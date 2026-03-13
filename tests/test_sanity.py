"""
tests/test_sanity.py
====================
Sanity checks that guard the three most dangerous failure modes
in systematic trading systems:

  1. Look-ahead bias in features
  2. Look-ahead bias in labels (label uses future data)
  3. Overlapping trades in the backtest (position discipline)

Run with:
    python -m pytest tests/test_sanity.py -v
    -- or --
    python tests/test_sanity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering import build_features
from labeling import apply_triple_barrier
from backtest_engine import BacktestEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic 1h OHLCV data with a tz-aware DatetimeIndex."""
    rng = np.random.default_rng(seed)

    idx = pd.date_range(
        start="2024-01-02 09:30",
        periods=n,
        freq="1h",
        tz="America/New_York",
    )
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high  = close + rng.uniform(0.1, 1.0, n)
    low   = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def raw_df():
    return _make_synthetic_ohlcv(500)


@pytest.fixture
def featured_df(raw_df):
    return build_features(raw_df)


@pytest.fixture
def labeled_df(featured_df):
    return apply_triple_barrier(featured_df)


# ---------------------------------------------------------------------------
# Test 1: No look-ahead in features
# ---------------------------------------------------------------------------

class TestFeatureLookahead:
    """Features at bar t must not depend on prices from bar t+1 or later."""

    def test_log_ret_uses_only_past(self, featured_df):
        """log_ret at position i should equal log(close[i]/close[i-1]).
        Align by index because build_features drops warm-up NaN rows.
        """
        close    = featured_df["close"]
        expected = np.log(close / close.shift(1))
        shared   = featured_df.index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            featured_df["log_ret"].loc[shared],
            expected.loc[shared],
            check_names=False,
        )

    def test_rsi_no_future_data(self, featured_df):
        """RSI at bar t must not change if we remove bar t+1 onward.
        EWM accumulates tiny float differences; tolerance is 1e-5.
        """
        n = len(featured_df)
        row_idx = n // 2

        rsi_full = featured_df["rsi"].iloc[row_idx]

        from feature_engineering import _rsi
        truncated = featured_df["close"].iloc[: row_idx + 1]
        rsi_trunc = _rsi(truncated, 14).iloc[-1]

        assert abs(rsi_full - rsi_trunc) < 1e-5, (
            f"RSI at bar {row_idx} differs when future bars are removed: "
            f"{rsi_full} vs {rsi_trunc} (diff={abs(rsi_full - rsi_trunc):.2e})"
        )

    def test_momentum_no_future_data(self, featured_df):
        """momentum_3 at bar t = close[t]/close[t-3] - 1 (pure past).
        Align by index because build_features drops warm-up NaN rows.
        """
        close    = featured_df["close"]
        expected = close / close.shift(3) - 1
        shared   = featured_df.index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            featured_df["momentum_3"].loc[shared],
            expected.loc[shared],
            check_names=False,
        )

    def test_overnight_gap_only_on_day_boundary(self, featured_df):
        """overnight_gap must be 0 for all intraday bars (non-first bars)."""
        gap = featured_df["overnight_gap"]
        dates = featured_df.index.date
        date_arr = pd.Series(dates, index=featured_df.index)
        intraday_mask = date_arr == date_arr.shift(1)  # True = NOT first bar of day
        intraday_gaps = gap[intraday_mask]
        assert (intraday_gaps == 0.0).all(), (
            "overnight_gap is non-zero for intraday bars (look-ahead or logic bug)"
        )


# ---------------------------------------------------------------------------
# Test 2: No look-ahead in labels
# ---------------------------------------------------------------------------

class TestLabelLookahead:
    """Label for bar t must only use information from bars [t+1, t+TIME_STOP]."""

    def test_label_changes_when_future_removed(self, featured_df):
        """
        Take a labeled bar near the end of the dataset.
        If we shorten the dataset by TIME_STOP_BARS, that bar must become NaN
        (no valid forward window), proving the label depends on future data.
        """
        from config import TIME_STOP_BARS

        labeled_full = apply_triple_barrier(featured_df)
        n = len(labeled_full)
        assert n > 0, "No labeled rows produced"

        # The last labeled bar in the full dataset should not exist if we
        # truncate the OHLCV so there's no forward window for it
        last_ts = labeled_full.index[-1]

        # Truncate raw data to remove the forward window of last_ts
        cut_idx = featured_df.index.get_loc(last_ts) + 1  # one bar past last_ts
        truncated = featured_df.iloc[: cut_idx]           # no forward bars left

        labeled_short = apply_triple_barrier(truncated)

        assert last_ts not in labeled_short.index, (
            f"Bar {last_ts} still has a label even though its forward window "
            f"({TIME_STOP_BARS} bars) was removed — possible look-ahead in labeling."
        )

    def test_all_label_values_valid(self, labeled_df):
        """Labels must be exactly -1, 0, or +1."""
        valid = {-1, 0, 1}
        actual = set(labeled_df["label"].unique())
        assert actual.issubset(valid), f"Unexpected label values: {actual - valid}"

    def test_label_count_positive(self, labeled_df):
        """At least some labels of each class should exist (non-trivial dataset)."""
        counts = labeled_df["label"].value_counts()
        assert len(counts) >= 2, "Only one class found in labels — check TP/SL params"


# ---------------------------------------------------------------------------
# Test 3: No overlapping trades in the backtest
# ---------------------------------------------------------------------------

class TestBacktestPositionDiscipline:
    """Only one open position at a time; exit before next entry."""

    def _run_backtest(self, labeled_df, featured_df):
        """Build a minimal signals DataFrame and run the backtest."""
        from model_training import ModelTrainer

        trainer = ModelTrainer()
        trainer.fit(labeled_df)

        proba_df = trainer.predict_proba(labeled_df)
        signals  = pd.DataFrame({
            "pred":  trainer.predict(labeled_df),
        }, index=labeled_df.index)
        signals = pd.concat([signals, proba_df.add_prefix("proba_")], axis=1)

        engine = BacktestEngine(df=featured_df, signals=signals)
        engine.run()
        return engine

    def test_no_overlapping_trades(self, labeled_df, featured_df):
        """No two trades should overlap in time."""
        engine = self._run_backtest(labeled_df, featured_df)
        trades = engine.trades

        if len(trades) < 2:
            pytest.skip("Not enough trades to test for overlap")

        for i in range(len(trades) - 1):
            t_curr = trades[i]
            t_next = trades[i + 1]
            assert t_next.entry_time >= t_curr.exit_time, (
                f"Trade overlap detected!\n"
                f"  Trade {i}: entry={t_curr.entry_time}  exit={t_curr.exit_time}\n"
                f"  Trade {i+1}: entry={t_next.entry_time}  exit={t_next.exit_time}"
            )

    def test_max_trades_per_day_respected(self, labeled_df, featured_df):
        """No single day should have more than MAX_TRADES_PER_DAY trades."""
        from backtest_engine import MAX_TRADES_PER_DAY

        engine = self._run_backtest(labeled_df, featured_df)
        trades = engine.trades

        if not trades:
            pytest.skip("No trades produced")

        from collections import Counter
        daily_counts = Counter(t.entry_time.date() for t in trades)
        worst_day    = max(daily_counts.values())
        assert worst_day <= MAX_TRADES_PER_DAY, (
            f"MAX_TRADES_PER_DAY={MAX_TRADES_PER_DAY} violated: "
            f"found {worst_day} trades in a single day"
        )


# ---------------------------------------------------------------------------
# Run directly (without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    raw      = _make_synthetic_ohlcv(600)
    featured = build_features(raw)
    labeled  = apply_triple_barrier(featured)

    suites = [
        TestFeatureLookahead(),
        TestLabelLookahead(),
        TestBacktestPositionDiscipline(),
    ]

    passed = failed = 0
    for suite in suites:
        for name in dir(suite):
            if not name.startswith("test_"):
                continue
            try:
                method = getattr(suite, name)
                # inject fixtures manually
                import inspect
                params = inspect.signature(method).parameters
                kwargs = {}
                if "raw_df"      in params: kwargs["raw_df"]      = raw
                if "featured_df" in params: kwargs["featured_df"] = featured
                if "labeled_df"  in params: kwargs["labeled_df"]  = labeled
                method(**kwargs)
                print(f"  PASS  {suite.__class__.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {suite.__class__.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
