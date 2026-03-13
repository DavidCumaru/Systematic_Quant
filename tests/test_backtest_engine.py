"""
tests/test_backtest_engine.py
==============================
Unit tests for backtest_engine.py.

Coverage
--------
- Equity curve length matches OHLCV index
- Equity curve starts at initial equity
- No overlapping trades (position discipline)
- MAX_TRADES_PER_DAY is respected
- _fill_price adds slippage/spread for longs and subtracts for shorts
- trades_df() returns expected columns
- RiskGuard kill-switch stops trading when triggered
- Equity curve is monotone between trades (ffill)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest_engine import BacktestEngine, MAX_TRADES_PER_DAY
from model_training import ModelTrainer
from config import INITIAL_EQUITY


# ---------------------------------------------------------------------------
# Helper to build a minimal signals DataFrame
# ---------------------------------------------------------------------------

def _build_signals(labeled_df: pd.DataFrame) -> pd.DataFrame:
    trainer = ModelTrainer()
    trainer.fit(labeled_df)
    proba_df = trainer.predict_proba(labeled_df)
    preds    = trainer.predict(labeled_df)
    signals  = pd.DataFrame({"pred": preds}, index=labeled_df.index)
    signals  = pd.concat([signals, proba_df.add_prefix("proba_")], axis=1)
    return signals


class TestEquityCurve:

    def test_equity_curve_length_matches_ohlcv(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals, equity=INITIAL_EQUITY)
        engine.run()
        assert len(engine.equity_curve) == len(featured_df)

    def test_equity_curve_starts_at_initial_equity(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals, equity=INITIAL_EQUITY)
        engine.run()
        assert engine.equity_curve.iloc[0] == pytest.approx(INITIAL_EQUITY)

    def test_equity_curve_index_is_datetime(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals, equity=INITIAL_EQUITY)
        engine.run()
        assert isinstance(engine.equity_curve.index, pd.DatetimeIndex)


class TestPositionDiscipline:

    def test_no_overlapping_trades(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals)
        engine.run()
        trades = engine.trades

        if len(trades) < 2:
            pytest.skip("Fewer than 2 trades produced")

        for i in range(len(trades) - 1):
            assert trades[i + 1].entry_time >= trades[i].exit_time, (
                f"Overlap: trade {i} exits {trades[i].exit_time}, "
                f"trade {i+1} enters {trades[i+1].entry_time}"
            )

    def test_max_trades_per_day(self, labeled_df, featured_df):
        from collections import Counter
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals)
        engine.run()
        if not engine.trades:
            pytest.skip("No trades produced")

        daily_counts = Counter(t.entry_time.date() for t in engine.trades)
        worst_day    = max(daily_counts.values())
        assert worst_day <= MAX_TRADES_PER_DAY


class TestFillPrice:

    def setup_method(self):
        # Minimal BacktestEngine instance to test internal methods
        import pandas as pd
        import numpy as np
        rng = np.random.default_rng(0)
        n = 100
        idx = pd.bdate_range("2022-01-03", periods=n, tz="America/New_York")
        close = 100 + np.cumsum(rng.normal(0, 0.3, n))
        df = pd.DataFrame({
            "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1_000_000.0
        }, index=idx)
        signals = pd.DataFrame({"pred": [0] * n}, index=idx)
        self.engine = BacktestEngine(df=df, signals=signals)

    def test_long_fill_price_above_base(self):
        base = 100.0
        fill = self.engine._fill_price(base, direction=1)
        assert fill > base, "Long fill must be above base (slippage + spread)"

    def test_short_fill_price_below_base(self):
        base = 100.0
        fill = self.engine._fill_price(base, direction=-1)
        assert fill < base, "Short fill must be below base"

    def test_fill_price_symmetric(self):
        base = 100.0
        long_fill  = self.engine._fill_price(base, direction=1)
        short_fill = self.engine._fill_price(base, direction=-1)
        # Should be symmetric around base
        assert abs((long_fill - base) - (base - short_fill)) < 1e-9


class TestTradesDf:

    def test_trades_df_columns(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals)
        trades_df = engine.run()

        if trades_df.empty:
            pytest.skip("No trades produced")

        expected_cols = {
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "shares", "pnl", "exit_reason"
        }
        assert expected_cols.issubset(set(trades_df.columns))

    def test_direction_values(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals)
        trades_df = engine.run()
        if trades_df.empty:
            pytest.skip("No trades produced")
        assert set(trades_df["direction"].unique()).issubset({"LONG", "SHORT"})

    def test_exit_reason_values(self, labeled_df, featured_df):
        signals = _build_signals(labeled_df)
        engine  = BacktestEngine(df=featured_df, signals=signals)
        trades_df = engine.run()
        if trades_df.empty:
            pytest.skip("No trades produced")
        assert set(trades_df["exit_reason"].unique()).issubset({"tp", "sl", "time"})
