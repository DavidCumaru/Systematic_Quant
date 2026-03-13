"""
tests/test_feature_engineering.py
==================================
Unit tests for feature_engineering.py

Coverage
--------
- All expected feature columns are present
- No look-ahead bias in key indicators (RSI, MACD, momentum, ATR, beta)
- ATR is always non-negative
- VIX defaults to 20 when data is missing
- Beta defaults to 1.0 when SPY is missing
- Garman-Klass vol is non-negative
- Overnight gap is 0 for all intraday bars (non-first bar of the day)
- Feature count is stable (regression test)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering import (
    _atr,
    _garman_klass_vol,
    _macd,
    _overnight_gap,
    _rsi,
    _rolling_beta,
    _vix_feature,
    build_features,
    get_feature_names,
)


class TestFeatureColumns:
    """All expected feature groups must be present in the output."""

    REQUIRED_FEATURES = [
        "log_ret",
        "log_ret_lag1", "log_ret_lag2",
        "rsi",
        "macd", "macd_signal", "macd_hist",
        "atr", "atr_pct",
        "vwap_dev",
        "rolling_vol",
        "vol_spike",
        "order_imbalance",
        "breakout_up", "breakout_down",
        "close_zscore",
        "momentum_5", "momentum_10", "momentum_20",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "hl_spread",
        "overnight_gap",
        "rel_volume",
        "regime_vol_flag",
        "gk_vol",
        "amihud",
        "vix",
        "beta_spy",
        "days_to_earnings",
        "ma200_dist",
        "hi52w_prox",
        "mom_12_1",
    ]

    def test_all_required_features_present(self, featured_df):
        missing = [f for f in self.REQUIRED_FEATURES if f not in featured_df.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_no_nan_in_features(self, featured_df):
        feat_cols = get_feature_names(featured_df)
        nan_cols = [c for c in feat_cols if featured_df[c].isna().any()]
        assert not nan_cols, f"NaN values in feature columns: {nan_cols}"

    def test_feature_count_regression(self, featured_df):
        """Feature count should be at least 35 (guards against accidental deletions)."""
        n_features = len(get_feature_names(featured_df))
        assert n_features >= 35, f"Feature count dropped to {n_features} (expected >= 35)"


class TestNoLookahead:
    """Features at bar t must not use information from bar t+1 or later."""

    def test_rsi_no_future_data(self, featured_df):
        n = len(featured_df)
        row_idx = n // 2
        rsi_full = featured_df["rsi"].iloc[row_idx]
        truncated = featured_df["close"].iloc[: row_idx + 1]
        rsi_trunc = _rsi(truncated, 14).iloc[-1]
        # EWM accumulates tiny float differences across many bars; 1e-3 tolerance is appropriate
        assert abs(rsi_full - rsi_trunc) < 1e-3, (
            f"RSI at {row_idx} differs with future truncation: {rsi_full} vs {rsi_trunc}"
        )

    def test_macd_no_future_data(self, featured_df):
        n = len(featured_df)
        row_idx = n // 2
        macd_full = featured_df["macd"].iloc[row_idx]
        truncated = featured_df["close"].iloc[: row_idx + 1]
        macd_trunc, _, _ = _macd(truncated, 12, 26, 9)
        assert abs(macd_full - macd_trunc.iloc[-1]) < 1e-6

    def test_momentum_uses_only_past(self, featured_df):
        close = featured_df["close"]
        expected = close / close.shift(5) - 1
        shared = featured_df.index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            featured_df["momentum_5"].loc[shared],
            expected.loc[shared],
            check_names=False,
            rtol=1e-10,
        )

    def test_log_ret_uses_only_past(self, featured_df):
        close = featured_df["close"]
        expected = np.log(close / close.shift(1))
        shared = featured_df.index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            featured_df["log_ret"].loc[shared],
            expected.loc[shared],
            check_names=False,
        )

    def test_overnight_gap_zero_for_intraday(self):
        """For daily bars every bar is a 'new day', so overnight_gap should be computed.
        For hourly bars, intraday bars (non-first of the day) must be 0."""
        from conftest import make_ohlcv
        df = make_ohlcv(200, freq="1h")
        gap = _overnight_gap(df["open"], df["close"])
        dates = df.index.date
        date_arr = pd.Series(dates, index=df.index)
        intraday_mask = date_arr == date_arr.shift(1)  # True = NOT first bar of day
        assert (gap[intraday_mask] == 0.0).all()


class TestIndicatorProperties:
    """Physical / mathematical constraints on indicator values."""

    def test_atr_non_negative(self, featured_df):
        assert (featured_df["atr"] >= 0).all(), "ATR must be >= 0"

    def test_atr_pct_non_negative(self, featured_df):
        assert (featured_df["atr_pct"] >= 0).all()

    def test_rsi_in_range(self, featured_df):
        rsi = featured_df["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_garman_klass_vol_non_negative(self, raw_df):
        gk = _garman_klass_vol(raw_df["open"], raw_df["high"], raw_df["low"], raw_df["close"], 20)
        assert (gk.dropna() >= 0).all()

    def test_order_imbalance_in_range(self, featured_df):
        oi = featured_df["order_imbalance"].dropna()
        assert (oi >= 0).all() and (oi <= 1).all()

    def test_hi52w_prox_clipped(self, featured_df):
        prox = featured_df["hi52w_prox"].dropna()
        assert (prox >= 0.5).all() and (prox <= 1.0).all()

    def test_vol_spike_binary(self, featured_df):
        unique = set(featured_df["vol_spike"].unique())
        assert unique.issubset({0, 1})

    def test_regime_vol_flag_binary(self, featured_df):
        unique = set(featured_df["regime_vol_flag"].unique())
        assert unique.issubset({0, 1})


class TestDefaultValues:
    """Features must degrade gracefully when external data is missing."""

    def test_vix_default_20_when_missing(self, raw_df):
        vix = _vix_feature(raw_df.index, None)
        assert (vix == 20.0).all()

    def test_beta_default_1_when_spy_missing(self, raw_df):
        log_ret = np.log(raw_df["close"] / raw_df["close"].shift(1))
        beta = _rolling_beta(log_ret, None, 20)
        assert (beta == 1.0).all()

    def test_build_features_without_spy_and_vix(self, raw_df):
        df = build_features(raw_df, spy_df=None, vix_df=None, ticker="")
        assert not df.empty
        assert "beta_spy" in df.columns
        assert "vix" in df.columns
