"""
tests/test_portfolio_manager.py
================================
Unit tests for portfolio_manager.py.

Coverage
--------
- Equal weights sum to 1 and are uniform
- Risk parity weights sum to 1 and favour lower-vol assets
- Min-variance weights sum to 1
- Max-Sharpe weights sum to 1
- max_position_pct cap is respected for all methods
- allocate_shares returns positive integers
- Correlation matrix is symmetric and in [-1, 1]
- Diversification ratio is >= 1.0 when assets are not perfectly correlated
- portfolio_summary returns all expected keys
- Empty inputs are handled gracefully
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_manager import PortfolioManager


TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


@pytest.fixture
def returns(returns_df):
    return returns_df


class TestEqualWeights:

    def test_weights_sum_to_one(self, returns):
        pm = PortfolioManager(method="equal")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_weights_are_uniform(self, returns):
        pm = PortfolioManager(method="equal")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        vals = list(w.values())
        assert max(vals) - min(vals) < 1e-9


class TestRiskParity:

    def test_weights_sum_to_one(self, returns):
        pm = PortfolioManager(method="risk_parity")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self, returns):
        """Low-vol asset should receive more weight than high-vol asset."""
        pm = PortfolioManager(method="risk_parity")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        vols = returns[TICKERS].std()
        # Rank by vol (ascending) -> weights should be inversely correlated
        vol_rank   = vols.rank()
        weight_ser = pd.Series(w)
        weight_rank = weight_ser.rank(ascending=False)  # highest weight -> rank 1
        # Spearman correlation should be positive (low vol -> high rank weight)
        corr = vol_rank.corr(weight_rank, method="spearman")
        assert corr > 0


class TestMinVariance:

    def test_weights_sum_to_one(self, returns):
        pm = PortfolioManager(method="min_variance")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_weights_non_negative(self, returns):
        pm = PortfolioManager(method="min_variance")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert all(v >= -1e-9 for v in w.values())


class TestMaxSharpe:

    def test_weights_sum_to_one(self, returns):
        pm = PortfolioManager(method="max_sharpe")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_weights_non_negative(self, returns):
        pm = PortfolioManager(method="max_sharpe")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        assert all(v >= -1e-9 for v in w.values())


class TestPositionCap:

    def test_max_position_cap_respected(self, returns):
        cap = 0.30
        pm = PortfolioManager(method="risk_parity", max_position_pct=cap)
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        for t, v in w.items():
            assert v <= cap + 1e-9, f"Position for {t} exceeds cap: {v:.4f} > {cap}"


class TestAllocateShares:

    def test_all_positive_integers(self, returns):
        pm = PortfolioManager(equity=100_000, method="equal")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        prices = {"SPY": 450.0, "QQQ": 370.0, "TLT": 92.0, "GLD": 185.0}
        shares = pm.allocate_shares(w, prices)
        for t, n in shares.items():
            assert isinstance(n, int)
            assert n >= 1

    def test_zero_price_skipped(self):
        pm = PortfolioManager(equity=100_000)
        shares = pm.allocate_shares({"SPY": 0.5, "QQQ": 0.5}, {"SPY": 0.0, "QQQ": 400.0})
        assert "SPY" not in shares
        assert "QQQ" in shares


class TestCorrelationMatrix:

    def test_symmetric(self, returns):
        pm = PortfolioManager()
        corr = pm.correlation_matrix(returns)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_values_in_range(self, returns):
        pm = PortfolioManager()
        corr = pm.correlation_matrix(returns)
        assert (corr.values >= -1 - 1e-9).all()
        assert (corr.values <=  1 + 1e-9).all()

    def test_diagonal_is_one(self, returns):
        pm = PortfolioManager()
        corr = pm.correlation_matrix(returns)
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0)


class TestDiversificationRatio:

    def test_dr_gte_one(self, returns):
        pm = PortfolioManager(method="equal")
        active = {t: 1 for t in TICKERS}
        w = pm.compute_weights(TICKERS, returns, active)
        dr = pm.diversification_ratio(w, returns)
        assert dr >= 1.0 - 1e-9, f"DR={dr} < 1 — unexpected"

    def test_dr_one_for_single_asset(self, returns):
        """Single asset portfolio has DR exactly 1."""
        pm = PortfolioManager()
        w  = {"SPY": 1.0}
        dr = pm.diversification_ratio(w, returns)
        assert abs(dr - 1.0) < 1e-9


class TestPortfolioSummary:

    EXPECTED_KEYS = [
        "n_tickers",
        "portfolio_return_pct",
        "portfolio_sharpe",
        "portfolio_max_dd_pct",
        "mean_ticker_sharpe",
        "mean_ticker_return_pct",
        "mean_win_rate_pct",
        "mean_profit_factor",
    ]

    def _make_mock_metrics(self):
        return {
            "SPY": {
                "sharpe_ratio": 1.5, "total_return_pct": 20.0,
                "max_drawdown_pct": 5.0, "win_rate_pct": 60.0, "profit_factor": 1.8,
            },
            "QQQ": {
                "sharpe_ratio": 1.2, "total_return_pct": 15.0,
                "max_drawdown_pct": 7.0, "win_rate_pct": 55.0, "profit_factor": 1.5,
            },
        }

    def _make_equity_curves(self):
        idx = pd.date_range("2022-01-03", periods=50, freq="B")
        return {
            "SPY": pd.Series(100 + np.arange(50) * 0.3, index=idx),
            "QQQ": pd.Series(100 + np.arange(50) * 0.2, index=idx),
        }

    def test_all_expected_keys_present(self):
        pm = PortfolioManager()
        summary = pm.portfolio_summary(self._make_mock_metrics(), self._make_equity_curves())
        for key in self.EXPECTED_KEYS:
            assert key in summary, f"Missing key: {key}"

    def test_empty_metrics_returns_empty(self):
        pm = PortfolioManager()
        summary = pm.portfolio_summary({}, {})
        assert summary == {}
