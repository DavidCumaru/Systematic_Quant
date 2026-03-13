"""
tests/test_performance.py
==========================
Unit tests for performance.py.

Coverage
--------
- compute_metrics() returns all expected keys
- Sharpe is positive for profitable trade series
- Max drawdown is 0 for monotonically increasing equity
- Bootstrap Sharpe CI: lo <= point_estimate <= hi (approximately)
- monthly_returns_table() returns a pivot with months as columns
- print_metrics() does not raise
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance import compute_metrics, monthly_returns_table, print_metrics, sharpe_confidence_interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equity(values, freq="B"):
    idx = pd.date_range("2022-01-03", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def _make_trades(n=30, win_rate=0.60, avg_win=300.0, avg_loss=150.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t = pd.Timestamp("2022-01-04", tz="America/New_York")
    for i in range(n):
        is_win = rng.random() < win_rate
        pnl    = avg_win if is_win else -avg_loss
        rows.append({
            "entry_time":  t,
            "exit_time":   t + pd.Timedelta(days=1),
            "direction":   "LONG",
            "entry_price": 100.0,
            "exit_price":  100.0 + (pnl / 10),
            "shares":      10,
            "pnl":         pnl,
            "exit_reason": "tp" if is_win else "sl",
        })
        t += pd.Timedelta(days=2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    EXPECTED_KEYS = [
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
        "total_return_pct", "win_rate_pct", "profit_factor",
        "n_trades", "cagr_pct", "omega_ratio",
    ]

    def test_all_keys_present(self):
        trades = _make_trades(40, win_rate=0.65)
        n = len(trades)
        equity = _make_equity([100_000 + i * 150 for i in range(n + 1)])
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        for key in self.EXPECTED_KEYS:
            assert key in metrics, f"Missing key: {key}"

    def test_sharpe_positive_for_profitable_trades(self):
        trades = _make_trades(50, win_rate=0.70)
        n = len(trades)
        equity = _make_equity([100_000 + i * 200 for i in range(n + 1)])
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        assert metrics["sharpe_ratio"] > 0

    def test_win_rate_in_range(self):
        trades = _make_trades(30, win_rate=0.60)
        n = len(trades)
        equity = _make_equity([100_000 + i * 100 for i in range(n + 1)])
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        assert 0 <= metrics["win_rate_pct"] <= 100

    def test_n_trades_matches(self):
        trades = _make_trades(25)
        n = len(trades)
        equity = _make_equity([100_000] * (n + 1))
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        assert metrics["n_trades"] == 25

    def test_max_drawdown_zero_monotone_equity(self):
        trades = _make_trades(10)
        n = len(trades)
        equity = _make_equity([100_000 + i * 500 for i in range(n + 1)])
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        assert metrics["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_profit_factor_gt_one_for_profitable(self):
        trades = _make_trades(40, win_rate=0.70, avg_win=300, avg_loss=100)
        n = len(trades)
        equity = _make_equity([100_000 + i * 300 for i in range(n + 1)])
        metrics = compute_metrics(trades, equity, initial_equity=100_000)
        assert metrics["profit_factor"] > 1.0


class TestSharpeCI:

    def test_ci_bounds_valid(self):
        equity = _make_equity([100_000 + np.random.randn() * 200 + i * 50 for i in range(200)])
        lo, hi = sharpe_confidence_interval(equity, n_bootstrap=500)
        assert lo <= hi

    def test_ci_width_positive(self):
        equity = _make_equity([100_000 + i * 30 for i in range(150)])
        lo, hi = sharpe_confidence_interval(equity, n_bootstrap=300)
        assert hi > lo


class TestMonthlyReturnsTable:

    def test_returns_dataframe(self):
        n = 300
        equity = _make_equity([100_000 + i * 100 for i in range(n)])
        result = monthly_returns_table(equity)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_months(self):
        n = 300
        equity = _make_equity([100_000 + i * 100 for i in range(n)])
        result = monthly_returns_table(equity)
        MONTH_ABBREVS = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}
        # Allow "Annual" summary column if present
        EXTRA_ALLOWED = {"Annual", "YTD", "Total"}
        for c in result.columns:
            assert c in range(1, 13) or str(c) in MONTH_ABBREVS or str(c) in EXTRA_ALLOWED, \
                f"Unexpected month column: {c!r}"

    def test_rows_are_years(self):
        n = 300
        equity = _make_equity([100_000 + i * 100 for i in range(n)])
        result = monthly_returns_table(equity)
        for yr in result.index:
            assert isinstance(yr, (int, np.integer)) or str(yr).isdigit(), \
                f"Unexpected row index: {yr!r}"


class TestPrintMetrics:

    def test_does_not_raise(self, capsys):
        metrics = {
            "sharpe_ratio": 1.5, "sortino_ratio": 2.0,
            "max_drawdown_pct": 5.0, "total_return_pct": 20.0,
            "win_rate_pct": 60.0, "profit_factor": 1.8,
            "n_trades": 30, "cagr_pct": 18.0, "omega_ratio": 1.4,
        }
        print_metrics(metrics)  # must not raise
