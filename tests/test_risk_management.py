"""
tests/test_risk_management.py
==============================
Unit tests for risk_management.py.

Coverage
--------
PositionSizer
  - Zero shares when stop_pct or entry_price is 0
  - Notional cap limits position size
  - Fixed-risk sizing correct formula
  - Kelly returns 0 when expectancy is negative
  - Kelly uses KELLY_FRACTION dampening

RiskGuard
  - Daily stop triggers halt
  - Max drawdown triggers halt
  - new_day() clears daily-stop halt but not max-drawdown halt
  - can_trade() returns False after halt
  - Peak equity updates correctly

RiskMetrics
  - drawdown_series produces values in [0, 1]
  - max_drawdown is 0 for monotonically increasing equity
  - VaR is positive for a series with losses
  - Expected shortfall >= VaR
  - Calmar ratio = CAGR / max_dd
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_management import PositionSizer, RiskGuard, RiskMetrics


class TestPositionSizer:

    def setup_method(self):
        self.sizer = PositionSizer(risk_per_trade=0.01, max_notional_pct=0.20)

    def test_zero_shares_on_zero_stop(self):
        assert self.sizer.shares(100_000, 100.0, 0.0) == 0

    def test_zero_shares_on_zero_price(self):
        assert self.sizer.shares(100_000, 0.0, 0.01) == 0

    def test_fixed_risk_formula(self):
        equity       = 100_000
        entry_price  = 50.0
        stop_pct     = 0.02          # 2% stop
        dollar_risk  = equity * 0.01  # 1% of equity = $1000
        stop_dist    = entry_price * stop_pct  # $1
        raw_shares   = dollar_risk / stop_dist  # 1000
        # notional cap: 20% of 100_000 / 50 = 400 shares
        expected = min(int(raw_shares), int(equity * 0.20 / entry_price))
        assert self.sizer.shares(equity, entry_price, stop_pct) == expected

    def test_notional_cap_limits_size(self):
        # Very small stop_pct -> huge raw_shares, must be capped
        equity = 100_000
        shares = self.sizer.shares(equity, 10.0, 0.0001)
        notional = shares * 10.0
        assert notional <= equity * 0.20 + 10.0  # allow 1-share rounding

    def test_kelly_negative_expectancy_returns_zero(self):
        """If avg_win < avg_loss and win_rate is low, Kelly fraction is negative -> 0 shares."""
        shares = self.sizer.kelly_shares(
            equity=100_000,
            entry_price=100.0,
            stop_pct=0.02,
            win_rate=0.30,   # only 30% win rate
            avg_win=50.0,    # small wins
            avg_loss=200.0,  # large losses
        )
        assert shares == 0

    def test_kelly_positive_expectancy_returns_nonzero(self):
        shares = self.sizer.kelly_shares(
            equity=100_000,
            entry_price=50.0,
            stop_pct=0.01,
            win_rate=0.60,
            avg_win=200.0,
            avg_loss=100.0,
        )
        assert shares > 0

    def test_notional_method(self):
        equity = 100_000
        shares = self.sizer.shares(equity, 100.0, 0.01)
        assert self.sizer.notional(equity, 100.0, 0.01) == shares * 100.0


class TestRiskGuard:

    def test_initial_state(self):
        guard = RiskGuard(equity=100_000)
        assert not guard.is_halted
        assert guard.can_trade()
        assert guard.equity == 100_000

    def test_daily_stop_triggers_halt(self):
        guard = RiskGuard(equity=100_000)
        # Lose 4% — above the 3% daily stop limit
        guard.update(-4_000)
        assert guard.is_halted
        assert not guard.can_trade()
        assert "Daily" in guard.halt_reason

    def test_max_drawdown_triggers_halt(self):
        # Simulate multi-day losses accumulating past max drawdown (10%)
        # Each day: lose 2.9% (just under 3% daily stop) across 4 days -> ~11% total.
        guard = RiskGuard(equity=100_000)
        for _ in range(4):
            loss = guard.equity * 0.029  # 2.9%/day, under the 3% daily limit
            guard.update(-loss)
            if guard.is_halted:
                break
            guard.new_day()   # reset daily counter

        assert guard.is_halted
        assert guard.current_drawdown >= 0.10

    def test_new_day_clears_daily_stop(self):
        guard = RiskGuard(equity=100_000)
        guard.update(-4_000)
        assert guard.is_halted
        guard.new_day()
        assert not guard.is_halted
        assert guard.can_trade()

    def test_new_day_does_not_clear_max_drawdown_halt(self):
        # Accumulate a max-drawdown halt across multiple days
        guard = RiskGuard(equity=100_000)
        for _ in range(4):
            loss = guard.equity * 0.029
            guard.update(-loss)
            if guard.is_halted:
                break
            guard.new_day()
        assert guard.is_halted
        # Calling new_day() again should NOT clear the max-drawdown halt
        guard.new_day()
        assert guard.is_halted

    def test_peak_equity_updates(self):
        guard = RiskGuard(equity=100_000)
        guard.update(5_000)
        assert guard.peak_equity == 105_000

    def test_current_drawdown(self):
        guard = RiskGuard(equity=100_000)
        guard.update(10_000)  # peak = 110_000
        guard.update(-20_000) # equity = 90_000
        dd = guard.current_drawdown
        expected = (110_000 - 90_000) / 110_000
        assert abs(dd - expected) < 1e-9

    def test_daily_pnl(self):
        guard = RiskGuard(equity=100_000)
        guard.update(1_500)
        assert guard.daily_pnl == 1_500

    def test_no_double_halt(self):
        guard = RiskGuard(equity=100_000)
        guard.update(-4_000)
        first_reason = guard.halt_reason
        guard.update(-1_000)
        assert guard.halt_reason == first_reason  # not overwritten


class TestRiskMetrics:

    def _make_equity(self, values):
        idx = pd.date_range("2023-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=idx)

    def test_drawdown_series_zero_for_monotone_up(self):
        eq = self._make_equity([100, 101, 102, 103, 104])
        dd = RiskMetrics.drawdown_series(eq)
        assert (dd == 0.0).all()

    def test_drawdown_series_in_range(self):
        eq = self._make_equity([100, 110, 90, 105, 95])
        dd = RiskMetrics.drawdown_series(eq)
        assert (dd >= 0).all() and (dd <= 1).all()

    def test_max_drawdown_zero_monotone(self):
        eq = self._make_equity([100, 101, 102, 103])
        assert RiskMetrics.max_drawdown(eq) == 0.0

    def test_max_drawdown_correct(self):
        eq = self._make_equity([100, 120, 80, 110])
        mdd = RiskMetrics.max_drawdown(eq)
        expected = (120 - 80) / 120
        assert abs(mdd - expected) < 1e-9

    def test_var_positive_for_loss_series(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(-0.01, 0.02, 500))
        var = RiskMetrics.value_at_risk(returns, confidence=0.95)
        assert var > 0, "VaR should be positive (representing a loss)"

    def test_expected_shortfall_gte_var(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.02, 500))
        var = RiskMetrics.value_at_risk(returns, 0.95)
        es  = RiskMetrics.expected_shortfall(returns, 0.95)
        assert es >= var - 1e-9, "CVaR must be >= VaR"

    def test_calmar_ratio(self):
        eq = self._make_equity([100, 120, 80, 130])
        calmar = RiskMetrics.calmar_ratio(eq, annual_return=0.20)
        mdd    = RiskMetrics.max_drawdown(eq)
        assert abs(calmar - 0.20 / mdd) < 1e-9
