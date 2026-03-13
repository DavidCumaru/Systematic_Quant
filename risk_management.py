"""
risk_management.py
==================
Professional-grade risk controls for the systematic pipeline.

Components
----------
  PositionSizer   — dynamic Kelly-inspired / fixed-risk position sizing
  RiskGuard       — stateful per-session kill-switch logic
  RiskMetrics     — utility functions (drawdown series, VaR, etc.)

All sizing is in shares (integer) given an equity amount, entry price,
and a stop-loss distance.

Risk per trade is capped at RISK_PER_TRADE × current_equity.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DAILY_STOP_PCT,
    INITIAL_EQUITY,
    KELLY_FRACTION,
    KELLY_WARMUP,
    MAX_DRAWDOWN_PCT,
    RISK_PER_TRADE,
    USE_KELLY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Computes share quantity based on fixed fractional risk.

    Formula
    -------
    dollar_risk    = equity × RISK_PER_TRADE
    stop_distance  = entry_price × stop_pct
    shares         = floor( dollar_risk / stop_distance )

    The position is further clipped by a maximum notional cap
    (equity × max_notional_pct) to avoid over-sizing on tight stops.
    """

    def __init__(
        self,
        risk_per_trade: float = RISK_PER_TRADE,
        max_notional_pct: float = 0.20,   # max 20% of equity per trade
    ):
        self.risk_per_trade    = risk_per_trade
        self.max_notional_pct  = max_notional_pct

    def shares(
        self,
        equity: float,
        entry_price: float,
        stop_pct: float,
    ) -> int:
        """
        Calculate number of shares to buy/sell.

        Parameters
        ----------
        equity      : current account equity in USD
        entry_price : expected fill price per share
        stop_pct    : stop-loss distance as fraction (e.g. 0.003 = 0.3%)

        Returns
        -------
        int  — number of shares (≥ 0)
        """
        if entry_price <= 0 or stop_pct <= 0:
            return 0

        dollar_risk   = equity * self.risk_per_trade
        stop_distance = entry_price * stop_pct
        raw_shares    = dollar_risk / stop_distance

        # Notional cap
        max_notional  = equity * self.max_notional_pct
        max_shares    = max_notional / entry_price

        shares = int(min(raw_shares, max_shares))
        logger.debug(
            "Sizing: equity=%.0f  entry=%.2f  stop_pct=%.4f  -> %d shares",
            equity, entry_price, stop_pct, shares,
        )
        return shares

    def notional(self, equity: float, entry_price: float, stop_pct: float) -> float:
        """Dollar value of the computed position."""
        return self.shares(equity, entry_price, stop_pct) * entry_price

    def kelly_shares(
        self,
        equity: float,
        entry_price: float,
        stop_pct: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> int:
        """
        Kelly Criterion position sizing (fractional Kelly).

        Formula: f* = (b*p - q) / b
          b = avg_win / avg_loss  (win/loss ratio)
          p = win_rate
          q = 1 - p

        Uses KELLY_FRACTION (default 0.25) of the Kelly fraction
        as a conservative safety margin.

        Falls back to fixed-risk sizing when Kelly fraction is negative
        (negative expectancy — model has no edge in this regime).
        """
        if avg_loss <= 0 or entry_price <= 0 or stop_pct <= 0:
            return self.shares(equity, entry_price, stop_pct)

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_f = (b * p - q) / b if b > 0 else 0.0
        kelly_f = max(0.0, kelly_f) * KELLY_FRACTION  # fractional Kelly, never negative

        if kelly_f == 0.0:
            # No edge detected — return 0 (skip trade)
            return 0

        dollar_risk   = equity * kelly_f
        stop_distance = entry_price * stop_pct
        raw_shares    = dollar_risk / stop_distance

        # Notional cap: never risk more than 20% of equity on one trade
        max_shares = (equity * 0.20) / entry_price
        shares = int(min(raw_shares, max_shares))

        logger.debug(
            "Kelly sizing: win_rate=%.2f  b=%.2f  kelly_f=%.4f  -> %d shares",
            win_rate, b, kelly_f, shares,
        )
        return shares


# ---------------------------------------------------------------------------
# Risk Guard (stateful kill-switch)
# ---------------------------------------------------------------------------

@dataclass
class RiskGuard:
    """
    Stateful risk controller that tracks intraday and cumulative drawdown.

    Attributes
    ----------
    equity          : current account equity
    peak_equity     : highest equity achieved so far (for drawdown calc)
    daily_start_eq  : equity at start of current trading day
    is_halted       : True when trading is suspended by a kill-switch
    halt_reason     : human-readable explanation
    trade_count     : total number of trades taken today
    """

    equity:         float = INITIAL_EQUITY
    peak_equity:    float = field(init=False)
    daily_start_eq: float = field(init=False)
    is_halted:      bool  = False
    halt_reason:    str   = ""
    trade_count:    int   = 0

    def __post_init__(self):
        self.peak_equity    = self.equity
        self.daily_start_eq = self.equity

    # ------------------------------------------------------------------
    def update(self, pnl: float) -> None:
        """
        Apply a realised P&L to the equity account and check all
        risk limits.  Call after every closed trade.
        """
        self.equity    += pnl
        self.trade_count += 1

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Check daily stop
        daily_loss_pct = (self.equity - self.daily_start_eq) / self.daily_start_eq
        if daily_loss_pct <= -DAILY_STOP_PCT:
            self._halt(
                f"Daily stop triggered: {daily_loss_pct:.2%} loss "
                f"(limit={-DAILY_STOP_PCT:.2%})"
            )
            return

        # Check max drawdown from peak
        dd_pct = (self.equity - self.peak_equity) / self.peak_equity
        if dd_pct <= -MAX_DRAWDOWN_PCT:
            self._halt(
                f"Max drawdown triggered: {dd_pct:.2%} "
                f"(limit={-MAX_DRAWDOWN_PCT:.2%})"
            )

    # ------------------------------------------------------------------
    def new_day(self) -> None:
        """Reset daily counters at the start of each trading session."""
        self.daily_start_eq = self.equity
        self.trade_count    = 0
        # Only reset halt if it was a daily stop (not a max-drawdown halt)
        if "Daily" in self.halt_reason:
            self.is_halted   = False
            self.halt_reason = ""
            logger.info("RiskGuard: daily stop cleared — new session started")

    # ------------------------------------------------------------------
    def can_trade(self) -> bool:
        """Return True if the guard allows a new trade to be opened."""
        return not self.is_halted

    # ------------------------------------------------------------------
    def _halt(self, reason: str) -> None:
        if not self.is_halted:
            self.is_halted   = True
            self.halt_reason = reason
            logger.warning("KILL SWITCH ACTIVATED: %s", reason)

    # ------------------------------------------------------------------
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak as a positive fraction."""
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_pnl(self) -> float:
        """Intraday P&L in USD."""
        return self.equity - self.daily_start_eq

    def __repr__(self) -> str:
        return (
            f"RiskGuard(equity={self.equity:,.0f}  "
            f"dd={self.current_drawdown:.2%}  "
            f"halted={self.is_halted})"
        )


# ---------------------------------------------------------------------------
# Risk Metrics (stateless utilities)
# ---------------------------------------------------------------------------

class RiskMetrics:
    """Static utility methods for risk analysis on equity / return series."""

    @staticmethod
    def drawdown_series(equity: pd.Series) -> pd.Series:
        """
        Compute the rolling drawdown from the running peak.

        Returns a Series of values in [0, 1] (positive = loss).
        """
        peak = equity.cummax()
        return (peak - equity) / peak.replace(0, np.nan)

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Maximum drawdown (positive fraction)."""
        return RiskMetrics.drawdown_series(equity).max()

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Historical VaR at *confidence* level (e.g. 0.95 → 5th percentile loss).
        Returned as a positive number representing loss.
        """
        return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

    @staticmethod
    def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
        """CVaR / Expected Shortfall at *confidence* level."""
        var = RiskMetrics.value_at_risk(returns, confidence)
        tail = returns[returns <= -var]
        return float(-tail.mean()) if not tail.empty else var

    @staticmethod
    def calmar_ratio(equity: pd.Series, annual_return: float) -> float:
        """Calmar = annualised return / max drawdown."""
        mdd = RiskMetrics.max_drawdown(equity)
        return annual_return / mdd if mdd > 0 else np.nan
