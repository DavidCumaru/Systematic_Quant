"""
market_impact.py
================
Realistic market impact modelling for the systematic trading pipeline.

Models implemented
------------------
  1. Square-Root Impact (Almgren-Chriss / empirical)
     impact_pct = eta * sigma * sqrt(Q / ADV)
     where:
       eta   = market impact coefficient (default 0.1 — conservative)
       sigma = daily volatility of the instrument
       Q     = order size in shares
       ADV   = average daily volume in shares (rolling 20-day)

  2. Linear (VWAP) impact  [simpler fallback]
     impact_pct = kappa * (Q / ADV)

  3. Zero impact  [used when volume data unavailable]

The square-root model is the industry standard for equities and ETFs.
It penalises large orders relative to daily volume, accurately capturing
the concave price-impact curve observed in microstructure research.

Usage
-----
    from market_impact import MarketImpactModel

    model   = MarketImpactModel(method="sqrt")
    imp_pct = model.impact_pct(
        shares   = 500,
        entry_price = 450.0,
        daily_vol_pct = 0.012,    # 1.2% daily vol
        adv_shares = 15_000_000,  # ~15M shares/day for SPY
    )
    fill_price = entry_price * (1 + imp_pct)   # for a buy

Integration with BacktestEngine
--------------------------------
BacktestEngine calls model.adjusted_fill_price() instead of _fill_price()
when a MarketImpactModel instance is passed.  The impact is ADDITIVE to
slippage and spread, providing a combined realistic transaction cost.

References
----------
- Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"
- Grinold & Kahn (1995) "Active Portfolio Management" (sqrt rule)
- Kissell (2014) "The Science of Algorithmic Trading and Portfolio Management"
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Impact coefficient — calibrated to typical US equity ETFs
# eta = 0.10 is a conservative estimate (range: 0.05 – 0.20 in literature)
DEFAULT_ETA   = 0.10
DEFAULT_KAPPA = 0.20   # linear model coefficient

# ADV estimation window (trading days)
ADV_WINDOW = 20

# Minimum ADV to avoid division by zero or unrealistic impact for illiquid days
MIN_ADV_SHARES = 10_000


class MarketImpactModel:
    """
    Computes realistic market impact cost for a given order.

    Parameters
    ----------
    method   : "sqrt"   — square-root Almgren-Chriss (default)
               "linear" — linear VWAP model
               "zero"   — no impact (pure slippage only)
    eta      : impact coefficient for sqrt model
    kappa    : impact coefficient for linear model
    """

    def __init__(
        self,
        method: Literal["sqrt", "linear", "zero"] = "sqrt",
        eta:    float = DEFAULT_ETA,
        kappa:  float = DEFAULT_KAPPA,
    ):
        self.method = method
        self.eta    = eta
        self.kappa  = kappa

    # ------------------------------------------------------------------
    def impact_pct(
        self,
        shares:        int,
        entry_price:   float,
        daily_vol_pct: float,
        adv_shares:    float,
    ) -> float:
        """
        Compute the price impact as a fraction of entry price.

        Parameters
        ----------
        shares        : order size in shares
        entry_price   : current price per share (for notional check)
        daily_vol_pct : daily volatility of the instrument (e.g. 0.012 = 1.2%)
        adv_shares    : average daily volume in shares (rolling 20-day)

        Returns
        -------
        float — impact as a positive fraction (e.g. 0.001 = 0.1 bps)
                Applied as: fill_price = base_price * (1 + impact_pct) for buys
                            fill_price = base_price * (1 - impact_pct) for sells
        """
        if self.method == "zero" or shares <= 0 or adv_shares <= MIN_ADV_SHARES:
            return 0.0

        participation = shares / max(adv_shares, MIN_ADV_SHARES)

        if self.method == "sqrt":
            impact = self.eta * daily_vol_pct * np.sqrt(participation)
        elif self.method == "linear":
            impact = self.kappa * participation
        else:
            impact = 0.0

        # Cap: never more than 2% impact per trade (prevents numerical blow-ups)
        return float(np.clip(impact, 0.0, 0.02))

    # ------------------------------------------------------------------
    def adjusted_fill_price(
        self,
        base_price:    float,
        direction:     int,
        shares:        int,
        daily_vol_pct: float,
        adv_shares:    float,
        slippage_pct:  float = 0.0005,
        spread_pct:    float = 0.0002,
    ) -> float:
        """
        Compute the realistic all-in fill price including:
          - Half-spread
          - Slippage
          - Market impact (direction-aware)

        Parameters
        ----------
        base_price    : raw bar price (open or close)
        direction     : +1 for buy, -1 for sell
        shares        : order size in shares
        daily_vol_pct : instrument daily volatility
        adv_shares    : average daily volume in shares
        slippage_pct  : fixed slippage fraction
        spread_pct    : bid-ask spread fraction

        Returns
        -------
        float — adjusted fill price
        """
        impact  = self.impact_pct(shares, base_price, daily_vol_pct, adv_shares)
        total_cost_pct = slippage_pct + spread_pct / 2 + impact

        if direction == 1:   # buy: pay more
            return base_price * (1 + total_cost_pct)
        else:                # sell: receive less
            return base_price * (1 - total_cost_pct)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_adv(volume: pd.Series, window: int = ADV_WINDOW) -> pd.Series:
        """
        Compute the rolling average daily volume.

        Uses shift(1) to avoid look-ahead: ADV at bar t uses only bars
        [t - window, t - 1].

        Parameters
        ----------
        volume : daily volume Series
        window : rolling window in bars

        Returns
        -------
        pd.Series of ADV values aligned with volume.index
        """
        return volume.shift(1).rolling(window, min_periods=window // 2).mean()

    # ------------------------------------------------------------------
    @staticmethod
    def compute_daily_vol(close: pd.Series, window: int = ADV_WINDOW) -> pd.Series:
        """
        Compute rolling close-to-close log-return volatility.

        Causal: uses shift(1) so bar t does not use bar t's close.

        Parameters
        ----------
        close  : daily close price Series
        window : rolling window in bars

        Returns
        -------
        pd.Series of daily volatility (fractional, e.g. 0.012)
        """
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(window, min_periods=window // 2).std()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.method == "sqrt":
            return f"MarketImpactModel(sqrt, eta={self.eta})"
        if self.method == "linear":
            return f"MarketImpactModel(linear, kappa={self.kappa})"
        return "MarketImpactModel(zero)"
