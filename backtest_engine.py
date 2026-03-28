"""
backtest_engine.py
==================
Event-driven backtest simulator with realistic execution modelling.

Execution realism
-----------------
  - Slippage:   fixed % applied to entry and exit prices
  - Commission: flat USD per trade (both sides)
  - Spread:     half-spread applied on each side
  - Delay:      signal at bar t -> fill at bar t + EXECUTION_DELAY_BARS
  - No look-ahead: the signal for bar t uses only data up to bar t close

Position discipline
-------------------
  - Only ONE open position at a time (no overlapping trades)
  - Maximum MAX_TRADES_PER_DAY per calendar day
  - Session filter: only enter between SESSION_START and SESSION_END
  - Cooldown of COOLDOWN_BARS bars between trade exit and next entry

Signal conventions
------------------
  +1  ->  long trade  (buy at open of delayed bar + slippage)
  -1  ->  short trade (sell short at open of delayed bar - slippage)
   0  ->  no trade

Trade lifecycle
---------------
  Entry bar   : delayed open price +/- slippage +/- half-spread
  Exit bar    : stop or take-profit hit (check high/low on each subsequent bar)
                or time-stop (close at delayed bar close)
  P&L         : (exit - entry) * shares * direction - 2 * commission
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    COMMISSION_PER_TRADE,
    EXECUTION_DELAY_BARS,
    INITIAL_EQUITY,
    KELLY_WARMUP,
    MIN_PROBA_THRESHOLD,
    REGIME_VOL_THRESHOLD,
    REGIME_VOL_WINDOW,
    SLIPPAGE_PCT,
    SPREAD_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TIME_STOP_BARS,
    TREND_MA_BARS,
    USE_KELLY,
    USE_SESSION_FILTER,
    USE_TREND_FILTER,
)
from risk_management import PositionSizer, RiskGuard
from market_impact import MarketImpactModel

logger = logging.getLogger(__name__)

# Session filter (bars/timestamps, ET)
SESSION_START_HOUR   = 9
SESSION_START_MINUTE = 45    # avoid first 15 min of chaotic open
SESSION_END_HOUR     = 15
SESSION_END_MINUTE   = 15    # avoid last 45 min of close

MAX_TRADES_PER_DAY   = 2     # para barras diarias: max 2 novas entradas por dia
COOLDOWN_BARS        = 1     # 1 dia de espera apos saida antes de re-entrar


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    direction:   int              # +1 long, -1 short
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float
    exit_reason: str              # "tp" | "sl" | "time"


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Simulates trade execution on out-of-sample signals.

    Parameters
    ----------
    df       : OHLCV DataFrame with DatetimeIndex (used for price simulation)
    signals  : signals DataFrame with columns [pred, proba_1, proba_-1, ...]
               produced by WalkForwardValidator.signals_df or ModelTrainer
    equity   : starting equity in USD
    params   : optional dict of per-ticker overrides (from ticker_config).
               Supported keys:
                 min_proba_threshold, stop_loss_pct, take_profit_pct,
                 time_stop_bars, use_trend_filter, trend_ma_bars,
                 direction  ("both" | "long_only" | "short_only"),
                 regime_filter ("all" | "Bull" | "Bear" | "Sideways" |
                                "Bull+Sideways" | "Bear+Sideways" | "Bear+Bull")
    regimes  : optional pd.Series with regime labels aligned to df.index
               (output of RegimeDetector.fit_predict) — required when
               params["regime_filter"] != "all"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        equity: float = INITIAL_EQUITY,
        impact_model: Optional[MarketImpactModel] = None,
        params: Optional[dict] = None,
        regimes: Optional[pd.Series] = None,
    ):
        self.df      = df.sort_index()
        self.signals = signals.sort_index()
        self.sizer   = PositionSizer()
        self.guard   = RiskGuard(equity=equity)
        self.impact  = impact_model or MarketImpactModel(method="sqrt")
        self.trades:  list[Trade]  = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)

        # ── Per-ticker parameter overrides ───────────────────────────────────
        p = params or {}
        self._min_proba    = float(p.get("min_proba_threshold", MIN_PROBA_THRESHOLD))
        self._sl_pct       = float(p.get("stop_loss_pct",       STOP_LOSS_PCT))
        self._tp_pct       = float(p.get("take_profit_pct",     TAKE_PROFIT_PCT))
        self._time_stop    = int(p.get("time_stop_bars",        TIME_STOP_BARS))
        self._use_trend    = bool(p.get("use_trend_filter",     USE_TREND_FILTER))
        self._trend_bars   = int(p.get("trend_ma_bars",         TREND_MA_BARS))
        self._direction    = str(p.get("direction",             "both"))
        self._regime_filt  = str(p.get("regime_filter",        "all"))

        # Regime labels (aligned to df.index) — used when regime_filter != "all"
        self._regimes: Optional[pd.Series] = None
        if regimes is not None:
            self._regimes = regimes.reindex(self.df.index, method="ffill")

        # Pre-compute rolling volatility for O(1) regime lookup during the loop
        log_ret = np.log(self.df["close"] / self.df["close"].shift(1))
        self._regime_vol = (
            log_ret
            .rolling(REGIME_VOL_WINDOW, min_periods=REGIME_VOL_WINDOW // 2)
            .std()
            .fillna(0.0)
        )

        # Pre-compute trend MA for the trend filter
        # Only enter LONG above MA, only enter SHORT below MA.
        self._trend_ma = (
            self.df["close"]
            .rolling(self._trend_bars, min_periods=self._trend_bars // 2)
            .mean()
        )

        # Pre-compute ADV and daily vol for market impact model (causal — no look-ahead)
        self._adv = MarketImpactModel.compute_adv(
            self.df["volume"], window=20
        ).fillna(1_000_000.0)
        self._daily_vol = MarketImpactModel.compute_daily_vol(
            self.df["close"], window=20
        ).fillna(0.012)

        # Online Kelly stats — updated after each closed trade
        self._win_count  = 0
        self._loss_count = 0
        self._sum_wins   = 0.0
        self._sum_losses = 0.0

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Execute the backtest loop.

        Returns
        -------
        pd.DataFrame of Trade records.
        """
        bars     = self.df
        signals  = self.signals
        n        = len(bars)
        idx      = bars.index

        equity_ts       = {idx[0]: self.guard.equity}
        current_date    = None
        trades_today    = 0
        in_position     = False      # True while a trade is simulated open
        position_exit_i = -1         # bar index where current trade exits
        cooldown_until  = -1         # bar index until which entries are blocked

        for i, ts in enumerate(idx):
            # Day boundary
            bar_date = ts.date()
            if bar_date != current_date:
                self.guard.new_day()
                current_date = bar_date
                trades_today = 0

            # Release position flag once the simulated trade has closed
            if in_position and i > position_exit_i:
                in_position = False

            # Update equity snapshot
            equity_ts[ts] = self.guard.equity

            # --- Guard rail checks ---
            if not self.guard.can_trade():
                continue
            if in_position:
                continue
            if i < cooldown_until:
                continue
            if trades_today >= MAX_TRADES_PER_DAY:
                continue

            # Session filter: only enter during liquid hours (intraday only)
            if USE_SESSION_FILTER:
                t_hour, t_min = ts.hour, ts.minute
                if (t_hour, t_min) < (SESSION_START_HOUR, SESSION_START_MINUTE):
                    continue
                if (t_hour, t_min) >= (SESSION_END_HOUR, SESSION_END_MINUTE):
                    continue

            # Regime filter: skip entry when volatility is dangerously high.
            if self._regime_vol.iloc[i] > REGIME_VOL_THRESHOLD:
                continue

            # Look up signal for this bar
            if ts not in signals.index:
                continue

            sig_row = signals.loc[ts]
            signal  = int(sig_row["pred"]) if "pred" in sig_row.index else 0

            if signal == 0:
                continue

            # Direction filter
            if self._direction == "long_only"  and signal == -1:
                continue
            if self._direction == "short_only" and signal == 1:
                continue

            # Regime filter — only trade in allowed regimes
            if self._regime_filt != "all" and self._regimes is not None:
                regime_label = self._regimes.get(ts)
                if regime_label is not None:
                    allowed = self._regime_filt.split("+")
                    if regime_label not in allowed:
                        continue

            # Trend filter: only long above MA, only short below MA.
            # Eliminates the primary source of losses: fighting the dominant trend.
            if self._use_trend:
                ma_val = self._trend_ma.iloc[i]
                price  = bars["close"].iloc[i]
                if signal == 1  and price < ma_val:   # long in downtrend
                    continue
                if signal == -1 and price > ma_val:   # short in uptrend
                    continue

            # Probability filter
            proba_col = f"proba_{signal}"
            if proba_col in sig_row.index:
                proba = float(sig_row[proba_col])
                if proba < self._min_proba:
                    continue

            # Entry bar (delayed)
            entry_bar_idx = i + EXECUTION_DELAY_BARS
            if entry_bar_idx >= n:
                continue

            entry_ts    = idx[entry_bar_idx]
            entry_bar   = bars.iloc[entry_bar_idx]
            # shares not yet sized here; use sizer estimate for impact calc
            _pre_shares = self.sizer.shares(self.guard.equity, float(entry_bar["open"]), self._sl_pct)
            entry_price = self._fill_price(
                float(entry_bar["open"]), signal, shares=_pre_shares, bar_idx=entry_bar_idx
            )

            # Position sizing: Kelly after warmup, fixed-risk during warmup
            n_trades_so_far = self._win_count + self._loss_count
            if USE_KELLY and n_trades_so_far >= KELLY_WARMUP:
                win_rate = self._win_count / n_trades_so_far
                avg_win  = self._sum_wins  / self._win_count  if self._win_count  > 0 else 0.0
                avg_loss = self._sum_losses / self._loss_count if self._loss_count > 0 else 1.0
                shares = self.sizer.kelly_shares(
                    self.guard.equity, entry_price, self._sl_pct,
                    win_rate, avg_win, avg_loss,
                )
            else:
                shares = self.sizer.shares(self.guard.equity, entry_price, self._sl_pct)

            if shares == 0:
                continue

            # Simulate full trade lifetime
            trade = self._simulate_trade(
                bars, entry_bar_idx, entry_ts, entry_price, signal, shares
            )
            if trade is None:
                continue

            # Compute exit bar index — O(log n) binary search vs O(n) get_loc
            exit_bar_i = idx.searchsorted(trade.exit_time, side="left")
            exit_bar_i = min(exit_bar_i, n - 1)

            in_position     = True
            position_exit_i = exit_bar_i
            cooldown_until  = exit_bar_i + COOLDOWN_BARS
            trades_today   += 1

            self.guard.update(trade.pnl)
            self.trades.append(trade)
            equity_ts[trade.exit_time] = self.guard.equity

            # Update online Kelly stats for next sizing decision
            if trade.pnl > 0:
                self._win_count  += 1
                self._sum_wins   += trade.pnl
            else:
                self._loss_count += 1
                self._sum_losses += abs(trade.pnl)

        # Build equity curve (forward-fill between trade events)
        self.equity_curve = (
            pd.Series(equity_ts)
            .sort_index()
            .reindex(idx, method="ffill")
        )

        logger.info(
            "Backtest complete: %d trades | final equity=%.0f",
            len(self.trades), self.guard.equity,
        )
        return self.trades_df()

    # ------------------------------------------------------------------
    def _fill_price(
        self,
        base_price: float,
        direction: int,
        shares: int = 0,
        bar_idx: int = 0,
    ) -> float:
        """
        Apply slippage, half-spread, and market impact to the raw price.

        When shares > 0 and a MarketImpactModel is attached, the
        square-root impact is included on top of fixed slippage/spread.
        """
        adv       = float(self._adv.iloc[bar_idx]) if bar_idx < len(self._adv) else 1_000_000.0
        daily_vol = float(self._daily_vol.iloc[bar_idx]) if bar_idx < len(self._daily_vol) else 0.012

        return self.impact.adjusted_fill_price(
            base_price    = base_price,
            direction     = direction,
            shares        = shares,
            daily_vol_pct = daily_vol,
            adv_shares    = adv,
            slippage_pct  = SLIPPAGE_PCT,
            spread_pct    = SPREAD_PCT,
        )

    # ------------------------------------------------------------------
    def _simulate_trade(
        self,
        bars: pd.DataFrame,
        entry_idx: int,
        entry_ts: pd.Timestamp,
        entry_price: float,
        direction: int,
        shares: int,
    ) -> Optional[Trade]:
        """
        Walk forward bar by bar to find the first barrier touched.
        Uses bar high/low to check TP and SL intrabar.
        """
        tp_price = entry_price * (1 + direction * self._tp_pct)
        sl_price = entry_price * (1 - direction * self._sl_pct)
        n        = len(bars)
        idx      = bars.index

        exit_price  = None
        exit_reason = "time"
        exit_ts     = None

        for j in range(entry_idx + 1, min(entry_idx + self._time_stop + 1, n)):
            bar  = bars.iloc[j]
            high = bar["high"]
            low  = bar["low"]

            if direction == 1:
                if high >= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif low <= sl_price:
                    exit_price, exit_reason = sl_price, "sl"
            else:
                if low <= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif high >= sl_price:
                    exit_price, exit_reason = sl_price, "sl"

            if exit_price is not None:
                exit_ts = idx[j]
                break

        if exit_price is None:
            last_idx    = min(entry_idx + self._time_stop, n - 1)
            exit_price  = bars.iloc[last_idx]["close"]
            exit_ts     = idx[last_idx]
            exit_reason = "time"

        exit_bar_i = min(entry_idx + self._time_stop, len(bars) - 1)
        exit_fill = self._fill_price(exit_price, -direction, shares=shares, bar_idx=exit_bar_i)
        gross_pnl = (exit_fill - entry_price) * shares * direction
        net_pnl   = gross_pnl - 2 * COMMISSION_PER_TRADE

        return Trade(
            entry_time=entry_ts,
            exit_time=exit_ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_fill,
            shares=shares,
            pnl=net_pnl,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    def trades_df(self) -> pd.DataFrame:
        """Return all trades as a structured DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        rows = [
            {
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "direction":   "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 4),
                "exit_price":  round(t.exit_price,  4),
                "shares":      t.shares,
                "pnl":         round(t.pnl, 2),
                "exit_reason": t.exit_reason,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)
