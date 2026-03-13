"""
position_manager.py
===================
Stateful management of concurrent open positions across multiple tickers.

This module provides the infrastructure for live multi-asset execution,
tracking open positions, checking for conflicts, and enforcing portfolio-level
risk limits across simultaneous positions.

Classes
-------
  Position       — dataclass describing a single open position
  PositionManager — stateful tracker for all concurrent open positions

Key features
------------
  - Maximum N concurrent positions (configurable)
  - Per-ticker position lock (no doubling-up on the same instrument)
  - Portfolio-level notional cap (e.g. total exposure <= 80% of equity)
  - Correlation-aware blocking (skip a new signal if a correlated position is open)
  - P&L tracking per position and portfolio-wide
  - Clean close/update interface for live reconciliation with broker

Design philosophy
-----------------
  The PositionManager is deliberately broker-agnostic: it tracks state
  in-memory and delegates actual order submission to ExecutionEngine.
  This makes it easy to swap out the broker (Alpaca -> IB -> Binance)
  without changing portfolio logic.

Usage
-----
    from position_manager import PositionManager, Position

    pm = PositionManager(max_positions=5, max_notional_pct=0.80)

    # Attempt to open a position
    ok = pm.can_open("SPY", direction=1, notional=20_000, equity=100_000)
    if ok:
        pos = pm.open_position(
            ticker="SPY", direction=1,
            entry_price=450.0, shares=44, notional=19_800,
            signal_id="abc12345",
        )

    # After the fill is confirmed
    pm.update_price("SPY", current_price=452.0)

    # On close signal or stop/TP hit
    pm.close_position("SPY", exit_price=453.5, exit_reason="tp")

    # Portfolio summary
    summary = pm.portfolio_pnl()
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_CONCURRENT_POSITIONS = 5      # maximum simultaneous open positions
MAX_NOTIONAL_PCT         = 0.80   # max total notional as % of equity
MAX_CORRELATION_BLOCK    = 0.80   # block new position if existing correlated pos open


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open position."""
    ticker:       str
    direction:    int          # +1 long, -1 short
    entry_price:  float
    shares:       int
    notional:     float        # abs(shares * entry_price)
    signal_id:    str
    opened_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    stop_loss:    Optional[float] = None
    take_profit:  Optional[float] = None

    def update_price(self, price: float) -> None:
        self.current_price  = price
        self.unrealised_pnl = (price - self.entry_price) * self.shares * self.direction

    @property
    def side(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * self.direction * 100

    def __repr__(self) -> str:
        return (
            f"Position({self.ticker} {self.side} "
            f"{self.shares}@{self.entry_price:.2f} "
            f"upnl={self.unrealised_pnl:+.2f})"
        )


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """
    Stateful multi-position tracker for live trading.

    Parameters
    ----------
    max_positions        : maximum number of concurrent open positions
    max_notional_pct     : max total notional as fraction of equity
    max_correlation_block: skip new position if correlation to any open
                           position exceeds this threshold
    """

    def __init__(
        self,
        max_positions:         int   = MAX_CONCURRENT_POSITIONS,
        max_notional_pct:      float = MAX_NOTIONAL_PCT,
        max_correlation_block: float = MAX_CORRELATION_BLOCK,
    ):
        self.max_positions         = max_positions
        self.max_notional_pct      = max_notional_pct
        self.max_correlation_block = max_correlation_block

        # {ticker: Position}
        self._positions: dict[str, Position] = {}

        # Closed position history (for session P&L)
        self._closed: list[dict] = []

        # Realised P&L for the session
        self._realised_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def open_tickers(self) -> list[str]:
        return list(self._positions.keys())

    @property
    def n_open(self) -> int:
        return len(self._positions)

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self._positions.values())

    @property
    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self._positions.values())

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._positions.get(ticker)

    def is_open(self, ticker: str) -> bool:
        return ticker in self._positions

    # ------------------------------------------------------------------
    # Eligibility check
    # ------------------------------------------------------------------

    def can_open(
        self,
        ticker:   str,
        direction: int,
        notional: float,
        equity:   float,
        returns:  Optional[pd.DataFrame] = None,
    ) -> tuple[bool, str]:
        """
        Determine whether a new position can be opened.

        Parameters
        ----------
        ticker    : instrument symbol
        direction : +1 long, -1 short
        notional  : proposed position notional in USD
        equity    : current account equity
        returns   : optional DataFrame of daily returns for correlation check
                    columns = tickers, index = dates

        Returns
        -------
        (allowed: bool, reason: str)
          allowed=True  if the position can be opened
          allowed=False with a human-readable reason if it must be blocked
        """
        # 1. Already have an open position in this ticker
        if self.is_open(ticker):
            return False, f"Already have an open position in {ticker}"

        # 2. Max concurrent positions
        if self.n_open >= self.max_positions:
            return False, (
                f"Max concurrent positions reached ({self.n_open}/{self.max_positions})"
            )

        # 3. Portfolio notional cap
        if equity > 0:
            projected_notional = self.total_notional + notional
            if projected_notional / equity > self.max_notional_pct:
                return False, (
                    f"Notional cap: projected {projected_notional/equity:.1%} "
                    f"> limit {self.max_notional_pct:.0%}"
                )

        # 4. Correlation block (requires returns DataFrame)
        if returns is not None and self._positions:
            corr_block = self._check_correlation(ticker, returns)
            if corr_block:
                return False, (
                    f"Correlation block: {ticker} is too correlated with open "
                    f"position {corr_block} (threshold={self.max_correlation_block:.0%})"
                )

        return True, "OK"

    def _check_correlation(self, new_ticker: str, returns: pd.DataFrame) -> Optional[str]:
        """
        Return the ticker of an existing open position that is too correlated
        with *new_ticker*, or None if no conflict.
        """
        if new_ticker not in returns.columns:
            return None

        for existing_ticker in self._positions:
            if existing_ticker not in returns.columns:
                continue
            corr = returns[new_ticker].corr(returns[existing_ticker])
            if abs(corr) >= self.max_correlation_block:
                logger.debug(
                    "Correlation block: %s x %s = %.3f >= %.3f",
                    new_ticker, existing_ticker, corr, self.max_correlation_block,
                )
                return existing_ticker
        return None

    # ------------------------------------------------------------------
    # Open / update / close
    # ------------------------------------------------------------------

    def open_position(
        self,
        ticker:      str,
        direction:   int,
        entry_price: float,
        shares:      int,
        notional:    float,
        signal_id:   Optional[str] = None,
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """
        Register a new open position.

        Call this after a broker fill confirmation (not before).

        Returns the Position object for the caller to store/log.
        """
        if self.is_open(ticker):
            logger.warning("open_position: %s already open — ignoring duplicate", ticker)
            return self._positions[ticker]

        pos = Position(
            ticker       = ticker,
            direction    = direction,
            entry_price  = entry_price,
            shares       = shares,
            notional     = notional,
            signal_id    = signal_id or str(uuid.uuid4())[:8],
            current_price = entry_price,
            stop_loss    = stop_loss,
            take_profit  = take_profit,
        )
        self._positions[ticker] = pos
        logger.info(
            "POSITION OPENED: %s %s %d@%.2f  notional=$%.0f  total_open=%d",
            ticker, pos.side, shares, entry_price, notional, self.n_open,
        )
        return pos

    def update_price(self, ticker: str, current_price: float) -> Optional[Position]:
        """Update the mark-to-market price of an open position."""
        pos = self._positions.get(ticker)
        if pos is None:
            return None
        pos.update_price(current_price)
        return pos

    def update_all_prices(self, prices: dict[str, float]) -> None:
        """Batch update mark-to-market prices for all open positions."""
        for ticker, price in prices.items():
            self.update_price(ticker, price)

    def close_position(
        self,
        ticker:      str,
        exit_price:  float,
        exit_reason: str = "signal",
    ) -> Optional[dict]:
        """
        Close an open position and record it in the closed history.

        Parameters
        ----------
        ticker      : instrument symbol
        exit_price  : fill price at close
        exit_reason : "tp" | "sl" | "signal" | "eod" | "manual"

        Returns the closed trade dict, or None if ticker was not open.
        """
        pos = self._positions.pop(ticker, None)
        if pos is None:
            logger.warning("close_position: %s not in open positions", ticker)
            return None

        gross_pnl = (exit_price - pos.entry_price) * pos.shares * pos.direction
        self._realised_pnl += gross_pnl

        record = {
            "ticker":       ticker,
            "direction":    pos.side,
            "entry_price":  pos.entry_price,
            "exit_price":   exit_price,
            "shares":       pos.shares,
            "notional":     pos.notional,
            "gross_pnl":    round(gross_pnl, 2),
            "return_pct":   round(
                (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100, 4
            ),
            "exit_reason":  exit_reason,
            "opened_at":    pos.opened_at,
            "closed_at":    datetime.now(timezone.utc),
            "signal_id":    pos.signal_id,
        }
        self._closed.append(record)

        logger.info(
            "POSITION CLOSED: %s %s %d@%.2f -> %.2f  pnl=$%.2f  reason=%s  total_open=%d",
            ticker, pos.side, pos.shares, pos.entry_price, exit_price,
            gross_pnl, exit_reason, self.n_open,
        )
        return record

    # ------------------------------------------------------------------
    # Stop / TP checks
    # ------------------------------------------------------------------

    def check_stops(self, prices: dict[str, float]) -> list[dict]:
        """
        Check all open positions against their stop-loss and take-profit levels.

        Call this on every price update (e.g. every bar or tick).

        Parameters
        ----------
        prices : {ticker: current_price}

        Returns
        -------
        list of closed trade dicts for positions that hit stop or TP.
        """
        triggered = []
        for ticker, price in prices.items():
            pos = self._positions.get(ticker)
            if pos is None:
                continue
            pos.update_price(price)

            if pos.direction == 1:   # long
                if pos.stop_loss is not None and price <= pos.stop_loss:
                    record = self.close_position(ticker, price, exit_reason="sl")
                    if record:
                        triggered.append(record)
                elif pos.take_profit is not None and price >= pos.take_profit:
                    record = self.close_position(ticker, price, exit_reason="tp")
                    if record:
                        triggered.append(record)
            else:                    # short
                if pos.stop_loss is not None and price >= pos.stop_loss:
                    record = self.close_position(ticker, price, exit_reason="sl")
                    if record:
                        triggered.append(record)
                elif pos.take_profit is not None and price <= pos.take_profit:
                    record = self.close_position(ticker, price, exit_reason="tp")
                    if record:
                        triggered.append(record)

        return triggered

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def portfolio_pnl(self) -> dict:
        """
        Return a summary of current portfolio P&L.

        Includes both unrealised (open positions) and realised (closed).
        """
        return {
            "n_open":              self.n_open,
            "open_tickers":        self.open_tickers,
            "total_notional":      round(self.total_notional, 2),
            "unrealised_pnl":      round(self.total_unrealised_pnl, 2),
            "realised_pnl":        round(self._realised_pnl, 2),
            "total_pnl":           round(self.total_unrealised_pnl + self._realised_pnl, 2),
            "n_closed_trades":     len(self._closed),
        }

    def positions_df(self) -> pd.DataFrame:
        """Return all open positions as a structured DataFrame."""
        if not self._positions:
            return pd.DataFrame()
        rows = [
            {
                "ticker":         t,
                "side":           p.side,
                "shares":         p.shares,
                "entry_price":    p.entry_price,
                "current_price":  p.current_price,
                "notional":       p.notional,
                "unrealised_pnl": round(p.unrealised_pnl, 2),
                "return_pct":     round(p.return_pct, 4),
                "stop_loss":      p.stop_loss,
                "take_profit":    p.take_profit,
                "opened_at":      p.opened_at,
                "signal_id":      p.signal_id,
            }
            for t, p in self._positions.items()
        ]
        return pd.DataFrame(rows).set_index("ticker")

    def closed_trades_df(self) -> pd.DataFrame:
        """Return all closed trades as a structured DataFrame."""
        if not self._closed:
            return pd.DataFrame()
        return pd.DataFrame(self._closed)

    def print_positions(self) -> None:
        """Pretty-print current open positions."""
        sep = "=" * 65
        logger.info(sep)
        logger.info("  OPEN POSITIONS  (%d / %d)", self.n_open, self.max_positions)
        logger.info(sep)
        if not self._positions:
            logger.info("  (none)")
        else:
            df = self.positions_df()
            for col in df.columns:
                logger.info("  %s", col)
            logger.info("\n%s", df.to_string())
        summary = self.portfolio_pnl()
        logger.info("  Total notional : $%,.0f", summary["total_notional"])
        logger.info("  Unrealised P&L : $%+,.2f", summary["unrealised_pnl"])
        logger.info("  Realised P&L   : $%+,.2f", summary["realised_pnl"])
        logger.info(sep)
