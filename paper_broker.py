"""
paper_broker.py
===============
Local paper trading broker — zero API keys required.

Simulates realistic order fills using the next available yfinance price
(next-bar open) after signal submission.  State is persisted as a JSON
file so the portfolio survives process restarts.

Architecture
------------
  submit_order(signal)   — queue a pending order
  fill_pending()         — fetch next prices via yfinance and fill queued orders
  update_positions()     — mark-to-market open positions, check stops/TPs
  portfolio_state()      — returns a snapshot dict (equity, positions, trades)

Storage schema (JSON)
---------------------
  {
    "initial_equity": 100000.0,
    "cash": 97500.0,
    "pending": [ {order}, ... ],
    "open_positions": { "SPY": {position}, ... },
    "closed_trades":  [ {trade}, ... ]
  }

Usage
-----
    from paper_broker import PaperBroker
    broker = PaperBroker()
    broker.submit_order(signal_dict)
    broker.fill_pending()          # call once per bar / day
    state = broker.portfolio_state()
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

from config import PAPER_BROKER_PATH

logger = logging.getLogger(__name__)

_EMPTY_STATE = {
    "initial_equity": 100_000.0,
    "cash": 100_000.0,
    "pending": [],
    "open_positions": {},
    "closed_trades": [],
}


class PaperBroker:
    """
    Local paper broker with yfinance fill simulation.

    Parameters
    ----------
    initial_equity : starting cash (only used if no existing state file)
    state_path     : path to the JSON persistence file
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        state_path: Path = PAPER_BROKER_PATH,
    ):
        self._path = Path(state_path)
        self._state = self._load(initial_equity)

    # ------------------------------------------------------------------ I/O
    def _load(self, initial_equity: float) -> dict:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    state = json.load(f)
                logger.debug("PaperBroker: loaded state from %s", self._path)
                return state
            except Exception as exc:
                logger.warning("PaperBroker: corrupt state file — resetting. (%s)", exc)

        state = dict(_EMPTY_STATE)
        state["initial_equity"] = initial_equity
        state["cash"] = initial_equity
        self._save(state)
        return state

    def _save(self, state: Optional[dict] = None) -> None:
        s = state or self._state
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("PaperBroker: could not save state: %s", exc)

    # --------------------------------------------------------------- Orders
    def submit_order(self, signal: dict) -> dict:
        """
        Queue a pending order.  Fill is deferred to the next fill_pending() call.

        Parameters
        ----------
        signal : dict with keys: ticker, direction, entry_price,
                 stop_loss, take_profit, position_size, notional_usd,
                 confidence, signal_id (optional)

        Returns
        -------
        order dict with order_id and status='pending'
        """
        order = {
            "order_id":     signal.get("signal_id") or str(uuid.uuid4())[:8],
            "submitted_at": datetime.now(tz=timezone.utc).isoformat(),
            "ticker":       signal.get("ticker", ""),
            "direction":    signal.get("direction", "BUY").upper(),
            "qty":          int(signal.get("position_size", 0)),
            "est_price":    float(signal.get("entry_price", 0.0)),
            "stop_loss":    signal.get("stop_loss"),
            "take_profit":  signal.get("take_profit"),
            "confidence":   float(signal.get("confidence", 0.5)),
            "status":       "pending",
        }

        if not order["ticker"] or order["qty"] <= 0:
            logger.warning("PaperBroker: invalid order skipped — %s", order)
            return order

        self._state["pending"].append(order)
        self._save()
        logger.info(
            "PaperBroker: order queued  %s %s x%d  est=%.2f",
            order["direction"], order["ticker"], order["qty"], order["est_price"],
        )
        return order

    # --------------------------------------------------------------- Fills
    def fill_pending(self) -> list[dict]:
        """
        Attempt to fill all pending orders using the latest yfinance price.

        For each pending order:
          - Fetches the most recent close price via yfinance (1d bar, period=5d)
          - Uses that close as the fill price (simulates next-bar market fill)
          - Deducts notional from cash (BUY) or credits it (SELL/SHORT)
          - Registers an open position

        Returns list of filled order dicts.
        """
        if not self._state["pending"]:
            return []

        filled = []
        still_pending = []

        for order in self._state["pending"]:
            ticker = order["ticker"]
            price  = self._fetch_price(ticker)

            if price is None:
                still_pending.append(order)
                continue

            qty       = order["qty"]
            direction = order["direction"]
            notional  = round(price * qty, 2)

            # Check cash for BUY
            if direction == "BUY" and notional > self._state["cash"]:
                logger.warning(
                    "PaperBroker: insufficient cash for %s %s x%d  need=%.2f  have=%.2f",
                    direction, ticker, qty, notional, self._state["cash"],
                )
                still_pending.append(order)
                continue

            # Register open position (allow simple cover for existing short, etc.)
            pos_key = ticker
            self._state["open_positions"][pos_key] = {
                "ticker":      ticker,
                "direction":   direction,
                "qty":         qty,
                "entry_price": price,
                "notional":    notional,
                "stop_loss":   order.get("stop_loss"),
                "take_profit": order.get("take_profit"),
                "opened_at":   datetime.now(tz=timezone.utc).isoformat(),
                "order_id":    order["order_id"],
                "confidence":  order["confidence"],
            }

            # Adjust cash
            if direction == "BUY":
                self._state["cash"] -= notional
            else:
                self._state["cash"] += notional   # short proceeds

            order["fill_price"] = price
            order["fill_notional"] = notional
            order["status"] = "filled"
            order["filled_at"] = datetime.now(tz=timezone.utc).isoformat()
            filled.append(order)

            logger.info(
                "PaperBroker: FILLED  %s %s x%d @ %.2f  cash=%.2f",
                direction, ticker, qty, price, self._state["cash"],
            )

        self._state["pending"] = still_pending
        self._save()
        return filled

    # --------------------------------------------------- Mark-to-market / exits
    def update_positions(self) -> list[dict]:
        """
        Fetch current prices, mark-to-market open positions, and auto-close
        any that have hit their stop-loss or take-profit.

        Returns list of closed trade records.
        """
        if not self._state["open_positions"]:
            return []

        tickers = list(self._state["open_positions"].keys())
        prices  = self._fetch_prices(tickers)

        closed = []
        still_open = {}

        for ticker, pos in self._state["open_positions"].items():
            price = prices.get(ticker)
            if price is None:
                still_open[ticker] = pos
                continue

            direction  = pos["direction"]
            entry      = pos["entry_price"]
            qty        = pos["qty"]
            sl         = pos.get("stop_loss")
            tp         = pos.get("take_profit")

            exit_reason = None
            if direction == "BUY":
                if sl and price <= sl:
                    exit_reason = "stop_loss"
                elif tp and price >= tp:
                    exit_reason = "take_profit"
            else:  # SHORT
                if sl and price >= sl:
                    exit_reason = "stop_loss"
                elif tp and price <= tp:
                    exit_reason = "take_profit"

            if exit_reason:
                trade = self._close_position(ticker, pos, price, exit_reason)
                closed.append(trade)
            else:
                pos["current_price"] = price
                pos["unrealised_pnl"] = round(
                    (price - entry) * qty * (1 if direction == "BUY" else -1), 2
                )
                still_open[ticker] = pos

        self._state["open_positions"] = still_open
        self._save()
        return closed

    def close_position_manual(self, ticker: str, reason: str = "manual") -> Optional[dict]:
        """Manually close an open position at current market price."""
        pos = self._state["open_positions"].get(ticker)
        if pos is None:
            logger.warning("PaperBroker: no open position for %s", ticker)
            return None
        price = self._fetch_price(ticker) or pos["entry_price"]
        trade = self._close_position(ticker, pos, price, reason)
        self._state["open_positions"].pop(ticker, None)
        self._save()
        return trade

    def _close_position(self, ticker: str, pos: dict, exit_price: float, reason: str) -> dict:
        """Internal: record a closed trade and adjust cash."""
        direction = pos["direction"]
        qty       = pos["qty"]
        entry     = pos["entry_price"]

        if direction == "BUY":
            gross_pnl = round((exit_price - entry) * qty, 2)
            self._state["cash"] += round(exit_price * qty, 2)
        else:
            gross_pnl = round((entry - exit_price) * qty, 2)
            self._state["cash"] -= round(exit_price * qty, 2)

        trade = {
            "trade_id":    str(uuid.uuid4())[:8],
            "ticker":      ticker,
            "direction":   direction,
            "qty":         qty,
            "entry_price": entry,
            "exit_price":  exit_price,
            "gross_pnl":   gross_pnl,
            "exit_reason": reason,
            "opened_at":   pos.get("opened_at"),
            "closed_at":   datetime.now(tz=timezone.utc).isoformat(),
            "confidence":  pos.get("confidence", 0.0),
        }
        self._state["closed_trades"].append(trade)

        logger.info(
            "PaperBroker: CLOSED  %s %s x%d  entry=%.2f exit=%.2f  pnl=$%.2f  reason=%s",
            direction, ticker, qty, entry, exit_price, gross_pnl, reason,
        )
        return trade

    # ---------------------------------------------------------- State snapshot
    def portfolio_state(self) -> dict:
        """
        Returns a complete portfolio snapshot for the dashboard.

        Keys
        ----
        initial_equity, cash, market_value, total_equity,
        unrealised_pnl, realised_pnl, total_return_pct,
        open_positions (list), closed_trades (list), equity_curve (list)
        """
        open_pos = list(self._state["open_positions"].values())
        closed   = self._state["closed_trades"]

        market_value  = sum(
            p.get("current_price", p["entry_price"]) * p["qty"]
            for p in open_pos
            if p["direction"] == "BUY"
        )
        unrealised = sum(p.get("unrealised_pnl", 0.0) for p in open_pos)
        realised   = sum(t["gross_pnl"] for t in closed)
        total_eq   = round(self._state["cash"] + market_value, 2)
        init_eq    = self._state["initial_equity"]

        return {
            "initial_equity":  init_eq,
            "cash":            round(self._state["cash"], 2),
            "market_value":    round(market_value, 2),
            "total_equity":    total_eq,
            "unrealised_pnl":  round(unrealised, 2),
            "realised_pnl":    round(realised, 2),
            "total_return_pct": round((total_eq / init_eq - 1) * 100, 4),
            "n_open":          len(open_pos),
            "n_closed":        len(closed),
            "open_positions":  open_pos,
            "closed_trades":   closed,
        }

    def reset(self, initial_equity: Optional[float] = None) -> None:
        """Wipe all state and start fresh."""
        eq = initial_equity or self._state.get("initial_equity", 100_000.0)
        self._state = dict(_EMPTY_STATE)
        self._state["initial_equity"] = eq
        self._state["cash"] = eq
        self._save()
        logger.info("PaperBroker: state reset  initial_equity=%.2f", eq)

    # ---------------------------------------------------------- yfinance helpers
    @staticmethod
    def _fetch_price(ticker: str) -> Optional[float]:
        """Fetch the latest close price for a single ticker."""
        if not _YF_OK:
            return None
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                return None
            close = df["Close"]
            if hasattr(close, "iloc"):
                val = close.iloc[-1]
                if hasattr(val, "item"):
                    return float(val.item())
                return float(val)
            return None
        except Exception as exc:
            logger.warning("PaperBroker: price fetch failed for %s: %s", ticker, exc)
            return None

    @staticmethod
    def _fetch_prices(tickers: list[str]) -> dict[str, float]:
        """Fetch latest close prices for multiple tickers at once."""
        if not _YF_OK or not tickers:
            return {}
        try:
            raw = yf.download(
                tickers, period="5d", interval="1d",
                progress=False, auto_adjust=True,
            )
            close = raw["Close"] if "Close" in raw.columns else raw
            if close.empty:
                return {}
            last = close.iloc[-1]
            return {
                t: float(last[t])
                for t in tickers
                if t in last.index and pd.notna(last[t])
            }
        except Exception as exc:
            logger.warning("PaperBroker: batch price fetch failed: %s", exc)
            return {}
