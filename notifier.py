"""
notifier.py
===========
Sends trade signals and pipeline alerts via Telegram Bot API.

Setup
-----
1. Create a Telegram bot: message @BotFather, run /newbot
2. Get your chat_id: message @userinfobot
3. Set environment variables:
     set TELEGRAM_TOKEN=<your_bot_token>
     set TELEGRAM_CHAT_ID=<your_chat_id>

The notifier fails silently when credentials are missing — it never
blocks the trading pipeline.

Usage
-----
    from notifier import Notifier
    n = Notifier()
    n.signal(ticker="AAPL", direction="BUY", price=185.50,
             stop=183.00, tp=188.50, shares=10, confidence=0.74)
    n.alert("Pipeline run complete for AAPL")
"""

import logging
from typing import Optional

from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

logger = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


class Notifier:
    """
    Sends formatted messages to a Telegram chat.

    All methods return True on success, False on failure (no exceptions raised).
    When TELEGRAM_TOKEN or TELEGRAM_CHAT_ID are not set, all calls are no-ops.
    """

    def __init__(
        self,
        token: str = TELEGRAM_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID,
    ):
        self.token   = token
        self.chat_id = chat_id
        self._active = bool(token and chat_id and _REQUESTS_OK)

        if not _REQUESTS_OK:
            logger.debug("Notifier: 'requests' not installed — Telegram disabled.")
        elif not self._active:
            logger.debug(
                "Notifier: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set — "
                "set env vars to enable notifications."
            )

    # ------------------------------------------------------------------
    def signal(
        self,
        ticker: str,
        direction: str,
        price: float,
        stop: float,
        tp: float,
        shares: int,
        confidence: float,
        signal_id: str = "",
    ) -> bool:
        """Send a trade signal notification."""
        arrow  = "BUY" if direction.upper() == "BUY" else "SELL"
        emoji  = "+" if direction.upper() == "BUY" else "-"
        msg = (
            f"[SIGNAL {emoji}]\n"
            f"  Ticker:     {ticker}\n"
            f"  Direction:  {arrow}\n"
            f"  Entry:      ${price:.2f}\n"
            f"  Stop Loss:  ${stop:.2f}\n"
            f"  Take Profit:${tp:.2f}\n"
            f"  Shares:     {shares}\n"
            f"  Confidence: {confidence:.1%}\n"
            + (f"  ID: {signal_id}" if signal_id else "")
        )
        return self._send(msg)

    # ------------------------------------------------------------------
    def alert(self, text: str) -> bool:
        """Send a free-form alert message."""
        return self._send(f"[ALERT] {text}")

    # ------------------------------------------------------------------
    def performance(self, ticker: str, metrics: dict) -> bool:
        """Send a performance summary after a backtest or live session."""
        lines = [f"[PERFORMANCE] {ticker}"]
        keys  = [
            ("total_return_pct", "Total Return"),
            ("sharpe_ratio",     "Sharpe"),
            ("max_drawdown_pct", "Max DD"),
            ("win_rate_pct",     "Win Rate"),
            ("profit_factor",    "Profit Factor"),
            ("n_trades",         "Trades"),
        ]
        for key, label in keys:
            val = metrics.get(key)
            if val is not None:
                lines.append(f"  {label:<14} {val}")
        return self._send("\n".join(lines))

    # ------------------------------------------------------------------
    def _send(self, text: str) -> bool:
        if not self._active:
            return False
        try:
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = _requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text},
                timeout=5,
            )
            if not resp.ok:
                logger.warning("Telegram API error: %s", resp.text)
                return False
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False
