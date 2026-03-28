"""
notifier.py
===========
Log-based notifier — prints trade signals and pipeline alerts to the
Python logging system (console + file).  No external service or API key
required.  Implements the same interface as the previous Telegram notifier
so that call-sites in main.py need no changes.
"""

import logging

logger = logging.getLogger(__name__)


class Notifier:
    """
    Writes formatted messages to the Python logger.

    All methods return True unconditionally (maintains backward compatibility).
    """

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
        arrow = "BUY" if direction.upper() == "BUY" else "SELL"
        logger.info(
            "[SIGNAL] %s %s  entry=%.2f  sl=%.2f  tp=%.2f  shares=%d  conf=%.1f%%%s",
            ticker, arrow, price, stop, tp, shares, confidence * 100,
            f"  id={signal_id}" if signal_id else "",
        )
        return True

    # ------------------------------------------------------------------
    def alert(self, text: str) -> bool:
        logger.info("[ALERT] %s", text)
        return True

    # ------------------------------------------------------------------
    def performance(self, ticker: str, metrics: dict) -> bool:
        keys = [
            ("total_return_pct", "Total Return"),
            ("sharpe_ratio",     "Sharpe"),
            ("max_drawdown_pct", "Max DD"),
            ("win_rate_pct",     "Win Rate"),
            ("profit_factor",    "Profit Factor"),
            ("n_trades",         "Trades"),
        ]
        parts = "  ".join(
            f"{label}={metrics[k]}"
            for k, label in keys
            if k in metrics
        )
        logger.info("[PERFORMANCE] %s  %s", ticker, parts)
        return True
