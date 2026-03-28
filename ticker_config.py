"""
ticker_config.py
================
Per-ticker parameter management.

Each ticker can have independent values for every execution/backtest
parameter.  Values are loaded from data/ticker_params.json (written by
grid_search.py) and fall back to the global config defaults when a
key or file is missing.

Usage
-----
    from ticker_config import load_ticker_params, save_ticker_params

    params = load_ticker_params("QQQ")
    # {'min_proba_threshold': 0.52, 'stop_loss_pct': 0.007, ...}

    save_ticker_params("QQQ", best_params)
"""

import json
import logging
from pathlib import Path

from config import (
    MIN_PROBA_THRESHOLD,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TIME_STOP_BARS,
    TREND_MA_BARS,
    USE_TREND_FILTER,
)

logger = logging.getLogger(__name__)

PARAMS_FILE = Path(__file__).parent / "data" / "ticker_params.json"

# Global defaults — used as fallback for any missing key
DEFAULTS: dict = {
    "min_proba_threshold": MIN_PROBA_THRESHOLD,
    "stop_loss_pct":       STOP_LOSS_PCT,
    "take_profit_pct":     TAKE_PROFIT_PCT,
    "time_stop_bars":      TIME_STOP_BARS,
    "direction":           "both",        # "both" | "long_only" | "short_only"
    "regime_filter":       "all",         # "all" | "Bull" | "Bear" | "Sideways"
                                          # | "Bull+Sideways" | "Bear+Sideways" | "Bear+Bull"
    "use_trend_filter":    USE_TREND_FILTER,
    "trend_ma_bars":       TREND_MA_BARS,
}


def load_ticker_params(ticker: str) -> dict:
    """
    Return the params dict for *ticker*.

    Missing keys fall back to DEFAULTS.  If the file does not exist,
    returns a copy of DEFAULTS unchanged.
    """
    params = dict(DEFAULTS)
    if not PARAMS_FILE.exists():
        return params
    try:
        with open(PARAMS_FILE, encoding="utf-8") as f:
            all_params: dict = json.load(f)
        params.update(all_params.get(ticker, {}))
    except Exception as exc:
        logger.warning("ticker_config: could not load %s — %s", PARAMS_FILE, exc)
    return params


def save_ticker_params(ticker: str, params: dict) -> None:
    """Persist *params* for *ticker* into the JSON file."""
    all_params: dict = {}
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE, encoding="utf-8") as f:
                all_params = json.load(f)
        except Exception:
            pass
    all_params[ticker] = {k: v for k, v in params.items() if k in DEFAULTS}
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2)
    logger.info("ticker_config: saved params for %s -> %s", ticker, PARAMS_FILE)


def load_all_params() -> dict:
    """Return the full ticker_params.json as a dict (empty if missing)."""
    if not PARAMS_FILE.exists():
        return {}
    try:
        with open(PARAMS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_all_params() -> None:
    """Pretty-print the current per-ticker params."""
    data = load_all_params()
    if not data:
        print("No per-ticker params saved yet. Run grid_search.py first.")
        return
    header = f"{'Ticker':<6}  " + "  ".join(f"{k:<22}" for k in DEFAULTS)
    print(header)
    print("-" * len(header))
    for ticker, p in sorted(data.items()):
        vals = "  ".join(f"{str(p.get(k, DEFAULTS[k])):<22}" for k in DEFAULTS)
        print(f"{ticker:<6}  {vals}")
