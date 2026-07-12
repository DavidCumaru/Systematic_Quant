"""
core.py
=======
Shared signal decision logic used by both BacktestEngine and ExecutionEngine.

Having one source of truth for signal filtering (trend, probability) and
SL/TP/sizing calculations prevents live vs. backtest divergence.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: int          # +1 BUY, -1 SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float


def check_trend_filter(
    price: float,
    direction: int,
    ma_value: float,
    use_trend: bool,
) -> bool:
    """
    Return True if the signal PASSES the trend filter (allowed to trade).

    Logic (identical to BacktestEngine):
      - BUY only when price >= MA  (with the trend)
      - SELL only when price <= MA (with the trend)
    """
    if not use_trend or pd.isna(ma_value):
        return True
    if direction == 1 and price < ma_value:
        return False
    if direction == -1 and price > ma_value:
        return False
    return True


def compute_sl_tp(
    entry_price: float,
    direction: int,
    sl_pct: float,
    tp_pct: float,
) -> tuple[float, float]:
    """
    Compute stop-loss and take-profit prices.

    Returns (stop_loss, take_profit).
    """
    if direction == 1:
        stop_loss   = round(entry_price * (1 - sl_pct),  4)
        take_profit = round(entry_price * (1 + tp_pct), 4)
    else:
        stop_loss   = round(entry_price * (1 + sl_pct),  4)
        take_profit = round(entry_price * (1 - tp_pct), 4)
    return stop_loss, take_profit


def should_trade(
    pred: int,
    proba: float,
    price: float,
    ma_value: float,
    params: dict,
) -> bool:
    """
    Decide whether a model prediction passes all signal filters.

    Applies probability filter and trend filter using the same rules
    as both BacktestEngine and ExecutionEngine. The caller is responsible
    for computing fill prices, SL/TP, and sizing (engine-specific).

    Parameters
    ----------
    pred     : model prediction (+1 BUY, -1 SELL, 0 neutral)
    proba    : model confidence for the predicted class
    price    : current close price (for trend filter comparison)
    ma_value : rolling MA value at this bar (for trend filter)
    params   : per-ticker params dict (from load_ticker_params)

    Returns True if the signal passes all filters.
    """
    if pred == 0:
        return False

    min_proba = params.get("min_proba_threshold", 0.58)
    use_trend = params.get("use_trend_filter", True)

    if proba < min_proba:
        return False

    if not check_trend_filter(price, pred, ma_value, use_trend):
        return False

    return True
