"""
labeling.py
===========
Triple-Barrier labeling for day/swing trade classification.

Labels
------
  +1  ->  take-profit barrier touched first
  -1  ->  stop-loss barrier touched first
   0  ->  time-stop reached (neutral / flat)

Two barrier modes
-----------------
  Fixed %     : barriers at entry * (1 +/- pct)  -- simple, easy to tune
  ATR-based   : barriers at entry +/- ATR * multiplier
                Adapts to current volatility: wider in choppy markets,
                tighter in calm markets. Produces more balanced labels.

Consistency with backtest
-------------------------
Barriers are checked against bar HIGH (for TP) and bar LOW (for SL),
matching exactly how the BacktestEngine simulates fills.
This eliminates the label/backtest inconsistency that arises when
using close prices for labeling but high/low for execution.

No look-ahead: label for bar t uses only bars [t+1, t+TIME_STOP_BARS].
"""

import logging

import numpy as np
import pandas as pd

from config import (
    ATR_BARRIER_MULT_SL,
    ATR_BARRIER_MULT_TP,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TIME_STOP_BARS,
    USE_ATR_BARRIERS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core single-trade labeler (uses high/low arrays for realism)
# ---------------------------------------------------------------------------

def _label_single_hl(
    highs: np.ndarray,
    lows: np.ndarray,
    tp_level: float,
    sl_level: float,
) -> int:
    """
    Scan forward bars using high/low to detect barrier touches.
    Consistent with BacktestEngine fill logic.
    """
    for h, l in zip(highs, lows):
        if h >= tp_level:
            return 1
        if l <= sl_level:
            return -1
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    df: pd.DataFrame,
    tp_pct: float = TAKE_PROFIT_PCT,
    sl_pct: float = STOP_LOSS_PCT,
    time_stop: int = TIME_STOP_BARS,
    use_atr: bool = USE_ATR_BARRIERS,
) -> pd.DataFrame:
    """
    Attach a 'label' column using the Triple-Barrier Method.

    Parameters
    ----------
    df        : DataFrame with columns [close, high, low] and DatetimeIndex.
                If use_atr=True, must also have an 'atr' column (from
                feature_engineering.build_features).
    tp_pct    : fixed take-profit fraction (used when use_atr=False)
    sl_pct    : fixed stop-loss fraction   (used when use_atr=False)
    time_stop : max bars to hold before flat exit
    use_atr   : if True, use ATR-based dynamic barriers

    Returns
    -------
    pd.DataFrame with 'label' column appended; last time_stop rows dropped.
    """
    df = df.copy()

    if use_atr and "atr" not in df.columns:
        logger.warning("use_atr=True but 'atr' column missing; falling back to fixed %.")
        use_atr = False

    close_arr = df["close"].to_numpy()
    high_arr  = df["high"].to_numpy()
    low_arr   = df["low"].to_numpy()
    atr_arr   = df["atr"].to_numpy() if use_atr else None

    n      = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - time_stop):
        entry = close_arr[i]

        if use_atr:
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            tp_level = entry + atr_val * ATR_BARRIER_MULT_TP
            sl_level = entry - atr_val * ATR_BARRIER_MULT_SL
        else:
            tp_level = entry * (1 + tp_pct)
            sl_level = entry * (1 - sl_pct)

        fwd_h = high_arr[i + 1 : i + 1 + time_stop]
        fwd_l = low_arr [i + 1 : i + 1 + time_stop]

        labels[i] = _label_single_hl(fwd_h, fwd_l, tp_level, sl_level)

    # Invalidate last time_stop rows (no valid forward window)
    label_series = pd.Series(labels, index=df.index, dtype=float)
    label_series.iloc[-time_stop:] = np.nan

    df["label"] = label_series
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    counts = df["label"].value_counts().sort_index()
    mode   = "ATR-based" if use_atr else f"fixed tp={tp_pct:.3%} sl={sl_pct:.3%}"
    logger.info(
        "Labels applied [%s]: total=%d  [-1: %d  0: %d  +1: %d]  time_stop=%d bars",
        mode, len(df),
        counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0),
        time_stop,
    )
    return df


def label_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return label distribution summary as a DataFrame."""
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    report = pd.DataFrame({
        "count": counts,
        "pct":   (counts / total * 100).round(2),
    })
    report.index = report.index.map(
        {-1: "-1 (stop)", 0: "0 (neutral)", 1: "+1 (target)"}
    )
    report.index.name = "label"
    return report
