"""
performance.py
==============
Institutional-grade performance analytics and visualisation.

Metrics computed
----------------
  Total Return          — cumulative net P&L / initial equity
  CAGR                  — compound annual growth rate (from actual date span)
  Sharpe Ratio          — annualised, auto-detects bar frequency
  Sortino Ratio         — downside-deviation adjusted
  Max Drawdown          — peak-to-trough equity decline
  Calmar Ratio          — CAGR / Max Drawdown
  MAR Ratio             — annualised return / max drawdown (same as Calmar)
  Omega Ratio           — probability-weighted ratio of gains to losses
  Win Rate              — fraction of profitable trades
  Profit Factor         — gross_profit / gross_loss
  Expectancy            — average P&L per trade
  Average Win / Loss    — mean winner and loser size
  Max consecutive W/L   — longest winning and losing streaks
  Trade count           — total, long, short
  Sharpe CI             — 95% bootstrap confidence interval
  Equity curve plot     — with drawdown panel

All Sharpe/Sortino calculations auto-detect bar frequency from the
equity curve DatetimeIndex to produce correctly annualised figures.

Bug fixed (v2):
  Previous code used BARS_PER_DAY_5M=78 for annualisation regardless of
  bar frequency.  For daily data this over-stated Sharpe by sqrt(78) ≈ 8.8x.
  Now uses _annualization_factor() which detects daily vs intraday.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_management import RiskMetrics

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Annualisation helper — detects bar frequency automatically
# ---------------------------------------------------------------------------

def _annualization_factor(equity_curve: pd.Series) -> int:
    """
    Infer the annualisation factor from the equity curve's DatetimeIndex.

    Returns bars-per-year so that:
        Sharpe = mean_bar_return / std_bar_return * sqrt(ann_factor)

    Frequency map
    -------------
      daily  (>= 20h median gap)  -> 252
      hourly (>= 55min gap)       -> 252 * 7  = 1764
      30-min (>= 25min gap)       -> 252 * 13 = 3276
      5-min  (default)            -> 252 * 78 = 19656
    """
    if len(equity_curve) < 2:
        return TRADING_DAYS_PER_YEAR

    idx = equity_curve.index
    if not isinstance(idx, pd.DatetimeIndex):
        return TRADING_DAYS_PER_YEAR

    # Sample up to 500 consecutive gaps to estimate median timedelta
    diffs = pd.Series(idx[1:] - idx[:-1]).iloc[:500]
    median_td = diffs.median()

    if median_td >= pd.Timedelta(hours=20):          # daily bars
        return TRADING_DAYS_PER_YEAR                 # 252
    elif median_td >= pd.Timedelta(minutes=55):      # 1-hour bars
        return TRADING_DAYS_PER_YEAR * 7             # 1764
    elif median_td >= pd.Timedelta(minutes=25):      # 30-min bars
        return TRADING_DAYS_PER_YEAR * 13            # 3276
    else:                                            # 5-min or finer
        return TRADING_DAYS_PER_YEAR * 78            # 19656


def _years_from_curve(equity_curve: pd.Series) -> float:
    """Compute actual elapsed years from the equity curve's date span."""
    if len(equity_curve) < 2:
        return 1.0
    idx = equity_curve.index
    if isinstance(idx, pd.DatetimeIndex):
        delta_sec = (idx[-1] - idx[0]).total_seconds()
        return max(delta_sec / (365.25 * 86400), 1 / 252)
    # Fallback: assume daily
    return max(len(equity_curve) / TRADING_DAYS_PER_YEAR, 1 / 252)


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    initial_equity: float,
) -> dict:
    """
    Compute the full set of institutional performance metrics.

    Parameters
    ----------
    trades_df     : DataFrame from BacktestEngine.trades_df()
    equity_curve  : pd.Series indexed by timestamp, values = equity USD
    initial_equity: float — starting account value

    Returns
    -------
    dict of metric_name -> value
    """
    if trades_df.empty or equity_curve.empty:
        logger.warning("No trades or equity curve available for metrics.")
        return {}

    pnl_series  = trades_df["pnl"]
    final_eq    = equity_curve.iloc[-1]
    total_return = (final_eq - initial_equity) / initial_equity

    # Correct CAGR from actual date span (not bar count)
    years = _years_from_curve(equity_curve)
    cagr  = (final_eq / initial_equity) ** (1 / years) - 1

    # Bar-level returns with auto-detected annualisation factor
    bar_returns = equity_curve.pct_change().dropna()
    ann_factor  = _annualization_factor(equity_curve)

    mean_ret = bar_returns.mean()
    std_ret  = bar_returns.std()
    down_std = bar_returns[bar_returns < 0].std()

    sharpe  = (mean_ret / std_ret  * np.sqrt(ann_factor)) if std_ret  > 1e-12 else np.nan
    sortino = (mean_ret / down_std * np.sqrt(ann_factor)) if down_std > 1e-12 else np.nan

    mdd    = RiskMetrics.max_drawdown(equity_curve)
    calmar = cagr / mdd if mdd > 1e-12 else np.nan

    # Omega ratio — probability-weighted gains / losses above threshold 0
    omega = _omega_ratio(bar_returns)

    # Trade-level statistics
    n_trades = len(pnl_series)
    winners  = pnl_series[pnl_series > 0]
    losers   = pnl_series[pnl_series < 0]

    win_rate      = len(winners) / n_trades if n_trades > 0 else 0.0
    gross_profit  = winners.sum()
    gross_loss    = abs(losers.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.nan
    expectancy    = pnl_series.mean()
    avg_win       = winners.mean() if not winners.empty else 0.0
    avg_loss      = losers.mean()  if not losers.empty  else 0.0

    # Consecutive win/loss streaks
    max_consec_wins, max_consec_losses = _consecutive_streaks(pnl_series)

    # VaR and CVaR (95%)
    var_95  = RiskMetrics.value_at_risk(bar_returns, 0.95)
    cvar_95 = RiskMetrics.expected_shortfall(bar_returns, 0.95)

    n_long  = (trades_df["direction"] == "LONG").sum()
    n_short = (trades_df["direction"] == "SHORT").sum()

    # Average holding period
    avg_hold = _avg_holding_period(trades_df)

    metrics = {
        # Returns
        "total_return_pct":     round(total_return * 100, 2),
        "cagr_pct":             round(cagr * 100, 2),
        # Risk-adjusted
        "sharpe_ratio":         round(sharpe,        4),
        "sortino_ratio":        round(sortino,        4),
        "calmar_ratio":         round(calmar,         4),
        "omega_ratio":          round(omega,          4),
        # Drawdown
        "max_drawdown_pct":     round(mdd * 100,      2),
        # Tail risk
        "var_95_pct":           round(var_95 * 100,   4),
        "cvar_95_pct":          round(cvar_95 * 100,  4),
        # Trade stats
        "win_rate_pct":         round(win_rate * 100, 2),
        "profit_factor":        round(profit_factor,  4),
        "expectancy_usd":       round(expectancy,     2),
        "avg_win_usd":          round(avg_win,        2),
        "avg_loss_usd":         round(avg_loss,       2),
        "max_consec_wins":      max_consec_wins,
        "max_consec_losses":    max_consec_losses,
        "avg_hold_days":        avg_hold,
        # Counts
        "n_trades":             n_trades,
        "n_long":               n_long,
        "n_short":              n_short,
        # P&L
        "gross_profit_usd":     round(gross_profit,   2),
        "gross_loss_usd":       round(gross_loss,     2),
        "final_equity_usd":     round(final_eq,       2),
        # Meta
        "years_tested":         round(years,          2),
        "ann_factor_used":      ann_factor,
    }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict to the logger."""
    divider = "=" * 60
    logger.info(divider)
    logger.info("  PERFORMANCE REPORT")
    logger.info(divider)
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        logger.info("  %-32s  %s", label, v)
    logger.info(divider)


# ---------------------------------------------------------------------------
# Bootstrap Sharpe Confidence Interval
# ---------------------------------------------------------------------------

def sharpe_confidence_interval(
    equity_curve: pd.Series,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval for the Sharpe ratio.

    Resamples bar-level returns with replacement to estimate sampling
    uncertainty.  A wide CI (e.g. [0.1, 2.5]) indicates insufficient
    history to be confident in the measured Sharpe.

    Parameters
    ----------
    equity_curve : pd.Series — equity indexed by datetime
    n_bootstrap  : number of bootstrap samples
    ci           : confidence level (default 0.95 -> 95% CI)

    Returns
    -------
    (lower_bound, upper_bound)
    """
    bar_returns = equity_curve.pct_change().dropna().values
    ann_factor  = _annualization_factor(equity_curve)

    if len(bar_returns) < 20:
        return (np.nan, np.nan)

    rng     = np.random.default_rng(42)
    sharpes = []

    for _ in range(n_bootstrap):
        sample = rng.choice(bar_returns, size=len(bar_returns), replace=True)
        std    = sample.std()
        if std > 1e-12:
            sharpes.append(sample.mean() / std * np.sqrt(ann_factor))

    if not sharpes:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    lo    = float(np.quantile(sharpes, alpha))
    hi    = float(np.quantile(sharpes, 1 - alpha))
    return (round(lo, 4), round(hi, 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _omega_ratio(bar_returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio = E[max(R - threshold, 0)] / E[max(threshold - R, 0)]

    > 1 means more probability-weighted gains than losses above threshold.
    An Omega > 1.5 is generally considered good.
    """
    r    = bar_returns.dropna()
    gain = (r - threshold).clip(lower=0).mean()
    loss = (threshold - r).clip(lower=0).mean()
    return float(gain / loss) if loss > 1e-12 else np.nan


def _consecutive_streaks(pnl: pd.Series) -> tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_wins = max_losses = cur_wins = cur_losses = 0
    for p in pnl:
        if p > 0:
            cur_wins  += 1
            cur_losses = 0
        elif p < 0:
            cur_losses += 1
            cur_wins   = 0
        else:
            cur_wins = cur_losses = 0
        max_wins   = max(max_wins,   cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def _avg_holding_period(trades_df: pd.DataFrame) -> float:
    """Average holding period in calendar days."""
    if trades_df.empty:
        return 0.0
    if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
        return 0.0
    try:
        durations = (
            pd.to_datetime(trades_df["exit_time"])
            - pd.to_datetime(trades_df["entry_time"])
        ).dt.total_seconds() / 86400
        return round(durations.mean(), 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    metrics: dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate a 4-panel chart:
      1. Equity curve with fill
      2. Drawdown series
      3. Per-trade P&L bar chart
      4. Rolling 30-trade win rate

    Parameters
    ----------
    equity_curve : pd.Series — timestamp-indexed account equity
    trades_df    : pd.DataFrame — trade records
    metrics      : dict — from compute_metrics()
    save_path    : Path — if provided, save the figure to disk
    """
    dd_series = RiskMetrics.drawdown_series(equity_curve) * 100

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    sharpe_str   = metrics.get("sharpe_ratio", "N/A")
    sortino_str  = metrics.get("sortino_ratio", "N/A")
    mdd_str      = metrics.get("max_drawdown_pct", "N/A")
    wr_str       = metrics.get("win_rate_pct", "N/A")
    omega_str    = metrics.get("omega_ratio", "N/A")
    cagr_str     = metrics.get("cagr_pct", "N/A")
    years_str    = metrics.get("years_tested", "N/A")

    fig.suptitle(
        f"Systematic Alpha — Backtest Results  ({years_str}y tested)\n"
        f"Sharpe={sharpe_str}  Sortino={sortino_str}  Omega={omega_str}  "
        f"CAGR={cagr_str}%  MaxDD={mdd_str}%  WinRate={wr_str}%",
        fontsize=10,
    )

    # Panel 1: Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, lw=1.2, color="#2196F3", label="Equity")
    ax1.fill_between(equity_curve.index, equity_curve.values,
                     equity_curve.min(), alpha=0.08, color="#2196F3")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title("Equity Curve")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    ax2 = axes[1]
    ax2.fill_between(dd_series.index, -dd_series.values, 0,
                     color="#F44336", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Per-trade P&L
    ax3 = axes[2]
    if not trades_df.empty:
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades_df["pnl"]]
        ax3.bar(range(len(trades_df)), trades_df["pnl"].values, color=colors, alpha=0.7)
        ax3.axhline(0, color="black", lw=0.8)
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("P&L (USD)")
        ax3.set_title("Per-Trade P&L")
        ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling win rate (30-trade window)
    ax4 = axes[3]
    if not trades_df.empty and len(trades_df) >= 10:
        win_binary = (trades_df["pnl"] > 0).astype(float)
        window     = min(30, len(win_binary))
        roll_wr    = win_binary.rolling(window).mean() * 100
        ax4.plot(range(len(roll_wr)), roll_wr.values, color="#9C27B0", lw=1.2)
        ax4.axhline(50, color="gray", lw=0.8, linestyle="--", label="50%")
        ax4.set_xlabel("Trade #")
        ax4.set_ylabel("Win Rate (%)")
        ax4.set_title(f"Rolling {window}-Trade Win Rate")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Equity curve chart saved -> %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def monthly_returns_table(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Compute month-by-month returns and return a pivot table
    (rows = year, columns = month).
    """
    monthly = equity_curve.resample("ME").last().pct_change().dropna() * 100
    tbl = pd.DataFrame({
        "year":  monthly.index.year,
        "month": monthly.index.month,
        "ret":   monthly.values,
    })
    pivot = tbl.pivot(index="year", columns="month", values="ret").round(2)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = month_names[:len(pivot.columns)]
    pivot["Annual"] = pivot.sum(axis=1).round(2)
    return pivot