"""
factor_analysis.py
==================
Quantitative signal quality metrics — industry-standard alpha evaluation.

IC (Information Coefficient)
    Spearman rank correlation between model predictions and realized forward
    returns.  Computed per walk-forward fold to avoid look-ahead.

    IC > 0.02   : weak but potentially exploitable signal
    IC > 0.05   : meaningful alpha
    IC > 0.10   : strong signal (institutional-grade)

ICIR (IC Information Ratio)
    IC_mean / IC_std — measures consistency of the IC across time periods.
    ICIR > 0.5  : viable for trading
    ICIR > 1.0  : institutional-grade signal

Signal Decay
    IC at horizons 1..N bars. A strategy with IC decaying sharply after
    h=3 should have a short hold period (as ours does: TIME_STOP_BARS=3).

Factor Attribution
    OLS regression of strategy returns on market (SPY) returns to separate:
      alpha — skill / structural edge
      beta  — market exposure (systematic risk)

Turnover Analysis
    Fraction of bars where signal changes direction. High turnover erodes
    edge through transaction costs (break-even cost analysis included).

Usage
-----
    from factor_analysis import FactorAnalyzer
    fa  = FactorAnalyzer()
    ic  = fa.ic_summary(signals_df, prices)
    dec = fa.signal_decay(signals_df, prices)
    att = fa.factor_attribution(trades_df, spy_prices)
    to  = fa.turnover_analysis(signals_df, trades_df)
    fa.print_report(ic, dec, att, to)
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FactorAnalyzer
# ---------------------------------------------------------------------------

class FactorAnalyzer:
    """
    Computes IC, ICIR, signal decay, factor attribution and turnover.

    All methods are stateless — pass the relevant DataFrames each time.
    """

    # ------------------------------------------------------------------
    def information_coefficient(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series,
        horizon: int = 1,
    ) -> pd.Series:
        """
        Compute per-fold IC: Spearman(pred_rank, fwd_return_rank).

        Parameters
        ----------
        signals_df : WFV output with columns ['pred', 'fold']
        prices     : close price Series aligned to signals_df index
        horizon    : forward return horizon in bars

        Returns
        -------
        pd.Series of IC values indexed by fold_id
        """
        fwd_ret = prices.pct_change(horizon).shift(-horizon)

        merged = signals_df[["pred", "fold"]].copy()
        merged["fwd_ret"] = fwd_ret.reindex(merged.index)
        merged = merged.dropna()

        if merged.empty:
            return pd.Series(dtype=float, name=f"IC_h{horizon}")

        ic_by_fold: dict[int, float] = {}
        for fold_id, grp in merged.groupby("fold"):
            if len(grp) < 5:
                continue
            if grp["pred"].nunique() < 2:
                continue
            ic, _ = stats.spearmanr(grp["pred"], grp["fwd_ret"])
            ic_by_fold[int(fold_id)] = round(float(ic), 6)

        return pd.Series(ic_by_fold, name=f"IC_h{horizon}")

    # ------------------------------------------------------------------
    def ic_summary(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        IC, ICIR, t-stat and % positive-IC folds for multiple horizons.

        Returns
        -------
        pd.DataFrame indexed by horizon with columns:
          IC_mean, IC_std, ICIR, t_stat, pct_positive, n_folds
        """
        if horizons is None:
            horizons = [1, 2, 3, 5, 10]

        rows = []
        for h in horizons:
            ic_series = self.information_coefficient(signals_df, prices, horizon=h)
            if ic_series.empty:
                continue

            ic_mean = ic_series.mean()
            ic_std  = ic_series.std()
            n       = len(ic_series)
            icir    = ic_mean / ic_std if ic_std > 1e-12 else np.nan
            t_stat  = ic_mean / (ic_std / np.sqrt(n)) if (ic_std > 1e-12 and n > 1) else np.nan

            rows.append({
                "horizon":       h,
                "IC_mean":       round(ic_mean, 4),
                "IC_std":        round(ic_std,  4),
                "ICIR":          round(icir,    4),
                "t_stat":        round(t_stat,  4),
                "pct_positive":  round((ic_series > 0).mean(), 4),
                "n_folds":       n,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("horizon")
        logger.info("IC Summary:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------
    def signal_decay(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series,
        max_horizon: int = 10,
    ) -> pd.DataFrame:
        """
        IC at horizons 1..max_horizon — reveals how fast the edge decays.

        A sharp IC drop from h=1 to h=5 validates a short hold period
        (consistent with TIME_STOP_BARS=3 in this strategy).
        """
        return self.ic_summary(
            signals_df, prices,
            horizons=list(range(1, max_horizon + 1)),
        )

    # ------------------------------------------------------------------
    def factor_attribution(
        self,
        trades_df: pd.DataFrame,
        benchmark_prices: pd.Series,
        annual_factor: int = 252,
    ) -> dict:
        """
        Decompose strategy returns into alpha + beta * market_return.

        Uses OLS regression on trade-level returns:
            R_strategy_i ~ alpha + beta * R_market_at_exit_i

        Parameters
        ----------
        trades_df        : from BacktestEngine.trades_df()
        benchmark_prices : SPY (or other benchmark) daily close prices
        annual_factor    : bars-per-year for alpha annualisation

        Returns dict with:
          alpha_annualised, beta, r_squared, p_value_beta,
          information_ratio, tracking_error
        """
        if trades_df.empty or benchmark_prices.empty:
            return {}

        # Per-trade % return
        trade_rets = pd.Series(
            (trades_df["exit_price"] / trades_df["entry_price"] - 1).values
            * trades_df["direction"].map({"LONG": 1, "SHORT": -1}).values,
            index=pd.DatetimeIndex(trades_df["exit_time"]),
            name="strategy",
        )

        # Benchmark daily returns aligned to trade exit dates
        mkt_rets = benchmark_prices.pct_change().reindex(
            trade_rets.index, method="nearest", tolerance="2D"
        )

        merged = pd.concat([trade_rets, mkt_rets], axis=1).dropna()
        merged.columns = ["strategy", "market"]

        if len(merged) < 10:
            logger.warning("Too few aligned trades for factor attribution (%d).", len(merged))
            return {}

        y = merged["strategy"].values
        X = merged["market"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        ann_alpha      = intercept * annual_factor
        excess_returns = merged["strategy"] - slope * merged["market"]
        tracking_error = excess_returns.std() * np.sqrt(annual_factor)
        info_ratio     = ann_alpha / tracking_error if tracking_error > 1e-12 else np.nan

        result = {
            "alpha_annualised_pct": round(ann_alpha * 100,     4),
            "beta":                 round(slope,               4),
            "r_squared":            round(r_value ** 2,        4),
            "p_value_beta":         round(p_value,             4),
            "information_ratio":    round(info_ratio,          4),
            "tracking_error_pct":   round(tracking_error * 100, 4),
            "n_trades_used":        len(merged),
        }
        logger.info("Factor attribution: %s", result)
        return result

    # ------------------------------------------------------------------
    def turnover_analysis(
        self,
        signals_df: pd.DataFrame,
        trades_df: pd.DataFrame | None = None,
        commission_per_trade_usd: float = 1.0,
        slippage_bps: float = 5.0,
        initial_equity: float = 100_000.0,
    ) -> dict:
        """
        Measure signal turnover and estimate break-even transaction cost.

        Parameters
        ----------
        signals_df              : WFV output with 'pred' column
        trades_df               : optional — used for notional sizing
        commission_per_trade_usd: flat commission per side
        slippage_bps            : one-way slippage in basis points
        initial_equity          : for cost ratio calculation

        Returns dict with:
          total_signals, non_neutral_pct, turnover_rate, avg_run_length,
          estimated_annual_cost_usd, break_even_edge_bps
        """
        if signals_df.empty or "pred" not in signals_df.columns:
            return {}

        pred        = signals_df["pred"]
        total       = len(pred)
        non_neutral = int((pred != 0).sum())
        changes     = int((pred != pred.shift(1)).sum())

        avg_run = round(total / max(changes, 1), 2)

        # Annual cost estimate
        avg_notional = initial_equity * 0.10   # assume 10% position size
        slippage_usd = avg_notional * slippage_bps / 10_000
        cost_per_rt  = 2 * (commission_per_trade_usd + slippage_usd)

        # Trades per year: infer from signals date span
        if isinstance(signals_df.index, pd.DatetimeIndex) and len(signals_df.index) > 1:
            days_span = (signals_df.index[-1] - signals_df.index[0]).days
            years = max(days_span / 365.25, 1 / 52)
        else:
            years = 1.0

        n_trades_year     = non_neutral / years
        annual_cost_usd   = n_trades_year * cost_per_rt
        bep_bps           = (annual_cost_usd / (initial_equity / 100)) * 100  # bps of equity

        result = {
            "total_signals":          total,
            "non_neutral_pct":        round(non_neutral / total * 100, 2),
            "turnover_rate_pct":      round(changes / total * 100, 2),
            "avg_run_length_bars":    avg_run,
            "trades_per_year_est":    round(n_trades_year, 1),
            "annual_cost_est_usd":    round(annual_cost_usd, 2),
            "break_even_edge_bps":    round(bep_bps, 2),
        }
        logger.info("Turnover analysis: %s", result)
        return result

    # ------------------------------------------------------------------
    def print_report(
        self,
        ic_df: pd.DataFrame,
        decay_df: pd.DataFrame,
        attribution: dict,
        turnover: dict,
    ) -> None:
        """Pretty-print the complete factor analysis report."""
        sep = "=" * 60

        logger.info(sep)
        logger.info("  FACTOR ANALYSIS REPORT")
        logger.info(sep)

        if not ic_df.empty:
            logger.info("--- IC Summary (key horizons) ---")
            logger.info("\n%s", ic_df.to_string())
            ic1 = ic_df["IC_mean"].get(1, np.nan)
            icir1 = ic_df["ICIR"].get(1, np.nan)
            grade = "STRONG" if abs(icir1) > 1.0 else ("VIABLE" if abs(icir1) > 0.5 else "WEAK")
            logger.info("Signal grade at h=1: %s  (ICIR=%.4f)", grade, icir1)

        if not decay_df.empty:
            logger.info("--- Signal Decay (IC by horizon) ---")
            logger.info("\n%s", decay_df[["IC_mean", "ICIR"]].to_string())

        if attribution:
            logger.info("--- Factor Attribution ---")
            for k, v in attribution.items():
                logger.info("  %-30s  %s", k.replace("_", " ").title(), v)

        if turnover:
            logger.info("--- Turnover & Cost Analysis ---")
            for k, v in turnover.items():
                logger.info("  %-30s  %s", k.replace("_", " ").title(), v)

        logger.info(sep)
