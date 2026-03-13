"""
portfolio_manager.py
====================
Multi-ticker portfolio construction with correlation-aware position sizing.

Allocation Methods
------------------
  equal         — uniform weight across all active signals
  risk_parity   — inverse-volatility weighting (default)
                  Each ticker contributes equal RISK, not equal capital.
  min_variance  — Markowitz minimum-variance via scipy.optimize
  max_sharpe    — Tangency portfolio (maximize Sharpe via optimization)

Additional features
-------------------
  - Correlation matrix with flagged high-correlation pairs
  - Diversification ratio (DR > 1 = genuine diversification benefit)
  - Portfolio-level summary across all tickers
  - Break-even correlation threshold filter (skip correlated redundant signals)

Usage
-----
    from portfolio_manager import PortfolioManager

    pm = PortfolioManager(equity=100_000, method="risk_parity")

    # Compute weights given active signals
    weights = pm.compute_weights(
        tickers=["SPY", "QQQ", "TLT"],
        returns=returns_df,           # DataFrame of daily log-returns
        active_signals={"SPY": 1, "QQQ": 1, "TLT": -1},
    )
    # Convert to integer share counts
    shares = pm.allocate_shares(weights, prices={"SPY": 520.0, "QQQ": 440.0, "TLT": 95.0})

    # Portfolio-level summary
    summary = pm.portfolio_summary(all_metrics, all_equity_curves)
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import INITIAL_EQUITY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Coordinates position sizing across multiple tickers.

    Parameters
    ----------
    equity                     : current account equity in USD
    method                     : allocation method (see module docstring)
    max_position_pct           : hard cap per ticker (default 25%)
    max_correlation_threshold  : skip signal if correlation to existing
                                 position exceeds this (default 0.80)
    """

    def __init__(
        self,
        equity: float = INITIAL_EQUITY,
        method: str = "risk_parity",
        max_position_pct: float = 0.25,
        max_correlation_threshold: float = 0.80,
    ):
        self.equity                    = equity
        self.method                    = method
        self.max_position_pct          = max_position_pct
        self.max_correlation_threshold = max_correlation_threshold

    # ------------------------------------------------------------------
    def compute_weights(
        self,
        tickers: list[str],
        returns: pd.DataFrame,
        active_signals: dict[str, int],
    ) -> dict[str, float]:
        """
        Compute portfolio weights for tickers with active signals.

        Parameters
        ----------
        tickers        : full universe list
        returns        : DataFrame of daily log-returns (columns = tickers)
        active_signals : {ticker: direction} where direction in {-1, +1}

        Returns
        -------
        dict {ticker: weight} — weights sum to <= 1.0
        """
        active = [t for t in tickers if active_signals.get(t, 0) != 0]
        if not active:
            return {}

        # Filter to tickers with returns data
        avail = [t for t in active if t in returns.columns]
        if not avail:
            return {t: 1.0 / len(active) for t in active}

        ret_sub = returns[avail].dropna(how="all")

        # Dispatch to allocation method
        if self.method == "equal":
            w = {t: 1.0 / len(avail) for t in avail}
        elif self.method == "risk_parity":
            w = self._risk_parity(avail, ret_sub)
        elif self.method == "min_variance":
            w = self._min_variance(avail, ret_sub)
        elif self.method == "max_sharpe":
            w = self._max_sharpe(avail, ret_sub)
        else:
            w = {t: 1.0 / len(avail) for t in avail}

        # Cap each position
        for t in w:
            w[t] = min(w[t], self.max_position_pct)

        # Normalise to sum <= 1
        total = sum(w.values())
        if total > 1e-12:
            w = {t: v / total for t, v in w.items()}

        logger.info("Portfolio weights [%s]: %s", self.method,
                    {t: round(v, 4) for t, v in w.items()})
        return w

    # ------------------------------------------------------------------
    def _risk_parity(self, tickers: list[str], returns: pd.DataFrame) -> dict[str, float]:
        """Inverse-volatility weighting: each ticker contributes equal risk."""
        vol     = returns[tickers].std().replace(0, np.nan).fillna(1.0)
        inv_vol = 1.0 / vol
        total   = inv_vol.sum()
        return {t: float(inv_vol[t] / total) for t in tickers}

    # ------------------------------------------------------------------
    def _min_variance(self, tickers: list[str], returns: pd.DataFrame) -> dict[str, float]:
        """Markowitz minimum-variance portfolio via SLSQP."""
        n   = len(tickers)
        cov = returns[tickers].cov().values

        if n < 2 or len(returns) < n + 5:
            return self._risk_parity(tickers, returns)

        def portfolio_var(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds      = [(0.0, self.max_position_pct)] * n
        w0          = np.ones(n) / n

        result = minimize(
            portfolio_var, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if result.success:
            return {t: float(w) for t, w in zip(tickers, result.x)}
        logger.warning("min_variance optimization failed — falling back to risk parity.")
        return self._risk_parity(tickers, returns)

    # ------------------------------------------------------------------
    def _max_sharpe(self, tickers: list[str], returns: pd.DataFrame) -> dict[str, float]:
        """Tangency portfolio: maximize Sharpe via SLSQP."""
        n   = len(tickers)
        mu  = returns[tickers].mean().values
        cov = returns[tickers].cov().values

        if n < 2 or len(returns) < n + 5:
            return self._risk_parity(tickers, returns)

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w))
            return -(port_ret / port_vol) if port_vol > 1e-12 else 0.0

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds      = [(0.0, self.max_position_pct)] * n
        w0          = np.ones(n) / n

        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if result.success:
            return {t: float(w) for t, w in zip(tickers, result.x)}
        logger.warning("max_sharpe optimization failed — falling back to risk parity.")
        return self._risk_parity(tickers, returns)

    # ------------------------------------------------------------------
    def allocate_shares(
        self,
        weights: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, int]:
        """
        Convert portfolio weights to integer share counts.

        Parameters
        ----------
        weights : {ticker: weight} from compute_weights()
        prices  : {ticker: entry_price_usd}

        Returns {ticker: n_shares}
        """
        shares: dict[str, int] = {}
        for ticker, weight in weights.items():
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue
            dollar_alloc = self.equity * weight
            n = max(1, int(dollar_alloc / price))
            shares[ticker] = n
        return shares

    # ------------------------------------------------------------------
    def correlation_matrix(
        self,
        returns: pd.DataFrame,
        flag_threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Return correlation matrix.

        If flag_threshold is provided, logs pairs that exceed it
        (these pairs offer little diversification benefit).
        """
        corr = returns.corr().round(4)

        if flag_threshold is not None:
            high_pairs = []
            cols = corr.columns.tolist()
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    if abs(corr.loc[a, b]) >= flag_threshold:
                        high_pairs.append(
                            f"{a}-{b}: {corr.loc[a, b]:.4f}"
                        )
            if high_pairs:
                logger.warning(
                    "High-correlation pairs (>= %.2f): %s",
                    flag_threshold, ", ".join(high_pairs),
                )

        return corr

    # ------------------------------------------------------------------
    def diversification_ratio(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
    ) -> float:
        """
        Diversification Ratio = weighted_avg_vol / portfolio_vol.

        DR = 1.0  -> no diversification benefit (all assets move together)
        DR > 1.0  -> diversification reduces portfolio risk below avg asset risk
        DR > 1.5  -> strong diversification (typical for equity + bond mix)
        """
        tickers = [t for t in weights if t in returns.columns]
        if not tickers:
            return 1.0

        w    = np.array([weights[t] for t in tickers])
        vols = returns[tickers].std().values
        cov  = returns[tickers].cov().values

        weighted_avg_vol = float(w @ vols)
        portfolio_vol    = float(np.sqrt(w @ cov @ w))

        dr = weighted_avg_vol / portfolio_vol if portfolio_vol > 1e-12 else 1.0
        logger.info("Diversification Ratio: %.4f", dr)
        return round(dr, 4)

    # ------------------------------------------------------------------
    def portfolio_summary(
        self,
        all_metrics: dict[str, dict],
        all_equity_curves: dict[str, pd.Series],
    ) -> dict:
        """
        Aggregate portfolio-level metrics across all tickers.

        Blends equity curves with equal weight to compute portfolio-level
        Sharpe and return.

        Parameters
        ----------
        all_metrics      : {ticker: metrics_dict} from stage_performance()
        all_equity_curves: {ticker: equity_curve}

        Returns dict with portfolio-level stats + cross-ticker summary.
        """
        if not all_metrics:
            return {}

        valid_metrics  = {t: m for t, m in all_metrics.items() if m}
        sharpes        = [m.get("sharpe_ratio", 0)      for m in valid_metrics.values()]
        returns_pct    = [m.get("total_return_pct", 0)  for m in valid_metrics.values()]
        max_dds        = [m.get("max_drawdown_pct", 0)  for m in valid_metrics.values()]
        win_rates      = [m.get("win_rate_pct", 0)      for m in valid_metrics.values()]
        profit_factors = [m.get("profit_factor", 0) or 0 for m in valid_metrics.values()]

        # Equal-weight blended equity curve
        portfolio_return = np.nan
        portfolio_sharpe = np.nan
        portfolio_maxdd  = np.nan

        curves = {t: c for t, c in all_equity_curves.items() if not c.empty}
        if curves:
            combined = pd.concat(curves.values(), axis=1).ffill()
            eq_norm  = combined.div(combined.iloc[0])   # normalise to 1
            blended  = eq_norm.mean(axis=1)

            portfolio_return = round((blended.iloc[-1] - 1) * 100, 2)

            rets = blended.pct_change().dropna()
            if rets.std() > 1e-12:
                # Use 252 (daily) — curves are daily bars
                portfolio_sharpe = round(
                    rets.mean() / rets.std() * np.sqrt(252), 4
                )

            peak = blended.cummax()
            dd   = (peak - blended) / peak.replace(0, np.nan)
            portfolio_maxdd = round(dd.max() * 100, 2)

        return {
            "n_tickers":               len(valid_metrics),
            "portfolio_return_pct":    portfolio_return,
            "portfolio_sharpe":        portfolio_sharpe,
            "portfolio_max_dd_pct":    portfolio_maxdd,
            "mean_ticker_sharpe":      round(float(np.mean(sharpes)),        4),
            "best_ticker_sharpe":      round(float(np.max(sharpes)),         4),
            "worst_ticker_sharpe":     round(float(np.min(sharpes)),         4),
            "mean_ticker_return_pct":  round(float(np.mean(returns_pct)),    2),
            "mean_max_drawdown_pct":   round(float(np.mean(max_dds)),        2),
            "mean_win_rate_pct":       round(float(np.mean(win_rates)),      2),
            "mean_profit_factor":      round(float(np.mean(profit_factors)), 4),
        }

    # ------------------------------------------------------------------
    def print_summary(self, summary: dict) -> None:
        """Pretty-print portfolio summary to logger."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("  PORTFOLIO-LEVEL SUMMARY")
        logger.info(sep)
        for k, v in summary.items():
            logger.info("  %-34s  %s", k.replace("_", " ").title(), v)
        logger.info(sep)
