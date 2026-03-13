"""
regime_detection.py
===================
Market regime classification for conditional strategy evaluation.

Regimes
-------
  0  — Bear  / High-Volatility  (negative rolling return, high vol)
  1  — Sideways / Ranging       (near-zero return, moderate vol)
  2  — Bull  / Trending         (positive rolling return, low vol)

Methods
-------
  GMM (default)
    Gaussian Mixture Model on [rolling_return, rolling_vol] features.
    No extra dependencies beyond scikit-learn (already required).

  HMM (optional)
    Hidden Markov Model on log-returns — captures regime transitions
    more naturally than GMM (regimes have memory / persistence).
    Requires: pip install hmmlearn

Usage
-----
    from regime_detection import RegimeDetector
    rd  = RegimeDetector(method="gmm")
    reg = rd.fit_predict(spy_prices)   # Series: 0/1/2 per bar

    breakdown = rd.performance_by_regime(trades_df, reg)
    print(breakdown)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

REGIME_NAMES = {0: "Bear", 1: "Sideways", 2: "Bull"}


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Fits a regime model on price data and classifies each bar.

    Parameters
    ----------
    method      : "gmm" (default) or "hmm"
    n_regimes   : number of distinct regimes (default 3)
    window      : rolling window for feature computation (bars)
    random_state: reproducibility seed
    """

    def __init__(
        self,
        method: str = "gmm",
        n_regimes: int = 3,
        window: int = 20,
        random_state: int = 42,
    ):
        self.method       = method if (method == "hmm" and _HMM_OK) else "gmm"
        self.n_regimes    = n_regimes
        self.window       = window
        self.random_state = random_state
        self._model       = None
        self._scaler      = None
        self._rank_map: dict[int, int] = {}

        if method == "hmm" and not _HMM_OK:
            logger.warning(
                "hmmlearn not installed — falling back to GMM. "
                "Install with: pip install hmmlearn"
            )

    # ------------------------------------------------------------------
    def fit_predict(self, prices: pd.Series) -> pd.Series:
        """
        Fit the regime model and classify all bars.

        Returns pd.Series of int {0,1,2} aligned to prices.index.
        0=Bear, 1=Sideways, 2=Bull (relabelled by mean return).
        """
        if self.method == "hmm":
            return self._fit_hmm(prices)
        return self._fit_gmm(prices)

    # ------------------------------------------------------------------
    def _feature_matrix(self, prices: pd.Series) -> pd.DataFrame:
        """Build [rolling_return, rolling_vol] feature DataFrame."""
        log_ret  = np.log(prices / prices.shift(1))
        roll_ret = log_ret.rolling(self.window, min_periods=self.window // 2).mean()
        roll_vol = log_ret.rolling(self.window, min_periods=self.window // 2).std()
        feat = pd.concat([roll_ret, roll_vol], axis=1).dropna()
        feat.columns = ["ret", "vol"]
        return feat

    # ------------------------------------------------------------------
    def _fit_gmm(self, prices: pd.Series) -> pd.Series:
        feat = self._feature_matrix(prices)
        if len(feat) < self.n_regimes * 10:
            logger.warning("Too few bars for GMM — returning neutral regime.")
            return pd.Series(1, index=prices.index, name="regime", dtype=int)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(feat.values)

        self._model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
            max_iter=200,
        )
        raw_labels = self._model.fit_predict(X)

        # Relabel by ascending mean return (0=Bear ... 2=Bull)
        cluster_means = {
            k: feat["ret"].values[raw_labels == k].mean()
            for k in range(self.n_regimes)
        }
        self._rank_map = {
            k: i for i, (k, _) in
            enumerate(sorted(cluster_means.items(), key=lambda x: x[1]))
        }
        relabelled = pd.Series(
            [self._rank_map[lbl] for lbl in raw_labels],
            index=feat.index,
            name="regime",
        )
        return (
            relabelled
            .reindex(prices.index, method="ffill")
            .fillna(1)
            .astype(int)
        )

    # ------------------------------------------------------------------
    def _fit_hmm(self, prices: pd.Series) -> pd.Series:
        log_ret = np.log(prices / prices.shift(1)).dropna()
        X = log_ret.values.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        self._model.fit(X)
        raw_labels = self._model.predict(X)

        cluster_means = {k: X[raw_labels == k].mean() for k in range(self.n_regimes)}
        self._rank_map = {
            k: i for i, (k, _) in
            enumerate(sorted(cluster_means.items(), key=lambda x: x[1]))
        }
        relabelled = pd.Series(
            [self._rank_map[lbl] for lbl in raw_labels],
            index=log_ret.index,
            name="regime",
        )
        return (
            relabelled
            .reindex(prices.index, method="ffill")
            .fillna(1)
            .astype(int)
        )

    # ------------------------------------------------------------------
    def performance_by_regime(
        self,
        trades_df: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Break down trade P&L by market regime at entry time.

        Returns pd.DataFrame indexed by regime name with columns:
          n_trades, win_rate_pct, avg_pnl_usd, total_pnl_usd,
          profit_factor, sharpe_proxy
        """
        if trades_df.empty or regimes.empty:
            return pd.DataFrame()

        trades = trades_df.copy()
        reg_sorted = regimes.sort_index()

        def _get_regime(ts: pd.Timestamp) -> int:
            pos = reg_sorted.index.searchsorted(ts, side="right") - 1
            return int(reg_sorted.iloc[max(pos, 0)])

        trades["regime"] = trades["entry_time"].apply(
            lambda ts: REGIME_NAMES.get(_get_regime(ts), "Unknown")
        )

        rows = []
        for regime_name, group in trades.groupby("regime"):
            pnl     = group["pnl"]
            winners = pnl[pnl > 0]
            losers  = pnl[pnl < 0]
            gross_l = abs(losers.sum())
            pf      = round(winners.sum() / gross_l, 4) if gross_l > 1e-12 else np.nan
            sp      = round(pnl.mean() / pnl.std(), 4)  if pnl.std() > 1e-12 else np.nan

            rows.append({
                "regime":        regime_name,
                "n_trades":      len(group),
                "win_rate_pct":  round(len(winners) / len(group) * 100, 2),
                "avg_pnl_usd":   round(pnl.mean(), 2),
                "total_pnl_usd": round(pnl.sum(),  2),
                "profit_factor": pf,
                "sharpe_proxy":  sp,
            })

        df = pd.DataFrame(rows).set_index("regime")
        logger.info("Regime breakdown:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------
    def regime_stats(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Summary: frequency (% of bars) and avg spell duration per regime.
        """
        freq = regimes.value_counts(normalize=True) * 100
        freq.index = freq.index.map(lambda x: REGIME_NAMES.get(int(x), str(x)))

        durations: dict[str, list[int]] = {v: [] for v in REGIME_NAMES.values()}
        cur_regime, cur_count = None, 0
        for val in regimes:
            name = REGIME_NAMES.get(int(val), str(val))
            if name == cur_regime:
                cur_count += 1
            else:
                if cur_regime is not None:
                    durations[cur_regime].append(cur_count)
                cur_regime, cur_count = name, 1
        if cur_regime:
            durations[cur_regime].append(cur_count)

        avg_dur = {
            name: round(np.mean(spells), 1) if spells else 0.0
            for name, spells in durations.items()
        }

        stats_df = pd.DataFrame({
            "frequency_pct":     freq.reindex(REGIME_NAMES.values()).fillna(0).round(2),
            "avg_duration_bars": pd.Series(avg_dur),
        })
        logger.info("Regime statistics:\n%s", stats_df.to_string())
        return stats_df
