"""
walk_forward.py
===============
Rolling Walk-Forward Validation (WFV) with IC tracking.

Walk-Forward scheme (rolling)
------------------------------
    [== TRAIN (24m) ==][= TEST (6m) =]
                  [== TRAIN (24m) ==][= TEST =]
                                ...

Walk-Forward scheme (expanding / anchored)
------------------------------------------
    [====== TRAIN (grows) ======][= TEST =]
    [============ TRAIN ========][= TEST =]
                                ...

The expanding window is more data-efficient but can be sensitive to
structural breaks (model trained on 2002 crisis influences 2023 predictions).
Rolling window is more adaptive and is the default.

Per-fold IC tracking
--------------------
After each fold the out-of-sample Information Coefficient (Spearman
correlation of predicted rank vs realized forward return) is computed.
This gives a fold-by-fold view of signal quality beyond simple accuracy.

MLflow integration
------------------
All fold-level and aggregate metrics are tracked in MLflow for
experiment comparison and hyperparameter audit trail.

Outputs
-------
- FoldResult per fold (accuracy, F1, IC, predictions DataFrame)
- signals_df : concatenated OOS predictions across all folds
- summary()  : aggregate metrics DataFrame
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats as scipy_stats

from config import (
    EMBARGO_BARS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    TEST_MONTHS,
    TRAIN_MONTHS,
    USE_MLFLOW,
)
from model_training import ModelTrainer

try:
    import mlflow
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold result data structure
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_id:     int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_train:     int
    n_test:      int
    accuracy:    float
    f1_macro:    float
    ic:          float = 0.0   # Information Coefficient for this fold
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Rolling or expanding walk-forward cross-validator for time-series ML.

    Parameters
    ----------
    df           : labeled DataFrame with DatetimeIndex (from labeling.py)
    train_months : months of training data per fold
    test_months  : months of test data per fold
    embargo_bars : bars purged from the end of each train fold
    expanding    : if True, use expanding (anchored) window instead of rolling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_months: int = TRAIN_MONTHS,
        test_months:  int = TEST_MONTHS,
        embargo_bars: int = EMBARGO_BARS,
        expanding:    bool = False,
    ):
        self.df           = df.sort_index()
        self.train_months = train_months
        self.test_months  = test_months
        self.embargo_bars = embargo_bars
        self.expanding    = expanding
        self.results:     list[FoldResult]   = []
        self.signals_df:  Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def _generate_windows(self) -> list[tuple]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.

        Rolling  : training window of fixed size slides forward.
        Expanding: training window always starts from day 1 and grows.

        Falls back to a single 70/30 chronological split when the dataset
        is too short for month-based windows.
        """
        idx   = self.df.index
        start = idx[0].normalize()
        end   = idx[-1].normalize()

        windows: list[tuple] = []

        if self.expanding:
            # Expanding window — train always from start
            test_start = start + relativedelta(months=self.train_months)
            while test_start < end:
                test_end = min(test_start + relativedelta(months=self.test_months), end)
                windows.append((start, test_start, test_start, test_end))
                test_start = test_end
        else:
            # Rolling window
            train_start = start
            while True:
                train_end  = train_start + relativedelta(months=self.train_months)
                test_start = train_end
                test_end   = test_start + relativedelta(months=self.test_months)

                if test_start >= end:
                    break
                if test_end > end:
                    test_end = end

                windows.append((train_start, train_end, test_start, test_end))
                train_start = test_start

        # Fallback: not enough data for month-based windows
        if not windows:
            available_days = max((end - start).days, 1)
            split_days     = int(available_days * 0.70)
            fb_train_end   = start + pd.Timedelta(days=split_days)
            logger.warning(
                "Only %d days available; need >= %d months. "
                "Falling back to single 70/30 chronological split.",
                available_days,
                self.train_months + self.test_months,
            )
            windows.append((start, fb_train_end, fb_train_end, end))

        return windows

    # ------------------------------------------------------------------
    def _compute_fold_ic(
        self,
        pred_df: pd.DataFrame,
        prices: Optional[pd.Series] = None,
    ) -> float:
        """
        Compute IC for a single fold.

        IC = Spearman(pred, fwd_return_1bar).
        Uses the label column as a proxy when prices are not passed
        (label is already a signed categorical target).
        """
        if "pred" not in pred_df.columns or "label" not in pred_df.columns:
            return 0.0

        # Use realized label as forward-return proxy (available OOS)
        pred  = pred_df["pred"].values
        label = pred_df["label"].values

        if len(pred) < 5 or np.std(pred) < 1e-12:
            return 0.0

        ic, _ = scipy_stats.spearmanr(pred, label)
        return round(float(ic), 6) if not np.isnan(ic) else 0.0

    # ------------------------------------------------------------------
    def run(
        self,
        model_type: Optional[str] = None,
        ticker: str = "",
    ) -> list[FoldResult]:
        """
        Execute the full walk-forward loop.

        Parameters
        ----------
        model_type : override config MODEL_TYPE (optional)
        ticker     : instrument name for MLflow tagging

        Returns list of FoldResult (one per completed fold).
        """
        windows = self._generate_windows()
        mode    = "expanding" if self.expanding else "rolling"
        logger.info(
            "Walk-forward [%s]: %d folds | train=%dm  test=%dm  embargo=%d",
            mode, len(windows), self.train_months, self.test_months, self.embargo_bars,
        )

        # MLflow setup
        _mlflow_active = USE_MLFLOW and _MLFLOW_OK
        if _mlflow_active:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            run = mlflow.start_run(run_name=ticker or "wfv")
            mlflow.log_params({
                "ticker":       ticker,
                "train_months": self.train_months,
                "test_months":  self.test_months,
                "n_folds":      len(windows),
                "model_type":   model_type or "default",
                "window_mode":  mode,
            })

        all_preds: list[pd.DataFrame] = []

        for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
            train_df = self.df[
                (self.df.index >= tr_s) & (self.df.index < tr_e)
            ]

            # Embargo: purge last N bars from train set whose Triple-Barrier
            # labels look forward into the test period.
            if self.embargo_bars > 0 and len(train_df) > self.embargo_bars:
                train_df = train_df.iloc[: -self.embargo_bars]

            test_df = self.df[
                (self.df.index >= te_s) & (self.df.index < te_e)
            ]

            if len(train_df) < 50 or len(test_df) < 10:
                logger.warning(
                    "Fold %d skipped — insufficient data (train=%d, test=%d)",
                    fold_id, len(train_df), len(test_df),
                )
                continue

            # Train fresh model on this fold
            kwargs  = {"model_type": model_type} if model_type else {}
            trainer = ModelTrainer(**kwargs)
            trainer.fit(train_df)

            # Evaluate
            metrics  = trainer.evaluate(test_df)

            # Collect OOS predictions + probabilities
            proba_df = trainer.predict_proba(test_df)
            pred_df  = pd.DataFrame({
                "label": test_df["label"].values,
                "pred":  trainer.predict(test_df),
            }, index=test_df.index)
            pred_df  = pd.concat([pred_df, proba_df.add_prefix("proba_")], axis=1)
            pred_df["fold"] = fold_id

            # Per-fold IC
            fold_ic = self._compute_fold_ic(pred_df)

            all_preds.append(pred_df)

            result = FoldResult(
                fold_id=fold_id,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                n_train=len(train_df),
                n_test=len(test_df),
                accuracy=metrics["accuracy"],
                f1_macro=metrics["f1_macro"],
                ic=fold_ic,
                predictions=pred_df,
            )
            self.results.append(result)

            logger.info(
                "Fold %2d | train[%s->%s] n=%d | test[%s->%s] n=%d | "
                "acc=%.4f  F1=%.4f  IC=%.4f",
                fold_id,
                tr_s.date(), tr_e.date(), len(train_df),
                te_s.date(), te_e.date(), len(test_df),
                metrics["accuracy"], metrics["f1_macro"], fold_ic,
            )

            if _mlflow_active:
                mlflow.log_metrics({
                    f"fold_{fold_id}_accuracy": metrics["accuracy"],
                    f"fold_{fold_id}_f1_macro": metrics["f1_macro"],
                    f"fold_{fold_id}_ic":       fold_ic,
                }, step=fold_id)

        if all_preds:
            self.signals_df = pd.concat(all_preds).sort_index()

        # Aggregate MLflow metrics
        if _mlflow_active and self.results:
            accs = [r.accuracy for r in self.results]
            f1s  = [r.f1_macro for r in self.results]
            ics  = [r.ic       for r in self.results]
            mlflow.log_metrics({
                "mean_accuracy": float(np.mean(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "mean_ic":       float(np.mean(ics)),
                "icir":          float(np.mean(ics) / np.std(ics))
                                 if np.std(ics) > 1e-12 else 0.0,
                "n_folds_run":   len(self.results),
            })
            mlflow.end_run()

        return self.results

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """
        Aggregate per-fold metrics into a summary DataFrame.

        Returns pd.DataFrame with columns:
          fold_id, train_start, train_end, test_start, test_end,
          n_train, n_test, accuracy, f1_macro, ic
        """
        if not self.results:
            raise RuntimeError("No results — call .run() first.")

        rows = []
        for r in self.results:
            rows.append({
                "fold_id":     r.fold_id,
                "train_start": r.train_start.date(),
                "train_end":   r.train_end.date(),
                "test_start":  r.test_start.date(),
                "test_end":    r.test_end.date(),
                "n_train":     r.n_train,
                "n_test":      r.n_test,
                "accuracy":    round(r.accuracy, 4),
                "f1_macro":    round(r.f1_macro, 4),
                "ic":          round(r.ic,       4),
            })

        summary_df = pd.DataFrame(rows).set_index("fold_id")
        means = summary_df[["accuracy", "f1_macro", "ic"]].mean()
        ic_std = summary_df["ic"].std()
        icir   = means["ic"] / ic_std if ic_std > 1e-12 else np.nan

        logger.info(
            "WFV summary — mean acc=%.4f  F1=%.4f  IC=%.4f  ICIR=%.4f  folds=%d",
            means["accuracy"], means["f1_macro"], means["ic"], icir,
            len(self.results),
        )
        return summary_df
