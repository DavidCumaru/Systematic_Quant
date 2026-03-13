"""
model_training.py
=================
Trains classifiers on labeled feature data.

Model types
-----------
  "lightgbm"            — single 3-class LightGBM (fast, accurate)
  "lightgbm_specialized" — two binary LightGBMs: one for LONG, one for SHORT.
                           Each model specialises in its own market regime,
                           reducing the confusion inherent in 3-class prediction.
  "random_forest"       — sklearn RF (legacy)
  "gradient_boosting"   — sklearn GBM (legacy)

Optuna integration
------------------
  When USE_OPTUNA=True each fold runs a 20-trial Bayesian search over
  LightGBM hyperparams using a 3-fold inner time-series CV.
  Best params are used for the final per-fold fit.

Two-pass feature selection
--------------------------
  Pass 1: fit to get importances.
  Pass 2: drop features below MIN_FEATURE_IMPORTANCE and re-fit.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# LightGBM inside sklearn Pipeline produces a harmless feature-name warning
# because StandardScaler returns numpy arrays (no column names). Suppress it.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CLASS_WEIGHT,
    LEARNING_RATE,
    MAX_DEPTH,
    MIN_FEATURE_IMPORTANCE,
    MIN_PROBA_THRESHOLD,
    MIN_SAMPLES_LEAF,
    MODEL_TYPE,
    MODELS_DIR,
    N_ESTIMATORS,
    NUM_LEAVES,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    USE_OPTUNA,
)
from feature_engineering import get_feature_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import lightgbm as lgb
    _LIGHTGBM_OK = True
except ImportError:
    _LIGHTGBM_OK = False
    logger.warning("lightgbm not installed — falling back to GradientBoosting.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    _OPTUNA_OK = False


# ---------------------------------------------------------------------------
# Helper: temporal train/test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n     = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]


# ---------------------------------------------------------------------------
# Optuna tuner
# ---------------------------------------------------------------------------

def _tune_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Run Optuna Bayesian search for LightGBM hyperparams.

    Uses a 3-fold inner time-series CV (no shuffling).
    Returns the best param dict found.
    """
    if not (_OPTUNA_OK and _LIGHTGBM_OK):
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        clf = lgb.LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        n = len(X)
        fold_size = max(n // 4, 30)
        scores = []
        for k in range(1, 4):
            t_end = k * fold_size
            v_end = min((k + 1) * fold_size, n)
            if t_end >= v_end:
                continue
            X_tr, X_v = X[:t_end], X[t_end:v_end]
            y_tr, y_v = y[:t_end], y[t_end:v_end]
            if len(np.unique(y_tr)) < 2:
                continue
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_v)
            scores.append(f1_score(y_v, preds, average="macro", zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best F1=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Unified trainer supporting multiple model backends.

    For "lightgbm_specialized": internally trains a LONG binary model and a
    SHORT binary model, then combines them at prediction time. This allows
    each model to specialise in its own market regime rather than learning
    a confused 3-way classification.

    For all other types: single Pipeline (StandardScaler + classifier).
    """

    def __init__(self, model_type: str = MODEL_TYPE):
        self.model_type    = model_type
        self.feature_cols: list[str] = []

        # For specialized mode
        self._long_trainer:  Optional["ModelTrainer"] = None
        self._short_trainer: Optional["ModelTrainer"] = None

        # For standard mode
        self.pipeline: Optional[Pipeline] = None

        if model_type != "lightgbm_specialized":
            self._build_pipeline()

    # ------------------------------------------------------------------
    def _build_pipeline(self) -> None:
        """Construct sklearn Pipeline for the configured model type."""
        mt = self.model_type

        if mt == "lightgbm" and _LIGHTGBM_OK:
            clf = lgb.LGBMClassifier(
                n_estimators=N_ESTIMATORS,
                num_leaves=NUM_LEAVES,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                min_child_samples=MIN_SAMPLES_LEAF,
                class_weight=CLASS_WEIGHT,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1,
            )
        elif mt == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH if MAX_DEPTH != -1 else None,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                class_weight=CLASS_WEIGHT,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        else:
            # gradient_boosting or lightgbm fallback when lgb not installed
            clf = GradientBoostingClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH if MAX_DEPTH != -1 else 4,
                learning_rate=LEARNING_RATE,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_SEED,
                subsample=0.8,
            )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "ModelTrainer":
        """
        Fit on *train_df*.

        For specialized mode: trains separate Long and Short binary models.
        For standard mode: two-pass feature selection + fit.
        """
        if self.model_type == "lightgbm_specialized":
            return self._fit_specialized(train_df)
        return self._fit_standard(train_df)

    # ------------------------------------------------------------------
    def _fit_standard(self, train_df: pd.DataFrame) -> "ModelTrainer":
        self.feature_cols = get_feature_names(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["label"].values

        logger.info(
            "Training %s on %d samples, %d features",
            self.model_type, len(X), len(self.feature_cols),
        )

        # --- Optuna hyperparameter tuning ---
        if USE_OPTUNA and _OPTUNA_OK and self.model_type == "lightgbm" and _LIGHTGBM_OK:
            best_params = _tune_lightgbm(X, y)
            if best_params:
                clf = lgb.LGBMClassifier(
                    **best_params,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                )
                self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # Pass 1: fit to get importances
        self.pipeline.fit(X, y)
        importances = self.pipeline.named_steps["clf"].feature_importances_

        # Pass 2: feature selection
        mask    = importances >= MIN_FEATURE_IMPORTANCE
        n_kept  = int(mask.sum())
        n_total = len(self.feature_cols)

        if n_kept >= 5 and n_kept < n_total:
            selected = [f for f, keep in zip(self.feature_cols, mask) if keep]
            logger.info(
                "Feature selection: %d -> %d features (threshold=%.4f)",
                n_total, n_kept, MIN_FEATURE_IMPORTANCE,
            )
            self.feature_cols = selected
            X = train_df[self.feature_cols].values
            self._build_pipeline()
            self.pipeline.fit(X, y)
        else:
            logger.info(
                "Feature selection: keeping all %d features (n_kept=%d)",
                n_total, n_kept,
            )

        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------
    def _fit_specialized(self, train_df: pd.DataFrame) -> "ModelTrainer":
        """
        Train two binary classifiers:
          - Long model:  predicts P(label == +1)  vs rest
          - Short model: predicts P(label == -1) vs rest

        Each model can focus on the patterns specific to its direction,
        avoiding the confusion of mixed 3-class training.
        """
        logger.info(
            "Training SPECIALIZED Long+Short models on %d samples",
            len(train_df),
        )

        # Binary labels for each specialised model
        df_long  = train_df.copy(); df_long["label"]  = (train_df["label"] == 1).astype(int)
        df_short = train_df.copy(); df_short["label"] = (train_df["label"] == -1).astype(int)

        self._long_trainer  = ModelTrainer(model_type="lightgbm")
        self._short_trainer = ModelTrainer(model_type="lightgbm")

        self._long_trainer.fit(df_long)
        self._short_trainer.fit(df_short)

        # Use the feature set selected by the long model (they should be similar)
        self.feature_cols = self._long_trainer.feature_cols
        logger.info("Specialized training complete.")
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return class predictions {-1, 0, +1}."""
        self._check_fitted()

        if self.model_type == "lightgbm_specialized":
            return self._predict_specialized(df)

        return self.pipeline.predict(df[self.feature_cols].values)

    def _predict_specialized(self, df: pd.DataFrame) -> np.ndarray:
        long_p  = self._long_proba(df)
        short_p = self._short_proba(df)
        result  = np.zeros(len(df), dtype=int)
        result[long_p  >= MIN_PROBA_THRESHOLD] = 1
        result[short_p >= MIN_PROBA_THRESHOLD] = -1
        # Conflict resolution: take the stronger signal
        both = (long_p >= MIN_PROBA_THRESHOLD) & (short_p >= MIN_PROBA_THRESHOLD)
        result[both] = np.where(long_p[both] >= short_p[both], 1, -1)
        return result

    def _long_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self._long_trainer.predict_proba(df)
        return p[1].values if 1 in p.columns else np.zeros(len(df))

    def _short_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self._short_trainer.predict_proba(df)
        return p[1].values if 1 in p.columns else np.zeros(len(df))

    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return class probabilities as DataFrame with columns matching classes.
        For specialized: columns are {-1, 0, +1}.
        """
        self._check_fitted()

        if self.model_type == "lightgbm_specialized":
            return self._predict_proba_specialized(df)

        X     = df[self.feature_cols].values
        proba = self.pipeline.predict_proba(X)
        classes = self.pipeline.named_steps["clf"].classes_
        return pd.DataFrame(proba, index=df.index, columns=classes)

    def _predict_proba_specialized(self, df: pd.DataFrame) -> pd.DataFrame:
        long_p  = self._long_proba(df)
        short_p = self._short_proba(df)
        neutral = np.clip(1.0 - long_p - short_p, 0.0, 1.0)
        return pd.DataFrame(
            {-1: short_p, 0: neutral, 1: long_p},
            index=df.index,
        )

    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute classification metrics on *test_df*."""
        self._check_fitted()
        y_true = test_df["label"].values
        y_pred = self.predict(test_df)

        accuracy = (y_true == y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report   = classification_report(y_true, y_pred, zero_division=0)

        logger.info("Evaluation -> accuracy=%.4f  F1-macro=%.4f", accuracy, f1_macro)
        logger.info("\n%s", report)

        return {"accuracy": accuracy, "f1_macro": f1_macro, "report": report}

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """Return sorted feature importances (descending)."""
        self._check_fitted()
        if self.model_type == "lightgbm_specialized":
            # Average of long and short importances
            imp_l = self._long_trainer.feature_importance()
            imp_s = self._short_trainer.feature_importance()
            combined = (imp_l.reindex(imp_l.index.union(imp_s.index), fill_value=0)
                        + imp_s.reindex(imp_l.index.union(imp_s.index), fill_value=0)) / 2
            return combined.sort_values(ascending=False)

        clf = self.pipeline.named_steps["clf"]
        return (
            pd.Series(clf.feature_importances_, index=self.feature_cols)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted model to disk."""
        self._check_fitted()
        if path is None:
            path = MODELS_DIR / f"model_{self.model_type}.pkl"
        payload = {
            "model_type":    self.model_type,
            "feature_cols":  self.feature_cols,
            "pipeline":      self.pipeline,
            "_long_trainer": self._long_trainer,
            "_short_trainer":self._short_trainer,
        }
        joblib.dump(payload, path)
        logger.info("Model saved -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelTrainer":
        """Load a previously saved ModelTrainer."""
        data    = joblib.load(path)
        trainer = cls.__new__(cls)
        trainer.model_type     = data["model_type"]
        trainer.feature_cols   = data["feature_cols"]
        trainer.pipeline       = data.get("pipeline")
        trainer._long_trainer  = data.get("_long_trainer")
        trainer._short_trainer = data.get("_short_trainer")
        logger.info("Model loaded from -> %s", path)
        return trainer

    # ------------------------------------------------------------------
    def shap_importance(
        self,
        df: pd.DataFrame,
        max_display: int = 20,
    ) -> pd.Series:
        """
        Compute SHAP-based feature importance (mean |SHAP value| per feature).

        SHAP (SHapley Additive exPlanations) provides model-agnostic feature
        attribution that is more reliable than native split-based importances:
          - Split importance is biased toward high-cardinality features
          - SHAP values are consistent and locally accurate

        Requires: pip install shap

        Parameters
        ----------
        df          : feature DataFrame (same schema used during training)
        max_display : return top-N features only

        Returns
        -------
        pd.Series sorted descending by mean |SHAP value|
        """
        self._check_fitted()

        try:
            import shap
        except ImportError:
            logger.warning(
                "shap not installed — run: pip install shap  "
                "Falling back to native feature importances."
            )
            return self.feature_importance().head(max_display)

        # For specialized model use the long trainer's LightGBM as reference
        if self.model_type == "lightgbm_specialized":
            trainer_ref = self._long_trainer
        else:
            trainer_ref = self

        feat_cols = trainer_ref.feature_cols
        X = df[feat_cols].values

        clf = trainer_ref.pipeline.named_steps["clf"]

        try:
            if hasattr(clf, "booster_"):
                # LightGBM — use TreeExplainer (fast, exact)
                explainer  = shap.TreeExplainer(clf)
                shap_vals  = explainer.shap_values(X)
            else:
                # Fallback for RF / GBM — use generic Explainer
                explainer  = shap.Explainer(clf, X)
                shap_vals  = explainer(X).values

            # shap_vals may be list[array] for multi-class
            if isinstance(shap_vals, list):
                # Average absolute SHAP across all classes
                abs_mean = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0
                )
            elif shap_vals.ndim == 3:
                abs_mean = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                abs_mean = np.abs(shap_vals).mean(axis=0)

            return (
                pd.Series(abs_mean, index=feat_cols, name="shap_importance")
                .sort_values(ascending=False)
                .head(max_display)
            )

        except Exception as exc:
            logger.warning("SHAP computation failed: %s — using native importances.", exc)
            return self.feature_importance().head(max_display)

    # ------------------------------------------------------------------
    def feature_stability(self, folds_importances: list[pd.Series]) -> pd.DataFrame:
        """
        Measure feature importance stability across walk-forward folds.

        A feature that ranks highly in every fold is more trustworthy than
        one that is important only occasionally (potential overfitting).

        Parameters
        ----------
        folds_importances : list of feature_importance() Series, one per fold

        Returns
        -------
        pd.DataFrame with columns: mean_rank, std_rank, mean_importance, cv
          (cv = coefficient of variation — lower is more stable)
        """
        if not folds_importances:
            return pd.DataFrame()

        # Convert importance to rank per fold (1 = most important)
        ranks = pd.concat([
            s.rank(ascending=False).rename(f"fold_{i}")
            for i, s in enumerate(folds_importances)
        ], axis=1)

        importances = pd.concat([
            s.rename(f"fold_{i}")
            for i, s in enumerate(folds_importances)
        ], axis=1)

        result = pd.DataFrame({
            "mean_rank":        ranks.mean(axis=1).round(2),
            "std_rank":         ranks.std(axis=1).round(2),
            "mean_importance":  importances.mean(axis=1).round(6),
            "cv":               (importances.std(axis=1) / importances.mean(axis=1).replace(0, np.nan)).round(4),
        }).sort_values("mean_rank")

        logger.info("Feature stability (top 10):\n%s", result.head(10).to_string())
        return result

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        is_standard     = self.pipeline is not None
        is_specialized  = (self._long_trainer is not None and
                           self._short_trainer is not None)
        if not (is_standard or is_specialized):
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
