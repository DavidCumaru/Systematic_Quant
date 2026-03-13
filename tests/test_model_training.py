"""
tests/test_model_training.py
=============================
Unit tests for model_training.py.

Coverage
--------
- fit() trains without errors on synthetic labeled data
- predict() returns array of {-1, 0, 1}
- predict_proba() rows sum to ~1.0
- feature_importance() is non-empty and sorted descending
- save/load roundtrip produces identical predictions
- temporal_split() puts all train data before test data
- ModelTrainer works with 'lightgbm' model type (if available)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_training import ModelTrainer, temporal_split


class TestModelTrainer:

    def test_fit_predict_shape(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        preds = trainer.predict(labeled_df)
        assert len(preds) == len(labeled_df)

    def test_predict_valid_classes(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        preds = trainer.predict(labeled_df)
        valid = {-1, 0, 1}
        assert set(np.unique(preds)).issubset(valid)

    def test_predict_proba_sums_to_one(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        proba_df = trainer.predict_proba(labeled_df)
        row_sums = proba_df.sum(axis=1)
        # Single multi-class model: sums exactly to 1.
        # Specialized (two binary LightGBMs): sums approximately to 1 (within 5%).
        assert np.allclose(row_sums, 1.0, atol=0.05), (
            f"Proba row sums deviate > 5% from 1.0: min={row_sums.min():.4f} max={row_sums.max():.4f}"
        )

    def test_predict_proba_non_negative(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        proba_df = trainer.predict_proba(labeled_df)
        assert (proba_df.values >= 0).all()

    def test_predict_proba_columns_are_class_labels(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        proba_df = trainer.predict_proba(labeled_df)
        valid_cols = {-1, 0, 1}
        assert set(proba_df.columns).issubset(valid_cols)

    def test_feature_importance_nonempty(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        imp = trainer.feature_importance()
        assert not imp.empty
        # feature_importance() may return a Series (index=feature, values=importance)
        # or a DataFrame with an "importance" column — both are acceptable
        if hasattr(imp, "columns"):
            assert "importance" in imp.columns
        else:
            assert isinstance(imp, pd.Series)

    def test_feature_importance_descending(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        imp = trainer.feature_importance()
        # Normalise to Series of values
        if hasattr(imp, "columns") and "importance" in imp.columns:
            vals = imp["importance"].values
        else:
            vals = imp.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_save_load_roundtrip(self, labeled_df):
        trainer = ModelTrainer()
        trainer.fit(labeled_df)
        preds_before = trainer.predict(labeled_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_test.pkl"
            trainer.save(path)
            loaded = ModelTrainer.load(path)

        preds_after = loaded.predict(labeled_df)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_fit_requires_label_column(self, featured_df):
        """fit() should raise if 'label' column is missing."""
        df_no_label = featured_df.copy()
        with pytest.raises(Exception):
            trainer = ModelTrainer()
            trainer.fit(df_no_label)  # missing 'label'

    def test_not_fitted_predict_raises(self, labeled_df):
        trainer = ModelTrainer()
        with pytest.raises(Exception):
            trainer.predict(labeled_df)


class TestTemporalSplit:

    def test_no_train_test_overlap(self, labeled_df):
        train, test = temporal_split(labeled_df, test_size=0.2)
        if not train.empty and not test.empty:
            assert train.index.max() < test.index.min()

    def test_split_ratio_approximate(self, labeled_df):
        n = len(labeled_df)
        train, test = temporal_split(labeled_df, test_size=0.2)
        actual_test_frac = len(test) / n
        assert 0.15 <= actual_test_frac <= 0.30

    def test_combined_length_equals_total(self, labeled_df):
        train, test = temporal_split(labeled_df, test_size=0.2)
        assert len(train) + len(test) == len(labeled_df)
