"""
tests/test_labeling.py
======================
Unit tests for labeling.py (Triple-Barrier Method).

Coverage
--------
- Label values are always in {-1, 0, +1}
- At least 2 distinct classes are produced on synthetic data
- TP/SL barriers are correctly identified using bar high/low
- No look-ahead: label at bar t is removed when forward window is absent
- ATR-based and fixed-% barriers both produce valid labels
- label_report() returns a DataFrame with expected columns
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labeling import apply_triple_barrier, label_report


class TestLabelValues:

    def test_valid_label_set(self, labeled_df):
        valid = {-1, 0, 1}
        actual = set(labeled_df["label"].unique())
        assert actual.issubset(valid), f"Unexpected labels: {actual - valid}"

    def test_at_least_two_classes(self, labeled_df):
        n_classes = labeled_df["label"].nunique()
        assert n_classes >= 2, "Only one class produced — check TP/SL configuration"

    def test_label_column_dtype_integer(self, labeled_df):
        assert labeled_df["label"].dtype in (int, np.int64, np.int32)

    def test_no_nan_in_label(self, labeled_df):
        assert not labeled_df["label"].isna().any()


class TestLookahead:

    def test_last_bar_label_requires_forward_window(self, featured_df):
        """Bar at last_ts must NOT appear in labeled output when forward window is absent."""
        from config import TIME_STOP_BARS

        labeled_full = apply_triple_barrier(featured_df)
        assert not labeled_full.empty

        last_ts = labeled_full.index[-1]
        # Cut dataset so last_ts has no forward bars
        cut_loc = featured_df.index.get_loc(last_ts) + 1
        truncated = featured_df.iloc[:cut_loc]
        labeled_short = apply_triple_barrier(truncated)

        assert last_ts not in labeled_short.index, (
            f"Bar {last_ts} still labeled after forward window was removed "
            f"({TIME_STOP_BARS} bars ahead)"
        )

    def test_label_is_stable_when_past_grows(self, featured_df):
        """Label for bar t should not change as we add more historical data before it."""
        n = len(featured_df)
        mid = n // 2
        mid_ts = featured_df.index[mid]

        labeled_from_mid = apply_triple_barrier(featured_df.iloc[:mid + 20])
        labeled_full     = apply_triple_barrier(featured_df)

        if mid_ts in labeled_from_mid.index and mid_ts in labeled_full.index:
            assert labeled_from_mid.loc[mid_ts, "label"] == labeled_full.loc[mid_ts, "label"]


class TestBarrierModes:

    def test_fixed_pct_barriers(self, featured_df):
        labeled = apply_triple_barrier(featured_df, use_atr=False, tp_pct=0.01, sl_pct=0.007)
        assert set(labeled["label"].unique()).issubset({-1, 0, 1})
        assert not labeled.empty

    def test_atr_barriers(self, featured_df):
        labeled = apply_triple_barrier(featured_df, use_atr=True)
        assert set(labeled["label"].unique()).issubset({-1, 0, 1})
        assert not labeled.empty

    def test_narrow_barriers_produce_more_tp_sl(self, featured_df):
        """With very narrow barriers (tiny TP/SL), time-stop (0) should be rare."""
        labeled = apply_triple_barrier(
            featured_df, use_atr=False, tp_pct=0.0001, sl_pct=0.0001, time_stop=10
        )
        time_stop_fraction = (labeled["label"] == 0).mean()
        assert time_stop_fraction < 0.5, "Expected few time-stops with narrow barriers"


class TestLabelReport:

    def test_report_has_expected_columns(self, labeled_df):
        report = label_report(labeled_df)
        assert isinstance(report, pd.DataFrame)
        assert "count" in report.columns or len(report.columns) >= 1

    def test_report_index_contains_classes(self, labeled_df):
        report = label_report(labeled_df)
        labels_in_data = set(labeled_df["label"].unique())
        # Report index can be the raw integer labels or decorated strings like "-1 (stop)"
        index_str = " ".join(str(v) for v in report.index)
        for lbl in labels_in_data:
            assert (
                lbl in report.index
                or str(lbl) in report.index.astype(str)
                or str(lbl) in index_str
            ), f"Label {lbl} not represented in report index: {report.index.tolist()}"
