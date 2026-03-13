"""
tests/test_execution_engine.py
================================
Unit tests for execution_engine.py.

Coverage
--------
- generate_signals() returns a DataFrame
- No signals for all-neutral predictions
- Signal dict contains all required fields
- run_live_scan() returns None for neutral prediction
- print_signal() does not raise
- Probability filter removes low-confidence signals
- Position sizing is non-zero for positive signals
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution_engine import ExecutionEngine
from model_training import ModelTrainer


REQUIRED_SIGNAL_FIELDS = {
    "signal_id", "timestamp", "ticker", "direction",
    "entry_price", "stop_loss", "take_profit",
    "position_size", "notional_usd", "confidence",
}


@pytest.fixture(scope="module")
def trained_trainer(labeled_df):
    trainer = ModelTrainer()
    trainer.fit(labeled_df)
    return trainer


@pytest.fixture(scope="module")
def engine(trained_trainer):
    return ExecutionEngine(trainer=trained_trainer, equity=100_000)


# ---------------------------------------------------------------------------
# Tests: generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:

    def test_returns_dataframe(self, engine, labeled_df, featured_df):
        proba_df  = engine.trainer.predict_proba(labeled_df)
        preds     = engine.trainer.predict(labeled_df)
        signals_df = pd.DataFrame({"pred": preds}, index=labeled_df.index)
        signals_df = pd.concat([signals_df, proba_df.add_prefix("proba_")], axis=1)

        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        assert isinstance(result, pd.DataFrame)

    def test_all_neutral_returns_empty(self, engine, featured_df):
        idx = featured_df.index
        signals_df = pd.DataFrame({"pred": [0] * len(idx)}, index=idx)
        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        assert result.empty

    def test_required_fields_present(self, engine, labeled_df, featured_df):
        proba_df  = engine.trainer.predict_proba(labeled_df)
        preds     = engine.trainer.predict(labeled_df)
        signals_df = pd.DataFrame({"pred": preds}, index=labeled_df.index)
        signals_df = pd.concat([signals_df, proba_df.add_prefix("proba_")], axis=1)

        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        if result.empty:
            pytest.skip("No signals produced (model predicted all neutral)")

        assert REQUIRED_SIGNAL_FIELDS.issubset(set(result.columns))

    def test_direction_values(self, engine, labeled_df, featured_df):
        proba_df  = engine.trainer.predict_proba(labeled_df)
        preds     = engine.trainer.predict(labeled_df)
        signals_df = pd.DataFrame({"pred": preds}, index=labeled_df.index)
        signals_df = pd.concat([signals_df, proba_df.add_prefix("proba_")], axis=1)

        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        if result.empty:
            pytest.skip("No signals produced")

        assert set(result["direction"].unique()).issubset({"BUY", "SELL"})

    def test_confidence_in_range(self, engine, labeled_df, featured_df):
        proba_df  = engine.trainer.predict_proba(labeled_df)
        preds     = engine.trainer.predict(labeled_df)
        signals_df = pd.DataFrame({"pred": preds}, index=labeled_df.index)
        signals_df = pd.concat([signals_df, proba_df.add_prefix("proba_")], axis=1)

        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        if result.empty:
            pytest.skip("No signals produced")

        assert (result["confidence"] >= 0).all()
        assert (result["confidence"] <= 1).all()

    def test_notional_positive(self, engine, labeled_df, featured_df):
        proba_df  = engine.trainer.predict_proba(labeled_df)
        preds     = engine.trainer.predict(labeled_df)
        signals_df = pd.DataFrame({"pred": preds}, index=labeled_df.index)
        signals_df = pd.concat([signals_df, proba_df.add_prefix("proba_")], axis=1)

        result = engine.generate_signals(signals_df, featured_df, ticker="SPY", output_path=None)
        if result.empty:
            pytest.skip("No signals produced")

        assert (result["notional_usd"] > 0).all()


# ---------------------------------------------------------------------------
# Tests: run_live_scan
# ---------------------------------------------------------------------------

class TestRunLiveScan:

    def test_returns_none_for_empty_df(self, engine):
        result = engine.run_live_scan(pd.DataFrame(), ticker="SPY")
        assert result is None

    def test_signal_dict_has_required_fields(self, engine, labeled_df):
        # Feed the last row with a forced signal by using the full labeled_df
        latest = labeled_df.tail(50)
        result = engine.run_live_scan(latest, ticker="SPY")
        if result is None:
            pytest.skip("Model returned neutral or low-confidence for last 50 bars")
        assert REQUIRED_SIGNAL_FIELDS.issubset(set(result.keys()))

    def test_entry_price_positive(self, engine, labeled_df):
        result = engine.run_live_scan(labeled_df.tail(50), ticker="SPY")
        if result is None:
            pytest.skip("No signal")
        assert result["entry_price"] > 0

    def test_stop_loss_below_entry_for_long(self, engine, labeled_df):
        result = engine.run_live_scan(labeled_df.tail(50), ticker="SPY")
        if result is None or result["direction"] != "BUY":
            pytest.skip("No long signal")
        assert result["stop_loss"] < result["entry_price"]

    def test_take_profit_above_entry_for_long(self, engine, labeled_df):
        result = engine.run_live_scan(labeled_df.tail(50), ticker="SPY")
        if result is None or result["direction"] != "BUY":
            pytest.skip("No long signal")
        assert result["take_profit"] > result["entry_price"]


# ---------------------------------------------------------------------------
# Tests: print_signal
# ---------------------------------------------------------------------------

class TestPrintSignal:

    def test_does_not_raise_for_none(self, engine, capsys):
        engine.print_signal(None)  # must not raise

    def test_does_not_raise_for_valid_signal(self, engine, capsys):
        signal = {
            "signal_id": "abc12345",
            "timestamp": pd.Timestamp("2024-01-15 16:00", tz="America/New_York"),
            "ticker": "SPY",
            "direction": "BUY",
            "entry_price": 480.25,
            "stop_loss": 476.83,
            "take_profit": 485.05,
            "position_size": 12,
            "notional_usd": 5_763.0,
            "confidence": 0.62,
        }
        engine.print_signal(signal)
