"""
tests/test_data_pipeline.py
============================
Unit tests for data_pipeline.py using an in-memory SQLite database.

Coverage
--------
- _ensure_table() creates table with correct schema
- _upsert_bars() inserts new rows
- _upsert_bars() ignores duplicates (INSERT OR IGNORE)
- load_data() returns empty DataFrame for unknown ticker
- load_data() respects start/end date filters
- Incremental update: only new bars are added
- _last_stored_dt() returns None for empty table
- get_available_tickers() lists all tables
"""

import sys
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_pipeline as dp


# ---------------------------------------------------------------------------
# Fixture: patch DB_PATH to an isolated temp file for each test
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect all DB operations to a temporary file."""
    db_path = tmp_path / "test_market_data.db"
    monkeypatch.setattr(dp, "DB_PATH", db_path)
    return db_path


def _make_bars(n: int = 20, start: str = "2023-01-02", tz: str = "America/New_York") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n, tz=tz)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open":   close,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": 1_000_000.0,
    }, index=idx)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestEnsureTable:

    def test_creates_table(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        dp._ensure_table(conn, "AAPL")
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "AAPL" in tables

    def test_idempotent(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        dp._ensure_table(conn, "AAPL")
        dp._ensure_table(conn, "AAPL")  # second call must not raise
        conn.close()


# ---------------------------------------------------------------------------
# Upsert tests
# ---------------------------------------------------------------------------

class TestUpsertBars:

    def test_inserts_new_rows(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        bars = _make_bars(10)
        n = dp._upsert_bars(conn, "SPY", bars)
        conn.close()
        assert n == 10

    def test_ignores_duplicates(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        bars = _make_bars(10)
        dp._upsert_bars(conn, "SPY", bars)
        n2 = dp._upsert_bars(conn, "SPY", bars)  # same data again
        conn.close()
        assert n2 == 0  # INSERT OR IGNORE -> 0 new rows

    def test_empty_df_returns_zero(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        n = dp._upsert_bars(conn, "SPY", pd.DataFrame())
        conn.close()
        assert n == 0


# ---------------------------------------------------------------------------
# Load / filter tests
# ---------------------------------------------------------------------------

class TestLoadData:

    def test_empty_for_unknown_ticker(self, tmp_db):
        df = dp.load_data("UNKNOWN_XYZ")
        assert df.empty

    def test_loads_persisted_bars(self, tmp_db):
        bars = _make_bars(15)
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "GLD", bars)
        conn.close()
        loaded = dp.load_data("GLD")
        assert len(loaded) == 15

    def test_start_filter(self, tmp_db):
        bars = _make_bars(30, start="2023-01-02")
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "TLT", bars)
        conn.close()

        mid_date = bars.index[15].date().isoformat()
        loaded = dp.load_data("TLT", start=mid_date)
        assert (loaded.index >= pd.Timestamp(mid_date, tz="America/New_York")).all()

    def test_end_filter(self, tmp_db):
        bars = _make_bars(30, start="2023-01-02")
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "IEF", bars)
        conn.close()

        end_date = bars.index[10].date().isoformat()
        loaded = dp.load_data("IEF", end=end_date)
        assert (loaded.index <= pd.Timestamp(end_date, tz="America/New_York")).all()

    def test_index_is_datetime(self, tmp_db):
        bars = _make_bars(10)
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "QQQ", bars)
        conn.close()
        loaded = dp.load_data("QQQ")
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_sorted_ascending(self, tmp_db):
        bars = _make_bars(20)
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "IWM", bars)
        conn.close()
        loaded = dp.load_data("IWM")
        assert loaded.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# last_stored_dt tests
# ---------------------------------------------------------------------------

class TestLastStoredDt:

    def test_returns_none_for_empty_table(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        dp._ensure_table(conn, "EMPTY")
        last = dp._last_stored_dt(conn, "EMPTY")
        conn.close()
        assert last is None

    def test_returns_correct_last_dt(self, tmp_db):
        bars = _make_bars(10)
        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "SPY", bars)
        last = dp._last_stored_dt(conn, "SPY")
        conn.close()
        assert last is not None
        assert last.date() == bars.index[-1].date()


# ---------------------------------------------------------------------------
# Incremental update test
# ---------------------------------------------------------------------------

class TestIncrementalUpdate:

    def test_only_new_bars_added(self, tmp_db):
        first_batch  = _make_bars(20, start="2023-01-02")
        second_batch = _make_bars(30, start="2023-01-02")  # overlaps + extends

        conn = sqlite3.connect(tmp_db)
        dp._upsert_bars(conn, "SPY", first_batch)
        # Only send bars after the last stored timestamp
        last_dt = dp._last_stored_dt(conn, "SPY")
        new_bars = second_batch[second_batch.index > last_dt]
        n_new = dp._upsert_bars(conn, "SPY", new_bars)
        conn.close()

        assert n_new == len(new_bars)


# ---------------------------------------------------------------------------
# get_available_tickers
# ---------------------------------------------------------------------------

class TestGetAvailableTickers:

    def test_lists_all_tables(self, tmp_db):
        tickers = ["AAPL", "MSFT", "GOOG"]
        conn = sqlite3.connect(tmp_db)
        for t in tickers:
            dp._ensure_table(conn, t)
        conn.close()
        available = dp.get_available_tickers()
        for t in tickers:
            assert t in available
