"""
tests/test_journal.py
=====================
Tests for journal.py trade recording and weekly close logic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestJournalDataSource:
    """journal_weekly_close must read from the same DB as the scan (load_data)."""

    def test_uses_load_data_not_yf_download(self):
        """Regression: journal_weekly_close must use load_data (DB), not yf.download."""
        import inspect
        from autotrader.utils import journal as jmod
        source = inspect.getsource(jmod.journal_weekly_close)
        assert "load_data" in source, (
            "journal_weekly_close must use load_data() to read from the DB — "
            "same source as the scan, ensuring consistent price base"
        )
        assert "yf.download" not in source, (
            "journal_weekly_close must NOT call yf.download directly — "
            "use load_data() from the DB instead"
        )
