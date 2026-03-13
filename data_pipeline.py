"""
data_pipeline.py
================
Responsible for:
  - Downloading intraday OHLCV data via yfinance (free, no API key)
  - Persisting raw bars to a local SQLite database
  - Incremental updates (only fetches missing bars)
  - Basic cleaning: NaN removal, timezone normalisation
  - Providing a clean DataFrame ready for feature engineering
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import DB_PATH, INTERVAL, INTRADAY_PERIOD_MAP, PERIOD, TICKERS, TIMEZONE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode for concurrency safety."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_table(conn: sqlite3.Connection, ticker: str, interval: str = "1d") -> None:
    """Create per-ticker OHLCV table if not already present."""
    table = _table_name(ticker, interval)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            datetime    TEXT PRIMARY KEY,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL
        )
    """)
    conn.commit()


def _table_name(ticker: str, interval: str = "1d") -> str:
    safe_ticker   = ticker.upper().replace("-", "_").replace(".", "_")
    safe_interval = interval.replace("m", "m").replace("h", "h")
    # For daily data keep the original name for backwards compatibility
    if interval == "1d":
        return safe_ticker
    return f"{safe_ticker}_{safe_interval}"


def _last_stored_dt(
    conn: sqlite3.Connection, ticker: str, interval: str = "1d"
) -> Optional[pd.Timestamp]:
    """Return the most recent datetime stored for *ticker* / *interval*, or None."""
    table = _table_name(ticker, interval)
    try:
        row = conn.execute(
            f'SELECT MAX(datetime) FROM "{table}"'
        ).fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0], tz=TIMEZONE)
    except sqlite3.OperationalError:
        pass
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download intraday bars from Yahoo Finance.

    Returns a clean DataFrame with columns:
        open, high, low, close, volume
    indexed by tz-aware timestamps in TIMEZONE.
    """
    logger.info("Downloading %s  period=%s  interval=%s", ticker, period, interval)
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    # yfinance may return MultiIndex columns when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "datetime"

    # Normalise timezone
    if raw.index.tz is None:
        raw = raw.tz_localize("UTC")
    raw = raw.tz_convert(TIMEZONE)

    # Drop rows with any NaN in OHLCV
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    # Remove zero-volume bars (market closed / extended hours artefacts)
    raw = raw[raw["volume"] > 0]

    return raw


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _upsert_bars(
    conn: sqlite3.Connection, ticker: str, df: pd.DataFrame, interval: str = "1d"
) -> int:
    """
    Insert new bars into SQLite, ignoring duplicates (UPSERT on PK).
    Returns the number of new rows inserted.
    """
    if df.empty:
        return 0

    table = _table_name(ticker, interval)
    _ensure_table(conn, ticker, interval)

    records = [
        (str(ts), row.open, row.high, row.low, row.close, row.volume)
        for ts, row in df.iterrows()
    ]

    cursor = conn.executemany(
        f"""
        INSERT OR IGNORE INTO "{table}"
            (datetime, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_data(
    tickers: list[str] = TICKERS,
    interval: str = INTERVAL,
    period: str = PERIOD,
) -> None:
    """
    Download and persist OHLCV bars for all *tickers*.

    Supports both daily ("1d") and intraday intervals ("1h", "5m", "1m").
    For intraday intervals the maximum free yfinance period is used unless
    *period* is explicitly provided.

    Performs incremental updates: only bars newer than the last stored
    timestamp are written to the database.
    """
    # Auto-select maximum free period for intraday when caller uses default
    effective_period = period
    if period == "max" and interval != "1d":
        effective_period = INTRADAY_PERIOD_MAP.get(interval, "60d")
        logger.info(
            "Intraday interval=%s — using period=%s (max free via yfinance)",
            interval, effective_period,
        )

    conn = _get_connection()
    try:
        for ticker in tickers:
            _ensure_table(conn, ticker, interval)
            last_dt = _last_stored_dt(conn, ticker, interval)

            df = _download_ticker(ticker, period=effective_period, interval=interval)
            if df.empty:
                continue

            # Incremental filter
            if last_dt is not None:
                df = df[df.index > last_dt]

            n = _upsert_bars(conn, ticker, df, interval)
            logger.info(
                "%s [%s] -> %d new bars stored (last=%s)",
                ticker, interval, n, df.index[-1] if not df.empty else "N/A",
            )
    finally:
        conn.close()


def load_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load OHLCV bars from SQLite for a given *ticker* and *interval*.

    Parameters
    ----------
    ticker   : str
    start    : ISO date string, e.g. "2024-01-01"  (optional)
    end      : ISO date string  (optional)
    interval : bar interval matching the table stored by update_data()
               e.g. "1d" (default), "1h", "5m"

    Returns
    -------
    pd.DataFrame  with tz-aware DatetimeIndex and columns
                  [open, high, low, close, volume]
    """
    conn = _get_connection()
    table = _table_name(ticker, interval)

    try:
        _ensure_table(conn, ticker, interval)
        query = f'SELECT * FROM "{table}" ORDER BY datetime'
        # Do NOT use parse_dates — data spans DST transitions,
        # producing mixed "+/-04:00"/"+/-05:00" strings that pandas rejects.
        # Instead read raw strings and convert via UTC to safely handle DST.
        df = pd.read_sql(query, conn, index_col="datetime")
    finally:
        conn.close()

    if df.empty:
        logger.warning("No data in DB for %s — run update_data() first.", ticker)
        return df

    # Convert mixed-tz strings -> UTC -> target timezone (handles DST correctly)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TIMEZONE)

    # Date filters
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=TIMEZONE)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=TIMEZONE)]

    return df.sort_index()


def get_available_tickers() -> list[str]:
    """Return the list of tickers that have data stored in the DB."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def load_vix_data(period: str = "max") -> pd.DataFrame:
    """
    Download daily VIX (^VIX) data from yfinance.

    Returns a daily DataFrame with columns [open, high, low, close, volume]
    indexed by tz-naive date timestamps.  Returns empty DataFrame on failure.

    VIX is used as a market fear gauge feature: high VIX = fearful market.
    We use the PREVIOUS day's VIX for each hourly bar (causal, no look-ahead).
    """
    try:
        raw = yf.download("^VIX", period=period, interval="1d",
                           auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("VIX download returned empty DataFrame.")
            return pd.DataFrame()

        raw.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                       for c in raw.columns]
        # Drop MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]

        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        # Make index tz-naive (VIX is daily — no intraday tz needed)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        logger.info("VIX loaded: %d daily bars  [%s -> %s]",
                    len(raw), raw.index[0].date(), raw.index[-1].date())
        return raw
    except Exception as exc:
        logger.warning("VIX download failed: %s — VIX features will use defaults.", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    update_data()
    for t in TICKERS:
        df = load_data(t)
        logger.info("Loaded %s: %d bars  [%s -> %s]", t, len(df), df.index[0], df.index[-1])
