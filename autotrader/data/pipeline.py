"""
pipeline.py
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

from autotrader.config.settings import DATA_DIR, DB_PATH, INTERVAL, INTRADAY_PERIOD_MAP, PERIOD, TICKERS, TIMEZONE

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
# B3 COTAHIST — dados oficiais da bolsa brasileira
# ---------------------------------------------------------------------------
# O arquivo COTAHIST_AXXXX.ZIP contém todos os negócios do ano em formato
# de largura fixa, publicado diariamente pela B3 gratuitamente.
# Documentação: https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/
#
# Uso:
#   from autotrader.data.pipeline import load_b3_cotahist, update_data_b3
#   df = load_b3_cotahist(2024, ["PETR4.SA", "BBAS3.SA"])
#   update_data_b3([2024, 2025], tickers=TICKERS)
# ---------------------------------------------------------------------------

_B3_COTAHIST_URL = (
    "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
)

# Posições exatas no arquivo COTAHIST (baseadas na especificação oficial B3)
_COTAHIST_COLSPECS = [
    (0,   2),    # tipo_registro
    (2,   10),   # data_pregao   YYYYMMDD
    (12,  24),   # codisi        ticker
    (24,  27),   # tipo_mercado  010=à vista
    (56,  69),   # preco_abertura  (×100)
    (69,  82),   # preco_maximo    (×100)
    (82,  95),   # preco_minimo    (×100)
    (108, 121),  # preco_fechamento (×100)
    (170, 188),  # volume_negociado (×100)
]

_COTAHIST_NAMES = [
    "tipo", "data", "ticker", "mercado",
    "open", "high", "low", "close", "volume"
]


def _parse_cotahist_price(s: pd.Series) -> pd.Series:
    """Converte campo de preço COTAHIST (string com 2 decimais implícitos) para float."""
    return pd.to_numeric(s.str.strip(), errors="coerce") / 100.0


def load_b3_cotahist(
    year: int,
    tickers: list[str] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Baixa e parseia o arquivo COTAHIST anual da B3.

    Dados oficiais — qualidade superior ao Yahoo Finance:
      - Ajuste correto de splits e proventos (campo preco_oferta)
      - Sem gaps ou erros de timestamp
      - Cobre todas as ações listadas na B3

    Parâmetros
    ----------
    year      : Ano (ex: 2024)
    tickers   : Lista de tickers .SA (ex: ["PETR4.SA", "BBAS3.SA"]).
                None = retorna todos os tickers.
    cache_dir : Diretório para cache do ZIP. None = usa DATA_DIR.

    Retorna
    -------
    pd.DataFrame com colunas [open, high, low, close, volume]
    e DatetimeIndex tz-aware (America/Sao_Paulo), por ticker filtrado.
    """
    import io
    import zipfile

    import requests

    if cache_dir is None:
        cache_dir = DATA_DIR

    cache_file = cache_dir / f"COTAHIST_A{year}.ZIP"

    # Download com cache local
    if not cache_file.exists():
        url = _B3_COTAHIST_URL.format(year=year)
        logger.info("B3 COTAHIST: baixando %s ...", url)
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            cache_file.write_bytes(resp.content)
            logger.info("B3 COTAHIST: salvo em %s (%d KB)", cache_file, len(resp.content) // 1024)
        except Exception as exc:
            logger.error("B3 COTAHIST download falhou para %d: %s", year, exc)
            return pd.DataFrame()
    else:
        logger.info("B3 COTAHIST: usando cache %s", cache_file)

    # Parse do arquivo de largura fixa
    try:
        with zipfile.ZipFile(cache_file) as zf:
            txt_name = [n for n in zf.namelist() if n.upper().startswith("COTAHIST")][0]
            raw_bytes = zf.read(txt_name)
        raw_text = raw_bytes.decode("latin-1")
    except Exception as exc:
        logger.error("B3 COTAHIST parse falhou: %s", exc)
        return pd.DataFrame()

    df = pd.read_fwf(
        io.StringIO(raw_text),
        colspecs=_COTAHIST_COLSPECS,
        names=_COTAHIST_NAMES,
        dtype=str,
        header=None,
    )

    # Filtra apenas registros de cotação à vista (tipo=01, mercado=010)
    df = df[df["tipo"] == "01"]
    df = df[df["mercado"].str.strip() == "010"]

    # Converte ticker: COTAHIST usa 12 chars padded com espaços, sem sufixo .SA
    df["ticker"] = df["ticker"].str.strip()

    # Filtra tickers se fornecidos
    if tickers:
        ticker_map = {t.replace(".SA", "").strip(): t for t in tickers}
        df = df[df["ticker"].isin(ticker_map.keys())]
        df["ticker"] = df["ticker"].map(ticker_map)

    if df.empty:
        logger.warning("B3 COTAHIST %d: nenhum dado encontrado para os tickers solicitados.", year)
        return pd.DataFrame()

    # Converte preços e datas
    df["date"] = pd.to_datetime(df["data"].str.strip(), format="%Y%m%d", errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = _parse_cotahist_price(df[col])
    df["volume"] = pd.to_numeric(df["volume"].str.strip(), errors="coerce") / 100.0

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df["close"] > 0]

    # Formata como DataFrame OHLCV com DatetimeIndex tz-aware (padrão do pipeline)
    result_frames = []
    for ticker_sa, grp in df.groupby("ticker"):
        grp = grp.set_index("date").sort_index()
        grp.index = pd.DatetimeIndex(grp.index).tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        ohlcv = grp[["open", "high", "low", "close", "volume"]].copy()
        ohlcv["_ticker"] = ticker_sa
        result_frames.append(ohlcv)

    if not result_frames:
        return pd.DataFrame()

    out = pd.concat(result_frames).sort_index()
    logger.info(
        "B3 COTAHIST %d: %d barras carregadas para %d ticker(s)",
        year, len(out), out["_ticker"].nunique(),
    )
    return out


def update_data_b3(
    years: list[int] | None = None,
    tickers: list[str] = TICKERS,
) -> None:
    """
    Atualiza o banco de dados local com dados B3 oficiais (COTAHIST).

    Complementa o Yahoo Finance com dados de alta qualidade:
      - Usa como fonte primária para dados históricos (anos completos)
      - yfinance continua sendo usado para dados do dia corrente

    Parâmetros
    ----------
    years   : Anos a importar (ex: [2022, 2023, 2024]). None = ano atual.
    tickers : Lista de tickers com sufixo .SA.
    """
    import datetime

    if years is None:
        years = [datetime.date.today().year]

    conn = _get_connection()
    try:
        for year in years:
            logger.info("B3 COTAHIST: importando ano %d para %d tickers...", year, len(tickers))
            cotahist_df = load_b3_cotahist(year, tickers=tickers)

            if cotahist_df.empty:
                continue

            for ticker in tickers:
                ticker_df = cotahist_df[cotahist_df["_ticker"] == ticker].drop(columns=["_ticker"])
                if ticker_df.empty:
                    continue
                n = _upsert_bars(conn, ticker, ticker_df, interval="1d")
                logger.info("B3 COTAHIST %d [%s]: %d novas barras inseridas", year, ticker, n)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    update_data()
    for t in TICKERS:
        df = load_data(t)
        logger.info("Loaded %s: %d bars  [%s -> %s]", t, len(df), df.index[0], df.index[-1])
