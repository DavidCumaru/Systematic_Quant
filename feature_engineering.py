"""
feature_engineering.py
=======================
Quantitative features for the Systematic Alpha pipeline.

All calculations are strictly causal (zero look-ahead bias):
  - Every indicator uses only information available at bar-close time t.
  - Shift(1) patterns are applied explicitly where needed.

Feature groups
--------------
  1.  Log returns (multi-lag)
  2.  RSI
  3.  MACD line + histogram
  4.  ATR (normalised)
  5.  VWAP deviation (rolling proxy)
  6.  Rolling volatility
  7.  Volume spike flag (z-score)
  8.  Order-imbalance proxy (HL range position)
  9.  Price breakout flags (20-bar)
  10. Z-score of close price
  11. Multi-window momentum
  12. Sazonalidade semanal/mensal (dia-da-semana + mes do ano sin/cos)
  13. High-low spread normalised
  14. Overnight gap (open vs prior close)
  15. Relative volume (volume / rolling mean)
  16. Regime volatility flag
  17. Garman-Klass volatility (more efficient estimator)
  18. Amihud illiquidity (price impact per dollar traded)
  19. VIX level (market fear gauge — from external daily data)
  20. Rolling beta vs SPY (systematic risk exposure)
  21. Earnings proximity (days to next earnings announcement)
  22. Distancia da MA200 (posicao relativa na tendencia)
  23. Proximidade ao maximo de 52 semanas (momentum classico)
  24. Momentum 12-1 mes (cross-sectional momentum padrao)
"""

import logging

import numpy as np
import pandas as pd

from config import (
    ATR_WINDOW,
    BETA_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MOMENTUM_WINDOWS,
    REGIME_VOL_THRESHOLD,
    REGIME_VOL_WINDOW,
    REL_VOL_WINDOW,
    RSI_WINDOW,
    VOL_WINDOW,
    VOLUME_SPIKE_Z,
    VWAP_WINDOW,
    ZSCORE_WINDOW,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers — existing indicators
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _vwap_deviation(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    cum_tpv = (close * volume).rolling(window, min_periods=1).sum()
    cum_vol = volume.rolling(window, min_periods=1).sum()
    vwap    = cum_tpv / cum_vol.replace(0, np.nan)
    return (close - vwap) / vwap.replace(0, np.nan)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / std.replace(0, np.nan)


def _volume_spike(volume: pd.Series, window: int, z_threshold: float) -> pd.Series:
    return (_rolling_zscore(volume, window) > z_threshold).astype(int)


def _order_imbalance_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    bar_range = (high - low).replace(0, np.nan)
    return (close - low) / bar_range


def _breakout_flag(close: pd.Series, window: int, direction: str) -> pd.Series:
    if direction == "up":
        level = close.shift(1).rolling(window, min_periods=window).max()
        return (close > level).astype(int)
    level = close.shift(1).rolling(window, min_periods=window).min()
    return (close < level).astype(int)


def _overnight_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """(open_t - close_{t-1}) / close_{t-1}, non-zero only on first bar of each day."""
    prev_close = close.shift(1)
    raw_gap    = (open_ - prev_close) / prev_close.replace(0, np.nan)
    date_arr   = pd.Series(open_.index.date, index=open_.index)
    day_change = date_arr != date_arr.shift(1)
    return raw_gap.where(day_change, other=0.0)


def _relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """volume / rolling_mean_volume — normalised liquidity context."""
    mean_vol = volume.rolling(window, min_periods=window // 2).mean()
    return volume / mean_vol.replace(0, np.nan)


def _regime_vol_flag(log_ret: pd.Series, window: int, threshold: float) -> pd.Series:
    """Binary flag: 1 when rolling volatility > threshold."""
    rolling_vol = log_ret.rolling(window, min_periods=window // 2).std()
    return (rolling_vol > threshold).astype(int)


# ---------------------------------------------------------------------------
# New indicators — Phase 2
# ---------------------------------------------------------------------------

def _garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    Uses OHLC data for a more efficient volatility estimate than
    simple close-to-close returns (~7x more efficient).

    GK = sqrt( 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2 )
    """
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    gk_sq  = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return gk_sq.rolling(window, min_periods=window // 2).mean().apply(
        lambda x: np.sqrt(max(x, 0))
    )


def _amihud_illiquidity(log_ret: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Amihud (2002) illiquidity measure: |return| / dollar_volume.

    High values = large price impact per dollar traded (illiquid).
    Normalised by rolling mean to make it comparable across time.
    """
    dollar_vol   = close * volume
    raw_amihud   = log_ret.abs() / dollar_vol.replace(0, np.nan)
    rolling_mean = raw_amihud.rolling(window, min_periods=window // 2).mean()
    rolling_std  = raw_amihud.rolling(window, min_periods=window // 2).std()
    return (raw_amihud - rolling_mean) / rolling_std.replace(0, np.nan)


def _vix_feature(df_index: pd.DatetimeIndex, vix_df: pd.DataFrame | None) -> pd.Series:
    """
    Map daily VIX close (prev day) onto hourly bars.

    Uses previous day's VIX to avoid any look-ahead — VIX is only
    published at 4pm ET, so intraday bars must use the prior day's value.
    Default = 20 (long-run average) when VIX data unavailable.
    """
    if vix_df is None or vix_df.empty:
        return pd.Series(20.0, index=df_index, name="vix")

    vix_col = "close" if "close" in vix_df.columns else vix_df.columns[0]
    vix_prev = vix_df[vix_col].shift(1)   # causal: use yesterday's VIX

    # Build date -> vix mapping
    vix_map = {ts.date(): val for ts, val in vix_prev.items() if not pd.isna(val)}

    values = []
    last_known = 20.0
    for ts in df_index:
        d = ts.date()
        if d in vix_map:
            last_known = vix_map[d]
        values.append(last_known)

    return pd.Series(values, index=df_index, name="vix")


def _rolling_beta(
    log_ret: pd.Series, spy_df: pd.DataFrame | None, window: int
) -> pd.Series:
    """
    Rolling beta of the stock vs SPY.

    beta = Cov(stock, SPY) / Var(SPY)

    Beta > 1 = more volatile than market, < 1 = more stable.
    Default = 1.0 (market beta) when SPY data unavailable.
    """
    if spy_df is None or spy_df.empty:
        return pd.Series(1.0, index=log_ret.index, name="beta_spy")

    spy_ret = np.log(spy_df["close"] / spy_df["close"].shift(1))
    spy_ret = spy_ret.reindex(log_ret.index, method="nearest", tolerance="1D")

    covariance = (
        log_ret.rolling(window, min_periods=window // 2)
        .cov(spy_ret)
    )
    variance = spy_ret.rolling(window, min_periods=window // 2).var()
    beta = covariance / variance.replace(0, np.nan)
    return beta.fillna(1.0).rename("beta_spy")


def _earnings_proximity(close_index: pd.DatetimeIndex, ticker: str) -> pd.Series:
    """
    Days until next earnings announcement, clipped to [0, 30].

    Uses yfinance earnings dates (free). Defaults to 30 (no upcoming earnings)
    when data is unavailable. Lower values = higher event risk.
    """
    result = pd.Series(30.0, index=close_index, name="days_to_earnings")

    if not ticker:
        return result

    try:
        import yfinance as yf
        dates_df = yf.Ticker(ticker).earnings_dates
        if dates_df is None or dates_df.empty:
            return result

        # Convert to date objects (tz-naive)
        earnings_dates = sorted(
            set(pd.DatetimeIndex(dates_df.index).tz_localize(None).normalize().date)
        )

        bar_dates = [ts.date() for ts in close_index]
        for i, d in enumerate(bar_dates):
            # Find nearest future earnings date
            future = [e for e in earnings_dates if e >= d]
            if future:
                result.iloc[i] = min((future[0] - d).days, 30)

    except Exception:
        pass  # silently fall back to default 30

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    """
    Compute all features and append to a copy of *df*.

    Parameters
    ----------
    df      : OHLCV DataFrame with tz-aware DatetimeIndex
    spy_df  : SPY OHLCV DataFrame (same tz). Used for rolling beta.
              Pass None to skip — beta feature defaults to 1.0.
    vix_df  : VIX daily DataFrame. Used for market fear context.
              Pass None to skip — VIX feature defaults to 20.0.
    ticker  : ticker symbol string (used for earnings calendar).
              Pass "" to skip earnings feature.

    Returns DataFrame with original OHLCV + all feature columns,
    NaN warm-up rows dropped.
    """
    df = df.copy()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    log_ret = np.log(close / close.shift(1))

    # 1. Log returns (lags 1-5)
    df["log_ret"] = log_ret
    for lag in range(1, 6):
        df[f"log_ret_lag{lag}"] = log_ret.shift(lag - 1)

    # 2. RSI
    df["rsi"] = _rsi(close, RSI_WINDOW)

    # 3. MACD
    macd_line, signal_line, histogram = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = histogram

    # 4. ATR (normalised by close)
    df["atr"]     = _atr(high, low, close, ATR_WINDOW)
    df["atr_pct"] = df["atr"] / close

    # 5. VWAP deviation
    df["vwap_dev"] = _vwap_deviation(close, volume, VWAP_WINDOW)

    # 6. Rolling volatility
    df["rolling_vol"] = log_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()

    # 7. Volume spike flag
    df["vol_spike"] = _volume_spike(volume, VOL_WINDOW, VOLUME_SPIKE_Z)

    # 8. Order imbalance proxy
    df["order_imbalance"] = _order_imbalance_proxy(high, low, close)

    # 9. Breakout flags (20-bar lookback)
    df["breakout_up"]   = _breakout_flag(close, 20, "up")
    df["breakout_down"] = _breakout_flag(close, 20, "down")

    # 10. Z-score of close price
    df["close_zscore"] = _rolling_zscore(close, ZSCORE_WINDOW)

    # 11. Multi-window momentum
    for w in MOMENTUM_WINDOWS:
        df[f"momentum_{w}"] = close / close.shift(w) - 1

    # 12. Sazonalidade: dia da semana + mes do ano (sin/cos encoding)
    dow = df.index.dayofweek.to_numpy().astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    month = df.index.month.to_numpy().astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 13. High-low spread
    df["hl_spread"] = (high - low) / close

    # 14. Overnight gap
    df["overnight_gap"] = _overnight_gap(open_, close)

    # 15. Relative volume
    df["rel_volume"] = _relative_volume(volume, REL_VOL_WINDOW)

    # 16. Regime volatility flag
    df["regime_vol_flag"] = _regime_vol_flag(log_ret, REGIME_VOL_WINDOW, REGIME_VOL_THRESHOLD)

    # 17. Garman-Klass volatility (more efficient than close-to-close)
    df["gk_vol"] = _garman_klass_vol(open_, high, low, close, VOL_WINDOW)

    # 18. Amihud illiquidity (z-scored)
    df["amihud"] = _amihud_illiquidity(log_ret, close, volume, VOL_WINDOW)

    # 19. VIX level (external — market fear gauge)
    df["vix"] = _vix_feature(df.index, vix_df).values

    # 20. Rolling beta vs SPY
    df["beta_spy"] = _rolling_beta(log_ret, spy_df, BETA_WINDOW).values

    # 21. Earnings proximity (days to next earnings, clipped to 30)
    df["days_to_earnings"] = _earnings_proximity(df.index, ticker).values

    # 22. Distancia da MA200 — posicao relativa na tendencia de longo prazo
    ma200 = close.rolling(200, min_periods=100).mean()
    df["ma200_dist"] = (close / ma200.replace(0, np.nan)) - 1.0

    # 23. Proximidade ao maximo de 52 semanas — forca de momentum classico
    hi52 = close.rolling(252, min_periods=126).max()
    df["hi52w_prox"] = (close / hi52.replace(0, np.nan)).clip(0.5, 1.0)

    # 24. Momentum 12-1 mes — retorno dos ultimos 12 meses excluindo o ultimo mes
    df["mom_12_1"] = close.shift(21) / close.shift(252).replace(0, np.nan) - 1.0

    # Drop warm-up NaN rows
    feature_cols = get_feature_names(df)
    df = df.dropna(subset=feature_cols)

    logger.info("Features built: %d rows x %d columns", len(df), len(df.columns))
    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes OHLCV and label)."""
    exclude = {"open", "high", "low", "close", "volume", "label"}
    return [c for c in df.columns if c not in exclude]
