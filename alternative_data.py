"""
alternative_data.py
===================
Free alternative data sources for the Systematic Alpha pipeline.

All data is obtained from free, public APIs — no paid subscriptions required.

Sources
-------
  1. FRED (Federal Reserve Economic Data)
     - 10Y-2Y Treasury yield spread (recession indicator)
     - Fed Funds Rate (monetary policy stance)
     - CPI YoY (inflation regime)
     Base URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES

  2. Fear & Greed Proxy (derived from market data)
     Composite score approximating CNN Fear & Greed index using:
       - VIX level
       - SPY 125-day momentum
       - High-yield spread proxy (via yfinance: HYG/IEF ratio)
     Normalised to [0, 100]: 0 = extreme fear, 100 = extreme greed

  3. Yield Curve
     - 10Y-2Y spread in basis points
     - Inversion flag (spread < 0 = inverted = recession warning)

  4. Macro Regime
     - Simple 3-state macro regime:
         0 = Contractionary (inverted yield curve + high inflation)
         1 = Neutral
         2 = Expansionary (normal yield curve + moderate inflation)

All series are returned as daily DataFrames with a tz-naive DatetimeIndex.
They are intended to be merged with the main feature DataFrame as additional
signal-agnostic context features.

Usage
-----
    from alternative_data import AlternativeDataLoader

    loader = AlternativeDataLoader()

    yield_curve = loader.yield_curve()          # daily DataFrame
    fear_greed  = loader.fear_greed_proxy()     # daily Series [0, 100]
    macro       = loader.macro_regime()         # daily Series {0, 1, 2}

    # Merge with feature DataFrame (daily bars)
    df = df.join(yield_curve, how="left").ffill()

FRED data requires no API key for public series (CSV endpoint).
If FRED is unreachable, all methods gracefully return default/fallback values.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series IDs
# ---------------------------------------------------------------------------
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    "t10y2y":      "T10Y2Y",       # 10Y-2Y Treasury spread (bps)
    "fedfunds":    "FEDFUNDS",     # Effective Federal Funds Rate
    "cpi_yoy":     "CPIAUCSL",     # CPI All Urban Consumers (level — compute YoY internally)
    "t10y":        "DGS10",        # 10-Year Treasury yield
    "t2y":         "DGS2",         # 2-Year Treasury yield
}

_REQUEST_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# FRED download helper
# ---------------------------------------------------------------------------

def _fetch_fred(series_id: str) -> pd.Series:
    """
    Download a FRED series as a daily pandas Series.

    Returns an empty Series on failure (network error, rate limit, etc.).
    No API key required — uses the public CSV endpoint.
    """
    url = f"{FRED_BASE_URL}?id={series_id}"
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        series = df.iloc[:, 0].replace(".", np.nan).astype(float)
        series.name = series_id
        logger.info("FRED %s: %d observations [%s -> %s]",
                    series_id, series.count(), series.first_valid_index(), series.last_valid_index())
        return series
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float, name=series_id)


# ---------------------------------------------------------------------------
# Alternative Data Loader
# ---------------------------------------------------------------------------

class AlternativeDataLoader:
    """
    Loads and caches free alternative data for systematic trading features.

    All methods return pandas DataFrames or Series indexed by date (tz-naive).
    Missing data is filled with sensible economic defaults.

    Parameters
    ----------
    cache : bool — if True, FRED series are cached in-memory after first fetch.
                   Avoids repeated HTTP calls during a single pipeline run.
    """

    def __init__(self, cache: bool = True):
        self._cache: dict[str, pd.Series] = {}
        self._use_cache = cache

    # ------------------------------------------------------------------
    def _get_fred(self, series_id: str) -> pd.Series:
        if self._use_cache and series_id in self._cache:
            return self._cache[series_id]
        data = _fetch_fred(series_id)
        if self._use_cache:
            self._cache[series_id] = data
        return data

    # ------------------------------------------------------------------
    def yield_curve(self) -> pd.DataFrame:
        """
        Return a DataFrame with Treasury yield data and derived features.

        Columns
        -------
        t10y        : 10-Year Treasury yield (%)
        t2y         : 2-Year Treasury yield (%)
        spread_10_2 : 10Y minus 2Y spread in percentage points
        inverted    : 1 if spread < 0 (yield curve inversion), 0 otherwise

        Fallback defaults (when FRED is unavailable):
          t10y=4.0, t2y=3.5, spread=0.5, inverted=0
        """
        t10y_s = self._get_fred(FRED_SERIES["t10y"])
        t2y_s  = self._get_fred(FRED_SERIES["t2y"])

        # Use pre-computed T10Y2Y if individual series failed
        spread_direct = self._get_fred(FRED_SERIES["t10y2y"])

        if t10y_s.empty and t2y_s.empty and spread_direct.empty:
            logger.warning("Yield curve: all FRED fetches failed — using defaults")
            idx = pd.date_range("2000-01-03", periods=1, freq="B")
            return pd.DataFrame(
                {"t10y": 4.0, "t2y": 3.5, "spread_10_2": 0.5, "inverted": 0},
                index=idx,
            )

        # Combine into daily index
        all_idx = pd.date_range(
            start=min(
                s.first_valid_index() for s in [t10y_s, t2y_s, spread_direct] if not s.empty
            ),
            end=max(
                s.last_valid_index() for s in [t10y_s, t2y_s, spread_direct] if not s.empty
            ),
            freq="B",
        )

        df = pd.DataFrame(index=all_idx)
        if not t10y_s.empty:
            df["t10y"] = t10y_s.reindex(all_idx).ffill().fillna(4.0)
        else:
            df["t10y"] = 4.0

        if not t2y_s.empty:
            df["t2y"] = t2y_s.reindex(all_idx).ffill().fillna(3.5)
        else:
            df["t2y"] = 3.5

        if not spread_direct.empty:
            df["spread_10_2"] = spread_direct.reindex(all_idx).ffill().fillna(df["t10y"] - df["t2y"])
        else:
            df["spread_10_2"] = df["t10y"] - df["t2y"]

        df["inverted"] = (df["spread_10_2"] < 0).astype(int)

        return df.round(4)

    # ------------------------------------------------------------------
    def fed_funds_rate(self) -> pd.Series:
        """
        Federal Funds Rate — monetary policy stance.

        Returns a daily Series (forward-filled from monthly releases).
        Default: 4.5% when FRED is unavailable.
        """
        ffr = self._get_fred(FRED_SERIES["fedfunds"])
        if ffr.empty:
            logger.warning("Fed Funds Rate: FRED unavailable — using default 4.5%%")
            return pd.Series(dtype=float, name="fedfunds")

        daily_idx = pd.date_range(ffr.first_valid_index(), ffr.last_valid_index(), freq="B")
        return ffr.reindex(daily_idx).ffill().fillna(4.5).rename("fedfunds")

    # ------------------------------------------------------------------
    def cpi_yoy(self) -> pd.Series:
        """
        CPI Year-over-Year inflation rate (%).

        Computed from monthly CPI levels as:
            cpi_yoy = (CPI_t / CPI_{t-12} - 1) * 100

        Returns a daily Series (forward-filled).
        Default: 3.0% when FRED is unavailable.
        """
        cpi = self._get_fred(FRED_SERIES["cpi_yoy"])
        if cpi.empty:
            logger.warning("CPI: FRED unavailable — using default 3.0%%")
            return pd.Series(dtype=float, name="cpi_yoy")

        yoy = (cpi / cpi.shift(12) - 1) * 100
        daily_idx = pd.date_range(yoy.first_valid_index(), yoy.last_valid_index(), freq="B")
        return yoy.reindex(daily_idx).ffill().fillna(3.0).rename("cpi_yoy")

    # ------------------------------------------------------------------
    def fear_greed_proxy(
        self,
        vix_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
        hyg_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Approximate CNN Fear & Greed index [0=extreme fear, 100=extreme greed].

        Composite of three sub-indicators normalised to [0, 100]:
          1. VIX momentum: low VIX relative to 30-day avg = greed
          2. SPY 125-day price momentum (trend strength)
          3. HYG/IEF ratio momentum (credit appetite vs safety)

        Each component is scaled to [0, 100] via rolling percentile rank
        (250-day window) to produce a regime-normalised score.

        Parameters
        ----------
        vix_df : daily VIX DataFrame (optional — downloaded if not provided)
        spy_df : daily SPY OHLCV DataFrame (optional)
        hyg_df : daily HYG OHLCV DataFrame (optional — credit spread proxy)

        Returns
        -------
        pd.Series named 'fear_greed' with daily values in [0, 100]
        """
        try:
            import yfinance as yf

            if vix_df is None or vix_df.empty:
                raw_vix = yf.download("^VIX", period="5y", interval="1d",
                                      auto_adjust=True, progress=False)
                if isinstance(raw_vix.columns, pd.MultiIndex):
                    raw_vix.columns = [c[0].lower() for c in raw_vix.columns]
                else:
                    raw_vix.columns = [c.lower() for c in raw_vix.columns]
                if raw_vix.index.tz is not None:
                    raw_vix.index = raw_vix.index.tz_localize(None)
                vix_close = raw_vix["close"]
            else:
                vix_close = vix_df["close"].copy()
                if vix_close.index.tz is not None:
                    vix_close.index = vix_close.index.tz_localize(None)

            if spy_df is None or spy_df.empty:
                raw_spy = yf.download("SPY", period="5y", interval="1d",
                                      auto_adjust=True, progress=False)
                if isinstance(raw_spy.columns, pd.MultiIndex):
                    raw_spy.columns = [c[0].lower() for c in raw_spy.columns]
                else:
                    raw_spy.columns = [c.lower() for c in raw_spy.columns]
                if raw_spy.index.tz is not None:
                    raw_spy.index = raw_spy.index.tz_localize(None)
                spy_close = raw_spy["close"]
            else:
                spy_close = spy_df["close"].copy()
                if spy_close.index.tz is not None:
                    spy_close.index = spy_close.index.tz_localize(None)

            def _percentile_rank(s: pd.Series, window: int = 250) -> pd.Series:
                """Roll-rank each value within its trailing window: [0, 100]."""
                return s.rolling(window, min_periods=50).apply(
                    lambda x: float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1) * 100,
                    raw=True,
                )

            # Sub-indicator 1: VIX momentum — low VIX = greed (invert the rank)
            vix_ma30   = vix_close.rolling(30, min_periods=15).mean()
            vix_signal = vix_close / vix_ma30.replace(0, np.nan)
            vix_fg     = 100 - _percentile_rank(vix_signal)   # inverted: high VIX = fear

            # Sub-indicator 2: SPY 125-day momentum — positive momentum = greed
            spy_mom  = spy_close / spy_close.shift(125).replace(0, np.nan) - 1
            spy_fg   = _percentile_rank(spy_mom)

            # Sub-indicator 3: HYG/IEF credit spread proxy
            try:
                if hyg_df is None or hyg_df.empty:
                    raw_hyg = yf.download("HYG", period="5y", interval="1d",
                                          auto_adjust=True, progress=False)
                    raw_ief = yf.download("IEF", period="5y", interval="1d",
                                          auto_adjust=True, progress=False)
                    for raw in [raw_hyg, raw_ief]:
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = [c[0].lower() for c in raw.columns]
                        else:
                            raw.columns = [c.lower() for c in raw.columns]
                        if raw.index.tz is not None:
                            raw.index = raw.index.tz_localize(None)
                    hyg_close = raw_hyg["close"]
                    ief_close = raw_ief["close"]
                else:
                    hyg_close = hyg_df["close"].copy()
                    if hyg_close.index.tz is not None:
                        hyg_close.index = hyg_close.index.tz_localize(None)
                    ief_close = spy_close  # fallback if HYG not separate

                credit_ratio = hyg_close / ief_close.reindex(hyg_close.index).ffill().replace(0, np.nan)
                credit_mom   = credit_ratio / credit_ratio.shift(20).replace(0, np.nan) - 1
                credit_fg    = _percentile_rank(credit_mom)
            except Exception:
                credit_fg = pd.Series(50.0, index=spy_close.index)

            # Combine on common index (equal weights)
            common_idx = vix_fg.index.intersection(spy_fg.index).intersection(credit_fg.index)
            composite  = (
                vix_fg.reindex(common_idx) * 0.40
                + spy_fg.reindex(common_idx) * 0.35
                + credit_fg.reindex(common_idx) * 0.25
            ).clip(0, 100)

            return composite.rename("fear_greed")

        except Exception as exc:
            logger.warning("fear_greed_proxy failed: %s — returning neutral 50", exc)
            return pd.Series(dtype=float, name="fear_greed")

    # ------------------------------------------------------------------
    def macro_regime(
        self,
        yield_curve_df: Optional[pd.DataFrame] = None,
        cpi_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Simple 3-state macro regime indicator.

        States
        ------
          0 = Contractionary  (inverted yield curve OR high inflation > 5%)
          1 = Neutral         (baseline)
          2 = Expansionary    (normal yield curve AND inflation 1-3%)

        Parameters
        ----------
        yield_curve_df : output of yield_curve() (optional — fetched if None)
        cpi_series     : output of cpi_yoy() (optional — fetched if None)

        Returns
        -------
        pd.Series of {0, 1, 2} indexed by business day
        """
        if yield_curve_df is None:
            yield_curve_df = self.yield_curve()
        if cpi_series is None:
            cpi_series = self.cpi_yoy()

        if yield_curve_df.empty:
            return pd.Series(dtype=int, name="macro_regime")

        df = yield_curve_df.copy()

        if not cpi_series.empty:
            df["cpi_yoy"] = cpi_series.reindex(df.index).ffill().fillna(3.0)
        else:
            df["cpi_yoy"] = 3.0

        def _classify(row):
            inverted   = row.get("inverted", 0) == 1
            high_infl  = row.get("cpi_yoy", 3.0) > 5.0
            low_infl   = row.get("cpi_yoy", 3.0) < 3.0
            steep_curv = row.get("spread_10_2", 0.5) > 0.75

            if inverted or high_infl:
                return 0   # Contractionary
            if steep_curv and low_infl:
                return 2   # Expansionary
            return 1       # Neutral

        regime = df.apply(_classify, axis=1).rename("macro_regime").astype(int)
        return regime

    # ------------------------------------------------------------------
    def build_macro_features(
        self,
        df_index: pd.DatetimeIndex,
        vix_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame aligned with *df_index* for integration
        into the main feature engineering pipeline.

        Merges all alternative data sources into a single DataFrame:
          - spread_10_2   : 10Y-2Y yield spread
          - inverted      : yield curve inversion flag
          - fedfunds      : Fed Funds Rate
          - cpi_yoy       : CPI year-over-year inflation
          - fear_greed    : Fear & Greed proxy [0, 100]
          - macro_regime  : {0, 1, 2}

        Parameters
        ----------
        df_index : DatetimeIndex of the main OHLCV DataFrame
                   (used for alignment and forward-filling)
        vix_df   : VIX daily DataFrame (optional)
        spy_df   : SPY daily DataFrame (optional)

        Returns
        -------
        pd.DataFrame aligned with df_index, all NaN filled with defaults
        """
        # Normalise index to tz-naive date for alignment
        if df_index.tz is not None:
            dates = df_index.tz_localize(None)
        else:
            dates = df_index

        macro_features = pd.DataFrame(index=dates)

        # 1. Yield curve
        try:
            yc = self.yield_curve()
            for col in ["spread_10_2", "inverted", "t10y", "t2y"]:
                if col in yc.columns:
                    macro_features[col] = yc[col].reindex(dates).ffill().fillna(
                        {"spread_10_2": 0.5, "inverted": 0, "t10y": 4.0, "t2y": 3.5}[col]
                    )
        except Exception as e:
            logger.warning("Yield curve feature skipped: %s", e)
            macro_features["spread_10_2"] = 0.5
            macro_features["inverted"]    = 0

        # 2. Fed Funds Rate
        try:
            ffr = self.fed_funds_rate()
            if not ffr.empty:
                macro_features["fedfunds"] = ffr.reindex(dates).ffill().fillna(4.5)
            else:
                macro_features["fedfunds"] = 4.5
        except Exception as e:
            logger.warning("Fed Funds feature skipped: %s", e)
            macro_features["fedfunds"] = 4.5

        # 3. CPI YoY
        try:
            cpi = self.cpi_yoy()
            if not cpi.empty:
                macro_features["cpi_yoy"] = cpi.reindex(dates).ffill().fillna(3.0)
            else:
                macro_features["cpi_yoy"] = 3.0
        except Exception as e:
            logger.warning("CPI feature skipped: %s", e)
            macro_features["cpi_yoy"] = 3.0

        # 4. Fear & Greed proxy
        try:
            fg = self.fear_greed_proxy(vix_df=vix_df, spy_df=spy_df)
            if not fg.empty:
                fg_dates = fg.index
                if fg_dates.tz is not None:
                    fg_dates = fg_dates.tz_localize(None)
                fg_reindexed = fg.copy()
                fg_reindexed.index = fg_dates
                macro_features["fear_greed"] = fg_reindexed.reindex(dates).ffill().fillna(50.0)
            else:
                macro_features["fear_greed"] = 50.0
        except Exception as e:
            logger.warning("Fear & Greed feature skipped: %s", e)
            macro_features["fear_greed"] = 50.0

        # 5. Macro regime
        try:
            regime = self.macro_regime()
            if not regime.empty:
                macro_features["macro_regime"] = regime.reindex(dates).ffill().fillna(1).astype(int)
            else:
                macro_features["macro_regime"] = 1
        except Exception as e:
            logger.warning("Macro regime feature skipped: %s", e)
            macro_features["macro_regime"] = 1

        # Restore original index with timezone if needed
        macro_features.index = df_index

        logger.info(
            "Alternative data features built: %d rows x %d columns",
            len(macro_features), len(macro_features.columns),
        )
        return macro_features


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------

def load_macro_features(
    df_index: pd.DatetimeIndex,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    One-call loader for all macro/alternative features.

    Returns a DataFrame aligned with *df_index* ready to be pd.concat'd
    with the main feature DataFrame.

    Usage in main.py / feature_engineering.py:
        macro_df = load_macro_features(df.index, vix_df=vix_df, spy_df=spy_raw)
        df = pd.concat([df, macro_df], axis=1)
    """
    loader = AlternativeDataLoader(cache=cache)
    return loader.build_macro_features(df_index, vix_df=vix_df, spy_df=spy_df)
