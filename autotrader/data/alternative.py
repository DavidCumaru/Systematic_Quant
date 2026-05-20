"""
alternative.py
===================
Dados alternativos gratuitos para o pipeline Systematic Alpha (mercado brasileiro).

Todos os dados são obtidos de APIs públicas — sem necessidade de chaves pagas.

Fontes
------
  1. FRED (Federal Reserve Economic Data) — séries macroeconômicas do Brasil
     - Taxa de juros longa (IRLTLT01BRM156N) — equivalente à NTN-B longa
     - SELIC (IRSTCI01BRM156N) — taxa básica de juros
     - IPCA YoY (BRACPIALLMINMEI) — inflação ao consumidor
     - Spread longo-SELIC: indicador de inclinação da curva de juros BR
     Base URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES

  2. Proxy de Medo & Ganância (derivado de dados de mercado)
     Índice composto aproximando o sentimento de mercado usando:
       - VIX (medo global — CBOE)
       - BOVA11.SA momentum de 125 dias (tendência do Ibovespa)
       - Proxy de spread de crédito BR (BOVA11 vs IMAB11)
     Normalizado para [0, 100]: 0 = medo extremo, 100 = ganância extrema

  3. Curva de Juros Brasil
     - Spread longo-SELIC (equivalente ao 10Y-2Y americano)
     - Flag de inversão (spread < 0 = curva invertida = alerta recessivo)

  4. Regime Macro
     - 3 estados:
         0 = Contracionista (curva invertida OU IPCA > 6%)
         1 = Neutro
         2 = Expansionista (curva normal E IPCA 2-4%)

Todas as séries são retornadas como DataFrames diários com DatetimeIndex tz-naive.
Devem ser mescladas com o DataFrame de features como contexto macro adicional.

Uso
---
    from autotrader.data.alternative import AlternativeDataLoader

    loader = AlternativeDataLoader()

    yield_curve = loader.yield_curve()          # daily DataFrame
    fear_greed  = loader.fear_greed_proxy()     # daily Series [0, 100]
    macro       = loader.macro_regime()         # daily Series {0, 1, 2}

    # Merge com o DataFrame de features (barras diárias)
    df = df.join(yield_curve, how="left").ffill()

O FRED não requer chave de API para séries públicas (endpoint CSV).
Se o FRED estiver inacessível, todos os métodos retornam valores default.
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Mapa de ticker B3 → termo de busca no Google Trends (BR)
# Busca pelo ticker sem sufixo .SA é mais preciso que nome da empresa
_TICKER_SEARCH_MAP: dict[str, str] = {
    "PETR4.SA":  "PETR4",
    "VALE3.SA":  "VALE3",
    "BBAS3.SA":  "BBAS3",
    "ITUB4.SA":  "ITUB4",
    "WEGE3.SA":  "WEGE3",
    "PRIO3.SA":  "PRIO3",
    "CSNA3.SA":  "CSNA3",
    "TOTS3.SA":  "TOTS3",
    "GGBR4.SA":  "GGBR4",
    "CSAN3.SA":  "CSAN3",
    "RENT3.SA":  "RENT3",
    "MGLU3.SA":  "MGLU3",
    "MRVE3.SA":  "MRVE3",
    "KLBN11.SA": "KLBN11",
    "SMAL11.SA": "SMAL11",
    "BOVA11.SA": "BOVA11",
    "ABEV3.SA":  "ABEV3",
    "LREN3.SA":  "LREN3",
}

# ---------------------------------------------------------------------------
# FRED series IDs — Brasil
# ---------------------------------------------------------------------------
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    "juro_longo":  "IRLTLT01BRM156N",  # Taxa de juro longa do Brasil (% a.a., mensal)
    "selic":       "IRSTCI01BRM156N",  # Taxa SELIC (% a.a., mensal)
    "ipca":        "BRACPIALLMINMEI",  # IPCA — índice de preços ao consumidor (nível mensal)
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
        Curva de juros Brasil: juro longo vs SELIC.

        Columns
        -------
        t10y        : Taxa de juro longa do Brasil (% a.a.)
        t2y         : Taxa SELIC (% a.a.) — proxy de juro curto
        spread_10_2 : Spread longo-SELIC (inclinação da curva)
        inverted    : 1 se spread < 0 (curva invertida = alerta recessivo)

        Defaults (quando FRED indisponível):
          t10y=12.0, t2y=10.5, spread=1.5, inverted=0
        """
        juro_longo_s = self._get_fred(FRED_SERIES["juro_longo"])
        selic_s      = self._get_fred(FRED_SERIES["selic"])

        if juro_longo_s.empty and selic_s.empty:
            logger.warning("Curva de juros BR: FRED indisponível — usando defaults")
            idx = pd.date_range("2000-01-03", periods=1, freq="B")
            return pd.DataFrame(
                {"t10y": 12.0, "t2y": 10.5, "spread_10_2": 1.5, "inverted": 0},
                index=idx,
            )

        valid_series = [s for s in [juro_longo_s, selic_s] if not s.empty]
        all_idx = pd.date_range(
            start=min(s.first_valid_index() for s in valid_series),
            end=max(s.last_valid_index() for s in valid_series),
            freq="B",
        )

        df = pd.DataFrame(index=all_idx)
        df["t10y"] = juro_longo_s.reindex(all_idx).ffill().fillna(12.0) if not juro_longo_s.empty else 12.0
        df["t2y"]  = selic_s.reindex(all_idx).ffill().fillna(10.5)      if not selic_s.empty      else 10.5
        df["spread_10_2"] = df["t10y"] - df["t2y"]
        df["inverted"]    = (df["spread_10_2"] < 0).astype(int)

        return df.round(4)

    # ------------------------------------------------------------------
    def fed_funds_rate(self) -> pd.Series:
        """
        Taxa SELIC — política monetária brasileira.

        Retorna uma Series diária (forward-filled das divulgações mensais).
        Default: 10.5% quando FRED indisponível.
        """
        selic = self._get_fred(FRED_SERIES["selic"])
        if selic.empty:
            logger.warning("SELIC: FRED indisponível — usando default 10.5%%")
            return pd.Series(dtype=float, name="fedfunds")

        daily_idx = pd.date_range(selic.first_valid_index(), selic.last_valid_index(), freq="B")
        return selic.reindex(daily_idx).ffill().fillna(10.5).rename("fedfunds")

    # ------------------------------------------------------------------
    def cpi_yoy(self) -> pd.Series:
        """
        IPCA acumulado 12 meses (%).

        Calculado a partir do índice mensal:
            ipca_yoy = (IPCA_t / IPCA_{t-12} - 1) * 100

        Retorna uma Series diária (forward-filled).
        Default: 4.5% quando FRED indisponível.
        """
        ipca = self._get_fred(FRED_SERIES["ipca"])
        if ipca.empty:
            logger.warning("IPCA: FRED indisponível — usando default 4.5%%")
            return pd.Series(dtype=float, name="cpi_yoy")

        yoy = (ipca / ipca.shift(12) - 1) * 100
        daily_idx = pd.date_range(yoy.first_valid_index(), yoy.last_valid_index(), freq="B")
        return yoy.reindex(daily_idx).ffill().fillna(4.5).rename("cpi_yoy")

    # ------------------------------------------------------------------
    def fear_greed_proxy(
        self,
        vix_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
        hyg_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Proxy de Medo & Ganância para o mercado brasileiro [0=medo extremo, 100=ganância extrema].

        Composto de três sub-indicadores normalizados para [0, 100]:
          1. VIX momentum: VIX baixo vs média 30d = ganância (medo global)
          2. BOVA11.SA momentum de 125 dias (tendência do Ibovespa)
          3. BOVA11 vs IMAB11 ratio (apetite por risco BR vs renda fixa inflação)

        Parâmetros
        ----------
        vix_df : DataFrame diário do VIX (opcional — baixado se não fornecido)
        spy_df : DataFrame OHLCV do BOVA11.SA (opcional — reutiliza se disponível)
        hyg_df : não utilizado (mantido por compatibilidade de interface)

        Retorna
        -------
        pd.Series 'fear_greed' com valores diários em [0, 100]
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

            # Usa BOVA11.SA como benchmark do mercado brasileiro
            if spy_df is None or spy_df.empty:
                raw_bova = yf.download("BOVA11.SA", period="5y", interval="1d",
                                       auto_adjust=True, progress=False)
                if isinstance(raw_bova.columns, pd.MultiIndex):
                    raw_bova.columns = [c[0].lower() for c in raw_bova.columns]
                else:
                    raw_bova.columns = [c.lower() for c in raw_bova.columns]
                if raw_bova.index.tz is not None:
                    raw_bova.index = raw_bova.index.tz_localize(None)
                bova_close = raw_bova["close"]
            else:
                bova_close = spy_df["close"].copy()
                if bova_close.index.tz is not None:
                    bova_close.index = bova_close.index.tz_localize(None)

            def _percentile_rank(s: pd.Series, window: int = 250) -> pd.Series:
                """Roll-rank cada valor dentro da janela trailing: [0, 100]."""
                return s.rolling(window, min_periods=50).apply(
                    lambda x: float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1) * 100,
                    raw=True,
                )

            # Sub-indicador 1: VIX momentum — VIX baixo = ganância (invertido)
            vix_ma30   = vix_close.rolling(30, min_periods=15).mean()
            vix_signal = vix_close / vix_ma30.replace(0, np.nan)
            vix_fg     = 100 - _percentile_rank(vix_signal)

            # Sub-indicador 2: BOVA11 momentum 125d — momentum positivo = ganância
            bova_mom = bova_close / bova_close.shift(125).replace(0, np.nan) - 1
            bova_fg  = _percentile_rank(bova_mom)

            # Sub-indicador 3: BOVA11 vs IMAB11 (ações vs renda fixa inflação BR)
            try:
                raw_imab = yf.download("IMAB11.SA", period="5y", interval="1d",
                                       auto_adjust=True, progress=False)
                if isinstance(raw_imab.columns, pd.MultiIndex):
                    raw_imab.columns = [c[0].lower() for c in raw_imab.columns]
                else:
                    raw_imab.columns = [c.lower() for c in raw_imab.columns]
                if raw_imab.index.tz is not None:
                    raw_imab.index = raw_imab.index.tz_localize(None)
                imab_close = raw_imab["close"]

                credit_ratio = bova_close / imab_close.reindex(bova_close.index).ffill().replace(0, np.nan)
                credit_mom   = credit_ratio / credit_ratio.shift(20).replace(0, np.nan) - 1
                credit_fg    = _percentile_rank(credit_mom)
            except Exception:
                credit_fg = pd.Series(50.0, index=bova_close.index)

            # Combina nos índices comuns (pesos iguais)
            common_idx = vix_fg.index.intersection(bova_fg.index).intersection(credit_fg.index)
            composite  = (
                vix_fg.reindex(common_idx)    * 0.40
                + bova_fg.reindex(common_idx) * 0.35
                + credit_fg.reindex(common_idx) * 0.25
            ).clip(0, 100)

            return composite.rename("fear_greed")

        except Exception as exc:
            logger.warning("fear_greed_proxy falhou: %s — retornando neutro 50", exc)
            return pd.Series(dtype=float, name="fear_greed")

    # ------------------------------------------------------------------
    def google_trends_interest(
        self,
        ticker: str,
        period_years: int = 3,
    ) -> pd.Series:
        """
        Interesse de busca no Google Trends para um ticker B3 (Brasil).

        Retorna uma Series semanal interpolada para diária com valores [0, 100]:
          0   = nenhuma busca relativa (desinteresse)
          100 = pico histórico de buscas (atenção máxima do mercado)

        Correlação comprovada em pesquisas acadêmicas:
          - Picos de busca precedem aumento de volatilidade em 1-3 dias
          - Útil como sinal de atenção de varejo (retail attention proxy)

        Parâmetros
        ----------
        ticker       : ticker B3 com sufixo .SA (ex: "PETR4.SA")
        period_years : anos de histórico (máx 5 para dados semanais via pytrends)

        Requer: pip install pytrends

        Retorna pd.Series com índice diário tz-naive.
        Retorna Series vazia em caso de falha (sem interromper o pipeline).
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.warning(
                "pytrends não instalado — Google Trends desativado. "
                "Instale com: pip install pytrends"
            )
            return pd.Series(dtype=float, name="google_trends")

        # Usa cache em memória para evitar múltiplas chamadas por ticker
        cache_key = f"trends_{ticker}_{period_years}"
        if self._use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Resolve o termo de busca: usa mapa ou extrai do ticker
        search_term = _TICKER_SEARCH_MAP.get(
            ticker,
            ticker.replace(".SA", "").strip()
        )

        end_date   = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=period_years)
        timeframe  = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

        try:
            pytrends = TrendReq(hl="pt-BR", tz=180, timeout=(10, 25))
            pytrends.build_payload(
                [search_term],
                cat=0,
                timeframe=timeframe,
                geo="BR",
                gprop="",
            )
            time.sleep(0.5)   # evita rate-limit do Google
            df_trends = pytrends.interest_over_time()

            if df_trends.empty or search_term not in df_trends.columns:
                logger.warning("Google Trends: sem dados para '%s'.", search_term)
                return pd.Series(dtype=float, name="google_trends")

            # Série semanal → diária por interpolação linear
            weekly = df_trends[search_term].astype(float)
            if weekly.index.tz is not None:
                weekly.index = weekly.index.tz_localize(None)

            daily_idx = pd.date_range(weekly.index[0], weekly.index[-1], freq="D")
            daily     = weekly.reindex(daily_idx).interpolate(method="time")

            # Z-score rolling 52 semanas para remover sazonalidade
            z = (daily - daily.rolling(365, min_periods=52).mean()) / (
                daily.rolling(365, min_periods=52).std().replace(0, np.nan)
            )
            result = z.rename("google_trends")

            logger.info(
                "Google Trends [%s / '%s']: %d dias carregados",
                ticker, search_term, len(result.dropna()),
            )

            if self._use_cache:
                self._cache[cache_key] = result

            return result

        except Exception as exc:
            logger.warning("Google Trends falhou para '%s': %s", ticker, exc)
            return pd.Series(dtype=float, name="google_trends")

    # ------------------------------------------------------------------
    def macro_regime(
        self,
        yield_curve_df: Optional[pd.DataFrame] = None,
        cpi_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Indicador de regime macro brasileiro — 3 estados.

        Estados
        -------
          0 = Contracionista  (curva invertida OU IPCA > 6%)
          1 = Neutro          (baseline)
          2 = Expansionista   (curva normal E IPCA 2-4%)

        Parâmetros
        ----------
        yield_curve_df : saída de yield_curve() (opcional — buscado se None)
        cpi_series     : saída de cpi_yoy()     (opcional — buscado se None)

        Retorna
        -------
        pd.Series de {0, 1, 2} indexado por dia útil
        """
        if yield_curve_df is None:
            yield_curve_df = self.yield_curve()
        if cpi_series is None:
            cpi_series = self.cpi_yoy()

        if yield_curve_df.empty:
            return pd.Series(dtype=int, name="macro_regime")

        df = yield_curve_df.copy()

        if not cpi_series.empty:
            df["cpi_yoy"] = cpi_series.reindex(df.index).ffill().fillna(4.5)
        else:
            df["cpi_yoy"] = 4.5

        def _classify(row):
            inverted   = row.get("inverted", 0) == 1
            high_infl  = row.get("cpi_yoy", 4.5) > 6.0   # IPCA acima da banda superior
            low_infl   = row.get("cpi_yoy", 4.5) < 4.0   # IPCA dentro/abaixo da meta BR
            steep_curv = row.get("spread_10_2", 1.5) > 2.0  # curva inclinada positivamente

            if inverted or high_infl:
                return 0   # Contracionista
            if steep_curv and low_infl:
                return 2   # Expansionista
            return 1       # Neutro

        regime = df.apply(_classify, axis=1).rename("macro_regime").astype(int)
        return regime

    # ------------------------------------------------------------------
    def build_macro_features(
        self,
        df_index: pd.DatetimeIndex,
        vix_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None,
        ticker: str = "",
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

        # 6. Google Trends — interesse de busca pelo ticker (retail attention proxy)
        if ticker:
            try:
                gt = self.google_trends_interest(ticker)
                if not gt.empty:
                    gt_dates = gt.index
                    if gt_dates.tz is not None:
                        gt_dates = gt_dates.tz_localize(None)
                    gt_reindexed = gt.copy()
                    gt_reindexed.index = gt_dates
                    macro_features["google_trends"] = (
                        gt_reindexed.reindex(dates).ffill().bfill().fillna(0.0)
                    )
                else:
                    macro_features["google_trends"] = 0.0
            except Exception as e:
                logger.warning("Google Trends feature skipped: %s", e)
                macro_features["google_trends"] = 0.0

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
    ticker: str = "",
    cache: bool = True,
) -> pd.DataFrame:
    """
    One-call loader for all macro/alternative features.

    Returns a DataFrame aligned with *df_index* ready to be pd.concat'd
    with the main feature DataFrame.

    Usage in main.py / feature_engineering.py:
        macro_df = load_macro_features(df.index, vix_df=vix_df, spy_df=spy_raw, ticker="PETR4.SA")
        df = pd.concat([df, macro_df], axis=1)
    """
    loader = AlternativeDataLoader(cache=cache)
    return loader.build_macro_features(df_index, vix_df=vix_df, spy_df=spy_df, ticker=ticker)
