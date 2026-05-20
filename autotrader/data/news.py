"""
news.py
============
Features de notícias e eventos para o modelo quantitativo.

Três fontes implementadas:
  1. Fatos Relevantes (CVM) — divulgações obrigatórias: resultados, dividendos,
     fusões, etc. Feature binária: houve fato relevante nos últimos N dias.

  2. Sentimento de manchetes — RSS do InfoMoney + Valor Econômico.
     Score positivo/negativo por ticker. Abordagem lexicon-based em português.

  3. Surpresa de resultados (CVM DFP/ITR) — compara lucro líquido divulgado
     com a média dos últimos 4 trimestres. Feature: pct de surpresa.

Todas as features são causais (usam apenas dados disponíveis até a data).
Dados são cacheados em data/news_cache/ para evitar re-downloads.

Uso:
    loader = NewsDataLoader(cache=True)
    feat   = loader.build_news_features("BBAS3.SA", dates=feat_df.index)
    # retorna DataFrame com colunas:
    #   fato_rel_1d, fato_rel_5d, fato_rel_tipo
    #   news_sentiment_5d, news_sentiment_10d
    #   earnings_surprise, days_since_earnings
"""

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

_CACHE_DIR  = Path("data/news_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CVM_REGISTRY_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
_CVM_DFP_URL      = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
_CVM_ITR_URL      = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_{year}.zip"
_CVM_IPE_URL      = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{year}.zip"

# Categorias IPE que representam eventos materiais
_IPE_MATERIAL_CATS = {
    "Fato Relevante", "Comunicado ao Mercado", "Aviso aos Acionistas",
    "Resultado", "Dividendo", "Fusão", "Aquisição", "Desinvestimento",
    "Oferta Publica", "Mudança de Controle",
}

_RSS_SOURCES = [
    "https://www.infomoney.com.br/feed/",
    "https://br.reuters.com/rssFeed/businessNews",
]

# Mapeamento ticker → código CVM (CD_CVM)
_TICKER_TO_CVM: dict[str, int] = {
    "BBAS3.SA":  1023,
    "ITUB4.SA":  19348,   # Itaú Unibanco Holding
    "PRIO3.SA":  22187,   # PetroRio
    "EQTL3.SA":  16608,   # Equatorial Energia (holding)
    "LREN3.SA":  8133,
    "SUZB3.SA":  4065,
    "VIVT3.SA":  22470,   # Telefônica Brasil
    "CSNA3.SA":  4030,
    "KLBN11.SA": 12653,
    "CYRE3.SA":  14460,
    "VALE3.SA":  4170,
    "PETR4.SA":  9512,
    "WEGE3.SA":  5410,
    "ABEV3.SA":  14761,   # também usado por LREN no passado; ABEV = 14761 não encontrado, placeholder
    "MGLU3.SA":  20303,
    "RENT3.SA":  14869,
    "RADL3.SA":  16284,
    "BEEF3.SA":  16152,
    "GGBR4.SA":  4170,
    "CSAN3.SA":  18724,
    "TIMS3.SA":  22640,
    "MRVE3.SA":  20532,
    "SBSP3.SA":  5444,
    "TOTS3.SA":  10629,
    "UGPA3.SA":  12441,
}

# Mapeamento ticker → palavras-chave para filtrar manchetes
_TICKER_KEYWORDS: dict[str, list[str]] = {
    "BBAS3.SA":  ["banco do brasil", "bb sa", "bb s.a", "bbas"],
    "ITUB4.SA":  ["itau", "itaú", "unibanco", "itub"],
    "PRIO3.SA":  ["petrorio", "petro rio", "prio"],
    "EQTL3.SA":  ["equatorial", "eqtl"],
    "LREN3.SA":  ["renner", "lojas renner", "lren"],
    "SUZB3.SA":  ["suzano", "suzb"],
    "VIVT3.SA":  ["telefonica", "telefônica", "vivo", "vivt"],
    "CSNA3.SA":  ["csn", "siderurgica nacional", "csna"],
    "KLBN11.SA": ["klabin", "klbn"],
    "CYRE3.SA":  ["cyrela", "cyre"],
    "VALE3.SA":  ["vale", "vale3"],
    "PETR4.SA":  ["petrobras", "petro", "petr4"],
    "WEGE3.SA":  ["weg ", "weg,", "weg."],
    "ABEV3.SA":  ["ambev", "abev"],
    "MGLU3.SA":  ["magazine luiza", "magalu", "mglu"],
    "RENT3.SA":  ["localiza", "rent3"],
    "RADL3.SA":  ["raia drogasil", "radl", "rd saude", "rd saúde"],
    "BEEF3.SA":  ["minerva", "beef"],
    "GGBR4.SA":  ["gerdau", "ggbr"],
    "CSAN3.SA":  ["cosan", "csan"],
    "TIMS3.SA":  ["tim brasil", "tim s.a", "tims"],
    "MRVE3.SA":  ["mrv", "engenharia"],
    "SBSP3.SA":  ["sabesp", "sbsp"],
    "TOTS3.SA":  ["totvs", "tots"],
    "UGPA3.SA":  ["ultrapar", "ultra", "ugpa"],
}

# Léxico financeiro em português: palavras positivas e negativas
_POSITIVE_WORDS = {
    "alta", "subiu", "cresceu", "crescimento", "lucro", "lucros", "resultado",
    "positivo", "recorde", "máximo", "valorização", "valorizou", "superou",
    "surpreendeu", "acima", "melhor", "elevou", "ganho", "ganhos", "receita",
    "forte", "robusto", "expansão", "compra", "recomenda", "upgrades",
    "dividend", "dividendo", "aprovado", "aprovação", "acordo", "contrato",
    "parceria", "investimento", "captação", "emissão bem-sucedida",
    "supera", "bate", "recorde", "aprovados", "elevar", "elevámos",
}

_NEGATIVE_WORDS = {
    "queda", "caiu", "recuou", "prejuízo", "perda", "perdas", "negativo",
    "mínimo", "desvalorização", "desvalorizou", "abaixo", "pior", "reduziu",
    "fraco", "contração", "venda", "rebaixamento", "downgrade", "downgrades",
    "multa", "processo", "ação judicial", "crise", "problema", "dificuldade",
    "atraso", "cancelamento", "cancelado", "suspensão", "suspendeu",
    "impacto negativo", "revisou para baixo", "corte", "cortou",
}


# ---------------------------------------------------------------------------
# NewsDataLoader
# ---------------------------------------------------------------------------

class NewsDataLoader:
    """
    Carrega e processa dados de notícias e eventos para features do modelo.

    Parâmetros
    ----------
    cache : bool
        Se True, armazena resultados em data/news_cache/ para reutilização.
    """

    def __init__(self, cache: bool = True):
        self.cache = cache
        self._registry:   Optional[pd.DataFrame] = None
        self._fatos_cache: dict[int, pd.DataFrame] = {}
        self._earnings_cache: dict[str, pd.DataFrame] = {}

    # ==========================================================================
    # PARTE 1 — FATOS RELEVANTES CVM
    # ==========================================================================

    def _get_registry(self) -> pd.DataFrame:
        """Baixa e cacheia o registro de empresas da CVM."""
        if self._registry is not None:
            return self._registry
        cache_path = _CACHE_DIR / "cvm_registry.parquet"
        if self.cache and cache_path.exists():
            self._registry = pd.read_parquet(cache_path)
            return self._registry
        try:
            r = requests.get(_CVM_REGISTRY_URL, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.content.decode("latin-1")), sep=";")
            if self.cache:
                df.to_parquet(cache_path)
            self._registry = df
        except Exception as e:
            logger.warning("CVM registry download failed: %s", e)
            self._registry = pd.DataFrame()
        return self._registry

    def _load_fatos_year(self, year: int) -> pd.DataFrame:
        """
        Baixa divulgações IPE da CVM para um ano.
        IPE contém: Fatos Relevantes, Comunicados, Resultados, Dividendos, etc.
        """
        if year in self._fatos_cache:
            return self._fatos_cache[year]

        cache_path = _CACHE_DIR / f"ipe_{year}.parquet"
        if self.cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._fatos_cache[year] = df
            return df

        url = _CVM_IPE_URL.format(year=year)
        df  = pd.DataFrame()
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                z       = zipfile.ZipFile(io.BytesIO(r.content))
                csv_f   = [f for f in z.namelist() if f.endswith(".csv")]
                if csv_f:
                    df = pd.read_csv(z.open(csv_f[0]), sep=";", encoding="latin-1")
                    # Normaliza colunas de data
                    for dc in ["Data_Referencia", "Data_Entrega"]:
                        if dc in df.columns:
                            df[dc] = pd.to_datetime(df[dc], errors="coerce")
        except Exception as e:
            logger.warning("IPE download failed (%d): %s", year, e)

        if self.cache and not df.empty:
            df.to_parquet(cache_path)
        self._fatos_cache[year] = df
        return df

    def _load_dfp_earnings(self, year: int) -> pd.DataFrame:
        """Baixa resultados anuais (DFP/DRE) da CVM para um ano."""
        cache_path = _CACHE_DIR / f"dfp_dre_{year}.parquet"
        if self.cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        url = _CVM_DFP_URL.format(year=year)
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            z   = zipfile.ZipFile(io.BytesIO(r.content))
            dre = [f for f in z.namelist() if "DRE_con" in f]
            if not dre:
                dre = [f for f in z.namelist() if "DRE" in f]
            if not dre:
                return pd.DataFrame()
            df = pd.read_csv(z.open(dre[0]), sep=";", encoding="latin-1")
            df["DT_REFER"] = pd.to_datetime(df["DT_REFER"], errors="coerce")
            if self.cache:
                df.to_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning("DFP download failed (%d): %s", year, e)
            return pd.DataFrame()

    def _load_itr_earnings(self, year: int) -> pd.DataFrame:
        """Baixa resultados trimestrais (ITR/DRE) da CVM para um ano."""
        cache_path = _CACHE_DIR / f"itr_dre_{year}.parquet"
        if self.cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        url = _CVM_ITR_URL.format(year=year)
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            z   = zipfile.ZipFile(io.BytesIO(r.content))
            dre = [f for f in z.namelist() if "DRE_con" in f]
            if not dre:
                dre = [f for f in z.namelist() if "DRE" in f]
            if not dre:
                return pd.DataFrame()
            df = pd.read_csv(z.open(dre[0]), sep=";", encoding="latin-1")
            df["DT_REFER"] = pd.to_datetime(df["DT_REFER"], errors="coerce")
            if self.cache:
                df.to_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning("ITR download failed (%d): %s", year, e)
            return pd.DataFrame()

    # --------------------------------------------------------------------------
    def get_fato_features(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
        years: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Cria features de Fatos Relevantes para o ticker.

        Retorna DataFrame com índice=dates e colunas:
          fato_rel_1d  : 1 se houve fato relevante hoje
          fato_rel_5d  : contagem de fatos relevantes nos últimos 5 dias
        """
        dates_norm = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
        result = pd.DataFrame(
            {"fato_rel_1d": 0.0, "fato_rel_5d": 0.0},
            index=dates_norm,
        )

        cvm_code = _TICKER_TO_CVM.get(ticker)
        if cvm_code is None:
            logger.debug("fatos: no CVM code for %s", ticker)
            return result

        if years is None:
            min_year = min(d.year for d in dates_norm)
            max_year = max(d.year for d in dates_norm)
            years = list(range(min_year, max_year + 1))

        all_fatos: list[pd.Timestamp] = []
        for yr in years:
            df = self._load_fatos_year(yr)
            if df.empty:
                continue
            # Filtra pela empresa (IPE usa Codigo_CVM)
            cvm_col  = next((c for c in ["Codigo_CVM", "CD_CVM"] if c in df.columns), None)
            date_col = next((c for c in ["Data_Entrega", "Data_Referencia", "DT_REFER"] if c in df.columns), None)
            if cvm_col is None or date_col is None:
                continue
            sub = df[df[cvm_col] == cvm_code]
            # Filtra apenas categorias de eventos materiais (se coluna existir)
            if "Categoria" in sub.columns:
                mat_mask = sub["Categoria"].str.contains(
                    "Fato Relevante|Comunicado|Resultado|Dividendo|Fusão|Aquisição|Oferta",
                    na=False, case=False,
                )
                sub = sub[mat_mask]
            sub = sub[[date_col]].dropna()
            sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce").dt.normalize()
            all_fatos.extend(sub[date_col].dropna().tolist())

        if not all_fatos:
            return result

        fato_dates = pd.DatetimeIndex(sorted(set(all_fatos)))

        for date in dates_norm:
            # 1d
            result.loc[date, "fato_rel_1d"] = float(date in fato_dates)
            # 5d
            window_start = date - pd.Timedelta(days=5)
            count = int(((fato_dates >= window_start) & (fato_dates <= date)).sum())
            result.loc[date, "fato_rel_5d"] = float(count)

        return result

    # ==========================================================================
    # PARTE 2 — SENTIMENTO DE MANCHETES (RSS)
    # ==========================================================================

    def _fetch_headlines(self, ticker: str, days_back: int = 30) -> list[dict]:
        """
        Busca manchetes recentes de fontes RSS para um ticker.
        Retorna lista de {date, title, score}.
        """
        import feedparser

        keywords = _TICKER_KEYWORDS.get(ticker, [ticker.replace(".SA", "").lower()])
        results  = []

        for rss_url in _RSS_SOURCES:
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()
                    text = title + " " + summary

                    # Verifica se é sobre o ticker
                    if not any(kw in text for kw in keywords):
                        continue

                    # Data
                    pub = entry.get("published_parsed") or entry.get("updated_parsed")
                    if pub:
                        date = pd.Timestamp(*pub[:6]).normalize()
                    else:
                        date = pd.Timestamp.now().normalize()

                    score = self._score_sentiment(text)
                    results.append({"date": date, "title": title[:100], "score": score})
                time.sleep(0.3)
            except Exception as e:
                logger.debug("RSS fetch failed (%s): %s", rss_url, e)

        return results

    @staticmethod
    def _score_sentiment(text: str) -> float:
        """
        Escore de sentimento lexical em português.
        Retorna valor entre -1 (negativo) e +1 (positivo).
        """
        text  = text.lower()
        words = set(text.split())
        pos   = len(words & _POSITIVE_WORDS)
        neg   = len(words & _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 4)

    def get_sentiment_features(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Cria features de sentimento de manchetes para o ticker.

        Retorna DataFrame com:
          news_sentiment_5d  : sentimento médio dos últimos 5 dias
          news_sentiment_10d : sentimento médio dos últimos 10 dias
        """
        dates_norm = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
        result = pd.DataFrame(
            {"news_sentiment_5d": 0.0, "news_sentiment_10d": 0.0},
            index=dates_norm,
        )

        # Tenta cache de manchetes históricas
        cache_path = _CACHE_DIR / f"headlines_{ticker.replace('.','_')}.parquet"

        headlines_df = pd.DataFrame()
        if self.cache and cache_path.exists():
            try:
                headlines_df = pd.read_parquet(cache_path)
            except Exception:
                pass

        # Busca manchetes recentes (só se não tiver cache ou cache estiver velho)
        needs_refresh = headlines_df.empty or (
            not headlines_df.empty and
            (pd.Timestamp.now() - headlines_df["date"].max()).days > 1
        )

        if needs_refresh:
            fresh = self._fetch_headlines(ticker, days_back=60)
            if fresh:
                fresh_df = pd.DataFrame(fresh)
                if not headlines_df.empty:
                    headlines_df = pd.concat([headlines_df, fresh_df]).drop_duplicates("title")
                else:
                    headlines_df = fresh_df
                if self.cache:
                    headlines_df.to_parquet(cache_path)

        if headlines_df.empty:
            return result

        headlines_df["date"] = pd.to_datetime(headlines_df["date"]).dt.normalize()

        for date in dates_norm:
            for col, days in [("news_sentiment_5d", 5), ("news_sentiment_10d", 10)]:
                start = date - pd.Timedelta(days=days)
                sub   = headlines_df[(headlines_df["date"] >= start) & (headlines_df["date"] <= date)]
                if not sub.empty:
                    result.loc[date, col] = round(sub["score"].mean(), 4)

        return result

    # ==========================================================================
    # PARTE 3 — SURPRESA DE RESULTADOS
    # ==========================================================================

    def _get_earnings_series(self, ticker: str, years: list[int]) -> pd.DataFrame:
        """
        Retorna série de lucro líquido trimestral (ITR) + anual (DFP).
        Colunas: date, lucro_liq, tipo (ITR/DFP)
        """
        cache_path = _CACHE_DIR / f"earnings_{ticker.replace('.','_')}.parquet"
        if self.cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        cvm_code = _TICKER_TO_CVM.get(ticker)
        if cvm_code is None:
            return pd.DataFrame()

        rows = []
        for yr in years:
            for loader, tipo in [(self._load_itr_earnings, "ITR"), (self._load_dfp_earnings, "DFP")]:
                df = loader(yr)
                if df.empty:
                    continue
                if "CD_CVM" not in df.columns:
                    continue
                sub = df[df["CD_CVM"] == cvm_code].copy()
                if sub.empty:
                    continue

                # Conta de lucro líquido (CD_CONTA começa com 3.11 em DRE padrão)
                lucro_mask = (
                    sub["CD_CONTA"].astype(str).str.startswith("3.11") |
                    sub["DS_CONTA"].str.contains("Lucro Líquido|Resultado Líquido", na=False, case=False)
                )
                lucro_rows = sub[lucro_mask]
                if lucro_rows.empty:
                    continue

                # Pega o maior valor absoluto (lucro líquido consolidado)
                lucro_row = lucro_rows.loc[lucro_rows["VL_CONTA"].abs().idxmax()]
                rows.append({
                    "date":      pd.Timestamp(lucro_row["DT_REFER"]).normalize(),
                    "lucro_liq": float(lucro_row["VL_CONTA"]),
                    "tipo":      tipo,
                })

        if not rows:
            return pd.DataFrame()

        result = (
            pd.DataFrame(rows)
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if self.cache:
            result.to_parquet(cache_path)
        return result

    def get_earnings_features(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
        years: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Cria features de surpresa de resultado para o ticker.

        Retorna DataFrame com:
          earnings_surprise     : (lucro_atual - media_4q) / abs(media_4q) * 100
          days_since_earnings   : dias desde o último resultado divulgado
        """
        dates_norm = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
        result = pd.DataFrame(
            {"earnings_surprise": 0.0, "days_since_earnings": 365.0},
            index=dates_norm,
        )

        if years is None:
            min_year = min(d.year for d in dates_norm)
            max_year = max(d.year for d in dates_norm)
            years = list(range(max(min_year - 3, 2018), max_year + 1))

        earnings = self._get_earnings_series(ticker, years)
        if earnings.empty:
            return result

        for date in dates_norm:
            # Resultados disponíveis até esta data (sem look-ahead)
            past = earnings[earnings["date"] <= date].tail(5)
            if past.empty:
                continue

            last_date  = past["date"].iloc[-1]
            last_lucro = past["lucro_liq"].iloc[-1]
            days_since = (date - last_date).days
            result.loc[date, "days_since_earnings"] = float(min(days_since, 365))

            # Surpresa: compara com média dos 4 trimestres anteriores
            prev = past.iloc[:-1]
            if len(prev) >= 2:
                avg = prev["lucro_liq"].mean()
                if abs(avg) > 1:
                    surprise = (last_lucro - avg) / abs(avg) * 100
                    result.loc[date, "earnings_surprise"] = round(float(np.clip(surprise, -200, 200)), 2)

        return result

    # ==========================================================================
    # MÉTODO PRINCIPAL — combina as 3 fontes
    # ==========================================================================

    def build_news_features(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Combina todas as features de notícias para um ticker.

        Colunas retornadas:
          fato_rel_1d        — fato relevante hoje (0/1)
          fato_rel_5d        — qtd fatos últimos 5 dias
          news_sentiment_5d  — sentimento médio das manchetes (últimos 5d)
          news_sentiment_10d — sentimento médio das manchetes (últimos 10d)
          earnings_surprise  — surpresa vs média 4 trimestres (%)
          days_since_earnings— dias desde último resultado
        """
        dates_norm = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])

        fato     = self.get_fato_features(ticker, dates_norm)
        senti    = self.get_sentiment_features(ticker, dates_norm)
        earnings = self.get_earnings_features(ticker, dates_norm)

        result = fato.join(senti).join(earnings)
        result = result.fillna(0.0)
        result.index = dates_norm
        return result
