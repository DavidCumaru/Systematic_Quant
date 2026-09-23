"""
Testes (sem internet) das fontes de notícias e do mapa CVM em news.py.
"""
import pytest

from autotrader.data.news import (
    _FALSOS_POSITIVOS,
    _RSS_SOURCES,
    _TICKER_KEYWORDS,
    _TICKER_TO_CVM,
    _menciona,
)


def test_codigos_cvm_sem_repeticao():
    # GGBR4 usava o mesmo código da VALE3 (4170): cada empresa tem o seu.
    codigos = list(_TICKER_TO_CVM.values())
    repetidos = {c for c in codigos if codigos.count(c) > 1}
    assert not repetidos, f"códigos CVM repetidos: {repetidos}"


@pytest.mark.parametrize("ticker, codigo", [
    ("VIVT3.SA", 17671),  # Telefônica Brasil (antes: 22470 = Magazine Luiza)
    ("SUZB3.SA", 13986),  # Suzano S.A.
    ("EQTL3.SA", 20010),  # Equatorial S.A. (antes: subsidiária do Maranhão)
    ("GGBR4.SA", 3980),   # Gerdau (antes: código da Vale)
])
def test_codigos_cvm_conferidos(ticker, codigo):
    assert _TICKER_TO_CVM[ticker] == codigo


def test_feed_da_reuters_removido():
    assert not any("reuters" in url for url in _RSS_SOURCES)


@pytest.mark.parametrize("texto, ticker, esperado", [
    ("ibovespa hoje ao vivo: bolsa recua", "VIVT3.SA", False),
    ("vivo lança plano 5g para empresas", "VIVT3.SA", True),
    ("petróleo na margem equatorial avança", "EQTL3.SA", False),
    ("equatorial energia tem lucro recorde", "EQTL3.SA", True),
    ("prioridade do governo é a reforma", "PRIO3.SA", False),
    ("prio anuncia recompra de ações", "PRIO3.SA", True),
    ("vale a pena investir em bancos?", "VALE3.SA", False),
    ("vale anuncia dividendos extraordinários", "VALE3.SA", True),
    ("gcm de suzano prende suspeito", "SUZB3.SA", False),
    ("suzano eleva preço da celulose", "SUZB3.SA", True),
])
def test_menciona_palavra_inteira_e_falsos_positivos(texto, ticker, esperado):
    assert _menciona(texto, ticker, _TICKER_KEYWORDS[ticker]) is esperado


def test_falsos_positivos_so_de_tickers_conhecidos():
    assert set(_FALSOS_POSITIVOS) <= set(_TICKER_KEYWORDS)
