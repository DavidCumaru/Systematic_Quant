# Systematic Alpha — Relatório Final de Auditoria

**Data**: 2026-06-28  
**Artefato de backtest**: `backtest_PRIO3_LREN3_SUZB3_EQTL3_+4_20260628_213847.json`  
**DB Hash**: ver artefato JSON  
**Git commit**: ver artefato JSON  
**Seed**: 42  
**Modelo**: lightgbm_specialized (2 LGBMs binários: long + short)  

---

## 1. Métricas do Walk-Forward (DB reconstruído, modelos retreinados)

| Ticker | Sharpe | Sortino | WR% | Trades | Return% | MaxDD% | PF | Calmar |
|---|---|---|---|---|---|---|---|---|
| PRIO3.SA | 0.96 | 8.96 | 65.0 | 40 | 665% | 54% | 4.35 | 0.77 |
| KLBN11.SA | 0.80 | 46.79 | 44.0 | 100 | 677% | 101% | 1.85 | 0.52 |
| LREN3.SA | 0.60 | 32.44 | 43.6 | 101 | 658% | 135% | 1.83 | 0.18 |
| VIVT3.SA | 0.53 | 11.41 | 43.6 | 101 | 655% | 65% | 1.81 | 0.37 |
| EQTL3.SA | 0.35 | 11.67 | 34.2 | 41 | 278% | 96% | 1.74 | 0.16 |
| ITUB4.SA | 0.25 | 15.39 | 28.2 | 124 | 382% | 114% | 1.31 | 0.13 |
| SUZB3.SA | -0.01 | -0.12 | 40.0 | 20 | -2% | 41% | 0.99 | -0.01 |
| BBAS3.SA | -0.01 | -0.20 | 40.0 | 30 | -5% | 84% | 0.98 | 0.00 |
| **Agregado** | **0.43** | — | **42.3** | **557** | **413%** | **86%** | — | — |

**Notas sobre os números**:
- MaxDD >100% em LREN3 e ITUB4 indica que a equity curve ficou negativa em algum ponto com compounding. Na prática, a estratégia teria sido interrompida pelo kill switch (MAX_DRAWDOWN_PCT=10%) muito antes.
- SUZB3 e BBAS3 têm Sharpe ~0 e PF <1 — sem edge detectável nesses tickers.
- Os retornos altos são nominais acumulados ao longo de todo o período do walk-forward (vários anos), não anualizados.

---

## 2. Comparação com o número antigo do comentário em settings.py

**Número antigo** (settings.py:135-136, não auditável):
> "Opcao C (Fix 3%/1% TIME=3): +54.50% retorno, 6.1% MaxDD, 284 trades"

**Número novo** (artefato auditável):
- Agregado: 557 trades, Sharpe médio 0.43, MaxDD médio 86%
- O melhor ticker individual (PRIO3): Sharpe 0.96, 40 trades, MaxDD 54%

**Os números são muito diferentes. Isso é esperado** por 6 razões acumuladas:
1. O número antigo usava parâmetros globais (SL=1%, TP=3%); o novo usa params otimizados por ticker (SL=0.5%, TP=1.2-2.5%)
2. O DB antigo tinha preços historicamente inconsistentes (13 tickers com ajustes stale + 1 com revisão de dados do yfinance)
3. O trend filter não existia no caminho de execução anterior
4. Os modelos foram retreinados do zero sobre dados reconstruídos
5. A calibração condicional foi aplicada (7/16 calibradores mantidos)
6. O número antigo não tem artefato, seed, hash ou commit — não há como saber exatamente quais dados/código/params foram usados

---

## 3. Resultado do live (journal.csv, 39 trades fechados)

O resultado do journal de 39 trades fechados oscilou entre **-R$178.78 (20.5% win rate)** e **+R$57.10 (28.2% win rate)** ao longo desta auditoria, dependendo de qual fonte/momento de dado de preço é usado para verificar as barreiras — uma oscilação de ~R$236, equivalente a ~2.5% do capital por trade.

Isso reflete **instabilidade de dado na fonte** (yfinance revisa histórico retroativamente, conforme caso IMAB11.SA documentado abaixo), não apenas os bugs corrigidos nesta sessão. Os 2 trades de maior impacto financeiro (ITUB4 SELL 04/mai e 25/mai, |delta|=R$75 cada) foram verificados com auto_adjust=True vs False como cross-check semi-independente: as barreiras SL e TP ficam a poucos bps uma da outra nesses trades, e o resultado muda dependendo de qual base de preço é usada.

**O resultado qualitativo central não muda com essa oscilação: marginal ou breakeven, não há edge positivo robusto detectável nesta amostra de 39 trades.**

---

## 4. Limitações documentadas

### 4.1. Custo de execução
Gap mediano de **76 bps** (preço raw) entre entry_price registrado e o open real do dia. O backtest assume open+7bps de slippage — subestima o custo real de execução em ~69 bps na mediana. Causa: ~64% dos trades têm entry próximo do open (movimento real de 40 min); ~36% usam close de D-1 porque yfinance não disponibilizou o candle de D no momento do scan (10:10 BRT).

### 4.2. Sobreposição de posições
Guard contra sinais duplicados no mesmo ticker ativo desde commit 8476877 (20/mai/2026), confirmado sem reversões posteriores (`git log -p 8476877..HEAD -- scheduler.py` = vazio). Trades de 09/abr a 19/mai (pré-guard) tiveram até 9x a exposição de capital assumida pelo position sizer e não são comparáveis ao backtest, que assume posição única por ticker.

### 4.3. Instabilidade de preço na fonte (yfinance)
O PnL do journal oscila dependendo do momento/modo de download dos dados de verificação de barreiras. Dois mecanismos identificados:
- **Dividendos/splits**: `auto_adjust=True` retroativamente altera preços históricos quando um novo corporate action ocorre
- **Revisão de dados**: o caso IMAB11.SA (Jun 2025) mostrou que yfinance revisa volume (58004→0) e close (103→79.50) sem corporate action correspondente. Guard de anomalia implementado em pipeline.py para detectar e bloquear essas revisões silenciosas.

### 4.4. DB de preços historicamente inconsistente (corrigido)
O market_data.db tinha **13/30 tickers** com ajustes históricos inconsistentes causados por `INSERT OR IGNORE` em pipeline.py que nunca propagava re-ajustes de dividendo/split ao histórico já armazenado. Além disso, 1 ticker (IMAB11.SA) tinha inconsistência por revisão de dados do yfinance (causa diferente).

**Tickers afetados e causa**:

| Causa | Tickers | Gap range |
|---|---|---|
| Split major | SBSP3.SA (5:1), VIVT3.SA (2:1) | 4.6-400% |
| Bonificação | ITUB4, MGLU3, RADL3, TIMS3 | 0.4-1.0% |
| Dividendo | ABEV3, PETR4, BBAS3, BEEF3, GGBR4, TOTS3, WEGE3 | 0.2-6.0% |
| Revisão yfinance | IMAB11 | 29.6% (falso: zero-volume bar) |

**Corrigido via**: `INSERT OR REPLACE` (pipeline.py), guard de anomalia (>3% sem corporate action → skip + log), re-download completo do DB (30 tickers), 30 modelos retreinados, grid search re-rodado para 8 LIVE_TICKERS. Qualquer artefato de backtest anterior a 2026-06-28 não é comparável a este.

### 4.5. Calibração condicional
Calibração isotonic aplicada condicionalmente (threshold 0.03 de melhoria no Brier, 3-fold CV dentro do holdout). Resultado nos 8 LIVE_TICKERS pós-rebuild:

| Ticker | Long | Short | Flag |
|---|---|---|---|
| PRIO3 | KEPT (-0.022) | KEPT (-0.035) | |
| SUZB3 | KEPT (-0.058) | KEPT (-0.039) | |
| EQTL3 | KEPT (-0.029) | KEPT (-0.033) | |
| BBAS3 | DISC | KEPT (-0.035) | |
| LREN3 | DISC | DISC | S>baseline |
| ITUB4 | DISC | DISC | **S Brier 0.32 > baseline 0.25** |
| VIVT3 | DISC | DISC | **S Brier 0.30 > baseline 0.25** |
| KLBN11 | DISC | DISC | **S Brier 0.28 > baseline 0.25** |

ITUB4, VIVT3, KLBN11 e LREN3 têm modelos short com Brier pior que o baseline de 0.25 — sem sinal preditivo real na direção short para esses tickers.

---

## 5. Correções implementadas nesta auditoria

| # | Correção | Arquivo(s) | Teste |
|---|---|---|---|
| 1 | Trend filter portado para execution engine | execution/engine.py | test_buy_below_ma_rejected + 5 testes |
| 2 | Ticker params (load_ticker_params) no caminho de execução | execution/engine.py, backtesting/engine.py | test_ticker_params_used_for_sl_tp |
| 3 | Lógica de sinal unificada (signals/core.py) | autotrader/signals/core.py | test_same_signals_produced |
| 4 | MA200 sem candle parcial (look-ahead fix) | execution/engine.py | test_ma_excludes_partial_candle |
| 5 | Calibração condicional (3-fold CV, threshold 0.03) | models/trainer.py | test_calibrator_discarded_when_not_helpful |
| 6 | Journal versionado (strategy_version) | journal.py, journal.csv | — |
| 7 | journal_weekly_close re-harmonizado (load_data do DB) | journal.py | test_uses_load_data_not_yf_download |
| 8 | INSERT OR REPLACE + guard de anomalia | data/pipeline.py | test_replace_updates_existing_bars + test_anomaly_guard |
| 9 | DB reconstruído do zero | data/market_data.db | 30/30 tickers gap ~0 |
| 10 | 30 modelos retreinados + 8 grid searches re-rodados | models/*.pkl, ticker_params.json | — |
| 11 | test_sanity.py fixture corrigido (600 barras diárias) | tests/test_sanity.py | 9/9 sanity tests passando |
| 12 | test_data_pipeline.py import corrigido | tests/test_data_pipeline.py | 17/17 pipeline tests passando |
| 13 | test_feature_engineering.py beta_spy→beta_ibov | tests/test_feature_engineering.py | 19/19 feature tests passando |

**Suite de testes final: 161 passed, 2 skipped, 0 failed.**

---

## 6. Artefato de backtest

Salvo em: `data/backtest_artifacts/backtest_PRIO3_LREN3_SUZB3_EQTL3_+4_20260628_213847.json`

Conteúdo: timestamp, git_commit, seed, period_start/end, tickers, params por ticker, métricas por ticker e agregadas, hash do DB.

**Este é o primeiro resultado de backtest auditável e reproduzível deste pipeline.**
