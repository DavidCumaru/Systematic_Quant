"""
settings.py
===========
Central configuration for the Systematic Alpha Research Pipeline.
All parameters are defined here to ensure reproducibility and easy tuning.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR     = BASE_DIR / "data"
MODELS_DIR   = BASE_DIR / "models"
LOGS_DIR     = BASE_DIR / "logs"
TESTS_DIR    = BASE_DIR / "tests"
DB_PATH      = DATA_DIR / "market_data.db"
SIGNALS_PATH = BASE_DIR / "signals_output.csv"

for _dir in (DATA_DIR, MODELS_DIR, LOGS_DIR, TESTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TICKERS  = [
    # ETFs de índice (benchmark e small caps)
    "BOVA11.SA", "SMAL11.SA", "IVVB11.SA",

    # Energia / Petróleo
    "PETR4.SA",  "PRIO3.SA",  "CSAN3.SA",  "UGPA3.SA",

    # Mineração / Siderurgia
    "VALE3.SA",  "GGBR4.SA",  "CSNA3.SA",  "SUZB3.SA",  "KLBN11.SA",

    # Setor Financeiro
    "ITUB4.SA",  "BBAS3.SA",

    # Alimentos / Bebidas / Proteína
    "ABEV3.SA",  "BEEF3.SA",

    # Varejo / Consumo
    "LREN3.SA",  "MGLU3.SA",  "RENT3.SA",  "RADL3.SA",

    # Telecom
    "VIVT3.SA",  "TIMS3.SA",

    # Industrial / Tecnologia
    "WEGE3.SA",  "TOTS3.SA",

    # Utilities / Infraestrutura / Saneamento
    "EQTL3.SA",  "SBSP3.SA",

    # Imobiliário / Construção
    "CYRE3.SA",  "MRVE3.SA",

    # Renda Fixa / Proteção
    "IMAB11.SA", "GOLD11.SA",
]
# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------
# Daily (default): free via yfinance, up to 10+ years of history
# Intraday:        yfinance provides ~60 days for 1h, ~7 days for 5m/1m
#
# Set INTERVAL to one of:
#   "1d"  — daily bars (default, longest history)
#   "1h"  — hourly bars (~60 days of history via yfinance)
#   "5m"  — 5-minute bars (~60 days of history via yfinance)
#   "1m"  — 1-minute bars (~7 days of history via yfinance)
#
# When using intraday bars:
#   - PERIOD should be set to "60d" (1h) or "7d" (1m/5m)
#   - SESSION_FILTER in backtest_engine should be True
#   - Features gain extra intraday context (VWAP, session time, volume patterns)
#
INTERVAL = "1d"
PERIOD   = "max"      # for daily; set "60d" for 1h, "7d" for 5m/1m

# Intraday-specific settings (active only when INTERVAL != "1d")
INTRADAY_SESSION_START = "10:05"   # BRT — abertura B3 + buffer leilão
INTRADAY_SESSION_END   = "16:45"   # BRT — antes do fechamento B3 (17:30)
INTRADAY_PERIOD_MAP = {            # max free history per interval
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",    # yfinance allows up to 730 days for 1h
    "1d": "max",
}

TIMEZONE = "America/Sao_Paulo"

# Reference instruments (downloaded alongside tickers)
VIX_TICKER    = "^VIX"      # CBOE Volatility Index — indicador global de medo
BENCHMARK_TICKER = "BOVA11.SA"  # IBOVESPA ETF — usado para beta e correlação (substitui SPY)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RSI_WINDOW       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
ATR_WINDOW       = 14
VWAP_WINDOW      = 20
VOL_WINDOW       = 20
VOLUME_SPIKE_Z   = 2.0
MOMENTUM_WINDOWS = [5, 10, 20]   # 5d, 10d, 20d (daily bars)
ZSCORE_WINDOW    = 20

# Relative volume: volume / rolling-mean-volume
REL_VOL_WINDOW   = 20

# Beta vs SPY: rolling window for beta computation
BETA_WINDOW      = 20

# Regime: rolling daily vol above this -> skip new entries
REGIME_VOL_WINDOW     = 20
REGIME_VOL_THRESHOLD  = 0.030    # 3.0% per daily bar -> only extreme crash regimes blocked

# ---------------------------------------------------------------------------
# Trend filter (Phase 1 quick win)
# ---------------------------------------------------------------------------
# Only take LONG signals when price > MA(TREND_MA_BARS)
# Only take SHORT signals when price < MA(TREND_MA_BARS)
# Eliminates the dominant source of losses: shorting during bull trends
USE_TREND_FILTER = True
TREND_MA_BARS    = 200          # MA200 diaria — filtro de tendencia classico

# ---------------------------------------------------------------------------
# Labeling (Triple-Barrier)
# ---------------------------------------------------------------------------
# Barreiras globais (fallback). Params por ticker via ticker_params.json.
# Backtest auditavel: data/backtest_artifacts/backtest_PRIO3_LREN3_SUZB3_EQTL3_+4_20260628_213847.json
USE_ATR_BARRIERS    = False
ATR_BARRIER_MULT_TP = 2.0        # nao usado (USE_ATR_BARRIERS=False)
ATR_BARRIER_MULT_SL = 1.0        # nao usado (USE_ATR_BARRIERS=False)

# Fixed-% barriers (ativos quando USE_ATR_BARRIERS=False)
TAKE_PROFIT_PCT = 0.030   # 3.0% take-profit (R:R 3:1)
STOP_LOSS_PCT   = 0.010   # 1.0% stop-loss

TIME_STOP_BARS  = 3              # 3 dias max hold (swing trade)

# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
TRAIN_MONTHS  = 24               # 2 anos de treino
TEST_MONTHS   = 6                # 6 meses de teste
EMBARGO_BARS  = TIME_STOP_BARS   # = 5 dias purged from end of each train fold

# ---------------------------------------------------------------------------
# Model — LightGBM (replaces sklearn GradientBoosting)
# ---------------------------------------------------------------------------
# "lightgbm"            — single 3-class LightGBM (default)
# "lightgbm_specialized" — two binary LightGBMs (long-model + short-model)
# "random_forest"       — sklearn RF (legacy fallback)
# "gradient_boosting"   — sklearn GBM (legacy fallback)
MODEL_TYPE       = "lightgbm_specialized"
RANDOM_SEED      = 42
N_ESTIMATORS     = 300
MAX_DEPTH        = -1            # unlimited depth — controlled via NUM_LEAVES
NUM_LEAVES       = 31            # LightGBM primary complexity control
LEARNING_RATE    = 0.05
MIN_SAMPLES_LEAF = 20
CLASS_WEIGHT     = "balanced"

# Feature selection: drop features whose importance < this threshold
MIN_FEATURE_IMPORTANCE = 0.005

# ---------------------------------------------------------------------------
# Optuna — Bayesian hyperparameter tuning
# ---------------------------------------------------------------------------
USE_OPTUNA    = True
OPTUNA_TRIALS = 15              # trials por fold — reduzido de 20 para limitar overfitting de hiperparâmetros

# ---------------------------------------------------------------------------
# MLflow — experiment tracking
# ---------------------------------------------------------------------------
USE_MLFLOW         = True
MLFLOW_EXPERIMENT  = "systematic_alpha"
# Use file:/// URI — required on Windows paths with spaces/non-ASCII chars
MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()  # e.g. file:///C:/Users/...

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
RISK_PER_TRADE   = 0.01      # 1% of equity per trade
DAILY_STOP_PCT   = 0.03      # 3% daily loss -> kill switch
MAX_DRAWDOWN_PCT = 0.10      # 10% portfolio drawdown -> halt
INITIAL_EQUITY   = float(os.environ.get("INITIAL_EQUITY", "8000"))

# Kelly criterion position sizing
USE_KELLY       = True
KELLY_FRACTION  = 0.25       # fractional Kelly (25%) — conservative
KELLY_WARMUP    = 20         # minimum trades before activating Kelly

# ---------------------------------------------------------------------------
# Backtest / execution simulation
# ---------------------------------------------------------------------------
USE_SESSION_FILTER   = False    # False para barras diarias (sem sessao intraday)
SLIPPAGE_PCT         = 0.0005   # 5 bps per side
COMMISSION_PER_TRADE = 1.0      # USD per trade (flat)
SPREAD_PCT           = 0.0002   # 2 bps spread
EXECUTION_DELAY_BARS = 1        # bars delayed before fill

# ---------------------------------------------------------------------------
# Signal filter
# ---------------------------------------------------------------------------
MIN_PROBA_THRESHOLD = 0.58   # minimo de confianca para gerar sinal (elevado de 0.52 apos analise semana 14/04)

# Gap protection — cancela ordem se o preco atual se afastou demais do fechamento
# Para BUY:  se preco subiu > MAX_ENTRY_GAP_PCT -> R/R comprometido -> cancela
# Para SELL: se preco caiu  > MAX_ENTRY_GAP_PCT -> R/R comprometido -> cancela
MAX_ENTRY_GAP_PCT = 0.015    # 1.5% de gap maximo aceito antes de cancelar

# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------
REGIME_METHOD   = "gmm"   # "gmm" (default, no extra deps) | "hmm" (pip install hmmlearn)
REGIME_N_STATES = 3       # Bear=0 / Sideways=1 / Bull=2

# ---------------------------------------------------------------------------
# Factor Analysis
# ---------------------------------------------------------------------------
IC_HORIZONS     = [1, 2, 3, 5, 10]   # forward-return horizons for IC decay curve
ICIR_MIN_VIABLE = 0.50                # ICIR below this = signal too noisy for trading

# ---------------------------------------------------------------------------
# Portfolio Management
# ---------------------------------------------------------------------------
PORTFOLIO_METHOD       = "risk_parity"   # "equal" | "risk_parity" | "min_variance" | "max_sharpe"
MAX_POSITION_PCT       = 0.25            # hard cap: max 25% of equity per ticker
MAX_CORRELATION_FILTER = 0.80            # log warning for pairs above this correlation

# ---------------------------------------------------------------------------
# Walk-Forward — extra options
# ---------------------------------------------------------------------------
WFV_EXPANDING   = False   # False = rolling window (default) | True = expanding/anchored
HOLDOUT_MONTHS  = 12      # Últimos N meses reservados como holdout final
                          # O WFV e o grid search nunca veem esses dados
                          # Isso reduz overfitting de segundo nível (tuning em OOS)

# ---------------------------------------------------------------------------
# Paper Broker — local simulation (no API key required)
# ---------------------------------------------------------------------------
PAPER_BROKER_PATH = DATA_DIR / "paper_broker.json"   # persistent state file

# ---------------------------------------------------------------------------
# Broker — execução de ordens
# ---------------------------------------------------------------------------
# BROKER_MODE = "paper"    → simulação local (padrão, seguro)
# BROKER_MODE = "mt5"      → MetaTrader 5, dinheiro real
# BROKER_MODE = "mt5_dry"  → MT5 conectado mas sem enviar ordens (para testar)
#
# Para usar MT5:
#   1. Instale: pip install MetaTrader5
#   2. Instale o MetaTrader 5 no Windows e faça login na corretora
#   3. Preencha MT5_LOGIN, MT5_PASSWORD, MT5_SERVER abaixo
#   4. Mude BROKER_MODE para "mt5_dry" primeiro para testar a conexão
#   5. Mude para "mt5" só quando estiver pronto para operar com dinheiro real
# ---------------------------------------------------------------------------
BROKER_MODE  = os.environ.get("BROKER_MODE",  "mt5_dry")
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "0"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "MetaQuotes-Demo")

# ---------------------------------------------------------------------------
# Live Trading — tickers selecionados pelo holdout backtest (melhores 8)
# ---------------------------------------------------------------------------
LIVE_TICKERS = [
    "PRIO3.SA",   # PetroRio      — maior lucro individual no holdout
    "LREN3.SA",   # Lojas Renner  — consistência alta
    "SUZB3.SA",   # Suzano        — bom sinal de tendência
    "EQTL3.SA",   # Equatorial    — utilitária, baixa correlação
    "ITUB4.SA",   # Itaú          — financeiro liquido
    "VIVT3.SA",   # Telefônica    — maior taxa de acerto (41,5%)
    "KLBN11.SA",  # Klabin        — papel e celulose, descorrelacionado
    "BBAS3.SA",   # Banco do Brasil
]

# ---------------------------------------------------------------------------
# Notificações Telegram
# ---------------------------------------------------------------------------
# Como configurar:
#   1. Abra o Telegram e procure @BotFather
#   2. Digite /newbot e siga as instruções para criar um bot
#   3. Copie o token gerado (ex: "123456:ABC-DEF...")
#   4. Abra seu bot no Telegram e envie qualquer mensagem
#   5. Acesse: https://api.telegram.org/bot<TOKEN>/getUpdates
#   6. Copie o "chat_id" que aparece no JSON
#   7. Cole token e chat_id abaixo
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
