"""
config.py
=========
Central configuration for the Systematic Alpha Research Pipeline.
All parameters are defined here to ensure reproducibility and easy tuning.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
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
    "SPY", "QQQ", "IWM",   # US Equity ETFs
    "EFA", "EEM",            # Internacional
    "TLT", "IEF",            # Bonds (inversamente correlacionados)
    "GLD",                   # Ouro (safe haven)
    "UUP",                   # Dolar americano
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
INTRADAY_SESSION_START = "09:30"   # ET — earliest entry time
INTRADAY_SESSION_END   = "15:45"   # ET — latest entry time (avoid close)
INTRADAY_PERIOD_MAP = {            # max free history per interval
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",    # yfinance allows up to 730 days for 1h
    "1d": "max",
}

TIMEZONE = "America/New_York"

# Reference instruments (downloaded alongside tickers)
VIX_TICKER = "^VIX"   # CBOE Volatility Index — market fear gauge
SPY_TICKER = "SPY"    # S&P 500 ETF — used for beta and correlation features

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
# Flipped ratio: TP = 1.0×ATR, SL = 1.5×ATR
# With TP < SL distance, TP is easier to reach -> more +1 labels -> better balance
USE_ATR_BARRIERS    = True
ATR_BARRIER_MULT_TP = 1.0        # 1.0x ATR take-profit (easier to reach)
ATR_BARRIER_MULT_SL = 1.5        # 1.5x ATR stop-loss   (wider stop = fewer stops)

# Fixed-% fallback (used when USE_ATR_BARRIERS=False or ATR unavailable)
TAKE_PROFIT_PCT = 0.010
STOP_LOSS_PCT   = 0.007

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
OPTUNA_TRIALS = 20              # trials per fold (balance quality vs speed)

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
INITIAL_EQUITY   = 100_000.0

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
MIN_PROBA_THRESHOLD = 0.48   # calibrado para barras diarias com modelo especializado

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
WFV_EXPANDING = False   # False = rolling window (default) | True = expanding/anchored

# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------
# Set via environment variables — never hardcode secrets
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Alpaca Markets — paper / live trading
# ---------------------------------------------------------------------------
ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_PAPER  = True          # True = paper trading endpoint (safe default)
