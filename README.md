# Systematic Alpha

A professional-grade ML-driven quantitative trading research pipeline that discovers and validates alpha signals through rigorous walk-forward validation, realistic backtesting, and institutional-grade risk management.

---

## Overview

**Systematic Alpha** combines machine learning (LightGBM), quantitative feature engineering, and event-driven backtesting to systematically research, validate, and trade multi-ticker alpha strategies. The system is designed with production-quality standards: no look-ahead bias, realistic execution costs, and robust out-of-sample validation.

**Key capabilities:**
- 24+ quantitative features (technical, macro, seasonality, market structure)
- Triple-Barrier labeling with adaptive ATR-based barriers
- Walk-forward validation with per-fold IC tracking
- Event-driven backtester with slippage, commission, spread, and market impact
- Institutional metrics: Sharpe (with CI), Sortino, Omega, Calmar, IC/ICIR
- Multi-ticker portfolio allocation (risk parity, min-variance, max-Sharpe)
- GMM/HMM regime detection and conditional performance analysis
- Live signal generation and Alpaca broker integration
- MLflow experiment tracking and Telegram notifications

---

## Architecture

```
systematic_alpha/
├── main.py                  # Orchestrator — 5 execution modes
├── config.py                # Central configuration (all parameters)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage Docker build
├── .github/workflows/ci.yml # CI/CD pipeline
│
├── data_pipeline.py         # yfinance download + SQLite storage
├── feature_engineering.py   # 24+ quantitative features
├── labeling.py              # Triple-Barrier method (-1/0/+1)
├── model_training.py        # LightGBM + Optuna HPO
├── walk_forward.py          # Walk-forward validation + MLflow
│
├── backtest_engine.py       # Event-driven backtester
├── execution_engine.py      # Signal generation + Alpaca integration
├── position_manager.py      # Multi-position state tracking
├── risk_management.py       # Kelly sizing + RiskGuard
├── market_impact.py         # Almgren-Chriss impact model
│
├── performance.py           # Institutional metrics
├── portfolio_manager.py     # Multi-ticker allocation
├── regime_detection.py      # GMM/HMM regime classification
├── factor_analysis.py       # IC, ICIR, signal decay, attribution
├── alternative_data.py      # FRED macro data
│
├── notifier.py              # Telegram Bot alerts
├── scheduler.py             # Daily automation
│
├── tests/                   # 10 pytest modules (~95% coverage)
│
├── data/                    # SQLite OHLCV storage
├── models/                  # Saved LightGBM models
├── logs/                    # Equity curves, signals CSV, pipeline log
└── mlruns/                  # MLflow experiment tracking
```

---

## Pipeline Stages

The research pipeline runs 9 sequential stages per ticker:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_pipeline` | Download OHLCV bars (yfinance), store in SQLite incrementally |
| 2 | `feature_engineering` | Build 24 causally-correct quantitative features |
| 3 | `labeling` | Assign Triple-Barrier labels (-1/0/+1) with ATR-based barriers |
| 4 | `walk_forward` | Rolling 24m train / 6m test WFV with IC tracking |
| 5 | `backtest_engine` | Event-driven simulation with realistic execution costs |
| 6 | `performance` | Institutional metrics + equity curve plot |
| 7 | `factor_analysis` | IC, ICIR, signal decay, factor attribution |
| 8 | `regime_detection` | GMM regime classification + conditional performance |
| 9 | `model_training` | Train final model on all data + save to disk |

---

## Installation

### Requirements

- Python 3.11+
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/systematic-alpha.git
cd systematic-alpha

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file (never commit this):

```env
# Alpaca Markets (paper trading or live)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live URL

# Telegram notifications (optional)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

All secrets are loaded from environment variables — never hardcoded.

---

## Usage

### Execution Modes

```bash
# Update market data for all configured tickers
python main.py --mode update

# Full research pipeline (features → labels → WFV → backtest → analysis)
python main.py --mode research --tickers SPY QQQ IWM

# Single ticker research
python main.py --mode research --ticker SPY

# Multi-ticker portfolio analysis
python main.py --mode portfolio

# Train and save final model for a ticker
python main.py --mode train --ticker SPY

# Generate live signals for execution
python main.py --mode live --ticker SPY
```

### Additional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Single ticker symbol |
| `--tickers` | config list | Space-separated list of tickers |
| `--equity` | `100000` | Starting capital for backtest |
| `--expanding` | `False` | Use expanding window WFV instead of rolling |
| `--verbose / -v` | `False` | Enable DEBUG-level logging |

### Examples

```bash
# Research with $50k capital, expanding window, verbose output
python main.py --mode research --ticker QQQ --equity 50000 --expanding --verbose

# Live signals for multiple tickers
python main.py --mode live --tickers SPY QQQ GLD

# Portfolio analysis with custom universe
python main.py --mode portfolio
```

---

## Configuration

All parameters are centralized in [config.py](config.py):

```python
# Ticker universe (equities, bonds, commodities, currencies)
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "UUP"]

# Data resolution
INTERVAL = "1d"   # daily bars (also: "1h", "5m", "1m")

# Walk-forward parameters
TRAIN_MONTHS = 24
TEST_MONTHS  = 6
EMBARGO_DAYS = 3   # prevents look-ahead leakage

# Risk parameters
RISK_PER_TRADE    = 0.01   # 1% of equity per trade
DAILY_STOP_LOSS   = 0.03   # 3% daily drawdown kill-switch
MAX_DRAWDOWN      = 0.10   # 10% drawdown kill-switch

# Execution costs
SLIPPAGE_BPS  = 5    # 5 basis points per side
COMMISSION    = 1.0  # $1 per trade
SPREAD_BPS    = 2    # 2 basis points bid-ask spread
BAR_DELAY     = 1    # 1-bar execution delay

# Signal threshold for live trading
LIVE_PROB_THRESHOLD = 0.48
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | pandas 2.1+, numpy 1.26+, yfinance |
| ML | LightGBM 4.3+, scikit-learn 1.4+ |
| HPO | Optuna 3.6+ (Bayesian, 20 trials/fold) |
| Experiment Tracking | MLflow 2.12+ |
| Backtesting | Custom event-driven engine |
| Broker | Alpaca Markets (alpaca-py 0.20+) |
| Notifications | Telegram Bot API |
| Scheduling | schedule 1.2+ |
| Visualization | matplotlib 3.8+ |
| Containerization | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Linting | ruff |
| Security | bandit |
| Testing | pytest + pytest-cov |

All dependencies are free and open-source.

---

## Features

### Quantitative Features (24+)

- **Returns**: Log returns (lags 1-5), multi-window momentum, 12-1 month momentum
- **Technical**: RSI-14, MACD (12/26/9), ATR-14, VWAP deviation, MA200 distance
- **Volatility**: Rolling vol (5/21d), Garman-Klass realized volatility
- **Volume**: Volume spike ratio, Amihud illiquidity proxy
- **Market Structure**: Order imbalance, 52-week high proximity, breakout flags
- **Macro/Regime**: VIX level, rolling Beta vs SPY, FRED yield curve
- **Seasonality**: Day-of-week, month-of-year, earnings proximity
- **Z-scores**: Price and volume normalization across rolling windows

### Labeling

Triple-Barrier method with adaptive ATR-based barriers:

- **Take Profit**: 1.0× ATR (or 1.0% fixed fallback)
- **Stop Loss**: 1.5× ATR (or 0.7% fixed fallback)
- **Max Hold**: 3 bars
- **Labels**: +1 (TP hit first), -1 (SL hit first), 0 (timeout)

### Backtesting Realism

- 5 bps slippage per side
- $1 commission per trade
- 2 bps bid-ask spread
- 1-bar execution delay
- Almgren-Chriss square-root market impact model
- Trend filter + volatility regime gating
- Daily stop loss and max drawdown kill-switches
- Cooldown between consecutive trades

### Portfolio Allocation Methods

- Equal weight
- Risk parity (inverse volatility)
- Minimum variance (quadratic optimization)
- Maximum Sharpe ratio (mean-variance optimization)

---

## Output Artifacts

| Path | Description |
|------|-------------|
| `data/market_data.db` | SQLite OHLCV database (all tickers, all intervals) |
| `models/model_final_<ticker>.pkl` | Trained LightGBM model for live trading |
| `logs/equity_curve_<ticker>.png` | Backtest equity curve with drawdown panel |
| `logs/signals_<ticker>.csv` | Out-of-sample signal predictions |
| `logs/pipeline.log` | Full execution log (DEBUG/INFO) |
| `signals_output.csv` | Latest live signals ready for execution |
| `mlruns/` | MLflow experiment runs (metrics, params, artifacts) |

---

## MLflow Experiment Tracking

View experiment results in the MLflow UI:

```bash
mlflow ui
# Open http://localhost:5000
```

Logged per fold: IC, accuracy, loss, model params, feature importances.

---

## Automation (Scheduler)

```bash
# Runs continuous daily automation in background
python scheduler.py
```

| Time (ET) | Job |
|-----------|-----|
| 08:30 | Incremental data update for all tickers |
| 09:40 | Live signal scan + Telegram notification |
| Saturday 07:00 | Weekly model retraining |

---

## Docker

```bash
# Build image
docker build -t systematic-alpha .

# Run research pipeline
docker run --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    systematic-alpha \
    python main.py --mode research --tickers SPY QQQ

# Run live signal scan
docker run --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    systematic-alpha \
    python main.py --mode live --ticker SPY
```

The Docker image uses a multi-stage build and runs as a non-root user.

---

## Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=. -v

# Run with timeout (recommended for CI)
pytest tests/ --cov=. --timeout=120

# Run specific test module
pytest tests/test_backtest_engine.py -v

# Skip slow integration tests
pytest tests/ -v --ignore=tests/test_sanity.py
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_data_pipeline.py` | DB operations, incremental updates, yfinance mocking |
| `test_feature_engineering.py` | All 24 features, look-ahead validation, edge cases |
| `test_labeling.py` | Triple-Barrier, ATR vs fixed %, label distribution |
| `test_model_training.py` | LightGBM fit, Optuna integration, feature selection |
| `test_backtest_engine.py` | Trade execution, P&L, slippage, commission |
| `test_execution_engine.py` | Signal generation, probability filtering |
| `test_portfolio_manager.py` | Allocation methods, correlation blocking |
| `test_risk_management.py` | Position sizing, Kelly, RiskGuard kill-switches |
| `test_performance.py` | Sharpe annualization (daily/intraday), drawdown |
| `conftest.py` | Shared fixtures (synthetic OHLCV, models) |

---

## CI/CD

GitHub Actions pipeline runs on every push and pull request:

1. **Lint** — ruff (E, W, F, I rules)
2. **Tests** — pytest with coverage, 120s timeout
3. **Docker Build** — validates multi-stage Dockerfile
4. **Security Scan** — bandit static analysis

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Past backtest performance does not guarantee future results. Algorithmic trading involves significant financial risk. Use paper trading accounts before deploying any real capital.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
