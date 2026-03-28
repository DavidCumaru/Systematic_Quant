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
- Institutional metrics: Sharpe (with 95% CI bootstrap), Sortino, Omega, Calmar, IC/ICIR
- Multi-ticker portfolio allocation (risk parity, min-variance, max-Sharpe)
- GMM/HMM regime detection and conditional performance analysis
- Per-ticker parameter optimization via exhaustive grid search (2,688 combinations/ticker)
- Local paper broker — zero API key required, JSON-persisted state
- Interactive Streamlit dashboard with live monitoring
- MLflow experiment tracking

---

## Backtested Results (Walk-Forward Validation, 2004–2026)

> Strategy: LightGBM long/short classifier, Triple-Barrier labels, 24-month rolling train / 6-month test, 10 folds.
> Capital: USD 100,000. Execution costs: 5 bps slippage + USD 1 commission + 2 bps spread per trade.
> Parameters per ticker optimized via exhaustive grid search (Sharpe-ranked, min. 10 trades).

### Per-Ticker Results (optimized params)

| Ticker | Total Return | CAGR | Sharpe | Sharpe 95% CI | Sortino | Omega | Max DD | Win Rate | Profit Factor | Trades |
|--------|-------------|------|--------|---------------|---------|-------|--------|----------|---------------|--------|
| SPY | +11.28% | 0.32% | 0.52 | [0.23, 0.78] | 1.54 | 1.71 | 2.68% | 33.2% | 1.71 | 202 |
| QQQ | +8.76% | 0.31% | 0.65 | [0.32, 0.95] | 1.95 | 1.99 | 0.79% | 46.1% | 1.99 | 141 |
| IWM | +1.57% | 0.06% | 0.15 | [-0.26, 0.47] | 15.34 | 1.26 | 1.38% | 25.4% | 1.26 | 71 |
| TLT | +4.22% | 0.18% | 0.28 | [-0.12, 0.65] | 0.10 | 1.35 | 3.01% | 54.1% | 1.34 | 109 |
| GLD | +6.88% | 0.31% | 0.60 | [0.21, 0.93] | 1.20 | 1.75 | 1.19% | 43.0% | 1.75 | 142 |

### Portfolio-Level Summary

| Metric | Value |
|--------|-------|
| Portfolio Return | +11.28% |
| Portfolio Sharpe | 0.52 |
| Portfolio Max Drawdown | 2.68% |
| Mean Ticker Sharpe | 0.44 |
| Best Ticker Sharpe | 0.65 (QQQ) |
| Mean Win Rate | 40.3% |
| Mean Profit Factor | 1.61 |
| Avg Hold Period | ~2–5 bars |

### Optimized Parameters per Ticker

| Ticker | Min Conf | Stop Loss | Take Profit | Time Stop | Direction | Regime Filter |
|--------|----------|-----------|-------------|-----------|-----------|---------------|
| SPY | 56% | 0.5% | 2.5% | 5 bars | Both | All |
| QQQ | 52% | 0.5% | 1.8% | 2 bars | Both | All |
| IWM | 60% | 0.5% | 2.5% | 5 bars | Long Only | All |
| TLT | 52% | 1.5% | 2.5% | 5 bars | Both | All |
| GLD | 48% | 0.5% | 1.8% | 2 bars | Long Only | All |

### 2025 Simulation (R$ 1.000 / USD ~180)

> Applying optimized models and params from Jan/2025 to Mar/2026.
> Note: R$ 1.000 = USD ~180 at BRL/USD 5.55 — results are proportionally scaled from full backtest.
> WFV test folds cover Jul/2025–Mar/2026; results limited to periods with out-of-sample signals.

| Ticker | Trades | Win Rate | P&L BRL |
|--------|--------|----------|---------|
| SPY | 4 | 0% | -R$ 3,92 |
| QQQ | 0 | — | — |
| IWM | 4 | 50% | +R$ 7,46 |
| TLT | 8 | 38% | -R$ 5,79 |
| GLD | 11 | 45% | +R$ 9,28 |
| **TOTAL** | **27** | **41%** | **+R$ 7,03** |

**Capital inicial: R$ 1.000,00 → Capital final: R$ 1.007,03 (+0.70%)**

> GLD foi o melhor ticker — tendência de alta do ouro no 2S2025 favoreceu a estratégia long-only.
> SPY sofreu 4 stops seguidos em fev/mar 2026 (queda pós-tarifas). IWM teve 2 TPs em jul e nov/2025.

---

## Database

- **Source:** Yahoo Finance (yfinance), daily bars
- **Storage:** SQLite WAL-mode (`data/market_data.db`)
- **Universe:** SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, UUP (9 tickers)
- **Total bars:** ~55,556 daily records
- **Date range:** ~1993–2026 (varies by ticker)
- **VIX:** CBOE Volatility Index (macro feature)

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
├── model_training.py        # LightGBM + Optuna HPO (specialized Long/Short)
├── walk_forward.py          # Walk-forward validation + MLflow
│
├── backtest_engine.py       # Event-driven backtester (per-ticker params)
├── execution_engine.py      # Signal generation + paper broker integration
├── paper_broker.py          # Local paper broker (JSON state, yfinance fills)
├── position_manager.py      # Multi-position state tracking
├── risk_management.py       # Kelly sizing + RiskGuard kill-switches
├── market_impact.py         # Almgren-Chriss square-root impact model
│
├── performance.py           # Institutional metrics + equity curve
├── portfolio_manager.py     # Multi-ticker allocation (risk parity etc.)
├── regime_detection.py      # GMM/HMM regime classification
├── factor_analysis.py       # IC, ICIR, signal decay, factor attribution
├── alternative_data.py      # FRED macro data
│
├── ticker_config.py         # Per-ticker parameter storage (JSON)
├── grid_search.py           # Exhaustive grid search (2,688 combos/ticker)
├── dashboard.py             # Streamlit dashboard (5 tabs, live monitoring)
├── notifier.py              # Log-based notifier (no API keys required)
├── scheduler.py             # Daily automation
│
├── tests/                   # 10 pytest modules
│
├── data/
│   ├── market_data.db       # SQLite OHLCV database
│   ├── paper_broker.json    # Paper broker persisted state
│   └── ticker_params.json   # Per-ticker optimized parameters
├── models/                  # Saved LightGBM models (.pkl)
├── logs/                    # Equity curves, signals CSV, pipeline log, grid_search.csv
└── mlruns/                  # MLflow experiment tracking
```

---

## Pipeline Stages

The research pipeline runs 10 sequential stages per ticker:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_pipeline` | Download OHLCV bars (yfinance), store in SQLite incrementally |
| 2 | `feature_engineering` | Build 24+ causally-correct quantitative features |
| 3 | `labeling` | Assign Triple-Barrier labels (-1/0/+1) with ATR-based barriers |
| 4 | `walk_forward` | Rolling 24m train / 6m test WFV — 10 folds, IC tracking |
| 5 | `backtest_engine` | Event-driven simulation with per-ticker params + regime filter |
| 6 | `performance` | Institutional metrics + bootstrap Sharpe CI + equity curve |
| 7 | `factor_analysis` | IC, ICIR, signal decay, factor attribution, turnover cost |
| 8 | `regime_detection` | GMM regime classification + conditional performance breakdown |
| 9 | `model_training` | Train final Long/Short specialized models on all data |
| 10 | `execution_engine` | Export signals CSV + paper broker order routing |

---

## Installation

### Requirements

- Python 3.11+
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/DavidCumaru/Systematic_Quant.git
cd Systematic_Quant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

No API keys required. All data via yfinance (free).

---

## Usage

### Execution Modes

```bash
# Update market data for all configured tickers
python main.py --mode update

# Full research pipeline (features -> labels -> WFV -> backtest -> analysis)
python main.py --mode research --tickers SPY QQQ IWM TLT GLD

# Research with per-ticker optimized parameters
python main.py --mode research --use-ticker-params --tickers SPY QQQ IWM TLT GLD

# Multi-ticker portfolio analysis
python main.py --mode portfolio

# Train and save final model for a ticker
python main.py --mode train --ticker GLD

# Generate live signals (routes to paper broker)
python main.py --mode live --ticker SPY
```

### Per-Ticker Grid Search (Parameter Optimization)

```bash
# Run exhaustive grid search for all tickers (~1h total)
python grid_search.py --tickers SPY QQQ IWM TLT GLD

# Grid search for specific tickers, show top 3 configs
python grid_search.py --tickers GLD QQQ --top 3

# Results saved automatically to:
#   data/ticker_params.json   (best params per ticker)
#   logs/grid_search.csv      (full 13,440 results)
```

**Grid search space (2,688 combinations per ticker):**

| Parameter | Values |
|-----------|--------|
| `min_proba_threshold` | 0.48, 0.52, 0.56, 0.60 |
| `stop_loss_pct` | 0.5%, 0.7%, 1.0%, 1.5% |
| `take_profit_pct` | 0.8%, 1.2%, 1.8%, 2.5% |
| `time_stop_bars` | 2, 3, 5 |
| `direction` | both, long_only |
| `regime_filter` | all, Bull, Bear, Sideways, Bull+Sideways, Bear+Sideways, Bear+Bull |

### Streamlit Dashboard

```bash
# Launch interactive dashboard
venv/Scripts/streamlit.exe run dashboard.py   # Windows
streamlit run dashboard.py                     # Linux/macOS

# Open http://localhost:8501
```

**Dashboard tabs:**

| Tab | Content |
|-----|---------|
| Overview | Portfolio KPIs, equity, total P&L, win rate, open positions |
| Signals | Signal feed with direction/ticker/confidence filters |
| Paper Broker | Open positions, closed trades, cash balance |
| Performance | Equity curves per ticker (PNG), monthly returns |
| Factor & Regime | IC/ICIR table, signal grade, regime performance breakdown |

### Additional CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Single ticker symbol |
| `--tickers` | config list | Space-separated list of tickers |
| `--equity` | `100000` | Starting capital for backtest |
| `--expanding` | `False` | Use expanding window WFV instead of rolling |
| `--use-ticker-params` | `False` | Load per-ticker optimized params from `data/ticker_params.json` |
| `--verbose / -v` | `False` | Enable DEBUG-level logging |

---

## Configuration

All parameters are centralized in [config.py](config.py):

```python
# Ticker universe
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "UUP"]

# Walk-forward parameters
TRAIN_MONTHS = 24
TEST_MONTHS  = 6
EMBARGO_DAYS = 3   # prevents look-ahead leakage

# Default execution params (overridden per-ticker after grid search)
MIN_PROBA_THRESHOLD = 0.52
STOP_LOSS_PCT       = 0.007
TAKE_PROFIT_PCT     = 0.012
TIME_STOP_BARS      = 3

# Risk parameters
RISK_PER_TRADE    = 0.01   # 1% of equity per trade
DAILY_STOP_LOSS   = 0.03   # 3% daily drawdown kill-switch
MAX_DRAWDOWN      = 0.10   # 10% drawdown kill-switch

# Execution costs
SLIPPAGE_BPS  = 5    # 5 basis points per side
COMMISSION    = 1.0  # $1 per trade
SPREAD_BPS    = 2    # 2 basis points bid-ask spread
BAR_DELAY     = 1    # 1-bar execution delay
```

Per-ticker overrides are stored in `data/ticker_params.json` (generated by `grid_search.py`).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | pandas 2.1+, numpy 1.26+, yfinance |
| ML | LightGBM 4.3+ (specialized Long/Short models) |
| HPO | Optuna 3.6+ (Bayesian, 20 trials/fold) |
| Experiment Tracking | MLflow 2.12+ |
| Backtesting | Custom event-driven engine (per-ticker params) |
| Param Optimization | Custom exhaustive grid search (2,688 combos/ticker) |
| Paper Broker | Local JSON broker (yfinance next-bar fills) |
| Dashboard | Streamlit 1.35+ + Plotly 5.20+ |
| Notifications | Python logging (no external API required) |
| Scheduling | schedule 1.2+ |
| Visualization | matplotlib 3.8+, Plotly 5.20+, Pillow 10+ |
| Containerization | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Linting | ruff |
| Security | bandit |
| Testing | pytest + pytest-cov |

All dependencies are **free and open-source**. No paid API keys required.

---

## Features

### Quantitative Features (24+)

- **Returns**: Log returns (lags 1–5), multi-window momentum (5/10/20), 12-1 month momentum
- **Technical**: RSI-14, MACD (12/26/9), ATR-14, VWAP deviation, MA200 distance
- **Volatility**: Rolling vol (5/21d), Garman-Klass realized volatility
- **Volume**: Volume spike ratio, Amihud illiquidity proxy
- **Market Structure**: Order imbalance, 52-week high proximity, breakout flags
- **Macro/Regime**: VIX level, rolling Beta vs SPY, FRED yield curve
- **Seasonality**: Day-of-week, month-of-year, earnings proximity
- **Z-scores**: Price and volume normalization across rolling windows

### Model: Specialized Long/Short Classifiers

- Two separate LightGBM binary classifiers trained per fold: one for LONG signals, one for SHORT
- Optuna Bayesian HPO: 20 trials per model per fold
- Feature selection: importance threshold 0.005 (reduces ~38 → ~28–34 active features)
- Labels: Triple-Barrier method — +1 (TP hit), -1 (SL hit), 0 (timeout)

### Walk-Forward Validation

- Rolling window: 24-month train / 6-month test
- Embargo: 3 bars between train and test to prevent leakage
- 10 folds covering ~2006–2026
- Per-fold metrics: accuracy, F1-macro, IC (Information Coefficient)
- Mean IC = 0.04 (GLD), ICIR = 0.33–0.85 depending on ticker/horizon

### Backtesting Realism

- 5 bps slippage per side
- $1 commission per trade
- 2 bps bid-ask spread
- 1-bar execution delay
- Almgren-Chriss square-root market impact model
- Direction filter (long_only / short_only / both) per ticker
- Regime filter (skip Bull/Bear/Sideways regimes per ticker config)
- Trend filter + volatility regime gating
- Daily stop loss and max drawdown kill-switches
- Cooldown between consecutive trades, no overlapping positions

### Portfolio Allocation Methods

- Equal weight
- Risk parity (inverse volatility)
- Minimum variance (quadratic optimization)
- Maximum Sharpe ratio (mean-variance optimization)

---

## Output Artifacts

| Path | Description |
|------|-------------|
| `data/market_data.db` | SQLite OHLCV database (~55k bars, 9 tickers) |
| `data/paper_broker.json` | Paper broker state (positions, trades, cash) |
| `data/ticker_params.json` | Per-ticker optimized parameters from grid search |
| `models/model_final_<ticker>.pkl` | Trained LightGBM model for live trading |
| `logs/equity_curve_<ticker>.png` | Backtest equity curve with drawdown panel |
| `logs/signals_<ticker>.csv` | Out-of-sample WFV signal predictions |
| `logs/grid_search.csv` | Full grid search results (13,440 rows) |
| `logs/pipeline.log` | Full execution log (DEBUG/INFO) |
| `mlruns/` | MLflow experiment runs (metrics, params, artifacts) |

---

## Paper Broker

The local paper broker (`paper_broker.py`) simulates live trading without any broker account:

```python
from paper_broker import PaperBroker

broker = PaperBroker(initial_equity=100_000)

# Submit order
broker.submit_order({"ticker": "GLD", "direction": "LONG", ...})

# Fill via yfinance next-bar close
broker.fill_pending()

# Mark-to-market and auto-close stops/TPs
broker.update_positions()

# Full portfolio snapshot
state = broker.portfolio_state()
# Returns: cash, market_value, total_equity, unrealised_pnl, realised_pnl,
#          open_positions, closed_trades, n_open, n_closed, total_return_pct
```

State persists in `data/paper_broker.json` across restarts.

---

## MLflow Experiment Tracking

```bash
mlflow ui
# Open http://localhost:5000
```

Logged per fold: IC, accuracy, F1-macro, model params, feature importances, Sharpe.

---

## Automation (Scheduler)

```bash
python scheduler.py
```

| Time (ET) | Job |
|-----------|-----|
| 08:30 | Incremental data update for all tickers |
| 09:40 | Live signal scan + paper broker routing |
| Saturday 07:00 | Weekly model retraining |

---

## Docker

```bash
# Build image
docker build -t systematic-alpha .

# Run research pipeline with optimized params
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    systematic-alpha \
    python main.py --mode research --use-ticker-params --tickers SPY QQQ GLD

# Run live signal scan
docker run \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    systematic-alpha \
    python main.py --mode live --ticker GLD
```

No `.env` file required — no external API keys needed.

---

## Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=. -v

# Run with timeout (recommended for CI)
pytest tests/ --cov=. --timeout=120

# Run sanity checks (look-ahead bias, position discipline)
pytest tests/test_sanity.py -v
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
| `test_sanity.py` | Look-ahead bias (RSI, momentum, labels), position discipline |
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

This software is for **research and educational purposes only**. It does not constitute financial advice. Past backtest performance does not guarantee future results. Algorithmic trading involves significant financial risk. Use paper trading before deploying any real capital.

Backtest results shown above are **out-of-sample** (walk-forward validation), not in-sample. Returns are small by design — the strategy prioritizes low drawdown and controlled risk over absolute returns.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
