# AutoTraderAI — Architecture

## Project Structure

```
AutoTraderAI/
├── autotrader/                   # Main Python package
│   ├── config/                   # Configuration & per-ticker parameters
│   │   ├── settings.py           # Central config (paths, tickers, risk params)
│   │   └── ticker_config.py      # Per-ticker param persistence (JSON)
│   ├── data/                     # Data acquisition & storage
│   │   ├── pipeline.py           # yfinance download → SQLite (incremental)
│   │   ├── alternative.py        # FRED macro data, fear/greed proxy
│   │   └── news.py               # CVM filings, news sentiment
│   ├── features/                 # Feature engineering
│   │   └── engineering.py        # 37+ quantitative features (causal)
│   ├── labeling/                 # Target variable construction
│   │   └── barriers.py           # Triple-Barrier labeling (+1/0/-1)
│   ├── models/                   # ML models & validation
│   │   ├── trainer.py            # LightGBM + Optuna HPO (ModelTrainer)
│   │   ├── walk_forward.py       # Rolling/expanding WFV with IC tracking
│   │   └── regime.py             # GMM/HMM market regime classifier
│   ├── backtesting/              # Simulation & validation
│   │   ├── engine.py             # Event-driven backtest (slippage/commission)
│   │   ├── holdout.py            # Honest out-of-sample holdout backtest
│   │   └── wfv_monthly.py        # Monthly walk-forward backtest variant
│   ├── execution/                # Order routing & broker integration
│   │   ├── engine.py             # Live signal generation + multi-ticker scan
│   │   ├── paper_broker.py       # Local paper broker (JSON-persisted state)
│   │   └── broker_mt5.py         # MetaTrader 5 integration (B3)
│   ├── risk/                     # Risk management & portfolio construction
│   │   ├── management.py         # Kelly sizing, RiskGuard kill-switch
│   │   ├── portfolio.py          # Equal/risk-parity/min-var/max-Sharpe weights
│   │   ├── position.py           # Multi-position state tracker
│   │   └── market_impact.py      # Almgren-Chriss square-root impact model
│   ├── analysis/                 # Performance & signal quality analytics
│   │   ├── performance.py        # Sharpe CI, Sortino, Calmar, monthly table
│   │   └── factors.py            # IC/ICIR, signal decay, factor attribution
│   └── utils/                    # Shared utilities
│       ├── logging.py            # Centralized setup_logging()
│       ├── notifier.py           # Telegram alerts
│       └── journal.py            # Trade journal CSV (signal → P&L reconciliation)
├── scripts/                      # One-off optimization scripts (not a library)
│   ├── grid_search.py            # Exhaustive per-ticker parameter search (2688 combos)
│   └── optimize_rr.py            # TP/SL/TIME risk-reward optimization
├── docs/
│   └── ARCHITECTURE.md           # This file
├── tests/                        # pytest suite (11 files, ~60 test classes)
├── data/                         # Runtime data (SQLite, broker state, ticker params)
├── models/                       # Trained LightGBM artifacts (.pkl, 42 tickers)
├── trades/                       # Trade journal (journal.csv)
├── .github/workflows/            # CI/CD: daily scan + weekly retrain
├── main.py                       # CLI entry point (update/research/portfolio/live/train)
├── scheduler.py                  # Daily automation (09:30/10:10/17:35 BRT)
├── dashboard.py                  # Streamlit dashboard (streamlit run dashboard.py)
├── Dockerfile                    # Multi-stage build (builder + runtime)
└── requirements.txt              # Pinned dependencies
```

---

## Module Dependency Graph

Dependencies flow strictly downward — no circular imports:

```
autotrader.config.settings           (no internal deps)
        ↓
autotrader.config.ticker_config
autotrader.data.{pipeline, alternative, news}
        ↓
autotrader.features.engineering      (lazy: data.alternative, data.news)
        ↓
autotrader.labeling.barriers
        ↓
autotrader.models.trainer
autotrader.models.walk_forward       (→ models.trainer)
autotrader.models.regime
        ↓
autotrader.risk.{management, portfolio, position, market_impact}
autotrader.backtesting.engine        (→ risk.management, risk.market_impact)
autotrader.execution.{engine, paper_broker, broker_mt5}
        ↓
autotrader.analysis.{performance, factors}  (→ risk.management)
autotrader.utils.{notifier, journal, logging}
        ↓
main.py / scheduler.py / dashboard.py      (entry points — top of DAG)
```

---

## Naming Conventions

| Scope | Convention | Example |
|---|---|---|
| Files | `snake_case` | `walk_forward.py`, `market_impact.py` |
| Classes | `PascalCase` | `ModelTrainer`, `BacktestEngine`, `RiskGuard` |
| Functions | `snake_case` | `build_features()`, `apply_triple_barrier()` |
| Private helpers | `_snake_case` | `_rsi()`, `_download_ticker()` |
| Config constants | `UPPER_CASE` | `INITIAL_EQUITY`, `TICKERS` |
| Dataclasses | `PascalCase` | `Trade`, `FoldResult`, `Position` |

---

## Pipeline Execution Modes (`main.py`)

| Mode | Description |
|---|---|
| `--mode update` | Download/refresh OHLCV bars for all tickers |
| `--mode research` | Full pipeline: features → labels → WFV → backtest → analysis |
| `--mode portfolio` | Multi-ticker allocation, correlation matrix, diversification ratio |
| `--mode live` | Generate live signals, submit to broker (paper or MT5) |
| `--mode train` | Train final model on all historical data, save to models/ |

---

## Key Design Decisions

### Causal Feature Engineering
All 37+ features are computed with `shift(1)` or rolling windows that only look backward. Tests in `tests/test_sanity.py` explicitly validate zero look-ahead bias.

### Triple-Barrier Labeling
Labels (+1/0/-1) are determined by which barrier (TP/SL/time-stop) is touched first on future bars. Barriers checked against bar high/low to match backtest fill logic exactly.

### Walk-Forward Validation
24-month rolling train / 6-month test with a 3-bar embargo (purged from train end) to prevent label leakage. The last 12 months are reserved as a blind holdout.

### Specialized LightGBM Models
Two separate binary classifiers per ticker: one for LONG (+1 vs rest) and one for SHORT (-1 vs rest). Reduces the confusion of mixed 3-class training and allows direction-specific feature selection.

### Realistic Backtesting
5 bps slippage + $1 commission + 2 bps spread + Almgren-Chriss square-root market impact. Execution delayed 1 bar. Max 1 concurrent position per ticker; 1-bar cooldown after exit.

### Paper Broker Architecture
State persisted in `data/paper_broker.json` so positions survive process restarts. Fills simulated using next-bar open via yfinance. Compatible interface with MT5Broker for live trading.

---

## Running the Project

```bash
# Update market data
python main.py --mode update

# Full research pipeline for selected tickers
python main.py --mode research --tickers PETR4.SA VALE3.SA ITUB4.SA

# Live signal scan
python main.py --mode live

# Launch dashboard
streamlit run dashboard.py

# Run tests
pytest tests/ -v

# Grid search parameter optimization
python scripts/grid_search.py --tickers PETR4.SA --top 5

# Docker
docker build -t autotrader .
docker run --env-file .env autotrader python main.py --mode live
```
