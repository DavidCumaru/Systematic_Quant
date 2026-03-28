"""
grid_search.py
==============
Exhaustive per-ticker parameter search to maximise out-of-sample Sharpe.

Strategy
--------
1.  Run Walk-Forward Validation ONCE per ticker with default params to get
    the OOS signals DataFrame (model predictions).
2.  For each combination in PARAM_GRID:
      a. Instantiate BacktestEngine with the custom params dict.
      b. Run backtest on the OOS signals.
      c. Compute Sharpe ratio (requires >= MIN_TRADES).
3.  Select the combination with the highest Sharpe.
4.  Persist the best params to data/ticker_params.json via ticker_config.

Run
---
    python grid_search.py                        # all tickers in config
    python grid_search.py --tickers SPY QQQ      # specific tickers
    python grid_search.py --tickers GLD --top 5  # show top-5 combos
    python grid_search.py --no-wfv               # skip WFV (use cached signals)

Output
------
  data/ticker_params.json   — best params per ticker
  logs/grid_search.csv      — all results for analysis
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from backtest_engine import BacktestEngine
from config import INITIAL_EQUITY, LOGS_DIR, TICKERS
from data_pipeline import load_data, load_vix_data
from feature_engineering import build_features
from labeling import apply_triple_barrier
from performance import compute_metrics
from regime_detection import RegimeDetector
from ticker_config import DEFAULTS, save_ticker_params
from walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("grid_search")
logger.setLevel(logging.INFO)

# ── Search space ──────────────────────────────────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "min_proba_threshold": [0.48, 0.52, 0.56, 0.60],
    "stop_loss_pct":       [0.005, 0.007, 0.010, 0.015],
    "take_profit_pct":     [0.008, 0.012, 0.018, 0.025],
    "time_stop_bars":      [2, 3, 5],
    "direction":           ["both", "long_only"],
    "regime_filter":       [
        "all", "Bull", "Bear", "Sideways",
        "Bull+Sideways", "Bear+Sideways", "Bear+Bull",
    ],
}

# Minimum trades required to consider a combination valid
MIN_TRADES = 10

# Metric used to rank combinations
RANK_METRIC = "sharpe_ratio"   # alternatives: "profit_factor", "total_return_pct"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combinations() -> list[dict]:
    """Return every combination from PARAM_GRID as a list of dicts."""
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_wfv(ticker: str) -> tuple[WalkForwardValidator, pd.DataFrame, pd.Series]:
    """Run WFV for a ticker and return (wfv, raw_df, regimes)."""
    logger.info("[%s] Running Walk-Forward Validation …", ticker)
    vix_df  = load_vix_data()
    spy_raw = load_data("SPY")
    raw     = load_data(ticker)

    spy_in  = spy_raw if ticker != "SPY" else None
    feat    = build_features(raw, spy_df=spy_in, vix_df=vix_df, ticker=ticker)
    labeled = apply_triple_barrier(feat)

    wfv = WalkForwardValidator(labeled, expanding=False)
    wfv.run(ticker=ticker)

    # Fit regimes on the full price series
    rd      = RegimeDetector(method="gmm", n_regimes=3)
    regimes = rd.fit_predict(raw["close"])

    return wfv, raw, regimes


def _search_ticker(
    ticker: str,
    wfv: WalkForwardValidator,
    raw: pd.DataFrame,
    regimes: pd.Series,
    top_n: int = 1,
    equity: float = INITIAL_EQUITY,
) -> tuple[dict, list[dict]]:
    """
    Run the full grid search for one ticker.

    Returns
    -------
    best_params : dict   — the winning combination
    all_results : list[dict] — all valid combinations sorted by RANK_METRIC
    """
    if wfv.signals_df is None or wfv.signals_df.empty:
        logger.warning("[%s] No WFV signals — skipping.", ticker)
        return dict(DEFAULTS), []

    combos = _combinations()
    logger.info(
        "[%s] Testing %d combinations …", ticker, len(combos)
    )

    results = []
    t0 = time.time()

    for combo in combos:
        try:
            eng = BacktestEngine(
                df=raw,
                signals=wfv.signals_df,
                equity=equity,
                params=combo,
                regimes=regimes,
            )
            trades_df = eng.run()
            eq        = eng.equity_curve

            if trades_df.empty or len(trades_df) < MIN_TRADES:
                continue

            met = compute_metrics(trades_df, eq, equity)
            row = {**combo,
                   "n_trades":        int(met.get("n_trades", 0)),
                   "sharpe_ratio":    round(met.get("sharpe_ratio", -9), 4),
                   "sortino_ratio":   round(met.get("sortino_ratio", -9), 4),
                   "profit_factor":   round(met.get("profit_factor", 0), 4),
                   "win_rate_pct":    round(met.get("win_rate_pct", 0), 2),
                   "total_return_pct":round(met.get("total_return_pct", -100), 4),
                   "max_drawdown_pct":round(met.get("max_drawdown_pct", 100), 4),
                   "ticker":          ticker}
            results.append(row)
        except Exception:
            continue

    elapsed = time.time() - t0
    logger.info(
        "[%s] %d valid combos in %.1fs (%.0f/s)",
        ticker, len(results), elapsed, len(combos) / max(elapsed, 0.01),
    )

    if not results:
        logger.warning("[%s] No valid combinations found.", ticker)
        return dict(DEFAULTS), []

    results.sort(key=lambda r: r[RANK_METRIC], reverse=True)
    best_raw = results[0]

    # Extract only the param keys (not metrics); fill missing keys from DEFAULTS
    best_params = {k: best_raw.get(k, DEFAULTS[k]) for k in DEFAULTS}

    logger.info(
        "[%s] BEST  Sharpe=%.4f  PF=%.2f  WinRate=%.1f%%  Trades=%d  "
        "regime=%s  dir=%s  conf>=%.2f  sl=%.3f  tp=%.3f  bars=%d",
        ticker,
        best_raw["sharpe_ratio"],
        best_raw["profit_factor"],
        best_raw["win_rate_pct"],
        best_raw["n_trades"],
        best_raw["regime_filter"],
        best_raw["direction"],
        best_raw["min_proba_threshold"],
        best_raw["stop_loss_pct"],
        best_raw["take_profit_pct"],
        best_raw["time_stop_bars"],
    )

    return best_params, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-ticker grid search")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to search (default: all in config)")
    parser.add_argument("--top",     type=int, default=1,
                        help="Show top-N combinations per ticker")
    parser.add_argument("--no-wfv",  action="store_true",
                        help="Skip WFV (requires existing signals CSVs — not yet supported)")
    parser.add_argument("--equity",  type=float, default=INITIAL_EQUITY)
    args = parser.parse_args()

    ticker_list = args.tickers or ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    total_combos = len(_combinations())
    logger.info(
        "Grid search: %d tickers × %d combos = %d backtests",
        len(ticker_list), total_combos, len(ticker_list) * total_combos,
    )

    all_rows: list[dict] = []
    summary: list[dict]  = []

    for ticker in ticker_list:
        print(f"\n{'='*60}")
        print(f"  TICKER: {ticker}")
        print(f"{'='*60}")

        wfv, raw, regimes = _run_wfv(ticker)
        best_params, results = _search_ticker(
            ticker, wfv, raw, regimes,
            top_n=args.top, equity=args.equity,
        )

        if results:
            # Save best params
            save_ticker_params(ticker, best_params)

            # Print top-N
            top = results[:args.top]
            for rank, r in enumerate(top, 1):
                print(
                    f"  #{rank:2d}  Sharpe={r['sharpe_ratio']:+.4f}  "
                    f"PF={r['profit_factor']:.2f}  WR={r['win_rate_pct']:.1f}%  "
                    f"Trades={r['n_trades']}  regime={r['regime_filter']}  "
                    f"dir={r['direction']}  conf>={r['min_proba_threshold']:.2f}  "
                    f"sl={r['stop_loss_pct']:.3f}  tp={r['take_profit_pct']:.3f}  "
                    f"bars={r['time_stop_bars']}"
                )

            summary.append({
                "ticker":         ticker,
                **{k: best_params[k] for k in DEFAULTS},
                "best_sharpe":    results[0]["sharpe_ratio"],
                "best_pf":        results[0]["profit_factor"],
                "best_win_rate":  results[0]["win_rate_pct"],
                "best_trades":    results[0]["n_trades"],
            })
            all_rows.extend(results)

    # ── Save full results CSV
    if all_rows:
        csv_path = LOGS_DIR / "grid_search.csv"
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        logger.info("Full results saved -> %s", csv_path)

    # ── Final summary table
    if summary:
        print(f"\n{'='*70}")
        print("  FINAL SUMMARY — Best params per ticker")
        print(f"{'='*70}")
        df_s = pd.DataFrame(summary).set_index("ticker")
        print(df_s.to_string())
        print()
        print(f"Per-ticker params saved -> data/ticker_params.json")
        print(f"Full grid results     -> logs/grid_search.csv")
        print(f"Apply with:  python main.py --mode research --use-ticker-params")


if __name__ == "__main__":
    main()
