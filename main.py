"""
main.py
=======
Orchestrates the full Systematic Alpha Research Pipeline.

Modes
-----
  --mode update      : download / refresh market data
  --mode research    : full pipeline per ticker (features -> labels -> WFV
                       -> backtest -> factor analysis -> regime breakdown)
  --mode portfolio   : multi-ticker portfolio-level analysis (IC aggregate,
                       correlation matrix, risk-parity allocation summary)
  --mode live        : generate live signals for today's bars
  --mode train       : train final model on all data and save

Run examples
------------
  python main.py --mode update
  python main.py --mode research --tickers SPY QQQ
  python main.py --mode portfolio
  python main.py --mode live --ticker SPY
  python main.py --mode train --ticker SPY
  python main.py --mode research --tickers SPY QQQ --expanding
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── project modules ──────────────────────────────────────────────────────────
from config import (
    INITIAL_EQUITY,
    LOGS_DIR,
    MODELS_DIR,
    SIGNALS_PATH,
    TICKERS,
)
from data_pipeline import load_data, load_vix_data, update_data
from feature_engineering import build_features, get_feature_names
from labeling import apply_triple_barrier, label_report
from model_training import ModelTrainer, temporal_split
from walk_forward import WalkForwardValidator
from backtest_engine import BacktestEngine
from execution_engine import ExecutionEngine
from performance import (
    compute_metrics,
    monthly_returns_table,
    plot_equity_curve,
    print_metrics,
    sharpe_confidence_interval,
)
from notifier import Notifier
from regime_detection import RegimeDetector
from factor_analysis import FactorAnalyzer
from portfolio_manager import PortfolioManager


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level    = logging.DEBUG if verbose else logging.INFO
    log_file = LOGS_DIR / "pipeline.log"
    stream_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    handlers: list[logging.Handler] = [
        stream_handler,
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_update(tickers: list[str]) -> None:
    """Stage 1 — Download and persist market data."""
    logger.info("=" * 60)
    logger.info("STAGE: DATA UPDATE")
    logger.info("=" * 60)
    update_data(tickers=tickers)
    logger.info("Data update complete.")


def stage_prepare(
    ticker: str,
    vix_df: pd.DataFrame | None = None,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stages 2-3 — Load data, build features, apply Triple-Barrier labels."""
    logger.info("=" * 60)
    logger.info("STAGE: FEATURE ENGINEERING + LABELING  [%s]", ticker)
    logger.info("=" * 60)

    raw = load_data(ticker)
    if raw.empty:
        raise RuntimeError(f"No data for {ticker}. Run --mode update first.")

    logger.info("Raw bars: %d  [%s -> %s]", len(raw), raw.index[0], raw.index[-1])

    featured = build_features(
        raw,
        spy_df=spy_df if ticker != "SPY" else None,
        vix_df=vix_df,
        ticker=ticker,
    )
    labeled = apply_triple_barrier(featured)
    logger.info("\n%s", label_report(labeled).to_string())
    return labeled


def stage_walk_forward(
    labeled: pd.DataFrame,
    ticker: str = "",
    expanding: bool = False,
) -> WalkForwardValidator:
    """Stage 4 — Walk-forward validation with per-fold IC tracking."""
    logger.info("=" * 60)
    logger.info("STAGE: WALK-FORWARD VALIDATION  [expanding=%s]", expanding)
    logger.info("=" * 60)
    wfv = WalkForwardValidator(labeled, expanding=expanding)
    wfv.run(ticker=ticker)
    summary = wfv.summary()
    logger.info("\n%s", summary.to_string())
    return wfv


def stage_backtest(
    wfv: WalkForwardValidator,
    raw_df: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 5 — Realistic backtest on OOS signal stream."""
    logger.info("=" * 60)
    logger.info("STAGE: BACKTEST SIMULATION  [%s]", ticker)
    logger.info("=" * 60)

    if wfv.signals_df is None or wfv.signals_df.empty:
        logger.warning("No signals produced by WFV — skipping backtest.")
        return pd.DataFrame(), pd.Series(dtype=float)

    engine       = BacktestEngine(df=raw_df, signals=wfv.signals_df, equity=INITIAL_EQUITY)
    trades_df    = engine.run()
    equity_curve = engine.equity_curve

    logger.info("Trades: %d", len(trades_df))
    if not trades_df.empty:
        logger.info("\n%s", trades_df.tail(10).to_string())

    return trades_df, equity_curve


def stage_performance(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    ticker: str,
) -> dict:
    """Stage 6 — Institutional performance metrics with bootstrap Sharpe CI."""
    logger.info("=" * 60)
    logger.info("STAGE: PERFORMANCE REPORT  [%s]", ticker)
    logger.info("=" * 60)

    if trades_df.empty:
        logger.warning("No trades to report.")
        return {}

    metrics = compute_metrics(trades_df, equity_curve, INITIAL_EQUITY)
    print_metrics(metrics)

    # Bootstrap Sharpe confidence interval
    lo, hi = sharpe_confidence_interval(equity_curve)
    logger.info("Sharpe 95%% CI (bootstrap n=2000): [%.4f, %.4f]", lo, hi)
    metrics["sharpe_ci_lo"] = lo
    metrics["sharpe_ci_hi"] = hi

    try:
        monthly = monthly_returns_table(equity_curve)
        logger.info("\nMonthly Returns (%%):\n%s", monthly.to_string())
    except Exception as e:
        logger.debug("Monthly returns table skipped: %s", e)

    chart_path = LOGS_DIR / f"equity_curve_{ticker}.png"
    try:
        plot_equity_curve(equity_curve, trades_df, metrics, save_path=chart_path)
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    return metrics


def stage_factor_analysis(
    wfv: WalkForwardValidator,
    raw_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    spy_raw: pd.DataFrame,
    ticker: str,
) -> None:
    """Stage 7 — IC/ICIR, signal decay, factor attribution, turnover."""
    logger.info("=" * 60)
    logger.info("STAGE: FACTOR ANALYSIS  [%s]", ticker)
    logger.info("=" * 60)

    if wfv.signals_df is None or wfv.signals_df.empty:
        logger.warning("No signals — skipping factor analysis.")
        return

    fa     = FactorAnalyzer()
    prices = raw_df["close"]

    ic_df    = fa.ic_summary(wfv.signals_df, prices)
    decay_df = fa.signal_decay(wfv.signals_df, prices, max_horizon=10)
    turnover = fa.turnover_analysis(wfv.signals_df, trades_df)

    attribution = {}
    if not spy_raw.empty and not trades_df.empty:
        attribution = fa.factor_attribution(trades_df, spy_raw["close"])

    fa.print_report(ic_df, decay_df, attribution, turnover)


def stage_regime_analysis(
    trades_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ticker: str,
) -> None:
    """Stage 8 — GMM regime detection and conditional performance breakdown."""
    logger.info("=" * 60)
    logger.info("STAGE: REGIME ANALYSIS  [%s]", ticker)
    logger.info("=" * 60)

    if trades_df.empty:
        logger.warning("No trades — skipping regime analysis.")
        return

    rd      = RegimeDetector(method="gmm", n_regimes=3)
    regimes = rd.fit_predict(raw_df["close"])

    stats     = rd.regime_stats(regimes)
    logger.info("Regime distribution:\n%s", stats.to_string())

    breakdown = rd.performance_by_regime(trades_df, regimes)
    if not breakdown.empty:
        logger.info("Performance by regime:\n%s", breakdown.to_string())


def stage_train_final(labeled: pd.DataFrame, ticker: str) -> ModelTrainer:
    """Stage 9 — Train final model on all data and persist."""
    logger.info("=" * 60)
    logger.info("STAGE: TRAIN FINAL MODEL  [%s]", ticker)
    logger.info("=" * 60)
    trainer = ModelTrainer()
    trainer.fit(labeled)

    imp = trainer.feature_importance().head(10)
    logger.info("Top-10 feature importances:\n%s", imp.to_string())

    path = trainer.save(MODELS_DIR / f"model_final_{ticker}.pkl")
    logger.info("Final model saved -> %s", path)
    return trainer


def stage_live(trainer: ModelTrainer, ticker: str, equity: float) -> None:
    """Stage 10 — Generate live signals from latest bars."""
    logger.info("=" * 60)
    logger.info("STAGE: LIVE SIGNAL SCAN  [%s]", ticker)
    logger.info("=" * 60)

    update_data(tickers=[ticker])
    raw = load_data(ticker)
    if raw.empty:
        logger.error("No data available for live scan.")
        return

    vix_df   = load_vix_data()
    spy_raw  = load_data("SPY") if ticker != "SPY" else None
    featured = build_features(raw, spy_df=spy_raw, vix_df=vix_df, ticker=ticker)

    latest_df = featured.tail(100)
    engine    = ExecutionEngine(trainer=trainer, equity=equity)
    signal    = engine.run_live_scan(latest_df, ticker=ticker)
    engine.print_signal(signal)

    if signal:
        sig_df = pd.DataFrame([signal])
        sig_df.to_csv(SIGNALS_PATH, index=False)
        logger.info("Signal written -> %s", SIGNALS_PATH)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic Alpha Research Pipeline")
    parser.add_argument(
        "--mode",
        choices=["update", "research", "portfolio", "live", "train"],
        default="research",
    )
    parser.add_argument("--ticker",  type=str, default=TICKERS[0])
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--equity",  type=float, default=INITIAL_EQUITY)
    parser.add_argument(
        "--expanding", action="store_true",
        help="Use expanding (anchored) WFV window instead of rolling",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    ticker_list = args.tickers if args.tickers else [args.ticker]
    logger.info("Pipeline — mode=%s  tickers=%s", args.mode, ticker_list)

    # ── UPDATE ──────────────────────────────────────────────────────────────
    if args.mode == "update":
        stage_update(TICKERS)

    # ── RESEARCH ────────────────────────────────────────────────────────────
    elif args.mode == "research":
        stage_update(ticker_list)

        all_metrics:       dict[str, dict]      = {}
        all_equity_curves: dict[str, pd.Series] = {}
        notif = Notifier()

        logger.info("Loading VIX and SPY reference data...")
        vix_df  = load_vix_data()
        spy_raw = load_data("SPY")

        for ticker in ticker_list:
            logger.info("=" * 60)
            logger.info("TICKER: %s", ticker)
            logger.info("=" * 60)
            try:
                spy_input = spy_raw if ticker != "SPY" else None
                labeled   = stage_prepare(ticker, vix_df=vix_df, spy_df=spy_input)
                wfv       = stage_walk_forward(labeled, ticker=ticker,
                                               expanding=args.expanding)
                raw       = load_data(ticker)
                trades_df, equity_curve = stage_backtest(wfv, raw, ticker)
                metrics   = stage_performance(trades_df, equity_curve, ticker)
                all_metrics[ticker]       = metrics
                all_equity_curves[ticker] = equity_curve

                if metrics:
                    notif.performance(ticker, metrics)

                stage_factor_analysis(wfv, raw, trades_df, spy_raw, ticker)
                stage_regime_analysis(trades_df, raw, ticker)

                if wfv.signals_df is not None and not wfv.signals_df.empty:
                    trainer_tmp = ModelTrainer()
                    trainer_tmp.fit(labeled)
                    trainer_tmp.save(MODELS_DIR / f"model_final_{ticker}.pkl")
                    exec_engine = ExecutionEngine(trainer=trainer_tmp, equity=args.equity)
                    sig_path    = LOGS_DIR / f"signals_{ticker}.csv"
                    exec_engine.generate_signals(
                        signals_df=wfv.signals_df,
                        ohlcv_df=raw,
                        ticker=ticker,
                        output_path=sig_path,
                    )

            except Exception as exc:
                logger.error("Ticker %s failed: %s", ticker, exc, exc_info=True)
                notif.alert(f"Pipeline failed for {ticker}: {exc}")

        # Cross-ticker aggregate
        if all_metrics:
            logger.info("=" * 60)
            logger.info("AGGREGATE SUMMARY — %d tickers", len(all_metrics))
            logger.info("=" * 60)
            summary_rows = [
                {
                    "ticker": t,
                    **{k: v for k, v in m.items() if k in (
                        "sharpe_ratio", "sortino_ratio", "omega_ratio",
                        "max_drawdown_pct", "win_rate_pct", "profit_factor",
                        "n_trades", "total_return_pct", "cagr_pct",
                        "sharpe_ci_lo", "sharpe_ci_hi",
                    )},
                }
                for t, m in all_metrics.items() if m
            ]
            summary_df = pd.DataFrame(summary_rows).set_index("ticker")
            logger.info("\n%s", summary_df.to_string())

            # Portfolio-level blended stats
            pm      = PortfolioManager(equity=args.equity)
            port_s  = pm.portfolio_summary(all_metrics, all_equity_curves)
            pm.print_summary(port_s)

    # ── PORTFOLIO ────────────────────────────────────────────────────────────
    elif args.mode == "portfolio":
        stage_update(TICKERS)

        vix_df  = load_vix_data()
        spy_raw = load_data("SPY")

        all_metrics:       dict[str, dict]      = {}
        all_equity_curves: dict[str, pd.Series] = {}
        all_returns:       dict[str, pd.Series] = {}

        for ticker in TICKERS:
            logger.info("PORTFOLIO TICKER: %s", ticker)
            try:
                spy_input = spy_raw if ticker != "SPY" else None
                labeled   = stage_prepare(ticker, vix_df=vix_df, spy_df=spy_input)
                wfv       = stage_walk_forward(labeled, ticker=ticker)
                raw       = load_data(ticker)
                trades_df, equity_curve = stage_backtest(wfv, raw, ticker)
                metrics   = stage_performance(trades_df, equity_curve, ticker)
                all_metrics[ticker]       = metrics
                all_equity_curves[ticker] = equity_curve
                all_returns[ticker]       = raw["close"].pct_change().dropna()
            except Exception as exc:
                logger.error("Portfolio ticker %s failed: %s", ticker, exc, exc_info=True)

        pm      = PortfolioManager(equity=args.equity, method="risk_parity")
        summary = pm.portfolio_summary(all_metrics, all_equity_curves)
        pm.print_summary(summary)

        if all_returns:
            returns_df     = pd.DataFrame(all_returns).dropna(how="all")
            corr           = pm.correlation_matrix(returns_df, flag_threshold=0.80)
            logger.info("Correlation matrix:\n%s", corr.to_string())

            active_signals = {t: 1 for t in TICKERS}
            w_rp  = pm.compute_weights(TICKERS, returns_df, active_signals)
            dr_rp = pm.diversification_ratio(w_rp, returns_df)
            logger.info("Diversification Ratio (risk-parity):  %.4f", dr_rp)

            pm_mv = PortfolioManager(equity=args.equity, method="min_variance")
            w_mv  = pm_mv.compute_weights(TICKERS, returns_df, active_signals)
            dr_mv = pm_mv.diversification_ratio(w_mv, returns_df)
            logger.info("Diversification Ratio (min-variance): %.4f", dr_mv)

            pm_ms = PortfolioManager(equity=args.equity, method="max_sharpe")
            w_ms  = pm_ms.compute_weights(TICKERS, returns_df, active_signals)
            dr_ms = pm_ms.diversification_ratio(w_ms, returns_df)
            logger.info("Diversification Ratio (max-sharpe):   %.4f", dr_ms)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    elif args.mode == "train":
        vix_df  = load_vix_data()
        spy_raw = load_data("SPY")
        for ticker in ticker_list:
            spy_input = spy_raw if ticker != "SPY" else None
            labeled   = stage_prepare(ticker, vix_df=vix_df, spy_df=spy_input)
            trainer   = stage_train_final(labeled, ticker)

            # SHAP importance (requires: pip install shap)
            try:
                shap_imp = trainer.shap_importance(labeled.tail(500))
                logger.info("SHAP top-10:\n%s", shap_imp.head(10).to_string())
            except Exception as e:
                logger.debug("SHAP skipped: %s", e)

    # ── LIVE ─────────────────────────────────────────────────────────────────
    elif args.mode == "live":
        vix_df  = load_vix_data()
        spy_raw = load_data("SPY")
        for ticker in ticker_list:
            model_path = MODELS_DIR / f"model_final_{ticker}.pkl"
            if not model_path.exists():
                logger.info("No saved model for %s — training now...", ticker)
                spy_input = spy_raw if ticker != "SPY" else None
                labeled   = stage_prepare(ticker, vix_df=vix_df, spy_df=spy_input)
                trainer   = stage_train_final(labeled, ticker)
            else:
                trainer = ModelTrainer.load(model_path)
            stage_live(trainer, ticker, args.equity)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
