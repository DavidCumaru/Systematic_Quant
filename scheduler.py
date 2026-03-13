"""
scheduler.py
============
Daily automation for the Systematic Alpha pipeline.

Schedule
--------
  08:30 ET (weekdays) — update data (incremental download)
  09:40 ET (weekdays) — live signal scan for all tickers + Telegram alerts
  Saturday 07:00      — full research re-run + model retrain (weekly refresh)

Usage
-----
  # Run in background (keep terminal open or use a process manager)
  python scheduler.py

  # Or import and call manually:
  from scheduler import run_daily_update, run_live_scan

Environment variables required for live execution:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   — for signal notifications
  ALPACA_KEY, ALPACA_SECRET          — for paper/live order execution

Dependencies:  pip install schedule
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Add project root to path so imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import schedule
    _SCHEDULE_OK = True
except ImportError:
    _SCHEDULE_OK = False

from config import INITIAL_EQUITY, MODELS_DIR, TICKERS, TIMEZONE
from data_pipeline import load_data, load_vix_data, update_data
from feature_engineering import build_features
from model_training import ModelTrainer
from notifier import Notifier
from execution_engine import ExecutionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")
notifier = Notifier()

ET = ZoneInfo(TIMEZONE)


# ---------------------------------------------------------------------------
# Core jobs
# ---------------------------------------------------------------------------

def run_daily_update() -> None:
    """08:30 ET — refresh intraday data for all tickers."""
    now = datetime.now(ET)
    logger.info("=== Daily data update started  %s ===", now.strftime("%Y-%m-%d %H:%M ET"))
    try:
        update_data(tickers=TICKERS)
        logger.info("Data update complete.")
        notifier.alert("Data update complete for " + ", ".join(TICKERS))
    except Exception as exc:
        logger.error("Data update failed: %s", exc, exc_info=True)
        notifier.alert(f"ERROR: Data update failed — {exc}")


def run_live_scan(equity: float = INITIAL_EQUITY) -> None:
    """
    09:40 ET — load saved models, scan latest bar, emit signals.

    For each ticker:
      1. Load the saved final model (train if missing)
      2. Build features on the latest data
      3. Run live scan — emit signal if confidence >= threshold
      4. Send Telegram notification
      5. Submit paper order via Alpaca if configured
    """
    now = datetime.now(ET)
    logger.info("=== Live signal scan started  %s ===", now.strftime("%Y-%m-%d %H:%M ET"))

    # Load VIX and SPY once (shared across all tickers)
    vix_df  = load_vix_data()
    spy_raw = load_data("SPY")

    for ticker in TICKERS:
        try:
            # Load or train model
            model_path = MODELS_DIR / f"model_final_{ticker}.pkl"
            if model_path.exists():
                trainer = ModelTrainer.load(model_path)
            else:
                logger.info("No saved model for %s — training now...", ticker)
                raw     = load_data(ticker)
                from labeling import apply_triple_barrier
                featured = build_features(raw, spy_df=spy_raw if ticker != "SPY" else None,
                                          vix_df=vix_df, ticker=ticker)
                labeled  = apply_triple_barrier(featured)
                trainer  = ModelTrainer()
                trainer.fit(labeled)
                trainer.save(model_path)

            # Build features on latest data
            raw      = load_data(ticker)
            featured = build_features(
                raw,
                spy_df=spy_raw if ticker != "SPY" else None,
                vix_df=vix_df,
                ticker=ticker,
            )
            latest   = featured.tail(200)  # warm-up window

            # Generate signal
            engine = ExecutionEngine(trainer=trainer, equity=equity)
            signal = engine.run_live_scan(latest, ticker=ticker)

            if signal:
                engine.print_signal(signal)
                # Telegram notification
                notifier.signal(
                    ticker=ticker,
                    direction=signal["direction"],
                    price=signal["entry_price"],
                    stop=signal["stop_loss"],
                    tp=signal["take_profit"],
                    shares=signal["position_size"],
                    confidence=signal["confidence"],
                    signal_id=signal["signal_id"],
                )
                # Alpaca paper order submission
                engine.submit_alpaca_order(signal)
            else:
                logger.info("[%s] No signal this bar.", ticker)

        except Exception as exc:
            logger.error("Live scan failed for %s: %s", ticker, exc, exc_info=True)
            notifier.alert(f"ERROR: Live scan failed for {ticker} — {exc}")


def run_weekly_research() -> None:
    """Saturday 07:00 ET — full research pipeline + model retrain for all tickers."""
    logger.info("=== Weekly research re-run started ===")
    notifier.alert("Weekly model retraining started for " + ", ".join(TICKERS))
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--mode", "research",
             "--tickers"] + TICKERS,
            capture_output=False,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            notifier.alert("Weekly research complete.")
        else:
            notifier.alert("Weekly research finished with errors — check logs.")
    except Exception as exc:
        logger.error("Weekly research failed: %s", exc, exc_info=True)
        notifier.alert(f"ERROR: Weekly research failed — {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _is_market_day() -> bool:
    """Return True if today is a weekday (Mon-Fri). Simple market day proxy."""
    return datetime.now(ET).weekday() < 5  # 0=Mon, 4=Fri


def main() -> None:
    if not _SCHEDULE_OK:
        logger.error("'schedule' not installed. Run: pip install schedule")
        sys.exit(1)

    logger.info("Systematic Alpha Scheduler starting...")
    logger.info("Tickers: %s", TICKERS)
    logger.info("Timezone: %s", TIMEZONE)

    # Register jobs
    schedule.every().day.at("08:30").do(
        lambda: run_daily_update() if _is_market_day() else None
    )
    schedule.every().day.at("09:40").do(
        lambda: run_live_scan()    if _is_market_day() else None
    )
    schedule.every().saturday.at("07:00").do(run_weekly_research)

    logger.info("Jobs scheduled:")
    logger.info("  08:30 ET (weekdays) — data update")
    logger.info("  09:40 ET (weekdays) — live signal scan")
    logger.info("  Saturday 07:00      — weekly model retrain")
    logger.info("Scheduler running — press Ctrl+C to stop.\n")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
