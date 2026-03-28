"""
execution_engine.py
===================
Generates actionable trade signals for manual or automated execution.

Outputs a structured CSV file with one row per signal, containing:
  - timestamp        : bar datetime when signal was generated
  - ticker           : instrument
  - direction        : BUY | SELL (short)
  - entry_price      : suggested limit/market price (current close)
  - stop_loss        : hard stop price
  - take_profit      : target price
  - position_size    : number of shares
  - notional         : position value in USD
  - confidence       : model probability of the predicted class
  - signal_id        : unique UUID for downstream reconciliation

The engine also performs a live signal scan with multi-ticker support:
  - run_live_scan()        : single ticker, returns signal dict
  - run_multi_ticker_scan(): scans all tickers, respects PositionManager
                             (concurrent positions, correlation blocks, notional caps)
  - submit_paper_order()   : forwards the order to the local PaperBroker

Multi-position management
-------------------------
The engine holds a PositionManager instance that tracks all open positions
across tickers, enforces max concurrent position limits, notional caps,
and correlation-based blocking.

Usage
-----
    from execution_engine import ExecutionEngine
    from model_training import ModelTrainer

    trainers = {"SPY": ModelTrainer.load("models/model_final_SPY.pkl"), ...}
    engine = ExecutionEngine(trainers=trainers, equity=100_000, max_positions=4)

    # Single ticker (backward-compatible)
    engine.generate_signals(signals_df, df_ohlcv, ticker="SPY")
    engine.run_live_scan(latest_bar_df, ticker="SPY")

    # Multi-ticker live scan
    results = engine.run_multi_ticker_scan(latest_bars, returns_df=returns)
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import (
    INITIAL_EQUITY,
    MIN_PROBA_THRESHOLD,
    SIGNALS_PATH,
    SLIPPAGE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TICKERS,
)
from model_training import ModelTrainer
from paper_broker import PaperBroker
from position_manager import PositionManager
from risk_management import PositionSizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Converts model predictions into structured execution signals.

    Supports both single-ticker and multi-ticker (portfolio) live scanning.

    Parameters
    ----------
    trainer        : fitted ModelTrainer, OR dict {ticker: ModelTrainer}
                     for multi-ticker operation
    equity         : current account equity in USD
    max_positions  : maximum concurrent open positions (multi-ticker mode)
    max_notional_pct: max total notional as fraction of equity
    """

    def __init__(
        self,
        trainer: Union[ModelTrainer, dict],
        equity: float = INITIAL_EQUITY,
        max_positions: int = 5,
        max_notional_pct: float = 0.80,
    ):
        # Support both single trainer and {ticker: trainer} dict
        if isinstance(trainer, dict):
            self._trainers: dict[str, ModelTrainer] = trainer
            # Use the first trainer as default for backward-compatible single-ticker calls
            self.trainer = next(iter(trainer.values())) if trainer else None
        else:
            self.trainer   = trainer
            self._trainers = {}   # populated lazily on demand

        self.equity  = equity
        self.sizer   = PositionSizer()
        self.broker  = PaperBroker(initial_equity=equity)

        # Multi-position state
        self.positions = PositionManager(
            max_positions    = max_positions,
            max_notional_pct = max_notional_pct,
        )

    # ------------------------------------------------------------------
    def generate_signals(
        self,
        signals_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        ticker: str,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Convert a walk-forward predictions DataFrame into an execution
        signal file.

        Parameters
        ----------
        signals_df  : output of WalkForwardValidator.signals_df
                      (must contain 'pred' and probability columns)
        ohlcv_df    : raw OHLCV bars aligned with signals_df
        ticker      : instrument name
        output_path : CSV save path (defaults to config.SIGNALS_PATH)

        Returns
        -------
        pd.DataFrame of actionable signals (non-neutral only)
        """
        # Align OHLCV with signals
        aligned = signals_df.join(ohlcv_df[["close", "high", "low", "volume"]], how="left")
        aligned = aligned.dropna(subset=["pred", "close"])

        rows = []
        for ts, row in aligned.iterrows():
            pred = int(row["pred"])
            if pred == 0:
                continue

            # Probability filter
            proba_col = f"proba_{pred}"
            proba = float(row.get(proba_col, 0.5))
            if proba < MIN_PROBA_THRESHOLD:
                continue

            entry_price  = float(row["close"])
            direction    = "BUY" if pred == 1 else "SELL"

            # Prices with slippage baked in
            slippage = entry_price * SLIPPAGE_PCT
            if pred == 1:
                fill_est   = entry_price + slippage
                stop_price = round(fill_est * (1 - STOP_LOSS_PCT),  4)
                tp_price   = round(fill_est * (1 + TAKE_PROFIT_PCT), 4)
            else:
                fill_est   = entry_price - slippage
                stop_price = round(fill_est * (1 + STOP_LOSS_PCT),  4)
                tp_price   = round(fill_est * (1 - TAKE_PROFIT_PCT), 4)

            shares   = self.sizer.shares(self.equity, fill_est, STOP_LOSS_PCT)
            notional = round(shares * fill_est, 2)

            rows.append({
                "signal_id":     str(uuid.uuid4())[:8],
                "timestamp":     ts,
                "ticker":        ticker,
                "direction":     direction,
                "entry_price":   round(fill_est, 4),
                "stop_loss":     stop_price,
                "take_profit":   tp_price,
                "position_size": shares,
                "notional_usd":  notional,
                "confidence":    round(proba, 4),
            })

        sig_df = pd.DataFrame(rows)
        if sig_df.empty:
            logger.warning("No actionable signals generated for %s", ticker)
            return sig_df

        # Save to CSV
        path = output_path or SIGNALS_PATH
        sig_df.to_csv(path, index=False)
        logger.info(
            "Signals exported: %d rows -> %s", len(sig_df), path
        )
        return sig_df

    # ------------------------------------------------------------------
    def run_live_scan(
        self,
        latest_df: pd.DataFrame,
        ticker: str,
    ) -> Optional[dict]:
        """
        Produce a single live signal from the most recent feature bar.

        Intended to be called at bar close (e.g. every 5 minutes) with
        the latest prepared feature row.

        Parameters
        ----------
        latest_df : single-row (or multi-row) feature DataFrame;
                    the LAST row is used for prediction
        ticker    : instrument name

        Returns
        -------
        dict with signal fields, or None if no trade is warranted
        """
        if latest_df.empty:
            logger.warning("live_scan: empty DataFrame received")
            return None

        row_df = latest_df.tail(1)
        proba_df = self.trainer.predict_proba(row_df)
        pred = int(self.trainer.predict(row_df)[0])

        if pred == 0:
            logger.info("live_scan [%s]: no signal (neutral)", ticker)
            return None

        proba_col = pred
        proba = float(proba_df[proba_col].iloc[0]) if proba_col in proba_df.columns else 0.5

        if proba < MIN_PROBA_THRESHOLD:
            logger.info(
                "live_scan [%s]: signal %+d filtered (proba=%.3f < %.3f)",
                ticker, pred, proba, MIN_PROBA_THRESHOLD,
            )
            return None

        # Assume entry at current close
        entry_price = float(row_df["close"].iloc[0])
        slippage    = entry_price * SLIPPAGE_PCT
        direction   = "BUY" if pred == 1 else "SELL"

        if pred == 1:
            fill_est   = entry_price + slippage
            stop_price = round(fill_est * (1 - STOP_LOSS_PCT),  4)
            tp_price   = round(fill_est * (1 + TAKE_PROFIT_PCT), 4)
        else:
            fill_est   = entry_price - slippage
            stop_price = round(fill_est * (1 + STOP_LOSS_PCT),  4)
            tp_price   = round(fill_est * (1 - TAKE_PROFIT_PCT), 4)

        shares   = self.sizer.shares(self.equity, fill_est, STOP_LOSS_PCT)
        notional = round(shares * fill_est, 2)

        signal = {
            "signal_id":     str(uuid.uuid4())[:8],
            "timestamp":     row_df.index[-1],
            "ticker":        ticker,
            "direction":     direction,
            "entry_price":   round(fill_est, 4),
            "stop_loss":     stop_price,
            "take_profit":   tp_price,
            "position_size": shares,
            "notional_usd":  notional,
            "confidence":    round(proba, 4),
        }

        logger.info(
            "LIVE SIGNAL [%s]  %s  entry=%.2f  sl=%.2f  tp=%.2f  "
            "shares=%d  confidence=%.3f",
            ticker, direction, fill_est, stop_price, tp_price, shares, proba,
        )
        return signal

    # ------------------------------------------------------------------
    def submit_paper_order(self, signal: dict) -> Optional[dict]:
        """
        Forward a signal to the local PaperBroker for simulated execution.

        Parameters
        ----------
        signal : dict produced by run_live_scan() or generate_signals()

        Returns the broker order record, or None if skipped.
        """
        if not signal:
            return None
        return self.broker.submit_order(signal)

    # ------------------------------------------------------------------
    def run_multi_ticker_scan(
        self,
        latest_bars: dict[str, pd.DataFrame],
        returns_df:  Optional[pd.DataFrame] = None,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Scan multiple tickers simultaneously and return actionable signals
        that pass the PositionManager eligibility checks.

        Parameters
        ----------
        latest_bars : {ticker: feature_DataFrame} — one entry per instrument.
                      Each DataFrame must have feature columns compatible with
                      the corresponding ModelTrainer.
        returns_df  : optional daily returns DataFrame (columns = tickers) used
                      for correlation-based blocking.
        output_path : if provided, all signals are appended to this CSV.

        Returns
        -------
        list of signal dicts (one per accepted new position).
        Each dict has the same structure as run_live_scan().

        Workflow
        --------
        1. For each ticker: run_live_scan() to generate a candidate signal
        2. Check PositionManager.can_open() — enforces max positions,
           notional cap, and correlation blocks
        3. If allowed: register with PositionManager.open_position()
        4. Collect and return all accepted signals
        """
        accepted_signals = []

        # Check existing positions for stop/TP breaches first
        current_prices = {}
        for ticker, df in latest_bars.items():
            if not df.empty and "close" in df.columns:
                current_prices[ticker] = float(df["close"].iloc[-1])

        triggered = self.positions.check_stops(current_prices)
        if triggered:
            for rec in triggered:
                logger.info(
                    "Auto-close [%s]: %s exit=%.2f  pnl=$%.2f",
                    rec["ticker"], rec["exit_reason"], rec["exit_price"], rec["gross_pnl"],
                )

        # Scan each ticker for new signals
        for ticker, feature_df in latest_bars.items():
            if feature_df.empty:
                continue

            # Resolve trainer for this ticker
            trainer = self._trainers.get(ticker, self.trainer)
            if trainer is None:
                logger.warning("No trainer available for %s — skipping", ticker)
                continue

            # Save current trainer, run scan, restore
            _orig = self.trainer
            self.trainer = trainer
            signal = self.run_live_scan(feature_df, ticker=ticker)
            self.trainer = _orig

            if signal is None:
                continue

            # Portfolio eligibility check
            notional  = signal.get("notional_usd", 0.0)
            direction = 1 if signal.get("direction") == "BUY" else -1

            allowed, reason = self.positions.can_open(
                ticker    = ticker,
                direction = direction,
                notional  = notional,
                equity    = self.equity,
                returns   = returns_df,
            )

            if not allowed:
                logger.info(
                    "Signal [%s] blocked by PositionManager: %s", ticker, reason
                )
                continue

            # Register the position
            shares       = signal.get("position_size", 0)
            entry_price  = signal.get("entry_price", 0.0)
            stop_loss    = signal.get("stop_loss")
            take_profit  = signal.get("take_profit")

            self.positions.open_position(
                ticker      = ticker,
                direction   = direction,
                entry_price = entry_price,
                shares      = shares,
                notional    = notional,
                signal_id   = signal.get("signal_id", ""),
                stop_loss   = stop_loss,
                take_profit = take_profit,
            )

            accepted_signals.append(signal)

            # Forward to local paper broker
            self.submit_paper_order(signal)

        # Log portfolio state
        if accepted_signals or triggered:
            self.positions.print_positions()

        # Persist signals to CSV
        if accepted_signals and output_path:
            sig_df = pd.DataFrame(accepted_signals)
            mode   = "a" if Path(output_path).exists() else "w"
            header = mode == "w"
            sig_df.to_csv(output_path, mode=mode, header=header, index=False)
            logger.info(
                "Multi-ticker signals appended: %d rows -> %s",
                len(accepted_signals), output_path,
            )

        return accepted_signals

    # ------------------------------------------------------------------
    def close_position(
        self,
        ticker:      str,
        exit_price:  float,
        exit_reason: str = "signal",
    ) -> Optional[dict]:
        """
        Manually close a position tracked by the PositionManager.

        Submits the closing order to Alpaca if connected.

        Returns the closed trade record, or None if ticker was not open.
        """
        record = self.positions.close_position(ticker, exit_price, exit_reason)
        if record is None:
            return None

        # Forward closing order to paper broker
        direction = -1 if record["direction"] == "LONG" else 1
        close_signal = {
            "ticker":        ticker,
            "direction":     "SELL" if direction == -1 else "BUY",
            "entry_price":   exit_price,
            "position_size": record["shares"],
            "notional_usd":  round(record["shares"] * exit_price, 2),
            "confidence":    1.0,
        }
        self.submit_paper_order(close_signal)

        return record

    # ------------------------------------------------------------------
    def print_signal(self, signal: dict) -> None:
        """Pretty-print a single signal dict."""
        if not signal:
            return
        print("\n" + "=" * 55)
        print("  EXECUTION SIGNAL")
        print("=" * 55)
        for k, v in signal.items():
            print(f"  {k:<20} {v}")
        print("=" * 55 + "\n")
