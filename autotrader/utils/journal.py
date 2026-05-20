"""
journal.py
================
Registra sinais gerados e calcula resultado real apos o prazo de holding.

Fluxo:
  1. Sinal gerado -> journal_add_signal()  -> salva em trades/journal.csv
  2. Sexta-feira  -> journal_weekly_close() -> busca precos, calcula PnL, retorna resumo
  3. Resumo enviado via Telegram pelo scheduler/GitHub Actions

Colunas do journal.csv:
  signal_id, date, ticker, direction, entry, sl, tp, shares, capital,
  confidence, status, exit_price, exit_reason, pnl, pnl_pct, closed_date
"""

import csv
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("trades/journal.csv")

FIELDNAMES = [
    "signal_id", "date", "ticker", "direction",
    "entry", "sl", "tp", "shares", "capital",
    "confidence", "status",
    "exit_price", "exit_reason", "pnl", "pnl_pct", "closed_date",
]


def _ensure_journal():
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def journal_add_signal(signal: dict, capital: float) -> None:
    """Registra um novo sinal no journal."""
    _ensure_journal()
    row = {
        "signal_id":   signal.get("signal_id", ""),
        "date":        date.today().isoformat(),
        "ticker":      signal.get("ticker", ""),
        "direction":   signal.get("direction", ""),
        "entry":       round(float(signal.get("entry_price", 0)), 4),
        "sl":          round(float(signal.get("stop_loss", 0)), 4),
        "tp":          round(float(signal.get("take_profit", 0)), 4),
        "shares":      int(signal.get("position_size", 0)),
        "capital":     round(capital, 2),
        "confidence":  round(float(signal.get("confidence", 0)), 4),
        "status":      "open",
        "exit_price":  "",
        "exit_reason": "",
        "pnl":         "",
        "pnl_pct":     "",
        "closed_date": "",
    }
    with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    logger.info("Journal: sinal registrado %s %s %s", row["ticker"], row["direction"], row["date"])


def journal_open_tickers() -> set:
    """Retorna conjunto de tickers com posicao aberta no journal."""
    _ensure_journal()
    try:
        df = pd.read_csv(JOURNAL_PATH)
        if df.empty:
            return set()
        return set(df[df["status"] == "open"]["ticker"].tolist())
    except Exception:
        return set()


def journal_weekly_close(capital: float, time_stop_bars: int = 3) -> dict:
    """
    Verifica sinais abertos com mais de time_stop_bars dias uteis.
    Busca precos via yfinance e calcula resultado (TP/SL/TIME).
    Retorna dict com resumo da semana.
    """
    _ensure_journal()

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance nao instalado")
        return {}

    try:
        df = pd.read_csv(JOURNAL_PATH)
    except Exception:
        return {}

    if df.empty:
        return {"trades": 0, "message": "Nenhum trade registrado ainda."}

    today      = date.today()
    cutoff     = today - timedelta(days=time_stop_bars + 2)  # +2 para dias nao uteis
    open_mask  = (df["status"] == "open") & (pd.to_datetime(df["date"]).dt.date <= cutoff)
    to_close   = df[open_mask].copy()

    closed_rows = []

    for _, row in to_close.iterrows():
        ticker    = row["ticker"]
        direction = row["direction"]
        entry     = float(row["entry"])
        sl        = float(row["sl"])
        tp        = float(row["tp"])
        shares    = int(row["shares"])
        sig_date  = row["date"]

        try:
            # Busca historico desde a data do sinal
            hist = yf.download(ticker, start=sig_date, period="10d",
                               interval="1d", progress=False, auto_adjust=True)
            if hist.empty:
                continue

            # Remove multiindex se necessario
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0].lower() for c in hist.columns]
            else:
                hist.columns = [c.lower() for c in hist.columns]

            # Pula o primeiro dia (barra de entrada)
            hist = hist.iloc[1:]
            if hist.empty:
                continue

            exit_price  = None
            exit_reason = "TIME"

            for _, bar in hist.iterrows():
                high = float(bar.get("high", entry))
                low  = float(bar.get("low",  entry))
                close= float(bar.get("close", entry))

                if direction == "BUY":
                    if low <= sl:
                        exit_price  = sl
                        exit_reason = "SL"
                        break
                    elif high >= tp:
                        exit_price  = tp
                        exit_reason = "TP"
                        break
                else:
                    if high >= sl:
                        exit_price  = sl
                        exit_reason = "SL"
                        break
                    elif low <= tp:
                        exit_price  = tp
                        exit_reason = "TP"
                        break
            else:
                # TIME STOP: usa o ultimo fechamento disponivel
                exit_price = float(hist["close"].iloc[-1])

            if exit_price is None:
                exit_price = float(hist["close"].iloc[-1])

            if direction == "BUY":
                pnl = round((exit_price - entry) * shares, 2)
            else:
                pnl = round((entry - exit_price) * shares, 2)

            pnl_pct = round(pnl / (entry * shares) * 100, 2) if entry * shares > 0 else 0

            closed_rows.append({
                "signal_id":   row["signal_id"],
                "exit_price":  round(exit_price, 4),
                "exit_reason": exit_reason,
                "pnl":         pnl,
                "pnl_pct":     pnl_pct,
                "closed_date": today.isoformat(),
            })

            logger.info("Journal: fechado %s %s -> %s pnl=R$%.2f",
                        ticker, direction, exit_reason, pnl)

        except Exception as e:
            logger.warning("Journal: erro ao fechar %s: %s", ticker, e)

    # Atualiza o CSV
    if closed_rows:
        closed_map = {r["signal_id"]: r for r in closed_rows}
        for i, row in df.iterrows():
            if row["signal_id"] in closed_map:
                c = closed_map[row["signal_id"]]
                df.at[i, "status"]      = "closed"
                df.at[i, "exit_price"]  = c["exit_price"]
                df.at[i, "exit_reason"] = c["exit_reason"]
                df.at[i, "pnl"]         = c["pnl"]
                df.at[i, "pnl_pct"]     = c["pnl_pct"]
                df.at[i, "closed_date"] = c["closed_date"]
        df.to_csv(JOURNAL_PATH, index=False)

    # Resumo semanal — trades fechados nos ultimos 7 dias
    week_ago   = (today - timedelta(days=7)).isoformat()
    df["closed_date"] = df["closed_date"].fillna("").astype(str)
    df_closed  = df[
        (df["status"] == "closed") &
        (df["closed_date"] >= week_ago)
    ].copy()

    if df_closed.empty:
        return {
            "trades":    0,
            "wins":      0,
            "losses":    0,
            "pnl_total": 0.0,
            "win_rate":  0.0,
            "capital":   capital,
            "message":   "Nenhum trade fechado esta semana.",
        }

    df_closed["pnl"] = pd.to_numeric(df_closed["pnl"], errors="coerce").fillna(0)
    total_pnl  = round(df_closed["pnl"].sum(), 2)
    wins       = int((df_closed["pnl"] > 0).sum())
    losses     = int((df_closed["pnl"] <= 0).sum())
    n_trades   = len(df_closed)
    win_rate   = round(wins / n_trades * 100, 1) if n_trades > 0 else 0
    new_capital= round(capital + total_pnl, 2)
    ret_pct    = round(total_pnl / capital * 100, 2) if capital > 0 else 0

    tp_count  = int((df_closed["exit_reason"] == "TP").sum())
    sl_count  = int((df_closed["exit_reason"] == "SL").sum())
    tm_count  = int((df_closed["exit_reason"] == "TIME").sum())

    best  = df_closed.loc[df_closed["pnl"].idxmax()]
    worst = df_closed.loc[df_closed["pnl"].idxmin()]

    return {
        "trades":    n_trades,
        "wins":      wins,
        "losses":    losses,
        "pnl_total": total_pnl,
        "ret_pct":   ret_pct,
        "win_rate":  win_rate,
        "capital":   new_capital,
        "tp_count":  tp_count,
        "sl_count":  sl_count,
        "tm_count":  tm_count,
        "best":      {"ticker": best["ticker"], "pnl": float(best["pnl"])},
        "worst":     {"ticker": worst["ticker"], "pnl": float(worst["pnl"])},
        "rows":      df_closed.to_dict("records"),
    }


def journal_send_weekly_summary(capital: float, notifier) -> None:
    """Calcula resultado semanal e envia via Telegram."""
    result = journal_weekly_close(capital=capital)

    if not result or result.get("trades", 0) == 0:
        notifier.alert("Resumo semanal: nenhum trade fechado esta semana.")
        return

    sinal    = "+" if result["pnl_total"] >= 0 else ""
    icone    = "📈" if result["pnl_total"] >= 0 else "📉"
    best     = result["best"]
    worst    = result["worst"]

    msg = (
        f"{icone} <b>RESUMO SEMANAL</b>\n"
        f"\n"
        f"Capital atual:  <b>R$ {result['capital']:,.2f}</b>\n"
        f"PnL da semana:  <b>R$ {sinal}{result['pnl_total']:.2f} ({sinal}{result['ret_pct']:.2f}%)</b>\n"
        f"\n"
        f"Trades:  {result['trades']}  |  Acerto: {result['win_rate']:.0f}%\n"
        f"TP: {result['tp_count']}  |  SL: {result['sl_count']}  |  TIME: {result['tm_count']}\n"
        f"\n"
        f"Melhor:  {best['ticker']}  R$ {best['pnl']:+.2f}\n"
        f"Pior:    {worst['ticker']}  R$ {worst['pnl']:+.2f}"
    )

    from autotrader.utils.notifier import _send_telegram
    _send_telegram(msg)
    logger.info("Resumo semanal enviado: %d trades, PnL=R$%.2f", result["trades"], result["pnl_total"])
