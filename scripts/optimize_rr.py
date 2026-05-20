"""
optimize_rr.py — testa combinacoes de TP/SL/TIME para encontrar o melhor R:R
Treino: dados ate 31/Dez/2024   Teste: Jan-Dez/2025
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import autotrader.config.settings as _cfg
_cfg.USE_OPTUNA = False

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from autotrader.models.trainer import ModelTrainer
from autotrader.features.engineering import build_features
from autotrader.data.pipeline import load_data
from autotrader.labeling.barriers import apply_triple_barrier
from autotrader.config.settings import LIVE_TICKERS, INITIAL_EQUITY, RISK_PER_TRADE, MIN_PROBA_THRESHOLD

TZ = "America/Sao_Paulo"
TRAIN_END  = pd.Timestamp("2024-12-31", tz=TZ)
TEST_START = pd.Timestamp("2025-01-01", tz=TZ)
TEST_END = pd.Timestamp("2025-12-31", tz=TZ)

# Combinacoes a testar
CONFIGS = [
    {"name": "Atual (ATR 1.0/1.5 TIME=3)", "atr": True,  "tp_m": 1.0, "sl_m": 1.5, "time": 3},
    {"name": "Opcao A (Fix 2%/1% TIME=5)", "atr": False, "tp_p": 0.02, "sl_p": 0.01, "time": 5},
    {"name": "Opcao B (Fix 3%/1% TIME=5)", "atr": False, "tp_p": 0.03, "sl_p": 0.01, "time": 5},
    {"name": "Opcao C (Fix 3%/1% TIME=3)", "atr": False, "tp_p": 0.03, "sl_p": 0.01, "time": 3},
    {"name": "Opcao D (ATR 2.0/1.0 TIME=5)", "atr": True,  "tp_m": 2.0, "sl_m": 1.0, "time": 5},
]

print("Treinando modelos (corte 31/Dez/2024)...")
models = {}
feat_all = {}

for ticker in LIVE_TICKERS:
    raw = load_data(ticker)
    if raw is None or raw.empty: continue
    feat = build_features(raw, ticker=ticker)
    if feat is None or feat.empty: continue
    feat_all[ticker] = feat

    train = feat[feat.index <= TRAIN_END].copy()
    if len(train) < 200: continue
    labeled = apply_triple_barrier(train).dropna(subset=["label"])
    if len(labeled) < 100: continue
    try:
        trainer = ModelTrainer()
        trainer.fit(labeled)
        models[ticker] = trainer
    except: continue

print(f"{len(models)} modelos prontos\n")

def run_config(cfg):
    all_trades = []
    for ticker, trainer in models.items():
        feat = feat_all[ticker]
        test = feat[(feat.index >= TEST_START) & (feat.index <= TEST_END)].copy()
        if test.empty: continue

        feat_cols = trainer.feature_cols
        for c in feat_cols:
            if c not in test.columns: test[c] = 0.0

        proba_df = trainer.predict_proba(test)
        p_long = proba_df[1].values  if 1  in proba_df.columns else np.zeros(len(test))
        p_short = proba_df[-1].values if -1 in proba_df.columns else np.zeros(len(test))

        close_arr = test["close"].values
        high_arr = test["high"].values
        low_arr = test["low"].values
        atr_arr = test["atr"].values if "atr" in test.columns else np.zeros(len(test))
        dates = test.index
        TIME = cfg["time"]

        i = 0
        while i < len(test):
            pl = p_long[i]; ps = p_short[i]
            if pl >= MIN_PROBA_THRESHOLD and pl >= ps:
                direction = "BUY";  confidence = pl
            elif ps >= MIN_PROBA_THRESHOLD and ps > pl:
                direction = "SELL"; confidence = ps
            else:
                i += 1; continue

            entry = close_arr[i]
            atr = atr_arr[i]

            if cfg["atr"] and atr > 0:
                tp_dist = atr * cfg["tp_m"]
                sl_dist = atr * cfg["sl_m"]
            else:
                tp_dist = entry * cfg["tp_p"]
                sl_dist = entry * cfg["sl_p"]

            tp = entry + tp_dist if direction == "BUY" else entry - tp_dist
            sl = entry - sl_dist if direction == "BUY" else entry + sl_dist

            risk_r = abs(entry - sl)
            if risk_r <= 0: i += 1; continue
            shares = max(1, int((INITIAL_EQUITY * RISK_PER_TRADE) / risk_r))

            exit_price = close_arr[min(i + TIME, len(close_arr)-1)]
            exit_reason = "TIME"
            exit_bar = min(i + TIME, len(close_arr)-1)

            for j in range(i+1, min(i+1+TIME, len(close_arr))):
                if direction == "BUY":
                    if low_arr[j]  <= sl: exit_price=sl; exit_reason="SL"; exit_bar=j; break
                    if high_arr[j] >= tp: exit_price=tp; exit_reason="TP"; exit_bar=j; break
                else:
                    if high_arr[j] >= sl: exit_price=sl; exit_reason="SL"; exit_bar=j; break
                    if low_arr[j]  <= tp: exit_price=tp; exit_reason="TP"; exit_bar=j; break

            pnl = (exit_price-entry)*shares if direction=="BUY" else (entry-exit_price)*shares
            corr = entry * shares * 0.00025 * 2
            pnl -= corr

            all_trades.append({
                "mes": dates[i].strftime("%Y-%m"),
                "ticker": ticker,
                "pnl": round(pnl, 2),
                "exit_reason": exit_reason,
            })
            i = exit_bar + 1

    return all_trades

print(f"{'Configuracao':<40} {'Trades':>6} {'Acerto':>7} {'TP':>4} {'SL':>4} {'PnL':>10} {'Retorno':>8} {'MaxDD':>7}")
print("-" * 95)

best_pnl = -999999
best_cfg = None

for cfg in CONFIGS:
    trades = run_config(cfg)
    if not trades:
        print(f"{cfg['name']}: sem trades"); continue

    pnl_total = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    tp_c = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_c = sum(1 for t in trades if t["exit_reason"] == "SL")
    wr = wins / len(trades) * 100

    # Max drawdown mensal
    from collections import defaultdict
    mes_pnl = defaultdict(float)
    for t in trades: mes_pnl[t["mes"]] += t["pnl"]
    equity = INITIAL_EQUITY; peak = equity; mdd = 0.0
    for m in sorted(mes_pnl):
        equity += mes_pnl[m]
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd: mdd = dd

    retorno = pnl_total / INITIAL_EQUITY * 100
    print(f"{cfg['name']:<38} {len(trades):>6} {wr:>6.1f}%"
          f" {tp_c:>4} {sl_c:>4} R${pnl_total:>+8.2f} {retorno:>+7.2f}% {mdd:>5.1f}%")

    if pnl_total > best_pnl:
        best_pnl = pnl_total
        best_cfg = cfg

print()
print(f"Melhor configuracao: {best_cfg['name'].strip()}")
