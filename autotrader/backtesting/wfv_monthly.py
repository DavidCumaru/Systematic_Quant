"""
wfv_monthly.py
=======================
Walk-Forward out-of-sample backtest mês a mês.
Para cada mês de teste, treina APENAS com dados anteriores — sem vazamento.
"""
import warnings, logging, time
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Desativa Optuna para velocidade (treina com hiperparâmetros padrão)
import autotrader.config.settings as _cfg
_cfg.USE_OPTUNA    = False
_cfg.OPTUNA_TRIALS = 0

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta
from autotrader.models.trainer import ModelTrainer
from autotrader.features.engineering import build_features
from autotrader.data.pipeline import load_data
from autotrader.risk.management import PositionSizer
from autotrader.labeling.barriers import apply_triple_barrier

TICKERS        = ["BBAS3.SA", "PRIO3.SA", "WEGE3.SA", "PETR4.SA"]
INITIAL_EQUITY = 1000.0
SL_PCT         = 0.010
TP_PCT         = 0.020
TIME_STOP_BARS = 3
TRAIN_MONTHS   = 24
TZ             = "America/Sao_Paulo"
TEST_START     = pd.Timestamp("2023-01-01", tz=TZ)
TEST_END       = pd.Timestamp("2026-03-31", tz=TZ)

MESES_PT = {
    "01":"Jan","02":"Fev","03":"Mar","04":"Abr","05":"Mai","06":"Jun",
    "07":"Jul","08":"Ago","09":"Set","10":"Out","11":"Nov","12":"Dez"
}

sizer = PositionSizer()

# ---------------------------------------------------------------------------
print("Carregando dados...")
raw_dfs  = {}
feat_dfs = {}
for ticker in TICKERS:
    df = load_data(ticker)
    if df is None or df.empty:
        continue
    feat = build_features(df, ticker=ticker)
    if feat is None or feat.empty:
        continue
    raw_dfs[ticker]  = df
    feat_dfs[ticker] = feat
print(f"  {len(feat_dfs)} tickers prontos")

# ---------------------------------------------------------------------------
# Gera janelas mensais
def ts(dt):
    return pd.Timestamp(dt).tz_localize(TZ) if dt.tzinfo is None else dt.tz_convert(TZ)

windows = []
cur = TEST_START
while cur <= TEST_END:
    windows.append((ts(cur - relativedelta(months=TRAIN_MONTHS)),
                    ts(cur),
                    ts(cur),
                    ts(cur + relativedelta(months=1))))
    cur = cur + relativedelta(months=1)

print(f"  {len(windows)} janelas mensais  ({TEST_START.date()} -> {TEST_END.date()})")
print()

# ---------------------------------------------------------------------------
monthly_results = {}

t0 = time.time()
for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    mes_key = te_s.strftime("%Y-%m")
    monthly_results[mes_key] = {"pnl": 0.0, "trades": 0, "wins": 0}

    for ticker in TICKERS:
        if ticker not in feat_dfs:
            continue
        feat_df = feat_dfs[ticker]
        raw_df  = raw_dfs[ticker]

        train_feat = feat_df[(feat_df.index >= tr_s) & (feat_df.index < tr_e)]
        if len(train_feat) < 100:
            continue

        labeled = apply_triple_barrier(raw_df)
        if labeled is None or labeled.empty:
            continue
        train_labeled = labeled[(labeled.index >= tr_s) & (labeled.index < tr_e)]
        if len(train_labeled) < 100:
            continue

        train_df = train_feat.join(train_labeled[["label"]], how="inner").dropna(subset=["label"])
        if len(train_df) < 50:
            continue

        trainer = ModelTrainer()
        try:
            trainer.fit(train_df)
        except Exception:
            continue

        test_feat = feat_df[(feat_df.index >= te_s) & (feat_df.index < te_e)]
        if test_feat.empty:
            continue

        test_raw = raw_df[(raw_df.index >= te_s) & (raw_df.index < te_e)]

        open_pos = None
        for date in test_feat.index:
            # --- verifica saída ---
            if open_pos is not None:
                if date in test_raw.index:
                    bar_h = float(test_raw.loc[date, "high"])
                    bar_l = float(test_raw.loc[date, "low"])
                    bar_c = float(test_raw.loc[date, "close"])
                else:
                    bar_c = float(test_feat.loc[date, "close"])
                    bar_h = bar_c * 1.005
                    bar_l = bar_c * 0.995

                bars_held   = open_pos["bars_held"] + 1
                hit_sl = hit_tp = False
                exit_price  = bar_c

                if open_pos["direction"] == "BUY":
                    if bar_l <= open_pos["sl"]:
                        hit_sl = True; exit_price = open_pos["sl"]
                    elif bar_h >= open_pos["tp"]:
                        hit_tp = True; exit_price = open_pos["tp"]
                else:
                    if bar_h >= open_pos["sl"]:
                        hit_sl = True; exit_price = open_pos["sl"]
                    elif bar_l <= open_pos["tp"]:
                        hit_tp = True; exit_price = open_pos["tp"]

                if hit_sl or hit_tp or bars_held >= TIME_STOP_BARS:
                    if open_pos["direction"] == "BUY":
                        pnl = (exit_price - open_pos["entry"]) * open_pos["shares"]
                    else:
                        pnl = (open_pos["entry"] - exit_price) * open_pos["shares"]
                    monthly_results[mes_key]["pnl"]    += pnl
                    monthly_results[mes_key]["trades"] += 1
                    if pnl > 0:
                        monthly_results[mes_key]["wins"] += 1
                    open_pos = None

            # --- gera sinal ---
            if open_pos is None:
                row = test_feat.loc[[date]]
                try:
                    pred = int(trainer.predict(row)[0])
                except Exception:
                    continue
                if pred == 0:
                    continue
                proba_df = trainer.predict_proba(row)
                proba = float(proba_df[pred].iloc[0]) if pred in proba_df.columns else 0.5
                if proba < 0.48:
                    continue
                close_px  = float(row["close"].iloc[0])
                direction = "BUY" if pred == 1 else "SELL"
                sl = round(close_px * (1 - SL_PCT if direction == "BUY" else 1 + SL_PCT), 2)
                tp = round(close_px * (1 + TP_PCT if direction == "BUY" else 1 - TP_PCT), 2)
                cap_now = INITIAL_EQUITY + sum(v["pnl"] for v in monthly_results.values())
                shares  = sizer.shares(cap_now, close_px, SL_PCT)
                if shares <= 0:
                    continue
                open_pos = {
                    "direction": direction, "entry": close_px,
                    "sl": sl, "tp": tp, "shares": shares, "bars_held": 0,
                }

    elapsed = time.time() - t0
    pct     = (wi + 1) / len(windows) * 100
    eta     = elapsed / (wi + 1) * (len(windows) - wi - 1)
    mes_pnl = monthly_results[mes_key]["pnl"]
    print(f"[{pct:5.1f}%] {mes_key}  PnL=R${mes_pnl:>+8.2f}  {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

# ---------------------------------------------------------------------------
print()
print(f"WALK-FORWARD OUT-OF-SAMPLE — Capital inicial: R${INITIAL_EQUITY:,.2f}")
print(f"Tickers: {' | '.join(TICKERS)}  | SL=1%  TP=2%  Treino=24m  SEM Optuna")
print("=" * 75)
print(f"{'Mes':<10} {'Trades':<9} {'Acerto':<10} {'PnL':<14} {'Retorno':<10} {'Capital'}")
print("=" * 75)

equity = INITIAL_EQUITY
for mes_key in sorted(monthly_results.keys()):
    r   = monthly_results[mes_key]
    pnl = r["pnl"]
    n   = r["trades"]
    wr  = r["wins"] / n * 100 if n > 0 else 0
    ret = pnl / equity * 100 if equity > 0 else 0
    equity += pnl
    y, m  = mes_key[:4], mes_key[5:]
    nome  = f"{MESES_PT[m]}/{y}"
    bar   = "$" * int(abs(ret) / 0.5) if pnl >= 0 else "-" * int(abs(ret) / 0.5)
    print(f"{nome:<10} {n:<9} {wr:<9.1f}%  R${pnl:>+9.2f}   {ret:>+6.2f}%   R${equity:>9,.2f}  {bar}")

print("=" * 75)
total_ret = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
print(f"TOTAL                         R${equity - INITIAL_EQUITY:>+9,.2f}   {total_ret:>+6.2f}%   R${equity:>9,.2f}")

print()
pos   = [v for v in monthly_results.values() if v["pnl"] > 0]
neg   = [v for v in monthly_results.values() if v["pnl"] <= 0]
best  = max(monthly_results.items(), key=lambda x: x[1]["pnl"])
worst = min(monthly_results.items(), key=lambda x: x[1]["pnl"])

print(f"Meses positivos:  {len(pos)}/{len(monthly_results)}  ({len(pos)/len(monthly_results)*100:.0f}%)")
y, m = best[0][:4],  best[0][5:]
print(f"Melhor mes:       {MESES_PT[m]}/{y}  +R${best[1]['pnl']:.2f}")
y, m = worst[0][:4], worst[0][5:]
print(f"Pior mes:         {MESES_PT[m]}/{y}   R${worst[1]['pnl']:.2f}")
print(f"Tempo total:      {(time.time()-t0)/60:.1f} min")
