"""
holdout.py
===================
Backtest honesto no período de holdout: Apr/2025 → Mar/2026.

Regras para evitar viés:
  1. Modelos retreinados APENAS com dados até dez/2024
  2. Features causais (indicadores só olham para trás)
  3. Deduplicação de barras (Yahoo + B3 geravam barras duplas)
  4. Sem Optuna (treino rápido, sem overfitting de hiperparâmetros)
  5. Capital inicial R$1.000 com compounding real
"""
import warnings, logging, time
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

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

# ---------------------------------------------------------------------------
# Seleção final: 8 tickers lucrativos com features completas (técnicas + macro + news)
TICKERS = [
    "PRIO3.SA","LREN3.SA","SUZB3.SA","VIVT3.SA",
    "EQTL3.SA","ITUB4.SA","KLBN11.SA","BBAS3.SA",
]
INITIAL_EQUITY = 1000.0
SL_PCT         = 0.010
TP_PCT         = 0.020
TIME_STOP_BARS = 3
TRAIN_END      = "2024-12-31"   # treina só até aqui
HOLDOUT_START  = "2025-04-01"   # 3 meses de gap extra (jan-mar/2025 = embargo)
HOLDOUT_END    = "2026-03-31"

MESES_PT = {
    "01":"Jan","02":"Fev","03":"Mar","04":"Abr","05":"Mai","06":"Jun",
    "07":"Jul","08":"Ago","09":"Set","10":"Out","11":"Nov","12":"Dez"
}

TZ    = "America/Sao_Paulo"
sizer = PositionSizer()


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove barras duplicadas: mantém uma barra por dia de calendário."""
    df = df.copy()
    # Converte para UTC primeiro para evitar ambiguidade de horário de verão
    if df.index.tzinfo is not None:
        dates = df.index.tz_convert("UTC").normalize().tz_localize(None)
    else:
        dates = df.index.normalize()
    df.index = dates
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
print("=" * 60)
print("HOLDOUT BACKTEST  Apr/2025 -> Mar/2026")
print("Capital inicial: R$1.000  |  SL=1%  TP=2%")
print("=" * 60)
print()
print("Carregando e deduplicando dados...")

raw_dfs  = {}
feat_dfs = {}

for ticker in TICKERS:
    df = load_data(ticker)
    if df is None or df.empty:
        continue
    df   = dedup(df)
    feat = build_features(df, ticker=ticker)
    if feat is None or feat.empty:
        continue
    feat = dedup(feat)
    raw_dfs[ticker]  = df
    feat_dfs[ticker] = feat
    n_month = len(feat[(feat.index >= "2026-03-01") & (feat.index <= "2026-03-31")])
    print(f"  {ticker}: {len(feat)} barras totais  |  mar/2026 = {n_month} barras")

print()

# ---------------------------------------------------------------------------
print("Treinando modelos com dados até dez/2024...")
t0 = time.time()
trainers = {}

for ticker in TICKERS:
    if ticker not in feat_dfs:
        continue
    feat_df = feat_dfs[ticker]
    raw_df  = raw_dfs[ticker]

    train_feat = feat_df[feat_df.index <= TRAIN_END]
    labeled    = apply_triple_barrier(raw_df)
    if labeled is None or labeled.empty:
        continue
    labeled    = dedup(labeled)
    train_lab  = labeled[labeled.index <= TRAIN_END]

    train_df = train_feat.join(train_lab[["label"]], how="inner").dropna(subset=["label"])
    if len(train_df) < 100:
        print(f"  {ticker}: dados insuficientes ({len(train_df)} barras)")
        continue

    trainer = ModelTrainer()
    try:
        trainer.fit(train_df)
        trainers[ticker] = trainer
        print(f"  {ticker}: treinado com {len(train_df)} barras")
    except Exception as e:
        print(f"  {ticker}: erro no treino — {e}")

print(f"  Treino concluído em {time.time()-t0:.0f}s")
print()

# ---------------------------------------------------------------------------
# Gera janelas mensais do holdout
windows = []
cur = pd.Timestamp(HOLDOUT_START)
end = pd.Timestamp(HOLDOUT_END)
while cur <= end:
    windows.append((cur, cur + relativedelta(months=1)))
    cur += relativedelta(months=1)

print(f"Simulando {len(windows)} meses de holdout ({HOLDOUT_START} -> {HOLDOUT_END})...")
print()

monthly_results = {}
equity = INITIAL_EQUITY
all_trades = []

for te_s, te_e in windows:
    mes_key = te_s.strftime("%Y-%m")
    monthly_results[mes_key] = {"pnl": 0.0, "trades": 0, "wins": 0}

    for ticker in TICKERS:
        if ticker not in trainers:
            continue
        feat_df = feat_dfs[ticker]
        raw_df  = raw_dfs[ticker]
        trainer = trainers[ticker]

        # Filtra período do mês (index já é tz-naive após dedup/normalize)
        te_s_str = te_s.strftime("%Y-%m-%d")
        te_e_str = te_e.strftime("%Y-%m-%d")
        test_feat = feat_df[(feat_df.index >= te_s_str) & (feat_df.index < te_e_str)]
        test_raw  = raw_df[(raw_df.index >= te_s_str) & (raw_df.index < te_e_str)]

        if test_feat.empty:
            continue

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
                    equity += pnl
                    monthly_results[mes_key]["pnl"]    += pnl
                    monthly_results[mes_key]["trades"] += 1
                    if pnl > 0:
                        monthly_results[mes_key]["wins"] += 1
                    reason = "TP" if hit_tp else ("SL" if hit_sl else "TIME")
                    all_trades.append({
                        "mes": mes_key, "ticker": ticker,
                        "direction": open_pos["direction"],
                        "entry": open_pos["entry"], "exit": exit_price,
                        "shares": open_pos["shares"], "pnl": round(pnl, 2),
                        "saida": reason,
                    })
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
                shares = sizer.shares(equity, close_px, SL_PCT)
                if shares <= 0:
                    continue
                open_pos = {
                    "direction": direction, "entry": close_px,
                    "sl": sl, "tp": tp, "shares": shares, "bars_held": 0,
                }

# ---------------------------------------------------------------------------
print("RESULTADO MÊS A MÊS — HOLDOUT OUT-OF-SAMPLE")
print("=" * 72)
print(f"{'Mes':<10} {'Trades':<8} {'Acerto':<10} {'PnL':>12}  {'Retorno':>8}  {'Capital':>12}")
print("=" * 72)

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
    sinal = "+" if pnl >= 0 else ""
    bar   = "#" * max(0, int(abs(ret) / 0.5))
    print(f"{nome:<10} {n:<8} {wr:<9.1f}%  R${pnl:>+9.2f}  {ret:>+7.2f}%  R${equity:>10,.2f}  {bar}")

print("=" * 72)

total_pnl = equity - INITIAL_EQUITY
total_ret = total_pnl / INITIAL_EQUITY * 100
all_n   = sum(r["trades"] for r in monthly_results.values())
all_w   = sum(r["wins"]   for r in monthly_results.values())
global_wr = all_w / all_n * 100 if all_n > 0 else 0

print(f"{'TOTAL':<10} {all_n:<8} {global_wr:<9.1f}%  R${total_pnl:>+9.2f}  {total_ret:>+7.2f}%  R${equity:>10,.2f}")
print()

pos_months = sum(1 for r in monthly_results.values() if r["pnl"] > 0)
best  = max(monthly_results.items(), key=lambda x: x[1]["pnl"])
worst = min(monthly_results.items(), key=lambda x: x[1]["pnl"])

# --- Breakdown por ticker ---
print()
print("PERFORMANCE POR TICKER")
print("=" * 60)
print(f"{'Ticker':<14} {'Trades':>8} {'Acerto':>8} {'PnL':>12} {'Status'}")
print("=" * 60)
ticker_stats = {}
for t in all_trades:
    tk = t["ticker"]
    if tk not in ticker_stats:
        ticker_stats[tk] = {"pnl": 0.0, "trades": 0, "wins": 0}
    ticker_stats[tk]["pnl"]    += t["pnl"]
    ticker_stats[tk]["trades"] += 1
    if t["pnl"] > 0:
        ticker_stats[tk]["wins"] += 1

for tk, s in sorted(ticker_stats.items(), key=lambda x: -x[1]["pnl"]):
    n  = s["trades"]
    wr = s["wins"] / n * 100 if n > 0 else 0
    ok = "LUCRO" if s["pnl"] > 0 else "PERDA"
    print(f"{tk:<14} {n:>8} {wr:>7.1f}%  R${s['pnl']:>+9.2f}  {ok}")
print("=" * 60)

lucrativos = [tk for tk, s in ticker_stats.items() if s["pnl"] > 0]
print(f"Lucrativos: {len(lucrativos)}/{len(ticker_stats)}  -> {sorted(lucrativos)}")

print()
print(f"Meses positivos:  {pos_months}/{len(monthly_results)}  ({pos_months/len(monthly_results)*100:.0f}%)")
y, m = best[0][:4],  best[0][5:]
print(f"Melhor mês:       {MESES_PT[m]}/{y}  +R${best[1]['pnl']:.2f}")
y, m = worst[0][:4], worst[0][5:]
print(f"Pior mês:         {MESES_PT[m]}/{y}   R${worst[1]['pnl']:.2f}")

wins_all  = [t for t in all_trades if t["pnl"] > 0]
loss_all  = [t for t in all_trades if t["pnl"] <= 0]
tp_count  = sum(1 for t in all_trades if t["saida"] == "TP")
sl_count  = sum(1 for t in all_trades if t["saida"] == "SL")
tm_count  = sum(1 for t in all_trades if t["saida"] == "TIME")

if wins_all:  print(f"Ganho médio:      R${sum(t['pnl'] for t in wins_all)/len(wins_all):.2f}")
if loss_all:  print(f"Perda média:      R${sum(t['pnl'] for t in loss_all)/len(loss_all):.2f}")
print(f"Saídas TP/SL/TIME: {tp_count} / {sl_count} / {tm_count}")
print(f"Tempo total:       {(time.time()-t0)/60:.1f} min")
