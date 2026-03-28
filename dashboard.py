"""
dashboard.py
============
Streamlit dashboard for the Systematic Alpha Research Pipeline.

Run
---
    streamlit run dashboard.py

Tabs
----
  Overview       — portfolio equity, P&L summary, KPIs
  Signals        — live signal feed per ticker
  Paper Broker   — open positions + closed trades + broker equity
  Performance    — per-ticker backtest metrics, equity curves, monthly returns
  Factor & Regime— IC/ICIR, signal decay, regime breakdown
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── project paths ────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
LOGS   = BASE / "logs"
DATA   = BASE / "data"
BROKER = DATA / "paper_broker.json"

st.set_page_config(
    page_title="Systematic Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_signals(ticker: str | None = None) -> pd.DataFrame:
    """Load signal CSVs from logs/. Optionally filter by ticker."""
    frames = []
    pattern = f"signals_{ticker}.csv" if ticker else "signals_*.csv"
    for p in LOGS.glob(pattern):
        try:
            df = pd.read_csv(p, parse_dates=["timestamp"])
            if "ticker" not in df.columns:
                df["ticker"] = p.stem.replace("signals_", "")
            frames.append(df)
        except Exception:
            pass
    # also try signals_output.csv (single-ticker live scan)
    if (BASE / "signals_output.csv").exists():
        try:
            df = pd.read_csv(BASE / "signals_output.csv", parse_dates=["timestamp"])
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["signal_id"] if "signal_id" in frames[0].columns else None
    )
    return combined.sort_values("timestamp", ascending=False)


@st.cache_data(ttl=60)
def load_broker_state() -> dict:
    """Load paper broker JSON state."""
    if not BROKER.exists():
        return {}
    try:
        with open(BROKER, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_equity_curves() -> dict[str, pd.Series]:
    """Reconstruct equity curves from per-ticker signals CSVs if available."""
    curves: dict[str, pd.Series] = {}
    # Look for equity curve data embedded in signals (running equity)
    for p in LOGS.glob("signals_*.csv"):
        ticker = p.stem.replace("signals_", "")
        try:
            df = pd.read_csv(p, parse_dates=["timestamp"])
            if "entry_price" in df.columns and "position_size" in df.columns:
                # crude equity proxy: cumulative notional
                df = df.sort_values("timestamp")
                curves[ticker] = df.set_index("timestamp")["notional_usd"].cumsum()
        except Exception:
            pass
    return curves


def _fmt(val: float | None, decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def _color_metric(val: float, positive_good: bool = True) -> str:
    if val > 0:
        return "normal" if positive_good else "inverse"
    return "inverse" if positive_good else "normal"


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Systematic Alpha")
    st.caption("Research · Paper Trading · Analytics")
    st.divider()

    # Auto-refresh
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if auto_refresh:
        import time
        st.caption(f"Última atualização: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    st.divider()

    # Ticker filter
    all_signals = load_signals()
    available_tickers = (
        sorted(all_signals["ticker"].unique().tolist())
        if not all_signals.empty and "ticker" in all_signals.columns
        else []
    )
    selected_tickers = st.multiselect(
        "Filtrar tickers", options=available_tickers, default=available_tickers
    )

    st.divider()
    st.caption("Broker local — sem API key\nyfinance fill simulation")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_signals, tab_broker, tab_performance, tab_factor = st.tabs([
    "📊 Overview",
    "🔔 Signals",
    "🏦 Paper Broker",
    "📈 Performance",
    "🔬 Factor & Regime",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Portfolio Overview")

    broker = load_broker_state()

    if broker:
        init_eq  = broker.get("initial_equity", 100_000)
        cash     = broker.get("cash", init_eq)
        open_pos = broker.get("open_positions", {})
        closed   = broker.get("closed_trades", [])

        market_val = sum(
            p.get("current_price", p["entry_price"]) * p["qty"]
            for p in open_pos.values()
            if p.get("direction") == "BUY"
        )
        total_eq = cash + market_val
        ret_pct  = (total_eq / init_eq - 1) * 100
        realised = sum(t["gross_pnl"] for t in closed)
        unrealised = sum(p.get("unrealised_pnl", 0) for p in open_pos.values())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Equity Total",   f"${total_eq:,.2f}",
                  f"{ret_pct:+.2f}%")
        c2.metric("Cash",           f"${cash:,.2f}")
        c3.metric("Valor em Mercado",f"${market_val:,.2f}")
        c4.metric("P&L Realizado",  f"${realised:+,.2f}",
                  delta_color="normal" if realised >= 0 else "inverse")
        c5.metric("P&L Não Realizado", f"${unrealised:+,.2f}",
                  delta_color="normal" if unrealised >= 0 else "inverse")

        st.divider()

        # Trade summary
        if closed:
            df_c = pd.DataFrame(closed)
            wins = (df_c["gross_pnl"] > 0).sum()
            total_t = len(df_c)
            win_rate = wins / total_t * 100
            gross_p  = df_c[df_c["gross_pnl"] > 0]["gross_pnl"].sum()
            gross_l  = df_c[df_c["gross_pnl"] < 0]["gross_pnl"].sum().abs()
            pf = gross_p / gross_l if gross_l > 0 else float("inf")

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Trades Fechados", total_t)
            cc2.metric("Win Rate",  f"{win_rate:.1f}%")
            cc3.metric("Profit Factor", f"{pf:.2f}")
            cc4.metric("Posições Abertas", len(open_pos))
    else:
        st.info(
            "Nenhum estado do paper broker encontrado.  "
            "Rode `python main.py --mode live` para gerar sinais e preencher posições."
        )

    # ── Backtest aggregate table (from pipeline logs)
    st.subheader("Backtest Aggregate — Últimos Resultados")
    frames = []
    for p in LOGS.glob("signals_*.csv"):
        ticker = p.stem.replace("signals_", "")
        if selected_tickers and ticker not in selected_tickers:
            continue
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            n = len(df)
            buy_pct = (df["direction"] == "BUY").mean() * 100 if "direction" in df.columns else None
            avg_conf = df["confidence"].mean() * 100 if "confidence" in df.columns else None
            frames.append({"Ticker": ticker, "Sinais": n,
                           "% BUY": _fmt(buy_pct, 1, "%"),
                           "Conf. Média": _fmt(avg_conf, 1, "%")})
        except Exception:
            pass
    if frames:
        st.dataframe(pd.DataFrame(frames).set_index("Ticker"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.header("Feed de Sinais")

    sigs = load_signals()
    if sigs.empty:
        st.warning("Nenhum sinal encontrado em logs/. Rode o pipeline primeiro.")
    else:
        # Filter
        if selected_tickers:
            sigs = sigs[sigs["ticker"].isin(selected_tickers)]

        # Direction filter
        dir_filter = st.radio("Direção", ["Todos", "BUY", "SELL"], horizontal=True)
        if dir_filter != "Todos":
            sigs = sigs[sigs["direction"] == dir_filter]

        # Confidence slider
        if "confidence" in sigs.columns:
            min_conf = float(sigs["confidence"].min())
            max_conf = float(sigs["confidence"].max())
            if min_conf < max_conf:
                conf_range = st.slider(
                    "Confiança mínima", min_conf, max_conf,
                    (min_conf, max_conf), format="%.2f"
                )
                sigs = sigs[sigs["confidence"].between(*conf_range)]

        st.caption(f"{len(sigs)} sinais exibidos")

        # Color-code direction
        def _style_direction(val: str) -> str:
            return "color: #00c853" if val == "BUY" else "color: #ff1744"

        styled = sigs.style.map(_style_direction, subset=["direction"])
        st.dataframe(styled, use_container_width=True, height=420)

        # Distribuição temporal
        if "timestamp" in sigs.columns and len(sigs) > 1:
            st.subheader("Distribuição Temporal de Sinais")
            sigs_t = sigs.copy()
            sigs_t["date"] = pd.to_datetime(sigs_t["timestamp"]).dt.date
            cnt = sigs_t.groupby(["date", "ticker"]).size().reset_index(name="count")
            fig = px.bar(cnt, x="date", y="count", color="ticker",
                         title="Sinais por Dia", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Confiança por ticker
        if "confidence" in sigs.columns and "ticker" in sigs.columns:
            st.subheader("Distribuição de Confiança")
            fig2 = px.box(sigs, x="ticker", y="confidence", color="direction",
                          title="Confiança do Modelo por Ticker", height=350)
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PAPER BROKER
# ══════════════════════════════════════════════════════════════════════════════
with tab_broker:
    st.header("Paper Broker — Simulação Local")
    st.caption("Fills simulados com preços reais via yfinance.  Sem API key.")

    broker = load_broker_state()
    if not broker:
        st.info("Estado do broker vazio. Execute `python main.py --mode live` para criar ordens.")
    else:
        # ── Open positions
        st.subheader("Posições Abertas")
        open_pos = broker.get("open_positions", {})
        if open_pos:
            rows = []
            for ticker, p in open_pos.items():
                curr = p.get("current_price", p["entry_price"])
                entry = p["entry_price"]
                qty   = p["qty"]
                direction = p["direction"]
                upnl  = p.get("unrealised_pnl", round((curr - entry) * qty * (1 if direction == "BUY" else -1), 2))
                rows.append({
                    "Ticker":     ticker,
                    "Direção":    direction,
                    "Qtd":        qty,
                    "Entrada":    f"${entry:.2f}",
                    "Atual":      f"${curr:.2f}",
                    "Stop":       f"${p['stop_loss']:.2f}" if p.get("stop_loss") else "—",
                    "Target":     f"${p['take_profit']:.2f}" if p.get("take_profit") else "—",
                    "P&L Não Real.": f"${upnl:+,.2f}",
                    "Confiança":  f"{p.get('confidence', 0):.1%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Sem posições abertas no momento.")

        st.divider()

        # ── Closed trades
        st.subheader("Trades Fechados")
        closed = broker.get("closed_trades", [])
        if closed:
            df_c = pd.DataFrame(closed)
            df_c["pnl_color"] = df_c["gross_pnl"].apply(
                lambda x: "Win" if x > 0 else "Loss"
            )

            # Summary KPIs
            wins    = (df_c["gross_pnl"] > 0).sum()
            total_t = len(df_c)
            gross_p = df_c[df_c["gross_pnl"] > 0]["gross_pnl"].sum()
            gross_l = df_c[df_c["gross_pnl"] < 0]["gross_pnl"].sum().abs()
            pf = gross_p / gross_l if gross_l > 0 else float("inf")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Trades", total_t)
            k2.metric("Win Rate", f"{wins / total_t * 100:.1f}%")
            k3.metric("Profit Factor", f"{pf:.2f}")
            k4.metric("P&L Total", f"${df_c['gross_pnl'].sum():+,.2f}")

            # P&L chart
            df_plot = df_c.copy()
            if "closed_at" in df_plot.columns:
                df_plot["closed_at"] = pd.to_datetime(df_plot["closed_at"])
                df_plot = df_plot.sort_values("closed_at")
                df_plot["cumulative_pnl"] = df_plot["gross_pnl"].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot["closed_at"], y=df_plot["cumulative_pnl"],
                    mode="lines+markers",
                    line=dict(color="#00c853", width=2),
                    name="P&L Acumulado",
                ))
                fig.update_layout(title="Curva de P&L — Paper Broker",
                                  xaxis_title="Data", yaxis_title="P&L ($)",
                                  height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Per-trade bars
            if "closed_at" in df_plot.columns:
                fig2 = px.bar(df_plot, x="closed_at", y="gross_pnl",
                              color="pnl_color",
                              color_discrete_map={"Win": "#00c853", "Loss": "#ff1744"},
                              title="P&L por Trade", height=280)
                st.plotly_chart(fig2, use_container_width=True)

            # Table
            disp_cols = [c for c in ["trade_id","ticker","direction","qty","entry_price",
                                     "exit_price","gross_pnl","exit_reason","closed_at"]
                         if c in df_c.columns]
            st.dataframe(df_c[disp_cols].sort_values(
                "closed_at", ascending=False) if "closed_at" in df_c.columns
                else df_c[disp_cols], use_container_width=True)
        else:
            st.info("Nenhum trade fechado ainda.")

        # ── Pending orders
        pending = broker.get("pending", [])
        if pending:
            st.divider()
            st.subheader(f"Ordens Pendentes ({len(pending)})")
            st.dataframe(pd.DataFrame(pending), use_container_width=True)

        # ── Reset button
        st.divider()
        with st.expander("Configurações do Broker", expanded=False):
            st.warning("Reset apaga todos os trades e posições do paper broker.")
            if st.button("Reset Paper Broker", type="secondary"):
                import sys
                sys.path.insert(0, str(BASE))
                from paper_broker import PaperBroker
                init_eq = broker.get("initial_equity", 100_000)
                PaperBroker(initial_equity=init_eq).reset(init_eq)
                st.success("Broker resetado.")
                st.cache_data.clear()
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_performance:
    st.header("Performance por Ticker — Backtest")

    # ── Equity curve images
    img_files = sorted(LOGS.glob("equity_curve_*.png"))
    if img_files:
        tickers_img = [p.stem.replace("equity_curve_", "") for p in img_files]
        if selected_tickers:
            img_files   = [p for p, t in zip(img_files, tickers_img) if t in selected_tickers]
            tickers_img = [t for t in tickers_img if t in selected_tickers]

        if img_files:
            st.subheader("Equity Curves")
            cols = st.columns(min(len(img_files), 2))
            for i, (p, t) in enumerate(zip(img_files, tickers_img)):
                with cols[i % 2]:
                    try:
                        img = Image.open(p)
                        st.image(img, caption=t, use_container_width=True)
                    except Exception:
                        st.warning(f"Não foi possível carregar {p.name}")
    else:
        st.info("Rode `python main.py --mode research` para gerar equity curves.")

    st.divider()

    # ── Per-ticker signal stats (proxy for backtest summary)
    st.subheader("Estatísticas de Sinais por Ticker")

    rows = []
    for p in sorted(LOGS.glob("signals_*.csv")):
        ticker = p.stem.replace("signals_", "")
        if selected_tickers and ticker not in selected_tickers:
            continue
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            n_total = len(df)
            n_buy   = (df["direction"] == "BUY").sum() if "direction" in df.columns else "—"
            n_sell  = (df["direction"] == "SELL").sum() if "direction" in df.columns else "—"
            avg_c   = df["confidence"].mean() if "confidence" in df.columns else None
            avg_not = df["notional_usd"].mean() if "notional_usd" in df.columns else None
            rows.append({
                "Ticker":          ticker,
                "Total Sinais":    n_total,
                "BUY":             n_buy,
                "SELL":            n_sell,
                "Conf. Média":     f"{avg_c:.2%}" if avg_c else "—",
                "Notional Médio":  f"${avg_not:,.0f}" if avg_not else "—",
            })
        except Exception:
            pass

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    # ── Notional distribution
    all_sigs = load_signals()
    if not all_sigs.empty and "notional_usd" in all_sigs.columns:
        st.subheader("Distribuição de Notional por Ticker")
        if selected_tickers:
            all_sigs = all_sigs[all_sigs["ticker"].isin(selected_tickers)]
        fig = px.box(all_sigs, x="ticker", y="notional_usd",
                     color="ticker", title="Notional USD", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Confidence histogram
    if not all_sigs.empty and "confidence" in all_sigs.columns:
        st.subheader("Histograma de Confiança")
        fig2 = px.histogram(all_sigs, x="confidence", color="ticker",
                            nbins=30, opacity=0.7,
                            title="Distribuição de Confiança do Modelo", height=320)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FACTOR & REGIME
# ══════════════════════════════════════════════════════════════════════════════
with tab_factor:
    st.header("Factor Analysis & Regime")

    # ── Parse pipeline.log for IC/ICIR and regime data
    log_path = LOGS / "pipeline.log"

    if not log_path.exists():
        st.info("Rode o pipeline completo (`--mode research`) para ver análise de fatores.")
    else:
        # Extract relevant lines
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                log_lines = f.readlines()
        except Exception:
            log_lines = []

        # ── IC Summary
        ic_rows = []
        for line in log_lines:
            if "IC Mean" in line or "ICIR" in line or "IC mean" in line:
                ic_rows.append(line.strip())

        if ic_rows:
            st.subheader("Information Coefficient (IC / ICIR)")
            # Try to parse structured IC table lines
            parsed = []
            for l in log_lines:
                if "ic_mean" in l.lower() or "icir" in l.lower():
                    parts = l.strip().split(" - ", 2)
                    if len(parts) >= 3:
                        parsed.append(parts[2])
            if parsed:
                st.code("\n".join(parsed[:30]))
            else:
                st.code("\n".join(ic_rows[:20]))

        # ── Regime table
        st.subheader("Performance por Regime")
        regime_sections = []
        in_regime = False
        regime_lines = []
        for line in log_lines:
            if "Performance by regime:" in line:
                in_regime = True
                regime_lines = []
            elif in_regime:
                if line.strip() == "" or "=====" in line:
                    if regime_lines:
                        regime_sections.append(regime_lines[:])
                    in_regime = False
                    regime_lines = []
                else:
                    regime_lines.append(line.rstrip())

        if regime_sections:
            for i, section in enumerate(regime_sections[:10]):
                content = "\n".join(section)
                st.code(content)
        else:
            st.info("Nenhum dado de regime encontrado no log.")

        # ── Signal decay visualization (if logged)
        decay_data = {}
        for line in log_lines:
            if "IC_" in line and "horizon" in line.lower():
                parts = line.strip().split()
                for p in parts:
                    if p.startswith("IC_") and "=" in p:
                        k, v = p.split("=", 1)
                        try:
                            decay_data[int(k.replace("IC_", ""))] = float(v)
                        except Exception:
                            pass

        if decay_data:
            st.subheader("Signal Decay (IC por Horizonte)")
            decay_df = pd.DataFrame(
                list(decay_data.items()), columns=["Horizon (days)", "IC"]
            ).sort_values("Horizon (days)")
            fig = px.line(decay_df, x="Horizon (days)", y="IC",
                          markers=True,
                          title="IC Decay — quanto o sinal decai ao longo do tempo",
                          height=300)
            fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                          annotation_text="IC=0.05 (meaningful)")
            fig.add_hline(y=0.10, line_dash="dash", line_color="gold",
                          annotation_text="IC=0.10 (institutional)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Last N log lines (diagnostics)
        with st.expander("Últimas 100 linhas do pipeline.log", expanded=False):
            st.code("".join(log_lines[-100:]))


# ── footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Systematic Alpha Research Pipeline · Paper Broker Local · yfinance Data · "
    "LightGBM + Optuna · MLflow · Streamlit"
)
