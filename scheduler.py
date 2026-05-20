"""
scheduler.py
============
Automacao diaria do pipeline Systematic Alpha — mercado B3.

Agenda (horario de Brasilia):
  09:30  dias uteis  — atualiza dados (download incremental)
  10:10  dias uteis  — scan de sinais + envia ordens ao MT5
  17:35  dias uteis  — resumo do dia via Telegram
  Sabado 07:00       — retreinamento semanal dos modelos

Uso:
  python scheduler.py            # roda continuamente (deixe aberto ou use Task Scheduler)
  python scheduler.py --agora    # executa o scan imediatamente (teste)

Prerequisitos:
  pip install schedule
  MetaTrader 5 aberto e logado
  TELEGRAM_TOKEN e TELEGRAM_CHAT_ID preenchidos em config.py (opcional)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import schedule
except ImportError:
    print("Instale o schedule: pip install schedule")
    sys.exit(1)

from autotrader.config.settings import (
    BROKER_MODE, INITIAL_EQUITY, LIVE_TICKERS, MODELS_DIR,
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TIMEZONE,
)
from autotrader.data.pipeline import load_data, update_data
from autotrader.features.engineering import build_features
from autotrader.models.trainer import ModelTrainer
from autotrader.execution.engine import ExecutionEngine
from autotrader.execution.broker_mt5 import get_broker
from autotrader.utils.notifier import Notifier
from autotrader.utils.journal import journal_add_signal, journal_send_weekly_summary, journal_open_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
    ],
)
logger   = logging.getLogger("scheduler")
notifier = Notifier()
BRT      = ZoneInfo(TIMEZONE)


# Feriados B3 fixos (ano corrente) — acrescente conforme necessário
_FERIADOS_B3 = {
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03",
    "2026-04-21", "2026-05-01", "2026-06-04", "2026-09-07",
    "2026-10-12", "2026-11-02", "2026-11-15", "2026-11-20",
    "2026-12-25",
}

def _is_market_day() -> bool:
    hoje = datetime.now(BRT)
    if hoje.weekday() >= 5: # sabado/domingo
        return False
    if hoje.strftime("%Y-%m-%d") in _FERIADOS_B3:
        return False
    return True


# Job 1 — Atualiza dados (09:30 BRT)
def job_atualiza_dados() -> None:
    if not _is_market_day():
        return
    logger.info("=== JOB: Atualizacao de dados ===")
    try:
        update_data(tickers=LIVE_TICKERS)
        logger.info("Dados atualizados: %d tickers", len(LIVE_TICKERS))
    except Exception as e:
        logger.error("Falha na atualizacao: %s", e, exc_info=True)
        notifier.alert(f"ERRO: atualizacao de dados falhou — {e}")


# Job 2 — Scan de sinais e envio de ordens (10:10 BRT)
def job_scan_sinais() -> None:
    if not _is_market_day():
        return

    agora = datetime.now(BRT).strftime("%Y-%m-%d %H:%M BRT")
    logger.info("=== JOB: Scan de sinais  %s ===", agora)

    # Inicializa broker (MT5 ou paper)
    try:
        broker = get_broker(
            BROKER_MODE,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )
        info = broker.account_info()
        if info:
            equity = float(info.get("equity", INITIAL_EQUITY))
            logger.info("Broker: %s | Conta: %s | Equity: R$ %.2f",
                        BROKER_MODE, info.get("login",""), equity)
        else:
            equity = INITIAL_EQUITY
            logger.warning("Broker nao retornou info de conta — usando equity padrao")
    except Exception as e:
        logger.error("Falha ao conectar broker: %s", e)
        notifier.alert(f"ERRO: broker nao conectou — {e}")
        equity = INITIAL_EQUITY
        broker = None

    sinais_gerados = []
    open_tickers = journal_open_tickers()   # tickers com posicao aberta hoje

    for ticker in LIVE_TICKERS:
        try:
            # Bloqueia novo sinal se ja existe posicao aberta neste ticker
            if ticker in open_tickers:
                logger.info("[%s] Posicao ja aberta — pulando sinal", ticker)
                continue

            # Carrega modelo
            model_path = MODELS_DIR / f"model_final_{ticker}.pkl"
            if not model_path.exists():
                logger.warning("[%s] Modelo nao encontrado — pulando", ticker)
                continue
            trainer = ModelTrainer.load(model_path)

            # Carrega e prepara features
            raw_df = load_data(ticker)
            if raw_df is None or raw_df.empty:
                logger.warning("[%s] Sem dados — pulando", ticker)
                continue

            feat_df = build_features(raw_df, ticker=ticker)
            if feat_df is None or feat_df.empty:
                logger.warning("[%s] Sem features — pulando", ticker)
                continue

            latest = feat_df.tail(200)   # janela de aquecimento

            # Gera sinal
            engine = ExecutionEngine(trainer=trainer, equity=equity)
            signal = engine.run_live_scan(latest, ticker=ticker)

            if signal:
                logger.info("[%s] SINAL: %s @ R$%.2f  conf=%.1f%%",
                            ticker, signal["direction"], signal["entry_price"],
                            signal["confidence"] * 100)

                # Injeta equity no sinal para recalculo de tamanho apos gap adjustment
                signal["_equity"] = equity

                # Envia ordem ao broker (gap protection + recalculo SL/TP automatico)
                result = engine.submit_paper_order(signal)

                if result is None:
                    logger.info("[%s] Sinal cancelado pela gap protection", ticker)
                    continue

                status = result.get("status", "?")
                logger.info("[%s] Ordem enviada: status=%s", ticker, status)

                # Notifica com preco real apos ajuste
                notifier.signal(
                    ticker=ticker,
                    direction=signal["direction"],
                    price=float(result.get("fill_price", signal["entry_price"])),
                    stop=signal["stop_loss"],
                    tp=signal["take_profit"],
                    shares=int(result.get("shares", signal["position_size"])),
                    confidence=signal["confidence"],
                    signal_id=signal.get("signal_id", ""),
                    broker_mode=BROKER_MODE,
                )
                # Registra no journal de trades
                journal_add_signal(signal, capital=equity)
                sinais_gerados.append(signal)
            else:
                logger.info("[%s] Sem sinal hoje", ticker)

        except Exception as e:
            logger.error("[%s] Erro no scan: %s", ticker, e, exc_info=True)
            notifier.alert(f"ERRO no scan de {ticker}: {e}")

    if not sinais_gerados:
        logger.info("Nenhum sinal gerado hoje.")
    else:
        logger.info("Total de sinais: %d", len(sinais_gerados))


# Job 3 — Resumo do dia (17:35 BRT)
def job_resumo_dia() -> None:
    if not _is_market_day():
        return
    logger.info("=== JOB: Resumo do dia ===")
    try:
        broker = get_broker(BROKER_MODE, login=MT5_LOGIN,
                            password=MT5_PASSWORD, server=MT5_SERVER)
        info = broker.account_info() or {}
        equity = float(info.get("equity", INITIAL_EQUITY))
        balance = float(info.get("balance", INITIAL_EQUITY))
        pnl_dia = equity - balance
        posicoes = broker.get_open_positions()
        notifier.daily_summary(equity=equity, pnl_day=pnl_dia, open_positions=posicoes)

        # Sexta-feira: envia resumo semanal do journal
        from datetime import datetime
        from zoneinfo import ZoneInfo
        if datetime.now(ZoneInfo(TIMEZONE)).weekday() == 4:  # 4 = sexta
            logger.info("Sexta-feira: enviando resumo semanal...")
            journal_send_weekly_summary(capital=equity, notifier=notifier)

    except Exception as e:
        logger.error("Resumo falhou: %s", e)


# Job 4 — Retreinamento semanal (Sabado 07:00 BRT)
def job_retreinamento() -> None:
    logger.info("=== JOB: Retreinamento semanal ===")
    notifier.alert("Retreinamento semanal iniciado...")
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, "main.py", "--mode", "research",
             "--tickers"] + LIVE_TICKERS,
            cwd=str(Path(__file__).resolve().parent),
        )
        msg = "Retreinamento concluido." if r.returncode == 0 else "Retreinamento terminou com erros."
        notifier.alert(msg)
        logger.info(msg)
    except Exception as e:
        logger.error("Retreinamento falhou: %s", e, exc_info=True)
        notifier.alert(f"ERRO no retreinamento: {e}")

# Entry point
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agora", action="store_true",
                        help="Executa o scan imediatamente (modo teste)")
    args = parser.parse_args()

    if args.agora:
        logger.info("Modo --agora: executando scan imediato...")
        job_atualiza_dados()
        job_scan_sinais()
        return

    logger.info("=" * 55)
    logger.info("Systematic Alpha — Scheduler B3")
    logger.info("Tickers: %s", ", ".join(LIVE_TICKERS))
    logger.info("Broker:  %s", BROKER_MODE)
    logger.info("=" * 55)

    schedule.every().day.at("09:30").do(job_atualiza_dados)
    schedule.every().day.at("10:10").do(job_scan_sinais)
    schedule.every().day.at("17:35").do(job_resumo_dia)
    schedule.every().saturday.at("07:00").do(job_retreinamento)

    logger.info("Agendamentos:")
    logger.info("09:30 BRT (dias uteis) — atualizacao de dados")
    logger.info("10:10 BRT (dias uteis) — scan de sinais + ordens MT5")
    logger.info("17:35 BRT (dias uteis) — resumo do dia")
    logger.info("Sabado 07:00 — retreinamento semanal")
    logger.info("Rodando... (Ctrl+C para parar)\n")

    notifier.alert(
        f"Systematic Alpha iniciado\n"
        f"Tickers: {', '.join(LIVE_TICKERS)}\n"
        f"Broker: {BROKER_MODE}"
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
