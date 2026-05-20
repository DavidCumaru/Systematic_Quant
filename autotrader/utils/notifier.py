"""
notifier.py
===========
Envia notificações de sinais e alertas via Telegram + log local.

Configuração:
  Em config.py, preencha TELEGRAM_TOKEN e TELEGRAM_CHAT_ID.
  Se vazios, notificações vão apenas para o log (modo silencioso).
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

try:
    from autotrader.config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds between retries


def _send_telegram(text: str) -> bool:
    """Envia mensagem via Telegram Bot API com retry. Retorna True se enviado."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            data = resp.json()

            if data.get("ok"):
                return True

            # API retornou erro HTTP (4xx/5xx)
            code = data.get("error_code", resp.status_code)
            desc = data.get("description", "sem descrição")

            # Rate limit — espera o tempo indicado pela API antes de tentar novamente
            if resp.status_code == 429:
                retry_after = data.get("parameters", {}).get("retry_after", _RETRY_DELAY)
                logger.warning(
                    "Telegram rate limit (tentativa %d/%d). Aguardando %ss.",
                    attempt, _MAX_RETRIES, retry_after,
                )
                time.sleep(retry_after)
                continue

            # Erros 4xx (exceto 429) são definitivos — não adianta repetir
            if 400 <= code < 500:
                logger.warning(
                    "Telegram API erro %s: %s  (mensagem descartada)",
                    code, desc,
                )
                return False

            # Erros 5xx — servidor do Telegram com problema, vale tentar novamente
            logger.warning(
                "Telegram servidor erro %s: %s  (tentativa %d/%d)",
                code, desc, attempt, _MAX_RETRIES,
            )

        except requests.exceptions.Timeout:
            logger.warning(
                "Telegram timeout (tentativa %d/%d)",
                attempt, _MAX_RETRIES,
            )
        except requests.exceptions.ConnectionError as e:
            logger.warning(
                "Telegram conexão falhou (tentativa %d/%d): %s",
                attempt, _MAX_RETRIES, e,
            )
        except Exception as e:
            logger.warning("Telegram erro inesperado: %s", e)
            return False

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)

    logger.error(
        "Telegram: falha após %d tentativas — mensagem não enviada.", _MAX_RETRIES
    )
    return False


class Notifier:
    """
    Notificações de sinais e alertas do pipeline.
    Envia para Telegram (se configurado) e sempre para o log local.
    """

    def signal(
        self,
        ticker: str,
        direction: str,
        price: float,
        stop: float,
        tp: float,
        shares: int,
        confidence: float,
        signal_id: str = "",
        broker_mode: str = "paper",
    ) -> bool:
        arrow = "BUY" if direction.upper() == "BUY" else "SELL"
        modo  = "REAL" if broker_mode == "mt5" else ("SIMULADO" if broker_mode == "mt5_dry" else "PAPEL")

        logger.info(
            "[SINAL] %s %s  entrada=%.2f  sl=%.2f  tp=%.2f  acoes=%d  conf=%.0f%%  [%s]%s",
            ticker, arrow, price, stop, tp, shares, confidence * 100, modo,
            f"  id={signal_id}" if signal_id else "",
        )

        risco = round(abs(price - stop) * shares, 2)
        ganho = round(abs(tp - price) * shares, 2)
        icone = "\U0001f7e2" if arrow == "BUY" else "\U0001f534"
        msg = (
            f"{icone} <b>SINAL {arrow} — {ticker}</b>\n"
            f"\n"
            f"Entrada:    <b>R$ {price:.2f}</b>\n"
            f"Stop Loss:  R$ {stop:.2f}  (-{abs(price-stop)/price*100:.1f}%)\n"
            f"Take Profit: R$ {tp:.2f}  (+{abs(tp-price)/price*100:.1f}%)\n"
            f"Acoes:      {shares}x\n"
            f"Risco:      R$ {risco:.2f}  |  Alvo: R$ {ganho:.2f}\n"
            f"Confianca:  {confidence*100:.0f}%\n"
            f"Modo:       {modo}"
        )
        sent = _send_telegram(msg)
        if not sent and (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
            logger.warning("[SINAL] Telegram nao enviado para %s %s", ticker, arrow)
        return True

    def order_filled(self, ticker: str, direction: str, price: float,
                     shares: int, order_id: str = "") -> bool:
        logger.info("[ORDEM] %s %s  preco=%.2f  acoes=%d%s",
                    ticker, direction, price, shares,
                    f"  id={order_id}" if order_id else "")
        msg = (
            f"✅ <b>ORDEM EXECUTADA — {ticker}</b>\n"
            f"Direcao: {direction}  |  Preco: R$ {price:.2f}  |  Acoes: {shares}"
        )
        sent = _send_telegram(msg)
        if not sent and (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
            logger.warning("[ORDEM] Telegram nao enviado para %s %s", ticker, direction)
        return True

    def alert(self, text: str) -> bool:
        logger.info("[ALERTA] %s", text)
        sent = _send_telegram(f"⚠️ {text}")
        if not sent and (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
            logger.warning("[ALERTA] Telegram nao enviado: %s", text[:80])
        return True

    def performance(self, ticker: str, metrics: dict) -> bool:
        keys = [
            ("total_return_pct", "Retorno"),
            ("sharpe_ratio",     "Sharpe"),
            ("max_drawdown_pct", "MaxDD"),
            ("win_rate_pct",     "Acerto"),
            ("n_trades",         "Trades"),
        ]
        parts = "  ".join(
            f"{label}={metrics[k]}"
            for k, label in keys
            if k in metrics
        )
        logger.info("[PERFORMANCE] %s  %s", ticker, parts)
        return True

    def daily_summary(self, equity: float, pnl_day: float, open_positions: list) -> bool:
        """Resumo diário enviado ao fim do pregão."""
        sinal = "+" if pnl_day >= 0 else ""
        icone = "\U0001f4c8" if pnl_day >= 0 else "\U0001f4c9"
        pos_str = "\n".join(
            f"  {p['ticker']} {p['direction']} {p['shares']}x @ R${p['entry_price']:.2f}"
            for p in open_positions
        ) or "  Nenhuma"

        logger.info("[RESUMO] Patrimonio=R$%.2f  PnL_dia=R$%s%.2f  Posicoes=%d",
                    equity, sinal, pnl_day, len(open_positions))

        msg = (
            f"{icone} <b>RESUMO DO DIA</b>\n"
            f"\n"
            f"Patrimonio: <b>R$ {equity:,.2f}</b>\n"
            f"PnL hoje:   <b>R$ {sinal}{pnl_day:.2f}</b>\n"
            f"\n"
            f"<b>Posicoes abertas:</b>\n{pos_str}"
        )
        sent = _send_telegram(msg)
        if not sent and (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
            logger.warning("[RESUMO] Telegram nao enviado")
        return True
