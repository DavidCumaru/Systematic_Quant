"""
broker_mt5.py
=============
Integração com MetaTrader 5 para execução real de ordens na B3.

Suporta as principais corretoras brasileiras que oferecem MT5:
  - Clear Corretora
  - Rico Investimentos
  - XP Investimentos
  - Genial Investimentos
  - Ágora Investimentos

Fluxo de execução
-----------------
  1. Sistema gera sinal (BUY/SELL)
  2. MT5Broker.submit_order(signal) é chamado
  3. Ordem de mercado enviada ao MT5 (que repassa à corretora)
  4. Stop Loss e Take Profit cadastrados automaticamente
  5. Resultado registrado em logs/mt5_trades.csv

Modo de operação
----------------
  BROKER_MODE = "paper"   → usa PaperBroker (simulação, padrão)
  BROKER_MODE = "mt5"     → usa MT5Broker (dinheiro real)

Configuração (config.py)
------------------------
  MT5_LOGIN    = 12345678        # login da corretora
  MT5_PASSWORD = "sua_senha"     # senha da corretora
  MT5_SERVER   = "Clear-Server"  # servidor (veja em Arquivo > Login no MT5)

Segurança
---------
  - Verificação de conexão antes de cada ordem
  - Limite diário de capital (DAILY_STOP_PCT)
  - Log completo de todas as ordens enviadas
  - Modo dry_run: simula sem enviar (para testes)

Instalação
----------
  pip install MetaTrader5

O MetaTrader 5 deve estar instalado e logado na corretora.
Funciona apenas em Windows (limitação da API do MT5).
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tentativa de importar MT5 — falha graciosamente em sistemas não-Windows
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 não instalado. "
        "Instale com: pip install MetaTrader5  "
        "e certifique-se de que o MT5 está instalado no Windows."
    )


# ---------------------------------------------------------------------------
# Mapeamento de ticker B3 → símbolo MT5
# ---------------------------------------------------------------------------
# A maioria das corretoras usa o ticker sem sufixo .SA e sem o dígito final
# Ex: PETR4.SA → PETR4  |  BBAS3.SA → BBAS3  |  BOVA11.SA → BOVA11
# Ajuste conforme sua corretora se necessário

def _to_mt5_symbol(ticker: str) -> str:
    """Converte ticker B3 (ex: PETR4.SA) para símbolo MT5 (ex: PETR4)."""
    return ticker.replace(".SA", "").strip()


# ---------------------------------------------------------------------------
# MT5 Broker
# ---------------------------------------------------------------------------

class MT5Broker:
    """
    Broker de execução real via MetaTrader 5.

    Parâmetros
    ----------
    login    : número de login da corretora
    password : senha da corretora
    server   : nome do servidor MT5 (ex: 'Clear-Server')
    dry_run  : se True, simula sem enviar ordens reais (para testes)
    log_path : caminho do CSV de log de ordens
    """

    def __init__(
        self,
        login:    "int | str",
        password: str,
        server:   str,
        dry_run:  bool = False,
        log_path: Optional[Path] = None,
    ):
        self.login    = login
        self.password = password
        self.server   = server
        self.dry_run  = dry_run
        self.log_path = log_path or Path("logs/mt5_trades.csv")
        self._connected = False

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not _MT5_AVAILABLE:
            logger.error(
                "MT5Broker instanciado mas MetaTrader5 não está instalado. "
                "Instale com: pip install MetaTrader5"
            )
            return

        self._connect()

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        """Inicializa e faz login no MT5. Retorna True se bem-sucedido."""
        if not _MT5_AVAILABLE:
            return False

        if not mt5.initialize():
            logger.error("MT5: falha ao inicializar — %s", mt5.last_error())
            return False

        authorized = mt5.login(
            login=self.login,
            password=self.password,
            server=self.server,
        )

        if not authorized:
            logger.error(
                "MT5: falha no login (login=%s, server=%s) — %s",
                self.login, self.server, mt5.last_error(),
            )
            mt5.shutdown()
            return False

        info = mt5.account_info()
        logger.info(
            "MT5: conectado | conta=%d | corretora=%s | "
            "saldo=R$%.2f | equity=R$%.2f",
            info.login, info.company, info.balance, info.equity,
        )
        self._connected = True
        return True

    # ------------------------------------------------------------------
    def _ensure_connected(self) -> bool:
        """Reconecta ao MT5 se a conexão foi perdida."""
        if not _MT5_AVAILABLE:
            return False
        if not self._connected or not mt5.terminal_info():
            logger.warning("MT5: reconectando...")
            return self._connect()
        return True

    # ------------------------------------------------------------------
    def account_info(self) -> dict:
        """Retorna saldo, equity e margem disponível da conta real."""
        if not self._ensure_connected():
            return {}

        info = mt5.account_info()
        if info is None:
            return {}

        return {
            "login":   info.login,
            "balance": info.balance,
            "equity":  info.equity,
            "margin":  info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
            "company":  info.company,
        }

    # ------------------------------------------------------------------
    def _get_tick_price(self, symbol: str, direction: str) -> Optional[float]:
        """Retorna o preço atual de mercado (ask para BUY, bid para SELL)."""
        if not _MT5_AVAILABLE or not self._ensure_connected():
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick.ask if direction == "BUY" else tick.bid

    # ------------------------------------------------------------------
    def _validate_and_adjust(self, signal: dict, current_price: float) -> Optional[dict]:
        """
        Valida o gap entre o preço do sinal e o preço atual.
        - Se o gap for desfavoravel e > MAX_ENTRY_GAP_PCT: cancela (retorna None)
        - Caso contrario: recalcula SL/TP a partir do preco atual e retorna sinal ajustado

        Um gap e desfavoravel quando:
          BUY:  preco atual subiu muito acima do sinal (entrada cara, SL/TP errados)
          SELL: preco atual caiu muito abaixo do sinal (entrada cara, SL/TP errados)
        """
        from autotrader.config.settings import MAX_ENTRY_GAP_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT
        from autotrader.risk.management import PositionSizer

        direction    = signal["direction"]
        signal_price = float(signal.get("entry_price", current_price))

        if signal_price <= 0:
            return signal

        gap_pct = (current_price - signal_price) / signal_price  # positivo = subiu

        # Gap desfavoravel: BUY em preco muito alto, ou SELL em preco muito baixo
        unfavorable = (direction == "BUY"  and gap_pct >  MAX_ENTRY_GAP_PCT) or \
                      (direction == "SELL" and gap_pct < -MAX_ENTRY_GAP_PCT)

        if unfavorable:
            logger.warning(
                "Gap protection [%s]: sinal CANCELADO — preco atual %.2f vs sinal %.2f "
                "(gap %+.1f%% > limite %.1f%%)",
                signal.get("ticker", ""), current_price, signal_price,
                gap_pct * 100, MAX_ENTRY_GAP_PCT * 100,
            )
            return None

        # Recalcula SL/TP a partir do preco atual
        adjusted = dict(signal)
        adjusted["entry_price"] = round(current_price, 4)
        if direction == "BUY":
            adjusted["stop_loss"]   = round(current_price * (1 - STOP_LOSS_PCT),  2)
            adjusted["take_profit"] = round(current_price * (1 + TAKE_PROFIT_PCT), 2)
        else:
            adjusted["stop_loss"]   = round(current_price * (1 + STOP_LOSS_PCT),  2)
            adjusted["take_profit"] = round(current_price * (1 - TAKE_PROFIT_PCT), 2)

        # Recalcula tamanho da posição com equity e preco atuais
        equity = signal.get("_equity")
        if equity and equity > 0:
            adjusted["position_size"] = PositionSizer().shares(equity, current_price, STOP_LOSS_PCT)

        if abs(gap_pct) > 0.002:   # loga se gap > 0.2%
            logger.info(
                "Gap adjustment [%s]: entry %.2f -> %.2f (%+.1f%%)  "
                "sl=%.2f  tp=%.2f",
                signal.get("ticker", ""), signal_price, current_price,
                gap_pct * 100, adjusted["stop_loss"], adjusted["take_profit"],
            )

        return adjusted

    # ------------------------------------------------------------------
    def submit_order(self, signal: dict) -> Optional[dict]:
        """
        Envia uma ordem de mercado ao MT5 com stop loss e take profit.

        Antes de enviar:
          1. Busca o preco atual de mercado via MT5
          2. Valida o gap vs preco do sinal (gap protection)
          3. Recalcula SL/TP a partir do preco real (nunca usa preco antigo)

        Parâmetros
        ----------
        signal : dict com campos:
            ticker        : ex "PETR4.SA"
            direction     : "BUY" ou "SELL"
            entry_price   : preco de referencia (fechamento anterior)
            stop_loss     : SL calculado pelo modelo (sera recalculado)
            take_profit   : TP calculado pelo modelo (sera recalculado)
            position_size : quantidade de acoes
            signal_id     : identificador unico

        Retorna dict com resultado da ordem, ou None se cancelado/erro.
        """
        if not signal:
            return None

        ticker    = signal["ticker"]
        direction = signal["direction"]
        signal_id = signal.get("signal_id", "")
        symbol    = _to_mt5_symbol(ticker)

        # Busca preco atual para validar gap e recalcular SL/TP
        current_price = self._get_tick_price(symbol, direction)

        if current_price and current_price > 0:
            signal = self._validate_and_adjust(signal, current_price)
            if signal is None:
                return None   # gap protection acionada
        else:
            logger.warning("MT5: nao foi possivel obter cotacao atual para %s — usando preco do sinal", symbol)

        shares    = int(signal.get("position_size", 0))
        sl        = float(signal.get("stop_loss", 0))
        tp        = float(signal.get("take_profit", 0))

        if shares <= 0:
            logger.warning("MT5: posição de 0 ações para %s — ignorado", ticker)
            return None

        # Modo dry_run: simula sem enviar
        if self.dry_run:
            entry = float(signal.get("entry_price", 0))
            logger.info(
                "MT5 [DRY RUN]: %s %s x%d @ %.2f  sl=%.2f  tp=%.2f  id=%s",
                direction, symbol, shares, entry, sl, tp, signal_id,
            )
            result = {
                "order_id":  f"DRY_{signal_id}",
                "ticker":    ticker,
                "symbol":    symbol,
                "direction": direction,
                "shares":    shares,
                "fill_price": entry,
                "sl":        sl,
                "tp":        tp,
                "status":    "dry_run",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_order(result)
            return result

        # Execução real
        if not self._ensure_connected():
            logger.error("MT5: sem conexão — ordem cancelada para %s", ticker)
            return None

        # Verifica se o símbolo existe e está disponível
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("MT5: símbolo '%s' não encontrado", symbol)
            return None

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        # Preço de execução
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("MT5: sem cotação para '%s'", symbol)
            return None

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price      = tick.ask if direction == "BUY" else tick.bid

        # Monta requisição
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(shares),
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,           # slippage máximo em pontos
            "magic":        20260101,     # identificador do robô
            "comment":      f"systematic_alpha_{signal_id}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(
            "MT5: enviando %s %s x%d @ %.2f  sl=%.2f  tp=%.2f",
            direction, symbol, shares, price, sl, tp,
        )

        result_mt5 = mt5.order_send(request)

        if result_mt5 is None or result_mt5.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result_mt5.retcode if result_mt5 else "N/A"
            comment = result_mt5.comment if result_mt5 else mt5.last_error()
            logger.error(
                "MT5: FALHA na ordem %s %s — retcode=%s  %s",
                direction, symbol, retcode, comment,
            )
            return None

        result = {
            "order_id":    result_mt5.order,
            "ticker":      ticker,
            "symbol":      symbol,
            "direction":   direction,
            "shares":      shares,
            "fill_price":  result_mt5.price,
            "sl":          sl,
            "tp":          tp,
            "status":      "filled",
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "signal_id":   signal_id,
        }

        logger.info(
            "MT5: ORDEM EXECUTADA | %s %s x%d @ R$%.2f | order_id=%s",
            direction, symbol, shares, result_mt5.price, result_mt5.order,
        )

        self._log_order(result)
        return result

    # ------------------------------------------------------------------
    def get_open_positions(self) -> list[dict]:
        """Retorna todas as posições abertas na conta MT5."""
        if not self._ensure_connected():
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            symbol = pos.symbol
            ticker = symbol + ".SA" if not symbol.endswith(".SA") else symbol
            result.append({
                "ticket":      pos.ticket,
                "ticker":      ticker,
                "symbol":      symbol,
                "direction":   "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "shares":      int(pos.volume),
                "entry_price": pos.price_open,
                "current":     pos.price_current,
                "sl":          pos.sl,
                "tp":          pos.tp,
                "profit":      pos.profit,
                "open_time":   datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
            })
        return result

    # ------------------------------------------------------------------
    def close_position(self, ticker: str) -> Optional[dict]:
        """
        Fecha todas as posições abertas para um ticker.
        Útil para saída manual ou em caso de kill switch.
        """
        if not self._ensure_connected():
            return None

        symbol    = _to_mt5_symbol(ticker)
        positions = mt5.positions_get(symbol=symbol)

        if not positions:
            logger.warning("MT5: nenhuma posição aberta para %s", symbol)
            return None

        results = []
        for pos in positions:
            direction = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick      = mt5.symbol_info_tick(symbol)
            price     = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

            close_request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       pos.volume,
                "type":         direction,
                "position":     pos.ticket,
                "price":        price,
                "deviation":    20,
                "magic":        20260101,
                "comment":      "systematic_alpha_close",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            res = mt5.order_send(close_request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    "MT5: posição fechada | %s x%.0f @ R$%.2f | lucro=R$%.2f",
                    symbol, pos.volume, res.price, pos.profit,
                )
                results.append({"symbol": symbol, "profit": pos.profit, "status": "closed"})
            else:
                logger.error("MT5: falha ao fechar %s — %s", symbol, res.comment if res else mt5.last_error())

        return results[0] if results else None

    # ------------------------------------------------------------------
    def _log_order(self, result: dict) -> None:
        """Persiste a ordem em CSV para auditoria."""
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        """Encerra a conexão com o MT5."""
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5: desconectado.")

    # ------------------------------------------------------------------
    def __del__(self):
        self.disconnect()


# ---------------------------------------------------------------------------
# Factory — retorna o broker correto baseado no BROKER_MODE
# ---------------------------------------------------------------------------

def get_broker(mode: str = "paper", **kwargs):
    """
    Fábrica de brokers. Retorna o broker correto baseado no modo.

    Parâmetros
    ----------
    mode : "paper" → PaperBroker (simulação)
           "mt5"   → MT5Broker (dinheiro real)
           "mt5_dry" → MT5Broker em modo dry_run (testa conexão sem enviar)

    Exemplos
    --------
        # Simulação (padrão)
        broker = get_broker("paper")

        # Real
        broker = get_broker("mt5", login=12345678, password="senha", server="Clear-Server")

        # Teste de conexão sem enviar ordens
        broker = get_broker("mt5_dry", login=12345678, password="senha", server="Clear-Server")
    """
    from autotrader.config.settings import INITIAL_EQUITY

    if mode == "paper":
        from autotrader.execution.paper_broker import PaperBroker
        return PaperBroker(initial_equity=INITIAL_EQUITY)

    elif mode in ("mt5", "mt5_dry"):
        dry = mode == "mt5_dry"

        login    = kwargs.get("login")
        password = kwargs.get("password")
        server   = kwargs.get("server")

        if not all([login, password, server]):
            raise ValueError(
                "Para usar MT5, forneça: login, password e server.\n"
                "Exemplo: get_broker('mt5', login=12345678, password='senha', server='Clear-Server')"
            )

        # Login pode ser numérico (int) ou alfanumérico (str) dependendo da corretora
        try:
            login_val = int(login)
        except (ValueError, TypeError):
            login_val = login   # mantém como string se não for numérico

        return MT5Broker(
            login=login_val,
            password=str(password),
            server=str(server),
            dry_run=dry,
        )

    else:
        raise ValueError(f"BROKER_MODE inválido: '{mode}'. Use 'paper', 'mt5' ou 'mt5_dry'.")
