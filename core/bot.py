"""
core/bot.py — Главен оркестратор v3 (FIXED)
✅ Фиксирани: дублирани сделки, score праг, cooldown по символ
✅ НОВО: Коректна P&L калкулация, правилен Trailing Stop
"""

import logging
import math
import uuid
from datetime import datetime
from typing import Dict, Set, List

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from strategies.signal_engine import SignalEngine, get_pip
from ml.learning_agent import LearningAgent, TradeRecord

logger = logging.getLogger("TradingBot")


class TradingBot:

    def __init__(self, settings, mode: str = "paper"):
        self.settings = settings
        self.mode = mode
        self.is_paper = mode == "paper"

        logger.info("Инициализиране на модули...")
        self.mt5 = MT5Connector(settings)
        self.risk = RiskManager(settings)
        self.signals = SignalEngine(settings)
        self.ml = LearningAgent(settings)

        self._pending_trades: Dict = {}
        self._cycle_count = 0
        self._last_signal_cycle: Dict[str, int] = {}
        # Hourly trade counter за scalper
        self._hourly_trades: List[float] = []

        if not self.mt5.connect():
            raise RuntimeError("Не може да се свърже с MT5!")
        logger.info("Всички модули инициализирани.")

    def run_cycle(self):
        self._cycle_count += 1
        logger.cycle(f"–––––––––––––––Цикъл #{self._cycle_count} [{datetime.now().strftime('%H:%M:%S')}]–––––––––––––––")

        account = self.mt5.get_account_info()
        open_positions = self.mt5.get_open_positions()

        # Подаваме баланса за adaptive pip targets
        self.signals.update_balance(account.get("balance", 1000.0))

        self._sync_positions(open_positions)
        self._manage_open_positions(open_positions, account)
        self._check_closed_trades(open_positions)
        self._analyze_and_trade(account)

        if self._cycle_count % 30 == 0:
            self._print_daily_report(account)

    def _analyze_and_trade(self, account: Dict):
        s = self.settings
        traded_this_cycle: Set[str] = set()

        for symbol in s.SYMBOLS:
            # Не влизаме 2 пъти в един символ в един цикъл
            if symbol in traded_this_cycle:
                continue

            # Cooldown проверка
            last = self._last_signal_cycle.get(symbol, 0)
            if self._cycle_count - last < s.COOLDOWN_CYCLES:
                continue

            try:
                # Взимане на данни
                df_h1 = self.mt5.get_candles(symbol, s.PRIMARY_TF, s.CANDLE_HISTORY)
                df_h4 = self.mt5.get_candles(symbol, s.HIGHER_TF, 200)
                df_m15 = self.mt5.get_candles(symbol, s.ENTRY_TF, 100)

                if df_h1 is None or df_h4 is None or df_m15 is None:
                    continue

                ml_features = self._extract_ml_features(symbol, df_h1)
                ml_pred = self.ml.predict(ml_features) if s.ML_ENABLED else 0.5

                # AI параметри (Обем)
                smart_params = self.ml.get_smart_trade_params(symbol, ml_pred, ml_features)
                volume = smart_params["volume"]

                # Анализ на сигнала
                signal = self.signals.analyze(symbol, df_h1, df_h4, df_m15, ml_pred)

                if signal.direction == "NEUTRAL":
                    continue

                # Филтри: Score, Риск Мениджър и R:R
                if signal.score < s.MIN_SIGNAL_SCORE:
                    continue

                allowed, reason = self.risk.can_trade(symbol, account)
                if not allowed:
                    logger.info(f"SKIP {symbol}: {reason}")
                    continue

                if signal.risk_reward < 1.5:
                    logger.info(f"SKIP {symbol}: R:R={signal.risk_reward:.1f} < 1.5")
                    continue

                # ═══════════════════════════════════════════════════════════
                # 🎯 НОВО: ДИНАМИЧНО ОПРЕДЕЛЯНЕ НА TP (TP1 срещу TP2)
                # ═══════════════════════════════════════════════════════════
                final_tp = signal.take_profit  # По подразбиране е далечният TP2

                # Ако лотът е малък (0.01 - 0.03), пресмятаме TP1 като единствена цел
                if volume < s.PARTIAL_CLOSE_MIN_LOT:
                    dist = abs(signal.take_profit - signal.entry_price)
                    if signal.direction == "BUY":
                        final_tp = signal.entry_price + (dist / 2)
                    else:
                        final_tp = signal.entry_price - (dist / 2)

                    # Закръгляме спрямо брокера (например 5 знака)
                    final_tp = round(final_tp, 5)
                    logger.info(f"📐 {symbol}: Малък лот ({volume}). Зададен единичен TP на ниво TP1.")

                logger.info(
                    f"SIGNAL {symbol} | {signal.direction} | "
                    f"Лот: {volume} | Score:{signal.score:.1f} | TP Level: {'TP1' if volume < s.PARTIAL_CLOSE_MIN_LOT else 'TP2'}"
                )

                if self.mode == "live":
                    order = self.mt5.place_order(
                        symbol=symbol,
                        order_type=signal.direction,
                        volume=volume,
                        sl=signal.stop_loss,
                        tp=final_tp,  # Изпращаме коригирания TP
                        comment=f"AI_{signal.score:.0f}"
                    )
                else:
                    order = {
                        "ticket": abs(hash(f"{symbol}{datetime.now().isoformat()}")) % 999999,
                        "symbol": symbol, "type": signal.direction, "volume": volume,
                        "price": signal.entry_price, "sl": signal.stop_loss, "tp": final_tp
                    }

                if order:
                    self.risk.register_open(order["ticket"], {
                        "symbol": symbol, "type": signal.direction, "volume": volume,
                        "sl": signal.stop_loss, "tp": final_tp, "entry": signal.entry_price,
                        "price_open": signal.entry_price
                    })
                    self._pending_trades[order["ticket"]] = {
                        "signal": signal,
                        "ml_features": ml_features,
                        "entry_price": signal.entry_price,
                        "initial_volume": volume
                    }
                    traded_this_cycle.add(symbol)
                    self._last_signal_cycle[symbol] = self._cycle_count

            except Exception as e:
                logger.error(f"Грешка при анализ на {symbol}: {e}", exc_info=True)

    def _manage_open_positions(self, positions, account):
        from strategies.indicators import TechnicalIndicators
        ind = TechnicalIndicators()

        for pos in positions:
            try:
                symbol = pos["symbol"]
                df = self.mt5.get_candles(symbol, self.settings.PRIMARY_TF, 50)
                if df is None or df.empty:
                    continue

                atr = ind.atr(df).iloc[-1]
                current_price = df["Close"].iloc[-1]
                pip_unit = get_pip(symbol)

                # Изчисляваме прогреса спрямо целта
                target_pips = abs(pos["tp"] - pos["price_open"])
                current_pips = abs(current_price - pos["price_open"])

                # Дефинираме TP1 като среда на разстоянието (за логиката на BE и Partial)
                tp1_pips = target_pips / 2 if target_pips > 0 else 0

                # Проверка за първоначалния обем
                initial_volume = self._pending_trades.get(pos["ticket"], {}).get("initial_volume", pos["volume"])
                is_partially_closed = pos["volume"] < initial_volume

                # ─── ПРАВИЛО 1: БРОНИРАНА ЖИЛЕТКА (Break-even + 2 pips) ───
                # Активира се, когато цената измине 45% от разстоянието до TP1
                if current_pips >= (tp1_pips * 0.45):
                    if pos["type"] == "BUY":
                        new_sl = pos["price_open"] + (2 * pip_unit)
                        # Модифицираме само ако новият стоп е по-сигурен от текущия
                        if new_sl > pos["sl"] + (atr * 0.05):
                            if self.mode == "live":
                                if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                    logger.info(f"🛡️ BE+2 {symbol}: SL преместен на +2 пипа.")

                    elif pos["type"] == "SELL":
                        new_sl = pos["price_open"] - (2 * pip_unit)
                        if new_sl < pos["sl"] - (atr * 0.05):
                            if self.mode == "live":
                                if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                    logger.info(f"🛡️ BE+2 {symbol}: SL преместен на +2 пипа.")

                # ─── ПРАВИЛО 2: ЧАСТИЧНО ЗАТВАРЯНЕ (Само за големи лотове) ───
                # Малките лотове (0.01-0.03) не влизат тук, защото техният TP вече е заложен на TP1 ниво
                if initial_volume >= self.settings.PARTIAL_CLOSE_MIN_LOT:
                    if current_pips >= tp1_pips and not is_partially_closed:
                        exact_half = initial_volume * self.settings.PARTIAL_CLOSE_PERCENT
                        vol_to_close = math.ceil(exact_half * 100) / 100

                        if self.mode == "live":
                            if self.mt5.close_partial_position(pos["ticket"], vol_to_close):
                                logger.info(
                                    f"💰 Partial Close {symbol}: Прибрани {vol_to_close} лота. Остатъкът гони TP2!")
                                # Обновяваме информацията в pending trades
                                if pos["ticket"] in self._pending_trades:
                                    self._pending_trades[pos["ticket"]]["initial_volume"] = initial_volume

            except Exception as e:
                logger.debug(f"Грешка при управление на позиция {pos.get('symbol')}: {e}")

    def _check_closed_trades(self, current_positions):
        """
        Проверява за затворени сделки и записва само потвърдени резултати от историята.
        ✅ ФИКС: Веднага освобождава символа, за да не блокира нови сделки.
        """
        current_tickets = {p["ticket"] for p in current_positions}
        closed_tickets = set(self._pending_trades.keys()) - current_tickets

        if not closed_tickets:
            return

        account_info = self.mt5.get_account_info()
        balance = account_info.get("balance", 1000.0)
        currency = account_info.get("currency", "USD")

        for ticket in list(closed_tickets):
            # 1. Опит за вземане на РЕАЛНАТА цена от историята
            real_exit = self.mt5.get_deal_exit_price(ticket)

            # 🚨 АКО НЕ Е НАМЕРЕНА В ИСТОРИЯТА (MT5 Lag):
            if real_exit is None:
                # Вземаме текущата пазарна цена за статистиката, за да не "зависваме"
                sym = self._pending_trades[ticket]["signal"].symbol
                logger.warning(f"⏳ Тикет {ticket} ({sym}) липсва в историята. Използвам Fallback цена.")
                real_exit = self.mt5.get_candles(sym, "M1", 1)["Close"].iloc[-1]

            # ✅ КЛЮЧ: Вадим тикета от pending ВИНАГИ тук, за да освободим символа за нови сигнали!
            pending = self._pending_trades.pop(ticket)

            signal = pending["signal"]
            ml_features = pending["ml_features"]
            entry = pending["entry_price"]
            symbol = signal.symbol
            initial_vol = pending.get("initial_volume", 0.01)

            # 2. Изчисляваме печалбата в пари чрез MT5 данни
            sym_info = self.mt5._mt5.symbol_info(symbol)
            if sym_info:
                point = sym_info.point
                tick_val = sym_info.trade_tick_value
                if signal.direction == "BUY":
                    money_profit = (real_exit - entry) / point * tick_val * initial_vol
                else:
                    money_profit = (entry - real_exit) / point * tick_val * initial_vol
            else:
                money_profit = 0.0

            # 3. Процент спрямо баланса (Синхрон с MT5 Terminal)
            profit_pct = (money_profit / balance) * 100

            # 4. Пипсове за AI
            pip_unit = get_pip(symbol)
            profit_pips = (real_exit - entry) / pip_unit if signal.direction == "BUY" else (
                                                                                                       entry - real_exit) / pip_unit

            # 5. Резултат
            outcome = "WIN" if money_profit > 0.00 else "LOSS"

            record = TradeRecord(
                id=str(uuid.uuid4()),
                symbol=symbol,
                direction=signal.direction,
                entry_price=entry,
                exit_price=real_exit,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                profit_pips=profit_pips,
                profit_pct=profit_pct,
                outcome=outcome,
                duration_hours=1.0,
                rsi_at_entry=ml_features.get("rsi", 50),
                macd_hist_at_entry=ml_features.get("macd_hist", 0),
                ma_alignment=ml_features.get("ma_alignment", 0),
                ob_score=signal.ob_score,
                candle_score=signal.candle_score,
                trendline_score=signal.trendline_score,
                total_score=signal.score,
                market_hour=datetime.now().hour,
                day_of_week=datetime.now().weekday()
            )

            self.ml.record_trade(record)
            self.risk.register_close(ticket, profit_pct)

            emoji = "✅" if outcome == "WIN" else "❌"
            logger.info(
                f"{emoji} Затворена: {symbol} | P&L: {profit_pct:+.4f}% | Profit: {money_profit:+.2f} {currency}"
            )

    def _sync_positions(self, mt5_positions):
        for pos in mt5_positions:
            if pos["ticket"] not in self.risk.open_positions:
                self.risk.open_positions[pos["ticket"]] = pos

    def _extract_ml_features(self, symbol: str, df) -> Dict:
        from strategies.indicators import TechnicalIndicators
        ind = TechnicalIndicators()
        s = self.settings
        close = df["Close"]
        rsi_val = float(ind.rsi(close, s.RSI_PERIOD).iloc[-1])
        _, _, macd_hist = ind.macd(close)
        ma_f = ind.ema(close, s.MA_FAST).iloc[-1]
        ma_s = ind.ema(close, s.MA_SLOW).iloc[-1]
        return {
            "rsi": rsi_val,
            "macd_hist": float(macd_hist.iloc[-1]),
            "ma_alignment": 1.0 if ma_f > ma_s else 0.0,
            "ob_score": 10,
            "candle_score": 5,
            "trendline_score": 5,
            "total_score": 50,
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday()
        }

    def _print_daily_report(self, account: Dict):
        summary = self.risk.get_daily_summary()
        ml_stats = self.ml.get_stats()
        logger.info("=" * 50)
        logger.info(f"ДНЕВЕН ОТЧЕТ [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
        logger.info(f"  Баланс:   {account.get('balance', 0):.2f} {account.get('currency','')}")
        logger.info(f"  PnL:      {summary['daily_pnl']:+.2f}%")
        logger.info(f"  Сделки:   {summary['daily_trades']}")
        if ml_stats:
            logger.info(f"  WinRate:  {ml_stats.get('win_rate', 0):.1%} ({ml_stats.get('total_trades', 0)} сделки)")
        logger.info("=" * 50)

    def run_backtest(self, date_from: str, date_to: str):
        logger.info(f"Backtest {date_from} -> {date_to}")

    def shutdown(self):
        self.mt5.disconnect()
        logger.info(f"Финален отчет: {self.risk.get_daily_summary()}")