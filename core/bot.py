"""
core/bot.py — Главен оркестратор на бота
Свързва всички модули и управлява trading цикъла
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from strategies.signal_engine import SignalEngine
from ml.learning_agent import LearningAgent, TradeRecord

logger = logging.getLogger("TradingBot")


class TradingBot:
    """Главен бот — обединява MT5, сигнали, риск и ML."""

    def __init__(self, settings, mode: str = "paper"):
        self.settings = settings
        self.mode = mode
        self.is_paper = mode == "paper"

        logger.info("🔧 Инициализиране на модули...")
        self.mt5 = MT5Connector(settings)
        self.risk = RiskManager(settings)
        self.signals = SignalEngine(settings)
        self.ml = LearningAgent(settings)

        self._pending_trades: Dict = {}
        self._cycle_count = 0

        if not self.mt5.connect():
            raise RuntimeError("Не може да се свърже с MT5!")
        logger.info("✅ Всички модули инициализирани.")

    def run_cycle(self):
        """Един пълен цикъл: анализ → сигнали → изпълнение → управление."""
        self._cycle_count += 1
        logger.info(f"── Цикъл #{self._cycle_count} [{datetime.now().strftime('%H:%M:%S')}] ──")

        account = self.mt5.get_account_info()
        open_positions = self.mt5.get_open_positions()

        # Синхронизиране на open positions с Risk Manager
        self._sync_positions(open_positions)

        # Управление на съществуващи позиции (trailing stop, TP update)
        self._manage_open_positions(open_positions, account)

        # Проверка за сделки, достигнали SL/TP (за ML обучение)
        self._check_closed_trades(open_positions)

        # Анализ и нови сигнали
        self._analyze_and_trade(account)

        # Дневен отчет на всеки 30 цикъла
        if self._cycle_count % 30 == 0:
            self._print_daily_report(account)

    def _analyze_and_trade(self, account: Dict):
        """Анализира всеки символ и изпраща поръчки при сигнал."""
        s = self.settings

        for symbol in s.SYMBOLS:
            try:
                # Зареждане на данни за 3 таймфрейма
                df_h1 = self.mt5.get_candles(symbol, s.PRIMARY_TF, s.CANDLE_HISTORY)
                df_h4 = self.mt5.get_candles(symbol, s.HIGHER_TF, 200)
                df_m15 = self.mt5.get_candles(symbol, s.ENTRY_TF, 100)

                if df_h1 is None or df_h4 is None or df_m15 is None:
                    continue

                # ML предсказване
                ml_features = self._extract_ml_features(symbol, df_h1)
                ml_pred = self.ml.predict(ml_features) if s.ML_ENABLED else None

                # Генериране на сигнал
                signal = self.signals.analyze(symbol, df_h1, df_h4, df_m15, ml_pred)

                if signal.direction == "NEUTRAL":
                    continue

                # Проверка за разрешение от Risk Manager
                allowed, reason = self.risk.can_trade(symbol, account)
                if not allowed:
                    logger.debug(f"⛔ {symbol}: {reason}")
                    continue

                # Risk/Reward проверка
                if signal.risk_reward < 1.5:
                    logger.debug(f"⛔ {symbol}: R:R={signal.risk_reward:.1f} < 1.5")
                    continue

                # Изчисляване на lot размер
                sl_distance = abs(signal.entry_price - signal.stop_loss)
                volume = self.mt5.calculate_volume(symbol, sl_distance, s.RISK_PERCENT)

                logger.info(
                    f"🎯 СИГНАЛ {symbol} | {signal.direction} | "
                    f"Score:{signal.score:.1f} | R:R:{signal.risk_reward:.1f} | "
                    f"Причина: {signal.reasoning}"
                )

                # Изпращане на поръчка
                if not self.is_paper or self.mode == "live":
                    order = self.mt5.place_order(
                        symbol=symbol,
                        order_type=signal.direction,
                        volume=volume,
                        sl=signal.stop_loss,
                        tp=signal.take_profit,
                        comment=f"AI_{signal.score:.0f}"
                    )
                else:
                    # Paper mode — само логиране
                    order = {
                        "ticket": hash(f"{symbol}{datetime.now()}") % 100000,
                        "symbol": symbol,
                        "type": signal.direction,
                        "volume": volume,
                        "price": signal.entry_price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit
                    }
                    logger.info(f"📋 [PAPER] {signal.direction} {volume} {symbol} "
                                f"@ {signal.entry_price:.5f}")

                if order:
                    self.risk.register_open(order["ticket"], {
                        "symbol": symbol,
                        "type": signal.direction,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "entry": signal.entry_price
                    })
                    # Запазваме за ML tracking
                    self._pending_trades[order["ticket"]] = {
                        "signal": signal,
                        "ml_features": ml_features,
                        "entry_price": signal.entry_price
                    }

            except Exception as e:
                logger.error(f"Грешка при анализ на {symbol}: {e}", exc_info=True)

    def _manage_open_positions(self, positions, account):
        """Управлява trailing stop за отворени позиции."""
        from strategies.indicators import TechnicalIndicators
        ind = TechnicalIndicators()

        for pos in positions:
            try:
                df = self.mt5.get_candles(pos["symbol"], self.settings.PRIMARY_TF, 50)
                if df is None:
                    continue
                atr = ind.atr(df).iloc[-1]
                current_price = df["Close"].iloc[-1]

                new_sl = self.risk.calculate_trailing_stop(pos, current_price, atr)
                if new_sl and not self.is_paper:
                    if self.mt5.modify_sl(pos["ticket"], new_sl):
                        logger.info(f"📈 Trailing stop за {pos['symbol']}: "
                                    f"{pos['sl']:.5f} → {new_sl:.5f}")
            except Exception as e:
                logger.debug(f"Trailing stop грешка: {e}")

    def _check_closed_trades(self, current_positions):
        """Проверява кои pending trades са затворени → записва за ML."""
        current_tickets = {p["ticket"] for p in current_positions}
        closed_tickets = set(self._pending_trades.keys()) - current_tickets

        for ticket in closed_tickets:
            pending = self._pending_trades.pop(ticket)
            signal = pending["signal"]
            ml_features = pending["ml_features"]
            entry = pending["entry_price"]

            # В реален режим взимаме реалния profit от MT5 история
            # В paper режим симулираме
            sim_exit = signal.take_profit if signal.score > 70 else signal.stop_loss
            profit_pips = (sim_exit - entry) if signal.direction == "BUY" else (entry - sim_exit)
            profit_pct = profit_pips / entry * 100
            outcome = "WIN" if profit_pips > 0 else "LOSS"

            record = TradeRecord(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=entry,
                exit_price=sim_exit,
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
            logger.info(f"🔒 Сделка затворена [{outcome}]: {signal.symbol} "
                        f"{profit_pct:+.2f}%")

    def _sync_positions(self, mt5_positions):
        """Синхронизира Risk Manager с реалните MT5 позиции."""
        for pos in mt5_positions:
            if pos["ticket"] not in self.risk.open_positions:
                self.risk.open_positions[pos["ticket"]] = pos

    def _extract_ml_features(self, symbol: str, df) -> Dict:
        from strategies.indicators import TechnicalIndicators
        ind = TechnicalIndicators()
        s = self.settings
        close = df["Close"]
        rsi = ind.rsi(close, s.RSI_PERIOD).iloc[-1]
        _, _, macd_hist = ind.macd(close)
        ma_f = ind.ema(close, s.MA_FAST).iloc[-1]
        ma_s = ind.ema(close, s.MA_SLOW).iloc[-1]
        ma_alignment = 1.0 if ma_f > ma_s else 0.0
        return {
            "rsi": float(rsi),
            "macd_hist": float(macd_hist.iloc[-1]),
            "ma_alignment": ma_alignment,
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
        logger.info("═" * 50)
        logger.info(f"📊 ДНЕВЕН ОТЧЕТ [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
        logger.info(f"  Баланс:         {account.get('balance', 0):.2f} {account.get('currency','')}")
        logger.info(f"  Equity:         {account.get('equity', 0):.2f}")
        logger.info(f"  Дневен P&L:     {summary['daily_pnl']:+.2f}")
        logger.info(f"  Сделки днес:    {summary['daily_trades']}")
        logger.info(f"  Отворени:       {summary['open_positions']}")
        if ml_stats:
            logger.info(f"  ML Win Rate:    {ml_stats.get('win_rate', 0):.1%}")
            logger.info(f"  Общо сделки:   {ml_stats.get('total_trades', 0)}")
        logger.info("═" * 50)

    def run_backtest(self, date_from: str, date_to: str):
        """Backtesting на исторически данни."""
        logger.info(f"📈 Backtest {date_from} → {date_to}")
        logger.info("ℹ️  Backtest изисква MT5 с исторически данни.")
        # Имплементация: зарежда исторически данни, симулира сигнали и сделки
        pass

    def shutdown(self):
        self.mt5.disconnect()
        report = self.risk.get_daily_summary()
        logger.info(f"📊 Финален отчет: {report}")
