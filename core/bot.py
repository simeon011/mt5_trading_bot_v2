"""
core/bot.py — Главен оркестратор v3
Оправени: дублирани сделки, score праг, cooldown по символ
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Set

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from strategies.signal_engine import SignalEngine
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
        from typing import List as _L
        self._hourly_trades: list = []

        if not self.mt5.connect():
            raise RuntimeError("Не може да се свърже с MT5!")
        logger.info("Всички модули инициализирани.")

    def run_cycle(self):
        self._cycle_count += 1
        logger.info(f"Цикъл #{self._cycle_count} [{datetime.now().strftime('%H:%M:%S')}]")

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

            # Cooldown (0 за scalper = без чакане)
            last = self._last_signal_cycle.get(symbol, 0)
            if self._cycle_count - last < s.COOLDOWN_CYCLES:
                continue

            try:
                df_h1 = self.mt5.get_candles(symbol, s.PRIMARY_TF, s.CANDLE_HISTORY)
                df_h4 = self.mt5.get_candles(symbol, s.HIGHER_TF, 200)
                df_m15 = self.mt5.get_candles(symbol, s.ENTRY_TF, 100)

                if df_h1 is None or df_h4 is None or df_m15 is None:
                    continue

                ml_features = self._extract_ml_features(symbol, df_h1)
                ml_pred = self.ml.predict(ml_features) if s.ML_ENABLED else None

                signal = self.signals.analyze(symbol, df_h1, df_h4, df_m15, ml_pred)

                if signal.direction == "NEUTRAL":
                    continue

                # Score под прага — пропускаме
                if signal.score < s.MIN_SIGNAL_SCORE:
                    logger.info(f"SKIP {symbol}: Score {signal.score:.1f} < {s.MIN_SIGNAL_SCORE} (праг)")
                    continue

                allowed, reason = self.risk.can_trade(symbol, account)
                if not allowed:
                    logger.info(f"SKIP {symbol}: {reason}")
                    continue

                if signal.risk_reward < 1.5:
                    logger.info(f"SKIP {symbol}: R:R={signal.risk_reward:.1f} < 1.5")
                    continue

                sl_distance = abs(signal.entry_price - signal.stop_loss)
                volume = self.mt5.calculate_volume(symbol, sl_distance, s.RISK_PERCENT)

                logger.info(
                    f"SIGNAL {symbol} | {signal.direction} | "
                    f"Score:{signal.score:.1f} | R:R:{signal.risk_reward:.1f} | "
                    f"{signal.reasoning}"
                )

                if self.mode == "live":
                    order = self.mt5.place_order(
                        symbol=symbol,
                        order_type=signal.direction,
                        volume=volume,
                        sl=signal.stop_loss,
                        tp=signal.take_profit,
                        comment=f"AI_{signal.score:.0f}"
                    )
                else:
                    order = {
                        "ticket": abs(hash(f"{symbol}{datetime.now().isoformat()}")) % 999999,
                        "symbol": symbol,
                        "type": signal.direction,
                        "volume": volume,
                        "price": signal.entry_price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit
                    }
                    logger.info(
                        f"[PAPER] {signal.direction} {volume} {symbol} "
                        f"@ {signal.entry_price:.5f} | "
                        f"SL:{signal.stop_loss:.5f} | TP:{signal.take_profit:.5f}"
                    )

                if order:
                    self.risk.register_open(order["ticket"], {
                        "symbol": symbol,
                        "type": signal.direction,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "entry": signal.entry_price
                    })
                    self._pending_trades[order["ticket"]] = {
                        "signal": signal,
                        "ml_features": ml_features,
                        "entry_price": signal.entry_price
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
                # --- ТУК СЛАГАШ НОВИЯ КОД ---
                # 1. Изчисляваме разстоянието до целта
                target_pips = abs(pos["tp"] - pos["price_open"])
                current_pips = abs(current_price - pos["price_open"])
                # 2. Проверяваме дали сме изминали поне 50% от пътя до TP
                # Ако не сме - прескачаме местенето на стопа (continue)
                if current_pips < (target_pips * 0.4):
                    continue
                    # ---------------------------
                # Едва ако сме минали 50%, изпълняваме долния код:
                new_sl = self.risk.calculate_trailing_stop(pos, current_price, atr)
                if new_sl and self.mode == "live":
                    if self.mt5.modify_sl(pos["ticket"], new_sl):
                        logger.info(f"✅ TrailingStop {symbol}: {pos['sl']:.5f} -> {new_sl:.5f}")

            except Exception as e:
                logger.debug(f"Грешка при управление на позиция {pos.get('symbol')}: {e}")

    def _check_closed_trades(self, current_positions):
        current_tickets = {p["ticket"] for p in current_positions}
        closed_tickets = set(self._pending_trades.keys()) - current_tickets

        for ticket in closed_tickets:
            pending = self._pending_trades.pop(ticket)
            signal = pending["signal"]
            ml_features = pending["ml_features"]
            entry = pending["entry_price"]

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
            logger.info(f"Затворена [{outcome}]: {signal.symbol} {profit_pct:+.2f}%")

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
        logger.info(f"  PnL:      {summary['daily_pnl']:+.2f}")
        logger.info(f"  Сделки:   {summary['daily_trades']}")
        if ml_stats:
            logger.info(f"  WinRate:  {ml_stats.get('win_rate', 0):.1%} ({ml_stats.get('total_trades', 0)} сделки)")
        logger.info("=" * 50)

    def run_backtest(self, date_from: str, date_to: str):
        logger.info(f"Backtest {date_from} -> {date_to}")

    def shutdown(self):
        self.mt5.disconnect()
        logger.info(f"Финален отчет: {self.risk.get_daily_summary()}")
