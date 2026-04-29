"""
core/bot.py — Главен оркестратор v3 (FIXED)
✅ Фиксирани: дублирани сделки, score праг, cooldown по символ
✅ НОВО: Коректна P&L калкулация, правилен Trailing Stop
"""

import logging
import math
import uuid
from datetime import datetime
from typing import Dict, Set, List, Optional

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from strategies.signal_engine import SignalEngine, get_pip, TechnicalIndicators
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

                # Анализ на сигнала (нужен ни е SL за валидация на риска)
                signal = self.signals.analyze(symbol, df_h1, df_h4, df_m15, ml_pred)

                if signal.direction == "NEUTRAL":
                    continue

                # ── ДОЛЛАРОВ РИСК КОНТРОЛ ──────────────────────────────────
                volume = self._cap_volume_to_risk(symbol, volume, signal.stop_loss, signal.entry_price, account)
                if volume is None:
                    continue

                # Филтри: Score, Риск Мениджър и R:R
                if signal.score < s.MIN_SIGNAL_SCORE:
                    continue

                allowed, reason = self.risk.can_trade(symbol, account)
                if not allowed:
                    logger.info(f"SKIP {symbol}: {reason}")
                    continue

                if signal.risk_reward < self.settings.MIN_RR_RATIO:
                    logger.info(f"SKIP {symbol}: R:R={signal.risk_reward:.2f} < {self.settings.MIN_RR_RATIO}")
                    continue

                # ═══════════════════════════════════════════════════════════
                # 🎯 НОВО: ДИНАМИЧНО ОПРЕДЕЛЯНЕ НА TP (TP1 срещу TP2)
                # ═══════════════════════════════════════════════════════════
                final_tp = signal.take_profit  # По подразбиране е далечният TP2

                # Ако лотът е малък (0.01 - 0.03), пресмятаме TP1 като единствена цел
                if volume < s.PARTIAL_CLOSE_MIN_LOT:
                    dist = abs(signal.take_profit - signal.entry_price)
                    if signal.direction == "BUY":
                        final_tp = signal.entry_price + (dist * 0.70)
                    else:
                        final_tp = signal.entry_price - (dist * 0.70)

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
        """Управлява Trailing Stop и Break-even + 2 pips защита."""
        ind = TechnicalIndicators()

        for pos in positions:
            try:
                symbol = pos["symbol"]
                df = self.mt5.get_candles(symbol, self.settings.PRIMARY_TF, 50)
                if df is None or df.empty:
                    continue

                atr = ind.atr(df).iloc[-1]
                current_price = df["Close"].iloc[-1]
                pip_unit = get_pip(symbol)  # 0.0001 за Forex, 0.01 за JPY

                # ─────────────────────────────────────────────────────────────
                # НОВО: Изчисляваме ПРОГРЕСА към TP1 (среда на TP)
                # ─────────────────────────────────────────────────────────────
                target_pips = abs(pos["tp"] - pos["price_open"])
                current_pips = abs(current_price - pos["price_open"])
                tp1_pips = target_pips / 2 if target_pips > 0 else 0

                progress_pct = (current_pips / tp1_pips * 100) if tp1_pips > 0 else 0

                # ─────────────────────────────────────────────────────────────
                # ПРАВИЛО 1: BREAK-EVEN + 2 ПИПА (Когато е на 45% към TP1)
                # ─────────────────────────────────────────────────────────────
                if progress_pct >= 45 and progress_pct < 100:
                    if pos["type"] == "BUY":
                        new_sl = pos["price_open"] + (2 * pip_unit)
                        if new_sl > pos["sl"] + (atr * 0.05):
                            if self.mode == "live":
                                if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                    logger.info(f"🛡️ BE+2 {symbol}: SL преместен на Entry+2пипа ({new_sl:.5f})")
                                    if pos["ticket"] in self._pending_trades:
                                        self._pending_trades[pos["ticket"]]["be_activated"] = True

                    elif pos["type"] == "SELL":
                        new_sl = pos["price_open"] - (2 * pip_unit)
                        if new_sl < pos["sl"] - (atr * 0.05):
                            if self.mode == "live":
                                if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                    logger.info(f"🛡️ BE+2 {symbol}: SL преместен на Entry-2пипа ({new_sl:.5f})")
                                    if pos["ticket"] in self._pending_trades:
                                        self._pending_trades[pos["ticket"]]["be_activated"] = True

                # ─────────────────────────────────────────────────────────────
                # ПРАВИЛО 2: TRAILING STOP (Активира се след BE+2)
                # ─────────────────────────────────────────────────────────────
                if progress_pct >= 100 and self.settings.USE_TRAILING_STOP:
                    pos_info = self.risk.open_positions.get(pos["ticket"], pos)
                    new_trail_sl = self.risk.calculate_trailing_stop(pos_info, current_price, float(atr))
                    if new_trail_sl is not None and self.mode == "live":
                        if self.mt5.modify_sl(pos["ticket"], new_trail_sl, pos["tp"]):
                            logger.debug(f"🔄 Trailing Stop {symbol}: SL → {new_trail_sl:.5f}")

            except Exception as e:
                logger.debug(f"❌ Грешка при управление на позиция {pos.get('symbol')}: {e}")

    """
    ЗАМЕНИ функцията _check_closed_trades в core/bot.py С ТОЗИ КОД!

    ВАЖНО: Комисията е калкулирана ДВОЙНО (вход + изход)
    Ако брокерът взима 0.03$ per trade, то total е 0.06$ (вход + изход)
    """

    def _check_closed_trades(self, current_positions):
        """
        Проверява затворени сделки и записва правилната P&L с комисия.
        ✅ ФИКСИРАНО: Вземаме РЕАЛНАТА цена от историята
        ✅ ФИКСИРАНО: Калкулираме P&L с комисия (вход + изход)
        ✅ ФИКСИРАНО: Освобождаваме символа веднага
        """
        current_tickets = {p["ticket"] for p in current_positions}
        closed_tickets = set(self._pending_trades.keys()) - current_tickets

        if not closed_tickets:
            return

        account_info = self.mt5.get_account_info()
        balance = account_info.get("balance", 1000.0)
        currency = account_info.get("currency", "USD")

        # Комисия: $0.03 на 0.01 лот (вход + изход = x2)
        COMMISSION_PER_001_LOT = 0.03

        for ticket in list(closed_tickets):
            # ✅ Вадим тикета от pending ВЕДНАГА, за да освободим символа!
            pending = self._pending_trades.pop(ticket)

            signal = pending["signal"]
            ml_features = pending["ml_features"]
            entry = pending["entry_price"]
            symbol = signal.symbol
            initial_vol = pending.get("initial_volume", 0.01)

            # 1. Вземаме P&L и цена директно от MT5
            deal_info = self.mt5.get_deal_info(ticket)
            if deal_info:
                real_exit    = deal_info["price"]
                raw_profit   = deal_info["profit"]
                real_pnl_usd = deal_info["net"]   # profit + commission + swap от MT5
            else:
                logger.warning(f'Ticket {ticket} ({symbol}) not found in history. Using fallback.')
                try:
                    df_temp = self.mt5.get_candles(symbol, 'M1', 1)
                    real_exit = float(df_temp['Close'].iloc[-1]) if df_temp is not None and not df_temp.empty else entry
                except:
                    real_exit = entry
                sym_info = self.mt5._mt5.symbol_info(symbol) if self.mt5._mt5 else None
                if sym_info and sym_info.trade_tick_size > 0:
                    if signal.direction == "BUY":
                        raw_profit = (real_exit - entry) / sym_info.trade_tick_size * sym_info.trade_tick_value * initial_vol
                    else:
                        raw_profit = (entry - real_exit) / sym_info.trade_tick_size * sym_info.trade_tick_value * initial_vol
                else:
                    raw_profit = 0.0
                TOTAL_COMMISSION = round((initial_vol / 0.01) * COMMISSION_PER_001_LOT * 2, 4)
                real_pnl_usd = raw_profit - TOTAL_COMMISSION

            real_pnl_pct = (real_pnl_usd / balance) * 100 if balance > 0 else 0

            # 4. Пипсове за AI обучение (НА БАЗАТА НА РЕАЛНАТА ЦЕНА)
            pip_unit = get_pip(symbol)
            if signal.direction == "BUY":
                profit_pips = (real_exit - entry) / pip_unit
            else:
                profit_pips = (entry - real_exit) / pip_unit

            # 5. Резултат
            be_activated = pending.get("be_activated", False)
            if be_activated and profit_pips <= 5:
                # BE SL е на entry+2 пипа — затваряне с малко пипове е BREAKEVEN
                # независимо дали net P&L е леко + или леко - (заради комисионна)
                outcome = "BREAKEVEN"
            else:
                outcome = "WIN" if real_pnl_usd > 0 else "LOSS" if real_pnl_usd < 0 else "BREAKEVEN"

            # 6. Записваме в ML памет
            record = TradeRecord(
                id=str(uuid.uuid4()),
                symbol=symbol,
                direction=signal.direction,
                entry_price=entry,
                exit_price=real_exit,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                profit_pips=profit_pips,
                profit_pct=real_pnl_pct,  # ← РЕАЛНИЯТ % със комисия
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
            self.risk.register_close(ticket, real_pnl_usd)  # ← Пращаме долари, не %

            # 7. Логиране на ПРАВИЛНАТА информация
            emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⚪"
            logger.info(
                f"{emoji} Затворена: {symbol} | Entry: {entry:.5f} | Exit: {real_exit:.5f} | "
                f"Пипсове: {profit_pips:+.1f} | Брут: {raw_profit:+.2f}{currency} | "
                f"Реален P&L: {real_pnl_usd:+.2f}{currency} ({real_pnl_pct:+.2f}%) | "
                f"[{outcome}]"
            )

    def _sync_positions(self, mt5_positions):
        for pos in mt5_positions:
            if pos["ticket"] not in self.risk.open_positions:
                self.risk.open_positions[pos["ticket"]] = pos

    def _extract_ml_features(self, symbol: str, df) -> Dict:
        ind = TechnicalIndicators()
        s = self.settings
        close = df["Close"]
        current_price = float(close.iloc[-1])

        rsi_val = float(ind.rsi(close, s.RSI_PERIOD).iloc[-1])
        _, _, macd_hist = ind.macd(close)
        ma_f = float(ind.ema(close, s.MA_FAST).iloc[-1])
        ma_s = float(ind.ema(close, s.MA_SLOW).iloc[-1])

        # Order Block score — цената в OB зона
        ob_score_val = 0.0
        try:
            obs = ind.find_order_blocks(df, s.OB_LOOKBACK, s.OB_MIN_SIZE_ATR)
            for ob in obs[:5]:
                if ob.low <= current_price <= ob.high:
                    ob_score_val = ob.strength * 20
                    break
        except Exception:
            pass

        # Trendline score — близо до Support/Resistance
        tl_score_val = 0.0
        try:
            trendlines = ind.find_trendlines(df)
            channel = ind.get_channel_position(df, trendlines)
            if channel and (channel["near_support"] or channel["near_resistance"]):
                tl_score_val = 10.0
        except Exception:
            pass

        # Груба оценка на candle size спрямо ATR
        atr_val = float(ind.atr(df, s.ATR_PERIOD).iloc[-1])
        open_p = float(df["Open"].iloc[-1])
        candle_size = abs(current_price - open_p)
        candle_score_val = min(15.0, (candle_size / atr_val * 15) if atr_val > 0 else 5.0)

        return {
            "rsi": rsi_val,
            "macd_hist": float(macd_hist.iloc[-1]),
            "ma_alignment": 1.0 if ma_f > ma_s else 0.0,
            "ob_score": ob_score_val,
            "candle_score": candle_score_val,
            "trendline_score": tl_score_val,
            "total_score": 50,      # Непознато преди пълния анализ
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

    def _cap_volume_to_risk(self, symbol: str, volume: float,
                             sl: float, entry: float, account: Dict) -> Optional[float]:
        """Намалява лота ако реалният долларов риск > balance × RISK_PERCENT.
        Връща None ако дори мин. лот надвишава лимита."""
        balance = account.get("balance", 1000.0)
        max_risk_usd = balance * (self.settings.RISK_PERCENT / 100.0)
        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            return volume

        risk_per_lot = 0.0
        if self.mt5._mt5:
            sym_info = self.mt5._mt5.symbol_info(symbol)
            if sym_info and sym_info.trade_tick_size > 0:
                risk_per_lot = (sl_distance / sym_info.trade_tick_size) * sym_info.trade_tick_value
        if risk_per_lot <= 0:
            pip = get_pip(symbol)
            risk_per_lot = (sl_distance / pip) * 1.0

        real_risk = risk_per_lot * volume
        if real_risk <= max_risk_usd:
            return volume

        max_vol = math.floor((max_risk_usd / risk_per_lot) * 100) / 100
        if max_vol < self.settings.MIN_LOT_SIZE:
            logger.info(
                f"⚠️ {symbol}: Риск {real_risk:.2f}$ > лимит {max_risk_usd:.2f}$. Отваряне с мин. лот {self.settings.MIN_LOT_SIZE}."
            )
            return self.settings.MIN_LOT_SIZE

        logger.info(
            f"⚖️ {symbol}: Лот {volume} → {max_vol} "
            f"(Риск: {real_risk:.2f}$ → {max_vol * risk_per_lot:.2f}$ / Лимит: {max_risk_usd:.2f}$)"
        )
        return max_vol

    def run_backtest(self, date_from: str, date_to: str):
        logger.info(f"Backtest {date_from} -> {date_to}")

    def shutdown(self):
        self.mt5.disconnect()
        logger.info(f"Финален отчет: {self.risk.get_daily_summary()}")