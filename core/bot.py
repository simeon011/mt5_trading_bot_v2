"""
core/bot.py — Главен оркестратор
✅ Paper mode премахнат — само live
✅ ML tp/sl мултипликатори се прилагат
✅ Hourly trade limit работи
✅ candle_score е нормализиран (0-15)
✅ OB score мултипликатор унифициран (x15)
✅ be_activated работи без mode проверка
✅ Почистен floating docstring
"""

import logging
import math
import time
import uuid
from datetime import datetime
from typing import Dict, Set, List, Optional

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from strategies.signal_engine import SignalEngine, get_pip, TechnicalIndicators
from ml.learning_agent import LearningAgent, TradeRecord

logger = logging.getLogger("TradingBot")


class TradingBot:

    def __init__(self, settings):
        self.settings = settings

        logger.info("Инициализиране на модули...")
        self.mt5 = MT5Connector(settings)
        self.risk = RiskManager(settings)
        self.signals = SignalEngine(settings)
        self.ml = LearningAgent(settings)

        self._pending_trades: Dict = {}
        self._cycle_count = 0
        self._last_signal_cycle: Dict[str, int] = {}
        self._hourly_trades: List[float] = []   # timestamps на отворени сделки

        if not self.mt5.connect():
            raise RuntimeError("Не може да се свърже с MT5!")
        logger.info("Всички модули инициализирани.")

    def run_cycle(self):
        self._cycle_count += 1
        (logger.cycle if hasattr(logger, 'cycle') else logger.info)(
            f"–––––––––––––––Цикъл #{self._cycle_count} "
            f"[{datetime.now().strftime('%H:%M:%S')}]–––––––––––––––"
        )

        account = self.mt5.get_account_info()
        open_positions = self.mt5.get_open_positions()

        self.signals.update_balance(account.get("balance", 1000.0))

        self._sync_positions(open_positions)
        self._manage_open_positions(open_positions, account)
        self._check_closed_trades(open_positions)
        self._analyze_and_trade(account)

        if self._cycle_count % 30 == 0:
            self._print_daily_report(account)

    # ─────────────────────────────────────────────────────────────────────
    def _analyze_and_trade(self, account: Dict):
        s = self.settings
        traded_this_cycle: Set[str] = set()

        # ── Hourly trade limit ────────────────────────────────────────────
        now_ts = time.time()
        self._hourly_trades = [t for t in self._hourly_trades if now_ts - t < 3600]
        if len(self._hourly_trades) >= s.MAX_TRADES_PER_HOUR:
            logger.info(f"⏰ Лимит {s.MAX_TRADES_PER_HOUR} сделки/час достигнат")
            return

        for symbol in s.SYMBOLS:
            if symbol in traded_this_cycle:
                continue

            last = self._last_signal_cycle.get(symbol, 0)
            if self._cycle_count - last < s.COOLDOWN_CYCLES:
                continue

            try:
                # ── Данни (правилни имена) ─────────────────────────────────
                df_primary = self.mt5.get_candles(symbol, s.PRIMARY_TF, s.CANDLE_HISTORY)
                df_higher  = self.mt5.get_candles(symbol, s.HIGHER_TF, 200)
                df_entry   = self.mt5.get_candles(symbol, s.ENTRY_TF, 100)

                if df_primary is None or df_higher is None or df_entry is None:
                    continue

                ml_features = self._extract_ml_features(symbol, df_primary)
                ml_pred = self.ml.predict(ml_features) if s.ML_ENABLED else 0.5

                # ── ML параметри (обем + tp/sl мултипликатори) ────────────
                smart_params = self.ml.get_smart_trade_params(symbol, ml_pred, ml_features)
                volume    = smart_params["volume"]
                tp_mult   = smart_params.get("tp_multiplier", 1.5)
                sl_mult   = smart_params.get("sl_multiplier", 1.0)

                # ── Сигнал с ML мултипликатори ────────────────────────────
                signal = self.signals.analyze(
                    symbol, df_primary, df_higher, df_entry, ml_pred,
                    tp_mult=tp_mult, sl_mult=sl_mult
                )

                if signal.direction in ("NEUTRAL", "WAIT"):
                    continue

                # ── Долларов риск контрол ─────────────────────────────────
                volume = self._cap_volume_to_risk(
                    symbol, volume, signal.stop_loss, signal.entry_price, account
                )
                if volume is None:
                    continue

                if signal.score < s.MIN_SIGNAL_SCORE:
                    continue

                allowed, reason = self.risk.can_trade(symbol, account)
                if not allowed:
                    logger.info(f"SKIP {symbol}: {reason}")
                    continue

                if signal.risk_reward < s.MIN_RR_RATIO:
                    logger.info(
                        f"SKIP {symbol}: R:R={signal.risk_reward:.2f} < {s.MIN_RR_RATIO}"
                    )
                    continue

                # ── Динамично TP (TP1 за малки лотове) ───────────────────
                final_tp = signal.take_profit
                if volume < s.PARTIAL_CLOSE_MIN_LOT:
                    dist = abs(signal.take_profit - signal.entry_price)
                    if signal.direction == "BUY":
                        final_tp = round(signal.entry_price + dist * 0.70, 5)
                    else:
                        final_tp = round(signal.entry_price - dist * 0.70, 5)
                    logger.info(
                        f"📐 {symbol}: Малък лот ({volume}). Зададен единичен TP на ниво TP1."
                    )

                logger.info(
                    f"SIGNAL {symbol} | {signal.direction} | Лот: {volume} | "
                    f"Score:{signal.score:.1f} | "
                    f"TP Level: {'TP1' if volume < s.PARTIAL_CLOSE_MIN_LOT else 'TP2'}"
                )

                order = self.mt5.place_order(
                    symbol=symbol,
                    order_type=signal.direction,
                    volume=volume,
                    sl=signal.stop_loss,
                    tp=final_tp,
                    comment=f"AI_{signal.score:.0f}"
                )

                if order:
                    self.risk.register_open(order["ticket"], {
                        "symbol": symbol,
                        "type": signal.direction,
                        "volume": volume,
                        "sl": signal.stop_loss,
                        "tp": final_tp,
                        "entry": signal.entry_price,
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
                    self._hourly_trades.append(time.time())   # ← брои за hourly limit

            except Exception as e:
                logger.error(f"Грешка при анализ на {symbol}: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────
    def _manage_open_positions(self, positions, account):
        """Trailing Stop и Break-even + 2 pips защита."""
        ind = TechnicalIndicators()

        for pos in positions:
            try:
                symbol = pos["symbol"]
                df = self.mt5.get_candles(symbol, self.settings.PRIMARY_TF, 50)
                if df is None or df.empty:
                    continue

                atr           = ind.atr(df).iloc[-1]
                current_price = df["Close"].iloc[-1]
                pip_unit      = get_pip(symbol)

                target_pips  = abs(pos["tp"] - pos["price_open"])
                current_pips = abs(current_price - pos["price_open"])
                # fix: progress спрямо пълния TP, не половината
                progress_pct = (current_pips / target_pips * 100) if target_pips > 0 else 0

                # ── ПРАВИЛО 1: BE+2 (45% към TP1) ────────────────────────
                if 45 <= progress_pct < 100:
                    if pos["type"] == "BUY":
                        new_sl = pos["price_open"] + (2 * pip_unit)
                        if new_sl > pos["sl"] + (atr * 0.05):
                            if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                logger.info(
                                    f"🛡️ BE+2 {symbol}: SL → Entry+2пипа ({new_sl:.5f})"
                                )
                                if pos["ticket"] in self._pending_trades:
                                    self._pending_trades[pos["ticket"]]["be_activated"] = True

                    elif pos["type"] == "SELL":
                        new_sl = pos["price_open"] - (2 * pip_unit)
                        if new_sl < pos["sl"] - (atr * 0.05):
                            if self.mt5.modify_sl(pos["ticket"], new_sl, pos["tp"]):
                                logger.info(
                                    f"🛡️ BE+2 {symbol}: SL → Entry-2пипа ({new_sl:.5f})"
                                )
                                if pos["ticket"] in self._pending_trades:
                                    self._pending_trades[pos["ticket"]]["be_activated"] = True

                # ── ПРАВИЛО 2: TRAILING STOP (след TP1) ──────────────────
                if progress_pct >= 100 and self.settings.USE_TRAILING_STOP:
                    pos_info = self.risk.open_positions.get(pos["ticket"], pos)
                    new_trail_sl = self.risk.calculate_trailing_stop(
                        pos_info, current_price, float(atr)
                    )
                    if new_trail_sl is not None:
                        if self.mt5.modify_sl(pos["ticket"], new_trail_sl, pos["tp"]):
                            logger.debug(
                                f"🔄 Trailing Stop {symbol}: SL → {new_trail_sl:.5f}"
                            )

            except Exception as e:
                logger.debug(
                    f"❌ Грешка при управление на позиция {pos.get('symbol')}: {e}"
                )

    # ─────────────────────────────────────────────────────────────────────
    def _check_closed_trades(self, current_positions):
        """Проверява затворени сделки и записва P&L."""
        current_tickets = {p["ticket"] for p in current_positions}
        closed_tickets  = set(self._pending_trades.keys()) - current_tickets

        if not closed_tickets:
            return

        account_info = self.mt5.get_account_info()
        balance  = account_info.get("balance", 1000.0)
        currency = account_info.get("currency", "USD")

        COMMISSION_PER_001_LOT = 0.03

        for ticket in list(closed_tickets):
            pending    = self._pending_trades.pop(ticket)
            signal     = pending["signal"]
            ml_features = pending["ml_features"]
            entry      = pending["entry_price"]
            symbol     = signal.symbol
            initial_vol = pending.get("initial_volume", 0.01)

            # 1. P&L от MT5
            deal_info = self.mt5.get_deal_info(ticket)
            if deal_info:
                real_exit    = deal_info["price"]
                raw_profit   = deal_info["profit"]
                real_pnl_usd = deal_info["net"]
            else:
                logger.warning(
                    f"Ticket {ticket} ({symbol}) не е намерен в историята. Fallback."
                )
                try:
                    df_temp   = self.mt5.get_candles(symbol, "M1", 1)
                    real_exit = float(df_temp["Close"].iloc[-1]) \
                                if df_temp is not None and not df_temp.empty else entry
                except Exception:
                    real_exit = entry

                sym_info = self.mt5._mt5.symbol_info(symbol) if self.mt5._mt5 else None
                if sym_info and sym_info.trade_tick_size > 0:
                    if signal.direction == "BUY":
                        raw_profit = ((real_exit - entry)
                                      / sym_info.trade_tick_size
                                      * sym_info.trade_tick_value
                                      * initial_vol)
                    else:
                        raw_profit = ((entry - real_exit)
                                      / sym_info.trade_tick_size
                                      * sym_info.trade_tick_value
                                      * initial_vol)
                else:
                    raw_profit = 0.0

                TOTAL_COMMISSION = round((initial_vol / 0.01) * COMMISSION_PER_001_LOT * 2, 4)
                real_pnl_usd = raw_profit - TOTAL_COMMISSION

            real_pnl_pct = (real_pnl_usd / balance) * 100 if balance > 0 else 0

            # 2. Пипсове
            pip_unit = get_pip(symbol)
            if signal.direction == "BUY":
                profit_pips = (real_exit - entry) / pip_unit
            else:
                profit_pips = (entry - real_exit) / pip_unit

            # 3. Резултат
            be_activated = pending.get("be_activated", False)
            if be_activated and profit_pips <= 5:
                outcome = "BREAKEVEN"
            else:
                outcome = ("WIN" if real_pnl_usd > 0
                           else "LOSS" if real_pnl_usd < 0
                           else "BREAKEVEN")

            # 4. ML запис (candle_score вече е нормализиран в signal_engine)
            record = TradeRecord(
                id=str(uuid.uuid4()),
                symbol=symbol,
                direction=signal.direction,
                entry_price=entry,
                exit_price=real_exit,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                profit_pips=profit_pips,
                profit_pct=real_pnl_pct,
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
            self.risk.register_close(ticket, real_pnl_usd)

            # 6. Лог
            emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⚪"
            logger.info(
                f"{emoji} Затворена: {symbol} | Entry: {entry:.5f} | Exit: {real_exit:.5f} | "
                f"Пипсове: {profit_pips:+.1f} | Брут: {raw_profit:+.2f}{currency} | "
                f"Реален P&L: {real_pnl_usd:+.2f}{currency} ({real_pnl_pct:+.2f}%) | "
                f"[{outcome}]"
            )

    # ─────────────────────────────────────────────────────────────────────
    def _sync_positions(self, mt5_positions):
        for pos in mt5_positions:
            if pos["ticket"] not in self.risk.open_positions:
                self.risk.open_positions[pos["ticket"]] = pos

    def _extract_ml_features(self, symbol: str, df) -> Dict:
        ind = TechnicalIndicators()
        s   = self.settings
        close         = df["Close"]
        current_price = float(close.iloc[-1])

        rsi_val         = float(ind.rsi(close, s.RSI_PERIOD).iloc[-1])
        _, _, macd_hist = ind.macd(close)
        ma_f            = float(ind.ema(close, s.MA_FAST).iloc[-1])
        ma_s            = float(ind.ema(close, s.MA_SLOW).iloc[-1])
        atr_val         = float(ind.atr(df, s.ATR_PERIOD).iloc[-1])

        # OB score (унифициран x15)
        ob_score_val = 0.0
        try:
            obs = ind.find_order_blocks(df, s.OB_LOOKBACK, s.OB_MIN_SIZE_ATR)
            for ob in obs[:5]:
                if ob.low <= current_price <= ob.high:
                    ob_score_val = ob.strength * 15   # ← унифициран (беше 20)
                    break
        except Exception:
            pass

        # Trendline score
        tl_score_val = 0.0
        try:
            trendlines = ind.find_trendlines(df)
            channel    = ind.get_channel_position(df, trendlines)
            if channel and (channel["near_support"] or channel["near_resistance"]):
                tl_score_val = 10.0
        except Exception:
            pass

        # Нормализиран candle score (0-15)
        open_p      = float(df["Open"].iloc[-1])
        candle_size = abs(current_price - open_p)
        candle_score_val = min(15.0, (candle_size / atr_val * 15) if atr_val > 0 else 5.0)

        # Приблизителен total_score от основните индикатори (за predict())
        # Бонусите от SMC/Volume не са известни тук; signal.score се записва при затваряне
        _b, _s = 0, 0
        if rsi_val < 40:       _b += 20
        elif rsi_val > 60:     _s += 20
        _mh = float(macd_hist.iloc[-1])
        if _mh > 0:            _b += 15
        elif _mh < 0:          _s += 15
        if ma_f > ma_s:        _b += 25
        else:                  _s += 25
        _b += ob_score_val + candle_score_val + tl_score_val
        total_score_approx = min(100.0, float(max(_b, _s)))

        return {
            "rsi":           rsi_val,
            "macd_hist":     _mh,
            "ma_alignment":  1.0 if ma_f > ma_s else 0.0,
            "ob_score":      ob_score_val,
            "candle_score":  candle_score_val,
            "trendline_score": tl_score_val,
            "total_score":   total_score_approx,
            "hour":          datetime.now().hour,
            "day_of_week":   datetime.now().weekday(),
            "direction":     "BUY" if ma_f > ma_s else "SELL"
        }

    def _print_daily_report(self, account: Dict):
        summary  = self.risk.get_daily_summary()
        ml_stats = self.ml.get_stats()
        logger.info("=" * 50)
        logger.info(f"ДНЕВЕН ОТЧЕТ [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
        logger.info(f"  Баланс:  {account.get('balance', 0):.2f} {account.get('currency', '')}")
        logger.info(f"  PnL:     {summary['daily_pnl']:+.2f}$")
        logger.info(f"  Сделки:  {summary['daily_trades']}")
        if ml_stats:
            logger.info(
                f"  WinRate: {ml_stats.get('win_rate', 0):.1%} "
                f"({ml_stats.get('total_trades', 0)} сделки)"
            )
        logger.info("=" * 50)

    def _cap_volume_to_risk(self, symbol: str, volume: float,
                             sl: float, entry: float, account: Dict) -> Optional[float]:
        balance      = account.get("balance", 1000.0)
        max_risk_usd = balance * (self.settings.RISK_PERCENT / 100.0)
        sl_distance  = abs(entry - sl)
        if sl_distance == 0:
            return volume

        risk_per_lot = 0.0
        if self.mt5._mt5:
            sym_info = self.mt5._mt5.symbol_info(symbol)
            if sym_info and sym_info.trade_tick_size > 0:
                risk_per_lot = (sl_distance / sym_info.trade_tick_size) * sym_info.trade_tick_value
        if risk_per_lot <= 0:
            pip          = get_pip(symbol)
            risk_per_lot = (sl_distance / pip) * 1.0

        real_risk = risk_per_lot * volume
        if real_risk <= max_risk_usd:
            return volume

        max_vol = math.floor((max_risk_usd / risk_per_lot) * 100) / 100
        if max_vol < self.settings.MIN_LOT_SIZE:
            logger.info(
                f"⚠️ {symbol}: Риск {real_risk:.2f}$ > лимит {max_risk_usd:.2f}$. "
                f"Отваряне с мин. лот {self.settings.MIN_LOT_SIZE}."
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
