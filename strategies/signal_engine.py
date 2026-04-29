"""
strategies/signal_engine.py — Главен анализатор на сигнали
✅ Включва: Многократен анализ (M5, H1, H4), ML Интеграция, ATR Снайпер филтър
✅ НОВО: Анализ на обема (Volume Spike) и подобрени логове
"""

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import math

from strategies.indicators import TechnicalIndicators

logger = logging.getLogger("ScalperEngine")

def get_pip(symbol: str) -> float:
    """Връща стойността на 1 пип за дадения символ."""
    return 0.01 if "JPY" in symbol else 0.0001

@dataclass
class TradeSignal:
    symbol: str
    direction: str          # "BUY", "SELL", "NEUTRAL", "WAIT"
    score: float            # Мощност на сигнала
    confidence: float       # Процент сигурност (напр. 85.0)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasoning: str = ""
    ob_score: float = 0.0
    candle_score: float = 0.0
    trendline_score: float = 0.0


class SignalEngine:
    def __init__(self, settings):
        self.s = settings
        self.ind = TechnicalIndicators()
        self.account_balance = 1000.0  # Ще се обновява динамично от бота

    def update_balance(self, balance: float):
        self.account_balance = balance

    def analyze(self, symbol: str, df_primary: pd.DataFrame,
                df_higher: pd.DataFrame, df_entry: pd.DataFrame,
                ml_prediction: Optional[float] = None) -> TradeSignal:

        ind = self.ind
        s = self.s
        close = df_primary["Close"]
        current_price = float(close.iloc[-1])
        pip = get_pip(symbol)

        # ── ⚡ ОПТИМИЗАЦИЯ: Изчисляваме ATR само ВЕДНЪЖ тук най-отгоре
        atr_series = ind.atr(df_primary, s.ATR_PERIOD)
        current_atr = float(atr_series.iloc[-1])

        # ── ATR Filter ───────────────────────────────────────
        if s.ATR_FILTER_ENABLED:
            avg_atr = float(atr_series.tail(s.ATR_AVERAGE_PERIOD).mean())

            if avg_atr > 0:
                atr_ratio = current_atr / avg_atr

                if atr_ratio > s.ATR_MAX_MULTIPLIER:
                    logger.info(f"⚠️ {symbol:8} HIGH VOL | Ratio: {atr_ratio:.2f}x (max {s.ATR_MAX_MULTIPLIER}x)")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0, confidence=0,
                        entry_price=current_price, stop_loss=current_price, take_profit=current_price, risk_reward=0
                    )

                if atr_ratio < s.ATR_MIN_MULTIPLIER:
                    logger.info(f"💤 {symbol:8} LOW VOL  | Ratio: {atr_ratio:.2f}x (min {s.ATR_MIN_MULTIPLIER}x)")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0, confidence=0,
                        entry_price=current_price, stop_loss=current_price, take_profit=current_price, risk_reward=0
                    )

        # ── Изчисляване на индикаторите ──────────────────────────────────
        # RSI
        rsi_series = ind.rsi(close, s.RSI_PERIOD)
        current_rsi = float(rsi_series.iloc[-1])

        # MACD
        macd_line, signal_line, macd_hist = ind.macd(close)
        current_macd_hist = float(macd_hist.iloc[-1])

        # Moving Averages (Trend)
        ma_fast = float(ind.ema(close, s.MA_FAST).iloc[-1])
        ma_slow = float(ind.ema(close, s.MA_SLOW).iloc[-1])

        # ── ТОЧКОВА СИСТЕМА (Scoring) ────────────────────────────────────
        b_score = 0
        s_score = 0

        # 1. RSI Логика
        if current_rsi < 40:
            b_score += 20
        elif current_rsi > 60:
            s_score += 20

        # 2. MACD Логика
        if current_macd_hist > 0:
            b_score += 15
        elif current_macd_hist < 0:
            s_score += 15

        # 3. Trend Логика (Moving Averages)
        if ma_fast > ma_slow:
            b_score += 25
        elif ma_fast < ma_slow:
            s_score += 25

        # 4. Анализ на Свещите (Price Action)
        open_p = float(df_primary["Open"].iloc[-1])
        close_p = float(df_primary["Close"].iloc[-1])
        candle_size = abs(close_p - open_p)

        if candle_size > current_atr * 0.5:
            if close_p > open_p:
                b_score += 15
            elif close_p < open_p:
                s_score += 15

        # ── 📊 АНАЛИЗ НА ОБЕМА (VOLUME SPIKE) ──
        vol_ratio = float(ind.volume_spike(df_primary, period=20))

        # Ако обемът е с поне 50% по-голям от нормалното (ratio > 1.5)
        if vol_ratio > 1.5:
            if close_p > open_p:  # Бича свещ + Голям обем
                b_score += 15
            elif close_p < open_p: # Меча свещ + Голям обем
                s_score += 15

        # 5. Order Blocks (Институционални зони — 15 точки)
        # Цената влиза в Bullish OB → силен BUY сигнал
        # Цената влиза в Bearish OB → силен SELL сигнал
        obs = ind.find_order_blocks(df_primary, s.OB_LOOKBACK, s.OB_MIN_SIZE_ATR)
        ob_score_val = 0.0
        for ob in obs[:5]:
            if ob.low <= current_price <= ob.high:
                if ob.ob_type == "BULLISH":
                    b_score += 15
                    ob_score_val = ob.strength * 15
                elif ob.ob_type == "BEARISH":
                    s_score += 15
                    ob_score_val = ob.strength * 15
                break

        # 6. Trendlines / Channel (Структура на пазара — 10 точки)
        trendlines = ind.find_trendlines(df_primary)
        channel = ind.get_channel_position(df_primary, trendlines)
        tl_score_val = 0.0
        if channel:
            if channel["near_support"]:
                b_score += 10
                tl_score_val = 10.0
            elif channel["near_resistance"]:
                s_score += 10
                tl_score_val = 10.0

        # 7. ML Интеграция (Изкуствен Интелект — 10 точки)
        ml_val = ml_prediction if ml_prediction is not None else 0.5
        if ml_val >= 0.60:
            b_score += 10
        elif ml_val <= 0.40:
            s_score += 10

        # ── СТЪПКА 1: Tentative посока само от класическите индикатори ──
        # RSI, MACD, MA, Candle, Volume, OB, Trendline, ML
        classic_total = b_score + s_score
        if classic_total == 0:
            b_pct_c, s_pct_c = 50.0, 50.0
        else:
            b_pct_c = (b_score / classic_total) * 100
            s_pct_c = (s_score / classic_total) * 100

        if b_pct_c > s_pct_c:
            tentative = "BUY"
        elif s_pct_c > b_pct_c:
            tentative = "SELL"
        else:
            tentative = "WAIT"

        # ── СТЪПКА 2: SMC като GATE + BONUS ─────────────────────────────
        # BOS/CHOCH/EQL никога не дават точки на обратната посока.
        # Те или ПОТВЪРЖДАВАТ (добавят бонус) или БЛОКИРАТ сигнала изцяло.
        ms  = ind.detect_market_structure(df_primary)
        eq  = ind.find_equal_levels(df_primary)

        bos_icon   = ""
        choch_icon = ""
        eq_icon    = ""

        # BOS Бонус: само потвърждава tentative посоката (не блокира)
        if ms.bos_direction == "BULLISH":
            if tentative == "BUY":
                b_score += 20
                bos_icon = "📈 BOS↑"
        elif ms.bos_direction == "BEARISH":
            if tentative == "SELL":
                s_score += 20
                bos_icon = "📉 BOS↓"

        # CHOCH Бонус: по-силен от BOS (не блокира)
        if ms.choch_detected:
            if ms.choch_direction == "BULLISH" and tentative == "BUY":
                b_score += 25
                choch_icon = "🔄 CHOCH↑"
            elif ms.choch_direction == "BEARISH" and tentative == "SELL":
                s_score += 25
                choch_icon = "🔄 CHOCH↓"

        # EQL Бонус: само потвърждава (не блокира)
        if eq.price_swept_eq_low or eq.price_broke_eq_high:
            if tentative == "BUY":
                b_score += 15
                eq_icon = "💧EQL↑" if eq.price_swept_eq_low else "🚀EQH↑"
        elif eq.price_swept_eq_high or eq.price_broke_eq_low:
            if tentative == "SELL":
                s_score += 15
                eq_icon = "💧EQH↓" if eq.price_swept_eq_high else "🔻EQL↓"

        # ── СТЪПКА 3: Финална посока ─────────────────────────────────────
        total_possible = b_score + s_score
        if total_possible == 0:
            b_pct, s_pct = 50.0, 50.0
        else:
            b_pct = (b_score / total_possible) * 100
            s_pct = (s_score / total_possible) * 100

        final_score = max(b_score, s_score)
        direction   = "WAIT"
        reason_str  = ""

        if b_pct > s_pct and b_pct >= 60 and final_score >= s.MIN_SIGNAL_SCORE:
            direction  = "BUY"
            reason_str = "Trend-UP" if ma_fast > ma_slow else "Reversal-UP"
        elif s_pct > b_pct and s_pct >= 60 and final_score >= s.MIN_SIGNAL_SCORE:
            direction  = "SELL"
            reason_str = "Trend-DOWN" if ma_fast < ma_slow else "Reversal-DOWN"

        # ── 🎯 ДИНАМИЧНО ИЗЧИСЛЯВАНЕ НА TP / SL ЧРЕЗ ATR ──
        tp_pips = (current_atr * getattr(self.s, 'tp_multiplier', 1.5)) / pip
        sl_pips = (current_atr * getattr(self.s, 'sl_multiplier', 1.0)) / pip

        # Защита: SL не може да е под 4 пипса (или 5 за JPY)
        min_sl = 5.0 if "JPY" in symbol else 4.0
        sl_pips = max(sl_pips, min_sl)

        # ── 🎯 СНАЙПЕР ФИЛТЪР (Защита от слаби движения) ──
        if tp_pips < s.MIN_TP_PIPS:
            logger.info(
                f"💤 {symbol:8} | СКИП | 🐂 {b_pct:>2.0f}% / 🐻 {s_pct:>2.0f}% | Мощност: {final_score:>2.0f} | 🧠 ML: {ml_val*100:.0f}% | "
                f"⚠️ Отказ: TP ({tp_pips:.1f}p) е под {s.MIN_TP_PIPS}"
            )
            return TradeSignal(
                symbol=symbol, direction="NEUTRAL", score=0, confidence=0, entry_price=current_price,
                stop_loss=current_price, take_profit=current_price, risk_reward=0, reasoning="Sniper Filter"
            )

        # Превръщане на пипсовете в реални ценови нива
        if direction == "BUY":
            sl_price = current_price - (sl_pips * pip)
            tp_price = current_price + (tp_pips * pip)
        elif direction == "SELL":
            sl_price = current_price + (sl_pips * pip)
            tp_price = current_price - (tp_pips * pip)
        else:
            sl_price = current_price
            tp_price = current_price

        # Закръгляне до 5 знака (3 за JPY) за MT5
        decimals = 3 if "JPY" in symbol else 5
        sl_price = round(sl_price, decimals)
        tp_price = round(tp_price, decimals)

        risk = abs(current_price - sl_price)
        reward = abs(tp_price - current_price)
        rr_ratio = reward / risk if risk > 0 else 0

        # ── 📊 ВИЗУАЛНО ТАБЛО (За приетите сигнали) ───────────
        if direction != "WAIT":
            emoji    = "🟢" if direction == "BUY" else "🔴"
            vol_icon = "🔥 VOL!" if vol_ratio > 1.5 else "📊 vol"
            ob_icon  = f"🧱 OB:{ob_score_val:.0f}" if ob_score_val > 0 else ""
            tl_icon  = f"📐 TL:{tl_score_val:.0f}" if tl_score_val > 0 else ""
            smc_icons = " ".join(filter(None, [bos_icon, choch_icon, eq_icon]))

            logger.info(
                f"{emoji} {symbol:8} | {direction:4} | 🐂 {b_pct:>2.0f}% / 🐻 {s_pct:>2.0f}% | "
                f"Мощност: {final_score:>2.0f} | 🧠 ML: {ml_val*100:.0f}% | {vol_icon} ({vol_ratio:.1f}x) | "
                f"{ob_icon} {tl_icon} {smc_icons} | 🎯 SL: {sl_pips:.0f}p / TP: {tp_pips:.0f}p | {reason_str}"
            )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            score=final_score,
            confidence=max(b_pct, s_pct),
            entry_price=current_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            risk_reward=rr_ratio,
            reasoning=reason_str,
            ob_score=ob_score_val,
            candle_score=candle_size,
            trendline_score=tl_score_val
        )