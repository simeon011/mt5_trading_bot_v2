"""
strategies/signal_engine.py — Комбинира всички индикатори в един търговски сигнал
Всеки индикатор дава точки → общ score 0-100 → решение BUY/SELL/NEUTRAL
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging

from strategies.indicators import (
    TechnicalIndicators, OrderBlock, Trendline, CandlePattern
)

logger = logging.getLogger("SignalEngine")


@dataclass
class TradeSignal:
    symbol: str
    direction: str          # "BUY", "SELL", "NEUTRAL"
    score: float            # 0-100
    confidence: float       # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float

    # Детайли по компонент
    rsi_score: float = 0
    macd_score: float = 0
    ma_score: float = 0
    ob_score: float = 0
    trendline_score: float = 0
    candle_score: float = 0
    ml_score: float = 0

    # Контекст
    active_ob: Optional[OrderBlock] = None
    candle_patterns: List[CandlePattern] = field(default_factory=list)
    reasoning: str = ""


class SignalEngine:
    """
    Scoring система:
    ─────────────────────────────────────────
    RSI                    → 0-15 точки
    MACD                   → 0-15 точки
    Moving Averages        → 0-20 точки
    Order Blocks           → 0-20 точки
    Trendlines/Channels    → 0-15 точки
    Candlestick Patterns   → 0-15 точки
    ML корекция            → ±10 точки
    ─────────────────────────────────────────
    TOTAL                  → 0-110 → нормализира до 0-100
    Праг за сигнал         → 65 точки
    """

    def __init__(self, settings):
        self.s = settings
        self.ind = TechnicalIndicators()

    def analyze(self, symbol: str, df_h1: pd.DataFrame,
                df_h4: pd.DataFrame, df_m15: pd.DataFrame,
                ml_prediction: Optional[float] = None) -> TradeSignal:
        """
        Пълен анализ на символ върху 3 таймфрейма.
        Връща TradeSignal с BUY/SELL/NEUTRAL и score.
        """
        ind = self.ind
        s = self.s

        # ── Изчисляване на индикатори (H1) ──────────────────
        close = df_h1["Close"]
        atr = ind.atr(df_h1, s.ATR_PERIOD)
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        rsi = ind.rsi(close, s.RSI_PERIOD)
        macd_line, macd_sig, macd_hist = ind.macd(close, s.MACD_FAST, s.MACD_SLOW, s.MACD_SIGNAL)
        ma_fast = ind.ema(close, s.MA_FAST)
        ma_slow = ind.ema(close, s.MA_SLOW)
        ma_trend = ind.ema(close, s.MA_TREND)

        # ── H4 тренд ────────────────────────────────────────
        h4_ma_fast = ind.ema(df_h4["Close"], s.MA_FAST)
        h4_ma_slow = ind.ema(df_h4["Close"], s.MA_SLOW)
        h4_bullish = h4_ma_fast.iloc[-1] > h4_ma_slow.iloc[-1]
        h4_bearish = h4_ma_fast.iloc[-1] < h4_ma_slow.iloc[-1]

        # ── Order Blocks ─────────────────────────────────────
        order_blocks = ind.find_order_blocks(df_h1, s.OB_LOOKBACK, s.OB_MIN_SIZE_ATR)

        # ── Trendlines & Channels ────────────────────────────
        trendlines = ind.find_trendlines(df_h1)
        channel = ind.get_channel_position(df_h1, trendlines)

        # ── Candle Patterns (M15 за прецизност) ─────────────
        candle_patterns_m15 = ind.find_candle_patterns(df_m15)
        candle_patterns_h1 = ind.find_candle_patterns(df_h1)
        all_patterns = candle_patterns_h1 + candle_patterns_m15

        # ═══════════════════════════════════════════════════
        # SCORING — Bullish
        # ═══════════════════════════════════════════════════
        bull_score = 0
        bear_score = 0
        reasoning_parts = []

        # 1. RSI (0-15)
        rsi_val = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2] if len(rsi) > 1 else rsi_val
        if rsi_val < s.RSI_OVERSOLD:
            bull_rsi = 15
            bear_rsi = 0
            reasoning_parts.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val > s.RSI_OVERBOUGHT:
            bull_rsi = 0
            bear_rsi = 15
            reasoning_parts.append(f"RSI overbought ({rsi_val:.1f})")
        elif 40 <= rsi_val <= 60:
            # Зона на несигурност
            bull_rsi = 5
            bear_rsi = 5
        elif rsi_val < 50:
            bull_rsi = 3
            bear_rsi = 8
        else:
            bull_rsi = 8
            bear_rsi = 3
        # RSI momentum (пресичане на 50)
        if rsi_prev < 50 <= rsi_val:
            bull_rsi += 3
        elif rsi_prev > 50 >= rsi_val:
            bear_rsi += 3
        bull_score += bull_rsi
        bear_score += bear_rsi

        # 2. MACD (0-15)
        macd_val = macd_line.iloc[-1]
        macd_sig_val = macd_sig.iloc[-1]
        macd_hist_val = macd_hist.iloc[-1]
        macd_hist_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        # Crossover
        macd_cross_bull = macd_hist_prev < 0 and macd_hist_val > 0
        macd_cross_bear = macd_hist_prev > 0 and macd_hist_val < 0
        if macd_cross_bull:
            bull_macd = 15
            bear_macd = 0
            reasoning_parts.append("MACD bullish crossover")
        elif macd_cross_bear:
            bull_macd = 0
            bear_macd = 15
            reasoning_parts.append("MACD bearish crossover")
        elif macd_hist_val > 0 and macd_hist_val > macd_hist_prev:
            bull_macd = 10
            bear_macd = 2
        elif macd_hist_val < 0 and macd_hist_val < macd_hist_prev:
            bull_macd = 2
            bear_macd = 10
        else:
            bull_macd = 5
            bear_macd = 5
        bull_score += bull_macd
        bear_score += bear_macd

        # 3. Moving Averages (0-20, включително H4 потвърждение)
        bull_ma = bear_ma = 0
        ma_f = ma_fast.iloc[-1]
        ma_s = ma_slow.iloc[-1]
        ma_t = ma_trend.iloc[-1]
        if current_price > ma_t:
            bull_ma += 5
        else:
            bear_ma += 5
        if ma_f > ma_s:
            bull_ma += 5
        else:
            bear_ma += 5
        if current_price > ma_f > ma_s:
            bull_ma += 5
            reasoning_parts.append("Price above MA stack")
        elif current_price < ma_f < ma_s:
            bear_ma += 5
            reasoning_parts.append("Price below MA stack")
        # H4 потвърждение
        if h4_bullish:
            bull_ma += 5
        elif h4_bearish:
            bear_ma += 5
        bull_score += bull_ma
        bear_score += bear_ma

        # 4. Order Blocks (0-20)
        bull_ob = bear_ob = 0
        nearest_bull_ob = None
        nearest_bear_ob = None
        for ob in order_blocks:
            dist = abs(current_price - (ob.high + ob.low) / 2)
            if dist < current_atr * 1.5 and not ob.broken:
                if ob.ob_type == "BULLISH" and current_price >= ob.low and current_price <= ob.high * 1.01:
                    bull_ob = int(15 * ob.strength)
                    nearest_bull_ob = ob
                    reasoning_parts.append(f"Price at Bullish OB ({ob.strength:.0%})")
                    break
                elif ob.ob_type == "BEARISH" and current_price <= ob.high and current_price >= ob.low * 0.99:
                    bear_ob = int(15 * ob.strength)
                    nearest_bear_ob = ob
                    reasoning_parts.append(f"Price at Bearish OB ({ob.strength:.0%})")
                    break
        bull_score += bull_ob
        bear_score += bear_ob

        # 5. Trendlines & Channels (0-15)
        bull_tl = bear_tl = 0
        if channel:
            if channel["near_support"]:
                bull_tl = 12
                reasoning_parts.append("Price at channel support")
            elif channel["near_resistance"]:
                bear_tl = 12
                reasoning_parts.append("Price at channel resistance")
            else:
                bull_tl = bear_tl = 5
        else:
            # Проверка за trendline break
            for tl in trendlines[:3]:
                tl_price = tl.slope * (len(df_h1)-1) + tl.intercept
                if tl.line_type == "RESISTANCE" and current_price > tl_price:
                    bull_tl = max(bull_tl, int(10 * tl.strength))
                    reasoning_parts.append("Resistance trendline break")
                elif tl.line_type == "SUPPORT" and current_price < tl_price:
                    bear_tl = max(bear_tl, int(10 * tl.strength))
                    reasoning_parts.append("Support trendline break")
        bull_score += bull_tl
        bear_score += bear_tl

        # 6. Candlestick Patterns (0-15)
        bull_cp = bear_cp = 0
        for cp in all_patterns:
            if cp.direction == "BULLISH":
                bull_cp = max(bull_cp, int(15 * cp.strength))
                reasoning_parts.append(f"{cp.name} pattern")
            elif cp.direction == "BEARISH":
                bear_cp = max(bear_cp, int(15 * cp.strength))
                reasoning_parts.append(f"{cp.name} pattern")
        bull_score += bull_cp
        bear_score += bear_cp

        # 7. ML корекция (±10)
        ml_bull = ml_bear = 0
        if ml_prediction is not None:
            if ml_prediction > 0.6:
                ml_bull = int(10 * (ml_prediction - 0.5) * 2)
                reasoning_parts.append(f"ML: bullish ({ml_prediction:.0%})")
            elif ml_prediction < 0.4:
                ml_bear = int(10 * (0.5 - ml_prediction) * 2)
                reasoning_parts.append(f"ML: bearish ({1-ml_prediction:.0%})")
        bull_score += ml_bull
        bear_score += ml_bear

        # ═══════════════════════════════════════════════════
        # РЕШЕНИЕ
        # ═══════════════════════════════════════════════════
        max_possible = 110
        bull_pct = (bull_score / max_possible) * 100
        bear_pct = (bear_score / max_possible) * 100

        active_ob = nearest_bull_ob if bull_pct > bear_pct else nearest_bear_ob

        if bull_pct >= s.MIN_SIGNAL_SCORE and bull_pct > bear_pct + 10:
            direction = "BUY"
            final_score = bull_pct
        elif bear_pct >= s.MIN_SIGNAL_SCORE and bear_pct > bull_pct + 10:
            direction = "SELL"
            final_score = bear_pct
        else:
            direction = "NEUTRAL"
            final_score = max(bull_pct, bear_pct)

        # ── SL/TP изчисляване ────────────────────────────────
        sl, tp = self._calculate_sl_tp(
            direction, current_price, current_atr, active_ob, df_h1, trendlines
        )
        rr = abs(tp - current_price) / abs(sl - current_price) if sl != current_price else 0

        confidence = min(1.0, final_score / 100)
        reasoning = " | ".join(reasoning_parts[:5]) if reasoning_parts else "Няма силни сигнали"

        logger.info(f"{symbol} → {direction} | Score: {final_score:.1f} "
                    f"(Bull:{bull_pct:.1f} Bear:{bear_pct:.1f}) | {reasoning}")

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            score=final_score,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            risk_reward=rr,
            rsi_score=bull_rsi if direction != "SELL" else bear_rsi,
            macd_score=bull_macd if direction != "SELL" else bear_macd,
            ma_score=bull_ma if direction != "SELL" else bear_ma,
            ob_score=bull_ob if direction != "SELL" else bear_ob,
            trendline_score=bull_tl if direction != "SELL" else bear_tl,
            candle_score=bull_cp if direction != "SELL" else bear_cp,
            ml_score=ml_bull if direction != "SELL" else ml_bear,
            active_ob=active_ob,
            candle_patterns=[p for p in all_patterns if p.direction != "NEUTRAL"],
            reasoning=reasoning
        )

    def _calculate_sl_tp(self, direction: str, price: float, atr: float,
                         ob: Optional[OrderBlock], df: pd.DataFrame,
                         trendlines: List) -> tuple:
        """Изчислява SL и TP базирани на структурата."""
        s = self.s

        if direction == "BUY":
            # SL: под Order Block ако има, иначе ATR-базиран
            if ob and ob.ob_type == "BULLISH":
                sl = ob.low - atr * 0.2
            else:
                sl = price - atr * s.DEFAULT_SL_ATR_MULT
            tp = price + (price - sl) * s.DEFAULT_TP_RR_RATIO
        elif direction == "SELL":
            if ob and ob.ob_type == "BEARISH":
                sl = ob.high + atr * 0.2
            else:
                sl = price + atr * s.DEFAULT_SL_ATR_MULT
            tp = price - (sl - price) * s.DEFAULT_TP_RR_RATIO
        else:
            sl = price - atr
            tp = price + atr

        return round(sl, 5), round(tp, 5)
