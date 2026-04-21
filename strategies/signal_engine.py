"""
strategies/signal_engine.py — SCALPER VERSION
Бърза scoring система за high-frequency trading на M5/M1
Adaptive TP/SL спрямо баланса на акаунта
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging

from strategies.indicators import TechnicalIndicators, OrderBlock, Trendline, CandlePattern
from config.settings import get_adaptive_pips

logger = logging.getLogger("ScalperEngine")

# Pip стойности за различни символи (1 pip в price units)
PIP_SIZE = {
    "EURUSD": 0.0001, "EURUSDm": 0.0001,
    "GBPUSD": 0.0001, "GBPUSDm": 0.0001,
    "USDJPY": 0.01,   "USDJPYm": 0.01,
    "USDCHF": 0.0001, "USDCHFm": 0.0001,
    "AUDUSD": 0.0001, "AUDUSDm": 0.0001,
    "XAUUSD": 0.1,    "XAUUSDm": 0.1,   "XAUUSD+": 0.1,
    "XAGUSD": 0.01,
    "NAS100": 1.0,    "NAS100m": 1.0,
    "US500":  0.1,    "SP500":   0.1,
    "GER40":  1.0,    "UK100":   1.0,
    "US30":   1.0,
}

def get_pip(symbol: str) -> float:
    """Връща pip стойността за символ. Default: 0.0001"""
    for key, val in PIP_SIZE.items():
        if symbol.upper().startswith(key.upper()):
            return val
    return 0.0001


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    score: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    sl_pips: float = 0
    tp_pips: float = 0
    rsi_score: float = 0
    macd_score: float = 0
    ma_score: float = 0
    ob_score: float = 0
    candle_score: float = 0
    ml_score: float = 0
    trendline_score: float = 0
    candle_patterns: List[CandlePattern] = field(default_factory=list)
    active_ob: Optional[OrderBlock] = None
    reasoning: str = ""


class SignalEngine:
    """
    Scalper scoring система (0-100):
    ─────────────────────────────────
    RSI (M5)          → 0-20 точки
    MACD (M5)         → 0-20 точки
    Moving Averages   → 0-15 точки
    Candle Patterns   → 0-20 точки
    Order Blocks      → 0-15 точки
    M15 Тренд         → 0-10 точки
    ─────────────────────────────────
    TOTAL             → 0-100
    Праг              → 52 точки
    """

    def __init__(self, settings):
        self.s = settings
        self.ind = TechnicalIndicators()
        self._account_balance: float = 1000.0  # Default, обновява се от бота

    def update_balance(self, balance: float):
        """Ботът извиква това за да обнови баланса → adaptive pips."""
        self._account_balance = balance

    def analyze(self, symbol: str, df_primary: pd.DataFrame,
                df_higher: pd.DataFrame, df_entry: pd.DataFrame,
                ml_prediction: Optional[float] = None) -> TradeSignal:
        """
        Анализира символ и връща TradeSignal.
        df_primary = M5, df_higher = M15, df_entry = M1
        """
        ind = self.ind
        s = self.s

        close = df_primary["Close"]
        current_price = float(close.iloc[-1])
        pip = get_pip(symbol)

        # Adaptive пипове спрямо баланса
        if s.USE_ADAPTIVE_PIPS:
            pips = get_adaptive_pips(self._account_balance)
            tp_pips = pips["tp"]
            sl_pips = pips["sl"]
        else:
            tp_pips = s.MANUAL_TP_PIPS
            sl_pips = s.MANUAL_SL_PIPS

        # ── Индикатори (M5) ──────────────────────────────────
        rsi = ind.rsi(close, s.RSI_PERIOD)
        macd_line, macd_sig, macd_hist = ind.macd(close, s.MACD_FAST, s.MACD_SLOW, s.MACD_SIGNAL)
        ma_fast = ind.ema(close, s.MA_FAST)
        ma_slow = ind.ema(close, s.MA_SLOW)
        ma_trend = ind.ema(close, s.MA_TREND)

        # ── M15 тренд ────────────────────────────────────────
        m15_ema_fast = ind.ema(df_higher["Close"], s.MA_FAST)
        m15_ema_slow = ind.ema(df_higher["Close"], s.MA_SLOW)
        m15_bull = m15_ema_fast.iloc[-1] > m15_ema_slow.iloc[-1]
        m15_bear = m15_ema_fast.iloc[-1] < m15_ema_slow.iloc[-1]

        # ── Order Blocks ─────────────────────────────────────
        order_blocks = ind.find_order_blocks(df_primary, s.OB_LOOKBACK, s.OB_MIN_SIZE_ATR)

        # ── Candle patterns (M1 + M5) ────────────────────────
        patterns_m1 = ind.find_candle_patterns(df_entry)
        patterns_m5 = ind.find_candle_patterns(df_primary)
        all_patterns = patterns_m5 + patterns_m1

        rsi_val = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2]) if len(rsi) > 1 else rsi_val
        macd_h = float(macd_hist.iloc[-1])
        macd_h_prev = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0
        ma_f = float(ma_fast.iloc[-1])
        ma_s = float(ma_slow.iloc[-1])
        ma_t = float(ma_trend.iloc[-1])

        bull = 0
        bear = 0
        reasons = []

        # ── 1. RSI (0-20) ─────────────────────────────────────
        if rsi_val < s.RSI_OVERSOLD:
            bull += 20
            reasons.append(f"RSI oversold {rsi_val:.0f}")
        elif rsi_val > s.RSI_OVERBOUGHT:
            bear += 20
            reasons.append(f"RSI overbought {rsi_val:.0f}")
        elif rsi_val < 45:
            bull += 8
            bear += 3
        elif rsi_val > 55:
            bear += 8
            bull += 3
        else:
            bull += 5
            bear += 5
        # Momentum cross 50
        if rsi_prev < 50 <= rsi_val:
            bull += 5
            reasons.append("RSI cross 50 up")
        elif rsi_prev > 50 >= rsi_val:
            bear += 5
            reasons.append("RSI cross 50 down")

        # ── 2. MACD (0-20) ────────────────────────────────────
        cross_bull = macd_h_prev < 0 < macd_h
        cross_bear = macd_h_prev > 0 > macd_h
        if cross_bull:
            bull += 20
            reasons.append("MACD cross up")
        elif cross_bear:
            bear += 20
            reasons.append("MACD cross down")
        elif macd_h > 0 and macd_h > macd_h_prev:
            bull += 12
            bear += 2
        elif macd_h < 0 and macd_h < macd_h_prev:
            bear += 12
            bull += 2
        elif macd_h > 0:
            bull += 7
        elif macd_h < 0:
            bear += 7
        else:
            bull += 4
            bear += 4

        # ── 3. Moving Averages (0-15) ─────────────────────────
        ma_bull = ma_bear = 0
        if current_price > ma_t:
            ma_bull += 4
        else:
            ma_bear += 4
        if ma_f > ma_s:
            ma_bull += 4
        else:
            ma_bear += 4
        if current_price > ma_f > ma_s:
            ma_bull += 7
            reasons.append("Above MA stack")
        elif current_price < ma_f < ma_s:
            ma_bear += 7
            reasons.append("Below MA stack")
        bull += ma_bull
        bear += ma_bear

        # ── 4. Candle Patterns (0-20) ─────────────────────────
        cp_bull = cp_bear = 0
        best_bull_cp = best_bear_cp = None
        for cp in all_patterns:
            if cp.direction == "BULLISH":
                score = int(20 * cp.strength)
                if score > cp_bull:
                    cp_bull = score
                    best_bull_cp = cp
            elif cp.direction == "BEARISH":
                score = int(20 * cp.strength)
                if score > cp_bear:
                    cp_bear = score
                    best_bear_cp = cp
        if best_bull_cp:
            reasons.append(best_bull_cp.name)
        if best_bear_cp:
            reasons.append(best_bear_cp.name)
        bull += cp_bull
        bear += cp_bear

        # ── 5. Order Blocks (0-15) ────────────────────────────
        ob_bull = ob_bear = 0
        nearest_ob = None
        atr_val = float(ind.atr(df_primary, s.ATR_PERIOD).iloc[-1])
        for ob in order_blocks:
            dist = abs(current_price - (ob.high + ob.low) / 2)
            if dist < atr_val * 1.0 and not ob.broken:
                if ob.ob_type == "BULLISH" and ob.low <= current_price <= ob.high * 1.005:
                    ob_bull = int(15 * ob.strength)
                    nearest_ob = ob
                    reasons.append(f"Bullish OB")
                    break
                elif ob.ob_type == "BEARISH" and ob.low * 0.995 <= current_price <= ob.high:
                    ob_bear = int(15 * ob.strength)
                    nearest_ob = ob
                    reasons.append(f"Bearish OB")
                    break
        bull += ob_bull
        bear += ob_bear

        # ── 6. M15 Тренд потвърждение (0-10) ─────────────────
        if m15_bull:
            bull += 10
        elif m15_bear:
            bear += 10

        # ── 7. ML корекция (±8) ───────────────────────────────
        if ml_prediction is not None:
            if ml_prediction > 0.62:
                adj = int(8 * (ml_prediction - 0.5) * 2)
                bull += adj
                reasons.append(f"ML bull {ml_prediction:.0%}")
            elif ml_prediction < 0.38:
                adj = int(8 * (0.5 - ml_prediction) * 2)
                bear += adj
                reasons.append(f"ML bear {1-ml_prediction:.0%}")

        # ── Нормализиране ─────────────────────────────────────
        max_possible = 100
        bull_score = min(100, (bull / max_possible) * 100)
        bear_score = min(100, (bear / max_possible) * 100)

        # ── Решение ───────────────────────────────────────────
        if bull_score >= s.MIN_SIGNAL_SCORE and bull_score > bear_score + 8:
            direction = "BUY"
            final_score = bull_score
        elif bear_score >= s.MIN_SIGNAL_SCORE and bear_score > bull_score + 8:
            direction = "SELL"
            final_score = bear_score
        else:
            direction = "NEUTRAL"
            final_score = max(bull_score, bear_score)

        # ── SL/TP в пипове → price ────────────────────────────
        if direction == "BUY":
            sl = round(current_price - sl_pips * pip, 5)
            tp = round(current_price + tp_pips * pip, 5)
        elif direction == "SELL":
            sl = round(current_price + sl_pips * pip, 5)
            tp = round(current_price - tp_pips * pip, 5)
        else:
            sl = round(current_price - sl_pips * pip, 5)
            tp = round(current_price + tp_pips * pip, 5)

        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        reasoning = " | ".join(reasons[:4]) if reasons else "Weak signal"

        logger.info(
            f"{symbol} {direction} | Score:{final_score:.0f} "
            f"(B:{bull_score:.0f}/S:{bear_score:.0f}) | "
            f"TP:{tp_pips}pip SL:{sl_pips}pip | {reasoning}"
        )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            score=final_score,
            confidence=min(1.0, final_score / 100),
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            risk_reward=rr,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rsi_score=bull if direction != "SELL" else bear,
            macd_score=0,
            ma_score=ma_bull if direction != "SELL" else ma_bear,
            ob_score=ob_bull if direction != "SELL" else ob_bear,
            candle_score=cp_bull if direction != "SELL" else cp_bear,
            ml_score=0,
            trendline_score=0,
            active_ob=nearest_ob,
            candle_patterns=all_patterns,
            reasoning=reasoning
        )
