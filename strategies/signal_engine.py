"""
strategies/signal_engine.py — SCALPER VERSION (FIXED)
✅ Фиксирани: JPN225ft pip размер, валидация на SL/TP разстояние
Minimalist Design | Bull/Bear Power Meter | Professional & Clean Logs
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging

from strategies.indicators import TechnicalIndicators, OrderBlock, Trendline, CandlePattern
from config.settings import get_adaptive_pips

logger = logging.getLogger("ScalperEngine")

# ═══════════════════════════════════════════════════════════════
# ✅ ФИКСИРАНИ PIP РАЗМЕРИ (Проблем 4)
# Обновени правилни стойности за всички символи
# ═══════════════════════════════════════════════════════════════
PIP_SIZE = {
    # Forex
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
    "USDCHF": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001, # Добавен NZDUSD
    "USDCAD": 0.0001, "EURGBP": 0.0001, "EURJPY": 0.01,   # Добавени USDCAD, EURGBP, EURJPY
    # Metals & Indices
    "XAUUSD": 0.1,    "XAGUSD": 0.01,   "NAS100": 1.0,
    "US500":  0.1,    "SP500":  0.1,    "GER40":  1.0,
    "UK100":  1.0,    "US30":   1.0,    "JPN225": 1.0
}

def get_pip(symbol: str) -> float:
    """
    Връща pip стойността за символ.
    Default: 0.0001 за неизвестни символи
    """
    # Проверяваме точно съвпадение първо
    if symbol.upper() in PIP_SIZE:
        return PIP_SIZE[symbol.upper()]

    # След това проверяваме с префикс (за символи с суфикси)
    for key, val in PIP_SIZE.items():
        if symbol.upper().startswith(key.upper()):
            return val

    # Default за неизвестни символи
    logger.warning(f"⚠️  Неизвестен символ: {symbol}. Използвам default pip 0.0001")
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
    def __init__(self, settings):
        self.s = settings
        self.ind = TechnicalIndicators()
        self._account_balance: float = 1000.0

    def update_balance(self, balance: float):
        self._account_balance = balance

    def analyze(self, symbol: str, df_primary: pd.DataFrame,
                df_higher: pd.DataFrame, df_entry: pd.DataFrame,
                ml_prediction: Optional[float] = None) -> TradeSignal:

        ind = self.ind
        s = self.s
        close = df_primary["Close"]
        current_price = float(close.iloc[-1])
        pip = get_pip(symbol)

        # ── ATR Filter ───────────────────────────────────────
        if s.ATR_FILTER_ENABLED:
            atr_series = ind.atr(df_primary, s.ATR_PERIOD)
            current_atr = float(atr_series.iloc[-1])
            avg_atr = float(atr_series.tail(s.ATR_AVERAGE_PERIOD).mean())

            if avg_atr > 0:
                atr_ratio = current_atr / avg_atr

                # ПРЕДУПРЕЖДЕНИЕ ЗА ВИСОКА ВОЛАТИЛНОСТ (Пазарът е луд)
                if atr_ratio > s.ATR_MAX_MULTIPLIER:
                    logger.warning(f"⚠️ {symbol:8} HIGH VOLATILITY | Ratio: {atr_ratio:.2f}x | Trading Paused")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0,
                        confidence=0, entry_price=current_price,
                        stop_loss=current_price, take_profit=current_price,
                        risk_reward=0, reasoning=f"High Volatility ({atr_ratio:.2f}x)"
                    )

                # ИНФОРМАЦИЯ ЗА НИСКА ВОЛАТИЛНОСТ (Пазарът спи)
                if atr_ratio < s.ATR_MIN_MULTIPLIER:
                    logger.info(f"💤 {symbol:8} LOW VOLATILITY  | Ratio: {atr_ratio:.2f}x | Waiting for movement")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0,
                        confidence=0, entry_price=current_price,
                        stop_loss=current_price, take_profit=current_price,
                        risk_reward=0, reasoning=f"Low Volatility ({atr_ratio:.2f}x)"
                    )

        # ── Adaptive Pips ────────────────────────────────────
        pips = get_adaptive_pips(self._account_balance) if s.USE_ADAPTIVE_PIPS else {"tp": s.MANUAL_TP_PIPS, "sl": s.MANUAL_SL_PIPS}
        tp_pips, sl_pips = pips["tp"], pips["sl"]

        # ─────────────────────────────────────────────────────
        # ✅ НОВО: Валидация на SL/TP разстояние за различните символи
        # За някои символи (JPN225ft, indices) минималното разстояние е по-голямо
        # ─────────────────────────────────────────────────────

        # Определяме минимално разстояние спрямо символ
        min_sl_distance = 3  # Default за forex
        if symbol.upper().startswith("JPN"):
            min_sl_distance = 5  # JPN225 нужда поне 5 точки
        elif symbol.upper().startswith("US") or symbol.upper().startswith("NAS") or "500" in symbol.upper():
            min_sl_distance = 3   # Indices нужда поне 3 точки

        # Ако адаптивните пипа са твърде малки, увеличаваме ги
        if sl_pips < min_sl_distance:
            sl_pips = min_sl_distance
            logger.info(f"⚙️  {symbol}: Увеличен SL от {pips['sl']} на {sl_pips} (минимум за символ)")

        if tp_pips < min_sl_distance * 1.5:
            tp_pips = min_sl_distance * 2
            logger.info(f"⚙️  {symbol}: Увеличен TP от {pips['tp']} на {tp_pips}")

        # ── Индикатори ───────────────────────────────────────
        rsi_val = float(ind.rsi(close, s.RSI_PERIOD).iloc[-1])
        macd_line, macd_sig, macd_hist = ind.macd(close, s.MACD_FAST, s.MACD_SLOW, s.MACD_SIGNAL)
        macd_h = float(macd_hist.iloc[-1])
        ma_fast_val = float(ind.ema(close, s.MA_FAST).iloc[-1])
        ma_slow_val = float(ind.ema(close, s.MA_SLOW).iloc[-1])

        m15_ema_f = ind.ema(df_higher["Close"], s.MA_FAST).iloc[-1]
        m15_ema_s = ind.ema(df_higher["Close"], s.MA_SLOW).iloc[-1]

        bull = bear = 0
        reasons = []

        # ── Simple Scoring ────────────────────────────────────
        if rsi_val < s.RSI_OVERSOLD: bull += 25; reasons.append("RSI-OS")
        elif rsi_val > s.RSI_OVERBOUGHT: bear += 25; reasons.append("RSI-OB")

        if macd_h > 0: bull += 15
        else: bear += 15

        if current_price > ma_fast_val > ma_slow_val: bull += 20; reasons.append("Trend-UP")
        elif current_price < ma_fast_val < ma_slow_val: bear += 20; reasons.append("Trend-DOWN")

        if m15_ema_f > m15_ema_s: bull += 10
        else: bear += 10

        ml_val = ml_prediction if ml_prediction else 0.5
        if ml_val > 0.6: bull += 20
        elif ml_val < 0.4: bear += 20

        # ── Calculations ──────────────────────────────────────
        total_p = bull + bear
        b_pct = (bull / total_p * 100) if total_p > 0 else 50
        s_pct = (bear / total_p * 100) if total_p > 0 else 50

        if b_pct >= s.MIN_SIGNAL_SCORE and b_pct > s_pct + 10:
            direction, emoji = "BUY", "🟢"
            final_score = b_pct
        elif s_pct >= s.MIN_SIGNAL_SCORE and s_pct > b_pct + 10:
            direction, emoji = "SELL", "🔴"
            final_score = s_pct
        else:
            direction, emoji = "WAIT", "🟡"
            final_score = max(b_pct, s_pct)

        # ── Professional Log ─────────────────────────────────
        reason_str = "/".join(list(dict.fromkeys(reasons))[:3])

        logger.info(
            f"{emoji} {symbol:8} {direction:7} | "
            f"Bias: {b_pct:>3.0f}%/{s_pct:<3.0f}% | "
            f"Score: {final_score:>2.0f} | "
            f"ML: {ml_val:>4.0%} | "
            f"SL:{sl_pips:.0f}pip TP:{tp_pips:.0f}pip"
            f"{' | ' + reason_str if reason_str else ''}"
        )

        # ── TradeSignal Object ───────────────────────────────
        if direction == "BUY":
            sl, tp = current_price - sl_pips * pip, current_price + tp_pips * pip
        elif direction == "SELL":
            sl, tp = current_price + sl_pips * pip, current_price - tp_pips * pip
        else:
            sl = tp = current_price  # NEUTRAL

        return TradeSignal(
            symbol=symbol, direction=direction if direction != "WAIT" else "NEUTRAL",
            score=final_score, confidence=max(b_pct, s_pct)/100, entry_price=current_price,
            stop_loss=round(sl, 5), take_profit=round(tp, 5),
            risk_reward=tp_pips/sl_pips if sl_pips > 0 else 0,
            sl_pips=sl_pips, tp_pips=tp_pips,
            rsi_score=bull, ma_score=bull, reasoning=reason_str
        )