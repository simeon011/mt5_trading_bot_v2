"""
strategies/signal_engine.py — SCALPER VERSION (FIXED)
✅ Фиксирани: JPN225ft pip размер, валидация на SL/TP разстояние
✅ Динамични ATR Цели & Снайпер Филтър (Преместен след индикаторите)
✅ Визуално Табло Логване (STYLE 2)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging

from strategies.indicators import TechnicalIndicators, OrderBlock, Trendline, CandlePattern

logger = logging.getLogger("ScalperEngine")

# ═══════════════════════════════════════════════════════════════
# ✅ ФИКСИРАНИ PIP РАЗМЕРИ
# ═══════════════════════════════════════════════════════════════
PIP_SIZE = {
    # Forex
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
    "USDCHF": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
    "USDCAD": 0.0001, "EURGBP": 0.0001, "EURJPY": 0.01,
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
    if symbol.upper() in PIP_SIZE:
        return PIP_SIZE[symbol.upper()]

    for key, val in PIP_SIZE.items():
        if symbol.upper().startswith(key.upper()):
            return val

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

                if atr_ratio > s.ATR_MAX_MULTIPLIER:
                    logger.warning(f"⚠️ {symbol:8} HIGH VOLATILITY | Ratio: {atr_ratio:.2f}x | Trading Paused")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0, confidence=0,
                        entry_price=current_price, stop_loss=current_price, take_profit=current_price, risk_reward=0
                    )

                if atr_ratio < s.ATR_MIN_MULTIPLIER:
                    logger.info(f"💤 {symbol:8} LOW VOLATILITY  | Ratio: {atr_ratio:.2f}x | Waiting for movement")
                    return TradeSignal(
                        symbol=symbol, direction="NEUTRAL", score=0, confidence=0,
                        entry_price=current_price, stop_loss=current_price, take_profit=current_price, risk_reward=0
                    )

        # ── Динамични ATR Цели (Снайперист) ──────────────────
        atr_series = ind.atr(df_primary, s.ATR_PERIOD)
        current_atr = float(atr_series.iloc[-1])
        atr_pips = current_atr / pip

        tp_multiplier = 1.5
        sl_multiplier = 1.0

        tp_pips = atr_pips * tp_multiplier
        sl_pips = atr_pips * sl_multiplier

        # ─────────────────────────────────────────────────────
        # Валидация на SL/TP разстояние за различните символи
        # ─────────────────────────────────────────────────────
        min_sl_distance = 3  # Default за forex
        if symbol.upper().startswith("JPN"):
            min_sl_distance = 5
        elif symbol.upper().startswith("US") or symbol.upper().startswith("NAS") or "500" in symbol.upper():
            min_sl_distance = 3

        if sl_pips < min_sl_distance:
            sl_pips = min_sl_distance
        if tp_pips < min_sl_distance * 1.5:
            tp_pips = min_sl_distance * 2

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

        reason_str = "/".join(list(dict.fromkeys(reasons))[:3])
        if not reason_str: reason_str = "ML-Driven"

        # ── 🎯 СНАЙПЕР ФИЛТЪР (След като имаме процентите) ────
        if tp_pips < s.MIN_TP_PIPS:
            logger.info(
                f"💤 {symbol:8} | СКИП | 🐂 {b_pct:>2.0f}% / 🐻 {s_pct:>2.0f}% | Мощност: {final_score:>2.0f} | "
                f"⚠️ Отказ: TP ({tp_pips:.1f}p) е под {s.MIN_TP_PIPS}"
            )
            return TradeSignal(
                symbol=symbol, direction="NEUTRAL", score=0, confidence=0, entry_price=current_price,
                stop_loss=current_price, take_profit=current_price, risk_reward=0, reasoning="Sniper Filter"
            )

        # ── 📊 ВИЗУАЛНО ТАБЛО (За приетите сигнали) ───────────
        if direction != "WAIT":
            logger.info(
                f"{emoji} {symbol:8} | {direction:4} | 🐂 {b_pct:>2.0f}% / 🐻 {s_pct:>2.0f}% | Мощност: {final_score:>2.0f} | "
                f"🎯 SL: {sl_pips:.0f}p / TP: {tp_pips:.0f}p | {reason_str}"
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