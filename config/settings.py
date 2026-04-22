"""
config/settings.py — SCALPER MODE (FIXED)
✅ Добавени: Trailing Stop параметри
Adaptive pip targets спрямо баланса на акаунта
"""
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:

    # ── MT5 Връзка ──────────────────────────────────────────
    MT5_LOGIN: int = 25080519
    MT5_PASSWORD: str = "!g&H2rUT"
    MT5_SERVER: str = "VantageInternational-Demo"
    MT5_PATH: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    # ── Символи ─────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        # "XAUUSD+",  # Gold
        "EURUSD+",  # Forex
        "GBPUSD+",  # Forex
        "USDJPY+",  # Forex
        # "SP500",  # S&P 500 Index
        # "JPN225ft",  # JPN index
        # "NAS100",  # NASDAQ Index
    ])

    # ── Таймфреймове (SCALPER: M1 + M5 + M15) ───────────────
    PRIMARY_TF: str = "M5"       # Основен сигнал
    HIGHER_TF: str = "M15"      # Тренд потвърждение
    ENTRY_TF: str = "M1"        # Прецизен вход

    # ── Adaptive Pip Target ──────────────────────────────────
    # Автоматично се изчислява в signal_engine.py
    USE_ADAPTIVE_PIPS: bool = True
    # Ако искаш ръчно — постави USE_ADAPTIVE_PIPS = False
    MANUAL_TP_PIPS: float = 10.0
    MANUAL_SL_PIPS: float = 6.0

    # ── Риск Management ──────────────────────────────────────
    RISK_PERCENT: float = 0.5        # 0.5% на сделка (скалпинг = по-малко)
    MAX_DAILY_LOSS_PERCENT: float = 4.0
    MAX_OPEN_TRADES: int = 10        # Scalper отваря повече едновременно
    MAX_TRADES_PER_SYMBOL: int = 2   # До 2 позиции на символ
    MAX_TRADES_PER_HOUR: int = 30    # Hard limit за час

    # ── Scalper Signal Settings ──────────────────────────────
    MIN_SIGNAL_SCORE: float = 60.0   # По-нисък праг = повече сигнали
    COOLDOWN_CYCLES: int = 0         # Без cooldown между сделки
    MIN_RR_RATIO: float = 1.5        # Минимален R:R

    # ── Индикатори (оптимизирани за M5) ─────────────────────
    RSI_PERIOD: int = 7              # По-бърз RSI за скалпинг
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    MACD_FAST: int = 5
    MACD_SLOW: int = 13
    MACD_SIGNAL: int = 3
    MA_FAST: int = 8
    MA_SLOW: int = 21
    MA_TREND: int = 50               # По-кратък тренд за M5
    ATR_PERIOD: int = 7

    # ── ATR Volatility Filter ────────────────────────────────
    # Ботът НЕ трейдва ако ATR е извън тези граници
    ATR_FILTER_ENABLED: bool = True
    ATR_MIN_MULTIPLIER: float = 0.5  # Пазарът е твърде тих (< 0.5x средния ATR)
    ATR_MAX_MULTIPLIER: float = 3.0  # Пазарът е твърде луд  (> 3x средния ATR)
    ATR_AVERAGE_PERIOD: int = 20  # Средно ATR за последните 20 свещи

    # ── Trailing Stop (НОВО) ─────────────────────────────────
    # ✅ Когато достигнем 40% от TP/SL → местим SL на entry
    USE_TRAILING_STOP: bool = True          # Включен trailing stop
    TRAILING_STOP_ATR_MULT: float = 1.5     # Разстояние = ATR * 1.5
    TRAILING_STOP_PROFIT_THRESHOLD: float = 0.4  # 40% прогрес към TP

    # ── Order Blocks ─────────────────────────────────────────
    OB_LOOKBACK: int = 20
    OB_MIN_SIZE_ATR: float = 0.3

    # ── ML ───────────────────────────────────────────────────
    ML_ENABLED: bool = True
    ML_MIN_TRADES_TO_TRAIN: int = 50
    ML_RETRAIN_INTERVAL: int = 50
    ML_FEATURE_LOOKBACK: int = 10

    # ── Performance ──────────────────────────────────────────
    CYCLE_INTERVAL_SECONDS: int = 15  # Проверява на всеки 15 сек!
    CANDLE_HISTORY: int = 200

    # ── Telegram (опционално) ───────────────────────────────
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ── Пътища ──────────────────────────────────────────────
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"


def get_adaptive_pips(balance: float) -> dict:
    """
    Автоматично изчислява TP и SL спрямо баланса.
    Използва логаритмична прогресия, за да не растат целите прекалено бързо.
    """
    # Базови стойности за много малък акаунт ($10-$50)
    base_tp = 15
    base_sl = 10

    # Коефициент на растеж: на всеки дублиран баланс добавяме малко към целите
    # Използваме log2, за да имаме плавна крива
    growth = math.log2(max(balance, 10) / 10)

    # Изчисляване на TP и SL
    # Пример: при $50 TP ще е ~20, при $500 TP ще е ~30
    calculated_tp = int(base_tp + (growth * 5))
    calculated_sl = int(base_sl + (growth * 2.5))

    # Ограничители (Caps), за да не станат прекалено огромни или малки
    tp = max(15, min(calculated_tp, 50))
    sl = max(10, min(calculated_sl, 25))

    return {
        "tp": tp,
        "sl": sl,
        "label": f"auto (dist: {tp}/{sl})"
    }