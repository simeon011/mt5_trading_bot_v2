"""
config/settings.py — SCALPER MODE
Adaptive pip targets спрямо баланса на акаунта
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:

    # ── MT5 Връзка ──────────────────────────────────────────
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""
    MT5_PATH: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    # ── Символи ─────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "XAUUSD+",  # Gold
        "EURUSD+",  # Forex
        "GBPUSD+",  # Forex
        "USDJPY+",  # Forex
        "SP500",  # S&P 500 Index
        "JPN225ft",  # JPN index
        "NAS100",  # NASDAQ Index
    ])

    # ── Таймфреймове (SCALPER: M1 + M5 + M15) ───────────────
    PRIMARY_TF: str = "M5"       # Основен сигнал
    HIGHER_TF: str = "M15"      # Тренд потвърждение
    ENTRY_TF: str = "M1"        # Прецизен вход

    # ── Adaptive Pip Target ──────────────────────────────────
    # Автоматично се изчислява в scalper_engine.py
    # Логика:
    #   баланс < $500   → 5 пипа TP,  3 пипа SL
    #   баланс $500-2K  → 8 пипа TP,  5 пипа SL
    #   баланс $2K-10K  → 12 пипа TP, 7 пипа SL
    #   баланс > $10K   → 20 пипа TP, 10 пипа SL
    USE_ADAPTIVE_PIPS: bool = True
    # Ако искаш ръчно — постави USE_ADAPTIVE_PIPS = False
    MANUAL_TP_PIPS: float = 10.0
    MANUAL_SL_PIPS: float = 6.0

    # ── Риск Management ──────────────────────────────────────
    RISK_PERCENT: float = 0.5        # 0.5% на сделка (скалпинг = по-малко)
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_OPEN_TRADES: int = 10        # Scalper отваря повече едновременно
    MAX_TRADES_PER_SYMBOL: int = 2   # До 2 позиции на символ
    MAX_TRADES_PER_HOUR: int = 20    # Hard limit за час

    # ── Scalper Signal Settings ──────────────────────────────
    MIN_SIGNAL_SCORE: float = 52.0   # По-нисък праг = повече сигнали
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
    Връща TP и SL в пипове спрямо баланса.
    Колкото повече пари, толкова по-широки цели.
    """
    if balance < 100:
        return {"tp": 3, "sl": 2, "label": "micro (<$100)"}
    elif balance < 500:
        return {"tp": 5, "sl": 3, "label": "small (<$500)"}
    elif balance < 2000:
        return {"tp": 8, "sl": 5, "label": "medium (<$2K)"}
    elif balance < 10000:
        return {"tp": 12, "sl": 7, "label": "standard (<$10K)"}
    else:
        return {"tp": 20, "sl": 10, "label": "large (>$10K)"}
