"""
config/settings.py — Всички настройки на бота
Редактирай тук преди стартиране!
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # ── MT5 Връзка ──────────────────────────────────────────
    MT5_LOGIN: int = 0              # Твоят MT5 акаунт номер
    MT5_PASSWORD: str = ""          # Парола
    MT5_SERVER: str = ""            # напр. "ICMarkets-Demo"
    MT5_PATH: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    # ── Символи за трейдване ────────────────────────────────
    # Vantage International символи:
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "XAUUSD+",  # Gold
        "EURUSD+",  # Forex
        "GBPUSD+",  # Forex
        "USDJPY+",  # Forex
        "SP500",  # S&P 500 Index
        "JPN225ft",  # JPN index
        "NAS100",  # NASDAQ Index
    ])

    # ЗАБЕЛЕЖКА: Провери точните имена в MT5:
    # View -> Market Watch -> десен клик -> Symbols -> търси символа
    # Vantage често добавя 'm' или '.' след символа

    # ── Таймфреймове ────────────────────────────────────────
    # Бота анализира няколко таймфрейма едновременно (Multi-TF)
    PRIMARY_TF: str = "H1"          # Основен таймфрейм за сигнали
    HIGHER_TF: str = "H4"           # За тренд потвърждение
    ENTRY_TF: str = "M15"           # За прецизен вход

    # ── Управление на риска ─────────────────────────────────
    RISK_PERCENT: float = 1.0       # % от баланса на сделка (MAX 2%)
    MAX_DAILY_LOSS_PERCENT: float = 3.0   # Спри бота при 3% дневна загуба
    MAX_OPEN_TRADES: int = 3        # Максимум едновременни позиции
    MAX_TRADES_PER_SYMBOL: int = 1  # 1 позиция на символ

    # ── Stop Loss / Take Profit ─────────────────────────────
    DEFAULT_SL_ATR_MULT: float = 1.5   # SL = 1.5x ATR
    DEFAULT_TP_RR_RATIO: float = 2.0   # TP = 2:1 Risk/Reward
    USE_TRAILING_STOP: bool = True
    TRAILING_STOP_ATR_MULT: float = 1.0

    # ── Стратегия & Сигнали ─────────────────────────────────
    MIN_SIGNAL_SCORE: float = 65.0     # Минимален скор за влизане (0-100)
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    MA_FAST: int = 21
    MA_SLOW: int = 50
    MA_TREND: int = 200
    ATR_PERIOD: int = 14

    # ── Order Blocks ────────────────────────────────────────
    OB_LOOKBACK: int = 50           # Свещи назад за намиране на OB
    OB_MIN_SIZE_ATR: float = 0.5    # Мин. размер на OB в ATR единици

    # ── ML Модел ────────────────────────────────────────────
    ML_ENABLED: bool = True
    ML_MIN_TRADES_TO_TRAIN: int = 30   # Мин. затворени сделки за обучение
    ML_RETRAIN_INTERVAL: int = 100     # Преобучаване на всеки N затворени сделки
    ML_FEATURE_LOOKBACK: int = 20      # Брой свещи за ML features

    # ── Производителност ────────────────────────────────────
    CYCLE_INTERVAL_SECONDS: int = 60   # Проверка на всяка минута
    CANDLE_HISTORY: int = 500          # Брой свещи история

    # ── Известия (опционално) ───────────────────────────────
    TELEGRAM_TOKEN: str = ""           # Bot token за Telegram известия
    TELEGRAM_CHAT_ID: str = ""         # Твоят chat ID

    # ── Пътища ──────────────────────────────────────────────
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"
