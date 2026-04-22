"""
╔══════════════════════════════════════════════════════╗
║         MT5 AI TRADING BOT — ГЛАВЕН МОДУЛ            ║
║   Gold · Forex · Indices | Self-Learning Agent       ║
╚══════════════════════════════════════════════════════╝

Стартиране:
    pip install -r requirements.txt
    python main.py --mode paper     # Тест без реални пари
    python main.py --mode live      # Реален трейдинг (внимание!)
    python main.py --mode backtest  # Тестване на исторически данни
"""

import argparse
import logging
import time
import sys
import os
from datetime import datetime
from core.bot import TradingBot
from config.settings import Settings

CYCLE_LEVEL = 21
STUDY_LEVEL = 22
MODEL_LEVEL = 23

# 2. Казваме на Python какъв текст да изписва в скобите [ ]
logging.addLevelName(CYCLE_LEVEL, "CYCLE")
logging.addLevelName(STUDY_LEVEL, "SELF-STUDYING")
logging.addLevelName(MODEL_LEVEL, "CREATE NEW MODEL")

# 3. Създаваме функциите, за да можеш да ги викаш лесно с logger.neshto()
def log_cycle(self, message, *args, **kws):
    if self.isEnabledFor(CYCLE_LEVEL):
        self._log(CYCLE_LEVEL, message, args, **kws)

def log_self_study(self, message, *args, **kws):
    if self.isEnabledFor(STUDY_LEVEL):
        self._log(STUDY_LEVEL, message, args, **kws)

def log_new_model(self, message, *args, **kws):
    if self.isEnabledFor(MODEL_LEVEL):
        self._log(MODEL_LEVEL, message, args, **kws)

# 4. Прикачваме ги към главния Logger клас
logging.Logger.cycle = log_cycle
logging.Logger.self_study = log_self_study
logging.Logger.new_model = log_new_model

# Fix Windows Unicode (Bulgarian + emoji)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        ),
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="MT5 AI Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default="paper")
    parser.add_argument("--symbol", default=None, help="Конкретен символ (напр. EURUSD)")
    parser.add_argument("--backtest-from", default="2024-01-01")
    parser.add_argument("--backtest-to", default="2024-12-31")
    args = parser.parse_args()

    settings = Settings()
    if args.symbol:
        settings.SYMBOLS = [args.symbol]

    logger.info(f"🚀 Стартиране в режим: {args.mode.upper()}")
    logger.info(f"📊 Символи: {settings.SYMBOLS}")

    bot = TradingBot(settings, mode=args.mode)

    if args.mode == "backtest":
        bot.run_backtest(args.backtest_from, args.backtest_to)
    else:
        logger.info("⏳ Бот е активен. Ctrl+C за спиране.")
        try:
            while True:
                bot.run_cycle()
                time.sleep(settings.CYCLE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("🛑 Бот спрян от потребителя.")
            bot.shutdown()



if __name__ == "__main__":
    main()
