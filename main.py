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
