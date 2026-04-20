"""
core/risk_manager.py — Управление на риска
Защита от прекомерни загуби, position sizing, trailing stop
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("RiskManager")


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    last_reset: date = field(default_factory=date.today)


class RiskManager:
    """Контролира риска на ниво портфолио."""

    def __init__(self, settings):
        self.s = settings
        self.state = RiskState()
        self.open_positions: Dict = {}

    def reset_daily_if_needed(self):
        today = date.today()
        if self.state.last_reset < today:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_losses = 0
            self.state.trading_halted = False
            self.state.halt_reason = ""
            self.state.last_reset = today
            logger.info("🔄 Дневен риск reset.")

    def can_trade(self, symbol: str, account_info: Dict) -> tuple:
        """
        Проверява дали е позволено да се отвори нова сделка.
        Връща (allowed: bool, reason: str)
        """
        self.reset_daily_if_needed()

        if self.state.trading_halted:
            return False, f"Трейдинг спрян: {self.state.halt_reason}"

        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)

        # Проверка за дневна загуба
        daily_loss_pct = (self.state.daily_pnl / balance * 100) if balance > 0 else 0
        if daily_loss_pct <= -self.s.MAX_DAILY_LOSS_PERCENT:
            self.state.trading_halted = True
            self.state.halt_reason = f"Дневна загуба {daily_loss_pct:.1f}% достигна лимит"
            logger.warning(f"🛑 {self.state.halt_reason}")
            return False, self.state.halt_reason

        # Проверка за drawdown
        drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        if drawdown > 5.0:
            return False, f"Drawdown {drawdown:.1f}% е твърде висок"

        # Брой отворени позиции
        total_open = len(self.open_positions)
        if total_open >= self.s.MAX_OPEN_TRADES:
            return False, f"Максимум {self.s.MAX_OPEN_TRADES} позиции достигнат"

        # Проверка за символ
        symbol_positions = sum(1 for p in self.open_positions.values()
                               if p.get("symbol") == symbol)
        if symbol_positions >= self.s.MAX_TRADES_PER_SYMBOL:
            return False, f"Вече има позиция за {symbol}"

        return True, "OK"

    def register_open(self, ticket: int, trade_info: Dict):
        self.open_positions[ticket] = {**trade_info, "open_time": datetime.now()}
        self.state.daily_trades += 1
        logger.info(f"📌 Позиция регистрирана: {ticket} | "
                    f"Общо отворени: {len(self.open_positions)}")

    def register_close(self, ticket: int, profit: float):
        if ticket in self.open_positions:
            del self.open_positions[ticket]
        self.state.daily_pnl += profit
        if profit < 0:
            self.state.daily_losses += 1
        logger.info(f"📤 Позиция затворена: {ticket} | P&L: {profit:+.2f} | "
                    f"Дневен P&L: {self.state.daily_pnl:+.2f}")

    def calculate_trailing_stop(self, position: Dict, current_price: float,
                                  atr: float) -> Optional[float]:
        """Изчислява нов trailing stop ако е нужен."""
        if not self.s.USE_TRAILING_STOP:
            return None

        current_sl = position.get("sl", 0)
        direction = position.get("type", "BUY")
        trail_dist = atr * self.s.TRAILING_STOP_ATR_MULT

        if direction == "BUY":
            new_sl = current_price - trail_dist
            if new_sl > current_sl + atr * 0.1:  # Движи само ако е значително
                return round(new_sl, 5)
        else:
            new_sl = current_price + trail_dist
            if new_sl < current_sl - atr * 0.1:
                return round(new_sl, 5)

        return None

    def get_daily_summary(self) -> Dict:
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_losses": self.state.daily_losses,
            "open_positions": len(self.open_positions),
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason
        }
