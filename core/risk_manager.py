"""
core/risk_manager.py — Управление на риска (FIXED)
✅ Добавени: USE_TRAILING_STOP, TRAILING_STOP_ATR_MULT параметри
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
    start_balance: float = 0.0  # Балансът в началото на деня


class RiskManager:
    """Контролира риска на ниво портфолио."""

    def __init__(self, settings):
        self.s = settings
        self.state = RiskState()
        self.open_positions: Dict = {}

        # ✅ НОВО: Параметри за Trailing Stop (ако не са дефинирани в settings)
        self._init_trailing_stop_settings()

    def _init_trailing_stop_settings(self):
        """Инициализира Trailing Stop параметрите ако не съществуват."""
        if not hasattr(self.s, 'USE_TRAILING_STOP'):
            self.s.USE_TRAILING_STOP = True
            logger.info("✅ Добавен USE_TRAILING_STOP = True (по подразбиране)")

        if not hasattr(self.s, 'TRAILING_STOP_ATR_MULT'):
            self.s.TRAILING_STOP_ATR_MULT = 1.5
            logger.info("✅ Добавен TRAILING_STOP_ATR_MULT = 1.5 (по подразбиране)")

        if not hasattr(self.s, 'TRAILING_STOP_PROFIT_THRESHOLD'):
            self.s.TRAILING_STOP_PROFIT_THRESHOLD = 0.4  # 40% от TP
            logger.info("✅ Добавен TRAILING_STOP_PROFIT_THRESHOLD = 0.4 (по подразбиране)")

    def reset_daily_if_needed(self, current_balance: float):
        """Рестартира броячите и записва началния баланс за деня."""
        today = date.today()
        if self.state.last_reset < today or self.state.start_balance == 0:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_losses = 0
            self.state.trading_halted = False
            self.state.halt_reason = ""
            self.state.last_reset = today
            self.state.start_balance = current_balance
            logger.info(f"🔄 Дневен риск reset. Начален баланс: {current_balance:.2f} USD")

    def can_trade(self, symbol: str, account_info: Dict) -> tuple:
        """Проверява дали е позволено да се отвори нова сделка."""
        balance = account_info.get("balance", 0)
        self.reset_daily_if_needed(balance)

        if self.state.trading_halted:
            return False, f"Трейдинг спрян: {self.state.halt_reason}"

        # 1. Изчисляване на % загуба спрямо началния баланс
        if self.state.daily_pnl < 0:
            loss_pct = (abs(self.state.daily_pnl) / self.state.start_balance * 100) if self.state.start_balance > 0 else 0
            if loss_pct >= self.s.MAX_DAILY_LOSS_PERCENT:
                self.state.trading_halted = True
                self.state.halt_reason = f"Дневна загуба {loss_pct:.1f}% достигна лимит"
                logger.warning(f"🛑 {self.state.halt_reason}")
                return False, self.state.halt_reason

        # Проверка за текущ drawdown на отворените позиции
        equity = account_info.get("equity", 0)
        drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        if drawdown > 5.0:
            return False, f"Drawdown {drawdown:.1f}% е твърде висок"

        # Брой отворени позиции
        if len(self.open_positions) >= self.s.MAX_OPEN_TRADES:
            return False, f"Максимум {self.s.MAX_OPEN_TRADES} позиции достигнат"

        # Проверка за символ
        symbol_positions = sum(1 for p in self.open_positions.values() if p.get("symbol") == symbol)
        if symbol_positions >= self.s.MAX_TRADES_PER_SYMBOL:
            return False, f"Вече има позиция за {symbol}"

        return True, "OK"

    def register_open(self, ticket: int, trade_info: Dict):
        self.open_positions[ticket] = {**trade_info, "open_time": datetime.now()}
        self.state.daily_trades += 1
        logger.info(f"📌 Позиция регистрирана: {ticket} | Общо отворени: {len(self.open_positions)}")

    def register_close(self, ticket: int, profit: float):
        """
        profit: Трябва да бъде реалната сума в долари (profit + commission + swap)
        """
        if ticket in self.open_positions:
            del self.open_positions[ticket]

        self.state.daily_pnl += profit

        if profit < 0:
            self.state.daily_losses += 1

        # Изчисляваме текущия % загуба за лога
        current_loss_pct = (abs(self.state.daily_pnl) / self.state.start_balance * 100) if (self.state.daily_pnl < 0 and self.state.start_balance > 0) else 0

        logger.info(f"📤 Позиция затворена: {ticket} | P&L: {profit:+.2f}$ | "
                    f"Дневен P&L: {self.state.daily_pnl:+.2f}$ ({current_loss_pct:.1f}%)")

    def calculate_trailing_stop(self, position: Dict, current_price: float, atr: float) -> Optional[float]:
        """
        Изчислява новия Trailing Stop.

        Логика:
        - Трябва да сме в печалба преди да се движи SL
        - Движим SL нагоре/надолу в посока на профита с ATR*множител
        - Но с минимален праг да се движи
        """
        if not hasattr(self.s, 'USE_TRAILING_STOP') or not self.s.USE_TRAILING_STOP:
            return None

        current_sl = position.get("sl", 0)
        direction = position.get("type", "BUY")  # BUY или SELL
        entry_price = position.get("entry", position.get("price_open", current_price))

        # Трябва ни по-широк стоп за малки акаунти
        trail_dist = atr * self.s.TRAILING_STOP_ATR_MULT

        if direction == "BUY":
            # За BUY: SL се движи нагоре (за защита на печалба)
            new_sl = current_price - trail_dist
            # Само ако новия SL е по-висок от текущия И по-висок от entry
            if new_sl > current_sl and new_sl > entry_price:
                return round(new_sl, 5)
        else:  # SELL
            # За SELL: SL се движи НАДОЛУ (намалява числово) за защита на печалба
            # new_sl = current_price + trail_dist (SL е над текущата цена)
            # Печалба = когато current_price < entry_price
            # Trailing влиза само ако new_sl < entry_price (в зона на печалба)
            new_sl = current_price + trail_dist
            if (current_sl == 0 or new_sl < current_sl) and new_sl < entry_price:
                return round(new_sl, 5)

        return None

    def get_daily_summary(self) -> Dict:
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_losses": self.state.daily_losses,
            "open_positions": len(self.open_positions),
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "start_balance": self.state.start_balance
        }