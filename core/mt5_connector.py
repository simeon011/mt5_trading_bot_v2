"""
core/mt5_connector.py — MT5 връзка и изпълнение на поръчки (ОПТИМИЗИРАН)
✅ Премахнат дублиращ се код
✅ Обединена логика за затваряне на позиции
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict

logger = logging.getLogger("MT5Connector")

# Таймфрейм mapping
TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
    "W1": 32769, "MN1": 49153
}


class MT5Connector:
    """Управлява всички операции с MetaTrader 5."""

    def __init__(self, settings):
        self.settings = settings
        self.connected = False
        self._mt5 = None
        self._import_mt5()

    def _import_mt5(self):
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            logger.info("✅ MetaTrader5 пакет намерен.")
        except ImportError:
            logger.warning("⚠️ MetaTrader5 не е инсталиран. Режим симулация.")
            self._mt5 = None

    def connect(self) -> bool:
        """Свързване с MT5 терминал."""
        if self._mt5 is None:
            logger.info("📊 Симулационен режим (без MT5).")
            self.connected = True
            return True

        mt5 = self._mt5
        kwargs = {}
        if self.settings.MT5_PATH:
            kwargs["path"] = self.settings.MT5_PATH

        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 initialize() неуспешно: {mt5.last_error()}")
            return False

        if self.settings.MT5_LOGIN:
            authorized = mt5.login(
                self.settings.MT5_LOGIN,
                password=self.settings.MT5_PASSWORD,
                server=self.settings.MT5_SERVER
            )
            if not authorized:
                logger.error(f"MT5 login неуспешен: {mt5.last_error()}")
                return False

        info = mt5.account_info()
        logger.info(f"✅ Свързан с MT5 | Акаунт: {info.login} | "
                    f"Баланс: {info.balance:.2f} {info.currency} | "
                    f"Сървър: {info.server}")
        self.connected = True
        return True

    def disconnect(self):
        if self._mt5:
            self._mt5.shutdown()
        self.connected = False
        logger.info("MT5 връзката затворена.")

    def get_candles(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Взима OHLCV данни като DataFrame."""
        if self._mt5 is None:
            return self._simulate_candles(symbol, count)

        tf_code = TF_MAP.get(timeframe, 16385)
        rates = self._mt5.copy_rates_from_pos(symbol, tf_code, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"Няма данни за {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                           "close": "Close", "tick_volume": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_account_info(self) -> Dict:
        if self._mt5 is None:
            return {"balance": 10000.0, "equity": 10000.0, "margin_free": 9500.0, "currency": "USD"}
        info = self._mt5.account_info()
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin_free": info.margin_free,
            "currency": info.currency
        }

    def get_open_positions(self) -> List[Dict]:
        """Взима отворени позиции от MT5."""
        if self._mt5 is None:
            return []
        positions = self._mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "comment": p.comment
            })
        return result

    def place_order(self, symbol: str, order_type: str, volume: float,
                    sl: float, tp: float, comment: str = "AI_BOT") -> Optional[Dict]:
        """Изпраща поръчка към MT5 с валидация."""
        if self._mt5 is None:
            logger.info(f"📋 [СИМУЛАЦИЯ] {order_type} {volume} {symbol} | SL:{sl:.5f} TP:{tp:.5f}")
            return {"ticket": np.random.randint(100000, 999999), "symbol": symbol,
                    "type": order_type, "volume": volume, "sl": sl, "tp": tp, "price_open": 0}

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Не може да се вземе tick за {symbol}")
            return None

        price = tick.ask if order_type == "BUY" else tick.bid
        order_type_code = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL

        # Валидация на SL и TP разстояние
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            min_distance = symbol_info.trade_stops_level or 0
            if min_distance > 0:
                if order_type == "BUY":
                    if sl >= price:
                        logger.error(f"❌ {symbol}: SL ({sl}) трябва да е по-нисък от entry ({price})")
                        return None
                    if tp <= price:
                        logger.error(f"❌ {symbol}: TP ({tp}) трябва да е по-висок от entry ({price})")
                        return None
                else:  # SELL
                    if sl <= price:
                        logger.error(f"❌ {symbol}: SL ({sl}) трябва да е по-висок от entry ({price})")
                        return None
                    if tp >= price:
                        logger.error(f"❌ {symbol}: TP ({tp}) трябва да е по-нисък от entry ({price})")
                        return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_code,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 20250101,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ {symbol} Поръчката неуспешна: {result.retcode} — {result.comment}")
            return None

        logger.info(f"✅ {order_type} {volume} {symbol} @ {price:.5f} | Ticket: {result.order}")
        return {"ticket": result.order, "symbol": symbol, "type": order_type,
                "volume": volume, "price": price, "price_open": price, "sl": sl, "tp": tp}

    def close_position(self, ticket: int) -> bool:
        """Затваря напълно дадена позиция (Пренасочва към partial с None обем)."""
        return self.close_partial_position(ticket, None)

    def close_partial_position(self, ticket: int, volume_to_close: Optional[float] = None) -> bool:
        """Обединена функция за пълно и частично затваряне."""
        if self._mt5 is None:
            vol_str = f"{volume_to_close} лота" if volume_to_close else "всичко"
            logger.info(f"📋 [СИМУЛАЦИЯ] Затваряне на {vol_str} от позиция {ticket}")
            return True

        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)

        if not position:
            logger.error(f"❌ Грешка: Позиция {ticket} не е намерена за затваряне.")
            return False

        pos = position[0]

        # Ако не е подаден обем или подаденият е по-голям от наличния, затваряме всичко
        close_vol = float(volume_to_close) if volume_to_close and volume_to_close < pos.volume else pos.volume

        if close_vol <= 0:
            return False

        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)

        if tick is None:
            return False

        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_vol,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 20250101,
            "comment": "AI_PARTIAL" if volume_to_close else "AI_BOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Неуспешно затваряне ({close_vol} лота): {result.retcode} — {result.comment}")
            return False

        return True

    def modify_sl(self, ticket: int, new_sl: float, current_tp: float) -> bool:
        """Модифицира Stop Loss (за trailing stop)."""
        if self._mt5 is None:
            return True
        mt5 = self._mt5
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(new_sl),
            "tp": float(current_tp)
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def get_deal_exit_price(self, ticket: int) -> Optional[float]:
        """Търси реалната цена на затваряне в историята с изчакване."""
        import MetaTrader5 as mt5
        import time
        from datetime import datetime, timedelta

        if self._mt5 is None: return None

        # Опитваме в продължение на 5 секунди (през 1 сек)
        for attempt in range(5):
            from_date = datetime.now() - timedelta(hours=4)
            deals = mt5.history_deals_get(from_date, datetime.now())

            if deals:
                for d in deals:
                    if d.position_id == ticket and d.entry == mt5.DEAL_ENTRY_OUT:
                        return float(d.price)

            time.sleep(1.0)

        return None

    def _simulate_candles(self, symbol: str, count: int) -> pd.DataFrame:
        """Генерира симулирани свещи за тестване без MT5."""
        np.random.seed(hash(symbol) % 2 ** 31)
        dates = pd.date_range(end=datetime.now(), periods=count, freq="1h")

        base = {
            "EURUSD": 1.0850, "GBPUSD": 1.2540, "USDJPY": 155.0,
            "USDCHF": 0.9050, "AUDUSD": 0.6500, "USDCAD": 1.3650,
            "NZDUSD": 0.5950, "EURJPY": 168.00, "EURGBP": 0.8550,
            "XAUUSD": 2350.0, "NAS100": 18000.0, "SP500": 5100.0
        }

        clean_symbol = symbol.replace("+", "").upper()
        price = base.get(clean_symbol, 100.0)

        closes = [price]
        for _ in range(count - 1):
            closes.append(closes[-1] * (1 + np.random.normal(0, 0.001)))

        closes = np.array(closes)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.0005, count)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.0005, count)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0] * 0.999

        volumes = np.random.randint(1000, 50000, count).astype(float)
        return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                             "Close": closes, "Volume": volumes}, index=dates)