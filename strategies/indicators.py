"""
strategies/indicators.py — Всички технически индикатори
RSI, MACD, MA, ATR, Order Blocks, Trendlines, Channels, Candlestick Patterns
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger("Indicators")


@dataclass
class OrderBlock:
    index: int
    time: pd.Timestamp
    high: float
    low: float
    ob_type: str        # "BULLISH" или "BEARISH"
    strength: float     # 0-1, колко силен е OB
    tested: bool = False
    broken: bool = False


@dataclass
class Trendline:
    start_idx: int
    end_idx: int
    slope: float
    intercept: float
    line_type: str      # "SUPPORT" или "RESISTANCE"
    touches: int        # Брой допирания
    strength: float


@dataclass
class CandlePattern:
    name: str
    direction: str      # "BULLISH" eller "BEARISH"
    strength: float     # 0-1
    index: int


@dataclass
class MarketStructure:
    bos_direction: str       # "BULLISH", "BEARISH", "NONE"
    choch_detected: bool
    choch_direction: str     # "BULLISH" (обрат нагоре), "BEARISH" (обрат надолу), "NONE"
    last_swing_high: float
    last_swing_low: float
    trend: str               # "UPTREND", "DOWNTREND", "RANGING"


@dataclass
class EqualLevels:
    equal_highs: List[float]          # Ценови нива на Equal Highs
    equal_lows: List[float]           # Ценови нива на Equal Lows
    price_swept_eq_high: bool         # Цената е ударила и отблъсната от EQH (ликвидност взета)
    price_swept_eq_low: bool          # Цената е ударила и отблъсната от EQL (ликвидност взета)
    price_broke_eq_high: bool         # Цената е пробила и затворила над EQH (силен BUY)
    price_broke_eq_low: bool          # Цената е пробила и затворила под EQL (силен SELL)
    nearest_eq_high: Optional[float]  # Най-близкото EQH над цената
    nearest_eq_low: Optional[float]   # Най-близкото EQL под цената


class TechnicalIndicators:
    """Изчислява всички технически индикатори."""

    # ── Trend Indicators ────────────────────────────────────

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20,
                        std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = mid + std * std_dev
        lower = mid - std * std_dev
        return upper, mid, lower

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14,
                   d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = df["Low"].rolling(window=k_period).min()
        highest_high = df["High"].rolling(window=k_period).max()
        k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return k, d

    # ── Order Blocks ────────────────────────────────────────

    @staticmethod
    def find_order_blocks(df: pd.DataFrame, lookback: int = 50,
                          min_size_atr_mult: float = 0.5) -> List[OrderBlock]:
        """
        Order Block = последната свещ преди силно движение.
        Bullish OB: последна мечи свещ преди импулс нагоре.
        Bearish OB: последна бича свещ преди импулс надолу.
        """
        blocks = []
        atr_vals = TechnicalIndicators.atr(df).values
        closes = df["Close"].values
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)

        for i in range(3, min(lookback, n - 3)):
            idx = n - 1 - i
            if idx < 3:
                continue

            atr_val = atr_vals[idx]
            if atr_val <= 0:
                continue

            # Движение след OB (3 свещи напред)
            move_up = max(highs[idx+1:idx+4]) - closes[idx]
            move_down = closes[idx] - min(lows[idx+1:idx+4])

            # Bullish OB: мечи свещ + силно движение нагоре
            if (opens[idx] > closes[idx] and  # bearish candle
                    move_up > atr_val * min_size_atr_mult * 2):
                strength = min(1.0, move_up / (atr_val * 4))
                # Проверка дали е бил тестван или пробит
                tested = bool((lows[idx+4:] < highs[idx]).any()) if idx+4 < n else False
                broken = bool((closes[idx+1:] < lows[idx]).any()) if idx+1 < n else False
                if not broken:
                    blocks.append(OrderBlock(
                        index=idx,
                        time=df.index[idx],
                        high=highs[idx],
                        low=lows[idx],
                        ob_type="BULLISH",
                        strength=strength,
                        tested=tested,
                        broken=broken
                    ))

            # Bearish OB: бича свещ + силно движение надолу
            elif (closes[idx] > opens[idx] and  # bullish candle
                      move_down > atr_val * min_size_atr_mult * 2):
                strength = min(1.0, move_down / (atr_val * 4))
                broken = bool((closes[idx+1:] > highs[idx]).any()) if idx+1 < n else False
                if not broken:
                    blocks.append(OrderBlock(
                        index=idx,
                        time=df.index[idx],
                        high=highs[idx],
                        low=lows[idx],
                        ob_type="BEARISH",
                        strength=strength,
                        broken=broken
                    ))

        return sorted(blocks, key=lambda x: x.strength, reverse=True)[:10]

    # ── Trendlines & Channels ────────────────────────────────

    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Намира swing highs и swing lows."""
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        swing_highs = []
        swing_lows = []

        for i in range(window, n - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append(i)
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append(i)

        return swing_highs, swing_lows

    @staticmethod
    def find_trendlines(df: pd.DataFrame, min_touches: int = 2) -> List[Trendline]:
        """Намира trendlines от swing points."""
        lines = []
        swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df)
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        atr_val = TechnicalIndicators.atr(df).iloc[-1]
        tolerance = atr_val * 0.3

        def fit_line(points, values):
            if len(points) < 2:
                return []
            found = []
            for i in range(len(points)-1):
                for j in range(i+1, len(points)):
                    x1, x2 = points[i], points[j]
                    y1, y2 = values[x1], values[x2]
                    if x2 == x1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    touches = sum(
                        1 for k in points
                        if abs(values[k] - (slope * k + intercept)) < tolerance
                    )
                    if touches >= min_touches:
                        found.append((slope, intercept, touches, x1, x2))
            return found

        # Resistance trendlines (от swing highs)
        for slope, intercept, touches, s, e in fit_line(swing_highs[-20:], highs):
            strength = min(1.0, touches / 5)
            lines.append(Trendline(s, e, slope, intercept, "RESISTANCE", touches, strength))

        # Support trendlines (от swing lows)
        for slope, intercept, touches, s, e in fit_line(swing_lows[-20:], lows):
            strength = min(1.0, touches / 5)
            lines.append(Trendline(s, e, slope, intercept, "SUPPORT", touches, strength))

        return sorted(lines, key=lambda x: x.strength, reverse=True)[:8]

    @staticmethod
    def get_channel_position(df: pd.DataFrame, trendlines: List[Trendline]) -> Optional[Dict]:
        """Определя позицията на цената спрямо channel."""
        supports = [t for t in trendlines if t.line_type == "SUPPORT"]
        resistances = [t for t in trendlines if t.line_type == "RESISTANCE"]
        if not supports or not resistances:
            return None

        n = len(df) - 1
        current_price = df["Close"].iloc[-1]
        support = supports[0]
        resistance = resistances[0]

        sup_price = support.slope * n + support.intercept
        res_price = resistance.slope * n + resistance.intercept

        if res_price <= sup_price:
            return None

        channel_height = res_price - sup_price
        position = (current_price - sup_price) / channel_height

        return {
            "support_price": sup_price,
            "resistance_price": res_price,
            "position": position,      # 0=bottom, 1=top
            "channel_height": channel_height,
            "near_support": position < 0.2,
            "near_resistance": position > 0.8
        }

    # ── Candlestick Patterns ─────────────────────────────────

    @staticmethod
    def find_candle_patterns(df: pd.DataFrame) -> List[CandlePattern]:
        """Разпознава свещни модели."""
        patterns = []
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        n = len(df)

        if n < 3:
            return patterns

        def body(i): return abs(c[i] - o[i])
        def upper_wick(i): return h[i] - max(o[i], c[i])
        def lower_wick(i): return min(o[i], c[i]) - l[i]
        def is_bullish(i): return c[i] > o[i]
        def is_bearish(i): return c[i] < o[i]

        i = n - 1  # Последна свещ

        # Hammer / Pin Bar (bullish)
        if (lower_wick(i) > body(i) * 2 and
                upper_wick(i) < body(i) * 0.5 and
                body(i) > 0):
            strength = min(1.0, lower_wick(i) / (body(i) * 4))
            patterns.append(CandlePattern("Hammer/Pin Bar", "BULLISH", strength, i))

        # Shooting Star / Bearish Pin Bar
        if (upper_wick(i) > body(i) * 2 and
                lower_wick(i) < body(i) * 0.5 and
                body(i) > 0):
            strength = min(1.0, upper_wick(i) / (body(i) * 4))
            patterns.append(CandlePattern("Shooting Star", "BEARISH", strength, i))

        # Doji
        if body(i) < (h[i] - l[i]) * 0.1 and (h[i] - l[i]) > 0:
            patterns.append(CandlePattern("Doji", "NEUTRAL", 0.5, i))

        if n >= 2:
            # Engulfing Bullish
            if (is_bearish(i-1) and is_bullish(i) and
                    c[i] > o[i-1] and o[i] < c[i-1] and
                    body(i) > body(i-1)):
                strength = min(1.0, body(i) / (body(i-1) + 1e-10))
                patterns.append(CandlePattern("Bullish Engulfing", "BULLISH", strength, i))

            # Engulfing Bearish
            if (is_bullish(i-1) and is_bearish(i) and
                    c[i] < o[i-1] and o[i] > c[i-1] and
                    body(i) > body(i-1)):
                strength = min(1.0, body(i) / (body(i-1) + 1e-10))
                patterns.append(CandlePattern("Bearish Engulfing", "BEARISH", strength, i))

        if n >= 3:
            # Morning Star (bullish reversal)
            if (is_bearish(i-2) and
                    body(i-1) < body(i-2) * 0.3 and
                    is_bullish(i) and
                    c[i] > (o[i-2] + c[i-2]) / 2):
                patterns.append(CandlePattern("Morning Star", "BULLISH", 0.8, i))

            # Evening Star (bearish reversal)
            if (is_bullish(i-2) and
                    body(i-1) < body(i-2) * 0.3 and
                    is_bearish(i) and
                    c[i] < (o[i-2] + c[i-2]) / 2):
                patterns.append(CandlePattern("Evening Star", "BEARISH", 0.8, i))

            # Three White Soldiers
            if all(is_bullish(i-k) for k in range(3)):
                if c[i] > c[i-1] > c[i-2]:
                    patterns.append(CandlePattern("Three White Soldiers", "BULLISH", 0.85, i))

            # Three Black Crows
            if all(is_bearish(i-k) for k in range(3)):
                if c[i] < c[i-1] < c[i-2]:
                    patterns.append(CandlePattern("Three Black Crows", "BEARISH", 0.85, i))

        return patterns

    # ── Market Structure: BOS & CHOCH ───────────────────────────

    @staticmethod
    def detect_market_structure(df: pd.DataFrame, window: int = 5) -> MarketStructure:
        """
        Открива BOS (Break of Structure) и CHOCH (Change of Character).

        BOS: Цената пробива последния Swing High/Low → потвърждение на тренда.
        CHOCH: Трендът се обръща → цената пробива в обратна посока.
        """
        swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, window)

        closes = df["Close"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        current_close = float(closes[-1])

        # Нужни са поне 2 swing high и 2 swing low за анализ
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure(
                bos_direction="NONE", choch_detected=False,
                choch_direction="NONE", last_swing_high=0.0,
                last_swing_low=0.0, trend="RANGING"
            )

        last_sh  = float(highs[swing_highs[-1]])
        prev_sh  = float(highs[swing_highs[-2]])
        last_sl  = float(lows[swing_lows[-1]])
        prev_sl  = float(lows[swing_lows[-2]])

        # Определяме текущата структура
        # Uptrend  = Higher High + Higher Low
        # Downtrend = Lower High  + Lower Low
        uptrend   = last_sh > prev_sh and last_sl > prev_sl
        downtrend = last_sh < prev_sh and last_sl < prev_sl
        trend = "UPTREND" if uptrend else ("DOWNTREND" if downtrend else "RANGING")

        # ── BOS ──────────────────────────────────────────────────
        # Цената затваря над последния Swing High → Bullish BOS (тренд нагоре потвърден)
        # Цената затваря под последния Swing Low  → Bearish BOS (тренд надолу потвърден)
        bos_direction = "NONE"
        if current_close > last_sh:
            bos_direction = "BULLISH"
        elif current_close < last_sl:
            bos_direction = "BEARISH"

        # ── CHOCH ────────────────────────────────────────────────
        # В Uptrend: цената пробива под последния Swing Low → обрат надолу
        # В Downtrend: цената пробива над последния Swing High → обрат нагоре
        choch_detected  = False
        choch_direction = "NONE"

        if uptrend and current_close < last_sl:
            choch_detected  = True
            choch_direction = "BEARISH"
        elif downtrend and current_close > last_sh:
            choch_detected  = True
            choch_direction = "BULLISH"

        return MarketStructure(
            bos_direction=bos_direction,
            choch_detected=choch_detected,
            choch_direction=choch_direction,
            last_swing_high=last_sh,
            last_swing_low=last_sl,
            trend=trend
        )

    # ── Equal Highs & Equal Lows ─────────────────────────────────

    @staticmethod
    def find_equal_levels(df: pd.DataFrame, window: int = 5,
                          tolerance_atr_mult: float = 0.25) -> EqualLevels:
        """
        Открива Equal Highs и Equal Lows (ликвидни зони).

        Equal High: два или повече Swing High на почти еднакво ниво
        → Там седят стоповете на retail купувачи → институционалните ги таргетират

        Equal Low: два или повече Swing Low на почти еднакво ниво
        → Там седят стоповете на retail продавачи → институционалните ги таргетират

        Sweep: Цената удари нивото и се върна → ликвидността е взета → обрат очакван
        Break: Цената затвори зад нивото → продължение очаквано
        """
        swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, window)

        highs  = df["High"].values
        lows   = df["Low"].values
        closes = df["Close"].values
        current_close = float(closes[-1])
        current_high  = float(highs[-1])
        current_low   = float(lows[-1])

        atr_val   = float(TechnicalIndicators.atr(df).iloc[-1])
        tolerance = atr_val * tolerance_atr_mult

        # ── Equal Highs ──────────────────────────────────────────
        # Изискваме поне 2 различни swing highs в зоната (не просто шум)
        eq_highs: List[float] = []
        sh_prices = [float(highs[i]) for i in swing_highs[-15:]]
        for i in range(len(sh_prices)):
            cluster = [sh_prices[i]]
            for j in range(len(sh_prices)):
                if i != j and abs(sh_prices[i] - sh_prices[j]) <= tolerance:
                    cluster.append(sh_prices[j])
            if len(cluster) >= 2:   # Минимум 2 touch-а за валиден Equal High
                level = round(sum(cluster) / len(cluster), 5)
                if level not in eq_highs:
                    eq_highs.append(level)

        # ── Equal Lows ───────────────────────────────────────────
        eq_lows: List[float] = []
        sl_prices = [float(lows[i]) for i in swing_lows[-15:]]
        for i in range(len(sl_prices)):
            cluster = [sl_prices[i]]
            for j in range(len(sl_prices)):
                if i != j and abs(sl_prices[i] - sl_prices[j]) <= tolerance:
                    cluster.append(sl_prices[j])
            if len(cluster) >= 2:   # Минимум 2 touch-а за валиден Equal Low
                level = round(sum(cluster) / len(cluster), 5)
                if level not in eq_lows:
                    eq_lows.append(level)

        # ── Sweep vs Break Detection ──────────────────────────────
        # Sweep на EQH: Цената е надхвърлила нивото (wick) но затворила под него
        price_swept_eq_high = any(
            current_high >= lvl and current_close < lvl
            for lvl in eq_highs
        )
        # Break на EQH: Цената затвори над нивото → bullish continuation
        price_broke_eq_high = any(current_close > lvl for lvl in eq_highs)

        # Sweep на EQL: Цената е паднала под нивото (wick) но затворила над него
        price_swept_eq_low = any(
            current_low <= lvl and current_close > lvl
            for lvl in eq_lows
        )
        # Break на EQL: Цената затвори под нивото → bearish continuation
        price_broke_eq_low = any(current_close < lvl for lvl in eq_lows)

        # Намираме най-близкото EQH над цената и EQL под цената
        above = [lvl for lvl in eq_highs if lvl > current_close]
        below = [lvl for lvl in eq_lows  if lvl < current_close]
        nearest_eq_high = min(above) if above else None
        nearest_eq_low  = max(below) if below else None

        return EqualLevels(
            equal_highs=sorted(eq_highs),
            equal_lows=sorted(eq_lows),
            price_swept_eq_high=price_swept_eq_high,
            price_swept_eq_low=price_swept_eq_low,
            price_broke_eq_high=price_broke_eq_high,
            price_broke_eq_low=price_broke_eq_low,
            nearest_eq_high=nearest_eq_high,
            nearest_eq_low=nearest_eq_low
        )

    @staticmethod
    def volume_spike(df, period=20):
        """
        Изчислява дали текущият обем е необичайно висок спрямо последните 'period' свещи.
        Връща съотношението (ratio).
        Пример: 1.5 означава, че обемът е 50% над средния.
        """
        # MetaTrader 5 връща обема в колона 'tick_volume'
        vol_col = "tick_volume" if "tick_volume" in df.columns else "Volume"

        if vol_col not in df.columns:
            return 1.0  # Защита, ако липсва колоната

        # Изчисляваме средния обем за последните 20 свещи
        vol_sma = df[vol_col].rolling(window=period).mean()

        # Взимаме обема на последната свещ и средния обем
        current_vol = df[vol_col].iloc[-1]
        avg_vol = vol_sma.iloc[-1]

        # Пресмятаме съотношението
        if avg_vol > 0:
            ratio = current_vol / avg_vol
        else:
            ratio = 1.0

        return ratio
