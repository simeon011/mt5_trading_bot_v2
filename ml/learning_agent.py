"""
ml/learning_agent.py — Самообучаващ се ML агент (FIXED)
✅ Ново: По-добро логиране на преобучаването

Методи на обучение:
1. Random Forest — класификация на сигнали
2. Reinforcement Learning (Q-Learning) — оптимизиране на решения
3. Pattern Memory — запомня кои конфигурации работят
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("LearningAgent")


@dataclass
class TradeRecord:
    """Запис на затворена сделка за обучение."""
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    profit_pips: float
    profit_pct: float
    outcome: str        # "WIN", "LOSS", "BREAKEVEN"
    duration_hours: float

    # Сигнали при влизане
    rsi_at_entry: float
    macd_hist_at_entry: float
    ma_alignment: float
    ob_score: float
    candle_score: float
    trendline_score: float
    total_score: float
    market_hour: int
    day_of_week: int

    timestamp: str = ""

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.rsi_at_entry / 100,
            self.macd_hist_at_entry,
            self.ma_alignment,
            self.ob_score / 20,
            self.candle_score / 15,
            self.trendline_score / 15,
            self.total_score / 100,
            self.market_hour / 24,
            self.day_of_week / 7,
            1 if self.direction == "BUY" else 0,
        ])


class LearningAgent:
    """
    Самообучаващ се агент базиран на затворени сделки.

    Алгоритъм:
    1. Всяка затворена сделка се записва с пълен контекст
    2. На всеки N сделки → преобучаване на модела
    3. Моделът предсказва вероятност за успех на нов сигнал
    4. Q-Learning reward система: +1 WIN, -1 LOSS
    5. Pattern Memory: запомня какви комбинации работят
    """

    def __init__(self, settings):
        self.s = settings
        self.model_path = os.path.join(settings.MODEL_DIR, "trading_model.pkl")
        self.memory_path = os.path.join(settings.DATA_DIR, "trade_memory.json")
        self.q_table_path = os.path.join(settings.DATA_DIR, "q_table.json")

        self.model = None
        self.trade_history: List[TradeRecord] = []
        self.pattern_memory: Dict = {}
        self.q_table: Dict = {}
        self.performance_stats: Dict = {}

        # ✅ НОВО: Брой за счетоводство на преобучаванията
        self.last_retrain_count = 0

        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self._load_memory()
        self._load_model()
        logger.info(f"🧠 ML агент зареден | {len(self.trade_history)} сделки в паметта")

    # ── Запис на сделки ──────────────────────────────────────

    def record_trade(self, trade: TradeRecord):
        """Записва затворена сделка и задейства обучение."""
        trade.timestamp = datetime.now().isoformat()
        self.trade_history.append(trade)
        self._update_pattern_memory(trade)
        self._update_q_table(trade)
        self._save_memory()

        n = len(self.trade_history)

        # ═════════════════════════════════════════════════════════════
        # ✅ ФИКС за преобучаване логиране (Проблем 2)
        # Проверяваме ако е време за переучаване И показваме съобщение
        # ═════════════════════════════════════════════════════════════

        if n >= self.s.ML_MIN_TRADES_TO_TRAIN:
            if (n - self.last_retrain_count) >= self.s.ML_RETRAIN_INTERVAL:
                logger.info(f"{'='*60}")
                logger.self_study(f"🔄 ПРЕОБУЧАВАНЕ НА ML МОДЕЛ")
                logger.info(f"   Сделки за обучение: {n}")
                logger.info(f"   Последно преобучаване беше при: {self.last_retrain_count}")
                logger.info(f"   Разлика: {n - self.last_retrain_count} сделки")
                logger.info(f"{'='*60}")

                metrics = self.train()
                self.last_retrain_count = n

                if metrics:
                    logger.new_model(f"✅ МОДЕЛ ПРЕОБУЧЕН УСПЕШНО")
                    logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
                    logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
                    logger.info(f"   Топ фактори: {list(metrics.get('feature_importance', {}).items())[:3]}")

        self._update_performance_stats()
        logger.info(f"📝 Сделка записана: {trade.outcome} | "
                    f"Profit: {trade.profit_pct:+.2f}% | "
                    f"Общо: {n} сделки | "
                    f"До преобучаване: {self.s.ML_RETRAIN_INTERVAL - ((n - self.last_retrain_count) % self.s.ML_RETRAIN_INTERVAL)}")

    # ── Обучение ─────────────────────────────────────────────

    def train(self) -> Dict:
        """Обучава Random Forest модел от историята на сделките."""
        if len(self.trade_history) < self.s.ML_MIN_TRADES_TO_TRAIN:
            logger.info(f"Нужни са {self.s.ML_MIN_TRADES_TO_TRAIN} сделки. "
                        f"Имаме {len(self.trade_history)}.")
            return {}

        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("sklearn не е инсталиран. pip install scikit-learn")
            return {}

        X = np.array([t.to_feature_vector() for t in self.trade_history])
        # ✅ Правилна WIN/LOSS логика
        y = np.array([1 if t.outcome == "WIN" else 0 for t in self.trade_history])

        if len(np.unique(y)) < 2:
            logger.warning("Нужни са и WIN и LOSS сделки за обучение.")
            return {}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ensemble от 2 модела
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)

        # Cross-validation
        rf_cv = cross_val_score(rf, X_scaled, y, cv=min(5, len(y)//5+1), scoring="accuracy")

        self.model = {"rf": rf, "gb": gb, "scaler": scaler}

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Feature importance
        feature_names = ["RSI", "MACD", "MA_Align", "OB", "Candle",
                         "Trendline", "Total_Score", "Hour", "DayOfWeek", "Direction"]
        importances = rf.feature_importances_
        importance_dict = dict(zip(feature_names, importances))

        win_rate = y.mean()
        metrics = {
            "accuracy": float(rf_cv.mean()),
            "win_rate": float(win_rate),
            "total_trades": len(self.trade_history),
            "feature_importance": {k: round(float(v), 3)
                                   for k, v in sorted(importance_dict.items(),
                                                       key=lambda x: x[1], reverse=True)}
        }

        logger.info(f"✅ Модел обучен | Accuracy: {metrics['accuracy']:.1%} | "
                    f"Win Rate: {win_rate:.1%} | Сделки: {len(y)}")
        logger.info(f"Топ фактори: {list(metrics['feature_importance'].items())[:3]}")

        return metrics

    # ── Предсказване ─────────────────────────────────────────

    def predict(self, features: Dict) -> float:
        """
        Предсказва вероятност за успех на сигнал.
        Връща float 0.0-1.0 (>0.6 = bullish bias, <0.4 = bearish bias)
        """
        if self.model is None or len(self.trade_history) < self.s.ML_MIN_TRADES_TO_TRAIN:
            return 0.5  # Неутрален без достатъчно данни

        feature_vec = np.array([[
            features.get("rsi", 50) / 100,
            features.get("macd_hist", 0),
            features.get("ma_alignment", 0),
            features.get("ob_score", 0) / 20,
            features.get("candle_score", 0) / 15,
            features.get("trendline_score", 0) / 15,
            features.get("total_score", 50) / 100,
            features.get("hour", 12) / 24,
            features.get("day_of_week", 2) / 7,
            1 if features.get("direction") == "BUY" else 0,
        ]])

        X_scaled = self.model["scaler"].transform(feature_vec)
        rf_prob = self.model["rf"].predict_proba(X_scaled)[0][1]
        gb_prob = self.model["gb"].predict_proba(X_scaled)[0][1]
        ensemble = (rf_prob * 0.6 + gb_prob * 0.4)

        # Q-learning корекция
        state = self._get_state_key(features)
        q_adj = self.q_table.get(state, {}).get("value", 0) * 0.1
        result = float(np.clip(ensemble + q_adj, 0.1, 0.9))

        return result

    # ── Динамични Параметри (AI Решения) ─────────────────────

    def get_smart_trade_params(self, symbol: str, confidence: float, features: Dict) -> Dict[str, float]:
        """
        Умен алгоритъм за избор на лот.
        Колкото повече е научен ботът и колкото по-добре се справя → толкова по-смело влиза.

        Формула: base_lot × опит × win_rate × confidence × сесия
        """
        base_lot = self.s.MIN_LOT_SIZE
        n_trades = len(self.trade_history)

        # ── 1. ОПИТ: Колко сделки е научил ботът ─────────────────────
        # Начинаещ бот е консервативен. Опитен бот може да рискува повече.
        if n_trades < 30:
            exp_mult = 1.0      # Новак: само минимален лот
        elif n_trades < 100:
            exp_mult = 1.2      # Малко опит
        elif n_trades < 250:
            exp_mult = 1.5      # Среден опит
        else:
            exp_mult = 2.0      # Опитен: до двоен лот

        # ── 2. ПОСЛЕДНИ РЕЗУЛТАТИ: Как се справя в момента ───────────
        # Ако последните 20 сделки са лоши → намалява лота (предпазливост)
        # Ако последните 20 са добри → увеличава лота (увереност)
        recent = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        if recent:
            recent_wr = sum(1 for t in recent if t.outcome == "WIN") / len(recent)
            if recent_wr < 0.35:
                wr_mult = 0.6   # Лоша серия → силно намаляване
            elif recent_wr < 0.45:
                wr_mult = 0.8   # Под средно → малко намаляване
            elif recent_wr > 0.65:
                wr_mult = 1.3   # Добра серия → увеличаване
            else:
                wr_mult = 1.0   # Нормално
        else:
            wr_mult = 1.0

        # ── 3. УВЕРЕНОСТ НА ML СИГНАЛА ────────────────────────────────
        if confidence >= 0.80:
            conf_mult = 1.5     # Много уверен сигнал
        elif confidence >= 0.70:
            conf_mult = 1.2
        elif confidence >= 0.60:
            conf_mult = 1.0
        else:
            conf_mult = 0.8     # Слаб сигнал → намален лот

        # ── 4. ТЪРГОВСКА СЕСИЯ ────────────────────────────────────────
        # Активните сесии имат по-голяма ликвидност → по-предвидими движения
        hour = features.get("hour", 12)
        if 8 <= hour < 12:
            session_mult = 1.4      # Лондон отваряне (най-активно)
        elif 12 <= hour < 13:
            session_mult = 1.6      # Лондон/NY Overlap (максимален обем)
        elif 13 <= hour < 17:
            session_mult = 1.3      # Нюйоркска сесия
        elif 0 <= hour < 8:
            session_mult = 0.6      # Азиатска сесия (по-тихо) → малък лот
        else:
            session_mult = 0.8      # Извън активни сесии

        # ── 5. ФИНАЛЕН ЛОТ ───────────────────────────────────────────
        raw_volume = base_lot * exp_mult * wr_mult * conf_mult * session_mult
        # Закръгляме до 0.01 и гарантираме минимум
        volume = max(base_lot, round(raw_volume, 2))

        # ── TP/SL множители (зависят от сесията) ─────────────────────
        state = self._get_state_key(features)
        if "asia" in state:
            best_tp_mult = 1.0      # По-малки цели при Азия
        else:
            best_tp_mult = 1.5

        logger.info(
            f"🎲 Лот: {volume} | Опит:{n_trades} ({exp_mult:.1f}x) | "
            f"WR:{recent_wr*100:.0f}% ({wr_mult:.1f}x) | "
            f"ML:{confidence*100:.0f}% ({conf_mult:.1f}x) | "
            f"Сесия:{hour}ч ({session_mult:.1f}x)"
            if recent else
            f"🎲 Лот: {volume} | Опит:{n_trades} (новак)"
        )

        return {
            "volume": volume,
            "tp_multiplier": best_tp_mult,
            "sl_multiplier": 1.0
        }

    # ── Q-Learning ────────────────────────────────────────────

    def _get_state_key(self, features: Dict) -> str:
        """Дискретизира features в state за Q-таблица."""
        rsi_zone = "low" if features.get("rsi", 50) < 40 else \
                   "high" if features.get("rsi", 50) > 60 else "mid"
        trend = "bull" if features.get("ma_alignment", 0) > 0.5 else "bear"
        hour_zone = "asia" if features.get("hour", 12) < 8 else \
                    "london" if features.get("hour", 12) < 16 else "ny"
        return f"{rsi_zone}_{trend}_{hour_zone}"

    def _update_q_table(self, trade: TradeRecord):
        """Q-Learning update: учи кои market conditions работят."""
        features = {
            "rsi": trade.rsi_at_entry,
            "ma_alignment": trade.ma_alignment,
            "hour": trade.market_hour,
            "direction": trade.direction
        }
        state = self._get_state_key(features)
        reward = 1.0 if trade.outcome == "WIN" else -1.0
        reward *= abs(trade.profit_pct) / 2  # По-силна корекция за по-голяма печалба/загуба

        alpha = 0.1   # Learning rate
        gamma = 0.95  # Discount factor

        if state not in self.q_table:
            self.q_table[state] = {"value": 0.0, "count": 0}

        old_val = self.q_table[state]["value"]
        self.q_table[state]["value"] = old_val + alpha * (reward - old_val)
        self.q_table[state]["count"] += 1

        with open(self.q_table_path, "w") as f:
            json.dump(self.q_table, f, indent=2)

    # ── Pattern Memory ───────────────────────────────────────

    def _update_pattern_memory(self, trade: TradeRecord):
        """Запомня кои pattern комбинации имат висок win rate."""
        key = f"{trade.symbol}_{trade.direction}_ob{int(trade.ob_score)}_candle{int(trade.candle_score)}"
        if key not in self.pattern_memory:
            self.pattern_memory[key] = {"wins": 0, "losses": 0, "total_profit": 0}
        if trade.outcome == "WIN":
            self.pattern_memory[key]["wins"] += 1
        else:
            self.pattern_memory[key]["losses"] += 1
        self.pattern_memory[key]["total_profit"] += trade.profit_pct

    def get_pattern_win_rate(self, symbol: str, direction: str,
                              ob_score: float, candle_score: float) -> Optional[float]:
        key = f"{symbol}_{direction}_ob{int(ob_score)}_candle{int(candle_score)}"
        if key in self.pattern_memory:
            m = self.pattern_memory[key]
            total = m["wins"] + m["losses"]
            if total >= 3:
                return m["wins"] / total
        return None

    # ── Статистики ───────────────────────────────────────────

    def _update_performance_stats(self):
        if not self.trade_history:
            return
        wins = [t for t in self.trade_history if t.outcome == "WIN"]
        losses = [t for t in self.trade_history if t.outcome == "LOSS"]
        profits = [t.profit_pct for t in self.trade_history]

        self.performance_stats = {
            "total_trades": len(self.trade_history),
            "win_rate": len(wins) / len(self.trade_history),
            "avg_profit": float(np.mean(profits)),
            "best_trade": float(max(profits)) if profits else 0,
            "worst_trade": float(min(profits)) if profits else 0,
            "profit_factor": (sum(t.profit_pct for t in wins) /
                              abs(sum(t.profit_pct for t in losses)))
                              if losses else float("inf"),
            "avg_win": float(np.mean([t.profit_pct for t in wins])) if wins else 0,
            "avg_loss": float(np.mean([t.profit_pct for t in losses])) if losses else 0,
        }

    def get_stats(self) -> Dict:
        return self.performance_stats

    # ── Persistence ──────────────────────────────────────────

    def _save_memory(self):
        data = {
            "trades": [asdict(t) for t in self.trade_history],
            "pattern_memory": self.pattern_memory
        }
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_memory(self):
        if not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path) as f:
                data = json.load(f)
            self.trade_history = [TradeRecord(**t) for t in data.get("trades", [])]
            self.pattern_memory = data.get("pattern_memory", {})
            if os.path.exists(self.q_table_path):
                with open(self.q_table_path) as f:
                    self.q_table = json.load(f)
        except Exception as e:
            logger.error(f"Грешка при зареждане на памет: {e}")

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("✅ ML модел зареден от диск.")
            except Exception as e:
                logger.error(f"Грешка при зареждане на модел: {e}")