<img width="720" height="400" alt="ChatGPT Image 28 04 2026 г , 18_47_45" src="https://github.com/user-attachments/assets/87c1236e-76a2-4a49-8ecc-37d83a565b12" />



# MT5 AI Trading Bot 🤖📈

Самообучаващ се алготрейдинг бот за MetaTrader 5.  
Поддържа **Gold (XAUUSD)**, **Forex** и **Indices**.

---

## 🗂️ Структура на проекта

```
mt5_bot/
├── main.py                  ← Стартиране
├── requirements.txt
├── config/
│   └── settings.py          ← ВСИЧКИ НАСТРОЙКИ (редактирай тук!)
├── core/
│   ├── mt5_connector.py     ← MT5 връзка & поръчки
│   ├── risk_manager.py      ← Управление на риска
│   └── bot.py               ← Главен оркестратор
├── strategies/
│   ├── indicators.py        ← RSI, MACD, MA, OB, Trendlines, Patterns
│   └── signal_engine.py     ← Scoring система (0-100)
├── ml/
│   └── learning_agent.py    ← Self-learning (RF + Q-Learning)
├── data/                    ← Trade история (auto-created)
├── models/                  ← ML модели (auto-created)
└── logs/                    ← Логове (auto-created)
```

---

## ⚙️ Инсталация

### 1. Изисквания
- Windows 10/11 (MT5 работи само на Windows)
- Python 3.10+
- MetaTrader 5 терминал инсталиран

### 2. Инсталирай библиотеките
```bash
pip install -r requirements.txt
```

### 3. Конфигурирай `config/settings.py`
```python
MT5_LOGIN = 123456          # Твоят акаунт номер
MT5_PASSWORD = "парола"
MT5_SERVER = "ICMarkets-Demo"
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"
```

---

## 🚀 Стартиране

```bash
# Paper trading (без реални пари) — ЗАПОЧНИ ТУК!
python main.py --mode paper

# Backtest на исторически данни
python main.py --mode backtest --backtest-from 2024-01-01 --backtest-to 2024-12-31

# Live trading (САМО след тестване!)
python main.py --mode live
```

---

## 🧠 Как се учи ботът

```
Сделка затворена
      ↓
Записва резултат + пазарен контекст
      ↓
   ┌──────────────────────────────┐
   │  Random Forest               │ ← Учи кои индикатори предсказват WIN
   │  Gradient Boosting           │ ← Допълнителен ensemble модел
   │  Q-Learning таблица          │ ← Учи кои market conditions работят
   │  Pattern Memory              │ ← Запомня кои pattern комбинации работят
   └──────────────────────────────┘
      ↓
На всеки 100 сделки → преобучаване
      ↓
ML score коригира бъдещите сигнали (±10 точки)
```

---

## 📊 Scoring система

| Индикатор | Макс точки | Описание |
|-----------|-----------|----------|
| RSI | 15 | Oversold/Overbought + momentum |
| MACD | 15 | Crossover + histogram |
| Moving Averages | 20 | MA stack + H4 потвърждение |
| Order Blocks | 20 | Цена в зона на OB |
| Trendlines/Channels | 15 | Support/Resistance + breakout |
| Candlestick Patterns | 15 | Engulfing, Pin Bar, Stars... |
| ML корекция | ±10 | Self-learning adjustment |
| **ОБЩО** | **100** | **Праг за сделка: 65** |

---

## 🛡️ Управление на риска

- **Risk per trade**: 1% от баланса (настройва се)
- **Max daily loss**: 3% → бот спира автоматично
- **Max open trades**: 3 едновременно
- **Trailing Stop**: активен, следва цената с ATR
- **SL базиран на**: Order Block структура или 1.5x ATR
- **TP базиран на**: 2:1 Risk/Reward (минимум 1.5:1)

---

## ⚠️ ВАЖНИ ПРЕДУПРЕЖДЕНИЯ

> **НИКОГА не стартирай на реален акаунт без:**
> 1. ✅ Минимум 1 месец paper trading тест
> 2. ✅ Backtest на поне 6 месеца данни
> 3. ✅ Разбиране на всеки модул
> 4. ✅ Тест с минимален реален акаунт ($100-500)

> Алготрейдингът носи висок финансов риск.  
> Пазарите могат да се държат непредвидимо.  
> Използвай само средства, чиято загуба можеш да понесеш.

---

## 🔧 Следващи стъпки (подобрения)

- [ ] Telegram известия за всяка сделка
- [ ] Dashboard с реално-времеви статистики
- [ ] News filter (не трейдва около новини)
- [ ] Session filter (само London/NY сесии)
- [ ] Backtest engine с детайлни отчети
- [ ] Deep Learning модел (LSTM за price prediction)

---

## 📞 Поддръжка

Кодът е коментиран на Bulgarian/English.  
Всеки модул може да се тества независимо.
