"""
Kairos Monitor — Telegram alerting agent
Runs alongside the bot and watches the log file for issues.
"""

import time
import os
import re
import glob
import requests
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# ── Telegram Config (от .env) ────────────────────────────────
TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Settings ─────────────────────────────────────────────────
LOG_DIR              = "logs"
CHECK_INTERVAL       = 15        # секунди между проверките
MAX_SILENCE_MINUTES  = 3         # минути без лог → бота е замръзнал
CONSECUTIVE_LOSS_LIMIT = 3       # колко поредни загуби = alert
DAILY_LOSS_WARN_PCT  = 3.0       # % дневна загуба → предупреждение


def send(msg: str):
    """Праща Telegram съобщение."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"[Telegram ERROR] {e}")


def register_commands():
    """Регистрира командите в Telegram — показват се при натискане на /"""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/setMyCommands",
            json={"commands": [
                {"command": "stats",  "description": "Дневна статистика и баланс"},
                {"command": "help",   "description": "Списък с команди"},
            ]},
            timeout=10
        )
    except Exception as e:
        print(f"[Telegram ERROR] {e}")


def get_latest_log() -> str:
    """Намира най-новия лог файл."""
    files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    return max(files, key=os.path.getmtime) if files else None


def tail_file(path: str, last_pos: int) -> tuple:
    """Чете новите редове от лог файла."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            new_pos = f.tell()
        return new_lines, new_pos
    except:
        return [], last_pos


class KairosMonitor:

    def __init__(self):
        self.last_pos           = 0
        self.last_log_time      = datetime.now()
        self.consecutive_losses = 0
        self.daily_pnl          = 0.0
        self.start_balance      = 0.0
        self.current_balance    = 0.0
        self.seen_errors        = deque(maxlen=50)
        self.bot_halted         = False
        self.current_log        = None
        self.trade_count        = 0
        self.wins               = 0
        self.losses             = 0
        self.breakevens         = 0
        self.last_update_id     = 0

        register_commands()
        send("👻 <b>Kairos Monitor стартиран!</b>\nНаблюдавам бота в реално време...\n\nНапиши / за да видиш командите.")
        print(f"[{datetime.now():%H:%M:%S}] Monitor started. Watching logs...")

    def run(self):
        while True:
            try:
                self._check_commands()
                self._check()
            except Exception as e:
                print(f"[Monitor ERROR] {e}")
            time.sleep(CHECK_INTERVAL)

    def _check_commands(self):
        """Проверява за Telegram команди (/stats)."""
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={"offset": self.last_update_id + 1, "timeout": 1},
                timeout=5
            )
            data = resp.json()
            for update in data.get("result", []):
                self.last_update_id = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                if text == "/stats":
                    self._send_stats()
                elif text == "/help":
                    send(
                        "👻 <b>Kairos — Команди</b>\n\n"
                        "/stats — Дневна статистика и баланс\n"
                        "/help — Този списък"
                    )
        except:
            pass

    def _send_stats(self):
        """Праща дневна статистика."""
        total = self.wins + self.losses
        wr = (self.wins / total * 100) if total > 0 else 0
        loss_pct = (abs(self.daily_pnl) / self.start_balance * 100) if (self.start_balance > 0 and self.daily_pnl < 0) else 0
        balance_str = f"{self.current_balance:.2f}$" if self.current_balance > 0 else "N/A"

        send(
            f"👻 <b>Kairos — Дневна статистика</b>\n\n"
            f"💰 Баланс: <b>{balance_str}</b>\n"
            f"📊 Сделки днес: <b>{self.trade_count}</b>\n"
            f"✅ Победи: <b>{self.wins}</b>\n"
            f"❌ Загуби: <b>{self.losses}</b>\n"
            f"⚪ Неутрални (BE): <b>{self.breakevens}</b>\n"
            f"🎯 Win Rate: <b>{wr:.0f}%</b>\n"
            f"📈 Дневен P&L (брут): <b>{self.daily_pnl:+.2f}$</b>\n"
            f"🔻 Дневна загуба: <b>{loss_pct:.1f}%</b> / 4%\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )

    def _check(self):
        log_path = get_latest_log()
        if not log_path:
            return

        # Нов ден → нов лог файл
        if log_path != self.current_log:
            self.current_log        = log_path
            self.last_pos           = 0
            self.daily_pnl          = 0.0
            self.consecutive_losses = 0
            self.trade_count        = 0
            self.wins               = 0
            self.losses             = 0
            self.breakevens         = 0

        lines, self.last_pos = tail_file(log_path, self.last_pos)

        if lines:
            self.last_log_time = datetime.now()

        for line in lines:
            self._process_line(line.strip())

        # Проверка за замразен бот
        silence = (datetime.now() - self.last_log_time).total_seconds() / 60
        if silence >= MAX_SILENCE_MINUTES:
            send(f"🔴 <b>Kairos не отговаря!</b>\nНяма активност от {silence:.0f} минути.")
            self.last_log_time = datetime.now()  # Reset за да не спамим

    def _process_line(self, line: str):
        if not line:
            return

        # ── Грешки ───────────────────────────────────────────
        if "[ERROR]" in line:
            key = line[-80:]
            if key not in self.seen_errors:
                self.seen_errors.append(key)
                send(f"❌ <b>ГРЕШКА:</b>\n<code>{line[-200:]}</code>")

        # ── Бот спрян (дневен лимит) ──────────────────────────
        if "Трейдинг спрян" in line or "trading_halted" in line.lower():
            if not self.bot_halted:
                self.bot_halted = True
                send(f"🛑 <b>Kairos е СПРЯН!</b>\n<code>{line[-150:]}</code>")

        # ── Дневен reset → бот продължава ────────────────────
        if "Дневен риск reset" in line:
            self.bot_halted = False
            self.daily_pnl  = 0.0
            bal = re.search(r"([\d.]+) USD", line)
            if bal:
                self.start_balance   = float(bal.group(1))
                self.current_balance = float(bal.group(1))

        # ── Обновяване на баланса от дневния отчет ────────────
        if "Баланс:" in line and "USD" in line:
            bal = re.search(r"Баланс:\s+([\d.]+)", line)
            if bal:
                self.current_balance = float(bal.group(1))
                # Ако мониторът е стартиран по средата на деня и start_balance не е зададен
                if self.start_balance == 0:
                    self.start_balance = float(bal.group(1))

        # ── Нова сделка отворена (само брой, без spam) ───────
        if "MT5Connector" in line and ("✅ BUY" in line or "✅ SELL" in line):
            self.trade_count += 1

        # ── Затворена сделка ──────────────────────────────────
        if ("✅ Затворена" in line or "❌ Затворена" in line or "⚪ Затворена" in line):
            gross_m = re.search(r"Брут: ([+-]?[\d.]+)", line)
            pips_m  = re.search(r"Пипсове: ([+-]?[\d.]+)", line)
            sym_m   = re.search(r"Затворена: (\S+)", line)

            gross = float(gross_m.group(1)) if gross_m else 0.0
            pips  = float(pips_m.group(1))  if pips_m  else 0.0
            sym   = sym_m.group(1)           if sym_m   else "?"

            # Четем изхода директно от тага в лога
            if "| [WIN]" in line:
                outcome = "WIN"
            elif "| [LOSS]" in line:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

            self.daily_pnl += gross

            if outcome == "WIN":
                self.consecutive_losses = 0
                self.wins += 1
                self.current_balance += gross
            elif outcome == "BREAKEVEN":
                self.breakevens += 1
            else:  # LOSS
                self.consecutive_losses += 1
                self.losses += 1
                self.current_balance += gross
                send(f"❌ <b>{sym} LOSS</b> | {pips:+.1f}p | {gross:+.2f}$ | Дневен: {self.daily_pnl:+.2f}$")

                if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                    send(f"⚠️ <b>{self.consecutive_losses} поредни загуби!</b>\nОбмисли дали да спреш бота.")

            # Дневна загуба предупреждение
            if self.start_balance > 0 and self.daily_pnl < 0:
                loss_pct = abs(self.daily_pnl) / self.start_balance * 100
                if loss_pct >= DAILY_LOSS_WARN_PCT:
                    send(f"🟡 <b>Дневна загуба {loss_pct:.1f}%</b>\n"
                         f"Лимит: 4% = {self.start_balance * 0.04:.2f}$")


if __name__ == "__main__":
    monitor = KairosMonitor()
    monitor.run()
