"""
trade_memory.py — Память сделок v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
НОВОЕ v3.0:
  - MFE / MAE / TTE поля в каждой сделке
  - Полный временной профиль: hour, dow (день недели), session
  - Анализ прибыльности по времени суток и дням недели
  - Метод get_time_profile() для ИИ-анализа
"""

import json
import os
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
from config import TRADE_HISTORY_FILE, MAGIC, SYMBOL


# ── Дни недели (для читаемости) ──────────────────
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class TradeMemory:
    def __init__(self):
        self.history_file = TRADE_HISTORY_FILE
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    # ─────────────────────────────────────────
    #  ЗАПИСЬ НОВОЙ СДЕЛКИ
    # ─────────────────────────────────────────

    def record_trade(
        self,
        direction, lot, entry_price, sl, tp,
        confidence, ai_logic, rsi, atr, session,
        spread=0, slippage=0,
    ):
        now = datetime.now()
        trade = {
            # ── Идентификаторы ───────────────
            "id":           now.strftime("%Y%m%d_%H%M%S"),
            "time":         now.isoformat(),

            # ── Временной профиль ────────────
            "entry_hour":   now.hour,                      # 0-23
            "entry_dow":    now.weekday(),                  # 0=Пн 6=Вс
            "entry_dow_name": DOW_NAMES[now.weekday()],
            "entry_date":   now.strftime("%Y-%m-%d"),

            # ── Параметры входа ──────────────
            "direction":    direction,
            "lot":          lot,
            "entry_price":  entry_price,
            "sl":           sl,
            "tp":           tp,

            # ── Контекст ────────────────────
            "confidence":   confidence,
            "ai_logic":     ai_logic,
            "rsi_at_entry": rsi,
            "atr_at_entry": atr,
            "session":      session,
            "spread_at_entry": spread,
            "slippage":     slippage,

            # ── Экскурсионные метрики ────────
            "mfe_points":   None,   # Maximum Favorable Excursion (в пунктах)
            "mae_points":   None,   # Maximum Adverse Excursion  (в пунктах)
            "tte_candles":  None,   # Time to Exit в свечах M15
            "tte_minutes":  None,   # Time to Exit в минутах

            # ── Результат ───────────────────
            "result":       None,
            "profit":       None,
            "commission":   None,
            "swap":         None,
            "close_time":   None,
            "close_reason": None,
            "close_price":  None,
            "duration_minutes": None,

            # ── Флаги аудита ─────────────────
            "audited":      False,
            "trade_class":  None,   # PERFECT_WIN / NOISE_STOP / WASTED_WIN / etc.
            "audit_rule":   None,   # Правило от аудитора
        }

        history = self._load()
        history.append(trade)
        self._save(history)
        return trade["id"]

    # ─────────────────────────────────────────
    #  ОБНОВЛЕНИЕ MFE / MAE
    # ─────────────────────────────────────────

    def update_excursion(
        self, trade_id: str,
        mfe_points: float,
        mae_points: float,
        tte_candles: int = 0,
        tte_minutes: int = 0,
    ):
        """Обновляет метрики экскурсии для открытой/закрытой сделки."""
        history = self._load()
        for t in history:
            if t["id"] == trade_id:
                t["mfe_points"]  = round(mfe_points, 1)
                t["mae_points"]  = round(mae_points, 1)
                t["tte_candles"] = int(tte_candles)
                t["tte_minutes"] = int(tte_minutes)
                break
        self._save(history)

    def update_closed_trade(self, trade_id: str, close_price: float, profit: float):
        history = self._load()
        for t in history:
            if t["id"] == trade_id:
                t["result"]      = "WIN" if profit > 0 else "LOSS"
                t["profit"]      = profit
                t["close_price"] = close_price
                t["close_time"]  = datetime.now().isoformat()
                t["close_reason"] = "TRAINING"
                # Считаем duration если есть entry time
                try:
                    open_t = datetime.fromisoformat(t["time"])
                    t["duration_minutes"] = int(
                        (datetime.now() - open_t).total_seconds() / 60
                    )
                    t["tte_minutes"] = t["duration_minutes"]
                except Exception:
                    pass
                break
        self._save(history)

    def update_audit_result(
        self, trade_id: str,
        trade_class: str, audit_rule: str,
    ):
        """Записывает результат ИИ-аудита в сделку."""
        history = self._load()
        for t in history:
            if t["id"] == trade_id:
                t["audited"]     = True
                t["trade_class"] = trade_class
                t["audit_rule"]  = audit_rule
                break
        self._save(history)

    # ─────────────────────────────────────────
    #  ВРЕМЕННОЙ ПРОФИЛЬ ДЛЯ ИИ
    # ─────────────────────────────────────────

    def get_time_profile(self, last_n: int = 100) -> str:
        """
        Возвращает текстовый отчёт о прибыльности по часам и дням.
        Передаётся в промпт Консилиума перед принятием решения.
        """
        history = self._load()
        closed = [t for t in history if t.get("result") is not None][-last_n:]
        if len(closed) < 5:
            return "Time Profile: недостаточно данных (нужно ≥5 сделок)."

        # ── По часам ─────────────────────────
        hour_stats: dict = {}
        for t in closed:
            h = t.get("entry_hour")
            if h is None:
                continue
            if h not in hour_stats:
                hour_stats[h] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t["result"] == "WIN":
                hour_stats[h]["wins"] += 1
            else:
                hour_stats[h]["losses"] += 1
            hour_stats[h]["pnl"] += t.get("profit", 0) or 0

        # ── По дням недели ───────────────────
        dow_stats: dict = {}
        for t in closed:
            d = t.get("entry_dow")
            if d is None:
                continue
            name = DOW_NAMES[d]
            if name not in dow_stats:
                dow_stats[name] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t["result"] == "WIN":
                dow_stats[name]["wins"] += 1
            else:
                dow_stats[name]["losses"] += 1
            dow_stats[name]["pnl"] += t.get("profit", 0) or 0

        # ── По сессиям ────────────────────────
        sess_stats: dict = {}
        for t in closed:
            s = t.get("session", "unknown")
            if s not in sess_stats:
                sess_stats[s] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t["result"] == "WIN":
                sess_stats[s]["wins"] += 1
            else:
                sess_stats[s]["losses"] += 1
            sess_stats[s]["pnl"] += t.get("profit", 0) or 0

        lines = [f"=== TEMPORAL TRADE PROFILE (последние {len(closed)} сделок) ==="]

        # Часы
        best_hours  = []
        worst_hours = []
        lines.append("\nПо часам (UTC):")
        for h in sorted(hour_stats.keys()):
            s = hour_stats[h]
            total_h = s["wins"] + s["losses"]
            wr = s["wins"] / total_h * 100 if total_h > 0 else 0
            mark = " ✅ ЛУЧШИЙ" if wr >= 65 and total_h >= 3 else \
                   " ❌ ХУДШИЙ" if wr <= 35 and total_h >= 3 else ""
            if wr >= 65 and total_h >= 3:
                best_hours.append(h)
            if wr <= 35 and total_h >= 3:
                worst_hours.append(h)
            lines.append(
                f"  {h:02d}:00  сделок:{total_h}  WR:{wr:.0f}%  P/L:${s['pnl']:.1f}{mark}"
            )

        # Дни
        lines.append("\nПо дням недели:")
        for name in DOW_NAMES:
            if name not in dow_stats:
                continue
            s = dow_stats[name]
            total_d = s["wins"] + s["losses"]
            wr = s["wins"] / total_d * 100 if total_d > 0 else 0
            mark = " ✅" if wr >= 60 and total_d >= 3 else \
                   " ❌" if wr <= 40 and total_d >= 3 else ""
            lines.append(
                f"  {name}  сделок:{total_d}  WR:{wr:.0f}%  P/L:${s['pnl']:.1f}{mark}"
            )

        # Сессии
        lines.append("\nПо торговым сессиям:")
        for name, s in sorted(sess_stats.items(), key=lambda x: -x[1]["pnl"]):
            total_s = s["wins"] + s["losses"]
            wr = s["wins"] / total_s * 100 if total_s > 0 else 0
            lines.append(
                f"  {name:<20} сделок:{total_s}  WR:{wr:.0f}%  P/L:${s['pnl']:.1f}"
            )

        # Рекомендации
        lines.append("\n>>> РЕКОМЕНДАЦИИ ПО ВРЕМЕНИ:")
        if best_hours:
            lines.append(
                f"  ЛУЧШЕЕ время входа: {', '.join(f'{h:02d}:xx' for h in best_hours)}"
            )
        if worst_hours:
            lines.append(
                f"  ИЗБЕГАТЬ часы: {', '.join(f'{h:02d}:xx' for h in worst_hours)}"
            )

        # Текущий час — опасен?
        now_h = datetime.now().hour
        if now_h in worst_hours:
            lines.append(
                f"  ⚠️ ТЕКУЩИЙ ЧАС {now_h:02d}:xx — в зоне риска! Повышен MIN_CONFIDENCE."
            )
        elif now_h in best_hours:
            lines.append(
                f"  ✅ ТЕКУЩИЙ ЧАС {now_h:02d}:xx — исторически прибыльный."
            )

        return "\n".join(lines)

    def get_time_confidence_adjustment(self) -> int:
        """
        Возвращает поправку к уверенности на основе текущего часа.
        +1 если час исторически прибыльный, -1 если убыточный.
        """
        history  = self._load()
        closed   = [t for t in history if t.get("result") is not None]
        if len(closed) < 10:
            return 0

        now_h = datetime.now().hour
        hour_trades = [t for t in closed if t.get("entry_hour") == now_h]
        if len(hour_trades) < 3:
            return 0

        wins = sum(1 for t in hour_trades if t["result"] == "WIN")
        wr   = wins / len(hour_trades)

        if wr >= 0.65:
            return +1
        elif wr <= 0.35:
            return -1
        return 0

    # ─────────────────────────────────────────
    #  СВОДКА ДЛЯ ПРОМПТА (оригинальный метод + время)
    # ─────────────────────────────────────────

    def get_performance_summary(self, last_n: int = 20) -> str:
        history = self._load()
        closed  = [t for t in history if t.get("result") is not None]
        if not closed:
            return "No trade history. Be very cautious with first trades."

        recent = closed[-last_n:]
        wins   = [t for t in recent if t["result"] == "WIN"]
        losses = [t for t in recent if t["result"] == "LOSS"]
        total  = len(recent)

        total_profit  = sum(t.get("profit", 0) for t in recent if t.get("profit"))
        avg_win       = sum(t.get("profit", 0) for t in wins)  / len(wins)   if wins   else 0
        avg_loss      = sum(t.get("profit", 0) for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # MFE/MAE статистика
        mfe_list = [t.get("mfe_points") for t in recent if t.get("mfe_points") is not None]
        mae_list = [t.get("mae_points") for t in recent if t.get("mae_points") is not None]
        avg_mfe  = sum(mfe_list) / len(mfe_list) if mfe_list else 0
        avg_mae  = sum(mae_list) / len(mae_list) if mae_list else 0

        # Серия проигрышей
        losing_streak = current_streak = 0
        for t in recent:
            if t["result"] == "LOSS":
                current_streak += 1
                losing_streak = max(losing_streak, current_streak)
            else:
                current_streak = 0

        summary = (
            f"=== LAST {total} TRADES ===\n"
            f"Wins: {len(wins)} | Losses: {len(losses)} | "
            f"Winrate: {len(wins)/total*100:.0f}%\n"
            f"Net P/L: ${total_profit:.2f} | Profit Factor: {profit_factor:.2f}\n"
            f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}\n"
            f"Max losing streak: {losing_streak} | Current: {current_streak}\n"
            f"Avg MFE: {avg_mfe:.1f}p | Avg MAE: {avg_mae:.1f}p\n"
        )

        # Предупреждения
        warnings = []
        if current_streak >= 3:
            warnings.append("CRITICAL: 3+ losses — только уверенность 9+!")
        if current_streak >= 5:
            warnings.append("EMERGENCY: 5+ losses! Рекомендую остановить торговлю.")
        if profit_factor < 1.0 and total > 5:
            warnings.append(f"Profit factor {profit_factor:.2f} < 1.0 — система теряет деньги!")

        # Временная поправка
        time_adj = self.get_time_confidence_adjustment()
        if time_adj == +1:
            warnings.append("✅ Текущий час — исторически прибыльный.")
        elif time_adj == -1:
            warnings.append("⚠️ Текущий час — исторически убыточный! Повышаем планку.")

        if warnings:
            summary += "WARNINGS:\n" + "\n".join(warnings)
        else:
            summary += "Status: Normal"

        return summary

    def get_last_trades_for_prompt(self, n: int = 5) -> str:
        history = self._load()
        closed  = [t for t in history if t.get("result") is not None][-n:]
        if not closed:
            return "No completed trades."
        lines = []
        for t in closed:
            mfe  = t.get("mfe_points", "?")
            mae  = t.get("mae_points", "?")
            hour = t.get("entry_hour", "?")
            lines.append(
                f"  {t['direction']} | {t['result']} | "
                f"P/L:${t.get('profit', 0):.2f} | Conf:{t.get('confidence', 0)} | "
                f"RSI:{t.get('rsi_at_entry', 0):.0f} | "
                f"Hour:{hour} | Session:{t.get('session', '?')} | "
                f"MFE:{mfe}p MAE:{mae}p | Class:{t.get('trade_class', '-')}"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────
    #  СИНХРОНИЗАЦИЯ С MT5
    # ─────────────────────────────────────────

    def sync_closed_trades(self):
        history    = self._load()
        open_trades = [t for t in history if t.get("result") is None]
        if not open_trades:
            return
        week_ago = datetime.now() - timedelta(days=7)
        deals    = mt5.history_deals_get(week_ago, datetime.now())
        if not deals:
            return
        changed = False
        for trade in open_trades:
            for deal in deals:
                if deal.magic == MAGIC and deal.symbol == SYMBOL and deal.entry == 1:
                    trade["result"]       = "WIN" if deal.profit > 0 else "LOSS"
                    trade["profit"]       = deal.profit
                    trade["commission"]   = deal.commission
                    trade["swap"]         = deal.swap
                    trade["close_time"]   = datetime.now().isoformat()
                    trade["close_reason"] = "TP/SL"
                    try:
                        open_t = datetime.fromisoformat(trade["time"])
                        dur    = int((datetime.now() - open_t).total_seconds() / 60)
                        trade["duration_minutes"] = dur
                        trade["tte_minutes"]      = dur
                    except Exception:
                        pass
                    changed = True
        if changed:
            self._save(history)

    def get_daily_realized_loss(self) -> float:
        history = self._load()
        today   = datetime.now().strftime("%Y-%m-%d")
        today_closed = [
            t for t in history
            if t.get("result") is not None
            and t.get("close_time", "").startswith(today)
        ]
        return sum(t.get("profit", 0) for t in today_closed if t.get("profit", 0) < 0)

    # ─────────────────────────────────────────
    #  PRIVATE
    # ─────────────────────────────────────────

    def _load(self) -> list:
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, data: list):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

memory = TradeMemory()
