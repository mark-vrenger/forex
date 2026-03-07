"""
machine_state.py — Контроллер Состояний v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Состояния системы:
  WARMUP     — прогрев после запуска, торговля запрещена
  LIVE       — нормальная работа
  DEGRADED   — проблемы с данными или риском, торговля ограничена
  SAFE_MODE  — серия убытков или drawdown, MIN_CONFIDENCE повышен
  PAPER      — shadow-режим, сигналы логируются но не исполняются
  KILL       — аварийная остановка, вернуться можно только вручную

Автоматические переходы:
  WARMUP     → LIVE         при накоплении N баров ATR/VWAP
  LIVE       → DEGRADED     при data quality < 0.6
  LIVE       → SAFE_MODE    при 3+ убытках подряд / drawdown > 5%
  LIVE       → KILL         при drawdown > MAX_DAILY_LOSS
  SAFE_MODE  → LIVE         при 2 профитных сделках / восстановлении
  KILL       → ручное       только через reset()
  любое      → PAPER        через set_paper()
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

STATE_FILE = "logs/machine_state.json"

# ── Конфиг переходов ────────────────────────
WARMUP_BARS_REQUIRED  = 50    # баров M15 до выхода из WARMUP
SAFE_LOSS_STREAK      = 3     # убытков подряд → SAFE_MODE
SAFE_DRAWDOWN_PCT     = 5.0   # % просадки от пика → SAFE_MODE
KILL_DAILY_LOSS       = 400   # $ потерь за день → KILL
SAFE_RECOVERY_WINS    = 2     # побед подряд → обратно в LIVE
DEGRADED_QUALITY      = 0.60  # quality_score ниже → DEGRADED
SAFE_MIN_CONFIDENCE   = 8     # в SAFE_MODE минимальная уверенность
NORMAL_MIN_CONFIDENCE = 7     # в LIVE минимальная уверенность


@dataclass
class MachineStateSnapshot:
    state:             str      = "WARMUP"
    entered_at:        str      = ""
    reason:            str      = ""
    bars_since_start:  int      = 0
    current_streak:    int      = 0   # + победы, - убытки
    peak_balance:      float    = 0.0
    current_balance:   float    = 0.0
    daily_loss:        float    = 0.0
    last_loss_date:    str      = ""
    total_transitions: int      = 0
    paper_mode:        bool     = False
    manual_override:   bool     = False


class MachineStateController:

    VALID_STATES = {"WARMUP", "LIVE", "DEGRADED", "SAFE_MODE", "PAPER", "KILL"}

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.snap = self._load()
        if not self.snap.entered_at:
            self.snap.entered_at = datetime.now().isoformat()
            self._save()

    # ─────────────────────────────────────────
    #  ГЛАВНОЕ API
    # ─────────────────────────────────────────

    @property
    def state(self) -> str:
        return self.snap.state

    @property
    def can_trade(self) -> bool:
        """Можно ли открывать позиции."""
        if self.snap.paper_mode:
            return False
        return self.snap.state in ("LIVE", "SAFE_MODE")

    @property
    def min_confidence(self) -> int:
        """Минимальная уверенность для текущего состояния."""
        if self.snap.state == "SAFE_MODE":
            return SAFE_MIN_CONFIDENCE
        if self.snap.state == "DEGRADED":
            return SAFE_MIN_CONFIDENCE + 1
        return NORMAL_MIN_CONFIDENCE

    def tick(
        self,
        balance:        float,
        last_result:    Optional[str] = None,   # "WIN" / "LOSS" / None
        data_quality:   float = 1.0,
        bars_added:     int   = 1,
    ) -> dict:
        """
        Вызывается на каждом баре/итерации.
        Обновляет состояние и возвращает статус.

        Возвращает:
        {
          "state":          str,
          "can_trade":      bool,
          "min_confidence": int,
          "changed":        bool,
          "reason":         str,
        }
        """
        prev_state = self.snap.state
        self.snap.bars_since_start += bars_added
        self.snap.current_balance   = balance

        if balance > self.snap.peak_balance:
            self.snap.peak_balance = balance

        # Обновляем серию побед/убытков
        if last_result == "WIN":
            self.snap.current_streak = max(0, self.snap.current_streak) + 1
        elif last_result == "LOSS":
            self.snap.current_streak = min(0, self.snap.current_streak) - 1
            # Дневной убыток
            today = datetime.now().strftime("%Y-%m-%d")
            if self.snap.last_loss_date != today:
                self.snap.daily_loss    = 0.0
                self.snap.last_loss_date = today
            self.snap.daily_loss += balance - self.snap.peak_balance

        # Обновляем дневной убыток
        today = datetime.now().strftime("%Y-%m-%d")
        if self.snap.last_loss_date != today:
            self.snap.daily_loss = 0.0

        reason = ""

        # ── Автоматические переходы ─────────────
        if not self.snap.manual_override:
            reason = self._auto_transition(balance, data_quality)

        changed = (self.snap.state != prev_state)
        if changed:
            self.snap.entered_at       = datetime.now().isoformat()
            self.snap.total_transitions += 1
            print(f"[MachineState] {prev_state} → {self.snap.state} | {reason}")

        self._save()

        return {
            "state":          self.snap.state,
            "can_trade":      self.can_trade,
            "min_confidence": self.min_confidence,
            "changed":        changed,
            "reason":         reason,
            "paper_mode":     self.snap.paper_mode,
        }

    def _auto_transition(self, balance: float, data_quality: float) -> str:
        state = self.snap.state

        # ── WARMUP → LIVE ────────────────────────
        if state == "WARMUP":
            if self.snap.bars_since_start >= WARMUP_BARS_REQUIRED:
                self.snap.state  = "LIVE"
                return f"Прогрев завершён ({self.snap.bars_since_start} баров)"
            return ""

        # ── KILL: выход только вручную ───────────
        if state == "KILL":
            return ""

        # ── Проверка дневного лимита убытков ─────
        if abs(self.snap.daily_loss) >= KILL_DAILY_LOSS:
            if state != "KILL":
                self.snap.state = "KILL"
                return f"Дневной лимит убытков ${KILL_DAILY_LOSS} достигнут"

        # ── Drawdown к пику ──────────────────────
        if self.snap.peak_balance > 0:
            dd_pct = (self.snap.peak_balance - balance) / self.snap.peak_balance * 100
            if dd_pct >= SAFE_DRAWDOWN_PCT and state not in ("SAFE_MODE", "DEGRADED"):
                self.snap.state  = "SAFE_MODE"
                return f"Просадка {dd_pct:.1f}% от пика"

        # ── Серия убытков ────────────────────────
        if self.snap.current_streak <= -SAFE_LOSS_STREAK and state == "LIVE":
            self.snap.state = "SAFE_MODE"
            return f"{abs(self.snap.current_streak)} убытков подряд"

        # ── Качество данных ──────────────────────
        if data_quality < DEGRADED_QUALITY and state not in ("DEGRADED", "KILL"):
            self.snap.state = "DEGRADED"
            return f"Качество данных {data_quality:.0%} < {DEGRADED_QUALITY:.0%}"
        elif data_quality >= DEGRADED_QUALITY and state == "DEGRADED":
            self.snap.state = "LIVE"
            return f"Качество данных восстановлено {data_quality:.0%}"

        # ── Восстановление SAFE_MODE → LIVE ──────
        if state == "SAFE_MODE" and self.snap.current_streak >= SAFE_RECOVERY_WINS:
            self.snap.state = "LIVE"
            return f"{SAFE_RECOVERY_WINS} побед подряд — возврат в LIVE"

        return ""

    # ─────────────────────────────────────────
    #  РУЧНОЕ УПРАВЛЕНИЕ
    # ─────────────────────────────────────────

    def set_state(self, new_state: str, reason: str = "Ручное управление"):
        assert new_state in self.VALID_STATES, f"Неверное состояние: {new_state}"
        old = self.snap.state
        self.snap.state          = new_state
        self.snap.entered_at     = datetime.now().isoformat()
        self.snap.reason         = reason
        self.snap.manual_override = True
        self.snap.total_transitions += 1
        self._save()
        print(f"[MachineState] РУЧНОЙ ПЕРЕХОД: {old} → {new_state} | {reason}")

    def set_paper(self, enabled: bool = True):
        self.snap.paper_mode = enabled
        self._save()
        print(f"[MachineState] Paper mode: {'ON' if enabled else 'OFF'}")

    def reset_kill(self, reason: str = ""):
        """Ручной выход из KILL. Требует явного вызова."""
        if self.snap.state != "KILL":
            print("[MachineState] Не в KILL состоянии.")
            return
        self.snap.state           = "SAFE_MODE"
        self.snap.manual_override = False
        self.snap.entered_at      = datetime.now().isoformat()
        self.snap.reason          = reason or "Ручной сброс KILL"
        self.snap.daily_loss      = 0.0
        self.snap.current_streak  = 0
        self._save()
        print(f"[MachineState] KILL сброшен → SAFE_MODE | {self.snap.reason}")

    def reset_manual_override(self):
        self.snap.manual_override = False
        self._save()

    # ─────────────────────────────────────────
    #  СТАТУС ДЛЯ ПРОМПТА
    # ─────────────────────────────────────────

    def get_status_text(self) -> str:
        s   = self.snap
        dd  = 0.0
        if s.peak_balance > 0:
            dd = (s.peak_balance - s.current_balance) / s.peak_balance * 100

        state_emoji = {
            "WARMUP":    "🔄",
            "LIVE":      "✅",
            "DEGRADED":  "⚠️",
            "SAFE_MODE": "🔶",
            "PAPER":     "📄",
            "KILL":      "🛑",
        }.get(s.state, "❓")

        lines = [
            f"=== MACHINE STATE: {state_emoji} {s.state} ===",
            f"  Баланс: ${s.current_balance:.1f} | Пик: ${s.peak_balance:.1f} | Просадка: {dd:.1f}%",
            f"  Серия: {s.current_streak:+d} | Дневной убыток: ${abs(s.daily_loss):.1f}",
            f"  Баров обработано: {s.bars_since_start} | Переходов: {s.total_transitions}",
            f"  MIN_CONFIDENCE: {self.min_confidence} | Торговля: {'разрешена' if self.can_trade else 'ЗАПРЕЩЕНА'}",
        ]
        if s.paper_mode:
            lines.append("  📄 PAPER MODE: сигналы логируются, не исполняются")
        if s.manual_override:
            lines.append("  🔧 Ручное управление активно")
        return "\n".join(lines)

    # ─────────────────────────────────────────
    #  PERSISTENCE
    # ─────────────────────────────────────────

    def _load(self) -> MachineStateSnapshot:
        if not os.path.exists(STATE_FILE):
            return MachineStateSnapshot()
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            snap = MachineStateSnapshot(**data)
            # Если была KILL при предыдущем запуске — сохраняем
            return snap
        except Exception:
            return MachineStateSnapshot()

    def _save(self):
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self.snap), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MachineState] Save error: {e}")


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

machine = MachineStateController()
