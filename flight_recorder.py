"""
flight_recorder.py — Чёрный Ящик v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Сохраняет ПОЛНЫЙ снапшот состояния системы в момент
каждого решения. Позволяет детерминированно воспроизвести
любую сделку через replay().

Что сохраняется:
  - Точный слепок OHLCV по всем TF (последние N свечей)
  - Значения всех признаков в момент решения
  - Голоса каждого агента + промпты
  - Метрики риска
  - Спред, объём, сессия
  - Версия конфига
  - Качество данных (от DataSteward)
  - Версия кода (git hash или timestamp)

Без этого Аудитор анализирует иллюзию, а не реальность.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Optional
import gzip

FLIGHT_DIR     = "logs/flight_recorder"
MAX_RECORDS    = 10_000   # старые сжимаются в .gz
SNAPSHOT_CANDLES = 50     # свечей сохраняем на каждом TF


class FlightRecorder:

    def __init__(self):
        os.makedirs(FLIGHT_DIR, exist_ok=True)
        self._record_count = self._count_records()

    # ─────────────────────────────────────────
    #  ЗАПИСЬ СНАПШОТА
    # ─────────────────────────────────────────

    def record_decision(
        self,
        decision_id: str,
        signal: str,
        confidence: int,

        # Данные рынка
        tf_candles: dict,       # {"M15": list_of_dicts, "H1": ...}
        current_price: float,
        spread: float = 0,
        session: str = "",

        # Признаки в момент решения (point-in-time)
        features: dict = None,  # {"rsi": 65.2, "atr": 0.0012, "pgi": 0.71, ...}

        # Агенты
        agent_votes: list = None,  # [{"name": "IMPULSE", "signal": "BUY", ...}]

        # Риск
        risk_params: dict = None,  # {"lot": 0.01, "sl": ..., "tp": ...}

        # Качество данных
        data_quality: dict = None,

        # Контекст
        alpha_patterns_used: list = None,
        custom_signals: dict = None,
        news_blocked: bool = False,
        arbiter_score: float = 0.0,
        machine_state: str = "live",

        # Конфиг-версия
        config_hash: str = "",
    ) -> str:
        """
        Записывает полный снапшот. Возвращает decision_id.
        """
        snapshot = {
            # ── Идентификация ──────────────────
            "decision_id":    decision_id,
            "timestamp":      datetime.now().isoformat(),
            "machine_state":  machine_state,
            "config_hash":    config_hash or self._get_config_hash(),

            # ── Решение ───────────────────────
            "signal":         signal,
            "confidence":     confidence,
            "arbiter_score":  arbiter_score,
            "news_blocked":   news_blocked,

            # ── Рынок ─────────────────────────
            "current_price":  current_price,
            "spread":         spread,
            "session":        session,

            # ── Свечи (point-in-time snapshot) ─
            "candles": self._serialize_candles(tf_candles),

            # ── Признаки (point-in-time) ───────
            "features": features or {},

            # ── Агенты ─────────────────────────
            "agent_votes": agent_votes or [],

            # ── Риск ──────────────────────────
            "risk_params": risk_params or {},

            # ── Качество данных ────────────────
            "data_quality": {
                "score":     data_quality.get("quality_score", 1.0) if data_quality else 1.0,
                "tradeable": data_quality.get("tradeable", True) if data_quality else True,
                "issues":    len(data_quality.get("issues", [])) if data_quality else 0,
            },

            # ── Alpha Patterns ─────────────────
            "alpha_patterns_used": alpha_patterns_used or [],

            # ── Custom Signals ─────────────────
            "custom_signals": custom_signals or {},

            # ── Исход (заполняется позже) ──────
            "outcome": None,   # "WIN" / "LOSS" / "HOLD"
            "profit":  None,
            "mfe":     None,
            "mae":     None,
            "tte":     None,
            "audited": False,
        }

        self._write(decision_id, snapshot)
        self._record_count += 1
        if self._record_count % 1000 == 0:
            self._archive_old()

        return decision_id

    def update_outcome(
        self,
        decision_id: str,
        outcome: str,
        profit: float,
        mfe: float = 0,
        mae: float = 0,
        tte: int   = 0,
    ):
        """Дополняет запись результатом после закрытия сделки."""
        path = self._path(decision_id)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            snap["outcome"] = outcome
            snap["profit"]  = profit
            snap["mfe"]     = mfe
            snap["mae"]     = mae
            snap["tte"]     = tte
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[FlightRecorder] update_outcome error: {e}")

    def mark_audited(self, decision_id: str, audit_summary: str):
        """Отмечает запись как аудированную."""
        path = self._path(decision_id)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            snap["audited"]       = True
            snap["audit_summary"] = audit_summary[:500]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ─────────────────────────────────────────
    #  REPLAY
    # ─────────────────────────────────────────

    def replay(self, decision_id: str) -> Optional[dict]:
        """
        Загружает полный снапшот по ID.
        Позволяет детерминированно воспроизвести любое решение.
        """
        path = self._path(decision_id)
        if not os.path.exists(path):
            # Попробуем в архивах
            return self._search_archive(decision_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def replay_summary(self, decision_id: str) -> str:
        """Читаемый текст для ИИ-аудитора."""
        snap = self.replay(decision_id)
        if not snap:
            return f"Снапшот {decision_id} не найден."

        votes_text = ""
        for v in snap.get("agent_votes", []):
            votes_text += (
                f"\n    {v.get('name','?')}: {v.get('signal','?')} "
                f"(уверенность {v.get('confidence',0)}/10) — {str(v.get('reasoning',''))[:80]}"
            )

        features = snap.get("features", {})
        feat_text = " | ".join(
            f"{k}:{v:.3f}" if isinstance(v, float) else f"{k}:{v}"
            for k, v in features.items()
        )

        patterns = snap.get("alpha_patterns_used", [])
        pat_text  = ", ".join(str(p) for p in patterns[:3]) if patterns else "нет"

        return f"""
=== FLIGHT RECORDER: {decision_id} ===
Время:     {snap.get('timestamp', '?')}
Решение:   {snap.get('signal', '?')} | Уверенность: {snap.get('confidence', 0)}/10
Арбитр:    {snap.get('arbiter_score', 0):.3f} | Состояние: {snap.get('machine_state', '?')}
Цена:      {snap.get('current_price', 0)} | Спред: {snap.get('spread', 0)} | Сессия: {snap.get('session', '?')}
Данные:    quality={snap['data_quality'].get('score', '?'):.0%} tradeable={snap['data_quality'].get('tradeable', '?')}
Признаки:  {feat_text or 'нет данных'}
Паттерны:  {pat_text}
Голоса агентов:{votes_text or ' нет данных'}
Исход:     {snap.get('outcome', 'ещё не закрыта')} | P/L: {snap.get('profit', '?')} | MFE: {snap.get('mfe', '?')}p MAE: {snap.get('mae', '?')}p
Аудит:     {'✅' if snap.get('audited') else '⏳ ожидает'}
""".strip()

    # ─────────────────────────────────────────
    #  ПОИСК И СТАТИСТИКА
    # ─────────────────────────────────────────

    def get_recent(self, n: int = 10) -> list:
        """Возвращает N последних снапшотов."""
        files = sorted(
            [f for f in os.listdir(FLIGHT_DIR) if f.endswith(".json")],
            reverse=True
        )[:n]
        result = []
        for fname in files:
            try:
                with open(os.path.join(FLIGHT_DIR, fname), "r", encoding="utf-8") as f:
                    result.append(json.load(f))
            except Exception:
                continue
        return result

    def get_stats(self) -> str:
        files   = [f for f in os.listdir(FLIGHT_DIR) if f.endswith(".json")]
        gz_files = [f for f in os.listdir(FLIGHT_DIR) if f.endswith(".gz")]
        total   = len(files) + len(gz_files) * 100  # приблизительно
        with_outcome = sum(
            1 for f in files[:500]
            if self._quick_check_outcome(os.path.join(FLIGHT_DIR, f))
        )
        return (
            f"FlightRecorder: {len(files)} активных снапшотов | "
            f"{len(gz_files)} архивов | {with_outcome}/{len(files)} с исходами"
        )

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────

    def _serialize_candles(self, tf_candles: dict) -> dict:
        """
        Сериализует свечи в JSON-совместимый формат.
        Конвертирует pd.Timestamp, numpy types и т.д.
        """
        result = {}
        for tf_name, candles in tf_candles.items():
            if candles is None:
                result[tf_name] = []
                continue
            if hasattr(candles, "tail"):
                # DataFrame
                rows = candles.tail(SNAPSHOT_CANDLES).to_dict("records")
            elif isinstance(candles, list):
                rows = candles[-SNAPSHOT_CANDLES:]
            else:
                rows = []
            result[tf_name] = [self._clean_row(r) for r in rows]
        return result

    @staticmethod
    def _clean_row(row: dict) -> dict:
        """Конвертирует нативные типы pandas/numpy в JSON-совместимые."""
        import pandas as pd
        import numpy as np
        clean = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, float) and (v != v):  # NaN check
                clean[k] = None
            else:
                clean[k] = v
        return clean

    def _path(self, decision_id: str) -> str:
        safe_id = decision_id.replace(":", "").replace(" ", "_")[:50]
        return os.path.join(FLIGHT_DIR, f"{safe_id}.json")

    def _write(self, decision_id: str, snapshot: dict):
        try:
            path = self._path(decision_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[FlightRecorder] Write error: {e}")

    def _count_records(self) -> int:
        try:
            return len([f for f in os.listdir(FLIGHT_DIR) if f.endswith(".json")])
        except Exception:
            return 0

    def _archive_old(self):
        """Сжимает старые записи в .gz для экономии места."""
        files = sorted(
            [f for f in os.listdir(FLIGHT_DIR) if f.endswith(".json")]
        )
        if len(files) <= MAX_RECORDS:
            return
        to_archive = files[:len(files) - MAX_RECORDS]
        for fname in to_archive:
            fpath = os.path.join(FLIGHT_DIR, fname)
            try:
                with open(fpath, "rb") as f_in:
                    with gzip.open(fpath + ".gz", "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(fpath)
            except Exception:
                pass

    def _search_archive(self, decision_id: str) -> Optional[dict]:
        safe_id = decision_id.replace(":", "").replace(" ", "_")[:50]
        gz_path = os.path.join(FLIGHT_DIR, f"{safe_id}.json.gz")
        if not os.path.exists(gz_path):
            return None
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _get_config_hash() -> str:
        try:
            with open("config.py", "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        except Exception:
            return "unknown"

    @staticmethod
    def _quick_check_outcome(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("outcome") is not None
        except Exception:
            return False


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

flight_recorder = FlightRecorder()
