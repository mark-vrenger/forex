"""
auditor_agent.py — ИИ-Аудитор сделок v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Запускается ПОСЛЕ закрытия каждой сделки.
Анализирует MFE / MAE / TTE и качество входа.
Синтезирует паттерны в alpha_patterns.json.
"""

import os
import json
import re
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from config import GPT_TIMEOUT, GPT_MAX_RETRIES

load_dotenv()

AUDITOR_MODEL   = "deepseek/deepseek-v3.2"
ALPHA_FILE      = "knowledge/alpha_patterns.json"
AUDIT_LOG_FILE  = "logs/audit_log.jsonl"


# ─────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────

def _make_llm():
    return ChatOpenAI(
        model=AUDITOR_MODEL,
        temperature=0.2,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        request_timeout=GPT_TIMEOUT,
        max_retries=GPT_MAX_RETRIES,
    )


# ─────────────────────────────────────────
#  CORE AUDIT
# ─────────────────────────────────────────

class AuditorAgent:
    """
    Агент-Аудитор.
    Вызов: auditor.audit_trade(trade_dict, candles_during_trade)
    """

    def __init__(self):
        self.llm = _make_llm()
        os.makedirs("knowledge", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        self._ensure_alpha_file()

    # ── PUBLIC ──────────────────────────────

    def audit_trade(
        self,
        trade: dict,
        candles_during_trade: Optional[list] = None,
    ) -> dict:
        """
        Полный аудит одной закрытой сделки.

        trade — словарь из trade_memory / database
        candles_during_trade — список словарей OHLCV (опционально)

        Возвращает словарь audit_result.
        """

        # ── 1. Вычислить MFE / MAE / TTE ───────
        metrics = self._calc_excursion_metrics(trade, candles_during_trade)

        # ── 2. Классифицировать сделку ──────────
        trade_class = self._classify_trade(trade, metrics)

        # ── 3. Вызвать ИИ для глубокого анализа ─
        ai_verdict = self._call_ai(trade, metrics, trade_class)

        # ── 4. Сохранить в лог ──────────────────
        audit_result = {
            "trade_id":    trade.get("id"),
            "time":        datetime.now().isoformat(),
            "direction":   trade.get("direction"),
            "result":      trade.get("result"),
            "profit":      trade.get("profit", 0),
            "mfe_points":  metrics["mfe_points"],
            "mae_points":  metrics["mae_points"],
            "tte_candles": metrics["tte_candles"],
            "trade_class": trade_class,
            "efficiency":  metrics["efficiency"],
            "noise_ratio": metrics["noise_ratio"],
            "ai_verdict":  ai_verdict,
            "entry_hour":  trade.get("entry_hour"),
            "entry_dow":   trade.get("entry_dow"),
            "session":     trade.get("session"),
        }

        self._save_audit(audit_result)

        # ── 5. Обновить alpha_patterns при идеальном входе ─
        if trade_class in ("PERFECT_WIN", "EFFICIENT_WIN"):
            self._update_alpha_patterns(trade, metrics, ai_verdict)

        return audit_result

    def get_audit_summary(self, last_n: int = 20) -> str:
        """Сводка последних N аудитов для промпта Консилиума."""
        records = self._load_audit_log(last_n)
        if not records:
            return "Audit: No data yet."

        perfect  = [r for r in records if r.get("trade_class") == "PERFECT_WIN"]
        noisy_sl = [r for r in records if r.get("trade_class") == "NOISE_STOP"]
        wasted   = [r for r in records if r.get("trade_class") == "WASTED_WIN"]
        avg_eff  = sum(r.get("efficiency", 0) for r in records) / len(records)

        lines = [
            f"=== LAST {len(records)} AUDITS ===",
            f"  Perfect entries:    {len(perfect)}",
            f"  Wasted wins (early TP): {len(wasted)}",
            f"  Noise stops:        {len(noisy_sl)}",
            f"  Avg entry efficiency: {avg_eff:.1f}%",
        ]

        if wasted:
            avg_wasted_mfe = sum(r.get("mfe_points", 0) for r in wasted) / len(wasted)
            lines.append(f"  Avg MFE on wasted wins: {avg_wasted_mfe:.1f}p (TP was too tight)")

        return "\n".join(lines)

    # ── METRICS ─────────────────────────────

    def _calc_excursion_metrics(self, trade: dict, candles: Optional[list]) -> dict:
        """
        Вычисляет MFE, MAE, TTE и производные метрики.
        Если candles не переданы — использует данные из полей trade.
        """
        direction   = trade.get("direction", "BUY")
        entry_price = trade.get("entry_price", 0)
        sl          = trade.get("sl", 0)
        tp          = trade.get("tp", 0)
        point_size  = 0.00001

        # Из поля backtester (max_float / min_float)
        if candles is None:
            mfe_money = trade.get("max_float", trade.get("mfe_money", 0)) or 0
            mae_money = abs(trade.get("min_float", trade.get("mae_money", 0)) or 0)
            tte_candles = trade.get("tte_candles", trade.get("duration_minutes", 0)) or 0

            lot        = trade.get("lot", 0.01)
            multiplier = (lot * 100_000 * point_size) or 0.00001
            mfe_points = mfe_money / multiplier if multiplier else 0
            mae_points = mae_money / multiplier if multiplier else 0
        else:
            mfe_points = 0.0
            mae_points = 0.0
            tte_candles = len(candles)

            for c in candles:
                high  = c.get("high", entry_price)
                low   = c.get("low",  entry_price)
                if direction == "BUY":
                    mfe_points = max(mfe_points, (high - entry_price) / point_size)
                    mae_points = max(mae_points, (entry_price - low)  / point_size)
                else:
                    mfe_points = max(mfe_points, (entry_price - low)  / point_size)
                    mae_points = max(mae_points, (high - entry_price) / point_size)

        # Целевые дистанции
        tp_dist = abs(tp - entry_price) / point_size if tp and entry_price else 1
        sl_dist = abs(sl - entry_price) / point_size if sl and entry_price else 1

        # Эффективность входа: сколько % от MFE забрали через TP
        efficiency = min(100.0, (tp_dist / mfe_points * 100)) if mfe_points > 0 else 0

        # Шум: насколько MAE велик относительно SL
        noise_ratio = mae_points / sl_dist if sl_dist > 0 else 0

        return {
            "mfe_points":  round(mfe_points,  1),
            "mae_points":  round(mae_points,   1),
            "tte_candles": int(tte_candles),
            "efficiency":  round(efficiency,   1),
            "noise_ratio": round(noise_ratio,  2),
            "tp_dist":     round(tp_dist,      1),
            "sl_dist":     round(sl_dist,      1),
        }

    def _classify_trade(self, trade: dict, metrics: dict) -> str:
        """
        Классифицирует сделку по 6 типам.
        """
        result      = trade.get("result", "")
        mfe         = metrics["mfe_points"]
        mae         = metrics["mae_points"]
        efficiency  = metrics["efficiency"]
        noise_ratio = metrics["noise_ratio"]
        sl_dist     = metrics["sl_dist"]
        tp_dist     = metrics["tp_dist"]

        if result == "WIN":
            if mae < sl_dist * 0.15 and efficiency >= 80:
                return "PERFECT_WIN"       # Вошли точно, взяли почти весь ход
            elif mfe > tp_dist * 2.5:
                return "WASTED_WIN"        # Взяли крохи, цена ушла намного дальше
            else:
                return "EFFICIENT_WIN"     # Нормальная победа

        else:  # LOSS
            if noise_ratio > 0.7 and mfe > sl_dist * 0.5:
                return "NOISE_STOP"        # Стоп выбит шумом, цена шла в нашу сторону
            elif mae > sl_dist * 1.5:
                return "WRONG_DIRECTION"   # Цена сразу пошла против нас
            else:
                return "NORMAL_LOSS"       # Обычный стоп

    def _call_ai(self, trade: dict, metrics: dict, trade_class: str) -> str:
        """Вызывает DeepSeek V3.2 для анализа конкретной сделки."""
        prompt = f"""Ты — Аудитор торговых сделок (DeepSeek V3.2).
Твоя задача: глубокий post-trade разбор.

=== ДАННЫЕ СДЕЛКИ ===
Направление: {trade.get('direction')} | Результат: {trade.get('result')} | P/L: ${trade.get('profit', 0):.2f}
Вход: {trade.get('entry_price')} | SL: {trade.get('sl')} | TP: {trade.get('tp')}
Время входа: {trade.get('time')} | Сессия: {trade.get('session')}
Час входа: {trade.get('entry_hour', '?')} | День недели: {trade.get('entry_dow', '?')}
RSI при входе: {trade.get('rsi_at_entry', '?')} | ATR при входе: {trade.get('atr_at_entry', '?')}
Уверенность ИИ: {trade.get('confidence', '?')}/10

=== МЕТРИКИ ЭКСКУРСИИ ===
MFE (макс. в нашу сторону): {metrics['mfe_points']:.1f} пунктов
MAE (макс. против нас): {metrics['mae_points']:.1f} пунктов
TTE (свечей до закрытия): {metrics['tte_candles']}
Эффективность входа: {metrics['efficiency']:.1f}%
Noise ratio (шум / SL): {metrics['noise_ratio']:.2f}
Класс сделки: {trade_class}

=== ЛОГИКА КОНСИЛИУМА ПРИ ВХОДЕ ===
{str(trade.get('ai_logic', 'Нет данных'))[:500]}

=== ЗАДАЧА ===
1. Оцени качество входа (был ли момент оптимальным?)
2. Оцени качество выхода (правильно ли выставлен TP/SL?)
3. Найди главную причину победы/поражения
4. Сформулируй ОДНО конкретное правило улучшения для будущих сделок
5. Если это NOISE_STOP: предложи, как можно было поставить SL шире/умнее
6. Если это WASTED_WIN: предложи новое правило трейлинга
7. Если это PERFECT_WIN: опиши точный паттерн входа (для базы знаний)

ВЫВОД:
КАЧЕСТВО_ВХОДА: (1-10)
КАЧЕСТВО_ВЫХОДА: (1-10)
ГЛАВНАЯ_ПРИЧИНА: (одна строка)
ПРАВИЛО: (одно конкретное правило, начинающееся со слова "Входить" / "Не входить" / "Выходить")
ПАТТЕРН_ДЛЯ_БАЗЫ: (только если PERFECT_WIN или EFFICIENT_WIN — JSON-описание паттерна)
"""
        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"КАЧЕСТВО_ВХОДА: 0\nКАЧЕСТВО_ВЫХОДА: 0\nГЛАВНАЯ_ПРИЧИНА: Ошибка ИИ ({e})\nПРАВИЛО: Нет\nПАТТЕРН_ДЛЯ_БАЗЫ: Нет"

    # ── ALPHA PATTERNS ──────────────────────

    def _update_alpha_patterns(self, trade: dict, metrics: dict, ai_verdict: str):
        """
        Добавляет паттерн в alpha_patterns.json.
        Только если накоплено ≥ 3 похожих паттернов — помечает как «подтверждённый».
        """
        patterns = self._load_alpha_patterns()

        # Извлекаем описание паттерна из ответа ИИ
        pattern_desc = ""
        for line in ai_verdict.split("\n"):
            if "ПАТТЕРН_ДЛЯ_БАЗЫ:" in line.upper():
                pattern_desc = line.split(":", 1)[-1].strip()
                break

        if not pattern_desc or pattern_desc.lower() in ("нет", "no", ""):
            return

        new_entry = {
            "id":           f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created":      datetime.now().isoformat(),
            "direction":    trade.get("direction"),
            "session":      trade.get("session"),
            "entry_hour":   trade.get("entry_hour"),
            "rsi_at_entry": trade.get("rsi_at_entry"),
            "atr_at_entry": trade.get("atr_at_entry"),
            "confidence":   trade.get("confidence"),
            "mfe_points":   metrics["mfe_points"],
            "mae_points":   metrics["mae_points"],
            "efficiency":   metrics["efficiency"],
            "description":  pattern_desc,
            "occurrences":  1,
            "confirmed":    False,
            "win_count":    1,
        }

        # Проверяем: есть ли уже похожий паттерн (по сессии + направлению + RSI-зоне)?
        rsi = trade.get("rsi_at_entry", 50) or 50
        merged = False
        for p in patterns:
            same_dir     = p.get("direction") == new_entry["direction"]
            same_session = p.get("session")   == new_entry["session"]
            rsi_close    = abs((p.get("rsi_at_entry") or 50) - rsi) < 10
            if same_dir and same_session and rsi_close:
                p["occurrences"] = p.get("occurrences", 1) + 1
                p["win_count"]   = p.get("win_count", 1)   + 1
                p["confirmed"]   = p["occurrences"] >= 3
                p["last_seen"]   = datetime.now().isoformat()
                merged = True
                break

        if not merged:
            patterns.append(new_entry)

        # Сохранить (не более 100 паттернов, чистим старые неподтверждённые)
        patterns = self._prune_patterns(patterns)
        self._save_alpha_patterns(patterns)

    def _prune_patterns(self, patterns: list) -> list:
        """Оставляет топ-100: сначала подтверждённые, потом новые."""
        confirmed   = [p for p in patterns if p.get("confirmed")]
        unconfirmed = [p for p in patterns if not p.get("confirmed")]
        unconfirmed.sort(key=lambda x: x.get("created", ""), reverse=True)
        return (confirmed + unconfirmed)[:100]

    # ── PERSISTENCE ─────────────────────────

    def _save_audit(self, result: dict):
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[Auditor] Save error: {e}")

    def _load_audit_log(self, last_n: int = 20) -> list:
        if not os.path.exists(AUDIT_LOG_FILE):
            return []
        records = []
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            return []
        return records[-last_n:]

    def _ensure_alpha_file(self):
        if not os.path.exists(ALPHA_FILE):
            with open(ALPHA_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_alpha_patterns(self) -> list:
        try:
            with open(ALPHA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_alpha_patterns(self, patterns: list):
        with open(ALPHA_FILE, "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)

    def get_alpha_context(self) -> str:
        """Возвращает текст подтверждённых паттернов для промпта."""
        patterns = self._load_alpha_patterns()
        confirmed = [p for p in patterns if p.get("confirmed")]
        if not confirmed:
            return "Alpha Patterns: пока нет подтверждённых паттернов (нужно ≥3 повторений)."
        lines = ["=== ALPHA PATTERNS (ПОДТВЕРЖДЁННЫЕ ПАТТЕРНЫ) ==="]
        for p in confirmed:
            lines.append(
                f"  [{p['id']}] {p['direction']} | {p['session']} | "
                f"RSI≈{p.get('rsi_at_entry', '?')} | "
                f"Встреч: {p['occurrences']} | WR: {p['win_count']}/{p['occurrences']} | "
                f"{p['description'][:120]}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

auditor = AuditorAgent()
