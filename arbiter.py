"""
arbiter.py — Агент-Арбитр (Meta-Policy) v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Не ищет сигнал сам. Принимает структурированные голоса агентов
и вычисляет финальный вердикт по формуле:

  final_score = Σ(agent_weight × confidence × quality_factor)
              - penalty_disagreement
              - penalty_stale_data
              - penalty_event_risk
              - penalty_degraded_state

"NO TRADE" — полноценный класс решения, а не отсутствие совпадения.

Каждый агент должен передавать типизированный контракт:
{
  "name":              str,
  "signal":            "BUY" | "SELL" | "WAIT",
  "confidence":        int (1-10),
  "horizon":           "SCALP" | "INTRADAY" | "SWING",
  "invalidation_level": float,      ← при каком цене идея сломана
  "expected_move_atr": float,       ← ожидаемое движение в ATR
  "data_freshness":    float (0-1), ← насколько свежи данные агента
  "veto":              bool,        ← жёсткое вето (блокирует всё)
  "reasoning":         str,
}
"""

import json
import os
from datetime import datetime
from typing import Optional

ARBITER_LOG = "logs/arbiter_decisions.jsonl"

# ── Базовые веса агентов (настраиваются optimizer'ом) ─
DEFAULT_AGENT_WEIGHTS = {
    "IMPULSE": 0.30,
    "TREND":   0.40,
    "ANALYST": 0.30,
    "QUANT":   0.35,
    "MACRO":   0.25,
}

# ── Пороги ───────────────────────────────────────────
MIN_SCORE_TO_TRADE    = 4.5    # ниже → NO TRADE
DISAGREEMENT_PENALTY  = 0.8    # штраф за несогласие агентов
STALE_DATA_PENALTY    = 1.5    # штраф за несвежие данные
EVENT_RISK_PENALTY    = 2.0    # штраф за новостной риск
DEGRADED_PENALTY      = 1.0    # штраф за DEGRADED state
VETO_BLOCKS_ALL       = True   # любое veto = NO TRADE


class Arbiter:

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.agent_weights = dict(DEFAULT_AGENT_WEIGHTS)

    # ─────────────────────────────────────────
    #  ГЛАВНЫЙ МЕТОД
    # ─────────────────────────────────────────

    def decide(
        self,
        agent_votes:    list,        # список typed-контрактов агентов
        data_quality:   float = 1.0, # от DataSteward
        event_risk:     bool  = False,
        machine_state:  str   = "LIVE",
        current_regime: dict  = None, # {"trend_prob": 0.6, "range_prob": 0.3, ...}
        custom_signals: dict  = None, # от custom_indicators
    ) -> dict:
        """
        Возвращает финальный вердикт арбитра.
        """

        # ── 1. Проверка вето ────────────────
        for vote in agent_votes:
            if vote.get("veto") and VETO_BLOCKS_ALL:
                return self._build_result(
                    "NO_TRADE", 0.0, f"VETO от {vote.get('name', '?')}: {vote.get('reasoning', '')[:80]}",
                    agent_votes, data_quality
                )

        # ── 2. Разделяем голоса ─────────────
        buy_votes  = [v for v in agent_votes if v.get("signal") == "BUY"]
        sell_votes = [v for v in agent_votes if v.get("signal") == "SELL"]
        wait_votes = [v for v in agent_votes if v.get("signal") == "WAIT"]

        # ── 3. Считаем взвешенный score ─────
        buy_score  = self._weighted_score(buy_votes,  data_quality)
        sell_score = self._weighted_score(sell_votes, data_quality)

        # ── 4. Штрафы ───────────────────────
        penalties = 0.0
        penalty_reasons = []

        # Несогласие: если есть сильные голоса в обе стороны
        disagreement = min(buy_score, sell_score)
        if disagreement > 1.5:
            p = disagreement * DISAGREEMENT_PENALTY
            penalties += p
            penalty_reasons.append(f"Несогласие агентов -{p:.1f}")

        # Несвежие данные
        avg_freshness = self._avg_freshness(agent_votes)
        if avg_freshness < 0.7:
            p = (1.0 - avg_freshness) * STALE_DATA_PENALTY * 3
            penalties += p
            penalty_reasons.append(f"Несвежие данные (свежесть {avg_freshness:.0%}) -{p:.1f}")

        # Новостной риск
        if event_risk:
            penalties += EVENT_RISK_PENALTY
            penalty_reasons.append(f"Новостной риск -{EVENT_RISK_PENALTY}")

        # Состояние машины
        if machine_state == "DEGRADED":
            penalties += DEGRADED_PENALTY
            penalty_reasons.append(f"DEGRADED state -{DEGRADED_PENALTY}")
        elif machine_state == "SAFE_MODE":
            penalties += DEGRADED_PENALTY * 0.5
            penalty_reasons.append(f"SAFE_MODE -{DEGRADED_PENALTY*0.5}")

        # ── 5. Итоговые score ───────────────
        final_buy  = max(0.0, buy_score  - penalties)
        final_sell = max(0.0, sell_score - penalties)

        # ── 6. Бонус от Custom Indicators ──
        if custom_signals:
            ci_buy  = custom_signals.get("buy",  0) * 0.2
            ci_sell = custom_signals.get("sell", 0) * 0.2
            final_buy  += ci_buy
            final_sell += ci_sell

        # ── 7. Бонус от Regime ──────────────
        if current_regime:
            trend_p = current_regime.get("trend_prob", 0.5)
            # В трендовом рынке даём бонус
            if trend_p > 0.6 and (buy_score > sell_score or sell_score > buy_score):
                dominant = max(final_buy, final_sell)
                bonus = (trend_p - 0.5) * 2.0
                if final_buy > final_sell:
                    final_buy  += bonus
                else:
                    final_sell += bonus

        # ── 8. Решение ───────────────────────
        if final_buy > final_sell and final_buy >= MIN_SCORE_TO_TRADE:
            signal = "BUY"
            score  = final_buy
        elif final_sell > final_buy and final_sell >= MIN_SCORE_TO_TRADE:
            signal = "SELL"
            score  = final_sell
        else:
            signal = "NO_TRADE"
            score  = max(final_buy, final_sell)

        # Причина NO_TRADE
        reason = ""
        if signal == "NO_TRADE":
            if max(buy_score, sell_score) < 1.0:
                reason = "Нет значимых сигналов от агентов"
            elif penalties > 2.0:
                reason = f"Штрафы перевесили: {' | '.join(penalty_reasons)}"
            else:
                reason = f"Score {score:.2f} ниже порога {MIN_SCORE_TO_TRADE}"
        else:
            reason = f"Score {score:.2f} | " + (
                " | ".join(penalty_reasons) if penalty_reasons else "Чисто"
            )

        return self._build_result(signal, score, reason, agent_votes, data_quality,
                                   final_buy, final_sell, penalty_reasons)

    # ─────────────────────────────────────────
    #  SCORE FORMULA
    # ─────────────────────────────────────────

    def _weighted_score(self, votes: list, data_quality: float) -> float:
        """
        Σ(agent_weight × confidence × data_freshness × quality_factor)
        """
        total = 0.0
        for v in votes:
            name       = v.get("name", "").upper()
            weight     = self.agent_weights.get(name, 0.30)
            confidence = v.get("confidence", 5) / 10.0  # нормируем 0-1
            freshness  = v.get("data_freshness", 1.0)
            quality_f  = (data_quality + freshness) / 2.0  # среднее
            total += weight * confidence * quality_f
        return round(total, 3)

    def _avg_freshness(self, votes: list) -> float:
        fresh = [v.get("data_freshness", 1.0) for v in votes]
        return sum(fresh) / len(fresh) if fresh else 1.0

    # ─────────────────────────────────────────
    #  АДАПТАЦИЯ ВЕСОВ
    # ─────────────────────────────────────────

    def update_agent_weight(self, agent_name: str, new_weight: float):
        """Обновляет вес агента (из optimizer'а)."""
        self.agent_weights[agent_name.upper()] = round(max(0.05, min(1.0, new_weight)), 3)

    def load_weights_from_config(self, config_dict: dict):
        """Загружает веса из конфига (IMPULSE_WEIGHT, TREND_WEIGHT, etc.)."""
        mapping = {
            "IMPULSE_WEIGHT": "IMPULSE",
            "TREND_WEIGHT":   "TREND",
            "ANALYST_WEIGHT": "ANALYST",
        }
        for config_key, agent_name in mapping.items():
            if config_key in config_dict:
                self.agent_weights[agent_name] = config_dict[config_key]

    # ─────────────────────────────────────────
    #  DIVERSITY CHECK
    # ─────────────────────────────────────────

    def check_diversity(self, recent_votes_history: list) -> dict:
        """
        Проверяет насколько агенты реально независимы.
        recent_votes_history — список списков голосов за последние N решений.
        Возвращает correlation matrix и рекомендации.
        """
        if len(recent_votes_history) < 10:
            return {"status": "insufficient_data", "message": "Нужно ≥10 решений"}

        # Собираем сигналы каждого агента в числовой вектор
        agent_signals: dict = {}
        for votes in recent_votes_history:
            for v in votes:
                name = v.get("name", "?")
                sig  = 1 if v.get("signal") == "BUY" else (-1 if v.get("signal") == "SELL" else 0)
                if name not in agent_signals:
                    agent_signals[name] = []
                agent_signals[name].append(sig)

        agents  = list(agent_signals.keys())
        corr    = {}
        high_corr_pairs = []

        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                s1 = agent_signals[a1]
                s2 = agent_signals[a2]
                n  = min(len(s1), len(s2))
                if n < 5:
                    continue
                # Ручная корреляция Пирсона
                s1n, s2n = s1[:n], s2[:n]
                mean1 = sum(s1n) / n
                mean2 = sum(s2n) / n
                num   = sum((a - mean1) * (b - mean2) for a, b in zip(s1n, s2n))
                den1  = sum((a - mean1)**2 for a in s1n) ** 0.5
                den2  = sum((b - mean2)**2 for b in s2n) ** 0.5
                r     = num / (den1 * den2 + 1e-10)
                corr[f"{a1}/{a2}"] = round(r, 3)
                if abs(r) > 0.75:
                    high_corr_pairs.append((a1, a2, r))

        warnings = []
        for a1, a2, r in high_corr_pairs:
            warnings.append(
                f"⚠️ {a1} и {a2} коррелируют на {r:.0%} — "
                f"возможно дублируют друг друга (снизить вес одного)"
            )

        return {
            "status":          "ok",
            "correlations":    corr,
            "high_corr_pairs": high_corr_pairs,
            "warnings":        warnings,
            "message":         "\n".join(warnings) if warnings else "Агенты достаточно независимы ✅",
        }

    # ─────────────────────────────────────────
    #  BUILD + LOG
    # ─────────────────────────────────────────

    def _build_result(
        self, signal: str, score: float, reason: str,
        agent_votes: list, data_quality: float,
        buy_score: float = 0.0, sell_score: float = 0.0,
        penalties: list = None,
    ) -> dict:
        result = {
            "signal":      signal,
            "score":       round(score, 3),
            "buy_score":   round(buy_score, 3),
            "sell_score":  round(sell_score, 3),
            "reason":      reason,
            "data_quality": data_quality,
            "penalties":   penalties or [],
            "timestamp":   datetime.now().isoformat(),
            "summary":     self._build_summary(signal, score, reason, buy_score, sell_score),
        }
        self._log(result)
        return result

    def _build_summary(
        self, signal: str, score: float, reason: str,
        buy_score: float, sell_score: float
    ) -> str:
        icon = {"BUY": "🟢", "SELL": "🔴", "NO_TRADE": "⬜"}.get(signal, "❓")
        return (
            f"=== АРБИТР: {icon} {signal} (score={score:.2f}) ===\n"
            f"  BUY-score: {buy_score:.2f} | SELL-score: {sell_score:.2f}\n"
            f"  Причина: {reason}"
        )

    def _log(self, result: dict):
        try:
            entry = {
                "time":       result["timestamp"],
                "signal":     result["signal"],
                "score":      result["score"],
                "buy_score":  result["buy_score"],
                "sell_score": result["sell_score"],
            }
            with open(ARBITER_LOG, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass


# ─────────────────────────────────────────
#  TYPED CONTRACT HELPER
# ─────────────────────────────────────────

def make_vote(
    name:              str,
    signal:            str,   # "BUY" | "SELL" | "WAIT"
    confidence:        int,   # 1-10
    reasoning:         str   = "",
    horizon:           str   = "INTRADAY",
    invalidation_level: float = 0.0,
    expected_move_atr:  float = 1.0,
    data_freshness:     float = 1.0,
    veto:              bool  = False,
) -> dict:
    """
    Создаёт типизированный контракт голоса агента.
    Используется всеми агентами Консилиума.
    """
    return {
        "name":               name,
        "signal":             signal,
        "confidence":         max(1, min(10, confidence)),
        "reasoning":          reasoning[:200],
        "horizon":            horizon,
        "invalidation_level": invalidation_level,
        "expected_move_atr":  expected_move_atr,
        "data_freshness":     max(0.0, min(1.0, data_freshness)),
        "veto":               veto,
        "timestamp":          datetime.now().isoformat(),
    }


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

arbiter = Arbiter()
