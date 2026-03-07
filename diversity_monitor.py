"""
diversity_monitor.py — Монитор Разнообразия + Drift Watchdog v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Два модуля в одном файле:

1. DiversityMonitor
   Следит за тем, насколько агенты реально независимы.
   Если агенты коррелируют >75% — они описывают одно и то же движение
   разными словами. Консилиум кажется единогласным, но это не
   независимые подтверждения, а эхо-камера.

2. DriftWatchdog
   Отслеживает два вида дрейфа:
   - feature drift: распределение признаков изменилось
   - concept drift: признаки те же, но их связь с результатом изменилась

   Если дрейф высокий → система переходит в SAFE_MODE.

3. ExpectancyTracker
   Заменяет голый win-rate на правильные метрики:
   - expectancy = avg_win × WR - avg_loss × (1-WR)
   - payoff_ratio = avg_win / avg_loss
   - ulcer_index
   - rolling_expectancy по последним N сделкам
"""

import json
import os
import math
from datetime import datetime
from collections import deque
from typing import Optional

DIVERSITY_LOG = "logs/diversity_monitor.jsonl"
DRIFT_LOG     = "logs/drift_watchdog.jsonl"

# ── Пороги ───────────────────────────────────────────
HIGH_CORRELATION    = 0.75    # выше → агент дублирует другого
FEATURE_DRIFT_SIGMA = 3.0     # сдвиг > 3σ → feature drift
CONCEPT_DRIFT_DELTA = 0.15    # ухудшение expectancy на 15% → concept drift
MIN_TRADES_DRIFT    = 20      # минимум сделок для оценки дрейфа
WINDOW_REFERENCE    = 100     # эталонное окно (сделок)
WINDOW_RECENT       = 30      # сравниваемое окно (сделок)


# ═══════════════════════════════════════════════
#  1. DIVERSITY MONITOR
# ═══════════════════════════════════════════════

class DiversityMonitor:
    """
    Мерит корреляцию сигналов агентов и вклад каждого в PnL.
    Если агент дублирует других — снижает его вес.
    """

    def __init__(self, window: int = 50):
        self.window     = window
        self.history    = deque(maxlen=window)  # list of vote_dicts per decision
        os.makedirs("logs", exist_ok=True)

    def record_votes(self, votes: list, outcome: Optional[str] = None, pnl: float = 0):
        """Записывает голоса и исход решения."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "votes":     [{"name": v.get("name"), "signal": v.get("signal"),
                           "confidence": v.get("confidence")} for v in votes],
            "outcome":   outcome,
            "pnl":       pnl,
        }
        self.history.append(entry)

    def analyze(self) -> dict:
        """
        Анализирует разнообразие агентов.
        Возвращает корреляции, предупреждения, рекомендации весов.
        """
        if len(self.history) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Нужно ≥10 решений (сейчас {len(self.history)})",
                "weight_adjustments": {},
            }

        # Собираем сигналы по агентам
        agent_sigs: dict = {}
        agent_pnl:  dict = {}

        for record in self.history:
            pnl = record.get("pnl", 0)
            for v in record.get("votes", []):
                name = v.get("name", "?")
                sig  = 1 if v.get("signal") == "BUY" else (
                      -1 if v.get("signal") == "SELL" else 0)
                if name not in agent_sigs:
                    agent_sigs[name] = []
                    agent_pnl[name]  = []
                agent_sigs[name].append(sig)
                # PnL агента = pnl если его сигнал совпал с решением
                agent_pnl[name].append(pnl)

        agents = list(agent_sigs.keys())
        corr_matrix  = {}
        high_corr    = []
        warnings     = []
        weight_adj   = {}

        # Попарные корреляции
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                r = self._pearson(agent_sigs[a1], agent_sigs[a2])
                corr_matrix[f"{a1}/{a2}"] = round(r, 3)
                if abs(r) >= HIGH_CORRELATION:
                    high_corr.append((a1, a2, r))
                    warnings.append(
                        f"⚠️ {a1} и {a2}: корреляция {r:.0%} — "
                        f"один из них дублирует другого"
                    )
                    # Рекомендуем снизить вес менее результативного
                    pnl1 = sum(agent_pnl.get(a1, [0]))
                    pnl2 = sum(agent_pnl.get(a2, [0]))
                    weaker = a1 if pnl1 < pnl2 else a2
                    weight_adj[weaker] = weight_adj.get(weaker, 1.0) * 0.8
                    warnings.append(
                        f"  → Рекомендую снизить вес {weaker} (PnL ниже)"
                    )

        # Вклад каждого агента в PnL
        agent_contributions = {}
        for name in agents:
            total = sum(agent_pnl.get(name, [0]))
            avg   = total / len(agent_pnl[name]) if agent_pnl[name] else 0
            agent_contributions[name] = round(avg, 3)

        result = {
            "status":             "ok",
            "agents_analyzed":    len(agents),
            "corr_matrix":        corr_matrix,
            "high_corr_pairs":    [(a, b, round(r, 3)) for a, b, r in high_corr],
            "warnings":           warnings,
            "weight_adjustments": weight_adj,
            "agent_pnl_contrib":  agent_contributions,
            "message":            "\n".join(warnings) if warnings else
                                  "✅ Агенты достаточно независимы",
        }

        self._log(result)
        return result

    @staticmethod
    def _pearson(x: list, y: list) -> float:
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        xn, yn   = x[:n], y[:n]
        mx, my   = sum(xn)/n, sum(yn)/n
        num      = sum((a-mx)*(b-my) for a, b in zip(xn, yn))
        den      = (sum((a-mx)**2 for a in xn)**0.5) * (sum((b-my)**2 for b in yn)**0.5)
        return num / (den + 1e-10)

    def _log(self, result: dict):
        try:
            with open(DIVERSITY_LOG, "a", encoding="utf-8") as f:
                json.dump({
                    "time":    datetime.now().isoformat(),
                    "n":       len(self.history),
                    "corr":    result["corr_matrix"],
                    "warnings": len(result["warnings"]),
                }, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  2. DRIFT WATCHDOG
# ═══════════════════════════════════════════════

class DriftWatchdog:
    """
    Обнаруживает feature drift и concept drift.

    Feature drift: распределение входных признаков изменилось
                   (рынок перешёл в другой режим)

    Concept drift: признаки те же, связь с результатом изменилась
                   (стратегия перестала работать)
    """

    def __init__(self):
        self.reference_features: dict = {}  # эталон (первые 100 сделок)
        self.reference_expectancy: float = 0.0
        self.recent_trades:  deque = deque(maxlen=WINDOW_RECENT)
        self.all_trades:     deque = deque(maxlen=WINDOW_REFERENCE + WINDOW_RECENT)
        self._ref_built = False
        os.makedirs("logs", exist_ok=True)

    def record_trade(
        self,
        features: dict,  # {"rsi": 65, "atr": 0.0012, "pgi": 0.71, ...}
        result:   str,   # "WIN" / "LOSS"
        profit:   float,
    ):
        """Записывает завершённую сделку с признаками."""
        record = {
            "features": features,
            "result":   result,
            "profit":   profit,
            "time":     datetime.now().isoformat(),
        }
        self.recent_trades.append(record)
        self.all_trades.append(record)

        # Строим эталон из первых WINDOW_REFERENCE сделок
        if not self._ref_built and len(self.all_trades) >= WINDOW_REFERENCE:
            self._build_reference()

    def check(self) -> dict:
        """
        Проверяет дрейф. Вызывать после каждых 10 сделок.
        """
        if not self._ref_built:
            return {
                "feature_drift": False,
                "concept_drift": False,
                "drift_score":   0.0,
                "message": f"Эталон строится ({len(self.all_trades)}/{WINDOW_REFERENCE})",
                "action": "none",
            }

        if len(self.recent_trades) < 10:
            return {
                "feature_drift": False,
                "concept_drift": False,
                "drift_score":   0.0,
                "message": "Мало свежих сделок для оценки дрейфа",
                "action": "none",
            }

        fd_score, fd_details = self._check_feature_drift()
        cd_score, cd_details = self._check_concept_drift()

        drift_score = max(fd_score, cd_score)
        feature_drift = fd_score > 0.5
        concept_drift = cd_score > 0.5

        messages = []
        if feature_drift:
            messages.append(f"📊 Feature drift (score={fd_score:.2f}): {fd_details}")
        if concept_drift:
            messages.append(f"🔁 Concept drift (score={cd_score:.2f}): {cd_details}")

        # Рекомендуемое действие
        if drift_score > 0.8:
            action = "SAFE_MODE"
        elif drift_score > 0.5:
            action = "reduce_risk"
        else:
            action = "none"

        result = {
            "feature_drift":  feature_drift,
            "concept_drift":  concept_drift,
            "drift_score":    round(drift_score, 3),
            "fd_score":       round(fd_score, 3),
            "cd_score":       round(cd_score, 3),
            "message":        "\n".join(messages) if messages else "✅ Дрейф не обнаружен",
            "action":         action,
            "timestamp":      datetime.now().isoformat(),
        }

        if feature_drift or concept_drift:
            self._log(result)

        return result

    def _build_reference(self):
        """Строит эталонное распределение признаков."""
        ref_trades = list(self.all_trades)[:WINDOW_REFERENCE]
        for record in ref_trades:
            for k, v in record.get("features", {}).items():
                if isinstance(v, (int, float)):
                    if k not in self.reference_features:
                        self.reference_features[k] = []
                    self.reference_features[k].append(float(v))

        # Считаем μ и σ для каждого признака
        self.ref_stats = {}
        for k, vals in self.reference_features.items():
            mu  = sum(vals) / len(vals)
            var = sum((v - mu)**2 for v in vals) / len(vals)
            self.ref_stats[k] = {"mu": mu, "sigma": max(var**0.5, 1e-10)}

        # Эталонная expectancy
        self.reference_expectancy = self._calc_expectancy(ref_trades)
        self._ref_built = True
        print(f"[DriftWatchdog] Эталон построен на {len(ref_trades)} сделках. "
              f"Expectancy: ${self.reference_expectancy:.2f}")

    def _check_feature_drift(self) -> tuple:
        """Проверяет сдвиг признаков (z-score сдвига μ)."""
        if not self.ref_stats:
            return 0.0, ""

        recent_vals: dict = {}
        for record in self.recent_trades:
            for k, v in record.get("features", {}).items():
                if isinstance(v, (int, float)):
                    if k not in recent_vals:
                        recent_vals[k] = []
                    recent_vals[k].append(float(v))

        drifted = []
        for k, vals in recent_vals.items():
            if k not in self.ref_stats or len(vals) < 5:
                continue
            recent_mu = sum(vals) / len(vals)
            ref_mu    = self.ref_stats[k]["mu"]
            ref_sigma = self.ref_stats[k]["sigma"]
            z_score   = abs(recent_mu - ref_mu) / ref_sigma
            if z_score > FEATURE_DRIFT_SIGMA:
                drifted.append(f"{k} z={z_score:.1f}")

        score   = min(1.0, len(drifted) / max(len(self.ref_stats), 1))
        details = ", ".join(drifted[:5]) if drifted else "нет"
        return score, details

    def _check_concept_drift(self) -> tuple:
        """Проверяет изменение expectancy."""
        recent_expectancy = self._calc_expectancy(list(self.recent_trades))
        if self.reference_expectancy == 0:
            return 0.0, ""

        delta = (self.reference_expectancy - recent_expectancy) / abs(self.reference_expectancy)
        delta = max(0.0, delta)  # нас интересует только ухудшение
        details = (f"Эталон ${self.reference_expectancy:.2f} → "
                   f"Сейчас ${recent_expectancy:.2f} (ухудшение {delta:.0%})")
        score = min(1.0, delta / CONCEPT_DRIFT_DELTA)
        return score, details

    @staticmethod
    def _calc_expectancy(trades: list) -> float:
        wins   = [t["profit"] for t in trades if t.get("result") == "WIN" and t.get("profit")]
        losses = [t["profit"] for t in trades if t.get("result") == "LOSS" and t.get("profit")]
        if not trades:
            return 0.0
        wr     = len(wins) / len(trades)
        avg_w  = sum(wins)   / len(wins)   if wins   else 0
        avg_l  = sum(losses) / len(losses) if losses else 0
        return wr * avg_w + (1 - wr) * avg_l

    def _log(self, result: dict):
        try:
            with open(DRIFT_LOG, "a", encoding="utf-8") as f:
                json.dump({k: v for k, v in result.items() if k != "timestamp"},
                          f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  3. EXPECTANCY TRACKER
# ═══════════════════════════════════════════════

class ExpectancyTracker:
    """
    Правильные метрики вместо голого win-rate.
    Используется optimizer'ом для принятия решений.
    """

    def analyze(self, trades: list) -> dict:
        """
        trades — список сделок с полями: result, profit
        Возвращает полную метрику качества стратегии.
        """
        closed = [t for t in trades if t.get("result") in ("WIN", "LOSS")]
        if len(closed) < 5:
            return {"status": "insufficient", "message": "Нужно ≥5 сделок"}

        wins   = [t for t in closed if t.get("result") == "WIN"]
        losses = [t for t in closed if t.get("result") == "LOSS"]
        profits = [t.get("profit", 0) for t in closed]

        wr      = len(wins) / len(closed)
        avg_w   = sum(t.get("profit", 0) for t in wins)   / len(wins)   if wins   else 0
        avg_l   = abs(sum(t.get("profit", 0) for t in losses) / len(losses)) if losses else 1
        payoff  = avg_w / avg_l if avg_l > 0 else 0
        expect  = wr * avg_w - (1 - wr) * avg_l

        # Max drawdown
        equity     = 0.0
        peak       = 0.0
        max_dd     = 0.0
        for p in profits:
            equity += p
            peak    = max(peak, equity)
            max_dd  = max(max_dd, peak - equity)

        # Ulcer Index (среднеквадратичная просадка)
        equities = []
        eq = 0.0
        for p in profits:
            eq += p
            equities.append(eq)
        peak_eq = [max(equities[:i+1]) for i in range(len(equities))]
        dd_pct  = [(peak_eq[i] - equities[i]) / abs(peak_eq[i]) * 100
                   if peak_eq[i] != 0 else 0 for i in range(len(equities))]
        ulcer   = math.sqrt(sum(d**2 for d in dd_pct) / len(dd_pct)) if dd_pct else 0

        # Rolling expectancy (последние 20)
        recent = closed[-20:]
        r_wins = [t for t in recent if t.get("result") == "WIN"]
        r_loss = [t for t in recent if t.get("result") == "LOSS"]
        r_wr   = len(r_wins) / len(recent)
        r_avgw = sum(t.get("profit", 0) for t in r_wins) / len(r_wins) if r_wins else 0
        r_avgl = abs(sum(t.get("profit", 0) for t in r_loss) / len(r_loss)) if r_loss else 1
        rolling_expect = r_wr * r_avgw - (1 - r_wr) * r_avgl

        # Проверка на достаточность edge
        edge_sufficient = (
            expect > 0 and
            payoff > 0.8 and
            wr > 0.35 and
            len(closed) >= 20
        )

        result = {
            "status":           "ok",
            "n_trades":         len(closed),
            "winrate":          round(wr, 3),
            "avg_win":          round(avg_w, 2),
            "avg_loss":         round(avg_l, 2),
            "payoff_ratio":     round(payoff, 3),
            "expectancy":       round(expect, 3),
            "rolling_expect":   round(rolling_expect, 3),
            "max_drawdown":     round(max_dd, 2),
            "ulcer_index":      round(ulcer, 2),
            "edge_sufficient":  edge_sufficient,
            "summary":          self._build_summary(
                wr, expect, payoff, rolling_expect, ulcer, edge_sufficient, len(closed)
            ),
        }
        return result

    @staticmethod
    def _build_summary(wr, expect, payoff, rolling, ulcer, edge_ok, n) -> str:
        icon = "✅" if edge_ok else "⚠️"
        return (
            f"=== EXPECTANCY REPORT ({n} сделок) {icon} ===\n"
            f"  WR: {wr:.0%} | Avg Win/Loss: {payoff:.2f}:1 | "
            f"Expectancy: ${expect:.2f}/сделка\n"
            f"  Rolling(20): ${rolling:.2f} | "
            f"Max DD: ${-abs(0):.0f} | Ulcer: {ulcer:.1f}\n"
            f"  Edge: {'ДОСТАТОЧНЫЙ' if edge_ok else 'НЕДОСТАТОЧНЫЙ — не применять паттерн'}"
        )


# ─────────────────────────────────────────
#  SINGLETONS
# ─────────────────────────────────────────

diversity_monitor  = DiversityMonitor()
drift_watchdog     = DriftWatchdog()
expectancy_tracker = ExpectancyTracker()
