"""
data_steward.py — Агент-Стюард Данных v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Задача: проверить качество входных данных ДО того как
        агенты получат их. Если данные грязные — сигнал
        не выдаётся вообще.

Проверки:
  1. Пропущенные свечи (gaps > 2× expected interval)
  2. Дубликаты по timestamp
  3. Выбросы цены (spike > N×ATR)
  4. Несинхронность таймфреймов (M15 и H1 смотрят на разное время)
  5. Нулевые / отрицательные OHLC
  6. Инверсия OHLC (high < low, close outside range)
  7. Объём ноль при активной сессии
  8. Staleness — данные устарели

Каждый батч получает quality_score 0.0–1.0.
Ниже порога MIN_QUALITY → блокируем торговлю.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import json
import os

MIN_QUALITY       = 0.75   # ниже — блокировать торговлю
STALE_MINUTES     = 30     # данные старше 30 мин — stale
SPIKE_ATR_MULT    = 5.0    # свеча > 5×ATR — выброс
MAX_GAP_MULT      = 3.0    # разрыв > 3× ожидаемого интервала — gap
LOG_FILE          = "logs/data_quality.jsonl"

# Ожидаемые интервалы в минутах
TF_INTERVALS = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


class DataSteward:

    def __init__(self):
        os.makedirs("logs", exist_ok=True)

    # ─────────────────────────────────────────
    #  ГЛАВНЫЙ МЕТОД
    # ─────────────────────────────────────────

    def validate_batch(
        self,
        tf_data: dict,           # {"M15": df, "H1": df, ...}
        current_time: Optional[datetime] = None,
        live_mode: bool = False,
    ) -> dict:
        """
        Проверяет все таймфреймы в батче.

        Возвращает:
        {
          "quality_score": 0.0–1.0,
          "tradeable":     bool,
          "issues":        [{"tf": "M15", "type": "gap", "severity": "warn", "detail": "..."}],
          "tf_scores":     {"M15": 0.9, "H1": 0.85},
          "summary":       str,  ← для промпта агентов
        }
        """
        issues      = []
        tf_scores   = {}
        now         = current_time or datetime.now()

        for tf_name, df in tf_data.items():
            if df is None or len(df) == 0:
                issues.append({
                    "tf": tf_name, "type": "missing",
                    "severity": "critical", "detail": "DataFrame пустой или None"
                })
                tf_scores[tf_name] = 0.0
                continue

            score, tf_issues = self._check_dataframe(df, tf_name, now, live_mode)
            tf_scores[tf_name] = score
            issues.extend(tf_issues)

        # Синхронность между таймфреймами
        sync_issues = self._check_sync(tf_data, now)
        issues.extend(sync_issues)

        # Итоговый score = среднее по TF с весами
        weights = {"M1": 0.5, "M5": 0.7, "M15": 1.0, "M30": 0.9,
                   "H1": 1.0, "H4": 1.0, "D1": 0.8}
        total_w = total_s = 0.0
        for tf_name, score in tf_scores.items():
            w = weights.get(tf_name, 1.0)
            total_w += w
            total_s += score * w
        quality_score = total_s / total_w if total_w > 0 else 0.0

        # Критические ошибки → score в 0
        critical = [i for i in issues if i["severity"] == "critical"]
        if critical:
            quality_score = min(quality_score, 0.3)

        tradeable = quality_score >= MIN_QUALITY
        summary   = self._build_summary(quality_score, tradeable, issues)

        result = {
            "quality_score": round(quality_score, 3),
            "tradeable":     tradeable,
            "issues":        issues,
            "tf_scores":     {k: round(v, 3) for k, v in tf_scores.items()},
            "summary":       summary,
            "timestamp":     now.isoformat(),
        }

        self._log(result)
        return result

    # ─────────────────────────────────────────
    #  ПРОВЕРКИ ОДНОГО ДАТАФРЕЙМА
    # ─────────────────────────────────────────

    def _check_dataframe(
        self, df: pd.DataFrame, tf_name: str,
        now: datetime, live_mode: bool
    ) -> tuple:
        """Возвращает (score 0-1, list of issues)."""
        issues  = []
        penalty = 0.0
        interval_min = TF_INTERVALS.get(tf_name, 15)

        # ── 1. Нулевые / отрицательные OHLC ──
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                bad = (df[col] <= 0).sum()
                if bad > 0:
                    issues.append({
                        "tf": tf_name, "type": "zero_price",
                        "severity": "critical",
                        "detail": f"{col}: {bad} нулевых/отрицательных значений"
                    })
                    penalty += 0.4

        # ── 2. Инверсия OHLC ──
        if all(c in df.columns for c in ["high", "low", "open", "close"]):
            inv_hl = (df["high"] < df["low"]).sum()
            inv_oc = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
            if inv_hl > 0:
                issues.append({
                    "tf": tf_name, "type": "ohlc_inversion",
                    "severity": "critical",
                    "detail": f"high < low: {inv_hl} свечей"
                })
                penalty += 0.3
            if inv_oc > 0:
                issues.append({
                    "tf": tf_name, "type": "ohlc_inversion",
                    "severity": "warn",
                    "detail": f"close вне диапазона: {inv_oc} свечей"
                })
                penalty += 0.1

        # ── 3. Дубликаты по времени ──
        if "time" in df.columns:
            dups = df["time"].duplicated().sum()
            if dups > 0:
                issues.append({
                    "tf": tf_name, "type": "duplicate",
                    "severity": "warn",
                    "detail": f"{dups} дублированных timestamp"
                })
                penalty += min(0.2, dups * 0.02)

        # ── 4. Пропущенные свечи (gaps) ──
        if "time" in df.columns and len(df) > 2:
            times    = pd.to_datetime(df["time"])
            diffs    = times.diff().dropna()
            expected = pd.Timedelta(minutes=interval_min)
            gaps     = (diffs > expected * MAX_GAP_MULT).sum()
            max_gap  = diffs.max()
            if gaps > 0:
                sev = "critical" if gaps > 5 else "warn"
                issues.append({
                    "tf": tf_name, "type": "gap",
                    "severity": sev,
                    "detail": f"{gaps} пропусков, макс разрыв {max_gap}"
                })
                penalty += min(0.25, gaps * 0.03)

        # ── 5. Ценовые выбросы (спайки) ──
        if all(c in df.columns for c in ["high", "low", "close"]):
            tr     = (df["high"] - df["low"]).abs()
            atr14  = tr.rolling(14).mean()
            spikes = (tr > atr14 * SPIKE_ATR_MULT).sum()
            if spikes > 0:
                issues.append({
                    "tf": tf_name, "type": "spike",
                    "severity": "warn",
                    "detail": f"{spikes} ценовых выбросов > {SPIKE_ATR_MULT}×ATR"
                })
                penalty += min(0.15, spikes * 0.05)

        # ── 6. Нулевой объём в активные часы ──
        if "tick_volume" in df.columns and "time" in df.columns:
            times   = pd.to_datetime(df["time"])
            active  = times.dt.hour.between(7, 21)
            zero_v  = ((df["tick_volume"] == 0) & active).sum()
            if zero_v > 2:
                issues.append({
                    "tf": tf_name, "type": "zero_volume",
                    "severity": "warn",
                    "detail": f"{zero_v} свечей с нулевым объёмом в торговые часы"
                })
                penalty += min(0.15, zero_v * 0.01)

        # ── 7. Staleness (только в live-режиме) ──
        if live_mode and "time" in df.columns:
            last_bar_time = pd.to_datetime(df["time"].iloc[-1])
            if hasattr(last_bar_time, "to_pydatetime"):
                last_bar_time = last_bar_time.to_pydatetime()
            age_min = (now - last_bar_time.replace(tzinfo=None)).total_seconds() / 60
            if age_min > STALE_MINUTES:
                issues.append({
                    "tf": tf_name, "type": "stale",
                    "severity": "critical",
                    "detail": f"Последний бар {age_min:.0f} мин назад (лимит {STALE_MINUTES})"
                })
                penalty += 0.4

        score = max(0.0, 1.0 - penalty)
        return round(score, 3), issues

    # ─────────────────────────────────────────
    #  СИНХРОННОСТЬ ТАЙМФРЕЙМОВ
    # ─────────────────────────────────────────

    def _check_sync(self, tf_data: dict, now: datetime) -> list:
        """
        Проверяет что M15 и H1 смотрят на совместимые временные окна.
        """
        issues = []
        times  = {}
        for tf_name, df in tf_data.items():
            if df is not None and "time" in df.columns and len(df) > 0:
                t = pd.to_datetime(df["time"].iloc[-1])
                if hasattr(t, "to_pydatetime"):
                    t = t.to_pydatetime().replace(tzinfo=None)
                times[tf_name] = t

        if len(times) < 2:
            return issues

        # Максимальная разница между таймфреймами не должна быть слишком большой
        t_vals = list(times.values())
        max_diff = max((abs((a - b).total_seconds()) for a in t_vals for b in t_vals
                        if a != b), default=0)

        if max_diff > 3600 * 4:  # 4 часа
            issues.append({
                "tf": "SYNC", "type": "desync",
                "severity": "critical",
                "detail": f"Таймфреймы рассинхронизированы на {max_diff/3600:.1f}ч"
            })
        elif max_diff > 3600:
            issues.append({
                "tf": "SYNC", "type": "desync",
                "severity": "warn",
                "detail": f"Небольшая рассинхронизация {max_diff/60:.0f}мин"
            })
        return issues

    # ─────────────────────────────────────────
    #  CSV ВАЛИДАЦИЯ (для slow_trainer)
    # ─────────────────────────────────────────

    def validate_csv(self, df: pd.DataFrame, tf_name: str) -> dict:
        """
        Полная проверка CSV файла перед обучением.
        Возвращает отчёт + очищенный DataFrame.
        """
        original_len = len(df)
        report       = {"tf": tf_name, "original_rows": original_len, "fixes": []}

        # Удаляем строки с нулевыми OHLC
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                bad = df[col] <= 0
                if bad.any():
                    df = df[~bad]
                    report["fixes"].append(f"Удалено {bad.sum()} строк с {col}<=0")

        # Исправляем инверсию high/low (swap)
        if "high" in df.columns and "low" in df.columns:
            inv = df["high"] < df["low"]
            if inv.any():
                df.loc[inv, ["high", "low"]] = df.loc[inv, ["low", "high"]].values
                report["fixes"].append(f"Исправлено {inv.sum()} инвертированных high/low")

        # Удаляем дубликаты
        if "time" in df.columns:
            dups = df.duplicated(subset=["time"])
            if dups.any():
                df = df[~dups]
                report["fixes"].append(f"Удалено {dups.sum()} дублированных строк")

        # Клипаем объём
        if "tick_volume" in df.columns:
            df["tick_volume"] = df["tick_volume"].clip(lower=0)

        # Заполняем нулевой объём медианой
        if "tick_volume" in df.columns:
            zero_v = df["tick_volume"] == 0
            if zero_v.any():
                median_v = df["tick_volume"][~zero_v].median()
                df.loc[zero_v, "tick_volume"] = median_v
                report["fixes"].append(
                    f"Заполнено {zero_v.sum()} нулевых объёмов медианой ({median_v:.0f})"
                )

        df = df.reset_index(drop=True)
        report["final_rows"]  = len(df)
        report["removed_rows"] = original_len - len(df)
        report["clean_ratio"] = round(len(df) / original_len, 3) if original_len > 0 else 1.0

        print(f"[DataSteward] {tf_name}: "
              f"{original_len}→{len(df)} строк "
              f"(убрано {report['removed_rows']})")
        for fix in report["fixes"]:
            print(f"  ↳ {fix}")

        return {"report": report, "df": df}

    # ─────────────────────────────────────────
    #  SUMMARY ДЛЯ ПРОМПТА
    # ─────────────────────────────────────────

    def _build_summary(
        self, score: float, tradeable: bool, issues: list
    ) -> str:
        status = "✅ ТОРГОВЛЯ РАЗРЕШЕНА" if tradeable else "🚫 ТОРГОВЛЯ ЗАБЛОКИРОВАНА"
        lines  = [
            f"=== DATA QUALITY: {score:.0%} | {status} ===",
        ]
        if issues:
            crits = [i for i in issues if i["severity"] == "critical"]
            warns = [i for i in issues if i["severity"] == "warn"]
            if crits:
                for i in crits:
                    lines.append(f"  ❌ [{i['tf']}] {i['type']}: {i['detail']}")
            if warns:
                for i in warns[:3]:  # Не спамим промпт
                    lines.append(f"  ⚠️ [{i['tf']}] {i['type']}: {i['detail']}")
        else:
            lines.append("  Данные чистые, нет замечаний.")
        return "\n".join(lines)

    def _log(self, result: dict):
        try:
            entry = {
                "time":          result["timestamp"],
                "quality_score": result["quality_score"],
                "tradeable":     result["tradeable"],
                "issue_count":   len(result["issues"]),
                "tf_scores":     result["tf_scores"],
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

steward = DataSteward()
