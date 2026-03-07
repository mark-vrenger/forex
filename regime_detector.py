"""
regime_detector.py — Детектор Режима Рынка v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
НОВОЕ v2.0: Вместо жёстких ярлыков ("TREND") — вероятности:

  {
    "trend_prob":    0.62,
    "range_prob":    0.21,
    "breakout_prob": 0.11,
    "reversal_prob": 0.06,
  }

+ Гистерезис: режим не прыгает туда-обратно на границе.
  Смена режима только если новый доминирует ≥ 3 бара подряд.

Метрики для классификации:
  - ADX (14) — сила тренда
  - BB Width — ширина канала (сжатие = breakout вероятен)
  - ATR / SMA ratio — нормированная волатильность
  - Price vs SMA — выше/ниже скользящей
  - RSI extremes — перекупленность/перепроданность
  - Candle momentum — последние 5 свечей
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── Гистерезис ──────────────────────────────────
HYSTERESIS_BARS = 3    # подтверждений для смены режима


class RegimeDetector:

    def __init__(self):
        self._last_probs     = {"trend": 0.25, "range": 0.25, "breakout": 0.25, "reversal": 0.25}
        self._candidate      = None   # кандидат на смену режима
        self._candidate_cnt  = 0      # сколько баров подряд кандидат доминирует

    # ─────────────────────────────────────────
    #  ГЛАВНЫЙ МЕТОД
    # ─────────────────────────────────────────

    def detect(self, df: pd.DataFrame, tf_name: str = "H1") -> dict:
        """
        Возвращает вероятности режимов + текущий dominant + контекст для промпта.

        df — DataFrame с open/high/low/close/tick_volume
        """
        if df is None or len(df) < 30:
            return self._empty()

        # ── Вычисляем сырые метрики ──────────
        adx_val, plus_di, minus_di = self._calc_adx(df)
        bb_width     = self._calc_bb_width(df)
        atr_ratio    = self._calc_atr_ratio(df)
        rsi_val      = self._calc_rsi(df)
        trend_slope  = self._calc_trend_slope(df)
        momentum     = self._calc_momentum(df)

        # ── Вычисляем сырые баллы ────────────
        raw = {
            "trend":    self._score_trend(adx_val, plus_di, minus_di, trend_slope),
            "range":    self._score_range(adx_val, bb_width, atr_ratio),
            "breakout": self._score_breakout(bb_width, atr_ratio, momentum),
            "reversal": self._score_reversal(rsi_val, momentum, adx_val),
        }

        # ── Нормируем в вероятности ──────────
        total = sum(raw.values()) + 1e-10
        probs = {k: round(v / total, 3) for k, v in raw.items()}

        # ── Гистерезис ───────────────────────
        dominant = max(probs, key=probs.get)
        probs, dominant = self._apply_hysteresis(probs, dominant)

        # ── Направление тренда ───────────────
        trend_direction = "UP" if plus_di > minus_di else "DOWN"
        if probs["trend"] < 0.4:
            trend_direction = "NONE"

        result = {
            # Вероятности
            "trend_prob":    probs["trend"],
            "range_prob":    probs["range"],
            "breakout_prob": probs["breakout"],
            "reversal_prob": probs["reversal"],

            # Итог
            "dominant":       dominant,
            "trend_direction": trend_direction,
            "confidence":     round(probs[dominant], 3),

            # Метрики для отладки
            "adx":         round(adx_val, 1),
            "rsi":         round(rsi_val, 1),
            "bb_width_pct": round(bb_width * 100, 2),
            "atr_ratio":   round(atr_ratio, 4),
            "momentum":    round(momentum, 3),
            "trend_slope": round(trend_slope, 4),

            # Текст для промпта
            "summary":     self._build_summary(probs, dominant, trend_direction,
                                               adx_val, rsi_val),
        }

        self._last_probs = probs
        return result

    # ─────────────────────────────────────────
    #  СКОРИНГ ПО РЕЖИМУ
    # ─────────────────────────────────────────

    def _score_trend(self, adx, plus_di, minus_di, slope) -> float:
        score = 0.0
        # ADX > 25 = сильный тренд
        if adx > 30:   score += 3.0
        elif adx > 25: score += 2.0
        elif adx > 20: score += 1.0
        # Разница DI
        di_diff = abs(plus_di - minus_di)
        score += min(di_diff / 10.0, 2.0)
        # Наклон цены
        if abs(slope) > 0.0003:
            score += 1.5
        elif abs(slope) > 0.0001:
            score += 0.7
        return score

    def _score_range(self, adx, bb_width, atr_ratio) -> float:
        score = 0.0
        # Слабый ADX
        if adx < 20:   score += 3.0
        elif adx < 25: score += 1.5
        # Узкий BB
        if bb_width < 0.005:   score += 2.5
        elif bb_width < 0.010: score += 1.0
        # Низкий ATR
        if atr_ratio < 0.003:  score += 1.5
        return score

    def _score_breakout(self, bb_width, atr_ratio, momentum) -> float:
        score = 0.0
        # Сжатие BB перед взрывом
        if bb_width < 0.004:   score += 2.5
        elif bb_width < 0.007: score += 1.0
        # Высокий momentum после сжатия
        if abs(momentum) > 0.5: score += 2.0
        elif abs(momentum) > 0.3: score += 1.0
        # Рост ATR
        if atr_ratio > 0.008: score += 1.0
        return score

    def _score_reversal(self, rsi, momentum, adx) -> float:
        score = 0.0
        # Экстремальный RSI
        if rsi > 75 or rsi < 25:   score += 3.0
        elif rsi > 70 or rsi < 30: score += 1.5
        # Ослабление momentum (тренд выдыхается)
        if adx > 25 and abs(momentum) < 0.1: score += 2.0
        # Противоположный momentum при экстремальном RSI
        if rsi > 70 and momentum < -0.2: score += 1.5
        if rsi < 30 and momentum > 0.2:  score += 1.5
        return score

    # ─────────────────────────────────────────
    #  ГИСТЕРЕЗИС
    # ─────────────────────────────────────────

    def _apply_hysteresis(self, probs: dict, new_dominant: str) -> tuple:
        """
        Режим меняется только если новый доминант держится HYSTERESIS_BARS баров.
        Предотвращает мигание на границе.
        """
        current_dominant = max(self._last_probs, key=self._last_probs.get)

        if new_dominant == current_dominant:
            self._candidate     = None
            self._candidate_cnt = 0
            return probs, new_dominant

        # Кандидат на смену
        if self._candidate == new_dominant:
            self._candidate_cnt += 1
        else:
            self._candidate     = new_dominant
            self._candidate_cnt = 1

        if self._candidate_cnt >= HYSTERESIS_BARS:
            # Смена подтверждена
            self._candidate     = None
            self._candidate_cnt = 0
            return probs, new_dominant
        else:
            # Держим старый режим (но показываем актуальные вероятности)
            return probs, current_dominant

    # ─────────────────────────────────────────
    #  РАСЧЁТ МЕТРИК
    # ─────────────────────────────────────────

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        try:
            h, l, c = df["high"], df["low"], df["close"]
            prev_c  = c.shift(1)

            tr = pd.concat([
                h - l,
                (h - prev_c).abs(),
                (l - prev_c).abs(),
            ], axis=1).max(axis=1)

            plus_dm  = h.diff().clip(lower=0)
            minus_dm = (-l.diff()).clip(lower=0)
            plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

            atr14      = self._ema(tr.values, period)
            plus_di14  = 100 * self._ema(plus_dm.values, period)  / (atr14 + 1e-10)
            minus_di14 = 100 * self._ema(minus_dm.values, period) / (atr14 + 1e-10)

            dx  = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
            adx = self._ema(dx, period)

            return float(adx[-1]), float(plus_di14[-1]), float(minus_di14[-1])
        except Exception:
            return 20.0, 20.0, 20.0

    def _calc_bb_width(self, df: pd.DataFrame, period: int = 20) -> float:
        try:
            c   = df["close"].tail(period + 5)
            sma = c.rolling(period).mean()
            std = c.rolling(period).std()
            bb_width = (2 * std / (sma + 1e-10)).iloc[-1]
            return float(bb_width)
        except Exception:
            return 0.01

    def _calc_atr_ratio(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            tr  = (df["high"] - df["low"]).tail(period)
            atr = tr.mean()
            price = df["close"].iloc[-1]
            return float(atr / (price + 1e-10))
        except Exception:
            return 0.005

    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            delta = df["close"].diff().tail(period + 5)
            gain  = delta.where(delta > 0, 0).rolling(period).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
            return float(100 - 100 / (1 + rs))
        except Exception:
            return 50.0

    def _calc_trend_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        try:
            c = df["close"].tail(period).values
            x = np.arange(len(c))
            slope = np.polyfit(x, c, 1)[0]
            return float(slope)
        except Exception:
            return 0.0

    def _calc_momentum(self, df: pd.DataFrame, period: int = 5) -> float:
        try:
            last = df["close"].tail(period + 1).values
            moves = [(last[i+1] - last[i]) / (last[i] + 1e-10) for i in range(len(last)-1)]
            return float(sum(moves) / len(moves))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────
    #  SUMMARY ДЛЯ ПРОМПТА
    # ─────────────────────────────────────────

    def _build_summary(self, probs, dominant, trend_dir, adx, rsi) -> str:
        bars = {
            "trend":    "████" if probs["trend"]    > 0.5 else "██" if probs["trend"]    > 0.3 else "▌",
            "range":    "████" if probs["range"]    > 0.5 else "██" if probs["range"]    > 0.3 else "▌",
            "breakout": "████" if probs["breakout"] > 0.5 else "██" if probs["breakout"] > 0.3 else "▌",
            "reversal": "████" if probs["reversal"] > 0.5 else "██" if probs["reversal"] > 0.3 else "▌",
        }
        return (
            f"=== РЕЖИМ РЫНКА: {dominant.upper()} ({probs[dominant]:.0%}) ===\n"
            f"  trend    {bars['trend']} {probs['trend']:.0%}\n"
            f"  range    {bars['range']} {probs['range']:.0%}\n"
            f"  breakout {bars['breakout']} {probs['breakout']:.0%}\n"
            f"  reversal {bars['reversal']} {probs['reversal']:.0%}\n"
            f"  ADX:{adx:.0f} | RSI:{rsi:.0f}"
            + (f" | Тренд: {trend_dir}" if trend_dir != "NONE" else "")
        )

    def _empty(self) -> dict:
        return {
            "trend_prob": 0.25, "range_prob": 0.25,
            "breakout_prob": 0.25, "reversal_prob": 0.25,
            "dominant": "unknown", "trend_direction": "NONE",
            "confidence": 0.25, "adx": 0, "rsi": 50,
            "bb_width_pct": 0, "atr_ratio": 0,
            "momentum": 0, "trend_slope": 0,
            "summary": "Режим: недостаточно данных",
        }

    @staticmethod
    def _ema(arr, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        out   = np.zeros_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

regime_detector = RegimeDetector()
