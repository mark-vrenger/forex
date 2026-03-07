"""
custom_indicators.py — Фирменные индикаторы v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Три оригинальных инструмента анализа свечей.
Нигде не описаны — разработаны специально для этой системы.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PGI  — Pressure Gradient Index
   Измеряет АСИММЕТРИЮ давления покупателей/продавцов
   сразу на 4 таймфреймах и вычисляет градиент (ускорение).

2. TVA  — Temporal Volatility Asymmetry
   Сравнивает ATR бычьих свечей против медвежьих.
   Определяет: кто «агрессивнее» — быки или медведи.

3. CandleDNA — Свечной Геном
   Кодирует каждую свечу в 6-мерный вектор.
   Ищет повторяющиеся «геномные последовательности»
   из 3 свечей, предшествующие движению.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
from typing import Optional
import hashlib
import json
import os

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

CANDLE_DNA_MEMORY = "knowledge/candle_dna_memory.json"


# ═══════════════════════════════════════════════
#  1.  PGI — Pressure Gradient Index
# ═══════════════════════════════════════════════

class PressureGradientIndex:
    """
    PGI рассчитывается как:
      pressure_candle = (close - low) / (high - low + ε)
      — это «где внутри свечи закрылась цена» (0 = у дна, 1 = у вершины)

    PGI_tf = EMA( pressure_candle × volume_ratio, period )

    PGI_gradient = скорость изменения PGI (первая производная)

    Сигнал формируется при:
      - Конвергенции PGI на 3+ таймфреймах (все растут или все падают)
      - И при PGI_gradient > порога (ускорение давления)

    Интерпретация:
      PGI > 0.65 + gradient > 0 → накопление, вероятен рост
      PGI < 0.35 + gradient < 0 → распределение, вероятно падение
    """

    def __init__(self, ema_period: int = 14, gradient_period: int = 5):
        self.ema_period      = ema_period
        self.gradient_period = gradient_period

    def calculate(self, df: pd.DataFrame) -> dict:
        """
        df — DataFrame с колонками: open, high, low, close, tick_volume
        Возвращает словарь с PGI, gradient, signal.
        """
        if df is None or len(df) < self.ema_period + self.gradient_period + 5:
            return {"pgi": 0.5, "gradient": 0.0, "signal": "NEUTRAL",
                    "buy": 0, "sell": 0, "text": "PGI: недостаточно данных"}

        high   = df["high"].values
        low    = df["low"].values
        close  = df["close"].values
        volume = df["tick_volume"].values

        # Нормированное давление покупателей на каждой свече
        rng = high - low + 1e-10
        pressure = (close - low) / rng  # 0..1

        # Взвешивание по объёму
        avg_vol   = np.mean(volume[-20:]) + 1e-10
        vol_ratio = volume / avg_vol
        weighted  = pressure * np.clip(vol_ratio, 0.2, 3.0)

        # EMA взвешенного давления
        pgi_series = self._ema(weighted, self.ema_period)

        last_pgi  = float(pgi_series[-1])
        # Gradient = наклон за последние N точек (линейная регрессия)
        window = pgi_series[-self.gradient_period:]
        x = np.arange(len(window))
        gradient = float(np.polyfit(x, window, 1)[0])

        # Нормируем градиент в [-1, +1] относительно типичного масштаба
        pgi_std = float(np.std(pgi_series[-30:])) + 1e-8
        norm_gradient = np.clip(gradient / pgi_std, -1.0, 1.0)

        # Сигнал
        buy, sell = 0, 0
        notes = []

        if last_pgi > 0.65 and norm_gradient > 0.3:
            notes.append("Давление покупателей нарастает")
            buy += 3
        elif last_pgi > 0.55 and norm_gradient > 0.1:
            notes.append("Умеренный рост давления BUY")
            buy += 1
        elif last_pgi < 0.35 and norm_gradient < -0.3:
            notes.append("Давление продавцов нарастает")
            sell += 3
        elif last_pgi < 0.45 and norm_gradient < -0.1:
            notes.append("Умеренный рост давления SELL")
            sell += 1
        else:
            notes.append("Давление сбалансировано")

        if buy > sell:
            signal = "BUY" if buy >= 3 else "LEAN_BUY"
        elif sell > buy:
            signal = "SELL" if sell >= 3 else "LEAN_SELL"
        else:
            signal = "NEUTRAL"

        text = (
            f"  PGI: {last_pgi:.3f} | Градиент: {norm_gradient:+.2f} | "
            f"{' | '.join(notes)} → {signal}"
        )

        return {
            "pgi":      round(last_pgi, 4),
            "gradient": round(norm_gradient, 3),
            "signal":   signal,
            "buy":      buy,
            "sell":     sell,
            "text":     text,
        }

    def multi_tf_signal(self, tf_data: dict) -> dict:
        """
        Рассчитывает PGI на нескольких таймфреймах.
        tf_data = {"M15": df_m15, "H1": df_h1, "H4": df_h4}
        Возвращает общий сигнал по конвергенции.
        """
        results = {}
        buy_total = 0
        sell_total = 0

        for tf_name, df in tf_data.items():
            if df is None:
                continue
            r = self.calculate(df)
            results[tf_name] = r
            buy_total  += r["buy"]
            sell_total += r["sell"]

        # Конвергенция: сколько TF согласны
        buy_tfs  = sum(1 for r in results.values() if r["buy"] > r["sell"])
        sell_tfs = sum(1 for r in results.values() if r["sell"] > r["buy"])
        total_tfs = len(results)

        if buy_tfs == total_tfs and buy_total >= total_tfs * 2:
            consensus = "STRONG_BUY"
            buy_total += 2
        elif sell_tfs == total_tfs and sell_total >= total_tfs * 2:
            consensus = "STRONG_SELL"
            sell_total += 2
        elif buy_tfs > sell_tfs:
            consensus = "LEAN_BUY"
        elif sell_tfs > buy_tfs:
            consensus = "LEAN_SELL"
        else:
            consensus = "DIVERGED"
            buy_total = sell_total = 0

        lines = [f"  PGI Multi-TF: {consensus}"]
        for tf_name, r in results.items():
            lines.append(f"    {tf_name}: PGI={r['pgi']:.3f} grad={r['gradient']:+.2f} → {r['signal']}")

        return {
            "consensus": consensus,
            "buy":       buy_total,
            "sell":      sell_total,
            "details":   results,
            "text":      "\n".join(lines),
        }

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        out = np.zeros_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out


# ═══════════════════════════════════════════════
#  2.  TVA — Temporal Volatility Asymmetry
# ═══════════════════════════════════════════════

class TemporalVolatilityAsymmetry:
    """
    TVA = ATR_bull / ATR_bear

    ATR_bull — средний True Range только БЫЧЬИХ свечей (close > open)
    ATR_bear — средний True Range только МЕДВЕЖЬИХ свечей (close < open)

    TVA > 1.2  → быки «размашистее», вероятен продолжающийся рост
    TVA < 0.83 → медведи агрессивнее, вероятно падение

    Дополнительно анализируется TVA по часам суток (TimeProfile):
    В какое время быки / медведи исторически агрессивнее?
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < self.period:
            return {"tva": 1.0, "signal": "NEUTRAL", "buy": 0, "sell": 0,
                    "text": "TVA: недостаточно данных"}

        d = df.tail(self.period).copy()

        # True Range
        d["prev_close"] = d["close"].shift(1)
        d["tr"] = d[["high", "low", "prev_close"]].apply(
            lambda r: max(
                r["high"] - r["low"],
                abs(r["high"] - (r["prev_close"] or r["low"])),
                abs(r["low"]  - (r["prev_close"] or r["high"])),
            ),
            axis=1,
        )

        bulls = d[d["close"] >= d["open"]]
        bears = d[d["close"] <  d["open"]]

        atr_bull = bulls["tr"].mean() if len(bulls) >= 3 else 0
        atr_bear = bears["tr"].mean() if len(bears) >= 3 else 0

        if atr_bear == 0 or atr_bull == 0:
            tva = 1.0
        else:
            tva = atr_bull / atr_bear

        # Дополнительно: соотношение объёмов быков vs медведей
        vol_bull = bulls["tick_volume"].mean() if len(bulls) >= 3 else 1
        vol_bear = bears["tick_volume"].mean() if len(bears) >= 3 else 1
        vol_ratio = vol_bull / vol_bear if vol_bear > 0 else 1.0

        buy, sell = 0, 0
        notes = []

        # TVA сигнал
        if tva > 1.25:
            notes.append(f"Быки агрессивнее (ATR×{tva:.2f})")
            buy += 2
        elif tva > 1.10:
            notes.append(f"Лёгкое доминирование быков (×{tva:.2f})")
            buy += 1
        elif tva < 0.80:
            notes.append(f"Медведи агрессивнее (ATR×{tva:.2f})")
            sell += 2
        elif tva < 0.90:
            notes.append(f"Лёгкое доминирование медведей (×{tva:.2f})")
            sell += 1
        else:
            notes.append(f"Волатильность симметрична (×{tva:.2f})")

        # Объёмный тест
        if vol_ratio > 1.3:
            notes.append(f"Объём быков выше (×{vol_ratio:.2f})")
            buy += 1
        elif vol_ratio < 0.77:
            notes.append(f"Объём медведей выше (×{vol_ratio:.2f})")
            sell += 1

        signal = "BUY" if buy > sell + 1 else "SELL" if sell > buy + 1 else "NEUTRAL"
        text = f"  TVA: {' | '.join(notes)} → {signal}"

        return {
            "tva":       round(tva, 3),
            "vol_ratio": round(vol_ratio, 3),
            "signal":    signal,
            "buy":       buy,
            "sell":      sell,
            "text":      text,
            "bull_count": len(bulls),
            "bear_count": len(bears),
        }

    def hourly_profile(self, df: pd.DataFrame) -> dict:
        """
        Анализирует TVA по часам суток (если в df есть колонка 'time').
        Возвращает словарь hour -> {'tva': float, 'bias': str}
        Полезно для анализа: в какое время чаще выигрышные / убыточные сделки.
        """
        if df is None or "time" not in df.columns:
            return {}

        d = df.copy()
        d["hour"] = pd.to_datetime(d["time"]).dt.hour
        d["is_bull"] = d["close"] >= d["open"]
        d["tr"] = (d["high"] - d["low"]).abs()

        profile = {}
        for hour in range(24):
            hourly = d[d["hour"] == hour]
            if len(hourly) < 5:
                continue
            bulls_h = hourly[hourly["is_bull"]]
            bears_h = hourly[~hourly["is_bull"]]
            atr_b = bulls_h["tr"].mean() if len(bulls_h) >= 2 else 0
            atr_s = bears_h["tr"].mean() if len(bears_h) >= 2 else 0
            tva_h = (atr_b / atr_s) if atr_s > 0 else 1.0
            profile[hour] = {
                "tva":   round(tva_h, 2),
                "bias":  "bulls" if tva_h > 1.1 else "bears" if tva_h < 0.9 else "neutral",
                "count": len(hourly),
            }
        return profile


# ═══════════════════════════════════════════════
#  3.  CandleDNA — Свечной Геном
# ═══════════════════════════════════════════════

class CandleDNA:
    """
    Каждая свеча кодируется в «геном» — вектор из 6 генов:

    Gene 1: body_size   — размер тела / ATR (нормировано)
    Gene 2: direction   — 1 (бычья) / -1 (медвежья) / 0 (дожи)
    Gene 3: upper_wick  — верхняя тень / (high-low)
    Gene 4: lower_wick  — нижняя тень / (high-low)
    Gene 5: position    — где закрылась внутри high-low (0..1)
    Gene 6: volume_gene — volume_ratio квантован в 5 уровней (0..4)

    Последовательность из 3 свечей образует «ДНК-цепочку» из 18 генов.
    Цепочки хэшируются и сохраняются с результатом следующих N свечей.

    При повторении цепочки ≥ 3 раз с winrate > 60% — сигнал активен.
    """

    DOJI_THRESHOLD   = 0.1   # тело < 10% от high-low => дожи
    SEQUENCE_LENGTH  = 3     # длина анализируемой последовательности
    MIN_OCCURRENCES  = 3     # минимум повторений для сигнала
    MIN_WINRATE      = 0.60  # минимальный winrate для сигнала
    FORWARD_CANDLES  = 10    # смотрим на N свечей вперёд

    def __init__(self):
        os.makedirs("knowledge", exist_ok=True)
        self.memory = self._load_memory()

    # ── ENCODING ────────────────────────────

    def encode_candle(self, candle: dict, atr: float) -> tuple:
        """Кодирует одну свечу в 6-ген вектор."""
        o = candle.get("open",  0)
        h = candle.get("high",  0)
        l = candle.get("low",   0)
        c = candle.get("close", 0)
        v = candle.get("tick_volume", 1)

        rng = max(h - l, 1e-10)
        body = abs(c - o)

        # Gene 1: размер тела (квантуем в 5 уровней)
        body_norm = min(body / (atr + 1e-10), 2.0)
        body_gene = int(body_norm * 2.5)  # 0-5

        # Gene 2: направление
        if body / rng < self.DOJI_THRESHOLD:
            dir_gene = 0   # дожи
        elif c > o:
            dir_gene = 1   # бычья
        else:
            dir_gene = -1  # медвежья

        # Gene 3: верхняя тень (квантуем в 4 уровня)
        upper = h - max(o, c)
        upper_gene = int(min(upper / rng, 1.0) * 3)

        # Gene 4: нижняя тень
        lower = min(o, c) - l
        lower_gene = int(min(lower / rng, 1.0) * 3)

        # Gene 5: позиция закрытия (квантуем в 5 уровней)
        position = (c - l) / rng
        pos_gene = int(position * 4)

        # Gene 6: объёмный ген (передаётся снаружи)
        vol_gene = int(min(v, 4))  # 0-4 (нормируется снаружи)

        return (body_gene, dir_gene, upper_gene, lower_gene, pos_gene, vol_gene)

    def encode_sequence(
        self, candles: list, atr: float, vol_avg: float
    ) -> Optional[str]:
        """
        Кодирует последовательность из SEQUENCE_LENGTH свечей в строковый хэш.
        candles — список словарей (от старой к новой)
        """
        if len(candles) < self.SEQUENCE_LENGTH:
            return None

        genes = []
        for c in candles[-self.SEQUENCE_LENGTH:]:
            vol_norm = int(min(c.get("tick_volume", 1) / (vol_avg + 1e-10) * 2, 4))
            c_mod = dict(c, tick_volume=vol_norm)
            g = self.encode_candle(c_mod, atr)
            genes.extend(g)

        # Строковое представление генома
        genome_str = ",".join(str(g) for g in genes)
        # MD5 хэш для быстрого поиска
        dna_hash = hashlib.md5(genome_str.encode()).hexdigest()[:12]
        return dna_hash

    # ── TRAINING ────────────────────────────

    def train_on_history(self, df: pd.DataFrame) -> int:
        """
        Обучает CandleDNA на исторических данных.
        Проходит по всему df, кодирует последовательности
        и запоминает, что произошло через FORWARD_CANDLES свечей.
        Возвращает количество обученных паттернов.
        """
        if df is None or len(df) < self.SEQUENCE_LENGTH + self.FORWARD_CANDLES + 15:
            return 0

        # ATR и средний объём
        atr_series = self._calc_atr_series(df)
        vol_avg    = df["tick_volume"].mean()

        records = df.to_dict("records")
        new_patterns = 0

        for i in range(self.SEQUENCE_LENGTH, len(records) - self.FORWARD_CANDLES):
            seq   = records[i - self.SEQUENCE_LENGTH: i]
            atr_i = float(atr_series.iloc[i]) if i < len(atr_series) else 0.0001
            dna   = self.encode_sequence(seq, atr_i, vol_avg)
            if dna is None:
                continue

            # Что произошло дальше?
            future = records[i: i + self.FORWARD_CANDLES]
            entry_price = records[i]["close"]
            direction, magnitude = self._measure_move(future, entry_price, atr_i)

            # Обновляем память
            if dna not in self.memory:
                self.memory[dna] = {
                    "buy_wins":  0, "buy_total":  0,
                    "sell_wins": 0, "sell_total": 0,
                    "occurrences": 0,
                    "example_genes": {k: v for k, v in seq[-1].items() if k != "time"},
                }
                new_patterns += 1

            entry = self.memory[dna]
            entry["occurrences"] += 1
            if direction == "UP":
                entry["buy_total"]  += 1
                if magnitude > 0.5:  # прошло > 0.5 ATR вверх
                    entry["buy_wins"] += 1
            else:
                entry["sell_total"] += 1
                if magnitude > 0.5:
                    entry["sell_wins"] += 1

        self._save_memory()
        return new_patterns

    # ── SIGNAL ──────────────────────────────

    def get_signal(
        self, recent_candles: list, atr: float, vol_avg: float
    ) -> dict:
        """
        Возвращает сигнал на основе текущей ДНК-последовательности.
        """
        dna = self.encode_sequence(recent_candles, atr, vol_avg)
        if dna is None or dna not in self.memory:
            return {
                "signal": "NEUTRAL", "buy": 0, "sell": 0,
                "text":   "  CandleDNA: паттерн не распознан",
                "dna_hash": dna,
            }

        entry = self.memory[dna]
        occ   = entry.get("occurrences", 0)

        if occ < self.MIN_OCCURRENCES:
            return {
                "signal": "NEUTRAL", "buy": 0, "sell": 0,
                "text":   f"  CandleDNA: паттерн {dna} (только {occ} примеров, нужно {self.MIN_OCCURRENCES})",
                "dna_hash": dna,
            }

        buy_wr  = entry["buy_wins"]  / entry["buy_total"]  if entry["buy_total"]  > 0 else 0
        sell_wr = entry["sell_wins"] / entry["sell_total"] if entry["sell_total"] > 0 else 0

        buy, sell = 0, 0
        notes = []

        if buy_wr >= self.MIN_WINRATE and entry["buy_total"] >= self.MIN_OCCURRENCES:
            notes.append(f"BUY-паттерн WR={buy_wr:.0%} ({entry['buy_total']} случаев)")
            buy += 3 if buy_wr >= 0.70 else 2
        if sell_wr >= self.MIN_WINRATE and entry["sell_total"] >= self.MIN_OCCURRENCES:
            notes.append(f"SELL-паттерн WR={sell_wr:.0%} ({entry['sell_total']} случаев)")
            sell += 3 if sell_wr >= 0.70 else 2

        if buy > sell:
            signal = "BUY"
        elif sell > buy:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        text = f"  CandleDNA [{dna}]: {' | '.join(notes) if notes else 'нет чёткого паттерна'} → {signal}"

        return {
            "signal":  signal,
            "buy":     buy,
            "sell":    sell,
            "text":    text,
            "dna_hash": dna,
            "buy_wr":  round(buy_wr, 3),
            "sell_wr": round(sell_wr, 3),
            "occurrences": occ,
        }

    # ── HELPERS ─────────────────────────────

    def _measure_move(
        self, future_candles: list, entry_price: float, atr: float
    ) -> tuple:
        """
        Определяет направление и magnitude движения после входа.
        Возвращает ('UP' | 'DOWN', magnitude_in_atr_units).
        """
        if not future_candles:
            return "UP", 0.0
        max_up   = max(c.get("high",  entry_price) for c in future_candles) - entry_price
        max_down = entry_price - min(c.get("low", entry_price) for c in future_candles)
        atr_safe = max(atr, 1e-8)
        if max_up >= max_down:
            return "UP",   max_up   / atr_safe
        else:
            return "DOWN", max_down / atr_safe

    def _calc_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().bfill()

    def _load_memory(self) -> dict:
        if not os.path.exists(CANDLE_DNA_MEMORY):
            return {}
        try:
            with open(CANDLE_DNA_MEMORY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _sanitize_for_json(obj):
        """Рекурсивно конвертирует pandas/numpy типы в базовые Python-типы.
        Решает: TypeError: Object of type Timestamp is not JSON serializable."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: CandleDNA._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [CandleDNA._sanitize_for_json(i) for i in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _save_memory(self):
        # Ограничиваем размер: топ-2000 по occurrences
        if len(self.memory) > 2000:
            sorted_items = sorted(
                self.memory.items(),
                key=lambda x: x[1].get("occurrences", 0),
                reverse=True,
            )
            self.memory = dict(sorted_items[:2000])
        # Чистим Timestamp и numpy-типы перед сериализацией
        clean = self._sanitize_for_json(self.memory)
        with open(CANDLE_DNA_MEMORY, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False)

    def get_stats(self) -> str:
        total = len(self.memory)
        strong = sum(
            1 for v in self.memory.values()
            if v.get("occurrences", 0) >= self.MIN_OCCURRENCES
        )
        return f"CandleDNA: {total} паттернов в памяти, {strong} активных (≥{self.MIN_OCCURRENCES} примеров)"


# ═══════════════════════════════════════════════
#  COMBINED SIGNAL ENGINE
# ═══════════════════════════════════════════════

class CustomSignalEngine:
    """
    Объединяет PGI + TVA + CandleDNA в единый блок сигналов.
    Добавляется в SignalEngine как четвёртый слой.
    """

    def __init__(self):
        self.pgi        = PressureGradientIndex()
        self.tva        = TemporalVolatilityAsymmetry()
        self.candle_dna = CandleDNA()

    def analyze(
        self,
        df_m15: Optional[pd.DataFrame] = None,
        df_h1:  Optional[pd.DataFrame] = None,
        df_h4:  Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Полный анализ кастомными индикаторами.
        Возвращает {'summary': str, 'buy': int, 'sell': int, 'details': dict}
        """
        buy_total  = 0
        sell_total = 0
        lines      = ["=== CUSTOM INDICATORS (PGI / TVA / CandleDNA) ==="]

        # ── PGI Multi-TF ────────────────────
        tf_data = {}
        if df_m15 is not None: tf_data["M15"] = df_m15
        if df_h1  is not None: tf_data["H1"]  = df_h1
        if df_h4  is not None: tf_data["H4"]  = df_h4

        if tf_data:
            pgi_result = self.pgi.multi_tf_signal(tf_data)
            buy_total  += pgi_result["buy"]
            sell_total += pgi_result["sell"]
            lines.append(pgi_result["text"])

        # ── TVA (на M15 для быстрых сигналов) ─
        tva_df = df_m15 if df_m15 is not None else df_h1
        if tva_df is not None:
            tva_result = self.tva.calculate(tva_df)
            buy_total  += tva_result["buy"]
            sell_total += tva_result["sell"]
            lines.append(tva_result["text"])

        # ── CandleDNA ────────────────────────
        if df_m15 is not None and len(df_m15) >= 10:
            recent = df_m15.tail(10).to_dict("records")
            atr    = (df_m15["high"] - df_m15["low"]).tail(14).mean()
            vol_avg = df_m15["tick_volume"].tail(20).mean()
            dna_result = self.candle_dna.get_signal(recent, atr, vol_avg)
            buy_total  += dna_result["buy"]
            sell_total += dna_result["sell"]
            lines.append(dna_result["text"])

        # ── Итоговый вердикт ────────────────
        if buy_total > sell_total + 3:
            verdict = f"CUSTOM STRONG BUY ({buy_total}v{sell_total})"
        elif sell_total > buy_total + 3:
            verdict = f"CUSTOM STRONG SELL ({sell_total}v{buy_total})"
        elif buy_total > sell_total:
            verdict = f"CUSTOM LEAN BUY ({buy_total}v{sell_total})"
        elif sell_total > buy_total:
            verdict = f"CUSTOM LEAN SELL ({sell_total}v{buy_total})"
        else:
            verdict = "CUSTOM NEUTRAL"

        lines.insert(1, f"  Вердикт: {verdict}")
        summary = "\n".join(lines)

        return {
            "summary":   summary,
            "buy":       buy_total,
            "sell":      sell_total,
            "verdict":   verdict,
        }

    def train_dna(self, df: pd.DataFrame) -> int:
        """Обучает CandleDNA на переданном датафрейме."""
        n = self.candle_dna.train_on_history(df)
        print(f"[CandleDNA] Обучено {n} новых паттернов. {self.candle_dna.get_stats()}")
        return n

    def get_hourly_profile(
        self, df: pd.DataFrame
    ) -> dict:
        """Профиль по часам суток — для анализа времени сделок."""
        return self.tva.hourly_profile(df)


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

custom_signals = CustomSignalEngine()
