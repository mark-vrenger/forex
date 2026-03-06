"""
signal_engine.py — Движок технических сигналов v2.0
Исправлен ADX (нет DeprecationWarning). Добавлены: Stochastic, Ichimoku cloud.
Добавлены get_for_agent() и get_*_summary() для Консилиума.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from config import SYMBOL


class SignalEngine:

    def analyze_all(self) -> tuple:
        """Полный анализ всех 10 индикаторов. Возвращает (summary, buy_score, sell_score)."""
        results = []
        buy_score = 0
        sell_score = 0
        for func in [
            self._ma_crossover,
            self._rsi_divergence,
            self._level_breakout,
            self._volume_analysis,
            self._candle_patterns,
            self._bollinger_bands,
            self._golden_death_cross,
            self._adx_strength,
            self._stochastic,
            self._ichimoku,
        ]:
            try:
                sig = func()
                if sig:
                    results.append(sig["text"])
                    buy_score += sig["buy"]
                    sell_score += sig["sell"]
            except Exception:
                continue

        total = buy_score + sell_score
        if total == 0:
            verdict = "NEUTRAL"
        elif buy_score > sell_score + 2:
            verdict = f"STRONG BUY ({buy_score}v{sell_score})"
        elif sell_score > buy_score + 2:
            verdict = f"STRONG SELL ({sell_score}v{buy_score})"
        elif buy_score > sell_score:
            verdict = f"LEAN BUY ({buy_score}v{sell_score})"
        elif sell_score > buy_score:
            verdict = f"LEAN SELL ({sell_score}v{buy_score})"
        else:
            verdict = f"MIXED ({buy_score}v{sell_score})"

        summary = f"-- TECH SIGNALS (10 indicators) --\nVerdict: {verdict}\n"
        summary += "\n".join(results) if results else "  No signals"
        return summary, buy_score, sell_score

    def get_for_agent(self, agent_name: str) -> str:
        """
        Возвращает только релевантные индикаторы для конкретного агента Консилиума.
        agent_name: 'GROK' | 'CONSERVATIVE' | 'ANALYST'
        """
        if agent_name == "IMPULSE":
            # IMPULSE: агрессивный скальпер
            parts = []
            for func in [self._candle_patterns, self._volume_analysis,
                         self._bollinger_bands, self._stochastic]:
                try:
                    sig = func()
                    if sig:
                        parts.append(sig["text"])
                except Exception:
                    pass
            return "\n".join(parts) if parts else "No scalping signals"

        elif agent_name == "TREND":
            # TREND: консерватор, тренд
            parts = []
            for func in [self._ma_crossover, self._golden_death_cross,
                         self._adx_strength, self._ichimoku]:
                try:
                    sig = func()
                    if sig:
                        parts.append(sig["text"])
                except Exception:
                    pass
            return "\n".join(parts) if parts else "No trend signals"

        elif agent_name == "ANALYST":
            # Аналитик: Price Action и дивергенции
            parts = []
            for func in [self._rsi_divergence, self._level_breakout,
                         self._bollinger_bands, self._candle_patterns]:
                try:
                    sig = func()
                    if sig:
                        parts.append(sig["text"])
                except Exception:
                    pass
            return "\n".join(parts) if parts else "No PA signals"

        else:
            summary, _, _ = self.analyze_all()
            return summary

    def get_m15_summary(self) -> str:
        """Быстрый агрегат по M15 для Консилиума."""
        df = self._get_data(tf=mt5.TIMEFRAME_M15, count=50)
        if df is None:
            return "M15: нет данных"
        sma5 = df['close'].tail(5).mean()
        sma15 = df['close'].tail(15).mean()
        trend = "UP" if sma5 > sma15 else "DOWN"
        avg_v = df['tick_volume'].tail(20).mean()
        last_v = df['tick_volume'].iloc[-1]
        ratio = last_v / avg_v if avg_v > 0 else 1.0
        return f"M15: trend={trend} vol_ratio={ratio:.2f} close={df['close'].iloc[-1]:.5f}"

    def get_h4_summary(self) -> str:
        """Быстрый агрегат по H4 для Консилиума."""
        df = self._get_data(tf=mt5.TIMEFRAME_H4, count=50)
        if df is None:
            return "H4: нет данных"
        sma5 = df['close'].tail(5).mean()
        sma15 = df['close'].tail(15).mean()
        trend = "UP" if sma5 > sma15 else "DOWN"
        return f"H4: trend={trend} close={df['close'].iloc[-1]:.5f} sma5={sma5:.5f} sma15={sma15:.5f}"

    def get_confidence_adjustment(self, ai_direction: str) -> int:
        _, bs, ss = self.analyze_all()
        if ai_direction == "BUY":
            if bs > ss + 3:  return +1
            elif ss > bs + 3: return -1
        elif ai_direction == "SELL":
            if ss > bs + 3:  return +1
            elif bs > ss + 3: return -1
        return 0

    # ── DATA FETCHER ─────────────────────────────

    def _get_data(self, tf=None, count=200) -> pd.DataFrame | None:
        if tf is None:
            tf = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, count)
        if rates is None or len(rates) < 50:
            return None
        df = pd.DataFrame(rates)
        if 'close' not in df.columns and len(df.columns) >= 5:
            df.columns = ['time', 'open', 'high', 'low', 'close',
                          'tick_volume', 'spread', 'real_volume'][:len(df.columns)]
        return df

    # ── INDICATORS ──────────────────────────────

    def _ma_crossover(self):
        df = self._get_data()
        if df is None: return None
        df['sma10'] = df['close'].rolling(10).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        p, c = df.iloc[-2], df.iloc[-1]
        buy, sell, sigs = 0, 0, []
        if p['sma10'] <= p['sma20'] and c['sma10'] > c['sma20']:
            sigs.append("SMA10>SMA20 (BUY)"); buy += 2
        elif p['sma10'] >= p['sma20'] and c['sma10'] < c['sma20']:
            sigs.append("SMA10<SMA20 (SELL)"); sell += 2
        if c['close'] > c['sma50']:
            sigs.append("Above SMA50"); buy += 1
        else:
            sigs.append("Below SMA50"); sell += 1
        if not sigs: return None
        return {"text": "  MA: " + " | ".join(sigs), "buy": buy, "sell": sell}

    def _rsi_divergence(self):
        df = self._get_data()
        if df is None: return None
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        r = df.tail(20).dropna()
        if len(r) < 10: return None
        pl1, pl2 = r['close'].iloc[:10].min(), r['close'].iloc[10:].min()
        rl1, rl2 = r['rsi'].iloc[:10].min(), r['rsi'].iloc[10:].min()
        ph1, ph2 = r['close'].iloc[:10].max(), r['close'].iloc[10:].max()
        rh1, rh2 = r['rsi'].iloc[:10].max(), r['rsi'].iloc[10:].max()
        if pl2 < pl1 and rl2 > rl1:
            return {"text": "  DIVERGENCE: Bullish (STRONG BUY)", "buy": 3, "sell": 0}
        elif ph2 > ph1 and rh2 < rh1:
            return {"text": "  DIVERGENCE: Bearish (STRONG SELL)", "buy": 0, "sell": 3}
        return None

    def _level_breakout(self):
        df = self._get_data(count=100)
        if df is None: return None
        h = df.iloc[-50:-5]
        res = h['high'].max()
        sup = h['low'].min()
        cur = df['close'].iloc[-1]
        prv = df['close'].iloc[-2]
        sym = mt5.symbol_info(SYMBOL)
        pt = sym.point if sym else 0.00001
        if prv < res and cur > res:
            p = (cur - res) / pt
            return {"text": f"  BREAKOUT: Resistance {res:.5f} (+{p:.0f}p) = BUY", "buy": 2, "sell": 0}
        elif prv > sup and cur < sup:
            p = (sup - cur) / pt
            return {"text": f"  BREAKOUT: Support {sup:.5f} (-{p:.0f}p) = SELL", "buy": 0, "sell": 2}
        dr = (res - cur) / pt
        ds = (cur - sup) / pt
        return {"text": f"  LEVELS: R:{res:.5f}({dr:.0f}p) S:{sup:.5f}({ds:.0f}p)", "buy": 0, "sell": 0}

    def _volume_analysis(self):
        df = self._get_data()
        if df is None: return None
        avg = df['tick_volume'].tail(20).mean()
        last = df['tick_volume'].iloc[-1]
        ratio = last / avg if avg > 0 else 1
        if ratio > 2.0:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                return {"text": f"  VOLUME: Spike x{ratio:.1f} bullish", "buy": 2, "sell": 0}
            else:
                return {"text": f"  VOLUME: Spike x{ratio:.1f} bearish", "buy": 0, "sell": 2}
        return {"text": f"  VOLUME: x{ratio:.1f}", "buy": 0, "sell": 0}

    def _candle_patterns(self):
        df = self._get_data(tf=mt5.TIMEFRAME_M15, count=10)
        if df is None or len(df) < 3: return None
        c2, c3 = df.iloc[-2], df.iloc[-1]
        b2 = abs(c2['close'] - c2['open'])
        b3 = abs(c3['close'] - c3['open'])
        buy, sell, pats = 0, 0, []
        # Bullish Engulfing
        if (c2['close'] < c2['open'] and c3['close'] > c3['open']
                and c3['open'] <= c2['close'] and c3['close'] >= c2['open'] and b3 > b2):
            pats.append("Bullish Engulfing"); buy += 2
        # Bearish Engulfing
        if (c2['close'] > c2['open'] and c3['close'] < c3['open']
                and c3['open'] >= c2['close'] and c3['close'] <= c2['open'] and b3 > b2):
            pats.append("Bearish Engulfing"); sell += 2
        # Hammer
        sl = min(c3['open'], c3['close']) - c3['low']
        sh = c3['high'] - max(c3['open'], c3['close'])
        if b3 > 0 and sl > b3 * 2 and sh < b3 * 0.5:
            pats.append("Hammer"); buy += 1
        if b3 > 0 and sh > b3 * 2 and sl < b3 * 0.5:
            pats.append("Shooting Star"); sell += 1
        if not pats: return None
        return {"text": "  PATTERNS: " + ", ".join(pats), "buy": buy, "sell": sell}

    def _bollinger_bands(self):
        df = self._get_data()
        if df is None: return None
        df['sma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['upper'] = df['sma20'] + 2 * df['std20']
        df['lower'] = df['sma20'] - 2 * df['std20']
        l = df.iloc[-1]
        if l['close'] <= l['lower']:
            return {"text": "  BB: At LOWER band (oversold) = BUY", "buy": 2, "sell": 0}
        elif l['close'] >= l['upper']:
            return {"text": "  BB: At UPPER band (overbought) = SELL", "buy": 0, "sell": 2}
        w = (l['upper'] - l['lower']) / l['sma20'] * 100
        if w < 0.3:
            return {"text": f"  BB: Squeeze ({w:.2f}%) — breakout soon", "buy": 0, "sell": 0}
        return {"text": f"  BB: In range. W:{w:.2f}%", "buy": 0, "sell": 0}

    def _golden_death_cross(self):
        df = self._get_data(tf=mt5.TIMEFRAME_H4, count=200)
        if df is None or len(df) < 200: return None
        df['sma50']  = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        c, p = df.iloc[-1], df.iloc[-2]
        if p['sma50'] <= p['sma200'] and c['sma50'] > c['sma200']:
            return {"text": "  CROSS: Golden Cross H4 = STRONG BUY", "buy": 3, "sell": 0}
        elif p['sma50'] >= p['sma200'] and c['sma50'] < c['sma200']:
            return {"text": "  CROSS: Death Cross H4 = STRONG SELL", "buy": 0, "sell": 3}
        if c['sma50'] > c['sma200']:
            return {"text": "  CROSS: SMA50>SMA200 bullish", "buy": 1, "sell": 0}
        return {"text": "  CROSS: SMA50<SMA200 bearish", "buy": 0, "sell": 1}

    def _adx_strength(self):
        """ADX — исправлен без DeprecationWarning (нет iloc-присваиваний)."""
        df = self._get_data(count=100)
        if df is None or len(df) < 30: return None
        period = 14

        high  = df['high'].values
        low   = df['low'].values
        close = df['close'].values
        n     = len(high)

        plus_dm  = np.zeros(n)
        minus_dm = np.zeros(n)
        tr_arr   = np.zeros(n)

        for i in range(1, n):
            up   = high[i]  - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i]  = up   if up > down and up > 0 else 0.0
            minus_dm[i] = down if down > up and down > 0 else 0.0
            tr_arr[i]   = max(
                high[i] - low[i],
                abs(high[i]  - close[i - 1]),
                abs(low[i]   - close[i - 1]),
            )

        # Wilder smoothing (EMA-like with period)
        def wilder(arr, p):
            out = np.zeros(len(arr))
            out[p] = arr[1:p + 1].sum()
            for i in range(p + 1, len(arr)):
                out[i] = out[i - 1] - out[i - 1] / p + arr[i]
            return out

        tr_s  = wilder(tr_arr,   period)
        pdm_s = wilder(plus_dm,  period)
        mdm_s = wilder(minus_dm, period)

        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di  = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
            minus_di = np.where(tr_s > 0, 100 * mdm_s / tr_s, 0.0)
            dx = np.where(
                (plus_di + minus_di) > 0,
                100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
                0.0,
            )

        adx_arr = wilder(dx, period)
        last_adx   = adx_arr[-1]
        last_plus  = plus_di[-1]
        last_minus = minus_di[-1]

        if np.isnan(last_adx) or last_adx == 0:
            return None

        buy, sell = 0, 0
        if last_adx > 25:
            strength = "STRONG"
            buy, sell = (2, 0) if last_plus > last_minus else (0, 2)
        elif last_adx > 20:
            strength = "MODERATE"
            buy, sell = (1, 0) if last_plus > last_minus else (0, 1)
        else:
            strength = "WEAK/FLAT"

        trend_dir = "Bullish" if last_plus > last_minus else "Bearish"
        return {
            "text": (f"  ADX: {last_adx:.1f} ({strength}) {trend_dir} | "
                     f"+DI:{last_plus:.1f} -DI:{last_minus:.1f}"),
            "buy": buy, "sell": sell,
        }

    def _stochastic(self):
        """Stochastic Oscillator %K/%D — нужен агрессивному агенту."""
        df = self._get_data(tf=mt5.TIMEFRAME_M15, count=50)
        if df is None or len(df) < 20: return None
        period_k, period_d = 14, 3
        low_min  = df['low'].rolling(period_k).min()
        high_max = df['high'].rolling(period_k).max()
        rng = high_max - low_min
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.where(rng > 0, (df['close'] - low_min) / rng * 100, 50.0)
        k_series = pd.Series(k)
        d_series = k_series.rolling(period_d).mean()
        last_k = float(k_series.iloc[-1])
        last_d = float(d_series.iloc[-1])
        prev_k = float(k_series.iloc[-2])
        prev_d = float(d_series.iloc[-2])
        buy, sell = 0, 0
        note = ""
        if last_k < 20 and last_d < 20:
            note = "Oversold"
            buy += 2
        elif last_k > 80 and last_d > 80:
            note = "Overbought"
            sell += 2
        # Crossover
        if prev_k <= prev_d and last_k > last_d and last_k < 50:
            note += " BullCross"
            buy += 1
        elif prev_k >= prev_d and last_k < last_d and last_k > 50:
            note += " BearCross"
            sell += 1
        return {
            "text": f"  STOCH: %K:{last_k:.1f} %D:{last_d:.1f} {note}",
            "buy": buy, "sell": sell,
        }

    def _ichimoku(self):
        """Упрощённое облако Ишимоку — нужно консервативному агенту."""
        df = self._get_data(tf=mt5.TIMEFRAME_H1, count=60)
        if df is None or len(df) < 52: return None
        # Tenkan-sen (9), Kijun-sen (26), Senkou A/B (52)
        tenkan = (df['high'].rolling(9).max()  + df['low'].rolling(9).min())  / 2
        kijun  = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        span_a = (tenkan + kijun) / 2
        span_b = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
        price = df['close'].iloc[-1]
        tk = tenkan.iloc[-1]
        kj = kijun.iloc[-1]
        sa = span_a.iloc[-1]
        sb = span_b.iloc[-1]
        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)
        buy, sell = 0, 0
        notes = []
        if price > cloud_top:
            notes.append("Above cloud"); buy += 2
        elif price < cloud_bot:
            notes.append("Below cloud"); sell += 2
        else:
            notes.append("Inside cloud")
        if tk > kj:
            notes.append("TK>KJ↑"); buy += 1
        else:
            notes.append("TK<KJ↓"); sell += 1
        return {
            "text": f"  ICHIMOKU: {' | '.join(notes)} Price:{price:.5f}",
            "buy": buy, "sell": sell,
        }


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

signals = SignalEngine()
