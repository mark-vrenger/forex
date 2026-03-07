"""
slow_trainer.py — Оффлайн-тренер v3.1 FAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ускорения v3.1:
  - Все sleep() убраны
  - Пауза охлаждения убрана (не нужна при обучении)
  - Аудитор работает в фоновом потоке (не блокирует цикл)
  - Dashboard обновляется каждые 20 итераций (не каждую)
  - State сохраняется каждые 20 итераций
  - candles_cache обновляется каждые 5 итераций
  - Пре-фильтр отсекает ~80% баров без вызова AI
"""

import time
import os
import json
import threading
import pandas as pd
from datetime import datetime
from collections import deque

from pro_trading_agent_pc import agent
from auditor_agent        import auditor
from custom_indicators    import custom_signals
from dashboard_generator  import generate_dashboard

STATE_FILE = "trainer_state.json"

# ─── Глобальные счётчики ────────────────
stats = {
    "buy": 0, "sell": 0, "wait": 0,
    "wins": 0, "losses": 0,
    "balance": 1000.0,
    "profit_sum": 0.0,
    "last_custom_text": "",
    "bars_skipped": 0,
    "bars_analyzed": 0,
}
all_trades    = []
council_log   = deque(maxlen=20)
audit_log     = deque(maxlen=20)
candles_cache = []

# ─── Фоновый аудит (не блокирует главный цикл) ──
_audit_lock   = threading.Lock()
_audit_queue  = deque(maxlen=50)

def _audit_worker():
    """Фоновый поток: аудирует сделки из очереди."""
    while True:
        if _audit_queue:
            trade_record = _audit_queue.popleft()
            try:
                result = auditor.audit_trade(trade_record)
                with _audit_lock:
                    # Обновляем запись в all_trades
                    for t in all_trades:
                        if t.get("id") == trade_record.get("id"):
                            t["trade_class"] = result.get("trade_class")
                            for line in result.get("ai_verdict", "").split("\n"):
                                if "ПРАВИЛО:" in line.upper():
                                    t["audit_rule"] = line.split(":", 1)[-1].strip()[:80]
                                    break
                            break
                    audit_log.append(result)
            except Exception as e:
                pass  # Аудит не критичен для обучения
        else:
            time.sleep(0.5)

# Запускаем фоновый поток аудита
_audit_thread = threading.Thread(target=_audit_worker, daemon=True)
_audit_thread.start()


# ═══════════════════════════════════════
#  СИМУЛЯЦИЯ СДЕЛКИ
# ═══════════════════════════════════════

def simulate_trade(df_future, direction, entry_price, logic_text):
    is_scalp = "SCALP" in logic_text.upper() or "СКАЛЬПИНГ" in logic_text.upper()

    if is_scalp:
        sl_dist, tp_dist   = 0.0010, 0.0020
        risk_money, reward = -10.0,  20.0
    else:
        sl_dist, tp_dist   = 0.0025, 0.0050
        risk_money, reward = -25.0,  50.0

    sl = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist
    tp = entry_price + tp_dist if direction == "BUY" else entry_price - tp_dist

    mfe = mae = 0.0
    point = 0.00001

    for candles_held, (_, candle) in enumerate(df_future.iterrows(), 1):
        h, l = candle["high"], candle["low"]
        if direction == "BUY":
            mfe = max(mfe, (h - entry_price) / point)
            mae = max(mae, (entry_price - l) / point)
            if l <= sl: return "LOSS", risk_money, mfe, mae, candles_held
            if h >= tp: return "WIN",  reward,     mfe, mae, candles_held
        else:
            mfe = max(mfe, (entry_price - l) / point)
            mae = max(mae, (h - entry_price) / point)
            if h >= sl: return "LOSS", risk_money, mfe, mae, candles_held
            if l <= tp: return "WIN",  reward,     mfe, mae, candles_held

    return "HOLD", 0.0, mfe, mae, len(df_future)


# ═══════════════════════════════════════
#  ЗАГРУЗКА CSV
# ═══════════════════════════════════════

def load_csv(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            header = f.readline()
        sep = "\t" if "\t" in header else ";" if ";" in header else ","
        df  = pd.read_csv(filename, sep=sep)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        if "date" in df.columns and "time" in df.columns:
            df["time"] = pd.to_datetime(df["date"] + " " + df["time"])
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"tickvol": "tick_volume", "vol": "real_volume"})
        if "tick_volume" not in df.columns: df["tick_volume"] = 100
        if "spread"      not in df.columns: df["spread"]      = 1.0
        if "real_volume" not in df.columns: df["real_volume"]  = 0
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"Ошибка файла {filename}: {e}")
        return None


def get_slice(df, target_time, count=50):
    if df is None:
        return None
    idx = df["time"].searchsorted(target_time, side="right")
    if idx == 0:
        return None
    return df.iloc[max(0, idx - count): idx]


# ═══════════════════════════════════════
#  БЫСТРЫЙ ПРЕ-ФИЛЬТР
# ═══════════════════════════════════════

def _quick_prefilter(m15_df, h1_df) -> bool:
    """
    Без вызова AI. Отсекает ~80% баров.
    True = есть смысл звать AI.
    """
    try:
        if m15_df is None or h1_df is None:
            return False
        if len(m15_df) < 20 or len(h1_df) < 20:
            return False

        # 1. Не флэт H1
        sma5  = h1_df["close"].tail(5).mean()
        sma15 = h1_df["close"].tail(15).mean()
        if abs(sma5 - sma15) / sma15 * 100 < 0.02:
            return False

        # 2. Объём не мёртвый
        avg_vol  = m15_df["tick_volume"].tail(20).mean()
        last_vol = m15_df["tick_volume"].iloc[-1]
        if avg_vol > 0 and last_vol < avg_vol * 0.5:
            return False

        # 3. RSI не нейтрален (42-58)
        delta = m15_df["close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))
        if 42 < rsi < 58:
            return False

        return True
    except Exception:
        return True


# ═══════════════════════════════════════
#  ДАШБОРД
# ═══════════════════════════════════════

def update_dashboard(idx, total, current_time):
    wins   = stats["wins"]
    losses = stats["losses"]
    total_t = wins + losses
    winrate = wins / total_t * 100 if total_t > 0 else 0
    pf_w = sum(t.get("profit", 0) for t in all_trades if (t.get("profit") or 0) > 0)
    pf_l = abs(sum(t.get("profit", 0) for t in all_trades if (t.get("profit") or 0) < 0))

    dash_stats = {
        "balance":       round(stats["balance"], 2),
        "wins":          wins,
        "losses":        losses,
        "winrate":       round(winrate, 1),
        "profit":        round(stats["profit_sum"], 2),
        "profit_factor": round(pf_w / pf_l, 2) if pf_l > 0 else 0,
    }

    from trade_memory import memory as mem_s
    time_profile = mem_s.get_time_profile(last_n=100) if len(all_trades) >= 5 else ""

    generate_dashboard(
        candles=candles_cache,
        trades=list(all_trades)[-50:],
        council_log=list(council_log),
        audit_log=list(audit_log),
        stats=dash_stats,
        time_profile=time_profile,
        alpha_context=auditor.get_alpha_context(),
        custom_verdict=stats.get("last_custom_text", ""),
        title=f"Тренер | {idx}/{total} | {current_time} | "
              f"AI:{stats['bars_analyzed']} skip:{stats['bars_skipped']}",
    )


# ═══════════════════════════════════════
#  ОСНОВНОЙ ЦИКЛ
# ═══════════════════════════════════════

def start_training():
    global candles_cache

    print("Загрузка CSV...")
    df_m1  = load_csv("M1.csv")
    df_m5  = load_csv("M5.csv")
    df_m15 = load_csv("M15.csv")
    df_m30 = load_csv("M30.csv")
    df_h1  = load_csv("H1.csv")
    df_h4  = load_csv("H4.csv")

    if df_m15 is None:
        print("ОШИБКА: M15.csv не найден!")
        return

    total_bars = len(df_m15)
    print(f"Загружено {total_bars} баров M15.")

    # Обучение CandleDNA один раз
    print("Обучение CandleDNA...")
    n = custom_signals.train_dna(df_m15)
    print(f"CandleDNA: {n} паттернов.")

    # Восстановление прогресса
    idx = 100
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
                idx                  = s.get("current_index", 100)
                stats["balance"]     = s.get("balance",     1000.0)
                stats["wins"]        = s.get("wins",        0)
                stats["losses"]      = s.get("losses",      0)
                stats["profit_sum"]  = s.get("profit_sum",  0.0)
        except Exception:
            pass

    print(f"Старт с бара {idx}. Открой dashboard.html в браузере.")
    update_dashboard(idx, total_bars, "Старт...")

    _iter  = 0          # счётчик итераций для throttling
    _t_start = time.time()

    while idx < total_bars - 100:

        row         = df_m15.iloc[idx]
        target_time = row["time"]
        dt_str      = target_time.strftime("%Y-%m-%d %H:%M")
        _iter      += 1

        # ── Кэш свечей — каждые 5 итераций ────
        if _iter % 5 == 0 and df_m15 is not None:
            raw = df_m15.iloc[max(0, idx-100): idx+1].copy()
            raw["time"] = raw["time"].apply(
                lambda t: int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
            )
            candles_cache = raw.to_dict("records")

        # ── Нарезка таймфреймов ─────────────────
        m1  = get_slice(df_m1,  target_time)
        m5  = get_slice(df_m5,  target_time)
        m15 = get_slice(df_m15, target_time)
        m30 = get_slice(df_m30, target_time)
        h1  = get_slice(df_h1,  target_time)
        h4  = get_slice(df_h4,  target_time)

        if h1 is None or len(h1) < 10:
            idx += 1
            continue

        m15_df = pd.DataFrame(m15) if m15 is not None else None
        h1_df  = pd.DataFrame(h1)
        h4_df  = pd.DataFrame(h4) if h4 is not None else None

        # ── Пре-фильтр ─────────────────────────
        if not _quick_prefilter(m15_df, h1_df):
            stats["bars_skipped"] += 1
            idx += 1
            # Дашборд при пропуске — каждые 100
            if _iter % 100 == 0:
                update_dashboard(idx, total_bars, dt_str)
                _save_state(idx)
            continue

        stats["bars_analyzed"] += 1
        price = row["close"]

        # ── Custom Indicators ───────────────────
        custom_result = custom_signals.analyze(m15_df, h1_df, h4_df)
        stats["last_custom_text"] = custom_result.get("summary", "")

        # ── AI Консилиум ────────────────────────
        ans            = agent.analyze_market(m1, m5, m15, m30, h1, h4, price, "Offline_CSV", 1.0)
        conf, d, logic = agent.parse_response(ans)

        # Корректировка от кастомных индикаторов
        adj = 0
        if d == "BUY"  and custom_result["buy"]  > custom_result["sell"] + 3: adj = +1
        if d == "SELL" and custom_result["sell"] > custom_result["buy"]  + 3: adj = +1
        final_conf = min(10, conf + adj)

        council_log.append({
            "time": dt_str, "signal": d,
            "confidence": final_conf, "votes": [],
            "reasoning": logic[:150],
        })

        # ── Сделка ─────────────────────────────
        if "BUY" in d or "SELL" in d:
            direction = "BUY" if "BUY" in d else "SELL"
            stats["buy" if direction == "BUY" else "sell"] += 1

            future = df_m15.iloc[idx + 1: idx + 101]
            result, pnl, mfe, mae, tte = simulate_trade(future, direction, price, logic)

            trade_record = {
                "id":          f"T_{dt_str.replace(' ', '_').replace(':', '')}_{idx}",
                "time":        target_time.isoformat(),
                "entry_time":  target_time.isoformat(),
                "direction":   direction,
                "entry_price": price,
                "sl":          price - 0.0025 if direction == "BUY" else price + 0.0025,
                "tp":          price + 0.0050 if direction == "BUY" else price - 0.0050,
                "confidence":  final_conf,
                "ai_logic":    logic[:200],
                "result":      result if result != "HOLD" else None,
                "profit":      pnl,
                "mfe_points":  round(mfe, 1),
                "mae_points":  round(mae, 1),
                "tte_candles": tte,
                "tte_minutes": tte * 15,
                "entry_hour":  target_time.hour,
                "entry_dow":   target_time.weekday(),
                "session":     _get_session(target_time.hour),
                "trade_class": None,
                "audit_rule":  None,
            }

            if result in ("WIN", "LOSS"):
                stats["balance"]    += pnl
                stats["profit_sum"] += pnl
                stats["wins" if result == "WIN" else "losses"] += 1
                # Аудит — в фоновый поток, НЕ блокируем цикл
                _audit_queue.append(trade_record)

            all_trades.append(trade_record)

            elapsed = time.time() - _t_start
            speed   = stats["bars_analyzed"] / elapsed if elapsed > 0 else 0
            print(
                f"[{dt_str}] {direction}(Ув:{final_conf}) "
                f"=> {result} ${pnl:.0f} | "
                f"MFE:{mfe:.0f}p MAE:{mae:.0f}p | "
                f"Скорость:{speed:.1f} бар/сек"
            )
        else:
            stats["wait"] += 1

        idx += 1

        # ── Сохранение и дашборд — каждые 20 итераций ──
        if _iter % 20 == 0:
            _save_state(idx)
            update_dashboard(idx, total_bars, dt_str)


def _save_state(idx):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "current_index": idx,
                "balance":       stats["balance"],
                "wins":          stats["wins"],
                "losses":        stats["losses"],
                "profit_sum":    stats["profit_sum"],
            }, f)
    except Exception:
        pass


def _get_session(hour: int) -> str:
    if 7  <= hour < 10: return "London Open"
    if 10 <= hour < 13: return "London"
    if 13 <= hour < 17: return "London+NY"
    if 17 <= hour < 21: return "New York"
    return "Asia"


if __name__ == "__main__":
    start_training()
