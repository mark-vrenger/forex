"""
slow_trainer.py — Исторический тренинг v2.0
Убран бесполезный sleep(3600). Добавлен прогресс-бар и запись в БД.
Поддержка --start-from аргумента командной строки.
"""

import MetaTrader5 as mt5
import time
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pro_trading_agent_pc import agent
from config import SYMBOL

STATE_FILE = "trainer_state.json"
BATCH_SIZE = 50       # баров на батч
PAUSE_BETWEEN = 0.5   # сек между запросами (чтобы не спамить API)


def log_tr(msg: str):
    print(f"[ОБУЧЕНИЕ {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    pct   = current / total if total > 0 else 0
    done  = int(pct * width)
    bar   = "█" * done + "░" * (width - done)
    return f"[{bar}] {pct*100:.1f}% ({current}/{total})"


def estimate_eta(start_time: float, current: int, total: int) -> str:
    elapsed = time.time() - start_time
    if current == 0:
        return "вычисляется..."
    rate    = current / elapsed          # баров в секунду
    remain  = (total - current) / rate   # секунд осталось
    h, rem  = divmod(int(remain), 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_state(idx: int, total: int, stats: dict):
    with open(STATE_FILE, "w") as f:
        json.dump({
            "current_index": idx,
            "total": total,
            "stats": stats,
            "last_update": datetime.now().isoformat(),
        }, f)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_index": 100, "total": 0, "stats": {}}


def start_training(start_from_date: str = None, start_from_idx: int = None):
    if not mt5.initialize():
        print("❌ MT5 не подключён")
        return

    log_tr("Загружаю историю (5 лет)...")
    rates = mt5.copy_rates_range(
        SYMBOL, mt5.TIMEFRAME_H1,
        datetime.now() - timedelta(days=1825),
        datetime.now()
    )
    if rates is None:
        log_tr("Ошибка: история не получена!")
        mt5.shutdown()
        return

    df_h1 = pd.DataFrame(rates)
    df_h1.columns = ['time', 'open', 'high', 'low', 'close',
                      'tick_volume', 'spread', 'real_volume'][:len(df_h1.columns)]
    total = len(df_h1)
    log_tr(f"Загружено {total} баров H1")

    # ── Определяем стартовый индекс ──────────
    if start_from_date:
        # Ищем ближайший бар к дате
        try:
            target = pd.to_datetime(start_from_date)
            df_h1['dt'] = pd.to_datetime(df_h1['time'], unit='s')
            diffs = abs(df_h1['dt'] - target)
            idx = int(diffs.idxmin())
            log_tr(f"Старт с даты {start_from_date} → индекс {idx}")
        except Exception as e:
            log_tr(f"Ошибка разбора даты: {e}. Используем сохранённое состояние.")
            idx = None
    elif start_from_idx is not None:
        idx = max(100, start_from_idx)
    else:
        state = load_state()
        idx = state.get("current_index", 100)

    if idx is None:
        state = load_state()
        idx = state.get("current_index", 100)

    idx = max(100, min(idx, total - 101))

    # ── Статистика ────────────────────────────
    stats = {
        "total_analyzed": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "wait_signals": 0,
        "errors": 0,
        "start_idx": idx,
        "start_time": datetime.now().isoformat(),
    }

    log_tr(f"Старт с бара {idx}/{total} | Осталось: {total - idx} баров")

    # ── Попытка импортировать БД ──────────────
    try:
        from database import db as _db
        use_db = _db is not None
    except Exception:
        use_db = False
        log_tr("БД недоступна — результаты только в логах")

    start_time = time.time()
    last_log_time = start_time

    try:
        while idx < total - 100:
            ts_val = int(df_h1.iloc[idx]['time'])

            # Получаем данные для анализа
            m5  = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M5,  ts_val, 50)
            h1  = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_H1,  ts_val, 50)
            m15 = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M15, ts_val, 50)

            if m5 is not None and h1 is not None:
                price = float(df_h1.iloc[idx]['close'])
                try:
                    raw  = agent.analyze_market(m5, m15, h1, None, price, "History", 1.0)
                    resp = agent.parse_response(raw)

                    stats["total_analyzed"] += 1
                    if resp.signal == "BUY":    stats["buy_signals"]  += 1
                    elif resp.signal == "SELL": stats["sell_signals"] += 1
                    else:                       stats["wait_signals"] += 1

                    # Запись в БД для последующей оценки точности
                    if use_db and resp.signal in ("BUY", "SELL"):
                        try:
                            _db.record_signal(
                                price=price,
                                direction=resp.signal,
                                confidence=resp.confidence,
                                final_conf=resp.confidence,
                                rsi=50, atr=0.001, spread=1.5,
                                session="History",
                                tech_buy=0, tech_sell=0, tech_summary="",
                                corr_buy=0, corr_sell=0,
                                corr_verdict="", corr_summary="",
                                sentiment_summary="",
                                corr_adj=0, tech_adj=0,
                                logic=resp.reasoning[:200],
                                acted=False,
                            )
                        except Exception:
                            pass

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        log_tr(f"Ошибка анализа: {e}")
            else:
                stats["errors"] += 1

            idx += 1

            # ── Прогресс каждые 10 секунд ──────
            now = time.time()
            if now - last_log_time >= 10:
                pb  = progress_bar(idx - stats["start_idx"],
                                   total - stats["start_idx"])
                eta = estimate_eta(start_time, idx - stats["start_idx"],
                                   total - stats["start_idx"])
                log_tr(f"{pb} | ETA: {eta} | "
                       f"B:{stats['buy_signals']} S:{stats['sell_signals']} "
                       f"W:{stats['wait_signals']} E:{stats['errors']}")
                last_log_time = now

            # ── Сохранение состояния каждые 100 баров ──
            if idx % 100 == 0:
                save_state(idx, total, stats)

            # Небольшая пауза между запросами к API
            time.sleep(PAUSE_BETWEEN)

    except KeyboardInterrupt:
        log_tr("Остановлено пользователем (Ctrl+C)")

    # Финальная статистика
    save_state(idx, total, stats)
    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)

    log_tr("=" * 50)
    log_tr(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    log_tr(f"Время: {h:02d}:{m:02d}:{s:02d}")
    log_tr(f"Проанализировано: {stats['total_analyzed']} баров")
    log_tr(f"Сигналы: BUY={stats['buy_signals']} SELL={stats['sell_signals']} WAIT={stats['wait_signals']}")
    log_tr(f"Ошибок: {stats['errors']}")
    log_tr("=" * 50)

    mt5.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Исторический тренинг бота")
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Дата начала обучения (например: 2024-01-01)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Начальный индекс бара (например: 500)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Сбросить состояние и начать с начала",
    )
    args = parser.parse_args()

    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("Состояние сброшено.")

    start_training(
        start_from_date=args.start_from,
        start_from_idx=args.start_idx,
    )
