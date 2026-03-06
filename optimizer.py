"""
optimizer.py — Оптимизатор параметров v2.0
Добавлена оптимизация весов агентов Консилиума.
Добавлен analyze_agent_accuracy() и find_consensus_threshold().
Исправлен битый символ в run_full().
"""

import pandas as pd
import numpy as np
from itertools import product
from database import db
from datetime import datetime
import os
import json


def header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ─────────────────────────────────────────
#  LAYER WEIGHT OPTIMIZER (оригинал)
# ─────────────────────────────────────────

def simulate_with_weights(signals_df, trades_df,
                          corr_weight, tech_weight,
                          min_confidence):
    """Симуляция с заданными весами по истории сигналов/сделок."""
    profits_list = []
    trades_taken = 0

    for _, sig in signals_df.iterrows():
        direction = sig['direction']
        raw_conf  = sig.get('confidence', 0) or 0
        corr_adj  = sig.get('corr_adj', 0) or 0
        tech_adj  = sig.get('tech_adj', 0) or 0
        sig_id    = sig.get('id', None)

        adj_conf = raw_conf + (corr_adj * corr_weight) + (tech_adj * tech_weight)
        adj_conf = max(1, min(10, int(adj_conf)))

        if direction == "WAIT" or adj_conf < min_confidence:
            continue

        # Поиск сделки по signal_id
        if sig_id is not None and 'signal_id' in trades_df.columns:
            matching = trades_df[
                (trades_df['signal_id'] == sig_id) &
                (trades_df['result'].notna())
            ]
            if len(matching) > 0:
                profit = matching.iloc[0].get('profit', 0) or 0
                profits_list.append(profit)
                trades_taken += 1
                continue

        # Fallback: поиск по времени (5 мин окно)
        sig_time = sig.get('time', '')
        if not sig_time:
            continue
        try:
            sig_dt = pd.to_datetime(sig_time)
            matching = trades_df[
                (trades_df['direction'] == direction) &
                (trades_df['result'].notna())
            ].copy()
            if len(matching) == 0:
                continue
            matching['time_dt'] = pd.to_datetime(matching['time'])
            matching['diff'] = abs((matching['time_dt'] - sig_dt).dt.total_seconds())
            closest = matching.loc[matching['diff'].idxmin()]
            if closest['diff'] > 300:
                continue
            profit = closest.get('profit', 0) or 0
            profits_list.append(profit)
            trades_taken += 1
        except Exception:
            continue

    if trades_taken == 0:
        return {"profit": 0, "trades": 0, "winrate": 0,
                "profit_factor": 0, "score": -999}

    wins = len([p for p in profits_list if p > 0])
    total_profit = sum(profits_list)
    winrate = wins / trades_taken * 100
    total_gain = sum(p for p in profits_list if p > 0)
    total_loss = abs(sum(p for p in profits_list if p < 0))
    pf = total_gain / total_loss if total_loss > 0 else 0
    score = total_profit * (winrate / 100) * (trades_taken ** 0.5)

    return {
        "profit": round(total_profit, 2),
        "trades": trades_taken,
        "wins": wins,
        "losses": trades_taken - wins,
        "winrate": round(winrate, 1),
        "profit_factor": round(pf, 2),
        "score": round(score, 2),
    }


def optimize_weights(auto_apply: bool = False):
    header("LAYER WEIGHT OPTIMIZER")

    signals_df = db.get_signals_df()
    trades_df  = db.get_trades_df()

    if len(signals_df) == 0:
        print("  Нет сигналов. Запустите бота сначала.")
        return None

    closed = trades_df[trades_df['result'].notna()]
    if len(closed) < 5:
        print(f"  Только {len(closed)} сделок. Нужно минимум 5.")
        return None

    print(f"  Загружено {len(signals_df)} сигналов, {len(closed)} сделок")

    corr_weights   = [0.0, 0.5, 1.0, 1.5, 2.0]
    tech_weights   = [0.0, 0.5, 1.0, 1.5, 2.0]
    min_confidences = [6, 7, 8, 9]

    total_combos = len(corr_weights) * len(tech_weights) * len(min_confidences)
    print(f"  Тестирую {total_combos} комбинаций...")

    results = []
    best_score  = -999
    best_params = None

    for cw, tw, mc in product(corr_weights, tech_weights, min_confidences):
        metrics = simulate_with_weights(signals_df, trades_df, cw, tw, mc)
        metrics['corr_weight'] = cw
        metrics['tech_weight'] = tw
        metrics['min_conf']    = mc
        results.append(metrics)
        if metrics['score'] > best_score and metrics['trades'] >= 3:
            best_score  = metrics['score']
            best_params = metrics

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    header("TOP 10 КОМБИНАЦИЙ")
    print(f"  {'CW':>5} {'TW':>5} {'MC':>4} {'Сделок':>7} {'WR%':>6} {'P/L':>10} {'PF':>6} {'Score':>10}")
    print("  " + "-" * 55)
    for r in results[:10]:
        if r['trades'] == 0:
            continue
        print(
            f"  {r['corr_weight']:5.1f} {r['tech_weight']:5.1f} "
            f"{r['min_conf']:4d} {r['trades']:7d} "
            f"{r['winrate']:6.1f} ${r['profit']:9.2f} "
            f"{r['profit_factor']:6.2f} {r['score']:10.2f}"
        )

    if not best_params:
        print("\n  Нет валидных комбинаций.")
        return None

    header("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ")
    print(f"""
  Вес корреляций:  {best_params['corr_weight']}
  Вес техсигналов: {best_params['tech_weight']}
  Мин. уверенность: {best_params['min_conf']}
  Ожидаемо:
    Сделок:        {best_params['trades']}
    Winrate:       {best_params['winrate']}%
    Прибыль:       ${best_params['profit']:.2f}
    Profit factor: {best_params['profit_factor']}
    """)

    from config import MIN_CONFIDENCE, CORR_WEIGHT, TECH_WEIGHT
    current = simulate_with_weights(
        signals_df, trades_df, CORR_WEIGHT, TECH_WEIGHT, MIN_CONFIDENCE
    )
    print(f"  Текущие (cw={CORR_WEIGHT}, tw={TECH_WEIGHT}, mc={MIN_CONFIDENCE}):")
    print(f"    Сделок:{current['trades']} WR:{current['winrate']}% P/L:${current['profit']:.2f}")

    improvement = best_params['profit'] - current.get('profit', 0)
    if improvement > 0:
        print(f"\n  >>> ПОТЕНЦИАЛЬНОЕ УЛУЧШЕНИЕ: +${improvement:.2f}")
    else:
        print(f"\n  Текущие настройки уже оптимальны!")

    db.record_optimization(
        best_params['corr_weight'], best_params['tech_weight'],
        best_params['min_conf'], best_params['profit'],
        best_params['winrate'], best_params['trades'],
        auto_apply and improvement > 0
    )

    if auto_apply and improvement > 0:
        apply_params(
            best_params['corr_weight'],
            best_params['tech_weight'],
            best_params['min_conf'],
        )

    return best_params


def apply_params(corr_weight, tech_weight, min_conf):
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith('MIN_CONFIDENCE'):
                new_lines.append(f'MIN_CONFIDENCE = {min_conf}\n')
            elif line.startswith('CORR_WEIGHT'):
                new_lines.append(f'CORR_WEIGHT = {corr_weight}\n')
            elif line.startswith('TECH_WEIGHT'):
                new_lines.append(f'TECH_WEIGHT = {tech_weight}\n')
            else:
                new_lines.append(line)
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(f"\n  config.py ОБНОВЛЁН:")
        print(f"     MIN_CONFIDENCE = {min_conf}")
        print(f"     CORR_WEIGHT = {corr_weight}")
        print(f"     TECH_WEIGHT = {tech_weight}")
        print(f"     Перезапустите бота для применения.")
    except Exception as e:
        print(f"\n  Ошибка обновления config: {e}")


# ─────────────────────────────────────────
#  AGENT WEIGHTS OPTIMIZER (Консилиум)
# ─────────────────────────────────────────

def optimize_agent_weights():
    """
    Подбирает оптимальные веса голосов агентов Консилиума
    на основе истории council_sessions.
    """
    header("ОПТИМИЗАЦИЯ ВЕСОВ АГЕНТОВ КОНСИЛИУМА")

    sessions_df = db.get_council_sessions_df()
    if sessions_df is None or len(sessions_df) == 0:
        print("  Нет сессий Консилиума. Нужно накопить историю.")
        return None

    completed = sessions_df[sessions_df['trade_result'].notna()].copy()
    if len(completed) < 5:
        print(f"  Только {len(completed)} завершённых сделок. Нужно минимум 5.")
        return None

    print(f"  Загружено {len(completed)} завершённых сессий")

    agent_names = ["IMPULSE", "TREND", "ANALYST"]
    weight_variants = [0.5, 1.0, 1.5, 2.0]

    best_score  = -999
    best_params = None
    results     = []

    for gw, cw, aw in product(weight_variants, weight_variants, weight_variants):
        wins = 0
        losses = 0
        total_profit = 0.0
        trades_taken = 0

        for _, row in completed.iterrows():
            try:
                votes = json.loads(row['votes_r1'])
            except Exception:
                continue

            # Взвешенное голосование
            score = {"BUY": 0.0, "SELL": 0.0, "WAIT": 0.0}
            w_map = {"IMPULSE": gw, "TREND": cw, "ANALYST": aw}
            for v in votes:
                name   = v.get("name", "")
                signal = v.get("signal", "WAIT")
                conf   = v.get("confidence", 0) or 0
                w      = w_map.get(name, 1.0)
                s      = signal if signal in score else "WAIT"
                score[s] += conf * w

            best_sig = max(score, key=score.get)
            if best_sig == "WAIT" or score[best_sig] == 0:
                continue

            # Порог: итоговый взвешенный скор должен быть > 35
            if score[best_sig] < 35:
                continue

            trades_taken += 1
            result = row['trade_result']
            profit = row.get('trade_profit', 0) or 0
            total_profit += profit
            if result == "WIN":
                wins += 1
            else:
                losses += 1

        if trades_taken == 0:
            continue

        winrate = wins / trades_taken * 100
        total_gain = 0.0
        total_loss = 0.0
        for _, row in completed.iterrows():
            p = row.get('trade_profit', 0) or 0
            if p > 0: total_gain += p
            else: total_loss += abs(p)
        pf = total_gain / total_loss if total_loss > 0 else 0
        score_val = total_profit * (winrate / 100) * (trades_taken ** 0.5)

        r = {
            "impulse_w": gw, "trend_w": cw, "analyst_w": aw,
            "trades": trades_taken, "wins": wins, "losses": losses,
            "winrate": round(winrate, 1),
            "profit": round(total_profit, 2),
            "profit_factor": round(pf, 2),
            "score": round(score_val, 2),
        }
        results.append(r)
        if score_val > best_score:
            best_score  = score_val
            best_params = r

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    header("TOP 10 ВЕСОВ АГЕНТОВ")
    print(f"  {'IMPULSE':>5} {'CONS':>5} {'ANAL':>5} {'Сделок':>7} {'WR%':>6} {'P/L':>10} {'Score':>10}")
    print("  " + "-" * 50)
    for r in results[:10]:
        print(
            f"  {r['impulse_w']:5.1f} {r['trend_w']:5.1f} {r['analyst_w']:5.1f} "
            f"{r['trades']:7d} {r['winrate']:6.1f} ${r['profit']:9.2f} {r['score']:10.2f}"
        )

    if best_params:
        header("ОПТИМАЛЬНЫЕ ВЕСА АГЕНТОВ")
        print(f"  IMPULSE:      {best_params['impulse_w']}")
        print(f"  TREND:        {best_params['trend_w']}")
        print(f"  ANALYST:      {best_params['analyst_w']}")
        print(f"  Сделок: {best_params['trades']} | WR: {best_params['winrate']}% | P/L: ${best_params['profit']:.2f}")

    return best_params


def analyze_agent_accuracy():
    """Показывает кто из агентов Консилиума чаще прав."""
    header("ТОЧНОСТЬ АГЕНТОВ КОНСИЛИУМА")

    accuracy = db.get_agent_accuracy()
    if not accuracy:
        print("  Нет данных. Нужны завершённые сессии Консилиума.")
        return

    print(f"\n  {'Агент':<16} {'Точность':>9} {'Правильно':>10} {'Всего':>7}")
    print("  " + "-" * 45)

    for name, stats in sorted(accuracy.items(), key=lambda x: -x[1]['accuracy']):
        acc = stats['accuracy']
        mark = " 🥇" if acc >= 65 else " 🥈" if acc >= 55 else " ⚠️" if acc < 45 else ""
        print(f"  {name:<16} {acc:>8.1f}% {stats['correct']:>10} {stats['total']:>7}{mark}")


def find_consensus_threshold():
    """
    Находит оптимальный порог консенсуса (% голосов 'за')
    для открытия сделки.
    """
    header("ОПТИМАЛЬНЫЙ ПОРОГ КОНСЕНСУСА")

    sessions_df = db.get_council_sessions_df()
    if sessions_df is None or len(sessions_df) == 0:
        print("  Нет данных.")
        return

    completed = sessions_df[sessions_df['trade_result'].notna()].copy()
    if len(completed) < 5:
        print("  Мало данных.")
        return

    print(f"\n  {'Порог%':>8} {'Сделок':>7} {'WR%':>6} {'P/L':>10}")
    print("  " + "-" * 35)

    for threshold in [50, 60, 67, 75, 100]:
        subset = completed[completed['consensus_pct'] >= threshold]
        if len(subset) == 0:
            continue
        wins = (subset['trade_result'] == 'WIN').sum()
        wr   = wins / len(subset) * 100
        pnl  = subset['trade_profit'].sum()
        mark = " <-- ЛУЧШИЙ" if 67 <= threshold <= 75 else ""
        print(f"  {threshold:>7}% {len(subset):>7} {wr:>5.1f}% ${pnl:>9.2f}{mark}")


# ─────────────────────────────────────────
#  LAYER ACCURACY (оригинал)
# ─────────────────────────────────────────

def analyze_layer_accuracy():
    header("ТОЧНОСТЬ СЛОЁВ")
    trades_df = db.get_trades_df()
    closed = trades_df[trades_df['result'].notna()]
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    tech_correct, tech_total = 0, 0
    corr_correct, corr_total = 0, 0
    ai_total, ai_correct     = 0, 0

    for _, t in closed.iterrows():
        direction = t['direction']
        result    = t['result']
        tb = t.get('tech_buy', 0) or 0
        ts = t.get('tech_sell', 0) or 0
        cb = t.get('corr_buy', 0) or 0
        cs = t.get('corr_sell', 0) or 0

        if tb != ts:
            tech_total += 1
            tech_pred = "BUY" if tb > ts else "SELL"
            if (tech_pred == direction and result == "WIN") or \
               (tech_pred != direction and result == "LOSS"):
                tech_correct += 1
        if cb != cs:
            corr_total += 1
            corr_pred = "BUY" if cb > cs else "SELL"
            if (corr_pred == direction and result == "WIN") or \
               (corr_pred != direction and result == "LOSS"):
                corr_correct += 1
        ai_total += 1
        if result == "WIN":
            ai_correct += 1

    print(f"\n  Точность предсказаний слоёв:\n")
    if tech_total > 0:
        print(f"  📈 Технический: {tech_correct/tech_total*100:.1f}% ({tech_correct}/{tech_total})")
    if corr_total > 0:
        print(f"  🔗 Корреляции: {corr_correct/corr_total*100:.1f}% ({corr_correct}/{corr_total})")
    if ai_total > 0:
        print(f"  🧠 AI (Консилиум): {ai_correct/ai_total*100:.1f}% ({ai_correct}/{ai_total})")


def find_best_conditions():
    header("ЛУЧШИЕ УСЛОВИЯ ДЛЯ ТОРГОВЛИ")
    trades_df = db.get_trades_df()
    closed = trades_df[trades_df['result'].notna()].copy()
    if len(closed) < 10:
        print("  Нужно 10+ сделок.")
        return

    for col in ['profit', 'rsi', 'atr', 'spread', 'confidence']:
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors='coerce')

    wins   = closed[closed['result'] == 'WIN']
    losses = closed[closed['result'] == 'LOSS']

    print(f"\n  {'Метрика':<15} {'WIN':>10} {'LOSS':>10} {'DIFF':>10}")
    print(f"  {'-'*45}")

    for col in ['confidence', 'rsi', 'atr', 'spread', 'tech_buy', 'tech_sell', 'corr_buy', 'corr_sell']:
        if col not in closed.columns:
            continue
        w_val = pd.to_numeric(wins[col],   errors='coerce').mean()
        l_val = pd.to_numeric(losses[col], errors='coerce').mean()
        if pd.isna(w_val) or pd.isna(l_val):
            continue
        diff = w_val - l_val
        mark = "  OK" if abs(diff) > 0.5 else ""
        print(f"  {col:<15} {w_val:>10.2f} {l_val:>10.2f} {diff:>+10.2f}{mark}")

    if 'session' in closed.columns:
        print(f"\n  По сессиям:")
        for s in closed['session'].unique():
            sub = closed[closed['session'] == s]
            if len(sub) >= 2:
                wr  = (sub['result'] == 'WIN').mean() * 100
                pnl = sub['profit'].sum()
                mark = " ЛУЧШИЙ" if wr > 60 else " ПЛОХОЙ" if wr < 40 else ""
                print(f"    {s:<25} WR:{wr:5.1f}% P/L:${pnl:8.2f}{mark}")


def run_auto_optimization() -> bool:
    """Вызывается из главного цикла бота."""
    from config import AUTO_OPTIMIZE, OPTIMIZE_INTERVAL_DAYS, OPTIMIZE_MIN_TRADES
    if not AUTO_OPTIMIZE:
        return False
    last_opt = db.get_last_optimization_date()
    if last_opt:
        days_since = (datetime.now() - last_opt).days
        if days_since < OPTIMIZE_INTERVAL_DAYS:
            return False
    trade_count = db.get_closed_trades_count()
    if trade_count < OPTIMIZE_MIN_TRADES:
        return False
    print("\n[АВТО-ОПТИМИЗАЦИЯ] Запуск еженедельной оптимизации...")
    result = optimize_weights(auto_apply=True)
    return result is not None


def run_full():
    print("""
    +==========================================+
    |    OPTIMIZER + ANALYTICS v2.0           |
    +==========================================+
    """)
    optimize_weights(auto_apply=False)
    optimize_agent_weights()
    analyze_layer_accuracy()
    analyze_agent_accuracy()
    find_consensus_threshold()
    find_best_conditions()
    header("ГОТОВО")
    ans = input("\n  Применить оптимальные настройки к config.py? (y/n): ").strip().lower()
    if ans == 'y':
        optimize_weights(auto_apply=True)


if __name__ == "__main__":
    run_full()
