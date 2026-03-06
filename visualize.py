"""
visualize.py — Визуальная аналитика v2.0
Добавлены: plot_council_agreement(), plot_agent_accuracy(),
           plot_debate_impact(), export_session_log().
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from database import db

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ─────────────────────────────────────────
#  ORIGINAL CHARTS
# ─────────────────────────────────────────

def plot_equity_curve():
    header("EQUITY CURVE")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) == 0:
        print("  Нет закрытых сделок.")
        return

    closed['profit'] = closed['profit'].astype(float)
    closed = closed.sort_values('time')
    closed['cumulative'] = closed['profit'].cumsum()

    values = closed['cumulative'].tolist()
    max_val = max(max(values), 1)
    min_val = min(min(values), -1)
    chart_height = 20
    chart_width  = min(len(values), 60)

    if len(values) > chart_width:
        step = len(values) // chart_width
        values_sampled = values[::step]
    else:
        values_sampled = values

    print(f"\n  Кривая P/L ({len(closed)} сделок):")
    print(f"  Max: ${max_val:.2f} | Min: ${min_val:.2f} | Итог: ${values[-1]:.2f}")
    print()

    val_range = max_val - min_val if max_val != min_val else 1
    for row in range(chart_height, -1, -1):
        level = min_val + (row / chart_height) * val_range
        line = "  "
        if row == chart_height:
            line += f"${max_val:>8.0f} |"
        elif row == 0:
            line += f"${min_val:>8.0f} |"
        elif row == chart_height // 2:
            mid = (max_val + min_val) / 2
            line += f"${mid:>8.0f} |"
        else:
            line += "          |"

        for v in values_sampled:
            v_row = int((v - min_val) / val_range * chart_height)
            if v_row == row:
                line += "█" if v >= 0 else "░"
            elif row == int((0 - min_val) / val_range * chart_height):
                line += "─"
            else:
                line += " "
        print(line)

    print("  " + " " * 10 + "+" + "─" * len(values_sampled))
    print(f"  " + " " * 10 + f"Сделка 1" + " " * (len(values_sampled) - 10) + f"Сделка {len(closed)}")

    csv_path = os.path.join(PLOT_DIR, "equity_curve.csv")
    closed[['time', 'direction', 'profit', 'cumulative', 'confidence',
            'rsi', 'session']].to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")

    peak = 0
    max_dd = 0
    dd_start = 0
    for i, v in enumerate(values):
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
            dd_start = i
    print(f"  Макс. просадка: ${max_dd:.2f} (на сделке #{dd_start + 1})")


def plot_heatmap_rsi_profit():
    header("HEATMAP: RSI vs PROFIT")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed['profit'] = closed['profit'].astype(float)
    closed['rsi']    = closed['rsi'].astype(float)
    bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]

    print(f"\n  {'RSI':12} {'Сделок':>7} {'WIN':>6} {'WR%':>6} {'Avg P/L':>9} {'Итого':>10} {'Оценка':>8}")
    print("  " + "-" * 60)

    for low, high in bins:
        subset = closed[(closed['rsi'] >= low) & (closed['rsi'] < high)]
        if len(subset) == 0:
            continue
        wins    = (subset['result'] == 'WIN').sum()
        wr      = wins / len(subset) * 100
        avg_pnl = subset['profit'].mean()
        total   = subset['profit'].sum()
        rating  = "ЛУЧШИЙ" if total > 0 and wr > 55 else "OK" if total > 0 else "СЛАБЫЙ" if wr > 45 else "ПЛОХОЙ"
        print(f"  {low:>3}-{high:<3}     {len(subset):>7} {wins:>6} {wr:>5.1f}% ${avg_pnl:>8.2f} ${total:>9.2f} {rating}")


def plot_heatmap_confidence_session():
    header("HEATMAP: УВЕРЕННОСТЬ x СЕССИЯ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed['profit']     = closed['profit'].astype(float)
    closed['confidence'] = closed['confidence'].astype(int)
    sessions = sorted(closed['session'].unique())
    confs    = sorted(closed['confidence'].unique())
    session_short = {s: s[:8] for s in sessions}
    header_line = f"  {'Уверен':<6}"
    for s in sessions:
        header_line += f" {session_short[s]:>10}"
    print(f"\n{header_line}")
    print("  " + "-" * (6 + 11 * len(sessions)))

    for conf in confs:
        line = f"  {conf:<6}"
        for s in sessions:
            subset = closed[(closed['confidence'] == conf) & (closed['session'] == s)]
            if len(subset) == 0:
                line += f" {'---':>10}"
            else:
                pnl = subset['profit'].sum()
                line += f" ${pnl:>7.0f}{'OK' if pnl > 0 else 'XX'}"
        print(line)


def plot_hourly_distribution():
    header("РАСПРЕДЕЛЕНИЕ ПО ЧАСАМ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed['profit'] = closed['profit'].astype(float)
    closed['hour']   = pd.to_datetime(closed['time']).dt.hour
    max_trades = closed.groupby('hour').size().max()

    print(f"\n  {'Час':<6} {'Сделок':>7} {'WR%':>6} {'P/L':>10} {'Бар':>30}")
    print("  " + "-" * 60)

    for h in range(7, 22):
        subset = closed[closed['hour'] == h]
        if len(subset) == 0:
            continue
        trades  = len(subset)
        wr      = (subset['result'] == 'WIN').mean() * 100
        pnl     = subset['profit'].sum()
        bar_len = int(trades / max_trades * 25) if max_trades > 0 else 0
        bar     = ("█" if pnl > 0 else "░") * bar_len
        sign    = "+" if pnl >= 0 else ""
        print(f"  {h:02d}:00 {trades:>7} {wr:>5.1f}% ${sign}{pnl:>8.2f}  {bar}")


def plot_weekday_distribution():
    header("РАСПРЕДЕЛЕНИЕ ПО ДНЯМ НЕДЕЛИ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed['profit']  = closed['profit'].astype(float)
    closed['weekday'] = pd.to_datetime(closed['time']).dt.day_name()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    print(f"\n  {'День':<12} {'Сделок':>7} {'WR%':>6} {'P/L':>10}")
    print("  " + "-" * 35)

    for d in days:
        subset = closed[closed['weekday'] == d]
        if len(subset) == 0:
            continue
        wr  = (subset['result'] == 'WIN').mean() * 100
        pnl = subset['profit'].sum()
        mark = " ЛУЧШИЙ" if pnl > 0 and wr > 55 else " ПЛОХОЙ" if pnl < 0 else ""
        print(f"  {d:<12} {len(subset):>7} {wr:>5.1f}% ${pnl:>9.2f}{mark}")


def plot_layer_comparison():
    header("АНАЛИЗ СОГЛАСОВАННОСТИ СЛОЁВ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed['profit'] = closed['profit'].astype(float)
    all_agree, partial, disagree = [], [], []

    for _, t in closed.iterrows():
        tb = t.get('tech_buy', 0) or 0
        ts = t.get('tech_sell', 0) or 0
        cb = t.get('corr_buy', 0) or 0
        cs = t.get('corr_sell', 0) or 0
        direction = t['direction']

        tech_agrees = (direction == "BUY" and tb > ts) or (direction == "SELL" and ts > tb)
        corr_agrees = (direction == "BUY" and cb > cs) or (direction == "SELL" and cs > cb)

        if tech_agrees and corr_agrees:
            all_agree.append(t)
        elif tech_agrees or corr_agrees:
            partial.append(t)
        else:
            disagree.append(t)

    categories = [
        ("Все слои согласны",   all_agree),
        ("Частичное согласие",  partial),
        ("Слои НЕ согласны",    disagree),
    ]

    print(f"\n  {'Категория':<22} {'Сделок':>7} {'WR%':>6} {'Avg P/L':>9} {'Итого':>10}")
    print("  " + "-" * 55)

    for name, trades_list in categories:
        if not trades_list:
            continue
        tdf = pd.DataFrame(trades_list)
        wr  = (tdf['result'] == 'WIN').mean() * 100
        avg = tdf['profit'].mean()
        tot = tdf['profit'].sum()
        mark = " OK" if tot > 0 else " XX"
        print(f"  {name:<22} {len(tdf):>7} {wr:>5.1f}% ${avg:>8.2f} ${tot:>9.2f}{mark}")


def plot_consecutive_streaks():
    header("СЕРИИ ПОБЕД / ПОРАЖЕНИЙ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) < 5:
        print("  Нужно больше сделок.")
        return

    closed = closed.sort_values('time')
    streaks = []
    current_type  = None
    current_count = 0

    for _, t in closed.iterrows():
        if t['result'] == current_type:
            current_count += 1
        else:
            if current_type is not None:
                streaks.append((current_type, current_count))
            current_type  = t['result']
            current_count = 1
    if current_type:
        streaks.append((current_type, current_count))

    win_streaks  = [c for t, c in streaks if t == 'WIN']
    loss_streaks = [c for t, c in streaks if t == 'LOSS']

    print(f"\n  Макс. серия побед:    {max(win_streaks)  if win_streaks  else 0}")
    print(f"  Макс. серия поражений: {max(loss_streaks) if loss_streaks else 0}")
    if win_streaks:
        print(f"  Ср. серия побед:    {sum(win_streaks)/len(win_streaks):.1f}")
    if loss_streaks:
        print(f"  Ср. серия поражений: {sum(loss_streaks)/len(loss_streaks):.1f}")

    print(f"\n  Последние 30 сделок:")
    results = closed['result'].tail(30).tolist()
    line = "  "
    for r in results:
        line += "█" if r == "WIN" else "░"
    print(line)
    print(f"  █=WIN ░=LOSS")


# ─────────────────────────────────────────
#  NEW: COUNCIL CHARTS
# ─────────────────────────────────────────

def plot_council_agreement():
    """% сделок при разном уровне консенсуса (1/3, 2/3, 3/3 голосов)."""
    header("АНАЛИЗ КОНСИЛИУМА: КОНСЕНСУС vs РЕЗУЛЬТАТ")

    sessions_df = db.get_council_sessions_df()
    if sessions_df is None or len(sessions_df) == 0:
        print("  Нет данных Консилиума.")
        return

    completed = sessions_df[sessions_df['trade_result'].notna()].copy()
    if len(completed) < 3:
        print("  Нужно минимум 3 завершённые сессии.")
        return

    completed['trade_profit'] = pd.to_numeric(completed['trade_profit'], errors='coerce').fillna(0)
    completed['consensus_pct'] = pd.to_numeric(completed['consensus_pct'], errors='coerce').fillna(0)

    print(f"\n  Всего сессий с результатом: {len(completed)}")
    print(f"\n  {'Консенсус':>12} {'Сделок':>7} {'WIN':>5} {'WR%':>6} {'P/L':>10}")
    print("  " + "-" * 43)

    for label, low, high in [
        ("33% (1/3)",  0,  40),
        ("67% (2/3)",  40, 70),
        ("100% (3/3)", 70, 101),
    ]:
        sub = completed[
            (completed['consensus_pct'] >= low) &
            (completed['consensus_pct'] < high)
        ]
        if len(sub) == 0:
            continue
        wins = (sub['trade_result'] == 'WIN').sum()
        wr   = wins / len(sub) * 100
        pnl  = sub['trade_profit'].sum()
        mark = " <-- ТОРГОВАТЬ ЗДЕСЬ" if low >= 67 and wr > 55 else ""
        print(f"  {label:>12} {len(sub):>7} {wins:>5} {wr:>5.1f}% ${pnl:>9.2f}{mark}")

    # Scatter: консенсус -> прибыль
    print(f"\n  Scatter: Консенсус → результаты")
    print(f"  {'Консенсус':>12}  Result")
    for _, row in completed.tail(20).iterrows():
        pct    = row['consensus_pct']
        result = row['trade_result']
        bar    = int(pct / 100 * 20)
        icon   = "█" if result == "WIN" else "░"
        print(f"  {pct:>10.0f}%  {'█' * bar}{icon}")


def plot_agent_accuracy():
    """Точность каждого агента Консилиума — ASCII радар."""
    header("ТОЧНОСТЬ АГЕНТОВ КОНСИЛИУМА")

    accuracy = db.get_agent_accuracy()
    if not accuracy:
        print("  Нет данных. Нужны завершённые сессии Консилиума.")
        return

    agents = ["GROK", "CONSERVATIVE", "ANALYST"]
    print(f"\n  {'Агент':<16} {'Точность':>8} {'Правильно':>10} {'Всего':>7}  Бар")
    print("  " + "-" * 60)

    max_total = max((s['total'] for s in accuracy.values()), default=1)

    for name in agents:
        if name not in accuracy:
            print(f"  {name:<16} {'нет данных':>8}")
            continue
        stats = accuracy[name]
        acc   = stats['accuracy']
        bar_len = int(acc / 100 * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        medal   = " ЛУЧШИЙ" if acc >= 65 else " ХУДШИЙ" if acc < 45 else ""
        print(f"  {name:<16} {acc:>7.1f}% {stats['correct']:>10} {stats['total']:>7}  {bar}{medal}")

    # Вывод рекомендаций
    if accuracy:
        best = max(accuracy, key=lambda n: accuracy[n]['accuracy'])
        worst = min(accuracy, key=lambda n: accuracy[n]['accuracy'])
        print(f"\n  Лучший агент:  {best} ({accuracy[best]['accuracy']:.1f}%)")
        print(f"  Худший агент:  {worst} ({accuracy[worst]['accuracy']:.1f}%)")
        diff = accuracy[best]['accuracy'] - accuracy[worst]['accuracy']
        if diff > 15:
            print(f"  >>> Рекомендую увеличить вес {best} в optimizer.py")


def plot_debate_impact():
    """% случаев когда дебаты изменили финальное решение Консилиума."""
    header("ВЛИЯНИЕ ДЕБАТОВ НА РЕШЕНИЕ")

    sessions_df = db.get_council_sessions_df()
    if sessions_df is None or len(sessions_df) == 0:
        print("  Нет данных Консилиума.")
        return

    has_debate = sessions_df[
        sessions_df['votes_r2'].notna() &
        (sessions_df['votes_r2'] != '[]') &
        (sessions_df['votes_r2'] != '')
    ]

    if len(has_debate) == 0:
        print("  Нет сессий с дебатами.")
        return

    changed_count  = 0
    changed_helped = 0
    total_debates  = 0

    for _, row in has_debate.iterrows():
        try:
            r1 = json.loads(row['votes_r1'])
            r2 = json.loads(row['votes_r2'])
        except Exception:
            continue

        total_debates += 1
        any_changed = False
        for v1, v2 in zip(r1, r2):
            if v1.get('signal') != v2.get('signal'):
                any_changed = True
                break

        if any_changed:
            changed_count += 1
            result = row.get('trade_result')
            if result == 'WIN':
                changed_helped += 1

    if total_debates == 0:
        print("  Нет данных для анализа.")
        return

    change_pct  = changed_count / total_debates * 100
    helpful_pct = changed_helped / changed_count * 100 if changed_count > 0 else 0

    print(f"\n  Всего сессий с дебатами:  {total_debates}")
    print(f"  Изменили мнение:          {changed_count} ({change_pct:.1f}%)")
    print(f"  Изменения привели к WIN:  {changed_helped} ({helpful_pct:.1f}%)")
    print()

    if helpful_pct > 60:
        print("  >>> Дебаты ПОЛЕЗНЫ — агенты улучшают решения при обмене аргументами")
    elif helpful_pct < 40:
        print("  >>> Дебаты ВРЕДНЫ — первоначальные инстинкты агентов точнее")
    else:
        print("  >>> Влияние дебатов нейтральное")


def export_session_log(session_id: int):
    """Полный протокол одной сессии Консилиума."""
    header(f"ЭКСПОРТ СЕССИИ #{session_id}")

    sessions_df = db.get_council_sessions_df()
    if sessions_df is None or len(sessions_df) == 0:
        print("  Нет данных.")
        return

    row = sessions_df[sessions_df['id'] == session_id]
    if len(row) == 0:
        print(f"  Сессия #{session_id} не найдена.")
        return

    row = row.iloc[0]
    print(f"\n  Время:       {row['time']}")
    print(f"  Цена:        {row['price']:.5f}")
    print(f"  Сигнал:      {row['signal']} (уверенность: {row['confidence']}/10)")
    print(f"  Консенсус:   {row['consensus_pct']:.0f}%")
    if row.get('blocked_by'):
        print(f"  Блокировка:  {row['blocked_by']} — {row['block_reason']}")
    print(f"  Результат:   {row.get('trade_result', 'N/A')} (${row.get('trade_profit', 0):.2f})")

    print(f"\n  --- Раунд 1 (первичные голоса) ---")
    try:
        for v in json.loads(row['votes_r1']):
            print(f"  {v.get('emoji','')} {v.get('name',''):<16} {v.get('signal',''):<5} "
                  f"({v.get('confidence',0)}/10) | {v.get('reasoning','')[:80]}")
    except Exception:
        print("  Нет данных R1")

    print(f"\n  --- Раунд 2 (после дебатов) ---")
    try:
        r2 = json.loads(row['votes_r2'])
        if r2:
            for v in r2:
                changed = f" (был {v['changed_from']})" if v.get('changed_from') else ""
                print(f"  {v.get('emoji','')} {v.get('name',''):<16} {v.get('signal',''):<5} "
                      f"({v.get('confidence',0)}/10){changed} | {v.get('reasoning','')[:80]}")
        else:
            print("  Нет дебатов")
    except Exception:
        print("  Нет данных R2")

    # Сохранить в файл
    path = os.path.join(PLOT_DIR, f"session_{session_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Session #{session_id}\n")
        f.write(f"Time: {row['time']}\n")
        f.write(f"Signal: {row['signal']} ({row['confidence']}/10)\n")
        f.write(f"Result: {row.get('trade_result','?')}\n")
    print(f"\n  Сохранено: {path}")


# ─────────────────────────────────────────
#  SUMMARY REPORT
# ─────────────────────────────────────────

def generate_summary_report():
    header("СВОДНЫЙ ОТЧЁТ")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) == 0:
        print("  Нет сделок.")
        return

    closed['profit'] = closed['profit'].astype(float)
    wins   = closed[closed['result'] == 'WIN']
    losses = closed[closed['result'] == 'LOSS']
    total_pnl = closed['profit'].sum()
    total = len(closed)
    wr    = len(wins) / total * 100
    avg_win  = wins['profit'].mean()   if len(wins)   > 0 else 0
    avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
    win_sum  = sum(p for p in closed['profit'] if p > 0)
    loss_sum = abs(sum(p for p in closed['profit'] if p < 0))
    pf = win_sum / loss_sum if loss_sum > 0 else 0

    print(f"""
  +---------------------------------------+
  |  ИТОГО P/L:    ${total_pnl:>10.2f}          |
  |  СДЕЛОК:       {total:>5}                 |
  |  WINRATE:      {wr:>5.1f}%                |
  |  PROFIT FACTOR: {pf:>5.2f}               |
  |  AVG WIN:      ${avg_win:>10.2f}          |
  |  AVG LOSS:     ${avg_loss:>10.2f}          |
  +---------------------------------------+
    """)

    cum = closed['profit'].cumsum().tolist()
    sparkline = cum[::max(len(cum) // 40, 1)] if len(cum) > 40 else cum
    min_v = min(sparkline)
    max_v = max(sparkline)
    rng   = max_v - min_v if max_v != min_v else 1
    chars = " ▁▂▃▄▅▆▇█"
    spark = ""
    for v in sparkline:
        idx   = int((v - min_v) / rng * 8)
        spark += chars[idx]
    print(f"  Equity: {spark}")
    print(f"          ${min_v:.0f}" + " " * (len(spark) - 10) + f"${max_v:.0f}")

    # Данные Консилиума
    sessions_df = db.get_council_sessions_df()
    if sessions_df is not None and len(sessions_df) > 0:
        completed = sessions_df[sessions_df['trade_result'].notna()]
        total_sessions = len(sessions_df)
        acted = sessions_df[sessions_df['acted'] == 1]
        print(f"\n  Консилиум: {total_sessions} сессий | "
              f"{len(acted)} открытых сделок | "
              f"{len(completed)} закрытых")

    report_path = os.path.join(PLOT_DIR, f"report_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"AUTOPILOT REPORT {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Trades: {total} | WR: {wr:.1f}% | PF: {pf:.2f} | P/L: ${total_pnl:.2f}\n")
    print(f"\n  Отчёт: {report_path}")


def run_all():
    print("""
    +==========================================+
    |    VISUAL ANALYTICS v2.0                |
    +==========================================+
    """)
    generate_summary_report()
    plot_equity_curve()
    plot_heatmap_rsi_profit()
    plot_heatmap_confidence_session()
    plot_hourly_distribution()
    plot_weekday_distribution()
    plot_layer_comparison()
    plot_consecutive_streaks()
    # Новые графики Консилиума
    plot_council_agreement()
    plot_agent_accuracy()
    plot_debate_impact()
    header("ВСЕ ГРАФИКИ ПОСТРОЕНЫ")
    print("  CSV данные в: plots/")
    print("  Рекомендуется после 20+ сделок.")


if __name__ == "__main__":
    run_all()
