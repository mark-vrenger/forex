import pandas as pd
import numpy as np
from datetime import datetime
from database import db

# ═══════════════════════════════════════
#  DEEP ANALYTICS v1.0
#  Анализирует все данные из SQLite
#  Даёт рекомендации по настройкам
# ═══════════════════════════════════════


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def analyze_trades():
    header("TRADE ANALYSIS")
    df = db.get_trades_df()
    closed = df[df['result'].notna()].copy()
    if len(closed) == 0:
        print("  No closed trades yet.")
        return None
    closed['profit'] = closed['profit'].astype(float)
    closed['confidence'] = closed['confidence'].astype(int)
    closed['rsi'] = closed['rsi'].astype(float)
    closed['atr'] = closed['atr'].astype(float)
    closed['spread'] = closed['spread'].astype(float)
    wins = closed[closed['result'] == 'WIN']
    losses = closed[closed['result'] == 'LOSS']
    total = len(closed)
    wr = len(wins) / total * 100
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
    pf = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else 0
    total_profit = closed['profit'].sum()
    # Equity curve
    equity = [0]
    peak = 0
    max_dd = 0
    for _, t in closed.iterrows():
        equity.append(equity[-1] + t['profit'])
        if equity[-1] > peak:
            peak = equity[-1]
        dd = peak - equity[-1]
        if dd > max_dd:
            max_dd = dd
    print(f"""
  Total trades:     {total}
  Wins:             {len(wins)} ({wr:.1f}%)
  Losses:           {len(losses)}
  Profit factor:    {pf:.2f}
  Net P/L:          ${total_profit:.2f}
  Avg win:          ${avg_win:.2f}
  Avg loss:         ${avg_loss:.2f}
  Max drawdown:     ${max_dd:.2f}
  Best trade:       ${closed['profit'].max():.2f}
  Worst trade:      ${closed['profit'].min():.2f}
    """)
    return closed


def analyze_by_session(closed):
    header("BY SESSION")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    sessions = closed.groupby('session').agg(
        trades=('profit', 'count'),
        profit=('profit', 'sum'),
        avg_profit=('profit', 'mean'),
        winrate=('result', lambda x: (x == 'WIN').mean() * 100),
        avg_confidence=('confidence', 'mean')
    ).round(2)
    sessions = sessions.sort_values('profit', ascending=False)
    print(sessions.to_string())
    best = sessions.index[0]
    worst = sessions.index[-1]
    print(f"\n  BEST session:  {best} (${sessions.loc[best, 'profit']:.2f})")
    print(f"  WORST session: {worst} (${sessions.loc[worst, 'profit']:.2f})")
    if sessions.loc[worst, 'winrate'] < 40 and sessions.loc[worst, 'trades'] >= 3:
        print(f"\n  >>> RECOMMENDATION: Consider disabling trading during '{worst}'")


def analyze_by_direction(closed):
    header("BY DIRECTION")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    for d in ['BUY', 'SELL']:
        subset = closed[closed['direction'] == d]
        if len(subset) == 0:
            continue
        wr = (subset['result'] == 'WIN').mean() * 100
        profit = subset['profit'].sum()
        avg = subset['profit'].mean()
        print(f"\n  {d}:")
        print(f"    Trades: {len(subset)} | Winrate: {wr:.1f}%")
        print(f"    Total P/L: ${profit:.2f} | Avg: ${avg:.2f}")
        if wr < 40 and len(subset) >= 5:
            print(f"    >>> WARNING: {d} trades are losing! Consider avoiding.")


def analyze_by_confidence(closed):
    header("BY CONFIDENCE LEVEL")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    for conf in range(5, 11):
        subset = closed[closed['confidence'] == conf]
        if len(subset) == 0:
            continue
        wr = (subset['result'] == 'WIN').mean() * 100
        profit = subset['profit'].sum()
        print(f"  Conf {conf}: {len(subset)} trades | WR: {wr:.1f}% | P/L: ${profit:.2f}")
    # Find optimal threshold
    best_conf = 7
    best_profit = -999999
    for threshold in range(5, 10):
        subset = closed[closed['confidence'] >= threshold]
        if len(subset) < 3:
            continue
        profit = subset['profit'].sum()
        wr = (subset['result'] == 'WIN').mean() * 100
        if profit > best_profit:
            best_profit = profit
            best_conf = threshold
    print(f"\n  >>> OPTIMAL MIN_CONFIDENCE: {best_conf} (profit: ${best_profit:.2f})")


def analyze_by_rsi(closed):
    header("BY RSI ZONES")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    zones = [
        ("Oversold (< 30)", closed[closed['rsi'] < 30]),
        ("Low (30-40)", closed[(closed['rsi'] >= 30) & (closed['rsi'] < 40)]),
        ("Neutral (40-60)", closed[(closed['rsi'] >= 40) & (closed['rsi'] < 60)]),
        ("High (60-70)", closed[(closed['rsi'] >= 60) & (closed['rsi'] < 70)]),
        ("Overbought (> 70)", closed[closed['rsi'] >= 70]),
    ]
    for name, subset in zones:
        if len(subset) == 0:
            continue
        wr = (subset['result'] == 'WIN').mean() * 100
        profit = subset['profit'].sum()
        print(f"  {name}: {len(subset)} trades | WR: {wr:.1f}% | P/L: ${profit:.2f}")
    # Best RSI range for BUY
    buys = closed[closed['direction'] == 'BUY']
    if len(buys) >= 5:
        winning_buys = buys[buys['result'] == 'WIN']
        losing_buys = buys[buys['result'] == 'LOSS']
        if len(winning_buys) > 0 and len(losing_buys) > 0:
            print(f"\n  BUY wins avg RSI:   {winning_buys['rsi'].mean():.1f}")
            print(f"  BUY losses avg RSI: {losing_buys['rsi'].mean():.1f}")
    sells = closed[closed['direction'] == 'SELL']
    if len(sells) >= 5:
        winning_sells = sells[sells['result'] == 'WIN']
        losing_sells = sells[sells['result'] == 'LOSS']
        if len(winning_sells) > 0 and len(losing_sells) > 0:
            print(f"  SELL wins avg RSI:   {winning_sells['rsi'].mean():.1f}")
            print(f"  SELL losses avg RSI: {losing_sells['rsi'].mean():.1f}")


def analyze_by_spread(closed):
    header("BY SPREAD")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    zones = [
        ("Tight (< 1.0p)", closed[closed['spread'] < 1.0]),
        ("Normal (1.0-2.0p)", closed[(closed['spread'] >= 1.0) & (closed['spread'] < 2.0)]),
        ("Wide (2.0-3.0p)", closed[(closed['spread'] >= 2.0) & (closed['spread'] < 3.0)]),
        ("Very wide (> 3.0p)", closed[closed['spread'] >= 3.0]),
    ]
    for name, subset in zones:
        if len(subset) == 0:
            continue
        wr = (subset['result'] == 'WIN').mean() * 100
        profit = subset['profit'].sum()
        print(f"  {name}: {len(subset)} trades | WR: {wr:.1f}% | P/L: ${profit:.2f}")


def analyze_tech_signals(closed):
    header("TECH SIGNALS ACCURACY")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    if 'tech_buy' not in closed.columns or 'tech_sell' not in closed.columns:
        print("  No tech signal data.")
        return
    # When tech agreed with direction
    agreed = []
    disagreed = []
    for _, t in closed.iterrows():
        tb = t.get('tech_buy', 0) or 0
        ts = t.get('tech_sell', 0) or 0
        if t['direction'] == 'BUY' and tb > ts:
            agreed.append(t)
        elif t['direction'] == 'SELL' and ts > tb:
            agreed.append(t)
        else:
            disagreed.append(t)
    if agreed:
        ag = pd.DataFrame(agreed)
        ag_wr = (ag['result'] == 'WIN').mean() * 100
        ag_profit = ag['profit'].sum()
        print(f"  Tech AGREED with AI: {len(ag)} trades | WR: {ag_wr:.1f}% | P/L: ${ag_profit:.2f}")
    if disagreed:
        dg = pd.DataFrame(disagreed)
        dg_wr = (dg['result'] == 'WIN').mean() * 100
        dg_profit = dg['profit'].sum()
        print(f"  Tech DISAGREED:      {len(dg)} trades | WR: {dg_wr:.1f}% | P/L: ${dg_profit:.2f}")
    if agreed and disagreed:
        ag = pd.DataFrame(agreed)
        dg = pd.DataFrame(disagreed)
        ag_wr = (ag['result'] == 'WIN').mean() * 100
        dg_wr = (dg['result'] == 'WIN').mean() * 100
        if ag_wr > dg_wr + 10:
            print(f"\n  >>> Tech signals IMPROVE accuracy by {ag_wr - dg_wr:.1f}%")
        elif dg_wr > ag_wr + 10:
            print(f"\n  >>> WARNING: Tech signals HURT accuracy! Consider reducing weight.")


def analyze_corr_signals(closed):
    header("CORRELATION ACCURACY")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    if 'corr_buy' not in closed.columns or 'corr_sell' not in closed.columns:
        print("  No correlation data.")
        return
    agreed = []
    disagreed = []
    for _, t in closed.iterrows():
        cb = t.get('corr_buy', 0) or 0
        cs = t.get('corr_sell', 0) or 0
        if t['direction'] == 'BUY' and cb > cs:
            agreed.append(t)
        elif t['direction'] == 'SELL' and cs > cb:
            agreed.append(t)
        else:
            disagreed.append(t)
    if agreed:
        ag = pd.DataFrame(agreed)
        ag_wr = (ag['result'] == 'WIN').mean() * 100
        ag_profit = ag['profit'].sum()
        print(f"  Corr AGREED with AI: {len(ag)} trades | WR: {ag_wr:.1f}% | P/L: ${ag_profit:.2f}")
    if disagreed:
        dg = pd.DataFrame(disagreed)
        dg_wr = (dg['result'] == 'WIN').mean() * 100
        dg_profit = dg['profit'].sum()
        print(f"  Corr DISAGREED:      {len(dg)} trades | WR: {dg_wr:.1f}% | P/L: ${dg_profit:.2f}")


def analyze_signals():
    header("SIGNAL ANALYSIS (ALL — including WAIT)")
    df = db.get_signals_df()
    if len(df) == 0:
        print("  No signals recorded.")
        return
    total = len(df)
    acted = len(df[df['acted'] == 1])
    wait = len(df[df['direction'] == 'WAIT'])
    buy = len(df[df['direction'] == 'BUY'])
    sell = len(df[df['direction'] == 'SELL'])
    print(f"""
  Total signals:    {total}
  Acted on:         {acted} ({acted/total*100:.1f}%)
  Skipped (WAIT):   {wait} ({wait/total*100:.1f}%)
  BUY signals:      {buy}
  SELL signals:     {sell}
    """)
    # Confidence distribution
    if 'final_confidence' in df.columns:
        print("  Confidence distribution:")
        for c in range(1, 11):
            count = len(df[df['final_confidence'] == c])
            if count > 0:
                bar = "#" * min(count, 40)
                print(f"    {c:2d}: {bar} ({count})")
    # Average tech/corr scores
    if 'tech_buy' in df.columns:
        avg_tb = df['tech_buy'].mean()
        avg_ts = df['tech_sell'].mean()
        avg_cb = df['corr_buy'].mean()
        avg_cs = df['corr_sell'].mean()
        print(f"\n  Avg Tech scores: BUY={avg_tb:.1f} SELL={avg_ts:.1f}")
        print(f"  Avg Corr scores: BUY={avg_cb:.1f} SELL={avg_cs:.1f}")


def analyze_equity_curve(closed):
    header("EQUITY CURVE")
    if closed is None or len(closed) == 0:
        print("  No data.")
        return
    equity = 0
    peak = 0
    in_dd = False
    dd_start = None
    longest_dd_days = 0
    current_dd_start = None
    print("\n  Balance progression:")
    for i, (_, t) in enumerate(closed.iterrows()):
        equity += t['profit']
        if equity > peak:
            peak = equity
            if in_dd and current_dd_start:
                dd_days = i
                if dd_days > longest_dd_days:
                    longest_dd_days = dd_days
            in_dd = False
            current_dd_start = None
        else:
            if not in_dd:
                in_dd = True
                current_dd_start = i
        # Print every 10th trade or last
        if i % 10 == 0 or i == len(closed) - 1:
            marker = "+" if t['profit'] > 0 else "-"
            bar_len = min(int(abs(equity) / 10), 40)
            if equity >= 0:
                bar = "█" * bar_len
                print(f"    Trade {i+1:4d}: ${equity:8.2f} |{bar}")
            else:
                bar = "░" * bar_len
                print(f"    Trade {i+1:4d}: ${equity:8.2f} |{bar}")
    # Streak analysis
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current = 0
    for _, t in closed.iterrows():
        if t['result'] == 'WIN':
            if current > 0:
                current += 1
            else:
                current = 1
            max_win_streak = max(max_win_streak, current)
        else:
            if current < 0:
                current -= 1
            else:
                current = -1
            max_loss_streak = max(max_loss_streak, abs(current))
    print(f"\n  Max winning streak: {max_win_streak}")
    print(f"  Max losing streak:  {max_loss_streak}")


def generate_recommendations(closed):
    header("RECOMMENDATIONS")
    if closed is None or len(closed) < 5:
        print("  Need at least 5 trades for recommendations.")
        return
    recs = []
    # 1. Winrate check
    wr = (closed['result'] == 'WIN').mean() * 100
    if wr < 45:
        recs.append("LOW WINRATE: Consider increasing MIN_CONFIDENCE to 8")
    elif wr > 65:
        recs.append("HIGH WINRATE: System is performing well")
    # 2. Session check
    for session in closed['session'].unique():
        subset = closed[closed['session'] == session]
        if len(subset) >= 3:
            s_wr = (subset['result'] == 'WIN').mean() * 100
            s_profit = subset['profit'].sum()
            if s_wr < 35 and s_profit < 0:
                recs.append(f"DISABLE '{session}': {s_wr:.0f}% winrate, ${s_profit:.2f} loss")
    # 3. Direction check
    for d in ['BUY', 'SELL']:
        subset = closed[closed['direction'] == d]
        if len(subset) >= 5:
            d_wr = (subset['result'] == 'WIN').mean() * 100
            if d_wr < 35:
                recs.append(f"REDUCE {d}: Only {d_wr:.0f}% winrate")
    # 4. Confidence check
    high = closed[closed['confidence'] >= 8]
    low = closed[closed['confidence'] <= 7]
    if len(high) >= 3 and len(low) >= 3:
        h_wr = (high['result'] == 'WIN').mean() * 100
        l_wr = (low['result'] == 'WIN').mean() * 100
        if h_wr > l_wr + 15:
            recs.append(f"RAISE MIN_CONFIDENCE: High conf WR={h_wr:.0f}% vs Low={l_wr:.0f}%")
    # 5. Spread check
    tight = closed[closed['spread'] < 1.5]
    wide = closed[closed['spread'] >= 2.0]
    if len(tight) >= 3 and len(wide) >= 3:
        t_wr = (tight['result'] == 'WIN').mean() * 100
        w_wr = (wide['result'] == 'WIN').mean() * 100
        if t_wr > w_wr + 10:
            recs.append(f"LOWER MAX_SPREAD: Tight spread WR={t_wr:.0f}% vs Wide={w_wr:.0f}%")
    # 6. Duration
    if 'duration_minutes' in closed.columns:
        dur = closed['duration_minutes'].dropna()
        if len(dur) > 0:
            wins_dur = closed[closed['result'] == 'WIN']['duration_minutes'].dropna()
            loss_dur = closed[closed['result'] == 'LOSS']['duration_minutes'].dropna()
            if len(wins_dur) > 0 and len(loss_dur) > 0:
                if wins_dur.mean() < loss_dur.mean() * 0.5:
                    recs.append("WINNERS are faster than losers - trailing stop is working well")
    if not recs:
        recs.append("System looks balanced. Keep monitoring.")
    for i, r in enumerate(recs, 1):
        icon = "✅" if "well" in r.lower() or "balanced" in r.lower() else "⚠️"
        print(f"  {icon} {i}. {r}")


def analyze_daily():
    header("DAILY STATS")
    db.update_daily_stats()
    df = db.get_daily_stats_df()
    if len(df) == 0:
        print("  No daily stats yet.")
        return
    print(df[['date', 'total_trades', 'winning_trades', 'losing_trades',
              'total_profit', 'profit_factor']].to_string(index=False))
    # Best/worst day
    if len(df) > 1:
        best_day = df.loc[df['total_profit'].idxmax()]
        worst_day = df.loc[df['total_profit'].idxmin()]
        print(f"\n  Best day:  {best_day['date']} (${best_day['total_profit']:.2f})")
        print(f"  Worst day: {worst_day['date']} (${worst_day['total_profit']:.2f})")


def run_full_analysis():
    print("""
    ╔══════════════════════════════════════════╗
    ║       DEEP ANALYTICS v1.0               ║
    ║       Analyzing all data from SQLite     ║
    ╚══════════════════════════════════════════╝
    """)
    closed = analyze_trades()
    analyze_by_session(closed)
    analyze_by_direction(closed)
    analyze_by_confidence(closed)
    analyze_by_rsi(closed)
    analyze_by_spread(closed)
    analyze_tech_signals(closed)
    analyze_corr_signals(closed)
    analyze_signals()
    analyze_equity_curve(closed)
    analyze_daily()
    generate_recommendations(closed)
    header("DONE")
    print("  Run this script periodically to track improvements.")
    print("  Data file: logs/autopilot.db")


if __name__ == "__main__":
    run_full_analysis()