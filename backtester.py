import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import json
import os
from config import SYMBOL, RISK_PERCENT, MIN_CONFIDENCE, MAX_DAILY_LOSS, MAX_DAILY_TRADES


class VirtualAccount:
    def __init__(self, balance=10000, spread=1.5, slippage=1.0, commission=7.0):
        self.initial = balance
        self.balance = balance
        self.equity = [balance]
        self.spread = spread
        self.slippage = slippage
        self.commission = commission
        self.position = None
        self.trades = []
        self.daily_pnl = {}
        self.daily_count = {}

    def open_pos(self, direction, price, lot, sl, tp, meta, bar_time):
        if self.position:
            return False
        point = 0.00001
        slip = self.slippage * point * (1 if np.random.random() > 0.5 else -1)
        if direction == "BUY":
            entry = price + (self.spread * point / 2) + slip
        else:
            entry = price - (self.spread * point / 2) + slip
        self.position = {
            "direction": direction, "entry": entry, "lot": lot,
            "sl": sl, "tp": tp, "time": bar_time,
            "meta": meta, "max_p": 0, "min_p": 0
        }
        return True

    def check(self, high, low, close, bar_time):
        if not self.position:
            return None
        p = self.position
        if p["direction"] == "BUY":
            if low <= p["sl"]:
                return self._close(p["sl"], "SL", bar_time)
            elif high >= p["tp"]:
                return self._close(p["tp"], "TP", bar_time)
            pnl = (close - p["entry"]) * p["lot"] * 100000
            p["max_p"] = max(p["max_p"], pnl)
            p["min_p"] = min(p["min_p"], pnl)
        else:
            if high >= p["sl"]:
                return self._close(p["sl"], "SL", bar_time)
            elif low <= p["tp"]:
                return self._close(p["tp"], "TP", bar_time)
            pnl = (p["entry"] - close) * p["lot"] * 100000
            p["max_p"] = max(p["max_p"], pnl)
            p["min_p"] = min(p["min_p"], pnl)
        return None

    def _close(self, exit_price, reason, bar_time):
        p = self.position
        if p["direction"] == "BUY":
            profit = (exit_price - p["entry"]) * p["lot"] * 100000
        else:
            profit = (p["entry"] - exit_price) * p["lot"] * 100000
        profit -= self.commission * p["lot"]
        self.balance += profit
        self.equity.append(self.balance)
        day = bar_time.strftime("%Y-%m-%d")
        self.daily_pnl[day] = self.daily_pnl.get(day, 0) + profit
        self.daily_count[day] = self.daily_count.get(day, 0) + 1
        dur = (bar_time - p["time"]).total_seconds() / 3600
        trade = {
            "entry_time": p["time"].isoformat(), "exit_time": bar_time.isoformat(),
            "direction": p["direction"], "entry": p["entry"], "exit": exit_price,
            "lot": p["lot"], "sl": p["sl"], "tp": p["tp"],
            "profit": round(profit, 2), "result": "WIN" if profit > 0 else "LOSS",
            "reason": reason, "duration_h": round(dur, 1),
            "max_float": round(p["max_p"], 2), "min_float": round(p["min_p"], 2),
            **p.get("meta", {})
        }
        self.trades.append(trade)
        self.position = None
        return trade

    def force_close(self, price, bar_time):
        if self.position:
            return self._close(price, "END", bar_time)

    def can_trade(self, bar_time):
        day = bar_time.strftime("%Y-%m-%d")
        if self.daily_pnl.get(day, 0) <= -MAX_DAILY_LOSS:
            return False
        if self.daily_count.get(day, 0) >= MAX_DAILY_TRADES:
            return False
        return True


def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50


def calc_atr(df, period=14):
    d = df.copy()
    d['tr'] = pd.concat([
        d['high'] - d['low'],
        abs(d['high'] - d['close'].shift(1)),
        abs(d['low'] - d['close'].shift(1))
    ], axis=1).max(axis=1)
    return d['tr'].rolling(period).mean().iloc[-1]


def get_session(h):
    if 7 <= h < 10: return "London Open"
    elif 10 <= h < 13: return "London"
    elif 13 <= h < 17: return "London+NY"
    elif 17 <= h < 21: return "New York"
    else: return "Asia"


def calc_lot(balance, atr, risk=RISK_PERCENT):
    risk_money = balance * (risk / 100)
    sl_pips = (atr * 1.5) / 0.00001
    if sl_pips == 0: return 0.01
    return round(max(0.01, min(risk_money / (sl_pips * 10), 0.5)), 2)


def rule_signal(h4, h1, m15, rsi, tech_buy, tech_sell):
    sma5_h4 = h4['close'].tail(5).mean()
    sma15_h4 = h4['close'].tail(15).mean()
    sma5_h1 = h1['close'].tail(5).mean()
    sma15_h1 = h1['close'].tail(15).mean()
    h4t = "UP" if sma5_h4 > sma15_h4 else "DOWN" if sma5_h4 < sma15_h4 else "FLAT"
    h1t = "UP" if sma5_h1 > sma15_h1 else "DOWN" if sma5_h1 < sma15_h1 else "FLAT"
    if h4t == "FLAT" or h1t == "FLAT" or h4t != h1t:
        return 3, "WAIT", ""
    ema12 = h1['close'].ewm(span=12).mean().iloc[-1]
    ema26 = h1['close'].ewm(span=26).mean().iloc[-1]
    macd = ema12 > ema26
    direction = "BUY" if h4t == "UP" else "SELL"
    if direction == "BUY" and not macd: return 4, "WAIT", ""
    if direction == "SELL" and macd: return 4, "WAIT", ""
    if direction == "BUY" and rsi > 70: return 3, "WAIT", ""
    if direction == "SELL" and rsi < 30: return 3, "WAIT", ""
    conf = 6
    m15t = "UP" if m15['close'].tail(5).mean() > m15['close'].tail(15).mean() else "DOWN"
    if m15t == h4t: conf += 1
    if direction == "BUY" and rsi < 40: conf += 1
    if direction == "SELL" and rsi > 60: conf += 1
    if direction == "BUY" and tech_buy > tech_sell + 2: conf += 1
    if direction == "SELL" and tech_sell > tech_buy + 2: conf += 1
    return min(conf, 10), direction, f"H4:{h4t} H1:{h1t} RSI:{rsi:.0f}"


def tech_signals(h1):
    buy, sell = 0, 0
    if len(h1) < 50: return 0, 0
    sma10 = h1['close'].rolling(10).mean()
    sma20 = h1['close'].rolling(20).mean()
    if sma10.iloc[-1] > sma20.iloc[-1]: buy += 1
    else: sell += 1
    rsi = calc_rsi(h1['close'])
    if rsi < 30: buy += 2
    elif rsi > 70: sell += 2
    sma20v = h1['close'].rolling(20).mean().iloc[-1]
    std20 = h1['close'].rolling(20).std().iloc[-1]
    if h1['close'].iloc[-1] <= sma20v - 2 * std20: buy += 2
    elif h1['close'].iloc[-1] >= sma20v + 2 * std20: sell += 2
    return buy, sell


def run_backtest(months=6, balance=10000, spread=1.5, slippage=1.0,
                 commission=7.0, min_conf=MIN_CONFIDENCE,
                 corr_weight=1.0, tech_weight=1.0):
    if not mt5.initialize():
        print("MT5 Error")
        return None
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    m15 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
    h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start, end)
    h4 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4, start, end)
    if any(x is None for x in [m15, h1, h4]):
        print("No data")
        return None
    for name, arr in [("M15", m15), ("H1", h1), ("H4", h4)]:
        df = pd.DataFrame(arr)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if name == "M15": df_m15 = df.set_index('time')
        elif name == "H1": df_h1 = df.set_index('time')
        else: df_h4 = df.set_index('time')
    acc = VirtualAccount(balance, spread, slippage, commission)
    step = 4
    checks = range(200, len(df_m15), step)
    for i, idx in enumerate(checks):
        bt = df_m15.index[idx]
        bar = df_m15.iloc[idx]
        if acc.position:
            for j in range(max(0, idx - step), idx + 1):
                b = df_m15.iloc[j]
                r = acc.check(b['high'], b['low'], b['close'], df_m15.index[j])
                if r: break
            if acc.position: continue
        h = bt.hour
        wd = bt.weekday()
        if 0 <= h <= 6 or (wd == 4 and h >= 18) or wd in [5, 6]: continue
        if not acc.can_trade(bt): continue
        s_m15 = df_m15.loc[:bt].tail(30)
        s_h1 = df_h1.loc[:bt].tail(30)
        s_h4 = df_h4.loc[:bt].tail(30)
        if len(s_m15) < 20 or len(s_h1) < 15 or len(s_h4) < 10: continue
        rsi = calc_rsi(s_m15['close'])
        atr = calc_atr(s_m15)
        price = bar['close']
        tb, ts = tech_signals(s_h1)
        conf, direction, logic = rule_signal(s_h4, s_h1, s_m15, rsi, tb, ts)
        # Apply weights
        t_adj = 1 if (direction == "BUY" and tb > ts + 3) or (direction == "SELL" and ts > tb + 3) else \
                -1 if (direction == "BUY" and ts > tb + 3) or (direction == "SELL" and tb > ts + 3) else 0
        conf = max(1, min(10, conf + int(t_adj * tech_weight)))
        if direction == "WAIT" or conf < min_conf: continue
        lot = calc_lot(acc.balance, atr)
        sl_d = atr * 1.5
        tp_d = atr * 2.5
        if direction == "BUY":
            sl, tp = round(price - sl_d, 5), round(price + tp_d, 5)
        else:
            sl, tp = round(price + sl_d, 5), round(price - tp_d, 5)
        meta = {"confidence": conf, "rsi": rsi, "atr": atr,
                "session": get_session(h), "tech_buy": tb, "tech_sell": ts}
        acc.open_pos(direction, price, lot, sl, tp, meta, bt)
    if acc.position:
        acc.force_close(df_m15['close'].iloc[-1], df_m15.index[-1])
    return report(acc, months, spread, slippage, commission, min_conf, corr_weight, tech_weight)


def report(acc, months, spread, slip, comm, mc, cw, tw):
    trades = acc.trades
    if not trades:
        return {"profit": 0, "trades": 0, "winrate": 0, "pf": 0, "drawdown": 0}
    df = pd.DataFrame(trades)
    wins = df[df['result'] == 'WIN']
    losses = df[df['result'] == 'LOSS']
    total_p = df['profit'].sum()
    wr = len(wins) / len(df) * 100
    gain = sum(p for p in df['profit'] if p > 0)
    loss = abs(sum(p for p in df['profit'] if p < 0))
    pf = gain / loss if loss > 0 else 0
    # Max drawdown
    peak = acc.equity[0]
    max_dd = 0
    for eq in acc.equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    return {
        "profit": round(total_p, 2), "trades": len(df),
        "wins": len(wins), "losses": len(losses),
        "winrate": round(wr, 1), "pf": round(pf, 2),
        "drawdown": round(max_dd, 1),
        "avg_win": round(wins['profit'].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses['profit'].mean(), 2) if len(losses) > 0 else 0,
        "best": round(df['profit'].max(), 2),
        "worst": round(df['profit'].min(), 2),
        "mc": mc, "cw": cw, "tw": tw,
        "all_trades": trades
    }


def multi_backtest(months=6):
    """Test multiple parameter combinations"""
    print(f"\n  MULTI-BACKTEST: Testing parameter combinations...")
    min_confs = [6, 7, 8]
    results = []
    for mc in min_confs:
        print(f"    Testing MIN_CONFIDENCE={mc}...")
        r = run_backtest(months=months, min_conf=mc)
        if r and r['trades'] > 0:
            results.append(r)
            print(f"      Trades:{r['trades']} WR:{r['winrate']}% P/L:${r['profit']} PF:{r['pf']} DD:{r['drawdown']}%")
    if not results:
        print("  No results.")
        return
    results.sort(key=lambda x: x['profit'], reverse=True)
    print(f"\n  BEST: MC={results[0]['mc']} -> ${results[0]['profit']} ({results[0]['winrate']}% WR)")
    print(f"  WORST: MC={results[-1]['mc']} -> ${results[-1]['profit']}")
    # Save
    os.makedirs("logs", exist_ok=True)
    fp = f"logs/multitest_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fp, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != 'all_trades'} for r in results], f, indent=2)
    print(f"  Saved: {fp}")


def print_report(r):
    if not r or r['trades'] == 0:
        print("  No trades.")
        return
    print(f"""
  {'='*50}
  BACKTEST RESULTS
  {'='*50}
  Profit:        ${r['profit']:.2f}
  Trades:        {r['trades']}
  Wins:          {r['wins']} ({r['winrate']}%)
  Losses:        {r['losses']}
  Profit Factor: {r['pf']}
  Max Drawdown:  {r['drawdown']}%
  Avg Win:       ${r['avg_win']}
  Avg Loss:      ${r['avg_loss']}
  Best:          ${r['best']}
  Worst:         ${r['worst']}
  {'='*50}
    """)
    # Rating
    if r['pf'] > 2 and r['winrate'] > 55 and r['drawdown'] < 15:
        print("  RATING: ⭐⭐⭐⭐⭐ EXCELLENT")
    elif r['pf'] > 1.5 and r['winrate'] > 50 and r['drawdown'] < 25:
        print("  RATING: ⭐⭐⭐⭐ GOOD")
    elif r['pf'] > 1.2 and r['winrate'] > 45:
        print("  RATING: ⭐⭐⭐ AVERAGE")
    elif r['pf'] > 1.0:
        print("  RATING: ⭐⭐ BELOW AVERAGE")
    else:
        print("  RATING: ⭐ NEEDS WORK")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════╗
    ║         BACKTESTER v2.0                  ║
    ║  1 = Single test (fast)                  ║
    ║  2 = Multi-parameter test                ║
    ╚══════════════════════════════════════════╝
    """)
    mode = input("  Mode (1/2): ").strip()
    months = input("  Months (default 6): ").strip()
    months = int(months) if months else 6
    if mode == "2":
        multi_backtest(months)
    else:
        r = run_backtest(months)
        print_report(r)