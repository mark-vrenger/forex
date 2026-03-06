import json
import os
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from config import TRADE_HISTORY_FILE, MAGIC, SYMBOL


class TradeMemory:
    def __init__(self):
        self.history_file = TRADE_HISTORY_FILE
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def record_trade(self, direction, lot, entry_price, sl, tp,
                     confidence, ai_logic, rsi, atr, session,
                     spread=0, slippage=0):
        trade = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "time": datetime.now().isoformat(),
            "direction": direction,
            "lot": lot,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "ai_logic": ai_logic,
            "rsi_at_entry": rsi,
            "atr_at_entry": atr,
            "session": session,
            "spread_at_entry": spread,
            "slippage": slippage,
            "result": None,
            "profit": None,
            "commission": None,
            "swap": None,
            "close_time": None,
            "close_reason": None,
            "duration_minutes": None
        }
        history = self._load()
        history.append(trade)
        self._save(history)
        return trade["id"]

    def get_performance_summary(self, last_n=20):
        history = self._load()
        closed = [t for t in history if t["result"] is not None]
        if not closed:
            return "No trade history. Be very cautious with first trades."
        recent = closed[-last_n:]
        wins = [t for t in recent if t["result"] == "WIN"]
        losses = [t for t in recent if t["result"] == "LOSS"]
        total_profit = sum(t.get("profit", 0) for t in recent if t.get("profit"))
        total_commission = sum(t.get("commission", 0) for t in recent if t.get("commission"))
        total_swap = sum(t.get("swap", 0) for t in recent if t.get("swap"))
        buy_trades = [t for t in recent if t["direction"] == "BUY"]
        sell_trades = [t for t in recent if t["direction"] == "SELL"]
        buy_wins = len([t for t in buy_trades if t["result"] == "WIN"])
        sell_wins = len([t for t in sell_trades if t["result"] == "WIN"])
        avg_win = 0
        avg_loss = 0
        if wins:
            avg_win = sum(t.get("profit", 0) for t in wins) / len(wins)
        if losses:
            avg_loss = sum(t.get("profit", 0) for t in losses) / len(losses)
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        losing_streak = 0
        current_streak = 0
        for t in recent:
            if t["result"] == "LOSS":
                current_streak += 1
                losing_streak = max(losing_streak, current_streak)
            else:
                current_streak = 0
        # Session analysis
        session_stats = {}
        for t in recent:
            s = t.get("session", "unknown")
            if s not in session_stats:
                session_stats[s] = {"wins": 0, "losses": 0, "pnl": 0}
            if t["result"] == "WIN":
                session_stats[s]["wins"] += 1
            else:
                session_stats[s]["losses"] += 1
            session_stats[s]["pnl"] += t.get("profit", 0)
        # Confidence analysis
        high_conf = [t for t in recent if t.get("confidence", 0) >= 8]
        high_conf_wins = len([t for t in high_conf if t["result"] == "WIN"])
        low_conf = [t for t in recent if t.get("confidence", 0) <= 7]
        low_conf_wins = len([t for t in low_conf if t["result"] == "WIN"])
        
        total_trades_count = len(recent)
        
        summary = (
            f"=== LAST {total_trades_count} TRADES ===\n"
            f"Wins: {len(wins)} | Losses: {len(losses)} | "
            f"Winrate: {len(wins)/total_trades_count*100:.0f}%\n"
            f"Net P/L: ${total_profit:.2f} | "
            f"Commission: ${total_commission:.2f} | Swap: ${total_swap:.2f}\n"
            f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f} | "
            f"Profit Factor: {profit_factor:.2f}\n"
            f"BUY: {buy_wins}/{len(buy_trades)} wins | "
            f"SELL: {sell_wins}/{len(sell_trades)} wins\n"
            f"Max losing streak: {losing_streak} | Current: {current_streak}\n"
        )
        # Session breakdown
        session_lines = []
        for s, st in session_stats.items():
            total_s = st["wins"] + st["losses"]
            wr_s = (st["wins"] / total_s * 100) if total_s > 0 else 0
            session_lines.append(f"  {s}: {st['wins']}/{total_s} wins ({wr_s:.0f}%) P/L: ${st['pnl']:.2f}")
        if session_lines:
            summary += "Sessions:\n" + "\n".join(session_lines) + "\n"
        # Confidence breakdown
        if high_conf:
            summary += f"High confidence (8+): {high_conf_wins}/{len(high_conf)} wins\n"
        if low_conf:
            summary += f"Low confidence (7): {low_conf_wins}/{len(low_conf)} wins\n"
        # Warnings
        warnings = []
        if current_streak >= 3:
            warnings.append("CRITICAL: 3+ losses in a row - only enter with confidence 9+!")
        if current_streak >= 5:
            warnings.append("EMERGENCY: 5+ losses! Consider stopping for the day.")
        if len(buy_trades) >= 3 and buy_wins / len(buy_trades) < 0.3:
            warnings.append("BUY trades losing (<30%) - strongly avoid BUY.")
        if len(sell_trades) >= 3 and sell_wins / len(sell_trades) < 0.3:
            warnings.append("SELL trades losing (<30%) - strongly avoid SELL.")
        if profit_factor < 1.0 and total_trades_count > 5:
            warnings.append(f"Profit factor {profit_factor:.2f} < 1.0 - system is losing money!")
        for s, st in session_stats.items():
            total_s = st["wins"] + st["losses"]
            if total_s >= 3 and st["wins"] / total_s < 0.3:
                warnings.append(f"Session '{s}' is unprofitable - consider pausing.")
        if high_conf and len(high_conf) >= 3:
            hcwr = high_conf_wins / len(high_conf)
            if hcwr < 0.4:
                warnings.append("Even high-confidence trades are losing - review strategy!")
        if warnings:
            summary += "WARNINGS:\n" + "\n".join(warnings)
        else:
            summary += "Status: Normal"
        return summary

    def get_last_trades_for_prompt(self, n=5):
        history = self._load()
        closed = [t for t in history if t["result"] is not None][-n:]
        if not closed:
            return "No completed trades."
        lines = []
        for t in closed:
            duration = t.get("duration_minutes", "?")
            lines.append(
                f"  {t['direction']} | {t['result']} | "
                f"P/L: ${t.get('profit', 0):.2f} | Conf: {t.get('confidence', 0)} | "
                f"RSI: {t.get('rsi_at_entry', 0):.0f} | "
                f"Session: {t.get('session', '?')} | Duration: {duration}min"
            )
        return "\n".join(lines)

    def sync_closed_trades(self):
        history = self._load()
        open_trades = [t for t in history if t["result"] is None]
        if not open_trades:
            return
        week_ago = datetime.now() - timedelta(days=7)
        deals = mt5.history_deals_get(week_ago, datetime.now())
        if not deals:
            return
        changed = False
        for trade in open_trades:
            for deal in deals:
                if deal.magic == MAGIC and deal.symbol == SYMBOL and deal.entry == 1:
                    trade["result"] = "WIN" if deal.profit > 0 else "LOSS"
                    trade["profit"] = deal.profit
                    trade["commission"] = deal.commission
                    trade["swap"] = deal.swap
                    trade["close_time"] = datetime.now().isoformat()
                    trade["close_reason"] = "TP/SL"
                    try:
                        open_time = datetime.fromisoformat(trade["time"])
                        trade["duration_minutes"] = int((datetime.now() - open_time).total_seconds() / 60)
                    except Exception:
                        trade["duration_minutes"] = 0
                    changed = True
        if changed:
            self._save(history)

    def update_closed_trade(self, trade_id, close_price, profit):
        history = self._load()
        for t in history:
            if t["id"] == trade_id:
                t["result"] = "WIN" if profit > 0 else "LOSS"
                t["profit"] = profit
                t["close_time"] = datetime.now().isoformat()
                t["close_reason"] = "TRAINING"
                break
        self._save(history)

    def get_daily_realized_loss(self):
        history = self._load()
        today = datetime.now().strftime("%Y-%m-%d")
        today_closed = [
            t for t in history
            if t.get("result") is not None
            and t.get("close_time", "").startswith(today)
        ]
        return sum(t.get("profit", 0) for t in today_closed if t.get("profit", 0) < 0)

    def _load(self):
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, data):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


memory = TradeMemory()