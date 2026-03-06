"""
database.py — SQLite хранилище v2.0
Добавлена таблица council_sessions для протоколов Консилиума.
Добавлен метод get_agent_accuracy() для оценки точности агентов.
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
from config import DB_FILE


class Database:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()

        # ── Сделки ──────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            signal_id INTEGER,
            time TEXT,
            direction TEXT,
            lot REAL,
            entry_price REAL,
            sl REAL,
            tp REAL,
            confidence INTEGER,
            ai_logic TEXT,
            rsi REAL,
            atr REAL,
            session TEXT,
            spread REAL,
            slippage REAL,
            tech_buy INTEGER,
            tech_sell INTEGER,
            corr_buy INTEGER,
            corr_sell INTEGER,
            result TEXT,
            profit REAL,
            commission REAL,
            swap REAL,
            close_time TEXT,
            close_reason TEXT,
            duration_minutes INTEGER,
            dynamic_risk_pct REAL
        )""")

        # ── Сигналы ─────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            price REAL,
            direction TEXT,
            confidence INTEGER,
            final_confidence INTEGER,
            rsi REAL,
            atr REAL,
            spread REAL,
            session TEXT,
            tech_buy INTEGER,
            tech_sell INTEGER,
            tech_summary TEXT,
            corr_buy INTEGER,
            corr_sell INTEGER,
            corr_verdict TEXT,
            corr_summary TEXT,
            sentiment_summary TEXT,
            corr_adj INTEGER,
            tech_adj INTEGER,
            logic TEXT,
            acted INTEGER DEFAULT 0,
            dynamic_risk_pct REAL
        )""")

        # ── Ошибки ──────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            type TEXT,
            message TEXT
        )""")

        # ── Heartbeat ───────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS heartbeat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            balance REAL,
            equity REAL,
            open_pnl REAL,
            positions INTEGER,
            status TEXT
        )""")

        # ── Дневная статистика ───────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            start_balance REAL,
            end_balance REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            total_profit REAL,
            total_commission REAL,
            total_swap REAL,
            max_drawdown REAL,
            profit_factor REAL,
            avg_confidence REAL,
            signals_generated INTEGER,
            signals_acted INTEGER,
            best_trade REAL,
            worst_trade REAL
        )""")

        # ── Лог оптимизации ─────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS optimization_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            best_corr_weight REAL,
            best_tech_weight REAL,
            best_min_conf INTEGER,
            expected_profit REAL,
            expected_winrate REAL,
            trades_analyzed INTEGER,
            applied INTEGER DEFAULT 0
        )""")

        # ── Сессии Консилиума ─────────────────── NEW
        c.execute("""CREATE TABLE IF NOT EXISTS council_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            price REAL,
            signal TEXT,
            confidence INTEGER,
            consensus_pct REAL,
            blocked_by TEXT,
            block_reason TEXT,
            corr_adj INTEGER,
            sl REAL,
            tp REAL,
            lot REAL,
            votes_r1 TEXT,
            votes_r2 TEXT,
            acted INTEGER DEFAULT 0,
            trade_result TEXT,
            trade_profit REAL
        )""")

        self.conn.commit()

    # ── SIGNALS ─────────────────────────────────

    def record_signal(self, price, direction, confidence, final_conf,
                      rsi, atr, spread, session,
                      tech_buy, tech_sell, tech_summary,
                      corr_buy, corr_sell, corr_verdict, corr_summary,
                      sentiment_summary,
                      corr_adj, tech_adj, logic, acted, dynamic_risk=0):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO signals (time,price,direction,confidence,"
            "final_confidence,rsi,atr,spread,session,"
            "tech_buy,tech_sell,tech_summary,"
            "corr_buy,corr_sell,corr_verdict,corr_summary,"
            "sentiment_summary,"
            "corr_adj,tech_adj,logic,acted,dynamic_risk_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), price, direction, confidence,
             final_conf, rsi, atr, spread, session,
             tech_buy, tech_sell, str(tech_summary)[:500],
             corr_buy, corr_sell, corr_verdict, str(corr_summary)[:500],
             str(sentiment_summary)[:500],
             corr_adj, tech_adj, logic, 1 if acted else 0, dynamic_risk)
        )
        self.conn.commit()
        return c.lastrowid

    # ── TRADES ──────────────────────────────────

    def record_trade(self, trade_data: dict):
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO trades VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade_data.get("id"), trade_data.get("signal_id"),
                trade_data.get("time"),
                trade_data.get("direction"), trade_data.get("lot"),
                trade_data.get("entry_price"), trade_data.get("sl"),
                trade_data.get("tp"), trade_data.get("confidence"),
                trade_data.get("ai_logic"), trade_data.get("rsi"),
                trade_data.get("atr"), trade_data.get("session"),
                trade_data.get("spread", 0), trade_data.get("slippage", 0),
                trade_data.get("tech_buy", 0), trade_data.get("tech_sell", 0),
                trade_data.get("corr_buy", 0), trade_data.get("corr_sell", 0),
                trade_data.get("result"), trade_data.get("profit"),
                trade_data.get("commission"), trade_data.get("swap"),
                trade_data.get("close_time"), trade_data.get("close_reason"),
                trade_data.get("duration_minutes"),
                trade_data.get("dynamic_risk_pct", 0)
            )
        )
        self.conn.commit()

    # ── COUNCIL SESSIONS ─────────────────────── NEW

    def record_council_session(self, session_data: dict) -> int:
        """Сохраняет полный протокол сессии Консилиума."""
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO council_sessions "
            "(time,price,signal,confidence,consensus_pct,"
            "blocked_by,block_reason,corr_adj,sl,tp,lot,"
            "votes_r1,votes_r2,acted) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_data.get("time", datetime.now().isoformat()),
                session_data.get("price", 0),
                session_data.get("signal", "WAIT"),
                session_data.get("confidence", 0),
                session_data.get("consensus_pct", 0),
                session_data.get("blocked_by", ""),
                session_data.get("block_reason", ""),
                session_data.get("corr_adj", 0),
                session_data.get("sl", 0),
                session_data.get("tp", 0),
                session_data.get("lot", 0),
                json.dumps(session_data.get("votes_r1", []), ensure_ascii=False),
                json.dumps(session_data.get("votes_r2", []), ensure_ascii=False),
                1 if session_data.get("signal") in ("BUY", "SELL") else 0,
            )
        )
        self.conn.commit()
        return c.lastrowid

    def update_council_session_result(self, session_id: int, result: str, profit: float):
        """Обновляет результат сделки для сессии Консилиума."""
        c = self.conn.cursor()
        c.execute(
            "UPDATE council_sessions SET trade_result=?, trade_profit=? WHERE id=?",
            (result, profit, session_id)
        )
        self.conn.commit()

    def get_agent_accuracy(self) -> dict:
        """
        Возвращает точность каждого агента Консилиума.
        Сравнивает сигнал агента с итоговым результатом сделки.
        """
        c = self.conn.cursor()
        c.execute(
            "SELECT votes_r1, signal, trade_result "
            "FROM council_sessions "
            "WHERE trade_result IS NOT NULL AND votes_r1 IS NOT NULL"
        )
        rows = c.fetchall()
        if not rows:
            return {}

        stats = {}
        for row in rows:
            try:
                votes = json.loads(row[0])
                final_signal = row[1]
                trade_result = row[2]
                for v in votes:
                    name = v.get("name", "UNKNOWN")
                    agent_signal = v.get("signal", "WAIT")
                    if name not in stats:
                        stats[name] = {"correct": 0, "total": 0}
                    stats[name]["total"] += 1
                    # Правильно если агент согласился с финалом и сделка выиграна
                    # или если агент сказал WAIT/противоположный и сделка проиграна
                    if ((agent_signal == final_signal and trade_result == "WIN") or
                            (agent_signal != final_signal and trade_result == "LOSS")):
                        stats[name]["correct"] += 1
            except Exception:
                continue

        result = {}
        for name, s in stats.items():
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            result[name] = {
                "accuracy": round(acc, 1),
                "correct": s["correct"],
                "total": s["total"],
            }
        return result

    # ── ERRORS ──────────────────────────────────

    def record_error(self, error_type: str, message: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO errors (time,type,message) VALUES (?,?,?)",
            (datetime.now().isoformat(), error_type, str(message)[:500])
        )
        self.conn.commit()

    # ── HEARTBEAT ───────────────────────────────

    def record_heartbeat(self, balance, equity, open_pnl, positions, status):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO heartbeat (time,balance,equity,open_pnl,positions,status) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(), balance, equity, open_pnl, positions, status)
        )
        self.conn.commit()

    # ── OPTIMIZATION ────────────────────────────

    def record_optimization(self, cw, tw, mc, profit, wr, trades, applied):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO optimization_log "
            "(time,best_corr_weight,best_tech_weight,best_min_conf,"
            "expected_profit,expected_winrate,trades_analyzed,applied) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), cw, tw, mc, profit, wr, trades,
             1 if applied else 0)
        )
        self.conn.commit()

    # ── DAILY STATS ─────────────────────────────

    def update_daily_stats(self, date_str=None):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        c = self.conn.cursor()
        c.execute(
            "SELECT * FROM trades WHERE time LIKE ? AND result IS NOT NULL",
            (date_str + "%",)
        )
        trades = c.fetchall()
        c.execute("SELECT * FROM signals WHERE time LIKE ?", (date_str + "%",))
        sigs = c.fetchall()
        if not trades and not sigs:
            return
        total = len(trades)
        wins    = len([t for t in trades if t[19] == "WIN"])
        losses  = len([t for t in trades if t[19] == "LOSS"])
        profits = [t[20] for t in trades if t[20] is not None]
        total_profit = sum(profits) if profits else 0
        comms    = [t[21] for t in trades if t[21] is not None]
        swaps    = [t[22] for t in trades if t[22] is not None]
        total_comm = sum(comms) if comms else 0
        total_swap = sum(swaps) if swaps else 0
        best  = max(profits) if profits else 0
        worst = min(profits) if profits else 0
        win_sum  = sum(p for p in profits if p > 0)
        loss_sum = abs(sum(p for p in profits if p < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 0
        confs = [t[8] for t in trades if t[8] is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0
        signals_total = len(sigs)
        signals_acted = len([s for s in sigs if s[21] == 1])
        equity = 0
        peak = 0
        max_dd = 0
        for p in profits:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        c.execute(
            "SELECT balance FROM heartbeat WHERE time LIKE ? ORDER BY time DESC LIMIT 1",
            (date_str + "%",)
        )
        hb = c.fetchone()
        end_bal   = hb[0] if hb else 0
        start_bal = end_bal - total_profit if end_bal else 0
        c.execute(
            "INSERT OR REPLACE INTO daily_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (date_str, round(start_bal, 2), round(end_bal, 2),
             total, wins, losses, round(total_profit, 2),
             round(total_comm, 2), round(total_swap, 2),
             round(max_dd, 2), round(pf, 2), round(avg_conf, 1),
             signals_total, signals_acted, round(best, 2), round(worst, 2))
        )
        self.conn.commit()

    # ── QUERY HELPERS ───────────────────────────

    def get_situation_history(self, direction, rsi_range=(30, 70)):
        c = self.conn.cursor()
        c.execute(
            "SELECT direction, result, profit, confidence, rsi, ai_logic "
            "FROM trades WHERE direction=? AND rsi BETWEEN ? AND ? "
            "AND result IS NOT NULL ORDER BY time DESC LIMIT 5",
            (direction, rsi_range[0], rsi_range[1])
        )
        rows = c.fetchall()
        if not rows:
            return "No similar past trades."
        lines = ["Similar past trades:"]
        for r in rows:
            lines.append(
                f"  {r[0]}|{r[1]}|${r[2]:.2f}|C:{r[3]}|RSI:{r[4]:.0f}|{str(r[5])[:60]}"
            )
        return "\n".join(lines)

    def get_last_optimization_date(self):
        c = self.conn.cursor()
        c.execute("SELECT time FROM optimization_log ORDER BY time DESC LIMIT 1")
        row = c.fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def get_closed_trades_count(self) -> int:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades WHERE result IS NOT NULL")
        return c.fetchone()[0]

    def get_trades_df(self)         -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM trades ORDER BY time", self.conn)

    def get_signals_df(self)        -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM signals ORDER BY time", self.conn)

    def get_daily_stats_df(self)    -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM daily_stats ORDER BY date", self.conn)

    def get_errors_df(self)         -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM errors ORDER BY time", self.conn)

    def get_heartbeat_df(self)      -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM heartbeat ORDER BY time", self.conn)

    def get_council_sessions_df(self) -> pd.DataFrame:
        """Возвращает все сессии Консилиума."""
        return pd.read_sql("SELECT * FROM council_sessions ORDER BY time", self.conn)


# ─────────────────────────────────────────
#  SINGLETON — безопасная инициализация
# ─────────────────────────────────────────

try:
    db = Database()
except Exception as _e:
    import sys
    print(f"[Database] Init error: {_e}", file=sys.stderr)
    db = None  # type: ignore
