"""
pro_auto_trade_pc.py — Боевой автопилот v3.0
Интегрирован Консилиум трейдеров. Реальное открытие/закрытие позиций.
"""

import MetaTrader5 as mt5
import time, os, sys, io
from datetime import datetime
from config import (
    SYMBOL, MAGIC, ACCOUNT_ID,
    MAX_DAILY_LOSS, MAX_DAILY_TRADES, MAX_OPEN_POSITIONS,
    MAX_SPREAD_PIPS, CLOSE_FRIDAY_HOUR,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    LOG_DIR,
)
from council import council, send_telegram
from signal_engine import signals
from correlation_analyzer import correlator
from news_manager import get_upcoming_news
from trailing_manager import trailing
from database import db

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "   ", "WARN": "⚠️ ", "ERR": "❌ ", "OK": "✅ ", "TRADE": "💰 "}.get(level, "   ")
    line = f"[{ts}] {prefix}{msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        with open(os.path.join(LOG_DIR, f"{today}_autopilot.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────
#  MT5 HELPERS
# ─────────────────────────────────────────

def get_account():
    acc = mt5.account_info()
    if not acc:
        return None, None, None
    return acc.balance, acc.equity, acc.profit


def get_open_position():
    """Возвращает первую открытую позицию по нашему MAGIC, или None."""
    pos = mt5.positions_get(symbol=SYMBOL)
    if not pos:
        return None
    for p in pos:
        if p.magic == MAGIC:
            return p
    return None


def get_spread_pips() -> float:
    tick = mt5.symbol_info_tick(SYMBOL)
    sym = mt5.symbol_info(SYMBOL)
    if not tick or not sym:
        return 999.0
    return (tick.ask - tick.bid) / sym.point


def get_session_name(h: int) -> str:
    if 7 <= h < 10:   return "London Open"
    if 10 <= h < 13:  return "London"
    if 13 <= h < 17:  return "London+NY"
    if 17 <= h < 21:  return "New York"
    return "Asia"


def calc_rsi(closes, period=14):
    import pandas as pd
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return round(float(v), 2) if not pd.isna(v) else 50.0


def calc_atr(m15, period=14):
    import pandas as pd
    df = pd.DataFrame(m15)
    if 'close' not in df.columns and len(df.columns) >= 5:
        df.columns = ['time','open','high','low','close','tick_volume','spread','real_volume'][:len(df.columns)]
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    v = df['tr'].rolling(period).mean().iloc[-1]
    return float(v) if v == v else 0.0010


# ─────────────────────────────────────────
#  DAILY LIMITS CHECK
# ─────────────────────────────────────────

_daily_trades = 0
_daily_pnl = 0.0
_daily_date = ""


def reset_daily_if_new_day():
    global _daily_trades, _daily_pnl, _daily_date
    today = datetime.now().strftime("%Y-%m-%d")
    if today != _daily_date:
        _daily_trades = 0
        _daily_pnl = 0.0
        _daily_date = today
        log(f"Новый торговый день: {today}")


def can_open_trade() -> tuple:
    """Возвращает (True/False, причина)."""
    reset_daily_if_new_day()
    now = datetime.now()
    h, wd = now.hour, now.weekday()

    # Выходные
    if wd in (5, 6):
        return False, "Выходной день"
    # Азиатская сессия (по умолчанию не торгуем)
    if h < 7 or h >= 22:
        return False, f"Нерабочий час ({h}:00)"
    # Закрытие в пятницу
    if wd == 4 and h >= CLOSE_FRIDAY_HOUR:
        return False, f"Пятничное закрытие (>{CLOSE_FRIDAY_HOUR}:00)"
    # Дневной убыток
    if _daily_pnl <= -MAX_DAILY_LOSS:
        return False, f"Дневной лимит убытка: ${_daily_pnl:.2f}"
    # Дневное количество сделок
    if _daily_trades >= MAX_DAILY_TRADES:
        return False, f"Дневной лимит сделок: {_daily_trades}"
    # Уже открытая позиция
    if get_open_position():
        return False, "Уже есть открытая позиция"

    return True, "OK"


# ─────────────────────────────────────────
#  TRADE EXECUTION
# ─────────────────────────────────────────

def open_trade(direction: str, lot: float, sl: float, tp: float,
               confidence: int, logic: str) -> bool:
    global _daily_trades
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if not tick or not info:
        log("Нет тика/инфо символа", "ERR")
        return False

    price = tick.ask if direction == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl, info.digits),
        "tp": round(tp, info.digits),
        "deviation": 20,
        "magic": MAGIC,
        "comment": f"COUNCIL_C{confidence}",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        _daily_trades += 1
        msg = (f"{direction} {lot} лот | Цена: {price:.5f} | "
               f"SL: {sl:.5f} | TP: {tp:.5f} | Уверенность: {confidence}")
        log(f"СДЕЛКА ОТКРЫТА: {msg}", "TRADE")
        # Запись в БД
        db.record_heartbeat(0, 0, 0, 1, f"OPEN_{direction}")
        return True
    else:
        code = result.retcode if result else "None"
        log(f"Ошибка открытия: retcode={code}", "ERR")
        db.record_error("OPEN_TRADE", f"retcode={code} dir={direction} lot={lot}")
        return False


def close_trade(position, reason: str = "COUNCIL") -> bool:
    global _daily_pnl
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return False

    close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    price = tick.bid if position.type == 0 else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC,
        "comment": f"CLOSE_{reason}",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        pnl = position.profit
        _daily_pnl += pnl
        log(f"СДЕЛКА ЗАКРЫТА: {reason} | P/L: ${pnl:.2f}", "TRADE" if pnl >= 0 else "WARN")
        db.record_heartbeat(0, 0, pnl, 0, f"CLOSE_{reason}")
        return True
    return False


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

def start_autopilot():
    if not mt5.initialize():
        print("❌ Не удалось подключиться к MT5")
        return

    acc_info = mt5.account_info()
    log(f"БОЕВОЙ РЕЖИМ ВКЛЮЧЁН | Счёт: {acc_info.login} | Баланс: ${acc_info.balance:.2f}", "OK")
    send_telegram(f"🚀 Автопилот запущен\nСчёт: {acc_info.login}\nБаланс: ${acc_info.balance:.2f}")

    cycle = 0

    while True:
        try:
            cycle += 1
            reset_daily_if_new_day()
            now = datetime.now()
            balance, equity, open_pnl = get_account()

            # ── Heartbeat каждые 30 минут ──────────────
            if cycle % 6 == 0:
                pos_count = len(mt5.positions_get(symbol=SYMBOL) or [])
                db.record_heartbeat(balance or 0, equity or 0, open_pnl or 0,
                                    pos_count, "RUNNING")

            # ── Данные с рынка ─────────────────────────
            m5  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5,  0, 50)
            m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
            h1  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1,  0, 50)
            h4  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4,  0, 50)
            tick = mt5.symbol_info_tick(SYMBOL)

            if not tick or any(x is None for x in [m5, h1]):
                log("Синхронизация данных...", "WARN")
                time.sleep(15)
                continue

            price  = tick.ask
            spread = get_spread_pips()
            h      = now.hour
            session = get_session_name(h)

            # ── Управление открытой позицией ───────────
            position = get_open_position()
            if position:
                import pandas as pd
                m15_df = pd.DataFrame(m15)
                if 'close' not in m15_df.columns and len(m15_df.columns) >= 5:
                    m15_df.columns = ['time','open','high','low','close',
                                      'tick_volume','spread','real_volume'][:len(m15_df.columns)]
                atr_val = calc_atr(m15, 14)
                trailing.set_atr(position.ticket, atr_val)
                actions = trailing.manage_positions()
                for a in actions:
                    log(f"Trailing: {a}", "OK")

                # Проверка разворотного сигнала (раз в 5 мин)
                tech_sum, tb, ts = signals.analyze_all()
                corr_sum, cb, cs, _ = correlator.analyze()
                news = get_upcoming_news(30)
                rsi = calc_rsi(m15_df['close'].tolist(), 14)
                atr = calc_atr(m15, 14)

                decision = council.run_session(
                    m5, m15, h1, h4, price, spread,
                    rsi, atr, session, balance or 10000,
                    tech_sum, cb, cs, str(corr_sum), news
                )
                council.print_protocol(decision)
                council.save_protocol(decision)

                # Закрыть при противоположном сигнале с высокой уверенностью
                pos_dir = "BUY" if position.type == 0 else "SELL"
                opposite = "SELL" if pos_dir == "BUY" else "BUY"
                if decision.signal == opposite and decision.confidence >= 8:
                    log(f"Разворот: {pos_dir}→{opposite} (уверенность {decision.confidence})", "WARN")
                    close_trade(position, "REVERSAL")

                time.sleep(300)
                continue

            # ── Проверка условий для новой сделки ─────
            can_trade, reason = can_open_trade()
            if not can_trade:
                log(f"Торговля заблокирована: {reason}")
                time.sleep(300)
                continue

            # ── Технический и корреляционный анализ ───
            tech_sum, tb, ts = signals.analyze_all()
            corr_sum, cb, cs, _ = correlator.analyze()
            news = get_upcoming_news(30)

            import pandas as pd
            m15_df = pd.DataFrame(m15)
            if 'close' not in m15_df.columns and len(m15_df.columns) >= 5:
                m15_df.columns = ['time','open','high','low','close',
                                  'tick_volume','spread','real_volume'][:len(m15_df.columns)]

            rsi = calc_rsi(m15_df['close'].tolist(), 14)
            atr = calc_atr(m15, 14)

            log(f"RSI:{rsi:.1f} ATR:{atr:.5f} Spread:{spread:.1f}p | "
                f"Tech B:{tb}/S:{ts} Corr B:{cb}/S:{cs}")

            # ── КОНСИЛИУМ ──────────────────────────────
            log("Созываю Консилиум...")
            decision = council.run_session(
                m5, m15, h1, h4, price, spread,
                rsi, atr, session, balance or 10000,
                tech_sum, cb, cs, str(corr_sum), news
            )
            council.print_protocol(decision)
            council.save_protocol(decision)

            # Запись сигнала в БД
            db.record_signal(
                price=price, direction=decision.signal,
                confidence=decision.confidence, final_conf=decision.confidence,
                rsi=rsi, atr=atr, spread=spread, session=session,
                tech_buy=tb, tech_sell=ts, tech_summary=str(tech_sum),
                corr_buy=cb, corr_sell=cs,
                corr_verdict="", corr_summary=str(corr_sum),
                sentiment_summary="",
                corr_adj=decision.corr_adj, tech_adj=0,
                logic=decision.votes[0].reasoning if decision.votes else "",
                acted=decision.signal in ("BUY", "SELL")
            )

            # Telegram
            tg_msg = council.build_telegram_message(decision)
            send_telegram(tg_msg)

            # ── Открытие сделки ────────────────────────
            if decision.signal in ("BUY", "SELL") and not decision.blocked_by:
                log(f"ОТКРЫВАЮ: {decision.signal} | Уверенность: {decision.confidence}/10", "OK")
                success = open_trade(
                    direction=decision.signal,
                    lot=decision.lot,
                    sl=decision.sl,
                    tp=decision.tp,
                    confidence=decision.confidence,
                    logic=decision.votes[0].reasoning if decision.votes else "",
                )
                if success:
                    send_telegram(
                        f"✅ СДЕЛКА ОТКРЫТА: {decision.signal} {decision.lot} лот\n"
                        f"SL: {decision.sl:.5f} | TP: {decision.tp:.5f}"
                    )
            else:
                log(f"ПРОПУСК: {decision.signal} | Блок: {decision.blocked_by or 'нет'} | "
                    f"Консенсус: {decision.consensus_pct:.0f}%")

            time.sleep(300)   # 5 минут между циклами

        except KeyboardInterrupt:
            log("Останов по Ctrl+C", "WARN")
            send_telegram("⛔ Автопилот остановлен вручную")
            break
        except Exception as e:
            log(f"Ошибка цикла: {e}", "ERR")
            db.record_error("MAIN_LOOP", str(e))
            time.sleep(60)

    mt5.shutdown()
    log("MT5 отключён. Автопилот завершён.")


if __name__ == "__main__":
    start_autopilot()
