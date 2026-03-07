import MetaTrader5 as mt5
import time, os, sys, io
from datetime import datetime
from config import *
from pro_trading_agent_pc import agent

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log_it(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", f"{today}_logic.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass

def start_autopilot():
    if not mt5.initialize(): 
        log_it("!!! КРИТИЧЕСКАЯ ОШИБКА: MT5 не запущен!"); return
    
    mt5.market_book_add(SYMBOL)
    log_it(f"БОЕВОЙ РЕЖИМ (Скальпинг + Среднесрок) | Счет: {mt5.account_info().login}")

    while True:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            sym = mt5.symbol_info(SYMBOL)
            if not tick or not sym: time.sleep(5); continue

            spr = (tick.ask - tick.bid) / sym.point
            
            # Загружаем все 6 таймфреймов
            m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 50)
            m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 50)
            m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
            m30 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, 50)
            h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 50)
            h4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 50)
            
            # Ждем только старшие ТФ, если M1 не прогрузился - ничего страшного
            if h1 is None or h4 is None:
                log_it("Ожидание синхронизации данных H1/H4..."); time.sleep(10); continue

            log_it("--- ЗАПРОС КОНСИЛИУМУ ИИ ---")
            analysis = agent.analyze_market(m1, m5, m15, m30, h1, h4, tick.ask, "Book: OK", spr)
            conf, dir, logic = agent.parse_response(analysis)
            
            log_it(f"РЕШЕНИЕ: {dir} (Ув:{conf})")
            log_it(f"ЛОГИКА: {logic}")

            time.sleep(300) 
        except Exception as e:
            log_it(f"ОШИБКА В ЦИКЛЕ: {e}"); time.sleep(60)

if __name__ == "__main__":
    start_autopilot()