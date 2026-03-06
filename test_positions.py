import MetaTrader5 as mt5

if not mt5.initialize():
    print("Ошибка MT5")
else:
    # ВСЕ позиции на счете
    positions = mt5.positions_get()
    if positions:
        print(f"Всего позиций: {len(positions)}\n")
        for p in positions:
            direction = "BUY" if p.type == 0 else "SELL"
            print(f"  {p.symbol} | {direction} {p.volume} лот")
            print(f"  Вход: {p.price_open} | SL: {p.sl} | TP: {p.tp}")
            print(f"  P/L: {p.profit} | Magic: {p.magic}")
            print(f"  Комментарий: {p.comment}")
            print("  ---")
    else:
        print("Нет открытых позиций")
    mt5.shutdown()