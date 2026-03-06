import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 Error")
else:
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.bid if pos.type == 0 else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "comment": "MANUAL CLOSE",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"CLOSED: {pos.symbol} {pos.volume} lot | P/L: {pos.profit}")
            else:
                code = result.retcode if result else "None"
                print(f"ERROR: {code}")
    else:
        print("No open positions")
    mt5.shutdown()