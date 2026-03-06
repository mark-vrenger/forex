import MetaTrader5 as mt5

if mt5.initialize():
    acc = mt5.account_info()
    print(f"✅ Связь установлена! Счет: {acc.login}, Баланс: {acc.balance}") # cite: image_08605f.png
    mt5.shutdown()
else:
    print("❌ Ошибка связи с MT5")