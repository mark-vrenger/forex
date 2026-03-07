import os
from datetime import datetime

def analyze_logs():
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join("logs", f"{today}_logic.txt")
    
    if not os.path.exists(log_file):
        print(f"Лог-файл за сегодня ({log_file}) не найден.")
        return

    stats = {"BUY": 0, "SELL": 0, "WAIT": 0, "CLOSE": 0, "HOLD": 0, "ОШИБКИ": 0}
    total_signals = 0

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "ВЕРДИКТ:" in line or "РЕШЕНИЕ:" in line or "СИГНАЛ:" in line or "]" in line and "(Ув:" in line:
                for key in ["BUY", "SELL", "WAIT", "CLOSE", "HOLD"]:
                    if key in line:
                        stats[key] += 1
                        total_signals += 1
                        break # Считаем только первый найденный сигнал в строке
            if "ОШИБКА" in line or "Error" in line:
                stats["ОШИБКИ"] += 1

    print("\n" + "="*40)
    print(f"📊 СТАТИСТИКА АВТОПИЛОТА ({today})")
    print("="*40)
    print(f"Всего решений ИИ: {total_signals}")
    print(f"🟢 BUY (Покупки):   {stats['BUY']}")
    print(f"🔴 SELL (Продажи):  {stats['SELL']}")
    print(f"⏳ WAIT (Ожидание): {stats['WAIT']}")
    print(f"🔒 CLOSE (Закрыто): {stats['CLOSE']}")
    print(f"🛡 HOLD (Держим):   {stats['HOLD']}")
    print("-" * 40)
    print(f"⚠️ Ошибки/Сбои:     {stats['ОШИБКИ']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    analyze_logs()