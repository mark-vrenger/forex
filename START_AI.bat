@echo off
:: Добавляем эту строку для исправления иероглифов
chcp 65001 >nul
title ЗАПУСК ТОРГОВОЙ СИСТЕМЫ ИИ
color 0A

echo [⚙️] СТАРТ МОДУЛЕЙ ИЗ C:\II...

:: Запуск твоих 3-х основных окон
start "АВТОПИЛОТ" powershell -NoExit -Command "python 'C:\ii\pro_auto_trade_pc.py'"
start "АНАЛИТИКА" powershell -NoExit -Command "python 'C:\ii\pro_trading_agent_pc.py'"
start "ТЕСТ СВЯЗИ" powershell -NoExit -Command "python 'C:\ii\test_ai.py'"

echo.
echo ✅ Все окна запущены!
pause