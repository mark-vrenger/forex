# 🔌 ИНТЕГРАЦИЯ — Как подключить новые модули

## Быстрый старт: что куда вставить

### 1. `slow_trainer.py` — начало `start_training()`

```python
from data_steward    import steward
from flight_recorder import flight_recorder
from machine_state   import machine
from arbiter         import arbiter, make_vote
from diversity_monitor import diversity_monitor, drift_watchdog, expectancy_tracker
from regime_detector import regime_detector

# После загрузки CSV — очищаем данные:
if df_m15 is not None:
    result = steward.validate_csv(df_m15, "M15")
    df_m15 = result["df"]
if df_h1 is not None:
    result = steward.validate_csv(df_h1, "H1")
    df_h1 = result["df"]
# ... и остальные TF
```

### 2. `slow_trainer.py` — внутри цикла (перед AI вызовом)

```python
# ── Machine State tick ─────────────────────────────
ms = machine.tick(
    balance      = stats["balance"],
    last_result  = last_trade_result,   # "WIN"/"LOSS"/None
    data_quality = 1.0,                 # до DataSteward
    bars_added   = 1,
)
if not ms["can_trade"] and ms["state"] not in ("SAFE_MODE",):
    idx += 1
    continue

# ── Валидация данных DataSteward ───────────────────
tf_data = {"M15": m15_df, "H1": h1_df, "H4": h4_df}
dq = steward.validate_batch(tf_data, current_time=target_time)

# Передаём quality в machine state
ms = machine.tick(stats["balance"], data_quality=dq["quality_score"])

if not dq["tradeable"]:
    idx += 1
    continue

# ── Детектор режима ────────────────────────────────
regime = regime_detector.detect(h1_df, "H1")
# Передаём режим агентам как контекст

# ── Арбитр вместо простого голосования ────────────
# Формируем typed votes из ответов агентов:
votes = [
    make_vote("IMPULSE", impulse_signal, impulse_conf, impulse_reasoning),
    make_vote("TREND",   trend_signal,   trend_conf,   trend_reasoning),
    make_vote("ANALYST", analyst_signal, analyst_conf, analyst_reasoning),
]

arbiter_result = arbiter.decide(
    agent_votes   = votes,
    data_quality  = dq["quality_score"],
    event_risk    = news_blocked,
    machine_state = ms["state"],
    current_regime = regime,
    custom_signals = custom_result,
)

# Итоговый сигнал из арбитра вместо простого conf/direction:
final_signal = arbiter_result["signal"]   # "BUY"/"SELL"/"NO_TRADE"
final_score  = arbiter_result["score"]

if final_signal == "NO_TRADE":
    idx += 1
    continue

# ── Flight Recorder — снапшот ──────────────────────
decision_id = f"D_{dt_str.replace(' ','_').replace(':','')}_{idx}"
flight_recorder.record_decision(
    decision_id   = decision_id,
    signal        = final_signal,
    confidence    = final_conf,
    tf_candles    = {"M15": m15, "H1": h1, "H4": h4},
    current_price = price,
    spread        = row.get("spread", 0),
    session       = _get_session(target_time.hour),
    features      = {
        "rsi": rsi_val, "atr": atr_val, "adx": regime["adx"],
        "pgi": custom_result.get("pgi", 0),
        "regime": regime["dominant"],
    },
    agent_votes   = votes,
    data_quality  = dq,
    custom_signals = custom_result,
    arbiter_score = final_score,
    machine_state = ms["state"],
)

# ── После закрытия сделки ──────────────────────────
flight_recorder.update_outcome(decision_id, result, pnl, mfe, mae, tte)

# ── Diversity Monitor ──────────────────────────────
diversity_monitor.record_votes(votes, result, pnl)
if len(all_trades) % 20 == 0:
    div_report = diversity_monitor.analyze()
    if div_report.get("weight_adjustments"):
        for agent, adj in div_report["weight_adjustments"].items():
            arbiter.update_agent_weight(agent, adj)

# ── Drift Watchdog ─────────────────────────────────
drift_watchdog.record_trade(
    features = {"rsi": rsi_val, "adx": regime["adx"]},
    result   = result,
    profit   = pnl,
)
if len(all_trades) % 10 == 0:
    drift = drift_watchdog.check()
    if drift["action"] == "SAFE_MODE":
        machine.set_state("SAFE_MODE", f"Drift: {drift['message'][:80]}")
```

### 3. `council.py` — в `build_market_context()`

```python
from regime_detector import regime_detector
from machine_state   import machine

# Добавить в контекст:
regime = regime_detector.detect(df_h1, "H1")
context += f"\n{regime['summary']}"
context += f"\n{machine.get_status_text()}"
```

### 4. `optimizer.py` — заменить win-rate на expectancy

```python
from diversity_monitor import expectancy_tracker

# Вместо:
# if win_rate > 0.60 and n_trades >= 20:

# Использовать:
metrics = expectancy_tracker.analyze(trades)
if metrics["edge_sufficient"] and metrics["n_trades"] >= 30:
    # Применять паттерн
```

### 5. `pro_auto_trade_pc.py` — боевой режим

```python
from machine_state import machine

# В начале торгового цикла:
ms = machine.tick(current_balance, last_result, data_quality_score)
if not ms["can_trade"]:
    print(f"[{ms['state']}] Торговля запрещена: {ms['reason']}")
    continue

# Корректировать MIN_CONFIDENCE динамически:
effective_min_confidence = ms["min_confidence"]
```

---

## Файлы, которые нужно положить в папку `ii\`

```
data_steward.py      ← Валидация данных
flight_recorder.py   ← Черный ящик
machine_state.py     ← Контроллер состояний
arbiter.py           ← Арбитр (заменяет голосование)
diversity_monitor.py ← Корреляция агентов + Drift + Expectancy
regime_detector.py   ← Режим рынка с вероятностями
```

## Папки которые создадутся автоматически

```
logs/flight_recorder/    ← JSON снапшоты каждого решения
logs/data_quality.jsonl  ← Лог качества данных
logs/machine_state.json  ← Состояние машины
logs/arbiter_decisions.jsonl
logs/diversity_monitor.jsonl
logs/drift_watchdog.jsonl
```

---

## Порядок внедрения (от простого к сложному)

**День 1:** `machine_state.py` — просто добавить в цикл `machine.tick()`.
Ничего не сломает, сразу даёт защиту от серий убытков.

**День 2:** `data_steward.py` — очистить CSV перед обучением.
Запусти `steward.validate_csv(df, "M15")` перед `train_dna()`.

**День 3:** `regime_detector.py` — добавить в контекст промпта.
Агенты получат `regime["summary"]` и будут знать тренд/флэт.

**День 4:** `flight_recorder.py` — включить запись снапшотов.
Аудитор сможет replay любую сделку.

**День 5:** `arbiter.py` — заменить простое голосование.
Самое сложное, требует рефакторинга council.py.

**День 6:** `diversity_monitor.py` — мониторинг после 50+ сделок.
