"""
dashboard_generator.py — Генератор HTML-дашборда v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
НОВОЕ v3.0:
  - TradingView Lightweight Charts (японские свечи)
  - Маркеры сделок на графике (стрелки BUY/SELL, крестики закрытия)
  - Блок аудитора (MFE/MAE/класс сделки)
  - Тепловая карта по часам (WR по времени суток)
  - Equity curve через recharts-подобный canvas
  - Споры Консилиума под графиком
"""

import json
import os
from datetime import datetime
from typing import Optional

DASHBOARD_FILE = "dashboard.html"


def generate_dashboard(
    candles: list,           # список dict: {time, open, high, low, close, tick_volume}
    trades: list,            # список dict из trade_memory / backtester
    council_log: list,       # список dict: {time, signal, confidence, votes, reasoning}
    audit_log: list,         # список dict из auditor_agent
    stats: dict,             # {'balance', 'wins', 'losses', 'winrate', 'profit'}
    time_profile: str = "",  # текст из trade_memory.get_time_profile()
    alpha_context: str = "", # текст из auditor.get_alpha_context()
    custom_verdict: str = "", # текст из custom_indicators
    title: str = "Autopilot Dashboard",
) -> str:
    """
    Генерирует HTML-дашборд и сохраняет в DASHBOARD_FILE.
    Возвращает путь к файлу.
    """

    # ── Подготовка данных для JS ─────────────────────
    candles_js    = _prepare_candles(candles)
    markers_js    = _prepare_markers(trades)
    equity_js     = _prepare_equity(trades)
    hourly_js     = _prepare_hourly(trades)
    audit_cards   = _prepare_audit_cards(audit_log)
    council_html  = _prepare_council_html(council_log)
    trade_log_html = _prepare_trade_log(trades)

    bal_color = "#00ff88" if stats.get("balance", 0) >= 0 else "#ff4444"
    winrate   = stats.get("winrate", 0)
    wr_color  = "#00ff88" if winrate >= 55 else "#ffaa00" if winrate >= 45 else "#ff4444"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="5">
  <title>{title}</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {{
      --bg:      #0d0d0d;
      --panel:   #141414;
      --border:  #222;
      --green:   #00ff88;
      --red:     #ff4455;
      --blue:    #4488ff;
      --yellow:  #ffcc44;
      --gray:    #666;
      --text:    #e0e0e0;
      --font:    'Courier New', monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 13px;
      padding: 12px;
    }}

    /* ── Header ── */
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 12px;
    }}
    .header h1 {{ color: #fff; font-size: 16px; letter-spacing: 2px; }}
    .ts {{ color: var(--gray); font-size: 11px; }}

    /* ── Stat boxes ── */
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .stat-box {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 8px;
      text-align: center;
    }}
    .stat-box .label {{ color: var(--gray); font-size: 10px; margin-bottom: 4px; }}
    .stat-box .value {{ font-size: 18px; font-weight: bold; }}

    /* ── Chart ── */
    .chart-wrapper {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 12px;
    }}
    .chart-title {{
      color: var(--gray);
      font-size: 11px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    #tradingview-chart {{ width: 100%; height: 420px; }}

    /* ── Two columns ── */
    .cols-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .cols-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}

    /* ── Panel ── */
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
    }}
    .panel-title {{
      color: var(--yellow);
      font-size: 11px;
      letter-spacing: 1px;
      text-transform: uppercase;
      margin-bottom: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
    }}

    /* ── Equity curve ── */
    #equity-canvas {{ width: 100%; height: 160px; }}

    /* ── Audit cards ── */
    .audit-card {{
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 8px;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .audit-card.PERFECT_WIN   {{ border-color: #00ff88; }}
    .audit-card.WASTED_WIN    {{ border-color: #ffaa00; }}
    .audit-card.NOISE_STOP    {{ border-color: #ff8844; }}
    .audit-card.WRONG_DIR     {{ border-color: #ff4455; }}
    .audit-card.NORMAL_LOSS   {{ border-color: #444; }}
    .tag {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      margin-right: 4px;
      font-weight: bold;
    }}
    .tag-win    {{ background: #003322; color: #00ff88; }}
    .tag-loss   {{ background: #220011; color: #ff4455; }}
    .tag-class  {{ background: #222200; color: #ffaa00; }}
    .metric {{
      display: inline-block;
      margin-right: 10px;
      color: var(--gray);
    }}
    .metric span {{ color: var(--text); }}

    /* ── Council log ── */
    .council-entry {{
      border-bottom: 1px solid var(--border);
      padding: 8px 0;
      font-size: 11px;
    }}
    .council-signal {{ font-size: 13px; font-weight: bold; margin-right: 8px; }}
    .sig-buy    {{ color: var(--green); }}
    .sig-sell   {{ color: var(--red); }}
    .sig-wait   {{ color: var(--gray); }}
    .agent-line {{ color: #888; margin-top: 3px; }}
    .agent-name {{ font-weight: bold; }}
    .agent-impulse  {{ color: #ffaa00; }}
    .agent-trend    {{ color: #4488ff; }}
    .agent-analyst  {{ color: #cc88ff; }}

    /* ── Hourly heatmap ── */
    .heatmap-grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 3px;
      margin-top: 8px;
    }}
    .hmap-cell {{
      border-radius: 3px;
      padding: 4px 2px;
      text-align: center;
      font-size: 10px;
      cursor: default;
    }}

    /* ── Trade log ── */
    .trade-row {{
      display: grid;
      grid-template-columns: 90px 50px 60px 70px 70px 80px 1fr;
      gap: 6px;
      padding: 4px 0;
      border-bottom: 1px solid #1a1a1a;
      font-size: 11px;
    }}
    .trade-row:hover {{ background: #1a1a1a; }}

    /* ── Alpha patterns ── */
    .alpha-block {{
      background: #0a1a0a;
      border: 1px solid #1a3a1a;
      border-radius: 4px;
      padding: 10px;
      font-size: 11px;
      color: #88cc88;
      white-space: pre-wrap;
      max-height: 180px;
      overflow-y: auto;
    }}
    .custom-block {{
      background: #0a0a1a;
      border: 1px solid #1a1a3a;
      border-radius: 4px;
      padding: 10px;
      font-size: 11px;
      color: #8888cc;
      white-space: pre-wrap;
      max-height: 180px;
      overflow-y: auto;
    }}
  </style>
</head>
<body>

  <!-- HEADER -->
  <div class="header">
    <h1>⚡ {title.upper()}</h1>
    <span class="ts">Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 5s</span>
  </div>

  <!-- STATS -->
  <div class="stats-row">
    <div class="stat-box">
      <div class="label">БАЛАНС</div>
      <div class="value" style="color:{bal_color}">${stats.get('balance', 0):.1f}</div>
    </div>
    <div class="stat-box">
      <div class="label">СДЕЛОК</div>
      <div class="value">{stats.get('wins', 0) + stats.get('losses', 0)}</div>
    </div>
    <div class="stat-box">
      <div class="label">WIN / LOSS</div>
      <div class="value">{stats.get('wins', 0)} / {stats.get('losses', 0)}</div>
    </div>
    <div class="stat-box">
      <div class="label">WINRATE</div>
      <div class="value" style="color:{wr_color}">{winrate:.1f}%</div>
    </div>
    <div class="stat-box">
      <div class="label">P&L</div>
      <div class="value" style="color:{bal_color}">${stats.get('profit', 0):.2f}</div>
    </div>
    <div class="stat-box">
      <div class="label">P. FACTOR</div>
      <div class="value">{stats.get('profit_factor', 0):.2f}</div>
    </div>
  </div>

  <!-- MAIN CHART -->
  <div class="chart-wrapper">
    <div class="chart-title">📈 EURUSD — Японские свечи (M15) + Сделки</div>
    <div id="tradingview-chart"></div>
  </div>

  <!-- EQUITY + HOURLY HEATMAP -->
  <div class="cols-2">
    <div class="panel">
      <div class="panel-title">📉 Equity Curve</div>
      <canvas id="equity-canvas"></canvas>
    </div>
    <div class="panel">
      <div class="panel-title">🕐 Тепловая карта по часам (WR%)</div>
      <div class="heatmap-grid" id="heatmap-grid"></div>
      <div style="margin-top:8px; font-size:10px; color:#666;">
        Зелёный = WR≥60% | Красный = WR≤40% | Серый = мало данных
      </div>
    </div>
  </div>

  <!-- AUDIT + COUNCIL -->
  <div class="cols-2">

    <!-- AUDIT BLOCK -->
    <div class="panel">
      <div class="panel-title">🔍 ИИ-Аудитор (последние сделки)</div>
      <div id="audit-cards">
        {audit_cards}
      </div>
    </div>

    <!-- COUNCIL LOG -->
    <div class="panel">
      <div class="panel-title">🧠 Консилиум (последние решения)</div>
      <div id="council-log" style="max-height:320px; overflow-y:auto;">
        {council_html}
      </div>
    </div>

  </div>

  <!-- ALPHA PATTERNS + CUSTOM SIGNALS -->
  <div class="cols-2" style="margin-bottom:12px;">
    <div class="panel">
      <div class="panel-title">🧬 Alpha Patterns (подтверждённые)</div>
      <div class="alpha-block">{alpha_context or 'Паттерны накапливаются...'}</div>
    </div>
    <div class="panel">
      <div class="panel-title">⚗️ Custom Indicators (PGI / TVA / CandleDNA)</div>
      <div class="custom-block">{custom_verdict or 'Ожидание данных...'}</div>
    </div>
  </div>

  <!-- TIME PROFILE -->
  <div class="panel" style="margin-bottom:12px;">
    <div class="panel-title">⏱️ Временной профиль (прибыль по времени)</div>
    <pre style="color:#aaa; font-size:11px; white-space:pre-wrap;">{time_profile or 'Накапливается история...'}</pre>
  </div>

  <!-- TRADE LOG -->
  <div class="panel">
    <div class="panel-title">📋 Журнал сделок</div>
    <div class="trade-row" style="color:#666; font-size:10px; border-bottom:1px solid #333;">
      <span>ВРЕМЯ</span><span>ТИП</span><span>РЕЗУЛЬТАТ</span>
      <span>P/L</span><span>MFE</span><span>MAE</span><span>КЛАСС / ПРАВИЛО</span>
    </div>
    {trade_log_html}
  </div>

  <!-- SCRIPTS -->
  <script>
  // ══════════════════════════════════════════
  //  TRADINGVIEW LIGHTWEIGHT CHARTS
  // ══════════════════════════════════════════
  (function() {{
    const el = document.getElementById('tradingview-chart');
    const chart = LightweightCharts.createChart(el, {{
      layout: {{
        background: {{ color: '#0d0d0d' }},
        textColor:  '#888',
      }},
      grid: {{
        vertLines: {{ color: '#1a1a1a' }},
        horzLines: {{ color: '#1a1a1a' }},
      }},
      crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      rightPriceScale: {{ borderColor: '#333' }},
      timeScale: {{
        borderColor: '#333',
        timeVisible: true,
        secondsVisible: false,
      }},
    }});

    const candleSeries = chart.addCandlestickSeries({{
      upColor:   '#00ff88',
      downColor: '#ff4455',
      borderUpColor:   '#00ff88',
      borderDownColor: '#ff4455',
      wickUpColor:   '#00aa55',
      wickDownColor: '#cc2233',
    }});

    const candles = {candles_js};
    if (candles.length > 0) {{
      candleSeries.setData(candles);
    }}

    // Маркеры сделок
    const markers = {markers_js};
    if (markers.length > 0) {{
      candleSeries.setMarkers(markers);
    }}

    // Авторесайз
    new ResizeObserver(() => chart.applyOptions({{ width: el.clientWidth }}))
      .observe(el);
  }})();

  // ══════════════════════════════════════════
  //  EQUITY CURVE (Canvas)
  // ══════════════════════════════════════════
  (function() {{
    const canvas = document.getElementById('equity-canvas');
    const ctx    = canvas.getContext('2d');
    const equity = {equity_js};
    if (!equity || equity.length < 2) {{
      ctx.fillStyle = '#444';
      ctx.font = '12px monospace';
      ctx.fillText('Нет данных', 10, 80);
      return;
    }}
    canvas.width  = canvas.offsetWidth || 400;
    canvas.height = 160;
    const W = canvas.width, H = canvas.height;
    const pad = 30;
    const min_v = Math.min(...equity);
    const max_v = Math.max(...equity);
    const range = max_v - min_v || 1;

    function toX(i) {{ return pad + (i / (equity.length - 1)) * (W - pad * 2); }}
    function toY(v) {{ return H - pad - ((v - min_v) / range) * (H - pad * 2); }}

    // Нулевая линия
    const zeroY = toY(0);
    ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(W - pad, zeroY); ctx.stroke();

    // Кривая
    ctx.lineWidth = 2;
    ctx.beginPath();
    equity.forEach((v, i) => {{
      ctx.strokeStyle = v >= 0 ? '#00ff88' : '#ff4455';
      if (i === 0) ctx.moveTo(toX(i), toY(v));
      else ctx.lineTo(toX(i), toY(v));
    }});
    ctx.stroke();

    // Последнее значение
    const last = equity[equity.length - 1];
    ctx.fillStyle = last >= 0 ? '#00ff88' : '#ff4455';
    ctx.font = 'bold 13px monospace';
    ctx.fillText('$' + last.toFixed(2), W - 80, 20);
  }})();

  // ══════════════════════════════════════════
  //  HEATMAP ПО ЧАСАМ
  // ══════════════════════════════════════════
  (function() {{
    const grid    = document.getElementById('heatmap-grid');
    const hourly  = {hourly_js};

    for (let h = 0; h < 24; h++) {{
      const cell  = document.createElement('div');
      cell.classList.add('hmap-cell');
      const d = hourly[h];
      if (!d || d.total < 2) {{
        cell.style.background = '#1a1a1a';
        cell.style.color = '#444';
        cell.innerHTML = `<div>${{h}}</div><div>—</div>`;
      }} else {{
        const wr = d.wr;
        const r  = Math.round(255 * (1 - wr / 100));
        const g  = Math.round(200 * (wr / 100));
        cell.style.background = `rgba(${{r}},${{g}},30,0.5)`;
        cell.style.color = '#fff';
        cell.innerHTML = `<div>${{h}}h</div><div>${{Math.round(wr)}}%</div>`;
        cell.title = `${{h}}:00 | WR ${{wr.toFixed(1)}}% | ${{d.total}} сделок`;
      }}
      grid.appendChild(cell);
    }}
  }})();
  </script>
</body>
</html>"""

    try:
        with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        print(f"[Dashboard] Save error: {e}")

    return DASHBOARD_FILE


# ─────────────────────────────────────────
#  HELPERS — подготовка данных для JS
# ─────────────────────────────────────────

def _prepare_candles(candles: list) -> str:
    """Конвертирует список свечей в JS-массив для LightweightCharts.
    Обрабатывает pd.Timestamp, numpy.int64, строки ISO, Unix int.
    """
    import pandas as pd
    import numpy as np

    if not candles:
        return "[]"

    def to_unix(t):
        """Конвертирует любой временной тип в Unix timestamp (int)."""
        if isinstance(t, (int, float)):
            # Если это секунды (разумный диапазон 2000-2100)
            v = int(t)
            if v > 1_000_000_000_000:   # миллисекунды → секунды
                v = v // 1000
            return v
        if isinstance(t, pd.Timestamp):
            return int(t.timestamp())
        if isinstance(t, str):
            try:
                from datetime import datetime as _dt
                return int(_dt.fromisoformat(t.replace("Z", "+00:00")).timestamp())
            except Exception:
                return 0
        try:
            return int(t.timestamp())
        except Exception:
            return int(t)

    result = []
    prev_ts = 0
    for c in candles[-200:]:
        try:
            ts = to_unix(c.get("time", 0))
            if ts == 0 or ts <= prev_ts:
                continue  # пропускаем дубли и нулевые
            o = float(c.get("open",  0))
            h = float(c.get("high",  0))
            l = float(c.get("low",   0))
            cl = float(c.get("close", 0))
            if o == 0 or h == 0:
                continue
            result.append({"time": ts, "open": o, "high": h, "low": l, "close": cl})
            prev_ts = ts
        except Exception:
            continue
    return json.dumps(result)


def _prepare_markers(trades: list) -> str:
    """Создаёт маркеры сделок для LightweightCharts."""
    if not trades:
        return "[]"
    markers = []
    for t in trades:
        try:
            time_str = t.get("time") or t.get("entry_time", "")
            if not time_str:
                continue
            from datetime import datetime as _dt
            ts = int(_dt.fromisoformat(str(time_str)).timestamp())
            direction = t.get("direction", "BUY")
            result    = t.get("result", "")

            # Маркер входа
            markers.append({
                "time":     ts,
                "position": "belowBar" if direction == "BUY" else "aboveBar",
                "color":    "#00ff88" if direction == "BUY" else "#ff4455",
                "shape":    "arrowUp" if direction == "BUY" else "arrowDown",
                "text":     f"{direction} {t.get('confidence', '')}",
            })

            # Маркер закрытия
            close_str = t.get("close_time") or t.get("exit_time", "")
            if close_str:
                close_ts = int(_dt.fromisoformat(str(close_str)).timestamp())
                markers.append({
                    "time":     close_ts,
                    "position": "aboveBar",
                    "color":    "#00ff88" if result == "WIN" else "#ff4455",
                    "shape":    "circle",
                    "text":     f"✓ {result}" if result else "✗",
                })
        except Exception:
            continue
    # Сортируем по времени (обязательно для LightweightCharts)
    markers.sort(key=lambda m: m["time"])
    return json.dumps(markers)


def _prepare_equity(trades: list) -> str:
    """Строит кривую equity из списка сделок."""
    closed = [t for t in trades if t.get("result") is not None]
    if not closed:
        return "[0]"
    equity = 0.0
    curve  = [0.0]
    for t in closed:
        equity += t.get("profit", 0) or 0
        curve.append(round(equity, 2))
    return json.dumps(curve)


def _prepare_hourly(trades: list) -> str:
    """Считает winrate по часам суток для тепловой карты."""
    hourly = {h: {"wins": 0, "total": 0} for h in range(24)}
    for t in trades:
        if t.get("result") is None:
            continue
        h = t.get("entry_hour")
        if h is None:
            try:
                from datetime import datetime as _dt
                h = _dt.fromisoformat(str(t.get("time", ""))).hour
            except Exception:
                continue
        hourly[h]["total"] += 1
        if t.get("result") == "WIN":
            hourly[h]["wins"] += 1
    result = {}
    for h, d in hourly.items():
        result[h] = {
            "total": d["total"],
            "wr":    (d["wins"] / d["total"] * 100) if d["total"] > 0 else 50,
        }
    return json.dumps(result)


def _prepare_audit_cards(audit_log: list) -> str:
    """Генерирует HTML карточки аудита."""
    if not audit_log:
        return "<div style='color:#555'>Аудит запустится после первых сделок...</div>"

    cards = []
    for a in reversed(audit_log[-8:]):
        cls    = a.get("trade_class", "NORMAL_LOSS")
        result = a.get("result", "?")
        profit = a.get("profit", 0) or 0
        mfe    = a.get("mfe_points", "?")
        mae    = a.get("mae_points", "?")
        eff    = a.get("efficiency", 0)
        ts     = a.get("time", "")[:16]
        verdict_raw = a.get("ai_verdict", "")

        # Извлечь ПРАВИЛО из вердикта
        rule = ""
        for line in verdict_raw.split("\n"):
            if "ПРАВИЛО:" in line.upper():
                rule = line.split(":", 1)[-1].strip()[:100]
                break

        tag_class = "tag-win" if result == "WIN" else "tag-loss"
        card = f"""
        <div class="audit-card {cls}">
          <span class="tag {tag_class}">{result}</span>
          <span class="tag tag-class">{cls}</span>
          <span style="color:#666; font-size:10px;">{ts}</span>
          <br>
          <span class="metric">P/L: <span style="color:{'#00ff88' if profit >= 0 else '#ff4455'}">${profit:.1f}</span></span>
          <span class="metric">MFE: <span>{mfe}p</span></span>
          <span class="metric">MAE: <span>{mae}p</span></span>
          <span class="metric">Эфф: <span>{eff}%</span></span>
          {"<br><span style='color:#aaa; font-size:11px;'>📌 " + rule + "</span>" if rule else ""}
        </div>"""
        cards.append(card)
    return "\n".join(cards)


def _prepare_council_html(council_log: list) -> str:
    """Генерирует HTML для лога Консилиума."""
    if not council_log:
        return "<div style='color:#555'>Ожидание решений Консилиума...</div>"

    entries = []
    for c in reversed(council_log[-10:]):
        signal = c.get("signal", "WAIT")
        conf   = c.get("confidence", 0)
        ts     = c.get("time", "")[:16]
        sig_class = "sig-buy" if signal == "BUY" else "sig-sell" if signal == "SELL" else "sig-wait"

        votes_html = ""
        for v in c.get("votes", []):
            name    = v.get("name", "?")
            vsignal = v.get("signal", "WAIT")
            vconf   = v.get("confidence", 0)
            reason  = v.get("reasoning", "")[:80]
            agent_cls = (
                "agent-impulse" if "IMPULSE" in name else
                "agent-trend"   if "TREND"   in name else
                "agent-analyst" if "ANALYST" in name else ""
            )
            vsig_cls = "sig-buy" if vsignal == "BUY" else "sig-sell" if vsignal == "SELL" else "sig-wait"
            votes_html += (
                f'<div class="agent-line">'
                f'  <span class="agent-name {agent_cls}">{name}</span>: '
                f'  <span class="{vsig_cls}">{vsignal}</span> ({vconf}/10) — {reason}'
                f'</div>'
            )

        blocked = c.get("blocked_by", "")
        block_html = (
            f'<div style="color:#ff8844; font-size:10px;">🚫 Заблокировано: {blocked}</div>'
            if blocked else ""
        )

        entry = f"""
        <div class="council-entry">
          <span class="council-signal {sig_class}">{signal}</span>
          <span style="color:#888">{conf}/10</span>
          <span style="color:#555; font-size:10px;"> — {ts}</span>
          {block_html}
          {votes_html}
        </div>"""
        entries.append(entry)

    return "\n".join(entries)


def _prepare_trade_log(trades: list) -> str:
    """Генерирует HTML строк журнала сделок."""
    if not trades:
        return "<div style='color:#555'>Нет сделок</div>"

    rows = []
    for t in reversed(trades[-30:]):
        result    = t.get("result")
        direction = t.get("direction", "?")
        profit    = t.get("profit", 0) or 0
        mfe       = t.get("mfe_points", "-")
        mae       = t.get("mae_points", "-")
        cls       = t.get("trade_class", "-")
        rule      = (t.get("audit_rule") or "")[:60]
        ts        = (t.get("time") or t.get("entry_time") or "")[:16]

        dir_color    = "#00ff88" if direction == "BUY" else "#ff4455"
        result_color = "#00ff88" if result == "WIN" else "#ff4455" if result == "LOSS" else "#888"
        profit_color = "#00ff88" if profit >= 0 else "#ff4455"
        result_text  = result or "OPEN"

        rows.append(f"""
        <div class="trade-row">
          <span style="color:#666">{ts}</span>
          <span style="color:{dir_color}">{direction}</span>
          <span style="color:{result_color}">{result_text}</span>
          <span style="color:{profit_color}">${profit:.2f}</span>
          <span style="color:#4488ff">{mfe}p</span>
          <span style="color:#ff8844">{mae}p</span>
          <span style="color:#888">{cls} {rule}</span>
        </div>""")

    return "\n".join(rows)
