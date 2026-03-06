import json
import os
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from config import TRADE_HISTORY_FILE, LOG_DIR, SYMBOL

app = Flask(__name__)

# Хранилище данных от слоёв (обновляется из главного цикла)
layer_data = {
    "last_update": "",
    "signal": "WAIT",
    "confidence": 0,
    "price": 0,
    "spread": 0,
    "rsi": 50,
    "atr": 0,
    "session": "",
    "logic": "",
    # Слой 1: Техника
    "tech_buy": 0,
    "tech_sell": 0,
    "tech_details": [],
    # Слой 2: Корреляции
    "corr_buy": 0,
    "corr_sell": 0,
    "corr_verdict": "NEUTRAL",
    "corr_details": [],
    # Слой 3: Настроения
    "sentiment": "No data",
    # Слой 4: ИИ
    "ai_raw_confidence": 0,
    "corr_adjustment": 0,
    "tech_adjustment": 0,
    "final_confidence": 0,
    # История уверенности
    "confidence_history": [],
    # Позиции
    "open_positions": [],
}


def update_layer_data(data):
    """Вызывается из главного цикла для обновления дашборда"""
    global layer_data
    layer_data.update(data)
    layer_data["last_update"] = datetime.now().strftime("%H:%M:%S")
    # Сохраняем историю уверенности (макс 50 точек)
    if data.get("final_confidence", 0) > 0:
        layer_data["confidence_history"].append({
            "time": datetime.now().strftime("%H:%M"),
            "conf": data.get("final_confidence", 0),
            "signal": data.get("signal", "WAIT"),
            "tech_buy": data.get("tech_buy", 0),
            "tech_sell": data.get("tech_sell", 0),
            "corr_buy": data.get("corr_buy", 0),
            "corr_sell": data.get("corr_sell", 0),
        })
        if len(layer_data["confidence_history"]) > 50:
            layer_data["confidence_history"] = layer_data["confidence_history"][-50:]


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>Autopilot 5.0 Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#e0e0e0;font-family:'Segoe UI',monospace;padding:15px}
.header{text-align:center;padding:15px;background:linear-gradient(135deg,#1a1f35,#0d1117);border:1px solid #30363d;border-radius:12px;margin-bottom:15px}
.header h1{color:#58a6ff;font-size:22px}
.status-line{margin-top:8px;font-size:14px}
.signal-badge{display:inline-block;padding:4px 16px;border-radius:20px;font-weight:bold;font-size:18px;margin:0 8px}
.signal-buy{background:#1a4d2e;color:#3fb950;border:2px solid #3fb950}
.signal-sell{background:#4d1a1a;color:#f85149;border:2px solid #f85149}
.signal-wait{background:#4d3d1a;color:#d29922;border:2px solid #d29922}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:15px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}
.card h2{color:#58a6ff;font-size:13px;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.stat{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;font-size:13px}
.stat .label{color:#8b949e}
.positive{color:#3fb950}.negative{color:#f85149}.neutral{color:#d29922}.muted{color:#8b949e}
.meter{height:24px;border-radius:4px;background:#21262d;margin:4px 0;position:relative;overflow:hidden}
.meter-fill{height:100%;border-radius:4px;transition:width 0.5s}
.meter-label{position:absolute;top:3px;left:8px;font-size:12px;font-weight:bold;color:#fff;z-index:1}
.layer-row{display:flex;align-items:center;padding:6px 0;border-bottom:1px solid #21262d}
.layer-name{width:120px;color:#8b949e;font-size:12px}
.layer-bar{flex:1;display:flex;height:20px;gap:2px}
.bar-buy{background:#3fb950;border-radius:3px;transition:width 0.5s}
.bar-sell{background:#f85149;border-radius:3px;transition:width 0.5s}
.bar-neutral{background:#30363d;border-radius:3px;flex:1}
.layer-score{width:80px;text-align:right;font-size:12px;font-weight:bold}
.conf-chart{display:flex;align-items:flex-end;gap:3px;height:120px;padding:10px 0;border-top:1px solid #21262d;margin-top:10px}
.conf-bar{flex:1;min-width:12px;border-radius:3px 3px 0 0;position:relative;transition:height 0.3s}
.conf-bar:hover .conf-tooltip{display:block}
.conf-tooltip{display:none;position:absolute;bottom:105%;left:50%;transform:translateX(-50%);background:#30363d;color:#e0e0e0;padding:4px 8px;border-radius:4px;font-size:10px;white-space:nowrap;z-index:10}
.trade-item{padding:8px;margin:4px 0;border-radius:6px;border-left:3px solid;font-size:12px}
.trade-win{border-color:#3fb950;background:#0d1f0d}
.trade-loss{border-color:#f85149;background:#1f0d0d}
.trade-open{border-color:#d29922;background:#1f1a0d}
.log-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px;max-height:250px;overflow-y:auto;font-size:11px;line-height:1.5;white-space:pre-wrap}
.detail-item{font-size:11px;padding:3px 0;color:#c9d1d9}
.pos-card{background:#1a1f2e;border:1px solid #30363d;border-radius:8px;padding:12px;margin:6px 0}
.adj-tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin:2px}
.adj-pos{background:#1a4d2e;color:#3fb950}
.adj-neg{background:#4d1a1a;color:#f85149}
.adj-zero{background:#21262d;color:#8b949e}
</style>
</head>
<body>

<div class="header">
<h1>AUTOPILOT 5.0 — LAYER MONITOR</h1>
<div class="status-line">
  <span class="signal-badge signal-{{ signal_css }}">{{ d.signal }}</span>
  Final Confidence: <strong>{{ d.final_confidence }}/10</strong> |
  Updated: {{ d.last_update }} |
  {{ d.session }}
</div>
</div>

<!-- ROW 1: Signal + Layers + Confidence -->
<div class="grid">

<!-- CURRENT SIGNAL -->
<div class="card">
<h2>🎯 Current Signal</h2>
<div class="stat"><span class="label">Direction</span><span class="{{ 'positive' if d.signal=='BUY' else 'negative' if d.signal=='SELL' else 'neutral' }}"><strong>{{ d.signal }}</strong></span></div>
<div class="stat"><span class="label">Price</span><span>{{ "%.5f"|format(d.price) }}</span></div>
<div class="stat"><span class="label">Spread</span><span>{{ "%.1f"|format(d.spread) }}p</span></div>
<div class="stat"><span class="label">RSI</span><span class="{{ 'negative' if d.rsi > 70 else 'positive' if d.rsi < 30 else 'muted' }}">{{ "%.1f"|format(d.rsi) }}</span></div>
<div class="stat"><span class="label">ATR</span><span>{{ "%.6f"|format(d.atr) }}</span></div>
<div class="stat"><span class="label">AI Logic</span><span style="font-size:11px;max-width:200px;text-align:right">{{ d.logic[:80] }}</span></div>
<h2 style="margin-top:12px">📐 Confidence Breakdown</h2>
<div class="stat"><span class="label">AI Raw</span><span>{{ d.ai_raw_confidence }}/10</span></div>
<div class="stat"><span class="label">Corr Adjust</span><span><span class="adj-tag {{ 'adj-pos' if d.corr_adjustment > 0 else 'adj-neg' if d.corr_adjustment < 0 else 'adj-zero' }}">{{ '+' if d.corr_adjustment > 0 else '' }}{{ d.corr_adjustment }}</span></span></div>
<div class="stat"><span class="label">Tech Adjust</span><span><span class="adj-tag {{ 'adj-pos' if d.tech_adjustment > 0 else 'adj-neg' if d.tech_adjustment < 0 else 'adj-zero' }}">{{ '+' if d.tech_adjustment > 0 else '' }}{{ d.tech_adjustment }}</span></span></div>
<div class="stat"><span class="label"><strong>FINAL</strong></span><span><strong>{{ d.final_confidence }}/10</strong></span></div>
</div>

<!-- LAYER ANALYSIS -->
<div class="card">
<h2>📊 Layer Analysis (BUY vs SELL scores)</h2>

<!-- Tech Layer -->
<div class="layer-row">
<span class="layer-name">📈 Technical</span>
<div class="layer-bar">
{% set tech_total = d.tech_buy + d.tech_sell if (d.tech_buy + d.tech_sell) > 0 else 1 %}
<div class="bar-buy" style="width:{{ (d.tech_buy/tech_total*100)|int }}%"></div>
<div class="bar-sell" style="width:{{ (d.tech_sell/tech_total*100)|int }}%"></div>
</div>
<span class="layer-score"><span class="positive">{{ d.tech_buy }}</span>/<span class="negative">{{ d.tech_sell }}</span></span>
</div>

<!-- Corr Layer -->
<div class="layer-row">
<span class="layer-name">🔗 Correlations</span>
<div class="layer-bar">
{% set corr_total = d.corr_buy + d.corr_sell if (d.corr_buy + d.corr_sell) > 0 else 1 %}
<div class="bar-buy" style="width:{{ (d.corr_buy/corr_total*100)|int }}%"></div>
<div class="bar-sell" style="width:{{ (d.corr_sell/corr_total*100)|int }}%"></div>
</div>
<span class="layer-score"><span class="positive">{{ d.corr_buy }}</span>/<span class="negative">{{ d.corr_sell }}</span></span>
</div>

<!-- AI Layer -->
<div class="layer-row">
<span class="layer-name">🧠 AI (GPT-4o)</span>
<div class="layer-bar">
{% if d.signal == 'BUY' %}
<div class="bar-buy" style="width:{{ d.ai_raw_confidence * 10 }}%"></div>
<div class="bar-neutral"></div>
{% elif d.signal == 'SELL' %}
<div class="bar-sell" style="width:{{ d.ai_raw_confidence * 10 }}%"></div>
<div class="bar-neutral"></div>
{% else %}
<div class="bar-neutral"></div>
{% endif %}
</div>
<span class="layer-score">{{ d.ai_raw_confidence }}/10</span>
</div>

<!-- Combined meter -->
<div style="margin-top:15px">
<div style="font-size:12px;color:#8b949e;margin-bottom:4px">Combined Confidence</div>
<div class="meter">
<div class="meter-label">{{ d.final_confidence }}/10</div>
<div class="meter-fill" style="width:{{ d.final_confidence * 10 }}%;background:{{ '#3fb950' if d.final_confidence >= 7 else '#d29922' if d.final_confidence >= 5 else '#f85149' }}"></div>
</div>
</div>

<!-- Tech Details -->
<h2 style="margin-top:15px">📈 Technical Indicators</h2>
{% for detail in d.tech_details %}
<div class="detail-item">{{ detail }}</div>
{% endfor %}
{% if not d.tech_details %}<div class="detail-item muted">Waiting for data...</div>{% endif %}

<!-- Corr Details -->
<h2 style="margin-top:10px">🔗 Correlation Details</h2>
{% for detail in d.corr_details %}
<div class="detail-item">{{ detail }}</div>
{% endfor %}
{% if not d.corr_details %}<div class="detail-item muted">Waiting for data...</div>{% endif %}
</div>

<!-- CONFIDENCE CHART -->
<div class="card">
<h2>📉 Confidence Over Time</h2>
{% if history %}
<div class="conf-chart">
{% for h in history %}
{% set color = '#3fb950' if h.signal == 'BUY' else '#f85149' if h.signal == 'SELL' else '#d29922' %}
<div class="conf-bar" style="height:{{ h.conf * 10 }}%;background:{{ color }}">
<div class="conf-tooltip">{{ h.time }} | {{ h.signal }} {{ h.conf }}/10<br>Tech:{{ h.tech_buy }}/{{ h.tech_sell }} Corr:{{ h.corr_buy }}/{{ h.corr_sell }}</div>
</div>
{% endfor %}
</div>
<div style="display:flex;justify-content:space-between;font-size:10px;color:#8b949e;margin-top:4px">
<span>{{ history[0].time if history else '' }}</span>
<span>{{ history[-1].time if history else '' }}</span>
</div>
<div style="margin-top:8px;font-size:11px">
<span class="positive">■</span> BUY
<span class="negative" style="margin-left:10px">■</span> SELL
<span class="neutral" style="margin-left:10px">■</span> WAIT
<span class="muted" style="margin-left:10px">— Min confidence line (7)</span>
</div>
{% else %}
<p class="muted">Chart will appear after first analysis cycle</p>
{% endif %}

<!-- POSITIONS -->
<h2 style="margin-top:15px">📌 Open Positions</h2>
{% for pos in positions %}
<div class="pos-card">
<strong>{{ pos.dir }} {{ pos.vol }} lot</strong> @ {{ pos.entry }}<br>
SL: {{ pos.sl }} | TP: {{ pos.tp }}<br>
<span class="{{ 'positive' if pos.pnl >= 0 else 'negative' }}">P/L: ${{ "%.2f"|format(pos.pnl) }}</span>
</div>
{% endfor %}
{% if not positions %}<p class="muted">No open positions</p>{% endif %}

<!-- NEWS SENTIMENT -->
<h2 style="margin-top:15px">📰 News Sentiment</h2>
<div style="font-size:11px;color:#c9d1d9;white-space:pre-wrap">{{ d.sentiment }}</div>
</div>
</div>

<!-- ROW 2: Stats + Trades + Log -->
<div class="grid">
<div class="card">
<h2>📊 Performance</h2>
<div class="stat"><span class="label">Total Trades</span><span>{{ total }}</span></div>
<div class="stat"><span class="label">Winrate</span><span class="{{ 'positive' if wr >= 50 else 'negative' }}">{{ "%.1f"|format(wr) }}%</span></div>
<div class="stat"><span class="label">Wins / Losses</span><span><span class="positive">{{ w }}</span> / <span class="negative">{{ l }}</span></span></div>
<div class="stat"><span class="label">Total P/L</span><span class="{{ 'positive' if pnl >= 0 else 'negative' }}">${{ "%.2f"|format(pnl) }}</span></div>
<div class="stat"><span class="label">Avg Win</span><span class="positive">${{ "%.2f"|format(avg_win) }}</span></div>
<div class="stat"><span class="label">Avg Loss</span><span class="negative">${{ "%.2f"|format(avg_loss) }}</span></div>
<div class="stat"><span class="label">Profit Factor</span><span>{{ "%.2f"|format(pf) }}</span></div>
<div class="stat"><span class="label">Today Trades</span><span>{{ today_trades }}/5</span></div>
<div class="stat"><span class="label">Today P/L</span><span class="{{ 'positive' if today_pnl >= 0 else 'negative' }}">${{ "%.2f"|format(today_pnl) }}</span></div>
</div>

<div class="card">
<h2>📜 Recent Trades</h2>
{% for t in trades %}
<div class="trade-item {{ 'trade-win' if t.r == 'WIN' else 'trade-loss' }}">
{{ t.time }} | {{ t.dir }} {{ t.lot }}lot | <span class="{{ 'positive' if t.pnl >= 0 else 'negative' }}">${{ "%.2f"|format(t.pnl) }}</span> | Conf:{{ t.conf }}
</div>
{% endfor %}
{% if not trades %}<p class="muted">No completed trades yet</p>{% endif %}
</div>

<div class="card">
<h2>📝 Live Log</h2>
<div class="log-box">{{ log }}</div>
</div>
</div>

</body>
</html>"""


@app.route('/')
def dashboard():
    import MetaTrader5 as mt5
    history = []
    try:
        with open(TRADE_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        pass
    closed = [t for t in history if t.get("result")]
    w = len([t for t in closed if t["result"] == "WIN"])
    l = len([t for t in closed if t["result"] == "LOSS"])
    total = len(closed)
    wr = (w / total * 100) if total > 0 else 0
    pnl = sum(t.get("profit", 0) for t in closed if t.get("profit"))
    wins_list = [t for t in closed if t["result"] == "WIN"]
    loss_list = [t for t in closed if t["result"] == "LOSS"]
    avg_win = sum(t.get("profit", 0) for t in wins_list) / len(wins_list) if wins_list else 0
    avg_loss = sum(t.get("profit", 0) for t in loss_list) / len(loss_list) if loss_list else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    today = datetime.now().strftime("%Y-%m-%d")
    today_closed = [t for t in closed if t.get("close_time", "").startswith(today)]
    today_pnl = sum(t.get("profit", 0) for t in today_closed if t.get("profit"))
    trades = []
    for t in reversed(closed[-15:]):
        trades.append({
            "time": t.get("time", "")[:16],
            "dir": t.get("direction", "?"),
            "lot": t.get("lot", 0),
            "pnl": t.get("profit", 0),
            "conf": t.get("confidence", 0),
            "r": t.get("result", "?")
        })
    # Positions
    positions = []
    try:
        if mt5.initialize():
            pos_list = mt5.positions_get(symbol=SYMBOL)
            if pos_list:
                for p in pos_list:
                    positions.append({
                        "dir": "BUY" if p.type == 0 else "SELL",
                        "vol": p.volume,
                        "entry": f"{p.price_open:.5f}",
                        "sl": f"{p.sl:.5f}",
                        "tp": f"{p.tp:.5f}",
                        "pnl": p.profit
                    })
    except Exception:
        pass
    # Log
    log_content = "Waiting..."
    log_path = os.path.join(LOG_DIR, today + "_logic.txt")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            log_content = "".join(lines[-60:])
    except Exception:
        pass
    # Signal CSS
    sig = layer_data.get("signal", "WAIT")
    signal_css = "buy" if sig == "BUY" else "sell" if sig == "SELL" else "wait"
    # Tech details as list
    tech_details = []
    tech_summary = layer_data.get("tech_summary", "")
    if tech_summary:
        for line in tech_summary.split("\n"):
            line = line.strip()
            if line and not line.startswith("--") and not line.startswith("Verdict"):
                tech_details.append(line)
    layer_data["tech_details"] = tech_details
    # Corr details
    corr_details = []
    corr_raw = layer_data.get("corr_summary", "")
    if corr_raw:
        for line in corr_raw.split("\n"):
            line = line.strip()
            if line and not line.startswith("--") and not line.startswith("BUY:"):
                corr_details.append(line)
    layer_data["corr_details"] = corr_details
    return render_template_string(
        HTML,
        d=layer_data,
        signal_css=signal_css,
        history=layer_data.get("confidence_history", []),
        positions=positions,
        total=total, wr=wr, w=w, l=l, pnl=pnl,
        avg_win=avg_win, avg_loss=avg_loss, pf=pf,
        today_trades=len(today_closed), today_pnl=today_pnl,
        trades=trades, log=log_content
    )


@app.route('/api/layers')
def api_layers():
    return jsonify(layer_data)


def start_dashboard():
    app.run(host='0.0.0.0', port=8080, debug=False)


def start_dashboard_thread():
    t = threading.Thread(target=start_dashboard, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    print("Dashboard: http://localhost:8080")
    start_dashboard()