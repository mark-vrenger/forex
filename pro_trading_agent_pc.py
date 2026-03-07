import os
import pandas as pd
import numpy as np
import re
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from config import GPT_TIMEOUT, GPT_MAX_RETRIES

load_dotenv()

class TradingAgent:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-v3.2",
            temperature=0.3, 
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            request_timeout=GPT_TIMEOUT,
            max_retries=GPT_MAX_RETRIES
        )
        self.knowledge_path = os.path.join("knowledge", "strategy_handbook.txt")

    def _calculate_channels(self, df, period=30):
        if df is None or len(df) < period: return "N/A"
        y = df['close'].tail(period).values
        x = np.arange(period)
        m, b = np.polyfit(x, y, 1)
        std_dev = np.std(y - (m * x + b))
        if std_dev == 0: return {"slope": "FLAT", "pos": 50}
        pos = ((y[-1] - (m * (period-1) + b - 2*std_dev)) / (4*std_dev)) * 100
        return {"slope": "UP" if m > 0 else "DOWN", "pos": round(pos, 1)}

    def _summarize(self, raw_data, label):
        if raw_data is None or len(raw_data) == 0: return f"[{label}] Нет данных"
        df = pd.DataFrame(raw_data)
        if len(df.columns) >= 5 and 'close' not in df.columns:
            df.columns = ['time','open','high','low','close','tick_volume','spread','real_volume'][:len(df.columns)]
        
        chan = self._calculate_channels(df)
        avg_v = df['tick_volume'].tail(20).mean()
        v_ratio = df['tick_volume'].iloc[-1] / avg_v if avg_v > 0 else 1
        return f"[{label}] Цена:{df['close'].iloc[-1]:.5f} | Тренд:{chan['slope']}({chan['pos']}%) | VolRatio:{v_ratio:.2f}"

    def analyze_market(self, d_m1, d_m5, d_m15, d_m30, d_h1, d_h4, price, book_info, spread):
        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                knowledge = f.read()
        except: knowledge = "База знаний недоступна."

        prompt = f"""You are an ELITE AI TRADING COUNCIL analyzing 6 timeframes.

=== MARKET DATA ===
Price: {price:.5f} | Spread: {spread:.1f}p | {book_info}
Scalping TF: {self._summarize(d_m1, "M1")} | {self._summarize(d_m5, "M5")} | {self._summarize(d_m15, "M15")}
Swing TF:    {self._summarize(d_m30, "M30")} | {self._summarize(d_h1, "H1")} | {self._summarize(d_h4, "H4")}

=== HANDBOOK RULES ===
{knowledge}

=== COUNCIL ROLES ===
1. Grok (Micro-Scalper): Look for extreme channel bounces on M1/M5. Quick entries.
2. Claude 3.5 (Swing Trader): Ignore M1/M5 noise. Look for trend continuation setups on M30/H1/H4 using Brooks Price Action.
3. GPT-4o (Risk Manager): Weigh both Scalping and Swing opportunities. Choose the one with the highest probability right now.

=== CONSENSUS DECISION ===
- Majority rules (2 out of 3 votes). 
- Specify if the setup is "SCALPING" or "SWING".
- Take calculated risks if confidence is 6/10 or higher.

ЛОГИКА: (Детальный анализ на русском: кто победил в обсуждении и какой стиль торговли выбран?)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT/CLOSE/HOLD)
"""
        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"ЛОГИКА: Ошибка ИИ ({e})\nУВЕРЕННОСТЬ: 0\nСИГНАЛ: WAIT"

    def parse_response(self, analysis):
        conf, direction, logic = 0, "WAIT", "Анализ не расшифрован"
        clean_text = analysis.replace('**', '').replace('###', '')
        for line in clean_text.split('\n'):
            up = line.upper().strip()
            if "ЛОГИКА:" in up: logic = line.split(":", 1)[1].strip() if ":" in line else line
            if "УВЕРЕННОСТЬ:" in up:
                nums = re.findall(r'\d+', line)
                if nums: conf = min(int(nums[0]), 10)
            if "СИГНАЛ:" in up:
                for s in ["BUY", "SELL", "WAIT", "CLOSE", "HOLD"]:
                    if s in up: direction = s; break
        return conf, direction, logic

agent = TradingAgent()