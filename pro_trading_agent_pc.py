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
            temperature=0.2, 
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            request_timeout=GPT_TIMEOUT,
            max_retries=GPT_MAX_RETRIES
        )
        self.knowledge_path = os.path.join("knowledge", "strategy_handbook.txt")

    def _load_knowledge(self):
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    return f.read()
            except: pass
        return "Используй Price Action и уровни."

    def _calculate_channels(self, df, period=30):
        if df is None or len(df) < period: return "N/A"
        y = df['close'].tail(period).values
        x = np.arange(period)
        m, b = np.polyfit(x, y, 1)
        std_dev = np.std(y - (m * x + b))
        pos = ((y[-1] - (m * (period-1) + b - 2*std_dev)) / (4*std_dev)) * 100
        return {"slope": "UP" if m > 0 else "DOWN", "pos": round(pos, 1)}

    def _summarize(self, raw_data, label):
        if raw_data is None or len(raw_data) == 0: return f"{label}: нет данных"
        df = pd.DataFrame(raw_data)
        # Фикс KeyError 'close': приводим имена колонок к стандарту
        if 'close' not in df.columns and len(df.columns) >= 5:
            df.columns = ['time','open','high','low','close','tick_volume','spread','real_volume'][:len(df.columns)]
        
        chan = self._calculate_channels(df)
        avg_v = df['tick_volume'].tail(20).mean()
        v_ratio = df['tick_volume'].iloc[-1] / avg_v if avg_v > 0 else 1
        return f"[{label}] Цена:{df['close'].iloc[-1]:.5f} | Тренд:{chan['slope']}({chan['pos']}%) | VolRatio:{v_ratio:.2f}"

    def analyze_market(self, d_m5, d_m15, d_h1, d_h4, price, book_info, spread):
        knowledge = self._load_knowledge()
        prompt = f"""You are a COUNCIL of ELITE AI TRADERS.

=== MARKET DATA ===
Price: {price:.5f} | Spread: {spread:.1f}p | {book_info}
MTF: {self._summarize(d_h4, "H4")} | {self._summarize(d_h1, "H1")} | {self._summarize(d_m15, "M15")} | {self._summarize(d_m5, "M5")}

=== KNOWLEDGE BASE ===
{knowledge}

=== COUNCIL VOICES ===
1. Grok (Aggressive): Scan for M5/M15 scalping momentum and volume spikes.
2. GPT-4o (Conservative): Focus on H1/H4 trend alignment and channel safety.
3. Claude 3.5 (Analyst): Look for Nison candle patterns and Brooks Price Action.

=== CONSENSUS DECISION ===
Combine all perspectives.
ЛОГИКА: (Подробный технический разбор на русском)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT/CLOSE/HOLD)
"""
        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"Error: {e}\nЛОГИКА: Ошибка ИИ\nУВЕРЕННОСТЬ: 0\nСИГНАЛ: WAIT"

    def parse_response(self, analysis):
        conf, direction, logic = 0, "WAIT", "Анализ не получен"
        lines = analysis.split('\n')
        for line in lines:
            up = line.upper().strip()
            if "ЛОГИКА:" in up: logic = line.split(":", 1)[1].strip() if ":" in line else ""
            if "УВЕРЕННОСТЬ:" in up:
                nums = re.findall(r'\d+', line)
                if nums: conf = min(int(nums[0]), 10)
            if "СИГНАЛ:" in up:
                for s in ["BUY", "SELL", "CLOSE", "HOLD"]:
                    if s in up: direction = s; break
        return conf, direction, logic

agent = TradingAgent()