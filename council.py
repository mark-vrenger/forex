"""
council.py — Консилиум трейдеров v2.1
Три агента DeepSeek V3.2 с разными ролями: IMPULSE / TREND / ANALYST
Исправлено: правильное имя модели deepseek/deepseek-v3.2, ASCII вывод.
"""

import os
import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from config import (
    GPT_TIMEOUT, GPT_MAX_RETRIES,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    MIN_CONFIDENCE,
)

load_dotenv()

# Рабочее имя модели — то же что работало в оригинале
DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"


# ─────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class AgentVoice:
    name: str        # 'IMPULSE' | 'TREND' | 'ANALYST'
    label: str       # ASCII-лейбл для вывода
    signal: str      # 'BUY' | 'SELL' | 'WAIT'
    confidence: int  # 1-10
    reasoning: str
    changed_from: str = ""
    round_num: int = 1


@dataclass
class CouncilDecision:
    signal: str
    confidence: int
    votes: List[AgentVoice] = field(default_factory=list)
    debate_votes: List[AgentVoice] = field(default_factory=list)
    blocked_by: str = ""
    block_reason: str = ""
    corr_adj: int = 0
    corr_summary: str = ""
    news_clear: bool = True
    news_detail: str = ""
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    price: float = 0.0
    session_time: str = ""
    consensus_pct: float = 0.0


# ─────────────────────────────────────────
#  LLM FACTORY
# ─────────────────────────────────────────

def _make_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        request_timeout=GPT_TIMEOUT,
        max_retries=GPT_MAX_RETRIES,
    )


# ─────────────────────────────────────────
#  BASE AGENT
# ─────────────────────────────────────────

class _BaseAgent:
    NAME  = "BASE"
    LABEL = "[BASE           ]"
    TEMPERATURE = 0.2

    def __init__(self):
        self.llm = _make_llm(self.TEMPERATURE)

    def _call(self, prompt: str) -> str:
        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"ЛОГИКА: Ошибка: {e}\nУВЕРЕННОСТЬ: 0\nСИГНАЛ: WAIT"

    def _parse(self, text: str, round_num: int = 1) -> AgentVoice:
        conf = 0
        signal = "WAIT"
        reasoning = ""
        for line in text.split("\n"):
            up = line.upper().strip()
            if "ЛОГИКА:" in up:
                reasoning = line.split(":", 1)[-1].strip()
            if "УВЕРЕННОСТЬ:" in up:
                nums = re.findall(r"\d+", line)
                if nums:
                    conf = min(int(nums[0]), 10)
            if "СИГНАЛ:" in up:
                for s in ["BUY", "SELL"]:
                    if s in up:
                        signal = s
                        break
        if not reasoning:
            for line in text.split("\n"):
                line = line.strip()
                if (len(line) > 20
                        and "УВЕРЕН" not in line.upper()
                        and "СИГНАЛ" not in line.upper()):
                    reasoning = line[:120]
                    break
            if not reasoning:
                reasoning = text.strip()[:120]
        return AgentVoice(
            name=self.NAME,
            label=self.LABEL,
            signal=signal,
            confidence=conf,
            reasoning=reasoning,
            round_num=round_num,
        )

    def initial_vote(self, market_ctx: str) -> AgentVoice:
        raise NotImplementedError

    def debate_vote(self, market_ctx: str, others: List[AgentVoice]) -> AgentVoice:
        others_text = "\n".join(
            f"  [{v.name}]: {v.signal} ({v.confidence}/10) -- {v.reasoning[:100]}"
            for v in others
        )
        prompt = f"""{market_ctx}

=== POZITSII DRUGIKH AGENTOV ===
{others_text}

Eto raund DEBATOV. Prochitay argumenty kolleg.
Mozesh ostatsya pri svoyom ili izmenit mneniye.

ЛОГИКА: (na russkom, s uchetom kolleg)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT)
"""
        raw = self._call(prompt)
        return self._parse(raw, round_num=2)


# ─────────────────────────────────────────
#  THREE DEEPSEEK V3.2 AGENTS
# ─────────────────────────────────────────

class ImpulseAgent(_BaseAgent):
    """
    DeepSeek V3.2 -- агент импульса.
    Заменяет Grok. Фокус: M5/M15 моментум, объём, паттерны свечей.
    """
    NAME  = "IMPULSE"
    LABEL = "[IMPULSE/DSv3.2]"
    TEMPERATURE = 0.4

    def initial_vote(self, market_ctx: str) -> AgentVoice:
        prompt = f"""Ty -- agent impulsnoy torgovli IMPULSE (DeepSeek V3.2).
Tvoy stil: agressivnyy skalper, lovish kratkosrochnye dvizheniya.

{market_ctx}

ZADACHA:
- Analizi M5 i M15: momentum, obem (ratio > 1.5 = signal)
- Ishi patterny: Engulfing, Hammer, Shooting Star
- Esli net chetkogo impulsa -- govori WAIT
- RSI: pokupay pri < 35, prodavay pri > 65
- Torguyi TOLKO pri silnom impulse

ЛОГИКА: (kratko i konkretno na russkom)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT)
"""
        return self._parse(self._call(prompt), round_num=1)


class TrendAgent(_BaseAgent):
    """
    DeepSeek V3.2 -- агент тренда.
    Заменяет GPT-4o Conservative. Фокус: H1/H4 тренд.
    """
    NAME  = "TREND"
    LABEL = "[TREND  /DSv3.2]"
    TEMPERATURE = 0.1

    def initial_vote(self, market_ctx: str) -> AgentVoice:
        prompt = f"""Ty -- agent trendovoy torgovli TREND (DeepSeek V3.2).
Tvoy printsip: snachala ne navredi. Torguy tolko s trendom.

{market_ctx}

ZADACHA:
- Prover trend H1 i H4 (SMA5 vs SMA15)
- WAIT esli H1 i H4 trendy ne sovpadayut
- Prover ADX: esli < 20 -- rynok fleytovyy, luchshe WAIT
- Uverennost > 7 tolko pri ochen chetkom trende
- Ne torguy protiv osnovnogo trenda H4

ЛОГИКА: (na russkom, konservativno)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT)
"""
        return self._parse(self._call(prompt), round_num=1)


class AnalystAgent(_BaseAgent):
    """
    DeepSeek V3.2 -- агент Price Action.
    Заменяет Claude Analyst. Фокус: уровни, Nison/Brooks, RSI div.
    """
    NAME  = "ANALYST"
    LABEL = "[ANALYST/DSv3.2]"
    TEMPERATURE = 0.2

    def initial_vote(self, market_ctx: str) -> AgentVoice:
        prompt = f"""Ty -- analitik Price Action ANALYST (DeepSeek V3.2).
Chitaesh rynok cherez strukturu, urovni i povedeniye tseny.

{market_ctx}

ZADACHA:
- Naydi klyuchevye urovni podderzhki i soprotivleniya
- Otsen patterny svechey po Nison i Brooks
- Prover divergentsiyu RSI (bychia/medvezhya)
- Otsen shirinu Bollindzherab (szhatiye = proriv skoro)
- Day nezavisimuyu tekhnicheskuyu otsenku

ЛОГИКА: (na russkom, tekhnicheski detalno)
УВЕРЕННОСТЬ: (1-10)
СИГНАЛ: (BUY/SELL/WAIT)
"""
        return self._parse(self._call(prompt), round_num=1)


# ─────────────────────────────────────────
#  MARKET CONTEXT BUILDER
# ─────────────────────────────────────────

def build_market_context(
    m5, m15, h1, h4,
    price: float,
    spread: float,
    tech_summary: str,
    rsi: float,
    atr: float,
    session: str,
    balance: float,
) -> str:
    import pandas as pd

    def _tf_line(raw, label):
        if raw is None or len(raw) == 0:
            return f"[{label}]: net dannykh"
        df = pd.DataFrame(raw)
        if 'close' not in df.columns and len(df.columns) >= 5:
            df.columns = ['time','open','high','low','close',
                          'tick_volume','spread','real_volume'][:len(df.columns)]
        sma5  = df['close'].tail(5).mean()
        sma15 = df['close'].tail(15).mean()
        trend = "UP" if sma5 > sma15 else "DOWN" if sma5 < sma15 else "FLAT"
        avg_v   = df['tick_volume'].tail(20).mean()
        v_ratio = df['tick_volume'].iloc[-1] / avg_v if avg_v > 0 else 1.0
        return (f"[{label}] Tsena:{df['close'].iloc[-1]:.5f} | "
                f"Trend:{trend} | SMA5:{sma5:.5f} SMA15:{sma15:.5f} | "
                f"Vol ratio:{v_ratio:.2f}")

    lines = [
        "=== RYNOCHNYY KONTEKST (EURUSD) ===",
        f"Vremya: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sessiya: {session}",
        f"Tsena: {price:.5f} | Spred: {spread:.1f}p | ATR: {atr:.5f} | RSI: {rsi:.1f}",
        f"Balans: ${balance:.2f}",
        "",
        _tf_line(h4,  "H4"),
        _tf_line(h1,  "H1"),
        _tf_line(m15, "M15"),
        _tf_line(m5,  "M5"),
        "",
        "=== TEKHN. INDIKATORY ===",
        tech_summary[:600],
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────
#  COUNCIL ORCHESTRATOR
# ─────────────────────────────────────────

class Council:
    """
    Konsilium iz trekh agentov DeepSeek V3.2.
    R1: nezavisimye golosa
    R2: debaty (vidyat pozitsii drug druga)
    R3: veto (novosti, spred, korrelyatsiya)
    R4: finalnyy vzveshenny verdikt
    """

    def __init__(self):
        self.impulse  = ImpulseAgent()
        self.trend    = TrendAgent()
        self.analyst  = AnalystAgent()
        self._agents  = [self.impulse, self.trend, self.analyst]
        self._last_decision: Optional[CouncilDecision] = None

    # ── PUBLIC ──────────────────────────────

    def run_session(
        self,
        m5, m15, h1, h4,
        price: float,
        spread: float,
        rsi: float,
        atr: float,
        session: str,
        balance: float,
        tech_summary: str,
        corr_buy: int,
        corr_sell: int,
        corr_summary: str,
        news_events: list,
    ) -> CouncilDecision:

        ctx = build_market_context(
            m5, m15, h1, h4, price, spread,
            tech_summary, rsi, atr, session, balance
        )

        print("  [R1] Poluchayu golosa agentov...")
        r1_votes = self._round1(ctx)

        print("  [R2] Raund debatov...")
        r2_votes = self._round2(ctx, r1_votes)

        block_by, block_reason = self._round3_veto(
            r2_votes, news_events, corr_buy, corr_sell, spread
        )

        decision = self._round4_verdict(
            r1_votes, r2_votes,
            block_by, block_reason,
            corr_buy, corr_sell, corr_summary,
            not bool(news_events), str(news_events),
            price, atr, balance,
        )
        self._last_decision = decision
        return decision

    def print_protocol(self, decision: CouncilDecision):
        """ASCII protokol -- sovmestim s lyuboy kodirovkoy."""
        W = 68
        bar = "=" * W

        def box(text=""):
            pad = max(0, W - 2 - len(text))
            return f"| {text}{' ' * pad} |"

        def sep():
            return f"+{'-' * W}+"

        def sig_mark(s):
            return {
                "BUY":  ">> BUY ",
                "SELL": ">> SELL",
            }.get(s, "-- WAIT")

        lines = [f"+{bar}+"]
        ts = decision.session_time or datetime.now().strftime("%H:%M:%S")
        lines.append(box(f"  KONSILIUM (3x DeepSeek V3.2) | {ts} | {decision.price:.5f}"))
        lines.append(f"+{bar}+")

        # Round 1
        lines.append(box("  ROUND 1 -- Pervichnye golosa"))
        lines.append(sep())
        for v in decision.votes:
            lines.append(box(f"  {v.label}  {sig_mark(v.signal)}  ({v.confidence}/10)"))
            rsn = v.reasoning[:60] if v.reasoning else "---"
            lines.append(box(f"    Logika: {rsn}"))

        # Round 2
        if decision.debate_votes:
            lines.append(sep())
            lines.append(box("  ROUND 2 -- Debaty"))
            lines.append(sep())
            for v in decision.debate_votes:
                changed = f"  [byl: {v.changed_from}]" if v.changed_from else ""
                lines.append(box(f"  {v.label}  {sig_mark(v.signal)}  ({v.confidence}/10){changed}"))
                rsn = v.reasoning[:55] if v.reasoning else "---"
                lines.append(box(f"    Logika: {rsn}"))

        # Filters
        lines.append(sep())
        lines.append(box("  FILTRY"))
        lines.append(sep())
        news_str = "OK" if decision.news_clear else f"BLOCK: {decision.news_detail[:40]}"
        lines.append(box(f"  [NOVOSTI  ] {news_str}"))
        corr_str = f"{decision.corr_adj:+d} | {decision.corr_summary[:42]}"
        lines.append(box(f"  [KORREL   ] {corr_str}"))

        if decision.blocked_by:
            lines.append(sep())
            lines.append(box(f"  !! BLOKIROVKA: {decision.blocked_by}"))
            lines.append(box(f"     Prichina: {decision.block_reason[:55]}"))

        # Verdict
        lines.append(f"+{bar}+")
        v_str = f">> {decision.signal}"
        lines.append(box(
            f"  VERDIKT: {v_str:<10} Uverennost: {decision.confidence}/10"
            f"  Konsensus: {decision.consensus_pct:.0f}%"
        ))
        if decision.signal in ("BUY", "SELL") and decision.lot > 0:
            lines.append(box(
                f"  Lot: {decision.lot}   SL: {decision.sl:.5f}   TP: {decision.tp:.5f}"
            ))
        lines.append(f"+{bar}+")
        print("\n".join(lines))

    def save_protocol(self, decision: CouncilDecision):
        try:
            from database import db
            if db:
                db.record_council_session({
                    "time":          decision.session_time,
                    "price":         decision.price,
                    "signal":        decision.signal,
                    "confidence":    decision.confidence,
                    "consensus_pct": decision.consensus_pct,
                    "blocked_by":    decision.blocked_by,
                    "block_reason":  decision.block_reason,
                    "corr_adj":      decision.corr_adj,
                    "sl": decision.sl, "tp": decision.tp, "lot": decision.lot,
                    "votes_r1": [asdict(v) for v in decision.votes],
                    "votes_r2": [asdict(v) for v in decision.debate_votes],
                })
        except Exception as e:
            print(f"[Council] DB error: {e}")

        try:
            import os
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d')}_council.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                json.dump({
                    "time":      decision.session_time,
                    "signal":    decision.signal,
                    "confidence": decision.confidence,
                    "votes":     [asdict(v) for v in decision.votes],
                    "debate":    [asdict(v) for v in decision.debate_votes],
                }, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[Council] Log error: {e}")

    def build_telegram_message(self, decision: CouncilDecision) -> str:
        ts = decision.session_time or datetime.now().strftime("%H:%M")
        lines = [
            "*KONSILIUM RESHENIYE*",
            f"EURUSD | {ts} | {decision.price:.5f}",
        ]
        if decision.blocked_by:
            lines += [f"BLOKIROVKA: {decision.blocked_by}", decision.block_reason]
        else:
            lines += [
                f"SIGNAL: *{decision.signal}* | Uverennost: *{decision.confidence}/10*",
                f"Konsensus: {decision.consensus_pct:.0f}%",
                "",
                "Golosa agentov:",
            ]
            for v in decision.votes:
                lines.append(f"  {v.label}: {v.signal} ({v.confidence}) | {v.reasoning[:60]}")
            if decision.lot > 0:
                lines += [
                    "",
                    f"Lot: {decision.lot} | SL: {decision.sl:.5f} | TP: {decision.tp:.5f}",
                ]
        return "\n".join(lines)

    # ── PRIVATE ─────────────────────────────

    def _round1(self, ctx: str) -> List[AgentVoice]:
        votes = []
        for agent in self._agents:
            try:
                v = agent.initial_vote(ctx)
                votes.append(v)
                print(f"    {agent.LABEL}: {v.signal} ({v.confidence}/10) | {v.reasoning[:55]}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    {agent.NAME} ERROR: {e}")
                votes.append(AgentVoice(
                    name=agent.NAME, label=agent.LABEL,
                    signal="WAIT", confidence=0,
                    reasoning=str(e)[:80], round_num=1,
                ))
        return votes

    def _round2(self, ctx: str, r1_votes: List[AgentVoice]) -> List[AgentVoice]:
        r2_votes = []
        for i, agent in enumerate(self._agents):
            others = [v for j, v in enumerate(r1_votes) if j != i]
            try:
                v2 = agent.debate_vote(ctx, others)
                if v2.signal != r1_votes[i].signal:
                    v2.changed_from = r1_votes[i].signal
                r2_votes.append(v2)
                ch = f" [IZMENIL: {r1_votes[i].signal}->{v2.signal}]" if v2.changed_from else ""
                print(f"    {agent.LABEL}: {v2.signal} ({v2.confidence}/10){ch}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    {agent.NAME} debate ERROR: {e}")
                fallback = AgentVoice(
                    name=r1_votes[i].name, label=r1_votes[i].label,
                    signal=r1_votes[i].signal, confidence=r1_votes[i].confidence,
                    reasoning=r1_votes[i].reasoning, round_num=2,
                )
                r2_votes.append(fallback)
        return r2_votes

    def _round3_veto(
        self,
        votes: List[AgentVoice],
        news_events: list,
        corr_buy: int,
        corr_sell: int,
        spread: float,
    ):
        from config import MAX_SPREAD_PIPS, CLOSE_BEFORE_NEWS_MINUTES

        # 1) Высокорисковые новости
        if news_events:
            names = ", ".join(e.get("name", "?") for e in news_events[:3])
            return "SENTINEL", f"Novosti <{CLOSE_BEFORE_NEWS_MINUTES}min: {names}"

        # 2) Спред — мягкий лимит (x3 от config, т.к. rfd-символы шире)
        spread_limit = MAX_SPREAD_PIPS * 3
        if spread > spread_limit:
            return "SPREAD_GUARD", f"Spred {spread:.1f}p > {spread_limit:.0f}p"

        # 3) Жёсткое противоречие корреляции (только при разнице > 6)
        majority = self._majority_signal(votes)
        if majority == "BUY" and corr_sell > corr_buy + 6:
            return "CORRELATOR", f"Korrelyatsii protiv BUY (S={corr_sell} B={corr_buy})"
        if majority == "SELL" and corr_buy > corr_sell + 6:
            return "CORRELATOR", f"Korrelyatsii protiv SELL (B={corr_buy} S={corr_sell})"

        return "", ""

    def _round4_verdict(
        self,
        r1_votes, r2_votes,
        block_by, block_reason,
        corr_buy, corr_sell, corr_summary,
        news_clear, news_detail,
        price, atr, balance,
    ) -> CouncilDecision:

        active   = r2_votes if r2_votes else r1_votes
        majority = self._majority_signal(active)
        matching = [v for v in active if v.signal == majority]
        consensus_pct = len(matching) / len(active) * 100 if active else 0
        avg_conf = sum(v.confidence for v in matching) / len(matching) if matching else 0

        corr_adj = 0
        if majority == "BUY":
            if corr_buy > corr_sell + 2:    corr_adj = +1
            elif corr_sell > corr_buy + 2:  corr_adj = -1
        elif majority == "SELL":
            if corr_sell > corr_buy + 2:    corr_adj = +1
            elif corr_buy > corr_sell + 2:  corr_adj = -1

        final_conf = max(1, min(10, int(avg_conf) + corr_adj))

        if block_by:
            final_signal = "WAIT"
            final_conf   = 0
        elif consensus_pct < 67 or final_conf < MIN_CONFIDENCE:
            final_signal = "WAIT"
        else:
            final_signal = majority

        sl = tp = lot = 0.0
        if final_signal in ("BUY", "SELL") and atr > 0:
            sl_d = atr * 1.5
            tp_d = atr * 2.5
            if final_signal == "BUY":
                sl = round(price - sl_d, 5)
                tp = round(price + tp_d, 5)
            else:
                sl = round(price + sl_d, 5)
                tp = round(price - tp_d, 5)
            risk_money = balance * 0.01
            sl_pips    = sl_d / 0.00001
            lot = round(max(0.01, min(risk_money / (sl_pips * 10), 0.5)), 2) if sl_pips > 0 else 0.01

        return CouncilDecision(
            signal=final_signal,
            confidence=final_conf,
            votes=r1_votes,
            debate_votes=r2_votes,
            blocked_by=block_by,
            block_reason=block_reason,
            corr_adj=corr_adj,
            corr_summary=corr_summary[:60],
            news_clear=news_clear,
            news_detail=news_detail[:60],
            sl=sl, tp=tp, lot=lot,
            price=price,
            session_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            consensus_pct=consensus_pct,
        )

    @staticmethod
    def _majority_signal(votes: List[AgentVoice]) -> str:
        counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
        for v in votes:
            counts[v.signal if v.signal in counts else "WAIT"] += 1
        return max(counts, key=counts.get)


# ─────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.request, urllib.parse
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "Markdown",
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        print(f"[Telegram] Error: {e}")


# ─────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────

council = Council()
