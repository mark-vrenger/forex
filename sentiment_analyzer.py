import urllib.request
import json
import re
from datetime import datetime


class SentimentAnalyzer:
    """
    Парсит бесплатные RSS/JSON ленты финансовых новостей.
    Извлекает заголовки и передаёт в GPT для анализа настроений.
    """

    def __init__(self):
        self.sources = [
            {
                "name": "ForexLive",
                "url": "https://www.forexlive.com/feed/news",
                "type": "rss"
            },
            {
                "name": "DailyFX",
                "url": "https://www.dailyfx.com/feeds/market-news",
                "type": "rss"
            },
        ]
        self.cache = []
        self.cache_time = None
        self.cache_ttl = 900  # 15 минут

    def get_headlines(self, max_headlines=10):
        """Получает последние заголовки новостей"""
        now = datetime.now()
        if self.cache and self.cache_time:
            diff = (now - self.cache_time).total_seconds()
            if diff < self.cache_ttl:
                return self.cache

        headlines = []
        for source in self.sources:
            try:
                fetched = self._fetch_rss(source["url"], source["name"])
                headlines.extend(fetched)
            except Exception:
                continue

        # Фильтруем только EUR/USD релевантные
        keywords = [
            'eur', 'usd', 'euro', 'dollar', 'fed', 'ecb',
            'forex', 'fx', 'currency', 'rate', 'inflation',
            'gdp', 'employment', 'nfp', 'fomc', 'cpi', 'pmi',
            'trade', 'tariff', 'recession', 'growth'
        ]
        filtered = []
        for h in headlines:
            lower = h["title"].lower()
            if any(kw in lower for kw in keywords):
                filtered.append(h)

        self.cache = filtered[:max_headlines]
        self.cache_time = now
        return self.cache

    def get_sentiment_summary(self):
        """Формирует сводку для GPT промпта"""
        headlines = self.get_headlines()
        if not headlines:
            return "No recent news headlines available."

        lines = ["-- NEWS HEADLINES (for sentiment) --"]
        for h in headlines:
            lines.append(f"  [{h['source']}] {h['title']}")

        lines.append("")
        lines.append(
            "Analyze these headlines for EUR/USD sentiment. "
            "Are they bullish or bearish for EUR? For USD?"
        )
        return "\n".join(lines)

    def _fetch_rss(self, url, source_name, timeout=10):
        """Простой парсер RSS без внешних библиотек"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Trading Bot)'
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
            content = response.read().decode('utf-8', errors='ignore')
        except Exception:
            return []

        # Извлекаем заголовки из RSS XML
        titles = re.findall(r'<title[^>]*>(.*?)</title>', content, re.DOTALL)
        headlines = []
        for title in titles[:10]:
            # Очистка от CDATA и HTML
            clean = re.sub(r'<!\[CDATA\[|\]\]>', '', title)
            clean = re.sub(r'<[^>]+>', '', clean).strip()
            if clean and len(clean) > 10:
                headlines.append({
                    "source": source_name,
                    "title": clean,
                    "time": datetime.now().strftime("%H:%M")
                })
        return headlines


sentiment = SentimentAnalyzer()