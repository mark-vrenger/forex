import MetaTrader5 as mt5
from datetime import datetime, timedelta


def get_upcoming_news(minutes_ahead=60):
    if not hasattr(mt5, 'calendar_get'):
        return []
    try:
        now = datetime.now()
        end_time = now + timedelta(minutes=minutes_ahead)
        news_usd = mt5.calendar_get(time_from=now, time_to=end_time, currency="USD")
        news_eur = mt5.calendar_get(time_from=now, time_to=end_time, currency="EUR")
        all_news = []
        if news_usd is not None:
            all_news = all_news + list(news_usd)
        if news_eur is not None:
            all_news = all_news + list(news_eur)
        high_impact = []
        for event in all_news:
            if hasattr(event, 'importance') and event.importance == 3:
                event_time = datetime.fromtimestamp(event.time)
                time_str = event_time.strftime("%H:%M")
                high_impact.append({
                    "time": time_str,
                    "name": event.name,
                    "currency": event.currency
                })
        return high_impact
    except Exception:
        return []