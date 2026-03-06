import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from config import CORR_RECALC_HOURS, CORR_WINDOW


class CorrelationAnalyzer:
    def __init__(self):
        self.instruments = {
            "GBPUSDrfd": (+1, "Direct"),
            "USDCHFrfd": (-1, "Inverse"),
            "USDJPYrfd": (-1, "Inverse"),
            "XAUUSDrfd": (+1, "Direct"),
        }
        self.live_correlations = {}
        self.last_recalc = None

    def _should_recalculate(self):
        if self.last_recalc is None:
            return True
        hours = (datetime.now() - self.last_recalc).total_seconds() / 3600
        return hours >= CORR_RECALC_HOURS

    def _recalculate_correlations(self):
        from config import SYMBOL
        main_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, CORR_WINDOW)
        if main_rates is None or len(main_rates) < 30:
            return
        main_returns = pd.DataFrame(main_rates)['close'].pct_change().dropna()
        for symbol in self.instruments:
            try:
                info = mt5.symbol_info(symbol)
                if info is None:
                    continue
                if not info.visible:
                    mt5.symbol_select(symbol, True)
                other_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, CORR_WINDOW)
                if other_rates is None or len(other_rates) < 30:
                    continue
                other_returns = pd.DataFrame(other_rates)['close'].pct_change().dropna()
                min_len = min(len(main_returns), len(other_returns))
                if min_len < 20:
                    continue
                corr = np.corrcoef(
                    main_returns.tail(min_len).values,
                    other_returns.tail(min_len).values
                )[0, 1]
                self.live_correlations[symbol] = round(corr, 3)
            except Exception:
                continue
        self.last_recalc = datetime.now()

    def analyze(self):
        if self._should_recalculate():
            self._recalculate_correlations()
        results = []
        buy_votes = 0
        sell_votes = 0
        for symbol, (expected_corr, desc) in self.instruments.items():
            trend = self._get_trend(symbol)
            if trend is None:
                continue
            direction = trend["direction"]
            strength = trend["strength"]
            live_corr = self.live_correlations.get(symbol)
            corr_str = f"{live_corr:.2f}" if live_corr is not None else "N/A"
            actual_corr = live_corr if live_corr is not None else expected_corr
            if actual_corr > 0:
                if direction == "UP":
                    buy_votes += strength
                elif direction == "DOWN":
                    sell_votes += strength
            else:
                if direction == "UP":
                    sell_votes += strength
                elif direction == "DOWN":
                    buy_votes += strength
            signal = self._interpret(direction, 1 if actual_corr > 0 else -1)
            results.append(
                f"  {symbol}: {direction} str:{strength}/3 "
                f"corr:{corr_str} -> {signal}"
            )
        if buy_votes > sell_votes + 2:
            verdict = "STRONG BUY"
        elif sell_votes > buy_votes + 2:
            verdict = "STRONG SELL"
        elif buy_votes > sell_votes:
            verdict = "LEAN BUY"
        elif sell_votes > buy_votes:
            verdict = "LEAN SELL"
        else:
            verdict = "NEUTRAL"
        recalc = self.last_recalc.strftime("%H:%M") if self.last_recalc else "never"
        header = f"BUY:{buy_votes} SELL:{sell_votes} -> {verdict} (recalc:{recalc})"
        summary = "-- CORRELATIONS --\n" + header + "\n"
        summary += "\n".join(results) if results else "  No data"
        return summary, buy_votes, sell_votes, verdict

    def _get_trend(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
        if rates is None or len(rates) < 15:
            return None
        df = pd.DataFrame(rates)
        sma5 = df['close'].tail(5).mean()
        sma15 = df['close'].tail(15).mean()
        diff = abs(sma5 - sma15) / sma15 * 100
        if diff > 0.1:
            strength = 3
        elif diff > 0.03:
            strength = 2
        else:
            strength = 1
        if sma5 > sma15:
            d = "UP"
        elif sma5 < sma15:
            d = "DOWN"
        else:
            d = "FLAT"
            strength = 0
        return {"direction": d, "strength": strength}

    def _interpret(self, direction, corr):
        if corr > 0:
            return "supports BUY" if direction == "UP" else "supports SELL" if direction == "DOWN" else "neutral"
        else:
            return "supports SELL" if direction == "UP" else "supports BUY" if direction == "DOWN" else "neutral"

    def get_confidence_adjustment(self, ai_direction):
        _, bv, sv, _ = self.analyze()
        if ai_direction == "BUY":
            if bv > sv + 2:
                return +1
            elif sv > bv + 2:
                return -2
        elif ai_direction == "SELL":
            if sv > bv + 2:
                return +1
            elif bv > sv + 2:
                return -2
        return 0


correlator = CorrelationAnalyzer()