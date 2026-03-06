import MetaTrader5 as mt5
from config import SYMBOL, BREAKEVEN_ATR, TRAILING_STEP_ATR, TRAILING_ACTIVATION_ATR


class TrailingManager:
    def __init__(self):
        self.last_atr = {}

    def set_atr(self, ticket, atr):
        self.last_atr[ticket] = atr

    def manage_positions(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return []
        actions = []
        for pos in positions:
            info = mt5.symbol_info(SYMBOL)
            if not info:
                continue
            point = info.point
            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                continue
            atr = self.last_atr.get(pos.ticket, 0.0010)
            be_dist = atr * BREAKEVEN_ATR
            trail_act = atr * TRAILING_ACTIVATION_ATR
            trail_step = atr * TRAILING_STEP_ATR
            be_offset = point * 2
            if pos.type == 0:  # BUY
                current_price = tick.bid
                profit_dist = current_price - pos.price_open
                if profit_dist >= be_dist:
                    new_sl = pos.price_open + be_offset
                    if pos.sl < new_sl:
                        if self._modify_sl(pos, new_sl):
                            pips = profit_dist / point
                            actions.append(f"BE BUY #{pos.ticket}: SL={new_sl:.5f} (+{pips:.0f}p)")
                if profit_dist >= trail_act:
                    trail_sl = current_price - trail_step
                    if trail_sl > pos.sl:
                        if self._modify_sl(pos, trail_sl):
                            pips = profit_dist / point
                            actions.append(f"TRAIL BUY #{pos.ticket}: SL={trail_sl:.5f} (+{pips:.0f}p)")
            elif pos.type == 1:  # SELL
                current_price = tick.ask
                profit_dist = pos.price_open - current_price
                if profit_dist >= be_dist:
                    new_sl = pos.price_open - be_offset
                    if pos.sl > new_sl or pos.sl == 0:
                        if self._modify_sl(pos, new_sl):
                            pips = profit_dist / point
                            actions.append(f"BE SELL #{pos.ticket}: SL={new_sl:.5f} (+{pips:.0f}p)")
                if profit_dist >= trail_act:
                    trail_sl = current_price + trail_step
                    if trail_sl < pos.sl:
                        if self._modify_sl(pos, trail_sl):
                            pips = profit_dist / point
                            actions.append(f"TRAIL SELL #{pos.ticket}: SL={trail_sl:.5f} (+{pips:.0f}p)")
        return actions

    def _modify_sl(self, position, new_sl):
        info = mt5.symbol_info(SYMBOL)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": SYMBOL,
            "position": position.ticket,
            "sl": round(new_sl, info.digits),
            "tp": position.tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        return False


trailing = TrailingManager()