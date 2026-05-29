import numpy as np
from backtest.events import SignalEvent


class MomentumStrategy:
    """
    Rank-based momentum. Buys the symbol if its return over
    the lookback period exceeds the threshold, and exits otherwise.
    Works best across a basket of symbols.
    """

    def __init__(self, data_handler, events, lookback=90, threshold=0.0):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = data_handler.symbol_list
        self.lookback = lookback
        self.threshold = threshold
        self.in_trade = {s: False for s in self.symbol_list}

    def calculate_signals(self, event):
        for symbol in self.symbol_list:
            bars = self.data_handler.get_latest_bars_values(
                symbol, "adj close", N=self.lookback
            )
            if len(bars) < self.lookback:
                continue
            momentum = (bars[-1] - bars[0]) / bars[0]
            if momentum > self.threshold and not self.in_trade[symbol]:
                self.events.put(SignalEvent(symbol, "LONG"))
                self.in_trade[symbol] = True
            elif momentum <= self.threshold and self.in_trade[symbol]:
                self.events.put(SignalEvent(symbol, "EXIT"))
                self.in_trade[symbol] = False
