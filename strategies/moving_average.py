import numpy as np
from backtest.events import SignalEvent


class SMACrossoverStrategy:
    """
    Simple moving average crossover. Goes long when the short SMA
    crosses above the long SMA, and exits when it crosses back below.
    """

    def __init__(self, data_handler, events, short_window=20, long_window=50):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = data_handler.symbol_list
        self.short_window = short_window
        self.long_window = long_window
        self.bought = {s: False for s in self.symbol_list}

    def calculate_signals(self, event):
        for symbol in self.symbol_list:
            bars = self.data_handler.get_latest_bars_values(
                symbol, "adj close", N=self.long_window
            )
            if len(bars) < self.long_window:
                continue
            short_avg = np.mean(bars[-self.short_window:])
            long_avg = np.mean(bars[-self.long_window:])
            if short_avg > long_avg and not self.bought[symbol]:
                self.events.put(SignalEvent(symbol, "LONG"))
                self.bought[symbol] = True
            elif short_avg < long_avg and self.bought[symbol]:
                self.events.put(SignalEvent(symbol, "EXIT"))
                self.bought[symbol] = False


class EMACrossoverStrategy:
    """
    EMA crossover variant. Uses exponential moving averages instead
    of simple averages, reacting faster to recent price changes.
    """

    def __init__(self, data_handler, events, short_window=12, long_window=26):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = data_handler.symbol_list
        self.short_window = short_window
        self.long_window = long_window
        self.bought = {s: False for s in self.symbol_list}

    def _ema(self, values, period):
        alpha = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def calculate_signals(self, event):
        for symbol in self.symbol_list:
            bars = self.data_handler.get_latest_bars_values(
                symbol, "adj close", N=self.long_window
            )
            if len(bars) < self.long_window:
                continue
            short_ema = self._ema(bars[-self.short_window:], self.short_window)
            long_ema = self._ema(bars, self.long_window)
            if short_ema > long_ema and not self.bought[symbol]:
                self.events.put(SignalEvent(symbol, "LONG"))
                self.bought[symbol] = True
            elif short_ema < long_ema and self.bought[symbol]:
                self.events.put(SignalEvent(symbol, "EXIT"))
                self.bought[symbol] = False
