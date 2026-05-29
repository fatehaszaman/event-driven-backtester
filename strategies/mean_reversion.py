import numpy as np
from backtest.events import SignalEvent


class MeanReversionStrategy:
    """
    Z-score mean reversion. Enters long when price falls more than
    `entry_z` standard deviations below the rolling mean, and exits
    once it reverts back within `exit_z` standard deviations.
    """

    def __init__(self, data_handler, events, lookback=20, entry_z=2.0, exit_z=0.5):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = data_handler.symbol_list
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.in_trade = {s: False for s in self.symbol_list}

    def calculate_signals(self, event):
        for symbol in self.symbol_list:
            bars = self.data_handler.get_latest_bars_values(
                symbol, "adj close", N=self.lookback
            )
            if len(bars) < self.lookback:
                continue
            mean = np.mean(bars)
            std = np.std(bars)
            if std == 0:
                continue
            z = (bars[-1] - mean) / std
            if z < -self.entry_z and not self.in_trade[symbol]:
                self.events.put(SignalEvent(symbol, "LONG"))
                self.in_trade[symbol] = True
            elif abs(z) < self.exit_z and self.in_trade[symbol]:
                self.events.put(SignalEvent(symbol, "EXIT"))
                self.in_trade[symbol] = False
