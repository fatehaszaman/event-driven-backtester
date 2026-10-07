import numpy as np
import pandas as pd

from backtest.performance import create_sharpe_ratio
from abc import ABC, abstractmethod
from backtest.events import OrderEvent


class Portfolio(ABC):

    @abstractmethod
    def update_signal(self, event):
        raise NotImplementedError

    @abstractmethod
    def update_fill(self, event):
        raise NotImplementedError


class NaivePortfolio(Portfolio):
    """
    Tracks positions and cash across the backtest.
    Converts signals into fixed-size market orders and logs
    every fill for performance analysis.
    """

    def __init__(self, data_handler, events, start_date, initial_capital=100_000.0):
        self.data_handler = data_handler
        self.events = events
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.symbol_list = data_handler.symbol_list
        self.all_positions = []
        self.current_positions = {s: 0 for s in self.symbol_list}
        self.all_holdings = []
        self.current_holdings = self._init_holdings()

    def _init_holdings(self):
        h = {s: 0.0 for s in self.symbol_list}
        h["datetime"] = self.start_date
        h["cash"] = self.initial_capital
        h["commission"] = 0.0
        h["total"] = self.initial_capital
        return h

    def update_timeindex(self, event):
        bars = {s: self.data_handler.get_latest_bar(s) for s in self.symbol_list}
        dt = bars[self.symbol_list[0]][0]

        dp = {"datetime": dt}
        for s in self.symbol_list:
            dp[s] = self.current_positions[s]
        self.all_positions.append(dp)

        dh = {"datetime": dt, "cash": self.current_holdings["cash"],
              "commission": self.current_holdings["commission"],
              "total": self.current_holdings["cash"]}
        for s in self.symbol_list:
            mv = self.current_positions[s] * self.data_handler.get_latest_bar_value(s, "adj close")
            dh[s] = mv
            dh["total"] += mv
        self.all_holdings.append(dh)

    def update_signal(self, event):
        order = self._generate_order(event)
        if order:
            self.events.put(order)

    def _generate_order(self, signal):
        order = None
        symbol = signal.symbol
        direction = signal.signal_type
        qty = int(100 * signal.strength)
        cur = self.current_positions[symbol]

        if direction == "LONG" and cur == 0:
            order = OrderEvent(symbol, "MKT", qty, "BUY")
        elif direction == "SHORT" and cur == 0:
            order = OrderEvent(symbol, "MKT", qty, "SELL")
        elif direction == "EXIT" and cur > 0:
            order = OrderEvent(symbol, "MKT", abs(cur), "SELL")
        elif direction == "EXIT" and cur < 0:
            order = OrderEvent(symbol, "MKT", abs(cur), "BUY")
        return order

    def update_fill(self, event):
        factor = 1 if event.direction == "BUY" else -1
        self.current_positions[event.symbol] += factor * event.quantity
        self.current_holdings[event.symbol] += factor * event.fill_cost
        self.current_holdings["commission"] += event.commission
        self.current_holdings["cash"] -= factor * event.fill_cost + event.commission
        self.current_holdings["total"] -= event.commission

    def output_summary_stats(self):
        totals = pd.DataFrame(self.all_holdings)["total"]
        total_return = totals.iloc[-1] / self.initial_capital - 1.0
        returns = totals.pct_change().dropna()
        sharpe = create_sharpe_ratio(returns, periods=252)
        max_dd = ((totals - totals.cummax()) / totals.cummax()).min()
        return [
            ("Total Return", f"{total_return * 100:.2f}%"),
            ("Sharpe Ratio", f"{sharpe:.4f}"),
            ("Max Drawdown", f"{max_dd * 100:.2f}%"),
        ]
