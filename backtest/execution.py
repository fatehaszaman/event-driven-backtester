from abc import ABC, abstractmethod
from backtest.events import FillEvent


class ExecutionHandler(ABC):

    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Fills orders at the current market price with no latency.
    Slippage and market impact are not modeled here -- keep that
    in mind when interpreting results on illiquid instruments.
    """

    def __init__(self, events, data_handler, slippage=0.0):
        self.events = events
        self.data_handler = data_handler
        self.slippage = slippage

    def execute_order(self, event):
        if event.type.value != "ORDER":
            return

        symbol = event.symbol
        direction = event.direction
        quantity = event.quantity

        try:
            fill_price = self.data_handler.get_latest_bar_value(symbol, "adj close")
        except Exception:
            return

        if direction == "BUY":
            fill_price *= (1 + self.slippage)
        else:
            fill_price *= (1 - self.slippage)

        fill_cost = fill_price * quantity
        timeindex = self.data_handler.get_latest_bar(symbol)[0]

        fill = FillEvent(
            timeindex=timeindex,
            symbol=symbol,
            exchange="SIM",
            quantity=quantity,
            direction=direction,
            fill_cost=fill_cost,
        )
        self.events.put(fill)
