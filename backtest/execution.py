import logging
from abc import ABC, abstractmethod

from backtest.events import FillEvent

logger = logging.getLogger(__name__)


class ExecutionHandler(ABC):

    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError

    def on_market(self, event):
        """Called when a new bar arrives. Returns fills to apply before the bar is marked."""
        return []

    def cancel_pending(self):
        """Called when the data ends. Handlers that hold orders release them here."""


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Fills market orders at the next bar's open, plus proportional slippage.

    An order created while bar t is the latest bar is held until bar t+1
    arrives, then filled at bar t+1's open. The open is scaled by
    adj close / close so it is on the same adjusted basis the portfolio uses
    to mark positions. An order with no next bar is not filled; it is logged
    and recorded in `cancelled_orders`.

    Market impact and partial fills are not modeled.
    """

    def __init__(self, events, data_handler, slippage=0.0):
        self.events = events
        self.data_handler = data_handler
        self.slippage = slippage
        self.pending_orders = []
        self.cancelled_orders = []

    def execute_order(self, event):
        if event.type.value != "ORDER":
            return

        try:
            placed_at = self.data_handler.get_latest_bar(event.symbol)[0]
        except Exception:
            return  # KNOWN_ISSUES.md #3: unpriceable orders are discarded silently

        self.pending_orders.append((placed_at, event))

    def on_market(self, event=None):
        fills, still_pending = [], []
        for placed_at, order in self.pending_orders:
            timeindex, bar = self.data_handler.get_latest_bar(order.symbol)
            if timeindex <= placed_at:
                # No new bar for this symbol yet, for example the extra market
                # event at the end of the data (KNOWN_ISSUES.md #1).
                still_pending.append((placed_at, order))
                continue

            try:
                fill_price = bar["open"] * bar["adj close"] / bar["close"]
            except Exception:
                continue  # KNOWN_ISSUES.md #3: unpriceable orders are discarded silently

            if order.direction == "BUY":
                fill_price *= (1 + self.slippage)
            else:
                fill_price *= (1 - self.slippage)

            fills.append(FillEvent(
                timeindex=timeindex,
                symbol=order.symbol,
                exchange="SIM",
                quantity=order.quantity,
                direction=order.direction,
                fill_cost=fill_price * order.quantity,
            ))
        self.pending_orders = still_pending
        return fills

    def cancel_pending(self):
        for placed_at, order in self.pending_orders:
            logger.warning(
                "Order not filled: no bar after %s for %r", placed_at, order
            )
            self.cancelled_orders.append(order)
        self.pending_orders = []
