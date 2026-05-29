from enum import Enum


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Event:
    pass


class MarketEvent(Event):
    def __init__(self):
        self.type = EventType.MARKET


class SignalEvent(Event):
    def __init__(self, symbol, signal_type, strength=1.0):
        self.type = EventType.SIGNAL
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent(Event):
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = EventType.ORDER
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def __repr__(self):
        return (
            f"OrderEvent(symbol={self.symbol}, type={self.order_type}, "
            f"qty={self.quantity}, direction={self.direction})"
        )


class FillEvent(Event):
    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):
        self.type = EventType.FILL
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission if commission is not None else self._ib_commission()

    def _ib_commission(self):
        if self.quantity <= 500:
            cost = max(1.3, 0.013 * self.quantity)
        else:
            cost = max(1.3, 0.008 * self.quantity)
        return min(cost, 0.005 * self.fill_cost)
