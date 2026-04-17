"""
events.py
---------
Core event types for the event-driven backtesting engine.

The engine is structured around an event queue. Each bar of market data
produces a MarketEvent, strategies consume MarketEvents and emit
SignalEvents, the order manager converts SignalEvents to OrderEvents,
and the fill simulator converts OrderEvents to FillEvents.

This separation means the same strategy and order logic runs identically
in backtesting and live trading — only the data feed and fill simulator
swap out.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER  = "ORDER"
    FILL   = "FILL"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class FillStatus(str, Enum):
    FULL    = "FULL"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


@dataclass
class Bar:
    """A single OHLCV bar for one instrument."""
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float          # Best bid at bar close
    ask: float          # Best ask at bar close

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class MarketEvent:
    """Fired when a new bar of data is available."""
    type: EventType = field(default=EventType.MARKET, init=False)
    bars: dict[str, Bar] = field(default_factory=dict)   # symbol -> Bar

    @property
    def timestamp(self) -> Optional[pd.Timestamp]:
        if self.bars:
            return next(iter(self.bars.values())).timestamp
        return None


@dataclass
class SignalEvent:
    """Emitted by a strategy to indicate a desired position direction."""
    type: EventType = field(default=EventType.SIGNAL, init=False)
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    strength: float = 1.0       # 0–1, used for position sizing
    strategy_id: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class OrderEvent:
    """A concrete order ready to be sent to the fill simulator."""
    type: EventType = field(default=EventType.ORDER, init=False)
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    order_id: str = ""


@dataclass
class FillEvent:
    """Confirmation of an executed order."""
    type: EventType = field(default=EventType.FILL, init=False)
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    fill_price: float           # Volume-weighted average fill price
    commission: float
    slippage: float             # Total slippage cost (absolute)
    status: FillStatus
    order_id: str = ""
