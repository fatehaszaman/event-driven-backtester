"""
strategy.py
-----------
Base strategy interface.

All strategies inherit from BaseStrategy and implement on_bar().
The strategy receives a MarketEvent, can access bar history, and
emits SignalEvents via self.signal().

Design goals:
  - Strategies are stateful but isolated from execution concerns
  - The same strategy class runs in backtest and live with no changes
  - Signal strength (0–1) is used by the position sizer for scaling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from .events import MarketEvent, SignalEvent, OrderSide

if TYPE_CHECKING:
    from .data_feed import HistoricalDataFeed


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclasses implement on_bar() and call self.signal() to emit signals.

    Parameters
    ----------
    symbols : list[str]
        Instruments this strategy trades.
    strategy_id : str
        Unique identifier for this strategy instance.
    """

    def __init__(self, symbols: list[str], strategy_id: str = ""):
        self.symbols = symbols
        self.strategy_id = strategy_id or self.__class__.__name__
        self._signals: list[SignalEvent] = []
        self._feed: "HistoricalDataFeed | None" = None

    def attach_feed(self, feed: "HistoricalDataFeed") -> None:
        self._feed = feed

    def signal(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: OrderSide,
        strength: float = 1.0,
        meta: dict | None = None,
    ) -> None:
        """Emit a signal. Called from on_bar()."""
        self._signals.append(SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            strength=strength,
            strategy_id=self.strategy_id,
            meta=meta or {},
        ))

    def pop_signals(self) -> list[SignalEvent]:
        """Return and clear pending signals."""
        signals = self._signals[:]
        self._signals.clear()
        return signals

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> None:
        """
        Called on every new bar. Implement trading logic here.
        Call self.signal() to emit trade signals.
        """
        ...
