"""
backtest
--------
Event-driven backtesting engine.

Components
----------
events          : Core event types (Bar, MarketEvent, SignalEvent, OrderEvent, FillEvent)
data_feed       : HistoricalDataFeed + synthetic data generator
fill_simulator  : Realistic order execution with spread, slippage, partial fills
portfolio       : Position tracker, cash management, equity curve
strategy        : BaseStrategy interface
engine          : BacktestEngine — drives the event loop
analytics       : Performance metrics and reporting
"""

from .events import (
    Bar, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, OrderSide, OrderType, FillStatus,
)
from .data_feed import HistoricalDataFeed, generate_synthetic_data
from .fill_simulator import FillSimulator, FillSimulatorConfig
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .engine import BacktestEngine, PositionSizer
from .analytics import compute_metrics, print_report

__all__ = [
    "Bar", "MarketEvent", "SignalEvent", "OrderEvent", "FillEvent",
    "EventType", "OrderSide", "OrderType", "FillStatus",
    "HistoricalDataFeed", "generate_synthetic_data",
    "FillSimulator", "FillSimulatorConfig",
    "Portfolio",
    "BaseStrategy",
    "BacktestEngine", "PositionSizer",
    "compute_metrics", "print_report",
]
