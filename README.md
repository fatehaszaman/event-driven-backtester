# Event-Driven Backtesting Engine

A Python backtesting engine built around an event-driven architecture — the same design pattern used in production trading systems. Strategies, execution logic, and data feeds are fully decoupled, so the same strategy code runs identically in backtesting and live trading.

---

## Why event-driven?

Most backtesting frameworks process data in vectorized batches. Fast, but it introduces look-ahead bias and makes it impossible to model realistic execution — partial fills, order sequencing, and latency all require per-event state.

This engine processes one bar at a time through an event queue:

```
MarketEvent → Strategy → SignalEvent → PositionSizer → OrderEvent → FillSimulator → FillEvent → Portfolio
```

Signals generated on bar N are filled on bar N+1 (next-bar execution). No look-ahead, no free fills at close.

---

## Components

### `backtest/data_feed.py` — Data Feed
Generates `MarketEvent` objects from historical OHLCV data. Handles multi-symbol timestamp alignment (LOCF for missing bars) and synthesizes bid/ask spreads when not present in source data.

Includes a synthetic data generator (GBM price paths with realistic spread and volume profiles) for testing.

### `backtest/fill_simulator.py` — Fill Simulator
Models the key sources of execution cost and constraint:

| Feature | Description |
|---|---|
| Bid/ask spread | Buys fill at ask, sells at bid |
| Market impact | Price moves against order proportional to volume participation |
| Partial fills | Orders exceeding `max_volume_pct` of bar volume are partially filled |
| Wide-spread rejection | Orders rejected when spread exceeds configurable threshold |
| Limit order logic | Limit price checked against bar bid/ask before filling |

### `backtest/portfolio.py` — Portfolio Tracker
Maintains cash balance, open positions (average entry price, realized/unrealized P&L), and a full trade log. Produces an equity curve with per-bar snapshots.

### `backtest/strategy.py` — Strategy Interface
`BaseStrategy` defines the interface all strategies implement. Strategies call `self.signal()` to emit `SignalEvent` objects — isolated from order sizing and execution concerns.

### `backtest/engine.py` — Backtest Runner
Drives the event loop, connects all components, and returns performance metrics. The `PositionSizer` converts signal strength to order quantity using fixed fractional sizing.

### `backtest/analytics.py` — Performance Analytics
Computes standard metrics from the equity curve and trade history:
- Total return, CAGR
- Sharpe, Sortino, Calmar ratios
- Max drawdown and drawdown duration
- Win rate, profit factor
- Commission and slippage breakdown

---

## Example Strategy

`strategies/mean_reversion.py` implements a mean-reversion strategy with volatility regime gating:

1. **Z-score**: price relative to a rolling fair-value band (rolling mean ± rolling std)
2. **Entry**: z-score crosses entry threshold → signal in the direction of reversion
3. **Exit**: z-score reverts to within the exit band
4. **Vol regime gate**: when realized vol exceeds its rolling average by a configurable factor, entry signals are suppressed — mean-reversion is unreliable in trending or high-vol conditions

---

## Quickstart

```bash
pip install -r requirements.txt
python examples/demo.py
```

The demo runs a 2-year backtest across three synthetic instruments and prints a full performance report.

---

## Project Structure

```
backtest_engine/
├── backtest/
│   ├── events.py           # Core event types
│   ├── data_feed.py        # Historical data feed + synthetic generator
│   ├── fill_simulator.py   # Execution simulator
│   ├── portfolio.py        # Position tracker and equity curve
│   ├── strategy.py         # BaseStrategy interface
│   ├── engine.py           # Backtest runner and position sizer
│   ├── analytics.py        # Performance metrics
│   └── __init__.py
├── strategies/
│   └── mean_reversion.py   # Mean-reversion + vol regime strategy
├── examples/
│   └── demo.py             # End-to-end demo
├── requirements.txt
└── README.md
```

---

## Writing a Custom Strategy

```python
from backtest import BaseStrategy, MarketEvent, OrderSide

class MyStrategy(BaseStrategy):
    def on_bar(self, event: MarketEvent) -> None:
        for symbol in self.symbols:
            if symbol not in event.bars:
                continue
            bar = event.bars[symbol]
            # Your logic here
            if some_condition:
                self.signal(bar.timestamp, symbol, OrderSide.BUY, strength=0.8)
```

---

## Requirements

- Python 3.10+
- pandas
- numpy
