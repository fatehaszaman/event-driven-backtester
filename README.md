# High-Fidelity Event-Driven Backtester

### Discrete-Event Simulation for Systematic Strategies

This engine is built around the "speed vs. realism" trade-off that most
Python backtesters silently lose. Vectorized frameworks compute P&L by
broadcasting NumPy operations across an entire price series at once.
That is fast, but it leaks information across bars — strategies end up
trading on prices and statistics they could not have seen at decision
time, and execution costs (spread, partial fills, rejections) are either
ignored or bolted on after the fact.

This engine instead processes one bar at a time through an explicit
event queue, using a discrete-event simulation (DES) model that mirrors
the architecture of a production trading system. The same strategy
code runs identically in backtest and live trading — only the data
feed and fill simulator are swapped.

```
MarketEvent -> Strategy -> SignalEvent -> PositionSizer -> OrderEvent -> FillSimulator -> FillEvent -> Portfolio
```

Signals generated on bar `N` are filled on bar `N+1` (next-bar
execution). No look-ahead, no free fills at the close.

---

## Performance

The engine is pure Python on top of NumPy and pandas. Hot paths (synthetic
data generation, equity-curve analytics, market-impact math) are
vectorized through NumPy; the event loop itself is intentionally
imperative, because per-event state is what gives the simulation its
realism.

Reproducible numbers on a single CPU core, `python 3.12`, no GPU and no
external services. Each row runs the full mean-reversion strategy
end-to-end (signal generation, sizing, fill simulation, mark-to-market,
analytics):

| Bars  | Instruments | Wall time | Market events / s | Symbol-bars / s | Peak RSS |
|-------|-------------|-----------|-------------------|-----------------|----------|
|   500 | 3           | 0.40 s    | 1,258             | 3,774           | 71 MiB   |
| 2,000 | 5           | 1.98 s    | 1,008             | 5,041           | 80 MiB   |
| 5,000 | 10          | 8.35 s    | 599               | 5,990           | 117 MiB  |

> Reproduce with `python scripts/bench.py`. These are real measurements
> from this repo, not headline numbers from a different system.
> Throughput here is dominated by the per-bar DataFrame slicing inside
> the data feed and the Python-level event dispatch — both deliberate
> choices in favor of clarity over raw ticks/sec.

### Hardening roadmap (next steps, not current claims)

Things I would build next if pushing this toward production-grade
throughput, and which are explicitly **not** in the repo today:

- **NumPy-native data feed**: replace the per-timestamp `DataFrame`
  filter with column-array slicing to eliminate the dominant bottleneck.
  Target: 50–100x throughput on the same hardware.
- **Numba / Cython JIT on the fill simulator**: the market-impact and
  spread math is a tight numeric kernel and a natural fit for JIT.
- **C++ event-loop core** with a thin Python binding (pybind11) for
  cases where 10⁵+ events/sec are needed (intraday tick data,
  agent-based simulations).
- **State snapshots / deterministic replay**: serialize portfolio +
  strategy + RNG state between bars to enable bit-exact replay from any
  point. The architecture already supports this — the engine is purely
  a function of the prior state plus the next event — but it is not
  yet wired up.

---

## Architecture

### Event types ([`backtest/events.py`](backtest/events.py))

Five concrete event dataclasses (`Bar`, `MarketEvent`, `SignalEvent`,
`OrderEvent`, `FillEvent`) form the simulation's vocabulary. Every
piece of state that crosses a component boundary is an immutable event
— this is what makes the same code reusable in live trading and what
makes the system amenable to snapshot-based replay.

### Microstructure realism ([`backtest/fill_simulator.py`](backtest/fill_simulator.py))

The fill simulator models the execution frictions that determine
whether a strategy is actually tradable:

| Feature                | Behavior                                                                                |
|------------------------|------------------------------------------------------------------------------------------|
| Bid/ask crossing       | Buys fill at the ask, sells fill at the bid                                              |
| Linear market impact   | Fill price moves against the order, proportional to volume participation                 |
| Partial fills          | Orders exceeding `max_volume_pct` of bar volume are partially filled (or rejected)       |
| Wide-spread rejection  | Orders rejected when `spread / mid` exceeds a configurable threshold (thin-book guard)   |
| Limit-order semantics  | Limits checked against bar bid/ask before any fill — no peeking at the next bar         |
| Commission             | Linear in notional, configurable per venue                                               |

Bid/ask are taken from the source data when present and synthesized
from a configurable spread model otherwise — this lets the same engine
consume daily OHLCV, hourly data with a spread overlay, or true L1
quotes without changing the rest of the system.

### Deterministic execution ([`backtest/engine.py`](backtest/engine.py), [`backtest/data_feed.py`](backtest/data_feed.py))

- Event order is fixed: fills from the previous bar settle first,
  then mark-to-market, then strategy, then order generation.
- Multi-symbol timestamp alignment uses last-observation-carried-forward
  (LOCF) so strategies always see a consistent cross-sectional snapshot.
- Synthetic data generation accepts an explicit RNG seed — given the
  same seed and inputs, the engine produces identical equity curves and
  trade logs across runs.

### Portfolio & analytics ([`backtest/portfolio.py`](backtest/portfolio.py), [`backtest/analytics.py`](backtest/analytics.py))

The portfolio tracks cash, per-symbol positions (average entry,
realized/unrealized P&L), and a full trade log. Per-bar snapshots feed
the equity curve. Analytics computes the standard set:

- Total return, CAGR, annualized volatility
- Sharpe, Sortino, Calmar
- Max drawdown and drawdown duration
- Win rate, profit factor, average win/loss
- Commission and slippage broken out separately, so you can see how
  much of a loss comes from market direction versus pure execution
  friction

### Strategy interface ([`backtest/strategy.py`](backtest/strategy.py))

`BaseStrategy` is intentionally small: subclass it, implement
`on_bar`, and call `self.signal(...)` to emit signals. Sizing and
execution are not the strategy's concern. The included
[`strategies/mean_reversion.py`](strategies/mean_reversion.py)
demonstrates a real strategy on top of the interface: z-score entry/exit
bands plus a realized-vol regime gate that suppresses entries when
volatility is elevated relative to its rolling average.

---

## How to run

```bash
pip install -r requirements.txt   # pandas, numpy
python examples/demo.py           # 2-year, 3-instrument backtest with full report
```

The demo prints the data summary, a per-100-bar equity ticker, the
performance report, and a per-instrument trade breakdown.

To reproduce the performance table above:

```bash
python scripts/bench.py
```

---

## Writing a custom strategy

```python
from backtest import BaseStrategy, MarketEvent, OrderSide

class MyStrategy(BaseStrategy):
    def on_bar(self, event: MarketEvent) -> None:
        for symbol in self.symbols:
            if symbol not in event.bars:
                continue
            bar = event.bars[symbol]
            if some_condition(bar):
                self.signal(bar.timestamp, symbol, OrderSide.BUY, strength=0.8)
```

The engine handles sizing, order generation, fills, and accounting.
The strategy is responsible only for deciding *what direction* to be
in, at *which bar*, with *what conviction*.

---

## Project structure

```
event-driven-backtester/
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
│   └── mean_reversion.py   # Mean-reversion + vol regime example
├── examples/
│   └── demo.py             # End-to-end demo
├── scripts/
│   └── bench.py            # Reproduces the performance table
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+, pandas, numpy.
