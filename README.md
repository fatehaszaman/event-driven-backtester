# Event-Driven Backtester

A discrete-event backtesting engine in Python. Market, signal, order, and fill
events flow through a queue, which keeps the data, strategy, portfolio, and
execution layers genuinely decoupled rather than decoupled in description only.

68 tests. `KNOWN_ISSUES.md` lists ten defects found by writing them, with
severity and a fix for each.

## What is actually implemented

| | |
|---|---|
| Discrete-event loop over a shared queue | yes |
| CSV data handler, bars streamed strictly in date order | yes |
| SMA and EMA crossover, z-score mean reversion, momentum | yes |
| Long and short positions, per-fill cash and position accounting | yes |
| Fixed proportional slippage, applied against the trade direction | yes |
| Tiered per-share commission with a notional cap | yes |
| Sharpe, max drawdown, CAGR, equity and drawdown tearsheet | yes |

## What is not implemented

An earlier version of this README claimed several of these. It was wrong, and
the list is kept explicit so nobody has to read the source to find out.

| | |
|---|---|
| Bid/ask spread modelling | no — slippage is a single proportional factor |
| Market impact | no — `execution.py` says so directly |
| Partial fills | no — every order fills in full |
| Yahoo Finance or any non-CSV source | no — `CSVDataHandler` only |
| Calmar ratio, win rate | no |
| Fill latency | no — see the honest accounting below |

## Honest accounting of the fill assumption

The data handler does guarantee ordered delivery: after *k* calls to
`update_bars()`, exactly *k* bars are visible and no later bar can be read.
That property is tested directly.

Execution is a different matter. The engine drains the event queue within a
single bar, so a signal computed from that bar's closing price is filled at
that same closing price. Zero latency. The close was not tradable at the moment
it was observed, and the realistic assumption is a fill at the next bar's open.

For a crossover strategy this assumption is favourable, not neutral. **Every
return figure this engine produces is optimistic by an unmeasured amount.**
That is `KNOWN_ISSUES.md` #2 and it is the first thing worth fixing.

## Install

```bash
git clone https://github.com/fatehaszaman/event-driven-backtester.git
cd event-driven-backtester
pip install -r requirements-dev.txt
pytest tests -q
```

## Quick start

```python
from backtest.data import CSVDataHandler
from backtest.engine import Backtest
from backtest.execution import SimulatedExecutionHandler
from backtest.portfolio import NaivePortfolio
from strategies.moving_average import SMACrossoverStrategy

bt = Backtest(
    csv_dir="./data",
    symbol_list=["AAPL"],
    initial_capital=100_000,
    heartbeat=0.0,
    start_date="2018-01-01",
    end_date="2023-12-31",
    data_handler_cls=CSVDataHandler,
    execution_handler_cls=SimulatedExecutionHandler,
    portfolio_cls=NaivePortfolio,
    strategy_cls=SMACrossoverStrategy,
    strategy_params={"short_window": 20, "long_window": 50},
)

bt.run()
bt.print_performance()
bt.plot_tearsheet()
```

The constructor takes `data_handler_cls`, `execution_handler_cls`,
`portfolio_cls`, and `strategy_cls` — classes, not instances. Pass
`NaivePortfolio`; `Portfolio` is an abstract base and cannot be instantiated.
The previous README got both of these wrong, so the snippet it showed could not
run.

## Layout

```
backtest/
  engine.py         event loop
  events.py         Market, Signal, Order, Fill
  data.py           CSVDataHandler — ordered bar delivery
  portfolio.py      NaivePortfolio — positions, cash, order sizing
  execution.py      SimulatedExecutionHandler — slippage, commission
  performance.py    Sharpe, drawdown, CAGR, tearsheet
strategies/
  moving_average.py SMA and EMA crossover
  mean_reversion.py z-score
  momentum.py       lookback momentum
examples/
  sma_crossover.py
tests/               68 tests
KNOWN_ISSUES.md      ten defects, with severity and fixes
```

## Testing approach

The fixtures use price series whose correct answer can be worked out by hand,
so the assertions test intent rather than freezing whatever the code currently
returns.

The three that carry the most weight:

- **`test_bars_revealed_one_at_a_time`** — after *k* updates the latest bar is
  the *k*-th, never a later one. This is the look-ahead guarantee.
- **`test_a_flat_market_cannot_produce_profit`** — a constant price series run
  end to end must not finish above starting capital. The strongest available
  check that the accounting invents nothing.
- **`test_cash_plus_market_value_equals_reported_total`** — an identity that
  must hold on every single bar.

Tests that assert a *defect* say so in the docstring and point at the
`KNOWN_ISSUES.md` entry. They pass today by pinning the wrong behaviour so it
cannot change unnoticed; fixing the bug breaks the test, which is the intended
signal to delete both.

### Worth knowing about the highest-severity bug

`create_sharpe_ratio` guards against division by zero with
`if returns.std() == 0`. Exact float equality almost never holds: a constant 1%
return series has a standard deviation of `1.83e-18`, so the guard is bypassed
and the function returns a Sharpe of **8.68e16** where it intends to return
`0.0`. The same pattern appears a second time in
`NaivePortfolio.output_summary_stats`.

I predicted a different failure mode before running the test, and the real one
was worse. That is the argument for the test suite, not for my judgement.

## Data format

Yahoo Finance CSV export. Column names are lowercased on load, and `adj close`
is the field used for both signals and fills.

```
Date,Open,High,Low,Close,Volume,Adj Close
2018-01-02,170.16,172.30,169.26,172.23,25555934,172.23
```

Rows are sorted by date on load, so a reverse-ordered file streams correctly.
`start_date` and `end_date` are both inclusive.

## Requirements

Python 3.11+, NumPy ≥ 1.24, Pandas ≥ 2.0, Matplotlib ≥ 3.7, SciPy ≥ 1.10.
CI runs the suite on 3.11 and 3.12.

## License

MIT. See [LICENSE](LICENSE).
