# Event-Driven Backtester

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

A discrete-event backtesting engine in Python. Market, signal, order, and fill
events flow through a queue, which keeps the data, strategy, portfolio, and
execution layers genuinely decoupled rather than decoupled in description only.

90 tests. `KNOWN_ISSUES.md` lists the defects found by writing them, with
severity and a fix for each: eight open, two fixed.

## What is actually implemented

| | |
|---|---|
| Discrete-event loop over a shared queue | yes |
| CSV data handler, bars streamed strictly in date order | yes |
| SMA and EMA crossover, z-score mean reversion, momentum | yes |
| Long and short positions, per-fill cash and position accounting | yes |
| Fixed proportional slippage, applied against the trade direction | yes |
| Tiered per-share commission with a notional cap | yes |
| Orders fill at the next bar's open, with one bar of latency | yes |
| Sharpe, max drawdown, CAGR, equity and drawdown tearsheet | yes |
| Mean-variance optimizer with a turnover penalty (standalone module) | yes |

## What is not implemented

An earlier version of this README claimed several of these. It was wrong, and
the list is kept explicit so nobody has to read the source to find out.

| | |
|---|---|
| Bid/ask spread modelling | no - slippage is a single proportional factor |
| Market impact | no - `execution.py` says so directly |
| Partial fills | no - every order fills in full |
| Yahoo Finance or any non-CSV source | no - `CSVDataHandler` only |
| Calmar ratio, win rate | no |
| Optimizer-driven rebalancing inside the event loop | no - `optimizer.py` is not called by the engine |

## Honest accounting of the fill assumption

The data handler does guarantee ordered delivery: after *k* calls to
`update_bars()`, exactly *k* bars are visible and no later bar can be read.
That property is tested directly.

**Fills now occur at the next bar's open; earlier results used the same-bar
close.** A signal computed from bar *t*'s close used to be filled at that same
close, a price that was not tradable at the moment it was observed. An order
created on bar *t* is now held until bar *t+1* arrives and is filled at bar
*t+1*'s open, plus slippage. The open is scaled by `adj close / close` so fills
and position marks use the same adjusted basis. An order placed on the final
bar has no next bar; it is not filled, and a warning is logged. This was
`KNOWN_ISSUES.md` #2.

Re-running the four strategies on AAPL daily bars from 2018 to 2023 (Yahoo
Finance, 100,000 starting capital, default parameters) shows the change is not
a uniform haircut:

| Strategy | Total return, same-bar close | Total return, next-bar open | Annualized Sharpe, before | Annualized Sharpe, after |
|---|---|---|---|---|
| EMA crossover | 9.54% | 9.88% | 0.69 | 0.71 |
| SMA crossover | 8.32% | 8.32% | 0.59 | 0.59 |
| Momentum | 7.99% | 8.82% | 0.50 | 0.56 |
| Mean reversion | 3.14% | 1.88% | 0.33 | 0.20 |

Total return is cumulative over the full six years (final equity divided by
starting capital, minus one); it is not annualized and is not CAGR. Sharpe is
annualized from daily returns with 252 periods and a zero risk-free rate
(`KNOWN_ISSUES.md` #6).

Delayed entries and delayed exits both move, so the sign of the change depends
on the strategy and on overnight gaps. The earlier results were not tradable as
stated, and figures from before this fix should not be compared with later ones.

## Portfolio construction

`backtest/optimizer.py` chooses weights for n assets given expected returns
mu, a covariance matrix Sigma and the current weights w_prev:

```
maximize    w'mu - lam * w'Sigma w - kappa * sum_i |w_i - w_prev_i|
subject to  sum_i w_i = 1,   lower_i <= w_i <= upper_i
```

- `w'mu` is expected return, `lam` is risk aversion and `w'Sigma w` is
  portfolio variance.
- `kappa` is a proportional cost per unit of turnover, so a trade is made only
  when its improvement in risk-adjusted return exceeds its cost.
- The absolute value is made smooth by splitting each trade into buys and
  sells, `w - w_prev = u - v` with `u, v >= 0`, which turns the problem into a
  convex quadratic program solved with SciPy's SLSQP.

With no turnover cost and no bounds the optimum has a closed form:

```
w* = Sigma^-1 mu / (2 lam) + (1 - 1'Sigma^-1 mu / (2 lam)) * Sigma^-1 1 / (1'Sigma^-1 1)
```

That is the minimum-variance portfolio plus a tilt toward `Sigma^-1 mu` that
shrinks as `lam` rises. Rescaling `Sigma^-1 mu` to sum to one is a different
portfolio (the tangency portfolio at a zero risk-free rate), which does not
depend on `lam`.

`ewma_covariance` gives an optional RiskMetrics-style estimate with decay
0.94, using only the returns passed in, so it is point-in-time when the caller
supplies data up to the decision date.

Tests check the solver against the closed form, the budget and bound
constraints, and three comparative statics: raising `kappa` never increases
turnover, raising `lam` never increases variance, and raising one asset's
expected return never lowers its weight. Test inputs are synthetic. The module
is standalone; the backtest engine does not call it yet.

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
`portfolio_cls`, and `strategy_cls` - classes, not instances. Pass
`NaivePortfolio`; `Portfolio` is an abstract base and cannot be instantiated.
The previous README got both of these wrong, so the snippet it showed could not
run.

## Layout

```
backtest/
  engine.py         event loop
  events.py         Market, Signal, Order, Fill
  data.py           CSVDataHandler - ordered bar delivery
  portfolio.py      NaivePortfolio - positions, cash, order sizing
  execution.py      SimulatedExecutionHandler - slippage, commission
  performance.py    Sharpe, drawdown, CAGR, tearsheet
  optimizer.py      mean-variance weights with a turnover penalty, EWMA covariance
strategies/
  moving_average.py SMA and EMA crossover
  mean_reversion.py z-score
  momentum.py       lookback momentum
examples/
  sma_crossover.py
tests/               90 tests
KNOWN_ISSUES.md      eight open defects and two fixed, with severity and fixes
```

## Testing approach

The fixtures use price series whose correct answer can be worked out by hand,
so the assertions test intent rather than freezing whatever the code currently
returns.

The three that carry the most weight:

- **`test_bars_revealed_one_at_a_time`** - after *k* updates the latest bar is
  the *k*-th, never a later one. This is the look-ahead guarantee.
- **`test_a_flat_market_cannot_produce_profit`** - a constant price series run
  end to end must not finish above starting capital. The strongest available
  check that the accounting invents nothing.
- **`test_cash_plus_market_value_equals_reported_total`** - an identity that
  must hold on every single bar.

Tests that assert a *defect* say so in the docstring and point at the
`KNOWN_ISSUES.md` entry. They pass today by pinning the wrong behaviour so it
cannot change unnoticed; fixing the bug breaks the test, which is the intended
signal to delete both.

### Worth knowing about the highest-severity bug (fixed in #8)

`create_sharpe_ratio` originally guarded against division by zero with
`if returns.std() == 0`. Exact float equality almost never holds: a constant 1%
return series has a standard deviation of `1.83e-18`, so the guard was bypassed
and the function returned a Sharpe of **8.68e16** where it intended to return
`0.0`. The same pattern appeared a second time in
`NaivePortfolio.output_summary_stats`. Both now route through a shared `1e-12`
near-zero volatility tolerance (`KNOWN_ISSUES.md` #8).

I predicted a different failure mode before running the test, and the real one
was worse. That is the argument for the test suite, not for my judgement. Along
with the remaining open entries in `KNOWN_ISSUES.md`, these pinned defects and
seeded runs also serve as ground-truth test cases for my own LLM-as-a-judge
code and quantitative review experiments.

## Data format

Yahoo Finance CSV export. Column names are lowercased on load. `adj close` is
used for signals and position marks; fills use `open` scaled to the adjusted
basis.

Price data is not included in this repository, and `data/` is ignored by git.
To reproduce the results above, export AAPL daily history from Yahoo Finance
to `data/AAPL.csv` yourself. Yahoo Finance data is subject to Yahoo's terms
and is used here for personal research only. The row below shows the expected
columns; its values are illustrative.

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
