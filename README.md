# Event-Driven Backtester

A production-grade, discrete-event simulation backtester written in Python. Built for realistic strategy evaluation with bid/ask spread modeling, market impact, partial fills, and zero look-ahead bias.

## Features

- **Discrete-event architecture**: clean separation of data, strategy, portfolio, and execution layers
- **Realistic fill simulation**: bid/ask spreads, slippage, partial fills, and commission modeling
- **Market impact modeling**: volume-weighted price impact for large orders
- **Zero look-ahead bias**: strict event ordering with no future data leaking into signals
- **Multiple data sources**: CSV, Yahoo Finance, and an extensible handler interface
- **Strategy framework**: plug-and-play strategy interface with full portfolio state access
- **Risk metrics**: Sharpe ratio, max drawdown, CAGR, Calmar ratio, and more
- **Tearsheet generation**: performance tearsheets with equity curves and drawdown plots

## Architecture

```
event-driven-backtester/
├── backtest/
│   ├── __init__.py
│   ├── engine.py          # Core event loop and backtesting engine
│   ├── events.py          # Event class hierarchy (Market, Signal, Order, Fill)
│   ├── data.py            # Data handlers (CSV, Yahoo Finance)
│   ├── portfolio.py       # Portfolio management and position tracking
│   ├── execution.py       # Execution handler with slippage/commission models
│   └── performance.py     # Performance metrics and tearsheet generation
├── strategies/
│   ├── __init__.py
│   ├── base.py            # Abstract strategy base class
│   ├── moving_average.py  # SMA/EMA crossover strategies
│   ├── mean_reversion.py  # Z-score mean reversion strategy
│   └── momentum.py        # Momentum/trend-following strategy
├── examples/
│   ├── sma_crossover.py   # Simple moving average crossover example
│   ├── mean_reversion.py  # Pairs trading mean reversion example
│   └── momentum.py        # Momentum strategy example
├── scripts/
│   └── run_backtest.py    # CLI entrypoint for running backtests
├── tests/
│   ├── test_events.py
│   ├── test_portfolio.py
│   └── test_execution.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
git clone https://github.com/fatehaszaman/event-driven-backtester.git
cd event-driven-backtester
pip install -r requirements.txt
```

## Quick Start

```python
from backtest.engine import Backtest
from backtest.data import CSVDataHandler
from backtest.portfolio import Portfolio
from backtest.execution import SimulatedExecutionHandler
from strategies.moving_average import SMACrossoverStrategy

bt = Backtest(
    csv_dir="./data",
    symbol_list=["AAPL"],
    initial_capital=100_000,
    heartbeat=0.0,
    start_date="2018-01-01",
    end_date="2023-12-31",
    data_handler=CSVDataHandler,
    execution_handler=SimulatedExecutionHandler,
    portfolio=Portfolio,
    strategy=SMACrossoverStrategy,
    strategy_params={"short_window": 20, "long_window": 50},
)

bt.run()
bt.print_performance()
bt.plot_tearsheet()
```

## Strategies

### SMA Crossover
Generates buy signals when the short-term moving average crosses above the long-term, and sell signals on the reverse.

### Mean Reversion
Z-score based mean reversion. Enters long when the z-score drops below `-entry_z` and exits when it reverts toward zero.

### Momentum
Rank-based momentum strategy that goes long the top decile of performers over a lookback window.

## Performance Metrics

| Metric | Description |
|---|---|
| Total Return | Cumulative portfolio return over the backtest period |
| Sharpe Ratio | Annualized risk-adjusted return (252 trading days) |
| Max Drawdown | Largest peak-to-trough decline in portfolio value |
| CAGR | Compound annual growth rate |
| Calmar Ratio | CAGR divided by maximum drawdown |
| Win Rate | Percentage of profitable trades |

## Configuration

All backtest parameters are passed to the `Backtest` constructor. Key parameters:

| Parameter | Type | Description |
|---|---|---|
| `csv_dir` | str | Path to directory containing OHLCV CSV files |
| `symbol_list` | list | List of ticker symbols to trade |
| `initial_capital` | float | Starting portfolio value in USD |
| `start_date` | str | Backtest start date (YYYY-MM-DD) |
| `end_date` | str | Backtest end date (YYYY-MM-DD) |
| `strategy_params` | dict | Strategy-specific hyperparameters |

## Data Format

CSV files should follow Yahoo Finance export format:

```
Date,Open,High,Low,Close,Volume,Adj Close
2018-01-02,170.16,172.30,169.26,172.23,25555934,172.23
```

## Requirements

- Python 3.9+
- NumPy >= 1.24
- Pandas >= 2.0
- Matplotlib >= 3.7
- SciPy >= 1.10

## License

MIT License. See [LICENSE](LICENSE) for details.
