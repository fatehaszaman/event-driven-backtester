"""
Example: SMA Crossover on AAPL

Runs a 20/50 SMA crossover strategy on AAPL from 2018 to 2023.
Place AAPL.csv in the ./data directory before running.
CSV format should match Yahoo Finance exports.
"""

from backtest.engine import Backtest
from backtest.data import CSVDataHandler
from backtest.portfolio import NaivePortfolio
from backtest.execution import SimulatedExecutionHandler
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
bt.plot_tearsheet(title="AAPL SMA 20/50 Crossover")
