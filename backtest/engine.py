import queue
import time
import pandas as pd

from backtest.events import EventType
from backtest.performance import plot_tearsheet


class Backtest:
    """
    The main engine. Wires together the data handler, strategy,
    portfolio, and execution handler, then runs the event loop.
    """

    def __init__(
        self,
        csv_dir,
        symbol_list,
        initial_capital,
        heartbeat,
        start_date,
        end_date,
        data_handler_cls,
        execution_handler_cls,
        portfolio_cls,
        strategy_cls,
        strategy_params=None,
    ):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_params = strategy_params or {}
        self.events = queue.Queue()
        self.num_events = 0

        self.data_handler = data_handler_cls(
            self.events, csv_dir, symbol_list, start_date, end_date
        )
        self.portfolio = portfolio_cls(
            self.data_handler, self.events, start_date, initial_capital
        )
        self.execution_handler = execution_handler_cls(
            self.events, self.data_handler
        )
        self.strategy = strategy_cls(
            self.data_handler, self.events, **self.strategy_params
        )

    def _run_loop(self):
        while True:
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break

            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break

                self.num_events += 1

                if event.type == EventType.MARKET:
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                elif event.type == EventType.SIGNAL:
                    self.portfolio.update_signal(event)
                elif event.type == EventType.ORDER:
                    self.execution_handler.execute_order(event)
                elif event.type == EventType.FILL:
                    self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def run(self):
        self._run_loop()

    def print_performance(self):
        stats = self.portfolio.output_summary_stats()
        print("\n=== Backtest Results ===")
        for label, value in stats:
            print(f"  {label}: {value}")
        print(f"  Events processed: {self.num_events}")

    def plot_tearsheet(self, title="Strategy Tearsheet"):
        holdings_df = pd.DataFrame(self.portfolio.all_holdings)
        holdings_df.set_index("datetime", inplace=True)
        plot_tearsheet(holdings_df, self.initial_capital, title=title)
