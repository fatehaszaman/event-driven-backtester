import os
import pandas as pd
from abc import ABC, abstractmethod
from backtest.events import MarketEvent


class DataHandler(ABC):

    @abstractmethod
    def get_latest_bar(self, symbol):
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        raise NotImplementedError

    @abstractmethod
    def update_bars(self):
        raise NotImplementedError


class CSVDataHandler(DataHandler):
    """
    Streams OHLCV bars from CSV files one at a time.
    Guarantees no look-ahead bias -- bars are revealed in order.
    """

    def __init__(self, events, csv_dir, symbol_list, start_date=None, end_date=None):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._load_csv_files()

    def _load_csv_files(self):
        for symbol in self.symbol_list:
            path = os.path.join(self.csv_dir, f"{symbol}.csv")
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            if self.start_date:
                df = df[df.index >= pd.Timestamp(self.start_date)]
            if self.end_date:
                df = df[df.index <= pd.Timestamp(self.end_date)]
            df.columns = [c.lower().strip() for c in df.columns]
            self.symbol_data[symbol] = df.iterrows()
            self.latest_symbol_data[symbol] = []

    def _get_new_bar(self, symbol):
        try:
            return next(self.symbol_data[symbol])
        except StopIteration:
            return None

    def get_latest_bar(self, symbol):
        bars = self.latest_symbol_data[symbol]
        if not bars:
            raise ValueError(f"No bars loaded for {symbol}")
        return bars[-1]

    def get_latest_bars(self, symbol, N=1):
        return self.latest_symbol_data[symbol][-N:]

    def get_latest_bar_value(self, symbol, val_type):
        return self.get_latest_bar(symbol)[1][val_type]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        return [b[1][val_type] for b in self.get_latest_bars(symbol, N)]

    def update_bars(self):
        for symbol in self.symbol_list:
            bar = self._get_new_bar(symbol)
            if bar is not None:
                self.latest_symbol_data[symbol].append(bar)
            else:
                self.continue_backtest = False
        self.events.put(MarketEvent())
