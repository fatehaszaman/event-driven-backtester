"""Deterministic fixtures. Every price series here is chosen so the correct
answer can be computed by hand, which is what makes the assertions meaningful
rather than snapshots of current behaviour."""
import queue

import pandas as pd
import pytest

from backtest.data import CSVDataHandler


def write_csv(path, dates, closes):
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.to_datetime(dates),
    )
    df.index.name = "Date"
    df.to_csv(path)
    return df


@pytest.fixture
def events():
    return queue.Queue()


@pytest.fixture
def flat_prices(tmp_path, events):
    """Ten bars, price constant at 100. A strategy must not make money here."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    write_csv(tmp_path / "FLAT.csv", dates, [100.0] * 10)
    return CSVDataHandler(events, str(tmp_path), ["FLAT"])


@pytest.fixture
def ramp_prices(tmp_path, events):
    """Prices 100..109, one per bar. Monotone, so P&L signs are unambiguous."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    write_csv(tmp_path / "RAMP.csv", dates, [100.0 + i for i in range(10)])
    return CSVDataHandler(events, str(tmp_path), ["RAMP"])


@pytest.fixture
def csv_dir_ramp(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60)
    write_csv(tmp_path / "RAMP.csv", dates, [100.0 + i for i in range(60)])
    return str(tmp_path)
