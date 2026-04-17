"""
data_feed.py
------------
Data feed: generates MarketEvents from OHLCV data.

The feed iterates over a DataFrame of bars in timestamp order,
aligning multiple symbols so strategies always see a consistent
cross-sectional snapshot. Timestamps are aligned — if one symbol
has no bar at a given timestamp, the last known bar is carried forward
(last-observation-carried-forward / LOCF).

Supports both historical DataFrames (backtesting) and can be
subclassed for live data sources.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterator, Optional
from .events import Bar, MarketEvent


class HistoricalDataFeed:
    """
    Generates MarketEvents from a historical OHLCV DataFrame.

    Expected DataFrame columns (per symbol):
        timestamp, symbol, open, high, low, close, volume

    Bid/ask spread is synthesized from a configurable spread model
    if not present in the source data (common for daily/hourly bars).

    Parameters
    ----------
    data : pd.DataFrame
        Must have columns: timestamp, symbol, open, high, low, close, volume.
        Additional bid/ask columns are used if present.
    spread_pct : float
        Synthetic spread as a percentage of close price.
        Used when bid/ask not present in source data. Default 0.05%.
    symbols : list[str], optional
        Subset of symbols to include. All symbols used if None.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        spread_pct: float = 0.0005,
        symbols: Optional[list[str]] = None,
    ):
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        if symbols:
            data = data[data["symbol"].isin(symbols)]

        # Synthesize bid/ask if not present
        if "bid" not in data.columns:
            data["bid"] = data["close"] * (1 - spread_pct / 2)
        if "ask" not in data.columns:
            data["ask"] = data["close"] * (1 + spread_pct / 2)

        self._data = data
        self._symbols = data["symbol"].unique().tolist()
        self._timestamps = sorted(data["timestamp"].unique())
        self._current_idx = 0
        self._latest_bars: dict[str, Bar] = {}

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def timestamps(self) -> list[pd.Timestamp]:
        return self._timestamps

    def __len__(self) -> int:
        return len(self._timestamps)

    def __iter__(self) -> Iterator[MarketEvent]:
        self._current_idx = 0
        self._latest_bars = {}
        return self

    def __next__(self) -> MarketEvent:
        if self._current_idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._current_idx]
        self._current_idx += 1

        slice_ = self._data[self._data["timestamp"] == ts]
        event_bars: dict[str, Bar] = {}

        for _, row in slice_.iterrows():
            bar = Bar(
                timestamp=row["timestamp"],
                symbol=row["symbol"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                bid=row["bid"],
                ask=row["ask"],
            )
            self._latest_bars[row["symbol"]] = bar
            event_bars[row["symbol"]] = bar

        # Carry forward any symbols not in this bar
        for sym in self._symbols:
            if sym not in event_bars and sym in self._latest_bars:
                event_bars[sym] = self._latest_bars[sym]

        return MarketEvent(bars=event_bars)

    def latest_bar(self, symbol: str) -> Optional[Bar]:
        return self._latest_bars.get(symbol)

    def history(self, symbol: str, n: int) -> pd.DataFrame:
        """Return last n bars for a symbol as a DataFrame."""
        sym_data = self._data[self._data["symbol"] == symbol]
        idx = self._current_idx
        start = max(0, idx - n)
        return sym_data.iloc[start:idx].reset_index(drop=True)


def generate_synthetic_data(
    symbols: list[str],
    n_bars: int = 500,
    start: str = "2023-01-02",
    freq: str = "B",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing and demonstration.

    Uses geometric Brownian motion for price paths with realistic
    spread and volume profiles.

    Parameters
    ----------
    symbols : list[str]
    n_bars : int
    start : str
        Start date string.
    freq : str
        Pandas frequency string. Default 'B' (business days).
    seed : int
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars, freq=freq)
    rows = []

    for symbol in symbols:
        # GBM price path
        mu = rng.uniform(0.00005, 0.0002)
        sigma = rng.uniform(0.008, 0.018)
        s0 = rng.uniform(50, 500)

        returns = rng.normal(mu, sigma, n_bars)
        closes = s0 * np.exp(np.cumsum(returns))

        # Realistic OHLCV from close
        daily_range_pct = np.abs(rng.normal(0, sigma * 1.5, n_bars))
        highs = closes * (1 + daily_range_pct)
        lows  = closes * (1 - daily_range_pct)
        opens = np.roll(closes, 1)
        opens[0] = s0

        volumes = rng.integers(100_000, 2_000_000, n_bars).astype(float)
        spread_pct = rng.uniform(0.0003, 0.0008)

        for i, date in enumerate(dates):
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "open": round(opens[i], 4),
                "high": round(highs[i], 4),
                "low": round(lows[i], 4),
                "close": round(closes[i], 4),
                "volume": volumes[i],
                "bid": round(closes[i] * (1 - spread_pct / 2), 4),
                "ask": round(closes[i] * (1 + spread_pct / 2), 4),
            })

    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
