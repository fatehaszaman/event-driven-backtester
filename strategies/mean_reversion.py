"""
mean_reversion.py
-----------------
Mean-reversion strategy with volatility regime gating.

Signal logic
------------
1. Compute a z-score of the current price relative to a rolling fair-value
   band (rolling mean ± rolling std).
2. Gate signals through a volatility regime classifier:
   - LOW vol regime: full signal strength — mean-reversion is more reliable
   - HIGH vol regime: signals suppressed — trending / fat-tail conditions
     make mean-reversion dangerous
3. Entry: z-score crosses threshold in the direction of reversion
4. Exit: z-score reverts to within the exit band

Volatility regime
-----------------
Regime is determined by comparing realized volatility (rolling std of
returns) to its own rolling average. When vol is above its average by
more than vol_regime_threshold, we're in a high-vol regime and mean-
reversion signals are suppressed.

This gating is important: mean-reversion strategies without vol-regime
filters tend to buy falling knives in trending or crisis conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Deque

from backtest.strategy import BaseStrategy
from backtest.events import MarketEvent, OrderSide


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy with volatility regime gating.

    Parameters
    ----------
    symbols : list[str]
    lookback : int
        Rolling window for fair-value band. Default 20 bars.
    entry_zscore : float
        Z-score threshold to enter a trade. Default 1.5.
    exit_zscore : float
        Z-score threshold to exit. Default 0.25.
    vol_lookback : int
        Rolling window for volatility regime. Default 20 bars.
    vol_regime_threshold : float
        Vol ratio above which high-vol regime is declared (suppress signals).
        Default 1.5 — vol must be 50% above its average to suppress.
    min_bars : int
        Minimum bars required before generating signals. Default = lookback.
    """

    def __init__(
        self,
        symbols: list[str],
        lookback: int = 20,
        entry_zscore: float = 1.5,
        exit_zscore: float = 0.25,
        vol_lookback: int = 20,
        vol_regime_threshold: float = 1.5,
        min_bars: int | None = None,
        strategy_id: str = "MeanReversion",
    ):
        super().__init__(symbols, strategy_id)
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.vol_lookback = vol_lookback
        self.vol_regime_threshold = vol_regime_threshold
        self.min_bars = min_bars if min_bars is not None else lookback

        # Per-symbol price and return history
        self._prices: dict[str, Deque[float]] = {
            s: deque(maxlen=max(lookback, vol_lookback) + 1) for s in symbols
        }
        self._returns: dict[str, Deque[float]] = {
            s: deque(maxlen=vol_lookback + 1) for s in symbols
        }
        self._bar_counts: dict[str, int] = {s: 0 for s in symbols}
        self._positions: dict[str, int] = {s: 0 for s in symbols}  # -1, 0, 1

    def _zscore(self, prices: Deque[float]) -> float:
        arr = np.array(prices)
        mu = arr.mean()
        std = arr.std()
        if std < 1e-10:
            return 0.0
        return (arr[-1] - mu) / std

    def _is_high_vol_regime(self, returns: Deque[float]) -> bool:
        if len(returns) < self.vol_lookback:
            return False
        arr = np.array(returns)
        realized_vol = arr[-self.vol_lookback:].std()
        avg_vol = arr.std()
        if avg_vol < 1e-10:
            return False
        return realized_vol / avg_vol > self.vol_regime_threshold

    def on_bar(self, event: MarketEvent) -> None:
        ts = event.timestamp

        for symbol in self.symbols:
            if symbol not in event.bars:
                continue

            bar = event.bars[symbol]
            prices = self._prices[symbol]
            rets = self._returns[symbol]

            # Update history
            if len(prices) > 0:
                ret = (bar.close - prices[-1]) / prices[-1] if prices[-1] > 0 else 0.0
                rets.append(ret)
            prices.append(bar.close)
            self._bar_counts[symbol] += 1

            if self._bar_counts[symbol] < self.min_bars:
                continue

            # Compute signals
            z = self._zscore(prices)
            high_vol = self._is_high_vol_regime(rets)
            current_pos = self._positions[symbol]

            # Suppress entries in high-vol regime
            if high_vol and current_pos == 0:
                continue

            # Entry logic
            if current_pos == 0:
                if z <= -self.entry_zscore:
                    # Price well below fair value → long
                    strength = min(1.0, abs(z) / (self.entry_zscore * 2))
                    self.signal(ts, symbol, OrderSide.BUY, strength=strength,
                                meta={"zscore": round(z, 3), "high_vol": high_vol})
                    self._positions[symbol] = 1

                elif z >= self.entry_zscore:
                    # Price well above fair value → short
                    strength = min(1.0, abs(z) / (self.entry_zscore * 2))
                    self.signal(ts, symbol, OrderSide.SELL, strength=strength,
                                meta={"zscore": round(z, 3), "high_vol": high_vol})
                    self._positions[symbol] = -1

            # Exit logic
            elif current_pos == 1 and z >= -self.exit_zscore:
                self.signal(ts, symbol, OrderSide.SELL, strength=1.0,
                            meta={"zscore": round(z, 3), "exit": True})
                self._positions[symbol] = 0

            elif current_pos == -1 and z <= self.exit_zscore:
                self.signal(ts, symbol, OrderSide.BUY, strength=1.0,
                            meta={"zscore": round(z, 3), "exit": True})
                self._positions[symbol] = 0
