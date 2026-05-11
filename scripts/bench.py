"""
bench.py
--------
Reproduces the performance table in README.md.

Runs the full mean-reversion + vol-regime strategy end-to-end at three
problem sizes and prints wall-time, throughput, and peak RSS for each.
Pure Python on NumPy and pandas, single core, no external services.
"""

from __future__ import annotations

import os
import resource
import sys
import time

# Allow running from repo root without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_feed import HistoricalDataFeed, generate_synthetic_data
from backtest.engine import BacktestEngine
from backtest.portfolio import Portfolio
from strategies.mean_reversion import MeanReversionStrategy


CONFIGS = [
    (500, 3),
    (2000, 5),
    (5000, 10),
]


def run_one(n_bars: int, n_syms: int) -> dict:
    data = generate_synthetic_data(
        [f"S{i}" for i in range(n_syms)], n_bars=n_bars, seed=42
    )
    feed = HistoricalDataFeed(data)
    strategy = MeanReversionStrategy(symbols=feed.symbols)
    portfolio = Portfolio(initial_capital=100_000)
    engine = BacktestEngine(feed, strategy, portfolio, verbose=False)

    t0 = time.perf_counter()
    engine.run()
    dt = time.perf_counter() - t0

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "bars": n_bars,
        "syms": n_syms,
        "seconds": dt,
        "events_per_s": n_bars / dt,
        "sym_bars_per_s": n_bars * n_syms / dt,
        "peak_rss_mib": rss_kib / 1024,
    }


def main() -> None:
    header = f"{'bars':>6} | {'syms':>4} | {'time':>8} | {'events/s':>10} | {'sym-bars/s':>12} | {'peak RSS':>10}"
    print(header)
    print("-" * len(header))
    for n_bars, n_syms in CONFIGS:
        r = run_one(n_bars, n_syms)
        print(
            f"{r['bars']:>6} | {r['syms']:>4} | {r['seconds']:>7.2f}s | "
            f"{r['events_per_s']:>10,.0f} | {r['sym_bars_per_s']:>12,.0f} | "
            f"{r['peak_rss_mib']:>7.1f} MiB"
        )


if __name__ == "__main__":
    main()
