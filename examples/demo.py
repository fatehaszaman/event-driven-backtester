"""
demo.py
-------
End-to-end demonstration of the backtesting engine.

Runs a mean-reversion + volatility regime strategy across three
synthetic instruments over a two-year period, then prints a full
performance report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest import (
    generate_synthetic_data,
    HistoricalDataFeed,
    FillSimulator, FillSimulatorConfig,
    Portfolio,
    BacktestEngine, PositionSizer,
)
from strategies.mean_reversion import MeanReversionStrategy

# ── 1. Generate synthetic data ────────────────────────────────────────────────

symbols = ["INST_A", "INST_B", "INST_C"]
data = generate_synthetic_data(
    symbols=symbols,
    n_bars=504,         # ~2 years of daily bars
    start="2022-01-03",
    seed=42,
)

print(f"Data: {len(data)} bars across {len(symbols)} instruments")
print(f"Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
print()

# ── 2. Set up components ──────────────────────────────────────────────────────

feed = HistoricalDataFeed(data, spread_pct=0.0005)

strategy = MeanReversionStrategy(
    symbols=symbols,
    lookback=20,
    entry_zscore=1.5,
    exit_zscore=0.25,
    vol_regime_threshold=1.5,
)

portfolio = Portfolio(initial_capital=100_000)

fill_config = FillSimulatorConfig(
    commission_pct=0.001,           # 10bps
    market_impact_coeff=0.1,
    max_volume_pct=0.05,            # Max 5% of bar volume per order
    max_spread_pct=0.005,           # Reject if spread > 50bps
    allow_partial_fills=True,
)
fill_sim = FillSimulator(fill_config)

sizer = PositionSizer(equity_fraction=0.10)

engine = BacktestEngine(
    feed=feed,
    strategy=strategy,
    portfolio=portfolio,
    fill_simulator=fill_sim,
    position_sizer=sizer,
    verbose=True,
)

# ── 3. Run backtest ───────────────────────────────────────────────────────────

metrics = engine.run_and_report("Mean-Reversion + Vol Regime | 3 Instruments | 2Y")

# ── 4. Trade summary ──────────────────────────────────────────────────────────

trades = portfolio.trade_history()
if not trades.empty:
    print(f"Trade breakdown by instrument:")
    print(trades.groupby("symbol")["quantity"].count().rename("n_fills").to_string())
    print()
    print(f"Fill status breakdown:")
    print(trades.groupby("status")["quantity"].count().to_string())
    print()

# ── 5. Equity curve sample ────────────────────────────────────────────────────

curve = portfolio.equity_curve()
if not curve.empty:
    print("Equity curve (every 50 bars):")
    sample = curve.iloc[::50][["timestamp", "equity", "net_pnl"]].copy()
    sample["equity"] = sample["equity"].map("{:,.2f}".format)
    sample["net_pnl"] = sample["net_pnl"].map("{:,.2f}".format)
    print(sample.to_string(index=False))
