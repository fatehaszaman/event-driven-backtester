"""
engine.py
---------
Backtest runner — ties all components together.

The engine drives the event loop:
  1. Fetch next MarketEvent from the data feed
  2. Mark portfolio to market
  3. Pass event to strategy → collect SignalEvents
  4. Convert signals to OrderEvents via position sizer
  5. Fill orders via fill simulator → FillEvents
  6. Update portfolio on fills
  7. Repeat until data is exhausted

Timestamp alignment
-------------------
Signals generated on bar N are filled on bar N+1 (next-bar execution).
This prevents look-ahead bias — you can't trade on the close of the bar
you used to generate the signal.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .data_feed import HistoricalDataFeed
from .events import (
    MarketEvent, SignalEvent, OrderEvent, FillEvent,
    OrderSide, OrderType, FillStatus,
)
from .fill_simulator import FillSimulator, FillSimulatorConfig
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .analytics import compute_metrics, print_report


class PositionSizer:
    """
    Converts signal strength to order quantity.

    Uses a fixed fractional approach: each signal allocates a fixed
    percentage of current equity to the position.

    Parameters
    ----------
    equity_fraction : float
        Fraction of current equity to allocate per signal. Default 0.10.
    max_position_value : float, optional
        Hard cap on single position value in currency units.
    """

    def __init__(
        self,
        equity_fraction: float = 0.10,
        max_position_value: Optional[float] = None,
    ):
        self.equity_fraction = equity_fraction
        self.max_position_value = max_position_value

    def size(
        self,
        signal: SignalEvent,
        current_price: float,
        current_equity: float,
    ) -> float:
        """Return order quantity for a signal."""
        alloc = current_equity * self.equity_fraction * signal.strength
        if self.max_position_value:
            alloc = min(alloc, self.max_position_value)
        return alloc / current_price if current_price > 0 else 0.0


class BacktestEngine:
    """
    Drives the backtest event loop.

    Parameters
    ----------
    feed : HistoricalDataFeed
    strategy : BaseStrategy
    portfolio : Portfolio
    fill_simulator : FillSimulator, optional
    position_sizer : PositionSizer, optional
    verbose : bool
        Print progress every N bars.
    """

    def __init__(
        self,
        feed: HistoricalDataFeed,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        fill_simulator: Optional[FillSimulator] = None,
        position_sizer: Optional[PositionSizer] = None,
        verbose: bool = False,
    ):
        self.feed = feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.fill_simulator = fill_simulator or FillSimulator()
        self.position_sizer = position_sizer or PositionSizer()
        self.verbose = verbose

        self.strategy.attach_feed(feed)
        self._pending_orders: list[OrderEvent] = []

    def _signals_to_orders(
        self,
        signals: list[SignalEvent],
        bars: dict,
    ) -> list[OrderEvent]:
        """Convert signals to orders using the position sizer."""
        orders = []
        equity = self.portfolio.cash + sum(
            pos.unrealized_pnl(bars[sym].close)
            for sym, pos in self.portfolio.positions.items()
            if sym in bars and not pos.is_flat()
        )

        for sig in signals:
            if sig.symbol not in bars:
                continue
            price = bars[sig.symbol].mid
            qty = self.position_sizer.size(sig, price, equity)
            if qty <= 0:
                continue

            # Close opposite position first
            existing = self.portfolio.get_position(sig.symbol)
            if not existing.is_flat():
                opposite = (
                    existing.quantity < 0 and sig.side == OrderSide.BUY
                ) or (
                    existing.quantity > 0 and sig.side == OrderSide.SELL
                )
                if opposite:
                    close_qty = abs(existing.quantity)
                    orders.append(OrderEvent(
                        timestamp=sig.timestamp,
                        symbol=sig.symbol,
                        side=sig.side,
                        order_type=OrderType.MARKET,
                        quantity=close_qty,
                        order_id=str(uuid.uuid4())[:8],
                    ))
                elif (existing.quantity > 0 and sig.side == OrderSide.BUY) or \
                     (existing.quantity < 0 and sig.side == OrderSide.SELL):
                    # Already in position same direction — skip
                    continue

            orders.append(OrderEvent(
                timestamp=sig.timestamp,
                symbol=sig.symbol,
                side=sig.side,
                order_type=OrderType.MARKET,
                quantity=qty,
                order_id=str(uuid.uuid4())[:8],
            ))

        return orders

    def run(self) -> dict:
        """
        Run the full backtest. Returns performance metrics dict.
        """
        pending_orders: list[OrderEvent] = []
        bar_count = 0

        for market_event in self.feed:
            bars = market_event.bars
            if not bars:
                continue

            # ── Fill pending orders from previous bar (next-bar execution) ──
            fills: list[FillEvent] = []
            remaining_orders = []
            for order in pending_orders:
                if order.symbol in bars:
                    fill = self.fill_simulator.execute(order, bars[order.symbol])
                    fills.append(fill)
                    self.portfolio.on_fill(fill)
                else:
                    remaining_orders.append(order)
            pending_orders = remaining_orders

            # ── Mark to market ───────────────────────────────────────────────
            prices = {sym: bar.close for sym, bar in bars.items()}
            snap = self.portfolio.mark_to_market(market_event.timestamp, prices)

            # ── Strategy on_bar ──────────────────────────────────────────────
            self.strategy.on_bar(market_event)
            signals = self.strategy.pop_signals()

            # ── Convert signals → orders (filled next bar) ───────────────────
            new_orders = self._signals_to_orders(signals, bars)
            pending_orders.extend(new_orders)

            bar_count += 1
            if self.verbose and bar_count % 100 == 0:
                print(f"  Bar {bar_count}/{len(self.feed)} | "
                      f"Equity: {snap.equity:,.2f}")

        # ── Performance metrics ───────────────────────────────────────────────
        metrics = compute_metrics(
            self.portfolio.equity_curve(),
            self.portfolio.trade_history(),
        )
        return metrics

    def run_and_report(self, label: str = "Backtest Results") -> dict:
        metrics = self.run()
        print_report(metrics, label)
        return metrics
