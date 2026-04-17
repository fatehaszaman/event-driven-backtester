"""
portfolio.py
------------
Portfolio and position tracker.

Maintains the state of all open positions, cash balance, and trade history.
Updated on every FillEvent. Provides equity curve, position snapshots,
and P&L attribution at any point during the backtest.

Position accounting
-------------------
- Long positions: positive quantity, cost basis = average entry price
- Short positions: negative quantity
- Realized P&L: computed on close/reduce of a position
- Unrealized P&L: computed mark-to-market at current bar prices
- Total equity = cash + sum(unrealized P&L across all positions)
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .events import FillEvent, FillStatus, OrderSide, Bar


@dataclass
class Position:
    """Tracks a single instrument position."""
    symbol: str
    quantity: float = 0.0           # Positive = long, negative = short
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9

    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_entry_price)

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def update_on_fill(self, fill: FillEvent) -> None:
        """Update position state from a fill."""
        if fill.status == FillStatus.REJECTED or fill.filled_quantity == 0:
            return

        qty = fill.filled_quantity
        price = fill.fill_price
        sign = 1.0 if fill.side == OrderSide.BUY else -1.0
        delta_qty = sign * qty

        old_qty = self.quantity
        new_qty = old_qty + delta_qty

        # Realize P&L on closes / reduces
        if old_qty != 0 and (
            (old_qty > 0 and delta_qty < 0) or
            (old_qty < 0 and delta_qty > 0)
        ):
            closed_qty = min(abs(old_qty), abs(delta_qty))
            self.realized_pnl += (price - self.avg_entry_price) * closed_qty * (
                1 if old_qty > 0 else -1
            )

        # Update average entry price on adds
        if new_qty != 0:
            if old_qty == 0 or (old_qty > 0 and delta_qty > 0) or (old_qty < 0 and delta_qty < 0):
                # Opening or adding to position
                total_cost = abs(old_qty) * self.avg_entry_price + abs(delta_qty) * price
                self.avg_entry_price = total_cost / abs(new_qty)
            elif abs(new_qty) < abs(old_qty):
                # Partial close — entry price unchanged
                pass
            else:
                # Flip — new direction, reset entry price
                self.avg_entry_price = price

        self.quantity = new_qty
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage


@dataclass
class EquitySnapshot:
    """Portfolio state at a single timestamp."""
    timestamp: pd.Timestamp
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float
    total_slippage: float

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized_pnl

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.total_commission - self.total_slippage


class Portfolio:
    """
    Tracks cash, positions, and equity across a backtest.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict] = []
        self._equity_curve: list[EquitySnapshot] = []

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def on_fill(self, fill: FillEvent) -> None:
        """Process a fill and update cash + position."""
        if fill.status == FillStatus.REJECTED or fill.filled_quantity == 0:
            return

        pos = self.get_position(fill.symbol)
        pos.update_on_fill(fill)

        # Update cash
        notional = fill.filled_quantity * fill.fill_price
        if fill.side == OrderSide.BUY:
            self.cash -= notional + fill.commission
        else:
            self.cash += notional - fill.commission

        self.trade_log.append({
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.filled_quantity,
            "price": fill.fill_price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "status": fill.status.value,
        })

    def mark_to_market(
        self,
        timestamp: pd.Timestamp,
        prices: dict[str, float],
    ) -> EquitySnapshot:
        """Compute and record a portfolio snapshot at current prices."""
        unrealized = sum(
            pos.unrealized_pnl(prices[sym])
            for sym, pos in self.positions.items()
            if sym in prices and not pos.is_flat()
        )
        positions_value = sum(
            pos.market_value(prices[sym])
            for sym, pos in self.positions.items()
            if sym in prices
        )
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_commission = sum(pos.total_commission for pos in self.positions.values())
        total_slippage = sum(pos.total_slippage for pos in self.positions.values())

        snap = EquitySnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_commission=total_commission,
            total_slippage=total_slippage,
        )
        self._equity_curve.append(snap)
        return snap

    def equity_curve(self) -> pd.DataFrame:
        """Return the full equity curve as a DataFrame."""
        if not self._equity_curve:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": s.timestamp,
            "equity": s.equity,
            "cash": s.cash,
            "unrealized_pnl": s.unrealized_pnl,
            "realized_pnl": s.realized_pnl,
            "net_pnl": s.net_pnl,
            "total_commission": s.total_commission,
            "total_slippage": s.total_slippage,
        } for s in self._equity_curve])

    def trade_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

    def summary(self) -> dict:
        curve = self.equity_curve()
        if curve.empty:
            return {}
        final_equity = curve["equity"].iloc[-1]
        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round((final_equity / self.initial_capital - 1) * 100, 2),
            "total_trades": len(self.trade_log),
            "total_commission": round(curve["total_commission"].iloc[-1], 2),
            "total_slippage": round(curve["total_slippage"].iloc[-1], 2),
        }
