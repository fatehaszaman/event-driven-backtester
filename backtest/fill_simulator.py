"""
fill_simulator.py
-----------------
Order fill simulator with realistic execution modeling.

Naive backtesting fills orders at the close price with no friction.
This produces results that are impossible to replicate in live trading.
This simulator models the key sources of execution cost and constraint:

  1. Bid/ask spread: buys fill at ask, sells fill at bid
  2. Market impact slippage: larger orders move the price against you
  3. Partial fills: orders too large relative to available volume
     are partially filled; remainder can be carried or cancelled
  4. Size constraints: configurable max order size as % of bar volume
  5. Wide-spread avoidance: orders rejected when spread exceeds threshold

These mechanics produce fill prices and quantities that closely track
what would actually happen in a live market, making strategy performance
estimates more reliable.
"""

from __future__ import annotations

import uuid
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .events import (
    Bar, OrderEvent, FillEvent, OrderSide, OrderType, FillStatus
)


@dataclass
class FillSimulatorConfig:
    """
    Configuration for the fill simulator.

    Parameters
    ----------
    commission_pct : float
        Commission as a fraction of trade notional. Default 0.001 (10bps).
    market_impact_coeff : float
        Linear market impact coefficient. Impact (in price units) =
        coeff * (order_size / bar_volume) * close_price.
        Default 0.1 — calibrate to the instrument / venue.
    max_volume_pct : float
        Maximum order size as a fraction of bar volume.
        Orders above this are partially filled. Default 0.10 (10%).
    max_spread_pct : float
        Maximum spread (as % of mid) above which orders are rejected.
        Models wide-spread / thin-book avoidance. Default 0.5%.
    allow_partial_fills : bool
        If True, orders exceeding max_volume_pct are partially filled.
        If False, they are rejected entirely. Default True.
    """
    commission_pct: float = 0.001
    market_impact_coeff: float = 0.1
    max_volume_pct: float = 0.10
    max_spread_pct: float = 0.005
    allow_partial_fills: bool = True


class FillSimulator:
    """
    Simulates order execution against a bar.

    Parameters
    ----------
    config : FillSimulatorConfig
    """

    def __init__(self, config: Optional[FillSimulatorConfig] = None):
        self.config = config or FillSimulatorConfig()

    def execute(self, order: OrderEvent, bar: Bar) -> FillEvent:
        """
        Attempt to fill an order against the given bar.

        Execution logic
        ---------------
        1. Reject if spread exceeds max_spread_pct (thin book / wide spread).
        2. Determine available fill quantity (capped at max_volume_pct of bar volume).
        3. Compute base fill price: buys at ask, sells at bid.
        4. Apply market impact: price moves against order direction proportional
           to order size relative to bar volume.
        5. Compute commission on filled notional.
        6. Return FillEvent with status FULL, PARTIAL, or REJECTED.

        Parameters
        ----------
        order : OrderEvent
        bar : Bar
            The bar against which the order is being filled.

        Returns
        -------
        FillEvent
        """
        cfg = self.config
        order_id = order.order_id or str(uuid.uuid4())[:8]

        # ── 1. Wide-spread check ─────────────────────────────────────────────
        spread_pct = bar.spread / bar.mid if bar.mid > 0 else 0
        if spread_pct > cfg.max_spread_pct:
            return FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                requested_quantity=order.quantity,
                filled_quantity=0.0,
                fill_price=0.0,
                commission=0.0,
                slippage=0.0,
                status=FillStatus.REJECTED,
                order_id=order_id,
            )

        # ── 2. Available volume ──────────────────────────────────────────────
        max_fill = bar.volume * cfg.max_volume_pct
        filled_qty = min(order.quantity, max_fill)
        status = FillStatus.FULL if filled_qty >= order.quantity else (
            FillStatus.PARTIAL if cfg.allow_partial_fills else FillStatus.REJECTED
        )

        if status == FillStatus.REJECTED:
            return FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                requested_quantity=order.quantity,
                filled_quantity=0.0,
                fill_price=0.0,
                commission=0.0,
                slippage=0.0,
                status=FillStatus.REJECTED,
                order_id=order_id,
            )

        # ── 3. Base fill price: cross the spread ─────────────────────────────
        base_price = bar.ask if order.side == OrderSide.BUY else bar.bid

        # For limit orders: check if limit is accessible
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and order.limit_price < bar.ask:
                # Limit too low — not filled
                return FillEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    requested_quantity=order.quantity,
                    filled_quantity=0.0,
                    fill_price=0.0,
                    commission=0.0,
                    slippage=0.0,
                    status=FillStatus.REJECTED,
                    order_id=order_id,
                )
            elif order.side == OrderSide.SELL and order.limit_price > bar.bid:
                return FillEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    requested_quantity=order.quantity,
                    filled_quantity=0.0,
                    fill_price=0.0,
                    commission=0.0,
                    slippage=0.0,
                    status=FillStatus.REJECTED,
                    order_id=order_id,
                )
            base_price = order.limit_price

        # ── 4. Market impact ─────────────────────────────────────────────────
        volume_participation = filled_qty / bar.volume if bar.volume > 0 else 0
        impact = cfg.market_impact_coeff * volume_participation * bar.close
        direction = 1 if order.side == OrderSide.BUY else -1
        fill_price = base_price + direction * impact

        # ── 5. Commission and slippage ───────────────────────────────────────
        notional = filled_qty * fill_price
        commission = notional * cfg.commission_pct
        slippage = abs(fill_price - bar.mid) * filled_qty

        return FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=filled_qty,
            fill_price=round(fill_price, 6),
            commission=round(commission, 4),
            slippage=round(slippage, 4),
            status=status,
            order_id=order_id,
        )
