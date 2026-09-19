"""Fill pricing and commission.

The fill price is where a backtest most often flatters itself.
"""
import pytest

from backtest.events import FillEvent, OrderEvent
from backtest.execution import SimulatedExecutionHandler


def test_fill_uses_the_current_bar_close(ramp_prices, events):
    ramp_prices.update_bars()  # bar 0, close 100
    ramp_prices.update_bars()  # bar 1, close 101
    ex = SimulatedExecutionHandler(events, ramp_prices)
    while not events.empty():
        events.get_nowait()

    ex.execute_order(OrderEvent("RAMP", "MKT", 10, "BUY"))
    fill = events.get_nowait()
    assert fill.fill_cost == pytest.approx(1010.0), "10 shares at the 101 close"


def test_signal_and_fill_share_the_same_bar_zero_latency(ramp_prices, events):
    """DEFECT: a signal derived from a bar's close is filled at that same close.

    In live trading the close is not tradable at the moment it is observed; the
    realistic assumption is the next bar's open. This is an optimistic
    assumption, not a neutral one, and it is not currently configurable.
    See KNOWN_ISSUES.md #2.
    """
    ramp_prices.update_bars()
    observed_close = ramp_prices.get_latest_bar_value("RAMP", "adj close")
    ex = SimulatedExecutionHandler(events, ramp_prices)
    while not events.empty():
        events.get_nowait()
    ex.execute_order(OrderEvent("RAMP", "MKT", 1, "BUY"))
    fill = events.get_nowait()
    assert fill.fill_cost == pytest.approx(observed_close), (
        "fill price equals the close the strategy just saw; no latency is modelled"
    )


def test_slippage_penalises_both_sides(ramp_prices, events):
    ramp_prices.update_bars()
    ex = SimulatedExecutionHandler(events, ramp_prices, slippage=0.01)
    while not events.empty():
        events.get_nowait()

    ex.execute_order(OrderEvent("RAMP", "MKT", 1, "BUY"))
    buy = events.get_nowait().fill_cost
    ex.execute_order(OrderEvent("RAMP", "MKT", 1, "SELL"))
    sell = events.get_nowait().fill_cost
    assert buy > 100.0 > sell, "slippage must hurt the buyer and the seller"


def test_order_is_silently_dropped_when_no_bar_exists(flat_prices, events):
    """DEFECT: a bare `except Exception: return` swallows the order.

    No fill, no warning, no counter. A strategy that tracks its own position
    then believes it holds something it does not. See KNOWN_ISSUES.md #3.
    """
    ex = SimulatedExecutionHandler(events, flat_prices)  # no update_bars() yet
    ex.execute_order(OrderEvent("FLAT", "MKT", 10, "BUY"))
    assert events.empty(), "order vanished with no trace — documented defect"


def test_non_order_events_are_ignored(ramp_prices, events):
    ramp_prices.update_bars()
    ex = SimulatedExecutionHandler(events, ramp_prices)
    while not events.empty():
        events.get_nowait()
    ex.execute_order(FillEvent("2024-01-01", "RAMP", "SIM", 1, "BUY", 100.0))
    assert events.empty()


# ----------------------------------------------------------------- commission

def test_commission_respects_the_stated_minimum_for_normal_trades():
    fill = FillEvent("2024-01-01", "X", "SIM", 100, "BUY", 10_000.0)
    assert fill.commission == pytest.approx(1.3)


def test_commission_scales_with_quantity_above_the_floor():
    fill = FillEvent("2024-01-01", "X", "SIM", 400, "BUY", 100_000.0)
    assert fill.commission == pytest.approx(0.013 * 400)


def test_large_order_uses_the_cheaper_tier():
    small = FillEvent("2024-01-01", "X", "SIM", 500, "BUY", 1_000_000.0).commission
    large = FillEvent("2024-01-01", "X", "SIM", 501, "BUY", 1_000_000.0).commission
    assert large < small, "crossing 500 shares must move to the cheaper per-share tier"


def test_commission_floor_is_breached_on_small_notional():
    """DEFECT: the 0.5%-of-notional cap overrides the 1.30 minimum.

    A 100-dollar trade is charged 0.50, below the documented floor. Real broker
    schedules apply the cap and the minimum together. See KNOWN_ISSUES.md #4.
    """
    fill = FillEvent("2024-01-01", "X", "SIM", 10, "BUY", 100.0)
    assert fill.commission == pytest.approx(0.5)
    assert fill.commission < 1.3


def test_zero_notional_fill_is_free():
    """Edge case worth pinning: a zero-quantity order costs nothing."""
    assert FillEvent("2024-01-01", "X", "SIM", 0, "BUY", 0.0).commission == 0.0
