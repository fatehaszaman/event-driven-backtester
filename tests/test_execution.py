"""Fill pricing and commission.

The fill price is where a backtest most often flatters itself. Orders fill at
the next bar's open, never at the close the signal was computed from.
"""
import logging

import pandas as pd
import pytest

from backtest.data import CSVDataHandler
from backtest.events import FillEvent, OrderEvent
from backtest.execution import SimulatedExecutionHandler


@pytest.fixture
def gapped_prices(tmp_path, events):
    """Opens sit 0.5 below each close, so a next-open fill and a same-close
    fill give different prices. Adj close equals close."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    closes = [100.0, 101.0, 102.0]
    df = pd.DataFrame(
        {"Open": [c - 0.5 for c in closes], "High": closes, "Low": closes,
         "Close": closes, "Adj Close": closes, "Volume": [1000] * 3},
        index=pd.Index(dates, name="Date"),
    )
    df.to_csv(tmp_path / "GAP.csv")
    return CSVDataHandler(events, str(tmp_path), ["GAP"])


def _drain(events):
    while not events.empty():
        events.get_nowait()


def test_order_waits_for_the_next_bar_and_fills_at_its_open(gapped_prices, events):
    gapped_prices.update_bars()  # bar 0, open 99.5, close 100
    ex = SimulatedExecutionHandler(events, gapped_prices)
    _drain(events)

    ex.execute_order(OrderEvent("GAP", "MKT", 10, "BUY"))
    assert events.empty(), "no fill may be produced on the bar that created the order"
    assert ex.on_market() == [], "bar 0 has not ended, so nothing fills yet"

    gapped_prices.update_bars()  # bar 1, open 100.5, close 101
    (fill,) = ex.on_market()
    assert fill.fill_cost == pytest.approx(1005.0), "10 shares at the 100.5 open of bar 1"
    assert fill.timeindex == pd.Timestamp("2024-01-02")


def test_signal_is_not_filled_at_the_close_it_was_computed_from(gapped_prices, events):
    """Fixed KNOWN_ISSUES.md #2. A signal computed from bar t's close used to
    fill at that same close. It now fills at bar t+1's open."""
    gapped_prices.update_bars()
    observed_close = gapped_prices.get_latest_bar_value("GAP", "adj close")
    ex = SimulatedExecutionHandler(events, gapped_prices)
    ex.execute_order(OrderEvent("GAP", "MKT", 1, "BUY"))
    gapped_prices.update_bars()
    (fill,) = ex.on_market()
    assert fill.fill_cost != pytest.approx(observed_close)
    assert fill.fill_cost == pytest.approx(100.5)


def test_open_is_put_on_the_adjusted_basis(tmp_path, events):
    """Positions are marked at adj close, so the fill uses open * adj close / close."""
    dates = pd.bdate_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {"Open": [100.0, 100.0], "High": [100.0, 100.0], "Low": [100.0, 100.0],
         "Close": [100.0, 100.0], "Adj Close": [50.0, 50.0], "Volume": [1000, 1000]},
        index=pd.Index(dates, name="Date"),
    )
    df.to_csv(tmp_path / "ADJ.csv")
    data = CSVDataHandler(events, str(tmp_path), ["ADJ"])
    data.update_bars()
    ex = SimulatedExecutionHandler(events, data)
    ex.execute_order(OrderEvent("ADJ", "MKT", 1, "BUY"))
    data.update_bars()
    (fill,) = ex.on_market()
    assert fill.fill_cost == pytest.approx(50.0)


def _corporate_action_handler(tmp_path, events, sym, opens, closes, adj):
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame(
        {"Open": opens, "High": closes, "Low": closes, "Close": closes,
         "Adj Close": adj, "Volume": [1000] * len(closes)},
        index=pd.Index(dates, name="Date"),
    )
    df.to_csv(tmp_path / f"{sym}.csv")
    return CSVDataHandler(events, str(tmp_path), [sym])


def test_fill_after_a_split_uses_the_fill_bar_ratio(tmp_path, events):
    """2-for-1 split goes ex on bar 1 and a 1% dividend goes ex on bar 2, so
    Yahoo-style back-adjustment gives adj/close ratios of 0.495, 0.99 and 1.
    An order from bar 0 fills at bar 1's open on bar 1's ratio:
    101 * 0.99 = 99.99. Using bar 0's ratio gives 49.995; no scaling gives 101."""
    closes = [200.0, 100.0, 100.0]
    data = _corporate_action_handler(
        tmp_path, events, "SPL",
        opens=[199.0, 101.0, 100.0], closes=closes,
        adj=[c * r for c, r in zip(closes, [0.495, 0.99, 1.0])])
    data.update_bars()
    ex = SimulatedExecutionHandler(events, data)
    ex.execute_order(OrderEvent("SPL", "MKT", 1, "BUY"))
    data.update_bars()
    (fill,) = ex.on_market()
    assert fill.fill_cost == pytest.approx(99.99)


def test_fill_on_an_ex_dividend_bar_uses_the_fill_bar_ratio(tmp_path, events):
    """Dividends go ex on bar 1 (1%) and bar 2 (2%), giving ratios of 0.9702,
    0.98 and 1. An order from bar 0 fills at bar 1's open of 98 on bar 1's
    ratio: 96.04. Using bar 0's ratio gives 95.08; no scaling gives 98."""
    closes = [100.0, 98.5, 98.0]
    data = _corporate_action_handler(
        tmp_path, events, "DIV",
        opens=[100.0, 98.0, 98.0], closes=closes,
        adj=[c * r for c, r in zip(closes, [0.9702, 0.98, 1.0])])
    data.update_bars()
    ex = SimulatedExecutionHandler(events, data)
    ex.execute_order(OrderEvent("DIV", "MKT", 1, "BUY"))
    data.update_bars()
    (fill,) = ex.on_market()
    assert fill.fill_cost == pytest.approx(96.04)


def test_order_on_the_last_bar_is_not_filled_and_is_logged(gapped_prices, events, caplog):
    for _ in range(3):
        gapped_prices.update_bars()  # all three bars loaded
    ex = SimulatedExecutionHandler(events, gapped_prices)
    order = OrderEvent("GAP", "MKT", 1, "BUY")
    ex.execute_order(order)

    gapped_prices.update_bars()  # data exhausted: market event with no new bar
    assert ex.on_market() == [], "the repeated final bar must not fill the order"

    with caplog.at_level(logging.WARNING, logger="backtest.execution"):
        ex.cancel_pending()
    assert ex.cancelled_orders == [order]
    assert ex.pending_orders == []
    assert "Order not filled" in caplog.text


def test_slippage_penalises_both_sides(ramp_prices, events):
    ramp_prices.update_bars()
    ex = SimulatedExecutionHandler(events, ramp_prices, slippage=0.01)
    ex.execute_order(OrderEvent("RAMP", "MKT", 1, "BUY"))
    ex.execute_order(OrderEvent("RAMP", "MKT", 1, "SELL"))
    ramp_prices.update_bars()  # bar 1 opens at 101
    buy, sell = (f.fill_cost for f in ex.on_market())
    assert buy > 101.0 > sell, "slippage must hurt the buyer and the seller"


def test_order_is_silently_dropped_when_no_bar_exists(flat_prices, events):
    """DEFECT: a bare `except Exception: return` swallows the order.

    No fill, no warning, no counter. A strategy that tracks its own position
    then believes it holds something it does not. See KNOWN_ISSUES.md #3.
    """
    ex = SimulatedExecutionHandler(events, flat_prices)  # no update_bars() yet
    ex.execute_order(OrderEvent("FLAT", "MKT", 10, "BUY"))
    assert events.empty(), "order vanished with no trace — documented defect"
    assert ex.pending_orders == [], "and it is not held for a later fill either"


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
