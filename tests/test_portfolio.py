"""Position and cash accounting. Every assertion is hand-computable."""
import pytest

from backtest.events import FillEvent, SignalEvent
from backtest.portfolio import NaivePortfolio


@pytest.fixture
def pf(ramp_prices, events):
    return NaivePortfolio(ramp_prices, events, "2024-01-01", initial_capital=100_000.0)


def _fill(qty, direction, cost, commission=0.0):
    return FillEvent("2024-01-02", "RAMP", "SIM", qty, direction, cost, commission)


def test_buy_reduces_cash_by_cost_plus_commission(pf):
    pf.update_fill(_fill(10, "BUY", 1000.0, commission=1.3))
    assert pf.current_positions["RAMP"] == 10
    assert pf.current_holdings["cash"] == pytest.approx(100_000.0 - 1000.0 - 1.3)


def test_sell_increases_cash(pf):
    pf.update_fill(_fill(10, "BUY", 1000.0))
    pf.update_fill(_fill(10, "SELL", 1100.0))
    assert pf.current_positions["RAMP"] == 0
    assert pf.current_holdings["cash"] == pytest.approx(100_000.0 + 100.0)


def test_short_sale_credits_cash_and_makes_position_negative(pf):
    pf.update_fill(_fill(10, "SELL", 1000.0))
    assert pf.current_positions["RAMP"] == -10
    assert pf.current_holdings["cash"] == pytest.approx(101_000.0)


def test_round_trip_at_the_same_price_only_loses_commission(pf):
    pf.update_fill(_fill(10, "BUY", 1000.0, commission=1.3))
    pf.update_fill(_fill(10, "SELL", 1000.0, commission=1.3))
    assert pf.current_holdings["cash"] == pytest.approx(100_000.0 - 2.6)
    assert pf.current_positions["RAMP"] == 0


def test_equity_marks_position_at_the_current_close(pf, ramp_prices):
    ramp_prices.update_bars()          # close 100
    pf.update_fill(_fill(10, "BUY", 1000.0))
    ramp_prices.update_bars()          # close 101
    pf.update_timeindex(None)
    row = pf.all_holdings[-1]
    assert row["RAMP"] == pytest.approx(1010.0)
    assert row["total"] == pytest.approx(99_000.0 + 1010.0)


def test_flat_market_leaves_equity_unchanged_apart_from_costs(flat_prices, events):
    """Sanity floor: a constant price series cannot generate P&L."""
    pf = NaivePortfolio(flat_prices, events, "2024-01-01", 100_000.0)
    flat_prices.update_bars()
    pf.update_fill(
        FillEvent("2024-01-02", "FLAT", "SIM", 10, "BUY", 1000.0, 0.0)
    )
    for _ in range(5):
        flat_prices.update_bars()
        pf.update_timeindex(None)
    assert pf.all_holdings[-1]["total"] == pytest.approx(100_000.0)


# ------------------------------------------------------------ order generation

def test_long_signal_when_flat_creates_a_buy(pf):
    order = pf._generate_order(SignalEvent("RAMP", "LONG"))
    assert (order.direction, order.quantity) == ("BUY", 100)


def test_long_signal_is_ignored_when_already_long(pf):
    pf.current_positions["RAMP"] = 100
    assert pf._generate_order(SignalEvent("RAMP", "LONG")) is None


def test_exit_closes_the_full_position_either_way(pf):
    pf.current_positions["RAMP"] = 250
    o = pf._generate_order(SignalEvent("RAMP", "EXIT"))
    assert (o.direction, o.quantity) == ("SELL", 250)
    pf.current_positions["RAMP"] = -250
    o = pf._generate_order(SignalEvent("RAMP", "EXIT"))
    assert (o.direction, o.quantity) == ("BUY", 250)


def test_exit_when_already_flat_creates_nothing(pf):
    assert pf._generate_order(SignalEvent("RAMP", "EXIT")) is None


def test_signal_strength_scales_order_size(pf):
    assert pf._generate_order(SignalEvent("RAMP", "LONG", strength=2.5)).quantity == 250


def test_zero_strength_produces_a_zero_quantity_order(pf):
    """DEFECT: strength 0 yields a 0-share order rather than no order.

    It reaches the execution handler, produces a zero-cost fill, and still
    pays nothing — harmless today, but it pollutes trade counts and would
    become a real problem the moment a fixed per-order fee is added.
    See KNOWN_ISSUES.md #5.
    """
    order = pf._generate_order(SignalEvent("RAMP", "LONG", strength=0.0))
    assert order is not None and order.quantity == 0


def test_total_field_on_current_holdings_is_stale_between_marks(pf):
    """update_fill adjusts current_holdings['total'] but update_timeindex
    recomputes it from scratch. The in-between value is not meaningful and
    nothing should read it. Pinned so a future refactor notices."""
    before = pf.current_holdings["total"]
    pf.update_fill(_fill(10, "BUY", 1000.0, commission=5.0))
    assert pf.current_holdings["total"] == pytest.approx(before - 5.0)


def test_fill_for_an_unknown_symbol_fails_loudly(pf):
    """Good behaviour, pinned: a fill for an untracked symbol raises rather
    than silently creating a position the portfolio never sized."""
    with pytest.raises(KeyError):
        pf.update_fill(FillEvent("2024-01-02", "NOPE", "SIM", 1, "BUY", 100.0, 0.0))
