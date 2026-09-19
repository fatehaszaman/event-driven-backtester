"""The data handler carries the repository's central claim: no look-ahead.

README states "Guarantees no look-ahead bias -- bars are revealed in order."
These tests hold that claim to account.
"""
import pandas as pd
import pytest

from backtest.data import CSVDataHandler
from backtest.events import EventType
from tests.conftest import write_csv


def test_no_bars_visible_before_first_update(flat_prices):
    """Before update_bars(), nothing is knowable. Raising beats returning None."""
    with pytest.raises(ValueError, match="No bars loaded"):
        flat_prices.get_latest_bar("FLAT")


def test_bars_revealed_one_at_a_time(ramp_prices):
    """THE LOOK-AHEAD TEST. After k updates, exactly k bars are visible and the
    latest is the k-th, never a later one."""
    for k in range(1, 6):
        ramp_prices.update_bars()
        visible = ramp_prices.latest_symbol_data["RAMP"]
        assert len(visible) == k
        assert ramp_prices.get_latest_bar_value("RAMP", "adj close") == 100.0 + (k - 1)


def test_history_window_never_reaches_into_the_future(ramp_prices):
    for _ in range(4):
        ramp_prices.update_bars()
    vals = ramp_prices.get_latest_bars_values("RAMP", "adj close", N=10)
    assert vals == [100.0, 101.0, 102.0, 103.0], (
        "requesting more history than exists must return what is known, "
        "never pad with future bars"
    )


def test_market_event_emitted_per_update(ramp_prices, events):
    ramp_prices.update_bars()
    ev = events.get_nowait()
    assert ev.type == EventType.MARKET


def test_columns_are_normalised_to_lowercase(ramp_prices):
    ramp_prices.update_bars()
    assert ramp_prices.get_latest_bar_value("RAMP", "adj close") == 100.0


def test_date_filters_are_inclusive(tmp_path, events):
    dates = pd.bdate_range("2024-01-01", periods=10)
    write_csv(tmp_path / "X.csv", dates, [100.0 + i for i in range(10)])
    dh = CSVDataHandler(
        events, str(tmp_path), ["X"], start_date=dates[2], end_date=dates[5]
    )
    n = 0
    while dh.continue_backtest:
        dh.update_bars()
        n += 1
    assert len(dh.latest_symbol_data["X"]) == 4, "start and end dates are inclusive"


def test_unsorted_input_is_sorted_before_streaming(tmp_path, events):
    """A CSV in reverse order must not stream the future first."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {"Open": [104.0, 103, 102, 101, 100], "High": [1] * 5, "Low": [1] * 5,
         "Close": [104.0, 103, 102, 101, 100],
         "Adj Close": [104.0, 103, 102, 101, 100], "Volume": [1] * 5},
        index=pd.to_datetime(dates[::-1]),
    )
    df.index.name = "Date"
    df.to_csv(tmp_path / "R.csv")
    dh = CSVDataHandler(events, str(tmp_path), ["R"])
    dh.update_bars()
    assert dh.get_latest_bar_value("R", "adj close") == 100.0, (
        "first streamed bar must be the earliest date, not the first CSV row"
    )


def test_emits_a_market_event_even_when_data_is_exhausted(ramp_prices, events):
    """DEFECT: update_bars() emits MARKET unconditionally.

    On the final call no new bar is appended, yet a MARKET event still fires.
    Downstream that re-processes the previous bar. See KNOWN_ISSUES.md #1.
    """
    while ramp_prices.continue_backtest:
        ramp_prices.update_bars()
    n_bars = len(ramp_prices.latest_symbol_data["RAMP"])
    n_market = 0
    while not events.empty():
        if events.get_nowait().type == EventType.MARKET:
            n_market += 1
    assert n_bars == 10
    assert n_market == 11, (
        "documents the off-by-one: 11 market events for 10 bars. If this test "
        "starts failing because the extra event was removed, delete the test "
        "and KNOWN_ISSUES.md #1."
    )
