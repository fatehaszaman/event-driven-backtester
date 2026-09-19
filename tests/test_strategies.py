"""Strategy signal logic on series where the correct signal is unambiguous."""
import queue

import numpy as np
import pytest

from backtest.data import CSVDataHandler
from backtest.events import EventType
from strategies.moving_average import EMACrossoverStrategy, SMACrossoverStrategy
from tests.conftest import write_csv


def _handler(tmp_path, closes, symbol="S"):
    import pandas as pd
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    write_csv(tmp_path / f"{symbol}.csv", dates, closes)
    q = queue.Queue()
    return CSVDataHandler(q, str(tmp_path), [symbol]), q


def _drain(q):
    out = []
    while not q.empty():
        e = q.get_nowait()
        if e.type == EventType.SIGNAL:
            out.append(e.signal_type)
    return out


def test_no_signal_before_the_long_window_is_filled(tmp_path):
    dh, q = _handler(tmp_path, [100.0 + i for i in range(30)])
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(19):
        dh.update_bars()
        strat.calculate_signals(None)
    assert _drain(q) == [], "a strategy must not act on an incomplete window"


def test_rising_series_produces_exactly_one_long(tmp_path):
    dh, q = _handler(tmp_path, [100.0 + i for i in range(40)])
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(40):
        dh.update_bars()
        strat.calculate_signals(None)
    assert _drain(q) == ["LONG"], "no repeat entries while already long"


def test_reversal_produces_long_then_exit(tmp_path):
    closes = [100.0 + i for i in range(30)] + [130.0 - 2 * i for i in range(30)]
    dh, q = _handler(tmp_path, closes)
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(len(closes)):
        dh.update_bars()
        strat.calculate_signals(None)
    assert _drain(q) == ["LONG", "EXIT"]


def test_falling_series_never_goes_long(tmp_path):
    dh, q = _handler(tmp_path, [200.0 - i for i in range(40)])
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(40):
        dh.update_bars()
        strat.calculate_signals(None)
    assert "LONG" not in _drain(q)


def test_flat_series_produces_no_signal(tmp_path):
    dh, q = _handler(tmp_path, [100.0] * 40)
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(40):
        dh.update_bars()
        strat.calculate_signals(None)
    assert _drain(q) == [], "equal averages must not trigger a crossover"


def test_internal_flag_tracks_signals_not_fills(tmp_path):
    """DEFECT: `bought` flips when the SIGNAL is emitted, not when a fill
    arrives.

    The execution handler silently drops orders it cannot price (see
    test_execution.py). When that happens the strategy believes it is long
    while the portfolio holds nothing, and because it believes it is long it
    never signals LONG again. The desync is permanent and silent.
    See KNOWN_ISSUES.md #9.
    """
    dh, q = _handler(tmp_path, [100.0 + i for i in range(40)])
    strat = SMACrossoverStrategy(dh, q, short_window=5, long_window=20)
    for _ in range(25):
        dh.update_bars()
        strat.calculate_signals(None)
    assert strat.bought["S"] is True, "flag set with no fill having occurred"


def test_ema_weights_recent_observations_more_heavily():
    dh = type("D", (), {"symbol_list": []})()
    strat = EMACrossoverStrategy(dh, queue.Queue())
    rising = [float(i) for i in range(1, 11)]
    assert strat._ema(rising, 5) > np.mean(rising)


def test_ema_of_a_constant_series_is_that_constant():
    strat = EMACrossoverStrategy(type("D", (), {"symbol_list": []})(), queue.Queue())
    assert strat._ema([7.0] * 20, 5) == pytest.approx(7.0)


def test_ema_windows_are_seeded_on_different_histories(tmp_path):
    """DEFECT: the short EMA is seeded from the last `short_window` bars while
    the long EMA is seeded from all `long_window` bars.

    The two averages therefore start from different initial values, so the
    crossover compares series built on inconsistent bases. With enough bars the
    effect decays, but on the first signals after warm-up it does not.
    See KNOWN_ISSUES.md #10.
    """
    strat = EMACrossoverStrategy(type("D", (), {"symbol_list": []})(), queue.Queue())
    bars = [float(i) for i in range(1, 27)]
    short_on_tail = strat._ema(bars[-12:], 12)
    short_on_full = strat._ema(bars, 12)
    assert short_on_tail != pytest.approx(short_on_full), (
        "seeding choice changes the EMA, so it changes signal timing"
    )
