"""End-to-end runs. The README shows an SMA crossover example; this makes sure
that example actually executes and that the results obey basic identities.
"""
import pandas as pd
import pytest

from backtest.data import CSVDataHandler
from backtest.engine import Backtest
from backtest.execution import SimulatedExecutionHandler
from backtest.portfolio import NaivePortfolio
from strategies.moving_average import SMACrossoverStrategy
from tests.conftest import write_csv


def _run(csv_dir, symbol, capital=100_000.0, **params):
    bt = Backtest(
        csv_dir=csv_dir,
        symbol_list=[symbol],
        initial_capital=capital,
        heartbeat=0.0,
        start_date=None,
        end_date=None,
        data_handler_cls=CSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=NaivePortfolio,
        strategy_cls=SMACrossoverStrategy,
        strategy_params=params or {"short_window": 5, "long_window": 20},
    )
    bt.run()
    return bt


@pytest.fixture
def ramp_dir(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60)
    write_csv(tmp_path / "RAMP.csv", dates, [100.0 + i for i in range(60)])
    return str(tmp_path)


@pytest.fixture
def flat_dir(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60)
    write_csv(tmp_path / "FLAT.csv", dates, [100.0] * 60)
    return str(tmp_path)


def test_full_backtest_runs_and_produces_an_equity_curve(ramp_dir):
    bt = _run(ramp_dir, "RAMP")
    assert len(bt.portfolio.all_holdings) > 0
    assert bt.num_events > 0


def test_a_rising_market_with_a_long_only_trend_strategy_makes_money(ramp_dir):
    bt = _run(ramp_dir, "RAMP")
    final = bt.portfolio.all_holdings[-1]["total"]
    assert final > 100_000.0, f"long-only on a monotone ramp ended at {final:.2f}"


def test_a_flat_market_cannot_produce_profit(flat_dir):
    """The strongest sanity check available: no price change, so no strategy
    may end above its starting capital."""
    bt = _run(flat_dir, "FLAT")
    final = bt.portfolio.all_holdings[-1]["total"]
    assert final <= 100_000.0 + 1e-6, f"manufactured {final - 100_000:.2f} from nothing"


def test_cash_plus_market_value_equals_reported_total(ramp_dir):
    """Accounting identity that must hold on every bar."""
    bt = _run(ramp_dir, "RAMP")
    for row in bt.portfolio.all_holdings:
        assert row["total"] == pytest.approx(row["cash"] + row["RAMP"])


def test_positions_and_holdings_have_matching_lengths(ramp_dir):
    bt = _run(ramp_dir, "RAMP")
    assert len(bt.portfolio.all_positions) == len(bt.portfolio.all_holdings)


def test_summary_stats_are_finite_and_parseable(ramp_dir):
    bt = _run(ramp_dir, "RAMP")
    stats = dict(bt.portfolio.output_summary_stats())
    assert set(stats) == {"Total Return", "Sharpe Ratio", "Max Drawdown"}
    for label, val in stats.items():
        assert "nan" not in val.lower(), f"{label} reported {val}"


def test_run_is_deterministic(ramp_dir):
    """Same inputs, same outputs. No hidden state or randomness."""
    a = _run(ramp_dir, "RAMP").portfolio.all_holdings[-1]["total"]
    b = _run(ramp_dir, "RAMP").portfolio.all_holdings[-1]["total"]
    assert a == pytest.approx(b)


def test_equity_curve_has_one_extra_row_from_the_duplicate_final_bar(ramp_dir):
    """DEFECT, consequence of KNOWN_ISSUES.md #1.

    60 bars produce 61 holdings rows, because the final update_bars() emits a
    MARKET event with no new data and the portfolio marks the previous bar
    again. The duplicate row adds a spurious 0% return, which dilutes the
    standard deviation and therefore inflates the reported Sharpe ratio.
    """
    bt = _run(ramp_dir, "RAMP")
    assert len(bt.portfolio.all_holdings) == 61
    last, second_last = bt.portfolio.all_holdings[-1], bt.portfolio.all_holdings[-2]
    assert last["datetime"] == second_last["datetime"]
    assert last["total"] == pytest.approx(second_last["total"])


def test_no_trade_is_ever_priced_outside_the_observed_range(ramp_dir):
    """Fills must occur at prices that existed. Catches accidental use of a
    future bar's price, which would show up as a fill above the ramp's max."""
    bt = _run(ramp_dir, "RAMP")
    for row in bt.portfolio.all_holdings:
        if row["RAMP"] != 0:
            implied = abs(row["RAMP"]) / 100
            assert 99.0 <= implied <= 160.0


class _LongOnBar:
    """Emits one LONG signal on the chosen 1-based bar and logs each call."""

    def __init__(self, data_handler, events, bar=1, log=None):
        from backtest.events import SignalEvent

        self._signal = SignalEvent
        self.data_handler = data_handler
        self.events = events
        self.bar = bar
        self.log = log if log is not None else []
        self.seen = 0

    def calculate_signals(self, event):
        bars = self.data_handler.get_latest_bars("GAP", N=10_000)
        if len(bars) == self.seen:
            return  # repeated final market event, no new bar
        self.seen = len(bars)
        self.log.append(("signals", bars[-1][0]))
        if self.seen == self.bar:
            self.events.put(self._signal("GAP", "LONG"))


@pytest.fixture
def gap_dir(tmp_path):
    """Opens sit 0.5 below each close, so open and close fills differ."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    closes = [100.0, 101.0, 102.0, 103.0]
    df = pd.DataFrame(
        {"Open": [c - 0.5 for c in closes], "High": closes, "Low": closes,
         "Close": closes, "Adj Close": closes, "Volume": [1000] * 4},
        index=pd.Index(dates, name="Date"),
    )
    df.to_csv(tmp_path / "GAP.csv")
    return str(tmp_path), dates


def _run_logged(csv_dir, bar):
    log = []

    class LoggedExecution(SimulatedExecutionHandler):
        def on_market(self, event=None):
            fills = super().on_market(event)
            for f in fills:
                log.append(("fill", f.timeindex))
            return fills

    bt = Backtest(
        csv_dir=csv_dir, symbol_list=["GAP"], initial_capital=100_000.0,
        heartbeat=0.0, start_date=None, end_date=None,
        data_handler_cls=CSVDataHandler,
        execution_handler_cls=LoggedExecution,
        portfolio_cls=NaivePortfolio, strategy_cls=_LongOnBar,
        strategy_params={"bar": bar, "log": log},
    )
    bt.run()
    return bt, log


def test_on_bar_t_plus_1_the_fill_comes_before_new_signals(gap_dir):
    """An order from bar t fills at bar t+1's open before the strategy sees
    bar t+1, and the t+1 holdings row is marked after that fill."""
    csv_dir, dates = gap_dir
    bt, log = _run_logged(csv_dir, bar=1)

    assert log[:3] == [("signals", dates[0]), ("fill", dates[1]), ("signals", dates[1])]
    row = next(r for r in bt.portfolio.all_holdings if r["datetime"] == dates[1])
    assert bt.portfolio.all_positions[-1]["GAP"] == 100
    assert row["cash"] == pytest.approx(100_000.0 - 100 * 100.5 - row["commission"])
    assert row["GAP"] == pytest.approx(100 * 101.0), "marked at the t+1 close after the fill"


def test_duplicate_final_market_event_cannot_fill_a_last_bar_order(gap_dir):
    """KNOWN_ISSUES.md #1 re-emits the last bar. An order placed on that bar
    must stay unfilled rather than fill at the bar's own price."""
    csv_dir, dates = gap_dir
    bt, log = _run_logged(csv_dir, bar=4)

    assert not any(kind == "fill" for kind, _ in log)
    assert len(bt.execution_handler.cancelled_orders) == 1
    assert bt.portfolio.all_positions[-1]["GAP"] == 0
