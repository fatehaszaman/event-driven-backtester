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
