"""Performance statistics.

These are the numbers a reader of the README actually judges, so they get
tested against hand-computed values and against known degenerate inputs.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.performance import create_cagr, create_drawdowns, create_sharpe_ratio


# ---------------------------------------------------------------------- sharpe

def test_sharpe_matches_hand_computation():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = np.sqrt(252) * r.mean() / r.std()
    assert create_sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_guard_catches_exactly_zero_volatility():
    """The guard works when the standard deviation is bit-exactly zero."""
    assert create_sharpe_ratio(pd.Series([0.0] * 10)) == 0.0


def test_sharpe_is_zero_on_near_constant_returns():
    """Fixed (formerly KNOWN_ISSUES.md #8): a constant 1% return series has
    std ~1.8e-18 rather than 0.0. The guard now uses a tolerance, so the
    Sharpe is 0.0 instead of ~8.7e16."""
    assert create_sharpe_ratio(pd.Series([0.01] * 10)) == 0.0


def test_portfolio_summary_stats_reuses_the_guarded_sharpe(ramp_prices, events):
    """NaivePortfolio.output_summary_stats used its own `std() > 0` guard,
    the second home of the #8 defect. It now delegates to
    create_sharpe_ratio so there is one guarded implementation."""
    import inspect

    from backtest.portfolio import NaivePortfolio

    src = inspect.getsource(NaivePortfolio.output_summary_stats)
    assert "create_sharpe_ratio(" in src
    assert "returns.std() > 0" not in src


def test_sharpe_sign_follows_mean_return():
    assert create_sharpe_ratio(pd.Series([0.02, 0.01, 0.03, 0.015])) > 0
    assert create_sharpe_ratio(pd.Series([-0.02, -0.01, -0.03, -0.015])) < 0


def test_sharpe_scales_with_the_declared_period():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    assert create_sharpe_ratio(r, periods=252) / create_sharpe_ratio(r, periods=52) == (
        pytest.approx(np.sqrt(252 / 52))
    )


def test_sharpe_uses_sample_standard_deviation():
    """pandas defaults to ddof=1 while numpy defaults to ddof=0. Which one is
    in use changes the reported figure, so it is pinned deliberately."""
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    ddof1 = np.sqrt(252) * r.mean() / r.std(ddof=1)
    ddof0 = np.sqrt(252) * r.mean() / r.std(ddof=0)
    assert create_sharpe_ratio(r) == pytest.approx(ddof1)
    assert create_sharpe_ratio(r) != pytest.approx(ddof0)


def test_sharpe_assumes_a_zero_risk_free_rate():
    """Documented assumption, not a bug: there is no rf argument. In a
    4-5% rate environment this overstates the ratio, and the README should
    say so. See KNOWN_ISSUES.md #6."""
    import inspect
    assert "risk_free" not in inspect.signature(create_sharpe_ratio).parameters


# -------------------------------------------------------------------- drawdown

def test_drawdown_is_zero_on_a_monotonically_rising_curve():
    eq = pd.Series([100.0, 101, 102, 103])
    dd, mx = create_drawdowns(eq)
    assert mx == pytest.approx(0.0)
    assert (dd <= 0).all()


def test_max_drawdown_matches_hand_computation():
    # peak 120 -> trough 90 = -25%
    eq = pd.Series([100.0, 120.0, 90.0, 110.0])
    _, mx = create_drawdowns(eq)
    assert mx == pytest.approx(-0.25)


def test_drawdown_measures_from_the_high_water_mark_not_the_start():
    eq = pd.Series([100.0, 50.0, 200.0, 150.0])
    _, mx = create_drawdowns(eq)
    assert mx == pytest.approx(-0.5), "the 50% early fall is the worst, not the later 25%"


def test_drawdown_is_never_positive():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    dd, _ = create_drawdowns(eq)
    assert dd.max() <= 1e-12


# ------------------------------------------------------------------------ cagr

def test_cagr_of_a_doubling_over_one_year():
    eq = pd.Series([100.0] * 251 + [200.0])
    assert create_cagr(eq, periods=252) == pytest.approx(1.0, abs=0.02)


def test_cagr_is_zero_for_a_flat_curve():
    assert create_cagr(pd.Series([100.0] * 252)) == pytest.approx(0.0)


def test_cagr_is_negative_for_a_losing_curve():
    assert create_cagr(pd.Series(np.linspace(100, 80, 252))) < 0


def test_cagr_reports_worse_than_total_loss_when_equity_goes_negative():
    """DEFECT: negative terminal equity over exactly one year gives -110%.

    A loss worse than -100% is not a meaningful growth rate. The function
    should refuse rather than return a number. See KNOWN_ISSUES.md #7.
    """
    eq = pd.Series([100.0] * 251 + [-10.0])   # exactly one year
    assert create_cagr(eq) == pytest.approx(-1.1)


def test_cagr_returns_nan_when_equity_goes_negative_over_a_fractional_year():
    """DEFECT, same root cause: a fractional power of a negative base is nan,
    so the tearsheet title renders 'CAGR: nan%'. See KNOWN_ISSUES.md #7."""
    eq = pd.Series([100.0] * 400 + [-10.0])
    assert np.isnan(create_cagr(eq))


def test_cagr_annualises_two_bars_without_complaint():
    """DEFECT: a 1% gain over two bars is annualised to +250% with no warning.

    There is no minimum-length check, so a short or truncated backtest
    produces a headline growth rate with no statistical basis.
    See KNOWN_ISSUES.md #7.
    """
    val = create_cagr(pd.Series([100.0, 101.0]), periods=252)
    assert val == pytest.approx(2.503, abs=0.01)
