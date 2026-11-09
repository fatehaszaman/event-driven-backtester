"""Mean-variance optimizer with a turnover penalty.

The inputs are synthetic: a fixed four-asset covariance and expected returns
chosen so the answers can be checked against closed-form results and the
first-order conditions, not market estimates.
"""
from itertools import pairwise

import numpy as np
import pytest

from backtest.optimizer import closed_form_weights, ewma_covariance, optimize

MU = np.array([0.08, 0.05, 0.11, 0.03])
VOLS = np.array([0.20, 0.15, 0.30, 0.10])
CORR = np.array([
    [1.0, 0.3, 0.5, 0.1],
    [0.3, 1.0, 0.2, 0.0],
    [0.5, 0.2, 1.0, 0.2],
    [0.1, 0.0, 0.2, 1.0],
])
SIGMA = CORR * np.outer(VOLS, VOLS)
PREV = np.array([0.25, 0.25, 0.25, 0.25])


# ------------------------------------------------------------- closed form

@pytest.mark.parametrize("lam", [1.0, 3.0, 10.0])
def test_numerical_solver_matches_the_closed_form_without_turnover_cost(lam):
    """kappa = 0 and bounds wide enough never to bind, so SLSQP must land on
    the analytic optimum."""
    numeric = optimize(MU, SIGMA, lam, bounds=(-50.0, 50.0)).weights
    assert numeric == pytest.approx(closed_form_weights(MU, SIGMA, lam), abs=1e-6)


def test_closed_form_satisfies_the_first_order_condition():
    """mu - 2 lam Sigma w must equal the same multiplier for every asset."""
    lam = 3.0
    w = closed_form_weights(MU, SIGMA, lam)
    marginal = MU - 2.0 * lam * SIGMA @ w
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert marginal == pytest.approx(np.full(4, marginal[0]), abs=1e-12)


def test_rescaled_inverse_sigma_mu_is_not_the_optimum():
    """Rescaling Sigma^-1 mu to sum to one gives the tangency portfolio,
    which ignores lam. The constrained optimum moves with lam."""
    tangency = np.linalg.solve(SIGMA, MU)
    tangency /= tangency.sum()
    low, high = closed_form_weights(MU, SIGMA, 1.0), closed_form_weights(MU, SIGMA, 10.0)
    assert np.abs(low - high).max() > 0.05
    assert np.abs(low - tangency).max() > 0.05


# ------------------------------------------------------------- constraints

@pytest.mark.parametrize("bounds", [
    (0.0, 0.6),                                          # long-only, 60% cap
    [(0.1, 0.4), (0.0, 0.3), (0.0, 0.5), (0.05, 0.2)],   # per-asset bounds
])
def test_weights_sum_to_one_and_stay_within_bounds(bounds):
    w = optimize(MU, SIGMA, 3.0, turnover_cost=0.01, prev_weights=PREV, bounds=bounds).weights
    pairs = [bounds] * 4 if isinstance(bounds, tuple) else bounds
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    for wi, (lo, hi) in zip(w, pairs):
        assert lo - 1e-9 <= wi <= hi + 1e-9


def test_reported_turnover_and_objective_are_consistent():
    r = optimize(MU, SIGMA, 3.0, turnover_cost=0.02, prev_weights=PREV, bounds=(0.0, 1.0))
    assert r.turnover == pytest.approx(np.abs(r.weights - PREV).sum(), abs=1e-12)
    expected = r.weights @ MU - 3.0 * r.weights @ SIGMA @ r.weights - 0.02 * r.turnover
    assert r.objective == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------- comparative statics

def test_raising_turnover_cost_never_increases_turnover():
    turnovers = [
        optimize(MU, SIGMA, 3.0, turnover_cost=k, prev_weights=PREV, bounds=(0.0, 1.0)).turnover
        for k in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    ]
    assert all(b <= a + 1e-7 for a, b in pairwise(turnovers))
    assert turnovers[-1] < turnovers[0]


def test_large_turnover_cost_keeps_the_current_portfolio():
    w = optimize(MU, SIGMA, 3.0, turnover_cost=10.0, prev_weights=PREV, bounds=(0.0, 1.0)).weights
    assert w == pytest.approx(PREV, abs=1e-7)


def test_raising_risk_aversion_never_increases_variance():
    variances = [
        optimize(MU, SIGMA, lam, turnover_cost=0.005, prev_weights=PREV, bounds=(0.0, 1.0)).variance
        for lam in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    ]
    assert all(b <= a + 1e-9 for a, b in pairwise(variances))
    assert variances[-1] < variances[0]


def test_raising_one_assets_expected_return_never_lowers_its_weight():
    weights = []
    for bump in [0.0, 0.01, 0.02, 0.04, 0.08]:
        mu = MU.copy()
        mu[1] += bump
        weights.append(optimize(mu, SIGMA, 3.0, turnover_cost=0.005,
                                prev_weights=PREV, bounds=(0.0, 1.0)).weights[1])
    assert all(b >= a - 1e-7 for a, b in pairwise(weights))
    assert weights[-1] > weights[0]


# ------------------------------------------------------------------- EWMA

def test_ewma_covariance_matches_the_weighted_sum():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, size=(50, 3))
    decay = 0.94
    w = (1 - decay) * decay ** np.arange(49, -1, -1)
    expected = sum(wk * np.outer(row, row) for wk, row in zip(w, r)) / w.sum()
    assert ewma_covariance(r, decay) == pytest.approx(expected, abs=1e-15)


def test_ewma_puts_most_weight_on_recent_returns():
    """Calm history then one large recent shock: EWMA variance reacts far more
    than an equally weighted estimate would."""
    r = np.vstack([np.full((99, 1), 0.001), [[0.05]]])
    ewma = ewma_covariance(r, 0.94)[0, 0]
    equal = (r ** 2).mean()
    assert ewma > 5 * equal


# --------------------------------------------------------------- validation

@pytest.mark.parametrize("kwargs", [
    {"sigma": np.diag([0.04, 0.0225, 0.09, -0.01])},   # not positive definite
    {"bounds": (0.0, 0.2)},                             # 4 x 20% cannot reach 100%
])
def test_invalid_inputs_are_rejected(kwargs):
    args = {"mu": MU, "sigma": SIGMA, "risk_aversion": 3.0, **kwargs}
    with pytest.raises(ValueError):
        optimize(**args)
