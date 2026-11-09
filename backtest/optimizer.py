"""
Mean-variance portfolio construction with a turnover penalty.

Problem, for n assets with expected returns mu, covariance Sigma, current
weights w_prev, risk aversion lam and turnover cost kappa:

    maximize    w'mu - lam * w'Sigma w - kappa * sum_i |w_i - w_prev_i|
    subject to  sum_i w_i = 1
                lower_i <= w_i <= upper_i

The absolute value is not differentiable at zero, so each trade is split
into a buy part u and a sell part v:

    w - w_prev = u - v,   u >= 0,   v >= 0,   turnover = sum_i (u_i + v_i)

At the optimum at most one of u_i, v_i is non-zero (holding both would pay
kappa twice for no change in w), so sum(u + v) equals sum|w - w_prev| and
the problem becomes a smooth convex quadratic program in (w, u, v), solved
here with SLSQP.

Closed form with no turnover cost and no bounds. The Lagrangian condition
mu - 2 lam Sigma w - gamma 1 = 0 with 1'w = 1 gives

    w* = Sigma^-1 mu / (2 lam)  +  (1 - 1'Sigma^-1 mu / (2 lam)) * w_mv
    w_mv = Sigma^-1 1 / (1'Sigma^-1 1)          (minimum-variance portfolio)

so the optimum is the minimum-variance portfolio plus a self-financing
tilt toward Sigma^-1 mu that shrinks as lam rises. Rescaling Sigma^-1 mu to
sum to one gives the tangency portfolio (zero risk-free rate) instead,
which does not depend on lam and is generally not the solution of the
problem above.

This module is standalone: the backtest engine does not call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    expected_return: float     # w'mu
    variance: float            # w'Sigma w
    turnover: float            # sum |w - w_prev|
    objective: float           # w'mu - lam w'Sigma w - kappa * turnover


def _as_inputs(mu, sigma):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.shape[0]
    if mu.ndim != 1 or sigma.shape != (n, n):
        raise ValueError("mu must be length n and sigma must be n x n")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        raise ValueError("mu and sigma must be finite")
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("sigma must be symmetric")
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("sigma must be positive definite")
    return mu, sigma


def closed_form_weights(mu, sigma, risk_aversion: float) -> np.ndarray:
    """Exact optimum with weights summing to one, no bounds and no turnover cost."""
    mu, sigma = _as_inputs(mu, sigma)
    if not risk_aversion > 0:
        raise ValueError("risk_aversion must be positive")
    ones = np.ones_like(mu)
    inv_mu = np.linalg.solve(sigma, mu)
    inv_one = np.linalg.solve(sigma, ones)
    w_mv = inv_one / inv_one.sum()
    tilt = inv_mu / (2.0 * risk_aversion)
    return tilt + (1.0 - tilt.sum()) * w_mv


def optimize(
    mu,
    sigma,
    risk_aversion: float,
    turnover_cost: float = 0.0,
    prev_weights=None,
    bounds=None,
) -> OptimizationResult:
    """
    Solve the mean-variance problem with a turnover penalty.

    Parameters
    ----------
    mu : array of n expected returns per period
    sigma : n x n covariance matrix, positive definite
    risk_aversion : lam > 0
    turnover_cost : kappa >= 0, charged per unit of sum |w - w_prev|
    prev_weights : current weights; defaults to equal weights
    bounds : (lower, upper) applied to every asset, or a list of n pairs;
        None means unbounded (shorting and leverage allowed)
    """
    mu, sigma = _as_inputs(mu, sigma)
    n = mu.shape[0]
    if not risk_aversion > 0:
        raise ValueError("risk_aversion must be positive")
    if not turnover_cost >= 0:
        raise ValueError("turnover_cost must be non-negative")
    w_prev = np.full(n, 1.0 / n) if prev_weights is None else np.asarray(prev_weights, float)
    if w_prev.shape != (n,):
        raise ValueError("prev_weights must have length n")

    if bounds is None:
        w_bounds = [(None, None)] * n
    elif len(bounds) == 2 and np.isscalar(bounds[0]):
        w_bounds = [tuple(bounds)] * n
    else:
        w_bounds = [tuple(b) for b in bounds]
    if len(w_bounds) != n:
        raise ValueError("bounds must be one pair or n pairs")
    lows = [-np.inf if lo is None else lo for lo, _ in w_bounds]
    highs = [np.inf if hi is None else hi for _, hi in w_bounds]
    if any(lo > hi for lo, hi in zip(lows, highs)) or sum(lows) > 1 or sum(highs) < 1:
        raise ValueError("bounds leave no fully invested portfolio")

    if turnover_cost == 0 and bounds is None:
        w = closed_form_weights(mu, sigma, risk_aversion)
        return _result(w, mu, sigma, risk_aversion, turnover_cost, w_prev)

    # Variables x = [w, u, v]. Minimize the negative objective.
    def f(x):
        w, u, v = x[:n], x[n:2 * n], x[2 * n:]
        return -(w @ mu) + risk_aversion * (w @ sigma @ w) + turnover_cost * (u.sum() + v.sum())

    def grad(x):
        w = x[:n]
        g_w = -mu + 2.0 * risk_aversion * (sigma @ w)
        return np.concatenate([g_w, np.full(2 * n, turnover_cost)])

    eye = np.eye(n)
    constraints = [
        {"type": "eq", "fun": lambda x: x[:n].sum() - 1.0,
         "jac": lambda x: np.concatenate([np.ones(n), np.zeros(2 * n)])},
        {"type": "eq", "fun": lambda x: x[:n] - w_prev - x[n:2 * n] + x[2 * n:],
         "jac": lambda x: np.hstack([eye, -eye, eye])},
    ]
    w0 = np.clip(w_prev, lows, highs)     # SLSQP does not need a feasible start
    diff = w0 - w_prev
    x0 = np.concatenate([w0, np.maximum(diff, 0.0), np.maximum(-diff, 0.0)])

    res = minimize(
        f, x0, jac=grad, method="SLSQP",
        bounds=w_bounds + [(0.0, None)] * (2 * n),
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"optimizer did not converge: {res.message}")
    w = np.clip(res.x[:n], lows, highs)
    return _result(w, mu, sigma, risk_aversion, turnover_cost, w_prev)


def _result(w, mu, sigma, lam, kappa, w_prev) -> OptimizationResult:
    variance = float(w @ sigma @ w)
    expected = float(w @ mu)
    turnover = float(np.abs(w - w_prev).sum())
    return OptimizationResult(
        weights=w,
        expected_return=expected,
        variance=variance,
        turnover=turnover,
        objective=expected - lam * variance - kappa * turnover,
    )


def ewma_covariance(returns, decay: float = 0.94) -> np.ndarray:
    """
    RiskMetrics-style exponentially weighted covariance of a T x n array of
    returns, oldest row first, assuming zero mean:

        Sigma = sum_k (1 - decay) decay^k r_{T-k} r_{T-k}' / (1 - decay^T)

    The most recent row has the largest weight. Only the rows passed in are
    used, so the estimate is point-in-time if the caller passes returns up
    to the decision date only. Dividing by 1 - decay^T makes the weights sum
    to one on a finite sample.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("returns must be a T x n array with T >= 2")
    if not 0 < decay < 1:
        raise ValueError("decay must be strictly between 0 and 1")
    t = r.shape[0]
    weights = (1 - decay) * decay ** np.arange(t - 1, -1, -1)
    weights /= weights.sum()
    return (r * weights[:, None]).T @ r
