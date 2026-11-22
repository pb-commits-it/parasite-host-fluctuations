"""Second-order environmental growth costs for a continuous-time stage ring.

Only numpy and scipy are required.  See derivation_ct.md for the derivation,
the meaning of the two cycle times, and the small-noise qualification.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Cycle:
    r0: float
    q: np.ndarray
    mean: float
    variance: float
    ordinary_mean: float


def cycle_parameters(nu, mu, b):
    """Return r0 and the growth-weighted Exp(r0 + nu + mu) cycle law.

    Scalars broadcast against one-dimensional arrays.  `ordinary_mean` is
    the cycle duration conditional on completing all stages, without the
    additional population-growth weighting.
    """
    arrays = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, dtype=float))
                                   for a in (nu, mu, b)))
    if any(a.ndim != 1 or a.size == 0 for a in arrays):
        raise ValueError("nu, mu, b must be nonempty scalars or 1-D arrays")
    nu, mu, b = arrays
    if (not all(np.all(np.isfinite(a)) for a in arrays)
            or np.any(nu <= 0) or np.any(mu < 0) or np.any(b <= 0)):
        raise ValueError("finite nu > 0, mu >= 0, and b > 0 are required")
    d = nu + mu
    if not np.all(np.isfinite(d)):
        raise ValueError("nu + mu must be finite")
    log_c = np.log(nu) + np.log(b)
    delta = d - d.min()
    # Solve in log(r0 + min(d)); this avoids a root endpoint at log(0).
    with np.errstate(divide="ignore"):
        log_delta = np.log(delta)

    def equation(log_t):
        return np.sum(np.logaddexp(log_t, log_delta) - log_c)

    hi = float(np.mean(log_c))
    if np.all(delta == 0):
        log_t = hi
    else:
        width = 1.0
        while equation(hi - width) > 0:
            width *= 2
        log_t = brentq(equation, hi - width, hi, xtol=1e-13,
                       rtol=4 * np.finfo(float).eps)
    t = np.exp(log_t)
    q = t + delta
    if not np.all(np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("rates outside supported floating-point range")
    return Cycle(float(t - d.min()), q, float(np.sum(1 / q)),
                 float(np.sum(1 / q**2)), float(np.sum(1 / d)))


def cost_ct(nu, mu, b, alpha, sigma):
    """Return an array: O(sigma**2) cost per unit time from EACH host.

    The total second-order cost is cost_ct(...).sum(). All hosts have the
    same coefficient, even for heterogeneous stages. This is not an exact
    finite-sigma formula, except when n=1 (where the cost is exactly zero).
    Hosts must be independent, stationary OU log-hosts of variance one,
    with the same alpha > 0, and h = exp(sigma*z - sigma**2/2).
    """
    alpha, sigma = float(alpha), float(sigma)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    cycle = cycle_parameters(nu, mu, b)
    n, m = len(cycle.q), cycle.mean
    if n == 1 or sigma == 0:
        return np.zeros(n)
    a = alpha / cycle.q
    if np.sum(np.log1p(a)) > 600:
        fraction = 1.0  # exponentially tiny second term
    else:
        # P - 1 = linear + higher. Accumulate positive terms instead of
        # subtracting nearly equal quantities when alpha is very small.
        linear = higher = 0.0
        for ai in a:
            higher = higher * (1 + ai) + linear * ai
            linear += ai
        fraction = higher / (linear + higher)
    value = sigma**2 / (alpha * m**2) * fraction
    return np.full(n, value)
