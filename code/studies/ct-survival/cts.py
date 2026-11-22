"""Second-order continuous-time cost of OU mortality in one cyclic stage.

Derived here; see derivation_cts.md.  A negative cost is a growth gain.
Requires numpy and scipy.  mu contains mean death rates; j is zero-based.
"""

import operator

import numpy as np
from scipy.optimize import brentq


def lineage_parameters(nu, mu, b):
    """Return (r0, a, p), where p[k] = (1/a[k])/sum(1/a)."""
    nu, mu, b = (np.asarray(x, dtype=float) for x in (nu, mu, b))
    if nu.ndim != 1 or not nu.size or mu.shape != nu.shape or b.shape != nu.shape:
        raise ValueError("nu, mu, b must be nonempty one-dimensional arrays of equal length")
    if not all(np.all(np.isfinite(x)) for x in (nu, mu, b)):
        raise ValueError("rates and offspring factors must be finite")
    if np.any(nu <= 0) or np.any(b <= 0):
        raise ValueError("nu and b must be strictly positive")
    d = nu + mu
    log_c = np.log(nu) + np.log(b)
    if len(nu) == 1:
        a = np.exp(log_c)
        return float(a[0] - d[0]), a, np.ones(1)

    # Set a[k] = exp(y) + d[k] - min(d), and solve in y.  This
    # avoids subtracting nearly equal r0 and d at the root boundary.
    dmin = float(np.min(d))
    offset = d - dmin
    log_offset = np.full_like(offset, -np.inf)
    np.log(offset, out=log_offset, where=offset > 0)
    target = float(np.sum(log_c))

    def equation(y):
        return float(np.sum(np.logaddexp(y, log_offset)) - target)

    # mean(log_c) is an upper bound in exact arithmetic.  Add slack:
    # at equal d[k], rounding its residual slightly negative would
    # otherwise make brentq reject a valid root at that endpoint.
    hi = float(np.mean(log_c)) + 1.0
    lo = hi - 1.0
    width = 1.0
    while equation(lo) > 0:
        width *= 2.0
        lo = hi - width
    y = brentq(equation, lo, hi, xtol=1e-13, rtol=1e-14)
    a = np.exp(y) + offset
    r0 = float(np.exp(y) - dmin)
    t = 1.0 / a
    p = t / np.sum(t)
    return r0, a, p


def cost_death(nu, mu, b, alpha, sigma, j):
    """Return sigma**2 times the small-noise cost coefficient per unit time.

    The exact growth cost is this value + o(sigma**2), for fixed alpha>0.
    alpha=0 returns the limit as alpha decreases to zero, not a claim
    about a nonergodic, frozen OU process.  n=1 returns zero exactly.
    """
    alpha, sigma = float(alpha), float(sigma)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    j = operator.index(j)
    _, a, p = lineage_parameters(nu, mu, b)
    if not 0 <= j < len(a):
        raise IndexError("j is outside the stage array")
    if len(a) == 1 or sigma == 0:
        return 0.0

    # U=(1-psi)/alpha, V=(alpha*S_other-1+psi)/alpha**2,
    # psi=product_{k != j}(1+alpha/a[k])**(-1).  This positive
    # recursion avoids cancellation, including in the alpha=0 limit.
    U = V = 0.0
    for k, ak in enumerate(a):
        if k == j:
            continue
        t = 1.0 / ak
        denominator = 1.0 + alpha * t
        V += t * (t + U) / denominator
        U = (U + t) / denominator
    gain_coefficient = p[j] ** 2 * a[j] * V / (1.0 + a[j] * U)
    return float(-sigma ** 2 * gain_coefficient)
