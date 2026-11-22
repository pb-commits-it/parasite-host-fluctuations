"""Second-order fertility-noise costs for primitive Leslie life tables.

The returned coefficient C means log(lambda) - stochastic_growth =
sigma**2 * C + o(sigma**2), with fixed life table and fixed |phi| < 1.
Dependencies: NumPy and SciPy. See derivation_leslie.md for the derivation.
"""

from functools import reduce
from math import gcd

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def demographic_distribution(m, p, *, require_primitive=True):
    """Return (lambda, q), where q[a-1] = l_a m_a lambda**(-a).

    m has length d; p has length d-1. The final fertility must be positive.
    Lambda is obtained from Euler--Lotka, independently of an eigensolver.
    """
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)
    if m.ndim != 1 or len(m) == 0 or p.ndim != 1 or len(p) != len(m) - 1:
        raise ValueError("m must have length d >= 1, and p length d-1")
    if not np.all(np.isfinite(m)) or np.any(m < 0) or m[-1] <= 0:
        raise ValueError("fertilities must be finite, nonnegative, with m[-1] > 0")
    if not np.all(np.isfinite(p)) or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("survivals must be finite and in (0, 1]")
    ages = np.arange(1, len(m) + 1)
    support = ages[m > 0]
    if require_primitive and reduce(gcd, map(int, support)) != 1:
        raise ValueError("nonprimitive life table: the small-noise formula does not apply")
    log_l = np.r_[0.0, np.cumsum(np.log(p))]
    positive = m > 0
    log_lm = log_l[positive] + np.log(m[positive])
    repro_ages = ages[positive]

    def equation(log_lambda):
        return logsumexp(log_lm - repro_ages * log_lambda)

    lower, upper = -1.0, 1.0
    while equation(lower) < 0:
        lower *= 2
    while equation(upper) > 0:
        upper *= 2
    log_lambda = brentq(equation, lower, upper, xtol=5e-15, rtol=1e-15)
    q = np.zeros(len(m))
    q[positive] = np.exp(log_lm - repro_ages * log_lambda)
    q /= q.sum()  # Remove only root-solver roundoff.
    return float(np.exp(log_lambda)), q


def _coefficient_from_q(q, phi, independent=False):
    """Pole-free polynomial form; internally also usable at phi = 1.

    D(s) = sum q_a sum_{j=0}^{a-1} s**j,
    H(s) = (T-D(s))/(1-s), evaluated as a finite polynomial.
    These avoid subtracting divergent terms when phi is close to one.
    """
    q = np.asarray(q, dtype=np.longdouble)
    s = np.longdouble(phi)
    ages = np.arange(1, len(q) + 1, dtype=np.longdouble)
    T = ages @ q
    tails = np.cumsum(q[::-1])[::-1]
    D = np.polynomial.polynomial.polyval(s, tails)
    h_coeff = np.cumsum(tails[:0:-1])[::-1]
    H = np.polynomial.polynomial.polyval(s, h_coeff) if len(h_coeff) else 0
    subtraction = (T + (1 + s) * H) / (2 * T * T * D)
    if not independent:
        return float(1 / T - subtraction)
    q_squared = q * q
    S2 = q_squared.sum()
    tails_squared = np.cumsum(q_squared[::-1])[::-1]
    D2 = np.polynomial.polynomial.polyval(s, tails_squared)
    return float(D2 / (T * D) - S2 * subtraction)


def _validate_phi(phi):
    if not np.isfinite(phi) or not -1 < phi < 1:
        raise ValueError("phi must be finite and strictly between -1 and 1")


def cost_common_driver(m, p, phi):
    """Return second-order cost per sigma**2 for one common fertility factor.

    With T=sum(a*q_a) and G=sum(q_a*phi**a), the result is
      (1+phi)/(2*T**2*(1-phi)) - G/(T*(1-G)).
    Negative values indicate a growth benefit relative to the mean matrix.
    A periodic life table raises ValueError, including semelparity at d > 1.
    """
    _validate_phi(phi)
    _, q = demographic_distribution(m, p)
    return _coefficient_from_q(q, phi)


def cost_independent_drivers(m, p, phi):
    """Return second-order cost with mutually independent age-specific factors.

    S2=sum(q_a**2), R=sum(q_a**2*phi**a), and T,G as above:
      S2*(1+phi)/(2*T**2*(1-phi)) - R/(T*(1-G)).
    """
    _validate_phi(phi)
    _, q = demographic_distribution(m, p)
    return _coefficient_from_q(q, phi, independent=True)


def cost_persistent_limit(m, p, *, independent=False):
    """Limit of the second-order coefficient as phi increases to 1.

    This is not a claim about an ergodic environment at phi exactly 1.
    """
    _, q = demographic_distribution(m, p)
    ages = np.arange(1, len(q) + 1)
    T = ages @ q
    V = (ages - T) ** 2 @ q
    if independent:
        S2 = q @ q
        U = ages @ (q * q)
        return float((2 * T * U - S2 * (T * T + V)) / (2 * T**3))
    return float((T * T - V) / (2 * T**3))


def cost_semelparous(age):
    """Exact cost per sigma**2 for reproduction only at integer age >= 1."""
    if not isinstance(age, (int, np.integer)) or age < 1:
        raise ValueError("age must be a positive integer")
    return 1 / (2 * age)
