"""Equal-stage spectra for the two cyclic life-cycle models.

Derived here; see derivation_shared.md.  Thresholds are in L = log(R0).
An instability threshold means a positive spectral abscissa for L strictly
above it.  At a finite threshold the linearisation is nonhyperbolic.
"""

import math
from numbers import Integral

import numpy as np


def _n(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    return int(n)


def _parameters(n, lnR0, a):
    n = _n(n)
    lnR0, a = float(lnR0), float(a)
    if not math.isfinite(lnR0) or lnR0 < 0:
        raise ValueError("lnR0 must be finite and nonnegative")
    if not math.isfinite(a) or a <= 0:
        raise ValueError("a must be finite and positive")
    return n, lnR0, a


def threshold_A(n):
    """Threshold in log(R0) for one regulated transfer (derived here)."""
    n = _n(n)
    if n <= 2:
        return math.inf
    return 1.0 + math.exp(-n * math.log(math.cos(math.pi / n)))


def threshold_B(n):
    """Threshold for the symmetric, equally regulated equilibrium (derived)."""
    n = _n(n)
    if n == 1:
        return math.inf
    if n % 2 == 0:
        return float(2 * n)
    return n * (1.0 + 1.0 / math.cos(math.pi / n))


def eigenvalues_A(n, lnR0, a):
    """All eigenvalues, including multiplicities, for model A (derived)."""
    n, lnR0, a = _parameters(n, lnR0, a)
    if n == 1:
        return np.array([-a * lnR0], dtype=complex)
    p = 1.0 - lnR0
    if p == 0:
        return np.full(n, -a, dtype=complex)
    radius = abs(p) ** (1.0 / n)
    angles = (2.0 * np.pi * np.arange(n) + (np.pi if p < 0 else 0.0)) / n
    return a * (radius * np.exp(1j * angles) - 1.0)


def eigenvalues_B(n, lnR0, a):
    """All eigenvalues at B's symmetric equilibrium (derived here)."""
    n, lnR0, a = _parameters(n, lnR0, a)
    angles = 2.0 * np.pi * np.arange(n) / n
    return a * ((1.0 - lnR0 / n) * np.exp(1j * angles) - 1.0)
