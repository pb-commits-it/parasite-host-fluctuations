"""Independent ring life-cycle calculations; times are conditional transfer times.

C uses a_k = nu_k + mu_k. D uses mean holding time 1/(1-s_k),
in updates. Nonlinear functions use relative abundance, equilibrium y=1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def nth_roots(q, n):
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer")
    if q == 0:
        return np.zeros(n, dtype=complex)
    angle = 0.0 if q > 0 else np.pi
    return abs(q) ** (1.0 / n) * np.exp(1j * (angle + 2*np.pi*np.arange(n))/n)


def char_roots_C(n, tau, kappa):
    """All eigenvalues, equal stages, one locally sensed regulated transfer."""
    if tau <= 0 or kappa < 0:
        raise ValueError("tau > 0 and kappa >= 0 required")
    return (n / tau) * (nth_roots(1-kappa, n) - 1)


def dominant_C(n, tau, kappa):
    """One dominant eigenvalue; choose positive imaginary member of a pair."""
    a = n/tau
    if kappa <= 1:
        return complex(a*((1-kappa)**(1/n)-1))
    if n == 1:
        return complex(-a*kappa)
    return a*((kappa-1)**(1/n)*np.exp(1j*np.pi/n)-1)


def critical_equal_C(n):
    return np.inf if n <= 2 else 1 + (1/np.cos(np.pi/n))**n


def local_elasticity(R0, n=1, form="bh"):
    """Symmetric equilibrium elasticity; n=1 for a single regulated edge."""
    if R0 <= 1:
        raise ValueError("R0 must exceed 1")
    L = np.log(R0)/n
    if form == "bh":
        return -np.expm1(-L)
    if form == "ricker":
        return L
    raise ValueError("form must be 'bh' or 'ricker'")


def every_edge_factor(n, R0, form):
    # Direct exponential avoids cancellation for strong Beverton-Holt feedback.
    return np.exp(-np.log(R0)/n) if form == "bh" else 1-local_elasticity(R0, n, form)


def every_roots_C(n, tau, R0, form="bh"):
    b = every_edge_factor(n, R0, form)
    return (n/tau)*(b*np.exp(2j*np.pi*np.arange(n)/n)-1)


def return_rate_every_stage(n, tau, R0, form="bh"):
    """Signed spectral decay rate. Default BH; form='ricker' uses exp(-x/K).

    Positive means exponential decay, negative means instability; assumes the
    symmetric equilibrium in the Ricker case. tau is the mean entire cycle.
    """
    b = every_edge_factor(n, R0, form)
    if b >= 0 or n == 1:
        spectral_part = b
    elif n % 2 == 0:
        spectral_part = -b
    else:
        spectral_part = -b*np.cos(np.pi/n)
    return (n/tau)*(1-spectral_part)


def critical_logR_every_C(n):
    if n == 1:
        return np.inf
    return 2*n if n % 2 == 0 else n*(1+1/np.cos(np.pi/n))


def char_roots_D(n, s, kappa):
    if not 0 <= s < 1:
        raise ValueError("0 <= s < 1 required")
    return s + (1-s)*nth_roots(1-kappa, n)


def radius_D(n, s, kappa):
    d = 1-s
    if kappa <= 1:
        return s+d*(1-kappa)**(1/n)
    r = (kappa-1)**(1/n)
    if n == 1:
        return abs(1-d*kappa)
    return np.sqrt(s*s+d*d*r*r+2*s*d*r*np.cos(np.pi/n))


def negative_gain_radius_boundary(n, s):
    """r at the first unit-circle crossing of s+(1-s) r exp(i*pi/n)."""
    theta = np.pi/n
    return (np.sqrt(1-s*s*np.sin(theta)**2)-s*np.cos(theta))/(1-s)


def critical_equal_D(n, s):
    return 1+negative_gain_radius_boundary(n, s)**n


def every_roots_D(n, s, R0, form="bh"):
    b = every_edge_factor(n, R0, form)
    return s+(1-s)*b*np.exp(2j*np.pi*np.arange(n)/n)


def radius_every_D(n, s, R0, form="bh"):
    b = every_edge_factor(n, R0, form)
    if b >= 0:
        return s+(1-s)*b
    if n == 1:
        return abs(s+(1-s)*b)
    if n % 2 == 0:
        return s+(1-s)*abs(b)
    r = abs(b)
    return np.sqrt(s*s+(1-s)**2*r*r+2*s*(1-s)*r*np.cos(np.pi/n))


def critical_logR_every_D(n, s):
    return 2*n if n % 2 == 0 else n*(1+negative_gain_radius_boundary(n, s))


def decay_rate_D(radius):
    return np.inf if radius == 0 else -np.log(radius)


def jacobian_C(a, kappa=0., sensor=None, all_kappas=None):
    """Relative-coordinate Jacobian, sensor is a zero-based stage index."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    J = -np.diag(a)
    for k in range(n):
        previous = (k-1) % n
        factor = 1 if all_kappas is None else 1-all_kappas[previous]
        J[k, previous] += a[k]*factor
    if all_kappas is None:
        J[0, n-1 if sensor is None else sensor] -= a[0]*kappa
    return J


def jacobian_D(s, kappa=0., sensor=None, all_kappas=None):
    s = np.asarray(s, dtype=float)
    # C and D normalized Jacobians obey J_D = I + J_C(a=1-s).
    return np.eye(len(s))+jacobian_C(1-s, kappa, sensor, all_kappas)


def characteristic_terms_C(lam, a, kappa=0., sensor=None, all_kappas=None):
    a = np.asarray(a)
    w = 1+lam/a
    if all_kappas is not None:
        return np.array([np.prod(w), -np.prod(1-np.asarray(all_kappas))])
    j = len(a)-1 if sensor is None else sensor
    return np.array([np.prod(w), -1, kappa*np.prod(w[j+1:])])


def characteristic_terms_D(z, s, kappa=0., sensor=None, all_kappas=None):
    return characteristic_terms_C(z-1, 1-np.asarray(s), kappa, sensor, all_kappas)


def critical_kappa_C(times):
    """Exact first crossing, arbitrary positive conditional mean durations."""
    times = np.asarray(times, dtype=float)
    if np.any(times <= 0):
        raise ValueError("positive durations required")
    if len(times) <= 2:
        return np.inf
    scaled = times/times.mean()
    hi = 1.
    phase = lambda w: np.arctan(w*scaled).sum()-np.pi
    while phase(hi) < 0:
        hi *= 2
    w = brentq(phase, 0, hi, xtol=1e-13)
    return 1+np.exp(.5*np.log1p((w*scaled)**2).sum())


def critical_kappa_D(times):
    """Exact first unit-circle crossing; durations are in discrete updates."""
    times = np.asarray(times, dtype=float)
    if np.any(times < 1):
        raise ValueError("discrete mean durations must be >= 1")
    if len(times) == 1:
        return 2*times[0]
    s = 1-1/times
    phase = lambda w: np.angle(np.exp(1j*w)-s).sum()-np.pi
    w = brentq(phase, 0, np.pi, xtol=1e-13)
    return 1+np.exp(np.log(np.abs(np.exp(1j*w)-s)*times).sum())


def undercomp_rate_C(times, kappa):
    times = np.asarray(times, dtype=float)
    if not 0 < kappa < 1:
        raise ValueError("0 < kappa < 1 required")
    return brentq(lambda r: np.prod(1-r*times)-(1-kappa),
                  0, 1/times.max(), xtol=1e-14)


def regulation_ratio(y, kappa, form):
    """f(y*x*)/f(x*), with equilibrium y=1."""
    if form == "ricker":
        return np.exp(-kappa*(y-1))
    if form == "bh":
        return 1/(1+kappa*(y-1))
    raise ValueError(form)


def nonlinear_C(t, y, a, kappa, form="ricker", sensor=None, every=False):
    a = np.asarray(a)
    incoming = np.roll(y, 1).copy()
    if every:
        incoming *= regulation_ratio(np.roll(y, 1), kappa, form)
    else:
        j = len(y)-1 if sensor is None else sensor
        incoming[0] *= regulation_ratio(y[j], kappa, form)
    return a*(incoming-y)


def nonlinear_D(y, s, kappa, form="ricker", sensor=None, every=False):
    return y+nonlinear_C(0., y, 1-np.asarray(s), kappa, form, sensor, every)
