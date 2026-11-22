"""Simulator for a parasite life cycle written as a ring of stages with time-varying rates.

Model (indices mod n):

    x_{k+1}(t+1) = S_{k+1}(t) * x_{k+1}(t) + G_k(t) * x_k(t)

S_k(t) is survival in stage k per time step; G_k(t) is the number of stage-(k+1) parasites
produced per stage-k parasite. An environment object supplies S and G each step. The engine
knows nothing about how they are generated.

Verified against known results by `checks/r0_verification.py` (protocol `protocols/R0-verification.md`).
Do not use it for a new result unless that check passes on the current file.
"""
from __future__ import annotations

import numpy as np


def mean_matrix(s, g):
    """n x n matrix with s on the diagonal and g_k in row k+1, column k (cyclic)."""
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    a = np.diag(s).astype(float)
    for k in range(n):
        a[(k + 1) % n, k] += g[k]
    return a


def perron(a):
    """Dominant eigenvalue lam0, right eigenvector u (sums to one), left eigenvector v (v @ u = 1)."""
    w, vr = np.linalg.eig(a)
    i = int(np.argmax(w.real))
    lam0 = float(w[i].real)
    u = np.abs(vr[:, i].real)
    u /= u.sum()
    wl, vl = np.linalg.eig(a.T)
    j = int(np.argmax(wl.real))
    v = np.abs(vl[:, j].real)
    v /= v @ u
    return lam0, u, v


def elasticities(s, g):
    """lam0 and the elasticity of lam0 to each production rate g_k and each survival s_k."""
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    lam0, u, v = perron(mean_matrix(s, g))
    e_g = np.array([v[(k + 1) % n] * g[k] * u[k] for k in range(n)]) / lam0
    e_s = v * s * u / lam0
    return lam0, e_g, e_s


class LognormalAR1:
    """n independent host series per replicate: h = exp(z - sigma^2/2), z Gaussian AR(1).

    Each h has mean one, log standard deviation sigma and lag-one autocorrelation phi (of z).
    """

    def __init__(self, n, sigma, phi, reps, rng):
        self.sigma, self.phi, self.rng = float(sigma), float(phi), rng
        self.shape = (reps, n)
        self.z = rng.normal(0.0, self.sigma, self.shape) if sigma > 0 else np.zeros(self.shape)
        self.innov = self.sigma * np.sqrt(1.0 - self.phi ** 2)

    def next(self):
        if self.sigma == 0.0:
            return np.ones(self.shape)
        self.z = self.phi * self.z + self.innov * self.rng.normal(0.0, 1.0, self.shape)
        return np.exp(self.z - 0.5 * self.sigma ** 2)


class FluctuatingProduction:
    """Environment in which production follows host abundance: G_k(t) = g_k * h_k(t); S constant.

    `mask` selects which steps fluctuate (default all). The control variate is
    sum_k e_k (h_k - 1) over fluctuating steps, with e_k the elasticity of lam0 to g_k; its mean
    is exactly zero, so subtracting it changes the variance of the estimate and not its mean.
    """

    def __init__(self, s, g, sigma, phi, reps, rng, mask=None, hosts=None):
        self.s = np.asarray(s, float)
        self.g = np.asarray(g, float)
        n = len(self.s)
        self.mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        self.hosts = hosts if hosts is not None else LognormalAR1(n, sigma, phi, reps, rng)
        self.lam0, e_g, _ = elasticities(self.s, self.g)
        self.coef = np.where(self.mask, e_g, 0.0)

    def step(self):
        h = self.hosts.next()
        h = np.where(self.mask[None, :], h, 1.0)
        return self.s[None, :], self.g[None, :] * h, (h - 1.0) @ self.coef


def run(env, n, reps, steps, burn=0, x0=None):
    """Iterate the ring and return per-replicate growth statistics.

    env.step() returns (S, G, cv): S and G broadcastable to (reps, n); cv a zero-mean control
    variate per replicate, or None.

    Returns a dict with
      rate      mean log growth per step after burn-in, per replicate
      rate_cv   the same with the control variate subtracted (equals rate if cv is None)
      total     summed log growth after burn-in, per replicate
      last      log growth in the final step, per replicate
    """
    x = np.full((reps, n), 1.0 / n) if x0 is None else np.broadcast_to(np.asarray(x0, float), (reps, n)).copy()
    x = x / x.sum(axis=1, keepdims=True)
    total = np.zeros(reps)
    cv_sum = np.zeros(reps)
    last = np.zeros(reps)
    for t in range(steps):
        s_t, g_t, cv = env.step()
        x = x * s_t + np.roll(x * g_t, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        last = np.log(tot)
        if t >= burn:
            total += last
            if cv is not None:
                cv_sum += cv
    length = steps - burn
    return {"rate": total / length, "rate_cv": (total - cv_sum) / length, "total": total, "last": last}


def mean_se(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
