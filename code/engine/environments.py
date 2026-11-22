"""Environments for `engine/ring.py` beyond the one tested in rung R0.

`FluctuatingRing` lets transfer and survival both fluctuate, each stage with its own noise level and
autocorrelation:

    G_k(t) = g_k * h_k(t),      S_k(t) = s_k * u_k(t)

Every h_k and u_k is an independent lognormal AR(1) series with mean one. A log standard deviation of
zero switches a series off.

Verified against known results by `checks/r0b_survival_noise.py` (protocol `protocols/R0b-survival-noise.md`).
Do not use it for a new result unless that check passes on the current file.
"""
from __future__ import annotations

import numpy as np

from . import ring


class LognormalAR1Vec:
    """One independent series per stage and replicate: exp(z - sigma_k^2/2), z Gaussian AR(1).

    sigma and phi are arrays of length n (log standard deviation and lag-one autocorrelation of z).
    """

    def __init__(self, sigma, phi, reps, rng):
        self.sigma = np.asarray(sigma, float)
        self.phi = np.broadcast_to(np.asarray(phi, float), self.sigma.shape).copy()
        self.rng = rng
        self.shape = (reps, len(self.sigma))
        self.z = rng.normal(0.0, 1.0, self.shape) * self.sigma[None, :]
        self.innov = self.sigma * np.sqrt(1.0 - self.phi ** 2)
        self.active = bool(np.any(self.sigma > 0))

    def next(self):
        if not self.active:
            return np.ones(self.shape)
        self.z = self.phi[None, :] * self.z + self.innov[None, :] * self.rng.normal(0.0, 1.0, self.shape)
        return np.exp(self.z - 0.5 * self.sigma[None, :] ** 2)


class FluctuatingRing:
    """Transfer and survival both fluctuate. Control variate: sum of elasticity * (multiplier - 1).

    s, g                mean staying and mean transfer per stage (g_k is from stage k to k+1)
    sig_g, phi_g        log standard deviation and autocorrelation of the transfer multipliers h_k
    sig_s, phi_s        the same for the survival multipliers u_k
    """

    def __init__(self, s, g, sig_g, phi_g, sig_s, phi_s, reps, rng):
        self.s = np.asarray(s, float)
        self.g = np.asarray(g, float)
        n = len(self.s)
        zeros = np.zeros(n)
        self.h = LognormalAR1Vec(np.broadcast_to(np.asarray(sig_g, float), (n,)) + zeros, phi_g, reps, rng)
        self.u = LognormalAR1Vec(np.broadcast_to(np.asarray(sig_s, float), (n,)) + zeros, phi_s, reps, rng)
        self.lam0, self.e_g, self.e_s = ring.elasticities(self.s, self.g)

    def step(self):
        h = self.h.next()
        u = self.u.next()
        cv = (h - 1.0) @ self.e_g + (u - 1.0) @ self.e_s
        return self.s[None, :] * u, self.g[None, :] * h, cv
