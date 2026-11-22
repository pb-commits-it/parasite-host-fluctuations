"""Environment with several alternative hosts at each stage (rung R3).

Transfer out of stage k is g_k times a weighted mean of the multipliers of M hosts:

    G_k(t) = g_k * sum_m pi_m h_{k,m}(t),      h_{k,m} = exp(z - sigma_m^2/2)

with z = sigma_m * ( sqrt(rho) * c_k + sqrt(1 - rho) * e_{k,m} ), where c_k (common to the hosts of stage k) and
e_{k,m} (specific to each host) are independent stationary Gaussian AR(1) series of unit variance, with
autocorrelations phi_c and phi_m. So each h has mean one and log standard deviation sigma_m, and the log abundances
of two hosts of the same stage have correlation rho. Survival is constant.

Verified by check K0 of `checks/r3_alternative_hosts.py` (protocol `protocols/R3-alternative-hosts.md`).
"""
from __future__ import annotations

import numpy as np

from . import ring


class AlternativeHosts:
    def __init__(self, s, g, shares, sigma, phi, reps, rng, rho=0.0, phi_c=0.0):
        self.s, self.g = np.asarray(s, float), np.asarray(g, float)
        n = len(self.s)
        self.pi = np.asarray(shares, float)
        m = len(self.pi)
        self.sigma = np.broadcast_to(np.asarray(sigma, float), (m,)).copy()
        self.phi = np.broadcast_to(np.asarray(phi, float), (m,)).copy()
        self.rho, self.phi_c, self.rng = float(rho), float(phi_c), rng
        self.shape = (reps, n, m)
        self.e = rng.normal(size=self.shape)
        self.c = rng.normal(size=(reps, n))
        self.lam0, self.e_g, _ = ring.elasticities(self.s, self.g)

    def step(self):
        self.e = self.phi[None, None, :] * self.e + np.sqrt(1 - self.phi ** 2)[None, None, :] * self.rng.normal(size=self.shape)
        self.c = self.phi_c * self.c + np.sqrt(1 - self.phi_c ** 2) * self.rng.normal(size=self.c.shape)
        z = self.sigma[None, None, :] * (np.sqrt(self.rho) * self.c[:, :, None] + np.sqrt(1 - self.rho) * self.e)
        h = np.exp(z - 0.5 * self.sigma[None, None, :] ** 2)
        agg = h @ self.pi                                   # (reps, n): weighted mean over the hosts of each stage
        return self.s[None, :], self.g[None, :] * agg, (agg - 1.0) @ self.e_g
