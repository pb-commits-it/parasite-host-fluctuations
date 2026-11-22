#!/usr/bin/env python3
"""Independent check of two notes of closed forms that were each derived once, by one model.

  Note B ("encounters and transmission routes"), formulas (1) to (17).
  Note C ("dispersal"): F = q S - (1 - q) H for identical island patches, the long-stay coefficient K,
  the boundaries, the two-patch log cosh formula, the R0 ratios and the bound.

Claim tested: every closed form equals the general second-order coefficient computed here on the
model's own matrix; every quoted boundary and number is reproduced by code written here; every
statement said to hold at any noise survives a search for a counterexample.  Nothing in this file is
taken from the notes' own script.  The notes are read for the model definitions and the formulas only.

Own tools (section H checks them against known answers before they are used):
  (a) second_order(): the coefficient
        C = sum_ij { c_i c_j rho_ij(0)/2 - sum_{m>=0} rho_ij(m+1) w^T E_i T^m Q E_j v }
      for any primitive non-negative matrix, any perturbation matrices E_i and any covariances
      rho_ij(l) = E xi_i(t) xi_j(t-l), by the lag sum and by the resolvent (I - phi_i T)^(-1);
  (b) floquet_root(): the exact loss in a periodic environment (product over one period,
      spectral radius, root in y);
  (c) run_growth() with find_root(), batch_growth(), spatial_growth(): products of the actual
      random matrices, with standard errors, for Gaussian AR(1) drivers (antithetic pairs) and for
      bounded finite-state Markov environments; the time mean of (h - 1), whose expectation is
      exactly zero, is subtracted as a control variate;
  (d) Ez(), Ez2(): Gaussian expectations by the trapezoid rule (and Gauss-Hermite as a cross-check)
      for the models that reduce to one stage.

The loss D: the birth transfer(s) are multiplied by e^y; D is the y at which the growth rate is zero
when the matrix of mean rates is at its threshold.  D = sigma^2 C / b to second order, b = w^T R v,
R the birth part of the mean matrix.

TOLERANCES, fixed before the first run:
  TOL_ALG   = 1e-8   closed form against general coefficient, relative to the larger of the two
                     values (absolute floor 1e-12); where the terms of a coefficient nearly cancel,
                     relative to 1e-3 of the sum of the magnitudes of its terms
  TOL_NUM   = 1e-9   a quoted number against my number, relative; a quoted number with fewer
                     digits is compared to the digits given
  TOL_FLOQ  = 1e-3   second-order value against exact periodic loss at small amplitude, relative
                     (after one Richardson step in the amplitude)
  small-noise stochastic check: |D_sim - sigma^2 K| <= 4 SE + 5% of |sigma^2 K|
  a simulated number against a formula with no expansion: 4 SE; a simulated number of mine
  against a simulated number of the note: 4 combined SE
  a quoted number that is a small difference of two roots (the "one side each" examples), or
  that needs a two-dimensional quadrature: relative 1e-6 (the roots themselves: TOL_NUM, or
  1e-8 for the two-dimensional quadrature); the agreement actually reached is printed
  searches for a counterexample: a case counts against the claim when it is on the wrong side
  by more than 4 SE (exact periodic cases: by more than 1e-9); such a case is rerun longer
  with a new seed before it is reported.

Run:  python3 -I checks/verify_mechanisms_routes_dispersal.py
      (writes to standard output; the main process and two workers; 215 to 245 s measured on a
      busy ten-core machine, about 620 s of processor time)
      --only NAME[,NAME]  runs named sections; --list lists them.
Output: results/verify-mechanisms-routes-dispersal_20261114.txt.  Failed outputs caused by this
harness are kept: ..._attempt1_... (single sections, while the script was being written) and
..._attempt2_... (the first complete run, one check of my own mis-specified).
Report: notes/mechanisms-verification.md.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import math
import time
import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter

TOL_ALG = 1e-8
ABS_FLOOR = 1e-12
TOL_NUM = 1e-9
TOL_FLOQ = 1e-3
NSE = 4.0
SMALL_REL = 0.05


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------
class Rep:
    def __init__(self, title):
        self.lines = ["", "=" * 100, title, "=" * 100]
        self.checks = []
        self.t0 = time.time()

    def p(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def check(self, key, ok, text):
        self.checks.append((key, bool(ok), text))
        self.p(("  PASS  " if ok else "  FAIL  ") + key + ": " + text)

    def done(self):
        self.p("  [section time %.1f s]" % (time.time() - self.t0))
        return "\n".join(self.lines), self.checks


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def close_alg(a, b):
    return abs(a - b) <= TOL_ALG * max(abs(a), abs(b)) + ABS_FLOOR


def close_num(mine, quoted, tol=TOL_NUM):
    return abs(mine - quoted) <= tol * max(abs(quoted), 1e-300)


# --------------------------------------------------------------------------------------------
# tool (a): the general second-order coefficient
# --------------------------------------------------------------------------------------------
def perron(A):
    """Growth factor, right and left Perron vectors (w^T v = 1) of a non-negative matrix."""
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.real))
    lam = float(ev[i].real)
    v = np.abs(V[:, i].real)
    ev2, W = np.linalg.eig(A.T)
    j = int(np.argmax(ev2.real))
    w = np.abs(W[:, j].real)
    v = v / v.sum()
    w = w / (w @ v)
    return lam, v, w


def perron_at_one(A):
    """Perron vectors of a matrix whose growth factor is exactly one, by linear solves only
    (so that it can be used with a complex step)."""
    n = A.shape[0]
    M = A - np.eye(n)
    rhs = np.zeros(n, dtype=A.dtype)
    rhs[-1] = 1.0
    M1 = M.copy()
    M1[-1, :] = 1.0
    v = np.linalg.solve(M1, rhs)
    M2 = M.T.copy()
    M2[-1, :] = 1.0
    w = np.linalg.solve(M2, rhs)
    w = w / (w @ v)
    return v, w


def second_order(A, Es, phis=None, R=None, method="resolvent", rho=None, vw=None, maxlag=400000):
    """C = sum_ij { c_i c_j rho_ij(0)/2 - sum_{m>=0} rho_ij(m+1) w^T E_i T^m Q E_j v }.

    Drivers: either AR(1) with coefficients phis[i] and contemporaneous covariance R[i][j]
    (then rho_ij(l) = phis[i]^l R[i][j]: the later factor i carries the memory), or any
    function rho(i, j, l) = E xi_i(t) xi_j(t - l), l >= 0 (lag sum only).
    Returns C, growth factor, list of c_i, v, w.  A is divided by its growth factor first."""
    if vw is None:
        lam, v, w = perron(A)
    else:
        lam, v, w = vw
    n = A.shape[0]
    M = A / lam
    Es = [E / lam for E in Es]
    I = len(Es)
    Q = np.eye(n) - np.outer(v, w)
    T = M @ Q
    c = [w @ E @ v for E in Es]
    if rho is None:
        R = np.asarray(R)

        def rho0(i, j):
            return R[i, j]
    else:
        def rho0(i, j):
            return rho(i, j, 0)
    C = 0.0
    for i in range(I):
        wEi = w @ Es[i]
        for j in range(I):
            r0 = rho0(i, j)
            QEjv = Q @ (Es[j] @ v)
            if rho is None:
                if r0 == 0:
                    continue
                f = phis[i]
                if method == "resolvent":
                    lagsum = f * r0 * (wEi @ np.linalg.solve(np.eye(n) - f * T, QEjv))
                else:
                    lagsum = 0.0
                    vec = QEjv.copy()
                    fm = f
                    for m in range(maxlag):
                        lagsum += fm * r0 * (wEi @ vec)
                        vec = T @ vec
                        fm *= f
                        if abs(fm) * np.abs(vec).sum() < 1e-18:
                            break
            else:
                lagsum = 0.0
                vec = QEjv.copy()
                for m in range(maxlag):
                    lagsum += rho(i, j, m + 1) * (wEi @ vec)
                    vec = T @ vec
                    if np.abs(vec).sum() < 1e-18:
                        break
            C = C + c[i] * c[j] * r0 / 2 - lagsum
    return C, lam, c, v, w


class Model:
    """A(t, y) = C0 + e^y C1 + sum_i h_i(t) (B0_i + e^y B1_i).  The h_i are mean-one multipliers;
    everything multiplied by e^y is a birth.  Mean matrix: y = 0, h = 1."""

    def __init__(self, C0, C1, B0, B1):
        self.C0 = np.array(C0, float)
        self.C1 = np.array(C1, float)
        self.B0 = np.array(B0, float)
        self.B1 = np.array(B1, float)
        self.n = self.C0.shape[0]
        self.I = self.B0.shape[0]

    def mat(self, y, h):
        ey = math.exp(y)
        return self.C0 + ey * self.C1 + np.tensordot(np.asarray(h, float), self.B0 + ey * self.B1, axes=(0, 0))

    def mean(self, y=0.0):
        return self.mat(y, np.ones(self.I))

    def birth(self):
        return self.C1 + self.B1.sum(0)

    def E(self, i):
        return self.B0[i] + self.B1[i]


def K_model(model, kappa, lat, phis_lat, corr_lat, method="resolvent"):
    """Second-order coefficient of the loss D (per unit sigma^2) of a Model whose multiplier i has
    first-order part kappa[i] * sigma * z_{lat[i]}, the z_l being AR(1) with coefficients phis_lat
    and contemporaneous correlation corr_lat."""
    A = model.mean()
    Es = [kappa[i] * model.E(i) for i in range(model.I)]
    I = model.I
    R = np.array([[corr_lat[lat[i]][lat[j]] for j in range(I)] for i in range(I)], float)
    phis = [phis_lat[lat[i]] for i in range(I)]
    C, lam, c, v, w = second_order(A, Es, phis, R, method=method)
    b = (w @ model.birth() @ v) / lam
    return C / b, lam, b


# --------------------------------------------------------------------------------------------
# tool (b): exact loss in a periodic environment
# --------------------------------------------------------------------------------------------
def spec_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def floquet_growth(model, hseq, y):
    """log spectral radius of the product over one period, per step.  hseq: (N, I)."""
    M = np.eye(model.n)
    for t in range(len(hseq)):
        M = model.mat(y, hseq[t]) @ M
    return math.log(spec_radius(M)) / len(hseq)


def floquet_root(model, hseq, lo=-3.0, hi=3.0):
    f = lambda y: floquet_growth(model, hseq, y)
    flo, fhi = f(lo), f(hi)
    k = 0
    while flo > 0 and k < 20:
        lo -= 3.0
        flo = f(lo)
        k += 1
    while fhi < 0 and k < 40:
        hi += 3.0
        fhi = f(hi)
        k += 1
    return brentq(f, lo, hi, xtol=1e-15, rtol=8.9e-16, maxiter=300)


# --------------------------------------------------------------------------------------------
# tool (d): Gaussian expectations
# --------------------------------------------------------------------------------------------
_ZG = np.arange(-14.0, 14.0 + 1e-9, 0.02)
_WG = np.exp(-0.5 * _ZG * _ZG) / math.sqrt(2 * math.pi) * 0.02


def Ez(f):
    """E f(Z), Z standard normal, trapezoid rule on [-14, 14] with step 0.02 (the integrands used
    are analytic near the real line, so the rule converges exponentially)."""
    return float(np.sum(_WG * f(_ZG)))


def Ez_gh(f, n=300):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return float(np.sum(w * f(x)) / math.sqrt(2 * math.pi))


_Z2 = np.arange(-12.0, 12.0 + 1e-9, 0.04)
_W2 = np.exp(-0.5 * _Z2 * _Z2) / math.sqrt(2 * math.pi) * 0.04


def Ez2(f, z=_Z2, w=_W2):
    """E f(Z1, Z2) for independent standard normals, tensor trapezoid rule."""
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    return float(np.sum(np.outer(w, w) * f(Z1, Z2)))


def scalar_root(mult, s, lo=-30.0, hi=30.0, E=Ez):
    """y with E log[s + (1 - s) e^y mult(Z)] = 0."""
    return brentq(lambda y: E(lambda z: np.log(s + (1 - s) * math.exp(y) * mult(z))), lo, hi,
                  xtol=1e-15, rtol=8.9e-16, maxiter=300)


# --------------------------------------------------------------------------------------------
# models of note B
# --------------------------------------------------------------------------------------------
def t_fun(phi, tau):
    return phi / (phi + (1 - phi) * tau)


def A_phi(phi):
    return (1 + phi) / (2 * (1 - phi))


def ring_parts(taus, gauge=None):
    taus = np.asarray(taus, float)
    s = 1 - 1 / taus
    g = 1 / taus
    if gauge is not None:
        g = g * np.asarray(gauge)
    return s, g


def model_ring(taus, mult, contact=None, gauge=None, birth=None, nmult=None):
    """Ring at the threshold of its mean matrix.  Stage k stays with s_k = 1 - 1/tau_k; the transfer
    k -> k+1 has mean g_k = 1/tau_k (times a gauge factor; the gauge factors have product one).
    mult: {k: i}: transfer k is multiplied by the multiplier h_i.  contact = (k, d, i): stage k is
    contact-limited with death fraction d: its transfer is g_k h_i and its staying rate is
    s_k - c (h_i - 1), c = (1 - d)(1 - s_k).  The birth is the last transfer unless given."""
    taus = np.asarray(taus, float)
    n = len(taus)
    s, g = ring_parts(taus, gauge)
    if birth is None:
        birth = n - 1
    I = nmult if nmult is not None else (max(list(mult.values()) + ([contact[2]] if contact else [])) + 1)
    C0 = np.zeros((n, n))
    C1 = np.zeros((n, n))
    B0 = np.zeros((I, n, n))
    B1 = np.zeros((I, n, n))
    for k in range(n):
        C0[k, k] += s[k]
        to = (k + 1) % n
        isb = (k == birth)
        if k in mult or (contact is not None and contact[0] == k):
            i = mult[k] if k in mult else contact[2]
            (B1 if isb else B0)[i, to, k] += g[k]
        else:
            (C1 if isb else C0)[to, k] += g[k]
    if contact is not None:
        k, d, i = contact
        c = (1 - d) * (1 - s[k])
        C0[k, k] += c
        B0[i, k, k] += -c
    return Model(C0, C1, B0, B1)


def Kg_closed(taus, phi):
    """Note B (2): K_g = A_phi/G - P/(1 - P)."""
    taus = np.asarray(taus, float)
    G = taus.sum()
    P = np.prod(t_fun(phi, taus))
    return A_phi(phi) / G - P / (1 - P)


def Lk_closed(taus, k, phi):
    taus = np.asarray(taus, float)
    P = np.prod(t_fun(phi, taus))
    return (t_fun(phi, taus[k]) - P) / (1 - P)


def Kc_closed(taus, k, d, phi):
    """Note B (2): K_c = d^2 K_g + d (1 - d) L_k."""
    return d * d * Kg_closed(taus, phi) + d * (1 - d) * Lk_closed(taus, k, phi)


def dKc_db_closed(taus_other, k_other, d, phi, b):
    """Note B (4).  taus_other: the stays of the stages other than the parking stage;
    k_other: position of the contact stage in that list."""
    taus_other = np.asarray(taus_other, float)
    S = taus_other.sum()
    theta = phi / (1 - phi)
    U = theta * np.prod(t_fun(phi, taus_other))
    alpha = (1 - d) * t_fun(phi, taus_other[k_other])
    return -d * d * (theta + 0.5) / (b + S) ** 2 + d * U * (1 - alpha) / (b + theta - U) ** 2


def K6_closed(taus, a, phi):
    """Note B (6): one driver on several transfers with signed elasticities a_j (transfer j leaves
    stage j).  L_ij = product of t over the stages j+1, ..., i (a full circuit if i = j)."""
    taus = np.asarray(taus, float)
    n = len(taus)
    G = taus.sum()
    t = t_fun(phi, taus)
    P = np.prod(t)
    tot = 0.0
    for i in range(n):
        for j in range(n):
            if a[i] == 0 or a[j] == 0:
                continue
            if i == j:
                L = P
            else:
                L = 1.0
                k = (j + 1) % n
                while True:
                    L *= t[k]
                    if k == i:
                        break
                    k = (k + 1) % n
            tot += a[i] * a[j] * L
    return A_phi(phi) / G * (sum(a)) ** 2 - tot / (1 - P)


def model_parallel(tau0, taus, w, gauge=None):
    """Central stage 0 (stay tau0) enters host j (stage j, stay taus[j-1]) with mean transfer
    p0 w_j times the multiplier h_{j-1}; host j returns at rate 1/tau_j; all returns are births."""
    taus = np.asarray(taus, float)
    J = len(taus)
    n = J + 1
    s0, p0 = 1 - 1 / tau0, 1 / tau0
    C0 = np.zeros((n, n))
    C1 = np.zeros((n, n))
    B0 = np.zeros((J, n, n))
    B1 = np.zeros((J, n, n))
    C0[0, 0] = s0
    for j in range(J):
        gj = 1.0 if gauge is None else gauge[j]
        C0[j + 1, j + 1] = 1 - 1 / taus[j]
        B0[j, j + 1, 0] = p0 * w[j] * gj
        C1[0, j + 1] = (1 / taus[j]) / gj
    return Model(C0, C1, B0, B1)


def Kpar_closed(tau0, taus, w, kappa, R, phi):
    """Note B (8), (9)."""
    taus = np.asarray(taus, float)
    w = np.asarray(w, float)
    u = w * np.asarray(kappa, float)
    R = np.asarray(R, float)
    V = u @ R @ u
    G = tau0 + (w * taus).sum()
    if phi == 0:
        return V / (2 * G)
    t0 = t_fun(phi, tau0)
    t = t_fun(phi, taus)
    P = t0 * (w * t).sum()
    return A_phi(phi) * V / G - t0 / (1 - P) * sum(u[j] * t[j] * (R[:, j] @ u) for j in range(len(u)))


def Kpar_closed_multi(tau0, taus, w, kappa, R, phis):
    """Note B (10)."""
    taus = np.asarray(taus, float)
    w = np.asarray(w, float)
    G = tau0 + (w * taus).sum()
    J = len(taus)
    tot = 0.0
    for i in range(J):
        ph = phis[i]
        t0 = t_fun(ph, tau0)
        t = t_fun(ph, taus)
        den = 1 - t0 * (w * t).sum()
        for j in range(J):
            tot += R[i][j] * w[i] * w[j] * kappa[i] * kappa[j] * (A_phi(ph) / G - t0 * t[j] / den)
    return tot


def model_optional(tau0, tau1, taup, f, q, equal_R0=True):
    """Stage 0 -> 1 is the noisy entrance.  A fraction f of the exits from stage 1 goes through a
    steady stage 2 (stay taup, survival q), the rest returns directly; all returns are births.
    equal_R0: the returns are divided by Y = 1 - f + f q."""
    Y = (1 - f + f * q) if equal_R0 else 1.0
    n = 3
    C0 = np.zeros((n, n))
    C1 = np.zeros((n, n))
    B0 = np.zeros((1, n, n))
    B1 = np.zeros((1, n, n))
    C0[0, 0] = 1 - 1 / tau0
    C0[1, 1] = 1 - 1 / tau1
    C0[2, 2] = 1 - 1 / taup
    B0[0, 1, 0] = 1 / tau0
    C1[0, 1] = (1 / tau1) * (1 - f) / Y
    C0[2, 1] = (1 / tau1) * f * q
    C1[0, 2] = (1 / taup) / Y
    return Model(C0, C1, B0, B1)


def Kopt_closed(tau0, tau1, taup, f, q, phi):
    """Note B (12)."""
    Y = 1 - f + f * q
    al = f * q / Y
    G = tau0 + tau1 + al * taup
    P = t_fun(phi, tau0) * t_fun(phi, tau1) * ((1 - al) + al * t_fun(phi, taup))
    return A_phi(phi) / G - P / (1 - P)


def model_bypass(tau0, tau1, f, q, equal_R0=True):
    """Central stage 0.  A fraction f of its exits is a direct birth of relative yield q (back into
    stage 0); the rest enters the noisy host (stage 1), which returns (a birth)."""
    Y = (1 - f + f * q) if equal_R0 else 1.0
    n = 2
    C0 = np.zeros((n, n))
    C1 = np.zeros((n, n))
    B0 = np.zeros((1, n, n))
    B1 = np.zeros((1, n, n))
    C0[0, 0] = 1 - 1 / tau0
    C1[0, 0] = (1 / tau0) * f * q / Y
    B0[0, 1, 0] = (1 / tau0) * (1 - f)
    C0[1, 1] = 1 - 1 / tau1
    C1[0, 1] = (1 / tau1) / Y
    return Model(C0, C1, B0, B1)


def Kbyp_closed(tau0, tau1, f, q, phi):
    """Note B (14)."""
    Y = 1 - f + f * q
    al = (1 - f) / Y
    G = tau0 + al * tau1
    t0, t1 = t_fun(phi, tau0), t_fun(phi, tau1)
    P = t0 * ((1 - al) + al * t1)
    return al * al * (A_phi(phi) / G - t0 * t1 / (1 - P))


# --------------------------------------------------------------------------------------------
# section H: the tools against known answers
# --------------------------------------------------------------------------------------------
def rand_corr(rng, I):
    X = rng.standard_normal((I, I + 2))
    S = X @ X.T
    dd = np.sqrt(np.diag(S))
    return S / np.outer(dd, dd)


def admissible_R(rng, phis):
    """A contemporaneous correlation matrix that componentwise AR(1) drivers with these
    coefficients can have: R_ij = C_ij sqrt((1-phi_i^2)(1-phi_j^2))/(1 - phi_i phi_j)."""
    I = len(phis)
    Cm = rand_corr(rng, I)
    ph = np.asarray(phis)
    return Cm * np.sqrt(np.outer(1 - ph ** 2, 1 - ph ** 2)) / (1 - np.outer(ph, ph))


def sec_H():
    rp = Rep("H. The tools against known answers (before they are used on the notes)")
    rng = np.random.default_rng(1001)
    # H1 lag sum against resolvent, dense random matrices, several correlated drivers
    worst = 0.0
    for case in range(200):
        n = int(rng.integers(2, 7))
        A = rng.random((n, n)) * (rng.random((n, n)) < 0.7)
        A += np.diag(rng.random(n) * 0.5) + 0.02
        I = int(rng.integers(1, 4))
        Es = [A * rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.5) for _ in range(I)]
        phis = list(rng.uniform(-0.9, 0.95, I))
        R = admissible_R(rng, phis)
        C1, lam, c, v, w = second_order(A, Es, phis, R, method="resolvent")
        C2 = second_order(A, Es, phis, R, method="lagsum")[0]
        gross = sum(abs(ci) for ci in c) ** 2 + 1e-300
        worst = max(worst, abs(C1 - C2) / max(abs(C1), abs(C2), 1e-3 * gross))
    rp.check("H1", worst < 1e-9, "lag sum = resolvent form of the general coefficient, 200 random dense matrices "
             "with 1 to 3 correlated AR(1) drivers of different memory: largest relative difference %.2e" % worst)

    # H2 general coefficient against the exact loss in a periodic environment of small amplitude
    worst = 0.0
    ncase = 60
    for case in range(ncase):
        n = int(rng.integers(2, 6))
        A = rng.random((n, n)) * (rng.random((n, n)) < 0.7) + 0.05
        A += np.diag(rng.random(n) * 0.5)
        lam = perron(A)[0]
        A = A / lam
        I = int(rng.integers(1, 4))
        N = int(rng.integers(2, 8))
        xi = rng.standard_normal((N, I))
        xi -= xi.mean(0)
        xi /= np.abs(xi).max()
        signs = [rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < 0.6) for _ in range(I)]
        bmask = (rng.random((n, n)) < 0.4)
        bmask[0, :] = True                       # births: a random set of entries containing a full row
        B0 = np.array([A * sg * (~bmask) for sg in signs])
        B1 = np.array([A * sg * bmask for sg in signs])
        C0 = A * (~bmask) - B0.sum(0)
        C1 = A * bmask - B1.sum(0)

        def rho(i, j, l, xi=xi, N=N):
            return float(np.mean(xi[:, i] * np.roll(xi[:, j], l)))   # mean_t xi_i(t) xi_j(t - l)

        Es = [A * sg for sg in signs]
        Cg, lam1, c, v, w = second_order(A, Es, rho=rho)
        b = w @ (A * bmask) @ v
        Kpred = Cg / b
        ests = []
        for amp in (4e-3, 2e-3):
            mdl = Model(C0, C1, B0, B1)
            hseq = 1 + amp * xi
            ests.append(floquet_root(mdl, hseq, -0.5, 0.5) / amp ** 2)
        Kex = 2 * ests[1] - ests[0]
        gross = sum(abs(ci) for ci in c) ** 2 / abs(b)
        worst = max(worst, abs(Kex - Kpred) / max(abs(Kpred), 1e-2 * gross))
    rp.check("H2", worst < TOL_FLOQ, "general coefficient (lag sum with the periodic covariance, cross terms "
             "between drivers included) = exact loss/amplitude^2 of a periodic environment, %d random dense "
             "matrices, periods 2 to 7, amplitudes 4e-3 and 2e-3 extrapolated: largest relative difference %.2e"
             % (ncase, worst))
    # H3 quadrature: trapezoid against Gauss-Hermite and a closed form
    f = lambda z: np.log(0.3 + 0.7 * np.exp(1.3 * z - 0.845))
    a1, a2 = Ez(f), Ez_gh(f, 300)
    a3 = Ez(lambda z: z ** 4)
    rp.check("H3", abs(a1 - a2) < 1e-11 and abs(a3 - 3) < 1e-11,
             "trapezoid rule against Gauss-Hermite (300 nodes) on E log[0.3 + 0.7 e^(1.3 Z - 0.845)]: "
             "%.13f against %.13f; E Z^4 = %.13f" % (a1, a2, a3))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B: formulas (2), (3), (4)
# --------------------------------------------------------------------------------------------
def rand_taus(rng, n, lo=1.0, hi=12.0):
    return 1.0 + rng.random(n) * (hi - lo) * rng.random(n) + (lo - 1.0)


def rand_gauge(rng, n):
    g = np.exp(rng.uniform(-1, 1, n))
    g /= g.prod() ** (1.0 / n)
    return g


def Kc_general(taus, k, d, phi, gauge=None, method="resolvent"):
    mdl = model_ring(taus, {}, contact=(k, d, 0), gauge=gauge)
    return K_model(mdl, [1.0], [0], [phi], [[1.0]], method=method)[0]


def Kg_general(taus, k, phi, gauge=None, method="resolvent"):
    mdl = model_ring(taus, {k: 0}, gauge=gauge)
    return K_model(mdl, [1.0], [0], [phi], [[1.0]], method=method)[0]


def Kc_complex(taus, k, d, phi):
    """K_c through linear solves only (growth factor exactly one), usable with complex stays."""
    taus = np.asarray(taus)
    n = len(taus)
    s = 1 - 1 / taus
    g = 1 / taus
    A = np.zeros((n, n), dtype=taus.dtype)
    for j in range(n):
        A[j, j] += s[j]
        A[(j + 1) % n, j] += g[j]
    E = np.zeros((n, n), dtype=taus.dtype)
    E[(k + 1) % n, k] += g[k]
    E[k, k] += -(1 - d) * (1 - s[k])
    Bm = np.zeros((n, n), dtype=taus.dtype)
    Bm[0, n - 1] = g[n - 1]
    v, w = perron_at_one(A)
    Q = np.eye(n) - np.outer(v, w)
    T = A @ Q
    c = w @ E @ v
    C = c * c / 2 - phi * (w @ E @ np.linalg.solve(np.eye(n) - phi * T, Q @ E @ v))
    return C / (w @ Bm @ v)


def contact_square_wave_D(tau_k, b, d, amp, half):
    """Exact loss, contact stays (tau_k, b), multiplier 1+amp for `half` steps then 1-amp for `half`."""
    mdl = model_ring([tau_k, b], {}, contact=(0, d, 0))
    hseq = np.array([[1 + amp]] * half + [[1 - amp]] * half)
    return floquet_root(mdl, hseq)


def sec_B234():
    rp = Rep("B (2), (3), (4): contact-limited transfer, its ratio to gain-only, parking")
    rng = np.random.default_rng(2002)
    # (2) against the general coefficient
    worst = 0.0
    worst_ls = 0.0
    ncase = 300
    for case in range(ncase):
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n)
        k = int(rng.integers(0, n))
        d = rng.random()
        phi = rng.uniform(-0.9, 0.95)
        a = Kc_closed(taus, k, d, phi)
        b = Kc_general(taus, k, d, phi, gauge=rand_gauge(rng, n))
        worst = max(worst, abs(a - b) / (max(abs(a), abs(b)) + ABS_FLOOR / TOL_ALG))
        if case < 40:
            b2 = Kc_general(taus, k, d, phi, method="lagsum")
            worst_ls = max(worst_ls, abs(b - b2) / (max(abs(a), abs(b)) + 1e-4))
    rp.check("B2.i", worst < TOL_ALG, "(2) K_c = d^2 K_g + d(1-d) L_k against the general coefficient, %d random rings "
             "(1 to 5 stages, stays 1 to 12, d in (0,1), memory -0.9 to 0.95, random gauge of the transfers): "
             "largest relative difference %.2e (lag sum against resolvent on 40 of them: %.1e)" % (ncase, worst, worst_ls))
    # the one comparison with the earlier independent derivation (C1) of notes/contact-limited-transfer.md
    worst = 0.0
    for case in range(100):
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n)
        k = int(rng.integers(0, n))
        d = rng.random()
        phi = rng.uniform(0.01, 0.95)
        th = phi / (1 - phi)
        qk = 1 / (1 + taus / th)
        Rr = qk.prod()
        c1 = d * d * (1 + 2 * th) / (2 * taus.sum()) - d * (Rr - (1 - d) * qk[k]) / (1 - Rr)
        worst = max(worst, relerr(c1, Kc_closed(taus, k, d, phi)))
    rp.check("B2.C1", worst < TOL_ALG, "(2) is the same function as (C1) of notes/contact-limited-transfer.md, "
             "100 random rings: largest relative difference %.2e" % worst)
    # zero memory
    worst = 0.0
    for case in range(100):
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n)
        k = int(rng.integers(0, n))
        d = rng.random()
        worst = max(worst, relerr(Kc_general(taus, k, d, 0.0), d * d / (2 * taus.sum())))
    rp.check("B2.phi0", worst < TOL_ALG, "without memory K_c = d^2/(2G) (general coefficient, 100 random rings): %.2e" % worst)

    # (3)
    worst = 0.0
    wrong = 0
    nbig = 0
    ex_hi = ex_lo = None
    worstL = 0.0
    for case in range(400):
        n = int(rng.integers(2, 6))
        taus = rand_taus(rng, n, hi=40.0 if case % 2 else 12.0)
        k = int(rng.integers(0, n))
        d = rng.uniform(0.02, 0.98)
        phi = rng.uniform(-0.9, 0.95)
        gz = rand_gauge(rng, n)
        Kc = Kc_general(taus, k, d, phi, gauge=gz)
        Kg = Kg_general(taus, k, phi, gauge=gz)
        Lk = Lk_closed(taus, k, phi)
        # L_k recovered from the general coefficient at a second death fraction
        d2 = rng.uniform(0.02, 0.98)
        Lgen = (Kc_general(taus, k, d2, phi, gauge=gz) - d2 * d2 * Kg) / (d2 * (1 - d2))
        worstL = max(worstL, abs(Lgen - Lk) / (abs(Lk) + 1e-6))
        if abs(Kg) > 1e-6:
            worst = max(worst, relerr(Kc / Kg, d * d + d * (1 - d) * Lk / Kg))
        lhs = Kc > Kg
        rhs = d * Lk > (1 + d) * Kg
        if abs(Kc - Kg) > 1e-12 and lhs != rhs:
            wrong += 1
        if lhs:
            nbig += 1
            if ex_hi is None and phi > 0:
                ex_hi = (taus.copy(), k, d, phi, Kc, Kg)
        elif ex_lo is None and phi > 0:
            ex_lo = (taus.copy(), k, d, phi, Kc, Kg)
    rp.check("B3.ratio", worst < TOL_ALG and worstL < 1e-7,
             "(3) K_c/K_g = d^2 + d(1-d) L_k/K_g with K_c and K_g from the general coefficient, 400 random rings: "
             "largest relative difference %.2e; L_k recovered from the general coefficient at another d: %.1e" % (worst, worstL))
    rp.check("B3.ineq", wrong == 0, "K_c > K_g exactly when d L_k > (1+d) K_g: the two sides disagree in %d of 400 "
             "random rings (K_c > K_g in %d of them)" % (wrong, nbig))
    for lab, ex in (("K_c > K_g", ex_hi), ("K_c < K_g", ex_lo)):
        if ex is not None:
            rp.p("      example with %s: stays %s, contact stage %d, d = %.4f, memory %.4f: K_c = %.6f, K_g = %.6f"
                 % (lab, np.array2string(ex[0], precision=3), ex[1], ex[2], ex[3], ex[4], ex[5]))
    # limits of the ratio, and the linear small-d term
    taus = np.array([5.0, 3.0, 2.0])
    phi = 0.7
    Kg = Kg_general(taus, 0, phi)
    r_small = Kc_general(taus, 0, 1e-7, phi) / Kg
    r_big = Kc_general(taus, 0, 1 - 1e-7, phi) / Kg
    lin = Kc_general(taus, 0, 1e-7, phi) / 1e-7
    rp.check("B3.limits", abs(r_small) < 1e-6 and abs(r_big - 1) < 1e-6 and relerr(lin, Lk_closed(taus, 0, phi)) < 1e-6,
             "stays (5,3,2), memory 0.7: K_c/K_g = %.3e at d = 1e-7 and %.9f at d = 1 - 1e-7; K_c/d at d = 1e-7 is "
             "%.9f = L_k = %.9f (the small-d term is linear in d)" % (r_small, r_big, lin, Lk_closed(taus, 0, phi)))
    # K_g >= 1/(2G), L_k >= 0, K_c > 0 for non-negative memory
    bad = 0
    minrat = 1e9
    minL = 1e9
    minKc = 1e9
    for case in range(2000):
        n = int(rng.integers(1, 7))
        taus = rand_taus(rng, n, hi=60.0)
        k = int(rng.integers(0, n))
        d = rng.uniform(0.001, 1.0)
        phi = rng.uniform(0.0, 0.99)
        Kg = Kg_general(taus, k, phi)
        Kc = Kc_general(taus, k, d, phi)
        Lk = (Kc - d * d * Kg) / (d * (1 - d)) if d < 0.999 else 0.0
        minrat = min(minrat, Kg * 2 * taus.sum())
        minL = min(minL, Lk)
        minKc = min(minKc, Kc / d)
        if Kg * 2 * taus.sum() < 1 - 1e-9 or Kc <= 0 or Lk < -1e-9:
            bad += 1
    rp.check("B3.sign", bad == 0, "for memory in [0, 0.99], 2,000 random rings (general coefficient): K_g >= 1/(2G) "
             "(smallest 2 G K_g = %.6f), L_k >= 0 (smallest %.2e), K_c > 0 (smallest K_c/d = %.3e); exceptions %d"
             % (minrat, minL, minKc, bad))
    # two equal stays
    worst = 0.0
    for case in range(100):
        a = rng.uniform(1.0, 20.0)
        phi = rng.uniform(-0.9, 0.95)
        cl = (1 / (4 * a)) * (1 + phi * (1 + 2 / a)) / (1 - phi * (1 - 2 / a))
        worst = max(worst, abs(cl - Kg_general([a, a], 0, phi)) / (abs(cl) + 1e-3))
    k1 = Kg_general([2.0, 2.0], 0, -0.8)
    k2 = Kg_general([2.0, 2.0], 0, 0.0)
    bnd = brentq(lambda ph: Kg_general([2.0, 2.0], 0, ph), -0.9, 0.0, xtol=1e-14)
    rp.check("B.Kg2", worst < TOL_ALG and close_num(k1, -0.075) and close_num(k2, 0.125) and abs(bnd + 0.5) < 1e-10,
             "two equal stays a: K_g = (1/4a)(1+phi(1+2/a))/(1-phi(1-2/a)), 100 random (a, phi): %.2e; stays (2,2): "
             "K_g(-0.8) = %.12f (note -0.075), K_g(0) = %.12f (note 0.125), zero of my general coefficient at phi = %.12f "
             "(note -0.5)" % (worst, k1, k2, bnd))
    # exact alternating counterexamples
    mdl = model_ring([2.0, 2.0], {0: 0})
    D1 = floquet_root(mdl, np.array([[1.6], [0.4]]))
    mdl = model_ring([5.0, 2.0], {}, contact=(0, 0.1, 0))
    D2 = floquet_root(mdl, np.array([[1.6], [0.4]]))
    # same with the multiplier on the other transfer / the birth on the other transfer (must not matter)
    D1b = floquet_root(model_ring([2.0, 2.0], {1: 0}), np.array([[1.6], [0.4]]))
    D2b = floquet_root(model_ring([5.0, 2.0], {}, contact=(0, 0.1, 0), birth=0), np.array([[0.4], [1.6]]))
    rp.check("B.alt1", close_num(D1, -0.0420311234843) and abs(D1 - D1b) < 1e-13,
             "gain-only stays (2,2), multiplier alternating 1 +- 0.6, exact: D = %.13f (note -0.0420311234843); "
             "noise on the other transfer: %.13f" % (D1, D1b))
    rp.check("B.alt2", close_num(D2, -0.00487832384055) and abs(D2 - D2b) < 1e-13,
             "contact stays (5,2), d = 0.1, same multiplier, exact: D = %.14f (note -0.00487832384055); birth on the "
             "contact transfer and the other phase: %.14f" % (D2, D2b))
    rp.p("      second-order values at memory -1 (the alternating environment): gain-only 0.36 K_g(-1) = %.6f, "
         "contact 0.36 K_c(-1) = %.6f" % (0.36 * Kg_closed([2.0, 2.0], -1.0), 0.36 * Kc_closed([5.0, 2.0], 0, 0.1, -1.0)))

    # (4)
    worst = 0.0
    worst_fd = 0.0
    ncase = 200
    for case in range(ncase):
        nother = int(rng.integers(1, 5))
        to = rand_taus(rng, nother)
        ko = int(rng.integers(0, nother))
        d = rng.uniform(0.02, 1.0)
        phi = rng.uniform(0.02, 0.95)
        b = rng.uniform(1.2, 40.0)
        pos = int(rng.integers(0, nother + 1))          # where the parking stage sits in the ring
        full = np.insert(to.astype(complex), pos, b + 1e-30j)
        kfull = ko if ko < pos else ko + 1
        der = Kc_complex(full, kfull, d, phi).imag / 1e-30
        cl = dKc_db_closed(to, ko, d, phi, b)
        scale = d * d * (phi / (1 - phi) + 0.5) / (b + to.sum()) ** 2
        worst = max(worst, abs(der - cl) / max(abs(der), abs(cl), 1e-3 * scale))
        if case < 30:                                    # the complex step against a real difference quotient
            hh = 1e-4 * b
            def Kr(bb):
                return Kc_general(np.insert(to, pos, bb), kfull, d, phi)
            fd = (8 * (Kr(b + hh) - Kr(b - hh)) - (Kr(b + 2 * hh) - Kr(b - 2 * hh))) / (12 * hh)
            worst_fd = max(worst_fd, abs(fd - der) / max(abs(der), 1e-3 * scale))
    rp.check("B4.i", worst < TOL_ALG, "(4) dK_c/db against the derivative of the general coefficient (complex step), "
             "%d random rings with the parking stage at a random position, memory 0.02 to 0.95: largest relative "
             "difference %.2e (complex step against a five-point real difference on 30 of them: %.1e)" % (ncase, worst, worst_fd))
    worst0 = 0.0
    for case in range(50):
        to = rand_taus(rng, 2)
        d = rng.random()
        b = rng.uniform(1.2, 30)
        der = Kc_complex(np.array([to[0], to[1], b + 1e-30j]), 0, d, 0.0).imag / 1e-30
        worst0 = max(worst0, relerr(der, -d * d / (2 * (b + to.sum()) ** 2)))
    rp.check("B4.phi0", worst0 < TOL_ALG, "without memory dK_c/db = -d^2/[2(b+S)^2], 50 random rings: %.2e" % worst0)
    # the quoted turning stay
    def dK(b, d, phi=0.8, tk=5.0):
        return Kc_complex(np.array([tk, b + 1e-30j]), 0, d, phi).imag / 1e-30
    bstar = brentq(lambda b: dK(b, 0.8), 1.5, 20.0, xtol=1e-14)
    dlo, dhi = dK(bstar - 1, 0.8), dK(bstar + 1, 0.8)
    rp.check("B4.b", close_num(bstar, 3.43717648559) and close_num(dlo, 0.00761825473637) and close_num(dhi, -0.00311841081797),
             "contact stay 5, d = 0.8, memory 0.8: zero of the derivative of my general coefficient at b = %.11f "
             "(note 3.43717648559); derivative one unit below %.14f (note 0.00761825473637), one unit above %.14f "
             "(note -0.00311841081797)" % (bstar, dlo, dhi))
    bs = [1.5, 2, 5, 10, 20, 40, 100, 1e3, 1e4, 1e5, 1e6]
    ders = [dK(b, 0.1) for b in bs]
    th = 4.0
    tk = t_fun(0.8, 5.0)
    U = th * tk
    lead = U * (1 - 0.9 * tk) - 0.1 * (th + 0.5)
    rp.check("B4.d01", all(x > 0 for x in ders) and lead > 0,
             "same ring with d = 0.1: derivative positive at b = 1.5 ... 1e6 (smallest value times b^2: %.5f); "
             "leading coefficient U(1-alpha) - d(theta+1/2) = %.6f > 0, and b^2 dK/db at b = 1e6 is %.6f = d times it = %.6f"
             % (min(x * b * b for x, b in zip(ders, bs)), lead, ders[-1] * 1e12, 0.1 * lead))
    # the long-stay limit
    worst = 0.0
    for case in range(100):
        nother = int(rng.integers(1, 4))
        to = rand_taus(rng, nother)
        ko = int(rng.integers(0, nother))
        d = rng.random()
        phi = rng.uniform(0.02, 0.95)
        lim = Kc_complex(np.append(to, 1e9), ko, d, phi).real
        worst = max(worst, abs(lim - d * (1 - d) * t_fun(phi, to[ko])))
    rp.check("B4.limit", worst < 1e-6, "long parking: K_c -> d(1-d) t_k (parking stay 1e9, 100 random rings): largest "
             "absolute difference %.1e" % worst)
    bad = 0
    for case in range(1000):
        nother = int(rng.integers(1, 5))
        to = rand_taus(rng, nother)
        phi = rng.uniform(0.0, 0.97)
        b = 10 ** rng.uniform(3, 6)
        if Kc_complex(np.append(to.astype(complex), b + 1e-30j), 0, 1.0, phi).imag > 0:
            bad += 1
    rp.check("B4.gain", bad == 0, "gain-only (d = 1), memory in [0, 0.97]: derivative negative at long parking "
             "(b from 1e3 to 1e6) in all of 1,000 random rings; exceptions %d" % bad)
    # the exact finite-amplitude parking boundary, square wave of period 100
    def diff(d):
        return contact_square_wave_D(5.0, 40.0, d, 0.6, 50) - contact_square_wave_D(5.0, 20.0, d, 0.6, 50)
    dstar = brentq(diff, 0.2, 0.7, xtol=1e-14)
    e1, e2 = diff(dstar - 0.03), diff(dstar + 0.03)
    rp.check("B4.period100", close_num(dstar, 0.436417797968) and close_num(e1, 0.000734413962806, 1e-7)
             and close_num(e2, -0.000814303130121, 1e-7),
             "contact stays (5,b), 50 steps at 1.6 then 50 at 0.4, exact: D(b=40) = D(b=20) at d = %.12f (note "
             "0.436417797968); D40 - D20 at d - 0.03: %+.15f (note +0.000734413962806); at d + 0.03: %+.15f "
             "(note -0.000814303130121)" % (dstar, e1, e2))
    rp.p("      the losses themselves at the boundary: D20 = %.10f, D40 = %.10f; stay rate of the contact stage "
         "ranges %.4f to %.4f (a probability)" % (contact_square_wave_D(5.0, 20.0, dstar, 0.6, 50),
                                                contact_square_wave_D(5.0, 40.0, dstar, 0.6, 50),
                                                0.8 - 0.6 * (1 - dstar) * 0.2, 0.8 + 0.6 * (1 - dstar) * 0.2))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (5): curved encounter functions
# --------------------------------------------------------------------------------------------
def resp_F(kind, par):
    if kind == "II":
        return lambda H: H / (1 + H)
    if kind == "III":
        return lambda H: H * H / (1 + H * H)
    if kind == "power":
        return lambda H: H ** par
    if kind == "inverse":
        return lambda H: 1 / H
    raise ValueError(kind)


def eta_chi_numeric(F, x):
    """eta = x F'/F and chi = x^2 F''/F at mean abundance x, by a complex step for F' and a
    five-point difference of that for F''."""
    def F1(u):
        return (F(u + 1e-30j)).imag / 1e-30
    hh = 1e-3 * x
    F2 = (8 * (F1(x + hh) - F1(x - hh)) - (F1(x + 2 * hh) - F1(x - 2 * hh))) / (12 * hh)
    return x * F1(x) / F(x), x * x * F2 / F(x)


def eta_chi_table(kind, x, par):
    """The table of note B."""
    if kind == "II":
        return 1 / (1 + x), -2 * x / (1 + x) ** 2
    if kind == "III":
        return 2 / (1 + x * x), 2 * (1 - 3 * x * x) / (1 + x * x) ** 2
    if kind == "power":
        return par, par * (par - 1)
    return -1.0, 2.0


def benefit_table(kind, x, par, Kg):
    """The 'local benefit against baseline (i)' column of note B's table."""
    if kind == "II":
        return False if Kg >= 0 else None
    if kind == "III":
        return (1 - 4 * Kg) > 0 and x * x < (1 - 4 * Kg) / 3
    if kind == "power":
        return par * ((1 - 2 * Kg) * par - 1) > 0
    return Kg < 1


def curved_shift(F, x, sigma, s, baseline):
    """Exact one-stage threshold shift.  baseline 'i': root y of E log[s + (1-s) e^y F(H)/F(x)] = 0;
    baseline 'ii': the same with F(H)/E F(H).  H = x exp(sigma Z - sigma^2/2)."""
    H = lambda z: x * np.exp(sigma * z - sigma * sigma / 2)
    if baseline == "i":
        den = F(x)
    else:
        den = Ez(lambda z: F(H(z)))
    return scalar_root(lambda z: F(H(z)) / den, s)


def sec_B5():
    rp = Rep("B (5): curved encounter functions (elasticity eta, curvature chi, the table, the one-stage benchmark)")
    rng = np.random.default_rng(5005)
    kinds = ["II", "III", "power", "inverse"]
    worst = {k: 0.0 for k in kinds}
    wrong = {k: 0 for k in kinds}
    nben = {k: 0 for k in kinds}
    ncase = 120
    for kind in kinds:
        for case in range(ncase):
            x = float(np.exp(rng.uniform(-2.5, 2.0)))
            par = float(rng.uniform(-3, 4)) if kind == "power" else None
            F = resp_F(kind, par)
            xx = 1.0 if kind in ("power", "inverse") else x
            e1, c1 = eta_chi_numeric(F, xx)
            e2, c2 = eta_chi_table(kind, xx, par)
            worst[kind] = max(worst[kind], abs(e1 - e2) / max(abs(e2), 1e-6), abs(c1 - c2) / max(abs(c2), 1e-3))
            Kg = float(rng.uniform(-0.3, 1.3)) if kind == "inverse" else float(rng.uniform(-0.2, 0.5))
            ben = c1 / 2 - e1 * e1 * Kg > 0
            tb = benefit_table(kind, xx, par, Kg)
            if tb is None:
                continue
            nben[kind] += int(ben)
            if ben != tb and abs(c1 / 2 - e1 * e1 * Kg) > 1e-9:
                wrong[kind] += 1
    for kind in kinds:
        rp.check("B5.table." + kind, worst[kind] < TOL_ALG and wrong[kind] == 0,
                 "type %s: eta and chi of the table against numerical derivatives of F, %d random cases: largest "
                 "relative difference %.2e; the table's benefit condition against the sign of chi/2 - eta^2 K_g "
                 "(K_g random): %d disagreements (benefit in %d cases)" % (kind, ncase, worst[kind], wrong[kind], nben[kind]))
    # (5) against the exact one-stage growth law at small noise (stay 3/4: K_g = 1/8 at any memory)
    s = 0.75
    Kg = 1 / 8
    worst_i = worst_ii = 0.0
    examples = [("II", 0.7, None), ("II", 3.0, None), ("III", 0.25, None), ("III", 0.408248290464, None),
                ("III", 1.5, None), ("power", 1.0, 2.5), ("power", 1.0, 0.5), ("power", 1.0, -0.7),
                ("power", 1.0, 3.0), ("inverse", 1.0, None)]
    for kind, x, par in examples:
        F = resp_F(kind, par)
        e, c = eta_chi_numeric(F, x)
        est = {}
        for bl in ("i", "ii"):
            vals = [curved_shift(F, x, sg, s, bl) / sg ** 2 for sg in (0.04, 0.02)]
            est[bl] = (4 * vals[1] - vals[0]) / 3
        pred_ii = e * e * Kg
        pred_i = -(c / 2 - e * e * Kg)
        worst_ii = max(worst_ii, abs(est["ii"] - pred_ii) / max(abs(pred_ii), 1e-3))
        worst_i = max(worst_i, abs(est["i"] - pred_i) / max(abs(pred_i), 1e-3))
        rp.p("      type %-7s x = %.4f%s: exact shift/sigma^2 -> %+.7f (ii), %+.7f (i); (5) gives %+.7f, %+.7f"
             % (kind, x, "" if par is None else " power %.1f" % par, est["ii"], est["i"], pred_ii, pred_i))
    rp.check("B5.small", worst_i < TOL_FLOQ and worst_ii < TOL_FLOQ,
             "(5) against the exact one-stage shift at sigma = 0.04 and 0.02 (extrapolated), %d responses: largest relative "
             "difference %.2e for D(ii) = eta^2 K_g sigma^2 and %.2e for the shift against baseline (i), "
             "-(chi/2 - eta^2 K_g) sigma^2" % (len(examples), worst_ii, worst_i))
    # (5) on rings with memory: eta^2 K_g is the general coefficient with E = eta * (transfer)
    worst = 0.0
    for case in range(100):
        n = int(rng.integers(1, 5))
        taus = rand_taus(rng, n)
        phi = rng.uniform(-0.9, 0.95)
        eta = rng.uniform(-2, 3)
        k = int(rng.integers(0, n))
        mdl = model_ring(taus, {k: 0})
        Kgen = K_model(mdl, [eta], [0], [phi], [[1.0]])[0]
        worst = max(worst, abs(Kgen - eta * eta * Kg_closed(taus, phi)) / (abs(Kgen) + 1e-6))
    rp.check("B5.i", worst < TOL_ALG, "D(ii) coefficient eta^2 K_g against the general coefficient with the first-order "
             "perturbation eta * transfer, 100 random rings: %.2e" % worst)
    # type III boundaries
    F3 = resp_F("III", None)
    xloc = math.sqrt((1 - 4 * Kg) / 3)
    e, c = eta_chi_numeric(F3, xloc)
    rp.check("B5.x_local", close_num(xloc, 0.408248290464) and abs(c / 2 - e * e * Kg) < 1e-9,
             "type III, K_g = 1/8: local boundary x = %.12f (note 0.408248290464); chi/2 - eta^2 K_g there = %.1e; "
             "one side each: x = 0.3: %+.6f (benefit), x = 0.5: %+.6f (cost)"
             % (xloc, c / 2 - e * e * Kg,
                eta_chi_numeric(F3, 0.3)[1] / 2 - eta_chi_numeric(F3, 0.3)[0] ** 2 * Kg,
                eta_chi_numeric(F3, 0.5)[1] / 2 - eta_chi_numeric(F3, 0.5)[0] ** 2 * Kg))
    quoted = {0.3: (0.382645737133, 0.306116589707, 0.459174884560, -0.0118416347363, 0.0113768991662),
              1.0: (0.229888315533, 0.183910652426, 0.275865978639, -0.058600689333, 0.0559916666033)}
    for sg, (xq, xlo, xhi, slo, shi) in quoted.items():
        xb = brentq(lambda x: Ez(lambda z: np.log(s + (1 - s) * F3(x * np.exp(sg * z - sg * sg / 2)) / F3(x))),
                    0.05, 1.0, xtol=1e-15)
        xb_gh = brentq(lambda x: Ez_gh(lambda z: np.log(s + (1 - s) * F3(x * np.exp(sg * z - sg * sg / 2)) / F3(x)), 300),
                       0.05, 1.0, xtol=1e-15)
        ylo = curved_shift(F3, xlo, sg, s, "i")
        yhi = curved_shift(F3, xhi, sg, s, "i")
        rp.check("B5.x_sigma%.1f" % sg, close_num(xb, xq) and close_num(ylo, slo, 1e-8) and close_num(yhi, shi, 1e-8)
                 and abs(xb - xb_gh) < 1e-10,
                 "type III, stay 3/4, sigma = %.1f, exact one-stage: the shift against baseline (i) is zero at x = %.12f "
                 "(note %.12f; Gauss-Hermite 300 nodes gives %.12f); shift at x = %.12f: %+.13f (note %+.13f); at x = %.12f: "
                 "%+.13f (note %+.13f)" % (sg, xb, xq, xb_gh, xlo, ylo, slo, xhi, yhi, shi))
    # the stay 1/2 benchmark has no type III boundary at second order (K_g = 1/4): 1 - 4 K_g = 0
    rp.p("      (stay 1/2 gives K_g = 1/4 and 1 - 4 K_g = 0: no local type III boundary, as the note's protocol records)")
    # eta = 3, no staying: exact loss eta^2 sigma^2/2
    vals = []
    for sg in (0.3, 1.0):
        den = Ez(lambda z: np.exp(3 * (sg * z - sg * sg / 2)))
        vals.append(-Ez(lambda z: 3 * (sg * z - sg * sg / 2) - math.log(den)))
    rp.check("B.eta3", close_num(vals[0], 0.405) and close_num(vals[1], 4.5),
             "power 3, one stage without staying: exact loss by quadrature %.12f at host sigma 0.3 (note 0.405; host-only "
             "bound 0.045) and %.12f at sigma 1 (note 4.5; host-only bound 0.5)" % (vals[0], vals[1]))
    for eta in (0.9, 1.1):
        sg = 0.5
        den = Ez(lambda z: np.exp(eta * (sg * z - sg * sg / 2)))
        Dv = -Ez(lambda z: eta * (sg * z - sg * sg / 2) - math.log(den))
        rp.p("      power %.1f at host sigma 0.5: loss %.6f, host-only bound %.6f (%s)"
             % (eta, Dv, sg * sg / 2, "within" if Dv <= sg * sg / 2 else "exceeds it"))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (6), (7): one host at two transfers
# --------------------------------------------------------------------------------------------
def alt_K(model, signs):
    """Exact loss/amplitude^2 in the strictly alternating environment h_i = 1 + a signs_i (-1)^t,
    amplitudes 0.02 and 0.01, extrapolated (the loss is even in a)."""
    vals = []
    for a in (0.02, 0.01):
        hseq = np.array([[1 + a * sg for sg in signs], [1 - a * sg for sg in signs]])
        vals.append(floquet_root(model, hseq, -0.5, 0.5) / a ** 2)
    return (4 * vals[1] - vals[0]) / 3


def sec_B67():
    rp = Rep("B (6), (7): one host driving two or more transfers")
    rng = np.random.default_rng(6006)
    worst = 0.0
    worst_alt = 0.0
    ncase = 300
    for case in range(ncase):
        n = int(rng.integers(2, 6))
        taus = rand_taus(rng, n)
        a = rng.uniform(-2, 2, n) * (rng.random(n) < 0.7)
        if np.all(a == 0):
            a[0] = 1.0
        phi = rng.uniform(-0.9, 0.95)
        idx = [k for k in range(n) if a[k] != 0]
        mdl = model_ring(taus, {k: i for i, k in enumerate(idx)}, gauge=rand_gauge(rng, n))
        Kgen = K_model(mdl, [a[k] for k in idx], [0] * len(idx), [phi], [[1.0]])[0]
        cl = K6_closed(taus, a, phi)
        scale = A_phi(abs(phi)) / taus.sum() * np.abs(a).sum() ** 2
        worst = max(worst, abs(Kgen - cl) / max(abs(Kgen), abs(cl), 1e-3 * scale))
        if case < 40:
            Kex = alt_K(mdl, [a[k] for k in idx])
            cl1 = K6_closed(taus, a, -1.0)
            worst_alt = max(worst_alt, abs(Kex - cl1) / max(abs(cl1), 1e-2 * np.abs(a).sum() ** 2 / taus.sum()))
    rp.check("B6.i", worst < TOL_ALG, "(6) against the general coefficient, %d random rings of 2 to 5 stages with a common "
             "driver on a random set of transfers, signed elasticities in (-2,2), memory -0.9 to 0.95: largest relative "
             "difference %.2e" % (ncase, worst))
    rp.check("B6.alt", worst_alt < TOL_FLOQ, "(6) continued to memory -1 against the exact loss of a strictly alternating "
             "environment (amplitudes 0.02, 0.01), 40 of those rings: largest relative difference %.2e" % worst_alt)
    # (7)
    worst = 0.0
    for case in range(150):
        a = rng.uniform(1.0, 20.0)
        phi = rng.uniform(-0.9, 0.95)
        mdl = model_ring([a, a], {0: 0, 1: 1})
        Kpm = K_model(mdl, [1.0, -1.0], [0, 0], [phi], [[1.0]])[0]
        Kpp = K_model(mdl, [1.0, 1.0], [0, 0], [phi], [[1.0]])[0]
        cl = (2 * phi / a) / (1 - phi * (1 - 2 / a))
        worst = max(worst, abs(Kpm - cl) / (abs(cl) + 1e-4), abs(Kpp - 1 / a) * a)
    mdl = model_ring([2.0, 2.0], {0: 0, 1: 1})
    kneg = K_model(mdl, [1.0, -1.0], [0, 0], [-0.3], [[1.0]])[0]
    kpos = K_model(mdl, [1.0, -1.0], [0, 0], [0.3], [[1.0]])[0]
    kzero = K_model(mdl, [1.0, -1.0], [0, 0], [0.0], [[1.0]])[0]
    rp.check("B7.i", worst < TOL_ALG and abs(kzero) < 1e-14 and kneg < 0 < kpos,
             "(7) K_+- = (2 phi/a)/(1 - phi(1-2/a)) and K_++ = 1/a against the general coefficient, 150 random (a, phi): "
             "%.2e; stays (2,2): K_+- = %.6f at memory -0.3, %.1e at 0, %.6f at +0.3 (the local boundary is zero memory)"
             % (worst, kneg, kzero, kpos))
    # zero memory: K = (sum a)^2/(2G)
    worst = 0.0
    for case in range(100):
        n = int(rng.integers(2, 6))
        taus = rand_taus(rng, n)
        a = rng.uniform(-2, 2, n)
        mdl = model_ring(taus, {k: k for k in range(n)})
        Kgen = K_model(mdl, list(a), [0] * n, [0.0], [[1.0]])[0]
        worst = max(worst, abs(Kgen - a.sum() ** 2 / (2 * taus.sum())) / (np.abs(a).sum() ** 2 / taus.sum()))
    rp.check("B6.phi0", worst < TOL_ALG, "without memory K = (sum a)^2/(2G), 100 random rings: %.2e" % worst)
    # D_++ = 2 d_s in exact periodic environments of large amplitude
    worst = 0.0
    ncase = 500
    for case in range(ncase):
        a = rng.uniform(1.05, 12.0)
        s = 1 - 1 / a
        N = int(rng.integers(2, 13))
        sg = rng.uniform(0.3, 2.5)
        h = np.exp(sg * rng.standard_normal(N))
        h /= h.mean()
        mdl = model_ring([a, a], {0: 0, 1: 0})
        D = floquet_root(mdl, h[:, None], -10, 10)
        ds = brentq(lambda y: np.mean(np.log(s + (1 - s) * math.exp(y) * h)), -40, 40, xtol=1e-15)
        worst = max(worst, abs(D - 2 * ds))
    rp.check("B7.Dpp_periodic", worst < 1e-10, "D_++ = 2 d_s exactly: %d random periodic environments (period 2 to 12, "
             "log-amplitude 0.3 to 2.5, equal stays 1.05 to 12), full-period products against the one-stage root: largest "
             "absolute difference %.1e" % (ncase, worst))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (8) to (11): parallel alternatives
# --------------------------------------------------------------------------------------------
def Kpar_general(tau0, taus, w, kappa, R, phis, gauge=None):
    mdl = model_parallel(tau0, taus, w, gauge=gauge)
    J = len(taus)
    return K_model(mdl, list(kappa), list(range(J)), list(phis), R)[0]


def scalar_parallel_D(f, sigma, s, kap2, grid=None):
    """Exact one-stage loss with two alternative independent hosts: root y of
    E log[s + (1-s) e^y ((1-f) h1 + f h2)] = 0, h1 = exp(sigma Z1 - sigma^2/2),
    h2 = exp(kap2 sigma Z2 - kap2^2 sigma^2/2)."""
    z, w = grid if grid is not None else (_Z2, _W2)
    h1 = np.exp(sigma * z - sigma * sigma / 2)[:, None]
    h2 = np.exp(kap2 * sigma * z - (kap2 * sigma) ** 2 / 2)[None, :]
    mix = (1 - f) * h1 + f * h2
    W = np.outer(w, w)
    return brentq(lambda y: float(np.sum(W * np.log(s + (1 - s) * math.exp(y) * mix))), -30, 30, xtol=1e-15, rtol=8.9e-16)


def sec_B811():
    rp = Rep("B (8) to (11): parallel alternative hosts")
    rng = np.random.default_rng(8008)
    worst8 = worst9 = worst10 = worst_alt = 0.0
    ncase = 300
    for case in range(ncase):
        J = int(rng.integers(1, 5))
        tau0 = float(rand_taus(rng, 1)[0])
        taus = rand_taus(rng, J)
        w = rng.dirichlet(np.ones(J))
        kappa = rng.uniform(0.2, 3.0, J) * np.where(rng.random(J) < 0.15, -1, 1)
        R = rand_corr(rng, J)
        gz = np.exp(rng.uniform(-1, 1, J))
        phi = rng.uniform(-0.9, 0.95)
        scale = (np.abs(w * kappa).sum()) ** 2 / (tau0 + (w * taus).sum())
        g0 = Kpar_general(tau0, taus, w, kappa, R, [0.0] * J, gauge=gz)
        worst8 = max(worst8, abs(g0 - Kpar_closed(tau0, taus, w, kappa, R, 0.0)) / max(abs(g0), 1e-3 * scale))
        g1 = Kpar_general(tau0, taus, w, kappa, R, [phi] * J, gauge=gz)
        c1 = Kpar_closed(tau0, taus, w, kappa, R, phi)
        worst9 = max(worst9, abs(g1 - c1) / max(abs(g1), abs(c1), 1e-3 * scale * A_phi(abs(phi))))
        phis = rng.uniform(-0.9, 0.95, J)
        Ra = admissible_R(rng, phis)
        g2 = Kpar_general(tau0, taus, w, kappa, Ra, phis, gauge=gz)
        c2 = Kpar_closed_multi(tau0, taus, w, kappa, Ra, phis)
        worst10 = max(worst10, abs(g2 - c2) / max(abs(g2), abs(c2), 1e-3 * scale * A_phi(np.abs(phis).max())))
        if case < 30:
            sg = np.where(rng.random(J) < 0.5, -1.0, 1.0)
            Kex = alt_K(model_parallel(tau0, taus, w), list(sg * kappa))
            cl = Kpar_closed(tau0, taus, w, kappa, np.outer(sg, sg), -1.0)
            worst_alt = max(worst_alt, abs(Kex - cl) / max(abs(cl), 1e-2 * scale))
    rp.check("B8.i", worst8 < TOL_ALG, "(8) K = V/(2G) without memory against the general coefficient, %d random networks "
             "(1 to 4 hosts, random shares, stays, log-sd multipliers and correlation matrix): %.2e" % (ncase, worst8))
    rp.check("B9.i", worst9 < TOL_ALG, "(9) common memory -0.9 to 0.95, same networks: largest relative difference %.2e" % worst9)
    rp.check("B10.i", worst10 < TOL_ALG, "(10) a different memory for each host, correlation matrices that componentwise "
             "AR(1) drivers can have (rho_ij(l) = phi_i^l R_ij, i the later use), same networks: %.2e" % worst10)
    rp.check("B9.alt", worst_alt < TOL_FLOQ, "(9) continued to memory -1 with correlations +-1 against the exact loss of a "
             "strictly alternating environment, 30 networks: %.2e" % worst_alt)
    # is (10) changed if the memory of the EARLIER driver is used?  (a convention the note must state)
    J = 2
    tau0, taus, w, kappa = 2.0, np.array([8.0, 1.0]), np.array([0.6, 0.4]), np.array([1.0, 1.0])
    phis = np.array([0.2, 0.8])
    Ra = np.array([[1.0, 0.4], [0.4, 1.0]])
    c_later = Kpar_closed_multi(tau0, taus, w, kappa, Ra, phis)
    g_later = Kpar_general(tau0, taus, w, kappa, Ra, phis)
    mdl = model_parallel(tau0, taus, w)
    Es = [mdl.E(i) for i in range(2)]
    g_earlier = second_order(mdl.mean(), Es, rho=lambda i, j, l: phis[j] ** l * Ra[i, j])
    g_earlier = g_earlier[0] / (g_earlier[4] @ mdl.birth() @ g_earlier[3])
    inn = Ra * (1 - np.outer(phis, phis))
    rp.p("      stay 2, hosts (8,1), shares (0.6,0.4), memories (0.2,0.8), correlation 0.4: (10) = %.9f, general coefficient "
         "with rho_ij(l) = phi_i^l R_ij: %.9f; with the earlier driver's memory instead: %.9f; smallest eigenvalue of the "
         "innovation covariance %.4f (admissible)" % (c_later, g_later, g_earlier, np.linalg.eigvalsh(inn).min()))
    # (11)
    wrong = 0
    nlower = 0
    ncase = 400
    for case in range(ncase):
        J = int(rng.integers(1, 4))
        tau0 = float(rand_taus(rng, 1)[0])
        taus = rand_taus(rng, J + 1)
        wold = rng.dirichlet(np.ones(J))
        kappa = rng.uniform(0.2, 3.0, J + 1)
        R = rand_corr(rng, J + 1)
        f = rng.uniform(0.01, 0.99)
        Kold = Kpar_general(tau0, taus[:J], wold, kappa[:J], R[:J, :J], [0.0] * J)
        wnew = np.append((1 - f) * wold, f)
        Knew = Kpar_general(tau0, taus, wnew, kappa, R, [0.0] * (J + 1))
        uo = wold * kappa[:J]
        V = uo @ R[:J, :J] @ uo
        U = kappa[J] ** 2
        Cc = kappa[J] * (R[J, :J] @ uo)
        G = tau0 + (wold * taus[:J]).sum()
        dtau = taus[J] - (wold * taus[:J]).sum()
        crit = f * G * (V - 2 * Cc + U) < 2 * G * (V - Cc) + V * dtau
        if abs(Knew - Kold) > 1e-12 and (Knew < Kold) != crit:
            wrong += 1
        nlower += int(Knew < Kold)
    rp.check("B11.i", wrong == 0, "(11): 'the newcomer lowers the loss' from the general coefficient of the old and new "
             "networks (no memory) against the inequality f G (V - 2C + U) < 2G(V - C) + V dtau, %d random cases "
             "(old aggregate of 1 to 3 hosts): %d disagreements; the newcomer lowers the loss in %d" % (ncase, wrong, nlower))

    def Knew(f, tau2, k2, rho, phi=0.0):
        return Kpar_general(2.0, np.array([8.0, tau2]), np.array([1 - f, f]), np.array([1.0, k2]),
                            np.array([[1.0, rho], [rho, 1.0]]), [phi, phi])
    Kold = Kpar_general(2.0, np.array([8.0]), np.array([1.0]), np.array([1.0]), np.array([[1.0]]), [0.0])
    f1 = brentq(lambda f: Knew(f, 1.0, 1.0, 0.0) - Kold, 0.05, 0.99, xtol=1e-14)
    f2 = brentq(lambda f: Knew(f, 8.0, 3.0, 0.0) - Kold, 0.05, 0.99, xtol=1e-14)
    rp.check("B11.f065", abs(f1 - 0.65) < 1e-10 and Knew(0.4, 1.0, 1.0, 0.0) < Kold < Knew(0.8, 1.0, 1.0, 0.0),
             "central stay 2, old host stay 8 (K = %.6f): independent equal-variance newcomer of stay 1: K_new = K_old at "
             "f = %.12f (note 0.65); K_new(0.4) = %.6f (lower), K_new(0.8) = %.6f (higher)"
             % (Kold, f1, Knew(0.4, 1.0, 1.0, 0.0), Knew(0.8, 1.0, 1.0, 0.0)))
    rp.check("B11.f02", abs(f2 - 0.2) < 1e-10 and Knew(0.1, 8.0, 3.0, 0.0) < Kold < Knew(0.3, 8.0, 3.0, 0.0),
             "independent newcomer with three times the elasticity, stay 8: boundary f = %.12f (note 0.2); K_new(0.1) = %.6f, "
             "K_new(0.3) = %.6f" % (f2, Knew(0.1, 8.0, 3.0, 0.0), Knew(0.3, 8.0, 3.0, 0.0)))
    fs = np.linspace(0.001, 1.0, 400)
    mn = min(Knew(f, 8.0, 3.0, 0.8) - Kold for f in fs)
    rp.check("B11.rho08", mn > 0, "same newcomer with correlation 0.8: K_new - K_old > 0 at 400 shares from 0.001 to 1 "
             "(smallest %.3e, at the smallest share); formal root of (11): f = %.6f (negative, as the note says)"
             % (mn, 2 * 10 * (1 - 2.4) / (10 * (1 - 4.8 + 9))))
    # the exact one-stage ("scalar") benchmark, stay 3/4, newcomer with elasticity 3
    fine = (np.arange(-13.0, 13.0 + 1e-9, 0.025), None)
    fine = (fine[0], np.exp(-0.5 * fine[0] ** 2) / math.sqrt(2 * math.pi) * 0.025)
    quoted = {0.3: (0.146102538428, 0.116882030742, 0.175323046113, -0.000498405189041, 0.000726781891139),
              1.0: (0.006939193724, 0.005551354979, 0.008327032468, -0.000133848202874, 0.000171102416373)}
    for sg, (fq, flo, fhi, dlo, dhi) in quoted.items():
        D0 = scalar_parallel_D(0.0, sg, 0.75, 3.0)
        fb = brentq(lambda f: scalar_parallel_D(f, sg, 0.75, 3.0) - D0, 1e-4, 0.6, xtol=1e-14)
        e1 = scalar_parallel_D(flo, sg, 0.75, 3.0) - D0
        e2 = scalar_parallel_D(fhi, sg, 0.75, 3.0) - D0
        chk = scalar_parallel_D(fb, sg, 0.75, 3.0, grid=fine) - scalar_parallel_D(0.0, sg, 0.75, 3.0, grid=fine)
        rp.check("B11.scalar%.1f" % sg, close_num(fb, fq, 1e-8) and close_num(e1, dlo, 1e-6) and close_num(e2, dhi, 1e-6)
                 and abs(chk) < 1e-11,
                 "one stage, stay 3/4, independent newcomer with elasticity 3, sigma = %.1f, exact (two-dimensional "
                 "quadrature): D(f) = D(0) at f = %.12f (note %.12f; residual on a finer, wider grid %.1e); D(f) - D(0) at "
                 "f = %.12f: %+.15f (note %+.15f); at f = %.12f: %+.15f (note %+.15f); D(0) = %.9f"
                 % (sg, fb, fq, chk, flo, e1, dlo, fhi, e2, dhi, D0))
    # why the last digits of the note's sigma = 1 boundary differ: Gauss-Hermite orders
    vals = []
    for nn in (80, 160, 240):
        xg, wg = np.polynomial.hermite_e.hermegauss(nn)
        wg = wg / math.sqrt(2 * math.pi)
        D0 = scalar_parallel_D(0.0, 1.0, 0.75, 3.0, grid=(xg, wg))
        vals.append((nn, brentq(lambda f: scalar_parallel_D(f, 1.0, 0.75, 3.0, grid=(xg, wg)) - D0, 1e-4, 0.6, xtol=1e-14)))
    rp.p("      the sigma = 1 boundary with tensor Gauss-Hermite quadrature instead of the trapezoid rule: "
         + ", ".join("%d nodes: %.12f" % v for v in vals) + " (the note's 0.006939193724 lies between the 80- and "
         "160-node values; the converged value is 0.006939193740)")
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (12), (13), (14): an optional steady host; a bypass
# --------------------------------------------------------------------------------------------
def square_seq(amp, half):
    return np.array([[1 + amp]] * half + [[1 - amp]] * half)


def advantage13(q, hseq, tau0=2.0, tau1=2.0, taup=8.0, f=0.5):
    """(13): log Y + D_base - D_optional, every term an exact periodic root."""
    Y = 1 - f + f * q
    Dbase = floquet_root(model_ring([tau0, tau1], {0: 0}), hseq)
    Dopt = floquet_root(model_optional(tau0, tau1, taup, f, q, equal_R0=True), hseq)
    return math.log(Y) + Dbase - Dopt


def scalar_bypass_root(q, sigma, s, f):
    """Root y of E log[s + (1-s) e^y ((1-f) h + f q)] = 0 (fixed other rates)."""
    return scalar_root(lambda z: (1 - f) * np.exp(sigma * z - sigma * sigma / 2) + f * q, s)


def sec_B1214():
    rp = Rep("B (12), (13), (14): an optional steady host and a bypass of the fluctuating host")
    rng = np.random.default_rng(12012)
    worst12 = worst14 = worst_a12 = worst_a14 = worst14z = 0.0
    ncase = 300
    for case in range(ncase):
        tau0, tau1, taup = rand_taus(rng, 3)
        f, q = rng.random(), rng.uniform(0.02, 1.0)
        phi = rng.uniform(-0.9, 0.95)
        m12 = model_optional(tau0, tau1, taup, f, q)
        g = K_model(m12, [1.0], [0], [phi], [[1.0]])
        c = Kopt_closed(tau0, tau1, taup, f, q, phi)
        worst12 = max(worst12, abs(g[0] - c) / max(abs(g[0]), abs(c), 1e-3 * A_phi(abs(phi)) / (tau0 + tau1)))
        assert abs(g[1] - 1) < 1e-12
        m14 = model_bypass(tau0, tau1, f, q)
        g = K_model(m14, [1.0], [0], [phi], [[1.0]])
        c = Kbyp_closed(tau0, tau1, f, q, phi)
        Y = 1 - f + f * q
        al = (1 - f) / Y
        worst14 = max(worst14, abs(g[0] - c) / max(abs(g[0]), abs(c), 1e-3 * al * al * A_phi(abs(phi)) / (tau0 + tau1)))
        assert abs(g[1] - 1) < 1e-12
        g0 = K_model(m14, [1.0], [0], [0.0], [[1.0]])[0]
        worst14z = max(worst14z, abs(g0 - al * al / (2 * (tau0 + al * tau1))) / (abs(g0) + 1e-9))
        if case < 30:
            worst_a12 = max(worst_a12, abs(alt_K(m12, [1.0]) - Kopt_closed(tau0, tau1, taup, f, q, -1.0))
                            / max(abs(Kopt_closed(tau0, tau1, taup, f, q, -1.0)), 1e-2 / (tau0 + tau1)))
            worst_a14 = max(worst_a14, abs(alt_K(m14, [1.0]) - Kbyp_closed(tau0, tau1, f, q, -1.0))
                            / max(abs(Kbyp_closed(tau0, tau1, f, q, -1.0)), 1e-2 * al * al / (tau0 + tau1)))
    rp.check("B12.i", worst12 < TOL_ALG, "(12) against the general coefficient, %d random (stays, f, q, memory -0.9 to "
             "0.95), returns divided by Y so that the mean matrix is at threshold: %.2e" % (ncase, worst12))
    rp.check("B14.i", worst14 < TOL_ALG and worst14z < TOL_ALG, "(14) against the general coefficient, %d random cases: "
             "%.2e; without memory alpha^2/[2(tau0 + alpha tau1)]: %.2e" % (ncase, worst14, worst14z))
    rp.check("B12.alt", worst_a12 < TOL_FLOQ and worst_a14 < TOL_FLOQ, "(12) and (14) continued to memory -1 against the "
             "exact loss in a strictly alternating environment, 30 cases each: %.2e and %.2e" % (worst_a12, worst_a14))
    # (13) is an identity: the root at fixed other rates is -log Y + the root at equal mean R0
    worst = 0.0
    for case in range(100):
        tau0, tau1, taup = rand_taus(rng, 3)
        f, q = rng.random(), rng.uniform(0.05, 1.0)
        N = int(rng.integers(2, 9))
        h = np.exp(rng.uniform(0.2, 1.5) * rng.standard_normal(N))
        h = (h / h.mean())[:, None]
        Y = 1 - f + f * q
        y_fixed = floquet_root(model_optional(tau0, tau1, taup, f, q, equal_R0=False), h, -10, 10)
        y_equal = floquet_root(model_optional(tau0, tau1, taup, f, q, equal_R0=True), h, -10, 10)
        yb_fixed = floquet_root(model_bypass(tau0, tau1, f, q, equal_R0=False), h, -10, 10)
        yb_equal = floquet_root(model_bypass(tau0, tau1, f, q, equal_R0=True), h, -10, 10)
        worst = max(worst, abs(y_fixed - (y_equal - math.log(Y))), abs(yb_fixed - (yb_equal - math.log(Y))))
    rp.check("B13.identity", worst < 1e-11, "(13): the threshold at fixed other rates is the equal-R0 loss minus log Y, "
             "optional and bypass networks, 100 random periodic environments: largest difference %.1e" % worst)
    h20 = square_seq(0.6, 10)
    qb = brentq(lambda q: advantage13(q, h20), 0.5, 1.0, xtol=1e-14)
    e1, e2 = advantage13(qb - 0.01, h20), advantage13(qb + 0.01, h20)
    rp.check("B13.q", close_num(qb, 0.859387632729) and close_num(e1, -0.00556253815021, 1e-8) and close_num(e2, 0.00552874654846, 1e-8),
             "ten steps at 1.6 then ten at 0.4, base stays (2,2), f = 1/2, extra stay 8, exact: (13) is zero at q = %.12f "
             "(note 0.859387632729); at q - 0.01: %+.14f (note -0.00556253815021); at q + 0.01: %+.14f (note "
             "+0.00552874654846)" % (qb, e1, e2))
    h2 = square_seq(0.6, 1)
    adv = advantage13(1.0, h2)
    Db = floquet_root(model_ring([2.0, 2.0], {0: 0}), h2)
    Do = floquet_root(model_optional(2.0, 2.0, 8.0, 0.5, 1.0), h2)
    rp.check("B13.alt", close_num(adv, -0.022920059653, 1e-9), "strict alternation of 1.6 and 0.4, q = 1: (13) = %.12f "
             "(note -0.022920059653); D_base = %.10f, D_optional = %.10f (the optional host with perfect survival raises "
             "the loss here)" % (adv, Db, Do))
    qs = np.linspace(0.3, 1.0, 36)
    advs = [advantage13(q, h20) for q in qs]
    rp.check("B13.monotone", all(b > a for a, b in zip(advs, advs[1:])),
             "(13) increases with q on 36 values from 0.3 to 1 in the period-20 environment (from %.5f to %.5f)" % (advs[0], advs[-1]))
    # complete bypass
    worst = 0.0
    for case in range(50):
        tau0, tau1 = rand_taus(rng, 2)
        N = int(rng.integers(2, 9))
        h = np.exp(rng.standard_normal(N))
        h = (h / h.mean())[:, None]
        worst = max(worst, abs(floquet_root(model_bypass(tau0, tau1, 1.0, rng.uniform(0.1, 1)), h)))
    rp.check("B14.f1", worst < 1e-12, "complete bypass (f = 1): the exact loss is zero in 50 random periodic environments "
             "(largest |D| %.1e)" % worst)
    # the exact one-stage bypass benchmark
    quoted = {0.3: (0.966801568423, -0.00515708684824, 0.00513033594805),
              1.0: (0.687697785698, -0.00697579753729, 0.00692252295029)}
    for sg, (qq, alo, ahi) in quoted.items():
        Dbase = scalar_root(lambda z: np.exp(sg * z - sg * sg / 2), 0.5)
        adv = lambda q: Dbase - scalar_bypass_root(q, sg, 0.5, 0.5)
        qb = brentq(adv, 0.2, 1.0, xtol=1e-14)
        e1, e2 = adv(qb - 0.01), adv(qb + 0.01)
        rp.check("B14.scalar%.1f" % sg, close_num(qb, qq) and close_num(e1, alo, 1e-8) and close_num(e2, ahi, 1e-8),
                 "one stage, stay 1/2, allocation 1/2, sigma = %.1f, exact: the bypass breaks even at relative yield q = "
                 "%.12f (note %.12f); advantage at q - 0.01: %+.14f (note %+.14f); at q + 0.01: %+.14f (note %+.14f); "
                 "D_base = %.9f" % (sg, qb, qq, e1, alo, e2, ahi, Dbase))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (15): independent hosts in sequence
# --------------------------------------------------------------------------------------------
def Kseries(taus, phi):
    """Coefficient of sigma^2 when every transfer of the ring has its own independent driver of
    the same variance and memory: general coefficient, summed over the drivers."""
    n = len(taus)
    mdl = model_ring(taus, {k: k for k in range(n)})
    return K_model(mdl, [1.0] * n, list(range(n)), [phi] * n, np.eye(n))[0]


def sec_B15():
    rp = Rep("B (15): independent hosts in sequence")
    rng = np.random.default_rng(15015)
    worst = 0.0
    ncase = 300
    for case in range(ncase):
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n)
        sig = rng.uniform(0.2, 2.0, n)
        phis = rng.uniform(-0.9, 0.95, n)
        mdl = model_ring(taus, {k: k for k in range(n)}, gauge=rand_gauge(rng, n))
        Kgen = K_model(mdl, list(sig), list(range(n)), list(phis), np.eye(n))[0]
        cl = sum(sig[j] ** 2 * Kg_closed(taus, phis[j]) for j in range(n))
        worst = max(worst, abs(Kgen - cl) / max(abs(Kgen), abs(cl), 1e-3 * (sig ** 2).sum() / taus.sum()))
    rp.check("B15.i", worst < TOL_ALG, "(15) D = sum_j sigma_j^2 K_g(stays, phi_j) against the general coefficient with one "
             "independent driver per transfer, %d random rings (1 to 5 stages, different variances and memories): "
             "largest relative difference %.2e" % (ncase, worst))
    # the repartition boundaries
    K2 = Kseries([10.0, 10.0], 0.9)
    dif = lambda a: Kseries([a, (20 - a) / 2, (20 - a) / 2], 0.9) - K2
    ab = brentq(dif, 12.0, 17.99, xtol=1e-13)
    e1, e2 = dif(ab - 0.3), dif(ab + 0.3)
    rp.check("B15.a", close_num(ab, 17.3757414756) and close_num(e1, 0.0244677150771, 1e-8) and close_num(e2, -0.0260160842042, 1e-8),
             "generation time 20, memory 0.9, three independent hosts with stays (a, (20-a)/2, (20-a)/2) against two with "
             "(10,10): equal coefficients at a = %.10f (note 17.3757414756); new minus old at a - 0.3: %+.13f (note "
             "+0.0244677150771); at a + 0.3: %+.13f (note -0.0260160842042); two-host coefficient %.6f"
             % (ab, e1, e2, K2))
    difp = lambda ph: Kseries([18.0, 1.0, 1.0], ph) - Kseries([10.0, 10.0], ph)
    pb = brentq(difp, 0.5, 0.95, xtol=1e-14)
    e1, e2 = difp(pb - 0.03), difp(pb + 0.03)
    rp.check("B15.phi", close_num(pb, 0.816984251337) and close_num(e1, 0.0129879581011, 1e-8) and close_num(e2, -0.0163098249777, 1e-8),
             "stays (18,1,1) against (10,10): equal coefficients at memory %.12f (note 0.816984251337); difference at "
             "memory - 0.03: %+.13f (note +0.0129879581011); at + 0.03: %+.13f (note -0.0163098249777)" % (pb, e1, e2))
    vals = [(n, Kseries([20.0 / n] * n, 0.0)) for n in (1, 2, 3, 4)]
    rp.check("B15.phi0", all(abs(v - n / 40.0) < 1e-12 for n, v in vals),
             "no memory, equal stays, generation time 20: coefficient n/(2G) = " + ", ".join("%.6f (n=%d)" % (v, n) for n, v in vals))
    return rp.done()


# --------------------------------------------------------------------------------------------
# B (16), (17): the bound, in exact periodic environments; mu = 0
# --------------------------------------------------------------------------------------------
def bound17(model, ells):
    """(17): sum_e c_e l_e / b with c_e = w_i a_e v_j the frequency of edge e in the mean matrix,
    b the frequency of the birth edges.  Valid when multiplier i multiplies edges only."""
    A = model.mean()
    lam, v, w = perron(A)
    b = w @ model.birth() @ v
    return sum((w @ model.E(i) @ v) * ells[i] for i in range(model.I)) / b


def rand_positive_seq(rng, N, I):
    sg = rng.uniform(0.3, 2.5, I)
    kind = rng.integers(0, 3)
    if kind == 0:
        h = np.exp(sg * rng.standard_normal((N, I)))
    elif kind == 1:
        h = rng.random((N, I)) ** rng.uniform(1, 6) + 1e-3
    else:                                           # one driver, random signed powers
        base = np.exp(rng.standard_normal(N))
        h = base[:, None] ** (sg * np.where(rng.random(I) < 0.5, -1, 1))[None, :]
    return h / h.mean(0)


def random_route_model(rng):
    kind = int(rng.integers(0, 4))
    if kind == 0:
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n)
        if rng.random() < 0.25:
            taus = np.ones(n) + (rng.random(n) < 0.3) * rng.random(n)
        ks = [k for k in range(n) if rng.random() < 0.7] or [0]
        return "ring", model_ring(taus, {k: i for i, k in enumerate(ks)})
    if kind == 1:
        J = int(rng.integers(2, 4))
        return "parallel", model_parallel(float(rand_taus(rng, 1)[0]), rand_taus(rng, J), rng.dirichlet(np.ones(J)))
    tau0, tau1, taup = rand_taus(rng, 3)
    if kind == 2:
        return "optional", model_optional(tau0, tau1, taup, rng.random(), rng.uniform(0.05, 1))
    return "bypass", model_bypass(tau0, tau1, rng.uniform(0, 0.95), rng.uniform(0.05, 1))


def sec_B1617():
    rp = Rep("B (16), (17): the bound D <= sum of visit-weighted log penalties, searched in exact periodic environments; "
             "contact limitation with no waiting mortality")
    rng = np.random.default_rng(16016)
    # route weights of (17) as the note states them
    worst = 0.0
    for case in range(100):
        J = 3
        tau0 = float(rand_taus(rng, 1)[0])
        taus = rand_taus(rng, J)
        w = rng.dirichlet(np.ones(J))
        ells = rng.random(J)
        worst = max(worst, abs(bound17(model_parallel(tau0, taus, w), ells) - (w * ells).sum()))
        t0, t1, tp = rand_taus(rng, 3)
        f, q = rng.random(), rng.uniform(0.05, 1)
        worst = max(worst, abs(bound17(model_optional(t0, t1, tp, f, q), [1.0]) - 1.0))
        worst = max(worst, abs(bound17(model_bypass(t0, t1, f, q), [1.0]) - (1 - f) / (1 - f + f * q)))
        n = int(rng.integers(1, 6))
        ts = rand_taus(rng, n)
        worst = max(worst, abs(bound17(model_ring(ts, {k: k for k in range(n)}), np.ones(n)) - n))
    rp.check("B17.weights", worst < 1e-11, "the weights of (17) from the Perron vectors: ring transfers 1 each, parallel "
             "entrances w_j, optional host's entrance 1, bypassed entrance alpha (100 random cases each): %.1e" % worst)
    counts = {}
    viol = 0
    mingrow = 1e9
    maxratio = -1e9
    ncase = 3000
    ex_tight = None
    for case in range(ncase):
        kind, mdl = random_route_model(rng)
        N = int(rng.integers(2, 13))
        h = rand_positive_seq(rng, N, mdl.I)
        ells = -np.log(h).mean(0)
        yb = bound17(mdl, ells)
        gb = floquet_growth(mdl, h, yb) * N
        counts[kind] = counts.get(kind, 0) + 1
        mingrow = min(mingrow, gb)
        if gb < -1e-9:
            viol += 1
        if case % 6 == 0:
            D = floquet_root(mdl, h, -20, 20)
            if yb > 1e-6:
                maxratio = max(maxratio, D / yb)
                if D / yb > 0.999 and ex_tight is None:
                    ex_tight = (kind, D, yb)
    rp.check("B16.periodic", viol == 0, "(16)-(17) in %d random exact periodic environments (%s; period 2 to 12; log-amplitude "
             "0.3 to 2.5; independent, or one driver with signed powers): growth at the bound is non-negative in all; "
             "smallest log multiplier over a period %.2e; violations %d; largest D/bound among %d solved cases %.6f"
             % (ncase, ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items())), mingrow, viol, ncase // 6, maxratio))
    # contact: D <= l_transfer + (tau_k - 1) l_stay; and the transfer-only bound (not claimed by the note)
    viol = 0
    viol_t = 0
    worst_t = 0.0
    ex_t = None
    mingrow = 1e9
    ncase = 3000
    for case in range(ncase):
        n = int(rng.integers(1, 5))
        taus = rand_taus(rng, n) + 0.05
        k = int(rng.integers(0, n))
        d = rng.uniform(0.0, 1.0) ** (2 if case % 2 else 1)
        s_k = 1 - 1 / taus[k]
        c = (1 - d) * (1 - s_k)
        N = int(rng.integers(2, 13))
        u = rng.random(N) ** rng.uniform(0.5, 4)
        u = u - u.mean()
        amax_lo = 1.0 / max(-u.min(), 1e-12)                        # h >= 0
        amax_hi = (s_k / c) / max(u.max(), 1e-12) if c > 0 else 1e9   # stay >= 0
        amp = rng.uniform(0.3, 0.98) * min(amax_lo, amax_hi)
        h = 1 + amp * u
        mdl = model_ring(taus, {}, contact=(k, d, 0))
        l_tr = -np.log(h).mean()
        l_st = -np.log((s_k - c * (h - 1)) / s_k).mean()
        yb = l_tr + (taus[k] - 1) * l_st
        gb = floquet_growth(mdl, h[:, None], yb) * N
        mingrow = min(mingrow, gb)
        if gb < -1e-9:
            viol += 1
        gt = floquet_growth(mdl, h[:, None], l_tr) * N
        if gt < -1e-9:
            viol_t += 1
            D = floquet_root(mdl, h[:, None], -20, 20)
            if D - l_tr > worst_t:
                worst_t = D - l_tr
                ex_t = (taus.copy(), k, d, h.copy(), D, l_tr, yb)
    rp.check("B16.contact", viol == 0, "contact version D <= l_transfer + (tau_k - 1) l_stay in %d random exact periodic "
             "environments (rings of 1 to 4 stages, any d, amplitude up to 98%% of the largest valid one): violations %d; "
             "smallest log multiplier at the bound %.2e" % (ncase, viol, mingrow))
    rp.p("      the transfer-only bound D <= l_transfer (which the note says it has neither proved nor refuted for contact "
         "limitation) is violated in %d of these %d cases." % (viol_t, ncase))
    if ex_t is not None:
        taus, k, d, h, D, l_tr, yb = ex_t
        rp.p("      largest excess found: stays %s, contact stage %d, d = %.4f, period %d, multipliers %s: D = %.6f, "
             "l_transfer = %.6f, full bound %.6f" % (np.array2string(taus, precision=3), k, d, len(h),
                                                    np.array2string(h, precision=3), D, l_tr, yb))
    # mu = 0: every environmental matrix conserves the left vector of the mean matrix
    worst_w = 0.0
    worst_D = 0.0
    ncase = 2000
    for case in range(ncase):
        n = int(rng.integers(1, 6))
        taus = rand_taus(rng, n) + 0.02
        k = int(rng.integers(0, n))
        mdl = model_ring(taus, {}, contact=(k, 0.0, 0), gauge=rand_gauge(rng, n))
        lam, v, w = perron(mdl.mean())
        hmax = 1 + (1 - 1 / taus[k]) * taus[k]          # stay rate 1 - c h >= 0 with c = 1/tau_k
        hs = rng.uniform(0, hmax, 6)
        for hv in hs:
            worst_w = max(worst_w, np.abs(w @ mdl.mat(0.0, [hv]) - w).max() / w.max())
        if case < 400:
            N = int(rng.integers(2, 13))
            u = rng.random(N)
            u -= u.mean()
            amp = rng.uniform(0.5, 1.0) * min(1 / -u.min(), (hmax - 1) / u.max())
            worst_D = max(worst_D, abs(floquet_root(mdl, (1 + amp * u)[:, None], -5, 5)))
    rp.check("B.mu0", worst_w < 1e-12 and worst_D < 1e-11,
             "contact limitation with no waiting mortality (d = 0): w^T A(h) = w^T for the left vector w of the mean matrix, "
             "%d random rings x 6 random valid multipliers each: largest relative defect %.1e; exact periodic loss in 400 "
             "of them at up to the full valid amplitude: largest |D| %.1e" % (ncase, worst_w, worst_D))
    return rp.done()


# --------------------------------------------------------------------------------------------
# tool (c): products of the actual random matrices
# --------------------------------------------------------------------------------------------
def gauss_chunks(rng, phis, corr, npairs, T, chunk=1000):
    """Stationary Gaussian AR(1) series z_l(t) = phi_l z_l(t-1) + innovation, unit variance,
    contemporaneous correlation corr (innovation covariance corr_ij (1 - phi_i phi_j)).
    Yields arrays (steps, npairs, L)."""
    phis = np.asarray(phis, float)
    corr = np.asarray(corr, float)
    L = len(phis)
    Sig = corr * (1 - np.outer(phis, phis))
    ch = np.linalg.cholesky(Sig + 1e-13 * np.eye(L))
    c0 = np.linalg.cholesky(corr + 1e-13 * np.eye(L))
    z = rng.standard_normal((npairs, L)) @ c0.T
    done = 0
    while done < T:
        m = min(chunk, T - done)
        eps = rng.standard_normal((m, npairs, L)) @ ch.T
        out = np.empty_like(eps)
        for l in range(L):
            out[:, :, l] = lfilter([1.0], [1.0, -phis[l]], eps[:, :, l], axis=0, zi=(phis[l] * z[:, l])[None, :])[0]
        z = out[-1]
        yield out
        done += m


def make_gauss_env(phis, corr, lat, ksig, hfuns=None):
    """Multiplier i is exp(ksig_i z_{lat_i} - ksig_i^2/2) (mean one), or hfuns[i](z_{lat_i}).
    Each pair of replicates uses z and -z (antithetic): rows [0, npairs) and [npairs, 2 npairs)."""
    lat = list(lat)
    ksig = np.asarray(ksig, float)

    def make(seed, T, npairs):
        rng = np.random.default_rng(seed)
        for z in gauss_chunks(rng, phis, corr, npairs, T):
            zz = np.concatenate([z, -z], axis=1)[:, :, lat]
            if hfuns is None:
                yield np.exp(ksig * zz - 0.5 * ksig * ksig)
            else:
                yield np.stack([hfuns[i](zz[:, :, i]) for i in range(len(lat))], axis=-1)
    return make


def make_markov_env(X, pi, phi, amp):
    """One bounded multiplier h = 1 + amp X, X a finite-state Markov chain with transition
    phi I + (1 - phi) 1 pi (stay with probability phi, otherwise redraw from pi).  E X = 0.
    The 2 npairs replicates are independent."""
    X = np.asarray(X, float)
    cum = np.cumsum(pi)
    cum[-1] = 1.0

    def make(seed, T, npairs):
        rng = np.random.default_rng(seed)
        R = 2 * npairs
        state = np.searchsorted(cum, rng.random(R))
        done = 0
        while done < T:
            m = min(1000, T - done)
            keep = rng.random((m, R)) < phi
            fresh = np.searchsorted(cum, rng.random((m, R)))
            idx = np.where(keep, -1, np.arange(m)[:, None])
            idx = np.maximum.accumulate(idx, axis=0)
            st = np.where(idx >= 0, np.take_along_axis(fresh, np.maximum(idx, 0), axis=0), state[None, :])
            state = st[-1]
            yield (1 + amp * X[st])[:, :, None]
            done += m
    return make


def run_growth(model, ys, hgen, burn):
    """Mean log growth per step of the product of A(t, y) for each y in ys (same environments).
    Returns r (R, Y) and the time mean of the multipliers (R, I) over the same steps.
    The population starts in stage 0 only.  In a ring without any staying the stages are separate
    cohorts that never mix; the log of their sum has a bias of order T^-1/2 that replicates share
    (found the hard way: see the attempt1 file).  One cohort has no such bias.  With any staying
    the start is forgotten during the burn-in."""
    n = model.n
    Y = len(ys)
    Ac = np.stack([model.C0 + math.exp(y) * model.C1 for y in ys])
    Ad = np.stack([model.B0 + math.exp(y) * model.B1 for y in ys], axis=1)
    x = None
    t = 0
    for h in hgen:
        m, R, I = h.shape
        if x is None:
            x = np.zeros((R, Y, n, 1))
            x[:, :, 0, 0] = 1.0                      # one cohort: see the note on rings without staying
            acc = np.zeros((R, Y))
            hs = np.zeros((R, I))
            cnt = 0
        sms = np.empty((m, R, Y))
        for tt in range(m):
            A = Ac + np.tensordot(h[tt], Ad, axes=(1, 0))
            x = A @ x
            sm = x.sum(axis=2, keepdims=True)
            x /= sm
            sms[tt] = sm[:, :, 0, 0]
        lo = max(0, burn - t)
        if lo < m:
            acc += np.log(sms[lo:]).sum(0)
            hs += h[lo:].sum(0)
            cnt += m - lo
        t += m
    return acc / cnt, hs / cnt


def cv_beta(model, ys):
    """First-order coefficient of the log growth in (h_i - 1): w^T E_i v / lambda at each y."""
    out = np.zeros((len(ys), model.I))
    for a, y in enumerate(ys):
        lam, v, w = perron(model.mean(y))
        for i in range(model.I):
            out[a, i] = (w @ (model.B0[i] + math.exp(y) * model.B1[i]) @ v) / lam
    return out


def model_evaluator(model, env, burn=2000, use_cv=True):
    """evalr(ys, T, npairs, seed) -> growth per step, shape (npairs, Y), each row the mean of a
    pair of replicates.  The time mean of (h - 1), whose expectation is exactly zero, is
    subtracted with a fixed coefficient (a control variate: it removes the first-order noise
    and leaves the expectation unchanged)."""
    def evalr(ys, T, npairs, seed):
        r, hbar = run_growth(model, ys, env(seed, T + burn, npairs), burn)
        if use_cv:
            r = r - (hbar - 1) @ cv_beta(model, ys).T
        return 0.5 * (r[:npairs] + r[npairs:])
    return evalr


def find_root(evalr, y0, T, npairs, seed, width=None, extra_ys=()):
    """The y at which the growth rate is zero.  A short pilot run on a wide grid of y brackets
    the root; the main run evaluates three y around it in the same environments and each pair
    of replicates gives its own root (quadratic through the three points).  Returns the mean
    root, its standard error over the pairs, and diagnostics."""
    width = width if width is not None else max(abs(y0), 0.01)
    grid = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    yc = y0
    Tp = 4000
    for it in range(12):
        ys = yc + width * grid
        rp_ = evalr(ys, Tp, 6, seed + 7919)
        mean = rp_.mean(0)
        if mean[0] < 0 < mean[-1]:
            break
        yc = yc + 3 * width if mean[-1] <= 0 else yc - 3 * width
        width *= 2
    else:
        raise RuntimeError("root not bracketed")
    j = int(np.argmax(mean > 0))
    slope = (mean[j] - mean[j - 1]) / (ys[j] - ys[j - 1])
    yp = ys[j - 1] - mean[j - 1] / slope
    spread = rp_[:, j].std(ddof=1) / slope
    delta = max(4 * spread * math.sqrt(Tp / T), 0.02 * abs(yp), 1e-5)
    ys = np.concatenate([yp + delta * np.array([-1.0, 0.0, 1.0]), np.asarray(extra_ys, float)])
    rm = evalr(ys, T, npairs, seed)
    extra = [(float(rm[:, 3 + a].mean()), float(rm[:, 3 + a].std(ddof=1) / math.sqrt(npairs))) for a in range(len(extra_ys))]
    a0 = rm[:, 1]
    a1 = (rm[:, 2] - rm[:, 0]) / (2 * delta)
    a2 = (rm[:, 2] + rm[:, 0] - 2 * rm[:, 1]) / (2 * delta * delta)
    disc = np.maximum(a1 * a1 - 4 * a0 * a2, 0.0)
    u = -2 * a0 / (a1 + np.sqrt(disc))
    return dict(D=float(yp + u.mean()), se=float(u.std(ddof=1) / math.sqrt(npairs)), roots=yp + u,
                slope=float(a1.mean()), umax=float(np.abs(u).max() / delta), yp=float(yp), delta=float(delta), extra=extra)


def growth_at(evalr, y, T, npairs, seed):
    r = evalr(np.array([y]), T, npairs, seed)[:, 0]
    return float(r.mean()), float(r.std(ddof=1) / math.sqrt(npairs))


def lognormal_env(sigma, phis, corr=None, lat=None, kappa=None):
    L = len(phis)
    corr = np.eye(L) if corr is None else corr
    lat = list(range(L)) if lat is None else lat
    kappa = np.ones(len(lat)) if kappa is None else np.asarray(kappa, float)
    return make_gauss_env(phis, corr, lat, sigma * kappa)


def within_small(Dsim, se, pred):
    return abs(Dsim - pred) <= NSE * se + SMALL_REL * abs(pred)


# --------------------------------------------------------------------------------------------
# HS: the simulator against known answers
# --------------------------------------------------------------------------------------------
def sec_HS():
    rp = Rep("HS. The simulator against known answers")
    # one stage, lognormal, any memory: exact root by quadrature
    for sg, phi, s, seed in ((1.0, 0.5, 0.75, 11), (0.3, -0.6, 0.5, 12)):
        mdl = model_ring([1 / (1 - s)], {0: 0})
        ex = scalar_root(lambda z: np.exp(sg * z - sg * sg / 2), s)
        res = find_root(model_evaluator(mdl, lognormal_env(sg, [phi])), 0.4 * sg * sg, 60000, 24, seed)
        rp.check("HS.scalar_sigma%.1f" % sg, abs(res["D"] - ex) <= NSE * res["se"],
                 "one stage, stay %.2f, lognormal sigma %.1f, memory %.1f: simulated loss %.6f (SE %.6f), quadrature %.6f"
                 % (s, sg, phi, res["D"], res["se"], ex))
    # one stage, two-state Markov multiplier: the exact root is a two-term sum
    s, amp, phi = 0.6, 0.8, 0.7
    X, pi = np.array([-0.5, 2.0]), np.array([0.8, 0.2])
    ex = brentq(lambda y: float(np.sum(pi * np.log(s + (1 - s) * math.exp(y) * (1 + amp * X)))), -5, 5, xtol=1e-15)
    mdl = model_ring([1 / (1 - s)], {0: 0})
    res = find_root(model_evaluator(mdl, make_markov_env(X, pi, phi, amp)), 0.1, 60000, 24, 13)
    rp.check("HS.markov", abs(res["D"] - ex) <= NSE * res["se"],
             "one stage, two-state Markov multiplier (values 0.6 and 2.6, memory 0.7): simulated loss %.6f (SE %.6f), exact %.6f"
             % (res["D"], res["se"], ex))
    # the ring formula of the manuscript at small noise (a result established before these notes)
    taus, phi, sg = [3.0, 2.0, 4.0], 0.6, 0.05
    mdl = model_ring(taus, {1: 0})
    pred = sg * sg * Kg_closed(taus, phi)
    res = find_root(model_evaluator(mdl, lognormal_env(sg, [phi])), pred, 200000, 32, 14)
    rp.check("HS.ring_small", within_small(res["D"], res["se"], pred),
             "gain-only ring (3,2,4), memory 0.6, sigma 0.05: simulated loss %.3e (SE %.1e), manuscript's second-order value %.3e"
             % (res["D"], res["se"], pred))
    return rp.done()


# --------------------------------------------------------------------------------------------
# S: second-order values of note B against exact products at small noise
# --------------------------------------------------------------------------------------------
def small_noise_case(rp, key, text, model, env, pred, seed, T=200000, npairs=32):
    res = find_root(model_evaluator(model, env), pred, T, npairs, seed)
    rp.check(key, within_small(res["D"], res["se"], pred),
             "%s: simulated D = %.4e (SE %.1e), second-order value %.4e, ratio %.4f" % (text, res["D"], res["se"], pred, res["D"] / pred))
    return res


def sec_SA():
    rp = Rep("S-A. Second-order values against products of the actual random matrices at small noise, part 1 "
             "(criterion: 4 SE + 5%)")
    Xs, pis = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    # (2) contact, bounded symmetric two-state host with geometric memory
    for (taus, k, d, phi, amp, seed) in (([5.0, 2.0], 0, 0.5, 0.8, 0.06, 21), ([3.0, 4.0, 2.0], 1, 0.3, 0.6, 0.06, 22),
                                         ([5.0, 20.0], 0, 0.1, 0.8, 0.06, 23)):
        mdl = model_ring(taus, {}, contact=(k, d, 0))
        pred = amp * amp * Kc_closed(taus, k, d, phi)
        small_noise_case(rp, "S2.%d" % seed, "(2) contact stays %s, contact stage %d, d = %.1f, two-state host 1 +- %.2f with "
                         "memory %.1f" % (taus, k, d, amp, phi), mdl, make_markov_env(Xs, pis, phi, amp), pred, seed)
    # (5) curved response on a ring with memory: the transfer is F(H)/E F(H), H lognormal
    sg = 0.05
    for (kind, x, par, taus, phi, seed) in (("III", 0.6, None, [2.0, 3.0], 0.5, 31), ("power", 1.0, 2.5, [3.0, 2.0], 0.7, 32),
                                            ("II", 1.5, None, [2.0, 2.0], -0.3, 33)):
        F = resp_F(kind, par)
        den = Ez(lambda z: F(x * np.exp(sg * z - sg * sg / 2)))
        hf = (lambda z, F=F, x=x, den=den: F(x * np.exp(sg * z - sg * sg / 2)) / den)
        env = make_gauss_env([phi], np.eye(1), [0], np.ones(1), hfuns=[hf])
        eta = eta_chi_numeric(F, x)[0]
        pred = eta * eta * sg * sg * Kg_closed(taus, phi)
        small_noise_case(rp, "S5.%d" % seed, "(5) type %s%s at x = %.1f on ring %s, memory %.1f, host sigma %.2f, rate "
                         "normalised to its mean (baseline ii), eta = %.4f" % (kind, "" if par is None else " %.1f" % par, x, taus, phi, sg, eta),
                         model_ring(taus, {0: 0}), env, pred, seed)
    # (6), (7)
    for (taus, a, phi, seed) in (([2.0, 3.0, 4.0], [1.0, -0.7, 0.0], 0.6, 41), ([2.0, 3.0, 4.0], [1.0, 0.0, 0.5], 0.6, 42),
                                 ([2.0, 2.0], [1.0, -1.0], 0.8, 43), ([2.0, 2.0], [1.0, -1.0], -0.5, 44),
                                 ([3.0, 3.0], [1.0, 1.0], 0.7, 45)):
        idx = [k for k in range(len(taus)) if a[k] != 0]
        mdl = model_ring(taus, {k: i for i, k in enumerate(idx)})
        env = lognormal_env(sg, [phi], lat=[0] * len(idx), kappa=[a[k] for k in idx])
        pred = sg * sg * K6_closed(taus, a, phi)
        small_noise_case(rp, "S6.%d" % seed, "(6)/(7) ring %s, one host with elasticities %s, memory %.1f, sigma %.2f"
                         % (taus, a, phi, sg), mdl, env, pred, seed)
    return rp.done()


def sec_SB():
    rp = Rep("S-B. Second-order values against products of the actual random matrices at small noise, part 2 "
             "(criterion: 4 SE + 5%)")
    sg = 0.05
    # (8), (9), (10)
    tau0, taus, w, kap = 2.0, np.array([8.0, 1.0]), np.array([0.6, 0.4]), np.array([1.0, 1.5])
    Rc = np.array([[1.0, 0.4], [0.4, 1.0]])
    mdl = model_parallel(tau0, taus, w)
    for phi, seed in ((0.0, 51), (0.7, 52)):
        env = lognormal_env(sg, [phi, phi], corr=Rc, kappa=kap)
        pred = sg * sg * Kpar_closed(tau0, taus, w, kap, Rc, phi)
        small_noise_case(rp, "S9.%d" % seed, "(%s) parallel hosts, central stay 2, host stays (8,1), shares (0.6,0.4), "
                         "log-sd multipliers (1,1.5), correlation 0.4, memory %.1f" % ("8" if phi == 0 else "9", phi), mdl, env, pred, seed)
    phis = [0.2, 0.8]
    kap1 = np.array([1.0, 1.0])
    w1 = np.array([0.6, 0.4])
    env = lognormal_env(sg, phis, corr=Rc, kappa=kap1)
    pred = sg * sg * Kpar_closed_multi(tau0, taus, w1, kap1, Rc, phis)
    small_noise_case(rp, "S10.53", "(10) the same network with unit multipliers and memories (0.2, 0.8), correlation 0.4",
                     model_parallel(tau0, taus, w1), env, pred, 53)
    # (12), (14)
    env = lognormal_env(sg, [0.8])
    pred = sg * sg * Kopt_closed(2.0, 2.0, 8.0, 0.5, 0.8, 0.8)
    small_noise_case(rp, "S12.61", "(12) optional host: stays (2,2), f = 1/2, extra stay 8, q = 0.8, memory 0.8",
                     model_optional(2.0, 2.0, 8.0, 0.5, 0.8), env, pred, 61)
    pred = sg * sg * Kbyp_closed(2.0, 4.0, 0.5, 0.5, 0.8)
    small_noise_case(rp, "S14.62", "(14) bypass: central stay 2, host stay 4, f = 1/2, q = 1/2, memory 0.8",
                     model_bypass(2.0, 4.0, 0.5, 0.5), env, pred, 62)
    # (15)
    taus = [3.0, 5.0, 2.0]
    phis = [0.5, 0.8, -0.4]
    kap = [1.0, 0.7, 1.2]
    env = lognormal_env(sg, phis, kappa=kap)
    pred = sg * sg * sum(kap[j] ** 2 * Kg_closed(taus, phis[j]) for j in range(3))
    small_noise_case(rp, "S15.71", "(15) ring (3,5,2), three independent hosts with memories (0.5,0.8,-0.4) and log-sd "
                     "multipliers (1,0.7,1.2)", model_ring(taus, {0: 0, 1: 1, 2: 2}), env, pred, 71)
    return rp.done()


# --------------------------------------------------------------------------------------------
# SP: numbers quoted in note B that need products of random matrices
# --------------------------------------------------------------------------------------------
def cmp_note(mine, se, theirs, se_theirs):
    return abs(mine - theirs) <= NSE * math.sqrt(se * se + se_theirs * se_theirs)


def sec_SP1():
    rp = Rep("S-spot 1. Note B: the log cosh identity (7a), D_++ = 2 d_s, and a spot check of the note's simulation "
             "tables for stays (2,2) (criterion: 4 SE; against a simulated number of the note, 4 combined SE)")
    # (7a): growth of the raw H and 1/H ring at the constant-mean-host threshold
    mdl = model_ring([2.0, 2.0], {0: 0, 1: 1})
    for sg, phi, seed in ((0.3, 0.0, 81), (1.0, 0.0, 82), (0.3, 0.8, 83), (1.0, 0.8, 84)):
        hf = [lambda z, sg=sg: np.exp(sg * z - sg * sg / 2), lambda z, sg=sg: np.exp(-sg * z + sg * sg / 2)]
        env = make_gauss_env([phi], np.eye(1), [0, 0], np.ones(2), hfuns=hf)
        r, se = growth_at(model_evaluator(mdl, env, use_cv=False), 0.0, 100000, 16, seed)
        ex = Ez(lambda z: np.log(np.cosh(sg * math.sqrt((1 - phi) / 2) * z)))
        rp.check("B7a.%d" % seed, abs(r - ex) <= NSE * se, "(7a) stays (2,2), transfers H and 1/H, sigma %.1f, memory %.1f: "
                 "growth per step %.6f (SE %.6f); E log cosh[sigma sqrt((1-phi)/2) Z] = %.6f" % (sg, phi, r, se, ex))
    # D_++ = 2 d_s
    for sg, phi, s, seed in ((1.0, 0.8, 0.5, 85), (1.5, -0.5, 0.8, 86)):
        tau = 1 / (1 - s)
        mdl2 = model_ring([tau, tau], {0: 0, 1: 0})
        ex = 2 * scalar_root(lambda z: np.exp(sg * z - sg * sg / 2), s)
        res = find_root(model_evaluator(mdl2, lognormal_env(sg, [phi])), ex, 100000, 32, seed)
        rp.check("B7.Dpp_sim%d" % seed, abs(res["D"] - ex) <= NSE * res["se"], "same-sign pair, equal stays %.0f, sigma %.1f, "
                 "memory %.1f: simulated D = %.5f (SE %.5f); 2 d_s by quadrature = %.5f%s"
                 % (tau, sg, phi, res["D"], res["se"], ex, " (note's table at this case: 0.52301 (0.0017))" if seed == 85 else ""))
    # the note's table: single transfer and opposite pair on stays (2,2)
    table = [("single", -0.8, 0.3, -0.00519904, 0.00023), ("single", -0.8, 1.0, 0.04949668, 0.0017),
             ("single", 0.8, 1.0, 0.32697814, 0.00045),
             ("opposite", 0.0, 0.3, 0.00266215, 0.00022), ("opposite", 0.0, 1.0, 0.19692712, 0.0017),
             ("opposite", -0.8, 0.3, -0.05966736, 0.00069), ("opposite", -0.8, 1.0, -0.15838507, 0.0042),
             ("opposite", 0.8, 0.3, 0.07204296, 3.3e-05), ("opposite", 0.8, 1.0, 0.80579042, 0.00034)]
    for i, (kind, phi, sg, theirs, se_t) in enumerate(table):
        if kind == "single":
            m, env = model_ring([2.0, 2.0], {0: 0}), lognormal_env(sg, [phi])
            k2 = Kg_closed([2.0, 2.0], phi)
        else:
            m, env = model_ring([2.0, 2.0], {0: 0, 1: 1}), lognormal_env(sg, [phi], lat=[0, 0], kappa=[1.0, -1.0])
            k2 = K6_closed([2.0, 2.0], [1.0, -1.0], phi)
        res = find_root(model_evaluator(m, env), sg * sg * k2 if k2 != 0 else 0.01, 100000, 32, 90 + i)
        rp.check("B.table.%s.phi%.1f.sigma%.1f" % (kind, phi, sg), cmp_note(res["D"], res["se"], theirs, se_t),
                 "%s transfer(s), stays (2,2), memory %.1f, sigma %.1f: my D = %+.5f (SE %.5f); note %+.5f (SE %.5f); "
                 "second-order value %+.5f" % (kind, phi, sg, res["D"], res["se"], theirs, se_t, sg * sg * k2))
    return rp.done()


def sec_SP2():
    rp = Rep("S-spot 2. Note B: parking under contact limitation at finite noise, and the repartition (18,1,1) against "
             "(10,10) (criterion against the note's simulated numbers: 4 combined SE)")
    X, pi = np.array([-0.5, 2.0]), np.array([0.8, 0.2])
    quoted = {0.3: (0.00300153, 5.9e-05, 0.00311170, 5.8e-05, 0.000110167, 1.1e-06),
              1.0: (0.05666975, 0.00029, 0.05857725, 0.00029, 0.001907503, 1.7e-05)}
    for sg, (q20, s20, q40, s40, qd, sd) in quoted.items():
        amp = math.sqrt(math.exp(sg * sg) - 1)
        env = make_markov_env(X, pi, 0.8, amp)
        out = {}
        for b in (20.0, 40.0):
            mdl = model_ring([5.0, b], {}, contact=(0, 0.1, 0))
            out[b] = find_root(model_evaluator(mdl, env), amp * amp * Kc_closed([5.0, b], 0, 0.1, 0.8), 200000, 32, 100 + int(10 * sg))
        dif = out[40.0]["roots"] - out[20.0]["roots"]
        dm, dse = float(dif.mean()), float(dif.std(ddof=1) / math.sqrt(len(dif)))
        ok = (cmp_note(out[20.0]["D"], out[20.0]["se"], q20, s20) and cmp_note(out[40.0]["D"], out[40.0]["se"], q40, s40)
              and cmp_note(dm, dse, qd, sd) and dm > NSE * dse)
        rp.check("B4.sim.sigma%.1f" % sg, ok,
                 "contact stays (5,b), d = 0.1, the note's bounded two-state host (values %.3f and %.3f, memory 0.8), sigma %.1f: "
                 "D(b=20) = %.6f (SE %.6f; note %.6f), D(b=40) = %.6f (SE %.6f; note %.6f); D40 - D20 in the same "
                 "environments = %+.7f (SE %.7f; note %+.7f (%.1e)); second-order values %.6f and %.6f"
                 % (1 - amp / 2, 1 + 2 * amp, sg, out[20.0]["D"], out[20.0]["se"], q20, out[40.0]["D"], out[40.0]["se"], q40,
                    dm, dse, qd, sd, amp * amp * Kc_closed([5.0, 20.0], 0, 0.1, 0.8), amp * amp * Kc_closed([5.0, 40.0], 0, 0.1, 0.8)))
    quoted = {0.3: (0.03350309, 0.00023, 0.02976608, 0.00079), 1.0: (0.38901993, 0.0023, 0.44930543, 0.0071)}
    for sg, (qa, sa, qb, sb) in quoted.items():
        ra = find_root(model_evaluator(model_ring([10.0, 10.0], {0: 0, 1: 1}), lognormal_env(sg, [0.9, 0.9])),
                       sg * sg * Kseries([10.0, 10.0], 0.9), 150000, 32, 120 + int(10 * sg))
        rb = find_root(model_evaluator(model_ring([18.0, 1.0, 1.0], {0: 0, 1: 1, 2: 2}), lognormal_env(sg, [0.9, 0.9, 0.9])),
                       sg * sg * Kseries([18.0, 1.0, 1.0], 0.9), 150000, 32, 130 + int(10 * sg))
        dm = rb["D"] - ra["D"]
        dse = math.sqrt(ra["se"] ** 2 + rb["se"] ** 2)
        rp.check("B15.sim.sigma%.1f" % sg, cmp_note(ra["D"], ra["se"], qa, sa) and cmp_note(rb["D"], rb["se"], qb, sb),
                 "memory 0.9, every transfer its own independent host, sigma %.1f: D(10,10) = %.5f (SE %.5f; note %.5f), "
                 "D(18,1,1) = %.5f (SE %.5f; note %.5f); three hosts minus two = %+.5f (SE %.5f): %s; second-order "
                 "values %.5f and %.5f" % (sg, ra["D"], ra["se"], qa, rb["D"], rb["se"], qb, dm, dse,
                                          "lower with three" if dm < -NSE * dse else ("higher with three" if dm > NSE * dse else "unresolved"),
                                          sg * sg * Kseries([10.0, 10.0], 0.9), sg * sg * Kseries([18.0, 1.0, 1.0], 0.9)))
    return rp.done()


# --------------------------------------------------------------------------------------------
# X: searches for a counterexample with products of random matrices, many cases at once
# --------------------------------------------------------------------------------------------
def stack_models(models, ys, nmax, Imax):
    C = len(models)
    Ac = np.zeros((C, nmax, nmax))
    Ad = np.zeros((C, Imax, nmax, nmax))
    for c, (m, y) in enumerate(zip(models, ys)):
        ey = math.exp(y)
        Ac[c, :m.n, :m.n] = m.C0 + ey * m.C1
        Ad[c, :m.I, :m.n, :m.n] = m.B0 + ey * m.B1
    return Ac, Ad


def batch_growth(Ac, Ad, hstep, T, burn, R):
    """Growth per step of C different models at once, R independent replicates each.
    Returns mean and standard error over the replicates, per model."""
    C, n = Ac.shape[0], Ac.shape[1]
    x = np.zeros((C, R, n, 1))
    x[:, :, 0, 0] = 1.0
    acc = np.zeros((C, R))
    for t in range(T + burn):
        h = hstep()
        A = Ac[:, None] + np.einsum("cri,cijk->crjk", h, Ad)
        x = A @ x
        sm = x.sum(axis=2, keepdims=True)
        x /= sm
        if t >= burn:
            acc += np.log(sm[:, :, 0, 0])
    r = acc / T
    return r.mean(1), r.std(1, ddof=1) / math.sqrt(R)


class LogNormalStep:
    """h_i = exp(sig_i z_i - sig_i^2/2), z_i = sum_k Lam_ik u_k (rows of Lam have unit length),
    u_k independent stationary AR(1) with coefficient phi_k."""

    def __init__(self, rng, sig, Lam, phi, R):
        self.rng, self.sig, self.Lam, self.phi = rng, sig[:, None, :], Lam, phi[:, None, :]
        self.sq = np.sqrt(1 - self.phi ** 2)
        self.u = rng.standard_normal((sig.shape[0], R, Lam.shape[2]))

    def __call__(self):
        self.u = self.phi * self.u + self.sq * self.rng.standard_normal(self.u.shape)
        z = np.einsum("cik,crk->cri", self.Lam, self.u)
        return np.exp(self.sig * z - 0.5 * self.sig ** 2)


class MarkovStep:
    """One multiplier per model: h = 1 + amp X, X in {X0, X1} with stationary law pi and
    transition phi I + (1 - phi) 1 pi."""

    def __init__(self, rng, amp, phi, R, X=(-0.5, 2.0), p1=0.2):
        self.rng, self.amp, self.phi, self.X, self.p1 = rng, amp[:, None], phi[:, None], X, p1
        self.st = rng.random((len(amp), R)) < p1

    def __call__(self):
        keep = self.rng.random(self.st.shape) < self.phi
        fresh = self.rng.random(self.st.shape) < self.p1
        self.st = np.where(keep, self.st, fresh)
        return (1 + self.amp * np.where(self.st, self.X[1], self.X[0]))[:, :, None]


def rand_loadings(rng, I, K):
    kind = int(rng.integers(0, 3))
    L = np.zeros((I, K))
    if kind == 0:
        L[np.arange(I), np.arange(I)] = 1.0
    elif kind == 1:
        L[:, 0] = np.where(rng.random(I) < 0.5, -1.0, 1.0)
    else:
        L = rng.standard_normal((I, K))
        L /= np.sqrt((L ** 2).sum(1, keepdims=True))
    return L


def lognormal_search(rng, models, ys, sig, Lam, phi, T, burn, R, batch=250):
    nmax = max(m.n for m in models)
    Imax = max(m.I for m in models)
    means, ses = [], []
    for a in range(0, len(models), batch):
        sl = slice(a, a + batch)
        Ac, Ad = stack_models(models[sl], ys[sl], nmax, Imax)
        step = LogNormalStep(rng, sig[sl], Lam[sl], phi[sl], R)
        mn, se = batch_growth(Ac, Ad, step, T, burn, R)
        means.append(mn)
        ses.append(se)
    return np.concatenate(means), np.concatenate(ses)


def summarise_search(rp, key, text, mean, se, want_sign, recheck=None, resolve=False):
    """want_sign = -1: the claim is growth <= 0; +1: growth >= 0.  A case counts against the
    claim when it is on the wrong side by more than 4 SE; such cases are rerun (ten times longer,
    16 replicates, new seed) by recheck(indices) -> (mean, se).  resolve=True also reruns the
    cases within 4 SE of zero and reports how many of them the longer run settles."""
    zs = want_sign * mean / np.maximum(se, 1e-300)
    bad = np.where(zs < -NSE)[0]
    mid = np.where(np.abs(zs) <= NSE)[0]
    resolved = int((zs > NSE).sum())
    nbad_after = len(bad)
    extra = ""
    if len(bad) and recheck is not None:
        m2, s2 = recheck(bad)
        z2 = want_sign * m2 / np.maximum(s2, 1e-300)
        nbad_after = int((z2 < -NSE).sum())
        extra += "; %d flagged in the first pass (z = %s); in a ten times longer rerun with a new seed their z = %s, so %d remain " \
                 "on the wrong side by 4 SE" % (len(bad), ", ".join("%.1f" % v for v in zs[bad]),
                                                 ", ".join("%.1f" % v for v in z2), nbad_after)
    if resolve and len(mid) and recheck is not None:
        m3, s3 = recheck(mid)
        z3 = want_sign * m3 / np.maximum(s3, 1e-300)
        nbad_after += int((z3 < -NSE).sum())
        extra += "; of the %d cases within 4 SE of zero, a ten times longer rerun puts %d on the claimed side by 4 SE, " \
                 "%d on the wrong side, %d still within (most adverse z there %.2f)" \
                 % (len(mid), int((z3 > NSE).sum()), int((z3 < -NSE).sum()), int((np.abs(z3) <= NSE).sum()), float(z3.min()))
    rp.check(key, nbad_after == 0, "%s: %d cases; first pass: on the claimed side by more than 4 SE in %d, within 4 SE of zero "
             "in %d, on the wrong side by more than 4 SE in %d (most adverse z = %.2f)%s"
             % (text, len(mean), resolved, len(mid), len(bad), float(zs.min()), extra))


def gen_lognormal_cases(rng, ncase, kinds, sig_lo, sig_hi, phi_lo, phi_hi, K=5):
    models, sig, Lam, phi = [], [], [], []
    for case in range(ncase):
        kind = kinds[case % len(kinds)]
        if kind == "ring":
            n = int(rng.integers(1, 6))
            taus = rand_taus(rng, n)
            if rng.random() < 0.2:
                taus = np.ones(n) + (rng.random(n) < 0.3) * rng.random(n)
            ks = [k for k in range(n) if rng.random() < 0.75] or [0]
            mdl = model_ring(taus, {k: i for i, k in enumerate(ks)})
        elif kind == "parallel":
            J = int(rng.integers(2, 5))
            mdl = model_parallel(float(rand_taus(rng, 1)[0]), rand_taus(rng, J), rng.dirichlet(np.ones(J)))
        elif kind == "optional":
            t0, t1, tp = rand_taus(rng, 3)
            mdl = model_optional(t0, t1, tp, rng.random(), rng.uniform(0.05, 1))
        else:
            t0, t1 = rand_taus(rng, 2)
            mdl = model_bypass(t0, t1, rng.uniform(0, 0.95), rng.uniform(0.05, 1))
        I = mdl.I
        s_ = np.zeros(5)
        s_[:I] = rng.uniform(sig_lo, sig_hi, I)
        L_ = np.zeros((5, K))
        L_[:I] = rand_loadings(rng, I, K)
        models.append(mdl)
        sig.append(s_)
        Lam.append(L_)
        phi.append(rng.uniform(phi_lo, phi_hi, K))
    return models, np.array(sig), np.array(Lam), np.array(phi)


def gen_contact_cases(rng, ncase, phi_lo, phi_hi, dzero=False):
    models, amp, phi, info = [], [], [], []
    for case in range(ncase):
        n = int(rng.integers(1, 5))
        taus = rand_taus(rng, n) + 0.05
        k = int(rng.integers(0, n))
        d = 0.0 if dzero else rng.random() ** (2 if case % 2 else 1)
        s_k = 1 - 1 / taus[k]
        c = (1 - d) * (1 - s_k)
        amax = min(2.0, s_k / (2 * c) if c > 0 else 2.0)        # h >= 0 and stay rate >= 0
        a = (1.0 if dzero else rng.uniform(0.3, 0.98)) * amax
        models.append(model_ring(taus, {}, contact=(k, d, 0)))
        amp.append(a)
        phi.append(rng.uniform(phi_lo, phi_hi))
        info.append((taus, k, d, s_k, c))
    return models, np.array(amp), np.array(phi), info


def sec_X1():
    rp = Rep("X1. Searches at large noise, environments independent in time: is the loss ever negative? "
             "(growth at the threshold of the mean matrix must be <= 0); two opposite lognormal transfers")
    rng = np.random.default_rng(31031)
    T, burn, R = 4000, 300, 16
    # rings and route networks, lognormal multipliers, correlated or signed across transfers
    kinds = ["ring", "ring", "parallel", "optional", "bypass", "ring", "parallel"]
    models, sig, Lam, phi = gen_lognormal_cases(rng, 1750, kinds, 0.3, 2.5, 0.0, 0.0)
    mean, se = lognormal_search(rng, models, np.zeros(len(models)), sig, Lam, phi, T, burn, R)

    def recheck(idx):
        return lognormal_search(np.random.default_rng(99001), [models[i] for i in idx], np.zeros(len(idx)), sig[idx], Lam[idx],
                                phi[idx], 10 * T, burn, 16)
    summarise_search(rp, "X.iid.lognormal", "independent in time, lognormal multipliers with sigma 0.3 to 2.5 on rings "
                     "(1 to 5 stages, several noisy transfers, independent, common with signs, or mixed) and on parallel, "
                     "optional and bypass networks: growth at the mean threshold <= 0", mean, se, -1, recheck)
    # contact rings, bounded two-state multiplier
    cm, amp, cphi, info = gen_contact_cases(rng, 500, 0.0, 0.0)
    Ac, Ad = stack_models(cm, np.zeros(len(cm)), 4, 1)
    mean, se = batch_growth(Ac, Ad, MarkovStep(rng, amp, cphi, R), T, burn, R)

    def recheck_c(idx):
        Ac2, Ad2 = stack_models([cm[i] for i in idx], np.zeros(len(idx)), 4, 1)
        return batch_growth(Ac2, Ad2, MarkovStep(np.random.default_rng(99002), amp[idx], cphi[idx], 16), 10 * T, burn, 16)
    summarise_search(rp, "X.iid.contact", "independent in time, contact-limited rings (1 to 4 stages, any d) with a "
                     "two-state multiplier at 30% to 98% of the largest valid amplitude: growth at the mean threshold <= 0",
                     mean, se, -1, recheck_c)
    # two opposite lognormal transfers
    om, osig, oLam = [], [], []
    for case in range(2000):
        n = int(rng.integers(2, 6))
        taus = rand_taus(rng, n)
        i, j = rng.choice(n, 2, replace=False)
        om.append(model_ring(taus, {int(i): 0, int(j): 1}))
        sg = rng.uniform(0.5, 2.5)
        osig.append([sg, sg])
        oLam.append([[1.0], [-1.0]])
    osig, oLam = np.array(osig), np.array(oLam)
    ophi = np.zeros((2000, 1))
    mean, se = lognormal_search(rng, om, np.zeros(2000), osig, oLam, ophi, T, burn, R)

    def recheck_o(idx):
        return lognormal_search(np.random.default_rng(99006), [om[i] for i in idx], np.zeros(len(idx)), osig[idx], oLam[idx],
                                ophi[idx], 10 * T, burn, 16)
    summarise_search(rp, "X.opposite", "two opposite lognormal transfers exp(+-sigma z - sigma^2/2), no memory, sigma 0.5 to "
                     "2.5, random rings of 2 to 5 stages: growth at the mean threshold < 0 (largest mean growth in the first "
                     "pass %.5f)" % float(mean.max()), mean, se, -1, recheck_o, resolve=True)
    return rp.done()


def sec_X2():
    rp = Rep("X2. Searches at large noise with memory: the bound (16)-(17) and its contact version; D_++ = 2 d_s and "
             "the conserved vector along random paths")
    rng = np.random.default_rng(32032)
    T, burn, R = 4000, 300, 16
    kinds = ["ring", "ring", "parallel", "optional", "bypass", "ring", "parallel"]
    models, sig, Lam, phi = gen_lognormal_cases(rng, 2100, kinds, 0.3, 2.5, -0.95, 0.98)
    yb = np.array([bound17(m, 0.5 * sig[c, :m.I] ** 2) for c, m in enumerate(models)])
    mean, se = lognormal_search(rng, models, yb, sig, Lam, phi, T, burn, R)

    def recheck(idx):
        return lognormal_search(np.random.default_rng(99003), [models[i] for i in idx], yb[idx], sig[idx], Lam[idx],
                                phi[idx], 10 * T, burn, 16)
    summarise_search(rp, "X.bound.lognormal", "the bound (16)-(17) with lognormal multipliers, sigma 0.3 to 2.5, memories "
                     "-0.95 to 0.98, correlated or signed across transfers, rings and the three route networks: growth with "
                     "the births multiplied by exp(bound) >= 0", mean, se, +1, recheck)
    # contact, two-state Markov multiplier with memory
    cm, amp, cphi, info = gen_contact_cases(rng, 700, 0.0, 0.98)
    X0, X1, p1 = -0.5, 2.0, 0.2
    ybc, ytr = [], []
    for (taus, k, d, s_k, c), a in zip(info, amp):
        hv = np.array([1 + a * X0, 1 + a * X1])
        pv = np.array([1 - p1, p1])
        with np.errstate(divide="ignore"):
            l_tr = -float(pv @ np.log(hv))
            l_st = -float(pv @ np.log((s_k - c * (hv - 1)) / s_k)) if s_k > 0 else 0.0
        ytr.append(l_tr)
        ybc.append(l_tr + (taus[k] - 1) * l_st)
    ybc, ytr = np.array(ybc), np.array(ytr)
    Ac, Ad = stack_models(cm, ybc, 4, 1)
    mean, se = batch_growth(Ac, Ad, MarkovStep(rng, amp, cphi, R), T, burn, R)

    def recheck_c(idx):
        Ac2, Ad2 = stack_models([cm[i] for i in idx], ybc[idx], 4, 1)
        return batch_growth(Ac2, Ad2, MarkovStep(np.random.default_rng(99004), amp[idx], cphi[idx], 16), 10 * T, burn, 16)
    summarise_search(rp, "X.bound.contact", "contact version D <= l_transfer + (tau_k - 1) l_stay, two-state multiplier with "
                     "memory 0 to 0.98 at 30% to 98% of the largest valid amplitude, rings of 1 to 4 stages: growth with the "
                     "births multiplied by exp(bound) >= 0", mean, se, +1, recheck_c)
    Ac, Ad = stack_models(cm, ytr, 4, 1)
    mean, se = batch_growth(Ac, Ad, MarkovStep(rng, amp, cphi, R), T, burn, R)
    zs = mean / se
    bad = np.where(zs < -2.0)[0]                          # for this extra, rerun everything below -2 SE
    nb = 0
    z2txt = "none"
    if len(bad):
        Ac2, Ad2 = stack_models([cm[i] for i in bad], ytr[bad], 4, 1)
        m2, s2 = batch_growth(Ac2, Ad2, MarkovStep(np.random.default_rng(99005), amp[bad], cphi[bad], 16), 10 * T, burn, 16)
        nb = int((m2 / s2 < -NSE).sum())
        z2txt = ", ".join("%.1f" % v for v in m2 / s2)
    rp.p("      extra (not claimed by the note): the transfer-only bound D <= l_transfer in the same 700 contact cases: "
         "growth at exp(l_transfer) is below zero by more than 2 SE in %d of the first pass (most adverse z = %.2f); "
         "their z in a ten times longer rerun: %s; below -4 SE after the rerun: %d"
         % (len(bad), float(zs.min()), z2txt, nb))
    # d = 0: the left vector of the mean matrix is conserved along every path
    cm0, amp0, phi0, info0 = gen_contact_cases(rng, 2000, 0.0, 0.98, dzero=True)
    worst = 0.0
    for a in range(0, 2000, 500):
        ms = cm0[a:a + 500]
        Ac, Ad = stack_models(ms, np.zeros(len(ms)), 4, 1)
        W = np.zeros((len(ms), 4))
        for c, m in enumerate(ms):
            W[c, :m.n] = perron(m.mean())[2]
        step = MarkovStep(rng, amp0[a:a + 500], phi0[a:a + 500], 4)
        x = rng.random((len(ms), 4, 4, 1))
        for c, m in enumerate(ms):
            x[c, :, m.n:, :] = 0.0
        v0 = np.einsum("ck,crk->cr", W, x[..., 0])
        for t in range(1500):
            A = Ac[:, None] + np.einsum("cri,cijk->crjk", step(), Ad)
            x = A @ x
            worst = max(worst, float(np.abs(np.einsum("ck,crk->cr", W, x[..., 0]) / v0 - 1).max()))
    rp.check("X.mu0.path", worst < 1e-9, "contact limitation with d = 0, two-state host with memory at the LARGEST valid "
             "amplitude (the staying rate reaches zero), 2,000 random rings x 4 paths x 1,500 steps: the weighted total "
             "w^T x never moves from its starting value by more than %.1e (relative), so the growth rate is exactly zero" % worst)
    # D_++ = 2 d_s along random lognormal AR(1) paths
    C, Tn = 2000, 3000
    s = rng.uniform(0.05, 0.95, C)
    sg = rng.uniform(0.3, 2.5, C)
    ph = rng.uniform(-0.95, 0.98, C)
    z = np.empty((Tn, C))
    z[0] = rng.standard_normal(C)
    for t in range(1, Tn):
        z[t] = ph * z[t - 1] + np.sqrt(1 - ph * ph) * rng.standard_normal(C)
    h = np.exp(sg * z - sg * sg / 2)
    lo, hi = np.full(C, -60.0), np.full(C, 60.0)
    for it in range(70):                               # bisection for the one-stage root on each path
        mid = 0.5 * (lo + hi)
        f = np.log(s + (1 - s) * np.exp(mid) * h).mean(0)
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    ds = 0.5 * (lo + hi)
    y = 2 * ds
    g = 1 - s
    worst_exact = worst_generic = 0.0
    for start in ("eigen", "generic"):
        x0 = np.ones(C)
        x1 = np.exp(-y / 2) if start == "eigen" else np.ones(C)
        acc = np.zeros(C)
        for t in range(Tn):
            n0 = s * x0 + g * np.exp(y) * h[t] * x1
            n1 = s * x1 + g * h[t] * x0
            tot = n0 + n1
            acc += np.log(tot / (x0 + x1))
            x0, x1 = n0 / tot, n1 / tot
        if start == "eigen":
            worst_exact = float(np.abs(acc / Tn).max())
        else:
            worst_generic = float((np.abs(acc) / (np.abs(y) + 5)).max())
    rp.check("X.Dpp.path", worst_exact < 1e-10 and worst_generic < 1.0,
             "D_++ = 2 d_s along 2,000 random lognormal AR(1) paths (sigma 0.3 to 2.5, memory -0.95 to 0.98, stay 0.05 to "
             "0.95, 3,000 steps): with the birth multiplied by exp(2 d_s), d_s the one-stage root ON THE SAME PATH, the "
             "two-stage product started on the tilted vector grows at rate %.1e per step (zero to rounding); started "
             "on (1,1) its total log growth over the whole path is at most %.2f x (|2 d_s| + 5), a boundary term"
             % (worst_exact, worst_generic))
    return rp.done()


# --------------------------------------------------------------------------------------------
# note C: stages x patches.  Demography first, then the mobile stage moves; census after movement.
# --------------------------------------------------------------------------------------------
def W_island(P):
    return np.full((P, P), 1.0 / P)


def W_circle(P):
    W = np.zeros((P, P))
    for p in range(P):
        W[(p + 1) % P, p] += 0.5
        W[(p - 1) % P, p] += 0.5
    return W


def W_line(P):
    W = np.zeros((P, P))
    for p in range(P):
        if p + 1 < P:
            W[p + 1, p] += 0.5
        if p - 1 >= 0:
            W[p - 1, p] += 0.5
    return W


def W_down_circle(P):
    W = np.zeros((P, P))
    for p in range(P):
        W[(p + 1) % P, p] += 1.0
    return W


def W_down_line(P):
    W = np.zeros((P, P))
    for p in range(P - 1):
        W[p + 1, p] += 1.0
    return W


TOPO = {"island": W_island, "circle": W_circle, "line": W_line, "downwind circle": W_down_circle,
        "downwind line": W_down_line}


def spec_radius_safe(M):
    """Spectral radius.  For a triangular matrix (the open downwind line) it is read off the
    diagonal: a general eigenvalue routine loses digits on a defective matrix."""
    if np.array_equal(M, np.tril(M)) or np.array_equal(M, np.triu(M)):
        return float(np.max(np.abs(np.diag(M))))
    return spec_radius(M)


class Spatial:
    """n stages in each of P patches.  Stage j stays with s_j = 1 - 1/stay_j; the transfer
    j -> j+1 in patch p has mean g[j, p].  After demography the mobile stage d moves with
    K_d = (1 - m) I + m (1 - ell) W (columns: patch of departure).  calibrate: transfer 0 is
    divided, in every patch, by the whole-system R0, so that R0 = 1."""

    def __init__(self, stays, P, d, m, ell, W, qual=None, calibrate=True):
        stays = np.asarray(stays, float)
        self.n, self.P, self.d = len(stays), P, d
        self.s = 1 - 1 / stays
        self.g = np.outer(1 / stays, np.ones(P))
        if qual is not None:
            self.g = self.g * qual
        self.Kd = (1 - m) * np.eye(P) + m * (1 - ell) * W
        self.R0_uncal = self.R0()
        if calibrate:
            self.g[0] = self.g[0] / self.R0_uncal

    def K(self, j):
        return self.Kd if j == self.d else np.eye(self.P)

    def R0(self):
        """Spectral radius of E_0 E_(n-1) ... E_1, E_j = (I - s_j K_j)^(-1) K_j G_(j-1)."""
        n, P = self.n, self.P
        Es = []
        for j in range(n):
            Kj = self.K(j)
            Es.append(np.linalg.solve(np.eye(P) - self.s[j] * Kj, Kj @ np.diag(self.g[(j - 1) % n])))
        M = Es[0]
        for j in range(n - 1, 0, -1):
            M = M @ Es[j]
        return spec_radius_safe(M)

    def move(self):
        n, P = self.n, self.P
        M = np.zeros((n * P, n * P))
        for j in range(n):
            M[j * P:(j + 1) * P, j * P:(j + 1) * P] = self.K(j)
        return M

    def transfer_full(self, k, p, M=None):
        """The full-matrix contribution of the transfer k -> k+1 in patch p (movement applied)."""
        n, P = self.n, self.P
        E0 = np.zeros((n * P, n * P))
        E0[((k + 1) % n) * P + p, k * P + p] = self.g[k, p]
        return (self.move() if M is None else M) @ E0

    def full(self):
        n, P = self.n, self.P
        M = self.move()
        Adem = np.zeros((n * P, n * P))
        for j in range(n):
            for p in range(P):
                Adem[j * P + p, j * P + p] += self.s[j]
                Adem[((j + 1) % n) * P + p, j * P + p] += self.g[j, p]
        return M @ Adem, M


def spatial_coef(sp, rho_s, sig, phis, birth_transfer=0):
    """General second-order coefficient of the loss on the FULL matrix (n P x n P).  sig: {k: s_k}
    log-sd multiplier of the host of transfer k; phis: {k: memory}.  Patches of one host have
    correlation rho_s; different hosts are independent.  The loss is the common log multiplier of
    one transfer in every patch."""
    A, M = sp.full()
    keys = [(k, p) for k in sorted(sig) for p in range(sp.P)]
    Es = [sig[k] * sp.transfer_full(k, p, M) for (k, p) in keys]
    I = len(keys)
    R = np.zeros((I, I))
    for a, (ka, pa) in enumerate(keys):
        for b_, (kb, pb) in enumerate(keys):
            if ka == kb:
                R[a, b_] = 1.0 if pa == pb else rho_s
    ph = [phis[k] for (k, p) in keys]
    C, lam, c, v, w = second_order(A, Es, ph, R)
    Bm = sum(sp.transfer_full(birth_transfer, p, M) for p in range(sp.P))
    b = (w @ Bm @ v) / lam
    return C / b, lam, b


def F_closed(stays, P, d, m, ell, rho_s, phi):
    """Note C: F = q S - (1 - q) H for identical island patches."""
    s = 1 - 1 / np.asarray(stays, float)
    a, b = 1 - m * ell, 1 - m
    u = s.copy()
    u[d] = a * s[d]
    t = 1 / (1 - u)
    T = t.sum()
    q = rho_s + (1 - rho_s) / P
    Ra = np.prod(phi * (1 - u) / (1 - phi * u))
    Rb = Ra * (b / a) * (1 - phi * a * s[d]) / (1 - phi * b * s[d])
    S = (1 + phi) / (2 * (1 - phi) * T) - Ra / (1 - Ra)
    H = Rb / (1 - Rb)
    return q * S - (1 - q) * H, S, H, T, Ra, Rb, q


def K_long_closed(taus_other, P, m, phi, rho_s, s_d):
    """Note C: F(L) = K/L + O(L^-2); s_d = staying rate of the mobile stage (1 if it is the long one)."""
    theta = phi / (1 - phi)
    U = np.prod(theta / (theta + np.asarray(taus_other, float)))
    f = (1 - m) * (1 - phi * s_d) / (1 - phi * (1 - m) * s_d)
    q = rho_s + (1 - rho_s) / P
    return q * (1 + 2 * theta) / 2 - theta * U * (q + (1 - q) * f)


def Ffull_island(stays, P, d, m, ell, rho_s, phi, qual=None):
    sp = Spatial(stays, P, d, m, ell, W_island(P), qual=qual)
    K, lam, b = spatial_coef(sp, rho_s, {0: 1.0}, {0: phi})
    assert abs(lam - 1) < 1e-8, lam
    return K


def sec_C1():
    rp = Rep("C (i). F = q S - (1 - q) H for identical island patches against the general coefficient on the FULL "
             "spatial matrix (patches x stages)")
    rng = np.random.default_rng(41041)
    worst = worstT = 0.0
    cat = {"before (stage 0)": 0, "after (stage 1)": 0, "a third, steady stage": 0}
    ncase = 240
    for case in range(ncase):
        P = 2 + case % 5
        n = 2 + (case // 5) % 3
        stays = rand_taus(rng, n)
        d = (case // 15) % n
        m = rng.uniform(0.02, 1.0)
        ell = 0.0 if rng.random() < 0.4 else rng.uniform(0, 0.9)
        rho_s = [0.0, 1.0, rng.random(), rng.random()][case % 4]
        phi = rng.uniform(-0.9, 0.95)
        sp = Spatial(stays, P, d, m, ell, W_island(P))
        Kf, lam, b = spatial_coef(sp, rho_s, {0: 1.0}, {0: phi})
        Fc, S, H, T, Ra, Rb, q = F_closed(stays, P, d, m, ell, rho_s, phi)
        scale = max(abs(Fc), abs(Kf), 1e-3 * (abs(q * S) + abs((1 - q) * H)))
        worst = max(worst, abs(Kf - Fc) / scale)
        worstT = max(worstT, abs(b * T - 1), abs(lam - 1))
        cat["before (stage 0)" if d == 0 else ("after (stage 1)" if d == 1 else "a third, steady stage")] += 1
    rp.check("C.F.i", worst < TOL_ALG and worstT < 1e-9,
             "F against the general coefficient on the full matrix, %d random cases (2 to 6 patches, 2 to 4 stages, stays 1 "
             "to 12, m 0.02 to 1, transit loss 0 to 0.9, spatial correlation 0, 1 or random, memory -0.9 to 0.95; mobile stage: %s): "
             "largest relative difference %.2e; the elasticity of the whole transfer is 1/T and the calibrated growth "
             "factor is one within %.1e" % (ncase, ", ".join("%s %d" % kv for kv in cat.items()), worst, worstT))
    # the noisy transfer need not be transfer 0; several independent hosts add
    worstk = worstsum = 0.0
    for case in range(80):
        P = int(rng.integers(2, 6))
        n = int(rng.integers(2, 5))
        stays = rand_taus(rng, n)
        d = int(rng.integers(0, n))
        m = rng.uniform(0.05, 1.0)
        ell = 0.0 if case % 2 else rng.uniform(0, 0.8)
        rho_s = rng.random()
        sp = Spatial(stays, P, d, m, ell, W_island(P))
        k = int(rng.integers(0, n))
        phi = rng.uniform(-0.9, 0.95)
        Kf = spatial_coef(sp, rho_s, {k: 1.0}, {k: phi}, birth_transfer=int(rng.integers(0, n)))[0]
        Fc, S, H, T, Ra, Rb, q = F_closed(stays, P, d, m, ell, rho_s, phi)
        sc = max(abs(Fc), 1e-3 * (abs(q * S) + abs((1 - q) * H)))
        worstk = max(worstk, abs(Kf - Fc) / sc)
        sig = {j: rng.uniform(0.3, 2.0) for j in range(n)}
        phs = {j: rng.uniform(-0.9, 0.95) for j in range(n)}
        Kall = spatial_coef(sp, rho_s, sig, phs)[0]
        Fs = [F_closed(stays, P, d, m, ell, rho_s, phs[j]) for j in range(n)]
        cl = sum(sig[j] ** 2 * Fs[j][0] for j in range(n))
        sc = max(abs(cl), 1e-3 * sum(sig[j] ** 2 * (abs(Fs[j][6] * Fs[j][1]) + abs((1 - Fs[j][6]) * Fs[j][2])) for j in range(n)))
        worstsum = max(worstsum, abs(Kall - cl) / sc)
    rp.check("C.F.any_transfer", worstk < TOL_ALG and worstsum < TOL_ALG,
             "80 random cases: the same F when the noisy transfer is any transfer k (and the loss is counted on any "
             "transfer): %.2e; every transfer with its own independent host of different variance and memory: "
             "coefficient = sum sigma_k^2 F(phi_k): %.2e" % (worstk, worstsum))
    # special cases and the amount removed
    w0 = w1 = wr = 0.0
    for case in range(60):
        P = int(rng.integers(2, 7))
        n = int(rng.integers(2, 5))
        stays = rand_taus(rng, n)
        d = int(rng.integers(0, n))
        m = rng.uniform(0.05, 1.0)
        rho_s = rng.random()
        phi = rng.uniform(0.05, 0.95)
        ell = 0.0 if case % 2 else rng.uniform(0, 0.8)
        q = rho_s + (1 - rho_s) / P
        F0 = Ffull_island(stays, P, d, m, ell, rho_s, 0.0)
        T = F_closed(stays, P, d, m, ell, rho_s, 0.0)[3]
        w0 = max(w0, relerr(F0, q / (2 * T)))
        F1 = Ffull_island(stays, P, d, 1.0, ell, rho_s, phi)
        S1 = F_closed(stays, P, d, 1.0, ell, rho_s, phi)[1]
        w1 = max(w1, abs(F1 - q * S1) / max(abs(q * S1), 1e-6))
        Fm = Ffull_island(stays, P, d, m, ell, rho_s, phi)
        Fc, S, H, T, Ra, Rb, q = F_closed(stays, P, d, m, ell, rho_s, phi)
        u = 1 - 1 / np.asarray(stays)
        u[d] *= (1 - m * ell)
        single = Kg_general(1 / (1 - u), 0, phi)          # one patch with the effective staying rates
        wr = max(wr, abs((single - Fm) - (1 - rho_s) * (1 - 1 / P) * (S + H)) / max(abs(S + H), 1e-6))
    rp.check("C.F.special", w0 < TOL_ALG and w1 < TOL_ALG and wr < TOL_ALG,
             "60 random cases, full matrix: without memory F = q/(2T) at any m > 0: %.2e; complete mixing (m = 1): "
             "F = q S: %.2e; amount removed relative to one patch with the effective staying rates = "
             "(1 - rho_s)(1 - 1/P)(S + H): %.2e" % (w0, w1, wr))
    return rp.done()


def sec_C2():
    rp = Rep("C (ii). The long-stay coefficient K and the quoted second-order boundaries, each from the general "
             "coefficient on the full matrix")
    rng = np.random.default_rng(42042)
    worst = worst_cf = 0.0
    ncase = 60
    for case in range(ncase):
        P = int(rng.integers(2, 6))
        n = int(rng.integers(2, 5))
        others = rand_taus(rng, n - 1)
        pos = int(rng.integers(0, n))                   # position of the long stage
        d = int(rng.integers(0, n))
        m = rng.uniform(0.05, 1.0)
        rho_s = rng.random() * (case % 3 > 0)
        phi = rng.uniform(0.05, 0.9)
        vals = [L * Ffull_island(np.insert(others, pos, L), P, d, m, 0.0, rho_s, phi) for L in (4000.0, 8000.0)]
        Kex = 2 * vals[1] - vals[0]
        vals = [L * F_closed(np.insert(others, pos, L), P, d, m, 0.0, rho_s, phi)[0] for L in (1e6, 2e6)]
        Kex_cf = 2 * vals[1] - vals[0]
        stays = np.insert(others, pos, 1.0)
        s_d = 1.0 if d == pos else 1 - 1 / stays[d]
        Kc = K_long_closed(others, P, m, phi, rho_s, s_d)
        q = rho_s + (1 - rho_s) / P
        sc = max(abs(Kc), 1e-2 * q * (0.5 + phi / (1 - phi)))
        worst = max(worst, abs(Kex - Kc) / sc)
        worst_cf = max(worst_cf, abs(Kex_cf - Kc) / sc)
    rp.check("C.K", worst < 1e-4 and worst_cf < 1e-8, "K against L F(L), %d random conservative cases (the long stage at "
             "any position, mobile or not, memory 0.05 to 0.9): from the full matrix at L = 4000 and 8000 (extrapolated; the "
             "remainder of my extrapolation is of order L^-2, about 1e-5): largest relative difference %.2e; from the "
             "closed F (which C (i) ties to the full matrix) at L = 1e6 and 2e6: %.2e" % (ncase, worst, worst_cf))
    # boundaries of the two-stage example: stays (2,2), P = 4, mobile stage one
    f_phi = lambda ph: Ffull_island([2.0, 2.0], 4, 1, 0.15, 0.0, 0.0, ph)
    pb = brentq(f_phi, 0.05, 0.9, xtol=1e-13)
    rp.check("C.phi_star", close_num(pb, 0.4644635334), "stays (2,2), P = 4, mobile stage one, m = 0.15, rho_s = 0: the "
             "full-matrix coefficient is zero at memory %.10f (note 0.4644635334); at 0.3 it is %+.6f (loss), at 0.6 %+.6f "
             "(gain); at the note's example memory 0.8 it is %+.6f" % (pb, f_phi(0.3), f_phi(0.6), f_phi(0.8)))
    f_m = lambda m: Ffull_island([2.0, 2.0], 4, 1, m, 0.0, 0.0, 0.8)
    mb = brentq(f_m, 0.2, 0.99, xtol=1e-13)
    Fc, S, H, T, Ra, Rb, q = F_closed([2.0, 2.0], 4, 1, 0.15, 0.0, 0.0, 0.8)
    xq = q * S / (1 - q + q * S)
    m_formula = 1 - xq / (Ra * (1 - 0.8 * 0.5) + xq * 0.8 * 0.5)
    rp.check("C.m_star", close_num(mb, 0.6803278689) and abs(m_formula - mb) < 1e-9,
             "same, memory 0.8: zero at m = %.10f (note 0.6803278689; the note's formula for m_* gives %.10f); "
             "at m = 0.5: %+.6f (gain), at m = 0.85: %+.6f (loss)" % (mb, m_formula, f_m(0.5), f_m(0.85)))
    f_r = lambda r: Ffull_island([2.0, 2.0], 4, 1, 0.15, 0.0, r, 0.8)
    rb = brentq(f_r, 0.0, 1.0, xtol=1e-13)
    r_formula = (4 * H / (S + H) - 1) / 3
    rp.check("C.rho_star", close_num(rb, 0.4890400605) and abs(r_formula - rb) < 1e-9,
             "same, memory 0.8, m = 0.15: zero at spatial correlation %.10f (note 0.4890400605; the note's formula for "
             "rho_* gives %.10f); at 0.3: %+.6f (gain), at 0.7: %+.6f (loss)" % (rb, r_formula, f_r(0.3), f_r(0.7)))
    # small m: is there a second crossing?  (the note warns that small m is singular at fixed noise)
    ms = [1e-6, 1e-4, 1e-2, 0.05]
    rp.p("      second-order coefficient at small m (memory 0.8): " + ", ".join("m = %g: %+.5f" % (m, f_m(m)) for m in ms)
         + "; at m -> 0 the closed form tends to q S - (1 - q) R_a/(1 - R_a) = %+.5f, not to the single-patch S = %+.5f"
         % (q * S - (1 - q) * Ra / (1 - Ra), S))
    # heterogeneous quality: which transfer carries the quality multipliers?
    qv = np.array([0.4, 0.8, 1.2, 1.6])
    readings = {"on the noisy transfer (0 -> 1)": np.array([qv, np.ones(4)]),
                "on the other transfer (1 -> 0)": np.array([np.ones(4), qv]),
                "on both transfers": np.array([qv, qv])}
    found = {}
    for lab, qual in readings.items():
        fq = lambda ph, qual=qual: Ffull_island([2.0, 2.0], 4, 1, 0.15, 0.0, 0.0, ph, qual=qual)
        grid = np.linspace(0.02, 0.995, 80)
        vals = [fq(ph) for ph in grid]
        roots = [brentq(fq, grid[i], grid[i + 1], xtol=1e-13) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
        found[lab] = roots
        rp.p("      quality (0.4,0.8,1.2,1.6) %s: zero(s) of the full-matrix coefficient at memory %s; value at 0.5: %+.6f, "
             "at 0.99: %+.6f" % (lab, ", ".join("%.10f" % r for r in roots) or "none in (0.02, 0.995)", fq(0.5), fq(0.99)))
    match = [lab for lab, roots in found.items() if any(close_num(r, 0.9732283441) for r in roots)]
    rp.check("C.hetero", len(match) > 0, "heterogeneous quality: the note's crossing 0.9732283441 is reproduced with the "
             "multipliers %s" % (" and ".join(match) if match else "under NONE of the three readings tried"))
    # H2: long-stay boundary and the finite comparison, stays (2,2,L), P = 4, m = 0.2, mobile stage one
    def Kfull(ph):
        v1 = 4000.0 * Ffull_island([2.0, 2.0, 4000.0], 4, 1, 0.2, 0.0, 0.0, ph)
        v2 = 8000.0 * Ffull_island([2.0, 2.0, 8000.0], 4, 1, 0.2, 0.0, 0.0, ph)
        return 2 * v2 - v1
    kb = brentq(Kfull, 0.5, 0.9, xtol=1e-12)
    kb_closed = brentq(lambda ph: K_long_closed([2.0, 2.0], 4, 0.2, ph, 0.0, 0.5), 0.5, 0.9, xtol=1e-14)
    FL = lambda L, ph: Ffull_island([2.0, 2.0, L], 4, 1, 0.2, 0.0, 0.0, ph)
    side_lo = FL(2000.0, kb - 0.03) - FL(1000.0, kb - 0.03)
    side_hi = FL(2000.0, kb + 0.03) - FL(1000.0, kb + 0.03)
    rp.check("C.K_boundary", close_num(kb_closed, 0.7563100600) and abs(kb - kb_closed) < 1e-6 and side_lo < 0 < side_hi,
             "stays (2,2,L), P = 4, m = 0.2, mobile stage one, rho_s = 0: K = 0 at memory %.10f from the closed K (note "
             "0.7563100600) and %.7f from the full matrix at L = 4000, 8000; F(2000) - F(1000) from the full matrix at "
             "memory - 0.03: %+.3e (longer stay lowers the loss), at + 0.03: %+.3e (raises it)" % (kb_closed, kb, side_lo, side_hi))
    f_fin = lambda ph: FL(40.0, ph) - FL(10.0, ph)
    fb = brentq(f_fin, 0.5, 0.9, xtol=1e-13)
    rp.check("C.L10_40", close_num(fb, 0.7127169883), "same: F(L=40) = F(L=10) at memory %.10f (note 0.7127169883); "
             "F(40) - F(10) at 0.65: %+.6f (helps), at 0.78: %+.6f (harms); F(10), F(40) at the crossing: %+.6f, %+.6f"
             % (fb, f_fin(0.65), f_fin(0.78), FL(10.0, fb), FL(40.0, fb)))
    # H3: two against four hosts at generation time 12, final stage mobile, every host noisy
    def Dn(n, ph):
        sp = Spatial([12.0 / n] * n, 4, n - 1, 0.2, 0.0, W_island(4))
        return spatial_coef(sp, 0.0, {k: 1.0 for k in range(n)}, {k: ph for k in range(n)})[0]
    f_h3 = lambda ph: Dn(4, ph) - Dn(2, ph)
    hb = brentq(f_h3, 0.5, 0.95, xtol=1e-13)
    F2 = F_closed([6.0, 6.0], 4, 1, 0.2, 0.0, 0.0, hb)[0]
    F4 = F_closed([3.0] * 4, 4, 3, 0.2, 0.0, 0.0, hb)[0]
    rp.check("C.H3", close_num(hb, 0.8550757674) and abs(2 * F2 - 4 * F4) < 1e-9,
             "generation time 12, P = 4, m = 0.2, rho_s = 0, final stage mobile, every transfer its own independent host: "
             "the full-matrix coefficients of four and of two hosts are equal at memory %.10f (note 0.8550757674; there "
             "2 F_2 = %.8f, 4 F_4 = %.8f); four minus two at 0.8: %+.6f (more hosts cost more), at 0.9: %+.6f (less); "
             "without memory: %+.6f" % (hb, 2 * F2, 4 * F4, f_h3(0.8), f_h3(0.9), f_h3(0.0)))
    return rp.done()


def sec_C3():
    rp = Rep("C (iv), (v). Exact invariance at full synchrony; the R0 ratios with transit loss and on the open downwind line")
    rng = np.random.default_rng(43043)
    # (v) R0 ratios
    w_isl = w_open = 0.0
    for case in range(300):
        P = int(rng.integers(2, 7))
        n = int(rng.integers(2, 5))
        stays = rand_taus(rng, n)
        d = int(rng.integers(0, n))
        m = rng.uniform(0.0, 1.0) if case % 10 else 1.0
        ell = rng.uniform(0, 0.95)
        s_d = 1 - 1 / stays[d]
        a = 1 - m * ell
        for topo in ("island", "circle", "downwind circle"):
            sp = Spatial(stays, P, d, m, ell, TOPO[topo](P), calibrate=False)
            w_isl = max(w_isl, abs(sp.R0_uncal - a * (1 - s_d) / (1 - a * s_d)))
        sp = Spatial(stays, P, d, m, ell, W_down_line(P), calibrate=False)
        w_open = max(w_open, abs(sp.R0_uncal - (1 - m) * (1 - s_d) / (1 - (1 - m) * s_d)))
    rp.check("C.R0.transit", w_isl < 1e-10, "identical patches, island, circle and downwind circle with transit loss: "
             "R0(m)/R0(0) = a(1 - s_d)/(1 - a s_d), a = 1 - m ell; 300 random cases x 3 topologies (R0 from the product of "
             "next-generation operators; R0(0) = 1): largest difference %.1e" % w_isl)
    rp.check("C.R0.open", w_open < 1e-10, "open downwind line: R0/R0(0) = (1 - m)(1 - s_d)/[1 - (1 - m) s_d], whatever the "
             "transit loss and the number of patches; same 300 cases (m = 1 in 30 of them, where R0 = 0): largest "
             "difference %.1e" % w_open)
    sp_l = Spatial([2.0, 3.0], 5, 1, 0.3, 0.0, W_line(5), calibrate=False)
    rp.p("      (open line with movement to both neighbours, stays (2,3), 5 patches, m = 0.3, no transit loss: R0 = %.6f; "
         "the downwind formula would give %.6f; the note gives no formula for this case)"
         % (sp_l.R0_uncal, 0.7 * (1 - 2 / 3) / (1 - 0.7 * 2 / 3)))
    # R0 > 1 exactly when the growth factor of the full mean matrix > 1; after calibration it is one
    wrong = 0
    worst_cal = 0.0
    for case in range(300):
        P = int(rng.integers(2, 6))
        n = int(rng.integers(2, 5))
        stays = rand_taus(rng, n)
        d = int(rng.integers(0, n))
        topo = list(TOPO)[case % 5]
        qual = np.exp(0.5 * rng.standard_normal((n, P)))
        m, ell = rng.uniform(0.02, 0.95), rng.uniform(0, 0.8) * (case % 2)
        sp = Spatial(stays, P, d, m, ell, TOPO[topo](P), qual=qual * np.exp(rng.uniform(-0.3, 0.3)), calibrate=False)
        lam = perron(sp.full()[0])[0]
        if abs(sp.R0_uncal - 1) > 1e-6 and (lam > 1) != (sp.R0_uncal > 1):
            wrong += 1
        sp2 = Spatial(stays, P, d, m, ell, TOPO[topo](P), qual=qual)
        worst_cal = max(worst_cal, abs(perron(sp2.full()[0])[0] - 1))
    rp.check("C.R0.def", wrong == 0 and worst_cal < 1e-9, "300 random cases with patch quality differing between patches "
             "(all five topologies, with and without transit loss): the growth factor of the full mean matrix exceeds one "
             "exactly when the note's R0 does (disagreements %d); after dividing transfer 0 by R0 the growth factor is one "
             "within %.1e" % (wrong, worst_cal))
    # (iv) full synchrony: stage totals follow the single ring path by path
    res = {}
    for topo in ("island", "circle", "downwind circle", "line", "downwind line"):
        for ell in (0.0, 0.4):
            worst = 0.0
            for case in range(20):
                P = int(rng.integers(2, 7))
                n = int(rng.integers(2, 5))
                stays = rand_taus(rng, n)
                d = int(rng.integers(0, n))
                m = rng.uniform(0.05, 1.0)
                sp = Spatial(stays, P, d, m, ell, TOPO[topo](P), calibrate=False)
                M = sp.move()
                a = 1 - m * ell
                x = rng.random(n * P)
                tot = x.reshape(n, P).sum(1)
                z = rng.standard_normal()
                for t in range(300):
                    z = 0.8 * z + 0.6 * rng.standard_normal()
                    h = math.exp(1.5 * z - 1.125)
                    Adem = np.zeros((n * P, n * P))
                    for j in range(n):
                        for p in range(P):
                            Adem[j * P + p, j * P + p] = sp.s[j]
                            Adem[((j + 1) % n) * P + p, j * P + p] += sp.g[j, p] * (h if j == 0 else 1.0)
                    x = M @ (Adem @ x)
                    new = sp.s * tot + np.roll(sp.g[:, 0] * tot * np.where(np.arange(n) == 0, h, 1.0), 1)
                    new[d] *= a
                    tot = new
                    worst = max(worst, float(np.abs(x.reshape(n, P).sum(1) / tot - 1).max()))
                    sc = tot.sum()
                    x /= sc
                    tot /= sc
            res[(topo, ell)] = worst
    ok_closed = all(res[(t, e)] < 1e-10 for t in ("island", "circle", "downwind circle") for e in (0.0, 0.4))
    rp.check("C.sync", ok_closed and res[("line", 0.0)] > 1e-3 and res[("downwind line", 0.0)] > 1e-3,
             "full synchrony (rho_s = 1), sigma 1.5, memory 0.8, 20 random systems x 300 steps each, patches started unequal: "
             "largest relative difference between the stage totals of the spatial system and the one-patch ring on the "
             "same path (mobile row multiplied by a = 1 - m ell): " + "; ".join("%s, transit loss %.1f: %.1e" % (t, e, v)
                                                                             for (t, e), v in res.items())
             + " (the open lines are not expected to agree)")
    return rp.done()


# --------------------------------------------------------------------------------------------
# note C: products of the actual random matrices
# --------------------------------------------------------------------------------------------
def spatial_growth(s, g, Kall, sig, rho_s, phi, ys, T, burn, R, rng, antithetic=False, common=False, dmob=None):
    """Growth per step of C spatial systems at once.  s (C,n); g (C,n,P); Kall (C,n,P,P) movement
    of each stage; sig (C,n) log-sd of the host of transfer k (0: steady); rho_s (C,); phi (C,n);
    ys (C,) log multiplier of the last transfer (n-1 -> 0) in every patch.  Each host has one
    Gaussian AR(1) series per patch, correlation rho_s between patches.
    common: all C systems see the same environments (noise parameters taken from system 0).
    antithetic: rows [R/2, R) use -z.  dmob: if given, only that stage moves, with Kall[0, dmob].
    Returns r (C, R) and the time mean of h - 1, shape (Cz, R, n, P)."""
    C, n, P = g.shape
    Cz = 1 if common else C
    Rz = R // 2 if antithetic else R
    sgz = sig[:Cz]
    act = np.where(np.abs(sgz).sum(0) > 0)[0]
    na = len(act)
    sg = sgz[:, None, act, None]
    ph = phi[:Cz][:, None, act, None]
    sq = np.sqrt(1 - ph ** 2)
    ac = np.sqrt(rho_s[:Cz])[:, None, None, None]
    ao = np.sqrt(1 - rho_s[:Cz])[:, None, None, None]
    z = ac * rng.standard_normal((Cz, Rz, na, 1)) + ao * rng.standard_normal((Cz, Rz, na, P))
    x = np.zeros((C, R, n, P))
    x[:, :, 0, :] = 1.0 / P                          # one cohort, spread evenly over the patches
    acc = np.zeros((C, R))
    hs = np.zeros((Cz, R, n, P))
    ey = np.exp(ys)[:, None, None]
    sx = s[:, None, :, None]
    gx = g[:, None]
    h = np.ones((Cz, R, n, P))
    KT = None if dmob is None else Kall[0, dmob].T
    Kx = Kall[:, None]
    for t in range(T + burn):
        z = ph * z + sq * (ac * rng.standard_normal((Cz, Rz, na, 1)) + ao * rng.standard_normal((Cz, Rz, na, P)))
        zz = np.concatenate([z, -z], axis=1) if antithetic else z
        h[:, :, act, :] = np.exp(sg * zz - 0.5 * sg * sg)
        tr = gx * h * x
        tr[:, :, n - 1, :] *= ey
        new = sx * x + np.roll(tr, 1, axis=2)
        if dmob is None:
            new = (Kx @ new[..., None])[..., 0]
        else:
            new[:, :, dmob, :] = new[:, :, dmob, :] @ KT
        sm = new.sum(axis=(2, 3))
        x = new / sm[:, :, None, None]
        if t >= burn:
            acc += np.log(sm)
            hs += h
    return acc / T, hs / T - 1.0


def spatial_evaluator(sp, sig, rho_s, phis, burn=2000, use_cv=True):
    """evalr(ys, T, npairs, seed) for one Spatial system: antithetic pairs, the same environments
    for every y, control variate = time mean of (h - 1) times the first-order coefficients."""
    n, P = sp.n, sp.P
    Kall = np.stack([sp.K(j) for j in range(n)])
    A, M = sp.full()
    lam, v, w = perron(A)
    beta = np.zeros((n, P))
    for k in range(n):
        if sig[k] != 0:
            for p in range(P):
                beta[k, p] = w @ sp.transfer_full(k, p, M) @ v / lam

    def evalr(ys, T, npairs, seed):
        C = len(ys)
        rng = np.random.default_rng(seed)
        tile = lambda a: np.broadcast_to(a, (C,) + a.shape).copy()
        r, hbar = spatial_growth(tile(sp.s), tile(sp.g), tile(Kall), tile(np.asarray(sig, float)), np.full(C, rho_s),
                                 tile(np.asarray(phis, float)), np.asarray(ys, float), T, burn, 2 * npairs, rng,
                                 antithetic=True, common=True, dmob=sp.d)
        if use_cv:
            r = r - np.einsum("rkp,kp->r", hbar[0], beta)[None, :]
        r = r.T
        return 0.5 * (r[:npairs] + r[npairs:])
    return evalr


def sec_C4():
    rp = Rep("C (iii). Two patches, complete mixing, no staying: D = sigma^2/2 - E log cosh[sigma sqrt((1 - rho_s)/2) Z] "
             "against products of random matrices (criterion: 4 SE)")
    for sg, rho_s, phi, n, seed in ((0.5, 0.0, 0.6, 2, 201), (1.5, 0.0, 0.6, 2, 202), (0.5, 0.5, 0.0, 2, 203),
                                    (1.5, 0.5, 0.9, 3, 204)):
        sp = Spatial([1.0] * n, 2, 1, 1.0, 0.0, W_island(2))
        sig = np.zeros(n)
        sig[0] = sg
        ev = spatial_evaluator(sp, sig, rho_s, [phi] * n)
        ex = sg * sg / 2 - Ez(lambda z: np.log(np.cosh(sg * math.sqrt((1 - rho_s) / 2) * z)))
        res = find_root(ev, ex, 100000, 32, seed)
        rp.check("C.cosh.%d" % seed, abs(res["D"] - ex) <= NSE * res["se"],
                 "%d stages without staying, 2 patches, complete island mixing of stage one, sigma %.1f, rho_s %.1f, memory "
                 "%.1f: simulated D = %.5f (SE %.5f); formula by quadrature %.5f; one patch alone would lose %.5f"
                 % (n, sg, rho_s, phi, res["D"], res["se"], ex, sg * sg / 2))
    # the formula is specific to no staying: with staying the removed amount is different
    sg = 1.5
    sp = Spatial([2.0, 2.0], 2, 1, 1.0, 0.0, W_island(2))
    res = find_root(spatial_evaluator(sp, np.array([sg, 0.0]), 0.0, [0.0, 0.0]), 0.3, 100000, 32, 205)
    one = scalar_root(lambda z: np.exp(sg * z - sg * sg / 2), 0.5)
    rp.p("      for contrast, stays (2,2), same mixing, sigma 1.5, no memory: D = %.5f (SE %.5f); the log cosh expression "
         "is for the ring without staying only (it would give %.5f)" % (res["D"], res["se"],
                                                                     sg * sg / 2 - Ez(lambda z: np.log(np.cosh(sg * math.sqrt(0.5) * z)))))
    return rp.done()


def neg_loss_case(rp, key, label, stays, P, d, m, rho_s, phi, sigmas, seed, T_main=200000, T_extra=100000, npairs=32,
                  main=(0.1, 0.3, 1.0), claim=True, small_sigma=0.1):
    """claim=True (the note's own example): 'the loss is negative' is a check at each main sigma.
    claim=False (cases chosen here): the sign at each sigma is reported as a finding; only the
    agreement with the second-order value at sigma = 0.1 is a check."""
    n = len(stays)
    sp = Spatial(stays, P, d, m, 0.0, W_island(P))
    Fc, S, H, Tg, Ra, Rb, q = F_closed(stays, P, d, m, 0.0, rho_s, phi)
    Ff = spatial_coef(sp, rho_s, {0: 1.0}, {0: phi})[0]
    rp.p("  %s: stays %s, %d patches, mobile stage %d, m = %.2f, rho_s = %.2f, memory %.2f, identical patches, island "
         "movement, no transit loss" % (label, stays, P, d, m, rho_s, phi))
    rp.p("      second-order coefficient: closed form F = %+.6f, full matrix %+.6f (S = %.5f, H = %.5f, q = %.4f, "
         "generation time %.1f)" % (Fc, Ff, S, H, q, Tg))
    rows = []
    for i, sg in enumerate(sigmas):
        sig = np.zeros(n)
        sig[0] = sg
        ev = spatial_evaluator(sp, sig, rho_s, [phi] * n)
        T = T_main if sg in main else T_extra
        res = find_root(ev, sg * sg * Fc, T, npairs, seed + i, extra_ys=(0.0,))
        r0, r0se = res["extra"][0]
        rows.append((sg, res["D"], res["se"], r0, r0se))
        verdict = "NEGATIVE loss (gain)" if res["D"] < -NSE * res["se"] else ("positive loss" if res["D"] > NSE * res["se"] else "unresolved")
        rp.p("      sigma = %-4.2f  D = %+.6f (SE %.6f)  [second order: %+.6f; D/sigma^2 = %+.4f]   growth at the mean "
             "threshold %+.3e (SE %.1e) per step   -> %s   (%d steps x %d antithetic pairs)"
             % (sg, res["D"], res["se"], sg * sg * Fc, res["D"] / sg ** 2, r0, r0se, verdict, T, npairs))
        if sg in main and claim:
            rp.check("%s.sigma%.1f" % (key, sg), res["D"] < -NSE * res["se"] and r0 > NSE * r0se,
                     "%s, sigma = %.1f: D = %+.6f (SE %.6f), negative by %.0f SE; growth at the mean threshold %+.3e "
                     "(SE %.1e), positive by %.0f SE" % (label, sg, res["D"], res["se"], -res["D"] / res["se"], r0, r0se, r0 / r0se))
        elif sg in main:
            rp.p("  FINDING  %s, sigma = %.1f: D = %+.6f (SE %.6f): %s (%.0f SE from zero); growth at the mean threshold "
                 "%+.3e (SE %.1e)" % (label, sg, res["D"], res["se"], verdict, abs(res["D"]) / res["se"], r0, r0se))
        if abs(sg - small_sigma) < 1e-12:
            rp.check("%s.small" % key, within_small(res["D"], res["se"], sg * sg * Fc),
                     "%s, sigma = %g against the second-order value (4 SE + 5%%): D = %.4e (SE %.1e), sigma^2 F = %.4e, "
                     "ratio %.4f" % (label, sg, res["D"], res["se"], sg * sg * Fc, res["D"] / (sg * sg * Fc)))
    # where does the gain end?
    cross = None
    for (s1, D1, e1, _, _), (s2, D2, e2, _, _) in zip(rows, rows[1:]):
        if D1 < 0 <= D2:
            cross = s1 + (s2 - s1) * (-D1) / (D2 - D1)
            rp.p("      the loss changes sign between sigma = %.2f and %.2f (linear interpolation: about %.2f)" % (s1, s2, cross))
    if cross is None:
        if all(r[1] < -NSE * r[2] for r in rows):
            rp.p("      the loss is negative by more than 4 SE at every sigma tried, up to %.1f: the gain does not disappear in this range" % rows[-1][0])
        else:
            rp.p("      no sign change located among the sigmas tried")
    best = min(rows, key=lambda r: r[1])
    rp.p("      most negative loss among the sigmas tried: D = %+.5f at sigma = %.2f" % (best[1], best[0]))
    return rows


SIG_SWEEP = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.7, 2.0, 3.0)


def spatial_model(sp):
    """The spatial system as a general Model: one multiplier per patch on transfer 0 -> 1, the last
    transfer (n-1 -> 0) in every patch is the birth.  A second, independent implementation of the
    same process, used to cross-check spatial_growth()."""
    n, P = sp.n, sp.P
    M = sp.move()
    N = n * P
    C0 = np.zeros((N, N))
    C1 = np.zeros((N, N))
    for j in range(n):
        for p in range(P):
            C0[j * P + p, j * P + p] += sp.s[j]
    C0 = M @ C0
    for k in range(1, n):
        for p in range(P):
            if k == n - 1:
                C1 += sp.transfer_full(k, p, M)
            else:
                C0 += sp.transfer_full(k, p, M)
    B0 = np.array([sp.transfer_full(0, p, M) for p in range(P)])
    return Model(C0, C1, B0, np.zeros_like(B0))


def sec_C5a():
    rp = Rep("C (vi) a. Negative loss under dispersal at FINITE noise: the note's example")
    rows = neg_loss_case(rp, "C.neg.note", "note's example", [2.0, 2.0], 4, 1, 0.15, 0.0, 0.8, SIG_SWEEP, 300)
    # the same process through the general matrix simulator (a second implementation of mine)
    sp = Spatial([2.0, 2.0], 4, 1, 0.15, 0.0, W_island(4))
    mdl = spatial_model(sp)
    assert np.abs(mdl.mean() - sp.full()[0]).max() < 1e-14
    for sg in (0.3, 1.0):
        res = find_root(model_evaluator(mdl, lognormal_env(sg, [0.8] * 4)), -0.3 * sg * sg, 100000, 32, 390 + int(10 * sg))
        ref = [r for r in rows if abs(r[0] - sg) < 1e-12][0]
        rp.check("C.neg.cross.sigma%.1f" % sg, abs(res["D"] - ref[1]) <= NSE * math.sqrt(res["se"] ** 2 + ref[2] ** 2),
                 "note's example at sigma = %.1f through the general 8 x 8 matrix simulator with four independent drivers: "
                 "D = %+.6f (SE %.6f); the structured spatial simulator gave %+.6f (SE %.6f)" % (sg, res["D"], res["se"], ref[1], ref[2]))
    # exact periodic environments, no random number: four patches switching in turn
    rp.p("      exact periodic environments (full-period product of the 8 x 8 matrices, no simulation): each patch's host is "
         "1 + a for half the period and 1 - a for the other half; patches shifted by a quarter period (asynchronous) or not "
         "at all (synchronous)")
    ok_sync = True
    worst_small = 0.0
    A_full = mdl.mean()
    for period in (8, 20, 40):
        sq = np.array([1.0] * (period // 2) + [-1.0] * (period // 2))
        xi = np.stack([np.roll(sq, p * period // 4) for p in range(4)], axis=1)
        # second-order value with the periodic covariance, and the exact loss at small amplitude
        Cg, lam, c, v, w = second_order(A_full, [mdl.E(p) for p in range(4)],
                                        rho=lambda i, j, l, xi=xi: float(np.mean(xi[:, i] * np.roll(xi[:, j], l))))
        Kpred = Cg / (w @ mdl.birth() @ v)
        vals = [floquet_root(mdl, 1 + a * xi, -0.5, 0.5) / a ** 2 for a in (0.02, 0.01)]
        Kex = (4 * vals[1] - vals[0]) / 3
        worst_small = max(worst_small, abs(Kex - Kpred) / abs(Kpred))
        rp.p("        period %2d, asynchronous: second-order coefficient from the general formula on the full matrix %+.6f; "
             "exact loss/a^2 at a = 0.02, 0.01 (extrapolated) %+.6f" % (period, Kpred, Kex))
        for a in (0.3, 0.6, 0.9):
            Da = floquet_root(mdl, 1 + a * xi)
            Ds = floquet_root(mdl, 1 + a * np.stack([sq] * 4, axis=1))
            D1 = floquet_root(model_ring([2.0, 2.0], {0: 0}), (1 + a * sq)[:, None])
            ok_sync = ok_sync and abs(Ds - D1) < 1e-10
            rp.p("  FINDING   period %2d, a = %.1f: asynchronous D = %+.6f (%s); synchronous D = %+.6f (one patch alone: %+.6f)"
                 % (period, a, Da, "a gain, exact" if Da < 0 else "a loss, smaller than one patch's", Ds, D1))
    rp.check("C.periodic.small", worst_small < TOL_FLOQ, "the general second-order coefficient on the full spatial matrix, with "
             "the covariance between patches of the periodic environment, against the exact loss at small amplitude, periods "
             "8, 20, 40: largest relative difference %.2e" % worst_small)
    rp.check("C.periodic.sync", ok_sync, "with the patches in phase the exact loss of the spatial system equals that of one "
             "patch (nine environments, within 1e-10)")
    return rp.done()


def sec_C5b():
    rp = Rep("C (vi) b. Negative loss under dispersal at FINITE noise: a case of my own (three stages, six patches, the "
             "stage before the noisy transfer moves, patches partly synchronised)")
    neg_loss_case(rp, "C.neg.own1", "own case 1", [3.0, 1.5, 4.0], 6, 0, 0.10, 0.2, 0.9, SIG_SWEEP, 320, claim=False)
    return rp.done()


def sec_C5c():
    rp = Rep("C (vi) c. Negative loss under dispersal at FINITE noise: a second case of my own (unequal stays, three "
             "patches, weaker memory, more movement)")
    neg_loss_case(rp, "C.neg.own2", "own case 2", [1.5, 5.0], 3, 1, 0.30, 0.0, 0.7, SIG_SWEEP, 340, claim=False)
    return rp.done()


def sec_C5d():
    rp = Rep("C (vi) d. The note's example with a hundredth of the movement (m = 0.01), where the second-order gain is "
             "largest and the note says the expansion is least trustworthy")
    rows = neg_loss_case(rp, "C.neg.own3", "own case 3 (small m)", [2.0, 2.0], 4, 1, 0.01, 0.0, 0.8,
                         (0.02, 0.05, 0.1, 0.3, 0.5, 1.0), 360, claim=False, main=(0.1, 0.3, 1.0), small_sigma=0.02)
    F = F_closed([2.0, 2.0], 4, 1, 0.01, 0.0, 0.0, 0.8)[0]
    rp.p("  FINDING  with m = 0.01 the ratio of the simulated loss to sigma^2 F is "
         + ", ".join("%.3f at sigma = %g" % (r[1] / (r[0] ** 2 * F), r[0]) for r in rows)
         + ": the second-order value is the small-noise limit, but it is within 5 per cent only when sigma^2 is small against m")
    return rp.done()


def sec_C5e():
    rp = Rep("C (vi) e. The note's example read the other way: 'mobile stage one' taken as the FIRST stage (stage 0, the "
             "source of the noisy transfer) instead of stage 1.  The second-order coefficient is the same; is the "
             "finite-noise loss?")
    neg_loss_case(rp, "C.neg.note_d0", "note's example with stage 0 mobile", [2.0, 2.0], 4, 0, 0.15, 0.0, 0.8,
                  (0.1, 0.3, 1.0), 380, claim=True)
    return rp.done()


def sec_C6():
    rp = Rep("C (vii). The bound D <= sum sigma_k^2/2 at sigma = 1 in random spatial systems; and no gain when the "
             "environment is independent in time")
    rng = np.random.default_rng(46046)
    groups = [(2, 2), (2, 3), (2, 4), (2, 6), (3, 2), (3, 3), (3, 5), (4, 2), (4, 4), (4, 6)]
    per = 56
    T, burn, R = 3000, 300, 16
    tot = dict(n=0, bad=0, mid=0, good=0, zmin=1e9, flagged=0)
    tot_iid = dict(n=0, bad=0, mid=0, good=0, zmin=1e9, flagged=0)
    topo_count = {k: 0 for k in TOPO}
    for (n, P) in groups:
        S_, G_, K_, SIG, RHO, PHI, YB = [], [], [], [], [], [], []
        for case in range(per):
            topo = list(TOPO)[int(rng.integers(0, 5))]
            topo_count[topo] += 1
            stays = rand_taus(rng, n)
            if rng.random() < 0.2:
                stays = np.ones(n) + (rng.random(n) < 0.3) * rng.random(n)
            d = int(rng.integers(0, n))
            m = rng.uniform(0.02, 0.95)
            ell = 0.0 if rng.random() < 0.5 else rng.uniform(0, 0.8)
            qual = np.exp(0.5 * rng.standard_normal((n, P))) if rng.random() < 0.5 else None
            sp = Spatial(stays, P, d, m, ell, TOPO[topo](P), qual=qual)
            noisy = [k for k in range(n) if rng.random() < 0.6] or [0]
            sig = np.zeros(n)
            sig[noisy] = 1.0
            S_.append(sp.s)
            G_.append(sp.g)
            K_.append(np.stack([sp.K(j) for j in range(n)]))
            SIG.append(sig)
            RHO.append([0.0, 1.0, rng.random(), rng.random()][case % 4])
            PHI.append(rng.uniform(-0.9, 0.98, n))
            YB.append(0.5 * len(noisy))
        S_, G_, K_, SIG, RHO, PHI, YB = map(np.array, (S_, G_, K_, SIG, RHO, PHI, YB))
        for which, tt in (("bound", tot), ("iid", tot_iid)):
            ys = YB if which == "bound" else np.zeros(per)
            ph = PHI if which == "bound" else np.zeros_like(PHI)
            sign = 1.0 if which == "bound" else -1.0
            r, _ = spatial_growth(S_, G_, K_, SIG, RHO, ph, ys, T, burn, R, rng)
            mean, se = r.mean(1), r.std(1, ddof=1) / math.sqrt(R)
            zs = sign * mean / se
            bad = np.where(zs < -NSE)[0]
            tt["n"] += per
            tt["good"] += int((zs > NSE).sum())
            tt["mid"] += int((np.abs(zs) <= NSE).sum())
            tt["zmin"] = min(tt["zmin"], float(zs.min()))
            tt["flagged"] += len(bad)
            if len(bad):
                r2, _ = spatial_growth(S_[bad], G_[bad], K_[bad], SIG[bad], RHO[bad], ph[bad], ys[bad], 10 * T, burn, R,
                                       np.random.default_rng(99100 + n * 10 + P))
                z2 = sign * r2.mean(1) / (r2.std(1, ddof=1) / math.sqrt(R))
                tt["bad"] += int((z2 < -NSE).sum())
    rp.check("C.bound", tot["bad"] == 0, "D <= (number of noisy hosts)/2 at sigma = 1: %d random spatial systems (2 to 6 "
             "patches, 2 to 4 stages; %s; patch quality equal or unequal; with and without transit loss; 1 to n noisy "
             "hosts with memories -0.9 to 0.98; spatial correlation 0, 1 or random; mean matrix calibrated to R0 = 1): "
             "growth with the transfer multiplied by exp(bound) is positive by more than 4 SE in %d, within 4 SE of zero "
             "in %d, negative by more than 4 SE in %d of the first pass (most adverse z = %.2f) and in %d after a ten "
             "times longer rerun" % (tot["n"], ", ".join("%d %s" % (v, k) for k, v in topo_count.items()), tot["good"],
                                     tot["mid"], tot["flagged"], tot["zmin"], tot["bad"]))
    rp.check("C.iid", tot_iid["bad"] == 0, "the same %d systems with environments independent in time: growth at the mean "
             "threshold is negative by more than 4 SE in %d, within 4 SE of zero in %d, positive by more than 4 SE in %d "
             "of the first pass (most adverse z = %.2f) and in %d after rerun (the note's sufficient condition for a "
             "non-negative loss)" % (tot_iid["n"], tot_iid["good"], tot_iid["mid"], tot_iid["flagged"], tot_iid["zmin"], tot_iid["bad"]))
    return rp.done()


SECTIONS = {}
SECTIONS["H"] = sec_H
SECTIONS["HS"] = sec_HS
SECTIONS["C1"] = sec_C1
SECTIONS["C2"] = sec_C2
SECTIONS["C3"] = sec_C3
SECTIONS["C4"] = sec_C4
SECTIONS["C5a"] = sec_C5a
SECTIONS["C5b"] = sec_C5b
SECTIONS["C5c"] = sec_C5c
SECTIONS["C5d"] = sec_C5d
SECTIONS["C5e"] = sec_C5e
SECTIONS["C6"] = sec_C6
SECTIONS["SP1"] = sec_SP1
SECTIONS["SP2"] = sec_SP2
SECTIONS["X1"] = sec_X1
SECTIONS["X2"] = sec_X2
SECTIONS["SA"] = sec_SA
SECTIONS["SB"] = sec_SB
SECTIONS["B234"] = sec_B234
SECTIONS["B5"] = sec_B5
SECTIONS["B67"] = sec_B67
SECTIONS["B811"] = sec_B811
SECTIONS["B1214"] = sec_B1214
SECTIONS["B15"] = sec_B15
SECTIONS["B1617"] = sec_B1617

ORDER = ["H", "HS", "B234", "B5", "B67", "B811", "B1214", "B15", "B1617", "SA", "SB", "SP1", "SP2", "X1", "X2",
         "C1", "C2", "C3", "C4", "C5a", "C5b", "C5c", "C5d", "C5e", "C6"]
POOL_SECTIONS = ["C5b", "C5c", "X1", "X2", "SP2", "C5d", "SB", "C5e", "C4"]                # two worker processes, longest first
MAIN_SECTIONS = ["H", "B234", "B5", "B67", "B811", "B1214", "B15", "B1617", "C1", "C2", "C3", "HS", "C5a", "SA", "SP1", "C6"]


def run_section(name):
    t0 = time.time()
    txt, checks = SECTIONS[name]()
    sys.stderr.write("[%s done in %.0f s]\n" % (name, time.time() - t0))
    sys.stderr.flush()
    return name, txt, checks


if __name__ == "__main__":
    if "--list" in sys.argv:
        print(" ".join(ORDER))
        sys.exit(0)
    t_start = time.time()
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    results = {}
    if only is not None:
        for name in ORDER:
            if name in only:
                results[name] = run_section(name)
    else:
        from multiprocessing import Pool
        with Pool(2) as pool:                                   # main process + 2 workers = 3 processes
            pending = [pool.apply_async(run_section, (name,)) for name in POOL_SECTIONS]
            for name in MAIN_SECTIONS:
                results[name] = run_section(name)
            for pr in pending:
                r = pr.get()
                results[r[0]] = r
    print("verify_mechanisms_routes_dispersal.py   started %s   python %s, numpy %s"
          % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)), sys.version.split()[0], np.__version__))
    print("Tolerances (fixed before the first run): closed form against general coefficient %.0e relative (of the larger "
          "value, or of 1e-3 of the sum of the magnitudes of the terms where the terms nearly cancel; absolute floor %.0e); "
          "quoted numbers %.0e relative (see the header for the two exceptions); exact periodic loss at small amplitude "
          "%.0e relative; small-noise products: %g SE + %g%%; searches: wrong side by more than %g SE after a tenfold rerun."
          % (TOL_ALG, ABS_FLOOR, TOL_NUM, TOL_FLOQ, NSE, 100 * SMALL_REL, NSE))
    allchecks = []
    for name in ORDER:
        if name in results:
            print(results[name][1])
            allchecks += [(name,) + c for c in results[name][2]]
    npass = sum(1 for c in allchecks if c[2])
    print("")
    print("=" * 100)
    print("SUMMARY: %d checks, %d pass, %d fail.   Wall time %.0f s." % (len(allchecks), npass, len(allchecks) - npass, time.time() - t_start))
    for c in allchecks:
        if not c[2]:
            print("  FAILED  [%s] %s: %s" % (c[0], c[1], c[3]))
    print("=" * 100)
    sys.exit(0 if npass == len(allchecks) else 1)
