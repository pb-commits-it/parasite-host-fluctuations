"""Helper library for run R15a (protocols/R15a-temperature-incubation.md).

Self-contained: models, exact computations, the second-order formula coded twice, and the simulators.
Nothing here imports the project's engine.  Every thermal parameter is ILLUSTRATIVE (protocol, section 3).

Conventions.  A model gives, for an array of temperatures, two stacks of matrices: `rest` and `scal`.
The matrix at log multiplier y is rest + exp(y) * scal; `scal` holds one sheltered transfer, which every
circuit of the life cycle uses exactly once, so R0 is proportional to exp(y).  For any environment the
value y* at which the long-run growth rate is zero gives ln R0* = -y* (for the model as written, y = 0).
With y_mean and y_tbar the same roots for the matrix of mean rates and the matrix at the mean temperature:
    curvature part  J = y_tbar - y_mean,   loss  D = y* - y_mean,   total = y_tbar - y* = J - D.
"""
from __future__ import annotations

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter

ZG = np.linspace(-10.0, 10.0, 20001)          # quadrature grid for a standard Gaussian
WG = np.exp(-ZG * ZG / 2)
WG /= WG.sum()

LN2, LN25, LN3 = math.log(2.0), math.log(2.5), math.log(3.0)


# ----------------------------------------------------------------------------- thermal responses
def briere(T, c, lo, hi):
    """c T (T - lo) sqrt(hi - T) between lo and hi, zero outside."""
    T = np.asarray(T, float)
    ok = (T > lo) & (T < hi)
    Tc = np.where(ok, T, 0.5 * (lo + hi))
    return np.where(ok, c * Tc * (Tc - lo) * np.sqrt(hi - Tc), 0.0)


def q10rate(T, m0, Q, Tref=20.0):
    return m0 * np.exp(math.log(Q) * (np.asarray(T, float) - Tref) / 10.0)


class TPC2:
    """Stage 2 of model TPC: development (Briere, 10 to 35 C) and death (Q10 = 2).  Illustrative."""
    def __call__(self, T):
        return briere(T, 7.0e-5, 10.0, 35.0), q10rate(T, 0.03, 2.0)


class TPC3:
    """Stage 3 of model TPC: contact (Briere, 8 to 40 C) and death (Q10 = 2.5).  Illustrative."""
    def __call__(self, T):
        return briere(T, 1.2e-4, 8.0, 40.0), q10rate(T, 0.05, 2.5)


class ExpStage:
    """nu = nu0 exp(bnu (T - Tref)), mu = mu0 exp(bmu (T - Tref)).  Illustrative."""
    def __init__(self, nu0, mu0, bnu, bmu, Tref=20.0):
        self.p = (nu0, mu0, bnu, bmu, Tref)

    def __call__(self, T):
        nu0, mu0, bnu, bmu, Tref = self.p
        d = np.asarray(T, float) - Tref
        return nu0 * np.exp(bnu * d), mu0 * np.exp(bmu * d)


def qp(nu, mu):
    """Clock q (chance of leaving in one day) and completion probability p, rates constant within the day."""
    r = nu + mu
    return -np.expm1(-r), nu / r


# ----------------------------------------------------------------------------- models
class Ring:
    """stages: list of ("fixed", s, g) or ("exposed", fn) with fn(T) -> (nu, mu).  The transfer out of
    stage `k0` (a fixed one unless there is none) carries exp(y)."""

    def __init__(self, stages, k0=0):
        self.stages = list(stages)
        self.n = len(self.stages)
        self.k0 = k0
        self.exposed = [k for k, st in enumerate(self.stages) if st[0] == "exposed"]

    def sg(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        s = np.empty(T.shape + (self.n,))
        g = np.empty(T.shape + (self.n,))
        for k, st in enumerate(self.stages):
            if st[0] == "fixed":
                s[..., k] = st[1]
                g[..., k] = st[2]
            else:
                q, p = qp(*st[1](T))
                s[..., k] = 1 - q
                g[..., k] = p * q
        return s, g

    def parts(self, T):
        s, g = self.sg(T)
        K, n = s.shape[0], self.n
        rest = np.zeros((K, n, n))
        scal = np.zeros((K, n, n))
        for k in range(n):
            rest[:, k, k] += s[:, k]
            (scal if k == self.k0 else rest)[:, (k + 1) % n, k] += g[:, k]
        return rest, scal


def dd_weights(nu, mu, m):
    """One-day map of a chain of m sub-stages, each left at rate m*nu, death at rate mu, rates constant in
    the day.  Returns W[..., k] = chance of surviving the day and advancing exactly k sub-stages (k < m),
    and E[..., k-1] = chance of leaving the chain alive within the day when k advances are needed."""
    nu = np.asarray(nu, float)
    mu = np.asarray(mu, float)
    a = m * nu
    lam = a + mu
    W = np.empty(nu.shape + (m,))
    P = np.empty(nu.shape + (m,))
    W[..., 0] = np.exp(-lam)
    P[..., 0] = W[..., 0]
    for k in range(1, m):
        W[..., k] = W[..., k - 1] * a / k
        P[..., k] = P[..., k - 1] * lam / k
    tail = np.clip(1.0 - np.cumsum(P, axis=-1), 0.0, 1.0)      # P(Poisson(lam) >= k), k = 1..m
    ratio = a / lam
    E = tail * np.cumprod(np.repeat(ratio[..., None], m, axis=-1), axis=-1)
    return W, E


class DDModel:
    """Model DD: stage 0 sheltered (stay 1/(1 - s1)), then an incubation chain of m sub-stages."""

    def __init__(self, m, s1=1 - 1 / 30.0, fn=None):
        self.m, self.s1, self.fn = m, s1, fn or TPC2()
        self.n = m + 1

    def parts(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        m, K = self.m, T.size
        W, E = dd_weights(*self.fn(T), m)
        rest = np.zeros((K, m + 1, m + 1))
        scal = np.zeros((K, m + 1, m + 1))
        rest[:, 0, 0] = self.s1
        for j in range(m):
            rest[:, 0, 1 + j] = E[:, m - 1 - j]
            for i in range(j, m):
                rest[:, 1 + i, 1 + j] = W[:, i - j]
        scal[:, 1, 0] = 1 - self.s1
        return rest, scal


def make_model(spec):
    """Models by name, so that tasks can be sent to worker processes."""
    kind = spec[0]
    if kind == "TPC":                       # ("TPC", variant, extra)
        variant = spec[1]
        s1 = 1 - 1 / 30.0
        st1 = ("fixed", s1, 1 - s1)
        if variant == "full":
            return Ring([st1, ("exposed", TPC2()), ("exposed", TPC3())])
        if variant == "split1":             # the sheltered stay divided between two sheltered stages
            h = 1 - 1 / 15.0
            return Ring([("fixed", h, 1 - h), ("fixed", h, 1 - h), ("exposed", TPC2()), ("exposed", TPC3())])
        sbar, gbar = spec[2]                # mean rates of the three stages for this environment
        if variant == "shelter2":
            return Ring([st1, ("fixed", sbar[1], gbar[1]), ("exposed", TPC3())])
        if variant == "shelter3":
            return Ring([st1, ("exposed", TPC2()), ("fixed", sbar[2], gbar[2])])
        raise ValueError(variant)
    if kind == "EXP":                       # ("EXP", bnu, bmu) two stages; ("EXP1", bnu, bmu) the exposed stage alone
        s1 = 1 - 1 / 20.0
        return Ring([("fixed", s1, 1 - s1), ("exposed", ExpStage(0.10, 0.05, spec[1], spec[2]))])
    if kind == "EXP1":
        return Ring([("exposed", ExpStage(0.10, 0.05, spec[1], spec[2]))])
    if kind == "EXPs":                      # as EXP with a sheltered stay of 5 days (amendment 3)
        s1 = 1 - 1 / 5.0
        return Ring([("fixed", s1, 1 - s1), ("exposed", ExpStage(0.10, 0.05, spec[1], spec[2]))])
    if kind == "TPCinc":                    # sheltered stage and stage 2 of model TPC only (model DD with m = 1)
        s1 = 1 - 1 / 30.0
        return Ring([("fixed", s1, 1 - s1), ("exposed", TPC2())])
    if kind == "DD":
        return DDModel(spec[1])
    raise ValueError(kind)


# ----------------------------------------------------------------------------- exact quantities of constant matrices
def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def perron(A):
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(vals.real))
    v = np.abs(vecs[:, i].real)
    vl, wl = np.linalg.eig(A.T)
    j = int(np.argmax(vl.real))
    w = np.abs(wl[:, j].real)
    v = v / v.sum()
    w = w / (w @ v)
    return float(vals[i].real), v, w


def y_threshold(rest, scal):
    """y at which rest + exp(y) scal has dominant eigenvalue one; +inf if no y does (R0 = 0)."""
    f = lambda y: math.log(max(spectral_radius(rest + math.exp(y) * scal), 1e-300))
    if f(60.0) < 0:
        return math.inf
    if f(-60.0) > 0:
        return -math.inf
    return brentq(f, -60.0, 60.0, xtol=1e-13, rtol=1e-15)


def gen_time(rest, scal, y):
    """1 / (elasticity of the dominant eigenvalue to the scaled transfer): the generation time of a ring."""
    A = rest + math.exp(y) * scal
    lam, v, w = perron(A)
    return lam / float(w @ (math.exp(y) * scal) @ v)


def gauss_mean_parts(model, Tbar, sigT):
    rest, scal = model.parts(Tbar + sigT * ZG)
    return np.tensordot(WG, rest, 1), np.tensordot(WG, scal, 1)


def season_temps(Tbar, amp, P):
    return Tbar + amp * np.sin(2 * math.pi * np.arange(P) / P)


def const_parts(model, Tbar):
    rest, scal = model.parts(np.array([Tbar]))
    return rest[0], scal[0]


# ----------------------------------------------------------------------------- second-order formula, coded twice
def cpp_resolvent(A0, dA, phi):
    """C'' for one AR(1) driver.  dA = sum over driven entries of kappa_e * (mean entry) * E_e (the first-order
    change of the matrix per unit of the driver).  C'' = (w N v)^2/2 - phi w N (I - phi T)^-1 Q N v, N = dA/lambda."""
    lam, v, w = perron(A0)
    n = A0.shape[0]
    Q = np.eye(n) - np.outer(v, w)
    T = (A0 / lam) @ Q
    N = dA / lam
    c = float(w @ N @ v)
    if phi == 0:
        return c * c / 2
    return c * c / 2 - phi * float(w @ N @ np.linalg.solve(np.eye(n) - phi * T, Q @ N @ v))


def cpp_lagsum(A0, entries, kappas, phi, nlag=20000, tol=1e-17):
    """The same coefficient as a sum over pairs of entries and over lags (the coding of the function `terms`
    in checks/host_crash_kills_parasites.py, written again here).  entries: list of (i, j, amount)."""
    lam, v, w = perron(A0)
    n = A0.shape[0]
    Q = np.eye(n) - np.outer(v, w)
    T = (A0 / lam) @ Q
    B = []
    for (i, j, a) in entries:
        Be = np.zeros((n, n))
        Be[i, j] = a / lam
        B.append(Be)
    c = [float(w @ Be @ v) for Be in B]
    tot = 0.0
    for e in range(len(B)):
        for e2 in range(len(B)):
            acc = 0.0
            if phi != 0:
                Tm = np.eye(n)
                for m in range(nlag):
                    rho = phi ** (m + 1)
                    acc += rho * float(w @ B[e] @ Tm @ Q @ B[e2] @ v)
                    Tm = Tm @ T
                    if abs(rho) * np.abs(Tm).max() < tol:
                        break
            tot += kappas[e] * kappas[e2] * (c[e] * c[e2] / 2 - acc)
    return tot


def cpp_season_tangent(A0, dA, omega):
    """Sinusoid of unit amplitude on the driver, first-order change dA of the matrix per unit of the driver."""
    lam, v, w = perron(A0)
    n = A0.shape[0]
    Q = np.eye(n) - np.outer(v, w)
    T = (A0 / lam) @ Q
    N = dA / lam
    c = float(w @ N @ v)
    z = complex(math.cos(omega), math.sin(omega))
    return c * c / 4 - 0.5 * (z * (w @ N @ np.linalg.solve(np.eye(n) - z * T, Q @ N @ v))).real


def cpp_season_full(Aseq):
    """Second order in the departure of the daily matrices from their mean over one period, all harmonics."""
    P, n, _ = Aseq.shape
    Abar = Aseq.mean(axis=0)
    lam, v, w = perron(Abar)
    Q = np.eye(n) - np.outer(v, w)
    T = (Abar / lam) @ Q
    F = np.fft.fft(Aseq, axis=0) / P
    tot = 0.0
    for k in range(1, (P - 1) // 2 + 1):
        N = F[k] / lam
        z = complex(math.cos(2 * math.pi * k / P), math.sin(2 * math.pi * k / P))
        c = w @ N @ v
        tot += abs(c) ** 2 - 2 * (z * (w @ N @ np.linalg.solve(np.eye(n) - z * T, Q @ np.conj(N) @ v))).real
    return tot


def gamma0_one_transfer(taus, phi):
    """Manuscript equation (2), per unit sigma^2, with geometric stays."""
    taus = np.asarray(taus, float)
    TT = taus.sum()
    if phi == 0:
        return 1 / (2 * TT)
    th = phi / (1 - phi)
    R = float(np.prod(1 / (1 + taus / th)))
    return (1 + 2 * th) / (2 * TT) - R / (1 - R)


def gamma0_general_R(TT, R, phi):
    """Equation (2) with R = E[phi^L] supplied (claim I1)."""
    if phi == 0:
        return 1 / (2 * TT)
    th = phi / (1 - phi)
    return (1 + 2 * th) / (2 * TT) - R / (1 - R)


def gamma0_clock(taus, expo, phi):
    """Claim T2: second-order loss per unit variance of the driver, ring at its threshold, geometric stays taus,
    expo = {stage: (pi, chi)} the log-sensitivities of the completion probability and of the clock."""
    taus = np.asarray(taus, float)
    n = len(taus)
    TT = taus.sum()
    ks = sorted(expo)
    if phi == 0:
        return sum(expo[k][0] for k in ks) ** 2 / (2 * TT)
    th = phi / (1 - phi)
    rho = 1 / (1 + taus / th)
    R = float(np.prod(rho))

    def seg(k, j):                       # product of rho over the stages after k up to and including j
        prod, i = 1.0, (k + 1) % n
        while True:
            prod *= rho[i]
            if i == j:
                return prod
            i = (i + 1) % n

    K = (1 + 2 * th) / (2 * TT) - R / (1 - R)
    tot = 0.0
    for a, j in enumerate(ks):
        pj, cj = expo[j]
        tot += pj * pj * K + pj * cj * (rho[j] - R) / (1 - R)
        for k in ks[a + 1:]:
            tot += pj * expo[k][0] * ((1 + 2 * th) / TT - (seg(k, j) + seg(j, k)) / (1 - R))
        for k in ks:
            if k != j:
                tot -= pj * expo[k][1] * seg(k, j) * (1 - rho[k]) / (1 - R)
    return tot


def hermite_psi(z, J):
    """Normalised Hermite polynomials He_j(z)/sqrt(j!), j = 0..J, by the stable recurrence."""
    out = np.empty((J + 1, z.size))
    out[0] = 1.0
    if J >= 1:
        out[1] = z
    for j in range(1, J):
        out[j + 1] = (z * out[j] - math.sqrt(j) * out[j - 1]) / math.sqrt(j + 1)
    return out


class Formulas:
    """The three versions of the second-order loss for a Gaussian AR(1) temperature (protocol, section 4)."""

    def __init__(self, model, Tbar, sigT, J=200):
        self.model, self.Tbar, self.sigT, self.J = model, Tbar, sigT, J
        self.rest, self.scal = model.parts(Tbar + sigT * ZG)
        self.rbar = np.tensordot(WG, self.rest, 1)
        self.sbar = np.tensordot(WG, self.scal, 1)
        self.y_mean = y_threshold(self.rbar, self.sbar)
        r0, s0 = const_parts(model, Tbar)
        self.r0, self.s0 = r0, s0
        self.y_tbar = y_threshold(r0, s0)
        self._coef = None

    def J_part(self):
        return self.y_tbar - self.y_mean

    def tangent(self, phi, h=1e-4):
        """(a): everything evaluated at Tbar."""
        if not math.isfinite(self.y_tbar):
            return math.nan
        ey = math.exp(self.y_tbar)
        rp, sp = const_parts(self.model, self.Tbar + h)
        rm, sm = const_parts(self.model, self.Tbar - h)
        dA = self.sigT * ((rp - rm) + ey * (sp - sm)) / (2 * h)
        A0 = self.r0 + ey * self.s0
        return gen_time(self.r0, self.s0, self.y_tbar) * cpp_resolvent(A0, dA, phi)

    def _coefs(self):
        if self._coef is None:
            K = ZG.size
            psiw = hermite_psi(ZG, self.J) * WG
            n = self.rest.shape[1]
            self._coef = (psiw @ self.rest.reshape(K, n * n)).reshape(self.J + 1, n, n), \
                         (psiw @ self.scal.reshape(K, n * n)).reshape(self.J + 1, n, n)
        return self._coef

    def cost_hermite(self, phi, y):
        """Cost per step at log multiplier y: (first Hermite term only, all terms, share of the variance of the
        first-order growth captured by the J terms kept)."""
        cr, cs = self._coefs()
        ey = math.exp(y)
        A0 = self.rbar + ey * self.sbar
        lam, v, w = perron(A0)
        n = A0.shape[0]
        Q = np.eye(n) - np.outer(v, w)
        T = (A0 / lam) @ Q
        wAv = ((self.rest + ey * self.scal) @ v) @ w / lam
        var0 = float(np.sum(WG * wAv * wAv) - 1.0)
        mem, first, cap = 0.0, 0.0, 0.0
        for j in range(1, self.J + 1):
            N = (cr[j] + ey * cs[j]) / lam
            c = float(w @ N @ v)
            cap += c * c
            if phi != 0:
                pj = phi ** j
                term = pj * float(w @ N @ np.linalg.solve(np.eye(n) - pj * T, Q @ N @ v))
                mem += term
            else:
                term = 0.0
            if j == 1:
                first = c * c / 2 - term
        return first, var0 / 2 - mem, (cap / var0 if var0 > 0 else 1.0)

    def hermite(self, phi):
        """(b) and (c) at the threshold of the matrix of mean rates, as losses of ln R0."""
        TT = gen_time(self.rbar, self.sbar, self.y_mean)
        first, full, cap = self.cost_hermite(phi, self.y_mean)
        return TT * first, TT * full, cap

    def hermite_root(self, phi):
        """(c'): the root in y of ln lambda(y) - cost(y) = 0, minus y_mean."""
        f = lambda y: math.log(spectral_radius(self.rbar + math.exp(y) * self.sbar)) - self.cost_hermite(phi, y)[1]
        lo, hi = self.y_mean - 1.0, self.y_mean + 1.0
        for _ in range(8):
            if f(lo) < 0 < f(hi):
                return brentq(f, lo, hi, xtol=1e-10) - self.y_mean
            lo -= 2.0
            hi += 2.0
        return math.nan


# ----------------------------------------------------------------------------- periodic environment: Floquet and brute force
def floquet_r(rest, scal, y):
    ey = math.exp(y)
    P, n, _ = rest.shape
    Mo = np.eye(n)
    logs = 0.0
    for t in range(P):
        Mo = (rest[t] + ey * scal[t]) @ Mo
        if t % 16 == 15:
            nrm = np.abs(Mo).max()
            Mo /= nrm
            logs += math.log(nrm)
    return (math.log(spectral_radius(Mo)) + logs) / P


def floquet_root(rest, scal, y_guess=0.0):
    f = lambda y: floquet_r(rest, scal, y)
    lo, hi = y_guess - 1.0, y_guess + 1.0
    for _ in range(40):
        if f(lo) < 0:
            break
        lo -= 2.0
    for _ in range(40):
        if f(hi) > 0:
            break
        hi += 2.0
    return brentq(f, lo, hi, xtol=1e-13, rtol=1e-15)


def brute_r(rest, scal, y, years=400, discard=100):
    """Iterate the periodic system and read the growth per step over the years after `discard`."""
    ey = math.exp(y)
    P, n, _ = rest.shape
    A = rest + ey * scal
    x = np.ones(n) / n
    tot = 0.0
    for yr in range(years):
        g = 0.0
        for t in range(P):
            x = A[t] @ x
            if t % 32 == 31:
                s = x.sum()
                x /= s
                g += math.log(s)
        s = x.sum()
        x /= s
        g += math.log(s)
        if yr >= discard:
            tot += g
    return tot / ((years - discard) * P)


def season_case(model, Tbar, amp, P=365):
    """Exact J, D and total for a sinusoidal temperature, by Floquet."""
    rest, scal = model.parts(season_temps(Tbar, amp, P))
    y_mean = y_threshold(rest.mean(axis=0), scal.mean(axis=0))
    r0, s0 = const_parts(model, Tbar)
    y_tbar = y_threshold(r0, s0)
    y_star = floquet_root(rest, scal, y_mean)
    return dict(y_mean=y_mean, y_tbar=y_tbar, y_star=y_star, J=y_tbar - y_mean, D=y_star - y_mean,
                total=y_tbar - y_star, rest=rest, scal=scal)


def season_formulas(model, Tbar, amp, P=365, h=1e-4):
    """Second-order loss for the sinusoid: tangent at Tbar, and all harmonics about the matrix of mean rates."""
    out = {}
    r0, s0 = const_parts(model, Tbar)
    y_tbar = y_threshold(r0, s0)
    if math.isfinite(y_tbar):
        ey = math.exp(y_tbar)
        rp, sp = const_parts(model, Tbar + h)
        rm, sm = const_parts(model, Tbar - h)
        dA = amp * ((rp - rm) + ey * (sp - sm)) / (2 * h)
        out["tangent"] = gen_time(r0, s0, y_tbar) * cpp_season_tangent(r0 + ey * s0, dA, 2 * math.pi / P)
    else:
        out["tangent"] = math.nan
    rest, scal = model.parts(season_temps(Tbar, amp, P))
    rb, sb = rest.mean(axis=0), scal.mean(axis=0)
    y_mean = y_threshold(rb, sb)
    out["full"] = gen_time(rb, sb, y_mean) * cpp_season_full(rest + math.exp(y_mean) * scal)
    return out


# ----------------------------------------------------------------------------- simulators
def ar1_block(rng, zlast, phi, C, M):
    eps = rng.standard_normal((C, M))
    if phi == 0:
        return eps, eps[-1]
    z, _ = lfilter([math.sqrt(1 - phi * phi)], [1.0, -phi], eps, axis=0, zi=(phi * zlast)[None, :])
    return z, z[-1]


def sim_ring_A(n, s0, g0, ks, vary, phi, ys, M, T, burn, seed, chunk=400, start=None):
    """Path A.  Ring with fixed rates s0, g0 except the entries returned by vary(z_block) as a list of
    (kind, stage, array[C, M]) with kind 0 = staying and 1 = transfer.  exp(y) multiplies the transfer out
    of stage ks.  Returns the growth rate per step of each replicate at each y: array (len(ys), M)."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ny = ys.size
    ey = np.exp(ys)[:, None]
    s0 = np.asarray(s0, float)[:, None, None]
    g0 = np.asarray(g0, float)[:, None, None]
    zlast = rng.standard_normal(M)
    X = np.full((n, ny, M), 1.0 / n)             # stage first, so that every slice is contiguous
    if start is not None:
        X[:] = np.asarray(start, float)[:, None, None]
    acc = np.zeros((ny, M))
    done, total = 0, burn + T
    while done < total:
        C = min(chunk, total - done)
        z, zlast = ar1_block(rng, zlast, phi, C, M)
        var = vary(z)
        tots = np.empty((C, ny, M))
        for t in range(C):
            SX = X * s0
            GX = X * g0
            for kind, k, arr in var:
                if kind == 0:
                    np.multiply(X[k], arr[t], out=SX[k])
                else:
                    np.multiply(X[k], arr[t], out=GX[k])
            GX[ks] *= ey
            SX[1:] += GX[:-1]
            SX[0] += GX[-1]
            tot = SX.sum(axis=0)
            tots[t] = tot
            X = SX / tot
        lo = max(burn - done, 0)
        if lo < C:
            acc += np.log(tots[lo:]).sum(axis=0)
        done += C
    return acc / T


def sim_B(parts_of_z, n, phi, ys, M, T, burn, seed):
    """Path B.  Explicit matrices for every replicate, AR(1) one step at a time, state normalised by its
    largest entry.  parts_of_z(z) -> (rest[M, n, n], scal[M, n, n])."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ey = np.exp(ys)[:, None, None]
    c = math.sqrt(1 - phi * phi)
    z = rng.standard_normal(M)
    X = np.ones((ys.size, M, n))
    acc = np.zeros((ys.size, M))
    for t in range(burn + T):
        z = phi * z + c * rng.standard_normal(M)
        rest, scal = parts_of_z(z)
        Y = np.einsum("mij,ymj->ymi", rest, X) + ey * np.einsum("mij,ymj->ymi", scal, X)
        nrm = Y.max(axis=-1)
        if t >= burn:
            acc += np.log(nrm)
        X = Y / nrm[..., None]
    return acc / T


def sim_delay(sA, gA, G, b, sigma, phi, ys, M, T, burn, seed, chunk=400):
    """Path C.  Stage A as a chain of len(sA) sub-stages (staying sA, transfers gA between them), the
    transfer out of its last sub-stage multiplied by exp(y) G h(t), then a fixed delay of b steps back into
    the first sub-stage.  h = exp(sigma z - sigma^2/2).  Growth is read from the last sub-stage of A.  If no
    sub-stage has staying the ring is a pure cycle of b + len(sA) steps whose cohorts never mix; growth is
    then read between two times a whole number of cycles apart, so that one cohort is followed."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ny = ys.size
    ey = np.exp(ys)[:, None]
    sA = np.asarray(sA, float)[:, None, None]
    gA = np.asarray(gA, float)[:, None, None]
    mA = sA.shape[0]
    pure = bool(np.all(sA == 0))
    cyc = b + mA
    zlast = rng.standard_normal(M)
    X = np.ones((mA, ny, M))
    buf = np.ones((b + 1, ny, M))
    logacc = np.zeros((ny, M))
    L0 = L1 = None
    n1 = T
    done, total = 0, burn + T
    while done < total:
        C = min(chunk, total - done)
        z, zlast = ar1_block(rng, zlast, phi, C, M)
        H = np.exp(sigma * z - sigma * sigma / 2) * G
        for t in range(C):
            tt = done + t
            if tt == burn:
                L0 = logacc + np.log(X[-1])
            elif pure and tt > burn and (tt - burn) % cyc == 0:
                L1 = logacc + np.log(X[-1])
                n1 = tt - burn
            buf[tt % (b + 1)] = ey * H[t] * X[-1]
            back = buf[(tt + 1) % (b + 1)]
            Xn = X * sA
            if mA > 1:
                Xn[1:] += X[:-1] * gA[:-1]
            Xn[0] += back
            X = Xn
            if tt % 16 == 15:
                f = X.sum(axis=0)
                X /= f
                buf /= f
                logacc += np.log(f)
        done += C
    if not pure or (total - burn) % cyc == 0:
        L1 = logacc + np.log(X[-1])
        n1 = T
    return (L1 - L0) / n1


def _dd_block(fn, T, m):
    """Weights for a block of temperatures T[C, M], sub-stage index first within each step: W[C, m, M], Erev[C, m, M]."""
    W, E = dd_weights(*fn(T), m)
    return np.ascontiguousarray(np.moveaxis(W, -1, 1)), np.ascontiguousarray(np.moveaxis(E[..., ::-1], -1, 1))


def sim_dd_A(m, s1, fn, Tbar, sigT, phi, ys, M, T, burn, seed, chunk=200):
    """Model DD, path A: sheltered stage and incubation chain with the exact one-day map."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ny = ys.size
    ey = np.exp(ys)[:, None]
    zlast = rng.standard_normal(M)
    x1 = np.full((ny, M), 0.5)
    U = np.full((m, ny, M), 0.5 / m)
    acc = np.zeros((ny, M))
    done, total = 0, burn + T
    while done < total:
        C = min(chunk, total - done)
        z, zlast = ar1_block(rng, zlast, phi, C, M)
        W, Erev = _dd_block(fn, Tbar + sigT * z, m)
        tots = np.empty((C, ny, M))
        for t in range(C):
            Wt = W[t][:, None, :]
            out = (U * Erev[t][:, None, :]).sum(axis=0)
            Un = U * Wt[0]
            for k in range(1, m):
                Un[k:] += U[:m - k] * Wt[k]
            Un[0] += ey * (1 - s1) * x1
            x1 = s1 * x1 + out
            tot = x1 + Un.sum(axis=0)
            tots[t] = tot
            x1 = x1 / tot
            U = Un / tot
        lo = max(burn - done, 0)
        if lo < C:
            acc += np.log(tots[lo:]).sum(axis=0)
        done += C
    return acc / T


def sim_dd_cohort(m, fn, Tbar, sigT, phi, M, T, burn, seed, chunk=200):
    """One individual enters the chain every day.  Returns per replicate the mean number completing per day
    (the completion probability averaged over entry days) and the mean age in days at completion."""
    rng = np.random.default_rng(seed)
    zlast = rng.standard_normal(M)
    U = np.zeros((m, M))
    A = np.zeros((m, M))                 # age-weighted mass
    out_mass = np.zeros(M)
    out_age = np.zeros(M)
    done, total = 0, burn + T
    while done < total:
        C = min(chunk, total - done)
        z, zlast = ar1_block(rng, zlast, phi, C, M)
        W, Erev = _dd_block(fn, Tbar + sigT * z, m)
        for t in range(C):
            U[0] += 1.0
            AU = A + U
            if done + t >= burn:
                out_mass += (U * Erev[t]).sum(axis=0)
                out_age += (AU * Erev[t]).sum(axis=0)
            Wt = W[t]
            Un = U * Wt[0]
            An = AU * Wt[0]
            for k in range(1, m):
                Un[k:] += U[:m - k] * Wt[k]
                An[k:] += AU[:m - k] * Wt[k]
            U, A = Un, An
        done += C
    return out_mass / T, out_age / np.maximum(out_mass, 1e-300)


def dd_constant_stats(W, E):
    """Completion probability and mean age at completion for a constant one-day map (W, E of length m)."""
    m = W.size
    L = np.zeros((m, m))
    for j in range(m):
        for i in range(j, m):
            L[i, j] = W[i - j]
    ex = E[::-1]
    e0 = np.zeros(m)
    e0[0] = 1.0
    occ = np.linalg.solve(np.eye(m) - L, e0)
    occ2 = np.linalg.solve(np.eye(m) - L, occ)
    prob = float(ex @ occ)
    return prob, float(ex @ occ2) / prob


def dd_season_stats(m, fn, Tbar, amp, P=365):
    """Exact periodic counterpart of sim_dd_cohort."""
    W, E = dd_weights(*fn(season_temps(Tbar, amp, P)), m)
    Ls = np.zeros((P, m, m))
    for j in range(m):
        for i in range(j, m):
            Ls[:, i, j] = W[:, i - j]
    Erev = E[:, ::-1]
    e0 = np.zeros(m)
    e0[0] = 1.0
    # periodic state of (U, A) at the start of day 0, before that day's entrant: solve the linear fixed point
    # by propagating a basis.  State vector s = (U, A), affine map over one year.
    dim = 2 * m

    def year(s):
        U, A = s[:m].copy(), s[m:].copy()
        om = oa = 0.0
        for t in range(P):
            U = U + e0
            AU = A + U
            om += Erev[t] @ U
            oa += Erev[t] @ AU
            U, A = Ls[t] @ U, Ls[t] @ AU
        return np.concatenate([U, A]), om, oa

    base, _, _ = year(np.zeros(dim))
    Mlin = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        Mlin[:, i] = year(e)[0] - base
    s0 = np.linalg.solve(np.eye(dim) - Mlin, base)
    _, om, oa = year(s0)
    return om / P, oa / om


# ----------------------------------------------------------------------------- root finding with common random numbers
def quad_root(ys, rbar):
    """Root of the quadratic through three points, nearest the middle one; and the slope there."""
    co = np.polyfit(ys - ys[1], rbar, 2)
    roots = np.roots(co)
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-12]
    if co[0] == 0 or not real:
        slope = (rbar[2] - rbar[0]) / (ys[2] - ys[0])
        x = -rbar[1] / slope
    else:
        x = min(real, key=abs)
        slope = 2 * co[0] * x + co[1]
    return ys[1] + x, float(slope)


def find_root(run, y0, delta, M, T, burn, seed, max_iter=6):
    """run(ys, T, burn, seed) -> growth rates (3, M).  A pilot of one-eighth length locates the root; the
    final run brackets it closely.  Returns root, its standard error, the per-replicate roots (for paired
    comparisons between cases run with the same seed), and the number of runs."""
    Tp = max(T // 8, 2000)
    centre, d = y0, delta
    runs = 0
    for _ in range(max_iter):
        ys = centre + d * np.array([-1.0, 0.0, 1.0])
        r = run(ys, Tp, burn, seed + 7919)
        runs += 1
        root, slope = quad_root(ys, r.mean(axis=1))
        if abs(root - centre) <= 2 * d:
            break
        centre = centre + float(np.clip(root - centre, -4 * d, 4 * d))
    se_p = float(r[1].std(ddof=1) / math.sqrt(M) / abs(slope))
    centre = root
    d = float(np.clip(4 * se_p, 0.005, 0.05))
    for _ in range(max_iter):
        ys = centre + d * np.array([-1.0, 0.0, 1.0])
        r = run(ys, T, burn, seed)
        runs += 1
        root, slope = quad_root(ys, r.mean(axis=1))
        if abs(root - centre) <= 1.5 * d:
            break
        centre = root
    x = root
    L = np.array([(x - ys[1]) * (x - ys[2]) / ((ys[0] - ys[1]) * (ys[0] - ys[2])),
                  (x - ys[0]) * (x - ys[2]) / ((ys[1] - ys[0]) * (ys[1] - ys[2])),
                  (x - ys[0]) * (x - ys[1]) / ((ys[2] - ys[0]) * (ys[2] - ys[1]))])
    rm = L @ r
    Dm = root - rm / slope
    return float(root), float(Dm.std(ddof=1) / math.sqrt(M)), Dm.astype(np.float64), runs


def one_stage_exact(s_of_z, g_of_z):
    """Exact D for one stage: root of E ln(s + e^y g) = 0 minus the root for the mean rates."""
    s, g = s_of_z, g_of_z
    sb, gb = float(WG @ s), float(WG @ g)
    y_mean = math.log((1 - sb) / gb)
    f = lambda y: float(WG @ np.log(s + math.exp(y) * g))
    return brentq(f, y_mean - 5, y_mean + 20, xtol=1e-14) - y_mean


def stage_bounds(fn, Tbar, sigT):
    """One exposed stage under Gaussian temperature: mean staying and transfer, the two Jensen gaps
    lg = -E ln(g/mean g) and ls = -E ln(s/mean s) (ln s = -(nu + mu) exactly, so no underflow), the mean stay,
    and the exact one-stage loss."""
    nu, mu = fn(Tbar + sigT * ZG)
    q, p = qp(nu, mu)
    sv, gv = 1 - q, p * q
    sb, gb = float(WG @ sv), float(WG @ gv)
    lg = -float(WG @ (np.log(p) + np.log(q))) + math.log(gb)
    ls = float(WG @ (nu + mu)) + math.log(sb)
    return dict(sb=sb, gb=gb, lg=lg, ls=ls, tau=1 / (1 - sb), D_one=one_stage_exact(sv, gv))
