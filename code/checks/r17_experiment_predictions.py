"""R17.  The decisive experiment, predictions redone (protocol: protocols/R17-experiment-predictions.md).

The first design (checks/experiment_predictions.py) predicts what a microcosm shows when host density
is switched between a high and a low level at the same arithmetic mean.  It assumes that scarcity
WASTES the stage that waits for a host, that every stay is memoryless, and that the schedule is a
square wave.  This script computes the same predictions without those three assumptions, with one
engine and no small-noise expansion.

Model (continuous time; host density constant between switches, so every piece is a matrix
exponential).  In hosts: mean stay a = 5 days, a chain of m_a equal sub-stages.  Outside hosts: mean
stay b days at the mean host density, a chain of m_b equal sub-stages; only the last one meets hosts.
With h = host density / its mean and f the encounter function (f(h) = h, or the saturating
f(h) = 2h/(1 + h)), the last sub-stage outside is left at rate (m_b/b)[d + (1 - d) f(h)] and sends
e^y (m_b/b) f(h) per head into hosts.  d = 1 is WASTE (the first design); d < 1 is WAIT, d being the
fraction that dies waiting at the mean density.  y = ln R0 of the constant treatment.

Schedules, all with arithmetic mean density one: square wave of period P (high half first); random
telegraph with mean dwell m (exponential dwells); a three-level random schedule with the same mean,
variance and autocorrelation as the telegraph.

D, the loss of ln R0, is the y at which the long-run growth rate is zero.
  square wave            Floquet product, root in y                              (computed)
  telegraph, 2 stages    quadrature of the stationary law of the stage fraction  (computed)
  telegraph with chains, three-level schedule
                         simulation, the vector carried exactly through each dwell by products of
                         precomputed matrix exponentials; standard errors from batch means (simulated)
What an experimenter would see: the model is RUN for 60 days from a stated start and read at days 30
and 60, over 16 starting phases (square wave) or 2,000 schedules (telegraph).  No asymptotic rate is
extrapolated.

Validation checks V1 to V7, with the tolerances of the protocol, are printed first.
Run:  python3 -I checks/r17_experiment_predictions.py        (3 processes)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import math
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy import integrate, linalg, optimize

# ----------------------------------------------------------------------------------------------
# grid (protocol, section 3)
A_HOST = 5.0
B_OUT = (2.0, 10.0, 50.0)
DELTAS = (0.8, 0.5)
KS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
KLAB = ("1/4", "1/2", "1", "2", "4", "8")
D_SET = (1.0, 0.99, 0.9, 0.5, 0.1)
SHAPES = [(mb, ma) for mb in (1, 4, 16) for ma in (1, 4, 16)]      # (outside, in hosts)
RESP = ("prop", "sat")
VARS = [(mb, ma, d, resp) for resp in RESP for d in D_SET for (mb, ma) in SHAPES]
Y_STATED = 0.5
DAYS = (30.0, 60.0)
NPHASE = 16
NREAL = 1000                     # telegraph schedules per starting level (2,000 per cell; amendment 1)
SIGMA_M2 = math.log(1.0 + 0.2 ** 2)   # variance of ln(count) at a coefficient of variation of 20%
SIGMA_M = math.sqrt(SIGMA_M2)

# simulation sizes (fixed after the timing test; protocol amendment 1)
N_PILOT = 1000
N_FINAL = 60000
N_BURN = 100
N_BATCH = 20
GRID_SIM = 65536                 # dwell times of the simulated schedules are rounded to m/65536
NDIG_SIM = 6
GRID_RUN = 1024                  # switching times in the 60-day telegraph runs are rounded to 1/1024 day
NDIG_RUN = 4
PILOT_GRID_TEL = (-0.2, 0.1, 0.4, 0.7, 1.0)
PILOT_GRID_3 = (-0.2, 0.2, 0.6, 1.0, 1.4)
HALF_BRACKET = 0.04
SEED = 20261114

V3_CELLS3 = [(1, 1, 1.0, "prop"), (1, 1, 0.5, "prop"), (16, 16, 1.0, "prop"), (16, 16, 0.5, "prop")]
SENS_VARS = [(1, 1, 1.0, "prop"), (1, 1, 0.5, "prop"), (16, 16, 1.0, "prop"), (16, 16, 0.5, "prop")]
D_FINE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)

FAILS = []
QUAD_MASS_ERR = [0.0]


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def vlabel(var):
    mb, ma, d, resp = var
    how = "waste      " if d == 1.0 else f"wait d={d:<4g}"
    return f"{how} out={mb:<2d} in={ma:<2d} {resp}"


# ----------------------------------------------------------------------------------------------
# the model
def f_enc(h, resp):
    return h if resp == "prop" else 2.0 * h / (1.0 + h)


def generator(var, b, fh, y):
    """Generator of the two-stage cycle.  Order of states: sub-stages in hosts, then outside."""
    mb, ma, d, _ = var
    n = ma + mb
    A = np.zeros((n, n))
    ra = ma / A_HOST
    rb = mb / b
    for i in range(ma):
        A[i, i] = -ra
        A[i + 1, i] = ra
    for j in range(mb - 1):
        k = ma + j
        A[k, k] = -rb
        A[k + 1, k] = rb
    k = n - 1
    A[k, k] = -rb * (d + (1.0 - d) * fh)
    A[0, k] = math.exp(y) * rb * fh
    return A


def levels_tel(var, b, delta, y):
    resp = var[3]
    return [generator(var, b, f_enc(1 + delta, resp), y), generator(var, b, f_enc(1 - delta, resp), y)]


def levels_three(var, b, delta, y):
    resp = var[3]
    d3 = delta * math.sqrt(1.5)
    return [generator(var, b, f_enc(h, resp), y) for h in (1 - d3, 1.0, 1 + d3)]


def dominant(A):
    return float(np.linalg.eigvals(A).real.max())


def perron_right(A):
    w, V = np.linalg.eig(A)
    v = V[:, np.argmax(w.real)].real
    v = np.abs(v)
    return v / v.sum()


def perron_left(A):
    return perron_right(A.T)


# ----------------------------------------------------------------------------------------------
# square wave: Floquet
def floquet_rate(var, b, delta, P, y):
    Ah, Al = levels_tel(var, b, delta, y)
    M = linalg.expm(Al * (P / 2)) @ linalg.expm(Ah * (P / 2))
    return math.log(max(abs(np.linalg.eigvals(M)))) / P


def root_in_y(fun, lo=-0.6, hi=1.6):
    flo, fhi = fun(lo), fun(hi)
    while flo > 0:
        lo -= 1.0
        flo = fun(lo)
    while fhi < 0:
        hi += 1.0
        fhi = fun(hi)
    return optimize.brentq(fun, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=300)


def floquet_D(var, b, delta, P):
    if delta == 0.0:
        return root_in_y(lambda y: dominant(generator(var, b, 1.0, y)))
    return root_in_y(lambda y: floquet_rate(var, b, delta, P, y))


# the same generator in steps of dt days (dt = 1: the model of the first design)
def step_rate(var, b, delta, half_steps, dt, y):
    Ah, Al = levels_tel(var, b, delta, y)
    n = Ah.shape[0]
    Mh = np.eye(n) + Ah * dt
    Ml = np.eye(n) + Al * dt
    M = np.linalg.matrix_power(Ml, half_steps) @ np.linalg.matrix_power(Mh, half_steps)
    return math.log(max(abs(np.linalg.eigvals(M)))) / (2 * half_steps * dt)


def step_D(var, b, delta, half_steps, dt):
    return root_in_y(lambda y: step_rate(var, b, delta, half_steps, dt, y))


# ----------------------------------------------------------------------------------------------
# telegraph, two stages: quadrature of the stationary law
def tele2d_rate(var, b, delta, m, y):
    """Growth rate of the two-stage cycle (memoryless stays) under a telegraph with mean dwell m.

    u = fraction of the parasites that is outside hosts.  In level i: du/dt = F_i(u) =
    aH (1 - u) - l_i u - q_i u^2 and d ln N/dt = q_i u, with k_i = e^y f_i/b, l_i = (d + (1-d) f_i)/b,
    q_i = k_i - l_i.  F_i = -c_i (u - u_i)(1 - s_i u), u_i its root in (0, 1).  Zero net flux in the
    stationary law gives F_hi p_hi = -F_lo p_lo = -phi and
    phi = prod_i |u - u_i|^(kappa/gamma_i) (1 - s_i u)^(-kappa/gamma_i), gamma_i = -F_i'(u_i).
    """
    mb, ma, d, resp = var
    assert mb == 1 and ma == 1
    aH = 1.0 / A_HOST
    kap = 1.0 / m

    def lev(f):
        k = math.exp(y) * f / b
        l = (d + (1.0 - d) * f) / b
        q = k - l
        gam = math.sqrt((aH + l) ** 2 + 4.0 * q * aH)
        c = 0.5 * (aH + l + gam)
        return q, gam, c, aH / c, -q / c

    qh, gh, ch, uh, sh = lev(f_enc(1 + delta, resp))
    ql, gl, cl, ul, sl = lev(f_enc(1 - delta, resp))
    if ul - uh < 1e-13:
        return qh * uh
    eh, el = kap / gh, kap / gl

    def W(u):
        return (1.0 - sh * u) ** (-eh) * (1.0 - sl * u) ** (-el)

    def dh(u):
        return W(u) / (ch * (1.0 - sh * u))

    def dl(u):
        return W(u) / (cl * (1.0 - sl * u))

    kw = dict(epsabs=0.0, epsrel=1e-12, limit=400)
    Ih = integrate.quad(dh, uh, ul, weight="alg", wvar=(eh - 1.0, el), **kw)[0]
    Il = integrate.quad(dl, uh, ul, weight="alg", wvar=(eh, el - 1.0), **kw)[0]
    Nh = integrate.quad(lambda u: qh * u * dh(u), uh, ul, weight="alg", wvar=(eh - 1.0, el), **kw)[0]
    Nl = integrate.quad(lambda u: ql * u * dl(u), uh, ul, weight="alg", wvar=(eh, el - 1.0), **kw)[0]
    QUAD_MASS_ERR[0] = max(QUAD_MASS_ERR[0], abs(Ih / Il - 1.0))
    return (Nh + Nl) / (Ih + Il)


def tele2d_D(var, b, delta, m):
    return root_in_y(lambda y: tele2d_rate(var, b, delta, m, y))


# ----------------------------------------------------------------------------------------------
# second-order values (used only in the checks)
def stays(var, b):
    mb, ma = var[0], var[1]
    return np.array([A_HOST / ma] * ma + [b / mb] * mb)


def d2_random(var, b, delta, m):
    """Result 1 summed for a ring in continuous time (d = 1), and (C2) of the contact-limited note."""
    tau = stays(var, b)
    theta = m / 2.0
    q = 1.0 / (1.0 + tau / theta)
    R = float(np.prod(q))
    d = var[2]
    return delta ** 2 * (d * d * theta / tau.sum() - d * (R - (1.0 - d) * q[-1]) / (1.0 - R))


def d2_square_gain(var, b, delta, P, jmax=20001):
    """Result 7 summed over the harmonics of the square wave (gain-only)."""
    tau = stays(var, b)
    j = np.arange(1, jmax + 1, 2)
    w = 2.0 * math.pi * j / P
    g = np.prod(1.0 / (1.0 + 1j * np.outer(w, tau)), axis=1)
    eps = 4.0 * delta / (math.pi * j)
    return float(-np.sum(0.5 * eps ** 2 * (g / (1.0 - g)).real))


def d2_square_matrix(var, b, delta, P, jmax=4001):
    """Second-order loss under a square wave from the perturbation of the Floquet exponent, for any
    proportional variant: r2 = sum_j (eps_j^2/2) Re[w'B (i w_j - A0)^-1 B v]/(w'v).  Derived for this
    check (not refereed); it reduces to Result 7 when d = 1."""
    A0 = generator(var, b, 1.0, 0.0)
    B = generator(var, b, 2.0, 0.0) - A0
    n = A0.shape[0]
    v = perron_right(A0)
    w = perron_left(A0)
    dAdy = np.zeros((n, n))
    dAdy[0, n - 1] = A0[0, n - 1]
    c = (w @ dAdy @ v) / (w @ v)
    tot = 0.0
    Bv = B @ v
    wB = w @ B
    for j in range(1, jmax + 1, 2):
        wj = 2.0 * math.pi * j / P
        eps = 4.0 * delta / (math.pi * j)
        tot += 0.5 * eps ** 2 * (wB @ np.linalg.solve(1j * wj * np.eye(n) - A0, Bv)).real / (w @ v)
    return -tot / c


# ----------------------------------------------------------------------------------------------
# simulation of random schedules
def build_ladder(As, dt, ndig):
    """L[g, q] = exp(As * q * 16^g * dt), q = 0..15, for a stack of matrices As [U, n, n]."""
    U, n, _ = As.shape
    base = np.stack([linalg.expm(As[u] * dt) for u in range(U)])
    L = np.empty((ndig, 16, U, n, n))
    eye = np.broadcast_to(np.eye(n), (U, n, n))
    for g in range(ndig):
        L[g, 0] = eye
        L[g, 1] = base
        for q in range(2, 16):
            L[g, q] = np.matmul(L[g, q - 1], base)
        base = np.matmul(L[g, 15], base)
    return L


def ladder_error(L, As, dt, K):
    """Largest relative difference between the ladder product for K steps and the matrix exponential."""
    U = As.shape[0]
    worst = 0.0
    for u in range(0, U, max(1, U // 6)):
        M = np.eye(As.shape[1])
        g, kk = 0, K
        while kk:
            q = kk & 15
            if q:
                M = L[g, q, u] @ M
            kk >>= 4
            g += 1
        E = linalg.expm(As[u] * (K * dt))
        worst = max(worst, float(np.abs(M - E).max() / np.abs(E).max()))
    return worst


def draw_schedule(kind, ndwell, rng):
    """Levels and dwell lengths (in units of m/GRID_SIM) of a random schedule with mean dwell m
    (telegraph) or 3m/4 (three-level).  Dwell times are exponential, rounded to the grid."""
    if kind == "tel":
        lev = np.arange(ndwell) % 2
        K = np.rint(rng.exponential(1.0, ndwell) * GRID_SIM).astype(np.int64)
    else:
        lev = np.empty(ndwell, dtype=np.int64)
        lev[0] = rng.integers(3)
        jump = rng.integers(1, 3, ndwell)
        for i in range(1, ndwell):
            lev[i] = (lev[i - 1] + jump[i]) % 3
        K = np.rint(rng.exponential(0.75, ndwell) * GRID_SIM).astype(np.int64)
    return lev, K


def mean_rounded_dwell(mean, dt):
    """E[dt * round(X/dt)] for X exponential with the given mean."""
    return dt / (2.0 * math.sinh(dt / (2.0 * mean)))


def sim_growth(mats, wts, lev, K, dt, mean_dwell):
    """Long-run growth rates of a batch of units driven by ONE schedule.

    mats [U, Lv, n, n] generators per level; wts [U, n] positive weights (log growth is measured with
    them).  A cycle is two consecutive dwells.  Returns (rate [U], standard error [U], ladder error).
    Estimator: (mean log growth per cycle, adjusted by regression on the time spent at each level in
    the cycle minus its known expectation) / (expected cycle length); standard error from batch means.
    """
    U, Lv, n, _ = mats.shape
    lam = np.empty((Lv, U))
    ladders = []
    lerr = 0.0
    for l in range(Lv):
        lam[l] = np.linalg.eigvals(mats[:, l]).real.max(axis=1)
        shifted = mats[:, l] - lam[l][:, None, None] * np.eye(n)
        Ld = build_ladder(shifted, dt, NDIG_SIM)
        lerr = max(lerr, ladder_error(Ld, shifted, dt, 3 * GRID_SIM + 12345))
        ladders.append(Ld)
    ncyc = len(K) // 2
    x = np.full((U, n), 1.0 / n)
    Y = np.empty((ncyc, U))
    Tl = np.zeros((ncyc, Lv))
    acc = np.zeros(U)
    for i in range(2 * ncyc):
        l = int(lev[i])
        kk = int(K[i])
        t = kk * dt
        acc += lam[l] * t
        Tl[i >> 1, l] += t
        Ld = ladders[l]
        g = 0
        while kk:
            q = kk & 15
            if q:
                x = np.matmul(Ld[g, q], x[:, :, None])[:, :, 0]
            kk >>= 4
            g += 1
        if i & 1:
            s = (wts * x).sum(axis=1)
            Y[i >> 1] = acc + np.log(s)
            x /= s[:, None]
            acc[:] = 0.0
    Y = Y[N_BURN:]
    Tl = Tl[N_BURN:]
    md = mean_rounded_dwell(mean_dwell, dt)
    C = Tl - 2.0 * md / Lv
    Cc = C - C.mean(axis=0)
    beta = np.linalg.lstsq(Cc, Y - Y.mean(axis=0), rcond=None)[0]
    Yadj = Y - C @ beta
    cyc = 2.0 * md
    rate = Yadj.mean(axis=0) / cyc
    nb = Yadj.shape[0] // N_BATCH
    bm = Yadj[: nb * N_BATCH].reshape(N_BATCH, nb, U).mean(axis=1)
    se = bm.std(axis=0, ddof=1) / math.sqrt(N_BATCH) / cyc
    return rate, se, lerr


def quad_root(ys, rs, near=None):
    """Root of the quadratic through three points, nearest `near` (default: the middle point)."""
    c = np.polyfit(ys, rs, 2)
    roots = np.roots(c)
    roots = roots[np.abs(roots.imag) < 1e-9].real
    if len(roots) == 0:
        return float("nan"), float("nan")
    r = roots[np.argmin(np.abs(roots - (ys[1] if near is None else near)))]
    return float(r), float(2 * c[0] * r + c[1])


def sim_cells(cells, b, m, kind, seed, n_pilot, n_final):
    """D and the growth rate at the stated R0 for cells (var, delta) of one dimension, one schedule."""
    levels = levels_tel if kind == "tel" else levels_three
    grid = PILOT_GRID_TEL if kind == "tel" else PILOT_GRID_3
    mean_dwell = m if kind == "tel" else 0.75 * m
    dt = m / GRID_SIM
    nc = len(cells)

    def run(ylists, ncyc, sd):
        rng = np.random.default_rng(sd)
        lev, K = draw_schedule(kind, 2 * (ncyc + N_BURN), rng)
        mats, wts = [], []
        for (var, delta), ys in zip(cells_run, ylists):
            for y in ys:
                mats.append(np.stack(levels(var, b, delta, y)))
                wts.append(perron_left(generator(var, b, 1.0, y)))
        rate, se, lerr = sim_growth(np.stack(mats), np.stack(wts), lev, K, dt, mean_dwell)
        ny = len(ylists[0])
        return rate.reshape(len(cells_run), ny), se.reshape(len(cells_run), ny), lerr

    # pilot
    cells_run = cells
    rp, _, lerr = run([grid] * nc, n_pilot, seed + 1)
    gy = np.array(grid)
    centre = np.empty(nc)
    for i in range(nc):
        r = rp[i]
        j = int(np.searchsorted(r > 0, True))          # first grid point with positive growth
        j = min(max(j, 1), len(gy) - 1)
        lo = max(0, min(j - 1, len(gy) - 3))
        lin = gy[j - 1] - r[j - 1] * (gy[j] - gy[j - 1]) / (r[j] - r[j - 1])
        root, _ = quad_root(gy[lo:lo + 3], r[lo:lo + 3], near=lin)
        centre[i] = root if np.isfinite(root) else lin
    D = np.full(nc, np.nan)
    Dse = np.full(nc, np.nan)
    r05 = np.full(nc, np.nan)
    r05se = np.full(nc, np.nan)
    nrerun = 0
    todo = list(range(nc))
    for attempt in range(4):
        if not todo:
            break
        cells_run = [cells[i] for i in todo]
        ylists = [(centre[i] - HALF_BRACKET, centre[i], centre[i] + HALF_BRACKET, Y_STATED) for i in todo]
        rf, sf, le = run(ylists, n_final, seed + 2)
        lerr = max(lerr, le)
        again = []
        for jj, i in enumerate(todo):
            ys = np.array(ylists[jj][:3])
            root, slope = quad_root(ys, rf[jj, :3])
            r05[i], r05se[i] = rf[jj, 3], sf[jj, 3]
            if np.isfinite(root) and abs(root - centre[i]) <= HALF_BRACKET:
                D[i], Dse[i] = root, sf[jj, 1] / slope
            else:
                if not np.isfinite(root):
                    root = centre[i] - rf[jj, 1] * 2 * HALF_BRACKET / (rf[jj, 2] - rf[jj, 0])
                centre[i] = root
                again.append(i)
        nrerun += len(again)
        todo = again
    return D, Dse, r05, r05se, lerr, nrerun, len(todo)


# ----------------------------------------------------------------------------------------------
# actual runs of 60 days
def start_vector(var, b, y, start):
    A = generator(var, b, 1.0, y)
    if start == "stable":
        return perron_right(A)
    x = np.zeros(A.shape[0])
    x[var[1]] = 1.0                                   # every parasite in the first sub-stage outside
    return x


def const_counts(var, b, y, x0):
    A = generator(var, b, 1.0, y)
    ma = var[1]
    out = np.empty((2, 2))
    for i, day in enumerate(DAYS):
        x = linalg.expm(A * day) @ x0
        out[i] = x.sum(), x[:ma].sum()
    return out


def run_square(var, b, delta, P, y, x0):
    """Counts (total, in hosts) at days 30 and 60 for 16 starting phases.  [phase, day, 2]"""
    Ah, Al = levels_tel(var, b, delta, y)
    ma = var[1]
    half = P / 2.0
    cache = {}

    def E(l, dt):
        key = (l, round(dt, 10))
        if key not in cache:
            cache[key] = linalg.expm((Ah if l == 0 else Al) * dt)
        return cache[key]

    out = np.empty((NPHASE, 2, 2))
    for j in range(NPHASE):
        l = 0 if j < NPHASE // 2 else 1
        rem = (NPHASE // 2 - j) * P / NPHASE if l == 0 else (NPHASE - j) * P / NPHASE
        x = x0.copy()
        t = 0.0
        for i, day in enumerate(DAYS):
            while day - t > 1e-9:
                dt = min(rem, day - t)
                x = E(l, dt) @ x
                t += dt
                rem -= dt
                if rem <= 1e-9:
                    l ^= 1
                    rem = half
            out[j, i] = x.sum(), x[:ma].sum()
    return out


def draw_run_schedules(m, rng):
    """Dwell lengths, in units of 1/GRID_RUN day, for NREAL telegraph schedules.  Each is used twice,
    once starting high and once starting low with the same switching times (a mirrored pair), so the
    time spent high and low is balanced over the 2*NREAL runs exactly (amendment 2)."""
    S = int(60.0 / m * 3) + 12
    while True:
        E = rng.exponential(m, size=(NREAL, S))
        if (E.sum(axis=1) > 61.0).all():
            break
        S *= 2
    K = np.rint(E * GRID_RUN).astype(np.int64)
    return np.stack([K, K])


def run_telegraph(var, b, delta, y, x0, Kint):
    """Counts (total, in hosts) at days 30 and 60 for the 2*NREAL schedules.  [2*NREAL, day, 2]"""
    Ah, Al = levels_tel(var, b, delta, y)
    ma = var[1]
    Lf = [build_ladder(A[None], 1.0 / GRID_RUN, NDIG_RUN)[:, :, 0] for A in (Ah, Al)]
    out = np.empty((2 * NREAL, 2, 2))
    for g0 in (0, 1):
        K = Kint[g0]
        R, S = K.shape
        start = np.cumsum(K, axis=1) - K
        X = np.tile(x0, (R, 1))
        lo = 0
        for i, day in enumerate(DAYS):
            hi = int(day * GRID_RUN)
            dur = np.clip(np.minimum(start + K, hi) - np.maximum(start, lo), 0, None)
            for s in range(S):
                ds = dur[:, s]
                if not ds.any():
                    continue
                Ll = Lf[(g0 + s) & 1]
                for g in range(NDIG_RUN):
                    q = (ds >> (4 * g)) & 15
                    if q.any():
                        X = np.matmul(Ll[g][q], X[:, :, None])[:, :, 0]
            out[g0 * R:(g0 + 1) * R, i, 0] = X.sum(axis=1)
            out[g0 * R:(g0 + 1) * R, i, 1] = X[:, :ma].sum(axis=1)
            lo = hi
    return out


NSTAT = 6


def summarise(lnr, kind):
    """[mean, low, high, sd, standard error of the mean, sd of pair means] of ln(N_const/N_fluct).
    low/high = min/max over phases (square) or the 10th/90th percentiles over schedules (telegraph).
    A pair is a phase with the phase half a period later, or a schedule with its mirror image; the
    two halves of the array are in that order.  The square wave's mean over 16 phases is not sampled,
    so its standard error is zero."""
    h = len(lnr) // 2
    pm = 0.5 * (lnr[:h] + lnr[h:])
    if kind == "sq":
        return np.array([lnr.mean(), lnr.min(), lnr.max(), lnr.std(), 0.0, pm.std()])
    return np.array([lnr.mean(), np.percentile(lnr, 10), np.percentile(lnr, 90), lnr.std(), pm.std(ddof=1) / math.sqrt(h), pm.std()])


# ----------------------------------------------------------------------------------------------
# one task: everything for one outside stay b and one time scale k
def task_bk(bi, ki, n_pilot, n_final):
    t0 = time.time()
    b = B_OUT[bi]
    T = A_HOST + b
    P = m = KS[ki] * T
    nv = len(VARS)
    res = {"bi": bi, "ki": ki}
    D = np.full((2, nv, 2), np.nan)
    Dse = np.zeros((2, nv, 2))
    r05 = np.full((2, nv, 2), np.nan)
    r05se = np.zeros((2, nv, 2))
    rc = np.empty(nv)
    Lr = np.empty((2, nv, 2, 2, 2, NSTAT))            # schedule, variant, delta, day, (total, hosts), stat
    Lks = np.empty((nv, 2, 2, 2, NPHASE), dtype=np.float32)       # ln(N_const/N_fluct) per starting phase
    Lkt = np.empty((nv, 2, 2, 2, 2 * NREAL), dtype=np.float32)    # ... and per telegraph schedule
    Dsim = np.full((nv, 2), np.nan)
    Dsimse = np.full((nv, 2), np.nan)
    rng_run = np.random.default_rng(SEED + 100 * bi + ki)
    Kint = draw_run_schedules(m, rng_run)
    for iv, var in enumerate(VARS):
        rc[iv] = dominant(generator(var, b, 1.0, Y_STATED))
        x0 = start_vector(var, b, Y_STATED, "stable")
        cc = const_counts(var, b, Y_STATED, x0)
        for di, delta in enumerate(DELTAS):
            D[0, iv, di] = floquet_D(var, b, delta, P)
            r05[0, iv, di] = floquet_rate(var, b, delta, P, Y_STATED)
            sq = run_square(var, b, delta, P, Y_STATED, x0)
            te = run_telegraph(var, b, delta, Y_STATED, x0, Kint)
            for day in range(2):
                for c in range(2):
                    Lks[iv, di, day, c] = np.log(cc[day, c] / sq[:, day, c])
                    Lkt[iv, di, day, c] = np.log(cc[day, c] / te[:, day, c])
                    Lr[0, iv, di, day, c] = summarise(np.log(cc[day, c] / sq[:, day, c]), "sq")
                    Lr[1, iv, di, day, c] = summarise(np.log(cc[day, c] / te[:, day, c]), "tel")
            if var[0] == 1 and var[1] == 1:
                D[1, iv, di] = tele2d_D(var, b, delta, m)
                r05[1, iv, di] = tele2d_rate(var, b, delta, m, Y_STATED)
    res["t_det"] = time.time() - t0
    # simulation, telegraph: every variant (the two-stage ones only to check the simulation, V5)
    lerr = 0.0
    nrer = 0
    nfail = 0
    for n in sorted({v[0] + v[1] for v in VARS}):
        cells = [(iv, di) for iv, v in enumerate(VARS) if v[0] + v[1] == n for di in range(2)]
        out = sim_cells([(VARS[iv], DELTAS[di]) for iv, di in cells], b, m, "tel", SEED + 1000 * bi + 10 * ki, n_pilot, n_final)
        lerr = max(lerr, out[4])
        nrer += out[5]
        nfail += out[6]
        for j, (iv, di) in enumerate(cells):
            Dsim[iv, di], Dsimse[iv, di] = out[0][j], out[1][j]
            if n > 2:
                D[1, iv, di], Dse[1, iv, di] = out[0][j], out[1][j]
                r05[1, iv, di], r05se[1, iv, di] = out[2][j], out[3][j]
    # three-level schedule
    D3 = np.empty((len(V3_CELLS3), 2))
    D3se = np.empty((len(V3_CELLS3), 2))
    for n in (2, 32):
        cells = [(j, di) for j, v in enumerate(V3_CELLS3) if v[0] + v[1] == n for di in range(2)]
        out = sim_cells([(V3_CELLS3[j], DELTAS[di]) for j, di in cells], b, m, "three", SEED + 1000 * bi + 10 * ki + 5, n_pilot, n_final)
        lerr = max(lerr, out[4])
        nrer += out[5]
        nfail += out[6]
        for jj, (j, di) in enumerate(cells):
            D3[j, di], D3se[j, di] = out[0][jj], out[1][jj]
    # sensitivity of the 60-day runs: a higher R0, and another start (swing 1.8/0.2 only)
    sens = np.empty((len(SENS_VARS), 2, 2, 2, NSTAT))  # variant, (R0 = e^1.5 | start outside), schedule, day, stat
    for j, var in enumerate(SENS_VARS):
        for s, (y, start) in enumerate(((1.5, "stable"), (Y_STATED, "outside"))):
            x0 = start_vector(var, b, y, start)
            cc = const_counts(var, b, y, x0)
            sq = run_square(var, b, 0.8, P, y, x0)
            te = run_telegraph(var, b, 0.8, y, x0, Kint)
            for day in range(2):
                sens[j, s, 0, day] = summarise(np.log(cc[day, 0] / sq[:, day, 0]), "sq")
                sens[j, s, 1, day] = summarise(np.log(cc[day, 0] / te[:, day, 0]), "tel")
    res.update(D=D, Dse=Dse, r05=r05, r05se=r05se, rc=rc, L=Lr, Lks=Lks, Lkt=Lkt, Dsim=Dsim, Dsimse=Dsimse, D3=D3, D3se=D3se,
               sens=sens, lerr=lerr, nrerun=nrer, nfail=nfail, quad=QUAD_MASS_ERR[0], secs=time.time() - t0)
    return res


def task_v2d(n_pilot, n_final):
    """Chains under the telegraph at a small swing, by simulation (V2d); and the chain with d = 0 at
    the large swing, for which the growth rate at y = 0 must be zero on every path (V7)."""
    b, m = 10.0, 30.0
    cells = [((4, 4, 1.0, "prop"), 0.2), ((4, 4, 0.5, "prop"), 0.2)]
    out = sim_cells(cells, b, m, "tel", SEED + 7, n_pilot, n_final)
    var0 = (4, 4, 0.0, "prop")
    rng = np.random.default_rng(SEED + 8)
    lev, K = draw_schedule("tel", 2 * (2000 + N_BURN), rng)
    r0, _, le = sim_growth(np.stack([np.stack(levels_tel(var0, b, 0.8, 0.0))]), np.stack([perron_left(generator(var0, b, 1.0, 0.0))]),
                           lev, K, m / GRID_SIM, m)
    return {"D": out[0], "se": out[1], "lerr": max(out[4], le), "v7": float(r0[0]) * (A_HOST + b)}


def run_lane(tasks):
    return [task_bk(*t) if t[0] != "v2d" else task_v2d(*t[1:]) for t in tasks]


# ----------------------------------------------------------------------------------------------
# reporting helpers
def nrep(delta, extra=0.0):
    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return 10 ** 9
    return int(math.ceil(16.0 * (2.0 * SIGMA_M2 + extra) / delta ** 2 - 1e-12))


def nfmt(n):
    return f"{n:d}" if n < 100000 else ">1e5"


TREATS = [(s, di, ki) for s in range(2) for di in range(2) for ki in range(6)]


HELD = (1.8, 0.2, 1.5, 0.5)             # densities held constant for the whole run (not variability treatments)


def tlabel(t):
    if t is None:
        return "constant"
    if t[0] == "held":
        return f"held at {HELD[t[1]]:.1f}"
    s, di, ki = t
    d = DELTAS[di]
    return f"{'square' if s == 0 else 'telegraph'} {1 + d:.1f}/{1 - d:.1f} {'period' if s == 0 else 'dwell'} {KLAB[ki]} gen"


def main(argv):
    t_start = time.time()
    n_pilot, n_final = N_PILOT, N_FINAL
    only = None
    for a in argv[1:]:
        if a.startswith("--nfinal="):
            n_final = int(a.split("=")[1])
        if a.startswith("--npilot="):
            n_pilot = int(a.split("=")[1])
        if a.startswith("--only="):
            only = [tuple(int(x) for x in p.split(".")) for p in a.split("=")[1].split(",")]
    print("R17: the decisive experiment, predictions redone.  Protocol: protocols/R17-experiment-predictions.md")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; python {sys.version.split()[0]}, numpy {np.__version__}; 3 processes")
    print(f"simulation sizes: pilot {n_pilot} cycles, final {n_final} cycles, burn-in {N_BURN}, {N_BATCH} batches; "
          f"dwell grid m/{GRID_SIM}; 60-day runs: {NPHASE} phases, {2 * NREAL} schedules in {NREAL} mirrored pairs, switch times on a grid of 1/{GRID_RUN} day")
    print(f"a = {A_HOST:g} days in hosts; outside stay b = {B_OUT}; generation T = a + b; ln R0 for rates and runs = {Y_STATED}")
    print()

    # ---------------- launch the heavy part on two workers; the main process takes a third share
    pairs = [(bi, ki) for ki in range(6) for bi in range(3)] if only is None else only
    tasks = [(bi, ki, n_pilot, n_final) for bi, ki in pairs]
    lanes = [[], [], []]
    for t in tasks:
        lanes[(t[0] + t[1]) % 3].append(t)
    lanes[2].append(("v2d", n_pilot, max(n_final, 40000)))
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(2)
    pending = [pool.apply_async(run_lane, (lanes[1],)), pool.apply_async(run_lane, (lanes[2],))]

    # ---------------- validation checks that need no simulation (main process, before its lane)
    print("=" * 110)
    print("VALIDATION (tolerances fixed in the protocol, section 5)")
    base = (1, 1, 1.0, "prop")
    # V1
    rows, worst = [], 0.0
    for b, mult, printed in ((2.0, 2, 0.1511), (10.0, 1, 0.0634), (50.0, 8, 0.0619)):
        T = A_HOST + b
        Pd = max(2, int(round(mult * T / 2)) * 2)
        Dd = step_D(base, b, 0.8, Pd // 2, 1.0)
        worst = max(worst, abs(Dd - printed))
        rows.append(f"b={b:g}, period {Pd} d: {Dd:.6f} (printed {printed:.4f})")
    report("V1 daily steps reproduce three numbers of the first design", worst < 5e-5, "; ".join(rows) + f"; largest difference {worst:.1e}")
    r_old = step_rate(base, 2.0, 0.8, 7, 1.0, 0.5)
    print(f"      also: growth per day in switched hosts, b = 2, period 14 d, ln R0 = 0.5: {r_old:+.4f} (first design printed +0.0498)")

    # V2a
    rows, worst = [], 0.0
    v2a = [((1, 1), 10.0, 0.5), ((1, 1), 2.0, 2.0), ((1, 1), 50.0, 4.0), ((4, 4), 10.0, 2.0), ((16, 16), 10.0, 1.0), ((16, 1), 2.0, 4.0)]
    for (mb, ma), b, k in v2a:
        var = (mb, ma, 1.0, "prop")
        P = k * (A_HOST + b)
        ex = floquet_D(var, b, 0.05, P)
        ex2 = floquet_D(var, b, 0.025, P)
        d2 = d2_square_gain(var, b, 0.05, P)
        rel = abs(ex / d2 - 1)
        ok = rel < 0.02 or abs(ex - d2) < 2e-7
        worst = max(worst, rel if not abs(ex - d2) < 2e-7 else 0.0)
        rows.append(f"      shape ({mb},{ma}) b={b:g} P={k:g}T: D {ex:+.4e}, Result 7 {d2:+.4e}, rel. diff. {rel:.5f} (at half the swing {abs(ex2 / (d2 / 4) - 1):.5f})" + ("" if ok else "  <-- outside tolerance"))
        if not ok:
            worst = max(worst, 1.0)
    report("V2a gain-only, square wave, swing 0.05: D equals Result 7 summed over harmonics", worst < 0.02, f"largest relative difference {worst:.5f} over 6 designs")
    print("\n".join(rows))

    # V2b, V2c
    rows, worst = [], 0.0
    for b in B_OUT:
        for k in (0.5, 2.0):
            m = k * (A_HOST + b)
            ex = tele2d_D(base, b, 0.05, m)
            d2 = d2_random(base, b, 0.05, m)
            worst = max(worst, abs(ex / d2 - 1))
            rows.append(f"b={b:g} m={k:g}T: {ex:.4e} vs {d2:.4e}")
    report("V2b gain-only, telegraph, swing 0.05: D (quadrature) equals Result 1 summed for the ring", worst < 0.02, f"largest relative difference {worst:.5f}; " + "; ".join(rows))
    rows, worst = [], 0.0
    for d in (0.1, 0.5, 0.9):
        for b in B_OUT:
            for k in (0.5, 2.0):
                var = (1, 1, d, "prop")
                m = k * (A_HOST + b)
                ex = tele2d_D(var, b, 0.05, m)
                d2 = d2_random(var, b, 0.05, m)
                worst = max(worst, abs(ex / d2 - 1))
                rows.append(f"d={d:g} b={b:g} m={k:g}T: {ex:.4e} vs {d2:.4e}")
    report("V2c contact-limited, telegraph, swing 0.05: D (quadrature) equals (C2)", worst < 0.02, f"largest relative difference {worst:.5f} over 18 designs; e.g. " + "; ".join(rows[6:9]))

    # V2e (reported)
    rows = []
    for (mb, ma), d, b, k in (((1, 1), 0.5, 10.0, 2.0), ((1, 1), 0.1, 2.0, 1.0), ((4, 4), 0.5, 10.0, 2.0), ((16, 16), 0.5, 10.0, 1.0)):
        var = (mb, ma, d, "prop")
        P = k * (A_HOST + b)
        ex = floquet_D(var, b, 0.05, P)
        d2 = d2_square_matrix(var, b, 0.05, P)
        rows.append(f"shape ({mb},{ma}) d={d:g} b={b:g} P={k:g}T: D {ex:+.4e}, matrix second order {d2:+.4e}")
    print("      reported, not a gate (V2e): contact-limited under a SQUARE wave against the second-order perturbation of the Floquet exponent")
    print("        (derived for this check, not refereed): " + "; ".join(rows))

    # V4
    rows, ok_all = [], True
    for name, var in (("waste", base), ("wait d=0.5", (1, 1, 0.5, "prop"))):
        Dc = floquet_D(var, 10.0, 0.8, 30.0)
        Ds = {}
        for p in range(0, 7):
            dt = 1.0 / 2 ** p
            Ds[p] = step_D(var, 10.0, 0.8, int(round(15.0 / dt)), dt)
        e64 = abs(Ds[6] / Dc - 1)
        ex = abs((2 * Ds[6] - Ds[5]) / Dc - 1)
        ok_all = ok_all and e64 < 0.02 and ex < 0.002
        rows.append(f"      {name}: continuous {Dc:.6f}; steps of 1, 1/2, ..., 1/64 day: " + ", ".join(f"{Ds[p]:.6f}" for p in range(7))
                    + f"; step 1/64 off by {e64:.4%}; extrapolated off by {ex:.5%}; the daily model is off by {Ds[0] / Dc - 1:+.2%}")
    report("V4 continuous time against steps, two cases (b = 10, swing 1.8/0.2, period 30 d)", ok_all, "step 1/64 within 2% and the extrapolation within 0.2% in both")
    print("\n".join(rows))

    # V6a, V6d, V7 (deterministic parts)
    worst = max(abs(floquet_D(v, b, 0.0, 10.0)) for v, b in (((1, 1, 1.0, "prop"), 2.0), ((4, 4, 0.5, "prop"), 10.0), ((16, 16, 1.0, "sat"), 50.0),
                                                              ((16, 1, 0.1, "sat"), 2.0), ((1, 16, 0.9, "prop"), 10.0), ((4, 16, 0.99, "prop"), 50.0)))
    report("V6a no fluctuation, no loss", worst < 1e-10, f"largest |D| over six variants {worst:.1e}")
    worst = 0.0
    for var, b in (((1, 1, 1.0, "prop"), 2.0), ((16, 16, 0.5, "prop"), 10.0), ((4, 4, 0.9, "sat"), 50.0)):
        for P in (1.5, 7.5):                           # 60 days = 40 periods, or 8
            Ah, Al = levels_tel(var, b, 0.8, Y_STATED)
            M = linalg.expm(Al * (P / 2)) @ linalg.expm(Ah * (P / 2))
            w, V = np.linalg.eig(M)
            x0 = np.abs(V[:, np.argmax(abs(w))].real)
            x0 /= x0.sum()
            sq = run_square(var, b, 0.8, P, Y_STATED, x0)      # phase 0 starts a period
            rf = floquet_rate(var, b, 0.8, P, Y_STATED)
            worst = max(worst, abs(math.log(sq[0, 1, 0]) / 60.0 / rf - 1), abs(math.log(sq[0, 0, 0]) / 30.0 / rf - 1))
    report("V6d the 60-day run routine started on the Floquet eigenvector grows at the Floquet rate", worst < 1e-9, f"largest relative difference {worst:.1e} (three cells; 40 periods of 1.5 days and 8 periods of 7.5 days)")
    worst = 0.0
    for (mb, ma) in ((1, 1), (16, 16), (4, 1)):
        for b in B_OUT:
            for k in (0.5, 2.0):
                worst = max(worst, abs(floquet_D((mb, ma, 0.0, "prop"), b, 0.8, k * (A_HOST + b))))
    for b in B_OUT:
        worst = max(worst, abs(tele2d_D((1, 1, 0.0, "prop"), b, 0.8, 2.0 * (A_HOST + b))))
    v7det = worst

    # ---------------- the main process's lane, then collect
    results = run_lane(lanes[0])
    for p in pending:
        results += p.get()
    pool.close()
    pool.join()
    v2d = [r for r in results if "bi" not in r][0]
    results = [r for r in results if "bi" in r]
    nv = len(VARS)
    shp = (2, nv, 3, 2, 6)
    D = np.full(shp, np.nan)
    Dse = np.zeros(shp)
    r05 = np.full(shp, np.nan)
    r05se = np.zeros(shp)
    rc = np.full((nv, 3), np.nan)
    L = np.full((2, nv, 3, 2, 6, 2, 2, NSTAT), np.nan) # schedule, variant, b, delta, k, day, (total, hosts), stat
    LK = [np.full((nv, 3, 2, 6, 2, 2, NPHASE), np.nan, dtype=np.float32), np.full((nv, 3, 2, 6, 2, 2, 2 * NREAL), np.nan, dtype=np.float32)]
    Dsim = np.full((nv, 3, 2, 6), np.nan)
    Dsimse = np.full((nv, 3, 2, 6), np.nan)
    D3 = np.full((len(V3_CELLS3), 3, 2, 6), np.nan)
    D3se = np.full((len(V3_CELLS3), 3, 2, 6), np.nan)
    sens = np.full((len(SENS_VARS), 3, 6, 2, 2, 2, NSTAT), np.nan)
    lerr = v2d["lerr"]
    quad = QUAD_MASS_ERR[0]
    nrer = nfail = 0
    secs = []
    for r in results:
        bi, ki = r["bi"], r["ki"]
        D[:, :, bi, :, ki] = r["D"]
        Dse[:, :, bi, :, ki] = r["Dse"]
        r05[:, :, bi, :, ki] = r["r05"]
        r05se[:, :, bi, :, ki] = r["r05se"]
        rc[:, bi] = r["rc"]
        L[:, :, bi, :, ki] = r["L"]
        LK[0][:, bi, :, ki] = r["Lks"]
        LK[1][:, bi, :, ki] = r["Lkt"]
        Dsim[:, bi, :, ki] = r["Dsim"]
        Dsimse[:, bi, :, ki] = r["Dsimse"]
        D3[:, bi, :, ki] = r["D3"]
        D3se[:, bi, :, ki] = r["D3se"]
        sens[:, bi, ki] = r["sens"]
        lerr = max(lerr, r["lerr"])
        quad = max(quad, r["quad"])
        nrer += r["nrerun"]
        nfail += r["nfail"]
        secs.append((bi, ki, r["secs"]))
    have = ~np.isnan(D[0, 0, :, 0, :])                 # which (b, k) were run
    full = bool(have.all())
    VI = {v: i for i, v in enumerate(VARS)}

    # V2d
    rows, ok_all = [], True
    for j, (name, var) in enumerate((("gain-only", (4, 4, 1.0, "prop")), ("contact-limited d=0.5", (4, 4, 0.5, "prop")))):
        d2 = d2_random(var, 10.0, 0.2, 30.0)
        ok = abs(v2d["D"][j] - d2) <= 4 * v2d["se"][j] + 0.06 * abs(d2)
        ok_all = ok_all and ok
        rows.append(f"{name}: simulated {v2d['D'][j]:.6f} +- {v2d['se'][j]:.6f}, second order {d2:.6f}")
    report("V2d chains (4,4) under the telegraph, swing 0.2, b = 10, m = 2T, by simulation", ok_all, "; ".join(rows) + " (4 standard errors + 6%)")

    # V3
    bad = 0
    ncell = 0
    closest = (9.0, "")
    for iv, var in enumerate(VARS):
        if var[2] != 1.0 or var[3] != "prop":
            continue
        for bi in range(3):
            for di, delta in enumerate(DELTAS):
                ell = -0.5 * (math.log(1 + delta) + math.log(1 - delta))
                for ki in range(6):
                    if not have[bi, ki]:
                        continue
                    for s in range(2):
                        ncell += 1
                        tol = 1e-9 + 4 * Dse[s, iv, bi, di, ki]
                        if D[s, iv, bi, di, ki] > ell + tol:
                            bad += 1
                        frac = D[s, iv, bi, di, ki] / ell
                        if 1 - frac < closest[0]:
                            closest = (1 - frac, f"{'square' if s == 0 else 'telegraph'}, {vlabel(var)}, b={B_OUT[bi]:g}, swing {delta}, {KLAB[ki]} gen: D = {D[s, iv, bi, di, ki]:.4f} of {ell:.4f}")
    bad3 = n3 = 0
    for j, var in enumerate(V3_CELLS3):
        if var[2] != 1.0:
            continue
        for di, delta in enumerate(DELTAS):
            d3 = delta * math.sqrt(1.5)
            ell3 = -(math.log(1 + d3) + math.log(1 - d3)) / 3.0
            for bi in range(3):
                for ki in range(6):
                    if have[bi, ki]:
                        n3 += 1
                        bad3 += D3[j, bi, di, ki] > ell3 + 4 * D3se[j, bi, di, ki]
    report("V3 the bound D <= -(1/2)[ln(1+delta) + ln(1-delta)] for every gain-only variant with fixed staying", bad == 0 and bad3 == 0,
           f"violations {bad} of {ncell} cells (square wave and telegraph), {bad3} of {n3} three-level cells; nearest the bound: {closest[1]}")
    # reported: the bound for the saturating function
    bads = ns = 0
    for iv, var in enumerate(VARS):
        if var[2] != 1.0 or var[3] != "sat":
            continue
        for di, delta in enumerate(DELTAS):
            ell = -0.5 * (math.log(f_enc(1 + delta, "sat")) + math.log(f_enc(1 - delta, "sat")))
            for bi in range(3):
                for ki in range(6):
                    if have[bi, ki]:
                        for s in range(2):
                            ns += 1
                            bads += D[s, iv, bi, di, ki] > ell + 1e-9 + 4 * Dse[s, iv, bi, di, ki]
    print(f"      reported, not a gate: waste with the saturating function, D <= -E ln f(h): violations {bads} of {ns} cells")

    # V5
    rows, ok_all = [], True
    for d in (1.0, 0.5):
        iv = VI[(1, 1, d, "prop")]
        for bi in range(3):
            for ki in (1, 4):
                if not have[bi, ki]:
                    continue
                dq, ds, se = D[1, iv, bi, 0, ki], Dsim[iv, bi, 0, ki], Dsimse[iv, bi, 0, ki]
                ok = abs(dq - ds) <= 4 * se + 0.0005
                ok_all = ok_all and ok
                rows.append(f"d={d:g} b={B_OUT[bi]:g} m={KLAB[ki]}T: {dq:.5f} vs {ds:.5f}+-{se:.5f}")
    two = [iv for iv, v in enumerate(VARS) if v[0] == 1 and v[1] == 1]
    zz = (Dsim[two] - D[1, two]) / np.where(Dsimse[two] > 0, Dsimse[two], np.nan)
    report("V5 simulation against quadrature, telegraph, memoryless stays, swing 1.8/0.2 (12 cells)", ok_all and len(rows) > 0, "; ".join(rows[:4]) + " ...")
    print(f"      all 12: " + "; ".join(rows))
    print(f"      reported: over all {int(np.isfinite(zz).sum())} two-stage telegraph cells, largest |simulated - quadrature| = {np.nanmax(np.abs(Dsim[two] - D[1, two])):.5f}, "
          f"largest |difference|/standard error = {np.nanmax(np.abs(zz)):.2f}, root mean square {math.sqrt(np.nanmean(zz ** 2)):.2f}")

    report("V6b products of precomputed matrix exponentials equal the matrix exponential (every simulated batch)", lerr < 1e-8, f"largest relative difference {lerr:.1e}")
    report("V6c quadrature: the two levels carry equal mass", quad < 1e-6, f"largest relative difference {quad:.1e}")
    report("V7 nobody dies waiting (d = 0): no loss at any amplitude", v7det < 1e-8 and abs(v2d["v7"]) < 1e-6,
           f"largest |D| {v7det:.1e} (square wave: 18 cells; quadrature: 3 cells); simulated chain (4,4), swing 1.8/0.2: growth at ln R0 = 0, per generation, {v2d['v7']:.1e}")
    semax = np.nanmax(Dse[1])
    target = np.maximum(0.003, 0.03 * np.abs(D[1]))
    miss = int(np.nansum(Dse[1] > target))
    print(f"      simulation precision: largest standard error of D {semax:.5f}; cells missing the target (larger of 0.003 and 3% of D): {miss}; "
          f"cells rerun because the root left the bracket: {nrer}; cells left without a root: {nfail}")
    print()

    # ---------------- tables
    def row6(x, fmt):
        return " ".join(fmt.format(v) if np.isfinite(v) else "    n/a" for v in x)

    print("=" * 110)
    print("TABLE 0.  Constants")
    for di, delta in enumerate(DELTAS):
        ell = -0.5 * (math.log(1 + delta) + math.log(1 - delta))
        fb = 0.5 * (f_enc(1 + delta, "sat") + f_enc(1 - delta, "sat"))
        print(f"  swing {1 + delta:.1f}/{1 - delta:.1f}: bound for waste, proportional {ell:.4f}; saturating function: mean rate {fb:.4f} of the rate at the mean density; "
              "curvature part D_curv = -ln[fbar/(d + (1-d) fbar)]: " + ", ".join(f"d={d:g}: {-math.log(fb / (d + (1 - d) * fb)):.4f}" for d in D_SET))
    print(f"  measurement: coefficient of variation 20% on a count -> sd of ln(count) {SIGMA_M:.4f}; replicates per treatment n = ceil(32 x {SIGMA_M2:.5f} / Delta^2) = ceil({32 * SIGMA_M2:.4f}/Delta^2)")
    print(f"  growth per day of the constant treatment at ln R0 = {Y_STATED} (memoryless stays): " + ", ".join(f"b={B_OUT[bi]:g}: {rc[VI[base], bi]:+.4f}" for bi in range(3))
          + "; (16,16): " + ", ".join(f"b={B_OUT[bi]:g}: {rc[VI[(16, 16, 1.0, 'prop')], bi]:+.4f}" for bi in range(3)))
    print()

    print("=" * 110)
    print("TABLE 0b.  The first design's model (daily steps) against continuous time.  D for waste, memoryless stays, proportional, square wave,")
    print("  the period rounded to an even number of days as in the first design; columns 1/4, 1/2, 1, 2, 4, 8 generations")
    for delta in DELTAS:
        for b in B_OUT:
            Ps = [max(2, int(round(k * (A_HOST + b) / 2)) * 2) for k in KS]
            dd = np.array([step_D(base, b, delta, Pd // 2, 1.0) for Pd in Ps])
            dc = np.array([floquet_D(base, b, delta, float(Pd)) for Pd in Ps])
            print(f"  swing {1 + delta:.1f}/{1 - delta:.1f} b={b:<2g} periods {'/'.join(str(x) for x in Ps)} d | daily steps " + row6(dd, "{:.4f}") + " | continuous " + row6(dc, "{:.4f}")
                  + " | daily/continuous " + row6(dd / dc, "{:.2f}"))
    print()

    print("=" * 110)
    print("TABLE 1.  Loss D of ln R0, and r_const - r_fluct per day at ln R0 = 0.5 (x1000), every variant.")
    print("  Columns: time scale 1/4, 1/2, 1, 2, 4, 8 generations (square wave: period; telegraph: mean dwell).  sq = square wave (Floquet, computed);")
    print("  te = telegraph (two-stage variants: quadrature, computed; chains: simulated, largest standard error of D in the row at the end).")
    for s, sname in enumerate(("sq", "te")):
        for di, delta in enumerate(DELTAS):
            for bi, b in enumerate(B_OUT):
                print(f"  -- {sname}, swing {1 + delta:.1f}/{1 - delta:.1f}, outside stay b = {b:g} d (T = {A_HOST + b:g} d)")
                for iv, var in enumerate(VARS):
                    dr = (rc[iv, bi] - r05[s, iv, bi, di]) * 1000
                    tail = f" | se {Dse[s, iv, bi, di].max():.4f}" if s == 1 and var[0] + var[1] > 2 else ""
                    print(f"  {sname} {1 + delta:.1f} b={b:<2g} {vlabel(var)} | D " + row6(D[s, iv, bi, di], "{:+7.4f}") + " | 1000dr " + row6(dr, "{:+7.2f}") + tail)
    print()

    print("=" * 110)
    print("TABLE 2.  What an experimenter would see: N_const/N_fluct (all parasites) after 30 and 60 days, from actual runs.")
    print("  Start: the stable stage composition of the constant treatment at ln R0 = 0.5, split into the treatments at day 0.")
    print(f"  sq: geometric mean over 16 starting phases [smallest, largest].  te: geometric mean over {2 * NREAL} schedules [10th, 90th percentile];")
    print("  at the end of each telegraph row, the largest standard error of the mean of ln(ratio) in the row (from the mirrored pairs).")
    print("  Columns: time scale 1/4, 1/2, 1, 2, 4, 8 generations.")

    def cell(st):
        return f"{math.exp(st[0]):5.2f}[{math.exp(st[1]):4.2f},{math.exp(st[2]):5.2f}]"

    for s, sname in enumerate(("sq", "te")):
        for di, delta in enumerate(DELTAS):
            for bi, b in enumerate(B_OUT):
                print(f"  -- {sname}, swing {1 + delta:.1f}/{1 - delta:.1f}, outside stay b = {b:g} d")
                for iv, var in enumerate(VARS):
                    for day in range(2):
                        print(f"  {sname} {1 + delta:.1f} b={b:<2g} {vlabel(var)} day {int(DAYS[day])} | "
                              + " ".join(cell(L[s, iv, bi, di, ki, day, 0]) if have[bi, ki] else "n/a" for ki in range(6))
                              + (f" | se {np.nanmax(L[s, iv, bi, di, :, day, 0, 4]):.3f}" if s == 1 else ""))
    print()

    print("=" * 110)
    print("TABLE 3.  Compact: swing 1.8/0.2, ten variants.  D, and N_const/N_fluct at day 60 (geometric mean).")
    compact = [(1, 1, 1.0, "prop"), (1, 1, 0.9, "prop"), (1, 1, 0.5, "prop"), (1, 1, 0.1, "prop"), (16, 16, 1.0, "prop"), (16, 1, 1.0, "prop"),
               (1, 16, 1.0, "prop"), (16, 16, 0.5, "prop"), (1, 1, 1.0, "sat"), (1, 1, 0.5, "sat")]
    for s, sname in enumerate(("square wave (period)", "telegraph (mean dwell)")):
        for bi, b in enumerate(B_OUT):
            print(f"  -- {sname}, outside stay b = {b:g} d; columns 1/4, 1/2, 1, 2, 4, 8 generations"
                  + (f"; standard error of ln(ratio), largest over the ten variants, by column: day 60 " + row6(np.nanmax(L[1, [VI[v] for v in compact], bi, 0, :, 1, 0, 4], axis=0), "{:.3f}")
                     + "; day 30 " + row6(np.nanmax(L[1, [VI[v] for v in compact], bi, 0, :, 0, 0, 4], axis=0), "{:.3f}") if s == 1 else ""))
            for var in compact:
                iv = VI[var]
                print(f"  {vlabel(var)} | D " + row6(D[s, iv, bi, 0], "{:+6.3f}") + " | ratio day 60 " + row6(np.exp(L[s, iv, bi, 0, :, 1, 0, 0]), "{:5.2f}")
                      + " | ratio day 30 " + row6(np.exp(L[s, iv, bi, 0, :, 0, 0, 0]), "{:5.2f}"))
    print("  the same, counting only the parasites in hosts, day 60:")
    for s, sname in enumerate(("square wave", "telegraph")):
        for bi, b in enumerate(B_OUT):
            for var in compact[:5]:
                iv = VI[var]
                print(f"  {sname:<11s} b={b:<2g} {vlabel(var)} | in hosts, day 60 " + row6(np.exp(L[s, iv, bi, 0, :, 1, 1, 0]), "{:5.2f}"))
    print()

    print("=" * 110)
    print("TABLE 4.  D against d, the fraction that dies waiting (memoryless stays, proportional, swing 1.8/0.2; computed).")
    for s, sname in enumerate(("square wave, period", "telegraph, mean dwell")):
        for bi, b in enumerate(B_OUT):
            for ki in (2, 3, 5):
                Tt = KS[ki] * (A_HOST + b)
                vals = [floquet_D((1, 1, d, "prop"), b, 0.8, Tt) if s == 0 else tele2d_D((1, 1, d, "prop"), b, 0.8, Tt) for d in D_FINE]
                print(f"  {sname} {KLAB[ki]} gen, b={b:<2g} | d = " + " ".join(f"{d:g}" for d in D_FINE) + " | D " + " ".join(f"{v:.4f}" for v in vals))
    print()

    print("=" * 110)
    print("TABLE 5.  Three-level random schedule (same mean, variance and autocorrelation as the telegraph) against the telegraph: D, simulated.")
    print("  columns 1/4 ... 8 generations; each entry three-level D (telegraph D); standard errors of the three-level value at the end")
    for j, var in enumerate(V3_CELLS3):
        iv = VI[var]
        for di, delta in enumerate(DELTAS):
            for bi, b in enumerate(B_OUT):
                print(f"  {vlabel(var)} swing {1 + delta:.1f}/{1 - delta:.1f} b={b:<2g} | " + " ".join(f"{D3[j, bi, di, ki]:+.4f} ({D[1, iv, bi, di, ki]:+.4f})" for ki in range(6))
                      + f" | se <= {np.nanmax(D3se[j, bi, di]):.4f}")
    rel = []
    for j, var in enumerate(V3_CELLS3):
        iv = VI[var]
        rel.append(D3[j] / D[1, iv] - 1)
    rel = np.array(rel)
    print(f"  three-level D relative to the telegraph D: median {np.nanmedian(rel):+.3f}, smallest {np.nanmin(rel):+.3f}, largest {np.nanmax(rel):+.3f} over {int(np.isfinite(rel).sum())} cells;"
          f" swing 1.5/0.5 only: smallest {np.nanmin(rel[:, :, 1]):+.3f}, largest {np.nanmax(rel[:, :, 1]):+.3f}")
    print("  as a fraction of each schedule's own bound -E ln h (waste, nearly fixed stays (16,16)); three-level levels are "
          + ", ".join(f"{1 - dl * math.sqrt(1.5):.3f}/1/{1 + dl * math.sqrt(1.5):.3f}" for dl in DELTAS) + ":")
    j16 = V3_CELLS3.index((16, 16, 1.0, "prop"))
    for di, delta in enumerate(DELTAS):
        d3 = delta * math.sqrt(1.5)
        ell3 = -(math.log(1 + d3) + math.log(1 - d3)) / 3.0
        ell2 = -0.5 * (math.log(1 + delta) + math.log(1 - delta))
        for bi, b in enumerate(B_OUT):
            print(f"      swing {1 + delta:.1f}/{1 - delta:.1f} b={b:<2g} | three-level (bound {ell3:.4f}) " + row6(D3[j16, bi, di] / ell3, "{:.2f}")
                  + f" | telegraph (bound {ell2:.4f}) " + row6(D[1, VI[(16, 16, 1.0, 'prop')], bi, di] / ell2, "{:.2f}"))
    print()

    print("=" * 110)
    print("TABLE 6.  Sensitivity of the 60-day ratio (swing 1.8/0.2): a higher R0, and a start with every parasite in the first sub-stage outside.")
    print("  entries: geometric mean of N_const/N_fluct at day 60, columns 1/4 ... 8 generations; 'base' is Table 2")
    for j, var in enumerate(SENS_VARS):
        iv = VI[var]
        for bi, b in enumerate(B_OUT):
            for s, sname in enumerate(("sq", "te")):
                print(f"  {sname} b={b:<2g} {vlabel(var)} | base " + row6(np.exp(L[s, iv, bi, 0, :, 1, 0, 0]), "{:5.2f}")
                      + " | ln R0 = 1.5 " + row6(np.exp(sens[j, bi, :, 0, s, 1, 0]), "{:5.2f}") + " | start outside " + row6(np.exp(sens[j, bi, :, 1, s, 1, 0]), "{:5.2f}"))
    print()

    # ---------------- open questions
    print("=" * 110)
    print("OPEN QUESTIONS (protocol, section 6)")
    for s, sname in enumerate(("square wave", "telegraph")):
        Ds = D[s]
        nneg = int(np.sum(Ds < -4 * Dse[s] - 1e-9))
        neg = np.argwhere(Ds < -4 * Dse[s] - 0.001)
        print(f"  O1(i)/O2 {sname}: cells with D < 0 (beyond 4 standard errors for simulated values): {nneg} of {int(np.isfinite(Ds).sum())}; smallest D {np.nanmin(Ds):+.4f}; "
              f"cells with D < -0.001: {len(neg)}, listed")
        if len(neg):
            byvar = {}
            for iv, bi, di, ki in neg:
                byvar.setdefault(iv, []).append((bi, di, ki))
            for iv, lst in sorted(byvar.items()):
                print(f"      {vlabel(VARS[iv])}: " + ", ".join(f"b={B_OUT[bi]:g} {1 + DELTAS[di]:.1f} {KLAB[ki]}gen D={Ds[iv, bi, di, ki]:+.4f}" for bi, di, ki in lst))
        if full:
            mono = np.all(np.diff(Ds, axis=3) > -4 * (Dse[s][..., 1:] + Dse[s][..., :-1]) - 1e-9, axis=3)
            print(f"  O1(ii) {sname}: D rises with the time scale at every step in {int(mono.sum())} of {mono.size} (variant, b, swing) rows")
            for resp in RESP:
                for d in D_SET:
                    sel = [iv for iv, v in enumerate(VARS) if v[2] == d and v[3] == resp]
                    print(f"      {'waste' if d == 1 else 'wait d=%g' % d:<12s} {resp}: {int(mono[sel].sum())} of {mono[sel].size}"
                          + f"; 8 generations above 1/4 generation in {int((Ds[sel][..., 5] > Ds[sel][..., 0]).sum())} of {mono[sel].size}")
            prot = Ds[:, 2, :, 2:] < Ds[:, 0, :, 2:]
            print(f"  O1(iii) {sname}: at 1 generation or more, D with a 50-day outside stage is below D with a 2-day one in {int(prot.sum())} of {prot.size} (variant, swing, time scale) cells")
            for resp in RESP:
                for d in D_SET:
                    sel = [iv for iv, v in enumerate(VARS) if v[2] == d and v[3] == resp]
                    print(f"      {'waste' if d == 1 else 'wait d=%g' % d:<12s} {resp}: {int(prot[sel].sum())} of {prot[sel].size}; median D(b=50)/D(b=2) at 2 generations, swing 1.8/0.2: {np.median(Ds[sel, 2, 0, 3] / Ds[sel, 0, 0, 3]):.2f}")
    if full:
        print("  O3 range of D across the 90 variants, swing 1.8/0.2 (smallest, median, largest), columns 1/4 ... 8 generations")
        for s, sname in enumerate(("square wave", "telegraph")):
            for bi, b in enumerate(B_OUT):
                print(f"      {sname:<11s} b={b:<2g} | min " + row6(np.nanmin(D[s, :, bi, 0], axis=0), "{:+6.3f}") + " | median " + row6(np.nanmedian(D[s, :, bi, 0], axis=0), "{:+6.3f}")
                      + " | max " + row6(np.nanmax(D[s, :, bi, 0], axis=0), "{:+6.3f}"))
        print("  O3 the same for the geometric-mean ratio N_const/N_fluct at day 60")
        for s, sname in enumerate(("square wave", "telegraph")):
            for bi, b in enumerate(B_OUT):
                R = np.exp(L[s, :, bi, 0, :, 1, 0, 0])
                print(f"      {sname:<11s} b={b:<2g} | min " + row6(R.min(axis=0), "{:5.2f}") + " | median " + row6(np.median(R, axis=0), "{:5.2f}") + " | max " + row6(R.max(axis=0), "{:5.2f}"))
        print("  O3 D for nearly fixed stays (16,16), waste, proportional, as a fraction of the bound, telegraph, swing 1.8/0.2: "
              + "; ".join(f"b={B_OUT[bi]:g}: " + row6(D[1, VI[(16, 16, 1.0, 'prop')], bi, 0] / 0.5108256, "{:.2f}") for bi in range(3)))
    print()

    # ---------------- the design question
    if full:
        print("=" * 110)
        print("THE DESIGN QUESTION (protocol, section 7, amendments 1 and 2).  L = ln(N_const/N_fluct) for one phase or schedule; a treatment's outcome")
        print("  is the mean of ln(count) over its replicates.  For two alternatives A and B and two treatments T1, T2: contrast = L(T2) - L(T1);")
        print("  Delta = contrast under A minus contrast under B.  Replicates per treatment for the two predictions to be 4 standard errors apart:")
        print(f"  n   measurement error only (sd of ln count {SIGMA_M:.3f}).  Valid when the phases or schedules are balanced by design and recorded, and")
        print("      the prediction is made for them: a square wave with replicates spread evenly over its phases, or one drawn set of schedules shared by all.")
        print("  nm  replicates in mirrored pairs (a phase and the phase half a period later; a schedule and the same switching times with high and low")
        print("      exchanged), pairs drawn independently, their spread treated as error.")
        print("  n*  replicates with independent phases or schedules, not paired, their spread treated as error.")
        print("  ns  every replicate compared with the prediction for its own phase or schedule (uses Delta^2 + sdD^2); for (a) to (c) only: against")
        print("      'no effect' it would only show that growth follows the level of density.")
        print("  sdD = sd over phases or schedules of the difference between the two alternatives.")
        print("  'any' = any schedule of the grid (the protocol's search).  'adm' = admissible schedules only: square wave with at least two periods in")
        print("  60 days (period <= 30 d), telegraph with mean dwell <= 15 d.  A schedule that hardly switches within the run compares constant densities.")
        print("  'held' = densities held constant for the whole run at 1.8, 0.2, 1.5 or 0.5 times the mean (and the mean itself): not variability")
        print("  treatments; they measure how growth responds to the level of density.")
        CNT = ("all parasites", "in hosts")

        def admissible(bi, t):
            sch, di, ki = t
            return KS[ki] * (A_HOST + B_OUT[bi]) <= (30.0 if sch == 0 else 15.0) + 1e-9

        for bi in range(3):
            print(f"  admissible for b = {B_OUT[bi]:g}: square periods " + ", ".join(KLAB[ki] for ki in range(6) if admissible(bi, (0, 0, ki))) + " gen; telegraph dwells "
                  + (", ".join(KLAB[ki] for ki in range(6) if admissible(bi, (1, 0, ki))) or "none") + " gen")

        # densities held constant at one of the levels, from the same start: ln(N at the mean density / N at the held density)
        LH = np.empty((nv, 3, len(HELD), 2, 2))
        for iv, var in enumerate(VARS):
            for bi, b in enumerate(B_OUT):
                x0 = start_vector(var, b, Y_STATED, "stable")
                cc = const_counts(var, b, Y_STATED, x0)
                for hi, h in enumerate(HELD):
                    Ahd = generator(var, b, f_enc(h, var[3]), Y_STATED)
                    for day in range(2):
                        x = linalg.expm(Ahd * DAYS[day]) @ x0
                        LH[iv, bi, hi, day] = np.log(cc[day] / np.array([x.sum(), x[:var[1]].sum()]))
        HELD_T = [("held", hi) for hi in range(len(HELD))]

        def lk(iv, bi, t, day, c):
            if t[0] == "held":
                return np.array([LH[iv, bi, t[1], day, c]] * 2)
            sch, di, ki = t
            return LK[sch][iv, bi, di, ki, day, c].astype(float)

        def pairsd(x):
            h = len(x) // 2
            return (0.5 * (x[:h] + x[h:])).std()

        def dstat(ivA, ivB, bi, t, day, c):
            """(mean L under A, mean L under B, sd of L_A - L_B, larger sd of L, larger sd of pair means) over phases or schedules;
            ivB None = 'no effect' (L = 0)."""
            if t is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            xa = lk(ivA, bi, t, day, c)
            if ivB is None:
                return xa.mean(), 0.0, xa.std(), xa.std(), pairsd(xa)
            xb = lk(ivB, bi, t, day, c)
            return xa.mean(), xb.mean(), (xa - xb).std(), max(xa.std(), xb.std()), max(pairsd(xa), pairsd(xb))

        def search(ivA, ivB, bi, c, adm, crit, with_const_only=False):
            best = None
            for day in range(2):
                cand = [None] + (HELD_T if adm == "held" else [t for t in TREATS if (not adm or admissible(bi, t))])
                st = [dstat(ivA, ivB, bi, t, day, c) for t in cand]
                for i in range(len(cand)):
                    for j in range(i + 1, len(cand)):
                        if with_const_only and i != 0:
                            continue
                        delta = (st[j][0] - st[j][1]) - (st[i][0] - st[i][1])
                        extra = st[i][3] ** 2 + st[j][3] ** 2
                        extram = 2.0 * (st[i][4] ** 2 + st[j][4] ** 2)
                        score = abs(delta) if crit == "delta" else delta ** 2 / (2 * SIGMA_M2 + extram)
                        if best is None or score > best[0]:
                            i1, i2 = (i, j) if delta >= 0 else (j, i)
                            best = (score, day, cand[i1], cand[i2], abs(delta), st[i2][0] - st[i1][0], st[i2][1] - st[i1][1], math.hypot(st[i][2], st[j][2]), extra, extram)
            return best

        def show(tag, ivA, ivB, bi, c, adm, crit):
            r = search(ivA, ivB, bi, c, adm, crit)
            kind = {False: "any ", True: "adm ", "held": "held"}[adm]
            cname = "max Delta" if crit == "delta" else "min nm   "
            if r is None or r[4] < 1e-12:
                print(f"  {tag} b={B_OUT[bi]:<2g} {CNT[c]:<13s} {kind} {cname}: none")
                return
            _, day, t1, t2, delta, ca, cb, sdd, extra, extram = r
            ns = "" if ivB is None else f"ns = {nfmt(nrep(math.hypot(delta, sdd)))}, "
            print(f"  {tag} b={B_OUT[bi]:<2g} {CNT[c]:<13s} {kind} {cname}: day {int(DAYS[day])}, [{tlabel(t1)}] vs [{tlabel(t2)}]: "
                  f"contrast A {ca:+.3f}, B {cb:+.3f}, Delta {delta:.3f}, sdD {sdd:.3f}, n = {nfmt(nrep(delta))}, {ns}nm = {nfmt(nrep(delta, extram))}, n* = {nfmt(nrep(delta, extra))}")

        def block(tag, ivA, ivB, held=True):
            for bi in range(3):
                show(tag, ivA, ivB, bi, 0, False, "delta")
                for c in range(2):
                    show(tag, ivA, ivB, bi, c, True, "delta")
                    show(tag, ivA, ivB, bi, c, True, "nm")
                    if held:
                        show(tag, ivA, ivB, bi, c, "held", "delta")

        print("  (a) waste (A) against wait (B)")
        for shape in ((1, 1), (16, 16)):
            for d in (0.1, 0.5, 0.9, 0.99):
                block(f"(a) shapes ({shape[0]},{shape[1]}), d={d:<4g}", VI[(shape[0], shape[1], 1.0, "prop")], VI[(shape[0], shape[1], d, "prop")])
        print("  (b) memoryless stays (A) against regular stays (B)")
        for d in (1.0, 0.5):
            for shape in ((16, 16), (4, 4), (16, 1), (1, 16)):
                block(f"(b) {'waste' if d == 1 else 'wait d=0.5'}, B = out {shape[0]} in {shape[1]}", VI[(1, 1, d, "prop")], VI[(shape[0], shape[1], d, "prop")])
        print("  (c) proportional (A) against saturating (B) encounter")
        for d in (1.0, 0.5):
            block(f"(c) {'waste' if d == 1 else 'wait d=0.5'}, memoryless", VI[(1, 1, d, "prop")], VI[(1, 1, d, "sat")])
        print("  (d) the theory under each variant (A) against 'no effect of variability at equal mean' (B: mean L = 0 in every balanced treatment)")
        print("      first, ten variants in full; then every variant, one line per count")
        for var in compact:
            block(f"(d) {vlabel(var)}", VI[var], None, held=False)
        print("      every variant: the admissible treatment that, against the constant one, needs the fewest replicates in mirrored pairs (nm):")
        print("      treatment, day, mean L, n, nm, n*; 'blind' = admissible treatments with |mean L| < 0.32 at both days (not told from no effect by")
        print("      12 replicates even with measurement error only), of the number admissible")
        for c in range(2):
            for iv, var in enumerate(VARS):
                parts = []
                for bi in range(3):
                    r = search(iv, None, bi, c, True, "nm", with_const_only=True)
                    adm = [t for t in TREATS if admissible(bi, t)]
                    blind = sum(1 for t in adm if abs(lk(iv, bi, t, 0, c).mean()) < 0.32 and abs(lk(iv, bi, t, 1, c).mean()) < 0.32)
                    _, day, t1, t2, delta, ca, cb, sdd, extra, extram = r
                    tt = t2 if t1 is None else t1
                    parts.append(f"b={B_OUT[bi]:g}: [{tlabel(tt)}, day {int(DAYS[day])}] L {(ca if t1 is None else -ca):+.2f} n {nfmt(nrep(delta))} nm {nfmt(nrep(delta, extram))} n* {nfmt(nrep(delta, extra))} blind {blind}/{len(adm)}")
                print(f"  (d) {CNT[c]:<13s} {vlabel(var)} | " + " | ".join(parts))
        print("  (c'), (d') the part that is not a change of the mean rate: a slow schedule minus a fast one with the same levels (swing 1.8/0.2, day 60).")
        print("      'any': telegraph dwell 8 gen minus dwell 1/4 gen (the protocol's contrast).  'adm': the slowest admissible minus the fastest (1/4 gen),")
        print("      telegraph (te) and square wave (sq).  Entries: contrast, n, nm, n*.")

        def slowfast(iv, bi, sch, kslow, c):
            xs, xf = lk(iv, bi, (sch, 0, kslow), 1, c), lk(iv, bi, (sch, 0, 0), 1, c)
            con = xs.mean() - xf.mean()
            return con, nrep(con), nrep(con, 2.0 * (pairsd(xs) ** 2 + pairsd(xf) ** 2)), nrep(con, xs.std() ** 2 + xf.std() ** 2)

        for c in range(2):
            for iv, var in enumerate(VARS):
                parts = []
                for bi in range(3):
                    con, n1, n2, n3 = slowfast(iv, bi, 1, 5, c)
                    txt = f"b={B_OUT[bi]:g}: any te 8-1/4 {con:+.2f} n {nfmt(n1)} nm {nfmt(n2)} n* {nfmt(n3)}"
                    for sch, nm in ((1, "te"), (0, "sq")):
                        ks = [ki for ki in range(1, 6) if admissible(bi, (sch, 0, ki))]
                        if ks:
                            con, n1, n2, n3 = slowfast(iv, bi, sch, ks[-1], c)
                            txt += f"; adm {nm} {KLAB[ks[-1]]}-1/4 {con:+.2f} n {nfmt(n1)} nm {nfmt(n2)} n* {nfmt(n3)}"
                    parts.append(txt)
                print(f"  (d') {CNT[c]:<13s} {vlabel(var)} | " + " | ".join(parts))
        print("  (c') Delta between proportional and saturating for the slow-minus-fast contrasts, memoryless stays, all parasites, day 60:")
        for d in (1.0, 0.5):
            for bi in range(3):
                specs = [(1, "any te", 5)]
                for sch, nm in ((1, "te"), (0, "sq")):
                    ks = [ki for ki in range(1, 6) if admissible(bi, (sch, 0, ki))]
                    if ks:
                        specs.append((sch, "adm " + nm, ks[-1]))
                txt = []
                for sch, nm, kk in specs:
                    xa = lk(VI[(1, 1, d, "prop")], bi, (sch, 0, kk), 1, 0).mean() - lk(VI[(1, 1, d, "prop")], bi, (sch, 0, 0), 1, 0).mean()
                    xb = lk(VI[(1, 1, d, "sat")], bi, (sch, 0, kk), 1, 0).mean() - lk(VI[(1, 1, d, "sat")], bi, (sch, 0, 0), 1, 0).mean()
                    txt.append(f"{nm} {KLAB[kk]}-1/4: prop {xa:+.3f}, sat {xb:+.3f}, Delta {abs(xa - xb):.3f}, n {nfmt(nrep(xa - xb))}")
                print(f"  (c') {'waste' if d == 1 else 'wait d=0.5'} b={B_OUT[bi]:<2g}: " + "; ".join(txt))
        print("  (e) spread of ln N between starting phases (square) and between schedules (telegraph), swing 1.8/0.2, day 60, all parasites, columns 1/4 ... 8 gen:")
        print("      sd over single phases or schedules, then sd over mirrored pairs")
        for var in (base, (16, 16, 1.0, "prop")):
            for sch, sname in enumerate(("square wave", "telegraph")):
                for bi in range(3):
                    print(f"      {vlabel(var)} {sname:<11s} b={B_OUT[bi]:<2g} | single " + row6(L[sch, VI[var], bi, 0, :, 1, 0, 3], "{:5.2f}") + " | pairs " + row6(L[sch, VI[var], bi, 0, :, 1, 0, 5], "{:5.2f}"))
        print("  (f) the sign of the outcome: (treatment, day) cells in which the fluctuating treatment has MORE parasites on average (mean L < -0.05), all parasites")
        for resp in RESP:
            for d in D_SET:
                sel = [iv for iv, v in enumerate(VARS) if v[2] == d and v[3] == resp]
                cnt = [int(np.sum(L[sch][sel][..., 0, 0] < -0.05)) for sch in range(2)]
                tot = L[0][sel][..., 0, 0].size
                print(f"      {'waste' if d == 1 else 'wait d=%g' % d:<12s} {resp}: square wave {cnt[0]} of {tot}, telegraph {cnt[1]} of {tot}; most negative mean L: square {np.min(L[0][sel][..., 0, 0]):+.3f}, telegraph {np.min(L[1][sel][..., 0, 0]):+.3f}")
        print()

    print("=" * 110)
    print("time per task (b index, k index, seconds): " + ", ".join(f"({bi},{ki}) {sc:.0f}" for bi, ki, sc in sorted(secs)))
    print(f"wall time {time.time() - t_start:.0f} s; finished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
