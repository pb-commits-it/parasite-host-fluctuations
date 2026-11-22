"""Protocol R11 (protocols/R11-dobson-trophic-model.md): Dobson's nonlinear model with a predator
and its prey as the two hosts of a trophically transmitted parasite, in the corrected reading
fixed by the protocol.  One run; the report is reports/R11-dobson-trophic-model.md.

Run from problems/dobson-r0:   python3 -I checks/r11_dobson_trophic.py
(The options --draws, --seed and --procs exist only so that the code could be debugged on small
samples drawn with another seed.  The registered run is the one with no options: 20,000 draws,
seed 20261114, 6 processes.)

THE MODEL.  H predators (definitive hosts), P adult parasites in predators, W free-living
infective stages, C uninfected prey (intermediate hosts), I infected prey; N = C + I.

    dH/dt = e a H N/(C50 + N) - d H - alpha P
    dP/dt = nu a H I/(C50 + N) - (mu + d + alpha) P - alpha kappa P^2/H
    dW/dt = lambda P - sigma W - beta N W
    dC/dt = f N (1 - N/K) - d_C C - a H C/(C50 + N) - beta C W
    dI/dt = beta C W - d_C I - a H I/(C50 + N)

kappa = (k + 1)/k for clumping parameter k, and kappa = 1 for the printed form ("k = infinity").
Units: d = 1 (time in predator lifetimes), C50 = 1 (unit of prey density).  K = infinity means the
prey has no regulation of its own.

DERIVED HERE (each is compared below with the model's right-hand side or its Jacobian).
 - Parasite-free equilibrium: C0 = d C50/(e a - d), H0 = (C50 + C0)(f(1 - C0/K) - d_C)/a.
   Because f(1 - C/K) - d_C = (f - d_C)(1 - C/K') with K' = K(1 - d_C/f), the prey alone settles at
   K' (its carrying capacity net of deaths), and H0 > 0 needs C0 < K'.
 - Rosenzweig-MacArthur condition for this model.  The prey isocline is
   H = (C50 + C)(f - d_C)(1 - C/K')/a, with its peak at C = (K' - C50)/2.  The predator-prey
   Jacobian at (H0, C0) has positive determinant and trace
   (f - d_C) C0 (K' - C50 - 2 C0)/(K'(C50 + C0)), so the equilibrium is unstable exactly when
   C0 < (K' - C50)/2.  With K = infinity the trace is (f - d_C) C0/(C50 + C0) > 0: never stable.
 - R0 of the parasite at the parasite-free equilibrium, with phi0 = a H0/(C50 + C0) the rate at
   which a prey is eaten:
       R0 = [lambda/(mu + d + alpha)] x [beta C0/(sigma + beta C0)] x [phi0/(d_C + phi0)] x nu
   (eggs per adult lifetime x chance a free-living stage is eaten by a prey x chance an infected
   prey is eaten rather than dying x adults established per infected prey eaten).
 - Every endemic equilibrium has H = (C50 + N)(f(1 - N/K) - d_C)/a,
   P/H = (e a N/(C50 + N) - d)/alpha, I = (C50 + N)(P/H)(mu + d + alpha + alpha kappa P/H)/(nu a),
   W = lambda P/(sigma + beta N), and N solves one equation,
   G(N) = beta (N - I) W - (d_C + a H/(C50 + N)) I = 0 on C0 < N < K'.  This reduction is not in
   the protocol.  It is used for some of the random starting points of Newton's method and for
   the supplementary scan S; every equilibrium it suggests is accepted only after Newton's method
   on the full model and the residual test.

THE DIRECT CYCLE (H4).  Same predator, same prey as food, same adult parasite; the prey is not
a host; predators pick the free-living stage up directly:

    dH/dt = e a H C/(C50 + C) - d H - alpha P
    dP/dt = beta_H W H - (mu + d + alpha) P - alpha kappa P^2/H
    dW/dt = lambda_D P - sigma W - beta_H W H
    dC/dt = f C (1 - C/K) - d_C C - a H C/(C50 + C)

Its R0 is [lambda_D/(mu + d + alpha)] x [beta_H H0/(sigma + beta_H H0)], at the same (H0, C0).
beta_H and lambda_D are chosen so that (i) this R0 equals the trophic R0 and (ii) the mean time
round the cycle at the direct cycle's own endemic equilibrium,
    1/(mu + d + alpha + alpha kappa P/H) + 1/(sigma + beta_H H),
equals the trophic cycle's at its endemic equilibrium,
    1/(mu + d + alpha + alpha kappa P/H) + 1/(sigma + beta N) + 1/(d_C + a H/(C50 + N)).
Every other parameter is unchanged.  How the pair is found: at the direct cycle's equilibrium the
prey density C_D fixes P/H and H (first and fourth equations), condition (ii) then fixes
beta_H H, the second and third equations fix lambda_D, and condition (i) is one equation in C_D,
solved on the same grid as S with a bracketing root-finder.  Sets with no solution are skipped
and counted; if there are several, the one with the smallest C_D is used and the set is counted.
A match counts only if R0 and the time agree to 1e-9 relative and the direct equilibrium has
relative residual below 1e-9.

WHAT IS TESTED, AND THE TOLERANCES (fixed before the run).
H0  (check; must pass before anything else is used)
    (a-i)  1,000 random predator-prey parameter sets, K placed 1% above and 1% below the boundary
           K' = C50 + 2 C0: the largest real part of the predator-prey block of the
           finite-difference Jacobian of the full model is > +1e-6 above and < -1e-6 below.
           The peak of the prey isocline, found by numerical maximisation, equals (K' - C50)/2
           (1e-6 of K'; this second tolerance is mine, the protocol states none for it).
    (a-ii) 1,000 sets with K drawn from the grid: unstable exactly when C0 < (K' - C50)/2.
    (b)    1,000 sets with K = infinity: largest real part > +1e-6 in all.
R0  (check) 2,000 unfiltered draws: the parasite block (P, W, I) of the finite-difference Jacobian
    at the parasite-free equilibrium has a positive eigenvalue exactly when R0 > 1.  The same for
    the direct cycle's block (P, W) with a random beta_H.  Zero exceptions.  Draws within 1e-6 of
    R0 = 1 are skipped and counted.
M   (check) the model is coded twice (natural variables; log variables for the integration);
    the two agree to 1e-10 of the size of the terms at 500 random states.
J   (check) finite-difference Jacobian against the analytic one at 200 endemic equilibria:
    largest difference of an entry of X^-1 J X (X = diag of the state), relative to the largest
    entry, below 1e-5.
H1  (open) share of endemic equilibria that are locally stable against the share of
    parasite-free equilibria that are stable to disturbances of predator and prey alone (every
    kept set has R0 > 1, so no kept parasite-free equilibrium is stable to the parasite).
H2  (open) sigma and beta divided by c, c in {0.01, 0.1, 0.3, 1, 3, 10, 100, 1e4}, at the first
    2,000 sets (in draw order) with an endemic equilibrium.  Before any eigenvalue: the
    equilibrium with W multiplied by c must satisfy the equations with relative residual below
    1e-9 (a check).
H3  (open) stability against the position (C* + I*)/K, sets with finite K.
H4  (open) stability of the matched direct cycle against the trophic cycle.
T   (check) twelve integrations of the nonlinear model at six stability boundaries found in H2,
    one on each side (the two neighbouring values of c between which stability changes): a
    disturbance of 1e-4 relative along the leading eigenvector, followed for 6/|Re z| with
    solve_ivp(method="LSODA", rtol=1e-10), ends below 0.1 of its early size (stable side) or above
    10 times it (unstable side); "early" and "late" are the first and last tenths of the time.

HOW EQUILIBRIA ARE FOUND (the protocol's method).  The model is integrated in log variables with
scipy's LSODA (the solver class behind solve_ivp(method="LSODA"), stepped directly so that a run
can be stopped at a step limit), relative tolerance 1e-9, from the parasite-free equilibrium with
the parasite added along the leading eigenvector of its block, scaled so that P <= 1e-3 H0 and
I <= 1e-3 C0.  "Settled" means: Newton's method on the full model from the state reached converges
to an equilibrium with all five components positive (above 1e-10 of the reference values below)
and relative residual below 1e-9 (the residual of each equation divided by the sum of the sizes
of its terms), and the state reached is within 1e-6 relative of it.  This is tried at the end of
stretches of time that double in length (10, 20, 40, ...), whenever the state reached has a
relative residual below 1e-3.  An integration stops when it settles;
when any variable falls below 1e-30 of its reference value (H0 for H and P, lambda H0/(sigma +
beta C0) for W, C0 for C and I) or the prey rises above 1e10 of it; at 150,000 steps; or at time
1e6.  What a trajectory that did not settle was doing is read from its extremes over its last two
stretches (function `behaviour`).  Sets that do not settle get Newton (scipy fsolve on the log
variables, then Newton on the full model) from 20 random starts (function `newton_random`).  If a
set has several positive equilibria the one with the smallest C* + I* is used.
S   (supplementary, not in the protocol) all sign changes of G(N) on a grid of 1,199 points
    between C0 and K' (1,200 points over twenty decades above C0 if K = infinity), each root then
    polished by Newton on the full model; used only to say how complete the protocol's method was.

"Stable" means the largest real part of the eigenvalues of the finite-difference Jacobian (central
differences, relative step 1e-6) is negative.  Proportions carry binomial standard errors.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
from scipy.integrate import LSODA, solve_ivp
from scipy.optimize import brentq, fsolve, minimize_scalar

SEED = 20261114
N_DRAWS = 20000
PROCS = 6
C50 = 1.0
D = 1.0
C_GRID = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1.0e4)
RES_TOL = 1e-9            # relative residual of an equilibrium
FD_REL = 1e-6             # relative step of the finite-difference Jacobian
RTOL, ATOL = 1e-9, 1e-11  # integration, log variables
MAX_STEPS = 150000
T_MAX = 1.0e6
WIN0 = 10.0
LOG_LOST = 30.0 * math.log(10.0)      # a variable below 1e-30 of its reference value: stop
LOG_PREY_UP = 10.0 * math.log(10.0)   # prey above 1e10 of its reference value: stop
SETTLE_DIST = 1e-6
N_STARTS = 20
SWEEP_BUDGET_S = 30 * 60  # the sweep stops here and the run reports what finished
FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def se(k, n):
    """Binomial standard error of the proportion k/n."""
    if n == 0:
        return float("nan")
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def frac(k, n):
    """k/n with its binomial standard error.  At 0 or n of n the standard error is zero by formula
    and says nothing, so the rule-of-three 95% bound (3/n) is given instead."""
    if n == 0:
        return f"{k}/0"
    if k == 0:
        return f"0/{n} = 0.000 (95% upper bound about {min(1.0, 3 / n):.3f})"
    if k == n:
        return f"{n}/{n} = 1.000 (95% lower bound about {max(0.0, 1 - 3 / n):.3f})"
    return f"{k}/{n} = {k / n:.3f} +- {se(k, n):.3f}"


# ------------------------------------------------------------------ parameters

def draw_sets(n, seed):
    """n parameter sets, log-uniform in the protocol's ranges.  Every fifth set has K = infinity."""
    rng = np.random.default_rng(seed)

    def logu(lo, hi):
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

    sets, redraws = [], 0
    for i in range(n):
        dC = logu(2, 50)
        f = dC * logu(1.5, 10)
        K = math.inf if i % 5 == 0 else logu(2, 200)
        while True:
            a = logu(5, 100)
            e = logu(0.02, 0.5)
            if e * a > D:
                break
            redraws += 1
        mu = logu(0.2, 20)
        al = logu(1e-3, 1)
        k = (0.2, 1.0, 5.0, math.inf)[int(rng.integers(4))]
        lam = logu(1, 1e4)
        sig = logu(1, 200)
        be = logu(1e-3, 10)
        nu = logu(0.1, 10)
        sets.append((dC, f, K, a, e, mu, al, k, lam, sig, be, nu))
    return sets, redraws


def derived(p):
    """(d_C, f, 1/K, a, e, mu, alpha, kappa, lambda, sigma, beta, nu)."""
    dC, f, K, a, e, mu, al, k, lam, sig, be, nu = p
    return (dC, f, 0.0 if math.isinf(K) else 1.0 / K, a, e, mu, al,
            1.0 if math.isinf(k) else (k + 1.0) / k, lam, sig, be, nu)


def scaled_stage(q, x, c):
    """sigma and beta divided by c; the equilibrium with W multiplied by c."""
    q2 = list(q)
    q2[9] = q[9] / c
    q2[10] = q[10] / c
    x2 = [float(v) for v in x]
    x2[2] *= c
    return tuple(q2), x2


# ------------------------------------------------------------------ the trophic model

def rhs_parts(q, x):
    """Right-hand side and, for each equation, the sum of the sizes of its terms."""
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    H, P, W, C, I = x
    N = C + I
    S = C50 + N
    ph = a * H / S
    t1 = e * ph * N
    t2 = nu * ph * I
    t3 = (mu + D + al) * P
    t4 = al * kap * P * P / H
    t5 = be * N * W
    t6 = f * N
    t7 = f * N * N * invK
    t8 = be * C * W
    F = [t1 - D * H - al * P,
         t2 - t3 - t4,
         lam * P - sig * W - t5,
         t6 - t7 - dC * C - ph * C - t8,
         t8 - dC * I - ph * I]
    Sc = [abs(t1) + abs(D * H) + abs(al * P),
          abs(t2) + abs(t3) + abs(t4),
          abs(lam * P) + abs(sig * W) + abs(t5),
          abs(t6) + abs(t7) + abs(dC * C) + abs(ph * C) + abs(t8),
          abs(t8) + abs(dC * I) + abs(ph * I)]
    return F, Sc


def rhs(q, x):
    return rhs_parts(q, x)[0]


def rel_residual(q, x):
    F, Sc = rhs_parts(q, x)
    return max((abs(u) / s if s > 0 else math.inf) for u, s in zip(F, Sc))


def jac_an(q, x):
    """Analytic Jacobian of the trophic model."""
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    H, P, W, C, I = x
    N = C + I
    S = C50 + N
    S2 = S * S
    J = np.zeros((5, 5))
    J[0, 0] = e * a * N / S - D
    J[0, 1] = -al
    J[0, 3] = J[0, 4] = e * a * H * C50 / S2
    J[1, 0] = nu * a * I / S + al * kap * P * P / (H * H)
    J[1, 1] = -(mu + D + al) - 2 * al * kap * P / H
    J[1, 3] = -nu * a * H * I / S2
    J[1, 4] = nu * a * H * (S - I) / S2
    J[2, 1] = lam
    J[2, 2] = -sig - be * N
    J[2, 3] = J[2, 4] = -be * W
    J[3, 0] = -a * C / S
    J[3, 2] = -be * C
    J[3, 3] = f * (1 - 2 * N * invK) - dC - a * H / S + a * H * C / S2 - be * W
    J[3, 4] = f * (1 - 2 * N * invK) + a * H * C / S2
    J[4, 0] = -a * I / S
    J[4, 2] = be * C
    J[4, 3] = be * W + a * H * I / S2
    J[4, 4] = -dC - a * H / S + a * H * I / S2
    return J


def jac_fd(fun, x, h):
    """Central-difference Jacobian of fun at x with steps h."""
    n = len(x)
    J = np.empty((n, n))
    for j in range(n):
        xp = [float(v) for v in x]
        xm = [float(v) for v in x]
        xp[j] += h[j]
        xm[j] -= h[j]
        J[:, j] = (np.array(fun(xp)) - np.array(fun(xm))) / (2.0 * h[j])
    return J


def max_real(J):
    ev = np.linalg.eigvals(J)
    return float(np.max(ev.real)), ev


def classify(q, x):
    """Largest real part from the finite-difference Jacobian, the same from the analytic one,
    and the largest modulus."""
    xl = [float(v) for v in x]
    mf, ev = max_real(jac_fd(lambda z: rhs(q, z), xl, [FD_REL * v for v in xl]))
    ma, _ = max_real(jac_an(q, xl))
    return mf, ma, float(np.max(np.abs(ev)))


def pfe(q):
    """Parasite-free equilibrium (H0, C0), or None if the predator cannot persist."""
    dC, f, invK, a, e = q[:5]
    if e * a <= D:
        return None
    C0 = C50 * D / (e * a - D)
    H0 = (C50 + C0) * (f * (1 - C0 * invK) - dC) / a
    if H0 <= 0:
        return None
    return H0, C0


def r0_trophic(q, H0, C0):
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    ph0 = a * H0 / (C50 + C0)
    return lam / (mu + D + al) * (be * C0 / (sig + be * C0)) * (ph0 / (dC + ph0)) * nu


def ref_scales(q, H0, C0):
    lam, sig, be = q[8], q[9], q[10]
    return [H0, H0, lam * H0 / (sig + be * C0), C0, C0]


def pfe_jacobian(q, H0, C0):
    """Finite-difference Jacobian of the full model at the parasite-free equilibrium."""
    return jac_fd(lambda z: rhs(q, z), [H0, 0.0, 0.0, C0, 0.0], [FD_REL * v for v in ref_scales(q, H0, C0)])


def make_logrhs(q):
    """d(ln x)/dt as a function of y = ln x, written with exponentials of differences so that
    very small densities do not divide zero by zero."""
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    mtot = mu + D + al
    ak = al * kap
    exp = math.exp

    def g(t, y):
        yH, yP, yW, yC, yI = y
        N = exp(yC) + exp(yI)
        S = C50 + N
        ph = a * exp(yH) / S
        r = exp(yP - yH)
        return [e * a * N / S - D - al * r,
                nu * a * exp(yH + yI - yP) / S - mtot - ak * r,
                lam * exp(yP - yW) - sig - be * N,
                f * (1.0 - N * invK) * (1.0 + exp(yI - yC)) - dC - ph - be * exp(yW),
                be * exp(yC + yW - yI) - dC - ph]
    return g


def polish(q, x):
    """Newton's method on the full model from x.  Returns (equilibrium, relative residual)."""
    x = [float(v) for v in x]
    try:
        prev = math.inf
        for _ in range(60):
            F, Sc = rhs_parts(q, x)
            rel = max((abs(u) / s if s > 0 else math.inf) for u, s in zip(F, Sc))
            if rel < 1e-14 or (rel < 1e-11 and rel > 0.5 * prev):
                break
            prev = rel
            dx = np.linalg.solve(jac_an(q, x), -np.array(F))
            x = [xi + float(di) for xi, di in zip(x, dx)]
            if not all(math.isfinite(v) for v in x):
                return None, math.inf
        rel = rel_residual(q, x)
    except (ZeroDivisionError, OverflowError, np.linalg.LinAlgError, ValueError):
        return None, math.inf
    return np.array(x), rel


def accept(q, x, ref):
    """Polish x; return the equilibrium if it is positive and its residual is small enough."""
    xs, rel = polish(q, x)
    if xs is None or not (rel < RES_TOL):
        return None
    if any(xs[j] <= 1e-10 * ref[j] for j in range(5)):
        return None
    return xs


def add_unique(found, xs):
    for z in found:
        if np.max(np.abs(np.log(xs / z))) < 1e-6:
            return
    found.append(xs)


# ------------------------------------------------------------------ the one-equation reduction

def reduction(q, N):
    """State on the curve of equilibria given N (works for arrays), and G(N)."""
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    S = C50 + N
    m = (e * a * N / S - D) / al
    H = S * (f * (1 - N * invK) - dC) / a
    I = S * m * (mu + D + al + al * kap * m) / (nu * a)
    W = lam * m * H / (sig + be * N)
    G = be * (N - I) * W - (dC + a * H / S) * I
    return (H, m * H, W, N - I, I), G


def n_grid(q, C0, n_half=600):
    """Grid of N between C0 and K' (or far above C0 if K = infinity), fine near both ends."""
    dC, f, invK = q[:3]
    if invK > 0:
        Kp = (1 - dC / f) / invK
        u = np.logspace(-10, math.log10(0.5), n_half)
        u = np.concatenate([u, 1 - u[::-1][1:]])
        return C0 + (Kp - C0) * u
    return C0 + max(C0, C50) * np.logspace(-10, 10, 2 * n_half)


def scan_equilibria(q, H0, C0):
    """Supplementary: every root of G on the grid, polished on the full model."""
    ref = ref_scales(q, H0, C0)
    N = n_grid(q, C0)
    with np.errstate(all="ignore"):
        _, G = reduction(q, N)
    found = []
    sg = np.sign(G)
    for j in np.nonzero(sg[:-1] * sg[1:] < 0)[0]:
        try:
            root = brentq(lambda z: reduction(q, z)[1], N[j], N[j + 1], xtol=1e-15 * N[j], rtol=8.9e-16, maxiter=200)
        except (ValueError, RuntimeError):
            continue
        x, _ = reduction(q, root)
        xs = accept(q, x, ref)
        if xs is not None:
            add_unique(found, xs)
    return found


# ------------------------------------------------------------------ integration from the parasite-free state

def integrate_from_pfe(q, H0, C0):
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    ph0 = a * H0 / (C50 + C0)
    A = np.array([[-(mu + D + al), 0.0, nu * ph0], [lam, -(sig + be * C0), 0.0], [0.0, be * C0, -(dC + ph0)]])
    w, v = np.linalg.eig(A)
    vec = np.abs(v[:, int(np.argmax(w.real))].real)
    eps = 1e-3 / max(vec[0] / H0, vec[2] / C0)
    y = [math.log(u) for u in (H0, eps * vec[0], eps * vec[1], C0, eps * vec[2])]
    ref = ref_scales(q, H0, C0)
    yref = [math.log(u) for u in ref]
    lo = [u - LOG_LOST for u in yref]
    hi = [yref[0] + LOG_LOST, yref[1] + LOG_LOST, yref[2] + LOG_LOST, yref[3] + LOG_PREY_UP, yref[4] + LOG_PREY_UP]
    g = make_logrhs(q)
    t = 0.0
    nsteps = restarts = 0
    windows = []
    wmin, wmax, wint, wt0 = list(y), list(y), [0.0] * 5, 0.0
    wlen = WIN0
    wend = WIN0
    status, which, xstar = "limit", -1, None
    solver = LSODA(g, t, y, T_MAX, rtol=RTOL, atol=ATOL)
    while True:
        failed = False
        try:
            solver.step()
        except (OverflowError, ZeroDivisionError, FloatingPointError, ValueError):
            failed = True
        if failed or solver.status == "failed":
            restarts += 1
            if restarts > 3:
                status = "failed"
                break
            try:
                solver = LSODA(g, t, y, T_MAX, rtol=RTOL, atol=ATOL)
            except Exception:
                status = "failed"
                break
            continue
        nsteps += 1
        tn, yn = solver.t, solver.y.tolist()
        dt = tn - t
        out = 0
        for j in range(5):
            u = yn[j]
            if u < wmin[j]:
                wmin[j] = u
            elif u > wmax[j]:
                wmax[j] = u
            wint[j] += 0.5 * (u + y[j]) * dt
            if u < lo[j]:
                out = -(j + 1)
            elif u > hi[j]:
                out = j + 1
        t, y = tn, yn
        if out:
            status, which = ("lost" if out < 0 else "unbounded"), abs(out) - 1
            break
        done = solver.status == "finished"
        if t >= wend or done:
            windows.append((wt0, t, np.array(wmin), np.array(wmax), np.array(wint) / (t - wt0)))
            x_now = np.exp(np.array(y))
            if rel_residual(q, [float(u) for u in x_now]) < 1e-3:
                xs = accept(q, x_now, ref)
                if xs is not None and float(np.max(np.abs(x_now / xs - 1.0))) < SETTLE_DIST:
                    status, xstar = "settled", xs
                    break
            wmin, wmax, wint, wt0 = list(y), list(y), [0.0] * 5, t
            wlen *= 2.0
            wend = t + wlen
        if done:
            status = "tmax"
            break
        if nsteps >= MAX_STEPS:
            break
    if status in ("limit", "tmax") and t > wt0 and windows and (t - wt0) >= 0.5 * (windows[-1][1] - windows[-1][0]):
        windows.append((wt0, t, np.array(wmin), np.array(wmax), np.array(wint) / (t - wt0)))
    return status, which, xstar, windows, t, nsteps, np.array(y)


def behaviour(status, which, windows, dy):
    """What a trajectory that did not settle was doing when it stopped.  dy = ln(state/reference)
    at the stop.  A stop because a variable fell below 1e-30 of its reference is put down to the
    prey if the uninfected prey is itself below 1e-20 of its reference (a crash of the prey takes
    the infected prey and the parasite with it), otherwise to the variable that crossed."""
    if status == "lost":
        if which == 3 or dy[3] < -20.0 * math.log(10.0):
            return "prey collapses"
        return "predator lost" if which == 0 else "parasite lost"
    if status == "unbounded":
        return "prey escapes" if which in (3, 4) else "unbounded " + "HPW"[which]
    if status == "failed":
        return "integration failed"
    if len(windows) < 2:
        return "undetermined"
    w1, w2 = windows[-2], windows[-1]
    r1, r2 = w1[3] - w1[2], w2[3] - w2[2]
    dmean = w2[4] - w1[4]
    par = (1, 2, 4)
    if all(dmean[j] < -1.0 for j in par) and all(w2[3][j] < w1[3][j] - 1.0 for j in par):
        return "parasite declining"
    if dmean[0] < -1.0 and w2[3][0] < w1[3][0] - 1.0:
        return "predator declining"
    if float(r2.max()) < 1e-3:
        return "nearly steady"
    if bool(np.all(np.abs(r2 - r1) <= 0.1 * np.maximum(r1, r2) + 1e-3)):
        return "sustained cycle"
    return "oscillation growing" if float(r2.sum()) > float(r1.sum()) else "oscillation shrinking"


def newton_random(q, H0, C0, centre, rng):
    """Newton (scipy fsolve on the log variables, then polish on the full model) from N_STARTS
    random starts.  With a centre (the time-average of ln x over the trajectory's last stretch)
    half the starts are scattered round it.  The others take N at random, one start in each of
    equal slices of the logarithmic range between C0 and K' (alternately counted from C0 and from
    K'; twelve decades above C0 if K = infinity), put the other variables on the curve of
    equilibria at that N, and multiply each by a random factor between 1/e and e."""
    dC, f, invK = q[:3]
    ref = ref_scales(q, H0, C0)
    g = make_logrhs(q)

    def fun(y):
        try:
            return g(0.0, np.clip(y, -300.0, 300.0))
        except OverflowError:
            return [1e30] * 5

    n_centre = N_STARTS // 2 if centre is not None else 0
    n_curve = N_STARTS - n_centre
    found = []
    for i in range(N_STARTS):
        if i < n_centre:
            y0 = centre + rng.normal(0.0, 1.0, 5)
        else:
            j = i - n_centre
            if invK > 0:
                Kp = (1 - dC / f) / invK
                half = (n_curve + 1) // 2
                u = 10.0 ** (-6.0 + 6.0 * (j // 2 + rng.random()) / half)
                if j % 2:
                    u = 1.0 - u
                N = C0 + (Kp - C0) * min(max(u, 1e-9), 1 - 1e-9)
            else:
                N = C0 + max(C0, C50) * 10.0 ** (-6.0 + 12.0 * (j + rng.random()) / n_curve)
            (H, P, W, C, I), _ = reduction(q, N)
            if C <= 0:
                C = 1e-2 * N
            x0 = np.array([H, P, W, C, I])
            if not np.all(np.isfinite(x0)) or np.any(x0 <= 0):
                x0 = np.array(ref) * np.exp(rng.uniform(-3.0, 3.0, 5))
            y0 = np.log(x0) + rng.uniform(-1.0, 1.0, 5)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ys = fsolve(fun, y0, xtol=1e-12, maxfev=400)
        except Exception:
            continue
        with np.errstate(all="ignore"):
            x = np.exp(np.clip(ys, -300.0, 300.0))
        xs = accept(q, x, ref)
        if xs is not None:
            add_unique(found, xs)
    return found


def process_set(task):
    """Everything done to one kept parameter set.  Runs in a worker process.  An unexpected
    error is recorded and counted, and the set is left out of every table."""
    try:
        return _process_set(task)
    except Exception as ex:  # noqa: BLE001
        return {"i": task[0], "status": "error", "error": repr(ex)}


def _process_set(task):
    i, p = task
    t0 = time.perf_counter()
    q = derived(p)
    H0, C0 = pfe(q)
    Jp = pfe_jacobian(q, H0, C0)
    pfe_re, _ = max_real(Jp[np.ix_([0, 3], [0, 3])])
    status, which, xstar, windows, t_end, nsteps, y_end = integrate_from_pfe(q, H0, C0)
    rec = {"i": i, "status": status, "t_end": t_end, "nsteps": nsteps, "pfe_re": pfe_re,
           "behaviour": "settled", "eqs": [], "scan": [], "sec_int": time.perf_counter() - t0}
    if status == "settled":
        rec["eqs"].append((xstar,) + classify(q, xstar))
    else:
        yref = np.log(np.array(ref_scales(q, H0, C0)))
        rec["behaviour"] = behaviour(status, which, windows, y_end - yref)
        centre = windows[-1][4] if (status in ("limit", "tmax") and windows) else None
        if status in ("limit", "tmax") and windows:
            rec["trough"] = float(np.min(windows[-1][2] - yref)) / math.log(10.0)
        rng = np.random.default_rng([SEED, 7, i])
        for xs in newton_random(q, H0, C0, centre, rng):
            rec["eqs"].append((xs,) + classify(q, xs))
        if rec["eqs"] and status in ("limit", "tmax") and len(windows) >= 2:
            # is the trajectory closing in on a stable equilibrium that Newton found?
            for xs, mf, ma, _ in rec["eqs"]:
                if mf < 0:
                    ys = np.log(xs)
                    d1 = max(float(np.max(np.abs(windows[-2][k] - ys))) for k in (2, 3))
                    d2 = max(float(np.max(np.abs(windows[-1][k] - ys))) for k in (2, 3))
                    if d2 < d1 and d2 < 0.1:
                        rec["behaviour"] = "still converging"
    for xs in scan_equilibria(q, H0, C0):
        rec["scan"].append((xs,) + classify(q, xs))
    rec["seconds"] = time.perf_counter() - t0
    return rec


def _quiet():
    """Workers return results and print nothing; the Fortran solver's own messages are dropped."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    np.seterr(all="ignore")


# ------------------------------------------------------------------ H0 and the R0 formula

def h0():
    rng = np.random.default_rng([SEED, 1])

    def logu(lo, hi):
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

    def pred_prey():
        dC = logu(2, 50)
        f = dC * logu(1.5, 10)
        while True:
            a = logu(5, 100)
            e = logu(0.02, 0.5)
            if e * a > D:
                return dC, f, a, e

    def block_re(dC, f, invK, a, e):
        q = (dC, f, invK, a, e, 1.0, 0.1, 2.0, 10.0, 5.0, 0.1, 1.0)  # the parasite's values do not enter
        H0, C0 = pfe(q)
        return max_real(pfe_jacobian(q, H0, C0)[np.ix_([0, 3], [0, 3])])[0], H0, C0

    bad = 0
    worst_peak = 0.0
    smallest = math.inf
    for _ in range(1000):
        dC, f, a, e = pred_prey()
        C0 = C50 * D / (e * a - D)
        Kcrit = (C50 + 2 * C0) / (1 - dC / f)
        for side, K in ((+1, 1.01 * Kcrit), (-1, 0.99 * Kcrit)):
            Kp = K * (1 - dC / f)
            res = minimize_scalar(lambda C: -(C50 + C) * (f * (1 - C / K) - dC) / a, bounds=(0.0, Kp),
                                  method="bounded", options={"xatol": 1e-13 * Kp})
            worst_peak = max(worst_peak, abs(res.x - (Kp - C50) / 2) / Kp)
            mr, _, _ = block_re(dC, f, 1.0 / K, a, e)
            smallest = min(smallest, abs(mr))
            if not ((mr > 1e-6) if side > 0 else (mr < -1e-6)):
                bad += 1
    report("H0a-i Rosenzweig-MacArthur boundary", bad == 0 and worst_peak < 1e-6,
           f"1,000 sets at K 1% above and 1% below the boundary, exceptions {bad}, smallest |Re z| {smallest:.2e}; "
           f"isocline peak by maximisation against (K' - C50)/2: largest error {worst_peak:.1e} of K'")

    bad = n_un = n_st = skipped = 0
    for _ in range(1000):
        dC, f, a, e = pred_prey()
        K = logu(2, 200)
        Kp = K * (1 - dC / f)
        C0 = C50 * D / (e * a - D)
        if C0 >= Kp:
            skipped += 1
            continue
        mr, _, _ = block_re(dC, f, 1.0 / K, a, e)
        want_unstable = C0 < (Kp - C50) / 2
        n_un += want_unstable
        n_st += not want_unstable
        if (mr > 0) != want_unstable:
            bad += 1
    report("H0a-ii the condition away from the boundary", bad == 0,
           f"{n_un + n_st} sets with K from the grid ({skipped} more had no positive equilibrium): "
           f"{n_un} predicted unstable, {n_st} predicted stable, exceptions {bad}")

    bad = 0
    smallest = math.inf
    for _ in range(1000):
        dC, f, a, e = pred_prey()
        mr, _, _ = block_re(dC, f, 0.0, a, e)
        smallest = min(smallest, mr)
        if not mr > 1e-6:
            bad += 1
    report("H0b prey without regulation (K = infinity)", bad == 0,
           f"1,000 sets, stable {bad}, smallest largest real part {smallest:.2e}")


def codings_check():
    """The model is coded twice: in natural variables (right-hand side, Jacobians, residuals) and
    in log variables (integration).  They must agree."""
    sets, _ = draw_sets(500, [SEED, 4])
    rng = np.random.default_rng([SEED, 5])
    worst = 0.0
    for p in sets:
        q = derived(p)
        x = np.exp(rng.uniform(-4.0, 4.0, 5))
        F, Sc = rhs_parts(q, [float(v) for v in x])
        gl = make_logrhs(q)(0.0, [float(v) for v in np.log(x)])
        worst = max(worst, max(abs(gl[j] - F[j] / x[j]) / (Sc[j] / x[j]) for j in range(5)))
    report("M the two codings of the model agree", worst < 1e-10,
           f"500 random parameter sets and states, largest difference {worst:.1e} of the size of the terms")


def r0_check():
    sets, _ = draw_sets(2000, [SEED, 2])
    rng = np.random.default_rng([SEED, 3])
    bad = n_gt = n_lt = skipped = nopfe = 0
    bad_d = n_gt_d = n_lt_d = 0
    for p in sets:
        q = derived(p)
        eq = pfe(q)
        if eq is None:
            nopfe += 1
            continue
        H0, C0 = eq
        R0 = r0_trophic(q, H0, C0)
        if abs(R0 - 1) < 1e-6:
            skipped += 1
            continue
        mr, _ = max_real(pfe_jacobian(q, H0, C0)[np.ix_([1, 2, 4], [1, 2, 4])])
        n_gt += R0 > 1
        n_lt += R0 < 1
        if (mr > 0) != (R0 > 1):
            bad += 1
        # the direct cycle with a random uptake rate beta_H and the same lambda
        dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
        beH = float(math.exp(rng.uniform(math.log(1e-3), math.log(10.0))))
        qd = (dC, f, invK, a, e, mu, al, kap, lam, sig, beH)
        R0d = lam / (mu + D + al) * beH * H0 / (sig + beH * H0)
        if abs(R0d - 1) < 1e-6:
            continue
        Jd = jac_fd(lambda z: rhs_direct_parts(qd, z)[0], [H0, 0.0, 0.0, C0], [FD_REL * H0, FD_REL * H0, FD_REL * lam * H0 / sig, FD_REL * C0])
        mrd, _ = max_real(Jd[np.ix_([1, 2], [1, 2])])
        n_gt_d += R0d > 1
        n_lt_d += R0d < 1
        if (mrd > 0) != (R0d > 1):
            bad_d += 1
    report("R0 formula against the Jacobian at the parasite-free equilibrium", bad == 0 and n_gt + n_lt >= 500,
           f"{n_gt + n_lt} sets ({n_gt} with R0 > 1, {n_lt} with R0 < 1; {nopfe} of 2,000 draws had no positive "
           f"parasite-free equilibrium, {skipped} within 1e-6 of R0 = 1), exceptions {bad}")
    report("R0 formula of the direct cycle against its Jacobian", bad_d == 0 and n_gt_d + n_lt_d >= 500,
           f"{n_gt_d + n_lt_d} sets with beta_H drawn log-uniformly in [1e-3, 10] ({n_gt_d} with R0 > 1, {n_lt_d} with R0 < 1), exceptions {bad_d}")


# ------------------------------------------------------------------ the direct cycle (H4)

def rhs_direct_parts(qd, x):
    dC, f, invK, a, e, mu, al, kap, lamD, sig, beH = qd
    H, P, W, C = x
    S = C50 + C
    ph = a * H / S
    t1 = e * ph * C
    t2 = beH * W * H
    t3 = (mu + D + al) * P
    t4 = al * kap * P * P / H
    F = [t1 - D * H - al * P,
         t2 - t3 - t4,
         lamD * P - sig * W - t2,
         f * C - f * C * C * invK - dC * C - ph * C]
    Sc = [abs(t1) + abs(D * H) + abs(al * P),
          abs(t2) + abs(t3) + abs(t4),
          abs(lamD * P) + abs(sig * W) + abs(t2),
          abs(f * C) + abs(f * C * C * invK) + abs(dC * C) + abs(ph * C)]
    return F, Sc


def direct_match(q, xs, R0T, H0, C0):
    """beta_H and lambda_D matching R0 and the mean time round the cycle.  Returns a dict."""
    dC, f, invK, a, e, mu, al, kap, lam, sig, be, nu = q
    mtot = mu + D + al
    H, P, W, C, I = [float(v) for v in xs]
    N = C + I
    t_prey = 1.0 / (dC + a * H / (C50 + N))
    TT = 1.0 / (mtot + al * kap * P / H) + 1.0 / (sig + be * N) + t_prey

    def psi(CD):
        S = C50 + CD
        mD = (e * a * CD / S - D) / al
        HD = S * (f * (1 - CD * invK) - dC) / a
        AD = mtot + al * kap * mD
        rem = TT - 1.0 / AD                    # time left for the free-living stage
        sW = 1.0 / rem
        b = sW - sig                           # beta_H H at the direct equilibrium
        val = (AD / mtot) * sW * H0 / (sig * HD + b * H0) - R0T
        ok = (mD > 0) & (HD > 0) & (rem > 0) & (b > 0)
        return val, ok, (mD, HD, AD, sW, b)

    grid = n_grid(q, C0)
    with np.errstate(all="ignore"):
        val, ok, _ = psi(grid)
    out = {"status": "no match: the time can be matched at some prey densities, R0 at none of them", "roots": 0,
           "share_prey": t_prey / TT}
    if not np.any(ok):
        out["status"] = "no match: the direct cycle cannot be made as long as the trophic one at any prey density"
        return out
    roots = []
    for j in range(len(grid) - 1):
        if ok[j] and ok[j + 1] and val[j] * val[j + 1] < 0:
            try:
                roots.append(brentq(lambda z: float(psi(z)[0]), grid[j], grid[j + 1], xtol=1e-15 * grid[j], rtol=8.9e-16, maxiter=200))
            except (ValueError, RuntimeError):
                pass
    out["roots"] = len(roots)
    if not roots:
        return out
    CD = min(roots)
    _, okr, (mD, HD, AD, sW, b) = psi(CD)
    beH = b / HD
    lamD = AD * sW / b
    PD = mD * HD
    WD = lamD * PD / (sig + beH * HD)
    xd = [HD, PD, WD, CD]
    qd = (dC, f, invK, a, e, mu, al, kap, lamD, sig, beH)
    F, Sc = rhs_direct_parts(qd, xd)
    res = max(abs(u) / s for u, s in zip(F, Sc))
    R0D = lamD / mtot * beH * H0 / (sig + beH * H0)
    TD = 1.0 / (mtot + al * kap * PD / HD) + 1.0 / (sig + beH * HD)
    err = max(abs(R0D / R0T - 1), abs(TD / TT - 1))
    if not (okr and res < RES_TOL and err < 1e-9 and min(xd) > 0):
        out["status"] = "a match found but not to 1e-9"
        return out
    mr, ev = max_real(jac_fd(lambda z: rhs_direct_parts(qd, z)[0], xd, [FD_REL * v for v in xd]))
    out.update(status="matched", re=mr, res=res, err=err, T=TT, beH=beH, lamD=lamD)
    return out


# ------------------------------------------------------------------ the twelve integrations

def confirm_task(task):
    """One integration of the nonlinear model near an equilibrium: does a small disturbance shrink
    or grow as the eigenvalues say?"""
    try:
        return _confirm_task(task)
    except Exception:  # noqa: BLE001
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False)


def _confirm_task(task):
    q, xs, want_stable = task
    xs = np.array(xs, dtype=float)
    w, v = np.linalg.eig(jac_an(q, xs))
    k = int(np.argmax(w.real))
    z = w[k]
    vec = v[:, k].real / xs
    vec = vec / np.max(np.abs(vec))
    T = 6.0 / abs(z.real)
    g = make_logrhs(q)
    y0 = np.log(xs * (1.0 + 1e-4 * vec))
    sol = solve_ivp(g, (0.0, T), y0, method="LSODA", rtol=1e-10, atol=1e-12, t_eval=np.linspace(0.0, T, 2001))
    if not sol.success or sol.y.shape[1] < 2001:
        return (z.real, z.imag, T, float("nan"), float("nan"), False)
    dev = np.max(np.abs(np.exp(sol.y) / xs[:, None] - 1.0), axis=0)
    early, late = float(np.max(dev[:200])), float(np.max(dev[-200:]))
    ok = (late < 0.1 * early) if want_stable else (late > 10.0 * early)
    return (z.real, z.imag, T, early, late, bool(ok))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--procs", type=int, default=PROCS)
    args = ap.parse_args()
    t_start = time.time()
    np.seterr(all="ignore")
    print("R11: Dobson's model with a predator and its prey as the two hosts (checks/r11_dobson_trophic.py)")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (clock); seed {args.seed}; draws {args.draws}; {args.procs} processes")
    if args.draws != N_DRAWS or args.seed != SEED:
        print("*** DEBUGGING RUN: not the registered run ***")
    print()

    h0()
    r0_check()
    codings_check()
    if FAILS:
        print(f"\nH0 FAILED ({FAILS}); nothing else is run or used.")
        sys.exit(1)
    print(f"      (these checks took {time.time() - t_start:.0f} s)\n", flush=True)

    # ---------------------------------------------------------------- draws and the keep rule
    sets, redraws = draw_sets(args.draws, args.seed)
    kept, n_nopfe, n_r0low = [], 0, 0
    info = {}
    for i, p in enumerate(sets):
        q = derived(p)
        eq = pfe(q)
        if eq is None:
            n_nopfe += 1
            continue
        R0 = r0_trophic(q, *eq)
        if not R0 > 1:
            n_r0low += 1
            continue
        kept.append((i, p))
        info[i] = (q, eq[0], eq[1], R0, math.isinf(p[2]))
    n_inf_drawn = sum(math.isinf(p[2]) for p in sets)
    print(f"Draws: {len(sets)} sets ({n_inf_drawn} with K = infinity); a and e redrawn {redraws} times for e a > d.")
    print(f"  no positive parasite-free equilibrium (C0 >= K'): {n_nopfe}")
    print(f"  positive parasite-free equilibrium but R0 <= 1:   {n_r0low}")
    n_kinf = sum(info[i][4] for i, _ in kept)
    print(f"  kept (positive parasite-free equilibrium, R0 > 1): {len(kept)}  (K finite {len(kept) - n_kinf}, K = infinity {n_kinf})")
    r0s = np.array([info[i][3] for i, _ in kept])
    if len(r0s):
        print(f"  R0 of the kept sets: median {np.median(r0s):.3g}, 10th to 90th percentile {np.percentile(r0s, 10):.3g} to {np.percentile(r0s, 90):.3g}")
    print(flush=True)

    # ---------------------------------------------------------------- the sweep
    recs = []
    stopped = False
    t_sweep = time.time()
    with mp.Pool(args.procs, initializer=_quiet) as pool:
        for rec in pool.imap(process_set, kept, chunksize=4):
            recs.append(rec)
            if time.time() - t_sweep > SWEEP_BUDGET_S:
                stopped = True
                break
    sweep_s = time.time() - t_sweep
    errors = [r for r in recs if r["status"] == "error"]
    recs = [r for r in recs if r["status"] != "error"]
    if errors:
        print(f"*** {len(errors)} sets raised an error in the worker and are left out of everything below; the first: "
              f"set {errors[0]['i']}: {errors[0]['error']}")
    print(f"Sweep: {len(recs)} of {len(kept)} kept sets processed in {sweep_s:.0f} s of wall-clock on {args.procs} processes "
          f"({sum(r['seconds'] for r in recs):.0f} s of processor time, {sum(r['sec_int'] for r in recs):.0f} s of it integrating; "
          f"slowest set {max(r['seconds'] for r in recs):.1f} s; {sum(r['nsteps'] >= MAX_STEPS for r in recs)} sets reached the step limit)."
          + ("  STOPPED AT THE BUDGET: the sets processed are the first in draw order." if stopped else ""), flush=True)

    def K_inf(r):
        return info[r["i"]][4]

    def stable(e):
        return e[1] < 0

    classes = (("K finite", lambda r: not K_inf(r)), ("K = infinity", K_inf))
    n_pfe_st = sum(r["pfe_re"] < 0 for r in recs)
    print(f"  parasite-free equilibrium stable to disturbances of predator and prey alone: {frac(n_pfe_st, len(recs))} of the kept sets "
          f"(K finite {frac(sum(r['pfe_re'] < 0 and not K_inf(r) for r in recs), sum(not K_inf(r) for r in recs))})")

    # ---------------------------------------------------------------- what the integrations did
    print("\nIntegration from the parasite-free equilibrium with the parasite added:")
    for label, sel in (("all kept sets", recs),) + tuple((lb, [r for r in recs if fl(r)]) for lb, fl in classes):
        n = len(sel)
        n_set = sum(r["status"] == "settled" for r in sel)
        print(f"  {label}: settled on a positive equilibrium {frac(n_set, n)}; did not settle {n - n_set}")
    settled = [r for r in recs if r["status"] == "settled"]
    n_set_stable = sum(stable(r["eqs"][0]) for r in settled)
    print(f"  of the {len(settled)} settled equilibria, linearly stable by the eigenvalues: {n_set_stable}"
          + (f"  (largest real part among the others: {max(r['eqs'][0][1] for r in settled if not stable(r['eqs'][0])):.2e})"
             if n_set_stable < len(settled) else ""))
    for label, sel in (("parasite-free equilibrium stable", [r for r in recs if r["pfe_re"] < 0]),
                       ("parasite-free equilibrium unstable", [r for r in recs if r["pfe_re"] >= 0])):
        print(f"  {label}: settled {frac(sum(r['status'] == 'settled' for r in sel), len(sel))}")
    uns = [r for r in recs if r["status"] != "settled"]
    print(f"  the {len(uns)} that did not settle, by what the trajectory was doing when it stopped (K finite / K = infinity):")
    kinds = sorted({r["behaviour"] for r in uns}, key=lambda s: -sum(r["behaviour"] == s for r in uns))
    gloss = {"prey collapses": "prey falls below 1e-30 of its reference value (predator-prey swing of extreme size; stopped there)",
             "parasite lost": "a parasite stage falls below 1e-30 of its reference value with predator and prey still present",
             "predator lost": "predator falls below 1e-30 of its reference value",
             "prey escapes": "prey rises above 1e10 of its reference value (escapes the predator; stopped there)",
             "unbounded H": "predator above 1e30 of its reference value",
             "unbounded P": "adult parasites above 1e30 of their reference value",
             "unbounded W": "free-living stages above 1e30 of their reference value",
             "sustained cycle": "cycle with all five variables present, same extremes over the last two stretches (step limit)",
             "oscillation growing": "oscillating with all five present, extremes still widening at the step limit",
             "oscillation shrinking": "oscillating with all five present, extremes still narrowing at the step limit",
             "parasite declining": "parasite declining at the step limit (not yet below 1e-30)",
             "predator declining": "predator declining at the step limit (not yet below 1e-30)",
             "still converging": "closing in on a linearly stable equilibrium that Newton found; not within 1e-6 of it at the step limit",
             "nearly steady": "nearly steady (range below 0.1%) but not within 1e-6 of an equilibrium at the step limit",
             "undetermined": "undetermined", "integration failed": "integration failed"}
    for kd in kinds:
        nf = sum(r["behaviour"] == kd and not K_inf(r) for r in uns)
        ni = sum(r["behaviour"] == kd and K_inf(r) for r in uns)
        print(f"    {nf + ni:6d}  ({nf} / {ni})  {gloss.get(kd, kd)}")
    cyc = [r for r in uns if r["behaviour"] in ("sustained cycle", "oscillation growing", "oscillation shrinking") and "trough" in r]
    if cyc:
        tr = np.array([r["trough"] for r in cyc])
        print(f"    of the {len(cyc)} cycling or oscillating with all five present: lowest density reached by any variable in the last stretch, "
              f"as a power of ten of its reference value: median {np.median(tr):.1f}, quartiles {np.percentile(tr, 25):.1f} and {np.percentile(tr, 75):.1f}; "
              f"every variable stays above 1e-6 of its reference in {int(np.sum(tr > -6))}")
    print(f"    integration time reached, sets that did not settle: median {np.median([r['t_end'] for r in uns]):.3g}; "
          f"sets that settled: median {np.median([r['t_end'] for r in settled]) if settled else float('nan'):.3g} "
          f"(largest {max([r['t_end'] for r in settled]) if settled else float('nan'):.3g})")

    # ---------------------------------------------------------------- Newton from random starts
    n_found = sum(len(r["eqs"]) > 0 for r in uns)
    n_multi = sum(len(r["eqs"]) > 1 for r in uns)
    n_anystable = sum(any(stable(e) for e in r["eqs"]) for r in uns)
    print(f"\nNewton from {N_STARTS} random starts in the {len(uns)} sets that did not settle:")
    print(f"  at least one positive equilibrium found in {n_found} ({n_multi} with more than one); none found in {len(uns) - n_found}")
    print(f"  sets with a linearly stable one among them: {n_anystable}"
          f"  (of which the trajectory was still closing in on it at the step limit: {sum(r['behaviour'] == 'still converging' for r in uns)})")
    bk = {}
    for r in uns:
        if any(stable(e) for e in r["eqs"]):
            bk[r["behaviour"]] = bk.get(r["behaviour"], 0) + 1
    if bk:
        print("    what those trajectories were doing: " + "; ".join(f"{k}: {v}" for k, v in sorted(bk.items(), key=lambda kv: -kv[1])))
    for label, flt in classes:
        sel = [r for r in uns if flt(r)]
        print(f"    {label}: found in {sum(len(r['eqs']) > 0 for r in sel)} of {len(sel)}; with a stable one {sum(any(stable(e) for e in r['eqs']) for r in sel)}")

    # ---------------------------------------------------------------- supplementary scan
    def same(x1, x2):
        return float(np.max(np.abs(np.log(x1 / x2)))) < 1e-6

    n_scan_any = sum(len(r["scan"]) > 0 for r in recs)
    n_scan_multi = sum(len(r["scan"]) > 1 for r in recs)
    miss_by_newton = sum(len(r["eqs"]) == 0 and len(r["scan"]) > 0 for r in recs)
    miss_by_scan = sum(any(not any(same(e[0], s[0]) for s in r["scan"]) for e in r["eqs"]) for r in recs)
    newton_incomplete = sum(len(r["eqs"]) > 0 and any(not any(same(e[0], s[0]) for e in r["eqs"]) for s in r["scan"]) for r in recs)
    print(f"\nS (supplementary, not in the protocol): roots of the one-equation reduction on a grid, {len(recs)} sets")
    print(f"  sets with at least one positive equilibrium by the scan: {n_scan_any} ({n_scan_multi} with more than one)")
    for label, flt in classes:
        sel = [r for r in recs if flt(r)]
        cnt = {}
        for r in sel:
            cnt[len(r["scan"])] = cnt.get(len(r["scan"]), 0) + 1
        print(f"    {label}: number of equilibria per set -> number of sets: " + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items())))
    print(f"  sets where the protocol's method found no equilibrium and the scan found one: {miss_by_newton}")
    print(f"  sets where the protocol's method found some but not all of the scan's: {newton_incomplete}")
    print(f"  sets where the protocol's method found an equilibrium that the scan did not: {miss_by_scan}")

    # ---------------------------------------------------------------- primary equilibrium of each set
    def primary(r):
        if r["status"] == "settled":
            return r["eqs"][0]
        return min(r["eqs"], key=lambda e: e[0][3] + e[0][4])

    prim = [(r, primary(r)) for r in recs if r["eqs"]]
    marginal = sum((e[1] < 0) != (e[2] < 0) for _, e in prim)
    print(f"\nSets with an endemic equilibrium by the protocol's method: {len(prim)} of {len(recs)}"
          f" (K finite {sum(not K_inf(r) for r, _ in prim)}, K = infinity {sum(K_inf(r) for r, _ in prim)})."
          f"  Where a set has several, the one with the smallest C* + I* is used.")
    print(f"  finite-difference and analytic Jacobians disagree on the sign of the largest real part in {marginal} of them;"
          f" smallest |largest real part| / largest modulus: {min(abs(e[1]) / e[3] for _, e in prim):.1e}")

    # ---------------------------------------------------------------- J check
    worst = worst_ev = 0.0
    nj = 0
    for r, e in prim[:200]:
        q = info[r["i"]][0]
        x = [float(v) for v in e[0]]
        Jf = jac_fd(lambda z: rhs(q, z), x, [FD_REL * v for v in x])
        Ja = jac_an(q, x)
        X = np.array(x)
        Sf, Sa = Jf * X[None, :] / X[:, None], Ja * X[None, :] / X[:, None]
        worst = max(worst, float(np.max(np.abs(Sf - Sa)) / np.max(np.abs(Sa))))
        ef, ea = np.linalg.eigvals(Jf), np.linalg.eigvals(Ja)
        worst_ev = max(worst_ev, abs(float(ef.real.max() - ea.real.max())) / float(np.max(np.abs(ea))))
        nj += 1
    report("J finite-difference Jacobian against the analytic one", worst < 1e-5 and nj == 200,
           f"{nj} endemic equilibria, largest relative difference of a scaled entry {worst:.1e}, "
           f"of the largest real part {worst_ev:.1e} (relative to the largest modulus)")

    # ---------------------------------------------------------------- H1
    print("\nH1 (open).  Sets with both a positive parasite-free and a positive endemic equilibrium.")
    print("  'parasite-free stable' = stable to disturbances of predator and prey alone (all kept sets have R0 > 1).")

    def h1_table(label, sel):
        n = len(sel)
        if n == 0:
            print(f"  {label}: no sets")
            return
        ss = sum(r["pfe_re"] < 0 and stable(e) for r, e in sel)
        su = sum(r["pfe_re"] < 0 and not stable(e) for r, e in sel)
        us = sum(r["pfe_re"] >= 0 and stable(e) for r, e in sel)
        uu = sum(r["pfe_re"] >= 0 and not stable(e) for r, e in sel)
        diff = (us - su) / n
        sd = math.sqrt(max(us + su - (us - su) ** 2 / n, 0.0)) / n
        print(f"  {label}: n = {n}")
        print(f"    endemic equilibrium stable:        {frac(ss + us, n)}")
        print(f"    parasite-free equilibrium stable:  {frac(ss + su, n)}")
        print(f"    difference (endemic - parasite-free) {diff:+.3f} +- {sd:.3f} (paired standard error)")
        print(f"    parasite-free stable   -> endemic stable {ss}, endemic unstable {su}")
        print(f"    parasite-free unstable -> endemic stable {us}, endemic unstable {uu}")

    h1_table("all", prim)
    for label, flt in classes:
        h1_table(label, [(r, e) for r, e in prim if flt(r)])
    prim_scan = [(r, min(r["scan"], key=lambda e: e[0][3] + e[0][4])) for r in recs if r["scan"]]
    print("  Supplementary: the same with the scan's equilibria (smallest C* + I* where a set has several):")
    h1_table("all, scan", prim_scan)
    for label, flt in classes:
        h1_table(label + ", scan", [(r, e) for r, e in prim_scan if flt(r)])
    print(f"  Kept sets with no endemic equilibrium found by the protocol's method (outside H1): {sum(len(r['eqs']) == 0 for r in recs)}"
          f"  (K finite {sum(len(r['eqs']) == 0 and not K_inf(r) for r in recs)}, K = infinity {sum(len(r['eqs']) == 0 and K_inf(r) for r in recs)});"
          f" none by the scan: {sum(len(r['scan']) == 0 for r in recs)}"
          f"  (K finite {sum(len(r['scan']) == 0 and not K_inf(r) for r in recs)}, K = infinity {sum(len(r['scan']) == 0 and K_inf(r) for r in recs)})")

    # ---------------------------------------------------------------- H3
    print("\nH3 (open).  Sets with finite K: stability against where the prey is held, x = (C* + I*)/K.")
    fin = [(r, e) for r, e in prim if not K_inf(r)]

    def pos(r, e):
        return (e[0][3] + e[0][4]) / sets[r["i"]][2]

    hi_ = [(r, e) for r, e in fin if pos(r, e) > 0.5]
    lo_ = [(r, e) for r, e in fin if pos(r, e) <= 0.5]
    print(f"    x > 1/2:  stable {frac(sum(stable(e) for _, e in hi_), len(hi_))}")
    print(f"    x <= 1/2: stable {frac(sum(stable(e) for _, e in lo_), len(lo_))}")
    edges = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sel = [(r, e) for r, e in fin if lo_e < pos(r, e) <= hi_e]
        print(f"      {lo_e:.2f} < x <= {hi_e:.2f}: stable {frac(sum(stable(e) for _, e in sel), len(sel))}")
    for label, sel in (("parasite-free equilibrium stable", [(r, e) for r, e in fin if r["pfe_re"] < 0]),
                       ("parasite-free equilibrium unstable", [(r, e) for r, e in fin if r["pfe_re"] >= 0])):
        a_ = [(r, e) for r, e in sel if pos(r, e) > 0.5]
        b_ = [(r, e) for r, e in sel if pos(r, e) <= 0.5]
        print(f"    among sets with the {label}: x > 1/2 stable {frac(sum(stable(e) for _, e in a_), len(a_))}; "
              f"x <= 1/2 stable {frac(sum(stable(e) for _, e in b_), len(b_))}")

    def kprime(r):
        p = sets[r["i"]]
        return p[2] * (1 - p[0] / p[1])

    for label, test in (("C* + I* > K'/2 (peak of the prey's net production, K' = K(1 - d_C/f))",
                         lambda r, e: e[0][3] + e[0][4] > kprime(r) / 2),
                        ("C* + I* > (K' - C50)/2 (peak of the prey isocline, as in H0)",
                         lambda r, e: e[0][3] + e[0][4] > (kprime(r) - C50) / 2)):
        a_ = [(r, e) for r, e in fin if test(r, e)]
        b_ = [(r, e) for r, e in fin if not test(r, e)]
        print(f"    supplementary split (not in the protocol), {label}:")
        print(f"      above: stable {frac(sum(stable(e) for _, e in a_), len(a_))};  below: stable {frac(sum(stable(e) for _, e in b_), len(b_))}")

    # ---------------------------------------------------------------- H2
    print("\nH2 (open).  sigma and beta divided by c (free-living stage c times longer), equilibrium otherwise unchanged.")
    h2 = prim[:2000]
    n_bad_res = n_marg = 0
    worst_res = 0.0
    patterns = []
    for r, e in h2:
        q = info[r["i"]][0]
        pat = ""
        for c in C_GRID:
            qc, xc = scaled_stage(q, e[0], c)
            res = rel_residual(qc, xc)
            worst_res = max(worst_res, res)
            if not res < RES_TOL:
                n_bad_res += 1
            mf, ma, _ = classify(qc, xc)
            n_marg += (mf < 0) != (ma < 0)
            pat += "S" if mf < 0 else "U"
        patterns.append(pat)
    if len(h2) < 2000:
        print(f"  *** only {len(h2)} sets with an endemic equilibrium were available; the protocol asks for 2,000 ***")
    report("H2 pre-check: the equilibrium with W multiplied by c satisfies the equations", n_bad_res == 0,
           f"{len(h2)} sets x {len(C_GRID)} values of c, largest relative residual {worst_res:.1e}, above 1e-9 in {n_bad_res}")
    print(f"      (finite-difference and analytic Jacobians disagree on the sign in {n_marg} of {len(h2) * len(C_GRID)} cases)")

    def shape(pat):
        runs = pat[0]
        for ch in pat[1:]:
            if ch != runs[-1]:
                runs += ch
        return runs

    shape_names = {"S": "stable at every c", "U": "unstable at every c", "SU": "stable when short, unstable when long",
                   "US": "unstable when short, stable when long", "SUS": "stable, unstable, stable (an island of instability)",
                   "USU": "unstable, stable, unstable"}
    for label, flt in classes:
        idx = [j for j, (r, _) in enumerate(h2) if flt(r)]
        n = len(idx)
        print(f"  {label}: {n} sets")
        if n == 0:
            continue
        print("    stable share by c: " + ", ".join(f"c={c:g}: {sum(patterns[j][ci] == 'S' for j in idx) / n:.3f}" for ci, c in enumerate(C_GRID)))
        cnt = {}
        for j in idx:
            cnt[shape(patterns[j])] = cnt.get(shape(patterns[j]), 0) + 1
        for sh, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"    {v:6d}  {shape_names.get(sh, 'more than two changes (' + sh + ')')}")
        nonmono = sum(len(shape(patterns[j])) >= 3 for j in idx)
        restored = sum("US" in shape(patterns[j]) for j in idx)
        ever_u = sum("U" in patterns[j] for j in idx)
        ever_u_short = sum("U" in patterns[j][:-1] for j in idx)
        end_s = sum(("U" in patterns[j][:-1]) and patterns[j][-1] == "S" for j in idx)
        print(f"    non-monotonic (two or more changes of stability along c): {frac(nonmono, n)}")
        print(f"    stability restored by a longer stage somewhere along c (unstable, then stable at a larger c): {frac(restored, n)}")
        print(f"    of the {ever_u_short} sets unstable at some c below 1e4, stable at c = 1e4: {frac(end_s, ever_u_short)}")
        print(f"    of the {ever_u} sets unstable at some c, unstable at c = 1e4: {frac(sum(patterns[j][-1] == 'U' for j in idx if 'U' in patterns[j]), ever_u)}")
    sys.stdout.flush()

    # ---------------------------------------------------------------- T: twelve integrations
    print("\nT (check).  Twelve integrations of the nonlinear model, one on each side of six boundaries found in H2")
    print("  (the first three sets with finite K and the first three with K = infinity that have a change of stability")
    print("  between neighbouring values of c; if one kind has fewer than three, the other kind fills the places).")
    pending, skipped_slow = [], 0
    for j, (r, e) in enumerate(h2):
        pat = patterns[j]
        ch = [ci for ci in range(len(C_GRID) - 1) if pat[ci] != pat[ci + 1]]
        if not ch:
            continue
        ci = ch[0]
        q = info[r["i"]][0]
        sides = []
        slow = False
        for cc in (C_GRID[ci], C_GRID[ci + 1]):
            qc, xc = scaled_stage(q, e[0], cc)
            w = np.linalg.eigvals(jac_an(qc, xc))
            z = w[int(np.argmax(w.real))]
            if 6.0 * abs(z.imag) / (2 * math.pi * abs(z.real)) > 3000:
                slow = True
            sides.append((qc, xc, bool(z.real < 0), cc))
        if slow:
            skipped_slow += 1
            continue
        pending.append((K_inf(r), r["i"], sides))
    chosen = [b for b in pending if not b[0]][:3] + [b for b in pending if b[0]][:3]
    if len(chosen) < 6:
        chosen += [b for b in pending if b not in chosen][:6 - len(chosen)]
    tasks = [(qc, xc, st) for _, _, sides in chosen for qc, xc, st, _ in sides]
    outs = []
    if tasks:
        with mp.Pool(min(args.procs, len(tasks)), initializer=_quiet) as pool:
            outs = pool.map(confirm_task, tasks, chunksize=1)
    k = n_ok = 0
    for kinf, i_set, sides in chosen:
        for qc, xc, st, cc in sides:
            zr, zi, T, early, late, ok = outs[k]
            k += 1
            n_ok += ok
            print(f"      set {i_set} ({'K = infinity' if kinf else 'K finite'}), c = {cc:g}: leading eigenvalue {zr:+.4g}{zi:+.4g}i, "
                  f"time {T:.4g}, disturbance {early:.2e} -> {late:.2e}, predicted {'to shrink' if st else 'to grow'}: "
                  f"{'as predicted' if ok else 'NOT as predicted'}")
    report("T nonlinear integrations either side of a boundary", n_ok == len(tasks) and len(tasks) == 12,
           f"{n_ok} of {len(tasks)} as predicted ({len(pending)} sets with a boundary available, "
           f"{sum(b[0] for b in pending)} of them with K = infinity; {skipped_slow} more skipped because "
           f"following them would take over 3,000 oscillations)")

    # ---------------------------------------------------------------- H4
    print("\nH4 (open).  The trophic cycle against a direct cycle matched for R0 and for the mean time round the cycle.")
    t4 = time.time()
    res4 = []
    for r, e in prim:
        q, H0, C0, R0, _ = info[r["i"]]
        try:
            res4.append((r, e, direct_match(q, e[0], R0, H0, C0)))
        except Exception as ex:  # noqa: BLE001
            res4.append((r, e, {"status": "error in the matching: " + type(ex).__name__, "roots": 0, "share_prey": float("nan")}))
    cnt = {}
    for _, _, d in res4:
        cnt[d["status"]] = cnt.get(d["status"], 0) + 1
    print(f"  {len(res4)} sets with an endemic equilibrium:")
    for k_, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {v:6d}  {k_}")
    matched = [(r, e, d) for r, e, d in res4 if d["status"] == "matched"]
    print(f"  matched sets with more than one matching direct cycle (the one with the smallest prey density used): {sum(d['roots'] > 1 for _, _, d in matched)}")
    if matched:
        print(f"  largest mismatch of R0 or of the time among matched sets {max(d['err'] for _, _, d in matched):.1e}; "
              f"largest relative residual of a direct equilibrium {max(d['res'] for _, _, d in matched):.1e}")
        print(f"  R0: median {np.median([info[r['i']][3] for r, _, _ in matched]):.3g} in matched sets, "
              f"{np.median([info[r['i']][3] for r, _, d in res4 if d['status'] != 'matched']):.3g} in the others; "
              f"share of the trophic cycle's time spent in the prey: median {np.median([d['share_prey'] for _, _, d in matched]):.3f} in matched sets, "
              f"{np.median([d['share_prey'] for _, _, d in res4 if d['status'] != 'matched']):.3f} in the others")

    def h4_table(label, sel):
        n = len(sel)
        if n == 0:
            print(f"  {label}: no matched sets")
            return
        ss = sum(stable(e) and d["re"] < 0 for _, e, d in sel)
        su = sum(stable(e) and not d["re"] < 0 for _, e, d in sel)
        us = sum(not stable(e) and d["re"] < 0 for _, e, d in sel)
        uu = sum(not stable(e) and not d["re"] < 0 for _, e, d in sel)
        diff = (su - us) / n
        sd = math.sqrt(max(su + us - (su - us) ** 2 / n, 0.0)) / n
        print(f"  {label}: n = {n}")
        print(f"    trophic cycle stable: {frac(ss + su, n)}")
        print(f"    direct cycle stable:  {frac(ss + us, n)}")
        print(f"    difference (trophic - direct) {diff:+.3f} +- {sd:.3f} (paired standard error)")
        print(f"    both stable {ss}; trophic stable, direct unstable {su}; trophic unstable, direct stable {us}; both unstable {uu}")

    h4_table("all matched", matched)
    for label, flt in classes:
        h4_table(label, [m_ for m_ in matched if flt(m_[0])])
    print(f"  For comparison, trophic cycle stable among the sets that could not be matched: "
          f"{frac(sum(stable(e) for _, e, d in res4 if d['status'] != 'matched'), len(res4) - len(matched))}")
    print(f"      (H4 took {time.time() - t4:.0f} s)")

    print()
    print(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')} (clock); {time.time() - t_start:.0f} s of wall-clock in all")
    print("ALL CHECKS PASSED (H0, R0, J, H2 pre-check, T).  H1 to H4 are open: they are reported above as counts, not as pass or fail."
          if not FAILS else f"FAILED CHECKS: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
