"""Checks made after referee report 14 on notes/mechanism-temperature-incubation.md.

Written 14 November 2026, after reading studies/referee-notes-14/referee_report_14.md in full and before running
anything below.  No protocol file may be written for this step, so the claims, tolerances and pass rules are
fixed HERE, in this docstring, before the first run.  The scripts whose outputs are already saved are not
edited; this script imports checks/r15a_lib.py and checks/r10b_real_systems_stay_shape.py.

Notation.  One fluctuating transfer, lognormal host of memory phi (theta = phi/(1 - phi)).  L is the time between
two uses of that transfer by a line of descent at the threshold, T = E[L], R = E[phi^L].  The second-order
loss per unit variance is G = (1 + 2 theta)/(2 T) - R/(1 - R).

Checks (pass rule in brackets):
  AR1  Referee's counterexample to "any distribution of stays" (report, point 3): exposed geometric stage of
       mean 2, then a stage that ends at day 1 with probability 0.99 and otherwise at day 101; phi = 0.99.
       My closed form, my 102-state matrix with the resolvent coding, and with the lag-sum coding.
       [closed form within 1e-9 of the referee's printed -2.09236743618; matrix (resolvent) within 1e-8
       relative of the closed form; lag-sum within 1e-8 relative of the resolvent.]
  AR2  Algebra of the gain condition: G < 0 exactly when R > (1 + 2 theta)/(1 + 2 theta + 2 T).
       [same sign in 5,000 random (T, R, phi) with |R - threshold| > 1e-9.]
  AR3  Serial chains of geometric sub-stages and fixed delays can never meet it for phi >= 0:
       R <= theta/(theta + T) < threshold, and G >= 1/(2T).  [20,000 random chains of 1 to 8 sub-stages, stays
       1 to 300 steps, phi from 0 to 0.9999: no violation beyond 1e-12.]
  AR4  Reported, no criterion: the values of phi for which the counterexample of AR1 gives a gain, and its
       slow-host limit (1 - Var L/T^2)/2.
  AR5  Dormancy mixture.  One stage A, geometric with mean a, whose exit follows the host; a fraction f of
       the leavers then pass through a dormant stay, geometric with mean 100 steps, before returning to A;
       the rest return at once.  L = G_a + B G_100, B = 1 with probability f.
       (i) closed form against my two-state matrix formula [1e-10 relative, 60 parameter sets];
       (ii) reported: the smallest and largest f that give a gain, by phi, for a = 2 and a = 4; the range of
       phi that gives a gain, by f; the limit of the smallest f as phi -> 1, which by hand is the smaller
       root of 20000 f^2 - (19900 - 200 a) f + a = 0 [numerical value at phi = 0.999999 within 1% of it].
  AR6  One unexpanded computation on each side of the second-order boundary, at sigma = 0.3: the dormancy
       mixture with a = 2, phi = 0.9 (second-order boundary near f = 0.0012, computed in AR5), at f = 0.0005
       (second order: a loss) and f = 0.005 (second order: a gain).  Simulation, two code paths (explicit
       matrices, r15a_lib.sim_B; a stage-wise simulator written here).
       [the two paths agree within 3.5 combined standard errors; D > +3 se at f = 0.0005 and D < -3 se at
       f = 0.005 on the stage-wise path.  If a sign is not as second order predicts, that is reported as a
       result about the boundary, not as a harness failure.]
  AR7  Referee's counterexample to "a stage with a proportional response is a sheltered stage" (point 5):
       stays (3.7, 4.4), phi = 0.76, stage 0 with (pi, chi) = (0.2, 0), stage 1 with (0, 0.4) or with its clock
       frozen (0, 0).  My closed form and my matrix formula.
       [each within 5e-9 of the referee's printed +0.008540788 and -0.018043202; closed form and matrix
       within 1e-10 of each other.]
  AR8  The endpoint of Proposition D (point 9): two fixed steps and a stay of 6 as a chain of m.  At phi = 0
       the formula gives 1/16 for m = 1, 2, 3 and also for m = 6, where the ring is not primitive and the
       true coefficient is 1/2.  [matrix formula = 1/16 to 1e-12 for m = 1, 2, 3; second eigenvalue of the
       mean matrix has modulus 1 to 1e-12 for m = 6.]
  AR9  The integer rule for a fixed delay with death (point 13): at second order without memory one more step
       pays exactly when mu < sigma^2/(2(a + b)(a + b + 1)), and no whole delay pays when
       mu >= sigma^2/(2a(a + 1)).  [the integer maximiser of -mu b - sigma^2/(2(a + b)) agrees with the rule
       in 2,000 random (a, sigma, mu).]  And for (a, sigma, mu) = (3.2, 0.4, 0.09) the geometric-stay quadratic
       has no positive root and the margin falls from b = 1 on.  [both roots negative; N2 decreasing on a grid.]
  AR10 The referee's unexpanded losses for a geometric stage of 4 steps and a fixed delay b (point 2):
       sigma = 1, phi = 0.9, b = 72, 120, 200; sigma = 0.3, phi = 0.9, b = 72, 120.  My delay simulator
       (r10b's), burn-in 8 relaxation times, at least 30 relaxation times of run.
       [each within 4 combined standard errors of the referee's printed value.  A miss is reported as "not
       reproduced" in the note; it does not stop the run.]

Amendments after the first run (started 12:50:06 by the system clock; its output is kept as
results/r15a-after-referee_attempt1_20261114.txt).  Written at 12:52 before the second run.
  1. AR3 and AR5(i) failed on rounding, not on the claims.  G is the difference of two terms that reach 10^4
     at phi = 0.9999, and the tolerances (1e-12 absolute, 1e-10 relative to G) did not allow for that; the
     misses were 5.7e-12 and 1.7e-10.  Both tolerances are now relative to the sum of the two terms.  The
     inequalities R <= theta/(theta + T) and R < threshold, which are the claim of AR3, had passed.
  2. AR6 gave a RESULT, not a harness fault: at f = 0.0005 (phi = 0.9, sigma = 0.3) the unexpanded loss is
     negative, a gain, where second order predicts a loss; the two code paths agree.  As the rule above says,
     this is reported as a result about the boundary.  AR6 now asserts only that the two paths agree, and
     prints whether each sign is as second order predicts.
  3. Added, with rules fixed before the second run, to find a loss side at sigma = 0.3 and to see whether the
     second-order boundary is right at smaller noise:
     AR6b  phi = 0.5, sigma = 0.3 (second-order boundary f = 0.0105): f = 0.003 and f = 0.03, both paths.
           [paths agree within 3.5 combined se; signs reported as predicted or not.]
     AR6c  phi = 0.9, sigma = 0.1: f = 0.0005 and 0.005.  [reported: signs as predicted or not.]
     AR6d  phi = 0.9, sigma = 0.3: f = 0.00002 and 0.0001, and the exact one-stage loss at f = 0.  [reported.]
     AR5 also prints the smallest memory at which any f gives a gain.
  4. Written at 12:53, after the second run (which passed; its numbers are unchanged in the final output, the
     seeds being fixed).  AR6c showed a gain at f = 0.0005 even at sigma = 0.1, where second order predicts a
     loss.  Added, to see whether the second-order sign is recovered at still smaller noise:
     AR6e  phi = 0.9, f = 0.0005, sigma = 0.05 and 0.03, 400 populations x 250,000 steps.
           [reported: the second-order sign counts as recovered at a sigma if D > +3 se there.]

Run: python3 -I checks/r15a_after_referee.py    (three processes, about one minute)
"""
from __future__ import annotations

import math
import os
import sys
import time

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import multiprocessing as mp

import numpy as np
from scipy.optimize import brentq

import r15a_lib as L
import r10b_real_systems_stay_shape as B

REFEREE = {"AR1": -2.09236743618, "AR7_frozen": 0.008540788, "AR7_clock": -0.018043202,
           # (sigma, b): (value, standard error), report 14, point 2 / point 12 and its output file
           (1.0, 72): (0.208386, 0.000416), (1.0, 120): (0.186582, 0.000393), (1.0, 200): (0.173432, 0.001046),
           (0.3, 72): (0.012953, 0.000137), (0.3, 120): (0.009485, 0.000338)}
ONE_STAGE = {1.0: None, 0.3: None}          # exact one-stage loss for a = 4, filled in main()


def G_of(T, R, phi):
    th = phi / (1 - phi)
    return (1 + 2 * th) / (2 * T) - R / (1 - R)


def threshold(T, phi):
    th = phi / (1 - phi)
    return (1 + 2 * th) / (1 + 2 * th + 2 * T)


# ----------------------------------------------------------------------------- AR1, AR4: the referee's stage
def counter_matrix():
    """States: 0 = exposed stage A (geometric, mean 2); 1..101 = day of the second stage.  Column = from."""
    n = 102
    A = np.zeros((n, n))
    A[0, 0] = 0.5
    A[1, 0] = 0.5                 # the fluctuating transfer
    A[0, 1] = 0.99                # ends at day 1 with probability 0.99
    A[2, 1] = 0.01
    for k in range(2, 101):
        A[k + 1, k] = 1.0
    A[0, 101] = 1.0               # otherwise ends at day 101
    dA = np.zeros((n, n))
    dA[1, 0] = 0.5
    return A, dA


def counter_R(phi):
    return phi / (2 - phi) * (0.99 * phi + 0.01 * phi ** 101)


# ----------------------------------------------------------------------------- AR5, AR6: the dormancy mixture
def dorm_RT(a, f, phi, d=100.0):
    th = phi / (1 - phi)
    R = (th / (th + a)) * (1 - f + f * th / (th + d))
    return R, a + d * f


def dorm_matrix(a, f, d=100.0):
    """States (A, dormant).  rest: staying and the return from dormancy; scal: the two parts of the exit from A."""
    sa, sd = 1 - 1 / a, 1 - 1 / d
    rest = np.array([[sa, 1 - sd], [0.0, sd]])
    scal = np.array([[(1 - sa) * (1 - f), 0.0], [(1 - sa) * f, 0.0]])
    return rest, scal


def dorm_matrix_G(a, f, phi, d=100.0):
    rest, scal = dorm_matrix(a, f, d)
    A0 = rest + scal
    lam, v, w = L.perron(A0)
    Tgen = lam / float(w @ scal @ v)
    return Tgen * L.cpp_resolvent(A0, scal, phi), Tgen


def f_range(a, phi, d=100.0):
    """Smallest and largest f in (0, 1) with a gain at second order, or None."""
    g = lambda f: dorm_RT(a, f, phi, d)[0] - threshold(a + d * f, phi)
    fs = np.concatenate([10.0 ** np.linspace(-7, -2, 400), np.linspace(0.0101, 0.9999, 2000)])
    vals = np.array([g(f) for f in fs])
    pos = np.nonzero(vals > 0)[0]
    if pos.size == 0:
        return None
    i, j = pos[0], pos[-1]
    lo = brentq(g, fs[i - 1], fs[i], xtol=1e-14) if i > 0 else fs[0]
    hi = brentq(g, fs[j], fs[j + 1], xtol=1e-12) if j < fs.size - 1 else fs[-1]
    return lo, hi


def phi_range(a, f, d=100.0):
    g = lambda phi: dorm_RT(a, f, phi, d)[0] - threshold(a + d * f, phi)
    ps = np.concatenate([np.linspace(0.0005, 0.99, 1980), 1 - 10.0 ** np.linspace(-2, -7, 200)])
    vals = np.array([g(p) for p in ps])
    pos = np.nonzero(vals > 0)[0]
    if pos.size == 0:
        return None
    i, j = pos[0], pos[-1]
    lo = brentq(g, ps[i - 1], ps[i], xtol=1e-12) if i > 0 else ps[0]
    return lo, (ps[j] if j == ps.size - 1 else brentq(g, ps[j], ps[j + 1], xtol=1e-12))


def sim_dorm(a, f, sigma, phi, ys, M, T, burn, seed, d=100.0, chunk=400):
    """Stage-wise simulator of the dormancy mixture (second code path)."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ey = np.exp(ys)[:, None]
    sa, sd = 1 - 1 / a, 1 - 1 / d
    zlast = rng.standard_normal(M)
    xa = np.full((ys.size, M), 0.5)
    xd = np.full((ys.size, M), 0.5)
    acc = np.zeros((ys.size, M))
    done, total = 0, burn + T
    while done < total:
        C = min(chunk, total - done)
        z, zlast = L.ar1_block(rng, zlast, phi, C, M)
        H = np.exp(sigma * z - sigma * sigma / 2) * (1 - sa)
        tots = np.empty((C, ys.size, M))
        for t in range(C):
            out = ey * H[t] * xa
            na = sa * xa + (1 - f) * out + (1 - sd) * xd
            nd = sd * xd + f * out
            tot = na + nd
            tots[t] = tot
            xa = na / tot
            xd = nd / tot
        lo = max(burn - done, 0)
        if lo < C:
            acc += np.log(tots[lo:]).sum(axis=0)
        done += C
    return acc / T


def task_dorm(a):
    t0 = time.time()
    aa, f, sigma, phi, M, T, burn = a["a"], a["f"], a["sigma"], a["phi"], a["M"], a["T"], a["burn"]
    R, TT = dorm_RT(aa, f, phi)
    guess = sigma ** 2 * G_of(TT, R, phi)
    if a["path"] == "stage":
        run = lambda ys, T_, b_, sd: sim_dorm(aa, f, sigma, phi, ys, M, T_, b_, sd)
    else:
        rest, scal = dorm_matrix(aa, f)

        def parts(z):
            h = np.exp(sigma * z - sigma * sigma / 2)
            return np.broadcast_to(rest, (M, 2, 2)), scal[None] * h[:, None, None]

        run = lambda ys, T_, b_, sd: L.sim_B(parts, 2, phi, ys, M, T_, b_, sd)
    D, se, Dm, runs = L.find_root(run, guess, max(0.02, 0.5 * abs(guess)), M, T, burn, a["seed"])
    return a["key"], dict(D=D, se=se, guess=guess, secs=time.time() - t0)


def task_delay(a):
    """AR10: geometric stage of 4 steps and a fixed delay b, with the delay simulator of run R10b."""
    t0 = time.time()
    b, sigma, phi, M = a["b"], a["sigma"], a["phi"], a["M"]
    subs = [0.75] + [0.0] * b
    tau = B.relax_time(subs)
    T = int(max(math.ceil(30 * tau), a["Tmin"]))
    T += T % 2
    burn = max(2000, int(math.ceil(8 * tau)))
    store = {}

    def run(ys, T_, b_, sd):
        r1, r2 = B.chain_sim([0.75], b, sigma, phi, ys, M, T_, b_, sd)
        store["last"] = (r1, r2)
        return 0.5 * (r1 + r2)

    D, se, Dm, runs = B.root(run, sigma ** 2 / 2, M, T, burn, a["seed"])
    r1, r2 = store["last"]
    dd = r2[1] - r1[1]
    z = float(dd.mean() / (dd.std(ddof=1) / math.sqrt(M)))
    so, TT = B.second_order(subs, sigma, phi)
    return a["key"], dict(D=D, se=se, z=z, tau=tau, T=T, burn=burn, so=so, secs=time.time() - t0)


def run_task(a):
    return (task_dorm if a["fn"] == "dorm" else task_delay)(a)


def chk(fails, name, ok, detail):
    if not ok:
        fails.append(name)
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def main():
    t_start = time.time()
    fails = []
    print("R15a after referee report 14.  Claims and tolerances: the docstring of checks/r15a_after_referee.py.")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (system clock).  ± is one standard error.  G = second-order loss per unit variance.")

    tasks = []
    seed = 21_000_000
    for f in (0.0005, 0.005):
        for path, M, T in (("stage", 400, 60000), ("matrix", 200, 20000)):
            seed += 1000
            tasks.append(dict(fn="dorm", key=("AR6", f, path), a=2.0, f=f, sigma=0.3, phi=0.9, M=M, T=T, burn=3000, path=path, seed=seed))
    seed2 = 22_000_000
    for f in (0.003, 0.03):                                       # AR6b
        for path, M, T in (("stage", 400, 60000), ("matrix", 200, 20000)):
            seed2 += 1000
            tasks.append(dict(fn="dorm", key=("AR6b", f, path), a=2.0, f=f, sigma=0.3, phi=0.5, M=M, T=T, burn=3000, path=path, seed=seed2))
    for f in (0.0005, 0.005):                                     # AR6c
        seed2 += 1000
        tasks.append(dict(fn="dorm", key=("AR6c", f, "stage"), a=2.0, f=f, sigma=0.1, phi=0.9, M=400, T=60000, burn=3000, path="stage", seed=seed2))
    for f in (0.00002, 0.0001):                                   # AR6d
        seed2 += 1000
        tasks.append(dict(fn="dorm", key=("AR6d", f, "stage"), a=2.0, f=f, sigma=0.3, phi=0.9, M=400, T=60000, burn=3000, path="stage", seed=seed2))
    for sg in (0.05, 0.03):                                       # AR6e
        seed2 += 1000
        tasks.append(dict(fn="dorm", key=("AR6e", sg, "stage"), a=2.0, f=0.0005, sigma=sg, phi=0.9, M=400, T=250000, burn=3000, path="stage", seed=seed2))
    for sigma, bs in ((1.0, (72, 120, 200)), (0.3, (72, 120))):
        for b in bs:
            seed += 1000
            tasks.append(dict(fn="delay", key=("AR10", sigma, b), b=b, sigma=sigma, phi=0.9, M=100, Tmin=400000, seed=seed))
    tasks.sort(key=lambda a: -(a.get("b", 0)))
    pool = mp.Pool(3)
    pending = pool.imap_unordered(run_task, tasks)

    # ---- AR1
    A, dA = counter_matrix()
    phi = 0.99
    closed = G_of(4.0, counter_R(phi), phi)
    lam, v, w = L.perron(A)
    Tgen = lam / float(w @ dA @ v)
    mres = Tgen * L.cpp_resolvent(A, dA, phi)
    mlag = Tgen * L.cpp_lagsum(A, [(1, 0, 0.5)], [1.0], phi, nlag=200000, tol=1e-18)
    ok = abs(closed - REFEREE["AR1"]) <= 1e-9 and abs(mres - closed) <= 1e-8 * abs(closed) and abs(mlag - mres) <= 1e-8 * abs(mres)
    print("\n" + chk(fails, "AR1 the referee's counterexample (geometric 2; then day 1 with probability 0.99, else day 101), phi = 0.99", ok,
                     f"closed form {closed:.11f}; referee {REFEREE['AR1']:.11f}; my 102-state matrix, resolvent {mres:.11f}, lag sum {mlag:.11f}; "
                     f"dominant eigenvalue {lam:.12f}, generation time {Tgen:.10f}"))

    # ---- AR2
    rng = np.random.default_rng(2114)
    bad = 0
    for _ in range(5000):
        T = float(np.exp(rng.uniform(0, math.log(500))))
        phi_ = float(rng.uniform(0.0, 0.9999))
        R = float(rng.uniform(0, 0.9999))
        thr = threshold(T, phi_)
        if abs(R - thr) > 1e-9:
            bad += (G_of(T, R, phi_) < 0) != (R > thr)
    print(chk(fails, "AR2 G < 0 exactly when R > (1 + 2 theta)/(1 + 2 theta + 2 T)", bad == 0, f"5,000 random (T, R, phi): {bad} disagreements of sign"))

    # ---- AR3
    worst_R = worst_thr = worst_G = -math.inf
    for _ in range(20000):
        n = int(rng.integers(1, 9))
        stays = np.exp(rng.uniform(0, math.log(300), n))
        fixed = rng.uniform(size=n) < 0.3
        stays[fixed] = 1.0
        if np.all(fixed):
            stays[0] = 1.0 + float(rng.uniform(0.01, 5))          # keep the ring primitive
        phi_ = float(rng.uniform(0, 0.9999)) if rng.uniform() > 0.05 else 0.0
        T = float(stays.sum())
        if phi_ == 0:
            R, geo = 0.0, 0.0
        else:
            th = phi_ / (1 - phi_)
            R = float(np.prod(1 / (1 + stays / th)))
            geo = th / (th + T)
        thr = threshold(T, phi_)
        worst_R = max(worst_R, R - geo)
        worst_thr = max(worst_thr, R - thr)
        th_ = phi_ / (1 - phi_)
        scale = (1 + 2 * th_) / (2 * T) + R / (1 - R) + 1.0
        worst_G = max(worst_G, (1 / (2 * T) - G_of(T, R, phi_)) / scale)
    ok = worst_R <= 1e-12 and worst_thr < 0 and worst_G <= 1e-12
    print(chk(fails, "AR3 serial chains of geometric sub-stages and fixed steps never meet the gain condition (phi >= 0)", ok,
              f"20,000 random chains: largest R - theta/(theta + T) = {worst_R:.1e}; largest R - threshold = {worst_thr:.2e} (negative: never met); "
              f"largest (1/(2T) - G) relative to the size of the two terms of G = {worst_G:.1e}"))

    # ---- AR4
    g = lambda p: counter_R(p) - threshold(4.0, p)
    ps = np.concatenate([np.linspace(0.001, 0.99, 990), 1 - 10.0 ** np.linspace(-2, -7, 100)])
    vals = [g(p) for p in ps]
    cross = [brentq(g, ps[i], ps[i + 1], xtol=1e-13) for i in range(len(ps) - 1) if vals[i] * vals[i + 1] < 0]
    varL, EL = 2.0 + 99.0, 4.0
    print("AR4 (reported) the same example at other memories: G changes sign at phi = " + ", ".join(f"{c:.5f}" for c in cross) +
          f"; G at phi = 0.5, 0.9, 0.99, 0.999: " + ", ".join(f"{G_of(4.0, counter_R(p), p):+.4f}" for p in (0.5, 0.9, 0.99, 0.999)) +
          f"; slow-host limit (1 - Var L/T^2)/2 = {(1 - varL / EL ** 2) / 2:+.4f} (Var L = {varL:.0f}, T = {EL:.0f})")

    # ---- AR5
    worst = 0.0
    for a_ in (1.5, 2.0, 4.0, 10.0):
        for f in (0.0003, 0.003, 0.03, 0.3, 0.9):
            for phi_ in (0.3, 0.9, 0.999):
                R, TT = dorm_RT(a_, f, phi_)
                gm, Tg = dorm_matrix_G(a_, f, phi_)
                th_ = phi_ / (1 - phi_)
                scale = (1 + 2 * th_) / (2 * TT) + R / (1 - R)
                worst = max(worst, abs(gm - G_of(TT, R, phi_)) / scale, abs(Tg - TT) / TT)
    print(chk(fails, "AR5 dormancy mixture: closed form with R = E[phi^L] = two-state matrix formula", worst < 1e-10, f"60 parameter sets, largest difference relative to the size of the two terms {worst:.1e}"))
    print("AR5 (reported) dormancy of mean 100 steps reached by a fraction f of the leavers of a geometric stage of mean a.  Second order.")
    lim_ok = True
    for a_ in (2.0, 4.0):
        qa, qb, qc = 20000.0, -(19900.0 - 200.0 * a_), a_
        flim = (-qb - math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        fnum = f_range(a_, 0.999999)[0]
        lim_ok &= abs(fnum - flim) <= 0.01 * flim
        print(f"      a = {a_:g}: fractions f that give a gain (smallest to largest), by phi:")
        for phi_ in (0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999):
            fr = f_range(a_, phi_)
            print(f"        phi = {phi_:<7}: " + ("none" if fr is None else f"{fr[0]:.3e} to {fr[1]:.4f}"))
        print(f"        limit of the smallest f as phi -> 1 (hand: smaller root of 20000 f^2 - (19900 - 200a) f + a = 0): {flim:.4e}; numerically at phi = 0.999999: {fnum:.4e}")
        print(f"      a = {a_:g}: memories phi that give a gain, by f (the upper end is the end of the grid, phi = 0.9999999):")
        best = (1.0, None)
        for f in (0.0001, 0.0002, 0.001, 0.01, 0.05, 0.2, 0.5, 0.9):
            pr = phi_range(a_, f)
            print(f"        f = {f:<7}: " + ("none" if pr is None else f"phi from {pr[0]:.4f} to {pr[1]:.7f}"))
        for f in np.linspace(0.02, 0.98, 97):
            pr = phi_range(a_, float(f))
            if pr is not None and pr[0] < best[0]:
                best = (pr[0], float(f))
        print(f"        smallest memory at which some f gives a gain (f on a grid of 0.01): phi = {best[0]:.4f}, at f = {best[1]:.2f}")
    print(chk(fails, "AR5 the smallest f as phi -> 1 is the root of the quadratic", lim_ok, "a = 2 and a = 4, within 1%"))

    # ---- AR7
    taus = np.array([3.7, 4.4])
    ph = 0.76
    frozen = L.gamma0_clock(taus, {0: (0.2, 0.0)}, ph)
    clock = L.gamma0_clock(taus, {0: (0.2, 0.0), 1: (0.0, 0.4)}, ph)

    def matrix_two(expo):
        s, g = 1 - 1 / taus, 1 / taus
        A = np.zeros((2, 2))
        dA_ = np.zeros((2, 2))
        for k in range(2):
            A[k, k] += s[k]
            A[(k + 1) % 2, k] += g[k]
        for k, (pi, chi) in expo.items():
            dA_[k, k] += -g[k] * chi
            dA_[(k + 1) % 2, k] += g[k] * (pi + chi)
        return taus.sum() * L.cpp_resolvent(A, dA_, ph)
    mf, mc = matrix_two({0: (0.2, 0.0)}), matrix_two({0: (0.2, 0.0), 1: (0.0, 0.4)})
    ok = abs(frozen - REFEREE["AR7_frozen"]) <= 5e-9 and abs(clock - REFEREE["AR7_clock"]) <= 5e-9 and abs(mf - frozen) <= 1e-10 and abs(mc - clock) <= 1e-10
    print(chk(fails, "AR7 the referee's counterexample: a stage whose completion probability is constant but whose clock responds", ok,
              f"clock of stage 1 frozen: closed form {frozen:+.9f}, matrix {mf:+.9f}, referee {REFEREE['AR7_frozen']:+.9f}; "
              f"clock responding: closed form {clock:+.9f}, matrix {mc:+.9f}, referee {REFEREE['AR7_clock']:+.9f}"))

    # ---- AR8
    vals8 = []
    for m in (1, 2, 3):
        subs = [0.0, 0.0] + [1 - m / 6.0] * m
        vals8.append(B.matrix_second_order(subs, 1.0, 0.0))
    ev = np.sort(np.abs(np.linalg.eigvals(B.ring_matrix([0.0] * 8))))[::-1]
    formula6 = B.second_order([0.0] * 8, 1.0, 0.0)[0]
    ok = all(abs(x - 1 / 16) < 1e-12 for x in vals8) and abs(ev[1] - 1) < 1e-12 and abs(formula6 - 1 / 16) < 1e-12
    print(chk(fails, "AR8 the endpoint of Proposition D: two fixed steps and a stay of 6 as a chain of m, phi = 0", ok,
              f"matrix formula for m = 1, 2, 3: {vals8[0]:.6f}, {vals8[1]:.6f}, {vals8[2]:.6f} (1/16 = 0.0625); for m = 6 the expression gives {formula6:.6f} "
              f"but the ring is not primitive (second eigenvalue modulus {ev[1]:.12f}); the true coefficient there is 1/2"))

    # ---- AR9
    bad = 0
    for _ in range(2000):
        a_ = float(np.exp(rng.uniform(0, math.log(30))))
        sg = float(rng.uniform(0.2, 2.0))
        mu = float(np.exp(rng.uniform(math.log(1e-5), math.log(0.5))))
        bs = np.arange(0, 20001)
        N = -mu * bs - sg ** 2 / (2 * (a_ + bs))
        arg = int(np.argmax(N))
        rule = 0
        while mu < sg ** 2 / (2 * (a_ + rule) * (a_ + rule + 1)):
            rule += 1
        none_rule = mu >= sg ** 2 / (2 * a_ * (a_ + 1))
        bad += (arg != rule) or ((arg == 0) != none_rule)
    a_, sg, mu = 3.2, 0.4, 0.09
    eps = math.expm1(mu)
    c = a_ * eps + 1 + eps
    qa, qb, qc = 2 * eps * c * c, 4 * eps * a_ * c - sg ** 2 * (1 + eps) * eps, 2 * eps * a_ * a_ - sg ** 2 * (1 + eps)
    disc = qb * qb - 4 * qa * qc
    roots = [(-qb + s_ * math.sqrt(disc)) / (2 * qa) for s_ in (1, -1)] if disc >= 0 else []
    N2g = lambda b: -math.log1p(b * eps) - sg ** 2 / (2 * (a_ + b * (1 + eps) / (1 + b * eps)))
    grid = np.linspace(1, 200, 2000)
    dec = bool(np.all(np.diff([N2g(b) for b in grid]) < 0))
    ok = bad == 0 and all(r < 0 for r in roots) and dec
    print(chk(fails, "AR9 fixed delay with death, second order, no memory: the rule for whole steps; and a geometric case with no positive root", ok,
              f"2,000 random (a, sigma, mu): {bad} disagreements between the integer maximum and the rule mu < sigma^2/(2(a+b)(a+b+1)); "
              f"(3.2, 0.4, 0.09): roots of the quadratic {', '.join(f'{r:.3f}' for r in roots) if roots else 'none real'}, margin decreasing from b = 1: {dec}"))

    # ---- simulations
    res = dict(pending)
    pool.close()
    pool.join()
    hz = lambda s: np.exp(s * L.ZG - s * s / 2)
    for s in (1.0, 0.3):
        ONE_STAGE[s] = L.one_stage_exact(np.full_like(L.ZG, 0.75), 0.25 * hz(s))

    print("\nAR6 dormancy mixture, a = 2, dormancy 100 steps, phi = 0.9, sigma = 0.3: unexpanded loss on each side of the second-order boundary")
    fb = f_range(2.0, 0.9)
    print(f"      second-order boundary: gain for f between {fb[0]:.5f} and {fb[1]:.4f}")
    def show(tag, f, want, both=True):
        rs = res[(tag, f, "stage")]
        sign_ok = (rs["D"] > 3 * rs["se"]) if want > 0 else (rs["D"] < -3 * rs["se"])
        txt = f"      f = {f}: second order {rs['guess']:+.5f};  stage-wise simulator {rs['D']:+.5f} ± {rs['se']:.5f}"
        z = 0.0
        if both:
            rm = res[(tag, f, "matrix")]
            z = (rs["D"] - rm["D"]) / math.hypot(rs["se"], rm["se"])
            txt += f";  explicit matrices {rm['D']:+.5f} ± {rm['se']:.5f} ({z:+.1f} combined se)"
        print(txt + f";  second order predicts a {'loss' if want > 0 else 'gain'}: {'as predicted' if sign_ok else 'NOT as predicted'}")
        return abs(z) <= 3.5, sign_ok

    okp, signs = True, []
    for f, want in ((0.0005, +1), (0.005, -1)):
        a_ok, s_ok = show("AR6", f, want)
        okp &= a_ok
        signs.append(s_ok)
    print(chk(fails, "AR6 the two code paths agree (amendment 2)", okp, "3.5 combined standard errors; "
              + ("both signs as second order predicts" if all(signs) else "RESULT: the sign at f = 0.0005 is not the one second order predicts; at sigma = 0.3 the second-order boundary f = 0.00124 is not where the loss changes sign")))
    print("AR6b the same model at phi = 0.5, sigma = 0.3")
    fb = f_range(2.0, 0.5)
    print(f"      second-order boundary: gain for f between {fb[0]:.5f} and {fb[1]:.4f}")
    okp, signs = True, []
    for f, want in ((0.003, +1), (0.03, -1)):
        a_ok, s_ok = show("AR6b", f, want)
        okp &= a_ok
        signs.append(s_ok)
    print(chk(fails, "AR6b the two code paths agree", okp, "3.5 combined standard errors; " + ("both signs as second order predicts" if all(signs) else "RESULT: a sign is not the one second order predicts")))
    print("AR6c (reported) phi = 0.9 at smaller noise, sigma = 0.1 (second-order boundary f = 0.00124 as in AR6)")
    for f, want in ((0.0005, +1), (0.005, -1)):
        show("AR6c", f, want, both=False)
    one = L.one_stage_exact(np.full_like(L.ZG, 0.5), 0.5 * np.exp(0.3 * L.ZG - 0.045))
    print(f"AR6d (reported) phi = 0.9, sigma = 0.3, smaller fractions; with no dormancy at all (f = 0) the exact one-stage loss is {one:+.5f}")
    for f in (0.00002, 0.0001):
        show("AR6d", f, +1, both=False)
    G6 = G_of(*reversed(dorm_RT(2.0, 0.0005, 0.9)), 0.9)
    print("AR6e (reported, amendment 4) phi = 0.9, f = 0.0005 at still smaller noise: D, D/sigma^2 beside the second-order coefficient, and (D - second order)/sigma^4")
    for sg, tag in ((0.3, "AR6"), (0.1, "AR6c"), (0.05, "AR6e"), (0.03, "AR6e")):
        rs = res[(tag, 0.0005 if tag != "AR6e" else sg, "stage")]
        rec = "second-order sign recovered" if rs["D"] > 3 * rs["se"] else ("gain" if rs["D"] < -3 * rs["se"] else "sign not resolved")
        print(f"      sigma = {sg:<5}: D = {rs['D']:+.6f} ± {rs['se']:.6f};  D/sigma^2 = {rs['D'] / sg ** 2:+.4f} ± {rs['se'] / sg ** 2:.4f} (second order {G6:+.5f});  "
              f"(D - second order)/sigma^4 = {(rs['D'] - G6 * sg ** 2) / sg ** 4:+.1f}  [{rec}]")

    print("\nAR10 geometric stage of 4 steps and a fixed delay b, phi = 0.9: my unexpanded loss against the referee's")
    ok = True
    for sigma, bs in ((1.0, (72, 120, 200)), (0.3, (72, 120))):
        for b in bs:
            r = res[("AR10", sigma, b)]
            ref, rse = REFEREE[(sigma, b)]
            z = (r["D"] - ref) / math.hypot(r["se"], rse)
            good = abs(z) <= 4
            ok &= good
            print(f"      sigma = {sigma}, b = {b:3d}: mine {r['D']:.5f} ± {r['se']:.5f}; referee {ref:.6f} ± {rse:.6f} ({z:+.1f} combined se) [{'reproduced' if good else 'NOT reproduced'}];  "
                  f"second order {r['so']:.5f};  one stage alone (exact) {ONE_STAGE[sigma]:.5f}: the delay {'helps' if r['D'] < ONE_STAGE[sigma] - 3 * r['se'] else ('hurts' if r['D'] > ONE_STAGE[sigma] + 3 * r['se'] else 'undecided')};  "
                  f"relaxation {r['tau']:.0f} steps, run {r['T']} after {r['burn']}; halves differ by {r['z']:+.1f} se")
    print(chk(fails, "AR10 the referee's five unexpanded values are reproduced", ok, "4 combined standard errors each"))

    secs = sum(r["secs"] for r in res.values())
    print(f"\nprocess time of the simulations {secs:.0f} s; wall time {time.time() - t_start:.0f} s; finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
