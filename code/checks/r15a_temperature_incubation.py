"""Run R15a (protocols/R15a-temperature-incubation.md): how far four conclusions of the manuscript survive
(1) temperature as one driver of several rates through curved responses, and (2) incubation times that are
nearly fixed, with or without death during the stay.

The claims, their tolerances and the pass/fail criteria are in the protocol, sections 5 to 7 (K1 to K7,
T1 to T9, I1 to I9) and its amendments.  This script computes every number of the report.  Every thermal
parameter is ILLUSTRATIVE (protocol, section 3).  Helper code: checks/r15a_lib.py.

Run: python3 -I checks/r15a_temperature_incubation.py      (three processes)
     add  --smoke N  to divide the lengths of the simulations by N (for a timing test only).
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import r15a_lib as L

SMOKE = 1
if "--smoke" in sys.argv:
    SMOKE = int(sys.argv[sys.argv.index("--smoke") + 1])

LN2, LN25, LN3 = L.LN2, L.LN25, L.LN3
TBARS = (14.0, 23.0, 30.0)
SIGTS = (2.0, 5.0, 8.0)
AMPS = tuple(math.sqrt(2) * s for s in SIGTS)
EXPS = {"E1": (LN2 / 10, LN2 / 10), "E2": (LN2 / 10, LN3 / 10), "E3": (LN2 / 10, 0.0), "E4": (0.0, LN3 / 10)}
A_STAY, B_STAY, SIGMA_H, MU_B = 4.0, 32, 1.0, 0.02
MU_I8 = 0.005


def size(kind):
    base = {"temp": (500, 20000, 2000), "table": (500, 40000, 2000), "tempB": (250, 20000, 2000),
            "fine": (500, 160000, 2000), "host": (400, 25000, 2000), "hostB": (400, 10000, 1000),
            "dd": (200, 20000, 2000), "cohort": (200, 10000, 3000)}[kind]
    return base[0], max(base[1] // SMOKE, 2000), base[2]


# ============================================================================= simulation tasks
def task_temp(a):
    """A ring driven by temperature: exact J, simulated D (path A or B), and the three formula versions."""
    t0 = time.time()
    model = L.make_model(a["spec"])
    Tbar, sigT, phi = a["Tbar"], a["sigT"], a["phi"]
    F = L.Formulas(model, Tbar, sigT)
    fa = F.tangent(phi)
    fb, fc, cap = F.hermite(phi)
    fcr = F.hermite_root(phi) if a.get("root_formula") else math.nan
    M, T, burn = a["size"]
    if a.get("path", "A") == "A":
        s0, g0 = model.sg(np.array([Tbar]))
        s0, g0 = s0[0], g0[0]
        ex = model.exposed

        def vary(z):
            ss, gg = model.sg(Tbar + sigT * z)
            out = []
            for k in ex:
                out.append((0, k, np.ascontiguousarray(ss[..., k])))
                out.append((1, k, np.ascontiguousarray(gg[..., k])))
            return out

        run = lambda ys, T_, b_, sd: L.sim_ring_A(model.n, s0, g0, model.k0, vary, phi, ys, M, T_, b_, sd)
    else:
        run = lambda ys, T_, b_, sd: L.sim_B(lambda z: model.parts(Tbar + sigT * z), model.n, phi, ys, M, T_, b_, sd)
    guess = fc if math.isfinite(fc) else 0.0
    root, se, Dm, runs = L.find_root(run, F.y_mean + guess, max(0.03, 0.5 * abs(guess)), M, T, burn, a["seed"])
    return a["key"], dict(D=root - F.y_mean, se=se, Dm=Dm - F.y_mean, J=F.J_part(), y_mean=F.y_mean,
                          y_tbar=F.y_tbar, fa=fa, fb=fb, fc=fc, fcr=fcr, cap=cap, runs=runs, secs=time.time() - t0)


def task_host(a):
    """Lognormal host on a ring at its threshold (g_k = 1 - s_k), path A."""
    t0 = time.time()
    s = np.asarray(a["s"], float)
    g = 1 - s
    n, kf, sigma, phi = s.size, a["kf"], a["sigma"], a["phi"]
    M, T, burn = a["size"]

    def vary(z):
        return [(1, kf, g[kf] * np.exp(sigma * z - sigma * sigma / 2))]

    run = lambda ys, T_, b_, sd: L.sim_ring_A(n, s, g, kf, vary, phi, ys, M, T_, b_, sd, start=a.get("start"))
    root, se, Dm, runs = L.find_root(run, a["y0"], max(0.03, 0.5 * abs(a["y0"])), M, T, burn, a["seed"])
    return a["key"], dict(D=root, se=se, Dm=Dm, runs=runs, secs=time.time() - t0)


def task_hostB(a):
    """One stage with a lognormal host, path B."""
    t0 = time.time()
    s, sigma, phi = a["s"], a["sigma"], a["phi"]
    M, T, burn = a["size"]

    def parts(z):
        return np.full((M, 1, 1), s), ((1 - s) * np.exp(sigma * z - sigma * sigma / 2))[:, None, None]

    run = lambda ys, T_, b_, sd: L.sim_B(parts, 1, phi, ys, M, T_, b_, sd)
    root, se, Dm, runs = L.find_root(run, a["y0"], max(0.03, 0.5 * abs(a["y0"])), M, T, burn, a["seed"])
    return a["key"], dict(D=root, se=se, Dm=Dm, runs=runs, secs=time.time() - t0)


def task_delay(a):
    """Stage A (chain of sub-stages with staying sA) and a fixed delay of b steps, path C."""
    t0 = time.time()
    sA = np.asarray(a["sA"], float)
    gA = 1 - sA
    M, T, burn = a["size"]
    run = lambda ys, T_, b_, sd: L.sim_delay(sA, gA, gA[-1], a["b"], a["sigma"], a["phi"], ys, M, T_, b_, sd)
    root, se, Dm, runs = L.find_root(run, a["y0"], max(0.03, 0.5 * abs(a["y0"])), M, T, burn, a["seed"])
    return a["key"], dict(D=root, se=se, Dm=Dm, runs=runs, secs=time.time() - t0)


def task_dd(a):
    """Model DD: D by simulation, J exactly, completion probability and incubation time."""
    t0 = time.time()
    m, Tbar, sigT, phi = a["m"], a["Tbar"], a["sigT"], a["phi"]
    model = L.DDModel(m)
    fn = model.fn
    F = L.Formulas(model, Tbar, sigT, J=120)
    fa = F.tangent(phi)
    fb, fc, cap = F.hermite(phi)
    M, T, burn = a["size"]
    run = lambda ys, T_, b_, sd: L.sim_dd_A(m, model.s1, fn, Tbar, sigT, phi, ys, M, T_, b_, sd)
    root, se, Dm, runs = L.find_root(run, F.y_mean + fc, max(0.03, 0.5 * abs(fc)), M, T, burn, a["seed"])
    Mc, Tc, bc = a["csize"]
    pm, age = L.sim_dd_cohort(m, fn, Tbar, sigT, phi, Mc, Tc, bc, a["seed"] + 31)
    W0, E0 = L.dd_weights(*fn(np.array(Tbar)), m)
    p_const, age_const = L.dd_constant_stats(W0, E0)
    Wg, Eg = L.dd_weights(*fn(Tbar + sigT * L.ZG), m)
    p_mean, age_mean = L.dd_constant_stats(L.WG @ Wg, L.WG @ Eg)
    return a["key"], dict(D=root - F.y_mean, se=se, J=F.J_part(), fa=fa, fc=fc, p_const=p_const, age_const=age_const,
                          p_mean=p_mean, age_mean=age_mean, p_var=float(pm.mean()),
                          p_var_se=float(pm.std(ddof=1) / math.sqrt(Mc)), age_var=float(age.mean()),
                          age_var_se=float(age.std(ddof=1) / math.sqrt(Mc)), runs=runs, secs=time.time() - t0)


# ============================================================================= exact blocks (no simulation)
def chk(fails, name, ok, detail):
    if not ok:
        fails.append(name)
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def sens(model, Tbar, h=1e-3):
    """pi = d ln p/dT, chi = d ln q/dT and p''/p of each exposed stage at Tbar, by central differences."""
    out = {}
    for k in model.exposed:
        fn = model.stages[k][1]

        def pq(T):
            q, p = L.qp(*fn(np.array([T])))
            return float(p[0]), float(q[0])

        p0, q0 = pq(Tbar)
        pp, qpl = pq(Tbar + h)
        pm, qm = pq(Tbar - h)
        out[k] = ((math.log(pp) - math.log(pm)) / (2 * h), (math.log(qpl) - math.log(qm)) / (2 * h),
                  (pp - 2 * p0 + pm) / (h * h * p0))
    return out


def second_order_small(model, Tbar, phi):
    """Per unit variance of temperature: curvature part J2 and loss D2 (claims T2 and T8)."""
    se = sens(model, Tbar)
    s, _ = model.sg(np.array([Tbar]))
    taus = 1 / (1 - s[0])
    J2 = 0.5 * sum(v[2] + 2 * v[0] * v[1] for v in se.values())
    D2 = L.gamma0_clock(taus, {k: (v[0], v[1]) for k, v in se.items()}, phi)
    return J2, D2


def sign_changes(xs, fs):
    return [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)
            if math.isfinite(fs[i]) and math.isfinite(fs[i + 1]) and fs[i] * fs[i + 1] < 0]


def block_K2(a):
    lines, fails = [], []
    worst = worst2 = 0.0
    for taus, phi in (((5, 2), 0.0), ((5, 2), 0.5), ((5, 20), 0.9), ((3, 8, 4), 0.3), ((10, 10), 0.9), ((2, 2, 1, 7), 0.7)):
        taus = np.array(taus, float)
        n = taus.size
        s, g = 1 - 1 / taus, 1 / taus
        A = np.zeros((n, n))
        for k in range(n):
            A[k, k] += s[k]
            A[(k + 1) % n, k] += g[k]
        dA = np.zeros((n, n))
        dA[1 % n, 0] = g[0]
        want = L.gamma0_one_transfer(taus, phi)
        worst = max(worst, abs(taus.sum() * L.cpp_resolvent(A, dA, phi) - want) / want)
        worst2 = max(worst2, abs(taus.sum() * L.cpp_lagsum(A, [(1 % n, 0, g[0])], [1.0], phi) - want) / want)
    lines.append(chk(fails, "K2a C'' (resolvent coding) = manuscript eq. (2)", worst < 1e-10, f"6 rings, largest relative difference {worst:.1e}"))
    lines.append(chk(fails, "K2a C'' (lag-sum coding) = manuscript eq. (2)", worst2 < 1e-10, f"6 rings, largest relative difference {worst2:.1e}"))
    rng = np.random.default_rng(1502)
    worst = 0.0
    for _ in range(50):
        n = int(rng.integers(2, 6))
        A = rng.uniform(0.05, 1.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.7)
        A += np.diag(rng.uniform(0.05, 0.5, n))
        for k in range(n):
            A[(k + 1) % n, k] += 0.2
        ne = int(rng.integers(2, 5))
        idx = np.argwhere(A > 0)
        pick = idx[rng.choice(len(idx), ne, replace=False)]
        kap = rng.uniform(-1, 1, ne)
        phi = float(rng.uniform(-0.5, 0.95))
        dA = np.zeros((n, n))
        ent = []
        for (i, j), kk in zip(pick, kap):
            dA[i, j] += kk * A[i, j]
            ent.append((int(i), int(j), float(A[i, j])))
        c1 = L.cpp_resolvent(A, dA, phi)
        c2 = L.cpp_lagsum(A, ent, list(kap), phi)
        lam, v, w = L.perron(A)
        scale = sum(abs(k1 * k2) * (w[i1] * a1 * v[j1] / lam) * (w[i2] * a2 * v[j2] / lam)
                    for (i1, j1, a1), k1 in zip(ent, kap) for (i2, j2, a2), k2 in zip(ent, kap))
        worst = max(worst, abs(c1 - c2) / max(scale, 1e-12))
    lines.append(chk(fails, "K2b the two codings of C'' agree, several entries on one driver, signed sizes", worst < 1e-9,
                     f"50 random models, largest difference relative to sum |kappa kappa' c c'| {worst:.1e}"))
    return "K2", dict(lines=lines, fails=fails)


def block_K5_K7_T1(a):
    lines, fails = [], []
    model = L.make_model(("TPC", "full", None))
    worst = 0.0
    for Tbar in TBARS:
        for amp in AMPS:
            sc = L.season_case(model, Tbar, amp)
            r0 = L.brute_r(sc["rest"], sc["scal"], sc["y_star"])
            r1 = L.brute_r(sc["rest"], sc["scal"], sc["y_star"] + 0.01)
            y_b = sc["y_star"] - r0 / ((r1 - r0) / 0.01)
            worst = max(worst, abs(y_b - sc["y_star"]))
    lines.append(chk(fails, "K5 seasonal ln R0*: Floquet against 400 years of iteration", worst < 1e-8,
                     f"9 cases (3 mean temperatures, 3 amplitudes), largest difference {worst:.1e}"))
    # K7
    fn = L.TPC2()
    worst = 0.0
    for m in (1, 4, 16):
        for T in TBARS:
            nu, mu = fn(np.array(T))
            W, E = L.dd_weights(nu, mu, m)
            prob, _ = L.dd_constant_stats(W, E)
            want = float((m * nu / (m * nu + mu)) ** m)
            worst = max(worst, abs(prob - want) / want)
    lines.append(chk(fails, "K7a one-day map of the chain: completion probability at constant temperature", worst < 1e-12,
                     f"m = 1, 4, 16 at 14, 23, 30 C: largest relative difference from (m nu/(m nu + mu))^m {worst:.1e}"))
    s1 = 1 - 1 / 30.0
    ring = L.Ring([("fixed", s1, 1 - s1), ("exposed", L.TPC2())])
    Ts = np.linspace(5, 38, 67)
    r1, c1 = L.DDModel(1).parts(Ts)
    r2, c2 = ring.parts(Ts)
    worst = max(np.abs(r1 - r2).max(), np.abs(c1 - c2).max())
    lines.append(chk(fails, "K7b the chain with m = 1 is the q, p ring", worst < 1e-12, f"67 temperatures, largest difference of an entry {worst:.1e}"))
    # T1 (a): conserved reproductive value along a random path; (b): seasonal, exact
    model = L.make_model(("EXP",) + EXPS["E1"])
    F = L.Formulas(model, 20.0, 8.0)
    pc = 0.10 / 0.15                         # the completion probability of E1, the same at every temperature
    y_exact = -math.log(pc)                  # threshold: R0 = exp(y) p = 1
    w = np.array([1.0, pc])                  # reproductive value at the threshold
    rng = np.random.default_rng(1501)
    z = rng.standard_normal()
    x = np.array([0.5, 0.5 / pc])
    dev = 0.0
    for t in range(5000):
        z = 0.9 * z + math.sqrt(1 - 0.81) * rng.standard_normal()
        r, sc_ = model.parts(np.array([20.0 + 8.0 * z]))
        x = (r[0] + math.exp(y_exact) * sc_[0]) @ x
        dev = max(dev, abs(w @ x - 1.0))
    dJ = max(abs(F.J_part()), abs(F.y_mean - y_exact))
    lines.append(chk(fails, "T1a proportional responses (E1): reproductive value is conserved on every path", dev < 1e-12 and dJ < 1e-12,
                     f"5,000 days at sigma_T = 8, phi = 0.9: largest |w.x(t) - 1| = {dev:.1e}; |J| and |ln R0_ii - ln R0_i| at most {dJ:.1e}"))
    worst = 0.0
    for amp in AMPS:
        for P in (7, 30, 365):
            sc = L.season_case(model, 20.0, amp, P)
            worst = max(worst, abs(sc["D"]), abs(sc["J"]))
    lines.append(chk(fails, "T1b proportional responses (E1), periodic temperature, exact", worst < 1e-10,
                     f"3 amplitudes, periods 7, 30, 365 days: largest |D| or |J| = {worst:.1e}"))
    return "K5K7T1", dict(lines=lines, fails=fails)


def block_T2(a):
    lines, fails = [], []
    rng = np.random.default_rng(1503)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 7))
        taus = np.exp(rng.uniform(math.log(1.05), math.log(40), n))
        if rng.uniform() < 0.3:
            taus[rng.integers(0, n)] = 1.0
        ne = int(rng.integers(1, min(3, n) + 1))
        ex = [int(k) for k in rng.choice(n, ne, replace=False)]
        for k in ex:
            taus[k] = max(taus[k], 1.05)
        phi = float(rng.uniform(0, 0.99)) if rng.uniform() > 0.1 else 0.0
        s, g = 1 - 1 / taus, 1 / taus
        A = np.zeros((n, n))
        dA = np.zeros((n, n))
        expo = {}
        for k in range(n):
            A[k, k] += s[k]
            A[(k + 1) % n, k] += g[k]
        for k in ex:
            pi, chi = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
            expo[k] = (pi, chi)
            dA[k, k] += -g[k] * chi                      # staying 1 - q, q = 1/tau
            dA[(k + 1) % n, k] += g[k] * (pi + chi)      # transfer p q
        want = taus.sum() * L.cpp_resolvent(A, dA, phi)
        got = L.gamma0_clock(taus, expo, phi)
        scale = sum(abs(pi) + abs(chi) for pi, chi in expo.values()) ** 2 / taus.sum()
        worst = max(worst, abs(got - want) / max(abs(want), 1e-3 * scale))
    lines.append(chk(fails, "T2 closed form with clock and completion sensitivities = matrix formula", worst < 1e-9,
                     f"200 random rings (2 to 6 stages, 1 to 3 exposed, either sign, phi 0 to 0.99): largest relative difference {worst:.1e}"))
    return "T2", dict(lines=lines, fails=fails)


def block_T3_T5_T6(a):
    lines, fails = [], []
    # ---- T3, seasonal: exact D against the tangent formula at small amplitude
    lines.append("T3 (seasonal, exact by Floquet): D_exact / D_formula(tangent) at small amplitude, period 365 d (and 30 d, extra)")
    ok_all = True
    cases = [("E2", L.make_model(("EXP",) + EXPS["E2"]), 20.0), ("E3", L.make_model(("EXP",) + EXPS["E3"]), 20.0),
             ("E4", L.make_model(("EXP",) + EXPS["E4"]), 20.0), ("TPC 14 C", L.make_model(("TPC", "full", None)), 14.0),
             ("TPC 30 C", L.make_model(("TPC", "full", None)), 30.0)]
    for name, model, Tbar in cases:
        for P in (365, 30):
            rel = {}
            for amp in (0.5, 0.25):
                D = L.season_case(model, Tbar, amp, P)["D"]
                f = L.season_formulas(model, Tbar, amp, P)["tangent"]
                rel[amp] = (D / f - 1, D, f)
            ratio = rel[0.5][0] / rel[0.25][0] if abs(rel[0.25][0]) > 1e-6 and abs(rel[0.5][0]) > 1e-6 else math.nan
            ok = abs(rel[0.5][0]) < 0.02 and (math.isnan(ratio) or 3 <= ratio <= 5)
            if P == 365:
                ok_all &= ok
            lines.append(f"      {name:9s} P={P:3d}: D = {rel[0.5][1]:+.4e} (formula {rel[0.5][2]:+.4e}) at 0.5 C; relative error {rel[0.5][0]:+.2e} at 0.5 C, "
                         f"{rel[0.25][0]:+.2e} at 0.25 C, ratio {ratio:.2f}  [{'ok' if ok else 'FAIL'}{'' if P == 365 else ', extra'}]")
    lines.append(chk(fails, "T3 seasonal: the matrix formula is the small-amplitude limit, cross terms included", ok_all,
                     "5 models at period 365 d: within 2% at 0.5 C and error falling as amplitude squared"))
    # ---- T5: where the closed form changes sign in phi; seasonal counterpart in the period
    lines.append("")
    lines.append("T5 closed form (T2) for the EXP models at 20 C, per unit variance of temperature (K, X as in the protocol)")
    for name in ("E2", "E3", "E4"):
        model = L.make_model(("EXP",) + EXPS[name])
        se = sens(model, 20.0)[1]
        s, _ = model.sg(np.array([20.0]))
        taus = 1 / (1 - s[0])
        row = []
        for phi in (0.0, 0.5, 0.8, 0.99):
            row.append(f"phi={phi}: {L.gamma0_clock(taus, {1: (se[0], se[1])}, phi):+.3e}")
        f = lambda ph: L.gamma0_clock(taus, {1: (se[0], se[1])}, ph)
        xs = np.linspace(0.0, 0.99, 100)
        sc = sign_changes(list(xs), [f(x) for x in xs])
        star = ", ".join(f"{brentq(f, lo, hi, xtol=1e-10):.4f}" for lo, hi in sc) if sc else "none"
        lines.append(f"      {name}: pi = {se[0]:+.5f}, chi = {se[1]:+.5f} per C, stays ({taus[0]:.2f}, {taus[1]:.2f}) d;  " + ";  ".join(row) + f";  sign change at phi = {star}")
    lines.append("T5 seasonal counterpart, exact (Floquet): E4, amplitude 7.07 C, D against the period")
    model = L.make_model(("EXP",) + EXPS["E4"])
    row = []
    for P in (3, 5, 7, 14, 30, 90, 365):
        sc = L.season_case(model, 20.0, AMPS[1], P)
        fo = L.season_formulas(model, 20.0, AMPS[1], P)
        row.append(f"P={P}: {sc['D']:+.3e} (tangent {fo['tangent']:+.3e}, full {fo['full']:+.3e})")
    lines.append("      " + ";  ".join(row))
    # ---- T6: one exposed stage alone, exact by quadrature
    lines.append("")
    lines.append("T6 one exposed stage alone, exact by quadrature (any memory): D against the bounds")
    lines.append("      B1 = -E ln(g/mean g) (fluctuating transfer only); B1' = B1 + -E ln(s/mean s); B2 = B1 + (tau - 1)(-E ln(s/mean s))")
    viol1 = viol1p = 0
    ok2 = True
    for name in ("E4", "E2", "E3"):
        fn = L.ExpStage(0.10, 0.05, *EXPS[name])
        for sigT in SIGTS:
            sb_ = L.stage_bounds(fn, 20.0, sigT)
            D, lg, ls, tau = sb_["D_one"], sb_["lg"], sb_["ls"], sb_["tau"]
            B1, B1p, B2 = lg, lg + ls, lg + (tau - 1) * ls
            viol1 += D > B1 * (1 + 1e-9)
            viol1p += D > B1p * (1 + 1e-9)
            ok2 &= D <= B2 * (1 + 1e-9)
            lines.append(f"      {name} sigma_T={sigT:.0f}: D = {D:.6f};  B1 = {B1:.6f} [{'violated' if D > B1 else 'holds'}];  "
                         f"B1' = {B1p:.6f} [{'violated' if D > B1p else 'holds'}];  B2 = {B2:.6f} [{'holds' if D <= B2 else 'VIOLATED'}];  tau = {tau:.2f} d")
    lines.append(chk(fails, "T6a counterexample to the uncorrected bounds (one stage, exact)", viol1 > 0 and viol1p > 0,
                     f"of 9 cases D exceeds B1 in {viol1} and B1' in {viol1p}"))
    lines.append(chk(fails, "T6b corrected bound B2 (one stage, exact)", ok2, "holds in all 9 cases"))
    return "T3T5T6", dict(lines=lines, fails=fails)


def tpc_total_season(model, Tbar, amp):
    return L.season_case(model, Tbar, amp)["total"]


def roots_and_sign(f, xs, xtol, fs=None):
    """Roots of f between grid points, and the intervals of the grid on which f > 0."""
    xs = [float(x) for x in xs]
    fs = [f(x) for x in xs] if fs is None else list(fs)
    roots = [brentq(f, lo, hi, xtol=xtol) for lo, hi in sign_changes(xs, fs)]
    edges = [xs[0]] + roots + [xs[-1]]
    pos = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if f(0.5 * (lo + hi)) > 0]
    return roots, pos


def show_pos(pos):
    return "; ".join(f"{lo:.2f} to {hi:.2f}" for lo, hi in pos) if pos else "nowhere on the grid"


T8_GRID = np.concatenate([np.arange(10.5, 34.51, 0.5), [34.6, 34.7, 34.8, 34.9, 34.95, 34.98]])


def block_T8(a):
    """Where more variable temperature raises ln R0 against baseline (i), model TPC: the small-variance
    limit, the exact seasonal case, and the prediction for the random case that the simulations then bracket."""
    lines, fails = [], []
    model = L.make_model(("TPC", "full", None))
    fine = np.arange(10.2, 34.81, 0.1)
    lines.append("T8 small-variance limit (second order).  Temperatures (C) at which J2 - D2 > 0, that is, at which a little variability raises ln R0;")
    lines.append("      grid 10.2 to 34.8 C.  J2 = (1/2) sum_k [p_k''/p_k + 2 pi_k chi_k], D2 from claim T2, both per unit variance.")
    for phi in (0.0, 0.8, 0.99):
        f = lambda T: (lambda jd: jd[0] - jd[1])(second_order_small(model, T, phi))
        roots, pos = roots_and_sign(f, fine, 1e-8)
        lines.append(f"      phi = {phi:<4}: raises for Tbar in {show_pos(pos)}")
    fJ = lambda T: second_order_small(model, T, 0.0)[0]
    roots, pos = roots_and_sign(fJ, fine, 1e-8)
    lines.append(f"      curvature part alone (J2 > 0): {show_pos(pos)}")

    def fs_(T):
        J2 = second_order_small(model, T, 0.0)[0]
        return J2 / 2 - L.season_formulas(model, T, 1.0)["tangent"]       # variance of a unit sinusoid is 1/2
    roots, pos = roots_and_sign(fs_, fine, 1e-8)
    lines.append(f"      seasonal, period 365 d: raises for Tbar in {show_pos(pos)}")
    for k, nm in ((1, "p2 (stage 2, development)"), (2, "p3 (stage 3, contact)")):
        g = lambda T: sens(model, T)[k][0]
        roots, _ = roots_and_sign(g, fine, 1e-8)
        lines.append(f"      optimum of the completion probability {nm}: " + ", ".join(f"{r:.2f}" for r in roots) + " C")
    gR = lambda T: sum(v[0] for v in sens(model, T).values())
    roots, _ = roots_and_sign(gR, fine, 1e-8)
    lines.append("      optimum of R0(T), proportional to p2 p3: " + ", ".join(f"{r:.2f}" for r in roots) + " C")
    for k, nm, fn in ((1, "development nu2", L.TPC2()), (2, "contact a3", L.TPC3())):
        g = lambda T: float(fn(np.array([T + 1e-4]))[0][0] - fn(np.array([T - 1e-4]))[0][0])
        roots, _ = roots_and_sign(g, fine, 1e-8)
        g2 = lambda T: float(fn(np.array([T + 1e-2]))[0][0] - 2 * fn(np.array([T]))[0][0] + fn(np.array([T - 1e-2]))[0][0])
        roots2, _ = roots_and_sign(g2, fine, 1e-8)
        lines.append(f"      {nm}: optimum " + ", ".join(f"{r:.2f}" for r in roots) + " C; inflection (convex below) " + ", ".join(f"{r:.2f}" for r in roots2) + " C")

    grid = T8_GRID
    lines.append("")
    lines.append("T8 curvature part alone at finite variance (exact, quadrature): temperatures at which J > 0 (mean rates give a larger R0 than the rates at Tbar)")
    for sigT in SIGTS:
        def fJx(T):
            rb, sb_ = L.gauss_mean_parts(model, T, sigT)
            return L.y_threshold(*L.const_parts(model, T)) - L.y_threshold(rb, sb_)
        roots, pos = roots_and_sign(fJx, grid, 1e-6)
        lines.append(f"      sigma_T = {sigT:.0f}: {show_pos(pos)}")
    lines.append("")
    lines.append("T8 seasonal, exact (Floquet): boundaries of total = ln R0* - ln R0(Tbar) on a grid from 10.5 to 34.98 C, with the exact total on each side")
    season_bounds = {}
    for amp in AMPS:
        f = lambda T: tpc_total_season(model, T, amp)
        roots, pos = roots_and_sign(f, grid, 1e-7)
        season_bounds[amp] = roots
        lines.append(f"      amplitude {amp:5.2f} C: variability raises ln R0 for Tbar in {show_pos(pos)}")
        for r in roots:
            d = 1.0 if r < 33.9 else (35.0 - r) / 2
            lines.append(f"            boundary {r:.3f} C;  total at {r - d:.3f} C = {f(r - d):+.5f},  at {r + d:.3f} C = {f(r + d):+.5f}")
    ok = all(len(v) >= 1 for v in season_bounds.values())
    lines.append(chk(fails, "T8 seasonal: a boundary exists at each amplitude, with an exact example on each side", ok,
                     "; ".join(f"A = {amp:.2f}: {len(v)} boundaries" for amp, v in season_bounds.items())))

    lines.append("")
    lines.append("T8 random, sigma_T = 5: boundaries predicted from the exact J and formula (c) on the same grid (simulated below)")
    pred = {}
    tots = {0.8: [], 0.99: []}
    for T in grid:
        F = L.Formulas(model, float(T), 5.0)
        for phi in tots:
            tots[phi].append(F.J_part() - F.hermite(phi)[1])
    for phi in (0.8, 0.99):
        f = lambda T: (lambda F: F.J_part() - F.hermite(phi)[1])(L.Formulas(model, T, 5.0))
        roots, pos = roots_and_sign(f, grid, 0.002, fs=tots[phi])
        pred[phi] = roots
        lines.append(f"      phi = {phi}: boundaries " + ", ".join(f"{r:.2f}" for r in roots) + f" C; predicted to raise ln R0 for Tbar in {show_pos(pos)}")
    return "T8", dict(lines=lines, fails=fails, pred=pred, season_bounds=season_bounds)


def block_T9_season(a):
    lines, fails = [], []
    full = L.make_model(("TPC", "full", None))
    split = L.make_model(("TPC", "split1", None))
    lines.append("T9 seasonal, exact (Floquet): loss D with stage 2 or stage 3 sheltered at its mean rates, and with the sheltered stay split in two")
    lines.append("      Tbar  amplitude   D both exposed   D only 3 exposed   D only 2 exposed   sum of the two   share that goes when 2 is sheltered   D with stage 1 split (15 + 15 d)")
    for Tbar in TBARS:
        for amp in AMPS:
            Ts = L.season_temps(Tbar, amp, 365)
            s, g = full.sg(Ts)
            sb, gb = s.mean(axis=0), g.mean(axis=0)
            D0 = L.season_case(full, Tbar, amp)["D"]
            D3 = L.season_case(L.make_model(("TPC", "shelter2", (sb, gb))), Tbar, amp)["D"]
            D2 = L.season_case(L.make_model(("TPC", "shelter3", (sb, gb))), Tbar, amp)["D"]
            Ds = L.season_case(split, Tbar, amp)["D"]
            lines.append(f"      {Tbar:4.0f}  {amp:8.2f}   {D0:+13.5f}   {D3:+15.5f}   {D2:+15.5f}   {D2 + D3:+13.5f}   {1 - D3 / D0:+10.3f}   {Ds:+13.5f} ({Ds / D0 - 1:+.1%})")
    return "T9s", dict(lines=lines, fails=fails)


def chain_ring(a, b, m, mu=0.0):
    """Staying rates of the ring [A, B_1..B_m]: A geometric with stay a; B a chain of m sub-stages with mean
    duration b and death at rate mu per step."""
    sub = math.exp(-mu) * (1 - m / b)
    return np.array([1 - 1 / a] + [sub] * m)


def chain_R(a, b, m, mu, phi):
    """R = E[phi^L] and the generation time for that ring (claim I1)."""
    sub = math.exp(-mu) * (1 - m / b)
    tau_sub = 1 / (1 - sub)
    TT = a + m * tau_sub
    if phi == 0:
        return 0.0, TT
    th = phi / (1 - phi)
    return (1 / (1 + a / th)) * (1 + tau_sub / th) ** (-m), TT


def chain_gamma0(a, b, m, mu, phi):
    R, TT = chain_R(a, b, m, mu, phi)
    return L.gamma0_general_R(TT, R, phi)


def matrix_gamma0(s, phi, kf=0):
    """Second-order loss per unit variance from the matrix formula, ring at threshold with staying rates s."""
    s = np.asarray(s, float)
    n = s.size
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] += s[k]
        A[(k + 1) % n, k] += 1 - s[k]
    dA = np.zeros((n, n))
    dA[(kf + 1) % n, kf] = 1 - s[kf]
    return float(np.sum(1 / (1 - s))) * L.cpp_resolvent(A, dA, phi)


def N2_fixed(b, a, sigma, mu, phi):
    """Second-order net margin for a fixed delay of b steps (b real), relative to ln R0 without the stage."""
    if phi == 0:
        R = 0.0
    else:
        th = phi / (1 - phi)
        R = (1 / (1 + a / th)) * phi ** b
    return -mu * b - sigma ** 2 * L.gamma0_general_R(a + b, R, phi)


def N2_geom(b, a, sigma, mu, phi):
    eps = math.expm1(mu)
    tauB = b * (1 + eps) / (1 + b * eps)
    if phi == 0:
        R = 0.0
    else:
        th = phi / (1 - phi)
        R = (1 / (1 + a / th)) / (1 + tauB / th)
    return -math.log1p(b * eps) - sigma ** 2 * L.gamma0_general_R(a + tauB, R, phi)


def block_I(a):
    lines, fails = [], []
    # ---- I1
    worst = 0.0
    for m in (1, 2, 4, 16, 32):
        for phi in (0.0, 0.5, 0.9, 0.99):
            for mu in (0.0, MU_B):
                want = matrix_gamma0(chain_ring(A_STAY, B_STAY, m, mu), phi)
                worst = max(worst, abs(chain_gamma0(A_STAY, B_STAY, m, mu, phi) - want) / want)
    lines.append(chk(fails, "I1a equation (2) with R = E[phi^L] for a chain of sub-stages = matrix formula (grid of I3)", worst < 1e-10,
                     f"40 cases, largest relative difference {worst:.1e}"))
    rng = np.random.default_rng(1504)
    worst = 0.0
    for _ in range(100):
        nst = int(rng.integers(2, 5))
        s_all, R, TT = [], 1.0, 0.0
        phi = float(rng.uniform(0.05, 0.99))
        th = phi / (1 - phi)
        for k in range(nst):
            m = int(rng.integers(1, 7))
            stay = float(np.exp(rng.uniform(0, math.log(30))))        # of one sub-stage, >= 1
            if k > 0 and rng.uniform() < 0.3:
                stay = 1.0                                           # a fixed delay of m steps
            mu = float(rng.uniform(0, 0.05)) if stay > 1 else 0.0
            sub = math.exp(-mu) * (1 - 1 / stay)
            if k == 0:
                sub = max(sub, 0.05)
            s_all += [sub] * m
            tau = 1 / (1 - sub)
            R *= (1 + tau / th) ** (-m)
            TT += m * tau
        want = matrix_gamma0(np.array(s_all), phi, kf=int(rng.integers(0, len(s_all))))
        worst = max(worst, abs(L.gamma0_general_R(TT, R, phi) - want) / want)
    lines.append(chk(fails, "I1b the same for 100 random chains, the fluctuating transfer at a random place", worst < 1e-10, f"largest relative difference {worst:.1e}"))
    # ---- I2
    ok = True
    for phi in (0.0, 0.5, 0.9, 0.99):
        vals = [chain_gamma0(A_STAY, B_STAY, m, 0.0, phi) for m in (1, 2, 4, 8, 16, 32)]
        d = np.diff(vals)
        ok &= bool(np.all(d >= -1e-12)) and (phi == 0 or bool(np.all(d > 0)))
        if phi == 0:
            ok &= bool(np.all(np.abs(d) < 1e-12))
    lines.append(chk(fails, "I2 at second order and fixed mean stays D does not fall as m rises (strictly rises if phi > 0)", ok, "a = 4, b = 32, m = 1, 2, 4, 8, 16, 32, phi = 0, 0.5, 0.9, 0.99"))
    # ---- I7
    a_, sg = A_STAY, SIGMA_H
    for mu in (0.005, 0.02, 0.001):
        bstar = sg / math.sqrt(2 * mu) - a_
        bi = np.arange(0, 400)
        Nint = [-mu * b - sg ** 2 * (matrix_gamma0(np.array([1 - 1 / a_] + [0.0] * int(b)), 0.0) if b <= 60 else 1 / (2 * (a_ + b))) for b in bi]
        arg = int(bi[int(np.argmax(Nint))])
        okF = (arg in (math.floor(bstar), math.ceil(bstar))) if bstar > 0 else arg == 0
        eps = math.expm1(mu)
        c = a_ * eps + 1 + eps
        qa, qb, qc = 2 * eps * c * c, 4 * eps * a_ * c - sg ** 2 * (1 + eps) * eps, 2 * eps * a_ * a_ - sg ** 2 * (1 + eps)
        disc = qb * qb - 4 * qa * qc
        bg = (-qb + math.sqrt(disc)) / (2 * qa) if disc >= 0 else math.nan
        if bg > 0:
            res = minimize_scalar(lambda b: -N2_geom(b, a_, sg, mu, 0.0), bounds=(1e-6, 5000), method="bounded", options={"xatol": 1e-10})
            okG = abs(res.x - bg) < 1e-5 * max(1, bg)
            num = res.x
        else:
            num, okG = 0.0, N2_geom(0.5, a_, sg, mu, 0.0) < N2_geom(1e-9, a_, sg, mu, 0.0)
        tauB = lambda b: b * (1 + eps) / (1 + b * eps)
        wantD = matrix_gamma0(np.array([1 - 1 / a_, math.exp(-mu) * (1 - 1 / 10.0)]), 0.0)
        okD = abs(1 / (2 * (a_ + tauB(10.0))) - wantD) < 1e-12
        lines.append(chk(fails, f"I7 closed forms for b* at second order, no memory, mu = {mu}", okF and okG and okD,
                         f"fixed delay: b* = sigma/sqrt(2 mu) - a = {bstar:.3f}, integer maximum of the matrix value at {arg}; "
                         f"geometric: root of the quadratic {bg:.4f}, numerical maximum {num:.4f}"))
    lines.append("I7 with memory (second order, numerical): the stay at which lengthening stops paying, a = 4, sigma = 1")
    lines.append("      'gain window' = the stays b at which one more step raises ln R0 - D; N(b*) - N(0) says whether the best delay beats no steady stage at all")
    for kind, N2 in (("fixed delay", N2_fixed), ("geometric  ", N2_geom)):
        for mu in (0.001, 0.005, 0.02):
            for phi in (0.0, 0.5, 0.9, 0.99):
                bs = np.concatenate([np.arange(0.05, 20, 0.05), np.arange(20, 3000, 0.5)])
                Nv = np.array([N2(b, a_, sg, mu, phi) for b in bs])
                d = np.diff(Nv)
                up = np.nonzero(d > 0)[0]
                if up.size == 0:
                    lines.append(f"      {kind} mu = {mu:<5} phi = {phi:<4}: no stay at which lengthening pays")
                    continue
                lo, hi = bs[up[0]], bs[up[-1] + 1]
                N0 = -sg ** 2 / (2 * a_)
                lines.append(f"      {kind} mu = {mu:<5} phi = {phi:<4}: gain window from b = {lo:.2f} to b* = {hi:.2f};  N(b*) - N(0) = {Nv[up[-1] + 1] - N0:+.4f}")
    return "I", dict(lines=lines, fails=fails)


def block_I9_season(a):
    lines, fails = [], []
    lines.append("I9 seasonal, exact (amplitude 7.07 C, period 365 d): incubation as a chain of m sub-stages")
    lines.append("      m  Tbar   ln P(Tbar)   J = ln P(mean rates) - ln P(Tbar)   -D      total    ln P(cohort mean) - ln P(Tbar)   days: const / mean rates / seasonal")
    fn = L.TPC2()
    for m in (1, 4, 16):
        model = L.DDModel(m)
        for Tbar in TBARS:
            sc = L.season_case(model, Tbar, AMPS[1])
            W0, E0 = L.dd_weights(*fn(np.array(Tbar)), m)
            p0, age0 = L.dd_constant_stats(W0, E0)
            Wg, Eg = L.dd_weights(*fn(L.season_temps(Tbar, AMPS[1], 365)), m)
            pm_, agem = L.dd_constant_stats(Wg.mean(axis=0), Eg.mean(axis=0))
            pv, agev = L.dd_season_stats(m, fn, Tbar, AMPS[1])
            lines.append(f"      {m:2d}  {Tbar:4.0f}   {math.log(p0):+8.4f}   {sc['J']:+10.4f}                     {-sc['D']:+8.4f}  {sc['total']:+8.4f}   {math.log(pv) - math.log(p0):+10.4f}"
                         f"                 {age0:6.2f} / {agem:6.2f} / {agev:6.2f}")
    return "I9s", dict(lines=lines, fails=fails)


def block_season_table(a):
    """Decomposition for the sinusoid, exact, with the two formula versions."""
    model = L.make_model(("TPC", "full", None))
    rows = {}
    for Tbar in TBARS:
        for amp in AMPS:
            sc = L.season_case(model, Tbar, amp)
            fo = L.season_formulas(model, Tbar, amp)
            rows[(Tbar, amp)] = (sc["J"], sc["D"], sc["total"], fo["tangent"], fo["full"])
    return "season_table", dict(rows=rows, lines=[], fails=[])


BLOCKS = {"K2": block_K2, "K5K7T1": block_K5_K7_T1, "T2": block_T2, "T3T5T6": block_T3_T5_T6, "T8": block_T8,
          "T9s": block_T9_season, "I": block_I, "I9s": block_I9_season, "season_table": block_season_table}
TASKS = {"temp": task_temp, "host": task_host, "hostB": task_hostB, "delay": task_delay, "dd": task_dd}


def run_task(a):
    t0 = time.time()
    if a["fn"] == "block":
        key, res = BLOCKS[a["name"]](a)
    else:
        key, res = TASKS[a["fn"]](a)
    res["secs"] = time.time() - t0
    return key, res


# ============================================================================= the task list
def build_tasks():
    tasks = []
    seed = [15_000_000]

    def nxt():
        seed[0] += 1000
        return seed[0]

    # K1: one stage, lognormal host
    for phi in (0.0, 0.9):
        sd = nxt()
        tasks.append(dict(fn="host", key=("K1", "A", phi), s=[0.75], kf=0, sigma=1.0, phi=phi, y0=0.1, size=size("host"), seed=sd))
        tasks.append(dict(fn="hostB", key=("K1", "B", phi), s=0.75, sigma=1.0, phi=phi, y0=0.1, size=size("hostB"), seed=sd + 1))
    # K3: small noise against the second-order value
    for phi in (0.5, 0.9):
        y0 = 0.04 * chain_gamma0(A_STAY, B_STAY, 4, 0.0, phi)
        tasks.append(dict(fn="host", key=("K3", 4, phi), s=list(chain_ring(A_STAY, B_STAY, 4)), kf=0, sigma=0.2, phi=phi, y0=y0,
                          size=(400, size("host")[1] * 2, 2000), seed=nxt()))
        y0 = 0.04 * chain_gamma0(A_STAY, B_STAY, 32, 0.0, phi)
        tasks.append(dict(fn="delay", key=("K3", 32, phi), sA=[1 - 1 / A_STAY], b=B_STAY, sigma=0.2, phi=phi, y0=y0,
                          size=(400, size("host")[1] * 2, 2000), seed=nxt()))
    # K4: one exposed stage alone (E4), both paths
    sd = nxt()
    tasks.append(dict(fn="temp", key=("K4", "A"), spec=("EXP1",) + EXPS["E4"], Tbar=20.0, sigT=5.0, phi=0.8, size=size("temp"), seed=sd))
    tasks.append(dict(fn="temp", key=("K4", "B"), spec=("EXP1",) + EXPS["E4"], Tbar=20.0, sigT=5.0, phi=0.8, size=size("tempB"), seed=sd + 1, path="B"))
    # T7 table (path A), K6 (path B at sigma_T = 5)
    tseed = {}
    for phi in (0.8, 0.99):
        for sigT in SIGTS:
            for Tbar in TBARS:
                tseed[(Tbar, sigT, phi)] = sd = nxt()
                tasks.append(dict(fn="temp", key=("T7", Tbar, sigT, phi), spec=("TPC", "full", None), Tbar=Tbar, sigT=sigT, phi=phi,
                                  size=size("table"), seed=sd, root_formula=True))
                if sigT == 5.0:
                    tasks.append(dict(fn="temp", key=("K6", Tbar, phi), spec=("TPC", "full", None), Tbar=Tbar, sigT=sigT, phi=phi,
                                      size=size("tempB"), seed=sd + 17, path="B"))
    # T1c, T3 random, T5
    for phi in (0.8, 0.99):
        tasks.append(dict(fn="temp", key=("T1c", phi), spec=("EXP",) + EXPS["E1"], Tbar=20.0, sigT=8.0, phi=phi, size=size("temp"), seed=nxt()))
    tasks.append(dict(fn="temp", key=("T3", "E3"), spec=("EXP",) + EXPS["E3"], Tbar=20.0, sigT=1.0, phi=0.8, size=size("fine"), seed=nxt()))
    tasks.append(dict(fn="temp", key=("T3", "TPC14"), spec=("TPC", "full", None), Tbar=14.0, sigT=1.0, phi=0.8, size=size("fine"), seed=nxt()))
    for name in ("E2", "E3", "E4"):
        for phi in (0.0, 0.8, 0.99):
            sz = size("fine") if phi == 0 else size("temp")
            tasks.append(dict(fn="temp", key=("T5", name, phi), spec=("EXP",) + EXPS[name], Tbar=20.0, sigT=5.0, phi=phi, size=sz, seed=nxt()))
    tasks.append(dict(fn="temp", key=("T6", "E4s"), spec=("EXPs",) + EXPS["E4"], Tbar=20.0, sigT=5.0, phi=0.0, size=size("table"), seed=nxt()))
    # T9 shelter variants, paired with the T7 runs at sigma_T = 5 (same seed, same size)
    full = L.make_model(("TPC", "full", None))
    for Tbar in (14.0, 30.0):
        s, g = full.sg(Tbar + 5.0 * L.ZG)
        sb, gb = tuple(L.WG @ s), tuple(L.WG @ g)
        for phi in (0.8, 0.99):
            for variant in ("shelter2", "shelter3", "split1"):
                tasks.append(dict(fn="temp", key=("T9", variant, Tbar, phi), spec=("TPC", variant, (sb, gb)), Tbar=Tbar, sigT=5.0, phi=phi,
                                  size=size("table"), seed=tseed[(Tbar, 5.0, phi)]))
    sd = nxt()
    for variant in ("full", "split1"):
        tasks.append(dict(fn="temp", key=("T9", variant, 14.0, 0.0), spec=("TPC", variant, None), Tbar=14.0, sigT=5.0, phi=0.0, size=size("table"), seed=sd))
    # I3: chains at sigma = 1
    i3seed = {}
    for phi in (0.0, 0.5, 0.9, 0.99):
        i3seed[phi] = sd = nxt()
        for mu in (0.0, MU_B):
            for m in (1, 2, 4, 16):
                y0 = chain_gamma0(A_STAY, B_STAY, m, mu, phi)
                tasks.append(dict(fn="host", key=("I3", m, phi, mu), s=list(chain_ring(A_STAY, B_STAY, m, mu)), kf=0, sigma=SIGMA_H, phi=phi,
                                  y0=y0, size=size("host"), seed=sd))
        y0 = chain_gamma0(A_STAY, B_STAY, 32, 0.0, phi)
        tasks.append(dict(fn="delay", key=("I3", 32, phi, 0.0), sA=[1 - 1 / A_STAY], b=B_STAY, sigma=SIGMA_H, phi=phi, y0=y0, size=size("host"), seed=sd))
    # I4: every stay fixed (exact value sigma^2/2), and stage A as a chain of 2 with B fixed
    for phi in (0.0, 0.9):
        tasks.append(dict(fn="delay", key=("I4", "A fixed", phi), sA=[0.0] * 4, b=B_STAY, sigma=SIGMA_H, phi=phi, y0=0.5, size=size("host"), seed=nxt()))
        tasks.append(dict(fn="delay", key=("I4", "A chain of 2", phi), sA=[0.5, 0.5], b=B_STAY, sigma=SIGMA_H, phi=phi, y0=0.3, size=size("host"), seed=nxt()))
    # I6 and K8: position of the fluctuating transfer; path A against path C for the fixed delay.
    # Same seed and size as the I3 runs at phi = 0.9, so that the comparisons are paired.
    y0 = chain_gamma0(A_STAY, B_STAY, 4, 0.0, 0.9)
    tasks.append(dict(fn="host", key=("I6", 4), s=list(chain_ring(A_STAY, B_STAY, 4)), kf=4, sigma=SIGMA_H, phi=0.9, y0=y0, size=size("host"), seed=i3seed[0.9]))
    y0 = chain_gamma0(A_STAY, B_STAY, 32, 0.0, 0.9)
    tasks.append(dict(fn="host", key=("I6", 32), s=list(chain_ring(A_STAY, B_STAY, 32)), kf=32, sigma=SIGMA_H, phi=0.9, y0=y0, size=size("host"), seed=i3seed[0.9]))
    tasks.append(dict(fn="host", key=("K8", 32), s=list(chain_ring(A_STAY, B_STAY, 32)), kf=0, sigma=SIGMA_H, phi=0.9, y0=y0, size=size("host"), seed=i3seed[0.9]))
    # I8: the net margin against the stay, common random numbers within each phi
    for phi, bsF in ((0.0, list(range(1, 15))), (0.5, [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24])):
        sd = nxt()
        for b in bsF:
            y0 = -N2_fixed(b, A_STAY, SIGMA_H, 0.0, phi)
            tasks.append(dict(fn="delay", key=("I8", "F", phi, b), sA=[1 - 1 / A_STAY], b=b, sigma=SIGMA_H, phi=phi, y0=y0, size=size("host"), seed=sd))
        for b in (2, 4, 6, 8, 12, 16, 24, 32, 48):
            sB = math.exp(-MU_I8) * (1 - 1 / b)
            y0 = SIGMA_H ** 2 * L.gamma0_one_transfer([A_STAY, 1 / (1 - sB)], phi)
            tasks.append(dict(fn="host", key=("I8", "G", phi, b), s=[1 - 1 / A_STAY, sB], kf=0, sigma=SIGMA_H, phi=phi, y0=y0, size=size("host"), seed=sd))
    # I9: model DD
    for m in (1, 4, 16):
        for Tbar in TBARS:
            for phi in (0.8, 0.99):
                tasks.append(dict(fn="dd", key=("I9", m, Tbar, phi), m=m, Tbar=Tbar, sigT=5.0, phi=phi, size=size("dd"), csize=size("cohort"), seed=nxt()))
    tasks.append(dict(fn="temp", key=("K9",), spec=("TPCinc",), Tbar=14.0, sigT=5.0, phi=0.8, size=size("temp"), seed=nxt()))
    return tasks


# ============================================================================= report
def main():
    t_start = time.time()
    print("R15a: temperature and incubation time.  Protocol: protocols/R15a-temperature-incubation.md")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (system clock); simulation lengths divided by {SMOKE}")
    print("Every thermal parameter is ILLUSTRATIVE (protocol, section 3).  ± is one standard error.  Day = one time step in Part 1 and 2(c).")
    tasks = build_tasks()
    res = {}
    with mp.Pool(3) as pool:
        first = pool.apply_async(run_task, (dict(fn="block", name="T8"),))
        pend = [pool.apply_async(run_task, (dict(fn="block", name=nm),)) for nm in BLOCKS if nm != "T8"]
        order = sorted(tasks, key=lambda a: -(a["size"][0] * a["size"][1] * (len(a.get("s", [0, 0, 0])) if a["fn"] == "host" else (a.get("m", 1) + 2))))
        pend += [pool.apply_async(run_task, (a,)) for a in order]
        k8, r8 = first.get()
        res[k8] = r8
        seed = 16_000_000
        t8keys = []
        for phi, roots in r8["pred"].items():
            for ib, root in enumerate(roots):
                gap = 35.0 - root                      # distance to the upper limit of development
                for off in ((-2.0, -1.0, 1.0, 2.0) if gap > 2.5 else (-2.0, -1.0, gap / 3, 2 * gap / 3)):
                    seed += 1000
                    Tb = round(root + off, 3)
                    if not (10.2 < Tb < 34.995):
                        continue
                    key = ("T8", phi, ib, off)
                    t8keys.append(key)
                    pend.append(pool.apply_async(run_task, (dict(fn="temp", key=key, spec=("TPC", "full", None), Tbar=Tb, sigT=5.0, phi=phi,
                                                                   size=size("temp"), seed=seed),)))
        for p in pend:
            k, r = p.get()
            res[k] = r
    fails = []
    for nm in BLOCKS:
        fails += res[nm]["fails"]

    def block(nm):
        for ln in res[nm]["lines"]:
            print(ln)

    def report(name, ok, detail):
        print(chk(fails, name, ok, detail))

    # ------------------------------------------------------------------ checks of the code
    print("\n" + "=" * 110 + "\nSECTION 5 OF THE PROTOCOL: CHECKS OF THE CODE AGAINST KNOWN RESULTS\n" + "=" * 110)
    hz = np.exp(SIGMA_H * L.ZG - SIGMA_H ** 2 / 2)
    D_one_075 = L.one_stage_exact(np.full_like(L.ZG, 0.75), 0.25 * hz)
    ok = True
    det = []
    for phi in (0.0, 0.9):
        for path in ("A", "B"):
            r = res[("K1", path, phi)]
            good = abs(r["D"] - D_one_075) <= max(3 * r["se"], 0.02 * D_one_075)
            ok &= good
            det.append(f"path {path} phi={phi}: {r['D']:.5f} ± {r['se']:.5f}")
    report("K1 one stage, lognormal host (s = 0.75, sigma = 1): simulation against quadrature", ok, f"exact {D_one_075:.5f}; " + "; ".join(det))
    block("K2")
    ok = True
    det = []
    for m in (4, 32):
        for phi in (0.5, 0.9):
            r = res[("K3", m, phi)]
            want = 0.04 * chain_gamma0(A_STAY, B_STAY, m, 0.0, phi)
            good = abs(r["D"] - want) <= max(0.05 * want, 3 * r["se"])
            ok &= good
            det.append(f"m={m} phi={phi}: {r['D']:.5f} ± {r['se']:.5f} against {want:.5f}")
    report("K3 chain at sigma = 0.2: simulation against the second-order value", ok, "; ".join(det))
    D_k4 = L.stage_bounds(L.ExpStage(0.10, 0.05, *EXPS["E4"]), 20.0, 5.0)["D_one"]
    ok = True
    det = []
    for path in ("A", "B"):
        r = res[("K4", path)]
        ok &= abs(r["D"] - D_k4) <= max(3 * r["se"], 0.02 * abs(D_k4))
        det.append(f"path {path}: {r['D']:.6f} ± {r['se']:.6f}")
    report("K4 one exposed stage alone (E4, sigma_T = 5, phi = 0.8): simulation against quadrature", ok, f"exact {D_k4:.6f}; " + "; ".join(det))
    block("K5K7T1")
    ok = True
    det = []
    for phi in (0.8, 0.99):
        for Tbar in TBARS:
            ra, rb = res[("T7", Tbar, 5.0, phi)], res[("K6", Tbar, phi)]
            zsc = (ra["D"] - rb["D"]) / math.hypot(ra["se"], rb["se"])
            ok &= abs(zsc) <= 3.5
            det.append(f"{Tbar:.0f} C phi={phi}: A {ra['D']:+.4f} ± {ra['se']:.4f}, B {rb['D']:+.4f} ± {rb['se']:.4f} ({zsc:+.1f} se)")
    report("K6 paths A and B agree on D (model TPC, sigma_T = 5)", ok, "; ".join(det))
    ra, rc = res[("K8", 32)], res[("I3", 32, 0.9, 0.0)]
    zsc = (ra["D"] - rc["D"]) / math.hypot(ra["se"], rc["se"])
    dd = ra["Dm"] - rc["Dm"]
    report("K8 (amendment 2) fixed delay of 32: ring of 33 stages (path A) against the delay recursion (path C), same random numbers", abs(zsc) <= 3,
           f"A {ra['D']:.4f} ± {ra['se']:.4f}, C {rc['D']:.4f} ± {rc['se']:.4f} ({zsc:+.1f} combined se); paired difference {dd.mean():+.5f} ± {dd.std(ddof=1) / math.sqrt(dd.size):.5f}")
    ra, rd = res[("K9",)], res[("I9", 1, 14.0, 0.8)]
    zsc = (ra["D"] - rd["D"]) / math.hypot(ra["se"], rd["se"])
    report("K9 (amendment 2) model DD with m = 1 (chain code) against the same ring through path A, 14 C, sigma_T = 5, phi = 0.8", abs(zsc) <= 3,
           f"ring {ra['D']:+.5f} ± {ra['se']:.5f}, chain code {rd['D']:+.5f} ± {rd['se']:.5f} ({zsc:+.1f} se)")
    okJ = max(abs(res[("I9", m, Tb, ph)]["J"] - (math.log(res[("I9", m, Tb, ph)]["p_mean"]) - math.log(res[("I9", m, Tb, ph)]["p_const"])))
              for m in (1, 4, 16) for Tb in TBARS for ph in (0.8, 0.99))
    report("K7c model DD: J from the eigenvalue = ln P(mean rates) - ln P(Tbar)", okJ < 1e-8, f"18 cases, largest difference {okJ:.1e}")

    # ------------------------------------------------------------------ Part 1
    print("\n" + "=" * 110 + "\nPART 1: TEMPERATURE\n" + "=" * 110)
    ok = all(abs(res[("T1c", phi)]["D"]) <= 3 * res[("T1c", phi)]["se"] for phi in (0.8, 0.99))
    report("T1c proportional responses (E1, sigma_T = 8): simulated D is zero", ok,
           "; ".join(f"phi={phi}: {res[('T1c', phi)]['D']:+.2e} ± {res[('T1c', phi)]['se']:.1e}" for phi in (0.8, 0.99)))
    block("T2")
    block("T3T5T6")
    print("")
    for key, nm in ((("T3", "E3"), "E3, sigma_T = 1, phi = 0.8"), (("T3", "TPC14"), "TPC at 14 C, sigma_T = 1, phi = 0.8")):
        r = res[key]
        inform = 3 * r["se"] <= 0.5 * abs(r["fa"])
        good = abs(r["D"] - r["fa"]) <= max(0.05 * abs(r["fa"]), 3 * r["se"])
        if not inform:
            print(f"[UNINFORMATIVE] T3 random, {nm}: D = {r['D']:+.3e} ± {r['se']:.1e}, formula (a) {r['fa']:+.3e}")
        else:
            report(f"T3 random, {nm}: simulated D against the tangent formula", good,
                   f"D = {r['D']:+.4e} ± {r['se']:.1e}; formula (a) {r['fa']:+.4e}; ratio {r['D'] / r['fa']:.3f}; formula (c) {r['fc']:+.4e}")
    print("\nT4 and T5: sign of the fluctuation part in the EXP models (sigma_T = 5, simulated) beside the closed form")
    print("      case  phi     D simulated              formula (a)   formula (c)   predicted sign   simulated sign")
    ok4, ok5 = True, True
    for name in ("E2", "E3", "E4"):
        for phi in (0.0, 0.8, 0.99):
            r = res[("T5", name, phi)]
            pred = "loss" if r["fa"] > 0 else "gain"
            sim = "loss" if r["D"] > 3 * r["se"] else ("gain" if r["D"] < -3 * r["se"] else "unresolved")
            ok5 &= pred == sim
            if phi == 0:
                ok4 &= r["D"] >= -3 * r["se"]
            print(f"      {name}    {phi:<5}  {r['D']:+.3e} ± {r['se']:.1e}   {r['fa']:+.3e}    {r['fc']:+.3e}    {pred:5s}            {sim}")
    for k in res:
        if isinstance(k, tuple) and k[0] == "T9" and k[-1] == 0.0:
            ok4 &= res[k]["D"] >= -3 * res[k]["se"]
    report("T4 without memory D is not negative", ok4, "every phi = 0 case of Part 1 (3 EXP cases and 2 TPC cases), 3 standard errors")
    report("T5 sign of the fluctuation part as the closed form predicts", ok5, "9 cases, each by more than 3 standard errors")
    # T6 two-stage
    print("\nT6 two-stage ring E4 (sigma_T = 5), simulated D against the bounds (stage 2 is the only exposed stage)")
    sb_ = L.stage_bounds(L.ExpStage(0.10, 0.05, *EXPS["E4"]), 20.0, 5.0)
    B1, B1p, B2 = sb_["lg"], sb_["lg"] + sb_["ls"], sb_["lg"] + (sb_["tau"] - 1) * sb_["ls"]
    r = res[("T5", "E4", 0.0)]
    report("T6c counterexample to the uncorrected bounds, two stages (sheltered stay 20 d), phi = 0", r["D"] > B1 + 3 * r["se"] and r["D"] > B1p + 3 * r["se"],
           f"D = {r['D']:.6f} ± {r['se']:.6f};  B1 = {B1:.6f} ({(r['D'] - B1) / r['se']:+.1f} se);  B1' = {B1p:.6f} ({(r['D'] - B1p) / r['se']:+.1f} se);  corrected B2 = {B2:.6f}")
    r = res[("T6", "E4s")]
    report("T6c' (amendment 3) the same with a sheltered stay of 5 d", r["D"] > B1 + 3 * r["se"] and r["D"] > B1p + 3 * r["se"] and r["D"] <= B2 + 3 * r["se"],
           f"D = {r['D']:.6f} ± {r['se']:.6f};  B1 = {B1:.6f} ({(r['D'] - B1) / r['se']:+.1f} se);  B1' = {B1p:.6f} ({(r['D'] - B1p) / r['se']:+.1f} se);  corrected B2 = {B2:.6f}")
    okb = True
    det = []
    for name in ("E2", "E3", "E4"):
        sb_ = L.stage_bounds(L.ExpStage(0.10, 0.05, *EXPS[name]), 20.0, 5.0)
        b2 = sb_["lg"] + (sb_["tau"] - 1) * sb_["ls"]
        for phi in (0.0, 0.8, 0.99):
            r = res[("T5", name, phi)]
            okb &= r["D"] <= b2 + 3 * r["se"]
        det.append(f"{name}: B2 = {b2:.5f}, largest D = {max(res[('T5', name, ph)]['D'] for ph in (0.0, 0.8, 0.99)):+.5f}")
    report("T6d corrected bound B2 in every EXP case", okb, "; ".join(det))

    # T7
    print("\nT7 DECOMPOSITION, model TPC, random temperature.  J = curvature part (exact), -D = fluctuation part (simulated), total = J - D")
    print("      = ln R0* - ln R0(constant temperature Tbar).  Formula versions for D: (a) tangent at Tbar, (b) first Hermite term, (c) all Hermite")
    print("      terms, (c') formula (c) solved at the root.  In brackets: formula / simulated D.")
    print("      phi   sigma_T  Tbar        J        -D (sim)              total               (a)              (b)              (c)              (c')")
    ratios = {}
    for phi in (0.8, 0.99):
        for sigT in SIGTS:
            for Tbar in TBARS:
                r = res[("T7", Tbar, sigT, phi)]
                D = r["D"]
                cells = []
                for nm in ("fa", "fb", "fc", "fcr"):
                    ratios.setdefault((sigT, nm), []).append(r[nm] / D)
                    cells.append(f"{r[nm]:+.4f} [{r[nm] / D:5.2f}]")
                print(f"      {phi:<5} {sigT:4.0f}    {Tbar:4.0f}   {r['J']:+8.4f}   {-D:+8.4f} ± {r['se']:.4f}   {r['J'] - D:+8.4f} ± {r['se']:.4f}   " + "  ".join(cells))
    print("      largest |formula - simulated D| over the six cases at each sigma_T, in ln R0 units (and the largest |D| itself):")
    for sigT in SIGTS:
        errs = {nm: max(abs(res[("T7", Tb, sigT, ph)][nm] - res[("T7", Tb, sigT, ph)]["D"]) for Tb in TBARS for ph in (0.8, 0.99)) for nm in ("fa", "fb", "fc", "fcr")}
        dmax = max(abs(res[("T7", Tb, sigT, ph)]["D"]) for Tb in TBARS for ph in (0.8, 0.99))
        print(f"        sigma_T = {sigT:.0f}: (a) {errs['fa']:.4f};  (b) {errs['fb']:.4f};  (c) {errs['fc']:.4f};  (c') {errs['fcr']:.4f};  largest |D| {dmax:.4f}")
    print("      range of formula / simulated D over the cases in which |D| exceeds both 0.005 and 5 standard errors:")
    for sigT in SIGTS:
        big = [(Tb, ph) for Tb in TBARS for ph in (0.8, 0.99)
               if abs(res[("T7", Tb, sigT, ph)]["D"]) > max(0.005, 5 * res[("T7", Tb, sigT, ph)]["se"])]
        if not big:
            print(f"        sigma_T = {sigT:.0f}: no such case")
            continue
        print(f"        sigma_T = {sigT:.0f} ({len(big)} cases): " + ";  ".join(
            f"({lab}) {min(res[('T7', Tb, sigT, ph)][nm] / res[('T7', Tb, sigT, ph)]['D'] for Tb, ph in big):.2f} to "
            f"{max(res[('T7', Tb, sigT, ph)][nm] / res[('T7', Tb, sigT, ph)]['D'] for Tb, ph in big):.2f}"
            for nm, lab in (("fa", "a"), ("fb", "b"), ("fc", "c"), ("fcr", "c'"))))
    print("\nT7 DECOMPOSITION, model TPC, seasonal sinusoid of period 365 d, exact (Floquet).  Formula: tangent at Tbar; all harmonics about the mean rates.")
    print("      amplitude  Tbar        J          -D         total      D tangent [ratio]     D all harmonics [ratio]")
    for amp in AMPS:
        for Tbar in TBARS:
            J, D, tot, ft, ff = res["season_table"]["rows"][(Tbar, amp)]
            print(f"      {amp:8.2f}  {Tbar:4.0f}   {J:+9.4f}   {-D:+9.4f}   {tot:+9.4f}     {ft:+.4f} [{ft / D:5.2f}]     {ff:+.4f} [{ff / D:5.2f}]")
    # T8
    print("")
    block("T8")
    print("T8 random, sigma_T = 5: simulated total = J - D at the predicted boundary -2, -1, +1, +2 C")
    okT8 = True
    for phi, roots in res["T8"]["pred"].items():
        for ib, root in enumerate(roots):
            pts = []
            for key in sorted(k for k in res if isinstance(k, tuple) and k[:3] == ("T8", phi, ib)):
                r = res[key]
                pts.append((root + key[3], r["J"] - r["D"], r["se"], r["J"], r["D"]))
            txt = ";  ".join(f"{p_[0]:.2f} C: {p_[1]:+.4f} ± {p_[2]:.4f}" for p_ in pts)
            have = pts
            brack = len(have) >= 2 and have[0][1] * have[-1][1] < 0 and abs(have[0][1]) > 3 * have[0][2] and abs(have[-1][1]) > 3 * have[-1][2]
            okT8 &= brack
            flip = [i for i in range(len(have) - 1) if have[i][1] * have[i + 1][1] < 0]
            if flip:
                i = flip[0]
                est = have[i][0] - have[i][1] * (have[i + 1][0] - have[i][0]) / (have[i + 1][1] - have[i][1])
                where = f"sign changes between {have[i][0]:.2f} and {have[i + 1][0]:.2f} C (linear interpolation {est:.2f} C)"
            else:
                where = "no sign change among the simulated points"
            print(f"      phi = {phi}, predicted {root:.2f} C: {txt};  {where}  [{'bracketed' if brack else 'NOT bracketed'}]")
    report("T8 random: each predicted boundary is bracketed by simulated totals of opposite sign", okT8, "outer points, each more than 3 standard errors from zero")
    print("      (a boundary within 2.5 C of the upper limit of development, 35 C, is simulated at -2 and -1 C and at one third and two thirds of the way to the limit: amendment 4)")
    # T9
    print("\nT9 SHELTER, model TPC, random temperature, sigma_T = 5 (simulated; pairs share their random numbers)")
    print("      Tbar  phi    D both exposed       D only 3 exposed (2 sheltered)   D only 2 exposed (3 sheltered)   sum of the two   share that goes when 2 is sheltered   D with stage 1 split 15 + 15 d   split - whole (paired)")
    for Tbar in (14.0, 30.0):
        for phi in (0.8, 0.99):
            r0 = res[("T7", Tbar, 5.0, phi)]
            r3, r2, rs = res[("T9", "shelter2", Tbar, phi)], res[("T9", "shelter3", Tbar, phi)], res[("T9", "split1", Tbar, phi)]
            dd = rs["Dm"] - r0["Dm"]
            print(f"      {Tbar:4.0f}  {phi:<5}  {r0['D']:+.4f} ± {r0['se']:.4f}    {r3['D']:+.4f} ± {r3['se']:.4f}                {r2['D']:+.4f} ± {r2['se']:.4f}                {r2['D'] + r3['D']:+.4f}          {1 - r3['D'] / r0['D']:+.3f}"
                  f"                               {rs['D']:+.4f} ± {rs['se']:.4f}                {dd.mean():+.4f} ± {dd.std(ddof=1) / math.sqrt(dd.size):.4f}")
            print(f"            formula (c):  {r0['fc']:+.4f}              {r3['fc']:+.4f}                          {r2['fc']:+.4f}                                                                              {rs['fc']:+.4f}")
    r0, rs = res[("T9", "full", 14.0, 0.0)], res[("T9", "split1", 14.0, 0.0)]
    dd = rs["Dm"] - r0["Dm"]
    print(f"        14  0      {r0['D']:+.5f} ± {r0['se']:.5f}  (no memory)   split: {rs['D']:+.5f} ± {rs['se']:.5f};  split - whole (paired) {dd.mean():+.5f} ± {dd.std(ddof=1) / math.sqrt(dd.size):.5f};  formula (c): {r0['fc']:+.5f} and {rs['fc']:+.5f}")
    block("T9s")

    # ------------------------------------------------------------------ Part 2
    print("\n" + "=" * 110 + "\nPART 2: INCUBATION TIME\n" + "=" * 110)
    block("I")
    D_one = L.one_stage_exact(np.full_like(L.ZG, 1 - 1 / A_STAY), (1 / A_STAY) * hz)
    print(f"\nI3 EXACT D at sigma = {SIGMA_H}: stage A geometric, stay {A_STAY:.0f}; stage B steady, mean duration {B_STAY}, chain of m sub-stages (m = 32: fixed delay).")
    print(f"      One stage alone (no stage B), exact by quadrature: D = {D_one:.5f} (second order {SIGMA_H ** 2 / (2 * A_STAY):.5f}).  Bound sigma^2/2 = {SIGMA_H ** 2 / 2}.")
    print("      Each cell: D simulated ± se (second-order value) | parking benefit = D(one stage) - D, with its sign [predicted sign at second order]")
    all_part2 = []
    for mu in (0.0, MU_B):
        print(f"      death rate in stage B: {mu} per step" + ("  (for the fixed delay the staying rates, and so D, do not depend on the death rate: same run)" if mu else ""))
        for m in (1, 2, 4, 16, 32):
            cells = []
            for phi in (0.0, 0.5, 0.9, 0.99):
                r = res[("I3", m, phi, mu if m < 32 else 0.0)]
                g0 = SIGMA_H ** 2 * chain_gamma0(A_STAY, B_STAY, m, mu, phi)
                ben = D_one - r["D"]
                pred = "+" if SIGMA_H ** 2 / (2 * A_STAY) - g0 > 0 else "-"
                sim = "+" if ben > 3 * r["se"] else ("-" if ben < -3 * r["se"] else "0")
                cells.append(f"phi={phi}: {r['D']:.4f} ± {r['se']:.4f} ({g0:.4f}) | {ben:+.4f} {sim} [{pred}]")
                all_part2.append((r["D"], r["se"]))
            print(f"        m = {m:2d}:  " + "   ".join(cells))
    print("")
    ok = True
    det = []
    for phi in (0.0, 0.9):
        r = res[("I4", "A fixed", phi)]
        ok &= abs(r["D"] - 0.5) <= max(3 * r["se"], 0.01)
        r2 = res[("I4", "A chain of 2", phi)]
        r1 = res[("I3", 32, phi, 0.0)]
        det.append(f"phi={phi}: A geometric {r1['D']:.4f} ± {r1['se']:.4f}, A chain of 2 {r2['D']:.4f} ± {r2['se']:.4f}, A fixed {r['D']:.4f} ± {r['se']:.4f}")
        all_part2 += [(r["D"], r["se"]), (r2["D"], r2["se"])]
    report("I4 every stay fixed (A a delay of 4, B a delay of 32): D = sigma^2/2 = 0.5", ok, "; ".join(det))
    det = []
    ok = True
    for m in (4, 32):
        ra, rb = res[("I6", m)], res[("I3", m, 0.9, 0.0)]
        zsc = (ra["D"] - rb["D"]) / math.hypot(ra["se"], rb["se"])
        dd = ra["Dm"] - rb["Dm"]
        ok &= abs(zsc) <= 3
        det.append(f"m={m}: out of B {ra['D']:.4f} ± {ra['se']:.4f}, out of A {rb['D']:.4f} ± {rb['se']:.4f} ({zsc:+.1f} combined se; "
                   f"paired difference, same random numbers, {dd.mean():+.5f} ± {dd.std(ddof=1) / math.sqrt(dd.size):.5f})")
        all_part2.append((ra["D"], ra["se"]))
    report("I6 the loss does not depend on which transfer fluctuates (phi = 0.9, sigma = 1)", ok, "; ".join(det))

    print(f"\nI8 NET MARGIN N(b) = ln R0(b) - D(b), relative to ln R0 with no stage B.  a = {A_STAY:.0f}, sigma = {SIGMA_H}, death {MU_I8} per step in stage B.")
    print(f"      No stage B at all: N(0) = -D(one stage) = {-D_one:+.5f} (exact).")
    for kind, label in (("F", "fixed delay of b steps: ln R0 falls by mu b"), ("G", "geometric stay, mean b without death: ln R0 falls by ln(1 + b(e^mu - 1))")):
        for phi in (0.0, 0.5):
            keys = sorted(k for k in res if isinstance(k, tuple) and k[0] == "I8" and k[1] == kind and k[2] == phi)
            bs = [k[3] for k in keys]
            lnS = [(-MU_I8 * b if kind == "F" else -math.log1p(b * math.expm1(MU_I8))) for b in bs]
            N = np.array([ls_ - res[k]["D"] for ls_, k in zip(lnS, keys)])
            N2 = [(N2_fixed if kind == "F" else N2_geom)(b, A_STAY, SIGMA_H, MU_I8, phi) for b in bs]
            ib = int(np.argmax(N))
            Dms = [res[k]["Dm"] for k in keys]
            for k in keys:
                all_part2.append((res[k]["D"], res[k]["se"]))

            def pse(i, j):
                d = Dms[i] - Dms[j]
                return float(d.std(ddof=1) / math.sqrt(d.size))

            near = [bs[i] for i in range(len(bs)) if i == ib or N[ib] - N[i] <= 2 * pse(i, ib)]
            lo = next((i for i in range(ib - 1, -1, -1) if N[ib] - N[i] > 3 * pse(i, ib)), None)
            hi = next((i for i in range(ib + 1, len(bs)) if N[ib] - N[i] > 3 * pse(i, ib)), None)
            b2 = bs[int(np.argmax(N2))]
            print(f"      {label}; phi = {phi}")
            print("        b:          " + " ".join(f"{b:8d}" for b in bs))
            print("        D exact:    " + " ".join(f"{res[k]['D']:8.4f}" for k in keys))
            print("        se:         " + " ".join(f"{res[k]['se']:8.4f}" for k in keys))
            print("        N exact:    " + " ".join(f"{x:+8.4f}" for x in N))
            print("        N 2nd order:" + " ".join(f"{x:+8.4f}" for x in N2))
            claim = lo is not None and hi is not None
            side = (f"N(b*) - N({bs[lo]}) = {N[ib] - N[lo]:+.5f} ± {pse(lo, ib):.5f}" if lo is not None else "no smaller grid stay is resolved") + "; " + \
                   (f"N(b*) - N({bs[hi]}) = {N[ib] - N[hi]:+.5f} ± {pse(hi, ib):.5f}" if hi is not None else "no larger grid stay is resolved")
            print(f"        exact maximum on the grid at b* = {bs[ib]} (within 2 paired se of the maximum: {near}); second-order maximum on the grid at {b2}; "
                  f"{side}  [{'boundary shown' if claim else 'boundary NOT shown'}]")
            print(f"        N(b*) - N(0) = {N[ib] + D_one:+.4f}: the best steady stage {'beats' if N[ib] + D_one > 0 else 'does not beat'} no steady stage")
    viol = [(D, se) for D, se in all_part2 if D > SIGMA_H ** 2 / 2 + 3 * se]
    report("I5 H4 for chains: D <= sigma^2/2 in every case of Part 2(a) and 2(b)", not viol,
           f"{len(all_part2)} cases, largest D = {max(d for d, _ in all_part2):.4f}, bound {SIGMA_H ** 2 / 2}")

    print("\nI9 MODEL DD, random temperature, sigma_T = 5 (simulated).  P = probability of completing incubation.")
    print("      m  Tbar  phi    ln P(Tbar)   J = ln P(mean rates) - ln P(Tbar)   -D (sim)            total = ln R0* - ln R0(Tbar)   ln P(cohort mean) - ln P(Tbar)   days to complete: const / mean rates / variable        formula D: (a), (c)")
    for m in (1, 4, 16):
        for Tbar in TBARS:
            for phi in (0.8, 0.99):
                r = res[("I9", m, Tbar, phi)]
                lpv = math.log(r["p_var"]) - math.log(r["p_const"])
                print(f"      {m:2d}  {Tbar:4.0f}  {phi:<5}  {math.log(r['p_const']):+8.4f}     {r['J']:+8.4f}                          {-r['D']:+8.4f} ± {r['se']:.4f}   {r['J'] - r['D']:+8.4f} ± {r['se']:.4f}"
                      f"             {lpv:+8.4f} ± {r['p_var_se'] / r['p_var']:.4f}              {r['age_const']:6.2f} / {r['age_mean']:6.2f} / {r['age_var']:6.2f} ± {r['age_var_se']:.2f}"
                      f"          {r['fa']:+.4f}, {r['fc']:+.4f}")
    block("I9s")

    secs = sorted(((r.get("secs", 0.0), k) for k, r in res.items()), reverse=True)
    grp = {}
    for sc_, k in secs:
        nm = k[0] if isinstance(k, tuple) else "exact blocks"
        grp[nm] = grp.get(nm, 0.0) + sc_
    print("\nprocess seconds by group: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(grp.items(), key=lambda kv: -kv[1])))
    print(f"\n{len(res)} tasks; process time {sum(s for s, _ in secs):.0f} s; slowest {secs[0][1]} {secs[0][0]:.0f} s; wall time {time.time() - t_start:.0f} s; "
          f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
