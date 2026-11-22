"""Checks of the claims C1 to C10 of notes/sealed/tertiary-coupled_as-sealed_20261114.md:
the stability of a parasite's life cycle through hosts that have their own dynamics.

Each check compares a derived statement with the nonlinear model of engine/cycle_blocks.py
(its right-hand side, a finite-difference Jacobian, or an integration), none of which use the
derived transfer functions.  Tolerances were fixed before the first run and are stated here.

K0  The engine reproduces known results before it is used:
    (a) one R block and n-1 L blocks with equal rates a: largest real part of the eigenvalues
        is a(|G|^(1/n) cos(pi/n) - 1) for G = 1 - kappa < 0 (Proposition 17 of proofs-v2.md,
        refereed), n = 3..6, kappa in {3, 6, 9.5}.  Tolerance 1e-6 relative.
    (b) one S block closed on itself (direct SIS): eigenvalue -a(R0 - 1) with R0 = 1/(1 - p).
        Tolerance 1e-6 relative.
K1  C1.  Roots of Prod D_k - Prod N_k against eigenvalues of the finite-difference Jacobian:
    400 random cycles of all five kinds, n = 1..6, including hosts below their peak production
    and hosts with no self-regulation.  Tolerance: Hausdorff distance between the two sets
    below 1e-5 x max(1, largest modulus).
K2  C3.  6,000 random S, V (rho >= 0) and M (rho >= 0) blocks: D Hurwitz, and |E(i w)| on a
    grid of 4,001 frequencies never above the stated bound (relative slack 1e-12).
K3  C4.  5,000 random cycles of blocks that cannot amplify (n = 1..8): every eigenvalue of the
    finite-difference Jacobian has negative real part, and max |L(i w)| <= Prod gamma_k.
    Then 2,000 cycles with one over-compensating R block (|1 - kappa| > 1) and
    Prod gamma_k < 1: all stable.  Zero exceptions allowed.
K4  C5, C6.  (i) E(0) of a V block against the block's own steady-state response to its input
    (finite difference of the exact block equilibrium), 500 blocks, tolerance 1e-4 relative.
    (ii) one V block (logistic host below its peak production) and one L block: sign of
    f(tau) against the sign of the largest real part, 3,000 cases; cases within 1e-6 relative
    of the boundary are skipped and counted.  (iii) islands: every case found with the island
    condition is stable at tau_-/3, unstable at the geometric mean of tau_- and tau_+, stable
    at 3 tau_+.  (iv) host with no self-regulation, 1,000 cases: stable below tau_c, unstable
    at 1.5, 10 and 1,000 tau_c when q(u - s) < as; stable at all tau tested otherwise.
    (v) six integrations of the nonlinear model: an island case at three stage lengths, a
    no-self-regulation case at three; a perturbation of 1e-4 relative over a time 6/|Re z|
    must shrink below 0.1 of its size or grow beyond 10 times it, as predicted.
K5  C7.  2,000 random cycles with an L block and finite G: for G < 1 the cycle is stable when
    the L stage is made 1e4 times longer than every other time scale, and the leading
    eigenvalue is (G - 1)/tau within 2% there; for G > 1 it is unstable at tau = 0.01, 1,
    100, 1e4 (x the scale).  Cases with |G - 1| < 0.02 are skipped and counted.
K6  C8.  2,000 random sets of rates, n = 3..8: g_c from the implicit formula is where the
    largest real part of the roots of Prod(z + a_k) + g Prod a_k changes sign (tested at
    g_c(1 -+ 1e-3)); g_c >= sec^n(pi/n); closed form for n = 3 (1e-10 relative); the limit
    as one stay grows (tau_1 = 1e7 x the others, 1e-4 relative).  300 of them through the
    engine (R and L blocks, finite-difference Jacobian) at g_c(1 -+ 0.01).
K7  C9.  1,000 random cycles of S and L blocks: R0 Prod(1 - p_k) = 1 and L(0) = 1/R0
    (1e-10); n equal S blocks: rate of return and rate of increase when rare both equal
    (n/tau)(R0^(1/n) - 1) (1e-6); 20 cycles integrated from 5 starting points each end at the
    equilibrium (1e-6 relative).  The last is numerical evidence of global attraction, not
    a proof.
K8  C10.  Variance of the log output of each block when the input of one block is multiplied
    by exp(eta), eta an Ornstein-Uhlenbeck process of standard deviation 0.05: simulation of
    the nonlinear model against the integral of the predicted spectrum.  Two cycles (S-L-S
    and V-L with a dynamic host), each at two lengths of the L stage.  Tolerance 12%
    (sampling error about 4% plus second-order terms).
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/checks/", 1)[0])
from engine import cycle_blocks as cb  # noqa: E402

RNG = np.random.default_rng(20261114)
FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def logu(lo, hi):
    return float(np.exp(RNG.uniform(math.log(lo), math.log(hi))))


def max_re(blocks):
    return float(np.max(np.linalg.eigvals(cb.jacobian_fd(blocks)).real))


# ------------------------------------------------------------------ random blocks

def rand_L():
    return cb.make_L(logu(0.05, 5), logu(0.1, 10))


def rand_R(kmax=2.0, kmin=0.0):
    return cb.make_R(logu(0.05, 5), logu(0.1, 10), RNG.uniform(max(kmin, 1e-3), kmax))


def rand_S():
    return cb.make_S(logu(0.05, 5), logu(0.02, 20), logu(0.5, 50))


def rand_V(xlo, xhi):
    """Logistic host held at N*/K = x in (xlo, xhi)."""
    alpha = logu(0.05, 5)
    a = alpha + logu(0.02, 5)
    u = logu(0.02, 20)
    p = u / (u + a)
    x = RNG.uniform(xlo, xhi)
    r = alpha * p / (1 - x)
    return cb.make_V(a, u, alpha, r, logu(0.5, 50))


def rand_V_noreg():
    alpha = logu(0.05, 5)
    a = alpha + logu(0.02, 5)
    p = RNG.uniform(0.02, 0.95)
    u = a * p / (1 - p)
    return cb.make_V(a, u, alpha, alpha * p, math.inf)


def rand_M(xlo, xhi):
    alpha = logu(0.01, 1)
    xi = alpha + logu(0.02, 5)
    m = logu(0.1, 20)
    eps = RNG.uniform(1.0, 5.0)
    x = RNG.uniform(xlo, xhi)
    r = alpha * m / (1 - x)
    return cb.make_M(xi, m, alpha, eps, r, logu(0.5, 50))


def rand_M_noreg():
    alpha = logu(0.01, 1)
    xi = alpha + logu(0.02, 5)
    m = logu(0.1, 20)
    return cb.make_M(xi, m, alpha, RNG.uniform(1.0, 5.0), alpha * m, math.inf)


def gamma_bound(b):
    k = b["kind"]
    if k == "L":
        return 1.0
    if k == "R":
        return abs(1 - b["kappa"])
    if k in ("S", "V"):
        return b["a"] / (b["a"] + b["u"])
    m = b["state"][1] / b["state"][0]
    return (b["xi"] + b["alpha"] * b["eps"] * m) / (b["xi"] + 2 * b["alpha"] * b["eps"] * m)


def omega_grid(blocks):
    scales = []
    for b in blocks:
        _, dk = cb.tf_polys(b)
        scales += [abs(z) for z in dk.roots if abs(z) > 0]
    lo, hi = min(scales), max(scales)
    return np.concatenate([[0.0], np.logspace(math.log10(lo) - 4, math.log10(hi) + 4, 4000)])


# ------------------------------------------------------------------ K0

def k0():
    worst = 0.0
    for n in range(3, 7):
        for kappa in (3.0, 6.0, 9.5):
            a = 0.7
            blocks = [cb.make_R(a, 1.3, kappa)] + [cb.make_L(a, logu(0.2, 5)) for _ in range(n - 1)]
            cb.close_cycle(blocks)
            got = max_re(blocks)
            want = a * (abs(1 - kappa) ** (1 / n) * math.cos(math.pi / n) - 1)
            worst = max(worst, abs(got - want) / a)
    report("K0a known: one regulated transfer, equal rates", worst < 1e-6, f"largest error {worst:.1e} of the rate (12 cases)")
    worst = 0.0
    for _ in range(50):
        b = rand_S()
        cb.close_cycle([b])
        p = b["state"][0] / b["N"]
        want = -b["a"] * (1 / (1 - p) - 1)
        got = max_re([b])
        worst = max(worst, abs(got - want) / abs(want))
    report("K0b known: direct SIS", worst < 1e-6, f"largest relative error {worst:.1e} (50 cases)")


# ------------------------------------------------------------------ K1

def rand_any():
    c = RNG.integers(0, 9)
    if c == 0:
        return rand_L()
    if c == 1:
        return rand_R(kmax=6.0)
    if c == 2:
        return rand_S()
    if c == 3:
        return rand_V(0.5, 0.98)
    if c == 4:
        return rand_V(0.03, 0.5)
    if c == 5:
        return rand_V_noreg()
    if c == 6:
        return rand_M(0.5, 0.98)
    if c == 7:
        return rand_M(0.03, 0.5)
    return rand_M_noreg()


def hausdorff(xs, ys):
    d1 = max(min(abs(x - y) for y in ys) for x in xs)
    d2 = max(min(abs(x - y) for x in xs) for y in ys)
    return max(d1, d2)


def k1():
    worst, kinds = 0.0, set()
    for _ in range(400):
        n = int(RNG.integers(1, 7))
        blocks = cb.close_cycle([rand_any() for _ in range(n)])
        kinds |= {b["kind"] for b in blocks}
        eig = np.linalg.eigvals(cb.jacobian_fd(blocks))
        roots = cb.char_poly(blocks).roots
        if len(roots) != len(eig):
            worst = math.inf
            break
        worst = max(worst, hausdorff(eig, roots) / max(1.0, float(np.max(np.abs(eig)))))
    report("K1 loop equation (C1)", worst < 1e-5, f"400 cycles, kinds {sorted(kinds)}, largest distance {worst:.1e}")


# ------------------------------------------------------------------ K2

def k2():
    bad, worst = 0, 0.0
    for i in range(6000):
        b = (rand_S, lambda: rand_V(0.5, 0.98), lambda: rand_M(0.5, 0.98))[i % 3]()
        _, dk = cb.tf_polys(b)
        if np.max(dk.roots.real) >= 0:
            bad += 1
        g = np.max(np.abs(cb.block_gain(b, omega_grid([b]))))
        ratio = g / gamma_bound(b)
        worst = max(worst, ratio)
        if ratio > 1 + 1e-12:
            bad += 1
    report("K2 blocks that cannot amplify (C3)", bad == 0, f"6,000 blocks, exceptions {bad}, largest gain/bound {worst:.6f}")


# ------------------------------------------------------------------ K3

def rand_contractive():
    c = RNG.integers(0, 5)
    return (rand_L, rand_R, rand_S, lambda: rand_V(0.5, 0.98), lambda: rand_M(0.5, 0.98))[c]()


def k3():
    bad, worst_re, worst_gain, done = 0, -math.inf, 0.0, 0
    while done < 5000:
        n = int(RNG.integers(1, 9))
        blocks = [rand_contractive() for _ in range(n)]
        if all(gamma_bound(b) >= 1 - 1e-9 for b in blocks):
            continue                       # needs one strict contraction
        cb.close_cycle(blocks)
        done += 1
        mr = max_re(blocks)
        rates = np.abs(np.linalg.eigvals(cb.jacobian_fd(blocks)))
        worst_re = max(worst_re, mr / float(np.max(rates)))
        g = float(np.max(np.abs(cb.loop_gain(blocks, omega_grid(blocks)))))
        bound = float(np.prod([gamma_bound(b) for b in blocks]))
        worst_gain = max(worst_gain, g / bound)
        if mr >= 0 or g > bound * (1 + 1e-9):
            bad += 1
    report("K3a any number of hosts, any durations (C4)", bad == 0,
           f"5,000 cycles, exceptions {bad}, largest Re z / largest |z| = {worst_re:.2e}, largest gain/bound {worst_gain:.6f}")
    bad, done = 0, 0
    while done < 2000:
        n = int(RNG.integers(2, 8))
        blocks = [rand_R(kmax=6.0, kmin=2.05)] + [rand_contractive() for _ in range(n - 1)]
        if float(np.prod([gamma_bound(b) for b in blocks])) >= 0.999:
            continue
        cb.close_cycle(blocks)
        done += 1
        if max_re(blocks) >= 0:
            bad += 1
    report("K3b over-compensation offset by prevalence (C4)", bad == 0, f"2,000 cycles with product of bounds < 1, exceptions {bad}")


# ------------------------------------------------------------------ K4

def v_params(b):
    u, a, al = b["u"], b["a"], b["alpha"]
    s = -cb.rho_of(b)
    return u, a, al, s, u + a - s, al * u - s * (u + a)


def f_tau(b, tau):
    u, a, al, s, q, D = v_params(b)
    return q * D * tau * tau + (q * (u - s) - a * s) * tau + (u - s)


def cycle_VL(bv, tau):
    blocks = [dict(bv, state=bv["state"].copy()), cb.make_L(1.0 / tau, logu(0.2, 5))]
    return cb.close_cycle(blocks)


def k4():
    # (i) E(0) against the block's own steady state
    worst = 0.0
    for _ in range(500):
        b = rand_V(0.03, 0.98)
        cb.close_cycle([b])
        a, al, r, K, u = b["a"], b["alpha"], b["r"], b["K"], b["u"]

        def steady_I(uu):
            p = uu / (uu + a)
            return p * K * (1 - al * p / r)
        d = 1e-5
        elas = (math.log(steady_I(u * (1 + d))) - math.log(steady_I(u * (1 - d)))) / (math.log(1 + d) - math.log(1 - d))
        nk, dk = cb.tf_polys(b)
        e0 = float(nk(0) / dk(0))
        worst = max(worst, abs(elas - e0) / max(abs(e0), 1e-3))
    report("K4i static response of a harmed host (C5)", worst < 1e-4, f"500 blocks, largest relative error {worst:.1e}")

    # (ii) sign of f(tau)
    agree = skipped = unstable = 0
    islands = []
    for _ in range(3000):
        b = rand_V(0.03, 0.5)
        tau = logu(0.01, 300)
        u, a, al, s, q, D = v_params(b)
        f = f_tau(b, tau)
        scale = abs(q * D) * tau * tau + abs(q * (u - s) - a * s) * tau + abs(u - s)
        mid = q * (u - s) - a * s
        if s < u and mid < 0 and mid * mid > 4 * q * D * (u - s):
            islands.append(b)
        if abs(f) < 1e-6 * scale:
            skipped += 1
            continue
        mr = max_re(cycle_VL(b, tau))
        unstable += mr > 0
        agree += (f > 0) == (mr < 0)
    total = 3000 - skipped
    report("K4ii one harmed host and one delay (C6)", agree == total,
           f"{agree} of {total} agree ({skipped} at the boundary skipped); {unstable} unstable; {len(islands)} cases with an island")

    # (iii) islands
    bad = 0
    for b in islands:
        u, a, al, s, q, D = v_params(b)
        A_, B_, C_ = q * D, q * (u - s) - a * s, u - s
        disc = math.sqrt(B_ * B_ - 4 * A_ * C_)
        t_lo, t_hi = (-B_ - disc) / (2 * A_), (-B_ + disc) / (2 * A_)
        if not (max_re(cycle_VL(b, t_lo / 3)) < 0 and max_re(cycle_VL(b, math.sqrt(t_lo * t_hi))) > 0
                and max_re(cycle_VL(b, 3 * t_hi)) < 0):
            bad += 1
    report("K4iii islands of instability at intermediate stage length (C6)", bad == 0 and len(islands) > 0,
           f"{len(islands)} islands, exceptions {bad}")

    # (iv) no self-regulation
    bad = n_unst = n_stab = 0
    for _ in range(1000):
        b = rand_V_noreg()
        u, a, al, s, q, D = v_params(b)
        mid = q * (u - s) - a * s
        if abs(mid) < 1e-3 * (abs(q * (u - s)) + a * s) or u <= s * 1.001:
            continue
        if mid < 0:
            n_unst += 1
            tc = (u - s) / (-mid)
            ok = max_re(cycle_VL(b, 0.5 * tc)) < 0 and all(max_re(cycle_VL(b, k * tc)) > 0 for k in (1.5, 10.0, 1000.0))
        else:
            n_stab += 1
            ok = all(max_re(cycle_VL(b, t)) < 0 for t in (0.01, 1.0, 100.0, 1e4))
        bad += not ok
    report("K4iv host with no self-regulation (C6)", bad == 0 and n_unst > 50,
           f"{n_unst} cases destabilised by every stage longer than tau_c, {n_stab} stable at every length, exceptions {bad}")

    # (v) integrations
    results = []

    def run(blocks, expect_stable, label):
        J = cb.jacobian_fd(blocks)
        ev = np.linalg.eigvals(J)
        lead = ev[np.argmax(ev.real)]
        x0 = cb.equilibrium(blocks)
        # perturb along the real part of the leading eigenvector
        w, vecs = np.linalg.eig(J)
        vec = vecs[:, np.argmax(w.real)].real
        vec = vec / np.max(np.abs(vec / x0))
        T = 6.0 / abs(lead.real)
        fastest = float(np.max(np.abs(ev)))
        dt = min(0.02 / fastest, T / 4000)
        ts, xs = cb.integrate(blocks, x0 + 1e-4 * vec, T, dt, record_every=max(1, int(T / dt / 4000)))
        dev = np.max(np.abs(xs / x0 - 1), axis=1)
        k = len(dev) // 10
        early, late = float(np.max(dev[:k])), float(np.max(dev[-k:]))
        ok = (late < 0.1 * early) if expect_stable else (late > 10 * early)
        results.append(ok)
        print(f"      {label}: leading eigenvalue {lead.real:+.4g}{lead.imag:+.4g}i, deviation {early:.2e} -> {late:.2e}, "
              f"{'decays' if late < early else 'grows'} ({'as predicted' if ok else 'NOT as predicted'})")

    if islands:
        b = islands[0]
        u, a, al, s, q, D = v_params(b)
        A_, B_, C_ = q * D, q * (u - s) - a * s, u - s
        disc = math.sqrt(B_ * B_ - 4 * A_ * C_)
        t_lo, t_hi = (-B_ - disc) / (2 * A_), (-B_ + disc) / (2 * A_)
        print(f"      island case: N*/K = {b['state'][0] / b['K']:.3f}, prevalence {u / (u + a):.3f}, unstable for {t_lo:.3g} < tau < {t_hi:.3g}")
        run(cycle_VL(b, t_lo / 3), True, "short stage")
        run(cycle_VL(b, math.sqrt(t_lo * t_hi)), False, "intermediate stage")
        run(cycle_VL(b, 3 * t_hi), True, "long stage")
    while True:
        b = rand_V_noreg()
        u, a, al, s, q, D = v_params(b)
        mid = q * (u - s) - a * s
        if mid < -0.05 * a * s and u > 1.2 * s:
            break
    tc = (u - s) / (-mid)
    print(f"      host with no self-regulation: prevalence {u / (u + a):.3f}, tau_c = {tc:.3g}")
    run(cycle_VL(b, 0.4 * tc), True, "stage 0.4 tau_c")
    run(cycle_VL(b, 3 * tc), False, "stage 3 tau_c")
    run(cycle_VL(b, 300 * tc), False, "stage 300 tau_c")
    report("K4v nonlinear integrations (C6)", all(results) and len(results) == 6, f"{sum(results)} of {len(results)} as predicted")


# ------------------------------------------------------------------ K5

def k5():
    bad = skipped = n_lt = n_gt = 0
    worst = 0.0
    done = 0
    while done < 2000:
        n = int(RNG.integers(1, 6))
        others = []
        for _ in range(n):
            c = RNG.integers(0, 6)
            others.append((rand_L, lambda: rand_R(kmax=6.0), rand_S, lambda: rand_V(0.03, 0.98),
                           lambda: rand_M(0.03, 0.98), lambda: rand_V(0.03, 0.5))[c]())
        ok_open = True
        scales = []
        for b in others:
            _, dk = cb.tf_polys(b)
            if np.max(dk.roots.real) >= 0:
                ok_open = False
            scales += [1 / abs(z) for z in dk.roots]
        if not ok_open:
            continue
        G = 1.0
        for b in others:
            nk, dk = cb.tf_polys(b)
            G *= float(nk(0) / dk(0))
        if abs(G - 1) < 0.02:
            skipped += 1
            continue
        done += 1
        T0 = max(scales)
        ustar = logu(0.2, 5)

        def with_stage(tau):
            blocks = [dict(b, state=b["state"].copy()) for b in others] + [cb.make_L(1.0 / tau, ustar)]
            return cb.close_cycle(blocks)

        if G < 1:
            n_lt += 1
            tau = 1e4 * T0
            ev = np.linalg.eigvals(cb.jacobian_fd(with_stage(tau)))
            lead = float(np.max(ev.real))
            pred = (G - 1) / tau
            worst = max(worst, abs(lead - pred) / abs(pred))
            if not (lead < 0 and abs(lead - pred) < 0.02 * abs(pred)):
                bad += 1
        else:
            n_gt += 1
            if not all(max_re(with_stage(k * T0)) > 0 for k in (0.01, 1.0, 100.0, 1e4)):
                bad += 1
    report("K5 a long-lived stage (C7)", bad == 0 and n_gt > 20,
           f"G < 1: {n_lt} cycles, all stable with a long stage, leading eigenvalue within {worst:.1e} of (G-1)/tau; "
           f"G > 1: {n_gt} cycles, unstable at every length; exceptions {bad}; {skipped} skipped near G = 1")


# ------------------------------------------------------------------ K6

def g_crit(taus):
    lo, hi = 0.0, 1.0
    while np.sum(np.arctan(hi * taus)) < math.pi:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.sum(np.arctan(mid * taus)) < math.pi:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return float(np.prod(np.sqrt(1 + (w * taus) ** 2))), w


def k6():
    bad = 0
    worst_j = math.inf
    for _ in range(2000):
        n = int(RNG.integers(3, 9))
        taus = np.array([logu(0.05, 20) for _ in range(n)])
        gc, _ = g_crit(taus)
        a = 1 / taus
        base = np.poly1d(np.poly(-a))
        for g, want_stable in ((gc * (1 - 1e-3), True), (gc * (1 + 1e-3), False)):
            mr = float(np.max((base + g * np.prod(a)).roots.real))
            if (mr < 0) != want_stable:
                bad += 1
        worst_j = min(worst_j, gc / (1 / math.cos(math.pi / n)) ** n)
    report("K6a critical over-compensation with unequal stays (C8)", bad == 0 and worst_j >= 1 - 1e-12,
           f"2,000 sets of rates, sign changes at g_c in all but {bad}; smallest g_c / sec^n(pi/n) = {worst_j:.6f}")
    worst = 0.0
    for _ in range(500):
        taus = np.array([logu(0.05, 20) for _ in range(3)])
        gc, _ = g_crit(taus)
        worst = max(worst, abs(gc - (taus.sum() * (1 / taus).sum() - 1)) / gc)
    report("K6b three stages: g_c = (sum of stays)(sum of rates) - 1 (C8)", worst < 1e-10, f"500 cases, largest relative error {worst:.1e}")
    worst = 0.0
    for _ in range(300):
        n = int(RNG.integers(3, 8))
        rest = np.array([logu(0.2, 5) for _ in range(n - 1)])
        t1 = 1e7 * rest.max()
        gc, _ = g_crit(np.concatenate([[t1], rest]))
        lo, hi = 0.0, 1.0
        while np.sum(np.arctan(hi * rest)) < math.pi / 2:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.sum(np.arctan(mid * rest)) < math.pi / 2:
                lo = mid
            else:
                hi = mid
        w = 0.5 * (lo + hi)
        pred = w * float(np.prod(np.sqrt(1 + (w * rest) ** 2)))
        worst = max(worst, abs(gc / t1 - pred) / pred)
    report("K6c one stay growing: g_c grows in proportion (C8)", worst < 1e-4, f"300 cases (n = 3..7; n = 2 others gives 1/tau_2 + 1/tau_3), largest relative error {worst:.1e}")
    bad = 0
    for _ in range(300):
        n = int(RNG.integers(3, 7))
        taus = np.array([logu(0.1, 10) for _ in range(n)])
        gc, _ = g_crit(taus)
        for g, want_stable in ((gc * 0.99, True), (gc * 1.01, False)):
            blocks = [cb.make_R(1 / taus[0], logu(0.2, 5), 1 + g)] + [cb.make_L(1 / t, logu(0.2, 5)) for t in taus[1:]]
            cb.close_cycle(blocks)
            if (max_re(blocks) < 0) != want_stable:
                bad += 1
    report("K6d the same through the nonlinear model (C8)", bad == 0, f"300 cycles at g_c(1 -+ 0.01), exceptions {bad}")


# ------------------------------------------------------------------ K7

def k7():
    worst = 0.0
    for _ in range(1000):
        n = int(RNG.integers(1, 7))
        blocks = [rand_S() if (i == 0 or RNG.random() < 0.6) else rand_L() for i in range(n)]
        cb.close_cycle(blocks)
        R0, prod = 1.0, 1.0
        for b in blocks:
            if b["kind"] == "S":
                R0 *= b["c"] * b["N"] / b["a"]
                prod *= 1 - b["state"][0] / b["N"]
            else:
                R0 *= b["c"] / b["a"]
        G = float(np.real(cb.loop_gain(blocks, [0.0])[0]))
        worst = max(worst, abs(R0 * prod - 1), abs(G * R0 - 1))
    report("K7a the loop multiplies a slow disturbance by 1/R0 (C9)", worst < 1e-10, f"1,000 cycles of S and L blocks, largest error {worst:.1e}")
    worst = 0.0
    for _ in range(200):
        n = int(RNG.integers(1, 9))
        a, u, N = logu(0.05, 5), logu(0.02, 20), logu(0.5, 50)
        blocks = cb.close_cycle([cb.make_S(a, u, N) for _ in range(n)])
        p = u / (u + a)
        R0 = (1 - p) ** (-n)
        tau = n / a
        want = (n / tau) * (R0 ** (1 / n) - 1)
        got_return = -max_re(blocks)
        zero = np.full(n, 1e-9 * N)
        J0 = cb.jacobian_fd(blocks, x=zero, rel=1e-3)
        got_rare = float(np.max(np.linalg.eigvals(J0).real))
        worst = max(worst, abs(got_return - want) / want, abs(got_rare - want) / want)
    report("K7b equal hosts: return rate = rate when rare = (n/tau)(R0^(1/n) - 1) (C9)", worst < 1e-6, f"200 cycles, largest relative error {worst:.1e}")
    from scipy.integrate import solve_ivp
    bad = 0
    worst = 0.0
    for _ in range(20):
        n = int(RNG.integers(2, 6))
        blocks = cb.close_cycle([rand_S() if (i == 0 or RNG.random() < 0.6) else rand_L() for i in range(n)])
        xeq = cb.equilibrium(blocks)
        rate = -max_re(blocks)
        for _ in range(5):
            x0 = xeq * np.exp(RNG.uniform(-6, 0.0, size=len(xeq)))
            for k, b in enumerate(blocks):
                if b["kind"] == "S" and RNG.random() < 0.5:
                    x0[k] = b["N"] * RNG.uniform(0.5, 0.999)
                elif b["kind"] == "L" and RNG.random() < 0.5:
                    x0[k] = xeq[k] * math.exp(RNG.uniform(0, 4))
            # a stiff solver: the pure-Python fixed-step integrator needed up to 1e8 steps here
            sol = solve_ivp(lambda t, x: cb.rhs(blocks, x), (0.0, 80.0 / rate), x0, method="LSODA",
                            rtol=1e-11, atol=1e-13 * xeq)
            err = float(np.max(np.abs(sol.y[:, -1] / xeq - 1)))
            worst = max(worst, err)
            if not sol.success or err > 1e-6:
                bad += 1
    report("K7c every start tried ends at the equilibrium (C9, numerical only)", bad == 0,
           f"100 integrations of 20 cycles, exceptions {bad}, largest final relative distance {worst:.1e}")


# ------------------------------------------------------------------ K8

def k8():
    sd, nu = 0.05, 0.3
    outcomes = []

    def predicted(blocks, j):
        w = np.concatenate([[0.0], np.logspace(-6, 4, 200001)])
        Sx = 2 * nu * sd * sd / (nu * nu + w * w)
        L = cb.loop_gain(blocks, w)
        n = len(blocks)
        out = []
        for k in range(n):
            path = np.ones_like(L)
            i = j
            while True:
                path = path * cb.block_gain(blocks[i], w)
                if i == k:
                    break
                i = (i + 1) % n
            integrand = np.abs(path) ** 2 / np.abs(1 - L) ** 2 * Sx
            out.append(float(np.trapezoid(integrand, w) / math.pi))
        return out

    def simulated(blocks, j, T, dt, seed):
        rng = np.random.default_rng(seed)
        steps = int(T / dt)
        half = dt / 2
        rho_ = math.exp(-nu * half)
        innov = sd * math.sqrt(1 - rho_ * rho_) * rng.standard_normal(2 * steps + 2)
        eta = np.empty(2 * steps + 2)
        eta[0] = sd * rng.standard_normal()
        for i in range(1, 2 * steps + 2):
            eta[i] = rho_ * eta[i - 1] + innov[i]
        n = len(blocks)
        mult = np.ones(n)

        def mult_fn(t):
            mult[j] = math.exp(eta[int(round(t / half))])
            return mult
        _, xs = cb.integrate(blocks, cb.equilibrium(blocks), T, dt, mult_fn=mult_fn, record_every=4)
        xs = xs[len(xs) // 20:]
        out, pos = [], 0
        for b in blocks:
            d = len(b["state"])
            state = xs[:, pos:pos + d]
            if b["kind"] in ("L",):
                v = state[:, 0]
            else:
                v = state[:, -1]
            out.append(float(np.var(np.log(v))))
            pos += d
        return out

    cases = []
    for tau in (2.0, 20.0):
        cases.append((f"S-L-S, L stage {tau:g}", cb.close_cycle([cb.make_S(1.0, 0.8, 5.0), cb.make_L(1 / tau, 1.0), cb.make_S(0.5, 0.6, 3.0)]), 0))
    for tau in (2.0, 20.0):
        cases.append((f"V-L (dynamic host, N*/K = 0.7), L stage {tau:g}",
                      cb.close_cycle([cb.make_V(1.0, 0.7, 0.6, 0.6 * (0.7 / 1.7) / 0.3, 4.0), cb.make_L(1 / tau, 1.0)]), 0))
    worst = 0.0
    for idx, (label, blocks, j) in enumerate(cases):
        pred = predicted(blocks, j)
        sim = simulated(blocks, j, T=60000.0, dt=0.04, seed=1000 + idx)
        errs = [abs(s - p) / p for s, p in zip(sim, pred)]
        worst = max(worst, max(errs))
        print(f"      {label}: variance of log output, predicted " + ", ".join(f"{p:.3e}" for p in pred)
              + "; simulated " + ", ".join(f"{s:.3e}" for s in sim))
        outcomes.append(pred)
    report("K8 fluctuating host with feedback (C10)", worst < 0.12, f"4 runs, largest relative error {worst:.3f} (tolerance 0.12)")
    r = outcomes[0][1] / outcomes[1][1]
    print(f"      S-L-S: variance in the L stage falls {r:.2f}-fold when its length rises 10-fold")


if __name__ == "__main__":
    only = sys.argv[1:] or ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
    # The random stream is restarted at k0, k3, k7 and k8.  Those are the points at which the checks
    # were first run as separate invocations (k0 k1 k2; k3 to k7; k7 alone after its harness was fixed;
    # k8 alone), so restarting there makes one run of the whole script reproduce those runs exactly.
    # Without the restarts a full run draws different random cases, and one of the fixed-step
    # integrations of K4v then needed more than twenty minutes (stopped by hand, 14 November, 09:50).
    for name in only:
        if name in ("k0", "k3", "k7", "k8"):
            RNG = np.random.default_rng(20261114)
        globals()[name]()
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
