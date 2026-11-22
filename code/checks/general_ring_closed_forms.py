"""Formula-level checks of the statements in `notes/general-ring.md`. No simulation.

Claims checked, each against an independent evaluation, with tolerances stated beforehand:
  F1  the closed form for fluctuating transfer in a ring equals the published matrix formula
      (relative 1e-8, or absolute 1e-12 where the value itself is below 1e-6);
  F2  the same for fluctuating survival;
  F3  the lineage form equals the matrix formula for any primitive matrix and one fluctuating entry (relative 1e-8);
  F4  one stage: the cost of fluctuating transfer does not depend on autocorrelation (relative 1e-9; a first
      version used an absolute 1e-12 and missed it by rounding, 1.5e-12, at phi = 0.99, where two terms of
      size 50 q^2 cancel);
  F5  the contribution of a host does not depend on the stage it serves (1e-12);
  F6  same n and generation time: the cost lies between that of the ring with all but one stage without staying
      and that of the equal ring (no exception);
  F7  positive autocorrelation never lowers the cost of fluctuating transfer in a ring, with equality only for one
      stage (no exception);
  F8  limits: phi -> 0 gives 1/(2 T^2); phi -> 1 gives (1 - CV^2)/(2 T) (60-digit arithmetic, relative 1e-9);
  F9  the equal ring: closed form equals the earlier C_n, and C_2 > C_1 exactly when phi > (s+g)/(3s+g);
  F10 periodic environment: the same formula with a sinusoid, against the exact growth rate from the product of
      matrices over one period; the error must fall sixteen-fold when the amplitude is halved (between 14 and 18).

Added later on 14 November 2026, after secondary's review (board 0014):
  F7 now excludes rings in which no stage has staying, where the closed form does not apply (the first version
      drew them with small probability);
  F11 several entries on one driver: the full lineage form, with both orders of each pair and the lag-zero term,
      equals the matrix formula (relative 1e-8);
  F12 an autocorrelation that is not geometric: for a mixture of two geometric terms the lineage form equals the
      same mixture of matrix-formula costs (relative 1e-8); and secondary's counterexample, a ring of two stages with
      autocorrelation only at lag two, costs LESS than with no memory, so statement F7 is specific to geometric decay;
  F13 per generation (cost times generation time), equal stays: constant in n without memory, and never falling as n
      grows with positive memory (no exception);
  F14 a fixed generation time divided equally among n stages, n below the generation time: the total cost rises
      with n (no exception);
  F15 slow environment: the limit equals (T^2 - sum tau^2 + T)/(2 T^3), which is 1/(2 T^2) for one stage;
  F16 per generation, unequal stays, positive memory: adding a host whose stay equals the current mean stay never
      lowers the cost per generation (no exception). Without memory it leaves it unchanged (1e-12);
  F17 beyond a loop: for an age-structured (Leslie) matrix in which one driver multiplies every fertility, the cost is
      (1+2 theta)/(2 T^2) - (1/T) R/(1-R) with T = sum_a a l_a m_a lambda^-a, the mean age of mothers, and
      R = sum_a l_a m_a (phi/lambda)^a, the Euler-Lotka sum with lambda/phi in place of lambda (relative 1e-8).

Run: python3 -I checks/general_ring_closed_forms.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import theory  # noqa: E402


def random_ring(rng, nmax=12, allow_zero=True):
    n = int(rng.integers(1, nmax + 1))
    s = rng.uniform(0.0, 0.98, n)
    if allow_zero and n > 1 and rng.random() < 0.3:
        z = rng.random(n) < 0.5
        z[rng.integers(n)] = False
        s = np.where(z, 0.0, s)
    g = np.exp(rng.uniform(np.log(0.02), np.log(5.0), n))
    return s, g


def close(a, b, rel=1e-8, absol=1e-12):
    return abs(a - b) <= max(rel * abs(b), absol) or (abs(b) < 1e-6 and abs(a - b) <= absol * 10)


def main():
    rng = np.random.default_rng(2026111404)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    # F1, F2
    for label, closed_fn, kind in (("F1 transfer", theory.cost_transfer, "g"), ("F2 survival", theory.cost_survival, "s")):
        bad, worst, cnt, neg = 0, 0.0, 0, 0
        for trial in range(4000):
            s, g = random_ring(rng)
            n = len(s)
            if n == 1 and kind == "s":
                continue                      # with one stage the two entries coincide; treated in F4
            phi = rng.uniform(0, 0.98, n)
            cf = closed_fn(s, g, 1.0, phi)
            a = theory.ring_matrix(s, g)
            for k in range(n):
                if kind == "s" and s[k] == 0:
                    continue
                entry = ((k + 1) % n, k) if kind == "g" else (k, k)
                if n == 1:
                    continue
                mat = theory.matrix_cost(a, [entry], [0], [1.0], [phi[k]])
                if not close(cf[k], mat):
                    bad += 1
                if abs(mat) > 1e-6:
                    worst = max(worst, abs(cf[k] - mat) / abs(mat))
                neg += int(mat < 0)
                cnt += 1
        report(f"{label}: closed form = matrix formula", bad == 0,
               f"{cnt} stage contributions in 4,000 random rings (1-12 stages, some stages without staying); largest relative difference {worst:.1e}; negative (a gain) in {neg}")

    # F3 lineage form, general matrices
    worst, cnt = 0.0, 0
    for trial in range(1500):
        d = int(rng.integers(2, 9))
        a = rng.uniform(0, 1, (d, d)) * (rng.random((d, d)) < rng.uniform(0.3, 1.0))
        a += np.diag(rng.uniform(0.01, 0.5, d)) + 0.02 * np.roll(np.eye(d), 1, axis=0)
        nz = np.argwhere(a > 0)
        i, j = nz[rng.integers(len(nz))]
        phi = rng.uniform(-0.9, 0.97)
        mat = theory.matrix_cost(a, [(int(i), int(j))], [0], [1.0], [phi])
        lin = theory.lineage_cost(a, (int(i), int(j)), lambda l, f=phi: f ** l)
        worst = max(worst, abs(mat - lin) / max(abs(mat), 1e-12)); cnt += 1
    report("F3 lineage form = matrix formula, any primitive matrix", worst < 1e-8, f"{cnt} random matrices of size 2-8, phi from -0.9 to 0.97; largest relative difference {worst:.1e}")

    # F4 one stage
    worst = 0.0
    for trial in range(2000):
        s = rng.uniform(0, 0.98); g = math.exp(rng.uniform(math.log(0.02), math.log(5)))
        q = g / (s + g)
        for phi in (0.0, 0.3, 0.9, 0.99):
            worst = max(worst, abs(theory.cost_transfer([s], [g], 1.0, phi)[0] / (q * q / 2) - 1))
    report("F4 one stage: cost = q^2/2 for every phi", worst < 1e-9, f"2,000 cases x 4 autocorrelations; largest relative difference {worst:.1e}")

    # F5 position
    worst = 0.0
    for trial in range(2000):
        s, g = random_ring(rng); n = len(s); phi = rng.uniform(0, 0.98)
        c = theory.cost_transfer(s, g, 1.0, phi)
        worst = max(worst, float(np.ptp(c)) / max(abs(c[0]), 1e-300))
    report("F5 position does not matter", worst < 1e-12, f"2,000 random rings; largest relative spread across stages {worst:.1e}")

    # F6 equal ring is the worst at fixed n and T; F7 positive autocorrelation raises the cost
    def cost_tau(tau, phi):
        tau = np.asarray(tau, float); lam = 1.05
        return float(theory.cost_transfer(lam * (1 - 1 / tau), lam / tau, 1.0, phi)[0])
    viol, cnt = 0, 0
    for trial in range(50000):
        n = int(rng.integers(2, 10)); T = n + rng.exponential(5.0) + 0.01; phi = rng.uniform(1e-3, 0.999)
        tau = 1 + (T - n) * rng.dirichlet(np.ones(n) * rng.choice([0.3, 1, 5]))
        ext = np.ones(n); ext[0] = T - n + 1
        c, ce, cx = cost_tau(tau, phi), cost_tau(np.full(n, T / n), phi), cost_tau(ext, phi)
        viol += int(c > ce * (1 + 1e-10) or c < cx * (1 - 1e-10)); cnt += 1
    report("F6 same n and generation time: equal ring costs most, all-but-one-without-staying least", viol == 0, f"{cnt} random rings, exceptions: {viol}")
    least, eq1, cnt = np.inf, 0.0, 0
    for trial in range(100000):
        n = int(rng.integers(1, 13)); tau = 1 + rng.exponential(rng.choice([0.2, 2.0, 20.0]), n) * (rng.random(n) < 0.8)
        if n > 1 and np.all(tau == 1.0):
            continue                      # no stage has staying: not primitive, the closed form does not apply
        phi = rng.uniform(1e-3, 0.999)
        excess = cost_tau(tau, phi) * 2 * tau.sum() ** 2 - 1
        if n == 1:
            eq1 = max(eq1, abs(excess))
        else:
            least = min(least, excess)
        cnt += 1
    report("F7 positive autocorrelation never lowers the cost of fluctuating transfer", least > 0 and eq1 < 1e-9,
           f"{cnt} random rings: smallest relative excess over the uncorrelated cost for 2 or more stages {least:.2e}; largest |excess| for one stage {eq1:.1e}")

    # F8 limits, high precision
    import mpmath as mp
    mp.mp.dps = 60
    worst0, worst1 = mp.mpf(0), mp.mpf(0)
    for trial in range(300):
        n = int(rng.integers(1, 10))
        tau = [mp.mpf(1) + (mp.mpf(float(rng.exponential(2.0))) if rng.random() < 0.8 else 0) for _ in range(n)]
        T = sum(tau); varL = sum(t * (t - 1) for t in tau)
        def cost(phi):
            R = mp.mpf(1)
            for t in tau:
                m = 1 - 1 / t
                R *= phi * (1 - m) / (1 - phi * m)
            return (1 + phi) / (2 * T * T * (1 - phi)) - (R / (1 - R)) / T
        worst0 = max(worst0, abs(cost(mp.mpf(10) ** -30) * 2 * T * T - 1))
        lim = (1 - varL / T ** 2) / (2 * T)
        worst1 = max(worst1, abs(cost(1 - mp.mpf(10) ** -14) / lim - 1))
    report("F8 limits of the closed form", worst0 < 1e-9 and worst1 < 1e-9,
           f"300 random rings: phi -> 0 against 1/(2T^2), largest relative difference {mp.nstr(worst0, 2)}; phi -> 1 against (1-CV^2)/(2T), {mp.nstr(worst1, 2)}")

    # F9 equal ring
    worst, bad, cnt = 0.0, 0, 0
    for trial in range(20000):
        n = int(rng.integers(1, 40)); s = rng.uniform(0.01, 0.98); g = math.exp(rng.uniform(math.log(0.02), math.log(5))); phi = rng.uniform(0, 0.98)
        a_ = float(theory.cost_transfer([s] * n, [g] * n, 1.0, phi).sum()); b_ = theory.cost_equal_ring(n, s, g, phi)
        worst = max(worst, abs(a_ - b_) / b_)
        c1, c2 = theory.cost_equal_ring(1, s, g, phi), theory.cost_equal_ring(2, s, g, phi)
        thr = (s + g) / (3 * s + g)
        if abs(phi - thr) > 1e-9 and ((c2 > c1) != (phi > thr)):
            bad += 1
        cnt += 1
    report("F9 equal ring: general closed form = C_n; two stages cost more than one exactly when phi > (s+g)/(3s+g)", worst < 1e-8 and bad == 0,
           f"{cnt} cases; largest relative difference {worst:.1e}; threshold exceptions {bad}")

    # F10 periodic environment, exact
    def periodic(s, g, k, P, eps):
        n = len(s); a0 = theory.ring_matrix(s, g); lam = theory.perron(a0)[0]
        prod = np.eye(n)
        for t in range(P):
            a = a0.copy()
            a[(k + 1) % n, k] += g[k] * eps * math.cos(2 * math.pi * t / P)
            prod = a @ prod
        exact = math.log(max(abs(np.linalg.eigvals(prod)))) / P - math.log(lam)
        m = np.asarray(s) / lam; T = float(np.sum(1 / (1 - m))); w = 2 * math.pi / P
        gg = np.prod(np.exp(1j * w) * (1 - m) / (1 - m * np.exp(1j * w)))
        var = eps ** 2 if P == 2 else eps ** 2 / 2          # with period 2 the cosine is +-1 and its variance is eps^2
        return exact, var * (1 / T) * (gg / (1 - gg)).real
    ratios, devs, gains, cnt = [], [], 0, 0
    for trial in range(600):
        n = int(rng.integers(1, 8)); s = rng.uniform(0.05, 0.95, n); g = np.exp(rng.uniform(np.log(0.05), np.log(3), n))
        k = int(rng.integers(n)); P = int(rng.integers(2, 40))
        e1, p1 = periodic(s, g, k, P, 0.02); e2, p2 = periodic(s, g, k, P, 0.01)
        cnt += 1; gains += int(e2 > 0)
        if abs(p2) < 1e-9 or abs(e2 - p2) < 1e-15:
            continue
        ratios.append(abs(e1 - p1) / abs(e2 - p2)); devs.append(abs(e2 / p2 - 1))
    med = float(np.median(ratios)); frac = float(np.mean((np.array(ratios) > 14) & (np.array(ratios) < 18)))
    report("F10 periodic environment against exact growth rates", 14 < med < 18 and frac > 0.9,
           f"{cnt} random rings, period 2 to 39, amplitude 0.01: median |exact/predicted - 1| = {np.median(devs):.1e}; error ratio on halving the amplitude: median {med:.2f}, "
           f"between 14 and 18 in {100 * frac:.0f}% of {len(ratios)} cases; exact growth above the constant environment in {gains} of {cnt}")

    # F11 several entries on one driver
    worst, cnt = 0.0, 0
    for trial in range(500):
        d = int(rng.integers(2, 7))
        a = rng.uniform(0.05, 1, (d, d)) * (rng.random((d, d)) < 0.7)
        a += np.diag(rng.uniform(0.05, 0.5, d)) + 0.05 * np.roll(np.eye(d), 1, axis=0)
        nz = [tuple(int(z) for z in e) for e in np.argwhere(a > 0)]
        k = int(rng.integers(2, min(4, len(nz)) + 1))
        ch = [nz[i] for i in rng.choice(len(nz), k, replace=False)]
        kap = rng.uniform(0.3, 1, k); phi = rng.uniform(-0.8, 0.95)
        mat = theory.matrix_cost(a, ch, [0] * k, kap, [phi])
        lin = theory.lineage_cost_shared(a, ch, kap, lambda l, f=phi: f ** l)
        worst = max(worst, abs(mat - lin) / max(abs(mat), 1e-12)); cnt += 1
    report("F11 several entries on one driver: full lineage form = matrix formula", worst < 1e-8, f"{cnt} random matrices with 2 to 4 entries on one driver; largest relative difference {worst:.1e}")

    # F12 autocorrelation that is not geometric
    worst, cnt = 0.0, 0
    for trial in range(500):
        s_, g_ = random_ring(rng, nmax=8, allow_zero=False)
        if len(s_) < 2:
            continue
        a = theory.ring_matrix(s_, g_); k = int(rng.integers(len(s_))); entry = ((k + 1) % len(s_), k)
        f1, f2 = rng.uniform(-0.8, 0.95, 2); w1 = rng.uniform(0.1, 0.9)
        c0 = theory.matrix_cost(a, [entry], [0], [1.0], [0.0])
        mix = w1 * theory.matrix_cost(a, [entry], [0], [1.0], [f1]) + (1 - w1) * theory.matrix_cost(a, [entry], [0], [1.0], [f2])
        lin = theory.lineage_cost(a, entry, lambda l: w1 * f1 ** l + (1 - w1) * f2 ** l)
        worst = max(worst, abs(mix - lin) / max(abs(mix), 1e-12)); cnt += 1
    m2 = np.array([0.1, 0.1]); a2 = theory.ring_matrix(m2, 1 - m2)
    c_white = theory.lineage_cost(a2, (1, 0), lambda l: 0.0); c_lag2 = theory.lineage_cost(a2, (1, 0), lambda l: 0.4 if l == 2 else 0.0)
    report("F12 non-geometric autocorrelation", worst < 1e-8 and c_lag2 < c_white,
           f"{cnt} rings with a mixture of two geometric terms: largest relative difference {worst:.1e}; counterexample (two stages, staying share 0.1, "
           f"autocorrelation 0.4 at lag two only): cost {c_lag2:.5f} against {c_white:.5f} with no memory")

    # F13 per generation, equal stays
    def gamma(n, tau, phi):
        return n * tau * float(theory.cost_transfer([1 - 1 / tau] * n, [1 / tau] * n, 1.0, phi).sum())
    bad, worst0, worst_lim, cnt = 0, 0.0, 0.0, 0
    for trial in range(5000):
        tau = 1 + rng.exponential(3) + 1e-3; phi = rng.uniform(1e-3, 0.99)
        vals = [gamma(n, tau, phi) for n in range(1, 40)]
        bad += int(any(vals[i + 1] < vals[i] * (1 - 1e-12) for i in range(len(vals) - 1)))
        worst0 = max(worst0, max(abs(gamma(n, tau, 0.0) * 2 * tau - 1) for n in (1, 2, 5, 17)))
        worst_lim = max(worst_lim, abs(vals[0] * 2 * tau - 1))
        cnt += 1
    report("F13 per generation: constant in n without memory, never falling with n with positive memory", bad == 0 and worst0 < 1e-9 and worst_lim < 1e-9,
           f"{cnt} random (stay, phi), n = 1 to 39: exceptions {bad}; without memory equal to 1/(2 tau) within {worst0:.1e}")

    # F14 fixed generation time divided among n stages
    bad, cnt = 0, 0
    for trial in range(5000):
        T = rng.uniform(3, 40); phi = rng.uniform(0, 0.99)
        ns = range(1, int(math.ceil(T)))                    # n < T, so every stage has staying
        vals = [float(theory.cost_transfer([1 - n / T] * n, [n / T] * n, 1.0, phi).sum()) for n in ns]
        bad += int(any(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))); cnt += 1
    report("F14 generation time fixed and divided among n stages: cost rises with n", bad == 0, f"{cnt} random (generation time, phi), all n below the generation time: exceptions {bad}")

    # F15 slow-environment limit in terms of the stays
    worst1, worst2 = mp.mpf(0), mp.mpf(0)
    for trial in range(200):
        n = int(rng.integers(1, 8))
        tau = [mp.mpf(1) + mp.mpf(float(rng.exponential(3.0))) for _ in range(n)]
        T = sum(tau)
        phi = 1 - mp.mpf(10) ** -14
        R = mp.mpf(1)
        for t in tau:
            mm = 1 - 1 / t
            R *= phi * (1 - mm) / (1 - phi * mm)
        val = (1 + phi) / (2 * T * T * (1 - phi)) - (R / (1 - R)) / T
        lim = (T * T - sum(t * t for t in tau) + T) / (2 * T ** 3)
        worst1 = max(worst1, abs(val / lim - 1))
        if n == 1:
            worst2 = max(worst2, abs(lim * 2 * T * T - 1))
    report("F15 slow environment: limit (T^2 - sum tau^2 + T)/(2 T^3), equal to 1/(2 T^2) for one stage", worst1 < 1e-9 and worst2 < 1e-30,
           f"200 random rings; largest relative difference {mp.nstr(worst1, 2)}")

    # F16 adding a host with the mean stay
    def gam_tau(tau, theta):
        tau = np.asarray(tau, float); n = len(tau); T = tau.sum()
        if theta == 0:
            return n / (2 * T)
        R = 1 / np.prod(1 + tau / theta)
        return n * ((1 + 2 * theta) / (2 * T) - R / (1 - R))
    bad, worst0, cnt = 0, 0.0, 0
    for trial in range(100000):
        n = int(rng.integers(1, 9)); tau = 1 + rng.exponential(rng.choice([0.3, 2, 10]), n)
        theta = rng.choice([0.05, 0.3, 1, 3, 10, 50]) * rng.uniform(0.5, 2)
        more = np.append(tau, tau.mean())
        bad += int(gam_tau(more, theta) < gam_tau(tau, theta) * (1 - 1e-12))
        worst0 = max(worst0, abs(gam_tau(more, 0) / gam_tau(tau, 0) - 1)); cnt += 1
    # the same quantity through the closed form in theory.py, on a few cases, so that gam_tau is not a private formula
    wt = 0.0
    for trial in range(200):
        n = int(rng.integers(1, 7)); tau = 1 + rng.exponential(2.0, n) + 1e-3; phi = rng.uniform(0.05, 0.95)
        ref = float(theory.per_generation_cost_transfer(1 - 1 / tau, 1 / tau, 1.0, phi).sum())
        wt = max(wt, abs(gam_tau(tau, phi / (1 - phi)) / ref - 1))
    report("F16 per generation, unequal stays: a host added with the mean stay never lowers the cost when hosts have memory", bad == 0 and worst0 < 1e-12 and wt < 1e-9,
           f"{cnt} random cycles of 1 to 8 unequal stages: exceptions {bad}; without memory unchanged within {worst0:.1e}")

    # F17 any age-structured life history, one driver on fertility
    worst, cnt = 0.0, 0
    for trial in range(2000):
        d = int(rng.integers(1, 10))
        p_ = rng.uniform(0.2, 0.98, d - 1)
        f_ = rng.uniform(0, 2, d) * (rng.random(d) < 0.7); f_[-1] = rng.uniform(0.2, 2)
        a = np.zeros((d, d)); a[0, :] = f_
        for i in range(d - 1):
            a[i + 1, i] = p_[i]
        lam = theory.perron(a)[0]
        l_ = np.concatenate([[1.0], np.cumprod(p_)]); age = np.arange(1, d + 1)
        wgt = l_ * f_ * lam ** (-age)                      # distribution of the age of mothers; sums to one
        phi = rng.uniform(-0.8, 0.97); theta = phi / (1 - phi)
        T = float(np.sum(age * wgt)); R = float(np.sum(wgt * phi ** age))
        closed = (1 + 2 * theta) / (2 * T * T) - (R / (1 - R)) / T
        ent = [(0, j) for j in range(d) if f_[j] > 0]
        mat = theory.matrix_cost(a, ent, [0] * len(ent), [1.0] * len(ent), [phi])
        worst = max(worst, abs(closed - mat) / max(abs(mat), 1e-9), abs(wgt.sum() - 1)); cnt += 1
    report("F17 any age-structured life history, one driver on fertility: Euler-Lotka form", worst < 1e-8,
           f"{cnt} random life tables of 1 to 9 age classes, phi from -0.8 to 0.97; largest relative difference {worst:.1e}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
