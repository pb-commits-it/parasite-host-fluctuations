"""Verification run R0: does `engine/ring.py` reproduce results known without it?

Claim tested: the simulator implements
    x_{k+1}(t+1) = S_{k+1}(t) x_{k+1}(t) + G_k(t) x_k(t)
with independent lognormal AR(1) hosts of mean one, and its growth-rate estimate is unbiased.
Checks V1 to V6 and V8 and their tolerances are fixed in `protocols/R0-verification.md`,
written before this script was first run. Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import ring  # noqa: E402

S, GBAR = 2.0 / 3.0, 0.4
REPS, STEPS, BURN = 400, 120000, 2000
results = []          # (check, case, passed, detail)


def record(check, case, passed, detail):
    results.append((check, case, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {check} {case}: {detail}", flush=True)


def exact_one_stage_rate(s, g, sigma, nodes=200):
    """E ln(s + g exp(sigma Z - sigma^2/2)), Z standard normal, by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return float(np.sum(w * np.log(s + g * np.exp(sigma * x - 0.5 * sigma ** 2))) / math.sqrt(2 * math.pi))


class Constant:
    def __init__(self, s, g):
        self.s, self.g = np.asarray(s, float)[None, :], np.asarray(g, float)[None, :]

    def step(self):
        return self.s, self.g, None


class Periodic:
    def __init__(self, s_seq, g_seq):
        self.s_seq, self.g_seq, self.t = s_seq, g_seq, 0

    def step(self):
        i = self.t % len(self.s_seq)
        self.t += 1
        return self.s_seq[i][None, :], self.g_seq[i][None, :], None


class UncorrectedHosts(ring.LognormalAR1):
    """Mutation m1: the mean correction is left out, so E h = exp(sigma^2/2), not one."""

    def next(self):
        self.z = self.phi * self.z + self.innov * self.rng.normal(0.0, 1.0, self.shape)
        return np.exp(self.z)


class SharedHost:
    """Mutation m2: every stage follows the same host series."""

    def __init__(self, n, sigma, phi, reps, rng):
        self.one = ring.LognormalAR1(1, sigma, phi, reps, rng)
        self.n = n

    def next(self):
        return np.repeat(self.one.next(), self.n, axis=1)


def v1():
    print("V1  constant rates: growth per step equals ln of the dominant eigenvalue")
    rng = np.random.default_rng(101)
    cases = [(f"equal n={n}", np.full(n, S), np.full(n, GBAR)) for n in (1, 2, 4, 8, 16, 32)]
    for i in range(20):
        n = int(rng.integers(2, 9))
        cases.append((f"random#{i + 1} n={n}", rng.uniform(0.1, 0.95, n), rng.uniform(0.05, 1.0, n)))
    worst = 0.0
    for name, s, g in cases:
        lam0 = max(np.linalg.eigvals(ring.mean_matrix(s, g)).real)
        out = ring.run(Constant(s, g), len(s), 1, 20000)
        err = abs(out["last"][0] - math.log(lam0))
        worst = max(worst, err)
        if err > 1e-9:
            record("V1", name, False, f"error {err:.2e}")
    record("V1", f"{len(cases)} cases", worst <= 1e-9, f"largest error {worst:.2e} (tolerance 1e-9)")


def v2(hosts_cls=ring.LognormalAR1, label="V2", cases=None, expect_fail=False):
    if not expect_fail:
        print("V2  one stage against the exact integral")
    lam0 = S + GBAR
    ok_all = True
    for sigma, phi in cases or [(sg, ph) for sg in (0.05, 0.15, 0.4, 0.8) for ph in (0.0, 0.9)]:
        exact = exact_one_stage_rate(S, GBAR, sigma)
        rng = np.random.default_rng(int(2000 + 1000 * sigma + 10 * phi))
        env = ring.FluctuatingProduction([S], [GBAR], sigma, phi, REPS, rng,
                                         hosts=hosts_cls(1, sigma, phi, REPS, rng))
        out = ring.run(env, 1, REPS, STEPS, BURN)
        m, se = ring.mean_se(out["rate_cv"])
        m_raw, se_raw = ring.mean_se(out["rate"])
        cost = math.log(lam0) - exact
        z, z_raw, rel = (m - exact) / se, (m_raw - exact) / se_raw, se / cost
        ok = abs(z) <= 3.5 and rel <= 0.01 and abs(z_raw) <= 3.5
        ok_all &= ok
        detail = (f"exact cost {cost:.6e}, simulated {math.log(lam0) - m:.6e}, z = {z:+.2f}, "
                  f"s.e./cost = {100 * rel:.3f}%, z without control variate = {z_raw:+.2f}")
        if expect_fail:
            record(label, f"sigma={sigma} phi={phi}", not ok, "broken copy " + ("rejected: " if not ok else "NOT rejected: ") + detail)
        else:
            record(label, f"sigma={sigma} phi={phi}", ok, detail)
    return ok_all


def v3():
    print("V3  one stage, small fluctuations: exact integral against the second-order approximation")
    e = GBAR / (S + GBAR)
    gaps = {}
    for sigma in (0.2, 0.1, 0.05):
        cost = math.log(S + GBAR) - exact_one_stage_rate(S, GBAR, sigma)
        gaps[sigma] = cost / (0.5 * e ** 2 * math.expm1(sigma ** 2)) - 1.0
    r1, r2 = gaps[0.2] / gaps[0.1], gaps[0.1] / gaps[0.05]
    coef, pred = gaps[0.05] / 0.05 ** 2, -2 * e * (1 - 0.75 * e)
    record("V3", "scaling", 3.5 <= r1 <= 4.5 and 3.5 <= r2 <= 4.5,
           f"gap {gaps[0.2]:+.5f}, {gaps[0.1]:+.5f}, {gaps[0.05]:+.6f} at sigma 0.2, 0.1, 0.05; ratios {r1:.3f}, {r2:.3f}")
    record("V3", "coefficient", abs(coef / pred - 1) <= 0.15, f"gap/sigma^2 = {coef:+.4f}, predicted {pred:+.4f}")
    from scipy import integrate
    worst = 0.0
    for sigma in (0.05, 0.15, 0.4, 0.8):
        f = lambda z: math.log(S + GBAR * math.exp(sigma * z - 0.5 * sigma ** 2)) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        q, _ = integrate.quad(f, -12, 12, epsabs=1e-14, epsrel=1e-13, limit=400)
        worst = max(worst, abs(q - exact_one_stage_rate(S, GBAR, sigma)))
    record("V3", "quadrature cross-check", worst <= 1e-10, f"Gauss-Hermite against scipy quad, largest difference {worst:.1e}")


def v4(hosts_cls=None, label="V4", cases=None, expect_fail=False):
    if not expect_fail:
        print("V4  no staying, one cohort: exact mean and variance of log growth")
    gbar, sigma, reps, steps = 1.1, 0.3, 4000, 20000
    for n, phi in cases or [(n, ph) for n in (1, 2, 4, 8) for ph in (0.0, 0.9)]:
        rng = np.random.default_rng(4000 + 10 * n + int(10 * phi))
        hosts = hosts_cls(n, sigma, phi, reps, rng) if hosts_cls else None
        env = ring.FluctuatingProduction(np.zeros(n), np.full(n, gbar), sigma, phi, reps, rng, hosts=hosts)
        x0 = np.zeros(n)
        x0[0] = 1.0
        out = ring.run(env, n, reps, steps, 0, x0=x0)
        m, se = ring.mean_se(out["rate"])
        exact = math.log(gbar) - 0.5 * sigma ** 2
        z = (m - exact) / se
        var_pred = sigma ** 2 * (steps + 2 * sum((steps - j * n) * phi ** (j * n) for j in range(1, (steps - 1) // n + 1)))
        var_obs = float(out["total"].var(ddof=1))
        ok_mean, ok_var = abs(z) <= 3.5, abs(var_obs / var_pred - 1) <= 0.08
        detail = f"mean z = {z:+.2f}; variance observed/predicted = {var_obs / var_pred:.3f}"
        if expect_fail:
            record(label, f"n={n} phi={phi}", not ok_var, "broken copy " + ("rejected: " if not ok_var else "NOT rejected: ") + detail)
        else:
            record(label, f"n={n} phi={phi}", ok_mean and ok_var, detail)


def v5():
    print("V5  periodic rates against the eigenvalue of the product over one period")
    rng = np.random.default_rng(505)
    worst = 0.0
    for i in range(10):
        n, p = int(rng.integers(2, 7)), int(rng.choice([2, 3, 5]))
        s_seq = rng.uniform(0.1, 0.95, (p, n))
        g_seq = rng.uniform(0.05, 1.0, n)[None, :] * rng.uniform(0.3, 1.7, (p, n))
        prod = np.eye(n)
        for t in range(p):
            a = np.zeros((n, n))
            for k in range(n):
                a[k, k] += s_seq[t, k]
                a[(k + 1) % n, k] += g_seq[t, k]
            prod = a @ prod
        exact = math.log(max(abs(np.linalg.eigvals(prod)))) / p
        out = ring.run(Periodic(s_seq, g_seq), n, 1, 20000 + 10 * p, 20000)
        worst = max(worst, abs(out["rate"][0] - exact))
    record("V5", "10 random rings", worst <= 1e-9, f"largest error {worst:.2e} (tolerance 1e-9)")


def v6():
    print("V6  several stages, uncorrelated years, against the published small-noise approximation")
    sigma = 0.05
    cases = [(f"equal n={n}", np.full(n, S), np.full(n, GBAR)) for n in (2, 4, 8)]
    cases.append(("unequal n=3", np.array([0.5, 0.8, 0.3]), np.array([0.6, 0.3, 0.9])))
    cases.append(("unequal n=5", np.array([0.9, 0.4, 0.6, 0.2, 0.7]), np.array([0.2, 0.7, 0.5, 0.9, 0.4])))
    for i, (name, s, g) in enumerate(cases):
        n = len(s)
        lam0, e_g, _ = ring.elasticities(s, g)
        approx = 0.5 * float(np.sum(e_g ** 2)) * math.expm1(sigma ** 2)
        rng = np.random.default_rng(6000 + i)
        out = ring.run(ring.FluctuatingProduction(s, g, sigma, 0.0, REPS, rng), n, REPS, STEPS, BURN)
        m, se = ring.mean_se(out["rate_cv"])
        cost = math.log(lam0) - m
        ok = abs(cost - approx) <= max(3.5 * se, 0.02 * approx)
        record("V6", name, ok, f"approximation {approx:.4e}, simulated {cost:.4e} (s.e. {se:.1e}), ratio {cost / approx:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    print("R0 verification of engine/ring.py; tolerances from protocols/R0-verification.md")
    v1()
    v2()
    v3()
    v4()
    v5()
    v6()
    print("V8  the checks reject two deliberately broken copies")
    v2(hosts_cls=UncorrectedHosts, label="V8 m1", cases=[(0.4, 0.0)], expect_fail=True)
    v4(hosts_cls=SharedHost, label="V8 m2", cases=[(4, 0.9)], expect_fail=True)
    print("V7  Jonsson (2021) two-stage result: not run; the full text has not been read")
    n_fail = sum(1 for r in results if not r[2])
    print(f"\n{len(results) - n_fail} of {len(results)} recorded checks passed; {n_fail} failed; "
          f"{time.time() - t0:.0f} s")
    sys.exit(1 if n_fail else 0)
