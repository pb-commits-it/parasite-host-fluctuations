"""Verification run R0b: does `engine/environments.py` reproduce results known without it?

Claim tested: `FluctuatingRing` implements G_k(t) = g_k h_k(t), S_k(t) = s_k u_k(t) with independent
lognormal AR(1) multipliers of mean one, each with its own noise level and autocorrelation.
Checks S1 to S6 and their tolerances are fixed in `protocols/R0b-survival-noise.md`, written before
this script was first run. Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import environments, ring  # noqa: E402

S, G = 2.0 / 3.0, 0.4
REPS, STEPS, BURN = 400, 120000, 2000
results = []


def record(check, case, passed, detail):
    results.append((check, case, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {check} {case}: {detail}", flush=True)


def exact_rate_1d(const, coef, sigma, nodes=200):
    """E ln(const + coef * exp(sigma Z - sigma^2/2))."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return float(np.sum(w * np.log(const + coef * np.exp(sigma * x - 0.5 * sigma ** 2))) / math.sqrt(2 * math.pi))


def exact_rate_2d(s, g, sig_s, sig_g, nodes=80):
    """E ln(s u + g h), u and h independent mean-one lognormals."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    u = s * np.exp(sig_s * x - 0.5 * sig_s ** 2)
    h = g * np.exp(sig_g * x - 0.5 * sig_g ** 2)
    return float(np.sum(w[:, None] * w[None, :] * np.log(u[:, None] + h[None, :])) / (2 * math.pi))


class UncorrectedSurvival(environments.FluctuatingRing):
    """Mutation m4: the survival multiplier has no mean correction, so E u = exp(sigma^2/2)."""

    def step(self):
        s_t, g_t, cv = super().step()
        return s_t * np.exp(0.5 * self.u.sigma[None, :] ** 2), g_t, cv


class ShiftedSurvival(environments.FluctuatingRing):
    """Mutation m3: the survival multiplier of stage k is applied to stage k+1."""

    def step(self):
        h = self.h.next()
        u = self.u.next()
        cv = (h - 1.0) @ self.e_g + (u - 1.0) @ self.e_s
        return self.s[None, :] * np.roll(u, 1, axis=1), self.g[None, :] * h, cv


def s1(env_cls=environments.FluctuatingRing, label="S1", cases=None, expect_fail=False):
    print(f"{label}  one stage, survival fluctuating: exact integral")
    cases = cases or [(sg, ph) for sg in (0.05, 0.15, 0.4, 0.8) for ph in (0.0, 0.9)]
    worst = 0.0
    for i, (sg, ph) in enumerate(cases):
        rng = np.random.default_rng(81000 + i)
        env = env_cls([S], [G], [0.0], [0.0], [sg], [ph], REPS, rng)
        out = ring.run(env, 1, REPS, STEPS, BURN)
        exact = exact_rate_1d(G, S, sg)
        cost_exact = math.log(S + G) - exact
        m_cv, se_cv = ring.mean_se(out["rate_cv"])
        m_raw, se_raw = ring.mean_se(out["rate"])
        z_cv, z_raw = (m_cv - exact) / se_cv, (m_raw - exact) / se_raw
        ok = abs(z_cv) < 3.5 and abs(z_raw) < 3.5 and se_cv <= 0.01 * cost_exact
        worst = max(worst, abs(z_cv))
        if expect_fail:
            record(label, f"sigma={sg} phi={ph}", not ok, f"broken copy rejected: z = {z_cv:+.1f}" if not ok else "broken copy NOT rejected")
        else:
            record(label, f"sigma={sg} phi={ph}", ok, f"z = {z_cv:+.2f} (no control variate {z_raw:+.2f}); s.e. is {100 * se_cv / cost_exact:.3f}% of the cost")
    return worst


def s2():
    print("S2  one stage, survival and transfer both fluctuating: two-dimensional integral")
    for i, (ps, pg) in enumerate(((0.0, 0.9), (0.9, 0.5))):
        rng = np.random.default_rng(82000 + i)
        env = environments.FluctuatingRing([S], [G], [0.2], [pg], [0.3], [ps], REPS, rng)
        out = ring.run(env, 1, REPS, STEPS, BURN)
        exact = exact_rate_2d(S, G, 0.3, 0.2)
        m, se = ring.mean_se(out["rate_cv"])
        z = (m - exact) / se
        record("S2", f"phi_s={ps} phi_g={pg}", abs(z) < 3.5, f"z = {z:+.2f}; cost {math.log(S + G) - m:.6f} against exact {math.log(S + G) - exact:.6f}")


def s3():
    print("S3  one stage, no transfer: exact mean and variance of total log growth")
    s, sg, reps, steps = 0.9, 0.3, 4000, 20000
    for i, ph in enumerate((0.0, 0.5, 0.9)):
        rng = np.random.default_rng(83000 + i)
        env = environments.FluctuatingRing([s], [0.0], [0.0], [0.0], [sg], [ph], reps, rng)
        env.lam0, env.e_g, env.e_s = s, np.zeros(1), np.ones(1)      # elasticities are undefined when g = 0
        out = ring.run(env, 1, reps, steps, 0)
        mean_exact = math.log(s) - 0.5 * sg ** 2
        m, se = ring.mean_se(out["rate"])
        z = (m - mean_exact) / se
        j = np.arange(1, steps)
        var_exact = sg ** 2 * (steps + 2 * np.sum((steps - j) * ph ** j))
        var = float(np.var(out["total"], ddof=1))
        record("S3", f"phi={ph}", abs(z) < 3.5 and abs(var / var_exact - 1) < 0.08, f"mean z = {z:+.2f}; variance / exact = {var / var_exact:.3f}")


def s4():
    print("S4  no staying, stage-specific transfer noise, one cohort: exact mean")
    g = np.array([1.3, 0.8, 1.1]); sg = np.array([0.1, 0.3, 0.5]); ph = np.array([0.0, 0.5, 0.9])
    reps, steps = 4000, 20001
    rng = np.random.default_rng(84000)
    env = environments.FluctuatingRing([0.0, 0.0, 0.0], g, sg, ph, [0.0] * 3, [0.0] * 3, reps, rng)
    out = ring.run(env, 3, reps, steps, 0, x0=[1.0, 0.0, 0.0])
    exact = float(np.mean(np.log(g) - 0.5 * sg ** 2))
    m, se = ring.mean_se(out["rate"])
    z = (m - exact) / se
    record("S4", "n=3", abs(z) < 3.5, f"mean {m:.6f} against exact {exact:.6f}; z = {z:+.2f}")


def s5_case(name, s, g, sig_g, sig_s, seed, env_cls=environments.FluctuatingRing):
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    sig_g = np.broadcast_to(np.asarray(sig_g, float), (n,)); sig_s = np.broadcast_to(np.asarray(sig_s, float), (n,))
    rng = np.random.default_rng(seed)
    env = env_cls(s, g, sig_g, np.zeros(n), sig_s, np.zeros(n), REPS, rng)
    out = ring.run(env, n, REPS, STEPS, BURN)
    lam0, e_g, e_s = ring.elasticities(s, g)
    approx = 0.5 * float(np.sum(e_g ** 2 * np.expm1(sig_g ** 2)) + np.sum(e_s ** 2 * np.expm1(sig_s ** 2)))
    m, se = ring.mean_se(out["rate_cv"])
    cost = math.log(lam0) - m
    return cost, se, approx


def s5():
    print("S5  several stages, uncorrelated years, small noise: published approximation")
    cases = [
        ("a equal n=2, survival noise in both", [S, S], [G, G], 0.0, 0.05),
        ("b equal n=4, survival noise in all", [S] * 4, [G] * 4, 0.0, 0.05),
        ("c unequal n=3, survival noise in stage 2 only", [0.3, 0.85, 0.5], [0.6, 0.2, 0.9], 0.0, [0.0, 0.05, 0.0]),
        ("d unequal n=5, survival and transfer noise everywhere", [0.75, 0.1, 0.5, 0.85, 0.35], [0.15, 0.8, 0.4, 0.1, 0.55], 0.05, 0.05),
    ]
    for i, (name, s, g, sg, ss) in enumerate(cases):
        cost, se, approx = s5_case(name, s, g, sg, ss, 85000 + i)
        ok = abs(cost - approx) < max(3.5 * se, 0.02 * approx)
        record("S5", name, ok, f"simulated / approximation = {cost / approx:.4f} (s.e. {se / approx:.4f})")


def s6():
    print("S6  the checks can fail")
    cost, se, approx = s5_case("c shifted", [0.3, 0.85, 0.5], [0.6, 0.2, 0.9], 0.0, [0.0, 0.05, 0.0], 86000, env_cls=ShiftedSurvival)
    rejected = not (abs(cost - approx) < max(3.5 * se, 0.02 * approx))
    record("S6", "m3 survival noise on the wrong stage", rejected, f"simulated / approximation = {cost / approx:.4f}; " + ("rejected" if rejected else "NOT rejected"))
    s1(env_cls=UncorrectedSurvival, label="S6 m4", cases=[(0.4, 0.0)], expect_fail=True)


def main():
    t0 = time.time()
    s1(); s2(); s3(); s4(); s5(); s6()
    n_fail = sum(1 for r in results if not r[2])
    print(f"\n{len(results)} checks recorded, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R0b PASSED" if n_fail == 0 else "R0b FAILED")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
