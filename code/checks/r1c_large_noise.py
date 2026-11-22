"""Run R1c: large fluctuations and the risk of extinction, for the ring of equal stages.

Claim tested: hypotheses L2, L3 and L5 of `protocols/R1c-large-noise-and-extinction.md` (checks), and the open
questions L1, L4, L6, L7, L8, with the grid, settings and criteria fixed there before this script was first run.
Exit status is 0 only if every check passes.

The simulation loop here repeats the update of `engine/ring.run` so that batch totals and running minima can be
recorded. It is tested inside this script in two ways: the one-stage runs must match the exact integral (L3), and
one case is compared with `engine/ring.run` on the same random numbers.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy.stats import norm

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import environments, ring, theory  # noqa: E402

spec = importlib.util.spec_from_file_location("h4", HERE.parent / "studies" / "h4" / "h4.py")
h4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h4)

REPS, STEPS, BURN, BATCH = 400, 60000, 3000, 6000
checks = []


def record(hyp, name, ok, txt):
    checks.append((hyp, name, bool(ok), txt))
    print(f"  [{'PASS' if ok else 'FAIL'}] {hyp} {name}: {txt}", flush=True)


def exact_one_stage_cost(s, g, sigma, nodes=200):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    rate = float(np.sum(w * np.log(s + g * np.exp(sigma * x - 0.5 * sigma ** 2))) / math.sqrt(2 * math.pi))
    return math.log(s + g) - rate


def simulate(n, s, g, sigma, phi, seed, reps=REPS, steps=STEPS, burn=BURN, batch=BATCH):
    """Cost (with control variate), its s.e., long-run variance per step from batch totals, and its s.e."""
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing([s] * n, [g] * n, [sigma] * n, [phi] * n, [0.0] * n, [0.0] * n, reps, rng)
    x = np.full((reps, n), 1.0 / n)
    total = np.zeros(reps); cv_sum = np.zeros(reps)
    nb = steps // batch
    btot = np.zeros((nb, reps)); acc = np.zeros(reps)
    for t in range(burn + steps):
        s_t, g_t, cv = env.step()
        x = x * s_t + np.roll(x * g_t, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= burn:
            lg = np.log(tot)
            total += lg; cv_sum += cv; acc += lg
            k = t - burn
            if (k + 1) % batch == 0:
                btot[(k + 1) // batch - 1] = acc; acc = np.zeros(reps)
    rate_cv = (total - cv_sum) / steps
    cost = math.log(env.lam0) - float(rate_cv.mean()); se = float(rate_cv.std(ddof=1) / math.sqrt(reps))
    dev2 = (btot - btot.mean()) ** 2 / batch
    var = float(dev2.mean()); var_se = float(dev2.std(ddof=1) / math.sqrt(dev2.size))
    return cost, se, var, var_se, float(total.mean() / steps)


def extinction(n, s, g, sigma, phi, seed, reps=4000, steps=2000, burn=1000, depth=math.log(100.0)):
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing([s] * n, [g] * n, [sigma] * n, [phi] * n, [0.0] * n, [0.0] * n, reps, rng)
    x = np.full((reps, n), 1.0 / n)
    cum = np.zeros(reps); low = np.zeros(reps)
    for t in range(burn + steps):
        s_t, g_t, cv = env.step()
        x = x * s_t + np.roll(x * g_t, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= burn:
            cum += np.log(tot); low = np.minimum(low, cum)
    ext = low <= -depth
    p = float(ext.mean()); p_se = math.sqrt(max(p * (1 - p), 1e-12) / reps)
    mu = float(cum.mean() / steps); v = float(cum.var(ddof=1) / steps)
    return p, p_se, mu, v


def lande_orzack(mu, v, x0, t):
    """Probability of having fallen by x0 in log size by time t (Lande & Orzack 1988, eq. 11)."""
    if v <= 0:
        return 0.0
    sd = math.sqrt(v * t)
    return float(norm.cdf((-x0 - mu * t) / sd) + math.exp(-2 * mu * x0 / v) * (1 - norm.cdf((x0 - mu * t) / sd)))


def main():
    t0 = time.time()
    out = {"A": [], "C": []}
    s, g = 2.0 / 3.0, 0.4
    lam0 = s + g; q = g / lam0; tau = 1 / q

    print("Self-test: this script's loop against engine/ring.run on the same random numbers")
    env = environments.FluctuatingRing([s] * 4, [g] * 4, [0.2] * 4, [0.9] * 4, [0.0] * 4, [0.0] * 4, 50, np.random.default_rng(5))
    ref = ring.run(env, 4, 50, 3000 + 6000, 3000)
    mine = simulate(4, s, g, 0.2, 0.9, 5, reps=50, steps=6000, burn=3000, batch=6000)
    d = abs((math.log(lam0) - float(ref["rate_cv"].mean())) - mine[0])
    record("self-test", "loop equals engine", d < 1e-12, f"difference in cost {d:.1e}")

    print("Parts A and B: cost and variance over the grid")
    for phi in (0.0, 0.5, 0.9):
        for n in (1, 2, 4, 8, 16):
            c2 = theory.cost_equal_ring(n, s, g, phi)
            hh = float(h4.fourth_order_coefficient(n, s, g, phi))
            vth = n * (1 + phi) / (1 - phi) / (n * tau) ** 2
            for i, sigma in enumerate((0.1, 0.2, 0.4, 0.8)):
                seed = 5000 + 1000 * int(round(phi * 10)) + 10 * n + i
                cost, se, var, var_se, rate = simulate(n, s, g, sigma, phi, seed)
                p2 = c2 * sigma ** 2; p4 = p2 + hh * sigma ** 4
                row = dict(n=n, phi=phi, sigma=sigma, cost=cost, se=se, second=p2, fourth=p4, var=var, var_se=var_se, var_theory=vth * sigma ** 2)
                if n == 1:
                    row["exact"] = exact_one_stage_cost(s, g, sigma)
                out["A"].append(row)
                print(f"    phi={phi} n={n:2d} sigma={sigma}: cost/2nd = {cost / p2:.4f} +- {se / p2:.4f}; cost/(2nd+4th) = {cost / p4:.4f}; "
                      f"variance/leading = {var / (vth * sigma ** 2):.4f} +- {var_se / (vth * sigma ** 2):.4f}", flush=True)

    print("L2  the fourth-order term improves the prediction where it is a small correction")
    bad = 0; cnt = 0
    for r in out["A"]:
        if abs(r["fourth"] - r["second"]) < 0.2 * abs(r["second"]):
            cnt += 1
            d2, d4 = abs(r["cost"] - r["second"]), abs(r["cost"] - r["fourth"])
            if not (d4 < d2 or (d2 < 3.5 * r["se"] and d4 < 3.5 * r["se"])):
                bad += 1
                print(f"      not improved: n={r['n']} phi={r['phi']} sigma={r['sigma']}: |sim - 2nd| = {d2:.3e}, |sim - (2nd+4th)| = {d4:.3e}, s.e. {r['se']:.1e}")
    record("L2", "fourth-order term", bad == 0, f"{cnt} runs in which the correction is below 20%; not improved in {bad}")
    print("L3  one stage against the exact integral")
    zs = [(r["cost"] - r["exact"]) / r["se"] for r in out["A"] if r["n"] == 1]
    record("L3", "one stage", max(abs(z) for z in zs) < 3.5, f"{len(zs)} runs; largest |z| = {max(abs(z) for z in zs):.2f}")
    print("L5  variance of growth at sigma = 0.1 against the leading-order formula")
    worst = 0.0; badv = 0; cntv = 0
    for r in out["A"]:
        if r["sigma"] == 0.1:
            cntv += 1
            dev = abs(r["var"] - r["var_theory"])
            worst = max(worst, dev / r["var_theory"])
            if dev > max(0.05 * r["var_theory"], 3.5 * r["var_se"]):
                badv += 1
    record("L5", "variance", badv == 0, f"{cntv} runs; largest relative departure {100 * worst:.1f}%; outside tolerance: {badv}")

    print("Part C: extinction within 2,000 steps, start 100 times the threshold, lambda0 = 1.01")
    s2, g2 = 0.63125, 0.37875
    q2 = g2 / (s2 + g2); tau2 = 1 / q2
    for phi in (0.0, 0.9):
        for n in (1, 2, 4, 8):
            for i, sigma in enumerate((0.2, 0.5)):
                seed = 9000 + 1000 * int(round(phi * 10)) + 10 * n + i
                p, p_se, mu, v = extinction(n, s2, g2, sigma, phi, seed)
                x0 = math.log(100.0)
                lo_meas = lande_orzack(mu, v, x0, 2000)
                mu_th = math.log(1.01) - theory.cost_equal_ring(n, s2, g2, phi) * sigma ** 2
                v_th = sigma ** 2 * n * (1 + phi) / (1 - phi) / (n * tau2) ** 2
                lo_th = lande_orzack(mu_th, v_th, x0, 2000)
                out["C"].append(dict(n=n, phi=phi, sigma=sigma, p=p, p_se=p_se, mu=mu, v=v, lo_measured=lo_meas, mu_theory=mu_th, v_theory=v_th, lo_theory=lo_th))
                print(f"    phi={phi} n={n} sigma={sigma}: extinct {p:.4f} +- {p_se:.4f}; diffusion with measured mean and variance {lo_meas:.4f}; "
                      f"with theory's {lo_th:.4f}  (mean {mu:+.5f} vs {mu_th:+.5f}; variance {v:.5f} vs {v_th:.5f})", flush=True)

    n_fail = sum(1 for c in checks if not c[2])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R1c PASSED" if n_fail == 0 else "R1c FAILED")
    (HERE.parent / "results" / "R1c-large-noise_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
