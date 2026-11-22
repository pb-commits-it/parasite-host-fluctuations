"""Run R3: several alternative hosts at one stage.

Claim tested: checks K0 to K4 of `protocols/R3-alternative-hosts.md`, fixed there before this script was first run.
Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import environments_r3, ring, theory  # noqa: E402

REPS, STEPS, BURN = 400, 120000, 2000
A1 = ([0.3, 0.85, 0.5], [0.6, 0.2, 0.9])
checks = []


def record(name, ok, txt):
    checks.append((name, bool(ok), txt))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {txt}", flush=True)


def F(s, g, phi):
    """Cost per unit variance of one host at any stage (the same for every stage)."""
    return float(theory.cost_transfer(s, g, 1.0, phi)[0])


def run_case(name, shares, sigma_rel, phi, rho=0.0, phi_c=0.0, sigma=0.05, seed=0):
    s, g = A1
    n = len(s)
    pi = np.asarray(shares, float); sr = np.broadcast_to(np.asarray(sigma_rel, float), pi.shape); ph = np.broadcast_to(np.asarray(phi, float), pi.shape)
    per_stage = sum(pi[m] ** 2 * sr[m] ** 2 * (1 - rho) * F(s, g, ph[m]) for m in range(len(pi))) + float(pi @ sr) ** 2 * rho * F(s, g, phi_c)
    pred = n * per_stage * sigma ** 2
    env = environments_r3.AlternativeHosts(s, g, pi, sigma * sr, ph, REPS, np.random.default_rng(seed), rho=rho, phi_c=phi_c)
    out = ring.run(env, n, REPS, BURN + STEPS, BURN)
    m_, se = ring.mean_se(out["rate_cv"])
    cost = math.log(env.lam0) - m_
    ok = abs(cost - pred) <= max(3.5 * se, 0.015 * abs(pred))
    record(name, ok, f"formula {pred / sigma ** 2:.6f}; simulated {cost / sigma ** 2:.6f} +- {se / sigma ** 2:.6f} per sigma^2; ratio {cost / pred:.4f}")
    return cost, se, pred


def main():
    t0 = time.time()
    print("K0  known answer: one stage, two alternative hosts, two-dimensional integral")
    s, g, pi, sig = 2 / 3, 0.4, np.array([0.7, 0.3]), 0.4
    x, w = np.polynomial.hermite_e.hermegauss(80)
    h = np.exp(sig * x - 0.5 * sig ** 2)
    exact = float(np.sum(w[:, None] * w[None, :] * np.log(s + g * (pi[0] * h[:, None] + pi[1] * h[None, :]))) / (2 * math.pi))
    env = environments_r3.AlternativeHosts([s], [g], pi, sig, [0.0, 0.9], REPS, np.random.default_rng(70))
    out = ring.run(env, 1, REPS, BURN + STEPS, BURN)
    m_, se = ring.mean_se(out["rate_cv"]); m_raw, se_raw = ring.mean_se(out["rate"])
    record("K0", abs(m_ - exact) < 3.5 * se and abs(m_raw - exact) < 3.5 * se_raw,
           f"z = {(m_ - exact) / se:+.2f} (no control variate {(m_raw - exact) / se_raw:+.2f}); cost {math.log(s + g) - m_:.6f} against exact {math.log(s + g) - exact:.6f}")
    print("K1  M equal independent hosts")
    res = {}
    for i, M in enumerate((1, 2, 4)):
        res[M] = run_case(f"K1 M={M}", [1.0 / M] * M, 1.0, 0.7, seed=71 + i)
    print(f"      simulated cost relative to one host: M=2 {res[2][0] / res[1][0]:.4f} (1/2 expected), M=4 {res[4][0] / res[1][0]:.4f} (1/4 expected)")
    print("K2  unequal shares")
    run_case("K2", [0.7, 0.2, 0.1], 1.0, 0.7, seed=75)
    print("K3  correlated hosts")
    run_case("K3", [0.5, 0.5], 1.0, 0.7, rho=0.5, phi_c=0.7, seed=76)
    print("K4  hosts with different memory")
    run_case("K4", [0.5, 0.5], 1.0, [0.0, 0.9], seed=77)
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R3 PASSED" if n_fail == 0 else "R3 FAILED")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
