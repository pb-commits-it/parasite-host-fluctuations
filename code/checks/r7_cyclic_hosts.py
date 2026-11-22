"""Run R7: a host that cycles (stationary AR(2) quasi-cycle), one fluctuating transfer of a ring.

Claim tested: checks Q0 to Q6 of `protocols/R7-cyclic-hosts.md`, fixed there before this script was first run.
Exit status is 0 only if all pass.
"""
from __future__ import annotations

import cmath
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

SIGMA, REPS, STEPS, BURN = 0.03, 128, 500_000, 2_000


def ar2_coeffs(period, r):
    w = 2 * math.pi / period
    return 2 * r * math.cos(w), -r * r


def cost_cyclic(stays, period, r):
    """Second-order cost per step, unit sigma^2, of one transfer of a ring (lam = 1) whose host is an AR(2) quasi-cycle."""
    tau = np.asarray(stays, float); m = 1 - 1 / tau; T = tau.sum(); c = 1 / T
    w = 2 * math.pi / period
    a1, a2 = ar2_coeffs(period, r)
    z = r * cmath.exp(1j * w); rho1 = a1 / (1 - a2)
    B = 1 + 1j * (math.cos(w) - rho1 / r) / math.sin(w)
    R = np.prod((1 - m) * z / (1 - m * z))
    return float(c * c / 2 + c * c * (B * z / (1 - z)).real - c * (B * R / (1 - R)).real)


def simulate(stays, a1, a2, seed):
    """Cost per step in units of sigma^2 and its standard error. The host series is AR(2) (AR(1) when a2 = 0)."""
    tau = np.asarray(stays, float); n = len(tau); s = 1 - 1 / tau; g = 1 / tau; c = 1 / tau.sum()
    rng = np.random.default_rng(seed)
    rho1 = a1 / (1 - a2)
    var_innov = (1 - a2) * (1 - rho1 * a1 - a2 * (a1 * rho1 + a2)) / (1 - a2) if a2 != 0 else 1 - a1 * a1
    # unit-variance stationary AR(2): innovation variance = 1 - a1 rho1 - a2 rho2, rho2 = a1 rho1 + a2
    rho2 = a1 * rho1 + a2
    var_innov = 1 - a1 * rho1 - a2 * rho2
    sd = math.sqrt(var_innov)
    z1 = rng.standard_normal(REPS); z0 = rho1 * z1 + math.sqrt(1 - rho1 * rho1) * rng.standard_normal(REPS)   # (z_{t-1}, z_t) stationary pair
    zprev, zcur = z1, z0
    x = np.full((REPS, n), 1.0 / n)
    cum = np.zeros(REPS); zsum = np.zeros(REPS)
    gg = np.tile(g, (REPS, 1))
    for t in range(BURN + STEPS):
        znew = a1 * zcur + a2 * zprev + sd * rng.standard_normal(REPS)
        zprev, zcur = zcur, znew
        gt = gg.copy(); gt[:, 0] *= np.exp(SIGMA * zcur - SIGMA * SIGMA / 2)
        x = x * s + np.roll(x * gt, 1, axis=1)
        tot = x.sum(axis=1); x /= tot[:, None]
        if t >= BURN:
            cum += np.log(tot); zsum += zcur
    growth = cum / STEPS - c * SIGMA * zsum / STEPS
    cost = -growth / SIGMA ** 2
    return float(cost.mean()), float(cost.std(ddof=1) / math.sqrt(REPS))


def main():
    t0 = time.time()
    out = []; checks = []
    a1, a2 = ar2_coeffs(12.0, 0.9)
    cases = [("Q0", [3.0] * 4, 0.5, 0.0, None), ("Q1", [12.0], a1, a2, 0.0417), ("Q2", [3.0] * 4, a1, a2, 0.1047), ("Q3", [1.5] * 8, a1, a2, -0.0764),
             ("Q4", [3.0] * 2, a1, a2, 0.2561), ("Q5", [1.5] * 4, a1, a2, 0.3903), ("Q6", [6.0] * 4, a1, a2, 0.0259)]
    for idx, (label, stays, b1, b2, written) in enumerate(cases):
        T = float(sum(stays))
        if label == "Q0":
            m = 1 - 1 / np.asarray(stays); pred = float(theory.cost_transfer(m, 1 - m, 1.0, 0.5)[0])
        else:
            pred = cost_cyclic(stays, 12.0, 0.9)
            assert abs(T * pred - written) < 6e-5, (label, T * pred, written)      # the prediction printed in the protocol
        est, se = simulate(stays, b1, b2, 31000 + idx)
        z = (est - pred) / se
        ok = abs(est - pred) <= max(3.5 * se, 0.015 * abs(pred))
        if label == "Q3":
            ok = ok and est < -3.5 * se
        out.append(dict(case=label, stays=stays, T=T, predicted_cost=pred, simulated_cost=est, se=se, z=z, gamma_pred=T * pred, gamma_sim=T * est, gamma_se=T * se, uncorrelated=1 / (2 * T)))
        txt = f"Gamma/sigma^2 simulated {T * est:+.4f} +- {T * se:.4f}, predicted {T * pred:+.4f} (z = {z:+.1f}; {100 * (est / pred - 1):+.2f}%); uncorrelated host {1 / (2 * T):.4f}"
        checks.append((label, ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] {label} n={len(stays)} stay={stays[0]:g}: {txt}", flush=True)
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R7 PASSED" if n_fail == 0 else "R7 FAILED")
    (HERE.parent / "results" / "R7-cyclic-hosts_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
