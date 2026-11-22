"""Run R5b: with density dependence, is the cost of fluctuation the threshold for persistence?

Claim tested: checks P0, P1, P2 of `protocols/R5b-persistence-threshold.md`, fixed there before this script was
first run; P3 is reported without a criterion. Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import environments, theory  # noqa: E402

spec = importlib.util.spec_from_file_location("h4", HERE.parent / "studies" / "h4" / "h4.py")
h4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h4)

P_SHARE, K = 0.625, 1.0
REPS, STEPS = 200, 200000
checks = []


def record(hyp, name, ok, txt):
    checks.append((hyp, name, bool(ok), txt))
    print(f"  [{'PASS' if ok else 'FAIL'}] {hyp} {name}: {txt}", flush=True)


def rates(lam0):
    return P_SHARE * lam0, (1 - P_SHARE) * lam0


def run(n, lam0, sigma, phi, log_start, seed, reps=REPS, steps=STEPS):
    """Regulated ring. Returns per-replicate slope of log total over the second half, and mean log total over the last tenth."""
    s, g = rates(lam0)
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing([s] * n, [g] * n, [sigma] * n, [phi] * n, [0.0] * n, [0.0] * n, reps, rng)
    y = np.full((reps, n), 1.0 / n)
    logn = np.full(reps, float(log_start))
    half = steps // 2; tail = steps - steps // 10
    at_half = None; tail_sum = np.zeros(reps)
    for t in range(steps):
        s_t, g_t, _ = env.step()
        g_t = np.broadcast_to(g_t, (reps, n)).copy()
        xn = np.exp(np.minimum(logn, 700.0)) * y[:, n - 1]
        g_t[:, n - 1] = g_t[:, n - 1] / (1.0 + xn / K)
        y = y * s_t + np.roll(y * g_t, 1, axis=1)
        tot = y.sum(axis=1)
        y /= tot[:, None]
        logn += np.log(tot)
        if t == half - 1:
            at_half = logn.copy()
        if t >= tail:
            tail_sum += logn
    slope = (logn - at_half) / (steps - half)
    return slope, tail_sum / (steps - tail)


def main():
    t0 = time.time()
    out = []
    print("P0  no fluctuation: equilibrium number in the regulated stage")
    for n in (1, 2, 4, 8):
        s, g = rates(1.05)
        y = np.full(n, 1.0 / n); logn = 0.0
        for t in range(20000):
            gt = np.full(n, g); xn = math.exp(logn) * y[n - 1]
            gt[n - 1] = g / (1 + xn / K)
            y = y * s + np.roll(y * gt, 1)
            tot = y.sum(); y /= tot; logn += math.log(tot)
        xn = math.exp(logn) * y[n - 1]; exact = K * ((g / (1 - s)) ** n - 1)
        record("P0", f"n={n}", abs(xn / exact - 1) < 1e-8, f"number in stage n {xn:.10f} against K(R0 - 1) = {exact:.10f}")

    sigma = 0.4
    print("P1, P2  margins of -0.01 and +0.01 about the threshold")
    for n in (1, 4):
        for phi in (0.0, 0.9):
            cost = theory.cost_equal_ring(n, *rates(1.0), phi) * sigma ** 2 + float(h4.fourth_order_coefficient(n, *rates(1.0), phi)) * sigma ** 4
            for margin in (-0.01, 0.01):
                lam0 = math.exp(cost + margin)
                seed = 3000 + 100 * n + int(phi * 10) + (1 if margin > 0 else 0)
                slope, level = run(n, lam0, sigma, phi, math.log(1e-6), seed)
                sl, sl_se = float(slope.mean()), float(slope.std(ddof=1) / math.sqrt(len(slope)))
                row = dict(n=n, phi=phi, cost=cost, margin=margin, slope=sl, slope_se=sl_se, level=float(level.mean()))
                if margin < 0:
                    ok = abs(sl - margin) <= max(3.5 * sl_se, 0.0005)
                    record("P1", f"n={n} phi={phi}", ok, f"cost {cost:.5f}; slope {sl:+.5f} +- {sl_se:.5f} against {margin:+.3f}")
                else:
                    slope2, level2 = run(n, lam0, sigma, phi, math.log(10.0), seed + 50)
                    l1, l2 = float(level.mean()), float(level2.mean())
                    se_d = math.hypot(float(level.std(ddof=1)), float(level2.std(ddof=1))) / math.sqrt(len(level))
                    ok = (abs(sl) <= max(3.5 * sl_se, 0.0002)) and abs(l1 - l2) <= 3.5 * se_d
                    row.update(level_other_start=l2)
                    record("P2", f"n={n} phi={phi}", ok, f"cost {cost:.5f}; slope {sl:+.6f} +- {sl_se:.6f}; mean log size {l1:+.3f} from 1e-6 and {l2:+.3f} from 10 (s.e. of difference {se_d:.3f})")
                out.append(row)
    print("P3  settled mean log size against the margin above the threshold (n = 4, phi = 0.9; reported, no criterion)")
    n, phi = 4, 0.9
    cost = theory.cost_equal_ring(n, *rates(1.0), phi) * sigma ** 2 + float(h4.fourth_order_coefficient(n, *rates(1.0), phi)) * sigma ** 4
    for i, margin in enumerate((0.002, 0.005, 0.02, 0.05)):
        slope, level = run(n, math.exp(cost + margin), sigma, phi, math.log(1e-6), 3900 + i)
        print(f"      margin {margin:+.3f}: slope over second half {slope.mean():+.6f}; mean log size at the end {level.mean():+.3f} +- {level.std(ddof=1) / math.sqrt(len(level)):.3f}", flush=True)
        out.append(dict(n=n, phi=phi, cost=cost, margin=margin, slope=float(slope.mean()), level=float(level.mean())))
    n_fail = sum(1 for c in checks if not c[2])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R5b PASSED" if n_fail == 0 else "R5b FAILED")
    (HERE.parent / "results" / "R5b-persistence_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
