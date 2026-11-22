"""Run R1d: risk of extinction for rings of different length at the same multiplication per generation.

Claim tested: checks M1 and M2 of `protocols/R1d-extinction-at-fixed-R0.md`, fixed there before this script was first
run. Exit status is 0 only if both pass.
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
from engine import theory  # noqa: E402

spec = importlib.util.spec_from_file_location("r1c", HERE / "r1c_large_noise.py")
r1c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1c)

X0 = math.log(10.0)
GENERATIONS = 1500
TAU = 8.0 / 3.0


def rates_for(n, ln_r0):
    x = math.exp(ln_r0 / n)
    lam0 = x / (0.375 + 0.625 * x)
    return 0.625 * lam0, 0.375 * lam0, lam0


def main():
    t0 = time.time()
    out = []; checks = []
    for label, phi, sigma, ln_r0 in (("M1", 0.0, 0.25, 0.01572), ("M2", 0.9, 0.12, 0.03)):
        print(f"{label}: phi = {phi}, sigma = {sigma}, ln R0 = {ln_r0}")
        ps = []
        for n in (1, 2, 4, 8):
            s, g, lam0 = rates_for(n, ln_r0)
            steps = int(round(GENERATIONS * TAU * n))
            p, p_se, mu, v = r1c.extinction(n, s, g, sigma, phi, 7000 + 100 * n + int(phi * 10), reps=2000, steps=steps, burn=1000, depth=X0)
            # leading-order prediction per generation, at the threshold rates
            gam = float(theory.per_generation_cost_transfer([0.625] * n, [0.375] * n, sigma ** 2, phi).sum())
            tv = sigma ** 2 * (1 + phi) / (1 - phi) / TAU
            pred = r1c.lande_orzack(ln_r0 - gam, tv, X0, GENERATIONS)
            T = n * TAU
            out.append(dict(test=label, n=n, lam0=lam0, p=p, p_se=p_se, mean_per_generation=mu * T, var_per_generation=v * T, gamma=gam, tv=tv, predicted=pred))
            ps.append((p, p_se))
            print(f"    n={n}: extinct {p:.4f} +- {p_se:.4f}; growth per generation {mu * T:+.5f} (theory {ln_r0 - gam:+.5f}); "
                  f"variance per generation {v * T:.5f} (theory {tv:.5f}); diffusion formula with the theory's values {pred:.4f}", flush=True)
        vals = [p for p, _ in ps]
        if label == "M1":
            ok = max(vals) - min(vals) <= 0.10
            txt = f"extinction probabilities {', '.join(f'{v:.3f}' for v in vals)}; spread {max(vals) - min(vals):.3f}"
        else:
            steps_ok = all(ps[i + 1][0] - ps[i][0] > 2 * math.hypot(ps[i + 1][1], ps[i][1]) for i in range(3))
            ok = steps_ok and (vals[3] - vals[0] > 0.3)
            txt = f"extinction probabilities {', '.join(f'{v:.3f}' for v in vals)}; each step up by more than 2 s.e.: {steps_ok}; eight stages minus one: {vals[3] - vals[0]:+.3f}"
        checks.append((label, ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {txt}")
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R1d PASSED" if n_fail == 0 else "R1d FAILED")
    (HERE.parent / "results" / "R1d-extinction-fixed-R0_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
