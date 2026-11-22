"""Run R9: the loss of ln R0 in a random environment is Gamma at the ordinary stays. Protocol: protocols/R9-random-R0.md.

For each case of random_r0.R9_CASES, simulates the ring with the same staying rates and
prod g_k/(1 - s_k) = exp(y), for y = Gamma_0 (run A) and y = Gamma at the given model's own stays (run B), and
measures the long-run growth of the logarithm of the total, with the first-order term subtracted.

Run: python3 -I checks/r9_random_r0.py
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
from engine import environments  # noqa: E402

spec = importlib.util.spec_from_file_location("random_r0", HERE / "random_r0.py")
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)

REPS, STEPS, BURN = 400, 200_000, 2_000


def growth(s, y, sig, phi, seed):
    s = np.asarray(s, float); n = len(s)
    g = (1 - s) * math.exp(y / n)
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing(list(s), list(g), list(sig), list(phi), [0.0] * n, [0.0] * n, REPS, rng)
    sig2 = np.asarray(sig, float) ** 2
    x = np.full((REPS, n), 1.0 / n); cum = np.zeros(REPS); ctrl = np.zeros(REPS)
    for t in range(BURN + STEPS):
        s_t, g_t, cv = env.step()
        x = x * s_t + np.roll(x * g_t, 1, axis=1)
        tot = x.sum(axis=1); x /= tot[:, None]
        if t >= BURN:
            cum += np.log(tot)
            ctrl += (np.log(g_t / g[None, :]) + 0.5 * sig2[None, :]) @ env.e_g
    per = (cum - ctrl) / STEPS
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(REPS))


def main():
    t0 = time.time(); out = []; fails = 0
    for idx, r in enumerate(rr.r9_predictions()):
        T0 = r["T0"]
        gA, seA = growth(r["s"], r["gamma0"], r["sigma"], r["phi"], 9000 + 10 * idx)
        gB, seB = growth(r["s"], r["gamma_given"], r["sigma"], r["phi"], 9005 + 10 * idx)
        c1 = bool(abs(gA * T0) <= max(3.5 * seA * T0, 0.06 * r["gamma0"]))
        pred = r["wrong_growth"]
        c2 = bool(gB * pred > 0 and abs(gB) > 5 * seB and abs(gB / pred - 1) < 0.25)
        fails += (not c1) + (not c2)
        row = dict(r, growth_A=gA, se_A=seA, growth_B=gB, se_B=seB, C1=c1, C2=c2); out.append(row)
        print(f"{r['case']}: Gamma_0 = {r['gamma0']:.6f}", flush=True)
        print(f"   [{'ok' if c1 else 'FAIL'}] C1 run A (scaled with Gamma_0): growth x T0 = {gA * T0:+.6f} +- {seA * T0:.6f}  ({100 * gA * T0 / r['gamma0']:+.1f}% of Gamma_0)", flush=True)
        print(f"   [{'ok' if c2 else 'FAIL'}] C2 run B (scaled with Gamma at the given model's stays): growth per step {gB:+.6f} +- {seB:.6f}, predicted {pred:+.6f} (ratio {gB / pred:.3f})", flush=True)
    res = (__import__('pathlib').Path(__file__).resolve().parent / '../results/R9-random-R0_20261114.json').resolve()
    res.write_text(json.dumps([{k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()} for row in out], indent=1))
    print()
    print(f"{time.time() - t0:.0f} s; " + ("ALL CRITERIA MET" if fails == 0 else f"{fails} CRITERIA NOT MET"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
