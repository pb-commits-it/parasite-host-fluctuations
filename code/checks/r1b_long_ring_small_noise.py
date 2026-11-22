"""R1b: is the closed form the exact second-order coefficient in long rings?

Claim tested: at noise levels below those of R1a, simulated cost / closed form equals
1 + c sigma^2 + d sigma^4 with c, d fixed beforehand from R1a and no constant term.
Predictions and pass rule: `protocols/R1b-long-ring-small-noise.md`, written before this ran.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import ring  # noqa: E402
from checks.r1a_large_ring_gap import closed_form  # noqa: E402

S, G, PHI = 2.0 / 3.0, 0.4, 0.9
CASES = {32: (0.02, 1.0034), 64: (0.012, 1.0044)}       # n: (sigma, predicted ratio)
BATCHES, REPS, STEPS, BURN = 8, 800, 120000, 5000


def batch(job):
    n, sigma, seed = job
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = ring.FluctuatingProduction(np.full(n, S), np.full(n, G), sigma, PHI, REPS, rng)
    out = ring.run(env, n, REPS, STEPS, BURN)
    return n, (math.log(S + G) - out["rate_cv"]).tolist(), time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, CASES[n][0], 9100 + 10 * n + b) for n in (64, 32) for b in range(BATCHES)]
    with mp.Pool(min(8, mp.cpu_count() - 1)) as pool:
        done = pool.map(batch, jobs, chunksize=1)
    out, ok = [], True
    for n, (sigma, pred) in CASES.items():
        costs = np.concatenate([np.array(c) for m, c, _ in done if m == n])
        f = closed_form(n, PHI, S, G, sigma)
        ratio, se = costs.mean() / f, costs.std(ddof=1) / math.sqrt(len(costs)) / f
        z = (ratio - pred) / se
        good = abs(z) <= 3 and se <= 0.0025
        ok &= good
        out.append({"n": n, "sigma": sigma, "replicates": int(len(costs)), "formula": f,
                    "ratio": float(ratio), "ratio_se": float(se), "predicted": pred, "z": float(z), "pass": bool(good)})
        print(f"n = {n}, sigma = {sigma}: ratio {ratio:.5f} ± {se:.5f} (s.e.), predicted {pred:.4f}, "
              f"z = {z:+.2f}, {len(costs)} replicates [{'PASS' if good else 'FAIL'}]")
    (HERE.parent / "results" / "R1b-long-ring-small-noise_20261113.json").write_text(json.dumps(out, indent=1))
    print(f"processor time {sum(c for _, _, c in done) / 60:.1f} min; wall {time.time() - t0:.0f} s")
    sys.exit(0 if ok else 1)
