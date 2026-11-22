"""R1a: is the gap between simulation and the closed form in long rings a higher-order term?

Claim tested: at fixed n and phi, (simulated cost / closed form) - 1 is proportional to sigma^2
and vanishes as sigma -> 0, so the closed form is correct to second order for long rings too.
Hypotheses, grid, tolerances, budget: `protocols/R1a-large-ring-gap.md`, written before this ran.
Uses `engine/ring.py`, verified by `checks/r0_verification.py`.
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

SIGMAS = (0.15, 0.106, 0.075, 0.053)
STEPS, BURN = 120000, 2000


def closed_form(n, phi, s, gbar, sigma):
    lam0 = s + gbar
    e = gbar / lam0
    tail = 0.0 if phi == 0 else n * lam0 * phi ** n * gbar ** (n - 1) / ((lam0 - phi * s) ** n - (phi * gbar) ** n)
    return e ** 2 * sigma ** 2 / n * (0.5 + (phi / (1 - phi) if phi else 0.0) - tail)


def one_case(case):
    label, n, phi, s, gbar, sigma, reps, steps, burn, seed = case
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = ring.FluctuatingProduction(np.full(n, s), np.full(n, gbar), sigma, phi, reps, rng)
    out = ring.run(env, n, reps, steps, burn)
    m, se = ring.mean_se(out["rate_cv"])
    cost = math.log(s + gbar) - m
    f = closed_form(n, phi, s, gbar, sigma)
    return {"label": label, "n": n, "phi": phi, "s": s, "gbar": gbar, "sigma": sigma, "reps": reps,
            "steps": steps, "cost": cost, "se": se, "formula": f, "ratio": cost / f,
            "ratio_se": se / f, "cpu_s": time.time() - t0}


def cases():
    out, seed = [], 7100
    s, g = 2.0 / 3.0, 0.4
    for label, n, phi, reps in (("main", 16, 0.9, 800), ("main", 32, 0.9, 1600),
                                ("main", 32, 0.0, 800), ("main", 64, 0.9, 1600)):
        for sigma in SIGMAS:
            seed += 1
            out.append((label, n, phi, s, g, sigma, reps, STEPS, BURN, seed))
    for sigma in SIGMAS:
        seed += 1
        out.append(("H4 little staying", 4, 0.0, 0.05, s + g - 0.05, sigma, 800, STEPS, BURN, seed))
    out.append(("H3 long run", 32, 0.9, s, g, 0.15, 1600, 4 * STEPS, 4 * BURN, seed + 1))
    return out


if __name__ == "__main__":
    t0 = time.time()
    todo = sorted(cases(), key=lambda c: -c[1] * c[6] * c[7])       # longest first
    with mp.Pool(min(8, mp.cpu_count() - 1)) as pool:
        res = pool.map(one_case, todo, chunksize=1)
    res.sort(key=lambda r: (r["label"], r["n"], -r["phi"], -r["sigma"]))
    (HERE.parent / "results" / "R1a-large-ring-gap_20261113.json").write_text(json.dumps(res, indent=1))
    print("label               n   phi    s      sigma   formula      simulated    s.e.      ratio    excess/sigma^2")
    for r in res:
        ex = (r["ratio"] - 1) / r["sigma"] ** 2
        print(f"{r['label']:18s} {r['n']:3d}  {r['phi']:3.1f}  {r['s']:5.3f}  {r['sigma']:5.3f}  "
              f"{r['formula']:.5e}  {r['cost']:.5e}  {r['se']:.1e}  {r['ratio']:7.4f}  {ex:8.3f} ± {r['ratio_se'] / r['sigma'] ** 2:.3f}")
    print(f"processor time {sum(r['cpu_s'] for r in res) / 60:.1f} min; wall {time.time() - t0:.0f} s")
