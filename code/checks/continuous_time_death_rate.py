"""A fluctuating death rate in the continuous-time ring. No simulation here; primary's simulation is in
studies/ct-survival/results_cts.txt.

Claims, tolerances stated beforehand:
  D1  primary's blind function `cost_death` (studies/ct-survival/cts.py) equals the limit tertiary wrote down beforehand
      (notes/continuous-time.md, section 7): sigma^2 p_j [ p_j/alpha - 1/((a_j + alpha)(1 - R)) ],
      p_j = (1/a_j)/T, R = prod a_k/(a_k + alpha) (relative 1e-9; absolute 1e-12 for one stage).
  D2  The discrete survival formula (theory.cost_survival) with step dt, staying share 1 - a_k dt, noise variance
      (sigma dt)^2 on the log of the staying multiplier and psi = exp(-alpha dt), divided by dt, tends to the same
      value at first order in the step: the difference falls about ten-fold from dt = 0.01 to 0.001 and again to
      0.0001 (median ratios between 8 and 12), and is below 1% at dt = 0.0001.
  D3  Sign: zero for one stage, negative (a gain) for every cycle of two or more stages (no exception).
  D4  Fast limit: alpha * cost -> - sigma^2 p_j (1 - p_j) (within 1% at alpha = 1000 times the largest rate), the
      form of Monmarche, Schreiber & Strickler (2025, Proposition 2) for a fluctuating diagonal entry.

Run: python3 -I checks/continuous_time_death_rate.py

First run, 14 November 2026 about 03:50 (output kept as results/continuous-time-death-rate_attempt1_20261114.txt):
D2 as first stated ("below 1% at dt = 0.001") failed in 4% of the 300 cases, the worst at 5.1%. The convergence was
first order in every case (ten-fold per decade of dt). The threshold of 1% at dt = 0.001 was a guess of mine that
did not allow for the size of the first-order term: the discrete model fixes the arithmetic mean of the staying
multiplier, which costs p^2 sigma^2 dt / 2 per unit time and vanishes with the step (4.9% of the limit in the worst
case, against 5.1% observed). D2 above is the restated claim. The formula under test was not changed.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
from scipy import optimize

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

spec = importlib.util.spec_from_file_location("primary_cts", HERE.parent / "studies" / "ct-survival" / "cts.py")
primary_cts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary_cts)


def death_cost(a, alpha, j):
    a = np.asarray(a, float)
    T = float(np.sum(1 / a)); p = (1 / a[j]) / T
    R = float(np.prod(a / (a + alpha)))
    return p * (p / alpha - 1 / ((a[j] + alpha) * (1 - R)))


def main():
    rng = np.random.default_rng(2026111409)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst, cnt = 0.0, 0
    for trial in range(500):
        n = int(rng.integers(1, 8)); nu = rng.uniform(0.3, 3, n); mu = rng.uniform(0, 1, n); b = rng.uniform(0.8, 3, n); alpha = rng.uniform(0.05, 8); j = int(rng.integers(0, n))
        r0 = optimize.brentq(lambda r: np.sum(np.log(b * nu)) - np.sum(np.log(r + nu + mu)), -np.min(nu + mu) + 1e-12, 1e3)
        mine = death_cost(r0 + nu + mu, alpha, j)
        theirs = float(primary_cts.cost_death(nu, mu, b, alpha, 1.0, j))
        d = abs(theirs - mine)
        worst = max(worst, d / abs(mine) if n > 1 else d * 1e3); cnt += 1
    report("D1 primary's blind function = tertiary's limit", worst < 1e-9, f"{cnt} random cycles of 1 to 7 stages; largest relative difference {worst:.1e}")

    r1, r2, r3, cnt = 0.0, [], [], 0
    for trial in range(300):
        n = int(rng.integers(2, 7)); a = rng.uniform(0.2, 3, n); alpha = rng.choice([0.05, 0.3, 1, 4]) * rng.uniform(0.5, 2); j = int(rng.integers(0, n))
        ref = death_cost(a, alpha, j)
        err = []
        for dt in (1e-2, 1e-3, 1e-4):
            m = 1 - a * dt
            disc = float(theory.cost_survival(m, 1 - m, dt * dt, math.exp(-alpha * dt))[j]) / dt
            err.append(abs(disc - ref) / abs(ref))
        r1 = max(r1, err[2]); r2.append(err[0] / err[1]); r3.append(err[1] / err[2]); cnt += 1
    report("D2 discrete survival formula / step tends to it at first order", r1 < 0.01 and 8 < float(np.median(r2)) < 12 and 8 < float(np.median(r3)) < 12,
           f"{cnt} random cycles: largest relative difference at step 0.0001 is {r1:.1e}; median ratios of the differences between successive steps {np.median(r2):.1f} and {np.median(r3):.1f}")

    one = max(abs(death_cost([a], al, 0)) for a, al in zip(rng.uniform(0.2, 3, 2000), rng.uniform(0.02, 8, 2000)))
    most = -np.inf
    for trial in range(20000):
        n = int(rng.integers(2, 8)); a = rng.uniform(0.2, 3, n)
        most = max(most, death_cost(a, rng.uniform(0.02, 8), int(rng.integers(0, n))))
    report("D3 zero for one stage; a gain for two or more", one < 1e-9 and most < 0, f"2,000 one-stage cases, largest |cost| {one:.1e}; 20,000 cases of 2 to 7 stages, largest (least negative) cost {most:.2e}")

    worst, cnt = 0.0, 0
    for trial in range(500):
        n = int(rng.integers(2, 7)); a = rng.uniform(0.2, 3, n); j = int(rng.integers(0, n)); alpha = 1000 * a.max()
        T = np.sum(1 / a); p = (1 / a[j]) / T
        worst = max(worst, abs(alpha * death_cost(a, alpha, j) / (-p * (1 - p)) - 1)); cnt += 1
    report("D4 fast limit: - p (1 - p) / alpha", worst < 0.01, f"{cnt} random cycles at alpha = 1000 times the largest rate: largest relative difference {worst:.1e}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
