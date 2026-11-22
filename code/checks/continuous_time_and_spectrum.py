"""The continuous-time form of the cost, the spectral reading of the discrete form, and two bounds. No simulation.

Claims checked, tolerances stated beforehand:
  T1  primary's blind continuous-time function `cost_ct` (`studies/ct/ct.py`) equals tertiary's sealed formula
          cost per unit time and per host = sigma^2 [ 1/(alpha T^2) - (1/T) R/(1 - R) ],  R = prod a_k/(a_k + alpha),
      a_k = r0 + nu_k + mu_k, T = sum 1/a_k (relative 1e-9; absolute 1e-12 for one stage, where both are zero);
  T2  the discrete closed form divided by the time step tends to the continuous form as the step goes to zero:
      the difference at step 0.001 is below 1% and about ten times smaller than at step 0.01;
  T3  in continuous time one stage costs nothing, and two or more stages cost something (no exception);
  T4  n equal stages: the total cost per generation tends to sigma^2 (n - 1)/2 in a slow environment and to
      sigma^2/(alpha tau) in a fast one (n >= 2), and rises with n at every alpha tried (no exception);
  T5  discrete time: the cost for an AR(1) host equals minus the average over the host's spectrum of the growth
      response to a sinusoid, (1/T) Re[g/(1-g)], g = E[exp(i w L)] (relative 1e-6);
  T6  discrete time, phi >= 0, primitive ring: the cost per generation and per host lies between sigma^2/(2T) and
      sigma^2 (1 + 2 theta)/(2T) (no exception).

Run: python3 -I checks/continuous_time_and_spectrum.py
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
from scipy import integrate, optimize

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

spec = importlib.util.spec_from_file_location("primary_ct", HERE.parent / "studies" / "ct" / "ct.py")
primary_ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary_ct)


def ct_from_rates(a, alpha):
    """Cost per unit time and per host, unit sigma^2, for lineage leaving rates a_k."""
    a = np.asarray(a, float)
    T = float(np.sum(1 / a))
    R = float(np.prod(a / (a + alpha)))
    return 1 / (alpha * T * T) - (R / (1 - R)) / T


def main():
    rng = np.random.default_rng(2026111407)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst, cnt = 0.0, 0
    for trial in range(500):
        n = int(rng.integers(1, 8)); nu = rng.uniform(0.3, 3, n); mu = rng.uniform(0, 1, n); b = rng.uniform(0.8, 3, n); alpha = rng.uniform(0.05, 8)
        r0 = optimize.brentq(lambda r: np.sum(np.log(b * nu)) - np.sum(np.log(r + nu + mu)), -np.min(nu + mu) + 1e-12, 1e3)
        mine = ct_from_rates(r0 + nu + mu, alpha)
        theirs = np.asarray(primary_ct.cost_ct(nu, mu, b, alpha, 1.0), float)
        d = float(np.max(np.abs(theirs - mine)))
        worst = max(worst, d / abs(mine) if n > 1 else d * 1e3); cnt += 1
    report("T1 primary's continuous-time function = tertiary's sealed formula", worst < 1e-9, f"{cnt} random cycles of 1 to 7 stages; largest relative difference {worst:.1e}")

    r1, r2, cnt = 0.0, [], 0
    for trial in range(300):
        n = int(rng.integers(2, 7)); a = rng.uniform(0.2, 3, n); alpha = rng.choice([0.05, 0.3, 1, 4]) * rng.uniform(0.5, 2)
        ref = ct_from_rates(a, alpha)
        err = []
        for dt in (1e-2, 1e-3):
            m = 1 - a * dt
            disc = float(theory.cost_transfer(m, 1 - m, 1.0, math.exp(-alpha * dt))[0]) / dt
            err.append(abs(disc - ref) / abs(ref))
        r1 = max(r1, err[1]); r2.append(err[0] / err[1]); cnt += 1
    report("T2 discrete form / step tends to the continuous form", r1 < 0.01 and 8 < float(np.median(r2)) < 12,
           f"{cnt} random cycles: largest relative difference at step 0.001 is {r1:.1e}; median ratio of the differences at steps 0.01 and 0.001 is {np.median(r2):.1f}")

    one = max(abs(ct_from_rates([a], al)) for a, al in zip(rng.uniform(0.2, 3, 2000), rng.uniform(0.02, 8, 2000)))
    least = min(ct_from_rates(rng.uniform(0.2, 3, int(rng.integers(2, 8))), rng.uniform(0.02, 8)) for _ in range(20000))
    report("T3 one stage costs nothing; two or more cost something", one < 1e-9 and least > 0, f"2,000 one-stage cases, largest |cost| {one:.1e}; 20,000 cases of 2 to 7 stages, smallest cost {least:.2e}")

    ok, w_slow, w_fast = True, 0.0, 0.0
    for alpha in (1e-6, 0.05, 0.2, 1.0, 5.0, 300.0):
        tot = [n * n * ct_from_rates(np.ones(n), alpha) for n in range(1, 13)]     # tau = 1: generation time n
        ok &= all(tot[i + 1] > tot[i] - 1e-12 for i in range(len(tot) - 1))
        if alpha == 1e-6:
            w_slow = max(abs(tot[n - 1] - (n - 1) / 2) for n in range(1, 13))
        if alpha == 300.0:
            w_fast = max(abs(tot[n - 1] * alpha - 1) for n in range(2, 13))
    report("T4 equal stages per generation: (n-1)/2 when slow, 1/(alpha tau) when fast, rising with n", ok and w_slow < 1e-3 and w_fast < 0.02,
           f"n = 1 to 12 at six values of alpha; slow limit off by at most {w_slow:.1e}; fast limit off by at most {100 * w_fast:.1f}% at alpha = 300")

    worst, cnt = 0.0, 0
    for trial in range(200):
        n = int(rng.integers(1, 7)); tau = 1 + rng.exponential(2.0, n) + 1e-2; phi = rng.uniform(0.05, 0.9)
        m = 1 - 1 / tau; T = tau.sum()
        cost = float(theory.cost_transfer(m, 1 - m, 1.0, phi)[0])
        def integrand(w):
            g = np.prod(np.exp(1j * w) * (1 - m) / (1 - m * np.exp(1j * w)))
            spec_ = (1 - phi * phi) / (1 - 2 * phi * math.cos(w) + phi * phi)
            return spec_ * (g / (1 - g)).real / T
        val = integrate.quad(integrand, 1e-9, math.pi, limit=400)[0] / math.pi
        worst = max(worst, abs(-val - cost) / cost); cnt += 1
    report("T5 the cost is minus the average over the spectrum of the response to a sinusoid", worst < 1e-6, f"{cnt} random rings of 1 to 6 stages; largest relative difference {worst:.1e}")

    lo, hi, cnt = np.inf, -np.inf, 0
    for trial in range(100000):
        n = int(rng.integers(1, 13)); tau = 1 + rng.exponential(rng.choice([0.2, 2.0, 20.0]), n) * (rng.random(n) < 0.8)
        if n > 1 and np.all(tau == 1.0):
            continue
        phi = rng.uniform(0, 0.999); theta = phi / (1 - phi); T = tau.sum(); m = 1 - 1 / tau
        gam = T * float(theory.cost_transfer(m, 1 - m, 1.0, phi)[0])
        k = gam * 2 * T                      # should lie between 1 and 1 + 2 theta
        lo = min(lo, k - 1); hi = max(hi, (k - (1 + 2 * theta)) / (1 + 2 * theta)); cnt += 1
    report("T6 bounds on the cost per generation: between 1/(2T) and (1 + 2 theta)/(2T)", lo > -1e-9 and hi < 1e-9, f"{cnt} random primitive rings: smallest (k - 1) = {lo:.1e}; largest relative excess over the upper bound = {hi:.1e}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
