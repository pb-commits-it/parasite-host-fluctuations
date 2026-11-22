"""Points raised by primary's referee report on manuscript draft 2, rerun with our own engine; the statements about
hosts in series that replace the ones it found too wide; and primary's blind Leslie derivation against ours.

Claims, tolerances stated beforehand. "Stays" (a, b, ...) means the ring with s_k = 1 - 1/tau_k, g_k = 1/tau_k, so
that lambda = 1 and R0 = 1. Gamma is the cost per generation summed over the n hosts, in units of sigma^2.

Counterexamples quoted by the referee (each reproduced to 1e-6 relative):
  X1  Same mean stay, more hosts, LOWER Gamma: (10,10) -> (1,1,28) at phi = 0.9: 0.371428571 -> 0.213884214.
  X2  Adding a host with the mean stay: (2,8) -> (2,8,5) at phi = 0.9: 0.371698113 -> 0.742244574.
  X3  Fixed generation time, unequal division: (10,10) -> (18,1,1) at phi = 0.9 lowers the total cost per step,
      0.018571429 -> 0.015770548.
  X4  With memory a longer-than-mean added stay can raise Gamma: (10,10) -> (10,10,11): 0.371429 -> 0.582427.
  X5  Survival: stays (100,10), noise on stage 1, psi = 0.95: -0.093058252 (a gain at elasticity 0.9);
      psi = 0.99: -0.269432296; stays (2,100), stage 1, psi = 0.001: +0.000043250239 (a cost at elasticity 0.0098).
  X6  Short memory is not no memory: (500,500), phi = 0.5: Gamma (two hosts) 0.002992032 against 0.001.
  X7  Lengthening one stay can raise Gamma: (100,1) -> (100,2) at phi = 0.9: 0.027563807 -> 0.041372900.
  X8  One noisy transfer; adding a second stage whose transfer does not fluctuate: (100) -> (100,100) at
      phi = 0.9: Gamma 0.005 -> 0.040635593.
  X9  R0 and lambda: two equal stages s = 0.5, g = 1: lambda = 1.5, R0 = 4, ln lambda = 0.405465,
      ln R0 / T(lambda) = 0.462098, ln R0 / T0 = 0.346574 with T0 the generation time at the threshold
      (the last added 14 November after the second referee report, point 7); and ln R0 = T0 r - (1/2) sum tau0 (tau0 - 1) r^2 + O(r^3) (ratio of the
      remainders at r and r/2 within 15% of 8, 200 random rings).
  X10 One stage, one multiplier on both staying and transfer, s = g = 0.5: cost 1/2, where the sum of the two
      separate-entry coefficients is 1/4.

Statements about hosts in series (every transfer on its own host, same sigma^2, 0 <= phi < 1):
  H1  Band: 1/(2 taubar) <= Gamma <= (1 + 2 theta)/(2 taubar), taubar = T/n (no exception; proof in the notes).
  H2  One stage sits on the lower edge of the band, so at a given mean stay no cycle has a lower Gamma than the
      one-stage cycle (no exception).
  H3  Adding a host with the current mean stay: the change in Gamma equals
      [P(x - 1) + 1] / [(P - 1){P(1 + x/n) - 1}], P = prod(1 + tau_k/theta), x = T/theta
      (difference below 1e-9 of Gamma), and is not negative (no exception beyond rounding, -1e-12).
  H4  Lengthening every stay in the same proportion lowers Gamma (no exception; proof by Newton's inequalities).
  H5  Equal stays: Gamma falls as the stay lengthens, for every n and phi (no exception; proof in the notes).

Leslie matrices, fertility fluctuating:
  L1  primary's blind function cost_common_driver = our closed form (1 + 2 theta)/(2 T^2) - (1/T) R/(1 - R)
      (relative 1e-9);
  L2  primary's cost_independent_drivers = the published matrix formula with one driver per fertility
      (relative 1e-8), and = sum_a [ q_a^2 (1 + 2 theta)/(2 T^2) - q_a^2 phi^a / (T (1 - R)) ] from the lineage form.

Run: python3 -I checks/referee_ms_points.py

First run, 14 November 2026 about 03:50 (output kept as results/referee-ms-points_attempt1_20261114.txt): X6 and H3
failed, both through faults in this script and not in the formulas. X6: the referee's two numbers are totals over
the two hosts and the script compared them with values per host (0.001496 and 0.0005, the same ratio). H3: the
tolerance was relative to the change in Gamma, which can be 2e-9 where Gamma is 0.05, so rounding in the difference
of two values of Gamma reached 1e-5 of it; measured against Gamma the largest difference is 5e-13. The claim
texts above are the corrected ones.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

spec = importlib.util.spec_from_file_location("primary_leslie", HERE.parent / "studies" / "leslie" / "leslie.py")
primary_leslie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary_leslie)


def ring(stays):
    tau = np.asarray(stays, float)
    return 1 - 1 / tau, 1 / tau


def gamma_total(stays, phi):
    s, g = ring(stays)
    return float(np.sum(theory.per_generation_cost_transfer(s, g, 1.0, phi)))


def gamma_host(stays, phi):
    s, g = ring(stays)
    return float(theory.per_generation_cost_transfer(s, g, 1.0, phi)[0])


def step_total(stays, phi):
    s, g = ring(stays)
    return float(np.sum(theory.cost_transfer(s, g, 1.0, phi)))


def main():
    rng = np.random.default_rng(2026111408)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    def same(a, b, rel=1e-6):
        return abs(a - b) <= rel * abs(b)

    a, b = gamma_total((10, 10), 0.9), gamma_total((1, 1, 28), 0.9)
    report("X1 same mean stay, three hosts in place of two, lower Gamma", same(a, 0.371428571) and same(b, 0.213884214), f"{a:.9f} -> {b:.9f}")
    a, b = gamma_total((2, 8), 0.9), gamma_total((2, 8, 5), 0.9)
    report("X2 adding a host with the mean stay raises Gamma", same(a, 0.371698113) and same(b, 0.742244574), f"{a:.9f} -> {b:.9f}")
    a, b = step_total((10, 10), 0.9), step_total((18, 1, 1), 0.9)
    report("X3 fixed generation time divided unequally among more hosts: lower cost per step", same(a, 0.018571429) and same(b, 0.015770548), f"{a:.9f} -> {b:.9f}")
    a, b = gamma_total((10, 10), 0.9), gamma_total((10, 10, 11), 0.9)
    report("X4 with memory, an added stay longer than the mean can raise Gamma", same(a, 0.371429, 1e-5) and same(b, 0.582427, 1e-5), f"{a:.6f} -> {b:.6f}")
    s, g = ring((100, 10)); c1 = float(theory.cost_survival(s, g, 1.0, 0.95)[0]); c2 = float(theory.cost_survival(s, g, 1.0, 0.99)[0])
    s, g = ring((2, 100)); c3 = float(theory.cost_survival(s, g, 1.0, 0.001)[0])
    report("X5 survival: a gain at elasticity 0.9, a cost at elasticity 0.0098", same(c1, -0.093058252) and same(c2, -0.269432296) and same(c3, 0.000043250239),
           f"(100,10), psi 0.95: {c1:.9f}; psi 0.99: {c2:.9f}; (2,100), psi 0.001: {c3:.12f}")
    a, b = gamma_total((500, 500), 0.5), gamma_total((500, 500), 0.0)
    report("X6 memory a thousandth of a generation still triples the cost", same(a, 0.002992032) and same(b, 0.001), f"Gamma {a:.9f} against {b:.9f} without memory")
    a, b = gamma_total((100, 1), 0.9), gamma_total((100, 2), 0.9)
    report("X7 lengthening the short stay raises Gamma", same(a, 0.027563807) and same(b, 0.041372900), f"{a:.9f} -> {b:.9f}")
    a, b = gamma_host((100,), 0.9), gamma_host((100, 100), 0.9)
    report("X8 adding a stage with a constant transfer raises the cost of the one fluctuating host", same(a, 0.005) and same(b, 0.040635593), f"{a:.9f} -> {b:.9f}")

    s, g = np.array([0.5, 0.5]), np.array([1.0, 1.0])
    lam, m, tau, T = theory.ring_quantities(s, g)
    R0 = float(np.prod(g / (1 - s)))
    T0 = float(np.sum(1 / (1 - s)))
    ok9 = same(lam, 1.5) and same(R0, 4.0) and same(math.log(lam), 0.405465, 1e-5) and same(math.log(R0) / T, 0.462098, 1e-5) and same(math.log(R0) / T0, 0.346574, 1e-5)
    ratios = []
    for trial in range(200):
        n = int(rng.integers(1, 6)); tau0 = 1 + rng.exponential(2.0, n) + 0.05; s0 = 1 - 1 / tau0
        rem = []
        for r in (0.02, 0.01):
            lam_ = math.exp(r)
            lnR0 = float(np.sum(np.log((lam_ - s0) / (1 - s0))))          # prod(lam - s_k) = prod g_k
            rem.append(lnR0 - (tau0.sum() * r - 0.5 * np.sum(tau0 * (tau0 - 1)) * r * r))
        ratios.append(rem[0] / rem[1])
    ratios = np.array(ratios)
    report("X9 ln lambda is ln R0 / T only near the threshold", ok9 and np.all(np.abs(ratios / 8 - 1) < 0.15),
           f"s = 0.5, g = 1: lambda {lam:.3f}, R0 {R0:.1f}, ln lambda {math.log(lam):.6f}, ln R0/T {math.log(R0) / T:.6f}, ln R0/T0 {math.log(R0) / T0:.6f}; 200 rings: remainder ratio between {ratios.min():.2f} and {ratios.max():.2f} (8 for a cubic remainder)")

    amat = np.array([[1.0]])
    shared = theory.lineage_cost_shared(amat, [(0, 0)], [1.0], lambda l: 0.0)
    sep = 2 * 0.25 / 2            # two entries, each with elasticity 1/2: c^2/2 each
    report("X10 one multiplier on staying and transfer together", same(shared, 0.5) and same(sep, 0.25), f"cost {shared:.3f}; the two separate coefficients sum to {sep:.3f}")

    lo, hi, low_edge, cnt = np.inf, -np.inf, 0.0, 0
    for trial in range(100000):
        n = int(rng.integers(1, 10)); tau = 1 + rng.exponential(rng.choice([0.3, 3.0, 30.0]), n) * (rng.random(n) < 0.8)
        if n > 1 and np.all(tau == 1.0):
            continue
        phi = rng.uniform(0, 0.995); theta = phi / (1 - phi); tb = tau.mean()
        G = gamma_total(tau, phi)
        lo = min(lo, G * 2 * tb - 1); hi = max(hi, G * 2 * tb / (1 + 2 * theta) - 1); cnt += 1
        if n == 1:
            low_edge = max(low_edge, abs(G * 2 * tb - 1))
    report("H1 band: 1/(2 taubar) <= Gamma <= (1 + 2 theta)/(2 taubar)", lo > -1e-9 and hi < 1e-9, f"{cnt} random rings of 1 to 9 stages: smallest relative margin below {lo:.1e}, above {-hi:.1e}")
    report("H2 one stage sits on the lower edge", low_edge < 1e-9, f"largest departure among the one-stage cases {low_edge:.1e}")

    worst, least, cnt = 0.0, np.inf, 0
    for trial in range(50000):
        n = int(rng.integers(1, 9)); tau = 1 + rng.exponential(rng.choice([0.3, 3.0, 30.0]), n) + (1e-3 if n > 0 else 0)
        phi = rng.uniform(0.02, 0.98); theta = phi / (1 - phi)
        G0 = gamma_total(tau, phi)
        d = gamma_total(np.append(tau, tau.mean()), phi) - G0
        P = float(np.prod(1 + tau / theta)); x = tau.sum() / theta
        f = (P * (x - 1) + 1) / ((P - 1) * (P * (1 + x / n) - 1))
        worst = max(worst, abs(d - f) / G0)
        least = min(least, d); cnt += 1
    report("H3 adding a host with the mean stay never lowers Gamma, and the change has the stated form", worst < 1e-9 and least > -1e-12, f"{cnt} random rings of 1 to 8 stages: largest difference from the formula, as a share of Gamma, {worst:.1e}; smallest change {least:.2e}")

    least, cnt = np.inf, 0
    for trial in range(50000):
        n = int(rng.integers(1, 9)); tau = 1 + rng.exponential(rng.choice([0.3, 3.0, 30.0]), n) * (rng.random(n) < 0.8)
        if n > 1 and np.all(tau == 1.0):
            continue
        phi = rng.uniform(0.0, 0.98); c = 1 + rng.exponential(0.5) + 1e-3
        least = min(least, (gamma_total(tau, phi) - gamma_total(c * tau, phi)) / gamma_total(tau, phi)); cnt += 1
    report("H4 lengthening every stay in the same proportion lowers Gamma", least > 0, f"{cnt} random rings and factors: smallest relative fall {least:.2e}")

    least, cnt = np.inf, 0
    for trial in range(50000):
        n = int(rng.integers(1, 30)); t1 = 1 + rng.exponential(3.0) + 1e-3; t2 = t1 * (1 + rng.exponential(0.3) + 1e-3); phi = rng.uniform(0, 0.99)
        least = min(least, (gamma_total([t1] * n, phi) - gamma_total([t2] * n, phi)) / gamma_total([t1] * n, phi)); cnt += 1
    report("H5 equal stays: Gamma falls as the stay lengthens", least > 0, f"{cnt} random cases, n from 1 to 29: smallest relative fall {least:.2e}")

    w1, w2, w3, cnt = 0.0, 0.0, 0.0, 0
    while cnt < 1500:
        d = int(rng.integers(2, 10)); mfert = rng.uniform(0, 2, d) * (rng.random(d) < 0.7); mfert[-1] = rng.uniform(0.1, 2)
        psurv = rng.uniform(0.3, 1.0, d - 1); phi = rng.uniform(0.02, 0.97)
        try:
            lam, q = primary_leslie.demographic_distribution(mfert, psurv)
        except ValueError:
            continue
        theta = phi / (1 - phi); ages = np.arange(1, d + 1); T = float(ages @ q); R = float(q @ phi ** ages)
        mine = (1 + 2 * theta) / (2 * T * T) - (R / (1 - R)) / T
        w1 = max(w1, abs(primary_leslie.cost_common_driver(mfert, psurv, phi) - mine) / max(abs(mine), 1e-6))
        A = np.zeros((d, d)); A[0, :] = mfert; A[np.arange(1, d), np.arange(d - 1)] = psurv
        ent = [(0, j) for j in range(d) if mfert[j] > 0]
        mat = theory.matrix_cost(A, ent, list(range(len(ent))), [1.0] * len(ent), [phi] * len(ent))
        ind = primary_leslie.cost_independent_drivers(mfert, psurv, phi)
        lin = float(np.sum(q * q) * (1 + 2 * theta) / (2 * T * T) - np.sum(q * q * phi ** ages) / (T * (1 - R)))
        w2 = max(w2, abs(ind - mat) / max(abs(mat), 1e-6)); w3 = max(w3, abs(lin - mat) / max(abs(mat), 1e-6)); cnt += 1
    report("L1 primary's blind Leslie function = our closed form (one common driver)", w1 < 1e-9, f"{cnt} random primitive life tables of 2 to 9 ages: largest relative difference {w1:.1e}")
    report("L2 one driver per fertility: primary's function = the published matrix formula = the lineage form", w2 < 1e-8 and w3 < 1e-8, f"largest relative differences {w2:.1e} and {w3:.1e}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
