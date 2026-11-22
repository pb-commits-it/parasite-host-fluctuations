"""The formulas of the manuscript's result on a steady ("parking") stage, after the third referee report. No simulation.

One fluctuating host governs the transfer out of a stage with lineage stay a; a second stage with stay b has a
constant transfer. theta = phi/(1 - phi). Gamma is the loss per generation in units of sigma^2.

Claims, tolerances stated beforehand:
  P1  Gamma(a, b) = (1 + 2 theta)/(2(a + b)) - theta^2/(theta(a + b) + a b) equals the engine's value
      (theory.per_generation_cost_transfer with one fluctuating transfer) on 2,000 random cases (relative 1e-9).
  P2  The referee's counterexamples and ours: a = 2, b = 1, phi = 0.1: 0.25 alone, 0.198413 with the steady stage
      (lower); a = b = 100, phi = 0.9: 0.005 alone, 0.040636 with (higher); a = 8/3, b = 4/3, phi = 0.9: 0.1875
      alone, 0.327247 with (higher) (relative 1e-4). (First run: this claim failed because I had written the last
      value to four figures, 0.3272, and asked for agreement to 1e-4; the computed value is 0.327247.)
  P3  Long steady stage: (a + b) Gamma -> (1 + 2 theta)/2 - theta^2/(theta + a); for a = 8/3, phi = 0.9 that is
      2.557143, and not 9.5 (relative 1e-4 at b = 1e7).
  P4  Without memory, adding a stage with variance v and stay t to a cycle with total variance S and generation
      time T changes Gamma by (v T - S t)/(2 T (T + t)) (2,000 random heterogeneous cycles, relative 1e-9), so it
      lowers Gamma exactly when v/t < S/T.
  P5  The diffusion index E = 2(ln R0 - Gamma)/(T V), T V = (1 + 2 theta)/(a + b) in units of sigma^2: writing
      k = 2 Gamma/(T V), 1/(1 + 2 theta) <= k <= 1 (no exception in 20,000 random cases); without memory k = 1.
  P6  Adding a steady stay D to a one-stage cycle raises E whenever D ln R0 > theta sigma^2 (no exception in
      20,000 random cases with positive drift before and after), and can lower it otherwise: a = b = 100,
      phi = 0.9, ln R0 = 0.05 sigma^2: E 0.4737 -> 0.1971 (relative 1e-3).

  P7  (added after the fourth referee report, whose identity this is) For a steady stage of stay b added to a
      one-stage cycle with stay a, with L = ln R0 / sigma^2:
          E(a, b) - E(a) = (2 b / (1 + 2 theta)) [ L - a theta / (a theta + b (a + theta)) ]
      (relative 1e-9 on 5,000 random cases), so the index rises exactly when L exceeds a theta/(a theta + b(a + theta)),
      that is when b > (a theta/(a + theta)) (1 - L)/L. For run R1f (a = 10, phi = 0.9, L = 0.2) the critical stay
      is 18.95 steps: a stay of 10 lowers the index and a stay of 40 raises it. The condition D ln R0 > theta sigma^2
      of P6 is sufficient and not necessary: a = b = 100, phi = 0.9, L = 0.08 has E rising, 0.7895 -> 0.8287.

  P8  (added after the fifth referee report, whose observation this is) The critical stay of P7 tends to infinity
      only as L -> 0. Where the one-stage cycle itself stops growing at leading order, L = 1/(2a), it is
      a theta (2a - 1)/(a + theta) (relative 1e-9 on 5,000 random cases, with E(a) = 0 and E(a, b) = 0 there);
      for a = 10, phi = 0.9 that is 90 steps.
  P9  (same report) The 18.95 steps of P7 is where the index E regains its value, not where the probability of
      falling within a finite time does. For the diffusion with drift ln R0 - Gamma and variance (1 + 2 theta)
      sigma^2/(a + b) per generation, the probability of falling by ln 10 within H = 6,000 generations is
          Phi((-x - mu H)/sqrt(W H)) + exp(-2 mu x/W) Phi((-x + mu H)/sqrt(W H)).
      With sigma = 0.15, a = 10: for phi = 0.9, L = 0.2 and b = 0, 10, 40, 160 it is 0.6835, 0.8086, 0.2763, 0.0009;
      for phi = 0, L = 0.06 and b = 0, 10 it is 0.4110, 0.0362 (the referee's values; absolute 5e-4). In the first
      family it regains its b = 0 value at b = 16.93 (absolute 0.01), below 18.95.

Run: python3 -I checks/parking_stage_formulas.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402


def gamma2(a, b, phi):
    if phi == 0:
        return 1 / (2 * (a + b))
    th = phi / (1 - phi)
    return (1 + 2 * th) / (2 * (a + b)) - th * th / (th * (a + b) + a * b)


def gamma1(a):
    return 1 / (2 * a)


def engine(stays, phi):
    tau = np.asarray(stays, float); m = 1 - 1 / tau
    return float(theory.per_generation_cost_transfer(m, 1 - m, 1.0, phi)[0])


def main():
    rng = np.random.default_rng(2026111414)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst = 0.0
    for trial in range(2000):
        a, b = 1 + rng.exponential(5.0, 2) + 1e-3; phi = rng.uniform(0, 0.98)
        worst = max(worst, abs(gamma2(a, b, phi) / engine([a, b], phi) - 1))
    report("P1 two-stage formula = engine", worst < 1e-9, f"2,000 random cases: largest relative difference {worst:.1e}")

    vals = [gamma1(2), gamma2(2, 1, 0.1), gamma1(100), gamma2(100, 100, 0.9), gamma1(8 / 3), gamma2(8 / 3, 4 / 3, 0.9)]
    want = [0.25, 0.198413, 0.005, 0.040636, 0.1875, 0.327247]
    report("P2 a steady stage can lower or raise the loss", all(abs(v / w - 1) < 1e-4 for v, w in zip(vals, want)), "; ".join(f"{v:.6f}" for v in vals))

    th = 9.0; a = 8 / 3; lim = (1 + 2 * th) / 2 - th * th / (th + a); got = (a + 1e7) * gamma2(a, 1e7, 0.9)
    report("P3 long steady stage: the constant", abs(got / lim - 1) < 1e-4 and abs(lim - 2.557143) < 1e-6, f"(a + b) Gamma at b = 1e7: {got:.6f}; limit {lim:.6f}; (1 + 2 theta)/2 = {(1 + 2 * th) / 2}")

    worst = 0.0
    for trial in range(2000):
        n = int(rng.integers(1, 7)); tau = 1 + rng.exponential(3.0, n) + 1e-3; var = rng.uniform(0, 2, n); t = 1 + rng.exponential(3.0); v = rng.uniform(0, 2)
        m = 1 - 1 / tau; old = float(theory.per_generation_cost_transfer(m, 1 - m, var, 0.0).sum())
        tau2 = np.append(tau, t); m2 = 1 - 1 / tau2; new = float(theory.per_generation_cost_transfer(m2, 1 - m2, np.append(var, v), 0.0).sum())
        T, S = tau.sum(), var.sum(); f = (v * T - S * t) / (2 * T * (T + t))
        worst = max(worst, abs((new - old) - f) / max(old, 1e-9))
    report("P4 no memory: the change on adding a host", worst < 1e-9, f"2,000 random cycles: largest difference from (vT - St)/(2T(T + t)), as a share of Gamma, {worst:.1e}")

    lo, hi = np.inf, -np.inf
    for trial in range(20000):
        a, b = 1 + rng.exponential(5.0, 2) + 1e-3; phi = rng.uniform(0, 0.98); th = phi / (1 - phi)
        k = 2 * gamma2(a, b, phi) * (a + b) / (1 + 2 * th)
        lo = min(lo, k - 1 / (1 + 2 * th)); hi = max(hi, k - 1)
    report("P5 the bounded term k of the index", lo > -1e-12 and hi < 1e-12, f"20,000 random cases: smallest k - 1/(1 + 2 theta) = {lo:.1e}; largest k - 1 = {hi:.1e}")

    least, cnt = np.inf, 0
    while cnt < 20000:
        a = 1 + rng.exponential(10.0) + 1e-3; D = 1 + rng.exponential(10.0); phi = rng.uniform(0.01, 0.98); th = phi / (1 - phi)
        lnr = rng.uniform(0, 3) * th / D            # ln R0 in units of sigma^2
        if D * lnr <= th or lnr <= gamma1(a) or lnr <= gamma2(a, D, phi):
            continue
        E0 = 2 * (lnr - gamma1(a)) * a / (1 + 2 * th); E1 = 2 * (lnr - gamma2(a, D, phi)) * (a + D) / (1 + 2 * th)
        least = min(least, E1 - E0); cnt += 1
    th = 9.0; lnr = 0.05
    E0 = 2 * (lnr - gamma1(100)) * 100 / (1 + 2 * th); E1 = 2 * (lnr - gamma2(100, 100, 0.9)) * 200 / (1 + 2 * th)
    report("P6 the index rises when D ln R0 > theta sigma^2, and can fall otherwise", least > 0 and abs(E0 / 0.4737 - 1) < 1e-3 and abs(E1 / 0.1971 - 1) < 1e-3,
           f"{cnt} random cases meeting the condition: smallest rise {least:.2e}; counterexample E {E0:.4f} -> {E1:.4f}")

    worst = 0.0
    for trial in range(5000):
        a = 1 + rng.exponential(10.0) + 1e-3; b = 1 + rng.exponential(10.0); phi = rng.uniform(0.01, 0.98); th = phi / (1 - phi); L = rng.uniform(0.0, 1.5)
        E0 = 2 * (L - gamma1(a)) * a / (1 + 2 * th); E1 = 2 * (L - gamma2(a, b, phi)) * (a + b) / (1 + 2 * th)
        f = 2 * b / (1 + 2 * th) * (L - a * th / (a * th + b * (a + th)))
        worst = max(worst, abs((E1 - E0) - f) / max(abs(E0), abs(E1), 1e-6))
    th = 9.0; bcrit = (10 * th / (10 + th)) * (1 - 0.2) / 0.2
    E0 = 2 * (0.08 - gamma1(100)) * 100 / 19; E1 = 2 * (0.08 - gamma2(100, 100, 0.9)) * 200 / 19
    report("P7 the exact condition for the index to rise when a steady stage is added to a one-stage cycle", worst < 1e-9 and abs(bcrit - 18.947) < 1e-3 and abs(E0 - 0.7895) < 1e-4 and abs(E1 - 0.8287) < 1e-4,
           f"5,000 random cases: largest relative difference {worst:.1e}; critical stay for run R1f {bcrit:.2f}; a = b = 100, L = 0.08: E {E0:.4f} -> {E1:.4f}")

    worst = 0.0
    for trial in range(5000):
        a = 1 + rng.exponential(10.0) + 1e-3; phi = rng.uniform(0.01, 0.98); th = phi / (1 - phi); L = 1 / (2 * a)
        b = a * th * (2 * a - 1) / (a + th)
        bc = (a * th / (a + th)) * (1 - L) / L
        E0 = 2 * (L - gamma1(a)) * a / (1 + 2 * th); E1 = 2 * (L - gamma2(a, b, phi)) * (a + b) / (1 + 2 * th)
        worst = max(worst, abs(b / bc - 1), abs(E0), abs(E1))
    b90 = 10 * 9.0 * 19 / 19.0
    report("P8 the critical stay where the one-stage cycle itself stops growing is finite", worst < 1e-9 and abs(b90 - 90) < 1e-12,
           f"5,000 random cases: largest of the relative difference of the two expressions and of |E| before and after, {worst:.1e}; a = 10, phi = 0.9: {b90:.1f} steps")

    import math
    from scipy import optimize
    Phi = lambda u: 0.5 * math.erfc(-u / math.sqrt(2.0))
    def within(a, b, phi, L, sig=0.15, H=6000.0, x=math.log(10.0)):
        th = phi / (1 - phi)
        mu = (L - (gamma1(a) if b == 0 else gamma2(a, b, phi))) * sig ** 2
        W = sig ** 2 * (1 + 2 * th) / (a + b)
        return Phi((-x - mu * H) / math.sqrt(W * H)) + math.exp(-2 * mu * x / W) * Phi((-x + mu * H) / math.sqrt(W * H))
    got = [within(10, b, 0.9, 0.2) for b in (0, 10, 40, 160)] + [within(10, b, 0.0, 0.06) for b in (0, 10)]
    ref = [0.683508, 0.808598, 0.276253, 0.000908, 0.411028, 0.036237]
    root = optimize.brentq(lambda b: within(10, b, 0.9, 0.2) - got[0], 12.0, 30.0)
    report("P9 the finite-time probability regains its starting value at a shorter stay than the index does",
           max(abs(g - r) for g, r in zip(got, ref)) < 5e-4 and abs(root - 16.926) < 0.01 and root < 18.947,
           "within 6,000 generations: " + ", ".join(f"{g:.4f}" for g in got) + f"; regains the value without a steady stage at b = {root:.2f}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
