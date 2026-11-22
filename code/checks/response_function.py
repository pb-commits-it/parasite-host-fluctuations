"""The cost of host fluctuation at each frequency, and hosts that cycle. No simulation (run R7 simulates).

For one fluctuating transfer of a primitive ring, define the response function
    K(w) = -(1/T) Re[ g(w) / (1 - g(w)) ],    g(w) = E[exp(i w L)] = prod_k (1 - m_k) e^{iw} / (1 - m_k e^{iw}),
so that (notes/continuous-time.md, section 5) cost / sigma^2 = (1/pi) * integral over (0, pi) of f(w) K(w) dw for a
host with normalised spectral density f.

Claims, tolerances stated beforehand:
  K1  One stage: K is the same at every frequency, 1/(2 tau^2) (relative 1e-10 on 2,000 random cases).
  K2  The average of K over (0, pi) is 1/(2 T^2), the cost of a host without memory (relative 1e-6, 300 rings).
  K3  K(w) -> (1 - CV^2)/(2T) as w -> 0, CV the coefficient of variation of L (relative 1e-4 at w = 1e-4, 300 rings),
      and for two or more stages this exceeds the average 1/(2 T^2) (no exception).
  K4  A host that is a stationary AR(2) quasi-cycle, autocorrelation Re[B z^l], z = r e^{i w0}: the closed form
          S0/(2 T^2) - (1/T) Re[ B R(z)/(1 - R(z)) ],  S0 = 1 + 2 Re[B z/(1 - z)],  R(z) = prod (1 - m_k) z/(1 - m_k z)
      equals (a) the direct sum over lags of the lineage form (relative 1e-9; rings of 2 to 6 stages) and
      (b) the integral of the AR(2) spectral density against K (relative 1e-6).
  K5  n equal stages with generation time T equal to the period P of a strongly cyclic host (r = 0.9, P = 12): the
      cost falls as n rises at fixed T and is negative for n = 8 (the generation becomes regular); with T = P/2 it
      is at least three times the cost of an uncorrelated host for n = 2 and n = 4 (as printed in protocol R7).

Run: python3 -I checks/response_function.py
"""
from __future__ import annotations

import cmath
import math
import pathlib
import sys

import numpy as np
from scipy import integrate

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402


def response(stays, w):
    """K(w) for a ring with the given lineage stays (lam = 1)."""
    tau = np.asarray(stays, float); m = 1 - 1 / tau; T = tau.sum()
    e = cmath.exp(1j * w)
    g = np.prod((1 - m) * e / (1 - m * e))
    return float(-(g / (1 - g)).real / T)


def ar2(period, r):
    w0 = 2 * math.pi / period
    a1, a2 = 2 * r * math.cos(w0), -r * r
    rho1 = a1 / (1 - a2)
    B = 1 + 1j * (math.cos(w0) - rho1 / r) / math.sin(w0)
    return a1, a2, w0, rho1, B


def cost_cyclic(stays, period, r):
    tau = np.asarray(stays, float); m = 1 - 1 / tau; T = tau.sum(); c = 1 / T
    a1, a2, w0, rho1, B = ar2(period, r)
    z = r * cmath.exp(1j * w0)
    R = np.prod((1 - m) * z / (1 - m * z))
    return float(c * c / 2 + c * c * (B * z / (1 - z)).real - c * (B * R / (1 - R)).real)


def main():
    rng = np.random.default_rng(2026111411)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst = 0.0
    for trial in range(2000):
        tau = 1 + rng.exponential(3.0) + 1e-3; w = rng.uniform(0.01, math.pi)
        worst = max(worst, abs(response([tau], w) * 2 * tau * tau - 1))
    report("K1 one stage: the same cost at every frequency", worst < 1e-10, f"2,000 random cases: largest relative departure from 1/(2 tau^2) is {worst:.1e}")

    w2, w3, least, cnt = 0.0, 0.0, np.inf, 0
    for trial in range(300):
        n = int(rng.integers(1, 7)); tau = 1 + rng.exponential(2.0, n) + 0.05; T = tau.sum()
        avg = integrate.quad(lambda w: response(tau, w), 1e-10, math.pi, limit=400)[0] / math.pi
        w2 = max(w2, abs(avg * 2 * T * T - 1))
        cv2 = float(np.sum(tau * (tau - 1))) / T ** 2
        k0 = (1 - cv2) / (2 * T)
        w3 = max(w3, abs(response(tau, 1e-4) / k0 - 1))
        if n >= 2:
            least = min(least, k0 * 2 * T * T - 1)
        cnt += 1
    report("K2 the average of K over frequency is 1/(2 T^2)", w2 < 1e-6, f"{cnt} random rings: largest relative difference {w2:.1e}")
    report("K3 at low frequency K is (1 - CV^2)/(2T), above the average for two or more stages", w3 < 1e-4 and least > 0, f"largest relative difference at w = 1e-4: {w3:.1e}; smallest excess of K(0) over the average, as a share of it: {least:.2e}")

    wa, wb, cnt = 0.0, 0.0, 0
    for trial in range(300):
        n = int(rng.integers(2, 7)); tau = 1 + rng.exponential(2.0, n) + 0.05; P = rng.uniform(2.5, 30); r = rng.uniform(0.3, 0.95)
        m = 1 - 1 / tau; A = theory.ring_matrix(m, 1 - m)
        a1, a2, w0, rho1, B = ar2(P, r)
        cache = [1.0, rho1]

        def rho(l):
            while len(cache) <= l:
                cache.append(a1 * cache[-1] + a2 * cache[-2])
            return cache[l]
        cf = cost_cyclic(tau, P, r)
        lin = theory.lineage_cost(A, (1, 0), rho, lags=20000)
        wa = max(wa, abs(lin - cf) / max(abs(cf), 1e-9))
        if trial < 100:
            innov = 1 - a1 * rho1 - a2 * (a1 * rho1 + a2)
            f = lambda w: innov / abs(1 - a1 * cmath.exp(-1j * w) - a2 * cmath.exp(-2j * w)) ** 2
            val = integrate.quad(lambda w: f(w) * response(tau, w), 1e-10, math.pi, limit=800, points=[w0])[0] / math.pi
            wb = max(wb, abs(val - cf) / max(abs(cf), 1e-4))
        cnt += 1
    report("K4 a cyclic (AR(2)) host: closed form = sum over lags = spectrum times K", wa < 1e-9 and wb < 1e-6, f"{cnt} random rings against the lag sum: {wa:.1e}; 100 against the spectral integral: {wb:.1e}")

    g12 = {n: 12 * cost_cyclic([12.0 / n] * n, 12.0, 0.9) for n in (2, 3, 4, 6, 8)}
    g6 = {n: 6 * cost_cyclic([6.0 / n] * n, 12.0, 0.9) for n in (2, 4)}
    ok = all(g12[a] > g12[b] for a, b in ((3, 4), (4, 6), (6, 8))) and g12[8] < 0 and all(v > 3 * (1 / 12) for v in g6.values())
    report("K5 resonance: a regular generation equal to the period gains; half the period costs at least three times the uncorrelated value", ok,
           "T = P = 12: " + ", ".join(f"n={n}: {v:+.4f}" for n, v in g12.items()) + f" (uncorrelated 0.0417); T = P/2: " + ", ".join(f"n={n}: {v:+.4f}" for n, v in g6.items()) + " (uncorrelated 0.0833)")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
