"""A periodic environment in continuous time: the reproduction number of a loop with one sinusoidal transfer.

Reproduces a published result with independent code, and tests the manuscript's statement that our loss of ln R0
reduces to it at one frequency. Bacaer (2007), as reported by secondary from the full text (verification/literature/
VERDICTS4.md): for an n-compartment cycle with exit rates b_1..b_n and one transfer a_n [1 + eps cos(w t)],
    R0 = G0 [ 1 + (eps^2/2) Re( g/(1 - g) ) + O(eps^4) ],   G0 = prod a_k / prod b_k,   g = prod b_k/(b_k + i w).
Ours (notes/random-R0.md, section 7): ln R0* - ln R0 = (eps^2/2) Re[g/(1 - g)] to second order, g the transform of
the time round the cycle at the ordinary exit rates.

Here R0* is computed directly from its definition: the R for which dx/dt = (A(t)/R - B) x has Floquet exponent
zero, A(t) the periodic transfer and B the rest. The monodromy matrix is integrated numerically (rtol 1e-11).

Claims, tolerances stated beforehand:
  E1  One stage: R0* = G0 for every eps and w (|ln R0* - ln G0| < 1e-7; 10 random cases, eps = 0.3).
  E2  Two to four stages, 30 random cases with |Re g/(1 - g)| > 0.05: the ratio of ln R0* - ln G0 to
      (eps^2/2) Re[g/(1 - g)] is within 0.02 of one at eps = 0.05 and within 0.08 of one at eps = 0.1.
  E3  The sign follows Re[g/(1 - g)]. In the 30 random cases of E2 (two to four stages) the computed
      ln R0* - ln G0 has the sign of Re[g/(1 - g)] in every case. And for eight equal stages with the period equal
      to the generation time (w = 2 pi b / 8), where Re[g/(1 - g)] = +0.074, the periodic transfer RAISES R0, with
      the ratio of E2 within 0.02 of one at eps = 0.05.
      First wording, as run at 07:05 (output kept in results/periodic-r0_attempt1_20261114.txt): "among the cases
      of E2 both signs of Re g/(1 - g) occur". That was false, and I should have known it: for two stages
      Re[g/(1 - g)] = -b1 b2 / (w^2 + (b1 + b2)^2) is negative at every frequency, and with two to four exponential
      stages the generation time is too variable for a gain. All 30 draws were losses. The agreement of sign and
      size held in every case; what failed was my expectation of what the draws would contain.

Run: python3 -I checks/periodic_r0.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp


def floquet_exponent(a, b, eps, w, R):
    """Exponent of dx_1/dt = a_n (1 + eps cos wt) x_n / R - b_1 x_1, dx_{k+1}/dt = a_k x_k - b_{k+1} x_{k+1}."""
    n = len(b); period = 2 * math.pi / w

    def rhs(t, xflat):
        X = xflat.reshape(n, n); dX = -b[:, None] * X
        if n == 1:
            dX[0] += a[0] * (1 + eps * math.cos(w * t)) / R * X[0]
        else:
            dX[0] += a[n - 1] * (1 + eps * math.cos(w * t)) / R * X[n - 1]
            dX[1:] += a[:n - 1, None] * X[:n - 1]
        return dX.ravel()

    sol = solve_ivp(rhs, (0, period), np.eye(n).ravel(), rtol=1e-11, atol=1e-13, method="DOP853")
    mu = np.max(np.abs(np.linalg.eigvals(sol.y[:, -1].reshape(n, n))))
    return math.log(mu) / period


def rstar(a, b, eps, w):
    g0 = float(np.prod(a) / np.prod(b))
    f = lambda lr: floquet_exponent(a, b, eps, w, math.exp(lr))
    return optimize.brentq(f, math.log(g0) - 1.0, math.log(g0) + 1.0, xtol=1e-12, rtol=1e-12), math.log(g0)


def main():
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    rng = np.random.default_rng(20261114)
    worst = 0.0
    for trial in range(10):
        b = rng.uniform(0.3, 3.0, 1); a = rng.uniform(0.3, 3.0, 1); w = rng.uniform(0.3, 6.0)
        lr, lg = rstar(a, b, 0.3, w); worst = max(worst, abs(lr - lg))
    report("E1 one stage: a sinusoidal transfer leaves R0 unchanged", worst < 1e-7, f"10 random cases at eps = 0.3: largest |ln R0* - ln G0| = {worst:.1e}")

    w05, w10, signs, sign_ok, used = 0.0, 0.0, set(), True, 0
    while used < 30:
        n = int(rng.integers(2, 5)); b = rng.uniform(0.3, 3.0, n); a = rng.uniform(0.3, 3.0, n); w = rng.uniform(0.3, 6.0)
        g = np.prod(b / (b + 1j * w)); k = (g / (1 - g)).real
        if abs(k) <= 0.05:
            continue
        used += 1; signs.add(k > 0)
        for eps in (0.05, 0.1):
            lr, lg = rstar(a, b, eps, w); ratio = (lr - lg) / (eps * eps / 2 * k)
            sign_ok &= (lr - lg) * k > 0
            if eps == 0.05:
                w05 = max(w05, abs(ratio - 1))
            else:
                w10 = max(w10, abs(ratio - 1))
    report("E2 two to four stages: ln R0* - ln G0 = (eps^2/2) Re[g/(1 - g)] to second order", w05 < 0.02 and w10 < 0.08,
           f"30 random cases: largest |ratio - 1| = {w05:.4f} at eps = 0.05 and {w10:.4f} at eps = 0.1")
    b8 = np.full(8, 1.0); a8 = np.full(8, 1.1); w8 = 2 * math.pi / 8
    g8 = np.prod(b8 / (b8 + 1j * w8)); k8 = (g8 / (1 - g8)).real
    lr, lg = rstar(a8, b8, 0.05, w8); ratio8 = (lr - lg) / (0.05 ** 2 / 2 * k8)
    report("E3 the sign follows Re[g/(1 - g)]: losses for two to four stages, a gain for eight stages at the matching period",
           sign_ok and signs == {False} and k8 > 0 and lr > lg and abs(ratio8 - 1) < 0.02 and abs(k8 - 0.074) < 0.001,
           f"30 random cases of two to four stages: all losses, sign agrees in every case: {sign_ok}; eight stages: Re[g/(1 - g)] = {k8:+.4f}, ln R0* - ln G0 = {lr - lg:+.3e}, ratio {ratio8:.4f}")
    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
