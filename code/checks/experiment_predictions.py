"""What the theory predicts for an experiment in which only the variability of a host is changed.

No existing open dataset isolates the effect the theory is about (data/INVENTORY2.md).  This script
computes, exactly and with no small-noise expansion, what the theory predicts for the simplest design
that would: replicate microcosms of a host and a parasite with a free-living (or otherwise host-free)
stage, in which host density is switched between a high and a low level on a fixed schedule, with the
SAME ARITHMETIC MEAN density in every treatment.

Model.  Discrete days.  Two stages in a ring: parasite in hosts (mean stay a days) and the free-living
stage (mean stay b days), geometric stays, threshold rates (s = 1 - 1/stay, g = 1/stay).  The transfer
from the free-living stage into hosts is multiplied by h(t) = host density / mean density:
h = 1 + delta for the first half of each period of P days and 1 - delta for the second.  One transfer
also carries exp(y).  The growth rate per day is ln(dominant eigenvalue of the product of the P daily
matrices)/P, which is exact (Floquet), and D is the y at which it is zero: the loss of ln R0, so that
the parasite invades the fluctuating treatment only if ln R0 > D, against ln R0 > 0 in the constant one.

Checks (tolerances fixed beforehand):
  X1  delta = 0 gives D = 0 (1e-10).
  X2  for small delta and a period that is not 1 or 2 days, D agrees with the second-order value for a
      periodic environment summed over the harmonics of the square wave (Result 7 of the manuscript:
      each harmonic of relative amplitude e_j contributes -(e_j^2/2) Re[g_j/(1 - g_j)] to ln R0, g_j the
      transform of the time round the cycle at that harmonic's frequency), within 2% at delta = 0.05
      for six designs.  This ties the exact computation to a result already refereed.
  X3  D <= l = -(1/2)[ln(1 + delta) + ln(1 - delta)], the bound of Proposition 21, in every case.
  Reported: D for delta in {0.5, 0.8, 0.95}, periods from a quarter of a generation to eight
  generations, and three free-living stays; the same with the two stages' stays exchanged does not
  matter (n = 2), so the table is by (a, b) only.  Also the difference in growth rate when rare between
  the constant and the fluctuating treatment at a given R0, D/T per day, and the factor by which the
  parasite's numbers differ after 60 days.

Run: python3 -I checks/experiment_predictions.py   (seconds)
"""
from __future__ import annotations

import cmath
import math
import sys

import numpy as np

FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def growth(a, b, delta, P, y):
    s = np.array([1 - 1 / a, 1 - 1 / b])            # stage 0: in hosts; stage 1: free-living
    g = np.array([1 / a, 1 / b])
    M = np.eye(2)
    for t in range(P):
        h = 1 + delta if t < P // 2 else 1 - delta
        A = np.array([[s[0], g[1] * h * math.exp(y)], [g[0], s[1]]])
        M = A @ M
    lam = max(abs(np.linalg.eigvals(M)))
    return math.log(lam) / P


def loss(a, b, delta, P):
    lo, hi = -1e-12, 3.0            # D is at most 1.2 for the largest swing used (Proposition 21)
    if growth(a, b, delta, P, 0.0) >= 0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if growth(a, b, delta, P, mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def second_order(a, b, delta, P):
    """Sum over the harmonics of the square wave of -(e_j^2/2) Re[g_j/(1 - g_j)] (discrete time)."""
    s = [1 - 1 / a, 1 - 1 / b]
    h = np.array([1 + delta if t < P // 2 else 1 - delta for t in range(P)])
    c = np.fft.fft(h - 1) / P                       # Fourier coefficients
    tot = 0.0
    for j in range(1, P):
        w = 2 * math.pi * j / P
        z = cmath.exp(1j * w)
        gj = 1.0
        for sk in s:
            gj *= (1 - sk) / (z - sk)               # transform of one geometric stay, at e^{i w}
        # a pair of conjugate coefficients c_j, c_{P-j} is a cosine of amplitude 2|c_j|; each coefficient
        # contributes |c_j|^2 Re[g/(1-g)] and the pair gives (e^2/2) Re[g/(1-g)] with e = 2|c_j|
        tot += abs(c[j]) ** 2 * (gj / (1 - gj)).real
    return -tot


def main():
    worst = max(abs(loss(a, b, 0.0, P)) for a, b, P in ((5, 5, 10), (10, 2, 8), (3, 30, 40)))
    report("X1 no fluctuation, no loss", worst < 1e-10, f"largest |D| = {worst:.1e}")

    worst = 0.0
    rows = []
    for a, b, P in ((5, 5, 10), (5, 5, 40), (10, 2, 12), (3, 30, 30), (3, 30, 120), (8, 8, 6)):
        d_exact = loss(a, b, 0.05, P)
        d2 = second_order(a, b, 0.05, P)
        worst = max(worst, abs(d_exact / d2 - 1))
        rows.append(f"({a},{b}) P={P}: exact {d_exact:.3e}, second order {d2:.3e}")
    report("X2 small swings: the exact loss equals the periodic second-order value", worst < 0.02, f"largest relative difference {worst:.4f}; " + "; ".join(rows[:2]))

    bad = 0
    print("\n      Loss D of ln R0 (the parasite invades only if ln R0 > D). Stay in hosts a = 5 days.")
    print("      rows: stay in the free-living stage b; columns: period of the host's swing as a multiple of the generation time T = a + b")
    for delta in (0.5, 0.8, 0.95):
        ell = -0.5 * (math.log(1 + delta) + math.log(1 - delta))
        print(f"      hosts switched between {1 + delta:.2f} and {1 - delta:.2f} times their mean (sd of ln density {0.5 * (math.log(1 + delta) - math.log(1 - delta)):.2f}; bound {ell:.3f})")
        print("        b \\ P/T      1/4      1/2       1        2        4        8")
        for b in (2, 10, 50):
            a = 5
            T = a + b
            cells = []
            for mult in (0.25, 0.5, 1, 2, 4, 8):
                P = max(2, int(round(mult * T / 2)) * 2)
                D = loss(a, b, delta, P)
                if D > ell + 1e-9:
                    bad += 1
                cells.append(f"{D:7.4f}")
            print(f"        {b:>3}       " + "  ".join(cells))
    report("X3 the bound of Proposition 21 holds in every design", bad == 0, f"violations {bad}")

    print("\n      What an experimenter would see: the same parasite (ln R0 = 0.5 per generation) in constant and in switched hosts,")
    print("      swings between 1.8 and 0.2 times the mean, period = 2 generations")
    print("        stay in hosts, free-living stay   growth per day, constant   growth per day, switched   ratio of parasite numbers after 60 days")
    for a, b in ((5, 2), (5, 10), (5, 50)):
        T = a + b
        P = max(2, int(round(2 * T / 2)) * 2)
        y = 0.5
        r_const = growth(a, b, 0.0, P, y)
        r_fluct = growth(a, b, 0.8, P, y)
        print(f"        ({a:>2}, {b:>2})                          {r_const:+.4f}                   {r_fluct:+.4f}                  {math.exp(60 * (r_const - r_fluct)):6.2f}")
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
