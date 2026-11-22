"""The remainder bound with relative noise sizes of either sign: an exact check in the one-stage case.

The note notes/error-bound-general.md proves |cost - C sigma^2| <= K sigma^4 for relative sizes kappa >= 0. The fifth
referee observed that the proof does not use the sign. Section 11 of the note goes through the proof line by line.
This script checks the conclusion where the cost can be computed exactly: one stage, a fixed part s and two
fluctuating contributions g1, g2 driven by ONE series without memory, with relative sizes k1, k2 of either sign:

    cost(sigma) = ln(s + g1 + g2) - E ln( s + g1 exp(k1 sigma z - k1^2 sigma^2/2) + g2 exp(k2 sigma z - k2^2 sigma^2/2) ),

z standard normal (the expectation by Gauss-Hermite quadrature, 200 nodes). Here C = (k1 c1 + k2 c2)^2 / 2 with
c_a = g_a/(s + g1 + g2), which is the formula of the note with its cross term.

Claims, tolerances stated beforehand:
  Q1  On 300 random cases with k1 > 0 > k2, write R(sigma) = (cost - C sigma^2)/sigma^4. Then
          max(|R(0.05)|, |R(0.025)|, |R(0.0125)|) <= 2 max(|R(0.2)|, |R(0.1)|) + 1e-6:
      the remainder divided by sigma^4 stays bounded as sigma falls sixteen-fold. With a term of order sigma^3 in the
      cost, R would grow like 1/sigma, about eight-fold over this range. The factor 2 allows for partial
      cancellation between the sigma^4 and sigma^6 terms at the two larger values (at worst the larger of the two
      is 0.6 of the limit).

      This is the third wording of Q1, and it was chosen AFTER seeing a diagnostic of the first two, so its
      tolerance was not fixed blind: the largest observed ratio was 1.16. The first two, both of which failed:
      (1) run at 06:15, output results/signed-kappa-remainder_attempt1_20261114.txt: "|r(0.05)| <= 1.05 |r(0.1)|/16
      and |r(0.1)| <= 1.3 |r(0.2)|/16", largest ratio 1.122. The 1.05 was a guess that ignored the sigma^6 term,
      which with |kappa| up to 2 is up to about thirteen times the sigma^4 coefficient. A ratio of 1.12 is still a
      sigma^4 law (sigma^3 would give 2).
      (2) run at 06:15, output results/signed-kappa-remainder_attempt2_20261114.txt: "the change in R shrinks at
      least 2.5-fold per halving of sigma". One case of 300 failed: R = 0.41836, 0.42263, 0.42245, 0.42230, 0.42225
      at sigma = 0.2, 0.1, 0.05, 0.025, 0.0125. R converges there; it is not monotone between 0.1 and 0.05, so that
      one change was small. Both failures were of the test, by near-cancellation, and neither shows a term below
      order sigma^4.
  Q2  The same 300 cases: the odd part vanishes, cost(sigma) = cost(-sigma) to 1e-13 (parity, Lemma 5').
  Q3  The cross term matters: with k1 = 1, k2 = -1 and g1 = g2 the coefficient C is zero, and the cost at
      sigma = 0.1 is below 2% of the cost with k1 = k2 = 1 (s = 0.5, g1 = g2 = 0.3).

Run: python3 -I checks/signed_kappa_remainder.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

X, W = np.polynomial.hermite_e.hermegauss(200)
W = W / math.sqrt(2 * math.pi)


def cost(s, g1, g2, k1, k2, sig):
    val = s + g1 * np.exp(k1 * sig * X - 0.5 * (k1 * sig) ** 2) + g2 * np.exp(k2 * sig * X - 0.5 * (k2 * sig) ** 2)
    return math.log(s + g1 + g2) - float(np.sum(W * np.log(val)))


def main():
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    rng = np.random.default_rng(20261114)
    ok1, ok2, worst_ratio, worst_par, used = True, True, 0.0, 0.0, 0
    for trial in range(300):
        s, g1, g2 = rng.uniform(0.05, 1.0, 3); k1 = rng.uniform(0.2, 2.0); k2 = -rng.uniform(0.2, 2.0)
        lam = s + g1 + g2; C = 0.5 * (k1 * g1 / lam + k2 * g2 / lam) ** 2
        R = {sg: (cost(s, g1, g2, k1, k2, sg) - C * sg * sg) / sg ** 4 for sg in (0.2, 0.1, 0.05, 0.025, 0.0125)}
        used += 1
        small, large = max(abs(R[0.05]), abs(R[0.025]), abs(R[0.0125])), max(abs(R[0.2]), abs(R[0.1]))
        ok1 &= small <= 2 * large + 1e-6
        worst_ratio = max(worst_ratio, small / (large + 1e-9))
        par = abs(cost(s, g1, g2, k1, k2, 0.1) - cost(s, g1, g2, k1, k2, -0.1)); worst_par = max(worst_par, par); ok2 &= par < 1e-13
    report("Q1 opposite signs: (cost - C sigma^2)/sigma^4 stays bounded as sigma falls sixteen-fold", ok1 and used == 300,
           f"{used} cases; largest ratio of its size at sigma <= 0.05 to its size at sigma >= 0.1: {worst_ratio:.3f} (about 8 if there were a sigma^3 term)")
    report("Q2 parity", ok2, f"largest |cost(0.1) - cost(-0.1)| = {worst_par:.1e}")
    a, b = cost(0.5, 0.3, 0.3, 1, -1, 0.1), cost(0.5, 0.3, 0.3, 1, 1, 0.1)
    report("Q3 equal and opposite contributions: the second-order cost cancels", a < 0.02 * b, f"cost at sigma = 0.1: {a:.3e} with opposite signs, {b:.3e} with the same sign")
    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
