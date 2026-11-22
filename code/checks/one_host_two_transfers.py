"""One host series governing two transfers of a ring (secondary's caveat, board 0027). No simulation.

Transfers out of stages i and j (i != j) are multiplied by exp(kappa_1 z - ...) and exp(kappa_2 z - ...) with ONE
series z of autocorrelation phi^l. Let R = E[phi^L] for the whole loop, R_12 = E[phi^(L_12)] with L_12 the time from
a use of transfer 1 to the next use of transfer 2 (the stays of the stages after i up to and including j), and
R_21 likewise, so that R_12 R_21 = R. Claimed closed form (notes/proofs.md, Proposition 4a):

    C = (kappa_1 + kappa_2)^2 (1 + 2 theta)/(2 T^2)
        - (1/T) [ (kappa_1^2 + kappa_2^2) R + kappa_1 kappa_2 (R_12 + R_21) ] / (1 - R).

Claims, tolerances stated beforehand:
  W1  The closed form equals the published matrix formula with the two entries on one driver (theory.matrix_cost)
      on 500 random rings of 2 to 7 stages, random positions, kappa of either sign, phi in (0, 0.97)
      (relative 1e-9, measured against the larger of |C| and 1e-6).
  W2  kappa_1 = kappa_2 = 1, no memory: C = 2/T^2, four times one transfer and twice two independent hosts
      (relative 1e-10).
  W3  kappa_1 = 1, kappa_2 = -1 (the host raises one step and lowers the other in proportion): the variance term
      vanishes and C = (R_12 + R_21 - 2R)/(T (1 - R)), which is positive for phi > 0 and zero without memory
      (no exception in 20,000 random rings).
  W4  kappa_1 = kappa_2 = 1: C is at most four times the cost of one transfer (the no-memory value). How far
      below it can fall was an open question when this was written; the script prints the range found.
      Outcome of the first run (14 November, about 04:30): the ratio runs from 4 down to large negative values.
      With memory and the two transfers close together in the cycle the host's fluctuation is a GAIN, for the
      reason fluctuating survival can be: the two multipliers a line of descent meets in quick succession are
      positively correlated, and the mean of their product exceeds the product of their means (for two transfers
      one step apart, E[h_t h_(t+1)] = exp(phi sigma^2)). Example: stays (30, 1), phi = 0.9: C = -0.0234.

Run: python3 -I checks/one_host_two_transfers.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402


def closed_form(tau, i, j, k1, k2, phi):
    tau = np.asarray(tau, float); n = len(tau); T = tau.sum()
    if phi == 0:
        return (k1 + k2) ** 2 / (2 * T * T)
    theta = phi / (1 - phi)
    r = 1 / (1 + tau / theta)                       # E[phi^stay] for each stage
    R = float(np.prod(r))
    idx12 = [(i + 1 + q) % n for q in range((j - i) % n)]      # stages i+1, ..., j
    R12 = float(np.prod(r[idx12])); R21 = R / R12
    return (k1 + k2) ** 2 * (1 + 2 * theta) / (2 * T * T) - ((k1 * k1 + k2 * k2) * R + k1 * k2 * (R12 + R21)) / (T * (1 - R))


def main():
    rng = np.random.default_rng(2026111413)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst, cnt = 0.0, 0
    for trial in range(500):
        n = int(rng.integers(2, 8)); tau = 1 + rng.exponential(2.0, n) + 0.02; m = 1 - 1 / tau
        i, j = rng.choice(n, 2, replace=False); k1, k2 = rng.uniform(-2, 2, 2); phi = rng.uniform(0.0, 0.97)
        A = theory.ring_matrix(m, 1 - m)
        mat = theory.matrix_cost(A, [((i + 1) % n, int(i)), ((j + 1) % n, int(j))], [0, 0], [k1, k2], [phi])
        cf = closed_form(tau, int(i), int(j), k1, k2, phi)
        worst = max(worst, abs(cf - mat) / max(abs(mat), 1e-6)); cnt += 1
    report("W1 closed form = published matrix formula", worst < 1e-9, f"{cnt} random rings: largest relative difference {worst:.1e}")

    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(2, 8)); tau = 1 + rng.exponential(2.0, n) + 0.02; i, j = rng.choice(n, 2, replace=False)
        worst = max(worst, abs(closed_form(tau, int(i), int(j), 1, 1, 0.0) * tau.sum() ** 2 / 2 - 1))
    report("W2 same sign, no memory: 2/T^2", worst < 1e-10, f"200 random rings: largest relative difference {worst:.1e}")

    least, zero, cnt = np.inf, 0.0, 0
    for trial in range(20000):
        n = int(rng.integers(2, 8)); tau = 1 + rng.exponential(rng.choice([0.3, 3.0, 30.0]), n) + 1e-3; i, j = rng.choice(n, 2, replace=False); phi = rng.uniform(0.01, 0.99)
        least = min(least, closed_form(tau, int(i), int(j), 1, -1, phi)); zero = max(zero, abs(closed_form(tau, int(i), int(j), 1, -1, 0.0))); cnt += 1
    report("W3 opposite signs: no variance term; a cost with memory, nothing without", least > 0 and zero < 1e-15, f"{cnt} random rings: smallest cost with memory {least:.2e}; largest |cost| without {zero:.1e}")

    lo, hi, cnt = np.inf, -np.inf, 0
    for trial in range(20000):
        n = int(rng.integers(2, 8)); tau = 1 + rng.exponential(rng.choice([0.3, 3.0, 30.0]), n) + 1e-3; i, j = rng.choice(n, 2, replace=False); phi = rng.uniform(0.0, 0.99)
        m = 1 - 1 / tau
        one = float(theory.cost_transfer(m, 1 - m, 1.0, phi)[0])
        ratio = closed_form(tau, int(i), int(j), 1, 1, phi) / one
        lo, hi = min(lo, ratio), max(hi, ratio); cnt += 1
    report("W4 same sign: cost relative to one transfer", hi <= 4 + 1e-9, f"{cnt} random rings: ratio between {lo:.3f} and {hi:.3f} (4 without memory; 2 would be two independent hosts)")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
