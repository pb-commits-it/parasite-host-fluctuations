"""Independent numerical check of the condition for a hump in the number of stages.

Claim (derived by primary on 2026-11-13, `studies/derivation.md` eq. 8): for a ring of n equal stages
with stay probability s > 0 and production gbar, the second-order cost of fluctuation has its
maximum over whole numbers of stages at some n > 1 exactly when
    phi > (s + gbar) / (3 s + gbar),
and otherwise is largest at n = 1. Also claimed: the cost is unimodal in n.

Method: evaluate the closed form as secondary wrote it (not primary's rearrangement) for n = 1..600
on a grid of s, gbar and phi, and compare the location of the maximum with the threshold.
Grid points within 1e-9 of the threshold are skipped. No simulation.
"""
import itertools
import sys

import numpy as np


def coefficient(n, phi, s, g):
    """cost / sigma^2 from secondary's expression, evaluated in logs to avoid overflow at large n."""
    lam0 = s + g
    e = g / lam0
    a, b = lam0 - phi * s, phi * g
    tail = n * lam0 / g * np.exp(n * np.log(b / a)) / (1.0 - np.exp(n * np.log(b / a))) if phi > 0 else 0.0
    return e ** 2 / n * (0.5 + phi / (1 - phi) - tail)


n = np.arange(1, 601)
checked = wrong = not_unimodal = 0
for s, g, phi in itertools.product(np.linspace(0.02, 0.98, 25), np.geomspace(0.02, 5.0, 25), np.linspace(0.01, 0.99, 99)):
    thr = (s + g) / (3 * s + g)
    if abs(phi - thr) < 1e-9:
        continue
    c = np.array([coefficient(k, phi, s, g) for k in n])
    peak = int(n[np.argmax(c)])
    checked += 1
    wrong += (peak > 1) != (phi > thr)
    d = np.sign(np.diff(c))
    d = d[np.abs(np.diff(c)) > 1e-15 * c.max()]
    not_unimodal += np.count_nonzero(np.diff(d) > 0) > 0        # a rise after a fall
print(f"{checked} parameter sets; hump condition wrong in {wrong}; not unimodal in {not_unimodal}")
sys.exit(0 if wrong == 0 and not_unimodal == 0 else 1)
