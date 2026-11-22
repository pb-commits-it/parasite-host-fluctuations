"""Do the exact fourth-order coefficients agree with simulations made before they existed?

secondary derived the exact coefficient H_n(phi) of sigma^4 in the cost (`verification/fourth_order_correlated.py`),
and primary gave it for phi = 0 (its eq. 10). Neither was available when the simulations in
`results/R1a-*.json`, `results/R1b-*.json` and `results/ring_cost_formula_20261113.txt` were run.
This script compares them. It runs no new simulation and fits nothing to choose a coefficient.

With c = H_n / C_n (C_n the closed-form second-order coefficient), theory says
    simulated cost / closed form = 1 + c sigma^2 + (terms in sigma^4 and beyond).

Three comparisons:
  A. Cases where c sigma^2 is small enough that the next term cannot matter at the precision reached.
  B. The coefficient fitted in R1a (two terms, through zero) against the exact c.
  C. For the long rings, c held at its exact value and the next two terms fitted (2 degrees of freedom).

A is the test. B and C are consistency checks made after the data were seen.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "verification"))
from fourth_order_correlated import exact_analytical_H4  # noqa: E402

RES = (__import__('pathlib').Path(__file__).resolve().parent / '../results').resolve()
S, G = 2.0 / 3.0, 0.4
# Test set for comparison A: cases whose predicted excess c sigma^2 is at most this. Chosen when the script
# was written and before it was first run, so that the following term is far below the standard errors.
TEST_SET_MAX_EXCESS = 0.006


def exact_c(n, phi, s=S, g=G):
    out = exact_analytical_H4(n, s, g, phi)
    return out["H_total"] / out["C2"]


def primary_c_phi0(n, s=S, g=G):
    lam0 = s + g
    p, q = s / lam0, g / lam0
    h = q ** 2 / (4 * n) + q ** 3 * (n * n - 1) / (24 * p * n * n) - q ** 3 / n ** 2 + 0.75 * q ** 4 / n ** 2
    return h / (q * q / (2 * n))


def figure_rows():
    rows = []
    for line in (RES / "ring_cost_formula_20261113.txt").read_text().splitlines()[2:]:
        f = line.split()
        if len(f) == 6:
            phi, n, formula, sim, se = float(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4])
            rows.append(("figure", n, phi, S, G, 0.15, sim / formula, se / formula))
    return rows


def r1_rows():
    rows = []
    for d in json.loads((RES / "R1a-large-ring-gap_20261113.json").read_text()):
        rows.append((d["label"], d["n"], d["phi"], d["s"], d["gbar"], d["sigma"], d["ratio"], d["ratio_se"]))
    for d in json.loads((RES / "R1b-long-ring-small-noise_20261113.json").read_text()):
        rows.append(("R1b", d["n"], 0.9, S, G, d["sigma"], d["ratio"], d["ratio_se"]))
    return rows


if __name__ == "__main__":
    # The phi = 0 values from secondary's code and from primary's eq. 10 must be the same number.
    worst = max(abs(exact_c(n, 0.0) - primary_c_phi0(n)) for n in (1, 2, 4, 8, 16, 32, 64))
    worst = max(worst, abs(exact_c(4, 0.0, 0.05, S + G - 0.05) - primary_c_phi0(4, 0.05, S + G - 0.05)))
    print(f"secondary's exact c at phi = 0 against primary eq. 10, 8 cases: largest difference {worst:.1e}\n")

    rows = figure_rows() + r1_rows()
    print("A. Every saved case, smallest predicted excess first. 'gap' = (simulated - predicted) / s.e.")
    print("   label                 n   phi    s      sigma   c exact   1 + c sigma^2   simulated (s.e.)    gap")
    table = []
    for label, n, phi, s, g, sigma, ratio, se in rows:
        c = exact_c(n, phi, s, g)
        pred = 1 + c * sigma ** 2
        table.append((abs(c) * sigma ** 2, label, n, phi, s, sigma, c, pred, ratio, se, (ratio - pred) / se))
    table.sort()
    small = []
    for ex, label, n, phi, s, sigma, c, pred, ratio, se, z in table:
        flag = ""
        if ex <= TEST_SET_MAX_EXCESS:
            small.append(z)
            flag = "  <- test set"
        print(f"   {label:<20s} {n:3d}  {phi:3.1f}  {s:5.3f}  {sigma:6.3f}  {c:8.3f}   {pred:10.5f}    "
              f"{ratio:8.5f} ({se:.5f})  {z:+6.2f}{flag}")
    small = np.array(small)
    print(f"\n   Test set (predicted excess at most 0.6%): {len(small)} cases; "
          f"{int((abs(small) <= 2).sum())} within 2 s.e., {int((abs(small) <= 3).sum())} within 3 s.e.; "
          f"largest |gap| {abs(small).max():.2f}; sum of squares {float((small ** 2).sum()):.1f}\n")

    print("B. Coefficient fitted in R1a (cost ratio - 1 = c sigma^2 + d sigma^4, no constant) against the exact c")
    groups = {}
    for label, n, phi, s, g, sigma, ratio, se in r1_rows():
        if label in ("main", "H4 little staying"):
            groups.setdefault((n, phi, s, g), []).append((sigma, ratio - 1, se))
    for (n, phi, s, g), pts in sorted(groups.items()):
        pts.sort()
        x = np.array([p[0] ** 2 for p in pts]); y = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        A = np.stack([x, x ** 2], axis=1) / e[:, None]
        cov = np.linalg.inv(A.T @ A)
        cfit = (cov @ A.T @ (y / e))[0]
        c = exact_c(n, phi, s, g)
        print(f"   n = {n:2d}, phi = {phi}, s = {s:.3f}: fitted {cfit:6.2f} ± {cov[0, 0] ** 0.5:.2f}; exact {c:6.2f}; "
              f"fitted - exact = {(cfit - c) / cov[0, 0] ** 0.5:+.1f} s.e.")

    print("\nC. Exact c held fixed; cost ratio - 1 - c sigma^2 = d sigma^4 + e sigma^6 fitted (4 points, 2 d.f.)")
    for (n, phi, s, g), pts in sorted(groups.items()):
        x = np.array([p[0] ** 2 for p in pts]); y = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        c = exact_c(n, phi, s, g)
        A = np.stack([x ** 2, x ** 3], axis=1) / e[:, None]
        coef, *_ = np.linalg.lstsq(A, (y - c * x) / e, rcond=None)
        chi2 = float((((y - c * x) / e - A @ coef) ** 2).sum())
        print(f"   n = {n:2d}, phi = {phi}, s = {s:.3f}: d = {coef[0]:9.1f}, e = {coef[1]:10.0f}, chi-square {chi2:.1f}")
