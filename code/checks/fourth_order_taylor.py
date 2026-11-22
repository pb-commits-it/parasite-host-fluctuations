"""The fourth-order coefficient of the cost, estimated without any algebra of ours.

Claim checked: secondary's exact coefficient H_n(phi) (`verification/fourth_order_correlated.py`,
`exact_analytical_H4`) equals minus one twenty-fourth of the long-run mean of the fourth derivative of
ln G_t with respect to sigma at sigma = 0.

Method. The exact recursion is carried through with Taylor arithmetic in sigma about sigma = 0, along
simulated noise paths. The fourth Taylor coefficient of ln G_t, averaged over time and paths after a
burn-in, estimates -H_n directly: there is no finite noise, so no bias from higher terms, and nothing
is fitted. The Taylor arithmetic is the one tested in `checks/error_bound_steps.py` (test T6c compares
it with finite differences).

Known answers first (rule 7), tolerance 3 standard errors, stated before the run:
  - one stage, any phi: H_1 = q^2 p (1 - 3q) / 4 (expansion of a one-dimensional integral);
  - phi = 0, any n: primary eq. (10).
Then the comparison with secondary's values for phi = 0.9 and 0.5, same tolerance, and at most one case in
twenty outside 2 standard errors expected by chance.

Run: python3 -I checks/fourth_order_taylor.py
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec = importlib.util.spec_from_file_location("ebs", HERE / "error_bound_steps.py")
ebs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebs)
spec_c = importlib.util.spec_from_file_location("secondary_h4", HERE.parent / "verification" / "fourth_order_correlated.py")
secondary_h4 = importlib.util.module_from_spec(spec_c)
spec_c.loader.exec_module(secondary_h4)


def h_phi0(n, s, gbar):
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    return q * q / (4 * n) + q ** 3 * (n * n - 1) / (24 * p * n * n) - q ** 3 / n ** 2 + 3 * q ** 4 / (4 * n * n)


def estimate(n, s, gbar, phi, reps, steps, rng):
    """Mean over time and paths of -(fourth Taylor coefficient of ln G_t) at sigma = 0, with its standard error
    from the spread of the per-path time averages."""
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    beta = math.sqrt(max(0.0, 1 - 4 * p * q * math.sin(math.pi / n) ** 2)) if n > 1 else 0.0
    burn = int(math.ceil(max(30.0 / (1.0 - beta), 30.0 / (1.0 - phi) if phi > 0 else 0.0, 50)))
    xi = rng.normal(size=(reps, n))
    c = math.sqrt(1 - phi * phi)
    w = np.zeros((reps, n, ebs.ORDER + 1))
    w[:, :, 0] = 1.0 / n
    acc = np.zeros(reps)
    for t in range(burn + steps):
        h = ebs.h_jet(xi, 0.0)
        new = s * w + np.roll(gbar * ebs.jmul(h, w), 1, axis=1)
        g = new.sum(axis=1)
        if t >= burn:
            acc -= ebs.jlog(g)[:, 4]
        w = ebs.jmul(new, ebs.jrecip(g)[:, None, :])
        xi = phi * xi + c * rng.normal(size=(reps, n))
    per_path = acc / steps
    return float(per_path.mean()), float(per_path.std(ddof=1) / math.sqrt(reps)), burn


def main():
    rng = np.random.default_rng(2026111402)
    s, gbar = 2 / 3, 0.4
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    rows = []
    print("Known answers")
    for (n, phi, reps, steps) in ((1, 0.0, 4000, 4000), (1, 0.9, 4000, 20000), (4, 0.0, 4000, 4000), (16, 0.0, 4000, 4000), (32, 0.0, 4000, 6000)):
        est, se, burn = estimate(n, s, gbar, phi, reps, steps, rng)
        exact = q * q * p * (1 - 3 * q) / 4 if n == 1 else h_phi0(n, s, gbar)
        z = (est - exact) / se
        rows.append(("known", n, phi, est, se, exact, z))
        print(f"  n={n:3d} phi={phi}: estimate {est:+.6f} +- {se:.6f}; exact {exact:+.6f}; z = {z:+.2f}  ({reps} paths x {steps} steps after {burn})")
    print("Against secondary's exact coefficient")
    for (n, phi, reps, steps) in ((2, 0.9, 4000, 20000), (4, 0.9, 4000, 20000), (8, 0.9, 4000, 20000), (16, 0.9, 4000, 20000),
                                  (32, 0.9, 4000, 20000), (64, 0.9, 2000, 30000), (4, 0.5, 4000, 8000), (16, 0.5, 4000, 8000)):
        est, se, burn = estimate(n, s, gbar, phi, reps, steps, rng)
        exact = float(secondary_h4.exact_analytical_H4(n, s, gbar, phi)["H_total"])
        z = (est - exact) / se
        rows.append(("verification", n, phi, est, se, exact, z))
        print(f"  n={n:3d} phi={phi}: estimate {est:+.6f} +- {se:.6f}; secondary {exact:+.6f}; z = {z:+.2f}  ({reps} paths x {steps} steps after {burn})")
    zs = np.array([r[6] for r in rows])
    known_ok = all(abs(r[6]) < 3 for r in rows if r[0] == "known")
    secondary_ok = all(abs(r[6]) < 3 for r in rows if r[0] == "verification")
    print()
    print(f"known answers within 3 standard errors: {known_ok}")
    print(f"secondary's values within 3 standard errors: {secondary_ok}; cases beyond 2: {int((np.abs(zs) > 2).sum())} of {len(zs)}; sum of z^2 = {float((zs ** 2).sum()):.1f} on {len(zs)} cases")
    return 0 if (known_ok and secondary_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
