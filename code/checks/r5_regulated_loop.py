"""Local stability of a regulated life cycle that is one loop of stages (rung R5, deterministic part).

Claims checked, each against eigenvalues of the Jacobian computed here, tolerance 1e-9 unless stated:
  D1  one regulated transition, sensed in the stage it leaves, continuous time: the eigenvalues satisfy
      prod_k (1 + lam/(nu_k + mu_k)) = 1 - kappa, kappa the strength of the density dependence;
  D2  equal stages, generation time tau: the leading eigenvalue is -(n/tau)(1 - (1-kappa)^(1/n)) for kappa < 1 and
      (n/tau)((kappa-1)^(1/n) cos(pi/n) - 1) for kappa > 1; the equilibrium is unstable exactly when
      kappa > 1 + sec(pi/n)^n (never for one or two stages);
  D3  regulation in every stage at whole-cycle factor R0: return rate (n/tau)(1 - R0^(-1/n)) when the regulation
      divides the transfer by 1 + x/K, and (n/tau)(R0^(1/n) - 1) when it is an added death rate proportional to
      density; both tend to ln(R0)/tau, the first from below and the second from above;
  D4  tertiary's closed forms equal primary's independently derived functions `char_roots_C` and
      `return_rate_every_stage` (`studies/r5/r5.py`), where the two overlap.

Run: python3 -I checks/r5_regulated_loop.py
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec = importlib.util.spec_from_file_location("primary_r5", HERE.parent / "studies" / "r5" / "r5.py")
primary_r5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary_r5)


def jac_one_regulated(nu, mu, kappa):
    """Linearisation at the equilibrium, scaled so that every stage holds one unit: inflow to a stage equals its
    total loss rate, and the regulated link (out of the last stage) carries the factor 1 - kappa."""
    n = len(nu)
    d = nu + mu
    j = -np.diag(d).astype(float)
    for k in range(n):
        j[(k + 1) % n, k] += d[(k + 1) % n] * ((1 - kappa) if k == n - 1 else 1.0)
    return j


def lead_closed(n, tau, kappa):
    a = n / tau
    if kappa <= 1:
        return -a * (1 - (1 - kappa) ** (1 / n))
    return a * ((kappa - 1) ** (1 / n) * math.cos(math.pi / n) - 1)


def kappa_c(n):
    return math.inf if n <= 2 else 1 + (1 / math.cos(math.pi / n)) ** n


def lead_every_stage(n, tau, r0, form):
    rho = r0 ** (1 / n)
    d = n / tau
    if form == "transfer":          # transfer divided by 1 + x/K
        j = -d * np.eye(n) + d * (1 / rho) * np.roll(np.eye(n), 1, axis=0)
    else:                           # added death rate proportional to density
        j = -(d * (2 * rho - 1)) * np.eye(n) + d * rho * np.roll(np.eye(n), 1, axis=0)
    return float(np.linalg.eigvals(j).real.max())


def main():
    rng = np.random.default_rng(2026111405)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    worst, cnt = 0.0, 0
    for trial in range(2000):
        n = int(rng.integers(1, 9)); nu = rng.uniform(0.2, 3, n); mu = rng.uniform(0, 1, n); kappa = rng.uniform(0.05, 3.0)
        d = nu + mu
        ev = np.linalg.eigvals(jac_one_regulated(nu, mu, kappa))
        worst = max(worst, float(np.max(np.abs(np.prod(1 + ev[:, None] / d[None, :], axis=1) - (1 - kappa))))); cnt += 1
    report("D1 characteristic equation, one regulated transition", worst < 1e-9, f"{cnt} random unequal cycles of 1 to 8 stages; largest residual {worst:.1e}")

    worst, bad, cnt = 0.0, 0, 0
    for n in (1, 2, 3, 4, 8, 16, 32):
        for kappa in np.linspace(0.07, 12.03, 240):          # grid chosen to miss the exact boundaries 5 and 9
            lead = float(np.linalg.eigvals(jac_one_regulated(np.full(n, n / 2.0), np.zeros(n), kappa)).real.max())
            worst = max(worst, abs(lead - lead_closed(n, 2.0, kappa)))
            bad += int((lead > 0) != (kappa > kappa_c(n))); cnt += 1
    report("D2 equal stages: leading eigenvalue and stability boundary", worst < 1e-9 and bad == 0,
           f"{cnt} cases (1 to 32 stages, kappa from 0.07 to 12); largest difference {worst:.1e}; boundary mismatches {bad}; "
           f"kappa_c = {', '.join(f'{n}: {kappa_c(n):.3f}' for n in (3, 4, 8, 16, 32))}")

    w1 = w2 = 0.0
    for n in (1, 2, 4, 8, 32):
        for r0 in (1.5, 4.0, 15.0):
            w1 = max(w1, abs(lead_every_stage(n, 2.0, r0, "transfer") + (n / 2.0) * (1 - r0 ** (-1 / n))))
            w2 = max(w2, abs(lead_every_stage(n, 2.0, r0, "death") + (n / 2.0) * (r0 ** (1 / n) - 1)))
    a_ = [round(-lead_every_stage(n, 2.0, 15.0, "transfer"), 3) for n in (1, 2, 4, 8, 32)]
    b_ = [round(-lead_every_stage(n, 2.0, 15.0, "death"), 3) for n in (1, 2, 4, 8, 32)]
    report("D3 regulation in every stage, two forms", w1 < 1e-9 and w2 < 1e-9,
           f"15 cases each; return rates at R0 = 15, tau = 2 for 1, 2, 4, 8, 32 stages: on the transfer {a_}, as a death rate {b_}; limit ln(R0)/tau = {math.log(15) / 2:.3f}")

    worst, cnt = 0.0, 0
    for n in (1, 2, 3, 4, 8, 16):
        for kappa in (0.2, 0.7, 1.5, 3.0, 8.0):
            roots = np.asarray(primary_r5.char_roots_C(n, 2.0, kappa))
            worst = max(worst, abs(float(roots.real.max()) - lead_closed(n, 2.0, kappa))); cnt += 1
        for r0 in (1.5, 4.0, 15.0):
            val = primary_r5.return_rate_every_stage(n, 2.0, r0)
            val = float(val[0] if isinstance(val, (tuple, list, np.ndarray)) else val)
            worst = max(worst, abs(val - (n / 2.0) * (1 - r0 ** (-1 / n)))); cnt += 1
    report("D4 agreement with primary's independent functions", worst < 1e-9, f"{cnt} cases; largest difference {worst:.1e}")

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
