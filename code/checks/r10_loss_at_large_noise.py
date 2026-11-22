"""Run R10 (protocols/R10-loss-at-large-noise.md): the loss D = ln R0 - ln R0* of a ring at its
threshold, computed without the small-noise expansion, against the second-order value Gamma_0.

Claims tested (tolerances fixed in the protocol):
  H1  the simulation reproduces (a) the exact one-stage D, (b) D = sum sigma^2/2 with no staying,
      (c) Gamma_0 within 5% (or 3 standard errors if larger) at sigma = 0.2.
  H2  D <= sum sigma_k^2 / 2 in every case, and D >= 0 when phi = 0 (3 standard errors).
  H3 to H5 are open questions; the script reports numbers and no verdict.

Run: python3 -I checks/r10_loss_at_large_noise.py   (about ten minutes on eight processes)
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.optimize import brentq

M, T, BURN = 200, 200_000, 2_000


def gamma0(stays, sig, phi):
    """Second-order loss of ln R0: sum_k sigma_k^2 [(1 + 2 theta_k)/(2 T0) - R_k/(1 - R_k)]."""
    tau = np.asarray(stays, float)
    T0 = tau.sum()
    tot = 0.0
    for s2, ph in zip(np.asarray(sig, float) ** 2, phi):
        if s2 == 0:
            continue
        if ph == 0:
            tot += s2 / (2 * T0)
        else:
            th = ph / (1 - ph)
            R = float(np.prod(1 / (1 + tau / th)))
            tot += s2 * ((1 + 2 * th) / (2 * T0) - R / (1 - R))
    return tot


def one_stage_exact(s, sigma):
    x = np.linspace(-14, 14, 200001)
    w = np.exp(-x * x / 2)
    w /= w.sum()
    f = lambda D: float(np.sum(w * np.log(s + (1 - s) * np.exp(D + sigma * x - sigma * sigma / 2))))
    return brentq(f, -1e-9, sigma * sigma / 2 + 1e-9, xtol=1e-13)


def rates(stays, sig, phi, y, seed):
    """Growth rate per step of each of M populations, with exp(y) on the transfer out of stage 0."""
    tau = np.asarray(stays, float)
    n = len(tau)
    s = 1 - 1 / tau
    g = 1 / tau
    g = g.copy()
    g[0] *= math.exp(y)
    sig = np.asarray(sig, float)
    phi = np.asarray(phi, float)
    fl = np.nonzero(sig > 0)[0]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((M, len(fl)))
    innov = np.sqrt(1 - phi[fl] ** 2)
    X = np.full((M, n), 1.0 / n)
    if np.all(tau == 1):
        # no staying: the n cohorts never mix, and the log of their sum converges like 1/sqrt(T);
        # start in one stage so that a single cohort is followed (protocol, amendment of 09:34)
        X = np.zeros((M, n))
        X[:, 0] = 1.0
    acc = np.zeros(M)
    h = np.ones((M, n))
    for t in range(BURN + T):
        z = phi[fl] * z + innov * rng.standard_normal((M, len(fl)))
        h[:, fl] = np.exp(sig[fl] * z - sig[fl] ** 2 / 2)
        out = g * h * X
        Xn = s * X + np.roll(out, 1, axis=1)
        tot = Xn.sum(axis=1)
        if t >= BURN:
            acc += np.log(tot)
        X = Xn / tot[:, None]
    return acc / T


def solve_case(case):
    label, stays, sig, phi, seed = case
    t0 = time.time()
    g0 = gamma0(stays, sig, phi) if not all(t == 1 for t in stays) else float(np.sum(np.square(sig)) / 2)
    y_prev = g0
    r_prev = rates(stays, sig, phi, y_prev, seed)
    y = g0 * 1.5 + 0.005
    slope = None
    for _ in range(14):
        r = rates(stays, sig, phi, y, seed)
        slope = (r.mean() - r_prev.mean()) / (y - y_prev)
        y_new = y - r.mean() / slope
        y_prev, r_prev = y, r
        if abs(y_new - y) < 1e-6:
            y = y_new
            break
        y = y_new
    se = float(r_prev.std(ddof=1) / math.sqrt(M) / slope)
    return label, stays, sig, phi, float(y), se, g0, time.time() - t0


def build_cases():
    cases = []
    seed = 20261114
    def add(label, stays, sig, phi):
        nonlocal seed
        seed += 1
        cases.append((label, list(stays), list(sig), list(phi), seed))
    # H1a
    for s in (0.5, 0.9, 0.96):
        for sigma in (0.6, 1.0, 1.4):
            for ph in (0.0, 0.75):
                add(f"H1a s={s} sigma={sigma} phi={ph}", [1 / (1 - s)], [sigma], [ph])
    # H1b
    for ph in (0.0, 0.75):
        add(f"H1b no staying phi={ph}", [1, 1, 1], [1.0, 1.0, 1.0], [ph, ph, ph])
    # H1c
    add("H1c (1,24) sigma=0.2 phi=0.4", [1, 24], [0.2, 0], [0.4, 0])
    add("H1c (1,1.25,2) sigma=0.2 phi=0.4", [1.25, 2, 1], [0.2, 0, 0], [0.4, 0, 0])
    # H3 scan.  The fluctuating transfer is the one out of stage 0, so the stage lists are rotated to put it first.
    for name, stays in (("schistosome (1,24)", [1, 24]), ("malaria (1,7.5)", [1, 7.5]), ("Borrelia (1,2.273)", [1, 10 / 4.4]),
                        ("Echinococcus (1,1.25,2), egg", [1.25, 2, 1])):
        for ph in (0.0, 0.4, 0.75):
            for sigma in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0):
                add(f"H3 {name} phi={ph} sigma={sigma}", stays, [sigma] + [0] * (len(stays) - 1), [ph] + [0] * (len(stays) - 1))
    # H4 real sets (secondary's parameterisation, SYSTEMS.md Table A)
    add("H4 1a S. haematobium", [1, 24], [0.589, 0], [0.398, 0])
    add("H4 1b S. mansoni", [1, 24], [1.380, 0], [0.367, 0])
    add("H4 2a E. multilocularis, transfer out of the egg", [1.25, 2, 1], [1.174, 0, 0], [0.75, 0, 0])
    add("H4 2a E. multilocularis, transfer out of the fox", [1, 1.25, 2], [1.174, 0, 0], [0.75, 0, 0])
    add("H4 2a E. multilocularis, transfer out of the vole", [2, 1, 1.25], [1.174, 0, 0], [0.75, 0, 0])
    add("H4 2b E. multilocularis, fox and vole", [1.25, 2, 1], [1.174, 0, 0.18], [0.75, 0, 0.50])
    add("H4 3a B. burgdorferi, oak forest", [1, 10 / 4.4], [0.668, 0], [0.30, 0])
    add("H4 3b B. burgdorferi, northern edge", [1, 10 / 4.4], [1.190, 0], [0.50, 0])
    add("H4 5 T. tenuis in red grouse", [1, 12], [0.70, 0], [0.85, 0])
    add("H4 6 P. falciparum", [1, 7.5], [1.40, 0], [0.75, 0])
    add("H4 7 T. circumcincta in Soay sheep", [1, 3], [0.35, 0], [0.80, 0])
    # H5 parking
    for sigma in (0.3, 0.6, 1.0, 1.4):
        for b in (2, 5, 19, 50, 200):
            add(f"H5 a=10 b={b} sigma={sigma}", [10, b], [sigma, 0], [0.9, 0])
    return cases


def main():
    t_start = time.time()
    cases = build_cases()
    print(f"R10: {len(cases)} cases, M = {M} populations, T = {T} steps, burn-in {BURN}")
    with mp.Pool(8) as pool:
        res = pool.map(solve_case, cases, chunksize=1)
    by = {r[0]: r for r in res}
    fails = []

    print("\nH1a  one stage: simulation against the exact value")
    for label, stays, sig, phi, D, se, g0, dt in res:
        if label.startswith("H1a"):
            s = 1 - 1 / stays[0]
            ex = one_stage_exact(s, sig[0])
            ok = abs(D - ex) <= max(3 * se, 0.02 * ex)
            if not ok:
                fails.append(label)
            print(f"  {label:34s} D = {D:.5f} +- {se:.5f}   exact {ex:.5f}   second order {g0:.5f}   [{'ok' if ok else 'FAIL'}]")
    print("\nH1b  three stages, no staying: D = sum sigma^2 / 2 = 1.5")
    for label, stays, sig, phi, D, se, g0, dt in res:
        if label.startswith("H1b"):
            ok = abs(D - 1.5) <= max(3 * se, 0.03)
            if not ok:
                fails.append(label)
            print(f"  {label:34s} D = {D:.5f} +- {se:.5f}   [{'ok' if ok else 'FAIL'}]")
    print("\nH1c  small noise: within 5% of the second-order value, or 3 standard errors if larger")
    for label, stays, sig, phi, D, se, g0, dt in res:
        if label.startswith("H1c"):
            ok = abs(D - g0) <= max(0.05 * g0, 3 * se)
            if not ok:
                fails.append(label)
            print(f"  {label:34s} D = {D:.6f} +- {se:.6f}   second order {g0:.6f}   ratio {D / g0:.3f}   [{'ok' if ok else 'FAIL'}]")

    print("\nH2  bounds: D <= sum sigma^2/2 always; D >= 0 without memory")
    n_hi = n_lo = 0
    worst_hi = -math.inf
    for label, stays, sig, phi, D, se, g0, dt in res:
        cap = float(np.sum(np.square(sig)) / 2)
        worst_hi = max(worst_hi, (D - cap) / se)
        if D > cap + 3 * se:
            n_hi += 1
            fails.append("H2 upper: " + label)
        if all(p == 0 for p in phi) and D < -3 * se:
            n_lo += 1
            fails.append("H2 lower: " + label)
    print(f"  {len(res)} cases: above the upper bound {n_hi}; below zero without memory {n_lo}; largest (D - bound)/se = {worst_hi:.1f}")

    print("\nH3  the loss against the second-order value as sigma rises (D, and D / Gamma_0)")
    for name in ("schistosome (1,24)", "malaria (1,7.5)", "Borrelia (1,2.273)", "Echinococcus (1,1.25,2), egg"):
        for ph in (0.0, 0.4, 0.75):
            row = []
            for sigma in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0):
                _, _, _, _, D, se, g0, _ = by[f"H3 {name} phi={ph} sigma={sigma}"]
                row.append(f"{D:.4f} ({D / g0:.2f})")
            print(f"  {name:30s} phi={ph:<4}: " + "  ".join(row))
    print("  columns: sigma = 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0")

    print("\nH4  the parameter sets of the real systems")
    for label, stays, sig, phi, D, se, g0, dt in res:
        if label.startswith("H4"):
            cap = float(np.sum(np.square(sig)) / 2)
            print(f"  {label[3:]:48s} D = {D:.4f} +- {se:.4f}   second order {g0:.4f}   ratio {D / g0:.2f}   no-staying value {cap:.3f}")

    print("\nH5  a steady stage of stay b added to a fluctuating stage of stay 10 (memory 0.9)")
    for sigma in (0.3, 0.6, 1.0, 1.4):
        ex = one_stage_exact(0.9, sigma)
        g_one = sigma * sigma / 20
        row = []
        for b in (2, 5, 19, 50, 200):
            _, stays, sig, phi, D, se, g0, _ = by[f"H5 a=10 b={b} sigma={sigma}"]
            row.append(f"b={b}: {D:.4f}+-{se:.4f} (2nd {g0:.4f})")
        print(f"  sigma={sigma}: one stage exact {ex:.4f} (2nd {g_one:.4f});  " + ";  ".join(row))

    print(f"\nwall time {time.time() - t_start:.0f} s; slowest case {max(r[7] for r in res):.0f} s")
    print("H1 and H2: ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
