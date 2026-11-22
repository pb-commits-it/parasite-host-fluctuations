"""Run R15a, two descriptive blocks added after the full run (protocol, amendment 12).  No simulation.

T8b  Where the fluctuation part of the temperature effect (against the cycle with every rate at its arithmetic
     mean) is a loss and where it is a gain, as the mean temperature varies: second order, formula (c), and
     exactly (Floquet) for the seasonal case, with an exact value on each side of every sign change.
I3b  The steady stays at which a steady stage lowers the second-order loss below its value with no steady
     stage, by the shape of the stay (geometric, chains of 2, 4, 16 sub-stages, fixed delay).

Neither block is a pass/fail check.  The output is appended to the results file of the main script.

Run: python3 -I checks/r15a_extra.py >> results/r15a-temperature-incubation_20261114.txt
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import r15a_lib as L
import r15a_temperature_incubation as R


def main():
    t0 = time.time()
    print("\n" + "=" * 110)
    print("ADDED AFTER THE FULL RUN (protocol, amendment 12): checks/r15a_extra.py, exact computations only, no pass/fail")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (system clock)")
    print("=" * 110)
    model = L.make_model(("TPC", "full", None))
    fine = np.arange(10.2, 34.81, 0.1)
    grid = R.T8_GRID
    print("T8b sign of the fluctuation part against baseline (ii), model TPC.  D > 0 is a loss, D < 0 a gain.")
    print("      Mean temperatures (C) at which D > 0:")
    for phi in (0.0, 0.8, 0.99):
        f = lambda T: R.second_order_small(model, T, phi)[1]
        roots, pos = R.roots_and_sign(f, fine, 1e-8)
        print(f"      second order (small variance), phi = {phi:<4}: {R.show_pos(pos)}" + ("   (zero at " + ", ".join(f"{r:.2f}" for r in roots) + ")" if roots else ""))
    f = lambda T: L.season_formulas(model, T, 1.0)["tangent"]
    roots, pos = R.roots_and_sign(f, fine, 1e-8)
    print(f"      second order (small amplitude), seasonal, 365 d: {R.show_pos(pos)}")
    dcs = {0.8: [], 0.99: []}
    for T in grid:
        F = L.Formulas(model, float(T), 5.0)
        for phi in dcs:
            dcs[phi].append(F.hermite(phi)[1])
    for phi in (0.8, 0.99):
        f = lambda T: L.Formulas(model, T, 5.0).hermite(phi)[1]
        roots, pos = R.roots_and_sign(f, grid, 0.002, fs=dcs[phi])
        print(f"      formula (c), random, sigma_T = 5, phi = {phi}: {R.show_pos(pos)}")
    for amp in R.AMPS:
        f = lambda T: L.season_case(model, T, amp)["D"]
        roots, pos = R.roots_and_sign(f, grid, 1e-7)
        print(f"      exact (Floquet), seasonal, amplitude {amp:5.2f} C: {R.show_pos(pos)}")
        for r in roots:
            d = 1.0 if 11.5 < r < 33.9 else 0.25
            print(f"            D = 0 at {r:.3f} C;  D at {r - d:.3f} C = {f(r - d):+.5f},  at {r + d:.3f} C = {f(r + d):+.5f}")

    a_ = R.A_STAY
    print("\nI3b second order, a = 4, no death.  Steady stays b (steps) at which the loss is below its value with no steady stage:")
    print("      (1 + 2 theta)/(2(a + b)) - R/(1 - R) < 1/(2a), with R = E[phi^L]; by the shape of the steady stay; b from m (or 1) to 20,000")
    for phi in (0.5, 0.9, 0.99):
        th = phi / (1 - phi)
        ra = 1 / (1 + a_ / th)
        for label, m in (("geometric (m = 1)", 1), ("chain, m = 2", 2), ("chain, m = 4", 4), ("chain, m = 16", 16), ("fixed delay", None)):
            def f(b):
                Rr = ra * (phi ** b if m is None else (1 + b / (m * th)) ** (-m))
                return 1 / (2 * a_) - L.gamma0_general_R(a_ + b, Rr, phi)
            lo = 1.0 if m is None else float(m)
            bs = np.concatenate([np.arange(lo, 50, 0.05), np.arange(50, 20001, 1.0)])
            roots, pos = R.roots_and_sign(f, bs, 1e-8)
            txt = "; ".join(f"{x:.2f} to {'beyond 20,000' if y >= 19999 else format(y, '.2f')}" for x, y in pos) if pos else "none up to 20,000"
            print(f"      phi = {phi:<4} {label:18s}: {txt}")
    print(f"\nextra blocks: wall time {time.time() - t0:.0f} s; finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
