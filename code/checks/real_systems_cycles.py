"""secondary's illustrations for real parasites (verification/systems/SYSTEMS.md), recomputed in continuous time, and the same
with the host's fluctuation treated as a cycle of fixed period instead of noise with fading memory.

Inputs are secondary's: stage durations in months, the variance of log abundance of the one fluctuating host, and its
autocorrelation phi over a step dt. These are published values measured for other purposes; see SYSTEMS.md for the
sources and the caveats. This script fits nothing.

Claims:
  Y1  With Ornstein-Uhlenbeck memory alpha = -ln(phi)/dt the formula of notes/continuous-time.md reproduces
      secondary's continuous-time values (Table A of SYSTEMS.md) to 3 significant figures.
  Y2  With the host's variance placed at a single period (12 months; 36 and 48 months for the vole), the loss per
      generation is within a factor of 2.5 of the value in Y1, in every system (printed; this is a description of
      six cases, not a general statement).

First run, 14 November 2026 about 04:25 (output kept as results/real-systems-cycles_attempt1_20261114.txt): Y2 as
first stated, "within a factor of two", failed for Borrelia at the northern edge of its range (0.084 for an annual
cycle against 0.203 with fading memory, a factor of 2.4). I had written "less than a factor of two" into the
manuscript after computing five of the six cases; the sixth is the same life cycle with a longer memory. The
statement is now "up to 2.4", here and in the manuscript.

Run: python3 -I checks/real_systems_cycles.py
"""
from __future__ import annotations

import math
import sys

import numpy as np


def response_ct(stays, period):
    """Loss per generation per unit of host variance at one period, exponential stays (continuous time)."""
    a = 1 / np.asarray(stays, float); T = np.sum(1 / a); w = 2 * math.pi / period
    g = np.prod(a / (a - 1j * w))
    return float(-(g / (1 - g)).real)


def gamma_ou(stays, alpha):
    a = 1 / np.asarray(stays, float); T = np.sum(1 / a); R = np.prod(a / (a + alpha))
    return float(1 / (alpha * T) - R / (1 - R))


SYSTEMS = [  # name, stays in months, sigma^2, phi, dt in months, secondary's continuous-time value, periods to try
    ("S. haematobium (snail 2, people 48)", [2, 48], 0.3469, 0.398, 2.0, 0.00707, (12,)),
    ("S. mansoni (snail 2, people 48)", [2, 48], 1.9044, 0.367, 2.0, 0.03727, (12,)),
    ("P. falciparum (mosquito 0.8, people 6)", [0.8, 6.0], 1.96, 0.75, 0.8, 0.16227, (12,)),
    ("B. burgdorferi, oak forest (mouse 4.4, tick 10)", [4.4, 10.0], 0.4462, 0.30, 4.4, 0.05157, (12,)),
    ("B. burgdorferi, northern edge (mouse 4.4, tick 10)", [4.4, 10.0], 1.4161, 0.50, 4.4, 0.20284, (12,)),
    ("E. multilocularis (fox 2, egg 2.5, vole 4)", [2, 2.5, 4], 1.3783, 0.75, 2.0, 0.34331, (36, 48)),
]


def main():
    fails = 0
    worst = 0.0
    for name, stays, s2, phi, dt, secondary, periods in SYSTEMS:
        ou = s2 * gamma_ou(stays, -math.log(phi) / dt)
        worst = max(worst, abs(ou / secondary - 1))
        cyc = {P: s2 * response_ct(stays, P) for P in periods}
        ratio = [v / ou for v in cyc.values()]
        ok2 = all(0.4 <= r <= 2.5 for r in ratio)
        fails += 0 if ok2 else 1
        cv2 = sum(x * x for x in stays) / sum(stays) ** 2
        print(f"{name}: generation {sum(stays):.1f} months; fading memory {ou:.4f} (secondary {secondary:.4f}); "
              + "; ".join(f"cycle of {P} months {v:.4f}" for P, v in cyc.items()) + f"; very slow host {s2 * (1 - cv2) / 2:.4f}  [{'ok' if ok2 else 'FAIL'} Y2]")
    ok1 = worst < 5e-3
    fails += 0 if ok1 else 1
    print(f"[{'ok' if ok1 else 'FAIL'}] Y1 secondary's continuous-time values reproduced: largest relative difference {worst:.1e}")
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
