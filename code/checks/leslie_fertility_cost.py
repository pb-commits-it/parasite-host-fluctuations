"""The closed form for fluctuating fertility in an age-structured population, against simulation.

Claim checked (`notes/general-ring-v2.md`, F2b): when one environmental factor with log standard deviation sigma and
autocorrelation phi multiplies every fertility of a Leslie matrix, the second-order cost of fluctuation is
    sigma^2 [ (1 + 2 theta)/(2 T^2) - (1/T) R/(1 - R) ],   T = sum_a a l_a m_a lambda^-a,   R = sum_a l_a m_a (phi/lambda)^a,
and in particular that the cost is NEGATIVE (fluctuation raises long-run growth) for a life table whose age of
reproduction has a coefficient of variation above one, once the environment has enough memory.

Cases and criteria, fixed before the first run. Four life tables, each at phi = 0, 0.5 and 0.9, sigma = 0.05:
  L1  three ages, reproduction at ages 2 and 3                      (CV^2 of the mother's age 0.04)
  L2  reproduction at age 1 and again at age 8                      (CV^2 1.31): gain predicted at phi = 0.5 and 0.9
  L3  reproduction spread evenly over six ages                      (CV^2 0.37)
  L4  reproduction at ages 4 and 5 only                             (CV^2 0.01)
Passes if every simulated cost is within the larger of 3.5 standard errors and 2% of the closed form, and the two
predicted gains are negative by more than 3.5 standard errors. 300 replicates of 60,000 steps after 1,000 discarded,
with the control variate. The simulator is `simulate_cost` of `checks/error_bound_general_steps.py`, which that
script tests against the matrix formula.

Run: python3 -I checks/leslie_fertility_cost.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

spec = importlib.util.spec_from_file_location("ebg", HERE / "error_bound_general_steps.py")
ebg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebg)

CASES = {
    "L1": ([0, 1.2, 1.5], [0.5, 0.6]),
    "L2": ([1.0, 0, 0, 0, 0, 0, 0, 6.0], [0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
    "L3": ([0.4] * 6, [0.8] * 5),
    "L4": ([0, 0, 0, 1.5, 2.5], [0.6, 0.8, 0.8, 0.8]),
}


def leslie(m, p):
    d = len(m)
    a = np.zeros((d, d))
    a[0, :] = m
    for i in range(d - 1):
        a[i + 1, i] = p[i]
    return a


def closed(m, p, phi):
    a = leslie(m, p)
    lam = theory.perron(a)[0]
    d = len(m)
    l_ = np.concatenate([[1.0], np.cumprod(p)])
    age = np.arange(1, d + 1)
    w = l_ * np.asarray(m, float) * lam ** (-age)
    T = float(np.sum(age * w))
    cv2 = float(np.sum(age * age * w) - T * T) / T ** 2
    if phi == 0:
        return 1 / (2 * T * T), T, cv2
    theta = phi / (1 - phi)
    R = float(np.sum(w * phi ** age))
    return (1 + 2 * theta) / (2 * T * T) - (R / (1 - R)) / T, T, cv2


def main():
    rng = np.random.default_rng(2026111406)
    sigma = 0.05
    fails = 0
    for name, (m, p) in CASES.items():
        a = leslie(m, p)
        ent = [(0, j) for j in range(len(m)) if m[j] > 0]
        for phi in (0.0, 0.5, 0.9):
            pred, T, cv2 = closed(m, p, phi)
            mat = theory.matrix_cost(a, ent, [0] * len(ent), [1.0] * len(ent), [phi])
            cost, se = ebg.simulate_cost(a, ent, [0] * len(ent), np.ones(len(ent)), np.array([phi]), sigma, 300, 60_000, 1_000, rng)
            c, s = cost / sigma ** 2, se / sigma ** 2
            ok = abs(c - pred) <= max(3.5 * s, 0.02 * abs(pred)) and abs(mat - pred) < 1e-8 * max(1.0, abs(pred))
            if pred < 0:
                ok = ok and (c < -3.5 * s)
            fails += 0 if ok else 1
            print(f"[{'ok' if ok else 'FAIL'}] {name} (T = {T:.2f}, CV^2 = {cv2:.2f}) phi = {phi}: closed form {pred:+.5f}; simulated {c:+.5f} +- {s:.5f}"
                  + ("   <- a gain" if pred < 0 else ""), flush=True)
    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
