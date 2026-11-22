#!/usr/bin/env python3
"""Generate CSV data for Section 3 figures (ring-1.csv and ring-2.csv).

Run from repository root or manuscript/jmb:
    python3 -I ../paper/figures/make_ring.py

Dependencies: Python 3 standard library and numpy only (SPEC.md section 2).

Rule 7 verification checks (tolerances fixed beforehand):
  1. Two-stage closed form Gamma(a, b)/sigma^2 in Eq. (eq:ring-steady) matches
     explicit 2x2 matrix resolvent C = c^2/2 - phi * w^T B (I - phi T_perp)^{-1} Q B v
     multiplied by (a + b) to within 1e-13 across (a, b, phi) test points, and matches
     the two published reference values in notes/proofs-v2.md Proposition 18(c)
     (a=2, b=1, phi=0.1 -> 0.198413; a=b=100, phi=0.9 -> 0.040636) to within 1e-6.
  2. Spectral response K(omega) in Eq. (eq:ring-spectrum) matches
     notes/proofs-v2.md Proposition 3a values:
     - 1 stage of stay 12: K(omega) = 1/288 across 50 frequencies to within 1e-14;
     - 8 equal stays of 1.5 at period 12 (K = -0.0650) and period 2 (K = -0.0003)
       to within 5e-5;
     - Trapezoidal frequency average of K(omega) over (0, pi) equals 1/(2 T^2) = 1/288
       to within 1e-10.
"""
from __future__ import annotations

import cmath
import math
from pathlib import Path

import numpy as np


def gamma_steady_per_var(a: float, b: float, phi: float) -> float:
    """Closed form Gamma(a, b) / sigma^2 from Eq. (eq:ring-steady)."""
    tot = a + b
    if phi == 0.0:
        return 1.0 / (2.0 * tot)
    theta = phi / (1.0 - phi)
    return (1.0 + 2.0 * theta) / (2.0 * tot) - (theta * theta) / (theta * tot + a * b)


def matrix_resolvent_gamma(a: float, b: float, phi: float) -> float:
    """Independent 2x2 matrix-resolvent evaluation of Gamma(a, b) / sigma^2 at lambda = 1."""
    s0, g0 = 1.0 - 1.0 / a, 1.0 / a
    s1, g1 = 1.0 - 1.0 / b, 1.0 / b
    tot = a + b
    m_mat = np.array([[s0, g1], [g0, s1]], dtype=float)
    v = np.array([a / tot, b / tot], dtype=float)
    w = np.ones(2, dtype=float)
    q_proj = np.eye(2) - np.outer(v, w)
    t_perp = m_mat - np.outer(v, w)
    b_edge = np.array([[0.0, 0.0], [g0, 0.0]], dtype=float)
    c_freq = float(w @ b_edge @ v)
    quad = float(w @ b_edge @ np.linalg.solve(np.eye(2) - phi * t_perp, q_proj @ b_edge @ v))
    return tot * (0.5 * c_freq * c_freq - phi * quad)


def response_k(stays: list[float], omega: float) -> float:
    """Periodic response function K(omega) from Eq. (eq:ring-spectrum)."""
    tau = np.asarray(stays, dtype=float)
    m_stay = 1.0 - 1.0 / tau
    tot = float(np.sum(tau))
    z = cmath.exp(1j * omega)
    g_val = complex(np.prod((1.0 - m_stay) * z / (1.0 - m_stay * z)))
    return -float((g_val / (1.0 - g_val)).real) / tot


def verify_known_results() -> None:
    # Check 1: matrix resolvent and proofs-v2.md Proposition 18(c) values
    for a, b, phi in [(10.0, 1.0, 0.0), (10.0, 15.0, 0.5), (10.0, 40.0, 0.9), (2.0, 1.0, 0.1), (100.0, 100.0, 0.9)]:
        cf = gamma_steady_per_var(a, b, phi)
        mr = matrix_resolvent_gamma(a, b, phi)
        if abs(cf - mr) > 1e-13:
            raise AssertionError(f"Resolvent mismatch at {(a, b, phi)}: {cf} vs {mr}")

    if abs(gamma_steady_per_var(2.0, 1.0, 0.1) - 0.1984127) > 1e-6:
        raise AssertionError("Mismatch on proofs-v2 Prop 18(c) case (2, 1, 0.1)")
    if abs(gamma_steady_per_var(100.0, 100.0, 0.9) - 0.0406360) > 1e-6:
        raise AssertionError("Mismatch on proofs-v2 Prop 18(c) case (100, 100, 0.9)")

    # Check 2: response function K(omega) against proofs-v2.md Proposition 3a
    omegas = np.linspace(0.02, math.pi - 0.02, 50)
    max_dev_1 = max(abs(response_k([12.0], float(w)) - 1.0 / 288.0) for w in omegas)
    if max_dev_1 > 1e-14:
        raise AssertionError(f"One-stage K not constant: max dev {max_dev_1}")

    k_p12 = response_k([1.5] * 8, 2.0 * math.pi / 12.0)
    k_p2 = response_k([1.5] * 8, math.pi)
    if abs(k_p12 - (-0.0650)) > 5e-5 or abs(k_p2 - (-0.0003)) > 5e-5:
        raise AssertionError(f"Mismatch on 8x1.5 benchmark values: {k_p12}, {k_p2}")

    # Midpoint rule on (0, pi) for smooth symmetric periodic integrand
    n_quad = 2000
    w_mid = (np.arange(n_quad) + 0.5) * (math.pi / n_quad)
    for stays in ([3.0] * 4, [1.5] * 8):
        avg = float(np.mean([response_k(stays, float(w)) for w in w_mid]))
        if abs(avg - 1.0 / 288.0) > 1e-10:
            raise AssertionError(f"Frequency average mismatch for {stays}: {avg}")


def build_ring_1(data_dir: Path) -> Path:
    a = 10.0
    # Dense grid near small b where curvature is sharpest, plus integer steps up to 200
    b_grid = np.unique(
        np.concatenate(
            [
                np.linspace(1.0, 30.0, 146),
                np.linspace(30.0, 200.0, 171),
            ]
        )
    )
    one_stage = 1.0 / (2.0 * a)
    lines = [
        "b,g_phi0,low_band,up_phi05,g_phi05,up_phi09,g_phi09,one_stage"
    ]
    for b in b_grid:
        tot = a + float(b)
        low_b = 1.0 / (2.0 * tot)
        g0 = gamma_steady_per_var(a, float(b), 0.0)
        g05 = gamma_steady_per_var(a, float(b), 0.5)
        g09 = gamma_steady_per_var(a, float(b), 0.9)
        up05 = 3.0 / (2.0 * tot)   # theta = 1 at phi = 0.5 -> 1 + 2*theta = 3
        up09 = 19.0 / (2.0 * tot)  # theta = 9 at phi = 0.9 -> 1 + 2*theta = 19
        lines.append(
            f"{b:.6f},{g0:.8f},{low_b:.8f},{up05:.8f},{g05:.8f},{up09:.8f},{g09:.8f},{one_stage:.8f}"
        )
    out = data_dir / "ring-1.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_ring_2(data_dir: Path) -> Path:
    periods = np.unique(
        np.concatenate(
            [
                np.geomspace(2.0, 60.0, 320),
                np.array([2.0, 3.0, 4.0, 6.0, 12.0, 24.0, 48.0], dtype=float),
            ]
        )
    )
    lines = ["period,omega,k_8x15,k_4x3,k_1x12,zero"]
    for p in periods:
        w = 2.0 * math.pi / float(p)
        k8 = response_k([1.5] * 8, w)
        k4 = response_k([3.0] * 4, w)
        k1 = response_k([12.0], w)
        lines.append(f"{p:.6f},{w:.8f},{k8:.8f},{k4:.8f},{k1:.8f},0.00000000")
    out = data_dir / "ring-2.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    verify_known_results()
    here = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p1 = build_ring_1(data_dir)
    p2 = build_ring_2(data_dir)
    print(f"Verified benchmarks and wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
