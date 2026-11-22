#!/usr/bin/env python3
"""Generate CSV data for Section 3 worked-example figure (example-1a.csv, example-1b.csv).

Run from manuscript working copy:
    python3 -I figures/make_example.py

Dependencies: Python 3 standard library and numpy only (SPEC.md section 2),
importing verified closed forms from ../../code/engine/closed_forms.py.

Rule 7 verification checks (tolerances fixed beforehand):
  1. Single-stage invariance: gamma_one([6], phi) == 1/12 at phi = 0 and phi = 0.8
     to within 1e-15 (Proposition prop:ring-band, left equality).
  2. Closed form gamma_one(taus, phi) matches independent n x n matrix-resolvent
     evaluation T * [c^2/2 - phi * w^T B (I - phi T_perp)^{-1} Q B v] across all
     five Table A life cycles and the (6, 24), (6, 26), (6, 27) two-stage rings
     at phi in {0.0, 0.8} to within 1e-13.
  3. Reversal threshold b* = a(2a - 1)theta / (a + theta) = 26.4 at a = 6, phi = 0.8:
     relative loss exceeds 1 at b = 26 (1.0114) and falls below 1 at b = 27 (0.9833)
     to within 1e-12.
  4. Parallel equal-share scaling: gamma_alt([6, 24], phi, [1/M]*M, rho) divided by
     gamma_one([6, 24], phi) equals 1/M at rho = 0 and 1 at rho = 1 for M = 1..5
     and phi in {0.0, 0.8} to within 1e-14; noisy newcomer variances (1, 9) at equal
     shares give factor 2.5 to within 1e-14.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parents[1] / "code/engine"))
from closed_forms import gamma_alt, gamma_one  # noqa: E402


def matrix_resolvent_gamma_n(taus: list[float], phi: float) -> float:
    """Independent n x n matrix-resolvent evaluation of Gamma / sigma^2 at lambda = 1."""
    tau = np.asarray(taus, dtype=float)
    n = len(tau)
    tot = float(np.sum(tau))
    s = 1.0 - 1.0 / tau
    g = 1.0 / tau
    if n == 1:
        return 1.0 / (2.0 * tot)
    m_mat = np.diag(s)
    for k in range(n):
        m_mat[(k + 1) % n, k] += g[k]
    v = tau / tot
    w = np.ones(n, dtype=float)
    q_proj = np.eye(n) - np.outer(v, w)
    t_perp = m_mat - np.outer(v, w)
    b_edge = np.zeros((n, n), dtype=float)
    b_edge[1 % n, 0] = g[0]
    c_freq = float(w @ b_edge @ v)
    quad = float(w @ b_edge @ np.linalg.solve(np.eye(n) - phi * t_perp, q_proj @ b_edge @ v))
    return tot * (0.5 * c_freq * c_freq - phi * quad)


SERIES_CASES: list[tuple[str, list[int]]] = [
    ("1_host", [6]),
    ("2_hosts_stay_3", [6, 3]),
    ("2_hosts_stay_60", [6, 60]),
    ("3_hosts_30_30", [6, 30, 30]),
    ("3_hosts_1_59", [6, 1, 59]),
]


def verify_known_results() -> None:
    ref0 = gamma_one([6], 0.0)
    ref08 = gamma_one([6], 0.8)
    if abs(ref0 - 1.0 / 12.0) > 1e-15 or abs(ref08 - 1.0 / 12.0) > 1e-15:
        raise AssertionError(f"One-stage reference mismatch: {ref0}, {ref08}")

    extra = [([6, 24], 0.8), ([6, 26], 0.8), ([6, 27], 0.8)]
    for _, taus in SERIES_CASES:
        for phi in (0.0, 0.8):
            cf = gamma_one(taus, phi)
            mr = matrix_resolvent_gamma_n([float(x) for x in taus], phi)
            if abs(cf - mr) > 1e-13:
                raise AssertionError(f"Resolvent mismatch for {taus} at phi={phi}: {cf} vs {mr}")
    for taus, phi in extra:
        cf = gamma_one(taus, phi)
        mr = matrix_resolvent_gamma_n([float(x) for x in taus], phi)
        if abs(cf - mr) > 1e-13:
            raise AssertionError(f"Resolvent mismatch for {taus} at phi={phi}: {cf} vs {mr}")

    rel26 = gamma_one([6, 26], 0.8) / ref08
    rel27 = gamma_one([6, 27], 0.8) / ref08
    if not (rel26 > 1.0 and rel27 < 1.0):
        raise AssertionError(f"Threshold crossing failed: b=26 -> {rel26}, b=27 -> {rel27}")
    if abs(rel26 - (9.0 / 64.0 - 4.0 / 71.0) * 12.0) > 1e-12:
        raise AssertionError(f"Exact b=26 ratio mismatch: {rel26}")

    for phi in (0.0, 0.8):
        base = gamma_one([6, 24], phi)
        for m in range(1, 6):
            shares = [1.0 / m] * m
            r_ind = gamma_alt([6, 24], phi, shares, rho=0.0) / base
            r_syn = gamma_alt([6, 24], phi, shares, rho=1.0) / base
            if abs(r_ind - 1.0 / m) > 1e-14:
                raise AssertionError(f"Parallel indep mismatch at M={m}, phi={phi}: {r_ind}")
            if abs(r_syn - 1.0) > 1e-14:
                raise AssertionError(f"Parallel sync mismatch at M={m}, phi={phi}: {r_syn}")

    noisy = gamma_alt([6, 24], 0.8, [0.5, 0.5], rel_var=[1.0, 9.0]) / gamma_one([6, 24], 0.8)
    if abs(noisy - 2.5) > 1e-14:
        raise AssertionError(f"Noisy newcomer factor mismatch: {noisy}")


def build_example_1a(data_dir: Path) -> Path:
    ref_g0 = gamma_one([6], 0.0)
    ref_g08 = gamma_one([6], 0.8)
    lines = ["idx,cycle,T,g_phi0,rel_phi0,g_phi08,rel_phi08,ref"]
    for idx, (label, taus) in enumerate(SERIES_CASES):
        tot = sum(taus)
        g0 = gamma_one(taus, 0.0)
        g08 = gamma_one(taus, 0.8)
        lines.append(
            f"{idx},{label},{tot},{g0:.8f},{g0 / ref_g0:.8f},{g08:.8f},{g08 / ref_g08:.8f},1.00000000"
        )
    out = data_dir / "example-1a.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_example_1b(data_dir: Path) -> Path:
    taus = [6, 24]
    base08 = gamma_one(taus, 0.8)
    base0 = gamma_one(taus, 0.0)
    lines = ["M,g_indep_phi08,rel_indep,g_sync_phi08,rel_sync,rel_indep_phi0,rel_sync_phi0"]
    for m in range(1, 6):
        shares = [1.0 / m] * m
        gi08 = gamma_alt(taus, 0.8, shares, rho=0.0)
        gs08 = gamma_alt(taus, 0.8, shares, rho=1.0)
        gi0 = gamma_alt(taus, 0.0, shares, rho=0.0)
        gs0 = gamma_alt(taus, 0.0, shares, rho=1.0)
        lines.append(
            f"{m},{gi08:.8f},{gi08 / base08:.8f},{gs08:.8f},{gs08 / base08:.8f},"
            f"{gi0 / base0:.8f},{gs0 / base0:.8f}"
        )
    out = data_dir / "example-1b.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    verify_known_results()
    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p1 = build_example_1a(data_dir)
    p2 = build_example_1b(data_dir)
    print(f"Verified benchmarks and wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
