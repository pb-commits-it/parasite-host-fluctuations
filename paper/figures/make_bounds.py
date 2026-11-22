#!/usr/bin/env python3
"""Generate CSV data for Section 4 figure (bounds-1.csv).

Run from repository root or manuscript/jmb:
    python3 -I ../paper/figures/make_bounds.py

Dependencies: Python 3 standard library and numpy only (SPEC.md section 2).
Data source:
    results/r10-loss-at-large-noise_20261114.txt (sections H3 and H4)
    and closed-form threshold approximation Eq. (eq:bounds-gamma0).

Rule 7 verification checks (tolerances fixed beforehand):
  1. Closed-form Gamma_0 for stays (1, 24) at phi = 0 and phi = 0.75 reproduces
     the printed H3 ratios D / Gamma_0 in r10-loss-at-large-noise_20261114.txt
     at sigma = 2.0 (3.37 at phi = 0 and 2.07 at phi = 0.75) to within 0.01, and
     reproduces H1c small-noise value (stays (1, 24), sigma = 0.2, phi = 0.4 ->
     0.001430) to within 1e-6.
  2. All 11 H4 rows parsed from r10-loss-at-large-noise_20261114.txt satisfy
     simulated D <= bound + 0.6 * SE and match the known row count (11).
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def gamma0_ring(stays: list[float], sigma: float, phi: float) -> float:
    """Threshold second-order loss Gamma_0 from Eq. (eq:bounds-gamma0) for one driver."""
    tau = np.asarray(stays, dtype=float)
    tot = float(np.sum(tau))
    if phi == 0.0:
        return (sigma * sigma) / (2.0 * tot)
    theta = phi / (1.0 - phi)
    r_val = float(np.prod(1.0 / (1.0 + tau / theta)))
    return (sigma * sigma) * ((1.0 + 2.0 * theta) / (2.0 * tot) - r_val / (1.0 - r_val))


def parse_h3_line(r10_txt: str, label_prefix: str) -> list[tuple[float, float]]:
    pattern = rf"^\s*{re.escape(label_prefix)}:\s*(.+)$"
    m = re.search(pattern, r10_txt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not find H3 row: {label_prefix}")
    pairs = re.findall(r"([0-9.]+)\s*\(([0-9.]+)\)", m.group(1))
    if len(pairs) != 9:
        raise RuntimeError(f"Expected 9 sigma columns for {label_prefix}, got {len(pairs)}")
    return [(float(d_str), float(rat_str)) for d_str, rat_str in pairs]


def parse_h4_rows(r10_txt: str) -> list[tuple[str, float, float, float, float]]:
    h4_block = r10_txt.split("H4  the parameter sets of the real systems", 1)[1].split("\n\n", 1)[0]
    rows: list[tuple[str, float, float, float, float]] = []
    for line in h4_block.strip().splitlines():
        m = re.match(
            r"^\s*(.+?)\s+D\s*=\s*([0-9.]+)\s*\+-\s*([0-9.]+)\s+second order\s+([0-9.]+)\s+ratio\s+[0-9.]+\s+no-staying value\s+([0-9.]+)\s*$",
            line,
        )
        if not m:
            raise RuntimeError(f"Could not parse H4 line: {line}")
        rows.append(
            (
                m.group(1).strip(),
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
            )
        )
    if len(rows) != 11:
        raise RuntimeError(f"Expected 11 H4 rows, got {len(rows)}")
    return rows


def main() -> None:
    here = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dobson = (__import__('pathlib').Path(__file__).resolve().parent / '../../code').resolve()
    r10_path = repo_dobson / "results" / "r10-loss-at-large-noise_20261114.txt"
    r10_txt = r10_path.read_text(encoding="utf-8")

    # Rule 7 check 1: H1c benchmark at (1, 24), sigma=0.2, phi=0.4 -> 0.001430
    g_h1c = gamma0_ring([1.0, 24.0], 0.2, 0.4)
    if abs(g_h1c - 0.001430) > 1e-6:
        raise AssertionError(f"H1c Gamma_0 mismatch: {g_h1c:.8f}")

    sigmas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
    h3_phi0 = parse_h3_line(r10_txt, "schistosome (1,24)             phi=0.0 ")
    h3_phi075 = parse_h3_line(r10_txt, "schistosome (1,24)             phi=0.75")

    # Verify printed ratios at sigma = 2.0
    r0_2 = h3_phi0[-1][0] / gamma0_ring([1.0, 24.0], 2.0, 0.0)
    r75_2 = h3_phi075[-1][0] / gamma0_ring([1.0, 24.0], 2.0, 0.75)
    if abs(r0_2 - h3_phi0[-1][1]) > 0.01 or abs(r75_2 - h3_phi075[-1][1]) > 0.01:
        raise AssertionError(f"H3 ratio mismatch at sigma=2.0: {r0_2:.3f}, {r75_2:.3f}")

    h4_rows = parse_h4_rows(r10_txt)
    for label, d_sim, d_se, g0, bnd in h4_rows:
        if d_sim > bnd + 0.6 * d_se:
            raise AssertionError(f"Bound violated in H4 row {label}: {d_sim} > {bnd}")

    # Combine panel (a) (9 rows + dense smooth curve for bound) and panel (b) (11 rows) into bounds-1.csv
    n_rows = max(len(sigmas), len(h4_rows))
    lines = [
        "sigma,d_phi0,g0_phi0,d_phi075,g0_phi075,bound_a,case_idx,h4_d,h4_se,h4_g0,h4_bound"
    ]
    for i in range(n_rows):
        if i < len(sigmas):
            s = sigmas[i]
            d0 = h3_phi0[i][0]
            g0_0 = gamma0_ring([1.0, 24.0], s, 0.0)
            d75 = h3_phi075[i][0]
            g0_75 = gamma0_ring([1.0, 24.0], s, 0.75)
            bnd_a = 0.5 * s * s
            col_a = f"{s:.4f},{d0:.6f},{g0_0:.6f},{d75:.6f},{g0_75:.6f},{bnd_a:.6f}"
        else:
            col_a = "nan,nan,nan,nan,nan,nan"

        if i < len(h4_rows):
            _, h4_d, h4_se, h4_g0, h4_bnd = h4_rows[i]
            col_b = f"{i + 1},{h4_d:.6f},{h4_se:.6f},{h4_g0:.6f},{h4_bnd:.6f}"
        else:
            col_b = "nan,nan,nan,nan,nan"

        lines.append(f"{col_a},{col_b}")

    out_path = data_dir / "bounds-1.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified benchmarks and wrote {out_path}")


if __name__ == "__main__":
    main()
