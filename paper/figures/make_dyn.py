"""Generate CSV figure data for Section 7 (hosts with their own dynamics).

Run from repository root or manuscript/jmb:
    python3 -I ../paper/figures/make_dyn.py

Dependencies: Python 3 standard library and numpy only (SPEC.md section 2).
Sources:
  - dyn-1.csv: Proposition 4 (K4) cubic and Routh-Hurwitz margin F(a_L), eq. (eq:dyn-fquad),
    evaluated at the admissible illustrative V block (a_V=1.05, alpha=1.0, p=0.20, rho=-0.196),
    together with the four stiff nonlinear ODE perturbation rates reported in
    verification/coupled_check_output.txt (lines 50-57) and
    verification/check-coupled-hosts.md (section 1.2).
  - dyn-2.csv: R11 locally stable equilibrium shares with corrected finite-K and infinite-K
    denominators from results/r11-dobson-trophic_20261114.txt (lines 64-75)
    and reports/R11-dobson-trophic-model.md (H1).
"""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np


def build_dyn_1(data_dir: Path, repo_dobson: Path) -> None:
    # Admissible illustrative V block (verification/check-coupled-hosts.md section 1.2)
    p = 0.20
    a_v = 1.05
    alpha = 1.00
    rho = -0.196

    q = a_v
    d = a_v / (1.0 - p)
    h = alpha * p

    A = rho + d
    B = d * (rho + h)
    J = rho + d - q
    C = rho * (d - q) + d * h

    disc = (A * J + q * rho) ** 2 - 4.0 * J * A * B
    a_minus = (-(A * J + q * rho) - math.sqrt(disc)) / (2.0 * J)
    a_plus = (-(A * J + q * rho) + math.sqrt(disc)) / (2.0 * J)

    # Parse the four saved ODE_fit rates directly from verification/coupled_check_output.txt
    check_txt = (repo_dobson / "verification" / "coupled_check_output.txt").read_text(encoding="utf-8")
    ode_rates: dict[str, float] = {}
    for tag in ("R2-1", "R2-2", "R2-3", "R2-4"):
        m = re.search(rf"\[PASS\] {tag} [^\n]*\n\s+Re\(lam_dom\)=[+\-0-9.]+, ODE_fit=([+\-0-9.]+)", check_txt)
        if not m:
            raise RuntimeError(f"Could not parse {tag} from coupled_check_output.txt")
        ode_rates[tag] = float(m.group(1))

    ode_points = [
        (0.35 * a_minus, ode_rates["R2-1"], True),
        (1.80 * a_minus, ode_rates["R2-2"], False),
        (0.65 * a_plus, ode_rates["R2-3"], False),
        (1.60 * a_plus, ode_rates["R2-4"], True),
    ]

    grid = np.unique(
        np.concatenate(
            [
                np.geomspace(0.008, 4.0, 260),
                np.array([a_minus, a_plus], dtype=float),
            ]
        )
    )

    out_path = data_dir / "dyn-1.csv"
    lines = ["a_L,F,re_max,ode_stable_a,ode_stable_re,ode_unstable_a,ode_unstable_re"]
    for idx, a_l in enumerate(grid):
        f_val = J * a_l * a_l + (A * J + q * rho) * a_l + A * B
        roots = np.roots(np.array([1.0, a_l + A, B + a_l * J, a_l * C], dtype=float))
        re_max = float(np.max(np.real(roots)))

        st_a = "nan"
        st_re = "nan"
        un_a = "nan"
        un_re = "nan"
        if idx < len(ode_points):
            pt_a, pt_re, is_stable = ode_points[idx]
            if is_stable:
                st_a = f"{pt_a:.10f}"
                st_re = f"{pt_re:.8f}"
            else:
                un_a = f"{pt_a:.10f}"
                un_re = f"{pt_re:.8f}"

        lines.append(
            f"{a_l:.10f},{f_val:.10f},{re_max:.10f},{st_a},{st_re},{un_a},{un_re}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dyn_2(data_dir: Path, repo_dobson: Path) -> None:
    r11_txt = (repo_dobson / "results" / "r11-dobson-trophic_20261114.txt").read_text(encoding="utf-8")

    m_fin = re.search(
        r"K finite: n = (\d+)\s+"
        r"endemic equilibrium stable:\s+(\d+)/\1\s+=\s+[0-9.]+\s+\+-\s+[0-9.]+\s+"
        r"parasite-free equilibrium stable:\s+(\d+)/\1\s+=\s+[0-9.]+\s+\+-\s+[0-9.]+\s+"
        r"difference[^\n]+\n\s+"
        r"parasite-free stable\s+->\s+endemic stable (\d+), endemic unstable (\d+)\s+"
        r"parasite-free unstable\s+->\s+endemic stable (\d+), endemic unstable (\d+)",
        r11_txt,
    )
    m_inf = re.search(
        r"K = infinity: n = (\d+)\s+endemic equilibrium stable:\s+(\d+)/\1",
        r11_txt,
    )
    if not m_fin or not m_inf:
        raise RuntimeError("Could not parse R11 H1 counts from r11-dobson-trophic_20261114.txt")

    n_fin = int(m_fin.group(1))
    end_fin = int(m_fin.group(2))
    pf_fin = int(m_fin.group(3))
    end_given_pf_unst = int(m_fin.group(6))
    pf_unst_denom = int(m_fin.group(6)) + int(m_fin.group(7))

    n_inf = int(m_inf.group(1))
    end_inf = int(m_inf.group(2))

    rows = [
        (0, 100.0 * end_fin / n_fin, end_fin, n_fin),
        (1, 100.0 * pf_fin / n_fin, pf_fin, n_fin),
        (2, 100.0 * end_given_pf_unst / pf_unst_denom, end_given_pf_unst, pf_unst_denom),
        (3, 100.0 * end_inf / n_inf, end_inf, n_inf),
    ]

    out_path = data_dir / "dyn-2.csv"
    lines = ["idx,pct,count,denom"]
    for idx, pct, count, denom in rows:
        lines.append(f"{idx},{pct:.6f},{count},{denom}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    here = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dobson = (__import__('pathlib').Path(__file__).resolve().parent / '../../code').resolve()

    build_dyn_1(data_dir, repo_dobson)
    build_dyn_2(data_dir, repo_dobson)
    print(f"Wrote {data_dir / 'dyn-1.csv'} and {data_dir / 'dyn-2.csv'}")


if __name__ == "__main__":
    main()
