#!/usr/bin/env python3
"""Convergence check for Method M3 (strictly positive row-stochastic linear Markov chain).

Verifies second-order spatial convergence O(dx^2) of the positive-weight Markov chain
(min_mu >= 0, TV = 1) toward the 55-digit discontinuous Gauss-Legendre solver, and
also checks the singular lattice case (small b where support has a Cantor gap).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_gap2_independent import GLKoopmanOperator, solve_positive_linear_chain, HERE


def main() -> None:
    print("=== Method M3 Positive-Weight Chain Convergence & Singular-Cantor Check ===", flush=True)
    cases = [
        # (L, delta, phi, label) -- note delta=0.25 has 3*atanh(0.25) < L so invariant law is Cantor-singular!
        (12.0, 0.5, -0.5, "smooth_or_overlapping_neg"),
        (12.0, 0.5, 0.5,  "smooth_or_overlapping_pos"),
        (20.0, 0.8, 0.0,  "large_amp_iid"),
        (10.0, 0.25, 0.0, "cantor_singular_iid"),
        (10.0, 0.25, -0.5, "cantor_singular_neg"),
    ]
    grids = [3201, 6401, 12801]
    rows = []
    all_pass = True
    for L_val, d_val, p_val, label in cases:
        with mp.workdps(55):
            gl_op = GLKoopmanOperator(mp.mpf(str(L_val)), mp.mpf(str(d_val)), mp.mpf(str(p_val)),
                                      n_pts=17, max_h=0.85, dps=55)
            q_gl = float(gl_op.solve_adjoint_invariant()["q_L"])
        diffs = []
        min_mus = []
        tvs = []
        q_pos_vals = []
        for ng in grids:
            res = solve_positive_linear_chain(L_val, d_val, p_val, n_grid=ng)
            q_pos_vals.append(res["q_L"])
            diffs.append(abs(res["q_L"] - q_gl))
            min_mus.append(res["min_mu"])
            tvs.append(res["tv"])
        # Richardson extrapolation between 6401 and 12801 assuming O(dx^2)
        q_rich = (4.0 * q_pos_vals[2] - q_pos_vals[1]) / 3.0
        rich_diff = abs(q_rich - q_gl)
        ok = (
            min(min_mus) >= -1e-16
            and max(abs(t - 1.0) for t in tvs) < 1e-12
            and diffs[2] < 3e-8
            and diffs[0] / diffs[2] > 6.5  # monotone refinement toward GL (L=20 has coarse dx~0.013)
            and rich_diff < 1.5e-8
        )
        all_pass = all_pass and ok
        row = dict(
            label=label, L=L_val, delta=d_val, phi=p_val,
            q_gl=q_gl, diffs=diffs, ratio_3200_to_12800=diffs[0] / diffs[2],
            richardson_diff=rich_diff, min_mu=min(min_mus), max_tv_err=max(abs(t - 1.0) for t in tvs),
            passed=ok,
        )
        rows.append(row)
        print(f"{'PASS' if ok else 'FAIL'} | {label} (L={L_val}, d={d_val}, p={p_val}) | "
              f"|dq|=[{diffs[0]:.3e}, {diffs[1]:.3e}, {diffs[2]:.3e}] "
              f"(ratio={diffs[0]/diffs[2]:.2f}x, Rich={rich_diff:.3e}) min_mu={min(min_mus):.2e}",
              flush=True)

    (HERE / "check_positive_chain_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
