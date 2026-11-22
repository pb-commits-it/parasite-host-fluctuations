#!/usr/bin/env python3
"""Auxiliary high-precision ground-truth audit for the finite-sample discrepancies
observed between the single-run `pilot.md` (n=200) and secondary's rerun (n=400).

Does NOT modify `pilot_rerun_results.json` or `pilot_rerun_cells.csv`.
Writes `highprec_discrepancy_audit.json` in this directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

for _env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_env_var] = "1"

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import numpy as np

from pilot_lib import POWER80_FACTOR, compute_finite_conditional_growth, wilson_ci95
from sim_world import (
    build_geometric_ring,
    exact_ring_small_noise_K,
    generate_lognormal_ar1_hosts,
    run_individual_world,
    sample_se,
    solve_threshold_D,
)

ORIG_RESULTS = (__import__('pathlib').Path(__file__).resolve().parent / '../../../virtualworld/results').resolve()


def main() -> int:
    t0 = time.monotonic()

    # 1. High-precision ground truth for s0.6_p0_t5 extinction (N = 20,000 trajectories)
    mdl_t5 = build_geometric_ring([2.0, 5.0])
    n_hp = 20_000
    h_t5 = generate_lognormal_ar1_hosts(500, n_hp, sigma=0.6, phi=0.0, seed=358100)
    sim_lr = run_individual_world(
        mdl_t5, np.broadcast_to([6, 14], (n_hp, 2)).copy(), 500, seed=358101, hosts=h_t5, tilt_y=0.05
    )
    sim_modal = run_individual_world(
        mdl_t5, np.broadcast_to([5, 15], (n_hp, 2)).copy(), 500, seed=358102, hosts=h_t5, tilt_y=0.05
    )
    k_lr = int(np.sum(sim_lr["extinction_step"] <= 500))
    k_modal = int(np.sum(sim_modal["extinction_step"] <= 500))
    p_lr = k_lr / n_hp
    p_modal = k_modal / n_hp
    p_const_t5_exact = 0.3536551469103162  # exact PGF at [6, 14]

    # 2. High-precision ground truth for s0.3_p0_t20 growth100 (N = 20,000 schedules)
    mdl_t20 = build_geometric_ring([2.0, 20.0])
    h_t20 = generate_lognormal_ar1_hosts(100, n_hp, sigma=0.3, phi=0.0, seed=358200)
    g100_t20 = compute_finite_conditional_growth(mdl_t20, h_t20, 0.05, [2, 18], horizons=(100,))[100]

    # 3. High-precision ground truth for s0.6_p0.6_t5 threshold D (256 antithetic pairs, 65,536 steps)
    h_d_hp = generate_lognormal_ar1_hosts(65_536 + 2_048, 256, sigma=0.6, phi=0.6, seed=358300, antithetic=True)
    s2k_t5 = 0.36 * exact_ring_small_noise_K([2.0, 5.0], 0.6)
    d_hp = solve_threshold_D(mdl_t5, h_d_hp, burn_steps=2_048, n_pairs=256, cv_weight=1.0 / 7.0, init_offset=s2k_t5)

    # 4. High-precision ground truth + bootstrap CI for continuous growth n80_per_cell
    mdl_t1 = build_geometric_ring([2.0, 1.0])
    h_m_t1 = generate_lognormal_ar1_hosts(100, n_hp, sigma=0.6, phi=0.6, seed=358400)
    h_m_t20 = generate_lognormal_ar1_hosts(100, n_hp, sigma=0.6, phi=0.6, seed=358401)
    g_m_t1 = compute_finite_conditional_growth(mdl_t1, h_m_t1, 0.05, [13, 7], horizons=(100,))[100]
    g_m_t20 = compute_finite_conditional_growth(mdl_t20, h_m_t20, 0.05, [2, 18], horizons=(100,))[100]
    g_c_t1 = 0.01676555451413363
    g_c_t20 = 0.0023177798609496274
    true_delta_g = (g_c_t1 - float(np.mean(g_m_t1))) - (g_c_t20 - float(np.mean(g_m_t20)))
    true_varsum_g = float(np.var(g_m_t1, ddof=1) + np.var(g_m_t20, ddof=1))
    true_raw_n80_g = POWER80_FACTOR * true_varsum_g / (true_delta_g * true_delta_g)

    # Nonparametric bootstrap (20,000 resamples) of primary's own saved n=200 arrays in results/
    orig_t1 = np.load(ORIG_RESULTS / "s0.6_p0.6_t1.npz")["growth100"]
    orig_t20 = np.load(ORIG_RESULTS / "s0.6_p0.6_t20.npz")["growth100"]
    orig_c_t1 = float(np.load(ORIG_RESULTS / "s0_p0.6_t1.npz")["growth100"][0])
    orig_c_t20 = float(np.load(ORIG_RESULTS / "s0_p0.6_t20.npz")["growth100"][0])
    rng = np.random.default_rng(358500)
    b_reps = 20_000
    idx1 = rng.integers(0, orig_t1.size, size=(b_reps, orig_t1.size))
    idx20 = rng.integers(0, orig_t20.size, size=(b_reps, orig_t20.size))
    b_t1 = orig_t1[idx1]
    b_t20 = orig_t20[idx20]
    b_delta = (orig_c_t1 - np.mean(b_t1, axis=1)) - (orig_c_t20 - np.mean(b_t20, axis=1))
    b_vsum = np.var(b_t1, axis=1, ddof=1) + np.var(b_t20, axis=1, ddof=1)
    b_n80 = np.ceil(POWER80_FACTOR * b_vsum / (b_delta * b_delta))

    out = {
        "elapsed_seconds": time.monotonic() - t0,
        "s0.6_p0_t5_extinction_N20000": {
            "p_largest_remainder_6_14": p_lr,
            "wilson95_lr": list(wilson_ci95(k_lr, n_hp)),
            "p_modal_5_15": p_modal,
            "wilson95_modal": list(wilson_ci95(k_modal, n_hp)),
            "true_simple_noise_diff_lr": p_lr - p_const_t5_exact,
        },
        "s0.3_p0_t20_growth100_N20000": {
            "mean": float(np.mean(g100_t20)),
            "se": sample_se(g100_t20),
        },
        "s0.6_p0.6_t5_D_256pairs_65536steps": {
            "D": float(d_hp["D"]),
            "se": float(d_hp["se"]),
            "ci95": [
                float(d_hp["D"]) - 1.9693105698458823 * float(d_hp["se"]),
                float(d_hp["D"]) + 1.9693105698458823 * float(d_hp["se"]),
            ],
        },
        "growth_n80_ground_truth_and_orig_bootstrap": {
            "N20000_delta_growth": true_delta_g,
            "N20000_variance_sum": true_varsum_g,
            "N20000_raw_n80": true_raw_n80_g,
            "N20000_ceil_n80": int(math.ceil(true_raw_n80_g)),
            "orig_n200_bootstrap_ci95_n80": np.quantile(b_n80, [0.025, 0.975]).tolist(),
            "orig_n200_bootstrap_10_90_n80": np.quantile(b_n80, [0.10, 0.90]).tolist(),
        },
    }
    (HERE / "highprec_discrepancy_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
