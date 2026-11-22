#!/usr/bin/env python3
"""Independent rerun of the full 18-cell simulated-world pilot at 400 replicates/cell (r35-vworld-B).

Reuses Step-A clean-room model (`sim_world.py`) after verifying Step-A validation
passed (`validation_rerun_results.json`) and executing all structural + Rule 7
pre-validation tests (`test_sim_world.py`, `test_pilot_rerun.py`).

Uses disjoint fresh seeds (`PILOT_SEED_BASE = 352000`, `JEFFREYS_SEED = 359000`)
and checkpoints every completed cell to `cells/<cell>.json` as it runs.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

for _env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_env_var] = "1"

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import stats

from pilot_lib import (
    compute_finite_conditional_growth,
    compute_four_cell_interaction,
    compute_simple_two_cell_contrast,
    largest_remainder_allocate,
    wilson_ci95,
)
from sim_world import (
    build_geometric_ring,
    exact_ring_small_noise_K,
    generate_lognormal_ar1_hosts,
    multitype_branching_extinction_curve,
    run_individual_world,
    sample_se,
    solve_threshold_D,
)
import test_pilot_rerun
import test_sim_world

PROB_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
ORIG_VWORLD = PROB_ROOT / "virtualworld"

PILOT_SEED_BASE = 352000
JEFFREYS_SEED = 359000
N_REPS = 400
HORIZON = 500
EXPOSED_STAY = 2.0
INIT_TOTAL = 20
LOG_R0 = 0.05
THRESH_PAIRS = 32
THRESH_RECORD_STEPS = 32_768
THRESH_BURN_STEPS = 2_048


def shell_date() -> str:
    return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S %Z"], text=True).strip()


def modal_stage_allocate_historical(total: int, weights: np.ndarray) -> np.ndarray:
    """Historical modal-stage rounding used in original `pilot.md` (for discrepancy audit only)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / float(np.sum(w))
    out = np.floor(total * w).astype(np.int64)
    out[int(np.argmax(w))] += int(total) - int(np.sum(out))
    return out


def compare_against_original(
    cells: dict[str, dict[str, object]],
    effects: dict[str, object],
) -> dict[str, object]:
    """Compare every cell, main contrast, power estimate, and simple contrast against `pilot.md`."""
    orig_cells: dict[str, dict[str, str]] = {}
    with (ORIG_VWORLD / "results" / "pilot_cells.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            orig_cells[row["cell"]] = row

    fix1_cells: dict[str, dict[str, str]] = {}
    fix1_csv = ORIG_VWORLD / "results" / "fix1" / "pilot" / "pilot_cells.csv"
    if fix1_csv.exists():
        with fix1_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fix1_cells[row["cell"]] = row

    orig_effects = json.loads((ORIG_VWORLD / "results" / "pilot_effects.json").read_text(encoding="utf-8"))
    fix1_effects_path = ORIG_VWORLD / "results" / "fix1" / "pilot" / "pilot_effects.json"
    fix1_effects = json.loads(fix1_effects_path.read_text(encoding="utf-8")) if fix1_effects_path.exists() else None

    cell_comparisons: list[dict[str, object]] = []
    for key, c in cells.items():
        oc = orig_cells[key]
        fc = fix1_cells.get(key)
        sig = float(c["sigma"])

        # 1. Extinction
        o_ext = float(oc["extinction"])
        o_ext_lo, o_ext_hi = float(oc["extinction_low"]), float(oc["extinction_high"])
        ext_in_ci = o_ext_lo <= float(c["extinction"]) <= o_ext_hi
        o_ext_se = math.sqrt(o_ext * (1.0 - o_ext) / 200.0)
        c_ext_se = math.sqrt(float(c["extinction"]) * (1.0 - float(c["extinction"])) / N_REPS)
        ext_z = (float(c["extinction"]) - o_ext) / math.hypot(c_ext_se, o_ext_se)

        # 2. Early growth/step
        o_g = float(oc["growth100"])
        o_g_lo, o_g_hi = float(oc["growth100_low"]), float(oc["growth100_high"])
        o_g_se = float(oc["growth100_se"])
        if sig == 0.0:
            g_rel = abs(float(c["growth100"]) - o_g) / abs(o_g)
            g_agree = g_rel <= 0.10
            g_mode = "deterministic_rel10pct"
            g_z = None
        else:
            g_in_ci = o_g_lo <= float(c["growth100"]) <= o_g_hi
            g_rel = abs(float(c["growth100"]) - o_g) / abs(o_g) if o_g != 0.0 else math.inf
            g_agree = g_in_ci
            g_mode = "interval95"
            g_z = (float(c["growth100"]) - o_g) / math.hypot(float(c["growth100_se"]), o_g_se)

        # 3. Threshold D
        o_d = float(oc["D"])
        o_d_lo, o_d_hi = float(oc["D_low"]), float(oc["D_high"])
        o_d_se = float(oc["D_se"])
        if sig == 0.0:
            d_rel = 0.0 if abs(float(c["D"])) <= 1e-12 else math.inf
            d_agree = abs(float(c["D"]) - o_d) <= 1e-12
            d_mode = "exact_zero"
            d_z = 0.0
            d_in_ci = True
        else:
            d_in_ci = o_d_lo <= float(c["D"]) <= o_d_hi
            d_rel = abs(float(c["D"]) - o_d) / abs(o_d)
            d_agree = d_in_ci
            d_mode = "interval95"
            d_z = (float(c["D"]) - o_d) / math.hypot(float(c["D_se"]), o_d_se)

        # 4. Analytical sigma^2 K
        o_d2 = float(oc["D2"])
        d2_abs = abs(float(c["sigma2_K"]) - o_d2)
        d2_rel = 0.0 if o_d2 == 0.0 and float(c["sigma2_K"]) == 0.0 else d2_abs / abs(o_d2)
        d2_agree = d2_abs <= 1e-12

        cell_comparisons.append(
            {
                "cell": key,
                "sigma": sig,
                "phi": float(c["phi"]),
                "stay": float(c["stay"]),
                "orig": {
                    "extinct": int(oc["extinct"]),
                    "n": int(oc["n"]),
                    "extinction": o_ext,
                    "extinction_ci": [o_ext_lo, o_ext_hi],
                    "growth100": o_g,
                    "growth100_ci": [o_g_lo, o_g_hi],
                    "growth100_se": o_g_se,
                    "D": o_d,
                    "D_ci": [o_d_lo, o_d_hi],
                    "D_se": o_d_se,
                    "sigma2_K": o_d2,
                    "init": [int(oc["initial_stage0"]), int(oc["initial_stage1"])],
                },
                "fix1": None
                if fc is None
                else {
                    "extinct": int(fc["extinct"]),
                    "extinction": float(fc["extinction"]),
                    "extinction_ci": [float(fc["extinction_low"]), float(fc["extinction_high"])],
                    "growth100": float(fc["growth100"]),
                    "D": float(fc["D"]),
                    "init": [int(fc["initial_stage0"]), int(fc["initial_stage1"])],
                },
                "checks": {
                    "extinction_in_orig_ci95": ext_in_ci,
                    "extinction_two_sample_z": ext_z,
                    "growth_agree": g_agree,
                    "growth_mode": g_mode,
                    "growth_rel_diff": g_rel,
                    "growth_two_sample_z": g_z,
                    "D_in_orig_ci95": d_in_ci,
                    "D_agree_strict_ci": d_agree,
                    "D_rel_diff": d_rel,
                    "D_two_sample_z": d_z,
                    "sigma2_K_agree": d2_agree,
                    "sigma2_K_abs_diff": d2_abs,
                    "sigma2_K_rel_diff": d2_rel,
                },
            }
        )

    # Main contrasts & power
    c_ext = effects["extinction"]
    o_ext_m = orig_effects["extinction"]
    c_gr = effects["growth"]
    o_gr_m = orig_effects["growth"]

    ext_int_in_wald = o_ext_m["ci95"][0] <= c_ext["effect"] <= o_ext_m["ci95"][1]
    ext_int_in_jeff = o_ext_m["jeffreys_interval95"][0] <= c_ext["effect"] <= o_ext_m["jeffreys_interval95"][1]
    ext_label_same = bool(c_ext["finite_upper_established"]) == bool(o_ext_m["finite_upper_established"])
    n80_ext_rel = abs(float(c_ext["n80_per_cell"]) - float(o_ext_m["n80_per_cell"])) / float(o_ext_m["n80_per_cell"])
    n80_ext_in_sens = o_ext_m["n80_sensitivity_10_90"][0] <= c_ext["n80_per_cell"] <= o_ext_m["n80_sensitivity_10_90"][1]

    gr_int_in_wald = o_gr_m["ci95"][0] <= c_gr["effect"] <= o_gr_m["ci95"][1]
    n80_gr_rel = abs(float(c_gr["n80_per_cell"]) - float(o_gr_m["n80_per_cell"])) / float(o_gr_m["n80_per_cell"])

    simple_comparisons: list[dict[str, object]] = []
    for c_sim, o_sim in zip(effects["simple"], orig_effects["simple"]):
        assert c_sim["label"] == o_sim["label"]
        e_in = o_sim["ext_low"] <= c_sim["extinction"] <= o_sim["ext_high"]
        is_zero_noise_stay = c_sim["label"].startswith("stay 1 minus 20, sigma=0,")
        if is_zero_noise_stay:
            g_rel = abs(c_sim["growth_loss"] - o_sim["growth_loss"]) / abs(o_sim["growth_loss"])
            g_ok = g_rel <= 0.10
            g_mode = "deterministic_rel10pct"
        else:
            g_rel = abs(c_sim["growth_loss"] - o_sim["growth_loss"]) / abs(o_sim["growth_loss"])
            g_ok = o_sim["growth_low"] <= c_sim["growth_loss"] <= o_sim["growth_high"]
            g_mode = "interval95"
        simple_comparisons.append(
            {
                "label": c_sim["label"],
                "secondary_ext": c_sim["extinction"],
                "secondary_ext_ci": [c_sim["ext_low"], c_sim["ext_high"]],
                "orig_ext": o_sim["extinction"],
                "orig_ext_ci": [o_sim["ext_low"], o_sim["ext_high"]],
                "ext_in_orig_ci95": e_in,
                "secondary_growth_loss": c_sim["growth_loss"],
                "secondary_growth_ci": [c_sim["growth_low"], c_sim["growth_high"]],
                "orig_growth_loss": o_sim["growth_loss"],
                "orig_growth_ci": [o_sim["growth_low"], o_sim["growth_high"]],
                "growth_agree": g_ok,
                "growth_mode": g_mode,
                "growth_rel_diff": g_rel,
            }
        )

    return {
        "cells": cell_comparisons,
        "main_extinction": {
            "secondary_effect": c_ext["effect"],
            "secondary_ci95": c_ext["ci95"],
            "secondary_jeffreys95": c_ext["jeffreys_interval95"],
            "secondary_n80": c_ext["n80_per_cell"],
            "secondary_n80_10_90": c_ext["n80_sensitivity_10_90"],
            "orig_effect": o_ext_m["effect"],
            "orig_ci95": o_ext_m["ci95"],
            "orig_jeffreys95": o_ext_m["jeffreys_interval95"],
            "orig_n80": o_ext_m["n80_per_cell"],
            "orig_n80_10_90": o_ext_m["n80_sensitivity_10_90"],
            "fix1_effect": None if fix1_effects is None else fix1_effects["extinction"]["effect"],
            "fix1_ci95": None if fix1_effects is None else fix1_effects["extinction"]["ci95"],
            "fix1_n80": None if fix1_effects is None else fix1_effects["extinction"]["n80_per_cell"],
            "in_orig_wald_ci95": ext_int_in_wald,
            "in_orig_jeffreys_ci95": ext_int_in_jeff,
            "same_outcome_label": ext_label_same,
            "n80_rel_diff": n80_ext_rel,
            "n80_within_10pct": n80_ext_rel <= 0.10,
            "n80_in_orig_10_90_range": n80_ext_in_sens,
        },
        "main_growth": {
            "secondary_effect": c_gr["effect"],
            "secondary_ci95": c_gr["ci95"],
            "secondary_n80": c_gr["n80_per_cell"],
            "orig_effect": o_gr_m["effect"],
            "orig_ci95": o_gr_m["ci95"],
            "orig_n80": o_gr_m["n80_per_cell"],
            "in_orig_wald_ci95": gr_int_in_wald,
            "n80_rel_diff": n80_gr_rel,
            "n80_within_10pct": n80_gr_rel <= 0.10,
        },
        "simple": simple_comparisons,
    }


def main() -> int:
    started_ts = shell_date()
    t0 = time.monotonic()

    # Step 0a: Confirm Step-A validation gates passed (`protocol.md` line 139)
    val_path = HERE / "validation_rerun_results.json"
    if not val_path.exists():
        raise RuntimeError("Missing validation_rerun_results.json; Step-A validation must pass first.")
    val_data = json.loads(val_path.read_text(encoding="utf-8"))
    if val_data.get("overall_status") != "PASS":
        raise RuntimeError("Step-A validation did not PASS; refusing to run pilot.")

    # Step 0b: Run structural & Rule 7 pre-validation unit tests before pilot execution
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromModule(test_sim_world),
            loader.loadTestsFromModule(test_pilot_rerun),
        ]
    )
    unit_res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not unit_res.wasSuccessful():
        raise RuntimeError("Pre-pilot unit/Rule-7 validation tests failed.")

    cells_dir = HERE / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict[str, object]] = {}
    growth_arrays: dict[str, np.ndarray] = {}
    roots_out: dict[str, dict[str, object]] = {}

    t_crit_growth = float(stats.t.ppf(0.975, N_REPS - 1))
    t_crit_D = float(stats.t.ppf(0.975, THRESH_PAIRS - 1))

    for sigma in (0.0, 0.3, 0.6):
        for phi in (0.0, 0.6):
            for stay in (1.0, 5.0, 20.0):
                idx = len(cells)
                cell_seed = PILOT_SEED_BASE + idx * 100
                key = f"s{sigma:g}_p{phi:g}_t{stay:g}"

                mdl = build_geometric_ring([EXPOSED_STAY, stay])
                lam_const, v_const, _ = mdl.perron_at_mean_host(LOG_R0)
                init_lr = largest_remainder_allocate(INIT_TOTAL, v_const)
                init_modal = modal_stage_allocate_historical(INIT_TOTAL, v_const)

                # 1. Forced lognormal AR(1) host schedules & forward individual branching
                init_pop = np.broadcast_to(init_lr, (N_REPS, mdl.dim)).copy()
                hosts_500 = generate_lognormal_ar1_hosts(
                    n_steps=HORIZON, n_reps=N_REPS, sigma=sigma, phi=phi, seed=cell_seed
                )
                sim = run_individual_world(
                    mdl, init_pop, n_steps=HORIZON, seed=cell_seed + 1, hosts=hosts_500, tilt_y=LOG_R0
                )
                ext_mask = sim["extinction_step"] <= HORIZON
                if not np.array_equal(ext_mask, sim["totals"][HORIZON] == 0):
                    raise RuntimeError(f"Absorption invariant violated in cell {key}.")

                k_ext = int(np.sum(ext_mask))
                p_ext = float(k_ext) / float(N_REPS)
                w_lo, w_hi = wilson_ci95(k_ext, N_REPS)

                pgf_ext_lr = (
                    float(
                        multitype_branching_extinction_curve(
                            (EXPOSED_STAY, stay), LOG_R0, (int(init_lr[0]), int(init_lr[1])), HORIZON
                        )[HORIZON]
                    )
                    if sigma == 0.0
                    else None
                )
                pgf_ext_modal = (
                    float(
                        multitype_branching_extinction_curve(
                            (EXPOSED_STAY, stay), LOG_R0, (int(init_modal[0]), int(init_modal[1])), HORIZON
                        )[HORIZON]
                    )
                    if sigma == 0.0
                    else None
                )

                # 2. All-schedule conditional-mean invasion growth over steps 0..100 and 0..500
                fin_g = compute_finite_conditional_growth(
                    mdl, hosts_500, tilt_y=LOG_R0, init_weights=init_lr, horizons=(100, 500)
                )
                g100 = fin_g[100]
                g500 = fin_g[500]
                if sigma == 0.0:
                    if float(np.ptp(g100)) > 1e-14:
                        raise RuntimeError(f"Unexpected spread in constant-host growth for {key}.")
                    g100_mean = float(g100[0])
                    g100_se = 0.0
                else:
                    g100_mean = float(np.mean(g100))
                    g100_se = sample_se(g100)
                g100_half = t_crit_growth * g100_se

                # Deterministic reference growths under modal-stage rounding and exact Perron start
                h_single_const = np.ones((100, 1, 1), dtype=np.float64)
                g100_const_modal = float(
                    compute_finite_conditional_growth(
                        mdl, h_single_const, tilt_y=LOG_R0, init_weights=init_modal, horizons=(100,)
                    )[100][0]
                )
                g100_const_perron = math.log(lam_const)

                # 3. Long-run threshold D on 32 independent antithetic pairs (32,768 + 2,048 steps)
                sigma2_K = sigma * sigma * exact_ring_small_noise_K([EXPOSED_STAY, stay], phi)
                hosts_thresh = generate_lognormal_ar1_hosts(
                    n_steps=THRESH_RECORD_STEPS + THRESH_BURN_STEPS,
                    n_reps=THRESH_PAIRS,
                    sigma=sigma,
                    phi=phi,
                    seed=cell_seed + 2,
                    antithetic=True,
                )
                d_res = solve_threshold_D(
                    mdl,
                    hosts_thresh,
                    burn_steps=THRESH_BURN_STEPS,
                    n_pairs=THRESH_PAIRS,
                    cv_weight=1.0 / (EXPOSED_STAY + stay),
                    init_offset=sigma2_K,
                    tol=1e-10,
                )
                root_res = float(d_res["root_residual"])
                if root_res > 1e-7:
                    raise RuntimeError(f"Root residual {root_res:.3e} exceeded 1e-7 in cell {key}.")
                if sigma == 0.0:
                    d_val = 0.0
                    d_se = 0.0
                else:
                    d_val = float(d_res["D"])
                    d_se = float(d_res["se"])
                d_half = t_crit_D * d_se

                alive_100 = sim["totals"][100] > 0
                surv_g100 = (
                    float(np.mean(np.log(sim["totals"][100, alive_100].astype(np.float64) / INIT_TOTAL) / 100.0))
                    if np.any(alive_100)
                    else None
                )
                path_means = np.mean(hosts_500[:, :, 0], axis=0)
                path_log_vars = np.var(np.log(hosts_500[:, :, 0]), axis=0, ddof=1)

                row = {
                    "cell": key,
                    "sigma": sigma,
                    "phi": phi,
                    "stay": stay,
                    "seed_hosts": cell_seed,
                    "seed_demography": cell_seed + 1,
                    "seed_threshold": cell_seed + 2,
                    "n": N_REPS,
                    "extinct": k_ext,
                    "extinction": p_ext,
                    "extinction_low": w_lo,
                    "extinction_high": w_hi,
                    "exact_pgf_extinction_lr": pgf_ext_lr,
                    "exact_pgf_extinction_modal": pgf_ext_modal,
                    "growth100": g100_mean,
                    "growth100_se": g100_se,
                    "growth100_low": g100_mean - g100_half,
                    "growth100_high": g100_mean + g100_half,
                    "growth500": float(np.mean(g500)),
                    "growth100_const_modal_init": g100_const_modal,
                    "growth100_const_perron_init": g100_const_perron,
                    "D": d_val,
                    "D_se": d_se,
                    "D_low": d_val - d_half,
                    "D_high": d_val + d_half,
                    "sigma2_K": sigma2_K,
                    "D_minus_sigma2_K": d_val - sigma2_K,
                    "root_residual": root_res,
                    "initial_stage0": int(init_lr[0]),
                    "initial_stage1": int(init_lr[1]),
                    "host_mean": float(np.mean(path_means)),
                    "host_mean_between_sd": float(np.std(path_means, ddof=1)),
                    "host_log_var_mean": float(np.mean(path_log_vars)),
                    "survivors_at100": int(np.sum(alive_100)),
                    "survivor_only_growth100": surv_g100,
                    "max_total_count": int(np.max(sim["totals"])),
                    "restricted_mean_persistence": float(np.mean(np.minimum(sim["extinction_step"], HORIZON))),
                }
                cells[key] = row
                growth_arrays[key] = g100
                roots_out[key] = {
                    "D": d_val,
                    "se": d_se,
                    "root_residual": root_res,
                    "iterations": int(d_res["iterations"]),
                    "slope": float(d_res["slope"]),
                    "half_diff": float(d_res["half_diff"]),
                    "half_se": float(d_res["half_se"]),
                    "samples": np.asarray(d_res["samples"], dtype=np.float64).tolist(),
                }

                # Checkpoint completed cell immediately ("write files as you go")
                (cells_dir / f"{key}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(
                    f"[{idx+1:2d}/18] {key:13s} | ext={k_ext:3d}/400 ({p_ext:.3f} [{w_lo:.3f}, {w_hi:.3f}]) "
                    f"| g100={g100_mean:+.6f} | D={d_val:.6f} | s2K={sigma2_K:.6f}",
                    flush=True,
                )

    # Main 4-cell interaction at phi = 0.6
    main_keys = ["s0.6_p0.6_t1", "s0_p0.6_t1", "s0.6_p0.6_t20", "s0_p0.6_t20"]
    ext_main = compute_four_cell_interaction(
        cells, growth_arrays, main_keys, [1.0, -1.0, -1.0, 1.0], "extinction", jeffreys_seed=JEFFREYS_SEED
    )
    gr_main = compute_four_cell_interaction(
        cells, growth_arrays, main_keys, [-1.0, 1.0, 1.0, -1.0], "growth100", jeffreys_seed=JEFFREYS_SEED
    )

    # 12 registered simple contrasts
    simple_specs: list[tuple[str, str, str]] = []
    for phi in (0.0, 0.6):
        for stay in (1.0, 5.0, 20.0):
            simple_specs.append(
                (f"sigma .6 minus 0, phi={phi:g}, stay={stay:g}", f"s0.6_p{phi:g}_t{stay:g}", f"s0_p{phi:g}_t{stay:g}")
            )
    for sigma in (0.0, 0.3, 0.6):
        for phi in (0.0, 0.6):
            simple_specs.append(
                (f"stay 1 minus 20, sigma={sigma:g}, phi={phi:g}", f"s{sigma:g}_p{phi:g}_t1", f"s{sigma:g}_p{phi:g}_t20")
            )
    simple_list = [compute_simple_two_cell_contrast(lbl, cells[a], cells[b]) for lbl, a, b in simple_specs]

    effects = {"extinction": ext_main, "growth": gr_main, "simple": simple_list}
    comparison = compare_against_original(cells, effects)

    finished_ts = shell_date()
    elapsed = time.monotonic() - t0

    # Write CSV & JSON artifacts
    csv_fields = list(next(iter(cells.values())).keys())
    with (HERE / "pilot_rerun_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(cells.values())

    (HERE / "pilot_rerun_roots.json").write_text(json.dumps(roots_out, indent=2), encoding="utf-8")

    full_payload = {
        "started": started_ts,
        "completed": finished_ts,
        "elapsed_seconds": elapsed,
        "pilot_seed_base": PILOT_SEED_BASE,
        "jeffreys_seed": JEFFREYS_SEED,
        "replicates_per_cell": N_REPS,
        "total_trajectories": 18 * N_REPS,
        "horizon": HORIZON,
        "exposed_stay": EXPOSED_STAY,
        "initial_total": INIT_TOTAL,
        "log_R0": LOG_R0,
        "unit_tests_run": unit_res.testsRun,
        "unit_tests_passed": unit_res.wasSuccessful(),
        "cells": cells,
        "effects": effects,
        "comparison_vs_pilot_md": comparison,
    }
    (HERE / "pilot_rerun_results.json").write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    print(f"\nCompleted 18-cell x 400-rep pilot rerun in {elapsed:.2f}s ({started_ts} -> {finished_ts}).")
    print(
        f"Main extinction interaction: {ext_main['effect']:.6f} "
        f"Wald 95% [{ext_main['ci95'][0]:.6f}, {ext_main['ci95'][1]:.6f}], "
        f"Jeffreys 95% [{ext_main['jeffreys_interval95'][0]:.6f}, {ext_main['jeffreys_interval95'][1]:.6f}], "
        f"n80={ext_main['n80_per_cell']} (10-90%: {ext_main['n80_sensitivity_10_90']})"
    )
    print(
        f"Main early-growth loss interaction: {gr_main['effect']:.6f} "
        f"Wald 95% [{gr_main['ci95'][0]:.6f}, {gr_main['ci95'][1]:.6f}], "
        f"n80={gr_main['n80_per_cell']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
