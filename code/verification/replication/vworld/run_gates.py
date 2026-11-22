#!/usr/bin/env python3
"""Independent execution of virtualworld validation gates (i)-(v) for r35-vworld-A.

Runs structural unit tests first (`protocol.md` line 90 & study rules Rule 7),
then executes Gates (i)-(v) sequentially using clean-room `sim_world.py` and
fresh independent seeds (`SEED_BASE = 351000`), stopping immediately if any gate
fails. Also cross-tabulates every check against primary's `validation.md` /
`results/validation.json` and the analytical reference values.
"""

from __future__ import annotations

import ast
import csv
import itertools
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
from scipy import linalg

from sim_world import (
    KernelSpec,
    StageSpec,
    TransferSpec,
    WorldModel,
    build_geometric_ring,
    exact_matrix_resolvent_K,
    exact_ring_small_noise_K,
    generate_bounded_refresh_hosts,
    generate_iid_gamma_hosts,
    generate_lognormal_ar1_hosts,
    memA_two_stage_coefficient,
    multitype_branching_extinction_curve,
    propagate_invasion_products,
    roerdink_exact_gamma_D,
    run_individual_world,
    sample_se,
    solve_threshold_D,
)
import test_sim_world

PROB_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()  # problems/dobson-r0
ORIG_VWORLD = PROB_ROOT / "virtualworld"
R16_CSV = PROB_ROOT / "studies" / "engine-R16" / "R16_results.csv"
MEMB_CSV = PROB_ROOT / "studies" / "memory-memB" / "memB_tables.csv"

SEED_BASE = 351000


def shell_date() -> str:
    return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S %Z"], text=True).strip()


def add_value_check(
    rows: list[dict[str, object]],
    case: str,
    actual: float,
    reference: float,
    tolerance: float,
    analytical: float | str | None = None,
    **meta: object,
) -> None:
    err = abs(float(actual) - float(reference))
    rows.append(
        {
            "case": case,
            "kind": "value",
            "actual": float(actual),
            "reference": float(reference),
            "analytical": float(reference) if analytical is None else analytical,
            "error": err,
            "tolerance": float(tolerance),
            "passed": bool(err <= float(tolerance)),
            **meta,
        }
    )


def add_ceiling_check(
    rows: list[dict[str, object]],
    case: str,
    observed_se: float,
    ceiling: float,
    **meta: object,
) -> None:
    shortfall = max(0.0, float(observed_se) - float(ceiling))
    rows.append(
        {
            "case": case,
            "kind": "precision",
            "actual": float(observed_se),
            "reference": float(ceiling),
            "analytical": f"<= {float(ceiling):.9g}",
            "error": shortfall,
            "tolerance": float(ceiling),
            "passed": bool(float(observed_se) <= float(ceiling)),
            **meta,
        }
    )


def add_direction_check(
    rows: list[dict[str, object]],
    case: str,
    delta: float,
    paired_se: float = 0.0,
    floor: float = 0.0,
    analytical: float | str = "> 0",
    **meta: object,
) -> None:
    req = max(float(floor), 4.0 * float(paired_se))
    shortfall = max(0.0, req - float(delta))
    rows.append(
        {
            "case": case,
            "kind": "direction",
            "actual": float(delta),
            "reference": "positive",
            "analytical": analytical,
            "error": shortfall,
            "tolerance": req,
            "passed": bool(float(delta) > req),
            "se": float(paired_se),
            **meta,
        }
    )


# ---------------------------------------------------------------------------
# Gate (i): small-noise second-order coefficient, ring band, one-step means
# ---------------------------------------------------------------------------


def run_gate_i() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ring_configs = ([2, 5], [1, 2, 7], [2, 3, 4, 5])
    for idx, stays in enumerate(ring_configs):
        mdl = build_geometric_ring(stays)
        total_T = float(sum(stays))
        mean_mat = mdl.expectation_matrix()

        # One-step empirical individual mean checks (50,000 independent parents/state)
        rng = np.random.default_rng(SEED_BASE + 100 + idx)
        n_ind = 50_000
        h_one = np.ones((n_ind, 1), dtype=np.float64)
        for src in range(mdl.dim):
            init = np.zeros((n_ind, mdl.dim), dtype=np.int64)
            init[:, src] = 1
            nxt = mdl.sample_step(init, h_one, rng)
            for dst in range(mdl.dim):
                emp_mean = float(np.mean(nxt[:, dst]))
                emp_se = float(np.std(nxt[:, dst], ddof=1) / math.sqrt(n_ind))
                exact_entry = float(mean_mat[dst, src])
                add_value_check(
                    rows,
                    f"individual mean {stays} {src}->{dst}",
                    emp_mean,
                    exact_entry,
                    4.0 * emp_se + 0.002,
                    analytical=exact_entry,
                    se=emp_se,
                )

        for phi in (0.0, 0.6):
            tag = f"{stays}, phi={phi}"
            K_ring = exact_ring_small_noise_K(stays, phi)
            K_res = exact_matrix_resolvent_K(mdl, phi)
            add_value_check(rows, f"resolvent {tag}", K_res, K_ring, 1e-10, analytical=K_ring)

            band_lo = 1.0 / (2.0 * total_T)
            band_hi = (1.0 + 2.0 * phi / (1.0 - phi)) / (2.0 * total_T)
            clipped_K = float(np.clip(K_ring, band_lo, band_hi))
            slack = min(K_ring - band_lo, band_hi - K_ring)
            add_value_check(
                rows,
                f"band {tag}",
                clipped_K,
                K_ring,
                1e-10,
                analytical=f"[{band_lo:.9g}, {band_hi:.9g}] (K={K_ring:.9g})",
                band_lo=band_lo,
                band_hi=band_hi,
                band_slack=slack,
            )

            sigma_sm = 0.03
            h_sched = generate_lognormal_ar1_hosts(
                n_steps=65_536 + 2_048,
                n_reps=32,
                sigma=sigma_sm,
                phi=phi,
                seed=SEED_BASE + 120 + idx,
                antithetic=True,
            )
            r_pairs, _, _ = propagate_invasion_products(
                mdl, h_sched, tilt_y=0.0, burn_steps=2_048, n_pairs=32, cv_weight=1.0 / total_T
            )
            k_samples = -total_T * r_pairs / (sigma_sm * sigma_sm)
            k_mean = float(np.mean(k_samples))
            k_se = sample_se(k_samples)
            add_value_check(
                rows,
                f"linear limit {tag}",
                k_mean,
                K_ring,
                0.05 * K_ring + 4.0 * k_se,
                analytical=K_ring,
                se=k_se,
            )
            add_ceiling_check(rows, f"precision {tag}", k_se, 0.03 * K_ring, se=k_se)
    return rows


# ---------------------------------------------------------------------------
# Gate (ii): six R16 cells (base, shape, alternatives, patches, mortality, delay)
# ---------------------------------------------------------------------------


def compile_r16_world_model(row: dict[str, str]) -> WorldModel:
    E, S, A, P, M = ast.literal_eval(row["bits"])
    fine = row["fine"] == "True"
    dt = 0.25 if fine else 1.0
    stay_steps = float(row["stay"]) / dt

    st_exposed = StageSpec(
        name="exposed",
        kernel=KernelSpec.geometric(1.0 / dt),
        mortality=0.5 * dt if E else 0.0,
        scarcity_mode="linear_delay" if E else "waste",
    )
    st_long = StageSpec(
        name="long",
        kernel=KernelSpec.erlang_chain(stay_steps, 4) if S else KernelSpec.geometric(stay_steps),
        mortality=(1.0 - 0.99**dt) if M else 0.0,
        host_driver=0,
        scarcity_mode="waste",
    )
    n_drv = 1 + A
    routes = [TransferSpec(0, 1, host_driver=d, share=1.0 / n_drv) for d in range(n_drv)] + [
        TransferSpec(1, 0, is_birth=True)
    ]
    n_pat = 4 if P else 1
    disp = (1.0 - 0.85**dt) if P else 0.0
    return WorldModel(
        stages=[st_exposed, st_long],
        transfers=routes,
        n_drivers=n_drv,
        n_patches=n_pat,
        dispersal_rate=disp,
        mobile_stage=1 if P else None,
    )


def explicit_r16_reference_matrix(row: dict[str, str], host_vec: np.ndarray) -> np.ndarray:
    """Closed-form critical expectation matrix built directly from R16 mathematical rules."""
    E, S, A, P, M = ast.literal_eval(row["bits"])
    fine = row["fine"] == "True"
    dt = 0.25 if fine else 1.0
    stay_steps = float(row["stay"]) / dt
    b_crit = float(row["critical_birth_scale"])

    n_drv = 1 + A
    n_pat = 4 if P else 1
    k_long = 4 if S else 1
    k_loc = 1 + k_long
    dim = n_pat * k_loc

    q0 = dt
    d0 = 0.5 * dt if E else 0.0
    surv_long = (0.99**dt) if M else 1.0
    q1 = (4.0 / stay_steps) if S else (1.0 / stay_steps)

    pre_disp = np.zeros((dim, dim), dtype=np.float64)
    for p in range(n_pat):
        off = p * k_loc
        h_patch = host_vec[p * n_drv : (p + 1) * n_drv]
        if E:
            c0 = q0 - d0
            pre_disp[off + 0, off + 0] = 1.0 - d0 - c0 * h_patch[0]
            pre_disp[off + 1, off + 0] = c0 * h_patch[0]
        else:
            h_eff = float(np.mean(h_patch))
            pre_disp[off + 0, off + 0] = 1.0 - q0
            pre_disp[off + 1, off + 0] = q0 * h_eff

        for j in range(k_long):
            pre_disp[off + 1 + j, off + 1 + j] = surv_long * (1.0 - q1)
            if j < k_long - 1:
                pre_disp[off + 2 + j, off + 1 + j] = surv_long * q1
            else:
                pre_disp[off + 0, off + 1 + j] = surv_long * q1 * b_crit

    if not P:
        return pre_disp

    m_move = 1.0 - 0.85**dt
    island = (1.0 - m_move) * np.eye(n_pat) + (m_move / n_pat) * np.ones((n_pat, n_pat))
    move_op = np.eye(dim, dtype=np.float64)
    for j in range(k_long):
        ids = np.arange(n_pat) * k_loc + (1 + j)
        move_op[np.ix_(ids, ids)] = island
    return move_op @ pre_disp


def r16_analytical_D2(mdl: WorldModel, sigma: float, phi_step: float) -> float:
    """Exact second-order resolvent coefficient D2 = C / b for independent identical bounded drivers."""
    y0 = mdl.critical_tilt_mean_host()
    _, v, w = mdl.perron_at_mean_host(y0)
    A_mat = mdl.expectation_matrix(None, tilt_y=y0)
    Q = np.eye(mdl.dim) - np.outer(v, w)
    # Note: R16 defines D2 with standardized variance sigma^2 (small-amplitude expansion in sigma)
    # Let's check R16_results.csv: lifecycle_engine sensitivities() uses sigma, not sqrt(expm1(sigma^2)).
    C_tot = 0.0
    for ch in range(mdl.n_channels):
        B_ch = (mdl.U1[ch] + math.exp(y0) * mdl.F1[ch]) * sigma
        c_ch = float(w @ B_ch @ v)
        sol = linalg.solve(np.eye(mdl.dim) - phi_step * (A_mat @ Q), Q @ (B_ch @ v))
        C_tot += 0.5 * c_ch * c_ch - phi_step * float(w @ B_ch @ sol)
    b_birth = float(w @ (math.exp(y0) * mdl.F0) @ v)
    return C_tot / b_birth


def run_gate_ii() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with R16_CSV.open(newline="", encoding="utf-8") as fh:
        r16_archive = {r["id"]: r for r in csv.DictReader(fh)}

    target_ids = ["0c-00000", "0c-01000", "0c-00100", "0c-00010", "0c-00001", "0f-10000"]
    for idx, cid in enumerate(target_ids):
        row = r16_archive[cid]
        mdl = compile_r16_world_model(row)
        fine = row["fine"] == "True"
        dt = 0.25 if fine else 1.0
        sig = float(row["sigma"])
        phi_step = float(row["phi"]) ** dt

        # Exhaustive bounded-state corner check + CSV invariant checks
        var_h = math.expm1(sig * sig)
        low_lv = 0.05
        high_lv = 1.0 + var_h / (1.0 - low_lv)
        corners = np.array(list(itertools.product([low_lv, high_lv], repeat=mdl.n_channels)), dtype=np.float64)
        y_crit = mdl.critical_tilt_mean_host()
        world_mats = mdl.expectation_matrix(corners, tilt_y=y_crit)
        ref_mats = np.stack([explicit_r16_reference_matrix(row, c_vec) for c_vec in corners], axis=0)
        max_mat_err = float(np.max(np.abs(world_mats - ref_mats)))
        d2_calc = r16_analytical_D2(mdl, sig, phi_step)
        d2_csv = float(row["D2"])
        if abs(d2_calc - d2_csv) > 1e-11:
            raise RuntimeError(f"Analytical D2 mismatch for {cid}: {d2_calc} vs {d2_csv}")

        add_value_check(rows, f"{cid} matrix", max_mat_err, 0.0, 1e-11, analytical=0.0, d2_exact=d2_calc)

        mult = 4 if fine else 1
        rec_steps = 32_768 * mult
        burn_steps = 2_048 * mult
        h_paths = generate_bounded_refresh_hosts(
            n_steps=rec_steps + burn_steps,
            n_reps=32,
            sigma=sig,
            phi=phi_step,
            seed=SEED_BASE + 210 + idx,
            n_drivers=mdl.n_drivers,
            n_patches=mdl.n_patches,
            low_level=0.05,
        )
        res = solve_threshold_D(mdl, h_paths, burn_steps=burn_steps, tol=1e-9)
        d_val = float(res["D"])
        d_se = float(res["se"])
        ref_d = float(row["D"])
        ref_se = float(row["se"])
        comb_se = math.hypot(d_se, ref_se)

        add_value_check(
            rows,
            f"{cid} D",
            d_val,
            ref_d,
            4.0 * comb_se + 0.002,
            analytical=f"D2={d2_csv:.9g} (R16_MC={ref_d:.9g})",
            se=d_se,
            reference_se=ref_se,
            d2_exact=d2_csv,
        )
        ceil_se = max(0.003, 0.05 * abs(d_val))
        add_ceiling_check(rows, f"{cid} precision", d_se, ceil_se, se=d_se)
        add_value_check(rows, f"{cid} root", float(res["root_residual"]), 0.0, 1e-7, analytical=0.0)
        add_value_check(
            rows,
            f"{cid} last half",
            float(res["half_diff"]),
            0.0,
            4.0 * float(res["half_se"]) + 0.002,
            analytical=0.0,
            se=float(res["half_se"]),
        )
    return rows


# ---------------------------------------------------------------------------
# Gate (iii): Roerdink's exact gamma-host model across 3 noise levels x 4 stays
# ---------------------------------------------------------------------------


def run_gate_iii() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, p_shape in enumerate((2.0, 1.0, 0.25)):
        h_gamma = generate_iid_gamma_hosts(
            n_steps=65_536 + 2_048, n_reps=64, shape_p=p_shape, seed=SEED_BASE + 300 + idx
        )
        prev_exact: float | None = None
        prev_samples: np.ndarray | None = None

        for s_stay in (0.2, 0.5, 0.8, 0.95):
            lbl = f"p={p_shape}, s={s_stay}"
            d_quad = roerdink_exact_gamma_D(s_stay, p_shape, refined=False)
            d_quad_ref = roerdink_exact_gamma_D(s_stay, p_shape, refined=True)
            add_value_check(rows, f"{lbl} quadrature", d_quad, d_quad_ref, 1e-8, analytical=d_quad_ref)

            tau_steady = 1.0 / (1.0 - s_stay)
            mdl = build_geometric_ring([1.0, tau_steady])
            res = solve_threshold_D(
                mdl,
                h_gamma,
                burn_steps=2_048,
                cv_weight=1.0 / (1.0 + tau_steady),
                init_offset=d_quad,
                tol=1e-10,
            )
            d_sim = float(res["D"])
            d_se = float(res["se"])
            d_samp = np.asarray(res["samples"], dtype=np.float64)

            add_value_check(rows, f"{lbl} D", d_sim, d_quad, 4.0 * d_se + 0.0005, analytical=d_quad, se=d_se)
            add_value_check(rows, f"{lbl} root", float(res["root_residual"]), 0.0, 1e-7, analytical=0.0)

            if prev_exact is not None and prev_samples is not None:
                exact_drop = prev_exact - d_quad
                add_direction_check(
                    rows, f"{lbl} exact decrease", exact_drop, floor=1e-8, analytical=exact_drop
                )
                sim_drops = prev_samples - d_samp
                add_direction_check(
                    rows,
                    f"{lbl} simulated decrease",
                    float(np.mean(sim_drops)),
                    paired_se=sample_se(sim_drops),
                    analytical=exact_drop,
                )

            prev_exact = d_quad
            prev_samples = d_samp
    return rows


# ---------------------------------------------------------------------------
# Gate (iv): memA coefficient and memB positive-memory direction
# ---------------------------------------------------------------------------


def run_gate_iv() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with MEMB_CSV.open(newline="", encoding="utf-8") as fh:
        memb_rows = list(csv.DictReader(fh))

    for s_idx, sigma in enumerate((0.3, 0.6)):
        # Common Gaussian innovations across stays AND phi (`protocol.md` line 123)
        common_seed = SEED_BASE + 400 + s_idx
        for phi in (0.6, 0.9):
            h_mem = generate_lognormal_ar1_hosts(
                n_steps=65_536 + 2_048,
                n_reps=32,
                sigma=sigma,
                phi=phi,
                seed=common_seed,
                antithetic=True,
            )
            prev_samples: np.ndarray | None = None
            prev_s2k: float | None = None
            prev_ref_d: float | None = None

            for tau in (2.0, 5.0, 10.0):
                lbl = f"sigma={sigma}, phi={phi}, tau={tau}"
                k_mema = memA_two_stage_coefficient(tau, phi)
                k_ring = exact_ring_small_noise_K([1.0, tau], phi)
                add_value_check(rows, f"{lbl} memA", k_ring, k_mema, 1e-10, analytical=k_mema)

                mdl = build_geometric_ring([1.0, tau])
                s2k = sigma * sigma * k_mema
                res = solve_threshold_D(
                    mdl,
                    h_mem,
                    burn_steps=2_048,
                    n_pairs=32,
                    cv_weight=1.0 / (1.0 + tau),
                    init_offset=s2k,
                    tol=1e-10,
                )
                matched = [
                    r
                    for r in memb_rows
                    if r["model"] == "lognormal"
                    and int(r["ring"]) == 2
                    and abs(float(r["amplitude"]) - sigma) < 1e-8
                    and abs(float(r["phi_input"]) - phi) < 1e-8
                    and abs(float(r["tau"]) - tau) < 1e-8
                ]
                if len(matched) != 1:
                    raise RuntimeError(f"Expected 1 memB match for {lbl}, got {len(matched)}")
                ref_d = float(matched[0]["D"])
                ref_se = float(matched[0]["SE_D"])

                d_sim = float(res["D"])
                d_se = float(res["se"])
                d_samp = np.asarray(res["samples"], dtype=np.float64)

                add_value_check(
                    rows,
                    f"{lbl} D",
                    d_sim,
                    ref_d,
                    4.0 * math.hypot(d_se, ref_se) + 0.001,
                    analytical=f"s2*K={s2k:.9g} (memB_MC={ref_d:.9g})",
                    se=d_se,
                    reference_se=ref_se,
                    s2k=s2k,
                )
                add_value_check(rows, f"{lbl} root", float(res["root_residual"]), 0.0, 1e-7, analytical=0.0)

                if prev_samples is not None and prev_s2k is not None and prev_ref_d is not None:
                    diffs = prev_samples - d_samp
                    add_direction_check(
                        rows,
                        f"{lbl} decrease",
                        float(np.mean(diffs)),
                        paired_se=sample_se(diffs),
                        analytical=f"ds2K={prev_s2k - s2k:.9g} (d_memB={prev_ref_d - ref_d:.9g})",
                    )

                prev_samples = d_samp
                prev_s2k = s2k
                prev_ref_d = ref_d
    return rows


# ---------------------------------------------------------------------------
# Gate (v): constant-host demographic persistence vs multitype PGF
# ---------------------------------------------------------------------------


def run_gate_v() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_traj = 10_000
    max_step = 120
    h_const = np.ones((max_step, n_traj, 1), dtype=np.float64)
    init_pop = np.zeros((n_traj, 2), dtype=np.int64)
    init_pop[:, 0] = 10

    for idx, log_r0 in enumerate((-0.1, 0.0, 0.1)):
        mdl = build_geometric_ring([2.0, 5.0])
        sim = run_individual_world(
            mdl, init_pop, max_step, seed=SEED_BASE + 500 + idx, hosts=h_const, tilt_y=log_r0
        )
        pgf = multitype_branching_extinction_curve((2.0, 5.0), log_r0, (10, 0), max_step)
        for step_h in (20, 60, 120):
            p_emp = float(np.mean(sim["extinction_step"] <= step_h))
            p_exact = float(pgf[step_h])
            binom_se = math.sqrt(p_exact * (1.0 - p_exact) / n_traj)
            add_value_check(
                rows,
                f"logR0={log_r0}, horizon={step_h}",
                p_emp,
                p_exact,
                4.0 * binom_se + 0.002,
                analytical=p_exact,
                se=binom_se,
            )
    return rows


# ---------------------------------------------------------------------------
# Cross-comparison with primary's `validation.md` / `validation.json`
# ---------------------------------------------------------------------------


def evaluate_agreement_with_independent(
    gates_out: dict[str, dict[str, object]],
) -> dict[str, object]:
    primary_json = json.loads((ORIG_VWORLD / "results" / "validation.json").read_text(encoding="utf-8"))
    primary_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for g_key, g_val in primary_json["gates"].items():
        for chk in g_val["checks"]:
            primary_lookup[(g_key, chk["case"])] = chk

    summary = {
        "total_checks": 0,
        "label_matches": 0,
        "point_estimate_checks": 0,
        "point_agreements": 0,
        "diagnostic_checks": 0,
        "disagreements": [],
    }

    for g_key, g_data in gates_out.items():
        for row in g_data["checks"]:
            case = str(row["case"])
            orig = primary_lookup[(g_key, case)]
            row["primary_actual"] = orig["actual"]
            row["primary_expected"] = orig["expected"]
            row["primary_tolerance"] = orig["tolerance"]
            row["primary_passed"] = orig["passed"]
            row["primary_se"] = orig.get("se")

            same_label = bool(row["passed"]) == bool(orig["passed"])
            summary["total_checks"] += 1
            if same_label:
                summary["label_matches"] += 1

            # Classify auxiliary numerical diagnostics (precision SE ceiling, Newton root residual,
            # matrix machine-eps norm, stationary half-run zero-difference diagnostic) vs scientific estimates
            is_diagnostic = (
                row["kind"] == "precision"
                or case.endswith(" root")
                or case.endswith(" matrix")
                or case.endswith(" last half")
            )
            if is_diagnostic:
                summary["diagnostic_checks"] += 1
                row["agreement_type"] = "diagnostic_label"
                row["agreed"] = same_label
                if not same_label:
                    summary["disagreements"].append((g_key, case, "outcome label mismatch"))
                continue

            summary["point_estimate_checks"] += 1
            my_val = float(row["actual"])
            cx_val = float(orig["actual"])
            cx_se = orig.get("se")

            in_ci95 = False
            ci_bounds = None
            if cx_se is not None and float(cx_se) > 0.0:
                half_w = 1.96 * float(cx_se)
                ci_bounds = [cx_val - half_w, cx_val + half_w]
                in_ci95 = (ci_bounds[0] <= my_val <= ci_bounds[1])

            if cx_val == 0.0:
                rel_diff = 0.0 if my_val == 0.0 else math.inf
                within_10pct = (my_val == 0.0)
            else:
                rel_diff = abs(my_val - cx_val) / abs(cx_val)
                within_10pct = (rel_diff <= 0.10)

            # Note: `validation.md` prints NO 95% interval column (except implicit SE for precision/tolerance);
            # check BOTH: within 10% relative (where no interval is printed in validation.md) OR inside 95% CI from SE.
            point_ok = bool(within_10pct or in_ci95)
            agreed = bool(same_label and point_ok)
            row["primary_ci95_from_se"] = ci_bounds
            row["in_independent_ci95"] = in_ci95
            row["rel_diff_vs_independent"] = rel_diff
            row["within_10pct_rel"] = within_10pct
            row["agreement_type"] = "point_estimate"
            row["agreed"] = agreed

            if agreed:
                summary["point_agreements"] += 1
            else:
                summary["disagreements"].append(
                    {
                        "gate": g_key,
                        "case": case,
                        "verification": my_val,
                        "studies": cx_val,
                        "primary_ci95": ci_bounds,
                        "rel_diff": rel_diff,
                        "same_label": same_label,
                    }
                )
    return summary


def main() -> int:
    started_ts = shell_date()
    t0 = time.monotonic()

    # Step 0: Run structural unit tests before executing gates (`protocol.md` line 90)
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_sim_world)
    runner = unittest.TextTestRunner(verbosity=1)
    unit_res = runner.run(suite)
    if not unit_res.wasSuccessful():
        print("STOP: Structural unit tests failed prior to scientific gates.", file=sys.stderr)
        return 1

    gates_spec = [
        ("i", run_gate_i),
        ("ii", run_gate_ii),
        ("iii", run_gate_iii),
        ("iv", run_gate_iv),
        ("v", run_gate_v),
    ]
    gates_out: dict[str, dict[str, object]] = {}
    overall_pass = True

    for g_key, g_fn in gates_spec:
        gt0 = time.monotonic()
        checks = g_fn()
        elapsed = time.monotonic() - gt0
        g_pass = bool(checks) and all(bool(c["passed"]) for c in checks)
        gates_out[g_key] = {
            "status": "PASS" if g_pass else "FAIL",
            "n_checks": len(checks),
            "seconds": elapsed,
            "checks": checks,
        }
        print(f"Gate {g_key}: {gates_out[g_key]['status']} ({len(checks)} checks in {elapsed:.2f}s)", flush=True)
        if not g_pass:
            overall_pass = False
            failed_cases = [c["case"] for c in checks if not c["passed"]]
            print(f"STOP at Gate {g_key}; failed cases: {failed_cases}", flush=True)
            break

    agreement = evaluate_agreement_with_independent(gates_out) if overall_pass else {}
    finished_ts = shell_date()
    total_sec = time.monotonic() - t0

    payload = {
        "started": started_ts,
        "completed": finished_ts,
        "seed_base": SEED_BASE,
        "unit_tests_run": unit_res.testsRun,
        "unit_tests_passed": unit_res.wasSuccessful(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "total_seconds": total_sec,
        "gates": gates_out,
        "agreement_summary": agreement,
    }
    out_path = HERE / "validation_rerun_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCompleted in {total_sec:.2f}s. Agreement summary: {json.dumps(agreement, indent=2)}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
