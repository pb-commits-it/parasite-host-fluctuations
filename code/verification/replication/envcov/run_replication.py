#!/usr/bin/env python3
"""End-to-end independent replication of case study `envcov` (`r36-rep-envcov`).

Run with:
    python3 -I verification/replication/envcov/run_replication.py

Zero imports from `casestudies/envcov/` or `studies/`.
Runs Rule 7 pre-analysis validation checks first before reading any empirical files.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW = ROOT / "data" / "raw"
ORIG_RESULTS = ROOT / "casestudies" / "envcov" / "results"
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import special

import data_and_covariates as dc
import lifecycle_solver as ls
import stats_methods as sm
import validate_before_use as vbu

DT_COARSE_DAYS = 4.4 * 30.4375  # 133.925 d
BORROWED_TICK_DAYS = 10.0 * 30.4375  # 304.375 d


def _solve_spec_worker(task: dict) -> dict:
    spec: ls.BorreliaSpec = task["spec"]
    res = ls.IndependentCycleModel(spec).solve_loss(
        steps=task.get("steps", 40000),
        burn=task.get("burn", 2000),
        pairs=task.get("pairs", 24),
        seed=task["seed"],
    )
    res["key"] = task["key"]
    return res


def _solve_coupled_worker(task: dict) -> dict:
    res = ls.solve_coupled_weather_cell(
        sigma_mouse=task["sigma"],
        kappa=task["kappa"],
        gamma=task["gamma"],
        site_climate=task["site_climate"],
        phi_mouse=task["phi_mouse"],
        phi_weather=task["phi_weather"],
        rho_hw=task["rho_hw"],
        weather_vals=task["weather"],
        duration_days=task["duration"],
        steps=40000,
        burn=2000,
        pairs=24,
        seed=task["seed"],
    )
    res["site"] = task["site"]
    res["gamma"] = task["gamma"]
    res["kappa"] = task["kappa"]
    return res


def write_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with (HERE / filename).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    t0_wall = time.time()
    date_str = subprocess.check_output(["date"], text=True).strip()
    date_utc = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
    print(f"=== Independent Replication r36-rep-envcov | {date_str} ===", flush=True)

    # 1. Rule 7 pre-analysis validation checks BEFORE reading any empirical data
    val_summary = vbu.run_all_validations(save_json=True)
    print(f"[STEP 1] Rule 7 validation passed ({val_summary['n_checks']} checks).", flush=True)

    # 2. Verify raw manifest and load raw datasets
    manifest_info = dc.verify_envcov_manifest(RAW / "envcov")
    if manifest_info["mismatches"]:
        raise RuntimeError(f"Raw manifest hash mismatch: {manifest_info['mismatches']}")
    print(
        f"[STEP 2] Verified {manifest_info['n_entries']} raw envcov files ({manifest_info['total_mb']:.2f} MB), 0 mismatches.",
        flush=True,
    )

    neon_bio = dc.load_neon_biological(RAW / "neon")
    cary_bio = dc.load_cary_biological(RAW / "cary_lyme")

    daymet_all, daymet_meta, cov_all = {}, {}, {}
    for p in sorted((RAW / "envcov" / "daymet").glob("*.csv")):
        loc = p.stem
        dm, meta = dc.parse_daymet_csv(p)
        daymet_all[loc] = dm
        daymet_meta[loc] = meta
        pw_path = RAW / "envcov" / "nasa_power" / f"{loc}.json"
        pw = dc.parse_nasa_power_json(pw_path) if pw_path.exists() else None
        cov_all[loc] = dc.compute_annual_covariates(dm, pw)

    # 3. Part 1 & Part 2A: NEON 6-site covariates, PCA1, correlations, and Freedman-Lane adjustments
    sigma_6 = np.array([neon_bio["sigma"][s] for s in dc.NEON_6], dtype=float)
    nip_6 = np.array([neon_bio["nip_site"][s] for s in dc.NEON_6], dtype=float)
    ntest_6 = np.array([neon_bio["tested"][s] for s in dc.NEON_6], dtype=int)

    neon_site_rows = []
    for idx, s in enumerate(dc.NEON_6):
        yrs = neon_bio["test_years"][s]
        m_tick = daymet_meta[f"{s}_tick"]
        m_tow = daymet_meta[f"{s}_tower"]
        row = {
            "site": s,
            "sigma": float(sigma_6[idx]),
            "tested": int(ntest_6[idx]),
            "positive": int(neon_bio["positive"][s]),
            "nip": float(nip_6[idx]),
            "r1": float(neon_bio["r1"][s]),
            "mean_index": float(neon_bio["mean_index"][s]),
            "mean_log_trapnights": float(neon_bio["mean_log_trapnights"][s]),
            "tick_to_tower_km": dc.great_circle_km(m_tick["lat"], m_tick["lon"], m_tow["lat"], m_tow["lon"]),
            **neon_bio["habitat"][s],
        }
        for ck in cov_all[f"{s}_tick"][yrs[0]].keys():
            row[ck] = float(np.mean([cov_all[f"{s}_tick"][y][ck] for y in yrs]))
        neon_site_rows.append(row)

    neon_pca = sm.fit_pca1_corrmat(
        np.array([[r[k] for k in dc.PRIMARY_COVS] for r in neon_site_rows], dtype=float),
        orient_col=0,
    )
    for r, sc in zip(neon_site_rows, neon_pca["scores"]):
        r["PC1_env"] = float(sc)

    neon_unadj = sm.exact_spearman_test_6(sigma_6, nip_6)
    wls_unadj = sm.quasibinomial_logit_wls(nip_6, ntest_6, sigma_6, None)

    neon_adjustments = [
        {
            "covariate": "unadjusted",
            "r": neon_unadj["rho"],
            "p_negative": neon_unadj["p_neg"],
            "p_positive": neon_unadj["p_pos"],
            "alpha": 0.05,
            "verdict": "not decided",
            **wls_unadj,
        }
    ]
    neon_correlations = []
    adj_keys = (
        ["PC1_env"]
        + dc.PRIMARY_COVS
        + ["P_gwettop_gs", "P_rh_gs", "P_ts_wmin", "forest_plot_frac", "elevation_m", "mean_index", "mean_log_trapnights"]
    )
    for k in adj_keys:
        zv = np.array([r[k] for r in neon_site_rows], dtype=float)
        fl = sm.freedman_lane_partial_test_6(nip_6, sigma_6, zv, rank_transform=True)
        wls = sm.quasibinomial_logit_wls(nip_6, ntest_6, sigma_6, zv)
        alpha = (0.05 / 6.0) if k in dc.PRIMARY_COVS else 0.05
        verdict = (
            "supports"
            if (fl["r_part"] < 0.0 and fl["p_neg"] < alpha)
            else ("contradicts" if (fl["r_part"] > 0.0 and fl["p_pos"] < alpha) else "not decided")
        )
        neon_adjustments.append({
            "covariate": k,
            "r": fl["r_part"],
            "p_negative": fl["p_neg"],
            "p_positive": fl["p_pos"],
            "alpha": alpha,
            "verdict": verdict,
            **wls,
        })
        neon_correlations.append({
            "covariate": k,
            "pearson_sigma": sm.pearson_corr(sigma_6, zv),
            "spearman_sigma": sm.spearman_corr(sigma_6, zv),
            "pearson_nip": sm.pearson_corr(nip_6, zv),
            "spearman_nip": sm.spearman_corr(nip_6, zv),
        })

    # Tower coordinate spatial check
    tower_mat = np.array(
        [[np.mean([cov_all[f"{s}_tower"][y][k] for y in neon_bio["test_years"][s]]) for k in dc.PRIMARY_COVS] for s in dc.NEON_6],
        dtype=float,
    )
    tower_pca = sm.fit_pca1_corrmat(tower_mat, orient_col=0)
    tower_fl = sm.freedman_lane_partial_test_6(nip_6, sigma_6, tower_pca["scores"], rank_transform=True)

    # NEON within-site T3 pooled slope (46 pairs)
    neon_t3_records = []
    for s in dc.NEON_6:
        for y, pval in sorted(neon_bio["yearly_nip"][s].items()):
            idx_prev = neon_bio["index"].get((s, y - 1), 0.0)
            if pval > 0.0 and idx_prev > 0.0:
                c_vec = [cov_all[f"{s}_tick"][y][k] for k in dc.PRIMARY_COVS]
                neon_t3_records.append({
                    "site": s,
                    "mouse_year": y - 1,
                    "nymph_year": y,
                    "x": math.log(idx_prev),
                    "y": math.log(pval),
                    "PC1_env": sm.project_pca1(c_vec, neon_pca),
                    "effort": math.log(neon_bio["effort"][(s, y - 1)]),
                    **cov_all[f"{s}_tick"][y],
                })

    neon_t3_results = {}
    for k in ["unadjusted", "PC1_env"] + dc.PRIMARY_COVS + ["effort"]:
        cols = [] if k == "unadjusted" else [k]
        res_reg = sm.neon_within_site_bootstrap(neon_t3_records, dc.NEON_6, cols, n_boot=2000, seed=(20261114, 3))
        res_ind = sm.neon_within_site_bootstrap(neon_t3_records, dc.NEON_6, cols, n_boot=2000, seed=99202601)
        neon_t3_results[k] = {"registered_seed": res_reg, "independent_seed": res_ind}

    # 4. Part 2B: Cary PCA1, station cross-validation, annual correlations, and within-plot slopes
    cary_pca = sm.fit_pca1_corrmat(
        np.array([[cov_all["CARY"][y][k] for k in dc.PRIMARY_COVS] for y in range(1995, 2014)], dtype=float),
        orient_col=0,
    )
    for y, crow in cov_all["CARY"].items():
        crow["PC1_env"] = sm.project_pca1([crow[k] for k in dc.PRIMARY_COVS], cary_pca)

    station_checks = []
    for st_col, dm_col in [
        ("mat", "mat_cal"),
        ("degree_days", "dd0_oct_sep"),
        ("cold_winter_days", "cold_winter_frac"),
        ("feb_apr_warm_days", "feb_apr_warm_dry_frac"),
        ("snowfall_cm", "C4_snow"),
    ]:
        ys = sorted(y for y, d in cary_bio["station_dry"].items() if st_col in d and y in cov_all["CARY"])
        va = np.array([cary_bio["station_dry"][y][st_col] for y in ys], dtype=float)
        vb = np.array([cov_all["CARY"][y][dm_col] for y in ys], dtype=float)
        station_checks.append({
            "station": st_col,
            "daymet": dm_col,
            "years": len(ys),
            "pearson_r": sm.pearson_corr(va, vb),
            "spearman_rho": sm.spearman_corr(va, vb),
        })

    # Build Cary pair datasets
    def build_cary_pairs(period_name: str) -> tuple[list[dict], list[str]]:
        plots = cary_bio["hplots"] if period_name == "holdout" else cary_bio["dplots"]
        recs = []
        for g in plots:
            if period_name == "holdout":
                for ny, pv in sorted(cary_bio["hol_bb"].get(g, {}).items()):
                    mv = cary_bio["hol_mice"].get(g, {}).get(ny, 0.0)
                    nt = cary_bio["hol_n"].get(g, {}).get(ny, 0.0)
                    if 2014 <= ny <= 2022 and nt >= 20 and pv > 0.0 and mv > 0.0:
                        recs.append({"plot": g, "mouse_year": ny - 1, "nymph_year": ny, "x": math.log(mv), "y": math.log(pv), "tested": int(nt)})
            else:
                resp_map = (
                    cary_bio["dry_nip"]
                    if period_name == "primary"
                    else (cary_bio["dry_don"] if period_name == "DON_exploratory" else cary_bio["dry_lpm"])
                )
                for ny, pv in sorted(resp_map.get(g, {}).items()):
                    my = ny if period_name == "LPM_exploratory" else (ny - 1)
                    mv = cary_bio["dry_mice"].get(g, {}).get(my, 0.0)
                    if 1995 <= ny <= 2013 and (period_name != "DON_exploratory" or my >= 1995) and pv > 0.0 and mv > 0.0:
                        recs.append({"plot": g, "mouse_year": my, "nymph_year": ny, "x": math.log(mv), "y": math.log(pv)})
        for r in recs:
            my, ny = r["mouse_year"], r["nymph_year"]
            for tag, yr_lookup in [("nymph", ny), ("larva", my), ("prev_yr", ny - 1)]:
                for k, v in cov_all["CARY"][yr_lookup].items():
                    r[f"{k}_{tag}"] = v
            r["trend"] = float(ny - 2004)
            r["assay"] = cary_bio["pcr_share"].get(r["plot"], {}).get(ny, float("nan"))
            r["chip"] = math.log(cary_bio["dry_chip"].get(r["plot"], {}).get(my, 0.0) + 1.0)
        return recs, plots

    cary_slope_results = []
    cary_pair_sets = {}
    for per in ["primary", "holdout", "DON_exploratory", "LPM_exploratory"]:
        recs, plots = build_cary_pairs(per)
        cary_pair_sets[per] = recs
        pbar = cary_bio["pbar_primary"] if per != "holdout" else float(np.mean([math.exp(r["y"]) for r in recs]))
        upper = 1.0 - pbar
        specs = [("unadjusted", []), ("PC1_nymph", ["PC1_env_nymph"]), ("PC1_larva", ["PC1_env_larva"])]
        if per in ("primary", "holdout"):
            specs += [(k, [f"{k}_nymph"]) for k in dc.PRIMARY_COVS]
            specs += [("all_six", [f"{k}_nymph" for k in dc.PRIMARY_COVS]), ("PC1_trend", ["PC1_env_nymph", "trend"])]
            if per == "primary":
                specs += [("PC1_assay", ["PC1_env_nymph", "assay"]), ("PC1_chip", ["PC1_env_nymph", "chip"])]
        for label, cols in specs:
            res_reg = sm.cary_two_year_block_bootstrap(recs, plots, cols, n_boot=2000, seed=20261114)
            res_ind = sm.cary_two_year_block_bootstrap(recs, plots, cols, n_boot=2000, seed=99202602)
            b = res_reg["b"]
            se = res_reg["se"]
            outside = (b + 2.0 * se < 0.0) or (b - 2.0 * se > upper)
            ref_b = 0.051 if per == "holdout" else 0.038
            ref_se = 0.118 if per == "holdout" else 0.058
            cary_slope_results.append({
                "period": per,
                "covariate": label,
                "n_pairs": res_reg["n_pairs"],
                "n_cohorts": res_reg["n_cohorts"],
                "b": b,
                "se_reg_seed": se,
                "ci_low_reg_seed": res_reg["ci_low"],
                "ci_high_reg_seed": res_reg["ci_high"],
                "se_ind_seed": res_ind["se"],
                "ci_low_ind_seed": res_ind["ci_low"],
                "ci_high_ind_seed": res_ind["ci_high"],
                "upper_band": upper,
                "band_compatible": bool(not outside),
                "shift_gt_1se": bool(abs(b - ref_b) > ref_se),
                "nonzero_2se": bool(abs(b) > 2.0 * se),
                "verdict": "contradicts" if outside else "not decided",
                "insufficient_distinct_years": res_reg["insufficient_distinct_years"],
            })

    # Annual Cary correlations (19 cohorts 1994-2012)
    prim_recs = cary_pair_sets["primary"]
    cary_ys = sorted({r["mouse_year"] for r in prim_recs})
    xx_cary = np.array([np.mean([r["x"] for r in prim_recs if r["mouse_year"] == y]) for y in cary_ys], dtype=float)
    yy_cary = np.array([np.mean([r["y"] for r in prim_recs if r["mouse_year"] == y]) for y in cary_ys], dtype=float)
    cary_annual_corrs = []
    for lag_label, offset in [("nymph", 1), ("larva", 0)]:
        for k in dc.PRIMARY_COVS + ["PC1_env"]:
            zv = np.array([cov_all["CARY"][y + offset][k] for y in cary_ys], dtype=float)
            for tname, tvec in [("log_mice", xx_cary), ("log_nip_next", yy_cary)]:
                st = sm.annual_block_corr_ci(tvec, zv, n_boot=2000, seed=20261114)
                cary_annual_corrs.append({"covariate": k, "lag": lag_label, "target": tname, **st})

    print("[STEP 3] Completed Part 1 & Part 2 empirical covariates and adjustments.", flush=True)

    # 5. Part 3: Thermal stage durations and independent stochastic D engine campaign
    thermal_durations = {}
    for loc_key in ["CARY"] + dc.NEON_6:
        dm_name = "CARY" if loc_key == "CARY" else f"{loc_key}_tick"
        yrs_larv = list(range(1994, 2013)) if loc_key == "CARY" else [y - 1 for y in neon_bio["test_years"][loc_key]]
        thermal_durations[loc_key] = {
            "primary": dc.evaluate_thermal_durations(daymet_all[dm_name], yrs_larv, 0.02, "wallace2019"),
            "quest_fast": dc.evaluate_thermal_durations(daymet_all[dm_name], yrs_larv, 0.04, "wallace2019"),
            "recalled_powerlaw": dc.evaluate_thermal_durations(daymet_all[dm_name], yrs_larv, 0.02, "ogden2004_recalled"),
        }

    # Joint annual mouse-weather covariance inputs for Cary and 6 NEON sites
    joint_inputs = {}
    for s in dc.NEON_6:
        yrs = sorted(y for (ss, y), ix in neon_bio["index"].items() if ss == s and ix > 0.0 and (y + 1) in cov_all[f"{s}_tick"])
        xm = np.array([math.log(neon_bio["index"][(s, y)]) for y in yrs], dtype=float)
        wm = np.array([sm.project_pca1([cov_all[f"{s}_tick"][y + 1][k] for k in dc.PRIMARY_COVS], neon_pca) for y in yrs], dtype=float)
        adj_mask = np.diff(np.array(yrs)) == 1
        joint_inputs[s] = {
            "years": yrs,
            "log_mouse": xm.tolist(),
            "weather": wm.tolist(),
            "sigma_mouse": float(xm.std(ddof=1)),
            "sd_weather": float(wm.std(ddof=1)),
            "rho": sm.pearson_corr(xm, wm),
            "phi_mouse": sm.pearson_corr(xm[:-1][adj_mask], xm[1:][adj_mask]),
            "phi_weather": sm.pearson_corr(wm[:-1][adj_mask], wm[1:][adj_mask]),
        }
    cw_cary = np.array([cov_all["CARY"][y + 1]["PC1_env"] for y in cary_ys], dtype=float)
    joint_inputs["CARY"] = {
        "years": cary_ys,
        "log_mouse": xx_cary.tolist(),
        "weather": cw_cary.tolist(),
        "sigma_mouse": float(xx_cary.std(ddof=1)),
        "sd_weather": float(cw_cary.std(ddof=1)),
        "rho": sm.pearson_corr(xx_cary, cw_cary),
        "phi_mouse": sm.pearson_corr(xx_cary[:-1], xx_cary[1:]),
        "phi_weather": sm.pearson_corr(cw_cary[:-1], cw_cary[1:]),
    }

    # Build independent stochastic D tasks (own seeds 810000 + i)
    phi_fine = 0.30 ** (6.0 / DT_COARSE_DAYS)
    spec_tasks = []

    def add_task(key: str, spec: ls.BorreliaSpec):
        spec_tasks.append({"key": key, "spec": spec, "steps": 40000, "burn": 2000, "pairs": 24, "seed": 810000 + len(spec_tasks)})

    for site in ["CARY"] + dc.NEON_6:
        sig = cary_bio["cary_sigma"] if site == "CARY" else float(neon_bio["sigma"][site])
        td = thermal_durations[site]["primary"]
        stay_d = td["phen_ln_mean_d"]
        B_fine = stay_d / 6.0
        m_disc, m_cont = ls.match_discrete_chain_substages(B_fine, td["phen_ln_total_cv"])

        add_task(f"{site}|M0_coarse_iid|k1", ls.BorreliaSpec(1.0, 10.0 / 4.4, sig, 0.0, kappa1=1.0))
        add_task(f"{site}|M3_coarse_phen_iid|k1", ls.BorreliaSpec(1.0, stay_d / DT_COARSE_DAYS, sig, 0.0, kappa1=1.0))
        add_task(f"{site}|M3a_fine_memoryless_memory|k1", ls.BorreliaSpec(133.0 / 6.0, B_fine, sig, phi_fine, kappa1=1.0))
        add_task(
            f"{site}|M3b_fine_chain_memory|k1",
            ls.BorreliaSpec(133.0 / 6.0, B_fine, sig, phi_fine, kappa1=1.0, tick_shape="chain", tick_substages=m_disc),
        )
        add_task(
            f"{site}|M3c_fine_continuousCV_memory|k1",
            ls.BorreliaSpec(133.0 / 6.0, B_fine, sig, phi_fine, kappa1=1.0, tick_shape="chain", tick_substages=m_cont),
        )

    # Cary mechanism sensitivities (Sensitivity 2) + coupled-weather fine-step cells
    sig_c = cary_bio["cary_sigma"]
    B_c = thermal_durations["CARY"]["primary"]["phen_ln_mean_d"] / 6.0
    add_task("CARY|alternative_host|k1", ls.BorreliaSpec(133.0 / 6.0, B_c, sig_c, phi_fine, alt_host_share=0.5))
    add_task("CARY|asynchronous_patches|k1", ls.BorreliaSpec(133.0 / 6.0, B_c, sig_c, phi_fine, n_patches=4, patch_move=0.15))
    add_task("CARY|dilution|k1", ls.BorreliaSpec(133.0 / 6.0, B_c, sig_c, phi_fine, kappa1=1.0, kappa2=-0.17))
    for c_flag in (False, True):
        for d_share in (0.0, 0.5):
            add_task(
                f"CARY|bounded_contact{c_flag}_deathshare{d_share}|k1",
                ls.BorreliaSpec(
                    133.0 / 6.0,
                    B_c,
                    sig_c,
                    phi_fine,
                    contact_tick=c_flag,
                    driver_kind="bounded",
                    bounded_low=0.2,
                    tick_death=d_share / B_c,
                ),
            )
    # Also audit matched-residence-time bounded_contactFalse control where total leaving hazard = 1/B_c!
    # We want (1 - d_match)*(1 - h_match) = 1 - 1/B_c with d_match = 0.5 / B_c => 1 - h_match = (1 - 1/B_c)/(1 - 0.5/B_c)
    d_m = 0.5 / B_c
    h_m = 1.0 - (1.0 - 1.0 / B_c) / (1.0 - d_m)
    add_task(
        "CARY|bounded_contactFalse_deathshare0.5_matchedGenTime|k1",
        ls.BorreliaSpec(
            133.0 / 6.0,
            1.0 / h_m,
            sig_c,
            phi_fine,
            contact_tick=False,
            driver_kind="bounded",
            bounded_low=0.2,
            tick_death=d_m,
        ),
    )

    rho_cary = joint_inputs["CARY"]["rho"]
    for gam in (-0.2, -0.1, 0.1, 0.2):
        add_task(
            f"CARY|coupled_weather{gam:+g}|k1",
            ls.BorreliaSpec(
                133.0 / 6.0,
                B_c,
                sig_c,
                phi_fine,
                sigma_weather=abs(gam),
                phi_weather=phi_fine,
                rho_hw=float(rho_cary * np.sign(gam)),
            ),
        )

    with ProcessPoolExecutor(max_workers=4) as pool:
        spec_out_list = list(pool.map(_solve_spec_worker, spec_tasks))
    engine_cells = {r["key"]: r for r in spec_out_list}

    prediction_shifts = []
    for site in ["CARY"] + dc.NEON_6:
        c_old = engine_cells[f"{site}|M0_coarse_iid|k1"]
        c_new = engine_cells[f"{site}|M3_coarse_phen_iid|k1"]
        c_fmem = engine_cells[f"{site}|M3a_fine_memoryless_memory|k1"]
        c_fchn = engine_cells[f"{site}|M3b_fine_chain_memory|k1"]
        prediction_shifts.append({
            "site": site,
            "old_D": c_old["D"],
            "old_se": c_old["se"],
            "thermal_same_clock_D": c_new["D"],
            "thermal_se": c_new["se"],
            "change": c_new["D"] - c_old["D"],
            "change_se": math.hypot(c_old["se"], c_new["se"]),
            "percent_change": 100.0 * (c_new["D"] / c_old["D"] - 1.0),
            "fine_memoryless_D": c_fmem["D"],
            "fine_memoryless_se": c_fmem["se"],
            "fine_chain_D": c_fchn["D"],
            "fine_chain_se": c_fchn["se"],
            "chain_ratio": c_fchn["D"] / c_fmem["D"],
        })

    print(f"[STEP 4] Completed {len(engine_cells)} Part 3 & sensitivity stochastic D cells.", flush=True)

    # 6. Part 4A: Coupled-development & mortality NEON 6-site campaign (54 cells + 9 scenarios x 2,000 studies)
    coupled_tasks = []
    for s in dc.NEON_6:
        js = joint_inputs[s]
        phen_map = thermal_durations[s]["primary"]["phen_ln_by_year"]
        yrs_common = [y for y in js["years"] if y in phen_map]
        w_paired = [js["weather"][js["years"].index(y)] - float(np.mean(js["weather"])) for y in yrs_common]
        d_paired = [phen_map[y] for y in yrs_common]
        s_row = next(r for r in neon_site_rows if r["site"] == s)
        for gam in (-0.2, 0.0, 0.2):
            for kap in (0.0, 0.5, 1.0):
                coupled_tasks.append({
                    "site": s,
                    "sigma": s_row["sigma"],
                    "kappa": kap,
                    "gamma": gam,
                    "site_climate": s_row["PC1_env"],
                    "phi_mouse": js["phi_mouse"],
                    "phi_weather": js["phi_weather"],
                    "rho_hw": js["rho"],
                    "weather": w_paired,
                    "duration": d_paired,
                    "seed": 820000 + len(coupled_tasks),
                })

    with ProcessPoolExecutor(max_workers=4) as pool:
        coupled_cells = list(pool.map(_solve_coupled_worker, coupled_tasks))

    # Simulate 2,000 synthetic 6-site studies per (gamma, kappa) with independent seeds
    n_mc = 2000
    clim_6 = np.array([r["PC1_env"] for r in neon_site_rows], dtype=float)
    pbar_neon = float(np.mean(nip_6))
    base_m = -math.log1p(-pbar_neon) / 0.5
    neon_counts_by_site = {s: [neon_bio["cells"][(s, y)][0] for y in neon_bio["test_years"][s]] for s in dc.NEON_6}

    driven_scenarios = []
    for gam in (-0.2, 0.0, 0.2):
        for kap in (0.0, 0.5, 1.0):
            scen_idx = len(driven_scenarios)
            rng_obs = np.random.default_rng(830000 + scen_idx)
            cells_gk = [next(c for c in coupled_cells if c["site"] == s and c["gamma"] == gam and c["kappa"] == kap) for s in dc.NEON_6]
            d_vec = np.array([c["D"] for c in cells_gk], dtype=float)
            ln_r_mean = np.log([c["mean_R0"] for c in cells_gk])
            margin = base_m - float(np.mean(ln_r_mean - d_vec))
            site_res = 0.17 * rng_obs.normal(size=(n_mc, 6))
            phat = np.zeros((n_mc, 6), dtype=float)

            for i, s in enumerate(dc.NEON_6):
                ns_yr = np.asarray(neon_counts_by_site[s], dtype=int)
                ny = len(ns_yr)
                st_ln_r = np.log(np.asarray(cells_gk[i]["state_R0"], dtype=float))
                wz = ls.sample_gaussian_ar1(
                    np.array([1.0]),
                    np.array([joint_inputs[s]["phi_weather"]]),
                    np.ones((1, 1)),
                    ny,
                    n_mc,
                    835000 + scen_idx * 10 + i,
                    antithetic=False,
                    standardized=True,
                )[:, :, 0].T
                bins = np.minimum((special.ndtr(wz) * len(st_ln_r)).astype(int), len(st_ln_r) - 1)
                ann_ln_r = margin + site_res[:, i, None] + st_ln_r[bins] - d_vec[i]
                p_yr = 1.0 - np.exp(-0.5 * np.maximum(ann_ln_r, 0.0))
                k_pos = rng_obs.binomial(ns_yr[None, :], p_yr)
                phat[:, i] = k_pos.sum(axis=1) / float(ns_yr.sum())

            tests = sm.batch_rank_tests_6(phat, sigma_6, clim_6)
            rho = tests["rho"]
            rpart = tests["r_part"]
            padj = tests["p_adj_neg"]
            pun = tests["p_un_neg"]

            ev_neg = sm.wilson_interval(int(np.sum(rho < 0.0)), n_mc)
            ev_sup = sm.wilson_interval(int(np.sum(rho <= -29.0 / 35.0 + 1e-12)), n_mc)
            ev_sup_ex = sm.wilson_interval(int(np.sum((rho < 0.0) & (pun < 0.05))), n_mc)
            ev_pos = sm.wilson_interval(int(np.sum(rho >= 17.0 / 35.0 - 1e-12)), n_mc)
            ev_adj = sm.wilson_interval(int(np.sum((rpart < 0.0) & (padj < 0.05))), n_mc)

            driven_scenarios.append({
                "gamma": gam,
                "kappa": kap,
                "n": n_mc,
                "mean_rho": float(np.nanmean(rho)),
                "mean_partial": float(np.nanmean(rpart)),
                "negative_unadjusted": ev_neg["freq"],
                "support_cutoff": ev_sup["freq"],
                "support_cutoff_count": ev_sup["count"],
                "support_cutoff_wilson_low": ev_sup["wilson_low"],
                "support_cutoff_wilson_high": ev_sup["wilson_high"],
                "support_exact": ev_sup_ex["freq"],
                "as_positive_as_R14": ev_pos["freq"],
                "support_adjusted": ev_adj["freq"],
                "support_adjusted_wilson_low": ev_adj["wilson_low"],
                "support_adjusted_wilson_high": ev_adj["wilson_high"],
            })

    # 7. Part 4B: Within-site Cary 18-year x 6-grid masking/mimicking simulation (independent seeds)
    plots_c = cary_bio["dplots"]
    mask_18 = np.zeros((6, 18), dtype=bool)
    ntest_18 = np.ones((6, 18), dtype=int)
    for r in prim_recs:
        if 1995 <= r["mouse_year"] <= 2012:
            gi = plots_c.index(r["plot"])
            ti = r["mouse_year"] - 1995
            nn = cary_bio["n_tested_dry"].get((r["plot"], r["nymph_year"]), 0)
            if nn > 0:
                mask_18[gi, ti] = True
                ntest_18[gi, ti] = nn
    yr_means_x = {y: float(np.mean([r["x"] for r in prim_recs if r["mouse_year"] == y])) for y in cary_ys}
    sd_grid = float(np.std([r["x"] - yr_means_x[r["mouse_year"]] for r in prim_recs], ddof=1))
    pbar_c = cary_bio["pbar_primary"]
    upper_c = 1.0 - pbar_c

    boot_starts_18 = np.random.default_rng(849999).integers(0, 17, size=(2000, 9))
    boot_weights_18 = np.array([np.bincount(np.concatenate([s, s + 1]), minlength=18) for s in boot_starts_18], dtype=float)

    jc = joint_inputs["CARY"]
    cary_sim_scenarios = []
    for gam in (-0.2, -0.1, 0.0, 0.1, 0.2):
        for kap in (0.0, 0.5, 1.0):
            s_idx = len(cary_sim_scenarios)
            rng_c = np.random.default_rng(840000 + s_idx)
            pair_draws = ls.sample_gaussian_ar1(
                np.array([1.0, 1.0]),
                np.array([jc["phi_mouse"], jc["phi_weather"]]),
                np.array([[1.0, jc["rho"]], [jc["rho"], 1.0]]),
                18,
                n_mc,
                845000 + s_idx,
                antithetic=False,
                standardized=True,
            ).transpose(1, 0, 2)
            mx = pair_draws[:, :, 0] * jc["sigma_mouse"]
            ww = pair_draws[:, :, 1] * jc["sd_weather"]
            b_true = kap * upper_c / 2.0

            xx_sim = mx[:, None, :] + sd_grid * rng_c.normal(size=(n_mc, 6, 18))
            zz_sim = np.broadcast_to(ww[:, None, :], xx_sim.shape).copy()
            lp = math.log(pbar_c) + b_true * (xx_sim - xx_sim.mean(axis=(1, 2), keepdims=True)) + gam * zz_sim + 0.14 * rng_c.normal(size=(n_mc, 1, 18))
            p_sim = np.clip(np.exp(lp), 1e-5, 0.95)
            kk = rng_c.binomial(ntest_18[None, :, :], p_sim)
            with np.errstate(divide="ignore"):
                yy_sim = np.log(kk / ntest_18[None, :, :])
            yy_sim = np.where(mask_18[None, :, :] & (kk > 0), yy_sim, np.nan)

            pt_un, pt_adj = sm.vectorized_panel_fe_bootstrap(xx_sim, yy_sim, zz_sim, np.ones((1, 18)))
            b_un = pt_un[:, 0]
            b_adj = pt_adj[:, 0]

            se_un = np.empty(n_mc, dtype=float)
            se_adj = np.empty(n_mc, dtype=float)
            for chunk0 in range(0, n_mc, 50):
                bu_b, ba_b = sm.vectorized_panel_fe_bootstrap(
                    xx_sim[chunk0 : chunk0 + 50],
                    yy_sim[chunk0 : chunk0 + 50],
                    zz_sim[chunk0 : chunk0 + 50],
                    boot_weights_18,
                )
                se_un[chunk0 : chunk0 + 50] = np.nanstd(bu_b, axis=1, ddof=1)
                se_adj[chunk0 : chunk0 + 50] = np.nanstd(ba_b, axis=1, ddof=1)

            cary_sim_scenarios.append({
                "gamma": gam,
                "kappa": kap,
                "b_true": b_true,
                "usable_positions": int(mask_18.sum()),
                "mean_b": float(b_un.mean()),
                "mean_b_adjusted": float(b_adj.mean()),
                "gt2se_unadjusted": float(np.mean(b_un > 2.0 * se_un)),
                "gt2se_adjusted": float(np.mean(b_adj > 2.0 * se_adj)),
                "band_compatible_unadjusted": float(np.mean((b_un >= -2.0 * se_un) & (b_un <= upper_c + 2.0 * se_un))),
                "band_compatible_adjusted": float(np.mean((b_adj >= -2.0 * se_adj) & (b_adj <= upper_c + 2.0 * se_adj))),
            })

    print("[STEP 5] Completed Part 4 NEON (54 cells + 9x2000 studies) and Cary (15x2000 studies) simulations.", flush=True)

    # 8. Quantitative discrepancy audits
    # Audit A: Drop-off DOY 256 (envcov_lib.py:246) vs DOY 196..258 inclusive (protocol.md:96)
    drop_258_append = np.append(np.arange(196, 257, 5, dtype=int), 258)
    audit_dropoff = {}
    for site in ["CARY"] + dc.NEON_6:
        dm_name = "CARY" if site == "CARY" else f"{site}_tick"
        yrs_larv = list(range(1994, 2013)) if site == "CARY" else [y - 1 for y in neon_bio["test_years"][site]]
        d256 = thermal_durations[site]["primary"]
        d258 = dc.evaluate_thermal_durations(daymet_all[dm_name], yrs_larv, 0.02, "wallace2019", drop_doys=drop_258_append)
        audit_dropoff[site] = {
            "mean_doy256": d256["phen_ln_mean_d"],
            "mean_doy258": d258["phen_ln_mean_d"],
            "diff_days": d258["phen_ln_mean_d"] - d256["phen_ln_mean_d"],
            "cv_doy256": d256["phen_ln_total_cv"],
            "cv_doy258": d258["phen_ln_total_cv"],
        }

    # Audit B: Coarse-step / memory conversion 133.925 d vs 133.8 d (protocol.md:104, 197)
    phi_133_8 = 0.30 ** (6.0 / 133.8)
    d2_cary_133_925 = ls.IndependentCycleModel(ls.BorreliaSpec(133.0 / 6.0, B_c, sig_c, phi_fine)).second_order_d2()
    d2_cary_133_8 = ls.IndependentCycleModel(ls.BorreliaSpec(133.0 / 6.0, B_c, sig_c, phi_133_8)).second_order_d2()

    # Audit C: Cary annual mouse aggregation over `rs` (omitting TX 2006 where NIP_{TX,2007}=0, and unbalanced plots)
    # vs all positive PL_grid in 1994-2012 vs plot-demeaned ln(PL_grid)
    xx_all_pos = np.array(
        [np.mean([math.log(cary_bio["dry_mice"][g][y]) for g in plots_c if cary_bio["dry_mice"][g].get(y, 0.0) > 0.0]) for y in cary_ys],
        dtype=float,
    )
    plot_means_ln = {
        g: float(np.mean([math.log(v) for y, v in cary_bio["dry_mice"][g].items() if 1994 <= y <= 2012 and v > 0.0]))
        for g in plots_c
    }
    xx_demeaned = np.array(
        [
            np.mean([math.log(cary_bio["dry_mice"][g][y]) - plot_means_ln[g] for g in plots_c if cary_bio["dry_mice"][g].get(y, 0.0) > 0.0])
            for y in cary_ys
        ],
        dtype=float,
    )
    audit_cary_agg = {
        "r_pc1_nymph_rs_subset": sm.pearson_corr(xx_cary, cw_cary),
        "r_pc1_nymph_all_positive_mice": sm.pearson_corr(xx_all_pos, cw_cary),
        "r_pc1_nymph_plot_demeaned": sm.pearson_corr(xx_demeaned, cw_cary),
        "r_pc1_larva_rs_subset": sm.pearson_corr(xx_cary, np.array([cov_all["CARY"][y]["PC1_env"] for y in cary_ys])),
        "r_pc1_larva_all_positive_mice": sm.pearson_corr(xx_all_pos, np.array([cov_all["CARY"][y]["PC1_env"] for y in cary_ys])),
        "r_pc1_larva_plot_demeaned": sm.pearson_corr(xx_demeaned, np.array([cov_all["CARY"][y]["PC1_env"] for y in cary_ys])),
    }

    # Audit D: LPM_exploratory lagged (t-1) weather vs duplicate contemporaneous row in cary_slopes.csv:35-36
    lpm_recs = cary_pair_sets["LPM_exploratory"]
    lpm_prev_yr = sm.cary_two_year_block_bootstrap(lpm_recs, plots_c, ["PC1_env_prev_yr"], n_boot=2000, seed=20261114)
    lpm_same_yr = sm.cary_two_year_block_bootstrap(lpm_recs, plots_c, ["PC1_env_nymph"], n_boot=2000, seed=20261114)

    # Audit E: Bounded contact=False death_share=0.5 generation-time contraction vs matched-residence-time control
    c_false_05_orig = engine_cells["CARY|bounded_contactFalse_deathshare0.5|k1"]
    c_false_05_match = engine_cells["CARY|bounded_contactFalse_deathshare0.5_matchedGenTime|k1"]
    c_true_05 = engine_cells["CARY|bounded_contactTrue_deathshare0.5|k1"]

    discrepancy_audits = {
        "dropoff_doy256_vs_doy258": audit_dropoff,
        "phi_133_925_vs_133_8": {
            "phi_133_925": phi_fine,
            "phi_133_8": phi_133_8,
            "d2_cary_133_925": d2_cary_133_925,
            "d2_cary_133_8": d2_cary_133_8,
            "rel_diff": abs(d2_cary_133_8 - d2_cary_133_925) / d2_cary_133_925,
        },
        "cary_annual_mouse_aggregation": audit_cary_agg,
        "lpm_exploratory_lag_weather": {
            "contemporaneous_PC1_t": lpm_same_yr["b"],
            "contemporaneous_se": lpm_same_yr["se"],
            "lagged_PC1_t_minus_1": lpm_prev_yr["b"],
            "lagged_se": lpm_prev_yr["se"],
        },
        "bounded_contact_false_generation_time": {
            "orig_contactFalse_death05_D": c_false_05_orig["D"],
            "orig_contactFalse_death05_gen_steps": c_false_05_orig["generation_steps"],
            "matched_contactFalse_death05_D": c_false_05_match["D"],
            "matched_contactFalse_death05_gen_steps": c_false_05_match["generation_steps"],
            "contactTrue_death05_D": c_true_05["D"],
            "contactTrue_death05_gen_steps": c_true_05["generation_steps"],
        },
    }

    # Save structured outputs
    write_csv("neon_adjustments_rep.csv", neon_adjustments)
    write_csv("cary_slopes_rep.csv", cary_slope_results)
    write_csv("prediction_shifts_rep.csv", prediction_shifts)
    write_csv("driven_weather_scenarios_rep.csv", driven_scenarios)
    write_csv("cary_sim_scenarios_rep.csv", cary_sim_scenarios)

    full_output = {
        "replication_id": "r36-rep-envcov",
        "date_local": date_str,
        "date_utc": date_utc,
        "wall_seconds": round(time.time() - t0_wall, 2),
        "manifest": manifest_info,
        "cary_holdout_individual_max_diff": cary_bio["hol_ind_max_diff"],
        "cary_sigma": cary_bio["cary_sigma"],
        "pbar_primary": cary_bio["pbar_primary"],
        "pca_var_explained": {
            "neon_6sites": neon_pca["var_explained"],
            "cary_primary_19yrs": cary_pca["var_explained"],
        },
        "neon_sites": neon_site_rows,
        "neon_adjustments": neon_adjustments,
        "neon_correlations": neon_correlations,
        "neon_tower_sensitivity": tower_fl,
        "neon_t3": neon_t3_results,
        "station_checks": station_checks,
        "cary_slopes": cary_slope_results,
        "cary_annual_correlations": cary_annual_corrs,
        "thermal_durations": {
            s: {
                v: {k: val for k, val in d.items() if not isinstance(val, dict)}
                for v, d in vs.items()
            }
            for s, vs in thermal_durations.items()
        },
        "engine_cells": engine_cells,
        "prediction_shifts": prediction_shifts,
        "coupled_weather_cells": coupled_cells,
        "driven_weather_scenarios": driven_scenarios,
        "cary_sim_scenarios": cary_sim_scenarios,
        "discrepancy_audits": discrepancy_audits,
    }
    (HERE / "replication_results.json").write_text(json.dumps(full_output, indent=2) + "\n", encoding="utf-8")
    print(f"=== Completed all replications in {full_output['wall_seconds']} s -> replication_results.json ===", flush=True)


if __name__ == "__main__":
    main()
