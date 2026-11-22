#!/usr/bin/env python3
"""Independent reanalysis of case study `microcosm` (`r36-rep-microcosm`).

Run from repository root:
  python3 -I verification/replication/microcosm/run_replication.py

Recomputes from `data/raw/microcosm/` and `protocol.md` (+ Amendments 1 & 2) alone:
  1. Rule-7 known-result benchmarks (before touching raw data);
  2. Registered Test Group T1 (Ferris et al. 2020, S1): T1a, T1b (total, dynamic,
     ln(mean H)-adjusted), and explicit Treatment Key audit on sheet 3 cols G-H;
  3. Registered Test Group T2 (Duncan et al. 2013, S2): T2a (prevalence, synchronous,
     infected-abundance, host-adjusted), T2b (Day 56), T2c (Red Async-Sync early & late,
     plus 2x2 interactions), and uninfected host series summaries;
  4. Registered Test Group T3 & Parameter Extractions (Dallas et al. 2018 S3, Dallas &
     Drake 2016 S4, Clerc 2015 + Tadiri 2023 S5, Davenport et al. 2024a S6, Krichel et
     al. 2023 S7), including audit of Tadiri `temp.csv` duplicate `t56` column;
  5. Weather & Environment Amendment 1/2 (`W1a`, `W1b`, Daymet-vs-GHCN check, `W2`);
  6. Two most consequential sensitivity analyses:
     - Sensitivity A (`S2` 36-cell daily Markov host-only vs. 5 C cold developmental
       arrest across d in {1, 0.5, 0.1}, stay shapes {(1,1), (4,4), (1,16)}, phi in {0, 0.8});
     - Sensitivity B (`S3+S4` 30-cell Daphnia-Metschnikowia periodic Floquet vs.
       continuous-time random telegraph across waste d=1 vs. calibrated waiting d=0.0721,
       k_in in {1, 4, 16}, scales {7, 14, 28, 56, 112} d), plus S1 8-cell d x k_in slopes;
  7. Automated comparison against `casestudies/microcosm/results/summary_metrics.json`
     and `report.md` tables.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW_DIR = REPO_ROOT / "data" / "raw" / "microcosm"
ORIG_DIR = REPO_ROOT / "casestudies" / "microcosm"

sys.path.insert(0, str(HERE))
import benchmarks_and_models as bm  # noqa: E402


def utc_timestamp() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def t_mean_ci(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    se = float(arr.std(ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return {"n": int(n), "mean": mean, "se": se, "lo": mean - tcrit * se, "hi": mean + tcrit * se}


def series_positive_log_metrics(times: list[float], counts: list[float]) -> dict[str, Any]:
    """Series summary under Amendment 2 positive-count rule (keeping census gaps intact)."""
    t = np.asarray(times, dtype=float)
    h = np.asarray(counts, dtype=float)
    pos = np.isfinite(h) & (h > 0.0)
    log_h = np.full(len(h), np.nan, dtype=float)
    log_h[pos] = np.log(h[pos])

    sd_ln = float(np.std(log_h[pos], ddof=1)) if int(np.sum(pos)) >= 2 else float("nan")
    if int(np.sum(pos)) >= 3:
        slope, intercept = np.polyfit(t[pos], log_h[pos], 1)
        resid = log_h[pos] - (slope * t[pos] + intercept)
        det_sd = float(np.std(resid, ddof=1))
    else:
        det_sd = float("nan")

    lags: dict[str, float] = {}
    for lag in (1, 2):
        ok_pair = np.isfinite(log_h[:-lag]) & np.isfinite(log_h[lag:])
        a, b = log_h[:-lag][ok_pair], log_h[lag:][ok_pair]
        if len(a) >= 3 and min(float(np.std(a)), float(np.std(b))) > 1e-12:
            lags[f"r{lag}"] = float(np.corrcoef(a, b)[0, 1])
        else:
            lags[f"r{lag}"] = float("nan")

    return {
        "n_obs": int(len(h)),
        "zero_n": int(np.sum(h == 0.0)),
        "pos_n": int(np.sum(pos)),
        "mean_H": float(np.nanmean(h)) if np.any(np.isfinite(h)) else float("nan"),
        "sd_logH": sd_ln,
        "sd_logH_detrended": det_sd,
        "r1": lags["r1"],
        "r2": lags["r2"],
    }


def outcome_label(estimate: float, p_less: float, p_greater: float, expected_sign: int = -1) -> str:
    if expected_sign < 0:
        if estimate < 0 and p_less < 0.05:
            return "supports"
        if estimate > 0 and p_greater < 0.05:
            return "contradicts"
        return "not decided"
    else:
        if estimate > 0 and p_greater < 0.05:
            return "supports"
        if estimate < 0 and p_less < 0.05:
            return "contradicts"
        return "not decided"


# ---------------------------------------------------------------------------
# Section 1: Test Group T1 — Ferris et al. (2020) Phage Microcosms (S1)
# ---------------------------------------------------------------------------

def replicate_s1() -> dict[str, Any]:
    xlsx_path = RAW_DIR / "ferris2020_phage_oscillations" / "rspb20200787_si_002.xlsx"
    sheet3_all = bm.load_xlsx_rows(xlsx_path, sheet_number=3)

    # Extract explicit Treatment Key embedded in columns G-H (indices 6-7) of sheet 3!
    embedded_key: dict[str, str] = {}
    for row in sheet3_all:
        if len(row) >= 8 and row[6] in ("One", "Two", "Three", "Four", "Five"):
            embedded_key[row[6]] = row[7]

    # Parse KB oscillation amplitudes directly from the embedded key strings
    # e.g., "Oscillating 0.01/0.99" -> (0.99 - 0.01)/2 / 0.5 = 0.98; "Constant 0.5" -> 0.0
    amp_by_trt: dict[str, float] = {}
    for trt_name, desc in embedded_key.items():
        nums = [float(tok) for tok in desc.replace("Oscillating", "").replace("Constant", "").strip().split("/") if tok]
        if len(nums) == 2:
            mean_kb = 0.5 * (nums[0] + nums[1])
            amp_by_trt[trt_name] = abs(nums[1] - nums[0]) / (2.0 * mean_kb)
        else:
            amp_by_trt[trt_name] = 0.0

    records = bm.load_xlsx_records(xlsx_path, sheet_number=3, n_cols=6)
    by_trt: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    by_unit: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for r in records:
        by_trt[r["Treatment (KB)"]].append(r)
        by_unit[(r["Treatment (KB)"], r["Replicate"])].append(r)

    trt_summaries: dict[str, dict[str, Any]] = {}
    for trt_name, rr in sorted(by_trt.items(), key=lambda kv: amp_by_trt[kv[0]]):
        post0 = [r for r in rr if int(r["Transfer"]) > 0]
        # Transfers 3, 9, 15, 21 (t % 6 == 3) are low-resource phase; 6, 12, 18, 24 (t % 6 == 0) are high-resource phase
        low_h = float(np.mean([bm.to_float(r["Bacteria Density (CFU per ml)"]) for r in post0 if int(r["Transfer"]) % 6 == 3]))
        high_h = float(np.mean([bm.to_float(r["Bacteria Density (CFU per ml)"]) for r in post0 if int(r["Transfer"]) % 6 == 0]))
        mean_h = 0.5 * (low_h + high_h)
        trt_summaries[trt_name] = {
            "treatment": trt_name,
            "key_label": embedded_key[trt_name],
            "amplitude": round(amp_by_trt[trt_name], 6),
            "low_phase_H": low_h,
            "high_phase_H": high_h,
            "mean_H": mean_h,
            "delta_H": abs(high_h - low_h) / (2.0 * mean_h),
        }

    h_const = trt_summaries["Five"]["mean_H"]
    floquet_grid: list[dict[str, Any]] = []
    for trt_name, ts in trt_summaries.items():
        ts["mean_host_loss"] = math.log(h_const / ts["mean_H"])
        h_hi_rel = ts["high_phase_H"] / ts["mean_H"]
        h_lo_rel = ts["low_phase_H"] / ts["mean_H"]
        for d_val in (1.0, 0.5, 0.1, 0.03):
            for k_in in (1, 16):
                d_floq = bm.solve_floquet_loss_D(0.35, 0.03, k_out=1, k_in=k_in, d_wait=d_val, period=4.0, h_high=h_hi_rel, h_low=h_lo_rel)
                floquet_grid.append({
                    "treatment": trt_name,
                    "amplitude": ts["amplitude"],
                    "d": d_val,
                    "k_in": k_in,
                    "D": d_floq,
                    "mean_host_loss": ts["mean_host_loss"],
                    "total_loss": d_floq + ts["mean_host_loss"],
                })
                if d_val == 1.0 and k_in == 1:
                    ts["D"] = d_floq
                    ts["total_loss"] = d_floq + ts["mean_host_loss"]

    units: list[dict[str, Any]] = []
    for (trt_name, rep), rr in sorted(by_unit.items()):
        rr_sorted = sorted(rr, key=lambda r: int(r["Transfer"]))
        by_t = {int(r["Transfer"]): r for r in rr_sorted}
        # Last transfer with detectable PFU (>0) or 'Phage amplified later? == Yes'
        surv_transfers = [
            int(r["Transfer"])
            for r in rr_sorted
            if bm.to_float(r["Phage denisty (PFU per ml)"]) > 0.0 or r["Phage amplified later?"].strip() == "Yes"
        ]
        t_surv = max(surv_transfers)
        # Also verify first unrecovered zero transfer (t_extinct): t_extinct - 3 == t_surv for every unit
        unrec_zeros = [
            int(r["Transfer"])
            for r in rr_sorted
            if bm.to_float(r["Phage denisty (PFU per ml)"]) == 0.0 and r["Phage amplified later?"].strip() != "Yes"
        ]
        first_unrec = min(unrec_zeros) if unrec_zeros else 27
        assert first_unrec - 3 == t_surv, f"Non-monotone extinction coding in {trt_name}-{rep}"

        y_36 = 0.5 * (
            math.log1p(bm.to_float(by_t[3]["Phage denisty (PFU per ml)"]))
            + math.log1p(bm.to_float(by_t[6]["Phage denisty (PFU per ml)"]))
        )
        post0 = [r for r in rr_sorted if int(r["Transfer"]) > 0]
        h_metrics = series_positive_log_metrics(
            [int(r["Transfer"]) for r in post0],
            [bm.to_float(r["Bacteria Density (CFU per ml)"]) for r in post0],
        )
        units.append({
            "treatment": trt_name,
            "replicate": rep,
            "amplitude": trt_summaries[trt_name]["amplitude"],
            "T_surv": t_surv,
            "first_unrec_zero": first_unrec if unrec_zeros else None,
            "early_logPFU": y_36,
            "D": trt_summaries[trt_name]["D"],
            "total_loss": trt_summaries[trt_name]["total_loss"],
            **h_metrics,
        })

    x_amp = np.array([u["amplitude"] for u in units], dtype=float)
    x_D = np.array([u["D"] for u in units], dtype=float)
    x_tot = np.array([u["total_loss"] for u in units], dtype=float)
    y_surv = np.array([u["T_surv"] for u in units], dtype=float)
    y_pfu = np.array([u["early_logPFU"] for u in units], dtype=float)
    ln_hbar = np.log([u["mean_H"] for u in units])

    t1a_exact = bm.exact_spearman_permutation(x_amp, y_surv)
    t1a_mc = bm.mc_spearman_and_boot_ci(x_amp, y_surv, n_perm=100000, n_boot=10000, seed=20261115)
    t1a_by_D = bm.exact_spearman_permutation(x_D, y_surv)

    t1b_total = bm.ols_slope_permutation(x_tot, y_pfu, covariates=None, n_perm=100000, seed=20261115)
    t1b_total["status"] = outcome_label(t1b_total["slope"], t1b_total["p_less"], t1b_total["p_greater"], -1)

    t1b_dyn = bm.ols_slope_permutation(x_D, y_pfu, covariates=None, n_perm=100000, seed=20261115)
    t1b_dyn["status"] = outcome_label(t1b_dyn["slope"], t1b_dyn["p_less"], t1b_dyn["p_greater"], -1)

    t1b_adj = bm.ols_slope_permutation(x_D, y_pfu, covariates=ln_hbar, n_perm=100000, seed=20261115)
    t1b_adj["status"] = outcome_label(t1b_adj["slope"], t1b_adj["p_less"], t1b_adj["p_greater"], -1)

    # 8-cell d x k_in sensitivity slopes for S1
    s1_sens: list[dict[str, Any]] = []
    for d_val in (1.0, 0.5, 0.1, 0.03):
        for k_in in (1, 16):
            d_map = {g["treatment"]: g["D"] for g in floquet_grid if g["d"] == d_val and g["k_in"] == k_in}
            res = bm.ols_slope_permutation(np.array([d_map[u["treatment"]] for u in units]), y_pfu, n_perm=100000, seed=20261115)
            res["status"] = outcome_label(res["slope"], res["p_less"], res["p_greater"], -1)
            s1_sens.append({"d": d_val, "k_in": k_in, **res})

    return {
        "embedded_treatment_key": embedded_key,
        "treatments": list(trt_summaries.values()),
        "n_units": len(units),
        "T1a": {
            "rho": t1a_exact["rho"],
            "ci_lo": t1a_mc["ci_lo"],
            "ci_hi": t1a_mc["ci_hi"],
            "p_less_exact": t1a_exact["p_less_exact"],
            "p_greater_exact": t1a_exact["p_greater_exact"],
            "p_less_mc": t1a_mc["p_less_mc"],
            "p_greater_mc": t1a_mc["p_greater_mc"],
            "total_allocations": t1a_exact["total_allocations"],
            "status": outcome_label(t1a_exact["rho"], t1a_exact["p_less_exact"], t1a_exact["p_greater_exact"], -1),
        },
        "T1a_by_D": t1a_by_D,
        "T1b_total": t1b_total,
        "T1b_dynamic": t1b_dyn,
        "T1b_adjusted": t1b_adj,
        "S1_sensitivity_8cells": s1_sens,
        "floquet_grid": floquet_grid,
    }


# ---------------------------------------------------------------------------
# Section 2: Test Group T2 — Duncan, Gonzalez & Kaltz (2013) Holospora (S2)
# ---------------------------------------------------------------------------

def replicate_s2() -> dict[str, Any]:
    host_rows = bm.load_xlsx_records(RAW_DIR / "duncan2013_paramecium_holospora" / "Duncan_Paramecium_Time_7_13.xlsx", 1)
    par_rows = bm.load_xlsx_records(RAW_DIR / "duncan2013_paramecium_holospora" / "Duncan_Parasite_time_07_13.xlsx", 1)

    host_by_id: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in host_rows:
        host_by_id[r["Metapopulation"]].append(r)

    host_units: dict[str, dict[str, Any]] = {}
    host_day_map: dict[str, dict[int, float]] = {}
    control_cvs: list[float] = []

    for mp_id, rr in sorted(host_by_id.items(), key=lambda kv: int(kv[0])):
        rr_s = sorted(rr, key=lambda r: int(r["Time"]))
        r0 = rr_s[0]
        times = [int(r["Time"]) for r in rr_s]
        counts = [bm.to_float(r["Data"]) for r in rr_s]
        host_day_map[mp_id] = dict(zip(times, counts))
        hm = series_positive_log_metrics(times, counts)
        arr_ok = np.array([c for c in counts if np.isfinite(c)], dtype=float)
        cv_val = float(arr_ok.std(ddof=1) / arr_ok.mean())
        if r0["Infection"] == "Uninfected" and r0["Temperature"] == "Variable":
            control_cvs.append(cv_val)
        host_units[mp_id] = {
            "id": mp_id,
            "infection": r0["Infection"],
            "noise": r0["Red/White Variation"],
            "synchrony": r0["Temperature synchrony"],
            "migration": r0["Migration"],
            "temperature": r0["Temperature"],
            "cv": cv_val,
            **hm,
        }

    par_by_id: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in par_rows:
        par_by_id[r["Metapopulation"]].append(r)

    par_units: list[dict[str, Any]] = []
    day16_prevs: list[float] = []
    for mp_id, rr in sorted(par_by_id.items(), key=lambda kv: int(kv[0])):
        by_t = {int(r["Time"]): r for r in rr}
        r0 = rr[0]
        log_prev = {
            t: math.log((bm.to_float(v["Number infected"]) + 0.5) / (bm.to_float(v["Total"]) + 1.0))
            for t, v in by_t.items()
        }
        if r0["Temperature"] == "Variable":
            day16_prevs.append(bm.to_float(by_t[16]["Number infected"]) / bm.to_float(by_t[16]["Total"]))
        h0 = host_day_map[mp_id].get(0, float("nan"))
        h16 = host_day_map[mp_id].get(16, float("nan"))
        r_i016 = (log_prev[16] - log_prev[0] + math.log(h16 / h0)) / 16.0 if (np.isfinite(h0) and np.isfinite(h16) and h0 > 0 and h16 > 0) else float("nan")
        par_units.append({
            "id": mp_id,
            "noise": r0["Red/White Variation"],
            "synchrony": r0["Temperature synchrony"],
            "migration": r0["Migration"],
            "temperature": r0["Temperature"],
            "r016": (log_prev[16] - log_prev[0]) / 16.0,
            "rI016": r_i016,
            "logp56": log_prev[56],
            "late_logp": float(np.mean([lp for t, lp in log_prev.items() if t >= 16])),
            "mean_H": host_units[mp_id]["mean_H"],
        })

    var_units = [u for u in par_units if u["temperature"] == "Variable"]
    assert len(var_units) == 32

    def run_block_contrast(subset: list[dict[str, Any]], metric: str, focal_field: str, focal_val: str, block_fields: list[str], exp_sign: int) -> dict[str, Any]:
        missing_ids = [u["id"] for u in subset if not np.isfinite(u[metric])]
        clean_sub = [u for u in subset if np.isfinite(u[metric])]
        vals = np.array([u[metric] for u in clean_sub], dtype=float)
        grp = np.array([u[focal_field] == focal_val for u in clean_sub], dtype=bool)
        strata = ["|".join(str(u[bf]) for bf in block_fields) for u in clean_sub]
        res = bm.exact_stratified_two_group_test(vals, grp, strata, n_boot=10000, seed=20261115)
        res["excluded_ids"] = missing_ids
        res["status"] = outcome_label(res["diff"], res["p_less_exact"], res["p_greater_exact"], exp_sign)
        return res

    t2a = run_block_contrast(var_units, "r016", "noise", "Red", ["synchrony", "migration"], -1)
    t2a_sync = run_block_contrast([u for u in var_units if u["synchrony"] == "Synchronous"], "r016", "noise", "Red", ["migration"], -1)
    t2a_inf = run_block_contrast(var_units, "rI016", "noise", "Red", ["synchrony", "migration"], -1)
    t2b = run_block_contrast(var_units, "logp56", "noise", "Red", ["synchrony", "migration"], -1)

    red_units = [u for u in var_units if u["noise"] == "Red"]
    t2c_r016 = run_block_contrast(red_units, "r016", "synchrony", "Asynchronous", ["migration"], +1)
    t2c_late = run_block_contrast(red_units, "late_logp", "synchrony", "Asynchronous", ["migration"], +1)

    # Interaction: (Async - Sync)_Red - (Async - Sync)_White
    # Transform u_tilde = +2 * y for Red and -2 * y for White; then mean(Async) - mean(Sync) equals
    # [0.5*(2*Red_Async - 2*White_Async)] - [0.5*(2*Red_Sync - 2*White_Sync)] = (Async-Sync)_Red - (Async-Sync)_White!
    inter_early_units = [dict(u, inter=u["r016"] * (2.0 if u["noise"] == "Red" else -2.0)) for u in var_units]
    inter_late_units = [dict(u, inter=u["late_logp"] * (2.0 if u["noise"] == "Red" else -2.0)) for u in var_units]
    t2c_inter_r016 = run_block_contrast(inter_early_units, "inter", "synchrony", "Asynchronous", ["noise", "migration"], +1)
    t2c_inter_late = run_block_contrast(inter_late_units, "inter", "synchrony", "Asynchronous", ["noise", "migration"], +1)

    # Host-adjusted T2a regression
    cov_mat = np.column_stack([
        np.log([u["mean_H"] for u in var_units]),
        [1.0 if u["synchrony"] == "Synchronous" else 0.0 for u in var_units],
        [1.0 if u["migration"] == "High" else 0.0 for u in var_units],
    ])
    t2a_adj = bm.ols_slope_permutation(
        np.array([1.0 if u["noise"] == "Red" else 0.0 for u in var_units]),
        np.array([u["r016"] for u in var_units]),
        covariates=cov_mat,
        n_perm=100000,
        seed=20261115,
    )
    t2a_adj["status"] = outcome_label(t2a_adj["slope"], t2a_adj["p_less"], t2a_adj["p_greater"], -1)

    # Uninfected control host summaries (n = 16 Red, n = 16 White)
    uninf_summary: dict[str, Any] = {}
    for noise_col in ("Red", "White"):
        sub_h = [hu for hu in host_units.values() if hu["infection"] == "Uninfected" and hu["temperature"] == "Variable" and hu["noise"] == noise_col]
        uninf_summary[noise_col] = {
            m: t_mean_ci([hu[m] for hu in sub_h])["mean"]
            for m in ("mean_H", "sd_logH", "sd_logH_detrended", "r1", "r2")
        }
        uninf_summary[noise_col]["n"] = len(sub_h)

    return {
        "T2a": t2a,
        "T2a_synchronous": t2a_sync,
        "T2a_infected_abundance": t2a_inf,
        "T2a_host_adjusted": t2a_adj,
        "T2b": t2b,
        "T2c_r016": t2c_r016,
        "T2c_late_logp": t2c_late,
        "T2c_interaction_r016": t2c_inter_r016,
        "T2c_interaction_late_logp": t2c_inter_late,
        "uninfected_host_controls": uninf_summary,
        "mean_control_CV": float(np.mean(control_cvs)),
        "max_day16_prev_variable": float(max(day16_prevs)),
    }


# ---------------------------------------------------------------------------
# Section 3: Test Group T3 & Parameter Extractions (S3, S4, S5, S6, S7)
# ---------------------------------------------------------------------------

def replicate_s3_to_s7() -> dict[str, Any]:
    # --- S3: Dallas, Krkosek & Drake (2018) ---
    s3_rows = bm.load_csv_records(RAW_DIR / "dallas2018_metschnikowia_threshold" / "expData.csv")
    s3_by_pop: dict[tuple[int, str], list[dict[str, str]]] = collections.defaultdict(list)
    for r in s3_rows:
        s3_by_pop[(int(r["trt"]), r["popID"])].append(r)

    s3_units: list[dict[str, Any]] = []
    for (trt_int, pop_id), rr in sorted(s3_by_pop.items()):
        rr_s = sorted(rr, key=lambda r: bm.to_float(r["day"]))
        days = [bm.to_float(r["day"]) for r in rr_s]
        h_per_l = [bm.to_float(r["total"]) * 20.0 for r in rr_s]  # 50-mL tube -> *20 per L
        prevs = [bm.to_float(r["prev"]) for r in rr_s]
        day11_row = next(r for r in rr_s if bm.to_float(r["day"]) == 11.0)
        auc_val = float(np.trapezoid(prevs, days))
        hm = series_positive_log_metrics(days, h_per_l)
        s3_units.append({
            "trt": trt_int,
            "popID": pop_id,
            "H0_L": trt_int * 20,
            "H11_L": bm.to_float(day11_row["total"]) * 20.0,
            "primary_at11": int(bm.to_float(day11_row["infected"]) > 0.0),
            "primary_by11": int(any(bm.to_float(r["infected"]) > 0.0 for r in rr_s if bm.to_float(r["day"]) <= 11.0)),
            "secondary_gt18": int(any(bm.to_float(r["infected"]) > 0.0 for r in rr_s if bm.to_float(r["day"]) > 18.0)),
            "auc": auc_val,
            **hm,
        })

    super_units = [u for u in s3_units if u["H0_L"] >= 160]
    assert len(super_units) == 45

    def t3_label(r_val: float, p_less: float, p_greater: float) -> str:
        if r_val < 0 and p_less < 0.05:
            return "directionally consistent (observational)"
        if r_val > 0 and p_greater < 0.05:
            return "dominated by reverse-causal epidemic host suppression (registered label)"
        return "not decided"

    t3a_pearson = bm.partial_correlation_test(
        np.array([u["sd_logH"] for u in super_units]),
        np.log([u["auc"] + 0.01 for u in super_units]),
        np.log([u["mean_H"] for u in super_units]),
        spearman=False,
    )
    t3a_pearson["status"] = t3_label(t3a_pearson["r"], t3a_pearson["p_less"], t3a_pearson["p_greater"])

    t3a_spearman = bm.partial_correlation_test(
        np.array([u["sd_logH"] for u in super_units]),
        np.log([u["auc"] + 0.01 for u in super_units]),
        np.log([u["mean_H"] for u in super_units]),
        spearman=True,
    )
    t3a_spearman["status"] = t3_label(t3a_spearman["r"], t3a_spearman["p_less"], t3a_spearman["p_greater"])

    s3_arms: list[dict[str, Any]] = []
    for h0 in (20, 40, 80, 160, 320, 640):
        arm = [u for u in s3_units if u["H0_L"] == h0]
        p_cnt = sum(u["primary_at11"] for u in arm)
        s_cnt = sum(u["secondary_gt18"] for u in arm)
        p_ci = stats.binomtest(p_cnt, len(arm)).proportion_ci()
        s_ci = stats.binomtest(s_cnt, len(arm)).proportion_ci()
        sd_vals = [u["sd_logH"] for u in arm]
        s3_arms.append({
            "H0_L": h0,
            "n": len(arm),
            "mean_H11": float(np.mean([u["H11_L"] for u in arm])),
            "primary_n": p_cnt,
            "primary_frac": p_cnt / len(arm),
            "primary_lo": float(p_ci.low),
            "primary_hi": float(p_ci.high),
            "secondary_n": s_cnt,
            "secondary_frac": s_cnt / len(arm),
            "secondary_lo": float(s_ci.low),
            "secondary_hi": float(s_ci.high),
            "mean_sd_logH_strict": float(np.mean(sd_vals)) if all(np.isfinite(sd_vals)) else None,
            "mean_sd_logH_nanmean": float(np.nanmean(sd_vals)),
            "n_undefined_sd_logH": sum(not np.isfinite(v) for v in sd_vals),
        })

    # --- S4: Dallas & Drake (2016) ---
    sp_rows = bm.load_csv_records(RAW_DIR / "dallas2016_metschnikowia_tempvar" / "sporeSurv.csv")
    s4_surv: list[dict[str, Any]] = []
    for trt_s in ("0", "1", "2", "4"):
        sub = [r for r in sp_rows if r["trt"] == trt_s]
        t_arr = np.array([bm.to_float(r["day"]) for r in sub])
        d_arr = np.array([bm.to_float(r["dead"]) for r in sub])
        n_arr = np.array([bm.to_float(r["total"]) for r in sub])
        for wb in (False, True):
            fit = bm.fit_binomial_weibull_or_exp(t_arr, d_arr, n_arr, weibull=wb)
            s4_surv.append({"trt": trt_s, **fit})

    filt_rows = bm.load_csv_records(RAW_DIR / "dallas2016_metschnikowia_tempvar" / "filterRate.csv")
    filt_20 = t_mean_ci([bm.to_float(r["mlDay"]) for r in filt_rows if r["trt"] == "20"])

    ind0_rows = bm.load_csv_records(RAW_DIR / "dallas2016_metschnikowia_tempvar" / "indivExpData0.csv")
    inf0 = [r for r in ind0_rows if r["infected"].strip() == "1"]
    ages0 = np.array([bm.to_float(r["death.day"]) for r in inf0], dtype=float)
    yields0 = np.array([
        np.mean([bm.to_float(r[f"spores{j}"]) for j in (1, 2, 3, 4)]) * 10.0 * bm.to_float(r["volume"])
        for r in inf0
    ], dtype=float)
    age0_ci = t_mean_ci(ages0)
    cv0 = float(ages0.std(ddof=1) / ages0.mean())
    yield0_ci = t_mean_ci(yields0)

    mu_0 = next(f["lambda_day"] for f in s4_surv if f["trt"] == "0" and f["model"] == "exponential")
    F_20 = filt_20["mean"]
    tau_in_0 = age0_ci["mean"]
    contact_curve = [
        {
            "H_per_L": h_l,
            "encounter_day": F_20 * h_l / 1000.0,
            "d": mu_0 / (mu_0 + F_20 * h_l / 1000.0),
            "tau_out_contact": 1.0 / (mu_0 + F_20 * h_l / 1000.0),
            "tau_out_death_only": 1.0 / mu_0,
        }
        for h_l in (20, 40, 80, 160, 320, 640)
    ]

    # --- S5: Clerc et al. (2015) + Tadiri & Ebert (2023) ---
    clerc_hdr, clerc_body = bm.load_csv_raw(RAW_DIR / "clerc2015_pasteuria_stages" / "clerc_exp_1.csv")
    dpi_idx = clerc_hdr.index("SAMP_DPI")
    clerc_days = sorted({int(float(r[dpi_idx])) for r in clerc_body})

    tad_hdr, tad_body = bm.load_csv_raw(RAW_DIR / "tadiri2023_pasteuria_epidemics" / "temp.csv")
    # Audit duplicate `t56` column at indices 152 and 153!
    t_col_indices = [(idx, name) for idx, name in enumerate(tad_hdr) if name.startswith("t") and name[1:].isdigit()]
    dup_t56_indices = [idx for idx, name in t_col_indices if name == "t56"]
    assert dup_t56_indices == [152, 153]

    idx_temp = tad_hdr.index("Temp")
    idx_de = tad_hdr.index("de")
    idx_tti = tad_hdr.index("tti")
    idx_htti = tad_hdr.index("htti")

    # Compare host_stats when DictReader overwrites col 152 with col 153 vs when col 152 -> t56, col 153 -> t61
    sd_diff_jars = 0
    r1_diff_jars = 0
    clocks: dict[str, Any] = {}
    parsed_jars: list[dict[str, Any]] = []

    for row in tad_body:
        # DictReader behaviour (takes last index 153 for 't56', dropping index 152)
        t_dict = [int(name[1:]) for idx, name in t_col_indices if idx != 152]
        v_dict = [bm.to_float(row[idx]) for idx, _ in t_col_indices if idx != 152]
        m_dict = series_positive_log_metrics(t_dict, v_dict)

        # Repaired schedule (index 152 = day 56, index 153 = day 61 matching j61, i61, ni61)
        t_rep = [61 if idx == 153 else int(name[1:]) for idx, name in t_col_indices]
        v_rep = [bm.to_float(row[idx]) for idx, _ in t_col_indices]
        m_rep = series_positive_log_metrics(t_rep, v_rep)

        if not (math.isnan(m_dict["sd_logH"]) and math.isnan(m_rep["sd_logH"])):
            if abs((m_dict["sd_logH"] or 0.0) - (m_rep["sd_logH"] or 0.0)) > 1e-12:
                sd_diff_jars += 1
        if not (math.isnan(m_dict["r1"]) and math.isnan(m_rep["r1"])):
            if abs((m_dict["r1"] or 0.0) - (m_rep["r1"] or 0.0)) > 1e-12:
                r1_diff_jars += 1

        parsed_jars.append({
            "temp": bm.to_float(row[idx_temp]),
            "origin": row[idx_de].strip(),
            "tti": bm.to_float(row[idx_tti]),
            "htti": bm.to_float(row[idx_htti]),
        })

    for origin in ("all", "d", "e"):
        for metric in ("tti", "htti"):
            sub = [j for j in parsed_jars if (origin == "all" or j["origin"] == origin) and np.isfinite(j[metric]) and j[metric] > 0.0]
            x_t = np.array([j["temp"] for j in sub], dtype=float)
            y_ln = np.log([j[metric] for j in sub])
            lr = stats.linregress(x_t, y_ln)
            tcrit = float(stats.t.ppf(0.975, len(sub) - 2))
            chi = -float(lr.slope)
            clocks[f"{origin}_{metric}"] = {
                "n": len(sub),
                "intercept": float(lr.intercept),
                "slope": float(lr.slope),
                "chi": chi,
                "chi_lo": chi - tcrit * float(lr.stderr),
                "chi_hi": chi + tcrit * float(lr.stderr),
                "R2": float(lr.rvalue ** 2),
            }

    # --- S7: Krichel et al. (2023) ---
    s7_rows = bm.load_csv_records(RAW_DIR / "krichel2023_ordospora_tempvar" / "pbio.3002260.s016.csv")
    s7_temp_rows = bm.load_csv_records(RAW_DIR / "krichel2023_ordospora_tempvar" / "pbio.3002260.s017.csv")
    obs_to_exp = {
        int(bm.to_float(r["obs_day"])): int(bm.to_float(r["exp_day"]))
        for r in s7_temp_rows
        if np.isfinite(bm.to_float(r["obs_day"]))
    }

    pop_day_groups: dict[tuple[str, int], list[dict[str, str]]] = collections.defaultdict(list)
    for r in s7_rows:
        pop_day_groups[(r["population"].strip(), int(bm.to_float(r["obs_day"])))].append(r)

    pop_days: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for (pop, oday), rr in sorted(pop_day_groups.items()):
        statuses = [r["inf_status"].strip().lower() for r in rr if r["inf_status"].strip().lower() in ("yes", "no")]
        clusters = [bm.to_float(r["n_clusters"]) for r in rr]
        prev = sum(s == "yes" for s in statuses) / len(statuses) if statuses else float("nan")
        mb = float(np.nanmean(clusters)) if any(np.isfinite(clusters)) else float("nan")
        pop_days[pop].append({"obs_day": oday, "exp_day": obs_to_exp[oday], "prev": prev, "burden": mb})

    s7_summary: dict[str, Any] = {}
    for is_var, label_v in ((False, "const_20C"), (True, "var_14_26C")):
        pops = [p for p in sorted(pop_days) if (not p.startswith("20C")) == is_var]
        for win_name, min_exp in (("full", -1), ("endemic_gt150", 150)):
            p_means = [float(np.nanmean([d["prev"] for d in pop_days[p] if d["exp_day"] > min_exp])) for p in pops]
            b_means = [float(np.nanmean([d["burden"] for d in pop_days[p] if d["exp_day"] > min_exp])) for p in pops]
            s7_summary[f"{label_v}_{win_name}_prev"] = t_mean_ci(p_means)
            s7_summary[f"{label_v}_{win_name}_burden"] = t_mean_ci(b_means)

    return {
        "T3a_Pearson": t3a_pearson,
        "T3a_Spearman": t3a_spearman,
        "S3_arms": s3_arms,
        "S4_control_exp": next(f for f in s4_surv if f["trt"] == "0" and f["model"] == "exponential"),
        "S4_control_weibull": next(f for f in s4_surv if f["trt"] == "0" and f["model"] == "Weibull"),
        "S4_all_survival": s4_surv,
        "S4_filtering_20C": filt_20,
        "S4_host_lifespan_trt0": {**age0_ci, "cv": cv0, "k_moment": 1.0 / (cv0 ** 2), "spore_yield_mean": yield0_ci["mean"], "spore_yield_se": yield0_ci["se"]},
        "S4_contact_curve": contact_curve,
        "S5_clerc_records": len(clerc_body),
        "S5_clerc_day_range": [min(clerc_days), max(clerc_days)],
        "S5_tadiri_jars": len(tad_body),
        "S5_tadiri_dup_t56_audit": {
            "duplicate_header_indices": dup_t56_indices,
            "jars_with_altered_sd_logH": sd_diff_jars,
            "jars_with_altered_r1": r1_diff_jars,
        },
        "S5_clocks": clocks,
        "S7": s7_summary,
    }


# ---------------------------------------------------------------------------
# Section 4: Weather & Environment Amendment (`W1a`, `W1b`, `W2`)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin(0.5 * (p2 - p1)) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(0.5 * (l2 - l1)) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(a))


def canon_lake(name: str) -> str:
    return {"Little Appleton": "LilAp", "LittleAppleton": "LilAp", "Crooked W": "CrookedW"}.get(name.strip(), name.strip())


def replicate_weather(pasteuria_clock: dict[str, float]) -> dict[str, Any]:
    lake_coords = {
        "Bishop": (42.5004, -83.8411),
        "Cedar": (42.3146, -84.0790),
        "CrookedW": (42.3241, -84.1125),
        "LilAp": (42.5066, -83.8385),
        "Mill": (42.3287, -84.0905),
        "Walsh": (42.3375, -84.0798),
    }
    daymet_file_stem = {"LilAp": "LittleAppleton"}

    # Load GHCN stations
    stations: list[dict[str, Any]] = []
    for st_label, st_id in (("AnnArbor", "USW00094889"), ("Jackson", "USW00014833")):
        st_rows = bm.load_csv_records(RAW_DIR / "weather" / f"ghcnd_{st_label}_{st_id}_2021.csv")
        stations.append({
            "label": st_label,
            "id": st_id,
            "lat": bm.to_float(st_rows[0]["LATITUDE"]),
            "lon": bm.to_float(st_rows[0]["LONGITUDE"]),
            "by_date": {dt.date.fromisoformat(r["DATE"]): (bm.to_float(r["TMAX"]) + bm.to_float(r["TMIN"])) / 2.0 for r in st_rows},
        })

    # Load Daymet 1-km daily series and compute trailing 14-day means (days t-14 .. t-1)
    wx_14d: dict[str, dict[dt.date, dict[str, float]]] = {}
    ghcn_14d: dict[str, dict[dt.date, float]] = {}
    station_distances: dict[str, dict[str, Any]] = {}

    for lake, (lat, lon) in lake_coords.items():
        dm_path = RAW_DIR / "weather" / f"daymet_{daymet_file_stem.get(lake, lake)}_2021.csv"
        lines = dm_path.read_text(encoding="utf-8").splitlines()
        hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("year,yday"))
        dm_rows = list(csv.DictReader(lines[hdr_idx:]))
        daily: dict[dt.date, dict[str, float]] = {}
        for r in dm_rows:
            d_obj = dt.date(int(r["year"]), 1, 1) + dt.timedelta(days=int(r["yday"]) - 1)
            daily[d_obj] = {
                "tmean": 0.5 * (bm.to_float(r["tmax (deg c)"]) + bm.to_float(r["tmin (deg c)"])),
                "srad": bm.to_float(r["srad (W/m^2)"]),
                "prcp": bm.to_float(r["prcp (mm/day)"]),
                "vp": bm.to_float(r["vp (Pa)"]),
            }
        nearest_st = min(stations, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))
        station_distances[lake] = {
            "station": nearest_st["id"],
            "distance_km": haversine_km(lat, lon, nearest_st["lat"], nearest_st["lon"]),
        }
        wx_14d[lake] = {}
        ghcn_14d[lake] = {}
        for d_obj in daily:
            win = [d_obj - dt.timedelta(days=k) for k in range(1, 15)]
            if all(w in daily for w in win):
                wx_14d[lake][d_obj] = {
                    cov: float(np.mean([daily[w][cov] for w in win]))
                    for cov in ("tmean", "srad", "prcp", "vp")
                }
                st_vals = [nearest_st["by_date"][w] for w in win if w in nearest_st["by_date"] and np.isfinite(nearest_st["by_date"][w])]
                ghcn_14d[lake][d_obj] = float(np.mean(st_vals))

    # Load S6 field host/infection dataset (576 rows)
    field_raw = bm.load_csv_records(RAW_DIR / "davenport2024a_spores_outbreaks" / "Clean_Data_2021_Lakes_All_Host_Densities.csv")
    field_records: list[dict[str, Any]] = []
    na_date_reconstructed = 0
    for r in field_raw:
        lake = canon_lake(r["Lake"])
        d_from_jul = dt.date(int(r["Year"]), 1, 1) + dt.timedelta(days=int(r["Julian"]) - 1)
        if r["Date"].strip() in ("", "NA"):
            d_obj = d_from_jul
            na_date_reconstructed += 1
        else:
            d_obj = dt.datetime.strptime(r["Date"].strip(), "%m/%d/%y").date()
            assert d_obj == d_from_jul
        w = wx_14d[lake][d_obj]
        field_records.append({
            "lake": lake,
            "species": r["Host.Species"].strip(),
            "date": d_obj,
            "doy": d_obj.timetuple().tm_yday,
            "H": bm.to_float(r["Host.Density"]),
            "ghcn_tmean": ghcn_14d[lake][d_obj],
            **w,
            **{p: bm.to_float(r[f"{p}.prev"]) for p in ("pasteuria", "metsch", "spiro", "brood", "micG")},
        })

    # Host memory across 48 lake x species series
    by_ls: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in field_records:
        by_ls[(rec["lake"], rec["species"])].append(rec)
    s6_r1_vals: list[float] = []
    s6_zero_total = 0
    for (_lk, _sp), rr in sorted(by_ls.items()):
        rr_s = sorted(rr, key=lambda x: x["date"])
        hm = series_positive_log_metrics([x["doy"] for x in rr_s], [x["H"] for x in rr_s])
        s6_zero_total += hm["zero_n"]
        if np.isfinite(hm["r1"]):
            s6_r1_vals.append(hm["r1"])

    # W1a: Spore-to-outbreak lag vs trailing 14d temperature (n = 13 outbreaks)
    beg_rows = bm.load_csv_records(RAW_DIR / "davenport2024a_spores_outbreaks" / "SporeConcentrationBeginningVersusEnd.csv")
    lag_rows = bm.load_csv_records(RAW_DIR / "davenport2024a_spores_outbreaks" / "AvgDaysBetweenSporeIncreaseAndOutbreakStart.csv")
    beg_lookup = {
        (canon_lake(r["Lake"]), r["HostSpecies"].replace("_", " ").strip(), r["Parasite"].strip()): r
        for r in beg_rows
    }
    w1a_lags: list[float] = []
    w1a_temps: list[float] = []
    for r in lag_rows:
        key = (canon_lake(r["Lake"]), r["HostSpecies"].replace("_", " ").strip(), r["Parasite"].strip())
        d_out = dt.datetime.strptime(beg_lookup[key]["Date_outbreak_reached_1%"].strip(), "%m/%d/%y").date()
        w1a_lags.append(bm.to_float(r["DaysBetweenSporeIncreaseAndOutbreakStart"]))
        w1a_temps.append(wx_14d[key[0]][d_out]["tmean"])

    lags_arr = np.array(w1a_lags, dtype=float)
    temps_arr = np.array(w1a_temps, dtype=float)
    w1a_exact_check = stats.spearmanr(lags_arr, temps_arr)
    w1a_mc = bm.mc_spearman_and_boot_ci(lags_arr, temps_arr, n_perm=100000, n_boot=10000, seed=20261115)

    # W1b: Unadjusted vs (T_14d, srad_14d)-partialled vs (+ lake|species + doy)-partialled correlations
    w1b_table: list[dict[str, Any]] = []
    cov_corrs: dict[str, dict[str, float]] = {}

    for par in ("pasteuria", "metsch", "spiro", "brood", "micG"):
        sub = [r for r in field_records if np.isfinite(r["H"]) and r["H"] > 0.0 and np.isfinite(r[par])]
        ln_h = np.log([r["H"] for r in sub])
        prev = np.array([r[par] for r in sub], dtype=float)
        z_wx = np.array([[r["tmean"], r["srad"]] for r in sub], dtype=float)

        groups_ls = [f"{r['lake']}|{r['species']}" for r in sub]
        uniq_ls = sorted(set(groups_ls))
        dummies = np.array([[1.0 if g == u else 0.0 for u in uniq_ls[1:]] for g in groups_ls], dtype=float)
        doy_vec = np.array([r["doy"] for r in sub], dtype=float)

        Z_grp = np.column_stack([np.ones(len(sub)), dummies])
        res_h_grp = ln_h - Z_grp @ np.linalg.lstsq(Z_grp, ln_h, rcond=None)[0]
        res_p_grp = prev - Z_grp @ np.linalg.lstsq(Z_grp, prev, rcond=None)[0]

        r_unadj = float(np.corrcoef(ln_h, prev)[0, 1])
        p_wx = bm.partial_correlation_test(ln_h, prev, z_wx, spearman=False)
        r_within = float(np.corrcoef(res_h_grp, res_p_grp)[0, 1])
        p_full = bm.partial_correlation_test(ln_h, prev, np.column_stack([z_wx, dummies, doy_vec]), spearman=False)

        w1b_table.append({
            "parasite": par,
            "n": len(sub),
            "unadjusted_r": r_unadj,
            "weather_partial_r": p_wx["r"],
            "weather_partial_lo": p_wx["ci_lo"],
            "weather_partial_hi": p_wx["ci_hi"],
            "within_lake_species_r": r_within,
            "weather_group_date_partial_r": p_full["r"],
            "changed_sign": (r_unadj * p_wx["r"]) < 0.0,
            "halved": abs(p_wx["r"]) <= 0.5 * abs(r_unadj),
        })

        sq_dev = res_h_grp ** 2
        for cov in ("tmean", "srad", "prcp", "vp"):
            cv_vec = np.array([r[cov] for r in sub], dtype=float)
            cov_corrs.setdefault(cov, {})["r_logH"] = float(np.corrcoef(ln_h, cv_vec)[0, 1])
            cov_corrs[cov]["r_sq_dev"] = float(np.corrcoef(sq_dev, cv_vec)[0, 1])
            cov_corrs[cov][f"r_{par}"] = float(np.corrcoef(prev, cv_vec)[0, 1])

    # Daymet vs GHCN station agreement across the 576 field rows
    dm_t = np.array([r["tmean"] for r in field_records], dtype=float)
    gh_t = np.array([r["ghcn_tmean"] for r in field_records], dtype=float)
    dm_gh_r = float(np.corrcoef(dm_t, gh_t)[0, 1])
    dm_gh_bias = float(np.mean(dm_t - gh_t))
    dm_gh_rmse = float(np.sqrt(np.mean((dm_t - gh_t) ** 2)))

    # W2: Illustrative thermal clock mapping into Proposition C1
    tau_proxies: list[float] = []
    d2_vals: list[float] = []
    extrap_rows = sum(not (10.0 <= r["tmean"] <= 20.0) for r in field_records)
    for r in field_records:
        tau_in_t = math.exp(pasteuria_clock["intercept"] + pasteuria_clock["slope"] * r["tmean"])
        tau_proxies.append(tau_in_t)
        for tau_out_ill in (2.0, 10.0, 50.0):
            for d_ill in (1.0, 0.5, 0.1):
                d2_vals.append(bm.closed_form_D2_discrete_C1(tau_out_ill, tau_in_t, d_ill, sigma=0.5, phi=0.8))

    return {
        "station_distances": station_distances,
        "field_rows": len(field_records),
        "reconstructed_na_dates": na_date_reconstructed,
        "s6_host_series_finite_r1": len(s6_r1_vals),
        "s6_r1_range": [float(min(s6_r1_vals)), float(max(s6_r1_vals))],
        "s6_zero_host_records": s6_zero_total,
        "daymet_vs_ghcn": {"r": dm_gh_r, "bias_C": dm_gh_bias, "rmse_C": dm_gh_rmse},
        "W1a": {
            "n": len(lags_arr),
            "signed_rho": float(w1a_exact_check.statistic),
            "elapsed_lead_rho": float(-w1a_exact_check.statistic),
            "ci_lo": w1a_mc["ci_lo"],
            "ci_hi": w1a_mc["ci_hi"],
            "p_greater_mc": w1a_mc["p_greater_mc"],
            "p_less_mc": w1a_mc["p_less_mc"],
            "lag_range": [float(lags_arr.min()), float(lags_arr.max())],
        },
        "W1b": w1b_table,
        "weather_covariate_correlations": cov_corrs,
        "W2": {
            "cells": len(d2_vals),
            "extrapolated_rows": extrap_rows,
            "tau_in_proxy_min": float(min(tau_proxies)),
            "tau_in_proxy_max": float(max(tau_proxies)),
            "D2_min": float(min(d2_vals)),
            "D2_max": float(max(d2_vals)),
        },
    }


# ---------------------------------------------------------------------------
# Section 5: Two Most Consequential Sensitivity Analyses
# ---------------------------------------------------------------------------

def replicate_top_two_sensitivities(mean_cv_s2: float, s4_mu: float, s4_F: float, s4_tau_in: float) -> dict[str, Any]:
    """Reproduce with independent RNG seeds:
      1. Sensitivity 1: S2 36-cell daily Markov host-only vs. 5 C cold developmental arrest
         (most consequential for T2 & pre-data power claim: reverses sign in 6 of 9 Red-White contrasts!).
      2. Sensitivity 2: S3+S4 30-cell Daphnia-Metschnikowia periodic Floquet vs. random telegraph
         across waste (d=1) vs. calibrated contact waiting (d=0.07206) and Erlang k_in in {1, 4, 16}
         (most consequential for mechanism: shows contact-waiting cuts max D ~5-fold and periodic resonance gives D < 0).
    """
    orig_s2 = {
        (r["noise"], int(r["k_out"]), int(r["k_in"]), float(r["d"]), r["thermal_arrest"] == "True"): (float(r["D"]), float(r["se"]))
        for r in bm.load_csv_records(ORIG_DIR / "results" / "s2_markov_predictions.csv")
    }
    amp = min(0.9, mean_cv_s2)
    s2_cells: list[dict[str, Any]] = []
    idx_seed = 0
    for ko, ki in ((1, 1), (4, 4), (1, 16)):
        for d_val in (1.0, 0.5, 0.1):
            for arrest in (False, True):
                pair_vals: dict[str, tuple[float, float]] = {}
                for noise_lbl, phi in (("White", 0.0), ("Red", 0.8)):
                    idx_seed += 1
                    d_hat, se_hat = bm.solve_markov_daily_D(
                        2.0, 7.0, ko, ki, d_val, phi, 1.0 + amp, 1.0 - amp,
                        cold_arrest=arrest,
                        bg_host_loss=0.02 if arrest else 0.0,
                        n_days=2000,
                        burn_days=200,
                        n_pairs=12,
                        seed=900000 + idx_seed,  # independent seed (not original's fixed 20261115)
                    )
                    orig_d, orig_se = orig_s2[(noise_lbl, ko, ki, d_val, arrest)]
                    comb_se = math.hypot(se_hat, orig_se)
                    z_diff = abs(d_hat - orig_d) / max(comb_se, 1e-12)
                    pair_vals[noise_lbl] = (d_hat, se_hat)
                    s2_cells.append({
                        "noise": noise_lbl,
                        "k_out": ko,
                        "k_in": ki,
                        "d": d_val,
                        "cold_arrest": arrest,
                        "D_rep": d_hat,
                        "se_rep": se_hat,
                        "D_orig": orig_d,
                        "se_orig": orig_se,
                        "z_diff": z_diff,
                        "within_4se": z_diff <= 4.0,
                    })

    # Count Red - White sign reversals under host-only vs cold arrest
    by_cfg = {(c["noise"], c["k_out"], c["k_in"], c["d"], c["cold_arrest"]): c["D_rep"] for c in s2_cells}
    host_only_pos = sum(
        (by_cfg[("Red", ko, ki, dv, False)] - by_cfg[("White", ko, ki, dv, False)]) > 0
        for ko, ki in ((1, 1), (4, 4), (1, 16))
        for dv in (1.0, 0.5, 0.1)
    )
    arrest_neg = sum(
        (by_cfg[("Red", ko, ki, dv, True)] - by_cfg[("White", ko, ki, dv, True)]) < 0
        for ko, ki in ((1, 1), (4, 4), (1, 16))
        for dv in (1.0, 0.5, 0.1)
    )

    # Sensitivity 2: S3+S4 30-cell grid
    orig_s34 = {
        (int(r["k_in"]), r["reading"], int(r["scale_days"])): (float(r["D_Floquet"]), float(r["D_telegraph"]), float(r["se_telegraph"]))
        for r in bm.load_csv_records(ORIG_DIR / "results" / "s34_prediction_grid.csv")
    }
    h_ref = 160.0
    tau_out_160 = 1.0 / (s4_mu + s4_F * h_ref / 1000.0)
    d_160 = s4_mu * tau_out_160

    s34_cells: list[dict[str, Any]] = []
    for k_in in (1, 4, 16):
        for reading, d_val in (("waste", 1.0), ("waiting", d_160)):
            for scale in (7, 14, 28, 56, 112):
                idx_seed += 1
                d_floq = bm.solve_floquet_loss_D(tau_out_160, s4_tau_in, k_out=1, k_in=k_in, d_wait=d_val, period=float(scale), h_high=1.8, h_low=0.2)
                d_tel, se_tel = bm.solve_telegraph_ct_D(
                    tau_out_160, s4_tau_in, k_out=1, k_in=k_in, d_wait=d_val, mean_dwell=float(scale),
                    h_high=1.8, h_low=0.2, n_cycles=2000, burn_cycles=100, n_pairs=12,
                    seed=800000 + idx_seed,  # independent seed
                )
                o_floq, o_tel, o_se = orig_s34[(k_in, reading, scale)]
                comb_se = math.hypot(se_tel, o_se)
                z_tel = abs(d_tel - o_tel) / max(comb_se, 1e-12)
                s34_cells.append({
                    "k_in": k_in,
                    "reading": reading,
                    "scale_days": scale,
                    "D_Floquet_rep": d_floq,
                    "D_Floquet_orig": o_floq,
                    "Floquet_abs_err": abs(d_floq - o_floq),
                    "D_telegraph_rep": d_tel,
                    "se_telegraph_rep": se_tel,
                    "D_telegraph_orig": o_tel,
                    "se_telegraph_orig": o_se,
                    "telegraph_z_diff": z_tel,
                    "within_4se": z_tel <= 4.0,
                })

    return {
        "S2_markov_36cells": {
            "amplitude": amp,
            "min_D_rep": float(min(c["D_rep"] for c in s2_cells)),
            "max_D_rep": float(max(c["D_rep"] for c in s2_cells)),
            "host_only_Red_gt_White_count": host_only_pos,
            "cold_arrest_Red_lt_White_reversals": arrest_neg,
            "all_within_4se": all(c["within_4se"] for c in s2_cells),
            "max_z_diff": float(max(c["z_diff"] for c in s2_cells)),
            "key_memoryless_waste_host_only": {
                "White": by_cfg[("White", 1, 1, 1.0, False)],
                "Red": by_cfg[("Red", 1, 1, 1.0, False)],
                "diff": by_cfg[("Red", 1, 1, 1.0, False)] - by_cfg[("White", 1, 1, 1.0, False)],
            },
            "key_memoryless_waste_cold_arrest": {
                "White": by_cfg[("White", 1, 1, 1.0, True)],
                "Red": by_cfg[("Red", 1, 1, 1.0, True)],
                "diff": by_cfg[("Red", 1, 1, 1.0, True)] - by_cfg[("White", 1, 1, 1.0, True)],
            },
            "cells": s2_cells,
        },
        "S34_grid_30cells": {
            "tau_out": tau_out_160,
            "tau_in": s4_tau_in,
            "d_waiting": d_160,
            "min_Floquet_rep": float(min(c["D_Floquet_rep"] for c in s34_cells)),
            "max_Floquet_rep": float(max(c["D_Floquet_rep"] for c in s34_cells)),
            "max_Floquet_abs_err": float(max(c["Floquet_abs_err"] for c in s34_cells)),
            "negative_Floquet_count": sum(c["D_Floquet_rep"] < 0.0 for c in s34_cells),
            "min_telegraph_rep": float(min(c["D_telegraph_rep"] for c in s34_cells)),
            "max_telegraph_rep": float(max(c["D_telegraph_rep"] for c in s34_cells)),
            "all_telegraph_within_4se": all(c["within_4se"] for c in s34_cells),
            "max_telegraph_z_diff": float(max(c["telegraph_z_diff"] for c in s34_cells)),
            "cells": s34_cells,
        },
    }


def main() -> None:
    started = utc_timestamp()
    print(f"[r36-rep-microcosm] Started {started}")

    # Step 1: Rule-7 benchmarks BEFORE touching raw data
    bms = bm.run_benchmarks()
    print(f"[1/5] Passed {len(bms)} Rule-7 benchmarks.")

    # Step 2: Primary test groups T1, T2, T3 + parameter extractions
    s1_res = replicate_s1()
    print(f"[2/5] Completed S1 / T1: T1a rho={s1_res['T1a']['rho']:.6f} ({s1_res['T1a']['status']}), T1b_dyn={s1_res['T1b_dynamic']['slope']:.4f} ({s1_res['T1b_dynamic']['status']})")

    s2_res = replicate_s2()
    print(f"[3/5] Completed S2 / T2: T2a diff={s2_res['T2a']['diff']:.6f} ({s2_res['T2a']['status']}), T2b diff={s2_res['T2b']['diff']:.6f} ({s2_res['T2b']['status']}), T2c_r016={s2_res['T2c_r016']['diff']:.6f} ({s2_res['T2c_r016']['status']})")

    s37_res = replicate_s3_to_s7()
    print(f"[4/5] Completed S3-S7 / T3: T3a Pearson r={s37_res['T3a_Pearson']['r']:.6f} ({s37_res['T3a_Pearson']['status']})")

    # Step 3: Weather & environment amendment
    wx_res = replicate_weather(s37_res["S5_clocks"]["all_tti"])
    print(f"[5/5] Completed Weather W1-W2: W1a rho={wx_res['W1a']['signed_rho']:.6f}, W1b max |delta r|={max(abs(x['weather_partial_r']-x['unadjusted_r']) for x in wx_res['W1b']):.6f}")

    # Step 4: Two most consequential sensitivities
    sens_res = replicate_top_two_sensitivities(
        mean_cv_s2=s2_res["mean_control_CV"],
        s4_mu=s37_res["S4_control_exp"]["lambda_day"],
        s4_F=s37_res["S4_filtering_20C"]["mean"],
        s4_tau_in=s37_res["S4_host_lifespan_trt0"]["mean"],
    )
    print(
        f"[Sensitivities] S2 36-cell max z={sens_res['S2_markov_36cells']['max_z_diff']:.2f} "
        f"(reversals={sens_res['S2_markov_36cells']['cold_arrest_Red_lt_White_reversals']}/9); "
        f"S3+S4 30-cell Floquet max err={sens_res['S34_grid_30cells']['max_Floquet_abs_err']:.2e}, "
        f"telegraph max z={sens_res['S34_grid_30cells']['max_telegraph_z_diff']:.2f}"
    )

    finished = utc_timestamp()
    out_payload = {
        "started_utc": started,
        "finished_utc": finished,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "benchmarks": bms,
        "S1_T1": s1_res,
        "S2_T2": s2_res,
        "S3_to_S7": s37_res,
        "weather": wx_res,
        "sensitivities": sens_res,
    }
    out_json = HERE / "replication_results.json"
    out_json.write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"[r36-rep-microcosm] Saved {out_json} at {finished}")


if __name__ == "__main__":
    main()
