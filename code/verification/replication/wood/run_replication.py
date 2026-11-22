#!/usr/bin/env python3
"""Master independent replication script for case study `wood` (r36-rep-wood).

Reproduces from raw data (`data/raw/wood/`) and `protocol.md` alone:
1. Raw dataset & blind trait registry audit + Rule 7 known-result checks.
2. Step 1 Phase 1 Wood et al. (2023) Laplace NB-GLMM reproduction gate (M1-M4).
3. Step 2 Registered primary trait models (omnibus ML AICc + 2,000-draw parametric
   bootstrap LRT, REML meta-regression t-contrasts, one-stage 8,754-row NB-GLMMs
   M3_nhosts_tax, M5_traits_tax, M6_joint_notax, M7_joint_tax, and within-3+ M8).
4. Two most consequential sensitivity analyses:
   - S4_storage_active (storage * active_fraction C1h interaction)
   - S2b_terminal (controlling for dissected life stage terminal_in_fish, n=78)
5. Weather and environment amendment's main adjusted result:
   - W1_ersst (NOAA ERSSTv6 annual mean + monthly SD by host-count group, 8,632 rows),
     plus unadjusted M7 on the exact same 8,632-row subset to separate sample
     restriction from thermal adjustment, plus W2_race (archived Race Rocks SST)
     and the descriptive species-centered annual environmental correlations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy import stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW_WOOD = REPO_ROOT / "data" / "raw" / "wood"
ORIG_WOOD = REPO_ROOT / "casestudies" / "wood"

sys.path.insert(0, str(HERE))
from laplace_nb_glmm import (
    IndependentLaplaceNBGLMM,
    NBGLMMDataset,
    run_rule7_glmm_validation,
)
from meta_regression import (
    fit_mixed_meta_regression,
    holm_bonferroni,
    parametric_bootstrap_lrt,
    run_rule7_meta_validation,
)

PRIMARY_TRAITS = [
    "n_fluc_obligate",
    "storage_stage",
    "spreading",
    "prop_active_transfers",
]


def now_date_str() -> str:
    return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S %Z"], text=True).strip()


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False) + "\n")


def parse_float_na(val: str) -> float:
    v = val.strip()
    if v == "" or v == "NA":
        return math.nan
    return float(v)


def haversine_km(lat1, lon1, lat2, lon2):
    r_earth = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(0.5 * dp) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(0.5 * dl) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(a))


def load_raw_wood_and_traits():
    """Load raw final_dataset.csv, raw psite_lc_complexity.csv, and locked traits_blind.csv."""
    final_csv = RAW_WOOD / "wood_zenodo_7448240" / "final_dataset.csv"
    lc_csv = RAW_WOOD / "wood_github" / "data" / "psite_lc_complexity.csv"
    blind_csv = ORIG_WOOD / "traits_blind.csv"

    blind_bytes = blind_csv.read_bytes()
    blind_sha256 = hashlib.sha256(blind_bytes).hexdigest()
    expected_sha256 = "f4d0202392697dd3cff69f10eecdcd97f6331c27386e3c82e94a5a394332fe35"
    assert blind_sha256 == expected_sha256, f"Hash mismatch on traits_blind.csv: {blind_sha256}"

    with final_csv.open(encoding="utf-8-sig") as f:
        obs_rows = list(csv.DictReader(f))

    with lc_csv.open(encoding="utf-8-sig") as f:
        lc_rows = {r["tim_code"]: r for r in csv.DictReader(f)}

    with blind_csv.open(encoding="utf-8-sig") as f:
        blind_rows_raw = list(csv.DictReader(f))

    # Canonical alphabetical ordering by para_spc
    blind_rows = sorted(blind_rows_raw, key=lambda r: r["para_spc"])
    spc_list = [r["para_spc"] for r in blind_rows]
    spc_to_idx = {s: i for i, s in enumerate(spc_list)}

    # Audit consistency between final_dataset.csv, psite_lc_complexity.csv, and traits_blind.csv
    obs_spc_set = {r["para.spc"] for r in obs_rows}
    assert obs_spc_set == set(spc_list) == set(lc_rows.keys())

    # Check rarity threshold pfoo >= 0.04 on all 97 host-parasite pairs
    hp_counts: dict[str, list[int]] = {}
    for r in obs_rows:
        hp_key = f"{r['fish.spc']} - {r['para.spc']}"
        hp_counts.setdefault(hp_key, []).append(int(r["count"]))
    min_pfoo = min(sum(1 for c in cs if c > 0) / len(cs) for cs in hp_counts.values())
    assert min_pfoo >= 0.04, f"Found pair below 0.04 prevalence: {min_pfoo}"

    # Verify nhosts and par_type match raw files for all 85 codes
    for br in blind_rows:
        s = br["para_spc"]
        assert int(br["nhosts_exact"]) == int(lc_rows[s]["n_hosts"])
        assert int(br["nhosts_wood_3plus"]) == min(int(lc_rows[s]["n_hosts"]), 3)
        assert br["par_type"] == lc_rows[s]["psite_type"]
        assert br["terminal_in_fish"] == lc_rows[s]["terminal"]

    # Verifynhosts in final_dataset.csv vs psite_lc_complexity.csv
    nh_obs_mismatch = 0
    for r in obs_rows:
        if int(r["nhosts"]) != int(lc_rows[r["para.spc"]]["n_hosts"]):
            nh_obs_mismatch += 1

    # Compute categorical design VIFs (matching Dated identifiability clarification)
    nh_3p = np.array([int(r["nhosts_wood_3plus"]) for r in blind_rows], dtype=np.int64)
    T_primary = np.array(
        [[float(r[col]) for col in PRIMARY_TRAITS] for r in blind_rows],
        dtype=np.float64,
    )
    X_cat_design = np.column_stack([
        np.ones(len(blind_rows)),
        (nh_3p == 2).astype(np.float64),
        (nh_3p == 3).astype(np.float64),
        T_primary,
    ])
    vif_names = ["host_2", "host_3plus"] + PRIMARY_TRAITS
    vifs = {}
    for j, nm in enumerate(vif_names, start=1):
        yj = X_cat_design[:, j]
        Xo = np.delete(X_cat_design, j, axis=1)
        b_hat, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
        rss = float(np.sum((yj - Xo @ b_hat) ** 2))
        tss = float(np.sum((yj - yj.mean()) ** 2))
        vifs[nm] = tss / rss

    data_audit = {
        "n_rows": len(obs_rows),
        "n_fish": len({r["fish.id"] for r in obs_rows}),
        "n_fish_species": len({r["fish.spc"] for r in obs_rows}),
        "n_taxa": len(spc_list),
        "n_host_parasite_pairs": len(hp_counts),
        "count_sum": int(sum(int(r["count"]) for r in obs_rows)),
        "year_min": int(min(int(r["year"]) for r in obs_rows)),
        "year_max": int(max(int(r["year"]) for r in obs_rows)),
        "year_mean": float(np.mean([float(r["year"]) for r in obs_rows])),
        "year_sd": float(np.std([float(r["year"]) for r in obs_rows], ddof=1)),
        "min_pair_prevalence_pfoo": float(min_pfoo),
        "nhosts_taxon_counts": {
            "1": int(np.sum(nh_3p == 1)),
            "2": int(np.sum(nh_3p == 2)),
            "3plus": int(np.sum(nh_3p == 3)),
        },
        "nhosts_obs_vs_lc_mismatches": nh_obs_mismatch,
        "blind_traits_sha256": blind_sha256,
        "blind_traits_sha256_verified": True,
        "categorical_design_rank": int(np.linalg.matrix_rank(X_cat_design)),
        "categorical_design_vifs": vifs,
    }
    return obs_rows, blind_rows, spc_list, spc_to_idx, data_audit


def build_glmm_dataset(
    obs_subset: list[dict],
    blind_subset: list[dict],
    full_year_mean: float,
    full_year_sd: float,
    full_lat_mean: float,
    full_lat_sd: float,
    tax_field: str = "par_type",
) -> NBGLMMDataset:
    """Construct NBGLMMDataset preserving full-sample year and latitude standardization."""
    spc_names = [r["para_spc"] for r in blind_subset]
    spc_map = {s: i for i, s in enumerate(spc_names)}

    hp_labels = sorted({f"{r['fish.spc']} - {r['para.spc']}" for r in obs_subset})
    hp_map = {h: i for i, h in enumerate(hp_labels)}

    fish_labels = sorted({r["fish.id"] for r in obs_subset})
    fish_map = {f_id: i for i, f_id in enumerate(fish_labels)}

    tax_labels = sorted({r[tax_field] for r in blind_subset})
    tax_map = {t: i for i, t in enumerate(tax_labels)}

    tax_spc = np.array([tax_map[r[tax_field]] for r in blind_subset], dtype=np.int64)
    nh_spc = np.array([min(int(r["nhosts_exact"]), 3) - 1 for r in blind_subset], dtype=np.int64)

    n = len(obs_subset)
    y = np.empty(n, dtype=np.float64)
    hp_idx = np.empty(n, dtype=np.int64)
    spc_idx = np.empty(n, dtype=np.int64)
    fish_idx = np.empty(n, dtype=np.int64)
    slat = np.empty(n, dtype=np.float64)
    slen = np.empty(n, dtype=np.float64)
    uyear = np.empty(n, dtype=np.float64)

    for i, r in enumerate(obs_subset):
        y[i] = float(r["count"])
        hp_idx[i] = hp_map[f"{r['fish.spc']} - {r['para.spc']}"]
        spc_idx[i] = spc_map[r["para.spc"]]
        fish_idx[i] = fish_map[r["fish.id"]]
        lat_val = parse_float_na(r["lat"])
        slat[i] = 0.0 if math.isnan(lat_val) else (lat_val - full_lat_mean) / full_lat_sd
        slen[i] = float(r["length"])
        uyear[i] = (float(r["year"]) - full_year_mean) / full_year_sd

    return NBGLMMDataset(
        y=y,
        hp_idx=hp_idx,
        spc_idx=spc_idx,
        fish_idx=fish_idx,
        tax_spc=tax_spc,
        tax_obs=tax_spc[spc_idx],
        nh_obs=nh_spc[spc_idx],
        slat=slat,
        slen=slen,
        uyear=uyear,
        n_hp=len(hp_labels),
        n_spc=len(spc_names),
        n_fish=len(fish_labels),
        n_tax=len(tax_labels),
    )


def warm_theta_from_m3(mdl: IndependentLaplaceNBGLMM, m3_pars: dict, slope_init: np.ndarray) -> np.ndarray:
    """Construct warm-start parameter vector from Phase 1 M3 hyperparameters."""
    th = []
    for nm in mdl.param_names:
        if nm.startswith("beta_slope_"):
            j = int(nm.split("_")[-1])
            th.append(float(slope_init[j]))
        elif nm.startswith("beta_env_"):
            th.append(0.0)
        elif nm == "logsigma_tax":
            th.append(-2.5)
        elif nm in m3_pars:
            th.append(float(m3_pars[nm]))
        else:
            th.append(0.0)
    return np.array(th, dtype=np.float64)


def parse_ersst_and_match_fish(obs_rows: list[dict]):
    """Independently parse NOAA ERSSTv6 ASCII and match unique fish specimens."""
    ersst_path = RAW_WOOD / "weather" / "ersst_v6_1880_2019.ascii"
    text = ersst_path.read_text(encoding="utf-8")

    # Grid dimensions: 1680 months (1880-01..2019-12), lat=[50, 48, 46], lon=[-128, -126, -124, -122]
    grid_vals = np.full((1680, 3, 4), np.nan, dtype=np.float64)
    for m in re.finditer(r"^\[(\d+)\]\[(\d+)\],\s*(.+)$", text, re.MULTILINE):
        t_i = int(m.group(1))
        lat_i = int(m.group(2))
        grid_vals[t_i, lat_i, :] = [float(x) for x in m.group(3).split(",")]
    grid_vals[grid_vals < -100.0] = np.nan

    time_line = text.split("sst.time[1680]\n")[1].split("\n")[0]
    days_since_1800 = [float(x) for x in time_line.split(",")]
    dates = [datetime(1800, 1, 1) + timedelta(days=d) for d in days_since_1800]
    assert len(dates) == 1680 and dates[0].year == 1880 and dates[-1].year == 2019 and dates[-1].month == 12

    lats = np.array([50.0, 48.0, 46.0])
    lons = np.array([-128.0, -126.0, -124.0, -122.0])

    valid_cells = []
    annual_lookup: dict[tuple[int, float, float], tuple[float, float]] = {}
    for i_la, la in enumerate(lats):
        for j_lo, lo in enumerate(lons):
            if not np.isfinite(grid_vals[:, i_la, j_lo]).any():
                continue
            valid_cells.append((la, lo))
            for yr in range(1880, 2020):
                m_slice = grid_vals[(yr - 1880) * 12:(yr - 1879) * 12, i_la, j_lo]
                if int(np.isfinite(m_slice).sum()) == 12:
                    annual_lookup[(yr, la, lo)] = (
                        float(np.mean(m_slice)),
                        float(np.std(m_slice, ddof=1)),
                    )
    valid_cells_arr = np.array(valid_cells, dtype=np.float64)

    # Extract unique fish records and total retained parasites per fish
    fish_records: dict[str, dict] = {}
    fish_totals: dict[str, int] = {}
    for r in obs_rows:
        fid = r["fish.id"]
        fish_totals[fid] = fish_totals.get(fid, 0) + int(r["count"])
        if fid not in fish_records:
            lat_v = parse_float_na(r["lat"])
            lon_v = parse_float_na(r["long"])
            yr_v = int(r["year"])
            sst_m, sst_sd, g_la, g_lo, g_dist = math.nan, math.nan, math.nan, math.nan, math.nan
            if not (math.isnan(lat_v) or math.isnan(lon_v)):
                dists = haversine_km(lat_v, lon_v, valid_cells_arr[:, 0], valid_cells_arr[:, 1])
                best_k = int(np.argmin(dists))
                g_la, g_lo = float(valid_cells_arr[best_k, 0]), float(valid_cells_arr[best_k, 1])
                g_dist = float(dists[best_k])
                if (yr_v, g_la, g_lo) in annual_lookup:
                    sst_m, sst_sd = annual_lookup[(yr_v, g_la, g_lo)]
            fish_records[fid] = {
                "fish_id": fid,
                "fish_spc": r["fish.spc"],
                "year": yr_v,
                "lat": lat_v,
                "long": lon_v,
                "grid_lat": g_la,
                "grid_lon": g_lo,
                "grid_dist_km": g_dist,
                "sst_mean": sst_m,
                "sst_monthly_sd": sst_sd,
                "temp_na_rm": parse_float_na(r["temp_na_rm"]),
                "sfish_density": parse_float_na(r["sfish_density"]),
                "ord1": parse_float_na(r["ord1"]),
                "ord2": parse_float_na(r["ord2"]),
            }

    for fid, rec in fish_records.items():
        rec["retained_parasites"] = fish_totals[fid]
        rec["log1p_burden"] = math.log1p(fish_totals[fid])

    # Descriptive species-centered annual pooled correlations
    by_sp_yr: dict[tuple[str, int], list[dict]] = {}
    for rec in fish_records.values():
        by_sp_yr.setdefault((rec["fish_spc"], rec["year"]), []).append(rec)

    cov_names = ["sst_mean", "sst_monthly_sd", "temp_na_rm", "ord1", "ord2"]
    annual_rows = []
    for (sp, yr), recs in sorted(by_sp_yr.items()):
        ar = {"fish_spc": sp, "year": yr}
        for c in cov_names:
            vals = [x[c] for x in recs if not math.isnan(x[c])]
            ar[c] = float(np.mean(vals)) if vals else math.nan
        dens_vals = [x["sfish_density"] for x in recs if not math.isnan(x["sfish_density"])]
        ar["sfish_density"] = dens_vals[0] if dens_vals else math.nan
        ar["log1p_burden"] = float(np.mean([x["log1p_burden"] for x in recs]))
        ar["host_abs_change"] = math.nan
        annual_rows.append(ar)

    # Consecutive-year |Delta sfish_density| within fish species
    sp_groups: dict[str, list[dict]] = {}
    for ar in annual_rows:
        sp_groups.setdefault(ar["fish_spc"], []).append(ar)
    for sp, grp in sp_groups.items():
        grp.sort(key=lambda z: z["year"])
        for i in range(1, len(grp)):
            if grp[i]["year"] - grp[i - 1]["year"] == 1:
                if not (math.isnan(grp[i]["sfish_density"]) or math.isnan(grp[i - 1]["sfish_density"])):
                    grp[i]["host_abs_change"] = abs(grp[i]["sfish_density"] - grp[i - 1]["sfish_density"])

    pooled_corrs = {}
    for target in ["log1p_burden", "sfish_density", "host_abs_change"]:
        sp_means = {}
        for sp, grp in sp_groups.items():
            tvals = [z[target] for z in grp if not math.isnan(z[target])]
            sp_means[sp] = float(np.mean(tvals)) if tvals else math.nan
        yr_agg: dict[int, list[dict]] = {}
        for ar in annual_rows:
            yr_agg.setdefault(ar["year"], []).append(ar)
        yr_summary = []
        for yr, items in sorted(yr_agg.items()):
            row_y = {"year": yr}
            for c in cov_names:
                cv = [z[c] for z in items if not math.isnan(z[c])]
                row_y[c] = float(np.mean(cv)) if cv else math.nan
            tv = [z[target] - sp_means[z["fish_spc"]] for z in items if not math.isnan(z[target])]
            row_y[target] = float(np.mean(tv)) if tv else math.nan
            yr_summary.append(row_y)
        for c in cov_names:
            pairs = [(z[c], z[target]) for z in yr_summary if not (math.isnan(z[c]) or math.isnan(z[target]))]
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            r_val = float(np.corrcoef(xs, ys)[0, 1]) if len(pairs) >= 4 else None
            pooled_corrs[f"{c}_vs_{target}"] = {"r": r_val, "n_years": len(pairs)}

    matched_dists = np.array([
        r["grid_dist_km"] for r in fish_records.values() if not math.isnan(r["grid_dist_km"])
    ])
    matched_years = sorted({
        r["year"] for r in fish_records.values() if not math.isnan(r["sst_mean"])
    })
    weather_meta = {
        "n_unique_fish": len(fish_records),
        "n_geolocated_fish": int(len(matched_dists)),
        "n_missing_coord_fish": int(sum(1 for r in fish_records.values() if math.isnan(r["lat"]))),
        "n_known_unique_coords": len({
            (r["lat"], r["long"]) for r in fish_records.values() if not math.isnan(r["lat"])
        }),
        "ersst_cell_used": sorted({
            (r["grid_lat"], r["grid_lon"]) for r in fish_records.values() if not math.isnan(r["grid_lat"])
        }),
        "ersst_matched_years": len(matched_years),
        "ersst_year_min": int(min(matched_years)),
        "ersst_year_max": int(max(matched_years)),
        "ersst_dist_km_min": float(np.min(matched_dists)),
        "ersst_dist_km_median": float(np.median(matched_dists)),
        "ersst_dist_km_max": float(np.max(matched_dists)),
        "pooled_annual_correlations": pooled_corrs,
    }
    return fish_records, weather_meta


def classify_trait_contrast(beta_val: float, p_val: float, expected_sign: int) -> str:
    if p_val < 0.05 and beta_val * expected_sign > 0:
        return "supports"
    if p_val < 0.05 and beta_val * expected_sign < 0:
        return "contradicts"
    return "not decided"


def main():
    t0_all = time.monotonic()
    out_path = HERE / "results_replication.json"
    results: dict = {"started_at": now_date_str()}

    # ------------------------------------------------------------------
    # Step 0: Rule 7 Validation + Raw Data & Blind Trait Audit
    # ------------------------------------------------------------------
    print("[Step 0] Running Rule 7 validations and raw data audit...", flush=True)
    r7_glmm = run_rule7_glmm_validation()
    r7_meta = run_rule7_meta_validation()
    obs_rows, blind_rows, spc_list, spc_to_idx, data_audit = load_raw_wood_and_traits()
    results["rule7_validation"] = {"glmm": r7_glmm, "meta": r7_meta}
    results["data_audit"] = data_audit
    save_json(out_path, results)

    all_years = np.array([float(r["year"]) for r in obs_rows], dtype=np.float64)
    full_year_mean = float(np.mean(all_years))
    full_year_sd = float(np.std(all_years, ddof=1))

    all_lats = np.array([parse_float_na(r["lat"]) for r in obs_rows], dtype=np.float64)
    full_lat_mean = float(np.nanmean(all_lats))
    full_lat_sd = float(np.nanstd(all_lats, ddof=1))

    ds_full = build_glmm_dataset(
        obs_rows, blind_rows, full_year_mean, full_year_sd, full_lat_mean, full_lat_sd
    )
    nh_spc_idx = np.array([min(int(r["nhosts_exact"]), 3) - 1 for r in blind_rows], dtype=np.int64)
    nh_3plus = nh_spc_idx + 1
    par_type_arr = np.array([r["par_type"] for r in blind_rows])

    # ------------------------------------------------------------------
    # Step 1: Phase 1 Reproduction Gate (Wood et al. 2023 Models 1-4)
    # ------------------------------------------------------------------
    print("[Step 1] Fitting Wood et al. (2023) Phase 1 Laplace NB-GLMMs...", flush=True)
    phase1_specs = [
        ("M1_null", dict(has_year=False)),
        ("M2_year", dict(has_year=True, X_slope=np.ones((len(spc_list), 1)))),
        ("M3_nhosts", dict(
            has_year=True,
            X_slope=np.eye(3)[nh_spc_idx],
            group_int_obs=ds_full.nh_obs,
            n_group_int=3,
        )),
        ("M4_partype", dict(
            has_year=True,
            X_slope=np.eye(ds_full.n_tax)[ds_full.tax_spc],
            group_int_obs=ds_full.tax_obs,
            n_group_int=ds_full.n_tax,
        )),
    ]
    phase1_fits = {}
    phase1_raw = {}
    for name, kw in phase1_specs:
        ts = time.monotonic()
        mdl = IndependentLaplaceNBGLMM(ds_full, **kw)
        fit = mdl.fit()
        elapsed = time.monotonic() - ts
        phase1_raw[name] = fit
        slope_idx = [i for i, nm in enumerate(fit["param_names"]) if nm.startswith("beta_slope_")]
        phase1_fits[name] = {
            "nll": fit["nll"],
            "k": fit["k"],
            "aicc": fit["aicc"],
            "converged": fit["converged"],
            "projected_grad_max": fit["projected_grad_max"],
            "inner_grad_max": fit["inner_grad_max"],
            "hessian_min_eig": fit["hessian_min_eig"],
            "beta_slope": fit["pars"].get("beta_slope"),
            "se_slope": fit["se_theta"][slope_idx] if slope_idx else None,
            "elapsed_s": elapsed,
        }
        print(f"  {name}: NLL={fit['nll']:.6f}, AICc={fit['aicc']:.3f} ({elapsed:.2f}s)", flush=True)

    b_m3 = np.asarray(phase1_fits["M3_nhosts"]["beta_slope"])
    se_m3 = np.asarray(phase1_fits["M3_nhosts"]["se_slope"])
    z_exact = float(stats.norm.ppf(0.975))
    pct_dec = 100.0 * np.expm1(10.0 * b_m3 / full_year_sd)
    pct_lo_z = 100.0 * np.expm1(10.0 * (b_m3 - z_exact * se_m3) / full_year_sd)
    pct_hi_z = 100.0 * np.expm1(10.0 * (b_m3 + z_exact * se_m3) / full_year_sd)
    pct_lo_196 = 100.0 * np.expm1(10.0 * (b_m3 - 1.96 * se_m3) / full_year_sd)
    pct_hi_196 = 100.0 * np.expm1(10.0 * (b_m3 + 1.96 * se_m3) / full_year_sd)

    aicc_order = sorted(phase1_fits.keys(), key=lambda m: phase1_fits[m]["aicc"])
    gate_pass = (
        abs(b_m3[2] - (-0.349)) <= 0.03
        and bool(np.all(np.abs(b_m3[:2]) < z_exact * se_m3[:2]))
        and aicc_order[0] == "M3_nhosts"
        and all(f["converged"] for f in phase1_fits.values())
    )
    assert gate_pass, "Phase 1 reproduction gate failed!"

    results["phase1"] = {
        "models": phase1_fits,
        "aicc_order": aicc_order,
        "delta_aicc_vs_M3": {
            m: phase1_fits[m]["aicc"] - phase1_fits["M3_nhosts"]["aicc"] for m in aicc_order
        },
        "null_minus_M2_aicc": phase1_fits["M1_null"]["aicc"] - phase1_fits["M2_year"]["aicc"],
        "host_group_slopes_per_sd_year": b_m3,
        "host_group_se_per_sd_year": se_m3,
        "percent_per_decade": pct_dec,
        "ci95_percent_exact_z": np.column_stack([pct_lo_z, pct_hi_z]),
        "ci95_percent_196": np.column_stack([pct_lo_196, pct_hi_196]),
        "gate_pass": gate_pass,
    }
    save_json(out_path, results)

    # Extract M2 taxon-level slopes and total ADREPORT SEs
    y_slopes = np.asarray(phase1_raw["M2_year"]["spc_slopes"], dtype=np.float64)
    se_slopes = np.asarray(phase1_raw["M2_year"]["spc_slopes_se"], dtype=np.float64)
    v_slopes = se_slopes ** 2

    # ------------------------------------------------------------------
    # Step 2A: Registered Meta-Regression Models + 2,000-draw Bootstrap LRT
    # ------------------------------------------------------------------
    print("[Step 2A] Fitting registered ML/REML meta-regressions and B=2,000 bootstrap...", flush=True)
    N_mat = np.column_stack([
        np.ones(len(spc_list)),
        (nh_3plus == 2).astype(np.float64),
        (nh_3plus == 3).astype(np.float64),
    ])
    T_mat = np.array(
        [[float(r[col]) for col in PRIMARY_TRAITS] for r in blind_rows],
        dtype=np.float64,
    )
    idx_all = np.arange(len(spc_list))
    mask_3p = nh_3plus == 3
    mask_term = np.array([r["terminal_in_fish"] != "NA" for r in blind_rows], dtype=bool)
    term_vec = np.array(
        [float(r["terminal_in_fish"]) if r["terminal_in_fish"] != "NA" else 0.0 for r in blind_rows],
        dtype=np.float64,
    )

    meta_specs = {
        "M3_nhosts_tax": (N_mat, par_type_arr, idx_all),
        "M5_traits_tax": (np.column_stack([np.ones(len(spc_list)), T_mat]), par_type_arr, idx_all),
        "M6_joint_notax": (np.column_stack([N_mat, T_mat]), None, idx_all),
        "M7_joint_tax": (np.column_stack([N_mat, T_mat]), par_type_arr, idx_all),
        "M8_within_3plus": (
            np.column_stack([np.ones(int(np.sum(mask_3p))), T_mat[mask_3p]]),
            par_type_arr[mask_3p],
            idx_all[mask_3p],
        ),
        # Two most consequential sensitivities (+ matched 78-taxon baseline for S2b)
        "S4_storage_active": (
            np.column_stack([N_mat, T_mat, T_mat[:, 1] * T_mat[:, 3]]),
            par_type_arr,
            idx_all,
        ),
        "S2b_terminal": (
            np.column_stack([N_mat[mask_term], T_mat[mask_term], term_vec[mask_term]]),
            par_type_arr[mask_term],
            idx_all[mask_term],
        ),
        "S2b_terminal_baseline": (
            np.column_stack([N_mat[mask_term], term_vec[mask_term]]),
            par_type_arr[mask_term],
            idx_all[mask_term],
        ),
    }

    meta_fits = {}
    for name, (X_m, grp_m, sub_i) in meta_specs.items():
        ml_res = fit_mixed_meta_regression(y_slopes[sub_i], X_m, v_slopes[sub_i], grp_m, reml=False)
        reml_res = fit_mixed_meta_regression(y_slopes[sub_i], X_m, v_slopes[sub_i], grp_m, reml=True)
        if name in ("M5_traits_tax", "M8_within_3plus"):
            reml_res["holm_4traits"] = holm_bonferroni(reml_res["pvalues"][1:5])
        elif X_m.shape[1] >= 7:
            reml_res["holm_4traits"] = holm_bonferroni(reml_res["pvalues"][3:7])
        meta_fits[name] = {"ML": ml_res, "REML": reml_res}

    boot_res = parametric_bootstrap_lrt(
        y_slopes,
        N_mat,
        np.column_stack([N_mat, T_mat]),
        v_slopes,
        par_type_arr,
        B=2000,
        seed=2026111501,
    )

    expected_signs = [-1, +1, +1, -1]
    m7_reml = meta_fits["M7_joint_tax"]["REML"]
    primary_contrasts_meta = []
    for j, (tr_name, sgn) in enumerate(zip(PRIMARY_TRAITS, expected_signs)):
        idx_c = 3 + j
        b_val = float(m7_reml["beta"][idx_c])
        p_val = float(m7_reml["pvalues"][idx_c])
        h_val = float(m7_reml["holm_4traits"][j])
        primary_contrasts_meta.append({
            "trait": tr_name,
            "expected_sign": sgn,
            "beta": b_val,
            "se": float(m7_reml["se"][idx_c]),
            "ci_lo": float(m7_reml["ci_lo"][idx_c]),
            "ci_hi": float(m7_reml["ci_hi"][idx_c]),
            "p_two_sided": p_val,
            "p_holm": h_val,
            "outcome": classify_trait_contrast(b_val, p_val, sgn),
        })

    m8_reml = meta_fits["M8_within_3plus"]["REML"]
    within3p_contrasts_meta = []
    for j, (tr_name, sgn) in enumerate(zip(PRIMARY_TRAITS, expected_signs)):
        idx_c = 1 + j
        b_val = float(m8_reml["beta"][idx_c])
        p_val = float(m8_reml["pvalues"][idx_c])
        h_val = float(m8_reml["holm_4traits"][j])
        within3p_contrasts_meta.append({
            "trait": tr_name,
            "expected_sign": sgn,
            "beta": b_val,
            "se": float(m8_reml["se"][idx_c]),
            "ci_lo": float(m8_reml["ci_lo"][idx_c]),
            "ci_hi": float(m8_reml["ci_hi"][idx_c]),
            "p_two_sided": p_val,
            "p_holm": h_val,
            "outcome": classify_trait_contrast(b_val, p_val, sgn),
        })

    any_support = any(c["outcome"] == "supports" for c in primary_contrasts_meta)
    any_contradict_abc = any(
        c["beta"] * c["expected_sign"] < 0 and c["p_holm"] < 0.05
        for c in primary_contrasts_meta[:3]
    )
    if boot_res["delta_aicc"] <= -2.0 and boot_res["p_boot"] < 0.05 and any_support:
        omnibus_meta_label = "supports"
    elif boot_res["delta_aicc"] <= -2.0 and any_contradict_abc:
        omnibus_meta_label = "contradicts"
    else:
        omnibus_meta_label = "not decided"

    results["meta_analysis"] = {
        "models": meta_fits,
        "omnibus_bootstrap": {
            k: v for k, v in boot_res.items() if k not in ("null_ml", "alt_ml")
        },
        "omnibus_outcome": omnibus_meta_label,
        "primary_contrasts_M7_REML": primary_contrasts_meta,
        "within_3plus_contrasts_M8_REML": within3p_contrasts_meta,
        "host_3plus_baseline_REML": float(meta_fits["M3_nhosts_tax"]["REML"]["beta"][2]),
        "host_3plus_joint_REML": float(meta_fits["M7_joint_tax"]["REML"]["beta"][2]),
        "S4_ml_delta_aicc_vs_M3": float(
            meta_fits["S4_storage_active"]["ML"]["aicc"] - meta_fits["M3_nhosts_tax"]["ML"]["aicc"]
        ),
        "S2b_ml_delta_aicc_vs_matched_baseline": float(
            meta_fits["S2b_terminal"]["ML"]["aicc"] - meta_fits["S2b_terminal_baseline"]["ML"]["aicc"]
        ),
    }
    save_json(out_path, results)
    print(
        f"  Meta omnibus: ΔAICc={boot_res['delta_aicc']:.3f}, LRT={boot_res['lrt_obs']:.3f}, "
        f"boot p={boot_res['p_boot']:.4f} -> {omnibus_meta_label}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2B & Step 3: Unified One-Stage Laplace NB-GLMM Trait & Sensitivity Fits
    # ------------------------------------------------------------------
    print("[Step 2B & 3] Fitting unified one-stage Laplace NB-GLMMs...", flush=True)
    m3_pars = phase1_raw["M3_nhosts"]["pars"]
    b_nh_contrast = np.array([b_m3[0], b_m3[1] - b_m3[0], b_m3[2] - b_m3[0]], dtype=np.float64)

    unified_specs = [
        ("M3_nhosts_tax", idx_all, N_mat, True, True, b_nh_contrast),
        ("M5_traits_tax", idx_all, np.column_stack([np.ones(len(spc_list)), T_mat]), True, True, np.r_[b_m3.mean(), np.zeros(4)]),
        ("M6_joint_notax", idx_all, np.column_stack([N_mat, T_mat]), False, True, np.r_[b_nh_contrast, np.zeros(4)]),
        ("M7_joint_tax", idx_all, np.column_stack([N_mat, T_mat]), True, True, np.r_[b_nh_contrast, np.zeros(4)]),
        # M8 both with proper n_group_int=0 and with original script's n_group_int=1
        ("M8_within_3plus_proper", idx_all[mask_3p], np.column_stack([np.ones(int(np.sum(mask_3p))), T_mat[mask_3p]]), True, False, np.r_[b_m3[2], np.zeros(4)]),
        ("M8_within_3plus_orig_bog1", idx_all[mask_3p], np.column_stack([np.ones(int(np.sum(mask_3p))), T_mat[mask_3p]]), True, True, np.r_[b_m3[2], np.zeros(4)]),
        # Two selected sensitivities
        ("S4_storage_active", idx_all, np.column_stack([N_mat, T_mat, T_mat[:, 1] * T_mat[:, 3]]), True, True, np.r_[b_nh_contrast, np.zeros(5)]),
        ("S2b_terminal", idx_all[mask_term], np.column_stack([N_mat[mask_term], T_mat[mask_term], term_vec[mask_term]]), True, True, np.r_[b_nh_contrast, np.zeros(5)]),
        ("S2b_terminal_baseline", idx_all[mask_term], np.column_stack([N_mat[mask_term], term_vec[mask_term]]), True, True, np.r_[b_nh_contrast, 0.0]),
    ]

    unified_fits = {}
    for name, sub_idx, X_sl, tax_re, use_bog, b_init in unified_specs:
        ts = time.monotonic()
        blind_sub = [blind_rows[i] for i in sub_idx]
        keep_spc = {r["para_spc"] for r in blind_sub}
        obs_sub = [r for r in obs_rows if r["para.spc"] in keep_spc]
        ds_sub = build_glmm_dataset(
            obs_sub, blind_sub, full_year_mean, full_year_sd, full_lat_mean, full_lat_sd
        )
        if use_bog:
            _, bog_inv = np.unique(ds_sub.nh_obs, return_inverse=True)
            n_bog = int(bog_inv.max()) + 1
        else:
            bog_inv = None
            n_bog = 0
        mdl = IndependentLaplaceNBGLMM(
            ds_sub,
            has_year=True,
            X_slope=X_sl,
            group_int_obs=bog_inv,
            n_group_int=n_bog,
            has_tax_slope_re=tax_re,
        )
        th0 = warm_theta_from_m3(mdl, m3_pars, b_init)
        fit = mdl.fit(theta0=th0)
        elapsed = time.monotonic() - ts
        sl_idx = [i for i, nm in enumerate(fit["param_names"]) if nm.startswith("beta_slope_")]
        b_hat = np.asarray(fit["pars"]["beta_slope"])
        se_hat = np.asarray(fit["se_theta"][sl_idx])
        unified_fits[name] = {
            "n_rows": len(obs_sub),
            "n_taxa": len(blind_sub),
            "k": fit["k"],
            "nll": fit["nll"],
            "aicc": fit["aicc"],
            "converged": fit["converged"],
            "projected_grad_max": fit["projected_grad_max"],
            "inner_grad_max": fit["inner_grad_max"],
            "hessian_min_eig": fit["hessian_min_eig"],
            "beta_slope": b_hat,
            "se_slope": se_hat,
            "ci95_lo_exact_z": b_hat - z_exact * se_hat,
            "ci95_hi_exact_z": b_hat + z_exact * se_hat,
            "ci95_lo_196": b_hat - 1.96 * se_hat,
            "ci95_hi_196": b_hat + 1.96 * se_hat,
            "p_wald": 2.0 * stats.norm.sf(np.abs(b_hat / se_hat)),
            "elapsed_s": elapsed,
        }
        print(f"  unified {name}: NLL={fit['nll']:.6f}, AICc={fit['aicc']:.3f} ({elapsed:.2f}s)", flush=True)

    u_m3 = unified_fits["M3_nhosts_tax"]
    u_m7 = unified_fits["M7_joint_tax"]
    u_m6 = unified_fits["M6_joint_notax"]
    results["unified_glmm"] = {
        "models": unified_fits,
        "delta_aicc_M7_minus_M3_tax": u_m7["aicc"] - u_m3["aicc"],
        "delta_aicc_M6_minus_M3_orig": u_m6["aicc"] - phase1_fits["M3_nhosts"]["aicc"],
        "host_3plus_attenuation": 1.0 - (u_m7["beta_slope"][2] / u_m3["beta_slope"][2]),
        "S4_delta_aicc_vs_M3_tax": unified_fits["S4_storage_active"]["aicc"] - u_m3["aicc"],
        "S2b_delta_aicc_vs_matched_baseline": (
            unified_fits["S2b_terminal"]["aicc"] - unified_fits["S2b_terminal_baseline"]["aicc"]
        ),
    }
    save_json(out_path, results)

    # ------------------------------------------------------------------
    # Step 4: Weather & Environment Amendment (W1_ersst primary + W2_race)
    # ------------------------------------------------------------------
    print("[Step 4] Parsing NOAA ERSSTv6 and fitting environmental adjustment models...", flush=True)
    fish_records, weather_meta = parse_ersst_and_match_fish(obs_rows)

    # Attach sst_mean and sst_monthly_sd to observation rows
    obs_with_env = []
    for r in obs_rows:
        rc = dict(r)
        f_rec = fish_records[r["fish.id"]]
        rc["sst_mean"] = str(f_rec["sst_mean"]) if not math.isnan(f_rec["sst_mean"]) else "NA"
        rc["sst_monthly_sd"] = str(f_rec["sst_monthly_sd"]) if not math.isnan(f_rec["sst_monthly_sd"]) else "NA"
        obs_with_env.append(rc)

    env_sets = [
        ("W1_ersst", ["sst_mean", "sst_monthly_sd"]),
        ("W2_race", ["temp_na_rm"]),
    ]
    weather_fits = {}
    for w_name, cov_cols in env_sets:
        sub_obs = [
            r for r in obs_with_env
            if all(not math.isnan(parse_float_na(r[c])) for c in cov_cols)
        ]
        present_spc = {r["para.spc"] for r in sub_obs}
        sub_blind = [r for r in blind_rows if r["para_spc"] in present_spc]
        ds_w = build_glmm_dataset(
            sub_obs, sub_blind, full_year_mean, full_year_sd, full_lat_mean, full_lat_sd
        )
        nh_sub_3p = np.array([min(int(r["nhosts_exact"]), 3) for r in sub_blind], dtype=np.int64)
        N_sub = np.column_stack([
            np.ones(len(sub_blind)),
            (nh_sub_3p == 2).astype(np.float64),
            (nh_sub_3p == 3).astype(np.float64),
        ])
        T_sub = np.array(
            [[float(r[col]) for col in PRIMARY_TRAITS] for r in sub_blind],
            dtype=np.float64,
        )

        cov_means = {}
        cov_sds = {}
        E_cols = []
        for c in cov_cols:
            arr_c = np.array([float(r[c]) for r in sub_obs], dtype=np.float64)
            m_c = float(np.mean(arr_c))
            s_c = float(np.std(arr_c, ddof=1))
            cov_means[c] = m_c
            cov_sds[c] = s_c
            E_cols.append((arr_c - m_c) / s_c)
        E_mat = np.column_stack(E_cols)
        G_obs = np.eye(3, dtype=np.float64)[ds_w.nh_obs]
        X_env = np.column_stack([
            E_mat[:, j] * G_obs[:, g_k]
            for j in range(len(cov_cols))
            for g_k in range(3)
        ])

        w_entry = {
            "covariates": cov_cols,
            "n_rows": len(sub_obs),
            "n_fish": len({r["fish.id"] for r in sub_obs}),
            "n_taxa": len(sub_blind),
            "n_years": len({int(r["year"]) for r in sub_obs}),
            "covariate_means": cov_means,
            "covariate_sds": cov_sds,
        }

        for lbl, X_sl, env_arg, b_init in [
            ("baseline", N_sub, X_env, b_nh_contrast),
            ("joint", np.column_stack([N_sub, T_sub]), X_env, np.r_[b_nh_contrast, np.zeros(4)]),
            ("unadjusted_joint_same_rows", np.column_stack([N_sub, T_sub]), None, np.r_[b_nh_contrast, np.zeros(4)]),
        ]:
            if w_name != "W1_ersst" and lbl == "unadjusted_joint_same_rows":
                continue
            ts = time.monotonic()
            mdl = IndependentLaplaceNBGLMM(
                ds_w,
                has_year=True,
                X_slope=X_sl,
                group_int_obs=ds_w.nh_obs,
                n_group_int=3,
                has_tax_slope_re=True,
                X_env_obs=env_arg,
            )
            th0 = warm_theta_from_m3(mdl, m3_pars, b_init)
            fit = mdl.fit(theta0=th0)
            elapsed = time.monotonic() - ts
            sl_idx = [i for i, nm in enumerate(fit["param_names"]) if nm.startswith("beta_slope_")]
            ev_idx = [i for i, nm in enumerate(fit["param_names"]) if nm.startswith("beta_env_")]
            w_entry[lbl] = {
                "k": fit["k"],
                "nll": fit["nll"],
                "aicc": fit["aicc"],
                "converged": fit["converged"],
                "projected_grad_max": fit["projected_grad_max"],
                "inner_grad_max": fit["inner_grad_max"],
                "hessian_min_eig": fit["hessian_min_eig"],
                "beta_slope": fit["pars"]["beta_slope"],
                "se_slope": fit["se_theta"][sl_idx],
                "beta_env": fit["pars"].get("beta_env"),
                "se_env": fit["se_theta"][ev_idx] if ev_idx else None,
                "elapsed_s": elapsed,
            }
            print(f"  {w_name}_{lbl}: NLL={fit['nll']:.6f}, AICc={fit['aicc']:.3f} ({elapsed:.2f}s)", flush=True)

        w_entry["delta_aicc"] = w_entry["joint"]["aicc"] - w_entry["baseline"]["aicc"]

        # 3+-host thermal association per degree C (both baseline and joint)
        c0 = cov_cols[0]
        sd0 = cov_sds[c0]
        for which in ("baseline", "joint"):
            b_deg = float(w_entry[which]["beta_env"][2]) / sd0
            se_deg = float(w_entry[which]["se_env"][2]) / sd0
            w_entry[f"thermal_3plus_{which}"] = {
                "log_count_per_C": b_deg,
                "se_per_C": se_deg,
                "pct_per_C": float(100.0 * np.expm1(b_deg)),
                "ci95_lo_pct_exact_z": float(100.0 * np.expm1(b_deg - z_exact * se_deg)),
                "ci95_hi_pct_exact_z": float(100.0 * np.expm1(b_deg + z_exact * se_deg)),
                "ci95_lo_pct_196": float(100.0 * np.expm1(b_deg - 1.96 * se_deg)),
                "ci95_hi_pct_196": float(100.0 * np.expm1(b_deg + 1.96 * se_deg)),
            }

        # Relative shifts vs full-sample M7_joint_tax (and vs same-rows unadjusted M7 for W1)
        b_j = np.asarray(w_entry["joint"]["beta_slope"])
        b_full = np.asarray(u_m7["beta_slope"])
        w_entry["rel_shift_vs_full_M7"] = {
            "host_3plus": float(abs((b_j[2] - b_full[2]) / b_full[2])),
            "n_fluc_obligate": float(abs((b_j[3] - b_full[3]) / b_full[3])),
            "storage_stage": float(abs((b_j[4] - b_full[4]) / b_full[4])),
        }
        if "unadjusted_joint_same_rows" in w_entry:
            b_same = np.asarray(w_entry["unadjusted_joint_same_rows"]["beta_slope"])
            w_entry["rel_shift_vs_same_rows_unadjusted_M7"] = {
                "host_3plus": float(abs((b_j[2] - b_same[2]) / b_same[2])),
                "n_fluc_obligate": float(abs((b_j[3] - b_same[3]) / b_same[3])),
                "storage_stage": float(abs((b_j[4] - b_same[4]) / b_same[4])),
            }
        weather_fits[w_name] = w_entry
        results["weather_amendment"] = {
            "metadata_and_descriptives": weather_meta,
            "adjusted_models": weather_fits,
        }
        save_json(out_path, results)

    results["completed_at"] = now_date_str()
    results["total_elapsed_seconds"] = time.monotonic() - t0_all
    save_json(out_path, results)
    print(f"ALL DONE in {results['total_elapsed_seconds']:.2f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
