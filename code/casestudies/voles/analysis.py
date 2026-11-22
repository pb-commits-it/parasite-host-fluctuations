#!/usr/bin/env python3
"""Step A registered tests (`T1_mem`, `T2_track`, `T3_order`) for case study `voles`.

Run with isolated Python:
    python3 -I casestudies/voles/analysis.py

Executes the three registered tests of `protocol.md` §4 (with Amendment 1, §7.1)
using `data_loader.py` and `voles_models.py`, after running the Rule 7 parser
and mathematical verification suites. Writes `results/tests.json` and
`results/tests.txt`.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data_loader  # noqa: E402
import voles_models  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _norm_verdict(raw_v: str) -> str:
    """Map internal verdict strings to exact CASESTUDY_SPEC.md strings:
    'supports', 'contradicts', or 'not decided'.
    """
    if raw_v == "supports":
        return "supports"
    if raw_v in ("contradicts", "contradicts_structured_supports_ar1"):
        return "contradicts"
    if raw_v in ("not_decided", "not decided"):
        return "not decided"
    raise ValueError(f"Unexpected verdict token: {raw_v}")


def _clean_memory_fit(fit_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON callables/arrays from `fit_host_memory_panel` output and attach normalized verdict."""
    out = {k: v for k, v in fit_dict.items() if not k.startswith("_")}
    out["verdict"] = _norm_verdict(fit_dict["verdict_T1_mem"])
    return out


def _audit_k15_traps_working() -> dict[str, Any]:
    """Check protocol §4.1 rule: `trapsWorking > 0` on >=90% of K15 capture rows."""
    csv_path = os.path.join(
        data_loader.RAW_VOLES,
        "kielder_eidc_2015_2017",
        "extracted",
        "data",
        "long_blood_qpcr_fv.2.csv",
    )
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        repaired = f.read().replace("TOLLIPA0001,", "TOLLIP\nA0001,", 1)
    reader = csv.DictReader(io.StringIO(repaired))
    n_tot = 0
    n_tw_pos = 0
    for row in reader:
        site = (row.get("site") or "").strip()
        dt = data_loader._parse_uk_or_iso_date(row.get("dateCapt"))
        if not site or dt is None:
            continue
        n_tot += 1
        tw = data_loader._parse_float(row.get("trapsWorking"))
        if tw is not None and tw > 0:
            n_tw_pos += 1
    frac = n_tw_pos / max(n_tot, 1)
    return {
        "n_valid_capture_rows": n_tot,
        "n_rows_trapsWorking_gt_0": n_tw_pos,
        "fraction_trapsWorking_gt_0": round(frac, 6),
        "meets_90pct_effort_rule": bool(frac >= 0.90),
    }


def _split_k08_campaigns(site_series: dict[str, list[dict[str, Any]]], max_gap_steps: int = 6) -> dict[str, list[dict[str, Any]]]:
    """Ensure K08 series are grouped by contiguous site-year campaigns (`protocol.md` §4.2).
    If a site was trapped in one ~12-month campaign, keep as `site`; if separated by >max_gap_steps,
    split into `site_camp1`, `site_camp2`.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for site, wins in sorted(site_series.items()):
        if not wins:
            continue
        ws = sorted(wins, key=lambda w: w["step_k"])
        chunks: list[list[dict[str, Any]]] = [[ws[0]]]
        for w in ws[1:]:
            if w["step_k"] - chunks[-1][-1]["step_k"] > max_gap_steps:
                chunks.append([w])
            else:
                chunks[-1].append(w)
        if len(chunks) == 1:
            out[site] = chunks[0]
        else:
            for idx_c, ch in enumerate(chunks, start=1):
                out[f"{site}_c{idx_c}"] = ch
    return out


def _build_ccf_grids(
    series_by_site: dict[str, list[dict[str, Any]]],
    h_key: str,
    pos_key: str,
    n_key: str,
    min_n: int = 8,
) -> dict[str, Any]:
    """Construct within-site centred 28-day step grids (`z` = ln H, `y` = Haldane-Anscombe ln p,
    `p_raw` = pos/n) over each site's active step span `[min_k, max_k]`.
    """
    site_grids: list[dict[str, Any]] = []
    raw_p_grids: list[dict[str, Any]] = []
    site_summaries: list[dict[str, Any]] = []

    all_z_centred: list[float] = []
    all_y_centred: list[float] = []
    binom_vars: list[float] = []
    total_pos = 0
    total_tested = 0
    n_windows_h_pos = 0
    n_windows_h_zero = 0
    n_windows_path_valid = 0
    n_windows_both_valid = 0

    for site, wins in sorted(series_by_site.items()):
        if not wins:
            continue
        steps = [int(w["step_k"]) for w in wins]
        k0, k1 = min(steps), max(steps)
        L = k1 - k0 + 1
        z_raw = np.full(L, np.nan, dtype=float)
        y_raw = np.full(L, np.nan, dtype=float)
        p_raw = np.full(L, np.nan, dtype=float)

        s_h_pos = 0
        s_h_zero = 0
        s_p_val = 0
        s_both = 0
        s_tested = 0
        s_pos = 0

        for w in wins:
            idx = int(w["step_k"]) - k0
            hv = w.get(h_key)
            h_ok = False
            if hv is not None and not math.isnan(float(hv)):
                if float(hv) > 0.0:
                    z_raw[idx] = math.log(float(hv))
                    s_h_pos += 1
                    h_ok = True
                else:
                    s_h_zero += 1

            nt = int(w.get(n_key, 0) or 0)
            ps = int(w.get(pos_key, 0) or 0)
            p_ok = False
            if nt >= min_n:
                p_hat_ha = (ps + 0.5) / (nt + 1.0)
                y_raw[idx] = math.log(p_hat_ha)
                p_raw[idx] = ps / float(nt)
                binom_vars.append((1.0 - p_hat_ha) / (nt * p_hat_ha))
                s_p_val += 1
                s_tested += nt
                s_pos += ps
                p_ok = True
            if h_ok and p_ok:
                s_both += 1

        n_windows_h_pos += s_h_pos
        n_windows_h_zero += s_h_zero
        n_windows_path_valid += s_p_val
        n_windows_both_valid += s_both
        total_tested += s_tested
        total_pos += s_pos

        valid_z = ~np.isnan(z_raw)
        valid_y = ~np.isnan(y_raw)
        included = bool(np.sum(valid_z) >= 2 and np.sum(valid_y) >= 2)

        site_summaries.append({
            "site": site,
            "step_min": k0,
            "step_max": k1,
            "span_steps": k1 - k0,
            "n_windows_total": len(wins),
            "n_windows_H_gt_0": s_h_pos,
            "n_windows_H_eq_0": s_h_zero,
            "n_windows_tested_ge_min_n": s_p_val,
            "n_windows_joint": s_both,
            "total_animals_tested": s_tested,
            "total_animals_positive": s_pos,
            "pooled_prevalence": round(s_pos / float(s_tested), 4) if s_tested > 0 else None,
            "included_in_pooled_ccf": included,
        })

        if included:
            zc = z_raw - float(np.nanmean(z_raw))
            yc = y_raw - float(np.nanmean(y_raw))
            pc = p_raw - float(np.nanmean(p_raw))
            site_grids.append({"site": site, "z": zc, "y": yc, "covs": {}})
            raw_p_grids.append({"site": site, "z": zc, "y": pc, "covs": {}})
            all_z_centred.extend(zc[valid_z].tolist())
            all_y_centred.extend(yc[valid_y].tolist())

    # Descriptive amplitude attenuation D_atten (protocol §4.4 & §6 #11)
    n_sites_inc = len(site_grids)
    dof_z = max(len(all_z_centred) - n_sites_inc, 1)
    dof_y = max(len(all_y_centred) - n_sites_inc, 1)
    var_z = float(np.sum(np.square(all_z_centred)) / dof_z) if all_z_centred else float("nan")
    var_y = float(np.sum(np.square(all_y_centred)) / dof_y) if all_y_centred else float("nan")
    mean_bvar = float(np.mean(binom_vars)) if binom_vars else 0.0
    var_y_corr = max(0.0, var_y - mean_bvar) if not math.isnan(var_y) else float("nan")
    sd_z = math.sqrt(max(var_z, 0.0)) if not math.isnan(var_z) else float("nan")
    sd_y = math.sqrt(max(var_y, 0.0)) if not math.isnan(var_y) else float("nan")
    sd_y_corr = math.sqrt(var_y_corr) if not math.isnan(var_y_corr) else float("nan")

    return {
        "site_grids": site_grids,
        "raw_p_grids": raw_p_grids,
        "site_summaries": site_summaries,
        "sample_summary": {
            "n_sites_total": len(series_by_site),
            "n_sites_included": n_sites_inc,
            "n_windows_H_gt_0": n_windows_h_pos,
            "n_windows_H_eq_0_excluded": n_windows_h_zero,
            "n_windows_pathogen_n_ge_8": n_windows_path_valid,
            "n_windows_joint_H_and_pathogen": n_windows_both_valid,
            "total_animals_tested_in_valid_windows": total_tested,
            "total_animals_positive_in_valid_windows": total_pos,
            "overall_pooled_prevalence": round(total_pos / float(total_tested), 4) if total_tested > 0 else None,
        },
        "D_atten_descriptive": {
            "sd_ln_H": round(sd_z, 4),
            "sd_ln_p_raw": round(sd_y, 4),
            "mean_binomial_log_var": round(mean_bvar, 4),
            "sd_ln_p_binom_corrected": round(sd_y_corr, 4),
            "ratio_sd_lnp_over_sd_lnH_raw": round(sd_y / sd_z, 4) if sd_z > 1e-12 else None,
            "ratio_sd_lnp_over_sd_lnH_binom_corrected": round(sd_y_corr / sd_z, 4) if sd_z > 1e-12 else None,
        },
    }


def _build_shared_order_sites(
    series_by_site: dict[str, list[dict[str, Any]]],
    h_key: str,
    fast_pos_key: str,
    fast_n_key: str,
    slow_pos_key: str,
    slow_n_key: str,
    min_n: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build shared-host site arrays `(z, y_fast, y_slow)` for `T3_order` block bootstrap."""
    shared_sites: list[dict[str, Any]] = []
    n_win_fast = 0
    n_win_slow = 0
    n_win_all3 = 0

    for site, wins in sorted(series_by_site.items()):
        if not wins:
            continue
        steps = [int(w["step_k"]) for w in wins]
        k0, k1 = min(steps), max(steps)
        L = k1 - k0 + 1
        z_raw = np.full(L, np.nan, dtype=float)
        yf_raw = np.full(L, np.nan, dtype=float)
        ys_raw = np.full(L, np.nan, dtype=float)

        for w in wins:
            idx = int(w["step_k"]) - k0
            hv = w.get(h_key)
            h_ok = hv is not None and not math.isnan(float(hv)) and float(hv) > 0.0
            if h_ok:
                z_raw[idx] = math.log(float(hv))
            nf = int(w.get(fast_n_key, 0) or 0)
            pf = int(w.get(fast_pos_key, 0) or 0)
            ns = int(w.get(slow_n_key, 0) or 0)
            ps = int(w.get(slow_pos_key, 0) or 0)
            if nf >= min_n:
                yf_raw[idx] = math.log((pf + 0.5) / (nf + 1.0))
            if ns >= min_n:
                ys_raw[idx] = math.log((ps + 0.5) / (ns + 1.0))

        vz = ~np.isnan(z_raw)
        vf = ~np.isnan(yf_raw)
        vs = ~np.isnan(ys_raw)
        if np.sum(vz) >= 2 and np.sum(vf) >= 2 and np.sum(vs) >= 2:
            n_win_fast += int(np.sum(vf))
            n_win_slow += int(np.sum(vs))
            n_win_all3 += int(np.sum(vz & vf & vs))
            shared_sites.append({
                "site": site,
                "z": z_raw - float(np.nanmean(z_raw)),
                "y_fast": yf_raw - float(np.nanmean(yf_raw)),
                "y_slow": ys_raw - float(np.nanmean(ys_raw)),
            })

    meta = {
        "n_sites_shared": len(shared_sites),
        "sites_included": [s["site"] for s in shared_sites],
        "n_windows_fast_n_ge_8": n_win_fast,
        "n_windows_slow_n_ge_8": n_win_slow,
        "n_windows_all_three_present": n_win_all3,
    }
    return shared_sites, meta


def run_step_a() -> dict[str, Any]:
    t0 = time.time()
    run_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Rule 7 verification checks before analysing empirical rows
    models_check = voles_models.verify_voles_models()
    bundle = data_loader.load_all_voles_data()
    tw_audit = _audit_k15_traps_working()
    assert tw_audit["meets_90pct_effort_rule"], f"K15 trapsWorking rule failed: {tw_audit}"

    with open(os.path.join(RESULTS_DIR, "pre_data_power.json"), "r", encoding="utf-8") as f:
        predata_doc = json.load(f)
    predata_theo = predata_doc["theoretical_predictions"]

    k15 = bundle["K15"]
    k08 = bundle["K08"]
    f02 = bundle["F02"]
    k08_campaigns = _split_k08_campaigns(k08["site_series"])

    # -----------------------------------------------------------------------
    # 2. Registered Test T1_mem (Host memory beyond AR(1), protocol §4.2-4.4)
    # -----------------------------------------------------------------------
    # F02 Konnevesi core (7-yr series)
    f02_core_wins = f02["core_windows"]
    f02_zero_steps = [w["step_k"] for w in f02_core_wins if not (w["H_per_100_trapnights"] > 0.0)]
    f02_pos_wins = [w for w in f02_core_wins if w["H_per_100_trapnights"] > 0.0]
    t_f02_int = np.array([float(w["step_k"]) for w in f02_pos_wins], dtype=float)
    t_f02_exact = np.array([float(w["t_exact_28d"]) for w in f02_pos_wins], dtype=float)
    z_f02 = np.array([math.log(float(w["H_per_100_trapnights"])) for w in f02_pos_wins], dtype=float)

    fit_f02_core = _clean_memory_fit(voles_models.fit_host_memory_panel([("core", t_f02_int, z_f02)]))
    fit_f02_core["span_steps"] = int(f02_pos_wins[-1]["step_k"] - f02_pos_wins[0]["step_k"])
    fit_f02_core["n_zero_host_windows_excluded"] = len(f02_zero_steps)
    fit_f02_core["zero_host_steps"] = f02_zero_steps

    fit_f02_exact_chk = _clean_memory_fit(voles_models.fit_host_memory_panel([("core", t_f02_exact, z_f02)]))

    # K15 Kielder grids (protocol §4.2: >=12 non-zero 28d obs spanning >=18 steps; §4.4: individual grids >=15 windows)
    k15_panel_qualifying: list[tuple[str, np.ndarray, np.ndarray]] = []
    k15_panel_all: list[tuple[str, np.ndarray, np.ndarray]] = []
    k15_site_inventory: dict[str, Any] = {}
    k15_individual_fits_ge15: dict[str, Any] = {}

    for site, wins in sorted(k15["site_series"].items()):
        pos_w = [w for w in wins if w["H_per_100_traps"] > 0.0]
        zero_w = [w for w in wins if not (w["H_per_100_traps"] > 0.0)]
        t_s = np.array([float(w["step_k"]) for w in pos_w], dtype=float)
        z_s = np.array([math.log(float(w["H_per_100_traps"])) for w in pos_w], dtype=float)
        span_s = int(pos_w[-1]["step_k"] - pos_w[0]["step_k"]) if len(pos_w) >= 2 else 0
        qualifies_4_2 = bool(len(pos_w) >= 12 and span_s >= 18)
        qualifies_indiv_15 = bool(len(pos_w) >= 15)

        k15_site_inventory[site] = {
            "gridTreat": wins[0]["gridTreat"] if wins else [],
            "n_windows_total": len(wins),
            "n_nonzero_windows": len(pos_w),
            "n_zero_windows": len(zero_w),
            "step_min": int(pos_w[0]["step_k"]) if pos_w else None,
            "step_max": int(pos_w[-1]["step_k"]) if pos_w else None,
            "span_steps": span_s,
            "meets_sec4_2_panel_rule_n12_span18": qualifies_4_2,
            "meets_sec4_4_single_grid_rule_n15": qualifies_indiv_15,
        }
        if len(pos_w) >= 2:
            k15_panel_all.append((site, t_s, z_s))
        if qualifies_4_2:
            k15_panel_qualifying.append((site, t_s, z_s))
        if qualifies_indiv_15:
            ind_fit = _clean_memory_fit(voles_models.fit_host_memory_panel([(site, t_s, z_s)]))
            ind_fit["span_steps"] = span_s
            ind_fit["n_zero_host_windows_excluded"] = len(zero_w)
            k15_individual_fits_ge15[site] = ind_fit

    fit_k15_pooled = _clean_memory_fit(voles_models.fit_host_memory_panel(k15_panel_qualifying))
    fit_k15_pooled["sites_included"] = [s for s, _, _ in k15_panel_qualifying]
    fit_k15_pooled["sites_excluded_by_sec4_2"] = [
        s for s, info in k15_site_inventory.items() if not info["meets_sec4_2_panel_rule_n12_span18"]
    ]
    fit_k15_all7_check = _clean_memory_fit(voles_models.fit_host_memory_panel(k15_panel_all))
    fit_k15_all7_check["sites_included"] = [s for s, _, _ in k15_panel_all]

    # K08 1-year campaigns (reported per §4.2 alongside §5.1 pre-data unidentifiability <=15% power)
    k08_panel: list[tuple[str, np.ndarray, np.ndarray]] = []
    k08_indiv_fits: dict[str, Any] = {}
    for camp, wins in sorted(k08_campaigns.items()):
        pos_w = [w for w in wins if w["H_per_100_traps"] > 0.0]
        zero_w = [w for w in wins if not (w["H_per_100_traps"] > 0.0)]
        t_c = np.array([float(w["step_k"]) for w in pos_w], dtype=float)
        z_c = np.array([math.log(float(w["H_per_100_traps"])) for w in pos_w], dtype=float)
        span_c = int(pos_w[-1]["step_k"] - pos_w[0]["step_k"]) if len(pos_w) >= 2 else 0
        if len(pos_w) >= 6:
            k08_panel.append((camp, t_c, z_c))
        if len(pos_w) >= 8:
            c_fit = _clean_memory_fit(voles_models.fit_host_memory_panel([(camp, t_c, z_c)]))
            c_fit["span_steps"] = span_c
            c_fit["n_zero_host_windows_excluded"] = len(zero_w)
            c_fit["note_predata_power"] = "<=15% power on 1-year series (protocol §5.1)"
            k08_indiv_fits[camp] = c_fit

    fit_k08_pooled = _clean_memory_fit(voles_models.fit_host_memory_panel(k08_panel))
    fit_k08_pooled["sites_included"] = [c for c, _, _ in k08_panel]
    fit_k08_pooled["note_predata_power"] = "1-year site campaigns (~9-12 steps); unidentifiable for multiannual cycles (protocol §5.1)"

    t1_results = {
        "criterion": {
            "statistic": "Delta_AICc = min(AICc(M0_iid), AICc(M1_ar1)) - min(AICc(M2_osc), AICc(M3_seas_ar1), AICc(M4_seas_osc))",
            "supports": "Delta_AICc >= 4.0 (structured memory beyond AR(1))",
            "contradicts": "Delta_AICc <= -2.0 (simple AR(1) or iid wins by >= 2.0 AICc)",
            "not_decided": "-2.0 < Delta_AICc < 4.0",
        },
        "primary_multiyear_series": {
            "F02_konnevesi_core_7yr": fit_f02_core,
            "K15_kielder_pooled_multigrid_3yr": fit_k15_pooled,
        },
        "k15_individual_grids_ge15": k15_individual_fits_ge15,
        "k15_site_inventory": k15_site_inventory,
        "secondary_checks": {
            "F02_konnevesi_core_exact_date_check": {
                "best_model": fit_f02_exact_chk["best_model"],
                "delta_aicc_beyond_ar1": fit_f02_exact_chk["delta_aicc_beyond_ar1"],
                "verdict": fit_f02_exact_chk["verdict"],
            },
            "K15_all_7_grids_pooled_check": {
                "n_sites": fit_k15_all7_check["n_sites"],
                "n_obs": fit_k15_all7_check["n_obs"],
                "best_model": fit_k15_all7_check["best_model"],
                "delta_aicc_beyond_ar1": fit_k15_all7_check["delta_aicc_beyond_ar1"],
                "verdict": fit_k15_all7_check["verdict"],
            },
            "K08_pooled_1yr_campaigns": fit_k08_pooled,
            "K08_individual_1yr_campaigns": k08_indiv_fits,
        },
    }

    # -----------------------------------------------------------------------
    # 3. Registered Test T2_track (K9 delayed tracking vs phase-surrogate null)
    # -----------------------------------------------------------------------
    t2_specs = [
        {
            "id": "F02_puuv_core",
            "tier": "primary",
            "label": "Konnevesi bank voles & Puumala hantavirus (F02 core, 2002-2009)",
            "series_map": {"core": f02["core_windows"]},
            "h_key": "H_per_100_trapnights",
            "pos_key": "puuv_pos",
            "n_key": "puuv_n",
            "allowed_window": (3, 8),
            "expect_neg0_late": True,
            "predata_key": "puumala_bank_vole",
            "predata_power_S_pos": 0.980,
        },
        {
            "id": "K15_bartonella",
            "tier": "primary",
            "label": "Kielder field voles & Bartonella spp. (K15 untreated qPCR RQ>=1, 2015-2017)",
            "series_map": k15["site_series"],
            "h_key": "H_per_100_traps",
            "pos_key": "bar_pos",
            "n_key": "bar_n",
            "allowed_window": (2, 8),
            "expect_neg0_late": True,
            "predata_key": "bartonella_flea_borne",
            "predata_power_S_pos": 0.977,
        },
        {
            "id": "K15_babesia",
            "tier": "primary",
            "label": "Kielder field voles & Babesia microti (K15 untreated qPCR RQ>=1, 2015-2017)",
            "series_map": k15["site_series"],
            "h_key": "H_per_100_traps",
            "pos_key": "bab_pos",
            "n_key": "bab_n",
            "allowed_window": (5, 10),
            "expect_neg0_late": True,
            "predata_key": "babesia_tick_borne",
            "predata_power_S_pos": 0.580,
        },
        {
            "id": "K08_cowpox",
            "tier": "secondary_lower_power",
            "label": "Kielder field voles & Cowpox IF serology (K08 longitudinal, 2008-2010)",
            "series_map": k08_campaigns,
            "h_key": "H_per_100_traps",
            "pos_key": "cow_pos",
            "n_key": "cow_n",
            "allowed_window": (1, 5),
            "expect_neg0_late": False,
            "predata_key": "cowpox_seroprevalence",
            "predata_power_S_pos": 0.793,
        },
        {
            "id": "K08_bartonella",
            "tier": "secondary_lower_power",
            "label": "Kielder field voles & Bartonella spp. (K08 longitudinal, 2008-2010)",
            "series_map": k08_campaigns,
            "h_key": "H_per_100_traps",
            "pos_key": "bar_pos",
            "n_key": "bar_n",
            "allowed_window": (2, 8),
            "expect_neg0_late": True,
            "predata_key": "bartonella_flea_borne",
            "predata_power_S_pos": 0.667,
        },
        {
            "id": "K08_babesia",
            "tier": "secondary_lower_power",
            "label": "Kielder field voles & Babesia microti (K08 longitudinal, 2008-2010)",
            "series_map": k08_campaigns,
            "h_key": "H_per_100_traps",
            "pos_key": "bab_pos",
            "n_key": "bab_n",
            "allowed_window": (5, 10),
            "expect_neg0_late": True,
            "predata_key": "babesia_tick_borne",
            "predata_power_S_pos": None,
        },
    ]

    t2_series_out: dict[str, Any] = {}
    for sp in t2_specs:
        built = _build_ccf_grids(
            sp["series_map"],
            h_key=sp["h_key"],
            pos_key=sp["pos_key"],
            n_key=sp["n_key"],
            min_n=8,
        )
        surr_res = voles_models.phase_surrogate_ccf_test(
            built["site_grids"],
            pred_iqr_extended=sp["allowed_window"],
            expect_neg_lag0_and_late_peak=sp["expect_neg0_late"],
            n_surr=2000,
            seed=20261114,
            cov_keys=None,
        )
        raw_p_ccf, raw_p_pairs = voles_models.compute_pooled_ccf(
            built["raw_p_grids"], max_lag=12, min_pairs=6, cov_keys=None
        )
        raw_p_metrics = voles_models.ccf_summary_metrics(raw_p_ccf)

        ccf_arr = np.array([np.nan if v is None else v for v in surr_res["ccf_0_to_12"]], dtype=float)
        pk_0_10 = int(np.nanargmax(ccf_arr[:11])) if not np.all(np.isnan(ccf_arr[:11])) else -1
        pk_0_12 = int(np.nanargmax(ccf_arr)) if not np.all(np.isnan(ccf_arr)) else -1

        theo = predata_theo[sp["predata_key"]]
        d_att = dict(built["D_atten_descriptive"])
        d_att["k9_predata_predicted_ratio"] = theo["attenuation_ratio_sd_lnp_over_sd_lnH"]

        t2_series_out[sp["id"]] = {
            "tier": sp["tier"],
            "label": sp["label"],
            "predata_power_S_pos": sp["predata_power_S_pos"],
            "verdict": _norm_verdict(surr_res["verdict_T2_track"]),
            "sample_summary": built["sample_summary"],
            "site_summaries": built["site_summaries"],
            "registered_test": {
                **surr_res,
                "verdict": _norm_verdict(surr_res["verdict_T2_track"]),
                "peak_lag_0_to_10_check": pk_0_10,
                "peak_lag_0_to_12_check": pk_0_12,
                "k9_predata_ccf_lag0": theo["ccf_lag0"],
                "k9_predata_peak_lag": theo["peak_positive_lag_steps"],
            },
            "untransformed_prevalence_ccf_check": {
                "ccf_0_to_12": [None if v is None else round(v, 4) for v in raw_p_ccf],
                "pair_counts_0_to_12": raw_p_pairs,
                "CCF_lag0": round(float(raw_p_metrics["CCF_lag0"]), 4) if not math.isnan(float(raw_p_metrics["CCF_lag0"])) else None,
                "S_pos_max_1_6": round(float(raw_p_metrics["S_pos_max_1_6"]), 4) if not math.isnan(float(raw_p_metrics["S_pos_max_1_6"])) else None,
                "peak_lag_0_to_8": int(raw_p_metrics["peak_lag_0_to_8"]),
                "L_com_0_to_8": round(float(raw_p_metrics["L_com_0_to_8"]), 4),
            },
            "D_atten_descriptive": d_att,
        }

    t2_results = {
        "criterion": {
            "statistic": "S_pos = max_{ell in 1..6} CCF(ell) via 2,000 phase-randomised host surrogates (seed 20261114), plus peak lag ell* = argmax_{0..8} CCF(ell)",
            "supports": "S_pos > 0 with one-sided surrogate p < 0.05 AND ell* inside pre-data K9 window (PUUV [3,8], Bartonella [2,8], Babesia [5,10], Cowpox sero [1,5])",
            "contradicts": "CCF(ell) < 0 across all ell in 1..6 with two-sided p < 0.05, OR significant positive peak at ell* = 0 when K9 predicts CCF(0) < 0 and ell* >= 4",
            "not_decided": "surrogate p >= 0.05 for S_pos (or significant S_pos outside allowed lag window without meeting contradiction clause)",
        },
        "series": t2_series_out,
    }

    # -----------------------------------------------------------------------
    # 4. Registered Test T3_order (Babesia vs Bartonella L_com lag ordering)
    # -----------------------------------------------------------------------
    k15_shared, k15_sh_meta = _build_shared_order_sites(
        k15["site_series"],
        h_key="H_per_100_traps",
        fast_pos_key="bar_pos",
        fast_n_key="bar_n",
        slow_pos_key="bab_pos",
        slow_n_key="bab_n",
        min_n=8,
    )
    t3_k15 = voles_models.block_bootstrap_lag_order(
        k15_shared, block_len=3, n_boot=2000, seed=20261114
    )
    t3_k15_clean = {
        **t3_k15,
        "verdict": _norm_verdict(t3_k15["verdict_T3_order"]),
        "sample_summary": k15_sh_meta,
    }

    k08_shared, k08_sh_meta = _build_shared_order_sites(
        k08_campaigns,
        h_key="H_per_100_traps",
        fast_pos_key="bar_pos",
        fast_n_key="bar_n",
        slow_pos_key="bab_pos",
        slow_n_key="bab_n",
        min_n=8,
    )
    t3_k08 = voles_models.block_bootstrap_lag_order(
        k08_shared, block_len=3, n_boot=2000, seed=20261114
    )
    t3_k08_clean = {
        **t3_k08,
        "verdict": _norm_verdict(t3_k08["verdict_T3_order"]),
        "sample_summary": k08_sh_meta,
    }

    # Descriptive multi-pathogen check in K08 involving Cowpox serology
    k08_cow_bar_sites, k08_cb_meta = _build_shared_order_sites(
        k08_campaigns,
        h_key="H_per_100_traps",
        fast_pos_key="cow_pos",
        fast_n_key="cow_n",
        slow_pos_key="bar_pos",
        slow_n_key="bar_n",
        min_n=8,
    )
    k08_cow_bar = voles_models.block_bootstrap_lag_order(
        k08_cow_bar_sites, block_len=3, n_boot=2000, seed=20261114
    )

    t3_results = {
        "criterion": {
            "statistic": "Delta L_com = L_com(Babesia, T0~11.2 steps) - L_com(Bartonella, T0~3.7 steps) over lags 0..8 via 2,000 within-site moving-block bootstraps (block_len=3, seed 20261114)",
            "supports": "Delta L_com > 0 with 95% bootstrap CI excluding <= 0 (one-sided bootstrap p < 0.05)",
            "contradicts": "Delta L_com < 0 with 95% bootstrap CI excluding >= 0 (one-sided bootstrap p < 0.05)",
            "not_decided": "95% bootstrap CI straddles 0",
        },
        "primary_K15_babesia_minus_bartonella": t3_k15_clean,
        "secondary_K08_babesia_minus_bartonella": t3_k08_clean,
        "descriptive_K08_bartonella_minus_cowpox": {
            **k08_cow_bar,
            "verdict_descriptive": _norm_verdict(k08_cow_bar["verdict_T3_order"]),
            "sample_summary": k08_cb_meta,
        },
    }

    # -----------------------------------------------------------------------
    # 5. Dataset & Confounder Controls Summary (for Step A confounder table)
    # -----------------------------------------------------------------------
    dataset_overview = {
        "K15": {
            "doi": k15["doi"],
            "date_span": [k15["t_origin"], k15["t_max"]],
            "total_capture_rows": k15["total_capture_rows"],
            "untreated_capture_rows": k15["untreated_capture_rows"],
            "treated_capture_rows_excluded_from_primary_pathogen": k15["treated_capture_rows"],
            "species_counts": k15["species_counts"],
            "grid_treat_by_site": k15["grid_treat_by_site"],
            "trapsWorking_audit": tw_audit,
        },
        "K08": {
            "doi": k08["doi"],
            "date_span": [k08["t_origin"], k08["t_max"]],
            "total_longitudinal_rows": k08["total_longitudinal_rows"],
            "cross_sectional_rows": k08["cross_sectional_workbook_rows"],
            "arriero_2018_rows": k08["arriero_2018_rows"],
            "arriero_distinct_tags": k08["arriero_distinct_tags"],
            "arriero_tags_in_k08_longitudinal": k08["arriero_tags_in_k08_longitudinal"],
            "n_campaigns": len(k08_campaigns),
            "campaigns": sorted(k08_campaigns.keys()),
        },
        "F02": {
            "doi": f02["doi"],
            "date_span": [f02["t_origin"], f02["t_max"]],
            "n_core_raw_sessions": f02["n_core_raw_sessions"],
            "n_core_28d_windows": f02["n_core_28d_windows"],
            "n_core_nonzero_host_windows": len(f02_pos_wins),
            "n_core_zero_host_windows_excluded": len(f02_zero_steps),
            "zero_host_steps": f02_zero_steps,
            "n_seroconversion_intervals": len(f02["seroconversion_series"]),
            "n_f23_annual_rows": len(f02["f23_annual_rows"]),
        },
    }

    output = {
        "step": "A_registered_tests",
        "timestamp_utc": run_utc,
        "elapsed_seconds": round(time.time() - t0, 2),
        "verification": {
            "parser_verification": bundle["parser_verification"],
            "models_verification": models_check,
            "k15_trapsWorking_rule": tw_audit,
        },
        "dataset_overview": dataset_overview,
        "T1_mem": t1_results,
        "T2_track": t2_results,
        "T3_order": t3_results,
    }
    return output


def format_tests_text(res: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CASE STUDY `voles` — STEP A: REGISTERED TESTS (`protocol.md` §4)")
    lines.append(f"Run timestamp (UTC): {res['timestamp_utc']}   Elapsed: {res['elapsed_seconds']:.2f} s")
    lines.append("=" * 78)

    v = res["verification"]
    lines.append("\n[0] CODE & SCHEMA VERIFICATION (study rules Rule 7 & Amendment 1)")
    lines.append(f"  - Parsers verified: {v['parser_verification']['passed']} (Haversine 1 deg = {v['parser_verification']['haversine_1deg_lat_km']} km)")
    lines.append(
        f"  - Models verified:  {v['models_verification']['passed']} "
        f"(V1 Prop A err={v['models_verification']['V1_prop_A_max_abs_err']:.2e}, "
        f"V2 Engine/B2/B8b rel_err={v['models_verification']['V2_engine_vs_B2_B8b_rel_err']:.2e}, "
        f"V3 Block V err={v['models_verification']['V3_block_V_reduction_err']:.2e})"
    )
    tw = v["k15_trapsWorking_rule"]
    lines.append(
        f"  - K15 effort audit: {tw['n_rows_trapsWorking_gt_0']}/{tw['n_valid_capture_rows']} rows "
        f"({100.0 * tw['fraction_trapsWorking_gt_0']:.2f}%) have trapsWorking > 0 (>=90% rule met: {tw['meets_90pct_effort_rule']})"
    )

    ov = res["dataset_overview"]
    lines.append("\n[1] DATASET SAMPLE SIZES & PROTOCOL CONTROLS")
    lines.append(
        f"  - K15 (2015-2017): {ov['K15']['total_capture_rows']} captures across 7 grids "
        f"({ov['K15']['untreated_capture_rows']} untreated control rows; "
        f"{ov['K15']['treated_capture_rows_excluded_from_primary_pathogen']} antiparasitic-treated rows excluded from primary prevalence); "
        f"species={ov['K15']['species_counts']}; gridTreat={ov['K15']['grid_treat_by_site']}"
    )
    lines.append(
        f"  - K08 (2008-2010): {ov['K08']['total_longitudinal_rows']} longitudinal captures across "
        f"{ov['K08']['n_campaigns']} campaigns {ov['K08']['campaigns']} "
        f"(Arriero 2018 overlap: {ov['K08']['arriero_tags_in_k08_longitudinal']}/{ov['K08']['arriero_distinct_tags']} tags)"
    )
    lines.append(
        f"  - F02 (2002-2009): {ov['F02']['n_core_raw_sessions']} core sessions -> {ov['F02']['n_core_28d_windows']} 28-day windows "
        f"({ov['F02']['n_core_nonzero_host_windows']} non-zero host windows; "
        f"{ov['F02']['n_core_zero_host_windows_excluded']} zero-host windows excluded at steps {ov['F02']['zero_host_steps']})"
    )

    # T1_mem
    lines.append("\n" + "-" * 78)
    lines.append("[2] REGISTERED TEST `T1_mem`: HOST MEMORY BEYOND AR(1) (protocol §4.4)")
    lines.append("    Criterion: Delta_AICc = min(M0, M1) - min(M2, M3, M4)")
    lines.append("               >= +4.0 -> supports | <= -2.0 -> contradicts | (-2.0, +4.0) -> not decided")
    lines.append("-" * 78)

    for key, label in [
        ("F02_konnevesi_core_7yr", "F02 Konnevesi core (7-yr bank vole series)"),
        ("K15_kielder_pooled_multigrid_3yr", "K15 Kielder pooled multi-grid (qualifying grids)"),
    ]:
        m = res["T1_mem"]["primary_multiyear_series"][key]
        mods = m["models"]
        lines.append(
            f"  * {label}: VERDICT = {m['verdict'].upper()}\n"
            f"    G={m['n_sites']} site(s), N={m['n_obs']} non-zero 28d obs, SD(ln H)={m['sd_ln_H']:.4f} (Var={m['var_ln_H']:.4f})\n"
            f"    AICc: M0_iid={mods['M0_iid']['aicc']:.2f}, M1_ar1={mods['M1_ar1']['aicc']:.2f} (phi={mods['M1_ar1']['phi']:.3f}), "
            f"M2_osc={mods['M2_osc']['aicc']:.2f}, M3_seas_ar1={mods['M3_seas_ar1']['aicc']:.2f} (phi={mods['M3_seas_ar1']['phi']:.3f}), "
            f"M4_seas_osc={mods['M4_seas_osc']['aicc']:.2f}\n"
            f"    Best model = {m['best_model']}, Delta_AICc = {m['delta_aicc_beyond_ar1']:+.3f}, "
            f"seasonal var share = {m['seasonal_variance_share_best']:.3f}\n"
            f"    Empirical ACF(1..6) = {m['acf_0_to_12'][1:7]}"
        )

    lines.append("\n  Individual K15 grids with >= 15 windows:")
    for site, m in sorted(res["T1_mem"]["k15_individual_grids_ge15"].items()):
        mods = m["models"]
        lines.append(
            f"    - {site} (N={m['n_obs']}, span={m['span_steps']} steps, SD={m['sd_ln_H']:.3f}): "
            f"best={m['best_model']}, Delta_AICc={m['delta_aicc_beyond_ar1']:+.2f} "
            f"[M1={mods['M1_ar1']['aicc']:.1f}, M2={mods['M2_osc']['aicc']:.1f}, M3={mods['M3_seas_ar1']['aicc']:.1f}, M4={mods['M4_seas_osc']['aicc']:.1f}] "
            f"-> {m['verdict']}"
        )

    sc = res["T1_mem"]["secondary_checks"]
    lines.append("\n  Secondary / completeness checks:")
    lines.append(
        f"    - F02 exact-date check: best={sc['F02_konnevesi_core_exact_date_check']['best_model']}, "
        f"Delta_AICc={sc['F02_konnevesi_core_exact_date_check']['delta_aicc_beyond_ar1']:+.3f} -> {sc['F02_konnevesi_core_exact_date_check']['verdict']}"
    )
    lines.append(
        f"    - K15 all 7 grids pooled (G={sc['K15_all_7_grids_pooled_check']['n_sites']}, N={sc['K15_all_7_grids_pooled_check']['n_obs']}): "
        f"best={sc['K15_all_7_grids_pooled_check']['best_model']}, "
        f"Delta_AICc={sc['K15_all_7_grids_pooled_check']['delta_aicc_beyond_ar1']:+.3f} -> {sc['K15_all_7_grids_pooled_check']['verdict']}"
    )
    k08p = sc["K08_pooled_1yr_campaigns"]
    lines.append(
        f"    - K08 pooled 1-yr campaigns (G={k08p['n_sites']}, N={k08p['n_obs']}, pre-data power <=15%): "
        f"best={k08p['best_model']}, Delta_AICc={k08p['delta_aicc_beyond_ar1']:+.3f} -> {k08p['verdict']}"
    )
    for site, m in sorted(sc["K08_individual_1yr_campaigns"].items()):
        lines.append(
            f"      . K08 {site} (N={m['n_obs']}, span={m['span_steps']}): best={m['best_model']}, "
            f"Delta_AICc={m['delta_aicc_beyond_ar1']:+.2f} -> {m['verdict']}"
        )

    # T2_track
    lines.append("\n" + "-" * 78)
    lines.append("[3] REGISTERED TEST `T2_track`: K9 DELAYED PREVALENCE TRACKING (protocol §4.4)")
    lines.append("    Criterion: S_pos = max_{1..6} CCF(ell) > 0 with 1-sided surrogate p < 0.05")
    lines.append("               AND ell* = argmax_{0..8} CCF(ell) inside pre-data K9 window")
    lines.append("-" * 78)
    for sid, srec in res["T2_track"]["series"].items():
        rt = srec["registered_test"]
        ss = srec["sample_summary"]
        da = srec["D_atten_descriptive"]
        raw_chk = srec["untransformed_prevalence_ccf_check"]
        lines.append(
            f"  * {sid} ({srec['tier']}): VERDICT = {srec['verdict'].upper()}\n"
            f"    Sites={ss['n_sites_included']}/{ss['n_sites_total']}, H>0 win={ss['n_windows_H_gt_0']} "
            f"(H=0 excl={ss['n_windows_H_eq_0_excluded']}), n>=8 pathogen win={ss['n_windows_pathogen_n_ge_8']} "
            f"(tested animals={ss['total_animals_tested_in_valid_windows']}, prev={ss['overall_pooled_prevalence']})\n"
            f"    CCF(0)={rt['CCF_lag0']:+.4f}, S_pos(max 1..6)={rt['S_pos_max_1_6']:+.4f} "
            f"(null 95%={rt['null_95pct_S_pos']:+.4f}, p_pos={rt['p_surrogate_S_pos']:.4f}), "
            f"S_delay={rt['S_delay']:+.4f} (p_delay={rt['p_surrogate_S_delay']:.4f})\n"
            f"    Peak lag ell*(0..8)={rt['peak_lag_0_to_8']} (allowed {rt['allowed_peak_lag_window']}; "
            f"0..10={rt['peak_lag_0_to_10_check']}, 0..12={rt['peak_lag_0_to_12_check']}), L_com(0..8)={rt['L_com_0_to_8']:.3f}\n"
            f"    CCF(0..8) = {rt['ccf_0_to_12'][:9]}   pairs(0..8) = {rt['pair_counts_0_to_12'][:9]}\n"
            f"    Raw-p check: CCF(0)={raw_chk['CCF_lag0']}, S_pos={raw_chk['S_pos_max_1_6']}, ell*={raw_chk['peak_lag_0_to_8']}, L_com={raw_chk['L_com_0_to_8']:.3f}\n"
            f"    D_atten: SD(ln p)/SD(ln H) raw={da['ratio_sd_lnp_over_sd_lnH_raw']}, "
            f"binom-corr={da['ratio_sd_lnp_over_sd_lnH_binom_corrected']} (K9 pre-data={da['k9_predata_predicted_ratio']})"
        )

    # T3_order
    lines.append("\n" + "-" * 78)
    lines.append("[4] REGISTERED TEST `T3_order`: RESIDENCE-TIME LAG ORDERING (protocol §4.4)")
    lines.append("    Criterion: Delta L_com = L_com(Babesia) - L_com(Bartonella) > 0 with 95% CI > 0")
    lines.append("-" * 78)
    for key, label in [
        ("primary_K15_babesia_minus_bartonella", "Primary K15 (Babesia minus Bartonella)"),
        ("secondary_K08_babesia_minus_bartonella", "Secondary K08 (Babesia minus Bartonella)"),
    ]:
        r3 = res["T3_order"][key]
        sm = r3["sample_summary"]
        lines.append(
            f"  * {label}: VERDICT = {r3['verdict'].upper()}\n"
            f"    Shared sites={sm['n_sites_shared']} ({sm['sites_included']}), "
            f"Bartonella win={sm['n_windows_fast_n_ge_8']}, Babesia win={sm['n_windows_slow_n_ge_8']}, joint={sm['n_windows_all_three_present']}\n"
            f"    L_com(Bartonella)={r3['L_com_fast']:.4f} (peak={r3['peak_lag_fast']}), "
            f"L_com(Babesia)={r3['L_com_slow']:.4f} (peak={r3['peak_lag_slow']})\n"
            f"    Delta L_com = {r3['diff_slow_minus_fast']:+.4f} steps, "
            f"boot 95% CI = [{r3['boot_ci95'][0]:+.4f}, {r3['boot_ci95'][1]:+.4f}], "
            f"1-sided p(Babesia <= Bartonella) = {r3['p_one_sided_slow_gt_fast']:.4f}"
        )

    dcb = res["T3_order"]["descriptive_K08_bartonella_minus_cowpox"]
    lines.append(
        f"  * Descriptive K08 (Bartonella minus Cowpox serology): "
        f"L_com(Cowpox)={dcb['L_com_fast']:.4f} (peak={dcb['peak_lag_fast']}), "
        f"L_com(Bartonella)={dcb['L_com_slow']:.4f} (peak={dcb['peak_lag_slow']}), "
        f"Delta={dcb['diff_slow_minus_fast']:+.4f}, 95% CI={dcb['boot_ci95']}, p={dcb['p_one_sided_slow_gt_fast']:.4f}"
    )
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run_step_a()
    json_path = os.path.join(RESULTS_DIR, "tests.json")
    txt_path = os.path.join(RESULTS_DIR, "tests.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt = format_tests_text(results)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    print(txt, end="")
    print(f"Saved JSON results to {os.path.relpath(json_path, data_loader.REPO_ROOT)}")
    print(f"Saved text summary to {os.path.relpath(txt_path, data_loader.REPO_ROOT)}")


if __name__ == "__main__":
    main()
