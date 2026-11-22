#!/usr/bin/env python3
"""Master independent replication runner for case study `lpd`.

Recomputes from raw data (`data/raw/lpd/`) and the registered protocol (`protocol.md`):
  1. Rule 7 validation checks (`validate_methods.py`);
  2. All registered primary results (counts, empirical distributions, Test T1 overall,
     CC-BY-only, by taxon, and by taxon x annual/subannual; direct stochastic life-cycle
     evaluation across all 5,526 primary cells; Test T2 and all primary regime fractions);
  3. Protocol-vs-implementation discrepancy audits (BioTIME seasonal modal cutoff,
     GPDD log-equal `PopulationUntransformed` series, CC0 VecDyn licensing);
  4. Weather & environment amendment main adjusted result (`W1`: paired before vs after
     adjustment overall, annual, and subannual, plus audit of pre-r27 vs current weather
     cache and physical calendar lag-1 lookup);
  5. The two sensitivity analyses that most affect the conclusions:
     - `S6_quality_fixedplot` (fixed-plot BioTIME without cross-plot pooling, excluding
       manipulations, and GPDD Reliability >= 3; M=20,177, K=80);
     - `S9b_contact_fine` (contact-limited waiting stage with mortality on skewed bounded
       host; M=1,326 in-grid and full admissible domain check).

Writes only summary statistics (never raw data) to `replication_results.json`.
"""
from __future__ import annotations

import collections
import datetime as dt
import functools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import core_math as cm
import load_compilations as lc
import validate_methods as vm
import weather_clean as wc

CYCLES = [("long", 24.0), ("moderate", 7.5), ("short", 2.3)]
PRIMARY_SIGMAS = [0.0, 0.05, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50, 1.85, 2.25]
PRIMARY_PHIS = [-0.85, -0.65, -0.45, -0.25, -0.10, 0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.96]
FINE_SIGMAS = [0.0, 0.15, 0.35, 0.60, 0.90, 1.20, 1.55, 2.00]
FINE_PHIS = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95]
CONTACT_SIGMAS = [0.0, 0.15, 0.35, 0.60, 0.85, 1.10, 1.32]


@functools.lru_cache(maxsize=8192)
def _cached_ar1_curve(rel_grid: tuple[int, ...]) -> np.ndarray:
    arr = np.array(rel_grid, dtype=float)
    return cm.expected_r1_ratio_ar1_projected(arr, cm.linear_detrend_basis(arr), cm.PHI_CAL_GRID)


@functools.lru_cache(maxsize=8192)
def _cached_spec_null(rel_grid: tuple[int, ...]) -> float:
    return cm.wn_null_spectral_slope(np.array(rel_grid, dtype=float))


def attach_series_estimands(s: dict) -> tuple[dict | None, str | None]:
    el, why = lc.check_primary_eligibility(s, coverage_min=0.75)
    if el is None:
        return None, why
    t_pos, y_log, _, cov = el

    m = cm.estimate_series_quantities(t_pos, y_log, detrend=True)
    if m["r1"] is None or m["sigma2"] <= 1e-10:
        return None, "zero_residual_variance"

    rel_t = tuple((t_pos - t_pos[0]).astype(int).tolist())
    mu0 = cm.wn_null_r1_linear(t_pos)
    r1_bc = float(np.interp(m["r1"], _cached_ar1_curve(rel_t), cm.PHI_CAL_GRID))
    mu_b0 = _cached_spec_null(rel_t)

    rec = {
        "series_id": s["series_id"],
        "database": s["database"],
        "study_id": s["study_id"],
        "licence": s["licence"],
        "cc_by_compatible": s["cc_by_compatible"],
        "taxon_group": s["taxon_group"],
        "taxon_subgroup": s["taxon_subgroup"],
        "species": s["species"],
        "interval_class": s["interval_class"],
        "realm_detail": s["realm_detail"],
        "environment_type": s["environment_type"],
        "reliability": s["reliability"],
        "manipulated": s["manipulated"],
        "coverage": cov,
        "n": m["n"],
        "sigma2": m["sigma2"],
        "sigma": m["sigma"],
        "sigma2_raw": m["sigma2_raw"],
        "r1": m["r1"],
        "mu0_r1": mu0,
        "delta_r1": m["r1"] - mu0,
        "r1_bc": r1_bc,
        "b_spec": m["b_spec"],
        "excess_reddening": mu_b0 - m["b_spec"],
    }
    for lag in range(1, 11):
        rec[f"acf_{lag}"] = m["acf"][lag]
    return rec, None


def attach_regime_classifications(
    rec: dict,
    coarse_eval_fn,
    fine_surfaces: dict[str, cm.CalibratedSurface2D] | None = None,
) -> dict:
    out = dict(rec)
    sig = float(rec["sigma"])
    phi = float(rec["r1_bc"])
    phi_plug = float(np.clip(float(rec["r1"]), -0.95, 0.98))

    for cname, b_val in CYCLES:
        res_d = coarse_eval_fn(cname, b_val, sig, phi)
        D = float(res_d["D"])
        D2 = cm.gamma0_ring_two_stage(1.0, b_val, sig, phi)
        out[f"{cname}_D"] = D
        out[f"{cname}_D2"] = D2
        out[f"{cname}_i_half_bound"] = bool(D <= 0.25 * sig * sig)
        same_sign = (D * D2 > 0.0) or (abs(D) < 1e-4 and abs(D2) < 1e-4)
        out[f"{cname}_iii_30pct"] = bool(same_sign and abs(D - D2) <= 0.30 * abs(D))
        out[f"{cname}_iii_10pct"] = bool(same_sign and abs(D - D2) <= 0.10 * abs(D))

        dE_035 = cm.delta_persistence_index_P2(1.0, b_val, sig, phi, 0.35)
        out[f"{cname}_i_index_P2_0.35"] = bool(dE_035 > 0.0)
        out[f"{cname}_ii_index_P2_0.35"] = bool(phi > 0.0 and dE_035 < 0.0)

        if fine_surfaces is not None and phi >= 0.0:
            f_ev = fine_surfaces[cname].evaluate(sig, phi, allow_outside_direct=True)
            d_fine = float(f_ev["D"])
            d_one = cm.one_stage_unexpanded_loss(4.0, sig)
            d2_fine = cm.gamma0_ring_two_stage(4.0, 4.0 * b_val, sig, (phi ** 0.25) if phi > 0.0 else 0.0)
            out[f"{cname}_i_loss_fine"] = bool(d_fine < d_one)
            out[f"{cname}_ii_loss_fine_unexpanded"] = bool(phi > 0.0 and d_fine > d_one)
            out[f"{cname}_ii_loss_fine_second_order"] = bool(phi > 0.0 and d2_fine > 0.125 * sig * sig)
        else:
            out[f"{cname}_i_loss_fine"] = None
            out[f"{cname}_ii_loss_fine_unexpanded"] = None
            out[f"{cname}_ii_loss_fine_second_order"] = None

    out["short_penalty_zone_T2"] = bool(phi > 0.1919)
    out["short_plugin_penalty_zone_T2"] = bool(phi_plug > 0.1919)
    out["long_strict_failure"] = bool(not out["long_iii_10pct"])
    return out


def audit_protocol_discrepancies(all_series: list[dict], audit_notes: dict) -> dict:
    """Quantify empirical impact of protocol-vs-code discrepancies found during clean-room audit."""
    out = {}

    # 1. GPDD log-equal PopulationUntransformed series (especially the 4 bird series 10039, 10040, 10113, 10114)
    gpdd_by_id = {s["series_id"]: s for s in all_series if s["database"] == "GPDD"}
    log_eq_details = []
    for mid, sp, notes in audit_notes.get("gpdd_log_equal_untransformed_series", []):
        sid = f"GPDD_{mid}"
        s = gpdd_by_id.get(sid)
        if s is None:
            continue
        el, _ = lc.check_primary_eligibility(s)
        vals = s["values"][s["values"] > 0]
        log_eq_details.append({
            "series_id": sid,
            "species": sp,
            "retained_in_primary": el is not None,
            "n_positive": int(len(vals)),
            "min_pos": round(float(np.min(vals)), 4),
            "max_pos": round(float(np.max(vals)), 4),
            "median_pos": round(float(np.median(vals)), 4),
            "units": s["units"],
        })
    out["gpdd_log_equal_series_summary"] = log_eq_details

    # 2. BioTIME seasonal cutoff: protocol.md §2.3 says "2–4 distinct months/year" -> seasonal,
    #    whereas helpers_continued.py:937 uses med_mo >= 2.5 (classifying med_mo == 2.0 as annual).
    med2_ids = {sid for sid, _, _ in audit_notes.get("biotime_studies_median_months_in_2_to_2.5", [])}
    affected_prim = [
        s["series_id"] for s in all_series
        if s["database"] == "BioTIME" and s["series_kind"] == "primary"
        and s["study_id"].replace("BT_STUDY_", "") in med2_ids
        and lc.check_primary_eligibility(s)[0] is not None
    ]
    out["biotime_med_months_2_primary_series_classified_annual"] = {
        "n_studies_in_biotime_meta": len(med2_ids),
        "n_retained_primary_series_affected": len(affected_prim),
        "affected_study_ids": sorted({s.split("_")[1] for s in affected_prim}),
    }

    # 3. CC-BY-only subset: report.md:100 excludes all 10 VecDyn series even though 3 NEON VecDyn series are CC0
    prim_vd_cc0 = [
        s["series_id"] for s in all_series
        if s["database"] == "VecDyn" and s["cc_by_compatible"] and lc.check_primary_eligibility(s)[0] is not None
    ]
    out["vecdyn_permissive_cc0_excluded_from_cc_by_subset"] = {
        "n_series": len(prim_vd_cc0),
        "series_ids": prim_vd_cc0,
    }
    return out


def main() -> None:
    t0 = time.monotonic()
    print("Step 1: Running Rule 7 validation checks...", flush=True)
    val_payload = vm.run_validations()
    c_kernel = vm.get_fast_2x2_kernel()

    print("\nStep 2: Loading raw compilations from data/raw/lpd/...", flush=True)
    all_series, loader_notes = lc.load_all_compilations()
    disc_audit = audit_protocol_discrepancies(all_series, loader_notes)

    primary_metrics = []
    excl_counts = collections.Counter()
    for s in all_series:
        if s["series_kind"] != "primary":
            continue
        rec, why = attach_series_estimands(s)
        if rec is not None:
            primary_metrics.append(rec)
        else:
            excl_counts[why] += 1

    print(f"  Retained primary series: M={len(primary_metrics)}, K={len({r['study_id'] for r in primary_metrics})}", flush=True)

    print("\nStep 3: Building calibrated surfaces (coarse, fine, and S9b contact_fine)...", flush=True)
    t_surf = time.monotonic()
    coarse_surfaces: dict[str, cm.CalibratedSurface2D] = {}
    fine_surfaces: dict[str, cm.CalibratedSurface2D] = {}
    contact_surfaces: dict[str, cm.CalibratedSurface2D] = {}
    for idx_c, (cname, b_val) in enumerate(CYCLES):
        coarse_surfaces[cname] = cm.CalibratedSurface2D(
            "primary_coarse", b_val, PRIMARY_SIGMAS, PRIMARY_PHIS,
            steps=24000, burn=1200, pairs=12, base_seed=710000 + idx_c * 1000,
            product_kernel=c_kernel,
        )
        fine_surfaces[cname] = cm.CalibratedSurface2D(
            "primary_fine", b_val, FINE_SIGMAS, FINE_PHIS,
            steps=24000, burn=1600, pairs=8, base_seed=720000 + idx_c * 1000,
            product_kernel=c_kernel,
        )
        contact_surfaces[cname] = cm.CalibratedSurface2D(
            "contact_fine", b_val, CONTACT_SIGMAS, FINE_PHIS,
            steps=40000, burn=2000, pairs=24, base_seed=730000 + idx_c * 1000,
            product_kernel=c_kernel,
        )
    print(f"  9 surfaces built in {time.monotonic() - t_surf:.2f}s", flush=True)

    print("\nStep 4: Direct stochastic evaluation of all primary coarse cells (1,842 x 3)...", flush=True)
    t_dir = time.monotonic()
    direct_cell_cache: dict[tuple[str, float, float], dict] = {}

    def eval_primary_direct(cname: str, b_val: float, sig: float, phi: float) -> dict:
        key = (cname, round(sig, 12), round(phi, 12))
        if key not in direct_cell_cache:
            direct_cell_cache[key] = cm.solve_threshold_loss_2x2(
                "primary_coarse", b_val, sig, phi,
                steps=24000, burn=1600, pairs=12, seed=810000 + len(direct_cell_cache),
                product_kernel=c_kernel,
            )
        return direct_cell_cache[key]

    primary_mapped = [
        attach_regime_classifications(r, eval_primary_direct, fine_surfaces=fine_surfaces)
        for r in primary_metrics
    ]
    print(f"  {len(direct_cell_cache)} distinct primary coarse cells solved directly in {time.monotonic() - t_dir:.2f}s", flush=True)

    bool_keys = [
        "long_i_half_bound", "long_iii_30pct", "long_iii_10pct", "long_strict_failure",
        "moderate_i_half_bound", "moderate_iii_30pct", "moderate_iii_10pct",
        "short_i_half_bound", "short_iii_30pct", "short_iii_10pct",
        "short_ii_loss_fine_unexpanded", "short_ii_loss_fine_second_order",
        "short_ii_index_P2_0.35", "short_penalty_zone_T2", "short_plugin_penalty_zone_T2",
    ]
    dist_keys = ["sigma2", "sigma", "r1", "r1_bc", "delta_r1", "b_spec", "excess_reddening"]

    print("\nStep 5: Computing study-cluster bootstrap summaries for primary strata...", flush=True)
    primary_summaries = {}
    strata = {
        "overall": primary_mapped,
        "CC_BY_only": [r for r in primary_mapped if r["database"] in ("BioTIME", "GPDD") and r["cc_by_compatible"]],
    }
    for tax in sorted({r["taxon_group"] for r in primary_mapped}):
        strata[f"taxon_group:{tax}"] = [r for r in primary_mapped if r["taxon_group"] == tax]
    for iv in sorted({r["interval_class"] for r in primary_mapped}):
        strata[f"interval_class:{iv}"] = [r for r in primary_mapped if r["interval_class"] == iv]
    for r in primary_mapped:
        broad_iv = "annual" if r["interval_class"] == "annual" else "subannual"
        strata.setdefault(f"taxon_interval:{r['taxon_group']}:{broad_iv}", []).append(r)

    for idx_s, (s_label, s_rows) in enumerate(strata.items()):
        primary_summaries[s_label] = cm.cluster_bootstrap_summary(
            s_rows,
            bool_keys=bool_keys,
            dist_keys=dist_keys if (s_label in ("overall", "CC_BY_only") or s_label.startswith(("taxon_group:", "interval_class:"))) else None,
            n_boot=1000,
            seed=950000 + idx_s * 17,
        )

    # Step 6: Two most consequential sensitivity analyses
    print("\nStep 6a: Reproducing Sensitivity 1 (`S6_quality_fixedplot`)...", flush=True)
    t_s6 = time.monotonic()
    prim_by_id = {r["series_id"]: r for r in primary_metrics}

    def eval_coarse_surface(cname: str, _b_val: float, sig: float, phi: float) -> dict:
        return coarse_surfaces[cname].evaluate(sig, phi, allow_outside_direct=True)

    s6_records = []
    for s in all_series:
        if s["database"] == "BioTIME":
            if s["series_kind"] == "primary" or s["manipulated"]:
                continue
        elif s["database"] == "GPDD":
            if s["reliability"] < 3:
                continue
        if s["series_id"] in prim_by_id:
            rec = prim_by_id[s["series_id"]]
        else:
            rec, _ = attach_series_estimands(s)
        if rec is not None:
            s6_records.append(attach_regime_classifications(rec, eval_coarse_surface, fine_surfaces=None))

    s6_summary = cm.cluster_bootstrap_summary(
        s6_records, bool_keys=bool_keys, dist_keys=["sigma2", "r1_bc", "delta_r1"], n_boot=1000, seed=960001
    )
    print(f"  S6_quality_fixedplot: M={s6_summary['M']}, K={s6_summary['K']}, mean_delta_r1={s6_summary['mean_delta_r1']['estimate']:.4f}, T1={s6_summary['T1']}, T2={s6_summary['T2']} ({time.monotonic() - t_s6:.2f}s)", flush=True)

    print("\nStep 6b: Reproducing Sensitivity 2 (`S9b_contact_fine`)...", flush=True)
    t_s9b = time.monotonic()
    s9b_rows = []
    admissible_max_sigma = math.sqrt(math.log(6.70))  # exact bound 0.875 - 0.125*h_high > 0 <=> Var(h) < 5.70
    s9b_full_admissible_counts = {c: [0, 0] for c, _ in CYCLES}
    for r in primary_metrics:
        sig = float(r["sigma"])
        phi = float(r["r1_bc"])
        row = {"series_id": r["series_id"], "study_id": r["study_id"], "delta_r1": r["delta_r1"], "r1_bc": r["r1_bc"]}
        for cname, b_val in CYCLES:
            ev_grid = contact_surfaces[cname].evaluate(sig, phi, allow_outside_direct=False)
            row[f"contact_grid_{cname}_half_bound"] = (
                bool(ev_grid["D"] <= 0.25 * sig * sig) if ev_grid["D"] is not None else None
            )
            if 0.0 <= phi < 1.0 and 0.0 <= sig < admissible_max_sigma - 1e-6:
                ev_adm = contact_surfaces[cname].evaluate(sig, phi, allow_outside_direct=True)
                s9b_full_admissible_counts[cname][1] += 1
                if ev_adm["D"] <= 0.25 * sig * sig:
                    s9b_full_admissible_counts[cname][0] += 1
        s9b_rows.append(row)

    s9b_bools = [f"contact_grid_{c}_half_bound" for c, _ in CYCLES]
    s9b_summary = cm.cluster_bootstrap_summary(s9b_rows, bool_keys=s9b_bools, dist_keys=None, n_boot=1000, seed=960002)
    s9b_summary["full_admissible_domain"] = {
        c: {
            "admissible_series": s9b_full_admissible_counts[c][1],
            "buffered_count": s9b_full_admissible_counts[c][0],
            "buffered_fraction": s9b_full_admissible_counts[c][0] / max(1, s9b_full_admissible_counts[c][1]),
        }
        for c, _ in CYCLES
    }
    print(f"  S9b_contact_fine in-grid eligible={s9b_summary['fractions']['contact_grid_long_half_bound']['denominator']}, long={s9b_summary['fractions']['contact_grid_long_half_bound']['estimate']:.4f} ({time.monotonic() - t_s9b:.2f}s)", flush=True)

    # Step 7: Weather & Environment Amendment main adjusted result (`W1`)
    print("\nStep 7: Reproducing Weather & Environment Amendment (`W1`)...", flush=True)
    t_w = time.monotonic()
    prim_raw = [s for s in all_series if s["series_kind"] == "primary" and s["series_id"] in prim_by_id]
    land_pts = sorted({
        (float(s["latitude"]), float(s["longitude"]))
        for s in prim_raw
        if s["realm_detail"] != "Marine" and math.isfinite(s["latitude"]) and math.isfinite(s["longitude"])
    })

    # Pre-Amendment-3 cutoff: 2026-11-15T08:30:00Z (between report.md §8 at 08:13:37Z and r27 fetch at 08:49Z)
    cutoff_pre_r27 = dt.datetime(2026, 11, 15, 8, 30, 0, tzinfo=dt.timezone.utc).timestamp()

    def run_w1_variant(label_tag: str, cutoff_ts: float | None, phys_lag: bool, seed_base: int) -> dict:
        store = wc.CleanWeatherStore(lc.RAW_LPD / "weather", land_pts, cutoff_epoch=cutoff_ts)
        paired_rows = []
        adj_rows = []
        reasons = collections.Counter()
        by_src = collections.Counter()
        for s in prim_raw:
            res = store.match_and_adjust_w1(s, physical_calendar_lag1=phys_lag)
            if not res["matched"]:
                reasons[res["reason"]] += 1
                continue
            by_src[res["source"]] += 1
            base_meta = {
                "series_id": s["series_id"],
                "study_id": s["study_id"],
                "taxon_group": s["taxon_group"],
                "interval_class": s["interval_class"],
            }
            p_rec = {**base_meta, **res["paired_metrics"]}
            a_rec = {**base_meta, **res["adjusted_metrics"]}
            paired_rows.append(attach_regime_classifications(p_rec, eval_coarse_surface, fine_surfaces=None))
            adj_rows.append(attach_regime_classifications(a_rec, eval_coarse_surface, fine_surfaces=None))

        w_bools = [
            "long_i_half_bound", "long_iii_30pct", "long_iii_10pct", "long_strict_failure",
            "moderate_i_half_bound", "moderate_iii_30pct", "moderate_iii_10pct",
            "short_i_half_bound", "short_iii_30pct", "short_iii_10pct", "short_penalty_zone_T2",
        ]
        out_v = {
            "variant": label_tag,
            "matched_series": len(adj_rows),
            "matched_clusters": len({r["study_id"] for r in adj_rows}),
            "by_source": dict(by_src),
            "unmatched_reasons": dict(reasons),
            "all_before": cm.cluster_bootstrap_summary(paired_rows, w_bools, None, 1000, seed_base + 1),
            "all_after": cm.cluster_bootstrap_summary(adj_rows, w_bools, None, 1000, seed_base + 2),
            "annual_before": cm.cluster_bootstrap_summary([r for r in paired_rows if r["interval_class"] == "annual"], w_bools, None, 1000, seed_base + 3),
            "annual_after": cm.cluster_bootstrap_summary([r for r in adj_rows if r["interval_class"] == "annual"], w_bools, None, 1000, seed_base + 4),
            "subannual_before": cm.cluster_bootstrap_summary([r for r in paired_rows if r["interval_class"] != "annual"], w_bools, None, 1000, seed_base + 5),
            "subannual_after": cm.cluster_bootstrap_summary([r for r in adj_rows if r["interval_class"] != "annual"], w_bools, None, 1000, seed_base + 6),
        }
        return out_v

    w1_pre_r27_archive = run_w1_variant("pre_r27_archive_pos_host_lag", cutoff_pre_r27, False, 970100)
    w1_current_archive_pos_lag = run_w1_variant("current_archive_pos_host_lag", None, False, 970200)
    w1_current_archive_phys_lag = run_w1_variant("current_archive_physical_calendar_lag", None, True, 970300)
    print(
        f"  W1 pre-r27 archive: {w1_pre_r27_archive['matched_series']}/{w1_pre_r27_archive['matched_clusters']} "
        f"after={w1_pre_r27_archive['all_after']['mean_delta_r1']['estimate']:.4f}; "
        f"current archive (pos lag): {w1_current_archive_pos_lag['matched_series']}/{w1_current_archive_pos_lag['matched_clusters']}; "
        f"current archive (physical calendar lag): {w1_current_archive_phys_lag['matched_series']}/{w1_current_archive_phys_lag['matched_clusters']} "
        f"after={w1_current_archive_phys_lag['all_after']['mean_delta_r1']['estimate']:.4f} ({time.monotonic() - t_w:.2f}s)",
        flush=True,
    )

    output = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "validation": val_payload,
        "eligibility": {
            "primary_retained": len(primary_metrics),
            "primary_clusters": len({r["study_id"] for r in primary_metrics}),
            "by_database": dict(collections.Counter(r["database"] for r in primary_metrics)),
            "by_taxon": dict(collections.Counter(r["taxon_group"] for r in primary_metrics)),
            "exclusions": dict(excl_counts),
        },
        "primary_summaries": primary_summaries,
        "sensitivities": {
            "S6_quality_fixedplot": s6_summary,
            "S9b_contact_fine": s9b_summary,
        },
        "weather_W1": {
            "pre_r27_archive_pos_host_lag": w1_pre_r27_archive,
            "current_archive_pos_host_lag": w1_current_archive_pos_lag,
            "current_archive_physical_calendar_lag": w1_current_archive_phys_lag,
        },
        "protocol_discrepancy_audits": disc_audit,
    }
    dest = HERE / "replication_results.json"
    dest.write_text(json.dumps(output, indent=2))
    print(f"\nCompleted full independent replication in {output['elapsed_seconds']}s -> {dest}", flush=True)


if __name__ == "__main__":
    main()
