#!/usr/bin/env python3
"""Main independent replication runner for `r36-rep-chemostat`.

Reproduces from raw data (`data/raw/chemostat/`) and registered `protocol.md` alone:
  1. All registered primary results across D1-D5 (T1 primary & secondary observables,
     T2a Proposition B bound, T2b contact-waiting ratio & Gamma_c accuracy, T3 competing
     mechanisms, plus D6 density-column exclusion check);
  2. Weather & environment amendment (`Amendment 3`) laboratory regime inventory,
     D4 reconstructed thermal treatment summaries, and descriptive Spearman correlations;
  3. The two sensitivity analyses that most affect the study's conclusions:
     - Sensitivity A (`S2`): Susceptible-host H_S(t) resistance adjustment on D1, D4, D5
       (and D1 abortive-adsorption sink bound), which flips D1 to `supports` (-0.8722);
     - Sensitivity B (`S4` D4 thermal confounding): matched-mean-30C switching lines (n=18),
       excluding lethal constant 32C (n=24), and linear thermal-adjusted r_const (n=30),
       plus an audit of multi-covariate Freedman-Lane thermal adjustment on full n=30;
  4. Quantitative discrepancy diagnostics (control vs infected t=0 mismatch in D1/D3/D4,
     non-positive lag-1 sub-step white-noise fallback in Gamma_c, D2 unweighted vs
     time-weighted washout in r_phage_emp and pulse operator alignment, D4/D5 first-valid-host
     interval drop when t=0 is NA).

Zero code imported or copied from `casestudies/chemostat/`.
Run with: python3 -I verification/replication/chemostat/run_replication.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy import optimize

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import core_math as cm
import raw_parsers as rp

ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW_DIR = ROOT / "data" / "raw" / "chemostat"
ORIG_RESULTS = ROOT / "casestudies" / "chemostat" / "results"


def prepare_unit_trajectory(
    rec: rp.TrajectoryRecord,
    min_pos_h: float,
    min_pos_p: float,
    floor_mult: float = 0.5,
) -> dict:
    """Apply pre-registered missing-value omission and half-LOD zero-floor rules (`protocol.md` §6.1)."""
    h_floor = floor_mult * min_pos_h
    p_floor = floor_mult * min_pos_p

    h_pts = [
        (float(rec.times_d[i]), float(rec.host_cells_ml[i]) if float(rec.host_cells_ml[i]) > 0.0 else h_floor, float(rec.washout_d[i]))
        for i in range(len(rec.times_d))
        if not math.isnan(rec.host_cells_ml[i])
    ]
    n_h_zeros = sum(1 for i in range(len(rec.times_d)) if not math.isnan(rec.host_cells_ml[i]) and rec.host_cells_ml[i] <= 0.0)

    if len(h_pts) >= 2:
        t_all = np.array([x[0] for x in h_pts], dtype=float)
        dts = np.diff(t_all)
        h_seq = np.array([x[1] for x in h_pts[1:]], dtype=float)
        w_seq = np.array([x[2] for x in h_pts[1:]], dtype=float)
        h_all = np.array([x[1] for x in h_pts], dtype=float)
    else:
        t_all = np.array([x[0] for x in h_pts], dtype=float)
        dts = np.array([], dtype=float)
        h_seq = np.array([], dtype=float)
        w_seq = np.array([], dtype=float)
        h_all = np.array([x[1] for x in h_pts], dtype=float)

    p_pts = [
        (float(rec.times_d[i]), float(rec.phage_pfu_ml[i]), float(rec.washout_d[i]))
        for i in range(len(rec.times_d))
        if not math.isnan(rec.phage_pfu_ml[i])
    ]
    n_p_zeros = sum(1 for _, pv, _ in p_pts if pv <= 0.0)
    extinct_end = bool(len(p_pts) > 0 and p_pts[-1][1] <= 0.0)

    if rec.infected and len(p_pts) >= 2:
        tp = np.array([x[0] for x in p_pts], dtype=float)
        pc = np.array([x[1] if x[1] > 0.0 else p_floor for x in p_pts], dtype=float)
        post = (tp > 0.0) if (tp[0] == 0.0 and np.sum(tp > 0.0) >= 1) else np.ones(len(tp), dtype=bool)
        y_mean = float(np.mean(np.log(pc[post])))

        # Registered calculation vs strict time-weighted washout audit:
        w_unweighted = float(np.mean([x[2] for x in p_pts[1:]]))
        dt_p = np.diff(tp)
        w_timeweighted = float(np.sum(dt_p * np.array([x[2] for x in p_pts[1:]], dtype=float)) / np.sum(dt_p))

        net_slope = (math.log(pc[-1]) - math.log(pc[0])) / max(tp[-1] - tp[0], 1e-9)
        r_emp_reg = net_slope + w_unweighted
        r_emp_tw_audit = net_slope + w_timeweighted

        if rec.dataset_id == "D1_larsen2019":
            idx_early = [j for j in range(len(tp)) if tp[j] <= 9.0]
            j_e = idx_early[-1] if len(idx_early) >= 2 else 1
        else:
            j_e = 1
        y_early = (math.log(pc[j_e]) - math.log(pc[0])) / max(tp[j_e] - tp[0], 1e-9) + float(p_pts[j_e][2])
    else:
        y_mean = math.nan
        r_emp_reg = math.nan
        r_emp_tw_audit = math.nan
        y_early = math.nan

    return {
        "t_all": t_all,
        "h_all": h_all,
        "h_seq": h_seq,
        "dts": dts,
        "w_seq": w_seq,
        "n_h_obs": len(h_pts),
        "n_p_obs": len(p_pts),
        "n_h_zeros": n_h_zeros,
        "n_p_zeros": n_p_zeros,
        "went_extinct": extinct_end,
        "y_mean_lnP": y_mean,
        "r_phage_emp_d": r_emp_reg,
        "r_phage_emp_tw_audit_d": r_emp_tw_audit,
        "y_early_d": y_early,
    }


def susceptible_host_series(
    rec: rp.TrajectoryRecord,
    interval_end_days: np.ndarray,
    h_seq: np.ndarray,
    ancestral_assay: bool,
    refuge_floor: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Interpolate assayed bacterial resistance onto interval end times (`protocol.md` §5 #13)."""
    if len(rec.res_days) == 0:
        return h_seq.copy(), math.nan
    r_src = rec.res_ancestral if ancestral_assay else rec.res_contemporary
    q_int = np.interp(interval_end_days, rec.res_days, r_src, left=float(r_src[0]), right=float(r_src[-1]))
    q_cap = np.clip(q_int, 0.0, 1.0 - refuge_floor)
    return h_seq * (1.0 - q_cap), float(np.mean(q_int))


def evaluate_d2_pulse_chloroform(rec: rp.TrajectoryRecord, lh: cm.PhageLifeHistory, align_start: bool = False) -> dict[str, float]:
    """Batch-incubation + discrete-dilution + chloroform-killing monodromy operator for D2."""
    nu = 1.0 / lh.latent_period_d
    alpha = lh.adsorption_ml_d
    mu0 = lh.virion_decay_d

    dts = np.diff(rec.times_d)
    hosts = rec.host_cells_ml[1:]
    k_tr = len(dts)
    span = float(np.sum(dts))
    h_bar = float(np.sum(dts * hosts) / span)

    f_vec = list(rec.extra["f_dilutions"])
    stypes = list(rec.extra["step_types"])
    chloro = ([True] * k_tr) if rec.extra["regime"] == "3h" else [(s == "T5") for s in stypes]

    def growth(h_arr: np.ndarray, f_arr: list[float], tilt: float) -> float:
        u_mat = np.eye(2, dtype=float)
        acc = 0.0
        b_rate = math.exp(tilt) * lh.burst_size * nu
        for k in range(k_tr):
            ek, ls = cm.exact_2x2_scaled_expm(-nu, alpha * float(h_arr[k]), b_rate, -(alpha * float(h_arr[k]) + mu0), float(dts[k]))
            acc += ls
            d_op = np.array([[0.0, 0.0], [0.0, float(f_arr[k])]]) if chloro[k] else (float(f_arr[k]) * np.eye(2))
            u_mat = (ek @ d_op @ u_mat) if align_start else (d_op @ ek @ u_mat)
            nrm = float(np.linalg.norm(u_mat, ord=np.inf))
            if nrm > 0.0:
                u_mat /= nrm
                acc += math.log(nrm)
        rad = float(np.max(np.abs(np.linalg.eigvals(u_mat))))
        return (acc + math.log(max(rad, 1e-300))) / span

    f_bar = float(np.mean(f_vec))
    h_const = np.full(k_tr, h_bar, dtype=float)
    f_const = [f_bar] * k_tr

    r_const = growth(h_const, f_const, 0.0)
    r_obs = growth(hosts, f_vec, 0.0)
    y_const = float(optimize.brentq(lambda y: growth(h_const, f_const, y), -15.0, 15.0, xtol=1e-10))
    y_obs = float(optimize.brentq(lambda y: growth(hosts, f_vec, y), -15.0, 15.0, xtol=1e-10))
    y_fonly = float(optimize.brentq(lambda y: growth(h_const, f_vec, y), -15.0, 15.0, xtol=1e-10))

    return {
        "r_pulse_const_d": r_const,
        "r_pulse_obs_d": r_obs,
        "D_pulse_total": y_obs - y_const,
        "D_pulse_from_dilution_var": y_fonly - y_const,
        "D_pulse_from_host_var": y_obs - y_fonly,
    }


def main() -> None:
    utc_stamp = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
    local_stamp = subprocess.check_output(["date"], text=True).strip()

    d1 = rp.parse_d1_larsen(RAW_DIR)
    d2, d2_checks = rp.parse_d2_reuter(RAW_DIR)
    d3, d3_checks = rp.parse_d3_chaudhry(RAW_DIR)
    d4 = rp.parse_d4_duncan(RAW_DIR)
    d5 = rp.parse_d5_zhao(RAW_DIR, od_cfu_factor=5.0e8)

    all_datasets = [
        ("D1_larsen2019", d1),
        ("D2_reuter2026", d2),
        ("D3_chaudhry2018", d3),
        ("D4_duncan2017", d4),
        ("D5_zhao2016", d5),
    ]

    unit_rows: list[dict] = []
    inf_by_ds: dict[str, list[dict]] = {}
    ctrl_by_ds: dict[str, list[dict]] = {}
    d2_pulse_reg: list[dict] = []
    d2_pulse_aligned_audit: list[dict] = []

    for ds_id, records in all_datasets:
        lh = cm.REGISTERED_LIFE_HISTORY[ds_id]
        pos_h = [float(v) for r in records for v in r.host_cells_ml if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for r in records for v in r.phage_pfu_ml if not math.isnan(v) and v > 0.0]
        min_h = min(pos_h) if pos_h else 1.0
        min_p = min(pos_p) if pos_p else 1.0

        inf_by_ds[ds_id] = []
        ctrl_by_ds[ds_id] = []

        for rec in records:
            prep = prepare_unit_trajectory(rec, min_h, min_p, floor_mult=0.5)
            if len(prep["h_seq"]) < 2:
                continue

            if not rec.infected:
                sig_all = float(np.std(np.log(prep["h_all"]), ddof=1))
                sig_post = float(np.std(np.log(prep["h_seq"]), ddof=1))
                ctrl_by_ds[ds_id].append({
                    "unit_id": rec.unit_id,
                    "sigma_lnH_incl_first": sig_all,
                    "sigma_lnH_excl_first": sig_post,
                })
                continue

            ct_contact = cm.evaluate_trajectory_ct(prep["h_seq"], prep["dts"], prep["w_seq"], prep["w_seq"], lh, "contact")
            ct_waste = cm.evaluate_trajectory_ct(prep["h_seq"], prep["dts"], prep["w_seq"], prep["w_seq"], lh, "waste")
            hm = cm.host_moments_and_gamma_c(
                prep["h_seq"], prep["dts"], ct_contact["tau_P_d"], ct_contact["tau_I_d"], ct_contact["d_threshold"], lh.erlang_m
            )
            pb = cm.verify_proposition_b_bound(prep["h_seq"], prep["dts"], prep["w_seq"], prep["w_seq"], lh)

            res_loss = 0.0
            d_sim_hs = math.nan
            d_thresh_hs = math.nan
            r_const_hs = math.nan
            mean_q = math.nan
            if len(rec.res_days) > 0:
                hs_seq, mean_q = susceptible_host_series(
                    rec, prep["t_all"][1:], prep["h_seq"], ancestral_assay=(ds_id == "D1_larsen2019")
                )
                ct_hs = cm.evaluate_trajectory_ct(hs_seq, prep["dts"], prep["w_seq"], prep["w_seq"], lh, "contact")
                res_loss = ct_contact["ln_R0_mean"] - ct_hs["ln_R0_mean"]
                d_sim_hs = ct_hs["D_sim"]
                d_thresh_hs = ct_hs["d_threshold"]
                r_const_hs = ct_hs["r_const_d"]

            if ds_id == "D2_reuter2026":
                d2_pulse_reg.append({"unit_id": rec.unit_id, "treatment": rec.treatment, **evaluate_d2_pulse_chloroform(rec, lh, align_start=False)})
                d2_pulse_aligned_audit.append({"unit_id": rec.unit_id, "treatment": rec.treatment, **evaluate_d2_pulse_chloroform(rec, lh, align_start=True)})

            urow = {
                "dataset_id": ds_id,
                "unit_id": rec.unit_id,
                "treatment": rec.treatment,
                "n_h_obs": prep["n_h_obs"],
                "n_p_obs": prep["n_p_obs"],
                "went_extinct": int(prep["went_extinct"]),
                "H_bar": ct_contact["H_bar"],
                "sigma_lnH": float(hm["sigma_lnH"]),
                "cv_H": float(hm["cv_H"]),
                "phi_lag1": float(hm["phi_lag1"]),
                "d_threshold": ct_contact["d_threshold"],
                "tau_P_d": ct_contact["tau_P_d"],
                "tau_I_d": ct_contact["tau_I_d"],
                "ln_R0_mean": ct_contact["ln_R0_mean"],
                "r_const_d": ct_contact["r_const_d"],
                "r_obs_d": ct_contact["r_obs_d"],
                "Delta_r_fluct_d": ct_contact["Delta_r_fluct_d"],
                "D_sim_contact": ct_contact["D_sim"],
                "D_sim_waste": ct_waste["D_sim"],
                "Gamma_c_2nd": float(hm["Gamma_c_2nd"]),
                "Gamma_c_pw_audit": float(hm["Gamma_c_pw_audit"]),
                "Gamma_waste_2nd": float(hm["Gamma_waste_2nd"]),
                "D_disc_propB": float(pb["D_disc"]),
                "bound_transfer_only": float(pb["bound_transfer_only"]),
                "stay_penalty_propB": float(pb["stay_penalty"]),
                "bound_prop_b": float(pb["bound_prop_b"]),
                "holds_prop_b": int(bool(pb["holds_prop_b"])),
                "violates_transfer_only": int(bool(pb["violates_transfer_only"])),
                "y_mean_lnP": prep["y_mean_lnP"],
                "r_phage_emp_d": prep["r_phage_emp_d"],
                "r_phage_emp_tw_audit_d": prep["r_phage_emp_tw_audit_d"],
                "y_early_d": prep["y_early_d"],
                "mean_res_frac": mean_q,
                "res_loss_lnR0": res_loss,
                "D_sim_susceptible": d_sim_hs,
                "d_threshold_susceptible": d_thresh_hs,
                "r_const_susceptible_d": r_const_hs,
                "extra": rec.extra,
            }
            unit_rows.append(urow)
            inf_by_ds[ds_id].append(urow)

    # -------------------------------------------------------------------------
    # 1. Primary Results Summary (T1, T2a, T2b, T3)
    # -------------------------------------------------------------------------
    primary_by_ds: dict[str, dict] = {}
    for ds_id, _ in all_datasets:
        rows = inf_by_ds[ds_id]
        ctrls = ctrl_by_ds[ds_id]
        n_inf = len(rows)

        sig_inf = float(np.mean([r["sigma_lnH"] for r in rows]))
        if ctrls:
            sig_ctrl_reg = float(np.mean([c["sigma_lnH_incl_first"] for c in ctrls]))
            sig_ctrl_matched = float(np.mean([c["sigma_lnH_excl_first"] for c in ctrls]))
            fb_reg = sig_inf / sig_ctrl_reg
            fb_matched = sig_inf / sig_ctrl_matched
        else:
            sig_ctrl_reg = math.nan
            sig_ctrl_matched = math.nan
            fb_reg = math.nan
            fb_matched = math.nan

        ym = np.array([r["y_mean_lnP"] for r in rows], dtype=float)
        re = np.array([r["r_phage_emp_d"] for r in rows], dtype=float)
        re_tw = np.array([r["r_phage_emp_tw_audit_d"] for r in rows], dtype=float)
        ye = np.array([r["y_early_d"] for r in rows], dtype=float)
        ds_c = np.array([r["D_sim_contact"] for r in rows], dtype=float)
        ds_w = np.array([r["D_sim_waste"] for r in rows], dtype=float)
        gc = np.array([r["Gamma_c_2nd"] for r in rows], dtype=float)
        gc_pw = np.array([r["Gamma_c_pw_audit"] for r in rows], dtype=float)
        rc = np.array([r["r_const_d"] for r in rows], dtype=float)

        t1_ym = cm.freedman_lane_partial_spearman(ym, ds_c, rc, seed=20261114, n_perm=10000)
        t1_ym["verdict"] = cm.classify_t1_outcome(t1_ym, fb_reg)
        # Independent 100,000-permutation check with fresh seed
        t1_ym_100k = cm.freedman_lane_partial_spearman(ym, ds_c, rc, seed=99112026, n_perm=100000)
        t1_ym_100k["verdict"] = cm.classify_t1_outcome(t1_ym_100k, fb_reg)

        t1_re = cm.freedman_lane_partial_spearman(re, ds_c, rc, seed=20261115, n_perm=10000)
        t1_re["verdict"] = cm.classify_t1_outcome(t1_re, fb_reg)
        t1_re_tw = cm.freedman_lane_partial_spearman(re_tw, ds_c, rc, seed=20261115, n_perm=10000)
        t1_re_tw["verdict"] = cm.classify_t1_outcome(t1_re_tw, fb_reg)

        t1_ye = cm.freedman_lane_partial_spearman(ye, ds_c, rc, seed=20261116, n_perm=10000)
        t1_ye["verdict"] = cm.classify_t1_outcome(t1_ye, fb_reg)
        t1_ye_100k = cm.freedman_lane_partial_spearman(ye, ds_c, rc, seed=99112027, n_perm=100000)
        t1_ye_100k["verdict"] = cm.classify_t1_outcome(t1_ye_100k, fb_reg)

        wait_ratios = [float(dc / dw) for dc, dw in zip(ds_c, ds_w) if abs(dw) > 1e-6]
        rel_err_reg = [abs(g - d) / max(abs(d), 1e-6) for g, d in zip(gc, ds_c) if abs(d) > 1e-5]
        rel_err_pw = [abs(g - d) / max(abs(d), 1e-6) for g, d in zip(gc_pw, ds_c) if abs(d) > 1e-5]

        primary_by_ds[ds_id] = {
            "n_infected": n_inf,
            "n_controls": len(ctrls),
            "sigma_lnH_infected_mean": sig_inf,
            "sigma_lnH_control_mean_reg": sig_ctrl_reg,
            "sigma_lnH_control_mean_matched_excl_first": sig_ctrl_matched,
            "feedback_ratio_reg": fb_reg,
            "feedback_ratio_matched_excl_first": fb_matched,
            "phi_lag1_median": float(np.median([r["phi_lag1"] for r in rows])),
            "phi_lag1_min_max": [float(np.min([r["phi_lag1"] for r in rows])), float(np.max([r["phi_lag1"] for r in rows]))],
            "d_threshold_mean": float(np.mean([r["d_threshold"] for r in rows])),
            "ln_R0_mean": float(np.mean([r["ln_R0_mean"] for r in rows])),
            "D_sim_contact_mean": float(np.mean(ds_c)),
            "D_sim_contact_range": [float(np.min(ds_c)), float(np.max(ds_c))],
            "Gamma_c_2nd_mean": float(np.mean(gc)),
            "Gamma_c_pw_audit_mean": float(np.mean(gc_pw)),
            "D_sim_waste_mean": float(np.mean(ds_w)),
            "waiting_ratio_median": float(np.median(wait_ratios)),
            "Gamma_c_median_rel_error_reg": float(np.median(rel_err_reg)),
            "Gamma_c_median_rel_error_pw_audit": float(np.median(rel_err_pw)),
            "PropB_holds_count": int(sum(r["holds_prop_b"] for r in rows)),
            "transfer_only_violations": int(sum(r["violates_transfer_only"] for r in rows)),
            "extinctions": int(sum(r["went_extinct"] for r in rows)),
            "r_const_mean_d": float(np.mean(rc)),
            "r_obs_mean_d": float(np.mean([r["r_obs_d"] for r in rows])),
            "r_phage_emp_mean_reg_d": float(np.mean(re)),
            "r_phage_emp_mean_tw_audit_d": float(np.mean(re_tw)),
            "res_loss_lnR0_mean": float(np.mean([r["res_loss_lnR0"] for r in rows])) if any(r["res_loss_lnR0"] > 0 for r in rows) else math.nan,
            "T1_primary_y_mean": t1_ym,
            "T1_primary_y_mean_100k_independent": t1_ym_100k,
            "T1_secondary_r_emp": t1_re,
            "T1_secondary_r_emp_tw_audit": t1_re_tw,
            "T1_early_invasion": t1_ye,
            "T1_early_invasion_100k_independent": t1_ye_100k,
        }

    # D2 pulse operator means
    d2_pulse_summary = {}
    for reg_lbl in ("3h_uniform", "30min_pulsed"):
        sub_reg = [x for x in d2_pulse_reg if x["treatment"] == reg_lbl]
        sub_aud = [x for x in d2_pulse_aligned_audit if x["treatment"] == reg_lbl]
        d2_pulse_summary[reg_lbl] = {
            "registered": {
                "r_pulse_const_d": float(np.mean([x["r_pulse_const_d"] for x in sub_reg])),
                "r_pulse_obs_d": float(np.mean([x["r_pulse_obs_d"] for x in sub_reg])),
                "D_pulse_total": float(np.mean([x["D_pulse_total"] for x in sub_reg])),
                "D_pulse_from_dilution_var": float(np.mean([x["D_pulse_from_dilution_var"] for x in sub_reg])),
                "D_pulse_from_host_var": float(np.mean([x["D_pulse_from_host_var"] for x in sub_reg])),
            },
            "start_aligned_audit": {
                "r_pulse_const_d": float(np.mean([x["r_pulse_const_d"] for x in sub_aud])),
                "r_pulse_obs_d": float(np.mean([x["r_pulse_obs_d"] for x in sub_aud])),
                "D_pulse_total": float(np.mean([x["D_pulse_total"] for x in sub_aud])),
                "D_pulse_from_dilution_var": float(np.mean([x["D_pulse_from_dilution_var"] for x in sub_aud])),
                "D_pulse_from_host_var": float(np.mean([x["D_pulse_from_host_var"] for x in sub_aud])),
            },
        }

    # -------------------------------------------------------------------------
    # 2. Weather & Environment Amendment (`Amendment 3`) on D4 (n = 30)
    # -------------------------------------------------------------------------
    d4_rows = inf_by_ds["D4_duncan2017"]
    weather_groups = {}
    for gname in ("constant_28C", "constant_32C", "switch_2d", "switch_4d", "switch_8d"):
        sub = [r for r in d4_rows if r["extra"]["weather_group"] == gname]
        sigs = [r["sigma_lnH"] for r in sub]
        yms = [r["y_mean_lnP"] for r in sub]
        weather_groups[gname] = {
            "n": len(sub),
            "sigma_lnH_mean": float(np.mean(sigs)),
            "sigma_lnH_range": [float(min(sigs)), float(max(sigs))],
            "y_mean_lnP_mean": float(np.mean(yms)),
            "y_mean_lnP_range": [float(min(yms)), float(max(yms))],
            "final_zero_count": int(sum(r["went_extinct"] for r in sub)),
        }

    t_mean_vec = np.array([r["extra"]["assigned_temp_mean_C"] for r in d4_rows], dtype=float)
    t_sd_vec = np.array([r["extra"]["assigned_temp_sd_C"] for r in d4_rows], dtype=float)
    d4_sw = [r for r in d4_rows if r["extra"]["freq_d"] > 0]
    sw_int_vec = np.array([r["extra"]["freq_d"] for r in d4_sw], dtype=float)

    weather_corrs = {
        "mean_T_vs_sigma_lnH": cm.spearman_corr(t_mean_vec, np.array([r["sigma_lnH"] for r in d4_rows])),
        "mean_T_vs_y_mean_lnP": cm.spearman_corr(t_mean_vec, np.array([r["y_mean_lnP"] for r in d4_rows])),
        "mean_T_vs_went_extinct": cm.spearman_corr(t_mean_vec, np.array([r["went_extinct"] for r in d4_rows])),
        "sd_T_vs_sigma_lnH": cm.spearman_corr(t_sd_vec, np.array([r["sigma_lnH"] for r in d4_rows])),
        "sd_T_vs_y_mean_lnP": cm.spearman_corr(t_sd_vec, np.array([r["y_mean_lnP"] for r in d4_rows])),
        "sd_T_vs_went_extinct": cm.spearman_corr(t_sd_vec, np.array([r["went_extinct"] for r in d4_rows])),
        "switch_interval_vs_sigma_lnH_n18": cm.spearman_corr(sw_int_vec, np.array([r["sigma_lnH"] for r in d4_sw])),
        "switch_interval_vs_y_mean_lnP_n18": cm.spearman_corr(sw_int_vec, np.array([r["y_mean_lnP"] for r in d4_sw])),
        "switch_interval_vs_went_extinct_n18": cm.spearman_corr(sw_int_vec, np.array([r["went_extinct"] for r in d4_sw])),
    }

    # Additional audit: what unexecuted Amendment 2's M1 partial Spearman controlling for (r_const, mean_T, sd_T) gives on full n=30!
    z_m1 = np.column_stack([
        np.array([r["r_const_d"] for r in d4_rows], dtype=float),
        t_mean_vec,
        t_sd_vec,
    ])
    d4_m1_audit = cm.freedman_lane_partial_spearman(
        np.array([r["y_mean_lnP"] for r in d4_rows], dtype=float),
        np.array([r["D_sim_contact"] for r in d4_rows], dtype=float),
        z_m1,
        seed=20261114,
        n_perm=10000,
    )
    d4_m1_audit["verdict"] = cm.classify_t1_outcome(d4_m1_audit, primary_by_ds["D4_duncan2017"]["feedback_ratio_reg"])

    # -------------------------------------------------------------------------
    # 3. Two Most Consequential Sensitivities (`S2` H_S(t) and `S4` D4 thermal)
    # -------------------------------------------------------------------------
    sens_s2: dict[str, dict] = {}
    for ds_id in ("D1_larsen2019", "D4_duncan2017", "D5_zhao2016"):
        rows = inf_by_ds[ds_id]
        d_hs = np.array([r["D_sim_susceptible"] for r in rows], dtype=float)
        rc_hs = np.array([r["r_const_susceptible_d"] for r in rows], dtype=float)
        ym = np.array([r["y_mean_lnP"] for r in rows], dtype=float)
        dth_hs = np.array([r["d_threshold_susceptible"] for r in rows], dtype=float)
        st = cm.freedman_lane_partial_spearman(ym, d_hs, rc_hs, seed=20261114, n_perm=10000)
        st["verdict"] = cm.classify_t1_outcome(st, primary_by_ds[ds_id]["feedback_ratio_reg"])
        st_100k = cm.freedman_lane_partial_spearman(ym, d_hs, rc_hs, seed=99112028, n_perm=100000)
        st_100k["verdict"] = cm.classify_t1_outcome(st_100k, primary_by_ds[ds_id]["feedback_ratio_reg"])
        sens_s2[ds_id] = {
            "mean_d_threshold_HS": float(np.mean(dth_hs)),
            "mean_D_sim_HS": float(np.mean(d_hs)),
            "T1_HS": st,
            "T1_HS_100k_independent": st_100k,
        }
    d1_abort = float(np.mean([-math.log(1.0 - min(max(r["mean_res_frac"], 0.0), 0.995)) for r in inf_by_ds["D1_larsen2019"]]))
    sens_s2["D1_abortive_sink_lnR0_loss_mean"] = d1_abort

    # Sensitivity B: D4 thermal confounding (`S4`)
    d4_no32 = [r for r in d4_rows if r["treatment"] != "freq0_T32"]
    st_sw18 = cm.freedman_lane_partial_spearman(
        np.array([r["y_mean_lnP"] for r in d4_sw]),
        np.array([r["D_sim_contact"] for r in d4_sw]),
        np.array([r["r_const_d"] for r in d4_sw]),
        seed=20261114,
        n_perm=10000,
    )
    st_sw18["verdict"] = cm.classify_t1_outcome(st_sw18, 1.18)

    st_no32 = cm.freedman_lane_partial_spearman(
        np.array([r["y_mean_lnP"] for r in d4_no32]),
        np.array([r["D_sim_contact"] for r in d4_no32]),
        np.array([r["r_const_d"] for r in d4_no32]),
        seed=20261114,
        n_perm=10000,
    )
    st_no32["verdict"] = cm.classify_t1_outcome(st_no32, 1.18)

    rc_therm = np.array([
        (1.0 if r["treatment"] == "freq0_T28" else (0.0 if r["treatment"] == "freq0_T32" else 0.5)) * r["r_const_d"]
        - (0.0 if r["treatment"] == "freq0_T28" else (1.0 if r["treatment"] == "freq0_T32" else 0.5)) * 2.45
        for r in d4_rows
    ], dtype=float)
    st_therm30 = cm.freedman_lane_partial_spearman(
        np.array([r["y_mean_lnP"] for r in d4_rows]),
        np.array([r["D_sim_contact"] for r in d4_rows]),
        rc_therm,
        seed=20261114,
        n_perm=10000,
    )
    st_therm30["verdict"] = cm.classify_t1_outcome(st_therm30, 1.18)

    sens_s4_d4 = {
        "matched_mean_30C_fluct_only_n18": st_sw18,
        "excluding_constant_32C_n24": st_no32,
        "thermal_adjusted_r_const_n30": st_therm30,
        "amendment2_M1_multicov_temp_adjusted_audit_n30": d4_m1_audit,
    }

    # -------------------------------------------------------------------------
    # 4. Unit-by-Unit Numerical Comparison Against Saved `unit_metrics.csv`
    # -------------------------------------------------------------------------
    with (ORIG_RESULTS / "unit_metrics.csv").open("r", encoding="utf-8") as fh:
        orig_units = {(r["dataset_id"], r["unit_id"]): r for r in csv.DictReader(fh)}

    compare_cols = [
        "H_bar", "sigma_lnH", "phi_lag1", "d_threshold", "ln_R0_mean",
        "r_const_d", "r_obs_d", "D_sim_contact", "D_sim_waste",
        "Gamma_c_2nd", "D_disc_propB", "bound_prop_b", "y_mean_lnP",
        "r_phage_emp_d", "y_early_d",
    ]
    max_col_diffs: dict[str, dict[str, float]] = {}
    for col in compare_cols:
        max_abs = 0.0
        max_rel = 0.0
        for r in unit_rows:
            o_val = float(orig_units[(r["dataset_id"], r["unit_id"])][col])
            c_val = float(r[col])
            ad = abs(c_val - o_val)
            rd = ad / max(abs(o_val), 1e-12)
            max_abs = max(max_abs, ad)
            max_rel = max(max_rel, rd)
        max_col_diffs[col] = {"max_abs_diff": max_abs, "max_rel_diff": max_rel}

    out = {
        "replication_utc_from_date": utc_stamp,
        "replication_local_date": local_stamp,
        "total_infected_units": len(unit_rows),
        "global_T2a": {
            "total_infected_units": len(unit_rows),
            "prop_b_holds": int(sum(r["holds_prop_b"] for r in unit_rows)),
            "transfer_only_violations": int(sum(r["violates_transfer_only"] for r in unit_rows)),
            "verdict": "supports" if sum(r["holds_prop_b"] for r in unit_rows) == len(unit_rows) else "contradicts",
        },
        "auxiliary_deposit_checks": {
            "D2": d2_checks,
            "D3": d3_checks,
        },
        "primary_by_dataset": primary_by_ds,
        "d2_pulse_chloroform_summary": d2_pulse_summary,
        "weather_amendment3_d4": {
            "field_record_count": 0,
            "groups": weather_groups,
            "spearman_correlations": weather_corrs,
        },
        "selected_sensitivities": {
            "S2_susceptible_host_HS": sens_s2,
            "S4_D4_thermal_confounding": sens_s4_d4,
        },
        "unit_level_104_agreement": max_col_diffs,
    }

    out_path = HERE / "replication_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
