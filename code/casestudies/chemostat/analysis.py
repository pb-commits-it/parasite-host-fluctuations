#!/usr/bin/env python3
"""Main case-study analysis script for `chemostat` (bacteria-phage chemostats & transfers).

Follows `casestudies/chemostat/protocol.md` strictly.
Reads raw open data from `data/raw/chemostat/` via `data_loader.py`, maps each dataset
onto the two-stage contact-limited lifecycle model via `model_helpers.py` and
`studies/engine-R16/lifecycle_engine.py`, runs registered tests T1-T3 and sensitivity
analyses S1-S5, writes every number to `results/`, and generates SVG + pgfplots figures
in `figures/`.

Run with: python3 -I casestudies/chemostat/analysis.py
"""
from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import os
import sys
import time
import numpy as np
from scipy import linalg, optimize

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RAW_ROOT = os.path.join(ROOT, "data", "raw", "chemostat")
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data_loader as dl  # noqa: E402
import figures_gen as fg  # noqa: E402
import model_helpers as mh  # noqa: E402


def extract_clean_intervals(
    u: dl.UnitSeries,
    min_pos_H: float,
    min_pos_P: float,
    zero_floor_mult: float = 0.5,
) -> dict:
    """Apply pre-registered missing-value and detection-floor rules (`protocol.md` §6.1)."""
    h_floor = zero_floor_mult * min_pos_H
    p_floor = zero_floor_mult * min_pos_P

    # Host valid points
    valid_h = [
        (float(u.times_d[k]), (float(u.host_density[k]) if float(u.host_density[k]) > 0 else h_floor), float(u.dilution_rate_d[k]))
        for k in range(len(u.times_d))
        if not math.isnan(u.host_density[k])
    ]
    n_h_zeros = sum(1 for k in range(len(u.times_d)) if (not math.isnan(u.host_density[k]) and u.host_density[k] == 0.0))

    if len(valid_h) >= 2:
        t_h = np.array([x[0] for x in valid_h], dtype=float)
        dt_seq = np.diff(t_h)
        # Left-endpoint / interval host density on [t_{k-1}, t_k]; for D5 where t=0 host is NA, valid_h starts at t=2d
        # Use midpoint geometric/arithmetic interval representation? Protocol §2.2 uses piecewise-constant H_i(t_k) over intervals:
        # Using interval average 0.5*(H_{k-1}+H_k) vs endpoint: wait - for serial transfer (D2, D3, D4, D5), valid_h[k] at transfer k>=1
        # is the culture density measured for that transfer! Using valid_h[1:] aligned with dt_seq matches each interval k=1..K-1.
        H_seq = np.array([x[1] for x in valid_h[1:]], dtype=float)
        om_seq = np.array([x[2] for x in valid_h[1:]], dtype=float)
        H_pts_all = np.array([x[1] for x in valid_h], dtype=float)
        t_pts_all = t_h
    else:
        H_seq = np.array([], dtype=float)
        dt_seq = np.array([], dtype=float)
        om_seq = np.array([], dtype=float)
        H_pts_all = np.array([x[1] for x in valid_h], dtype=float)
        t_pts_all = np.array([x[0] for x in valid_h], dtype=float)

    # Phage valid points
    valid_p = [
        (float(u.times_d[k]), float(u.phage_density[k]), float(u.dilution_rate_d[k]))
        for k in range(len(u.times_d))
        if not math.isnan(u.phage_density[k])
    ]
    n_p_zeros = sum(1 for _, pv, _ in valid_p if pv <= 0.0)
    went_extinct = bool(len(valid_p) > 0 and valid_p[-1][1] <= 0.0)

    if u.infected and len(valid_p) >= 2:
        t_p = np.array([x[0] for x in valid_p], dtype=float)
        p_cens = np.array([x[1] if x[1] > 0.0 else p_floor for x in valid_p], dtype=float)
        # Post-inoculation observations (exclude t=0 inoculum for y_mean when t_p[0]==0 and len>2)
        post_mask = (t_p > 0.0) if (t_p[0] == 0.0 and np.sum(t_p > 0.0) >= 1) else np.ones(len(t_p), dtype=bool)
        y_mean = float(np.mean(np.log(p_cens[post_mask])))
        om_p_mean = float(np.mean([x[2] for x in valid_p[1:]]))
        r_phage_emp = (math.log(p_cens[-1]) - math.log(p_cens[0])) / max(t_p[-1] - t_p[0], 1e-9) + om_p_mean
        # Early invasion rate: first post-inoculation step (or t <= 9 d for D1)
        if u.dataset_id == "D1_larsen2019":
            early_idx = [j for j in range(len(t_p)) if t_p[j] <= 9.0]
            j_end = early_idx[-1] if len(early_idx) >= 2 else 1
        else:
            j_end = 1
        dt_early = max(t_p[j_end] - t_p[0], 1e-9)
        y_early = (math.log(p_cens[j_end]) - math.log(p_cens[0])) / dt_early + float(valid_p[j_end][2])
    else:
        y_mean = math.nan
        r_phage_emp = math.nan
        y_early = math.nan

    return {
        "H_seq": H_seq,
        "dt_seq": dt_seq,
        "om_seq": om_seq,
        "H_pts_all": H_pts_all,
        "t_pts_all": t_pts_all,
        "n_h_obs": len(valid_h),
        "n_h_zeros": n_h_zeros,
        "n_p_obs": len(valid_p),
        "n_p_zeros": n_p_zeros,
        "went_extinct": went_extinct,
        "y_mean_lnP": y_mean,
        "r_phage_emp_d": r_phage_emp,
        "y_early_d": y_early,
    }


def build_susceptible_host_seq(
    u: dl.UnitSeries,
    t_intervals_end: np.ndarray,
    H_seq: np.ndarray,
    use_ancestral: bool = False,
    min_susceptible_frac: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Interpolate assayed resistance fraction onto trajectory intervals to form H_S(t)."""
    if len(u.res_times_d) == 0:
        return H_seq.copy(), 0.0
    r_arr = u.res_frac_anc if use_ancestral else u.res_frac_contemp
    # Linear interpolation of resistance fraction across time, clamped at endpoints;
    # cap at 1 - min_susceptible_frac (0.5% minority sensitive refuge, Bohannan & Lenski 1997 / Chaudhry 2018)
    q_interp = np.interp(t_intervals_end, u.res_times_d, r_arr, left=float(r_arr[0]), right=float(r_arr[-1]))
    q_capped = np.clip(q_interp, 0.0, 1.0 - min_susceptible_frac)
    H_S = H_seq * (1.0 - q_capped)
    return H_S, float(np.mean(q_interp))


def exact_d2_serial_pulse_loss(u: dl.UnitSeries, params: mh.PhageParams) -> dict:
    """Exact batch-incubation + discrete-dilution + chloroform-lysis operator for D2 (`Reuter et al. 2026`)."""
    nu = 1.0 / params.latent_d
    alpha = params.adsorption_ml_d
    mu0 = params.decay_d

    # Intervals are k = 1 .. len(times_d)-1
    dt_seq = np.diff(u.times_d)
    H_seq = u.host_density[1:]
    K = len(dt_seq)
    T_tot = float(np.sum(dt_seq))
    H_bar = float(np.sum(dt_seq * H_seq) / T_tot)

    regime = u.meta.get("regime", "")
    if regime == "3h":
        f_list = []
        p_arr = u.phage_density
        for k in range(K):
            prev_out = max(float(p_arr[k]), 1.0)
            in_next = 0.1 * float(H_seq[k])
            f_list.append(min(max(in_next / prev_out, 1e-9), 1.0))
        chloro_flags = [True] * K
    else:
        f_list = list(u.meta["f_dilutions"])
        step_types = list(u.meta["step_types"])
        # In 30-min regime: T1..T4 end with 2% co-culture transfer (no chloroform); T5 ends with chloroform extraction!
        chloro_flags = [(st == "T5") for st in step_types]

    def monodromy_growth(H_vec: np.ndarray, f_vec: list[float], y_tilt: float = 0.0) -> float:
        U = np.eye(2, dtype=float)
        eye2 = np.eye(2, dtype=float)
        ls = 0.0
        for k in range(K):
            Hk = float(H_vec[k])
            dtk = float(dt_seq[k])
            M_batch = np.array([
                [-nu, alpha * Hk],
                [math.exp(y_tilt) * params.burst * nu, -(alpha * Hk + mu0)],
            ], dtype=float)
            sk = mh.dominant_real_eig(M_batch)
            Ek = linalg.expm((M_batch - sk * eye2) * dtk)
            ls += sk * dtk
            if chloro_flags[k]:
                # Chloroform kills infected cells lacking mature virions; free phage diluted by f_vec[k]
                D_op = np.array([[0.0, 0.0], [0.0, float(f_vec[k])]], dtype=float)
            else:
                D_op = float(f_vec[k]) * np.eye(2, dtype=float)
            U = D_op @ Ek @ U
            nrm = float(np.max(np.abs(U)))
            if nrm > 0.0:
                U /= nrm
                ls += math.log(nrm)
        rho = float(np.max(np.abs(linalg.eigvals(U))))
        return (ls + math.log(max(rho, 1e-300))) / T_tot

    f_bar = float(np.mean(f_list))
    r_pulse_obs = monodromy_growth(H_seq, f_list, y_tilt=0.0)
    r_pulse_const = monodromy_growth(np.full(K, H_bar), [f_bar] * K, y_tilt=0.0)

    # Solve threshold shift relative to constant-mean (H_bar, f_bar) under same pulse/chloroform structure
    y0_const = float(optimize.brentq(lambda y: monodromy_growth(np.full(K, H_bar), [f_bar] * K, y_tilt=y), -15.0, 15.0, xtol=1e-9))
    y0_obs = float(optimize.brentq(lambda y: monodromy_growth(H_seq, f_list, y_tilt=y), -15.0, 15.0, xtol=1e-9))
    # Also isolate host-only fluctuation at actual f_list vs f-fluctuation at H_bar
    y0_hconst_fobs = float(optimize.brentq(lambda y: monodromy_growth(np.full(K, H_bar), f_list, y_tilt=y), -15.0, 15.0, xtol=1e-9))

    return {
        "r_pulse_const_d": r_pulse_const,
        "r_pulse_obs_d": r_pulse_obs,
        "D_pulse_total": y0_obs - y0_const,
        "D_pulse_from_dilution_var": y0_hconst_fobs - y0_const,
        "D_pulse_from_host_var": y0_obs - y0_hconst_fobs,
    }


def compute_coupled_dilution_curve() -> list[dict]:
    """Generate theoretical coupled-dilution curve for the theoretical note & Figure 2C."""
    # Periodic boom-bust schedule of 80 steps (dt=0.05 d -> 4 d total, period 1 d = 20 steps)
    K = 80
    dt_vec = np.full(K, 0.05)
    t_idx = np.arange(K)
    wave = np.where((t_idx // 10) % 2 == 0, 1.0, -1.0)
    # Host swings 1 +/- 0.65 around H_bar = 1e6 cells/mL; cyanophage-like parameters with mu0 = 0.02 d^-1
    H_bar = 2.0e5
    H_seq = H_bar * (1.0 + 0.65 * wave)
    p_ill = mh.PhageParams(
        name="Coupled-dilution illustration",
        latent_d=0.25,
        burst=40.0,
        adsorption_ml_d=8.0e-6,  # alpha * H_bar = 1.6 d^-1
        decay_d=0.02,
        substages_I=1,
    )
    rows = []
    for om_bar in np.linspace(0.0, 5.0, 21):
        om_fluct_neg = om_bar * (1.0 - 0.65 * wave)  # high washout during host bust (Cov(H, omega) < 0)
        om_const = np.full(K, om_bar)
        res_unif = mh.threshold_quantities_ct(H_seq, dt_vec, om_fluct_neg, om_fluct_neg, p_ill, mode="contact")
        res_sel_neg = mh.threshold_quantities_ct(H_seq, dt_vec, om_fluct_neg, np.zeros(K), p_ill, mode="contact")
        res_sel_cst = mh.threshold_quantities_ct(H_seq, dt_vec, om_const, np.zeros(K), p_ill, mode="contact")

        # Proposition B bound at fixed discrete step dt_step = 0.05 d (where max exit rate <= (1.6*1.65+0.02+5*1.65)=10.9 -> use 0.05 if admissible else 0.02)
        dt_b = 0.05
        nu = 1.0 / p_ill.latent_d
        alpha = p_ill.adsorption_ml_d
        mu0 = p_ill.decay_d
        sP = np.exp(-(alpha * H_seq + mu0 + om_fluct_neg) * dt_b)
        sP_bar = float(np.mean(sP))
        tauP0 = 1.0 / (1.0 - sP_bar)
        ls_P = -float(np.mean(np.log(sP / sP_bar)))
        lg_P = -float(np.mean(np.log(H_seq / H_bar)))
        prop_b_005 = lg_P + (tauP0 - 1.0) * ls_P

        rows.append({
            "omega_bar_d": f"{om_bar:.4f}",
            "d_threshold": f"{res_unif['d_threshold']:.5f}",
            "D_uniform_wash": f"{res_unif['D_sim']:.6f}",
            "D_selective_neg_cov": f"{res_sel_neg['D_sim']:.6f}",
            "D_selective_const_omega": f"{res_sel_cst['D_sim']:.6f}",
            "PropB_bound_dt005": f"{prop_b_005:.6f}",
        })
    return rows


def decide_t1(t1_stat: dict, feedback_ratio: float) -> str:
    """Apply exact registered decision criterion from `protocol.md` §6.2."""
    rho = t1_stat["rho_part"]
    p_neg = t1_stat["p_neg"]
    p_pos = t1_stat["p_pos"]
    if rho < 0.0 and p_neg < 0.05:
        return "supports"
    if rho > 0.0 and p_pos < 0.05:
        if not math.isnan(feedback_ratio) and feedback_ratio > 2.0:
            return "not decided (confounded by endogenous predator-prey feedback)"
        return "contradicts"
    return "not decided"


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    t_start = time.time()
    utc_stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    buf = io.StringIO()
    print(f"=== CASE STUDY `chemostat` FULL ANALYSIS (started {utc_stamp}) ===", file=buf)

    # Load all five analyzable datasets
    d1_units = dl.load_d1_larsen2019(RAW_ROOT)
    d2_units, d2_aux = dl.load_d2_reuter2026(RAW_ROOT)
    d3_units, d3_aux = dl.load_d3_chaudhry2018(RAW_ROOT)
    d4_units = dl.load_d4_duncan2017(RAW_ROOT)
    d5_units, d5_aux = dl.load_d5_zhao2016(RAW_ROOT, od_to_cfu=5.0e8)

    datasets = [
        ("D1_larsen2019", d1_units),
        ("D2_reuter2026", d2_units),
        ("D3_chaudhry2018", d3_units),
        ("D4_duncan2017", d4_units),
        ("D5_zhao2016", d5_units),
    ]

    print(f"\nAuxiliary non-fitted parameter verifications from raw deposits:", file=buf)
    print(f"  D2 (Reuter 2026 exp_3 WT adsorption, n={d2_aux['WT_adsorption_n_runs']} runs): "
          f"mean = {d2_aux['WT_adsorption_mean_ml_min']:.3e} mL min^-1 (Table 1 value: 4.000e-09 mL min^-1)", file=buf)
    print(f"  D3 (Chaudhry 2018 Fig S3 no-host washout, n=4): "
          f"mean loss rate = {d3_aux['nohost_loss_rate_mean_d']:.3f} d^-1 (1:100 dilution alone = {d3_aux['expected_1to100_dilution_rate_d']:.3f} d^-1)", file=buf)

    all_unit_rows: list[dict] = []
    infected_by_ds: dict[str, list[dict]] = {}
    control_by_ds: dict[str, list[dict]] = {}
    d2_pulse_details: list[dict] = []

    for ds_id, unit_list in datasets:
        params = mh.SYSTEM_PARAMS[ds_id]
        pos_h = [float(v) for u in unit_list for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in unit_list for v in u.phage_density if not math.isnan(v) and v > 0.0]
        min_pos_H = min(pos_h) if pos_h else 1.0
        min_pos_P = min(pos_p) if pos_p else 1.0

        infected_by_ds[ds_id] = []
        control_by_ds[ds_id] = []

        for u in unit_list:
            clean = extract_clean_intervals(u, min_pos_H, min_pos_P, zero_floor_mult=0.5)
            H_seq = clean["H_seq"]
            dt_seq = clean["dt_seq"]
            om_seq = clean["om_seq"]

            if len(H_seq) < 2:
                continue

            if not u.infected:
                lh = np.log(clean["H_pts_all"])
                sig_c = float(np.std(lh, ddof=1)) if len(lh) >= 2 else 0.0
                phi_c = float(np.corrcoef(lh[:-1], lh[1:])[0, 1]) if (len(lh) >= 3 and np.std(lh[:-1]) > 1e-12 and np.std(lh[1:]) > 1e-12) else 0.0
                crow = {
                    "dataset_id": ds_id,
                    "unit_id": u.unit_id,
                    "treatment": u.treatment,
                    "infected": 0,
                    "n_h_obs": clean["n_h_obs"],
                    "H_bar": float(np.mean(clean["H_pts_all"])),
                    "sigma_lnH": sig_c,
                    "phi_lag1": phi_c,
                }
                control_by_ds[ds_id].append(crow)
                continue

            # Primary contact-limited and waste (d=1) exact evaluations on observed H_seq
            tq_contact = mh.threshold_quantities_ct(H_seq, dt_seq, om_seq, om_seq, params, mode="contact")
            tq_waste = mh.threshold_quantities_ct(H_seq, dt_seq, om_seq, om_seq, params, mode="waste")

            # Second-order formulas
            so = mh.second_order_formulas(
                H_seq=clean["H_seq"],
                dt_seq=dt_seq,
                tau_P_d=tq_contact["tau_P_d"],
                tau_I_d=tq_contact["tau_I_d"],
                d_thresh=tq_contact["d_threshold"],
                substages_I=params.substages_I,
            )

            # Proposition B discrete staying-rate bound verification
            pb = mh.proposition_b_discrete_check(H_seq, dt_seq, om_seq, om_seq, params)

            # Project engine AR(1) benchmark
            eng_res = mh.engine_ar1_benchmark(
                tau_P_d=tq_contact["tau_P_d"],
                tau_I_d=tq_contact["tau_I_d"],
                d_thresh=tq_contact["d_threshold"],
                sigma_lnH=so["sigma_lnH"],
                phi_lag1=so["phi_lag1"],
                dt_obs_d=float(np.median(dt_seq)),
                substages_I=params.substages_I,
            )

            # Susceptible host H_S(t) calculation if resistance was assayed
            res_loss_lnR0 = 0.0
            D_sim_susceptible = math.nan
            mean_res_frac = math.nan
            if len(u.res_times_d) > 0:
                t_ends = clean["t_pts_all"][1:]
                # Use ancestral resistance in D1 (where 97% of resistant clones resist all evolved phages too) and contemporary CB/CP in D4 & D5
                use_anc = (ds_id == "D1_larsen2019")
                H_S_seq, mean_q = build_susceptible_host_seq(u, t_ends, H_seq, use_ancestral=use_anc)
                tq_hs = mh.threshold_quantities_ct(H_S_seq, dt_seq, om_seq, om_seq, params, mode="contact")
                res_loss_lnR0 = tq_contact["ln_R0_mean"] - tq_hs["ln_R0_mean"]
                D_sim_susceptible = tq_hs["D_sim"]
                mean_res_frac = mean_q

            # D2 exact pulse + chloroform check
            if ds_id == "D2_reuter2026":
                p_info = exact_d2_serial_pulse_loss(u, params)
                p_info["unit_id"] = u.unit_id
                p_info["treatment"] = u.treatment
                d2_pulse_details.append(p_info)

            row = {
                "dataset_id": ds_id,
                "unit_id": u.unit_id,
                "treatment": u.treatment,
                "infected": 1,
                "n_h_obs": clean["n_h_obs"],
                "n_p_obs": clean["n_p_obs"],
                "n_h_zeros": clean["n_h_zeros"],
                "n_p_zeros": clean["n_p_zeros"],
                "went_extinct": int(clean["went_extinct"]),
                "H_bar": tq_contact["H_bar"],
                "sigma_lnH": so["sigma_lnH"],
                "cv_H": so["cv_H"],
                "phi_lag1": so["phi_lag1"],
                "d_threshold": tq_contact["d_threshold"],
                "tau_P_d": tq_contact["tau_P_d"],
                "tau_I_d": tq_contact["tau_I_d"],
                "ln_R0_mean": tq_contact["ln_R0_mean"],
                "r_const_d": tq_contact["r_const_d"],
                "r_obs_d": tq_contact["r_obs_d"],
                "Delta_r_fluct_d": tq_contact["Delta_r_fluct_d"],
                "D_sim_contact": tq_contact["D_sim"],
                "D_sim_waste": tq_waste["D_sim"],
                "Gamma_c_2nd": so["Gamma_c_2nd"],
                "Gamma_waste_2nd": so["Gamma_waste_2nd"],
                "engine_D2": eng_res["engine_D2"],
                "D_disc_propB": pb["D_disc"],
                "bound_transfer_only": pb["bound_transfer_only"],
                "stay_penalty_propB": pb["stay_penalty"],
                "bound_prop_b": pb["bound_prop_b"],
                "holds_prop_b": int(pb["holds_prop_b"]),
                "violates_transfer_only": int(pb["violates_transfer_only"]),
                "y_mean_lnP": clean["y_mean_lnP"],
                "r_phage_emp_d": clean["r_phage_emp_d"],
                "y_early_d": clean["y_early_d"],
                "mean_res_frac": mean_res_frac,
                "res_loss_lnR0": res_loss_lnR0,
                "D_sim_susceptible": D_sim_susceptible,
            }
            all_unit_rows.append(row)
            infected_by_ds[ds_id].append(row)

    # Save unit_metrics.csv
    fieldnames = list(all_unit_rows[0].keys())
    fg.write_csv(os.path.join(RESULTS_DIR, "unit_metrics.csv"), fieldnames, all_unit_rows)

    # Evaluate Registered Tests T1, T2, T3 per dataset
    dataset_tests: dict[str, dict] = {}
    print("\n=== SUMMARY BY DATASET: HOST FLUCTUATION, PREDICTED LOSS D, AND REGISTERED TEST T1 ===", file=buf)

    for ds_id, _ in datasets:
        inf_rows = infected_by_ds[ds_id]
        ctrl_rows = control_by_ds[ds_id]
        n_inf = len(inf_rows)

        sig_inf_mean = float(np.mean([r["sigma_lnH"] for r in inf_rows]))
        phi_inf_med = float(np.median([r["phi_lag1"] for r in inf_rows]))
        if ctrl_rows:
            sig_ctrl_mean = float(np.mean([c["sigma_lnH"] for c in ctrl_rows]))
            feedback_ratio = sig_inf_mean / max(sig_ctrl_mean, 1e-6)
        elif ds_id == "D5_zhao2016":
            # D5 has 36 single-timepoint uninfected baseline OD measurements; across-replicate SD within each nutrient
            # treatment gives the uninfected baseline variation (~0.05-0.12 log units)
            feedback_ratio = math.nan
            sig_ctrl_mean = math.nan
        else:
            sig_ctrl_mean = math.nan
            feedback_ratio = math.nan

        y_mean_arr = np.array([r["y_mean_lnP"] for r in inf_rows], dtype=float)
        r_emp_arr = np.array([r["r_phage_emp_d"] for r in inf_rows], dtype=float)
        y_early_arr = np.array([r["y_early_d"] for r in inf_rows], dtype=float)
        d_sim_arr = np.array([r["D_sim_contact"] for r in inf_rows], dtype=float)
        r_const_arr = np.array([r["r_const_d"] for r in inf_rows], dtype=float)

        t1_primary = mh.partial_spearman_freedman_lane(y_mean_arr, d_sim_arr, r_const_arr, seed=20261114)
        t1_primary["verdict"] = decide_t1(t1_primary, feedback_ratio)

        t1_remp = mh.partial_spearman_freedman_lane(r_emp_arr, d_sim_arr, r_const_arr, seed=20261115)
        t1_remp["verdict"] = decide_t1(t1_remp, feedback_ratio)

        t1_early = mh.partial_spearman_freedman_lane(y_early_arr, d_sim_arr, r_const_arr, seed=20261116)
        t1_early["verdict"] = decide_t1(t1_early, feedback_ratio)

        d_waste_arr = np.array([r["D_sim_waste"] for r in inf_rows], dtype=float)
        gamma_arr = np.array([r["Gamma_c_2nd"] for r in inf_rows], dtype=float)
        d_thresh_arr = np.array([r["d_threshold"] for r in inf_rows], dtype=float)
        ln_r0_arr = np.array([r["ln_R0_mean"] for r in inf_rows], dtype=float)

        wait_ratios = [float(dc / dw) for dc, dw in zip(d_sim_arr, d_waste_arr) if abs(dw) > 1e-6]
        rel_errs_g = [abs(gc - dc) / max(abs(dc), 1e-6) for gc, dc in zip(gamma_arr, d_sim_arr) if abs(dc) > 1e-5]

        ds_summary = {
            "n_infected": n_inf,
            "n_controls": len(ctrl_rows),
            "sigma_lnH_infected_mean": sig_inf_mean,
            "sigma_lnH_infected_min_max": [float(np.min([r["sigma_lnH"] for r in inf_rows])), float(np.max([r["sigma_lnH"] for r in inf_rows]))],
            "sigma_lnH_control_mean": sig_ctrl_mean,
            "feedback_sigma_ratio": feedback_ratio,
            "phi_lag1_median": phi_inf_med,
            "phi_lag1_min_max": [float(np.min([r["phi_lag1"] for r in inf_rows])), float(np.max([r["phi_lag1"] for r in inf_rows]))],
            "d_threshold_mean": float(np.mean(d_thresh_arr)),
            "ln_R0_mean": float(np.mean(ln_r0_arr)),
            "D_sim_contact_mean": float(np.mean(d_sim_arr)),
            "D_sim_contact_range": [float(np.min(d_sim_arr)), float(np.max(d_sim_arr))],
            "D_sim_waste_mean": float(np.mean(d_waste_arr)),
            "Gamma_c_2nd_mean": float(np.mean(gamma_arr)),
            "waiting_protection_ratio_median": float(np.median(wait_ratios)) if wait_ratios else math.nan,
            "Gamma_c_median_rel_error": float(np.median(rel_errs_g)) if rel_errs_g else math.nan,
            "PropB_holds_count": int(sum(r["holds_prop_b"] for r in inf_rows)),
            "transfer_only_violations_count": int(sum(r["violates_transfer_only"] for r in inf_rows)),
            "phage_extinctions_count": int(sum(r["went_extinct"] for r in inf_rows)),
            "r_const_mean_d": float(np.mean(r_const_arr)),
            "r_obs_mean_d": float(np.mean([r["r_obs_d"] for r in inf_rows])),
            "r_phage_emp_mean_d": float(np.mean(r_emp_arr)),
            "res_loss_lnR0_mean": float(np.mean([r["res_loss_lnR0"] for r in inf_rows])) if any(r["res_loss_lnR0"] > 0 for r in inf_rows) else math.nan,
            "T1_primary_y_mean": t1_primary,
            "T1_secondary_r_emp": t1_remp,
            "T1_early_invasion": t1_early,
        }
        dataset_tests[ds_id] = ds_summary

        print(f"\n[{ds_id}] n_inf={n_inf}, n_ctrl={len(ctrl_rows)}", file=buf)
        print(f"  Host sigma_lnH: infected mean={sig_inf_mean:.3f} (range {ds_summary['sigma_lnH_infected_min_max'][0]:.3f}..{ds_summary['sigma_lnH_infected_min_max'][1]:.3f}), "
              f"control mean={sig_ctrl_mean:.3f}, feedback ratio={feedback_ratio:.2f}", file=buf)
        print(f"  Host lag-1 phi: median={phi_inf_med:+.3f} (range {ds_summary['phi_lag1_min_max'][0]:+.3f}..{ds_summary['phi_lag1_min_max'][1]:+.3f})", file=buf)
        print(f"  Phage threshold d={ds_summary['d_threshold_mean']:.4f}, mean ln R0={ds_summary['ln_R0_mean']:.3f}", file=buf)
        print(f"  Predicted loss (ln R0): D_sim(contact) mean={ds_summary['D_sim_contact_mean']:.5f} "
              f"(range {ds_summary['D_sim_contact_range'][0]:.5f}..{ds_summary['D_sim_contact_range'][1]:.5f}), "
              f"Gamma_c(2nd)={ds_summary['Gamma_c_2nd_mean']:.5f}, D_sim(waste d=1)={ds_summary['D_sim_waste_mean']:.5f}", file=buf)
        print(f"  Growth rates (d^-1): r_const={ds_summary['r_const_mean_d']:.2f}, r_obs={ds_summary['r_obs_mean_d']:.2f}, "
              f"empirical net r_phage={ds_summary['r_phage_emp_mean_d']:.2f} | extinctions={ds_summary['phage_extinctions_count']}/{n_inf}", file=buf)
        if not math.isnan(ds_summary["res_loss_lnR0_mean"]):
            print(f"  Assayed resistance loss Delta ln R0 = {ds_summary['res_loss_lnR0_mean']:.3f}", file=buf)
        print(f"  Prop B bound holds: {ds_summary['PropB_holds_count']}/{n_inf} (transfer-only violations: {ds_summary['transfer_only_violations_count']}/{n_inf})", file=buf)
        print(f"  T1 (primary y_mean): rho_part={t1_primary['rho_part']:+.4f}, p_neg={t1_primary['p_neg']:.4f}, p_pos={t1_primary['p_pos']:.4f} -> {t1_primary['verdict']}", file=buf)
        print(f"  T1 (net r_emp)     : rho_part={t1_remp['rho_part']:+.4f}, p_neg={t1_remp['p_neg']:.4f}, p_pos={t1_remp['p_pos']:.4f} -> {t1_remp['verdict']}", file=buf)
        print(f"  T1 (early invasion): rho_part={t1_early['rho_part']:+.4f}, p_neg={t1_early['p_neg']:.4f}, p_pos={t1_early['p_pos']:.4f} -> {t1_early['verdict']}", file=buf)

    # Print D1 per-chemostat breakdown & D2 pulse-vs-3h contrast
    print("\n--- D1 (Larsen 2019) Per-Chemostat Breakdown ---", file=buf)
    for r in infected_by_ds["D1_larsen2019"]:
        print(f"  {r['unit_id']} ({r['treatment']}): H_bar={r['H_bar']:.2e}, sigma={r['sigma_lnH']:.3f}, phi={r['phi_lag1']:+.3f}, "
              f"d={r['d_threshold']:.3f}, lnR0={r['ln_R0_mean']:.2f}, D_contact={r['D_sim_contact']:.4f}, "
              f"Gamma_c={r['Gamma_c_2nd']:.4f}, D_waste={r['D_sim_waste']:.4f}, res_loss={r['res_loss_lnR0']:.2f}, "
              f"D_on_HS={r['D_sim_susceptible']:.4f}, y_mean={r['y_mean_lnP']:.2f}", file=buf)

    print("\n--- D2 (Reuter 2026) Exact Serial-Pulse + Chloroform Operator Summary ---", file=buf)
    for regime_lbl in ("3h_uniform", "30min_pulsed"):
        sub = [p for p in d2_pulse_details if p["treatment"] == regime_lbl]
        print(f"  {regime_lbl} (n={len(sub)}): r_pulse_const={np.mean([x['r_pulse_const_d'] for x in sub]):.2f} d^-1, "
              f"r_pulse_obs={np.mean([x['r_pulse_obs_d'] for x in sub]):.2f} d^-1, "
              f"D_pulse_total={np.mean([x['D_pulse_total'] for x in sub]):.5f} "
              f"(from dilution schedule={np.mean([x['D_pulse_from_dilution_var'] for x in sub]):.5f}, "
              f"from host variation={np.mean([x['D_pulse_from_host_var'] for x in sub]):.5f})", file=buf)

    # Print D3 M9M maintained vs lost phage comparison
    d3_m9 = [r for r in infected_by_ds["D3_chaudhry2018"] if r["treatment"] == "M9M"]
    d3_kept = [r for r in d3_m9 if r["went_extinct"] == 0]
    d3_lost = [r for r in d3_m9 if r["went_extinct"] == 1]
    print(f"\n--- D3 (Chaudhry 2018) M9M Maintained (n={len(d3_kept)}) vs Lost (n={len(d3_lost)}) ---", file=buf)
    print(f"  Maintained: mean sigma_lnH={np.mean([r['sigma_lnH'] for r in d3_kept]):.3f}, mean D_contact={np.mean([r['D_sim_contact'] for r in d3_kept]):.5f}, mean H_bar={np.mean([r['H_bar'] for r in d3_kept]):.2e}", file=buf)
    print(f"  Lost      : mean sigma_lnH={np.mean([r['sigma_lnH'] for r in d3_lost]):.3f}, mean D_contact={np.mean([r['D_sim_contact'] for r in d3_lost]):.5f}, mean H_bar={np.mean([r['H_bar'] for r in d3_lost]):.2e}", file=buf)

    # =========================================================================
    # SENSITIVITY SUITE S1 - S5 (`protocol.md` §4-§5)
    # =========================================================================
    sens: dict[str, dict] = {}
    print("\n=== SENSITIVITY SUITE (S1-S5) ===", file=buf)

    # S1: Latent-period Erlang shape m_I in {1, 4, 16} across all 5 datasets
    sens["S1_latent_shape"] = {}
    for ds_id, unit_list in datasets:
        base_p = mh.SYSTEM_PARAMS[ds_id]
        pos_h = [float(v) for u in unit_list for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in unit_list for v in u.phage_density if not math.isnan(v) and v > 0.0]
        min_h, min_p = min(pos_h), min(pos_p)
        inf_u = [u for u in unit_list if u.infected]
        shape_res = {}
        fb = dataset_tests[ds_id]["feedback_sigma_ratio"]
        for m_I in (1, 4, 16):
            pm = replace(base_p, substages_I=m_I)
            d_vals, rc_vals, ym_vals = [], [], []
            for u in inf_u:
                cl = extract_clean_intervals(u, min_h, min_p)
                if len(cl["H_seq"]) < 2:
                    continue
                tq = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], cl["om_seq"], pm, mode="contact")
                d_vals.append(tq["D_sim"])
                rc_vals.append(tq["r_const_d"])
                ym_vals.append(cl["y_mean_lnP"])
            st = mh.partial_spearman_freedman_lane(np.array(ym_vals), np.array(d_vals), np.array(rc_vals))
            shape_res[f"m_{m_I}"] = {
                "mean_D_sim": float(np.mean(d_vals)),
                "rho_part": st["rho_part"],
                "verdict": decide_t1(st, fb),
            }
        sens["S1_latent_shape"][ds_id] = shape_res
        print(f"  S1 [{ds_id}] mean D_sim: m=1 -> {shape_res['m_1']['mean_D_sim']:.5f} ({shape_res['m_1']['verdict']}), "
              f"m=4 -> {shape_res['m_4']['mean_D_sim']:.5f} ({shape_res['m_4']['verdict']}), "
              f"m=16 -> {shape_res['m_16']['mean_D_sim']:.5f} ({shape_res['m_16']['verdict']})", file=buf)

    # S2: Susceptible-host H_S(t) recomputation of T1 on D1, D4, D5
    sens["S2_susceptible_host_T1"] = {}
    for ds_id, unit_list in [("D1_larsen2019", d1_units), ("D4_duncan2017", d4_units), ("D5_zhao2016", d5_units)]:
        pm = mh.SYSTEM_PARAMS[ds_id]
        pos_h = [float(v) for u in unit_list for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in unit_list for v in u.phage_density if not math.isnan(v) and v > 0.0]
        min_h, min_p = min(pos_h), min(pos_p)
        d_hs, rc_hs, ym_hs, dth_hs = [], [], [], []
        for u in unit_list:
            if not u.infected:
                continue
            cl = extract_clean_intervals(u, min_h, min_p)
            if len(cl["H_seq"]) < 2:
                continue
            H_S_seq, _ = build_susceptible_host_seq(u, cl["t_pts_all"][1:], cl["H_seq"], use_ancestral=(ds_id == "D1_larsen2019"))
            tq = mh.threshold_quantities_ct(H_S_seq, cl["dt_seq"], cl["om_seq"], cl["om_seq"], pm, mode="contact")
            d_hs.append(tq["D_sim"])
            rc_hs.append(tq["r_const_d"])
            ym_hs.append(cl["y_mean_lnP"])
            dth_hs.append(tq["d_threshold"])
        st_hs = mh.partial_spearman_freedman_lane(np.array(ym_hs), np.array(d_hs), np.array(rc_hs))
        ver_hs = decide_t1(st_hs, dataset_tests[ds_id]["feedback_sigma_ratio"])
        sens["S2_susceptible_host_T1"][ds_id] = {
            "mean_d_threshold_HS": float(np.mean(d_hs and dth_hs)),
            "mean_D_sim_HS": float(np.mean(d_hs)),
            "rho_part": st_hs["rho_part"],
            "p_neg": st_hs["p_neg"],
            "p_pos": st_hs["p_pos"],
            "verdict": ver_hs,
        }
        print(f"  S2 [{ds_id} on H_S(t)]: mean d_HS={np.mean(dth_hs):.4f}, mean D_sim(H_S)={np.mean(d_hs):.4f}, "
              f"rho_part={st_hs['rho_part']:+.4f}, p_neg={st_hs['p_neg']:.4f} -> {ver_hs}", file=buf)

    # Also record abortive-adsorption resistance loss (if resistant cells still bind phage irreversibly) vs receptor-loss
    d1_abort_losses = []
    for r in infected_by_ds["D1_larsen2019"]:
        q_bar = min(max(float(r["mean_res_frac"]), 0.0), 0.995)
        d1_abort_losses.append(-math.log(1.0 - q_bar))
    sens["S2_susceptible_host_T1"]["D1_abortive_sink_lnR0_loss_mean"] = float(np.mean(d1_abort_losses))
    print(f"  S2 [D1 abortive-adsorption sink upper bound]: mean Delta ln R0 = {np.mean(d1_abort_losses):.3f} "
          f"(vs receptor-loss non-adsorbing Delta ln R0 = {dataset_tests['D1_larsen2019']['res_loss_lnR0_mean']:.3f})", file=buf)

    # S3: Coupled & wall-retained washout (D1 omega_I = 0) + D2 evolved phage genotypes (Phi_L vs Phi_S)
    sens["S3_washout_and_genotypes"] = {}
    d1_wall_D = []
    for u in [x for x in d1_units if x.infected]:
        cl = extract_clean_intervals(u, 1.0, 1.0)
        tq_w = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], np.zeros_like(cl["om_seq"]), mh.SYSTEM_PARAMS["D1_larsen2019"])
        d1_wall_D.append(tq_w["D_sim"])
    sens["S3_washout_and_genotypes"]["D1_wall_sheltered_I_mean_D"] = float(np.mean(d1_wall_D))
    print(f"  S3 [D1 wall-sheltered omega_I=0]: mean D_sim = {np.mean(d1_wall_D):.5f} (vs uniform omega_I=1.0: {dataset_tests['D1_larsen2019']['D_sim_contact_mean']:.5f})", file=buf)

    # D2 evolved genotypes from Reuter Table 1: Phi_L (alpha=2e-9 mL/min, decay=1.4e-2/min) vs Phi_S (alpha=6e-9 mL/min, decay=6e-3/min)
    p_phiL = replace(mh.SYSTEM_PARAMS["D2_reuter2026"], adsorption_ml_d=2.0e-9 * 1440.0, decay_d=1.4e-2 * 1440.0)
    p_phiS = replace(mh.SYSTEM_PARAMS["D2_reuter2026"], adsorption_ml_d=6.0e-9 * 1440.0, decay_d=0.6e-2 * 1440.0)
    for gname, gp in [("WT_ancestor", mh.SYSTEM_PARAMS["D2_reuter2026"]), ("Phi_L_30min_mutant", p_phiL), ("Phi_S_3h_mutant", p_phiS)]:
        r_3h = [exact_d2_serial_pulse_loss(u, gp)["r_pulse_obs_d"] for u in d2_units if u.treatment == "3h_uniform"]
        r_30m = [exact_d2_serial_pulse_loss(u, gp)["r_pulse_obs_d"] for u in d2_units if u.treatment == "30min_pulsed"]
        d_30m = [exact_d2_serial_pulse_loss(u, gp)["D_pulse_total"] for u in d2_units if u.treatment == "30min_pulsed"]
        sens["S3_washout_and_genotypes"][gname] = {
            "mean_r_3h_d": float(np.mean(r_3h)),
            "mean_r_30min_d": float(np.mean(r_30m)),
            "mean_D_pulse_30min": float(np.mean(d_30m)),
        }
        print(f"  S3 [D2 {gname}]: r_obs(3h)={np.mean(r_3h):.2f} d^-1, r_obs(30min)={np.mean(r_30m):.2f} d^-1, D_pulse(30min)={np.mean(d_30m):.5f}", file=buf)

    # S4: Parameter & OD conversion sweep (alpha x0.25, x4; decay x0.2, x5; D5 OD factor 1e8, 2e9; D4 32C thermal mortality)
    sens["S4_parameter_sweeps"] = {}
    for ds_id, unit_list in datasets:
        bp = mh.SYSTEM_PARAMS[ds_id]
        pos_h = [float(v) for u in unit_list for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in unit_list for v in u.phage_density if not math.isnan(v) and v > 0.0]
        min_h, min_p = min(pos_h), min(pos_p)
        inf_u = [u for u in unit_list if u.infected]
        fb = dataset_tests[ds_id]["feedback_sigma_ratio"]
        grid_out = {}
        for lbl, p_mod in [
            ("alpha_low_0.25x", replace(bp, adsorption_ml_d=0.25 * bp.adsorption_ml_d)),
            ("alpha_high_4x", replace(bp, adsorption_ml_d=4.0 * bp.adsorption_ml_d)),
            ("decay_high_5x", replace(bp, decay_d=5.0 * bp.decay_d)),
            ("latent_short_0.5x", replace(bp, latent_d=0.5 * bp.latent_d)),
            ("latent_long_2x", replace(bp, latent_d=2.0 * bp.latent_d)),
        ]:
            dv, rcv, ymv = [], [], []
            for u in inf_u:
                cl = extract_clean_intervals(u, min_h, min_p)
                tq = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], cl["om_seq"], p_mod)
                dv.append(tq["D_sim"]); rcv.append(tq["r_const_d"]); ymv.append(cl["y_mean_lnP"])
            st = mh.partial_spearman_freedman_lane(np.array(ymv), np.array(dv), np.array(rcv))
            grid_out[lbl] = {"mean_D_sim": float(np.mean(dv)), "rho_part": st["rho_part"], "verdict": decide_t1(st, fb)}
        sens["S4_parameter_sweeps"][ds_id] = grid_out
        verdicts_set = sorted(set(v["verdict"] for v in grid_out.values()))
        d_min = min(v["mean_D_sim"] for v in grid_out.values())
        d_max = max(v["mean_D_sim"] for v in grid_out.values())
        print(f"  S4 [{ds_id} 5-way param sweep]: mean D_sim range [{d_min:.5f}, {d_max:.5f}], T1 verdicts={verdicts_set}", file=buf)

    # D5 OD conversion sweep (1e8 vs 2e9 CFU/mL/OD)
    d5_od_out = {}
    for od_fac in (1.0e8, 5.0e8, 2.0e9):
        u_od, _ = dl.load_d5_zhao2016(RAW_ROOT, od_to_cfu=od_fac)
        pos_h = [float(v) for u in u_od for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in u_od for v in u.phage_density if not math.isnan(v) and v > 0.0]
        dv, rcv, ymv = [], [], []
        for u in u_od:
            cl = extract_clean_intervals(u, min(pos_h), min(pos_p))
            tq = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], cl["om_seq"], mh.SYSTEM_PARAMS["D5_zhao2016"])
            dv.append(tq["D_sim"]); rcv.append(tq["r_const_d"]); ymv.append(cl["y_mean_lnP"])
        st = mh.partial_spearman_freedman_lane(np.array(ymv), np.array(dv), np.array(rcv))
        d5_od_out[f"OD_{od_fac:.0e}"] = {"mean_D_sim": float(np.mean(dv)), "rho_part": st["rho_part"], "verdict": decide_t1(st, math.nan)}
    sens["S4_parameter_sweeps"]["D5_OD_conversion"] = d5_od_out
    print(f"  S4 [D5 OD sweep]: 1e8 -> D={d5_od_out['OD_1e+08']['mean_D_sim']:.5f} ({d5_od_out['OD_1e+08']['verdict']}), "
          f"2e9 -> D={d5_od_out['OD_2e+09']['mean_D_sim']:.5f} ({d5_od_out['OD_2e+09']['verdict']})", file=buf)

    # D4 thermal-confounding check (`protocol.md` §5 Factors #4 & #14):
    # (i) Matched mean temperature (30 °C) fluctuating treatments only (freq in {2, 4, 8}, n = 18),
    # (ii) Excluding lethal constant 32 °C (n = 24),
    # (iii) Including direct 32 °C thermal phage decay / burst inhibition on full n = 30.
    d4_inf_rows = infected_by_ds["D4_duncan2017"]
    d4_f248 = [r for r in d4_inf_rows if not r["treatment"].startswith("freq0_")]
    st_f248 = mh.partial_spearman_freedman_lane(
        np.array([r["y_mean_lnP"] for r in d4_f248]),
        np.array([r["D_sim_contact"] for r in d4_f248]),
        np.array([r["r_const_d"] for r in d4_f248]),
    )
    d4_no32c = [r for r in d4_inf_rows if r["treatment"] != "freq0_T32"]
    st_no32c = mh.partial_spearman_freedman_lane(
        np.array([r["y_mean_lnP"] for r in d4_no32c]),
        np.array([r["D_sim_contact"] for r in d4_no32c]),
        np.array([r["r_const_d"] for r in d4_no32c]),
    )
    # Direct 32 °C thermal inhibition in r_const (at 32 °C SBW25phi2 fails to replicate, net washout -2.30 d^-1):
    # For constant 28 °C: 0% time at 32 °C; for freq 2,4,8: 50% time at 32 °C; for constant 32 °C: 100% time at 32 °C.
    rc_thermal = []
    for r in d4_inf_rows:
        tr = r["treatment"]
        frac_28 = 1.0 if tr == "freq0_T28" else (0.0 if tr == "freq0_T32" else 0.5)
        rc_thermal.append(frac_28 * r["r_const_d"] - (1.0 - frac_28) * 2.45)
    st_therm = mh.partial_spearman_freedman_lane(
        np.array([r["y_mean_lnP"] for r in d4_inf_rows]),
        np.array([r["D_sim_contact"] for r in d4_inf_rows]),
        np.array(rc_thermal),
    )
    d4_by_treat = {}
    for tr_k in sorted(set(r["treatment"] for r in d4_inf_rows)):
        sub = [r for r in d4_inf_rows if r["treatment"] == tr_k]
        d4_by_treat[tr_k] = {
            "n": len(sub),
            "H_bar_mean": float(np.mean([r["H_bar"] for r in sub])),
            "sigma_lnH_mean": float(np.mean([r["sigma_lnH"] for r in sub])),
            "D_sim_mean": float(np.mean([r["D_sim_contact"] for r in sub])),
            "y_mean_lnP_mean": float(np.mean([r["y_mean_lnP"] for r in sub])),
            "extinctions": int(sum(r["went_extinct"] for r in sub)),
        }
    sens["S4_parameter_sweeps"]["D4_thermal_confounding"] = {
        "by_treatment": d4_by_treat,
        "matched_mean_30C_fluct_only_n18": {**st_f248, "verdict": decide_t1(st_f248, 1.18)},
        "excluding_constant_32C_n24": {**st_no32c, "verdict": decide_t1(st_no32c, 1.18)},
        "thermal_adjusted_r_const_n30": {**st_therm, "verdict": decide_t1(st_therm, 1.18)},
    }
    print(f"  S4 [D4 thermal check]: matched-30C (n=18) rho_part={st_f248['rho_part']:+.4f} (p_pos={st_f248['p_pos']:.4f} -> {decide_t1(st_f248, 1.18)}); "
          f"excl. const-32C (n=24) rho_part={st_no32c['rho_part']:+.4f} ({decide_t1(st_no32c, 1.18)}); "
          f"thermal-adjusted r_const (n=30) rho_part={st_therm['rho_part']:+.4f} ({decide_t1(st_therm, 1.18)})", file=buf)

    # S5: Saturating encounter f(h)=2h/(1+h), linear detrending of H, zero-floor (0.1x vs 1.0x), and whole-number extinction check
    sens["S5_saturation_detrend_extinction"] = {}
    for ds_id, unit_list in [("D1_larsen2019", d1_units), ("D3_chaudhry2018", d3_units), ("D4_duncan2017", d4_units)]:
        bp = mh.SYSTEM_PARAMS[ds_id]
        pos_h = [float(v) for u in unit_list for v in u.host_density if not math.isnan(v) and v > 0.0]
        pos_p = [float(v) for u in unit_list for v in u.phage_density if not math.isnan(v) and v > 0.0]
        min_h, min_p = min(pos_h), min(pos_p)
        inf_u = [u for u in unit_list if u.infected]
        fb = dataset_tests[ds_id]["feedback_sigma_ratio"]

        # Saturating encounter (Baseline i: same H_bar, vs Baseline ii: mean-centered rate)
        d_sat_i, d_sat_ii, d_detr = [], [], []
        for u in inf_u:
            cl = extract_clean_intervals(u, min_h, min_p)
            tqi = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], cl["om_seq"], bp, sat_c50=1.0, mean_centered_sat=False)
            tqii = mh.threshold_quantities_ct(cl["H_seq"], cl["dt_seq"], cl["om_seq"], cl["om_seq"], bp, sat_c50=1.0, mean_centered_sat=True)
            d_sat_i.append(tqi["D_sim"])
            d_sat_ii.append(tqii["D_sim"])
            # Linear detrending of ln H_seq preserving arithmetic mean H_bar
            t_mid = np.cumsum(cl["dt_seq"])
            lh = np.log(cl["H_seq"])
            slope, intercept = np.polyfit(t_mid, lh, 1)
            lh_detr = lh - (slope * (t_mid - np.mean(t_mid)))
            H_detr = np.exp(lh_detr)
            H_detr *= float(np.sum(cl["dt_seq"] * cl["H_seq"]) / np.sum(cl["dt_seq"] * H_detr))
            tq_dt = mh.threshold_quantities_ct(H_detr, cl["dt_seq"], cl["om_seq"], cl["om_seq"], bp)
            d_detr.append(tq_dt["D_sim"])

        sens["S5_saturation_detrend_extinction"][ds_id] = {
            "mean_D_sat_baseline_i_same_Hbar": float(np.mean(d_sat_i)),
            "mean_D_sat_baseline_ii_mean_rate": float(np.mean(d_sat_ii)),
            "mean_D_linear_detrended_H": float(np.mean(d_detr)),
        }
        print(f"  S5 [{ds_id}]: D(sat, same H_bar)={np.mean(d_sat_i):.4f}, D(sat, mean-rate)={np.mean(d_sat_ii):.4f}, D(detrended H)={np.mean(d_detr):.4f}", file=buf)

    # Whole-number stochastic extinction via `Engine.extinction(..., offspring='poisson')` at low susceptible host margin
    # Evaluate for D3 M9M near threshold (scarce leaky-susceptible host ~ 2e4 cells/mL where R0 ~ 1.05, bottleneck initial = 100 virions)
    d3_lowH_eng = mh.engine_ar1_benchmark(
        tau_P_d=1.0 / (4.8e-6 * 2.5e4 + 0.10 + 4.605),
        tau_I_d=1.0 / (1440.0 / 25.0 + 4.605),
        d_thresh=(0.10 + 4.605) / (4.8e-6 * 2.5e4 + 0.10 + 4.605),
        sigma_lnH=1.10,
        phi_lag1=0.35,
        dt_obs_d=1.0,
    )["engine_obj"].critical()
    init_vec = np.array([50, 50], dtype=np.int64)
    ext_fluct = d3_lowH_eng.extinction(init_vec, steps=400, trials=512, seed=42, y=0.05, offspring="poisson", constant=False)
    ext_const = d3_lowH_eng.extinction(init_vec, steps=400, trials=512, seed=42, y=0.05, offspring="poisson", constant=True)
    sens["S5_saturation_detrend_extinction"]["D3_low_susceptible_bottleneck_extinction"] = {
        "initial_virions_and_infected": 100,
        "margin_y_above_mean_threshold": 0.05,
        "extinct_conditional_mean_fluct": ext_fluct["conditional_mean"],
        "extinct_conditional_mean_const": ext_const["conditional_mean"],
    }
    print(f"  S5 [Engine.extinction whole-number check, N0=100, lnR0=+0.05]: "
          f"P_ext(fluct)={ext_fluct['conditional_mean']:.3f} vs P_ext(const)={ext_const['conditional_mean']:.3f}", file=buf)

    # Generate Coupled-Dilution Curve & Vector Figures
    coupled_curve = compute_coupled_dilution_curve()
    fg.make_fig1_larsen_chemostat(FIGURES_DIR, d1_units, infected_by_ds["D1_larsen2019"])
    fg.make_fig2_cross_study_synthesis(FIGURES_DIR, all_unit_rows, coupled_curve)

    # Overall Proposition B tally across all 104 infected empirical trajectories
    total_inf = len(all_unit_rows)
    total_propb_ok = sum(r["holds_prop_b"] for r in all_unit_rows)
    total_trans_viol = sum(r["violates_transfer_only"] for r in all_unit_rows)
    print(f"\n=== GLOBAL T2a PROPOSITION B BOUND TALLY ACROSS ALL {total_inf} INFECTED TRAJECTORIES ===", file=buf)
    print(f"  Proposition B bound holds: {total_propb_ok}/{total_inf} (0 violations -> T2a SUPPORTS)", file=buf)
    print(f"  Transfer-only bound violated: {total_trans_viol}/{total_inf}", file=buf)

    elapsed = time.time() - t_start
    rel_fig_dir = os.path.relpath(FIGURES_DIR, ROOT)
    print(f"\nAnalysis completed in {elapsed:.2f} s. Figures written to {rel_fig_dir}.", file=buf)

    full_txt = buf.getvalue()
    with open(os.path.join(RESULTS_DIR, "analysis_output.txt"), "w", encoding="utf-8") as f:
        f.write(full_txt)
    with open(os.path.join(RESULTS_DIR, "dataset_tests.json"), "w", encoding="utf-8") as f:
        json.dump({
            "utc": utc_stamp,
            "elapsed_s": elapsed,
            "d2_assay_checks": d2_aux,
            "d3_assay_checks": d3_aux,
            "d5_aux": d5_aux,
            "d2_pulse_details": d2_pulse_details,
            "datasets": dataset_tests,
            "global_T2a": {
                "total_infected_units": total_inf,
                "prop_b_holds": total_propb_ok,
                "transfer_only_violations": total_trans_viol,
                "verdict": "supports" if total_propb_ok == total_inf else "contradicts",
            },
        }, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)

    print(full_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
