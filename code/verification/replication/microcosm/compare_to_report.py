#!/usr/bin/env python3
"""Compare independent replication results (`replication_results.json`) against
`casestudies/microcosm/results/summary_metrics.json` and `report.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
REP = json.loads((HERE / "replication_results.json").read_text())
ORIG = json.loads((REPO_ROOT / "casestudies/microcosm/results/summary_metrics.json").read_text())


def check_item(
    name: str,
    rep_est: float,
    orig_est: float,
    rep_ci: tuple[float, float] | None = None,
    orig_ci: tuple[float, float] | None = None,
    rep_p: float | None = None,
    orig_p: float | None = None,
    rep_label: str | None = None,
    orig_label: str | None = None,
) -> dict:
    abs_diff = abs(rep_est - orig_est)
    rel_diff = abs_diff / max(abs(orig_est), 1e-15)
    if orig_ci is not None:
        in_interval = (orig_ci[0] - 1e-12) <= rep_est <= (orig_ci[1] + 1e-12)
        crit_ok = in_interval
    else:
        in_interval = None
        crit_ok = rel_diff <= 0.10
    lbl_ok = (rep_label == orig_label) if (rep_label is not None and orig_label is not None) else True
    return {
        "name": name,
        "rep_est": rep_est,
        "orig_est": orig_est,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "rep_ci": list(rep_ci) if rep_ci else None,
        "orig_ci": list(orig_ci) if orig_ci else None,
        "rep_p": rep_p,
        "orig_p": orig_p,
        "rep_label": rep_label,
        "orig_label": orig_label,
        "in_orig_ci": in_interval,
        "agrees": bool(crit_ok and lbl_ok),
    }


def main() -> None:
    rows = []

    # --- T1 ---
    t1a_r = REP["S1_T1"]["T1a"]
    rows.append(check_item(
        "T1a: amplitude vs T_surv (Spearman rho)",
        t1a_r["rho"], ORIG["T1a"]["rho"],
        (t1a_r["ci_lo"], t1a_r["ci_hi"]), (ORIG["T1a_interval"]["lo"], ORIG["T1a_interval"]["hi"]),
        t1a_r["p_less_exact"], ORIG["T1a_exact"]["p_less_exact"],
        t1a_r["status"], ORIG["T1a"]["status"],
    ))
    rows.append(check_item(
        "T1a secondary: Floquet D vs T_surv (Spearman rho)",
        REP["S1_T1"]["T1a_by_D"]["rho"], ORIG["T1a_by_D"]["rho"],
        None, None,
        REP["S1_T1"]["T1a_by_D"]["p_less_exact"], ORIG["T1a_by_D"]["p_less"],
        "not decided", "not decided",
    ))
    for key, label_txt in [
        ("T1b_total", "T1b: early log PFU vs total loss (OLS slope)"),
        ("T1b_dynamic", "T1b: early log PFU vs dynamic Floquet D (OLS slope)"),
        ("T1b_adjusted", "T1b-adj: early log PFU vs dynamic D | ln(mean H)"),
    ]:
        rr = REP["S1_T1"][key]
        oo = ORIG[key]
        rows.append(check_item(
            label_txt,
            rr["slope"], oo["slope"],
            (rr["ci_lo"], rr["ci_hi"]), (oo["lo"], oo["hi"]),
            rr["p_less"], oo["p_less"],
            rr["status"], oo["status"],
        ))

    # --- T2 ---
    for key, label_txt, p_field in [
        ("T2a", "T2a: Red - White early prevalence growth r_{0->16}", "p_less_exact"),
        ("T2a_synchronous", "T2a (synchronous): Red - White r_{0->16}", "p_less_exact"),
        ("T2a_infected_abundance", "T2a (infected abundance): Red - White r_{I,0->16}", "p_less_exact"),
        ("T2b", "T2b: Red - White Day-56 log prevalence", "p_greater_exact"),
        ("T2c_r016", "T2c (early): Async - Sync under Red r_{0->16}", "p_greater_exact"),
        ("T2c_late_logp", "T2c (late): Async - Sync under Red Days 16-56", "p_greater_exact"),
        ("T2c_interaction_r016", "T2c interaction (early): (Async-Sync)_Red - (Async-Sync)_White", "p_greater_exact"),
        ("T2c_interaction_late_logp", "T2c interaction (late): (Async-Sync)_Red - (Async-Sync)_White", "p_greater_exact"),
    ]:
        rr = REP["S2_T2"][key]
        oo = ORIG[key]
        rows.append(check_item(
            label_txt,
            rr["diff"], oo["diff"],
            (rr["ci_lo"], rr["ci_hi"]), (oo["lo"], oo["hi"]),
            rr[p_field], oo[p_field],
            rr["status"], oo["status"],
        ))

    r_adj = REP["S2_T2"]["T2a_host_adjusted"]
    o_adj = ORIG["T2a_host_adjusted"]
    rows.append(check_item(
        "T2a-adj: Red - White r_{0->16} | ln(mean H), Sync, Mig",
        r_adj["slope"], o_adj["slope"],
        (r_adj["ci_lo"], r_adj["ci_hi"]), (o_adj["lo"], o_adj["hi"]),
        r_adj["p_less"], o_adj["p_less"],
        r_adj["status"], o_adj["status"],
    ))

    # --- T3 & Extractions ---
    for key, label_txt in [
        ("T3a_Pearson", "T3a: partial Pearson r(log AUC, sd_lnH | ln mean_H)"),
        ("T3a_Spearman", "T3a: partial Spearman r(log AUC, sd_lnH | ln mean_H)"),
    ]:
        rr = REP["S3_to_S7"][key]
        oo = ORIG[key]
        rows.append(check_item(
            label_txt,
            rr["r"], oo["rho"],
            (rr["ci_lo"], rr["ci_hi"]), (oo["lo"], oo["hi"]),
            rr["p_greater"], oo["p_greater"],
            rr["status"], oo["status"],
        ))

    # S4 calibrated parameters
    s4e = REP["S3_to_S7"]["S4_control_exp"]
    o4e = ORIG["S4"]["survival"][0]
    rows.append(check_item(
        "S4 exponential spore decay mu_out (/day)",
        s4e["lambda_day"], o4e["lambda_per_day"],
        (s4e["lambda_lo"], s4e["lambda_hi"]), (o4e["lambda_lo"], o4e["lambda_hi"]),
    ))
    s4w = REP["S3_to_S7"]["S4_control_weibull"]
    o4w = ORIG["S4"]["survival"][1]
    rows.append(check_item(
        "S4 Weibull spore survival shape k",
        s4w["k"], o4w["k"],
        (s4w["k_lo"], s4w["k_hi"]), (o4w["k_lo"], o4w["k_hi"]),
    ))
    s4f = REP["S3_to_S7"]["S4_filtering_20C"]
    o4f = next(x for x in ORIG["S4"]["filtering"] if x["trt"] == "20")
    rows.append(check_item(
        "S4 filtering rate F at 20 C (mL/host/day)",
        s4f["mean"], o4f["mean"],
        (s4f["lo"], s4f["hi"]), (o4f["lo"], o4f["hi"]),
    ))
    s4l = REP["S3_to_S7"]["S4_host_lifespan_trt0"]
    o4l = next(x for x in ORIG["S4"]["host_duration"] if x["trt"] == 0)
    rows.append(check_item(
        "S4 infected host lifespan tau_in (days)",
        s4l["mean"], o4l["mean_lifespan"],
        (s4l["lo"], s4l["hi"]), (o4l["lifespan_lo"], o4l["lifespan_hi"]),
    ))
    c160_r = next(x for x in REP["S3_to_S7"]["S4_contact_curve"] if x["H_per_L"] == 160)
    c160_o = next(x for x in ORIG["S4"]["contact_curve"] if x["H_per_L"] == 160)
    rows.append(check_item(
        "S4 contact-waiting death share d at 160 hosts/L",
        c160_r["d"], c160_o["d"],
    ))

    # S5 Pasteuria thermal clock
    c5_r = REP["S3_to_S7"]["S5_clocks"]["all_tti"]
    c5_o = next(x for x in ORIG["S5"]["clock"] if x["origin"] == "all" and x["metric"] == "tti")
    rows.append(check_item(
        "S5 Pasteuria thermal clock sensitivity chi_in (/C)",
        c5_r["chi"], c5_o["chi"],
        (c5_r["chi_lo"], c5_r["chi_hi"]), (c5_o["chi_lo"], c5_o["chi_hi"]),
    ))

    # S7 Krichel endemic prevalence
    s7c_r = REP["S3_to_S7"]["S7"]["const_20C_endemic_gt150_prev"]
    s7c_o = next(x for x in ORIG["S7"]["summary"] if x["window"] == "endemic_day150" and x["variable"] == 0 and x["metric"] == "mean_prevalence")
    rows.append(check_item(
        "S7 constant 20C endemic prevalence (day > 150)",
        s7c_r["mean"], s7c_o["mean"],
        (s7c_r["lo"], s7c_r["hi"]), (s7c_o["lo"], s7c_o["hi"]),
    ))
    s7v_r = REP["S3_to_S7"]["S7"]["var_14_26C_endemic_gt150_prev"]
    s7v_o = next(x for x in ORIG["S7"]["summary"] if x["window"] == "endemic_day150" and x["variable"] == 1 and x["metric"] == "mean_prevalence")
    rows.append(check_item(
        "S7 variable 14-26C endemic prevalence (day > 150)",
        s7v_r["mean"], s7v_o["mean"],
        (s7v_r["lo"], s7v_r["hi"]), (s7v_o["lo"], s7v_o["hi"]),
    ))

    # --- Weather Amendment W1a, W1b, W2 ---
    w1a_r = REP["weather"]["W1a"]
    w1a_o = ORIG["W1a"]
    rows.append(check_item(
        "W1a: spore-to-outbreak signed lag vs T_14d (Spearman rho)",
        w1a_r["signed_rho"], w1a_o["signed_rho"],
        (w1a_r["ci_lo"], w1a_r["ci_hi"]), None,
        w1a_r["p_greater_mc"], w1a_o["signed_p_greater"],
    ))
    for w_r, w_o in zip(REP["weather"]["W1b"], ORIG["W1b"]):
        assert w_r["parasite"] == w_o["parasite"]
        par = w_r["parasite"]
        rows.append(check_item(
            f"W1b ({par}): weather-adjusted partial r(ln H, prev | T_14d, srad_14d)",
            w_r["weather_partial_r"], w_o["weather_partial_r"],
            (w_r["weather_partial_lo"], w_r["weather_partial_hi"]), None,
            rep_label=f"changed_sign={w_r['changed_sign']}, halved={w_r['halved']}",
            orig_label=f"changed_sign={w_o['changed_sign']}, halved={w_o['halved']}",
        ))

    w2_r = REP["weather"]["W2"]
    w2_o = ORIG["W2"]
    rows.append(check_item("W2: proxy tau_in(T) min (days)", w2_r["tau_in_proxy_min"], w2_o["tau_proxy_min"]))
    rows.append(check_item("W2: proxy tau_in(T) max (days)", w2_r["tau_in_proxy_max"], w2_o["tau_proxy_max"]))
    rows.append(check_item("W2: second-order D2 min", w2_r["D2_min"], w2_o["D2_min"]))
    rows.append(check_item("W2: second-order D2 max", w2_r["D2_max"], w2_o["D2_max"]))

    # --- Sensitivities ---
    s2m = REP["sensitivities"]["S2_markov_36cells"]
    rows.append(check_item(
        "Sens 1a (S2 host-only memoryless waste): D_Red - D_White",
        s2m["key_memoryless_waste_host_only"]["diff"],
        0.014304348346504516 - 0.004991739795976902,
    ))
    rows.append(check_item(
        "Sens 1b (S2 5C cold-arrest memoryless waste): D_Red - D_White",
        s2m["key_memoryless_waste_cold_arrest"]["diff"],
        -0.042439015720040864 - 0.07475124241505324,
    ))
    s34g = REP["sensitivities"]["S34_grid_30cells"]
    rows.append(check_item(
        "Sens 2a (S3+S4 periodic Floquet min, k_in=16 waste P=14d)",
        s34g["min_Floquet_rep"], ORIG["S34_predictions"]["min_Floquet"],
    ))
    rows.append(check_item(
        "Sens 2b (S3+S4 periodic Floquet max, k_in=16 waste P=112d)",
        s34g["max_Floquet_rep"], ORIG["S34_predictions"]["max_Floquet"],
    ))
    rows.append(check_item(
        "Sens 2c (S3+S4 telegraph min, k_in=1 waste dwell=7d)",
        s34g["min_telegraph_rep"], ORIG["S34_predictions"]["min_telegraph"],
    ))
    rows.append(check_item(
        "Sens 2d (S3+S4 telegraph max, k_in=16 waste dwell=112d)",
        s34g["max_telegraph_rep"], ORIG["S34_predictions"]["max_telegraph"],
    ))

    out_path = HERE / "comparison_summary.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n")

    n_ok = sum(r["agrees"] for r in rows)
    print(f"Compared {len(rows)} items: {n_ok}/{len(rows)} agree.")
    for r in rows:
        flag = "OK" if r["agrees"] else "FAIL"
        print(f"[{flag}] {r['name']}: rep={r['rep_est']:.6g}, orig={r['orig_est']:.6g}, abs_diff={r['abs_diff']:.2e}, rel={r['rel_diff']:.2e}")
    print("\nTadiri duplicate t56 audit:", json.dumps(REP["S3_to_S7"]["S5_tadiri_dup_t56_audit"]))
    print("Ferris embedded Treatment Key:", json.dumps(REP["S1_T1"]["embedded_treatment_key"]))
    print("S3 arm sd_logH audit:", json.dumps([
        {"H0_L": a["H0_L"], "strict": a["mean_sd_logH_strict"], "nanmean": a["mean_sd_logH_nanmean"], "n_undef": a["n_undefined_sd_logH"]}
        for a in REP["S3_to_S7"]["S3_arms"]
    ]))


if __name__ == "__main__":
    main()
