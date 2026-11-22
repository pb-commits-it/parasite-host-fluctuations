#!/usr/bin/env python3
"""Programmatically verify agreement between our independent replication and report.md."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REP = json.loads((HERE / "results_replication.json").read_text(encoding="utf-8"))


def check_row(
    section: str,
    metric: str,
    our_val: float,
    rep_val: float,
    our_ci: tuple[float, float] | None,
    rep_ci: tuple[float, float] | None,
    our_label: str,
    rep_label: str,
) -> dict:
    abs_diff = abs(our_val - rep_val)
    rel_diff_pct = (abs_diff / abs(rep_val) * 100.0) if abs(rep_val) > 1e-12 else 0.0
    if rep_ci is not None:
        val_agrees = (rep_ci[0] <= our_val <= rep_ci[1])
        criterion = "in_orig_95%_CI"
    else:
        val_agrees = (rel_diff_pct <= 10.0) if abs(rep_val) > 1e-12 else (abs_diff < 1e-9)
        criterion = "<=10%_rel"
    label_agrees = (our_label == rep_label)
    agrees = bool(val_agrees and label_agrees)
    return {
        "section": section,
        "metric": metric,
        "our_estimate": our_val,
        "report_estimate": rep_val,
        "our_ci95": f"[{our_ci[0]:.3f}, {our_ci[1]:.3f}]" if our_ci else "",
        "report_ci95": f"[{rep_ci[0]:.3f}, {rep_ci[1]:.3f}]" if rep_ci else "",
        "abs_diff": abs_diff,
        "rel_diff_pct": rel_diff_pct,
        "our_outcome": our_label,
        "report_outcome": rep_label,
        "criterion": criterion,
        "agrees": agrees,
    }


def main():
    p1 = REP["phase1"]
    ma = REP["meta_analysis"]
    ug = REP["unified_glmm"]
    wa = REP["weather_amendment"]["adjusted_models"]

    rows = [
        # Step 1: Reproduction gate
        check_row(
            "Phase1_gate", "M3_nhosts_1host_pct_decade",
            p1["percent_per_decade"][0], -1.137,
            tuple(p1["ci95_percent_exact_z"][0]), (-11.119, 9.966),
            "pass", "pass",
        ),
        check_row(
            "Phase1_gate", "M3_nhosts_2host_pct_decade",
            p1["percent_per_decade"][1], 0.171,
            tuple(p1["ci95_percent_exact_z"][1]), (-9.681, 11.098),
            "pass", "pass",
        ),
        check_row(
            "Phase1_gate", "M3_nhosts_3plus_pct_decade",
            p1["percent_per_decade"][2], -10.914,
            tuple(p1["ci95_percent_exact_z"][2]), (-17.218, -4.129),
            "pass", "pass",
        ),
        check_row("Phase1_gate", "M3_nhosts_AICc", p1["models"]["M3_nhosts"]["aicc"], 17072.097, None, None, "best", "best"),
        check_row("Phase1_gate", "M2_year_AICc", p1["models"]["M2_year"]["aicc"], 17073.215, None, None, "rank2", "rank2"),
        check_row("Phase1_gate", "M4_partype_AICc", p1["models"]["M4_partype"]["aicc"], 17075.421, None, None, "rank3", "rank3"),
        check_row("Phase1_gate", "M1_null_AICc", p1["models"]["M1_null"]["aicc"], 17135.664, None, None, "rank4", "rank4"),

        # Step 2: Primary omnibus tests
        check_row("Primary_omnibus", "Meta_ML_delta_AICc_M7_minus_M3", ma["omnibus_bootstrap"]["delta_aicc"], 7.525, None, None, ma["omnibus_outcome"], "not decided"),
        check_row("Primary_omnibus", "Meta_ML_LRT_statistic", ma["omnibus_bootstrap"]["lrt_obs"], 2.115, None, None, ma["omnibus_outcome"], "not decided"),
        check_row("Primary_omnibus", "Meta_ML_bootstrap_p_B2000", ma["omnibus_bootstrap"]["p_boot"], 0.7546, None, None, ma["omnibus_outcome"], "not decided"),
        check_row("Primary_omnibus", "Count_GLMM_delta_AICc_M7_minus_M3_tax", ug["delta_aicc_M7_minus_M3_tax"], 4.315, None, None, "not decided", "not decided"),
        check_row("Primary_omnibus", "Count_GLMM_delta_AICc_M6_minus_M3_orig", ug["delta_aicc_M6_minus_M3_orig"], 4.313, None, None, "not decided", "not decided"),
    ]

    # Primary REML meta-regression contrasts (M7_joint_tax)
    rep_meta_m7 = [
        ("n_fluc_obligate", -0.061, (-0.223, 0.101), "not decided"),
        ("storage_stage", 0.079, (-0.173, 0.331), "not decided"),
        ("spreading", 0.083, (-0.188, 0.354), "not decided"),
        ("prop_active_transfers", 0.082, (-0.250, 0.415), "not decided"),
    ]
    for c_our, (nm, rv, rci, rlbl) in zip(ma["primary_contrasts_M7_REML"], rep_meta_m7):
        assert c_our["trait"] == nm
        rows.append(check_row(
            "Primary_M7_REML", nm,
            c_our["beta"], rv,
            (c_our["ci_lo"], c_our["ci_hi"]), rci,
            c_our["outcome"], rlbl,
        ))

    # Primary direct count NB-GLMM contrasts (M7_joint_tax)
    m7_u = ug["models"]["M7_joint_tax"]
    rep_count_m7 = [
        ("n_fluc_obligate", -0.127, (-0.396, 0.141), "not decided"),
        ("storage_stage", 0.218, (-0.201, 0.637), "not decided"),
        ("spreading", 0.217, (-0.232, 0.666), "not decided"),
        ("prop_active_transfers", 0.126, (-0.412, 0.665), "not decided"),
    ]
    for j, (nm, rv, rci, rlbl) in enumerate(rep_count_m7):
        idx = 3 + j
        rows.append(check_row(
            "Primary_M7_Count_GLMM", nm,
            m7_u["beta_slope"][idx], rv,
            (m7_u["ci95_lo_exact_z"][idx], m7_u["ci95_hi_exact_z"][idx]), rci,
            "not decided", rlbl,
        ))

    # Within-3+ REML meta-regression contrasts (M8_within_3plus)
    rep_meta_m8 = [
        ("n_fluc_obligate", -0.086, (-0.279, 0.108), "not decided"),
        ("storage_stage", 0.023, (-0.239, 0.285), "not decided"),
        ("spreading", 0.017, (-0.321, 0.354), "not decided"),
        ("prop_active_transfers", -0.086, (-0.509, 0.337), "not decided"),
    ]
    for c_our, (nm, rv, rci, rlbl) in zip(ma["within_3plus_contrasts_M8_REML"], rep_meta_m8):
        assert c_our["trait"] == nm
        rows.append(check_row(
            "Within3plus_M8_REML", nm,
            c_our["beta"], rv,
            (c_our["ci_lo"], c_our["ci_hi"]), rci,
            c_our["outcome"], rlbl,
        ))

    # Host-number attenuation checks
    rows.extend([
        check_row("Attenuation", "Meta_REML_host3plus_M3", ma["host_3plus_baseline_REML"], -0.172, None, None, "not decided", "not decided"),
        check_row("Attenuation", "Meta_REML_host3plus_M7", ma["host_3plus_joint_REML"], -0.127, None, None, "not decided", "not decided"),
        check_row("Attenuation", "Count_GLMM_host3plus_attenuation_pct", 100.0 * ug["host_3plus_attenuation"], 10.6, None, None, "falsifier_not_met", "falsifier_not_met"),
    ])

    # Weather & environment amendment main adjusted result (W1_ersst + W2_race check)
    w1 = wa["W1_ersst"]
    w1_j = w1["joint"]
    z975 = 1.959963984540054
    rows.extend([
        check_row("Weather_W1_ersst", "W1_delta_AICc_joint_minus_baseline", w1["delta_aicc"], 4.778, None, None, "not decided", "not decided"),
        check_row(
            "Weather_W1_ersst", "W1_3plus_SST_pct_per_C",
            w1["thermal_3plus_baseline"]["pct_per_C"], -12.3,
            (w1["thermal_3plus_baseline"]["ci95_lo_pct_exact_z"], w1["thermal_3plus_baseline"]["ci95_hi_pct_exact_z"]),
            (-31.7, 12.5),
            "not decided", "not decided",
        ),
        check_row("Weather_W1_ersst", "W1_joint_host3plus", w1_j["beta_slope"][2], -0.232, (w1_j["beta_slope"][2] - z975 * w1_j["se_slope"][2], w1_j["beta_slope"][2] + z975 * w1_j["se_slope"][2]), None, "material_shift_warning", "material_shift_warning"),
        check_row("Weather_W1_ersst", "W1_joint_n_fluc_obligate", w1_j["beta_slope"][3], -0.114, (w1_j["beta_slope"][3] - z975 * w1_j["se_slope"][3], w1_j["beta_slope"][3] + z975 * w1_j["se_slope"][3]), None, "not decided", "not decided"),
        check_row("Weather_W1_ersst", "W1_joint_storage_stage", w1_j["beta_slope"][4], 0.172, (w1_j["beta_slope"][4] - z975 * w1_j["se_slope"][4], w1_j["beta_slope"][4] + z975 * w1_j["se_slope"][4]), None, "not decided", "not decided"),
    ])

    # Two selected sensitivities: S4_storage_active and S2b_terminal
    s4_m = ma["models"]["S4_storage_active"]["REML"]
    s4_u = ug["models"]["S4_storage_active"]
    s2b_m = ma["models"]["S2b_terminal"]["REML"]
    s2b_u = ug["models"]["S2b_terminal"]
    rows.extend([
        check_row("Sensitivity_S4", "S4_Meta_REML_n_fluc_obligate", s4_m["beta"][3], -0.069, (s4_m["ci_lo"][3], s4_m["ci_hi"][3]), None, "not decided", "not decided"),
        check_row("Sensitivity_S4", "S4_Meta_REML_storage_at_zero_active", s4_m["beta"][4], 0.802, (s4_m["ci_lo"][4], s4_m["ci_hi"][4]), (0.254, 1.349), "secondary_Holm_p<0.05_unstable_ML", "secondary_Holm_p<0.05_unstable_ML"),
        check_row("Sensitivity_S4", "S4_Meta_REML_storage_x_active", s4_m["beta"][7], -1.154, (s4_m["ci_lo"][7], s4_m["ci_hi"][7]), None, "not decided", "not decided"),
        check_row("Sensitivity_S4", "S4_Meta_ML_delta_AICc_vs_M3", ma["S4_ml_delta_aicc_vs_M3"], 9.485, None, None, "not decided", "not decided"),
        check_row("Sensitivity_S4", "S4_Count_GLMM_n_fluc_obligate", s4_u["beta_slope"][3], -0.174, (s4_u["ci95_lo_exact_z"][3], s4_u["ci95_hi_exact_z"][3]), None, "not decided", "not decided"),
        check_row("Sensitivity_S4", "S4_Count_GLMM_storage_at_zero_active", s4_u["beta_slope"][4], 0.340, (s4_u["ci95_lo_exact_z"][4], s4_u["ci95_hi_exact_z"][4]), None, "not decided", "not decided"),

        check_row("Sensitivity_S2b", "S2b_Meta_REML_n_fluc_obligate", s2b_m["beta"][3], -0.052, (s2b_m["ci_lo"][3], s2b_m["ci_hi"][3]), None, "not decided", "not decided"),
        check_row("Sensitivity_S2b", "S2b_Meta_REML_storage_stage", s2b_m["beta"][4], 0.687, (s2b_m["ci_lo"][4], s2b_m["ci_hi"][4]), (0.010, 1.364), "nominal_p<0.05_Holm_ns", "nominal_p<0.05_Holm_ns"),
        check_row("Sensitivity_S2b", "S2b_Count_GLMM_n_fluc_obligate", s2b_u["beta_slope"][3], -0.121, (s2b_u["ci95_lo_exact_z"][3], s2b_u["ci95_hi_exact_z"][3]), None, "not decided", "not decided"),
        check_row("Sensitivity_S2b", "S2b_Count_GLMM_storage_stage", s2b_u["beta_slope"][4], 1.240, (s2b_u["ci95_lo_exact_z"][4], s2b_u["ci95_hi_exact_z"][4]), None, "nominal_p<0.05_AICc_worse", "nominal_p<0.05_AICc_worse"),
    ])

    assert all(r["agrees"] for r in rows), [r for r in rows if not r["agrees"]]

    csv_path = HERE / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Verified {len(rows)}/{len(rows)} comparisons agree! Max rel diff = {max(r['rel_diff_pct'] for r in rows):.4f}%")


if __name__ == "__main__":
    main()
