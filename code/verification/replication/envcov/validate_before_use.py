#!/usr/bin/env python3
"""Rule 7 pre-analysis validation checks for envcov replication (run BEFORE touching empirical data)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import integrate, special

import data_and_covariates as dc
import lifecycle_solver as ls
import stats_methods as sm


def run_all_validations(save_json: bool = True) -> dict:
    checks = []

    # V1: Exact 6-site Spearman cutoff and Freedman-Lane partial correlation vs two-step OLS + null Type I error
    x6 = np.arange(6.0)
    res_rev = sm.exact_spearman_test_6(x6, x6[::-1])
    # Enumerate all 720 orderings against cutoff -29/35
    cx = x6 - x6.mean()
    all_rhos = (cx[sm.ALL_PERMS_6] @ cx) / float(cx @ cx)
    n_le_cut = int(np.sum(all_rhos <= -29.0 / 35.0 + 1e-12))
    err_v1a = max(abs(res_rev["rho"] + 1.0), abs(res_rev["p_neg"] - 1.0 / 720.0), abs(n_le_cut - 21))
    checks.append({
        "id": "V1a_spearman6_exact_cutoff",
        "claim": "Exact 6-site Spearman gives rho=-1, p=1/720 on reverse rank and 21/720 orderings <= -29/35",
        "cases": 720,
        "tolerance": 1e-13,
        "observed_error": float(err_v1a),
        "pass": bool(err_v1a <= 1e-13),
    })

    rng = np.random.default_rng(104729)
    max_fl_ols_err = 0.0
    for _ in range(10):
        yv = rng.normal(size=6)
        xv = rng.normal(size=6)
        zv = rng.normal(size=6)
        fl = sm.freedman_lane_partial_test_6(yv, xv, zv, rank_transform=False)
        X_nuis = np.column_stack([np.ones(6), zv])
        ey = yv - X_nuis @ np.linalg.lstsq(X_nuis, yv, rcond=None)[0]
        ex = xv - X_nuis @ np.linalg.lstsq(X_nuis, xv, rcond=None)[0]
        r_ols = sm.pearson_corr(ex, ey)
        max_fl_ols_err = max(max_fl_ols_err, abs(fl["r_part"] - r_ols))
    # Also check batch rank Freedman-Lane matches scalar routine with ties
    yt_mat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 2.0, 5.0, 1.0, 4.0, 4.0]])
    xt_vec = np.array([0.4, 1.0, 0.5, 0.56, 0.57, 0.48])
    zt_vec = np.array([2.0, 0.0, 5.0, 1.0, 3.0, 3.0])
    bat = sm.batch_rank_tests_6(yt_mat, xt_vec, zt_vec)
    for idx in range(2):
        sc = sm.freedman_lane_partial_test_6(yt_mat[idx], xt_vec, zt_vec, rank_transform=True)
        max_fl_ols_err = max(max_fl_ols_err, abs(bat["r_part"][idx] - sc["r_part"]), abs(bat["p_adj_neg"][idx] - sc["p_neg"]))
    checks.append({
        "id": "V1b_freedman_lane_vs_ols_and_ties",
        "claim": "Freedman-Lane partial correlation matches explicit OLS residual correlation and tied scalar routine",
        "cases": 12,
        "tolerance": 1e-13,
        "observed_error": float(max_fl_ols_err),
        "pass": bool(max_fl_ols_err <= 1e-13),
    })

    # Null size check of 1-sided linear and rank Freedman-Lane at alpha = 0.05 across random nuisance Z (5,000 draws)
    x_std = np.array([-0.81, 1.98, -0.31, -0.03, 0.01, -0.84])
    x_std = (x_std - x_std.mean()) / x_std.std(ddof=1)
    n_null = 5000
    p_lin_hits = 0
    p_rnk_hits = 0
    for _ in range(n_null):
        z_draw = 0.6 * x_std + 0.8 * rng.normal(size=6)
        y_draw = 0.5 * z_draw + rng.normal(size=6)
        if sm.freedman_lane_partial_test_6(y_draw, x_std, z_draw, rank_transform=False)["p_neg"] < 0.05:
            p_lin_hits += 1
        if sm.freedman_lane_partial_test_6(y_draw, x_std, z_draw, rank_transform=True)["p_neg"] < 0.05:
            p_rnk_hits += 1
    type1_lin = p_lin_hits / n_null
    type1_rnk = p_rnk_hits / n_null
    checks.append({
        "id": "V1c_freedman_lane_null_calibration",
        "claim": "1-sided Freedman-Lane linear and rank tests at N=6 have null rejection rates in [0.030, 0.058]",
        "cases": n_null,
        "tolerance": "in [0.030, 0.058]",
        "observed_values": {"linear": type1_lin, "rank": type1_rnk},
        "pass": bool(0.030 <= type1_lin <= 0.058 and 0.030 <= type1_rnk <= 0.058),
    })

    # V2: Within-group fixed-effects OLS vs explicit dummy matrix + block bootstrap exact line with gaps
    groups_syn = []
    for g in range(4):
        xg = rng.normal(size=15)
        zg = rng.normal(size=(15, 2))
        yg = (-2.0 + 1.5 * g) + 0.65 * xg - 0.3 * zg[:, 0] + 0.4 * zg[:, 1] + 0.1 * rng.normal(size=15)
        groups_syn.append((xg, yg, zg))
    b_fe, _, _, _ = sm.within_group_ols(groups_syn)
    b_dum = sm.explicit_dummy_ols(groups_syn)
    fe_err = abs(b_fe - b_dum)

    recs_exact = []
    for g_idx, g_name in enumerate(["G1", "G2", "G3"]):
        for yr in range(1995, 2013):
            if (yr + g_idx) % 7 == 0:
                continue
            xv = float(rng.normal())
            wv = math.sin(0.5 * yr)
            yv = (10.0 * g_idx) + 0.42 * xv - 0.25 * wv
            recs_exact.append({"plot": g_name, "mouse_year": yr, "x": xv, "y": yv, "w": wv})
    boot_ex = sm.cary_two_year_block_bootstrap(recs_exact, ["G1", "G2", "G3"], ["w"], n_boot=500, seed=77)
    boot_err = max(abs(boot_ex["b"] - 0.42), boot_ex["se"])
    v2_err = max(fe_err, boot_err)
    checks.append({
        "id": "V2_fe_ols_and_block_bootstrap",
        "claim": "Within-group FE OLS matches dummy-variable OLS and 2-yr block bootstrap recovers exact gapped slope",
        "cases": 501,
        "tolerance": 1e-12,
        "observed_error": float(v2_err),
        "pass": bool(v2_err <= 1e-12),
    })

    # V3: Wallace (2019) constant-temperature threshold crossing
    r_ln_22, r_na_22 = dc.daily_maturation_rates(np.full(250, 22.0), scheme="wallace2019")
    c_ln = dc.first_crossing_days(r_ln_22, 0, 1.0)
    c_na = dc.first_crossing_days(r_na_22, 0, 1.0)
    r_ln_14, _ = dc.daily_maturation_rates(np.full(50, 14.9), scheme="wallace2019")
    c_below = dc.first_crossing_days(r_ln_14, 0, 1.0)
    v3_err = abs(c_ln - math.ceil(1.0 / float(r_ln_22[0]))) + abs(c_na - math.ceil(1.0 / float(r_na_22[0]))) + (0 if c_below is None else 1)
    checks.append({
        "id": "V3_wallace_thermal_crossing",
        "claim": "Constant-temperature Wallace maturation crossing equals ceil(1/m(T)) above 15 C and censors below 15 C",
        "cases": 3,
        "tolerance": 0,
        "observed_error": int(v3_err),
        "pass": bool(v3_err == 0),
    })

    # V4: Matrix-resolvent D2 vs closed-form serial-chain D2
    max_d2_rel = 0.0
    for m_sub in (1, 8, 49):
        for phi_val in (0.0, 0.75, 0.9475):
            sp = ls.BorreliaSpec(
                tau_mouse=22.166666666666668,
                tau_tick=54.0,
                sigma_mouse=0.85,
                phi_mouse=phi_val,
                kappa1=0.7,
                tick_shape="chain" if m_sub > 1 else "memoryless",
                tick_substages=m_sub,
            )
            d2_mat = ls.IndependentCycleModel(sp).second_order_d2()
            d2_cf = ls.closed_form_serial_d2([22.166666666666668, 54.0], [1, m_sub], 0.85, phi_val, kappa=0.7)
            max_d2_rel = max(max_d2_rel, abs(d2_mat - d2_cf) / abs(d2_cf))
    checks.append({
        "id": "V4_resolvent_vs_closed_form_d2",
        "claim": "Independent matrix-resolvent D2 agrees with closed-form serial-chain D2 across m in {1,8,49} and phi in {0,0.75,0.9475}",
        "cases": 9,
        "tolerance": 1e-12,
        "observed_error": float(max_d2_rel),
        "pass": bool(max_d2_rel <= 1e-12),
    })

    # V5: Compiled C kernel vs pure-NumPy product kernel + periodic both-fixed exact loss (kappa*sigma)^2 / 2
    test_specs = [
        ls.BorreliaSpec(2.5, 6.0, 0.7, 0.4),
        ls.BorreliaSpec(3.0, 12.0, 0.7, 0.8, tick_shape="chain", tick_substages=6),
        ls.BorreliaSpec(2.0, 5.0, 0.7, 0.4, mouse_shape="fixed", tick_shape="fixed"),
        ls.BorreliaSpec(2.5, 6.0, 0.7, 0.4, sigma_weather=0.2, phi_weather=0.3, rho_hw=-0.4),
        ls.BorreliaSpec(2.5, 6.0, 0.7, 0.4, n_patches=4, patch_move=0.15),
        ls.BorreliaSpec(2.5, 8.0, 0.6, 0.5, contact_tick=True, driver_kind="bounded", bounded_low=0.2, tick_death=0.05),
    ]
    max_c_err = 0.0
    for idx_s, sp in enumerate(test_specs):
        mod = ls.IndependentCycleModel(sp).critical()
        mc = mod._sample_mod_centered(260, 4, 9000 + idx_s, antithetic=True)
        for y_test in (0.0, 0.17):
            l_np, s_np = mod.eval_products_numpy(mc, y_test, 40, 4)
            l_c, s_c = mod.eval_products_c(mc, y_test, 40, 4)
            max_c_err = max(max_c_err, float(np.max(np.abs(l_np - l_c))), float(np.max(np.abs(s_np - s_c))))

    # Periodic both-fixed exact loss = (kappa * sigma)^2 / 2 when burn and steps are multiples of period
    sp_per = ls.BorreliaSpec(4.0, 7.0, 0.75, 0.9, kappa1=0.8, mouse_shape="fixed", tick_shape="fixed")
    # Period = 11; choose burn = 2200, steps = 44000
    res_per = ls.IndependentCycleModel(sp_per).solve_loss(steps=44000, burn=2200, pairs=16, seed=31415)
    exact_per = 0.5 * ((0.8 * 0.75) ** 2)
    per_err = abs(res_per["D"] - exact_per)
    v5_err = max(max_c_err, per_err)
    checks.append({
        "id": "V5_c_kernel_and_periodic_exact_loss",
        "claim": "Compiled C product kernel matches pure-NumPy across 12 cases and both-fixed periodic cycle equals (kappa*sigma)^2/2",
        "cases": 13,
        "tolerance": 1e-12,
        "observed_error": float(v5_err),
        "pass": bool(v5_err <= 1e-12),
    })

    # V6: 1-stage lognormal D vs 80-node Gauss-Hermite quadrature + R12 coarse Borrelia benchmark (0.2013 +- 0.0002)
    # 1-stage: s_stay = 0.5, birth = 0.5 * exp(y) * h_t, sigma = 0.6, phi = 0.75
    lib = ls._get_clib()
    sig1, phi1 = 0.6, 0.75
    hosts1 = ls.sample_gaussian_ar1(np.array([sig1]), np.array([phi1]), np.ones((1, 1)), 42000, 24, 7777, antithetic=True)
    mc1 = np.ascontiguousarray(hosts1 - 1.0, dtype=np.float64)
    row1 = np.array([0], dtype=np.int32)
    col1 = np.array([0], dtype=np.int32)
    u1 = np.array([0.5], dtype=np.float64)
    f1 = np.array([0.5], dtype=np.float64)
    um1 = np.array([[0.0]], dtype=np.float64)
    fm1 = np.array([[0.5]], dtype=np.float64)
    x1 = np.array([1.0], dtype=np.float64)
    y_est = 0.09
    for _ in range(12):
        lg = np.zeros(48, dtype=np.float64)
        sl = np.zeros(48, dtype=np.float64)
        lib.eval_general_products(42000, 48, 1, 1, 1, 2000, row1.ctypes.data, col1.ctypes.data, u1.ctypes.data, f1.ctypes.data, um1.ctypes.data, fm1.ctypes.data, mc1.ctypes.data, x1.ctypes.data, float(y_est), lg.ctypes.data, sl.ctypes.data)
        lg_p = 0.5 * (lg[:24] + lg[24:])
        sl_p = 0.5 * (sl[:24] + sl[24:])
        y_est -= float(lg_p.mean() / sl_p.mean())
    se1 = float((y_est - lg_p / sl_p.mean()).std(ddof=1) / math.sqrt(24))

    gh_x, gh_w = np.polynomial.hermite.hermgauss(80)
    z_nodes = gh_x * math.sqrt(2.0)
    w_nodes = gh_w / math.sqrt(math.pi)
    lo, hi = 0.0, 0.5
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        val = float(w_nodes @ np.log(0.5 + 0.5 * np.exp(mid + sig1 * z_nodes - 0.5 * sig1 * sig1)))
        if val < 0.0:
            lo = mid
        else:
            hi = mid
    d1_exact = 0.5 * (lo + hi)
    tol_1s = max(3.0 * se1, 0.02 * d1_exact)
    ok_1s = abs(y_est - d1_exact) <= tol_1s

    # R12 benchmark: stays [1, 10/4.4], sigma = 1.0304, phi = 0 -> D = 0.2013 +- 0.0002, D2 = 0.16221
    sp_r12 = ls.BorreliaSpec(tau_mouse=1.0, tau_tick=10.0 / 4.4, sigma_mouse=1.0304, phi_mouse=0.0)
    res_r12 = ls.IndependentCycleModel(sp_r12).solve_loss(steps=40000, burn=2000, pairs=24, seed=8888)
    tol_r12 = 4.0 * math.hypot(res_r12["se"], 0.0002)
    ok_r12 = (abs(res_r12["D"] - 0.2013) <= tol_r12) and (abs(res_r12["D2"] - 0.162208) < 2e-5)
    checks.append({
        "id": "V6_1stage_quadrature_and_r12_benchmark",
        "claim": "1-stage D matches Gauss-Hermite quadrature within max(3 SE, 2%) and R12 Borrelia benchmark reproduces 0.2013 +- 0.0002",
        "cases": 2,
        "tolerance": f"1-stage <= {tol_1s:.5f}, R12 <= {tol_r12:.5f}",
        "observed_values": {
            "d_1stage_sim": y_est,
            "d_1stage_se": se1,
            "d_1stage_exact": d1_exact,
            "d_r12_sim": res_r12["D"],
            "d_r12_se": res_r12["se"],
            "d2_r12": res_r12["D2"],
        },
        "pass": bool(ok_1s and ok_r12),
    })

    # V7: Coupled-weather truncated-normal bin expectations vs adaptive quadrature and zero-noise D = 0
    cw_zero = ls.solve_coupled_weather_cell(
        sigma_mouse=0.0,
        kappa=0.0,
        gamma=0.0,
        site_climate=0.0,
        phi_mouse=0.0,
        phi_weather=0.0,
        rho_hw=0.0,
        weather_vals=[-0.5, 0.0, 0.5],
        duration_days=[300.0, 300.0, 300.0],
        steps=4000,
        burn=1000,
        pairs=4,
        seed=4444,
    )
    cw_harv = ls.solve_coupled_weather_cell(
        sigma_mouse=0.95,
        kappa=1.0,
        gamma=0.2,
        site_climate=0.4,
        phi_mouse=-0.1,
        phi_weather=0.1,
        rho_hw=-0.35,
        weather_vals=np.linspace(-1.2, 1.2, 10),
        duration_days=np.linspace(315.0, 335.0, 10),
        steps=4000,
        burn=1000,
        pairs=4,
        seed=4445,
    )
    a_shift = 0.95 * 1.0 * (-0.35)
    cuts = cw_harv["cuts"]
    max_q_err = 0.0
    for k in range(len(cw_harv["qh"])):
        q_num = integrate.quad(lambda z: math.exp(-0.5 * (z - a_shift) ** 2) / math.sqrt(2.0 * math.pi), cuts[k], cuts[k + 1], epsabs=1e-13)[0]
        max_q_err = max(max_q_err, abs(q_num - cw_harv["qh"][k]))
    v7_err = max(abs(cw_zero["D"]), max_q_err)
    checks.append({
        "id": "V7_coupled_weather_quadrature_and_zero_limit",
        "claim": "Coupled-weather truncated-normal bin weights match numerical quadrature and constant-limit loss equals 0",
        "cases": 11,
        "tolerance": 1e-12,
        "observed_error": float(v7_err),
        "pass": bool(v7_err <= 1e-12),
    })

    all_passed = all(c["pass"] for c in checks)
    summary = {"all_passed": all_passed, "n_checks": len(checks), "checks": checks}
    if save_json:
        out_path = HERE / "validation_checks.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not all_passed:
        failed = [c["id"] for c in checks if not c["pass"]]
        raise RuntimeError(f"Rule 7 validation failed: {failed}")
    return summary


if __name__ == "__main__":
    res = run_all_validations(save_json=True)
    for c in res["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['id']}: {c['claim']}")
    print(f"ALL {res['n_checks']} RULE 7 VALIDATION CHECKS PASSED.")
