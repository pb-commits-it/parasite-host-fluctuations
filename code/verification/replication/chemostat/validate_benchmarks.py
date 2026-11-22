#!/usr/bin/env python3
"""Rule 7 pre-analysis known-result verification for `r36-rep-chemostat`.

Executed BEFORE reading or parsing any empirical density trajectory in `data/raw/chemostat/`.
Every benchmark states its tolerance beforehand and verifies our independent routines in
`core_math.py` against exact analytical identities.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import core_math as cm


def current_date_utc() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def check_b1_discrete_c1_vs_matrix_resolvent() -> dict:
    """B1: Closed-form C1 vs exact matrix resolvent `C = c_tilde^2/2 - phi w^T B (I - phi M Q)^-1 Q B v`."""
    tol = 1e-12
    rng = np.random.default_rng(2026111501)
    max_rel_err = 0.0
    n_cases = 30
    for _ in range(n_cases):
        tau_p = float(rng.uniform(2.2, 30.0))
        tau_i = float(rng.uniform(1.8, 25.0))
        d = float(rng.uniform(0.02, 0.98))
        phi = float(rng.uniform(-0.75, 0.94))
        sigma = float(rng.uniform(0.05, 0.80))

        # Threshold discrete 2x2 matrix on [I, P]^T:
        # Stay in I: s_I = 1 - 1/tau_I, transfer I->P: g_I = beta * (1/tau_I)
        # Total exit from P: 1/tau_P = c + mu, where d = mu/(c + mu) => c = (1-d)/tau_P, mu = d/tau_P
        # At threshold R0 = (1 - d) * beta = 1 => beta = 1/(1 - d), so g_I * c = (1/tau_I)*(1/tau_P)
        s_i = 1.0 - 1.0 / tau_i
        g_i = 1.0 / ((1.0 - d) * tau_i)
        c_rate = (1.0 - d) / tau_p
        s_p = 1.0 - 1.0 / tau_p

        m_mat = np.array([[s_i, c_rate], [g_i, s_p]], dtype=float)
        # Perturbation B_tilde when h(t) = 1 + eps(t): column P (index 1) has +c_rate on row 0 and -c_rate on row 1
        b_tilde = np.array([[0.0, c_rate], [0.0, -c_rate]], dtype=float)

        # Exact right and left Perron eigenvectors at lambda = 1
        # m_mat @ v = v => (1/tau_i) v_0 = c_rate v_1 => v proportional to [c_rate * tau_i, 1] = [(1-d)*tau_i/tau_p, 1]
        # w^T @ m_mat = w^T => (1/tau_i) w_0 = g_i w_1 => w_0 = w_1 / (1-d)
        v = np.array([(1.0 - d) * tau_i / tau_p, 1.0], dtype=float)
        w = np.array([1.0 / (1.0 - d), 1.0], dtype=float)
        # Normalize so sum(v)=1 and w^T v = 1:
        v /= float(np.sum(v))
        w /= float(np.dot(w, v))

        q_proj = np.eye(2) - np.outer(v, w)
        c_scal = float(w @ b_tilde @ v)
        resolvent = np.linalg.solve(np.eye(2) - phi * (m_mat @ q_proj), q_proj @ b_tilde @ v)
        c_step = 0.5 * (c_scal ** 2) - phi * float(w @ b_tilde @ resolvent)
        gamma_resolvent = (tau_p + tau_i) * (sigma ** 2) * c_step
        gamma_c1 = cm.closed_form_c1_discrete(tau_p, tau_i, d, phi, sigma, m_i=1)

        rel_err = abs(gamma_resolvent - gamma_c1) / max(abs(gamma_c1), 1e-15)
        max_rel_err = max(max_rel_err, rel_err)

    return {
        "claim": "Discrete 2-stage contact-limited closed form C1 equals exact matrix-resolvent 2nd-order perturbation",
        "n_cases": n_cases,
        "pre_stated_tolerance": tol,
        "observed_max_rel_error": max_rel_err,
        "passed": bool(max_rel_err < tol),
    }


def check_b2_ct_identities_and_c2_limit() -> dict:
    """B2: Analytical 2x2 expm vs scipy expm, d=0 exact conservation, uniform washout shift, and small-noise C2."""
    rng = np.random.default_rng(2026111502)

    # (a) Analytical 2x2 scaled expm vs scipy.linalg.expm across 40 stiff/non-stiff trajectories
    tol_expm = 1e-12
    max_expm_diff = 0.0
    for _ in range(40):
        hosts = rng.uniform(1e5, 2e8, size=25)
        dts = rng.uniform(0.05, 2.0, size=25)
        wp = rng.uniform(0.1, 4.0, size=25)
        lh = cm.PhageLifeHistory("test", latent_period_d=0.03, burst_size=80.0, adsorption_ml_d=5e-6, virion_decay_d=0.2)
        r_ana = cm.floquet_lyapunov_ct(hosts, dts, wp, wp, lh, "contact", use_analytical_2x2=True)
        r_sci = cm.floquet_lyapunov_ct(hosts, dts, wp, wp, lh, "contact", use_analytical_2x2=False)
        max_expm_diff = max(max_expm_diff, abs(r_ana - r_sci) / max(abs(r_sci), 1.0))

    # (b) Zero waiting death (mu0 = 0, wash_p = 0 => d = 0) gives D_sim = 0 for arbitrary positive host trajectory
    tol_d0 = 1e-11
    hosts_d0 = rng.uniform(0.1, 5.0, size=80)
    dts_d0 = rng.uniform(0.05, 0.5, size=80)
    lh_d0 = cm.PhageLifeHistory("d0", latent_period_d=0.4, burst_size=40.0, adsorption_ml_d=1.1, virion_decay_d=0.0)
    res_d0 = cm.evaluate_trajectory_ct(hosts_d0, dts_d0, np.zeros(80), np.zeros(80), lh_d0, "contact")
    abs_d0 = abs(res_d0["D_sim"])

    # (c) Uniform continuous washout shifts Floquet growth rate by -bar(omega) pathwise
    tol_wash = 1e-12
    max_wash_err = 0.0
    for _ in range(30):
        hosts_w = rng.uniform(0.2, 3.0, size=35)
        dts_w = rng.uniform(0.05, 0.4, size=35)
        om_w = rng.uniform(0.0, 2.5, size=35)
        om_bar = float(np.sum(dts_w * om_w) / np.sum(dts_w))
        lh_w = cm.PhageLifeHistory("wash", latent_period_d=0.5, burst_size=35.0, adsorption_ml_d=0.9, virion_decay_d=0.25)
        r_w = cm.floquet_lyapunov_ct(hosts_w, dts_w, om_w, om_w, lh_w, "contact")
        r_0 = cm.floquet_lyapunov_ct(hosts_w, dts_w, np.zeros(35), np.zeros(35), lh_w, "contact")
        max_wash_err = max(max_wash_err, abs((r_0 - om_bar) - r_w))

    # (d) Small-amplitude sinusoidal continuous-time Floquet vs exact single-frequency Fourier/Laplace C2 limit:
    # For H(t) = 1 + eps * cos(omega_f t), autocovariance is (eps^2 / 2) cos(omega_f u) = Re[(eps^2 / 2) exp(i omega_f u)].
    # Replacing 1/theta_ct by -i*omega_f (or testing telegraph / AR(1) fine-step process):
    # Test small-amplitude antithetic telegraph host against closed-form C2!
    tol_c2 = 1.0e-2
    delta = 0.03
    theta_ct = 1.5
    lam_sw = 1.0 / (2.0 * theta_ct)
    dwells = rng.exponential(scale=1.0 / lam_sw, size=30000)
    signs = np.where(np.arange(30000) % 2 == 0, 1.0, -1.0)
    lh_tel = cm.PhageLifeHistory("tel", latent_period_d=2.0, burst_size=1.0 / 0.6, adsorption_ml_d=0.6, virion_decay_d=0.4)
    # Symmetrized (+delta / -delta) exact Floquet root
    def sym_tel_r(d_loss: float) -> float:
        r1 = cm.floquet_lyapunov_ct(1.0 + delta * signs, dwells, np.zeros(30000), np.zeros(30000), lh_tel, "contact", birth_log_tilt=d_loss)
        r2 = cm.floquet_lyapunov_ct(1.0 - delta * signs, dwells, np.zeros(30000), np.zeros(30000), lh_tel, "contact", birth_log_tilt=d_loss)
        return 0.5 * (r1 + r2)

    from scipy import optimize
    d_tel_num = float(optimize.brentq(sym_tel_r, -0.005, 0.005, xtol=1e-11))
    d_tel_c2 = cm.closed_form_c2_continuous(tau_p=1.0, tau_i=2.0, d=0.4, theta_ct=theta_ct, sigma=delta, m_i=1)
    rel_c2 = abs(d_tel_num - d_tel_c2) / abs(d_tel_c2)

    passed = bool(max_expm_diff < tol_expm and abs_d0 < tol_d0 and max_wash_err < tol_wash and rel_c2 < tol_c2)
    return {
        "claim": "Analytical 2x2 Floquet solver matches Padé expm, d=0 gives D=0, uniform washout shifts by -omega_bar, and telegraph matches C2",
        "expm_tol": tol_expm,
        "expm_max_rel_diff": max_expm_diff,
        "d0_tol": tol_d0,
        "d0_abs_D": abs_d0,
        "washout_tol": tol_wash,
        "washout_max_abs_err": max_wash_err,
        "telegraph_c2_tol": tol_c2,
        "telegraph_c2_rel_err": rel_c2,
        "passed": passed,
    }


def check_b3_proposition_b_and_dt_scaling() -> dict:
    """B3: Proposition B holds across 100 random fluctuating-staying trajectories while transfer-only fails."""
    rng = np.random.default_rng(2026111503)
    prop_b_viol = 0
    trans_only_viol = 0
    n_cases = 100
    lh = cm.PhageLifeHistory("pb", latent_period_d=1.25, burst_size=25.0, adsorption_ml_d=0.7, virion_decay_d=0.2)
    for idx in range(n_cases):
        z = rng.choice([-1.0, 1.0], size=60)
        hosts = np.ones(60) if idx < 35 else (1.0 - 0.45 * z)
        wp = 0.6 * (1.0 + 0.55 * z)
        wi = np.zeros(60)
        dts = np.full(60, 0.08)
        res = cm.verify_proposition_b_bound(hosts, dts, wp, wi, lh, max_step_exit_prob=0.25)
        if not res["holds_prop_b"]:
            prop_b_viol += 1
        if res["violates_transfer_only"]:
            trans_only_viol += 1

    # O(dt) scaling of (tau_0 - 1) * l_s as dt halves from 0.01 to 0.005
    r_grid = np.array([0.6, 1.8], dtype=float)
    def stay_pen(dt_v: float) -> float:
        s = np.exp(-r_grid * dt_v)
        sb = float(np.mean(s))
        return (1.0 / (1.0 - sb) - 1.0) * (-float(np.mean(np.log(s / sb))))

    ratio_half = stay_pen(0.01) / stay_pen(0.005)
    passed = bool(prop_b_viol == 0 and trans_only_viol >= 25 and abs(ratio_half - 2.0) < 0.02)
    return {
        "claim": "Proposition B bound has 0 violations across 100 fluctuating-staying schedules and O(dt) staying term halves within 2%",
        "n_cases": n_cases,
        "prop_b_violations": prop_b_viol,
        "transfer_only_violations": trans_only_viol,
        "dt_halving_ratio": ratio_half,
        "pre_stated_ratio_tol": 0.02,
        "passed": passed,
    }


def check_b4_rank_and_freedman_lane_identities() -> dict:
    """B4: Tied-rank Spearman, partial rank correlation formula, and exact 6! permutation symmetry."""
    tol = 1e-13
    # 1. Tied-rank analytical benchmark: x=[1,2,2,4], y=[1,2,3,4] -> rho = sqrt(0.9)
    rho_tied = cm.spearman_corr(np.array([1.0, 2.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    err_tied = abs(float(rho_tied) - math.sqrt(0.9))

    # 2. Constant vector returns None
    const_is_none = (cm.spearman_corr(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) is None)

    # 3. Single-covariate partial Spearman equals closed-form (r_yx - r_yz*r_xz)/sqrt((1-r_yz^2)*(1-r_xz^2))
    rng = np.random.default_rng(2026111504)
    max_part_err = 0.0
    for _ in range(25):
        y = rng.normal(size=18)
        x = rng.normal(size=18)
        z = rng.normal(size=18)
        fl = cm.freedman_lane_partial_spearman(y, x, z, seed=1, n_perm=100)
        ryx = float(cm.spearman_corr(y, x))
        ryz = float(cm.spearman_corr(y, z))
        rxz = float(cm.spearman_corr(x, z))
        closed = (ryx - ryz * rxz) / math.sqrt((1.0 - ryz ** 2) * (1.0 - rxz ** 2))
        max_part_err = max(max_part_err, abs(float(fl["rho_part"]) - closed))

    # 4. Exact 6! = 720 permutation sign-reversal symmetry: reversing x negates rho_part and swaps p_neg <-> p_pos
    y6 = np.array([10.0, 12.0, 11.5, 15.0, 14.2, 19.0])
    x6 = np.array([0.4, 0.2, 0.7, 0.1, 0.5, 0.3])
    z6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fl_pos = cm.freedman_lane_partial_spearman(y6, x6, z6)
    fl_neg = cm.freedman_lane_partial_spearman(y6, -x6, z6)
    sym_err = max(
        abs(float(fl_pos["rho_part"]) + float(fl_neg["rho_part"])),
        abs(float(fl_pos["p_neg"]) - float(fl_neg["p_pos"])),
        abs(float(fl_pos["p_pos"]) - float(fl_neg["p_neg"])),
    )

    passed = bool(err_tied < tol and const_is_none and max_part_err < tol and sym_err < tol)
    return {
        "claim": "Midrank Spearman, closed-form partial rank correlation, and exact 6! Freedman-Lane symmetry hold to < 1e-13",
        "pre_stated_tolerance": tol,
        "tied_spearman_error": err_tied,
        "constant_returns_none": const_is_none,
        "max_partial_formula_error": max_part_err,
        "exact_720_symmetry_error": sym_err,
        "passed": passed,
    }


def main() -> None:
    stamp = current_date_utc()
    b1 = check_b1_discrete_c1_vs_matrix_resolvent()
    b2 = check_b2_ct_identities_and_c2_limit()
    b3 = check_b3_proposition_b_and_dt_scaling()
    b4 = check_b4_rank_and_freedman_lane_identities()

    all_passed = all(x["passed"] for x in (b1, b2, b3, b4))
    out = {
        "verified_utc_from_date": stamp,
        "empirics_opened_before_check": False,
        "all_passed": all_passed,
        "benchmarks": {
            "B1_discrete_C1_vs_resolvent": b1,
            "B2_continuous_Floquet_and_C2": b2,
            "B3_proposition_B_and_dt_scaling": b3,
            "B4_ranks_and_freedman_lane": b4,
        },
    }
    out_path = HERE / "validation_benchmarks.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
