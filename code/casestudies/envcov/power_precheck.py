#!/usr/bin/env python3
"""Pre-analysis power and identifiability calculation for case study envcov.

Written and executed BEFORE fetching or inspecting any environmental covariate values
(Rule 1 of CASESTUDY_SPEC.md). Uses only sample sizes, host-variability summaries, and
nymph test counts already published in reports R12, R12b, and R14.

Run:
    python3 -I casestudies/envcov/power_precheck.py
"""
from __future__ import annotations

import datetime
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
RESULTS = HERE / "results"

# Published R14 six-site summaries (from results/r14-posthoc-power_20261114.txt)
SITES = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]
SIGMA_R14 = np.array([0.399, 0.989, 0.505, 0.564, 0.571, 0.482], dtype=float)
NTEST_R14 = np.array([360, 1004, 178, 844, 662, 1446], dtype=int)
POBS_MEAN = float(np.mean([0.1861, 0.2291, 0.0618, 0.1481, 0.2523, 0.2275]))
OBS_RHO = 0.4857
D_KAPPA = {
    0.0: np.zeros(6),
    0.5: np.array([0.0062, 0.0403, 0.0099, 0.0124, 0.0128, 0.0090]),
    1.0: np.array([0.0256, 0.1837, 0.0419, 0.0531, 0.0544, 0.0380]),
}
CUT_RHO = -29.0 / 35.0  # exact 1-sided p = 21/720 = 0.0292 < 0.05 for n = 6

PERMS6 = np.array(list(itertools.permutations(range(6))), dtype=np.int64)


def spearman_batch(x_vec: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
    """Vectorized Spearman rank correlation between 1D x_vec (n,) and 2D y_mat (B, n)."""
    rx = np.argsort(np.argsort(x_vec)).astype(float)
    rx -= rx.mean()
    rx_norm = math.sqrt(float((rx * rx).sum()))
    ry = np.argsort(np.argsort(y_mat, axis=1), axis=1).astype(float)
    ry -= ry.mean(axis=1, keepdims=True)
    ry_norm = np.sqrt((ry * ry).sum(axis=1))
    return (ry @ rx) / (rx_norm * ry_norm)


def partial_corr_1cov(y_mat: np.ndarray, x_vec: np.ndarray, z_mat: np.ndarray) -> np.ndarray:
    """Batch partial correlation r(Y, X | Z) for Y (B, n), X (n,), Z (B, n)."""
    xc = x_vec - x_vec.mean()
    xn = xc / math.sqrt(float((xc * xc).sum()))
    yc = y_mat - y_mat.mean(axis=1, keepdims=True)
    yn = yc / np.maximum(np.sqrt((yc * yc).sum(axis=1, keepdims=True)), 1e-15)
    zc = z_mat - z_mat.mean(axis=1, keepdims=True)
    zn = zc / np.maximum(np.sqrt((zc * zc).sum(axis=1, keepdims=True)), 1e-15)
    ryx = yn @ xn
    ryz = (yn * zn).sum(axis=1)
    rxz = zn @ xn
    denom = np.sqrt(np.maximum((1.0 - ryz * ryz) * (1.0 - rxz * rxz), 1e-15))
    return (ryx - ryz * rxz) / denom


def partial_fl_perm_p(
    y_mat: np.ndarray, x_vec: np.ndarray, z_mat: np.ndarray, use_ranks: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 720-permutation 1-sided (negative) Freedman-Lane p-value for r(Y, X | Z).

    Note (from attempt 1): permuting Z-orthogonalized residuals ey alters their
    component along u_z = Z_centered / ||Z_centered||, so re-residualizing each
    permuted ey on [1, Z] divides the numerator (ey[perm] . ex) by
    sqrt(1 - (ey[perm] . u_z)^2), restoring the exact df = n - 3 null calibration.
    """
    if use_ranks:
        rx = np.argsort(np.argsort(x_vec)).astype(float)
        ry = np.argsort(np.argsort(y_mat, axis=1), axis=1).astype(float)
        rz = np.argsort(np.argsort(z_mat, axis=1), axis=1).astype(float)
    else:
        rx = np.asarray(x_vec, dtype=float).copy()
        ry = np.asarray(y_mat, dtype=float).copy()
        rz = np.asarray(z_mat, dtype=float).copy()
    rx -= rx.mean()
    ry -= ry.mean(axis=1, keepdims=True)
    rz -= rz.mean(axis=1, keepdims=True)
    rz_norm = np.maximum(np.sqrt((rz * rz).sum(axis=1, keepdims=True)), 1e-15)
    uz = rz / rz_norm

    ex = rx[None, :] - (uz @ rx)[:, None] * uz
    ex /= np.maximum(np.sqrt((ex * ex).sum(axis=1, keepdims=True)), 1e-15)

    ey = ry - (ry * uz).sum(axis=1, keepdims=True) * uz
    ey /= np.maximum(np.sqrt((ey * ey).sum(axis=1, keepdims=True)), 1e-15)

    obs_r = (ex * ey).sum(axis=1)
    pvals = np.empty(len(y_mat), dtype=float)
    chunk = 2500
    for i in range(0, len(y_mat), chunk):
        ey_c = ey[i : i + chunk]
        ex_c = ex[i : i + chunk]
        uz_c = uz[i : i + chunk]
        obs_c = obs_r[i : i + chunk, None]
        ey_perm = ey_c[:, PERMS6]
        num = np.einsum("bpn,bn->bp", ey_perm, ex_c)
        proj_z = np.einsum("bpn,bn->bp", ey_perm, uz_c)
        den = np.sqrt(np.maximum(1.0 - proj_z * proj_z, 1e-15))
        perm_r = num / den
        pvals[i : i + chunk] = np.mean(perm_r <= obs_c + 1e-12, axis=1)
    return obs_r, pvals


def solve_m_for_mean_prev(d_vec: np.ndarray, f: float = 0.5) -> float:
    lo, hi = 0.0, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pm = float(np.mean(1.0 - np.exp(-f * np.maximum(mid - d_vec, 0.0))))
        if pm < POBS_MEAN:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def code_checks() -> None:
    print("0. Pre-check code verification (Rule 7; tolerances fixed beforehand)")
    # P0a: exact cutoff and R14 posthoc power reproduction (tol = 0.004 on 200,000 MC)
    base = np.arange(6.0)
    s_vals = np.array([sum((a - b) ** 2 for a, b in zip(base, p)) for p in PERMS6])
    rho_vals = 1.0 - s_vals / 35.0
    n_cut = int(np.sum(rho_vals <= CUT_RHO + 1e-12))
    assert n_cut == 21, f"Expected 21 orderings, got {n_cut}"
    rng = np.random.default_rng(20261114)
    for kappa, ref_sup, ref_obs in [(0.0, 0.029, 0.177), (1.0, 0.068, 0.070)]:
        m = solve_m_for_mean_prev(D_KAPPA[kappa])
        ln_r = m + 0.17 * rng.standard_normal((200_000, 6)) - D_KAPPA[kappa]
        p = 1.0 - np.exp(-0.5 * np.maximum(ln_r, 0.0))
        k = rng.binomial(NTEST_R14, p)
        rho = spearman_batch(SIGMA_R14, k / NTEST_R14 + 1e-9 * rng.standard_normal(k.shape))
        p_sup = float(np.mean(rho <= CUT_RHO + 1e-9))
        p_obs = float(np.mean(rho >= OBS_RHO - 1e-9))
        ok = abs(p_sup - ref_sup) < 0.004 and abs(p_obs - ref_obs) < 0.004
        print(f"  P0a kappa={kappa:.1f}: P(sup)={p_sup:.4f} (ref {ref_sup:.3f}), P(rho>=+0.4857)={p_obs:.4f} (ref {ref_obs:.3f}) [{'ok' if ok else 'FAIL'}]")
        assert ok

    # P0b: partial correlation vs explicit OLS residuals (tol = 1e-12)
    rng2 = np.random.default_rng(42)
    x = rng2.normal(size=6)
    y = rng2.normal(size=(5, 6))
    z = rng2.normal(size=(5, 6))
    pc = partial_corr_1cov(y, x, z)
    for b in range(5):
        design = np.column_stack([np.ones(6), z[b]])
        rx_ols = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry_ols = y[b] - design @ np.linalg.lstsq(design, y[b], rcond=None)[0]
        ref = float(np.corrcoef(rx_ols, ry_ols)[0, 1])
        assert abs(pc[b] - ref) < 1e-12
    print("  P0b partial correlation matches OLS residual correlation to < 1e-12 [ok]")

    # P0c: null size of 1-covariate Freedman-Lane permutation test (tol: <= 0.055)
    rng3 = np.random.default_rng(99)
    y0 = rng3.normal(size=(20_000, 6))
    z0 = 0.6 * ((SIGMA_R14 - SIGMA_R14.mean()) / SIGMA_R14.std())[None, :] + 0.8 * rng3.normal(size=(20_000, 6))
    _, p0_lin = partial_fl_perm_p(y0 + 0.5 * z0, SIGMA_R14, z0, use_ranks=False)
    _, p0_rnk = partial_fl_perm_p(y0 + 0.5 * z0, SIGMA_R14, z0, use_ranks=True)
    null_lin = float(np.mean(p0_lin < 0.05))
    null_rnk = float(np.mean(p0_rnk < 0.05))
    ok_c = (0.020 <= null_lin <= 0.055) and (0.020 <= null_rnk <= 0.055)
    print(
        f"  P0c null Type I error of 1-sided Freedman-Lane test (n=6, r_xz=0.6): "
        f"linear={null_lin:.4f}, rank={null_rnk:.4f} [{'ok' if ok_c else 'FAIL'}]"
    )
    assert ok_c


def neon_identifiability_and_power() -> dict:
    print("\n1. Identifiability limits with N = 6 sites (NEON across-site comparison)")
    print("  Covariates adjusted (p) | Parameters (intercept + sigma + p) | Residual df = 6 - 2 - p | Identifiable?")
    id_rows = []
    for p in range(5):
        df = 6 - 2 - p
        status = "yes (unadjusted)" if p == 0 else ("yes, minimal (3 df)" if p == 1 else ("fragile (1-2 df)" if df > 0 else "NO: saturated / overfitted"))
        print(f"  {p:23d} | {p + 2:34d} | {df:23d} | {status}")
        id_rows.append({"p": p, "df": df, "status": status})

    print("\n  Variance inflation factor VIF = 1/(1 - r_xz^2) and effective SE multiplier on 3 df vs 4 df:")
    for rxz in (0.0, 0.3, 0.5, 0.7, 0.85):
        vif = 1.0 / (1.0 - rxz * rxz)
        print(f"    |r(sigma, cov)| = {rxz:.2f} -> VIF = {vif:.2f}, sqrt(VIF) = {math.sqrt(vif):.2f}")

    print("\n2. Simulated power and confounding rates across N = 6 NEON sites (50,000 replicates/cell)")
    rng = np.random.default_rng(2026111409)
    n_mc = 50_000
    xs = (SIGMA_R14 - SIGMA_R14.mean()) / SIGMA_R14.std(ddof=1)
    header = (
        f"  {'kappa':>5s} {'r_ZS':>5s} {'beta_Z':>6s} {'tau_res':>7s} | "
        f"{'mean_rho':>8s} {'P(unadj sup)':>12s} {'P(rho>=.486)':>12s} | "
        f"{'mean_pr':>8s} {'P(adj1 sup)':>11s}"
    )
    print(header)
    scenarios = [
        # Baseline no-confounder checks
        (0.0, 0.0, 0.00, 0.17),
        (0.5, 0.0, 0.00, 0.17),
        (1.0, 0.0, 0.00, 0.17),
        # Confounder positively correlated with sigma and raising prevalence (masks negative D, mimics +0.49!)
        (0.0, +0.6, +0.20, 0.10),
        (0.5, +0.6, +0.20, 0.10),
        (1.0, +0.6, +0.20, 0.10),
        (1.0, +0.6, +0.25, 0.10),
        # Confounder negatively correlated with sigma and raising prevalence (mimics theory when kappa=0!)
        (0.0, -0.6, +0.20, 0.10),
        (1.0, -0.6, +0.20, 0.10),
        # Strong confounder explaining nearly all inter-site variance (best possible case for adjustment)
        (1.0, +0.6, +0.20, 0.05),
        (1.0, 0.0, +0.20, 0.05),
    ]
    neon_out = []
    for kappa, r_zs, beta_z, tau_res in scenarios:
        m = solve_m_for_mean_prev(D_KAPPA[kappa])
        eps_z = rng.standard_normal((n_mc, 6))
        # Construct site covariate Z with expected correlation r_zs with standardized sigma
        z = r_zs * xs[None, :] + math.sqrt(max(0.0, 1.0 - r_zs * r_zs)) * eps_z
        ln_r = m + beta_z * z + tau_res * rng.standard_normal((n_mc, 6)) - D_KAPPA[kappa][None, :]
        p = 1.0 - np.exp(-0.5 * np.maximum(ln_r, 0.0))
        k = rng.binomial(NTEST_R14, p)
        p_hat = k / NTEST_R14 + 1e-9 * rng.standard_normal(k.shape)
        rho = spearman_batch(SIGMA_R14, p_hat)
        pr, p_adj = partial_fl_perm_p(p_hat, SIGMA_R14, z, use_ranks=True)
        row = {
            "kappa": kappa,
            "r_zs": r_zs,
            "beta_z": beta_z,
            "tau_res": tau_res,
            "mean_rho": float(rho.mean()),
            "p_unadj_sup": float(np.mean(rho <= CUT_RHO + 1e-9)),
            "p_unadj_pos49": float(np.mean(rho >= OBS_RHO - 1e-9)),
            "mean_partial_r": float(pr.mean()),
            "p_adj1_sup": float(np.mean(p_adj < 0.05)),
        }
        neon_out.append(row)
        print(
            f"  {kappa:5.2f} {r_zs:+5.2f} {beta_z:+6.2f} {tau_res:7.2f} | "
            f"{row['mean_rho']:+8.3f} {row['p_unadj_sup']:12.3f} {row['p_unadj_pos49']:12.3f} | "
            f"{row['mean_partial_r']:+8.3f} {row['p_adj1_sup']:11.3f}"
        )

    # Familywise false-positive rate when screening 6 independent candidate covariates under kappa=0
    y_null = solve_m_for_mean_prev(D_KAPPA[0.0]) + 0.17 * rng.standard_normal((20_000, 6))
    p_null = 1.0 - np.exp(-0.5 * np.maximum(y_null, 0.0))
    k_null = rng.binomial(NTEST_R14, p_null) / NTEST_R14 + 1e-9 * rng.standard_normal((20_000, 6))
    any_hit = np.zeros(20_000, dtype=bool)
    for _ in range(6):
        z_cand = rng.standard_normal((20_000, 6))
        _, p_c = partial_fl_perm_p(k_null, SIGMA_R14, z_cand, use_ranks=True)
        any_hit |= p_c < 0.05
    fwer6 = float(np.mean(any_hit))
    print(f"\n  Familywise false-support rate if screening 6 unrelated covariates at alpha=0.05 (kappa=0): {fwer6:.3f}")
    return {"identifiability": id_rows, "scenarios": neon_out, "fwer_6_covariates": fwer6}


def cary_identifiability_and_power() -> dict:
    print("\n3. Simulated bias, SE, and power for Cary within-plot annual slope (18 mouse-years x 6 grids, 20,000 replicates/cell)")
    print("   Note: weather at Cary is landscape-wide (1 value per year across all 6 grids), so weather")
    print("   confounding acts on the 18 annual degrees of freedom, not 102 independent grid-years.")
    rng = np.random.default_rng(2026111419)
    n_mc = 20_000
    n_years = 18
    n_grids = 6
    n_nymphs = 53  # R12 median tested per plot-year
    pbar = 0.413   # R12 mean NIP

    header = (
        f"  {'b_true':>6s} {'r_WM':>5s} {'gam_W':>6s} | "
        f"{'mean_b_un':>9s} {'SD_un':>6s} {'P(b_un>2SE)':>11s} {'P(P2 hold)':>10s} | "
        f"{'mean_b_a1':>9s} {'SD_a1':>6s} {'P(b_a1>2SE)':>11s} | {'SD_a6':>6s}"
    )
    print(header)
    cary_scenarios = [
        (0.00, 0.0, 0.00),
        (0.04, 0.0, 0.00),
        (0.15, 0.0, 0.00),
        (0.00, +0.5, +0.20),  # weather mimics positive mouse slope when b_true = 0
        (0.00, -0.5, +0.20),  # weather induces negative mouse slope when b_true = 0
        (0.15, -0.5, +0.20),  # weather masks true positive mouse slope (pushes b_unadj near 0.04!)
        (0.15, +0.5, -0.20),  # hot/dry weather covarying with mice masks mouse slope
    ]
    cary_out = []
    for b_true, r_wm, gam_w in cary_scenarios:
        # Annual landscape weather W_t (n_mc, n_years) and landscape mouse M_t
        w_yr = rng.standard_normal((n_mc, n_years))
        m_yr = r_wm * w_yr + math.sqrt(max(0.0, 1.0 - r_wm * r_wm)) * rng.standard_normal((n_mc, n_years))
        # Grid-level log mice: landscape sd ~ 1.00, within-year plot noise sd = 0.25 -> total sd ~ 1.03 (matches R12!)
        x = m_yr[:, :, None] + 0.25 * rng.standard_normal((n_mc, n_years, n_grids))
        # Log prevalence at t+1 around ln(pbar)
        yr_shock = 0.15 * rng.standard_normal((n_mc, n_years, 1))
        ln_p = math.log(pbar) + b_true * (x - x.mean(axis=(1, 2), keepdims=True)) + gam_w * w_yr[:, :, None] + yr_shock
        p = np.clip(np.exp(ln_p), 0.02, 0.95)
        k = rng.binomial(n_nymphs, p)
        y = np.log(np.maximum(k, 0.5) / n_nymphs)

        # Within-plot demeaned x and y
        xd = x - x.mean(axis=1, keepdims=True)
        yd = y - y.mean(axis=1, keepdims=True)
        b_un = (xd * yd).sum(axis=(1, 2)) / (xd * xd).sum(axis=(1, 2))

        # Analytical cluster-by-year SE approximation via year-level score (fast & exact for balanced panel)
        # Adjust for 1 weather covariate W_t (demeaned across years)
        wd = w_yr - w_yr.mean(axis=1, keepdims=True)
        # Project xd and yd orthogonal to wd[:,:,None]
        w_ss = (wd * wd).sum(axis=1, keepdims=True) * n_grids
        coef_xw = (xd * wd[:, :, None]).sum(axis=(1, 2), keepdims=True) / w_ss[:, :, None]
        coef_yw = (yd * wd[:, :, None]).sum(axis=(1, 2), keepdims=True) / w_ss[:, :, None]
        xd_a1 = xd - coef_xw * wd[:, :, None]
        yd_a1 = yd - coef_yw * wd[:, :, None]
        b_a1 = (xd_a1 * yd_a1).sum(axis=(1, 2)) / (xd_a1 * xd_a1).sum(axis=(1, 2))

        # Simultaneous 6-covariate adjustment (w_yr plus 5 noise weather series)
        w5 = rng.standard_normal((n_mc, n_years, 5))
        w6 = np.concatenate([w_yr[:, :, None], w5], axis=2)
        w6 -= w6.mean(axis=1, keepdims=True)
        # Batched projection of year-means; grid deviations from year-mean are untouched by landscape covariates!
        x_ym = xd.mean(axis=2)  # (n_mc, n_years)
        y_ym = yd.mean(axis=2)
        b_a6 = np.empty(n_mc)
        se_un = np.empty(n_mc)
        se_a1 = np.empty(n_mc)
        # Compute cluster-robust year-level jackknife SE for unadjusted and 1-cov adjusted, and b_a6
        x_within_yr = xd - x_ym[:, :, None]
        y_within_yr = yd - y_ym[:, :, None]
        ss_x_within = (x_within_yr * x_within_yr).sum(axis=(1, 2))
        xy_within = (x_within_yr * y_within_yr).sum(axis=(1, 2))
        for i in range(n_mc):
            # 6-cov projection on year-means
            q, _ = np.linalg.qr(w6[i], mode="reduced")
            rx_ym = x_ym[i] - q @ (q.T @ x_ym[i])
            ry_ym = y_ym[i] - q @ (q.T @ y_ym[i])
            b_a6[i] = (xy_within[i] + n_grids * float(rx_ym @ ry_ym)) / (ss_x_within[i] + n_grids * float(rx_ym @ rx_ym))

        # Cluster-by-year SEs (Liang-Zeger / year-score variance)
        res_un = yd - b_un[:, None, None] * xd
        score_yr_un = (xd * res_un).sum(axis=2)  # (n_mc, n_years)
        sxx_un = (xd * xd).sum(axis=(1, 2))
        se_un = np.sqrt((n_years / (n_years - 1)) * (score_yr_un * score_yr_un).sum(axis=1)) / sxx_un

        res_a1 = yd_a1 - b_a1[:, None, None] * xd_a1
        score_yr_a1 = (xd_a1 * res_a1).sum(axis=2)
        sxx_a1 = (xd_a1 * xd_a1).sum(axis=(1, 2))
        se_a1 = np.sqrt((n_years / (n_years - 2)) * (score_yr_a1 * score_yr_a1).sum(axis=1)) / sxx_a1

        upper = 1.0 - pbar
        p2_hold = np.mean((b_un >= -2.0 * se_un) & (b_un <= upper + 2.0 * se_un))
        row = {
            "b_true": b_true,
            "r_wm": r_wm,
            "gamma_w": gam_w,
            "mean_b_unadj": float(b_un.mean()),
            "sd_b_unadj": float(b_un.std(ddof=1)),
            "power_unadj_gt0": float(np.mean(b_un - 2.0 * se_un > 0.0)),
            "p2_hold_rate": float(p2_hold),
            "mean_b_adj1": float(b_a1.mean()),
            "sd_b_adj1": float(b_a1.std(ddof=1)),
            "power_adj1_gt0": float(np.mean(b_a1 - 2.0 * se_a1 > 0.0)),
            "sd_b_adj6": float(b_a6.std(ddof=1)),
        }
        cary_out.append(row)
        print(
            f"  {b_true:6.2f} {r_wm:+5.2f} {gam_w:+6.2f} | "
            f"{row['mean_b_unadj']:+9.4f} {row['sd_b_unadj']:6.4f} {row['power_unadj_gt0']:11.3f} {row['p2_hold_rate']:10.3f} | "
            f"{row['mean_b_adj1']:+9.4f} {row['sd_b_adj1']:6.4f} {row['power_adj1_gt0']:11.3f} | {row['sd_b_adj6']:6.4f}"
        )
    return {"scenarios": cary_out}


def main() -> int:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"envcov pre-analysis power and identifiability check. Started {stamp}.")
    print("No environmental covariate files have been fetched or read.")
    code_checks()
    neon_res = neon_identifiability_and_power()
    cary_res = cary_identifiability_and_power()
    out_json = RESULTS / "power_precheck_20261114.json"
    out_json.write_text(json.dumps({"timestamp": stamp, "neon": neon_res, "cary": cary_res}, indent=2))
    print(f"\nSaved JSON summary to {out_json.relative_to(HERE.parent.parent.parent)}")
    print(f"Completed in {time.time() - t0:.1f} s. ALL PRE-CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
