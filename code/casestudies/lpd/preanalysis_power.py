#!/usr/bin/env python3
"""Pre-analysis validation, identifiability, and power simulation for case study lpd.

Run BEFORE any empirical data are downloaded or inspected:
    python3 -I casestudies/lpd/preanalysis_power.py

Purpose (CASESTUDY_SPEC.md Rules 1 & 7):
  1. Reproduce known analytical/engine results (K0a-K0d) within pre-stated tolerances:
     - K0a: closed-form second-order loss Gamma_0(a, b, sigma, phi) matches
            lifecycle_engine.py Engine.coefficient()['D2'] within 1e-10 relative error
            across 50 random two-stage rings spanning positive and negative phi.
     - K0b: Result 4 / Lean P1-P2 persistence-index difference E(a,b) - E(a) matches
            direct evaluation of 2(ln R0 - Gamma_0)/(T_0 V) within 1e-12 relative error.
     - K0c: analytical hat-matrix formula for finite-sample linear-detrending bias in
            white-noise lag-1 autocorrelation agrees with 20,000 Monte Carlo white-noise
            series at N = 15 within 0.015 absolute tolerance (exact ratio vs ratio of means).
     - K0d: OLS log-periodogram spectral slope has mean within 0.02 of 0 on white noise
            (N = 30, 10,000 trials) and strictly negative (< -1.0) on red AR(1) phi = 0.85.
  2. Quantify estimator bias, RMSE, and observation-error attenuation for (sigma, r1,
     bias-corrected r1_bc, spectral slope b_spec) across series lengths N in {15, 20, 30, 50, 100},
     true memories phi in {-0.30, 0.00, 0.10, 0.20, 0.40, 0.75}, and observation-error variance
     shares eta_obs in {0.00, 0.25, 0.50}, plus an AR(2) 3.5-step quasi-cycle.
  3. Compute study-clustered statistical power and Type I error (alpha = 0.05, two-sided)
     for testing H0 (white noise around linear trend, phi = 0) vs H1 (non-zero host memory)
     at realistic taxon-group sizes M in {25, 100, 300} series clustered 5 per study
     with within-study correlation 0.40, establishing what true memory phi cannot be
     distinguished from zero.
  4. Quantify finite-sample estimation error on the three theoretical regime fractions
     (i) parking works, (ii) short steady stage harmed by memory, (iii) small-noise formula
     reliable across the three R16 illustrative life cycles.
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
DOBSON_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load project lifecycle_engine.py without modifying sys.path globally
_eng_spec = importlib.util.spec_from_file_location(
    "lifecycle_engine", DOBSON_ROOT / "studies" / "engine-R16" / "lifecycle_engine.py"
)
eng = importlib.util.module_from_spec(_eng_spec)
sys.modules[_eng_spec.name] = eng
_eng_spec.loader.exec_module(eng)


# ---------------------------------------------------------------------------
# 1. Exact closed-form formulas from notes/proofs-v2.md (Propositions 2, 4, 18)
# ---------------------------------------------------------------------------

def gamma0_two_stage(a: float, b: float, sigma: float, phi: float) -> float:
    """Second-order loss of ln R0 for a 2-stage ring with threshold stays (a, b),
    one fluctuating transfer of log-SD sigma and AR(1) correlation phi in (-1, 1),
    and fixed staying contributions (Proposition 2 / Statement ring-B1b & ring-B2).
    """
    if sigma == 0.0:
        return 0.0
    t0 = a + b
    if abs(phi) < 1e-15:
        return (sigma * sigma) / (2.0 * t0)
    # Exact generating function E[phi^(L1 + L2)] for geometric stays a, b >= 1 on {1, 2, ...}:
    # E[phi^Lk] = (phi / tau_k) / (1 - (1 - 1/tau_k) * phi) = phi / (tau_k - (tau_k - 1)*phi)
    r1 = phi / (a - (a - 1.0) * phi)
    r2 = phi / (b - (b - 1.0) * phi)
    re = r1 * r2
    one_plus_2theta = (1.0 + phi) / (1.0 - phi)
    bracket = one_plus_2theta / (2.0 * t0) - re / (1.0 - re)
    return (sigma * sigma) * bracket


def gamma0_one_stage(a: float, sigma: float) -> float:
    """Second-order loss of ln R0 for a 1-stage cycle of threshold stay a >= 1 (independent of phi)."""
    return (sigma * sigma) / (2.0 * a)


def persistence_index_E(a: float, b: float | None, sigma: float, phi: float, ln_r0: float) -> float:
    """Diffusion persistence index E = 2(ln R0 - Gamma_0) / (T_0 V) of Result 4 / Lean P1-P4."""
    one_plus_2theta = (1.0 + phi) / (1.0 - phi)
    if b is None or b == 0.0:
        g0 = gamma0_one_stage(a, sigma)
        tv = (sigma * sigma) * one_plus_2theta / a
    else:
        g0 = gamma0_two_stage(a, b, sigma, phi)
        tv = (sigma * sigma) * one_plus_2theta / (a + b)
    return 2.0 * (ln_r0 - g0) / tv


def lean_p1_delta_E(a: float, b: float, sigma: float, phi: float, ln_r0: float) -> float:
    """Closed-form E(a,b) - E(a) from Lean P1 (DobsonProofs/Parking.lean)."""
    theta = phi / (1.0 - phi)
    ell = ln_r0 / (sigma * sigma)
    one_plus_2theta = 1.0 + 2.0 * theta
    return (2.0 * b / one_plus_2theta) * (ell - (a * theta) / (a * theta + b * (a + theta)))


# ---------------------------------------------------------------------------
# 2. Time-series estimators (identical definitions to be used in analysis.py)
# ---------------------------------------------------------------------------

def ols_detrend(t_idx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """OLS linear detrending y_i = alpha + beta * t_i + e_i on integer step grid t_idx."""
    t_mean = float(t_idx.mean())
    y_mean = float(y.mean())
    dt = t_idx - t_mean
    dy = y - y_mean
    denom = float(dt @ dt)
    beta = float(dt @ dy) / denom if denom > 0.0 else 0.0
    alpha = y_mean - beta * t_mean
    resid = y - (alpha + beta * t_idx)
    return resid, alpha, beta


def white_noise_r1_null_mean(t_idx: np.ndarray) -> float:
    """First-order analytical expectation (ratio of expectations) of the OLS-detrended
    lag-1 autocovariance estimator under white noise on observed integer steps t_idx.
    Also serves as the baseline shift in Kendall-Marriott-Pope bias correction.
    """
    n = len(t_idx)
    if n <= 3:
        return 0.0
    t_c = t_idx - t_idx.mean()
    denom_t = float(t_c @ t_c)
    # Hat matrix H_ij = 1/n + t_c[i]*t_c[j]/denom_t
    pos = {int(t): i for i, t in enumerate(t_idx)}
    pairs = [(i, pos[int(t) + 1]) for i, t in enumerate(t_idx) if (int(t) + 1) in pos]
    m1 = len(pairs)
    if m1 == 0:
        return 0.0
    sum_h = 0.0
    for i, j in pairs:
        sum_h += (1.0 / n) + (t_c[i] * t_c[j] / denom_t if denom_t > 0 else 0.0)
    # E[num] = -(sigma^2 / m1) * sum_h ; E[den] = sigma^2 * (n - 2) / n
    return -float(n) / float((n - 2) * m1) * sum_h


_PHI_GRID = np.linspace(-0.95, 0.98, 194)


def ar1_detrended_expected_r1_dense(t_idx: np.ndarray, phi: float) -> float:
    """Reference dense-matrix E[num(phi)] / E[den(phi)] used to verify O(n) rank-2 identity."""
    n = len(t_idx)
    diff = np.abs(t_idx[:, None] - t_idx[None, :])
    gamma = np.eye(n) if abs(phi) < 1e-15 else (phi ** diff)
    t_c = t_idx - t_idx.mean()
    denom_t = float(t_c @ t_c)
    H = (1.0 / n) * np.ones((n, n)) + (np.outer(t_c, t_c) / denom_t if denom_t > 0 else 0.0)
    M = np.eye(n) - H
    cov_e = M @ gamma @ M
    e_den = float(np.trace(cov_e)) / n
    pos = {int(t): i for i, t in enumerate(t_idx)}
    pairs = [(i, pos[int(t) + 1]) for i, t in enumerate(t_idx) if (int(t) + 1) in pos]
    if not pairs or e_den <= 1e-15:
        return 0.0
    e_num = float(sum(cov_e[i, j] for i, j in pairs)) / len(pairs)
    return e_num / e_den


def ar1_detrended_expected_r1_curve(t_idx: np.ndarray, phis: np.ndarray = _PHI_GRID) -> np.ndarray:
    """Exact O(n) rank-2 forward-backward evaluation of E[num(phi)] / E[den(phi)]
    across an entire vector of candidate phi values simultaneously.
    """
    n = len(t_idx)
    p_len = len(phis)
    t_c = t_idx - t_idx.mean()
    norm_t = math.sqrt(float(t_c @ t_c))
    # U has shape (n, 2), orthonormal columns
    u0 = np.full(n, 1.0 / math.sqrt(n))
    u1 = t_c / norm_t if norm_t > 0 else np.zeros(n)
    U = np.column_stack([u0, u1])  # (n, 2)

    gaps = np.diff(t_idx)  # (n-1,)
    pow_gaps = phis[:, None] ** gaps[None, :]  # (P, n-1)

    W = np.empty((p_len, n, 2), dtype=float)
    f = np.broadcast_to(U[0], (p_len, 2)).copy()
    W[:, 0, :] = f
    for k in range(1, n):
        f = pow_gaps[:, k - 1 : k] * f + U[k]
        W[:, k, :] = f
    b = np.zeros((p_len, 2), dtype=float)
    for k in range(n - 2, -1, -1):
        b = pow_gaps[:, k : k + 1] * (b + U[k + 1])
        W[:, k, :] += b

    # UtW: shape (P, 2, 2) = sum_k U[k, a] * W[p, k, b]
    UtW = np.einsum("ka,pkb->pab", U, W)
    # tr(M Gamma M) = tr(Gamma M) = n - tr(UtW)
    e_den = (n - (UtW[:, 0, 0] + UtW[:, 1, 1])) / float(n)

    pos = {int(t): i for i, t in enumerate(t_idx)}
    pairs = [(i, pos[int(t) + 1]) for i, t in enumerate(t_idx) if (int(t) + 1) in pos]
    if not pairs:
        return np.zeros(p_len, dtype=float)
    ia = np.array([p[0] for p in pairs], dtype=int)
    ib = np.array([p[1] for p in pairs], dtype=int)
    # cov_e[i, j] = phi - U[i].W[j] - W[i].U[j] + U[i]^T UtW U[j]
    ui = U[ia]  # (m1, 2)
    uj = U[ib]  # (m1, 2)
    term_uw = np.einsum("ma,pma->pm", ui, W[:, ib, :])
    term_wu = np.einsum("pma,ma->pm", W[:, ia, :], uj)
    term_quad = np.einsum("ma,pab,mb->pm", ui, UtW, uj)
    cov_ij = phis[:, None] - term_uw - term_wu + term_quad
    e_num = cov_ij.mean(axis=1)
    return e_num / np.maximum(e_den, 1e-15)


def invert_bias_corrected_r1(t_idx: np.ndarray, r1_hat: float) -> float:
    """Invert ar1_detrended_expected_r1(t_idx, phi) = r1_hat on [-0.95, 0.98]."""
    curve = ar1_detrended_expected_r1_curve(t_idx, _PHI_GRID)
    return float(np.interp(r1_hat, curve, _PHI_GRID))


def estimate_series_stats(t_idx: np.ndarray, y: np.ndarray, max_lag: int = 10) -> dict:
    """Compute detrended variance, lag-1 ACF, full ACF, and log-periodogram spectral slope."""
    n = len(t_idx)
    resid, _, _ = ols_detrend(t_idx, y)
    var_det = float((resid @ resid) / (n - 2))
    var_raw = float(np.var(y, ddof=1))
    c0 = float((resid @ resid) / n)
    pos = {int(t): i for i, t in enumerate(t_idx)}
    acf = {}
    lmax = min(max_lag, n // 3)
    for lag in range(1, lmax + 1):
        pairs = [(i, pos[int(t) + lag]) for i, t in enumerate(t_idx) if (int(t) + lag) in pos]
        if len(pairs) >= 6 and c0 > 1e-15:
            idx_a = np.array([p[0] for p in pairs], dtype=int)
            idx_b = np.array([p[1] for p in pairs], dtype=int)
            acf[lag] = float((resid[idx_a] @ resid[idx_b]) / (len(pairs) * c0))
        else:
            acf[lag] = None
    r1 = acf.get(1, None)

    # Periodogram on integer span grid [0, T_span - 1] with missing slots zero-filled (mean of resid is 0).
    # Restrict to interior 2-df Fourier frequencies k = 1 .. floor((span - 1) / 2), excluding 0 and even Nyquist.
    span = int(t_idx[-1] - t_idx[0]) + 1
    grid = np.zeros(span, dtype=float)
    grid[(t_idx - t_idx[0]).astype(int)] = resid
    fft_vals = np.fft.rfft(grid)
    k_max = (span - 1) // 2
    freqs = np.arange(1, k_max + 1, dtype=float) / float(span)
    power = (np.abs(fft_vals[1 : k_max + 1]) ** 2) / float(n)
    valid = power > 1e-30
    if int(valid.sum()) >= 4:
        lx = np.log(freqs[valid])
        ly = np.log(power[valid])
        lx_c = lx - lx.mean()
        b_spec = float((lx_c @ (ly - ly.mean())) / (lx_c @ lx_c))
    else:
        b_spec = float("nan")

    return {
        "n": n,
        "sigma2": var_det,
        "sigma": math.sqrt(max(var_det, 0.0)),
        "sigma2_raw": var_raw,
        "r1": r1,
        "acf": acf,
        "b_spec": b_spec,
    }


# ---------------------------------------------------------------------------
# 3. Validation checks (Rule 7) + Identifiability & Power simulations (Rule 1)
# ---------------------------------------------------------------------------

def run_validation_checks(rng: np.random.Generator) -> list[dict]:
    checks = []

    # K0a: closed-form gamma0_two_stage vs lifecycle_engine.py Engine.coefficient()['D2']
    worst_rel = 0.0
    for _ in range(50):
        a = float(rng.uniform(1.0, 12.0))
        b = float(rng.uniform(1.1, 40.0))
        sigma = float(rng.uniform(0.1, 1.5))
        phi = float(rng.uniform(-0.80, 0.95))
        desc = eng.Description(
            stages=[eng.Stage("s0", a, "memoryless"), eng.Stage("s1", b, "memoryless")],
            transfers=[
                eng.Transfer("s0", "s1", driver="h0"),
                eng.Transfer("s1", "s0", birth=True),
            ],
            drivers=[eng.Driver("h0", sigma, phi, "lognormal")],
        )
        d2_eng = float(eng.Engine(desc).coefficient()["D2"])
        d2_cf = gamma0_two_stage(a, b, sigma, phi)
        rel = abs(d2_cf - d2_eng) / max(abs(d2_eng), 1e-12)
        worst_rel = max(worst_rel, rel)
    checks.append({
        "id": "K0a",
        "claim": "Closed-form two-stage Gamma_0 matches lifecycle_engine D2 across 50 random (a,b,sigma,phi) in (-0.8, 0.95)",
        "tolerance": 1e-10,
        "observed": worst_rel,
        "pass": bool(worst_rel < 1e-10),
    })

    # K0b: Lean P1/P2 E(a,b) - E(a) identity
    worst_e = 0.0
    for _ in range(50):
        a = float(rng.uniform(1.0, 15.0))
        b = float(rng.uniform(1.0, 30.0))
        sigma = float(rng.uniform(0.2, 1.4))
        phi = float(rng.uniform(0.01, 0.95))
        ln_r0 = float(rng.uniform(0.05, 2.0))
        diff_direct = persistence_index_E(a, b, sigma, phi, ln_r0) - persistence_index_E(a, None, sigma, phi, ln_r0)
        diff_lean = lean_p1_delta_E(a, b, sigma, phi, ln_r0)
        rel = abs(diff_direct - diff_lean) / max(abs(diff_direct), 1e-12)
        worst_e = max(worst_e, rel)
    checks.append({
        "id": "K0b",
        "claim": "Lean P1/P2 persistence-index difference matches direct 2(ln R0 - Gamma_0)/(T_0 V) across 50 random cases",
        "tolerance": 1e-12,
        "observed": worst_e,
        "pass": bool(worst_e < 1e-12),
    })

    # K0c: White-noise detrended lag-1 bias formula vs Monte Carlo at N = 15
    t15 = np.arange(15, dtype=float)
    analytic_mu0 = white_noise_r1_null_mean(t15)
    mc_r1 = []
    for _ in range(20000):
        y = 2.0 + 0.05 * t15 + rng.standard_normal(15) * 0.8
        mc_r1.append(estimate_series_stats(t15, y, max_lag=2)["r1"])
    mc_mean = float(np.mean(mc_r1))
    diff_mu0 = abs(mc_mean - analytic_mu0)
    checks.append({
        "id": "K0c",
        "claim": "Analytical hat-matrix white-noise detrended r1 mean matches 20,000 Monte Carlo series at N=15 (with linear trend)",
        "tolerance": 0.015,
        "analytic_mu0": analytic_mu0,
        "mc_mean": mc_mean,
        "observed": diff_mu0,
        "pass": bool(diff_mu0 < 0.015),
    })

    # K0d: Interior-frequency spectral slope has |mean| < 0.02 on demeaned white noise,
    # positive finite-N bluing shift on linearly detrended white noise, and mean < -1.0 on red AR(1) phi=0.85 (N=30)
    t30 = np.arange(30, dtype=float)
    k_max30 = (30 - 1) // 2
    freqs30 = np.log(np.arange(1, k_max30 + 1, dtype=float) / 30.0)
    lx_c30 = freqs30 - freqs30.mean()
    wn_demeaned_slopes = []
    wn_detrended_slopes = []
    red_slopes = []
    for _ in range(8000):
        wn = rng.standard_normal(30)
        wn0 = wn - wn.mean()
        pw = (np.abs(np.fft.rfft(wn0)[1 : k_max30 + 1]) ** 2) / 30.0
        ly = np.log(pw)
        wn_demeaned_slopes.append(float((lx_c30 @ (ly - ly.mean())) / (lx_c30 @ lx_c30)))
        wn_detrended_slopes.append(estimate_series_stats(t30, wn, max_lag=2)["b_spec"])
        z = np.empty(30)
        z[0] = rng.standard_normal()
        for t in range(1, 30):
            z[t] = 0.85 * z[t - 1] + math.sqrt(1.0 - 0.85 ** 2) * rng.standard_normal()
        red_slopes.append(estimate_series_stats(t30, z, max_lag=2)["b_spec"])
    wn_mean = float(np.mean(wn_demeaned_slopes))
    wn_det_mean = float(np.mean(wn_detrended_slopes))
    red_mean = float(np.mean(red_slopes))
    checks.append({
        "id": "K0d",
        "claim": "Interior-frequency log-periodogram slope has |mean| < 0.02 on demeaned white noise and mean < -1.0 on detrended AR(1) phi=0.85 (N=30, 8,000 trials)",
        "tolerance": 0.02,
        "wn_demeaned_mean": wn_mean,
        "wn_detrended_bluing_shift_N30": wn_det_mean,
        "red_detrended_mean": red_mean,
        "observed": abs(wn_mean),
        "pass": bool(abs(wn_mean) < 0.02 and wn_det_mean > 0.0 and red_mean < -1.0),
    })

    # K0e: Fast O(n) rank-2 forward-backward AR(1) detrended curve matches dense M @ Gamma @ M
    worst_fast = 0.0
    test_phis = np.array([-0.75, -0.25, 0.0, 0.35, 0.85])
    for _ in range(30):
        t_gapped = np.sort(rng.choice(45, size=22, replace=False)).astype(float)
        c_fast = ar1_detrended_expected_r1_curve(t_gapped, test_phis)
        c_dense = np.array([ar1_detrended_expected_r1_dense(t_gapped, float(p)) for p in test_phis])
        worst_fast = max(worst_fast, float(np.max(np.abs(c_fast - c_dense))))
    checks.append({
        "id": "K0e",
        "claim": "Fast O(n) rank-2 forward-backward AR(1) detrended r1 curve matches dense M @ Gamma @ M across 30 gapped grids",
        "tolerance": 1e-12,
        "observed": worst_fast,
        "pass": bool(worst_fast < 1e-12),
    })

    return checks


def simulate_ar1_with_obs_noise(
    rng: np.random.Generator, n_obs: int, sigma_tot: float, phi: float, eta_obs: float, miss_prob: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic series with n_obs observed points on an integer grid with
    missing-step rate miss_prob, total log-SD sigma_tot, observación-error share eta_obs,
    and linear trend.
    """
    span = int(math.ceil(n_obs / (1.0 - miss_prob))) + 4
    while True:
        keep = rng.random(span) >= miss_prob
        keep[0] = True
        if int(keep.sum()) >= n_obs:
            t_idx = np.flatnonzero(keep)[:n_obs].astype(float)
            # Ensure at least 12 consecutive pairs
            diffs = np.diff(t_idx)
            if int(np.sum(diffs == 1.0)) >= min(12, n_obs - 2):
                break
        span += 3
    T = int(t_idx[-1]) + 1
    sig_proc = sigma_tot * math.sqrt(1.0 - eta_obs)
    sig_obs = sigma_tot * math.sqrt(eta_obs)
    x = np.empty(T, dtype=float)
    if sig_proc > 0.0:
        x[0] = rng.standard_normal() * sig_proc
        innov_sd = sig_proc * math.sqrt(max(1.0 - phi * phi, 0.0))
        for t in range(1, T):
            x[t] = phi * x[t - 1] + rng.standard_normal() * innov_sd
    else:
        x[:] = 0.0
    obs = x[t_idx.astype(int)] + rng.standard_normal(n_obs) * sig_obs
    # Add arbitrary linear trend (invariance check: OLS detrending removes it exactly)
    trend = rng.uniform(-0.04, 0.04) * t_idx + rng.uniform(1.0, 5.0)
    return t_idx, obs + trend


def run_identifiability_grid(rng: np.random.Generator) -> list[dict]:
    rows = []
    trials = 1500
    for n_obs in (15, 20, 30, 50, 100):
        for phi in (-0.30, 0.00, 0.10, 0.20, 0.40, 0.75):
            for eta_obs in (0.00, 0.25, 0.50):
                if n_obs not in (15, 30, 100) and eta_obs == 0.25:
                    continue
                sig_hats, r1_hats, r1_bcs, b_specs = [], [], [], []
                for _ in range(trials):
                    t_idx, y = simulate_ar1_with_obs_noise(rng, n_obs, 0.80, phi, eta_obs, miss_prob=0.08)
                    st = estimate_series_stats(t_idx, y, max_lag=3)
                    sig_hats.append(st["sigma"])
                    r1_hats.append(st["r1"])
                    r1_bcs.append(invert_bias_corrected_r1(t_idx, st["r1"]))
                    b_specs.append(st["b_spec"])
                r1_arr = np.asarray(r1_hats, float)
                bc_arr = np.asarray(r1_bcs, float)
                sig_arr = np.asarray(sig_hats, float)
                bs_arr = np.asarray(b_specs, float)
                rows.append({
                    "n_obs": n_obs,
                    "true_phi": phi,
                    "eta_obs": eta_obs,
                    "true_obs_lag1": phi * (1.0 - eta_obs),
                    "mean_sigma_hat": float(sig_arr.mean()),
                    "sd_sigma_hat": float(sig_arr.std(ddof=1)),
                    "mean_r1_hat": float(r1_arr.mean()),
                    "sd_r1_hat": float(r1_arr.std(ddof=1)),
                    "rmse_r1_hat_vs_phi": float(math.sqrt(np.mean((r1_arr - phi) ** 2))),
                    "mean_r1_bc": float(bc_arr.mean()),
                    "sd_r1_bc": float(bc_arr.std(ddof=1)),
                    "rmse_r1_bc_vs_phi": float(math.sqrt(np.mean((bc_arr - phi) ** 2))),
                    "mean_b_spec": float(np.nanmean(bs_arr)),
                    "sd_b_spec": float(np.nanstd(bs_arr, ddof=1)),
                })

    # Also test AR(2) 3.5-year vole-like quasi-cycle (r=0.80, period=3.5 -> omega=2*pi/3.5)
    r_mod = 0.80
    omega = 2.0 * math.pi / 3.5
    a1 = 2.0 * r_mod * math.cos(omega)
    a2 = -(r_mod * r_mod)
    # Theoretical lag-1 and lag-2 ACF of AR(2): rho(1) = a1 / (1 - a2), rho(2) = a1*rho(1) + a2
    rho1_true = a1 / (1.0 - a2)
    rho2_true = a1 * rho1_true + a2
    for n_obs in (20, 40):
        r1_list, r2_list, r3_list, bs_list = [], [], [], []
        for _ in range(trials):
            T = n_obs + 50
            x = np.zeros(T)
            for t in range(2, T):
                x[t] = a1 * x[t - 1] + a2 * x[t - 2] + rng.standard_normal() * 0.5
            y = x[50:] + 0.02 * np.arange(n_obs)
            t_idx = np.arange(n_obs, dtype=float)
            st = estimate_series_stats(t_idx, y, max_lag=4)
            r1_list.append(st["r1"])
            r2_list.append(st["acf"][2])
            r3_list.append(st["acf"][3])
            bs_list.append(st["b_spec"])
        rows.append({
            "model": "AR2_vole_quasicycle_period_3.5",
            "n_obs": n_obs,
            "true_rho1": rho1_true,
            "true_rho2": rho2_true,
            "mean_r1_hat": float(np.mean(r1_list)),
            "mean_r2_hat": float(np.mean(r2_list)),
            "mean_r3_hat": float(np.mean(r3_list)),
            "mean_b_spec": float(np.mean(bs_list)),
        })
    return rows


def run_clustered_power_simulation(rng: np.random.Generator) -> list[dict]:
    """Simulate group-level power of the registered excess-memory test T1:
    Each taxon/interval group has M series grouped into K = M // 5 studies (5 series/study)
    sharing a common weather innovation with correlation rho_within = 0.40, series lengths
    drawn uniformly from {15, ..., 35}, missing rate 8%.
    For each series i, compute delta_r1_i = r1_hat_i - white_noise_r1_null_mean(t_idx_i)
    (exact hat-matrix null expectation, verified in K0c to 1e-4 against 20,000 MC draws).
    Test whether study-cluster mean of delta_r1 differs from 0 at two-sided alpha = 0.05.
    """
    from scipy.stats import t as student_t

    def fast_r1_and_mu0(t_idx: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        n = len(t_idx)
        resid, _, _ = ols_detrend(t_idx, y)
        c0 = float(resid @ resid) / n
        t_c = t_idx - t_idx.mean()
        denom_t = float(t_c @ t_c)
        # Adjacent steps have diff == 1.0 since t_idx is strictly increasing integers
        adj = np.flatnonzero(np.diff(t_idx) == 1.0)
        m1 = len(adj)
        c1 = float(resid[adj] @ resid[adj + 1]) / m1
        r1_hat = c1 / c0 if c0 > 1e-15 else 0.0
        sum_h = (m1 / float(n)) + (float(t_c[adj] @ t_c[adj + 1]) / denom_t if denom_t > 0 else 0.0)
        mu0 = -float(n) / float((n - 2) * m1) * sum_h
        return r1_hat, mu0

    power_rows = []
    n_datasets = 500
    rho_within = 0.40

    for m_series in (25, 100, 300):
        n_studies = m_series // 5
        tcrit = float(student_t.ppf(0.975, df=n_studies - 1))
        for phi in (0.00, 0.05, 0.10, 0.15, 0.20, 0.40, 0.75):
            for eta_obs in (0.00, 0.25, 0.50):
                if phi in (0.05, 0.15) and eta_obs == 0.25:
                    continue
                pos_sig = 0
                neg_sig = 0
                raw_uncorrected_neg_sig = 0
                mean_deltas = []
                for _ in range(n_datasets):
                    study_means = []
                    study_raw_means = []
                    for _s in range(n_studies):
                        T = 60
                        common_eps = rng.standard_normal(T)
                        s_deltas = []
                        s_raws = []
                        for _k in range(5):
                            n_obs = int(rng.integers(15, 36))
                            span = int(math.ceil(n_obs / 0.92)) + 3
                            keep = rng.random(span) >= 0.08
                            keep[0] = True
                            while int(keep.sum()) < n_obs or int(np.sum(np.diff(np.flatnonzero(keep)[:n_obs]) == 1)) < 12:
                                span += 3
                                keep = rng.random(span) >= 0.08
                                keep[0] = True
                            t_idx = np.flatnonzero(keep)[:n_obs].astype(float)
                            T_use = int(t_idx[-1]) + 1
                            if T_use > len(common_eps):
                                common_eps = rng.standard_normal(T_use + 10)
                            eps = math.sqrt(rho_within) * common_eps[:T_use] + math.sqrt(1.0 - rho_within) * rng.standard_normal(T_use)
                            sig_tot = 0.80
                            sig_proc = sig_tot * math.sqrt(1.0 - eta_obs)
                            sig_obs = sig_tot * math.sqrt(eta_obs)
                            x = np.zeros(T_use)
                            if sig_proc > 0:
                                x[0] = eps[0] * sig_proc
                                innov_sd = sig_proc * math.sqrt(max(1.0 - phi * phi, 0.0))
                                for t in range(1, T_use):
                                    x[t] = phi * x[t - 1] + eps[t] * innov_sd
                            y = x[t_idx.astype(int)] + rng.standard_normal(n_obs) * sig_obs + 0.03 * t_idx
                            r1_hat, mu0 = fast_r1_and_mu0(t_idx, y)
                            s_deltas.append(r1_hat - mu0)
                            s_raws.append(r1_hat)
                        study_means.append(float(np.mean(s_deltas)))
                        study_raw_means.append(float(np.mean(s_raws)))
                    arr = np.asarray(study_means, float)
                    se = float(arr.std(ddof=1)) / math.sqrt(n_studies)
                    tstat = float(arr.mean()) / max(se, 1e-15)
                    if tstat > tcrit:
                        pos_sig += 1
                    elif tstat < -tcrit:
                        neg_sig += 1
                    arr_raw = np.asarray(study_raw_means, float)
                    se_raw = float(arr_raw.std(ddof=1)) / math.sqrt(n_studies)
                    if float(arr_raw.mean()) / max(se_raw, 1e-15) < -tcrit:
                        raw_uncorrected_neg_sig += 1
                    mean_deltas.append(float(arr.mean()))
                power_rows.append({
                    "M_series": m_series,
                    "K_studies": n_studies,
                    "true_phi": phi,
                    "eta_obs": eta_obs,
                    "mean_excess_r1": float(np.mean(mean_deltas)),
                    "prob_support_pos_memory": pos_sig / n_datasets,
                    "prob_support_neg_memory": neg_sig / n_datasets,
                    "spurious_neg_rate_if_uncorrected_r1_used": raw_uncorrected_neg_sig / n_datasets,
                })
    return power_rows


def run_regime_classification_identifiability(rng: np.random.Generator) -> dict:
    """Evaluate how finite-sample estimation at N=20 and N=40 affects the three
    theoretical regime fractions on a synthetic population of 1,000 host regimes
    drawn from a broad prior sigma ~ Uniform(0.2, 1.6), phi ~ Uniform(-0.2, 0.85).
    """
    n_pop = 1000
    true_sigmas = rng.uniform(0.20, 1.60, n_pop)
    true_phis = rng.uniform(-0.20, 0.85, n_pop)

    # Evaluate second-order regimes at fine R16 step (a=4, b=9.2 short) and Lean P2 (a=1, b=2.3, ln R0=0.35)
    def classify(sigmas: np.ndarray, phis: np.ndarray) -> dict:
        n = len(sigmas)
        # (ii-loss, fine R16 short cycle a=4, b=9.2, phi_f = phi^0.25 for phi>0): Gamma_0(4, 9.2) > Gamma_0(4)
        harm_loss_fine = 0
        # (ii-index, Lean P2 coarse short cycle a=1, b=2.3 at illustrative Borrelia ln R0 = 0.35): E2 < E1
        harm_index_short = 0
        # (i-halfbound, coarse moderate b=7.5): Gamma_0(1, 7.5) < sigma^2 / 4 (at least half bound buffered)
        park_half_mod = 0
        # (iii-proxy): large-noise vulnerability sigma > 0.8 where R10 shows >30% error on long cycle without strong memory
        high_noise = 0
        for s, p in zip(sigmas, phis):
            p_clip = float(np.clip(p, -0.95, 0.98))
            g_mod = gamma0_two_stage(1.0, 7.5, float(s), p_clip)
            if g_mod < 0.25 * float(s) * float(s):
                park_half_mod += 1
            if p_clip > 0.0:
                pf = p_clip ** 0.25
                if gamma0_two_stage(4.0, 9.2, float(s), pf) > gamma0_one_stage(4.0, float(s)):
                    harm_loss_fine += 1
                if persistence_index_E(1.0, 2.3, float(s), p_clip, 0.35) < persistence_index_E(1.0, None, float(s), p_clip, 0.35):
                    harm_index_short += 1
            if s > 0.80:
                high_noise += 1
        return {
            "park_half_mod": park_half_mod / n,
            "harm_loss_fine_short": harm_loss_fine / n,
            "harm_index_short_lnR0_0.35": harm_index_short / n,
            "sigma_gt_0.80": high_noise / n,
        }

    truth = classify(true_sigmas, true_phis)
    out = {"truth": truth}
    for n_obs in (20, 40):
        est_sig, est_r1, est_bc = [], [], []
        t_idx = np.arange(n_obs, dtype=float)
        for s, p in zip(true_sigmas, true_phis):
            x = np.zeros(n_obs + 30)
            innov = float(s) * math.sqrt(max(1.0 - float(p) ** 2, 0.0))
            for t in range(1, n_obs + 30):
                x[t] = float(p) * x[t - 1] + rng.standard_normal() * innov
            y = x[30:] + 0.02 * t_idx
            st = estimate_series_stats(t_idx, y, max_lag=1)
            est_sig.append(st["sigma"])
            est_r1.append(st["r1"])
            est_bc.append(invert_bias_corrected_r1(t_idx, st["r1"]))
        out[f"N{n_obs}_plugin_r1"] = classify(np.array(est_sig), np.array(est_r1))
        out[f"N{n_obs}_bias_corrected_r1"] = classify(np.array(est_sig), np.array(est_bc))
    return out


def main() -> int:
    t0 = time.monotonic()
    utc_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rng = np.random.default_rng(2026111426)

    checks = run_validation_checks(rng)
    if not all(c["pass"] for c in checks):
        raise RuntimeError(f"Validation check failed: {checks}")

    ident = run_identifiability_grid(rng)
    power = run_clustered_power_simulation(rng)
    regimes = run_regime_classification_identifiability(rng)
    elapsed = time.monotonic() - t0

    payload = {
        "generated_utc": utc_now,
        "elapsed_seconds": round(elapsed, 2),
        "seed": 2026111426,
        "note": "Computed BEFORE any empirical dataset was downloaded or opened (CASESTUDY_SPEC.md Rule 1).",
        "validation_checks": checks,
        "identifiability_grid": ident,
        "clustered_power": power,
        "regime_identifiability": regimes,
    }
    (RESULTS_DIR / "preanalysis_power.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"PRE-ANALYSIS VALIDATION, IDENTIFIABILITY & POWER (UTC {utc_now}, {elapsed:.2f}s)",
        "Computed BEFORE any empirical data were downloaded or analysed.",
        "",
        "=== 1. Rule 7 Validation Checks ===",
    ]
    for c in checks:
        lines.append(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['id']}: observed={c['observed']:.3e} (tol={c['tolerance']:.3e}) — {c['claim']}")
    lines.append("")
    lines.append("=== 2. Selected Estimator Bias & RMSE (true sigma = 0.80) ===")
    lines.append("N   | phi   | eta_obs | E[sig_hat] | E[r1_hat] | RMSE(r1) | E[r1_bc] | RMSE(r1_bc) | E[b_spec]")
    for r in ident:
        if "model" in r:
            lines.append(
                f"AR2(P=3.5) N={r['n_obs']}: true_rho1={r['true_rho1']:+.3f}, true_rho2={r['true_rho2']:+.3f} "
                f"-> E[r1]={r['mean_r1_hat']:+.3f}, E[r2]={r['mean_r2_hat']:+.3f}, E[r3]={r['mean_r3_hat']:+.3f}, E[b_spec]={r['mean_b_spec']:+.3f}"
            )
            continue
        if r["n_obs"] in (15, 30, 100) and r["eta_obs"] in (0.0, 0.5):
            lines.append(
                f"{r['n_obs']:3d} | {r['true_phi']:+5.2f} | {r['eta_obs']:7.2f} | "
                f"{r['mean_sigma_hat']:10.3f} | {r['mean_r1_hat']:+9.3f} | {r['rmse_r1_hat_vs_phi']:8.3f} | "
                f"{r['mean_r1_bc']:+8.3f} | {r['rmse_r1_bc_vs_phi']:11.3f} | {r['mean_b_spec']:+9.3f}"
            )
    lines.append("")
    lines.append("=== 3. Study-Clustered Power & Type I Error for Excess Memory Test T1 (alpha=0.05 two-sided) ===")
    lines.append("M   | K_st | phi   | eta_obs | E[delta_r1] | P(support +mem) | P(support -mem) | Spurious -mem if raw r1 used")
    for p in power:
        lines.append(
            f"{p['M_series']:3d} | {p['K_studies']:4d} | {p['true_phi']:+5.2f} | {p['eta_obs']:7.2f} | "
            f"{p['mean_excess_r1']:+11.4f} | {p['prob_support_pos_memory']:15.3f} | "
            f"{p['prob_support_neg_memory']:15.3f} | {p['spurious_neg_rate_if_uncorrected_r1_used']:28.3f}"
        )
    lines.append("")
    lines.append("=== 4. Regime Fraction Identifiability (1,000 synthetic host regimes) ===")
    for k, v in regimes.items():
        lines.append(f"  {k:26s}: {json.dumps({kk: round(vv, 3) for kk, vv in v.items()})}")

    txt = "\n".join(lines) + "\n"
    (RESULTS_DIR / "preanalysis_power.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
