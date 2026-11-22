#!/usr/bin/env python3
"""Pre-analysis power and identifiability simulation for case study `devtimes`.

Run BEFORE any empirical dataset is fetched or analysed:
    python3 -I casestudies/devtimes/power_identifiability.py

Checks known analytical results first (Rule 7), then computes:
  P1. Sample-size bias, RMSE, and power to distinguish individual stage-duration
      CV from the exponential/geometric baseline (CV = 1) and from R10b's
      protection thresholds (CV = 0.043, 0.102, 0.21, 0.48).
  P2. Identifiability limits from interval censoring (observation spacing dt_obs),
      right-censoring (study truncation T_max), and group-mean / SE conflation.
  P3. Compilation-level Type I error and power for registered hypotheses H_dev1,
      H_sys, and H_transform (testing CV-sufficiency vs full transform E[phi^L]).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
from scipy import optimize, special, stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load project lifecycle engine (read-only import)
ENGINE_PATH = ROOT / "studies" / "engine-R16" / "lifecycle_engine.py"
_spec = importlib.util.spec_from_file_location("lifecycle_engine", ENGINE_PATH)
LE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = LE
_spec.loader.exec_module(LE)


def sample_cv(x: np.ndarray, corrected: bool = True) -> np.ndarray:
    """Sample CV along last axis; optional Sokal-Rohlf (1 + 1/(4n)) small-n bias correction."""
    n = x.shape[-1]
    mean = x.mean(axis=-1)
    sd = x.std(axis=-1, ddof=1)
    cv = sd / np.maximum(mean, 1e-300)
    return cv * (1.0 + 1.0 / (4.0 * n)) if corrected else cv


def closed_form_D2_chain(f_steps: float, f_m: int, l_steps: float, l_m: int, sigma: float, phi: float) -> float:
    """Second-order loss D2 for two-stage serial chain at threshold (Statement C2b)."""
    T = f_steps + l_steps
    if phi == 0.0:
        return sigma * sigma / (2.0 * T)
    theta = phi / (1.0 - phi)
    rho_f = (phi ** f_steps) if f_m == 0 else (1.0 + f_steps / (f_m * theta)) ** (-f_m)
    rho_l = (phi ** l_steps) if l_m == 0 else (1.0 + l_steps / (l_m * theta)) ** (-l_m)
    R = rho_f * rho_l
    return sigma * sigma * ((1.0 + 2.0 * theta) / (2.0 * T) - R / (1.0 - R))


def run_known_checks() -> list[dict]:
    """Rule 7: verify code against known analytical results with pre-set tolerances."""
    checks = []

    # K1: Discrete chain CV formula vs exact negative-binomial moments (tol = 1e-12)
    tau, m = 24.0, 4
    p = m / tau  # per-substage exit prob
    mean_nb = m / p
    var_nb = m * (1.0 - p) / (p * p)
    cv_exact = math.sqrt(var_nb) / mean_nb
    cv_formula = math.sqrt(1.0 / m - 1.0 / tau)
    err_k1 = abs(cv_exact - cv_formula)
    checks.append({
        "id": "K1",
        "claim": "Discrete m-substage chain CV equals sqrt(1/m - 1/tau)",
        "tol": 1e-12,
        "error": err_k1,
        "pass": bool(err_k1 < 1e-12),
    })

    # K2: Closed-form D2 equals lifecycle_engine.py resolvent D2 (tol = 1e-10 relative)
    desc = LE.Description(
        stages=[
            LE.Stage("exposed", mean=8.0, shape="fixed"),
            LE.Stage("long", mean=60.0, shape="chain", substages=4),
        ],
        transfers=[
            LE.Transfer("exposed", "long", driver="h"),
            LE.Transfer("long", "exposed", birth=True),
        ],
        drivers=[LE.Driver("h", sigma=1.4, phi=0.75 ** (1.0 / 8.0), kind="lognormal")],
    )
    d2_eng = LE.Engine(desc).coefficient()["D2"]
    d2_cf = closed_form_D2_chain(8.0, 0, 60.0, 4, 1.4, 0.75 ** (1.0 / 8.0))
    rel_k2 = abs(d2_eng - d2_cf) / abs(d2_cf)
    checks.append({
        "id": "K2",
        "claim": "Closed-form D2 matches lifecycle_engine resolvent D2 on fine-step malaria m=4",
        "tol": 1e-10,
        "error": rel_k2,
        "pass": bool(rel_k2 < 1e-10),
    })

    # K3: Interval binning adds dt_obs^2 / 12 variance on random entry phases (tol = 5e-3 rel)
    rng = np.random.default_rng(20261114)
    mu, k_shape, dt_obs = 30.0, 25.0, 7.0  # true CV = 0.20, weekly checks
    raw_t = rng.gamma(shape=k_shape, scale=mu / k_shape, size=400_000)
    phase = rng.uniform(0.0, dt_obs, size=400_000)
    # Midpoint of interval in which event is first detected relative to start check
    binned_t = np.ceil((raw_t - phase) / dt_obs) * dt_obs + 0.5 * dt_obs
    var_emp = float(np.var(binned_t, ddof=1))
    # Entry at uniform [0, dt_obs] before check 0 + exit in [k-1, k]*dt_obs adds 2*(dt^2/12) if entry also binned,
    # or 1*(dt^2/12) when infection time t=0 is exact (experimental cohort) and checks are at dt, 2*dt, ...:
    binned_exact_entry = (np.floor(raw_t / dt_obs) + 0.5) * dt_obs
    # Note: for smooth density far from 0, Var(floor(T/dt)*dt + dt/2) -> Var(T) + dt^2/12 up to Sheppard oscillatory terms
    var_exact_entry = float(np.var(binned_exact_entry, ddof=1))
    var_theory = (mu * mu / k_shape) + (dt_obs * dt_obs) / 12.0
    rel_k3 = abs(var_exact_entry - var_theory) / var_theory
    checks.append({
        "id": "K3",
        "claim": "Exact-entry interval-midpoint variance matches Var(T) + dt_obs^2 / 12",
        "tol": 5e-3,
        "error": rel_k3,
        "pass": bool(rel_k3 < 5e-3),
    })

    if not all(c["pass"] for c in checks):
        raise RuntimeError(f"Known check failed: {checks}")
    return checks


def sim_p1_sample_size_power(rng: np.random.Generator) -> list[dict]:
    """P1: Bias, RMSE, and power of individual CV across sample sizes n and true CVs."""
    ns = [5, 10, 20, 30, 50, 100, 250]
    true_cvs = [0.04, 0.10, 0.20, 0.35, 0.50, 0.80, 1.00, 1.30]
    trials = 4000
    rows = []
    for cv0 in true_cvs:
        k = 1.0 / (cv0 * cv0)
        for n in ns:
            x = rng.gamma(shape=k, scale=1.0 / k, size=(trials, n))
            cv_raw = sample_cv(x, corrected=False)
            cv_cor = sample_cv(x, corrected=True)
            # Asymptotic / McKay-Vangel SE of sample CV for gamma/normal-like positive variables:
            # SE(cv) approx cv * sqrt((1 + 2*cv^2)/(2*(n-1))) -- check coverage and one-sided tests
            se_est = cv_cor * np.sqrt((1.0 + 2.0 * cv_cor * cv_cor) / (2.0 * (n - 1)))
            ci_hi = cv_cor + 1.96 * se_est
            ci_lo = np.maximum(0.0, cv_cor - 1.96 * se_est)
            cov = float(np.mean((ci_lo <= cv0) & (cv0 <= ci_hi)))
            # Power to reject exponential baseline CV >= 1.0 (upper 95% bound < 0.85, or z-test vs CV=1)
            # Under exponential(1) null, get exact empirical 5th percentile cutoff at this n:
            x_exp = rng.exponential(scale=1.0, size=(trials, n))
            cv_exp = sample_cv(x_exp, corrected=True)
            q05_exp = float(np.quantile(cv_exp, 0.05))
            power_below_exp = float(np.mean(cv_cor < q05_exp))
            # Classification accuracy relative to R10b malaria threshold (CV < 0.21 vs CV > 0.21)
            # and schistosome fine-step bracket upper edge (CV > 0.102)
            prob_above_0102 = float(np.mean(ci_lo > 0.102))
            prob_above_0210 = float(np.mean(ci_lo > 0.210))
            prob_below_0480 = float(np.mean(ci_hi < 0.480))
            rows.append({
                "true_cv": cv0,
                "n": n,
                "mean_cv_raw": float(cv_raw.mean()),
                "mean_cv_cor": float(cv_cor.mean()),
                "bias_cor": float(cv_cor.mean() - cv0),
                "rmse_cor": float(np.sqrt(np.mean((cv_cor - cv0) ** 2))),
                "ci95_coverage": cov,
                "exp_null_q05": q05_exp,
                "power_reject_exponential": power_below_exp,
                "power_certify_above_schisto_0102": prob_above_0102,
                "power_certify_above_malaria_0210": prob_above_0210,
                "power_certify_below_malaria_m4_0480": prob_below_0480,
            })
    return rows


def weibull_mle_censored(t_obs: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    """Fit Weibull(shape k, scale lam) by exact censored log-likelihood; return (mean, cv)."""
    d = float(event.sum())
    if d < 3:
        return math.nan, math.nan

    def neg_profile_ll(log_k: float) -> float:
        k = math.exp(log_k)
        tk = t_obs ** k
        lam_k = float(tk.sum() / d)
        ll = d * math.log(k) - d * math.log(lam_k) + (k - 1.0) * float(np.log(np.maximum(t_obs[event], 1e-300)).sum()) - d
        return -ll

    res = optimize.minimize_scalar(neg_profile_ll, bounds=(-2.5, 4.5), method="bounded")
    k_hat = math.exp(float(res.x))
    lam_hat = float((t_obs ** k_hat).sum() / d) ** (1.0 / k_hat)
    mean_hat = lam_hat * special.gamma(1.0 + 1.0 / k_hat)
    var_hat = lam_hat * lam_hat * (special.gamma(1.0 + 2.0 / k_hat) - special.gamma(1.0 + 1.0 / k_hat) ** 2)
    cv_hat = math.sqrt(max(var_hat, 0.0)) / max(mean_hat, 1e-300)
    return float(mean_hat), float(cv_hat)


def sim_p2_identifiability_distortions(rng: np.random.Generator) -> dict:
    """P2: Quantify interval censoring, right censoring, and SE/group-mean misattribution."""
    out = {"interval_censoring": [], "right_censoring": [], "misattribution": []}

    # 2a. Interval censoring across ratios dt_obs / mean
    mu = 30.0
    for true_cv in (0.04, 0.10, 0.20, 0.50):
        k = 1.0 / (true_cv * true_cv)
        for ratio in (0.033, 0.10, 0.233, 0.467):  # ~1d, 3d, 7d, 14d on 30d mean
            dt_obs = ratio * mu
            t = rng.gamma(shape=k, scale=mu / k, size=(2000, 60))
            t_mid = (np.floor(t / dt_obs) + 0.5) * dt_obs
            cv_naive = sample_cv(t_mid, corrected=True)
            # Sheppard-debiased variance: max(s^2 - dt_obs^2/12, 0)
            m_hat = t_mid.mean(axis=-1)
            v_hat = t_mid.var(axis=-1, ddof=1)
            v_deb = np.maximum(v_hat - (dt_obs * dt_obs) / 12.0, 0.0)
            cv_deb = np.sqrt(v_deb) / m_hat
            out["interval_censoring"].append({
                "true_cv": true_cv,
                "dt_over_mean": ratio,
                "mean_cv_naive_midpoint": float(cv_naive.mean()),
                "mean_cv_sheppard_debiased": float(cv_deb.mean()),
                "floor_dt_sqrt12": ratio / math.sqrt(12.0),
            })

    # 2b. Right censoring (experiment stopped at T_max / mean)
    for true_cv in (0.35, 0.70, 1.00):
        # Match Weibull shape to true_cv
        f_cv = lambda k: math.sqrt(max(special.gamma(1 + 2 / k) / special.gamma(1 + 1 / k) ** 2 - 1.0, 0.0)) - true_cv
        k_w = optimize.brentq(f_cv, 0.4, 20.0)
        scale_w = mu / special.gamma(1.0 + 1.0 / k_w)
        for tmax_mult in (1.0, 1.5, 2.0, 3.0):
            tmax = tmax_mult * mu
            naive_cvs, mle_cvs, cens_fracs = [], [], []
            for _ in range(400):
                u = rng.weibull(k_w, size=80) * scale_w
                ev = u <= tmax
                t_obs = np.minimum(u, tmax)
                cens_fracs.append(1.0 - float(ev.mean()))
                if ev.sum() >= 5:
                    naive_cvs.append(float(sample_cv(u[ev], corrected=True)))
                _, cv_mle = weibull_mle_censored(t_obs, ev)
                if math.isfinite(cv_mle):
                    mle_cvs.append(cv_mle)
            out["right_censoring"].append({
                "true_cv": true_cv,
                "tmax_over_mean": tmax_mult,
                "censored_fraction": float(np.mean(cens_fracs)),
                "naive_completers_cv": float(np.mean(naive_cvs)),
                "weibull_censored_mle_cv": float(np.mean(mle_cvs)),
            })

    # 2c. Misattribution: treating SE or vial-mean SD (m=10 per vial, k=6 vials) as individual SD
    for true_cv in (0.50, 1.00):
        k = 1.0 / (true_cv * true_cv)
        # 6 vials of 10 individuals each (total N=60)
        x = rng.gamma(shape=k, scale=mu / k, size=(2000, 6, 10))
        vial_means = x.mean(axis=-1)
        cv_vial_sd = sample_cv(vial_means, corrected=True)  # mistakes vial SD for individual SD
        # Mistaking SE across N=25 individuals for SD
        x25 = rng.gamma(shape=k, scale=mu / k, size=(2000, 25))
        se_over_mean = (x25.std(axis=-1, ddof=1) / math.sqrt(25)) / x25.mean(axis=-1)
        recovered_cv = se_over_mean * math.sqrt(25) * (1.0 + 1.0 / 100.0)
        out["misattribution"].append({
            "true_cv": true_cv,
            "apparent_cv_from_vial_means_m10": float(cv_vial_sd.mean()),
            "apparent_cv_if_SE_n25_mistaken_for_SD": float(se_over_mean.mean()),
            "recovered_cv_when_SE_multiplied_by_sqrtn": float(recovered_cv.mean()),
        })
    return out


def sim_p3_compilation_power_and_transforms(rng: np.random.Generator) -> dict:
    """P3: Compilation-level power for H_dev1 and exact transform sensitivity for H_transform."""
    K_dev, K_surv = 40, 25
    B = 2000

    def eval_compilations(dev_lo, dev_hi, surv_lo, surv_hi):
        support_count = 0
        contradict_count = 0
        med_devs, med_survs = [], []
        for _ in range(B):
            n_dev = rng.integers(12, 120, size=K_dev)
            n_surv = rng.integers(12, 120, size=K_surv)
            true_dev = rng.uniform(dev_lo, dev_hi, size=K_dev)
            true_surv = rng.uniform(surv_lo, surv_hi, size=K_surv)
            obs_dev = np.empty(K_dev)
            obs_surv = np.empty(K_surv)
            for i in range(K_dev):
                k = 1.0 / (true_dev[i] ** 2)
                obs_dev[i] = float(sample_cv(rng.gamma(k, 1.0 / k, size=int(n_dev[i])), corrected=True))
            for i in range(K_surv):
                k = 1.0 / (true_surv[i] ** 2)
                obs_surv[i] = float(sample_cv(rng.gamma(k, 1.0 / k, size=int(n_surv[i])), corrected=True))
            med_devs.append(float(np.median(obs_dev)))
            med_survs.append(float(np.median(obs_surv)))
            # Fast bootstrap CI (250 resamples) for median_dev and (median_surv - median_dev)
            idx_d = rng.integers(0, K_dev, size=(250, K_dev))
            idx_s = rng.integers(0, K_surv, size=(250, K_surv))
            b_md = np.median(obs_dev[idx_d], axis=1)
            b_ms = np.median(obs_surv[idx_s], axis=1)
            b_diff = b_ms - b_md
            if np.quantile(b_md, 0.975) < 0.50 and np.quantile(b_diff, 0.025) > 0.0:
                support_count += 1
            elif np.quantile(b_md, 0.025) >= 0.80 or np.quantile(b_diff, 0.975) < 0.0:
                contradict_count += 1
        return {
            "mean_median_dev_cv": float(np.mean(med_devs)),
            "mean_median_surv_cv": float(np.mean(med_survs)),
            "prob_supports": support_count / B,
            "prob_contradicts": contradict_count / B,
            "prob_not_decided": 1.0 - (support_count + contradict_count) / B,
        }

    scen_theory = eval_compilations(0.10, 0.35, 0.55, 1.15)
    scen_mild = eval_compilations(0.25, 0.50, 0.45, 0.85)
    scen_null_exp = eval_compilations(1.00, 1.00, 1.00, 1.00)
    scen_null_equal = eval_compilations(0.45, 0.75, 0.45, 0.75)

    # Transform non-identifiability from CV alone (Statement C2d):
    # Compare four discrete laws on fine-step malaria (F=8 fixed steps, E[L_long]=60 steps, sigma=1.4, phi_fine=0.75^(1/8))
    # matched to the SAME mean=60 and target CV: (a) Erlang chain, (b) fixed + geometric, (c) discrete lognormal, (d) two-point bimodal
    phi_f = 0.75 ** (1.0 / 8.0)
    theta_f = phi_f / (1.0 - phi_f)
    sigma_m = 1.40
    F_steps, L_mean = 8, 60
    T_tot = F_steps + L_mean
    transform_rows = []
    for m_chain in (2, 4, 15):
        # Exact discrete chain CV on L_mean=60
        cv_t = math.sqrt(1.0 / m_chain - 1.0 / L_mean)
        rho_chain = (1.0 + L_mean / (m_chain * theta_f)) ** (-m_chain)
        # Match fixed (L_mean - g) + geometric(g): Var = g*(g-1) = (cv_t * L_mean)^2 => g = 0.5*(1 + sqrt(1 + 4*V))
        target_var = (cv_t * L_mean) ** 2
        g_cont = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * target_var))
        # Discrete mixture of floor(g) and ceil(g) or exact continuous-g transform comparison:
        g_int = max(1, int(round(g_cont)))
        cv_fg = math.sqrt(g_int * (g_int - 1.0)) / L_mean
        rho_fg = (phi_f ** (L_mean - g_int)) * ((1.0 / g_int) * phi_f / (1.0 - (1.0 - 1.0 / g_int) * phi_f))
        # Two-point law on {L_lo, L_hi} with exact mean=60 and skewness +2 vs -1
        # Put p=0.9 at L1 = 60 - s/3, 0.1 at L2 = 60 + 3*s where Var = 0.9*(s/3)^2 + 0.1*(3s)^2 = s^2*(0.1 + 0.9) = s^2!
        s_sd = cv_t * L_mean
        l1 = max(1.0, L_mean - s_sd / 3.0)
        l2 = L_mean + 9.0 * (L_mean - l1)  # exact mean 60: 0.9*l1 + 0.1*l2 = 60
        var_2pt = 0.9 * (l1 - L_mean) ** 2 + 0.1 * (l2 - L_mean) ** 2
        cv_2pt = math.sqrt(var_2pt) / L_mean
        rho_2pt = 0.9 * (phi_f ** l1) + 0.1 * (phi_f ** l2)

        def d2_from_rho(rho_l: float) -> float:
            R = (phi_f ** F_steps) * rho_l
            return sigma_m * sigma_m * ((1.0 + 2.0 * theta_f) / (2.0 * T_tot) - R / (1.0 - R))

        d2_c = d2_from_rho(rho_chain)
        d2_fg = d2_from_rho(rho_fg)
        d2_2p = d2_from_rho(rho_2pt)
        transform_rows.append({
            "m_chain": m_chain,
            "cv_chain": cv_t,
            "D2_chain": d2_c,
            "cv_fixed_plus_geom": cv_fg,
            "D2_fixed_plus_geom": d2_fg,
            "cv_right_skew_2pt": cv_2pt,
            "D2_right_skew_2pt": d2_2p,
            "max_rel_spread_at_matched_cv": (max(d2_c, d2_fg, d2_2p) - min(d2_c, d2_fg, d2_2p)) / d2_c,
        })

    return {
        "H_dev1_scenarios": {
            "mechanistic_separation_dev010_035_surv055_115": scen_theory,
            "moderate_separation_dev025_050_surv045_085": scen_mild,
            "null_all_exponential_cv1": scen_null_exp,
            "null_no_stage_difference_cv045_075": scen_null_equal,
        },
        "transform_vs_cv_only": transform_rows,
    }


def main() -> int:
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    rng = np.random.default_rng(20261114)

    # Confirm blinding: raw directory must not exist or must be empty before this run
    raw_dir = ROOT / "data" / "raw" / "devtimes"
    raw_files = list(raw_dir.rglob("*")) if raw_dir.exists() else []
    if raw_files:
        raise RuntimeError(f"Blinding check failed: {raw_dir} already contains files!")

    checks = run_known_checks()
    p1 = sim_p1_sample_size_power(rng)
    p2 = sim_p2_identifiability_distortions(rng)
    p3 = sim_p3_compilation_power_and_transforms(rng)

    elapsed = time.time() - t0
    payload = {
        "timestamp": stamp,
        "elapsed_seconds": elapsed,
        "blinding_verified_empty_raw_dir": True,
        "known_checks": checks,
        "P1_sample_size_power": p1,
        "P2_identifiability_distortions": p2,
        "P3_compilation_power_and_transforms": p3,
    }
    (RESULTS / "power_identifiability.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"PRE-ANALYSIS POWER & IDENTIFIABILITY CHECK (case study: devtimes)",
        f"Run timestamp: {stamp}   Wall time: {elapsed:.2f} s",
        f"Blinding check: data/raw/devtimes/ has 0 files (PASS)",
        "",
        "=== KNOWN ANALYTICAL CHECKS (Rule 7) ===",
    ]
    for c in checks:
        lines.append(f"  [{ 'PASS' if c['pass'] else 'FAIL' }] {c['id']}: {c['claim']} (err={c['error']:.2e}, tol={c['tol']:.1e})")

    lines.append("\n=== P1. SINGLE-COHORT SAMPLE-SIZE IDENTIFIABILITY & POWER (4,000 trials/cell) ===")
    lines.append("  true_cv    n   mean_cor    bias    rmse   cov95  P(reject Exp)  P(CI_lo>0.102)  P(CI_lo>0.210)  P(CI_hi<0.480)")
    for r in p1:
        if r["n"] in (10, 20, 50, 100):
            lines.append(
                f"    {r['true_cv']:5.2f}  {r['n']:3d}     {r['mean_cv_cor']:6.3f}  {r['bias_cor']:+6.3f}  "
                f"{r['rmse_cor']:6.3f}   {r['ci95_coverage']:5.3f}          {r['power_reject_exponential']:5.3f}"
                f"           {r['power_certify_above_schisto_0102']:5.3f}           {r['power_certify_above_malaria_0210']:5.3f}"
                f"           {r['power_certify_below_malaria_m4_0480']:5.3f}"
            )

    lines.append("\n=== P2a. INTERVAL-CENSORING FLOOR (exact entry, check spacing dt_obs / mean) ===")
    lines.append("  true_cv  dt/mean  floor(dt/sqrt12)  naive_midpoint_cv  sheppard_debiased_cv")
    for r in p2["interval_censoring"]:
        lines.append(
            f"    {r['true_cv']:5.2f}    {r['dt_over_mean']:5.3f}             {r['floor_dt_sqrt12']:5.3f}"
            f"              {r['mean_cv_naive_midpoint']:5.3f}                 {r['mean_cv_sheppard_debiased']:5.3f}"
        )

    lines.append("\n=== P2b. RIGHT-CENSORING BIAS (n=80 Weibull cohort truncated at T_max / mean) ===")
    lines.append("  true_cv  T_max/mean  cens_frac  naive_completers_cv  weibull_censored_mle_cv")
    for r in p2["right_censoring"]:
        lines.append(
            f"    {r['true_cv']:5.2f}       {r['tmax_over_mean']:5.2f}      {r['censored_fraction']:5.3f}"
            f"                {r['naive_completers_cv']:5.3f}                    {r['weibull_censored_mle_cv']:5.3f}"
        )

    lines.append("\n=== P2c. MISATTRIBUTION DISTORTION (confusing vial-mean SD or SE with individual SD) ===")
    for r in p2["misattribution"]:
        lines.append(
            f"  true_cv={r['true_cv']:.2f}: vial-mean SD (m=10) gives apparent CV={r['apparent_cv_from_vial_means_m10']:.3f}; "
            f"mistaking SE (n=25) for SD gives {r['apparent_cv_if_SE_n25_mistaken_for_SD']:.3f}; "
            f"SE*sqrt(n)*(1+1/4n) recovers {r['recovered_cv_when_SE_multiplied_by_sqrtn']:.3f}"
        )

    lines.append("\n=== P3a. COMPILATION-LEVEL POWER FOR H_dev1 (K_dev=40, K_surv=25, 2,000 compilations) ===")
    for name, s in p3["H_dev1_scenarios"].items():
        lines.append(
            f"  {name:50s}: E[med_dev]={s['mean_median_dev_cv']:.3f}, E[med_surv]={s['mean_median_surv_cv']:.3f} | "
            f"P(supports)={s['prob_supports']:.3f}, P(contradicts)={s['prob_contradicts']:.3f}, P(not_decided)={s['prob_not_decided']:.3f}"
        )

    lines.append("\n=== P3b. TRANSFORM NON-IDENTIFIABILITY AT MATCHED (MEAN, CV) ON FINE-STEP MALARIA ===")
    for r in p3["transform_vs_cv_only"]:
        lines.append(
            f"  m={r['m_chain']:2d} (CV~{r['cv_chain']:.3f}): D2_chain={r['D2_chain']:.4f}, "
            f"D2_fixed+geom={r['D2_fixed_plus_geom']:.4f}, D2_skew2pt={r['D2_right_skew_2pt']:.4f} "
            f"(spread={100*r['max_rel_spread_at_matched_cv']:.1f}%)"
        )

    txt = "\n".join(lines) + "\n"
    (RESULTS / "power_identifiability.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
