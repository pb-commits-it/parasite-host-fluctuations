#!/usr/bin/env python3
"""Pre-analysis theoretical predictions and statistical power / identifiability simulations.

Run BEFORE opening any outcome data rows in `data/raw/microcosm/` (CASESTUDY_SPEC.md Rule 1 & Rule 7).
Command:
  python3 -I casestudies/microcosm/preanalysis_power.py

No file under `data/raw/` is opened by this script. Only experimental design dimensions
(replicate counts, sampling schedules, binomial sample sizes) and published literature
parameter ranges are used.
"""
from __future__ import annotations

import math
import os
import sys
import time
import numpy as np
from scipy import linalg, optimize, stats

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ENGINE_DIR = os.path.join(REPO_ROOT, "problems", "dobson-r0", "studies", "engine-R16")
sys.path.insert(0, ENGINE_DIR)
import lifecycle_engine as le  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")

# Pre-stated benchmark tolerances (Rule 7)
TOL_FLOQUET_REF = 5.0e-4
TOL_ZERO_DEATH = 1.0e-12
TOL_ENGINE_C1 = 1.0e-10
TOL_NULL_ALPHA = (0.035, 0.060)  # 5,000 Monte Carlo trials at alpha = 0.05


def build_ct_generator(tau_out: float, tau_in: float, k_out: int, k_in: int,
                       d_wait: float, h_val: float, y_tilt: float) -> np.ndarray:
    """Continuous-time generator for two-stage Erlang chain (outside wait b, host stay a).

    Matches `notes/decisive-experiment-v2.md` §3:
      - `k_out` equal outside sub-stages of total mean `tau_out` at h=1; only last sub-stage meets hosts.
      - `k_in` equal in-host sub-stages of total mean `tau_in`; last sub-stage releases offspring into outside[0].
      - Waste (`d_wait = 1.0`): last outside sub-stage leaves at fixed rate `k_out / tau_out`,
        and entry to host[0] is multiplied by `e^{y_tilt} * h_val`.
      - Wait (`0 <= d_wait < 1`): last outside sub-stage leaves at rate `(k_out / tau_out) * (d_wait + (1 - d_wait) * h_val)`,
        with entry to host[0] at rate `(k_out / tau_out) * e^{y_tilt} * h_val` (at threshold y_tilt=0, h=1, R0 = 1).
    """
    n = k_out + k_in
    Q = np.zeros((n, n), dtype=float)
    r_out = k_out / tau_out
    r_in = k_in / tau_in

    for i in range(k_out - 1):
        Q[i, i] = -r_out
        Q[i + 1, i] = r_out

    last_out = k_out - 1
    first_in = k_out
    if abs(d_wait - 1.0) < 1e-15:
        Q[last_out, last_out] = -r_out
    else:
        Q[last_out, last_out] = -r_out * (d_wait + (1.0 - d_wait) * h_val)
    Q[first_in, last_out] = r_out * math.exp(y_tilt) * h_val

    for j in range(k_in - 1):
        idx = k_out + j
        Q[idx, idx] = -r_in
        Q[idx + 1, idx] = r_in

    last_in = k_out + k_in - 1
    Q[last_in, last_in] = -r_in
    Q[0, last_in] = r_in
    return Q


def floquet_growth(tau_out: float, tau_in: float, k_out: int, k_in: int,
                   d_wait: float, period: float, h_high: float, h_low: float,
                   y_tilt: float, frac_high: float = 0.5) -> float:
    """Exact Floquet growth rate (per day) for a two-level periodic square wave."""
    Q_hi = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_tilt)
    Q_lo = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_tilt)
    M = linalg.expm(Q_lo * (period * (1.0 - frac_high))) @ linalg.expm(Q_hi * (period * frac_high))
    rho = float(np.max(np.abs(linalg.eigvals(M))))
    return math.log(rho) / period


def floquet_loss_D(tau_out: float, tau_in: float, k_out: int = 1, k_in: int = 1,
                   d_wait: float = 1.0, period: float = 4.0,
                   h_high: float = 1.8, h_low: float = 0.2) -> float:
    """Solve for threshold tilt y = D where Floquet growth rate is zero."""
    sol = optimize.brentq(
        lambda y: floquet_growth(tau_out, tau_in, k_out, k_in, d_wait, period, h_high, h_low, y),
        -2.5, 6.0, xtol=1e-11
    )
    return float(sol)


def markov_daily_loss_D(tau_out: float, tau_in: float, k_out: int, k_in: int,
                        d_wait: float, phi_day: float, h_high: float, h_low: float,
                        horizon_days: int = 4000, n_pairs: int = 40, seed: int = 20261115) -> tuple[float, float]:
    """Threshold loss D for a symmetric two-state daily Markov environment with lag-1 correlation phi_day."""
    rng = np.random.default_rng(seed)
    p_stay = 0.5 * (1.0 + phi_day)
    states = np.empty((horizon_days, 2 * n_pairs), dtype=int)
    u0 = rng.random(n_pairs) < 0.5
    states[0, :n_pairs] = u0.astype(int)
    states[0, n_pairs:] = 1 - states[0, :n_pairs]
    flips = rng.random((horizon_days - 1, n_pairs)) >= p_stay
    for t in range(1, horizon_days):
        states[t, :n_pairs] = np.where(flips[t - 1], 1 - states[t - 1, :n_pairs], states[t - 1, :n_pairs])
        states[t, n_pairs:] = 1 - states[t, :n_pairs]

    burn = 200

    def eval_growth(y: float) -> tuple[float, np.ndarray]:
        M_hi = linalg.expm(build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y))
        M_lo = linalg.expm(build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y))
        dim = k_out + k_in
        x = np.full((2 * n_pairs, dim), 1.0 / dim)
        logs = np.zeros(2 * n_pairs)
        for t in range(horizon_days):
            hi_mask = states[t] == 1
            if np.any(hi_mask):
                x[hi_mask] = x[hi_mask] @ M_hi.T
            if np.any(~hi_mask):
                x[~hi_mask] = x[~hi_mask] @ M_lo.T
            s = x.sum(axis=1)
            x /= s[:, None]
            if t >= burn:
                logs += np.log(s)
        logs /= (horizon_days - burn)
        pair_means = 0.5 * (logs[:n_pairs] + logs[n_pairs:])
        return float(pair_means.mean()), pair_means

    y_root = optimize.brentq(lambda y: eval_growth(y)[0], -1.5, 4.0, xtol=1e-6)
    r_at_root, pm_0 = eval_growth(y_root)
    r_plus, _ = eval_growth(y_root + 1e-4)
    slope = (r_plus - r_at_root) / 1e-4
    se = float(pm_0.std(ddof=1) / math.sqrt(n_pairs) / max(abs(slope), 1e-8))
    return float(y_root), se


def c1_closed_form(tau_1: float, tau_2: float, phi: float, d_wait: float, sigma: float) -> float:
    """Closed-form second-order loss per generation C1 (`notes/contact-limited-transfer.md` eq. C1)."""
    T = tau_1 + tau_2
    if abs(phi) < 1e-15:
        return sigma * sigma * (d_wait * d_wait) / (2.0 * T)
    theta = phi / (1.0 - phi)
    q1 = 1.0 / (1.0 + tau_1 / theta)
    q2 = 1.0 / (1.0 + tau_2 / theta)
    R = q1 * q2
    return sigma * sigma * (d_wait * d_wait * (1.0 + 2.0 * theta) / (2.0 * T)
                            - d_wait * (R - (1.0 - d_wait) * q1) / (1.0 - R))


def run_benchmarks(lines: list[str]) -> None:
    lines.append("=== SECTION 0. BENCHMARK VERIFICATION (study rules Rule 7) ===")
    # B0a: Floquet D vs decisive-experiment-v2.md (b=10, a=5, P=30d [2 gen], swing 1.8/0.2)
    d_waste_10 = floquet_loss_D(10.0, 5.0, 1, 1, 1.0, 30.0, 1.8, 0.2)
    d_wait05_10 = floquet_loss_D(10.0, 5.0, 1, 1, 0.5, 30.0, 1.8, 0.2)
    err_a1 = abs(d_waste_10 - 0.0920)
    err_a2 = abs(d_wait05_10 - 0.0570)
    assert err_a1 <= TOL_FLOQUET_REF and err_a2 <= 1.0e-3, f"B0a failed: {d_waste_10}, {d_wait05_10}"
    lines.append(f"B0a PASS: Floquet (b=10, a=5, P=30, 1.8/0.2) waste D={d_waste_10:.6f} (ref 0.0920, diff={err_a1:.2e} <= {TOL_FLOQUET_REF:.1e}); wait d=0.5 D={d_wait05_10:.6f} (ref 0.057, diff={err_a2:.2e})")

    # B0b: Zero waiting mortality (d = 0) gives exact D = 0
    d_zero = floquet_loss_D(10.0, 5.0, 1, 1, 0.0, 30.0, 1.8, 0.2)
    assert abs(d_zero) <= TOL_ZERO_DEATH, f"B0b failed: {d_zero}"
    lines.append(f"B0b PASS: Contact-limited d=0 exact Floquet loss D={d_zero:.3e} (tol <= {TOL_ZERO_DEATH:.1e})")

    # B0c: lifecycle_engine.py coefficient() vs closed-form C1
    tau1, tau2, phi, d_val, sig = 3.0, 12.0, 0.8, 0.5, 0.25
    # Stage 1 contact-limited: mean = 3.0 => hazard = 1/3; d = death / hazard => death = d / 3
    desc = le.Description(
        stages=[
            le.Stage("free", mean=tau1, shape="memoryless", death=d_val / tau1, contact=True),
            le.Stage("host", mean=tau2, shape="memoryless", death=0.0, contact=False),
        ],
        transfers=[
            le.Transfer("free", "host", yield_=1.0 / (1.0 - d_val), driver="H", birth=False),
            le.Transfer("host", "free", yield_=1.0, driver=None, birth=True),
        ],
        drivers=[le.Driver("H", sigma=sig, phi=phi, kind="bounded", low=0.4)],
    )
    eng = le.Engine(desc)
    d2_eng = eng.coefficient()["D2"]
    d2_c1 = c1_closed_form(tau1, tau2, phi, d_val, sig)
    rel_c1 = abs(d2_eng - d2_c1) / abs(d2_c1)
    assert rel_c1 <= TOL_ENGINE_C1, f"B0c failed: {d2_eng} vs {d2_c1}"
    lines.append(f"B0c PASS: lifecycle_engine D2={d2_eng:.12f} vs closed-form C1={d2_c1:.12f} (rel diff={rel_c1:.2e} <= {TOL_ENGINE_C1:.1e})")

    # B0d: Calibration of exact/Monte-Carlo permutation test under null (alpha = 0.05)
    rng = np.random.default_rng(12345)
    n_mc = 5000
    x_levels = np.repeat([0.0, 0.1, 0.3, 0.5, 0.8], 6)
    rx = stats.rankdata(x_levels)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    fp = 0
    for _ in range(n_mc):
        y = rng.standard_normal(30)
        ry = stats.rankdata(y)
        ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
        r_obs = float(rx @ ry)
        idx = np.argsort(rng.random((499, 30)), axis=1)
        r_perm = ry[idx] @ rx
        p_val = (1 + int(np.sum(r_perm <= r_obs))) / 500.0
        if p_val < 0.05:
            fp += 1
    alpha_hat = fp / n_mc
    assert TOL_NULL_ALPHA[0] <= alpha_hat <= TOL_NULL_ALPHA[1], f"B0d failed: {alpha_hat}"
    lines.append(f"B0d PASS: Null false-positive rate over {n_mc} trials = {alpha_hat:.4f} (within {TOL_NULL_ALPHA})")
    lines.append("")


def study1_ferris_power(lines: list[str]) -> None:
    lines.append("=== SECTION 1. STUDY 1 (Ferris et al. 2020: P. fluorescens - phage SBW25phi2) ===")
    lines.append("Design: 5 resource oscillation amplitudes at fixed mean 0.50 KB, n=6 reps/arm (N=30),")
    lines.append("        period P = 4.0 d (alternating every 48 h transfer for 24 transfers = 48 d).")

    # Published/biological phage cycle ranges:
    # Latent period tau_in ~ 0.03 d (43 min); effective free-virion residence tau_out ~ 0.35 d
    # (bounded above by 99% serial dilution every 2 d -> 2 / ln(100) = 0.434 d, plus free decay).
    tau_out, tau_in = 0.35, 0.03
    period = 4.0
    # Illustrative host swings delta_H corresponding to resource swings (0, 0.50, 0.80, 0.90, 0.98)
    amplitudes_res = [0.00, 0.50, 0.80, 0.90, 0.98]
    swings_H = [0.00, 0.35, 0.65, 0.80, 0.92]

    lines.append(f"Nominal stage durations: tau_out={tau_out:.2f} d, tau_in={tau_in:.2f} d (T={tau_out+tau_in:.2f} d, P/T={period/(tau_out+tau_in):.1f} gen/cycle)")
    lines.append("Predicted Floquet loss D across resource amplitudes for gain-only (d=1) vs contact-limited (d=0.5, 0.1, 0.03):")
    D_table = {}
    for d_val in [1.0, 0.5, 0.1, 0.03]:
        row = [floquet_loss_D(tau_out, tau_in, 1, 1, d_val, period, 1.0 + dh, 1.0 - dh) if dh > 0 else 0.0
               for dh in swings_H]
        D_table[d_val] = np.array(row)
        lines.append(f"  d={d_val:4.2f}: " + ", ".join(f"amp {a:.2f}->D={v:.4f}" for a, v in zip(amplitudes_res, row)))

    # Pre-data Monte Carlo power simulation for T1a (phage survival transfer) and T1b (early log PFU slope)
    # Sample size: 5 treatments x 6 replicates = 30 microcosms.
    # Baseline phage extinction hazard per 3-transfer interval in constant control: p_ext0 ~ 0.05 (so ~26% die by T24, ~1.5/6)
    # Under theory, effective log-R0 margin drops by D(trt); suppose baseline margin M0 = 0.60,
    # and per-interval extinction hazard scales as logistic(-3.0 * (M0 - D(trt))) calibrated to match p_ext0=0.05 at D=0.
    rng = np.random.default_rng(20261115)
    n_sim = 4000
    x_trt = np.repeat(np.arange(5), 6)

    lines.append("\nPre-data Power Simulation for T1a (one-sided Spearman trend on survival transfer, N=30, alpha=0.05):")
    for label, d_vec in [
        ("Null (no fluctuation effect, D=0)", np.zeros(5)),
        ("Contact-limited fast adsorption (d=0.03)", D_table[0.03]),
        ("Contact-limited moderate loss (d=0.10)", D_table[0.10]),
        ("Contact-limited half-loss (d=0.50)", D_table[0.50]),
        ("Gain-only waste (d=1.00, baseline M0=0.8)", D_table[1.0]),
    ]:
        D_rep = np.repeat(d_vec, 6)
        # Calibrate per-check-interval (8 intervals: T3..T24) hazard: at D=0, p_step = 0.045 (~31% extinct by T24)
        logit0 = math.log(0.045 / (1.0 - 0.045))
        # Sensitivity of log-odds of extinction to threshold loss D: slope_k = 3.5 (strong demographic link)
        p_step = 1.0 / (1.0 + np.exp(-(logit0 + 3.5 * D_rep)))
        support_cnt = 0
        contra_cnt = 0
        rhos = []
        for _ in range(n_sim):
            # Simulate survival across 8 observation intervals (transfers 3,6,...,24)
            surv_time = np.full(30, 24.0)
            alive = np.ones(30, dtype=bool)
            for step_idx, tr_num in enumerate([3, 6, 9, 12, 15, 18, 21, 24]):
                dies = alive & (rng.random(30) < p_step)
                surv_time[dies] = tr_num
                alive[dies] = False
            if np.all(surv_time == surv_time[0]):
                rhos.append(0.0)
                continue
            r_obs, p_two = stats.spearmanr(D_rep if np.any(D_rep > 0) else x_trt, surv_time)
            rhos.append(r_obs)
            # Exact/asymptotic one-sided check (calibrated against permutation in B0d)
            if r_obs < 0 and (p_two / 2.0) < 0.05:
                support_cnt += 1
            elif r_obs > 0 and (p_two / 2.0) < 0.05:
                contra_cnt += 1
        lines.append(f"  {label:44s}: mean rho={np.mean(rhos):+.3f}, P(supports)={support_cnt/n_sim:.3f}, P(contradicts)={contra_cnt/n_sim:.3f}")

    # Power for T1b: early log-PFU (mean of Transfer 3 and Transfer 6) with replicate SD sigma_y = 0.8 or 1.5 (log-PFU units)
    lines.append("\nPre-data Power Simulation for T1b (OLS slope of early mean ln(PFU) on amplitude/D, N=30, alpha=0.05):")
    for sigma_y in [0.6, 1.2]:
        for label, d_vec in [
            ("Null (D=0)", np.zeros(5)),
            ("Contact-limited (d=0.10, 2 gen shift)", D_table[0.10]),
            ("Gain-only (d=1.00, 2 gen shift)", D_table[1.0]),
        ]:
            # Over one 2-transfer cycle (4 d ~ 10.5 phage generations), cumulative log deficit if unregulated ~ 4 * D,
            # or if partially regulated by carrying capacity, say 2.0 * D:
            D_rep = np.repeat(d_vec, 6)
            x_pred = D_rep if np.any(D_rep > 0) else x_trt.astype(float)
            sup_b, con_b = 0, 0
            for _ in range(n_sim):
                y_sim = 15.0 - 2.0 * D_rep + rng.normal(0.0, sigma_y, size=30)
                res = stats.linregress(x_pred, y_sim)
                if res.slope < 0 and (res.pvalue / 2.0) < 0.05:
                    sup_b += 1
                elif res.slope > 0 and (res.pvalue / 2.0) < 0.05:
                    con_b += 1
            lines.append(f"  sigma_lnPFU={sigma_y:.1f} | {label:38s}: P(supports)={sup_b/n_sim:.3f}, P(contradicts)={con_b/n_sim:.3f}")
    lines.append("")


def study2_duncan_power(lines: list[str]) -> None:
    lines.append("=== SECTION 2. STUDY 2 (Duncan, Gonzalez & Kaltz 2013: Paramecium - Holospora) ===")
    lines.append("Design: 8 variable cells = Red (phi=0.8) vs White (phi=0.0) x Sync vs Async (2 patches)")
    lines.append("        x High (0.0667) vs Low (0.0067) migration, n=4 infected + 4 uninfected reps/cell.")
    lines.append("        Sampled at Days 0, 4, 8, 16, 24, 32, 48, 56 (~45 hosts examined per sample).")

    # Biological stage durations at 23 C: tau_in = 7.0 d (Zilio et al. 2023), tau_out = 2.0 d (Fels & Kaltz 2006)
    tau_out, tau_in = 2.0, 7.0
    lines.append(f"Nominal Holospora stage durations: tau_out={tau_out:.1f} d (free spore), tau_in={tau_in:.1f} d (in host), T={tau_out+tau_in:.1f} d.")

    # 1) Continuous-time daily Markov switching (h = 1 +/- 0.6, matching empirical host CV ~ 0.6)
    for d_val in [1.0, 0.5, 0.1]:
        d_white, se_w = markov_daily_loss_D(tau_out, tau_in, 1, 1, d_val, 0.0, 1.6, 0.4, horizon_days=3000, n_pairs=24, seed=101)
        d_red, se_r = markov_daily_loss_D(tau_out, tau_in, 1, 1, d_val, 0.8, 1.6, 0.4, horizon_days=3000, n_pairs=24, seed=102)
        lines.append(f"  1-patch daily Markov (h=1+/-0.6, memoryless, d={d_val:.1f}): D_white={d_white:.4f}+/-{se_w:.4f}, D_red={d_red:.4f}+/-{se_r:.4f}, diff(Red-White)={d_red-d_white:+.4f}")

    # Erlang (4, 4) stays
    dw_e4, sew4 = markov_daily_loss_D(tau_out, tau_in, 4, 4, 1.0, 0.0, 1.6, 0.4, horizon_days=3000, n_pairs=24, seed=103)
    dr_e4, ser4 = markov_daily_loss_D(tau_out, tau_in, 4, 4, 1.0, 0.8, 1.6, 0.4, horizon_days=3000, n_pairs=24, seed=104)
    lines.append(f"  1-patch daily Markov (h=1+/-0.6, Erlang 4,4, d=1.0): D_white={dw_e4:.4f}+/-{sew4:.4f}, D_red={dr_e4:.4f}+/-{ser4:.4f}, diff(Red-White)={dr_e4-dw_e4:+.4f}")

    # 2) Two-patch discrete engine predictions (1-day step) across White/Red x Sync/Async x Low/High migration
    lines.append("\n2-patch `lifecycle_engine.py` predictions (1-d step, sigma=0.6, bounded low=0.3, gain-only):")
    cell_D = {}
    for noise_name, phi_val in [("White", 0.0), ("Red", 0.8)]:
        for sync_name, sync_val in [("Sync", 1.0), ("Async", 0.0)]:
            for mig_name, mig_val in [("Low", 0.0067), ("High", 0.0667)]:
                desc = le.Description(
                    stages=[
                        le.Stage("spore", mean=tau_out, shape="memoryless"),
                        le.Stage("host", mean=tau_in, shape="memoryless"),
                    ],
                    transfers=[
                        le.Transfer("spore", "host", yield_=1.0, driver="H", birth=False),
                        le.Transfer("host", "spore", yield_=1.0, driver=None, birth=True),
                    ],
                    drivers=[le.Driver("H", sigma=0.6, phi=phi_val, kind="bounded", low=0.3)],
                    patches=le.Patches(count=2, synchrony=sync_val, mobile_stage="host", movement=mig_val, topology="island"),
                )
                eng = le.Engine(desc)
                c_out = eng.coefficient()
                l_out = eng.loss(steps=12000, burn=600, pairs=16, seed=301)
                cell_D[(noise_name, sync_name, mig_name)] = l_out["D"]
                lines.append(f"  {noise_name:5s} | {sync_name:5s} | Mig={mig_name:4s}: D2={c_out['D2']:+.4f}, sim D={l_out['D']:+.4f}+/-{l_out['se']:.4f}")

    # Power simulation for T2a (early rare-growth rate Days 0->16, Red vs White, n=16 vs 16)
    # and T2c (Async vs Sync under Red noise, n=8 vs 8)
    # Over 16 days (~1.78 generations of T=9 d), difference in threshold D of delta_D ~ 0.12 (gain-only)
    # translates to growth rate difference delta_r = delta_D / T ~ 0.0133 d^-1, or cumulative log-growth ~ 0.213 over 16 days.
    # HOWEVER, random 16-day schedules of Red noise (phi=0.8, mean dwell 5 d) have huge realization-to-realization
    # variance in number of permissive days (~SD = sqrt(16 * 0.25 * (1+0.8)/(1-0.8)) = 6.0 days out of 16!),
    # plus binomial sampling error on ~45 examined hosts at low prevalence (p0=0.025 -> p16~0.15)!
    rng = np.random.default_rng(20261116)
    n_sim = 4000
    lines.append("\nPre-data Power Simulation for T2a (Early rare growth Day 0->16, Red n=16 vs White n=16, alpha=0.05)")
    lines.append("  Includes BOTH random 16-day Markov path variance AND binomial sampling error (n_exam=45 hosts/sample):")

    for scen_name, delta_D, path_sd in [
        ("Null (no difference, biological+schedule SD=0.35)", 0.00, 0.35),
        ("Theory host-only (delta_D=+0.11 over T=9d, schedule SD=0.35)", 0.11, 0.35),
        ("Theory host+thermal arrest (delta_D=+0.35 over T=9d, schedule SD=0.55)", 0.35, 0.55),
        ("Strong thermal-arrest penalty (delta_D=+0.80 over T=9d, schedule SD=0.55)", 0.80, 0.55),
    ]:
        sup_cnt, con_cnt = 0, 0
        diffs = []
        # Baseline growth from p0=0.03 to ~0.20 over 16 days: log-increase ~ 1.90 in White
        base_log_inc = 1.90
        effect_16d = (16.0 / 9.0) * delta_D  # expected reduction in Red over 16 days
        for _ in range(n_sim):
            # 16 White and 16 Red metapopulations
            true_p0_w = np.clip(rng.normal(0.03, 0.005, 16), 0.005, 0.2)
            true_p0_r = np.clip(rng.normal(0.03, 0.005, 16), 0.005, 0.2)
            # White has smaller 16-d schedule variance than Red
            lp16_w = np.log(true_p0_w / (1 - true_p0_w)) + base_log_inc + rng.normal(0.0, path_sd * 0.6, 16)
            lp16_r = np.log(true_p0_r / (1 - true_p0_r)) + (base_log_inc - effect_16d) + rng.normal(0.0, path_sd, 16)
            p16_w = 1.0 / (1.0 + np.exp(-lp16_w))
            p16_r = 1.0 / (1.0 + np.exp(-lp16_r))
            # Binomial sampling of n_exam = 45 hosts at Day 0 and Day 16
            k0_w = rng.binomial(45, true_p0_w)
            k16_w = rng.binomial(45, p16_w)
            k0_r = rng.binomial(45, true_p0_r)
            k16_r = rng.binomial(45, p16_r)
            r_w = (np.log((k16_w + 0.5) / 46.0) - np.log((k0_w + 0.5) / 46.0)) / 16.0
            r_r = (np.log((k16_r + 0.5) / 46.0) - np.log((k0_r + 0.5) / 46.0)) / 16.0
            diffs.append(float(r_r.mean() - r_w.mean()))
            t_stat, p_two = stats.ttest_ind(r_r, r_w, equal_var=False)
            if t_stat < 0 and (p_two / 2.0) < 0.05:
                sup_cnt += 1
            elif t_stat > 0 and (p_two / 2.0) < 0.05:
                con_cnt += 1
        lines.append(f"  {scen_name:68s}: mean(r_R-r_W)={np.mean(diffs):+.4f}/d, P(supports)={sup_cnt/n_sim:.3f}, P(contradicts)={con_cnt/n_sim:.3f}")

    # Power for T2c: Spatial rescue (Async n=8 vs Sync n=8 under Red noise)
    delta_D_spatial = cell_D[("Red", "Sync", "High")] - cell_D[("Red", "Async", "High")]
    lines.append(f"\nPredicted spatial rescue under Red+HighMig: D_sync - D_async = {delta_D_spatial:+.4f} (16-d log shift ~ {(16/9)*delta_D_spatial:+.4f})")
    for label, eff_log in [("Null (0)", 0.0), (f"Engine spatial rescue (+{(16/9)*delta_D_spatial:.3f})", (16.0 / 9.0) * delta_D_spatial)]:
        sup_s = 0
        for _ in range(n_sim):
            y_sync = rng.normal(0.0, 0.45, 8)
            y_async = rng.normal(eff_log, 0.45, 8)
            t_st, p2 = stats.ttest_ind(y_async, y_sync, equal_var=False)
            if t_st > 0 and (p2 / 2.0) < 0.05:
                sup_s += 1
        lines.append(f"  T2c Spatial power (n=8 vs 8) | {label:36s}: P(supports)={sup_s/n_sim:.3f}")
    lines.append("")


def study3_dallas_power(lines: list[str]) -> None:
    lines.append("=== SECTION 3. STUDY 3 (Dallas, Krkosek & Drake 2018: D. dentifera - M. bicuspidata) ===")
    lines.append("Design: 6 initial host densities (20, 40, 80, 160, 320, 640 hosts/L), n=15 microcosms/arm (N=90).")
    lines.append("Primary use: calibrate host-density threshold H_c, transmission rate, and endogenous sigma_lnH & r_1.")
    lines.append("Secondary observational check T3a: partial correlation of epidemic AUC with endogenous sigma_lnH")
    lines.append("  controlling for mean ln(H) in supercritical arms (160, 320, 640 hosts/L; n=45).")

    # Published Metschnikowia parameters (Dallas et al. 2018 Table 1; Dallas & Drake 2016):
    # tau_in ~ 13 d (death ~11-16 d); tau_out: mu=0.75/d in Table 1 (1.33 d) vs half-life 11 d in 2016 (tau_out ~ 15.9 d).
    for tau_out_m in [1.33, 15.9]:
        d_m_waste = markov_daily_loss_D(tau_out_m, 13.0, 1, 4, 1.0, 0.7, 1.5, 0.5, horizon_days=2500, n_pairs=16, seed=401)[0]
        d_m_wait = markov_daily_loss_D(tau_out_m, 13.0, 1, 4, 0.5, 0.7, 1.5, 0.5, horizon_days=2500, n_pairs=16, seed=402)[0]
        lines.append(f"  Predicted Metschnikowia D at sigma_H~0.5, phi_day=0.7 (tau_out={tau_out_m:5.2f} d, tau_in=13 d Erlang-4): waste D={d_m_waste:.4f}, wait(d=0.5) D={d_m_wait:.4f}")

    # Power simulation for T3a across n=45 supercritical microcosms where endogenous sigma_lnH varies only modestly (e.g. 0.35 +/- 0.12)
    rng = np.random.default_rng(20261117)
    n_sim = 4000
    for scen, slope_D in [("Null (no sigma effect)", 0.0), ("Theory (endogenous sigma range SD=0.12, waste)", -0.35), ("Confounded epidemic crash (+feedback)", +0.80)]:
        sup_cnt, con_cnt = 0, 0
        for _ in range(n_sim):
            ln_hbar = rng.normal(5.8, 0.45, 45)
            sig_h = np.clip(rng.normal(0.45, 0.12, 45), 0.10, 1.10)
            # Epidemic log-AUC driven strongly by mean host density + demographic noise (SD=0.75) + sigma effect
            y_auc = -1.5 + 0.9 * (ln_hbar - 5.8) + slope_D * (sig_h - 0.45) + rng.normal(0.0, 0.75, 45)
            # Partial correlation of y_auc with sig_h given ln_hbar
            res_y = y_auc - np.poly1d(np.polyfit(ln_hbar, y_auc, 1))(ln_hbar)
            res_s = sig_h - np.poly1d(np.polyfit(ln_hbar, sig_h, 1))(ln_hbar)
            r_p, p2 = stats.pearsonr(res_s, res_y)
            if r_p < 0 and (p2 / 2.0) < 0.05:
                sup_cnt += 1
            elif r_p > 0 and (p2 / 2.0) < 0.05:
                con_cnt += 1
        lines.append(f"  T3a Power (n=45 supercritical) | {scen:45s}: P(supports)={sup_cnt/n_sim:.3f}, P(contradicts)={con_cnt/n_sim:.3f}")
    lines.append("")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    lines = [
        "Pre-analysis Theoretical Predictions and Simulation Power Report (`microcosm`)",
        f"Generated UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "Data files in `data/raw/microcosm/` opened by this script: NONE.",
        "",
    ]
    run_benchmarks(lines)
    study1_ferris_power(lines)
    study2_duncan_power(lines)
    study3_dallas_power(lines)
    lines.append(f"Completed in {time.time() - t0:.2f} s.")

    out_text = "\n".join(lines) + "\n"
    out_path = os.path.join(RESULTS_DIR, "preanalysis_power_20261115.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(out_text)
    print("Saved:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
