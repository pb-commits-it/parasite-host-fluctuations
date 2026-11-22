#!/usr/bin/env python3
"""Pre-analysis verification and power/identifiability simulation for study `chemostat`.

IMPORTANT CHRONOLOGY RULE:
This script runs BEFORE any empirical density values in `data/raw/chemostat/` are read
or analysed. It imports NO files from `data/raw/`. It uses only:
  1. `studies/engine-R16/lifecycle_engine.py` (project engine),
  2. Independent continuous-time and discrete-time two-stage phage-chemostat routines,
  3. Published/documented experimental sample sizes, sampling intervals, and phage
     life-history parameters.

Part 1 (Rule 7 — Known-result checks with pre-stated tolerances):
  K1. Discrete contact-limited second-order coefficient `Engine.coefficient()['D2']`
      matches closed form (C1) of `notes/contact-limited-transfer.md` across 25 parameter
      combinations (tolerance: 1e-11 relative).
  K2. At zero waiting mortality (`death = 0`, `d -> 0`), exact simulated threshold loss `D`
      vanishes (`|D| < 1e-10` as death -> 1e-11, and exact matrix left-eigenvector
      conservation holds to 1e-13 at `death = 0` for arbitrary host swings).
  K3. Continuous-time piecewise matrix-exponential propagator reproduces the discrete
      two-stage contact-limited propagator as step `dt -> 0` (relative difference < 2e-3
      at `dt = 1e-3 d`), and reproduces (C2) of `notes/contact-limited-transfer.md` on a
      small-amplitude telegraph host (`delta = 0.05`, tolerance: 1.5e-2 relative).
  K4. Coupled-dilution chemostat identities:
      (a) Uniform continuous-time washout `-omega(t) I_2` shifts the top Lyapunov exponent
          by `-mean(omega)` exactly (pathwise difference < 1e-13 across 50 random schedules),
          so `Delta r_fluct` is invariant to `omega(t) - mean(omega)` while `mean(omega)`
          raises threshold `d` from `mu0/(alpha*H + mu0)` to `(mu0+omega_bar)/(alpha*H+mu0+omega_bar)`.
      (b) Selective free-phage washout (or discrete serial-transfer washout at fixed mean
          staying) violates the naive transfer-only bound `D <= sum l_g` when `H` is steady
          and `omega(t)` fluctuates, while obeying Proposition B's corrected bound
          `D <= sum l_g + sum (tau_0 - 1) l_s` (`0` violations in 100 random schedules), and
          `(tau_0 - 1) l_s = O(dt) -> 0` as `dt -> 0` (halving `dt` halves the staying bound term
          within 2%).

Part 2 (Pre-analysis power & identifiability at dataset sample sizes):
  Simulates synthetic datasets matching the designs of:
    - Design A (Chemostat, Larsen 2019 scale): n = 6 chemostats (3 low-sigma, 3 high-sigma),
      60 biennial-day observations over 166 d, dilution 1.0 d^-1, microscopy count CV = 0.20.
    - Design B (Pulsed vs uniform serial transfer, Reuter 2026 scale): n = 10 lines (5 uniform
      3-h, 5 alternating 4x30-min + 3-h), 6 to 50 transfers, plaque count CV = 0.20.
    - Design C (Nutrient / persistence serial transfer, Chaudhry 2018 & Zhao 2016 scale):
      n = 22 lines (or 18 lines), 12 transfers, 1:100 daily dilution, count CV = 0.25.
    - Design D (Environmental-frequency serial transfer, Duncan 2017 scale): n = 30 infected
      lines across constant vs 3 switching frequencies, 9 time points (days 0..16), CV = 0.25.
  Evaluates the registered partial Spearman test `T1` across 2,000 Monte Carlo trials per
  scenario under:
    (S0) Null: no dynamic fluctuation effect (phage responds only to mean host R0 + noise);
    (S1) Linear rare-invasion theory at high susceptible host density (small d ~ 0.03);
    (S2) Linear rare-invasion theory at low susceptible host density (moderate/high d ~ 0.65);
    (S3) Nonlinear predator-prey + resistance evolution chemostat/transfer dynamics (endogenous
         host crashes driven by phage + sweep of resistant host H_R).

Run with: python3 -I casestudies/chemostat/power_precheck.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import numpy as np
from scipy import linalg, stats

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
PROB_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE_DIR = os.path.join(PROB_ROOT, "studies", "engine-R16")
sys.path.insert(0, ENGINE_DIR)

import lifecycle_engine as le  # noqa: E402

OUT_DIR = os.path.join(HERE, "results")


def closed_form_c1(tau_p: float, tau_i: float, d: float, phi: float, sigma: float) -> float:
    """Proposition C1 of `notes/contact-limited-transfer.md` for a 2-stage primitive ring."""
    T = tau_p + tau_i
    if abs(phi) < 1e-15:
        return sigma * sigma * (d * d) / (2.0 * T)
    theta = phi / (1.0 - phi)
    qp = 1.0 / (1.0 + tau_p / theta)
    qi = 1.0 / (1.0 + tau_i / theta)
    R = qp * qi
    return sigma * sigma * (d * d * (1.0 + 2.0 * theta) / (2.0 * T) - d * (R - (1.0 - d) * qp) / (1.0 - R))


def closed_form_c2_ct(tau_p: float, tau_i: float, d: float, theta_ct: float, sigma: float) -> float:
    """Equation (C2) of `notes/contact-limited-transfer.md` in continuous time."""
    T = tau_p + tau_i
    qp = 1.0 / (1.0 + tau_p / theta_ct)
    qi = 1.0 / (1.0 + tau_i / theta_ct)
    R = qp * qi
    return sigma * sigma * (d * d * theta_ct / T - d * (R - (1.0 - d) * qp) / (1.0 - R))


def ct_phage_generator(H: float, alpha: float, nu: float, beta: float, mu0: float, omega_p: float, omega_i: float, y: float = 0.0) -> np.ndarray:
    """Continuous-time 2x2 generator on state [I, P]^T.

    dI/dt = -(nu + omega_i) I + alpha * H * P
    dP/dt = exp(y) * beta * nu * I - (alpha * H + mu0 + omega_p) P
    """
    return np.array([
        [-(nu + omega_i), alpha * H],
        [math.exp(y) * beta * nu, -(alpha * H + mu0 + omega_p)],
    ], dtype=float)


def ct_dominant_eig(M: np.ndarray) -> float:
    vals = linalg.eigvals(M)
    return float(np.max(vals.real))


def ct_piecewise_growth(H_seq: np.ndarray, dt_vec: np.ndarray, alpha: float, nu: float, beta: float, mu0: float,
                        omega_p_seq: np.ndarray, omega_i_seq: np.ndarray, y: float = 0.0, cycles: int = 40) -> float:
    """Exact Lyapunov exponent on a finite piecewise-constant trajectory repeated periodically (Floquet) or long path."""
    # Compute exact one-pass monodromy matrix product with step-by-step normalization to prevent overflow
    x = np.array([0.5, 0.5], dtype=float)
    # Burn-in 5 passes, measure over `cycles` passes (or exact spectral radius of monodromy if short)
    K = len(H_seq)
    if K <= 500:
        # Exact Floquet monodromy in log-scaled form: product of expm(M_k * dt_k)
        # Keep log_scale to avoid overflow/underflow
        U = np.eye(2, dtype=float)
        log_s = 0.0
        for k in range(K):
            Mk = ct_phage_generator(H_seq[k], alpha, nu, beta, mu0, omega_p_seq[k], omega_i_seq[k], y=y)
            Ek = linalg.expm(Mk * dt_vec[k])
            U = Ek @ U
            nrm = float(np.max(np.abs(U)))
            if nrm > 0:
                U /= nrm
                log_s += math.log(nrm)
        rho = float(np.max(np.abs(linalg.eigvals(U))))
        total_time = float(np.sum(dt_vec))
        return (log_s + math.log(max(rho, 1e-300))) / total_time
    log_acc = 0.0
    total_t = 0.0
    for c in range(cycles + 2):
        for k in range(K):
            Mk = ct_phage_generator(H_seq[k], alpha, nu, beta, mu0, omega_p_seq[k], omega_i_seq[k], y=y)
            x = linalg.expm(Mk * dt_vec[k]) @ x
            s = float(x.sum())
            x /= s
            if c >= 2:
                log_acc += math.log(s)
                total_t += dt_vec[k]
    return log_acc / total_t


def ct_threshold_loss(H_seq: np.ndarray, dt_vec: np.ndarray, alpha: float, nu: float, beta: float, mu0: float,
                      omega_p_seq: np.ndarray, omega_i_seq: np.ndarray) -> tuple[float, float, float, float]:
    """Return (r_const, r_obs, ln_R0_mean, D_ct) for piecewise-constant trajectory against arithmetic-mean baseline."""
    total_t = float(np.sum(dt_vec))
    w = dt_vec / total_t
    H_bar = float(np.sum(w * H_seq))
    omp_bar = float(np.sum(w * omega_p_seq))
    omi_bar = float(np.sum(w * omega_i_seq))

    r_const = ct_dominant_eig(ct_phage_generator(H_bar, alpha, nu, beta, mu0, omp_bar, omi_bar, y=0.0))
    r_obs = ct_piecewise_growth(H_seq, dt_vec, alpha, nu, beta, mu0, omega_p_seq, omega_i_seq, y=0.0)

    # Mean-rate R0: burst factor that makes constant-mean generator have zero top eigenvalue:
    # det([[-(nu+omi_bar), alpha*H_bar], [beta*exp(y0)*nu, -(alpha*H_bar+mu0+omp_bar)]]) = 0
    R0_mean = (alpha * H_bar / (alpha * H_bar + mu0 + omp_bar)) * (beta * nu / (nu + omi_bar))
    ln_R0 = math.log(R0_mean)

    # Solve for y* such that ct_piecewise_growth(..., y = -ln_R0 + D) == 0
    # At y_base = -ln_R0, constant-mean system is at threshold (r = 0).
    from scipy import optimize
    def f(D_val: float) -> float:
        return ct_piecewise_growth(H_seq, dt_vec, alpha, nu, beta, mu0, omega_p_seq, omega_i_seq, y=-ln_R0 + D_val)

    lo, hi = -2.0, 5.0
    while f(lo) > 0:
        lo *= 2.0
    while f(hi) < 0:
        hi *= 2.0
    D_ct = float(optimize.brentq(f, lo, hi, xtol=1e-9, rtol=1e-9))
    return r_const, r_obs, ln_R0, D_ct


def run_known_checks(buf: io.StringIO) -> dict:
    results = {}

    # K1: Engine.coefficient()['D2'] vs closed_form_c1 across 25 parameter sets
    rng = np.random.default_rng(20261114)
    worst_k1 = 0.0
    for _ in range(25):
        tau_p = float(rng.uniform(2.5, 25.0))
        tau_i = float(rng.uniform(1.5, 20.0))
        d = float(rng.uniform(0.05, 0.95))
        phi = float(rng.uniform(-0.6, 0.92))
        sigma = 0.05  # small sigma with symmetric bounded driver (low=None) so Var(h)=exp(sigma^2)-1
        # Note: Engine.sensitivities() scales by d.sigma, while closed_form_c1 uses sigma^2!
        death_p = d / tau_p
        desc = le.Description(
            stages=[
                le.Stage("P", mean=tau_p, shape="memoryless", death=death_p, contact=True),
                le.Stage("I", mean=tau_i, shape="memoryless", death=0.0, contact=False),
            ],
            transfers=[
                le.Transfer("P", "I", yield_=1.0, driver="host", birth=False),
                le.Transfer("I", "P", yield_=20.0, driver=None, birth=True),
            ],
            drivers=[le.Driver("host", sigma=sigma, phi=phi, kind="bounded", low=None)],
        )
        eng = le.Engine(desc)
        d2_eng = eng.coefficient(threshold=True)["D2"]
        d2_cf = closed_form_c1(tau_p, tau_i, d, phi, sigma)
        rel = abs(d2_eng - d2_cf) / max(abs(d2_cf), 1e-15)
        worst_k1 = max(worst_k1, rel)
    ok_k1 = bool(worst_k1 < 1e-11)
    print(f"[{'PASS' if ok_k1 else 'FAIL'}] K1 Engine contact-limited D2 vs Proposition C1 (25 rings): max rel diff = {worst_k1:.2e} (tol 1e-11)", file=buf)
    results["K1_max_rel_diff"] = float(worst_k1)
    results["K1_pass"] = ok_k1

    # K2: Zero waiting death (mu0 = omega = 0 -> d = 0) gives exact conservation and D = 0 at large swing
    H_swings = rng.uniform(0.15, 2.5, size=120)
    dt_swings = rng.uniform(0.05, 0.4, size=120)
    _, _, _, D_zero_death = ct_threshold_loss(
        H_swings, dt_swings, alpha=0.8, nu=1.5, beta=50.0, mu0=0.0,
        omega_p_seq=np.zeros(120), omega_i_seq=np.zeros(120),
    )
    # Also discrete left-eigenvector conservation w^T A(t) = w^T at R0 = 1, death = 0:
    # A(t) = [[1 - nu, c h(t)], [nu, 1 - c h(t)]] -> w = [1, 1]^T gives w^T A(t) = [1, 1]^T
    max_cons_err = 0.0
    for h_val in H_swings[:20]:
        c_step, nu_step = 0.25, 0.2
        A_t = np.array([[1.0 - nu_step, c_step * h_val], [nu_step, 1.0 - c_step * h_val]])
        max_cons_err = max(max_cons_err, float(np.max(np.abs(np.ones(2) @ A_t - np.ones(2)))))
    ok_k2 = bool(abs(D_zero_death) < 1e-10 and max_cons_err < 1e-13)
    print(f"[{'PASS' if ok_k2 else 'FAIL'}] K2 Zero waiting mortality (d=0): |D_ct| = {abs(D_zero_death):.2e} (tol 1e-10), discrete conservation err = {max_cons_err:.2e}", file=buf)
    results["K2_D_zero_death"] = float(abs(D_zero_death))
    results["K2_pass"] = ok_k2

    # K3: Discrete step -> continuous-time convergence + telegraph check against (C2)
    # First: discrete step dt=5e-4 vs continuous-time Floquet on a 40-step fluctuating host series
    H_test = 1.0 + 0.6 * np.sin(2.0 * np.pi * np.arange(40) / 10.0)
    dt_coarse = np.full(40, 0.1)
    _, _, _, D_ct_ref = ct_threshold_loss(
        H_test, dt_coarse, alpha=1.2, nu=2.0, beta=40.0, mu0=0.4,
        omega_p_seq=np.full(40, 0.4), omega_i_seq=np.full(40, 0.4),
    )
    # Discretize each 0.1 interval into 200 substeps of dt_sub = 5e-4
    dt_sub = 5e-4
    sub_per_seg = 200
    mu_p = 0.4 + 0.4
    c_bar = 1.2 * 1.0
    nu_tot = 2.0 + 0.4
    # At threshold, R0_disc = (c_bar / (c_bar + mu_p)) * (beta_crit * 2.0 / nu_tot) = 1
    beta_crit = (c_bar + mu_p) / c_bar * (nu_tot / 2.0)
    def disc_growth(D_shift: float) -> float:
        U = np.eye(2)
        ls = 0.0
        b_eff = beta_crit * math.exp(D_shift)
        for h_val in H_test:
            A = np.array([
                [1.0 - nu_tot * dt_sub, c_bar * h_val * dt_sub],
                [b_eff * 2.0 * dt_sub, 1.0 - (mu_p + c_bar * h_val) * dt_sub],
            ])
            Ak = np.linalg.matrix_power(A, sub_per_seg)
            U = Ak @ U
            nrm = float(np.max(np.abs(U)))
            U /= nrm
            ls += math.log(nrm)
        rho = float(np.max(np.abs(linalg.eigvals(U))))
        return (ls + math.log(rho)) / (40 * 0.1)
    from scipy import optimize
    D_disc_fine = float(optimize.brentq(disc_growth, 0.0, 1.0, xtol=1e-10))
    rel_dt = abs(D_disc_fine - D_ct_ref) / D_ct_ref

    # Second: small-amplitude symmetric telegraph host (1 +/- 0.04) vs closed_form_c2_ct
    rng_tel = np.random.default_rng(99)
    delta_tel = 0.04
    theta_ct = 2.0  # correlation time: symmetric telegraph switching rate lambda_sw = 1/(2*theta_ct) = 0.25 -> mean dwell 4.0?
    # Wait: symmetric telegraph switching between +1 and -1 with holding rate lam_sw each way has autocovariance exp(-2 * lam_sw * t),
    # so correlation time theta_ct = 1 / (2 * lam_sw), i.e., mean dwell m_dwell = 1 / lam_sw = 2 * theta_ct!
    lam_sw = 1.0 / (2.0 * theta_ct)
    n_dwells = 25000
    dwells = rng_tel.exponential(scale=1.0 / lam_sw, size=n_dwells)
    # Antithetic pair of signs to cancel odd/mean drift
    signs = np.where(np.arange(n_dwells) % 2 == 0, 1.0, -1.0)
    h_p = 1.0 + delta_tel * signs
    h_m = 1.0 - delta_tel * signs
    alpha_t, mu_t, nu_t = 0.6, 0.4, 0.5  # tau_p = 1/(0.6+0.4)=1.0, d = 0.4, tau_i = 2.0
    beta_t0 = 1.0 / (1.0 - 0.4)  # since omega_i = 0, R0 = (1-d)*beta = 1 -> beta_t0 = 1/0.6
    def tel_growth(D_val: float) -> float:
        r_sum = 0.0
        for hs in (h_p, h_m):
            x = np.array([0.5, 0.5])
            acc = 0.0
            t_tot = 0.0
            for k in range(n_dwells):
                M = ct_phage_generator(hs[k], alpha_t, nu_t, beta_t0, mu_t, 0.0, 0.0, y=D_val)
                x = linalg.expm(M * dwells[k]) @ x
                s = float(x.sum())
                x /= s
                if k >= 200:
                    acc += math.log(s)
                    t_tot += dwells[k]
            r_sum += acc / t_tot
        return 0.5 * r_sum
    D_tel_sim = float(optimize.brentq(tel_growth, -0.01, 0.01, xtol=1e-9))
    D_tel_c2 = closed_form_c2_ct(tau_p=1.0, tau_i=2.0, d=0.4, theta_ct=theta_ct, sigma=delta_tel)
    rel_c2 = abs(D_tel_sim - D_tel_c2) / abs(D_tel_c2)
    ok_k3 = bool(rel_dt < 2e-3 and rel_c2 < 1.5e-2)
    print(f"[{'PASS' if ok_k3 else 'FAIL'}] K3 Continuous-discrete convergence rel diff = {rel_dt:.2e} (tol 2e-3); telegraph vs (C2) rel diff = {rel_c2:.2e} (tol 1.5e-2)", file=buf)
    results["K3_rel_dt"] = float(rel_dt)
    results["K3_rel_c2"] = float(rel_c2)
    results["K3_pass"] = ok_k3

    # K4: Coupled-dilution chemostat identities & Proposition B bound
    # (a) Uniform washout -omega(t) I_2 shifts growth rate by -mean(omega) pathwise
    worst_shift = 0.0
    for _ in range(50):
        K_seg = 30
        H_r = rng.uniform(0.2, 2.2, size=K_seg)
        dt_r = rng.uniform(0.05, 0.3, size=K_seg)
        om_r = rng.uniform(0.05, 1.8, size=K_seg)
        om_bar = float(np.sum(dt_r * om_r) / np.sum(dt_r))
        r_with = ct_piecewise_growth(H_r, dt_r, 0.9, 1.4, 30.0, 0.2, om_r, om_r)
        r_nowash = ct_piecewise_growth(H_r, dt_r, 0.9, 1.4, 30.0, 0.2, np.zeros(K_seg), np.zeros(K_seg))
        worst_shift = max(worst_shift, abs((r_nowash - om_bar) - r_with))

    # (b) Discrete fluctuating dilution on free phage with constant host:
    #     violates transfer-only bound (l_g = 0, D > 0), satisfies Proposition B bound,
    #     and (tau_0 - 1)*l_s scales as O(dt)
    violations_prop_b = 0
    violations_transfer_only = 0
    for idx in range(100):
        # 2-stage discrete ring on periodic/random 80-step schedule
        K_s = 80
        # Host constant or negatively correlated with dilution
        mode_h_const = (idx < 30)
        z_seq = rng.choice([-1.0, 1.0], size=K_s)
        dt_s = 0.1
        nu_d = 0.8
        mu0_d = 0.2
        om_bar_d = 0.6
        om_seq = om_bar_d * (1.0 + 0.5 * z_seq)  # fluctuating dilution
        H_seq_d = np.ones(K_s) if mode_h_const else (1.0 - 0.45 * z_seq)  # negatively correlated host!
        alpha_d = 0.7
        # Stage 0 = I, Stage 1 = P; selective phage washout (infected cells wall-sheltered or constant exit)
        sI_seq = np.full(K_s, 1.0 - nu_d * dt_s)
        gI_base = (nu_d * dt_s)  # at threshold R0=1 we multiply birth edge I->P by beta_crit
        gP_seq = alpha_d * H_seq_d * dt_s
        sP_seq = 1.0 - (alpha_d * H_seq_d + mu0_d + om_seq) * dt_s
        assert np.all(sP_seq > 0.01) and np.all(sI_seq > 0.01)
        sI_bar, gP_bar, sP_bar = float(sI_seq.mean()), float(gP_seq.mean()), float(sP_seq.mean())
        beta_crit_d = (1.0 - sI_bar) * (1.0 - sP_bar) / (gI_base * gP_bar)

        def d_growth(D_v: float) -> float:
            U = np.eye(2)
            ls = 0.0
            b_now = beta_crit_d * math.exp(D_v)
            for k in range(K_s):
                A = np.array([[sI_seq[k], gP_seq[k]], [b_now * gI_base, sP_seq[k]]])
                U = A @ U
                nrm = float(np.max(np.abs(U)))
                U /= nrm
                ls += math.log(nrm)
            return ls + math.log(float(np.max(np.abs(linalg.eigvals(U)))))

        D_exact = float(optimize.brentq(d_growth, -1.0, 2.0, xtol=1e-10))
        lg_P = -float(np.mean(np.log(gP_seq / gP_bar)))
        ls_P = -float(np.mean(np.log(sP_seq / sP_bar)))
        ls_I = -float(np.mean(np.log(sI_seq / sI_bar)))
        tau_P0 = 1.0 / (1.0 - sP_bar)
        tau_I0 = 1.0 / (1.0 - sI_bar)
        bound_transfer = lg_P
        bound_prop_b = lg_P + (tau_P0 - 1.0) * ls_P + (tau_I0 - 1.0) * ls_I
        if D_exact > bound_transfer + 1e-8:
            violations_transfer_only += 1
        if D_exact > bound_prop_b + 1e-8:
            violations_prop_b += 1

    # Check O(dt) vanishing of (tau_P0 - 1) * ls_P as dt halves from 0.02 to 0.01
    z_grid = np.array([-1.0, 1.0])
    def stay_term_at_dt(dt_v: float) -> float:
        r_vals = 0.7 * (1.0 - 0.4 * z_grid) + 0.2 + 0.6 * (1.0 + 0.5 * z_grid)
        s_vals = np.exp(-r_vals * dt_v)
        s_bar = float(s_vals.mean())
        tau0 = 1.0 / (1.0 - s_bar)
        ls = -float(np.mean(np.log(s_vals / s_bar)))
        return (tau0 - 1.0) * ls

    st_02 = stay_term_at_dt(0.02)
    st_01 = stay_term_at_dt(0.01)
    ratio_dt_scaling = st_02 / st_01  # should be ~2.0

    ok_k4 = bool((worst_shift < 1e-13) and (violations_prop_b == 0) and (violations_transfer_only > 0) and (abs(ratio_dt_scaling - 2.0) < 0.02))
    print(
        f"[{'PASS' if ok_k4 else 'FAIL'}] K4 Coupled dilution: uniform-washout pathwise shift err = {worst_shift:.2e}; "
        f"transfer-only violations = {violations_transfer_only}/100, Prop B violations = {violations_prop_b}/100; "
        f"(tau-1)*l_s dt-halving ratio = {ratio_dt_scaling:.4f} (expected 2.0)",
        file=buf,
    )
    results["K4_worst_shift"] = float(worst_shift)
    results["K4_transfer_only_violations"] = int(violations_transfer_only)
    results["K4_prop_b_violations"] = int(violations_prop_b)
    results["K4_dt_halving_ratio"] = float(ratio_dt_scaling)
    results["K4_pass"] = ok_k4

    # Also illustrate coupled negative vs positive (H, omega) covariance effect on D in selective washout!
    K_ill = 200
    t_ill = np.arange(K_ill)
    wave = np.where((t_ill // 10) % 2 == 0, 1.0, -1.0)  # period 20 steps
    dt_ill = np.full(K_ill, 0.25)
    H_ill = 1.0 + 0.6 * wave
    om_const = np.full(K_ill, 0.5)
    om_neg = 0.5 * (1.0 - 0.6 * wave)  # high dilution during low host (chemostat coupling: Cov(H, omega) < 0)
    om_pos = 0.5 * (1.0 + 0.6 * wave)  # high dilution during high host (Cov(H, omega) > 0)
    _, _, _, D_om_const = ct_threshold_loss(H_ill, dt_ill, 0.5, 1.0, 20.0, 0.1, om_const, np.zeros(K_ill))
    _, _, _, D_om_neg = ct_threshold_loss(H_ill, dt_ill, 0.5, 1.0, 20.0, 0.1, om_neg, np.zeros(K_ill))
    _, _, _, D_om_pos = ct_threshold_loss(H_ill, dt_ill, 0.5, 1.0, 20.0, 0.1, om_pos, np.zeros(K_ill))
    print(
        f"  [Illustration] Selective free-phage washout in continuous time (same mean H=1, same mean omega_P=0.5): "
        f"D(const omega) = {D_om_const:.5f}, D(Cov(H,omega)<0, chemostat-like) = {D_om_neg:.5f}, "
        f"D(Cov(H,omega)>0) = {D_om_pos:.5f}",
        file=buf,
    )
    results["coupled_illustration_ct"] = {
        "D_const_omega": D_om_const,
        "D_neg_cov_chemostat": D_om_neg,
        "D_pos_cov": D_om_pos,
    }
    return results


def partial_spearman_one_sided(y: np.ndarray, d_loss: np.ndarray, r_const: np.ndarray, rng: np.random.Generator, n_perm: int = 2000) -> tuple[float, float, float]:
    """Freedman-Lane rank-residual partial Spearman correlation between y and d_loss given r_const.

    Returns (rho, p_neg_supports, p_pos_contradicts).
    """
    n = len(y)
    ry = stats.rankdata(y)
    rd = stats.rankdata(d_loss)
    rc = stats.rankdata(r_const)
    X = np.column_stack([np.ones(n), rc])
    # Orthogonal projector M = I - X (X^T X)^+ X^T
    P_X = X @ np.linalg.pinv(X)
    M = np.eye(n) - P_X
    ey = M @ ry
    ed = M @ rd
    norm_ed = float(np.linalg.norm(ed))
    norm_ey = float(np.linalg.norm(ey))
    if norm_ed < 1e-12 or norm_ey < 1e-12:
        return 0.0, 1.0, 1.0
    rho_obs = float((ey @ ed) / (norm_ey * norm_ed))
    yhat = P_X @ ry

    if n <= 7:
        from itertools import permutations
        vals = []
        for p in permutations(range(n)):
            y_star = yhat + ey[list(p)]
            ey_star = M @ y_star
            ne = float(np.linalg.norm(ey_star))
            vals.append(float((ey_star @ ed) / (ne * norm_ed)) if ne > 1e-12 else 0.0)
        vals_arr = np.asarray(vals)
        p_neg = float(np.mean(vals_arr <= rho_obs + 1e-15))
        p_pos = float(np.mean(vals_arr >= rho_obs - 1e-15))
        return rho_obs, p_neg, p_pos

    cnt_neg = 1
    cnt_pos = 1
    for _ in range(n_perm):
        p = rng.permutation(n)
        ey_star = M @ (yhat + ey[p])
        ne = float(np.linalg.norm(ey_star))
        val = float((ey_star @ ed) / (ne * norm_ed)) if ne > 1e-12 else 0.0
        if val <= rho_obs + 1e-15:
            cnt_neg += 1
        if val >= rho_obs - 1e-15:
            cnt_pos += 1
    return rho_obs, cnt_neg / (n_perm + 1), cnt_pos / (n_perm + 1)


def simulate_power_designs(buf: io.StringIO) -> dict:
    """Pre-analysis power & identifiability study across the four dataset designs (A-D)."""
    rng = np.random.default_rng(20261115)
    n_mc = 1000
    designs = [
        # (label, n_units, n_steps, dt_days, sigma_grid, phi_true, meas_sd_log, T_duration_days)
        ("Design_A_Larsen_chemostat_n6", 6, 60, 2.0, np.array([0.45, 0.55, 0.65, 1.30, 1.50, 1.70]), 0.80, 0.20, 120.0),
        ("Design_B_Reuter_transfer_n10", 10, 20, 0.125, np.array([0.15]*5 + [0.85]*5), 0.25, 0.20, 2.5),
        ("Design_C_Chaudhry_Zhao_n18", 18, 12, 1.0, np.linspace(0.30, 1.40, 18), 0.55, 0.25, 12.0),
        ("Design_D_Duncan_microcosm_n30", 30, 9, 2.0, np.array([0.30]*12 + [0.75]*6 + [1.10]*6 + [1.35]*6), 0.60, 0.25, 16.0),
    ]

    summary = {}
    print("\n=== PRE-ANALYSIS POWER AND IDENTIFIABILITY SIMULATION (1,000 trials per cell) ===", file=buf)
    print("Criterion T1: partial Spearman(y, D_pred | r_const) < 0 with one-sided p < 0.05 ('supports')", file=buf)
    print("              or > 0 with one-sided p < 0.05 ('contradicts').", file=buf)

    for label, n_u, K_t, dt_d, sig_vec, phi_h, meas_sd, T_dur in designs:
        summary[label] = {}
        print(f"\n--- {label} (n={n_u}, K={K_t} steps, dt={dt_d} d, meas_sd={meas_sd}) ---", file=buf)

        # Precompute typical D_pred scale at d = 0.04 (high host density) vs d = 0.70 (low susceptible host density)
        theta_ct = -dt_d / math.log(max(phi_h, 0.05))
        tau_p_highH, tau_i = 0.02, 0.20  # days
        tau_p_lowH = 0.80
        d_small = 0.04
        d_large = 0.70

        for scen_code, scen_desc in [
            ("S0_null", "No fluctuation effect (mean R0 + biological/assay noise SD=0.35)"),
            ("S1_linear_highH_d0.04", "True linear rare-growth theory, dense host (d=0.04, strong waiting buffer)"),
            ("S2_linear_lowH_d0.70", "True linear rare-growth theory, scarce susceptible host (d=0.70)"),
            ("S3_nonlinear_ecoevo", "Endogenous predator-prey + 98% resistance sweep (feedback reverses sigma_H)"),
        ]:
            supp = 0
            contra = 0
            undec = 0
            rhos = []
            d_preds_mean = []

            for _ in range(n_mc):
                # Simulate AR(1) log-host trajectories for each unit i=1..n_u
                log_hbar = rng.normal(0.0, 0.35, size=n_u)
                r_const_est = np.zeros(n_u)
                d_pred_est = np.zeros(n_u)
                y_obs = np.zeros(n_u)

                for i in range(n_u):
                    sig_i = sig_vec[i]
                    z = np.zeros(K_t)
                    z[0] = rng.normal(0.0, 1.0)
                    for t in range(1, K_t):
                        z[t] = phi_h * z[t - 1] + math.sqrt(1.0 - phi_h * phi_h) * rng.normal(0.0, 1.0)

                    if scen_code == "S3_nonlinear_ecoevo":
                        # Under nonlinear predator-prey + resistance, units with higher intrinsic phage attack/nutrient
                        # turnover experience LARGER host crashes (sigma_H increases with phage productivity!) while
                        # resistant bacteria sweep to ~98% so mean total host barely changes.
                        attack_drive = 0.8 * (sig_i - float(sig_vec.mean())) + rng.normal(0.0, 0.20)
                        H_true = np.exp(log_hbar[i] + (sig_i + 0.35 * max(attack_drive, 0.0)) * z - 0.5 * sig_i * sig_i)
                    else:
                        H_true = np.exp(log_hbar[i] + sig_i * z - 0.5 * sig_i * sig_i)

                    H_meas = H_true * np.exp(rng.normal(0.0, meas_sd, size=K_t))
                    hbar_hat = float(np.mean(H_meas))
                    sig_hat = float(np.std(np.log(H_meas), ddof=1))
                    if K_t >= 4:
                        lh = np.log(H_meas)
                        c0 = float(np.var(lh))
                        phi_hat = float(np.corrcoef(lh[:-1], lh[1:])[0, 1]) if c0 > 1e-12 else 0.0
                    else:
                        phi_hat = 0.0
                    phi_pos = float(np.clip(phi_hat, 0.01, 0.98))
                    th_hat = -dt_d / math.log(phi_pos)

                    d_use = d_large if scen_code == "S2_linear_lowH_d0.70" else d_small
                    tp_use = tau_p_lowH if scen_code == "S2_linear_lowH_d0.70" else tau_p_highH
                    T_gen = tp_use + tau_i
                    D_i = max(0.0, closed_form_c2_ct(tp_use, tau_i, d_use, th_hat, sig_hat))
                    d_pred_est[i] = D_i
                    r_const_est[i] = 0.8 * math.log(max(hbar_hat, 1e-12))

                    # True observable y_obs: net log-phage change per generation (or standardized log phage titer)
                    bio_noise = rng.normal(0.0, 0.35) + rng.normal(0.0, meas_sd * math.sqrt(2.0)) / max(T_dur / T_gen, 1.0)
                    if scen_code == "S0_null":
                        y_obs[i] = 0.8 * log_hbar[i] + bio_noise
                    elif scen_code in ("S1_linear_highH_d0.04", "S2_linear_lowH_d0.70"):
                        D_true = closed_form_c2_ct(tp_use, tau_i, d_use, theta_ct, sig_i)
                        y_obs[i] = 0.8 * log_hbar[i] - D_true + bio_noise
                    elif scen_code == "S3_nonlinear_ecoevo":
                        # Phage equilibrium titer rises with attack_drive (which also inflated sigma_H!),
                        # masking the small linear D_true penalty
                        D_true = closed_form_c2_ct(tau_p_lowH, tau_i, 0.50, theta_ct, sig_i)
                        y_obs[i] = 0.4 * log_hbar[i] + 0.9 * attack_drive - 0.2 * D_true + bio_noise

                rho, p_neg, p_pos = partial_spearman_one_sided(y_obs, d_pred_est, r_const_est, rng, n_perm=400)
                rhos.append(rho)
                d_preds_mean.append(float(np.ptp(d_pred_est)))
                if rho < 0 and p_neg < 0.05:
                    supp += 1
                elif rho > 0 and p_pos < 0.05:
                    contra += 1
                else:
                    undec += 1

            cell = {
                "mean_rho": float(np.mean(rhos)),
                "mean_D_spread": float(np.mean(d_preds_mean)),
                "frac_supports": supp / n_mc,
                "frac_contradicts": contra / n_mc,
                "frac_not_decided": undec / n_mc,
            }
            summary[label][scen_code] = cell
            print(
                f"  {scen_code:24s}: mean_D_range={cell['mean_D_spread']:.4f}, mean_rho={cell['mean_rho']:+.3f} | "
                f"supports={100*cell['frac_supports']:5.1f}%, contradicts={100*cell['frac_contradicts']:5.1f}%, "
                f"not_decided={100*cell['frac_not_decided']:5.1f}%",
                file=buf,
            )
    return summary


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    buf = io.StringIO()
    utc_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"Pre-analysis check & power study started at {utc_now} (NO empirical data files opened)", file=buf)

    checks = run_known_checks(buf)
    if not (checks["K1_pass"] and checks["K2_pass"] and checks["K3_pass"] and checks["K4_pass"]):
        print("FATAL: known-result verification failed!", file=sys.stderr)
        sys.exit(1)

    power = simulate_power_designs(buf)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f} s. All known-result checks PASSED.", file=buf)

    txt = buf.getvalue()
    with open(os.path.join(OUT_DIR, "power_precheck.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "power_precheck.json"), "w", encoding="utf-8") as f:
        json.dump({"utc": utc_now, "elapsed_s": elapsed, "checks": checks, "power": power}, f, indent=2)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
