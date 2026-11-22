#!/usr/bin/env python3
"""Two-stage phage-bacteria lifecycle operators, second-order formulas, bounds, and tests.

Implements the exact mathematical mapping registered in `protocol.md` (§2, §4–§6),
plus calls to `studies/engine-R16/lifecycle_engine.py`.
Run under `python3 -I`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import os
import sys
import numpy as np
from scipy import linalg, optimize, stats

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
PROB_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE_DIR = os.path.join(PROB_ROOT, "studies", "engine-R16")
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

import lifecycle_engine as le  # noqa: E402


@dataclass(frozen=True)
class PhageParams:
    name: str
    latent_d: float       # L_lat in days (mean stay in I without washout)
    burst: float          # beta (progeny per lysed cell)
    adsorption_ml_d: float  # alpha in mL cell^-1 d^-1
    decay_d: float        # mu_0 in d^-1 (intrinsic free-phage decay)
    substages_I: int = 1  # Erlang shape m_I for latent period
    provenance: str = ""


SYSTEM_PARAMS: dict[str, PhageParams] = {
    "D1_larsen2019": PhageParams(
        name="Synechococcus WH7803 - S-RIM8",
        latent_d=9.0 / 24.0,
        burst=50.0,
        adsorption_ml_d=6.06e-9 * 1440.0,
        decay_d=0.03,
        substages_I=1,
        provenance="[literature] Wilson et al. 1996 / Mann 2003 (S-PM2 on WH7803); Mruwat et al. 2022",
    ),
    "D2_reuter2026": PhageParams(
        name="E. coli C - PhiX174",
        latent_d=17.0 / 1440.0,
        burst=200.0,
        adsorption_ml_d=4.0e-9 * 1440.0,
        decay_d=1.7e-2 * 1440.0,
        substages_I=1,
        provenance="[paper] Reuter et al. 2026 Virus Evol. Table 1 (ancestor WT)",
    ),
    "D3_chaudhry2018": PhageParams(
        name="E. coli B - lambdaVIR",
        latent_d=25.0 / 1440.0,
        burst=55.0,
        adsorption_ml_d=2.0e-7 * 24.0,
        decay_d=0.10,
        substages_I=1,
        provenance="[paper] Chaudhry et al. 2018 PLOS Biol. Fig 1/S1/S4; decay [illustrative]",
    ),
    "D4_duncan2017": PhageParams(
        name="P. fluorescens SBW25 - phi2",
        latent_d=35.0 / 1440.0,
        burst=90.0,
        adsorption_ml_d=3.0e-9 * 1440.0,
        decay_d=0.15,
        substages_I=1,
        provenance="[literature/illustrative] Zhang et al. 2017 (VSW-3 burst 90) + Pseudomonas lytic phage range",
    ),
    "D5_zhao2016": PhageParams(
        name="P. fluorescens SBW25 - phi2 (OD)",
        latent_d=35.0 / 1440.0,
        burst=90.0,
        adsorption_ml_d=3.0e-9 * 1440.0,
        decay_d=0.15,
        substages_I=1,
        provenance="[literature/illustrative] Zhang et al. 2017 + OD600 conversion 5e8 CFU/mL/OD",
    ),
}


def build_ct_generator(
    H: float,
    H_bar: float,
    omega_P: float,
    omega_I: float,
    omega_P_bar: float,
    params: PhageParams,
    mode: str = "contact",
    y_tilt: float = 0.0,
    sat_c50: float | None = None,
    rate_norm: float = 1.0,
) -> np.ndarray:
    """Build (m_I + 1) x (m_I + 1) continuous-time generator for [I_1, ..., I_{m_I}, P]^T."""
    m = int(params.substages_I)
    nu = 1.0 / params.latent_d
    alpha = params.adsorption_ml_d
    mu0 = params.decay_d
    M = np.zeros((m + 1, m + 1), dtype=float)

    h_rel = H / max(H_bar, 1e-30)
    if sat_c50 is not None:
        eff_mult = ((1.0 + sat_c50) * h_rel / (sat_c50 + h_rel)) / rate_norm
    else:
        eff_mult = h_rel / rate_norm

    ads_rate = alpha * H_bar * eff_mult

    # Infected sub-stages 0 .. m-1
    for j in range(m):
        M[j, j] = -(m * nu + omega_I)
        if j + 1 < m:
            M[j + 1, j] = m * nu
    # Lysis from last sub-stage m-1 into P (index m)
    M[m, m - 1] = math.exp(y_tilt) * params.burst * m * nu

    # Free phage stage P (index m) -> I_1 (index 0)
    M[0, m] = ads_rate
    if mode == "contact":
        M[m, m] = -(ads_rate + mu0 + omega_P)
    elif mode == "waste":
        M[m, m] = -(alpha * H_bar + mu0 + omega_P_bar)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return M


def dominant_real_eig(M: np.ndarray) -> float:
    return float(np.max(linalg.eigvals(M).real))


def floquet_growth_ct(
    H_seq: np.ndarray,
    dt_seq: np.ndarray,
    omega_P_seq: np.ndarray,
    omega_I_seq: np.ndarray,
    params: PhageParams,
    mode: str = "contact",
    y_tilt: float = 0.0,
    sat_c50: float | None = None,
    mean_centered_sat: bool = False,
) -> float:
    """Exact Floquet top growth exponent (per day) for piecewise-constant trajectory."""
    T_tot = float(np.sum(dt_seq))
    w = dt_seq / T_tot
    H_bar = float(np.sum(w * H_seq))
    omp_bar = float(np.sum(w * omega_P_seq))

    rate_norm = 1.0
    if sat_c50 is not None and mean_centered_sat:
        h_rel = H_seq / max(H_bar, 1e-30)
        f_vals = (1.0 + sat_c50) * h_rel / (sat_c50 + h_rel)
        rate_norm = float(np.sum(w * f_vals))

    dim = int(params.substages_I) + 1
    eye_d = np.eye(dim, dtype=float)
    U = np.eye(dim, dtype=float)
    log_scale = 0.0
    for k in range(len(H_seq)):
        dtk = float(dt_seq[k])
        Mk = build_ct_generator(
            H=float(H_seq[k]),
            H_bar=H_bar,
            omega_P=float(omega_P_seq[k]),
            omega_I=float(omega_I_seq[k]),
            omega_P_bar=omp_bar,
            params=params,
            mode=mode,
            y_tilt=y_tilt,
            sat_c50=sat_c50,
            rate_norm=rate_norm,
        )
        sk = dominant_real_eig(Mk)
        Ek = linalg.expm((Mk - sk * eye_d) * dtk)
        log_scale += sk * dtk
        U = Ek @ U
        nrm = float(np.max(np.abs(U)))
        if nrm > 0.0:
            U /= nrm
            log_scale += math.log(nrm)
    rho = float(np.max(np.abs(linalg.eigvals(U))))
    return (log_scale + math.log(max(rho, 1e-300))) / T_tot


def threshold_quantities_ct(
    H_seq: np.ndarray,
    dt_seq: np.ndarray,
    omega_P_seq: np.ndarray,
    omega_I_seq: np.ndarray,
    params: PhageParams,
    mode: str = "contact",
    sat_c50: float | None = None,
    mean_centered_sat: bool = False,
) -> dict:
    """Compute r_const, r_obs, Delta_r, R0, d_threshold, tau_P, tau_I, and exact D_sim."""
    T_tot = float(np.sum(dt_seq))
    w = dt_seq / T_tot
    H_bar = float(np.sum(w * H_seq))
    omp_bar = float(np.sum(w * omega_P_seq))
    omi_bar = float(np.sum(w * omega_I_seq))

    m = int(params.substages_I)
    nu = 1.0 / params.latent_d
    alpha = params.adsorption_ml_d
    mu0 = params.decay_d

    # Constant arithmetic-mean baseline
    M_const = build_ct_generator(
        H=H_bar, H_bar=H_bar, omega_P=omp_bar, omega_I=omi_bar, omega_P_bar=omp_bar,
        params=params, mode=mode, y_tilt=0.0, sat_c50=None, rate_norm=1.0,
    )
    r_const = dominant_real_eig(M_const)
    r_obs = floquet_growth_ct(
        H_seq, dt_seq, omega_P_seq, omega_I_seq, params, mode=mode, y_tilt=0.0,
        sat_c50=sat_c50, mean_centered_sat=mean_centered_sat,
    )

    d_thresh = (mu0 + omp_bar) / (alpha * H_bar + mu0 + omp_bar)
    S_I = ((m * nu) / (m * nu + omi_bar)) ** m
    R0_val = (1.0 - d_thresh) * params.burst * S_I
    ln_R0 = math.log(max(R0_val, 1e-300))

    tau_P = 1.0 / (alpha * H_bar + mu0 + omp_bar)
    tau_I = m / (m * nu + omi_bar)
    T_gen0 = tau_P + tau_I

    def root_fn(D_val: float) -> float:
        return floquet_growth_ct(
            H_seq, dt_seq, omega_P_seq, omega_I_seq, params,
            mode=mode, y_tilt=-ln_R0 + D_val,
            sat_c50=sat_c50, mean_centered_sat=mean_centered_sat,
        )

    lo, hi = -2.0, 4.0
    flo = root_fn(lo)
    while flo > 0.0 and lo > -40.0:
        lo *= 2.0
        flo = root_fn(lo)
    fhi = root_fn(hi)
    while fhi < 0.0 and hi < 80.0:
        hi *= 2.0
        fhi = root_fn(hi)
    D_sim = float(optimize.brentq(root_fn, lo, hi, xtol=1e-9, rtol=1e-9))

    return {
        "H_bar": H_bar,
        "omega_P_bar": omp_bar,
        "omega_I_bar": omi_bar,
        "r_const_d": r_const,
        "r_obs_d": r_obs,
        "Delta_r_fluct_d": r_const - r_obs,
        "R0_mean": R0_val,
        "ln_R0_mean": ln_R0,
        "d_threshold": d_thresh,
        "tau_P_d": tau_P,
        "tau_I_d": tau_I,
        "T_gen0_d": T_gen0,
        "D_sim": D_sim,
    }


def second_order_formulas(
    H_seq: np.ndarray,
    dt_seq: np.ndarray,
    tau_P_d: float,
    tau_I_d: float,
    d_thresh: float,
    substages_I: int = 1,
) -> dict:
    """Estimate host log-SD, CV, lag-1 autocorrelation, and compute 2nd-order Gamma_c and Gamma_waste."""
    lh = np.log(H_seq)
    K = len(H_seq)
    w = dt_seq / float(np.sum(dt_seq))
    H_bar = float(np.sum(w * H_seq))
    sigma_lnH = float(np.std(lh, ddof=1)) if K >= 2 else 0.0
    cv_H = float(np.std(H_seq / H_bar, ddof=1)) if K >= 2 else 0.0

    if K >= 3 and np.std(lh[:-1]) > 1e-12 and np.std(lh[1:]) > 1e-12:
        phi_lag1 = float(np.corrcoef(lh[:-1], lh[1:])[0, 1])
    else:
        phi_lag1 = 0.0

    dt_med = float(np.median(dt_seq))
    T0 = tau_P_d + tau_I_d
    m = int(substages_I)

    # Continuous-time formula (C2) requires positive exponential memory 0 < phi < 1
    if 0.0 < phi_lag1 < 0.9999:
        theta_ct = -dt_med / math.log(phi_lag1)
        qP = 1.0 / (1.0 + tau_P_d / theta_ct)
        qI = (1.0 + tau_I_d / (m * theta_ct)) ** (-m)
        R = qP * qI
        gamma_c_sigma = sigma_lnH ** 2 * (d_thresh ** 2 * theta_ct / T0 - d_thresh * (R - (1.0 - d_thresh) * qP) / (1.0 - R))
        gamma_c_cv = cv_H ** 2 * (d_thresh ** 2 * theta_ct / T0 - d_thresh * (R - (1.0 - d_thresh) * qP) / (1.0 - R))
        gamma_waste = sigma_lnH ** 2 * (theta_ct / T0 - R / (1.0 - R))
        formula_used = "CT_C2"
    else:
        # Non-positive or degenerate lag-1 autocorrelation: fast-memory CT limit (theta_ct -> 0: R, qP -> 0 => 0 in CT,
        # so evaluate discrete step-level C1 on fine step dt_eff = min(dt_med, tau_P_d/2) with phi mapped or fast limit)
        theta_ct = 0.0
        # Also record discrete (C1) on a valid step dt_disc = min(dt_med, 0.4 * min(tau_P_d, tau_I_d))
        dt_disc = min(dt_med, 0.4 * min(tau_P_d, tau_I_d / m))
        tp_steps = tau_P_d / dt_disc
        ti_steps = tau_I_d / dt_disc
        T_steps = tp_steps + ti_steps
        # If phi_lag1 <= 0 at dt_med, evaluate C1 at phi = 0 (white noise on sub-step) and also report discrete C1 at phi_lag1 if dt_med == dt_disc
        phi_use = float(np.clip(phi_lag1, -0.95, 0.0)) if abs(dt_disc - dt_med) < 1e-9 else 0.0
        th_d = phi_use / (1.0 - phi_use)
        if abs(phi_use) < 1e-12:
            gamma_c_sigma = sigma_lnH ** 2 * (d_thresh ** 2) / (2.0 * T_steps)
            gamma_c_cv = cv_H ** 2 * (d_thresh ** 2) / (2.0 * T_steps)
            gamma_waste = sigma_lnH ** 2 / (2.0 * T_steps)
        else:
            qP_d = 1.0 / (1.0 + tp_steps / th_d)
            qI_d = (1.0 + ti_steps / (m * th_d)) ** (-m)
            R_d = qP_d * qI_d
            gamma_c_sigma = sigma_lnH ** 2 * (d_thresh ** 2 * (1.0 + 2.0 * th_d) / (2.0 * T_steps) - d_thresh * (R_d - (1.0 - d_thresh) * qP_d) / (1.0 - R_d))
            gamma_c_cv = cv_H ** 2 * (d_thresh ** 2 * (1.0 + 2.0 * th_d) / (2.0 * T_steps) - d_thresh * (R_d - (1.0 - d_thresh) * qP_d) / (1.0 - R_d))
            gamma_waste = sigma_lnH ** 2 * ((1.0 + 2.0 * th_d) / (2.0 * T_steps) - R_d / (1.0 - R_d))
        formula_used = "disc_C1_nonpos_phi"

    return {
        "sigma_lnH": sigma_lnH,
        "cv_H": cv_H,
        "phi_lag1": phi_lag1,
        "theta_ct_d": theta_ct,
        "Gamma_c_2nd": gamma_c_sigma,
        "Gamma_c_cv": gamma_c_cv,
        "Gamma_waste_2nd": gamma_waste,
        "formula_used": formula_used,
    }


def proposition_b_discrete_check(
    H_seq: np.ndarray,
    dt_seq: np.ndarray,
    omega_P_seq: np.ndarray,
    omega_I_seq: np.ndarray,
    params: PhageParams,
) -> dict:
    """Verify Proposition B bound (`notes/mechanism-temperature-incubation.md` §2.2) on fine discrete step."""
    nu = 1.0 / params.latent_d
    alpha = params.adsorption_ml_d
    mu0 = params.decay_d

    max_exit_P = float(np.max(alpha * H_seq + mu0 + omega_P_seq))
    max_exit_I = float(np.max(nu + omega_I_seq))
    max_rate = max(max_exit_P, max_exit_I, 1.0)

    # Choose fine sub-step delta_t so all exit probabilities <= 0.15 and each dt_seq[k] is subdivided cleanly
    dt_min = float(np.min(dt_seq))
    n_sub_base = max(1, int(math.ceil(dt_min * max_rate / 0.15)))
    delta_t = dt_min / n_sub_base

    sub_counts = [max(1, int(round(float(dtk) / delta_t))) for dtk in dt_seq]
    total_steps = sum(sub_counts)

    sI_vec = np.repeat(1.0 - (nu + omega_I_seq) * delta_t, sub_counts)
    gI_vec = np.full(total_steps, nu * delta_t)
    gP_vec = np.repeat(alpha * H_seq * delta_t, sub_counts)
    sP_vec = np.repeat(1.0 - (alpha * H_seq + mu0 + omega_P_seq) * delta_t, sub_counts)

    sI_bar = float(np.mean(sI_vec))
    gI_bar = float(np.mean(gI_vec))
    gP_bar = float(np.mean(gP_vec))
    sP_bar = float(np.mean(sP_vec))

    beta_crit = (1.0 - sI_bar) * (1.0 - sP_bar) / (gI_bar * gP_bar)

    def disc_monodromy_logrho(D_val: float) -> float:
        U = np.eye(2, dtype=float)
        ls = 0.0
        b_eff = beta_crit * math.exp(D_val)
        for idx, k_cnt in enumerate(sub_counts):
            sI_k = 1.0 - (nu + float(omega_I_seq[idx])) * delta_t
            gP_k = alpha * float(H_seq[idx]) * delta_t
            sP_k = 1.0 - (alpha * float(H_seq[idx]) + mu0 + float(omega_P_seq[idx])) * delta_t
            A = np.array([[sI_k, gP_k], [b_eff * nu * delta_t, sP_k]], dtype=float)
            rho_a = float(np.max(np.abs(linalg.eigvals(A))))
            Ak = np.linalg.matrix_power(A / rho_a, k_cnt)
            ls += k_cnt * math.log(rho_a)
            U = Ak @ U
            nrm = float(np.max(np.abs(U)))
            if nrm > 0.0:
                U /= nrm
                ls += math.log(nrm)
        rho = float(np.max(np.abs(linalg.eigvals(U))))
        return ls + math.log(max(rho, 1e-300))

    lo, hi = -2.0, 4.0
    while disc_monodromy_logrho(lo) > 0.0:
        lo *= 2.0
    while disc_monodromy_logrho(hi) < 0.0:
        hi *= 2.0
    D_disc = float(optimize.brentq(disc_monodromy_logrho, lo, hi, xtol=1e-9, rtol=1e-9))

    lg_P = -float(np.mean(np.log(gP_vec / gP_bar)))
    ls_P = -float(np.mean(np.log(sP_vec / sP_bar)))
    ls_I = -float(np.mean(np.log(sI_vec / sI_bar)))
    tau_P0 = 1.0 / (1.0 - sP_bar)
    tau_I0 = 1.0 / (1.0 - sI_bar)

    bound_transfer_only = lg_P
    stay_penalty = (tau_P0 - 1.0) * ls_P + (tau_I0 - 1.0) * ls_I
    bound_prop_b = lg_P + stay_penalty

    return {
        "delta_t_d": delta_t,
        "D_disc": D_disc,
        "bound_transfer_only": bound_transfer_only,
        "stay_penalty": stay_penalty,
        "bound_prop_b": bound_prop_b,
        "holds_prop_b": bool(D_disc <= bound_prop_b + 1e-6),
        "violates_transfer_only": bool(D_disc > bound_transfer_only + 1e-6),
    }


def engine_ar1_benchmark(
    tau_P_d: float,
    tau_I_d: float,
    d_thresh: float,
    sigma_lnH: float,
    phi_lag1: float,
    dt_obs_d: float,
    substages_I: int = 1,
) -> dict:
    """Call `lifecycle_engine.py` on a discretized AR(1) ring matched to the unit's (tau_P, tau_I, d, sigma, phi)."""
    m = int(substages_I)
    # Choose engine step dt_eng so tau_P_steps >= 2.5 and tau_I_steps >= max(2.0, m + 0.5)
    # AND bounded driver admissibility holds: stay(h_max) > 0 => c_step * h_max < 1 - mu_step
    var_req = min(math.expm1(min(sigma_lnH, 1.1) ** 2), 4.5)
    sig_use = math.sqrt(math.log1p(var_req))
    # With low = 0.05 (if var >= 1) or symmetric low = None (if var < 0.90):
    low_use = None if var_req < 0.85 else 0.05
    h_high = (1.0 + math.sqrt(var_req)) if low_use is None else (1.0 + var_req / 0.95)

    dt_max_stages = min(tau_P_d / 3.0, tau_I_d / (m + 1.0))
    # Ensure 1 - (1/tp_steps)*((1-d)*h_high + d) > 0.05 => tp_steps > ((1-d)*h_high + d) / 0.95
    min_tp_steps = max(3.0, ((1.0 - d_thresh) * h_high + d_thresh) / 0.85)
    dt_eng = min(dt_max_stages, tau_P_d / min_tp_steps)

    tp_s = tau_P_d / dt_eng
    ti_s = tau_I_d / dt_eng
    if phi_lag1 > 0.01:
        phi_eng = float(phi_lag1 ** (dt_eng / max(dt_obs_d, dt_eng)))
        phi_eng = min(max(phi_eng, 0.0), 0.995)
    else:
        phi_eng = 0.0

    desc = le.Description(
        stages=[
            le.Stage("P", mean=tp_s, shape="memoryless", death=d_thresh / tp_s, contact=True),
            le.Stage("I", mean=ti_s, shape=("chain" if m > 1 else "memoryless"), substages=m, death=0.0, contact=False),
        ],
        transfers=[
            le.Transfer("P", "I", yield_=1.0, driver="host", birth=False),
            le.Transfer("I", "P", yield_=20.0, driver=None, birth=True),
        ],
        drivers=[le.Driver("host", sigma=sig_use, phi=phi_eng, kind="bounded", low=low_use)],
    )
    eng = le.Engine(desc)
    coeff = eng.coefficient(threshold=True)
    return {
        "dt_eng_d": dt_eng,
        "sigma_eng": sig_use,
        "phi_eng": phi_eng,
        "engine_D2": float(coeff["D2"]),
        "engine_J_bound": float(eng.critical().jensen() / eng.critical().b),
        "engine_obj": eng,
    }


def partial_spearman_freedman_lane(
    y: np.ndarray,
    d_pred: np.ndarray,
    r_const: np.ndarray,
    seed: int = 20261114,
    n_perm: int = 10000,
) -> dict:
    """Exact/permutation Freedman-Lane rank-residual partial Spearman correlation."""
    n = len(y)
    ry = stats.rankdata(y)
    rd = stats.rankdata(d_pred)
    rc = stats.rankdata(r_const)
    X = np.column_stack([np.ones(n), rc])
    P_X = X @ np.linalg.pinv(X)
    M = np.eye(n) - P_X
    ey = M @ ry
    ed = M @ rd
    norm_ed = float(np.linalg.norm(ed))
    norm_ey = float(np.linalg.norm(ey))
    if norm_ed < 1e-12 or norm_ey < 1e-12:
        return {"n": n, "rho_part": 0.0, "p_neg": 1.0, "p_pos": 1.0, "method": "degenerate"}

    rho_obs = float((ey @ ed) / (norm_ey * norm_ed))
    yhat = P_X @ ry

    if n <= 7:
        vals = []
        for p in permutations(range(n)):
            ey_s = M @ (yhat + ey[list(p)])
            ne = float(np.linalg.norm(ey_s))
            vals.append(float((ey_s @ ed) / (ne * norm_ed)) if ne > 1e-12 else 0.0)
        arr = np.asarray(vals)
        p_neg = float(np.mean(arr <= rho_obs + 1e-15))
        p_pos = float(np.mean(arr >= rho_obs - 1e-15))
        method = f"exact_{len(vals)}_permutations"
    else:
        rng = np.random.default_rng(seed)
        c_neg = 1
        c_pos = 1
        for _ in range(n_perm):
            p = rng.permutation(n)
            ey_s = M @ (yhat + ey[p])
            ne = float(np.linalg.norm(ey_s))
            val = float((ey_s @ ed) / (ne * norm_ed)) if ne > 1e-12 else 0.0
            if val <= rho_obs + 1e-15:
                c_neg += 1
            if val >= rho_obs - 1e-15:
                c_pos += 1
        p_neg = c_neg / (n_perm + 1)
        p_pos = c_pos / (n_perm + 1)
        method = f"monte_carlo_{n_perm}_permutations"

    raw_rho, _ = stats.spearmanr(y, d_pred)
    return {
        "n": int(n),
        "rho_part": rho_obs,
        "rho_raw": float(raw_rho) if not math.isnan(raw_rho) else 0.0,
        "p_neg": p_neg,
        "p_pos": p_pos,
        "method": method,
    }
