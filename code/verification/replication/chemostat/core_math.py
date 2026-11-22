#!/usr/bin/env python3
"""Independent mathematical and statistical core for `r36-rep-chemostat`.

Clean-room implementation from the mathematical definitions in `protocol.md`
(§2, §4, §6 and Amendment 3) and `notes/contact-limited-transfer.md` /
`notes/mechanism-temperature-incubation.md`. Zero imports or copied code from
`casestudies/chemostat/`.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import numpy as np
from scipy import linalg, optimize


@dataclass(frozen=True)
class PhageLifeHistory:
    system_label: str
    latent_period_d: float      # L_lat (days)
    burst_size: float           # beta (virions per lysed cell)
    adsorption_ml_d: float      # alpha (mL cell^-1 d^-1)
    virion_decay_d: float       # mu_0 (d^-1)
    erlang_m: int = 1           # number of intracellular sub-stages m_I


REGISTERED_LIFE_HISTORY: dict[str, PhageLifeHistory] = {
    "D1_larsen2019": PhageLifeHistory(
        system_label="Synechococcus WH7803 + S-RIM8",
        latent_period_d=9.0 / 24.0,
        burst_size=50.0,
        adsorption_ml_d=6.06e-9 * 1440.0,
        virion_decay_d=0.03,
        erlang_m=1,
    ),
    "D2_reuter2026": PhageLifeHistory(
        system_label="Escherichia coli C + PhiX174",
        latent_period_d=17.0 / 1440.0,
        burst_size=200.0,
        adsorption_ml_d=4.0e-9 * 1440.0,
        virion_decay_d=1.7e-2 * 1440.0,
        erlang_m=1,
    ),
    "D3_chaudhry2018": PhageLifeHistory(
        system_label="Escherichia coli K12/B + lambdaVIR",
        latent_period_d=25.0 / 1440.0,
        burst_size=55.0,
        adsorption_ml_d=2.0e-7 * 24.0,
        virion_decay_d=0.10,
        erlang_m=1,
    ),
    "D4_duncan2017": PhageLifeHistory(
        system_label="Pseudomonas fluorescens SBW25 + SBW25phi2",
        latent_period_d=35.0 / 1440.0,
        burst_size=90.0,
        adsorption_ml_d=3.0e-9 * 1440.0,
        virion_decay_d=0.15,
        erlang_m=1,
    ),
    "D5_zhao2016": PhageLifeHistory(
        system_label="Pseudomonas fluorescens SBW25 + SBW25phi2 (OD)",
        latent_period_d=35.0 / 1440.0,
        burst_size=90.0,
        adsorption_ml_d=3.0e-9 * 1440.0,
        virion_decay_d=0.15,
        erlang_m=1,
    ),
}


def exact_2x2_scaled_expm(a: float, b: float, c: float, d: float, dt: float) -> tuple[np.ndarray, float]:
    """Analytical scaled matrix exponential for 2x2 Metzler matrix [[a, b], [c, d]] with b, c >= 0.

    Returns `(E_scaled, log_scale)` such that `expm(M * dt) = exp(log_scale) * E_scaled`,
    derived directly from the closed-form 2x2 spectral decomposition without Padé approximation.
    """
    tr = a + d
    diff = a - d
    disc = math.hypot(diff, 2.0 * math.sqrt(max(b * c, 0.0)))  # sqrt((a-d)^2 + 4bc) >= 0
    lam_max = 0.5 * (tr + disc)
    log_scale = lam_max * dt
    if disc * dt < 1e-12:
        # First-order confluent limit around 0.5 * tr
        half_diff = 0.5 * diff * dt
        e_shift = math.exp(-0.5 * disc * dt)
        mat = e_shift * np.array([
            [1.0 + half_diff, b * dt],
            [c * dt, 1.0 - half_diff],
        ], dtype=float)
        return mat, log_scale

    # With lam_1 = 0.5*(tr + disc), lam_2 = 0.5*(tr - disc):
    # exp(M*dt) = exp(lam_1*dt) * [ P_+ + exp(-disc*dt) * P_- ]
    # where P_+ + P_- = I and P_+ - P_- = (M - 0.5*tr*I) / (0.5*disc).
    z = math.exp(-disc * dt)
    avg_w = 0.5 * (1.0 + z)
    dif_w = 0.5 * (1.0 - z) / (0.5 * disc)  # -> dt as disc*dt -> 0
    h = 0.5 * diff
    mat = np.array([
        [avg_w + dif_w * h, dif_w * b],
        [dif_w * c, avg_w - dif_w * h],
    ], dtype=float)
    return mat, log_scale


def generator_matrix(
    host: float,
    host_mean: float,
    wash_p: float,
    wash_i: float,
    wash_p_mean: float,
    lh: PhageLifeHistory,
    scarcity_mode: str = "contact",
    birth_log_tilt: float = 0.0,
) -> np.ndarray:
    """Construct `(m_I + 1) x (m_I + 1)` continuous-time operator on `(I_1, ..., I_m, P)`."""
    m = int(lh.erlang_m)
    nu = 1.0 / lh.latent_period_d
    alpha = lh.adsorption_ml_d
    mu0 = lh.virion_decay_d
    ads = alpha * host

    mat = np.zeros((m + 1, m + 1), dtype=float)
    for idx in range(m):
        mat[idx, idx] = -(m * nu + wash_i)
        if idx + 1 < m:
            mat[idx + 1, idx] = m * nu
    mat[m, m - 1] = math.exp(birth_log_tilt) * lh.burst_size * (m * nu)
    mat[0, m] = ads
    if scarcity_mode == "contact":
        mat[m, m] = -(ads + mu0 + wash_p)
    elif scarcity_mode == "waste":
        mat[m, m] = -(alpha * host_mean + mu0 + wash_p_mean)
    else:
        raise ValueError(f"Unsupported scarcity_mode: {scarcity_mode}")
    return mat


def spectral_abscissa(mat: np.ndarray) -> float:
    """Largest real part of eigenvalues of `mat`."""
    if mat.shape == (2, 2):
        tr = float(mat[0, 0] + mat[1, 1])
        diff = float(mat[0, 0] - mat[1, 1])
        disc = math.sqrt(max(0.0, diff * diff + 4.0 * float(mat[0, 1] * mat[1, 0])))
        return 0.5 * (tr + disc)
    return float(np.max(linalg.eigvals(mat).real))


def floquet_lyapunov_ct(
    hosts: np.ndarray,
    dts: np.ndarray,
    wash_p: np.ndarray,
    wash_i: np.ndarray,
    lh: PhageLifeHistory,
    scarcity_mode: str = "contact",
    birth_log_tilt: float = 0.0,
    use_analytical_2x2: bool = True,
) -> float:
    """Top Floquet growth exponent (`d^-1`) over a piecewise-constant host/washout trajectory."""
    span = float(np.sum(dts))
    weights = dts / span
    h_mean = float(np.sum(weights * hosts))
    wp_mean = float(np.sum(weights * wash_p))

    m = int(lh.erlang_m)
    dim = m + 1
    monodromy = np.eye(dim, dtype=float)
    eye = np.eye(dim, dtype=float)
    acc_log = 0.0

    nu = 1.0 / lh.latent_period_d
    alpha = lh.adsorption_ml_d
    mu0 = lh.virion_decay_d
    burst_rate = math.exp(birth_log_tilt) * lh.burst_size * nu

    for k in range(len(hosts)):
        dt = float(dts[k])
        hk = float(hosts[k])
        wpk = float(wash_p[k])
        wik = float(wash_i[k])
        if m == 1 and use_analytical_2x2:
            a = -(nu + wik)
            b = alpha * hk
            c = burst_rate
            d = -(b + mu0 + wpk) if scarcity_mode == "contact" else -(alpha * h_mean + mu0 + wp_mean)
            step_mat, ls = exact_2x2_scaled_expm(a, b, c, d, dt)
        else:
            gen = generator_matrix(hk, h_mean, wpk, wik, wp_mean, lh, scarcity_mode, birth_log_tilt)
            shift = spectral_abscissa(gen)
            step_mat = linalg.expm((gen - shift * eye) * dt)
            ls = shift * dt

        acc_log += ls
        monodromy = step_mat @ monodromy
        scale = float(np.linalg.norm(monodromy, ord=np.inf))
        if scale > 0.0:
            monodromy /= scale
            acc_log += math.log(scale)

    spec_rad = float(np.max(np.abs(linalg.eigvals(monodromy))))
    return (acc_log + math.log(max(spec_rad, 1e-300))) / span


def evaluate_trajectory_ct(
    hosts: np.ndarray,
    dts: np.ndarray,
    wash_p: np.ndarray,
    wash_i: np.ndarray,
    lh: PhageLifeHistory,
    scarcity_mode: str = "contact",
) -> dict[str, float]:
    """Compute constant-mean growth, observed Floquet growth, threshold `d`, `ln R0`, and `D_sim`."""
    span = float(np.sum(dts))
    weights = dts / span
    h_bar = float(np.sum(weights * hosts))
    wp_bar = float(np.sum(weights * wash_p))
    wi_bar = float(np.sum(weights * wash_i))

    m = int(lh.erlang_m)
    nu = 1.0 / lh.latent_period_d
    alpha = lh.adsorption_ml_d
    mu0 = lh.virion_decay_d

    gen_const = generator_matrix(h_bar, h_bar, wp_bar, wi_bar, wp_bar, lh, scarcity_mode, 0.0)
    r_const = spectral_abscissa(gen_const)
    r_obs = floquet_lyapunov_ct(hosts, dts, wash_p, wash_i, lh, scarcity_mode, 0.0)

    d_val = (mu0 + wp_bar) / (alpha * h_bar + mu0 + wp_bar)
    surv_i = ((m * nu) / (m * nu + wi_bar)) ** m
    r0_val = (1.0 - d_val) * lh.burst_size * surv_i
    ln_r0 = math.log(max(r0_val, 1e-300))

    tau_p = 1.0 / (alpha * h_bar + mu0 + wp_bar)
    tau_i = m / (m * nu + wi_bar)

    def target(d_loss: float) -> float:
        return floquet_lyapunov_ct(
            hosts, dts, wash_p, wash_i, lh, scarcity_mode, birth_log_tilt=-ln_r0 + d_loss
        )

    lo, hi = -2.0, 4.0
    while target(lo) > 0.0 and lo > -64.0:
        lo *= 2.0
    while target(hi) < 0.0 and hi < 128.0:
        hi *= 2.0
    d_sim = float(optimize.brentq(target, lo, hi, xtol=1e-11, rtol=1e-11))

    return {
        "H_bar": h_bar,
        "omega_P_bar": wp_bar,
        "omega_I_bar": wi_bar,
        "d_threshold": d_val,
        "tau_P_d": tau_p,
        "tau_I_d": tau_i,
        "T0_d": tau_p + tau_i,
        "R0_mean": r0_val,
        "ln_R0_mean": ln_r0,
        "r_const_d": r_const,
        "r_obs_d": r_obs,
        "Delta_r_fluct_d": r_const - r_obs,
        "D_sim": d_sim,
    }


def closed_form_c1_discrete(tau_p: float, tau_i: float, d: float, phi: float, sigma: float, m_i: int = 1) -> float:
    """Discrete-time 2nd-order contact-limited loss `Gamma_c` (`notes/contact-limited-transfer.md` eq. C1)."""
    t_gen = tau_p + tau_i
    if abs(phi) < 1e-14:
        return (sigma * sigma) * (d * d) / (2.0 * t_gen)
    theta = phi / (1.0 - phi)
    qp = 1.0 / (1.0 + tau_p / theta)
    qi = (1.0 + tau_i / (m_i * theta)) ** (-m_i)
    r_prod = qp * qi
    return (sigma * sigma) * (
        (d * d) * (1.0 + 2.0 * theta) / (2.0 * t_gen)
        - d * (r_prod - (1.0 - d) * qp) / (1.0 - r_prod)
    )


def closed_form_c2_continuous(tau_p: float, tau_i: float, d: float, theta_ct: float, sigma: float, m_i: int = 1) -> float:
    """Continuous-time 2nd-order contact-limited loss `Gamma_c` (`protocol.md` eq. 5 / `C2`)."""
    t_gen = tau_p + tau_i
    qp = 1.0 / (1.0 + tau_p / theta_ct)
    qi = (1.0 + tau_i / (m_i * theta_ct)) ** (-m_i)
    r_prod = qp * qi
    return (sigma * sigma) * (
        (d * d) * theta_ct / t_gen
        - d * (r_prod - (1.0 - d) * qp) / (1.0 - r_prod)
    )


def host_moments_and_gamma_c(
    hosts: np.ndarray,
    dts: np.ndarray,
    tau_p_d: float,
    tau_i_d: float,
    d_val: float,
    m_i: int = 1,
) -> dict[str, float | str]:
    """Compute host variability/memory summaries and both registered `Gamma_c` and corrected piecewise-constant limit."""
    log_h = np.log(hosts)
    k_len = len(hosts)
    weights = dts / float(np.sum(dts))
    h_bar = float(np.sum(weights * hosts))

    sigma_lnh = float(np.std(log_h, ddof=1)) if k_len >= 2 else 0.0
    cv_h = float(np.std(hosts / h_bar, ddof=1)) if k_len >= 2 else 0.0
    if k_len >= 3 and np.std(log_h[:-1]) > 1e-12 and np.std(log_h[1:]) > 1e-12:
        phi_1 = float(np.corrcoef(log_h[:-1], log_h[1:])[0, 1])
    else:
        phi_1 = 0.0

    dt_med = float(np.median(dts))
    t0 = tau_p_d + tau_i_d

    if 0.0 < phi_1 < 0.9999:
        theta_ct = -dt_med / math.log(phi_1)
        gamma_c_reg = closed_form_c2_continuous(tau_p_d, tau_i_d, d_val, theta_ct, sigma_lnh, m_i=m_i)
        qp = 1.0 / (1.0 + tau_p_d / theta_ct)
        qi = (1.0 + tau_i_d / (m_i * theta_ct)) ** (-m_i)
        r_prod = qp * qi
        gamma_w_reg = (sigma_lnh ** 2) * (theta_ct / t0 - r_prod / (1.0 - r_prod))
        branch = "CT_C2"
    else:
        # Registered fallback rule (`protocol.md` §2.2 / `model_helpers.py:306-326`):
        theta_ct = 0.0
        dt_disc = min(dt_med, 0.4 * min(tau_p_d, tau_i_d / m_i))
        tp_s = tau_p_d / dt_disc
        ti_s = tau_i_d / dt_disc
        phi_use = float(np.clip(phi_1, -0.95, 0.0)) if abs(dt_disc - dt_med) < 1e-9 else 0.0
        gamma_c_reg = closed_form_c1_discrete(tp_s, ti_s, d_val, phi_use, sigma_lnh, m_i=m_i)
        gamma_w_reg = closed_form_c1_discrete(tp_s, ti_s, 1.0, phi_use, sigma_lnh, m_i=m_i)
        branch = "disc_C1_fallback"

    # Also compute physically consistent piecewise-constant step correlation time theta_pw = 0.5 * dt_med
    # when phi_1 <= 0 (for audit comparison only; not used to replace registered primary numbers):
    theta_audit = theta_ct if (0.0 < phi_1 < 0.9999) else (0.5 * dt_med * max(0.05, (1.0 + phi_1) / (1.0 - phi_1)))
    gamma_c_pw_audit = closed_form_c2_continuous(tau_p_d, tau_i_d, d_val, theta_audit, sigma_lnh, m_i=m_i)

    return {
        "sigma_lnH": sigma_lnh,
        "cv_H": cv_h,
        "phi_lag1": phi_1,
        "theta_ct_d": theta_ct,
        "Gamma_c_2nd": gamma_c_reg,
        "Gamma_waste_2nd": gamma_w_reg,
        "Gamma_c_pw_audit": gamma_c_pw_audit,
        "branch": branch,
    }


def verify_proposition_b_bound(
    hosts: np.ndarray,
    dts: np.ndarray,
    wash_p: np.ndarray,
    wash_i: np.ndarray,
    lh: PhageLifeHistory,
    max_step_exit_prob: float = 0.15,
) -> dict[str, float | bool]:
    """Check Proposition B fluctuating-staying bound on an admissible discretized trajectory."""
    nu = 1.0 / lh.latent_period_d
    alpha = lh.adsorption_ml_d
    mu0 = lh.virion_decay_d

    peak_exit = max(
        float(np.max(alpha * hosts + mu0 + wash_p)),
        float(np.max(nu + wash_i)),
        1.0,
    )
    dt_min = float(np.min(dts))
    n_div = max(1, int(math.ceil(dt_min * peak_exit / max_step_exit_prob)))
    dt_step = dt_min / n_div
    counts = np.array([max(1, int(round(float(d) / dt_step))) for d in dts], dtype=int)
    total_steps = int(np.sum(counts))

    s_i_seg = 1.0 - (nu + wash_i) * dt_step
    g_p_seg = alpha * hosts * dt_step
    s_p_seg = 1.0 - (alpha * hosts + mu0 + wash_p) * dt_step
    if np.any(s_i_seg <= 0.0) or np.any(s_p_seg <= 0.0):
        raise ValueError("Inadmissible discrete step: non-positive staying probability.")

    weights = counts / float(total_steps)
    s_i_bar = float(np.sum(weights * s_i_seg))
    g_p_bar = float(np.sum(weights * g_p_seg))
    s_p_bar = float(np.sum(weights * s_p_seg))
    g_i_const = nu * dt_step

    beta_thresh = (1.0 - s_i_bar) * (1.0 - s_p_bar) / (g_i_const * g_p_bar)

    def log_monodromy_radius(d_loss: float) -> float:
        u_mat = np.eye(2, dtype=float)
        acc = 0.0
        b_now = beta_thresh * math.exp(d_loss)
        for seg_idx in range(len(hosts)):
            k_rep = int(counts[seg_idx])
            a_step = np.array([
                [float(s_i_seg[seg_idx]), float(g_p_seg[seg_idx])],
                [b_now * g_i_const, float(s_p_seg[seg_idx])],
            ], dtype=float)
            # Exact 2x2 Perron/eigen power via matrix_power after spectral scaling
            tr = float(a_step[0, 0] + a_step[1, 1])
            df = float(a_step[0, 0] - a_step[1, 1])
            rho_step = 0.5 * (tr + math.sqrt(max(0.0, df * df + 4.0 * float(a_step[0, 1] * a_step[1, 0]))))
            blk = np.linalg.matrix_power(a_step / rho_step, k_rep)
            acc += k_rep * math.log(rho_step)
            u_mat = blk @ u_mat
            nrm = float(np.linalg.norm(u_mat, ord=np.inf))
            if nrm > 0.0:
                u_mat /= nrm
                acc += math.log(nrm)
        rad = float(np.max(np.abs(linalg.eigvals(u_mat))))
        return acc + math.log(max(rad, 1e-300))

    lo, hi = -2.0, 4.0
    while log_monodromy_radius(lo) > 0.0:
        lo *= 2.0
    while log_monodromy_radius(hi) < 0.0:
        hi *= 2.0
    d_disc = float(optimize.brentq(log_monodromy_radius, lo, hi, xtol=1e-11, rtol=1e-11))

    ell_g_p = -float(np.sum(weights * np.log(g_p_seg / g_p_bar)))
    ell_s_p = -float(np.sum(weights * np.log(s_p_seg / s_p_bar)))
    ell_s_i = -float(np.sum(weights * np.log(s_i_seg / s_i_bar)))
    tau_p0 = 1.0 / (1.0 - s_p_bar)
    tau_i0 = 1.0 / (1.0 - s_i_bar)

    stay_term = (tau_p0 - 1.0) * ell_s_p + (tau_i0 - 1.0) * ell_s_i
    prop_b_rhs = ell_g_p + stay_term

    return {
        "dt_step_d": dt_step,
        "total_steps": total_steps,
        "D_disc": d_disc,
        "bound_transfer_only": ell_g_p,
        "stay_penalty": stay_term,
        "bound_prop_b": prop_b_rhs,
        "holds_prop_b": bool(d_disc <= prop_b_rhs + 1e-6),
        "violates_transfer_only": bool(d_disc > ell_g_p + 1e-6),
    }


def average_tie_ranks(vals: np.ndarray) -> np.ndarray:
    """Compute 1-based midranks (average ranks for ties) without `scipy.stats`."""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        mid = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = mid
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Exact Pearson correlation on midranks; returns `None` if either vector is constant."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2:
        return None
    rx = average_tie_ranks(x_arr)
    ry = average_tie_ranks(y_arr)
    dx = rx - np.mean(rx)
    dy = ry - np.mean(ry)
    nx = float(np.dot(dx, dx))
    ny = float(np.dot(dy, dy))
    if nx <= 1e-15 or ny <= 1e-15:
        return None
    return float(np.dot(dx, dy) / math.sqrt(nx * ny))


def freedman_lane_partial_spearman(
    y: np.ndarray,
    x_pred: np.ndarray,
    z_cov: np.ndarray,
    seed: int = 20261114,
    n_perm: int = 10000,
) -> dict[str, float | int | str]:
    """Freedman-Lane rank-residual partial Spearman correlation `rho_part(y, x | z)`.

    Permutes reduced-model residuals `e_{y|z}` (`y* = yhat_z + P * e_{y|z}`),
    noting that `M_z (yhat_z + P e_{y|z}) = M_z P e_{y|z}` since `M_z yhat_z = 0`.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x_pred, dtype=float)
    z_arr = np.asarray(z_cov, dtype=float)
    n = len(y_arr)

    ry = average_tie_ranks(y_arr)
    rx = average_tie_ranks(x_arr)
    if z_arr.ndim == 1:
        rz = average_tie_ranks(z_arr)[:, None]
    else:
        rz = np.column_stack([average_tie_ranks(z_arr[:, col]) for col in range(z_arr.shape[1])])

    design = np.column_stack([np.ones(n, dtype=float), rz])
    q_orth, _ = np.linalg.qr(design, mode="reduced")
    # Residualize via orthogonal complement I - Q Q^T
    ey = ry - q_orth @ (q_orth.T @ ry)
    ex = rx - q_orth @ (q_orth.T @ rx)

    norm_ey = float(np.linalg.norm(ey))
    norm_ex = float(np.linalg.norm(ex))
    if norm_ey <= 1e-12 or norm_ex <= 1e-12:
        return {"n": n, "rho_part": 0.0, "rho_raw": 0.0, "p_neg": 1.0, "p_pos": 1.0, "method": "degenerate"}

    unit_ex = ex / norm_ex
    rho_obs = float(np.dot(ey, unit_ex) / norm_ey)
    raw = spearman_corr(y_arr, x_arr)
    rho_raw = 0.0 if raw is None else raw

    if n <= 7:
        perm_rhos = []
        for perm in itertools.permutations(range(n)):
            ey_p = ey[list(perm)]
            ey_proj = ey_p - q_orth @ (q_orth.T @ ey_p)
            nrm = float(np.linalg.norm(ey_proj))
            perm_rhos.append(float(np.dot(ey_proj, unit_ex) / nrm) if nrm > 1e-12 else 0.0)
        arr_p = np.asarray(perm_rhos, dtype=float)
        p_neg = float(np.mean(arr_p <= rho_obs + 1e-15))
        p_pos = float(np.mean(arr_p >= rho_obs - 1e-15))
        method = f"exact_{len(perm_rhos)}_permutations"
    else:
        rng = np.random.default_rng(seed)
        cnt_neg = 1
        cnt_pos = 1
        for _ in range(n_perm):
            perm = rng.permutation(n)
            ey_p = ey[perm]
            ey_proj = ey_p - q_orth @ (q_orth.T @ ey_p)
            nrm = float(np.linalg.norm(ey_proj))
            val = float(np.dot(ey_proj, unit_ex) / nrm) if nrm > 1e-12 else 0.0
            if val <= rho_obs + 1e-15:
                cnt_neg += 1
            if val >= rho_obs - 1e-15:
                cnt_pos += 1
        p_neg = cnt_neg / (n_perm + 1)
        p_pos = cnt_pos / (n_perm + 1)
        method = f"monte_carlo_{n_perm}_permutations"

    # Fisher-z approximate 95% interval on partial rank correlation (df = n - p_z - 2)
    df_eff = n - design.shape[1] - 1
    if df_eff >= 2 and abs(rho_obs) < 0.999999:
        z_f = math.atanh(rho_obs)
        se_z = 1.0 / math.sqrt(df_eff)
        ci_95 = [math.tanh(z_f - 1.959963984540054 * se_z), math.tanh(z_f + 1.959963984540054 * se_z)]
    else:
        ci_95 = [math.nan, math.nan]

    return {
        "n": int(n),
        "rho_part": rho_obs,
        "rho_raw": rho_raw,
        "p_neg": p_neg,
        "p_pos": p_pos,
        "fisher_z_95ci": ci_95,
        "method": method,
    }


def classify_t1_outcome(stat: dict, feedback_sigma_ratio: float) -> str:
    """Registered `T1` decision rule (`protocol.md` §6.2)."""
    rho = float(stat["rho_part"])
    p_neg = float(stat["p_neg"])
    p_pos = float(stat["p_pos"])
    if rho < 0.0 and p_neg < 0.05:
        return "supports"
    if rho > 0.0 and p_pos < 0.05:
        if not math.isnan(feedback_sigma_ratio) and feedback_sigma_ratio > 2.0:
            return "not decided (confounded by endogenous predator-prey feedback)"
        return "contradicts"
    return "not decided"
