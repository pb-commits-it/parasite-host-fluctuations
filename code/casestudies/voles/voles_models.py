#!/usr/bin/env python3
"""Mathematical, statistical, lifecycle-engine, and thermal-stage helpers for `voles`.

Implements:
  - Exact Gaussian maximum-likelihood host-memory fitters (single-site and pooled
    multi-site) for M0 (iid), M1 (AR1), M2 (CAR2 quasi-cycle), M3 (Seasonal + AR1),
    and M4 (Seasonal + CAR2), plus implied ACFs and normalised spectral densities.
  - Statement ring-B8b spectral R0 loss and Statement ring-B2 AR(1) loss.
  - K9 linearised prevalence frequency responses (Blocks S, L, SIR serology, and
    harmed-host Block V from `notes/coupled-hosts.md`).
  - Pooled within-site cross-correlations,weather-partial CCFs, phase-randomised
    surrogate tests, and within-site block-bootstrap CIs for L_com ordering.
  - Engine.loss() and Engine.coefficient() wrapper for finite-noise D, J, stay
    shapes, contact-limited waiting, spatial patches, alternative hosts, and
    backward-PGF trough extinction.
  - Thermal stage-duration & completion models (`notes/mechanism-temperature-incubation.md`
    §1, Proposition A §2.1, Proposition B §2.2, §3.2) driven by observed 28-day
    temperature and host trajectories.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
from scipy import linalg, optimize

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
ENGINE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "studies", "engine-R16"))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

from lifecycle_engine import Description, Driver, Engine, ModelError, Patches, Stage, Transfer  # noqa: E402
from protocol_power import discrete_ring_spectral_cost, k9_transfer_functions, theoretical_ccf_from_spectrum  # noqa: E402

PERIOD_YR_STEPS = 365.25 / 28.0  # 13.044642857 steps per solar year
OMEGA_YR = 2.0 * math.pi / PERIOD_YR_STEPS


# ---------------------------------------------------------------------------
# 1. Host Memory Models (Exact Gaussian MLE on Gapped Single or Multi-Site Data)
# ---------------------------------------------------------------------------

def car2_correlation_from_lags(dt_abs: np.ndarray, zeta: float, om0: float, nug: float) -> np.ndarray:
    """Continuous-time damped harmonic oscillator (CAR(2)) correlation with nugget:
      rho(tau) = (1 - nug) * exp(-zeta * tau) * [cos(om0 * tau) + (zeta / om0) * sin(om0 * tau)] + nug * I_{tau=0}.
    Positive-definite on any real time grid because its spectral density is strictly positive.
    """
    c = np.exp(-zeta * dt_abs) * (np.cos(om0 * dt_abs) + (zeta / max(om0, 1e-8)) * np.sin(om0 * dt_abs))
    out = (1.0 - nug) * c
    out[dt_abs == 0.0] = 1.0
    return out


def car2_two_sided_unit_psd(om: np.ndarray, zeta: float, om0: float) -> np.ndarray:
    """Continuous-time two-sided PSD normalised so (1/pi) int_0^inf S(om) dom = 1:
      int_{-inf}^{inf} [4*zeta*w2 / ((w2 - om^2)^2 + 4*zeta^2*om^2)] dom / (2*pi) = 1.
    """
    w2 = zeta ** 2 + om0 ** 2
    return (4.0 * zeta * w2) / ((w2 - om ** 2) ** 2 + 4.0 * (zeta ** 2) * (om ** 2))


def discrete_ar2_acf(max_lag: int, phi1: float, phi2: float) -> np.ndarray:
    """Exact stationary autocorrelation rho(0..max_lag) of discrete-time AR(2)
    z_t = phi1 * z_{t-1} + phi2 * z_{t-2} + eps_t inside the stationarity triangle.
    """
    rho = np.zeros(max_lag + 1, dtype=float)
    rho[0] = 1.0
    if max_lag >= 1:
        rho[1] = phi1 / max(1.0 - phi2, 1e-12)
    for ell in range(2, max_lag + 1):
        rho[ell] = phi1 * rho[ell - 1] + phi2 * rho[ell - 2]
    return rho


def discrete_ar2_unit_psd(om: np.ndarray, phi1: float, phi2: float) -> np.ndarray:
    """Exact discrete-time unit-variance PSD f(omega) on (0, pi) satisfying
    (1/pi) int_0^pi f(omega) d_omega = 1.
    """
    v_ratio = (1.0 + phi2) * (((1.0 - phi2) ** 2) - (phi1 ** 2)) / max(1.0 - phi2, 1e-12)
    denom = np.abs(1.0 - phi1 * np.exp(-1j * om) - phi2 * np.exp(-2j * om)) ** 2
    return np.maximum(v_ratio, 1e-15) / np.maximum(denom, 1e-15)


def fit_host_memory_panel(
    panel: list[tuple[str, np.ndarray, np.ndarray]],
    period_steps: float = PERIOD_YR_STEPS,
) -> dict[str, Any]:
    """Fit M0..M4 by exact Gaussian maximum likelihood on 1 or G sites.

    `panel`: list of `(site_name, t_steps_float, z_raw_log_H)` where each site has
    its own intercept (mean log density) and all sites share covariance parameters
    (and, in M3/M4, shared seasonal cos/sin coefficients).
    """
    g_sites = len(panel)
    y_all = np.concatenate([z for _, _, z in panel])
    t_all = np.concatenate([t for _, t, _ in panel])
    n_tot = len(y_all)
    om_yr = 2.0 * math.pi / period_steps

    # Build design matrices: G site intercepts (+ 2 seasonal columns in M3/M4)
    X_int = np.zeros((n_tot, g_sites))
    idx0 = 0
    dt_blocks = []
    for g, (_, tg, zg) in enumerate(panel):
        ng = len(zg)
        X_int[idx0 : idx0 + ng, g] = 1.0
        dt_blocks.append(np.abs(tg[:, None] - tg[None, :]))
        idx0 += ng
    X_seas = np.column_stack([X_int, np.cos(om_yr * t_all), np.sin(om_yr * t_all)])

    def eval_gls(X: np.ndarray, corr_blocks: list[np.ndarray]) -> tuple[float, np.ndarray, float]:
        logdet = 0.0
        Rinv_X_list = []
        Rinv_y_list = []
        i0 = 0
        for Rb in corr_blocks:
            nb = Rb.shape[0]
            Xb = X[i0 : i0 + nb]
            yb = y_all[i0 : i0 + nb]
            try:
                cho = linalg.cho_factor(Rb + 1e-10 * np.eye(nb), lower=True, check_finite=False)
            except linalg.LinAlgError:
                return 1e18, np.zeros(X.shape[1]), 1.0
            logdet += 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
            Rinv_X_list.append(linalg.cho_solve(cho, Xb, check_finite=False))
            Rinv_y_list.append(linalg.cho_solve(cho, yb, check_finite=False))
            i0 += nb
        Rinv_X = np.vstack(Rinv_X_list)
        Rinv_y = np.concatenate(Rinv_y_list)
        XtRX = X.T @ Rinv_X + 1e-12 * np.eye(X.shape[1])
        beta = linalg.solve(XtRX, X.T @ Rinv_y, assume_a="pos")
        resid = y_all - X @ beta
        # Recompute exact quadratic form block by block
        quad = 0.0
        i0 = 0
        for Rb in corr_blocks:
            nb = Rb.shape[0]
            rb = resid[i0 : i0 + nb]
            cho = linalg.cho_factor(Rb + 1e-10 * np.eye(nb), lower=True, check_finite=False)
            quad += float(rb @ linalg.cho_solve(cho, rb, check_finite=False))
            i0 += nb
        sigma2 = max(quad / n_tot, 1e-15)
        nll = 0.5 * (n_tot * math.log(2.0 * math.pi * sigma2) + logdet + n_tot)
        return nll, beta, sigma2

    def aicc(nll: float, k_params: int) -> float:
        aic = 2.0 * nll + 2.0 * k_params
        if n_tot - k_params - 1 > 0:
            return aic + (2.0 * k_params * (k_params + 1)) / (n_tot - k_params - 1)
        return aic + 1e6

    # M0: White noise (k = G + 1)
    nll0, beta0, s2_0 = eval_gls(X_int, [np.eye(len(b)) for b in dt_blocks])

    # M1: Continuous/discrete AR(1) phi in (0, 1) (plus check negative phi if all lags integer)
    def nll_ar1(X_mat: np.ndarray) -> tuple[float, float, np.ndarray, float]:
        best = (1e18, 0.0, np.zeros(X_mat.shape[1]), 1.0)
        for lo, hi in [(-7.0, 7.0)]:
            res = optimize.minimize_scalar(
                lambda lp: eval_gls(X_mat, [(1.0 / (1.0 + math.exp(-lp))) ** b for b in dt_blocks])[0],
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-6},
            )
            phi_v = 1.0 / (1.0 + math.exp(-float(res.x)))
            nl, bt, s2 = eval_gls(X_mat, [phi_v ** b for b in dt_blocks])
            if nl < best[0]:
                best = (nl, phi_v, bt, s2)
        # Check if all dt are integers -> also allow negative AR(1) phi in (-0.99, 0]
        if all(np.allclose(b, np.round(b), atol=1e-8) for b in dt_blocks):
            res_neg = optimize.minimize_scalar(
                lambda p_neg: eval_gls(X_mat, [(p_neg) ** np.round(b) for b in dt_blocks])[0],
                bounds=(-0.98, 0.0),
                method="bounded",
                options={"xatol": 1e-6},
            )
            pn = float(res_neg.x)
            nl, bt, s2 = eval_gls(X_mat, [pn ** np.round(b) for b in dt_blocks])
            if nl < best[0]:
                best = (nl, pn, bt, s2)
        return best

    nll1, phi1, beta1, s2_1 = nll_ar1(X_int)
    # Ensure nested M1 <= M0
    if nll0 < nll1:
        nll1, phi1, beta1, s2_1 = nll0, 0.0, beta0, s2_0

    # M3: Seasonal + AR(1)
    nll3, phi3, beta3, s2_3 = nll_ar1(X_seas)
    nll3_iid, beta3_iid, s2_3_iid = eval_gls(X_seas, [np.eye(len(b)) for b in dt_blocks])
    if nll3_iid < nll3:
        nll3, phi3, beta3, s2_3 = nll3_iid, 0.0, beta3_iid, s2_3_iid

    # M2 & M4: CAR(2) + nugget (multi-start Powell + Nelder-Mead polish so nested NLL <= M1/M3)
    def fit_car2(X_mat: np.ndarray, init_phi_nest: float) -> tuple[float, dict[str, float], np.ndarray, float]:
        def unpack(p):
            zeta = math.exp(float(np.clip(p[0], -5.5, 3.5)))
            om0 = math.pi / (1.0 + math.exp(-float(np.clip(p[1], -9.0, 9.5)))) + 1e-5
            nug = 0.98 / (1.0 + math.exp(-float(np.clip(p[2], -10.0, 8.0))))
            return zeta, om0, nug

        def obj(p):
            zeta, om0, nug = unpack(p)
            return eval_gls(X_mat, [car2_correlation_from_lags(b, zeta, om0, nug) for b in dt_blocks])[0]

        z_from_phi = math.log(max(-math.log(max(abs(init_phi_nest), 1e-3)), 1e-2))
        starts = [
            [z_from_phi, -8.0, -8.0],        # positive damped limit (om0 -> 0)
            [z_from_phi, 9.0, -8.0],         # alternating limit (om0 -> pi, matches negative AR1 on integers)
            [3.4, 0.0, 6.0],                 # white-noise limit
            [-2.2, -1.9, -4.5],              # ~3-4 yr multiannual cycle (om0 ~ 0.15)
            [-1.6, -0.85, -3.5],             # ~1 yr annual cycle (om0 ~ 0.48)
            [-1.2, 0.0, -3.0],               # ~4-5 step oscillation
            [-2.6, -2.2, -2.0],              # slow cycle with observation nugget
        ]
        best_f = 1e18
        best_p = starts[0]
        for s0 in starts:
            r_nm = optimize.minimize(obj, x0=s0, method="Nelder-Mead", options={"maxiter": 350, "xatol": 1e-4, "fatol": 1e-4})
            if r_nm.fun < best_f:
                best_f = float(r_nm.fun)
                best_p = list(r_nm.x)
        r_pol = optimize.minimize(obj, x0=best_p, method="Powell", options={"maxiter": 400, "xtol": 1e-5, "ftol": 1e-6})
        if r_pol.fun < best_f:
            best_p = list(r_pol.x)
        zeta, om0, nug = unpack(best_p)
        nl, bt, s2 = eval_gls(X_mat, [car2_correlation_from_lags(b, zeta, om0, nug) for b in dt_blocks])
        pars = {
            "zeta": zeta,
            "r_modulus_per_step": math.exp(-zeta),
            "omega_0_rad_per_step": om0,
            "period_steps": (2.0 * math.pi / om0) if om0 > 1e-3 else float("inf"),
            "period_years": (2.0 * math.pi / om0) / period_steps if om0 > 1e-3 else float("inf"),
            "nugget": nug,
        }
        return nl, pars, bt, s2

    nll2, pars2, beta2, s2_2 = fit_car2(X_int, phi1)
    nll4, pars4, beta4, s2_4 = fit_car2(X_seas, phi3)

    # Companion exact discrete-time AR(2) (via PACF bijection pacf1, pacf2 in (-1, 1)^2)
    max_int_lag = max(int(np.max(np.round(b))) for b in dt_blocks) + 2
    int_blocks = [np.round(b).astype(int) for b in dt_blocks]

    def fit_disc_ar2(X_mat: np.ndarray, init_phi: float) -> tuple[float, dict[str, float | None], np.ndarray, float]:
        def unpack_p(u):
            p1_pacf = math.tanh(float(np.clip(u[0], -4.5, 4.5)))
            p2_pacf = math.tanh(float(np.clip(u[1], -4.5, 4.5)))
            ph2 = p2_pacf
            ph1 = p1_pacf * (1.0 - ph2)
            return ph1, ph2

        def obj_d(u):
            ph1, ph2 = unpack_p(u)
            r_vec = discrete_ar2_acf(max_int_lag, ph1, ph2)
            return eval_gls(X_mat, [r_vec[ib] for ib in int_blocks])[0]

        u1_0 = math.atanh(float(np.clip(init_phi, -0.98, 0.98)))
        starts_d = [[u1_0, 0.0], [1.2, -0.8], [0.6, -0.6], [1.5, -1.1], [0.0, -0.5]]
        best_fd = 1e18
        best_ud = starts_d[0]
        for sd0 in starts_d:
            rd = optimize.minimize(obj_d, x0=sd0, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-5})
            if rd.fun < best_fd:
                best_fd = float(rd.fun)
                best_ud = list(rd.x)
        ph1, ph2 = unpack_p(best_ud)
        r_vec = discrete_ar2_acf(max_int_lag, ph1, ph2)
        nl_d, bt_d, s2_d = eval_gls(X_mat, [r_vec[ib] for ib in int_blocks])
        disc = ph1 ** 2 + 4.0 * ph2
        if disc < 0.0:
            r_mod = math.sqrt(-ph2)
            om_d = math.acos(float(np.clip(ph1 / max(2.0 * r_mod, 1e-12), -1.0, 1.0)))
            per_st = (2.0 * math.pi / om_d) if om_d > 1e-4 else None
            per_yr = (per_st / period_steps) if per_st is not None else None
        else:
            r_mod = None
            om_d = None
            per_st = None
            per_yr = None
        return nl_d, {
            "phi1": round(ph1, 4),
            "phi2": round(ph2, 4),
            "complex_roots": bool(disc < 0.0),
            "root_modulus": round(r_mod, 4) if r_mod is not None else None,
            "quasi_period_steps": round(per_st, 3) if per_st is not None else None,
            "quasi_period_years": round(per_yr, 3) if per_yr is not None else None,
        }, bt_d, s2_d

    nll2_d, pars2_d, _, s2_2d = fit_disc_ar2(X_int, phi1)
    nll4_d, pars4_d, beta4_d, s2_4d = fit_disc_ar2(X_seas, phi3)

    k0, k1, k2, k3, k4 = g_sites + 1, g_sites + 2, g_sites + 4, g_sites + 4, g_sites + 6
    k2_d, k4_d = g_sites + 3, g_sites + 5
    aicc_map = {
        "M0_iid": aicc(nll0, k0),
        "M1_ar1": aicc(nll1, k1),
        "M2_osc": aicc(nll2, k2),
        "M3_seas_ar1": aicc(nll3, k3),
        "M4_seas_osc": aicc(nll4, k4),
    }
    best_model = min(aicc_map, key=aicc_map.get)
    delta_beyond_ar1 = min(aicc_map["M0_iid"], aicc_map["M1_ar1"]) - min(
        aicc_map["M2_osc"], aicc_map["M3_seas_ar1"], aicc_map["M4_seas_osc"]
    )
    if delta_beyond_ar1 >= 4.0:
        verdict_t1 = "supports"
    elif delta_beyond_ar1 <= -2.0:
        verdict_t1 = "contradicts_structured_supports_ar1"
    else:
        verdict_t1 = "not_decided"

    # Compute implied ACF(0..12) and two-sided continuous PSD S_z(omega) for AR(1) and Best Model
    lags = np.arange(0, 13, dtype=float)
    ar1_acf = [float(phi1 ** l) for l in range(13)]

    def _seas_var_and_acf(bt: np.ndarray, s2_res: float, res_acf: np.ndarray) -> tuple[float, np.ndarray]:
        amp2 = float(bt[-2] ** 2 + bt[-1] ** 2)
        v_seas = 0.5 * amp2
        v_tot = v_seas + s2_res
        acf_tot = (v_seas * np.cos(om_yr * lags) + s2_res * res_acf) / max(v_tot, 1e-15)
        acf_tot[0] = 1.0
        return v_seas / max(v_tot, 1e-15), acf_tot

    if best_model == "M0_iid":
        best_acf = [1.0] + [0.0] * 12
        seas_share = 0.0
    elif best_model == "M1_ar1":
        best_acf = ar1_acf
        seas_share = 0.0
    elif best_model == "M2_osc":
        best_acf = car2_correlation_from_lags(lags, pars2["zeta"], pars2["omega_0_rad_per_step"], pars2["nugget"]).tolist()
        seas_share = 0.0
    elif best_model == "M3_seas_ar1":
        seas_share, acf_arr = _seas_var_and_acf(beta3, s2_3, phi3 ** lags)
        best_acf = acf_arr.tolist()
    else:
        r_acf4 = car2_correlation_from_lags(lags, pars4["zeta"], pars4["omega_0_rad_per_step"], pars4["nugget"])
        seas_share, acf_arr = _seas_var_and_acf(beta4, s2_4, r_acf4)
        best_acf = acf_arr.tolist()

    # Build continuous-time two-sided PSD S_z(omega) for K9 & discrete PSD f_H(omega) on (0, pi) for ring-B8b
    def _norm_on_pi(fn):
        g = np.linspace(1e-5, math.pi - 1e-5, 4096)
        nrm = float(np.trapezoid(fn(g), g) / math.pi)
        return lambda om: fn(om) / max(nrm, 1e-30)

    def make_best_psd_fns():
        if best_model == "M0_iid":
            return (lambda om: np.where(np.abs(om) <= math.pi, s2_0, 0.0)), (lambda om: np.ones_like(om))
        elif best_model == "M1_ar1":
            al = max(-math.log(max(abs(phi1), 1e-4)), 1e-3)
            return (
                lambda om: s2_1 * (2.0 * al) / (al ** 2 + om ** 2),
                lambda om: (1.0 - phi1 ** 2) / (1.0 + phi1 ** 2 - 2.0 * phi1 * np.cos(om)),
            )
        elif best_model == "M2_osc":
            zt, w0, ng = pars2["zeta"], pars2["omega_0_rad_per_step"], pars2["nugget"]
            d_osc = _norm_on_pi(lambda om: car2_two_sided_unit_psd(om, zt, w0))
            return (
                lambda om: s2_2 * ((1.0 - ng) * car2_two_sided_unit_psd(om, zt, w0) + ng * car2_two_sided_unit_psd(om, 3.0, 0.5)),
                lambda om: (1.0 - ng) * d_osc(om) + ng,
            )
        elif best_model == "M3_seas_ar1":
            amp2 = float(beta3[-2] ** 2 + beta3[-1] ** 2)
            vs = 0.5 * amp2
            al = max(-math.log(max(abs(phi3), 1e-4)), 1e-3)
            d_seas = _norm_on_pi(lambda om: car2_two_sided_unit_psd(om, 0.03, om_yr))
            d_ar1 = _norm_on_pi(lambda om: (1.0 - phi3 ** 2) / (1.0 + phi3 ** 2 - 2.0 * phi3 * np.cos(om)))
            return (
                lambda om: vs * car2_two_sided_unit_psd(om, 0.03, om_yr) + s2_3 * (2.0 * al) / (al ** 2 + om ** 2),
                lambda om: vs * d_seas(om) + s2_3 * d_ar1(om),
            )
        else:
            amp2 = float(beta4[-2] ** 2 + beta4[-1] ** 2)
            vs = 0.5 * amp2
            zt, w0, ng = pars4["zeta"], pars4["omega_0_rad_per_step"], pars4["nugget"]
            d_seas = _norm_on_pi(lambda om: car2_two_sided_unit_psd(om, 0.03, om_yr))
            d_osc = _norm_on_pi(lambda om: car2_two_sided_unit_psd(om, zt, w0))
            return (
                lambda om: vs * car2_two_sided_unit_psd(om, 0.03, om_yr) + s2_4 * ((1.0 - ng) * car2_two_sided_unit_psd(om, zt, w0) + ng * car2_two_sided_unit_psd(om, 3.0, 0.5)),
                lambda om: vs * d_seas(om) + s2_4 * ((1.0 - ng) * d_osc(om) + ng),
            )

    psd_ct_fn, psd_disc_fn = make_best_psd_fns()

    # Spectral power shares on (0, pi) from Lomb-Scargle periodogram of within-site centred residuals
    z_centred_list = [z - np.mean(z) for _, _, z in panel]
    z_cat = np.concatenate(z_centred_list)
    var_pooled = float(np.sum(z_cat ** 2) / max(n_tot - g_sites, 1))
    sd_pooled = math.sqrt(max(var_pooled, 0.0))

    om_grid = np.linspace(0.02, math.pi, 512)
    ls_pgram = np.zeros_like(om_grid)
    for _, tg, zg in panel:
        zc = zg - np.mean(zg)
        for i_o, w in enumerate(om_grid):
            tau_ls = 0.5 * math.atan2(float(np.sum(np.sin(2.0 * w * tg))), float(np.sum(np.cos(2.0 * w * tg))) + 1e-15) / w
            ct = np.cos(w * (tg - tau_ls))
            st = np.sin(w * (tg - tau_ls))
            p_val = 0.5 * (
                (float(np.sum(zc * ct)) ** 2) / max(float(np.sum(ct ** 2)), 1e-12)
                + (float(np.sum(zc * st)) ** 2) / max(float(np.sum(st ** 2)), 1e-12)
            )
            ls_pgram[i_o] += p_val
    tot_pow = float(np.trapezoid(ls_pgram, om_grid))
    m_multi = om_grid < 0.28                                  # periods > 22.4 steps (> 1.7 yr)
    m_ann = (om_grid >= 0.34) & (om_grid <= 0.64)             # periods ~ 9.8..18.5 steps (~1 yr)
    m_high = om_grid > 0.64
    frac_multi = float(np.trapezoid(ls_pgram[m_multi], om_grid[m_multi]) / max(tot_pow, 1e-15))
    frac_ann = float(np.trapezoid(ls_pgram[m_ann], om_grid[m_ann]) / max(tot_pow, 1e-15))
    frac_high = float(np.trapezoid(ls_pgram[m_high], om_grid[m_high]) / max(tot_pow, 1e-15))

    # Empirical ACF across integer lags 0..12
    emp_acf: list[float | None] = [1.0]
    emp_pairs: list[int] = [int(n_tot)]
    for lag in range(1, 13):
        xs, ys = [], []
        for _, tg, zg in panel:
            zc = zg - np.mean(zg)
            t_map = {int(round(float(tt))): val for tt, val in zip(tg, zc)}
            for k_step, val0 in t_map.items():
                if (k_step + lag) in t_map:
                    xs.append(val0)
                    ys.append(t_map[k_step + lag])
        emp_pairs.append(len(xs))
        if len(xs) >= 6:
            xa, ya = np.array(xs), np.array(ys)
            # Normalize by pooled within-site variance (standard stationary autocovariance / var) or Pearson:
            # Use Pearson on pairs (bounded in [-1, 1]) and also record covariance/var_pooled if needed
            emp_acf.append(float(np.corrcoef(xa, ya)[0, 1]))
        else:
            emp_acf.append(None)

    return {
        "n_sites": g_sites,
        "n_obs": int(n_tot),
        "var_ln_H": round(var_pooled, 5),
        "sd_ln_H": round(sd_pooled, 5),
        "acf_0_to_12": [None if v is None else round(v, 4) for v in emp_acf],
        "acf_pair_counts_0_to_12": emp_pairs,
        "lomb_scargle_power_shares": {
            "multiannual_lt_0.28rad": round(frac_multi, 4),
            "annual_0.34_to_0.64rad": round(frac_ann, 4),
            "subannual_gt_0.64rad": round(frac_high, 4),
        },
        "models": {
            "M0_iid": {"k": k0, "nll": round(nll0, 3), "aicc": round(aicc_map["M0_iid"], 3), "sigma2": round(s2_0, 5)},
            "M1_ar1": {"k": k1, "nll": round(nll1, 3), "aicc": round(aicc_map["M1_ar1"], 3), "phi": round(phi1, 4), "sigma2": round(s2_1, 5)},
            "M2_osc": {"k": k2, "nll": round(nll2, 3), "aicc": round(aicc_map["M2_osc"], 3), "sigma2": round(s2_2, 5), **{k: round(v, 4) if math.isfinite(v) else None for k, v in pars2.items()}},
            "M3_seas_ar1": {
                "k": k3,
                "nll": round(nll3, 3),
                "aicc": round(aicc_map["M3_seas_ar1"], 3),
                "phi": round(phi3, 4),
                "seas_amp": round(math.hypot(float(beta3[-2]), float(beta3[-1])), 4),
                "sigma2_resid": round(s2_3, 5),
            },
            "M4_seas_osc": {
                "k": k4,
                "nll": round(nll4, 3),
                "aicc": round(aicc_map["M4_seas_osc"], 3),
                "seas_amp": round(math.hypot(float(beta4[-2]), float(beta4[-1])), 4),
                "sigma2_resid": round(s2_4, 5),
                **{k: round(v, 4) if math.isfinite(v) else None for k, v in pars4.items()},
            },
        },
        "companion_discrete_ar2": {
            "M2_disc_ar2": {"k": k2_d, "nll": round(nll2_d, 3), "aicc": round(aicc(nll2_d, k2_d), 3), "sigma2": round(s2_2d, 5), **pars2_d},
            "M4_seas_disc_ar2": {
                "k": k4_d,
                "nll": round(nll4_d, 3),
                "aicc": round(aicc(nll4_d, k4_d), 3),
                "seas_amp": round(math.hypot(float(beta4_d[-2]), float(beta4_d[-1])), 4),
                "sigma2_resid": round(s2_4d, 5),
                **pars4_d,
            },
        },
        "best_model": best_model,
        "seasonal_variance_share_best": round(seas_share, 4),
        "delta_aicc_beyond_ar1": round(delta_beyond_ar1, 3),
        "verdict_T1_mem": verdict_t1,
        "ar1_acf_0_to_12": [round(v, 4) for v in ar1_acf],
        "best_model_acf_0_to_12": [round(v, 4) for v in best_acf],
        "_psd_ct_fn": psd_ct_fn,
        "_psd_disc_fn": psd_disc_fn,
        "_ls_om_grid": om_grid,
        "_ls_pgram": ls_pgram,
        "_phi_ar1": phi1,
    }


# ---------------------------------------------------------------------------
# 2. Harmed-Host Block V Extension for K9 (`notes/coupled-hosts.md` §1)
# ---------------------------------------------------------------------------

def k9_block_v_transfer_function(
    omega: np.ndarray,
    a_H: float,
    p_star: float,
    eta_H: float,
    alpha_harm: float,
    rho_reg: float,
    a_V: float | None = None,
    eta_VH: float = 1.0,
) -> dict[str, np.ndarray]:
    """Frequency response from external host/contact forcing z_H(i*omega) to y_p
    when the host is a dynamic Block V harmed at rate alpha_harm <= a_H with
    host net-production slope -rho_reg (`notes/coupled-hosts.md` §1):
      E_V_block(s) = q*(s + rho) / [(s + rho)*(s + d) + d*h],
      where q = a_H, d = a_H / (1 - p_star), h = alpha_harm * p_star.
    As alpha_harm -> 0 (h -> 0), E_V_block(s) reduces identically to a_H / (s + d) (Block S).
    """
    if alpha_harm > a_H + 1e-12:
        raise ValueError("Standing condition violated: Block V requires a_H >= alpha_harm")
    s = 1j * np.asarray(omega, dtype=float)
    q = a_H
    d = a_H / (1.0 - p_star)
    h = alpha_harm * p_star
    E_host = q * (s + rho_reg) / ((s + rho_reg) * (s + d) + d * h)

    if a_V is None:
        H_I = E_host * ((eta_H - 1.0) + 1.0 / (1.0 - p_star)) / (1.0 - E_host)
    else:
        eta_HV = eta_H + 1.0 - eta_VH
        E_vec = a_V / (s + a_V)
        H_I = E_host * (E_vec * (eta_VH - 1.0) + (eta_HV - 1.0) + 1.0 / (1.0 - p_star)) / (1.0 - E_host * E_vec)
    return {"H_I": H_I, "H_p": H_I - 1.0}


# ---------------------------------------------------------------------------
# 3. Cross-Correlation, Weather-Partial CCF, Phase Surrogates & Block Bootstrap
# ---------------------------------------------------------------------------

def compute_pooled_ccf(
    site_grids: list[dict[str, Any]],
    max_lag: int = 12,
    min_pairs: int = 6,
    cov_keys: list[str] | None = None,
) -> tuple[list[float | None], list[int]]:
    """Compute pooled within-site CCF(lag) = Corr(y_{s, t+lag}, z_{s, t}) for lag = 0..max_lag.

    Each item in `site_grids` has integer `max_k`, arrays `z` (centred within site),
    `y` (centred within site), and optional covariate dict `covs[name]` (centred within site).
    If `cov_keys` is provided, linearly partial out `[cov(t+lag), cov(t)]` across valid pairs!
    """
    ccfs: list[float | None] = []
    npairs: list[int] = []
    for lag in range(max_lag + 1):
        zs, ys, Xs = [], [], []
        for sg in site_grids:
            z_arr = sg["z"]
            y_arr = sg["y"]
            mk = len(z_arr)
            for t in range(mk - lag):
                zv, yv = z_arr[t], y_arr[t + lag]
                if math.isnan(zv) or math.isnan(yv):
                    continue
                if cov_keys:
                    row_x = []
                    ok = True
                    for ck in cov_keys:
                        c_t = sg["covs"][ck][t]
                        c_tl = sg["covs"][ck][t + lag]
                        if math.isnan(c_t) or math.isnan(c_tl):
                            ok = False
                            break
                        row_x.extend([c_t, c_tl])
                    if not ok:
                        continue
                    Xs.append(row_x)
                zs.append(zv)
                ys.append(yv)
        n_p = len(zs)
        npairs.append(n_p)
        if n_p < min_pairs:
            ccfs.append(None)
            continue
        za = np.array(zs, dtype=float)
        ya = np.array(ys, dtype=float)
        if cov_keys and len(Xs) == n_p and n_p > len(Xs[0]) + 3:
            Xmat = np.column_stack([np.ones(n_p), np.array(Xs, dtype=float)])
            bz = np.linalg.lstsq(Xmat, za, rcond=None)[0]
            by = np.linalg.lstsq(Xmat, ya, rcond=None)[0]
            rz = za - Xmat @ bz
            ry = ya - Xmat @ by
            denom = math.sqrt(float(np.sum(rz ** 2) * np.sum(ry ** 2)))
            ccfs.append(float(np.sum(rz * ry) / denom) if denom > 1e-15 else 0.0)
        else:
            if np.std(za) < 1e-12 or np.std(ya) < 1e-12:
                ccfs.append(0.0)
            else:
                ccfs.append(float(np.corrcoef(za, ya)[0, 1]))
    return ccfs, npairs


def ccf_summary_metrics(ccfs: list[float | None]) -> dict[str, float | int]:
    arr = np.array([np.nan if v is None else v for v in ccfs], dtype=float)
    sub16 = arr[1:7]
    sub08 = arr[0:9]
    s_pos = float(np.nanmax(sub16)) if not np.all(np.isnan(sub16)) else float("nan")
    c0 = float(arr[0]) if not math.isnan(arr[0]) else float("nan")
    s_delay = (s_pos - c0) if (not math.isnan(s_pos) and not math.isnan(c0)) else float("nan")
    pk_lag = int(np.nanargmax(sub08)) if not np.all(np.isnan(sub08)) else -1
    pos_w = np.maximum(np.nan_to_num(sub08, nan=0.0), 0.0)
    l_com = float(np.sum(np.arange(9) * pos_w) / np.sum(pos_w)) if np.sum(pos_w) > 1e-9 else 0.0
    all_neg_16 = bool(np.all(sub16[~np.isnan(sub16)] < 0.0)) if np.any(~np.isnan(sub16)) else False
    return {
        "S_pos_max_1_6": s_pos,
        "CCF_lag0": c0,
        "S_delay_max16_minus_lag0": s_delay,
        "peak_lag_0_to_8": pk_lag,
        "peak_val_0_to_8": float(sub08[pk_lag]) if pk_lag >= 0 else float("nan"),
        "L_com_0_to_8": l_com,
        "all_lags_1_6_negative": all_neg_16,
    }


def phase_surrogate_ccf_test(
    site_grids: list[dict[str, Any]],
    pred_iqr_extended: tuple[int, int],
    expect_neg_lag0_and_late_peak: bool,
    n_surr: int = 2000,
    seed: int = 20261114,
    cov_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Phase-randomised + random-circular-shift surrogate test preserving each site's
    exact missing-value pattern and host autocorrelation spectrum (`protocol.md` §4.4).
    """
    obs_ccf, obs_pairs = compute_pooled_ccf(site_grids, max_lag=12, min_pairs=6, cov_keys=cov_keys)
    obs_m = ccf_summary_metrics(obs_ccf)

    rng = np.random.default_rng(seed)
    # Pre-build filled FFT templates per site (linearly interpolating internal gaps before FFT phase
    # randomisation, then re-applying the exact observed NaN mask and re-centring!)
    templates = []
    for sg in site_grids:
        z = sg["z"].copy()
        valid = np.flatnonzero(~np.isnan(z))
        L = len(z)
        if len(valid) >= 4:
            grid_i = np.arange(L)
            z_fill = np.interp(grid_i, valid, z[valid])
            z_fill -= np.mean(z_fill)
            fft_z = np.fft.rfft(z_fill)
        else:
            fft_z = None
        templates.append((L, valid, fft_z, sg))

    null_s_pos = np.empty(n_surr)
    null_s_delay = np.empty(n_surr)
    null_min16 = np.empty(n_surr)
    null_c0 = np.empty(n_surr)

    for b in range(n_surr):
        surr_grids = []
        for L, valid, fft_z, sg in templates:
            z_s = np.full(L, np.nan)
            if fft_z is not None:
                phases = rng.uniform(0.0, 2.0 * math.pi, size=len(fft_z))
                phases[0] = 0.0
                if L % 2 == 0 and len(fft_z) > 1:
                    phases[-1] = 0.0
                synth = np.fft.irfft(np.abs(fft_z) * np.exp(1j * phases), n=L)
                shift = int(rng.integers(0, L))
                synth = np.roll(synth, shift)
                vals = synth[valid]
                vals -= np.mean(vals)
                s_orig = float(np.std(sg["z"][valid], ddof=1))
                s_syn = float(np.std(vals, ddof=1))
                if s_syn > 1e-12:
                    vals *= s_orig / s_syn
                z_s[valid] = vals
            surr_grids.append({"z": z_s, "y": sg["y"], "covs": sg.get("covs", {})})
        c_b, _ = compute_pooled_ccf(surr_grids, max_lag=6, min_pairs=6, cov_keys=cov_keys)
        arr_b = np.array([np.nan if v is None else v for v in c_b], dtype=float)
        sub = arr_b[1:7]
        sp = float(np.nanmax(sub)) if not np.all(np.isnan(sub)) else 0.0
        mn = float(np.nanmin(sub)) if not np.all(np.isnan(sub)) else 0.0
        c0 = float(arr_b[0]) if not math.isnan(arr_b[0]) else 0.0
        null_s_pos[b] = sp
        null_s_delay[b] = sp - c0
        null_min16[b] = mn
        null_c0[b] = c0

    p_pos = float((1 + np.sum(null_s_pos >= obs_m["S_pos_max_1_6"])) / (n_surr + 1))
    p_delay = float((1 + np.sum(null_s_delay >= obs_m["S_delay_max16_minus_lag0"])) / (n_surr + 1))
    crit_95 = float(np.quantile(null_s_pos, 0.95))
    crit_delay_95 = float(np.quantile(null_s_delay, 0.95))

    # Exact registered T2_track decision criterion (`protocol.md` §4.4):
    lo_iqr, hi_iqr = pred_iqr_extended
    pk = int(obs_m["peak_lag_0_to_8"])
    p_all_neg = float((1 + np.sum(null_s_pos <= obs_m["S_pos_max_1_6"])) / (n_surr + 1))
    p_c0_pos = float((1 + np.sum(null_c0 >= obs_m["CCF_lag0"])) / (n_surr + 1))

    if obs_m["S_pos_max_1_6"] > 0 and p_pos < 0.05 and (lo_iqr <= pk <= hi_iqr):
        verdict = "supports"
    elif (obs_m["all_lags_1_6_negative"] and p_all_neg < 0.025) or (
        expect_neg_lag0_and_late_peak and pk == 0 and obs_m["CCF_lag0"] > 0 and p_c0_pos < 0.05
    ):
        verdict = "contradicts"
    else:
        verdict = "not_decided"

    return {
        "ccf_0_to_12": [None if v is None else round(v, 4) for v in obs_ccf],
        "pair_counts_0_to_12": obs_pairs,
        "S_pos_max_1_6": round(float(obs_m["S_pos_max_1_6"]), 4),
        "CCF_lag0": round(float(obs_m["CCF_lag0"]), 4),
        "S_delay": round(float(obs_m["S_delay_max16_minus_lag0"]), 4),
        "peak_lag_0_to_8": pk,
        "L_com_0_to_8": round(float(obs_m["L_com_0_to_8"]), 4),
        "p_surrogate_S_pos": round(p_pos, 4),
        "p_surrogate_S_delay": round(p_delay, 4),
        "null_95pct_S_pos": round(crit_95, 4),
        "null_95pct_S_delay": round(crit_delay_95, 4),
        "allowed_peak_lag_window": [lo_iqr, hi_iqr],
        "verdict_T2_track": verdict,
    }


def block_bootstrap_lag_order(
    shared_sites: list[dict[str, Any]],
    block_len: int = 3,
    n_boot: int = 2000,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Within-site moving-block bootstrap of 28-day windows comparing L_com(slow) - L_com(fast)
    on shared host populations (`Test T3_order`, `protocol.md` §4.4).
    Each item in `shared_sites` has `z`, `y_fast`, `y_slow` of length `L`.
    """
    g_fast = [{"z": s["z"], "y": s["y_fast"]} for s in shared_sites]
    g_slow = [{"z": s["z"], "y": s["y_slow"]} for s in shared_sites]
    ccf_f, _ = compute_pooled_ccf(g_fast, max_lag=8, min_pairs=6)
    ccf_s, _ = compute_pooled_ccf(g_slow, max_lag=8, min_pairs=6)
    mf = ccf_summary_metrics(ccf_f + [None] * 4)
    ms = ccf_summary_metrics(ccf_s + [None] * 4)
    obs_diff = float(ms["L_com_0_to_8"] - mf["L_com_0_to_8"])

    # Resample pairs `(t, t+lag)` in contiguous starting-time blocks of length `block_len`
    # within each site so lag structure `(y(t+lag), z(t))` is preserved!
    rng = np.random.default_rng(seed + 77)
    diffs = np.empty(n_boot)
    l_fast_b = np.empty(n_boot)
    l_slow_b = np.empty(n_boot)

    for b in range(n_boot):
        cf_b = np.full(9, np.nan)
        cs_b = np.full(9, np.nan)
        # Draw block start indices for each site once per bootstrap replicate
        site_starts = []
        for s in shared_sites:
            L = len(s["z"])
            n_blk = max(1, math.ceil(L / block_len))
            max_st = max(1, L - block_len + 1)
            starts = rng.integers(0, max_st, size=n_blk)
            t_indices = np.concatenate([np.arange(st, min(st + block_len, L)) for st in starts])[:L]
            site_starts.append(t_indices)

        for lag in range(9):
            zf_l, yf_l, zs_l, ys_l = [], [], [], []
            for s, t_idx in zip(shared_sites, site_starts):
                L = len(s["z"])
                for t0 in t_idx:
                    if t0 + lag < L:
                        zv = s["z"][t0]
                        if not math.isnan(zv):
                            yfv = s["y_fast"][t0 + lag]
                            ysv = s["y_slow"][t0 + lag]
                            if not math.isnan(yfv):
                                zf_l.append(zv)
                                yf_l.append(yfv)
                            if not math.isnan(ysv):
                                zs_l.append(zv)
                                ys_l.append(ysv)
            if len(zf_l) >= 6 and np.std(zf_l) > 1e-12 and np.std(yf_l) > 1e-12:
                cf_b[lag] = float(np.corrcoef(zf_l, yf_l)[0, 1])
            if len(zs_l) >= 6 and np.std(zs_l) > 1e-12 and np.std(ys_l) > 1e-12:
                cs_b[lag] = float(np.corrcoef(zs_l, ys_l)[0, 1])

        wf = np.maximum(np.nan_to_num(cf_b, nan=0.0), 0.0)
        ws = np.maximum(np.nan_to_num(cs_b, nan=0.0), 0.0)
        lf = float(np.sum(np.arange(9) * wf) / np.sum(wf)) if np.sum(wf) > 1e-9 else 0.0
        ls = float(np.sum(np.arange(9) * ws) / np.sum(ws)) if np.sum(ws) > 1e-9 else 0.0
        l_fast_b[b] = lf
        l_slow_b[b] = ls
        diffs[b] = ls - lf

    ci_lo, ci_hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    p_one_pos = float((1 + np.sum(diffs <= 0.0)) / (n_boot + 1))
    p_one_neg = float((1 + np.sum(diffs >= 0.0)) / (n_boot + 1))
    if obs_diff > 0 and ci_lo > 0.0 and p_one_pos < 0.05:
        verdict = "supports"
    elif obs_diff < 0 and ci_hi < 0.0 and p_one_neg < 0.05:
        verdict = "contradicts"
    else:
        verdict = "not_decided"

    return {
        "L_com_fast": round(float(mf["L_com_0_to_8"]), 4),
        "L_com_slow": round(float(ms["L_com_0_to_8"]), 4),
        "peak_lag_fast": int(mf["peak_lag_0_to_8"]),
        "peak_lag_slow": int(ms["peak_lag_0_to_8"]),
        "diff_slow_minus_fast": round(obs_diff, 4),
        "boot_mean_diff": round(float(np.mean(diffs)), 4),
        "boot_se_diff": round(float(np.std(diffs, ddof=1)), 4),
        "boot_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "p_one_sided_slow_gt_fast": round(p_one_pos, 4),
        "verdict_T3_order": verdict,
    }


# ---------------------------------------------------------------------------
# 4. `lifecycle_engine.py` Mechanistic Sensitivity Runner
# ---------------------------------------------------------------------------

def run_engine_variant(
    host_stay_steps: float,
    vec_stay_steps: float,
    eta_H: float,
    sigma_H: float,
    phi_step: float,
    dt_sub: float = 0.25,
    vec_shape: str = "memoryless",
    vec_substages: int = 4,
    contact_wait: bool = False,
    wait_death_frac: float = 0.50,
    n_patches: int = 1,
    patch_move: float = 0.10,
    patch_sync: float = 0.20,
    alt_steady_share: float = 0.0,
    driver_kind: str = "lognormal",
    driver_low: float | None = None,
    steps_sim: int = 24000,
    pairs_sim: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build a discrete life cycle on sub-step `dt_sub` (default 1/4 of 28d = 7 days)
    and evaluate `Engine.coefficient()` and `Engine.loss()`.
    """
    m_h = host_stay_steps / dt_sub
    m_v = vec_stay_steps / dt_sub
    phi_sub = (abs(phi_step) ** dt_sub) * (1.0 if phi_step >= 0 else -1.0)
    move_sub = 1.0 - ((1.0 - patch_move) ** dt_sub) if n_patches > 1 else 0.0

    # If contact_wait is True, vector/free stage waits for focal host contact
    vec_death = (wait_death_frac / m_v) if contact_wait else 0.0
    if vec_shape == "fixed":
        m_v_use = float(max(1, int(round(m_v))))
    else:
        m_v_use = m_v

    stages = [
        Stage("host", mean=m_h, shape="memoryless"),
        Stage(
            "vector",
            mean=m_v_use,
            shape=vec_shape,
            substages=vec_substages if vec_shape == "chain" else 1,
            death=vec_death,
            contact=contact_wait,
        ),
    ]
    use_low = driver_low
    if driver_kind == "bounded" and use_low is None and np.expm1(sigma_H ** 2) >= 0.90:
        use_low = 0.15
    drv = Driver("vole", sigma=sigma_H, phi=phi_sub, kind=driver_kind, low=use_low)

    if alt_steady_share > 0.0:
        transfers = [
            Transfer("host", "vector", share=1.0, yield_=2.0, birth=True),
            Transfer("vector", "host", share=1.0 - alt_steady_share, yield_=1.5, driver="vole", elasticity=eta_H),
            Transfer("vector", "host", share=alt_steady_share, yield_=1.5, driver=None, elasticity=1.0),
        ]
    else:
        transfers = [
            Transfer("host", "vector", share=1.0, yield_=2.0, birth=True),
            Transfer("vector", "host", share=1.0, yield_=1.5, driver="vole", elasticity=eta_H),
        ]

    patches = (
        Patches(count=n_patches, synchrony=patch_sync, mobile_stage="host", movement=move_sub, topology="island")
        if n_patches > 1
        else Patches()
    )
    desc = Description(stages=stages, transfers=transfers, drivers=[drv], patches=patches)
    eng = Engine(desc)
    coeff = eng.coefficient()
    loss = eng.loss(steps=steps_sim, burn=1200, pairs=pairs_sim, seed=seed)
    return {
        "D2": round(float(coeff["D2"]), 5) if coeff["D2"] is not None else None,
        "J2_per_birth": round(float(coeff["J2"] / coeff["b"]), 5),
        "D_sim": round(float(loss["D"]), 5),
        "D_se": round(float(loss["se"]), 5),
        "J_bound": round(float(loss["J"]), 5),
        "J_transfer": round(float(loss["J_transfer"]), 5),
        "bound_holds": bool(loss["D"] <= loss["J"] + 1e-9),
    }


def run_engine_trough_extinction(
    host_stay_steps: float,
    vec_stay_steps: float,
    eta_H: float,
    sigma_H: float,
    phi_step: float,
    horizon_steps_28d: int = 36,
    margin_ln_r0: float = 0.05,
    n0_total: int = 20,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate exact conditional PGF extinction probability (`Engine.extinction`)
    near threshold (`y = D + margin_ln_r0` vs constant host `y = margin_ln_r0`).
    """
    dt_sub = 0.25
    m_h = host_stay_steps / dt_sub
    m_v = vec_stay_steps / dt_sub
    phi_sub = abs(phi_step) ** dt_sub
    desc = Description(
        stages=[Stage("host", mean=m_h), Stage("vector", mean=m_v)],
        transfers=[
            Transfer("host", "vector", yield_=2.0, birth=True),
            Transfer("vector", "host", yield_=1.5, driver="vole", elasticity=eta_H),
        ],
        drivers=[Driver("vole", sigma=sigma_H, phi=phi_sub, kind="lognormal")],
    )
    eng = Engine(desc).critical()
    d_est = eng.loss(steps=20000, burn=1000, pairs=16, seed=seed)["D"]
    init = np.array([n0_total // 2, n0_total - n0_total // 2], dtype=np.int64)
    tot_sub_steps = int(round(horizon_steps_28d / dt_sub))

    # Compare at the SAME mean-host R0 (`y = margin_ln_r0`): constant host vs fluctuating host!
    ext_const_pois = eng.extinction(init, steps=tot_sub_steps, trials=400, seed=seed, y=margin_ln_r0, offspring="poisson", constant=True)
    ext_fluct_pois = eng.extinction(init, steps=tot_sub_steps, trials=400, seed=seed, y=margin_ln_r0, offspring="poisson", constant=False)
    ext_fluct_nb = eng.extinction(init, steps=tot_sub_steps, trials=400, seed=seed, y=margin_ln_r0, offspring="negative_binomial", dispersion=0.5, constant=False)
    # Also at matched stochastic growth margin (`y = d_est + margin_ln_r0`):
    ext_matched_pois = eng.extinction(init, steps=tot_sub_steps, trials=400, seed=seed, y=d_est + margin_ln_r0, offspring="poisson", constant=False)

    return {
        "N0_cohort": n0_total,
        "horizon_28d_steps": horizon_steps_28d,
        "margin_ln_R0": margin_ln_r0,
        "D_loss": round(float(d_est), 5),
        "ext_prob_constant_poisson": round(float(ext_const_pois["conditional_mean"]), 4),
        "ext_prob_fluct_same_R0_poisson": round(float(ext_fluct_pois["conditional_mean"]), 4),
        "ext_prob_fluct_same_R0_negbin05": round(float(ext_fluct_nb["conditional_mean"]), 4),
        "ext_prob_fluct_matched_margin_poisson": round(float(ext_matched_pois["conditional_mean"]), 4),
    }


# ---------------------------------------------------------------------------
# 5. Empirical Weather -> Thermal Stage Duration & Trajectory Loss (`W2_thermal`)
# ---------------------------------------------------------------------------

def evaluate_thermal_trajectory_loss(
    T_series_C: np.ndarray,
    H_mult_series: np.ndarray,
    tau_host_steps: float,
    nu_ref_per_step: float,
    mu_ref_per_step: float,
    T_ref_C: float,
    Q10_dev: float,
    Q10_mort: float,
    m_buf: float,
    eta_H: float,
    cold_floor_C: float | None = None,
    dt_sub: float = 0.25,
) -> dict[str, Any]:
    """Evaluate exact trajectory growth root R0*, curvature shift J_temp, pure thermal
    loss D_temp, pure host loss D_host, joint loss D_joint(H, T), and Proposition B's
    staying-inclusive upper bound (`notes/mechanism-temperature-incubation.md` §§1, 2.2)
    along observed 28-day temperature `T_series_C` and mean-1 host multipliers `H_mult_series`.
    """
    T_arr = np.asarray(T_series_C, dtype=float)
    H_arr = np.asarray(H_mult_series, dtype=float)
    assert len(T_arr) == len(H_arr) and len(T_arr) >= 8

    # Normalize H^eta to arithmetic mean 1 (Baseline ii for host multiplier)
    h_raw = H_arr ** eta_H
    h_rate = h_raw / float(np.mean(h_raw))

    T_bar = float(np.mean(T_arr))
    T_eff = T_bar + m_buf * (T_arr - T_bar)

    def stage_rates(temp_c: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tc = np.asarray(temp_c, dtype=float)
        if cold_floor_C is not None:
            tc_dev = np.maximum(tc, cold_floor_C)
        else:
            tc_dev = tc
        nu = nu_ref_per_step * (Q10_dev ** ((tc_dev - T_ref_C) / 10.0))
        if cold_floor_C is not None:
            # Continuous steeper cold-diapause suppression below cold_floor_C (factor = 1 at tc == cold_floor_C)
            nu = np.where(tc < cold_floor_C, nu * (6.0 ** ((tc - cold_floor_C) / 10.0)), nu)
        mu = mu_ref_per_step * (Q10_mort ** ((tc - T_ref_C) / 10.0))
        # Exact sub-step probabilities over dt_sub:
        q_sub = 1.0 - np.exp(-(nu + mu) * dt_sub)
        p_comp = nu / np.maximum(nu + mu, 1e-15)
        s_sub = 1.0 - q_sub
        g_sub = p_comp * q_sub
        return s_sub, g_sub, nu, mu

    # Sheltered intra-host stage: constant geometric stay on sub-step dt_sub
    q_h = dt_sub / tau_host_steps
    s_h = 1.0 - q_h

    s_v_t, g_v_t, nu_t, mu_t = stage_rates(T_eff)
    # Implied 28-day mean stage stay tau_V(T_k) = 1 / (nu_t + mu_t)
    tau_v_28d = 1.0 / np.maximum(nu_t + mu_t, 1e-12)

    # Constant T_bar baseline (i)
    s_v_bar_arr, g_v_bar_arr, _, _ = stage_rates(T_bar)
    s_v_i = float(s_v_bar_arr)
    g_v_i = float(g_v_bar_arr)
    R0_i = g_v_i / (1.0 - s_v_i)  # choosing base fecundity b_0 = 1/(1-g_v_i...) -> normalize at R0_ii = 1!

    # Arithmetic-mean matrix baseline (ii)
    s_v_ii = float(np.mean(s_v_t))
    g_v_ii = float(np.mean(g_v_t))
    R0_ii_raw = g_v_ii / (1.0 - s_v_ii)
    J_temp = math.log(R0_ii_raw / R0_i)

    # Calibrate birth yield `b_cal` on host->vector edge so arithmetic-mean R0_ii == 1.0!
    b_cal = (1.0 - s_v_ii) / g_v_ii

    n_sub_per_step = int(round(1.0 / dt_sub))

    def lyapunov_growth(y_tilt: float, use_H: bool, use_T: bool, n_cycles: int = 60) -> float:
        x = np.array([0.5, 0.5], dtype=float)
        acc = 0.0
        tot_steps = 0
        eb = math.exp(y_tilt) * b_cal
        for cyc in range(n_cycles):
            for k in range(len(T_arr)):
                sv = float(s_v_t[k]) if use_T else s_v_ii
                gv = float(g_v_t[k]) if use_T else g_v_ii
                hm = float(h_rate[k]) if use_H else 1.0
                # Matrix: col 0 = host, col 1 = vector
                # x_host'   = s_h * x_host + gv * hm * x_vec
                # x_vec'    = eb * q_h * x_host + sv * x_vec
                a00, a01 = s_h, gv * hm
                a10, a11 = eb * q_h, sv
                for _ in range(n_sub_per_step):
                    nx0 = a00 * x[0] + a01 * x[1]
                    nx1 = a10 * x[0] + a11 * x[1]
                    nrm = nx0 + nx1
                    x[0] = nx0 / nrm
                    x[1] = nx1 / nrm
                    if cyc >= 5:
                        acc += math.log(nrm)
                        tot_steps += 1
        return acc / max(tot_steps, 1)

    def solve_D(use_H: bool, use_T: bool) -> float:
        return float(optimize.brentq(lambda y: lyapunov_growth(y, use_H, use_T), -3.5, 3.5, xtol=1e-9))

    D_host = solve_D(True, False)
    D_temp = solve_D(False, True)
    D_joint = solve_D(True, True)

    # Proposition B bound (`notes/mechanism-temperature-incubation.md` §2.2):
    # On joint (H, T), transfer g_vec->host(k) = g_v_t[k] * h_rate[k], mean g_bar_joint = mean(g_v_t * h_rate)
    # Wait: note that under correlated (H, T), E[g_v(T) * h(H)] can differ from E[g_v(T)] * E[h(H)]!
    # Relative to the joint mean matrix `A_bar_joint` (where g_bar_joint = E[g_v * h]):
    g_joint_series = g_v_t * h_rate
    g_bar_joint = float(np.mean(g_joint_series))
    ln_R0_mean_joint_vs_ii = math.log(g_bar_joint / g_v_ii)  # Cov(g_v, h) mean-rate shift!
    D_vs_joint_mean = D_joint + ln_R0_mean_joint_vs_ii

    ell_g_joint = -float(np.mean(np.log(np.maximum(g_joint_series / g_bar_joint, 1e-15))))
    ell_s_temp = -float(np.mean(np.log(np.maximum(s_v_t / s_v_ii, 1e-15))))
    tau_v_sub_0 = 1.0 / (1.0 - s_v_ii)
    prop_B_bound_joint = ell_g_joint + (tau_v_sub_0 - 1.0) * ell_s_temp

    ell_g_temp = -float(np.mean(np.log(np.maximum(g_v_t / g_v_ii, 1e-15))))
    prop_B_bound_temp = ell_g_temp + (tau_v_sub_0 - 1.0) * ell_s_temp

    return {
        "T_mean_C": round(T_bar, 2),
        "T_min_28d_C": round(float(np.min(T_arr)), 2),
        "T_max_28d_C": round(float(np.max(T_arr)), 2),
        "tau_V_mean_steps": round(float(np.mean(tau_v_28d)), 3),
        "tau_V_min_steps": round(float(np.min(tau_v_28d)), 3),
        "tau_V_max_steps": round(float(np.max(tau_v_28d)), 3),
        "tau_V_cv": round(float(np.std(tau_v_28d, ddof=1) / np.mean(tau_v_28d)), 4),
        "J_temp_curvature": round(J_temp, 5),
        "D_host_only": round(D_host, 5),
        "D_temp_only": round(D_temp, 5),
        "D_joint_HT": round(D_joint, 5),
        "cov_HT_mean_shift_lnR0": round(ln_R0_mean_joint_vs_ii, 5),
        "D_vs_joint_mean_matrix": round(D_vs_joint_mean, 5),
        "prop_B_bound_temp_only": round(prop_B_bound_temp, 5),
        "prop_B_bound_joint": round(prop_B_bound_joint, 5),
        "prop_B_holds_temp": bool(D_temp <= prop_B_bound_temp + 1e-7),
        "prop_B_holds_joint": bool(D_vs_joint_mean <= prop_B_bound_joint + 1e-7),
    }


def gp_posterior_mean_on_grid(
    t_obs: np.ndarray,
    z_obs: np.ndarray,
    t_grid: np.ndarray,
    fit_info: dict[str, Any],
    period_steps: float = PERIOD_YR_STEPS,
) -> np.ndarray:
    """Exact Gaussian-process conditional expectation E[z(t_grid) | z(t_obs)]
    under the best-fitting host memory model from `fit_host_memory_panel` (used
    only to supply winter host multipliers alongside contiguous 28-day weather in
    the W2_thermal joint trajectory check).
    """
    t_o = np.asarray(t_obs, dtype=float)
    y_o = np.asarray(z_obs, dtype=float)
    t_g = np.asarray(t_grid, dtype=float)
    om_yr = 2.0 * math.pi / period_steps
    bm = fit_info["best_model"]
    mods = fit_info["models"]

    has_seas = bm in ("M3_seas_ar1", "M4_seas_osc")
    X_o = np.column_stack([np.ones(len(t_o)), np.cos(om_yr * t_o), np.sin(om_yr * t_o)]) if has_seas else np.ones((len(t_o), 1))
    X_g = np.column_stack([np.ones(len(t_g)), np.cos(om_yr * t_g), np.sin(om_yr * t_g)]) if has_seas else np.ones((len(t_g), 1))

    dt_oo = np.abs(t_o[:, None] - t_o[None, :])
    dt_go = np.abs(t_g[:, None] - t_o[None, :])

    if bm == "M0_iid":
        K_oo = np.eye(len(t_o))
        K_go = np.where(dt_go == 0.0, 1.0, 0.0)
    elif bm in ("M1_ar1", "M3_seas_ar1"):
        phi_v = float(mods[bm]["phi"])
        K_oo = (abs(phi_v) ** dt_oo) * np.where((phi_v < 0) & (np.round(dt_oo) % 2 == 1), -1.0, 1.0)
        K_go = (abs(phi_v) ** dt_go) * np.where((phi_v < 0) & (np.round(dt_go) % 2 == 1), -1.0, 1.0)
    else:
        m_par = mods[bm]
        zt = float(m_par["zeta"])
        w0 = float(m_par["omega_0_rad_per_step"])
        ng = float(m_par["nugget"])
        K_oo = car2_correlation_from_lags(dt_oo, zt, w0, ng)
        # Latent signal covariance without measurement nugget on target grid
        c_go = np.exp(-zt * dt_go) * (np.cos(w0 * dt_go) + (zt / max(w0, 1e-8)) * np.sin(w0 * dt_go))
        K_go = (1.0 - ng) * c_go

    cho = linalg.cho_factor(K_oo + 1e-8 * np.eye(len(t_o)), lower=True, check_finite=False)
    Rinv_Xo = linalg.cho_solve(cho, X_o, check_finite=False)
    Rinv_yo = linalg.cho_solve(cho, y_o, check_finite=False)
    beta = linalg.solve(X_o.T @ Rinv_Xo + 1e-12 * np.eye(X_o.shape[1]), X_o.T @ Rinv_yo, assume_a="pos")
    resid_o = y_o - X_o @ beta
    alpha = linalg.cho_solve(cho, resid_o, check_finite=False)
    mean_g = X_g @ beta + K_go @ alpha
    return mean_g - float(np.mean(mean_g))


def verify_voles_models() -> dict[str, Any]:
    """Rule 7 verification suite with tolerances fixed beforehand:
      V1 (tol 1e-10): Proposition A thermal conservation (`J_temp = D_temp = 0` when Q10_dev == Q10_mort).
      V2 (tol 1e-4 rel): `Engine.coefficient()['D2']` == `Statement ring-B2` == `ring-B8b` spectral quadrature.
      V3 (tol 1e-12): Harmed-host Block V reduces to Block S at alpha=0 and obeys `|E_V(i*w)| <= 1 - p*` for rho >= 0.
      V4 (tol 1e-4): Exact Gaussian MLE nesting NLL(M4) <= NLL(M3) <= NLL(M0) and NLL(M2) <= NLL(M1) <= NLL(M0).
    """
    # V1: Proposition A conservation
    t_synth = 8.0 + 7.5 * np.sin(2.0 * math.pi * np.arange(52) / PERIOD_YR_STEPS) + 2.0 * np.cos(0.4 * np.arange(52))
    h_synth = np.exp(0.6 * np.cos(2.0 * math.pi * np.arange(52) / 39.0))
    res_prop_a = evaluate_thermal_trajectory_loss(
        t_synth,
        h_synth,
        tau_host_steps=2.5,
        nu_ref_per_step=0.45,
        mu_ref_per_step=0.25,
        T_ref_C=10.0,
        Q10_dev=2.4,
        Q10_mort=2.4,
        m_buf=0.7,
        eta_H=0.5,
        cold_floor_C=None,
        dt_sub=0.25,
    )
    err_v1 = max(abs(res_prop_a["J_temp_curvature"]), abs(res_prop_a["D_temp_only"]))
    assert err_v1 < 1e-10, f"V1 Proposition A failed: {res_prop_a}"

    # V2: Engine D2 vs Statement ring-B2 vs Statement ring-B8b quadrature on 1-step grid
    tau1, tau2 = 4.0, 9.0
    eta_v, sig_v, phi_v = 0.5, 0.8, 0.75
    th_v = phi_v / (1.0 - phi_v)
    R_v = (1.0 / (1.0 + tau1 / th_v)) * (1.0 / (1.0 + tau2 / th_v))
    T0_v = tau1 + tau2
    d2_b2 = (eta_v ** 2) * (sig_v ** 2) * ((1.0 + 2.0 * th_v) / (2.0 * T0_v) - R_v / (1.0 - R_v))
    psd_ar1 = lambda om: (1.0 - phi_v ** 2) / (1.0 + phi_v ** 2 - 2.0 * phi_v * np.cos(om))
    d2_b8b = discrete_ring_spectral_cost([tau1, tau2], eta=eta_v, sigma_H=sig_v, psd_unit_fn=psd_ar1)["Gamma_per_gen"]
    eng_v2 = Engine(
        Description(
            stages=[Stage("h", mean=tau1), Stage("v", mean=tau2)],
            transfers=[
                Transfer("h", "v", yield_=1.0, birth=True),
                Transfer("v", "h", yield_=1.0, driver="d", elasticity=eta_v),
            ],
            drivers=[Driver("d", sigma=sig_v, phi=phi_v, kind="lognormal")],
        )
    )
    d2_eng = float(eng_v2.coefficient()["D2"])
    rel_v2 = max(abs(d2_eng - d2_b2), abs(d2_b8b - d2_b2)) / abs(d2_b2)
    assert rel_v2 < 1e-4, f"V2 failed: eng={d2_eng}, b2={d2_b2}, b8b={d2_b8b}"

    # V3: Block V == Block S at alpha=0 and K3 bound |E_V(i*omega)| <= 1 - p*
    oms = np.linspace(0.0, 20.0, 500)
    hs0 = k9_transfer_functions(oms, a_H=1.0, p_star=0.35, eta_H=0.38, a_V=0.45)["H_p"]
    hv0 = k9_block_v_transfer_function(oms, a_H=1.0, p_star=0.35, eta_H=0.38, alpha_harm=0.0, rho_reg=0.25, a_V=0.45)["H_p"]
    err_v3 = float(np.max(np.abs(hs0 - hv0)))
    assert err_v3 < 1e-12, f"V3 reduction failed: {err_v3}"
    # Check K3 host-block bound for alpha_harm=0.22 <= a_H=1.0, rho_reg=0.20 >= 0
    s_im = 1j * oms
    q_b, d_b, h_b, r_b = 1.0, 1.0 / (1.0 - 0.35), 0.22 * 0.35, 0.20
    E_host_mod = np.abs(q_b * (s_im + r_b) / ((s_im + r_b) * (s_im + d_b) + d_b * h_b))
    assert float(np.max(E_host_mod)) <= (1.0 - 0.35) + 1e-12, "V3 K3 damping bound violated"

    # V4: Gaussian MLE nesting on gapped synthetic series
    rng_v4 = np.random.default_rng(20261115)
    t_v4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 30, 31], dtype=float)
    z_v4 = 0.7 * np.cos(OMEGA_YR * t_v4) + 0.4 * rng_v4.normal(size=len(t_v4))
    fit_v4 = fit_host_memory_panel([("s1", t_v4, z_v4)])
    nlls = {k: v["nll"] for k, v in fit_v4["models"].items()}
    nlls_d = {k: v["nll"] for k, v in fit_v4["companion_discrete_ar2"].items()}
    assert nlls["M1_ar1"] <= nlls["M0_iid"] + 1e-4
    assert nlls["M3_seas_ar1"] <= nlls["M1_ar1"] + 1e-4
    assert nlls_d["M2_disc_ar2"] <= nlls["M1_ar1"] + 1e-3
    assert nlls_d["M4_seas_disc_ar2"] <= nlls["M3_seas_ar1"] + 1e-3
    assert nlls["M2_osc"] <= nlls["M0_iid"] + 1e-3
    assert nlls["M4_seas_osc"] <= nlls["M3_seas_ar1"] + 5e-2

    return {
        "passed": True,
        "V1_prop_A_max_abs_err": err_v1,
        "V2_engine_vs_B2_B8b_rel_err": rel_v2,
        "V3_block_V_reduction_err": err_v3,
        "V4_mle_nlls": nlls,
    }
