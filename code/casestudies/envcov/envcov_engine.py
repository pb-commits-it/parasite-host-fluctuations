"""Lifecycle-engine evaluations, Part 4 masking/mimicking simulations, and factor sensitivities.

Uses `studies/engine-R16/lifecycle_engine.py` (Rule 3 of CASESTUDY_SPEC.md),
observing both caveats documented in `verification/engine-R16-check.md`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

ENGINE_PATH = (__import__('pathlib').Path(__file__).resolve().parent / '../../studies/engine-R16/lifecycle_engine.py').resolve()
_spec = importlib.util.spec_from_file_location("lifecycle_engine", ENGINE_PATH)
le = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = le
_spec.loader.exec_module(le)

Stage = le.Stage
Transfer = le.Transfer
Driver = le.Driver
Patches = le.Patches
Description = le.Description
Engine = le.Engine
ModelError = le.ModelError

DT_COARSE_DAYS = 4.4 * 30.4375  # 133.925 days (4.4 average Gregorians months, matching R10/R12/R14)
DT_FINE_DAYS = 6.0              # 6-day step: 22.32 steps in 4.4-mo mouse stay, ~50 steps in 10-mo tick stay
MOUSE_STAY_DAYS = 133.0         # Dunn et al. (2013, PMC3913058, Table 1, read in full)
BORROWED_TICK_DAYS = 10.0 * 30.4375  # 304.375 days (10 months, R10/R12/R14 benchmark)


def closed_form_d2_serial(
    stays: list[float],
    substages: list[int],
    sigma: float,
    phi: float,
    kappa: float = 1.0,
) -> float:
    """Analytical second-order loss D_2 for one lognormal transfer in a primitive serial chain."""
    t0 = float(sum(stays))
    ks2 = (kappa * sigma) ** 2
    if abs(phi) < 1e-15:
        return ks2 / (2.0 * t0)
    theta = phi / (1.0 - phi)
    r_prod = 1.0
    for tau, m in zip(stays, substages):
        r_prod *= (1.0 + tau / (m * theta)) ** (-m)
    return ks2 * ((1.0 + 2.0 * theta) / (2.0 * t0) - r_prod / (1.0 - r_prod))


def discrete_chain_substages(mean_steps: float, target_cv: float) -> tuple[int, int, float]:
    """Return (m_disc, m_cont, realized_cv_disc) for a discrete serial chain of mean_steps >= 1.

    Per Protocol Amendment 1: on discrete steps {1, 2, ...}, m equal geometric sub-stages of total
    mean B have Var(L)/B^2 = 1/m - 1/B, so m_disc = round(1 / (CV^2 + 1/B)) bounded to [1, floor(B)].
    """
    m_max = max(1, int(math.floor(mean_steps)))
    if target_cv <= 0.0:
        m_disc = m_max
        m_cont = m_max
    else:
        m_disc = max(1, min(int(round(1.0 / (target_cv * target_cv + 1.0 / mean_steps))), m_max))
        m_cont = max(1, min(int(round(1.0 / (target_cv * target_cv))), m_max))
    cv_real = math.sqrt(max(0.0, 1.0 / m_disc - 1.0 / mean_steps))
    return m_disc, m_cont, cv_real


def _eval_engine_job(job: dict) -> dict:
    """Worker function evaluating exact D, D2, and J for one Description specification."""
    desc: Description = job["desc"]
    steps: int = job.get("steps", 24000)
    burn: int = job.get("burn", 1500)
    pairs: int = job.get("pairs", 16)
    seed: int = job.get("seed", 101)

    model = Engine(desc).critical()
    try:
        d2 = model.coefficient()["D2"]
    except le.PeriodicModelError:
        d2 = None
    res = model.loss(steps=steps, burn=burn, pairs=pairs, seed=seed)
    out = {
        "key": job["key"],
        "D": float(res["D"]),
        "se": float(res["se"]),
        "D2": float(d2) if d2 is not None else None,
        "J": float(res["J"]),
        "J_transfer": float(res["J_transfer"]),
        "generation_steps": float(1.0 / model.b),
        "period": int(model.period),
    }
    if job.get("run_extinction", False):
        n0 = int(job.get("ext_initial", 500))
        y_margin = float(job.get("ext_margin", 0.05))
        horizon = int(math.ceil(job.get("ext_generations", 10.0) / model.b))
        init = np.zeros(model.n, dtype=np.int64)
        init[0] = n0
        ext = model.extinction(init, steps=horizon, trials=512, seed=seed + 77, y=out["D"] + y_margin)
        out["ext_conditional"] = float(ext["conditional_mean"])
        out["ext_fraction"] = float(ext["fraction"])
        out["ext_se"] = float(ext["se"])
        out["ext_steps"] = int(horizon)
    return out


def make_borrelia_desc(
    tau_mouse: float,
    tau_tick: float,
    sigma_mouse: float,
    phi_step: float = 0.0,
    kappa1: float = 1.0,
    kappa2: float = 0.0,
    mouse_shape: str = "memoryless",
    mouse_substages: int = 1,
    tick_shape: str = "memoryless",
    tick_substages: int = 1,
    tick_death: float = 0.0,
    contact_tick: bool = False,
    driver_kind: str = "lognormal",
    bounded_low: float | None = None,
    sigma_weather: float = 0.0,
    phi_weather: float = 0.0,
    rho_hw: float = 0.0,
    alt_host_sigma: float | None = None,
    alt_host_share: float = 0.5,
    alt_host_corr: float = 0.0,
    n_patches: int = 1,
    patch_move: float = 0.0,
) -> Description:
    """Build a discrete-time Borrelia cycle Description for lifecycle_engine.py."""
    s_mouse = Stage("mouse", mean=tau_mouse, shape=mouse_shape, substages=mouse_substages)
    s_tick = Stage(
        "tick",
        mean=tau_tick,
        shape=tick_shape,
        substages=tick_substages,
        death=tick_death,
        contact=contact_tick,
    )
    drivers = [Driver("mouse", sigma=sigma_mouse, phi=phi_step, kind=driver_kind, low=bounded_low)]
    corr_rows = [[1.0]]

    transfers = []
    if alt_host_sigma is not None:
        drivers.append(Driver("alt", sigma=alt_host_sigma, phi=phi_step, kind=driver_kind, low=bounded_low))
        corr_rows = [[1.0, alt_host_corr], [alt_host_corr, 1.0]]
        transfers.append(
            Transfer("tick", "mouse", share=1.0 - alt_host_share, driver="mouse", elasticity=kappa1)
        )
        transfers.append(
            Transfer("tick", "mouse", share=alt_host_share, driver="alt", elasticity=kappa1)
        )
    else:
        transfers.append(
            Transfer("tick", "mouse", share=1.0, driver="mouse", elasticity=kappa1)
        )

    if sigma_weather > 0.0:
        drivers.append(Driver("weather", sigma=sigma_weather, phi=phi_weather, kind="lognormal"))
        if len(corr_rows) == 1:
            corr_rows = [[1.0, rho_hw], [rho_hw, 1.0]]
        else:
            corr_rows = [
                [1.0, alt_host_corr, rho_hw],
                [alt_host_corr, 1.0, rho_hw],
                [rho_hw, rho_hw, 1.0],
            ]
        # Mouse -> tick carries optional dilution kappa2 on mouse; if kappa2==0, weather drives tick survival
        if abs(kappa2) < 1e-12:
            transfers.append(Transfer("mouse", "tick", driver="weather", elasticity=1.0, birth=True))
        else:
            # Split into intermediate 1-step survival stage if both mouse dilution and weather enter
            s_surv = Stage("overwinter", mean=max(1.0, tau_tick - 1.0), shape=tick_shape, substages=min(tick_substages, int(max(1.0, tau_tick - 1.0))))
            s_feed = Stage("feed", mean=1.0, shape="fixed")
            return Description(
                stages=[s_mouse, s_feed, s_surv],
                transfers=[
                    Transfer("mouse", "feed", driver="mouse", elasticity=kappa2),
                    Transfer("feed", "overwinter", driver="weather", elasticity=1.0, birth=True),
                    Transfer("overwinter", "mouse", driver="mouse", elasticity=kappa1),
                ],
                drivers=drivers,
                correlation=np.array(corr_rows, dtype=float),
                patches=Patches(count=n_patches, mobile_stage="overwinter" if n_patches > 1 else None, movement=patch_move),
            )
    else:
        transfers.append(
            Transfer(
                "mouse",
                "tick",
                driver="mouse" if abs(kappa2) > 1e-12 else None,
                elasticity=kappa2 if abs(kappa2) > 1e-12 else 1.0,
                birth=True,
            )
        )

    patches = Patches(
        count=n_patches,
        synchrony=0.0,
        mobile_stage="tick" if n_patches > 1 else None,
        movement=patch_move if n_patches > 1 else 0.0,
        topology="island",
    )
    return Description(
        stages=[s_mouse, s_tick],
        transfers=transfers,
        drivers=drivers,
        correlation=np.array(corr_rows, dtype=float),
        patches=patches,
    )


def run_engine_batch(jobs: list[dict], max_workers: int = 4) -> dict[str, dict]:
    """Evaluate a list of engine jobs in parallel and return a dict keyed by job['key']."""
    results: dict[str, dict] = {}
    if max_workers <= 1 or len(jobs) <= 2:
        for jb in jobs:
            r = _eval_engine_job(jb)
            results[r["key"]] = r
        return results
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(_eval_engine_job, jobs):
            results[r["key"]] = r
    return results


def engine_code_checks() -> None:
    """Rule 7 verification of lifecycle_engine.py against closed-form formulas (tolerances fixed)."""
    # C0a-1: Geometric and serial-chain D2 vs closed form (< 1e-12 relative)
    for m_sub in (1, 8):
        for phi in (0.0, 0.75):
            desc = make_borrelia_desc(
                tau_mouse=4.0,
                tau_tick=24.0,
                sigma_mouse=0.8,
                phi_step=phi,
                tick_shape="chain" if m_sub > 1 else "memoryless",
                tick_substages=m_sub,
            )
            d2_eng = Engine(desc).critical().coefficient()["D2"]
            d2_cf = closed_form_d2_serial([4.0, 24.0], [1, m_sub], sigma=0.8, phi=phi)
            rel_err = abs(d2_eng - d2_cf) / abs(d2_cf)
            assert rel_err < 1e-12, f"D2 mismatch m={m_sub}, phi={phi}: {d2_eng} vs {d2_cf}"

    # C0a-2: Exact 1-stage lognormal D vs Gauss-Hermite quadrature (within max(3 SE, 2%))
    s_stay, sig = 0.5, 0.6
    desc1 = Description(
        stages=[Stage("one", mean=2.0, shape="memoryless")],
        transfers=[Transfer("one", "one", driver="h", elasticity=1.0, birth=True)],
        drivers=[Driver("h", sigma=sig, phi=0.75, kind="lognormal")],
    )
    res1 = Engine(desc1).critical().loss(steps=40000, burn=2000, pairs=24, seed=2026)
    # Gauss-Hermite exact root of E[ln(0.5 + 0.5 * exp(D + sig*Z - sig^2/2))] = 0
    z_gh, w_gh = np.polynomial.hermite.hermgauss(60)
    z_std = z_gh * math.sqrt(2.0)
    w_std = w_gh / math.sqrt(math.pi)
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = float(w_std @ np.log(s_stay + (1.0 - s_stay) * np.exp(mid + sig * z_std - 0.5 * sig * sig)))
        if val < 0:
            lo = mid
        else:
            hi = mid
    d_exact = 0.5 * (lo + hi)
    diff = abs(res1["D"] - d_exact)
    tol = max(3.0 * res1["se"], 0.02 * d_exact)
    assert diff <= tol, f"1-stage exact D check failed: {res1['D']} vs {d_exact} (tol {tol})"

    # C0a-3: Reproduce R12 Part C (stays=[1, 10/4.4], sigma=1.0304, phi=0 -> R12 got 0.2013 +- 0.0002)
    desc_r12 = make_borrelia_desc(
        tau_mouse=1.0,
        tau_tick=10.0 / 4.4,
        sigma_mouse=1.0304,
        phi_step=0.0,
        mouse_shape="fixed",
        tick_shape="memoryless",
    )
    res_r12 = Engine(desc_r12).critical().loss(steps=40000, burn=2000, pairs=24, seed=202612)
    comb_se = math.hypot(res_r12["se"], 0.0002)
    assert abs(res_r12["D"] - 0.2013) <= 4.0 * comb_se, f"R12 D reproduction failed: {res_r12['D']} vs 0.2013"
    assert abs(Engine(desc_r12).critical().coefficient()["D2"] - 0.16221) < 1e-4
    print(
        f"  C0a Engine checks: D2 rel_err < 1e-12; 1-stage D={res1['D']:.5f}+-{res1['se']:.5f} (exact {d_exact:.5f}); "
        f"R12 D={res_r12['D']:.4f}+-{res_r12['se']:.4f} (R12 0.2013+-0.0002) [ok]"
    )


def simulate_neon_masking_mimicking(
    sigma_sites: np.ndarray,
    ntest_sites: np.ndarray,
    nyears_sites: np.ndarray,
    d_by_kappa: dict[float, np.ndarray],
    emp_r_zs: float,
    emp_beta_z: float,
    emp_tau_res: float,
    emp_sd_w_within: float,
    emp_gamma_w_within: float,
    p_obs_mean: float = 0.1842,
    n_mc: int = 20_000,
    seed: int = 2026111501,
) -> list[dict]:
    """Simulate 6-site multi-year weather + mouse studies calibrated to Engine losses D_site(kappa).

    In each synthetic study:
      - Site-mean climate Z_s (6,) is drawn with correlation `r_zs` with standardized `sigma_site`,
        shifting site baseline ln R0 by `beta_z * Z_s`, plus unmeasured site residual `tau_res * eps_s`.
      - Within each site s over `nyears_sites[s]` annual cohorts, annual weather shocks `w_{s,t}`
        perturb annual log reproductive success by `gamma_w * w_{s,t}` while long-run host fluctuation
        imposes the Engine-computed loss `D_site(kappa)`.
      - `ntest_sites[s]` nymphs are tested binomially across the site's study years, and both R14's
        registered unadjusted Spearman test and envcov's PC1_env Freedman-Lane partial rank test run.
    """
    from power_precheck import CUT_RHO, OBS_RHO, partial_fl_perm_p, spearman_batch

    rng = np.random.default_rng(seed)
    xs = (sigma_sites - sigma_sites.mean()) / sigma_sites.std(ddof=1)

    def solve_m(d_vec: np.ndarray, f: float = 0.5) -> float:
        lo, hi = 0.0, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            pm = float(np.mean(1.0 - np.exp(-f * np.maximum(mid - d_vec, 0.0))))
            if pm < p_obs_mean:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    regimes = [
        ("No weather confounding", 0.0, 0.0, emp_tau_res, 0.0),
        ("Empirical NEON climate-NIP cov", emp_r_zs, emp_beta_z, emp_tau_res, emp_gamma_w_within),
        ("Opposite climate cov (mimics D)", -abs(emp_r_zs), abs(emp_beta_z), emp_tau_res, emp_gamma_w_within),
    ]

    rows = []
    for scen_name, r_zs, beta_z, tau_res, gam_w in regimes:
        for kappa in (0.0, 0.5, 1.0):
            d_vec = d_by_kappa[kappa]
            m0 = solve_m(d_vec)
            eps_z = rng.standard_normal((n_mc, 6))
            z_site = r_zs * xs[None, :] + math.sqrt(max(0.0, 1.0 - r_zs * r_zs)) * eps_z
            site_base = m0 + beta_z * z_site + tau_res * rng.standard_normal((n_mc, 6)) - d_vec[None, :]

            # Average over annual weather shocks across each site's tested years
            p_site = np.zeros((n_mc, 6), dtype=float)
            z_obs = z_site.copy()
            for s_idx in range(6):
                ny = int(nyears_sites[s_idx])
                w_yr = emp_sd_w_within * rng.standard_normal((n_mc, ny))
                z_obs[:, s_idx] = z_site[:, s_idx] + w_yr.mean(axis=1) * 0.15
                ln_r_yr = site_base[:, s_idx : s_idx + 1] + gam_w * w_yr
                p_yr = 1.0 - np.exp(-0.5 * np.maximum(ln_r_yr, 0.0))
                p_site[:, s_idx] = p_yr.mean(axis=1)

            k_pos = rng.binomial(ntest_sites[None, :], np.clip(p_site, 1e-5, 1.0 - 1e-5))
            p_hat = k_pos / ntest_sites[None, :] + 1e-9 * rng.standard_normal(k_pos.shape)

            rho = spearman_batch(sigma_sites, p_hat)
            r_part, p_fl = partial_fl_perm_p(p_hat, sigma_sites, z_obs, use_ranks=True)

            rows.append(
                {
                    "scenario": scen_name,
                    "kappa": float(kappa),
                    "r_zs": float(r_zs),
                    "beta_z": float(beta_z),
                    "tau_res": float(tau_res),
                    "mean_rho_unadj": float(rho.mean()),
                    "p_sign_unadj": float(np.mean(rho < 0.0)),
                    "p_sup_unadj": float(np.mean(rho <= CUT_RHO + 1e-9)),
                    "p_pos49_unadj": float(np.mean(rho >= OBS_RHO - 1e-9)),
                    "mean_r_adj": float(r_part.mean()),
                    "p_sign_adj": float(np.mean(r_part < 0.0)),
                    "p_sup_adj": float(np.mean(p_fl < 0.05)),
                }
            )
    return rows


def simulate_cary_masking_mimicking(
    emp_r_wm: float,
    emp_gam_w: float,
    sigma_m_yr: float = 1.00,
    sigma_m_grid: float = 0.25,
    r1_m: float = -0.11,
    pbar: float = 0.413,
    n_years: int = 18,
    n_grids: int = 6,
    n_nymphs: int = 53,
    n_mc: int = 2_000,
    seed: int = 2026111502,
    b_values: tuple[tuple[str, float], ...] = (
        ("kappa=0 (no effect)", 0.0),
        ("empirical b=0.038", 0.038),
        ("kappa=0.5 (b=0.147)", 0.147),
        ("kappa=1.0 (b=0.294)", 0.294),
    ),
) -> list[dict]:
    """Simulate 18-year x 6-grid Cary cohort datasets under coupled annual weather and AR(1) mice."""
    rng = np.random.default_rng(seed)
    upper = 1.0 - pbar

    regimes = [
        ("Empirical Cary weather-mouse cov", emp_r_wm, emp_gam_w),
        ("Weather masking (r_WM=-0.45, gam_W=+0.20)", -0.45, +0.20),
        ("Weather mimicking (r_WM=+0.45, gam_W=+0.20)", +0.45, +0.20),
    ]

    rows = []
    for scen_name, r_wm, gam_w in regimes:
        for b_label, b_true in b_values:
            # Simulate AR(1) landscape mouse series correlated with landscape weather w_yr
            w_yr = rng.standard_normal((n_mc, n_years))
            eps_m = r_wm * w_yr + math.sqrt(max(0.0, 1.0 - r_wm * r_wm)) * rng.standard_normal((n_mc, n_years))
            m_yr = np.empty((n_mc, n_years), dtype=float)
            m_yr[:, 0] = sigma_m_yr * eps_m[:, 0]
            innov_scale = sigma_m_yr * math.sqrt(max(0.0, 1.0 - r1_m * r1_m))
            for t in range(1, n_years):
                m_yr[:, t] = r1_m * m_yr[:, t - 1] + innov_scale * eps_m[:, t]

            x = m_yr[:, :, None] + sigma_m_grid * rng.standard_normal((n_mc, n_years, n_grids))
            yr_shock = 0.14 * rng.standard_normal((n_mc, n_years, 1))
            ln_p = math.log(pbar) + b_true * (x - x.mean(axis=(1, 2), keepdims=True)) + gam_w * w_yr[:, :, None] + yr_shock
            p = np.clip(np.exp(ln_p), 0.02, 0.95)
            k = rng.binomial(n_nymphs, p)
            y = np.log(np.maximum(k, 0.5) / n_nymphs)

            xd = x - x.mean(axis=1, keepdims=True)
            yd = y - y.mean(axis=1, keepdims=True)
            sxx_un = (xd * xd).sum(axis=(1, 2))
            b_un = (xd * yd).sum(axis=(1, 2)) / sxx_un
            res_un = yd - b_un[:, None, None] * xd
            score_un = (xd * res_un).sum(axis=2)
            se_un = np.sqrt((n_years / (n_years - 1)) * (score_un * score_un).sum(axis=1)) / sxx_un

            wd = w_yr - w_yr.mean(axis=1, keepdims=True)
            w_ss = (wd * wd).sum(axis=1, keepdims=True) * n_grids
            cx = (xd * wd[:, :, None]).sum(axis=(1, 2), keepdims=True) / w_ss[:, :, None]
            cy = (yd * wd[:, :, None]).sum(axis=(1, 2), keepdims=True) / w_ss[:, :, None]
            xd_a = xd - cx * wd[:, :, None]
            yd_a = yd - cy * wd[:, :, None]
            sxx_a = (xd_a * xd_a).sum(axis=(1, 2))
            b_adj = (xd_a * yd_a).sum(axis=(1, 2)) / sxx_a
            res_a = yd_a - b_adj[:, None, None] * xd_a
            score_a = (xd_a * res_a).sum(axis=2)
            se_adj = np.sqrt((n_years / (n_years - 2)) * (score_a * score_a).sum(axis=1)) / sxx_a

            rows.append(
                {
                    "scenario": scen_name,
                    "b_label": b_label,
                    "b_true": float(b_true),
                    "r_wm": float(r_wm),
                    "gamma_w": float(gam_w),
                    "mean_b_unadj": float(b_un.mean()),
                    "sd_b_unadj": float(b_un.std(ddof=1)),
                    "p_pos_unadj": float(np.mean(b_un > 0.0)),
                    "p_gt2se_unadj": float(np.mean(b_un > 2.0 * se_un)),
                    "p_p2band_unadj": float(np.mean((b_un >= -2.0 * se_un) & (b_un <= upper + 2.0 * se_un))),
                    "mean_b_adj": float(b_adj.mean()),
                    "sd_b_adj": float(b_adj.std(ddof=1)),
                    "p_pos_adj": float(np.mean(b_adj > 0.0)),
                    "p_gt2se_adj": float(np.mean(b_adj > 2.0 * se_adj)),
                    "p_p2band_adj": float(np.mean((b_adj >= -2.0 * se_adj) & (b_adj <= upper + 2.0 * se_adj))),
                }
            )
    return rows
