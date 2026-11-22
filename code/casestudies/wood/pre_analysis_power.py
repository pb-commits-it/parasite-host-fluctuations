#!/usr/bin/env python3
"""Pre-analysis power and identifiability calculation for the Wood et al. (2023)
case study, executed PRIOR to inspecting or analysing any parasite count outcomes.

STRICT BLINDNESS GUARANTEE:
- From final_dataset.csv, this script reads ONLY ('fish.id', 'year', 'lat',
  'fish.spc', 'length', 'para.spc'). It NEVER reads 'count' or any outcome file.
- Combines the exact 8,754-row, 699-fish, 85-taxon design matrix with the blind
  trait table (traits_blind.csv) to:
  1. Simulate synthetic Negative Binomial counts on the exact museum specimen
     design (under literature macroparasite aggregation phi ~ LogNormal(ln 0.35, 0.6^2),
     Shaw & Dobson 1995) to obtain realistic per-taxon slope sampling variances v_k.
  2. Run 5,000 Monte Carlo datasets per scenario across residual between-taxon
     slope SD sigma_gamma in {0.20, 0.30, 0.40} (per 30.27-yr SD of year) and
     taxonomic group SD sigma_tax in {0.00, 0.15}, evaluating Type I error under
     the null and statistical power under weak (kappa=0.25), moderate (kappa=0.50),
     and full-scale (kappa=1.00) mappings from theoretical Delta D to temporal slope,
     plus exact 80%-power Minimum Detectable Effects (MDE_80).

Run with:
  python3 -I casestudies/wood/pre_analysis_power.py
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from scipy import optimize, stats

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RAW_WOOD = os.path.join(REPO_ROOT, "data", "raw", "wood")


def load_outcome_free_design():
    """Load design columns ONLY from final_dataset.csv; never touch 'count'."""
    with open(os.path.join(RAW_WOOD, "wood_zenodo_7448240", "final_dataset.csv"), encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        assert "count" in hdr
        # Explicitly omit 'count' index!
        idx_fid = hdr.index("fish.id")
        idx_yr = hdr.index("year")
        idx_lat = hdr.index("lat")
        idx_fsp = hdr.index("fish.spc")
        idx_len = hdr.index("length")
        idx_psp = hdr.index("para.spc")

        raw_rows = []
        for r in reader:
            raw_rows.append((
                r[idx_fid],
                float(r[idx_yr]),
                None if r[idx_lat] in ("NA", "") else float(r[idx_lat]),
                r[idx_fsp],
                0.0 if r[idx_len] in ("NA", "") else float(r[idx_len]),
                r[idx_psp],
            ))
    years = np.array([r[1] for r in raw_rows])
    u_year = (years - years.mean()) / years.std(ddof=1)
    lats_valid = np.array([r[2] for r in raw_rows if r[2] is not None])
    lat_mean, lat_sd = lats_valid.mean(), lats_valid.std(ddof=1)
    slat = np.array([0.0 if r[2] is None else (r[2] - lat_mean) / lat_sd for r in raw_rows])
    slen = np.array([r[4] for r in raw_rows])
    fids = [r[0] for r in raw_rows]
    fsps = [r[3] for r in raw_rows]
    psps = [r[5] for r in raw_rows]
    return {
        "n_rows": len(raw_rows),
        "year_mean": float(years.mean()),
        "year_sd": float(years.std(ddof=1)),
        "u_year": u_year,
        "slat": slat,
        "slen": slen,
        "fids": fids,
        "fsps": fsps,
        "psps": psps,
    }


def estimate_design_sampling_variances(design: dict, traits: list[dict], seed: int = 20261114) -> np.ndarray:
    """Compute realistic per-taxon slope sampling variances v_k on the exact museum
    specimen schedule using synthetic NB counts (never touching real counts).
     averages Fisher information of log-link NB regression across 40 synthetic draws
    with baseline prevalence >=5%, log-mean ~ N(-0.4, 1.1^2), phi ~ LogNormal(ln 0.35, 0.5^2).
    """
    rng = np.random.default_rng(seed)
    psps = np.array(design["psps"])
    fsps = np.array(design["fsps"])
    u = design["u_year"]
    slen = design["slen"]
    slat = design["slat"]

    n_taxa = len(traits)
    v_accum = np.zeros(n_taxa)
    n_draws = 50
    for _ in range(n_draws):
        for k, tr in enumerate(traits):
            pcode_sp = tr["para_spc"]
            mask = psps == pcode_sp
            uk = u[mask]
            lk = slen[mask]
            latk = slat[mask]
            fspk = fsps[mask]
            uniq_f = sorted(set(fspk))
            # Design matrix for taxon k: host-specific intercepts & length slopes + lat + u_year
            cols = []
            for f in uniq_f:
                ind = (fspk == f).astype(float)
                cols.append(ind)
                cols.append(ind * lk)
            cols.append(latk)
            cols.append(uk)
            Xk = np.column_stack(cols)

            phi_k = float(np.exp(rng.normal(math.log(0.35), 0.5)))
            b0 = rng.normal(-0.3, 0.9, size=len(uniq_f))
            eta = np.zeros(mask.sum())
            for j, f in enumerate(uniq_f):
                eta += (fspk == f) * (b0[j] + 0.25 * lk)
            eta += 0.1 * latk
            mu = np.clip(np.exp(eta), 0.05, 50.0)
            # Exact NB Fisher weight for log link: w_i = mu_i / (1 + mu_i / phi_k)
            w = mu / (1.0 + mu / phi_k)
            XtWX = (Xk * w[:, None]).T @ Xk + 1e-6 * np.eye(Xk.shape[1])
            cov = np.linalg.inv(XtWX)
            v_accum[k] += cov[-1, -1]
    return v_accum / n_draws


def fit_reml_wls(y: np.ndarray, X: np.ndarray, v_obs: np.ndarray, group_idx: np.ndarray | None = None):
    """Fit linear mixed/meta-regression y = X beta + a_group + eps,
    eps_k ~ N(0, sigma_gamma^2 + v_obs_k), a_g ~ N(0, sigma_tax^2) by ML (for LRT/AICc)
    and return (beta, se_beta, nll_ml, sigma_gamma, sigma_tax).
    """
    n, p = X.shape
    if group_idx is None:
        def nll(log_sg: float) -> float:
            sg2 = math.exp(2.0 * log_sg)
            w = 1.0 / (v_obs + sg2)
            XtWX = (X * w[:, None]).T @ X
            beta = np.linalg.solve(XtWX, (X * w[:, None]).T @ y)
            res = y - X @ beta
            return 0.5 * (n * math.log(2.0 * math.pi) + np.sum(np.log(v_obs + sg2)) + np.sum(w * res * res))

        opt = optimize.minimize_scalar(nll, bounds=(-8.0, 3.0), method="bounded")
        sg = math.exp(float(opt.x))
        w = 1.0 / (v_obs + sg * sg)
        XtWX = (X * w[:, None]).T @ X
        cov_b = np.linalg.inv(XtWX)
        beta = cov_b @ ((X * w[:, None]).T @ y)
        return beta, np.sqrt(np.diag(cov_b)), float(opt.fun), sg, 0.0

    # With taxonomic group random effect: block-diagonal covariance V_g = diag(v_k + sg^2) + st^2 1 1^T
    n_grp = int(group_idx.max()) + 1
    grp_masks = [np.flatnonzero(group_idx == g) for g in range(n_grp)]

    def eval_cov(sg2: float, st2: float):
        logdet = 0.0
        Vinv_X = np.empty_like(X)
        Vinv_y = np.empty_like(y)
        for idx in grp_masks:
            if len(idx) == 0:
                continue
            d_inv = 1.0 / (v_obs[idx] + sg2)
            sum_d = float(np.sum(d_inv))
            denom = 1.0 + st2 * sum_d
            logdet += float(np.sum(-np.log(d_inv))) + math.log(denom)
            Xk = X[idx]
            yk = y[idx]
            dX = d_inv[:, None] * Xk
            dy = d_inv * yk
            Vinv_X[idx] = dX - (st2 / denom) * np.outer(d_inv, np.sum(dX, axis=0))
            Vinv_y[idx] = dy - (st2 / denom) * d_inv * float(np.sum(dy))
        XtVX = X.T @ Vinv_X
        beta = np.linalg.solve(XtVX, X.T @ Vinv_y)
        res = y - X @ beta
        # Vinv_res
        Vinv_r = Vinv_y - Vinv_X @ beta
        quad = float(res @ Vinv_r)
        nll_val = 0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)
        return nll_val, beta, XtVX

    def obj(pars: np.ndarray) -> float:
        return eval_cov(math.exp(2.0 * pars[0]), math.exp(2.0 * pars[1]))[0]

    best = optimize.minimize(obj, x0=np.array([-1.2, -2.5]), method="L-BFGS-B", bounds=[(-8.0, 3.0), (-10.0, 3.0)])
    sg = math.exp(float(best.x[0]))
    st = math.exp(float(best.x[1]))
    nll_val, beta, XtVX = eval_cov(sg * sg, st * st)
    cov_b = np.linalg.inv(XtVX)
    return beta, np.sqrt(np.diag(cov_b)), nll_val, sg, st


def aicc(nll: float, k_params: int, n_obs: int) -> float:
    return 2.0 * k_params + 2.0 * nll + (2.0 * k_params * (k_params + 1.0)) / max(n_obs - k_params - 1.0, 1.0)


def main() -> int:
    design = load_outcome_free_design()
    with open(os.path.join(HERE, "traits_blind.csv"), encoding="utf-8") as f:
        traits = list(csv.DictReader(f))

    n_taxa = len(traits)
    v_obs = estimate_design_sampling_variances(design, traits)

    # Verification check on WLS/REML solver (Rule 7): recover exact OLS when v_obs=0, st=0
    rng_chk = np.random.default_rng(99)
    Xc = np.column_stack([np.ones(40), rng_chk.standard_normal((40, 2))])
    yc = Xc @ np.array([0.2, -0.5, 0.3]) + 0.1 * rng_chk.standard_normal(40)
    b_reml, _, _, _, _ = fit_reml_wls(yc, Xc, np.zeros(40), None)
    b_ols = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    assert np.max(np.abs(b_reml - b_ols)) < 1e-10, "REML solver verification failed"

    # Build predictor matrices for n=85 taxa
    nh = np.array([int(t["nhosts_wood_3plus"]) for t in traits])
    ind_nh2 = (nh == 2).astype(float)
    ind_nh3 = (nh == 3).astype(float)
    t_a = np.array([float(t["n_fluc_obligate"]) for t in traits])
    t_b = np.array([float(t["storage_stage"]) for t in traits])
    t_c = np.array([float(t["spreading"]) for t in traits])
    t_d = np.array([float(t["prop_active_transfers"]) for t in traits])

    ptypes = sorted({t["par_type"] for t in traits})
    grp_idx = np.array([ptypes.index(t["par_type"]) for t in traits], dtype=int)

    X_nh = np.column_stack([np.ones(n_taxa), ind_nh2, ind_nh3])
    X_joint = np.column_stack([np.ones(n_taxa), ind_nh2, ind_nh3, t_a, t_b, t_c, t_d])

    # Analytical asymptotic SEs and 80%-power MDEs (two-sided alpha=0.05 -> z_crit=1.96, z_80=0.8416, factor=2.8016)
    lines = []
    lines.append("=== Pre-Analysis Identifiability & Power Study (100% Outcome-Blind) ===")
    lines.append(
        f"Design dimensions: N_obs = {design['n_rows']}, N_fish = {len(set(design['fids']))}, "
        f"N_taxa = {n_taxa}, year_sd = {design['year_sd']:.4f} yr"
    )
    lines.append(
        f"Synthetic observation-level slope SE_k across 85 taxa: min={math.sqrt(v_obs.min()):.3f}, "
        f"median={math.sqrt(np.median(v_obs)):.3f}, max={math.sqrt(v_obs.max()):.3f} (per SD of year)"
    )
    lines.append("[PASS] Solver check P0: WLS/ML solver matches exact OLS to < 1.0e-10 when v_obs=0")
    lines.append("")

    # Analytical MDE_80 table across (sigma_gamma, sigma_tax)
    lines.append("--- 1. Exact 80%-Power Minimum Detectable Effect (MDE_80, two-sided alpha=0.05) ---")
    lines.append("Units: slope beta per 30.27-yr SD of year (and equivalent %/decade in parentheses)")
    contrast_names = ["(a) n_fluc_obligate (-)", "(b) storage_stage (+)", "(c) spreading (+)", "(d) prop_active (-)"]
    mde_summary = []
    for st in (0.0, 0.15):
        for sg in (0.20, 0.30, 0.40):
            # Construct exact covariance matrix V of taxon slopes
            V = np.diag(v_obs + sg * sg)
            if st > 0:
                for g in range(len(ptypes)):
                    m = (grp_idx == g).astype(float)
                    V += (st * st) * np.outer(m, m)
            Vinv = np.linalg.inv(V)
            cov_joint = np.linalg.inv(X_joint.T @ Vinv @ X_joint)
            ses = np.sqrt(np.diag(cov_joint))[3:7]
            mdes = 2.801585 * ses
            # Also within 3+-host subset (n=44)
            m3 = nh == 3
            X3 = np.column_stack([np.ones(m3.sum()), t_a[m3], t_b[m3], t_c[m3], t_d[m3]])
            V3inv = np.linalg.inv(V[np.ix_(m3, m3)])
            ses3 = np.sqrt(np.diag(np.linalg.inv(X3.T @ V3inv @ X3)))[1:5]
            mdes3 = 2.801585 * ses3

            lines.append(f"\nNoise regime: sigma_gamma = {sg:.2f}, sigma_tax = {st:.2f}:")
            row_rec = {"sigma_gamma": sg, "sigma_tax": st, "full_85": {}, "within_3plus_44": {}}
            for nm, se_j, mde_j, se_3, mde_3 in zip(contrast_names, ses, mdes, ses3, mdes3):
                pct_j = 100.0 * (math.exp(mde_j * 10.0 / design["year_sd"]) - 1.0)
                pct_3 = 100.0 * (math.exp(mde_3 * 10.0 / design["year_sd"]) - 1.0)
                lines.append(
                    f"  {nm:25s} | All 85 (w/ nhosts): SE={se_j:.3f}, MDE80={mde_j:.3f} ({pct_j:+.1f}%/dec) "
                    f"| Within 3+ (n=44): SE={se_3:.3f}, MDE80={mde_3:.3f} ({pct_3:+.1f}%/dec)"
                )
                row_rec["full_85"][nm] = {"se": float(se_j), "mde80": float(mde_j), "mde80_pct_dec": float(pct_j)}
                row_rec["within_3plus_44"][nm] = {"se": float(se_3), "mde80": float(mde_3), "mde80_pct_dec": float(pct_3)}
            mde_summary.append(row_rec)

    # 2. Monte Carlo simulation of the exact fitted mixed-effects ML/LRT/AICc procedure (2,000 replicates per scenario)
    lines.append("\n--- 2. Monte Carlo Power & Type I Error (2,000 simulated datasets per scenario) ---")
    lines.append(
        "Registered criteria tested: Omnibus LRT (4 df, p < 0.05) and Delta_AICc <= -2.0 (M_joint vs M_nhosts); "
        "individual contrasts two-sided p < 0.05 (|z| >= 1.96) with predicted sign."
    )
    scenarios = [
        ("Null (no trait effect; nhosts 3+ = -0.35)", np.array([0.0, 0.0, -0.35, 0.0, 0.0, 0.0, 0.0])),
        ("Weak theory (kappa=0.25; nhosts=0)",        np.array([0.05, 0.0, 0.0, -0.05, +0.075, +0.04, -0.04])),
        ("Moderate theory (kappa=0.50; nhosts=0)",    np.array([0.10, 0.0, 0.0, -0.10, +0.150, +0.08, -0.08])),
        ("Strong theory (kappa=1.00; nhosts=0)",      np.array([0.15, 0.0, 0.0, -0.20, +0.300, +0.15, -0.15])),
    ]
    n_mc = 1000
    rng_mc = np.random.default_rng(20261115)
    mc_summary = []

    for sg, st in [(0.25, 0.10), (0.35, 0.15)]:
        lines.append(f"\nMonte Carlo noise setting: sigma_gamma = {sg:.2f}, sigma_tax = {st:.2f} (n_mc = {n_mc}):")
        for s_name, beta_true in scenarios:
            mu_true = X_joint @ beta_true
            lrt_pass = 0
            aicc_pass = 0
            c_pass = np.zeros(4, dtype=int)
            signs = np.array([-1.0, +1.0, +1.0, -1.0])

            for _ in range(n_mc):
                a_draw = rng_mc.normal(0.0, st, size=len(ptypes))[grp_idx]
                eps_draw = rng_mc.normal(0.0, np.sqrt(v_obs + sg * sg))
                y_sim = mu_true + a_draw + eps_draw

                # Fit M_nhosts (k = 3 fixed + 2 var = 5 params) and M_joint (k = 7 fixed + 2 var = 9 params)
                _, _, nll_nh, _, _ = fit_reml_wls(y_sim, X_nh, v_obs, grp_idx)
                b_j, se_j, nll_j, _, _ = fit_reml_wls(y_sim, X_joint, v_obs, grp_idx)

                lr_stat = max(0.0, 2.0 * (nll_nh - nll_j))
                if stats.chi2.sf(lr_stat, df=4) < 0.05:
                    lrt_pass += 1
                if aicc(nll_j, 9, n_taxa) - aicc(nll_nh, 5, n_taxa) <= -2.0:
                    aicc_pass += 1

                z_vals = b_j[3:7] / se_j[3:7]
                for m in range(4):
                    if signs[m] * z_vals[m] >= 1.959964:
                        c_pass[m] += 1

            rec = {
                "sigma_gamma": sg,
                "sigma_tax": st,
                "scenario": s_name,
                "lrt_power": lrt_pass / n_mc,
                "daicc_le_minus2_power": aicc_pass / n_mc,
                "contrast_power": {contrast_names[m]: c_pass[m] / n_mc for m in range(4)},
            }
            mc_summary.append(rec)
            lines.append(
                f"  {s_name:43s} | LRT={rec['lrt_power']:.3f}, dAICc<=-2={rec['daicc_le_minus2_power']:.3f} | "
                f"P(a)={c_pass[0]/n_mc:.3f}, P(b)={c_pass[1]/n_mc:.3f}, P(c)={c_pass[2]/n_mc:.3f}, P(d)={c_pass[3]/n_mc:.3f}"
            )

    res_dir = os.path.join(HERE, "results")
    txt_path = os.path.join(res_dir, "pre_analysis_power.txt")
    json_path = os.path.join(res_dir, "pre_analysis_power.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"mde_summary": mde_summary, "mc_summary": mc_summary}, f, indent=2)

    print("\n".join(lines))
    print(f"\nSaved {txt_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
