#!/usr/bin/env python3
"""High-precision 20,000-study Monte Carlo verification of Part 4 NEON coupled masking/mimicking rates."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import special

import data_and_covariates as dc
import lifecycle_solver as ls
import stats_methods as sm


def main() -> None:
    rep = json.loads((HERE / "replication_results.json").read_text(encoding="utf-8"))
    neon_bio = dc.load_neon_biological(HERE.parents[4] / "data" / "raw" / "neon")
    counts_by_site = {s: [neon_bio["cells"][(s, y)][0] for y in neon_bio["test_years"][s]] for s in dc.NEON_6}

    neon_sites = rep["neon_sites"]
    sigma_6 = np.array([r["sigma"] for r in neon_sites], dtype=float)
    nip_6 = np.array([r["nip"] for r in neon_sites], dtype=float)
    clim_6 = np.array([r["PC1_env"] for r in neon_sites], dtype=float)
    base_m = -math.log1p(-float(np.mean(nip_6))) / 0.5
    coupled_cells = rep["coupled_weather_cells"]

    # Recompute phi_weather per site from raw data
    daymet_all, cov_all = {}, {}
    raw_env = (__import__('pathlib').Path(__file__).resolve().parent / '../../../data/raw/envcov').resolve()
    for s in dc.NEON_6:
        dm, _ = dc.parse_daymet_csv(raw_env / "daymet" / f"{s}_tick.csv")
        pw = dc.parse_nasa_power_json(raw_env / "nasa_power" / f"{s}_tick.json")
        cov_all[s] = dc.compute_annual_covariates(dm, pw)
    neon_pca = sm.fit_pca1_corrmat(np.array([[r[k] for k in dc.PRIMARY_COVS] for r in neon_sites], dtype=float), orient_col=0)
    phi_w_map = {}
    for s in dc.NEON_6:
        yrs = sorted(y for (ss, y), ix in neon_bio["index"].items() if ss == s and ix > 0.0 and (y + 1) in cov_all[s])
        wm = np.array([sm.project_pca1([cov_all[s][y + 1][k] for k in dc.PRIMARY_COVS], neon_pca) for y in yrs], dtype=float)
        adj = np.diff(np.array(yrs)) == 1
        phi_w_map[s] = sm.pearson_corr(wm[:-1][adj], wm[1:][adj])

    n_mc = 20000
    rows_20k = []
    for gam in (-0.2, 0.0, 0.2):
        for kap in (0.0, 0.5, 1.0):
            idx_s = len(rows_20k)
            rng = np.random.default_rng(950000 + idx_s)
            cells_gk = [next(c for c in coupled_cells if c["site"] == s and c["gamma"] == gam and c["kappa"] == kap) for s in dc.NEON_6]
            d_vec = np.array([c["D"] for c in cells_gk], dtype=float)
            ln_r_mean = np.log([c["mean_R0"] for c in cells_gk])
            margin = base_m - float(np.mean(ln_r_mean - d_vec))
            site_res = 0.17 * rng.normal(size=(n_mc, 6))
            phat = np.zeros((n_mc, 6), dtype=float)

            for i, s in enumerate(dc.NEON_6):
                ns_yr = np.asarray(counts_by_site[s], dtype=int)
                ny = len(ns_yr)
                st_ln_r = np.log(np.asarray(cells_gk[i]["state_R0"], dtype=float))
                wz = ls.sample_gaussian_ar1(
                    np.array([1.0]),
                    np.array([phi_w_map[s]]),
                    np.ones((1, 1)),
                    ny,
                    n_mc,
                    960000 + idx_s * 10 + i,
                    antithetic=False,
                    standardized=True,
                )[:, :, 0].T
                bins = np.minimum((special.ndtr(wz) * len(st_ln_r)).astype(int), len(st_ln_r) - 1)
                ann_ln_r = margin + site_res[:, i, None] + st_ln_r[bins] - d_vec[i]
                p_yr = 1.0 - np.exp(-0.5 * np.maximum(ann_ln_r, 0.0))
                k_pos = rng.binomial(ns_yr[None, :], p_yr)
                phat[:, i] = k_pos.sum(axis=1) / float(ns_yr.sum())

            rho_all, rpart_all, padj_all = [], [], []
            for start in range(0, n_mc, 2000):
                t = sm.batch_rank_tests_6(phat[start : start + 2000], sigma_6, clim_6)
                rho_all.append(t["rho"])
                rpart_all.append(t["r_part"])
                padj_all.append(t["p_adj_neg"])
            rho = np.concatenate(rho_all)
            rpart = np.concatenate(rpart_all)
            padj = np.concatenate(padj_all)

            ev_neg = sm.wilson_interval(int(np.sum(rho < 0.0)), n_mc)
            ev_sup = sm.wilson_interval(int(np.sum(rho <= -29.0 / 35.0 + 1e-12)), n_mc)
            ev_pos = sm.wilson_interval(int(np.sum(rho >= 17.0 / 35.0 - 1e-12)), n_mc)
            ev_adj = sm.wilson_interval(int(np.sum((rpart < 0.0) & (padj < 0.05))), n_mc)
            rows_20k.append({
                "gamma": gam,
                "kappa": kap,
                "n": n_mc,
                "negative_unadjusted": ev_neg["freq"],
                "support_cutoff": ev_sup["freq"],
                "support_cutoff_wilson_low": ev_sup["wilson_low"],
                "support_cutoff_wilson_high": ev_sup["wilson_high"],
                "as_positive_as_R14": ev_pos["freq"],
                "support_adjusted": ev_adj["freq"],
                "support_adjusted_wilson_low": ev_adj["wilson_low"],
                "support_adjusted_wilson_high": ev_adj["wilson_high"],
            })

    (HERE / "driven_weather_20k_check.json").write_text(json.dumps(rows_20k, indent=2) + "\n", encoding="utf-8")
    for r in rows_20k:
        print(
            f"gamma={r['gamma']:+.1f} kappa={r['kappa']:.1f} | neg={100*r['negative_unadjusted']:.2f}% "
            f"sup={100*r['support_cutoff']:.2f}% [{100*r['support_cutoff_wilson_low']:.2f}, {100*r['support_cutoff_wilson_high']:.2f}] "
            f"pos49={100*r['as_positive_as_R14']:.2f}% adj_sup={100*r['support_adjusted']:.2f}%"
        )


if __name__ == "__main__":
    main()
