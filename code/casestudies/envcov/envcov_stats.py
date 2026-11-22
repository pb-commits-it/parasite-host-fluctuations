"""Biological series readers (reproducing R12, R12b, R12c, R14) and statistical helpers for envcov."""
from __future__ import annotations

from collections import defaultdict
import csv
import itertools
import math
from pathlib import Path
import re

import numpy as np

PERMS6 = np.array(list(itertools.permutations(range(6))), dtype=np.int64)
R14_SITES13 = "BLAN DELA HARV KONZ MLBS ORNL OSBS SCBI SERC STEI TALL TREE UKFS".split()
R14_INCLUDED6 = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]


def midranks(a: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def pearson_r(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3:
        return float("nan")
    xm, ym = xa[m], ya[m]
    if xm.std() == 0 or ym.std() == 0:
        return float("nan")
    return float(np.corrcoef(xm, ym)[0, 1])


def spearman_rho(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3:
        return float("nan")
    return pearson_r(midranks(xa[m]), midranks(ya[m]))


def spearman_perm_exact6(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> tuple[float, float, float]:
    """Exact 720-permutation Spearman rho, P(rho* <= rho), P(rho* >= rho) for n = 6."""
    rx = midranks(x) - 3.5
    ry = midranks(y) - 3.5
    den = math.sqrt(float((rx @ rx) * (ry @ ry)))
    if den == 0:
        return float("nan"), float("nan"), float("nan")
    rho = float((rx @ ry) / den)
    vals = (ry[PERMS6] @ rx) / den
    return rho, float(np.mean(vals <= rho + 1e-12)), float(np.mean(vals >= rho - 1e-12))


def freedman_lane_partial6(
    y: np.ndarray | list[float],
    x: np.ndarray | list[float],
    z: np.ndarray | list[float],
    use_ranks: bool = True,
) -> tuple[float, float, float]:
    """Exact 720-permutation Freedman-Lane partial correlation r(Y, X | Z) and 1-sided p-values (n = 6)."""
    vy = midranks(y) if use_ranks else np.asarray(y, dtype=float).copy()
    vx = midranks(x) if use_ranks else np.asarray(x, dtype=float).copy()
    vz = midranks(z) if use_ranks else np.asarray(z, dtype=float).copy()
    vy -= vy.mean()
    vx -= vx.mean()
    vz -= vz.mean()
    uz = vz / max(math.sqrt(float(vz @ vz)), 1e-15)

    ex = vx - float(vx @ uz) * uz
    ex_norm = math.sqrt(float(ex @ ex))
    ey = vy - float(vy @ uz) * uz
    ey_norm = math.sqrt(float(ey @ ey))
    if ex_norm < 1e-14 or ey_norm < 1e-14:
        return float("nan"), float("nan"), float("nan")
    ex /= ex_norm
    ey /= ey_norm
    r_obs = float(ex @ ey)

    ey_perm = ey[PERMS6]
    num = ey_perm @ ex
    proj_z = ey_perm @ uz
    den = np.sqrt(np.maximum(1.0 - proj_z * proj_z, 1e-15))
    r_perm = num / den
    p_neg = float(np.mean(r_perm <= r_obs + 1e-12))
    p_pos = float(np.mean(r_perm >= r_obs - 1e-12))
    return r_obs, p_neg, p_pos


def logit_wls_6sites(
    nip: np.ndarray,
    ntest: np.ndarray,
    sigma: np.ndarray,
    z_cov: np.ndarray | None = None,
) -> dict[str, float]:
    """Binomial delta-method weighted least squares of logit(NIP) on sigma (+ optional 1 covariate)."""
    p = np.clip(np.asarray(nip, dtype=float), 1e-4, 1.0 - 1e-4)
    n = np.asarray(ntest, dtype=float)
    y = np.log(p / (1.0 - p))
    w = n * p * (1.0 - p)
    cols = [np.ones(len(p)), np.asarray(sigma, dtype=float)]
    if z_cov is not None:
        zc = np.asarray(z_cov, dtype=float)
        cols.append((zc - zc.mean()) / max(float(zc.std(ddof=1)), 1e-15))
    xmat = np.column_stack(cols)
    sw = np.sqrt(w)
    xw = xmat * sw[:, None]
    yw = y * sw
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    resid = y - xmat @ beta
    df = len(p) - xmat.shape[1]
    # Quasi-binomial / overdispersion-adjusted WLS covariance
    phi_disp = max(1.0, float(np.sum(w * (resid ** 2))) / max(df, 1))
    cov = phi_disp * np.linalg.inv(xw.T @ xw)
    se_sig = math.sqrt(float(cov[1, 1]))
    return {
        "beta_sigma": float(beta[1]),
        "se_sigma": se_sig,
        "t_sigma": float(beta[1] / max(se_sig, 1e-15)),
        "beta_z": float(beta[2]) if z_cov is not None else float("nan"),
        "se_z": math.sqrt(float(cov[2, 2])) if z_cov is not None else float("nan"),
        "df": int(df),
        "overdispersion": float(phi_disp),
    }


def fe_ols_with_covariates(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> tuple[float, np.ndarray, float, int]:
    """Fixed-effects OLS of y on x + Z with one intercept per group.

    Each element of `groups` is (x_g, y_g, Z_g), where Z_g is None or shape (n_g, p).
    Returns (b_x, beta_z, se_ols_x, df_resid).
    """
    xd_list, yd_list, zd_list = [], [], []
    n_tot = k_groups = 0
    p_cov = 0
    for xg, yg, zg in groups:
        if len(xg) < 2:
            continue
        k_groups += 1
        n_tot += len(xg)
        xd_list.append(xg - xg.mean())
        yd_list.append(yg - yg.mean())
        if zg is not None and zg.size > 0:
            p_cov = zg.shape[1]
            zd_list.append(zg - zg.mean(axis=0, keepdims=True))
    xd = np.concatenate(xd_list)
    yd = np.concatenate(yd_list)
    if p_cov == 0:
        sxx = float(xd @ xd)
        if sxx <= 1e-15:
            return float("nan"), np.zeros(0), float("nan"), 0
        bx = float((xd @ yd) / sxx)
        resid = yd - bx * xd
        df = n_tot - k_groups - 1
        se_ols = math.sqrt(float(resid @ resid) / max(df, 1) / sxx)
        return bx, np.zeros(0), se_ols, df
    zd = np.vstack(zd_list)
    wmat = np.column_stack([xd, zd])
    coef = np.linalg.lstsq(wmat, yd, rcond=None)[0]
    resid = yd - wmat @ coef
    df = n_tot - k_groups - 1 - p_cov
    xtwx_inv = np.linalg.pinv(wmat.T @ wmat)
    se_ols = math.sqrt(max(0.0, float(resid @ resid) / max(df, 1) * float(xtwx_inv[0, 0])))
    return float(coef[0]), coef[1:], se_ols, df


def fe_ols_dummies_check(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> float:
    """Explicit dummy-variable least squares check for Rule 7 verification."""
    used = [(xg, yg, zg) for (xg, yg, zg) in groups if len(xg) >= 2]
    k = len(used)
    rows, resp = [], []
    for g_idx, (xg, yg, zg) in enumerate(used):
        for i in range(len(xg)):
            z_row = list(zg[i]) if (zg is not None and zg.size > 0) else []
            dum = [1.0 if j == g_idx else 0.0 for j in range(k)]
            rows.append([float(xg[i])] + z_row + dum)
            resp.append(float(yg[i]))
    coef = np.linalg.lstsq(np.array(rows), np.array(resp), rcond=None)[0]
    return float(coef[0])


def cary_block_boot_slope(
    pair_records: list[dict],
    plots: list[str],
    cov_names: list[str],
    starts: np.ndarray,
    t0: int,
    t1: int,
) -> tuple[float, np.ndarray, float, tuple[float, float], int]:
    """Compute within-plot slope on Cary pairs + 2,000 2-yr block bootstrap SE (exact R12 blocks)."""
    # Build per-plot arrays for point estimate
    groups = []
    for p in plots:
        recs = [r for r in pair_records if r["plot"] == p]
        xg = np.array([r["x"] for r in recs], dtype=float)
        yg = np.array([r["y"] for r in recs], dtype=float)
        zg = np.array([[r[c] for c in cov_names] for r in recs], dtype=float) if cov_names else None
        groups.append((xg, yg, zg))
    bx, bz, _se_ols, df_res = fe_ols_with_covariates(groups)

    # Build year-indexed arrays [plot, year_offset] for exact R12 block resampling
    n_years = t1 - t0 + 1
    p_cov = len(cov_names)
    x_grid = np.full((len(plots), n_years), np.nan)
    y_grid = np.full((len(plots), n_years), np.nan)
    z_grid = np.full((len(plots), n_years, p_cov), np.nan) if p_cov else None
    p_idx = {p: i for i, p in enumerate(plots)}
    for r in pair_records:
        pi = p_idx[r["plot"]]
        yi = r["mouse_year"] - t0
        x_grid[pi, yi] = r["x"]
        y_grid[pi, yi] = r["y"]
        if p_cov:
            for ci, cname in enumerate(cov_names):
                z_grid[pi, yi, ci] = r[cname]

    n_boot = starts.shape[0]
    boots = np.full(n_boot, np.nan)
    draw_idx = np.concatenate([starts, starts + 1], axis=1)  # (NBOOT, 2 * n_blocks)
    for b in range(n_boot):
        cols = draw_idx[b]
        bgroups = []
        for pi in range(len(plots)):
            xv = x_grid[pi, cols]
            yv = y_grid[pi, cols]
            m = np.isfinite(xv) & np.isfinite(yv)
            if p_cov:
                zv = z_grid[pi, cols, :]
                m &= np.all(np.isfinite(zv), axis=1)
                if m.sum() >= 2:
                    bgroups.append((xv[m], yv[m], zv[m]))
            else:
                if m.sum() >= 2:
                    bgroups.append((xv[m], yv[m], None))
        if not bgroups:
            continue
        try:
            b_val, _, _, _ = fe_ols_with_covariates(bgroups)
            if math.isfinite(b_val):
                boots[b] = b_val
        except np.linalg.LinAlgError:
            continue

    ok = boots[np.isfinite(boots)]
    se = float(np.std(ok, ddof=1))
    ci = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))
    return bx, bz, se, ci, df_res


def neon_t3_boot_slope(
    site_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    n_boot: int = 2000,
    seed: list[int] | int = (20261114, 3),
) -> tuple[float, np.ndarray, float, tuple[float, float], float, int]:
    """Pooled within-site slope on NEON pairs + 2,000 within-site resamples (matching R14 T3)."""
    bx, bz, se_ols, df_res = fe_ols_with_covariates(site_groups)
    rng = np.random.default_rng(list(seed) if isinstance(seed, tuple) else seed)
    boots = []
    for _ in range(n_boot):
        bg = []
        for xg, yg, zg in site_groups:
            idx = rng.integers(0, len(xg), len(xg))
            xs, ys = xg[idx], yg[idx]
            zs = zg[idx] if zg is not None else None
            bg.append((xs, ys, zs))
        try:
            b_val, _, _, _ = fe_ols_with_covariates(bg)
            if math.isfinite(b_val):
                boots.append(b_val)
        except np.linalg.LinAlgError:
            continue
    arr = np.array(boots, dtype=float)
    se = float(np.std(arr, ddof=1))
    ci = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return bx, bz, se, ci, se_ols, df_res


def block_boot_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
) -> tuple[float, tuple[float, float], float]:
    """Pearson r, 2-year block bootstrap 95% CI, and Spearman rho for aligned annual series."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    r_obs = pearson_r(xa, ya)
    rho_obs = spearman_rho(xa, ya)
    cols = np.concatenate([starts, starts + 1], axis=1)
    vals = []
    for b in range(len(starts)):
        xb = xa[cols[b]]
        yb = ya[cols[b]]
        val = pearson_r(xb, yb)
        if math.isfinite(val):
            vals.append(val)
    arr = np.array(vals, dtype=float)
    ci = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) >= 10 else (float("nan"), float("nan"))
    return r_obs, ci, rho_obs


def load_cary_series(cary_raw: Path) -> dict:
    """Load Cary Dryad (1991-2015), Figshare long (1991-2025), and Holdout (2014-2022) series."""
    dry_path = cary_raw / "dryad_d1c8046_forest_food_web_2018" / "Lyme_Study_Dataset.csv"
    fig_path = cary_raw / "figshare_31872853_ecology_of_lyme_2026" / "Full_dataset_20260326.csv"
    hol_path = cary_raw / "figshare_27871269_nymphal_infection_2025" / "allthedata_grid.csv"

    with open(dry_path, newline="", encoding="latin-1") as f:
        dry_rows = [r for r in csv.DictReader(f) if r["Grid"] not in ("NA", "")]
    with open(fig_path, newline="", encoding="latin-1") as f:
        fig_rows = [r for r in csv.DictReader(f) if r["Plot"] not in ("NA", "")]
    with open(hol_path, newline="", encoding="latin-1") as f:
        hol_rows = [r for r in csv.DictReader(f) if r["Grid"] not in ("NA", "")]

    def col_by_plot(rows: list[dict], pcol: str, vcol: str) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for r in rows:
            p = r[pcol]
            out.setdefault(p, {})
            v = r.get(vcol, "")
            if v not in ("NA", "", None):
                out[p][int(r["Year"])] = float(v)
        return out

    dplots = sorted({r["Grid"] for r in dry_rows})
    fplots = sorted({r["Plot"] for r in fig_rows})
    hplots = sorted({r["Grid"] for r in hol_rows})

    # Landscape station weather columns in Dryad file (identical across plots within year)
    station_dry: dict[int, dict[str, float]] = {}
    for r in dry_rows:
        y = int(r["Year"])
        station_dry.setdefault(y, {})
        for c in ("mat", "degree_days", "cold_winter_days", "feb_apr_warm_days", "snowfall_cm", "Ave.Deer.Hr"):
            v = r.get(c, "")
            if v not in ("NA", "", None):
                station_dry[y][c] = float(v)

    # Exact PCR assay weight in [0, 1] for every NIP plot-year (0 = TFA only, 1 = PCR only, PCR_fraction = mixed)
    dry_pcr_share: dict[str, dict[int, float]] = {}
    for r in dry_rows:
        p = r["Grid"]
        y = int(r["Year"])
        dry_pcr_share.setdefault(p, {})
        has_tfa = r.get("TFA_NIP_CORR", "") not in ("NA", "", None)
        has_pcr = r.get("PCR_NIP", "") not in ("NA", "", None)
        pf = r.get("PCR_fraction", "")
        if pf not in ("NA", "", None):
            dry_pcr_share[p][y] = float(pf)
        elif has_pcr and not has_tfa:
            dry_pcr_share[p][y] = 1.0
        elif has_tfa and not has_pcr:
            dry_pcr_share[p][y] = 0.0

    return {
        "dplots": dplots,
        "fplots": fplots,
        "hplots": hplots,
        "dry_mice": col_by_plot(dry_rows, "Grid", "PL_grid"),
        "dry_chip": col_by_plot(dry_rows, "Grid", "TS_grid"),
        "dry_nip": col_by_plot(dry_rows, "Grid", "NIP"),
        "dry_don": col_by_plot(dry_rows, "Grid", "DON"),
        "dry_lpm": col_by_plot(dry_rows, "Grid", "LPM"),
        "dry_acorns": col_by_plot(dry_rows, "Grid", "TOT.MATURE.ACORNS"),
        "dry_pcr_frac": col_by_plot(dry_rows, "Grid", "PCR_fraction"),
        "dry_pcr_share": dry_pcr_share,
        "fig_mice": col_by_plot(fig_rows, "Plot", "PL_t"),
        "hol_n": col_by_plot(hol_rows, "Grid", "n"),
        "hol_bb": col_by_plot(hol_rows, "Grid", "Bburgd"),
        "hol_mice_prev": col_by_plot(hol_rows, "Grid", "PL.mnRawLyr"),
        "station_dry": station_dry,
    }


def load_neon_r14_series(neon_raw: Path) -> dict:
    """Recompute R14 mouse indices, sigma_site, r1, and nymphal B. burgdorferi s.l. counts."""
    name_re = re.compile(r"^NEON\.D\d\d\.([A-Z]{4})\.(DP1\.\d{5}\.\d{3})\.([A-Za-z_]+)\.(\d{4}-\d{2})\.basic\.(\d{8}T\d{6}Z)\.csv$")

    def latest_files(product: str, table: str) -> list[Path]:
        chosen = {}
        for s in R14_SITES13:
            d = neon_raw / product / s
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.csv")):
                m = name_re.match(p.name)
                if m and m.group(3) == table and m.group(1) == s:
                    key = (s, m.group(4))
                    if key not in chosen or m.group(5) > chosen[key][0]:
                        chosen[key] = (m.group(5), p)
        return [v[1] for k, v in sorted(chosen.items())]

    seen_mam = set()
    traps, tags, plot_lat = defaultdict(set), defaultdict(set), {}
    set_codes = {"2", "3", "4", "5", "6"}
    for path in latest_files("DP1.10072.001", "mam_pertrapnight"):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["uid"] in seen_mam:
                    continue
                seen_mam.add(r["uid"])
                date = r["collectDate"][:10]
                if not date[:4].isdigit():
                    continue
                s = r["siteID"]
                y = int(date[:4])
                code = r["trapStatus"].strip()[:1]
                if code in set_codes:
                    traps[(s, y)].add((r["plotID"], date, r["trapCoordinate"].strip()))
                w = r["scientificName"].split()
                if w and w[0] == "Peromyscus":
                    tag = r["tagID"].strip()
                    if tag:
                        tags[(s, y)].add(tag)
                lat = (r.get("decimalLatitude") or "").strip()
                if lat and (s, r["plotID"]) not in plot_lat:
                    plot_lat[(s, r["plotID"])] = float(lat)

    effort = {k: len(v) for k, v in traps.items()}
    index = {k: 100.0 * len(tags.get(k, ())) / tn for k, tn in effort.items() if tn >= 500}
    sigma, r1, mean_index, mean_log_tn = {}, {}, {}, {}
    for s in R14_SITES13:
        ys = sorted(y for (ss, y) in index if ss == s and index[(s, y)] > 0)
        ln_vals = {y: math.log(index[(s, y)]) for y in ys}
        sigma[s] = float(np.std(list(ln_vals.values()), ddof=1)) if len(ln_vals) >= 5 else None
        pairs = [(ln_vals[y], ln_vals[y + 1]) for y in ys if y + 1 in ln_vals]
        r1[s] = float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1]) if len(pairs) >= 3 else None
        mean_index[s] = float(np.mean([index[(s, y)] for y in ys])) if ys else float("nan")
        mean_log_tn[s] = float(np.mean([math.log(effort[(s, y)]) for y in ys])) if ys else float("nan")

    # Ticks & pathogens
    tax = {}
    seen_tax = set()
    for path in latest_files("DP1.10093.001", "tck_taxonomyProcessed"):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["uid"] in seen_tax:
                    continue
                seen_tax.add(r["uid"])
                sub = r["subsampleID"].strip()
                tax.setdefault(sub, (r["acceptedTaxonID"].strip(), r["sexOrAge"].strip()))

    ticks = {}
    seen_pat = set()
    for path in latest_files("DP1.10092.001", "tck_pathogen"):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["uid"] in seen_pat:
                    continue
                seen_pat.add(r["uid"])
                if not r["collectDate"][:4].isdigit():
                    continue
                tid = r["testingID"].strip()
                t = ticks.setdefault(
                    tid,
                    {"site": r["siteID"], "year": int(r["collectDate"][:4]), "sub": r["subsampleID"].strip(), "res": defaultdict(set)},
                )
                if r["testResult"] in ("Positive", "Negative"):
                    t["res"][r["testPathogenName"]].add(r["testResult"])

    sl_name = "Borrelia burgdorferi sensu lato"
    cells = defaultdict(lambda: [0, 0])
    for t in ticks.values():
        taxon, stage = tax.get(t["sub"], ("?", "?"))
        if taxon != "IXOSCA" or stage != "Nymph":
            continue
        res = t["res"].get(sl_name)
        if not res:
            continue
        dq = t["res"].get("HardTick DNA Quality")
        if dq is not None and "Positive" not in dq:
            continue
        c = cells[(t["site"], t["year"])]
        c[0] += 1
        c[1] += 1 if "Positive" in res else 0

    tested, positive, nip_site, test_years, yearly_nip = {}, {}, {}, {}, {}
    for s in R14_SITES13:
        ys = sorted(y for (ss, y) in cells if ss == s and cells[(s, y)][0] > 0)
        tot_n = sum(cells[(s, y)][0] for y in ys)
        tot_k = sum(cells[(s, y)][1] for y in ys)
        tested[s], positive[s] = tot_n, tot_k
        nip_site[s] = tot_k / tot_n if tot_n > 0 else None
        test_years[s] = ys
        yearly_nip[s] = {y: cells[(s, y)][1] / cells[(s, y)][0] for y in ys if cells[(s, y)][0] >= 20}

    lat = {s: float(np.mean([v for (ss, _), v in plot_lat.items() if ss == s])) for s in R14_SITES13}
    return {
        "effort": effort,
        "index": index,
        "sigma": sigma,
        "r1": r1,
        "mean_index": mean_index,
        "mean_log_tn": mean_log_tn,
        "cells": dict(cells),
        "tested": tested,
        "positive": positive,
        "nip_site": nip_site,
        "test_years": test_years,
        "yearly_nip": yearly_nip,
        "lat": lat,
    }
