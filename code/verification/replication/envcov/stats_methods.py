"""Independent statistical estimators, permutation tests, PCA, and block bootstraps for envcov replication."""
from __future__ import annotations

import itertools
import math
import numpy as np

ALL_PERMS_6 = np.array(list(itertools.permutations(range(6))), dtype=np.int64)


def compute_midranks(values: np.ndarray | list[float]) -> np.ndarray:
    """1-based midranks with exact tie averaging."""
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    idx = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and arr[idx[end]] == arr[idx[pos]]:
            end += 1
        ranks[idx[pos:end]] = 0.5 * (pos + end - 1) + 1.0
        pos = end
    return ranks


def compute_midranks_2d(mat: np.ndarray) -> np.ndarray:
    """Row-wise 1-based midranks for a 2D array (B, n)."""
    arr = np.asarray(mat, dtype=float)
    b, n = arr.shape
    idx = np.argsort(arr, axis=1, kind="mergesort")
    sorted_vals = np.take_along_axis(arr, idx, axis=1)
    out = np.empty_like(arr)
    for r in range(b):
        row = sorted_vals[r]
        ranks = np.empty(n, dtype=float)
        pos = 0
        while pos < n:
            end = pos + 1
            while end < n and row[end] == row[pos]:
                end += 1
            ranks[pos:end] = 0.5 * (pos + end - 1) + 1.0
            pos = end
        out[r, idx[r]] = ranks
    return out


def pearson_corr(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if int(keep.sum()) < 3:
        return float("nan")
    xc = xa[keep] - xa[keep].mean()
    yc = ya[keep] - ya[keep].mean()
    denom = math.sqrt(float((xc @ xc) * (yc @ yc)))
    if denom <= 0.0:
        return float("nan")
    return float((xc @ yc) / denom)


def spearman_corr(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if int(keep.sum()) < 3:
        return float("nan")
    return pearson_corr(compute_midranks(xa[keep]), compute_midranks(ya[keep]))


def exact_spearman_test_6(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict[str, float]:
    """Exhaustive 720-permutation Spearman rank test for N = 6."""
    rx = compute_midranks(x)
    ry = compute_midranks(y)
    cx = rx - rx.mean()
    cy = ry - ry.mean()
    denom = math.sqrt(float((cx @ cx) * (cy @ cy)))
    if denom <= 0.0:
        return {"rho": float("nan"), "p_neg": float("nan"), "p_pos": float("nan")}
    rho_obs = float((cx @ cy) / denom)
    perm_rhos = (cy[ALL_PERMS_6] @ cx) / denom
    p_neg = float(np.mean(perm_rhos <= rho_obs + 1e-12))
    p_pos = float(np.mean(perm_rhos >= rho_obs - 1e-12))
    return {"rho": rho_obs, "p_neg": p_neg, "p_pos": p_pos}


def freedman_lane_partial_test_6(
    y: np.ndarray | list[float],
    x: np.ndarray | list[float],
    z: np.ndarray | list[float],
    rank_transform: bool = True,
) -> dict[str, float]:
    """Exhaustive 720-permutation Freedman-Lane partial correlation test r(Y, X | Z) for N = 6."""
    vy = compute_midranks(y) if rank_transform else np.asarray(y, dtype=float).copy()
    vx = compute_midranks(x) if rank_transform else np.asarray(x, dtype=float).copy()
    vz = compute_midranks(z) if rank_transform else np.asarray(z, dtype=float).copy()

    cy = vy - vy.mean()
    cx = vx - vx.mean()
    cz = vz - vz.mean()
    nz = math.sqrt(float(cz @ cz))
    if nz <= 1e-15:
        return {"r_part": float("nan"), "p_neg": float("nan"), "p_pos": float("nan")}
    uz = cz / nz

    rx = cx - float(cx @ uz) * uz
    ry = cy - float(cy @ uz) * uz
    nx = math.sqrt(float(rx @ rx))
    ny = math.sqrt(float(ry @ ry))
    if nx <= 1e-14 or ny <= 1e-14:
        return {"r_part": float("nan"), "p_neg": float("nan"), "p_pos": float("nan")}
    ux = rx / nx
    uy = ry / ny
    r_obs = float(ux @ uy)

    # Under Freedman-Lane, reduced model is Y = a + g Z + e_Y; permute e_Y -> Y* = g_hat Z + P e_Y.
    # Re-residualizing Y* on [1, Z] equals projecting P u_y orthogonal to u_z, with norm sqrt(1 - (P u_y . u_z)^2).
    uy_perm = uy[ALL_PERMS_6]
    num = uy_perm @ ux
    proj_uz = uy_perm @ uz
    denom = np.sqrt(np.maximum(1.0 - proj_uz * proj_uz, 1e-15))
    r_perm = num / denom
    p_neg = float(np.mean(r_perm <= r_obs + 1e-12))
    p_pos = float(np.mean(r_perm >= r_obs - 1e-12))
    return {"r_part": r_obs, "p_neg": p_neg, "p_pos": p_pos}


def batch_rank_tests_6(y_mat: np.ndarray, x_vec: np.ndarray, z_vec: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized 720-permutation Spearman and Freedman-Lane partial Spearman across B studies."""
    rx = compute_midranks(x_vec)
    rx -= rx.mean()
    nx = math.sqrt(float(rx @ rx))

    rz = compute_midranks(z_vec)
    rz -= rz.mean()
    uz = rz / math.sqrt(float(rz @ rz))

    ry = compute_midranks_2d(y_mat)
    ry -= ry.mean(axis=1, keepdims=True)
    ny = np.sqrt((ry * ry).sum(axis=1))
    valid = ny > 1e-12

    denom_un = ny * nx
    rho = np.full(len(y_mat), np.nan)
    rho[valid] = (ry[valid] @ rx) / denom_un[valid]
    perm_un = np.einsum("bpj,j->bp", ry[:, ALL_PERMS_6], rx) / np.maximum(denom_un[:, None], 1e-15)
    p_un_neg = np.mean(perm_un <= rho[:, None] + 1e-12, axis=1)

    ex = rx - float(rx @ uz) * uz
    ux = ex / math.sqrt(float(ex @ ex))
    ey = ry - (ry @ uz)[:, None] * uz
    ney = np.sqrt((ey * ey).sum(axis=1))
    valid_part = ney > 1e-12
    uy = ey / np.maximum(ney[:, None], 1e-15)
    r_part = uy @ ux
    r_part[~valid_part] = np.nan

    uy_perm = uy[:, ALL_PERMS_6]
    num = np.einsum("bpj,j->bp", uy_perm, ux)
    proj = np.einsum("bpj,j->bp", uy_perm, uz)
    r_perm = num / np.sqrt(np.maximum(1.0 - proj * proj, 1e-15))
    p_adj_neg = np.mean(r_perm <= r_part[:, None] + 1e-12, axis=1)

    return {
        "rho": rho,
        "p_un_neg": p_un_neg,
        "r_part": r_part,
        "p_adj_neg": p_adj_neg,
        "valid": valid,
    }


def fit_pca1_corrmat(data_matrix: np.ndarray, orient_col: int = 0) -> dict:
    """First principal component via eigendecomposition of the sample correlation matrix."""
    x = np.asarray(data_matrix, dtype=float)
    n, p = x.shape
    col_mean = x.mean(axis=0)
    col_sd = x.std(axis=0, ddof=1)
    z = (x - col_mean[None, :]) / col_sd[None, :]
    corr_mat = (z.T @ z) / float(n - 1)
    evals, evecs = np.linalg.eigh(corr_mat)
    top_idx = int(np.argmax(evals))
    loading = evecs[:, top_idx].copy()
    raw_scores = z @ loading
    if pearson_corr(raw_scores, x[:, orient_col]) < 0.0:
        loading = -loading
        raw_scores = -raw_scores
    score_sd = float(raw_scores.std(ddof=1))
    scores = raw_scores / score_sd
    explained = float(evals[top_idx] / np.sum(evals))
    return {
        "scores": scores,
        "mean": col_mean,
        "sd": col_sd,
        "loading": loading,
        "score_sd": score_sd,
        "var_explained": explained,
    }


def project_pca1(values_vec: np.ndarray | list[float], pca_model: dict) -> float:
    v = np.asarray(values_vec, dtype=float)
    z = (v - pca_model["mean"]) / pca_model["sd"]
    return float((z @ pca_model["loading"]) / pca_model["score_sd"])


def quasibinomial_logit_wls(
    nip: np.ndarray,
    n_tested: np.ndarray,
    sigma: np.ndarray,
    cov: np.ndarray | None = None,
) -> dict[str, float]:
    """Overdispersion-adjusted binomial delta-method logit WLS across 6 sites."""
    p = np.clip(np.asarray(nip, dtype=float), 1e-4, 1.0 - 1e-4)
    n = np.asarray(n_tested, dtype=float)
    logit_p = np.log(p / (1.0 - p))
    weights = n * p * (1.0 - p)
    cols = [np.ones_like(p), np.asarray(sigma, dtype=float)]
    if cov is not None:
        zc = np.asarray(cov, dtype=float)
        cols.append((zc - zc.mean()) / float(zc.std(ddof=1)))
    design = np.column_stack(cols)
    root_w = np.sqrt(weights)
    w_design = design * root_w[:, None]
    w_resp = logit_p * root_w
    q_mat, r_mat = np.linalg.qr(w_design, mode="reduced")
    beta = np.linalg.solve(r_mat, q_mat.T @ w_resp)
    resid = logit_p - design @ beta
    df = int(len(p) - design.shape[1])
    disp = max(1.0, float(np.sum(weights * resid * resid)) / max(df, 1))
    r_inv = np.linalg.inv(r_mat)
    vcov = disp * (r_inv @ r_inv.T)
    se_sig = math.sqrt(float(vcov[1, 1]))
    return {
        "beta_sigma": float(beta[1]),
        "se_sigma": se_sig,
        "t_sigma": float(beta[1] / se_sig),
        "beta_z": float(beta[2]) if cov is not None else float("nan"),
        "se_z": math.sqrt(float(vcov[2, 2])) if cov is not None else float("nan"),
        "df": df,
        "overdispersion": disp,
    }


def within_group_ols(
    panel_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> tuple[float, np.ndarray, float, int]:
    """Fixed-effects OLS via within-group demeaning and QR least squares."""
    dx_parts, dy_parts, dz_parts = [], [], []
    n_obs = 0
    n_grp = 0
    n_cov = 0
    for xg, yg, zg in panel_groups:
        if len(xg) < 2:
            continue
        n_grp += 1
        n_obs += len(xg)
        dx_parts.append(xg - xg.mean())
        dy_parts.append(yg - yg.mean())
        if zg is not None and zg.size > 0:
            n_cov = zg.shape[1]
            dz_parts.append(zg - zg.mean(axis=0, keepdims=True))
    if n_grp == 0:
        return float("nan"), np.zeros(0), float("nan"), 0
    dx = np.concatenate(dx_parts)
    dy = np.concatenate(dy_parts)
    if n_cov == 0:
        sxx = float(dx @ dx)
        if sxx <= 1e-15:
            return float("nan"), np.zeros(0), float("nan"), 0
        b = float((dx @ dy) / sxx)
        resid = dy - b * dx
        df = n_obs - n_grp - 1
        se = math.sqrt(float(resid @ resid) / max(df, 1) / sxx)
        return b, np.zeros(0), se, df
    dz = np.vstack(dz_parts)
    w = np.column_stack([dx, dz])
    coef, _, rank, _ = np.linalg.lstsq(w, dy, rcond=None)
    if rank < w.shape[1]:
        # Near-collinear bootstrap draw check
        pass
    resid = dy - w @ coef
    df = n_obs - n_grp - 1 - n_cov
    gram_inv = np.linalg.pinv(w.T @ w)
    se = math.sqrt(max(0.0, float(resid @ resid) / max(df, 1) * float(gram_inv[0, 0])))
    return float(coef[0]), coef[1:].copy(), se, df


def explicit_dummy_ols(
    panel_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> float:
    """Explicit group-dummy design matrix solver for Rule 7 verification."""
    kept = [(xg, yg, zg) for (xg, yg, zg) in panel_groups if len(xg) >= 2]
    k = len(kept)
    rows, y_all = [], []
    for g_i, (xg, yg, zg) in enumerate(kept):
        dummies = [1.0 if j == g_i else 0.0 for j in range(k)]
        for idx in range(len(xg)):
            cov_vals = list(zg[idx]) if (zg is not None and zg.size > 0) else []
            rows.append([float(xg[idx])] + cov_vals + dummies)
            y_all.append(float(yg[idx]))
    beta = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(y_all, dtype=float), rcond=None)[0]
    return float(beta[0])


def cary_two_year_block_bootstrap(
    records: list[dict],
    plot_names: list[str],
    cov_cols: list[str],
    n_boot: int = 2000,
    seed: int = 20261114,
) -> dict:
    """Within-plot OLS slope and 2-year calendar-block bootstrap SE and 95% percentile CI."""
    work = [dict(r) for r in records if all(c in r and math.isfinite(r[c]) for c in cov_cols)]
    for col in cov_cols:
        vals = np.array([r[col] for r in work], dtype=float)
        mu = float(vals.mean())
        sd = max(float(vals.std(ddof=1)), 1e-15)
        for r in work:
            r[col] = (r[col] - mu) / sd

    groups = []
    for p in plot_names:
        sub = [r for r in work if r["plot"] == p]
        xg = np.array([r["x"] for r in sub], dtype=float)
        yg = np.array([r["y"] for r in sub], dtype=float)
        zg = np.array([[r[c] for c in cov_cols] for r in sub], dtype=float) if cov_cols else None
        groups.append((xg, yg, zg))
    b_hat, b_cov, se_ols, df_res = within_group_ols(groups)

    t0 = min(r["mouse_year"] for r in work)
    t1 = max(r["mouse_year"] for r in work)
    n_years = t1 - t0 + 1
    n_blocks = math.ceil(n_years / 2)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_years - 1, size=(n_boot, n_blocks))
    year_draws = np.concatenate([starts, starts + 1], axis=1)

    p_map = {p: i for i, p in enumerate(plot_names)}
    n_p = len(plot_names)
    p_c = len(cov_cols)
    x_mat = np.full((n_p, n_years), np.nan)
    y_mat = np.full((n_p, n_years), np.nan)
    z_mat = np.full((n_p, n_years, p_c), np.nan) if p_c > 0 else None
    for r in work:
        pi = p_map[r["plot"]]
        yi = r["mouse_year"] - t0
        x_mat[pi, yi] = r["x"]
        y_mat[pi, yi] = r["y"]
        for ci, col in enumerate(cov_cols):
            z_mat[pi, yi, ci] = r[col]

    boot_slopes = np.full(n_boot, np.nan)
    short_year_draws = 0
    for b_idx in range(n_boot):
        cols = year_draws[b_idx]
        if len(set(cols.tolist())) < p_c + 2:
            short_year_draws += 1
        bg = []
        for pi in range(n_p):
            xv = x_mat[pi, cols]
            yv = y_mat[pi, cols]
            ok = np.isfinite(xv) & np.isfinite(yv)
            if p_c > 0:
                zv = z_mat[pi, cols, :]
                ok &= np.all(np.isfinite(zv), axis=1)
                if int(ok.sum()) >= 2:
                    bg.append((xv[ok], yv[ok], zv[ok]))
            else:
                if int(ok.sum()) >= 2:
                    bg.append((xv[ok], yv[ok], None))
        if bg:
            val, _, _, _ = within_group_ols(bg)
            if math.isfinite(val):
                boot_slopes[b_idx] = val

    valid_b = boot_slopes[np.isfinite(boot_slopes)]
    se_boot = float(np.std(valid_b, ddof=1))
    ci_low, ci_high = np.percentile(valid_b, [2.5, 97.5])
    return {
        "b": b_hat,
        "beta_cov": b_cov.tolist(),
        "se": se_boot,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "se_ols": se_ols,
        "df_resid": df_res,
        "n_pairs": len(work),
        "n_cohorts": len({r["mouse_year"] for r in work}),
        "n_valid_boots": int(len(valid_b)),
        "insufficient_distinct_years": int(short_year_draws),
    }


def neon_within_site_bootstrap(
    records: list[dict],
    site_names: list[str],
    cov_cols: list[str],
    n_boot: int = 2000,
    seed: list[int] | int = (20261114, 3),
) -> dict:
    """Within-site pooled slope and within-site pair bootstrap SE and 95% percentile CI."""
    groups = []
    for s in site_names:
        sub = [r for r in records if r["site"] == s]
        if len(sub) < 2:
            continue
        xg = np.array([r["x"] for r in sub], dtype=float)
        yg = np.array([r["y"] for r in sub], dtype=float)
        zg = np.array([[r[c] for c in cov_cols] for r in sub], dtype=float) if cov_cols else None
        groups.append((xg, yg, zg))
    b_hat, b_cov, se_ols, df_res = within_group_ols(groups)

    rng = np.random.default_rng(list(seed) if isinstance(seed, tuple) else seed)
    boots = []
    for _ in range(n_boot):
        bg = []
        for xg, yg, zg in groups:
            draw = rng.integers(0, len(xg), size=len(xg))
            bg.append((xg[draw], yg[draw], zg[draw] if zg is not None else None))
        val, _, _, _ = within_group_ols(bg)
        if math.isfinite(val):
            boots.append(val)
    arr = np.asarray(boots, dtype=float)
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    return {
        "b": b_hat,
        "beta_cov": b_cov.tolist(),
        "se": float(np.std(arr, ddof=1)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "se_ols": se_ols,
        "df_resid": df_res,
        "n_pairs": sum(len(g[0]) for g in groups),
    }


def annual_block_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261114,
) -> dict[str, float]:
    """Pearson r, Spearman rho, and 2-year block-bootstrap 95% CIs on aligned annual series."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n_yr = len(xa)
    r_obs = pearson_corr(xa, ya)
    rho_obs = spearman_corr(xa, ya)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_yr - 1, size=(n_boot, math.ceil(n_yr / 2)))
    cols = np.concatenate([starts, starts + 1], axis=1)
    r_boots, rho_boots = [], []
    for b in range(n_boot):
        xb = xa[cols[b]]
        yb = ya[cols[b]]
        rv = pearson_corr(xb, yb)
        rhv = spearman_corr(xb, yb)
        if math.isfinite(rv):
            r_boots.append(rv)
        if math.isfinite(rhv):
            rho_boots.append(rhv)
    r_ci = np.percentile(r_boots, [2.5, 97.5]) if len(r_boots) >= 10 else [np.nan, np.nan]
    rho_ci = np.percentile(rho_boots, [2.5, 97.5]) if len(rho_boots) >= 10 else [np.nan, np.nan]
    return {
        "r": r_obs,
        "r_low": float(r_ci[0]),
        "r_high": float(r_ci[1]),
        "rho": rho_obs,
        "rho_low": float(rho_ci[0]),
        "rho_high": float(rho_ci[1]),
        "years": int(n_yr),
    }


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n)) / denom
    return {
        "count": int(k),
        "n": int(n),
        "freq": float(p),
        "mc_se": math.sqrt(p * (1.0 - p) / n),
        "wilson_low": float(center - half),
        "wilson_high": float(center + half),
    }


def vectorized_panel_fe_bootstrap(
    x_panel: np.ndarray,
    y_panel: np.ndarray,
    z_panel: np.ndarray,
    boot_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched within-plot OLS unadjusted and 1-covariate adjusted slopes across (B, G, T) studies and K weights."""
    valid = np.isfinite(x_panel) & np.isfinite(y_panel) & np.isfinite(z_panel)
    xv = np.where(valid, x_panel, 0.0)
    yv = np.where(valid, y_panel, 0.0)
    zv = np.where(valid, z_panel, 0.0)
    mv = valid.astype(float)

    def wsum(arr: np.ndarray) -> np.ndarray:
        return np.einsum("bgt,kt->bgk", arr, boot_weights, optimize=True)

    cnt = wsum(mv)
    denom_cnt = np.maximum(cnt, 1.0)
    sx = wsum(xv)
    sy = wsum(yv)
    sz = wsum(zv)
    sxx = (wsum(xv * xv) - (sx * sx) / denom_cnt).sum(axis=1)
    sxy = (wsum(xv * yv) - (sx * sy) / denom_cnt).sum(axis=1)
    sxz = (wsum(xv * zv) - (sx * sz) / denom_cnt).sum(axis=1)
    szz = (wsum(zv * zv) - (sz * sz) / denom_cnt).sum(axis=1)
    syz = (wsum(yv * zv) - (sy * sz) / denom_cnt).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b_un = sxy / sxx
        det = sxx * szz - sxz * sxz
        b_adj = (sxy * szz - sxz * syz) / det
    return b_un, b_adj
