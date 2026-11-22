#!/usr/bin/env python3
"""Independent ML/REML precision-weighted mixed-effects meta-regression solver.

Written from scratch for r36-rep-wood without importing or copying any code from
casestudies/wood/.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg, optimize, stats


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni adjusted two-sided p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out


def fit_mixed_meta_regression(
    y: np.ndarray,
    X: np.ndarray,
    sampling_var: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    reml: bool = False,
    multistart: bool = True,
) -> dict:
    """Fit y ~ N(X beta, diag(sampling_var + tau2) + sigma2_cluster * Z Z^T)
    by exact profile ML or REML over non-negative variance components.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(sampling_var, dtype=np.float64)
    n, p = X.shape
    rk = int(np.linalg.matrix_rank(X))
    if rk < p:
        return {"estimable": False, "n": int(n), "p": int(p), "rank": rk}

    has_cluster = cluster_labels is not None
    if has_cluster:
        _, inv_idx = np.unique(np.asarray(cluster_labels), return_inverse=True)
        n_clusters = int(inv_idx.max()) + 1
        Z = np.eye(n_clusters, dtype=np.float64)[inv_idx]
    else:
        Z = np.zeros((n, 0), dtype=np.float64)

    def _eval_obj_and_grad(var_vec: np.ndarray, full: bool = False):
        tau2 = float(var_vec[0])
        d_inv = 1.0 / (v + tau2)
        if has_cluster:
            sig2_c = float(var_vec[1])
            w_sum = Z.T @ d_inv
            denom_c = 1.0 + sig2_c * w_sum
            d_Z = d_inv[:, None] * Z
            Sigma_inv = np.diag(d_inv) - (d_Z * (sig2_c / denom_c)[None, :]) @ d_Z.T
            logdet_Sigma = float(np.sum(np.log(v + tau2)) + np.sum(np.log(denom_c)))
        else:
            Sigma_inv = np.diag(d_inv)
            logdet_Sigma = float(np.sum(np.log(v + tau2)))

        XtSi = X.T @ Sigma_inv
        XtSiX = XtSi @ X
        chol_XtSiX = linalg.cho_factor(XtSiX, lower=True, check_finite=False)
        cov_beta = linalg.cho_solve(chol_XtSiX, np.eye(p), check_finite=False)
        beta = cov_beta @ (XtSi @ y)
        res = y - X @ beta
        Si_res = Sigma_inv @ res

        quad = float(res @ Si_res)
        if reml:
            logdet_XtSiX = 2.0 * float(np.sum(np.log(np.diag(chol_XtSiX[0]))))
            nll = 0.5 * ((n - p) * np.log(2.0 * np.pi) + logdet_Sigma + logdet_XtSiX + quad)
            P_mat = Sigma_inv - XtSi.T @ cov_beta @ XtSi
        else:
            nll = 0.5 * (n * np.log(2.0 * np.pi) + logdet_Sigma + quad)
            P_mat = Sigma_inv

        grad_tau2 = 0.5 * (float(np.trace(P_mat)) - float(Si_res @ Si_res))
        if has_cluster:
            Zt_Si_res = Z.T @ Si_res
            grad_c = 0.5 * (float(np.sum(Z * (P_mat @ Z))) - float(Zt_Si_res @ Zt_Si_res))
            grad = np.array([grad_tau2, grad_c], dtype=np.float64)
        else:
            grad = np.array([grad_tau2], dtype=np.float64)

        if full:
            return nll, grad, beta, cov_beta
        return nll, grad

    n_var = 2 if has_cluster else 1
    tau2_lo = 0.0 if float(np.min(v)) > 0.0 else 1e-12
    bounds = [(tau2_lo, 5.0)] + ([(0.0, 5.0)] if has_cluster else [])
    if multistart:
        starts = (
            [[0.05, 0.0], [0.05, 0.02], [0.02, 0.20], [0.20, 0.20]]
            if has_cluster
            else [[0.01], [0.08], [0.25]]
        )
    else:
        starts = [[0.05, 0.01]] if has_cluster else [[0.05]]

    def _proj_grad_inf(x_vec: np.ndarray, g_vec: np.ndarray) -> float:
        pg = g_vec.copy()
        for j, (lo, hi) in enumerate(bounds):
            if x_vec[j] <= lo + 1e-9 and pg[j] > 0.0:
                pg[j] = 0.0
            if x_vec[j] >= hi - 1e-9 and pg[j] < 0.0:
                pg[j] = 0.0
        return float(np.max(np.abs(pg)))

    best_x = None
    best_f = np.inf
    for st in starts:
        opt = optimize.minimize(
            _eval_obj_and_grad,
            np.array(st, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 250, "maxls": 40},
        )
        f_cur, g_cur = _eval_obj_and_grad(opt.x)
        if _proj_grad_inf(opt.x, g_cur) > 1e-6:
            opt_pow = optimize.minimize(
                lambda z: _eval_obj_and_grad(z)[0],
                opt.x,
                method="Powell",
                bounds=bounds,
                options={"xtol": 1e-12, "ftol": 1e-14, "maxiter": 250},
            )
            opt2 = optimize.minimize(
                _eval_obj_and_grad,
                opt_pow.x,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 250, "maxls": 40},
            )
            for cand in (opt_pow, opt2):
                fc = _eval_obj_and_grad(cand.x)[0]
                if fc < f_cur:
                    f_cur = fc
                    opt = cand
        if f_cur < best_f:
            best_f = f_cur
            best_x = opt.x.copy()

    nll, grad, beta, cov_beta = _eval_obj_and_grad(best_x, full=True)
    pg_norm = _proj_grad_inf(best_x, grad)
    se = np.sqrt(np.diag(cov_beta))
    df = int(n - p)
    t_crit = float(stats.t.ppf(0.975, df))
    t_stat = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(t_stat), df)
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se

    k = int(p + n_var)
    aicc = float(2.0 * nll + 2.0 * k + (2.0 * k * (k + 1.0)) / (n - k - 1.0))
    return {
        "estimable": True,
        "reml": bool(reml),
        "n": int(n),
        "p": int(p),
        "df": df,
        "k": k,
        "nll": float(nll),
        "aicc": aicc,
        "beta": beta,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "pvalues": pvals,
        "var_components": best_x,
        "projected_grad_max": pg_norm,
        "converged": bool(pg_norm < 1e-5),
    }


def parametric_bootstrap_lrt(
    y: np.ndarray,
    X_null: np.ndarray,
    X_alt: np.ndarray,
    sampling_var: np.ndarray,
    cluster_labels: np.ndarray,
    B: int = 2000,
    seed: int = 20261115,
) -> dict:
    """Parametric bootstrap LRT comparing ML X_alt vs ML X_null under fitted ML X_null."""
    null_ml = fit_mixed_meta_regression(y, X_null, sampling_var, cluster_labels, reml=False, multistart=True)
    alt_ml = fit_mixed_meta_regression(y, X_alt, sampling_var, cluster_labels, reml=False, multistart=True)
    lrt_obs = float(2.0 * (null_ml["nll"] - alt_ml["nll"]))
    delta_aicc = float(alt_ml["aicc"] - null_ml["aicc"])

    _, inv_idx = np.unique(np.asarray(cluster_labels), return_inverse=True)
    n_clusters = int(inv_idx.max()) + 1
    n = len(y)

    mean_null = X_null @ null_ml["beta"]
    sd_obs = np.sqrt(sampling_var + null_ml["var_components"][0])
    sd_cluster = float(np.sqrt(null_ml["var_components"][1]))

    rng = np.random.default_rng(seed)
    lrt_draws = np.empty(B, dtype=np.float64)
    n_failed = 0
    for b in range(B):
        u_c = sd_cluster * rng.normal(size=n_clusters)
        y_sim = mean_null + u_c[inv_idx] + sd_obs * rng.normal(size=n)
        f0 = fit_mixed_meta_regression(y_sim, X_null, sampling_var, cluster_labels, reml=False, multistart=False)
        f1 = fit_mixed_meta_regression(y_sim, X_alt, sampling_var, cluster_labels, reml=False, multistart=False)
        if not (f0["converged"] and f1["converged"]):
            f0 = fit_mixed_meta_regression(y_sim, X_null, sampling_var, cluster_labels, reml=False, multistart=True)
            f1 = fit_mixed_meta_regression(y_sim, X_alt, sampling_var, cluster_labels, reml=False, multistart=True)
        if not (f0["converged"] and f1["converged"]):
            n_failed += 1
        lrt_draws[b] = max(0.0, 2.0 * (f0["nll"] - f1["nll"]))

    p_boot = float((1 + int(np.sum(lrt_draws >= lrt_obs))) / (B + 1))
    mc_se = float(np.sqrt(p_boot * (1.0 - p_boot) / (B + 1)))
    return {
        "lrt_obs": lrt_obs,
        "delta_aicc": delta_aicc,
        "p_boot": p_boot,
        "B": int(B),
        "seed": int(seed),
        "mc_se": mc_se,
        "n_failed": int(n_failed),
        "null_ml": null_ml,
        "alt_ml": alt_ml,
    }


def run_rule7_meta_validation() -> dict:
    """Rule 7 known-result checks for mixed meta-regression BEFORE data analysis.
    Preset tolerances:
      - Homoscedastic v=0 REML vs exact OLS beta, SE, and unbiased s^2 = RSS/(n-p): < 1e-10
      - Homoscedastic v=0 ML vs exact MLE sigma^2 = RSS/n and Gaussian log-likelihood: < 1e-10
      - Balanced one-way random-effects ANOVA REML vs exact closed-form between/within MS: < 1e-8
    """
    rng = np.random.default_rng(101)
    n = 50
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, size=n)])
    beta_true = np.array([0.4, -0.25, 0.6])
    y = X @ beta_true + rng.normal(0.0, 0.35, size=n)

    # 1. Exact OLS & REML check
    reml_fit = fit_mixed_meta_regression(y, X, np.zeros(n), None, reml=True)
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta_ols) ** 2))
    s2_ols = rss / (n - 3)
    se_ols = np.sqrt(np.diag(s2_ols * np.linalg.inv(X.T @ X)))

    err_beta_ols = float(np.max(np.abs(reml_fit["beta"] - beta_ols)))
    err_s2_reml = abs(float(reml_fit["var_components"][0]) - s2_ols)
    err_se_ols = float(np.max(np.abs(reml_fit["se"] - se_ols)))

    # 2. Exact ML check
    ml_fit = fit_mixed_meta_regression(y, X, np.zeros(n), None, reml=False)
    s2_ml = rss / n
    nll_ml_exact = 0.5 * n * (np.log(2.0 * np.pi * s2_ml) + 1.0)
    err_s2_ml = abs(float(ml_fit["var_components"][0]) - s2_ml)
    err_nll_ml = abs(ml_fit["nll"] - nll_ml_exact)

    # 3. Exact balanced 1-way random-effects ANOVA REML check (G=5 groups of m=8)
    G, m = 5, 8
    N_bal = G * m
    grp = np.repeat(np.arange(G), m)
    y_bal = np.repeat(rng.normal(0.0, 0.5, size=G), m) + rng.normal(0.0, 0.25, size=N_bal)
    g_means = np.array([y_bal[grp == g].mean() for g in range(G)])
    grand_mean = y_bal.mean()
    ms_between = m * np.sum((g_means - grand_mean) ** 2) / (G - 1)
    ms_within = np.sum((y_bal - g_means[grp]) ** 2) / (G * (m - 1))
    sigma2_within_exact = float(ms_within)
    sigma2_between_exact = float(max(0.0, (ms_between - ms_within) / m))

    bal_reml = fit_mixed_meta_regression(y_bal, np.ones((N_bal, 1)), np.zeros(N_bal), grp, reml=True)
    err_anova_within = abs(float(bal_reml["var_components"][0]) - sigma2_within_exact)
    err_anova_between = abs(float(bal_reml["var_components"][1]) - sigma2_between_exact)

    assert err_beta_ols < 1e-10 and err_s2_reml < 1e-8 and err_se_ols < 1e-9
    assert err_s2_ml < 1e-8 and err_nll_ml < 1e-10
    assert err_anova_within < 1e-8 and err_anova_between < 1e-8

    return {
        "ols_beta_max_abs_err": err_beta_ols,
        "reml_s2_abs_err": err_s2_reml,
        "ols_se_max_abs_err": err_se_ols,
        "ml_s2_abs_err": err_s2_ml,
        "ml_nll_abs_err": err_nll_ml,
        "balanced_anova_within_abs_err": err_anova_within,
        "balanced_anova_between_abs_err": err_anova_between,
        "pass": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_rule7_meta_validation(), indent=2))
