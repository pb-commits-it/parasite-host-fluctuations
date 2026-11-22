#!/usr/bin/env python3
"""Exact Laplace-approximated Negative Binomial GLMM engine in pure NumPy/SciPy.

Implements the exact hierarchical model family of Wood et al. (2023, PNAS;
src/TMB/glmm*.cpp) and its registered life-cycle trait extensions without
executing any external/downloaded R or C++ code.

All random effects are parameterized in standardized coordinates u_raw ~ N(0, I)
(an equivalent change of variables in the reference likelihood), combined with
a block Schur complement over the fish-specimen random effects. The NB
likelihood is not globally convex in dispersion: damped Newton steps and
mode/Hessian checks are required. The Laplace approximation is not an exact
integral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, sparse, special


def nb_loglik_and_derivs(y: np.ndarray, eta: np.ndarray, psi: np.ndarray):
    """Exact -log P(y | mu=exp(eta), phi=exp(psi)) and 1st/2nd derivatives of -ll
    with respect to (eta, psi), vectorized over observations.
    """
    eta_c = eta
    if np.max(np.abs(eta)) > 200 or np.max(np.abs(psi)) > 40:
        raise ValueError("Trial predictor outside numerically stable NB range")
    psi_c = psi
    mu = np.exp(eta_c)
    phi = np.exp(psi_c)
    denom = mu + phi
    p = mu / denom
    q = phi / denom
    log_denom = np.logaddexp(eta_c, psi_c)
    log_q = psi_c - log_denom

    ll = (
        special.gammaln(y + phi)
        - special.gammaln(phi)
        - special.gammaln(y + 1.0)
        + phi * log_q
        + y * (eta_c - log_denom)
    )

    g_eta = (y - mu) * q
    dig_diff = special.digamma(y + phi) - special.digamma(phi)
    A = dig_diff + log_q + (mu - y) / denom
    g_psi = phi * A

    pq = p * q
    h_ee = (y + phi) * pq
    h_ep = -(y - mu) * pq
    trig_diff = special.polygamma(1, y + phi) - special.polygamma(1, phi)
    dA_dphi = trig_diff + (1.0 / phi) - (1.0 / denom) - (mu - y) / (denom * denom)
    h_pp = -(phi * A + (phi * phi) * dA_dphi)

    return -float(np.sum(ll)), -g_eta, -g_psi, h_ee, h_ep, h_pp


@dataclass
class HierarchicalNBData:
    y: np.ndarray              # (n,) integer counts as float
    hostpar_idx: np.ndarray    # (n,) in 0..n_hp-1
    spc_idx: np.ndarray        # (n,) in 0..n_spc-1
    fish_idx: np.ndarray       # (n,) in 0..n_fish-1
    tax_of_spc: np.ndarray     # (n_spc,) in 0..n_tax-1
    tax_of_obs: np.ndarray     # (n,) in 0..n_tax-1
    nh_of_obs: np.ndarray      # (n,) in 0..2
    slat: np.ndarray           # (n,)
    slen: np.ndarray           # (n,)
    uyear: np.ndarray          # (n,)
    n_hp: int
    n_spc: int
    n_fish: int
    n_tax: int


class LaplaceNBGLMM:
    """Flexible specification matching Wood Models 1-4 and trait extensions.

    Standardized random-effect vector w = [w_v, w_z] ~ N(0, I):
      u = mu_u(theta) + sig_u(theta) * w
      v-block = [bo, (bog), blat, blen, (g), (atax), psi]
      z-block = [fish specimen random effects] (diagonal inner block)
    """

    def __init__(
        self,
        data: HierarchicalNBData,
        include_year: bool = True,
        X_slope: np.ndarray | None = None,
        bog_obs_idx: np.ndarray | None = None,
        n_bog: int = 0,
        has_tax_re: bool = False,
        extra_fixed_X: np.ndarray | None = None,
    ):
        self.d = data
        self.include_year = include_year
        self.has_bog = bog_obs_idx is not None and n_bog > 0
        self.bog_obs_idx = bog_obs_idx
        self.n_bog = n_bog
        self.has_tax_re = has_tax_re and include_year
        if include_year:
            assert X_slope is not None and X_slope.shape[0] == data.n_spc
            self.X_slope = np.asarray(X_slope, dtype=float)
            self.p_slope = self.X_slope.shape[1]
        else:
            self.X_slope = np.zeros((data.n_spc, 0))
            self.p_slope = 0

        if extra_fixed_X is not None:
            self.extra_fixed_X = np.asarray(extra_fixed_X, dtype=float)
            self.p_extra = self.extra_fixed_X.shape[1]
        else:
            self.extra_fixed_X = np.zeros((len(data.y), 0))
            self.p_extra = 0

        idx = 0
        self.sl_bo = slice(idx, idx + data.n_hp); idx += data.n_hp
        self.sl_bog = slice(idx, idx + (n_bog if self.has_bog else 0)); idx += (n_bog if self.has_bog else 0)
        self.sl_blat = slice(idx, idx + data.n_spc); idx += data.n_spc
        self.sl_blen = slice(idx, idx + data.n_hp); idx += data.n_hp
        self.sl_g = slice(idx, idx + (data.n_spc if include_year else 0)); idx += (data.n_spc if include_year else 0)
        self.sl_atax = slice(idx, idx + (data.n_tax if self.has_tax_re else 0)); idx += (data.n_tax if self.has_tax_re else 0)
        self.dim_eta_v = idx
        self.sl_psi = slice(idx, idx + data.n_spc); idx += data.n_spc
        self.dim_v = idx
        self.sl_z = slice(idx, idx + data.n_fish); idx += data.n_fish
        self.dim_u = idx

        n = len(data.y)
        ar = np.arange(n)
        rows, cols, vals = [], [], []

        def add_block(col_offset: int, c_idx: np.ndarray, weight: np.ndarray | float = 1.0):
            rows.append(ar)
            cols.append(col_offset + c_idx)
            vals.append(np.full(n, weight, dtype=float) if np.ndim(weight) == 0 else np.asarray(weight, float))

        add_block(self.sl_bo.start, data.hostpar_idx, 1.0)
        if self.has_bog:
            add_block(self.sl_bog.start, bog_obs_idx, 1.0)
        add_block(self.sl_blat.start, data.spc_idx, data.slat)
        add_block(self.sl_blen.start, data.hostpar_idx, data.slen)
        if include_year:
            add_block(self.sl_g.start, data.spc_idx, data.uyear)
        if self.has_tax_re:
            add_block(self.sl_atax.start, data.tax_of_obs, data.uyear)

        self.A_eta_v = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, self.dim_eta_v),
        )
        self.A_z = sparse.csr_matrix(
            (np.ones(n, dtype=float), (ar, data.fish_idx)),
            shape=(n, data.n_fish),
        )
        self.A_psi = sparse.csr_matrix(
            (np.ones(n, dtype=float), (ar, data.spc_idx)),
            shape=(n, data.n_spc),
        )
        self._w_cache = np.zeros(self.dim_u, dtype=float)

    def unpack_theta(self, theta: np.ndarray) -> dict:
        i = 0
        out = {}
        out["bobar"] = float(theta[i]); i += 1
        out["logsigma_bo"] = float(theta[i]); i += 1
        if self.has_bog:
            out["logsigma_bog"] = float(theta[i]); i += 1
        out["blatbar"] = float(theta[i]); i += 1
        out["logsigma_blat"] = float(theta[i]); i += 1
        out["blenbar"] = float(theta[i]); i += 1
        out["logsigma_blen"] = float(theta[i]); i += 1
        out["logsigma_z"] = float(theta[i]); i += 1
        out["logphibar"] = float(theta[i]); i += 1
        out["loglogsigma_phi"] = float(theta[i]); i += 1
        if self.include_year:
            out["beta_slope"] = np.asarray(theta[i:i + self.p_slope], dtype=float); i += self.p_slope
            out["logsigma_g"] = float(theta[i]); i += 1
        if self.has_tax_re:
            out["logsigma_tax"] = float(theta[i]); i += 1
        if self.p_extra > 0:
            out["beta_extra"] = np.asarray(theta[i:i + self.p_extra], dtype=float); i += self.p_extra
        assert i == len(theta)
        return out

    def initial_theta(self) -> np.ndarray:
        vals = [-1.1, 0.4]
        if self.has_bog:
            vals.append(-0.8)
        vals += [-0.12, -0.85, 0.50, -0.45, -0.78, -1.68, -0.18]
        if self.include_year:
            vals += [0.0] * self.p_slope + [-0.58]
        if self.has_tax_re:
            vals.append(-1.8)
        if self.p_extra > 0:
            vals += [0.0] * self.p_extra
        return np.array(vals, dtype=float)

    def _mu_and_sig(self, pars: dict):
        mu_u = np.zeros(self.dim_u)
        sig_u = np.ones(self.dim_u)
        if self.has_bog:
            mu_u[self.sl_bo] = 0.0
            sig_u[self.sl_bo] = math.exp(pars["logsigma_bo"])
            mu_u[self.sl_bog] = pars["bobar"]
            sig_u[self.sl_bog] = math.exp(pars["logsigma_bog"])
        else:
            mu_u[self.sl_bo] = pars["bobar"]
            sig_u[self.sl_bo] = math.exp(pars["logsigma_bo"])

        mu_u[self.sl_blat] = pars["blatbar"]
        sig_u[self.sl_blat] = math.exp(pars["logsigma_blat"])
        mu_u[self.sl_blen] = pars["blenbar"]
        sig_u[self.sl_blen] = math.exp(pars["logsigma_blen"])
        if self.include_year:
            mu_u[self.sl_g] = 0.0
            sig_u[self.sl_g] = math.exp(pars["logsigma_g"])
        if self.has_tax_re:
            mu_u[self.sl_atax] = 0.0
            sig_u[self.sl_atax] = math.exp(pars["logsigma_tax"])

        mu_u[self.sl_psi] = pars["logphibar"]
        sig_u[self.sl_psi] = math.exp(pars["loglogsigma_phi"])
        mu_u[self.sl_z] = 0.0
        sig_u[self.sl_z] = math.exp(pars["logsigma_z"])
        return mu_u, sig_u

    def joint_blocks(self, w: np.ndarray, pars: dict):
        """Evaluate joint NLL, gradient wrt w, and Schur blocks wrt w."""
        mu_u, sig_u = self._mu_and_sig(pars)
        u = mu_u + sig_u * w

        eta_fixed = np.zeros(len(self.d.y))
        if self.include_year and self.p_slope > 0:
            mean_slopes = self.X_slope @ pars["beta_slope"]
            eta_fixed = mean_slopes[self.d.spc_idx] * self.d.uyear
        if self.p_extra > 0:
            eta_fixed = eta_fixed + self.extra_fixed_X @ pars["beta_extra"]

        eta = eta_fixed + self.A_eta_v @ u[: self.dim_eta_v] + u[self.sl_z][self.d.fish_idx]
        psi = u[self.sl_psi][self.d.spc_idx]

        nll_d, g_eta, g_psi, h_ee, h_ep, h_pp = nb_loglik_and_derivs(self.d.y, eta, psi)
        # Standardized normal prior -sum(dnorm(w, 0, 1, true)):
        nll_p = 0.5 * self.dim_u * math.log(2.0 * math.pi) + 0.5 * float(w @ w)

        grad_u = np.empty(self.dim_u, dtype=float)
        grad_u[: self.dim_eta_v] = self.A_eta_v.T @ g_eta
        grad_u[self.sl_psi] = np.bincount(self.d.spc_idx, weights=g_psi, minlength=self.d.n_spc)
        grad_u[self.sl_z] = np.bincount(self.d.fish_idx, weights=g_eta, minlength=self.d.n_fish)
        grad_w = w + sig_u * grad_u

        sig_v = sig_u[: self.dim_v]
        sig_z = sig_u[self.sl_z]

        # Data Hessian wrt u_v, then scale by sig_v on both sides and add I:
        H_uu_vv = np.zeros((self.dim_v, self.dim_v), dtype=float)
        w_ee = sparse.diags(h_ee)
        w_ep = sparse.diags(h_ep)
        H_uu_vv[: self.dim_eta_v, : self.dim_eta_v] = (self.A_eta_v.T @ w_ee @ self.A_eta_v).toarray()
        H_ep = (self.A_eta_v.T @ w_ep @ self.A_psi).toarray()
        H_uu_vv[: self.dim_eta_v, self.sl_psi] = H_ep
        H_uu_vv[self.sl_psi, : self.dim_eta_v] = H_ep.T
        H_pp_diag = np.bincount(self.d.spc_idx, weights=h_pp, minlength=self.d.n_spc)
        H_uu_vv[self.sl_psi, self.sl_psi] = np.diag(H_pp_diag)

        H_vv = (sig_v[:, None] * H_uu_vv) * sig_v[None, :]
        H_vv[np.diag_indices(self.dim_v)] += 1.0

        H_uu_vz = np.zeros((self.dim_v, self.d.n_fish), dtype=float)
        H_uu_vz[: self.dim_eta_v, :] = (self.A_eta_v.T @ w_ee @ self.A_z).toarray()
        H_uu_vz[self.sl_psi, :] = (self.A_psi.T @ w_ep @ self.A_z).toarray()
        H_vz = (sig_v[:, None] * H_uu_vz) * sig_z[None, :]

        d_z_u = np.bincount(self.d.fish_idx, weights=h_ee, minlength=self.d.n_fish)
        d_z = 1.0 + (sig_z * sig_z) * d_z_u

        return nll_d + nll_p, grad_w, H_vv, H_vz, d_z

    def _schur_solve(self, grad_w: np.ndarray, H_vv: np.ndarray, H_vz: np.ndarray, d_z: np.ndarray, jitter: float = 0.0):
        dz_reg = np.maximum(d_z + jitter, 1e-12)
        inv_dz = 1.0 / dz_reg
        S = H_vv - (H_vz * inv_dz[None, :]) @ H_vz.T
        if jitter > 0.0:
            S[np.diag_indices(self.dim_v)] += jitter
        cf_S = linalg.cho_factor(S, lower=True, check_finite=False)
        gv = grad_w[: self.dim_v]
        gz = grad_w[self.sl_z]
        rhs_v = -(gv - H_vz @ (gz * inv_dz))
        dv = linalg.cho_solve(cf_S, rhs_v, check_finite=False)
        dz = -(gz + H_vz.T @ dv) * inv_dz
        step = np.concatenate([dv, dz])
        logdet = float(np.sum(np.log(dz_reg))) + 2.0 * float(np.sum(np.log(np.diag(cf_S[0]))))
        return step, cf_S, logdet

    def inner_solve(self, pars: dict, w_init: np.ndarray | None = None, tol: float = 1e-8, max_iter: int = 60):
        tol = max(tol, 1e-7)
        w = (self._w_cache if w_init is None else w_init).copy()
        f_val, grad, H_vv, H_vz, d_z = self.joint_blocks(w, pars)
        for _ in range(max_iter):
            if np.max(np.abs(grad)) < tol:
                break
            jitter = 0.0
            for _ in range(20):
                try:
                    step, _, _ = self._schur_solve(grad, H_vv, H_vz, d_z, jitter=jitter)
                    break
                except linalg.LinAlgError:
                    jitter = 1e-5 if jitter == 0.0 else jitter * 10.0
            else:
                raise RuntimeError("Inner Schur Hessian could not be regularized")

            alpha = 1.0
            slope = float(grad @ step)
            for _ in range(25):
                w_try = w + alpha * step
                try:
                    f_try, grad_try, H_vv_try, H_vz_try, d_z_try = self.joint_blocks(w_try, pars)
                except (ValueError, OverflowError):
                    alpha *= 0.5
                    continue
                if math.isfinite(f_try) and f_try <= f_val + 1e-4 * alpha * slope + 1e-10:
                    w, f_val, grad, H_vv, H_vz, d_z = w_try, f_try, grad_try, H_vv_try, H_vz_try, d_z_try
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"Inner line search failed: f={f_val}, gradient={np.max(np.abs(grad))}")

        if np.max(np.abs(grad)) > 1e-5:
            raise RuntimeError(f"Inner mode did not converge: {np.max(np.abs(grad))}")
        _, cf_S, logdet_H = self._schur_solve(grad, H_vv, H_vz, d_z, jitter=0.0)
        laplace_nll = f_val + 0.5 * logdet_H - 0.5 * self.dim_u * math.log(2.0 * math.pi)
        return laplace_nll, w, cf_S

    def laplace_nll(self, theta: np.ndarray) -> float:
        if np.any(np.abs(theta) > 14.0):
            return 1e12
        pars = self.unpack_theta(theta)
        try:
            val, w_hat, _ = self.inner_solve(pars)
            if math.isfinite(val):
                self._w_cache = w_hat
                return val
        except Exception:
            pass
        return 1e12

    def fit(self, theta0: np.ndarray | None = None, maxiter: int = 120):
        x0 = self.initial_theta() if theta0 is None else np.asarray(theta0, float).copy()
        _ = self.laplace_nll(x0)

        evaluations = [0]
        def obj_and_grad_impl(th: np.ndarray):
            f0 = self.laplace_nll(th)
            if f0 >= 1e11:
                return f0, np.zeros_like(th)
            w_center = self._w_cache.copy()
            eps = 2e-4
            g = np.zeros_like(th)
            for j in range(len(th)):
                th_p = th.copy(); th_p[j] += eps
                fp, _, _ = self.inner_solve(self.unpack_theta(th_p), w_init=w_center, tol=1e-9)
                th_m = th.copy(); th_m[j] -= eps
                fm, _, _ = self.inner_solve(self.unpack_theta(th_m), w_init=w_center, tol=1e-9)
                g[j] = (fp - fm) / (2.0 * eps)
            self._w_cache = w_center
            evaluations[0] += 1
            if evaluations[0] % 10 == 0:
                print("evaluation", evaluations[0], "nll", f0, "gradient", float(np.max(np.abs(g))), flush=True)
            return f0, g

        def obj_and_grad(th):
            try:
                return obj_and_grad_impl(th)
            except (RuntimeError, ValueError, linalg.LinAlgError, OverflowError):
                return 1e12, np.zeros_like(th)

        bounds = []
        for key, value in self.unpack_theta(x0).items():
            count = np.size(value)
            bound = (-7.0, 2.0) if "logsigma" in key else (-10.0, 8.0)
            bounds.extend([bound] * count)
        res = optimize.minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-4, "maxls": 30},
        )
        theta_hat = res.x
        pars_hat = self.unpack_theta(theta_hat)
        nll_hat, w_hat, cf_S = self.inner_solve(pars_hat, tol=1e-10)
        self._w_cache = w_hat
        mu_u, sig_u = self._mu_and_sig(pars_hat)
        u_hat = mu_u + sig_u * w_hat

        p = len(theta_hat)
        eps_h = 1e-3
        H_theta = np.zeros((p, p))
        for j in range(p):
            th_p = theta_hat.copy(); th_p[j] += eps_h
            _, gp = obj_and_grad(th_p)
            th_m = theta_hat.copy(); th_m[j] -= eps_h
            _, gm = obj_and_grad(th_m)
            H_theta[:, j] = (gp - gm) / (2.0 * eps_h)
        H_theta = 0.5 * (H_theta + H_theta.T)
        eigvals, eigvecs = linalg.eigh(H_theta)
        cov_theta = (eigvecs / np.maximum(eigvals, 1e-6)) @ eigvecs.T
        se_theta = np.sqrt(np.diag(cov_theta))

        spc_slopes = None
        spc_slopes_se = None
        spc_slopes_cond_se = None
        if self.include_year:
            def eval_slopes(th_vec: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
                pr = self.unpack_theta(th_vec)
                _, sg = self._mu_and_sig(pr)
                uv = sg * w_vec
                mp = self.X_slope @ pr["beta_slope"] if self.p_slope > 0 else np.zeros(self.d.n_spc)
                gp = uv[self.sl_g]
                tp = uv[self.sl_atax][self.d.tax_of_spc] if self.has_tax_re else 0.0
                return mp + gp + tp

            spc_slopes = eval_slopes(theta_hat, w_hat)
            S_inv = linalg.cho_solve(cf_S, np.eye(self.dim_v), check_finite=False)

            # Linear map L_w from w_v to spc_slopes at fixed theta_hat:
            L_w = np.zeros((self.d.n_spc, self.dim_v))
            sig_g = math.exp(pars_hat["logsigma_g"])
            L_w[np.arange(self.d.n_spc), np.arange(self.sl_g.start, self.sl_g.stop)] = sig_g
            if self.has_tax_re:
                sig_t = math.exp(pars_hat["logsigma_tax"])
                L_w[np.arange(self.d.n_spc), self.sl_atax.start + self.d.tax_of_spc] = sig_t

            cond_var = np.diag(L_w @ S_inv @ L_w.T)
            spc_slopes_cond_se = np.sqrt(np.maximum(cond_var, 1e-12))

            # Total TMB ADREPORT variance: L_w S_inv L_w^T + (d spc_slopes_hat / d theta) cov_theta (...)^T
            J_total = np.zeros((self.d.n_spc, p))
            for j in range(p):
                th_p = theta_hat.copy(); th_p[j] += eps_h
                _, wp, _ = self.inner_solve(self.unpack_theta(th_p), w_init=w_hat, tol=1e-10)
                th_m = theta_hat.copy(); th_m[j] -= eps_h
                _, wm, _ = self.inner_solve(self.unpack_theta(th_m), w_init=w_hat, tol=1e-10)
                J_total[:, j] = (eval_slopes(th_p, wp) - eval_slopes(th_m, wm)) / (2.0 * eps_h)

            cov_slopes = L_w @ S_inv @ L_w.T + J_total @ cov_theta @ J_total.T
            spc_slopes_se = np.sqrt(np.maximum(np.diag(cov_slopes), 1e-12))

        k_par = len(theta_hat)
        n_obs = len(self.d.y)
        aicc_val = 2.0 * k_par + 2.0 * nll_hat + (2.0 * k_par * (k_par + 1.0)) / (n_obs - k_par - 1.0)

        return {
            "converged": bool(res.success),
            "outer_grad_max": float(np.max(np.abs(res.jac))),
            "inner_grad_max": float(np.max(np.abs(self.joint_blocks(w_hat, pars_hat)[1]))),
            "hessian_min_eigenvalue": float(eigvals.min()),
            "iterations": int(res.nit),
            "message": str(res.message),
            "nll": float(nll_hat),
            "k": k_par,
            "aicc": float(aicc_val),
            "theta": theta_hat,
            "se_theta": se_theta,
            "cov_theta": cov_theta,
            "pars": pars_hat,
            "w_hat": w_hat,
            "u_hat": u_hat,
            "spc_slopes": spc_slopes,
            "spc_slopes_se": spc_slopes_se,
            "spc_slopes_cond_se": spc_slopes_cond_se,
        }
