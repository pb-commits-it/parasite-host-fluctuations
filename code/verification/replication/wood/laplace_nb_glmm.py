#!/usr/bin/env python3
"""Independent Laplace-approximated Negative Binomial GLMM solver.

Written from scratch for r36-rep-wood without importing or copying any code from
casestudies/wood/.

Mathematical specification (Wood et al. 2023, PNAS; protocol.md §4.1-4.2):
- Observations y_i ~ NegBin2(mu_i, phi_{s(i)}), Var(y_i) = mu_i + mu_i^2 / phi_{s(i)},
  with log(mu_i) = eta_i and log(phi_s) = psi_s.
- Random effects are represented in standardized coordinates w ~ N(0, I_d),
  u = mu_u(theta) + sigma_u(theta) * w.
- Inner mode w_hat(theta) solves grad_w f(theta, w) = 0 via damped Newton using
  the exact analytical inner Hessian and exact Schur complement over the diagonal
  specimen random-effect block.
- Outer marginal Laplace objective:
    L(theta) = nll_data(theta, w_hat) + 0.5 * ||w_hat||^2 + 0.5 * log det H_ww(theta, w_hat)
- Outer gradient grad_theta L(theta) is computed to machine precision via exact
  implicit differentiation (PyTorch float64 autodiff + Schur adjoint solve),
  avoiding finite-difference inner-mode perturbations during optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import linalg, optimize, sparse, special, stats

torch.set_num_threads(1)


@dataclass
class NBGLMMDataset:
    y: np.ndarray            # (n,) float64 counts
    hp_idx: np.ndarray       # (n,) int64 in 0..n_hp-1 (host-parasite pair)
    spc_idx: np.ndarray      # (n,) int64 in 0..n_spc-1 (parasite taxon code)
    fish_idx: np.ndarray     # (n,) int64 in 0..n_fish-1 (individual fish specimen)
    tax_spc: np.ndarray      # (n_spc,) int64 in 0..n_tax-1 (taxonomic group of spc)
    tax_obs: np.ndarray      # (n,) int64 in 0..n_tax-1 (taxonomic group of obs)
    nh_obs: np.ndarray       # (n,) int64 in 0..2 (obligate host-count group 1, 2, 3+)
    slat: np.ndarray         # (n,) float64 standardized latitude (NA -> 0)
    slen: np.ndarray         # (n,) float64 standardized fish length
    uyear: np.ndarray        # (n,) float64 standardized collection year
    n_hp: int
    n_spc: int
    n_fish: int
    n_tax: int


def nb2_nll_and_derivs_np(y: np.ndarray, eta: np.ndarray, psi: np.ndarray):
    """Return (-sum log P(y | mu=exp(eta), phi=exp(psi)), g_eta, g_psi, h_ee, h_ep, h_pp)
    where g_* and h_* are 1st and 2nd derivatives of the NEGATIVE log-likelihood
    per observation with respect to eta and psi.
    """
    if np.max(np.abs(eta)) > 180.0 or np.max(np.abs(psi)) > 40.0:
        raise FloatingPointError("Linear predictor or log-dispersion out of range")
    mu = np.exp(eta)
    phi = np.exp(psi)
    denom = mu + phi
    p = mu / denom
    q = phi / denom
    log_denom = np.logaddexp(eta, psi)
    log_q = psi - log_denom

    logpmf = (
        special.gammaln(y + phi)
        - special.gammaln(phi)
        - special.gammaln(y + 1.0)
        + phi * log_q
        + y * (eta - log_denom)
    )
    nll = -float(np.sum(logpmf))

    # First derivatives of -logpmf
    g_eta = (mu - y) * q
    dig = special.digamma(y + phi) - special.digamma(phi)
    a_term = dig + log_q + (mu - y) / denom
    g_psi = -phi * a_term

    # Second derivatives of -logpmf
    pq = p * q
    h_ee = (y + phi) * pq
    h_ep = (mu - y) * pq
    trig = special.polygamma(1, y + phi) - special.polygamma(1, phi)
    da_dphi = trig + (1.0 / phi) - (1.0 / denom) - (mu - y) / (denom * denom)
    h_pp = g_psi - (phi * phi) * da_dphi
    return nll, g_eta, g_psi, h_ee, h_ep, h_pp


class IndependentLaplaceNBGLMM:
    """Hierarchical Negative Binomial GLMM fitted by exact Laplace approximation."""

    def __init__(
        self,
        data: NBGLMMDataset,
        has_year: bool = True,
        X_slope: Optional[np.ndarray] = None,
        group_int_obs: Optional[np.ndarray] = None,
        n_group_int: int = 0,
        has_tax_slope_re: bool = False,
        X_env_obs: Optional[np.ndarray] = None,
        sigma_log_lower: float = -7.0,
        sigma_log_upper: float = 2.0,
    ):
        self.d = data
        self.n_obs = int(len(data.y))
        self.has_year = bool(has_year)
        self.has_bog = group_int_obs is not None and n_group_int > 0
        self.group_int_obs = None if not self.has_bog else np.asarray(group_int_obs, dtype=np.int64)
        self.n_bog = int(n_group_int) if self.has_bog else 0
        self.has_tax_re = bool(has_tax_slope_re and self.has_year)
        self.sigma_bounds = (float(sigma_log_lower), float(sigma_log_upper))

        if self.has_year:
            if X_slope is None:
                raise ValueError("X_slope must be provided when has_year=True")
            self.X_slope = np.asarray(X_slope, dtype=np.float64)
            assert self.X_slope.shape[0] == data.n_spc
            self.p_slope = int(self.X_slope.shape[1])
        else:
            self.X_slope = np.zeros((data.n_spc, 0), dtype=np.float64)
            self.p_slope = 0

        if X_env_obs is not None:
            self.X_env = np.asarray(X_env_obs, dtype=np.float64)
            assert self.X_env.shape[0] == self.n_obs
            self.p_env = int(self.X_env.shape[1])
        else:
            self.X_env = np.zeros((self.n_obs, 0), dtype=np.float64)
            self.p_env = 0

        # Layout of standardized random-effect vector w = [w_v, w_z]
        pos = 0
        self.sl_bo = slice(pos, pos + data.n_hp); pos += data.n_hp
        self.sl_bog = slice(pos, pos + self.n_bog); pos += self.n_bog
        self.sl_blat = slice(pos, pos + data.n_spc); pos += data.n_spc
        self.sl_blen = slice(pos, pos + data.n_hp); pos += data.n_hp
        self.sl_g = slice(pos, pos + (data.n_spc if self.has_year else 0))
        pos += (data.n_spc if self.has_year else 0)
        self.sl_atax = slice(pos, pos + (data.n_tax if self.has_tax_re else 0))
        pos += (data.n_tax if self.has_tax_re else 0)
        self.dim_eta_v = pos

        self.sl_psi = slice(pos, pos + data.n_spc); pos += data.n_spc
        self.dim_v = pos

        self.sl_z = slice(pos, pos + data.n_fish); pos += data.n_fish
        self.dim_w = pos

        # Sparse observation design matrices for fast inner Newton steps
        obs_ar = np.arange(self.n_obs, dtype=np.int64)
        r_list, c_list, v_list = [], [], []

        def _append_design(offset: int, idx_vec: np.ndarray, wt):
            r_list.append(obs_ar)
            c_list.append(offset + idx_vec)
            if np.ndim(wt) == 0:
                v_list.append(np.full(self.n_obs, float(wt), dtype=np.float64))
            else:
                v_list.append(np.asarray(wt, dtype=np.float64))

        _append_design(self.sl_bo.start, data.hp_idx, 1.0)
        if self.has_bog:
            _append_design(self.sl_bog.start, self.group_int_obs, 1.0)
        _append_design(self.sl_blat.start, data.spc_idx, data.slat)
        _append_design(self.sl_blen.start, data.hp_idx, data.slen)
        if self.has_year:
            _append_design(self.sl_g.start, data.spc_idx, data.uyear)
        if self.has_tax_re:
            _append_design(self.sl_atax.start, data.tax_obs, data.uyear)

        self.M_eta_v = sparse.csr_matrix(
            (np.concatenate(v_list), (np.concatenate(r_list), np.concatenate(c_list))),
            shape=(self.n_obs, self.dim_eta_v),
        )
        self.M_psi = sparse.csr_matrix(
            (np.ones(self.n_obs, dtype=np.float64), (obs_ar, data.spc_idx)),
            shape=(self.n_obs, data.n_spc),
        )
        self.M_z = sparse.csr_matrix(
            (np.ones(self.n_obs, dtype=np.float64), (obs_ar, data.fish_idx)),
            shape=(self.n_obs, data.n_fish),
        )

        # Torch tensors for exact autodiff outer gradient
        self._y_t = torch.tensor(data.y, dtype=torch.float64)
        self._hp_t = torch.tensor(data.hp_idx, dtype=torch.long)
        self._spc_t = torch.tensor(data.spc_idx, dtype=torch.long)
        self._fish_t = torch.tensor(data.fish_idx, dtype=torch.long)
        self._tax_obs_t = torch.tensor(data.tax_obs, dtype=torch.long)
        self._bog_obs_t = None if not self.has_bog else torch.tensor(self.group_int_obs, dtype=torch.long)
        self._slat_t = torch.tensor(data.slat, dtype=torch.float64)
        self._slen_t = torch.tensor(data.slen, dtype=torch.float64)
        self._uyear_t = torch.tensor(data.uyear, dtype=torch.float64)
        self._X_slope_t = torch.tensor(self.X_slope, dtype=torch.float64)
        self._X_env_t = torch.tensor(self.X_env, dtype=torch.float64)
        self._M_eta_v_dense_t = torch.tensor(self.M_eta_v.toarray(), dtype=torch.float64)
        self._M_psi_dense_t = torch.tensor(self.M_psi.toarray(), dtype=torch.float64)
        self._M_z_dense_t = torch.tensor(self.M_z.toarray(), dtype=torch.float64)

        self.param_names = self._build_param_spec()
        self.n_theta = len(self.param_names)
        self._w_mode_cache = np.zeros(self.dim_w, dtype=np.float64)

    def _build_param_spec(self) -> list[str]:
        names = ["bobar", "logsigma_bo"]
        if self.has_bog:
            names.append("logsigma_bog")
        names.extend([
            "blatbar",
            "logsigma_blat",
            "blengthbar",
            "logsigma_blength",
            "logsigma_z",
            "logphibar",
            "loglogsigma_phi",
        ])
        if self.has_year:
            names.extend([f"beta_slope_{j}" for j in range(self.p_slope)])
            names.append("logsigma_g")
        if self.has_tax_re:
            names.append("logsigma_tax")
        if self.p_env > 0:
            names.extend([f"beta_env_{j}" for j in range(self.p_env)])
        return names

    def default_theta0(self) -> np.ndarray:
        th = [-1.1, 0.35]
        if self.has_bog:
            th.append(-0.8)
        th.extend([-0.12, -0.84, 0.50, -0.46, -0.78, -1.68, -0.18])
        if self.has_year:
            th.extend([0.0] * self.p_slope)
            th.append(-0.58)
        if self.has_tax_re:
            th.append(-2.0)
        if self.p_env > 0:
            th.extend([0.0] * self.p_env)
        return np.array(th, dtype=np.float64)

    def unpack_theta(self, theta: np.ndarray) -> dict:
        i = 0
        p = {}
        p["bobar"] = float(theta[i]); i += 1
        p["logsigma_bo"] = float(theta[i]); i += 1
        if self.has_bog:
            p["logsigma_bog"] = float(theta[i]); i += 1
        p["blatbar"] = float(theta[i]); i += 1
        p["logsigma_blat"] = float(theta[i]); i += 1
        p["blengthbar"] = float(theta[i]); i += 1
        p["logsigma_blength"] = float(theta[i]); i += 1
        p["logsigma_z"] = float(theta[i]); i += 1
        p["logphibar"] = float(theta[i]); i += 1
        p["loglogsigma_phi"] = float(theta[i]); i += 1
        if self.has_year:
            p["beta_slope"] = np.asarray(theta[i:i + self.p_slope], dtype=np.float64)
            i += self.p_slope
            p["logsigma_g"] = float(theta[i]); i += 1
        if self.has_tax_re:
            p["logsigma_tax"] = float(theta[i]); i += 1
        if self.p_env > 0:
            p["beta_env"] = np.asarray(theta[i:i + self.p_env], dtype=np.float64)
            i += self.p_env
        assert i == len(theta)
        return p

    def _u_transform_np(self, p: dict):
        mu_u = np.zeros(self.dim_w, dtype=np.float64)
        sig_u = np.ones(self.dim_w, dtype=np.float64)
        if self.has_bog:
            mu_u[self.sl_bo] = 0.0
            sig_u[self.sl_bo] = math.exp(p["logsigma_bo"])
            mu_u[self.sl_bog] = p["bobar"]
            sig_u[self.sl_bog] = math.exp(p["logsigma_bog"])
        else:
            mu_u[self.sl_bo] = p["bobar"]
            sig_u[self.sl_bo] = math.exp(p["logsigma_bo"])
        mu_u[self.sl_blat] = p["blatbar"]
        sig_u[self.sl_blat] = math.exp(p["logsigma_blat"])
        mu_u[self.sl_blen] = p["blengthbar"]
        sig_u[self.sl_blen] = math.exp(p["logsigma_blength"])
        if self.has_year:
            sig_u[self.sl_g] = math.exp(p["logsigma_g"])
        if self.has_tax_re:
            sig_u[self.sl_atax] = math.exp(p["logsigma_tax"])
        mu_u[self.sl_psi] = p["logphibar"]
        sig_u[self.sl_psi] = math.exp(p["loglogsigma_phi"])
        sig_u[self.sl_z] = math.exp(p["logsigma_z"])
        return mu_u, sig_u

    def _inner_eval_np(self, w: np.ndarray, p: dict):
        """Return f_joint = nll_data + 0.5*||w||^2, grad_w, H_vv, H_vz, diag_zz."""
        mu_u, sig_u = self._u_transform_np(p)
        u = mu_u + sig_u * w

        eta = self.M_eta_v @ u[:self.dim_eta_v] + u[self.sl_z][self.d.fish_idx]
        if self.has_year and self.p_slope > 0:
            eta = eta + (self.X_slope @ p["beta_slope"])[self.d.spc_idx] * self.d.uyear
        if self.p_env > 0:
            eta = eta + self.X_env @ p["beta_env"]
        psi = u[self.sl_psi][self.d.spc_idx]

        nll_data, g_eta, g_psi, h_ee, h_ep, h_pp = nb2_nll_and_derivs_np(self.d.y, eta, psi)
        f_joint = nll_data + 0.5 * float(w @ w)

        grad_u = np.empty(self.dim_w, dtype=np.float64)
        grad_u[:self.dim_eta_v] = self.M_eta_v.T @ g_eta
        grad_u[self.sl_psi] = np.bincount(self.d.spc_idx, weights=g_psi, minlength=self.d.n_spc)
        grad_u[self.sl_z] = np.bincount(self.d.fish_idx, weights=g_eta, minlength=self.d.n_fish)
        grad_w = w + sig_u * grad_u

        sig_v = sig_u[:self.dim_v]
        sig_z = sig_u[self.sl_z]

        H_uu_vv = np.zeros((self.dim_v, self.dim_v), dtype=np.float64)
        D_ee = sparse.diags(h_ee)
        D_ep = sparse.diags(h_ep)
        H_uu_vv[:self.dim_eta_v, :self.dim_eta_v] = (self.M_eta_v.T @ D_ee @ self.M_eta_v).toarray()
        H_ep_blk = (self.M_eta_v.T @ D_ep @ self.M_psi).toarray()
        H_uu_vv[:self.dim_eta_v, self.sl_psi] = H_ep_blk
        H_uu_vv[self.sl_psi, :self.dim_eta_v] = H_ep_blk.T
        H_pp_diag = np.bincount(self.d.spc_idx, weights=h_pp, minlength=self.d.n_spc)
        H_uu_vv[self.sl_psi, self.sl_psi] = np.diag(H_pp_diag)

        H_vv = (sig_v[:, None] * H_uu_vv) * sig_v[None, :]
        H_vv[np.diag_indices(self.dim_v)] += 1.0

        H_uu_vz = np.zeros((self.dim_v, self.d.n_fish), dtype=np.float64)
        H_uu_vz[:self.dim_eta_v, :] = (self.M_eta_v.T @ D_ee @ self.M_z).toarray()
        H_uu_vz[self.sl_psi, :] = (self.M_psi.T @ D_ep @ self.M_z).toarray()
        H_vz = (sig_v[:, None] * H_uu_vz) * sig_z[None, :]

        diag_zz = 1.0 + (sig_z * sig_z) * np.bincount(
            self.d.fish_idx, weights=h_ee, minlength=self.d.n_fish
        )
        return f_joint, grad_w, H_vv, H_vz, diag_zz

    def _schur_factor_and_solve(
        self,
        rhs_w: np.ndarray,
        H_vv: np.ndarray,
        H_vz: np.ndarray,
        diag_zz: np.ndarray,
        damping: float = 0.0,
    ):
        dz_damped = np.maximum(diag_zz + damping, 1e-12)
        inv_dz = 1.0 / dz_damped
        S = H_vv - (H_vz * inv_dz[None, :]) @ H_vz.T
        if damping > 0.0:
            S = S.copy()
            S[np.diag_indices(self.dim_v)] += damping
        chol_S = linalg.cho_factor(S, lower=True, check_finite=False)
        rv = rhs_w[:self.dim_v]
        rz = rhs_w[self.sl_z]
        sol_v = linalg.cho_solve(chol_S, rv - H_vz @ (rz * inv_dz), check_finite=False)
        sol_z = (rz - H_vz.T @ sol_v) * inv_dz
        sol_w = np.concatenate([sol_v, sol_z])
        logdet = float(np.sum(np.log(dz_damped))) + 2.0 * float(np.sum(np.log(np.diag(chol_S[0]))))
        return sol_w, chol_S, logdet, inv_dz

    def solve_inner_mode(
        self,
        theta: np.ndarray,
        w_start: Optional[np.ndarray] = None,
        grad_tol: float = 1e-10,
        max_iter: int = 80,
    ):
        p = self.unpack_theta(theta)
        w = (self._w_mode_cache if w_start is None else w_start).copy()
        f_val, grad_w, H_vv, H_vz, diag_zz = self._inner_eval_np(w, p)

        for _ in range(max_iter):
            if float(np.max(np.abs(grad_w))) <= grad_tol:
                break
            damping = 0.0
            for _ in range(22):
                try:
                    neg_step, _, _, _ = self._schur_factor_and_solve(
                        grad_w, H_vv, H_vz, diag_zz, damping=damping
                    )
                    step = -neg_step
                    break
                except linalg.LinAlgError:
                    damping = 1e-5 if damping == 0.0 else damping * 10.0
            else:
                raise RuntimeError("Failed to factorize inner Schur Hessian")

            dir_deriv = float(grad_w @ step)
            if dir_deriv >= 0.0:
                step = -grad_w
                dir_deriv = -float(grad_w @ grad_w)

            step_scale = 1.0
            for _ in range(28):
                w_cand = w + step_scale * step
                try:
                    f_c, g_c, Hvv_c, Hvz_c, dzz_c = self._inner_eval_np(w_cand, p)
                except FloatingPointError:
                    step_scale *= 0.5
                    continue
                if math.isfinite(f_c) and f_c <= f_val + 1e-4 * step_scale * dir_deriv + 1e-11:
                    w, f_val, grad_w, H_vv, H_vz, diag_zz = w_cand, f_c, g_c, Hvv_c, Hvz_c, dzz_c
                    break
                step_scale *= 0.5
            else:
                raise RuntimeError(f"Inner line search stalled at |grad|={np.max(np.abs(grad_w)):.3e}")

        gnorm = float(np.max(np.abs(grad_w)))
        if gnorm > 1e-6:
            raise RuntimeError(f"Inner Newton failed to converge: |grad|={gnorm:.3e}")

        _, chol_S, logdet_H, inv_dz = self._schur_factor_and_solve(
            grad_w, H_vv, H_vz, diag_zz, damping=0.0
        )
        laplace_nll = f_val + 0.5 * logdet_H
        return laplace_nll, w, H_vv, H_vz, diag_zz, chol_S, inv_dz, gnorm

    def _torch_joint_and_half_logdet(self, th_t: torch.Tensor, w_t: torch.Tensor):
        """Differentiable PyTorch graph for f(theta, w), grad_w f(theta, w), and 0.5*log|H_ww|."""
        i = 0
        bobar = th_t[i]; i += 1
        sig_bo = torch.exp(th_t[i]); i += 1
        if self.has_bog:
            sig_bog = torch.exp(th_t[i]); i += 1
        blatbar = th_t[i]; i += 1
        sig_blat = torch.exp(th_t[i]); i += 1
        blenbar = th_t[i]; i += 1
        sig_blen = torch.exp(th_t[i]); i += 1
        sig_z = torch.exp(th_t[i]); i += 1
        logphibar = th_t[i]; i += 1
        sig_psi = torch.exp(th_t[i]); i += 1
        if self.has_year:
            beta_slope = th_t[i:i + self.p_slope]; i += self.p_slope
            sig_g = torch.exp(th_t[i]); i += 1
        if self.has_tax_re:
            sig_tax = torch.exp(th_t[i]); i += 1
        if self.p_env > 0:
            beta_env = th_t[i:i + self.p_env]; i += self.p_env

        mu_parts = []
        sig_parts = []
        if self.has_bog:
            mu_parts.append(torch.zeros(self.d.n_hp, dtype=torch.float64))
            sig_parts.append(sig_bo.expand(self.d.n_hp))
            mu_parts.append(bobar.expand(self.n_bog))
            sig_parts.append(sig_bog.expand(self.n_bog))
        else:
            mu_parts.append(bobar.expand(self.d.n_hp))
            sig_parts.append(sig_bo.expand(self.d.n_hp))

        mu_parts.append(blatbar.expand(self.d.n_spc))
        sig_parts.append(sig_blat.expand(self.d.n_spc))
        mu_parts.append(blenbar.expand(self.d.n_hp))
        sig_parts.append(sig_blen.expand(self.d.n_hp))

        if self.has_year:
            mu_parts.append(torch.zeros(self.d.n_spc, dtype=torch.float64))
            sig_parts.append(sig_g.expand(self.d.n_spc))
        if self.has_tax_re:
            mu_parts.append(torch.zeros(self.d.n_tax, dtype=torch.float64))
            sig_parts.append(sig_tax.expand(self.d.n_tax))

        mu_parts.append(logphibar.expand(self.d.n_spc))
        sig_parts.append(sig_psi.expand(self.d.n_spc))
        mu_parts.append(torch.zeros(self.d.n_fish, dtype=torch.float64))
        sig_parts.append(sig_z.expand(self.d.n_fish))

        mu_u = torch.cat(mu_parts)
        sig_u = torch.cat(sig_parts)
        u = mu_u + sig_u * w_t

        eta = self._M_eta_v_dense_t @ u[:self.dim_eta_v] + u[self.sl_z][self._fish_t]
        if self.has_year and self.p_slope > 0:
            eta = eta + (self._X_slope_t @ beta_slope)[self._spc_t] * self._uyear_t
        if self.p_env > 0:
            eta = eta + self._X_env_t @ beta_env
        psi = u[self.sl_psi][self._spc_t]

        mu = torch.exp(eta)
        phi = torch.exp(psi)
        denom = mu + phi
        p = mu / denom
        q = phi / denom
        log_denom = torch.logaddexp(eta, psi)
        log_q = psi - log_denom

        logpmf = (
            torch.lgamma(self._y_t + phi)
            - torch.lgamma(phi)
            - torch.lgamma(self._y_t + 1.0)
            + phi * log_q
            + self._y_t * (eta - log_denom)
        )
        f_joint = -torch.sum(logpmf) + 0.5 * torch.dot(w_t, w_t)

        g_eta = (mu - self._y_t) * q
        dig = torch.digamma(self._y_t + phi) - torch.digamma(phi)
        a_term = dig + log_q + (mu - self._y_t) / denom
        g_psi = -phi * a_term

        grad_u_eta_v = self._M_eta_v_dense_t.T @ g_eta
        grad_u_psi = self._M_psi_dense_t.T @ g_psi
        grad_u_z = self._M_z_dense_t.T @ g_eta
        grad_u = torch.cat([grad_u_eta_v, grad_u_psi, grad_u_z])
        grad_w = w_t + sig_u * grad_u

        pq = p * q
        h_ee = (self._y_t + phi) * pq
        h_ep = (mu - self._y_t) * pq
        trig = torch.polygamma(1, self._y_t + phi) - torch.polygamma(1, phi)
        da_dphi = trig + (1.0 / phi) - (1.0 / denom) - (mu - self._y_t) / (denom * denom)
        h_pp = g_psi - (phi * phi) * da_dphi

        H_ev_ev = self._M_eta_v_dense_t.T @ (h_ee[:, None] * self._M_eta_v_dense_t)
        H_ev_ps = self._M_eta_v_dense_t.T @ (h_ep[:, None] * self._M_psi_dense_t)
        H_ps_ps = torch.diag(self._M_psi_dense_t.T @ h_pp)

        top = torch.cat([H_ev_ev, H_ev_ps], dim=1)
        bot = torch.cat([H_ev_ps.T, H_ps_ps], dim=1)
        H_uu_vv = torch.cat([top, bot], dim=0)

        sig_v = sig_u[:self.dim_v]
        sig_z_vec = sig_u[self.sl_z]
        H_vv = (sig_v[:, None] * H_uu_vv) * sig_v[None, :] + torch.eye(self.dim_v, dtype=torch.float64)

        H_ev_z = self._M_eta_v_dense_t.T @ (h_ee[:, None] * self._M_z_dense_t)
        H_ps_z = self._M_psi_dense_t.T @ (h_ep[:, None] * self._M_z_dense_t)
        H_uu_vz = torch.cat([H_ev_z, H_ps_z], dim=0)
        H_vz = (sig_v[:, None] * H_uu_vz) * sig_z_vec[None, :]

        d_zz = 1.0 + (sig_z_vec * sig_z_vec) * (self._M_z_dense_t.T @ h_ee)
        inv_dzz = 1.0 / d_zz
        S = H_vv - (H_vz * inv_dzz[None, :]) @ H_vz.T
        L_S = torch.linalg.cholesky(S)
        half_logdet = 0.5 * torch.sum(torch.log(d_zz)) + torch.sum(torch.log(torch.diagonal(L_S)))
        return f_joint, grad_w, half_logdet

    def laplace_nll_and_exact_grad(
        self,
        theta: np.ndarray,
        w_init: Optional[np.ndarray] = None,
        update_cache: bool = True,
    ):
        """Compute exact Laplace NLL and exact analytical outer gradient wrt theta."""
        if np.any(np.abs(theta) > 15.0):
            return 1e12, np.zeros_like(theta)
        try:
            nll_val, w_hat, H_vv, H_vz, diag_zz, chol_S, inv_dz, _ = self.solve_inner_mode(
                theta, w_start=w_init, grad_tol=1e-12
            )
        except Exception:
            return 1e12, np.zeros_like(theta)

        if not math.isfinite(nll_val):
            return 1e12, np.zeros_like(theta)
        if update_cache:
            self._w_mode_cache = w_hat.copy()

        # Step 1: d(0.5*log|H_ww|)/dw at w_hat
        th_const = torch.tensor(theta, dtype=torch.float64, requires_grad=False)
        w_var = torch.tensor(w_hat, dtype=torch.float64, requires_grad=True)
        _, _, half_ld_w = self._torch_joint_and_half_logdet(th_const, w_var)
        g_ld_w = torch.autograd.grad(half_ld_w, w_var)[0].detach().numpy()

        # Step 2: Adjoint solve H_ww * v_adj = g_ld_w using already-factored Schur complement
        rv = g_ld_w[:self.dim_v]
        rz = g_ld_w[self.sl_z]
        v_adj_v = linalg.cho_solve(chol_S, rv - H_vz @ (rz * inv_dz), check_finite=False)
        v_adj_z = (rz - H_vz.T @ v_adj_v) * inv_dz
        v_adj_t = torch.tensor(np.concatenate([v_adj_v, v_adj_z]), dtype=torch.float64)

        # Step 3: Exact implicit-function gradient wrt theta in a single backward pass
        th_var = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        w_const = torch.tensor(w_hat, dtype=torch.float64, requires_grad=False)
        f_joint, grad_w, half_ld = self._torch_joint_and_half_logdet(th_var, w_const)
        exact_obj = f_joint + half_ld
        adj_obj = exact_obj - torch.dot(grad_w, v_adj_t)
        grad_theta = torch.autograd.grad(adj_obj, th_var)[0].detach().numpy()
        return float(exact_obj.item()), grad_theta

    def fit(self, theta0: Optional[np.ndarray] = None, maxiter: int = 250):
        x0 = self.default_theta0() if theta0 is None else np.asarray(theta0, dtype=np.float64).copy()
        # Prime inner cache
        _ = self.laplace_nll_and_exact_grad(x0, update_cache=True)

        bounds = []
        for nm in self.param_names:
            if "logsigma" in nm:
                bounds.append(self.sigma_bounds)
            else:
                bounds.append((-10.0, 10.0))

        opt = optimize.minimize(
            lambda th: self.laplace_nll_and_exact_grad(th, update_cache=True),
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-6, "maxls": 40},
        )
        theta_hat = opt.x.copy()
        nll_hat, exact_grad_hat = self.laplace_nll_and_exact_grad(theta_hat, update_cache=True)
        _, w_hat, H_vv, H_vz, diag_zz, chol_S, inv_dz, inner_gnorm = self.solve_inner_mode(
            theta_hat, w_start=self._w_mode_cache, grad_tol=1e-11
        )
        self._w_mode_cache = w_hat.copy()

        # Projected gradient norm respecting active box bounds
        proj_grad = exact_grad_hat.copy()
        for j, (lo, hi) in enumerate(bounds):
            if theta_hat[j] <= lo + 1e-9 and proj_grad[j] > 0.0:
                proj_grad[j] = 0.0
            if theta_hat[j] >= hi - 1e-9 and proj_grad[j] < 0.0:
                proj_grad[j] = 0.0
        proj_grad_max = float(np.max(np.abs(proj_grad)))

        # Outer Hessian via central differences of the EXACT analytical gradient
        eps_h = 1e-4
        H_theta = np.zeros((self.n_theta, self.n_theta), dtype=np.float64)
        for j in range(self.n_theta):
            tp = theta_hat.copy(); tp[j] += eps_h
            tm = theta_hat.copy(); tm[j] -= eps_h
            _, gp = self.laplace_nll_and_exact_grad(tp, w_init=w_hat, update_cache=False)
            _, gm = self.laplace_nll_and_exact_grad(tm, w_init=w_hat, update_cache=False)
            H_theta[:, j] = (gp - gm) / (2.0 * eps_h)
        H_theta = 0.5 * (H_theta + H_theta.T)
        evals, evecs = linalg.eigh(H_theta)
        cov_theta = (evecs / np.maximum(evals, 1e-6)) @ evecs.T
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))

        pars_hat = self.unpack_theta(theta_hat)

        # Taxon-specific temporal slopes and TMB sdreport total standard errors
        spc_slopes = None
        spc_slopes_se = None
        if self.has_year:
            def _compute_slopes(th_vec: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
                pr = self.unpack_theta(th_vec)
                _, sg = self._u_transform_np(pr)
                uv = sg * w_vec
                base_s = self.X_slope @ pr["beta_slope"] if self.p_slope > 0 else np.zeros(self.d.n_spc)
                re_s = uv[self.sl_g]
                tax_s = uv[self.sl_atax][self.d.tax_spc] if self.has_tax_re else 0.0
                return base_s + re_s + tax_s

            spc_slopes = _compute_slopes(theta_hat, w_hat)
            S_inv = linalg.cho_solve(chol_S, np.eye(self.dim_v), check_finite=False)
            L_map = np.zeros((self.d.n_spc, self.dim_v), dtype=np.float64)
            L_map[np.arange(self.d.n_spc), np.arange(self.sl_g.start, self.sl_g.stop)] = math.exp(pars_hat["logsigma_g"])
            if self.has_tax_re:
                L_map[np.arange(self.d.n_spc), self.sl_atax.start + self.d.tax_spc] = math.exp(pars_hat["logsigma_tax"])

            # Exact implicit Jacobian d(spc_slopes)/d(theta) via exact dw_hat/dtheta = -H_ww^{-1} d^2f/(dw dtheta)
            th_var = torch.tensor(theta_hat, dtype=torch.float64, requires_grad=True)
            w_const = torch.tensor(w_hat, dtype=torch.float64, requires_grad=False)
            _, grad_w_t, _ = self._torch_joint_and_half_logdet(th_var, w_const)
            D_w_th = np.zeros((self.dim_w, self.n_theta), dtype=np.float64)
            for r_idx in range(self.dim_w):
                g_th = torch.autograd.grad(grad_w_t[r_idx], th_var, retain_graph=(r_idx < self.dim_w - 1))[0]
                D_w_th[r_idx, :] = g_th.detach().numpy()

            rv_mat = -D_w_th[:self.dim_v, :]
            rz_mat = -D_w_th[self.sl_z, :]
            dw_v_dth = linalg.cho_solve(chol_S, rv_mat - H_vz @ (rz_mat * inv_dz[:, None]), check_finite=False)

            # Direct partial d(spc_slopes)/d(theta) at fixed w_hat
            J_direct = np.zeros((self.d.n_spc, self.n_theta), dtype=np.float64)
            for j, nm in enumerate(self.param_names):
                if nm.startswith("beta_slope_"):
                    col_j = int(nm.split("_")[-1])
                    J_direct[:, j] = self.X_slope[:, col_j]
                elif nm == "logsigma_g":
                    J_direct[:, j] = math.exp(pars_hat["logsigma_g"]) * w_hat[self.sl_g]
                elif nm == "logsigma_tax":
                    J_direct[:, j] = math.exp(pars_hat["logsigma_tax"]) * w_hat[self.sl_atax][self.d.tax_spc]

            J_total = J_direct + L_map @ dw_v_dth
            cov_slopes = L_map @ S_inv @ L_map.T + J_total @ cov_theta @ J_total.T
            spc_slopes_se = np.sqrt(np.maximum(np.diag(cov_slopes), 1e-14))

        k = self.n_theta
        n = self.n_obs
        aicc = 2.0 * nll_hat + 2.0 * k + (2.0 * k * (k + 1.0)) / (n - k - 1.0)

        return {
            "converged": bool(opt.success or proj_grad_max < 1e-4),
            "optimizer_success": bool(opt.success),
            "message": str(opt.message),
            "nit": int(opt.nit),
            "nll": float(nll_hat),
            "k": int(k),
            "n_obs": int(n),
            "aicc": float(aicc),
            "outer_grad_max": float(np.max(np.abs(exact_grad_hat))),
            "projected_grad_max": proj_grad_max,
            "inner_grad_max": inner_gnorm,
            "hessian_min_eig": float(evals.min()),
            "param_names": self.param_names,
            "theta": theta_hat,
            "se_theta": se_theta,
            "cov_theta": cov_theta,
            "pars": pars_hat,
            "w_hat": w_hat,
            "spc_slopes": spc_slopes,
            "spc_slopes_se": spc_slopes_se,
        }


def run_rule7_glmm_validation() -> dict:
    """Rule 7 validation of NB likelihood, derivatives, and exact implicit Laplace gradient
    against independent SciPy distributions and central finite differences BEFORE any data fit.
    Preset tolerances:
      - NB2 log-PMF vs scipy.stats.nbinom.logpmf: < 1e-12
      - Analytical 1st & 2nd NB2 derivatives vs central finite differences: < 1e-8
      - Exact implicit-differentiation Laplace outer gradient vs central finite differences: < 1e-8
    """
    y = np.array([0.0, 1.0, 3.0, 17.0, 42.0])
    eta = np.array([-1.8, -0.3, 0.7, 2.1, 3.3])
    psi = np.array([-1.2, -0.1, 0.6, 1.4, -0.5])
    nll, g_e, g_p, h_ee, h_ep, h_pp = nb2_nll_and_derivs_np(y, eta, psi)

    mu = np.exp(eta)
    phi = np.exp(psi)
    scipy_nll = -float(np.sum(stats.nbinom.logpmf(y, phi, phi / (phi + mu))))
    err_nll = abs(nll - scipy_nll)

    eps = 1e-6
    fd_errs = []
    for j in range(len(y)):
        de = np.zeros_like(eta); de[j] = eps
        dp = np.zeros_like(psi); dp[j] = eps
        ep = nb2_nll_and_derivs_np(y, eta + de, psi)
        em = nb2_nll_and_derivs_np(y, eta - de, psi)
        pp = nb2_nll_and_derivs_np(y, eta, psi + dp)
        pm = nb2_nll_and_derivs_np(y, eta, psi - dp)
        fd_errs.extend([
            abs((ep[0] - em[0]) / (2 * eps) - g_e[j]),
            abs((pp[0] - pm[0]) / (2 * eps) - g_p[j]),
            abs((ep[1][j] - em[1][j]) / (2 * eps) - h_ee[j]),
            abs((pp[1][j] - pm[1][j]) / (2 * eps) - h_ep[j]),
            abs((pp[2][j] - pm[2][j]) / (2 * eps) - h_pp[j]),
        ])
    max_nb_deriv_err = float(max(fd_errs))

    # Synthetic hierarchical NB dataset to test exact implicit Laplace outer gradient
    rng = np.random.default_rng(20261115)
    n_spc, n_hp, n_fish, n_tax = 6, 8, 20, 3
    n_obs = 80
    spc_idx = rng.integers(0, n_spc, size=n_obs)
    hp_idx = rng.integers(0, n_hp, size=n_obs)
    fish_idx = rng.integers(0, n_fish, size=n_obs)
    tax_spc = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    nh_spc = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    ds = NBGLMMDataset(
        y=rng.poisson(2.2, size=n_obs).astype(np.float64),
        hp_idx=hp_idx,
        spc_idx=spc_idx,
        fish_idx=fish_idx,
        tax_spc=tax_spc,
        tax_obs=tax_spc[spc_idx],
        nh_obs=nh_spc[spc_idx],
        slat=rng.normal(size=n_obs),
        slen=rng.normal(size=n_obs),
        uyear=rng.normal(size=n_obs),
        n_hp=n_hp,
        n_spc=n_spc,
        n_fish=n_fish,
        n_tax=n_tax,
    )
    X_sl = np.column_stack([np.ones(n_spc), rng.normal(size=n_spc)])
    X_ev = rng.normal(size=(n_obs, 2))
    mdl = IndependentLaplaceNBGLMM(
        ds,
        has_year=True,
        X_slope=X_sl,
        group_int_obs=ds.nh_obs,
        n_group_int=3,
        has_tax_slope_re=True,
        X_env_obs=X_ev,
    )
    th_test = mdl.default_theta0() + rng.normal(0.0, 0.15, size=mdl.n_theta)
    nll_exact, grad_exact = mdl.laplace_nll_and_exact_grad(th_test, update_cache=True)
    w_ref = mdl._w_mode_cache.copy()
    nll_scipy_mode, _, _, _, _, _, _, _ = mdl.solve_inner_mode(th_test, w_start=w_ref, grad_tol=1e-13)
    err_torch_scipy_nll = abs(nll_exact - nll_scipy_mode)

    eps_th = 1e-4
    grad_fd = np.zeros_like(th_test)
    for j in range(mdl.n_theta):
        tp1 = th_test.copy(); tp1[j] += eps_th
        tm1 = th_test.copy(); tm1[j] -= eps_th
        tp2 = th_test.copy(); tp2[j] += 2.0 * eps_th
        tm2 = th_test.copy(); tm2[j] -= 2.0 * eps_th
        fp1 = mdl.laplace_nll_and_exact_grad(tp1, w_init=w_ref, update_cache=False)[0]
        fm1 = mdl.laplace_nll_and_exact_grad(tm1, w_init=w_ref, update_cache=False)[0]
        fp2 = mdl.laplace_nll_and_exact_grad(tp2, w_init=w_ref, update_cache=False)[0]
        fm2 = mdl.laplace_nll_and_exact_grad(tm2, w_init=w_ref, update_cache=False)[0]
        grad_fd[j] = (8.0 * (fp1 - fm1) - (fp2 - fm2)) / (12.0 * eps_th)
    max_outer_grad_err = float(np.max(np.abs(grad_exact - grad_fd)))

    assert err_nll < 1e-12, f"NB2 NLL mismatch: {err_nll}"
    assert max_nb_deriv_err < 1e-8, f"NB2 derivative mismatch: {max_nb_deriv_err}"
    assert err_torch_scipy_nll < 1e-8, f"Torch vs SciPy Laplace NLL mismatch: {err_torch_scipy_nll}"
    assert max_outer_grad_err < 1e-8, f"Laplace outer gradient mismatch: {max_outer_grad_err}"

    return {
        "nb2_scipy_logpmf_abs_err": err_nll,
        "nb2_derivs_fd_max_abs_err": max_nb_deriv_err,
        "laplace_torch_vs_scipy_nll_abs_err": err_torch_scipy_nll,
        "laplace_outer_grad_fd_max_abs_err": max_outer_grad_err,
        "tolerances": {
            "nb2_scipy_logpmf": 1e-12,
            "nb2_derivs_fd": 1e-8,
            "laplace_torch_vs_scipy_nll": 1e-8,
            "laplace_outer_grad_fd": 1e-8,
        },
        "pass": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_rule7_glmm_validation(), indent=2))
