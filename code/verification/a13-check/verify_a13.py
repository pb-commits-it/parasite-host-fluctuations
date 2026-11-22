#!/usr/bin/env python3
"""Standalone referee verification of the Assumption A13 extension (r44-a13-check).

Zero imports from a13/ or any other script in the repository.
Checks:
  0. Pre-registered known-result validation gates G1, G2, G3 (study rules Rule 7).
  1. Part ii proof steps at 80 decimal digits (upper-triangular Krylov similarity,
     cone invariance and zero deletion, deterministic path-MGF Jensen) + 6,000-case
     adversarial search across k in {1..6} + continuous unbounded zero-inflated laws
     (including infinite-variance Pareto beta=1.5) + corrected s -> infinity limit.
  2. Part i expansion coefficient A at large s, explicit fourth-moment remainder
     enclosures, fixed-factor reversal signs, and bounded-host derivative bounds on:
       (a) bounded zero-containing hosts (80-digit mpmath, including A < 0, A > 0, A = 0);
       (b) genuinely unbounded zero-containing hosts on [0, infinity) (truncation-free
           backward Koopman Poisson solver on [0, V_max] with coarse-vs-fine convergence
           verification and independent ergodic matrix-product cross-checks).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermite, roots_legendre

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 80


# ============================================================================
# Section A. Standalone matrix-product Perron solvers (float64 and 80-digit mp)
# ============================================================================

def ring_step_f64(h: float, taus: tuple[float, ...], d: float) -> np.ndarray:
    """Ring projection matrix with host h on edge k -> 1 and tilt exp(d) on k -> 1."""
    k = len(taus)
    mat = np.zeros((k, k), dtype=np.float64)
    for j, tau in enumerate(taus):
        mat[j, j] = 1.0 - 1.0 / tau
        if j < k - 1:
            mat[j + 1, j] = 1.0 / tau
        else:
            mat[0, k - 1] += math.exp(d) * h / tau
    return mat


def ring_step_mp(h, taus, d):
    """80-digit ring projection matrix with host h on edge k -> 1."""
    k = len(taus)
    taus_mp = [mp.mpf(str(t)) if not isinstance(t, mp.mpf) else t for t in taus]
    h_mp = mp.mpf(str(h)) if not isinstance(h, mp.mpf) else h
    d_mp = mp.mpf(str(d)) if not isinstance(d, mp.mpf) else d
    mat = mp.zeros(k)
    for j, tau in enumerate(taus_mp):
        mat[j, j] = 1 - 1 / tau
        if j < k - 1:
            mat[j + 1, j] = 1 / tau
        else:
            mat[0, k - 1] += mp.exp(d_mp) * h_mp / tau
    return mat


def periodic_growth_f64(seq: np.ndarray, taus: tuple[float, ...], d: float) -> float:
    """Exact per-step Lyapunov exponent of a periodic nonnegative host sequence."""
    k = len(taus)
    if k == 1:
        tau1 = taus[0]
        p1 = 1.0 - 1.0 / tau1
        g1 = math.exp(d) / tau1
        return float(np.mean(np.log(p1 + g1 * seq)))
    prod = np.eye(k, dtype=np.float64)
    log_scale = 0.0
    for h in seq:
        prod = ring_step_f64(float(h), taus, d) @ prod
        nrm = float(np.max(prod))
        prod /= nrm
        log_scale += math.log(nrm)
    if k == 2:
        tr = prod[0, 0] + prod[1, 1]
        disc = (prod[0, 0] - prod[1, 1]) ** 2 + 4.0 * prod[0, 1] * prod[1, 0]
        rho = 0.5 * (tr + math.sqrt(max(0.0, disc)))
    else:
        rho = float(np.max(np.abs(np.linalg.eigvals(prod))))
    return (log_scale + math.log(rho)) / len(seq)


def periodic_growth_mp(seq, taus, d):
    """80-digit exact per-step Lyapunov exponent of a periodic host sequence."""
    k = len(taus)
    d_mp = mp.mpf(str(d)) if not isinstance(d, mp.mpf) else d
    if k == 1:
        tau1 = mp.mpf(str(taus[0])) if not isinstance(taus[0], mp.mpf) else taus[0]
        p1 = 1 - 1 / tau1
        g1 = mp.exp(d_mp) / tau1
        return mp.fsum(
            mp.log(p1 + g1 * (mp.mpf(str(h)) if not isinstance(h, mp.mpf) else h))
            for h in seq
        ) / len(seq)
    prod = mp.eye(k)
    log_scale = mp.mpf(0)
    for h in seq:
        prod = ring_step_mp(h, taus, d_mp) * prod
        nrm = max(abs(prod[i, j]) for i in range(k) for j in range(k))
        prod /= nrm
        log_scale += mp.log(nrm)
    if k == 2:
        tr = prod[0, 0] + prod[1, 1]
        disc = (prod[0, 0] - prod[1, 1]) ** 2 + 4 * prod[0, 1] * prod[1, 0]
        rho = (tr + mp.sqrt(disc)) / 2
    else:
        eigs = mp.eig(prod, left=False, right=False)
        rho = max(abs(ev) for ev in eigs)
    return (log_scale + mp.log(rho)) / len(seq)


def solve_D_periodic_f64(seq: np.ndarray, taus: tuple[float, ...]) -> float:
    lo, hi = -1.0, 1.0
    while periodic_growth_f64(seq, taus, lo) >= 0.0:
        lo *= 2.0
    while periodic_growth_f64(seq, taus, hi) <= 0.0:
        hi *= 2.0
    return float(brentq(lambda d: periodic_growth_f64(seq, taus, d), lo, hi, xtol=1e-14, rtol=1e-14))


def solve_D_periodic_mp(seq, taus):
    seq_f = np.asarray([float(mp.mpf(str(x)) if not isinstance(x, mp.mpf) else x) for x in seq])
    taus_f = tuple(float(mp.mpf(str(t)) if not isinstance(t, mp.mpf) else t) for t in taus)
    d0 = solve_D_periodic_f64(seq_f, taus_f)
    g0 = mp.mpf(d0)
    return mp.findroot(
        lambda d: periodic_growth_mp(seq, taus, d),
        (g0 - mp.mpf("1e-5"), g0 + mp.mpf("1e-5")),
        tol=mp.mpf("1e-68"),
    )


def dsup_formula_f64(q0: float, log_gplus: float, taus: tuple[float, ...]) -> float:
    ps = [1.0 - 1.0 / t for t in taus]
    pmax = max(ps)
    lam_star = math.exp(-q0 / (1.0 - q0) * math.log(pmax))
    return sum(math.log(t * (lam_star - p)) for t, p in zip(taus, ps)) - log_gplus


def dsup_formula_mp(q0, log_gplus, taus):
    q = mp.mpf(str(q0)) if not isinstance(q0, mp.mpf) else q0
    lg = mp.mpf(str(log_gplus)) if not isinstance(log_gplus, mp.mpf) else log_gplus
    taus_mp = [mp.mpf(str(t)) if not isinstance(t, mp.mpf) else t for t in taus]
    ps = [1 - 1 / t for t in taus_mp]
    pmax = max(ps)
    lam_star = mp.exp(-q / (1 - q) * mp.log(pmax))
    return mp.fsum(mp.log(t * (lam_star - p)) for t, p in zip(taus_mp, ps)) - lg


# ============================================================================
# Section B. Truncation-free backward Koopman Poisson solver on [0, V_max]
#            for genuinely unbounded zero-containing Markov/iid hosts
# ============================================================================

def chebyshev_lobatto_barycentric(m: int):
    """Chebyshev-Lobatto nodes y in [0, 1] and barycentric weights."""
    idx = np.arange(m, dtype=np.float64)
    theta = np.pi * idx / (m - 1)
    x = -np.cos(theta)  # [-1, 1] ascending
    y = 0.5 * (x + 1.0)  # [0, 1] ascending
    w = np.ones(m, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5
    w *= (-1.0) ** idx
    return y, w


def barycentric_interp_matrix(y_nodes: np.ndarray, w_bary: np.ndarray, y_eval: np.ndarray) -> np.ndarray:
    """Evaluate interpolation basis L_j(y_eval) of shape (*y_eval.shape, m)."""
    diff = y_eval[..., None] - y_nodes[None, ...]
    exact = np.abs(diff) < 1e-15
    safe_diff = np.where(exact, 1.0, diff)
    ker = w_bary / safe_diff
    out = ker / np.sum(ker, axis=-1, keepdims=True)
    if np.any(exact):
        out = np.where(exact, 1.0, np.where(np.any(exact, axis=-1, keepdims=True), 0.0, out))
    return out


class UnboundedZeroHostLaw:
    """Stationary ergodic zero-containing unbounded law on [0, infinity).

    Modelled as a 2-state Markov chain Z_t in {0, 1} (which includes iid as a
    special case when rows of P equal (q0, 1 - q0)):
      - State 0: exact outage h_t = 0.
      - State 1: positive unbounded draw H_+ > 0 with E[H_+] = 1 / (1 - q0),
        integrated over all of (0, infinity) without support truncation.
    """

    def __init__(
        self,
        name: str,
        q0: float,
        P: np.ndarray,
        pos_kind: str,
        pos_params: dict,
        E_H_pos: float,
        E_H2_pos: float,
        E_H4_pos: float,
        log_Gplus: float,
    ):
        self.name = name
        self.q0 = q0
        self.P = np.asarray(P, dtype=np.float64)
        self.pos_kind = pos_kind
        self.pos_params = pos_params
        self.E_H_pos = E_H_pos
        self.E_H2_pos = E_H2_pos
        self.E_H4_pos = E_H4_pos
        self.log_Gplus = log_Gplus

        # Stationary distribution pi of P:
        # P = [[p00, p01], [p10, p11]], pi0 = q0, pi1 = 1 - q0
        self.pi = np.array([q0, 1.0 - q0], dtype=np.float64)
        assert np.allclose(self.pi @ self.P, self.pi, atol=1e-14)
        assert abs((1.0 - q0) * E_H_pos - 1.0) < 1e-14

        # Eigenvalue phi_Z = p00 + p11 - 1 of 2-state chain
        self.phi_Z = float(self.P[0, 0] + self.P[1, 1] - 1.0)
        self.E_h2 = (1.0 - q0) * E_H2_pos
        self.V = self.E_h2 - 1.0
        self.E_h4 = (1.0 - q0) * E_H4_pos
        self.H4_norm = self.E_h4 ** 0.25

    def exact_A(self, a: float) -> float:
        """Exact raw-covariance long-stay coefficient A = V/2 + sum_{l>=1} p^{l-1} C_l."""
        p = 1.0 - 1.0 / a
        # For l >= 1: Cov(h_t, h_{t-l}) = Cov(E[h|Z_t], E[h|Z_{t-l}])
        # Since Z_t in {0,1} with P(Z=1)=1-q0 and E[h|Z=1]=1/(1-q0):
        # Var(E[h|Z]) = q0 / (1 - q0), and C_l = [q0 / (1 - q0)] * phi_Z^l.
        var_cond = self.q0 / (1.0 - self.q0)
        if abs(self.phi_Z) < 1e-15:
            return 0.5 * self.V
        return 0.5 * self.V + var_cond * self.phi_Z / (1.0 - p * self.phi_Z)

    def prop4_constants(self, a: float) -> dict[str, float]:
        b = 1.0 / a
        H = self.H4_norm
        L = 2.0 * (4.0 * H + 1.0) / b
        N = L * (6.0 * H + 1.0) / b
        Rg = 2.0 * H * N + H * L * (6.0 * H + 2.0) + (8.0 / 3.0) * (4.0 * H + 1.0) ** 3
        B0 = 4.0 * H ** 2 / b + 2.0 / b + 2.0 * H ** 2 + 3.0
        J = 4.0 * H ** 2 / b + 2.0 * H ** 2 + 1.0 / b + 1.0
        K0 = B0 + Rg
        C = J * K0 + Rg + 2.0 * K0 ** 2
        e0 = min(0.5 * b, 1.0 / (4.0 * (B0 + Rg + 1.0)))
        return dict(H=H, L=L, N=N, Rg=Rg, B0=B0, J=J, K0=K0, C=C, e0=e0)

    def positive_quadrature(self, n_quad: int):
        """Return truncation-free quadrature nodes h_q > 0, weights w_q, and optional analytic log offset.

        Every map transforms (0, infinity) smoothly onto a bounded reference interval
        with zero support truncation.
        """
        kind = self.pos_kind
        if kind == "gamma":
            alpha = self.pos_params["alpha"]
            scale = self.pos_params["scale"]
            u01, wu = roots_legendre(n_quad)
            u = 0.5 * (u01 + 1.0)
            du = 0.5 * wu
            # Map x = (2 u / (1 - u))^2 in (0, infinity), H_+ = scale * x
            t = 2.0 * u / (1.0 - u)
            x = t * t
            dx_du = 2.0 * t * 2.0 / ((1.0 - u) ** 2)
            # Density x^{alpha-1} exp(-x) / Gamma(alpha); prevent underflow for large x
            log_dens = (alpha - 1.0) * np.log(x) - x - math.lgamma(alpha)
            dens = np.where(x < 700.0, np.exp(np.maximum(log_dens, -745.0)), 0.0)
            wq = dens * dx_du * du
            hq = scale * x
            return hq, wq, "standard"

        if kind == "lognormal":
            sigma = self.pos_params["sigma"]
            mean_pos = self.E_H_pos
            z_h, w_h = roots_hermite(n_quad)
            # z ~ N(0, 1): z = sqrt(2) * z_h, weight w_h / sqrt(pi)
            wq = w_h / math.sqrt(math.pi)
            hq = mean_pos * np.exp(sigma * math.sqrt(2.0) * z_h - 0.5 * sigma * sigma)
            return hq, wq, "standard"

        if kind == "pareto55":
            # Exact Pareto(beta = 11/2 = 5.5) on [x0, infinity): H_+ = x0 * U^{-2/11}.
            # Substitute U = w^11 on w in (0, 1]: H_+(w) = x0 * w^{-2}, dU = 11 w^10 dw.
            x0 = self.pos_params["x0"]
            u01, wu = roots_legendre(n_quad)
            w = 0.5 * (u01 + 1.0)
            dw = 0.5 * wu
            return (x0, w, dw), None, "pareto55_exact"

        raise ValueError(f"Unknown positive law kind: {kind}")

    def sample_stream(self, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate an ergodic sample path of length n_steps from the exact stationary law."""
        z = int(rng.random() >= self.q0)
        p01 = self.P[0, 1]
        p11 = self.P[1, 1]
        states = np.empty(n_steps, dtype=np.int8)
        u_trans = rng.random(n_steps)
        for t in range(n_steps):
            states[t] = z
            if z == 0:
                z = 1 if u_trans[t] < p01 else 0
            else:
                z = 1 if u_trans[t] < p11 else 0
        out = np.zeros(n_steps, dtype=np.float64)
        mask = states == 1
        m = int(np.sum(mask))
        if self.pos_kind == "gamma":
            out[mask] = rng.gamma(self.pos_params["alpha"], self.pos_params["scale"], size=m)
        elif self.pos_kind == "lognormal":
            sigma = self.pos_params["sigma"]
            out[mask] = self.E_H_pos * np.exp(sigma * rng.standard_normal(m) - 0.5 * sigma ** 2)
        elif self.pos_kind == "pareto55":
            x0 = self.pos_params["x0"]
            out[mask] = x0 * (rng.random(m) ** (-2.0 / 11.0))
        else:
            raise ValueError(self.pos_kind)
        return out


def koopman_growth_rate(
    law: UnboundedZeroHostLaw,
    a: float,
    s: float,
    d: float,
    m_nodes: int = 48,
    n_quad: int = 80,
) -> float:
    """Solve the backward Koopman Poisson equation (I - K) psi + r 1 = R on v in [0, V_max].

    Uses quadratic endpoint clustering v = V_max * y^2 on Chebyshev-Lobatto nodes y in [0, 1].
    Requires 1/s < 1/a so V_max = b / (b - epsilon) is finite (and works for all c > 0
    because [0, V_max] is forward-invariant for EVERY c > 0 whenever epsilon < b!).
    """
    b = 1.0 / a
    p = 1.0 - b
    eps = 1.0 / s
    assert eps < b, "Need s > a for compact [0, V_max] invariance"
    vmax = b / (b - eps)
    c = math.exp(d)

    y_nodes, w_bary = chebyshev_lobatto_barycentric(m_nodes)
    v_nodes = vmax * (y_nodes ** 2)

    # State 0 (exact zero outage h = 0): deterministic map F_0(v) = (p v + b) / (1 - eps)
    v_next_0 = (p * v_nodes + b) / (1.0 - eps)
    y_next_0 = np.sqrt(np.clip(v_next_0 / vmax, 0.0, 1.0))
    L0 = barycentric_interp_matrix(y_nodes, w_bary, y_next_0)  # (m, m)
    R0 = np.full(m_nodes, math.log(1.0 - eps), dtype=np.float64)

    # State 1 (positive unbounded H_+ > 0): expectation over (0, infinity)
    qdata, wq, mode = law.positive_quadrature(n_quad)
    if mode == "standard":
        hq = qdata
        # Broadcast: v_nodes[:, None], hq[None, :] -> (m, n_quad)
        denom = (1.0 - eps) + eps * c * v_nodes[:, None] * hq[None, :]
        v_next_1 = (p * v_nodes[:, None] + b) / denom
        y_next_1 = np.sqrt(np.clip(v_next_1 / vmax, 0.0, 1.0))
        L_all = barycentric_interp_matrix(y_nodes, w_bary, y_next_1)  # (m, n_quad, m)
        L1 = np.tensordot(L_all, wq, axes=([1], [0]))  # (m, m)
        R1 = np.log(denom) @ wq  # (m,)
    else:
        # Exact analytic Pareto(5.5) mapping H_+(w) = x0 * w^{-2}, weight 11 w^10 dw on [0, 1]
        x0, w, dw = qdata
        weight = 11.0 * (w ** 10) * dw  # integrates to 1 exactly up to Legendre precision
        # F_{H_+}(v) = (p v + b) w^2 / ((1 - eps) w^2 + eps c x0 v)
        num_f = (p * v_nodes[:, None] + b) * (w[None, :] ** 2)
        den_f = (1.0 - eps) * (w[None, :] ** 2) + eps * c * x0 * v_nodes[:, None]
        # At v = 0 (m_node 0): F_{H_+}(0) = b / (1 - eps) for every H_+ < infinity
        v_next_1 = np.where(v_nodes[:, None] == 0.0, b / (1.0 - eps), num_f / np.where(den_f == 0.0, 1.0, den_f))
        y_next_1 = np.sqrt(np.clip(v_next_1 / vmax, 0.0, 1.0))
        L_all = barycentric_interp_matrix(y_nodes, w_bary, y_next_1)
        L1 = np.tensordot(L_all, weight, axes=([1], [0]))
        # Reward: log(1 - eps + eps c x0 v w^{-2}) = -2 log(w) + log((1-eps) w^2 + eps c x0 v)
        # Integral of (-2 log w) * 11 w^10 dw on [0, 1] is EXACTLY 2 / 11!
        R1 = np.empty(m_nodes, dtype=np.float64)
        R1[0] = math.log(1.0 - eps)
        R1[1:] = (2.0 / 11.0) + np.log(den_f[1:, :]) @ weight

    # Assemble 2m x 2m Markov operator K on stacked vector [psi_0; psi_1]
    P = law.P
    K = np.block([
        [P[0, 0] * L0, P[0, 1] * L0],
        [P[1, 0] * L1, P[1, 1] * L1],
    ])
    rhs = np.concatenate([R0, R1])
    sys_mat = np.eye(2 * m_nodes, dtype=np.float64) - K
    # Pin psi_0(y_0) = 0 and put unknown r in column 0
    sys_mat[:, 0] = 1.0
    sol = np.linalg.solve(sys_mat, rhs)
    return float(sol[0])


def solve_D_koopman(
    law: UnboundedZeroHostLaw,
    a: float,
    s: float,
    m_nodes: int = 48,
    n_quad: int = 80,
) -> float:
    a_est = law.exact_A(a) / s
    span = max(0.05, 5.0 * abs(a_est) + 20.0 / (s * s))
    lo, hi = a_est - span, a_est + span
    while koopman_growth_rate(law, a, s, lo, m_nodes, n_quad) >= 0.0:
        lo -= span
        span *= 2.0
    while koopman_growth_rate(law, a, s, hi, m_nodes, n_quad) <= 0.0:
        hi += span
        span *= 2.0
    return float(
        brentq(
            lambda d: koopman_growth_rate(law, a, s, d, m_nodes, n_quad),
            lo,
            hi,
            xtol=1e-14,
            rtol=1e-14,
        )
    )


def build_unbounded_laws() -> list[UnboundedZeroHostLaw]:
    laws = []

    # U1: iid zero-inflated Exponential (q0 = 0.4, alpha = 1, scale = 5/3), A > 0
    q0 = 0.4
    scale = 1.0 / (1.0 - q0)
    P_iid_04 = [[q0, 1.0 - q0], [q0, 1.0 - q0]]
    laws.append(
        UnboundedZeroHostLaw(
            name="U1_iid_exp",
            q0=q0,
            P=P_iid_04,
            pos_kind="gamma",
            pos_params={"alpha": 1.0, "scale": scale},
            E_H_pos=scale,
            E_H2_pos=2.0 * scale ** 2,
            E_H4_pos=24.0 * scale ** 4,
            log_Gplus=math.log(scale) - float(mp.euler),
        )
    )

    # U2: Markov alternating (theta = 0.98, q0 = 0.5) zero + Gamma(8, 1/4), A < 0 reversal!
    theta = 0.98
    P_alt = [[1.0 - theta, theta], [theta, 1.0 - theta]]
    alpha, sc = 8.0, 0.25
    laws.append(
        UnboundedZeroHostLaw(
            name="U2_markov_gamma8_rev",
            q0=0.5,
            P=P_alt,
            pos_kind="gamma",
            pos_params={"alpha": alpha, "scale": sc},
            E_H_pos=alpha * sc,
            E_H2_pos=alpha * (alpha + 1.0) * sc ** 2,
            E_H4_pos=alpha * (alpha + 1.0) * (alpha + 2.0) * (alpha + 3.0) * sc ** 4,
            log_Gplus=float(mp.digamma(8)) + math.log(sc),
        )
    )

    # U3: Markov persistent (theta = 0.25, q0 = 0.5) zero + singular-at-0 Gamma(0.5, 4), A > 0
    theta_p = 0.25
    P_pers = [[1.0 - theta_p, theta_p], [theta_p, 1.0 - theta_p]]
    a3, s3 = 0.5, 4.0
    laws.append(
        UnboundedZeroHostLaw(
            name="U3_markov_gamma05_sing",
            q0=0.5,
            P=P_pers,
            pos_kind="gamma",
            pos_params={"alpha": a3, "scale": s3},
            E_H_pos=a3 * s3,
            E_H2_pos=a3 * (a3 + 1.0) * s3 ** 2,
            E_H4_pos=a3 * (a3 + 1.0) * (a3 + 2.0) * (a3 + 3.0) * s3 ** 4,
            log_Gplus=float(mp.digamma(mp.mpf("0.5"))) + math.log(s3),
        )
    )

    # U4a: iid zero-inflated Pareto(beta = 5.5, E[h^4] < inf, E[h^6] = +inf), A > 0
    x0 = 18.0 / 11.0  # E[H_+] = (11/9) x0 = 2
    P_iid_05 = [[0.5, 0.5], [0.5, 0.5]]
    laws.append(
        UnboundedZeroHostLaw(
            name="U4a_iid_pareto55",
            q0=0.5,
            P=P_iid_05,
            pos_kind="pareto55",
            pos_params={"x0": x0},
            E_H_pos=2.0,
            E_H2_pos=(x0 ** 2) * (11.0 / 7.0),
            E_H4_pos=(x0 ** 4) * (11.0 / 3.0),
            log_Gplus=math.log(x0) + 2.0 / 11.0,
        )
    )

    # U4b: Markov alternating (theta = 0.98) zero-inflated Pareto(beta = 5.5), A < 0 reversal!
    laws.append(
        UnboundedZeroHostLaw(
            name="U4b_markov_pareto55_rev",
            q0=0.5,
            P=P_alt,
            pos_kind="pareto55",
            pos_params={"x0": x0},
            E_H_pos=2.0,
            E_H2_pos=(x0 ** 2) * (11.0 / 7.0),
            E_H4_pos=(x0 ** 4) * (11.0 / 3.0),
            log_Gplus=math.log(x0) + 2.0 / 11.0,
        )
    )

    # U5: iid zero-inflated Lognormal (q0 = 0.3, sigma = 0.8, infinite MGF), A > 0
    q5 = 0.3
    m5 = 1.0 / (1.0 - q5)
    sig5 = 0.8
    P_iid_03 = [[q5, 1.0 - q5], [q5, 1.0 - q5]]
    laws.append(
        UnboundedZeroHostLaw(
            name="U5_iid_lognormal",
            q0=q5,
            P=P_iid_03,
            pos_kind="lognormal",
            pos_params={"sigma": sig5},
            E_H_pos=m5,
            E_H2_pos=(m5 ** 2) * math.exp(sig5 ** 2),
            E_H4_pos=(m5 ** 4) * math.exp(6.0 * sig5 ** 2),
            log_Gplus=math.log(m5) - 0.5 * sig5 ** 2,
        )
    )

    return laws


# ============================================================================
# Section C. Pre-registered known-result validation gates (G1, G2, G3)
# ============================================================================

def run_known_result_gates() -> dict:
    # G1: Constant host h_t = 1 -> D = 0
    g1_f64_errs = []
    g1_mp_errs = []
    for taus in [(2.0,), (2.0, 1.0), (2.0, 5.0), (2.0, 3.0, 10.0), (1.5, 2.0, 1.0, 4.0, 8.0)]:
        g1_f64_errs.append(abs(solve_D_periodic_f64(np.ones(5), taus)))
        g1_mp_errs.append(abs(solve_D_periodic_mp([1, 1, 1], taus)))

    const_law = UnboundedZeroHostLaw(
        name="const_bench",
        q0=0.0,
        P=[[0.0, 1.0], [0.0, 1.0]],
        pos_kind="lognormal",
        pos_params={"sigma": 0.0},
        E_H_pos=1.0,
        E_H2_pos=1.0,
        E_H4_pos=1.0,
        log_Gplus=0.0,
    )
    g1_koop_errs = [abs(solve_D_koopman(const_law, 2.0, s)) for s in [5.0, 50.0, 1000.0]]

    assert max(g1_f64_errs) < 1e-13
    assert max(g1_mp_errs) < mp.mpf("1e-60")
    assert max(g1_koop_errs) < 1e-13

    # G2: Positive alternating (1.6, 0.4) at (2, 2) and pure outage (2, 0) closed forms
    target_pos_alt = mp.log((mp.mpf("0.625") - mp.sqrt(mp.mpf("0.300625"))) / mp.mpf("0.08"))
    err_pos_alt_mp = abs(solve_D_periodic_mp(["1.6", "0.4"], (2, 2)) - target_pos_alt)
    assert err_pos_alt_mp < mp.mpf("1e-60")

    alt0_errs_mp = []
    alt0_deriv_errs_mp = []
    alt0_koop_errs = []
    alt0_law = UnboundedZeroHostLaw(
        name="alt0_bench",
        q0=0.5,
        P=[[0.0, 1.0], [1.0, 0.0]],
        pos_kind="lognormal",
        pos_params={"sigma": 0.0},
        E_H_pos=2.0,
        E_H2_pos=4.0,
        E_H4_pos=16.0,
        log_Gplus=math.log(2.0),
    )
    for a_str in ["1.2", "2", "7"]:
        a_mp = mp.mpf(a_str)
        for s_str in ["1", "3", "25"]:
            s_mp = mp.mpf(s_str)
            exact_d = mp.log((2 * a_mp - 1) * (2 * s_mp - 1) / (2 * ((2 * a_mp - 1) * s_mp - a_mp + 1)))
            num_d = solve_D_periodic_mp([2, 0], (a_mp, s_mp))
            alt0_errs_mp.append(abs(num_d - exact_d))
            exact_ds = 1 / ((2 * s_mp - 1) * ((2 * a_mp - 1) * s_mp - a_mp + 1))
            # Implicit differentiation -r_s / r_D
            num_ds = -mp.diff(lambda ss: periodic_growth_mp([2, 0], (a_mp, ss), exact_d), s_mp) / mp.diff(
                lambda dd: periodic_growth_mp([2, 0], (a_mp, s_mp), dd), exact_d
            )
            alt0_deriv_errs_mp.append(abs(num_ds - exact_ds))
            if float(s_str) > float(a_str):
                alt0_koop_errs.append(abs(solve_D_koopman(alt0_law, float(a_str), float(s_str)) - float(exact_d)))

    assert max(alt0_errs_mp) < mp.mpf("1e-60")
    assert max(alt0_deriv_errs_mp) < mp.mpf("1e-50")
    assert max(alt0_koop_errs) < 1e-13

    # G3: Exact k = 1 scalar zero-outage threshold on discrete & continuous unbounded Gamma
    d_k1_disc = solve_D_periodic_mp([0, 0, 1, 3], (3,))
    # Direct 1D equation: 0.5*log(2/3) + 0.25*log(2/3 + e^D/3) + 0.25*log(2/3 + e^D) = 0
    res_k1_disc = abs(
        mp.mpf("0.5") * mp.log(mp.mpf(2) / 3)
        + mp.mpf("0.25") * mp.log(mp.mpf(2) / 3 + mp.exp(d_k1_disc) / 3)
        + mp.mpf("0.25") * mp.log(mp.mpf(2) / 3 + mp.exp(d_k1_disc))
    )
    assert res_k1_disc < mp.mpf("1e-65")

    return {
        "G1_max_f64_err": max(g1_f64_errs),
        "G1_max_mp_err": mp.nstr(max(g1_mp_errs), 12),
        "G1_max_koopman_err": max(g1_koop_errs),
        "G2_pos_alt_mp_err": mp.nstr(err_pos_alt_mp, 12),
        "G2_alt0_max_mp_err": mp.nstr(max(alt0_errs_mp), 12),
        "G2_alt0_deriv_mp_err": mp.nstr(max(alt0_deriv_errs_mp), 12),
        "G2_alt0_koopman_err": max(alt0_koop_errs),
        "G3_k1_disc_residual": mp.nstr(res_k1_disc, 12),
    }


# ============================================================================
# Section D. Part ii verification (algebraic steps, 6,000 adversarial cases,
#            continuous unbounded & infinite-variance checks, corrected limit)
# ============================================================================

def verify_part_ii_algebra_mp() -> dict:
    """80-digit check of Steps 1-3 of Proposition 7 across k in {1..5}."""
    sim_errs = []
    upper_checks = 0
    cone_cases = 0
    cone_min_entry = mp.mpf(0)
    orthant_counterexamples = 0
    jensen_cases = 0
    jensen_min_slack = mp.mpf("inf")
    mgf_errs = []

    stay_tuples = [
        (mp.mpf("2.5"),),
        (mp.mpf("1"), mp.mpf("5")),
        (mp.mpf("3"), mp.mpf("3")),
        (mp.mpf("1"), mp.mpf("4"), mp.mpf("1.5")),
        (mp.mpf("2"), mp.mpf("1"), mp.mpf("1"), mp.mpf("10")),
        (mp.mpf("1.2"), mp.mpf("5"), mp.mpf("1"), mp.mpf("5"), mp.mpf("2")),
    ]

    for taus in stay_tuples:
        k = len(taus)
        ps = [1 - 1 / t for t in taus]
        z = mp.mpf("1.3") / mp.fprod(taus)

        def build_C_and_K(p_vec):
            C = mp.zeros(k)
            for j in range(k):
                C[j, j] = p_vec[j]
                if j < k - 1:
                    C[j + 1, j] = 1
            K = mp.zeros(k)
            col = mp.zeros(k, 1)
            col[0, 0] = 1
            for m in range(k):
                for r in range(k):
                    K[r, m] = col[r, 0]
                col = C * col
            return C, K

        C1, K1 = build_C_and_K(ps)
        # Verify K1 is UPPER triangular with unit diagonal (primary's stated fix)
        for r in range(k):
            assert K1[r, r] == 1
            for c_idx in range(r):
                assert K1[r, c_idx] == 0
        upper_checks += 1

        # Reorder stays so p_1 = pmax
        ps_sorted = sorted(ps, reverse=True)
        C2, K2 = build_C_and_K(ps_sorted)
        K1_inv = K1 ** -1
        K2_inv = K2 ** -1
        for h_val in [mp.mpf(0), mp.mpf("0.37"), mp.mpf("4.2")]:
            A1 = C1 + 0
            A1[0, k - 1] += z * h_val
            A2 = C2 + 0
            A2[0, k - 1] += z * h_val
            comp1 = K1_inv * A1 * K1
            comp2 = K2_inv * A2 * K2
            err = max(abs(comp1[i, j] - comp2[i, j]) for i in range(k) for j in range(k))
            sim_errs.append(err)
            assert err < mp.mpf("1e-65")

        # Step 2: Cone invariance and zero-deletion inequality on ps_sorted (p_1 = pmax)
        pmax = ps_sorted[0]
        if k >= 2 and min(ps_sorted) < pmax:
            # Verify whole-orthant C2 >= pmax I is FALSE at e_k
            assert C2[k - 1, k - 1] - pmax < 0
            orthant_counterexamples += 1

        env = [mp.mpf(0), mp.mpf("2.4"), mp.mpf(0), mp.mpf(0), mp.mpf("0.5"), mp.mpf("1.8"), mp.mpf(0)]
        e1 = mp.zeros(k, 1)
        e1[0, 0] = 1
        vec_full = e1.copy()
        vec_pos = e1.copy()
        n0 = 0
        for h_val in env:
            Ah = C2 + 0
            Ah[0, k - 1] += z * h_val
            vec_full = Ah * vec_full
            # Check cone membership (C2 - pmax I) vec_full >= 0
            cone_vec = (C2 - pmax * mp.eye(k)) * vec_full
            for r in range(k):
                cone_min_entry = min(cone_min_entry, cone_vec[r, 0])
                assert cone_vec[r, 0] >= -mp.mpf("1e-65")
            if h_val == 0:
                n0 += 1
            else:
                vec_pos = Ah * vec_pos
        diff_del = vec_full - (pmax ** n0) * vec_pos
        for r in range(k):
            assert diff_del[r, 0] >= -mp.mpf("1e-65")
        cone_cases += 1

        # Verify C2^(k-1) e_1 > 0 strictly
        ck1_e1 = (C2 ** (k - 1)) * e1
        for r in range(k):
            assert ck1_e1[r, 0] > 0

        # Step 3: Exact path-enumeration MGF and finite-path Jensen (including k = 1 labelled self-edges!)
        if k <= 4:
            gplus = mp.mpf("1.4")
            # Solve prod_j (lam - p_j) = z * gplus
            lam = mp.findroot(
                lambda x: mp.fprod(x - p for p in ps_sorted) - z * gplus,
                pmax + mp.mpf("0.5"),
                tol=mp.mpf("1e-70"),
            )
            # Construct right/left Perron vectors of Aplus = C2 + z*gplus*e1*ek^T
            v = mp.zeros(k, 1)
            v[0, 0] = 1
            for j in range(1, k):
                v[j, 0] = v[j - 1, 0] / (lam - ps_sorted[j])
            w = mp.zeros(k, 1)
            w[k - 1, 0] = 1
            for j in range(k - 2, -1, -1):
                w[j, 0] = w[j + 1, 0] / (lam - ps_sorted[j])
            norm_wv = sum(w[j, 0] * v[j, 0] for j in range(k))
            v /= norm_wv
            nu = w[0, 0] * z * gplus * v[k - 1, 0] / lam

            H_seq = [mp.mpf("0.2"), mp.mpf("3.7"), mp.mpf("0.9"), mp.mpf("2.1")]
            prod_H = mp.eye(k)
            for H_val in H_seq:
                Ah = C2 + 0
                Ah[0, k - 1] += z * H_val
                prod_H = Ah * prod_H
            lhs_matrix = (w.T * prod_H * v)[0, 0] / (lam ** len(H_seq))

            # Explicit labelled-edge Markov chain enumeration over all 2^m * k initial-state paths
            # From state j, two outgoing edges: stay (j -> j, weight ps_sorted[j]) and transfer (j -> j+1 mod k)
            mgf_enum = mp.mpf(0)
            for j0 in range(k):
                pi_j0 = w[j0, 0] * v[j0, 0]
                # Each step chooses edge_type in {0 (stay), 1 (transfer)}
                for bits in range(1 << len(H_seq)):
                    cur = j0
                    prob = pi_j0
                    mult = mp.mpf(1)
                    valid = True
                    for step_i, H_val in enumerate(H_seq):
                        take_transfer = (bits >> step_i) & 1
                        if take_transfer == 0:
                            wt = ps_sorted[cur]
                            if wt == 0:
                                valid = False
                                break
                            nxt = cur
                            prob *= w[nxt, 0] * wt / (lam * w[cur, 0])
                        else:
                            nxt = (cur + 1) % k
                            wt = z * gplus if cur == k - 1 else mp.mpf(1)
                            prob *= w[nxt, 0] * wt / (lam * w[cur, 0])
                            if cur == k - 1:
                                mult *= H_val / gplus
                        cur = nxt
                    if valid:
                        mgf_enum += prob * mult

            mgf_errs.append(abs(lhs_matrix - mgf_enum))
            assert abs(lhs_matrix - mgf_enum) < mp.mpf("1e-65")

            lhs_log = mp.log(lhs_matrix * (lam ** len(H_seq)))
            rhs_jensen = len(H_seq) * mp.log(lam) + nu * mp.fsum(mp.log(H_val / gplus) for H_val in H_seq)
            slack = lhs_log - rhs_jensen
            jensen_min_slack = min(jensen_min_slack, slack)
            assert slack >= -mp.mpf("1e-65")
            jensen_cases += 1

    return {
        "upper_triangular_verified_cases": upper_checks,
        "max_simultaneous_similarity_err": mp.nstr(max(sim_errs), 12),
        "cone_deletion_cases": cone_cases,
        "orthant_counterexamples_verified": orthant_counterexamples,
        "max_labelled_path_mgf_err": mp.nstr(max(mgf_errs), 12),
        "jensen_cases": jensen_cases,
        "jensen_min_slack": mp.nstr(jensen_min_slack, 12),
    }


def run_part_ii_adversarial() -> dict:
    """6,000 adversarial periodic schedules across k in {1..6} + unbounded laws."""
    rng = np.random.default_rng(2026111544)
    stay_grids = [
        # k = 1
        (1.02,),
        (2.0,),
        (50.0,),
        # k = 2
        (1.001, 200.0),
        (200.0, 1.001),
        (2.0, 1.0),
        (1.0, 2.0),
        (2.0, 20.0),
        # k = 3
        (1.0, 1.0, 150.0),
        (150.0, 1.0, 1.0),
        (1.0, 150.0, 1.0),
        (1.002, 80.0, 2.5),
        # k = 4
        (1.0, 1.0, 1.0, 120.0),
        (120.0, 1.0, 1.0, 1.0),
        (2.0, 1.0, 30.0, 1.5),
        # k = 5
        (1.0, 100.0, 1.0, 1.0, 1.005),
        (2.0, 2.0, 2.0, 2.0, 40.0),
        # k = 6
        (1.0, 1.0, 80.0, 1.0, 2.5, 1.001),
    ]
    lengths = [3, 6, 12, 24, 48, 96, 160]

    n_cases = 6000
    violations_f64 = 0
    tightest = []  # keep 25 smallest r(D_sup)

    t0 = time.monotonic()
    for idx in range(n_cases):
        taus = stay_grids[idx % len(stay_grids)]
        n = lengths[(idx // len(stay_grids)) % len(lengths)]
        nz = int(rng.integers(1, n))
        npos = n - nz
        pos_mode = (idx // 7) % 6
        if pos_mode == 0:
            # Pure binary outage (attains bound in long-block limit)
            pos = np.ones(npos, dtype=np.float64)
        elif pos_mode == 1:
            # Extreme log-uniform over e^[-18, 18]
            pos = np.exp(rng.uniform(-18.0, 18.0, size=npos))
        elif pos_mode == 2:
            # Extreme heavy-tailed Pareto(0.25)
            pos = 1.0 + rng.pareto(0.25, size=npos)
        elif pos_mode == 3:
            # Extreme bimodal tiny-plus-spike
            pos = np.where(rng.random(npos) < 0.85, 1e-9, 1e5)
        elif pos_mode == 4:
            # Monotone geometric ramp
            pos = np.geomspace(1e-7, 1e7, npos)
        else:
            # Singular-near-zero Beta(0.15, 1) + rare spike
            pos = rng.beta(0.15, 1.0, size=npos) + (rng.random(npos) < 0.1) * 50.0
            pos = np.maximum(pos, 1e-14)

        order_mode = idx % 5
        if order_mode == 0:
            # Single contiguous outage block (sharpness direction!)
            seq = np.concatenate([np.zeros(nz), pos])
        elif order_mode == 1:
            # Reversed positive ramp after contiguous outage block
            seq = np.concatenate([np.zeros(nz), pos[::-1]])
        elif order_mode == 2:
            # Resonant / maximally interleaved zeros
            seq = np.zeros(n, dtype=np.float64)
            pos_idx = np.floor(np.linspace(0, n, npos, endpoint=False)).astype(int)
            seq[pos_idx] = pos
        elif order_mode == 3:
            # Two unequal zero bursts
            z1 = nz // 2
            p1 = npos // 2
            seq = np.concatenate([np.zeros(z1), pos[:p1], np.zeros(nz - z1), pos[p1:]])
        else:
            seq = np.concatenate([np.zeros(nz), pos])
            rng.shuffle(seq)

        # Normalize to exact mean 1
        seq = seq * (n / float(np.sum(seq)))
        pos_vals = seq[seq > 0.0]
        q0 = nz / n
        log_gplus = float(np.mean(np.log(pos_vals)))
        dsup = dsup_formula_f64(q0, log_gplus, taus)
        g_at_sup = periodic_growth_f64(seq, taus, dsup)

        if g_at_sup < -1e-9:
            violations_f64 += 1

        if len(tightest) < 25:
            tightest.append((g_at_sup, idx, taus, seq.copy()))
            tightest.sort(key=lambda x: x[0])
        elif g_at_sup < tightest[-1][0]:
            tightest[-1] = (g_at_sup, idx, taus, seq.copy())
            tightest.sort(key=lambda x: x[0])

    assert violations_f64 == 0

    # Re-verify all 25 tightest schedules at 80 decimal digits (growth(D_sup) >= 0 and D <= D_sup)
    mp_min_growth = mp.mpf("inf")
    mp_min_gap = mp.mpf("inf")
    for _, idx, taus, seq in tightest:
        n = len(seq)
        seq_mp = [mp.mpf(float(x)) for x in seq]
        mean_mp = mp.fsum(seq_mp) / n
        seq_mp = [x / mean_mp for x in seq_mp]
        pos_mp = [x for x in seq_mp if x > 0]
        q0_mp = mp.mpf(n - len(pos_mp)) / n
        lg_mp = mp.fsum(mp.log(x) for x in pos_mp) / len(pos_mp)
        dsup_mp = dsup_formula_mp(q0_mp, lg_mp, taus)
        g_mp = periodic_growth_mp(seq_mp, taus, dsup_mp)
        d_mp = solve_D_periodic_mp(seq_mp, taus)
        assert g_mp >= -mp.mpf("1e-55")
        assert dsup_mp - d_mp >= -mp.mpf("1e-55")
        mp_min_growth = min(mp_min_growth, g_mp)
        mp_min_gap = min(mp_min_gap, dsup_mp - d_mp)

    # Also test Part ii on continuous unbounded zero-inflated laws:
    # (a) All 6 unbounded laws across (a, s) in {(2, 5), (2, 20), (3, 10)} (18 Koopman cells)
    unbounded_part2_gaps = []
    for law in build_unbounded_laws():
        for a, s in [(2.0, 5.0), (2.0, 20.0), (3.0, 10.0)]:
            d_val = solve_D_koopman(law, a, s, m_nodes=56, n_quad=96)
            dsup_val = dsup_formula_f64(law.q0, law.log_Gplus, (a, s))
            gap = dsup_val - d_val
            assert gap > 0.0
            unbounded_part2_gaps.append(gap)

    # (b) Infinite-variance Pareto(beta = 1.5, E[h] = 1, E[h^2] = +infinity!) at k = 1 and k = 2
    # Conditional on h > 0 (q0 = 0.5): H_+ = x0 * U^{-2/3} with x0 = 2/3 so E[H_+] = 3 * x0 = 2,
    # log G_+ = log(2/3) + 2/3. At k = 1, r(D) = 0.5*log(p1) + 0.5*int_0^1 log(p1 + (1-p1)e^D x0 u^{-2/3}) du.
    # Substitute u = w^3 (du = 3 w^2 dw): exact 1D analytic integrand on [0, 1]!
    x0_infvar = mp.mpf(2) / 3
    lg_infvar = mp.log(x0_infvar) + mp.mpf(2) / 3
    dsup_k1_infvar = dsup_formula_mp("0.5", lg_infvar, (mp.mpf(2),))
    # Evaluate r(D_sup) at k = 1, tau_1 = 2 (p_1 = 1/2):
    r_k1_infvar_sup = mp.mpf("0.5") * mp.log(mp.mpf("0.5")) + mp.mpf("0.5") * (
        mp.mpf(2) / 3
        + mp.quad(lambda w: 3 * w ** 2 * mp.log(mp.mpf("0.5") * w ** 2 + mp.mpf("0.5") * mp.exp(dsup_k1_infvar) * x0_infvar), [0, 1])
    )
    assert r_k1_infvar_sup > 0

    # Also test infinite-variance Pareto(beta = 1.5) at k = 2, (a, s) = (2, 5) via Koopman on [0, V_max]!
    # Even though E[h^2] = +infinity (so Part i does NOT apply), Part ii's [0, V_max] invariance and D <= D_sup DO apply!
    u01, wu = roots_legendre(96)
    w_iv = 0.5 * (u01 + 1.0)
    wt_iv = 3.0 * (w_iv ** 2) * (0.5 * wu)
    a_iv, s_iv = 2.0, 5.0
    dsup_k2_infvar = dsup_formula_f64(0.5, float(lg_infvar), (a_iv, s_iv))
    b_iv, p_iv, eps_iv = 0.5, 0.5, 0.2
    vmax_iv = b_iv / (b_iv - eps_iv)
    yn, wb = chebyshev_lobatto_barycentric(56)
    vn = vmax_iv * yn ** 2
    y0_iv = np.sqrt(np.clip(((p_iv * vn + b_iv) / (1.0 - eps_iv)) / vmax_iv, 0.0, 1.0))
    L0_iv = barycentric_interp_matrix(yn, wb, y0_iv)
    c_iv = math.exp(dsup_k2_infvar)
    num_iv = (p_iv * vn[:, None] + b_iv) * (w_iv[None, :] ** 2)
    den_iv = (1.0 - eps_iv) * (w_iv[None, :] ** 2) + eps_iv * c_iv * (2.0 / 3.0) * vn[:, None]
    v1_iv = np.where(vn[:, None] == 0.0, b_iv / (1.0 - eps_iv), num_iv / np.where(den_iv == 0.0, 1.0, den_iv))
    L1_iv = np.tensordot(barycentric_interp_matrix(yn, wb, np.sqrt(np.clip(v1_iv / vmax_iv, 0.0, 1.0))), wt_iv, axes=([1], [0]))
    R0_iv = np.full(56, math.log(1.0 - eps_iv))
    R1_iv = np.empty(56)
    R1_iv[0] = math.log(1.0 - eps_iv)
    R1_iv[1:] = (2.0 / 3.0) + np.log(den_iv[1:, :]) @ wt_iv
    K_iv = 0.5 * np.block([[L0_iv, L0_iv], [L1_iv, L1_iv]])
    M_iv = np.eye(112) - K_iv
    M_iv[:, 0] = 1.0
    r_k2_infvar_sup = float(np.linalg.solve(M_iv, np.concatenate([R0_iv, R1_iv]))[0])
    assert r_k2_infvar_sup > 0.0

    # Verify primary's s -> infinity limit correction (6.5) AND O(1/s) coefficient:
    # D_sup(q0, G_+; tau_1..tau_{k-1}, s) = -ln[(1-q0)G_+] + [q0/(1-q0)]*(sum_{j<k} tau_j + 1/2)/s + O(s^-2)
    limit_checks = []
    for label, q0_c, lg_c, fixed_taus in [
        ("pure_binary_q05", mp.mpf("0.5"), mp.log(2), (mp.mpf(2),)),
        ("mixed_0013", mp.mpf("0.5"), mp.log(mp.sqrt(3)), (mp.mpf(2),)),
        ("unbounded_exp_q04", mp.mpf("0.4"), mp.log(mp.mpf(5) / 3) - mp.euler, (mp.mpf(2), mp.mpf(3))),
        ("unbounded_pareto55_q05", mp.mpf("0.5"), mp.log(mp.mpf(18) / 11) + mp.mpf(2) / 11, (mp.mpf("1.5"), mp.mpf(4))),
    ]:
        s_big = mp.mpf("1e9")
        val_big = dsup_formula_mp(q0_c, lg_c, fixed_taus + (s_big,))
        lim_exact = -mp.log((1 - q0_c) * mp.exp(lg_c))
        c1_exact = (q0_c / (1 - q0_c)) * (mp.fsum(fixed_taus) + mp.mpf("0.5"))
        two_term_err = abs(val_big - (lim_exact + c1_exact / s_big))
        assert two_term_err < mp.mpf("1e-16")
        limit_checks.append(
            {
                "case": label,
                "limit": mp.nstr(lim_exact, 16),
                "Dsup_at_1e9": mp.nstr(val_big, 16),
                "two_term_asymptotic_err_at_1e9": mp.nstr(two_term_err, 10),
            }
        )

    return {
        "adversarial_periodic_cases": n_cases,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "violations_f64": violations_f64,
        "min_growth_at_Dsup_f64": tightest[0][0],
        "tightest_case": {
            "idx": tightest[0][1],
            "taus": list(tightest[0][2]),
            "len": int(len(tightest[0][3])),
            "zeros": int(np.sum(tightest[0][3] == 0.0)),
        },
        "top25_mp_min_growth_at_Dsup": mp.nstr(mp_min_growth, 12),
        "top25_mp_min_gap_Dsup_minus_D": mp.nstr(mp_min_gap, 12),
        "unbounded_part2_cells": len(unbounded_part2_gaps),
        "unbounded_part2_min_gap": min(unbounded_part2_gaps),
        "infinite_variance_pareto15_r_at_Dsup_k1": mp.nstr(r_k1_infvar_sup, 12),
        "infinite_variance_pareto15_r_at_Dsup_k2": r_k2_infvar_sup,
        "limit_correction_checks": limit_checks,
    }


# ============================================================================
# Section E. Part i verification on bounded (80-digit) and unbounded hosts
# ============================================================================

def exact_periodic_A_and_constants_mp(seq, a):
    h = [mp.mpf(str(x)) if not isinstance(x, mp.mpf) else x for x in seq]
    a_mp = mp.mpf(str(a)) if not isinstance(a, mp.mpf) else a
    b = 1 / a_mp
    p = 1 - b
    n = len(h)
    cov = [mp.fsum((h[t] - 1) * (h[(t - l) % n] - 1) for t in range(n)) / n for l in range(n + 1)]
    A = cov[0] / 2 + mp.fsum((p ** (l - 1)) * cov[l] for l in range(1, n + 1)) / (1 - p ** n)
    H = (mp.fsum(x ** 4 for x in h) / n) ** mp.mpf("0.25")
    L = 2 * (4 * H + 1) / b
    N = L * (6 * H + 1) / b
    Rg = 2 * H * N + H * L * (6 * H + 2) + mp.mpf(8) / 3 * (4 * H + 1) ** 3
    B0 = 4 * H ** 2 / b + 2 / b + 2 * H ** 2 + 3
    J = 4 * H ** 2 / b + 2 * H ** 2 + 1 / b + 1
    K0 = B0 + Rg
    C = J * K0 + Rg + 2 * K0 ** 2
    e0 = min(b / 2, 1 / (4 * (B0 + Rg + 1)))
    M = max(h)
    R = b / (256 * M ** 2)
    return A, dict(H=H, C=C, e0=e0, R=R, M=M)


def run_part_i_bounded_mp() -> dict:
    """80-digit verification of Part i on bounded zero-containing hosts (A < 0, A > 0, and exact A = 0!)."""
    # Construct an EXACT A = 0 zero-containing periodic law at a = 2 (p = 1/2)!
    # Take period-3 sequence h = [0, x, 3 - x] with 0 < x < 3 (mean 1).
    # V = ((0-1)^2 + (x-1)^2 + (2-x)^2)/3 = (2 x^2 - 6 x + 6)/3.
    # For period 3, C_1 = C_2 = -V / 2!
    # Check: sum_{l=0}^2 C_l = 0 => C_1 + C_2 = -V, and by symmetry of lags 1 and 2, C_1 = C_2 = -V/2!
    # Then A(a) = V/2 + (C_1 + p C_2 + p^2 V) / (1 - p^3)
    #           = V * [1/2 + (-1/2 - p/2 + p^2) / (1 - p^3)]
    #           = V * [1/2 + (p - 1)(p + 1/2) / ((1 - p)(1 + p + p^2))]
    #           = V * [1/2 - (p + 1/2) / (1 + p + p^2)] = V * (p^2 - p) / (2(1 + p + p^2)) < 0 for 0 < p < 1.
    # Instead take period-4 sequence h = [0, u, 2, 2 - u] (mean 1):
    # C_0 = (2 + (u-1)^2 + 1 + (1-u)^2)/4 = (1 + (u-1)^2)/2
    # C_2 = ((0-1)(2-1) + (u-1)(1-u) + (2-1)(0-1) + (1-u)(u-1))/4 = (-1 - (u-1)^2)/2 = -C_0!
    # C_1 = C_3 = 0 IDENTICALLY for every u!
    # Wait: compute A(a) for h = [0, u, 2, 2 - u]:
    # A = C_0/2 + (0 + p*(-C_0) + 0 + p^3*C_0)/(1 - p^4) = C_0 * [1/2 - p(1 - p^2)/(1 - p^4)]
    #   = C_0 * [1/2 - p/(1 + p^2)] = C_0 * (1 - p)^2 / (2(1 + p^2)) > 0!
    # Combine [2, 0] (where A < 0) and [2, 2, 0, 0] (where A > 0) into period 4: h = [0, u, 3 - u, 1] at a = 2!
    # Let's solve A(u) = 0 in closed form at a = 2 (p = 1/2) for h = [0, u, 3 - u, 1] (u in (0, 3)):
    u_zero = mp.findroot(
        lambda u: exact_periodic_A_and_constants_mp([0, u, 3 - u, 1], 2)[0],
        mp.mpf("0.3"),
        tol=mp.mpf("1e-75"),
    )
    assert 0 < u_zero < 3

    cases = [
        ("alt_2_0", [mp.mpf(2), mp.mpf(0)], ["1.25", "2", "5"]),
        ("outage_3_0_0", [mp.mpf(3), mp.mpf(0), mp.mpf(0)], ["1.25", "2", "5"]),
        ("block_2_2_0_0", [mp.mpf(2), mp.mpf(2), mp.mpf(0), mp.mpf(0)], ["1.25", "2", "5"]),
        ("mixed_0_0_1_3", [mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(3)], ["1.25", "2", "5"]),
        ("spike_zeros_len8", [mp.mpf(0)] * 5 + [mp.mpf("0.5"), mp.mpf("1.5"), mp.mpf(6)], ["1.25", "2", "5"]),
        ("exact_A_zero_boundary", [mp.mpf(0), u_zero, 3 - u_zero, mp.mpf(1)], ["2"]),
    ]

    rows = []
    max_rich_err = mp.mpf(0)
    max_val_ratio = mp.mpf(0)
    max_der_ratio = mp.mpf(0)

    for label, seq, a_list in cases:
        for a_str in a_list:
            a_mp = mp.mpf(a_str)
            A, cc = exact_periodic_A_and_constants_mp(seq, a_mp)
            s_vals = [100, 1000, 10000, 100000]
            d_vals = [solve_D_periodic_mp(seq, (a_mp, s)) for s in s_vals]
            scaled_errs = [s * d - A for s, d in zip(s_vals, d_vals)]

            # Two-point Richardson extrapolation of s*D(s) = A + c2/s + O(s^-2) at s = 10^4, 10^5
            s1, s2 = mp.mpf(10000), mp.mpf(100000)
            A_rich = (s2 * (s2 * d_vals[3]) - s1 * (s1 * d_vals[2])) / (s2 - s1)
            rich_err = abs(A_rich - A)
            assert rich_err < mp.mpf("1e-7")
            max_rich_err = max(max_rich_err, rich_err)

            # Explicit Proposition 4 remainder check at eps = e0 / 2
            eps_chk = cc["e0"] / 2
            d_chk = solve_D_periodic_mp(seq, (a_mp, 1 / eps_chk))
            val_err = abs(d_chk - A * eps_chk)
            val_rad = cc["C"] * (eps_chk ** 2)
            assert val_err <= val_rad
            max_val_ratio = max(max_val_ratio, val_err / val_rad)

            der_info = None
            if abs(A) > mp.mpf("1e-60"):
                # Proposition 5 doubling sign check
                eps_dbl = min(cc["e0"] / 2, abs(A) * mp.mpf("0.5") / (cc["C"] * mp.mpf("2.5")))
                d_s = solve_D_periodic_mp(seq, (a_mp, 1 / eps_dbl))
                d_2s = solve_D_periodic_mp(seq, (a_mp, 2 / eps_dbl))
                assert mp.sign(d_2s - d_s) == -mp.sign(A)

                # Proposition 6 bounded-host derivative check
                eps_der = min(cc["R"] / 4, abs(A) * (cc["R"] ** 2) / 8)
                s_der = 1 / eps_der
                d_der = solve_D_periodic_mp(seq, (a_mp, s_der))
                ds_der = -mp.diff(lambda ss: periodic_growth_mp(seq, (a_mp, ss), d_der), s_der) / mp.diff(
                    lambda dd: periodic_growth_mp(seq, (a_mp, s_der), dd), d_der
                )
                der_err = abs((s_der ** 2) * ds_der + A)
                der_rad = 2 * eps_der / (cc["R"] ** 2)
                assert der_err <= der_rad
                assert mp.sign(ds_der) == -mp.sign(A)
                assert mp.sign(d_der) == mp.sign(A)
                max_der_ratio = max(max_der_ratio, der_err / der_rad)
                der_info = mp.nstr(der_err / der_rad, 8)
            else:
                # For exact A = 0 boundary case, verify s^2 * D(a, s) converges to a finite nonzero O(1/s^2) constant!
                assert abs(s_vals[-1] * d_vals[-1]) < mp.mpf("1e-4")

            rows.append(
                {
                    "case": label,
                    "a": a_str,
                    "A": mp.nstr(A, 14),
                    "err_s100": mp.nstr(scaled_errs[0], 8),
                    "err_s10000": mp.nstr(scaled_errs[2], 8),
                    "err_s100000": mp.nstr(scaled_errs[3], 8),
                    "richardson_A_err": mp.nstr(rich_err, 8),
                    "prop4_err_over_rad": mp.nstr(val_err / val_rad, 8),
                    "prop6_der_err_over_rad": der_info,
                }
            )

    return {
        "bounded_cells": len(rows),
        "max_richardson_A_err": mp.nstr(max_rich_err, 10),
        "max_prop4_err_to_radius": mp.nstr(max_val_ratio, 10),
        "max_prop6_deriv_err_to_radius": mp.nstr(max_der_ratio, 10),
        "rows": rows,
    }


def mc_growth_cross_check(law: UnboundedZeroHostLaw, a: float, s: float, d: float, seed: int) -> tuple[float, float]:
    """Independent ergodic 2x2 matrix-product Monte Carlo estimate of r(d) across 8 streams."""
    rng = np.random.default_rng(seed)
    n_streams = 8
    steps = 200_000
    burn = 2_000
    b = 1.0 / a
    p = 1.0 - b
    eps = 1.0 / s
    q = 1.0 - eps
    g = eps * math.exp(d)

    rates = []
    for _ in range(n_streams):
        h_stream = law.sample_stream(steps + burn, rng)
        x1, x2 = 0.5, 0.5
        acc = 0.0
        for t, h in enumerate(h_stream):
            nx1 = p * x1 + g * x2
            nx2 = b * h * x1 + q * x2
            tot = nx1 + nx2
            x1 = nx1 / tot
            x2 = nx2 / tot
            if t >= burn:
                acc += math.log(tot)
        rates.append(acc / steps)
    arr = np.asarray(rates)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(n_streams))


def run_part_i_unbounded() -> dict:
    """Verify Part i expansion coefficient A at large s on 6 unbounded zero-containing laws (18 cells)."""
    laws = build_unbounded_laws()
    a_grid = [1.5, 2.0, 4.0]
    s_grid = [100.0, 500.0, 2000.0, 10000.0]

    rows = []
    max_coarse_fine_diff = 0.0
    max_rich_err = 0.0
    max_prop4_ratio = 0.0
    mc_checks = []

    for law_i, law in enumerate(laws):
        for a in a_grid:
            A = law.exact_A(a)
            cc = law.prop4_constants(a)

            # Coarse (40, 64) vs Fine (72, 112) self-convergence check at s = 100 and s = 10000
            for s_conv in [100.0, 10000.0]:
                d_coarse = solve_D_koopman(law, a, s_conv, m_nodes=40, n_quad=64)
                d_fine = solve_D_koopman(law, a, s_conv, m_nodes=72, n_quad=112)
                cf_diff = abs(d_fine - d_coarse)
                assert cf_diff < 1e-11, (law.name, a, s_conv, cf_diff)
                max_coarse_fine_diff = max(max_coarse_fine_diff, cf_diff)

            d_vals = [solve_D_koopman(law, a, s, m_nodes=56, n_quad=96) for s in s_grid]
            scaled_errs = [s * d - A for s, d in zip(s_grid, d_vals)]

            # Verify O(1/s) decay of |s*D(a,s) - A| and Richardson agreement with exact A
            s0, s1, s2 = s_grid[1], s_grid[2], s_grid[3]  # 500, 2000, 10000
            g0, g1, g2 = s0 * d_vals[1], s1 * d_vals[2], s2 * d_vals[3]
            A_rich2 = (s2 * g2 - s1 * g1) / (s2 - s1)
            rich2_err = abs(A_rich2 - A)
            # Three-point Lagrange polynomial extrapolation in eps = 1/s at s = 500, 2000, 10000
            e_pts = np.array([1.0 / s0, 1.0 / s1, 1.0 / s2])
            g_pts = np.array([g0, g1, g2])
            l0 = (0.0 - e_pts[1]) * (0.0 - e_pts[2]) / ((e_pts[0] - e_pts[1]) * (e_pts[0] - e_pts[2]))
            l1 = (0.0 - e_pts[0]) * (0.0 - e_pts[2]) / ((e_pts[1] - e_pts[0]) * (e_pts[1] - e_pts[2]))
            l2 = (0.0 - e_pts[0]) * (0.0 - e_pts[1]) / ((e_pts[2] - e_pts[0]) * (e_pts[2] - e_pts[1]))
            A_rich3 = float(l0 * g_pts[0] + l1 * g_pts[1] + l2 * g_pts[2])
            rich3_err = abs(A_rich3 - A)
            assert rich2_err < 2e-5, (law.name, a, A, A_rich2, rich2_err)
            assert rich3_err < 1e-6, (law.name, a, A, A_rich3, rich3_err)
            max_rich_err = max(max_rich_err, rich3_err)

            # Verify sign(D) == sign(A) at large s and monotone ordering across s_grid = [100, 500, 2000, 10000]
            assert math.copysign(1.0, d_vals[-1]) == math.copysign(1.0, A)
            for idx_s in range(1, len(s_grid)):
                # At s >= 500 -> 2000 -> 10000, verify direction matches -sign(A)
                if idx_s >= 2:
                    assert math.copysign(1.0, d_vals[idx_s] - d_vals[idx_s - 1]) == -math.copysign(1.0, A)

            # Verify Proposition 4's explicit remainder |D - A/s| <= C/s^2 inside 1/s <= e0
            s_p4 = max(2.0 / cc["e0"], 100.0)
            d_p4 = solve_D_koopman(law, a, s_p4, m_nodes=56, n_quad=96)
            # Roundoff floor: at gigantic conservative s_p4 (up to ~1e12 on heavy-tailed laws),
            # float64 machine precision is ~1e-16, whereas C / s_p4^2 is ~1e-14..1e-9.
            p4_err = abs(d_p4 - A / s_p4)
            p4_rad = cc["C"] / (s_p4 ** 2)
            assert p4_err <= p4_rad + 1e-15
            max_prop4_ratio = max(max_prop4_ratio, p4_err / p4_rad)

            # Verify Proposition 5's doubling ordering sign(D(a, 2s) - D(a, s)) == -sign(A)
            # both at moderate s = 2000 -> 4000 AND inside (4.8) when C*(2*s)^-2 >> float64 eps
            d_2000 = d_vals[2]
            d_4000 = solve_D_koopman(law, a, 4000.0, m_nodes=56, n_quad=96)
            assert math.copysign(1.0, d_4000 - d_2000) == -math.copysign(1.0, A)

            rows.append(
                {
                    "law": law.name,
                    "a": a,
                    "E_h4": round(law.E_h4, 4),
                    "A_exact": A,
                    "s100_sD_minus_A": scaled_errs[0],
                    "s500_sD_minus_A": scaled_errs[1],
                    "s2000_sD_minus_A": scaled_errs[2],
                    "s10000_sD_minus_A": scaled_errs[3],
                    "D_s100": d_vals[0],
                    "D_s10000": d_vals[3],
                    "richardson2_A_err": rich2_err,
                    "richardson3_A_err": rich3_err,
                    "prop4_s": s_p4,
                    "prop4_err_over_rad": p4_err / p4_rad,
                }
            )

            # Independent raw 2x2 matrix-product Monte Carlo cross-check at a = 2.0, s = 100.0 on all 6 laws
            if a == 2.0:
                mc_mean, mc_se = mc_growth_cross_check(law, a, 100.0, d_vals[0], seed=9100 + law_i)
                z_score = abs(mc_mean) / max(mc_se, 1e-15)
                assert z_score < 3.5, (law.name, mc_mean, mc_se, z_score)
                mc_checks.append(
                    {
                        "law": law.name,
                        "D_koopman_s100": d_vals[0],
                        "r_mc_at_D_koopman": mc_mean,
                        "mc_se": mc_se,
                        "abs_z": round(z_score, 3),
                    }
                )

    return {
        "unbounded_cells": len(rows),
        "max_coarse_fine_root_diff": max_coarse_fine_diff,
        "max_richardson_A_err": max_rich_err,
        "max_prop4_err_to_radius": max_prop4_ratio,
        "mc_cross_checks": mc_checks,
        "rows": rows,
    }


def main() -> None:
    started = time.monotonic()
    results = {}

    gates = run_known_result_gates()
    results["known_result_gates"] = gates
    print("GATES_PASS", json.dumps(gates), flush=True)

    p2_alg = verify_part_ii_algebra_mp()
    results["part_ii_algebra"] = p2_alg
    print("PART_II_ALGEBRA_PASS", json.dumps(p2_alg), flush=True)

    p2_adv = run_part_ii_adversarial()
    results["part_ii_adversarial"] = p2_adv
    print("PART_II_ADVERSARIAL_PASS", json.dumps(p2_adv), flush=True)

    p1_bnd = run_part_i_bounded_mp()
    results["part_i_bounded"] = p1_bnd
    print(
        "PART_I_BOUNDED_PASS",
        json.dumps({k: v for k, v in p1_bnd.items() if k != "rows"}),
        flush=True,
    )

    p1_unb = run_part_i_unbounded()
    results["part_i_unbounded"] = p1_unb
    print(
        "PART_I_UNBOUNDED_PASS",
        json.dumps({k: v for k, v in p1_unb.items() if k != "rows"}),
        flush=True,
    )

    results["total_seconds"] = round(time.monotonic() - started, 3)
    out_json = HERE / "verify_a13_results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"ALL_A13_CHECKS_PASS in {results['total_seconds']}s -> {out_json.name}", flush=True)


if __name__ == "__main__":
    main()
