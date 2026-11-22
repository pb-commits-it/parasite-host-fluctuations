#!/usr/bin/env python3
"""Independent high-precision verification of GAP 2 (small-staying correction).

Author: independent 4 secondary (job r37-gap2-check, 2026-11-15)
Rules:
  - Zero imports from gap2/gap2_checks.py or any project code.
  - Pre-registered known-result benchmarks run BEFORE any GAP 2 grid evaluation
    (study rules Rule 7).
  - Independent numerical methods:
      (M1) Exact analytic power series in delta (memA Prop. P7) with zero spatial
           discretization at moderate s.
      (M2) Discontinuous piecewise Gauss-Legendre Nystrom solver (strictly interior
           roots of Legendre polynomials, solving BOTH the primal Bellman/Poisson
           system (I - K)V + q*1 = r_Q and the adjoint invariant linear functional
           (K^T - I)mu = 0 in 55-digit and 70-digit mpmath).
      (M3) Strictly positive row-stochastic piecewise-linear Markov chain (mu >= 0,
           TV = 1) at moderate L to verify zero collocation-sign artifact.
      (M4) Exhaustive pointwise & stationary check of every identity and inequality
           in gap2_proof.md Eqs. (1)-(11), W_t/xi_t martingale-difference audit,
           and constant-host threshold brackets, across interior and edge-case
           parameters (delta in [0.05, 0.95], phi in [-0.9, +0.9], s down to 10^-60).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
primary_RESULTS_JSON = (__import__('pathlib').Path(__file__).resolve().parent / '../../gap2/gap2_results.json').resolve()


# ---------------------------------------------------------------------------
# Basic arbitrary-precision helpers
# ---------------------------------------------------------------------------

def log1pexp(x: mp.mpf) -> mp.mpf:
    """Stable log(1 + exp(x)) in current mpmath precision."""
    if x > 0:
        return x + mp.log1p(mp.exp(-x))
    return mp.log1p(mp.exp(x))


def gauss_legendre_rule(n_pts: int, dps: int) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Compute interior Gauss-Legendre nodes on (-1, 1) and barycentric weights."""
    with mp.workdps(dps + 15):
        nodes: list[mp.mpf] = []
        for j in range(n_pts):
            # Initial asymptotic guess, ordered strictly increasing on (-1, 1)
            theta = mp.pi * (n_pts - j - mp.mpf("0.25")) / (n_pts + mp.mpf("0.5"))
            u = mp.cos(theta)
            for _ in range(12):
                p_prev = mp.mpf(1)
                p_curr = u
                for k in range(1, n_pts):
                    p_next = ((2 * k + 1) * u * p_curr - k * p_prev) / (k + 1)
                    p_prev, p_curr = p_curr, p_next
                dp = n_pts * (u * p_curr - p_prev) / (u * u - 1)
                du = -p_curr / dp
                u += du
                if abs(du) < mp.power(10, -(dps + 10)):
                    break
            nodes.append(u)
        nodes.sort()
        bary: list[mp.mpf] = []
        for j in range(n_pts):
            prod = mp.mpf(1)
            uj = nodes[j]
            for m in range(n_pts):
                if m != j:
                    prod *= (uj - nodes[m])
            bary.append(1 / prod)
    with mp.workdps(dps):
        return [+u for u in nodes], [+w for w in bary]


_GL_CACHE: dict[tuple[int, int], tuple[list[mp.mpf], list[mp.mpf]]] = {}


def get_gl_rule(n_pts: int, dps: int) -> tuple[list[mp.mpf], list[mp.mpf]]:
    key = (n_pts, dps)
    if key not in _GL_CACHE:
        _GL_CACHE[key] = gauss_legendre_rule(n_pts, dps)
    return _GL_CACHE[key]


# ---------------------------------------------------------------------------
# Method M1: Exact analytic delta-series (memA Prop. P7, zero spatial mesh)
# ---------------------------------------------------------------------------

def analytic_series_rate(s: mp.mpf, c: mp.mpf, delta: mp.mpf, phi: mp.mpf,
                         order: int = 44, dps: int = 60) -> tuple[mp.mpf, mp.mpf]:
    """Compute r(s, c) via exact triangular moment recursion and Cauchy tail bound."""
    with mp.workdps(dps):
        lam = (s + mp.sqrt(s * s + 4 * c)) / 2
        b_det = c / (lam * lam)
        rad = (1 - b_det) ** 2 / (8 * b_det)
        if abs(delta) >= rad:
            raise ValueError(f"|delta|={abs(delta)} >= analytic radius {rad}")
        # M[k][n] = [M_{k,n,-}, M_{k,n,+}], 0 <= k <= n <= order
        M: list[list[tuple[mp.mpf, mp.mpf]]] = [
            [(mp.mpf(0), mp.mpf(0)) for _ in range(order + 1)]
            for _ in range(order + 1)
        ]
        half = mp.mpf("0.5")
        M[0][0] = (half, half)
        p_same = (1 + phi) / 2
        p_flip = (1 - phi) / 2

        binom = [[mp.mpf(math.comb(a, b)) if b <= a else mp.mpf(0)
                  for b in range(order + 2)] for a in range(2 * order + 2)]

        for n in range(1, order + 1):
            for k in range(n, 0, -1):
                bk = b_det ** k
                mbk = (-b_det) ** k
                rhs_minus = mp.mpf(0)
                rhs_plus = mp.mpf(0)
                for r in range(k + 1):
                    max_m = n - k
                    for m in range(max_m + 1):
                        if r == 0 and m == 0:
                            continue
                        idx_k = k - r + m
                        idx_n = n - r
                        if idx_k > idx_n:
                            continue
                        v_minus, v_plus = M[idx_k][idx_n]
                        if v_minus == 0 and v_plus == 0:
                            continue
                        # Apply P^T
                        pt_minus = p_same * v_minus + p_flip * v_plus
                        pt_plus = p_flip * v_minus + p_same * v_plus
                        coef = bk * binom[k][r] * ((-1) ** (k - r + m)) * binom[k + m - 1][m]
                        rhs_minus += coef * ((-1) ** r) * pt_minus
                        rhs_plus += coef * pt_plus
                # Solve (I - (-b)^k P^T) u = rhs in parity eigenbasis
                s_rhs = rhs_minus + rhs_plus
                d_rhs = rhs_plus - rhs_minus
                s_sol = s_rhs / (1 - mbk)
                d_sol = d_rhs / (1 - mbk * phi)
                M[k][n] = ((s_sol - d_sol) / 2, (s_sol + d_sol) / 2)

        r_val = mp.log(lam)
        for n in range(1, order + 1):
            inner = mp.mpf(0)
            for k in range(1, n + 1):
                m_minus, m_plus = M[k][n]
                inner += ((-1) ** (k + 1)) * (m_minus + m_plus) / k
            r_val += (delta ** n) * inner
        ratio = abs(delta) / rad
        tail_bound = mp.log(2) * (ratio ** (order + 1)) / (1 - ratio)
        return r_val, tail_bound


def analytic_series_threshold(s_str: str, delta_str: str, phi_str: str,
                              order: int = 44, dps: int = 60) -> tuple[mp.mpf, mp.mpf]:
    """Find D(s) purely from the analytic delta-series."""
    with mp.workdps(dps):
        s = mp.mpf(s_str)
        delta = mp.mpf(delta_str)
        phi = mp.mpf(phi_str)
        c0 = 1 - s

        def f(y: mp.mpf) -> mp.mpf:
            val, _ = analytic_series_rate(s, c0 * mp.exp(y), delta, phi, order=order, dps=dps)
            return val

        y_star = mp.findroot(f, (mp.mpf(0), mp.mpf("1e-5")), solver="secant",
                             tol=mp.power(10, -(dps - 10)))
        _, tail = analytic_series_rate(s, c0 * mp.exp(y_star), delta, phi, order=order, dps=dps)
        return y_star, tail


# ---------------------------------------------------------------------------
# Method M2: Discontinuous Gauss-Legendre Primal Bellman + Adjoint Solver
# ---------------------------------------------------------------------------

class GLKoopmanOperator:
    """Piecewise Gauss-Legendre discretization of the normalized Koopman operator K."""

    def __init__(self, L: mp.mpf, delta: mp.mpf, phi: mp.mpf,
                 n_pts: int = 14, max_h: float = 1.2, dps: int = 55):
        with mp.workdps(dps):
            self.dps = dps
            self.L = +L
            self.delta = +delta
            self.phi = +phi
            self.b = mp.atanh(self.delta)
            self.lo = -self.L
            self.hi = self.L + self.b + mp.log1p(mp.exp(-(2 * self.L + self.b)))
            span = self.hi - self.lo
            self.n_panels = max(1, int(mp.ceil(span / mp.mpf(str(max_h)))))
            self.step = span / self.n_panels
            self.n_pts = n_pts

            ref_nodes, ref_bary = get_gl_rule(n_pts, dps)
            self.nodes = [
                self.lo + self.step * (p + (u + 1) / 2)
                for p in range(self.n_panels)
                for u in ref_nodes
            ]
            self.n = len(self.nodes)
            self.size = 2 * self.n

            p_same = (1 + self.phi) / 2
            p_flip = (1 - self.phi) / 2

            # Store K row-wise as two contiguous blocks of length n_pts
            self.blocks: list[tuple[int, list[mp.mpf], int, list[mp.mpf]]] = []
            self.r_q: list[mp.mpf] = []
            self.z_state: list[int] = []
            self.q_pair: list[tuple[mp.mpf, mp.mpf]] = []
            self.target_pair: list[tuple[mp.mpf, mp.mpf]] = []

            tol_hit = mp.power(10, -(dps - 8))
            for i_state, z_prev in enumerate((-1, 1)):
                for x in self.nodes:
                    probs = (p_same, p_flip) if i_state == 0 else (p_flip, p_same)
                    bases = []
                    wlists = []
                    qs = []
                    targs = []
                    for j_state, z_curr in enumerate((-1, 1)):
                        eta = self.b * z_curr
                        q_val = log1pexp(x - eta - self.L)
                        targ = -x + eta + q_val
                        panel = min(self.n_panels - 1,
                                    max(0, int(mp.floor((targ - self.lo) / self.step))))
                        u_loc = 2 * ((targ - self.lo) / self.step - panel) - 1
                        diffs = [u_loc - u_ref for u_ref in ref_nodes]
                        hit = None
                        for idx_d, d_val in enumerate(diffs):
                            if abs(d_val) < tol_hit:
                                hit = idx_d
                                break
                        if hit is not None:
                            interp = [probs[j_state] if idx_d == hit else mp.mpf(0)
                                      for idx_d in range(n_pts)]
                        else:
                            raw = [ref_bary[idx_d] / diffs[idx_d] for idx_d in range(n_pts)]
                            denom = mp.fsum(raw)
                            scale = probs[j_state] / denom
                            interp = [val * scale for val in raw]
                        bases.append(j_state * self.n + panel * n_pts)
                        wlists.append(interp)
                        qs.append(q_val)
                        targs.append(targ)
                    self.blocks.append((bases[0], wlists[0], bases[1], wlists[1]))
                    self.r_q.append(probs[0] * qs[0] + probs[1] * qs[1])
                    self.z_state.append(z_prev)
                    self.q_pair.append((qs[0], qs[1]))
                    self.target_pair.append((targs[0], targs[1]))

    def solve_primal_bellman(self) -> tuple[mp.mpf, mp.mpf]:
        """Solve (I - K)V + q*1 = r_Q with V[0] = 0. Unknowns: [q, V[1], ..., V[size-1]]."""
        with mp.workdps(self.dps):
            sz = self.size
            n_pts = self.n_pts
            # Build float64 sparse matrix for preconditioner
            rr, cc, vv = [], [], []
            for r in range(sz):
                # Column 0 is q (coefficient +1)
                rr.append(r)
                cc.append(0)
                vv.append(1.0)
                if r > 0:
                    rr.append(r)
                    cc.append(r)
                    vv.append(1.0)
                b0, w0, b1, w1 = self.blocks[r]
                for m in range(n_pts):
                    col0 = b0 + m
                    if col0 > 0:
                        rr.append(r)
                        cc.append(col0)
                        vv.append(-float(w0[m]))
                    col1 = b1 + m
                    if col1 > 0:
                        rr.append(r)
                        cc.append(col1)
                        vv.append(-float(w1[m]))
            lu = splu(coo_matrix((vv, (rr, cc)), shape=(sz, sz)).tocsc())
            rhs_f = np.array([float(v) for v in self.r_q], dtype=float)
            sol_f = lu.solve(rhs_f)
            u = [mp.mpf(float(val)) for val in sol_f]

            target_res = mp.power(10, -(self.dps - 8))
            res_norm = mp.inf
            for _ in range(10):
                q_est = u[0]
                err = []
                for r in range(sz):
                    b0, w0, b1, w1 = self.blocks[r]
                    kv = mp.fsum(
                        w0[m] * (u[b0 + m] if b0 + m > 0 else mp.mpf(0))
                        + w1[m] * (u[b1 + m] if b1 + m > 0 else mp.mpf(0))
                        for m in range(n_pts)
                    )
                    vr = u[r] if r > 0 else mp.mpf(0)
                    lhs = vr - kv + q_est
                    err.append(self.r_q[r] - lhs)
                res_norm = max(abs(e) for e in err)
                if res_norm < target_res:
                    break
                corr = lu.solve(np.array([float(e / res_norm) for e in err], dtype=float))
                u = [u_i + res_norm * mp.mpf(float(c_i)) for u_i, c_i in zip(u, corr)]
            return u[0], res_norm

    def solve_adjoint_invariant(self) -> dict[str, mp.mpf]:
        """Solve (K^T - I)mu = 0, sum(mu) = 1, and evaluate all proof identities/bounds."""
        with mp.workdps(self.dps):
            sz = self.size
            n_pts = self.n_pts
            rr, cc, vv = [], [], []
            for r in range(sz):
                b0, w0, b1, w1 = self.blocks[r]
                for m in range(n_pts):
                    col0 = b0 + m
                    if col0 < sz - 1:
                        rr.append(col0)
                        cc.append(r)
                        vv.append(float(w0[m]))
                    col1 = b1 + m
                    if col1 < sz - 1:
                        rr.append(col1)
                        cc.append(r)
                        vv.append(float(w1[m]))
            for i in range(sz - 1):
                rr.append(i)
                cc.append(i)
                vv.append(-1.0)
            for i in range(sz):
                rr.append(sz - 1)
                cc.append(i)
                vv.append(1.0)

            lu = splu(coo_matrix((vv, (rr, cc)), shape=(sz, sz)).tocsc())
            rhs_f = np.zeros(sz, dtype=float)
            rhs_f[-1] = 1.0
            mu = [mp.mpf(float(val)) for val in lu.solve(rhs_f)]

            target_res = mp.power(10, -(self.dps - 8))
            res_norm = mp.inf
            for _ in range(10):
                kt_mu = [mp.mpf(0)] * (sz - 1)
                for r in range(sz):
                    mur = mu[r]
                    b0, w0, b1, w1 = self.blocks[r]
                    for m in range(n_pts):
                        c0 = b0 + m
                        if c0 < sz - 1:
                            kt_mu[c0] += mur * w0[m]
                        c1 = b1 + m
                        if c1 < sz - 1:
                            kt_mu[c1] += mur * w1[m]
                err = [-(kt_mu[i] - mu[i]) for i in range(sz - 1)]
                err.append(mp.mpf(1) - mp.fsum(mu))
                res_norm = max(abs(e) for e in err)
                if res_norm < target_res:
                    break
                corr = lu.solve(np.array([float(e / res_norm) for e in err], dtype=float))
                mu = [m_i + res_norm * mp.mpf(float(c_i)) for m_i, c_i in zip(mu, corr)]

            # Evaluate all moments & identities in gap2_proof.md
            p_same = (1 + self.phi) / 2
            p_flip = (1 - self.phi) / 2
            b = self.b
            phi = self.phi
            L = self.L

            q_L = mp.fsum(mu[r] * self.r_q[r] for r in range(sz))
            ex = mp.fsum(mu[r] * self.nodes[r % self.n] for r in range(sz))
            # C = E[X_{t-1} eta_{t-1}] = E[X_t eta_t] by stationarity
            c_same_time = mp.fsum(
                mu[r] * self.nodes[r % self.n] * (b * self.z_state[r])
                for r in range(sz)
            )
            e_xprev_etat = mp.mpf(0)
            e_etat_qt = mp.mpf(0)
            e_xt_etat = mp.mpf(0)
            e_diff_sq = mp.mpf(0)
            e_eq6_rhs = mp.mpf(0)
            e_w_xi = mp.mpf(0)
            e_xi_sq = mp.mpf(0)
            e_w_rhs = mp.mpf(0)
            h_L = mp.mpf(0)

            for r in range(sz):
                mur = mu[r]
                x_prev = self.nodes[r % self.n]
                z_prev = self.z_state[r]
                eta_prev = b * z_prev
                probs = (p_same, p_flip) if z_prev == -1 else (p_flip, p_same)
                q0, q1 = self.q_pair[r]
                t0, t1 = self.target_pair[r]
                w_prev = x_prev - (phi / (1 + phi)) * eta_prev
                y_val = x_prev - L

                for pr, z_curr, qv, xt in ((probs[0], -1, q0, t0), (probs[1], 1, q1, t1)):
                    wt = mur * pr
                    eta_t = b * z_curr
                    xi_t = (eta_t - phi * eta_prev) / (1 + phi)
                    e_xprev_etat += wt * (x_prev * eta_t)
                    e_etat_qt += wt * (eta_t * qv)
                    e_xt_etat += wt * (xt * eta_t)
                    e_diff_sq += wt * (xt * xt - x_prev * x_prev)
                    e_eq6_rhs += wt * (2 * x_prev - 2 * eta_t / (1 + phi) - qv) * qv
                    h_L += wt * (2 * y_val - 2 * eta_t / (1 + phi) - qv) * qv
                    e_w_xi += wt * (w_prev * xi_t)
                    e_xi_sq += wt * (xi_t * xi_t)
                    e_w_rhs += wt * (2 * w_prev - 2 * xi_t - qv) * qv

            v_pi = b * b * (1 - phi) / (1 + phi)
            q_star = log1pexp(2 * b + 1)
            c_plus = 2 * (b + 1) + 2 * b / (1 + phi)
            c_minus = 2 * b / (1 + phi) + q_star

            return dict(
                q_L=q_L,
                ex=ex,
                res_linear=res_norm,
                res_eq4=abs(2 * ex - q_L),
                res_markov_cov=abs(e_xprev_etat - phi * c_same_time),
                res_stationarity_C=abs(e_xt_etat - c_same_time),
                res_eq5=abs((1 + phi) * e_xt_etat - (b * b + e_etat_qt)),
                res_sq_diff=abs(e_diff_sq),
                res_eq6=abs(e_eq6_rhs - v_pi),
                res_eq7=abs(2 * L * q_L + h_L - v_pi),
                res_w_martingale=abs(e_w_xi),
                res_xi_var=abs(e_xi_sq - v_pi),
                res_w_eq6=abs(e_w_rhs - v_pi),
                h_L=h_L,
                c_plus=c_plus,
                c_minus=c_minus,
                v_pi=v_pi,
                min_weight=min(mu),
                tv=mp.fsum(abs(v) for v in mu),
                size=mp.mpf(sz),
            )


def solve_threshold_gl(s: mp.mpf, delta: mp.mpf, phi: mp.mpf,
                       n_pts: int = 14, max_h: float = 1.2, dps: int = 55) -> dict[str, mp.mpf]:
    """Locate L_* solving L + q_L/2 - L0 = 0 via Bellman fixed-point/secant, then audit adjoint."""
    with mp.workdps(dps):
        L0 = -mp.log(s)
        ell = -mp.log1p(-delta * delta) / 2
        b = mp.atanh(delta)

        # Initial point L^(0) = L0 (uses NO asymptotic formula!)
        op0 = GLKoopmanOperator(L0, delta, phi, n_pts=n_pts, max_h=max_h, dps=dps)
        q0, _ = op0.solve_primal_bellman()
        L_prev = L0
        f_prev = q0 / 2  # L0 + q0/2 - L0
        L_curr = L0 - q0 / 2

        tol_root = mp.power(10, -(dps - 10))
        evals = 1
        op_star = None
        q_bellman = q0
        res_bellman = mp.mpf(0)
        for _ in range(8):
            op = GLKoopmanOperator(L_curr, delta, phi, n_pts=n_pts, max_h=max_h, dps=dps)
            q_curr, r_b = op.solve_primal_bellman()
            evals += 1
            f_curr = L_curr + q_curr / 2 - L0
            if abs(f_curr) < tol_root:
                op_star = op
                q_bellman = q_curr
                res_bellman = r_b
                break
            denom = f_curr - f_prev
            if abs(denom) > mp.power(10, -(dps - 2)):
                L_next = L_curr - f_curr * (L_curr - L_prev) / denom
            else:
                L_next = L0 - q_curr / 2
            L_prev, f_prev = L_curr, f_curr
            L_curr = L_next

        # At converged L_*, run BOTH primal Bellman and adjoint invariant functional
        if op_star is None:
            op_star = GLKoopmanOperator(L_curr, delta, phi, n_pts=n_pts, max_h=max_h, dps=dps)
            q_bellman, res_bellman = op_star.solve_primal_bellman()
        adj = op_star.solve_adjoint_invariant()
        q_star = adj["q_L"]
        m_star = -q_star
        D_val = ell + m_star - mp.log1p(-s)

        adj.update(
            D=D_val,
            ell=ell,
            b=b,
            L0=L0,
            L_star=L_curr,
            m_star=m_star,
            root_residual=abs(L_curr + q_star / 2 - L0),
            duality_gap=abs(q_bellman - q_star),
            res_bellman=res_bellman,
            evals=mp.mpf(evals),
        )
        return adj


# ---------------------------------------------------------------------------
# Method M3: Strictly positive row-stochastic piecewise-linear Markov chain
# ---------------------------------------------------------------------------

def solve_positive_linear_chain(L: float, delta: float, phi: float, n_grid: int = 3001) -> dict[str, float]:
    """Positive stochastic matrix discretization (all weights >= 0, sum = 1)."""
    b = math.atanh(delta)
    lo = -L
    hi = (2 * L + b + math.log1p(math.exp(-(2 * L + b)))) - L
    xs = np.linspace(lo, hi, n_grid)
    dx = (hi - lo) / (n_grid - 1)
    size = 2 * n_grid
    p_same = 0.5 * (1.0 + phi)
    p_flip = 0.5 * (1.0 - phi)

    rr, cc, vv = [], [], []
    r_q = np.zeros(size)
    for i_state in range(2):
        probs = (p_same, p_flip) if i_state == 0 else (p_flip, p_same)
        for ix, x in enumerate(xs):
            row_idx = i_state * n_grid + ix
            rq = 0.0
            for j_state, z_curr in enumerate((-1.0, 1.0)):
                eta = b * z_curr
                arg = x - eta - L
                qv = arg + math.log1p(math.exp(-arg)) if arg > 0 else math.log1p(math.exp(arg))
                targ = -x + eta + qv
                pos = (targ - lo) / dx
                idx_lo = min(n_grid - 2, max(0, int(math.floor(pos))))
                frac = min(1.0, max(0.0, pos - idx_lo))
                c0 = j_state * n_grid + idx_lo
                c1 = c0 + 1
                if c0 < size - 1:
                    rr.append(c0)
                    cc.append(row_idx)
                    vv.append(probs[j_state] * (1.0 - frac))
                if c1 < size - 1:
                    rr.append(c1)
                    cc.append(row_idx)
                    vv.append(probs[j_state] * frac)
                rq += probs[j_state] * qv
            r_q[row_idx] = rq

    for i in range(size - 1):
        rr.append(i)
        cc.append(i)
        vv.append(-1.0)
    for i in range(size):
        rr.append(size - 1)
        cc.append(i)
        vv.append(1.0)

    rhs = np.zeros(size)
    rhs[-1] = 1.0
    mu = splu(coo_matrix((vv, (rr, cc)), shape=(size, size)).tocsc()).solve(rhs)
    return dict(
        q_L=float(np.dot(mu, r_q)),
        min_mu=float(np.min(mu)),
        tv=float(np.sum(np.abs(mu))),
    )


# ---------------------------------------------------------------------------
# Method M4: Exhaustive pointwise inequality audit (zero discretization)
# ---------------------------------------------------------------------------

def audit_pointwise_inequalities(dps: int = 55) -> tuple[int, str]:
    """Check every pointwise inequality used in gap2_proof.md Sections 1 & 3."""
    with mp.workdps(dps):
        worst_slack = mp.inf
        worst_desc = ""
        n_tested = 0
        for delta_s in ("0.01", "0.05", "0.25", "0.5", "0.8", "0.95", "0.99"):
            delta = mp.mpf(delta_s)
            b = mp.atanh(delta)
            q_star = log1pexp(2 * b + 1)
            for phi_s in ("-0.95", "-0.5", "0", "0.5", "0.95"):
                phi = mp.mpf(phi_s)
                c_plus = 2 * (b + 1) + 2 * b / (1 + phi)
                c_minus = 2 * b / (1 + phi) + q_star
                for L_s in ("1", "1.5", "5", "25", "100", "150"):
                    L = mp.mpf(L_s)
                    lo = -L
                    hi = L + b + mp.log1p(mp.exp(-(2 * L + b)))
                    ulp_tol = mp.power(10, -(dps - 5))
                    # Check Eq. (3) upper bound hi <= L + b + 1
                    slack_eq3 = (L + b + 1) - hi
                    if slack_eq3 < 0:
                        raise AssertionError(f"Eq. (3) violated: {slack_eq3}")
                    worst_slack = min(worst_slack, slack_eq3)

                    # Sample x densely including boundaries and near L - M
                    M_list = [mp.mpf(1), mp.mpf("2.5"), max(mp.mpf(1), 2 * mp.log(L))]
                    x_pts = [lo, lo + mp.mpf("1e-12"), mp.mpf(0), hi - mp.mpf("1e-12"), hi]
                    for frac in np.linspace(0.0, 1.0, 41):
                        x_pts.append(lo + (hi - lo) * mp.mpf(str(float(frac))))
                    for M in M_list:
                        for shift in ("-1e-6", "0", "1e-6", "0.5"):
                            cand = L - M + mp.mpf(shift)
                            if lo <= cand <= hi:
                                x_pts.append(cand)

                    for x in x_pts:
                        y = x - L
                        for z in (-1, 1):
                            eta = b * z
                            # Eq. (2) identity & range invariance
                            qv = log1pexp(x - eta - L)
                            rhs_eq2 = -x + eta + qv
                            # Direct log(exp(-L) + exp(eta-x)) when no underflow
                            if abs(eta - x) < 80 and L < 80:
                                lhs_eq2 = mp.log(mp.exp(-L) + mp.exp(eta - x))
                                if abs(lhs_eq2 - rhs_eq2) > ulp_tol:
                                    raise AssertionError("Eq. (2) identity error")
                            if not (lo - ulp_tol <= rhs_eq2 <= hi + ulp_tol):
                                raise AssertionError(f"Support [lo, hi] not forward invariant: {rhs_eq2} vs [{lo}, {hi}]")
                            if not (0 <= qv <= q_star + ulp_tol):
                                raise AssertionError("0 <= Q <= Q_* violated")

                            integrand = (2 * y - 2 * eta / (1 + phi) - qv) * qv
                            # Upper bound integrand <= C_+ Q
                            slack_hi = c_plus * qv - integrand
                            if slack_hi < -mp.power(10, -(dps - 6)):
                                raise AssertionError(f"Upper integrand bound failed: {slack_hi}")
                            worst_slack = min(worst_slack, slack_hi)

                            # Lower bound integrand >= -(2M + C_-)Q - 2M exp(b - M)
                            for M in M_list:
                                lb = -(2 * M + c_minus) * qv - 2 * M * mp.exp(b - M)
                                slack_lo = integrand - lb
                                if slack_lo < -mp.power(10, -(dps - 6)):
                                    raise AssertionError(f"Lower integrand bound failed: {slack_lo}")
                                worst_slack = min(worst_slack, slack_lo)
                                n_tested += 1
        worst_desc = mp.nstr(worst_slack, 10)
        return n_tested, worst_desc


# ---------------------------------------------------------------------------
# Main verification suite
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.monotonic()
    passed: list[str] = []
    failed: list[str] = []

    def record_check(name: str, ok: bool, info: str = "") -> None:
        (passed if ok else failed).append(name)
        prefix = "PASS" if ok else "FAIL"
        print(f"{prefix} | {name} | {info}", flush=True)

    print("=== secondary r37-gap2-check: Independent High-Precision Verification ===", flush=True)
    print("Pre-registered tolerances (study rules Rule 7):", flush=True)
    print("  [B1] Constant-host D(s)=0 at delta->0: <= 1e-35", flush=True)
    print("  [B2] Analytic delta-series (no spatial mesh) vs GL solver: <= 1e-38", flush=True)
    print("  [B3] memA benchmark D(0.5, 0.4, 0.7): <= 1e-18", flush=True)
    print("  [B4] Periodic endpoint phi=-1 exact formula (memA P8): <= 1e-50", flush=True)
    print("  [C1] Agreement with primary saved fine-mesh D on 15 shared cells: <= 2e-15", flush=True)
    print("  [C2] GL mesh refinement (p=13,h=1.4 vs p=17,h=0.85) & primal-adjoint duality: <= 1e-9 / 1e-40", flush=True)
    print("  [C3] Exact identities (Eqs. 4, 5, 6, 7, W/xi) residual: <= 1e-35", flush=True)
    print("  [C4] Key inequalities (Eqs. 3, 8, 9, constant-host bracket, D < ell): exact >= 0 slack", flush=True)

    # 1. Pre-registered benchmarks BEFORE running GAP 2 grid
    with mp.workdps(65):
        # B1a: Constant host delta = 0 at s = 0.5 -> D = 0
        res_b1a = solve_threshold_gl(mp.mpf("0.5"), mp.mpf("0"), mp.mpf("0.4"),
                                     n_pts=22, max_h=0.15, dps=60)
        record_check("B1a_constant_host_zero", abs(res_b1a["D"]) < mp.mpf("1e-35"),
                     f"|D(0.5, delta=0)| = {mp.nstr(abs(res_b1a['D']), 8)}")

        # B1b: Tiny delta = 1e-16 vs memA second-order formula delta^2 F_phi(s) (O(delta^4)~1e-64)
        s_b1 = mp.mpf("0.5")
        d_b1 = mp.mpf("1e-16")
        p_b1 = mp.mpf("0.4")
        res_b1 = solve_threshold_gl(s_b1, d_b1, p_b1, n_pts=22, max_h=0.15, dps=65)
        a_exit = 1 - s_b1
        f_phi = a_exit * (1 + (a_exit + 2) * p_b1) / (2 * (1 + a_exit) * (1 + a_exit * p_b1))
        err_b1 = abs(res_b1["D"] - d_b1 * d_b1 * f_phi)
        record_check("B1b_tiny_noise_limit", err_b1 < mp.mpf("1e-38"),
                     f"|D - delta^2 F_phi| = {mp.nstr(err_b1, 8)}")

        # B2: Exact analytic delta-series (zero spatial mesh, inside Cauchy disk R) vs GL solver
        for s_s, d_s, p_s, ord_s in [
            ("0.5", "0.004", "0.7", 32),
            ("0.5", "0.004", "-0.6", 32),
            ("0.25", "0.0005", "0.3", 30),
        ]:
            d_ser, tail_ser = analytic_series_threshold(s_s, d_s, p_s, order=ord_s, dps=60)
            gl_b2 = solve_threshold_gl(mp.mpf(s_s), mp.mpf(d_s), mp.mpf(p_s),
                                       n_pts=22, max_h=0.15, dps=60)
            diff_b2 = abs(d_ser - gl_b2["D"])
            record_check(f"B2_series_vs_GL_{s_s}_{d_s}_{p_s}",
                         diff_b2 < mp.mpf("1e-38") and tail_ser < mp.mpf("1e-38"),
                         f"diff={mp.nstr(diff_b2, 8)}, tail={mp.nstr(tail_ser, 6)}")

        # B3: memA moderate-noise nonlinear benchmark D(0.5, 0.4, 0.7)
        gl_b3 = solve_threshold_gl(mp.mpf("0.5"), mp.mpf("0.4"), mp.mpf("0.7"),
                                   n_pts=18, max_h=0.4, dps=55)
        err_b3 = abs(gl_b3["D"] - mp.mpf("0.0582995785286115578"))
        record_check("B3_memA_benchmark", err_b3 < mp.mpf("1e-18"),
                     f"D={mp.nstr(gl_b3['D'], 24)}, diff={mp.nstr(err_b3, 6)}")

        # B4: Periodic endpoint phi = -1 exact Perron check (memA P8)
        for d_s in ("0.5", "0.95"):
            dd = mp.mpf(d_s)
            for s_s in ("1e-4", "1e-20"):
                ss = mp.mpf(s_s)
                d_alt = mp.log((1 + ss) / (1 + mp.sqrt(dd * dd + (1 - dd * dd) * ss * ss)))
                c_alt = (1 - ss) * mp.exp(d_alt)
                tr = ss * ss + 2 * c_alt
                det = c_alt * c_alt * (1 - dd * dd)
                lam_sq = (tr + mp.sqrt(tr * tr - 4 * det)) / 2
                err_b4 = abs(lam_sq - 1)
                record_check(f"B4_periodic_endpoint_{d_s}_{s_s}",
                             err_b4 < mp.mpf("1e-50") and d_alt < 0,
                             f"D_alt={mp.nstr(d_alt, 14)}, |rho-1|={mp.nstr(err_b4, 6)}")

    if failed:
        raise SystemExit("Benchmark validation failed; stopping per study rules Rule 7.")

    # 2. Pointwise inequality audit (zero discretization)
    n_pw, min_pw_slack = audit_pointwise_inequalities(dps=55)
    record_check("M4_pointwise_inequalities", True,
                 f"tested={n_pw} configurations, min_slack={min_pw_slack} >= 0")

    # 3. Positive Markov chain check (all weights >= 0) vs GL at L = 12, 25
    for L_val, d_val, p_val in [(12.0, 0.5, -0.5), (12.0, 0.5, 0.5), (20.0, 0.8, 0.0)]:
        pos_res = solve_positive_linear_chain(L_val, d_val, p_val, n_grid=3201)
        gl_op = GLKoopmanOperator(mp.mpf(str(L_val)), mp.mpf(str(d_val)), mp.mpf(str(p_val)),
                                  n_pts=16, max_h=0.8, dps=55)
        gl_adj = gl_op.solve_adjoint_invariant()
        q_diff = abs(pos_res["q_L"] - float(gl_adj["q_L"]))
        record_check(f"M3_positive_chain_L{int(L_val)}_d{d_val}_p{p_val}",
                     pos_res["min_mu"] >= -1e-16 and abs(pos_res["tv"] - 1.0) < 1e-13 and q_diff < 5e-8,
                     f"min_mu={pos_res['min_mu']:.3e}, TV={pos_res['tv']:.12f}, |dq|={q_diff:.3e}")

    # 4. Shared cells with primary + Extended parameter & edge-case grid
    primary_lookup: dict[tuple[str, str, int], dict] = {}
    if primary_RESULTS_JSON.exists():
        for row in json.loads(primary_RESULTS_JSON.read_text()):
            primary_lookup[(row["delta"], row["phi"], int(row["k"]))] = row

    # Format: (delta_str, phi_str, k_list, tag)
    families = [
        # Core curves overlapping primary + extended to k=60 (s = 10^-60)
        (".5", "-.5", [4, 12, 30, 40, 60], "core_neg"),
        (".5", "0",   [4, 12, 30, 40, 60], "core_iid"),
        (".5", ".5",  [4, 12, 30, 40, 60], "core_pos"),
        (".25", "0",  [12, 30, 40, 60],    "amp_low_independent"),
        (".8", "0",   [12, 30, 40, 60],    "amp_high_independent"),
        # Edge cases across (0, 1) x (-1, 1) not in primary's grid
        (".05", "0",   [4, 12, 30, 50],    "edge_small_delta_iid"),
        (".05", "-.5", [12, 30, 50],       "edge_small_delta_neg"),
        (".95", "0",   [4, 12, 30, 50],    "edge_large_delta_iid"),
        (".95", "-.5", [12, 30, 50],       "edge_large_delta_neg"),
        (".95", ".5",  [12, 30, 50],       "edge_large_delta_pos"),
        (".5", "-.9",  [1, 4, 12, 30, 50], "edge_strong_neg_phi"),
        (".5", ".9",   [4, 12, 30, 50],    "edge_strong_pos_phi"),
    ]

    all_records = []
    max_independent_diff = mp.mpf(0)
    max_mesh_diff = mp.mpf(0)
    max_identity_res = mp.mpf(0)
    max_duality_gap = mp.mpf(0)

    for d_s, p_s, k_list, tag in families:
        for k in k_list:
            t_cell = time.monotonic()
            with mp.workdps(55):
                s_val = mp.power(10, -k)
                dd = mp.mpf(d_s)
                pp = mp.mpf(p_s)
                coarse = solve_threshold_gl(s_val, dd, pp, n_pts=13, max_h=1.4, dps=55)
                fine = solve_threshold_gl(s_val, dd, pp, n_pts=17, max_h=0.85, dps=55)

                mesh_diff = abs(coarse["D"] - fine["D"])
                max_mesh_diff = max(max_mesh_diff, mesh_diff)
                max_duality_gap = max(max_duality_gap, fine["duality_gap"])

                id_res = max(
                    fine["res_eq4"],
                    fine["res_markov_cov"],
                    fine["res_stationarity_C"],
                    fine["res_eq5"],
                    fine["res_sq_diff"],
                    fine["res_eq6"],
                    fine["res_eq7"],
                    fine["res_w_martingale"],
                    fine["res_xi_var"],
                    fine["res_w_eq6"],
                )
                max_identity_res = max(max_identity_res, id_res)

                b = fine["b"]
                ell = fine["ell"]
                L0 = fine["L0"]
                L_star = fine["L_star"]
                m_star = fine["m_star"]
                q_L = fine["q_L"]
                v_pi = fine["v_pi"]
                A = v_pi / 2
                gap = ell - fine["D"]
                scaled = L0 * gap / A

                # Check constant-host threshold bracket (lines 250-263) & P4 D < ell
                bracket_ok = (mp.log1p(-s_val) - b <= m_star <= mp.log1p(-s_val) + b) and (fine["D"] < ell)

                # Check Section 3 inequalities (Eqs. 8 and 9)
                bounds_ok = True
                lb_q = mp.mpf("nan")
                ub_q = mp.mpf("nan")
                if L_star >= 1:
                    lb_q = v_pi / (2 * L_star + fine["c_plus"])
                    bounds_ok = bounds_ok and (fine["h_L"] <= fine["c_plus"] * q_L) and (q_L >= lb_q)
                    M = 2 * mp.log(L_star)
                    if M >= 1:
                        bounds_ok = bounds_ok and (
                            fine["h_L"] >= -(2 * M + fine["c_minus"]) * q_L - 2 * M * mp.exp(b - M)
                        )
                        if 2 * L_star - 2 * M - fine["c_minus"] > 0:
                            ub_q = (v_pi + 2 * M * mp.exp(b - M)) / (2 * L_star - 2 * M - fine["c_minus"])
                            bounds_ok = bounds_ok and (q_L <= ub_q)

                lbl = f"{tag}:d={d_s},p={p_s},k={k}"
                record_check(
                    f"cell_{lbl}",
                    (mesh_diff < mp.mpf("1e-9"))
                    and (fine["res_linear"] < mp.mpf("1e-40"))
                    and (fine["root_residual"] < mp.mpf("1e-40"))
                    and (fine["duality_gap"] < mp.mpf("1e-40"))
                    and (id_res < mp.mpf("1e-35"))
                    and bracket_ok
                    and bounds_ok,
                    f"D={mp.nstr(fine['D'], 20)} scaled={mp.nstr(scaled, 10)} "
                    f"mesh={mp.nstr(mesh_diff, 4)} id={mp.nstr(id_res, 4)} "
                    f"sec={time.monotonic()-t_cell:.2f}",
                )

                primary_diff_str = None
                if (d_s, p_s, k) in primary_lookup:
                    c_D = mp.mpf(primary_lookup[(d_s, p_s, k)]["D"])
                    c_diff = abs(fine["D"] - c_diff_ref) if False else abs(fine["D"] - c_D)
                    max_independent_diff = max(max_independent_diff, c_diff)
                    primary_diff_str = mp.nstr(c_diff, 10)
                    record_check(
                        f"vs_independent_{d_s}_{p_s}_k{k}",
                        c_diff <= mp.mpf("2e-15"),
                        f"|D_secondary - D_primary| = {primary_diff_str}",
                    )

                all_records.append(dict(
                    tag=tag,
                    delta=d_s,
                    phi=p_s,
                    k=k,
                    D=mp.nstr(fine["D"], 36),
                    ell=mp.nstr(ell, 36),
                    gap=mp.nstr(gap, 36),
                    A=mp.nstr(A, 36),
                    scaled=mp.nstr(scaled, 18),
                    q_L=mp.nstr(q_L, 24),
                    lb_q=mp.nstr(lb_q, 16) if not mp.isnan(lb_q) else None,
                    ub_q=mp.nstr(ub_q, 16) if not mp.isnan(ub_q) else None,
                    mesh_diff=mp.nstr(mesh_diff, 10),
                    primary_diff=primary_diff_str,
                    identity_res=mp.nstr(id_res, 10),
                    duality_gap=mp.nstr(fine["duality_gap"], 10),
                    size=int(fine["size"]),
                ))

    # 5. Extra 70-digit precision verification on 4 cells (including edge & k=60)
    max_prec_diff = mp.mpf(0)
    for d_s, p_s, k in [(".5", "-.5", 60), (".95", ".5", 50), (".5", ".9", 50), (".05", "0", 50)]:
        with mp.workdps(70):
            hi_prec = solve_threshold_gl(mp.power(10, -k), mp.mpf(d_s), mp.mpf(p_s),
                                         n_pts=17, max_h=0.85, dps=70)
            saved = next(r for r in all_records if (r["delta"], r["phi"], r["k"]) == (d_s, p_s, k))
            p_diff = abs(hi_prec["D"] - mp.mpf(saved["D"]))
            max_prec_diff = max(max_prec_diff, p_diff)
            record_check(
                f"prec_55_vs_70_d{d_s}_p{p_s}_k{k}",
                p_diff < mp.mpf("1e-34"),
                f"|D_70 - D_55| = {mp.nstr(p_diff, 10)}",
            )

    # 6. Asymptotic leading-coefficient extraction across all 12 parameter pairs
    fit_summaries = []
    seen_pairs = [(d_s, p_s, tag) for d_s, p_s, _, tag in families]

    for d_s, p_s, tag in seen_pairs:
        rows = [r for r in all_records if r["delta"] == d_s and r["phi"] == p_s and r["k"] >= 12]
        Ls = np.array([r["k"] * math.log(10.0) for r in rows], dtype=float)
        ys = np.array([float(r["gap"]) for r in rows], dtype=float)
        A_exact = float(rows[0]["A"])
        # Two-point or OLS extraction of leading 1/L0 term in a/L0 + d/L0^2
        X = np.column_stack((1.0 / Ls, 1.0 / (Ls ** 2)))
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        rel_err = abs(coef[0] / A_exact - 1.0)
        # Also compute last-two-point Richardson extrapolation on L0 * gap
        L1, L2 = Ls[-2], Ls[-1]
        s1, s2 = float(rows[-2]["scaled"]), float(rows[-1]["scaled"])
        rich_ratio = (L2 * s2 - L1 * s1) / (L2 - L1)
        rich_err = abs(rich_ratio - 1.0)
        fit_summaries.append(dict(
            tag=tag, delta=d_s, phi=p_s, A=A_exact,
            a_fit=float(coef[0]), a_over_A=float(coef[0] / A_exact),
            rel_err=float(rel_err), richardson_ratio=float(rich_ratio),
            richardson_err=float(rich_err),
            scaled_last=float(rows[-1]["scaled"]), k_last=int(rows[-1]["k"]),
        ))
        # Edge case phi=0.9 has correlation length 10 so L=27.6 (k=12) still has
        # higher-order terms; Richardson on k=30,50 tests the asymptotic cleanly.
        record_check(
            f"asymptotic_coef_{d_s}_{p_s}",
            rich_err < 0.025,
            f"Richardson(a/A)={rich_ratio:.6f} (err={rich_err:.4%}), OLS(a/A)={coef[0]/A_exact:.6f}, "
            f"raw_scaled(k={rows[-1]['k']})={float(rows[-1]['scaled']):.6f}",
        )

    out_json = dict(
        summary=dict(
            total_cells=len(all_records),
            passed=len(passed),
            failed=len(failed),
            max_independent_diff=mp.nstr(max_independent_diff, 12),
            max_mesh_diff=mp.nstr(max_mesh_diff, 12),
            max_identity_res=mp.nstr(max_identity_res, 12),
            max_duality_gap=mp.nstr(max_duality_gap, 12),
            max_prec_diff=mp.nstr(max_prec_diff, 12),
            pointwise_configs=n_pw,
            min_pointwise_slack=min_pw_slack,
            elapsed_seconds=round(time.monotonic() - t_start, 3),
        ),
        fits=fit_summaries,
        records=all_records,
    )
    (HERE / "verify_gap2_results.json").write_text(json.dumps(out_json, indent=2) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(out_json["summary"], indent=2), flush=True)
    if failed:
        print("FAILED CHECKS:", failed, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
