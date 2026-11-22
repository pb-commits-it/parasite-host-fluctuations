#!/usr/bin/env python3
"""Independent referee verification of GAP 1 partial proof (r37-gap1-check).

Standalone verification code by secondary. Imports NO code from
`gap1/` or elsewhere in the repository.

Methods implemented independently:
1. Exact rational polynomial ring over Q[delta, phi] (`fractions.Fraction`,
   no SymPy) verifying recurrence (4)-(5), all polynomial signs and L1 bounds
   through degree N=12, exact 256-word rational path-tree enumeration, and the
   rational margin 185403593/1312200000 at q = 1/30.
2. 80-digit `mpmath` analytic q-series with rigorous Cauchy remainder bounds on
   |q| <= 1/10, coupled with Newton root-solving for q_*(s) at exact target s.
3. 80-digit `mpmath` Gauss-Legendre modal Koopman + adjoint-Poisson solver
   (orthogonal Legendre basis P_m and exact three-term derivative P'_m, with
   scale-free Poisson normalization in v_half; zero Chebyshev-Lobatto or
   barycentric code shared with primary), cross-checked against full dense
   `mp.lu_solve` and 4th-order s-secant finite differences.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import lu_factor, lu_solve

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
GAP1_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '../../gap1').resolve()

mp.mp.dps = 80
ZERO = mp.mpf(0)


# ============================================================================
# 1. Exact rational algebra over Q[delta, phi] (zero SymPy, zero float)
# ============================================================================

Poly = dict[tuple[int, int], Fraction]


def poly_const(c: int | Fraction) -> Poly:
    f = Fraction(c)
    return {(0, 0): f} if f != 0 else {}


def poly_add(a: Poly, b: Poly, scale_b: int | Fraction = 1) -> Poly:
    sb = Fraction(scale_b)
    out = dict(a)
    for k, v in b.items():
        nv = out.get(k, Fraction(0)) + sb * v
        if nv == 0:
            out.pop(k, None)
        else:
            out[k] = nv
    return out


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for (da, pa), va in a.items():
        for (db, pb), vb in b.items():
            key = (da + db, pa + pb)
            nv = out.get(key, Fraction(0)) + va * vb
            if nv == 0:
                out.pop(key, None)
            else:
                out[key] = nv
    return out


def poly_scale(a: Poly, c: int | Fraction) -> Poly:
    fc = Fraction(c)
    if fc == 0:
        return {}
    return {k: v * fc for k, v in a.items()}


def exact_rational_recurrence(n_max: int = 12) -> tuple[list[Poly], list[Poly], list[Poly]]:
    """Compute r_n, e_n, g_n in Q[delta, phi] from Eqs. (4)-(5)."""
    p_same: Poly = {(0, 0): Fraction(1, 2), (0, 1): Fraction(1, 2)}
    p_diff: Poly = {(0, 0): Fraction(1, 2), (0, 1): Fraction(-1, 2)}

    pow_minus: list[Poly] = [poly_const(1)]
    pow_plus: list[Poly] = [poly_const(1)]
    fac_minus: Poly = {(0, 0): Fraction(1), (1, 0): Fraction(-1)}
    fac_plus: Poly = {(0, 0): Fraction(1), (1, 0): Fraction(1)}
    for _ in range(1, n_max + 1):
        pow_minus.append(poly_mul(pow_minus[-1], fac_minus))
        pow_plus.append(poly_mul(pow_plus[-1], fac_plus))

    M: list[list[tuple[Poly, Poly]]] = [
        [({}, {}) for _ in range(n_max + 1)] for _ in range(n_max + 1)
    ]
    M[0][0] = (poly_const(1), poly_const(1))

    r_polys: list[Poly] = [poly_const(0)]
    for n in range(1, n_max + 1):
        rn: Poly = {}
        for k in range(1, n + 1):
            acc_minus: Poly = {}
            acc_plus: Poly = {}
            for m in range(n - k + 1):
                binom_signed = ((-1) ** m) * math.comb(k + m - 1, m)
                m_minus, m_plus = M[m][n - k]
                if not m_minus and not m_plus:
                    continue
                mix_minus = poly_add(poly_mul(p_same, m_minus), poly_mul(p_diff, m_plus))
                mix_plus = poly_add(poly_mul(p_diff, m_minus), poly_mul(p_same, m_plus))
                acc_minus = poly_add(acc_minus, mix_minus, binom_signed)
                acc_plus = poly_add(acc_plus, mix_plus, binom_signed)
            mk_minus = poly_mul(pow_minus[k], acc_minus)
            mk_plus = poly_mul(pow_plus[k], acc_plus)
            M[k][n] = (mk_minus, mk_plus)
            weight = Fraction((-1) ** (k + 1), 2 * k)
            rn = poly_add(rn, poly_add(mk_minus, mk_plus), weight)
        r_polys.append(rn)

    e_polys: list[Poly] = [poly_const(1)]
    g_polys: list[Poly] = [poly_const(0)]
    for n in range(1, n_max + 1):
        en: Poly = {}
        for k in range(1, n + 1):
            en = poly_add(en, poly_mul(r_polys[k], e_polys[n - k]), Fraction(-k, n))
        e_polys.append(en)

        gn = poly_add(poly_scale(en, -1), r_polys[n], -2 * n)
        for k in range(1, n + 1):
            gn = poly_add(gn, poly_mul(e_polys[n - k], r_polys[k]), k)
        g_polys.append(gn)

    return r_polys, e_polys, g_polys


def poly_eval_fraction(p: Poly, d: Fraction, phi: Fraction) -> Fraction:
    total = Fraction(0)
    for (ed, ep), c in p.items():
        total += c * (d ** ed) * (phi ** ep)
    return total


def exact_rational_word_series(d: Fraction, phi: Fraction, deg: int = 8) -> list[Fraction]:
    """Exact rational enumeration of all 2^deg Markov histories of length deg."""
    p_same = (1 + phi) / 2
    p_diff = (1 - phi) / 2
    r_acc = [Fraction(0)] * (deg + 1)
    for word in itertools.product((-1, 1), repeat=deg):
        prob = Fraction(1, 2)
        for idx in range(1, deg):
            prob *= p_same if word[idx] == word[idx - 1] else p_diff
        w = [Fraction(0)] * (deg + 1)
        for z in word:
            inv = [Fraction(0)] * (deg + 1)
            inv[0] = Fraction(1)
            for n in range(1, deg + 1):
                inv[n] = -sum(w[k] * inv[n - k] for k in range(1, n + 1))
            hz = 1 + d * z
            w_next = [Fraction(0)] * (deg + 1)
            for n in range(1, deg + 1):
                w_next[n] = hz * inv[n - 1]
            w = w_next
        inv = [Fraction(0)] * (deg + 1)
        inv[0] = Fraction(1)
        for n in range(1, deg + 1):
            inv[n] = -sum(w[k] * inv[n - k] for k in range(1, n + 1))
            coeff = sum(k * w[k] * inv[n - k] for k in range(1, n + 1)) / n
            r_acc[n] += prob * coeff
    return r_acc


# ============================================================================
# 2. High-precision (80-digit) q-series solver with certified Cauchy bound
# ============================================================================

def hp_q_series_coeffs(delta: mp.mpf, phi: mp.mpf, deg: int = 130) -> list[mp.mpf]:
    """Compute r_0..r_deg in 80-digit mpmath arithmetic."""
    d = mp.mpmathify(delta)
    p = mp.mpmathify(phi)
    p_same = (1 + p) / 2
    p_diff = (1 - p) / 2
    hm = 1 - d
    hp = 1 + d
    hm_pow = [mp.mpf(1)]
    hp_pow = [mp.mpf(1)]
    for _ in range(1, deg + 1):
        hm_pow.append(hm_pow[-1] * hm)
        hp_pow.append(hp_pow[-1] * hp)

    M_m = [[ZERO] * (deg + 1) for _ in range(deg + 1)]
    M_p = [[ZERO] * (deg + 1) for _ in range(deg + 1)]
    M_m[0][0] = mp.mpf(1)
    M_p[0][0] = mp.mpf(1)

    r = [ZERO] * (deg + 1)
    comb = math.comb
    for n in range(1, deg + 1):
        rn = ZERO
        for k in range(1, n + 1):
            nk = n - k
            sm = ZERO
            sp = ZERO
            m_start = 0 if nk == 0 else 1
            for m in range(m_start, nk + 1):
                um = M_m[m][nk]
                up = M_p[m][nk]
                c = (-1) ** m * comb(k + m - 1, m)
                sm += c * (p_same * um + p_diff * up)
                sp += c * (p_diff * um + p_same * up)
            vm = hm_pow[k] * sm
            vp = hp_pow[k] * sp
            M_m[k][n] = vm
            M_p[k][n] = vp
            if k & 1:
                rn += (vm + vp) / (2 * k)
            else:
                rn -= (vm + vp) / (2 * k)
        r[n] = rn
    return r


def eval_q_series(r: list[mp.mpf], q: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Return (R(q), V(q)=q R'(q)) via Horner."""
    R_val = ZERO
    V_val = ZERO
    for n in range(len(r) - 1, 0, -1):
        rn = r[n]
        R_val = (R_val + rn) * q
        V_val = (V_val + n * rn) * q
    return R_val, V_val


def solve_q_from_s_series(r: list[mp.mpf], s: mp.mpf) -> dict[str, mp.mpf]:
    """Solve R(q) + log(s) = 0 to 80 digits by Newton's method on log(q)."""
    target_R = -mp.log(s)
    q = min((1 - s) / (s * s), mp.mpf("0.075"))
    if q <= 0:
        q = target_R
    for _ in range(40):
        R_val, V_val = eval_q_series(r, q)
        res = R_val - target_R
        if abs(res) < mp.mpf("1e-76"):
            break
        q = q * mp.exp(-res / V_val)
    R_val, V_val = eval_q_series(r, q)
    one_minus_s = 1 - s
    G_val = one_minus_s - (2 - s) * V_val
    D_val = mp.log(q * s * s / one_minus_s)
    Ds_val = -G_val / (s * one_minus_s * V_val)
    return {
        "q": q,
        "R": R_val,
        "V": V_val,
        "G": G_val,
        "D": D_val,
        "Ds": Ds_val,
        "residual": abs(R_val + mp.log(s)),
    }


# ============================================================================
# 3. Independent 80-digit Gauss-Legendre modal Koopman + Poisson solver
# ============================================================================

_GL_CACHE: dict[int, tuple[list[mp.mpf], list[list[mp.mpf]], list[list[mp.mpf]], np.ndarray]] = {}
MAX_LIN_RESIDUAL = ZERO


def gl_modal_data(M: int) -> tuple[list[mp.mpf], list[list[mp.mpf]], list[list[mp.mpf]], np.ndarray]:
    """Gauss-Legendre nodes on [-1,1] and inverse modal matrix V^-1 (and its transpose)."""
    if M in _GL_CACHE:
        return _GL_CACHE[M]
    nodes: list[mp.mpf] = []
    weights: list[mp.mpf] = []
    for k in range(M):
        x = -mp.cos(mp.pi * (k + mp.mpf("0.75")) / (M + mp.mpf("0.5")))
        for _ in range(60):
            p0, p1 = mp.mpf(1), x
            for m in range(1, M):
                p0, p1 = p1, ((2 * m + 1) * x * p1 - m * p0) / (m + 1)
            dp = M * (x * p1 - p0) / (x * x - 1)
            dx = -p1 / dp
            x += dx
            if abs(dx) < mp.mpf("1e-78"):
                break
        p0, p1 = mp.mpf(1), x
        for m in range(1, M):
            p0, p1 = p1, ((2 * m + 1) * x * p1 - m * p0) / (m + 1)
        dp = M * (x * p1 - p0) / (x * x - 1)
        w = 2 / ((1 - x * x) * dp * dp)
        nodes.append(x)
        weights.append(w)

    V_inv_rows: list[list[mp.mpf]] = [[ZERO] * M for _ in range(M)]
    for k, (xk, wk) in enumerate(zip(nodes, weights)):
        p_prev, p_curr = mp.mpf(1), xk
        V_inv_rows[0][k] = mp.mpf("0.5") * wk
        if M > 1:
            V_inv_rows[1][k] = mp.mpf("1.5") * wk * xk
        for m in range(1, M - 1):
            p_next = ((2 * m + 1) * xk * p_curr - m * p_prev) / (m + 1)
            p_prev, p_curr = p_curr, p_next
            V_inv_rows[m + 1][k] = (mp.mpf(2 * (m + 1) + 1) / 2) * wk * p_curr

    V_inv_T = [list(col) for col in zip(*V_inv_rows)]
    V_inv_f64 = np.array([[float(v) for v in row] for row in V_inv_rows], dtype=np.float64)
    _GL_CACHE[M] = (nodes, V_inv_rows, V_inv_T, V_inv_f64)
    return nodes, V_inv_rows, V_inv_T, V_inv_f64


def legendre_eval_row(xi: mp.mpf, M: int) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Evaluate P_0..P_{M-1}(xi) and P'_0..P'_{M-1}(xi) via three-term recurrence."""
    P = [ZERO] * M
    dP = [ZERO] * M
    P[0] = mp.mpf(1)
    if M > 1:
        P[1] = xi
        dP[1] = mp.mpf(1)
    for m in range(1, M - 1):
        P[m + 1] = ((2 * m + 1) * xi * P[m] - m * P[m - 1]) / (m + 1)
        dP[m + 1] = (2 * m + 1) * P[m] + dP[m - 1]
    return P, dP


def invariant_v_interval(q: mp.mpf, delta: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Minimal forward-invariant interval [v_lo, v_hi] for v_t = log(1 + w_t)."""
    if delta == 0:
        lam = (1 + mp.sqrt(1 + 4 * q)) / 2
        v0 = mp.log(lam)
        pad = mp.mpf("1e-6")
        return max(ZERO, v0 - pad), v0 + pad
    b = 2 * q * delta - 1
    disc = mp.sqrt(b * b + 4 * q * (1 + delta))
    z_min = 2 * q * (1 + delta) / (disc + b)
    z_max = z_min + 2 * q * delta
    return mp.log(z_min), mp.log(z_max)


def matvec_mp(mat: list[list[mp.mpf]], vec: list[mp.mpf]) -> list[mp.mpf]:
    fdot = mp.fdot
    return [fdot(row, vec) for row in mat]


def gl_koopman_eval(
    q: mp.mpf,
    delta: mp.mpf,
    phi: mp.mpf,
    M: int = 36,
    force_dense_lu: bool = False,
) -> tuple[mp.mpf, mp.mpf]:
    """Evaluate R(q) and V(q) = q R'(q) via 80-digit Gauss-Legendre modal Koopman.

    Uses scale-free Poisson right-hand side g = f / v_half solving
    (I - K + 1 mu^T) g = (v - R) / v_half = O(1), so d/dv = (1/v_half) d/dxi
    cancels v_half identically without small-delta roundoff amplification.
    """
    global MAX_LIN_RESIDUAL
    q = mp.mpmathify(q)
    d = mp.mpmathify(delta)
    p = mp.mpmathify(phi)
    v_lo, v_hi = invariant_v_interval(q, d)
    v_mid = (v_lo + v_hi) / 2
    v_half = (v_hi - v_lo) / 2
    nodes, V_inv_rows, V_inv_T, V_inv_f64 = gl_modal_data(M)
    vs = [v_mid + v_half * x for x in nodes]

    p_same = (1 + p) / 2
    p_diff = (1 - p) / 2

    P_eval_j: list[list[list[mp.mpf]]] = []
    P_eval_T_j: list[list[list[mp.mpf]]] = []
    dP_xi_j: list[list[list[mp.mpf]]] = []
    B_f64_j: list[np.ndarray] = []

    for z in (-1, 1):
        hz = 1 + d * z
        P_rows: list[list[mp.mpf]] = []
        dP_rows: list[list[mp.mpf]] = []
        for vk in vs:
            u = q * hz * mp.exp(-vk)
            vt = mp.log1p(u)
            sens = u / (1 + u)
            xi = (vt - v_mid) / v_half
            P_r, dP_r = legendre_eval_row(xi, M)
            P_rows.append(P_r)
            dP_rows.append([val * sens for val in dP_r])
        P_eval_j.append(P_rows)
        P_eval_T_j.append([list(col) for col in zip(*P_rows)])
        dP_xi_j.append(dP_rows)
        P_f64 = np.array([[float(v) for v in row] for row in P_rows], dtype=np.float64)
        B_f64_j.append(P_f64 @ V_inv_f64)

    N2 = 2 * M
    vec_v = vs + vs
    if force_dense_lu or abs(1 - p) < mp.mpf("1e-10"):
        V_inv_mp = mp.matrix(V_inv_rows)
        K_mp = mp.matrix(N2, N2)
        Kd_xi_mp = mp.matrix(N2, N2)
        for j in range(2):
            B_mp = mp.matrix(P_eval_j[j]) * V_inv_mp
            Bd_mp = mp.matrix(dP_xi_j[j]) * V_inv_mp
            for i in range(2):
                prob = p_same if i == j else p_diff
                for k in range(M):
                    for l in range(M):
                        K_mp[i * M + k, j * M + l] = prob * B_mp[k, l]
                        Kd_xi_mp[i * M + k, j * M + l] = prob * Bd_mp[k, l]
        A_mp = K_mp.T - mp.eye(N2)
        rhs_mp = mp.matrix(N2, 1)
        for l in range(N2):
            A_mp[N2 - 1, l] = mp.mpf(1)
        rhs_mp[N2 - 1] = mp.mpf(1)
        mu_mp = mp.lu_solve(A_mp, rhs_mp)
        vec_v_mp = mp.matrix(vec_v)
        R_val = (mu_mp.T * vec_v_mp)[0]
        H_mp = mp.eye(N2) - K_mp
        for k in range(N2):
            for l in range(N2):
                H_mp[k, l] += mu_mp[l]
        rhs_scaled = mp.matrix([(v - R_val) / v_half for v in vec_v])
        g_mp = mp.lu_solve(H_mp, rhs_scaled)
        V_val = (mu_mp.T * (Kd_xi_mp * g_mp))[0]
        return R_val, V_val

    ps_f, pd_f = float(p_same), float(p_diff)
    K_f64 = np.empty((N2, N2), dtype=np.float64)
    for j in range(2):
        K_f64[0:M, j * M:(j + 1) * M] = (ps_f if j == 0 else pd_f) * B_f64_j[j]
        K_f64[M:N2, j * M:(j + 1) * M] = (pd_f if j == 0 else ps_f) * B_f64_j[j]

    def apply_K(vec: list[mp.mpf]) -> list[mp.mpf]:
        c0 = matvec_mp(V_inv_rows, vec[0:M])
        c1 = matvec_mp(V_inv_rows, vec[M:N2])
        b0 = matvec_mp(P_eval_j[0], c0)
        b1 = matvec_mp(P_eval_j[1], c1)
        out0 = [p_same * b0[k] + p_diff * b1[k] for k in range(M)]
        out1 = [p_diff * b0[k] + p_same * b1[k] for k in range(M)]
        return out0 + out1

    def apply_K_dot_xi(vec: list[mp.mpf]) -> list[mp.mpf]:
        c0 = matvec_mp(V_inv_rows, vec[0:M])
        c1 = matvec_mp(V_inv_rows, vec[M:N2])
        b0 = matvec_mp(dP_xi_j[0], c0)
        b1 = matvec_mp(dP_xi_j[1], c1)
        out0 = [p_same * b0[k] + p_diff * b1[k] for k in range(M)]
        out1 = [p_diff * b0[k] + p_same * b1[k] for k in range(M)]
        return out0 + out1

    def apply_KT(mu_vec: list[mp.mpf]) -> list[mp.mpf]:
        mu0, mu1 = mu_vec[0:M], mu_vec[M:N2]
        mix0 = [p_same * mu0[k] + p_diff * mu1[k] for k in range(M)]
        mix1 = [p_diff * mu0[k] + p_same * mu1[k] for k in range(M)]
        t0 = matvec_mp(P_eval_T_j[0], mix0)
        t1 = matvec_mp(P_eval_T_j[1], mix1)
        return matvec_mp(V_inv_T, t0) + matvec_mp(V_inv_T, t1)

    A_f64 = K_f64.T - np.eye(N2)
    A_f64[-1, :] = 1.0
    fac_A = lu_factor(A_f64, check_finite=False)
    mu = [ZERO] * N2
    for _ in range(16):
        kt_mu = apply_KT(mu)
        res = [-kt_mu[i] + mu[i] for i in range(N2)]
        res[-1] = mp.mpf(1) - mp.fsum(mu)
        norm_r = max(abs(v) for v in res)
        if norm_r < mp.mpf("1e-75"):
            MAX_LIN_RESIDUAL = max(MAX_LIN_RESIDUAL, norm_r)
            break
        corr = lu_solve(fac_A, np.array([float(v) for v in res], dtype=np.float64), check_finite=False)
        for i in range(N2):
            mu[i] += mp.mpf(float(corr[i]))
    else:
        return gl_koopman_eval(q, d, p, M, force_dense_lu=True)

    R_val = mp.fdot(mu, vec_v)

    mu_f64 = np.array([float(v) for v in mu], dtype=np.float64)
    H_f64 = np.eye(N2) - K_f64 + np.ones((N2, 1)) * mu_f64[None, :]
    fac_H = lu_factor(H_f64, check_finite=False)
    rhs_H = [(v - R_val) / v_half for v in vec_v]
    g_vec = [ZERO] * N2
    for _ in range(16):
        kg = apply_K(g_vec)
        mu_dot_g = mp.fdot(mu, g_vec)
        res = [rhs_H[i] - (g_vec[i] - kg[i] + mu_dot_g) for i in range(N2)]
        norm_r = max(abs(v) for v in res)
        if norm_r < mp.mpf("1e-75"):
            MAX_LIN_RESIDUAL = max(MAX_LIN_RESIDUAL, norm_r)
            break
        corr = lu_solve(fac_H, np.array([float(v) for v in res], dtype=np.float64), check_finite=False)
        for i in range(N2):
            g_vec[i] += mp.mpf(float(corr[i]))
    else:
        return gl_koopman_eval(q, d, p, M, force_dense_lu=True)

    kd_g = apply_K_dot_xi(g_vec)
    V_val = mp.fdot(mu, kd_g)
    return R_val, V_val


def gl_koopman_eval_f64(q: float, d: float, p: float, M: int = 24) -> tuple[float, float]:
    """Fast binary64 warm-start evaluator used ONLY to seed 80-digit Newton."""
    if d == 0.0:
        lam = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * q))
        return math.log(lam), (lam - 1.0) / (2.0 * lam - 1.0)
    b = 2.0 * q * d - 1.0
    disc = math.sqrt(b * b + 4.0 * q * (1.0 + d))
    z_min = 2.0 * q * (1.0 + d) / (disc + b)
    z_max = z_min + 2.0 * q * d
    v_lo, v_hi = math.log(z_min), math.log(z_max)
    v_mid, v_half = 0.5 * (v_lo + v_hi), 0.5 * (v_hi - v_lo)
    nodes, _, _, V_inv_f64 = gl_modal_data(M)
    xs = np.array([float(x) for x in nodes], dtype=np.float64)
    vs = v_mid + v_half * xs

    ps, pd = 0.5 * (1.0 + p), 0.5 * (1.0 - p)
    N2 = 2 * M
    K = np.empty((N2, N2), dtype=np.float64)
    Kd = np.empty((N2, N2), dtype=np.float64)
    for j, z in enumerate((-1.0, 1.0)):
        u = q * (1.0 + d * z) * np.exp(-vs)
        vt = np.log1p(u)
        sens = u / (1.0 + u)
        xi = (vt - v_mid) / v_half
        P = np.zeros((M, M), dtype=np.float64)
        dP = np.zeros((M, M), dtype=np.float64)
        P[:, 0] = 1.0
        if M > 1:
            P[:, 1] = xi
            dP[:, 1] = 1.0
        for m in range(1, M - 1):
            P[:, m + 1] = ((2 * m + 1) * xi * P[:, m] - m * P[:, m - 1]) / (m + 1)
            dP[:, m + 1] = (2 * m + 1) * P[:, m] + dP[:, m - 1]
        B = P @ V_inv_f64
        Bd = (sens[:, None] * dP) @ V_inv_f64
        K[0:M, j * M:(j + 1) * M] = (ps if j == 0 else pd) * B
        K[M:N2, j * M:(j + 1) * M] = (pd if j == 0 else ps) * B
        Kd[0:M, j * M:(j + 1) * M] = (ps if j == 0 else pd) * Bd
        Kd[M:N2, j * M:(j + 1) * M] = (pd if j == 0 else ps) * Bd

    A = K.T - np.eye(N2)
    A[-1, :] = 1.0
    rhs = np.zeros(N2)
    rhs[-1] = 1.0
    mu = lu_solve(lu_factor(A, check_finite=False), rhs, check_finite=False)
    vec_v = np.tile(vs, 2)
    R_val = float(mu @ vec_v)
    H = np.eye(N2) - K + np.ones((N2, 1)) * mu[None, :]
    g = lu_solve(lu_factor(H, check_finite=False), (vec_v - R_val) / v_half, check_finite=False)
    V_val = float(mu @ (Kd @ g))
    return R_val, V_val


def solve_s_gl_koopman(
    s: mp.mpf,
    delta: mp.mpf,
    phi: mp.mpf,
    M: int = 36,
    q_init: mp.mpf | None = None,
) -> dict[str, mp.mpf]:
    """Root-solve R(q) + log(s) = 0 directly at target s to 80 digits."""
    s = mp.mpmathify(s)
    d = mp.mpmathify(delta)
    p = mp.mpmathify(phi)
    target_R = -mp.log(s)

    if q_init is None:
        qf = float((1 - s) / (s * s))
        tf = float(target_R)
        df, pf = float(d), float(p)
        if abs(1.0 - pf) > 1e-8 and 1e-12 < df < 1.0:
            mf = 24 if qf < 10.0 else 48
            for _ in range(12):
                Rf, Vf = gl_koopman_eval_f64(qf, df, pf, M=mf)
                if abs(Rf - tf) < 1e-13:
                    break
                qf *= math.exp(-(Rf - tf) / Vf)
        q = mp.mpf(qf)
    else:
        q = q_init

    for _ in range(25):
        R_val, V_val = gl_koopman_eval(q, d, p, M)
        res = R_val - target_R
        if abs(res) < mp.mpf("1e-74"):
            break
        q = q * mp.exp(-res / V_val)

    one_minus_s = 1 - s
    G_val = one_minus_s - (2 - s) * V_val
    D_val = mp.log(q * s * s / one_minus_s)
    Ds_val = -G_val / (s * one_minus_s * V_val)
    return {
        "q": q,
        "R": R_val,
        "V": V_val,
        "G": G_val,
        "D": D_val,
        "Ds": Ds_val,
        "residual": abs(R_val + mp.log(s)),
    }


# ============================================================================
# 4. Exact closed-form limit benchmarks (delta=0, phi=-1, phi->1-)
# ============================================================================

def exact_phi_one_limit(q: mp.mpf, delta: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """Closed-form limit as phi -> 1^- at fixed q > 0, delta in [0, 1]."""
    lam_m = (1 + mp.sqrt(1 + 4 * q * (1 - delta))) / 2
    lam_p = (1 + mp.sqrt(1 + 4 * q * (1 + delta))) / 2
    R_val = (mp.log(lam_m) + mp.log(lam_p)) / 2
    V_val = ((lam_m - 1) / (2 * lam_m - 1) + (lam_p - 1) / (2 * lam_p - 1)) / 2
    s_val = mp.exp(-R_val)
    G_val = (1 - s_val) - (2 - s_val) * V_val
    D_val = mp.log(q * s_val * s_val / (1 - s_val))
    Ds_val = -G_val / (s_val * (1 - s_val) * V_val)
    return s_val, R_val, V_val, D_val, Ds_val


# ============================================================================
# Main verification suite
# ============================================================================

def run_all() -> None:
    t0 = time.monotonic()
    results: dict[str, object] = {"dps": mp.mp.dps, "checks": []}
    failures: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}", flush=True)
        results["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append(name)

    print("=" * 78)
    print(f"secondary independent referee check (r37-gap1-check) | mpmath dps = {mp.mp.dps}")
    print("Pre-registered tolerances:")
    print("  G0 (O(M^2) modal refinement vs dense mp.lu_solve): 1e-74")
    print("  G1 (constant host): 1e-70 | G2 (alternating phi=-1): 1e-60")
    print("  G3 (phi->1^- closed form): 1e-60 | G4 (small-delta & s->1 asymptotics): 1e-30")
    print("  G5 (memA moderate-noise & 4th-order s-secant): 1e-16 / 1e-45")
    print("  Exact rational polynomial & 256-word identities: 0 (exact Fraction)")
    print("  Series vs GL-modal Koopman cross-check on proved region: 1e-65")
    print("=" * 78, flush=True)

    # ------------------------------------------------------------------------
    # PART A: Known-result validation gates (study rules Rule 7)
    # ------------------------------------------------------------------------
    R_dense, V_dense = gl_koopman_eval(mp.mpf("2.12"), mp.mpf("0.4"), mp.mpf("0.7"), M=32, force_dense_lu=True)
    R_fast, V_fast = gl_koopman_eval(mp.mpf("2.12"), mp.mpf("0.4"), mp.mpf("0.7"), M=32, force_dense_lu=False)
    err_g0 = max(abs(R_dense - R_fast), abs(V_dense - V_fast))
    record("G0 O(M^2) 80-digit modal refinement vs full dense mp.lu_solve", err_g0 < mp.mpf("1e-74"), f"max_err={mp.nstr(err_g0, 12)}")

    q_test = mp.mpf("0.025")
    r_const = hp_q_series_coeffs(ZERO, mp.mpf("0.6"), deg=110)
    R_ser, V_ser = eval_q_series(r_const, q_test)
    R_gl, V_gl = gl_koopman_eval(q_test, ZERO, mp.mpf("0.6"), M=28)
    lam_exact = (1 + mp.sqrt(1 + 4 * q_test)) / 2
    R_exact = mp.log(lam_exact)
    V_exact = (lam_exact - 1) / (2 * lam_exact - 1)
    err_g1 = max(abs(R_ser - R_exact), abs(V_ser - V_exact), abs(R_gl - R_exact), abs(V_gl - V_exact))
    record("G1 constant-host exact Perron root (series & GL Koopman)", err_g1 < mp.mpf("1e-70"), f"max_err={mp.nstr(err_g1, 12)}")

    err_g2 = ZERO
    s_crit = mp.mpf(4) / mp.sqrt(17)
    for d_str in ("0.3", "0.7"):
        d_val = mp.mpf(d_str)
        for s_val in (mp.mpf("0.985"), s_crit, mp.mpf("0.90"), mp.mpf("0.50")):
            M_g2 = 36 if s_val >= mp.mpf("0.90") else 64
            sol = solve_s_gl_koopman(s_val, d_val, mp.mpf("-1"), M=M_g2)
            Q_alt = mp.sqrt(d_val * d_val + (1 - d_val * d_val) * s_val * s_val)
            D_alt = mp.log((1 + s_val) / (1 + Q_alt))
            Ds_alt = 1 / (1 + s_val) - (1 - d_val * d_val) * s_val / (Q_alt * (1 + Q_alt))
            err_g2 = max(err_g2, abs(sol["D"] - D_alt), abs(sol["Ds"] - Ds_alt))
    record("G2 alternating host phi=-1 (memA P8 D and D')", err_g2 < mp.mpf("1e-60"), f"max_err={mp.nstr(err_g2, 12)}")

    err_g3 = ZERO
    for d_str in ("0.25", "0.85", "0.9999"):
        d_val = mp.mpf(d_str)
        r_phi1 = hp_q_series_coeffs(d_val, mp.mpf(1), deg=130)
        R_s1, V_s1 = eval_q_series(r_phi1, mp.mpf("0.02"))
        _, R_ex1, V_ex1, _, _ = exact_phi_one_limit(mp.mpf("0.02"), d_val)
        err_g3 = max(err_g3, abs(R_s1 - R_ex1), abs(V_s1 - V_ex1))
    r_corner = hp_q_series_coeffs(mp.mpf(1), mp.mpf(1), deg=140)
    sol_corner = solve_q_from_s_series(r_corner, s_crit)
    Ds_corner_exact = -(2 + s_crit) / (s_crit * (1 + s_crit))
    q_corner_exact = (1 - s_crit * s_crit) / (2 * s_crit ** 4)  # 17/512
    err_g3 = max(err_g3, abs(sol_corner["Ds"] - Ds_corner_exact), abs(sol_corner["q"] - q_corner_exact))
    record("G3 static-mixture phi->1^- & corner (delta->1^-, phi->1^-) exact formulas", err_g3 < mp.mpf("1e-60"), f"max_err={mp.nstr(err_g3, 12)}, q_corner(4/sqrt(17))={mp.nstr(sol_corner['q'], 16)} vs 17/512={17/512:.16f}")

    # G4: Small-noise delta -> 0 (memA P1/P2) and long-stay s -> 1^- (memA P5)
    d_tiny = mp.mpf("1e-16")
    p_g4 = mp.mpf("0.65")
    s_g4 = mp.mpf("0.45")
    sol_tiny = solve_s_gl_koopman(s_g4, d_tiny, p_g4, M=36)
    a_g4 = 1 - s_g4
    dF_ds = -(1 + 2 * p_g4 + 2 * a_g4 * p_g4 - (a_g4 * p_g4) ** 2) / (2 * (1 + a_g4) ** 2 * (1 + a_g4 * p_g4) ** 2)
    err_small_d = abs(sol_tiny["Ds"] / (d_tiny * d_tiny) - dF_ds)

    eps_tiny = mp.mpf("1e-16")
    d_p5, p_p5 = mp.mpf("0.7"), mp.mpf("0.8")
    r_p5 = hp_q_series_coeffs(d_p5, p_p5, deg=120)
    sol_s1 = solve_q_from_s_series(r_p5, 1 - eps_tiny)
    A_p5 = d_p5 ** 2 * (mp.mpf("0.5") + p_p5)
    B_p5 = -d_p5 ** 2 * (mp.mpf("0.5") + p_p5 + p_p5 ** 2) + mp.mpf("1.5") * A_p5 ** 2
    err_s1 = abs(sol_s1["Ds"] - (-A_p5 - 2 * B_p5 * eps_tiny))
    record("G4 small-noise delta->0 (memA P2) & s->1^- (memA P5) derivative asymptotics", max(err_small_d, err_s1) < mp.mpf("1e-30"), f"err_delta0={mp.nstr(err_small_d, 10)}, err_s1={mp.nstr(err_s1, 10)}")

    # G5: memA nonlinear moderate-noise benchmark + 4th-order s-secant check
    sol_m40 = solve_s_gl_koopman(mp.mpf("0.5"), mp.mpf("0.4"), mp.mpf("0.7"), M=40)
    sol_m56 = solve_s_gl_koopman(mp.mpf("0.5"), mp.mpf("0.4"), mp.mpf("0.7"), M=56, q_init=sol_m40["q"])
    D_memA = mp.mpf("0.0582995785286115578")
    err_memA = abs(sol_m56["D"] - D_memA)
    mesh_diff_50 = max(abs(sol_m56["D"] - sol_m40["D"]), abs(sol_m56["Ds"] - sol_m40["Ds"]))
    h_fd = mp.mpf("1e-14")
    Dm2 = solve_s_gl_koopman(mp.mpf("0.5") - 2 * h_fd, mp.mpf("0.4"), mp.mpf("0.7"), M=44, q_init=sol_m56["q"])["D"]
    Dm1 = solve_s_gl_koopman(mp.mpf("0.5") - h_fd, mp.mpf("0.4"), mp.mpf("0.7"), M=44, q_init=sol_m56["q"])["D"]
    Dp1 = solve_s_gl_koopman(mp.mpf("0.5") + h_fd, mp.mpf("0.4"), mp.mpf("0.7"), M=44, q_init=sol_m56["q"])["D"]
    Dp2 = solve_s_gl_koopman(mp.mpf("0.5") + 2 * h_fd, mp.mpf("0.4"), mp.mpf("0.7"), M=44, q_init=sol_m56["q"])["D"]
    Ds_fd4 = (Dm2 - 8 * Dm1 + 8 * Dp1 - Dp2) / (12 * h_fd)
    err_fd4 = abs(Ds_fd4 - sol_m56["Ds"])
    record(
        "G5 memA benchmark (s=0.5,d=0.4,phi=0.7) + M=40/56 refinement + 4th-order s-secant",
        err_memA < mp.mpf("1e-16") and mesh_diff_50 < mp.mpf("1e-45") and err_fd4 < mp.mpf("1e-45"),
        f"err_memA={mp.nstr(err_memA, 10)}, mesh_diff={mp.nstr(mesh_diff_50, 10)}, fd4_err={mp.nstr(err_fd4, 10)}",
    )

    # ------------------------------------------------------------------------
    # PART B: Exact rational verification of Eqs. (3)-(7) through N=12
    # ------------------------------------------------------------------------
    r_rat, e_rat, g_rat = exact_rational_recurrence(n_max=12)

    g2_expected: Poly = {(2, 0): Fraction(1, 2), (2, 1): Fraction(1)}
    g3_expected: Poly = {(2, 0): Fraction(-4), (2, 1): Fraction(-8), (2, 2): Fraction(-2)}
    div_even_ok = (
        g_rat[0] == {}
        and g_rat[1] == {}
        and g_rat[2] == g2_expected
        and g_rat[3] == g3_expected
        and all(ed >= 2 and ed % 2 == 0 for n in range(2, 13) for (ed, _) in g_rat[n])
    )
    record("B1 exact rational g_0=g_1=0, g_2, g_3, and delta^2*evenness through n=12", div_even_ok, f"g_2={g_rat[2]}, g_3={g_rat[3]}")

    expected_l1 = {
        2: Fraction(3, 2),
        3: Fraction(14, 1),
        4: Fraction(905, 8),
        5: Fraction(3531, 4),
        6: Fraction(109067, 16),
        7: Fraction(104991, 2),
        8: Fraction(51752565, 128),
    }
    l1_norms = {n: sum(abs(v) for v in g_rat[n].values()) for n in range(2, 13)}
    even_pos_4_8 = all(v > 0 for n in (4, 6, 8) for v in g_rat[n].values())
    odd_neg_5_7 = all(v < 0 for n in (5, 7) for v in g_rat[n].values())
    higher_alt = (
        all(v < 0 for n in (9, 11) for v in g_rat[n].values())
        and all(v > 0 for n in (10, 12) for v in g_rat[n].values())
    )
    l1_match = all(l1_norms[n] == expected_l1[n] for n in range(2, 9))
    record(
        "B2 exact monomial signs (g_4,6,8 > 0; g_5,7 < 0; continues for 9..12) & L1 norms",
        even_pos_4_8 and odd_neg_5_7 and l1_match and higher_alt,
        f"L1(g_5/d^2)={l1_norms[5]}, L1(g_7/d^2)={l1_norms[7]}, n=9..12 strict (-1)^n alternation={higher_alt}",
    )

    word_ok = True
    for d_f, p_f in [(Fraction(1, 3), Fraction(2, 5)), (Fraction(4, 5), Fraction(0, 1)), (Fraction(9, 10), Fraction(7, 8))]:
        r_words = exact_rational_word_series(d_f, p_f, deg=8)
        for n in range(1, 9):
            if poly_eval_fraction(r_rat[n], d_f, p_f) != r_words[n]:
                word_ok = False
    record("B3 exact 256-word rational path-tree enumeration vs Eq. (4) (0 error)", word_ok, "3 rational (delta, phi) pairs, degrees 1..8, exact Fraction equality")

    q_30 = Fraction(1, 30)
    margin_30 = (
        Fraction(1, 2)
        - 4 * q_30
        - l1_norms[5] * (q_30 ** 3)
        - l1_norms[7] * (q_30 ** 5)
        - Fraction(25, 9) * (Fraction(10) ** 9) * (q_30 ** 7) / (1 - 10 * q_30)
    )
    s2_crit = Fraction(16, 17)
    q_chord_crit = (1 - s2_crit) / (4 * s2_crit - 2)
    record(
        "B4 exact rational bound at q=1/30 & chord inversion at s^2=16/17",
        margin_30 == Fraction(185403593, 1312200000) and margin_30 > Fraction(14, 100) and q_chord_crit == Fraction(1, 30),
        f"margin={margin_30} ({float(margin_30):.12f}), q_chord(4/sqrt(17))={q_chord_crit}",
    )

    max_w_mod = ZERO
    max_b_mod = ZERO
    max_qwq_mod = ZERO
    max_G_over_d2 = ZERO
    for kq in range(16):
        qc = mp.mpf("0.1") * mp.exp(2j * mp.pi * kq / 16)
        for kd in range(16):
            dc = mp.mpf("1.2") * mp.exp(2j * mp.pi * kd / 16)
            for pat in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                w = mp.mpc(0)
                qwq = mp.mpc(0)
                for step in range(80):
                    z = pat[step & 1]
                    h = 1 + dc * z
                    b_step = qc * h / (1 + w) ** 2
                    w_new = qc * h / (1 + w)
                    qwq = w_new - b_step * qwq
                    w = w_new
                max_w_mod = max(max_w_mod, abs(w))
                max_b_mod = max(max_b_mod, abs(b_step))
                max_qwq_mod = max(max_qwq_mod, abs(qwq))
    for kq in range(12):
        qc = mp.mpf("0.06") * mp.exp(2j * mp.pi * kq / 12)
        for kd in range(12):
            dc = mp.mpf("1.2") * mp.exp(2j * mp.pi * kd / 12)
            for pv in (ZERO, mp.mpf("0.5"), mp.mpf("0.99")):
                R_c = sum(
                    sum(mp.mpf(c.numerator) / c.denominator * (dc ** ed) * (pv ** ep) for (ed, ep), c in r_rat[n].items())
                    * (qc ** n)
                    for n in range(1, 13)
                )
                V_c = sum(
                    n
                    * sum(mp.mpf(c.numerator) / c.denominator * (dc ** ed) * (pv ** ep) for (ed, ep), c in r_rat[n].items())
                    * (qc ** n)
                    for n in range(1, 13)
                )
                G_c = 1 - mp.exp(-R_c) - (2 - mp.exp(-R_c)) * V_c
                max_G_over_d2 = max(max_G_over_d2, abs(G_c) / abs(dc) ** 2)
    record(
        "B5 complex bidisk boundary audit (|q|=1/10, |delta|=6/5)",
        max_w_mod <= mp.mpf("0.33") + mp.mpf("1e-30")
        and max_b_mod <= mp.mpf("0.495") + mp.mpf("1e-30")
        and max_qwq_mod < mp.mpf(2) / 3
        and max_G_over_d2 < mp.mpf(25) / 9,
        f"max|w|={mp.nstr(max_w_mod, 8)} (<=0.33), max|b|={mp.nstr(max_b_mod, 8)} (<=0.495), max|qw_q|={mp.nstr(max_qwq_mod, 8)} (<2/3), max|G/d^2|={mp.nstr(max_G_over_d2, 8)} (<25/9=2.7778)",
    )

    # ------------------------------------------------------------------------
    # PART C: High-precision grid on s in [4/sqrt(17), 1), including
    #         delta -> 1^-, phi -> 1^-, and s -> 1^-
    # ------------------------------------------------------------------------
    deltas_proved = ["0.001", "0.05", "0.25", "0.5", "0.75", "0.9", "0.99", "0.999", "0.99999"]
    phis_proved = ["0", "0.25", "0.5", "0.75", "0.9", "0.99", "0.999", "0.99999"]
    s_proved = [
        ("4/sqrt(17)", s_crit),
        ("0.972", mp.mpf("0.972")),
        ("0.975", mp.mpf("0.975")),
        ("0.98", mp.mpf("0.98")),
        ("0.99", mp.mpf("0.99")),
        ("1-1e-3", 1 - mp.mpf("1e-3")),
        ("1-1e-6", 1 - mp.mpf("1e-6")),
        ("1-1e-10", 1 - mp.mpf("1e-10")),
        ("1-1e-16", 1 - mp.mpf("1e-16")),
    ]

    proved_rows = []
    max_Ds_proved = mp.mpf("-inf")
    min_scaled_G = mp.mpf("inf")
    max_q_at_crit = ZERO
    max_q_at_crit_pair = ("", "")
    max_cauchy_tail_ratio = ZERO
    ineq_violations = 0

    for d_str in deltas_proved:
        d_val = mp.mpf(d_str)
        for p_str in phis_proved:
            p_val = mp.mpf(p_str)
            r_coeffs = hp_q_series_coeffs(d_val, p_val, deg=130)
            for s_label, s_val in s_proved:
                sol = solve_q_from_s_series(r_coeffs, s_val)
                q_val = sol["q"]
                G_val = sol["G"]
                Ds_val = sol["Ds"]
                cauchy_tail = (mp.mpf(25) / 9) * (d_val ** 2) * (10 * q_val) ** 131 / (1 - 10 * q_val)
                tail_ratio = cauchy_tail / abs(G_val)
                max_cauchy_tail_ratio = max(max_cauchy_tail_ratio, tail_ratio)

                q_chord_ub = (1 - s_val * s_val) / (4 * s_val * s_val - 2)
                scaled_G = G_val / (d_val * d_val * q_val * q_val)
                if not (q_val <= q_chord_ub and q_val <= mp.mpf(1) / 30 and G_val - cauchy_tail > 0 and Ds_val < 0):
                    ineq_violations += 1
                if scaled_G < mp.mpf(185403593) / 1312200000:
                    ineq_violations += 1

                max_Ds_proved = max(max_Ds_proved, Ds_val)
                min_scaled_G = min(min_scaled_G, scaled_G)
                if s_label == "4/sqrt(17)" and q_val > max_q_at_crit:
                    max_q_at_crit = q_val
                    max_q_at_crit_pair = (d_str, p_str)

                proved_rows.append([
                    d_str,
                    p_str,
                    s_label,
                    mp.nstr(s_val, 25),
                    mp.nstr(q_val, 25),
                    mp.nstr(sol["D"], 25),
                    mp.nstr(Ds_val, 25),
                    mp.nstr(G_val, 25),
                    mp.nstr(scaled_G, 20),
                ])

    with (HERE / "proved_interval_grid.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "phi", "s_label", "s", "q", "D", "D_s", "G", "G_over_delta2_q2"])
        w.writerows(proved_rows)

    record(
        f"C1 80-digit root-solved grid on s in [4/sqrt(17), 1) ({len(proved_rows)} points, incl. d->1, phi->1, s->1)",
        ineq_violations == 0 and max_Ds_proved < 0 and max_cauchy_tail_ratio < mp.mpf("1e-50"),
        f"violations={ineq_violations}, max D'(s)={mp.nstr(max_Ds_proved, 12)}, min G/(d^2 q^2)={mp.nstr(min_scaled_G, 12)} (>0.141292), max q(4/sqrt(17))={mp.nstr(max_q_at_crit, 12)} at {max_q_at_crit_pair} (<1/30={1/30:.12f}), max Cauchy_tail/|G|={mp.nstr(max_cauchy_tail_ratio, 8)}",
    )

    corner_cases = [
        ("0.001", "0", s_crit),
        ("0.5", "0.5", s_crit),
        ("0.99", "0.99", s_crit),
        ("0.99999", "0", s_crit),
        ("0.99999", "0.99", s_crit),
        ("0.999", "0.999", mp.mpf("0.985")),
        ("0.99999", "0.99999", s_crit),
    ]
    max_cross_err = ZERO
    for d_str, p_str, s_val in corner_cases:
        d_val, p_val = mp.mpf(d_str), mp.mpf(p_str)
        r_c = hp_q_series_coeffs(d_val, p_val, deg=135)
        sol_ser = solve_q_from_s_series(r_c, s_val)
        sol_gl = solve_s_gl_koopman(s_val, d_val, p_val, M=36, q_init=sol_ser["q"])
        max_cross_err = max(
            max_cross_err,
            abs(sol_ser["q"] - sol_gl["q"]),
            abs(sol_ser["D"] - sol_gl["D"]),
            abs(sol_ser["Ds"] - sol_gl["Ds"]),
        )
    record(
        "C2 q-series vs GL modal Koopman on extreme corners (incl. d=0.99999, phi=0.99999, s=4/sqrt(17))",
        max_cross_err < mp.mpf("1e-65"),
        f"max |series - GL_Koopman| across q, D, D'(s) = {mp.nstr(max_cross_err, 12)}",
    )

    # ------------------------------------------------------------------------
    # PART D: Probing just below 4/sqrt(17) and across (0, 4/sqrt(17)):
    #         Is 4/sqrt(17) sharp or only a limit of the method?
    # ------------------------------------------------------------------------
    def proof_lb_rhs(qv: mp.mpf) -> mp.mpf:
        return (
            mp.mpf("0.5")
            - 4 * qv
            - (mp.mpf(3531) / 4) * qv ** 3
            - (mp.mpf(104991) / 2) * qv ** 5
            - (mp.mpf(25) / 9) * (10 ** 9) * (qv ** 7) / (1 - 10 * qv)
        )

    q_lb_zero = mp.findroot(proof_lb_rhs, mp.mpf("0.0365"))
    s_lb_zero = mp.sqrt((1 + 2 * q_lb_zero) / (1 + 4 * q_lb_zero))

    def proof_lb_rhs_N12(qv: mp.mpf) -> mp.mpf:
        return (
            mp.mpf("0.5")
            - 4 * qv
            - (mp.mpf(l1_norms[5].numerator) / l1_norms[5].denominator) * qv ** 3
            - (mp.mpf(l1_norms[7].numerator) / l1_norms[7].denominator) * qv ** 5
            - (mp.mpf(l1_norms[9].numerator) / l1_norms[9].denominator) * qv ** 7
            - (mp.mpf(l1_norms[11].numerator) / l1_norms[11].denominator) * qv ** 9
            - (mp.mpf(25) / 9) * (10 ** 13) * (qv ** 11) / (1 - 10 * qv)
        )

    q_lb_zero_12 = mp.findroot(proof_lb_rhs_N12, mp.mpf("0.045"))
    s_lb_zero_12 = mp.sqrt((1 + 2 * q_lb_zero_12) / (1 + 4 * q_lb_zero_12))

    s_below_near = [
        ("4/sqrt(17)-1e-6", s_crit - mp.mpf("1e-6")),
        ("4/sqrt(17)-1e-4", s_crit - mp.mpf("1e-4")),
        ("0.970", mp.mpf("0.970")),
        ("0.968", mp.mpf("0.968")),
        ("0.965", mp.mpf("0.965")),
        ("0.960", mp.mpf("0.960")),
        ("0.950", mp.mpf("0.950")),
        ("0.930", mp.mpf("0.930")),
        ("0.900", mp.mpf("0.900")),
    ]
    deltas_probe = ["0.01", "0.25", "0.5", "0.8", "0.95", "0.99", "0.9999"]
    phis_probe = ["0", "0.5", "0.9", "0.99", "0.9999"]

    below_rows = []
    max_Ds_below_near = mp.mpf("-inf")
    max_mesh_below_near = ZERO
    for d_str in deltas_probe:
        d_val = mp.mpf(d_str)
        for p_str in phis_probe:
            p_val = mp.mpf(p_str)
            for s_label, s_val in s_below_near:
                sol_36 = solve_s_gl_koopman(s_val, d_val, p_val, M=36)
                sol_52 = solve_s_gl_koopman(s_val, d_val, p_val, M=52, q_init=sol_36["q"])
                diff_Ds = abs(sol_52["Ds"] - sol_36["Ds"])
                max_mesh_below_near = max(max_mesh_below_near, diff_Ds)
                max_Ds_below_near = max(max_Ds_below_near, sol_52["Ds"])
                below_rows.append([
                    d_str,
                    p_str,
                    s_label,
                    mp.nstr(s_val, 25),
                    mp.nstr(sol_52["q"], 25),
                    mp.nstr(sol_52["D"], 25),
                    mp.nstr(sol_52["Ds"], 25),
                    mp.nstr(sol_52["G"], 25),
                    mp.nstr(diff_Ds, 12),
                ])

    s_global_probe = ["0.80", "0.60", "0.40", "0.20", "0.05", "0.01"]
    max_Ds_global_probe = mp.mpf("-inf")
    max_mesh_global_probe = ZERO
    global_probe_count = 0
    for d_str in ("0.1", "0.5", "0.9", "0.99"):
        d_val = mp.mpf(d_str)
        for p_str in ("0", "0.5", "0.9", "0.99"):
            p_val = mp.mpf(p_str)
            for s_str in s_global_probe:
                s_val = mp.mpf(s_str)
                M_lo = 48 if s_val >= mp.mpf("0.40") else (64 if s_val >= mp.mpf("0.20") else 96)
                sol_a = solve_s_gl_koopman(s_val, d_val, p_val, M=M_lo)
                sol_b = solve_s_gl_koopman(s_val, d_val, p_val, M=M_lo + 24, q_init=sol_a["q"])
                diff_Ds = abs(sol_b["Ds"] - sol_a["Ds"])
                max_mesh_global_probe = max(max_mesh_global_probe, diff_Ds)
                max_Ds_global_probe = max(max_Ds_global_probe, sol_b["Ds"])
                global_probe_count += 1
                below_rows.append([
                    d_str,
                    p_str,
                    s_str,
                    mp.nstr(s_val, 25),
                    mp.nstr(sol_b["q"], 25),
                    mp.nstr(sol_b["D"], 25),
                    mp.nstr(sol_b["Ds"], 25),
                    mp.nstr(sol_b["G"], 25),
                    mp.nstr(diff_Ds, 12),
                ])

    with (HERE / "below_threshold_probe.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "phi", "s_label", "s", "q", "D", "D_s", "G", "Ds_mesh_diff"])
        w.writerows(below_rows)

    record(
        f"D1 probe just below 4/sqrt(17) ({len(deltas_probe)*len(phis_probe)*len(s_below_near)} points on [0.90, 4/sqrt(17)))",
        max_Ds_below_near < 0 and max_mesh_below_near < mp.mpf("1e-55"),
        f"all negative: max D'(s)={mp.nstr(max_Ds_below_near, 12)}, max M=36/52 diff={mp.nstr(max_mesh_below_near, 10)}; N=8 bound root q*={mp.nstr(q_lb_zero, 10)} (s*={mp.nstr(s_lb_zero, 10)}); N=12 bound root q*={mp.nstr(q_lb_zero_12, 10)} (s*={mp.nstr(s_lb_zero_12, 10)})",
    )
    record(
        f"D2 global probe on s in [0.01, 0.80] ({global_probe_count} points, incl. d=0.99, phi=0.99)",
        max_Ds_global_probe < 0 and max_mesh_global_probe < mp.mpf("1e-18"),
        f"all negative: max D'(s)={mp.nstr(max_Ds_global_probe, 12)}, max mesh diff={mp.nstr(max_mesh_global_probe, 10)}",
    )

    # ------------------------------------------------------------------------
    # PART E: Spot-check primary's saved CSV rows with secondary's solver
    # ------------------------------------------------------------------------
    primary_hp_csv = GAP1_DIR / "high_precision_collocation_32.csv"
    max_independent_diff = ZERO
    with primary_hp_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_c = mp.mpf(row["q"])
            d_c = mp.mpf(row["delta"])
            p_c = mp.mpf(row["phi"])
            M_use = 48 if q_c < mp.mpf("1e4") else 96
            R_cb, V_cb = gl_koopman_eval(q_c, d_c, p_c, M=M_use)
            s_cb = mp.exp(-R_cb)
            one_m_s = -mp.expm1(-R_cb)
            G_cb = one_m_s - (2 - s_cb) * V_cb
            D_cb = mp.log(q_c) - 2 * R_cb - mp.log(one_m_s)
            Ds_cb = -G_cb / (s_cb * one_m_s * V_cb)
            max_independent_diff = max(
                max_independent_diff,
                abs(s_cb - mp.mpf(row["s"])),
                abs(D_cb - mp.mpf(row["D"])),
                abs(Ds_cb - mp.mpf(row["D_s"])),
            )

    q_worst = mp.mpf("691290300440.24")
    R_w176, V_w176 = gl_koopman_eval(q_worst, mp.mpf("0.95"), ZERO, M=176)
    R_w208, V_w208 = gl_koopman_eval(q_worst, mp.mpf("0.95"), ZERO, M=208)
    s_w = mp.exp(-R_w208)
    a_w = -mp.expm1(-R_w208)
    G_w = a_w - (2 - s_w) * V_w208
    Ds_w208 = -G_w / (s_w * a_w * V_w208)
    s_w0 = mp.exp(-R_w176)
    a_w0 = -mp.expm1(-R_w176)
    Ds_w176 = -(a_w0 - (2 - s_w0) * V_w176) / (s_w0 * a_w0 * V_w176)
    worst_cell_ref = abs(Ds_w208 - Ds_w176)

    record(
        "E1 cross-check against primary high_precision_collocation_32.csv + worst small-s cell",
        max_independent_diff < mp.mpf("1e-20") and Ds_w208 < -4429 and worst_cell_ref < mp.mpf("1e-9"),
        f"max |secondary_GL - primary_Chebyshev|={mp.nstr(max_independent_diff, 12)}, worst-cell Ds={mp.nstr(Ds_w208, 16)} (M=176/208 diff={mp.nstr(worst_cell_ref, 10)})",
    )

    elapsed = round(time.monotonic() - t0, 2)
    results["Summary"] = {
        "proved_grid_points": len(proved_rows),
        "max_Ds_proved": mp.nstr(max_Ds_proved, 20),
        "min_G_over_d2_q2_proved": mp.nstr(min_scaled_G, 20),
        "max_q_at_4_over_sqrt17": mp.nstr(max_q_at_crit, 20),
        "max_q_at_4_over_sqrt17_pair": max_q_at_crit_pair,
        "q_lb_zero_N8": mp.nstr(q_lb_zero, 16),
        "s_lb_zero_N8": mp.nstr(s_lb_zero, 16),
        "q_lb_zero_N12": mp.nstr(q_lb_zero_12, 16),
        "s_lb_zero_N12": mp.nstr(s_lb_zero_12, 16),
        "near_below_points": len(deltas_probe) * len(phis_probe) * len(s_below_near),
        "max_Ds_below_near": mp.nstr(max_Ds_below_near, 20),
        "global_probe_points": global_probe_count,
        "max_Ds_global_probe": mp.nstr(max_Ds_global_probe, 20),
        "max_linear_residual": mp.nstr(MAX_LIN_RESIDUAL, 12),
        "failures": failures,
        "elapsed_seconds": elapsed,
    }
    (HERE / "verify_gap1_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("=" * 78)
    print(f"Largest 80-digit linear residual: {mp.nstr(MAX_LIN_RESIDUAL, 12)}")
    print(f"COMPLETED in {elapsed}s | FAILED CHECKS: {len(failures)} {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
