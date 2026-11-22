#!/usr/bin/env python3
"""Independent mpmath.iv verification of GAP 4 (Part 2: operator coboundary, Gaussian chains, memB tallies).

Zero imports from gap4_certify.py or any collaborator script.
Implements:
  - Downward Chebyshev differentiation recurrence + l1-bounded local Taylor evaluator on mpmath.iv
  - Chebyshev-Gauss interior-node collocation candidate + dyadic cellwise Taylor-Lagrange coboundary enclosure
  - Pure-mpmath Hermite sign-change root brackets + Neumann-certified interval matrix inverses for n=12, 20 chains
  - Exact interval verification of all 4,320 rows (144 curves) of memB_tables.csv
"""

from collections import defaultdict
import csv
from decimal import Decimal, ROUND_CEILING, localcontext
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import mpmath
from mpmath import mp, iv

mp.dps = 85
iv.dps = 85

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
GAP4_JSON = (__import__('pathlib').Path(__file__).resolve().parent / '../../gap4/gap4_results.json').resolve()
PART1_JSON = HERE / "verify_gap4_part1_results.json"
MEMB_CSV = (__import__('pathlib').Path(__file__).resolve().parent / '../../studies/memory-memB/memB_tables.csv').resolve()

ZERO = iv.mpf(0)
ONE = iv.mpf(1)
HALF = iv.mpf("0.5")


def iv_lo_mpf(x):
    return mp.mpf(iv.mpf(x)._mpi_[0])


def iv_hi_mpf(x):
    return mp.mpf(iv.mpf(x)._mpi_[1])


def iv_ball(center, rad):
    c = iv.mpf(center)
    r = abs(iv.mpf(rad)).b
    return iv.mpf([(c - r).a, (c + r).b])


def iv_abs_upper(x):
    return abs(iv.mpf(x)).b


def iv_abs_lower(x):
    return abs(iv.mpf(x)).a


def iv_union(x, y):
    x = iv.mpf(x)
    y = iv.mpf(y)
    lo = x.a if bool(x.a <= y.a) else y.a
    hi = x.b if bool(x.b >= y.b) else y.b
    return iv.mpf([lo, hi])


def format_outward_pm(x, sig_mid=25, sig_rad=4):
    x = iv.mpf(x)
    lo_m = iv_lo_mpf(x)
    hi_m = iv_hi_mpf(x)
    if lo_m == 0 and hi_m == 0:
        return "0"
    mid_str = mpmath.nstr((lo_m + hi_m) / 2, sig_mid)
    m_iv = iv.mpf(mid_str)
    req_rad_m = iv_hi_mpf(abs(x - m_iv))
    if req_rad_m == 0 and bool(m_iv.a <= x.a) and bool(x.b <= m_iv.b):
        return mid_str
    raw_rad_str = mpmath.nstr(req_rad_m, sig_rad + 14)
    with localcontext() as dctx:
        dctx.prec = 65
        d_val = Decimal(raw_rad_str) * (Decimal(1) + Decimal(10) ** (-(sig_rad + 6)))
        if d_val == 0:
            d_val = Decimal("1e-90")
        exp = d_val.adjusted() - (sig_rad - 1)
        quant = Decimal(10) ** exp
        d_ceil = d_val.quantize(quant, rounding=ROUND_CEILING)
        rad_str = format(d_ceil.normalize(), "g")
    lo_chk = iv.mpf(mid_str) - iv.mpf(rad_str)
    hi_chk = iv.mpf(mid_str) + iv.mpf(rad_str)
    assert bool(lo_chk.a <= x.a) and bool(x.b <= hi_chk.b), f"Outward format failed for {x}"
    return f"[{mid_str} +/- {rad_str}]"


def parse_arb_str_to_iv(s):
    s = s.strip()
    m1 = re.match(r"^\[\s*([+-]?[0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)\s*\]$", s)
    if m1:
        m_iv = iv.mpf(m1.group(1))
        r_iv = iv.mpf(m1.group(2))
        return iv.mpf([(m_iv - r_iv).a, (m_iv + r_iv).b])
    m2 = re.match(r"^\[\s*\+/-\s*([0-9.eE+-]+)\s*\]$", s)
    if m2:
        r_iv = iv.mpf(m2.group(1))
        return iv.mpf([(-r_iv).a, r_iv.b])
    return iv.mpf(s)


def intervals_overlap(x, y):
    x = iv.mpf(x)
    y = iv.mpf(y)
    return bool((x.a <= y.b) and (y.a <= x.b))


def F_ar1(s, phi):
    a = 1 - s
    return a * (1 + (a + 2) * phi) / (2 * (1 + a) * (1 + a * phi))


def s_star_ar1(phi):
    return 1 - (1 - iv.sqrt(2 * (1 + phi))) / phi


# ===========================================================================
# 1. Chebyshev downward differentiation & Taylor-Lagrange coboundary operator
# ===========================================================================
def cheb_deriv_coeffs_iv(c):
    """Exact interval enclosure of Chebyshev coefficients of q'(z) given q(z) = sum_{k=0}^{M-1} c_k T_k(z)."""
    m = len(c)
    if m <= 1:
        return [ZERO]
    d = [ZERO] * (m - 1)
    for k in range(m - 1, 1, -1):
        d_kp1 = d[k + 1] if (k + 1) < (m - 1) else ZERO
        d[k - 1] = d_kp1 + iv.mpf(2 * k) * c[k]
    d2 = d[2] if 2 < m - 1 else ZERO
    d[0] = HALF * d2 + c[1]
    return d


def clenshaw_scalar_iv(coeffs, z_pt):
    """Evaluate sum_{k=0}^{K-1} coeffs[k] T_k(z_pt) for scalar/thin iv.mpf z_pt."""
    if not coeffs:
        return ZERO
    if len(coeffs) == 1:
        return coeffs[0]
    b1 = ZERO
    b2 = ZERO
    two_z = iv.mpf(2) * z_pt
    for ck in coeffs[:0:-1]:
        b0 = two_z * b1 - b2 + ck
        b2, b1 = b1, b0
    return coeffs[0] + z_pt * b1 - b2


class ChebPolyEvaluator:
    """Rigorous evaluator for q(z) and its scaled derivatives q^(r)(z)/r! on subintervals of [-1, 1],
    avoiding Clenshaw interval-dependency blowup via Mean Value Theorem + exact Chebyshev l1 slope bound.
    """

    def __init__(self, exact_dyadic_coeffs, max_deg):
        raw = [[iv.mpf(c) for c in exact_dyadic_coeffs]]
        for _ in range(max_deg + 1):
            raw.append(cheb_deriv_coeffs_iv(raw[-1]))
        self.max_deg = max_deg
        self.scaled_coeffs = []
        self.slope_l1 = []
        for r in range(max_deg + 1):
            fac_r = iv.mpf(math.factorial(r))
            self.scaled_coeffs.append([v / fac_r for v in raw[r]])
            # |(q^(r)/r!)'(z)| = |sum_l (raw[r+1][l]/r!) T_l(z)| <= sum_l |raw[r+1][l]|/r! on [-1, 1]
            l1 = ZERO
            for v in raw[r + 1]:
                l1 += iv_abs_upper(v)
            self.slope_l1.append(l1 / fac_r)

    def eval_scaled_deriv_on_interval(self, r, Z):
        """Enclose q^(r)(Z)/r! for interval Z subseteq [-1, 1]."""
        Z = iv.mpf(Z)
        assert bool(Z.a >= -ONE) and bool(Z.b <= ONE), f"Chebyshev argument {Z} exited [-1, 1]"
        zm_mpf = (iv_lo_mpf(Z) + iv_hi_mpf(Z)) / 2
        if zm_mpf < -1:
            zm_mpf = mp.mpf(-1)
        elif zm_mpf > 1:
            zm_mpf = mp.mpf(1)
        zm = iv.mpf(zm_mpf)
        val_mid = clenshaw_scalar_iv(self.scaled_coeffs[r], zm)
        rad_z = iv_abs_upper(Z - zm)
        if iv_hi_mpf(rad_z) == 0:
            return val_mid
        return iv_ball(val_mid, rad_z * self.slope_l1[r])

    def eval_jet_at_linear_arg(self, x0, h_step, deg):
        """Given linear jet X(t) = x0 + h_step * t (len deg+1), return jet of q(X(t)) up to t^deg."""
        g0 = self.eval_scaled_deriv_on_interval(0, x0)
        out = [g0]
        hp = ONE
        for r in range(1, deg + 1):
            hp = hp * h_step
            out.append(self.eval_scaled_deriv_on_interval(r, x0) * hp)
        return out

    def eval_jet_at_general_arg(self, X_jet, deg):
        """Given degree-`deg` jet X(t) = X_0 + Y(t) with X_0 subseteq [-1, 1] and Y(0) = 0,
        enclose q(X(t)) up to t^deg via degree-trimmed Horner in Y(t).
        """
        X0 = X_jet[0]
        g = [self.eval_scaled_deriv_on_interval(r, X0) for r in range(deg + 1)]
        res = [g[deg]]
        for r in range(deg - 1, -1, -1):
            lim = deg - r
            nxt = [g[r]] + [ZERO] * lim
            for i, ri in enumerate(res):
                if iv_hi_mpf(abs(ri)) == 0:
                    continue
                for j in range(1, lim - i + 1):
                    nxt[i + j] += ri * X_jet[j]
            res = nxt
        return res


def jet_log_positive(u_jet, deg):
    """Rigorous Taylor jet of log(u(t)) up to t^deg using u_0 * j * L_j = j * u_j - sum_{m=1}^{j-1} m * L_m * u_{j-m}."""
    u0 = u_jet[0]
    assert bool(u0 > 0), f"Non-positive log argument in jet: {u0}"
    L = [ZERO] * (deg + 1)
    L[0] = iv.log(u0)
    for j in range(1, deg + 1):
        acc = iv.mpf(j) * u_jet[j]
        for m in range(1, j):
            acc -= (iv.mpf(m) * L[m]) * u_jet[j - m]
        L[j] = acc / (iv.mpf(j) * u0)
    return L


def build_collocation_candidate(s_iv, c_iv, delta_iv, phi_iv, n):
    """Construct exact dyadic polynomial candidate coefficients (q_-, q_+) using
    Chebyshev-Gauss interior nodes x_k = cos((2k+1)*pi/(2n)) and float64 LU + mpmath
    iterative refinement. Also certify forward invariance of [vm - vr, vm + vr] in mpmath.iv!
    """
    # Certified forward-invariant interval in mpmath.iv
    width = iv.mpf(2) * c_iv * delta_iv / s_iv
    lo_u = iv.mpf(2) * (s_iv * width + c_iv * (ONE - delta_iv)) / (
        iv.sqrt((s_iv + width) ** 2 + iv.mpf(4) * c_iv * (ONE - delta_iv)) + width - s_iv
    )
    hi_u = lo_u + width
    vl = iv.log(lo_u).a - iv.mpf("1e-65")
    vh = iv.log(hi_u).b + iv.mpf("1e-65")
    vm = iv.mpf((iv_lo_mpf(vl) + iv_hi_mpf(vh)) / 2)
    vr = iv.mpf(iv_hi_mpf(iv_abs_upper(vh - vm) + iv_abs_upper(vm - vl)) / 2 + mp.mpf("1e-65"))
    # Strict verification of forward invariance on [vm - vr, vm + vr]
    f_minus_hi = iv.log(s_iv + c_iv * (ONE - delta_iv) * iv.exp(-(vm + vr)))
    f_plus_lo = iv.log(s_iv + c_iv * (ONE + delta_iv) * iv.exp(-(vm - vr)))
    assert bool(f_minus_hi >= vm - vr), "Forward invariance lower check failed"
    assert bool(f_plus_lo <= vm + vr), "Forward invariance upper check failed"

    # Assemble candidate collocation matrix in mpmath at 40 digits using Chebyshev-Gauss interior nodes
    with mp.workdps(40):
        s_m = iv_lo_mpf(s_iv)
        c_m = iv_lo_mpf(c_iv)
        d_m = iv_lo_mpf(delta_iv)
        p_m = iv_lo_mpf(phi_iv)
        vm_m = iv_lo_mpf(vm)
        vr_m = iv_lo_mpf(vr)

        dim = 2 * n + 1
        A_mp = [[mp.mpf(0)] * dim for _ in range(dim)]
        b_mp = [mp.mpf(0)] * dim

        def cheb_vals_mp(x_val, m_cnt):
            vals = [mp.mpf(1), x_val]
            for _ in range(2, m_cnt):
                vals.append(2 * x_val * vals[-1] - vals[-2])
            return vals[:m_cnt]

        row_idx = 0
        for i_state in range(2):
            for k in range(n):
                xk = mp.cos(mp.pi * (2 * k + 1) / (2 * n))
                vk = vm_m + vr_m * xk
                t_now = cheb_vals_mp(xk, n)
                row = A_mp[row_idx]
                row[-1] = mp.mpf(1)
                for l in range(n):
                    row[i_state * n + l] += t_now[l]
                for j_state, z_sgn in enumerate([-1, 1]):
                    prob = (1 + (p_m if i_state == j_state else -p_m)) / 2
                    xp = (mp.log(s_m + c_m * (1 + z_sgn * d_m) * mp.exp(-vk)) - vm_m) / vr_m
                    t_fut = cheb_vals_mp(xp, n)
                    for l in range(n):
                        row[j_state * n + l] -= prob * t_fut[l]
                b_mp[row_idx] = vk
                row_idx += 1
        A_mp[row_idx][0] = mp.mpf(1)
        b_mp[row_idx] = mp.mpf(0)

        # Solve in float64 via numpy.linalg.inv + 3 iterative refinement steps in 40-digit mpmath
        A_f64 = np.array([[float(v) for v in r] for r in A_mp], dtype=np.float64)
        invA_f64 = np.linalg.inv(A_f64)
        x_sol = [mp.mpf(0)] * dim
        for _ref in range(4):
            res_mp = [
                b_mp[r_i] - mp.fsum(A_mp[r_i][c_i] * x_sol[c_i] for c_i in range(dim))
                for r_i in range(dim)
            ]
            res_scale = max(abs(v) for v in res_mp)
            if res_scale == 0:
                break
            res_scaled_f64 = np.array([float(v / res_scale) for v in res_mp], dtype=np.float64)
            corr_f64 = invA_f64 @ res_scaled_f64
            for c_i in range(dim):
                x_sol[c_i] += mp.mpf(float(corr_f64[c_i])) * res_scale

        # Extract exact dyadic rational mpf coefficients
        coeff_exact = [[iv.mpf(x_sol[i_state * n + l]) for l in range(n)] for i_state in range(2)]

    return coeff_exact, vm, vr


def enclose_operator_rate(s_iv, c_iv, delta_iv, phi_iv, n=40, cells=64, degree=20):
    """Rigorous coboundary + cellwise Taylor-Lagrange interval enclosure of r(s, c)."""
    coeff_exact, vm, vr = build_collocation_candidate(s_iv, c_iv, delta_iv, phi_iv, n)
    evals = [ChebPolyEvaluator(coeff_exact[0], degree), ChebPolyEvaluator(coeff_exact[1], degree)]

    h_cell = ONE / iv.mpf(cells)
    p_same = (ONE + phi_iv) * HALF
    p_diff = (ONE - phi_iv) * HALF

    # Precompute (-vr)^j / j! for j = 0 .. degree
    neg_vr_pow = [ONE]
    for j in range(1, degree + 1):
        neg_vr_pow.append(neg_vr_pow[-1] * (-vr) / iv.mpf(j))

    def residual_jets_at(x_base):
        v0 = vm + vr * x_base
        exp_mv0 = iv.exp(-v0)
        fut_jets = []
        for j_state, z_sgn in enumerate([-1, 1]):
            scale = c_iv * (ONE + iv.mpf(z_sgn) * delta_iv) * exp_mv0
            u_jet = [scale * neg_vr_pow[j] for j in range(degree + 1)]
            u_jet[0] += s_iv
            log_u = jet_log_positive(u_jet, degree)
            xp_jet = [(log_u[0] - vm) / vr] + [log_u[j] / vr for j in range(1, degree + 1)]
            fut_jets.append(evals[j_state].eval_jet_at_general_arg(xp_jet, degree))
        cur_jets = [evals[i_state].eval_jet_at_linear_arg(x_base, ONE, degree) for i_state in range(2)]
        H_jets = []
        for i_state in range(2):
            hj = [ZERO] * (degree + 1)
            for m in range(degree + 1):
                hj[m] = (
                    p_same * fut_jets[i_state][m]
                    + p_diff * fut_jets[1 - i_state][m]
                    - cur_jets[i_state][m]
                )
            hj[0] += v0
            if degree >= 1:
                hj[1] += vr
            H_jets.append(hj)
        return H_jets

    h_pows = [ONE]
    for _ in range(1, degree + 1):
        h_pows.append(h_pows[-1] * h_cell)

    rate_enclosure = None
    for k in range(cells):
        x_mid = -ONE + iv.mpf(2 * k + 1) * h_cell
        x_dom = iv_ball(x_mid, h_cell)
        H_center = residual_jets_at(x_mid)
        H_domain = residual_jets_at(x_dom)
        for i_state in range(2):
            hc = H_center[i_state]
            hd = H_domain[i_state]
            rem = ZERO
            for m in range(1, degree):
                rem += iv_abs_upper(hc[m]) * h_pows[m]
            rem += iv_abs_upper(hd[degree]) * h_pows[degree]
            cell_enc = iv_ball(hc[0], rem)
            rate_enclosure = cell_enc if rate_enclosure is None else iv_union(rate_enclosure, cell_enc)

    return rate_enclosure


def enclose_operator_threshold(s_str, delta_str, phi_str, D0_str, n=40, cells=64, degree=20):
    s_iv = iv.mpf(s_str)
    d_iv = iv.mpf(delta_str)
    p_iv = iv.mpf(phi_str)
    D0_iv = iv.mpf(D0_str)
    c_iv = (ONE - s_iv) * iv.exp(D0_iv)
    rate_iv = enclose_operator_rate(s_iv, c_iv, d_iv, p_iv, n=n, cells=cells, degree=degree)
    c_min = (ONE - s_iv) * iv.exp(D0_iv - iv.mpf("0.01"))
    beta_iv = c_min * (ONE - d_iv) / (s_iv ** 2 + c_min * (ONE - d_iv))
    err_iv = (iv.mpf(3) * iv_abs_upper(rate_iv) / beta_iv).b
    assert bool(err_iv < iv.mpf("0.01")), f"Root error {err_iv} exceeded P3 slope window 0.01"
    D_iv = iv_ball(D0_iv, err_iv)
    return {"D": D_iv, "rate": rate_iv, "log_c_root_error": err_iv, "beta": beta_iv}


# ===========================================================================
# 2. Finite-state Gaussian Markov chains (Hermite sign brackets + Neumann inverse)
# ===========================================================================
def hermite_prob_iv(n, x):
    h0 = ONE
    if n == 0:
        return h0
    h1 = iv.mpf(x)
    for k in range(1, n):
        h0, h1 = h1, x * h1 - iv.mpf(k) * h0
    return h1


def hermite_prob_mp(n, x):
    h0 = mp.mpf(1)
    if n == 0:
        return h0
    h1 = mp.mpf(x)
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def enclose_hermite_quadrature(n):
    """Enclose all n roots and weights of probabilists' Hermite polynomial He_n
    using sign-change brackets of halfwidth 1e-76 and degree exhaustion.
    """
    # Seed approximations from numpy hermegauss, refined in 85-digit mpmath, then certified in iv!
    seeds_f64 = np.polynomial.hermite_e.hermegauss(n)[0]
    xs_iv = []
    h_br = iv.mpf("1e-76")
    for s_f64 in seeds_f64:
        xm = mp.mpf(float(s_f64))
        for _ in range(10):
            xm = xm - hermite_prob_mp(n, xm) / (n * hermite_prob_mp(n - 1, xm))
        xc = iv.mpf(xm)
        fa = hermite_prob_iv(n, xc - h_br)
        fb = hermite_prob_iv(n, xc + h_br)
        assert bool(fa * fb < 0), f"Sign bracket failed at Hermite seed {xm}"
        x_ball = iv_ball(xc, h_br)
        assert not (0 in hermite_prob_iv(n - 1, x_ball))
        if xs_iv:
            assert bool(xs_iv[-1] < x_ball), "Hermite root brackets not strictly ordered"
        xs_iv.append(x_ball)

    fac_nm1 = iv.mpf(math.factorial(n - 1))
    ws_iv = [fac_nm1 / (iv.mpf(n) * hermite_prob_iv(n - 1, x) ** 2) for x in xs_iv]
    assert 1 in sum(ws_iv, ZERO)
    assert 0 in sum((w * x for w, x in zip(ws_iv, xs_iv)), ZERO)
    assert 1 in sum((w * (x ** 2) for w, x in zip(ws_iv, xs_iv)), ZERO)
    return xs_iv, ws_iv


def neumann_certified_inverse(M_iv):
    """Rigorous n x n interval matrix inverse via exact dyadic midpoint inverse R + Neumann series
    M^{-1} = (I - (I - R M))^{-1} R. Certifies ||I - R M||_inf < 1 in mpmath.iv.
    """
    n = len(M_iv)
    with mp.workdps(85):
        M_mp = mp.matrix([[iv_lo_mpf(M_iv[i][j] + M_iv[i][j]) / 2 for j in range(n)] for i in range(n)])
        R_mp = M_mp ** (-1)
        R_iv = [[iv.mpf(R_mp[i, j]) for j in range(n)] for i in range(n)]

    # Compute E = I - R * M in mpmath.iv
    eta = ZERO
    R_row_norms = []
    for i in range(n):
        r_norm_i = ZERO
        e_norm_i = ZERO
        for j in range(n):
            r_norm_i += iv_abs_upper(R_iv[i][j])
            acc = ONE if i == j else ZERO
            for l in range(n):
                acc -= R_iv[i][l] * M_iv[l][j]
            e_norm_i += iv_abs_upper(acc)
        R_row_norms.append(r_norm_i)
        if bool(e_norm_i > eta):
            eta = e_norm_i
    assert bool(eta < ONE), f"Neumann condition failed: eta = {eta}"
    r_inf = max(R_row_norms, key=lambda v: iv_hi_mpf(v))
    entry_err = (eta * r_inf / (ONE - eta)).b
    return [[iv_ball(R_iv[i][j], entry_err) for j in range(n)] for i in range(n)]


def build_gaussian_finite_chain_iv(phi_iv, n, order):
    xs, ws = enclose_hermite_quadrature(n)
    one_m_p2 = ONE - phi_iv ** 2
    rt_omp2 = iv.sqrt(one_m_p2)
    K = [
        [
            ws[j]
            / rt_omp2
            * iv.exp(
                (iv.mpf(2) * phi_iv * xs[i] * xs[j] - (phi_iv ** 2) * (xs[i] ** 2 + xs[j] ** 2))
                / (iv.mpf(2) * one_m_p2)
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    row_sums = [sum(row, ZERO) for row in K]
    pi_raw = [ws[i] * row_sums[i] for i in range(n)]
    pi_norm = sum(pi_raw, ZERO)
    pi = [p / pi_norm for p in pi_raw]
    P = [[K[i][j] / row_sums[i] for j in range(n)] for i in range(n)]

    # Taylor coefficients in sigma up to `order` of (h_j(sigma) - 1)^r,
    # where h_j(sigma) = exp(sigma * x_j) / sum_i pi_i exp(sigma * x_i)
    norm_c = [
        sum((pi[j] * (xs[j] ** m) / iv.mpf(math.factorial(m)) for j in range(n)), ZERO)
        for m in range(order + 1)
    ]
    inv_norm = [ONE] + [ZERO] * order
    for m in range(1, order + 1):
        s_acc = ZERO
        for k in range(1, m + 1):
            s_acc += norm_c[k] * inv_norm[m - k]
        inv_norm[m] = -s_acc

    g_dev = [[ZERO] * n for _ in range(order + 1)]
    for m in range(1, order + 1):
        for j in range(n):
            acc = ZERO
            for k in range(m + 1):
                acc += (xs[j] ** k) / iv.mpf(math.factorial(k)) * inv_norm[m - k]
            g_dev[m][j] = acc

    # powers[r][m][j] = [sigma^m] (h_j(sigma) - 1)^r
    powers = [[[ZERO] * n for _ in range(order + 1)] for _ in range(order + 1)]
    powers[0][0] = [ONE] * n
    for r in range(1, order + 1):
        for m in range(r, order + 1):
            for j in range(n):
                acc = ZERO
                for k in range(1, m - r + 2):
                    acc += powers[r - 1][m - k][j] * g_dev[k][j]
                powers[r][m][j] = acc

    return xs, P, pi, powers


def mat_vec_iv(M, v):
    n = len(v)
    out = [ZERO] * n
    for i in range(n):
        acc = ZERO
        row = M[i]
        for j in range(n):
            acc += row[j] * v[j]
        out[i] = acc
    return out


def finite_gaussian_rate_iv(s_iv, c_iv, sigma_iv, chain, order):
    xs, P, pi, powers = chain
    n = len(xs)
    lam = (s_iv + iv.sqrt(s_iv ** 2 + iv.mpf(4) * c_iv)) * HALF
    b = c_iv / (lam ** 2)
    PT = [[P[j][i] for j in range(n)] for i in range(n)]

    inv_mats = {}
    sb = ONE
    for k in range(1, order + 1):
        sb = sb * (-b)
        M_k = [
            [(ONE if i == j else ZERO) - sb * PT[i][j] for j in range(n)]
            for i in range(n)
        ]
        inv_mats[k] = neumann_certified_inverse(M_k)

    mom = {}
    pmom = {(0, 0): mat_vec_iv(PT, pi)}
    coeffs = []
    comb = math.comb
    for tot in range(1, order + 1):
        for k in range(tot, 0, -1):
            rhs = [ZERO] * n
            bk = b ** k
            for r in range(k + 1):
                for m in range(tot - k + 1):
                    if r == 0 and m == 0:
                        continue
                    ell = k - r + m
                    fac = bk * iv.mpf(comb(k, r) * ((-1) ** (k - r + m)) * comb(k + m - 1, m))
                    p_range = [0] if r == 0 else range(r, tot - ell + 1)
                    for pwr in p_range:
                        prev = pmom.get((ell, tot - pwr))
                        if prev is None:
                            continue
                        pr_vec = powers[r][pwr]
                        for j in range(n):
                            rhs[j] += fac * pr_vec[j] * prev[j]
            vk = mat_vec_iv(inv_mats[k], rhs)
            mom[(k, tot)] = vk
            pmom[(k, tot)] = mat_vec_iv(PT, vk)
        c_tot = ZERO
        for k in range(1, tot + 1):
            c_tot += iv.mpf((-1) ** (k + 1)) / iv.mpf(k) * sum(mom[(k, tot)], ZERO)
        coeffs.append(c_tot)

    raw_rate = iv.log(lam)
    for idx, c_tot in enumerate(coeffs, start=1):
        raw_rate += c_tot * (sigma_iv ** idx)

    rd = (ONE - b) ** 2 / (iv.mpf(8) * b)
    x_max = max((iv_abs_upper(x) for x in xs), key=lambda v: iv_hi_mpf(v))
    sigma_rad = iv.log(ONE + rd / (iv.mpf(2) + rd)) / x_max
    q = iv_abs_upper(sigma_iv) / sigma_rad
    assert bool(q < ONE), f"Gaussian finite-chain sigma radius exceeded: q={q}"
    tail = iv.log(iv.mpf(2)) * (q ** (order + 1)) / (ONE - q)
    return iv_ball(raw_rate, tail), tail, sigma_rad, raw_rate, b


def main():
    t0 = time.monotonic()
    primary_records = {
        r["claim"]: r["enclosure"]
        for r in json.loads(GAP4_JSON.read_text())
    }
    part1_data = json.loads(PART1_JSON.read_text())
    ce2_part1_row = next(r for r in part1_data["results"] if r["claim_id"] == "Claim_4a_CE2_series_order88")
    ce2_series_iv = iv.mpf([
        mp.mpf(ce2_part1_row["details"]["D_raw_lo"]),
        mp.mpf(ce2_part1_row["details"]["D_raw_hi"]),
    ])

    results = []

    def log_check(claim_id, status, summary, details, overlaps_independent=True):
        row = {
            "claim_id": claim_id,
            "status": status,
            "overlaps_independent": overlaps_independent,
            "summary": summary,
            "details": details,
        }
        results.append(row)
        print(f"[{status}] {claim_id}: {summary}", flush=True)

    # ------------------------------------------------------------------
    # GATE G0_cheb_diff: pre-registered tolerance 1e-55
    # ------------------------------------------------------------------
    # Test polynomial q(z) = T_0 - 2 T_1 + (1/2) T_2 + (3/4) T_3 - (1/8) T_5
    # = (1 - 1/2) + (-2 - 9/4 - 5/8) z + 1 z^2 + (3 + 5/2) z^3 - 2 z^5
    # = 1/2 - 39/8 z + z^2 + 11/2 z^3 - 2 z^5
    c_test = [iv.mpf("0.5") * 2, iv.mpf(-2), HALF, iv.mpf("0.75"), ZERO, iv.mpf("-0.125")]
    ev_test = ChebPolyEvaluator(c_test, 6)

    def q_exact_scaled_derivs(z):
        # Monomial form of q(z) = 0.5 - 4.875 z + z^2 + 5.5 z^3 - 2 z^5
        a0 = HALF
        a1 = -iv.mpf(39) / iv.mpf(8)
        a2 = ONE
        a3 = iv.mpf(11) / iv.mpf(2)
        a5 = -iv.mpf(2)
        return [
            a0 + a1 * z + a2 * z ** 2 + a3 * z ** 3 + a5 * z ** 5,
            a1 + 2 * a2 * z + 3 * a3 * z ** 2 + 5 * a5 * z ** 4,
            a2 + 3 * a3 * z + 10 * a5 * z ** 3,
            a3 + 10 * a5 * z ** 2,
            5 * a5 * z,
            a5,
            ZERO,
        ]

    max_cheb_pt_w = ZERO
    for z_str in ["-0.75", "0.0", "0.3125"]:
        z_pt = iv.mpf(z_str)
        exact_d = q_exact_scaled_derivs(z_pt)
        for r in range(7):
            enc_r = ev_test.eval_scaled_deriv_on_interval(r, z_pt)
            diff_r = enc_r - exact_d[r]
            assert (0 in diff_r) and bool(enc_r.delta < iv.mpf("1e-55"))
            if bool(enc_r.delta > max_cheb_pt_w):
                max_cheb_pt_w = enc_r.delta
    # Also check non-degenerate interval inclusion on [0.2, 0.25]
    z_int = iv.mpf([mp.mpf("0.2"), mp.mpf("0.25")])
    for z_sample_str in ["0.2", "0.215", "0.25"]:
        ex_s = q_exact_scaled_derivs(iv.mpf(z_sample_str))
        for r in range(7):
            enc_int = ev_test.eval_scaled_deriv_on_interval(r, z_int)
            assert bool(enc_int.a <= ex_s[r].a) and bool(ex_s[r].b <= enc_int.b)

    log_check(
        "G0_cheb_diff",
        "PASS",
        f"Chebyshev downward differentiation + l1 MVT enclosure passed (max point width {mpmath.nstr(iv_hi_mpf(max_cheb_pt_w), 4)} < 1e-55)",
        {"max_point_width": mpmath.nstr(iv_hi_mpf(max_cheb_pt_w), 6)},
    )

    # ------------------------------------------------------------------
    # GATE G0_operator: pre-registered tolerance 1e-25 (alternating host benchmark)
    # ------------------------------------------------------------------
    s_b, d_b = iv.mpf("0.7"), iv.mpf("0.6")
    c_b = (ONE - s_b ** 2) / (ONE + iv.sqrt(d_b ** 2 + (ONE - d_b ** 2) * s_b ** 2))
    r_bench = enclose_operator_rate(s_b, c_b, d_b, iv.mpf(-1), n=40, cells=64, degree=20)
    assert (0 in r_bench) and bool(iv_abs_upper(r_bench) < iv.mpf("1e-25"))
    log_check(
        "G0_operator",
        "PASS",
        f"Exact alternating-host threshold benchmark passed: rate in {format_outward_pm(r_bench, 12, 3)} (< 1e-25)",
        {"rate_at_exact_alternating_threshold": format_outward_pm(r_bench, 12, 4)},
    )

    # ------------------------------------------------------------------
    # CLAIM 4b: CE2 independent exact-operator coboundary + cross-check with Part 1 series
    # ------------------------------------------------------------------
    ce2_op = enclose_operator_threshold(
        "0.9", "0.4", "-0.75", "-0.004482971848658596862157", n=40, cells=64, degree=20
    )
    assert intervals_overlap(ce2_op["D"], ce2_series_iv), "CE2 operator and series enclosures failed to overlap!"
    ov_ce2_op = intervals_overlap(
        ce2_op["D"], parse_arb_str_to_iv(primary_records["CE2 independent operator"]["D"])
    )
    assert ov_ce2_op
    log_check(
        "Claim_4b_CE2_independent_operator",
        "PASS",
        f"CE2 operator D = {format_outward_pm(ce2_op['D'])} overlaps both Part 1 order-88 series and primary Arb",
        {k: format_outward_pm(v) for k, v in ce2_op.items()},
        overlaps_independent=ov_ce2_op,
    )

    # ------------------------------------------------------------------
    # CLAIM 10: Moderate-s exact-operator coboundary enclosures (s=0.5 and s=0.7)
    # ------------------------------------------------------------------
    mod_05 = enclose_operator_threshold("0.5", "0.4", "0.7", "0.0582995785286115578", n=40, cells=64, degree=20)
    mod_07 = enclose_operator_threshold("0.7", "0.6", "0", "0.0456067551059600241", n=40, cells=64, degree=20)
    ov_mod = intervals_overlap(
        mod_05["D"], parse_arb_str_to_iv(primary_records["Moderate s=.5 delta=.4 phi=.7"]["D"])
    ) and intervals_overlap(
        mod_07["D"], parse_arb_str_to_iv(primary_records["Moderate s=.7 delta=.6 phi=0"]["D"])
    )
    assert ov_mod
    log_check(
        "Claim_10_Moderate_s_operator_enclosures",
        "PASS",
        f"Moderate-s coboundary certified: D(0.5,0.4,0.7) = {format_outward_pm(mod_05['D'])}, D(0.7,0.6,0) = {format_outward_pm(mod_07['D'])}",
        {
            "s=0.5_delta=0.4_phi=0.7": {k: format_outward_pm(v) for k, v in mod_05.items()},
            "s=0.7_delta=0.6_phi=0": {k: format_outward_pm(v) for k, v in mod_07.items()},
        },
        overlaps_independent=ov_mod,
    )

    # ------------------------------------------------------------------
    # CLAIM 11: Small-s exact-operator coboundary enclosures & sampled ordering
    # ------------------------------------------------------------------
    small_specs = [
        ("Small s=.1", "0.1", "0.4", "0.6", "0.0736141981986050458", 64, 128, 24),
        ("Small s=.03", "0.03", "0.4", "0.6", "0.07646264616186288", 96, 128, 24),
        ("Small s=.01", "0.01", "0.4", "0.6", "0.0781564790055211", 128, 128, 28),
        ("Small s=.003", "0.003", "0.4", "0.6", "0.07985126015331216", 160, 128, 28),
    ]
    small_Ds = []
    small_details = {}
    for cname, s_s, d_s, p_s, g_s, n_deg, n_cells, t_deg in small_specs:
        out_sm = enclose_operator_threshold(s_s, d_s, p_s, g_s, n=n_deg, cells=n_cells, degree=t_deg)
        assert intervals_overlap(out_sm["D"], parse_arb_str_to_iv(primary_records[cname]["D"]))
        small_Ds.append(out_sm["D"])
        small_details[cname] = {k: format_outward_pm(v) for k, v in out_sm.items()}

    ell_04 = -iv.log(ONE - iv.mpf("0.4") ** 2) * HALF
    increments = [small_Ds[k + 1] - small_Ds[k] for k in range(3)]
    assert all(bool(inc > 0) for inc in increments)
    assert bool(small_Ds[-1] < ell_04)
    for k in range(3):
        assert intervals_overlap(
            increments[k],
            parse_arb_str_to_iv(primary_records["Small-s sampled ordering"][f"increment_{k+1}"]),
        )
    small_details["ordering"] = {
        "ell": format_outward_pm(ell_04),
        **{f"increment_{k+1}": format_outward_pm(increments[k], 18, 4) for k in range(3)},
    }
    log_check(
        "Claim_11_Small_s_operator_and_ordering",
        "PASS",
        f"All 4 small-s enclosures (down to s=0.003: D = {format_outward_pm(small_Ds[-1], 18, 4)} < ell) and strict ordering increments > 0 certified",
        small_details,
        overlaps_independent=True,
    )

    # ------------------------------------------------------------------
    # CLAIM 12: Finite-state Gaussian Markov chains (n = 12 and n = 20)
    # ------------------------------------------------------------------
    phi_g = iv.mpf("-0.35")
    sig_g = iv.mpf("1e-5")
    st_g = s_star_ar1(phi_g)
    pts_g = [st_g - iv.mpf("0.03"), st_g, st_g + iv.mpf("0.03")]
    g_order = 12
    g_results = {}
    g_details = {}

    for n_nodes in [12, 20]:
        chain = build_gaussian_finite_chain_iv(phi_g, n_nodes, g_order)
        xs_c, P_c, pi_c, _ = chain
        mean_c = sum((pi_c[i] * xs_c[i] for i in range(n_nodes)), ZERO)
        var_c = sum((pi_c[i] * (xs_c[i] ** 2) for i in range(n_nodes)), ZERO) - mean_c ** 2
        lag1_c = (
            sum(
                (
                    pi_c[i]
                    * xs_c[i]
                    * sum((P_c[i][j] * xs_c[j] for j in range(n_nodes)), ZERO)
                    for i in range(n_nodes)
                ),
                ZERO,
            )
            / var_c
        )
        m_cname = f"Gaussian finite chain n={n_nodes} moments"
        assert intervals_overlap(var_c, parse_arb_str_to_iv(primary_records[m_cname]["variance"]))
        assert intervals_overlap(lag1_c, parse_arb_str_to_iv(primary_records[m_cname]["lag1"]))
        g_details[m_cname] = {
            "variance": format_outward_pm(var_c),
            "lag1": format_outward_pm(lag1_c),
        }

        g_results[n_nodes] = []
        for idx_pt, s_pt in enumerate(pts_g, start=1):
            D0_g = (sig_g ** 2) * F_ar1(s_pt, phi_g)
            for _ in range(2):
                _, _, _, raw_r8, b_8 = finite_gaussian_rate_iv(
                    s_pt, (ONE - s_pt) * iv.exp(D0_g), sig_g, chain, 8
                )
                D0_g = D0_g - raw_r8 * (ONE + b_8) / b_8
            _, tail8, _, raw8_fin, _ = finite_gaussian_rate_iv(
                s_pt, (ONE - s_pt) * iv.exp(D0_g), sig_g, chain, 8
            )
            assert bool(iv_abs_upper(raw8_fin) < iv.mpf("4e-46"))
            assert bool(tail8 < iv.mpf("1e-25"))

            c_cand = (ONE - s_pt) * iv.exp(D0_g)
            rate12, tail12, sig_rad, _, _ = finite_gaussian_rate_iv(s_pt, c_cand, sig_g, chain, g_order)
            raw_hosts = [iv.exp(sig_g * x) for x in xs_c]
            norm_h = sum((pi_c[j] * raw_hosts[j] for j in range(n_nodes)), ZERO)
            min_h = min(((h / norm_h).a for h in raw_hosts), key=lambda v: iv_lo_mpf(v))
            c_min_g = (ONE - s_pt) * iv.exp(D0_g - iv.mpf("0.01"))
            beta_g = c_min_g * min_h / (s_pt ** 2 + c_min_g * min_h)
            err_D = iv.mpf(3) * iv_abs_upper(rate12) / beta_g
            assert bool(err_D < iv.mpf("0.01"))
            D_over_s2 = iv_ball(D0_g, err_D) / (sig_g ** 2)
            g_results[n_nodes].append(D_over_s2)

            p_cname = f"Gaussian finite chain n={n_nodes} point={idx_pt}"
            assert intervals_overlap(
                D_over_s2, parse_arb_str_to_iv(primary_records[p_cname]["D_over_sigma2"])
            )
            g_details[p_cname] = {
                "D_over_sigma2": format_outward_pm(D_over_s2),
                "rate_tail": format_outward_pm(tail12, 16, 4),
                "sigma_radius": format_outward_pm(sig_rad),
                "root_error_D": format_outward_pm(err_D, 16, 4),
            }

    for idx_pt in range(3):
        diff_20_12 = g_results[20][idx_pt] - g_results[12][idx_pt]
        assert bool(abs(diff_20_12) < iv.mpf("2.4e-15"))
        r_cname = f"Gaussian finite-chain refinement point={idx_pt+1}"
        assert intervals_overlap(
            diff_20_12, parse_arb_str_to_iv(primary_records[r_cname]["normalized_difference"])
        )
        g_details[r_cname] = {"normalized_difference": format_outward_pm(diff_20_12, 16, 4)}

    sec_left = (g_results[20][1] - g_results[20][0]) / iv.mpf("0.03")
    sec_right = (g_results[20][2] - g_results[20][1]) / iv.mpf("0.03")
    assert bool(sec_left > 0) and bool(sec_right < 0)
    assert intervals_overlap(
        sec_left, parse_arb_str_to_iv(primary_records["Gaussian finite-chain secants"]["left"])
    )
    assert intervals_overlap(
        sec_right, parse_arb_str_to_iv(primary_records["Gaussian finite-chain secants"]["right"])
    )
    g_details["secants_n20"] = {
        "left": format_outward_pm(sec_left, 22, 4),
        "right": format_outward_pm(sec_right, 22, 4),
    }
    log_check(
        "Claim_12_Finite_state_Gaussian_chains_n12_n20",
        "PASS",
        f"Finite-state Gaussian chains (n=12, 20) certified: n=20 secants left = {format_outward_pm(sec_left, 16, 3)} > 0, right = {format_outward_pm(sec_right, 16, 3)} < 0 (finite chains only; continuous Gaussian rightly OPEN)",
        g_details,
        overlaps_independent=True,
    )

    # ------------------------------------------------------------------
    # CLAIM 13: Saved-grid tallies in memB_tables.csv (all 4,320 rows)
    # ------------------------------------------------------------------
    groups = defaultdict(list)
    row_count = 0
    with MEMB_CSV.open() as f:
        for row in csv.DictReader(f):
            row_count += 1
            groups[(row["model"], row["amplitude"], row["phi_input"], row["ring"])].append(row)
    assert row_count == 4320 and len(groups) == 144

    primary_counts = defaultdict(int)
    matched_counts = defaultdict(int)
    min_margin = None
    for key, rows in groups.items():
        rows.sort(key=lambda r: iv_lo_mpf(iv.mpf(r["tau"])))
        up_flags = []
        down_flags = []
        for row in rows[:-1]:
            diff_iv = iv.mpf(row["next_D_difference"])
            se_iv = iv.mpf(row["next_difference_SE"])
            is_up = bool(diff_iv - iv.mpf(4) * se_iv > 0)
            is_down = bool(diff_iv + iv.mpf(4) * se_iv < 0)
            up_flags.append(is_up)
            down_flags.append(is_down)
            # Distance of |diff| from 4*se decision boundary
            margin_iv = abs(abs(diff_iv) - iv.mpf(4) * se_iv)
            assert bool(margin_iv > 0), f"Ambiguous 4SE decision boundary on {key}"
            if min_margin is None or bool(margin_iv.a < min_margin.a):
                min_margin = margin_iv
        status = (
            "decrease"
            if all(down_flags)
            else ("nonmonotone" if (any(up_flags) and any(down_flags)) else "unresolved")
        )
        assert status == rows[0]["status"], f"Status mismatch on {key}: {status} vs {rows[0]['status']}"
        if key[0] == "gamma_matched":
            matched_counts[(key[3], status)] += 1
        else:
            primary_counts[(key[3], status)] += 1

    assert primary_counts[("2", "decrease")] == 112
    assert matched_counts[("2", "decrease")] == 8
    assert primary_counts[("3", "nonmonotone")] == 15
    assert primary_counts[("3", "decrease")] == 9
    assert intervals_overlap(
        min_margin,
        parse_arb_str_to_iv(primary_records["Manuscript sampled-grid tallies"]["minimum_distance_from_4SE_decision"]),
    )
    log_check(
        "Claim_13_memB_saved_grid_tallies",
        "PASS",
        f"All 4,320 rows (144 curves) verified: 112/112 primary 2-stage decrease (+8/8 matched gamma), 15/24 3-stage nonmonotone, min 4SE margin = {format_outward_pm(min_margin)}",
        {
            "rows_checked": row_count,
            "curves_checked": len(groups),
            "primary_2stage_decrease": primary_counts[("2", "decrease")],
            "matched_gamma_2stage_decrease": matched_counts[("2", "decrease")],
            "primary_3stage_nonmonotone": primary_counts[("3", "nonmonotone")],
            "primary_3stage_decrease": primary_counts[("3", "decrease")],
            "minimum_distance_from_4SE_decision": format_outward_pm(min_margin),
        },
        overlaps_independent=True,
    )

    elapsed = round(time.monotonic() - t0, 2)
    payload = {
        "script": "verify_gap4_part2_operator_gauss.py",
        "mpmath_version": mpmath.__version__,
        "iv_dps": iv.dps,
        "elapsed_seconds": elapsed,
        "all_passed": all(r["status"] == "PASS" for r in results),
        "results": results,
    }
    (HERE / "verify_gap4_part2_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Part 2 completed: {len(results)} check groups PASSED in {elapsed}s", flush=True)


if __name__ == "__main__":
    main()
