#!/usr/bin/env python3
"""Independent referee verification of a13i (r52-a13i-check).

Tests every claim of `a13i/a13i.md` with standalone code
(zero imports from `a13i/` or any other repository file):
  1. Pre-registered known-result gates (study rules Rule 7) in both 220-digit
     exact analytic Perron arithmetic and Chebyshev modal Poisson arithmetic.
  2. Section 2 regularity, exact finite-product Hessian convexity, edge-weight
     occupation derivatives, projective contraction (including rank-1 and h=0
     transpose edge cases), adversarial opposite-sign log-ratio perturbations,
     and O(1/n) convergence of finite-product gradients to infinite-horizon R_x, R_y.
  3. Sections 3-5 exact delta/eta/zeta recursions, L^4 / L^2 / L^(4/3) norms,
     cubic polynomial P(e,c) and R4*e^4 remainder, weighted absolute coefficient
     bounds B0 and C0, convex log-secant derivative enclosures, and the explicit
     cutoff |s^2 D_s + A| <= C_*/sqrt(s) across 8 periodic records spanning edge
     cases (V=0, alternating outage, multi-zero reversal [3,0,0], extreme outage
     q0=63/64, exact zero-containing A=0 boundary, tiny-|A| near-reversal, a->1+,
     a>>1, s->1+).
  4. Unbounded nonnegative host laws with zeros under E[h^4] < infinity across
     10 (law, a) cells x 4 stays s in {100, 500, 2500, 12500}, including
     borderline Pareto(alpha=4.05) (no 4.05-th moment), Pareto(4.5), singular-at-0
     Gamma(0.25) (infinite positive inverse moment), high-outage lognormal
     (q0=0.85, no MGF), and switching reversals, solved via Chebyshev modal
     T_k / U_{k-1} collocation + Gauss-Jacobi / Laguerre / Hermite quadrature
     with exact linear-system operator differentiation (cross-checked against
     centered finite differences).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_genlaguerre, roots_hermitenorm, roots_jacobi

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 220

ASSERTIONS = 0
FAILURES: list[dict] = []
SUMMARY: dict[str, float] = {}


def record_max(key: str, val) -> None:
    v = float(val)
    if key not in SUMMARY or v > SUMMARY[key]:
        SUMMARY[key] = v


def record_min(key: str, val) -> None:
    v = float(val)
    if key not in SUMMARY or v < SUMMARY[key]:
        SUMMARY[key] = v


def verify(label: str, cond: bool, detail: str = "") -> None:
    global ASSERTIONS
    ASSERTIONS += 1
    if not cond:
        FAILURES.append({"label": label, "detail": detail})
        print(f"FAIL [{label}]: {detail}", flush=True)


def verify_le(label: str, value, bound, metric_key: str | None = None) -> None:
    if metric_key is not None:
        record_max(metric_key, value)
    verify(
        label,
        value <= bound,
        f"value={mp.nstr(mp.mpf(value), 16)} > bound={mp.nstr(mp.mpf(bound), 16)}",
    )


# ---------------------------------------------------------------------------
# 1. 220-digit exact analytic matrix-product engine (no numerical mp.diff)
# ---------------------------------------------------------------------------


def step_matrix(h, a, q, g):
    b = mp.mpf(1) / a
    p = mp.mpf(1) - b
    return mp.matrix([[p, g], [b * h, q]])


def cycle_product_and_jet(seq, a, q, g):
    """Return (Pi, Pi_q, Pi_g, Pi_qq, Pi_gg, Pi_qg) by exact product rule."""
    b = mp.mpf(1) / a
    p = mp.mpf(1) - b
    Pi = mp.eye(2)
    Pq = mp.zeros(2)
    Pg = mp.zeros(2)
    Pqq = mp.zeros(2)
    Pgg = mp.zeros(2)
    Pqg = mp.zeros(2)
    Mq = mp.matrix([[0, 0], [0, 1]])
    Mg = mp.matrix([[0, 1], [0, 0]])
    for h in seq:
        M = mp.matrix([[p, g], [b * h, q]])
        nPqq = 2 * Mq * Pq + M * Pqq
        nPgg = 2 * Mg * Pg + M * Pgg
        nPqg = Mq * Pg + Mg * Pq + M * Pqg
        nPq = Mq * Pi + M * Pq
        nPg = Mg * Pi + M * Pg
        nPi = M * Pi
        Pi, Pq, Pg, Pqq, Pgg, Pqg = nPi, nPq, nPg, nPqq, nPgg, nPqg
    return Pi, Pq, Pg, Pqq, Pgg, Pqg


def perron_and_partials(seq, a, q, g):
    """Exact Perron eigenvalue lambda(q,g) and exact (lambda_q, lambda_g)."""
    Pi, Pq, Pg, _, _, _ = cycle_product_and_jet(seq, a, q, g)
    tr = Pi[0, 0] + Pi[1, 1]
    diff = Pi[0, 0] - Pi[1, 1]
    disc = diff * diff + 4 * Pi[0, 1] * Pi[1, 0]
    sdisc = mp.sqrt(disc)
    lam = (tr + sdisc) / 2

    def dlam(dP):
        dtr = dP[0, 0] + dP[1, 1]
        ddiff = dP[0, 0] - dP[1, 1]
        ddisc = 2 * diff * ddiff + 4 * (dP[0, 1] * Pi[1, 0] + Pi[0, 1] * dP[1, 0])
        return (dtr + ddisc / (2 * sdisc)) / 2

    return Pi, lam, dlam(Pq), dlam(Pg)


def exact_R_and_grad(seq, a, x, y):
    """Exact R(x,y), R_x(x,y), R_y(x,y) for x=log q, y=log g."""
    q = mp.exp(x)
    g = mp.exp(y)
    n = len(seq)
    _, lam, lq, lg = perron_and_partials(seq, a, q, g)
    R = mp.log(lam) / n
    Rx = q * lq / (n * lam)
    Ry = g * lg / (n * lam)
    return R, Rx, Ry


def exact_r_and_partials(seq, a, e, D):
    """Exact r(e, exp(D)), r_e, r_D at fixed (e, D)."""
    q = mp.mpf(1) - e
    c = mp.exp(D)
    g = e * c
    n = len(seq)
    _, lam, lq, lg = perron_and_partials(seq, a, q, g)
    r = mp.log(lam) / n
    Rx = q * lq / (n * lam)
    Ry = g * lg / (n * lam)
    re = -Rx / q + Ry / e
    rD = Ry
    return r, re, rD, Rx, Ry


def solve_threshold_mp(seq, a, s):
    """Solve r(1/s, exp(D)) = 0 by exact Newton from bracket and return (D, D_e, D_s)."""
    e = mp.mpf(1) / s
    # Find bracket first
    lo = mp.mpf("-2")
    hi = mp.mpf("2")
    while exact_r_and_partials(seq, a, e, lo)[0] > 0:
        lo *= 2
    while exact_r_and_partials(seq, a, e, hi)[0] < 0:
        hi *= 2
    # Bisection to 1e-25, then exact safeguarded Newton to 1e-205
    for _ in range(95):
        mid = (lo + hi) / 2
        if exact_r_and_partials(seq, a, e, mid)[0] <= 0:
            lo = mid
        else:
            hi = mid
    D = (lo + hi) / 2
    for _ in range(12):
        r, _, rD, _, _ = exact_r_and_partials(seq, a, e, D)
        step = r / rD
        D -= step
        if abs(step) < mp.mpf("1e-208"):
            break
    r, re, rD, Rx, Ry = exact_r_and_partials(seq, a, e, D)
    verify_le("newton_residual", abs(r), mp.mpf("1e-200"))
    De = -re / rD
    Ds = -De / (s * s)
    return D, De, Ds, Rx, Ry


def a13i_constants(a, H):
    """Explicit constants from equations (6), (12), (15), (21) of a13i.md."""
    b = mp.mpf(1) / a
    T = 2 * H + 1
    Q = 4 * H + 1
    U = T / b
    L = 2 * Q / b
    N = L * (Q + 2 * H) / b
    J = U * Q / b
    Z = (Q * N + 2 * H * L**2) / b
    B0 = 4 * H**2 / b + 2 / b + 2 * H**2 + 3
    C0 = 2 * H * J + 2 * H * T * U + T**3 / 3
    R4 = (
        2 * H * Z
        + 2 * H * T * N
        + 2 * H**2 * L**2
        + (2 * H * L / 3) * (Q**2 + Q * T + T**2)
        + 4 * Q**4
    )
    K = B0 + C0 + R4
    e0 = min(b / 4, mp.mpf(1) / 16, mp.mpf(1) / (8 * (K + 1)))
    Mx = 1 + 4 * B0 + 9 * C0
    My = 2 + 4 * B0 + 9 * C0
    Ex = Mx / 2 + 3 * R4
    Ey = My / 2 + 2 * R4
    E0 = 2 * Ex + Ey
    J0 = K + 2 * B0 + 3 * C0 + Ey
    N0 = 4 * B0 * K + 2 * C0 + R4 + E0
    estar = min(e0, mp.mpf(1) / (2 * J0))
    Cstar = 2 * (N0 + B0 * J0)
    return {
        "b": b, "T": T, "Q": Q, "U": U, "L": L, "N": N, "J": J, "Z": Z,
        "B0": B0, "C0": C0, "R4": R4, "K": K, "e0": e0,
        "Mx": Mx, "My": My, "Ex": Ex, "Ey": Ey, "E0": E0,
        "J0": J0, "N0": N0, "estar": estar, "Cstar": Cstar,
    }


def periodic_A_and_poly(seq, a, c):
    """Compute exact A, B(c), C(c), u_t(c), w_t(c) on a periodic record."""
    n = len(seq)
    b = mp.mpf(1) / a
    p = mp.mpf(1) - b

    def solve_filter(rhs):
        # z_{t+1} = p z_t + rhs_t  (periodic of period n)
        z0 = mp.fsum(p ** (n - 1 - j) * rhs[j] for j in range(n)) / (1 - p**n)
        out = [mp.mpf(0)] * n
        cur = z0
        for t in range(n):
            out[t] = cur
            cur = p * cur + rhs[t]
        return out

    k = [c * h - 1 for h in seq]
    u = solve_filter([-kt for kt in k])
    w = solve_filter(
        [-(c * seq[t] * u[t] + k[t] * u[(t + 1) % n]) for t in range(n)]
    )
    B = mp.fsum(k[t] ** 2 / 2 - c * seq[t] * u[t] for t in range(n)) / n
    C = mp.fsum(
        c * seq[t] * w[t] - k[t] * c * seq[t] * u[t] + k[t] ** 3 / 3
        for t in range(n)
    ) / n

    cov = [
        mp.fsum((seq[t] - 1) * (seq[(t - lag) % n] - 1) for t in range(n)) / n
        for lag in range(1, n + 1)
    ]
    var0 = mp.fsum((h - 1) ** 2 for h in seq) / n
    A = var0 / 2 + mp.fsum(p ** (lag - 1) * cov[lag - 1] for lag in range(1, n + 1)) / (1 - p**n)
    return A, B, C, u, w


def construct_zero_host_A0_record(a):
    """Construct exact h = [0, u_*, 3 - u_*, 1] with h >= 0, mean 1, Var > 0, and A(a) = 0."""
    # Letseq(u) = [0, u, 3 - u, 1], mean = 1 for all u in [0, 3].
    # At u = 0: [0, 0, 3, 1]; at u = 1.5: [0, 1.5, 1.5, 1]. Check if A changes sign!
    # Better: generalize to [0, u, 4 - 2u, u] or [u, 0, 4 - u, 0]?
    # Wait: in verification/a13-check, [0, u_*, 3 - u_*, 1] at a=2 had u_* ~ 0.29418? Wait: let's check!
    # Wait: for [0, u, 3-u, 1], cov(-lag) for lag=1,2,3,4:
    # Let's check A(u) at u=0 vs u=1.5, or try [0, u, 0, 4 - u]? For [0, 2, 0, 2], A < 0!
    # For [0, 4, 0, 0], A > 0!
    # YES!! On h(u) = [0, u, 0, 4 - u] for u in (0, 2):
    # At u = 2: [0, 2, 0, 2] = alternating (2, 0), where A = -1/[2(2a-1)] < 0 for EVERY a > 1!
    # At u = 0: [0, 0, 0, 4], where C_0 = 3, and for a = 2 (p=1/2): A > 0!
    # Let's solve A(u) = 0 on u in (0, 2) for [0, u, 0, 4 - u] (which has TWO zeros and mean 1)!
    def A_of_u(u):
        seq = [mp.mpf(0), u, mp.mpf(0), mp.mpf(4) - u]
        return periodic_A_and_poly(seq, a, mp.mpf(1))[0]

    lo, hi = mp.mpf("0.01"), mp.mpf("1.99")
    flo, fhi = A_of_u(lo), A_of_u(hi)
    assert flo > 0 and fhi < 0, f"Unexpected signs: {flo}, {fhi}"
    for _ in range(700):
        mid = (lo + hi) / 2
        if A_of_u(mid) > 0:
            lo = mid
        else:
            hi = mid
    ustar = (lo + hi) / 2
    return [mp.mpf(0), ustar, mp.mpf(0), mp.mpf(4) - ustar]


# ---------------------------------------------------------------------------
# 2. Chebyshev modal T_k / U_{k-1} Poisson solver with exact operator derivatives
# ---------------------------------------------------------------------------


def cheb_TU_and_deriv(xi: np.ndarray, dy_dtheta: np.ndarray, m: int):
    """Evaluate T_k(xi) and d/dtheta T_k(2*y - 1) = 2 * dT_k/dxi * dy_dtheta for k=0..m-1.

    Uses exact three-term recurrences for T_k and Chebyshev second-kind U_{k-1}
    (since T_k'(xi) = k * U_{k-1}(xi)), avoiding 0/0 at xi = +-1.
    """
    shape = xi.shape
    T = np.empty(shape + (m,), dtype=np.float64)
    dT = np.empty(shape + (m,), dtype=np.float64)
    T[..., 0] = 1.0
    dT[..., 0] = 0.0
    if m == 1:
        return T, dT
    T[..., 1] = xi
    U_prev2 = np.ones(shape, dtype=np.float64)  # U_0
    dT[..., 1] = 2.0 * U_prev2 * dy_dtheta
    if m == 2:
        return T, dT
    U_prev1 = 2.0 * xi  # U_1
    for k in range(2, m):
        T[..., k] = 2.0 * xi * T[..., k - 1] - T[..., k - 2]
        dT[..., k] = 2.0 * k * U_prev1 * dy_dtheta
        U_curr = 2.0 * xi * U_prev1 - U_prev2
        U_prev2, U_prev1 = U_prev1, U_curr
    return T, dT


class MarkovOutageLaw:
    """Two-state Markov outage process Z_t in {0,1} with conditionally i.i.d. positive H_+."""

    def __init__(self, name: str, q0: float, phi: float, dist: str, param: float):
        self.name = name
        self.q0 = float(q0)
        self.phi = float(phi)
        self.dist = dist
        self.param = float(param)
        self.P = np.array(
            [
                [q0 + (1.0 - q0) * phi, (1.0 - q0) * (1.0 - phi)],
                [q0 * (1.0 - phi), 1.0 - q0 + q0 * phi],
            ],
            dtype=np.float64,
        )

    def pos_mean(self) -> float:
        return 1.0 / (1.0 - self.q0)

    def pos_moment(self, k: int) -> float:
        mu = self.pos_mean()
        if self.dist == "const":
            return mu**k
        if self.dist == "gamma":
            shape = self.param
            prod = 1.0
            for j in range(k):
                prod *= shape + j
            return (mu / shape) ** k * prod
        if self.dist == "lognormal":
            sig = self.param
            return (mu**k) * math.exp(0.5 * (k * k - k) * sig * sig)
        if self.dist == "pareto":
            alpha = self.param
            x0 = mu * (alpha - 1.0) / alpha
            return alpha * (x0**k) / (alpha - k)
        raise ValueError(self.dist)

    def H4_norm(self) -> float:
        return ((1.0 - self.q0) * self.pos_moment(4)) ** 0.25

    def exact_A(self, a: float) -> float:
        p = 1.0 - 1.0 / a
        var_h = (1.0 - self.q0) * self.pos_moment(2) - 1.0
        if self.q0 == 0.0:
            return 0.5 * var_h
        cov_sum = (self.q0 / (1.0 - self.q0)) * self.phi / (1.0 - p * self.phi)
        return 0.5 * var_h + cov_sum

    def quadrature(self, nq: int):
        """Return positive quadrature representation for E_{H_+}[...]."""
        mu = self.pos_mean()
        if self.dist == "const":
            return "direct", np.array([mu]), np.array([1.0]), None
        if self.dist == "gamma":
            shape = self.param
            x, w = roots_genlaguerre(nq, shape - 1.0)
            return "direct", x * (mu / shape), w / math.gamma(shape), None
        if self.dist == "lognormal":
            sig = self.param
            z, w = roots_hermitenorm(nq)
            h = mu * np.exp(sig * z - 0.5 * sig * sig)
            return "direct", h, w / math.sqrt(2.0 * math.pi), None
        if self.dist == "pareto":
            alpha = self.param
            x0 = mu * (alpha - 1.0) / alpha
            # Exact Gauss-Jacobi on t in (-1, 1) with weight (1-t)^0 * (1+t)^(2*alpha - 1)
            # Mapped to w = (1 + t)/2 in (0, 1): integral_0^1 g(w) * 2*alpha*w^(2*alpha-1) dw
            t, wt = roots_jacobi(nq, 0.0, 2.0 * alpha - 1.0)
            w = 0.5 * (1.0 + t)
            weights = wt * (2.0 * alpha) / (2.0 ** (2.0 * alpha))
            return "pareto_w", w, weights, (alpha, x0)
        raise ValueError(self.dist)


def _phi1_and_deriv(x: np.ndarray):
    """Cancellation-free log1p(x)/x and its derivative d/dx [log1p(x)/x]."""
    small = np.abs(x) < 1e-3
    xs = np.where(small, x, 0.0)
    xb = np.where(small, 1.0, x)  # dummy nonzero away from small
    phi_s = 1.0 - xs * (
        0.5 - xs * (1.0 / 3.0 - xs * (0.25 - xs * (0.2 - xs * (1.0 / 6.0 - xs / 7.0))))
    )
    dphi_s = -0.5 + xs * (
        2.0 / 3.0 - xs * (0.75 - xs * (0.8 - xs * (5.0 / 6.0 - 6.0 * xs / 7.0)))
    )
    l1p = np.log1p(xb)
    phi_b = l1p / xb
    dphi_b = (xb / (1.0 + xb) - l1p) / (xb * xb)
    return np.where(small, phi_s, phi_b), np.where(small, dphi_s, dphi_b)


def modal_poisson_growth_and_partials(
    law: MarkovOutageLaw, a: float, e: float, D: float, m: int, nq: int
):
    """Compute rescaled (rtilde = r/e, rtilde_e, rtilde_D) without 1/e cancellation."""
    b = 1.0 / a
    p = 1.0 - b
    c = math.exp(D)
    gamma = b * e * c / (b - e)
    gamma_e = (b * b * c) / ((b - e) ** 2)
    gamma_D = gamma

    idx = np.arange(m, dtype=np.float64)
    xi_nodes = -np.cos(math.pi * idx / (m - 1))
    y = 0.5 * (1.0 + xi_nodes)
    y2 = y * y

    # Collocation basis matrix V_{j,k} = T_k(xi_j)
    V, _ = cheb_TU_and_deriv(xi_nodes, np.zeros_like(xi_nodes), m)

    # Numerator N_i(e) = p*y_i^2 + b - e > 0
    N = p * y2 + (b - e)

    # Outage branch z = 0 (h = 0, W = -1, x = -e): rtilde_0 = -phi1(-e)
    u0 = N / (1.0 - e)
    y0_next = np.sqrt(u0)
    du0_de = -p * (1.0 - y2) / ((1.0 - e) ** 2)
    dy0_de = 0.5 * du0_de / y0_next
    T0, dT0_de = cheb_TU_and_deriv(2.0 * y0_next - 1.0, dy0_de, m)
    dT0_dD = np.zeros_like(T0)

    phi0, dphi0 = _phi1_and_deriv(np.array([-e]))
    rew0 = np.full(m, -float(phi0[0]), dtype=np.float64)
    drew0_de = np.full(m, float(dphi0[0]), dtype=np.float64)
    drew0_dD = np.zeros(m, dtype=np.float64)

    # Positive branch z = 1 (h = H_+)
    mode, qnodes, qweights, extra = law.quadrature(nq)
    if mode == "direct":
        h = qnodes[None, :]  # (1, nq)
        Ncol = N[:, None]
        y2col = y2[:, None]
        W = (b * c / (b - e)) * y2col * h - 1.0
        We = (b * c / ((b - e) ** 2)) * y2col * h
        WD = W + 1.0
        x = e * W
        denom = 1.0 + x
        u1 = Ncol / denom
        y1_next = np.sqrt(u1)
        # d(N/denom)/de = [-p*(1 - y^2) - Ncol * gamma_e * y^2 * h - e*(...)] -> exact without O(1) cancel:
        # Ncol = p*y^2 + b - e, denom = 1 - e + gamma*y^2*h =>
        # dN/de * denom - N * d(denom)/de = -(1 - e + gamma*y^2*h) - (p*y^2 + b - e)*(-1 + gamma_e*y^2*h)
        # = -p*(1 - y^2) - (gamma + (p*y^2 + b - e)*gamma_e)*y^2*h
        num_du1_de = -p * (1.0 - y2col) - (gamma + Ncol * gamma_e) * y2col * h
        du1_de = num_du1_de / (denom * denom)
        du1_dD = -(Ncol * gamma_D * y2col * h) / (denom * denom)
        dy1_de = 0.5 * du1_de / y1_next
        dy1_dD = 0.5 * du1_dD / y1_next

        xi1 = 2.0 * y1_next - 1.0
        T1_q, dT1_de_q = cheb_TU_and_deriv(xi1, dy1_de, m)
        _, dT1_dD_q = cheb_TU_and_deriv(xi1, dy1_dD, m)
        T1 = np.einsum("iqk,q->ik", T1_q, qweights)
        dT1_de = np.einsum("iqk,q->ik", dT1_de_q, qweights)
        dT1_dD = np.einsum("iqk,q->ik", dT1_dD_q, qweights)

        phi1, dphi1 = _phi1_and_deriv(x)
        rew1 = (W * phi1) @ qweights
        drew1_de = (We / denom + (W * W) * dphi1) @ qweights
        drew1_dD = (WD / denom) @ qweights
    else:
        alpha, x0 = extra
        w = qnodes[None, :]  # (1, nq) in (0, 1)
        w2 = w * w
        T1 = np.empty((m, m), dtype=np.float64)
        dT1_de = np.empty((m, m), dtype=np.float64)
        dT1_dD = np.empty((m, m), dtype=np.float64)
        rew1 = np.empty(m, dtype=np.float64)
        drew1_de = np.empty(m, dtype=np.float64)
        drew1_dD = np.empty(m, dtype=np.float64)

        T1[0] = T0[0]
        dT1_de[0] = dT0_de[0]
        dT1_dD[0] = 0.0
        rew1[0] = rew0[0]
        drew1_de[0] = drew0_de[0]
        drew1_dD[0] = 0.0

        Np = N[1:, None]
        y2p = y2[1:, None]
        Dtilde = (1.0 - e) * w2 + gamma * y2p * x0
        rad = np.sqrt(Np / Dtilde)
        y1p = w * rad
        num_drad = -p * (1.0 - y2p) * w2 - (gamma + Np * gamma_e) * y2p * x0
        drad_de = 0.5 / rad * num_drad / (Dtilde * Dtilde)
        drad_dD = 0.5 / rad * (-(Np * gamma_D * y2p * x0)) / (Dtilde * Dtilde)
        dy1p_de = w * drad_de
        dy1p_dD = w * drad_dD

        xi1p = 2.0 * y1p - 1.0
        T1p_q, dT1p_de_q = cheb_TU_and_deriv(xi1p, dy1p_de, m)
        _, dT1p_dD_q = cheb_TU_and_deriv(xi1p, dy1p_dD, m)
        T1[1:] = np.einsum("iqk,q->ik", T1p_q, qweights)
        dT1_de[1:] = np.einsum("iqk,q->ik", dT1p_de_q, qweights)
        dT1_dD[1:] = np.einsum("iqk,q->ik", dT1p_dD_q, qweights)

        hp = x0 / w2
        Wp = (b * c / (b - e)) * y2p * hp - 1.0
        Wep = (b * c / ((b - e) ** 2)) * y2p * hp
        WDp = Wp + 1.0
        xp = e * Wp
        denomp = 1.0 + xp
        phi1p, dphi1p = _phi1_and_deriv(xp)
        rew1[1:] = (Wp * phi1p) @ qweights
        drew1_de[1:] = (Wep / denomp + (Wp * Wp) * dphi1p) @ qweights
        drew1_dD[1:] = (WDp / denomp) @ qweights

    P = law.P
    Zblock = np.zeros((m, m), dtype=np.float64)
    Vblock = np.block([[V, Zblock], [Zblock, V]])
    K = np.block([[P[0, 0] * T0, P[0, 1] * T0], [P[1, 0] * T1, P[1, 1] * T1]])
    Ke = np.block(
        [[P[0, 0] * dT0_de, P[0, 1] * dT0_de], [P[1, 0] * dT1_de, P[1, 1] * dT1_de]]
    )
    KD = np.block(
        [[P[0, 0] * dT0_dD, P[0, 1] * dT0_dD], [P[1, 0] * dT1_dD, P[1, 1] * dT1_dD]]
    )

    # Gauge: fix modal constant alpha_{0,0} = 0; column 0 carries r * 1_{2m}
    Msys = Vblock - K
    Msys[:, 0] = 1.0
    rhs = np.concatenate([rew0, rew1])
    sol = np.linalg.solve(Msys, rhs)
    r = float(sol[0])
    alpha_vec = sol.copy()
    alpha_vec[0] = 0.0

    rhs_e = np.concatenate([drew0_de, drew1_de]) + Ke @ alpha_vec
    rhs_D = np.concatenate([drew0_dD, drew1_dD]) + KD @ alpha_vec
    sol_e = np.linalg.solve(Msys, rhs_e)
    sol_D = np.linalg.solve(Msys, rhs_D)
    return r, float(sol_e[0]), float(sol_D[0])


def solve_modal_threshold(law: MarkovOutageLaw, a: float, e: float, m: int, nq: int):
    f = lambda D: modal_poisson_growth_and_partials(law, a, e, D, m, nq)[0]
    D = brentq(f, -0.35, 0.35, xtol=1e-15, rtol=1e-15)
    # Polish with one exact Newton step
    r, re, rD = modal_poisson_growth_and_partials(law, a, e, D, m, nq)
    D -= r / rD
    r, re, rD = modal_poisson_growth_and_partials(law, a, e, D, m, nq)
    De = -re / rD
    return D, De, r


# ---------------------------------------------------------------------------
# 3. Test suites
# ---------------------------------------------------------------------------


def run_known_result_gates() -> None:
    print("1. Pre-registered known-result gates (220-digit Perron & modal Chebyshev)", flush=True)
    # Gate G1: constant host h = [1] (zero-variance edge case V = 0, A = 0)
    for a_str in ["1.01", "2", "20"]:
        a = mp.mpf(a_str)
        for s_str in ["1.001", "10", "10000"]:
            s = mp.mpf(s_str)
            D, De, Ds, _, _ = solve_threshold_mp([mp.mpf(1)], a, s)
            verify_le("G1_mp_const_D", abs(D), mp.mpf("1e-180"), "G1_mp_const_D_max")
            verify_le("G1_mp_const_Ds", abs(Ds), mp.mpf("1e-180"), "G1_mp_const_Ds_max")

    const_law = MarkovOutageLaw("const", 0.0, 0.0, "const", 1.0)
    for e in [1e-2, 1e-3, 1e-4]:
        D, De, _ = solve_modal_threshold(const_law, 2.0, e, 40, 1)
        verify_le("G1_modal_const_D", abs(D), 1e-13, "G1_modal_const_D_max")
        verify_le("G1_modal_const_De", abs(De), 1e-11, "G1_modal_const_De_max")

    # Gate G2: exact closed-form alternating outage h = [2, 0] including near-boundary a, s
    seq20 = [mp.mpf(2), mp.mpf(0)]
    for a_str in ["1.01", "2", "50"]:
        a = mp.mpf(a_str)
        for s_str in ["1.001", "5", "5000"]:
            s = mp.mpf(s_str)
            D, _, Ds, _, _ = solve_threshold_mp(seq20, a, s)
            D_exact = mp.log((2 * a - 1) * (2 * s - 1) / (2 * ((2 * a - 1) * s - a + 1)))
            Ds_exact = mp.mpf(1) / ((2 * s - 1) * ((2 * a - 1) * s - a + 1))
            verify_le("G2_mp_alt_D", abs(D - D_exact), mp.mpf("1e-180"), "G2_mp_alt_D_max")
            verify_le("G2_mp_alt_Ds", abs(Ds - Ds_exact), mp.mpf("1e-180"), "G2_mp_alt_Ds_max")

    alt_law = MarkovOutageLaw("alt_2_0", 0.5, -1.0, "const", 1.0)
    for e in [1e-2, 1e-3, 1e-4]:
        s = mp.mpf(1) / mp.mpf(str(e))
        a_mp = mp.mpf(2)
        D_exact = float(mp.log((2 * a_mp - 1) * (2 * s - 1) / (2 * ((2 * a_mp - 1) * s - a_mp + 1))))
        De_exact = float(-s * s / ((2 * s - 1) * ((2 * a_mp - 1) * s - a_mp + 1)))
        D, De, _ = solve_modal_threshold(alt_law, 2.0, e, 48, 1)
        verify_le("G2_modal_alt_D", abs(D - D_exact), 1e-12, "G2_modal_alt_D_max")
        verify_le("G2_modal_alt_De", abs(De - De_exact), 1e-10, "G2_modal_alt_De_max")


def run_section2_audit() -> None:
    print("2. Section 2 audit: finite Hessian convexity, edge occupations, contraction & perturbations", flush=True)
    test_records = [
        [1],
        [2, 0],
        [3, 0, 0],
        [0, 0, 1, 3],
        [16] + [0] * 15,
    ]
    # 2a. Finite-product exact 1st & 2nd derivatives (joint convexity + eq. (4))
    for rec in test_records:
        seq = [mp.mpf(v) for v in (rec * 2 + [rec[0]])]  # non-multiple length
        n = len(seq)
        for a_str, q_str, g_str in [("1.01", "0.002", "0.95"), ("2", "0.96", "0.041"), ("20", "0.5", "0.3")]:
            a, q, g = mp.mpf(a_str), mp.mpf(q_str), mp.mpf(g_str)
            Pi, Pq, Pg, Pqq, Pgg, Pqg = cycle_product_and_jet(seq, a, q, g)
            ones = mp.matrix([1, 1])
            Z = (ones.T * Pi * ones)[0]
            Zq = (ones.T * Pq * ones)[0]
            Zg = (ones.T * Pg * ones)[0]
            Zqq = (ones.T * Pqq * ones)[0]
            Zgg = (ones.T * Pgg * ones)[0]
            Zqg = (ones.T * Pqg * ones)[0]
            Fx_exact = q * Zq / (n * Z)
            Fy_exact = g * Zg / (n * Z)
            Fxx = (q * Zq / Z + q * q * (Zqq * Z - Zq * Zq) / (Z * Z)) / n
            Fyy = (g * Zg / Z + g * g * (Zgg * Z - Zg * Zg) / (Z * Z)) / n
            Fxy = (q * g * (Zqg * Z - Zq * Zg) / (Z * Z)) / n
            verify_le("S2_hess_xx_nonneg", -Fxx, mp.mpf("1e-180"))
            verify_le("S2_hess_yy_nonneg", -Fyy, mp.mpf("1e-180"))
            verify_le("S2_hess_det_nonneg", -(Fxx * Fyy - Fxy * Fxy), mp.mpf("1e-180"))

            # Forward/backward edge weights (eq. 4)
            mats = [step_matrix(h, a, q, g) for h in seq]
            fw = [mp.matrix([1, 1])]
            for M in mats:
                fw.append(M * fw[-1])
            bw = [None] * (n + 1)
            bw[n] = mp.matrix([1, 1])
            for t in range(n - 1, -1, -1):
                bw[t] = mats[t].T * bw[t + 1]
            sum_pq = mp.mpf(0)
            sum_pg = mp.mpf(0)
            for t in range(n):
                pq_t = bw[t + 1][1] * q * fw[t][1] / Z
                pg_t = bw[t + 1][0] * g * fw[t][1] / Z
                verify("S2_prob_strict_01", 0 < pq_t < 1 and 0 < pg_t < 1 and (pq_t + pg_t) < 1)
                sum_pq += pq_t / n
                sum_pg += pg_t / n
            verify_le("S2_edge_grad_x", abs(sum_pq - Fx_exact), mp.mpf("1e-180"), "S2_edge_grad_err_max")
            verify_le("S2_edge_grad_y", abs(sum_pg - Fy_exact), mp.mpf("1e-180"), "S2_edge_grad_err_max")

    # 2b. s -> 1+ (e -> 1-, q -> 0+) C^1 positivity R_y > 0
    for rec in [[2, 0], [3, 0, 0], [16] + [0] * 15]:
        seq = [mp.mpf(v) for v in rec]
        for s_str in ["1.0001", "1.01", "1.1"]:
            D, De, Ds, Rx, Ry = solve_threshold_mp(seq, mp.mpf("1.02"), mp.mpf(s_str))
            verify("S2_Ry_positive_near_s1", Ry > 0, f"Ry={Ry}")
            verify("S2_Ds_finite_near_s1", mp.isfinite(Ds))

    # 2c. Projective contraction eq. (5) on M and M^T, including rank-1 (uz=vw) and h=0, 1e+-80
    # Explicit rank-1 positive check: u=0.6, v=0.3, w=0.4, z=0.2 => uz - vw = 0 => kappa = 0
    for u, v, w, z in [
        (mp.mpf("0.6"), mp.mpf("0.3"), mp.mpf("0.4"), mp.mpf("0.2")),  # rank-1
        (mp.mpf("0.5"), mp.mpf("0.02"), mp.mpf("0"), mp.mpf("0.98")),   # M at h=0
        (mp.mpf("0.5"), mp.mpf("0"), mp.mpf("0.02"), mp.mpf("0.98")),   # M^T at h=0
        (mp.mpf("0.01"), mp.mpf("1e-4"), mp.mpf("1e-80"), mp.mpf("0.99")),
        (mp.mpf("0.99"), mp.mpf("1e-4"), mp.mpf("1e80"), mp.mpf("0.01")),
    ]:
        has_zero = (v == 0) or (w == 0)
        kappa = mp.mpf(1) if has_zero else abs(mp.sqrt(u * z) - mp.sqrt(v * w)) / (mp.sqrt(u * z) + mp.sqrt(v * w))
        if u * z == v * w and not has_zero:
            verify_le("S2_rank1_kappa_zero", kappa, mp.mpf("1e-200"))
        for t_str in ["1e-80", "1e-10", "1", "1e10", "1e80"]:
            t = mp.mpf(t_str)
            deriv = abs((u * z - v * w) * t / ((u * t + v) * (w * t + z)))
            if has_zero:
                verify("S2_zero_strict_nonexp", deriv < 1)
            verify_le("S2_proj_contraction", deriv - kappa, mp.mpf("1e-180"))

    # 2d. Adversarial opposite-sign log-ratio perturbation test for |P(l',f') - P(l,f)| <= min(1, exp(d1+d2)-1)
    for h_str in ["0", "0.2", "50"]:
        M = step_matrix(mp.mpf(h_str), mp.mpf("2"), mp.mpf("0.7"), mp.mpf("0.15"))
        for d1_str, d2_str in [("0.01", "0.02"), ("0.3", "0.4"), ("1.2", "0.9")]:
            d1, d2 = mp.mpf(d1_str), mp.mpf(d2_str)
            bound = min(mp.mpf(1), mp.exp(d1 + d2) - 1)
            for sgn1 in [-1, 1]:
                for sgn2 in [-1, 1]:
                    for a0_str, b0_str in [("-2", "2"), ("0", "0"), ("3", "-1")]:
                        a0, b0 = mp.mpf(a0_str), mp.mpf(b0_str)
                        l1 = mp.matrix([mp.exp(a0), 1])
                        f1 = mp.matrix([mp.exp(b0), 1])
                        l2 = mp.matrix([mp.exp(a0 + sgn1 * d1), 1])
                        f2 = mp.matrix([mp.exp(b0 + sgn2 * d2), 1])
                        Z1 = (l1.T * M * f1)[0]
                        Z2 = (l2.T * M * f2)[0]
                        pq1 = l1[1] * M[1, 1] * f1[1] / Z1
                        pq2 = l2[1] * M[1, 1] * f2[1] / Z2
                        pg1 = l1[0] * M[0, 1] * f1[1] / Z1
                        pg2 = l2[0] * M[0, 1] * f2[1] / Z2
                        err = max(abs(pq1 - pq2), abs(pg1 - pg2))
                        verify_le("S2_edge_perturbation_bound", err - bound, mp.mpf("1e-180"))
                        record_max("S2_edge_perturbation_tightness", err / bound)

    # 2e. Convergence of random-phase finite-product gradient grad F_n -> grad R at rate O(1/n)
    rec = [mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(3)]
    per = len(rec)
    a = mp.mpf(2)
    x = mp.log(mp.mpf("0.92"))
    y = mp.log(mp.mpf("0.085"))
    _, Rx_inf, Ry_inf = exact_R_and_grad(rec, a, x, y)
    prev_err = None
    for n in [16, 64, 256]:
        # Stationary random-phase expectation over the `per` phases
        fx_avg = mp.mpf(0)
        fy_avg = mp.mpf(0)
        q, g = mp.exp(x), mp.exp(y)
        ones = mp.matrix([1, 1])
        for phase in range(per):
            window = [rec[(phase + t) % per] for t in range(n)]
            Pi, Pq, Pg, _, _, _ = cycle_product_and_jet(window, a, q, g)
            Z = (ones.T * Pi * ones)[0]
            fx_avg += (q * (ones.T * Pq * ones)[0] / (n * Z)) / per
            fy_avg += (g * (ones.T * Pg * ones)[0] / (n * Z)) / per
        err = max(abs(fx_avg - Rx_inf), abs(fy_avg - Ry_inf))
        if prev_err is not None:
            verify("S2_grad_Fn_converges", err < prev_err / 3)
        prev_err = err
    verify_le("S2_grad_Fn_n256_err", prev_err, mp.mpf("5e-3"), "S2_grad_Fn_n256_err")


def run_sections_3_to_5_audit() -> list[dict]:
    print("3. Sections 3-5 audit across 8 periodic edge-case records at 220 digits", flush=True)
    seq_A0 = construct_zero_host_A0_record(mp.mpf(2))
    # Perturb u_* by +1e-4 to create a tiny negative-A reversal record with two zeros
    u_tiny = seq_A0[1] + mp.mpf("1e-4")
    seq_tiny_negA = [mp.mpf(0), u_tiny, mp.mpf(0), mp.mpf(4) - u_tiny]

    cases: list[tuple[str, list, list]] = [
        ("const_1", [mp.mpf(1)], [mp.mpf("1.02"), mp.mpf("1.25"), mp.mpf(2), mp.mpf(20)]),
        ("alt_2_0", [mp.mpf(2), mp.mpf(0)], [mp.mpf("1.02"), mp.mpf("1.25"), mp.mpf(2), mp.mpf(20)]),
        ("multizero_rev_3_0_0", [mp.mpf(3), mp.mpf(0), mp.mpf(0)], [mp.mpf("1.25"), mp.mpf(2), mp.mpf(20)]),
        ("block_2_2_0_0", [mp.mpf(2), mp.mpf(2), mp.mpf(0), mp.mpf(0)], [mp.mpf("1.02"), mp.mpf(2), mp.mpf(20)]),
        ("skew_0_0_1_3", [mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(3)], [mp.mpf("1.02"), mp.mpf(2), mp.mpf(20)]),
        ("spike_64_zeros63", [mp.mpf(64)] + [mp.mpf(0)] * 63, [mp.mpf("1.25"), mp.mpf(2)]),
        ("exact_A0_with_zeros", seq_A0, [mp.mpf(2)]),
        ("tiny_negA_with_zeros", seq_tiny_negA, [mp.mpf(2)]),
    ]

    interval_rows = []
    for name, seq, a_list in cases:
        n = len(seq)
        H = (mp.fsum(h**4 for h in seq) / n) ** (mp.mpf(1) / 4)
        for a in a_list:
            b = mp.mpf(1) / a
            p = mp.mpf(1) - b
            co = a13i_constants(a, H)
            A, B_at_1, _, _, _ = periodic_A_and_poly(seq, a, mp.mpf(1))
            verify_le("S3_B1_eq_A", abs(B_at_1 - A), mp.mpf("1e-180"), "S3_B1_eq_A_max")
            if name == "exact_A0_with_zeros":
                verify_le("S3_constructed_A0", abs(A), mp.mpf("1e-200"))

            # Recover exact polynomial coefficients of B(c) (deg 2) and C(c) (deg 3)
            c_pts_B = [mp.mpf(0), mp.mpf(1), mp.mpf(2)]
            vals_B = [periodic_A_and_poly(seq, a, cp)[1] for cp in c_pts_B]
            vand_B = mp.matrix([[cp**j for j in range(3)] for cp in c_pts_B])
            beta = mp.lu_solve(vand_B, mp.matrix(vals_B))
            sum_beta = mp.fsum(abs(beta[j]) * (2**j) for j in range(3))
            verify_le("S3_B0_majorant", sum_beta / co["B0"], mp.mpf(1) + mp.mpf("1e-180"), "S3_B0_ratio_max")

            c_pts_C = [mp.mpf(0), mp.mpf(1), mp.mpf(2), mp.mpf(3)]
            vals_C = [periodic_A_and_poly(seq, a, cp)[2] for cp in c_pts_C]
            vand_C = mp.matrix([[cp**j for j in range(4)] for cp in c_pts_C])
            gamma = mp.lu_solve(vand_C, mp.matrix(vals_C))
            sum_gamma = mp.fsum(abs(gamma[j]) * (2**j) for j in range(4))
            verify_le("S3_C0_majorant", sum_gamma / co["C0"], mp.mpf(1) + mp.mpf("1e-180"), "S3_C0_ratio_max")

            # Check Section 3 recursions, Lp norms, and cubic remainder at c in {0.5, 1, 2}
            for c in [mp.mpf("0.5"), mp.mpf(1), mp.mpf(2)]:
                _, B_val, C_val, u, w = periodic_A_and_poly(seq, a, c)
                for e in [min(mp.mpf("0.005"), b / 4), min(mp.mpf("0.0005"), b / 40)]:
                    q = mp.mpf(1) - e
                    g = e * c
                    Pi, lam, _, _ = perron_and_partials(seq, a, q, g)
                    # Stationary pullback ratio v_0 from right Perron eigenvector of Pi
                    v0 = (b / (e * c)) * Pi[0, 1] / (lam - Pi[0, 0])
                    vs = [mp.mpf(0)] * n
                    cur = v0
                    for t in range(n):
                        vs[t] = cur
                        cur = (p * cur + b) / (1 + e * (c * seq[t] * cur - 1))
                    verify_le("S3_periodicity_v", abs(cur - v0), mp.mpf("1e-180"))
                    delta = [v - 1 for v in vs]
                    eta = [delta[t] - e * u[t] for t in range(n)]
                    zeta = [eta[t] - e * e * w[t] for t in range(n)]

                    for t in range(n):
                        tn = (t + 1) % n
                        kt = c * seq[t] - 1
                        Wt = c * seq[t] * vs[t] - 1
                        res_d = delta[tn] - p * delta[t] + e * vs[tn] * Wt
                        res_e = eta[tn] - p * eta[t] + e * (delta[tn] * Wt + c * seq[t] * delta[t])
                        res_z = zeta[tn] - p * zeta[t] + e * (
                            c * seq[t] * eta[t] + kt * eta[tn] + c * seq[t] * delta[t] * delta[tn]
                        )
                        verify_le("S3_rec_delta", abs(res_d), mp.mpf("1e-180"), "S3_rec_res_max")
                        verify_le("S3_rec_eta", abs(res_e), mp.mpf("1e-180"), "S3_rec_res_max")
                        verify_le("S3_rec_zeta", abs(res_z), mp.mpf("1e-180"), "S3_rec_res_max")

                    def lp_norm(arr, power):
                        return (mp.fsum(abs(x) ** power for x in arr) / n) ** (mp.mpf(1) / power)

                    verify_le("S3_u_L4", lp_norm(u, 4) / co["U"], 1 + mp.mpf("1e-180"), "S3_Lp_ratio_max")
                    verify_le("S3_w_L2", lp_norm(w, 2) / co["J"], 1 + mp.mpf("1e-180"), "S3_Lp_ratio_max")
                    verify_le("S3_delta_L4", lp_norm(delta, 4) / (co["L"] * e), 1 + mp.mpf("1e-180"), "S3_Lp_ratio_max")
                    verify_le("S3_eta_L2", lp_norm(eta, 2) / (co["N"] * e**2), 1 + mp.mpf("1e-180"), "S3_Lp_ratio_max")
                    verify_le(
                        "S3_zeta_L43",
                        lp_norm(zeta, mp.mpf(4) / 3) / (co["Z"] * e**3),
                        1 + mp.mpf("1e-180"),
                        "S3_zeta_L43_ratio_max",
                    )

                    r_exact = mp.log(lam) / n
                    P_val = e * (c - 1) - e**2 * B_val + e**3 * C_val
                    rem_ratio = abs(r_exact - P_val) / (co["R4"] * e**4)
                    verify_le("S3_R4_remainder", rem_ratio, 1 + mp.mpf("1e-180"), "S3_R4_ratio_max")

            # Check Sections 4-5 at BOTH e = estar (for eq. 2) AND near the strict sign cutoff (eq. 3)
            test_es = [co["estar"]]
            has_nonzero_A = abs(A) > mp.mpf("1e-150")
            if has_nonzero_A:
                sign_cut = min(co["estar"], (abs(A) / (2 * co["Cstar"])) ** 2)
                test_es.extend([mp.mpf("0.99") * sign_cut, mp.mpf("0.5") * sign_cut])

            for idx_e, e_test in enumerate(test_es):
                s_test = mp.mpf(1) / e_test
                D, De, Ds, Rx, Ry = solve_threshold_mp(seq, a, s_test)
                c_root = mp.exp(D)
                verify_le("S4_root_c_minus_1", abs(c_root - 1) / (co["K"] * e_test), 1 + mp.mpf("1e-150"))

                # Exact P_x and P_y at the root
                _, B_r, C_r, _, _ = periodic_A_and_poly(seq, a, c_root)
                B_prime = beta[1] + 2 * beta[2] * c_root
                C_prime = gamma[1] + 2 * gamma[2] * c_root + 3 * gamma[3] * c_root**2
                Py = e_test * c_root - e_test**2 * c_root * B_prime + e_test**3 * c_root * C_prime
                Pe = (c_root - 1) - 2 * e_test * B_r + 3 * e_test**2 * C_r
                q_test = 1 - e_test
                Px = q_test * (Py / e_test - Pe)

                verify_le(
                    "S4_Rx_secant_bound",
                    abs(Rx - Px) / (co["Ex"] * e_test**2),
                    1 + mp.mpf("1e-140"),
                    "S4_Rx_ratio_max",
                )
                verify_le(
                    "S4_Ry_secant_bound",
                    abs(Ry - Py) / (co["Ey"] * e_test ** mp.mpf("2.5")),
                    1 + mp.mpf("1e-140"),
                    "S4_Ry_ratio_max",
                )

                # Main theorem bound (2): |s^2 D_s + A| <= C_* / sqrt(s)
                lhs = abs(s_test * s_test * Ds + A)
                rhs_bound = co["Cstar"] / mp.sqrt(s_test)
                verify_le("S5_main_bound_eq2", lhs / rhs_bound, 1 + mp.mpf("1e-150"), "S5_eq2_ratio_max")

                if has_nonzero_A and idx_e >= 1:
                    # Strictly inside (3): sign(D_s) = -sign(A) and sign(D) = sign(A)
                    verify("S5_strict_Ds_sign", Ds * A < 0, f"Ds={Ds}, A={A}")
                    verify("S5_strict_D_sign", D * A > 0, f"D={D}, A={A}")

                if idx_e == len(test_es) - 1:
                    interval_rows.append(
                        {
                            "record": name,
                            "a": float(a),
                            "A": float(A),
                            "e_tested": mp.nstr(e_test, 6),
                            "s2_Ds_plus_A": mp.nstr(s_test * s_test * Ds + A, 6),
                            "bound_ratio": float(lhs / rhs_bound),
                        }
                    )
                    print(
                        f"  [{name}] a={mp.nstr(a,4)} A={mp.nstr(A,8)} "
                        f"e={mp.nstr(e_test,5)} -s^2*D_s={mp.nstr(-s_test*s_test*Ds,8)} "
                        f"err/bound={mp.nstr(lhs/rhs_bound,4)}",
                        flush=True,
                    )
    return interval_rows


def run_unbounded_laws_audit() -> list[dict]:
    print("4. Unbounded fourth-moment laws with zeros (including Pareto(4.05) & singular Gamma(0.25))", flush=True)
    laws = [
        MarkovOutageLaw("pareto405_iid", 0.5, 0.0, "pareto", 4.05),
        MarkovOutageLaw("pareto405_rev", 0.5, -0.96, "pareto", 4.05),
        MarkovOutageLaw("pareto450_iid", 0.5, 0.0, "pareto", 4.5),
        MarkovOutageLaw("pareto450_rev", 0.5, -0.96, "pareto", 4.5),
        MarkovOutageLaw("gamma025_sing_iid", 0.5, 0.0, "gamma", 0.25),
        MarkovOutageLaw("gamma025_sing_anti", 0.5, -0.985, "gamma", 0.25),
        MarkovOutageLaw("gamma8_switch", 0.5, -0.96, "gamma", 8.0),
        MarkovOutageLaw("lognormal_q85", 0.85, 0.0, "lognormal", 0.7),
    ]
    cells = [(law, 2.0) for law in laws] + [
        (laws[1], 1.25),
        (laws[6], 1.5),
        (laws[6], 4.0),
    ]
    stays = [100.0, 500.0, 2500.0, 12500.0]

    rows = []
    for law, a in cells:
        A_exact = law.exact_A(a)
        H4 = law.H4_norm()
        co = a13i_constants(mp.mpf(str(a)), mp.mpf(str(H4)))
        Cstar_f = float(co["Cstar"])
        De_fine_list = []

        for s in stays:
            e = 1.0 / s
            Dc, Dec, _ = solve_modal_threshold(law, a, e, 48, 112)
            Df, Def, _ = solve_modal_threshold(law, a, e, 80, 192)
            verify_le("U_D_refinement", abs(Df - Dc), 5e-10, "U_D_refinement_max")
            verify_le("U_De_refinement", abs(Def - Dec), 5e-7, "U_De_refinement_max")

            # Cross-check exact operator derivative De against centered finite difference of Df(e)
            if s in (500.0, 12500.0):
                h_step = e * 0.02
                Dp, _, _ = solve_modal_threshold(law, a, e + h_step, 64, 144)
                Dm, _, _ = solve_modal_threshold(law, a, e - h_step, 64, 144)
                De_fd = (Dp - Dm) / (2.0 * h_step)
                verify_le("U_De_analytic_vs_fd", abs(Def - De_fd), 1e-6, "U_De_analytic_vs_fd_max")

            De_fine_list.append(Def)
            Ds_f = -Def / (s * s)
            err_s = abs(s * s * Ds_f + A_exact)
            # Note: s <= 12500 is outside the conservative analytical threshold 1/e_*,
            # while empirically err_s = O(1/s) << C_*/sqrt(s).
            verify_le("U_empirical_within_Cstar", err_s / (Cstar_f / math.sqrt(s)), 1.0)

        # Check O(1/s) decay of |D_e - A|, sign at s=12500, and 2-point Richardson extrapolation
        err_2500 = abs(De_fine_list[-2] - A_exact)
        err_12500 = abs(De_fine_list[-1] - A_exact)
        verify("U_O_inv_s_decay", err_12500 < err_2500 / 4.5, f"{err_12500} vs {err_2500}")
        verify("U_sign_agreement_s12500", De_fine_list[-1] * A_exact > 0)

        e1, e2 = 1.0 / stays[-2], 1.0 / stays[-1]
        De_rich = (e1 * De_fine_list[-1] - e2 * De_fine_list[-2]) / (e1 - e2)
        verify_le("U_richardson_to_A", abs(De_rich - A_exact), 5e-4, "U_richardson_err_max")

        row = {
            "law": law.name,
            "a": a,
            "E_h4": round(H4**4, 4),
            "A": A_exact,
            "neg_s2_Ds_s100": De_fine_list[0],
            "neg_s2_Ds_s2500": De_fine_list[2],
            "neg_s2_Ds_s12500": De_fine_list[3],
            "richardson_err": abs(De_rich - A_exact),
        }
        rows.append(row)
        print(
            f"  [{law.name}] a={a} E[h^4]={H4**4:.2f} A={A_exact:+.8f} "
            f"-s^2*D_s(12500)={De_fine_list[-1]:+.8f} Rich.err={abs(De_rich - A_exact):.2e}",
            flush=True,
        )
    return rows


def main() -> None:
    t0 = time.monotonic()
    print("Independent referee check verify_a13i.py (220 decimal digits + modal Chebyshev)", flush=True)
    run_known_result_gates()
    if FAILURES:
        raise SystemExit(1)
    run_section2_audit()
    periodic_rows = run_sections_3_to_5_audit()
    unbounded_rows = run_unbounded_laws_audit()
    elapsed = time.monotonic() - t0

    out = {
        "assertions": ASSERTIONS,
        "failures": FAILURES,
        "summary_metrics": SUMMARY,
        "periodic_intervals": periodic_rows,
        "unbounded_cells": unbounded_rows,
        "seconds": round(elapsed, 3),
    }
    (HERE / "verify_a13i_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("SUMMARY_METRICS", json.dumps(SUMMARY, sort_keys=True), flush=True)
    print(f"ASSERTIONS {ASSERTIONS} FAILURES {len(FAILURES)} SECONDS {elapsed:.3f}", flush=True)
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
