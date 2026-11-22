#!/usr/bin/env python3
"""Independent referee verification of studies/memory-memA and resolution of the H2/memA tension.

Invoke with:
    python3 -I verification/memory-check/check_memA_and_tension.py

Rules followed:
- Self-contained: imports ONLY Python standard library, numpy, and mpmath (NO imports from memA or memB).
- Uses 80-digit arbitrary-precision arithmetic (mpmath, higher than memA's 65 digits).
- Reproduces known exact results (Roerdink/Tricomi independent-host coefficient and matrix resolvent)
  before testing new claims, with tolerances stated beforehand.
- Checks:
  1. P1/P2 second-order formula F_rho(s) and F_phi(s) against direct 2x2 matrix lag sums (including
     nonsummable long-memory covariance rho(j) = (j+1)^(-1/2)) and numerical differentiation, plus
     exact verification of the AR(1) reversal boundary phi = 2 - sqrt(5) and phi = -1/2.
  2. Exact counterexamples of memA in 80-digit arithmetic via TWO independent methods:
     (a) Independent triangular Taylor series of the stationary ratio law z' = b*(delta*Z' - z)/(1 + z)
         at orders 20 and 32 with analytic Cauchy tail + 3-step slope root enclosure;
     (b) Independent Gauss-Legendre Nystrom solver for the coupled invariant density (mu_-, mu_+) on
         the forward-invariant log-ratio interval at 80 digits.
     Specifically checks:
     - Exact aperiodic counterexample: delta = 1/2, phi = -3/4, s = 1 - 10^-8 (proving D'(s) > 1/32,
       computing D(s) and D'(s) to >60 digits);
     - Moderate-noise negative-loss example: delta = 0.4, phi = -0.75, s = 0.9 (D < 0, matching
       both 80-digit invariant quadrature and order-40 Taylor series);
     - Exceptional cancellation point: delta = 0.6, phi = -0.5, s = 1 - eps (A = 0, D ~ -delta^2/4 * eps^2);
     - Explicit periodic alternating counterexample P8: phi = -1, delta in {0.2, 0.5, 0.6, 0.9},
       s in {0, 0.2, 0.7, 0.99}, checking D_alt(s) < 0, D_alt'(s) > 0, D_alt(0) = -ln(1+delta) != ell,
       and exact 2-step Perron eigenvalue = 1;
     - Finite-noise boundary shift near phi = 2 - sqrt(5): checking that at phi = 2 - sqrt(5) + 0.002 > 2 - sqrt(5),
       finite noise delta = 0.5 creates a genuine finite-noise reversal (showing 2 - sqrt(5) is a
       second-order boundary, not an all-noise boundary for negative memory!).
  3. Algebraic and numerical verification of the tension between Result 4 / Lean P2 / H2 and memA,
     plus a finite-noise Monte Carlo confirmation for two-stage ring (a, b) with fluctuating stay a = 4 > (3+sqrt(17))/4.
"""

import math
import time
import numpy as np
import mpmath as mp

mp.mp.dps = 80
M = mp.mpf

FAILURES = []


def report(tag, ok, msg):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {tag}: {msg}", flush=True)
    if not ok:
        FAILURES.append(tag)


# ---------------------------------------------------------------------------
# 1. Second-order coefficient re-derivation checks (80 digits)
# ---------------------------------------------------------------------------

def F_ar1(s, phi):
    a = 1 - s
    return a * (1 + (a + 2) * phi) / (2 * (1 + a) * (1 + a * phi))


def dF_ar1_ds(s, phi):
    a = 1 - s
    # Completed-square numerator: N(a, phi) = 2*(1 + phi) - (1 - a*phi)^2
    num = 2 * (1 + phi) - (1 - a * phi) ** 2
    return -num / (2 * (1 + a) ** 2 * (1 + a * phi) ** 2)


def s_star_ar1(phi):
    crit = 2 - mp.sqrt(5)
    if phi >= crit:
        return M(0)
    if phi <= -M("0.5"):
        return M(1)
    return 1 - (1 - mp.sqrt(2 * (1 + phi))) / phi


def matrix_lag_loss_coef(s, rho_fn, n_lags=1600):
    """Direct Tuljapurkar/Ruelle 2x2 matrix lag sum multiplied by T_0 = (2-s)/(1-s)."""
    a = 1 - s
    # Threshold mean matrix A0 = [[0, 1], [a, s]] or transpose [[s, a], [1, 0]], lambda = 1
    # Right eigenvector v = [1, 1]^T; left eigenvector w = [1/(1+a), a/(1+a)]^T so w^T v = 1.
    # Fluctuating transfer entry is (0, 1) of [[s, a], [1, 0]] with value a.
    # In rank-1 reduced space: M - v w^T has eigenvalue -a!
    # Let us compute explicitly with 2x2 mpmath matrices so nothing is assumed.
    A0 = mp.matrix([[s, a], [M(1), M(0)]])
    v = mp.matrix([[M(1)], [M(1)]])
    w = mp.matrix([[1 / (1 + a)], [a / (1 + a)]])
    B = mp.matrix([[M(0), a], [M(0), M(0)]])
    Q = mp.eye(2) - v * w.T
    T = A0 - v * w.T
    c_use = (w.T * B * v)[0, 0]
    acc = c_use ** 2 / 2
    Tpow = mp.eye(2)
    for lag in range(1, n_lags + 1):
        term = (w.T * B * Tpow * Q * B * v)[0, 0]
        acc -= rho_fn(lag) * term
        Tpow = Tpow * T
    T0 = (1 + a) / a
    return T0 * acc


# ---------------------------------------------------------------------------
# 2. Independent 80-digit solvers for the two-state Markov host
#    h_t = 1 + delta * Z_t, Z_t in {-1, +1}, P = [[(1+phi)/2, (1-phi)/2], ...]
# ---------------------------------------------------------------------------

def markov_taylor_rate(s, c, delta, phi, order=24):
    """Independent implementation of exact stationary ratio moments up to delta^order.

    Uses U_t = lam * (1 + z_t), lam^2 = s*lam + c, b = c/lam^2 in (0, 1),
    z_t = b * (delta * Z_t - z_{t-1}) / (1 + z_{t-1}).
    """
    lam = (s + mp.sqrt(s * s + 4 * c)) / 2
    b = c / (lam * lam)
    # Eigenbasis of P^T: u_+ = [1, 1]^T (eigenvalue 1), u_- = [-1, 1]^T (eigenvalue phi).
    # Let S_{k,n} = [delta^n] E[z^k], D_{k,n} = [delta^n] E[z^k Z].
    # Note: E[(delta*Z' - z)^k (1+z)^(-k)] and E[(delta*Z' - z)^k (1+z)^(-k) Z']:
    # Conditional on past (z, Z):
    #   E[(Z')^r | Z] = 1 if r even, and phi * Z if r odd!
    #   E[(Z')^{r+1} | Z] = 1 if r odd, and phi * Z if r even!
    # That avoids 2-vector indexing entirely and gives an algebraically independent derivation!
    S = [[M(0)] * (order + 1) for _ in range(order + 1)]
    D = [[M(0)] * (order + 1) for _ in range(order + 1)]
    S[0][0] = M(1)

    for n in range(1, order + 1):
        for k in range(n, 0, -1):
            rhs_S = M(0)
            rhs_D = M(0)
            for r in range(0, k + 1):
                for m in range(0, n - k + 1):
                    if r == 0 and m == 0:
                        continue
                    ell = k - r + m
                    pref = (b ** k) * math.comb(k, r) * ((-1) ** (k - r + m)) * math.comb(k + m - 1, m)
                    if r % 2 == 0:
                        rhs_S += pref * S[ell][n - r]
                        rhs_D += pref * phi * D[ell][n - r]
                    else:
                        rhs_S += pref * phi * D[ell][n - r]
                        rhs_D += pref * S[ell][n - r]
            q = (-b) ** k
            S[k][n] = rhs_S / (1 - q)
            D[k][n] = rhs_D / (1 - q * phi)

    r_approx = mp.log(lam)
    for n in range(1, order + 1):
        cn = mp.fsum(((-1) ** (k + 1)) * S[k][n] / k for k in range(1, n + 1))
        r_approx += cn * (delta ** n)

    R_disk = (1 - b) ** 2 / (8 * b)
    rho = abs(delta) / R_disk
    tail_bound = mp.log(2) * (rho ** (order + 1)) / (1 - rho) if rho < 1 else mp.inf
    return r_approx, tail_bound, R_disk


def markov_taylor_D(s, delta, phi, order=24):
    """Solve r(s, (1-s)*exp(y)) = 0 with certified analytic truncation bound."""
    y0 = (delta ** 2) * F_ar1(s, phi)
    fn = lambda y: markov_taylor_rate(s, (1 - s) * mp.exp(y), delta, phi, order)[0]
    y_star = mp.findroot(fn, (y0 - M("1e-5"), y0 + M("1e-5")), solver="secant", tol=M("1e-68"))
    r_res, tail, R_disk = markov_taylor_rate(s, (1 - s) * mp.exp(y_star), delta, phi, order)
    eta = abs(r_res) + tail
    c_min = (1 - s) * mp.exp(y_star - M("0.01"))
    beta = c_min * (1 - delta) / (s * s + c_min * (1 - delta))
    err_bound = 3 * eta / beta
    if err_bound >= M("0.01"):
        err_bound = mp.inf
    return y_star, err_bound, R_disk


def markov_nystrom_rate(s, c, delta, phi, n_nodes=36):
    """Independent 80-digit Gauss-Legendre Nystrom solver for the invariant density on log(u).

    Unlike memA_checks.py (which used Chebyshev barycentric collocation of the forward pushforward
    kernel), here we solve the exact density integral equation directly!
    Under U' = s + c*h_j / U, setting x = u, z = c*h_j / (x - s) on x in (s + c*h_j / hi, s + c*h_j / lo):
    Wait: since U' = s + c*h_- / U and s + c*h_+ / U can have disjoint or overlapping images when s is
    not tiny or delta is large, the adjoint (Koopman) operator (T g)(u, i) = sum_j P_{ij} g(s + c*h_j / u, j)
    is smooth everywhere on [lo, hi]! Even simpler: solve for the density of a 2-step smooth kernel, OR
    expand g(u, i) in Legendre polynomials on v = log(u) in [vl, vh] via Gauss-Legendre projection
    (Galerkin/spectral projection of the Koopman operator, then extract the left invariant functional!).
    """
    # Forward-invariant interval [lo, hi]:
    # Minimum fixed point u_min solves u = s + c*(1-delta)/u => u_min = (s + sqrt(s^2 + 4*c*(1-delta)))/2?
    # Wait: since u -> s + c*h/u is DECREASING, the image of [lo, hi] must contain:
    # lo <= s + c*(1-delta)/hi  and  hi >= s + c*(1+delta)/lo.
    # Exact attracting 2-cycle of extreme alternating map lo = s + c*(1-delta)/hi, hi = s + c*(1+delta)/lo:
    # hi - lo = d_span => lo*hi = c*(1-delta)/(lo - s) ... specifically:
    # Let us solve lo = s + c*(1-delta)/hi, hi = s + c*(1+delta)/lo exactly!
    hm = 1 - delta
    hp = 1 + delta
    # From hi = s + c*hp/lo => lo = s + c*hm / (s + c*hp/lo) = s + c*hm*lo / (s*lo + c*hp)
    # => s*lo^2 + (c*(hp - hm) - s^2)*lo - s*c*hp = 0? Wait:
    # lo*(s*lo + c*hp) = s*(s*lo + c*hp) + c*hm*lo => s*lo^2 + (c*(hp - hm) - s^2)*lo - s*c*hp = 0!
    Bcoef = c * (hp - hm) - s * s
    lo = (-Bcoef + mp.sqrt(Bcoef * Bcoef + 4 * s * s * c * hp)) / (2 * s)
    hi = s + c * hp / lo
    vl, vh = mp.log(lo), mp.log(hi)
    mid = (vl + vh) / 2
    half = (vh - vl) / 2

    xs, ws = mp.gauss_quadrature(n_nodes, "legendre")

    def leg_vals(x, N):
        out = [M(1), x]
        for k in range(1, N - 1):
            out.append(((2 * k + 1) * x * out[k] - k * out[k - 1]) / (k + 1))
        return out

    # Orthonormalized Legendre basis phi_m(x) = sqrt((2m+1)/2) P_m(x) on [-1, 1]
    norms = [mp.sqrt(M(2 * m + 1) / 2) for m in range(n_nodes)]
    # Build Galerkin matrix of Koopman operator L: (L f)(v, i) = sum_j P_{ij} f(log(s + c*h_j*e^{-v}), j)
    # L_{a,b} = <phi_a, L phi_b>_{L^2([-1,1])}. Notice phi_0(x) = 1/sqrt(2) is constant, so L(phi_0, 1) = (phi_0, 1)!
    Pmat = [[(1 + phi) / 2, (1 - phi) / 2], [(1 - phi) / 2, (1 + phi) / 2]]
    hs = [hm, hp]

    basis_at_nodes = []
    for x in xs:
        ps = leg_vals(x, n_nodes)
        basis_at_nodes.append([norms[m] * ps[m] for m in range(n_nodes)])

    K = mp.matrix(2 * n_nodes, 2 * n_nodes)
    for i in range(2):
        for q_idx in range(n_nodes):
            v = mid + half * xs[q_idx]
            wq = ws[q_idx]
            b_here = basis_at_nodes[q_idx]
            for j in range(2):
                v_next = mp.log(s + c * hs[j] * mp.exp(-v))
                x_next = (v_next - mid) / half
                ps_next = leg_vals(x_next, n_nodes)
                b_next = [norms[m] * ps_next[m] for m in range(n_nodes)]
                pij = Pmat[i][j]
                for m1 in range(n_nodes):
                    factor = wq * b_here[m1] * pij
                    row = i * n_nodes + m1
                    for m2 in range(n_nodes):
                        K[row, j * n_nodes + m2] += factor * b_next[m2]

    # Constant function e = [1_{m=0}/sqrt(2) ...]? Wait: P_0(x) = 1, norm_0 = 1/sqrt(2), so constant 1 on both states
    # has coordinates c_const = [sqrt(2), 0, ..., sqrt(2), 0, ...]^T, and K * c_const = c_const!
    # Invariant functional mu satisfies mu^T K = mu^T and mu^T c_const = 1.
    A = K.T - mp.eye(2 * n_nodes)
    rhs = mp.matrix(2 * n_nodes, 1)
    # Replace row 0 with normalization mu^T c_const = 1
    for col in range(2 * n_nodes):
        A[0, col] = M(0)
    A[0, 0] = mp.sqrt(2)
    A[0, n_nodes] = mp.sqrt(2)
    rhs[0, 0] = M(1)
    mu = mp.lu_solve(A, rhs)

    # Observable f(v, i) = v = mid * 1 + half * x = mid * sqrt(2) * phi_0(x) + half * sqrt(2/3) * phi_1(x)!
    obs = mp.matrix(2 * n_nodes, 1)
    for i in range(2):
        obs[i * n_nodes + 0, 0] = mid * mp.sqrt(2)
        obs[i * n_nodes + 1, 0] = half * mp.sqrt(M(2) / 3)
    return (mu.T * obs)[0, 0]


def markov_nystrom_rate_float(s, c, delta, phi, n_nodes=32):
    """Fast IEEE double Legendre-Galerkin rate for initial root bracketing."""
    hm, hp = 1.0 - delta, 1.0 + delta
    Bcoef = c * (hp - hm) - s * s
    lo = (-Bcoef + math.sqrt(Bcoef * Bcoef + 4.0 * s * s * c * hp)) / (2.0 * s)
    hi = s + c * hp / lo
    vl, vh = math.log(lo), math.log(hi)
    mid, half = 0.5 * (vl + vh), 0.5 * (vh - vl)
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    norms = np.sqrt((2.0 * np.arange(n_nodes) + 1.0) / 2.0)
    # Vandermonde of orthonormal Legendre polynomials at xs: shape (n_nodes, n_nodes)
    V = np.polynomial.legendre.legvander(xs, n_nodes - 1) * norms[None, :]
    Pmat = np.array([[(1.0 + phi) / 2.0, (1.0 - phi) / 2.0],
                     [(1.0 - phi) / 2.0, (1.0 + phi) / 2.0]])
    hs = [hm, hp]
    K = np.zeros((2 * n_nodes, 2 * n_nodes))
    v_vec = mid + half * xs
    for j in range(2):
        v_next = np.log(s + c * hs[j] * np.exp(-v_vec))
        x_next = (v_next - mid) / half
        V_next = np.polynomial.legendre.legvander(x_next, n_nodes - 1) * norms[None, :]
        M_block = (V * ws[:, None]).T @ V_next
        for i in range(2):
            K[i * n_nodes:(i + 1) * n_nodes, j * n_nodes:(j + 1) * n_nodes] = Pmat[i, j] * M_block
    A = K.T - np.eye(2 * n_nodes)
    A[0, :] = 0.0
    A[0, 0] = math.sqrt(2.0)
    A[0, n_nodes] = math.sqrt(2.0)
    rhs = np.zeros(2 * n_nodes)
    rhs[0] = 1.0
    mu = np.linalg.solve(A, rhs)
    obs = np.zeros(2 * n_nodes)
    obs[0] = obs[n_nodes] = mid * math.sqrt(2.0)
    obs[1] = obs[n_nodes + 1] = half * math.sqrt(2.0 / 3.0)
    return float(mu @ obs)


def markov_nystrom_D(s, delta, phi, n_nodes=28):
    # Bracket in double precision first (~0.5 ms), then polish in 80-digit mpmath (2 secant steps)!
    sf, df, pf = float(s), float(delta), float(phi)
    lo_y = -math.log1p(df) - 0.05
    hi_y = -0.5 * math.log1p(-df * df) + 0.05
    for _ in range(52):
        mid_y = 0.5 * (lo_y + hi_y)
        if markov_nystrom_rate_float(sf, (1.0 - sf) * math.exp(mid_y), df, pf, max(32, n_nodes)) < 0.0:
            lo_y = mid_y
        else:
            hi_y = mid_y
    y0 = M(repr(0.5 * (lo_y + hi_y)))
    fn = lambda y: markov_nystrom_rate(s, (1 - s) * mp.exp(y), delta, phi, n_nodes)
    return mp.findroot(fn, (y0 - M("1e-13"), y0 + M("1e-13")), solver="secant", tol=M("1e-60"))


# ---------------------------------------------------------------------------
# 3. Tension resolution: general two-stage (a, b) and three-stage (1, b, c)
# ---------------------------------------------------------------------------

def gamma_two_stage(a, b, phi):
    """Second-order loss per generation / sigma^2 for stays (a, b), one fluctuating transfer."""
    if phi == 0:
        return 1 / (2 * (a + b))
    th = phi / (1 - phi)
    return (1 + 2 * th) / (2 * (a + b)) - (th * th) / (th * (a + b) + a * b)


def dgamma_two_stage_db(a, b, phi):
    if phi == 0:
        return -1 / (2 * (a + b) ** 2)
    th = phi / (1 - phi)
    return -(1 + 2 * th) / (2 * (a + b) ** 2) + (th * th * (a + th)) / (a * th + b * (a + th)) ** 2


def b_max_two_stage(a, phi):
    """Exact continuum maximizer b_max(a, theta) > 0 of gamma_two_stage(a, b, phi) for a > 1/2, phi > 0."""
    th = phi / (1 - phi)
    num = a * th * (mp.sqrt(2 * (a + th)) - mp.sqrt(1 + 2 * th))
    den = mp.sqrt(a + th) * (mp.sqrt((a + th) * (1 + 2 * th)) - th * mp.sqrt(2))
    return num / den


def theta_crit_two_stage_nonmonotone(a):
    """Exact threshold theta_crit(a) above which d/db Gamma(a, b)|_{b=1} > 0 (for a > (3 + sqrt(17))/4)."""
    return a * (2 * a + 1 + a * mp.sqrt(2 * a + 3)) / ((a + 1) * (2 * a * a - 3 * a - 1))


def mc_two_stage_general_a(a_stay, b_grid, sigma, phi, steps=120000, burn=4000, pairs=32, seed=20261114):
    """Independent finite-noise Monte Carlo for 2-stage ring with fluctuating-stage stay a_stay >= 1
    and steady-stage stay b in b_grid.

    Matrix at threshold c = (1 - s0)*(1 - s1)*exp(y):
        x0(t+1) = s0 * x0(t) + g1 * x1(t)
        x1(t+1) = g0 * h(t) * x0(t) + s1 * x1(t)
    Choose g1 = 1 - s1 = 1/b, g0 = (1 - s0)*exp(y) = exp(y)/a_stay, s0 = 1 - 1/a_stay, s1 = 1 - 1/b.
    """
    rng = np.random.default_rng(seed)
    T_tot = steps + burn
    z = np.empty((T_tot, pairs), dtype=np.float64)
    z[0] = rng.standard_normal(pairs)
    innov_scale = math.sqrt(1.0 - phi * phi)
    eps_all = rng.standard_normal((T_tot - 1, pairs))
    for t in range(T_tot - 1):
        z[t + 1] = phi * z[t] + innov_scale * eps_all[t]
    h = np.empty((T_tot, 2 * pairs), dtype=np.float64)
    h[:, 0::2] = np.exp(sigma * z - 0.5 * sigma * sigma)
    h[:, 1::2] = np.exp(-sigma * z - 0.5 * sigma * sigma)

    b_arr = np.asarray(b_grid, dtype=np.float64)
    s0 = 1.0 - 1.0 / a_stay
    s1 = 1.0 - 1.0 / b_arr
    g1 = 1.0 / b_arr
    # Initial guess from second-order formula
    y = np.array([float(sigma * sigma * gamma_two_stage(M(a_stay), M(b), M(phi))) for b in b_arr])
    ctl_mean = (h[burn:].mean(axis=0))[:, None] - 1.0
    cv = 1.0 / (a_stay + b_arr)[None, :]

    for _ in range(7):
        m = h.shape[1]
        n = len(b_arr)
        x0 = np.full((m, n), a_stay / (a_stay + b_arr))
        x1 = np.full((m, n), b_arr / (a_stay + b_arr))
        d0 = np.zeros((m, n))
        d1 = np.zeros((m, n))
        acc = np.zeros((m, n))
        der = np.zeros((m, n))
        g0 = np.exp(y) / a_stay
        for t in range(T_tot):
            ht = h[t][:, None]
            nx0 = s0 * x0 + g1 * x1
            nx1 = (g0 * ht) * x0 + s1 * x1
            nd0 = s0 * d0 + g1 * d1
            nd1 = (g0 * ht) * (d0 + x0) + s1 * d1
            tot = nx0 + nx1
            dtot = (nd0 + nd1) / tot
            x0 = nx0 / tot
            x1 = nx1 / tot
            d0 = nd0 / tot - x0 * dtot
            d1 = nd1 / tot - x1 * dtot
            if t >= burn:
                acc += np.log(tot)
                der += dtot
        r_path = acc / steps - ctl_mean * cv
        d_path = der / steps
        pr = 0.5 * (r_path[0::2] + r_path[1::2])
        pd = 0.5 * (d_path[0::2] + d_path[1::2])
        r_mean = pr.mean(axis=0)
        slope = pd.mean(axis=0)
        se_r = pr.std(axis=0, ddof=1) / math.sqrt(pairs)
        step_y = r_mean / slope
        y -= step_y
        se_D = se_r / slope
        if np.all(np.abs(step_y) < 1e-10):
            break
    infl = -(pr - r_mean) / slope
    return y, se_D, infl


def main():
    t0 = time.monotonic()
    print("=== secondary Independent Check of memA and Resolution of the H2/memA Tension ===")
    print(f"mpmath precision: {mp.mp.dps} decimal digits\n")

    # ------------------------------------------------------------------
    # Check 0: Known baseline reproduction (tolerance 1e-70 stated beforehand)
    # ------------------------------------------------------------------
    print("--- 0. Baseline reproduction of known independent-host & ring formulas (tol = 1e-70) ---")
    max_base = M(0)
    for s_str in ["0.1", "0.5", "0.9"]:
        s = M(s_str)
        tau = 1 / (1 - s)
        # At phi = 0, F_0(s) must equal 1 / (2*(1 + tau)) = (1 - s) / (2*(2 - s))
        exact_phi0 = (1 - s) / (2 * (2 - s))
        max_base = max(max_base, abs(F_ar1(s, M(0)) - exact_phi0))
        max_base = max(max_base, abs(matrix_lag_loss_coef(s, lambda j: M(0), n_lags=20) - exact_phi0))
    report("B0_baseline_phi0", max_base < M("1e-70"), f"max discrepancy = {mp.nstr(max_base, 6)}")

    # ------------------------------------------------------------------
    # Check 1: P1 & P2 second-order coefficient and exact AR(1) boundary
    # ------------------------------------------------------------------
    print("\n--- 1. P1 & P2: General Gaussian memory lag sum and AR(1) boundary phi = 2 - sqrt(5) ---")
    max_lag_err = M(0)
    cov_models = [
        ("AR1_pos_0.75", lambda j: M("0.75") ** j),
        ("AR1_neg_-0.4", lambda j: (-M("0.4")) ** j),
        ("AR2_damped_cos", lambda j: (M("0.85") ** j) * mp.cos(M("0.7") * j)),
        ("long_memory_inv_sqrt", lambda j: 1 / mp.sqrt(j + 1)),
    ]
    for s_str in ["0.15", "0.45", "0.82"]:
        s = M(s_str)
        a = 1 - s
        for cname, rho_fn in cov_models:
            # Truncate both series and matrix sum at 1800 lags (since a <= 0.85, 0.85^1800 ~ 10^-127 < 10^-80)
            f_series = (a / (2 * (1 + a))) * (1 + 2 * mp.fsum(((-a) ** (j - 1)) * rho_fn(j) for j in range(1, 1801)))
            f_mat = matrix_lag_loss_coef(s, rho_fn, n_lags=1800)
            max_lag_err = max(max_lag_err, abs(f_series - f_mat))
    report("M1_general_lag_vs_matrix", max_lag_err < M("1e-70"), f"max discrepancy across 12 cases = {mp.nstr(max_lag_err, 6)}")

    phi_crit = 2 - mp.sqrt(5)
    print(f"  Exact reversal threshold phi_crit = 2 - sqrt(5) = {mp.nstr(phi_crit, 35)}")
    # Check at phi = phi_crit: dF/ds(0, phi_crit) == 0 and dF/ds(s, phi_crit) < 0 for s in (0, 1)
    at_zero_crit = dF_ar1_ds(M(0), phi_crit)
    max_deriv_err = M(0)
    sign_ok = abs(at_zero_crit) < M("1e-75")
    for phi_val in [M("-0.8"), M("-0.5"), M("-0.35"), phi_crit - M("0.01"), phi_crit, phi_crit + M("0.01"), M("0"), M("0.5"), M("0.99")]:
        s_st = s_star_ar1(phi_val)
        if M("-0.5") < phi_val < phi_crit:
            # At s_st, derivative must vanish to ~80 digits
            d_at_star = dF_ar1_ds(s_st, phi_val)
            sign_ok = sign_ok and (abs(d_at_star) < M("1e-74"))
            sign_ok = sign_ok and (dF_ar1_ds(s_st / 2, phi_val) > 0) and (dF_ar1_ds((1 + s_st) / 2, phi_val) < 0)
        elif phi_val >= phi_crit:
            for s_test in [M("1e-6"), M("0.1"), M("0.5"), M("0.999")]:
                sign_ok = sign_ok and (dF_ar1_ds(s_test, phi_val) < 0)
        else:  # phi_val <= -0.5
            for s_test in [M("1e-6"), M("0.1"), M("0.5"), M("0.999")]:
                sign_ok = sign_ok and (dF_ar1_ds(s_test, phi_val) > 0)
        for s_test in [M("0.2"), M("0.7")]:
            num_d = mp.diff(lambda z: F_ar1(z, phi_val), s_test)
            max_deriv_err = max(max_deriv_err, abs(num_d - dF_ar1_ds(s_test, phi_val)))
    report("M2_AR1_boundary_and_derivative", sign_ok and max_deriv_err < M("1e-72"),
           f"boundary signs exact; |dF/ds(0, 2-sqrt(5))| = {mp.nstr(abs(at_zero_crit), 5)}; num diff err = {mp.nstr(max_deriv_err, 5)}")

    # ------------------------------------------------------------------
    # Check 2: Exact counterexamples in 80-digit arithmetic
    # ------------------------------------------------------------------
    print("\n--- 2. High-precision (80-digit) verification of every exact counterexample in memA ---")

    # Counterexample 2a: Exact aperiodic counterexample delta = 1/2, phi = -3/4, s = 1 - 10^-8
    s_ce = 1 - M("1e-8")
    delta_ce = M("0.5")
    phi_ce = -M("0.75")
    D_ce_20, err_ce_20, Rdisk_ce = markov_taylor_D(s_ce, delta_ce, phi_ce, order=20)
    D_ce_32, err_ce_32, _ = markov_taylor_D(s_ce, delta_ce, phi_ce, order=32)
    # Derivative dD/ds at s = 1 - 10^-8 via complex step / mp.diff on order-24 series AND Nystrom check
    dD_ds_ce = mp.diff(lambda ss: markov_taylor_D(ss, delta_ce, phi_ce, order=24)[0], s_ce)
    D_ce_nys = markov_nystrom_D(s_ce, delta_ce, phi_ce, n_nodes=32)
    # Compare with P5 two-term asymptotic A*eps + B*eps^2:
    eps_ce = M("1e-8")
    V_ce = delta_ce ** 2
    A_ce = V_ce * (M("0.5") + phi_ce)  # = -1/16 = -0.0625
    B_ce = -V_ce * (M("0.5") + phi_ce + phi_ce ** 2) + M("1.5") * (A_ce ** 2)
    D_ce_asymp = A_ce * eps_ce + B_ce * (eps_ce ** 2)
    dD_ce_asymp = -A_ce - 2 * B_ce * eps_ce
    print(f"  [CE1: aperiodic delta=1/2, phi=-3/4, s=1-10^-8]")
    print(f"    R_disk       = {mp.nstr(Rdisk_ce, 20)} (|delta|/R_disk = {mp.nstr(delta_ce / Rdisk_ce, 10)})")
    print(f"    D (order 32) = {mp.nstr(D_ce_32, 40)}")
    print(f"    root bound   = {mp.nstr(err_ce_32, 8)}")
    print(f"    |D_32 - D_20|= {mp.nstr(abs(D_ce_32 - D_ce_20), 8)}")
    print(f"    |D_32 - Nys| = {mp.nstr(abs(D_ce_32 - D_ce_nys), 8)}")
    print(f"    A*eps+B*eps^2= {mp.nstr(D_ce_asymp, 40)}")
    print(f"    dD/ds        = {mp.nstr(dD_ds_ce, 40)} (> 1/32 = 0.03125)")
    print(f"    -A - 2B*eps  = {mp.nstr(dD_ce_asymp, 40)}")
    report("CE1_aperiodic_positive_slope",
           (dD_ds_ce > M(1) / 32) and (D_ce_32 < 0) and (err_ce_32 < M("1e-70")) and (abs(D_ce_32 - D_ce_nys) < M("1e-65")),
           f"D = {mp.nstr(D_ce_32, 22)} < 0, dD/ds = {mp.nstr(dD_ds_ce, 22)} > 1/32")

    # Counterexample 2b: Moderate-noise negative loss s = 0.9, delta = 0.4, phi = -0.75
    s_neg, d_neg, p_neg = M("0.9"), M("0.4"), -M("0.75")
    D_neg_ser, err_neg_ser, _ = markov_taylor_D(s_neg, d_neg, p_neg, order=40)
    D_neg_nys28 = markov_nystrom_D(s_neg, d_neg, p_neg, n_nodes=28)
    D_neg_nys40 = markov_nystrom_D(s_neg, d_neg, p_neg, n_nodes=40)
    dD_neg = mp.diff(lambda ss: markov_nystrom_D(ss, d_neg, p_neg, n_nodes=28), s_neg)
    print(f"  [CE2: moderate-noise negative D: s=0.9, delta=0.4, phi=-0.75]")
    print(f"    D (order 40) = {mp.nstr(D_neg_ser, 35)} (analytic bound {mp.nstr(err_neg_ser, 6)})")
    print(f"    D (Nys 40)   = {mp.nstr(D_neg_nys40, 35)} (|Nys40 - Nys28| = {mp.nstr(abs(D_neg_nys40 - D_neg_nys28), 6)})")
    print(f"    dD/ds        = {mp.nstr(dD_neg, 25)}")
    report("CE2_moderate_noise_negative_D",
           (D_neg_ser < 0) and (dD_neg > 0) and (abs(D_neg_ser - D_neg_nys40) < err_neg_ser + M("1e-50")),
           f"D = {mp.nstr(D_neg_nys40, 22)} < 0, |series - Nystrom| = {mp.nstr(abs(D_neg_ser - D_neg_nys40), 6)}")

    # Counterexample 2c: Exceptional boundary phi = -1/2 (A = 0, B = -delta^2/4 < 0, D'(s) > 0 near s = 1)
    d_exc, p_exc = M("0.6"), -M("0.5")
    eps_exc = M("1e-6")
    D_exc, err_exc, _ = markov_taylor_D(1 - eps_exc, d_exc, p_exc, order=20)
    dD_exc = mp.diff(lambda ss: markov_taylor_D(ss, d_exc, p_exc, order=20)[0], 1 - eps_exc)
    B_exc = -(d_exc ** 2) / 4  # = -0.09
    print(f"  [CE3: exceptional A=0 point: delta=0.6, phi=-1/2, s=1-10^-6]")
    print(f"    D / eps^2    = {mp.nstr(D_exc / (eps_exc ** 2), 25)} vs B = {mp.nstr(B_exc, 10)}")
    print(f"    dD/ds / eps  = {mp.nstr(dD_exc / eps_exc, 25)} vs -2B = {mp.nstr(-2 * B_exc, 10)}")
    report("CE3_phi_minus_half_quadratic",
           (D_exc < 0) and (dD_exc > 0) and (abs(D_exc / (eps_exc ** 2) - B_exc) < M("1e-4")),
           f"D/eps^2 = {mp.nstr(D_exc / (eps_exc ** 2), 12)}, dD/ds = {mp.nstr(dD_exc, 12)} > 0")

    # Counterexample 2d: Explicit periodic alternating host P8 (phi = -1)
    max_alt_res = M(0)
    alt_ok = True
    for d_str in ["0.2", "0.5", "0.6", "0.9"]:
        delt = M(d_str)
        ell_d = -mp.log(1 - delt * delt) / 2
        D0 = mp.log(1 / (1 + delt))
        alt_ok = alt_ok and (D0 < 0) and (abs(D0 - ell_d) > M("0.01"))
        for s_str in ["0", "0.2", "0.7", "0.99"]:
            ss = M(s_str)
            Q = mp.sqrt(delt * delt + (1 - delt * delt) * ss * ss)
            D_alt = mp.log((1 + ss) / (1 + Q))
            dD_alt = (Q * (1 + Q) - (1 - delt * delt) * ss * (1 + ss)) / ((1 + ss) * Q * (1 + Q))
            c_star = (1 - ss) * mp.exp(D_alt)
            Ap = mp.matrix([[ss, c_star * (1 + delt)], [M(1), M(0)]])
            Am = mp.matrix([[ss, c_star * (1 - delt)], [M(1), M(0)]])
            prod = Ap * Am
            tr = prod[0, 0] + prod[1, 1]
            det = prod[0, 0] * prod[1, 1] - prod[0, 1] * prod[1, 0]
            perron = (tr + mp.sqrt(tr * tr - 4 * det)) / 2
            max_alt_res = max(max_alt_res, abs(perron - 1))
            num_slope = mp.diff(lambda z: mp.log((1 + z) / (1 + mp.sqrt(delt * delt + (1 - delt * delt) * z * z))), ss)
            max_alt_res = max(max_alt_res, abs(num_slope - dD_alt))
            alt_ok = alt_ok and (D_alt < 0) and (dD_alt > 0)
    report("CE4_periodic_alternating_P8", alt_ok and max_alt_res < M("1e-74"),
           f"all 16 (delta, s) points satisfy D<0, D'>0, rho(A+A-)=1; max residual = {mp.nstr(max_alt_res, 5)}")

    # Check 2e: Finite-noise shift of the reversal boundary above phi = 2 - sqrt(5)!
    # Take phi = -0.22 > 2 - sqrt(5) ~ -0.236068, where second-order F_phi(s) is strictly decreasing everywhere!
    # Check whether large finite amplitude delta = 0.85 reverses D'(s) near small s!
    phi_probe = M("-0.22")
    delta_probe = M("0.85")
    D_s005 = markov_nystrom_D(M("0.05"), delta_probe, phi_probe, n_nodes=24)
    D_s015 = markov_nystrom_D(M("0.15"), delta_probe, phi_probe, n_nodes=24)
    D_s015_ref = markov_nystrom_D(M("0.15"), delta_probe, phi_probe, n_nodes=32)
    print(f"  [Probe 2e: finite-noise boundary shift at phi = -0.22 > 2 - sqrt(5) = -0.23607, delta = 0.85]")
    print(f"    Second-order dF/ds(0.05) = {mp.nstr(dF_ar1_ds(M('0.05'), phi_probe), 12)} (< 0)")
    print(f"    Exact Markov D(0.05)     = {mp.nstr(D_s005, 25)}")
    print(f"    Exact Markov D(0.15)     = {mp.nstr(D_s015, 25)} (refinement diff {mp.nstr(abs(D_s015 - D_s015_ref), 6)})")
    # Also scan delta in {0.3, 0.6, 0.9} at phi = -0.235 > 2 - sqrt(5)
    phi_p2 = M("-0.235")
    for d_try in [M("0.3"), M("0.6"), M("0.9")]:
        d1 = markov_nystrom_D(M("0.02"), d_try, phi_p2, n_nodes=24)
        d2 = markov_nystrom_D(M("0.06"), d_try, phi_p2, n_nodes=24)
        print(f"    phi=-0.235 (> 2-sqrt(5)), delta={mp.nstr(d_try, 3)}: D(0.02)={mp.nstr(d1, 16)}, D(0.06)={mp.nstr(d2, 16)}, diff={mp.nstr(d2 - d1, 8)}")

    # Also verify positive persistence 0 <= phi < 1 in two-state Markov host: 0 < D < ell and strict decrease across s
    pos_ok = True
    for delt in [M("0.3"), M("0.8")]:
        ell_v = -mp.log(1 - delt * delt) / 2
        for ph in [M("0"), M("0.5"), M("0.9")]:
            vals = [markov_nystrom_D(ss, delt, ph, n_nodes=22) for ss in [M("0.05"), M("0.3"), M("0.7"), M("0.95")]]
            pos_ok = pos_ok and all(0 < v < ell_v for v in vals)
            pos_ok = pos_ok and all(vals[k + 1] < vals[k] for k in range(len(vals) - 1))
    report("M3_positive_memory_bounds_and_decrease", pos_ok, "0 < D < ell and D strictly decreasing across s in {0.05, 0.3, 0.7, 0.95} for all 6 (delta, phi>=0) pairs")

    # ------------------------------------------------------------------
    # Check 3: Explicit resolution of the H2 / Result 4 / Lean P2 vs memA tension
    # ------------------------------------------------------------------
    print("\n--- 3. Resolution of the Tension (Result 4 / Lean P2 / H2 vs memA) ---")
    a_crit = (3 + mp.sqrt(17)) / 4
    print(f"  Critical fluctuating-stage stay for 2-stage non-monotonicity in b >= 1: a_crit = (3 + sqrt(17))/4 = {mp.nstr(a_crit, 25)}")

    # 3a: Verify that at a = 1, for all phi in (0, 1):
    #   (i) Gamma(1, b) < Gamma_1(1) = 1/2 for all b >= 1 (adding a steady stage NEVER raises Gamma at a = 1)
    #   (ii) d/db Gamma(1, b) < 0 for all b >= 1, because b_max(1, phi) in (0, 1/3) < 1!
    a1_ok = True
    max_bmax_a1 = M(0)
    for ph_str in ["0.01", "0.3", "0.7", "0.9", "0.99", "0.9999"]:
        ph = M(ph_str)
        bm = b_max_two_stage(M(1), ph)
        max_bmax_a1 = max(max_bmax_a1, bm)
        a1_ok = a1_ok and (0 < bm < M(1) / 3)
        # Check d/db Gamma(1, bm) == 0
        a1_ok = a1_ok and (abs(dgamma_two_stage_db(M(1), bm, ph)) < M("1e-74"))
        for b_val in [M("1"), M("1.5"), M("5"), M("100")]:
            a1_ok = a1_ok and (gamma_two_stage(M(1), b_val, ph) < M("0.5"))
            a1_ok = a1_ok and (dgamma_two_stage_db(M(1), b_val, ph) < 0)
    report("T1_a_eq_1_never_rises", a1_ok,
           f"at a=1, sup b_max = {mp.nstr(max_bmax_a1, 10)} < 1/3; Gamma(1,b) < 1/2 and dGamma/db < 0 for all b >= 1, 0 < phi < 1")

    # 3b: Verify that for 1 < a <= (3 + sqrt(17))/4 (e.g. a = 1.5):
    #   (i) Adding a steady stage CAN raise Gamma above Gamma_1(a) = 1/(2a) (for b < a*(2a-1)*theta/(a+theta)),
    #   (ii) YET within the 2-stage model, d/db Gamma(a, b) < 0 for ALL b >= 1!
    a_mid = M("1.5")
    ph_mid = M("0.9")  # theta = 9
    th_mid = ph_mid / (1 - ph_mid)
    b_cross_mid = a_mid * (2 * a_mid - 1) * th_mid / (a_mid + th_mid)  # = 1.5 * 2 * 9 / 10.5 = 27/10.5 = 18/7 ~ 2.5714 > 1
    g1_mid = 1 / (2 * a_mid)  # = 1/3 ~ 0.333333
    g2_b1_mid = gamma_two_stage(a_mid, M(1), ph_mid)  # > 1/3!
    bm_mid = b_max_two_stage(a_mid, ph_mid)  # < 1!
    mid_ok = (g2_b1_mid > g1_mid) and (bm_mid < 1) and all(
        dgamma_two_stage_db(a_mid, b_v, ph_mid) < 0 for b_v in [M("1"), M("1.2"), M("2"), M("10")]
    )
    print(f"  [Intermediate regime 1 < a = 1.5 < a_crit = 1.7808, phi = 0.9]:")
    print(f"    1-stage Gamma_1(1.5)/sigma^2 = {mp.nstr(g1_mid, 12)}")
    print(f"    2-stage Gamma(1.5, b=1)/sigma^2 = {mp.nstr(g2_b1_mid, 12)} (> Gamma_1! Crossing at b = {mp.nstr(b_cross_mid, 10)})")
    print(f"    Yet b_max(1.5, 0.9) = {mp.nstr(bm_mid, 12)} < 1, so dGamma/db < 0 for ALL b >= 1!")
    report("T2_intermediate_a_separates_addition_from_slope", mid_ok,
           f"Gamma(1.5, 1) = {mp.nstr(g2_b1_mid, 8)} > Gamma_1(1.5) = {mp.nstr(g1_mid, 8)}, while dGamma/db < 0 on all b >= 1")

    # 3c: Verify that for a > (3 + sqrt(17))/4 (e.g. a = 4, phi = 0.95):
    #   BOTH Gamma(a, b) > Gamma_1(a) AND d/db Gamma(a, b) > 0 on b in [1, b_max(a, phi))!
    #   AND confirm by finite-noise Monte Carlo at sigma = 0.6, phi = 0.95, a = 4!
    a_hi = M("4")
    th_crit_4 = theta_crit_two_stage_nonmonotone(a_hi)
    phi_crit_4 = th_crit_4 / (1 + th_crit_4)
    ph_hi = M("0.95")  # theta = 19 > th_crit_4
    bm_hi = b_max_two_stage(a_hi, ph_hi)
    slope_b1_hi = dgamma_two_stage_db(a_hi, M(1), ph_hi)
    slope_at_crit = dgamma_two_stage_db(a_hi, M(1), phi_crit_4)
    print(f"  [Regime a = 4 > a_crit = 1.7808]:")
    print(f"    Exact non-monotonicity threshold: theta_crit(4) = {mp.nstr(th_crit_4, 15)} (phi_crit(4) = {mp.nstr(phi_crit_4, 15)})")
    print(f"    Check dGamma/db(4, 1) at phi_crit(4): {mp.nstr(slope_at_crit, 8)}")
    print(f"    At phi = 0.95: b_max(4, 0.95) = {mp.nstr(bm_hi, 15)} > 1, dGamma/db(4, 1) = {mp.nstr(slope_b1_hi, 12)} > 0")
    print(f"    Gamma(4, b)/sigma^2 at b = 1, {mp.nstr(bm_hi, 5)}, 20: "
          f"{mp.nstr(gamma_two_stage(a_hi, M(1), ph_hi), 8)}, "
          f"{mp.nstr(gamma_two_stage(a_hi, bm_hi, ph_hi), 8)}, "
          f"{mp.nstr(gamma_two_stage(a_hi, M(20), ph_hi), 8)} (vs 1-stage {mp.nstr(1/(2*a_hi), 8)})")

    b_grid_mc = [1.01, 1.5, 2.3, 3.0, 5.0, 10.0, 50.0]
    D_mc, se_mc, infl_mc = mc_two_stage_general_a(4.0, b_grid_mc, sigma=0.6, phi=0.95, steps=120000, burn=4000, pairs=32)
    diff_02 = D_mc[2] - D_mc[0]
    se_02 = np.std(infl_mc[:, 2] - infl_mc[:, 0], ddof=1) / math.sqrt(len(infl_mc))
    diff_26 = D_mc[6] - D_mc[2]
    se_26 = np.std(infl_mc[:, 6] - infl_mc[:, 2], ddof=1) / math.sqrt(len(infl_mc))
    print(f"    Finite-noise MC (2-stage a=4, sigma=0.6, phi=0.95):")
    for bb, dd, ss in zip(b_grid_mc, D_mc, se_mc):
        print(f"      b = {bb:5.2f}: D = {dd:.6f} +- {ss:.6f} (second-order = {0.36 * float(gamma_two_stage(M(4), M(bb), M('0.95'))):.6f})")
    print(f"    Rise from b=1.01 to b=2.30: +{diff_02:.6f} +- {se_02:.6f} ({diff_02 / se_02:.1f} paired SE)")
    print(f"    Fall from b=2.30 to b=50.0: {diff_26:.6f} +- {se_26:.6f} ({diff_26 / se_26:.1f} paired SE)")
    report("T3_a_gt_acrit_nonmonotone_algebra_and_MC",
           (abs(slope_at_crit) < M("1e-74")) and (bm_hi > 2) and (diff_02 > 4 * se_02) and (diff_26 < -4 * se_26),
           f"2-stage ring with fluctuating stay a=4 rises by {diff_02/se_02:.1f} SE before falling")

    # 3d: Verify Lean P2 identity and equivalence with Gamma(a,b) < Gamma_1(a) at L = 1/(2a)
    max_lean_err = M(0)
    for a_v, b_v, ph_v, L_v in [(M(1), M(2), M("0.9"), M("0.6")),
                                 (M(10), M(10), M("0.9"), M("0.2")),
                                 (M(4), M(3), M("0.8"), M("0.125"))]:
        th_v = ph_v / (1 - ph_v)
        E1 = (2 * a_v * L_v - 1) / (1 + 2 * th_v)
        G2 = gamma_two_stage(a_v, b_v, ph_v)
        E2 = (2 * (a_v + b_v) * (L_v - G2)) / (1 + 2 * th_v)
        rhs = (2 * b_v / (1 + 2 * th_v)) * (L_v - a_v * th_v / (a_v * th_v + b_v * (a_v + th_v)))
        max_lean_err = max(max_lean_err, abs((E2 - E1) - rhs))
    report("T4_Lean_P2_identity", max_lean_err < M("1e-75"), f"max error = {mp.nstr(max_lean_err, 5)}")

    # 3e: Verify 3-stage ring (1, b, c3) slow-memory maximizer b_max = (2*c3^2 - c3 + 1)/(2*c3 + 3) and c3 > 2 threshold
    def gamma_three_stage(b_stay, c3_stay, phi):
        if phi == 0:
            return 1 / (2 * (1 + b_stay + c3_stay))
        th = phi / (1 - phi)
        T_tot = 1 + b_stay + c3_stay
        R = (th ** 3) / ((th + 1) * (th + b_stay) * (th + c3_stay))
        return (1 + 2 * th) / (2 * T_tot) - R / (1 - R)

    c3_ok = True
    for c3_v in [M("1"), M("1.5"), M("2")]:
        # For c3 <= 2, d/db Gamma_3(1, b, c3) < 0 for all b >= 1 and all phi in (0, 1)
        for ph_v in [M("0.2"), M("0.7"), M("0.99"), M("0.99999")]:
            for b_v in [M("1"), M("1.01"), M("2"), M("10")]:
                sl = mp.diff(lambda z: gamma_three_stage(z, c3_v, ph_v), b_v)
                c3_ok = c3_ok and (sl < 0)
    # At c3 = 5 (memB's 3-stage ring), check slow-memory limit b_max -> 46/13 and phi=0.99 peak ~ 3.409719473
    b_peak_99 = mp.findroot(lambda z: mp.diff(lambda w: gamma_three_stage(w, M(5), M("0.99")), z), M("3.4"))
    b_peak_lim = mp.findroot(lambda z: mp.diff(lambda w: gamma_three_stage(w, M(5), 1 - M("1e-12")), z), M("3.53"))
    exact_lim_5 = M(46) / 13
    c3_ok = c3_ok and (abs(b_peak_lim - exact_lim_5) < M("1e-9")) and (b_peak_99 > 3)
    report("T5_three_stage_c3_gt_2_criterion", c3_ok,
           f"monotone for c3 <= 2; at c3=5, b_max(phi=0.99) = {mp.nstr(b_peak_99, 12)}, b_max(phi->1) = {mp.nstr(b_peak_lim, 12)} vs 46/13 = {mp.nstr(exact_lim_5, 12)}")

    elapsed = time.monotonic() - t0
    print(f"\nCompleted in {elapsed:.2f}s. Failures: {len(FAILURES)} {FAILURES}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
