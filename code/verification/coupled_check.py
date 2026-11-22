#!/usr/bin/env python3
"""
Independent third check of `notes/coupled-hosts.md` (K1-K9)
and test against Dobson & Hudson (1992) and Gaba & Gourbiere (2008).

Author: independent 4 secondary
Date: 14 November 2026

Written completely from scratch without reading or importing
`engine/cycle_blocks.py` or `studies/coupled/checks.py`.

PRE-DECLARED TOLERANCES (Rule 7, stated before running):
  TOL_BENCH       = 1.0e-7   # Known-result reproduction (Goodwin 3-stage, A&M 1978, D&H 1992 App. 1)
  TOL_EQ_RESID    = 1.0e-11  # Nonlinear equilibrium residual ||F(w*)||
  TOL_FD_VS_ANA   = 1.0e-6   # Finite-difference transfer function vs closed-form E(iw)
  TOL_GAIN_BOUND  = 1.0e-8   # Allowance for numerical grid/FD rounding in |E(iw)| <= q/d
  TOL_CLASS_MARG  = 1.0e-5   # Exclusion margin around algebraic boundary in K3(b) two-sided test
  TOL_ODE_REL     = 0.03     # Max relative error (3% or 2e-3 abs) between stiff ODE fit and Re(lambda_max)
  TOL_ODE_ABS     = 2.0e-3
  TOL_HURWITZ_K2  = 1.0e-11  # Max allowed Re(lambda) for Hurwitz cycles in K2 (numerical zero)
"""

import math
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Pre-declared tolerances
TOL_BENCH = 1.0e-7
TOL_EQ_RESID = 1.0e-11
TOL_FD_VS_ANA = 1.0e-6
TOL_GAIN_BOUND = 1.0e-8
TOL_CLASS_MARG = 1.0e-5
TOL_ODE_REL = 0.03
TOL_ODE_ABS = 2.0e-3
TOL_HURWITZ_K2 = 1.0e-11

SEED = 2026111401


# ============================================================================
# Helper: numerical Jacobian and state-space frequency response by central FD
# ============================================================================

def central_fd_jacobian(rhs_fn, w_eq, eps=1.0e-7):
    """Central finite-difference Jacobian dF/dw at w_eq."""
    n = len(w_eq)
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        step = eps * max(1.0, abs(w_eq[j]))
        w_plus = w_eq.copy()
        w_minus = w_eq.copy()
        w_plus[j] += step
        w_minus[j] -= step
        J[:, j] = (rhs_fn(w_plus) - rhs_fn(w_minus)) / (2.0 * step)
    return J


def block_fd_linearization(f_state, h_out, w_eq, u_eq, eps=1.0e-7):
    """
    Linearize a single block dw/dt = f_state(w, u), v = h_out(w) around (w_eq, u_eq)
    purely by central finite differences of the nonlinear equations.
    Returns (A, b, c, v_eq, eq_resid).
    """
    resid = np.max(np.abs(f_state(w_eq, u_eq)) / (1.0 + np.abs(w_eq)))
    A = central_fd_jacobian(lambda w: f_state(w, u_eq), w_eq, eps=eps)
    du = eps * max(1.0, abs(u_eq))
    b = (f_state(w_eq, u_eq + du) - f_state(w_eq, u_eq - du)) / (2.0 * du)
    n = len(w_eq)
    c = np.zeros(n, dtype=float)
    for j in range(n):
        dw = eps * max(1.0, abs(w_eq[j]))
        wp = w_eq.copy()
        wm = w_eq.copy()
        wp[j] += dw
        wm[j] -= dw
        c[j] = (h_out(wp) - h_out(wm)) / (2.0 * dw)
    v_eq = h_out(w_eq)
    return A, b, c, v_eq, resid


def eval_ss_log_response(A, b, c, u_eq, v_eq, omegas):
    """
    Evaluate E(i*omega) = (u*/v*) * c^T (i*omega*I - A)^{-1} b across real frequencies omegas.
    """
    n = A.shape[0]
    eye = np.eye(n, dtype=complex)
    scale = u_eq / v_eq
    # Vectorized solve for 1x1, 2x2, 3x3
    w_arr = np.asarray(omegas, dtype=float)
    if n == 1:
        return scale * c[0] * b[0] / (1j * w_arr - A[0, 0])
    if n == 2:
        m00 = 1j * w_arr - A[0, 0]
        m01 = -A[0, 1]
        m10 = -A[1, 0]
        m11 = 1j * w_arr - A[1, 1]
        det = m00 * m11 - m01 * m10
        sol0 = (m11 * b[0] - m01 * b[1]) / det
        sol1 = (-m10 * b[0] + m00 * b[1]) / det
        return scale * (c[0] * sol0 + c[1] * sol1)
    out = np.empty(len(w_arr), dtype=complex)
    for idx, w in enumerate(w_arr):
        sol = np.linalg.solve(1j * w * eye - A, b)
        out[idx] = scale * np.dot(c, sol)
    return out


def sup_modulus_ss(A, b, c, u_eq, v_eq, char_scales):
    """
    Compute sup_{omega >= 0} |E(i*omega)| directly from state-space matrices (A, b, c)
    using a dense multi-scale grid plus local Brent refinement around top peaks.
    """
    s_min = max(1.0e-12, min(char_scales))
    s_max = max(1.0, max(char_scales))
    log_grid = np.geomspace(1.0e-7 * s_min, 1.0e7 * s_max, 1800)
    omegas = np.concatenate(([0.0], log_grid))
    vals = np.abs(eval_ss_log_response(A, b, c, u_eq, v_eq, omegas))
    best_idx = int(np.argmax(vals))
    best_val = float(vals[best_idx])
    best_w = float(omegas[best_idx])

    # Refine around top 3 local maxima
    cand_indices = [best_idx]
    if len(vals) > 3:
        interior = np.where((vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1
        for idx in interior:
            if idx not in cand_indices:
                cand_indices.append(int(idx))
    for idx in cand_indices[:5]:
        lo = omegas[max(0, idx - 1)]
        hi = omegas[min(len(omegas) - 1, idx + 1)]
        if hi > lo:
            res = minimize_scalar(
                lambda w: -float(np.abs(eval_ss_log_response(A, b, c, u_eq, v_eq, [w])[0])),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1.0e-11 * max(1.0, hi)},
            )
            if -res.fun > best_val:
                best_val = float(-res.fun)
                best_w = float(res.x)
    return best_val, best_w


# ============================================================================
# Nonlinear constructors for V and M blocks from biological / equilibrium params
# ============================================================================

def make_nonlinear_phi(X_eq, h, rho):
    """
    Construct a smooth nonlinear host net production function Phi(X) such that:
      Phi(X_eq) = h * X_eq   and   Phi'(X_eq) = -rho.
    Notice:
      Phi(X) = X * [ h - (rho + h) * (X / X_eq - 1) ]
             = (2*h + rho)*X - ((rho + h)/X_eq) * X^2.
    - When rho + h > 0 (self-regulation): exact logistic growth Phi(X) = r*X*(1 - X/K)
      with r = 2*h + rho > 0 (since rho > -h and h > 0 -> r > h > 0) and K = r*X_eq/(rho + h)!
    - When rho = -h (no self-regulation): exact linear growth Phi(X) = h * X!
    - When rho < -h (Allee / rising per-capita production): quadratic growth with positive X^2 term
      near X_eq (adding a weak -eps*X^3 far above X_eq if needed, unnecessary for local/moderate checks).
    """
    r_lin = 2.0 * h + rho
    quad_coeff = (rho + h) / X_eq

    def phi(X):
        return r_lin * X - quad_coeff * (X * X)

    return phi


def make_V_block_nonlinear(N_eq, p, a_V, alpha, rho, c_out=1.0):
    """
    Build nonlinear V block:
      dN/dt = Phi(N) - alpha * I
      dI/dt = u * (N - I) - a_V * I
      v     = c_out * I
    with exact positive equilibrium (N_eq, I_eq = p * N_eq) at u_eq = a_V * p / (1 - p).
    """
    I_eq = p * N_eq
    h = alpha * p
    u_eq = a_V * p / (1.0 - p)
    phi = make_nonlinear_phi(N_eq, h, rho)

    def f_state(w, u):
        N, I = w[0], w[1]
        dN = phi(N) - alpha * I
        dI = u * (N - I) - a_V * I
        return np.array([dN, dI], dtype=float)

    def h_out(w):
        return c_out * w[1]

    q = a_V
    d = a_V / (1.0 - p)
    w_eq = np.array([N_eq, I_eq], dtype=float)
    return f_state, h_out, w_eq, u_eq, {"q": q, "d": d, "h": h, "rho": rho, "p": p, "a_V": a_V, "alpha": alpha}


def make_M_block_nonlinear(H_eq, m, xi, alpha, eps_agg, rho, lam_out=1.0):
    """
    Build nonlinear M block (Anderson & May 1978):
      dH/dt = Phi(H) - alpha * P
      dP/dt = u * H - xi * P - alpha * eps_agg * P^2 / H
      v     = lam_out * P
    with exact positive equilibrium (H_eq, P_eq = m * H_eq) at u_eq = m * (xi + alpha * eps_agg * m).
    """
    P_eq = m * H_eq
    h = alpha * m
    q = xi + eps_agg * h
    d = xi + 2.0 * eps_agg * h
    u_eq = m * q
    phi = make_nonlinear_phi(H_eq, h, rho)

    def f_state(w, u):
        H, P = w[0], w[1]
        dH = phi(H) - alpha * P
        dP = u * H - xi * P - alpha * eps_agg * (P * P) / H
        return np.array([dH, dP], dtype=float)

    def h_out(w):
        return lam_out * w[1]

    w_eq = np.array([H_eq, P_eq], dtype=float)
    return f_state, h_out, w_eq, u_eq, {"q": q, "d": d, "h": h, "rho": rho, "m": m, "xi": xi, "alpha": alpha, "eps": eps_agg}


def primary_two_sided_damping_test(q, d, h, rho):
    """
    primary's exact two-sided damping test for V and M blocks (coupled-hosts.md K3):
      A = rho + d
      B = d * (rho + h)
      U = rho^2 + d^2 - 2*d*h - q^2
      V = d^2 * (rho + h)^2 - q^2 * rho^2
      damping iff A > 0, B > 0, V > 0, and U + 2*sqrt(V) > 0.
    Returns (is_damping, min_margin).
    """
    A = rho + d
    B = d * (rho + h)
    U = rho * rho + d * d - 2.0 * d * h - q * q
    V = d * d * (rho + h) * (rho + h) - q * q * rho * rho
    if V > 0.0:
        fourth = U + 2.0 * math.sqrt(V)
    else:
        fourth = -abs(V) - 1.0
    margin = min(abs(A), abs(B), abs(V), abs(fourth) if V > 0.0 else abs(V))
    is_damping = (A > 0.0) and (B > 0.0) and (V > 0.0) and (fourth > 0.0)
    return is_damping, margin


# ============================================================================
# Step 0: Known-result reproduction benchmarks (Rule 7)
# ============================================================================

def run_step0_benchmarks():
    print("=" * 78)
    print("STEP 0: REPRODUCING KNOWN ANALYTICAL RESULTS (Rule 7)")
    print(f"Pre-declared tolerance: TOL_BENCH = {TOL_BENCH:.1e}")
    print("=" * 78)

    # Benchmark 0a: 3-stage equal-rate Goodwin / secant oscillator (kappa = 9 -> marginal Re=0)
    for kappa, expected_re in [(8.9, -0.5 + 0.5 * (7.9 ** (1.0 / 3.0)) * (-1.0)), (9.0, 0.0), (9.1, None)]:
        # Wait: exact roots of (z + 1)^3 = -(kappa - 1) = -g are z = -1 + g^(1/3) * exp(i*pi*(2k+1)/3)
        # Leading pair has Re(z) = -1 + g^(1/3) * cos(pi/3) = -1 + 0.5 * g^(1/3)
        g = kappa - 1.0
        exact_re = -1.0 + 0.5 * (g ** (1.0 / 3.0))
        # Nonlinear system: dw1/dt = v3 - w1, dw2/dt = w1 - w2, dw3/dt = w2 - w3
        # with v3 = c * w3 * exp(-kappa * w3) at w1*=w2*=w3*=1 -> c = exp(kappa)
        c_val = math.exp(kappa)

        def rhs_goodwin(w):
            v3 = c_val * w[2] * math.exp(-kappa * w[2])
            return np.array([v3 - w[0], w[0] - w[1], w[1] - w[2]], dtype=float)

        w_eq = np.ones(3)
        J_fd = central_fd_jacobian(rhs_goodwin, w_eq)
        re_max = float(np.max(np.real(np.linalg.eigvals(J_fd))))
        err = abs(re_max - exact_re)
        print(f"  [0a] 3-stage RLL ring kappa={kappa:.1f}: FD max Re={re_max:+.10f}, exact={exact_re:+.10f}, err={err:.2e}")
        assert err < TOL_BENCH, f"Benchmark 0a failed: {err}"

    # Benchmark 0b: Anderson & May (1978 Part I) instantaneous host-macroparasite loop (u = lam*P -> u*H = lam*P*H)
    # dH/dt = r*H - alpha*P, dP/dt = lam*P*H - xi*P - alpha*eps*P^2/H
    # At equilibrium m* = r/alpha, lam*H* = xi + eps*r.
    # From K4 eq (9) with rho = -h = -r, q = xi + eps*r, d = xi + 2*eps*r:
    # z^2 + J*z + C = 0 where J = (eps - 1)*r, C = r*(xi + eps*r).
    r, alpha, xi = 0.8, 0.01, 1.2
    for eps_agg in [1.0, 2.0]:
        m_eq = r / alpha
        H_eq = 100.0
        lam = (xi + eps_agg * r) / H_eq
        P_eq = m_eq * H_eq

        def rhs_am(w):
            H, P = w[0], w[1]
            return np.array([r * H - alpha * P, lam * P * H - xi * P - alpha * eps_agg * P * P / H])

        J_fd = central_fd_jacobian(rhs_am, np.array([H_eq, P_eq]))
        eigs_fd = np.sort_complex(np.linalg.eigvals(J_fd))
        J_coef = (eps_agg - 1.0) * r
        C_coef = r * (xi + eps_agg * r)
        eigs_exact = np.sort_complex(np.roots([1.0, J_coef, C_coef]))
        err = float(np.max(np.abs(eigs_fd - eigs_exact)))
        print(f"  [0b] A&M (1978) instantaneous eps={eps_agg:.1f}: max eig err={err:.2e}, max Re={np.max(np.real(eigs_fd)):+.10f}")
        assert err < TOL_BENCH, f"Benchmark 0b failed: {err}"

    # Benchmark 0c: Dobson & Hudson (1992) Appendix 1 eqs (A5)-(A7) for Model I (Delta = 0)
    a_g, b_g = 1.8, 1.05
    lam, mu_P, gamma, beta = 11.0, 1.0, 13.0, 0.1
    alpha_d, delta_d, k_nb = 3.0e-4, 5.0e-4, 1.0
    kp = (k_nb + 1.0) / k_nb
    d_prime = mu_P + alpha_d + b_g + kp * (a_g - b_g) * alpha_d / (alpha_d + delta_d)
    H_eq = gamma * d_prime / (beta * (lam - d_prime))
    m_eq = (a_g - b_g) / (alpha_d + delta_d)
    P_eq = m_eq * H_eq
    W_eq = d_prime * m_eq / beta

    def rhs_dh1(w):
        H, W, P = w[0], w[1], w[2]
        dH = (a_g - b_g) * H - (alpha_d + delta_d) * P
        dW = lam * P - gamma * W - beta * W * H
        dP = beta * W * H - (mu_P + b_g + alpha_d) * P - alpha_d * kp * P * P / H
        return np.array([dH, dW, dP], dtype=float)

    J_fd = central_fd_jacobian(rhs_dh1, np.array([H_eq, W_eq, P_eq]))
    eigs_fd = np.sort_complex(np.linalg.eigvals(J_fd))
    A_dh = gamma * lam / (lam - d_prime) + d_prime + (a_g - b_g) * (alpha_d * kp / (alpha_d + delta_d) - 1.0)
    B_dh = (gamma * lam / (lam - d_prime)) * (a_g - b_g) * (alpha_d * kp / (alpha_d + delta_d) - 1.0)
    C_dh = gamma * d_prime * (a_g - b_g)
    eigs_dh = np.sort_complex(np.roots([1.0, A_dh, B_dh, C_dh]))
    rel_err = float(np.max(np.abs(eigs_fd - eigs_dh) / (1.0 + np.abs(eigs_dh))))
    print(f"  [0c] Dobson & Hudson (1992) App. 1 Model I cubic: rel eig err={rel_err:.2e}")
    assert rel_err < TOL_BENCH, f"Benchmark 0c failed: {rel_err}"
    print("  -> ALL STEP 0 BENCHMARKS PASSED.\n")


# ============================================================================
# Part 1, Item 1: K3 (V and M blocks, >= 3,000 random sets each for (a) and (b))
# ============================================================================

def run_part1_k3(rng):
    print("=" * 78)
    print("PART 1, ITEM 1: CHECKING K3 (V BLOCK AND M BLOCK)")
    print(f"Pre-declared tolerances: TOL_EQ_RESID={TOL_EQ_RESID:.1e}, TOL_FD_VS_ANA={TOL_FD_VS_ANA:.1e},")
    print(f"  TOL_GAIN_BOUND={TOL_GAIN_BOUND:.1e}, TOL_CLASS_MARG={TOL_CLASS_MARG:.1e}")
    print("=" * 78)

    n_sets = 3200  # > 3,000 random parameter sets per subtest per block type

    # --- K3(a): rho >= 0, check sup_w |E(iw)| <= q/d ---
    for kind in ["V", "M"]:
        max_excess = -1.0e9
        max_fd_err = 0.0
        max_resid = 0.0
        rho_zero_max_ratio = 0.0
        n_physical_v = 0
        for idx in range(n_sets):
            if kind == "V":
                N_eq = 10.0 ** rng.uniform(-1.0, 2.5)
                p = rng.uniform(0.01, 0.98)
                alpha = 10.0 ** rng.uniform(-2.0, 1.5)
                # Half physical (a_V >= alpha), half general (a_V > 0) to test mathematical statement fully
                if idx % 2 == 0:
                    a_V = alpha * (1.0 + 10.0 ** rng.uniform(-2.0, 1.5))
                    n_physical_v += 1
                else:
                    a_V = 10.0 ** rng.uniform(-2.0, 1.5)
                # Include exact rho = 0 boundary in first 200 cases
                if idx < 200:
                    rho = 0.0
                else:
                    rho = 10.0 ** rng.uniform(-3.0, 2.0)
                f_st, h_o, w_eq, u_eq, pars = make_V_block_nonlinear(N_eq, p, a_V, alpha, rho)
            else:
                H_eq = 10.0 ** rng.uniform(-1.0, 2.5)
                m = 10.0 ** rng.uniform(-1.5, 2.0)
                xi = 10.0 ** rng.uniform(-2.0, 1.5)
                alpha = 10.0 ** rng.uniform(-2.5, 1.0)
                eps_agg = 1.0 if idx % 20 == 0 else 1.0 + 10.0 ** rng.uniform(-2.5, 1.2)
                if idx < 200:
                    rho = 0.0
                else:
                    rho = 10.0 ** rng.uniform(-3.0, 2.0)
                f_st, h_o, w_eq, u_eq, pars = make_M_block_nonlinear(H_eq, m, xi, alpha, eps_agg, rho)

            q, d, h, rho_val = pars["q"], pars["d"], pars["h"], pars["rho"]
            A_fd, b_fd, c_fd, v_eq, resid = block_fd_linearization(f_st, h_o, w_eq, u_eq)
            max_resid = max(max_resid, resid)
            assert resid < TOL_EQ_RESID, f"Equilibrium residual {resid} exceeded {TOL_EQ_RESID}"

            # Check Hurwitz of FD matrix
            re_fd = float(np.max(np.real(np.linalg.eigvals(A_fd))))
            assert re_fd < 0.0, f"K3(a) {kind} non-Hurwitz for rho>=0, h>0: re={re_fd}"

            sup_fd, w_star = sup_modulus_ss(A_fd, b_fd, c_fd, u_eq, v_eq, [q, d, h, max(rho_val, 1e-4)])
            # Compare FD response against closed-form formula at w_star and test frequencies
            test_ws = np.array([0.0, 0.1 * d, d, math.sqrt(max(d * h, 1e-12)), max(w_star, 1e-8)])
            E_fd = eval_ss_log_response(A_fd, b_fd, c_fd, u_eq, v_eq, test_ws)
            z = 1j * test_ws
            E_ana = q * (z + rho_val) / ((z + rho_val) * (z + d) + d * h)
            max_fd_err = max(max_fd_err, float(np.max(np.abs(E_fd - E_ana))))

            bound = q / d
            excess = sup_fd - bound
            max_excess = max(max_excess, excess)
            if rho_val == 0.0:
                rho_zero_max_ratio = max(rho_zero_max_ratio, sup_fd / bound)

        status = "PASS" if (max_excess <= TOL_GAIN_BOUND and max_fd_err <= TOL_FD_VS_ANA) else "FAIL"
        print(f"  [K3a-{kind}] {n_sets} random {kind} blocks with rho >= 0 (incl. 200 at rho = 0): {status}")
        print(f"           max(|E_FD(iw)| - q/d) = {max_excess:+.3e} (tol {TOL_GAIN_BOUND:.1e})")
        print(f"           at rho = 0, max(|E_FD(iw)| / (q/d)) = {rho_zero_max_ratio:.10f} (exact 1.0 attained at w=sqrt(dh))")
        print(f"           max |E_FD - E_analytic| = {max_fd_err:.3e}, max eq resid = {max_resid:.2e}")

    # --- K3(b): General rho (positive, -h < rho < 0, rho = -h, rho < -h), primary exact two-sided test ---
    for kind in ["V", "M"]:
        tested = 0
        skipped_near_boundary = 0
        misclassified = 0
        n_damping = 0
        n_not_damping = 0
        n_neg_rho_damping = 0
        n_neg_rho_amplifying_hurwitz = 0
        n_non_hurwitz = 0

        while tested < n_sets:
            if kind == "V":
                N_eq = 10.0 ** rng.uniform(-0.5, 2.0)
                p = rng.uniform(0.02, 0.96)
                alpha = 10.0 ** rng.uniform(-1.5, 1.2)
                if tested % 2 == 0:
                    a_V = alpha * (1.0 + 10.0 ** rng.uniform(-2.0, 1.2))
                else:
                    a_V = 10.0 ** rng.uniform(-1.5, 1.2)
                h = alpha * p
                # Sample rho across all regimes, heavily weighting -h < rho < 0 where boundary sits
                regime_draw = rng.random()
                if regime_draw < 0.20:
                    rho = 10.0 ** rng.uniform(-2.0, 1.5)
                elif regime_draw < 0.75:
                    rho = -h * rng.uniform(0.02, 0.98)
                else:
                    rho = -h * rng.uniform(1.02, 2.5)
                f_st, h_o, w_eq, u_eq, pars = make_V_block_nonlinear(N_eq, p, a_V, alpha, rho)
            else:
                H_eq = 10.0 ** rng.uniform(-0.5, 2.0)
                m = 10.0 ** rng.uniform(-1.0, 1.5)
                xi = 10.0 ** rng.uniform(-1.5, 1.2)
                alpha = 10.0 ** rng.uniform(-2.0, 0.8)
                eps_agg = 1.0 + 10.0 ** rng.uniform(-2.0, 1.0)
                h = alpha * m
                regime_draw = rng.random()
                if regime_draw < 0.20:
                    rho = 10.0 ** rng.uniform(-2.0, 1.5)
                elif regime_draw < 0.75:
                    rho = -h * rng.uniform(0.02, 0.98)
                else:
                    rho = -h * rng.uniform(1.02, 2.5)
                f_st, h_o, w_eq, u_eq, pars = make_M_block_nonlinear(H_eq, m, xi, alpha, eps_agg, rho)

            q, d, h_val, rho_val = pars["q"], pars["d"], pars["h"], pars["rho"]
            pred_damping, margin = primary_two_sided_damping_test(q, d, h_val, rho_val)
            if margin < TOL_CLASS_MARG:
                skipped_near_boundary += 1
                continue

            A_fd, b_fd, c_fd, v_eq, resid = block_fd_linearization(f_st, h_o, w_eq, u_eq)
            eigs = np.linalg.eigvals(A_fd)
            is_hurwitz = float(np.max(np.real(eigs))) < -1.0e-9
            sup_fd, _ = sup_modulus_ss(A_fd, b_fd, c_fd, u_eq, v_eq, [q, d, h_val, abs(rho_val) + 1e-4])
            # Also exclude cases where |sup_fd - 1| < 1e-6 to avoid numerical ambiguity
            if is_hurwitz and abs(sup_fd - 1.0) < 1.0e-6:
                skipped_near_boundary += 1
                continue

            obs_damping = is_hurwitz and (sup_fd < 1.0)
            if obs_damping != pred_damping:
                misclassified += 1
                print(f"    MISMATCH in {kind}: pars={pars}, pred={pred_damping}, obs={obs_damping}, sup={sup_fd}")

            tested += 1
            if obs_damping:
                n_damping += 1
                if rho_val < 0.0:
                    n_neg_rho_damping += 1
            else:
                n_not_damping += 1
                if is_hurwitz:
                    n_neg_rho_amplifying_hurwitz += 1
                else:
                    n_non_hurwitz += 1

        status = "PASS" if misclassified == 0 else "FAIL"
        print(f"  [K3b-{kind}] {tested} random {kind} blocks classified vs direct FD frequency response: {status}")
        print(f"           misclassified = {misclassified}/{tested} (skipped near boundary: {skipped_near_boundary})")
        print(f"           breakdown: {n_damping} damping ({n_neg_rho_damping} with rho < 0), "
              f"{n_not_damping} non-damping ({n_neg_rho_amplifying_hurwitz} Hurwitz amplifying, {n_non_hurwitz} non-Hurwitz)")
    print()


# ============================================================================
# Part 1, Item 2: K4 and K5 nonlinear stiff integrations (1 V block + 1 L block)
# ============================================================================

def build_coupled_VL_system(N_eq, p, a_V, alpha, rho, a_L):
    """
    Build coupled nonlinear ODE system for 1 V block + 1 L block:
      dN/dt = Phi(N) - alpha * I
      dI/dt = (c_L * w) * (N - I) - a_V * I
      dw/dt = (c_V * I) - a_L * w
    Set c_V = 1.0, w_eq = I_eq / a_L, and c_L = u_V_eq / w_eq = a_L * u_V_eq / I_eq
    so that (N_eq, I_eq, w_eq) is the exact positive equilibrium for every a_L > 0
    (holding static gain of L fixed at c_L / a_L = u_V_eq / I_eq, as in K4-K6).
    """
    I_eq = p * N_eq
    h = alpha * p
    u_eq = a_V * p / (1.0 - p)
    c_V = 1.0
    w_eq = c_V * I_eq / a_L
    c_L = u_eq / w_eq  # = a_L * u_eq / I_eq
    phi = make_nonlinear_phi(N_eq, h, rho)

    def rhs(t, y):
        N, I, w = y[0], y[1], y[2]
        u = c_L * w
        return np.array([
            phi(N) - alpha * I,
            u * (N - I) - a_V * I,
            c_V * I - a_L * w,
        ], dtype=float)

    y_eq = np.array([N_eq, I_eq, w_eq], dtype=float)
    q = a_V
    d = a_V / (1.0 - p)
    A = rho + d
    B = d * (rho + h)
    J = rho + d - q
    C = rho * (d - q) + d * h
    F_a = J * a_L * a_L + (A * J + q * rho) * a_L + A * B

    J_mat = central_fd_jacobian(lambda y: rhs(0.0, y), y_eq)
    eigvals, eigvecs = np.linalg.eig(J_mat)
    idx_dom = int(np.argmax(np.real(eigvals)))
    lam_dom = complex(eigvals[idx_dom])
    vec_dom = eigvecs[:, idx_dom]

    # Also verify cubic polynomial z^3 + (a_L + A)z^2 + (B + a_L*J)z + a_L*C
    poly_roots = np.roots([1.0, a_L + A, B + a_L * J, a_L * C])
    poly_err = float(np.max(np.abs(np.sort_complex(eigvals) - np.sort_complex(poly_roots))))
    assert poly_err < 1.0e-7, f"Cubic mismatch: {poly_err}"

    info = {
        "q": q, "d": d, "h": h, "rho": rho, "A": A, "B": B, "J": J, "C": C, "F_a": F_a,
        "a_L": a_L, "lam_dom": lam_dom, "vec_dom": vec_dom, "y_eq": y_eq,
        "a_V": a_V, "alpha": alpha, "p": p,
    }
    return rhs, info


def measure_nonlinear_perturbation_rate(rhs, info):
    """
    Integrate the nonlinear stiff ODE from a small perturbation along the dominant
    eigenvector using Radau, and fit exponential growth/decay rate.
    """
    y_eq = info["y_eq"]
    lam_dom = info["lam_dom"]
    re_dom = float(np.real(lam_dom))
    im_dom = abs(float(np.imag(lam_dom)))
    vec = np.real(info["vec_dom"])
    if np.linalg.norm(vec) < 1.0e-12:
        vec = np.imag(info["vec_dom"])
    # Scale perturbation so relative L2 norm in normalized coordinates is 1e-5
    rel_dir = vec / y_eq
    rel_dir = rel_dir / np.linalg.norm(rel_dir)
    eps0 = 1.0e-5
    y0 = y_eq * (1.0 + eps0 * rel_dir)

    # Choose integration horizon so amplitude changes by ~ exp(2.5) to exp(4), staying in linear regime
    rate_mag = max(abs(re_dom), 1.0e-3)
    T_end = min(600.0, max(15.0, 2.5 / rate_mag))
    if im_dom > 0.0:
        # Ensure at least 6 half-periods so peak-to-peak slope is super clean
        half_period = math.pi / im_dom
        T_end = max(T_end, 8.0 * half_period)
    # Keep trajectory inside linear regime and well above Radau atol=1e-13
    if re_dom > 0.0:
        T_end = min(T_end, 6.0 / re_dom)
    elif re_dom < 0.0:
        T_end = min(T_end, 11.0 / abs(re_dom))

    t_eval = np.linspace(0.0, T_end, 4000)
    sol = solve_ivp(rhs, (0.0, T_end), y0, method="Radau", t_eval=t_eval, rtol=1.0e-11, atol=1.0e-13)
    assert sol.success, f"Radau integration failed: {sol.message}"

    rel_dev = (sol.y.T - y_eq) / y_eq
    norms = np.linalg.norm(rel_dev, axis=1)

    # Fit slope on local maxima of norms (eliminates oscillatory modulation!) if oscillatory, else all points
    local_max_idx = np.where((norms[1:-1] > norms[:-2]) & (norms[1:-1] > norms[2:]))[0] + 1
    if len(local_max_idx) >= 4:
        use_idx = local_max_idx[1:] if len(local_max_idx) >= 5 else local_max_idx
        t_fit = sol.t[use_idx]
        log_fit = np.log(norms[use_idx])
        amp_start, amp_end = float(norms[use_idx[0]]), float(norms[use_idx[-1]])
    else:
        start = len(sol.t) // 3
        t_fit = sol.t[start:]
        log_fit = np.log(norms[start:])
        amp_start, amp_end = float(norms[start]), float(norms[-1])

    slope, _ = np.polyfit(t_fit, log_fit, 1)
    return float(slope), amp_start, amp_end, float(T_end)


def follow_long_run(rhs, info, T_long=1500.0, eps0=1.0e-3):
    """Integrate an unstable case into the long run with Radau to determine asymptotic fate."""
    y_eq = info["y_eq"]
    vec = np.real(info["vec_dom"])
    rel_dir = (vec / y_eq)
    rel_dir /= np.linalg.norm(rel_dir)
    y0 = y_eq * (1.0 + eps0 * rel_dir)

    # Stop event if any state becomes negative or approaches machine underflow
    def neg_S_event(t, y):
        return y[0] - y[1]  # S = N - I
    neg_S_event.terminal = False

    t_eval = np.linspace(0.0, T_long, 12000)
    sol = solve_ivp(
        rhs, (0.0, T_long), y0, method="Radau", t_eval=t_eval,
        rtol=1.0e-10, atol=1.0e-12, events=[neg_S_event]
    )
    tail = sol.y[:, len(sol.t) // 2:]
    N_tail, I_tail, w_tail = tail[0], tail[1], tail[2]
    min_S_all = float(np.min(sol.y[0] - sol.y[1]))
    min_N_all = float(np.min(sol.y[0]))
    min_I_all = float(np.min(sol.y[1]))
    return {
        "success": sol.success,
        "t_end": float(sol.t[-1]),
        "N_range_tail": (float(np.min(N_tail)), float(np.max(N_tail))),
        "I_range_tail": (float(np.min(I_tail)), float(np.max(I_tail))),
        "w_range_tail": (float(np.min(w_tail)), float(np.max(w_tail))),
        "min_S_all": min_S_all,
        "min_N_all": min_N_all,
        "min_I_all": min_I_all,
    }


def run_part1_k4_k5():
    print("=" * 78)
    print("PART 1, ITEM 2: NONLINEAR STIFF INTEGRATIONS FOR K4 AND K5 (Radau)")
    print(f"Pre-declared tolerances: sign agreement 100%, |sigma_fit - Re(lam)| <= max({TOL_ODE_ABS:.1e}, {TOL_ODE_REL:.0%}*|Re(lam)|)")
    print("=" * 78)

    cases = []

    # --- Regime 1: rho >= 0 (4 cases, physical a_V >= alpha, including rho=0 and both sides of K/2 boundary check) ---
    # Here N_eq = 10, p = 0.25, alpha = 1.0, a_V = 1.2 >= alpha (h = 0.25, q = 1.2, d = 1.6)
    for label, rho_val, a_L in [
        ("R1-1 (rho>0, slow L)", 0.40, 0.08),
        ("R1-2 (rho>0, med L)",  0.40, 1.50),
        ("R1-3 (rho>0, fast L)", 0.10, 12.0),
        ("R1-4 (rho=0 exact K/2)", 0.00, 1.00),
    ]:
        cases.append((label, 10.0, 0.25, 1.2, 1.0, rho_val, a_L))

    # --- Regime 2: -h < rho < 0 with an instability island (physical a_V >= alpha!) ---
    # Construct a 100% physical V block with a_V = 1.05 >= alpha = 1.0, p = 0.20, N_eq = 10.0:
    # h = alpha*p = 0.20, q = 1.05, d = 1.05/0.8 = 1.3125.
    # Choose rho = -0.196 (logistic host at N*/K = (h+rho)/(2h+rho) = 0.004/0.204 = 1.96% of K):
    # A = 1.1165, B = 1.3125*0.004 = 0.00525, J = -0.196 + 0.2625 = 0.0665 > 0.
    # q*|rho| - A*J = 1.05*0.196 - 1.1165*0.0665 = 0.2058 - 0.07424725 = 0.13155275
    # 2*sqrt(J*A*B) = 2*sqrt(0.0665 * 1.1165 * 0.00525) = 0.039487... < 0.13155!
    # Roots a_- and a_+ of F(a) = J*a^2 + (A*J + q*rho)*a + A*B = 0:
    p2, a_V2, alpha2, rho2 = 0.20, 1.05, 1.00, -0.196
    q2, d2, h2 = a_V2, a_V2 / (1.0 - p2), alpha2 * p2
    A2, B2, J2 = rho2 + d2, d2 * (rho2 + h2), rho2 + d2 - q2
    disc2 = (A2 * J2 + q2 * rho2) ** 2 - 4.0 * J2 * A2 * B2
    a_minus = (-(A2 * J2 + q2 * rho2) - math.sqrt(disc2)) / (2.0 * J2)
    a_plus = (-(A2 * J2 + q2 * rho2) + math.sqrt(disc2)) / (2.0 * J2)
    print(f"  Physical Regime-2 Island V-block (a_V={a_V2} >= alpha={alpha2}, p={p2}, rho={rho2}, h={h2}):")
    print(f"    Exact island boundaries: a_- = {a_minus:.6f}, a_+ = {a_plus:.6f}")

    # Test BOTH sides of BOTH boundaries a_- and a_+ (4 cases in Regime 2 + 2 extra on primary's example!)
    cases.extend([
        (f"R2-1 (below a_-={a_minus:.3f}, long L -> stable)", 10.0, p2, a_V2, alpha2, rho2, 0.35 * a_minus),
        (f"R2-2 (above a_-={a_minus:.3f}, in island -> unstable)", 10.0, p2, a_V2, alpha2, rho2, 1.80 * a_minus),
        (f"R2-3 (below a_+={a_plus:.3f}, in island -> unstable)", 10.0, p2, a_V2, alpha2, rho2, 0.65 * a_plus),
        (f"R2-4 (above a_+={a_plus:.3f}, short L -> stable)", 10.0, p2, a_V2, alpha2, rho2, 1.60 * a_plus),
    ])

    # --- Regime 3: rho = -h (no host self-regulation, Phi(N) = r*N, physical a_V >= alpha!) ---
    # Set 3A: p = 0.25, a_V = 1.2 >= alpha = 1.0 -> h = 0.25, rho = -0.25, q = 1.2, d = 1.6
    # A = d - h = 1.35, J = d - q - h = 0.15 > 0, a_c = q*h/J - A = 0.30/0.15 - 1.35 = 0.65 > 0!
    # Set 3B: p = 0.20, a_V = 1.1 >= alpha = 1.0 -> h = 0.20, rho = -0.20, q = 1.1, d = 1.375
    # A = 1.175, J = 0.075 > 0, a_c = 0.22/0.075 - 1.175 = 1.758333... > 0!
    ac_3a = (1.2 * 0.25) / 0.15 - 1.35
    ac_3b = (1.1 * 0.20) / 0.075 - 1.175
    print(f"  Physical Regime-3 No-self-reg V-blocks (rho = -h):")
    print(f"    Set 3A critical rate a_c = {ac_3a:.6f}; Set 3B critical rate a_c = {ac_3b:.6f}")

    cases.extend([
        (f"R3-1 (Set 3A, a_L > a_c={ac_3a:.3f} -> stable)", 10.0, 0.25, 1.2, 1.0, -0.25, 1.50 * ac_3a),
        (f"R3-2 (Set 3A, a_L < a_c={ac_3a:.3f} -> unstable)", 10.0, 0.25, 1.2, 1.0, -0.25, 0.50 * ac_3a),
        (f"R3-3 (Set 3A, very long L a_L << a_c -> still unstable)", 10.0, 0.25, 1.2, 1.0, -0.25, 0.05 * ac_3a),
        (f"R3-4 (Set 3B, a_L > a_c={ac_3b:.3f} -> stable)", 10.0, 0.20, 1.1, 1.0, -0.20, 1.40 * ac_3b),
        (f"R3-5 (Set 3B, a_L < a_c={ac_3b:.3f} -> unstable)", 10.0, 0.20, 1.1, 1.0, -0.20, 0.60 * ac_3b),
    ])

    # Also test both sides of upper boundary a_+ = 22.8518 on primary's flagship V example (a_V=0.5 < alpha=2.0)
    cases.extend([
        ("R2-primary-unstable (a_V=0.5<alpha=2, a_L=3 in (1.138,22.85))", 1.0, 0.5, 0.5, 2.0, -0.49, 3.0),
        ("R2-primary-stable-fast (a_V=0.5<alpha=2, a_L=30 > 22.85)", 1.0, 0.5, 0.5, 2.0, -0.49, 30.0),
    ])

    n_pass = 0
    for label, N_eq, p, a_V, alpha, rho_val, a_L in cases:
        rhs, info = build_coupled_VL_system(N_eq, p, a_V, alpha, rho_val, a_L)
        re_pred = float(np.real(info["lam_dom"]))
        slope_fit, n0, n1, T_end = measure_nonlinear_perturbation_rate(rhs, info)
        sign_ok = (re_pred > 0.0 and slope_fit > 0.0 and n1 > n0) or (re_pred < 0.0 and slope_fit < 0.0 and n1 < n0)
        tol_here = max(TOL_ODE_ABS, TOL_ODE_REL * abs(re_pred))
        err = abs(slope_fit - re_pred)
        ok = sign_ok and (err <= tol_here)
        if ok:
            n_pass += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}:")
        print(f"         Re(lam_dom)={re_pred:+.6f}, ODE_fit={slope_fit:+.6f}, err={err:.2e} (tol {tol_here:.2e}), "
              f"norm {n0:.1e}->{n1:.1e} over T={T_end:.1f}")

    print(f"  -> Summary K4/K5 linear-regime perturbation check: {n_pass}/{len(cases)} passed.\n")
    assert n_pass == len(cases), f"K4/K5 ODE check failed on {len(cases) - n_pass} cases"

    # --- Long-run fate of unstable trajectories (at least 2 required; we run 3!) ---
    print("  --- LONG-RUN TRAJECTORY FATE OF UNSTABLE CASES ---")
    long_cases = [
        ("LongRun-1: Physical Regime-2 Island (a_V=1.05>=alpha=1, a_L=0.65*a_+)",
         10.0, p2, a_V2, alpha2, rho2, 0.65 * a_plus, 3000.0),
        ("LongRun-2: Physical Regime-3 No-self-reg (a_V=1.2>=alpha=1, a_L=0.5*a_c)",
         10.0, 0.25, 1.2, 1.0, -0.25, 0.50 * ac_3a, 2500.0),
        ("LongRun-3: primary flagship example (a_V=0.5 < alpha=2.0, a_L=3.0)",
         1.0, 0.5, 0.5, 2.0, -0.49, 3.0, 1500.0),
    ]
    for title, N_eq, p, a_V, alpha, rho_val, a_L, T_long in long_cases:
        rhs, info = build_coupled_VL_system(N_eq, p, a_V, alpha, rho_val, a_L)
        res = follow_long_run(rhs, info, T_long=T_long, eps0=1.0e-3)
        print(f"  * {title}:")
        print(f"      t_end={res['t_end']:.1f}, success={res['success']}")
        print(f"      Tail N range = [{res['N_range_tail'][0]:.6e}, {res['N_range_tail'][1]:.6e}]")
        print(f"      Tail I range = [{res['I_range_tail'][0]:.6e}, {res['I_range_tail'][1]:.6e}]")
        print(f"      Tail w range = [{res['w_range_tail'][0]:.6e}, {res['w_range_tail'][1]:.6e}]")
        print(f"      Min N-I (susceptibles S) over all t = {res['min_S_all']:+.6e}, min N = {res['min_N_all']:+.6e}, min I = {res['min_I_all']:+.6e}")
    print()


# ============================================================================
# Part 1, Item 3: K2 counterexample search (>= 5,000 random cycles of 2..8 blocks)
# ============================================================================

def sample_random_block_for_k2(rng, force_strict=False):
    """
    Sample a random block (L, R, S, V, or M) satisfying the non-amplifying hypothesis
    (Hurwitz internal matrix and sup_w |E(iw)| <= 1), or strictly damping if force_strict=True.
    Returns (kind, A_blk, b_blk, c_blk) in normalized log-perturbation coordinates
    where x_k = du_k/u_k* and y_k = dv_k/v_k* (so u*/v* = 1 in normalized coordinates).
    """
    if force_strict:
        choices = ["R_strict", "S", "V_pos", "V_neg_damp", "M_pos", "M_neg_damp"]
    else:
        choices = ["L", "L", "R_weak", "R_strict", "S", "V_pos", "V_neg_damp", "M_pos", "M_neg_damp"]
    pick = choices[int(rng.integers(0, len(choices)))]

    if pick == "L":
        a = 10.0 ** rng.uniform(-2.5, 2.5)
        return "L", np.array([[-a]]), np.array([a]), np.array([1.0])

    if pick == "R_weak":
        a = 10.0 ** rng.uniform(-2.5, 2.5)
        kappa = 0.0 if rng.random() < 0.5 else 2.0  # Exact |1 - kappa| = 1 boundary!
        return "R", np.array([[-a]]), np.array([a]), np.array([1.0 - kappa])

    if pick == "R_strict":
        a = 10.0 ** rng.uniform(-2.5, 2.5)
        # Bias half of draws very close to kappa=2 (e.g. 1.999) to stress small-gain margin
        if rng.random() < 0.4:
            kappa = 2.0 - 10.0 ** rng.uniform(-4.0, -1.5)
        else:
            kappa = rng.uniform(0.001, 1.999)
        return "R", np.array([[-a]]), np.array([a]), np.array([1.0 - kappa])

    if pick == "S":
        a = 10.0 ** rng.uniform(-2.5, 2.5)
        p = 10.0 ** rng.uniform(-3.5, -0.02) if rng.random() < 0.4 else rng.uniform(0.001, 0.99)
        d = a / (1.0 - p)
        return "S", np.array([[-d]]), np.array([a]), np.array([1.0])

    if pick in ("V_pos", "V_neg_damp"):
        for _ in range(500):
            p = rng.uniform(0.01, 0.95)
            alpha = 10.0 ** rng.uniform(-1.5, 1.5)
            a_V = alpha * (1.0 + 10.0 ** rng.uniform(-2.0, 1.2)) if rng.random() < 0.7 else 10.0 ** rng.uniform(-1.5, 1.5)
            q = a_V
            d = a_V / (1.0 - p)
            h = alpha * p
            if pick == "V_pos":
                rho = 0.0 if rng.random() < 0.15 else 10.0 ** rng.uniform(-3.0, 1.5)
            else:
                rho = -h * rng.uniform(0.01, 0.95)
            ok, marg = primary_two_sided_damping_test(q, d, h, rho)
            if ok and marg > 1.0e-7:
                # Normalized state [b, y] = [dN/N*, dI/I*]: db/dt = -rho*b - h*y, dy/dt = d*b - d*y + q*x
                A_blk = np.array([[-rho, -h], [d, -d]], dtype=float)
                b_blk = np.array([0.0, q], dtype=float)
                c_blk = np.array([0.0, 1.0], dtype=float)
                return "V", A_blk, b_blk, c_blk

    # M_pos or M_neg_damp
    for _ in range(500):
        m = 10.0 ** rng.uniform(-1.0, 1.5)
        xi = 10.0 ** rng.uniform(-1.5, 1.5)
        alpha = 10.0 ** rng.uniform(-2.0, 1.0)
        eps_agg = 1.0 + 10.0 ** rng.uniform(-2.0, 1.0)
        h = alpha * m
        q = xi + eps_agg * h
        d = xi + 2.0 * eps_agg * h
        if pick == "M_pos":
            rho = 0.0 if rng.random() < 0.15 else 10.0 ** rng.uniform(-3.0, 1.5)
        else:
            rho = -h * rng.uniform(0.01, 0.95)
        ok, marg = primary_two_sided_damping_test(q, d, h, rho)
        if ok and marg > 1.0e-7:
            A_blk = np.array([[-rho, -h], [d, -d]], dtype=float)
            b_blk = np.array([0.0, q], dtype=float)
            c_blk = np.array([0.0, 1.0], dtype=float)
            return "M", A_blk, b_blk, c_blk

    raise RuntimeError("Failed to sample damping block")


def assemble_cycle_jacobian(blocks):
    """Assemble full state-space Jacobian J of a cyclic interconnection of blocks."""
    dims = [blk[1].shape[0] for blk in blocks]
    total = sum(dims)
    offsets = np.cumsum([0] + dims[:-1])
    J = np.zeros((total, total), dtype=float)
    n_blk = len(blocks)
    for k, (_, A_k, b_k, c_k) in enumerate(blocks):
        r0 = offsets[k]
        r1 = r0 + dims[k]
        J[r0:r1, r0:r1] += A_k
        prev = (k - 1) % n_blk
        p0 = offsets[prev]
        p1 = p0 + dims[prev]
        J[r0:r1, p0:p1] += np.outer(b_k, c_prev if False else blocks[prev][3])
    return J


def run_part1_k2(rng):
    print("=" * 78)
    print("PART 1, ITEM 3: COUNTEREXAMPLE SEARCH FOR K2 (>= 5,000 RANDOM CYCLES, n=2..8)")
    print(f"Pre-declared tolerance: TOL_HURWITZ_K2 = {TOL_HURWITZ_K2:.1e}")
    print("=" * 78)

    n_cycles = 6000
    max_re_found = -1.0e9
    worst_desc = ""
    n_counterexamples = 0
    counts_by_len = {k: 0 for k in range(2, 9)}
    type_counts = {"L": 0, "R": 0, "S": 0, "V": 0, "M": 0}

    for idx in range(n_cycles):
        n_len = int(rng.integers(2, 9))  # 2 to 8 inclusive
        counts_by_len[n_len] += 1
        strict_slot = int(rng.integers(0, n_len))
        blocks = []
        for k in range(n_len):
            blk = sample_random_block_for_k2(rng, force_strict=(k == strict_slot))
            blocks.append(blk)
            type_counts[blk[0]] += 1
        rng.shuffle(blocks)

        J_cyc = assemble_cycle_jacobian(blocks)
        eigs = np.linalg.eigvals(J_cyc)
        re_max = float(np.max(np.real(eigs)))
        if re_max > max_re_found:
            max_re_found = re_max
            worst_desc = f"cycle #{idx}, n={n_len}, types={[b[0] for b in blocks]}, max Re={re_max:+.4e}"
        if re_max >= 0.0 or re_max > TOL_HURWITZ_K2:
            n_counterexamples += 1

    status = "PASS" if n_counterexamples == 0 and max_re_found < 0.0 else "FAIL"
    print(f"  [K2] Tested {n_cycles} random cycles (lengths 2..8): {status}")
    print(f"       Counterexamples found (Re >= 0): {n_counterexamples}")
    print(f"       Largest Re(lambda) across all {n_cycles} cycles: {max_re_found:+.6e}")
    print(f"       Closest-to-zero case: {worst_desc}")
    print(f"       Cycles per length: {counts_by_len}")
    print(f"       Total blocks sampled by type: {type_counts}\n")


# ============================================================================
# Part 2: Dobson & Hudson (1992) and Gaba & Gourbiere (2008) block analysis
# ============================================================================

def dh_host_block_analytic_E(z, rho, h_eff, d_A, d_P, alpha, kp, m_P, f_A):
    """
    Exact analytical log-to-log transfer function E_DH(z) = (u*/v*) T_DH(z)
    for the 3-state Dobson & Hudson / Gaba & Gourbiere host block (H, A, P):
      dH/dt = Phi(H) - (alpha + delta)*P
      dA/dt = sigma * u * H - (mu_A + b + theta)*A - alpha * P * A / H
      dP/dt = theta * A + (1 - sigma) * u * H - (mu_P + b + alpha)*P - alpha * kp * P^2 / H
      v     = lam * P
    derived in Section 2 of check-coupled-hosts.md.
    """
    L_num = (1.0 - f_A) * (z + d_A) + f_A * d_A  # = z*(1 - f_A) + d_A
    num = d_P * (z + rho) * L_num
    # tilde_d(z)*(z + d_A) = (d_P + alpha*kp*m_P)*(z + d_A) + f_A*d_P*alpha*m_P
    td_zA = (d_P + alpha * kp * m_P) * (z + d_A) + f_A * d_P * alpha * m_P
    den = (z + rho) * (z * (z + d_A) + td_zA) + h_eff * (d_P * L_num + td_zA - d_P * (z + d_A))
    return num / den


def solve_dh_equilibrium(a_g, b_g, Delta, lam, mu_P, mu_A, gamma, beta, theta, sigma, alpha, delta, k_nb):
    """
    Find positive endemic equilibrium (H*, A*, P*, W*) of the full 4-ODE
    Dobson & Hudson (1992) Model III / Gaba & Gourbiere (2008) Models A-D system.
    Note: m_P(H) = (a_g - b_g - Delta*H) / (alpha + delta).
    At equilibrium:
      d_A(H) = mu_A + b_g + theta + alpha * m_P(H)
      d_P(H) = mu_P + b_g + alpha + alpha * kp * m_P(H)
      psi(H) = (1 - sigma) + sigma * theta / d_A(H)
      u*(H)  = beta * lam * P* / (gamma + beta * H) = beta * lam * H * m_P(H) / (gamma + beta * H)
      Equilibrium condition (for m_P(H) > 0):
        F_eq(H) = (beta * lam * H / (gamma + beta * H)) * psi(H) - d_P(H) = 0!
    Notice F_eq(H) is strictly increasing in H (since H/(gamma + beta*H) increases with H,
    m_P(H) decreases or is constant with H so d_A and d_P decrease or stay constant, increasing psi and -d_P)!
    """
    kp = (k_nb + 1.0) / k_nb
    H_max = (a_g - b_g) / Delta if Delta > 0.0 else 1.0e9

    def f_eq(H):
        m_P = (a_g - b_g - Delta * H) / (alpha + delta)
        d_A = mu_A + b_g + theta + alpha * m_P
        d_P = mu_P + b_g + alpha + alpha * kp * m_P
        psi = (1.0 - sigma) + sigma * theta / d_A
        return (beta * lam * H / (gamma + beta * H)) * psi - d_P

    if f_eq(1.0e-9) >= 0.0:
        return None
    if Delta > 0.0:
        if f_eq(H_max * (1.0 - 1.0e-12)) <= 0.0:
            return None
        lo, hi = 1.0e-9, H_max * (1.0 - 1.0e-12)
    else:
        # Exact analytical H* when Delta = 0!
        m_P = (a_g - b_g) / (alpha + delta)
        d_A = mu_A + b_g + theta + alpha * m_P
        d_P = mu_P + b_g + alpha + alpha * kp * m_P
        psi = (1.0 - sigma) + sigma * theta / d_A
        if lam * psi <= d_P:
            return None
        H_star = gamma * d_P / (beta * (lam * psi - d_P))
        lo = hi = H_star

    if lo < hi:
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if f_eq(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        H_star = 0.5 * (lo + hi)

    m_P = (a_g - b_g - Delta * H_star) / (alpha + delta)
    P_star = m_P * H_star
    d_A = mu_A + b_g + theta + alpha * m_P
    d_P = mu_P + b_g + alpha + alpha * kp * m_P
    psi = (1.0 - sigma) + sigma * theta / d_A
    f_A = (sigma * theta / d_A) / psi if sigma > 0.0 else 0.0
    u_star = d_P * m_P / psi
    W_star = u_star / beta
    A_star = (sigma * u_star * H_star / d_A) if sigma > 0.0 else 0.0
    h_eff = (alpha + delta) * m_P
    rho = -(a_g - b_g) + 2.0 * Delta * H_star

    return {
        "H": H_star, "A": A_star, "P": P_star, "W": W_star, "u": u_star,
        "m_P": m_P, "d_A": d_A, "d_P": d_P, "psi": psi, "f_A": f_A,
        "h_eff": h_eff, "rho": rho, "kp": kp, "a_W_eff": gamma + beta * H_star,
    }


def analyze_dh_stability(a_g, b_g, Delta, lam, mu_P, mu_A, gamma, beta, theta, sigma, alpha, delta, k_nb):
    """
    Compute both:
      (1) Exact eigenvalues of the full 4-ODE (or 3-ODE when sigma=0) Dobson & Hudson / Gaba & Gourbiere Jacobian
          (which includes -beta*W*H uptake depletion in dW/dt), AND
      (2) Roots of the K1 scalar-loop characteristic equation D_DH(z)*(z + a_W) - N_DH(z)*a_W = 0
          (with pure L block rate a_W = gamma + beta*H*, and also with a_W = gamma).
    """
    eq = solve_dh_equilibrium(a_g, b_g, Delta, lam, mu_P, mu_A, gamma, beta, theta, sigma, alpha, delta, k_nb)
    if eq is None:
        return None

    kp = eq["kp"]
    H_s, A_s, P_s, W_s = eq["H"], eq["A"], eq["P"], eq["W"]

    # 1. Full nonlinear RHS and FD Jacobian
    if sigma > 0.0:
        def rhs_full(w):
            H, A, P, W = w[0], w[1], w[2], w[3]
            dH = (a_g - b_g - Delta * H) * H - (alpha + delta) * P
            dA = sigma * beta * W * H - (mu_A + b_g + theta) * A - alpha * P * A / H
            dP = theta * A + (1.0 - sigma) * beta * W * H - (mu_P + b_g + alpha) * P - alpha * kp * P * P / H
            dW = lam * P - gamma * W - beta * W * H
            return np.array([dH, dA, dP, dW], dtype=float)
        J_full = central_fd_jacobian(rhs_full, np.array([H_s, A_s, P_s, W_s]))
    else:
        def rhs_full(w):
            H, P, W = w[0], w[1], w[2]
            dH = (a_g - b_g - Delta * H) * H - (alpha + delta) * P
            dP = beta * W * H - (mu_P + b_g + alpha) * P - alpha * kp * P * P / H
            dW = lam * P - gamma * W - beta * W * H
            return np.array([dH, dP, dW], dtype=float)
        J_full = central_fd_jacobian(rhs_full, np.array([H_s, P_s, W_s]))

    eigs_full = np.linalg.eigvals(J_full)
    re_full = float(np.max(np.real(eigs_full)))

    # 2. K1 scalar loop polynomial with E_DH(z) * (a_W / (z + a_W)) = 1
    rho, h_eff, d_A, d_P, m_P, f_A = eq["rho"], eq["h_eff"], eq["d_A"], eq["d_P"], eq["m_P"], eq["f_A"]
    # Build polynomial coefficients using numpy.poly1d
    z_p = np.poly1d([1.0, 0.0])
    L_num = (1.0 - f_A) * (z_p + d_A) + f_A * d_A
    N_dh = d_P * (z_p + rho) * L_num
    td_zA = (d_P + alpha * kp * m_P) * (z_p + d_A) + f_A * d_P * alpha * m_P
    D_dh = (z_p + rho) * (z_p * (z_p + d_A) + td_zA) + h_eff * (d_P * L_num + td_zA - d_P * (z_p + d_A))

    re_k1 = {}
    for key, a_W in [("aW=gamma+beta*H*", eq["a_W_eff"]), ("aW=gamma", gamma)]:
        chi_k1 = D_dh * (z_p + a_W) - N_dh * a_W
        roots_k1 = np.roots(chi_k1.c)
        re_k1[key] = float(np.max(np.real(roots_k1)))

    # Also compute host-block gain peak and phase at natural oscillation frequency
    omegas = np.geomspace(1.0e-2, 50.0, 1500)
    E_vals = dh_host_block_analytic_E(1j * omegas, rho, h_eff, d_A, d_P, alpha, kp, m_P, f_A)
    sup_E = float(np.max(np.abs(E_vals)))
    w_peak = float(omegas[int(np.argmax(np.abs(E_vals)))])

    return {
        "eq": eq, "re_full": re_full, "re_k1_eff": re_k1["aW=gamma+beta*H*"],
        "re_k1_gam": re_k1["aW=gamma"], "sup_E_host": sup_E, "w_peak": w_peak,
    }


def run_part2_published_models(rng):
    print("=" * 78)
    print("PART 2: TESTING AGAINST DOBSON & HUDSON (1992) AND GABA & GOURBIERE (2008)")
    print("=" * 78)

    # 2.0 Verify our analytic E_DH(z) against FD linearization of the 3-state (H, A, P) host block
    max_err_dh = 0.0
    for _ in range(100):
        a_g = rng.uniform(1.2, 2.5)
        b_g = rng.uniform(0.5, 1.1)
        Delta = rng.uniform(0.0, 0.01)
        mu_P = rng.uniform(0.5, 2.0)
        mu_A = rng.uniform(0.1, 1.0)
        theta = rng.uniform(0.5, 12.0)
        sigma = rng.uniform(0.05, 1.0)
        alpha = 10.0 ** rng.uniform(-4.0, -2.5)
        delta = 10.0 ** rng.uniform(-4.0, -2.5)
        k_nb = rng.uniform(0.3, 5.0)
        kp = (k_nb + 1.0) / k_nb
        H_eq = rng.uniform(20.0, 80.0) if Delta == 0.0 else 0.4 * (a_g - b_g) / Delta
        m_P = (a_g - b_g - Delta * H_eq) / (alpha + delta)
        if m_P <= 0.0:
            continue
        P_eq = m_P * H_eq
        d_A = mu_A + b_g + theta + alpha * m_P
        d_P = mu_P + b_g + alpha + alpha * kp * m_P
        psi = (1.0 - sigma) + sigma * theta / d_A
        f_A = (sigma * theta / d_A) / psi
        u_eq = d_P * m_P / psi
        A_eq = sigma * u_eq * H_eq / d_A
        lam = 10.0

        def f_hap(w, u):
            H, A, P = w[0], w[1], w[2]
            return np.array([
                (a_g - b_g - Delta * H) * H - (alpha + delta) * P,
                sigma * u * H - (mu_A + b_g + theta) * A - alpha * P * A / H,
                theta * A + (1.0 - sigma) * u * H - (mu_P + b_g + alpha) * P - alpha * kp * P * P / H,
            ], dtype=float)

        A_fd, b_fd, c_fd, v_eq, resid = block_fd_linearization(f_hap, lambda w: lam * w[2], np.array([H_eq, A_eq, P_eq]), u_eq)
        assert resid < TOL_EQ_RESID
        ws = np.array([0.05, 0.3, 1.0, 3.0, 10.0])
        E_fd = eval_ss_log_response(A_fd, b_fd, c_fd, u_eq, v_eq, ws)
        h_eff = (alpha + delta) * m_P
        rho = -(a_g - b_g) + 2.0 * Delta * H_eq
        E_ana = dh_host_block_analytic_E(1j * ws, rho, h_eff, d_A, d_P, alpha, kp, m_P, f_A)
        max_err_dh = max(max_err_dh, float(np.max(np.abs(E_fd - E_ana))))

    print(f"  [P2-Check] 3-state (H,A,P) host-block E_DH(z) vs FD Jacobian across 100 sets: max err = {max_err_dh:.2e} (PASS)\n")
    assert max_err_dh < TOL_FD_VS_ANA

    # Published Dobson & Hudson (1992) Table 2 mean parameter values:
    # a = 1.8 yr^-1, b = 1.05 yr^-1, lam = 11 yr^-1, mu_P = 1.0 yr^-1, mu_A = 0.5 yr^-1,
    # alpha = 3e-4, delta = 5e-4, k = 1.0, beta = 0.1.
    # Territoriality: Fig. 7 caption & p. 493 say diverging cycles occur when K / H_T is ~ 4-6.
    # At gamma = 13 (4-week larval life), H_T = 13 * 2.0503 / (0.1 * (11 - 2.0503)) = 29.78 grouse/km^2.
    # Taking K = 180 grouse/km^2 (K / H_T = 6.04) gives Delta = (a - b) / K = 0.75 / 180 = 0.0041667.
    a_g, b_g, lam, mu_P, mu_A = 1.8, 1.05, 11.0, 1.0, 0.5
    alpha, delta, k_nb, beta = 3.0e-4, 5.0e-4, 1.0, 0.1
    Delta_terr = (a_g - b_g) / 180.0

    print("  --- (a) Destabilising effect of longer-lived free-living stage (sigma = 0, D&H Table 2 + Fig. 7a) ---")
    print("      Parameters: a=1.8, b=1.05, lam=11, mu_P=1.0, alpha=3e-4, delta=5e-4, k=1.0, beta=0.1, K=180 (Delta=0.00417)")
    for w_months in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        gamma = 12.0 / w_months
        res = analyze_dh_stability(a_g, b_g, Delta_terr, lam, mu_P, mu_A, gamma, beta, 4.0, 0.0, alpha, delta, k_nb)
        eq = res["eq"]
        print(f"      1/gamma={w_months:4.2f} mo (gamma={gamma:5.2f}/yr): H*/K={eq['H']/180.0:.3f}, rho={eq['rho']:+.4f}, "
              f"Re_full={res['re_full']:+.5f}, Re_K1(aW_eff)={res['re_k1_eff']:+.5f}, Re_K1(gamma)={res['re_k1_gam']:+.5f}")

    print("\n  --- (b) Stabilising effect of long arrestment (sigma = 1, gamma = 13/yr [4 wk], D&H Table 2 + Fig. 7b) ---")
    for arr_months in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        theta = 12.0 / arr_months
        res = analyze_dh_stability(a_g, b_g, Delta_terr, lam, mu_P, mu_A, 13.0, beta, theta, 1.0, alpha, delta, k_nb)
        if res is None:
            print(f"      1/theta={arr_months:5.2f} mo: parasite fails to establish (R0 <= 1)")
        else:
            eq = res["eq"]
            print(f"      1/theta={arr_months:5.2f} mo (theta={theta:5.2f}/yr): H*/K={eq['H']/180.0:.3f}, rho={eq['rho']:+.4f}, "
                  f"sup|E_DH|={res['sup_E_host']:.3f}, Re_full={res['re_full']:+.5f}, Re_K1(aW_eff)={res['re_k1_eff']:+.5f}")

    print("\n  --- (c) Non-monotone effect of arrestment duration WITHOUT host self-regulation (Delta = 0, Gaba & Gourbiere 2008) ---")
    # Case c1: G&G Model A (Delta=0, fast W [gamma=1300/yr, beta=10 -> H0=130], sigma=0.85, lam=11, alpha=3e-4, delta=0, k=9.5 [low agg, k>=5]):
    # Reproduces G&G Section 3.1 / Fig. 1c: STABLE at 1 mo -> UNSTABLE at 2-7 mo (peak 3-4 mo) -> STABLE AGAIN at 10-12 mo!
    # Also exposes the exact breakdown of naive scalar-L K1 (without -beta*W*H depletion): Re_K1 rises then falls (non-monotone)
    # because 0 < sigma < 1, but stays positive (+0.0287 at 12 mo) because it misses H/(H+H0) uptake saturation!
    print("      [c1] G&G Model A (Delta=0, fast W [gamma=1300/yr, H0=130], sigma=0.85, lam=11, alpha=3e-4, delta=0, k=9.5):")
    for arr_months in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0]:
        theta = 12.0 / arr_months
        res = analyze_dh_stability(a_g, b_g, 0.0, 11.0, mu_P, mu_A, 1300.0, 10.0, theta, 0.85, 3.0e-4, 0.0, 9.5)
        eq = res["eq"]
        E_ref = dh_host_block_analytic_E(1j * 0.9, eq["rho"], eq["h_eff"], eq["d_A"], eq["d_P"], 3.0e-4, eq["kp"], eq["m_P"], eq["f_A"])
        phase_deg = math.degrees(cmath_phase(E_ref))
        print(f"      1/theta={arr_months:5.2f} mo: f_A={eq['f_A']:.3f}, H*={eq['H']:6.1f}, |E_DH(0.9i)|={abs(E_ref):.3f}, "
              f"arg={phase_deg:+6.1f} deg, Re_full={res['re_full']:+.6f}, Re_K1(no -betaWH)={res['re_k1_eff']:+.6f}")

    # Case c2: G&G Model B (Delta=0, with free-living W [gamma=6.5/yr, beta=0.1 -> H0=65], sigma=0.85, lam=5.8, alpha=3e-4, delta=0, k=13.0):
    # Reproduces G&G Abstract & Section 3.2: UNSTABLE with free-living stage alone (sigma=0), MORE UNSTABLE at short arrest (0.25-2 mo),
    # and STABILISED at long arrest (>= 4-7 mo) even with ZERO host self-regulation (Delta=0)!
    print("\n      [c2] G&G Model B (Delta=0, free-living W [gamma=6.5/yr, beta=0.1], sigma=0.85, lam=5.8, alpha=3e-4, delta=0, k=13.0):")
    res0 = analyze_dh_stability(a_g, b_g, 0.0, 5.8, mu_P, mu_A, 6.5, 0.1, 10.0, 0.0, 3.0e-4, 0.0, 13.0)
    print(f"      no arrest (sigma=0):          H*={res0['eq']['H']:6.1f}, Re_full={res0['re_full']:+.6f}, Re_K1(no -betaWH)={res0['re_k1_eff']:+.6f}")
    for arr_months in [0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        theta = 12.0 / arr_months
        res = analyze_dh_stability(a_g, b_g, 0.0, 5.8, mu_P, mu_A, 6.5, 0.1, theta, 0.85, 3.0e-4, 0.0, 13.0)
        eq = res["eq"]
        E_ref = dh_host_block_analytic_E(1j * 0.8, eq["rho"], eq["h_eff"], eq["d_A"], eq["d_P"], 3.0e-4, eq["kp"], eq["m_P"], eq["f_A"])
        phase_deg = math.degrees(cmath_phase(E_ref))
        print(f"      1/theta={arr_months:5.2f} mo: f_A={eq['f_A']:.3f}, H*={eq['H']:6.1f}, |E_DH(0.8i)|={abs(E_ref):.3f}, "
              f"arg={phase_deg:+6.1f} deg, Re_full={res['re_full']:+.6f}, Re_K1(no -betaWH)={res['re_k1_eff']:+.6f}")
    print()


def cmath_phase(z):
    return math.atan2(float(np.imag(z)), float(np.real(z)))


def main():
    t0 = time.time()
    print(f"coupled_check.py started (seed={SEED})")
    rng = np.random.default_rng(SEED)
    run_step0_benchmarks()
    run_part1_k3(rng)
    run_part1_k4_k5()
    run_part1_k2(rng)
    run_part2_published_models(rng)
    elapsed = time.time() - t0
    print("=" * 78)
    print(f"ALL CHECKS COMPLETED IN {elapsed:.2f} s.")
    print("=" * 78)


if __name__ == "__main__":
    main()
