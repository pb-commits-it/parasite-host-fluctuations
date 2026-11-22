"""Clean-room statistical estimators, exact second-order formulas, and stochastic
2x2 life-cycle threshold-loss solver for independent replication of case study `lpd`.

Written from scratch from the mathematical specifications in `protocol.md` without
importing or copying any original study scripts.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# 1. Time-series OLS detrending, empirical ACF, spectral slope, and bias nulls
# ---------------------------------------------------------------------------

PHI_CAL_GRID = np.linspace(-0.95, 0.98, 194)
_GAUSS_SPEC_REF = np.random.default_rng(2026111599).standard_normal((200, 1024))


def fit_ols_line(t_steps: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (residuals, intercept, slope) for y_k = alpha + beta * t_k + e_k."""
    t_bar = float(np.mean(t_steps))
    y_bar = float(np.mean(y))
    dt = t_steps - t_bar
    dy = y - y_bar
    ss_t = float(dt @ dt)
    slope = float(dt @ dy) / ss_t if ss_t > 0.0 else 0.0
    intercept = y_bar - slope * t_bar
    return y - (intercept + slope * t_steps), intercept, slope


def orthonormal_columns(X: np.ndarray) -> np.ndarray:
    """Return an (n, p_eff) orthonormal basis Q for the column space of X."""
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] <= 1e-14:
        return np.zeros((X.shape[0], 0), dtype=float)
    rank_mask = s > (s[0] * 1e-11)
    return u[:, rank_mask]


def wn_null_r1_linear(t_steps: np.ndarray) -> float:
    """Exact white-noise ratio-of-expectations mean of lag-1 autocorrelation after
    OLS linear detrending on integer grid t_steps (protocol.md §3.1 item 4).
    """
    n = len(t_steps)
    if n <= 3:
        return 0.0
    adj = np.flatnonzero(np.diff(t_steps) == 1.0)
    m1 = len(adj)
    if m1 == 0:
        return 0.0
    tc = t_steps - float(np.mean(t_steps))
    ss_t = float(tc @ tc)
    h_cross = (m1 / float(n)) + (float(tc[adj] @ tc[adj + 1]) / ss_t if ss_t > 0.0 else 0.0)
    return -float(n) / float((n - 2) * m1) * h_cross


def wn_null_r1_projected(t_steps: np.ndarray, Q: np.ndarray) -> float:
    """Exact white-noise ratio-of-expectations mean of lag-1 autocorrelation after
    projecting out orthonormal basis Q of shape (n, p).
    """
    n = len(t_steps)
    p = Q.shape[1]
    if n <= p + 1:
        return 0.0
    adj = np.flatnonzero(np.diff(t_steps) == 1.0)
    m1 = len(adj)
    if m1 == 0:
        return 0.0
    trace_cross = float(np.sum(Q[adj] * Q[adj + 1])) if p > 0 else 0.0
    return -float(n) / float((n - p) * m1) * trace_cross


def expected_r1_ratio_ar1_projected(
    t_steps: np.ndarray, Q: np.ndarray, phis: np.ndarray = PHI_CAL_GRID
) -> np.ndarray:
    """Exact O(n * p) evaluation of E_phi[numerator] / E_phi[denominator] for an AR(1)
    process on arbitrary integer steps t_steps after projecting out orthonormal Q (n x p).

    For unit-marginal-variance AR(1), Cov(z_a, z_b) = phi^{|t_b - t_a|}.
    With M = I - Q Q^T and e = M z:
      E[e^T e / n] = (n - tr(Q^T Sigma Q)) / n,
      E[e_a e_b]   = Sigma_{ab} - Q_a . (Sigma Q)_b - (Sigma Q)_a . Q_b + Q_a . (Q^T Sigma Q) Q_b.
    We compute W = Sigma Q in O(n * p) via one forward and one backward Markov sweep.
    """
    n = len(t_steps)
    m_phi = len(phis)
    p = Q.shape[1]
    gaps = np.diff(t_steps)
    adj = np.flatnonzero(gaps == 1.0)
    if len(adj) == 0:
        return np.zeros(m_phi, dtype=float)
    if p == 0:
        return phis.copy()

    step_decay = phis[:, None] ** gaps[None, :]  # (m_phi, n - 1)
    W = np.empty((m_phi, n, p), dtype=float)

    # Forward pass: sum_{j <= k} phi^{t_k - t_j} Q_j
    fwd = np.broadcast_to(Q[0], (m_phi, p)).copy()
    W[:, 0, :] = fwd
    for k in range(1, n):
        fwd = step_decay[:, k - 1 : k] * fwd + Q[k]
        W[:, k, :] = fwd

    # Backward pass: sum_{j > k} phi^{t_j - t_k} Q_j
    bwd = np.zeros((m_phi, p), dtype=float)
    for k in range(n - 2, -1, -1):
        bwd = step_decay[:, k : k + 1] * (bwd + Q[k + 1])
        W[:, k, :] += bwd

    qt_sigma_q = np.einsum("ka,mkb->mab", Q, W)  # (m_phi, p, p)
    tr_term = np.einsum("maa->m", qt_sigma_q)
    e_c0 = (float(n) - tr_term) / float(n)

    qa = Q[adj]
    qb = Q[adj + 1]
    cross_1 = np.einsum("ia,mia->mi", qa, W[:, adj + 1, :])
    cross_2 = np.einsum("mia,ia->mi", W[:, adj, :], qb)
    quad = np.einsum("ia,mab,ib->mi", qa, qt_sigma_q, qb)
    e_num = np.mean(phis[:, None] - cross_1 - cross_2 + quad, axis=1)
    return e_num / np.maximum(e_c0, 1e-15)


def linear_detrend_basis(t_steps: np.ndarray) -> np.ndarray:
    """Orthonormal 2-column basis for [1, t_steps]."""
    n = len(t_steps)
    u0 = np.full(n, 1.0 / math.sqrt(float(n)))
    tc = t_steps - float(np.mean(t_steps))
    norm_tc = math.sqrt(float(tc @ tc))
    if norm_tc > 0.0:
        return np.column_stack([u0, tc / norm_tc])
    return u0[:, None]


def calibrate_ar1_phi(t_steps: np.ndarray, r1_hat: float, Q: np.ndarray | None = None) -> float:
    """Invert E_phi[num]/E_phi[den] = r1_hat on [-0.95, 0.98]."""
    basis = linear_detrend_basis(t_steps) if Q is None else Q
    curve = expected_r1_ratio_ar1_projected(t_steps, basis, PHI_CAL_GRID)
    return float(np.interp(r1_hat, curve, PHI_CAL_GRID))


def wn_null_spectral_slope(t_steps: np.ndarray) -> float:
    """200-draw Gaussian white-noise expectation of detrended log-periodogram slope on t_steps."""
    n = len(t_steps)
    span = int(t_steps[-1] - t_steps[0]) + 1
    k_max = (span - 1) // 2
    if k_max < 4:
        return 0.0
    if n <= _GAUSS_SPEC_REF.shape[1]:
        draws = _GAUSS_SPEC_REF[:, :n]
    else:
        draws = np.random.default_rng(2026111500 + n).standard_normal((200, n))
    tc = t_steps - float(np.mean(t_steps))
    ss_t = float(tc @ tc)
    centered = draws - np.mean(draws, axis=1, keepdims=True)
    resid = centered - ((centered @ tc)[:, None] / ss_t) * tc[None, :] if ss_t > 0.0 else centered

    padded = np.zeros((200, span), dtype=float)
    padded[:, (t_steps - t_steps[0]).astype(int)] = resid
    spec = (np.abs(np.fft.rfft(padded, axis=1)[:, 1 : k_max + 1]) ** 2) / float(n)
    log_f = np.log(np.arange(1, k_max + 1, dtype=float) / float(span))
    log_f_c = log_f - float(np.mean(log_f))
    ss_f = float(log_f_c @ log_f_c)
    log_s = np.log(np.maximum(spec, 1e-30))
    log_s_c = log_s - np.mean(log_s, axis=1, keepdims=True)
    return float(np.mean((log_s_c @ log_f_c) / ss_f))


def estimate_series_quantities(
    t_steps: np.ndarray, y: np.ndarray, detrend: bool = True, max_lag: int = 10
) -> dict:
    """Compute per-series variance, ACF (lags 1..10), and log-periodogram spectral slope."""
    n = len(t_steps)
    if detrend:
        e, _, beta = fit_ols_line(t_steps, y)
        sigma2 = float((e @ e) / float(n - 2))
    else:
        e = y - float(np.mean(y))
        beta = 0.0
        sigma2 = float((e @ e) / float(n - 1))
    sigma2_raw = float(np.var(y, ddof=1))
    c0 = float((e @ e) / float(n))

    idx_by_step = {int(ts): idx for idx, ts in enumerate(t_steps)}
    lag_cap = min(max_lag, n // 3)
    acf: dict[int, float | None] = {}
    for lag in range(1, max_lag + 1):
        if lag > lag_cap:
            acf[lag] = None
            continue
        ia = [i for i, ts in enumerate(t_steps) if (int(ts) + lag) in idx_by_step]
        if len(ia) >= 6 and c0 > 1e-15:
            ib = [idx_by_step[int(t_steps[i]) + lag] for i in ia]
            acf[lag] = float((e[ia] @ e[ib]) / (float(len(ia)) * c0))
        else:
            acf[lag] = None

    span = int(t_steps[-1] - t_steps[0]) + 1
    padded = np.zeros(span, dtype=float)
    padded[(t_steps - t_steps[0]).astype(int)] = e
    k_max = (span - 1) // 2
    b_spec = float("nan")
    if k_max >= 4:
        freqs = np.arange(1, k_max + 1, dtype=float) / float(span)
        periodogram = (np.abs(np.fft.rfft(padded)[1 : k_max + 1]) ** 2) / float(n)
        ok = periodogram > 1e-30
        if int(np.sum(ok)) >= 4:
            lf = np.log(freqs[ok])
            lp = np.log(periodogram[ok])
            lf_c = lf - float(np.mean(lf))
            b_spec = float((lf_c @ (lp - float(np.mean(lp)))) / float(lf_c @ lf_c))

    return {
        "n": n,
        "span": span,
        "sigma2": sigma2,
        "sigma": math.sqrt(max(sigma2, 0.0)),
        "sigma2_raw": sigma2_raw,
        "sigma_raw": math.sqrt(max(sigma2_raw, 0.0)),
        "trend_slope": beta,
        "r1": acf[1],
        "acf": acf,
        "b_spec": b_spec,
    }


# ---------------------------------------------------------------------------
# 2. Closed-form theoretical quantities (protocol.md §3.2)
# ---------------------------------------------------------------------------

def gamma0_ring_two_stage(a: float, b: float, sigma: float, phi: float) -> float:
    """Second-order threshold loss Gamma_0(a, b; sigma, phi) (Statement ring-B2)."""
    if sigma <= 0.0:
        return 0.0
    t0 = a + b
    if abs(phi) < 1e-15:
        return (sigma * sigma) / (2.0 * t0)
    ra = phi / (a - (a - 1.0) * phi)
    rb = phi / (b - (b - 1.0) * phi)
    re = ra * rb
    return (sigma * sigma) * (((1.0 + phi) / (1.0 - phi)) / (2.0 * t0) - re / (1.0 - re))


def delta_persistence_index_P2(a: float, b: float, sigma: float, phi: float, ln_r0: float) -> float:
    """Diffusion persistence-index difference E(a, b) - E(a) (Result 4 / Lean P2)."""
    if sigma <= 1e-12:
        return 1.0
    theta = phi / (1.0 - phi)
    one_plus_2theta = (1.0 + phi) / (1.0 - phi)
    denom = a * theta + b * (a + theta)
    if abs(denom) < 1e-12:
        return 1.0
    return (2.0 * b / one_plus_2theta) * ((ln_r0 / (sigma * sigma)) - (a * theta) / denom)


_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(64)
_STD_NORM_Z = _GH_NODES * math.sqrt(2.0)
_STD_NORM_W = _GH_WEIGHTS / math.sqrt(math.pi)


def one_stage_unexpanded_loss(a: float, sigma: float) -> float:
    """Exact unexpanded threshold loss D(a; sigma) for a 1-stage geometric stay a >= 1
    with mean-one lognormal multiplier exp(sigma*Z - sigma^2/2).
    """
    if sigma <= 0.0:
        return 0.0
    if abs(a - 1.0) < 1e-14:
        return 0.5 * sigma * sigma
    s0 = 1.0 - 1.0 / a
    g0 = 1.0 / a
    log_mult = sigma * _STD_NORM_Z - 0.5 * sigma * sigma

    def root_fn(y: float) -> float:
        return float(np.sum(_STD_NORM_W * np.log(s0 + g0 * np.exp(y + log_mult))))

    return float(brentq(root_fn, 0.0, 0.5 * sigma * sigma + 1e-9, xtol=1e-14, rtol=1e-14))


def contact_fine_resolvent_d2(b_coarse: float, sigma: float, phi_coarse: float) -> float:
    """Exact matrix-resolvent second-order threshold loss D_2 for Sensitivity S9b
    (`contact_fine`: fine step dt = 1/4, exposed mean 4.0 with death = 0.125 and contact=True,
    long stage mean 4*b_coarse, bounded host with Var(h) = exp(sigma^2) - 1, phi_f = phi^{1/4}).
    Used in Rule 7 check V4d before empirical use.
    """
    if sigma <= 0.0:
        return 0.0
    phi_f = (phi_coarse ** 0.25) if phi_coarse > 0.0 else 0.0
    bf = 4.0 * b_coarse
    # Critical mean matrix M0 at birth_scale = 2.0:
    # [ 0.75,   2/(4b) ]
    # [ 0.125,  1 - 1/(4b) ]
    M0 = np.array([[0.75, 2.0 / bf], [0.125, 1.0 - 1.0 / bf]], dtype=float)
    # Right eigenvector v (sum 1): 0.25 v0 = (2/bf) v1 => v0 / v1 = 8 / bf
    v = np.array([8.0, bf], dtype=float)
    v /= np.sum(v)
    # Left eigenvector w (w @ M0 = w, w @ v = 1):
    # -0.25 w0 + 0.125 w1 = 0 => w1 = 2 w0
    w = np.array([1.0, 2.0], dtype=float)
    w /= float(w @ v)
    b_ref = float(w @ np.array([[0.0, 2.0 / bf], [0.0, 0.0]]) @ v)
    # Sensitivity to standardized host fluctuation (h - 1): sd_h = sqrt(exp(sigma^2) - 1) ~ sigma at second order
    # Note: Engine.coefficient() defines second-order expansion in sigma (where Var(h) = sigma^2 + O(sigma^4))
    B = sigma * np.array([[-0.125, 0.0], [0.125, 0.0]], dtype=float)
    Q_proj = np.eye(2) - np.outer(v, w)
    T_op = M0 @ Q_proj
    c_bar = float(w @ B @ v)
    rhs = Q_proj @ (B @ v)
    sol = np.linalg.solve(np.eye(2) - phi_f * T_op, rhs)
    C_step = 0.5 * c_bar * c_bar - phi_f * float(w @ B @ sol)
    return C_step / b_ref


# ---------------------------------------------------------------------------
# 3. Clean-room 2x2 Stochastic Life-Cycle Simulator (NumPy + optional C kernel)
# ---------------------------------------------------------------------------

def sample_lognormal_ar1(
    t_total: int, pairs: int, sigma: float, phi: float, seed: int
) -> np.ndarray:
    """Generate (t_total, 2 * pairs) antithetic stationary lognormal AR(1) multipliers."""
    rng = np.random.default_rng(seed)
    z_init = rng.standard_normal(pairs)
    eps = rng.standard_normal((t_total, pairs)) * math.sqrt(max(0.0, 1.0 - phi * phi))
    z_init_all = np.concatenate([z_init, -z_init])
    eps_all = np.concatenate([eps, -eps], axis=1)
    if abs(phi) > 0.0:
        z, _ = lfilter([1.0], [1.0, -phi], eps_all, axis=0, zi=(phi * z_init_all)[None, :])
    else:
        z = eps_all
    return np.exp(sigma * z - 0.5 * sigma * sigma)


def sample_bounded_markov(
    t_total: int, pairs: int, sigma: float, phi: float, low: float, seed: int
) -> np.ndarray:
    """Generate (t_total, 2 * pairs) antithetic stationary two-state Markov multipliers
    matching mean 1, variance exp(sigma^2) - 1, lower state `low`, and autocorrelation `phi >= 0`.
    """
    if phi < 0.0:
        raise ValueError("Bounded contact mode requires phi >= 0")
    var_h = math.expm1(sigma * sigma)
    if var_h <= 0.0:
        return np.ones((t_total, 2 * pairs), dtype=float)
    a = 1.0 - low
    high = 1.0 + var_h / a
    p_high = (a * a) / (a * a + var_h)
    p_low = 1.0 - p_high
    levels = np.array([low, high], dtype=float)

    rng = np.random.default_rng(seed)
    u_draw = rng.random((t_total, pairs))
    u_draw_all = np.concatenate([u_draw, 1.0 - u_draw], axis=1)
    u_ref = rng.random((t_total, pairs))
    u_ref_all = np.concatenate([u_ref, 1.0 - u_ref], axis=1)

    drawn_state = (u_draw_all >= p_low).astype(np.int64)
    refresh = u_ref_all < (1.0 - phi)
    refresh[0, :] = True
    step_idx = np.arange(t_total, dtype=np.int64)[:, None]
    last_refresh = np.maximum.accumulate(np.where(refresh, step_idx, 0), axis=0)
    col_idx = np.arange(2 * pairs, dtype=np.int64)[None, :]
    states = drawn_state[last_refresh, col_idx]
    return levels[states]


def _products_2x2_numpy(
    a00_t: np.ndarray,
    a10_t: np.ndarray,
    a01_base: float,
    a11_const: float,
    p_init: float,
    y: float,
    burn: int,
    pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy 2x2 Lyapunov growth & derivative accumulator on normalized state p_t = x0_t.
    a00_t, a10_t have shape (T, 2 * pairs).
    Returns paired (logs, slopes) each of shape (pairs,).
    """
    t_total, reps = a00_t.shape
    p = np.full(reps, p_init, dtype=float)
    q = np.zeros(reps, dtype=float)
    logs = np.zeros(reps, dtype=float)
    slopes = np.zeros(reps, dtype=float)
    a01_y = a01_base * math.exp(y)

    for t in range(t_total):
        a00 = a00_t[t]
        a10 = a10_t[t]
        one_minus_p = 1.0 - p
        u0 = a00 * p + a01_y * one_minus_p
        u1 = a10 * p + a11_const * one_minus_p
        tot = u0 + u1
        du0 = a00 * q + a01_y * (one_minus_p - q)
        du1 = (a10 - a11_const) * q
        dtot = du0 + du1
        inv_tot = 1.0 / tot
        p_next = u0 * inv_tot
        der = dtot * inv_tot
        q = (du0 - p_next * dtot) * inv_tot
        p = p_next
        if t >= burn:
            logs += np.log(tot)
            slopes += der

    keep = float(t_total - burn)
    logs /= keep
    slopes /= keep
    return 0.5 * (logs[:pairs] + logs[pairs:]), 0.5 * (slopes[:pairs] + slopes[pairs:])


def build_2x2_model_arrays(
    mode: str, b_coarse: float, sigma: float, phi_coarse: float, steps: int, burn: int, pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Construct (a00_t, a10_t, a01_base, a11_const, p_init, jensen_bound) for a 2x2 model."""
    t_total = steps + burn
    if mode == "primary_coarse":
        h = sample_lognormal_ar1(t_total, pairs, sigma, phi_coarse, seed)
        a00_t = np.zeros_like(h)
        a10_t = h
        a01_base = 1.0 / b_coarse
        a11_const = 1.0 - 1.0 / b_coarse
        p_init = 1.0 / (1.0 + b_coarse)
        j_bound = 0.5 * sigma * sigma
        return a00_t, a10_t, a01_base, a11_const, p_init, j_bound

    if mode == "primary_fine":
        pf = (phi_coarse ** 0.25) if phi_coarse > 0.0 else 0.0
        bf = 4.0 * b_coarse
        h = sample_lognormal_ar1(t_total, pairs, sigma, pf, seed)
        a00_t = np.full_like(h, 0.75)
        a10_t = 0.25 * h
        a01_base = 1.0 / bf
        a11_const = 1.0 - 1.0 / bf
        p_init = 4.0 / (4.0 + bf)
        j_bound = 0.5 * sigma * sigma
        return a00_t, a10_t, a01_base, a11_const, p_init, j_bound

    if mode == "contact_fine":
        pf = (phi_coarse ** 0.25) if phi_coarse > 0.0 else 0.0
        bf = 4.0 * b_coarse
        h = sample_bounded_markov(t_total, pairs, sigma, pf, 0.05, seed)
        a00_t = 0.875 - 0.125 * h
        if float(np.min(a00_t)) <= 1e-12:
            raise ValueError(f"Inadmissible contact_fine variance at sigma={sigma}")
        a10_t = 0.125 * h
        a01_base = 2.0 / bf
        a11_const = 1.0 - 1.0 / bf
        p_init = 8.0 / (8.0 + bf)
        # Edge Jensen bound for bounded contact stage
        var_h = math.expm1(sigma * sigma)
        if var_h > 0.0:
            high = 1.0 + var_h / 0.95
            ph = (0.95 * 0.95) / (0.95 * 0.95 + var_h)
            pl = 1.0 - ph
            # Staying edge (weight 0.75/0.125 = 6 per birth) + contact exit edge (weight 1 per birth)
            j_stay = -6.0 * (pl * math.log((0.875 - 0.125 * 0.05) / 0.75) + ph * math.log((0.875 - 0.125 * high) / 0.75))
            j_exit = -(pl * math.log(0.05) + ph * math.log(high))
            j_bound = j_stay + j_exit
        else:
            j_bound = 0.0
        return a00_t, a10_t, a01_base, a11_const, p_init, j_bound

    raise ValueError(f"Unsupported mode: {mode}")


def solve_threshold_loss_2x2(
    mode: str,
    b_coarse: float,
    sigma: float,
    phi_coarse: float,
    steps: int = 24000,
    burn: int = 1600,
    pairs: int = 12,
    seed: int = 2026111555,
    tol: float = 1e-7,
    product_kernel=_products_2x2_numpy,
) -> dict:
    """Compute exact unexpanded stochastic threshold loss D via Newton + bracketed fallback
    on common random numbers.
    """
    if sigma <= 0.0:
        return {"D": 0.0, "se": 0.0, "J": 0.0, "residual": 0.0, "method": "zero"}

    a00_t, a10_t, a01_base, a11_const, p_init, j_bound = build_2x2_model_arrays(
        mode, b_coarse, sigma, phi_coarse, steps, burn, pairs, seed
    )
    r0, s0 = product_kernel(a00_t, a10_t, a01_base, a11_const, p_init, 0.0, burn, pairs)
    y = float(np.clip(-float(np.mean(r0)) / float(np.mean(s0)), -3.0, 3.0))

    converged = False
    r_cur, s_cur = r0, s0
    corr = 0.0
    for _ in range(64):
        r_cur, s_cur = product_kernel(a00_t, a10_t, a01_base, a11_const, p_init, y, burn, pairs)
        corr = float(np.mean(r_cur)) / float(np.mean(s_cur))
        if abs(corr) < tol:
            converged = True
            break
        y -= float(np.clip(corr, -2.0, 2.0))

    method_tag = "newton"
    if not converged:
        method_tag = "brentq_fallback"

        def g_mean(y_try: float) -> float:
            rr, _ = product_kernel(a00_t, a10_t, a01_base, a11_const, p_init, y_try, burn, pairs)
            return float(np.mean(rr))

        lo = -max(1.0, j_bound + 1.0)
        hi = max(1.0, j_bound + 1.0)
        for _ in range(12):
            if g_mean(lo) < 0.0 < g_mean(hi):
                break
            lo *= 2.0
            hi *= 2.0
        y = float(brentq(g_mean, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=100))
        r_cur, s_cur = product_kernel(a00_t, a10_t, a01_base, a11_const, p_init, y, burn, pairs)
        corr = float(np.mean(r_cur)) / float(np.mean(s_cur))

    samples = y - r_cur / float(np.mean(s_cur))
    se = float(np.std(samples, ddof=1) / math.sqrt(float(pairs)))
    return {
        "D": y,
        "se": se,
        "J": j_bound,
        "residual": abs(corr),
        "method": method_tag,
    }


class CalibratedSurface2D:
    """Amplitude-aware D / sigma^2 grid interpolator with optional direct outside-box solver."""

    def __init__(
        self,
        mode: str,
        b_coarse: float,
        sigmas: list[float],
        phis: list[float],
        steps: int,
        burn: int,
        pairs: int,
        base_seed: int,
        product_kernel=_products_2x2_numpy,
    ) -> None:
        self.mode = mode
        self.b_coarse = b_coarse
        self.sigmas = np.array(sigmas, dtype=float)
        self.phis = np.array(phis, dtype=float)
        self.steps = steps
        self.burn = burn
        self.pairs = pairs
        self.base_seed = base_seed
        self.product_kernel = product_kernel

        ns, np_ = len(sigmas), len(phis)
        self.D_grid = np.zeros((ns, np_), dtype=float)
        self.se_grid = np.zeros((ns, np_), dtype=float)

        for i_s, sig in enumerate(self.sigmas):
            if sig <= 0.0:
                continue
            for i_p, phi in enumerate(self.phis):
                res = solve_threshold_loss_2x2(
                    mode, b_coarse, float(sig), float(phi),
                    steps=steps, burn=burn, pairs=pairs,
                    seed=base_seed + i_s * 31 + i_p,
                    product_kernel=product_kernel,
                )
                self.D_grid[i_s, i_p] = res["D"]
                self.se_grid[i_s, i_p] = res["se"]

        scaled = np.zeros_like(self.D_grid)
        scaled[1:, :] = self.D_grid[1:, :] / (self.sigmas[1:, None] ** 2)
        # Exact small-sigma quadratic limit at sigma = 0 from first positive sigma row / analytical D2
        if mode == "primary_coarse":
            scaled[0, :] = np.array([gamma0_ring_two_stage(1.0, b_coarse, 1.0, float(p)) for p in self.phis])
        elif mode == "primary_fine":
            scaled[0, :] = np.array([
                gamma0_ring_two_stage(4.0, 4.0 * b_coarse, 1.0, float(p) ** 0.25 if p > 0 else 0.0)
                for p in self.phis
            ])
        elif mode == "contact_fine":
            scaled[0, :] = np.array([contact_fine_resolvent_d2(b_coarse, 1.0, float(p)) for p in self.phis])

        self._rgi_scaled = RegularGridInterpolator((self.sigmas, self.phis), scaled, bounds_error=True)
        self._rgi_se = RegularGridInterpolator((self.sigmas, self.phis), self.se_grid, bounds_error=True)
        self._outside_cache: dict[tuple[float, float], dict] = {}

    def evaluate(self, sigma: float, phi: float, allow_outside_direct: bool = True) -> dict:
        if sigma <= 0.0:
            return {"D": 0.0, "se": 0.0, "source": "zero"}
        in_box = (self.sigmas[0] <= sigma <= self.sigmas[-1]) and (self.phis[0] <= phi <= self.phis[-1])
        if in_box:
            pt = [[sigma, phi]]
            d_val = float(self._rgi_scaled(pt)[0]) * (sigma * sigma)
            se_val = float(self._rgi_se(pt)[0])
            return {"D": d_val, "se": se_val, "source": "interpolated_D_over_sigma2"}
        if not allow_outside_direct:
            return {"D": None, "se": None, "source": "outside_grid"}
        key = (round(sigma, 12), round(phi, 12))
        if key not in self._outside_cache:
            res = solve_threshold_loss_2x2(
                self.mode, self.b_coarse, sigma, phi,
                steps=self.steps, burn=self.burn, pairs=self.pairs,
                seed=self.base_seed + 9999,
                product_kernel=self.product_kernel,
            )
            self._outside_cache[key] = {"D": res["D"], "se": res["se"], "source": "direct_outside"}
        return self._outside_cache[key]


# ---------------------------------------------------------------------------
# 4. Study-cluster bootstrap engine (protocol.md §3.1 item 6, §4.1)
# ---------------------------------------------------------------------------

def cluster_bootstrap_summary(
    records: list[dict],
    bool_keys: list[str],
    dist_keys: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 2026111501,
) -> dict:
    """Cluster-resample independent study/source clusters (`study_id`) `n_boot` times
    and evaluate exact registered T1 and T2 decision rules.
    """
    m_series = len(records)
    if m_series == 0:
        return {"M": 0, "K": 0, "T1": "not decided", "T2": "not decided"}

    clusters = sorted({r["study_id"] for r in records})
    k_clusters = len(clusters)
    c_map = {cid: idx for idx, cid in enumerate(clusters)}
    g_idx = np.array([c_map[r["study_id"]] for r in records], dtype=np.int64)

    rng = np.random.default_rng(seed)
    mult = rng.multinomial(k_clusters, np.full(k_clusters, 1.0 / float(k_clusters)), size=n_boot)  # (B, K)

    def _mean_and_ci(vals_raw: np.ndarray) -> dict:
        ok = np.isfinite(vals_raw)
        n_ok = int(np.sum(ok))
        if n_ok == 0:
            return {"estimate": None, "ci95": [None, None], "denominator": 0, "eligible_clusters": 0}
        cnt_per_c = np.bincount(g_idx, weights=ok.astype(float), minlength=k_clusters)
        sum_per_c = np.bincount(g_idx, weights=np.where(ok, vals_raw, 0.0), minlength=k_clusters)
        boot_den = mult @ cnt_per_c
        boot_num = mult @ sum_per_c
        boot_means = np.divide(boot_num, boot_den, out=np.full(n_boot, np.nan), where=boot_den > 0)
        valid_b = boot_means[np.isfinite(boot_means)]
        ci = np.quantile(valid_b, [0.025, 0.975]).tolist() if valid_b.size else [None, None]
        return {
            "estimate": float(np.mean(vals_raw[ok])),
            "ci95": [float(ci[0]), float(ci[1])] if ci[0] is not None else [None, None],
            "denominator": n_ok,
            "eligible_clusters": int(np.sum(cnt_per_c > 0)),
        }

    delta_arr = np.array(
        [float(r["delta_r1"]) if r.get("delta_r1") is not None and math.isfinite(r["delta_r1"]) else np.nan for r in records]
    )
    mean_delta = _mean_and_ci(delta_arr)
    med_r1_bc = float(np.median([float(r["r1_bc"]) for r in records]))

    fractions: dict[str, dict] = {}
    for bk in bool_keys:
        arr = np.array(
            [float(r[bk]) if isinstance(r.get(bk), (bool, np.bool_)) else np.nan for r in records],
            dtype=float,
        )
        fractions[bk] = _mean_and_ci(arr)

    # Registered Test T1 criterion (protocol.md §4.1)
    t1_label = "not decided"
    lo_t1, hi_t1 = mean_delta["ci95"]
    if k_clusters >= 5 and m_series >= 25 and lo_t1 is not None and hi_t1 is not None:
        if lo_t1 > 0.0 and med_r1_bc > 0.0:
            t1_label = "supports"
        elif hi_t1 < 0.0:
            t1_label = "contradicts"

    # Registered Test T2 criterion (protocol.md §4.1)
    t2_label = "not decided"
    f_long = fractions.get("long_strict_failure")
    f_mem = fractions.get("short_penalty_zone_T2")
    if f_long and f_mem and f_long["ci95"][0] is not None and f_mem["ci95"][0] is not None:
        if f_long["ci95"][0] > 0.20 or f_mem["ci95"][0] > 0.20:
            t2_label = "supports"
        elif f_long["ci95"][1] < 0.10 and f_mem["ci95"][1] < 0.10:
            t2_label = "contradicts"

    distributions: dict[str, dict] = {}
    if dist_keys:
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        for dk in dist_keys:
            x = np.array(
                [float(r[dk]) if r.get(dk) is not None and isinstance(r.get(dk), (int, float)) and math.isfinite(r[dk]) else np.nan for r in records],
                dtype=float,
            )
            ok = np.isfinite(x)
            if not np.any(ok):
                continue
            order = np.argsort(x[ok])
            xs = x[ok][order]
            gs = g_idx[ok][order]
            boot_q = np.empty((n_boot, len(qs)), dtype=float)
            for b0 in range(0, n_boot, 100):
                b1 = min(n_boot, b0 + 100)
                w = mult[b0:b1, gs]
                csum = np.cumsum(w, axis=1)
                tot = csum[:, -1]
                for jq, q in enumerate(qs):
                    idx = np.argmax(csum >= q * tot[:, None], axis=1)
                    vals = xs[idx]
                    vals[tot == 0] = np.nan
                    boot_q[b0:b1, jq] = vals
            q_out = {}
            for jq, q in enumerate(qs):
                vb = boot_q[:, jq]
                vb = vb[np.isfinite(vb)]
                ci_q = np.quantile(vb, [0.025, 0.975]).tolist() if vb.size else [None, None]
                q_out[str(q)] = {
                    "estimate": float(np.quantile(xs, q, method="inverted_cdf")),
                    "ci95": ci_q,
                }
            distributions[dk] = {
                "n": int(np.sum(ok)),
                "mean": float(np.mean(xs)),
                "median_standard": float(np.median(xs)),
                "quantiles_inverted_cdf": q_out,
            }

    return {
        "M": m_series,
        "K": k_clusters,
        "mean_delta_r1": mean_delta,
        "median_r1_bc": med_r1_bc,
        "T1": t1_label,
        "T2": t2_label,
        "fractions": fractions,
        "distributions": distributions,
    }
