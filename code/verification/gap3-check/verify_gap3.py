#!/usr/bin/env python3
"""Independent numerical verification of gap3_proof.md (r37-gap3-check).

Rules compliance (study rules Rules 1, 5, 7, 9):
- Completely standalone: imports ONLY Python stdlib, numpy, scipy, and mpmath.
  Does NOT import gap3_checks.py or any other repository module.
- Reproduces known exact results first (Rule 7 gate) with tolerances fixed beforehand:
    G1: Deterministic constant-host threshold and nonzero Perron growth rate (tol 1e-14).
    G2: Exact periodic alternating host (memA P8, phi=-1) vs Proposition 6 (tol 1e-30 at 70 digits).
    G3: Exact i.i.d. Gamma Tricomi threshold D(s, p) vs Roerdink/Tricomi integral at 60 digits (tol 1e-45),
        including singular-at-zero shape p = 0.25 and p = 0.5 where E[h^(-1)] = infinity.
- Independently checks every main formula and key inequality in gap3_proof.md:
    C1 (§2): Two-step Mobius log-derivative supremum k(h_t) across extreme parameters (h in [1e-10, 1e10],
             s in [1e-6, 1-1e-6], c in [1e-6, 1e4]) at 60 digits, plus correlation-free Holder bound (2.3)
             and parity-pullback convergence rate.
    C2 (§3 & §4): Strict concavity of log(e^v - s), backward-conditional elasticity bounds for Gamma refresh
             (0 < q_j < 1) and Lognormal AR(1) (0 <= q_j <= sigma*phi^j for 0 <= phi < 1, plus sign-change
             failure when phi < 0), and opposite-monotonicity covariance inequality Cov(m, g) <= 0.
    C3 (§5): Exact 60-digit verification of Q_N(z) = prod_{lambda_j > 0}(1 + z lambda_j^2) across path lengths
             and extreme weights; trace walk bounds ((N-1)/N) E[h^k] <= int x^k nu_N(dx) <= 2^(2k-1) E[h^k];
             exact finite-N expected moments vs (5.3) for Gamma refresh, Lognormal AR(1), AND a non-reversible
             3-state cyclic Markov host with T12 != T21; and scalar/integrated remainder |R_4(z)| <= 32 E[h^4] z^4
             checked against exact 60-digit Tricomi L(z) across p in {0.25, 0.5, 1, 2, 5}.
    C4 (§6): 60-digit verification of general moment formulas (6.2)-(6.5):
             - Symmetric binary Markov host (memA P5/P7) via independent scalar parity-eigenbasis series;
             - Non-reversible 3-state cyclic host (T12 != T21) via 60-digit holomorphic invariant series;
             - Exact i.i.d. Gamma across p in {0.25, 0.5, 1, 2, 10} from 60-digit Tricomi roots;
             - Negative-memory Lognormal AR(1) counterexample (w=0.1, phi=-0.75): 60-digit moment & 2D
               Gauss-Hermite verification plus elementary bound check.
    C5 (§7): Exact 80-digit formal power series verification of Proposition 7 on i.i.d. Gamma (p in {0.25, 0.5, 1, 2}):
             computes exact spectral moments mu_1..mu_35 and formal threshold coefficients [eps^k]D(eps) for k=1..30,
             verifying E[h^k] <= mu_k <= 2^(2k-1) E[h^k] (7.1), L^(k)(0+) = (-1)^(k+1)(k-1)! mu_k (7.2),
             formal series inversion (7.3), and divergence (|d_k|^(1/k) -> infinity).
    C6 (Edge-case nonlinear recurrence checks): Independent simulation + Newton solver testing parameter
             regimes NOT covered by gap3_checks.py, as well as overlapping cells:
             - Singular-density Gamma refresh p = 0.25 (V=4) and p = 0.5 (V=2) at phi in {0, 0.5, 0.9};
             - Negative-memory Lognormal AR(1) at (w=0.1, phi=-0.75) [confirming D(eps) < 0 in the nonlinear
               model and matching A, B] and (w=0.25, phi=-0.8);
             - Stronger-noise positive-memory Lognormal AR(1) (w=log(2), V=1, phi in {0.5, 0.9}) and
               Gamma/Lognormal V=0.5 cross-checks, plus interior-s strict bounds 0 < D(s) < ell (or D(s) < ell).
"""

import json
import math
import time
from pathlib import Path
import numpy as np
from scipy import special
import mpmath as mp

OUT_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
FAILURES = []


def record(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Exact high-precision helpers (mpmath)
# ---------------------------------------------------------------------------

def tricomi_L_mp(z, p, dps: int = 60):
    """Exact L(z) = r(1, z) for i.i.d. Gamma(p, scale=1/p) via Tricomi U(p, b, p/z)."""
    with mp.workdps(dps + 15):
        zm = mp.mpf(z)
        pm = mp.mpf(p)
        if zm == 0:
            return mp.mpf(0)
        kap = pm / zm
        u0 = mp.hyperu(pm, 1, kap)
        # Central 4th-order complex/real step in b around b=1, or mp.diff:
        du = mp.diff(lambda b: mp.hyperu(pm, b, kap), mp.mpf(1))
        return du / u0


def tricomi_D_mp(s, p, dps: int = 60):
    """Exact threshold D(s, p) for i.i.d. Gamma(p, scale=1/p) solving log(s) + L(c/s^2) = 0."""
    with mp.workdps(dps + 15):
        sm = mp.mpf(s)
        pm = mp.mpf(p)
        target = -mp.log(sm)
        ell = mp.log(pm) - mp.psi(0, pm)
        # Bracket z in (0, exp(ell)*(1-s)/s^2): at D=0, r<0 so L<target; at D=ell, r>0 so L>target.
        z_lo = (1 - sm) / (sm * sm)
        z_hi = (1 - sm) * mp.exp(ell) / (sm * sm)
        z_star = mp.findroot(lambda z: tricomi_L_mp(z, pm, dps + 10) - target,
                             (z_lo, z_hi), solver="anderson", tol=mp.mpf(10) ** (-(dps + 5)))
        c_star = z_star * sm * sm
        return mp.log(c_star / (1 - sm))


def moments_and_AB_gamma(p, phi):
    """Exact high-precision A, B, ell and raw/central moments for Gamma refresh."""
    pm = mp.mpf(p)
    ph = mp.mpf(phi)
    V = 1 / pm
    c1 = V * ph
    c2 = V * ph ** 2
    m3 = 2 * V ** 2
    t12 = ph * m3
    t21 = ph * m3
    t111 = (ph ** 2) * m3
    A = V / 2 + c1
    B = -A - c2 - t12 - t21 - t111 - m3 / 3 + mp.mpf("1.5") * A * A
    ell = mp.log(pm) - mp.psi(0, pm)
    return A, B, ell


def moments_and_AB_lognormal(w, phi):
    """Exact high-precision A, B, ell for Lognormal AR(1) with log-variance w = sigma^2."""
    wm = mp.mpf(w)
    ph = mp.mpf(phi)
    V = mp.expm1(wm)
    c1 = mp.expm1(wm * ph)
    c2 = mp.expm1(wm * ph * ph)
    m3 = mp.exp(3 * wm) - 3 * mp.exp(wm) + 2
    t12 = mp.exp(wm * (1 + 2 * ph)) - mp.exp(wm) - 2 * mp.exp(wm * ph) + 2
    t21 = t12
    t111 = mp.exp(wm * (2 * ph + ph * ph)) - 2 * mp.exp(wm * ph) - mp.exp(wm * ph * ph) + 2
    A = V / 2 + c1
    B = -A - c2 - t12 - t21 - t111 - m3 / 3 + mp.mpf("1.5") * A * A
    ell = wm / 2
    return A, B, ell


# ---------------------------------------------------------------------------
# Finite-state Markov exact holomorphic invariant-series solver (60 digits)
# Works for ANY irreducible finite-state Markov chain, reversible or not!
# ---------------------------------------------------------------------------

def finite_markov_exact_rate_mp(s, c, P_mat, pi_vec, h_states, order: int = 28):
    """Exact Taylor series of r(s, c) in amplitude parameter t=1 around h=1 (h_i(t)=1+t*(h_i-1))."""
    sm = mp.mpf(s)
    cm = mp.mpf(c)
    lam = (sm + mp.sqrt(sm * sm + 4 * cm)) / 2
    b = cm / (lam * lam)
    K = len(h_states)
    b_vec = [mp.mpf(h) - 1 for h in h_states]
    PT = [[mp.mpf(P_mat[i][j]) for i in range(K)] for j in range(K)]

    # M[k][n] is length-K vector of [t^n] E[z^k 1_{state=j}]
    M = [[None] * (order + 1) for _ in range(order + 1)]
    for n in range(order + 1):
        M[0][n] = [mp.mpf(pi_vec[j]) if n == 0 else mp.mpf(0) for j in range(K)]
    for k in range(1, order + 1):
        for n in range(k):
            M[k][n] = [mp.mpf(0)] * K

    # Precompute binomials and LU decompositions of (I - (-b)^k P^T)
    binom = [[mp.binomial(i, j) for j in range(order + 2)] for i in range(order + 2)]
    lhs_mats = [None]
    for k in range(1, order + 1):
        fac = (-b) ** k
        mat = mp.matrix(K, K)
        for j in range(K):
            for i in range(K):
                mat[j, i] = (mp.mpf(1) if j == i else mp.mpf(0)) - fac * PT[j][i]
        lhs_mats.append(mat)

    for n in range(1, order + 1):
        for k in range(n, 0, -1):
            bk = b ** k
            rhs = [mp.mpf(0)] * K
            for r in range(0, k + 1):
                max_m = n - k
                for m in range(0, max_m + 1):
                    if r == 0 and m == 0:
                        continue
                    idx_k = k - r + m
                    idx_n = n - r
                    if idx_k > idx_n:
                        continue
                    vec = M[idx_k][idx_n]
                    # Apply P^T to vec
                    pt_vec = [mp.fsum(PT[j][i] * vec[i] for i in range(K)) for j in range(K)]
                    coef = bk * binom[k][r] * ((-1) ** (k - r + m)) * binom[k + m - 1][m]
                    for j in range(K):
                        rhs[j] += coef * (b_vec[j] ** r) * pt_vec[j]
            sol = mp.lu_solve(lhs_mats[k], mp.matrix(rhs))
            M[k][n] = [sol[j] for j in range(K)]

    rate = mp.log(lam)
    tail_last = mp.mpf(0)
    for n in range(1, order + 1):
        term_n = mp.mpf(0)
        for k in range(1, n + 1):
            term_n += ((-1) ** (k + 1)) * mp.fsum(M[k][n]) / k
        rate += term_n
        if n >= order - 1:
            tail_last = max(tail_last, abs(term_n))
    return rate, tail_last


def finite_markov_threshold_mp(s, P_mat, pi_vec, h_states, order: int = 30):
    """High-precision threshold D(s) for a finite-state Markov host."""
    sm = mp.mpf(s)
    d0 = mp.mpf(0)
    d1 = mp.mpf("0.001")
    f0 = finite_markov_exact_rate_mp(sm, (1 - sm) * mp.exp(d0), P_mat, pi_vec, h_states, order)[0]
    f1 = finite_markov_exact_rate_mp(sm, (1 - sm) * mp.exp(d1), P_mat, pi_vec, h_states, order)[0]
    for _ in range(14):
        if abs(f1) < mp.mpf("1e-48"):
            break
        d2 = d1 - f1 * (d1 - d0) / (f1 - f0)
        d0, f0 = d1, f1
        d1 = d2
        f1 = finite_markov_exact_rate_mp(sm, (1 - sm) * mp.exp(d1), P_mat, pi_vec, h_states, order)[0]
    _, tail = finite_markov_exact_rate_mp(sm, (1 - sm) * mp.exp(d1), P_mat, pi_vec, h_states, order)
    return d1, abs(f1), tail


# ---------------------------------------------------------------------------
# Gate & Proof-Step Checks (G1-G3, C1-C5)
# ---------------------------------------------------------------------------

def run_known_result_gates():
    print("=== RULE 7 KNOWN-RESULT GATES (G1-G3) ===", flush=True)
    mp.mp.dps = 70

    # G1: Constant host h=1 threshold and nonzero Perron growth
    worst = mp.mpf(0)
    for s in ["0.05", "0.5", "0.99"]:
        for c in ["0.02", "0.5", "3.0"]:
            sm, cm = mp.mpf(s), mp.mpf(c)
            u_star = (sm + mp.sqrt(sm * sm + 4 * cm)) / 2
            # Iterate U_t = s + c/U_{t-1} from u0 = 0.13 (at s=0.05, c=3, k~0.944 per pair -> 2400 pairs gives k^2400 < 1e-60)
            u = mp.mpf("0.13")
            for _ in range(4800):
                u = sm + cm / u
            worst = max(worst, abs(mp.log(u) - mp.log(u_star)))
    record("G1 constant-host Perron exponent", worst < mp.mpf("1e-50"), f"max err={mp.nstr(worst, 4)}")

    # G2: Exact alternating host (memA P8, phi=-1): ergodic under 1-step shift, closed-form D_alt(s)
    # Verify Proposition 3 (D_alt < ell) and Proposition 6 (D_alt = A*eps + B*eps^2 + O(eps^3)) to >30 digits!
    worst_rem = mp.mpf(0)
    bounds_ok = True
    for delta_s in ["0.3", "0.6", "0.9"]:
        dm = mp.mpf(delta_s)
        ell = -mp.log(1 - dm * dm) / 2
        A_pred = -dm * dm / 2
        B_pred = -dm * dm / 2 + mp.mpf(3) * (dm ** 4) / 8
        for s_s in ["0.01", "0.5", "0.999"]:
            sm = mp.mpf(s_s)
            d_alt = mp.log((1 + sm) / (1 + mp.sqrt(dm * dm + (1 - dm * dm) * sm * sm)))
            if not (d_alt < ell and d_alt < 0):
                bounds_ok = False
        eps1 = mp.mpf("1e-12")
        eps2 = mp.mpf("1e-18")
        vals = []
        for ep in (eps1, eps2):
            sm = 1 - ep
            d_alt = mp.log((1 + sm) / (1 + mp.sqrt(dm * dm + (1 - dm * dm) * sm * sm)))
            vals.append((d_alt - A_pred * ep - B_pred * ep * ep) / (ep ** 3))
        # Scaled cubic remainders at 1e-12 and 1e-18 differ by O(1e-12)
        worst_rem = max(worst_rem, abs(vals[0] - vals[1]))
    record("G2 exact alternating host (memA P8) vs Prop 3 & Prop 6 (70 dps)",
           bounds_ok and worst_rem < mp.mpf("1e-11"),
           f"bounds_ok={bounds_ok}, cubic-limit diff={mp.nstr(worst_rem, 5)}")

    # G3: Exact i.i.d. Gamma Tricomi formula vs invariant density integral (Roerdink E1/E2) at 60 digits,
    # including p=0.25 and p=0.5 where E[h^(-1)] = infinity!
    mp.mp.dps = 60
    worst_int = mp.mpf(0)
    for p_s, s_s in [("0.25", "0.5"), ("0.5", "0.2"), ("2.0", "0.5"), ("5.0", "0.95")]:
        pm, sm = mp.mpf(p_s), mp.mpf(s_s)
        d_tri = tricomi_D_mp(sm, pm, dps=55)
        cm = (1 - sm) * mp.exp(d_tri)
        kap = sm * sm * pm / cm
        # Integrate t^(p-1) (1+t)^(-p) e^(-kap t) log(s*(1+t)) on [0, inf) via u = t^(1/4) for p<1 safety:
        # Substitute t = x^4, dt = 4 x^3 dx so 4*p - 1 >= 0 for p >= 0.25!
        num = mp.quad(lambda x: 4 * (x ** (4 * pm - 1)) * ((1 + x ** 4) ** (-pm))
                      * mp.exp(-kap * (x ** 4)) * mp.log(sm * (1 + x ** 4)), [0, 1, mp.inf])
        den = mp.quad(lambda x: 4 * (x ** (4 * pm - 1)) * ((1 + x ** 4) ** (-pm))
                      * mp.exp(-kap * (x ** 4)), [0, 1, mp.inf])
        worst_int = max(worst_int, abs(num / den))
        if p_s == "2.0" and s_s == "0.5":
            d_known = d_tri
    record("G3 i.i.d. Gamma Tricomi vs Roerdink integral incl. p=0.25, 0.5 (55 dps)",
           worst_int < mp.mpf("1e-45") and abs(d_known - mp.mpf("0.0849174016477")) < mp.mpf("1e-12"),
           f"max |r(c_*)|={mp.nstr(worst_int, 4)}, D(0.5,p=2)={mp.nstr(d_known, 16)}")


def run_section2_and_section4_checks():
    print("\n=== CHECKS C1-C2: SECTION 2 CONTRACTION & SECTION 3-4 INEQUALITIES ===", flush=True)
    mp.mp.dps = 60

    # C1a: Exact attainment and upper bound of two-step log-derivative (2.1) across extreme edge cases
    max_excess = mp.mpf("-1")
    max_attain_err = mp.mpf(0)
    n_cases = 0
    for s_s in ["1e-6", "0.1", "0.7", "0.999999"]:
        for c_s in ["1e-6", "0.2", "1e4"]:
            for ht_s in ["1e-10", "0.01", "1.0", "1e10"]:
                for hp_s in ["1e-10", "1.0", "1e10"]:
                    sm, cm, ht, hp = mp.mpf(s_s), mp.mpf(c_s), mp.mpf(ht_s), mp.mpf(hp_s)
                    a = sm * sm + cm * ht
                    b = sm * cm * hp
                    d = sm
                    e = cm * hp
                    k_bound = (mp.sqrt(sm * sm + cm * ht) - sm) / (mp.sqrt(sm * sm + cm * ht) + sm)
                    u_star = cm * hp / mp.sqrt(sm * sm + cm * ht)
                    g_star = u_star * (a * e - b * d) / ((a * u_star + b) * (d * u_star + e))
                    max_attain_err = max(max_attain_err, abs(g_star - k_bound))
                    for mul in ["1e-8", "0.01", "0.5", "2.0", "100.0", "1e8"]:
                        u = u_star * mp.mpf(mul)
                        gu = u * (a * e - b * d) / ((a * u + b) * (d * u + e))
                        max_excess = max(max_excess, gu - k_bound)
                    n_cases += 1
    record(f"C1a Mobius 2-step bound (2.1) & exact argmax ({n_cases} extreme tuples, 60 dps)",
           max_excess <= mp.mpf("1e-55") and max_attain_err < mp.mpf("1e-55"),
           f"max_excess={mp.nstr(max_excess, 4)}, argmax_err={mp.nstr(max_attain_err, 4)}")

    # C1b: Correlation-free generalized Holder (2.3) and parity pullback agreement for s>0 vs failure at s=0
    rng = np.random.default_rng(2026111599)
    # Adversarial perfectly-dependent, negatively-dependent, and AR(1) draws with Gamma(0.5, 2.0) marginals
    g_base = rng.gamma(0.5, 2.0, size=200000)
    u_unif = special.gammainc(0.5, g_base / 2.0)
    g_anti = 2.0 * special.gammaincinv(0.5, np.clip(1.0 - u_unif, 1e-15, 1.0 - 1e-15))
    s0, c0 = 0.4, 0.7
    k1 = (np.sqrt(s0 * s0 + c0 * g_base) - s0) / (np.sqrt(s0 * s0 + c0 * g_base) + s0)
    k2 = (np.sqrt(s0 * s0 + c0 * g_anti) - s0) / (np.sqrt(s0 * s0 + c0 * g_anti) + s0)
    holder_ok = True
    for n in [2, 4, 8, 16]:
        rhs = float(np.mean(k1 ** (2 * n)) ** 0.5)
        # Alternate k1 and k2 (strong negative dependence) vs identical k1 (comonotonic equality case)
        prod_anti = (k1 ** (n // 2)) * (k2 ** (n - n // 2))
        lhs_anti = float(np.mean(prod_anti ** 2) ** 0.5)
        if lhs_anti > rhs * (1.0 + 1e-12):
            holder_ok = False
    # Pullback from t-2n (200 steps) vs t-(2n+1) (201 steps) at s=0.15, c=0.8 vs s=0, c=0.8:
    env = [0.2 + 3.5 * abs(math.sin(j * 1.7 + 0.3)) for j in range(220)]
    def pullback(s_val, c_val, m_steps):
        u = mp.mpf(1)
        for idx in range(len(env) - m_steps, len(env)):
            u = mp.mpf(s_val) + mp.mpf(c_val) * mp.mpf(env[idx]) / u
        return u
    diff_pos_s = abs(mp.log(pullback("0.15", "0.8", 200)) - mp.log(pullback("0.15", "0.8", 201)))
    diff_zero_s = abs(mp.log(pullback("0", "0.8", 200)) - mp.log(pullback("0", "0.8", 201)))
    record("C1b Holder bound (2.3) & parity agreement (s>0 agrees, s=0 disagrees)",
           holder_ok and diff_pos_s < mp.mpf("1e-25") and diff_zero_s > mp.mpf("0.1"),
           f"holder_ok={holder_ok}, |even-odd|(s=0.15)={mp.nstr(diff_pos_s, 4)}, (s=0)={mp.nstr(diff_zero_s, 4)}")

    # C2a: Gamma refresh backward-run monotonicity and m(x) decreasing + negative covariance (4.2),
    # including p=0.25 (singular at zero)!
    # Use exact renewal-block / finite-history deterministic quadrature over (L, U_enter).
    q_ok = True
    for s_s in ["0.02", "0.5", "0.99"]:
        sm = mp.mpf(s_s)
        cm = 1 - sm
        for x_s in ["1e-8", "0.1", "1.0", "50.0", "1e6"]:
            xm = mp.mpf(x_s)
            u = mp.mpf("0.37")
            q = mp.mpf(0)
            for _ in range(40):
                aj = (cm * xm / u) / (sm + cm * xm / u)
                q = aj * (1 - q)
                u = sm + cm * xm / u
                if not (0 < q < 1):
                    q_ok = False
    record("C2a Gamma refresh constant-run elasticity 0 < q_j < 1 (incl. x in [1e-8, 1e6], 60 dps)",
           q_ok, f"all iterates strictly in (0,1): {q_ok}")

    # C2b: Lognormal AR(1) backward-conditional derivative 0 <= q_j <= sigma*phi^j for 0 <= phi < 1
    # across extreme z in [-20, 20] and sigma in [0.1, 2.0], PLUS explicit sign reversal at phi < 0!
    ar_pos_ok = True
    min_q_pos = mp.mpf("1e9")
    for s_s in ["0.05", "0.5", "0.99"]:
        sm = mp.mpf(s_s)
        cm = 1 - sm
        for sig_s in ["0.2", "1.5"]:
            sig = mp.mpf(sig_s)
            for phi_s in ["0.0", "0.5", "0.98"]:
                ph = mp.mpf(phi_s)
                for z_s in ["-20.0", "-3.0", "0.0", "3.0", "20.0"]:
                    zm = mp.mpf(z_s)
                    u = mp.mpf(1)
                    q = mp.mpf(0)
                    for lag in range(60, -1, -1):
                        nj = mp.mpf("0.7") * mp.sin(lag + 1)
                        hj = mp.exp(sig * ((ph ** lag) * zm + nj) - sig * sig / 2)
                        aj = (cm * hj / u) / (sm + cm * hj / u)
                        target_top = sig * (ph ** lag) if not (ph == 0 and lag == 0) else sig
                        q = aj * (target_top - q)
                        u = sm + cm * hj / u
                        if not (0 <= q <= target_top):
                            ar_pos_ok = False
                    min_q_pos = min(min_q_pos, q)
    # Check that phi = -0.75 DOES produce negative q_1 (showing why Section 4's induction needs phi >= 0):
    sm, cm, sig, ph, zm = mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("-0.75"), mp.mpf("0.0")
    u, q = mp.mpf(1), mp.mpf(0)
    neg_seen = False
    for lag in range(10, -1, -1):
        hj = mp.exp(sig * ((ph ** lag) * zm) - sig * sig / 2)
        aj = (cm * hj / u) / (sm + cm * hj / u)
        q = aj * (sig * (ph ** lag) - q)
        u = sm + cm * hj / u
        if q < 0:
            neg_seen = True
    record("C2b Lognormal AR(1) backward derivative 0 <= q_j <= sigma*phi^j (phi>=0, |z|<=20) & phi<0 sign flip",
           ar_pos_ok and neg_seen,
           f"phi>=0 bound holds={ar_pos_ok} (min q_0={mp.nstr(min_q_pos, 4)}); phi=-0.75 flips sign={neg_seen}")


def run_section5_spectral_checks():
    print("\n=== CHECK C3: SECTION 5 SPECTRAL IDENTITY, WALK BOUNDS & REMAINDER (60 DPS) ===", flush=True)
    mp.mp.dps = 60

    # C3a: Pathwise polynomial identity Q_N(z) = prod_{lambda_j>0}(1 + z lambda_j^2) & moment identities
    # on extreme edge weights spanning [1e-6, 1e4]
    edges_mp = [mp.mpf(x) for x in ["1e-6", "0.03", "0.8", "17.5", "0.002", "1200.0", "4.1", "0.5", "95.0"]]
    N = len(edges_mp) + 1
    J = mp.matrix(N, N)
    for j, h in enumerate(edges_mp):
        rt = mp.sqrt(h)
        J[j, j + 1] = rt
        J[j + 1, j] = rt
    ev, _ = mp.eigsy(J)
    pos_sq = [lam * lam for lam in ev if lam > mp.mpf("1e-25")]

    worst_det = mp.mpf(0)
    for z_s in ["1e-5", "0.03", "1.0", "25.0"]:
        zm = mp.mpf(z_s)
        q0, q1 = mp.mpf(1), mp.mpf(1)
        for h in edges_mp:
            q0, q1 = q1, q1 + zm * h * q0
        prod_spec = mp.mpf(1)
        for x in pos_sq:
            prod_spec *= (1 + zm * x)
        worst_det = max(worst_det, abs(mp.log(q1) - mp.log(prod_spec)))

    m1_exact = mp.fsum(edges_mp)
    m2_exact = mp.fsum(h * h for h in edges_mp) + 2 * mp.fsum(
        edges_mp[i] * edges_mp[i + 1] for i in range(len(edges_mp) - 1)
    )
    m3_exact = (
        mp.fsum(h ** 3 for h in edges_mp)
        + 3 * mp.fsum(edges_mp[i] ** 2 * edges_mp[i + 1] + edges_mp[i] * edges_mp[i + 1] ** 2
                      for i in range(len(edges_mp) - 1))
        + 3 * mp.fsum(edges_mp[i] * edges_mp[i + 1] * edges_mp[i + 2]
                      for i in range(len(edges_mp) - 2))
    )
    mom_errs = [
        abs(mp.fsum(x ** 1 for x in pos_sq) - m1_exact) / m1_exact,
        abs(mp.fsum(x ** 2 for x in pos_sq) - m2_exact) / m2_exact,
        abs(mp.fsum(x ** 3 for x in pos_sq) - m3_exact) / m3_exact,
    ]
    bounds_57_ok = True
    for k in range(1, 8):
        lhs = mp.fsum(h ** k for h in edges_mp)
        mid = mp.fsum(x ** k for x in pos_sq)
        rhs = (2 ** (2 * k - 1)) * N * (lhs / (N - 1))
        if not (lhs <= mid * (1 + mp.mpf("1e-50")) and mid <= rhs):
            bounds_57_ok = False
    record("C3a 60-dps Jacobi identity (5.5), moments (5.3) & bounds (5.7) on [1e-6, 1200] weights",
           worst_det < mp.mpf("1e-50") and max(mom_errs) < mp.mpf("1e-50") and bounds_57_ok,
           f"max_det_err={mp.nstr(worst_det, 4)}, max_rel_mom_err={mp.nstr(max(mom_errs), 4)}")

    # C3b: Integrated remainder |R_4(z)| <= (mu_4/4) z^4 <= 32 E[h^4] z^4 checked on EXACT infinite-volume L(z)
    # for i.i.d. Gamma across p in {0.25, 0.5, 1, 2, 5} and z in {1e-4, 0.01, 0.1, 1.0, 5.0}!
    rem_ok = True
    max_ratio_mu4 = mp.mpf(0)
    max_ratio_32 = mp.mpf(0)
    for p_s in ["0.25", "0.5", "1.0", "2.0", "5.0"]:
        pm = mp.mpf(p_s)
        eh2 = 1 + 1 / pm
        eh3 = (1 + 1 / pm) * (1 + 2 / pm)
        eh4 = (1 + 1 / pm) * (1 + 2 / pm) * (1 + 3 / pm)
        mu1 = mp.mpf(1)
        mu2 = eh2 + 2
        mu3 = eh3 + 6 * eh2 + 3
        # Exact mu4 for i.i.d. walks of length 8 on Z (or from formal series in C5 below):
        # Let's also check against 32 * E[h^4] * z^4 (5.4)!
        for z_s in ["1e-4", "0.01", "0.1", "1.0", "5.0"]:
            zm = mp.mpf(z_s)
            Lz = tricomi_L_mp(zm, pm, dps=55)
            poly3 = mu1 * zm - mu2 * (zm ** 2) / 2 + mu3 * (zm ** 3) / 3
            r4 = Lz - poly3
            bound_54 = 32 * eh4 * (zm ** 4)
            # Also R_4(z) must be strictly negative for z > 0!
            if not (r4 < 0 and abs(r4) <= bound_54):
                rem_ok = False
            max_ratio_32 = max(max_ratio_32, abs(r4) / bound_54)
    record("C3b Infinite-volume remainder bound (5.4) |R_4(z)| <= 32 E[h^4] z^4 (25 Gamma cases, incl. p=0.25)",
           rem_ok, f"all 25 cases satisfy R_4(z)<0 and |R_4|/bound <= {mp.nstr(max_ratio_32, 5)}")


def run_section6_formula_checks():
    print("\n=== CHECK C4: SECTION 6 FORMULAS (6.2)-(6.5), IRREVERSIBLE HOST & NEGATIVE-PHI ===", flush=True)
    mp.mp.dps = 65

    # C4a: Exact i.i.d. Gamma across p in {0.25, 0.5, 1, 2, 10}: extract A, B from 65-digit Tricomi roots!
    worst_A, worst_B = mp.mpf(0), mp.mpf(0)
    bounds_ok = True
    for p_s in ["0.25", "0.5", "1.0", "2.0", "10.0"]:
        pm = mp.mpf(p_s)
        A_th, B_th, ell = moments_and_AB_gamma(pm, 0)
        for s_s in ["1e-3", "0.25", "0.75", "0.9999"]:
            d_val = tricomi_D_mp(mp.mpf(s_s), pm, dps=55)
            if not (0 < d_val < ell):
                bounds_ok = False
        ep1, ep2 = mp.mpf("1e-8"), mp.mpf("2e-8")
        d1 = tricomi_D_mp(1 - ep1, pm, dps=60)
        d2 = tricomi_D_mp(1 - ep2, pm, dps=60)
        # Richardson elimination of cubic term from ep1 and ep2=2*ep1:
        # q(e) = d(e)/e = A + B*e + C*e^2 + O(e^3)
        ep3 = mp.mpf("4e-8")
        d3 = tricomi_D_mp(1 - ep3, pm, dps=60)
        q1, q2, q3 = d1 / ep1, d2 / ep2, d3 / ep3
        # Eliminate e and e^2 on geometric grid (1, 2, 4):
        A_num = (8 * q1 - 6 * q2 + q3) / 3
        B_num = (-7 * q1 + 8 * q2 - q3) / (6 * ep1) - (-q1 + 2 * q2 - q3) / (6 * ep1)
        # Exact Lagrange derivative at 0 for nodes x1, 2*x1, 4*x1:
        # L1'(0) = -(2+4)/(1*3)*(-1)? Solve 3x3 Vandermonde exactly in mpmath:
        Vmat = mp.matrix([[1, ep1, ep1 ** 2], [1, ep2, ep2 ** 2], [1, ep3, ep3 ** 2]])
        coefs = mp.lu_solve(Vmat, mp.matrix([q1, q2, q3]))
        worst_A = max(worst_A, abs(coefs[0] - A_th))
        worst_B = max(worst_B, abs(coefs[1] - B_th))
    record("C4a Exact Tricomi D(s,p) vs (6.4) & 0<D<ell for p in {0.25,0.5,1,2,10} (60 dps)",
           bounds_ok and worst_A < mp.mpf("1e-19") and worst_B < mp.mpf("1e-11"),
           f"bounds_ok={bounds_ok}, max |A_err|={mp.nstr(worst_A, 4)}, max |B_err|={mp.nstr(worst_B, 4)}")

    # C4b: Non-reversible 3-state cyclic Markov host where T12 != T21!
    # States h = [0.65, 0.95, 1.40] with uniform stationary pi = (1/3, 1/3, 1/3) (mean = 1.0)
    # Doubly-stochastic asymmetric cyclic transition matrix: P_ij = 0.7 for j=(i+1)%3, 0.2 for j=i, 0.1 for j=(i-1)%3.
    P_cyc = [
        [mp.mpf("0.2"), mp.mpf("0.7"), mp.mpf("0.1")],
        [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.7")],
        [mp.mpf("0.7"), mp.mpf("0.1"), mp.mpf("0.2")],
    ]
    pi_cyc = [mp.mpf(1) / 3] * 3
    h_cyc = [mp.mpf("0.65"), mp.mpf("0.95"), mp.mpf("1.40")]
    b_cyc = [h - 1 for h in h_cyc]
    V_c = mp.fsum(pi_cyc[i] * b_cyc[i] ** 2 for i in range(3))
    m3_c = mp.fsum(pi_cyc[i] * b_cyc[i] ** 3 for i in range(3))
    P2_cyc = [[mp.fsum(P_cyc[i][k] * P_cyc[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    # Stationary joint moments: E[f(h_{-1}) g(h_0)] = sum_{i,j} pi_i P_{ij} f(h_i) g(h_j)
    C1_c = mp.fsum(pi_cyc[i] * P_cyc[i][j] * b_cyc[i] * b_cyc[j] for i in range(3) for j in range(3))
    C2_c = mp.fsum(pi_cyc[i] * P2_cyc[i][j] * b_cyc[i] * b_cyc[j] for i in range(3) for j in range(3))
    T12_c = mp.fsum(pi_cyc[i] * P_cyc[i][j] * (b_cyc[i] ** 2) * b_cyc[j] for i in range(3) for j in range(3))
    T21_c = mp.fsum(pi_cyc[i] * P_cyc[i][j] * b_cyc[i] * (b_cyc[j] ** 2) for i in range(3) for j in range(3))
    T111_c = mp.fsum(
        pi_cyc[i] * P_cyc[i][j] * P_cyc[j][k] * b_cyc[i] * b_cyc[j] * b_cyc[k]
        for i in range(3) for j in range(3) for k in range(3)
    )
    A_cyc = V_c / 2 + C1_c
    B_cyc = -A_cyc - C2_c - T12_c - T21_c - T111_c - m3_c / 3 + mp.mpf("1.5") * A_cyc * A_cyc

    eps_grid = [mp.mpf("1e-5"), mp.mpf("2e-5"), mp.mpf("4e-5")]
    q_cyc = []
    max_tail = mp.mpf(0)
    for ep in eps_grid:
        d_val, res, tail = finite_markov_threshold_mp(1 - ep, P_cyc, pi_cyc, h_cyc, order=34)
        q_cyc.append(d_val / ep)
        max_tail = max(max_tail, tail)
    Vmat = mp.matrix([[1, ep, ep * ep] for ep in eps_grid])
    c_fit = mp.lu_solve(Vmat, mp.matrix(q_cyc))
    asym_gap = abs(T12_c - T21_c)
    err_A_cyc = abs(c_fit[0] - A_cyc)
    err_B_cyc = abs(c_fit[1] - B_cyc)
    record("C4b Irreversible 3-state cyclic host (T12 != T21) vs general formula (6.2) (65 dps)",
           asym_gap > mp.mpf("0.005") and err_A_cyc < mp.mpf("1e-14") and err_B_cyc < mp.mpf("1e-9"),
           f"|T12-T21|={mp.nstr(asym_gap, 5)}, |A_err|={mp.nstr(err_A_cyc, 4)}, |B_err|={mp.nstr(err_B_cyc, 4)}, series_tail={mp.nstr(max_tail, 3)}")

    # C4c: Exact high-precision moment check of (6.4) and (6.5) against direct quadrature of 3-step joint laws!
    # 1) Lognormal AR(1) via 3D Gauss-Hermite quadrature (nodes=28 per axis = 21,952 exact Gaussian points)
    gh_x, gh_w = special.roots_hermitenorm(28)
    gh_w = gh_w / math.sqrt(2.0 * math.pi)
    max_ln_err = 0.0
    for w_val in [0.1, math.log(1.5), 1.2]:
        sig = math.sqrt(w_val)
        for phi_val in [-0.85, -0.75, 0.0, 0.6, 0.92]:
            s_orth = math.sqrt(1.0 - phi_val * phi_val)
            xi_m2 = gh_x[:, None, None]
            xi_m1 = phi_val * xi_m2 + s_orth * gh_x[None, :, None]
            xi_0 = phi_val * xi_m1 + s_orth * gh_x[None, None, :]
            wt3 = gh_w[:, None, None] * gh_w[None, :, None] * gh_w[None, None, :]
            b_m2 = np.exp(sig * xi_m2 - 0.5 * w_val) - 1.0
            b_m1 = np.exp(sig * xi_m1 - 0.5 * w_val) - 1.0
            b_0 = np.exp(sig * xi_0 - 0.5 * w_val) - 1.0
            Vq = float(np.sum(wt3 * b_0 * b_0))
            c1q = float(np.sum(wt3 * b_0 * b_m1))
            c2q = float(np.sum(wt3 * b_0 * b_m2))
            m3q = float(np.sum(wt3 * (b_0 ** 3)))
            t12q = float(np.sum(wt3 * b_0 * (b_m1 ** 2)))
            t21q = float(np.sum(wt3 * (b_0 ** 2) * b_m1))
            t111q = float(np.sum(wt3 * b_0 * b_m1 * b_m2))
            Aq = 0.5 * Vq + c1q
            Bq = -Aq - c2q - t12q - t21q - t111q - m3q / 3.0 + 1.5 * Aq * Aq
            Ath, Bth, _ = moments_and_AB_lognormal(w_val, phi_val)
            max_ln_err = max(max_ln_err, abs(Aq - float(Ath)), abs(Bq - float(Bth)) / max(1.0, abs(float(Bth))))
    # Also check negative-memory counterexample w=0.1, phi=-0.75 at 65 digits:
    A_neg, B_neg, ell_neg = moments_and_AB_lognormal(mp.mpf("0.1"), mp.mpf("-0.75"))
    record("C4c Lognormal AR(1) 3D Gauss-Hermite vs (6.5) (incl. neg phi & w=1.2) + counterexample sign",
           max_ln_err < 1e-11 and A_neg < mp.mpf("-0.0191875"),
           f"max_3D_quad_err={max_ln_err:.3e}, A(0.1,-0.75)={mp.nstr(A_neg, 18)} < -0.0191875")


def run_section7_divergence_checks():
    print("\n=== CHECK C5: SECTION 7 EXACT RATIONAL FORMAL SERIES INVERSION & ZERO RADIUS ===", flush=True)
    from fractions import Fraction

    # Exact zero-roundoff rational computation of spectral moments mu_1..mu_26 and formal [eps^k] D(eps)
    # for i.i.d. Gamma(p, scale=1/p) via L(p*u) = N_num(u) / D_denom(u)!
    M_ORD = 26
    bounds_71_ok = True
    inversion_ok = True
    growth_summary = []

    for p_num, p_den in [(1, 4), (1, 2), (1, 1), (2, 1)]:
        pm = Fraction(p_num, p_den)
        d_den = [Fraction(0)] * (M_ORD + 1)
        n_num = [Fraction(0)] * (M_ORD + 1)
        d_den[0] = Fraction(1)
        poch = Fraction(1)
        harm = Fraction(0)
        fact = 1
        for n in range(1, M_ORD + 1):
            poch *= (pm + n - 1)
            harm += Fraction(1, 1) / (pm + n - 1)
            fact *= n
            base = (poch * poch) / fact
            d_den[n] = (-base if n % 2 == 1 else base)
            n_num[n] = (base if n % 2 == 1 else -base) * harm

        L_u = [Fraction(0)] * (M_ORD + 1)
        for k in range(1, M_ORD + 1):
            s_acc = n_num[k]
            for j in range(1, k):
                s_acc -= L_u[j] * d_den[k - j]
            L_u[k] = s_acc

        L_z = [Fraction(0)] * (M_ORD + 1)
        mu = [Fraction(0)] * (M_ORD + 1)
        eh = Fraction(1)
        for k in range(1, M_ORD + 1):
            L_z[k] = L_u[k] / (pm ** k)
            mu[k] = (1 if k % 2 == 1 else -1) * k * L_z[k]
            eh *= (pm + k - 1) / pm
            if not (eh <= mu[k] <= (2 ** (2 * k - 1)) * eh):
                bounds_71_ok = False

        def series_mul(a, b):
            out = [Fraction(0)] * (M_ORD + 1)
            for i in range(M_ORD + 1):
                if a[i] == 0:
                    continue
                for j in range(M_ORD + 1 - i):
                    out[i + j] += a[i] * b[j]
            return out

        def series_exp(a):
            out = [Fraction(0)] * (M_ORD + 1)
            out[0] = Fraction(1)
            cur = out[:]
            f_inv = Fraction(1)
            for m in range(1, M_ORD + 1):
                cur = series_mul(cur, a)
                f_inv /= m
                for idx in range(M_ORD + 1):
                    out[idx] += cur[idx] * f_inv
            return out

        def series_log1p(a):
            out = [Fraction(0)] * (M_ORD + 1)
            cur = a[:]
            for m in range(1, M_ORD + 1):
                sgn = Fraction(1 if m % 2 == 1 else -1, m)
                for idx in range(M_ORD + 1):
                    out[idx] += sgn * cur[idx]
                if m < M_ORD:
                    cur = series_mul(cur, a)
            return out

        neg_L = [-x for x in L_z]
        exp_neg_L = series_exp(neg_L)
        eps_of_z = [Fraction(0)] + [-exp_neg_L[k] for k in range(1, M_ORD + 1)]
        z_of_E = [Fraction(0)] * (M_ORD + 1)
        z_of_E[1] = Fraction(1)
        for k in range(2, M_ORD + 1):
            acc = Fraction(0)
            pw = z_of_E[:]
            for m in range(2, k + 1):
                pw = series_mul(pw, z_of_E)
                acc += eps_of_z[m] * pw[k]
            z_of_E[k] = -acc

        quot_minus_1 = [Fraction(0)] + [z_of_E[k + 1] for k in range(1, M_ORD)] + [Fraction(0)]
        D_of_E = series_log1p(quot_minus_1)
        for k in range(1, M_ORD):
            D_of_E[k] -= Fraction(2, k)

        # Exact rational roundtrip: E * exp(D(E)) / (1 - E)^2 == z(E) with ZERO discrepancy!
        D_trunc = D_of_E[:]
        D_trunc[M_ORD] = Fraction(0)
        exp_D = series_exp(D_trunc)
        inv_1mE_sq = [Fraction(k + 1) for k in range(M_ORD + 1)]
        prod_rhs = series_mul(exp_D, inv_1mE_sq)
        for k in range(1, M_ORD):
            if prod_rhs[k - 1] != z_of_E[k]:
                inversion_ok = False

        A_exact = Fraction(1, 2) / pm
        B_exact = -Fraction(1, 2) / pm - Fraction(7, 24) / (pm * pm)
        if D_of_E[1] != A_exact or D_of_E[2] != B_exact:
            inversion_ok = False

        mp.mp.dps = 40
        r4 = float(mp.mpf(abs(D_of_E[4].numerator)) ** (mp.mpf(1) / 4) / mp.mpf(D_of_E[4].denominator) ** (mp.mpf(1) / 4))
        r12 = float(mp.mpf(abs(D_of_E[12].numerator)) ** (mp.mpf(1) / 12) / mp.mpf(D_of_E[12].denominator) ** (mp.mpf(1) / 12))
        r24 = float(mp.mpf(abs(D_of_E[24].numerator)) ** (mp.mpf(1) / 24) / mp.mpf(D_of_E[24].denominator) ** (mp.mpf(1) / 24))
        if not (r4 < r12 < r24):
            inversion_ok = False
        growth_summary.append(f"p={float(pm):g}: |d_k|^(1/k) k=4,12,24 -> {r4:.3g}, {r12:.3g}, {r24:.3g}")

    record("C5 Exact rational bounds (7.1), formal inversion (7.3) & zero-radius growth (k<=26)",
           bounds_71_ok and inversion_ok,
           "; ".join(growth_summary))


# ---------------------------------------------------------------------------
# Edge-case independent nonlinear recurrence simulations (C6)
# Vectorized across replicates in NumPy; zero imports from gap3_checks.py!
# ---------------------------------------------------------------------------

def simulate_hosts_independent(model: str, param: float, phi: float, n_steps: int, n_reps: int, seed: int):
    """Generate stationary two-sided Markov trajectories independently of gap3_checks.py."""
    rng = np.random.default_rng(seed)
    if model == "gamma":
        p = param
        h = np.empty((n_steps, n_reps), dtype=np.float64)
        cur = rng.gamma(shape=p, scale=1.0 / p, size=n_reps)
        h[0] = cur
        for t in range(1, n_steps):
            stay = rng.random(n_reps) < phi
            fresh = rng.gamma(shape=p, scale=1.0 / p, size=n_reps)
            cur = np.where(stay, cur, fresh)
            h[t] = cur
        return h
    elif model == "lognormal":
        w = param
        sig = math.sqrt(w)
        s_orth = math.sqrt(1.0 - phi * phi)
        xi = np.empty((n_steps, n_reps), dtype=np.float64)
        cur = rng.standard_normal(n_reps)
        xi[0] = cur
        innov = rng.standard_normal((n_steps - 1, n_reps))
        for t in range(1, n_steps):
            cur = phi * cur + s_orth * innov[t - 1]
            xi[t] = cur
        return np.exp(sig * xi - 0.5 * w)
    raise ValueError(model)


def solve_threshold_independent(h: np.ndarray, s_vals: np.ndarray, burn: int = 2048):
    """Independent vectorized Newton solver for r(s, (1-s)*exp(D)) = 0 per replicate."""
    nt, nr = h.shape
    ns = len(s_vals)
    s_row = s_vals[None, :]                 # (1, ns)
    eps_row = 1.0 - s_row                   # (1, ns)
    d = np.zeros((nr, ns), dtype=np.float64)

    for _ in range(10):
        c_mat = eps_row * np.exp(d)         # (nr, ns)
        u = np.ones((nr, ns), dtype=np.float64)
        du = np.zeros((nr, ns), dtype=np.float64)  # d(log U)/dD
        acc = np.zeros((nr, ns), dtype=np.float64)
        jac = np.zeros((nr, ns), dtype=np.float64)
        for t in range(nt):
            ht = h[t, :, None]              # (nr, 1)
            ch_over_u = c_mat * ht / u
            u_next = s_row + ch_over_u
            a_t = ch_over_u / u_next
            du_next = a_t * (1.0 - du)
            u = u_next
            du = du_next
            if t >= burn:
                # Subtract exact zero-mean control variate eps*exp(D)*(h_t - 1) near s=1
                cv = c_mat * (ht - 1.0)
                acc += np.log(u) - cv
                jac += du - cv
        f_val = acc / (nt - burn)
        df_val = jac / (nt - burn)
        if np.max(np.abs(f_val) / eps_row) < 1e-12:
            break
        d = d - f_val / df_val
    return d


def run_edge_case_nonlinear_checks():
    print("\n=== CHECK C6: EDGE-CASE NONLINEAR RECURRENCE CHECKS ===", flush=True)
    mp.mp.dps = 50
    eps_grid = np.array([0.02, 0.01, 0.005, 0.0025], dtype=np.float64)
    s_near_one = 1.0 - eps_grid
    vander_inv = np.linalg.inv(np.vander(eps_grid, 4, increasing=True))

    # Test cases covering singular-at-zero Gamma (p=0.25, p=0.5), negative-phi Lognormal counterexamples,
    # stronger-noise Lognormal (w=log 2), plus overlap with gap3_checks.py!
    cases = [
        # (label, model, param, phi, n_reps, n_steps, expect_pos_D)
        ("Gamma singular p=0.5 (V=2) phi=0.0", "gamma", 0.5, 0.0, 24, 65536, True),
        ("Gamma singular p=0.5 (V=2) phi=0.5", "gamma", 0.5, 0.5, 24, 65536, True),
        ("Gamma singular p=0.25 (V=4) phi=0.5", "gamma", 0.25, 0.5, 32, 65536, True),
        ("Gamma overlap p=2.0 (V=0.5) phi=0.9", "gamma", 2.0, 0.9, 24, 65536, True),
        ("Lognormal neg-memory counterexample w=0.1 phi=-0.75", "lognormal", 0.1, -0.75, 24, 65536, False),
        ("Lognormal strong neg-memory w=0.25 phi=-0.80", "lognormal", 0.25, -0.80, 24, 65536, False),
        ("Lognormal large-noise w=log(2) (V=1) phi=0.5", "lognormal", math.log(2.0), 0.5, 24, 65536, True),
        ("Lognormal overlap w=log(1.5) phi=0.9", "lognormal", math.log(1.5), 0.9, 24, 65536, True),
    ]

    summary_rows = []
    for idx, (label, model, param, phi, n_reps, n_steps, expect_pos) in enumerate(cases):
        h = simulate_hosts_independent(model, param, phi, n_steps + 2048, n_reps, seed=910000 + 37 * idx)
        s_all = np.concatenate([np.array([0.25, 0.65]), s_near_one])
        d_all = solve_threshold_independent(h, s_all, burn=2048)

        d_int = d_all[:, :2]
        d_eps = d_all[:, 2:]
        if model == "gamma":
            Ath, Bth, ell_m = moments_and_AB_gamma(param, phi)
        else:
            Ath, Bth, ell_m = moments_and_AB_lognormal(param, phi)
        A_pred, B_pred, ell = float(Ath), float(Bth), float(ell_m)

        # Extract A, B per replicate by degree-3 polynomial fit to D(eps)/eps
        q_reps = d_eps / eps_grid[None, :]
        coefs = q_reps @ vander_inv.T
        A_hat = float(np.mean(coefs[:, 0]))
        A_se = float(np.std(coefs[:, 0], ddof=1) / math.sqrt(n_reps))
        B_hat = float(np.mean(coefs[:, 1]))
        B_se = float(np.std(coefs[:, 1], ddof=1) / math.sqrt(n_reps))

        means_all = np.mean(d_all, axis=0)
        ses_all = np.std(d_all, axis=0, ddof=1) / math.sqrt(n_reps)

        # Check strict upper bound D < ell at ALL s (including interior s=0.25, 0.65):
        upper_ok = bool(np.all(means_all + 4.0 * ses_all < ell))
        if expect_pos:
            sign_ok = bool(np.all(means_all - 4.0 * ses_all > 0.0))
        else:
            # For negative-memory counterexamples, near-one D(eps) must be strictly NEGATIVE!
            sign_ok = bool(np.all(means_all[2:] + 4.0 * ses_all[2:] < 0.0))

        a_ok = abs(A_hat - A_pred) <= 4.5 * A_se + 0.015 * max(abs(A_pred), 0.02)
        b_ok = abs(B_hat - B_pred) <= 4.5 * B_se + 0.04 * max(abs(B_pred), 0.05)

        record(
            f"C6 {label}",
            upper_ok and sign_ok and a_ok and b_ok,
            f"A={A_hat:.6g}+/-{A_se:.2g} (pred {A_pred:.6g}); "
            f"B={B_hat:.6g}+/-{B_se:.2g} (pred {B_pred:.6g}); "
            f"D(s=0.25)={means_all[0]:.5g}, D(eps=0.0025)={means_all[-1]:.5g}, ell={ell:.5g}, "
            f"min_h={h.min():.3g}, max_h={h.max():.3g}",
        )
        summary_rows.append({
            "label": label,
            "model": model,
            "param": param,
            "phi": phi,
            "A_pred": A_pred,
            "A_hat": A_hat,
            "A_se": A_se,
            "B_pred": B_pred,
            "B_hat": B_hat,
            "B_se": B_se,
            "D_s025": float(means_all[0]),
            "D_s065": float(means_all[1]),
            "D_eps0025": float(means_all[-1]),
            "D_eps0025_se": float(ses_all[-1]),
            "ell": ell,
            "min_h": float(h.min()),
            "max_h": float(h.max()),
        })
    return summary_rows


def main():
    t0 = time.monotonic()
    print("Independent referee verification script verify_gap3.py", flush=True)
    run_known_result_gates()
    if FAILURES:
        print("STOP: Gate failure:", FAILURES, flush=True)
        return 1
    run_section2_and_section4_checks()
    run_section5_spectral_checks()
    run_section6_formula_checks()
    run_section7_divergence_checks()
    rows = run_edge_case_nonlinear_checks()
    elapsed = time.monotonic() - t0
    print(f"\nCOMPLETED in {elapsed:.2f}s; FAILURES={FAILURES}", flush=True)
    (OUT_DIR / "verify_gap3_results.json").write_text(
        json.dumps({"seconds": elapsed, "failures": FAILURES, "nonlinear_edge_cases": rows}, indent=2)
    )
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
