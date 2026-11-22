"""Formal stationary fourth-order growth cost H_n(phi) for the correlated ring (Part 3 check).

Method 1 (sigma-free perturbative trajectory estimator):
  By Hermite orthogonality (E[He_4(Z)] = 0 and E[He_3(Z_1) Z_2] = 0 for jointly Gaussian Z),
  the formal stationary O(sigma^4) coefficient H_n(phi) in
      cost(n, sigma) = C_n(phi) * sigma^2 + H_n(phi) * sigma^4 + o(sigma^4)
    [O(sigma^6) conditional on smooth stationary differentiation / remainder control]
  is obtained by evolving the exact sigma-free Taylor recursions (v_1, v_2, v_3) on 1^perp
  driven by stationary Gaussian AR(1) xi(t) and eta(t) = (xi(t)^2 - 1)/2, with antithetic/control
  reduction.

Method 2 (exact deterministic closed/Fourier mode-sum analysis):
  Evaluates the exact deterministic Wick/Fourier mode sums for the formal stationary coefficient
  H_n(phi) and compares finite-n values against an asymptotic continuum-crossover approximation
  as n -> inf and phi -> 1 with theta_n = (1 - phi)*n / (2*sqrt(phi*q*(1 - phi*p))) fixed.

Correction note (14 November 2026; referee_report_h4.md points 43 and 44):
  1. In primary_C2, math.expm1(n * alpha) overflowed for valid small-phi, moderate-n inputs such as
     (n=40, s=0.625, gbar=0.375, phi=1e-10) where n*alpha > 709. Replaced by the overflow-safe
     form r = phi*q/(1 - phi*p) with |r| < 1 and r**n / (1 - r**n) (or exp(-n*alpha)/-expm1(-n*alpha)).
  2. Earlier wording in this docstring and in Section 4's print header overstated the remainder as
     unconditional O(sigma^6) (existence of H_n alone gives o(sigma^4) without higher-order stationary
     regularity, and identifying the formal filter value with the true Taylor coefficient requires
     justifying differentiation of the invariant law) and called the continuum crossover g(theta_n)
     "exact" (it is an asymptotic crossover approximation: e.g. at p=0.625, phi=0.9, relative error
     against finite-n H_n is +9.53% at n=16 and -6.65% at n=32).
"""
from __future__ import annotations

import math
import numpy as np


def primary_C2(n: int, s: float, gbar: float, phi: float) -> float:
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    if phi == 0.0 or n == 1:
        return 0.5 * q * q / n
    r = (phi * q) / (1.0 - phi * p)
    if r > 0.0:
        neg_n_alpha = n * math.log(r)
        corr = 0.0 if neg_n_alpha < -745.0 else q * math.exp(neg_n_alpha) / (-math.expm1(neg_n_alpha))
    else:
        rn = r ** n
        corr = q * rn / (1.0 - rn)
    return (q * q * (1.0 + phi)) / (2.0 * n * (1.0 - phi)) - corr


def primary_H4_phi0(n: int, s: float, gbar: float) -> float:
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    return (
        (q ** 2) / (4.0 * n)
        + (q ** 3) * (n * n - 1) / (24.0 * p * n * n)
        - (q ** 3) / (n * n)
        + 0.75 * (q ** 4) / (n * n)
    )


def exact_analytical_H4(n: int, s: float, gbar: float, phi: float) -> dict[str, float]:
    """Exact deterministic O(n^2) Fourier-mode / Wick-contraction evaluation of H_n(phi)
    for ANY n >= 1, 0 < s < 1, gbar > 0, 0 <= phi < 1!

    Let's trace EVERY diagram in:
      H_n(phi) = E[ -gamma_4(t) + gamma_1(t)*gamma_3(t) + 0.5*gamma_2(t)^2
                    - gamma_1(t)^2 * gamma_2(t) + 0.25*gamma_1(t)^4 ]
    using DFT x_hat_m = sum_{k=0}^{n-1} x_k exp(-2pi i m k / n).
    """
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0

    if n == 1:
        c2_1 = 0.5 * q * q
        h1 = 0.25 * q * q * p * (1.0 - 3.0 * q)
        return {"H_total": h1, "C2": c2_1, "H_over_C": h1 / c2_1,
                "T_A": 0.0, "T_B": 0.0, "T_C": 0.25 * q * q, "T_D": -(q ** 3), "T_E": 0.75 * (q ** 4),
                "E_q_xi_dot_v3": 0.0, "E_q_eta_dot_v2": 0.0, "E_v1_norm2": 0.0,
                "p34a_v1_piece": 0.0, "p31_piece": 0.0}

    m_all = np.arange(n)
    lam = np.exp(-2j * np.pi * m_all / n)  # eigenvalue of P: (Px)_k = x_{k-1}
    mu = p + q * lam                       # eigenvalue of M = pI + qP

    # Active non-zero modes m = 1..n-1:
    idx = np.arange(1, n)

    # Helper for geometric sum sum_{l=1}^inf r^l * z^(l-1) = r / (1 - r*z)
    # Wait: if lag starts at l=1 with factor alpha * beta^(l-1), sum_{l>=1} = alpha / (1 - beta).

    # 1. T_E = (1/4) E[gamma_1(t)^4]: gamma_1 = (q/n) xi_hat_0, E[xi_hat_0^2] = n -> E[xi_hat_0^4] = 3 n^2
    T_E = 0.75 * (q ** 4) / (n ** 2)

    # 2. T_D = -E[gamma_1(t)^2 * gamma_2(t)]:
    #    gamma_2(t) = (q/n) sum_{m!=0} xi_hat_{-m}(t) v1_hat_m(t) + (q/n) eta_hat_0(t).
    #    - With (q/n) eta_hat_0(t): E[xi_hat_0(t)^2 eta_hat_0(t)] = n * 1 * 1 = n -> (q^3 / n^3) * n = q^3 / n^2.
    #    - With (q/n) xi_hat_{-m}(t) v1_hat_m(t):
    #      xi_hat_0(t)^2 xi_hat_{-m}(t) xi_hat_m(t-l): only pairing (0 with -m) vanishes since m != 0!
    #      Pairing (0 with 0) and (-m with m): E[xi_hat_0^2] * E[xi_hat_{-m}(t) xi_hat_m(t-l)]!
    #      WAIT: that gives n * n * phi^l, summing to n * E[xi_t^T v_1(t)]!
    E_xi_dot_v1 = (q / n) * float(np.sum((phi * lam[idx]) / (1.0 - phi * mu[idx])).real)
    # E[gamma_1^2] = q^2 / n, and E[gamma_1^2 * (q xi_t^T v_1)] = (q^2 / n) * q * E_xi_dot_v1 (since m != 0 is independent of mode 0!)
    T_D = -(q ** 3) / (n ** 2) - (q ** 2 / n) * (q * E_xi_dot_v1)

    # 3. T_C = (1/2) E[gamma_2(t)^2], where gamma_2 = A + B, A = q xi_t^T v_1(t), B = (q/n) 1^T eta_t:
    #    - E[B^2] = (q^2 / n^2) E[eta_hat_0^2] = (q^2 / n^2) * (n / 2) = q^2 / (2n).
    #    - E[A * B]: (q^2 / n^2) sum_{m!=0} sum_{l>=1} (q lam_m / n) mu_m^{l-1} E[eta_hat_0(t) xi_hat_{-m}(t) xi_hat_m(t-l)].
    #      Recall E[eta_hat_0(t) xi_hat_{-m}(t) xi_hat_m(t-l)] = n * phi^(0 + l) = n * phi^l!
    #      So E[A * B] = (q^2 / n) * E_xi_dot_v1!
    #    - E[A^2]: Gaussian 4-point Wick contraction of q^2 xi_t^T v_1(t) xi_t^T v_1(t):
    #      = q^2 [ (E[xi_t^T v_1])^2 + E||v_1||^2 + tr(Cov(xi_t, v_1)^2) ]!
    E_v1_norm2 = (q * q / (n * n)) * float(
        np.sum((1.0 - (phi ** 2) * (np.abs(mu[idx]) ** 2)) / ((1.0 - np.abs(mu[idx]) ** 2) * (np.abs(1.0 - phi * mu[idx]) ** 2))).real
    )
    tr_cov2 = (q * q / (n * n)) * float(np.sum(((phi * lam[idx]) / (1.0 - phi * mu[idx])) ** 2).real)
    E_A2 = (q ** 2) * (E_xi_dot_v1 ** 2 + E_v1_norm2 + tr_cov2)
    T_C = 0.5 * (E_A2 + 2.0 * (q * q / n) * E_xi_dot_v1 + (q * q) / (2.0 * n))

    # 4. Compute exact mode-by-mode expectations needed for T_B = E[gamma_1(t) * gamma_3(t)]
    #    and T_A = -E[gamma_4(t)] = -q E[xi_t^T v_3(t)] - q E[eta_t^T v_2(t)]!
    #
    # Recall for m in 1..n-1:
    #   v2_hat_m(t) = sum_{l2>=1} mu_m^{l2-1} [
    #       sum_{a!=0} K_{m,a} xi_hat_{m-a}(t-l2) v1_hat_a(t-l2)
    #     - (q^2 lam_m / n^2) xi_hat_0(t-l2) xi_hat_m(t-l2)
    #     + (q lam_m / n) eta_hat_m(t-l2)
    #   ]
    # where K_{m,a} = (q/n) * (lam_m - delta_{m,a} * mu_m) for m, a in 1..n-1!

    m_mat = idx[:, None]   # shape (n-1, 1)
    a_mat = idx[None, :]   # shape (1, n-1)
    lam_m = lam[idx][:, None]
    lam_a = lam[idx][None, :]
    mu_m = mu[idx][:, None]
    mu_a = mu[idx][None, :]
    K_ma = (q / n) * (lam_m - np.eye(n - 1) * mu_m)  # shape (n-1, n-1)

    # First: -q E[eta_t^T v_2(t)] = -(q/n) sum_{m!=0} E[eta_hat_{-m}(t) v2_hat_m(t)]:
    #   Three pieces of v2_hat_m(t - l2) contracted with eta_hat_{-m}(t):
    #   (i) K_{m,a} xi_{m-a}(t-l2) * (q lam_a / n) sum_{l1>=1} mu_a^{l1-1} xi_a(t-l2-l1):
    #       E[eta_{-m}(t) xi_{m-a}(t-l2) xi_a(t-l2-l1)] = n * phi^(l2 + (l2 + l1)) = n * phi^(2*l2 + l1)!
    #       Sum over l1 >= 1: phi * (q lam_a / n) / (1 - phi * mu_a).
    #       Sum over l2 >= 1: phi^2 / (1 - phi^2 * mu_m).
    g1_a = (q * lam[idx] * phi) / (1.0 - phi * mu[idx])  # n * (1/n) * sum_{l1>=1} ... = q lam_a phi / (1 - phi mu_a)
    sum_l2_phi2 = (phi ** 2) / (1.0 - (phi ** 2) * mu[idx])
    eta_v2_i = float(np.sum(sum_l2_phi2 * (K_ma @ g1_a)).real)  # note (1/n) from v1 * n from E[...] = 1!
    #   (ii) -(q^2 lam_m / n^2) xi_0(t-l2) xi_m(t-l2):
    #       E[eta_{-m}(t) xi_0(t-l2) xi_m(t-l2)] = n * phi^(2*l2)!
    eta_v2_ii = -float(np.sum(sum_l2_phi2 * (q * q * lam[idx] / n)).real)
    #   (iii) +(q lam_m / n) eta_m(t-l2):
    #       E[eta_{-m}(t) eta_m(t-l2)] = (n / 2) * phi^(2*l2)!
    eta_v2_iii = float(np.sum(sum_l2_phi2 * (0.5 * q * lam[idx])).real)
    E_q_eta_dot_v2 = (q / n) * (eta_v2_i + eta_v2_ii + eta_v2_iii)

    # Second: T_B = E[gamma_1(t) * gamma_3(t)], where gamma_1(t) = (q/n) xi_hat_0(t)
    #   and gamma_3(t) = (q/n) sum_{m!=0} xi_hat_{-m}(t) v2_hat_m(t) + (q/n) sum_{m!=0} eta_hat_{-m}(t) v1_hat_m(t).
    #   - Second part of gamma_3 with gamma_1:
    #     E[xi_0(t) eta_{-m}(t) xi_m(t-l1)] = n * phi^(0 + l1) = n * phi^l1!
    #     Contribution to E[gamma_1 * gamma_3]: (q/n) * (q/n) * sum_{m!=0} (q lam_m / n) * n * phi / (1 - phi mu_m)
    #     = (q^2 / n) * E_xi_dot_v1!
    #   - First part of gamma_3 with gamma_1: (q^2 / n^2) sum_{m!=0} E[xi_0(t) xi_{-m}(t) v2_hat_m(t)].
    #     Only 4-xi pieces of v2_hat_m(t) survive (since E[xi_0(t) xi_{-m}(t) eta_m(t-l2)]:
    #     wait -- E[eta_m(t-l2) xi_0(t) xi_{-m}(t)] = n * phi^(l2 + l2) = n * phi^(2*l2) ALSO SURVIVES!!):
    #     Let's check ALL THREE pieces of v2_hat_m(t) contracted with xi_0(t) xi_{-m}(t)!
    #     (a) K_{m,a} xi_{m-a}(t-l2) v1_hat_a(t-l2): 4-point xi_0(t) xi_{-m}(t) xi_{m-a}(t-l2) xi_a(t-l2-l1) (m!=0, a!=0):
    #         - Pair (0 with -m): 0 since m != 0.
    #         - Pair (0 with m-a) => a = m! And (-m with a=m): gives n^2 * phi^l2 * phi^(l2+l1) = n^2 phi^(2*l2 + l1)!
    #         - Pair (0 with a): 0 since a != 0.
    #         So ONLY a = m contributes: K_{m,m} * (q lam_m / n) * n^2 * [phi^2 / (1 - phi^2 mu_m)] * [phi / (1 - phi mu_m)]!
    K_diag = np.diag(K_ma)  # = (q/n) * (lam_m - mu_m)
    tb_a = float(np.sum(n * K_diag * sum_l2_phi2 * g1_a).real)
    #     (b) -(q^2 lam_m / n^2) xi_0(t-l2) xi_m(t-l2) with xi_0(t) xi_{-m}(t) (m != 0):
    #         Only pairing (0 with 0) and (-m with m) survives: n^2 * phi^(2*l2)!
    tb_b = -float(np.sum((q * q * lam[idx]) * sum_l2_phi2).real)
    #     (c) +(q lam_m / n) eta_m(t-l2) with xi_0(t) xi_{-m}(t):
    #         E[eta_m(t-l2) xi_0(t) xi_{-m}(t)] = n * phi^(l2 + l2) = n * phi^(2*l2)!
    tb_c = float(np.sum((q * lam[idx]) * sum_l2_phi2).real)
    T_B = (q * q / n) * E_xi_dot_v1 + (q * q / (n * n)) * (tb_a + tb_b + tb_c)

    # 5. Finally, compute -q E[xi_t^T v_3(t)] = -(q/n) sum_{m!=0} E[xi_hat_{-m}(t) v3_hat_m(t)]!
    #    Look at v_3(s + 1) = M v_3(s) + F3(s), so for m != 0:
    #      E[xi_hat_{-m}(t) v3_hat_m(t)] = sum_{l3>=1} phi^{l3} mu_m^{l3-1} * { ... }?
    #    CAREFUL! xi_hat_{-m}(t) can contract with ANY field inside F3(t - l3), and every step l3
    #    simply multiplies xi_hat_{-m}(t)'s correlation with ALL past fields (at times <= t - l3) by phi^{l3}!!
    #    HOLD ON! VERIFY THAT MARKOV PROPERTY!!
    #    Because t > t - l3 >= all times inside F3(t - l3), for ANY polynomial P_past of fields at times <= tau = t - l3,
    #    since xi(t) = phi^{l3} xi(tau) + independent_innovation, we have IDENTICALLY:
    #      E[xi_hat_{-m}(t) * F3_hat_m(t - l3)] = phi^{l3} * E[xi_hat_{-m}(tau) * F3_hat_m(tau)]!!
    #    WHOA!! BY AR(1) MARKOVIANITY, THE SUM OVER l3 >= 1 FACTORS OUT COMPLETELY AS:
    #      E[xi_hat_{-m}(t) v3_hat_m(t)] = [phi / (1 - phi * mu_m)] * E[xi_hat_{-m}(tau) * F3_hat_m(tau)]!!
    #
    #    LET'S DOUBLE-CHECK THAT!!
    #    xi(t) = phi^{l3} xi(t - l3) + zeta, where zeta is independent of sigma-algebra F_{t - l3}!
    #    Since F3_hat_m(t - l3) is F_{t - l3}-measurable, E[zeta * F3_hat_m(t - l3)] = 0 IDENTICALLY!
    #    Therefore E[xi_hat_{-m}(t) F3_hat_m(t - l3)] = phi^{l3} E[xi_hat_{-m}(tau) F3_hat_m(tau)] (independent of l3 by stationarity)!
    #    Summing l3 = 1..inf of phi^{l3} mu_m^{l3-1} gives Simply R_m := phi / (1 - phi * mu_m)!!

    R_m = phi / (1.0 - phi * mu[idx])  # shape (n-1,)

    # Now write out F3_hat_m(tau) completely from:
    #   v_3(tau+1) = M v_3(tau) + q Q_0 P (xi_tau . v_2(tau)) + q Q_0 P (eta_tau . v_1(tau))
    #                - gamma_1(tau) v_2(tau+1) - gamma_2(tau) v_1(tau+1).
    # Substitute v_2(tau+1) = M v_2(tau) + [q Q_0 P(xi_tau . v_1(tau)) + (q/n) Q_0 P eta_tau - gamma_1(tau) v_1(tau+1)]:
    # Wait -- keep the 4 clean pieces of F3(tau):
    #   Piece 3.1: q Q_0 P(xi_tau . v_2(tau)) - gamma_1(tau) M v_2(tau)
    #              -> in mode m != 0: sum_{b!=0} K_{m,b} xi_hat_{m-b}(tau) v2_hat_b(tau)!
    #   Piece 3.2: q Q_0 P(eta_tau . v_1(tau)) - gamma_1(tau) * (q/n) Q_0 P eta_tau
    #              -> in mode m != 0: (q lam_m / n) sum_{a!=0} eta_hat_{m-a}(tau) v1_hat_a(tau)
    #                                 - (q^2 lam_m / n^2) xi_hat_0(tau) eta_hat_m(tau)!
    #   Piece 3.3: -gamma_1(tau) * [ q Q_0 P(xi_tau . v_1(tau)) - gamma_1(tau) v_1(tau+1) ]
    #              Wait: be exact! From -gamma_1(tau) v_2(tau+1), we already took -gamma_1 M v_2 in 3.1
    #              and -gamma_1 (q/n) Q_0 P eta_tau in 3.2; what remains of -gamma_1(tau) v_2(tau+1) is:
    #              -gamma_1(tau) * [ q Q_0 P(xi_tau . v_1(tau)) - gamma_1(tau) (M v_1(tau) + (q/n) Q_0 P xi_tau) ]!
    #   Piece 3.4: -gamma_2(tau) * v_1(tau+1)
    #              = -[ q xi_tau^T v_1(tau) + (q/n) 1^T eta_tau ] * [ M v_1(tau) + (q/n) Q_0 P xi_tau ]!

    # Let's evaluate E[xi_hat_{-m}(tau) * (Piece 3.k)_m] for k = 1, 2, 3, 4!

    # --- Piece 3.1: sum_{b!=0} K_{m,b} E[ xi_{-m}(tau) xi_{m-b}(tau) v2_hat_b(tau) ] ---
    # Notice at SAME time tau:
    # - For 4-xi part of v2_hat_b(tau) = sum_{l2>=1} mu_b^{l2-1} [ sum_{a!=0} K_{b,a} xi_{b-a}(tau-l2) v1_hat_a(tau-l2)
    #                                                             - (q^2 lam_b / n^2) xi_0(tau-l2) xi_b(tau-l2) ]:
    #   * First subpiece: xi_{-m}(tau) xi_{m-b}(tau) xi_{b-a}(tau-l2) xi_a(tau-l2-l1) (m, b, a != 0):
    #     Three Wick pairings:
    #     P1: (-m with m-b) => b = 0 (impossible since b != 0!). ZERO!
    #     P2: (-m with b-a) => a = b - m (with a != 0, i.e. b != m), AND (m-b with a = b-m) automatic!
    #         Wait: check (-m) + (b - a) = 0 => a = b - m (mod n)! Then (m - b) + a = (m - b) + (b - m) = 0 (mod n)!
    #         Lag factor: phi^{l2} * phi^{l2 + l1} = phi^{2*l2 + l1}!
    #     P3: (-m with a) => a = m, AND (m-b with b-m) automatic for ALL b != 0!
    #         Lag factor: phi^{l2 + l1} * phi^{l2} = phi^{2*l2 + l1}!
    #   * Second subpiece: -(q^2 lam_b / n^2) xi_{-m}(tau) xi_{m-b}(tau) xi_0(tau-l2) xi_b(tau-l2):
    #     - (-m with m-b): b = 0 (no).
    #     - (-m with 0): m = 0 (no).
    #     - (-m with b) => b = m, AND (m-b=0 with 0): YES! Only b = m! Lag factor: phi^{2*l2}!
    #   * Third subpiece (eta in v2): +(q lam_b / n) E[xi_{-m}(tau) xi_{m-b}(tau) eta_b(tau-l2)]
    #     = (q lam_b / n) * n * phi^{2*l2} FOR ALL b != 0!
    # Let's compute Piece 3.1 vector (length n-1, indexed by m in 1..n-1):
    p31 = np.zeros(n - 1, dtype=complex)
    # P3 contribution: a = m, all b != 0:
    # n^2 * (1/n from v1) = n * sum_{b!=0} K_{m,b} * sum_l2_phi2[b] * K_{b,m} * (g1_a[m] / (q lam_m / n)? Wait: let's be exact!)
    # Recall E[v1_hat_a(tau-l2) * xi_{-a}(tau-l2)] ... wait:
    # v1_hat_a(tau-l2) = (q lam_a / n) sum_{l1>=1} mu_a^{l1-1} xi_a(tau-l2-l1).
    # When we factor out n^2 from Wick pairing of 4 xi_hats, the remaining factor from v1_hat_a is:
    #   (q lam_a / n) * [phi / (1 - phi * mu_a)] = g1_a / n!
    # Multiply by n^2: gives n * g1_a!
    # Check tb_a earlier!! Look at line 139:
    #   In tb_a, we had K_{m,m} * v1_hat_m, and we wrote `n * K_diag * sum_l2_phi2 * g1_a`!
    #   Let's verify: K_{m,m} has 1/n, v1_hat_m has (q lam_m / n), Wick gives n^2 -> n^2 * K_{m,m} * (g1_a / n) = n * K_{m,m} * g1_a! EXACT!
    for mi_idx, m in enumerate(idx):
        # P3: a = m, sum over all b in 1..n-1
        val_p3 = n * g1_a[mi_idx] * np.sum(K_ma[mi_idx, :] * sum_l2_phi2 * K_ma[:, mi_idx])
        # P2: for each b in 1..n-1 with b != m, a = (b - m) % n (which is in 1..n-1!)
        val_p2 = 0.0 + 0.0j
        for bi_idx, b in enumerate(idx):
            if b == m:
                continue
            a = (b - m) % n
            ai_idx = a - 1
            val_p2 += n * K_ma[mi_idx, bi_idx] * sum_l2_phi2[bi_idx] * K_ma[bi_idx, ai_idx] * g1_a[ai_idx]
        # Second subpiece (b = m): -(q^2 lam_m / n^2) * n^2 * K_{m,m} * sum_l2_phi2[m]
        val_sub2 = -K_ma[mi_idx, mi_idx] * (q * q * lam[m]) * sum_l2_phi2[mi_idx]
        # Third subpiece (all b != 0): +(q lam_b / n) * n * K_{m,b} * sum_l2_phi2[b]
        val_sub3 = np.sum(K_ma[mi_idx, :] * (q * lam[idx]) * sum_l2_phi2)
        p31[mi_idx] = val_p2 + val_p3 + val_sub2 + val_sub3

    # --- Piece 3.2: (q lam_m / n) sum_{a!=0} E[xi_{-m}(tau) eta_{m-a}(tau) v1_hat_a(tau)]
    #                - (q^2 lam_m / n^2) E[xi_{-m}(tau) xi_0(tau) eta_m(tau)] ---
    # * First term: E[eta_{m-a}(tau) xi_{-m}(tau) xi_a(tau-l1)] = n * phi^(0 + l1) = n * phi^{l1} FOR ALL a != 0!
    #   Sum over l1: (q lam_a / n) * n * phi / (1 - phi mu_a) = g1_a[a]!
    #   Multiply by (q lam_m / n) and sum over a != 0: (q lam_m / n) * sum_{a!=0} g1_a!
    # * Second term: E[eta_m(tau) xi_{-m}(tau) xi_0(tau)] = n * 1 = n!
    #   Multiply by -(q^2 lam_m / n^2): -(q^2 lam_m / n)!
    p32 = (q * lam[idx] / n) * np.sum(g1_a) - (q * q * lam[idx] / n)

    # --- Piece 3.3: -gamma_1(tau) * [ q Q_0 P(xi_tau . v_1(tau)) - gamma_1(tau) (M v_1(tau) + (q/n) Q_0 P xi_tau) ] ---
    # Notice q Q_0 P(xi_tau . v_1(tau)) - gamma_1(tau) M v_1(tau) in mode m != 0 is simply:
    #   sum_{a!=0} K_{m,a} xi_{m-a}(tau) v1_hat_a(tau)!
    # Multiply by -gamma_1(tau) = -(q/n) xi_0(tau), and add +(q/n)^2 xi_0(tau)^2 * (q lam_m / n) xi_m(tau):
    # Now take E[xi_{-m}(tau) * (...)]:
    # * First part: -(q/n) sum_{a!=0} K_{m,a} E[xi_{-m}(tau) xi_0(tau) xi_{m-a}(tau) v1_hat_a(tau)] (m != 0, a != 0):
    #   Pairings of (-m, 0, m-a, a at tau-l1):
    #   - (-m with 0): 0.
    #   - (-m with m-a) => a = 0: 0.
    #   - (-m with a at tau-l1) => a = m, AND (0 with m-a = 0) at time tau!
    #     Gives n^2 * 1 * phi^{l1} -> -(q/n) * K_{m,m} * n * g1_a[m] = -q * K_diag * g1_a!
    # * Second part: +(q^3 lam_m / n^3) E[xi_{-m}(tau) xi_0(tau)^2 xi_m(tau)] (m != 0):
    #   Only (-m with m) and (0 with 0) survives: n^2!
    #   Gives +(q^3 lam_m / n)!
    p33 = -q * K_diag * g1_a + (q ** 3) * lam[idx] / n

    # --- Piece 3.4: -gamma_2(tau) * v_1(tau + 1) in mode m != 0 ---
    # Write gamma_2(tau) = (q/n) sum_{a!=0} xi_{-a}(tau) v1_hat_a(tau) + (q/n) eta_0(tau),
    # and v1_hat_m(tau + 1) = mu_m v1_hat_m(tau) + (q lam_m / n) xi_m(tau).
    # Multiply out the 4 sub-products and contract with xi_{-m}(tau):
    # (3.4a) -(q/n) mu_m sum_{a!=0} E[ xi_{-m}(tau) xi_{-a}(tau) v1_hat_a(tau) v1_hat_m(tau) ]!!
    #        LOOK AT (3.4a)!! THIS IS THE TWO-v1 COVARIANCE TERM WE SPOTTED ON PAPER!!
    #        Three Wick pairings of (xi_{-m}(tau), xi_{-a}(tau), v1_hat_a(tau), v1_hat_m(tau)):
    #        - Pair (xi_{-m}, xi_{-a}) => a = -m (mod n, which is in 1..n-1!), AND pair (v1_hat_{-m}(tau), v1_hat_m(tau))!
    #          What is E[v1_hat_{-m}(tau) v1_hat_m(tau)] = E|v1_hat_m(tau)|^2?!
    #          Let V1_m := E|v1_hat_m|^2 = (q^2 / n) * (1 - phi^2 |mu_m|^2) / ((1 - |mu_m|^2) |1 - phi mu_m|^2)!
    #          Wait: check factor of n in E|v1_hat_m|^2!
    #          v1_hat_m = (q lam_m / n) sum_{l>=1} mu_m^{l-1} xi_m(t-l), and E|xi_m|^2 = n -> (q^2 / n^2) * n * (...) = (q^2 / n) * (...)!
    #          Check Parseval: E||v_1||^2 = (1/n) sum_{m!=0} E|v1_hat_m|^2 = (q^2 / n^2) sum (...), matching E_v1_norm2!
    #          And E[xi_{-m}(tau) xi_m(tau)] = n!
    #          So this pairing gives: -(q/n) mu_m * n * V1_m = -q * mu_m * V1_m!!
    #        - Pair (xi_{-m}(tau), v1_hat_a(tau)) => a = m, AND (xi_{-m}(tau), v1_hat_m(tau)):
    #          Since E[xi_{-m}(tau) v1_hat_m(tau)] = g1_a[m], this gives -(q/n) mu_m * g1_a[m]^2!
    #        - Pair (xi_{-m}(tau), v1_hat_m(tau)) AND sum_{a!=0} E[xi_{-a}(tau) v1_hat_a(tau)]:
    #          Note sum_{a!=0} E[xi_{-a} v1_hat_a] = sum_{a!=0} g1_a = n * E_xi_dot_v1!
    #          Gives -(q/n) mu_m * g1_a[m] * sum_{a!=0} g1_a!
    V1_m = (q * q / n) * (1.0 - (phi ** 2) * (np.abs(mu[idx]) ** 2)) / (
        (1.0 - np.abs(mu[idx]) ** 2) * (np.abs(1.0 - phi * mu[idx]) ** 2)
    )
    p34a = -q * mu[idx] * V1_m - (q / n) * mu[idx] * (g1_a ** 2) - (q / n) * mu[idx] * g1_a * np.sum(g1_a)

    # (3.4b) -(q/n) * (q lam_m / n) sum_{a!=0} E[ xi_{-m}(tau) xi_m(tau) xi_{-a}(tau) v1_hat_a(tau) ]:
    #        Three Wick pairings:
    #        - (-m with m) [gives n] * sum_{a!=0} E[xi_{-a} v1_hat_a] [gives sum g1_a] -> n * sum g1_a!
    #        - (-m with -a) => a = n - m: E[xi_{-m} xi_m] [n] * E[xi_m(tau) v1_hat_{-m}(tau)] [g1_a[-m]] -> n * g1_a[-m]!
    #        - (-m with v1_hat_a) => a = m: g1_a[m] * E[xi_m xi_{-m}] [n] -> n * g1_a[m]!
    g1_neg_m = g1_a[::-1]  # since idx = 1..n-1, n - m corresponds to reversed index!
    p34b = -(q * q * lam[idx] / n) * (np.sum(g1_a) + g1_neg_m + g1_a)

    # (3.4c) -(q/n) mu_m E[ xi_{-m}(tau) eta_0(tau) v1_hat_m(tau) ]:
    #        E[eta_0(tau) xi_{-m}(tau) xi_m(tau-l1)] = n * phi^{0 + l1} = n * phi^{l1} -> n * (g1_a[m] / n) = g1_a[m]!
    #        Multiply by -(q/n) mu_m: -(q/n) mu_m * g1_a[m]!
    p34c = -(q / n) * mu[idx] * g1_a

    # (3.4d) -(q/n) * (q lam_m / n) E[ xi_{-m}(tau) eta_0(tau) xi_m(tau) ]:
    #        E[eta_0(tau) xi_{-m}(tau) xi_m(tau)] = n * 1 = n!
    #        Multiply by -(q^2 lam_m / n^2): -(q^2 lam_m / n)!
    p34d = -(q * q * lam[idx] / n)

    F3_corr = p31 + p32 + p33 + p34a + p34b + p34c + p34d
    E_q_xi_dot_v3 = (q / n) * float(np.sum(R_m * F3_corr).real)
    T_A = -E_q_xi_dot_v3 - E_q_eta_dot_v2

    H_total = T_A + T_B + T_C + T_D + T_E
    C2 = primary_C2(n, s, gbar, phi)
    p34a_v1_piece = -(q / n) * float(np.sum(R_m * (-q * mu[idx] * V1_m)).real)
    p31_piece = -(q / n) * float(np.sum(R_m * p31).real)
    return {
        "H_total": H_total,
        "C2": C2,
        "H_over_C": H_total / C2,
        "T_A": T_A,
        "T_B": T_B,
        "T_C": T_C,
        "T_D": T_D,
        "T_E": T_E,
        "E_q_xi_dot_v3": E_q_xi_dot_v3,
        "E_q_eta_dot_v2": E_q_eta_dot_v2,
        "E_v1_norm2": E_v1_norm2,
        "p34a_v1_piece": p34a_v1_piece,
        "p31_piece": p31_piece,
    }


def perturbative_mc_H4(
    n: int,
    s: float,
    gbar: float,
    phi: float,
    steps: int = 80000,
    burn: int = 4000,
    reps: int = 120,
    seed: int = 20261113,
) -> tuple[float, float]:
    """Direct sigma-free simulation of (v_1, v_2, v_3) in stage space (zero Wick/Fourier assumptions!)."""
    rng = np.random.default_rng(seed)
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    v1 = np.zeros((reps, n))
    v2 = np.zeros((reps, n))
    v3 = np.zeros((reps, n))
    xi = rng.normal(0.0, 1.0, size=(reps, n))
    innov = math.sqrt(max(0.0, 1.0 - phi * phi))
    acc = np.zeros(reps)

    for t in range(steps + burn):
        eta = 0.5 * (xi * xi - 1.0)
        g1 = (q / n) * xi.sum(axis=1)
        g2 = q * np.sum(xi * v1, axis=1) + (q / n) * eta.sum(axis=1)
        g3 = q * np.sum(xi * v2, axis=1) + q * np.sum(eta * v1, axis=1)
        g4 = q * np.sum(xi * v3, axis=1) + q * np.sum(eta * v2, axis=1)

        if t >= burn:
            # Subtract exact zero-mean control -(q/n)*0 + ... (g1^4/4 etc. are already O(1) variance)
            h_step = -g4 + g1 * g3 + 0.5 * (g2 ** 2) - (g1 ** 2) * g2 + 0.25 * (g1 ** 4)
            acc += h_step

        # Update v1, v2, v3 on 1^perp:
        # P acts as cyclic shift: (P z)_{k+1} = z_k -> np.roll(z, 1, axis=1)
        P_xi = np.roll(xi, 1, axis=1)
        P_eta = np.roll(eta, 1, axis=1)
        Q0P_xi = P_xi - P_xi.mean(axis=1, keepdims=True)
        Q0P_eta = P_eta - P_eta.mean(axis=1, keepdims=True)

        Mv1 = p * v1 + q * np.roll(v1, 1, axis=1)
        Mv2 = p * v2 + q * np.roll(v2, 1, axis=1)
        Mv3 = p * v3 + q * np.roll(v3, 1, axis=1)

        v1_next = Mv1 + (q / n) * Q0P_xi

        pxi_v1 = np.roll(xi * v1, 1, axis=1)
        q0_pxi_v1 = pxi_v1 - pxi_v1.mean(axis=1, keepdims=True)
        v2_next = Mv2 + q * q0_pxi_v1 + (q / n) * Q0P_eta - g1[:, None] * v1_next

        pxi_v2 = np.roll(xi * v2, 1, axis=1)
        q0_pxi_v2 = pxi_v2 - pxi_v2.mean(axis=1, keepdims=True)
        peta_v1 = np.roll(eta * v1, 1, axis=1)
        q0_peta_v1 = peta_v1 - peta_v1.mean(axis=1, keepdims=True)
        v3_next = Mv3 + q * q0_pxi_v2 + q * q0_peta_v1 - g1[:, None] * v2_next - g2[:, None] * v1_next

        v1, v2, v3 = v1_next, v2_next, v3_next
        xi = phi * xi + innov * rng.normal(0.0, 1.0, size=(reps, n))

    vals = acc / steps
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(reps))


if __name__ == "__main__":
    s, gbar = 2.0 / 3.0, 0.4
    print("=== 1. Bug check: exact_analytical_H4 at phi = 0 vs primary eq. (10) ===")
    for n in (1, 2, 3, 4, 8, 16, 32, 64):
        d0 = exact_analytical_H4(n, s, gbar, 0.0)
        h_independent = primary_H4_phi0(n, s, gbar)
        diff = abs(d0["H_total"] - h_independent)
        print(f"  n={n:2d}: exact_analytical={d0['H_total']:+.12f}  primary_eq10={h_independent:+.12f}  diff={diff:.2e}")
        assert diff < 1e-14

    print("\n=== 2. Bug check: exact_analytical_H4 vs sigma-free perturbative MC & primary paired MC at phi = 0.9 ===")
    primary_paired_phi09 = {
        2: (-8.5375014e-06, 1.52e-08),
        4: (-1.0418660e-05, 9.01e-09),
        8: (-5.6704458e-06, 1.88e-08),
        16: (+3.5777612e-05, 2.37e-07),
        32: (+1.6921927e-04, 1.25e-06),
    }
    denom = 0.025 ** 2 - 0.0125 ** 2
    import sys
    run_mc = "--no-mc" not in sys.argv
    if run_mc:
        for n in (2, 4, 8, 16, 32):
            res = exact_analytical_H4(n, s, gbar, 0.9)
            mc_h, mc_se = perturbative_mc_H4(n, s, gbar, 0.9, steps=60000, burn=5000, reps=96, seed=7000 + n)
            cdx_h = primary_paired_phi09[n][0] / denom
            cdx_se = primary_paired_phi09[n][1] / denom
            print(f"  n={n:2d}: H_exact={res['H_total']:+.7f}  H_pertMC={mc_h:+.7f}+/-{mc_se:.5f}  "
                  f"H_independent_paired={cdx_h:+.7f}+/-{cdx_se:.5f}  --> c_n = H_n/C_n = {res['H_over_C']:+.4f}")

    print("\n=== 3. Scaling of H_n(0.9) and c_n(0.9) = H_n / C_n from n = 1 to 512 ===")
    print("    n      C_n(0.9)       H_n(0.9)      c_n = H/C     diag=(q/12p)(1+phi)/(1-phi)*n    c_n/diag    n*E||v1||^2   p34a_v1    p31       T_C")
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    for n in (1, 2, 4, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512):
        res = exact_analytical_H4(n, s, gbar, 0.9)
        diag_c = (q / (12.0 * p)) * (1.9 / 0.1) * n
        print(f"  {n:4d}   {res['C2']:11.7f}   {res['H_total']:+12.7f}   {res['H_over_C']:+11.4f}   "
              f"{diag_c:18.4f}          {res['H_over_C']/diag_c:+8.4f}   {n*res['E_v1_norm2']:10.4f}  "
              f"{res['p34a_v1_piece']:+8.4f} {res['p31_piece']:+8.4f} {res['T_C']:+8.4f}")

    print("\n=== 4. Asymptotic continuum crossover approximation g(theta_n) vs formal finite-n H_n(0.9) ===")
    print("  theta_n = (1 - phi)*n / (2*sqrt(phi*q*(1 - phi*p)));  H_inf = (q^3 / (24*p)) * ((1+phi)/(1-phi))^2")
    h_inf = (q ** 3 / (24.0 * p)) * ((1.9 / 0.1) ** 2)
    n_phi = 2.0 * math.sqrt(0.9 * q * (1.0 - 0.9 * p)) / 0.1
    for n in (16, 24, 32, 48, 64, 96, 128, 256, 512):
        res = exact_analytical_H4(n, s, gbar, 0.9)
        theta = n / n_phi
        coth_t = 1.0 / math.tanh(theta)
        csch2_t = 1.0 / (math.sinh(theta) ** 2)
        f_t = 1.0 - 3.0 * (theta * coth_t - 1.0) / (theta * theta)
        tf_prime = -3.0 * ((coth_t - theta * csch2_t) / theta - 2.0 * (theta * coth_t - 1.0) / (theta * theta))
        g_t = f_t - 0.5 * tf_prime
        w_tc = (1.0 - 0.9) / (1.0 + 0.9)  # = 1 / 19
        h_asymp = h_inf * (w_tc * f_t + (1.0 - w_tc) * g_t)
        c_asymp = h_asymp / res["C2"]
        print(f"  n={n:4d}: theta={theta:6.3f}  f(th)={f_t:6.3f}  g(th)={g_t:6.3f}  "
              f"H_exact={res['H_total']:+8.4f}  H_cont={h_asymp:+8.4f}  c_exact={res['H_over_C']:+8.2f}  c_cont={c_asymp:+8.2f}")
