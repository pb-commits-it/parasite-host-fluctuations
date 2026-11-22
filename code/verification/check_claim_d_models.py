"""Quantitative check of tertiary's W1.1 assessment of Claim d across four ODE formulations:
  1. Round-1 toy ODE (`attack_stability.py` / `w11_replication.py`): regulation -c P_1^2 in stage 1 only,
     all stage durations shrunk as tau/n. Verifies exact analytic proof that the leading root is ALWAYS
     real for all R > 1 and n >= 2, and converges to -(1/tau) ln(2 - 1/R) as n -> inf.
  2. Symmetric comparably-regulated ring: identical regulation -c P_k^2 in EVERY stage k = 1..n at
     matched whole-cycle R and matched cycle time tau. Leading root is STILL real (-r_Malthusian),
     and STILL weakens monotonically by >5x as n grows (-7.00 -> -2.87 -> -1.94 -> -1.61 -> -1.35)!
  3. Fixed-adult-duration + n-stage larval Erlang delay (May & Anderson 1978 Part II / Blythe et al. 1984):
     adult worm stage has fixed residence tau_A = 1 with -c P^2, and larval transmission delay tau_L = 1
     is split into m = 1, 2, 3, 4, 6, 8 unregulated larval stages of exit rate m / tau_L at fixed R.
  4. Coupled host-macroparasite ODE (Dobson eqs. 1.6-1.11 corrected / Anderson & May 1978 + m larval stages)
     and Lloyd (2001) Erlang-n SIR model: regulation acts dynamically via host/susceptible population,
     producing oscillatory leading eigenvalues whose damping weakens steadily with stage number!
"""
from __future__ import annotations

import math
import numpy as np


def model1_toy(R: float = 15.0, tau: float = 2.0) -> None:
    print("=== Model 1: Round-1 toy (regulation in stage 1 only, all n stages exit at n/tau) ===")
    for n in (1, 2, 3, 4, 6, 8, 32, 128):
        g = n / tau
        if n == 1:
            ev = np.array([-g * (R - 1.0)])
        else:
            J = -g * np.eye(n) + np.roll(g * np.eye(n), 1, axis=0)
            J[0, 0] = -g * (2.0 * R - 1.0)
            J[0, -1] = g * R
            ev = np.linalg.eigvals(J)
        lead = ev[int(np.argmax(ev.real))]
        print(f"  n={n:3d}: lead = {lead.real:+.5f} {lead.imag:+.5f}i   max|Im|={np.max(np.abs(ev.imag)):.4f}")
    lim = -(1.0 / tau) * math.log(2.0 - 1.0 / R)
    print(f"  Exact n -> inf limit: -(1/tau)*ln(2 - 1/R) = {lim:+.5f}")


def model2_all_stages_regulated(R: float = 15.0, tau: float = 2.0) -> None:
    print("\n=== Model 2: Symmetric regulation -c P_k^2 in EVERY stage k = 1..n (matched R=15, tau=2) ===")
    for n in (1, 2, 3, 4, 6, 8, 32):
        g = n / tau
        r_step = R ** (1.0 / n)
        J = -g * (2.0 * r_step - 1.0) * np.eye(n) + np.roll(g * r_step * np.eye(n), 1, axis=0)
        ev = np.linalg.eigvals(J)
        lead = ev[int(np.argmax(ev.real))]
        exact_lead = -g * (r_step - 1.0)
        assert abs(lead.real - exact_lead) < 1e-11 and abs(lead.imag) < 1e-11
        print(f"  n={n:3d}: lead = {lead.real:+.5f} (exact -r_Malthusian = {exact_lead:+.5f})  max|Im|={np.max(np.abs(ev.imag)):.4f}")
    print(f"  Exact n -> inf limit: -ln(R)/tau = {-math.log(R)/tau:+.5f}")


def model3_fixed_adult_plus_larval_chain(R: float = 15.0, tau_A: float = 1.0, tau_L: float = 1.0) -> None:
    print("\n=== Model 3: Fixed adult stage (tau_A=1, -c P^2) + m larval delay stages (total tau_L=1, R=15) ===")
    gA = 1.0 / tau_A
    for m in (1, 2, 3, 4, 6, 8, 16):
        n = 1 + m
        gL = m / tau_L
        J = np.zeros((n, n))
        J[0, 0] = -gA * (2.0 * R - 1.0)
        J[0, -1] = gA * R
        for k in range(1, n):
            J[k, k] = -gL
            J[k, k - 1] = gL if k > 1 else gL  # scale invariant under diagonal similarity
        ev = np.linalg.eigvals(J)
        lead = ev[int(np.argmax(ev.real))]
        print(f"  larval stages m={m:2d} (total n={n:2d}): lead = {lead.real:+.5f} {lead.imag:+.5f}i  "
              f"({'OSCILLATORY' if abs(lead.imag) > 1e-7 else 'real'})")


def model4_lloyd_and_dobson_host_coupled() -> None:
    print("\n=== Model 4a: Lloyd (2001) Erlang-n SIR endemic equilibrium (R0=15, mu=0.02, D=2.0) ===")
    R0, mu, D = 15.0, 0.02, 2.0
    gamma = 1.0 / D
    # S* / N = 1 / R0; beta * I_tot* = mu * (R0 - 1)
    # Linearized system on (delta S, delta I_1, ..., delta I_n) with N=1:
    for n in (1, 2, 3, 4, 6, 8):
        ng = n * gamma
        # Exact R0 for Erlang-n with mortality mu: R0 = beta * [1 - (ng/(ng + mu))^n] / mu
        surv_factor = 1.0 - (ng / (ng + mu)) ** n
        beta = R0 * mu / surv_factor
        s_star = 1.0 / R0
        # At endemic eq: beta * s_star * I_tot* = inflow_I1, and I_tot* = mu * (1 - s_star) / (beta * s_star) * ...
        # Direct equilibrium: mu*(1 - s_star) = beta * s_star * I_tot* => beta * I_tot* = mu * (R0 - 1)!
        bI = mu * (R0 - 1.0)
        J = np.zeros((n + 1, n + 1))
        J[0, 0] = -mu - bI
        J[0, 1:] = -beta * s_star
        J[1, 0] = bI
        J[1, 1:] = beta * s_star
        for k in range(1, n + 1):
            J[k, k] -= (ng + mu)
            if k > 1:
                J[k, k - 1] += ng
        ev = np.linalg.eigvals(J)
        lead = ev[int(np.argmax(ev.real))]
        print(f"  n={n:2d}: lead = {lead.real:+.6f} {lead.imag:+.6f}i  ({'OSCILLATORY' if abs(lead.imag)>1e-8 else 'real'})")

    print("\n=== Model 4b: Corrected Dobson/Anderson-May coupled host-macroparasite ODE + m larval stages ===")
    # Definitive host H: dH/dt = (b - d)*H - alpha * P  (net host growth r_H = b - d > 0 regulated by parasite!)
    # Adult parasite P in H: dP/dt = g_L * L_m - (mu_P + d + alpha) * P - (alpha / k_clump) * P^2 / H
    # Larval chain L_1..L_m (total larval delay tau_L = 1.0, exit g_L = m / tau_L):
    #   dL_1/dt = lam_eff * P - g_L * L_1   (with linear uptake, or saturating H uptake; take standard A&M Model F!)
    # Check Anderson & May (1978 Part II, Model F): r_H = 0.2, mu_P+d+alpha = 1.0, alpha = 0.05, k_clump = 0.5,
    # lam_eff = 6.0 (above threshold mu_P+d+alpha + alpha*r_H/... ):
    r_H = 0.2
    gamma_P = 1.0     # mu_P + d + alpha (adult residence = 1.0)
    alpha = 0.05
    k_clump = 0.5     # moderately aggregated (Anderson & May stability requires k_clump small vs larval delay!)
    lam_eff = 5.0     # reproductive output surviving larval mortality at equilibrium
    tau_L = 1.0       # fixed total larval/intermediate-host delay = 1.0
    # Equilibrium: dH/dt = 0 => P* / H* = r_H / alpha = 4.0!
    # dL/dt = 0 => g_L L_m* = lam_eff * P*
    # dP/dt = 0 => lam_eff - gamma_P - (alpha / k_clump) * (P* / H*) = 0? Wait: choose lam_eff = gamma_P + (alpha/k_clump)*(r_H/alpha) = 1.0 + r_H/k_clump!
    # Wait: in Anderson & May (1978) with free-living uptake beta*H*W / (sigma + beta*H), R_0 depends on H*!
    # Specifically, transmission into L_1 or out of L_m saturates or scales with H: intake = beta * H * L_m / (H_0 + H) or beta * H * W!
    # Take standard Anderson & May (1978 eq. 18): transmission into H is proportional to H:
    #   dH/dt = r_H * H - alpha * P
    #   dL_1/dt = lam * P - g_L * L_1;  dL_j/dt = g_L L_{j-1} - g_L L_j (j=2..m-1);  dL_m/dt = g_L L_{m-1} - (sigma_0 + beta * H) L_m
    # Or simpler: every larval stage j=1..m has exit g_L = m / tau_L, and stage-1 production or stage-m uptake has factor H / (H_50 + H) (Dobson's Type II / mass-action host factor)!
    H_50 = 10.0
    # At eq: P* = (r_H / alpha) * H*. Need lam * [H* / (H_50 + H*)] = gamma_P + (alpha / k_clump)*(P*/H*) = gamma_P + r_H / k_clump.
    req = gamma_P + r_H / k_clump  # = 1.0 + 0.4 = 1.4
    lam = 2.8  # so H* / (H_50 + H*) = 0.5 => H* = H_50 = 10.0, P* = 40.0 (identical equilibrium for ALL m!)
    H_eq = H_50 * req / (lam - req)
    P_eq = (r_H / alpha) * H_eq
    for m in (0, 1, 2, 3, 4, 6, 8):
        # m = 0 means instantaneous transmission (1 parasite stage P + host H); m >= 1 means m larval delay stages of total mean time tau_L = 1.0
        def rhs(y: np.ndarray) -> np.ndarray:
            H = y[0]
            P = y[1]
            f_H = H / (H_50 + H)
            dH = r_H * H - alpha * P
            if m == 0:
                dP = lam * f_H * P - gamma_P * P - (alpha / k_clump) * (P * P / H)
                return np.array([dH, dP])
            L = y[2:]
            gL = m / tau_L
            dL = np.empty(m)
            dL[0] = lam * P - gL * L[0]
            if m > 1:
                dL[1:] = gL * L[:-1] - gL * L[1:]
            dP = gL * f_H * L[-1] - gamma_P * P - (alpha / k_clump) * (P * P / H)
            return np.concatenate(([dH, dP], dL))

        if m == 0:
            y_eq = np.array([H_eq, P_eq])
        else:
            gL = m / tau_L
            y_eq = np.concatenate(([H_eq, P_eq], np.full(m, lam * P_eq / gL)))
        assert np.max(np.abs(rhs(y_eq))) < 1e-12
        dim = len(y_eq)
        J = np.zeros((dim, dim))
        for j in range(dim):
            eps = 1e-7 * max(1.0, abs(y_eq[j]))
            up, dn = y_eq.copy(), y_eq.copy()
            up[j] += eps
            dn[j] -= eps
            J[:, j] = (rhs(up) - rhs(dn)) / (2.0 * eps)
        ev = np.linalg.eigvals(J)
        lead = ev[int(np.argmax(ev.real))]
        status = "UNSTABLE HOPF!" if lead.real > 0 else ("OSCILLATORY" if abs(lead.imag) > 1e-7 else "real")
        print(f"  larval stages m={m:2d} (total parasite stages={1+m:2d}): lead = {lead.real:+.5f} {lead.imag:+.5f}i  ({status})")


if __name__ == "__main__":
    model1_toy()
    model2_all_stages_regulated()
    model3_fixed_adult_plus_larval_chain()
    model4_lloyd_and_dobson_host_coupled()
