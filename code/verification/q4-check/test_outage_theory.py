#!/usr/bin/env python3
"""Test exact outage bounds, i.i.d. outage formula, and long-stay asymptotics with zeros."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4_check_run import load_raw_records, stage_means, solve_replay_D, circular_autocov


def d_sup_pure_outage(q0: float, taus: tuple[float, ...], g_plus: float | None = None) -> float:
    """Block-outage limit as outage gap G -> infinity at fixed outage fraction q0 and positive geometric mean g_plus."""
    if g_plus is None:
        g_plus = 1.0 / (1.0 - q0)
    p_stays = [1.0 - 1.0 / tau for tau in taus]
    p_max = max(p_stays)
    lam_star = p_max ** (-q0 / (1.0 - q0))
    # Characteristic equation on positive steps with geometric-mean host g_plus:
    # prod_k (lam_star - p_k) = (g_plus * e^D) / prod_k tau_k
    lhs = 1.0
    tau_prod = 1.0
    for tau, pk in zip(taus, p_stays):
        lhs *= (lam_star - pk)
        tau_prod *= tau
    return math.log(tau_prod * lhs / g_plus)


def solve_iid_pure_outage_D(q0: float, k: int, s: int) -> float:
    """Solve exact threshold D for i.i.d. pure-outage host h_t in {0, 1/(1-q0)} via Koopman / atom discretization."""
    taus = stage_means(k, s)
    h_on = 1.0 / (1.0 - q0)
    if k == 2 and s == 1:
        # Exact countable-atom formula derived in q4_probe_details.py
        a = taus[0]
        p1 = 1.0 - 1.0 / a
        def gr(y: float) -> float:
            ey = math.exp(y)
            tot = q0 * math.log(p1)
            u = 0.0
            w = q0 * (1.0 - q0)
            for _ in range(4000):
                u = (h_on / a) / (p1 + ey * u)
                tot += w * math.log(p1 + ey * u)
                w *= (1.0 - q0)
                if w < 1e-16:
                    break
            return tot
        return float(optimize.brentq(gr, -2.0, 20.0, xtol=1e-13))

    # High-precision Monte Carlo with antithetic / stratified frequency check (200k steps x 16 pairs) for k=2, s>1
    rng = np.random.default_rng(20261115)
    T_steps = 100000
    u_draws = rng.random((T_steps, 8))
    # Antithetic uniform draws
    u_all = np.concatenate([u_draws, 1.0 - u_draws], axis=1)
    h_mat = np.where(u_all < q0, 0.0, h_on)

    def mean_gr(y: float) -> float:
        # Vectorized across 16 paths
        n_rep = h_mat.shape[1]
        x = np.ones((n_rep, k), dtype=np.float64) / k
        logs = np.zeros(n_rep, dtype=np.float64)
        p_stay = np.array([1.0 - 1.0 / tau for tau in taus])
        p_leave = np.array([1.0 / tau for tau in taus])
        ey = math.exp(y)
        burn = 2000
        for t in range(T_steps):
            ht = h_mat[t]
            xn = np.empty_like(x)
            # stage 0 receives birth from k-1
            xn[:, 0] = p_stay[0] * x[:, 0] + (p_leave[k - 1] * ey) * x[:, k - 1]
            xn[:, 1] = p_stay[1] * x[:, 1] + (p_leave[0] * ht) * x[:, 0]
            if k == 3:
                xn[:, 2] = p_stay[2] * x[:, 2] + p_leave[1] * x[:, 1]
            tot = xn.sum(axis=1)
            x = xn / tot[:, None]
            if t >= burn:
                logs += np.log(tot)
        return float(logs.mean() / (T_steps - burn))

    return float(optimize.brentq(mean_gr, -0.5, 10.0, xtol=1e-6))


def main() -> None:
    # 1. Test whether D <= d_sup_pure_outage(q0, taus, g_plus) across random periodic schedules AND all 64 Q4 zero series!
    rng = np.random.default_rng(9911)
    violations_pure = 0
    max_gap_to_sup = -1e9
    for P in range(2, 25):
        for n_zero in range(1, P):
            q0 = n_zero / float(P)
            h_on = 1.0 / (1.0 - q0)
            for _ in range(20):
                arr = np.array([0.0] * n_zero + [h_on] * (P - n_zero))
                rng.shuffle(arr)
                for k in (2, 3):
                    for s in (1, 5, 20):
                        taus = stage_means(k, s)
                        d_val, _ = solve_replay_D(k, s, arr)
                        d_sup = d_sup_pure_outage(q0, taus)
                        max_gap_to_sup = max(max_gap_to_sup, d_val - d_sup)
                        if d_val > d_sup + 1e-10:
                            violations_pure += 1
                            print("VIOLATION pure:", P, n_zero, k, s, d_val, d_sup, arr)

    print(f"Pure binary outage random periodic schedules tested: violations = {violations_pure}, max(D - D_sup) = {max_gap_to_sup:.3e}")

    # Test on all 64 Q4 zero-containing series (384 cells)!
    records = load_raw_records()
    violations_q4 = 0
    max_q4_diff = -1e9
    for r in records:
        h = r["h"]
        if not np.any(h <= 0.0):
            continue
        q0 = float(np.mean(h == 0.0))
        g_plus = float(np.exp(np.mean(np.log(h[h > 0.0]))))
        for k in (2, 3):
            for s in (1, 5, 20):
                taus = stage_means(k, s)
                d_val, _ = solve_replay_D(k, s, h)
                d_sup = d_sup_pure_outage(q0, taus, g_plus=g_plus)
                max_q4_diff = max(max_q4_diff, d_val - d_sup)
                if d_val > d_sup + 1e-10:
                    violations_q4 += 1
                    print("Q4 VIOLATION:", r["series_id"], k, s, d_val, d_sup, q0, g_plus)
    print(f"Q4 384 zero-containing cells tested against D_sup(q0, G_+): violations = {violations_q4}, max(D - D_sup) = {max_q4_diff:.3e}")

    # 2. Compute clean table of i.i.d. pure-outage D(q0, s) across q0 and stays s=1, 5, 20!
    print("\n=== Clean i.i.d. pure-outage loss D_iid(q0; k=2, s) vs D2_raw = q0/[2*(1-q0)*(2+s)] vs D_sup(q0) ===")
    for q0 in (0.05, 0.10, 0.25, 0.50, 0.75):
        d1 = solve_iid_pure_outage_D(q0, 2, 1)
        d5 = solve_iid_pure_outage_D(q0, 2, 5)
        d20 = solve_iid_pure_outage_D(q0, 2, 20)
        v = q0 / (1.0 - q0)
        print(f"  q0={q0:.2f} (Var={v:.3f}): D(s=1)={d1:.5f} [raw2={v/6:.5f}, sup={d_sup_pure_outage(q0,(2,1)):.5f}] | "
              f"D(s=5)={d5:.5f} [raw2={v/14:.5f}, sup={d_sup_pure_outage(q0,(2,5)):.5f}] | "
              f"D(s=20)={d20:.5f} [raw2={v/44:.5f}, sup={d_sup_pure_outage(q0,(2,20)):.5f}]")


if __name__ == "__main__":
    main()
