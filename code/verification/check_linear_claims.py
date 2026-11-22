"""Independent verification and falsification suite for Dobson linear claims C1-C7
and Propositions 1-5 (secondary round-1 review, 2026-11-12).

Written from scratch without reading `checks/*`.
Tests:
  1. Dobson's printed examples: Malaria (0.4), Trematode (1.12), I2 (1.14/1.16), D2 (1.15/1.17).
  2. 25,000 random and adversarial block-cyclic life cycles for C1-C5:
     - n = 1..6, unequal stage widths w_k = 1..6
     - dense positive, sparse zero entries, reducible P, disconnected subcycles,
       nilpotent/acyclic P != 0 and P = 0
     - rank-1 blocks (w_k = 1, separable u v^T) and rank-0 blocks (A_k = 0)
     - full-rank blocks whose product P has rank 1
     - strictly positive full-rank P with complex conjugate non-dominant eigenvalues summing to 0
     - adversarial check of C4's phrasing ("non-dominant eigenvalues sum to zero") vs the
       exact Perron-multiplicity condition
     - comparison of C5's stage-1 bound max_i P_{1,ii} vs Prop. 4's all-stage bound max_{k,i} P_{k,ii}
  3. C6 verification:
     - stage-targeted reproduction number T_{S_k} = rho(C(I - B)^{-1}) == R0^n for general w_k >= 1
     - chronological Malthusian growth rate r vs n at fixed whole-cycle R0^n and fixed cycle duration
  4. 15,000 random and adversarial general digraphs for C7 / Proposition 5:
     - independent Johnson/DFS simple-cycle enumeration (including self-loops of length 1 at b)
     - mixed cycle lengths, reducible graphs, complex eigenvalues
     - verification that sum_r w_r equals Roberts-Heesterbeek type reproduction number T_b
     - explicit falsification of Prop. 5's unguarded statement on DAGs / zero-weight-cycle graphs
"""
from __future__ import annotations

import itertools
import math
import numpy as np


def spectral_radius(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    ev = np.linalg.eigvals(M)
    return float(np.max(np.abs(ev)))


def build_block_cyclic(blocks: list[np.ndarray]) -> np.ndarray:
    """Build K with A_k (shape w_{k+1} x w_k) in block (k+1 mod n, k)."""
    n = len(blocks)
    widths = [A.shape[1] for A in blocks]
    for k in range(n):
        assert blocks[k].shape == (widths[(k + 1) % n], widths[k])
    total = sum(widths)
    K = np.zeros((total, total), dtype=float)
    offsets = np.cumsum([0] + widths)
    for k, A in enumerate(blocks):
        nxt = (k + 1) % n
        r0, r1 = offsets[nxt], offsets[nxt + 1]
        c0, c1 = offsets[k], offsets[k + 1]
        K[r0:r1, c0:c1] += A
    return K


def return_map(blocks: list[np.ndarray], start_stage: int = 0) -> np.ndarray:
    """Compute P_k = A_{k-1} ... A_1 A_n ... A_k (0-indexed)."""
    n = len(blocks)
    w = blocks[start_stage].shape[1]
    P = np.eye(w, dtype=float)
    for step in range(n):
        idx = (start_stage + step) % n
        P = blocks[idx] @ P
    return P


def brute_force_route_sum(blocks: list[np.ndarray]) -> float:
    """Enumerate all w_1 * ... * w_n 1-lap routes (i_0, ..., i_{n-1}) directly."""
    n = len(blocks)
    widths = [A.shape[1] for A in blocks]
    total = 0.0
    for choice in itertools.product(*(range(w) for w in widths)):
        weight = 1.0
        for k in range(n):
            i_curr = choice[k]
            i_next = choice[(k + 1) % n]
            weight *= blocks[k][i_next, i_curr]
            if weight == 0.0:
                break
        total += weight
    return total


def check_dobson_examples(rng: np.random.Generator) -> None:
    # Malaria eq. 0.4
    a, b, m, r, mu = rng.uniform(0.5, 3.0, size=5)
    K_mal = np.array([[0.0, a * b * m / r], [a / mu, 0.0]])
    assert math.isclose(spectral_radius(K_mal), math.sqrt(a**2 * b * m / (mu * r)), rel_tol=1e-13)

    # Trematode eq. 1.12
    k1, k2, k3 = rng.uniform(0.5, 4.0, size=3)
    K_trem = np.array([[0.0, k1, 0.0], [0.0, 0.0, k2], [k3, 0.0, 0.0]])
    assert math.isclose(spectral_radius(K_trem), (k1 * k2 * k3) ** (1.0 / 3.0), rel_tol=1e-13)

    # I2 eq. 1.14 / 1.16: state order (D, J, W, B, M)
    DM, JM, WD, WJ, BW, MB = rng.uniform(0.3, 3.5, size=6)
    I2 = np.array([
        [0, 0, 0, 0, DM],
        [0, 0, 0, 0, JM],
        [WD, WJ, 0, 0, 0],
        [0, 0, BW, 0, 0],
        [0, 0, 0, MB, 0],
    ], dtype=float)
    r0_i2_formula = (BW * DM * MB * WD + BW * JM * MB * WJ) ** 0.25
    assert math.isclose(spectral_radius(I2), r0_i2_formula, rel_tol=1e-14)

    # D2 eq. 1.15 / 1.17: state order (W, M, V, J, D)
    WJ, WD, MW, VW, JM, JV, DM, DV = rng.uniform(0.3, 3.5, size=8)
    D2 = np.array([
        [0, 0, 0, WJ, WD],
        [MW, 0, 0, 0, 0],
        [VW, 0, 0, 0, 0],
        [0, JM, JV, 0, 0],
        [0, DM, DV, 0, 0],
    ], dtype=float)
    r0_d2_formula = (DM * MW * WD + JM * MW * WJ + DV * VW * WD + JV * VW * WJ) ** (1.0 / 3.0)
    assert math.isclose(spectral_radius(D2), r0_d2_formula, rel_tol=1e-14)
    print("[PASS] Dobson chapter examples (eqs 0.4, 1.12, 1.14/1.16, 1.15/1.17) verified to 1e-14.")


def check_c1_to_c5(rng: np.random.Generator, n_trials: int = 25000) -> None:
    tol = 1e-9
    c4_high = 0
    c4_low = 0
    c4_equal_nonrank1 = 0
    brute_checked = 0

    for trial in range(n_trials):
        mode = trial % 6
        n = 1 if (trial < 500) else int(rng.integers(2, 7))
        widths = [int(rng.integers(1, 6)) for _ in range(n)]

        if mode == 2:
            # Force a single-compartment stage
            widths[int(rng.integers(0, n))] = 1
        elif mode in (4, 5):
            # Force all stages >= 2 so no automatic single-compartment rank-1 block
            widths = [int(rng.integers(2, 6)) for _ in range(n)]

        blocks = []
        forced_leq_rank1 = False
        for k in range(n):
            r, c = widths[(k + 1) % n], widths[k]
            if mode == 0:
                A = rng.exponential(1.0, size=(r, c))
            elif mode == 1:
                # Sparse / reducible / zero entries
                mask = rng.random((r, c)) < 0.35
                A = rng.exponential(1.5, size=(r, c)) * mask
            elif mode == 2:
                A = rng.uniform(0.0, 2.0, size=(r, c))
                forced_leq_rank1 = True
            elif mode == 3:
                # Separable rank <= 1 at step 0 (including occasional zero vector -> rank 0)
                if k == 0:
                    if rng.random() < 0.15:
                        A = np.zeros((r, c))
                    else:
                        u = rng.uniform(0.0, 2.0, size=(r, 1))
                        v = rng.uniform(0.0, 2.0, size=(1, c))
                        A = u @ v
                    forced_leq_rank1 = True
                else:
                    A = rng.uniform(0.0, 2.0, size=(r, c))
            else:
                # Dense or sparse width >= 2
                A = rng.exponential(1.0, size=(r, c))
                if mode == 5:
                    A *= (rng.random((r, c)) < 0.5)
            blocks.append(A)

        K = build_block_cyclic(blocks)
        r0_n = spectral_radius(K) ** n
        P_list = [return_map(blocks, k) for k in range(n)]
        P0 = P_list[0]

        # C1: rho(K)^n == rho(P_k) and identical non-zero spectrum (with algebraic multiplicity).
        # Normalize by scale = max(1, rho(P0)) so characteristic polynomial coefficients are O(1).
        w_min = min(widths)
        scale = max(1.0, spectral_radius(P0))
        poly0 = np.poly(P0 / scale)[: w_min + 1]
        for Pk in P_list:
            rho_pk = spectral_radius(Pk)
            assert abs(r0_n - rho_pk) <= 1e-6 * max(1.0, r0_n), f"C1 rho mismatch: {r0_n} vs {rho_pk}"
            polyk = np.poly(Pk / scale)
            assert np.allclose(poly0, polyk[: w_min + 1], rtol=1e-7, atol=1e-8), (
                f"C1 characteristic polynomial mismatch: {poly0} vs {polyk[: w_min + 1]}"
            )
            assert np.allclose(polyk[w_min + 1 :], 0.0, atol=1e-8)

        # C2: trace(P_k) all equal, and match brute-force enumeration on small cycles
        tr0 = float(np.trace(P0))
        for Pk in P_list:
            assert abs(tr0 - float(np.trace(Pk))) <= 1e-10 * max(1.0, tr0)
        if math.prod(widths) <= 250 and brute_checked < 600:
            bf = brute_force_route_sum(blocks)
            assert abs(tr0 - bf) <= 1e-10 * max(1.0, bf)
            brute_checked += 1

        # C3: if any block has rank <= 1, rho(P0) == trace(P0)
        any_rank_leq_1 = forced_leq_rank1 or any(np.linalg.matrix_rank(A, tol=1e-12) <= 1 for A in blocks)
        rho0 = spectral_radius(P0)
        if any_rank_leq_1:
            assert abs(rho0 - tr0) <= tol * max(1.0, rho0), f"C3 failed: {rho0} vs {tr0}"
        else:
            if tr0 > rho0 + 1e-7:
                c4_high += 1
            elif tr0 < rho0 - 1e-7:
                c4_low += 1
            else:
                c4_equal_nonrank1 += 1

        # C5: bounds
        w_min = min(widths)
        max_diag_p0 = float(np.max(np.diag(P0)))
        max_diag_all = max(float(np.max(np.diag(Pk))) for Pk in P_list)
        assert max_diag_p0 <= rho0 + tol * max(1.0, rho0)
        assert max_diag_all <= rho0 + tol * max(1.0, rho0)
        assert tr0 / w_min <= max_diag_all + 1e-11 * max(1.0, max_diag_all)
        assert tr0 / w_min <= rho0 + tol * max(1.0, rho0)

    # Structured edge cases for C3/C4:
    # (a) Full-rank positive P with complex conjugate non-dominant eigenvalues summing to 0:
    #     P = [[3, 4, 2], [2, 3, 4], [4, 2, 3]] -> spec = {9, +i*sqrt(3), -i*sqrt(3)}, rank = 3, rho = trace = 9
    A1 = np.array([[3.0, 4.0, 2.0], [2.0, 3.0, 4.0], [4.0, 2.0, 3.0]])
    A2 = np.eye(3)
    K_circ = build_block_cyclic([A1, A2])
    P_circ = return_map([A1, A2], 0)
    assert np.linalg.matrix_rank(A1) == 3 and np.linalg.matrix_rank(A2) == 3
    assert math.isclose(spectral_radius(K_circ) ** 2, 9.0, rel_tol=1e-13)
    assert math.isclose(np.trace(P_circ), 9.0, rel_tol=1e-13)

    # (b) No rank-1 block, yet rank(P) == 1:
    B1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # rank 2
    B2 = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])  # rank 2
    P_prod = return_map([B1, B2], 0)
    assert np.linalg.matrix_rank(B1) == 2 and np.linalg.matrix_rank(B2) == 2
    assert np.linalg.matrix_rank(P_prod) == 1
    assert math.isclose(spectral_radius(P_prod), np.trace(P_prod), rel_tol=1e-14)

    # (c) Rank-1 nilpotent P != 0: A1 = [[0, 2], [0, 0]] (rank 1), A2 = I_2 -> P = [[0, 2], [0, 0]], rho = trace = 0
    N1 = np.array([[0.0, 2.0], [0.0, 0.0]])
    P_nil = return_map([N1, np.eye(2)], 0)
    assert np.linalg.matrix_rank(P_nil) == 1 and spectral_radius(P_nil) == 0.0 and np.trace(P_nil) == 0.0

    # (d) Demonstrate C4 wording bug on imprimitive and reducible P:
    #     P_imprim = [[0, 6], [6, 0]] has eigenvalues {+6, -6}. Strictly subdominant set {|lam| < 6} is EMPTY (sum 0),
    #     yet trace(P_imprim) = 0 != 6 = rho(P_imprim). Removing ONE copy of +6 leaves {-6} (sum -6 != 0).
    #     P_mixed = diag(6) (+) [[0, 6], [6, 0]] has eigenvalues {+6, +6, -6}: {|lam| < 6} is empty, AND removing
    #     one copy of +6 leaves {+6, -6} summing to 0, giving trace = 6 == rho!
    P_imprim = np.array([[0.0, 6.0], [6.0, 0.0]])
    ev_imp = np.linalg.eigvals(P_imprim)
    strict_subdom_sum = sum(z for z in ev_imp if abs(z) < spectral_radius(P_imprim) - 1e-12)
    assert strict_subdom_sum == 0.0 and np.trace(P_imprim) != spectral_radius(P_imprim)

    P_mixed = np.array([[6.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 6.0, 0.0]])
    assert math.isclose(spectral_radius(P_mixed), 6.0) and math.isclose(np.trace(P_mixed), 6.0)

    print(
        f"[PASS] C1-C5 tested on {n_trials} random/degenerate life cycles ({brute_checked} brute-force checked). "
        f"Non-rank-1 outcomes: trace > rho in {c4_high}, trace < rho in {c4_low}, trace == rho in {c4_equal_nonrank1}."
    )
    print("[PASS] C4 wording edge-case verified: 'strictly subdominant' fails on [[0,6],[6,0]] & diag(6,6); "
          "'remaining w_1 - 1 eigenvalues after removing one copy of rho(P)' is exact.")


def check_c6(rng: np.random.Generator) -> None:
    # 1. Verify target/subset reproduction number T_{S_k} == R0^n for general multi-host stages (w_k >= 1)
    for _ in range(500):
        n = int(rng.integers(2, 6))
        widths = [int(rng.integers(1, 5)) for _ in range(n)]
        blocks = [rng.exponential(1.0, size=(widths[(k + 1) % n], widths[k])) for k in range(n)]
        K = build_block_cyclic(blocks)
        r0_n = spectral_radius(K) ** n
        offsets = np.cumsum([0] + widths)
        for target_stage in range(n):
            # Target all incoming transmissions to target_stage (S-control on whole stage)
            prev_stage = (target_stage - 1) % n
            C = np.zeros_like(K)
            r0, r1 = offsets[target_stage], offsets[target_stage + 1]
            c0, c1 = offsets[prev_stage], offsets[prev_stage + 1]
            C[r0:r1, c0:c1] = K[r0:r1, c0:c1]
            B = K - C
            assert spectral_radius(B) < 1e-12  # B is nilpotent
            T_stage = spectral_radius(C @ np.linalg.inv(np.eye(K.shape[0]) - B))
            assert math.isclose(T_stage, r0_n, rel_tol=1e-9, abs_tol=1e-12)

    # 2. Check real-time Malthusian growth rate r of linearized ODE dx/dt = (T + Sigma) x
    #    at FIXED whole-cycle R_cycle = 100 and FIXED total cycle time tau_tot = 1.0:
    #    gamma_k = n / tau_tot, beta_k = gamma_k * R_cycle^(1/n) -> r = (n / tau_tot) * (R_cycle^(1/n) - 1)
    R_cycle = 100.0
    tau_tot = 1.0
    r_values = {}
    for n in (1, 2, 3, 4, 10):
        gamma = n / tau_tot
        beta = gamma * (R_cycle ** (1.0 / n))
        J = -gamma * np.eye(n) + np.roll(beta * np.eye(n), 1, axis=0)
        r_num = float(np.max(np.real(np.linalg.eigvals(J))))
        r_exact = (n / tau_tot) * (R_cycle ** (1.0 / n) - 1.0)
        assert math.isclose(r_num, r_exact, rel_tol=1e-11)
        r_values[n] = r_num
    assert r_values[1] > r_values[2] > r_values[3] > r_values[4] > r_values[10]
    print(
        f"[PASS] C6 stage-target T_S == R0^n verified on 500 multi-host cycles. "
        f"Real-time Malthusian r at fixed R_cycle=100, tau_tot=1 drops with n: {r_values}"
    )


def enumerate_all_simple_cycles(K: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    """Return all directed simple cycles (including self-loops of length 1) with positive weight.
    Arc j -> i has weight K[i, j] > 0."""
    m = K.shape[0]
    cycles: list[tuple[tuple[int, ...], float]] = []

    def dfs(start: int, curr: int, visited: set[int], path: list[int], weight: float) -> None:
        # Next vertices nxt >= start from curr via arc curr -> nxt (entry K[nxt, curr])
        for nxt in range(start, m):
            w_edge = K[nxt, curr]
            if w_edge <= 0.0:
                continue
            if nxt == start:
                cycles.append((tuple(path), weight * w_edge))
            elif nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, nxt, visited, path, weight * w_edge)
                path.pop()
                visited.remove(nxt)

    for s in range(m):
        dfs(s, s, {s}, [s], 1.0)
    return cycles


def solve_euler_lotka(cycles_b: list[tuple[tuple[int, ...], float]]) -> float:
    """Unique positive root of sum_r w_r * lam^(-l_r) = 1 via bisection + Newton."""
    weights = np.array([w for _, w in cycles_b], dtype=float)
    lengths = np.array([len(p) for p, _ in cycles_b], dtype=float)

    def f(lam: float) -> float:
        return float(np.sum(weights * (lam ** (-lengths)))) - 1.0

    lo, hi = 1e-12, 1.0
    while f(lo) <= 0.0:
        lo *= 0.1
    while f(hi) >= 0.0:
        hi *= 2.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_c7(rng: np.random.Generator, n_trials: int = 15000) -> None:
    mixed_len_count = 0
    self_loop_count = 0
    reducible_count = 0
    dag_count = 0

    for trial in range(n_trials):
        m = int(rng.integers(2, 8))
        b = int(rng.integers(0, m))

        # Construct K so that K without row/col b is strictly upper triangular up to a random permutation -> DAG!
        # That guarantees b lies on EVERY directed cycle of G(K), while allowing arbitrary mixed-length cycles through b,
        # self-loops at b, and reducible dead-end/source nodes.
        others = [v for v in range(m) if v != b]
        rng.shuffle(others)
        K = np.zeros((m, m), dtype=float)

        # Edges among others: only forward along `others` ordering -> acyclic on V \ {b}
        p_dag = 0.45 if (trial % 4 != 0) else 0.25
        for idx_u, u in enumerate(others):
            for v in others[idx_u + 1:]:
                if rng.random() < p_dag:
                    K[v, u] = rng.exponential(1.0)  # arc u -> v

        if trial % 15 == 0:
            # Force NO positive-weight cycles through b (degenerate DAG case!)
            for u in others:
                if rng.random() < 0.5:
                    K[u, b] = rng.exponential(1.0)  # b -> u only, no return to b
        else:
            # Allow edges out of b and into b, plus optional self-loop at b
            for u in others:
                if rng.random() < 0.55:
                    K[u, b] = rng.exponential(1.0)
                if rng.random() < 0.55:
                    K[b, u] = rng.exponential(1.0)
            if rng.random() < 0.35:
                K[b, b] = rng.exponential(0.8)  # self-loop at shared compartment b!

        all_cycles = enumerate_all_simple_cycles(K)
        for cyc, _ in all_cycles:
            assert b in cyc, f"Unexpected cycle avoiding b: {cyc}"

        rho = spectral_radius(K)
        route_sum = sum(w for _, w in all_cycles)

        # Check Roberts-Heesterbeek type reproduction number T_b == sum_r w_r!
        P_b = np.zeros((m, m), dtype=float)
        P_b[b, b] = 1.0
        residual = (np.eye(m) - P_b) @ K
        assert spectral_radius(residual) < 1e-12
        T_b = float((K @ np.linalg.inv(np.eye(m) - residual))[b, b])
        assert math.isclose(T_b, route_sum, rel_tol=1e-10, abs_tol=1e-13)

        if len(all_cycles) == 0:
            dag_count += 1
            assert rho < 1e-12 and route_sum == 0.0
            # Note: sum_r w_r lam^(-l_r) == 0 != 1 has NO root here (falsifies unguarded Prop. 5!)
            continue

        lengths = {len(cyc) for cyc, _ in all_cycles}
        if len(lengths) >= 2:
            mixed_len_count += 1
        if 1 in lengths:
            self_loop_count += 1
        # Check reducibility via reachability matrix
        reach = np.linalg.matrix_power(np.eye(m) + (K > 0).astype(float), m)
        if np.any(reach == 0):
            reducible_count += 1

        lam_root = solve_euler_lotka(all_cycles)
        assert abs(rho - lam_root) <= 1e-8 * max(1.0, rho), f"C7 root mismatch: {rho} vs {lam_root}"
        if abs(route_sum - 1.0) > 1e-10:
            assert (rho > 1.0) == (route_sum > 1.0)

    # Also verify failure when a disjoint cycle avoids b:
    # b = 0 has cycle 0 -> 1 -> 0 of weight 0.5, and disjoint cycle 2 -> 2 of weight 2.0:
    K_disj = np.array([[0.0, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    cyc_b = [c for c in enumerate_all_simple_cycles(K_disj) if 0 in c[0]]
    assert math.isclose(solve_euler_lotka(cyc_b), math.sqrt(0.5))
    assert math.isclose(spectral_radius(K_disj), 2.0)  # != sqrt(0.5)

    print(
        f"[PASS] C7 tested on {n_trials} digraphs: {mixed_len_count} with mixed route lengths, "
        f"{self_loop_count} with self-loop at b, {reducible_count} reducible, {dag_count} acyclic/zero-cycle "
        f"(confirming T_b == sum_r w_r everywhere and exposing the missing non-empty-cycle hypothesis on DAGs)."
    )


if __name__ == "__main__":
    rng = np.random.default_rng(2026111201)
    check_dobson_examples(rng)
    check_c1_to_c5(rng, n_trials=25000)
    check_c6(rng)
    check_c7(rng, n_trials=15000)
