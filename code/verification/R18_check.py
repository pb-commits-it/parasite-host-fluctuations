"""Independent check of Run R18: parasites in hosts, packets, and host outages (task r21).

Run with:
    python3 -I verification/R18_check.py

Rules observed:
  - Does NOT import or call r18_packets_and_outages.py or r18_analyze.py.
  - Does NOT read anything under sources/; uses no network.
  - Pre-declares all tolerances at the top before running (Rule 7).
  - Step 0 reproduces known analytical results (mean-matrix eigenstructure,
    one-step mean/variance, exact 11-type mixed-mortality PGF at a=0.5,
    lognormal AR(1) second-order loss D2, and single-gap cohort Monte Carlo)
    before checking any R18 result.
  - Item 1: quantifies structural & scaling quantities across L in {10, 40, 160}.
  - Item 2: independently calibrates continuous growth margins and recomputes
    21 whole-number cells (17 primary factorial cells spanning all a, K, L,
    environments, b0, and every headline cell + 4 supplemental cells), each
    with 2,000-3,000 replicates, checking agreement within 4 combined SE.
  - Item 3: recounts every contrast from r18_cells.json and r18_comparisons.json
    with full denominators and boundary-SE sensitivity checks.
  - Item 4: verifies the gap-survival formula and recomputes the 5%, 50%, 95%
    retention thresholds (including 7.58 and 3.48) plus discrete analogues.
  - Item 5: verifies the cross-law D-ordering reversal (0.004196 vs 0.002144),
    how D was computed, and all 4,293 within-environment pairwise counts.
"""
from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy import optimize, stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '../secondary').resolve()
R18_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '../studies/R18-packets-outages').resolve()
OUTPUT_PATH = HERE / "R18_check_output.txt"

# ---------------------------------------------------------------------------
# Pre-declared tolerances (fixed before running, Rule 7)
# ---------------------------------------------------------------------------
TOL_ALGEBRA_RTOL = 1.0e-10
TOL_GAP_RATIO_ATOL = 1.0e-9
TOL_KNOWN_SIM_SE = 4.0
TOL_CAL_COMBINED_SE = 4.0
TOL_CELL_COMBINED_SE = 4.0

# Independent simulation budgets
HORIZON_T = 1700
LARGE_SWITCH = 1.0e9
CAL_PAIRS = 96          # 1.5x R18's 64 antithetic pairs
CAL_STEPS = 80000
CAL_BURN = 2000
CELL_REPS_BASE = 2000   # >= 1,000 required; 2,000 standard, 3,000 for near-0 cells
CELL_REPS_NEAR_ZERO = 3000

# 17 primary factorial cells + 4 supplemental cells = 21 cells
# Covers all a in {0, 0.5, 1}, K in {1, 10, 100}, L in {10, 40, 160},
# env in {'A', 'B20', 'B200'}, b0 in {1, 100}, and every headline number.
PRIMARY_SAMPLE_SPECS = [
    # Headline B200 L10 -> L40 -> L160 at a=1, K=100, b0=100 (0.753, 0.637, 0.196)
    ("B200", 10,  1.0, 100, 100, False, CELL_REPS_BASE,      "headline B200 stay 10 (0.753)"),
    ("B200", 40,  1.0, 100, 100, False, CELL_REPS_BASE,      "headline B200 stay 40 (0.637)"),
    ("B200", 160, 1.0, 100, 100, False, CELL_REPS_BASE,      "headline B200 stay 160 (0.196)"),
    # Headline B200 stay 160 across shares a=0, 0.5, 1 (0.029, 0.061, 0.196)
    ("B200", 160, 0.0, 100, 100, False, CELL_REPS_BASE,      "headline B200 stay 160 a=0 (0.029)"),
    ("B200", 160, 0.5, 100, 100, False, CELL_REPS_BASE,      "headline B200 stay 160 a=0.5 (0.061)"),
    # Headline start-burden comparator in B200 stay 160 (0.121 vs 0.196) & singleton packet (0.029)
    ("B200", 160, 1.0, 100, 1,   False, CELL_REPS_BASE,      "headline B200 stay 160 b0=1 (0.121)"),
    ("B200", 160, 1.0, 1,   100, False, CELL_REPS_BASE,      "headline B200 stay 160 K=1 (0.029)"),
    # Midpoint B200 cell (a=0.5, K=10)
    ("B200", 40,  0.5, 10,  100, False, CELL_REPS_BASE,      "midpoint B200 L40 a=0.5 K=10"),
    # Headline B20 stay 40 packet contrast K=1 vs K=100 (0.055 vs 0.380) + spread across L=10, 160
    ("B20",  40,  1.0, 1,   100, False, CELL_REPS_BASE,      "headline B20 L40 K=1 (0.055)"),
    ("B20",  40,  1.0, 100, 100, False, CELL_REPS_BASE,      "headline B20 L40 K=100 (0.380)"),
    ("B20",  10,  0.0, 10,  1,   False, CELL_REPS_BASE,      "B20 L10 a=0 K=10 b0=1"),
    ("B20",  160, 1.0, 100, 100, False, CELL_REPS_NEAR_ZERO, "B20 L160 a=1 K=100 b0=100 (0.0077)"),
    # Headline Item 5 D-ordering pair in A (0.0000 at D=0.004196 vs 0.0490 at D=0.002144) + spread in A
    ("A",    10,  0.0, 1,   1,   False, CELL_REPS_NEAR_ZERO, "Item 5 A L10 a=0 K=1 b0=1 (0.000)"),
    ("A",    40,  1.0, 100, 1,   False, CELL_REPS_BASE,      "Item 5 A L40 a=1 K=100 b0=1 (0.049)"),
    ("A",    10,  0.0, 100, 100, False, CELL_REPS_BASE,      "A L10 a=0 K=100 b0=100 (0.162)"),
    ("A",    10,  0.5, 100, 100, False, CELL_REPS_BASE,      "A L10 a=0.5 K=100 b0=100 (0.186)"),
    ("A",    160, 1.0, 100, 100, False, CELL_REPS_NEAR_ZERO, "A L160 a=1 K=100 b0=100 (0.000)"),
    # Supplemental clumped 1/19 cell
    ("B200", 40,  0.0, 10,  100, True,  CELL_REPS_BASE,      "supplement clumped 1/19 B200 L40"),
]


# ---------------------------------------------------------------------------
# Independent stochastic environment, calibration, and host-burden simulator
# ---------------------------------------------------------------------------

def step_env(rng: np.random.Generator, state: np.ndarray, env: str) -> tuple[np.ndarray, np.ndarray]:
    if env == "A":
        nxt = 0.9 * state + math.sqrt(0.19) * rng.standard_normal(state.shape)
        return nxt, np.exp(0.15 * nxt - 0.5 * 0.15 * 0.15)
    if env == "constant":
        return state, np.ones_like(state, dtype=float)
    gap_mean = int(env[1:])
    flips = rng.random(state.shape) < (1.0 / gap_mean)
    nxt = np.where(flips, 1.0 - state, state)
    return nxt, 2.0 * nxt


def init_env(rng: np.random.Generator, n: int, env: str) -> np.ndarray:
    if env == "A":
        return rng.standard_normal(n)
    if env == "constant":
        return np.ones(n, dtype=float)
    return rng.integers(0, 2, size=n).astype(float)


def sample_binom_hybrid(rng: np.random.Generator, n: np.ndarray, p: float | np.ndarray, switch: float = LARGE_SWITCH) -> np.ndarray:
    n_arr = np.asarray(n, dtype=float)
    p_arr = np.broadcast_to(p, n_arr.shape)
    out = np.zeros_like(n_arr, dtype=float)
    pos = n_arr > 0
    if not np.any(pos):
        return out
    small = pos & (n_arr < switch)
    big = pos & (n_arr >= switch)
    if np.any(small):
        out[small] = rng.binomial(n_arr[small].astype(np.int64), p_arr[small])
    if np.any(big):
        mu = n_arr[big] * p_arr[big]
        sd = np.sqrt(mu * (1.0 - p_arr[big]))
        out[big] = np.clip(np.rint(mu + sd * rng.standard_normal(mu.shape)), 0.0, n_arr[big])
    return out


def sample_poisson_hybrid(rng: np.random.Generator, lam: np.ndarray, switch: float = LARGE_SWITCH) -> np.ndarray:
    lam_arr = np.asarray(lam, dtype=float)
    out = np.zeros_like(lam_arr, dtype=float)
    pos = lam_arr > 0
    if not np.any(pos):
        return out
    small = pos & (lam_arr < switch)
    big = pos & (lam_arr >= switch)
    if np.any(small):
        out[small] = rng.poisson(lam_arr[small])
    if np.any(big):
        out[big] = np.maximum(0.0, np.rint(lam_arr[big] + np.sqrt(lam_arr[big]) * rng.standard_normal(np.sum(big))))
    return out


def sample_multinom_hybrid(rng: np.random.Generator, counts: np.ndarray, probs: np.ndarray, switch: float = LARGE_SWITCH) -> np.ndarray:
    c_arr = np.asarray(counts, dtype=float)
    k_cat = len(probs)
    out = np.zeros((len(c_arr), k_cat), dtype=float)
    small = (c_arr > 0) & (c_arr < switch)
    big = c_arr >= switch
    if np.any(small):
        out[small] = rng.multinomial(c_arr[small].astype(np.int64), probs)
    if np.any(big):
        n_b = c_arr[big, None]
        mu = n_b * probs[None, :]
        z = np.sqrt(mu) * rng.standard_normal(mu.shape)
        draw = np.maximum(0.0, np.rint(mu + z - z.sum(axis=1, keepdims=True) * probs[None, :]))
        top = int(np.argmax(probs))
        fix = c_arr[big] - draw.sum(axis=1)
        if np.any(np.abs(fix) > np.maximum(1.0e-6 * c_arr[big], float(k_cat))):
            raise RuntimeError("Multinomial Gaussian rounding repair exceeded tolerance")
        draw[:, top] += fix
        if np.any(draw < 0):
            raise RuntimeError("Negative multinomial category after rounding repair")
        out[big] = draw
    return out


class IndependentHostBurdenModel:
    """Clean from-scratch implementation of the R18 host-burden packet branching law."""

    def __init__(self, L: int, a: float, k: int, b0: int, clump: bool = False, switch: float = LARGE_SWITCH):
        self.L = int(L)
        self.a = float(a)
        self.k = int(k)
        self.b0 = int(b0)
        self.clump = bool(clump)
        self.switch = float(switch)
        self.max_b = max(self.b0, 19 if self.clump else self.k)
        self.b_vec = np.arange(self.max_b + 1, dtype=float)
        s = 1.0 - 1.0 / self.L
        self.s_host = s ** self.a
        self.s_ind = s ** (1.0 - self.a)
        self.binom_pmf = [None]
        for b in range(1, self.max_b + 1):
            p = stats.binom.pmf(np.arange(b + 1), b, self.s_ind)
            p /= p.sum()
            self.binom_pmf.append(p)

    def one_step(
        self,
        rng: np.random.Generator,
        F: np.ndarray,
        C: np.ndarray,
        H: np.ndarray,
        gf: float,
        gs: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        S_tot = C @ self.b_vec
        # 1. Independent Poisson shedding from current steady-stage parasites before mortality
        F_shed = sample_poisson_hybrid(rng, gs * S_tot, self.switch)

        # 2. Steady-host and individual-parasite mortality
        if self.a == 1.0:
            # No within-host individual death (s_ind = 1); whole hosts survive with prob s_host
            C_next = sample_binom_hybrid(rng, C, self.s_host, self.switch)
            C_next[:, 0] = 0.0
        else:
            C_next = np.zeros_like(C)
            # Burden-1 hosts survive iff both individual and host clocks survive (prob s_ind * s_host = 1 - 1/L)
            C_next[:, 1] = sample_binom_hybrid(rng, C[:, 1], self.s_ind * self.s_host, self.switch)
            if self.max_b >= 2:
                multi_total = int(C[:, 2:].sum())
                if 0 < multi_total <= 2_000_000:
                    flat = C[:, 2:].ravel().astype(np.int64)
                    nz = np.flatnonzero(flat)
                    expanded = np.repeat(nz, flat[nz])
                    rep_idx = expanded // (self.max_b - 1)
                    b_vals = (expanded % (self.max_b - 1)) + 2
                    rem = rng.binomial(b_vals, self.s_ind)
                    if self.a > 0.0:
                        rem = np.where(rng.random(len(rem)) < self.s_host, rem, 0)
                    alive = rem > 0
                    if np.any(alive):
                        bins = rep_idx[alive] * (self.max_b + 1) + rem[alive]
                        C_next += np.bincount(bins, minlength=C.size).reshape(C.shape)
                elif multi_total > 0:
                    after_ind = np.zeros_like(C)
                    active_bs = np.flatnonzero(np.any(C[:, 2:] > 0, axis=0)) + 2
                    for b in active_bs:
                        rows = np.flatnonzero(C[:, b] > 0)
                        after_ind[rows, : b + 1] += sample_multinom_hybrid(
                            rng, C[rows, b], self.binom_pmf[b], self.switch
                        )
                    after_ind[:, 0] = 0.0
                    if self.a == 0.0:
                        C_next[:, 2:] += after_ind[:, 2:]
                        C_next[:, 1] += after_ind[:, 1]
                    else:
                        thinned = sample_binom_hybrid(rng, after_ind[:, 1:], self.s_host, self.switch)
                        C_next[:, 1:] += thinned

        # 3. Fluctuating-stage departures and new-host packet arrivals
        F_leave = sample_binom_hybrid(rng, F, 0.1, self.switch)
        F_next = F - F_leave + F_shed
        pkt_mean = F_leave * (gf / 0.1) * H / float(self.k)
        if self.clump:
            C_next[:, 1] += sample_poisson_hybrid(rng, 0.5 * pkt_mean, self.switch)
            C_next[:, 19] += sample_poisson_hybrid(rng, 0.5 * pkt_mean, self.switch)
        else:
            C_next[:, self.k] += sample_poisson_hybrid(rng, pkt_mean, self.switch)
        return F_next, C_next


def run_host_simulation(
    L: int,
    a: float,
    k: int,
    b0: int,
    env: str,
    gf: float,
    gs: float,
    reps: int,
    seed: int,
    steps: int = HORIZON_T,
    start_parasites: int = 10000,
    clump: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    model = IndependentHostBurdenModel(L=L, a=a, k=k, b0=b0, clump=clump)
    F = np.zeros(reps, dtype=float)
    C = np.zeros((reps, model.max_b + 1), dtype=float)
    C[:, b0] = float(start_parasites // b0)
    state = init_env(rng, reps, env)

    alive_ids = np.arange(reps)
    ext_step = np.full(reps, -1, dtype=np.int32)
    min_hosts = C.sum(axis=1).copy()
    approx_used = np.zeros(reps, dtype=bool)

    for t in range(1, steps + 1):
        if alive_ids.size == 0:
            break
        state, H = step_env(rng, state, env)
        hit_big = (F >= LARGE_SWITCH) | (C.max(axis=1) >= LARGE_SWITCH)
        approx_used[alive_ids[hit_big]] = True
        F, C = model.one_step(rng, F, C, H, gf, gs)
        occ = C.sum(axis=1)
        min_hosts[alive_ids] = np.minimum(min_hosts[alive_ids], occ)
        dead = (F == 0.0) & (occ == 0.0)
        if np.any(dead):
            ext_step[alive_ids[dead]] = t
            keep = ~dead
            alive_ids = alive_ids[keep]
            F = F[keep]
            C = C[keep]
            state = state[keep]

    ext_mask = ext_step > 0
    t_gen10 = 10 * (10 + L)
    gen_mask = (ext_step > 0) & (ext_step <= t_gen10)
    p_cal = float(ext_mask.mean())
    p_gen = float(gen_mask.mean())
    se_cal = math.sqrt(p_cal * (1.0 - p_cal) / reps)
    se_gen = math.sqrt(p_gen * (1.0 - p_gen) / reps)
    surv_med_min = float(np.median(min_hosts[~ext_mask])) if np.any(~ext_mask) else None
    return {
        "n": reps,
        "extinct": int(ext_mask.sum()),
        "p": p_cal,
        "se": se_cal,
        "p_gen": p_gen,
        "se_gen": se_gen,
        "survivor_min_median": surv_med_min,
        "approx_runs": int(approx_used.sum()),
    }


def eval_continuous_growth_antithetic(
    env: str,
    r_vec: np.ndarray,
    seed: int,
    n_pairs: int = CAL_PAIRS,
    steps: int = CAL_STEPS,
    burn: int = CAL_BURN,
) -> np.ndarray:
    """Evaluate per-step Lyapunov exponent for L in [10, 40, 160] using antithetic pairs."""
    rng = np.random.default_rng(seed)
    Ls = np.array([10.0, 40.0, 160.0])
    qs = 1.0 / Ls
    r_arr = np.asarray(r_vec, dtype=float)  # shape (3,)
    gf = (0.1 + r_arr)[:, None]
    gs = (qs + r_arr)[:, None]
    stay_s = (1.0 - qs)[:, None]

    x = np.full((3, 2 * n_pairs), 0.5)
    y = np.full((3, 2 * n_pairs), 0.5)
    acc = np.zeros((3, 2 * n_pairs))
    state = init_env(rng, n_pairs, env)
    for t in range(steps + burn):
        state, h1 = step_env(rng, state, env)
        h2 = np.exp(-0.15 * state - 0.5 * 0.15 * 0.15) if env == "A" else (2.0 - h1)
        h = np.concatenate([h1, h2])[None, :]
        nx = 0.9 * x + gs * y
        ny = gf * h * x + stay_s * y
        tot = nx + ny
        x = nx / tot
        y = ny / tot
        if t >= burn:
            acc += np.log(tot)
    per_path = acc / float(steps)
    return 0.5 * (per_path[:, :n_pairs] + per_path[:, n_pairs:])


def calibrate_env_task(args: tuple[str, int]) -> list[dict]:
    env, seed = args
    Ls = [10, 40, 160]
    t_gens = np.array([20.0, 50.0, 170.0])
    target = 0.005 / t_gens
    lo = np.zeros(3)
    hi = np.full(3, 0.15)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        vals = eval_continuous_growth_antithetic(env, mid, seed).mean(axis=1)
        below = vals < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    r_hat = 0.5 * (lo + hi)
    # Independent validation seed
    val_pairs = eval_continuous_growth_antithetic(env, r_hat, seed + 500)
    lam_mean = val_pairs.mean(axis=1)
    lam_se = val_pairs.std(axis=1, ddof=1) / math.sqrt(val_pairs.shape[1])
    out = []
    for j, L in enumerate(Ls):
        r_j = float(r_hat[j])
        tg = float(t_gens[j])
        m_j = float(lam_mean[j] * tg)
        se_j = float(lam_se[j] * tg)
        D_j = float(tg * (math.log1p(r_j) - lam_mean[j]))
        R0_j = float((1.0 + 10.0 * r_j) * (1.0 + L * r_j))
        out.append({
            "env": env,
            "L": L,
            "r": r_j,
            "gf": 0.1 + r_j,
            "gs": 1.0 / L + r_j,
            "R0": R0_j,
            "ln_R0": math.log(R0_j),
            "tg_ln1pr": tg * math.log1p(r_j),
            "D": D_j,
            "D_se": se_j,
            "margin": m_j,
            "margin_se": se_j,
        })
    return out


def calibrate_and_sim_chain4(seed_cal: int = 77100, seed_sim: int = 77200) -> list[dict]:
    """Independent calibration and simulation of the 3 four-phase B200 supplement cells."""
    Ls = np.array([10.0, 40.0, 160.0])
    t_gens = Ls + 10.0
    target = 0.005 / t_gens
    n_pairs = 96

    def eval_chain_growth(r_vec: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q4 = (4.0 / Ls)[:, None]
        gf = (0.1 + r_vec)[:, None]
        gs = (r_vec / (1.0 - (q4[:, 0] / (q4[:, 0] + r_vec)) ** 4))[:, None]
        F = np.full((3, 2 * n_pairs), 0.2)
        S = np.full((3, 2 * n_pairs, 4), 0.2)
        acc = np.zeros((3, 2 * n_pairs))
        state = init_env(rng, n_pairs, "B200")
        for t in range(CAL_STEPS + CAL_BURN):
            state, h1 = step_env(rng, state, "B200")
            h = np.concatenate([h1, 2.0 - h1])[None, :]
            Fn = 0.9 * F + gs * S.sum(axis=2)
            Sn = (1.0 - q4[:, :, None]) * S
            Sn[:, :, 1:] += q4[:, :, None] * S[:, :, :3]
            Sn[:, :, 0] += gf * h * F
            tot = Fn + Sn.sum(axis=2)
            F = Fn / tot
            S = Sn / tot[:, :, None]
            if t >= CAL_BURN:
                acc += np.log(tot)
        per = acc / float(CAL_STEPS)
        return 0.5 * (per[:, :n_pairs] + per[:, n_pairs:])

    lo = np.zeros(3)
    hi = np.full(3, 0.15)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        vals = eval_chain_growth(mid, seed_cal).mean(axis=1)
        below = vals < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    r_hat = 0.5 * (lo + hi)
    val_p = eval_chain_growth(r_hat, seed_cal + 500)
    lam_m = val_p.mean(axis=1)
    lam_s = val_p.std(axis=1, ddof=1) / math.sqrt(n_pairs)

    out = []
    for j, L_int in enumerate([10, 40, 160]):
        r_j = float(r_hat[j])
        q4 = 4.0 / L_int
        gf = 0.1 + r_j
        gs = r_j / (1.0 - (q4 / (q4 + r_j)) ** 4)
        tg = float(L_int + 10)
        D_j = float(tg * (math.log1p(r_j) - lam_m[j]))
        se_j = float(tg * lam_s[j])
        m_j = float(tg * lam_m[j])

        # Simulate 2,000 whole-number replicates of 4-phase pure shared death, K=100, b0=100
        reps = CELL_REPS_BASE
        rng = np.random.default_rng(seed_sim + j)
        F = np.zeros(reps, dtype=float)
        C = np.zeros((reps, 4), dtype=float)
        C[:, 0] = 100.0
        state = init_env(rng, reps, "B200")
        alive = np.arange(reps)
        ext_t = np.full(reps, -1, dtype=np.int32)
        for t in range(1, HORIZON_T + 1):
            if alive.size == 0:
                break
            state, H = step_env(rng, state, "B200")
            shed = sample_poisson_hybrid(rng, 100.0 * C.sum(axis=1) * gs)
            adv = sample_binom_hybrid(rng, C, q4)
            Cn = C - adv
            Cn[:, 1:] += adv[:, :3]
            dep = sample_binom_hybrid(rng, F, 0.1)
            Cn[:, 0] += sample_poisson_hybrid(rng, dep * (gf / 0.1) * H / 100.0)
            F = F - dep + shed
            C = Cn
            dead = (F == 0.0) & (C.sum(axis=1) == 0.0)
            if np.any(dead):
                ext_t[alive[dead]] = t
                keep = ~dead
                alive = alive[keep]
                F = F[keep]
                C = C[keep]
                state = state[keep]
        p_ext = float((ext_t > 0).mean())
        se_ext = math.sqrt(p_ext * (1.0 - p_ext) / reps)
        p_gen = float(((ext_t > 0) & (ext_t <= 10 * (L_int + 10))).mean())
        se_gen = math.sqrt(p_gen * (1.0 - p_gen) / reps)
        out.append({
            "key": f"chain4_B200_L{L_int}",
            "L": L_int,
            "r": r_j,
            "gf": gf,
            "gs": gs,
            "D": D_j,
            "D_se": se_j,
            "margin": m_j,
            "margin_se": se_j,
            "n": reps,
            "p": p_ext,
            "se": se_ext,
            "p_gen": p_gen,
            "se_gen": se_gen,
        })
    return out


# ---------------------------------------------------------------------------
# Independent single-gap retention functions (no scipy.special.gammaincc)
# ---------------------------------------------------------------------------

def erlang4_fresh_survival(ratio_g_over_tau: float) -> float:
    x = 4.0 * ratio_g_over_tau
    return math.exp(-x) * (1.0 + x + 0.5 * x * x + (x ** 3) / 6.0)


def erlang4_stationary_survival(ratio_g_over_tau: float) -> float:
    x = 4.0 * ratio_g_over_tau
    s1 = 1.0
    s2 = 1.0 + x
    s3 = s2 + 0.5 * x * x
    s4 = s3 + (x ** 3) / 6.0
    return 0.25 * math.exp(-x) * (s1 + s2 + s3 + s4)


def gap_retention_prob(h: float, p: float, burdens: np.ndarray) -> float:
    """Exact formula: P = 1 - prod_j [1 - h * (1 - (1 - p)^b_j)]."""
    b = np.asarray(burdens, dtype=float)
    if p >= 1.0:
        within = np.ones_like(b)
    elif p <= 0.0:
        within = np.zeros_like(b)
    else:
        within = -np.expm1(b * math.log1p(-p))
    u = h * within
    if np.any(u >= 1.0):
        return 1.0
    log_all_fail = float(np.sum(np.log1p(-u)))
    return -math.expm1(log_all_fail)


# ---------------------------------------------------------------------------
# Step 0: Reproduce known analytical results before checking R18 (Rule 7)
# ---------------------------------------------------------------------------

def run_step0_known_checks() -> list[tuple[str, bool, str]]:
    results = []

    # K0a: Mean-matrix eigenstructure (lambda = 1+r, left RV u = (1,1)) & one-step mean/variance
    max_eig_resid = 0.0
    for L in (10, 40, 160):
        for r in (0.0, 0.002, 0.04):
            A1 = np.array([[0.9, 1.0 / L + r], [0.1 + r, 1.0 - 1.0 / L]])
            u = np.array([1.0, 1.0])
            max_eig_resid = max(max_eig_resid, float(np.max(np.abs(u @ A1 - (1.0 + r) * u))))
    rng = np.random.default_rng(70001)
    n_mc = 4000
    F0, S0, b0 = 50000.0, 50000.0, 100
    L_t, a_t, k_t, H_t, gf_t, gs_t = 40, 0.5, 100, 2.0, 0.105, 0.030
    mod = IndependentHostBurdenModel(L_t, a_t, k_t, b0)
    C0 = np.zeros((n_mc, mod.max_b + 1))
    C0[:, b0] = S0 / b0
    Fn, Cn = mod.one_step(rng, np.full(n_mc, F0), C0, np.full(n_mc, H_t), gf_t, gs_t)
    Sn = Cn @ mod.b_vec
    exp_F = 0.9 * F0 + gs_t * S0
    exp_S = (1.0 - 1.0 / L_t) * S0 + gf_t * H_t * F0
    var_F = 0.09 * F0 + gs_t * S0
    c_pkt = (gf_t / 0.1) * H_t
    var_S = (S0 / b0) * (
        mod.s_host * b0 * mod.s_ind * (1.0 - mod.s_ind)
        + mod.s_host * (1.0 - mod.s_host) * (b0 * mod.s_ind) ** 2
    ) + F0 * (0.1 * k_t * c_pkt + 0.09 * c_pkt * c_pkt)
    z_F = abs(Fn.mean() - exp_F) / math.sqrt(var_F / n_mc)
    z_S = abs(Sn.mean() - exp_S) / math.sqrt(var_S / n_mc)
    ok_a = (max_eig_resid <= TOL_ALGEBRA_RTOL) and (max(z_F, z_S) <= TOL_KNOWN_SIM_SE)
    results.append(("K0a mean-matrix RV=(1,1) & one-step moments (a=0.5)", ok_a, f"eig_resid={max_eig_resid:.2e}, max_z={max(z_F, z_S):.2f}"))

    # K0b: Exact 11-type multitype PGF for mixed mortality (a=0.5, K=10, L=40, R0=1.35, T=500)
    L_p, a_p, k_p = 40, 0.5, 10
    gf_p = 0.1 * math.sqrt(1.35)
    gs_p = (1.0 / L_p) * math.sqrt(1.35)
    s_base = 1.0 - 1.0 / L_p
    sh_p = s_base ** a_p
    si_p = s_base ** (1.0 - a_p)
    trans_mat = np.zeros((k_p + 1, k_p + 1))
    for b in range(1, k_p + 1):
        trans_mat[b, : b + 1] = stats.binom.pmf(np.arange(b + 1), b, si_p)
    f_pgf = 0.0
    z_pgf = np.ones(k_p + 1)
    z_pgf[1:] = 0.0
    T_pgf = 500
    for _ in range(T_pgf):
        nf = 0.9 * f_pgf + 0.1 * math.exp((gf_p / (0.1 * k_p)) * (z_pgf[k_p] - 1.0))
        nz = np.ones(k_p + 1)
        shed_fac = np.exp(np.arange(1, k_p + 1) * gs_p * (f_pgf - 1.0))
        surv_sum = trans_mat[1:, :] @ z_pgf
        nz[1:] = shed_fac * ((1.0 - sh_p) + sh_p * surv_sum)
        f_pgf, z_pgf = nf, nz
    exact_ext_500 = float(z_pgf[k_p])
    sim_pgf = run_host_simulation(
        L=L_p, a=a_p, k=k_p, b0=k_p, env="constant", gf=gf_p, gs=gs_p,
        reps=5000, seed=70002, steps=T_pgf, start_parasites=k_p,
    )
    se_pgf = math.sqrt(exact_ext_500 * (1.0 - exact_ext_500) / 5000.0)
    z_err_pgf = abs(sim_pgf["p"] - exact_ext_500) / se_pgf
    results.append((
        "K0b exact 11-type PGF at mixed mortality (a=0.5, K=10)",
        z_err_pgf <= TOL_KNOWN_SIM_SE,
        f"sim={sim_pgf['p']:.4f}, exact={exact_ext_500:.5f} ({z_err_pgf:.2f} SE)",
    ))

    # K0c: Lognormal AR(1) ring closed-form D2 vs continuous simulator at r=0 (L=10, 40, 160)
    p_zero = eval_continuous_growth_antithetic("A", np.zeros(3), seed=70003, n_pairs=64, steps=60000, burn=2000)
    D_zero_sim = -p_zero.mean(axis=1) * np.array([20.0, 50.0, 170.0])
    theta = 0.9 / 0.1
    D2_exact = []
    for L_v in (10.0, 40.0, 160.0):
        R_prod = (theta / (theta + 10.0)) * (theta / (theta + L_v))
        D2_exact.append(0.15 ** 2 * ((1.0 + 2.0 * theta) / (2.0 * (10.0 + L_v)) - R_prod / (1.0 - R_prod)))
    max_d2_diff = float(np.max(np.abs(D_zero_sim - np.array(D2_exact))))
    results.append((
        "K0c lognormal AR(1) ring D2 closed form vs continuous sim",
        max_d2_diff < 5.0e-5,
        f"sim={np.round(D_zero_sim, 6).tolist()}, D2={np.round(D2_exact, 6).tolist()}",
    ))

    # K0d: Single-gap retention formula vs direct individual/host lifetime draws
    rng_g = np.random.default_rng(70004)
    n_coh = 20000
    # Mixed exponential clocks: a=0.5, M=100 hosts of b=100, G/tau = 10
    ratio = 10.0
    h_ex = math.exp(-0.5 * ratio)
    p_ex = math.exp(-0.5 * ratio)
    exact_g = gap_retention_prob(h_ex, p_ex, np.full(100, 100.0))
    host_alive = rng_g.exponential(scale=2.0, size=(n_coh, 100)) > ratio
    # Within surviving hosts, draw Binomial(100, p_ex)
    surv_par = rng_g.binomial(100, p_ex, size=(n_coh, 100)) * host_alive
    obs_g = float(np.any(surv_par > 0, axis=1).mean())
    se_g = math.sqrt(exact_g * (1.0 - exact_g) / n_coh)
    z_g = abs(obs_g - exact_g) / se_g
    results.append((
        "K0d single-gap retention formula vs direct lifetime draws",
        z_g <= TOL_KNOWN_SIM_SE,
        f"obs={obs_g:.5f}, exact={exact_g:.5f} ({z_g:.2f} SE)",
    ))
    return results


def sim_worker(spec_tuple: tuple) -> dict:
    env, L, a, k, b0, clump, reps, label, gf, gs, seed = spec_tuple
    key = f"{env}_L{L}_a{a:g}_k{k}_b{b0}" + ("_clump" if clump else "")
    res = run_host_simulation(L=L, a=a, k=k, b0=b0, env=env, gf=gf, gs=gs, reps=reps, seed=seed, clump=clump)
    res["key"] = key
    res["label"] = label
    return res


class Tee:
    def __init__(self, f):
        self.f = f

    def write(self, s: str):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def main() -> int:
    t_start = time.monotonic()
    with OUTPUT_PATH.open("w") as out_f:
        sys.stdout = Tee(out_f)
        print("==========================================================================")
        print("INDEPENDENT CHECK OF RUN R18: PACKETS, SHARED HOST DEATH, AND OUTAGES (r21)")
        print("Pre-declared tolerances:")
        print(f"  Exact algebra relative tol    : {TOL_ALGEBRA_RTOL:.1e}")
        print(f"  Gap threshold ratio abs tol   : {TOL_GAP_RATIO_ATOL:.1e}")
        print(f"  Step 0 known PGF/MC check     : <= {TOL_KNOWN_SIM_SE:.1f} SE")
        print(f"  Calibrated D comparison       : <= {TOL_CAL_COMBINED_SE:.1f} combined SE")
        print(f"  Simulated cell p_ext check    : <= {TOL_CELL_COMBINED_SE:.1f} combined SE")
        print(f"  Independent budget            : {CAL_PAIRS} cal pairs x {CAL_STEPS} steps; {CELL_REPS_BASE}-{CELL_REPS_NEAR_ZERO} reps/cell")
        print("==========================================================================\n")

        # Step 0
        print("--- STEP 0: REPRODUCING KNOWN ANALYTICAL RESULTS BEFORE R18 CHECKS ---")
        for name, ok, detail in run_step0_known_checks():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                raise RuntimeError(f"Step 0 check failed: {name}")
        print()

        # Load R18 saved JSON artifacts for comparison & recount
        theirs_cal = json.loads((R18_DIR / "r18_calibration.json").read_text())
        theirs_cells = json.loads((R18_DIR / "r18_cells.json").read_text())
        theirs_chain = json.loads((R18_DIR / "r18_chain_cells.json").read_text())
        theirs_comp = json.loads((R18_DIR / "r18_comparisons.json").read_text())
        cell_by_key = {r["key"]: r for r in theirs_cells}
        chain_by_key = {r["key"]: r for r in theirs_chain}
        cal_by_el = {(r["env"], r["L"]): r for rows in theirs_cal.values() for r in rows}

        # Item 1: Reviewer diagnostics on what changes with L
        print("--- ITEM 1: REVIEWER DIAGNOSTICS ON MATCHING & WHAT CHANGES WITH L ---")
        for env in ("A", "B20", "B200"):
            for L in (10, 40, 160):
                rc = cal_by_el[(env, L)]
                r_v = rc["r"]
                tg = L + 10
                lam_step = rc["margin"] / tg
                drift_1700 = 1700.0 * lam_step
                n_gens_1700 = 1700.0 / tg
                RF = 1.0 + 10.0 * r_v
                RS = 1.0 + L * r_v
                ln_R0 = math.log(rc["R0"])
                tg_ln1pr = tg * math.log1p(r_v)
                vS_share = (0.1 + r_v) / (0.1 + 1.0 / L + 2.0 * r_v)
                max_ap = max(c["approx_runs"] for c in theirs_cells if c["env"] == env and c["L"] == L and c["kind"] == "primary")
                print(
                    f"  {env:<4} L={L:<3}: r={r_v:.6f}, RF={RF:.4f}, RS={RS:.4f}, "
                    f"(L+10)ln(1+r)={tg_ln1pr:.5f} vs lnR0={ln_R0:.5f}, "
                    f"gens@1700={n_gens_1700:4.1f}, drift@1700={drift_1700:.3f}, "
                    f"stable_S_frac={vS_share:.3f}, max_approx_runs={max_ap}"
                )
        print()

        # Independent calibration across all 5 environments
        print("--- ITEMS 2 & 5A: INDEPENDENT CONTINUOUS-MATRIX CALIBRATION ---")
        ctx = mp.get_context("spawn")
        cal_tasks = [(env, 81000 + i * 1000) for i, env in enumerate(["A", "B5", "B20", "B60", "B200"])]
        with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
            cal_lists = list(pool.map(calibrate_env_task, cal_tasks))
        ours_cal_by_el = {
            (row["env"], row["L"]): row for sublist in cal_lists for row in sublist
        }

        cal_agree = 0
        for env in ("A", "B5", "B20", "B60", "B200"):
            for L in (10, 40, 160):
                o = ours_cal_by_el[(env, L)]
                t = cal_by_el[(env, L)]
                se_c = math.hypot(o["D_se"], t["D_se"])
                z_d = abs(o["D"] - t["D"]) / se_c
                ok = z_d <= TOL_CAL_COMBINED_SE
                cal_agree += int(ok)
                print(
                    f"  {env:<4} L={L:<3}: D_ours={o['D']:.6f}+-{o['D_se']:.6f} | "
                    f"D_theirs={t['D']:.6f}+-{t['D_se']:.6f} | |dD|/SEc={z_d:4.2f} | "
                    f"r_diff={o['r']-t['r']:+.2e} [{'AGREES' if ok else 'DIFFERS'}]"
                )
        print(f"Calibration D agreement within {TOL_CAL_COMBINED_SE:.1f} combined SE: {cal_agree}/15\n")

        # Item 2: Independent whole-number simulation of 18 primary/clumped + 3 chain4 = 21 cells
        print("--- ITEM 2: INDEPENDENT WHOLE-NUMBER RECOMPUTATION (21 CELLS) ---")
        sim_tasks = []
        for idx, (env, L, a, k, b0, clump, reps, label) in enumerate(PRIMARY_SAMPLE_SPECS):
            oc = ours_cal_by_el[(env, L)]
            sim_tasks.append((env, L, a, k, b0, clump, reps, label, oc["gf"], oc["gs"], 91000 + idx * 97))

        with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
            sim_results = list(pool.map(sim_worker, sim_tasks))

        chain_ours = calibrate_and_sim_chain4()

        print(f"{'Cell key':<28} {'n_ours':>6} {'p_ours+-SE':>17} {'p_theirs+-SE':>17} {'|dp|/SEc':>8} {'pgen_ours':>9} {'pgen_th':>9} {'Verdict'}")
        cell_agree = 0
        for res in sim_results:
            th = cell_by_key[res["key"]]
            se_c = math.hypot(res["se"], th["se"])
            z_p = 0.0 if (res["p"] == th["p"] and se_c == 0.0) else abs(res["p"] - th["p"]) / se_c
            se_gc = math.hypot(res["se_gen"], th["se_gen"])
            z_g = 0.0 if (res["p_gen"] == th["p_gen"] and se_gc == 0.0) else abs(res["p_gen"] - th["p_gen"]) / se_gc
            ok = (z_p <= TOL_CELL_COMBINED_SE) and (z_g <= TOL_CELL_COMBINED_SE)
            cell_agree += int(ok)
            o_str = f"{res['p']:.4f}+-{res['se']:.4f}"
            t_str = f"{th['p']:.4f}+-{th['se']:.4f}"
            print(
                f"{res['key']:<28} {res['n']:6d} {o_str:>17} {t_str:>17} {z_p:8.2f} "
                f"{res['p_gen']:9.4f} {th['p_gen']:9.4f} {'AGREES' if ok else 'DIFFERS'}"
            )
        for ch in chain_ours:
            th = chain_by_key[ch["key"]]
            se_c = math.hypot(ch["se"], th["se"])
            z_p = abs(ch["p"] - th["p"]) / se_c
            se_dc = math.hypot(ch["D_se"], th["D_se"])
            z_d = abs(ch["D"] - th["D"]) / se_dc
            ok = (z_p <= TOL_CELL_COMBINED_SE) and (z_d <= TOL_CAL_COMBINED_SE)
            cell_agree += int(ok)
            o_str = f"{ch['p']:.4f}+-{ch['se']:.4f}"
            t_str = f"{th['p']:.4f}+-{th['se']:.4f}"
            print(
                f"{ch['key']:<28} {ch['n']:6d} {o_str:>17} {t_str:>17} {z_p:8.2f} "
                f"{ch['p_gen']:9.4f} {th['p_gen']:9.4f} {'AGREES' if ok else 'DIFFERS'} "
                f"(D_ours={ch['D']:.4f}+-{ch['D_se']:.4f} vs {th['D']:.4f}+-{th['D_se']:.4f}, {z_d:.2f} SE)"
            )
        print(f"\nSimulated cell agreement within {TOL_CELL_COMBINED_SE:.1f} combined SE: {cell_agree}/{len(sim_results) + len(chain_ours)}\n")

        # Item 3: Recount all headline counts and denominators
        print("--- ITEM 3: INDEPENDENT RECOUNT OF ALL HEADLINE COUNTS & DENOMINATORS ---")
        prim = [r for r in theirs_cells if r["kind"] == "primary"]
        clumped = [r for r in theirs_cells if r["kind"] == "clumped"]
        extra_dur = [r for r in theirs_cells if r["kind"] == "extra_duration"]
        total_traj_192 = sum(r["n"] for r in theirs_cells)
        total_traj_chain = sum(r["n"] for r in theirs_chain)
        total_approx_192 = sum(r["approx_runs"] for r in theirs_cells)
        total_zero_host_surv = sum(r["survivors_with_zero_hosts"] for r in theirs_cells)
        print(
            f"Inventory: primary={len(prim)}, clumped={len(clumped)}, extra_duration={len(extra_dur)}, "
            f"chain4={len(theirs_chain)} => total={len(theirs_cells)+len(theirs_chain)} cells"
        )
        print(
            f"Trajectories: 192-cell={total_traj_192}, chain4={total_traj_chain}, total={total_traj_192+total_traj_chain} | "
            f"approx_runs(192)={total_approx_192} | zero-host survivors(192)={total_zero_host_surv}"
        )

        ix = {(r["env"], r["a"], r["k"], r["b0"], r["L"], bool(r.get("clump", False))): r for r in theirs_cells}

        def compare_pair(r1: dict, r2: dict, field: str = "p") -> dict:
            sef = "se_gen" if field == "p_gen" else "se"
            d = r1[field] - r2[field]
            se_plug = math.hypot(r1[sef], r2[sef])
            z_plug = d / se_plug if se_plug > 0.0 else 0.0
            # Also compute 1/n-floored SE and pooled 2-proportion SE for boundary sensitivity check
            se_floor = math.hypot(max(r1[sef], 1.0 / r1["n"]), max(r2[sef], 1.0 / r2["n"]))
            k1 = round(r1[field] * r1["n"])
            k2 = round(r2[field] * r2["n"])
            p_pool = (k1 + k2) / (r1["n"] + r2["n"])
            se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / r1["n"] + 1.0 / r2["n"]))
            z_pool = d / se_pool if se_pool > 0.0 else 0.0
            return {
                "d": d,
                "se": se_plug,
                "z": z_plug,
                "res_plug": abs(d) > 4.0 * se_plug,
                "res_floor": abs(d) > 4.0 * se_floor,
                "res_pool": abs(d) > 4.0 * se_pool,
                "vis_01": abs(d) >= 0.01,
            }

        for field_name, label_clk in (("p", "1,700-step calendar"), ("p_gen", "10-nominal-generation")):
            for L1, L2 in ((10, 40), (40, 160), (10, 160)):
                all_c = []
                out_c = []
                for env in ("A", "B20", "B200"):
                    for a in (0, 0.5, 1):
                        for k in (1, 10, 100):
                            for b0 in (1, 100):
                                c = compare_pair(ix[(env, a, k, b0, L1, False)], ix[(env, a, k, b0, L2, False)], field_name)
                                all_c.append(c)
                                if env in ("B20", "B200"):
                                    out_c.append(c)
                dec = sum(c["res_plug"] and c["d"] > 0 for c in all_c)
                dec_vis = sum(c["res_plug"] and c["d"] > 0 and c["vis_01"] for c in all_c)
                dec_floor = sum(c["res_floor"] and c["d"] > 0 for c in all_c)
                dec_pool = sum(c["res_pool"] and c["d"] > 0 for c in all_c)
                inc = sum(c["res_plug"] and c["d"] < 0 for c in all_c)
                unr = sum(not c["res_plug"] for c in all_c)
                out_dec = sum(c["res_plug"] and c["d"] > 0 for c in out_c)
                print(
                    f"  {label_clk:<23} L{L1}->L{L2}: {dec}/{len(all_c)} resolved decreases "
                    f"(>=0.01: {dec_vis}/{len(all_c)}; 1/n-floor: {dec_floor}; pooled-SE: {dec_pool}), "
                    f"{inc}/{len(all_c)} reversals, {unr}/{len(all_c)} unresolved | "
                    f"primary outages (B20,B200): {out_dec}/{len(out_c)}"
                )

        # Lognormal A cells on 10-generation clock
        a_prim_cells = [r for r in prim if r["env"] == "A"]
        a_all_cells = [r for r in theirs_cells if r["env"] == "A"]
        a_prim_zero_gen = sum(r["p_gen"] == 0.0 for r in a_prim_cells)
        a_all_zero_gen = sum(r["p_gen"] == 0.0 for r in a_all_cells)
        print(
            f"  Lognormal (A) 10-gen zero-event cells: primary={a_prim_zero_gen}/{len(a_prim_cells)}, "
            f"all (incl. 6 clumped)={a_all_zero_gen}/{len(a_all_cells)}"
        )

        # Supplemental outage L10->L160 counts (B5, B60, clumped B20/B200, chain4)
        supp_out_10_160 = []
        for env in ("B5", "B60"):
            for k in (1, 100):
                supp_out_10_160.append(compare_pair(ix[(env, 1, k, 100, 10, False)], ix[(env, 1, k, 100, 160, False)], "p"))
        for env in ("B20", "B200"):
            for a in (0, 1):
                supp_out_10_160.append(compare_pair(ix[(env, a, 10, 100, 10, True)], ix[(env, a, 10, 100, 160, True)], "p"))
        print(
            f"  Supplemental outage L10->L160 (4 in B5/B60 + 4 clumped in B20/B200): "
            f"{sum(c['res_plug'] and c['d'] > 0 for c in supp_out_10_160)}/{len(supp_out_10_160)} resolved decreases"
        )

        # Clumped vs fixed K=10 (calendar and 10-gen)
        clump_cal = [
            compare_pair(ix[(env, a, 10, 100, L, True)], ix[(env, a, 10, 100, L, False)], "p")
            for env in ("A", "B20", "B200") for L in (10, 40, 160) for a in (0, 1)
        ]
        clump_gen = [
            compare_pair(ix[(env, a, 10, 100, L, True)], ix[(env, a, 10, 100, L, False)], "p_gen")
            for env in ("A", "B20", "B200") for L in (10, 40, 160) for a in (0, 1)
        ]
        print(
            f"  Clumped 1/19 vs fixed 10: calendar resolved={sum(c['res_plug'] for c in clump_cal)}/{len(clump_cal)} "
            f"(max |z|={max(abs(c['z']) for c in clump_cal):.2f}); "
            f"10-gen resolved={sum(c['res_plug'] for c in clump_gen)}/{len(clump_gen)} "
            f"(max |z|={max(abs(c['z']) for c in clump_gen):.2f})"
        )

        # Share (a=1 vs a=0), Packet (K=100 vs K=1), Start (b0=100 vs b0=1)
        share_c = [
            compare_pair(ix[(env, 1, k, b0, L, False)], ix[(env, 0, k, b0, L, False)], "p")
            for env in ("A", "B20", "B200") for L in (10, 40, 160) for k in (1, 10, 100) for b0 in (1, 100)
        ]
        pkt_c = [
            compare_pair(ix[(env, a, 100, b0, L, False)], ix[(env, a, 1, b0, L, False)], "p")
            for env in ("A", "B20", "B200") for L in (10, 40, 160) for a in (0, 0.5, 1) for b0 in (1, 100)
        ]
        start_c = [
            (a, env, L, k, compare_pair(ix[(env, a, k, 100, L, False)], ix[(env, a, k, 1, L, False)], "p"))
            for env in ("A", "B20", "B200") for L in (10, 40, 160) for a in (0, 0.5, 1) for k in (1, 10, 100)
        ]
        print(
            f"  Share (a=1 vs a=0)   : +{sum(c['res_plug'] and c['d']>0 for c in share_c)}/{len(share_c)}, "
            f"-{sum(c['res_plug'] and c['d']<0 for c in share_c)}/{len(share_c)}, "
            f"unresolved={sum(not c['res_plug'] for c in share_c)}/{len(share_c)}"
        )
        print(
            f"  Packet (K=100 vs K=1): +{sum(c['res_plug'] and c['d']>0 for c in pkt_c)}/{len(pkt_c)}, "
            f"-{sum(c['res_plug'] and c['d']<0 for c in pkt_c)}/{len(pkt_c)}, "
            f"unresolved={sum(not c['res_plug'] for c in pkt_c)}/{len(pkt_c)}"
        )
        start_res = [(a, e, L, k, c) for a, e, L, k, c in start_c if c["res_plug"]]
        print(
            f"  Start (b0=100 vs b0=1): +{sum(c['res_plug'] and c['d']>0 for _,_,_,_,c in start_c)}/{len(start_c)} "
            f"(among a>0: {sum(c['res_plug'] for a,_,_,_,c in start_c if a>0)}/54; among a=0: {sum(c['res_plug'] for a,_,_,_,c in start_c if a==0)}/27) -> "
            f"{[(e, L, a, k, round(c['d'], 4), round(c['z'], 2)) for a, e, L, k, c in start_res]}"
        )
        # Verify exact match with r18_comparisons.json lengths
        assert len(theirs_comp["contrasts"]["p"]) == 162
        assert len(theirs_comp["contrasts"]["p_gen"]) == 162
        assert len(theirs_comp["contrasts"]["clump"]) == 18
        print()

        # Item 4: Gap-survival thresholds (continuous & discrete)
        print("--- ITEM 4: INDEPENDENT GAP-RETENTION THRESHOLDS (7.58 AND 3.48) ---")
        for shape, age in ((1, "fresh"), (4, "fresh"), (4, "stationary")):
            for M in (100, 10000):
                def surv_fn(u: float) -> float:
                    if shape == 1:
                        h_u = math.exp(-u)
                    elif age == "fresh":
                        h_u = erlang4_fresh_survival(u)
                    else:
                        h_u = erlang4_stationary_survival(u)
                    return gap_retention_prob(h_u, 1.0, np.ones(M))

                r95 = optimize.brentq(lambda u: surv_fn(u) - 0.95, 1e-9, 50.0, xtol=1e-14)
                r50 = optimize.brentq(lambda u: surv_fn(u) - 0.50, 1e-9, 50.0, xtol=1e-14)
                r05 = optimize.brentq(lambda u: surv_fn(u) - 0.05, 1e-9, 50.0, xtol=1e-14)
                print(f"  shape={shape}, age={age:<10}, M={M:5d}: 95%={r95:.6f}, 50%={r50:.6f}, 5%={r05:.6f}")
                if shape == 1 and age == "fresh" and M == 100:
                    closed_exp_05 = -math.log(1.0 - 0.95 ** (1.0 / 100.0))
                    assert abs(r05 - closed_exp_05) <= TOL_GAP_RATIO_ATOL
                    assert abs(round(r05, 3) - 7.576) <= 1e-12
                if shape == 4 and age == "fresh" and M == 100:
                    assert abs(round(r05, 3) - 3.476) <= 1e-12

        # Discrete geometric vs 4-phase binomial 5% retention step counts at M=100
        for L_d in (10, 40, 160):
            h_star = 1.0 - 0.95 ** 0.01
            g_geom = math.ceil(math.log(h_star) / math.log(1.0 - 1.0 / L_d))
            # Find smallest integer G with P(Binom(G, 4/L_d) <= 3) <= h_star
            g_bin = 1
            while stats.binom.cdf(3, g_bin, 4.0 / L_d) > h_star:
                g_bin += 1
            print(
                f"  Discrete M=100 5% retention steps at L={L_d:<3}: "
                f"geometric G={g_geom} (G/L={g_geom/L_d:.3f}) vs 4-phase G={g_bin} (G/L={g_bin/L_d:.3f})"
            )
        # Crossing point where single-host exponential and fresh 4-phase survival are equal
        u_cross = optimize.brentq(lambda u: erlang4_fresh_survival(u) - math.exp(-u), 0.1, 5.0, xtol=1e-14)
        print(f"  Single-host survival crossing (h_4(u) == h_1(u)): G/tau = {u_cross:.6f} (h = {math.exp(-u_cross):.6f})\n")

        # Item 5: Cross-law D-ordering audit & full within-environment denominators
        print("--- ITEM 5: D-ORDERING AUDIT & PAIRWISE DENOMINATORS ---")
        c1_th = cell_by_key["A_L10_a0_k1_b1"]
        c2_th = cell_by_key["A_L40_a1_k100_b1"]
        c_eq_th = cell_by_key["A_L40_a0_k1_b1"]
        c1_ours = next(x for x in sim_results if x["key"] == "A_L10_a0_k1_b1")
        c2_ours = next(x for x in sim_results if x["key"] == "A_L40_a1_k100_b1")
        d1_ours = ours_cal_by_el[("A", 10)]
        d2_ours = ours_cal_by_el[("A", 40)]
        print(
            f"  Headline pair:\n"
            f"    A_L10_a0_k1_b1  : D_ours={d1_ours['D']:.6f}+-{d1_ours['D_se']:.6f} (theirs {cal_by_el[('A',10)]['D']:.6f}), "
            f"p_ours={c1_ours['p']:.4f}+-{c1_ours['se']:.4f} (theirs {c1_th['p']:.4f}+-{c1_th['se']:.4f})\n"
            f"    A_L40_a1_k100_b1: D_ours={d2_ours['D']:.6f}+-{d2_ours['D_se']:.6f} (theirs {cal_by_el[('A',40)]['D']:.6f}), "
            f"p_ours={c2_ours['p']:.4f}+-{c2_ours['se']:.4f} (theirs {c2_th['p']:.4f}+-{c2_th['se']:.4f})\n"
            f"    Same-L control A_L40_a0_k1_b1 (exact same D={d2_ours['D']:.6f} & same per-step drift): theirs p={c_eq_th['p']:.4f} (0/3000)"
        )

        tot_within_env = 0
        tot_same_L = 0
        same_L_res = 0
        tot_cross_L = 0
        cross_L_d_res = 0
        cross_L_both_res_conc = 0
        cross_L_both_res_rev = 0
        cross_L_ext_unres = 0
        for env in ("A", "B20", "B200"):
            env_prim = [r for r in prim if r["env"] == env]
            for x, y in itertools.combinations(env_prim, 2):
                tot_within_env += 1
                cp = compare_pair(x, y, "p")
                if x["L"] == y["L"]:
                    tot_same_L += 1
                    if cp["res_plug"]:
                        same_L_res += 1
                else:
                    tot_cross_L += 1
                    cx = cal_by_el[(env, x["L"])]
                    cy = cal_by_el[(env, y["L"])]
                    dd = cx["D"] - cy["D"]
                    dse = math.hypot(cx["D_se"], cy["D_se"])
                    if abs(dd) > 4.0 * dse:
                        cross_L_d_res += 1
                    if cp["res_plug"] and abs(dd) > 4.0 * dse:
                        if dd * cp["d"] > 0:
                            cross_L_both_res_conc += 1
                        else:
                            cross_L_both_res_rev += 1
                    else:
                        cross_L_ext_unres += 1
        print(
            f"  All within-env primary pairs: {tot_within_env} = {tot_cross_L} cross-L + {tot_same_L} equal-L (same D)\n"
            f"    Cross-L pairs ({tot_cross_L}, D resolved in {cross_L_d_res}/{tot_cross_L}): "
            f"concordant={cross_L_both_res_conc}, reversed={cross_L_both_res_rev}, "
            f"extinction-unresolved={cross_L_ext_unres}\n"
            f"    Equal-L / equal-D pairs ({tot_same_L}): resolved different extinction={same_L_res}/{tot_same_L}, "
            f"unresolved={tot_same_L - same_L_res}/{tot_same_L}"
        )
        assert cross_L_both_res_conc == len(theirs_comp["D_concordant"]) == 2285
        assert cross_L_both_res_rev == len(theirs_comp["D_reversed"]) == 43
        assert same_L_res == len(theirs_comp["D_tied"]) == 494

        # Check within-law non-monotone D in the 4-phase supplement!
        c4_10, c4_40, c4_160 = chain_ours
        z_c4_10_40 = (c4_10["p"] - c4_40["p"]) / math.hypot(c4_10["se"], c4_40["se"])
        print(
            f"  Four-phase supplement non-monotone D check:\n"
            f"    L=10: D={c4_10['D']:.4f}, p={c4_10['p']:.4f}+-{c4_10['se']:.4f} | "
            f"L=40: D={c4_40['D']:.4f}, p={c4_40['p']:.4f}+-{c4_40['se']:.4f} | "
            f"L=160: D={c4_160['D']:.4f}, p={c4_160['p']:.4f}+-{c4_160['se']:.4f}\n"
            f"    (D rises +1.01 from L=10 to L=40 while calendar extinction falls: p(10)-p(40) = {c4_10['p']-c4_40['p']:+.4f}, {z_c4_10_40:.2f} SE)"
        )

        print(f"\nCompleted all checks cleanly in {time.monotonic() - t_start:.2f} s.")
        sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
