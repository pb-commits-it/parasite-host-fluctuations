#!/usr/bin/env python3
"""Independent clean-room reproduction and confounder audit of virtualworld pilot (r34).

Written solely from `protocol.md`, `pilot.md`, and `validation.md` BEFORE reading
anything in `virtualworld/world/` (zero imports or copied code from `virtualworld/`).

Pre-validation checks (study rules Rule 7, tolerances stated beforehand):
  V1a: Exact constant-host multitype branching PGF recursion reproduces all 9
       analytical reference probabilities in `validation.md` Gate v to <= 1e-9.
  V1b: Independent stochastic simulator (10,000 trajectories/setting) agrees with
       exact PGF within 4 * sqrt(p*(1-p)/10000) + 0.002 across all 9 Gate v cells.
  V2:  Conditional-mean invasion simulator at small noise (sigma = 0.03, 32
       antithetic pairs, 65,536 steps after 2,048 burn-in, control variate (h-1)/T)
       agrees with the exact ring coefficient K within 0.05*K + 4*SE with SE <= 0.03*K,
       and exact sigma^2 * K matches `pilot.md` to printed 6-decimal precision.

Tasks:
  1.   400-replicate reproduction of the 4 main-contrast cells at phi = 0.6,
       sigma in {0, 0.6}, steady stay in {1, 20}, log R0 = 0.05, horizon 500 steps.
  2a.  Fixed-generation horizon audit (exact PGF at sigma = 0 plus 400-replicate
       simulations up to G = 167 generations, evaluating G in {22.7, 23, 50, 100, 167}).
  2b.  Dependence on log R0: 400-replicate rerun of the 4 cells at log R0 = 0.2,
       horizon 500 steps.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, t as student_t


OUT_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
BINOM_CHUNK = 10**14  # < 2^53; exact IEEE-754 & int64 representation


def exact_binomial_draw(rng: np.random.Generator, n: int, p: float) -> int:
    """Exact Binomial(n, p) draw for arbitrary-precision Python int n >= 0 (no cap, no approximation)."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n <= BINOM_CHUNK:
        return int(rng.binomial(n, p))
    q, rem = divmod(n, BINOM_CHUNK)
    total = int(rng.binomial(rem, p)) if rem > 0 else 0
    batch_max = 100_000
    while q > 0:
        b = min(q, batch_max)
        arr = rng.binomial(BINOM_CHUNK, p, size=b)
        # Each element <= 10^14, batch <= 10^5 => batch sum <= 10^19; sum in object/Python ints
        total += sum(int(x) for x in np.add.reduceat(arr, range(0, b, 10_000)))
        q -= b
    return total


def exact_poisson_product_draw(rng: np.random.Generator, count: int, rate_per_ind: float) -> int:
    """Exact Poisson(count * rate_per_ind) draw for arbitrary-precision int `count` >= 0."""
    if count <= 0 or rate_per_ind <= 0.0:
        return 0
    max_chunk_lam = 1e14
    if count * rate_per_ind <= max_chunk_lam:
        return int(rng.poisson(count * rate_per_ind))
    step_count = max(1, int(max_chunk_lam // max(rate_per_ind, 1.0)))
    q, rem = divmod(count, step_count)
    total = int(rng.poisson(rem * rate_per_ind)) if rem > 0 else 0
    chunk_lam = step_count * rate_per_ind
    batch_max = 100_000
    while q > 0:
        b = min(q, batch_max)
        arr = rng.poisson(chunk_lam, size=b)
        total += sum(int(x) for x in np.add.reduceat(arr, range(0, b, 10_000)))
        q -= b
    return total


def exact_ring_K(stays: tuple[float, ...], phi: float) -> float:
    """Second-order small-noise loss coefficient K per circuit (protocol.md lines 30-35)."""
    T = float(sum(stays))
    theta = phi / (1.0 - phi)
    if phi == 0.0:
        return 1.0 / (2.0 * T)
    R = 1.0
    for tau in stays:
        R *= phi / (tau + (1.0 - tau) * phi)
    return (1.0 + 2.0 * theta) / (2.0 * T) - R / (1.0 - R)


def constant_env_matrix(tau0: float, tau1: float, log_r0: float) -> np.ndarray:
    """Constant-host (h=1) column projection matrix M for 2 geometric stages.

    Stage 0 (exposed): stay prob s0 = 1 - 1/tau0, transfer prob q0 = 1/tau0, yield 1.
    Stage 1 (steady):  stay prob s1 = 1 - 1/tau1, transfer prob q1 = 1/tau1, yield exp(log_r0).
    """
    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    r0 = math.exp(log_r0)
    return np.array([[s0, q1 * r0], [q0, s1]], dtype=np.float64)


def stable_initial_counts(total: int, tau0: float, tau1: float, log_r0: float) -> tuple[int, int]:
    """Allocate `total` individuals by constant-environment stable stage proportions.

    Uses exact Perron-Frobenius right eigenvector of M(h=1) with deterministic
    largest-remainder (Hamilton) rounding; ties go to stage 0.
    """
    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    r0 = math.exp(log_r0)
    disc = (s0 - s1) ** 2 + 4.0 * q0 * q1 * r0
    lam = 0.5 * ((s0 + s1) + math.sqrt(disc))
    v0 = lam - s1
    v1 = q0
    v_sum = v0 + v1
    raw0 = total * (v0 / v_sum)
    raw1 = total * (v1 / v_sum)
    n0 = int(math.floor(raw0))
    n1 = int(math.floor(raw1))
    rem = total - (n0 + n1)
    if rem == 1:
        if (raw0 - n0) >= (raw1 - n1):
            n0 += 1
        else:
            n1 += 1
    elif rem == 2:
        n0 += 1
        n1 += 1
    assert n0 + n1 == total
    return n0, n1


def exact_constant_host_pgf_curve(
    tau0: float,
    tau1: float,
    log_r0: float,
    init_counts: tuple[int, int],
    max_steps: int,
) -> np.ndarray:
    """Exact multitype Galton-Watson extinction probability P(N_t == 0) for steps 0..max_steps."""
    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    r0 = math.exp(log_r0)
    n0, n1 = init_counts
    u0, u1 = 0.0, 0.0
    p_ext = np.zeros(max_steps + 1, dtype=np.float64)
    p_ext[0] = (u0**n0) * (u1**n1)
    for step in range(1, max_steps + 1):
        nu0 = s0 * u0 + q0 * math.exp(1.0 * (u1 - 1.0))
        nu1 = s1 * u1 + q1 * math.exp(r0 * (u0 - 1.0))
        u0, u1 = nu0, nu1
        p_ext[step] = (u0**n0) * (u1**n1)
    return p_ext


def exact_asymptotic_extinction(log_r0: float, init_counts: tuple[int, int]) -> float:
    """Exact ultimate extinction probability P(N_t -> 0) under constant hosts (sigma=0).

    Without stage mortality, ultimate single-ancestor extinction (u0*, u1*) solves
    u0 = exp(u1 - 1), u1 = exp(R0 * (u0 - 1)), independent of geometric stage durations.
    """
    if log_r0 <= 0.0:
        return 1.0
    r0 = math.exp(log_r0)
    u0 = 0.5
    for _ in range(500):
        u1 = math.exp(r0 * (u0 - 1.0))
        u0_next = math.exp(u1 - 1.0)
        if abs(u0_next - u0) < 1e-15:
            u0 = u0_next
            break
        u0 = u0_next
    u1 = math.exp(r0 * (u0 - 1.0))
    n0, n1 = init_counts
    return (u0**n0) * (u1**n1)


def sample_host_schedule(
    rng: np.random.Generator,
    steps: int,
    sigma: float,
    phi: float,
) -> tuple[np.ndarray, float, float]:
    """Forced lognormal AR(1) host h_t = exp(sigma * z_t - sigma^2 / 2), stationary unit Gaussian z_t."""
    if sigma == 0.0:
        h = np.ones(steps, dtype=np.float64)
        return h, 1.0, 0.0
    eps = rng.standard_normal(steps)
    z = np.empty(steps, dtype=np.float64)
    z[0] = eps[0]
    if phi == 0.0:
        z[1:] = eps[1:]
    else:
        scale = math.sqrt(1.0 - phi * phi)
        for t in range(1, steps):
            z[t] = phi * z[t - 1] + scale * eps[t]
    log_h = sigma * z - 0.5 * sigma * sigma
    h = np.exp(log_h)
    realized_mean = float(np.mean(h))
    realized_log_var = float(np.var(log_h, ddof=1)) if steps > 1 else 0.0
    return h, realized_mean, realized_log_var


def simulate_cell_replicates(
    *,
    n_reps: int,
    tau0: float,
    tau1: float,
    sigma: float,
    phi: float,
    log_r0: float,
    horizon: int,
    init_counts: tuple[int, int] | None,
    base_seed: int,
    early_steps: int = 100,
) -> dict[str, Any]:
    """Run `n_reps` independent host + individual branching trajectories and invasion trackers."""
    if init_counts is None:
        init_counts = stable_initial_counts(20, tau0, tau1, log_r0)
    n0_init, n1_init = init_counts
    total_init = float(n0_init + n1_init)

    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    r0 = math.exp(log_r0)

    first_ext_step = np.full(n_reps, horizon + 1, dtype=np.int64)
    early_growth = np.zeros(n_reps, dtype=np.float64)
    horizon_growth = np.zeros(n_reps, dtype=np.float64)
    host_means = np.zeros(n_reps, dtype=np.float64)
    host_log_vars = np.zeros(n_reps, dtype=np.float64)
    int64_max = int(np.iinfo(np.int64).max)
    n_exceeded_int64 = 0
    max_log10_total = 0.0

    early_idx = min(early_steps, horizon)

    for rep in range(n_reps):
        rng = np.random.default_rng(base_seed + rep)
        h, h_mean, h_lvar = sample_host_schedule(rng, horizon, sigma, phi)
        host_means[rep] = h_mean
        host_log_vars[rep] = h_lvar

        # Individual branching trajectory (exact arbitrary-precision Python ints; no cap)
        n0, n1 = int(n0_init), int(n1_init)
        ext_t = horizon + 1
        rep_exceeded = False
        for t in range(horizon):
            if n0 == 0 and n1 == 0:
                if ext_t > horizon:
                    ext_t = t
                break
            leave0 = exact_binomial_draw(rng, n0, q0)
            stay0 = n0 - leave0
            if tau1 == 1.0:
                leave1 = n1
                stay1 = 0
            else:
                leave1 = exact_binomial_draw(rng, n1, q1)
                stay1 = n1 - leave1

            born1 = exact_poisson_product_draw(rng, leave0, float(h[t]))
            born0 = exact_poisson_product_draw(rng, leave1, r0)

            n0 = stay0 + born0
            n1 = stay1 + born1
            tot_n = n0 + n1
            if tot_n > int64_max:
                rep_exceeded = True
            if tot_n > 0:
                l10 = math.log10(tot_n)
                if l10 > max_log10_total:
                    max_log10_total = l10
            if tot_n == 0 and ext_t > horizon:
                ext_t = t + 1

        if rep_exceeded:
            n_exceeded_int64 += 1
        first_ext_step[rep] = ext_t

        # Conditional-on-environment linear invasion trajectory (log-rescaled to avoid under/overflow)
        m0 = float(n0_init) / total_init
        m1 = float(n1_init) / total_init
        cum_log = 0.0
        for t in range(horizon):
            nm0 = s0 * m0 + q1 * r0 * m1
            nm1 = q0 * float(h[t]) * m0 + s1 * m1
            tot = nm0 + nm1
            cum_log += math.log(tot)
            m0 = nm0 / tot
            m1 = nm1 / tot
            if t + 1 == early_idx:
                early_growth[rep] = cum_log / early_idx
        horizon_growth[rep] = cum_log / horizon

    extinct_mask = first_ext_step <= horizon
    k_ext = int(np.sum(extinct_mask))
    p_ext = k_ext / n_reps
    w_lo, w_hi = wilson_ci(k_ext, n_reps)

    eg_mean = float(np.mean(early_growth))
    eg_sd = float(np.std(early_growth, ddof=1))
    t_crit = float(student_t.ppf(0.975, df=n_reps - 1))
    eg_se = eg_sd / math.sqrt(n_reps)
    eg_ci = (eg_mean - t_crit * eg_se, eg_mean + t_crit * eg_se)

    return {
        "n_reps": n_reps,
        "tau0": tau0,
        "tau1": tau1,
        "sigma": sigma,
        "phi": phi,
        "log_r0": log_r0,
        "horizon": horizon,
        "init_counts": list(init_counts),
        "base_seed": base_seed,
        "extinct_count": k_ext,
        "p_ext": p_ext,
        "wilson_95": [w_lo, w_hi],
        "early_growth_mean": eg_mean,
        "early_growth_se": eg_se,
        "early_growth_sd": eg_sd,
        "early_growth_t95": [eg_ci[0], eg_ci[1]],
        "horizon_growth_mean": float(np.mean(horizon_growth)),
        "host_mean_avg": float(np.mean(host_means)),
        "host_log_var_avg": float(np.mean(host_log_vars)),
        "n_exceeded_int64": n_exceeded_int64,
        "max_log10_total": max_log10_total,
        "first_ext_step": first_ext_step,
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% interval for binomial proportion k/n."""
    z = float(norm.ppf(1.0 - 0.5 * alpha))
    p = k / n
    z2n = (z * z) / n
    denom = 1.0 + z2n
    center = (p + 0.5 * z2n) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return center - half, center + half


def compute_interaction_summary(
    c_s6_t1: dict[str, Any],
    c_s0_t1: dict[str, Any],
    c_s6_t20: dict[str, Any],
    c_s0_t20: dict[str, Any],
    jeffreys_seed: int = 9390001,
    jeffreys_draws: int = 100_000,
) -> dict[str, Any]:
    """Compute binary extinction and early-growth noise-by-stay interaction + intervals."""
    z = float(norm.ppf(0.975))

    cells = [c_s6_t1, c_s0_t1, c_s6_t20, c_s0_t20]
    ps = [c["p_ext"] for c in cells]
    ns = [c["n_reps"] for c in cells]
    ks = [c["extinct_count"] for c in cells]

    noise_t1 = ps[0] - ps[1]
    noise_t20 = ps[2] - ps[3]
    delta = noise_t1 - noise_t20
    var_sum = sum(p * (1.0 - p) / n for p, n in zip(ps, ns))
    se_delta = math.sqrt(var_sum)
    wald_ci = [delta - z * se_delta, delta + z * se_delta]
    wald_ci_196 = [delta - 1.96 * se_delta, delta + 1.96 * se_delta]

    se_t1 = math.sqrt(ps[0] * (1.0 - ps[0]) / ns[0] + ps[1] * (1.0 - ps[1]) / ns[1])
    se_t20 = math.sqrt(ps[2] * (1.0 - ps[2]) / ns[2] + ps[3] * (1.0 - ps[3]) / ns[3])

    rng = np.random.default_rng(jeffreys_seed)
    draws = [
        rng.beta(k + 0.5, n - k + 0.5, size=jeffreys_draws)
        for k, n in zip(ks, ns)
    ]
    post_delta = (draws[0] - draws[1]) - (draws[2] - draws[3])
    jeffreys_ci = [
        float(np.quantile(post_delta, 0.025)),
        float(np.quantile(post_delta, 0.975)),
    ]

    # Early-growth interaction oriented to loss: [r(0,1) - r(.6,1)] - [r(0,20) - r(.6,20)]
    gs = [c["early_growth_mean"] for c in cells]
    g_ses = [c["early_growth_se"] for c in cells]
    gloss_t1 = gs[1] - gs[0]
    gloss_t20 = gs[3] - gs[2]
    g_delta = gloss_t1 - gloss_t20
    g_se = math.sqrt(sum(se * se for se in g_ses))
    g_ci = [g_delta - z * g_se, g_delta + z * g_se]

    return {
        "noise_effect_stay1": noise_t1,
        "noise_effect_stay1_ci95": [noise_t1 - z * se_t1, noise_t1 + z * se_t1],
        "noise_effect_stay20": noise_t20,
        "noise_effect_stay20_ci95": [noise_t20 - z * se_t20, noise_t20 + z * se_t20],
        "interaction": delta,
        "interaction_se": se_delta,
        "interaction_wald_95": wald_ci,
        "interaction_wald_196": wald_ci_196,
        "interaction_jeffreys_95": jeffreys_ci,
        "early_growth_loss_stay1": gloss_t1,
        "early_growth_loss_stay20": gloss_t20,
        "early_growth_interaction": g_delta,
        "early_growth_interaction_se": g_se,
        "early_growth_interaction_95": g_ci,
    }


def extract_horizon_slice(cell: dict[str, Any], horizon_steps: int) -> dict[str, Any]:
    """Evaluate extinction at `horizon_steps` from an already-simulated cell with horizon >= horizon_steps."""
    assert cell["horizon"] >= horizon_steps
    n = cell["n_reps"]
    k = int(np.sum(cell["first_ext_step"] <= horizon_steps))
    w_lo, w_hi = wilson_ci(k, n)
    return {
        "n_reps": n,
        "extinct_count": k,
        "p_ext": k / n,
        "wilson_95": [w_lo, w_hi],
        "early_growth_mean": 0.0,
        "early_growth_se": 0.0,
    }


def run_prevalidations() -> dict[str, Any]:
    """study rules Rule 7: reproduce known analytical benchmarks before using new results."""
    results: dict[str, Any] = {}

    # V1a & V1b: Gate v multitype PGF recursion + 10,000-rep stochastic check
    gate_v_refs = {
        (-0.1, 20): 0.0590872813,
        (-0.1, 60): 0.464558519,
        (-0.1, 120): 0.790529106,
        (0.0, 20): 0.0404554605,
        (0.0, 60): 0.310148523,
        (0.0, 120): 0.553746734,
        (0.1, 20): 0.0267201583,
        (0.1, 60): 0.182635305,
        (0.1, 120): 0.300881377,
    }
    v1_rows = []
    for idx, lr in enumerate([-0.1, 0.0, 0.1]):
        pgf_curve = exact_constant_host_pgf_curve(2.0, 5.0, lr, (10, 0), 120)
        sim = simulate_cell_replicates(
            n_reps=10_000,
            tau0=2.0,
            tau1=5.0,
            sigma=0.0,
            phi=0.0,
            log_r0=lr,
            horizon=120,
            init_counts=(10, 0),
            base_seed=9300000 + idx * 20_000,
            early_steps=20,
        )
        for h_step in (20, 60, 120):
            p_exact = float(pgf_curve[h_step])
            p_ref = gate_v_refs[(lr, h_step)]
            pgf_err = abs(p_exact - p_ref)
            assert pgf_err <= 1e-9, f"PGF mismatch at ({lr}, {h_step}): {pgf_err}"

            p_sim = float(np.mean(sim["first_ext_step"] <= h_step))
            tol = 4.0 * math.sqrt(p_exact * (1.0 - p_exact) / 10_000.0) + 0.002
            sim_err = abs(p_sim - p_exact)
            assert sim_err <= tol, f"V1b failed at ({lr}, {h_step}): err={sim_err} > tol={tol}"
            v1_rows.append({
                "log_r0": lr,
                "horizon": h_step,
                "p_pgf": p_exact,
                "p_sim_10k": p_sim,
                "abs_err": sim_err,
                "tolerance": tol,
            })
    results["V1_gate_v"] = v1_rows

    # V2: Exact ring K vs pilot.md sigma^2*K table & small-noise conditional-mean check
    pilot_s2k_refs = {
        (0.3, 0.0, 1): 0.015000,
        (0.3, 0.0, 5): 0.006429,
        (0.3, 0.0, 20): 0.002045,
        (0.3, 0.6, 1): 0.028846,
        (0.3, 0.6, 5): 0.015836,
        (0.3, 0.6, 20): 0.005408,
        (0.6, 0.0, 1): 0.060000,
        (0.6, 0.0, 5): 0.025714,
        (0.6, 0.0, 20): 0.008182,
        (0.6, 0.6, 1): 0.115385,
        (0.6, 0.6, 5): 0.063345,
        (0.6, 0.6, 20): 0.021631,
    }
    for (sig, ph, st), ref_val in pilot_s2k_refs.items():
        val = sig * sig * exact_ring_K((2.0, float(st)), ph)
        assert abs(val - ref_val) <= 1e-6, f"sigma^2 K mismatch at {(sig, ph, st)}: {val} vs {ref_val}"

    # Small-noise (sigma=0.03) antithetic linear check on (2, 1) and (2, 20) at phi=0.6
    v2_rows = []
    sigma_sm = 0.03
    burn = 2048
    rec = 65536
    total_steps = burn + rec
    for p_idx, tau1 in enumerate([1.0, 20.0]):
        K_exact = exact_ring_K((2.0, tau1), 0.6)
        T = 2.0 + tau1
        s0, q0 = 0.5, 0.5
        s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
        pair_ests = []
        scale = math.sqrt(1.0 - 0.6 * 0.6)
        for pair in range(32):
            rng = np.random.default_rng(9320000 + p_idx * 1000 + pair)
            eps = rng.standard_normal(total_steps)
            z = np.empty(total_steps, dtype=np.float64)
            z[0] = eps[0]
            for t in range(1, total_steps):
                z[t] = 0.6 * z[t - 1] + scale * eps[t]
            vals = []
            for sign in (+1.0, -1.0):
                h = np.exp(sign * sigma_sm * z - 0.5 * sigma_sm * sigma_sm)
                m0, m1 = 2.0 / T, tau1 / T
                log_sum = 0.0
                for t in range(total_steps):
                    nm0 = s0 * m0 + q1 * m1
                    nm1 = q0 * float(h[t]) * m0 + s1 * m1
                    tot = nm0 + nm1
                    if t >= burn:
                        log_sum += math.log(tot)
                    m0 = nm0 / tot
                    m1 = nm1 / tot
                r_hat = log_sum / rec - float(np.mean(h[burn:] - 1.0)) / T
                vals.append(-T * r_hat / (sigma_sm * sigma_sm))
            pair_ests.append(0.5 * (vals[0] + vals[1]))
        est = float(np.mean(pair_ests))
        se = float(np.std(pair_ests, ddof=1) / math.sqrt(32))
        tol = 0.05 * K_exact + 4.0 * se
        assert abs(est - K_exact) <= tol and se <= 0.03 * K_exact, (
            f"V2 failed for tau1={tau1}: est={est}, K={K_exact}, se={se}"
        )
        v2_rows.append({
            "stays": [2.0, tau1],
            "phi": 0.6,
            "K_exact": K_exact,
            "K_sim": est,
            "se": se,
            "abs_err": abs(est - K_exact),
            "tolerance": tol,
        })
    results["V2_small_noise_K"] = v2_rows
    return results


def estimate_threshold_D(
    tau0: float,
    tau1: float,
    sigma: float,
    phi: float,
    base_seed: int,
    n_pairs: int = 32,
    burn: int = 2048,
    rec: int = 32768,
) -> dict[str, float]:
    """Estimate threshold log-R0 shift D by secant/Newton root-finding on 32 antithetic pairs."""
    if sigma == 0.0:
        return {"D_mean": 0.0, "D_se": 0.0, "D_t95_lo": 0.0, "D_t95_hi": 0.0, "root_resid": 0.0}
    total_steps = burn + rec
    s0, q0 = 1.0 - 1.0 / tau0, 1.0 / tau0
    s1, q1 = 1.0 - 1.0 / tau1, 1.0 / tau1
    T = tau0 + tau1
    scale = math.sqrt(1.0 - phi * phi)

    # Pre-generate 32 antithetic host path pairs
    h_paths = []
    for pair in range(n_pairs):
        rng = np.random.default_rng(base_seed + pair)
        eps = rng.standard_normal(total_steps)
        z = np.empty(total_steps, dtype=np.float64)
        z[0] = eps[0]
        if phi == 0.0:
            z[1:] = eps[1:]
        else:
            for t in range(1, total_steps):
                z[t] = phi * z[t - 1] + scale * eps[t]
        h_pos = np.exp(sigma * z - 0.5 * sigma * sigma)
        h_neg = np.exp(-sigma * z - 0.5 * sigma * sigma)
        h_paths.append((h_pos, h_neg))

    def eval_growth(y: float) -> tuple[float, np.ndarray]:
        r0 = math.exp(y)
        pair_r = np.empty(n_pairs, dtype=np.float64)
        for i, (hp, hn) in enumerate(h_paths):
            acc = 0.0
            for h in (hp, hn):
                m0, m1 = tau0 / T, tau1 / T
                ls = 0.0
                for t in range(total_steps):
                    nm0 = s0 * m0 + q1 * r0 * m1
                    nm1 = q0 * float(h[t]) * m0 + s1 * m1
                    tot = nm0 + nm1
                    if t >= burn:
                        ls += math.log(tot)
                    m0 = nm0 / tot
                    m1 = nm1 / tot
                acc += ls / rec
            pair_r[i] = 0.5 * acc
        return float(np.mean(pair_r)), pair_r

    y0 = sigma * sigma * exact_ring_K((tau0, tau1), phi)
    y1 = y0 + 0.01
    f0, _ = eval_growth(y0)
    f1, _ = eval_growth(y1)
    for _ in range(12):
        dy = -f1 * (y1 - y0) / (f1 - f0)
        y0, f0 = y1, f1
        y1 = y1 + dy
        f1, _ = eval_growth(y1)
        if abs(dy) <= 1e-9 and abs(f1) <= 1e-10:
            break
    # Pairwise influence / jackknife SE via numerical slope at root
    eps_y = 1e-5
    fp, rp = eval_growth(y1 + eps_y)
    fm, rm = eval_growth(y1 - eps_y)
    slope = (fp - fm) / (2.0 * eps_y)
    _, r_root = eval_growth(y1)
    pair_D = y1 - r_root / slope
    se_D = float(np.std(pair_D, ddof=1) / math.sqrt(n_pairs))
    t_crit = float(student_t.ppf(0.975, df=n_pairs - 1))
    return {
        "D_mean": y1,
        "D_se": se_D,
        "D_t95_lo": y1 - t_crit * se_D,
        "D_t95_hi": y1 + t_crit * se_D,
        "root_resid": abs(f1),
    }


def strip_arrays(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cell.items() if not isinstance(v, np.ndarray)}


def main() -> None:
    t0 = time.perf_counter()
    print("=== Step 0: Running Rule 7 Pre-Validation Checks ===")
    val = run_prevalidations()
    print(f"Pre-validations V1 & V2 PASSED in {time.perf_counter() - t0:.2f}s.")

    # ------------------------------------------------------------------
    # Task 1: 400-replicate reproduction of the 4 main-contrast cells
    # phi = 0.6, sigma in {0, 0.6}, steady stay in {1, 20}, log R0 = 0.05, H = 500
    # ------------------------------------------------------------------
    print("\n=== Step 1: 400-replicate reproduction of 4 main-contrast cells (log R0 = 0.05, H = 500) ===")
    cell_specs = [
        ("s0_t1", 0.0, 1.0, 9340000),
        ("s0_t20", 0.0, 20.0, 9341000),
        ("s0.6_t1", 0.6, 1.0, 9342000),
        ("s0.6_t20", 0.6, 20.0, 9343000),
    ]
    main_cells: dict[str, dict[str, Any]] = {}
    d_estimates: dict[str, dict[str, float]] = {}
    for key, sig, stay, seed in cell_specs:
        main_cells[key] = simulate_cell_replicates(
            n_reps=400,
            tau0=2.0,
            tau1=stay,
            sigma=sig,
            phi=0.6,
            log_r0=0.05,
            horizon=500,
            init_counts=None,
            base_seed=seed,
            early_steps=100,
        )
        d_estimates[key] = estimate_threshold_D(
            tau0=2.0,
            tau1=stay,
            sigma=sig,
            phi=0.6,
            base_seed=seed + 500,
        )
        c = main_cells[key]
        d = d_estimates[key]
        print(
            f"  {key:8s} init={c['init_counts']}: ext={c['extinct_count']}/400 = {c['p_ext']:.4f} "
            f"[{c['wilson_95'][0]:.3f}, {c['wilson_95'][1]:.3f}], "
            f"early_g={c['early_growth_mean']:.6f}, D={d['D_mean']:.6f}"
        )

    main_contrast = compute_interaction_summary(
        main_cells["s0.6_t1"],
        main_cells["s0_t1"],
        main_cells["s0.6_t20"],
        main_cells["s0_t20"],
        jeffreys_seed=9390001,
    )
    pilot_est = 0.43
    pilot_ci = [0.319451, 0.540549]
    my_est = main_contrast["interaction"]
    my_ci = main_contrast["interaction_wald_95"]
    intervals_overlap = max(my_ci[0], pilot_ci[0]) <= min(my_ci[1], pilot_ci[1])
    diff_lt_010 = abs(my_est - pilot_est) < 0.10
    print(
        f"  Main interaction = {my_est:.4f} (Wald 95% [{my_ci[0]:.4f}, {my_ci[1]:.4f}], "
        f"Jeffreys 95% [{main_contrast['interaction_jeffreys_95'][0]:.4f}, "
        f"{main_contrast['interaction_jeffreys_95'][1]:.4f}])"
    )
    print(
        f"  Comparison vs pilot.md (0.43, [0.32, 0.54]): |diff|={abs(my_est - pilot_est):.4f} (<0.10: {diff_lt_010}), "
        f"overlap={intervals_overlap}"
    )

    # Exact constant-host PGF numbers at log R0 = 0.05
    pgf_t1_005 = exact_constant_host_pgf_curve(2.0, 1.0, 0.05, (13, 7), 501)
    pgf_t20_005 = exact_constant_host_pgf_curve(2.0, 20.0, 0.05, (2, 18), 3674)
    exact_pgf_005 = {
        "stay1_init": [13, 7],
        "stay20_init": [2, 18],
        "stay1_step68_gen22.67": float(pgf_t1_005[68]),
        "stay1_step69_gen23": float(pgf_t1_005[69]),
        "stay1_step150_gen50": float(pgf_t1_005[150]),
        "stay1_step300_gen100": float(pgf_t1_005[300]),
        "stay1_step500_gen166.67": float(pgf_t1_005[500]),
        "stay1_step501_gen167": float(pgf_t1_005[501]),
        "stay1_asymptotic": exact_asymptotic_extinction(0.05, (13, 7)),
        "stay20_step500_gen22.73": float(pgf_t20_005[500]),
        "stay20_step506_gen23": float(pgf_t20_005[506]),
        "stay20_step1100_gen50": float(pgf_t20_005[1100]),
        "stay20_step2200_gen100": float(pgf_t20_005[2200]),
        "stay20_step3674_gen167": float(pgf_t20_005[3674]),
        "stay20_asymptotic": exact_asymptotic_extinction(0.05, (2, 18)),
    }

    # ------------------------------------------------------------------
    # Task 2a: Confounder audit -- fixed number of generations vs 500 steps
    # T_gen(stay=1) = 3 steps; T_gen(stay=20) = 22 steps.
    # Run fresh 400-rep trajectories out to G = 167 generations (501 steps vs 3674 steps)
    # and evaluate matched G in {23, 50, 100, 167}, plus G = 22.7 from Step 1 (68 vs 500).
    # ------------------------------------------------------------------
    print("\n=== Step 2a: Confounder audit -- matched generation horizons (log R0 = 0.05) ===")
    gen_specs = [
        ("s0_t1", 0.0, 1.0, 501, 9350000),
        ("s0_t20", 0.0, 20.0, 3674, 9351000),
        ("s0.6_t1", 0.6, 1.0, 501, 9352000),
        ("s0.6_t20", 0.6, 20.0, 3674, 9353000),
    ]
    gen_cells: dict[str, dict[str, Any]] = {}
    for key, sig, stay, h_max, seed in gen_specs:
        gen_cells[key] = simulate_cell_replicates(
            n_reps=400,
            tau0=2.0,
            tau1=stay,
            sigma=sig,
            phi=0.6,
            log_r0=0.05,
            horizon=h_max,
            init_counts=None,
            base_seed=seed,
            early_steps=100,
        )

    matched_gen_results: dict[str, Any] = {}
    # Also include G ≈ 22.7 directly from Step 1 trajectories (step 68 vs step 500)
    s1_68_s6 = extract_horizon_slice(main_cells["s0.6_t1"], 68)
    s1_68_s0 = extract_horizon_slice(main_cells["s0_t1"], 68)
    s20_500_s6 = extract_horizon_slice(main_cells["s0.6_t20"], 500)
    s20_500_s0 = extract_horizon_slice(main_cells["s0_t20"], 500)
    matched_gen_results["step1_G22.7_68v500"] = {
        "generations": 500.0 / 22.0,
        "steps_stay1": 68,
        "steps_stay20": 500,
        "cells": {
            "s0_t1": s1_68_s0,
            "s0_t20": s20_500_s0,
            "s0.6_t1": s1_68_s6,
            "s0.6_t20": s20_500_s6,
        },
        "contrast": compute_interaction_summary(
            s1_68_s6, s1_68_s0, s20_500_s6, s20_500_s0, jeffreys_seed=9390010
        ),
    }

    for g_idx, G in enumerate([23, 50, 100, 167]):
        st1 = 3 * G
        st20 = 22 * G
        c6_1 = extract_horizon_slice(gen_cells["s0.6_t1"], st1)
        c0_1 = extract_horizon_slice(gen_cells["s0_t1"], st1)
        c6_20 = extract_horizon_slice(gen_cells["s0.6_t20"], st20)
        c0_20 = extract_horizon_slice(gen_cells["s0_t20"], st20)
        summ = compute_interaction_summary(
            c6_1, c0_1, c6_20, c0_20, jeffreys_seed=9390020 + g_idx
        )
        matched_gen_results[f"G{G}"] = {
            "generations": G,
            "steps_stay1": st1,
            "steps_stay20": st20,
            "cells": {
                "s0_t1": c0_1,
                "s0_t20": c0_20,
                "s0.6_t1": c6_1,
                "s0.6_t20": c6_20,
            },
            "contrast": summ,
        }
        print(
            f"  G={G:3d} ({st1:3d} vs {st20:4d} steps): "
            f"Pext(0,1)={c0_1['p_ext']:.3f}, Pext(.6,1)={c6_1['p_ext']:.3f} (d1={summ['noise_effect_stay1']:.3f}); "
            f"Pext(0,20)={c0_20['p_ext']:.3f}, Pext(.6,20)={c6_20['p_ext']:.3f} (d20={summ['noise_effect_stay20']:.3f}); "
            f"interaction={summ['interaction']:.4f} [{summ['interaction_wald_95'][0]:.4f}, {summ['interaction_wald_95'][1]:.4f}]"
        )

    # ------------------------------------------------------------------
    # Task 2b: Confounder audit -- rerun 4 cells at log R0 = 0.2 (400 reps, H = 500)
    # ------------------------------------------------------------------
    print("\n=== Step 2b: Confounder audit -- log R0 = 0.2 (400 replicates/cell, H = 500) ===")
    r02_specs = [
        ("s0_t1", 0.0, 1.0, 9360000),
        ("s0_t20", 0.0, 20.0, 9361000),
        ("s0.6_t1", 0.6, 1.0, 9362000),
        ("s0.6_t20", 0.6, 20.0, 9363000),
    ]
    r02_cells: dict[str, dict[str, Any]] = {}
    for key, sig, stay, seed in r02_specs:
        r02_cells[key] = simulate_cell_replicates(
            n_reps=400,
            tau0=2.0,
            tau1=stay,
            sigma=sig,
            phi=0.6,
            log_r0=0.2,
            horizon=500,
            init_counts=None,
            base_seed=seed,
            early_steps=100,
        )
        c = r02_cells[key]
        print(
            f"  {key:8s} init={c['init_counts']}: ext={c['extinct_count']}/400 = {c['p_ext']:.4f} "
            f"[{c['wilson_95'][0]:.3f}, {c['wilson_95'][1]:.3f}], "
            f"early_g={c['early_growth_mean']:.6f}"
        )

    r02_contrast = compute_interaction_summary(
        r02_cells["s0.6_t1"],
        r02_cells["s0_t1"],
        r02_cells["s0.6_t20"],
        r02_cells["s0_t20"],
        jeffreys_seed=9390099,
    )
    init_t1_02 = stable_initial_counts(20, 2.0, 1.0, 0.2)
    init_t20_02 = stable_initial_counts(20, 2.0, 20.0, 0.2)
    exact_pgf_020 = {
        "stay1_init": list(init_t1_02),
        "stay20_init": list(init_t20_02),
        "stay1_step500": float(exact_constant_host_pgf_curve(2.0, 1.0, 0.2, init_t1_02, 500)[500]),
        "stay1_asymptotic": exact_asymptotic_extinction(0.2, init_t1_02),
        "stay20_step500": float(exact_constant_host_pgf_curve(2.0, 20.0, 0.2, init_t20_02, 500)[500]),
        "stay20_asymptotic": exact_asymptotic_extinction(0.2, init_t20_02),
    }
    print(
        f"  log R0 = 0.2 interaction = {r02_contrast['interaction']:.4f} "
        f"(Wald 95% [{r02_contrast['interaction_wald_95'][0]:.4f}, {r02_contrast['interaction_wald_95'][1]:.4f}], "
        f"Jeffreys 95% [{r02_contrast['interaction_jeffreys_95'][0]:.4f}, "
        f"{r02_contrast['interaction_jeffreys_95'][1]:.4f}])"
    )

    elapsed = time.perf_counter() - t0
    payload = {
        "elapsed_seconds": elapsed,
        "prevalidations": val,
        "task1_main_reproduction": {
            "cells": {k: strip_arrays(v) for k, v in main_cells.items()},
            "threshold_D": d_estimates,
            "contrast": main_contrast,
            "exact_constant_host_pgf": exact_pgf_005,
            "agreement_with_pilot_md": {
                "pilot_interaction": pilot_est,
                "pilot_ci95": pilot_ci,
                "secondary_interaction": my_est,
                "secondary_wald_ci95": my_ci,
                "abs_diff": abs(my_est - pilot_est),
                "diff_less_than_0_10": diff_lt_010,
                "intervals_overlap": intervals_overlap,
                "agreed": bool(intervals_overlap and diff_lt_010),
            },
        },
        "task2a_matched_generations": matched_gen_results,
        "task2b_log_r0_0_2": {
            "cells": {k: strip_arrays(v) for k, v in r02_cells.items()},
            "contrast": r02_contrast,
            "exact_constant_host_pgf": exact_pgf_020,
        },
    }
    out_json = OUT_DIR / "reproduce_and_audit_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json.name} in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
