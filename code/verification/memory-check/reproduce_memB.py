#!/usr/bin/env python3
"""Independent reproduction and audit of studies/memory-memB.

Invoke with:
    python3 -I verification/memory-check/reproduce_memB.py

Rules followed:
- Does NOT import memB.py or any project module. Uses only Python stdlib, numpy, scipy, mpmath.
- Sets BLAS/OpenMP thread env vars to 1 and uses at most 4 worker processes.
- Step 0 (Pre-validation, tolerances stated beforehand):
  * Deterministic 2-stage, 3-stage, and 12-step seasonal Floquet growth exponents (tol = 1e-12).
  * Six G4 independent-gamma benchmarks against 35-digit mpmath Tricomi exact roots and printed
    values in results/exact-gamma-parking_20261114.txt (printed rounding tol = 5.01e-6; fresh
    Monte Carlo tol = 4 SE).
- Step 1 (Full CSV audit of memB_tables.csv and memB_summary.csv):
  * Verifies all 4,320 rows, precision target SE_D <= max(5e-5, 0.02*|D|), curve counts and
    4-paired-SE monotonicity classifications (112/112 primary 2-stage decreasing, 8/8 matched-gamma
    decreasing, 15/24 3-stage non-monotone, 9/24 3-stage decreasing).
  * Checks all 120 independent-gamma (phi=0, ring=2) rows of memB_tables.csv against independent
    35-digit mpmath Tricomi roots (requiring all |D_memB - D_exact| <= 4 SE_memB).
- Step 2 (Independent Monte Carlo reproduction of 21 curves / 123 cells of memB_tables.csv):
  * Spread across all four host models (lognormal, telegraph, gamma, seasonal, plus gamma_matched)
    and both ring lengths (ring=2 and ring=3), using completely fresh random seeds.
  * Includes all four (plus matched-gamma) two-stage curves at strongest memory and largest noise
    (a = 1.5, phi = 0.99), both repaired cells of memB (telegraph a=1, phi=0.99; seasonal a=1.5, phi=0.6),
    five non-monotone three-stage curves across all four host models, and one decreasing high-memory
    three-stage curve (telegraph a=1.5, phi=0.90).
  * Requires pointwise agreement within 4 combined standard errors:
        |D_secondary - D_memB| <= 4 * sqrt(SE_secondary^2 + SE_memB^2),
    and checks global z-score distribution across all 123 cells.
  * Writes machine-readable reproduction table to verification/memory-check/memB_reproduction_cells.csv.
"""

import os
for _env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_env] = "1"

import concurrent.futures as cf
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy import special
import mpmath as mp

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
PROB_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
MEMB_CSV = PROB_ROOT / "studies" / "memory-memB" / "memB_tables.csv"
MEMB_SUM = PROB_ROOT / "studies" / "memory-memB" / "memB_summary.csv"
OUT_CSV = HERE / "memB_reproduction_cells.csv"

STEPS = 131076
BURN = 4104
FAILURES = []


def report(tag, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {tag}: {detail}", flush=True)
    if not ok:
        FAILURES.append(tag)


# ---------------------------------------------------------------------------
# Exact independent-gamma root via 35-digit mpmath Tricomi U(p, b, kappa)
# ---------------------------------------------------------------------------

def tricomi_gamma_exact_D(tau, p, dps=35):
    """Exact root y of ln(s) + d/db ln U(p, b, s^2 * p * tau * e^{-y})|_{b=1} = 0."""
    mp.mp.dps = dps
    tt = mp.mpf(repr(float(tau)))
    pp = mp.mpf(repr(float(p)))
    ss = 1 - 1 / tt
    ell = mp.log(pp) - mp.digamma(pp)

    # Initial guess via second-order or mid-bound, polished by secant in mpmath
    q2 = 1 / (2 * pp * (1 + tt))
    y_init = min(ell * mp.mpf("0.85"), max(mp.mpf("1e-10"), q2))

    def rate_at(y):
        kap = ss * ss * pp * tt * mp.exp(-y)
        # Use integral in u = log(t) around mode when kap or p is extreme, or mp.hyperu directly:
        # mp.hyperu(p, b, kap) is exact in 35-digit arithmetic across moderate p, kap!
        u0 = mp.hyperu(pp, mp.mpf(1), kap)
        eps = mp.mpf("1e-14")
        up = mp.hyperu(pp, 1 + eps, kap)
        um = mp.hyperu(pp, 1 - eps, kap)
        dlog_u = (mp.log(up) - mp.log(um)) / (2 * eps)
        return mp.log(ss) + dlog_u

    # Bracket cleanly between 0 and ell
    lo, hi = mp.mpf(0), ell
    # 12 bisection steps to get tight bracket, then secant to 10^-26
    for _ in range(14):
        mid = (lo + hi) / 2
        if rate_at(mid) < 0:
            lo = mid
        else:
            hi = mid
    root = mp.findroot(rate_at, (lo, hi), solver="secant", tol=mp.mpf("1e-25"))
    return float(root)


# ---------------------------------------------------------------------------
# Independent host environment generator (antithetic pairs)
# ---------------------------------------------------------------------------

def host_moments(model, a, phi):
    """Exact stationary host variance V and host lag-1 autocorrelation rho."""
    v_ln = math.expm1(a * a)
    rho_ln = math.expm1(a * a * phi) / v_ln if v_ln > 0 else phi
    if model == "gamma":
        return a * a, phi
    if model == "gamma_matched":
        return v_ln, rho_ln
    if model in ("lognormal", "telegraph"):
        return v_ln, rho_ln
    if model == "seasonal":
        v_s = 1.125 * math.exp(a * a) - 1.0
        rho_s = ((1.0 + 0.125 * math.cos(2.0 * math.pi / 12.0)) * math.exp(a * a * phi) - 1.0) / v_s
        return v_s, rho_s
    raise ValueError(f"Unknown model: {model}")


def sample_host_batch(model, a, phi, n_total, n_pairs, seed):
    """Generate (n_total, 2 * n_pairs) host array and (n_total,) deterministic mean baseline."""
    rng = np.random.default_rng(seed)
    if model in ("lognormal", "seasonal"):
        z = np.empty((n_total, n_pairs), dtype=np.float64)
        z[0] = rng.standard_normal(n_pairs)
        if phi == 0.0:
            z[1:] = rng.standard_normal((n_total - 1, n_pairs))
        else:
            scale = math.sqrt(1.0 - phi * phi)
            innov = rng.standard_normal((n_total - 1, n_pairs))
            for t in range(n_total - 1):
                z[t + 1] = phi * z[t] + scale * innov[t]
        h = np.empty((n_total, 2 * n_pairs), dtype=np.float64)
        shift = 0.5 * a * a
        h[:, 0::2] = np.exp(a * z - shift)
        h[:, 1::2] = np.exp(-a * z - shift)
        if model == "seasonal":
            season = 1.0 + 0.5 * np.cos(2.0 * math.pi * np.arange(n_total) / 12.0)
            h *= season[:, None]
            return h, season
        return h, np.ones(n_total, dtype=np.float64)

    V, rho = host_moments(model, a, phi)
    u_keep = rng.random((n_total, n_pairs))
    u_draw = rng.random((n_total, n_pairs))
    h = np.empty((n_total, 2 * n_pairs), dtype=np.float64)
    for anti in (0, 1):
        uk = u_keep if anti == 0 else (1.0 - u_keep)
        ud = u_draw if anti == 0 else (1.0 - u_draw)
        if model == "telegraph":
            d = min(math.sqrt(V), 0.8)
            p_hi = (d * d) / (V + d * d)
            fresh = np.where(ud < p_hi, 1.0 + V / d, 1.0 - d)
        else:  # gamma or gamma_matched: shape p = 1/V, scale V
            p_shape = 1.0 / V
            fresh = special.gammaincinv(p_shape, ud) * V
        col = np.empty((n_total, n_pairs), dtype=np.float64)
        col[0] = fresh[0]
        if rho == 0.0:
            col[1:] = fresh[1:]
        else:
            for t in range(1, n_total):
                keep_mask = uk[t] < rho
                col[t] = np.where(keep_mask, col[t - 1], fresh[t])
        h[:, anti::2] = col
    return h, np.ones(n_total, dtype=np.float64)


# ---------------------------------------------------------------------------
# Independent Lyapunov growth & pathwise derivative evaluator (ring = 2 or 3)
# ---------------------------------------------------------------------------

def eval_lyapunov_batch(h, baseline, taus, y_vec, burn, ring):
    """Evaluate controlled growth rate and d(rate)/dy across paths and stays."""
    taus = np.asarray(taus, dtype=np.float64)
    y_vec = np.asarray(y_vec, dtype=np.float64)
    n_tot, n_paths = h.shape
    n_rec = n_tot - burn
    n_taus = len(taus)
    T_gen = taus + (1.0 if ring == 2 else 6.0)

    # Mean-zero control variate: (1/n_rec) sum_{t>=burn} (h_t - E[h_t]) / T_gen
    cv_adj = ((h[burn:] - baseline[burn:, None]).mean(axis=0))[:, None] / T_gen[None, :]

    sum_log = np.zeros((n_paths, n_taus), dtype=np.float64)
    sum_der = np.zeros((n_paths, n_taus), dtype=np.float64)
    s_stay = 1.0 - 1.0 / taus

    if ring == 2:
        # u_t = x1(t+1)/x1(t) = s + c * h_t / u_{t-1}, c = (1 - s)*exp(y) = exp(y)/tau
        c_vec = np.exp(y_vec) / taus
        u = np.ones((n_paths, n_taus), dtype=np.float64)
        du_u = np.zeros((n_paths, n_taus), dtype=np.float64)  # d(log u)/dy
        for t in range(n_tot):
            q = (h[t][:, None] * c_vec[None, :]) / u
            u = s_stay[None, :] + q
            du_u = (q / u) * (1.0 - du_u)
            if t >= burn:
                sum_log += np.log(u)
                sum_der += du_u
    else:
        # 3-stage ring (1, tau, 5):
        # x0' = 0.2 * x2
        # x1' = exp(y)*h_t * x0 + (1 - 1/tau) * x1
        # x2' = (1/tau) * x1 + 0.8 * x2
        inv_tau = 1.0 / taus
        ey = np.exp(y_vec)
        x0 = np.broadcast_to(1.0 / T_gen, (n_paths, n_taus)).copy()
        x1 = np.broadcast_to(taus / T_gen, (n_paths, n_taus)).copy()
        x2 = np.broadcast_to(5.0 / T_gen, (n_paths, n_taus)).copy()
        # Track tangent vector dx/dy for normalized state x0 + x1 + x2 = 1
        dx0 = np.zeros_like(x0)
        dx1 = np.zeros_like(x1)
        dx2 = np.zeros_like(x2)
        for t in range(n_tot):
            g0h = h[t][:, None] * ey[None, :]
            nx0 = 0.2 * x2
            nx1 = g0h * x0 + s_stay[None, :] * x1
            nx2 = inv_tau[None, :] * x1 + 0.8 * x2

            ndx0 = 0.2 * dx2
            ndx1 = g0h * (dx0 + x0) + s_stay[None, :] * dx1
            ndx2 = inv_tau[None, :] * dx1 + 0.8 * dx2

            tot = nx0 + nx1 + nx2
            dlog_tot = (ndx0 + ndx1 + ndx2) / tot
            inv_tot = 1.0 / tot
            x0 = nx0 * inv_tot
            x1 = nx1 * inv_tot
            x2 = nx2 * inv_tot
            dx0 = ndx0 * inv_tot - x0 * dlog_tot
            dx1 = ndx1 * inv_tot - x1 * dlog_tot
            dx2 = ndx2 * inv_tot - x2 * dlog_tot
            if t >= burn:
                sum_log += np.log(tot)
                sum_der += dlog_tot

    r_ctrl = (sum_log / n_rec) - cv_adj
    dr_dy = sum_der / n_rec
    return r_ctrl, dr_dy


def solve_curve_mc(task):
    """Solve D(tau) by Newton root-finding on pooled antithetic batches with common random numbers."""
    model, a, phi, ring, taus, y_init, n_batches, pairs_per_batch, seed_base = task
    t0 = time.monotonic()
    taus = np.asarray(taus, dtype=np.float64)
    y = np.asarray(y_init, dtype=np.float64).copy()

    # Cache single-batch environments in memory; regenerate multi-batch deterministically per step
    cached = sample_host_batch(model, a, phi, STEPS + BURN, pairs_per_batch, seed_base) if n_batches == 1 else None

    for it in range(8):
        pair_r_list = []
        pair_d_list = []
        for b in range(n_batches):
            h, base = cached if cached is not None else sample_host_batch(
                model, a, phi, STEPS + BURN, pairs_per_batch, seed_base + 7919 * b
            )
            rc, dr = eval_lyapunov_batch(h, base, taus, y, BURN, ring)
            pair_r_list.append(0.5 * (rc[0::2] + rc[1::2]))
            pair_d_list.append(0.5 * (dr[0::2] + dr[1::2]))
            if cached is None:
                del h, base
        pr = np.concatenate(pair_r_list, axis=0)
        pd = np.concatenate(pair_d_list, axis=0)
        n_pairs_tot = pr.shape[0]
        r_bar = pr.mean(axis=0)
        slope = pd.mean(axis=0)
        se_r = pr.std(axis=0, ddof=1) / math.sqrt(n_pairs_tot)
        step = r_bar / slope
        se_D = se_r / slope
        y -= np.clip(step, -1.5, 1.5)
        if np.all(np.abs(step) < np.maximum(1e-9, 0.03 * se_D)):
            break

    infl = -(pr - r_bar[None, :]) / slope[None, :]
    return {
        "model": model,
        "a": a,
        "phi": phi,
        "ring": ring,
        "taus": taus,
        "D": y,
        "SE": se_D,
        "influence": infl,
        "pairs": n_pairs_tot,
        "seconds": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Step 0: Deterministic & G4 Exact Gamma Pre-Validation
# ---------------------------------------------------------------------------

def run_prevalidation():
    print("--- Step 0. Pre-validation of independent code against exact benchmarks ---")
    # 0a. Deterministic 2-stage, 3-stage, and 12-step seasonal Floquet matrix products (tol = 1e-12)
    taus_test = np.array([1.0 / 0.99, 2.0, 10.0, 100.0])
    y_test = np.full(4, 0.17)
    h_one = np.ones((10000, 2))
    base_one = np.ones(10000)
    max_det_err = 0.0
    for ring in (2, 3):
        rc, _ = eval_lyapunov_batch(h_one, base_one, taus_test, y_test, 8000, ring)
        for j, tau in enumerate(taus_test):
            if ring == 2:
                M2 = np.array([[1.0 - 1.0 / tau, math.exp(0.17) / tau], [1.0, 0.0]])
                ex = math.log(max(abs(np.linalg.eigvals(M2))))
            else:
                M3 = np.array([[0.0, 0.0, 0.2],
                               [math.exp(0.17), 1.0 - 1.0 / tau, 0.0],
                               [0.0, 1.0 / tau, 0.8]])
                ex = math.log(max(abs(np.linalg.eigvals(M3))))
            max_det_err = max(max_det_err, abs(rc[0, j] - ex))

    season = 1.0 + 0.5 * np.cos(2.0 * math.pi * np.arange(12000) / 12.0)
    h_seas = np.repeat(season[:, None], 2, axis=1)
    rc_s, _ = eval_lyapunov_batch(h_seas, season, taus_test, y_test, 9600, 2)
    for j, tau in enumerate(taus_test):
        P12 = np.eye(2)
        for m in range(12):
            P12 = np.array([[1.0 - 1.0 / tau, math.exp(0.17) * season[m] / tau], [1.0, 0.0]]) @ P12
        ex_s = math.log(max(abs(np.linalg.eigvals(P12)))) / 12.0
        max_det_err = max(max_det_err, abs(rc_s[0, j] - ex_s))
    report("V0_deterministic_and_floquet", max_det_err < 1e-12, f"max error = {max_det_err:.3e} (tol 1e-12)")

    # 0b. Six G4 exact independent-gamma benchmarks (results/exact-gamma-parking_20261114.txt)
    g4_cases = [
        (2.0, 1.0, 0.16991),
        (10.0, 0.5, 0.08555),
        (5.0, 2.0, 0.04091),
        (25.0, 1.0, 0.01888),
        (1.0 / 0.7, 4.0, 0.05399),
        (10.0, 0.25, 0.16303),
    ]
    g4_tasks = []
    exact_vals = []
    for idx, (tau, p, printed) in enumerate(g4_cases):
        ex = tricomi_gamma_exact_D(tau, p, dps=35)
        exact_vals.append(ex)
        g4_tasks.append(("gamma", 1.0 / math.sqrt(p), 0.0, 2, [tau], [ex], 1, 32, 880100 + 37 * idx))

    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        g4_res = list(pool.map(solve_curve_mc, g4_tasks))

    g4_ok = True
    max_g4_z = 0.0
    for (tau, p, printed), ex, res in zip(g4_cases, exact_vals, g4_res):
        d_mc = float(res["D"][0])
        se_mc = float(res["SE"][0])
        z = abs(d_mc - ex) / se_mc
        max_g4_z = max(max_g4_z, z)
        ok_i = (z <= 4.0) and (abs(ex - printed) <= 5.01e-6)
        g4_ok = g4_ok and ok_i
        print(f"  G4 tau={tau:8.4f}, p={p:4.2f}: Tricomi={ex:.8f} (printed {printed:.5f}), "
              f"secondary_MC={d_mc:.8f} +- {se_mc:.6f}, |z|={z:.2f}")
    report("V1_G4_exact_gamma_benchmarks", g4_ok, f"all 6 pass; max |z| = {max_g4_z:.2f} <= 4")


# ---------------------------------------------------------------------------
# Step 1: Audit memB_tables.csv and check all 120 independent-gamma rows
# ---------------------------------------------------------------------------

def audit_memB_csv():
    print("\n--- Step 1. Full audit of memB_tables.csv (4,320 rows) & exact gamma grid (120 rows) ---")
    with MEMB_CSV.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    with MEMB_SUM.open("r", newline="") as f:
        srows = list(csv.DictReader(f))

    n_rows = len(rows)
    n_sum = len(srows)
    prec_ok_count = 0
    by_curve = {}
    for r in rows:
        key = (r["model"], float(r["amplitude"]), float(r["phi_input"]), int(r["ring"]))
        by_curve.setdefault(key, []).append(r)
        D = float(r["D"])
        se = float(r["SE_D"])
        target = max(5e-5, 0.02 * abs(D))
        if se <= target:
            prec_ok_count += 1

    # Recompute curve monotonicity classifications directly from next_D_difference and next_difference_SE
    prim_counts = {"decrease": 0, "nonmonotone": 0, "unresolved": 0}
    matched_counts = {"decrease": 0, "nonmonotone": 0, "unresolved": 0}
    ring3_counts = {"decrease": 0, "nonmonotone": 0, "unresolved": 0}

    for key, crows in by_curve.items():
        crows_sorted = sorted(crows, key=lambda x: float(x["tau"]))
        ups = 0
        downs = 0
        for r in crows_sorted[:-1]:
            diff = float(r["next_D_difference"])
            dse = float(r["next_difference_SE"])
            if diff > 4.0 * dse:
                ups += 1
            elif diff < -4.0 * dse:
                downs += 1
        n_edges = len(crows_sorted) - 1
        st = "decrease" if downs == n_edges else ("nonmonotone" if (ups > 0 and downs > 0) else "unresolved")
        if key[3] == 2 and key[0] != "gamma_matched":
            prim_counts[st] += 1
        elif key[3] == 2 and key[0] == "gamma_matched":
            matched_counts[st] += 1
        else:
            ring3_counts[st] += 1

    report("A1_csv_counts_and_precision",
           n_rows == 4320 and n_sum == 144 and prec_ok_count == 4320,
           f"rows={n_rows}, curves={n_sum}, precision_pass={prec_ok_count}/4320")
    report("A2_curve_classifications",
           prim_counts == {"decrease": 112, "nonmonotone": 0, "unresolved": 0}
           and matched_counts == {"decrease": 8, "nonmonotone": 0, "unresolved": 0}
           and ring3_counts == {"decrease": 9, "nonmonotone": 15, "unresolved": 0},
           f"primary_2stage={prim_counts}, gamma_matched={matched_counts}, ring3={ring3_counts}")

    # Check all 120 gamma, phi=0, ring=2 rows against 35-digit mpmath Tricomi exact roots!
    # Also check the 60 gamma_matched, phi=0, ring=2 rows (2 amplitudes x 30 stays = 60 rows)!
    gamma_phi0_rows = [r for r in rows if r["model"] == "gamma" and float(r["phi_input"]) == 0.0 and int(r["ring"]) == 2]
    gmatch_phi0_rows = [r for r in rows if r["model"] == "gamma_matched" and float(r["phi_input"]) == 0.0 and int(r["ring"]) == 2]

    max_z_gamma120 = 0.0
    max_quad_diff = 0.0
    for r in gamma_phi0_rows:
        tau = float(r["tau"])
        amp = float(r["amplitude"])
        p = 1.0 / (amp * amp)
        ex = tricomi_gamma_exact_D(tau, p, dps=35)
        d_memb = float(r["D"])
        se_memb = float(r["SE_D"])
        z = abs(d_memb - ex) / se_memb
        max_z_gamma120 = max(max_z_gamma120, z)
        if r["gamma_exact"] != "":
            max_quad_diff = max(max_quad_diff, abs(float(r["gamma_exact"]) - ex))

    max_z_gmatch60 = 0.0
    for r in gmatch_phi0_rows:
        tau = float(r["tau"])
        v_host = float(r["host_variance"])
        p = 1.0 / v_host
        ex = tricomi_gamma_exact_D(tau, p, dps=35)
        z = abs(float(r["D"]) - ex) / float(r["SE_D"])
        max_z_gmatch60 = max(max_z_gmatch60, z)

    report("A3_exact_gamma_120_plus_60_grid",
           (len(gamma_phi0_rows) == 120) and (max_z_gamma120 <= 4.0) and (max_z_gmatch60 <= 4.0) and (max_quad_diff < 1e-11),
           f"120 gamma(phi=0) max |z| = {max_z_gamma120:.3f} SE, max |memB_quad - Tricomi| = {max_quad_diff:.3e}; "
           f"60 gamma_matched(phi=0) max |z| = {max_z_gmatch60:.3f} SE")

    return by_curve


# ---------------------------------------------------------------------------
# Step 2: Independent Monte Carlo reproduction of 21 curves (123 CSV cells)
# ---------------------------------------------------------------------------

def run_reproduction(by_curve):
    print("\n--- Step 2. Independent reproduction of 21 curves (123 cells of memB_tables.csv) ---")
    # Target stays for ring=2 (5 stays spanning entire range 1.01 to 1000)
    taus_r2_targets = [1.0 / 0.99, 2.0, 10.0, 100.0, 1000.0]
    # Target stays for ring=3 (8 stays resolving the short-stay rise, peak, and long-stay tail)
    taus_r3_targets = [1.0 / 0.99, 1.7947380553319678, 2.0, 2.3923196347875697,
                       3.1888738403842054, 4.250651218180415, 10.0, 1000.0]

    # 21 curves covering all 4 host models (+ gamma_matched), both ring=2 and ring=3,
    # strongest memory & largest noise, repaired cells, and 5 non-monotone 3-stage curves:
    specs = [
        # (model, a, phi, ring, n_batches, pairs_per_batch)
        ("lognormal", 0.3, 0.6, 2, 1, 64),
        ("lognormal", 0.6, 0.9, 2, 1, 64),
        ("lognormal", 1.0, 0.99, 2, 2, 64),
        ("lognormal", 1.5, 0.99, 2, 4, 64),       # largest noise, strongest memory
        ("telegraph", 0.3, 0.3, 2, 1, 64),
        ("telegraph", 0.6, 0.8, 2, 1, 64),
        ("telegraph", 1.0, 0.99, 2, 2, 64),       # repaired cell in memB
        ("telegraph", 1.5, 0.99, 2, 2, 64),       # largest noise, strongest memory
        ("gamma", 0.6, 0.9, 2, 1, 64),
        ("gamma", 1.0, 0.6, 2, 1, 64),
        ("gamma", 1.5, 0.99, 2, 2, 64),           # largest noise, strongest memory
        ("gamma_matched", 1.5, 0.99, 2, 2, 64),   # matched-moment supplement, largest noise & memory
        ("seasonal", 0.3, 0.95, 2, 1, 64),
        ("seasonal", 1.5, 0.6, 2, 2, 64),         # repaired cell in memB
        ("seasonal", 1.5, 0.99, 2, 4, 64),        # largest noise, strongest memory
        # Three-stage curves (ring = 3): 5 non-monotone + 1 high-noise decreasing
        ("lognormal", 0.6, 0.99, 3, 2, 64),       # non-monotone
        ("lognormal", 1.5, 0.99, 3, 4, 64),       # non-monotone (largest noise, strongest memory)
        ("telegraph", 0.6, 0.99, 3, 1, 64),       # non-monotone
        ("telegraph", 1.5, 0.9, 3, 1, 64),        # finite-noise decreasing vs 2nd-order peak 1.68
        ("gamma", 1.5, 0.99, 3, 2, 64),           # non-monotone (largest noise, strongest memory)
        ("seasonal", 1.5, 0.99, 3, 4, 64),        # non-monotone (largest noise, strongest memory)
    ]

    tasks = []
    ref_lookup = {}
    for idx, (model, a, phi, ring, nb, ppb) in enumerate(specs):
        crows = by_curve[(model, a, phi, ring)]
        targets = taus_r2_targets if ring == 2 else taus_r3_targets
        selected_rows = []
        for t_want in targets:
            best_row = min(crows, key=lambda r: abs(float(r["tau"]) - t_want))
            assert abs(float(best_row["tau"]) - t_want) < 1e-9, f"tau mismatch: {best_row['tau']} vs {t_want}"
            selected_rows.append(best_row)
        taus_exact = [float(r["tau"]) for r in selected_rows]
        # Independent initial guess: simple order-of-magnitude constant 0.05 (NOT memB's D!)
        y_init = [0.05] * len(taus_exact)
        seed_base = 5500000 + 9187 * idx
        tasks.append((model, a, phi, ring, taus_exact, y_init, nb, ppb, seed_base))
        ref_lookup[(model, a, phi, ring)] = selected_rows

    t_start = time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(solve_curve_mc, tasks))
    elapsed = time.monotonic() - t_start
    print(f"  Simulated 21 curves ({sum(len(r['taus']) for r in results)} cells) in {elapsed:.2f}s across 4 workers.")

    out_records = []
    max_abs_z = 0.0
    n_within_4se = 0
    n_within_2se = 0

    # Also verify non-monotonicity / decrease on each reproduced curve!
    curve_shape_checks = []
    for res in results:
        key = (res["model"], res["a"], res["phi"], res["ring"])
        ref_rows = ref_lookup[key]
        for j, r_ref in enumerate(ref_rows):
            tau = float(res["taus"][j])
            d_carb = float(res["D"][j])
            se_carb = float(res["SE"][j])
            d_memb = float(r_ref["D"])
            se_memb = float(r_ref["SE_D"])
            comb_se = math.hypot(se_carb, se_memb)
            z = (d_carb - d_memb) / comb_se
            max_abs_z = max(max_abs_z, abs(z))
            if abs(z) <= 4.0:
                n_within_4se += 1
            if abs(z) <= 2.0:
                n_within_2se += 1
            out_records.append({
                "model": res["model"],
                "amplitude": res["a"],
                "phi": res["phi"],
                "ring": res["ring"],
                "tau": tau,
                "memB_D": d_memb,
                "memB_SE": se_memb,
                "secondary_D": d_carb,
                "secondary_SE": se_carb,
                "combined_SE": comb_se,
                "diff_secondary_minus_memB": d_carb - d_memb,
                "z_combined": z,
                "secondary_pairs": res["pairs"],
                "memB_pairs": int(r_ref["independent_pairs"]),
            })

        # Check curve shape (decrease for ring=2 and telegraph 1.5/0.9 ring=3; non-monotone for the 5 ring=3 curves)
        D_vec = res["D"]
        infl = res["influence"]
        n_p = infl.shape[0]
        if res["ring"] == 2 or (res["model"] == "telegraph" and res["a"] == 1.5 and res["phi"] == 0.9):
            diffs = np.diff(D_vec)
            diff_ses = np.diff(infl, axis=1).std(axis=0, ddof=1) / math.sqrt(n_p)
            all_down = bool(np.all(diffs < -4.0 * diff_ses))
            curve_shape_checks.append((key, "decrease", all_down, float(np.max(diffs / diff_ses))))
            print(f"  Curve {key}: secondary status={'decrease' if all_down else 'NOT_ALL_DOWN'} "
                  f"(worst edge z = {np.max(diffs / diff_ses):.2f}), D[1.01]={D_vec[0]:.6f} -> D[1000]={D_vec[-1]:.6f}")
        else:
            pk = int(np.argmax(D_vec))
            rise = float(D_vec[pk] - D_vec[0])
            rise_se = float(np.std(infl[:, pk] - infl[:, 0], ddof=1) / math.sqrt(n_p))
            fall = float(D_vec[-1] - D_vec[pk])
            fall_se = float(np.std(infl[:, -1] - infl[:, pk], ddof=1) / math.sqrt(n_p))
            is_nonmono = (pk > 0) and (pk < len(D_vec) - 1) and (rise > 4.0 * rise_se) and (fall < -4.0 * fall_se)
            curve_shape_checks.append((key, "nonmonotone", is_nonmono, rise / rise_se))
            print(f"  Curve {key}: secondary status={'nonmonotone' if is_nonmono else 'UNRESOLVED'} "
                  f"peak tau={res['taus'][pk]:.2f}, rise=+{rise:.6f} ({rise/rise_se:.1f} SE), "
                  f"fall={fall:.6f} ({fall/fall_se:.1f} SE)")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_records[0].keys()))
        writer.writeheader()
        writer.writerows(out_records)

    n_tot_cells = len(out_records)
    shapes_ok = all(ok for _, _, ok, _ in curve_shape_checks)
    report("R1_123_cells_within_4_combined_SE",
           n_within_4se == n_tot_cells,
           f"{n_within_4se}/{n_tot_cells} within 4 combined SE ({n_within_2se}/{n_tot_cells} within 2 SE); max |z| = {max_abs_z:.3f}")
    report("R2_reproduced_curve_shapes",
           shapes_ok,
           f"all 15 two-stage curves + telegraph(1.5,0.9,ring=3) strictly decrease (>4 paired SE); "
           f"all 5 non-monotone three-stage curves rise >4 paired SE then fall >4 paired SE")


def main():
    t0 = time.monotonic()
    print("=== secondary Independent Reproduction and Audit of memB ===")
    run_prevalidation()
    by_curve = audit_memB_csv()
    run_reproduction(by_curve)
    print(f"\nTotal runtime: {time.monotonic() - t0:.2f}s. Failures: {len(FAILURES)} {FAILURES}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
