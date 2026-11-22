#!/usr/bin/env python3
"""Independent second-party reproduction and Rule 13 departure analysis for Q4-1.

Zero imports from virtualworld/q4/ or virtualworld/world/.
"""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import openpyxl
from scipy import optimize, stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
Q4_DIR = ROOT / "virtualworld/q4"


# ---------------------------------------------------------------------------
# 1. Standalone ring operator, circular-replay Perron root, PGF, and theory
# ---------------------------------------------------------------------------

def stage_means(k: int, s: int) -> tuple[float, ...]:
    if k == 2:
        return (2.0, float(s))
    if k == 3:
        return (2.0, 2.0, float(s))
    raise ValueError(f"Unsupported stage count k={k}")


def step_matrix(taus: tuple[float, ...], h_val: float, y: float) -> np.ndarray:
    """Single-step projection matrix for a primitive waste-mode geometric ring.

    Stage 0 has transfer yield multiplied by h_val; stage k-1 has birth tilt exp(y).
    At h_val = 1 and y = 0, every column sums to 1 so spectral radius is 1.
    """
    k = len(taus)
    A = np.zeros((k, k), dtype=np.float64)
    for i, tau in enumerate(taus):
        stay = 1.0 - 1.0 / tau
        leave = 1.0 / tau
        A[i, i] = stay
        nxt = (i + 1) % k
        if i == 0:
            A[nxt, i] = leave * h_val
        elif i == k - 1:
            A[nxt, i] = leave * math.exp(y)
        else:
            A[nxt, i] = leave
    return A


def replay_log_growth(taus: tuple[float, ...], h: np.ndarray, y: float) -> float:
    """Exact per-step asymptotic log growth of circular replay of finite record h."""
    k = len(taus)
    M = np.eye(k, dtype=np.float64)
    log_scale = 0.0
    for h_val in h:
        M = step_matrix(taus, float(h_val), y) @ M
        norm = float(np.max(np.abs(M)))
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("Degenerate monodromy matrix")
        M /= norm
        log_scale += math.log(norm)
    eigvals = np.linalg.eigvals(M)
    # Perron root is real positive for primitive non-negative monodromy
    real_pos = [float(ev.real) for ev in eigvals if abs(ev.imag) < 1e-9 and ev.real > 0]
    rho = max(real_pos) if real_pos else float(np.max(eigvals.real))
    if rho <= 0.0:
        raise FloatingPointError("Non-positive Perron root")
    return (math.log(rho) + log_scale) / len(h)


def solve_replay_D(k: int, s: int, h: np.ndarray) -> tuple[float, float]:
    """Solve for log birth tilt D making circular-replay Perron root equal to 1.

    At constant host h == 1, critical tilt is identically 0.
    Returns (D, growth_residual_at_D).
    """
    taus = stage_means(k, s)
    lo, hi = -1.0, 1.0
    while replay_log_growth(taus, h, lo) > 0.0:
        lo *= 2.0
        if lo < -64.0:
            raise RuntimeError("Failed to bracket lower root")
    while replay_log_growth(taus, h, hi) < 0.0:
        hi *= 2.0
        if hi > 64.0:
            raise RuntimeError("Failed to bracket upper root")
    root = float(optimize.brentq(lambda y: replay_log_growth(taus, h, y), lo, hi, xtol=1e-13, rtol=1e-13))
    res = abs(replay_log_growth(taus, h, root))
    return root, res


def ar1_ring_coeff(taus: tuple[float, ...], phi: float) -> float | None:
    """Analytical second-order coefficient Gamma_0 / sigma^2 for AR(1) memory."""
    if not (-1.0 < phi < 1.0):
        return None
    T0 = sum(taus)
    if abs(phi) < 1e-15:
        return 1.0 / (2.0 * T0)
    theta = phi / (1.0 - phi)
    R0 = 1.0
    for tau in taus:
        R0 *= phi / (tau + (1.0 - tau) * phi)
    return (1.0 + 2.0 * theta) / (2.0 * T0) - R0 / (1.0 - R0)


def spectral_kernel_T0K(taus: tuple[float, ...], omega: np.ndarray | float) -> np.ndarray:
    """Frequency-response kernel T_0 * K(omega) from Proposition prop:ring-spectrum.

    In prop:ring-spectrum, g(omega) = E[exp(i * omega * L)] where L = sum_j G_j
    with independent geometric stays G_j ~ Geom(1/tau_j) on {1, 2, ...}:
        E[exp(i * omega * G_j)] = (1/tau_j) * exp(i*omega) / (1 - (1 - 1/tau_j)*exp(i*omega)).
    Per-step cost is C(omega) = -(1/T_0) Re[g(omega)/(1 - g(omega))] for omega not in 2*pi*Z,
    and threshold loss per circuit multiplies per-step cost by T_0:
        Gamma_0(omega) = T_0 * K(omega) = -Re[g(omega)/(1 - g(omega))],
    with limit (1 - CV^2)/2 as omega -> 0 (mod 2*pi).
    """
    w = np.asarray(omega, dtype=np.float64)
    scalar = w.ndim == 0
    w1 = np.atleast_1d(w)
    T0 = sum(taus)
    var_L = sum(tau * (tau - 1.0) for tau in taus)
    cv2 = var_L / (T0 * T0)
    out = np.empty_like(w1, dtype=np.float64)
    z = np.exp(1j * w1)
    near_zero = np.abs(z - 1.0) < 1e-10
    out[near_zero] = 0.5 * (1.0 - cv2)
    if np.any(~near_zero):
        zn = z[~near_zero]
        g = np.ones_like(zn, dtype=np.complex128)
        for tau in taus:
            p_exit = 1.0 / tau
            g *= (p_exit * zn) / (1.0 - (1.0 - p_exit) * zn)
        out[~near_zero] = -np.real(g / (1.0 - g))
    return float(out[0]) if scalar else out


def matrix_lag_weights(taus: tuple[float, ...], n_lags: int = 4096) -> tuple[float, np.ndarray]:
    """Matrix perturbation lag-sum weights from Theorem thm:ring-rem-lin."""
    k = len(taus)
    T0 = sum(taus)
    A0 = step_matrix(taus, 1.0, 0.0)
    # At y=0, h=1: column sums of A0 are 1 -> left eigenvector w = (1,...,1), right v_i = tau_i / T0
    v = np.array(taus, dtype=np.float64) / T0
    w = np.ones(k, dtype=np.float64)
    Q = np.eye(k) - np.outer(v, w)
    B = np.zeros((k, k), dtype=np.float64)
    B[1, 0] = 1.0 / taus[0]
    b_sens = 1.0 / T0  # w @ B_birth @ v = (1/tau_{k-1}) * (tau_{k-1}/T0) = 1/T0
    c_use = float(w @ B @ v)  # = 1/T0
    leading = (c_use * c_use) / (2.0 * b_sens)  # = 1 / (2 * T0)
    M = A0 @ Q
    u = Q @ B @ v
    weights = np.empty(n_lags, dtype=np.float64)
    for ell in range(n_lags):
        weights[ell] = float(w @ B @ u) / b_sens
        u = M @ u
    if float(np.linalg.norm(u)) > 1e-12:
        raise RuntimeError("Lag kernel did not decay by 4096 lags")
    return leading, weights


def circular_autocov(x: np.ndarray) -> np.ndarray:
    xc = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    f = np.fft.fft(xc)
    return np.fft.ifft(np.abs(f) ** 2).real / len(xc)


def full_cov_prediction_spectral(taus: tuple[float, ...], series_centered: np.ndarray) -> float:
    """Exact discrete Fourier evaluation of second-order loss on circular replay."""
    n = len(series_centered)
    xc = series_centered - float(np.mean(series_centered))
    power = (np.abs(np.fft.fft(xc)) ** 2) / (n * n)
    # j = 0 has power[0] = 0 because xc is centered
    omegas = 2.0 * np.pi * np.arange(n) / n
    kern = spectral_kernel_T0K(taus, omegas)
    return float(np.dot(power[1:], kern[1:]))


def full_cov_prediction_lagsum(taus: tuple[float, ...], series_centered: np.ndarray) -> float:
    """Lag-sum evaluation of second-order loss on circular replay."""
    C = circular_autocov(series_centered)
    var = float(C[0])
    if var < 1e-25:
        return 0.0
    leading, weights = matrix_lag_weights(taus)
    idx = np.arange(1, len(weights) + 1) % len(series_centered)
    return float(leading * var - np.dot(weights, C[idx]))


def branching_pgf_survival(taus: tuple[float, ...], h: np.ndarray, initial: np.ndarray, horizons: tuple[int, ...], y: float = 0.05) -> dict[int, np.ndarray]:
    """Independent branching PGF survival by starting phase."""
    n = len(h)
    k = len(taus)
    q = np.zeros((n, k), dtype=np.float64)
    out = {}
    for step in range(1, max(horizons) + 1):
        q_next = np.roll(q, -1, axis=0)
        q_new = np.empty_like(q)
        for i, tau in enumerate(taus):
            p_leave = 1.0 / tau
            p_stay = 1.0 - p_leave
            nxt = (i + 1) % k
            if i == 0:
                mu = h
            elif i == k - 1:
                mu = math.exp(y)
            else:
                mu = 1.0
            q_new[:, i] = p_stay * q_next[:, i] + p_leave * np.exp(mu * (q_next[:, nxt] - 1.0))
        q = q_new
        if step in horizons:
            surv = 1.0 - np.prod(np.power(q, initial[None, :]), axis=1)
            out[step] = surv
    return out


# ---------------------------------------------------------------------------
# 2. Pre-registered benchmarks (Rule 7)
# ---------------------------------------------------------------------------

def run_benchmarks() -> list[dict]:
    results = []

    # K1: Constant host h == 1
    h_const = np.ones(30, dtype=np.float64)
    max_k1 = 0.0
    for k in (2, 3):
        for s in (1, 5, 20):
            D, _ = solve_replay_D(k, s, h_const)
            max_k1 = max(max_k1, abs(D))
    results.append({"id": "K1_constant_host", "max_err": max_k1, "tol": 1e-12, "passed": max_k1 <= 1e-12})

    # K2a: Period-2 alternating host 1 +- 0.6 at stays (2, 2)
    h_alt = np.array([1.6, 0.4], dtype=np.float64)
    # Exact root for taus=(2,2), h=(1.6, 0.4):
    # M(y) = [[0.5, e^y/2], [0.2, 0.5]] @ [[0.5, e^y/2], [0.8, 0.5]]
    #      = [[0.25 + 0.4 e^y, 0.5 e^y], [0.5, 0.25 + 0.1 e^y]]
    # det(I - M(y)) = 0 => (0.75 - 0.4 u)(0.75 - 0.1 u) - 0.25 u = 0 (u = e^y)
    # => 0.04 u^2 - 0.625 u + 0.5625 = 0 => u = (0.625 - sqrt(0.390625 - 0.09)) / 0.08
    exact_alt = math.log((0.625 - math.sqrt(0.300625)) / 0.08)
    # Solve with taus=(2,2) directly
    lo, hi = -1.0, 1.0
    d_alt = float(optimize.brentq(lambda y: replay_log_growth((2.0, 2.0), h_alt, y), lo, hi, xtol=1e-14))
    err_k2a = abs(d_alt - exact_alt)
    results.append({"id": "K2a_alternating_2_2", "actual": d_alt, "exact": exact_alt, "max_err": err_k2a, "tol": 1e-12, "passed": err_k2a <= 1e-12})

    # K2b: Period-2 zero-outage host h=(2, 0) at stays (2, s), s in {1, 5, 20}
    h_zero2 = np.array([2.0, 0.0], dtype=np.float64)
    max_k2b = 0.0
    for s in (1, 5, 20):
        d_num, _ = solve_replay_D(2, s, h_zero2)
        d_ex = math.log(3.0 * (2 * s - 1) / (2.0 * (3 * s - 1)))
        max_k2b = max(max_k2b, abs(d_num - d_ex))
    results.append({"id": "K2b_period2_outage_closed_form", "max_err": max_k2b, "tol": 1e-12, "passed": max_k2b <= 1e-12})

    # K3: AR(1) ring formula vs matrix lag-sum vs continuous Fourier integral of K(omega)
    max_k3 = 0.0
    w_grid = np.linspace(0.0, np.pi, 20001)
    for k in (2, 3):
        for s in (1, 5, 20):
            taus = stage_means(k, s)
            leading, weights = matrix_lag_weights(taus)
            kw = spectral_kernel_T0K(taus, w_grid)
            for phi in (-0.5, 0.0, 0.6, 0.9):
                c_closed = ar1_ring_coeff(taus, phi)
                lags = phi ** np.arange(1, len(weights) + 1)
                c_lagsum = leading - float(np.dot(weights, lags))
                # Spectral density of AR(1) on (0, pi): f(w) = (1 - phi^2) / (1 - 2*phi*cos(w) + phi^2)
                fw = (1.0 - phi * phi) / (1.0 - 2.0 * phi * np.cos(w_grid) + phi * phi)
                c_spec = float(np.trapezoid(fw * kw, w_grid) / np.pi)
                max_k3 = max(max_k3, abs(c_closed - c_lagsum), abs(c_closed - c_spec))
    results.append({"id": "K3_AR1_closed_vs_lagsum_vs_spectrum", "max_err": max_k3, "tol": 1e-11, "passed": max_k3 <= 1e-11})

    # K4: Small-amplitude symmetric finite difference vs spectral D2
    t_idx = np.arange(24, dtype=np.float64)
    z_syn = 0.7 * np.cos(2 * np.pi * t_idx / 6.0) + 0.3 * np.sin(2 * np.pi * t_idx / 8.0) + 0.2 * ((-1.0) ** t_idx)
    z_syn -= z_syn.mean()
    eps = 1e-3
    hp = np.exp(eps * z_syn); hp /= hp.mean()
    hm = np.exp(-eps * z_syn); hm /= hm.mean()
    max_k4 = 0.0
    for k in (2, 3):
        for s in (1, 5, 20):
            taus = stage_means(k, s)
            dp, _ = solve_replay_D(k, s, hp)
            dm, _ = solve_replay_D(k, s, hm)
            d2_sym = 0.5 * (dp + dm) / (eps * eps)
            d2_spec = full_cov_prediction_spectral(taus, z_syn)
            d2_lag = full_cov_prediction_lagsum(taus, z_syn)
            max_k4 = max(max_k4, abs(d2_sym - d2_spec), abs(d2_spec - d2_lag))
    results.append({"id": "K4_small_amplitude_perturbation", "max_err": max_k4, "tol": 1e-6, "passed": max_k4 <= 1e-6})

    # K5: Constant-host branching PGF vs scalar iteration
    max_k5 = 0.0
    for y in (-0.1, 0.0, 0.1):
        surv = branching_pgf_survival((2.0, 5.0), np.ones(1), np.array([10, 0]), (20, 60, 120), y=y)
        q0 = q1 = 0.0
        for step in range(1, 121):
            q0, q1 = 0.5 * q0 + 0.5 * math.exp(q1 - 1.0), 0.8 * q1 + 0.2 * math.exp(math.exp(y) * (q0 - 1.0))
            if step in surv:
                max_k5 = max(max_k5, abs(float(surv[step][0]) - (1.0 - q0 ** 10)))
    results.append({"id": "K5_constant_host_PGF", "max_err": max_k5, "tol": 1e-12, "passed": max_k5 <= 1e-12})

    if not all(r["passed"] for r in results):
        raise RuntimeError(f"Benchmark failure: {results}")
    return results


# ---------------------------------------------------------------------------
# 3. Independent raw data loading and frozen-sample verification
# ---------------------------------------------------------------------------

def load_raw_records() -> list[dict]:
    raw_path = ROOT / "data/raw/lpd/derived/candidate_series.json.gz"
    meta_path = ROOT / "casestudies/lpd/results/series_metrics.csv"
    wb_path = Q4_DIR / "raw/7972066/BeetleTeamTriboliumDataArchive.xlsx"

    with gzip.open(raw_path, "rt") as f:
        raw_list = json.load(f)
    by_id = {r["series_id"]: r for r in raw_list}
    with meta_path.open() as f:
        metrics = list(csv.DictReader(f))

    strata_map = collections.defaultdict(list)
    for m in metrics:
        sid = m["series_id"]
        r = by_id.get(sid)
        if r is None:
            continue
        try:
            vals = np.asarray(r["values"], dtype=np.float64)
            steps = np.asarray(r["steps"], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if len(vals) < 16 or len(vals) != len(steps):
            continue
        if np.any(~np.isfinite(vals)) or np.any(vals < 0.0) or np.any(~np.isfinite(steps)):
            continue
        if float(vals.mean()) <= 0.0:
            continue
        if not np.all(np.diff(steps) == 1.0):
            continue
        if not r.get("cc_by_compatible"):
            continue
        strata_map[(m["taxon_group"], m["interval_class"])].append(sid)

    keys = sorted(strata_map.keys())
    caps = np.array([len(strata_map[k]) for k in keys], dtype=int)
    target = min(120, int(caps.sum()))
    alloc = np.ones(len(keys), dtype=int)
    rem = target - len(keys)
    quotas = rem * (caps - 1) / float((caps - 1).sum())
    fl = np.floor(quotas).astype(int)
    alloc += fl
    order = sorted(range(len(keys)), key=lambda j: (-(quotas[j] - fl[j]), keys[j]))
    for j in order[: target - int(alloc.sum())]:
        alloc[j] += 1

    rng = np.random.default_rng(415111)
    comp_ids = []
    for k, n_pick in zip(keys, alloc):
        picked = sorted(rng.choice(sorted(strata_map[k]), int(n_pick), replace=False).tolist())
        comp_ids.extend(picked)
    comp_ids = sorted(comp_ids)

    records = []
    for sid in comp_ids:
        r = by_id[sid]
        vals = np.asarray(r["values"], dtype=np.float64)
        records.append({
            "series_id": sid,
            "source": r["database"],
            "taxon_group": r["taxon_group"],
            "interval_class": r["interval_class"],
            "dataset": "",
            "treatment": "",
            "block": "",
            "stage": "",
            "replicate": "",
            "values": vals,
            "h": vals / float(vals.mean()),
        })

    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
    for ds in (7, 8, 10):
        rows = list(wb[f"Data Set {ds}"].values)
        header = rows[0]
        grouped = collections.defaultdict(list)
        for row in rows[1:]:
            rec = dict(zip(header, row))
            if ds in (7, 10) and rec["Week"] < 18:
                continue
            key = (rec.get("Treatment", rec.get("HS")), rec.get("IC", 0), rec["Replicate"])
            grouped[key].append(rec)
        for (treat, block, rep), rs in sorted(grouped.items()):
            rs = sorted(rs, key=lambda x: x["Week"])
            for stage in ("L", "A"):
                vals_raw = [x[f"{stage}-stage"] for x in rs]
                if any(v is None for v in vals_raw):
                    continue
                vals = np.asarray(vals_raw, dtype=np.float64)
                if np.any(~np.isfinite(vals)) or np.any(vals < 0.0) or float(vals.mean()) <= 0.0:
                    continue
                sid = f"TRIB_D{ds}_T{treat}_IC{block}_R{rep}_{stage}"
                records.append({
                    "series_id": sid,
                    "source": "Tribolium",
                    "taxon_group": "laboratory_insects",
                    "interval_class": "biweekly",
                    "dataset": ds,
                    "treatment": treat,
                    "block": block,
                    "stage": stage,
                    "replicate": rep,
                    "values": vals,
                    "h": vals / float(vals.mean()),
                })
    for idx, r in enumerate(records):
        r["series_index"] = idx
    return records


def select_stratified_subsample(records: list[dict]) -> dict[str, list[str]]:
    """Select 20 compilation series, 10 Tribolium records, and all 4 non-zero reversals."""
    by_id = {r["series_id"]: r for r in records}
    nonzero_reversals = ["GPDD_10084", "GPDD_2004", "GPDD_9791", "GPDD_9897"]

    # 20 compilation series: 1 from each of the 11 (taxon_group, interval_class) strata,
    # plus 9 deterministic spread picks across large strata covering both BioTIME & GPDD
    # and both positive & zero-containing series, disjoint from nonzero_reversals.
    comp_pool = [r for r in records if r["source"] != "Tribolium" and r["series_id"] not in set(nonzero_reversals)]
    by_stratum = collections.defaultdict(list)
    for r in comp_pool:
        by_stratum[(r["taxon_group"], r["interval_class"])].append(r["series_id"])
    comp_20 = []
    for k in sorted(by_stratum.keys()):
        comp_20.append(sorted(by_stratum[k])[0])
    # Remaining slots up to 20 using evenly spaced picks across remaining compilation pool
    rem_needed = 20 - len(comp_20)
    rem_pool = [r["series_id"] for r in comp_pool if r["series_id"] not in set(comp_20)]
    idxs = np.linspace(0, len(rem_pool) - 1, rem_needed, dtype=int)
    for idx in idxs:
        comp_20.append(rem_pool[int(idx)])
    assert len(comp_20) == 20 and len(set(comp_20)) == 20

    # 10 Tribolium series stratified across D7, D8, D10 x (L, A) x (constant T=1, fluctuating T=2/3)
    trib_10 = [
        "TRIB_D7_T1_IC0_R13_A",   # D7 constant adult (positive)
        "TRIB_D7_T2_IC0_R5_A",    # D7 fluctuating adult (positive, H3 departure member)
        "TRIB_D7_T1_IC0_R13_L",   # D7 constant larva (zero-containing, H2 reversal)
        "TRIB_D7_T2_IC0_R5_L",    # D7 fluctuating larva (zero-containing, H2 reversal)
        "TRIB_D8_T1_IC1_R1_A",    # D8 constant adult (positive)
        "TRIB_D8_T2_IC1_R3_A",    # D8 fluctuating T2 adult (positive)
        "TRIB_D8_T3_IC2_R2_A",    # D8 fluctuating T3 adult (positive)
        "TRIB_D8_T2_IC1_R3_L",    # D8 fluctuating T2 larva (zero-containing, largest H2 reversal)
        "TRIB_D10_T2_IC0_R6_A",   # D10 fluctuating adult (positive)
        "TRIB_D10_T2_IC0_R6_L",   # D10 fluctuating larva (zero-containing, 3-stage H2 reversal)
    ]
    for sid in trib_10 + nonzero_reversals:
        assert sid in by_id, f"Missing {sid}"
    assert len(set(comp_20) | set(trib_10) | set(nonzero_reversals)) == 34
    return {
        "compilation_20": comp_20,
        "tribolium_10": trib_10,
        "nonzero_reversals_4": nonzero_reversals,
    }


# ---------------------------------------------------------------------------
# 4. Main check & Rule 13 departure analysis
# ---------------------------------------------------------------------------

def main() -> None:
    benchmarks = run_benchmarks()
    print("Benchmarks passed:", json.dumps(benchmarks, indent=2))

    records = load_raw_records()
    by_id = {r["series_id"]: r for r in records}

    # Verify selection against selection.json
    sel_saved = json.loads((Q4_DIR / "selection.json").read_text())
    assert [r["series_id"] for r in records] == sel_saved["ids"], "Selection IDs mismatch!"
    print(f"Verified exact frozen selection: {len(records)} records (120 compilation + 71 Tribolium).")

    # Load saved Q4 tables for comparison
    with (Q4_DIR / "cells.csv").open() as f:
        saved_cells = {(r["series_id"], int(r["stages"]), int(r["stay"])): r for r in csv.DictReader(f)}
    with (Q4_DIR / "monotonicity.csv").open() as f:
        saved_mono = list(csv.DictReader(f))
    with (Q4_DIR / "monotonicity-attribution.csv").open() as f:
        saved_mono_attr = list(csv.DictReader(f))
    with (Q4_DIR / "cell-attribution.csv").open() as f:
        saved_cell_attr = {(r["series_id"], int(r["stages"]), int(r["stay"])): r for r in csv.DictReader(f)}
    with (Q4_DIR / "contrasts.csv").open() as f:
        saved_contrasts = list(csv.DictReader(f))
    with (Q4_DIR / "contrast-attribution.csv").open() as f:
        saved_contrast_attr = {r["contrast_id"]: r for r in csv.DictReader(f)}
    with (Q4_DIR / "departures.csv").open() as f:
        saved_departures = list(csv.DictReader(f))

    subsample = select_stratified_subsample(records)
    subsample_set = set(subsample["compilation_20"] + subsample["tribolium_10"] + subsample["nonzero_reversals_4"])

    # Compute D, AR(1) prediction, full-cov spectral prediction, and raw-host second-order prediction for ALL 191 records
    our_cells = {}
    subsample_rows = []
    max_abs_sub = 0.0
    max_rel_sub = 0.0
    max_abs_all = 0.0
    max_pred_ar_err = 0.0
    max_pred_full_err = 0.0

    for r in records:
        sid = r["series_id"]
        h = r["h"]
        has_zero = bool(np.any(h <= 0.0))
        if not has_zero:
            z = np.log(h)
            zc = z - float(z.mean())
            C_log = circular_autocov(zc)
            var_log = float(C_log[0])
            phi_log = float(C_log[1] / var_log) if var_log > 1e-25 else 0.0
        else:
            zc = None
            var_log = None
            phi_log = None

        hc = h - 1.0

        for k in (2, 3):
            for s in (1, 5, 20):
                taus = stage_means(k, s)
                D_val, res = solve_replay_D(k, s, h)
                if not has_zero:
                    pred_ar = var_log * ar1_ring_coeff(taus, phi_log) if abs(phi_log) < 1.0 else None
                    pred_full = full_cov_prediction_spectral(taus, zc)
                else:
                    pred_ar = None
                    pred_full = None
                pred_raw_full = full_cov_prediction_spectral(taus, hc)

                sc = saved_cells[(sid, k, s)]
                d_saved = float(sc["D"])
                abs_diff = abs(D_val - d_saved)
                rel_diff = abs_diff / abs(d_saved) if abs(d_saved) > 1e-12 else 0.0
                max_abs_all = max(max_abs_all, abs_diff)

                if pred_ar is not None and sc["prediction_AR"] != "":
                    max_pred_ar_err = max(max_pred_ar_err, abs(pred_ar - float(sc["prediction_AR"])))
                if pred_full is not None and sc["prediction_full"] != "":
                    max_pred_full_err = max(max_pred_full_err, abs(pred_full - float(sc["prediction_full"])))

                agrees = (rel_diff < 0.01) or (abs_diff < 1e-4)
                our_cells[(sid, k, s)] = {
                    "series_id": sid,
                    "source": r["source"],
                    "stages": k,
                    "stay": s,
                    "D_ours": D_val,
                    "D_saved": d_saved,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "agrees": agrees,
                    "pred_ar": pred_ar,
                    "pred_full": pred_full,
                    "pred_raw_full": pred_raw_full,
                    "has_zero": has_zero,
                }
                if sid in subsample_set:
                    max_abs_sub = max(max_abs_sub, abs_diff)
                    if abs(d_saved) >= 1e-4:
                        max_rel_sub = max(max_rel_sub, rel_diff)
                    grp = (
                        "compilation_20" if sid in subsample["compilation_20"]
                        else ("tribolium_10" if sid in subsample["tribolium_10"] else "nonzero_reversals_4")
                    )
                    subsample_rows.append({
                        "stratum_group": grp,
                        **our_cells[(sid, k, s)],
                    })

    print(f"Subsample cells checked: {len(subsample_rows)} (all agree: {all(x['agrees'] for x in subsample_rows)})")
    print(f"Max abs diff (subsample 204 cells): {max_abs_sub:.3e}; max rel diff (|D|>=1e-4): {max_rel_sub:.3e}")
    print(f"Max abs diff (all 1146 cells):      {max_abs_all:.3e}")
    print(f"Max AR pred diff: {max_pred_ar_err:.3e}; Max full-ACF pred diff: {max_pred_full_err:.3e}")

    # Save subsample comparison CSV
    with (HERE / "subsample_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(subsample_rows[0].keys()))
        w.writeheader()
        w.writerows(subsample_rows)

    # -----------------------------------------------------------------------
    # Task 2(a): Detailed analysis of the 4 non-zero-support H2 reversals
    # -----------------------------------------------------------------------
    nonzero_rev_details = []
    for row in saved_mono_attr:
        sid = row["series_id"]
        r = by_id[sid]
        h = r["h"]
        if np.any(h <= 0.0):
            continue
        k = int(row["stages"])
        s1 = int(row["from_stay"])
        s2 = int(row["to_stay"])
        z = np.log(h)
        zc = z - float(z.mean())
        C = circular_autocov(zc)
        var_z = float(C[0])
        phi1 = float(C[1] / var_z)
        phi2 = float(C[2] / var_z)
        phi3 = float(C[3] / var_z)
        n = len(h)

        # Full-covariance, AR(1), and raw-host predictions across stays 1, 5, 20
        d_actual = {s: our_cells[(sid, k, s)]["D_ours"] for s in (1, 5, 20)}
        d2_full = {s: our_cells[(sid, k, s)]["pred_full"] for s in (1, 5, 20)}
        d2_ar = {s: our_cells[(sid, k, s)]["pred_ar"] for s in (1, 5, 20)}
        d2_raw = {s: our_cells[(sid, k, s)]["pred_raw_full"] for s in (1, 5, 20)}

        # Ultra-small amplitude eps=1e-3 verification
        eps = 1e-3
        he = np.exp(eps * zc); he /= he.mean()
        d_eps_rescaled = {s: solve_replay_D(k, s, he)[0] / (eps * eps) for s in (1, 5, 20)}

        # Spectral mode decomposition of D2_full(s1) - D2_full(s2)
        power = (np.abs(np.fft.fft(zc)) ** 2) / (n * n)
        omegas = 2.0 * np.pi * np.arange(n) / n
        k_s1 = spectral_kernel_T0K(stage_means(k, s1), omegas)
        k_s2 = spectral_kernel_T0K(stage_means(k, s2), omegas)
        # Fold conjugate frequencies into rfft modes m = 1..n//2
        rpower = np.zeros(n // 2 + 1)
        rk_s1 = np.zeros(n // 2 + 1)
        rk_s2 = np.zeros(n // 2 + 1)
        for m in range(1, n // 2 + 1):
            mult = 1.0 if (n % 2 == 0 and m == n // 2) else 2.0
            rpower[m] = mult * power[m]
            rk_s1[m] = k_s1[m]
            rk_s2[m] = k_s2[m]
        mode_contrib_diff = rpower * (rk_s1 - rk_s2)
        top_rev_modes = np.argsort(mode_contrib_diff)[:3]
        modes_info = []
        for m in top_rev_modes:
            if m == 0:
                continue
            modes_info.append({
                "mode_m": int(m),
                "period_steps": float(n / m),
                "power_share": float(rpower[m] / var_z),
                "T0K_from": float(rk_s1[m]),
                "T0K_to": float(rk_s2[m]),
                "delta_D2_contrib": float(mode_contrib_diff[m]),
            })

        nonzero_rev_details.append({
            "series_id": sid,
            "taxon_group": r["taxon_group"],
            "interval_class": r["interval_class"],
            "n": n,
            "stages": k,
            "from_stay": s1,
            "to_stay": s2,
            "sigma_log": math.sqrt(var_z),
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "log_skew": float(stats.skew(zc)),
            "ell_gap": float(-np.mean(z)),
            "D_actual": d_actual,
            "D_actual_decrease": d_actual[s1] - d_actual[s2],
            "D2_full": d2_full,
            "D2_full_decrease": d2_full[s1] - d2_full[s2],
            "D2_full_predicts_reversal": bool(d2_full[s1] < d2_full[s2]),
            "D2_AR": d2_ar,
            "D2_AR_decrease": d2_ar[s1] - d2_ar[s2],
            "D2_AR_predicts_reversal": bool(d2_ar[s1] < d2_ar[s2]),
            "D2_raw": d2_raw,
            "D2_raw_decrease": d2_raw[s1] - d2_raw[s2],
            "D_eps1e3_rescaled": d_eps_rescaled,
            "top_reversing_modes": modes_info,
            "saved_attribution": row["attribution"],
        })

    # -----------------------------------------------------------------------
    # Task 2(b): Detailed analysis of all 64 zero-containing series & 51 reversals
    # -----------------------------------------------------------------------
    zero_series_summary = []
    zero_rev_rows = [row for row in saved_mono_attr if "zero_support" in row["feature_flags"]]
    assert len(zero_rev_rows) == 51

    for r in records:
        h = r["h"]
        if not np.any(h <= 0.0):
            continue
        sid = r["series_id"]
        n = len(h)
        zero_mask = (h == 0.0)
        q0 = float(np.mean(zero_mask))
        # Circular zero-gap lengths
        if np.all(zero_mask):
            gaps = [n]
        elif not np.any(zero_mask):
            gaps = []
        else:
            # rotate to start at a positive value
            first_pos = int(np.argmax(~zero_mask))
            rot = np.roll(zero_mask, -first_pos)
            gaps = []
            cur = 0
            for flag in rot:
                if flag:
                    cur += 1
                elif cur > 0:
                    gaps.append(cur)
                    cur = 0
            if cur > 0:
                gaps.append(cur)
        hc = h - 1.0
        C_raw = circular_autocov(hc)
        var_raw = float(C_raw[0])
        rho_raw_1 = float(C_raw[1] / var_raw) if var_raw > 1e-25 else 0.0
        rho_raw_2 = float(C_raw[2] / var_raw) if var_raw > 1e-25 else 0.0

        d2 = [our_cells[(sid, 2, s)]["D_ours"] for s in (1, 5, 20)]
        d3 = [our_cells[(sid, 3, s)]["D_ours"] for s in (1, 5, 20)]
        raw2 = [our_cells[(sid, 2, s)]["pred_raw_full"] for s in (1, 5, 20)]
        raw3 = [our_cells[(sid, 3, s)]["pred_raw_full"] for s in (1, 5, 20)]

        rev_2_1_5 = (d2[0] - d2[1]) < -1e-6
        rev_2_5_20 = (d2[1] - d2[2]) < -1e-6
        rev_3_1_5 = (d3[0] - d3[1]) < -1e-6
        rev_3_5_20 = (d3[1] - d3[2]) < -1e-6

        zero_series_summary.append({
            "series_id": sid,
            "source": r["source"],
            "dataset": r["dataset"],
            "treatment": r["treatment"],
            "stage": r["stage"],
            "n": n,
            "zeros": int(np.sum(zero_mask)),
            "outage_freq_q0": q0,
            "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap": int(max(gaps)) if gaps else 0,
            "var_raw": var_raw,
            "rho_raw_1": rho_raw_1,
            "rho_raw_2": rho_raw_2,
            "D_k2": d2,
            "D_k3": d3,
            "D2_raw_k2": raw2,
            "D2_raw_k3": raw3,
            "reversals": [rev_2_1_5, rev_2_5_20, rev_3_1_5, rev_3_5_20],
        })

    # Summarize zero-series reversals
    rev_sources = collections.Counter()
    rev_signs = collections.Counter()
    raw_pred_agrees_dir = 0
    for row in zero_rev_rows:
        sid = row["series_id"]
        k = int(row["stages"])
        s1 = int(row["from_stay"])
        s2 = int(row["to_stay"])
        r = by_id[sid]
        rev_sources[(r["source"], r["dataset"], r["treatment"], r["stage"], k, f"{s1}->{s2}")] += 1
        d_from = our_cells[(sid, k, s1)]["D_ours"]
        d_to = our_cells[(sid, k, s2)]["D_ours"]
        rev_signs[(d_from < 0, d_to < 0)] += 1
        raw_from = our_cells[(sid, k, s1)]["pred_raw_full"]
        raw_to = our_cells[(sid, k, s2)]["pred_raw_full"]
        if raw_from < raw_to:
            raw_pred_agrees_dir += 1

    # Also test i.i.d. / Markov Bernoulli-type outage model directly across q0 and gap length G
    # to check exact behavior of D(q0, G, s)
    outage_grid = []
    # Exact 2-step / P-step periodic block outages: G zeros followed by (P - G) on-steps of height P/(P-G)
    for P, G in [(2, 1), (4, 2), (10, 5), (20, 10), (40, 20), (40, 30), (6, 1), (12, 6)]:
        h_blk = np.array([0.0] * G + [P / float(P - G)] * (P - G), dtype=np.float64)
        d_k2 = {s: solve_replay_D(2, s, h_blk)[0] for s in (1, 5, 20)}
        d_k3 = {s: solve_replay_D(3, s, h_blk)[0] for s in (1, 5, 20)}
        outage_grid.append({"P": P, "G": G, "q0": G / P, "D_k2": d_k2, "D_k3": d_k3})

    # -----------------------------------------------------------------------
    # Task 2(c): H1 departures (211 full-ACF departures; 224 AR(1) departures)
    # Check:
    # 1. Any-noise bounds of sections/04-bounds.tex:
    #    - Proposition prop:bounds-any(a): D <= ell = -E[ln h] for ALL 762 positive cells!
    #      Also check whether naive lognormal bound sigma^2/2 is ever exceeded when ell > sigma^2/2!
    #    - Proposition prop:bounds-any(b): liminf t^{-1} ln X(t) >= ln max_k s_k =>
    #      on ALL 1,146 cells (INCLUDING zero-containing cells!), check upper bound on D from staying alone:
    #      Wait: at D, rho(prod A(h_t, D)) = 1; also check single-positive-step analytical upper bound!
    #    - Negative D cells: how many among 762 positive cells and among 384 zero cells?
    # 2. Classification of the 211 H1-full departures (and 40 AR-only departures):
    #    - Check WHY 94 of 211 failed primary's one-sided eps=0.1 check:
    #      Evaluate symmetric eps=0.1 [0.5*(D(+0.1)+D(-0.1))/0.01] and smaller eps=0.01 [D(+0.01)/1e-4]!
    #      Compare D at eps=1 (actual) vs D2_full vs D2_raw vs ell = -mean(ln h)!
    # -----------------------------------------------------------------------
    bound_ell_violations = 0
    max_D_minus_ell = -1e9
    naive_half_sigma2_exceeded = []
    neg_D_positive_cells = []
    neg_D_zero_cells = []

    h1_full_departures_analysis = []
    for (sid, k, s), oc in our_cells.items():
        r = by_id[sid]
        h = r["h"]
        D_val = oc["D_ours"]
        if not oc["has_zero"]:
            ell = float(-np.mean(np.log(h)))
            var_z = float(np.var(np.log(h)))
            diff_ell = D_val - ell
            max_D_minus_ell = max(max_D_minus_ell, diff_ell)
            if diff_ell > 1e-12:
                bound_ell_violations += 1
            if D_val > 0.5 * var_z + 1e-12:
                naive_half_sigma2_exceeded.append({
                    "series_id": sid, "stages": k, "stay": s,
                    "D": D_val, "half_sigma2": 0.5 * var_z, "ell": ell,
                })
            if D_val < -1e-12:
                neg_D_positive_cells.append((sid, k, s, D_val))
        else:
            if D_val < -1e-12:
                neg_D_zero_cells.append((sid, k, s, D_val))

        sc = saved_cells[(sid, k, s)]
        if sc["H1_full"] == "departure":
            z = np.log(h)
            zc = z - float(z.mean())
            ell = float(-np.mean(z))
            var_z = float(np.var(zc))
            # One-sided eps=0.1, minus eps=-0.1, and eps=0.01
            hp01 = np.exp(0.1 * zc); hp01 /= hp01.mean()
            hm01 = np.exp(-0.1 * zc); hm01 /= hm01.mean()
            hp001 = np.exp(0.01 * zc); hp001 /= hp001.mean()
            dp01 = solve_replay_D(k, s, hp01)[0]
            dm01 = solve_replay_D(k, s, hm01)[0]
            dp001 = solve_replay_D(k, s, hp001)[0]

            d2 = oc["pred_full"]
            lim = max(0.001, 0.05 * abs(d2))
            onesided_01_err = abs(dp01 / 0.01 - d2)
            sym_01_err = abs(0.5 * (dp01 + dm01) / 0.01 - d2)
            onesided_001_err = abs(dp001 / 1e-4 - d2)
            # Estimate cubic coefficient D3 from 0.5*(D(+0.1) - D(-0.1)) / 0.001
            d3_est = 0.5 * (dp01 - dm01) / 0.001
            d4_est = (0.5 * (dp01 + dm01) - 0.01 * d2) / 1e-4

            ca = saved_cell_attr[(sid, k, s)]
            h1_full_departures_analysis.append({
                "series_id": sid,
                "source": r["source"],
                "stages": k,
                "stay": s,
                "D": D_val,
                "D2_full": d2,
                "D2_raw": oc["pred_raw_full"],
                "ell": ell,
                "half_sigma2": 0.5 * var_z,
                "sigma_log": math.sqrt(var_z),
                "log_skew": float(stats.skew(zc)),
                "log_kurt": float(stats.kurtosis(zc)),
                "small_agrees_independent_01": ca["small_agrees"] == "True",
                "sym_01_agrees": sym_01_err <= lim,
                "onesided_001_agrees": onesided_001_err <= lim,
                "onesided_01_err": onesided_01_err,
                "sym_01_err": sym_01_err,
                "onesided_001_err": onesided_001_err,
                "tol_small": lim,
                "D3_cubic_est": d3_est,
                "D4_quartic_est": d4_est,
                "phase_material": ca["phase_material"] == "True",
                "phase_mean_D": float(ca["phase_mean_D"]),
                "shuffle_material": ca["shuffle_material"] == "True",
                "detrend_material": ca["detrend_material"] == "True",
                "clip_material": ca["clip_material"] == "True",
                "mirror_material": ca["mirror_material"] == "True",
                "attribution": ca["attribution"],
                "feature_flags": ca["feature_flags"],
            })

    # Also check H1_AR vs H1_full cross-tabulation
    h1_crosstab = collections.Counter(
        (saved_cells[key]["H1_AR"], saved_cells[key]["H1_full"])
        for key in saved_cells
    )

    # -----------------------------------------------------------------------
    # Task 2(d): H3 departures (Tribolium experiment 7, adults)
    # -----------------------------------------------------------------------
    h3_departures_analysis = []
    for c in saved_contrasts:
        if c["H3_full"] != "departure" and c["H3_AR"] != "departure":
            continue
        cid = c["contrast_id"]
        k = int(c["stages"])
        s = int(c["stay"])
        members = c["members"].split(";")
        n_fl = int(c["n_fluct"])
        fl_ids = members[:n_fl]
        ct_ids = members[n_fl:]

        member_rows = []
        for sid in members:
            r = by_id[sid]
            h = r["h"]
            z = np.log(h)
            zc = z - float(z.mean())
            oc = our_cells[(sid, k, s)]
            sc = saved_cells[(sid, k, s)]
            hp01 = np.exp(0.1 * zc); hp01 /= hp01.mean()
            hm01 = np.exp(-0.1 * zc); hm01 /= hm01.mean()
            hp001 = np.exp(0.01 * zc); hp001 /= hp001.mean()
            dp01 = solve_replay_D(k, s, hp01)[0]
            dm01 = solve_replay_D(k, s, hm01)[0]
            dp001 = solve_replay_D(k, s, hp001)[0]
            member_rows.append({
                "series_id": sid,
                "arm": "fluctuating" if sid in fl_ids else "constant",
                "D": oc["D_ours"],
                "D2_full": oc["pred_full"],
                "D2_AR": oc["pred_ar"],
                "D2_raw": oc["pred_raw_full"],
                "ell": float(-np.mean(z)),
                "sigma_log": float(np.std(zc)),
                "log_skew": float(stats.skew(zc)),
                "H1_full": sc["H1_full"],
                "dp01_rescaled": dp01 / 0.01,
                "sym01_rescaled": 0.5 * (dp01 + dm01) / 0.01,
                "dp001_rescaled": dp001 / 1e-4,
            })

        fl_rows = member_rows[:n_fl]
        ct_rows = member_rows[n_fl:]
        d_obs = float(np.mean([x["D"] for x in fl_rows]) - np.mean([x["D"] for x in ct_rows]))
        d2_full_c = float(np.mean([x["D2_full"] for x in fl_rows]) - np.mean([x["D2_full"] for x in ct_rows]))
        d2_ar_c = float(np.mean([x["D2_AR"] for x in fl_rows]) - np.mean([x["D2_AR"] for x in ct_rows]))
        d2_raw_c = float(np.mean([x["D2_raw"] for x in fl_rows]) - np.mean([x["D2_raw"] for x in ct_rows]))
        sym01_c = float(np.mean([x["sym01_rescaled"] for x in fl_rows]) - np.mean([x["sym01_rescaled"] for x in ct_rows]))
        dp001_c = float(np.mean([x["dp001_rescaled"] for x in fl_rows]) - np.mean([x["dp001_rescaled"] for x in ct_rows]))

        h3_departures_analysis.append({
            "contrast_id": cid,
            "stages": k,
            "stay": s,
            "D_contrast": d_obs,
            "D2_full_contrast": d2_full_c,
            "D2_AR_contrast": d2_ar_c,
            "D2_raw_contrast": d2_raw_c,
            "sym01_rescaled_contrast": sym01_c,
            "dp001_rescaled_contrast": dp001_c,
            "tolerance_full": float(c["tolerance_full"]),
            "saved_attribution": saved_contrast_attr[cid]["attribution"],
            "members": member_rows,
        })

    # Aggregate summary statistics for printing and saving
    summary_out = {
        "benchmarks": benchmarks,
        "reproduction": {
            "total_records": len(records),
            "subsample_records": 34,
            "subsample_cells": len(subsample_rows),
            "subsample_all_agree": all(x["agrees"] for x in subsample_rows),
            "subsample_max_abs_diff": max_abs_sub,
            "subsample_max_rel_diff_above_1e4": max_rel_sub,
            "all_1146_max_abs_diff": max_abs_all,
            "max_pred_ar_err": max_pred_ar_err,
            "max_pred_full_err": max_pred_full_err,
            "subsample_ids": subsample,
        },
        "h1_crosstab": {f"{a}__{b}": cnt for (a, b), cnt in sorted(h1_crosstab.items())},
        "bounds_check": {
            "positive_cells_checked": sum(1 for x in our_cells.values() if not x["has_zero"]),
            "prop_bounds_any_ell_violations": bound_ell_violations,
            "max_D_minus_ell": max_D_minus_ell,
            "naive_half_sigma2_exceeded_count": len(naive_half_sigma2_exceeded),
            "naive_half_sigma2_exceeded_examples": naive_half_sigma2_exceeded[:10],
            "neg_D_positive_cells_count": len(neg_D_positive_cells),
            "neg_D_positive_cells": neg_D_positive_cells,
            "neg_D_zero_cells_count": len(neg_D_zero_cells),
        },
        "h2_nonzero_reversals": nonzero_rev_details,
        "h2_zero_reversals_summary": {
            "total_zero_series": len(zero_series_summary),
            "total_zero_reversals": len(zero_rev_rows),
            "by_source_ds_treat_stage_cell": {str(k): v for k, v in sorted(rev_sources.items())},
            "sign_pairs_from_neg_to_neg": {str(k): v for k, v in sorted(rev_signs.items())},
            "raw_second_order_predicts_reversal_count": raw_pred_agrees_dir,
            "outage_block_examples": outage_grid,
            "non_tribolium_zero_reversals": [
                row["series_id"] for row in zero_rev_rows if not row["series_id"].startswith("TRIB_")
            ],
            "tribolium_zero_series_not_reversing": [
                z["series_id"] for z in zero_series_summary
                if z["source"] == "Tribolium" and not any(z["reversals"])
            ],
            "compilation_zero_series_count": sum(1 for z in zero_series_summary if z["source"] != "Tribolium"),
            "tribolium_zero_series_count": sum(1 for z in zero_series_summary if z["source"] == "Tribolium"),
        },
        "h1_full_departures_summary": {
            "count": len(h1_full_departures_analysis),
            "primary_small_01_agrees": sum(1 for x in h1_full_departures_analysis if x["small_agrees_independent_01"]),
            "primary_small_01_unresolved": sum(1 for x in h1_full_departures_analysis if not x["small_agrees_independent_01"]),
            "symmetric_01_agrees": sum(1 for x in h1_full_departures_analysis if x["sym_01_agrees"]),
            "onesided_001_agrees": sum(1 for x in h1_full_departures_analysis if x["onesided_001_agrees"]),
            "D_less_than_D2_full": sum(1 for x in h1_full_departures_analysis if x["D"] < x["D2_full"]),
            "D_greater_than_D2_full": sum(1 for x in h1_full_departures_analysis if x["D"] > x["D2_full"]),
            "phase_material_count": sum(1 for x in h1_full_departures_analysis if x["phase_material"]),
            "phase_and_independent_small_01": sum(
                1 for x in h1_full_departures_analysis if x["small_agrees_independent_01"] and x["phase_material"]
            ),
            "primary_small_01_only_not_phase": sum(
                1 for x in h1_full_departures_analysis if x["small_agrees_independent_01"] and not x["phase_material"]
            ),
            "phase_not_independent_small_01": sum(
                1 for x in h1_full_departures_analysis if (not x["small_agrees_independent_01"]) and x["phase_material"]
            ),
            "neither_independent_small_nor_phase": [
                x for x in h1_full_departures_analysis
                if (not x["small_agrees_independent_01"]) and (not x["phase_material"])
            ],
        },
        "h3_departures": h3_departures_analysis,
    }

    (HERE / "q4_check_results.json").write_text(json.dumps(summary_out, indent=2) + "\n")
    print("Wrote q4_check_results.json successfully.")


if __name__ == "__main__":
    main()
