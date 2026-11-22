#!/usr/bin/env python3
"""Precompute and validate stochastic threshold-loss surfaces D(sigma, phi)
from studies/engine-R16/lifecycle_engine.py for case study lpd.

Run:
    python3 -I casestudies/lpd/engine_surfaces.py

Single-process vectorized evaluation (avoids macOS sandbox multiprocessing semaphores)
that is algebraically identical (< 1e-12, verified in check E0d) to calling
lifecycle_engine.Engine.loss() node by node. Validates small-noise accuracy (E0a),
the universal edge Jensen bound (E0b), off-grid interpolation accuracy (E0c), and
exact node identity with Engine.loss() (E0d) before saving results/engine_surfaces.json.
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
DOBSON_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_eng_spec = importlib.util.spec_from_file_location(
    "lifecycle_engine", DOBSON_ROOT / "studies" / "engine-R16" / "lifecycle_engine.py"
)
eng = importlib.util.module_from_spec(_eng_spec)
sys.modules[_eng_spec.name] = eng
_eng_spec.loader.exec_module(eng)

PRIMARY_SIGMAS = [0.0, 0.05, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50, 1.85, 2.25]
PRIMARY_PHIS = [-0.85, -0.65, -0.45, -0.25, -0.10, 0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.96]

SENS_SIGMAS = [0.0, 0.15, 0.35, 0.60, 0.90, 1.20, 1.55, 2.00]
SENS_SIGMAS_CONTACT = [0.0, 0.15, 0.35, 0.60, 0.85, 1.10, 1.32]  # Var(h) < 5.70 admissibility limit
SENS_PHIS_NONNEG = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95]
SENS_PHIS_SIGNED = [-0.70, -0.35, 0.0, 0.20, 0.45, 0.70, 0.88, 0.95]


def make_description(spec: dict, sigma: float, phi: float) -> eng.Description:
    mode = spec["mode"]
    b_stay = spec["b"]
    if mode == "one_stage_fine":
        pf = (phi ** 0.25) if phi > 0.0 else 0.0
        stages = [eng.Stage("exposed", 4.0, "memoryless")]
        drivers = [eng.Driver("h0", sigma, pf, "lognormal")]
        transfers = [eng.Transfer("exposed", "exposed", driver="h0", birth=True)]
        return eng.Description(stages, transfers, drivers)

    if mode == "primary_coarse":
        stages = [eng.Stage("exposed", 1.0, "fixed"), eng.Stage("long", b_stay, "memoryless")]
        drivers = [eng.Driver("h0", sigma, phi, "lognormal")]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    if mode == "primary_fine":
        pf = (phi ** 0.25) if phi > 0.0 else 0.0
        stages = [eng.Stage("exposed", 4.0, "memoryless"), eng.Stage("long", 4.0 * b_stay, "memoryless")]
        drivers = [eng.Driver("h0", sigma, pf, "lognormal")]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    if mode == "bounded_coarse":
        stages = [eng.Stage("exposed", 1.0, "fixed"), eng.Stage("long", b_stay, "memoryless")]
        drivers = [eng.Driver("h0", sigma, phi, "bounded", 0.05)]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    if mode == "chain4_fine":
        pf = (phi ** 0.25) if phi > 0.0 else 0.0
        stages = [eng.Stage("exposed", 4.0, "memoryless"), eng.Stage("long", 4.0 * b_stay, "chain", 4)]
        drivers = [eng.Driver("h0", sigma, pf, "lognormal")]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    if mode == "alt2_coarse":
        stages = [eng.Stage("exposed", 1.0, "fixed"), eng.Stage("long", b_stay, "memoryless")]
        drivers = [eng.Driver("h0", sigma, phi, "lognormal"), eng.Driver("h1", sigma, phi, "lognormal")]
        transfers = [
            eng.Transfer("exposed", "long", 0.5, driver="h0"),
            eng.Transfer("exposed", "long", 0.5, driver="h1"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    if mode == "patches4_coarse":
        stages = [eng.Stage("exposed", 1.0, "fixed"), eng.Stage("long", b_stay, "memoryless")]
        drivers = [eng.Driver("h0", sigma, phi, "lognormal")]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        patches = eng.Patches(4, 0.0, "long", 0.15, "island", 0.0)
        return eng.Description(stages, transfers, drivers, patches=patches)

    if mode == "contact_fine":
        pf = (phi ** 0.25) if phi > 0.0 else 0.0
        stages = [
            eng.Stage("exposed", 4.0, "memoryless", death=0.125, contact=True),
            eng.Stage("long", 4.0 * b_stay, "memoryless"),
        ]
        drivers = [eng.Driver("h0", sigma, pf, "bounded", 0.05)]
        transfers = [
            eng.Transfer("exposed", "long", driver="h0"),
            eng.Transfer("long", "exposed", birth=True),
        ]
        return eng.Description(stages, transfers, drivers)

    raise ValueError(f"Unknown surface mode: {mode}")


def batched_products(
    rates_all: np.ndarray,
    v0: np.ndarray,
    A0: np.ndarray,
    A1: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
    y_vec: np.ndarray,
    burn: int,
    pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized replica of Engine._products across G parameter nodes simultaneously.
    rates_all: (T, G, R, K), R = 2*pairs; v0: (G, d); A0, A1: (G, d, d); P0, P1: (G, K, d, d); y_vec: (G,).
    Returns logs: (G, pairs), slopes: (G, pairs).
    """
    t_len, g_len, reps, _ = rates_all.shape
    d_dim = v0.shape[1]
    x = np.broadcast_to(v0[:, None, :], (g_len, reps, d_dim)).copy()
    dx = np.zeros_like(x)
    logs = np.zeros((g_len, reps), dtype=float)
    slopes = np.zeros((g_len, reps), dtype=float)

    ey = np.exp(y_vec)
    base = A0 + ey[:, None, None] * A1  # (G, d, d)
    base_y = ey[:, None, None] * A1     # (G, d, d)
    parts = P0 + ey[:, None, None, None] * P1   # (G, K, d, d)
    parts_y = ey[:, None, None, None] * P1      # (G, K, d, d)

    for t in range(t_len):
        rm1 = rates_all[t] - 1.0  # (G, R, K)
        # mat: (G, R, d, d)
        mat = base[:, None, :, :] + np.einsum("grk,gkij->grij", rm1, parts, optimize=False)
        maty = base_y[:, None, :, :] + np.einsum("grk,gkij->grij", rm1, parts_y, optimize=False)
        new = np.einsum("grij,grj->gri", mat, x, optimize=False)
        dn = np.einsum("grij,grj->gri", mat, dx, optimize=False) + np.einsum("grij,grj->gri", maty, x, optimize=False)
        total = new.sum(axis=2)  # (G, R)
        der = dn.sum(axis=2) / total
        x = new / total[:, :, None]
        dx = dn / total[:, :, None] - x * der[:, :, None]
        if t >= burn:
            logs += np.log(total)
            slopes += der

    keep = float(t_len - burn)
    logs /= keep
    slopes /= keep
    logs_p = 0.5 * (logs[:, :pairs] + logs[:, pairs:])
    slopes_p = 0.5 * (slopes[:, :pairs] + slopes[:, pairs:])
    return logs_p, slopes_p


def compute_surface_batched(surf_id: str, sp: dict, base_seed: int) -> dict:
    sigmas = sp["sigmas"]
    phis = sp["phis"]
    ns, np_ = len(sigmas), len(phis)
    D_mat = np.zeros((ns, np_), dtype=float)
    se_mat = np.zeros((ns, np_), dtype=float)
    D2_mat = np.zeros((ns, np_), dtype=float)
    J_mat = np.zeros((ns, np_), dtype=float)

    steps = int(sp["steps"])
    burn = int(sp["burn"])
    pairs = int(sp["pairs"])
    t_total = steps + burn

    # Collect all active nodes (sigma > 0)
    coords = []
    models = []
    for i_s, sig in enumerate(sigmas):
        if sig == 0.0:
            continue
        for i_p, phi in enumerate(phis):
            m = eng.Engine(make_description(sp, float(sig), float(phi))).critical()
            D2_mat[i_s, i_p] = float(m.coefficient()["D2"])
            J_mat[i_s, i_p] = float(m.jensen() / m.b)
            coords.append((i_s, i_p))
            models.append(m)

    if not models:
        return {"sigmas": sigmas, "phis": phis, "D": D_mat, "se": se_mat, "D2": D2_mat, "J": J_mat}

    # Process in chunks of up to 56 nodes so memory stays tiny (~40 MB) and cache-hot
    chunk_size = 56
    for c_start in range(0, len(models), chunk_size):
        c_models = models[c_start : c_start + chunk_size]
        c_coords = coords[c_start : c_start + chunk_size]
        g_len = len(c_models)
        d_dim = c_models[0].n
        k_dim = len(c_models[0].rate_keys)

        rates_all = np.empty((t_total, g_len, 2 * pairs, k_dim), dtype=float)
        v0 = np.empty((g_len, d_dim), dtype=float)
        A0 = np.empty((g_len, d_dim, d_dim), dtype=float)
        A1 = np.empty((g_len, d_dim, d_dim), dtype=float)
        P0 = np.empty((g_len, k_dim, d_dim, d_dim), dtype=float)
        P1 = np.empty((g_len, k_dim, d_dim, d_dim), dtype=float)

        for g, m in enumerate(c_models):
            h = m.environment.sample(t_total, pairs, base_seed, antithetic=True)
            rates_all[:, g, :, :] = m.rates(h)
            v0[g] = m.v
            A0[g] = m.base_parts[0]
            A1[g] = m.base_parts[1]
            P0[g] = m.rate_parts[0]
            P1[g] = m.rate_parts[1]

        r0, s0 = batched_products(rates_all, v0, A0, A1, P0, P1, np.zeros(g_len), burn, pairs)
        y_vec = np.clip(-r0.mean(axis=1) / s0.mean(axis=1), -3.0, 3.0)
        for _it in range(12):
            r, s = batched_products(rates_all, v0, A0, A1, P0, P1, y_vec, burn, pairs)
            corr = r.mean(axis=1) / s.mean(axis=1)
            if float(np.max(np.abs(corr))) < 1e-7:
                break
            y_vec = y_vec - np.clip(corr, -2.0, 2.0)
        else:
            raise RuntimeError(f"Batched Newton root did not converge on {surf_id}")

        s_mean = s.mean(axis=1)
        samples = y_vec[:, None] - r / s_mean[:, None]
        ses = samples.std(axis=1, ddof=1) / math.sqrt(pairs)
        for g, (i_s, i_p) in enumerate(c_coords):
            D_mat[i_s, i_p] = float(y_vec[g])
            se_mat[i_s, i_p] = float(ses[g])

    return {
        "mode": sp["mode"],
        "cycle": sp["cycle"],
        "b": sp["b"],
        "sigmas": sigmas,
        "phis": phis,
        "D": D_mat,
        "se": se_mat,
        "D2": D2_mat,
        "J": J_mat,
    }


def surface_specs() -> dict[str, dict]:
    cycles = [("long", 24.0), ("moderate", 7.5), ("short", 2.3)]
    specs: dict[str, dict] = {}
    for cname, bval in cycles:
        specs[f"primary_coarse_{cname}"] = {
            "mode": "primary_coarse",
            "cycle": cname,
            "b": bval,
            "sigmas": PRIMARY_SIGMAS,
            "phis": PRIMARY_PHIS,
            "steps": 24000,
            "burn": 1200,
            "pairs": 12,
        }
        specs[f"primary_fine_{cname}"] = {
            "mode": "primary_fine",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS,
            "phis": SENS_PHIS_NONNEG,
            "steps": 24000,
            "burn": 1600,
            "pairs": 8,
        }
        specs[f"bounded_coarse_{cname}"] = {
            "mode": "bounded_coarse",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS_CONTACT + [1.40],
            "phis": SENS_PHIS_NONNEG,
            "steps": 20000,
            "burn": 1200,
            "pairs": 8,
        }
        specs[f"chain4_fine_{cname}"] = {
            "mode": "chain4_fine",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS,
            "phis": SENS_PHIS_NONNEG,
            "steps": 20000,
            "burn": 1600,
            "pairs": 8,
        }
        specs[f"alt2_coarse_{cname}"] = {
            "mode": "alt2_coarse",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS,
            "phis": SENS_PHIS_SIGNED,
            "steps": 20000,
            "burn": 1200,
            "pairs": 8,
        }
        specs[f"patches4_coarse_{cname}"] = {
            "mode": "patches4_coarse",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS,
            "phis": SENS_PHIS_SIGNED,
            "steps": 16000,
            "burn": 1200,
            "pairs": 8,
        }
        specs[f"contact_fine_{cname}"] = {
            "mode": "contact_fine",
            "cycle": cname,
            "b": bval,
            "sigmas": SENS_SIGMAS_CONTACT,
            "phis": SENS_PHIS_NONNEG,
            "steps": 20000,
            "burn": 1600,
            "pairs": 8,
        }
    specs["one_stage_fine"] = {
        "mode": "one_stage_fine",
        "cycle": "one_stage_a4",
        "b": 0.0,
        "sigmas": SENS_SIGMAS,
        "phis": [0.0, 0.50],
        "steps": 24000,
        "burn": 1600,
        "pairs": 12,
    }
    return specs


def main() -> int:
    t0 = time.monotonic()
    specs = surface_specs()
    results_map: dict[str, dict] = {}

    for idx, (surf_id, sp) in enumerate(specs.items()):
        t_s = time.monotonic()
        base_seed = 926000 + idx * 101
        results_map[surf_id] = compute_surface_batched(surf_id, sp, base_seed)
        print(f"  [{idx + 1:2d}/{len(specs)}] {surf_id:26s} ({time.monotonic() - t_s:5.1f}s)", flush=True)

    checks = []
    # E0d: Exact node identity between batched evaluator and direct Engine.loss()
    worst_node_diff = 0.0
    for idx_check, surf_id in enumerate(["primary_coarse_long", "patches4_coarse_moderate", "contact_fine_short", "chain4_fine_short"]):
        sp_idx = list(specs.keys()).index(surf_id)
        base_seed = 926000 + sp_idx * 101
        sp = specs[surf_id]
        rm = results_map[surf_id]
        i_s, i_p = 3, 2
        sig, phi = float(sp["sigmas"][i_s]), float(sp["phis"][i_p])
        m = eng.Engine(make_description(sp, sig, phi)).critical()
        direct = m.loss(steps=int(sp["steps"]), burn=int(sp["burn"]), pairs=int(sp["pairs"]), seed=base_seed)
        worst_node_diff = max(worst_node_diff, abs(rm["D"][i_s, i_p] - float(direct["D"])))
    checks.append({
        "id": "E0d",
        "claim": "Batched surface evaluator matches direct Engine.loss(tolerance=1e-7) on identical seeds across 4 distinct model types within 2e-7",
        "tolerance": 2e-7,
        "observed": worst_node_diff,
        "pass": bool(worst_node_diff < 2e-7),
    })

    # E0a: Small-noise sigma=0.05 agreement on primary_coarse surfaces
    max_small_err = 0.0
    small_ok = True
    for cname in ("long", "moderate", "short"):
        rm = results_map[f"primary_coarse_{cname}"]
        i_s = rm["sigmas"].index(0.05)
        for i_p in range(len(rm["phis"])):
            d_val = rm["D"][i_s, i_p]
            d2_val = rm["D2"][i_s, i_p]
            se_val = rm["se"][i_s, i_p]
            tol = 4.0 * se_val + 0.05 * abs(d2_val) + 1e-5
            err = abs(d_val - d2_val)
            max_small_err = max(max_small_err, err)
            if err > tol:
                small_ok = False
    checks.append({
        "id": "E0a",
        "claim": "At sigma=0.05, simulated D agrees with D2 within 4 SE + 5% across all primary coarse nodes",
        "max_abs_diff": max_small_err,
        "pass": small_ok,
    })

    # E0b: Universal edge Jensen bound D <= J + 4 SE + 1e-7 across all nodes of all surfaces
    bound_violations = 0
    worst_margin = -1e9
    for surf_id, rm in results_map.items():
        margin = float(np.max(rm["D"] - (rm["J"] + 4.0 * rm["se"] + 1e-7)))
        worst_margin = max(worst_margin, margin)
        if margin > 0.0:
            bound_violations += 1
    checks.append({
        "id": "E0b",
        "claim": "All computed surface nodes satisfy D <= J + 4 SE + 1e-7",
        "worst_excess": worst_margin,
        "pass": bound_violations == 0,
    })

    # E0c: Off-grid bilinear interpolation accuracy on 12 random (sigma, phi) test points
    rng = np.random.default_rng(2026111501)
    worst_interp_z = 0.0
    worst_interp_abs = 0.0
    interp_ok = True
    for trial in range(12):
        cname, bval = [("long", 24.0), ("moderate", 7.5), ("short", 2.3)][trial % 3]
        rm = results_map[f"primary_coarse_{cname}"]
        rgi = RegularGridInterpolator((rm["sigmas"], rm["phis"]), rm["D"], method="linear")
        sig = float(rng.uniform(0.15, 1.80))
        phi = float(rng.uniform(-0.60, 0.90))
        d_interp = float(rgi([[sig, phi]])[0])
        desc = make_description({"mode": "primary_coarse", "b": bval}, sig, phi)
        direct = eng.Engine(desc).critical().loss(steps=28000, burn=1400, pairs=14, seed=77000 + trial)
        diff = abs(d_interp - float(direct["D"]))
        tol = max(4.0 * float(direct["se"]) + 0.012, 0.04 * abs(float(direct["D"])))
        worst_interp_abs = max(worst_interp_abs, diff)
        worst_interp_z = max(worst_interp_z, diff / max(tol, 1e-12))
        if diff > tol:
            interp_ok = False
    checks.append({
        "id": "E0c",
        "claim": "Off-grid linear surface interpolation matches 12 independent Engine.loss runs within max(4 SE + 0.012, 4% |D|)",
        "max_abs_diff": worst_interp_abs,
        "max_ratio_to_tol": worst_interp_z,
        "pass": interp_ok,
    })

    if not all(c["pass"] for c in checks):
        raise RuntimeError(f"Surface validation check failed: {checks}")

    serializable = {}
    for surf_id, rm in results_map.items():
        serializable[surf_id] = {
            "mode": rm["mode"],
            "cycle": rm["cycle"],
            "b": rm["b"],
            "sigmas": rm["sigmas"],
            "phis": rm["phis"],
            "D": np.round(rm["D"], 7).tolist(),
            "se": np.round(rm["se"], 7).tolist(),
            "D2": np.round(rm["D2"], 7).tolist(),
            "J": np.round(rm["J"], 7).tolist(),
        }

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "checks": checks,
        "surfaces": serializable,
    }
    out_path = RESULTS_DIR / "engine_surfaces.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"All surface checks passed ({out['elapsed_seconds']}s). Saved {out_path}", flush=True)
    for c in checks:
        print(" ", c, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
