#!/usr/bin/env python3
"""Clean-room statistical and finite-horizon helpers for the 18-cell pilot rerun (r35-vworld-B).

Reuses the validated Step-A model (`sim_world.py`) without importing or copying
any code from `virtualworld/`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

from sim_world import WorldModel

Z95: float = float(stats.norm.ppf(0.975))
Z80: float = float(stats.norm.ppf(0.80))
POWER80_FACTOR: float = (Z95 + Z80) ** 2


def largest_remainder_allocate(total: int, weights: Sequence[float]) -> np.ndarray:
    """Allocate `total` integer individuals proportionally by largest fractional remainder.

    Ties in fractional remainder are broken deterministically by smallest state index.
    """
    if int(total) != total or total < 0:
        raise ValueError("total must be a nonnegative integer.")
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or np.any(~np.isfinite(w)) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
        raise ValueError("weights must be a positive-sum 1D nonnegative finite array.")
    quotas = float(total) * (w / float(np.sum(w)))
    floored = np.floor(quotas).astype(np.int64)
    leftover = int(total) - int(np.sum(floored))
    if leftover > 0:
        remainders = quotas - floored
        order = np.argsort(-remainders, kind="stable")
        floored[order[:leftover]] += 1
    return floored


def wilson_ci95(k_extinct: int, n_total: int) -> tuple[float, float]:
    """Pointwise 95% Wilson score confidence interval for a binomial proportion."""
    if not (0 <= k_extinct <= n_total) or n_total <= 0:
        raise ValueError("Invalid binomial counts for Wilson interval.")
    p = float(k_extinct) / float(n_total)
    n = float(n_total)
    z2 = Z95 * Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return float(center - half), float(center + half)


def compute_finite_conditional_growth(
    model: WorldModel,
    hosts: np.ndarray,
    tilt_y: float,
    init_weights: Sequence[float],
    horizons: Sequence[int] = (100, 500),
) -> dict[int, np.ndarray]:
    """All-schedule conditional-mean log growth per step over finite horizons.

    Includes every environment schedule regardless of whether the paired
    demographic realization goes extinct (`protocol.md` lines 84-86, 158-160).
    """
    h_arr = np.asarray(hosts, dtype=np.float64)
    if h_arr.ndim != 3 or h_arr.shape[2] != model.n_channels:
        raise ValueError("hosts must have shape (n_steps, n_reps, n_channels).")
    n_steps, n_reps, _ = h_arr.shape
    w0 = np.asarray(init_weights, dtype=np.float64)
    if w0.shape != (model.dim,) or np.any(w0 < 0.0) or float(np.sum(w0)) <= 0.0:
        raise ValueError("init_weights must be a nonnegative vector of length model.dim with positive sum.")

    comp = np.broadcast_to(w0 / float(np.sum(w0)), (n_reps, model.dim)).copy()
    cum_log = np.zeros(n_reps, dtype=np.float64)
    rates = model.host_response(h_arr)
    want = {int(ep) for ep in horizons}
    if any(ep < 1 or ep > n_steps for ep in want):
        raise ValueError("Requested growth horizon outside simulated step range.")

    out: dict[int, np.ndarray] = {}
    if model.is_rate_affine:
        ey = math.exp(tilt_y)
        base_m = model.U0 + ey * model.F0
        pert_m = model.U1 + ey * model.F1
        for t in range(1, n_steps + 1):
            m_t = base_m + np.einsum("rc,cij->rij", rates[t - 1] - 1.0, pert_m)
            nxt = np.einsum("rij,rj->ri", m_t, comp)
            tot = np.sum(nxt, axis=1)
            if np.any(tot <= 0.0) or np.any(nxt < -1e-12):
                raise FloatingPointError("Invalid conditional-mean vector during finite growth.")
            cum_log += np.log(tot)
            comp = nxt / tot[:, None]
            if t in want:
                out[t] = cum_log.copy() / float(t)
    else:
        for t in range(1, n_steps + 1):
            m_t = model.expectation_matrix(h_arr[t - 1], tilt_y=tilt_y)
            nxt = np.einsum("rij,rj->ri", m_t, comp)
            tot = np.sum(nxt, axis=1)
            if np.any(tot <= 0.0) or np.any(nxt < -1e-12):
                raise FloatingPointError("Invalid conditional-mean vector during finite growth.")
            cum_log += np.log(tot)
            comp = nxt / tot[:, None]
            if t in want:
                out[t] = cum_log.copy() / float(t)
    return out


def compute_four_cell_interaction(
    cells: dict[str, dict[str, object]],
    growth_samples: dict[str, np.ndarray],
    cell_keys: Sequence[str],
    weights: Sequence[float],
    outcome: str,
    jeffreys_seed: int = 359000,
    n_jeffreys_ci: int = 100_000,
    n_jeffreys_power: int = 10_000,
) -> dict[str, object]:
    """Compute four-cell linear contrast, Wald 95% CI, 80% power n/cell, and Jeffreys sensitivity."""
    if len(cell_keys) != 4 or len(weights) != 4:
        raise ValueError("Four-cell interaction requires 4 cell keys and 4 weights.")
    w = np.asarray(weights, dtype=np.float64)
    ns = np.array([int(cells[k]["n"]) for k in cell_keys], dtype=np.float64)

    if outcome == "extinction":
        means = np.array([float(cells[k]["extinction"]) for k in cell_keys], dtype=np.float64)
        unit_vars = means * (1.0 - means)
    elif outcome == "growth100":
        means = np.array([float(cells[k]["growth100"]) for k in cell_keys], dtype=np.float64)
        unit_vars = np.array(
            [0.0 if float(cells[k]["sigma"]) == 0.0 else float(np.var(growth_samples[k], ddof=1)) for k in cell_keys],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported outcome: {outcome}")

    delta = float(np.dot(w, means))
    var_est = float(np.sum((w * w) * unit_vars / ns))
    se_est = math.sqrt(var_est)
    ci95 = [float(delta - Z95 * se_est), float(delta + Z95 * se_est)]
    var_sum = float(np.sum(unit_vars))
    n80 = None if delta == 0.0 else max(2, int(math.ceil(POWER80_FACTOR * var_sum / (delta * delta))))

    result: dict[str, object] = {
        "effect": delta,
        "se": se_est,
        "ci95": ci95,
        "n80_per_cell": n80,
        "variance_sum": var_sum,
        "cells": list(cell_keys),
        "weights": [float(x) for x in w],
    }

    if outcome == "extinction":
        rng = np.random.default_rng(jeffreys_seed)
        ks = np.array([int(cells[k]["extinct"]) for k in cell_keys], dtype=np.float64)
        beta_draws = rng.beta(ks + 0.5, ns - ks + 0.5, size=(n_jeffreys_ci, 4))
        contrast_draws = beta_draws @ w
        j_lo, j_hi = np.quantile(contrast_draws, [0.025, 0.975])
        result["jeffreys_interval95"] = [float(j_lo), float(j_hi)]

        sub_p = beta_draws[:n_jeffreys_power]
        sub_d = contrast_draws[:n_jeffreys_power]
        n_draws = np.ceil(POWER80_FACTOR * np.sum(sub_p * (1.0 - sub_p), axis=1) / (sub_d * sub_d))
        p10, p90 = np.quantile(n_draws, [0.10, 0.90])
        result["n80_sensitivity_10_90"] = [float(p10), float(p90)]
        result["finite_upper_established"] = bool(not (j_lo <= 0.0 <= j_hi))

    return result


def compute_simple_two_cell_contrast(
    label: str,
    cell_a: dict[str, object],
    cell_b: dict[str, object],
) -> dict[str, object]:
    """Pointwise 95% normal contrasts for extinction (A - B) and early growth loss (B - A)."""
    na = float(cell_a["n"])
    nb = float(cell_b["n"])
    pa = float(cell_a["extinction"])
    pb = float(cell_b["extinction"])
    diff_ext = pa - pb
    se_ext = math.sqrt(pa * (1.0 - pa) / na + pb * (1.0 - pb) / nb)

    ra = float(cell_a["growth100"])
    rb = float(cell_b["growth100"])
    loss_growth = rb - ra
    se_growth = math.hypot(float(cell_a["growth100_se"]), float(cell_b["growth100_se"]))

    return {
        "label": label,
        "cell_a": str(cell_a["cell"]),
        "cell_b": str(cell_b["cell"]),
        "extinction": diff_ext,
        "extinction_se": se_ext,
        "ext_low": diff_ext - Z95 * se_ext,
        "ext_high": diff_ext + Z95 * se_ext,
        "growth_loss": loss_growth,
        "growth_loss_se": se_growth,
        "growth_low": loss_growth - Z95 * se_growth,
        "growth_high": loss_growth + Z95 * se_growth,
    }
