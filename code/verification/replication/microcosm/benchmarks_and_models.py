#!/usr/bin/env python3
"""Independent mathematical solvers, statistical routines, and Rule-7 benchmarks
for the `microcosm` case-study replication (`r36-rep-microcosm`).

Written from scratch from the mathematical specifications in `protocol.md`,
`notes/contact-limited-transfer.md`, and `notes/decisive-experiment-v2.md`.
No code from `casestudies/microcosm/` or `studies/` is imported or copied.
"""
from __future__ import annotations

import collections
import csv
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import integrate, linalg, optimize, special, stats


# ---------------------------------------------------------------------------
# 1. Independent safe raw-data readers (preserving full sheets & dup headers)
# ---------------------------------------------------------------------------

def load_xlsx_rows(xlsx_path: Path, sheet_number: int = 1) -> list[list[str]]:
    """Parse an `.xlsx` worksheet via XML without executing formulas or truncating columns."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall(".//s:si", ns):
                shared.append("".join( node.text or "" for node in si.findall(".//s:t", ns) ))

        sheet_xml = ET.fromstring(zf.read(f"xl/worksheets/sheet{sheet_number}.xml"))
        parsed_rows: list[list[str]] = []
        for row_el in sheet_xml.findall(".//s:row", ns):
            row_cells: dict[int, str] = {}
            max_idx = -1
            for cell in row_el.findall("s:c", ns):
                coord = cell.attrib.get("r", "")
                m = re.match(r"^([A-Z]+)", coord)
                if m:
                    col_idx = 0
                    for ch in m.group(1):
                        col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                    col_idx -= 1
                else:
                    col_idx = len(row_cells)
                max_idx = max(max_idx, col_idx)

                cell_type = cell.attrib.get("t", "")
                val_el = cell.find("s:v", ns)
                if val_el is not None and val_el.text is not None:
                    text_val = shared[int(val_el.text)] if cell_type == "s" else val_el.text
                else:
                    inline_el = cell.find(".//s:t", ns)
                    text_val = inline_el.text if (inline_el is not None and inline_el.text is not None) else ""
                row_cells[col_idx] = text_val.strip()
            if max_idx >= 0:
                parsed_rows.append([row_cells.get(j, "") for j in range(max_idx + 1)])
        return parsed_rows


def load_xlsx_records(xlsx_path: Path, sheet_number: int = 1, n_cols: int | None = None) -> list[dict[str, str]]:
    """Convert worksheet rows into header-keyed dictionaries using row 0 headers."""
    rows = load_xlsx_rows(xlsx_path, sheet_number=sheet_number)
    if not rows:
        return []
    header = rows[0] if n_cols is None else rows[0][:n_cols]
    w = len(header)
    out: list[dict[str, str]] = []
    for r in rows[1:]:
        padded = (r + [""] * max(0, w - len(r)))[:w]
        if any(cell != "" for cell in padded):
            out.append(dict(zip(header, padded)))
    return out


def load_csv_raw(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """Read CSV preserving duplicate column headers and handling UTF-8 BOM / Latin-1."""
    raw_bytes = csv_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    reader = csv.reader(text.splitlines())
    header = [h.strip() for h in next(reader)]
    body = [row for row in reader if any(c.strip() != "" for c in row)]
    return header, body


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    """Read CSV into dictionaries (requires unique headers or takes last duplicate if intentional)."""
    header, body = load_csv_raw(csv_path)
    w = len(header)
    return [dict(zip(header, (r + [""] * max(0, w - len(r)))[:w])) for r in body]


def to_float(val: Any) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s in ("", "NA", "N/A", "nan", "NaN", "-"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 2. Continuous-time life-cycle generators, Floquet, Markov, and Telegraph
# ---------------------------------------------------------------------------

def erlang_two_stage_generator(
    tau_out: float,
    tau_in: float,
    k_out: int,
    k_in: int,
    d_wait: float,
    h_rel: float,
    y_tilt: float,
    cold_arrest: bool = False,
    bg_host_loss: float = 0.0,
) -> np.ndarray:
    """Instantaneous transition rate matrix Q(h, y) for an Erlang(k_out) free stage
    and Erlang(k_in) within-host stage (`protocol.md` §2.1).

    Column-stochastic orientation: d x / dt = Q x, where:
      - Outside sub-stages 0 .. k_out-2 progress at rate r_out = k_out / tau_out.
      - Last outside sub-stage (k_out - 1) exits at rate r_out (if d_wait == 1, waste)
        or r_out * [d_wait + (1 - d_wait) * h_rel] (if 0 <= d_wait < 1, contact-limited waiting),
        and enters host sub-stage k_out at rate r_out * exp(y_tilt) * h_rel.
      - Host sub-stages k_out .. k_out + k_in - 1 progress at rate r_in = 0 (if cold_arrest)
        else k_in / tau_in, and suffer extra mortality `bg_host_loss`.
    """
    dim = k_out + k_in
    Q = np.zeros((dim, dim), dtype=float)
    r_out = k_out / tau_out
    r_in = 0.0 if cold_arrest else (k_in / tau_in)

    for j in range(k_out - 1):
        Q[j, j] = -r_out
        Q[j + 1, j] = r_out

    last_out = k_out - 1
    exit_out = r_out if abs(d_wait - 1.0) < 1e-15 else r_out * (d_wait + (1.0 - d_wait) * h_rel)
    Q[last_out, last_out] = -exit_out
    Q[k_out, last_out] = r_out * math.exp(y_tilt) * h_rel

    for j in range(k_in - 1):
        idx = k_out + j
        Q[idx, idx] = -(r_in + bg_host_loss)
        Q[idx + 1, idx] = r_in

    last_in = dim - 1
    Q[last_in, last_in] = -(r_in + bg_host_loss)
    Q[0, last_in] = r_in
    return Q


def floquet_growth_rate(
    tau_out: float,
    tau_in: float,
    k_out: int,
    k_in: int,
    d_wait: float,
    period: float,
    h_high: float,
    h_low: float,
    y_tilt: float,
    high_fraction: float = 0.5,
) -> float:
    """Exact Floquet growth exponent P^{-1} ln rho(M(y)) with spectral shift stabilization."""
    dim = k_out + k_in
    dt_hi = period * high_fraction
    dt_lo = period * (1.0 - high_fraction)

    Q_hi = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_tilt)
    Q_lo = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_tilt)

    s_hi = float(np.max(linalg.eigvals(Q_hi).real))
    s_lo = float(np.max(linalg.eigvals(Q_lo).real))

    M_hi = linalg.expm((Q_hi - s_hi * np.eye(dim)) * dt_hi)
    M_lo = linalg.expm((Q_lo - s_lo * np.eye(dim)) * dt_lo)
    M = M_lo @ M_hi
    rho = float(np.max(np.abs(linalg.eigvals(M))))
    return (math.log(rho) + s_hi * dt_hi + s_lo * dt_lo) / period


def floquet_growth_rate_ode(
    tau_out: float,
    tau_in: float,
    k_out: int,
    k_in: int,
    d_wait: float,
    period: float,
    h_high: float,
    h_low: float,
    y_tilt: float,
) -> float:
    """Independent Runge-Kutta DOP853 ODE monodromy integration for Rule-7 verification."""
    dim = k_out + k_in
    Q_hi = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_tilt)
    Q_lo = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_tilt)

    M = np.eye(dim, dtype=float)
    for Q_piece in (Q_hi, Q_lo):
        sol = integrate.solve_ivp(
            lambda _t, vec: (Q_piece @ vec.reshape(dim, dim)).ravel(),
            (0.0, 0.5 * period),
            M.ravel(),
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        M = sol.y[:, -1].reshape(dim, dim)
    rho = float(np.max(np.abs(linalg.eigvals(M))))
    return math.log(rho) / period


def solve_floquet_loss_D(
    tau_out: float,
    tau_in: float,
    k_out: int = 1,
    k_in: int = 1,
    d_wait: float = 1.0,
    period: float = 4.0,
    h_high: float = 1.8,
    h_low: float = 0.2,
) -> float:
    """Solve r_Floq(D) = 0 for exact periodic fluctuation loss D."""
    return float(
        optimize.brentq(
            lambda y: floquet_growth_rate(tau_out, tau_in, k_out, k_in, d_wait, period, h_high, h_low, y),
            -2.5,
            6.0,
            xtol=1e-12,
        )
    )


def closed_form_D2_discrete_C1(
    tau_out: float,
    tau_in: float,
    d_wait: float,
    sigma: float,
    phi: float,
) -> float:
    """Discrete-time second-order loss per generation Gamma_c (`Proposition C1`, `protocol.md` §2.2)."""
    T_gen = tau_out + tau_in
    if abs(phi) < 1e-15:
        return (sigma ** 2) * (d_wait ** 2) / (2.0 * T_gen)
    theta = phi / (1.0 - phi)
    q_out = 1.0 / (1.0 + tau_out / theta)
    q_in = 1.0 / (1.0 + tau_in / theta)
    R = q_out * q_in
    return (sigma ** 2) * (
        (d_wait ** 2) * (1.0 + 2.0 * theta) / (2.0 * T_gen)
        - d_wait * (R - (1.0 - d_wait) * q_out) / (1.0 - R)
    )


def closed_form_D2_continuous_C2(
    tau_out: float,
    tau_in: float,
    d_wait: float,
    sigma: float,
    theta_ct: float,
) -> float:
    """Continuous-time second-order loss per generation Gamma_c^{ct} (`eq. C2`, `protocol.md` §2.2)."""
    T_gen = tau_out + tau_in
    q_out = 1.0 / (1.0 + tau_out / theta_ct)
    q_in = 1.0 / (1.0 + tau_in / theta_ct)
    R = q_out * q_in
    return (sigma ** 2) * (
        (d_wait ** 2) * (theta_ct / T_gen)
        - d_wait * (R - (1.0 - d_wait) * q_out) / (1.0 - R)
    )


def solve_markov_daily_D(
    tau_out: float,
    tau_in: float,
    k_out: int,
    k_in: int,
    d_wait: float,
    phi_day: float,
    h_high: float,
    h_low: float,
    cold_arrest: bool = False,
    bg_host_loss: float = 0.0,
    n_days: int = 2000,
    burn_days: int = 200,
    n_pairs: int = 12,
    seed: int = 31415926,
) -> tuple[float, float]:
    """Independent continuous-time daily binary Markov Lyapunov threshold loss D (`protocol.md` §2.2)."""
    rng = np.random.default_rng(seed)
    p_stay = 0.5 * (1.0 + phi_day)
    n_paths = 2 * n_pairs
    states = np.empty((n_days, n_paths), dtype=np.int8)
    s0 = (rng.random(n_pairs) < 0.5).astype(np.int8)
    states[0, :n_pairs] = s0
    states[0, n_pairs:] = 1 - s0
    switch = rng.random((n_days - 1, n_pairs)) >= p_stay
    for t in range(1, n_days):
        states[t, :n_pairs] = np.where(switch[t - 1], 1 - states[t - 1, :n_pairs], states[t - 1, :n_pairs])
        states[t, n_pairs:] = 1 - states[t, :n_pairs]

    dim = k_out + k_in
    eye = np.eye(dim)

    def mean_lyapunov(y_val: float) -> tuple[float, np.ndarray]:
        Q_hi = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_val, False, 0.0)
        Q_lo = erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_val, cold_arrest, bg_host_loss)
        sh_hi = float(np.max(linalg.eigvals(Q_hi).real))
        sh_lo = float(np.max(linalg.eigvals(Q_lo).real))
        M_hi = linalg.expm(Q_hi - sh_hi * eye)
        M_lo = linalg.expm(Q_lo - sh_lo * eye)

        v = np.full((n_paths, dim), 1.0 / dim, dtype=float)
        acc = np.zeros(n_paths, dtype=float)
        for t in range(n_days):
            mask_hi = states[t] == 1
            if np.any(mask_hi):
                v[mask_hi] = v[mask_hi] @ M_hi.T
            if np.any(~mask_hi):
                v[~mask_hi] = v[~mask_hi] @ M_lo.T
            norms = v.sum(axis=1)
            v /= norms[:, None]
            if t >= burn_days:
                acc += np.log(norms) + np.where(mask_hi, sh_hi, sh_lo)
        rates = acc / (n_days - burn_days)
        pair_means = 0.5 * (rates[:n_pairs] + rates[n_pairs:])
        return float(pair_means.mean()), pair_means

    root = float(optimize.brentq(lambda y: mean_lyapunov(y)[0], -1.5, 6.0, xtol=1e-7))
    r0, pm0 = mean_lyapunov(root)
    r1, _ = mean_lyapunov(root + 1e-4)
    deriv = (r1 - r0) / 1e-4
    se = float(pm0.std(ddof=1) / math.sqrt(n_pairs) / max(abs(deriv), 1e-9))
    return root, se


def solve_telegraph_ct_D(
    tau_out: float,
    tau_in: float,
    k_out: int,
    k_in: int,
    d_wait: float,
    mean_dwell: float,
    h_high: float,
    h_low: float,
    n_cycles: int = 2000,
    burn_cycles: int = 100,
    n_pairs: int = 12,
    seed: int = 27182818,
) -> tuple[float, float]:
    """Exact unbinned continuous-time symmetric random telegraph loss D via spectral propagation."""
    rng = np.random.default_rng(seed)
    dwell_a = rng.exponential(mean_dwell, size=(n_cycles, n_pairs))
    dwell_b = rng.exponential(mean_dwell, size=(n_cycles, n_pairs))
    # Mirror-paired schedules swap high/low dwell assignments so every pair has exact time-average h = 1
    dt_hi = np.concatenate([dwell_a, dwell_b], axis=1)
    dt_lo = np.concatenate([dwell_b, dwell_a], axis=1)
    total_time = (dt_hi[burn_cycles:] + dt_lo[burn_cycles:]).sum(axis=0)
    dim = k_out + k_in
    n_paths = 2 * n_pairs
    eye = np.eye(dim)

    def decompose(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        evals, V = linalg.eig(Q)
        V_inv = linalg.inv(V)
        shift = float(np.max(evals.real))
        # Verify eigendecomposition accuracy against direct matrix exponential at mean_dwell
        check_mat = (V @ np.diag(np.exp((evals - shift) * mean_dwell)) @ V_inv).real
        ref_mat = linalg.expm((Q - shift * eye) * mean_dwell)
        if np.max(np.abs(check_mat - ref_mat)) > 1e-7:
            raise RuntimeError("Eigendecomposition accuracy check failed")
        return evals, V, V_inv, shift

    def mean_lyapunov(y_val: float) -> tuple[float, np.ndarray]:
        w_hi, V_hi, W_hi, sh_hi = decompose(erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_val))
        w_lo, V_lo, W_lo, sh_lo = decompose(erlang_two_stage_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_val))

        v = np.full((n_paths, dim), 1.0 / dim, dtype=float)
        acc = np.zeros(n_paths, dtype=float)
        for c in range(n_cycles):
            for dt_vec, w, V, W, sh in ((dt_hi[c], w_hi, V_hi, W_hi, sh_hi), (dt_lo[c], w_lo, V_lo, W_lo, sh_lo)):
                v = ((v @ W.T) * np.exp(dt_vec[:, None] * (w - sh))) @ V.T
                v = np.maximum(v.real, 0.0)
                norms = v.sum(axis=1)
                v /= norms[:, None]
                if c >= burn_cycles:
                    acc += np.log(norms) + sh * dt_vec
        rates = acc / total_time
        pair_means = 0.5 * (rates[:n_pairs] + rates[n_pairs:])
        return float(pair_means.mean()), pair_means

    root = float(optimize.brentq(lambda y: mean_lyapunov(y)[0], -1.0, 3.0, xtol=1e-9))
    r0, pm0 = mean_lyapunov(root)
    r1, _ = mean_lyapunov(root + 1e-4)
    deriv = (r1 - r0) / 1e-4
    se = float(pm0.std(ddof=1) / math.sqrt(n_pairs) / max(abs(deriv), 1e-9))
    return root, se


# ---------------------------------------------------------------------------
# 3. Independent statistical tests, exact enumerations, and survival MLEs
# ---------------------------------------------------------------------------

def exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Exact permutation distribution of Spearman's rho under tied group sizes via DP over rank counts."""
    rx = np.rint(2.0 * stats.rankdata(x)).astype(int)
    ry = np.rint(2.0 * stats.rankdata(y)).astype(int)
    obs_rho = float(stats.spearmanr(x, y).statistic)

    x_levels = sorted(set(rx))
    capacities = tuple(int(np.sum(rx == lv)) for lv in x_levels)
    y_counts = sorted(collections.Counter(int(v) for v in ry).items(), key=lambda kv: kv[1])
    obs_dot = int(np.sum(rx * ry))

    # Dynamic programming over distinct y-rank values
    dp: dict[tuple[tuple[int, ...], int], int] = {(capacities, 0): 1}

    def compositions(total: int, caps: tuple[int, ...]):
        if len(caps) == 1:
            if total <= caps[0]:
                yield (total,)
            return
        for take0 in range(min(total, caps[0]) + 1):
            for rest in compositions(total - take0, caps[1:]):
                yield (take0,) + rest

    for y_val, cnt in y_counts:
        next_dp: dict[tuple[tuple[int, ...], int], int] = collections.defaultdict(int)
        for (rem_caps, score), ways in dp.items():
            for alloc in compositions(cnt, rem_caps):
                mult = 1
                for cap_i, a_i in zip(rem_caps, alloc):
                    mult *= math.comb(cap_i, a_i)
                new_caps = tuple(cap_i - a_i for cap_i, a_i in zip(rem_caps, alloc))
                new_score = score + y_val * sum(lv * a_i for lv, a_i in zip(x_levels, alloc))
                next_dp[(new_caps, new_score)] += ways * mult
        dp = next_dp

    dist: dict[int, int] = {score: ways for (_caps, score), ways in dp.items()}
    tot_alloc = sum(dist.values())
    expected_alloc = math.factorial(len(y)) // math.prod(math.factorial(c) for _, c in y_counts)
    assert tot_alloc == expected_alloc, f"DP count mismatch: {tot_alloc} != {expected_alloc}"

    p_less = sum(w for s, w in dist.items() if s <= obs_dot) / tot_alloc
    p_greater = sum(w for s, w in dist.items() if s >= obs_dot) / tot_alloc
    return {
        "rho": obs_rho,
        "p_less_exact": float(p_less),
        "p_greater_exact": float(p_greater),
        "total_allocations": int(tot_alloc),
        "distinct_scores": len(dist),
    }


def mc_spearman_and_boot_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 100000,
    n_boot: int = 10000,
    seed: int = 20261115,
) -> dict[str, float]:
    """Monte Carlo permutation p-values (add-one) and unit-resampling bootstrap 95% CI for Spearman's rho."""
    rng = np.random.default_rng(seed)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    zx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    zy = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    rho = float(zx @ zy)

    perm_idx = np.argsort(rng.random((n_perm, len(y))), axis=1)
    perm_rho = zy[perm_idx] @ zx
    p_less_mc = float((1 + np.sum(perm_rho <= rho + 1e-15)) / (n_perm + 1))
    p_greater_mc = float((1 + np.sum(perm_rho >= rho - 1e-15)) / (n_perm + 1))

    rng_b = np.random.default_rng(seed + 7)
    boot_vals = np.empty(n_boot, dtype=float)
    n = len(x)
    for b in range(n_boot):
        ix = rng_b.integers(0, n, size=n)
        boot_vals[b] = stats.spearmanr(x[ix], y[ix]).statistic
    return {
        "rho": rho,
        "p_less_mc": p_less_mc,
        "p_greater_mc": p_greater_mc,
        "ci_lo": float(np.nanquantile(boot_vals, 0.025)),
        "ci_hi": float(np.nanquantile(boot_vals, 0.975)),
    }


def ols_slope_permutation(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None = None,
    n_perm: int = 100000,
    seed: int = 20261115,
) -> dict[str, float]:
    """Exact OLS / FWL multiple-regression slope, exact t-interval, and Freedman-Lane permutation test."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if covariates is None:
        Z = np.ones((n, 1), dtype=float)
    else:
        cov_2d = np.asarray(covariates, dtype=float).reshape(n, -1)
        Z = np.column_stack([np.ones(n, dtype=float), cov_2d])

    # Frisch-Waugh-Lovell projection orthogonal to Z
    Q_z, _ = np.linalg.qr(Z, mode="reduced")
    x_perp = x_arr - Q_z @ (Q_z.T @ x_arr)
    y_perp = y_arr - Q_z @ (Q_z.T @ y_arr)

    ssx = float(x_perp @ x_perp)
    beta = float((x_perp @ y_perp) / ssx)
    resid_full = y_perp - beta * x_perp
    df = n - Z.shape[1] - 1
    se = math.sqrt(float(resid_full @ resid_full) / df / ssx)
    t_crit = float(stats.t.ppf(0.975, df))
    t_stat = beta / se
    p_two_t = float(2.0 * stats.t.sf(abs(t_stat), df))

    # Exact Freedman-Lane permutation: permute reduced-model residuals y_perp,
    # add back fitted_z, and re-project orthogonal to Z before computing slope
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, n)), axis=1)
    y_perm_raw = y_perp[idx]  # (n_perm, n); fitted_z is orthogonal to x_perp and killed by I - Q_z Q_z^T
    # Project each permuted vector orthogonal to Z: Y_p_perp = Y_p - (Y_p @ Q_z) @ Q_z.T
    y_perm_perp = y_perm_raw - (y_perm_raw @ Q_z) @ Q_z.T
    # Denominator under Freedman-Lane t-statistic or slope: here slope is (y_perm_perp @ x_perp) / ssx
    # Note: since x_perp is already orthogonal to Q_z, y_perm_perp @ x_perp == y_perm_raw @ x_perp identically!
    slopes_perm = (y_perm_perp @ x_perp) / ssx
    p_less = float((1 + np.sum(slopes_perm <= beta + 1e-15)) / (n_perm + 1))
    p_greater = float((1 + np.sum(slopes_perm >= beta - 1e-15)) / (n_perm + 1))

    return {
        "slope": beta,
        "se": se,
        "ci_lo": beta - t_crit * se,
        "ci_hi": beta + t_crit * se,
        "df": int(df),
        "p_less": p_less,
        "p_greater": p_greater,
        "p_two_parametric": p_two_t,
    }


def exact_stratified_two_group_test(
    values: np.ndarray,
    is_focal: np.ndarray,
    strata: list[str],
    n_boot: int = 10000,
    seed: int = 20261115,
) -> dict[str, Any]:
    """Exact block-permutation test (meet-in-the-middle over stratum subset sums)
    plus stratified within-cell SE and stratified within-cell bootstrap 95% CI.
    """
    vals = np.asarray(values, dtype=float)
    grp = np.asarray(is_focal, dtype=bool)
    n1 = int(np.sum(grp))
    n0 = len(vals) - n1
    obs_diff = float(vals[grp].mean() - vals[~grp].mean())
    obs_sum1 = float(vals[grp].sum())

    unique_strata = sorted(set(strata))
    stratum_sums: list[np.ndarray] = []
    var_diff_strat = 0.0

    for st in unique_strata:
        idx = [i for i, s in enumerate(strata) if s == st]
        v_st = vals[idx]
        g_st = grp[idx]
        k1 = int(np.sum(g_st))
        k0 = len(idx) - k1
        # Within-cell variance contribution to mean1 - mean0
        if k1 > 1:
            var_diff_strat += float(v_st[g_st].var(ddof=1)) * (k1 / (n1 ** 2))
        if k0 > 1:
            var_diff_strat += float(v_st[~g_st].var(ddof=1)) * (k0 / (n0 ** 2))
        # All (len(idx) choose k1) subset sums within this stratum
        from itertools import combinations
        subs = np.fromiter((float(v_st[list(c)].sum()) for c in combinations(range(len(idx)), k1)), dtype=float)
        stratum_sums.append(subs)

    def minkowski_sum(arrays: list[np.ndarray]) -> np.ndarray:
        out = np.array([0.0], dtype=float)
        for arr in arrays:
            out = (out[:, None] + arr[None, :]).ravel()
        return out

    mid = len(stratum_sums) // 2
    left = minkowski_sum(stratum_sums[:mid])
    right = np.sort(minkowski_sum(stratum_sums[mid:]))
    tot = len(left) * len(right)

    le_cnt = sum(int(np.searchsorted(right, obs_sum1 - u + 1e-12, side="right")) for u in left)
    ge_cnt = sum(int(len(right) - np.searchsorted(right, obs_sum1 - u - 1e-12, side="left")) for u in left)

    # Stratified cell-preserving bootstrap
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=float)
    for st in unique_strata:
        mask_st = np.array([s == st for s in strata], dtype=bool)
        v1 = vals[mask_st & grp]
        v0 = vals[mask_st & (~grp)]
        if len(v1) > 0:
            boot += rng.choice(v1, size=(n_boot, len(v1)), replace=True).sum(axis=1) / n1
        if len(v0) > 0:
            boot -= rng.choice(v0, size=(n_boot, len(v0)), replace=True).sum(axis=1) / n0

    return {
        "diff": obs_diff,
        "mean_focal": float(vals[grp].mean()),
        "mean_ref": float(vals[~grp].mean()),
        "se_stratified": math.sqrt(var_diff_strat),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_less_exact": float(le_cnt / tot),
        "p_greater_exact": float(ge_cnt / tot),
        "exact_allocations": int(tot),
        "n": int(len(vals)),
    }


def partial_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    spearman: bool = False,
) -> dict[str, float]:
    """Partial Pearson or Spearman correlation with exact t-tail p-values and Fisher-z 95% CI."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    za = np.asarray(controls, dtype=float).reshape(len(xa), -1)
    valid = np.isfinite(xa) & np.isfinite(ya) & np.all(np.isfinite(za), axis=1)
    xa, ya, za = xa[valid], ya[valid], za[valid]
    n = len(xa)

    if spearman:
        xa = stats.rankdata(xa)
        ya = stats.rankdata(ya)
        za = np.column_stack([stats.rankdata(za[:, j]) for j in range(za.shape[1])])

    X_design = np.column_stack([np.ones(n, dtype=float), za])
    q_rank = int(np.linalg.matrix_rank(X_design))
    rx = xa - X_design @ np.linalg.lstsq(X_design, xa, rcond=None)[0]
    ry = ya - X_design @ np.linalg.lstsq(X_design, ya, rcond=None)[0]

    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - q_rank - 1
    t_val = r * math.sqrt(df / max(1.0 - r * r, 1e-15))
    p_less = float(stats.t.cdf(t_val, df))
    p_greater = float(stats.t.sf(t_val, df))

    z_fish = math.atanh(max(-0.9999999, min(0.9999999, r)))
    se_z = 1.0 / math.sqrt(max(n - q_rank - 2, 1))
    z_crit = float(stats.norm.ppf(0.975))
    return {
        "n": int(n),
        "r": r,
        "df": int(df),
        "p_less": p_less,
        "p_greater": p_greater,
        "ci_lo": math.tanh(z_fish - z_crit * se_z),
        "ci_hi": math.tanh(z_fish + z_crit * se_z),
    }


def fit_binomial_weibull_or_exp(
    days: np.ndarray,
    dead: np.ndarray,
    total: np.ndarray,
    weibull: bool = False,
) -> dict[str, Any]:
    """Maximum-likelihood binomial survival fit S(t) = exp(-(lambda * t)^k)
    using exact analytic gradients and observed Fisher information on log-parameters.
    """
    t = np.asarray(days, dtype=float)
    d = np.asarray(dead, dtype=float)
    n = np.asarray(total, dtype=float)
    ok = np.isfinite(t) & np.isfinite(d) & np.isfinite(n) & (n > 0) & (t > 0)
    t, d, n = t[ok], d[ok], n[ok]
    live = n - d

    if not weibull:
        # Exact 1D score equation for log(lambda): d(nll)/d(u) = sum( live * h - d * h / expm1(h) ), h = lam * t
        def nll_1d(u_arr: np.ndarray) -> float:
            lam = math.exp(float(u_arr[0]))
            h = lam * t
            return float(np.sum(live * h - d * np.log(-np.expm1(-h))))

        def grad_1d(u_arr: np.ndarray) -> np.ndarray:
            lam = math.exp(float(u_arr[0]))
            h = lam * t
            # d/du [-log(1 - exp(-h))] = - h * exp(-h) / (1 - exp(-h)) = - h / expm1(h)
            g = float(np.sum(live * h - d * h / np.expm1(h)))
            return np.array([g], dtype=float)

        sol = optimize.minimize(nll_1d, x0=np.array([math.log(0.06)]), jac=grad_1d, method="BFGS")
        u_hat = float(sol.x[0])
        lam_hat = math.exp(u_hat)
        h = lam_hat * t
        # Exact second derivative d^2(nll)/du^2 = sum( live * h - d * h * expm1(h - h_0)? ):
        # Since d/du (h / expm1(h)) = h * (expm1(h) - h * exp(h)) / expm1(h)^2
        em1 = np.expm1(h)
        d2_u = float(np.sum(live * h - d * h * (em1 - h * np.exp(h)) / (em1 ** 2)))
        se_u = 1.0 / math.sqrt(d2_u)
        z_crit = float(stats.norm.ppf(0.975))
        nll_val = nll_1d(sol.x)
        return {
            "model": "exponential",
            "fov_n": int(len(t)),
            "spores_n": int(n.sum()),
            "lambda_day": lam_hat,
            "lambda_lo": math.exp(u_hat - z_crit * se_u),
            "lambda_hi": math.exp(u_hat + z_crit * se_u),
            "k": 1.0,
            "k_lo": 1.0,
            "k_hi": 1.0,
            "mean_days": 1.0 / lam_hat,
            "nll": nll_val,
            "AIC": 2.0 * 1 + 2.0 * nll_val,
            "converged": bool(sol.success),
        }

    # 2D Weibull on (u = ln lambda, v = ln k): h(u, v) = exp(k * (u + ln t))
    ln_t = np.log(t)

    def nll_2d(theta: np.ndarray) -> float:
        u, v = float(theta[0]), float(theta[1])
        if abs(u) > 30.0 or abs(v) > 15.0:
            return 1e100
        k = math.exp(v)
        h = np.exp(np.clip(k * (u + ln_t), -500.0, 500.0))
        return float(np.sum(live * h - d * np.log(np.maximum(-np.expm1(-h), 1e-300))))

    def grad_2d(theta: np.ndarray) -> np.ndarray:
        u, v = float(theta[0]), float(theta[1])
        k = math.exp(min(v, 15.0))
        z = u + ln_t
        kz = np.clip(k * z, -500.0, 500.0)
        h = np.exp(kz)
        w = np.where(h > 50.0, live, live - d / np.expm1(np.minimum(h, 50.0)))
        return np.array([float(np.sum(w * k * h)), float(np.sum(w * k * z * h))], dtype=float)

    def hess_2d(theta: np.ndarray) -> np.ndarray:
        u, v = float(theta[0]), float(theta[1])
        k = math.exp(v)
        z = u + ln_t
        h = np.exp(k * z)
        em1 = np.expm1(h)
        w1 = live - d / em1
        w2 = d * np.exp(h) / (em1 ** 2)  # d(w1)/dh
        du = k * h
        dv = k * z * h
        h_uu = float(np.sum(w2 * du * du + w1 * k * du))
        h_uv = float(np.sum(w2 * du * dv + w1 * (du + k * z * du)))
        h_vv = float(np.sum(w2 * dv * dv + w1 * (dv + k * z * dv)))
        return np.array([[h_uu, h_uv], [h_uv, h_vv]], dtype=float)

    sol = optimize.minimize(nll_2d, x0=np.array([math.log(0.06), 0.0]), jac=grad_2d, method="BFGS")
    u_hat, v_hat = float(sol.x[0]), float(sol.x[1])
    cov = np.linalg.inv(hess_2d(sol.x))
    se_u, se_v = math.sqrt(float(cov[0, 0])), math.sqrt(float(cov[1, 1]))
    lam_hat, k_hat = math.exp(u_hat), math.exp(v_hat)
    z_crit = float(stats.norm.ppf(0.975))
    nll_val = nll_2d(sol.x)
    return {
        "model": "Weibull",
        "fov_n": int(len(t)),
        "spores_n": int(n.sum()),
        "lambda_day": lam_hat,
        "lambda_lo": math.exp(u_hat - z_crit * se_u),
        "lambda_hi": math.exp(u_hat + z_crit * se_u),
        "k": k_hat,
        "k_lo": math.exp(v_hat - z_crit * se_v),
        "k_hi": math.exp(v_hat + z_crit * se_v),
        "mean_days": float(special.gamma(1.0 + 1.0 / k_hat) / lam_hat),
        "nll": nll_val,
        "AIC": 2.0 * 2 + 2.0 * nll_val,
        "converged": bool(sol.success),
    }


# ---------------------------------------------------------------------------
# 4. Rule-7 Benchmarks (must pass before any raw data file is touched)
# ---------------------------------------------------------------------------

def run_benchmarks() -> list[dict[str, Any]]:
    """Verify known analytical identities, published reference values, and brute-force
    combinatorial distributions within pre-stated tolerances (`study rules` Rule 7).
    """
    records: list[dict[str, Any]] = []

    # BM1: Zero-mortality contact waiting (d = 0) gives exact D = 0 at large swing
    d0_floq = solve_floquet_loss_D(10.0, 5.0, k_out=1, k_in=4, d_wait=0.0, period=30.0, h_high=1.8, h_low=0.2)
    assert abs(d0_floq) < 1e-11, f"BM1a failed: {d0_floq}"
    records.append({"id": "BM1a_d0_Floquet", "value": d0_floq, "ref": 0.0, "error": abs(d0_floq), "tol": 1e-11, "passed": True})

    d0_tel, _ = solve_telegraph_ct_D(10.0, 5.0, k_out=1, k_in=4, d_wait=0.0, mean_dwell=15.0, h_high=1.8, h_low=0.2, n_cycles=600, n_pairs=6)
    assert abs(d0_tel) < 1e-10, f"BM1b failed: {d0_tel}"
    records.append({"id": "BM1b_d0_telegraph", "value": d0_tel, "ref": 0.0, "error": abs(d0_tel), "tol": 1e-10, "passed": True})

    # BM2: One-stage limit (tau_in -> 0, Statement ring-B9a) gives D -> 0
    d_1stage = solve_floquet_loss_D(1.0, 1e-7, k_out=1, k_in=1, d_wait=1.0, period=4.0, h_high=1.8, h_low=0.2)
    assert abs(d_1stage) < 1e-6, f"BM2 failed: {d_1stage}"
    records.append({"id": "BM2_one_stage_limit", "value": d_1stage, "ref": 0.0, "error": abs(d_1stage), "tol": 1e-6, "passed": True})

    # BM3: Matrix-exponential Floquet vs independent DOP853 ODE integration
    for d_val in (1.0, 0.5, 0.1):
        for ko, ki in ((1, 1), (4, 4)):
            r_exp = floquet_growth_rate(10.0, 5.0, ko, ki, d_val, 30.0, 1.8, 0.2, 0.12)
            r_ode = floquet_growth_rate_ode(10.0, 5.0, ko, ki, d_val, 30.0, 1.8, 0.2, 0.12)
            err = abs(r_exp - r_ode)
            assert err < 1e-7, f"BM3 failed at d={d_val}, k=({ko},{ki}): {err}"
            records.append({"id": f"BM3_Floquet_vs_ODE_d{d_val}_k{ko}x{ki}", "value": r_exp, "ref": r_ode, "error": err, "tol": 1e-7, "passed": True})

    # BM4: Known reference values from `notes/decisive-experiment-v2.md` Table 5.2
    for label_id, ko, ki, d_val, per, ref_val, tol in [
        ("BM4a_ref_waste_1x1_P30", 1, 1, 1.0, 30.0, 0.092043, 5e-4),
        ("BM4b_ref_wait05_1x1_P30", 1, 1, 0.5, 30.0, 0.056857, 5e-4),
        ("BM4c_ref_resonance_16x16_P15", 16, 16, 1.0, 15.0, -0.2797, 5e-4),
    ]:
        val = solve_floquet_loss_D(10.0, 5.0, ko, ki, d_val, per, 1.8, 0.2)
        err = abs(val - ref_val)
        assert err < tol, f"{label_id} failed: {val} vs {ref_val}"
        records.append({"id": label_id, "value": val, "ref": ref_val, "error": err, "tol": tol, "passed": True})

    # BM5: Small-noise telegraph simulation vs closed-form continuous-time eq. C2
    # Symmetric telegraph with mean dwell m has correlation exp(-2 t / m), so theta_ct = m / 2
    delta_small = 0.05
    m_dwell = 15.0
    c2_pred = closed_form_D2_continuous_C2(10.0, 5.0, d_wait=0.5, sigma=delta_small, theta_ct=0.5 * m_dwell)
    tel_small, tel_se = solve_telegraph_ct_D(10.0, 5.0, 1, 1, 0.5, m_dwell, 1.0 + delta_small, 1.0 - delta_small, n_cycles=4000, n_pairs=16)
    err_c2 = abs(tel_small - c2_pred)
    tol_c2 = 4.0 * tel_se + 5e-6
    assert err_c2 < tol_c2, f"BM5 failed: {tel_small} vs {c2_pred} (tol {tol_c2})"
    records.append({"id": "BM5_telegraph_vs_C2_small_noise", "value": tel_small, "ref": c2_pred, "error": err_c2, "tol": tol_c2, "passed": True})

    # BM6: Exact DP Spearman vs brute-force permutations on a tied toy sample (n = 8)
    from itertools import permutations
    toy_x = np.array([0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.0, 1.0])
    toy_y = np.array([3.0, 24.0, 24.0, 12.0, 24.0, 3.0, 18.0, 24.0])
    dp_res = exact_spearman_permutation(toy_x, toy_y)
    rx = stats.rankdata(toy_x)
    ry = stats.rankdata(toy_y)
    obs_s = float(rx @ ry)
    bf_le = sum(1 for p in permutations(ry) if float(rx @ np.array(p)) <= obs_s + 1e-12) / math.factorial(8)
    err_dp = abs(dp_res["p_less_exact"] - bf_le)
    assert err_dp < 1e-14, f"BM6 failed: {dp_res['p_less_exact']} vs {bf_le}"
    records.append({"id": "BM6_exact_DP_Spearman_vs_bruteforce", "value": dp_res["p_less_exact"], "ref": bf_le, "error": err_dp, "tol": 1e-14, "passed": True})

    return records


if __name__ == "__main__":
    bms = run_benchmarks()
    print(f"Passed {len(bms)} Rule-7 benchmarks:")
    for b in bms:
        print(f"  {b['id']}: value={b['value']:.8g}, ref={b['ref']:.8g}, err={b['error']:.2e} <= {b['tol']:.2e}")
