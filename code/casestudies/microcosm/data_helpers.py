"""Safe untrusted-data readers, statistical permutation tests, and continuous-time solvers.

Used by `casestudies/microcosm/analysis.py`. Standard library + numpy + scipy only.
"""
from __future__ import annotations

import csv
import math
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy import linalg, optimize, stats


def read_csv_dicts(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def read_xlsx_sheet(path: str, sheet_idx: int = 1) -> list[list[str]]:
    """Safe pure-Python XML reader for `.xlsx` worksheets (never executes macros/formulas)."""
    with zipfile.ZipFile(path, "r") as z:
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sroot.findall(".//m:si", ns):
                sst.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        sxml = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet_idx}.xml"))
        rows_out: list[list[str]] = []
        for row in sxml.findall(".//m:row", ns):
            rmap: dict[int, str] = {}
            max_col = -1
            for c in row.findall("m:c", ns):
                ref = c.attrib.get("r", "")
                m = re.match(r"([A-Z]+)", ref)
                col_idx = 0
                if m:
                    for ch in m.group(1):
                        col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                    col_idx -= 1
                else:
                    col_idx = len(rmap)
                max_col = max(max_col, col_idx)
                v = c.find("m:v", ns)
                t = c.attrib.get("t", "")
                if v is not None and v.text is not None:
                    val = sst[int(v.text)] if t == "s" else v.text
                else:
                    isel = c.find(".//m:t", ns)
                    val = isel.text if (isel is not None and isel.text is not None) else ""
                rmap[col_idx] = val.strip()
            if max_col >= 0:
                rows_out.append([rmap.get(i, "") for i in range(max_col + 1)])
        return rows_out


def series_log_stats(vals: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean_raw, sd_ln, detrended_sd_ln, lag1_r_ln) on positive entries."""
    arr = np.asarray([v for v in vals if v is not None and not math.isnan(v) and v > 0], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_raw = float(arr.mean())
    if len(arr) < 2:
        return mean_raw, float("nan"), float("nan"), float("nan")
    lns = np.log(arr)
    sd_ln = float(lns.std(ddof=1))
    t_idx = np.arange(len(lns), dtype=float)
    if len(lns) >= 3:
        slope, intercept = np.polyfit(t_idx, lns, 1)
        resid = lns - (slope * t_idx + intercept)
        det_sd = float(resid.std(ddof=1))
    else:
        det_sd = sd_ln
    if len(lns) >= 3 and np.std(lns[:-1]) > 1e-12 and np.std(lns[1:]) > 1e-12:
        r1 = float(np.corrcoef(lns[:-1], lns[1:])[0, 1])
    else:
        r1 = float("nan")
    return mean_raw, sd_ln, det_sd, r1


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 20000, seed: int = 20261115) -> dict[str, float]:
    """Vectorized Monte Carlo permutation test for Spearman's rho (one-sided lower & upper)."""
    rng = np.random.default_rng(seed)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx_n = (rx - rx.mean()) / max(np.linalg.norm(rx - rx.mean()), 1e-15)
    ry_n = (ry - ry.mean()) / max(np.linalg.norm(ry - ry.mean()), 1e-15)
    rho_obs = float(rx_n @ ry_n)
    idx = np.argsort(rng.random((n_perm, len(y))), axis=1)
    r_perm = ry_n[idx] @ rx_n
    p_less = float((1 + np.sum(r_perm <= rho_obs + 1e-15)) / (n_perm + 1))
    p_greater = float((1 + np.sum(r_perm >= rho_obs - 1e-15)) / (n_perm + 1))
    return {"rho": rho_obs, "p_less": p_less, "p_greater": p_greater}


def perm_ols_slope(x: np.ndarray, y: np.ndarray, z_cov: np.ndarray | None = None,
                   n_perm: int = 20000, seed: int = 20261115) -> dict[str, float]:
    """OLS slope (optionally Freedman-Lane residualized on covariate z_cov) with permutation p-values."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if z_cov is not None:
        z = np.asarray(z_cov, dtype=float)
        Xz = np.column_stack([np.ones(len(z)), z])
        bx, *_ = np.linalg.lstsq(Xz, x, rcond=None)
        by, *_ = np.linalg.lstsq(Xz, y, rcond=None)
        x_use = x - Xz @ bx
        y_use = y - Xz @ by
    else:
        x_use = x - x.mean()
        y_use = y - y.mean()
    denom = float(x_use @ x_use)
    slope = float((x_use @ y_use) / denom)
    dof = len(y) - (3 if z_cov is not None else 2)
    resid = y_use - slope * x_use
    se = math.sqrt(float((resid @ resid) / max(dof, 1)) / denom)
    idx = np.argsort(rng.random((n_perm, len(y))), axis=1)
    s_perm = (y_use[idx] @ x_use) / denom
    p_less = float((1 + np.sum(s_perm <= slope + 1e-15)) / (n_perm + 1))
    p_greater = float((1 + np.sum(s_perm >= slope - 1e-15)) / (n_perm + 1))
    return {"slope": slope, "se": se, "p_less": p_less, "p_greater": p_greater}


def stratified_diff_perm(values: np.ndarray, group_is_one: np.ndarray, strata: list[str],
                         n_perm: int = 20000, seed: int = 20261115) -> dict[str, float]:
    """Stratified permutation test for mean(group==1) - mean(group==0) within strata."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    g = np.asarray(group_is_one, dtype=bool)
    obs_diff = float(values[g].mean() - values[~g].mean())
    se_diff = math.sqrt(float(values[g].var(ddof=1) / np.sum(g) + values[~g].var(ddof=1) / np.sum(~g)))

    unique_s = sorted(set(strata))
    s_indices = [np.array([i for i, s in enumerate(strata) if s == us], dtype=int) for us in unique_s]
    s_counts = [int(np.sum(g[idxs])) for idxs in s_indices]
    n1_tot = int(np.sum(g))
    n0_tot = len(values) - n1_tot

    perm_diffs = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        sum1 = 0.0
        for idxs, k1 in zip(s_indices, s_counts):
            perm_idx = rng.permutation(idxs)
            sum1 += float(values[perm_idx[:k1]].sum())
        sum0 = float(values.sum()) - sum1
        perm_diffs[p] = sum1 / n1_tot - sum0 / n0_tot

    p_less = float((1 + np.sum(perm_diffs <= obs_diff + 1e-15)) / (n_perm + 1))
    p_greater = float((1 + np.sum(perm_diffs >= obs_diff - 1e-15)) / (n_perm + 1))
    return {
        "diff": obs_diff, "se": se_diff,
        "mean1": float(values[g].mean()), "mean0": float(values[~g].mean()),
        "p_less": p_less, "p_greater": p_greater,
    }


def build_ct_generator(tau_out: float, tau_in: float, k_out: int, k_in: int,
                       d_wait: float, h_val: float, y_tilt: float,
                       arrest_host: bool = False, host_bg_loss: float = 0.0) -> np.ndarray:
    """Continuous-time generator for two-stage Erlang chain (outside wait, in-host development).

    If `arrest_host=True` (used for 5 C cold arrest in *Holospora* Model M2B), in-host progression
    to infectious spore release is arrested while background host turnover/mortality `host_bg_loss` continues.
    """
    n = k_out + k_in
    Q = np.zeros((n, n), dtype=float)
    r_out = k_out / tau_out
    r_in = 0.0 if arrest_host else (k_in / tau_in)

    for i in range(k_out - 1):
        Q[i, i] = -r_out
        Q[i + 1, i] = r_out

    last_out = k_out - 1
    first_in = k_out
    if abs(d_wait - 1.0) < 1e-15:
        Q[last_out, last_out] = -r_out
    else:
        Q[last_out, last_out] = -r_out * (d_wait + (1.0 - d_wait) * h_val)
    Q[first_in, last_out] = r_out * math.exp(y_tilt) * h_val

    for j in range(k_in - 1):
        idx = k_out + j
        Q[idx, idx] = -(r_in + host_bg_loss)
        Q[idx + 1, idx] = r_in

    last_in = k_out + k_in - 1
    Q[last_in, last_in] = -(r_in + host_bg_loss)
    Q[0, last_in] = r_in
    return Q


def floquet_growth(tau_out: float, tau_in: float, k_out: int, k_in: int,
                   d_wait: float, period: float, h_high: float, h_low: float,
                   y_tilt: float, frac_high: float = 0.5) -> float:
    Q_hi = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y_tilt)
    Q_lo = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y_tilt)
    M = linalg.expm(Q_lo * (period * (1.0 - frac_high))) @ linalg.expm(Q_hi * (period * frac_high))
    rho = float(np.max(np.abs(linalg.eigvals(M))))
    return math.log(rho) / period


def floquet_loss_D(tau_out: float, tau_in: float, k_out: int = 1, k_in: int = 1,
                   d_wait: float = 1.0, period: float = 4.0,
                   h_high: float = 1.8, h_low: float = 0.2) -> float:
    return float(optimize.brentq(
        lambda y: floquet_growth(tau_out, tau_in, k_out, k_in, d_wait, period, h_high, h_low, y),
        -2.5, 6.0, xtol=1e-11
    ))


def telegraph_ct_loss_D(tau_out: float, tau_in: float, k_out: int, k_in: int,
                        d_wait: float, mean_dwell: float, h_high: float, h_low: float,
                        n_cycles: int = 8000, n_pairs: int = 20, seed: int = 20261115) -> tuple[float, float]:
    """Continuous-time symmetric random telegraph loss D via antithetic dwell pairs."""
    rng = np.random.default_rng(seed)
    # Each cycle consists of one high dwell and one low dwell (exponential of mean `mean_dwell`)
    u_hi = np.clip(rng.random((n_cycles, n_pairs)), 1e-12, 1.0 - 1e-12)
    u_lo = np.clip(rng.random((n_cycles, n_pairs)), 1e-12, 1.0 - 1e-12)
    # Antithetic quantiles reduce dwell-length Monte Carlo variance dramatically
    dt_hi = -mean_dwell * np.log(np.concatenate([u_hi, 1.0 - u_hi], axis=1))
    dt_lo = -mean_dwell * np.log(np.concatenate([u_lo, 1.0 - u_lo], axis=1))
    # Quantize dwells into 64 quantile bins per state for fast exact matrix exponentials
    q_edges = np.linspace(0.0, 1.0, 65)
    q_mids = 0.5 * (q_edges[:-1] + q_edges[1:])
    grid_dwells = -mean_dwell * np.log(1.0 - q_mids)
    bin_hi = np.clip(np.searchsorted(grid_dwells, dt_hi), 0, 63)
    bin_lo = np.clip(np.searchsorted(grid_dwells, dt_lo), 0, 63)
    tot_time = (grid_dwells[bin_hi][100:] + grid_dwells[bin_lo][100:]).sum(axis=0)

    def eval_rate(y: float) -> tuple[float, np.ndarray]:
        Q_hi = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y)
        Q_lo = build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y)
        M_hi_bank = np.array([linalg.expm(Q_hi * d) for d in grid_dwells])
        M_lo_bank = np.array([linalg.expm(Q_lo * d) for d in grid_dwells])
        dim = k_out + k_in
        x = np.full((2 * n_pairs, dim), 1.0 / dim)
        logs = np.zeros(2 * n_pairs)
        for c in range(n_cycles):
            x = np.einsum("rij,rj->ri", M_hi_bank[bin_hi[c]], x)
            s1 = x.sum(axis=1)
            x /= s1[:, None]
            x = np.einsum("rij,rj->ri", M_lo_bank[bin_lo[c]], x)
            s2 = x.sum(axis=1)
            x /= s2[:, None]
            if c >= 100:
                logs += np.log(s1) + np.log(s2)
        rates = logs / tot_time
        pm = 0.5 * (rates[:n_pairs] + rates[n_pairs:])
        return float(pm.mean()), pm

    y_root = optimize.brentq(lambda y: eval_rate(y)[0], -1.5, 4.0, xtol=1e-6)
    r0, pm0 = eval_rate(y_root)
    r1, _ = eval_rate(y_root + 1e-4)
    slope = (r1 - r0) / 1e-4
    se = float(pm0.std(ddof=1) / math.sqrt(n_pairs) / max(abs(slope), 1e-8))
    return float(y_root), se


def markov_daily_loss_D(tau_out: float, tau_in: float, k_out: int, k_in: int,
                        d_wait: float, phi_day: float, h_high: float, h_low: float,
                        arrest_low: bool = False, low_bg_loss: float = 0.0,
                        horizon_days: int = 3500, n_pairs: int = 32, seed: int = 20261115) -> tuple[float, float]:
    """Threshold loss D relative to constant-23C/mean generator for daily binary Markov switching.

    Note: when `arrest_low=True`, `D` is measured relative to the permissive 23 C generator (`y=0` gives R0=1 at h=1),
    capturing both the 50% time lost to 5 C arrest and the dynamic penalty of longer cold streaks under red noise.
    """
    rng = np.random.default_rng(seed)
    p_stay = 0.5 * (1.0 + phi_day)
    states = np.empty((horizon_days, 2 * n_pairs), dtype=int)
    u0 = rng.random(n_pairs) < 0.5
    states[0, :n_pairs] = u0.astype(int)
    states[0, n_pairs:] = 1 - states[0, :n_pairs]
    flips = rng.random((horizon_days - 1, n_pairs)) >= p_stay
    for t in range(1, horizon_days):
        states[t, :n_pairs] = np.where(flips[t - 1], 1 - states[t - 1, :n_pairs], states[t - 1, :n_pairs])
        states[t, n_pairs:] = 1 - states[t, :n_pairs]

    burn = 200

    def eval_growth(y: float) -> tuple[float, np.ndarray]:
        M_hi = linalg.expm(build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_high, y, False, 0.0))
        M_lo = linalg.expm(build_ct_generator(tau_out, tau_in, k_out, k_in, d_wait, h_low, y, arrest_low, low_bg_loss))
        dim = k_out + k_in
        x = np.full((2 * n_pairs, dim), 1.0 / dim)
        logs = np.zeros(2 * n_pairs)
        for t in range(horizon_days):
            hi_m = states[t] == 1
            if np.any(hi_m):
                x[hi_m] = x[hi_m] @ M_hi.T
            if np.any(~hi_m):
                x[~hi_m] = x[~hi_m] @ M_lo.T
            s = x.sum(axis=1)
            x /= s[:, None]
            if t >= burn:
                logs += np.log(s)
        logs /= (horizon_days - burn)
        pm = 0.5 * (logs[:n_pairs] + logs[n_pairs:])
        return float(pm.mean()), pm

    y_root = optimize.brentq(lambda y: eval_growth(y)[0], -1.5, 6.0, xtol=1e-6)
    r0, pm0 = eval_growth(y_root)
    r1, _ = eval_growth(y_root + 1e-4)
    slope = (r1 - r0) / 1e-4
    se = float(pm0.std(ddof=1) / math.sqrt(n_pairs) / max(abs(slope), 1e-8))
    return float(y_root), se
