#!/usr/bin/env python3
"""Helper library for `casestudies/devtimes/analysis.py`.

Contains:
  - Pure standard-library XML/ZIP readers for `.xlsx` and `.ods` files (`python3 -I` safe).
  - Statistical estimators for individual stage-duration spread (`CV_indiv`):
      * Sokal-Rohlf bias-corrected sample CV and Sheppard interval-censoring debiasing;
      * Right-censored Weibull MLE (`mean_mle`, `cv_mle`);
      * Cross-sectional destructive prevalence Gamma-CDF binomial MLE (`F_EIP(t) * p_max`);
      * Logistic/Gamma degree-day container transition fit for field enclosures.
  - Exact companion renewal matrix builder and CRN stochastic-loss simulator for
    arbitrary discrete empirical stay distributions `p(l)`, plus haversine distance
    and vapour-pressure-deficit utilities for the Weather & Environment section.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# 1. Safe XML / ZIP spreadsheet readers (no third-party xlsx/ods packages)
# ---------------------------------------------------------------------------

def _col_to_idx(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheet(path: Path, sheet_index: int = 0) -> list[list[str]]:
    """Read worksheet `sheet_index` (0-based) of an `.xlsx` archive into string rows."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as z:
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sroot.findall(".//m:si", ns):
                sst.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = wb.findall(".//m:sheet", ns)
        rels = {}
        if "xl/_rels/workbook.xml.rels" in z.namelist():
            rroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            for rel in rroot:
                rels[rel.attrib.get("Id", "")] = rel.attrib.get("Target", "")
        sh = sheets[sheet_index]
        rid = sh.attrib.get(rel_ns, "")
        target = rels.get(rid, f"worksheets/sheet{sheet_index + 1}.xml").lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sxml = ET.fromstring(z.read(target))
        out: list[list[str]] = []
        for r in sxml.findall(".//m:row", ns):
            row_map: dict[int, str] = {}
            max_c = -1
            for c in r.findall("m:c", ns):
                ref = c.attrib.get("r", "")
                col_str = "".join(ch for ch in ref if ch.isalpha())
                cidx = _col_to_idx(col_str) if col_str else (max_c + 1)
                max_c = max(max_c, cidx)
                t = c.attrib.get("t", "")
                v = c.find("m:v", ns)
                is_el = c.find("m:is", ns)
                if t == "s" and v is not None and v.text is not None:
                    val = sst[int(v.text)]
                elif is_el is not None:
                    val = "".join(x.text or "" for x in is_el.findall(".//m:t", ns))
                elif v is not None and v.text is not None:
                    val = v.text
                else:
                    val = ""
                row_map[cidx] = val.strip()
            if max_c >= 0:
                out.append([row_map.get(i, "") for i in range(max_c + 1)])
        return out


def read_ods_tables(path: Path) -> dict[str, list[list[str]]]:
    """Read all named sheets from an OpenDocument `.ods` archive."""
    out: dict[str, list[list[str]]] = {}
    with zipfile.ZipFile(path) as z:
        cxml = ET.fromstring(z.read("content.xml"))
        for tbl in cxml.iter():
            if not tbl.tag.endswith("}table"):
                continue
            tname = next((v for k, v in tbl.attrib.items() if k.endswith("}name")), "")
            rows_out: list[list[str]] = []
            for r in tbl:
                if not r.tag.endswith("}table-row"):
                    continue
                vals: list[str] = []
                for cell in r:
                    if not cell.tag.endswith("}table-cell"):
                        continue
                    rep = int(next((v for k, v in cell.attrib.items() if k.endswith("number-columns-repeated")), "1"))
                    txt = "".join(p.text or "" for p in cell.iter() if p.tag.endswith("}p")).strip()
                    vals.extend([txt] * min(rep, 32))
                while vals and vals[-1] == "":
                    vals.pop()
                if vals:
                    rows_out.append(vals)
            out[tname] = rows_out
    return out


# ---------------------------------------------------------------------------
# 2. Estimators for individual stage-duration spread (Section 4 of protocol)
# ---------------------------------------------------------------------------

def uncensored_stats(values: list[float] | np.ndarray, dt_obs: float | None = None) -> dict[str, float]:
    """Compute sample mean, SD, plug-in CV, Sokal-Rohlf bias-corrected CV_indiv,
    asymptotic SE of CV, and optional Sheppard-debiased CV."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    n = int(x.size)
    if n < 2:
        return {
            "n": n,
            "mean": float(x[0]) if n == 1 else math.nan,
            "sd": math.nan,
            "cv_raw": math.nan,
            "cv_indiv": math.nan,
            "cv_se": math.nan,
            "cv_sheppard": math.nan,
            "skewness": math.nan,
            "median": float(x[0]) if n == 1 else math.nan,
            "iqr": 0.0 if n == 1 else math.nan,
        }
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    cv_raw = sd / mean if mean > 0 else math.nan
    cv_cor = cv_raw * (1.0 + 1.0 / (4.0 * n))
    cv_se = cv_cor * math.sqrt((1.0 + 2.0 * cv_cor * cv_cor) / (2.0 * max(n - 1, 1)))
    if dt_obs is not None and dt_obs > 0 and (dt_obs / mean) > 0.05:
        v_deb = max(sd * sd - (dt_obs * dt_obs) / 12.0, 0.0)
        cv_shep = (math.sqrt(v_deb) / mean) * (1.0 + 1.0 / (4.0 * n))
    else:
        cv_shep = cv_cor
    skew = float(stats.skew(x, bias=False)) if n >= 3 and sd > 0 else 0.0
    q25, q50, q75 = np.percentile(x, [25.0, 50.0, 75.0])
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv_raw": cv_raw,
        "cv_indiv": cv_cor,
        "cv_se": cv_se,
        "cv_sheppard": cv_shep,
        "skewness": skew,
        "median": float(q50),
        "iqr": float(q75 - q25),
    }


def weibull_censored_mle(t_obs: np.ndarray, event: np.ndarray) -> dict[str, float]:
    """Exact right-censored Weibull MLE (Section 4.1 Rule 2, validated in Simulation P2b).
    `event` is boolean True for observed completion/death, False for right-censored."""
    t = np.asarray(t_obs, dtype=float)
    ev = np.asarray(event, dtype=bool)
    ok = np.isfinite(t) & (t > 0.0)
    t, ev = t[ok], ev[ok]
    n_tot = int(t.size)
    d = int(ev.sum())
    if d < 3 or n_tot < 3:
        return {"n_total": n_tot, "n_events": d, "cens_frac": math.nan, "mean_mle": math.nan, "sd_mle": math.nan, "cv_mle": math.nan, "cv_se": math.nan, "k_shape": math.nan}

    def neg_prof_ll(log_k: float) -> float:
        k = math.exp(log_k)
        tk = np.power(t, k)
        lam_k = float(np.sum(tk) / d)
        if lam_k <= 0 or not math.isfinite(lam_k):
            return 1e100
        ll = d * log_k - d * math.log(lam_k) + (k - 1.0) * float(np.sum(np.log(t[ev]))) - d
        return -ll

    res = optimize.minimize_scalar(neg_prof_ll, bounds=(-2.5, 4.5), method="bounded")
    log_k = float(res.x)
    k_hat = math.exp(log_k)
    lam_hat = float(np.sum(np.power(t, k_hat)) / d) ** (1.0 / k_hat)
    g1 = special.gamma(1.0 + 1.0 / k_hat)
    g2 = special.gamma(1.0 + 2.0 / k_hat)
    mean_hat = lam_hat * g1
    var_hat = max(lam_hat * lam_hat * (g2 - g1 * g1), 0.0)
    sd_hat = math.sqrt(var_hat)
    cv_hat = sd_hat / mean_hat if mean_hat > 0 else math.nan

    # Numerical observed Fisher curvature in log_k for delta-method SE of CV(k)
    h = 1e-4
    f_p = neg_prof_ll(log_k + h)
    f_0 = neg_prof_ll(log_k)
    f_m = neg_prof_ll(log_k - h)
    curv = max((f_p - 2.0 * f_0 + f_m) / (h * h), 1e-8)
    se_log_k = 1.0 / math.sqrt(curv)

    def cv_of_logk(lk: float) -> float:
        kk = math.exp(lk)
        gg1 = special.gamma(1.0 + 1.0 / kk)
        gg2 = special.gamma(1.0 + 2.0 / kk)
        return math.sqrt(max(gg2 / (gg1 * gg1) - 1.0, 0.0))

    dcv = abs(cv_of_logk(log_k + h) - cv_of_logk(log_k - h)) / (2.0 * h)
    cv_se = float(dcv * se_log_k)
    return {
        "n_total": n_tot,
        "n_events": d,
        "cens_frac": 1.0 - (d / n_tot),
        "mean_mle": float(mean_hat),
        "sd_mle": float(sd_hat),
        "cv_mle": float(cv_hat),
        "cv_se": cv_se,
        "k_shape": float(k_hat),
    }


def gamma_cdf_cross_sectional_mle(days: np.ndarray, pos: np.ndarray, tot: np.ndarray) -> dict[str, float]:
    """Binomial MLE of F(t) = p_max * GammaCDF(t; shape=k, scale=mu/k) on destructive
    cross-sectional prevalence cohorts (Section 4.1 Rule 3, e.g. Plasmodium EIP)."""
    days = np.asarray(days, dtype=float)
    pos = np.asarray(pos, dtype=float)
    tot = np.asarray(tot, dtype=float)
    ok = (tot > 0) & (days > 0)
    days, pos, tot = days[ok], pos[ok], tot[ok]
    if days.size < 3 or float(pos.sum()) < 5:
        return {"ok": False}

    obs_p = pos / tot
    pmax_init = float(np.clip(np.max(obs_p), 0.10, 0.999))
    half_idx = np.where(obs_p >= 0.5 * pmax_init)[0]
    mu_init = float(days[half_idx[0]]) if half_idx.size else float(np.median(days))

    def nll(theta: np.ndarray) -> float:
        log_k, log_mu, logit_p = float(theta[0]), float(theta[1]), float(theta[2])
        if not (-2.0 <= log_k <= 6.5 and -1.0 <= log_mu <= 6.0 and -6.0 <= logit_p <= 8.0):
            return 1e50
        k = math.exp(log_k)
        mu = math.exp(log_mu)
        pmax = 1.0 / (1.0 + math.exp(-logit_p))
        cdf = special.gammainc(k, k * days / mu)
        prob = np.clip(pmax * cdf, 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(pos * np.log(prob) + (tot - pos) * np.log1p(-prob)))
        return -ll

    best = None
    for lk0 in (1.5, 3.0, 4.2):
        x0 = np.array([lk0, math.log(max(mu_init, 1.0)), math.log(pmax_init / (1.0 - pmax_init))])
        r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    log_k, log_mu, logit_p = map(float, best.x)
    k_hat = math.exp(log_k)
    mu_hat = math.exp(log_mu)
    pmax_hat = 1.0 / (1.0 + math.exp(-logit_p))
    cv_hat = 1.0 / math.sqrt(k_hat)
    sd_hat = mu_hat * cv_hat

    # Check bracketing: observed prevalence starts < 0.25*pmax and reaches > 0.75*pmax
    bracketed = bool((np.min(obs_p) <= 0.25 * pmax_hat) and (np.max(obs_p) >= 0.75 * pmax_hat) and (days.size >= 4))

    # Profile curvature in log_k for SE(CV)
    h = 1e-3
    f0 = nll(best.x)
    fp = nll(best.x + np.array([h, 0.0, 0.0]))
    fm = nll(best.x - np.array([h, 0.0, 0.0]))
    curv = max((fp - 2.0 * f0 + fm) / (h * h), 1e-6)
    se_log_k = 1.0 / math.sqrt(curv)
    cv_se = 0.5 * cv_hat * se_log_k

    return {
        "ok": True,
        "bracketed": bracketed,
        "n_dissected": int(tot.sum()),
        "n_positive": int(pos.sum()),
        "n_days": int(days.size),
        "mean_eip": mu_hat,
        "sd_eip": sd_hat,
        "cv_indiv": cv_hat,
        "cv_se": cv_se,
        "k_gamma": k_hat,
        "p_max": pmax_hat,
    }


def weibull_survival_cross_sectional_mle(days: np.ndarray, surv: np.ndarray, tot: np.ndarray) -> dict[str, float]:
    """Binomial MLE of S(t) = s0 * exp(-(t / lam)^k) on cross-sectional container/cohort
    survival retrievals (Section 4.1 Rule 3, e.g. D6 flat tick containers, D9 cercarial survival)."""
    days = np.asarray(days, dtype=float)
    surv = np.asarray(surv, dtype=float)
    tot = np.asarray(tot, dtype=float)
    ok = (tot > 0) & (days > 0) & np.isfinite(surv)
    days, surv, tot = days[ok], np.clip(surv[ok], 0.0, tot[ok]), tot[ok]
    u_days = np.unique(np.round(days, 3))
    if u_days.size < 4 or float(tot.sum()) < 30:
        return {"ok": False}

    obs_s = surv / tot
    s0_init = float(np.clip(np.max(obs_s), 0.60, 0.999))
    half_idx = np.where(obs_s <= 0.5 * s0_init)[0]
    lam_init = float(days[half_idx[0]]) if half_idx.size else float(np.median(days))

    def nll(theta: np.ndarray) -> float:
        log_k, log_lam, logit_s0 = float(theta[0]), float(theta[1]), float(theta[2])
        if not (-1.5 <= log_k <= 4.5 and -2.5 <= log_lam <= 7.0 and -2.0 <= logit_s0 <= 9.0):
            return 1e50
        k = math.exp(log_k)
        lam = math.exp(log_lam)
        s0 = 1.0 / (1.0 + math.exp(-logit_s0))
        prob = np.clip(s0 * np.exp(-np.power(days / lam, k)), 1e-12, 1.0 - 1e-12)
        return -float(np.sum(surv * np.log(prob) + (tot - surv) * np.log1p(-prob)))

    best = None
    for lk0 in (0.0, 0.8, 1.8):
        x0 = np.array([lk0, math.log(max(lam_init, 0.05)), math.log(s0_init / (1.0 - s0_init))])
        r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 1800, "xatol": 1e-6, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    log_k, log_lam, logit_s0 = map(float, best.x)
    k_hat = math.exp(log_k)
    lam_hat = math.exp(log_lam)
    s0_hat = 1.0 / (1.0 + math.exp(-logit_s0))
    g1 = special.gamma(1.0 + 1.0 / k_hat)
    g2 = special.gamma(1.0 + 2.0 / k_hat)
    mean_hat = lam_hat * g1
    sd_hat = lam_hat * math.sqrt(max(g2 - g1 * g1, 0.0))
    cv_hat = sd_hat / mean_hat if mean_hat > 0 else math.nan
    bracketed = bool((np.max(obs_s) >= 0.70 * s0_hat) and (np.min(obs_s) <= 0.30 * s0_hat) and (u_days.size >= 4))

    h = 1e-3
    f0 = nll(best.x)
    fp = nll(best.x + np.array([h, 0.0, 0.0]))
    fm = nll(best.x - np.array([h, 0.0, 0.0]))
    curv = max((fp - 2.0 * f0 + fm) / (h * h), 1e-6)
    se_lk = 1.0 / math.sqrt(curv)

    def _cv_w(lk: float) -> float:
        kk = math.exp(lk)
        gg1, gg2 = special.gamma(1.0 + 1.0 / kk), special.gamma(1.0 + 2.0 / kk)
        return math.sqrt(max(gg2 / (gg1 * gg1) - 1.0, 0.0))

    cv_se = float(abs(_cv_w(log_k + h) - _cv_w(log_k - h)) / (2.0 * h) * se_lk)
    return {
        "ok": True,
        "bracketed": bracketed,
        "n_deployed": int(round(float(tot.sum()))),
        "n_survived": int(round(float(surv.sum()))),
        "n_days": int(u_days.size),
        "mean_surv": float(mean_hat),
        "sd_surv": float(sd_hat),
        "cv_indiv": float(cv_hat),
        "cv_se": cv_se,
        "k_weibull": float(k_hat),
        "s0": float(s0_hat),
    }


# ---------------------------------------------------------------------------
# 3. Exact companion renewal matrix & unexpanded simulator for empirical stays
# ---------------------------------------------------------------------------

def discretize_empirical_stay(durations: np.ndarray, target_mean_steps: float, max_age: int = 400) -> tuple[np.ndarray, float]:
    """Scale positive continuous individual durations `durations` so their discrete
    positive integer stay L >= 1 has exact mean close to `target_mean_steps`, returning
    probability vector `p[0..Lmax-1]` on `{1, ..., Lmax}` and discrete CV."""
    x = np.asarray(durations, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    scale = target_mean_steps / float(np.mean(x))
    scaled = np.clip(np.rint(x * scale), 1.0, float(max_age)).astype(int)
    lmax = int(scaled.max())
    counts = np.bincount(scaled, minlength=lmax + 1)[1:]
    p = counts.astype(float) / float(scaled.size)
    ell = np.arange(1, lmax + 1, dtype=float)
    mu = float(p @ ell)
    var = float(p @ ((ell - mu) ** 2))
    cv = math.sqrt(max(var, 0.0)) / mu
    return p, cv


def empirical_D2(f_fixed_steps: int, p_long: np.ndarray, sigma: float, phi: float) -> tuple[float, float, float]:
    """Exact primitive second-order loss D2 (Statement C2b) for a cycle with `f_fixed_steps`
    fixed exposure steps and empirical steady-stay pmf `p_long` on `{1, ..., Lmax}`."""
    ell = np.arange(1, p_long.size + 1, dtype=float)
    mean_long = float(p_long @ ell)
    T = float(f_fixed_steps) + mean_long
    if phi == 0.0:
        return sigma * sigma / (2.0 * T), T, 0.0
    theta = phi / (1.0 - phi)
    rho_long = float(p_long @ np.power(phi, ell))
    R = (phi ** f_fixed_steps) * rho_long
    if abs(1.0 - R) < 1e-14:
        return math.nan, T, R
    d2 = sigma * sigma * ((1.0 + 2.0 * theta) / (2.0 * T) - R / (1.0 - R))
    return d2, T, R


def companion_age_matrix(f_fixed_steps: int, p_long: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the exact finite survival-progression Leslie/renewal matrix for `f_fixed_steps`
    fixed steps followed by arbitrary discrete stay law `p_long` on `{1, ..., Lmax}` at threshold.
    Returns `(A_rest, A_birth)` where the noisy birth transfer enters age-0 of exposure."""
    p = np.asarray(p_long, dtype=float)
    p = p / p.sum()
    lmax = int(p.size)
    # Tail survival S_a = P(L >= a) for a = 1..lmax
    tail = np.cumsum(p[::-1])[::-1]
    hazards = np.where(tail > 1e-15, p / np.maximum(tail, 1e-300), 1.0)
    hazards[-1] = 1.0
    n = f_fixed_steps + lmax
    rest = np.zeros((n, n), dtype=float)
    birth = np.zeros((n, n), dtype=float)
    # States 0 .. f_fixed_steps-1: fixed exposure progression; transfer f-1 -> f is constant 1,
    # and we place the noisy birth multiplier on completion of the long stage -> state 0
    # (by single-noisy-transfer invariance Proposition C / Referee 14 point 27, identical top exponent).
    for j in range(f_fixed_steps):
        rest[j + 1, j] = 1.0
    for a in range(lmax):
        idx = f_fixed_steps + a
        h = float(np.clip(hazards[a], 0.0, 1.0))
        birth[0, idx] = h
        if a + 1 < lmax and (1.0 - h) > 0.0:
            rest[idx + 1, idx] = 1.0 - h
    return rest, birth


def simulate_empirical_loss(
    f_fixed_steps: int,
    p_long: np.ndarray,
    sigma: float,
    phi: float,
    steps: int = 48000,
    burn: int = 3000,
    pairs: int = 16,
    seed: int = 20261114,
) -> dict[str, float]:
    """Unexpanded finite-noise loss `D +- SE` for an empirical discrete stay distribution `p_long`
    using antithetic common-random-number Newton root finding on the exact age-hazard recursion."""
    p = np.asarray(p_long, dtype=float)
    p = p / p.sum()
    # All-fixed check: if support is a single point, exact cohort ergodicity gives sigma^2 / 2
    if np.count_nonzero(p > 1e-14) == 1:
        return {"D": 0.5 * sigma * sigma, "se": 0.0, "exact_periodic": True}

    tail = np.cumsum(p[::-1])[::-1]
    haz = np.clip(np.where(tail > 1e-15, p / np.maximum(tail, 1e-300), 1.0), 0.0, 1.0)
    haz[-1] = 1.0
    cont = 1.0 - haz[:-1]
    lmax = int(p.size)

    rng = np.random.default_rng(seed)
    total_steps = burn + steps
    z0 = rng.standard_normal(pairs)
    eps = rng.standard_normal((total_steps, pairs))
    z0_all = np.concatenate([z0, -z0])
    eps_all = np.concatenate([eps, -eps], axis=1)
    reps = 2 * pairs
    h_mult = np.empty((total_steps, reps), dtype=float)
    z = z0_all.copy()
    c_innov = math.sqrt(max(1.0 - phi * phi, 0.0))
    half_s2 = 0.5 * sigma * sigma
    for t in range(total_steps):
        z = phi * z + c_innov * eps_all[t]
        h_mult[t] = np.exp(sigma * z - half_s2)

    def eval_ys(ys: np.ndarray) -> np.ndarray:
        ny = ys.size
        ey = np.exp(ys)[:, None]
        exp_q = np.full((f_fixed_steps, ny, reps), 1.0 / (f_fixed_steps + lmax))
        age_q = np.full((lmax, ny, reps), 1.0 / (f_fixed_steps + lmax))
        acc = np.zeros((ny, reps), dtype=float)
        haz_col = haz[:, None, None]
        cont_col = cont[:, None, None]
        for t in range(total_steps):
            leaving = np.sum(age_q * haz_col, axis=0)
            births = ey * h_mult[t][None, :] * leaving
            entering_long = exp_q[-1].copy()
            if f_fixed_steps > 1:
                exp_q[1:] = exp_q[:-1]
            exp_q[0] = births
            if lmax > 1:
                age_q[1:] = age_q[:-1] * cont_col
            age_q[0] = entering_long
            if (t & 15) == 15 or t == total_steps - 1:
                tot = exp_q.sum(axis=0) + age_q.sum(axis=0)
                exp_q /= tot
                age_q /= tot
                if t >= burn:
                    acc += np.log(tot)
            elif t == burn - 1:
                tot = exp_q.sum(axis=0) + age_q.sum(axis=0)
                exp_q /= tot
                age_q /= tot
        # Pairwise antithetic averages
        g_rep = acc / steps
        return 0.5 * (g_rep[:, :pairs] + g_rep[:, pairs:])

    cap = 0.5 * sigma * sigma
    d2_init, _, _ = empirical_D2(f_fixed_steps, p, sigma, phi)
    centre = float(np.clip(d2_init if math.isfinite(d2_init) else 0.5 * cap, 0.02 * cap, 0.95 * cap))
    delta = max(0.03, 0.20 * centre)
    for _ in range(4):
        ys = centre + delta * np.array([-1.0, 0.0, 1.0])
        gbar = eval_ys(ys)
        rmean = gbar.mean(axis=1)
        slope = (rmean[2] - rmean[0]) / (2.0 * delta)
        root = centre - rmean[1] / slope
        centre = float(np.clip(root, -0.5, 1.5 * cap))
        delta = max(0.008, min(0.05, 4.0 * float(gbar[1].std(ddof=1) / math.sqrt(pairs) / max(abs(slope), 1e-8))))
    dm = root - gbar[1] / slope
    return {
        "D": float(root),
        "se": float(dm.std(ddof=1) / math.sqrt(pairs)),
        "exact_periodic": False,
    }


# ---------------------------------------------------------------------------
# 4. Haversine distance & saturation vapour pressure utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, a)))


def vpd_kpa_from_tmax_tmin_vp(tmax_c: float, tmin_c: float, vp_pa: float) -> float:
    """Tetens saturation vapour pressure minus ambient `vp_pa` (Pa), returned in kPa."""
    es_max = 0.61078 * math.exp(17.269 * tmax_c / (tmax_c + 237.3))
    es_min = 0.61078 * math.exp(17.269 * tmin_c / (tmin_c + 237.3))
    es = 0.5 * (es_max + es_min)
    return max(0.0, es - vp_pa / 1000.0)
