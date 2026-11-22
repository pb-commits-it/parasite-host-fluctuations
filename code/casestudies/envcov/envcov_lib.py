"""Data parsing, abiotic covariate construction, and thermal tick duration models for envcov.

All formulas follow `casestudies/envcov/protocol.md` (Section 4).
Standard library and numpy only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import datetime
import glob
import json
import math
from pathlib import Path
import re

import numpy as np

R14_INCLUDED = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]
R14_ALL13 = "BLAN DELA HARV KONZ MLBS ORNL OSBS SCBI SERC STEI TALL TREE UKFS".split()
COV_KEYS = ["C1_dd6", "C2_hotdry", "C3_wmin", "C4_snow", "C5_gsprcp", "C6_satdef"]
COV_LABELS = {
    "C1_dd6": "DD > 6 C (Oct-Sep, C*d)",
    "C2_hotdry": "Hot dry days (May-Sep, Tmax>25 & prcp=0)",
    "C3_wmin": "Winter min Tmin (Nov-Mar, C)",
    "C4_snow": "Winter snow days (Nov-Mar, SWE>0)",
    "C5_gsprcp": "Growing-season precip (Apr-Sep, mm)",
    "C6_satdef": "Questing saturation deficit (May-Sep, kPa)",
}


def tetens_es_kpa(temp_c: np.ndarray | float) -> np.ndarray | float:
    """Saturation vapor pressure in kPa by Tetens formula [recalled meteorological standard]."""
    t = np.asarray(temp_c, dtype=float)
    return 0.61078 * np.exp(17.269 * t / (t + 237.3))


def read_daymet_daily(path: Path) -> dict[str, np.ndarray]:
    """Read a Daymet V4 single-pixel CSV into aligned daily numpy arrays."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("year,yday,"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"No Daymet CSV header found in {path}")

    reader = csv.DictReader(lines[start_idx:])
    yr, yd, mo, dy = [], [], [], []
    tmax, tmin, prcp, vp, swe = [], [], [], [], []
    for r in reader:
        y = int(r["year"])
        doy = int(r["yday"])
        dt = datetime.date(y, 1, 1) + datetime.timedelta(days=doy - 1)
        yr.append(y)
        yd.append(doy)
        mo.append(dt.month)
        dy.append(dt.day)
        tmax.append(float(r["tmax (deg c)"]))
        tmin.append(float(r["tmin (deg c)"]))
        prcp.append(float(r["prcp (mm/day)"]))
        vp.append(float(r["vp (Pa)"]))
        swe.append(float(r["swe (kg/m^2)"]))

    tmax_a = np.array(tmax, dtype=float)
    tmin_a = np.array(tmin, dtype=float)
    tmean_a = 0.5 * (tmax_a + tmin_a)
    vp_kpa = np.array(vp, dtype=float) / 1000.0
    es_kpa = tetens_es_kpa(tmean_a)
    satdef_kpa = np.maximum(0.0, es_kpa - vp_kpa)
    rh_pct = np.clip(100.0 * vp_kpa / np.maximum(es_kpa, 1e-9), 0.0, 100.0)

    return {
        "year": np.array(yr, dtype=int),
        "yday": np.array(yd, dtype=int),
        "month": np.array(mo, dtype=int),
        "day": np.array(dy, dtype=int),
        "tmax": tmax_a,
        "tmin": tmin_a,
        "tmean": tmean_a,
        "prcp": np.array(prcp, dtype=float),
        "vp_pa": np.array(vp, dtype=float),
        "swe": np.array(swe, dtype=float),
        "satdef_kpa": satdef_kpa,
        "rh_daymet": rh_pct,
    }


def read_power_daily(path: Path) -> dict[str, np.ndarray | float]:
    """Read a NASA POWER daily JSON file into aligned daily numpy arrays + grid coordinates."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    coords = (raw.get("geometry") or {}).get("coordinates") or [np.nan, np.nan, np.nan]
    params = raw["properties"]["parameter"]
    dates = sorted(params["RH2M"].keys())
    yr, mo, dy = [], [], []
    out_cols: dict[str, list[float]] = {k: [] for k in params}
    for dstr in dates:
        yr.append(int(dstr[:4]))
        mo.append(int(dstr[4:6]))
        dy.append(int(dstr[6:8]))
        for k, series in params.items():
            v = float(series[dstr])
            out_cols[k].append(np.nan if v <= -900.0 else v)
    res: dict[str, np.ndarray | float] = {
        "year": np.array(yr, dtype=int),
        "month": np.array(mo, dtype=int),
        "day": np.array(dy, dtype=int),
        "grid_lon": float(coords[0]),
        "grid_lat": float(coords[1]),
        "grid_elev_m": float(coords[2]) if len(coords) > 2 else float("nan"),
    }
    for k, vals in out_cols.items():
        res[k] = np.array(vals, dtype=float)
    return res


def build_annual_covariates(
    dm: dict[str, np.ndarray],
    pw: dict[str, np.ndarray] | None = None,
    years: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Construct annual/seasonal tick covariates aligned to nymphal year t."""
    yr = dm["year"]
    mo = dm["month"]
    tmax = dm["tmax"]
    tmin = dm["tmin"]
    tmean = dm["tmean"]
    prcp = dm["prcp"]
    swe = dm["swe"]
    satdef = dm["satdef_kpa"]
    rh_dm = dm["rh_daymet"]

    min_yr = int(yr.min()) + 1
    max_yr = int(yr.max())
    target_years = years if years is not None else list(range(min_yr, max_yr + 1))
    out: dict[int, dict[str, float]] = {}

    for t in target_years:
        oct_sep = ((yr == t - 1) & (mo >= 10)) | ((yr == t) & (mo <= 9))
        nov_mar = ((yr == t - 1) & (mo >= 11)) | ((yr == t) & (mo <= 3))
        may_sep = (yr == t) & (mo >= 5) & (mo <= 9)
        may_jul = (yr == t) & (mo >= 5) & (mo <= 7)
        prev_aug_sep = (yr == t - 1) & (mo >= 8) & (mo <= 9)
        apr_sep = (yr == t) & (mo >= 4) & (mo <= 9)
        feb_apr = (yr == t) & (mo >= 2) & (mo <= 4)
        cal_yr = yr == t

        row = {
            "C1_dd6": float(np.sum(np.maximum(0.0, tmean[oct_sep] - 6.0))),
            "C2_hotdry": float(np.sum((tmax[may_sep] > 25.0) & (prcp[may_sep] == 0.0))),
            "C3_wmin": float(np.min(tmin[nov_mar])),
            "C4_snow": float(np.sum(swe[nov_mar] > 0.0)),
            "C5_gsprcp": float(np.sum(prcp[apr_sep])),
            "C6_satdef": float(np.mean(satdef[may_sep])),
            # Sensitivity & Cary cross-validation definitions
            "dd0_oct_sep": float(np.sum(np.maximum(0.0, tmean[oct_sep]))),
            "dd10_oct_sep": float(np.sum(np.maximum(0.0, tmean[oct_sep] - 10.0))),
            "mat_cal": float(np.mean(tmean[cal_yr])),
            "hotdry_nymph_may_jul": float(np.sum((tmax[may_jul] > 25.0) & (prcp[may_jul] == 0.0))),
            "hotdry_larv_prev_aug_sep": float(np.sum((tmax[prev_aug_sep] > 25.0) & (prcp[prev_aug_sep] == 0.0))),
            "cold_winter_frac": float(np.mean(tmin[nov_mar] < -10.0)),
            "swe_winter_mean": float(np.mean(swe[nov_mar])),
            "feb_apr_warm_dry_frac": float(np.mean((tmax[feb_apr] > 15.0) & (rh_dm[feb_apr] < 50.0))),
        }

        if pw is not None:
            pyr, pmo = pw["year"], pw["month"]
            p_jun = (pyr == t) & (pmo == 6)
            p_gs = (pyr == t) & (pmo >= 5) & (pmo <= 9)
            p_win = ((pyr == t - 1) & (pmo >= 11)) | ((pyr == t) & (pmo <= 3))
            row["P_rh_jun_sub82"] = float(np.sum(pw["RH2M"][p_jun] < 82.0))
            row["P_rh_gs"] = float(np.nanmean(pw["RH2M"][p_gs]))
            row["P_gwettop_gs"] = float(np.nanmean(pw["GWETTOP"][p_gs]))
            row["P_ts_wmin"] = float(np.nanmin(pw["TS"][p_win]))

        out[t] = row
    return out


def compute_pca1(mat: np.ndarray, orient_col: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """First principal component of standardized columns, oriented positively with orient_col."""
    x = np.asarray(mat, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, ddof=1, keepdims=True), 1e-15)
    z = (x - mu) / sd
    _u, svals, vt = np.linalg.svd(z, full_matrices=False)
    loadings = vt[0].copy()
    scores = z @ loadings
    if np.corrcoef(scores, x[:, orient_col])[0, 1] < 0:
        loadings = -loadings
        scores = -scores
    scores = (scores - scores.mean()) / max(float(scores.std(ddof=1)), 1e-15)
    var_explained = float((svals[0] ** 2) / np.sum(svals ** 2))
    return scores, loadings, var_explained


def maturation_rates(tmean: np.ndarray, model: str = "wallace2019") -> tuple[np.ndarray, np.ndarray]:
    """Daily maturation rates (day^-1) for engorged larva->nymph and engorged nymph->adult."""
    t = np.asarray(tmean, dtype=float)
    if model == "wallace2019":
        # Wallace et al. (2019, PMC6766261, Table 1, read in full), fitted to Ogden et al. (2004)
        active = (t >= 15.0).astype(float)
        m_ln = 0.04001 * np.exp(-(((t - 26.68) / 9.533) ** 2)) * active
        m_na = 0.03173 * np.exp(-(((t - 23.85) / 9.042) ** 2)) * active
        return m_ln, m_na
    if model == "ogden2004_recalled":
        # Ogden et al. (2004, J. Med. Entomol. 41:622, abstract read; power-law constants recalled)
        tp = np.maximum(t, 0.0)
        m_ln = np.where(t > 8.5, (tp ** 2.275) / 101181.0, 0.0)
        m_na = np.where(t > 8.0, (tp ** 1.210) / 1596.0, 0.0)
        return m_ln, m_na
    raise ValueError(f"Unknown maturation model: {model}")


def _days_to_threshold(rate_arr: np.ndarray, start_idx: int, target: float = 1.0) -> int | None:
    """Minimal integer days L >= 1 from start_idx until cumulative rate >= target, or None if censored."""
    rem = rate_arr[start_idx:]
    csum = np.cumsum(rem)
    hits = np.flatnonzero(csum >= target)
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def compute_tick_stage_durations(
    dm: dict[str, np.ndarray],
    cohort_years: list[int],
    quest_rate: float = 0.02,
    model: str = "wallace2019",
) -> dict:
    """Compute site- and year-specific larva->nymph and nymph->adult durations and spreads."""
    yr = dm["year"]
    yd = dm["yday"]
    mo = dm["month"]
    dy = dm["day"]
    tmean = dm["tmean"]

    m_ln, m_na = maturation_rates(tmean, model=model)
    q_haz = quest_rate * (tmean >= 10.0).astype(float)  # Allen (2025, PMC12040446, Table 1)

    idx_by_yd = {(int(y), int(d)): i for i, (y, d) in enumerate(zip(yr, yd))}
    mar1_idx = {int(y): i for i, (y, m, d) in enumerate(zip(yr, mo, dy)) if m == 3 and d == 1}

    # Seasonal larval drop-off dates: every 5 days from DOY 196 (15 Jul) to DOY 256 (13 Sep)
    drop_doys = np.arange(196, 257, 5, dtype=int)
    drop_weights = np.exp(-0.5 * (((drop_doys - 227.0) / 20.0) ** 2))
    drop_weights /= drop_weights.sum()

    premoult_ln_aug15 = {}
    premoult_na_jun1 = {}
    phen_stay_year_mean = {}
    phen_stay_samples = []
    phen_stay_sample_weights = []
    censored_cohorts = 0

    for y in cohort_years:
        i_aug15 = idx_by_yd.get((y, 227))
        i_jun1 = idx_by_yd.get((y, 152))
        if i_aug15 is not None:
            d_ln = _days_to_threshold(m_ln, i_aug15, 1.0)
            if d_ln is not None:
                premoult_ln_aug15[y] = d_ln
        if i_jun1 is not None:
            d_na = _days_to_threshold(m_na, i_jun1, 1.0)
            if d_na is not None:
                premoult_na_jun1[y] = d_na

        # Full phenological larva-feed to nymph-feed stay across seasonal drop-off window
        yr_stays, yr_wts = [], []
        i_mar1_next = mar1_idx.get(y + 1)
        if i_mar1_next is None:
            continue
        for doy, wt in zip(drop_doys, drop_weights):
            i0 = idx_by_yd.get((y, int(doy)))
            if i0 is None:
                continue
            d_moult = _days_to_threshold(m_ln, i0, 1.0)
            if d_moult is None:
                censored_cohorts += 1
                continue
            i_moult = i0 + d_moult
            # Post-solstice engorged larvae completing moult before spring wait in overwinter quiescence until 1 Mar
            i_quest = max(i_moult, i_mar1_next)
            d_quest = _days_to_threshold(q_haz, i_quest, 1.0)
            if d_quest is None:
                censored_cohorts += 1
                continue
            total_stay = (i_quest - i0) + d_quest
            yr_stays.append(float(total_stay))
            yr_wts.append(float(wt))
            phen_stay_samples.append(float(total_stay))
            phen_stay_sample_weights.append(float(wt))

        if yr_wts and sum(yr_wts) > 0.9:
            w_norm = np.array(yr_wts) / sum(yr_wts)
            phen_stay_year_mean[y] = float(np.sum(np.array(yr_stays) * w_norm))

    ln_vals = np.array(list(premoult_ln_aug15.values()), dtype=float)
    na_vals = np.array(list(premoult_na_jun1.values()), dtype=float)
    yr_means = np.array(list(phen_stay_year_mean.values()), dtype=float)
    s_all = np.array(phen_stay_samples, dtype=float)
    w_all = np.array(phen_stay_sample_weights, dtype=float)
    w_all /= np.maximum(w_all.sum(), 1e-15)

    mean_phen = float(np.sum(s_all * w_all))
    var_total = float(np.sum(w_all * ((s_all - mean_phen) ** 2)))
    sd_total = math.sqrt(max(var_total, 0.0))
    cv_total = sd_total / mean_phen if mean_phen > 0 else float("nan")
    sd_interannual = float(yr_means.std(ddof=1)) if len(yr_means) >= 2 else float("nan")
    cv_interannual = sd_interannual / float(yr_means.mean()) if len(yr_means) >= 2 else float("nan")

    return {
        "model": model,
        "quest_rate": quest_rate,
        "n_years_ln": int(len(ln_vals)),
        "premoult_ln_mean_d": float(ln_vals.mean()),
        "premoult_ln_sd_d": float(ln_vals.std(ddof=1)) if len(ln_vals) >= 2 else 0.0,
        "premoult_ln_cv": float(ln_vals.std(ddof=1) / ln_vals.mean()) if len(ln_vals) >= 2 else 0.0,
        "premoult_na_mean_d": float(na_vals.mean()),
        "premoult_na_sd_d": float(na_vals.std(ddof=1)) if len(na_vals) >= 2 else 0.0,
        "premoult_na_cv": float(na_vals.std(ddof=1) / na_vals.mean()) if len(na_vals) >= 2 else 0.0,
        "phen_ln_mean_d": mean_phen,
        "phen_ln_interannual_sd_d": sd_interannual,
        "phen_ln_interannual_cv": cv_interannual,
        "phen_ln_total_sd_d": sd_total,
        "phen_ln_total_cv": cv_total,
        "phen_ln_by_year": phen_stay_year_mean,
        "premoult_ln_by_year": premoult_ln_aug15,
        "censored_subcohorts": censored_cohorts,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    r_earth = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def read_daymet_meta(path: Path) -> dict[str, float | int | str]:
    """Extract tile, LCC coordinates, and pixel elevation from a Daymet V4 CSV header."""
    lines = path.read_text(encoding="utf-8").splitlines()[:10]
    meta: dict[str, float | int | str] = {}
    for ln in lines:
        if ln.startswith("Latitude:"):
            parts = ln.replace("Latitude:", "").replace("Longitude:", "").split()
            meta["lat"] = float(parts[0])
            meta["lon"] = float(parts[1])
        elif ln.startswith("X & Y"):
            parts = ln.split(":")[1].split()
            meta["lcc_x_m"] = float(parts[0])
            meta["lcc_y_m"] = float(parts[1])
            # Distance to nearest 1000-m Daymet LCC pixel center (centers at k*1000 + 250 / 500; bounded <= 0.707 km)
        elif ln.startswith("Tile:"):
            meta["tile"] = int(ln.split(":")[1].strip())
        elif ln.startswith("Elevation:"):
            meta["elev_m"] = float(ln.split(":")[1].split()[0])
    return meta


def extract_neon_habitat_summary(neon_raw: Path) -> dict[str, dict[str, float]]:
    """Compute static tick-plot forest fraction, elevation, and spatial spread per R14 site."""
    forest_classes = {"deciduousForest", "mixedForest", "evergreenForest", "woodyWetlands"}
    out = {}
    name_re = re.compile(r"^NEON\.D\d\d\.([A-Z]{4})\.DP1\.10093\.001\.tck_fielddata\.(\d{4}-\d{2})\.basic\.(\d{8}T\d{6}Z)\.csv$")
    mam_re = re.compile(r"^NEON\.D\d\d\.([A-Z]{4})\.DP1\.10072\.001\.mam_pertrapnight\.(\d{4}-\d{2})\.basic\.(\d{8}T\d{6}Z)\.csv$")
    for site in R14_INCLUDED:
        d = neon_raw / "DP1.10093.001" / site
        chosen = {}
        for p in sorted(d.glob("*.csv")):
            m = name_re.match(p.name)
            if m and (m.group(2) not in chosen or m.group(3) > chosen[m.group(2)][0]):
                chosen[m.group(2)] = (m.group(3), p)
        plot_info = {}
        plot_ll = {}
        drag_forest = Counter()
        for _stamp, p in chosen.values():
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    pid = (r.get("plotID") or "").strip()
                    nc = (r.get("nlcdClass") or "").strip()
                    el = (r.get("elevation") or "").strip()
                    la = (r.get("decimalLatitude") or "").strip()
                    lo = (r.get("decimalLongitude") or "").strip()
                    if pid and el and nc:
                        plot_info.setdefault(pid, (nc, float(el)))
                        drag_forest["forest" if nc in forest_classes else "open"] += 1
                    if pid and la and lo and pid not in plot_ll:
                        plot_ll[pid] = (float(la), float(lo))
        # Also compute mammal-plot centroid to check tick-vs-mammal grid separation
        dmam = neon_raw / "DP1.10072.001" / site
        mam_ll = {}
        for p in sorted(dmam.glob("*.csv")):
            if not mam_re.match(p.name):
                continue
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    pid = (r.get("plotID") or "").strip()
                    if not pid or pid in mam_ll:
                        continue
                    la = (r.get("decimalLatitude") or "").strip()
                    lo = (r.get("decimalLongitude") or "").strip()
                    if la and lo:
                        mam_ll[pid] = (float(la), float(lo))
        plots = list(plot_info.values())
        forest_plot_frac = sum(1 for nc, _ in plots if nc in forest_classes) / max(len(plots), 1)
        forest_drag_frac = drag_forest["forest"] / max(sum(drag_forest.values()), 1)
        mean_elev = sum(el for _, el in plots) / max(len(plots), 1)
        t_lat = sum(v[0] for v in plot_ll.values()) / max(len(plot_ll), 1)
        t_lon = sum(v[1] for v in plot_ll.values()) / max(len(plot_ll), 1)
        max_rad_km = max((haversine_km(t_lat, t_lon, la, lo) for la, lo in plot_ll.values()), default=0.0)
        m_lat = sum(v[0] for v in mam_ll.values()) / max(len(mam_ll), 1) if mam_ll else t_lat
        m_lon = sum(v[1] for v in mam_ll.values()) / max(len(mam_ll), 1) if mam_ll else t_lon
        tick_mam_km = haversine_km(t_lat, t_lon, m_lat, m_lon)
        out[site] = {
            "n_tick_plots": len(plots),
            "forest_plot_frac": float(forest_plot_frac),
            "forest_drag_frac": float(forest_drag_frac),
            "elevation_m": float(mean_elev),
            "tick_lat": float(t_lat),
            "tick_lon": float(t_lon),
            "max_tick_plot_radius_km": float(max_rad_km),
            "tick_to_mammal_centroid_km": float(tick_mam_km),
        }
    return out
