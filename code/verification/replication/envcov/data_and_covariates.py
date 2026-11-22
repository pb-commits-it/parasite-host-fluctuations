"""Independent raw data parsers, abiotic covariate builders, and thermal stage-duration solvers for envcov."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

NEON_6 = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]
NEON_13 = ["BLAN", "DELA", "HARV", "KONZ", "MLBS", "ORNL", "OSBS", "SCBI", "SERC", "STEI", "TALL", "TREE", "UKFS"]
PRIMARY_COVS = ["C1_dd6", "C2_hotdry", "C3_wmin", "C4_snow", "C5_gsprcp", "C6_satdef"]


def verify_envcov_manifest(envcov_raw: Path) -> dict:
    """Verify SHA-256 and byte counts for every file listed in data/raw/envcov/MANIFEST.json."""
    manifest_path = envcov_raw / "MANIFEST.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_bytes = 0
    mismatches = []
    for item in entries:
        fpath = envcov_raw / item["path"]
        raw_bytes = fpath.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        total_bytes += len(raw_bytes)
        if digest != item["sha256"] or len(raw_bytes) != int(item["bytes"]):
            mismatches.append(item["path"])
    return {
        "n_entries": len(entries),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1e6,
        "mismatches": mismatches,
    }


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(0.5 * dp) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(0.5 * dl) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(h)))


def saturation_vp_tetens_kpa(temp_c: np.ndarray | float) -> np.ndarray | float:
    """Tetens saturation vapor pressure in kPa (protocol.md Section 4; standard meteorological formula)."""
    t = np.asarray(temp_c, dtype=float)
    return 0.61078 * np.exp(17.269 * t / (t + 237.3))


def parse_daymet_csv(csv_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Parse Daymet V4 R1 single-pixel CSV metadata header and daily series."""
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header_row = None
    meta = {}
    for idx, line in enumerate(lines):
        if line.startswith("Latitude:"):
            toks = line.replace("Latitude:", "").replace("Longitude:", "").split()
            meta["lat"] = float(toks[0])
            meta["lon"] = float(toks[1])
        elif line.startswith("Tile:"):
            meta["tile"] = int(line.split(":")[1].strip())
        elif line.startswith("Elevation:"):
            meta["elev_m"] = float(line.split(":")[1].split()[0])
        elif line.startswith("year,yday,"):
            header_row = idx
            break
    if header_row is None:
        raise ValueError(f"Missing Daymet table header in {csv_path}")

    years, ydays, months, days = [], [], [], []
    tmax_l, tmin_l, prcp_l, vp_l, swe_l = [], [], [], [], []
    for row in csv.DictReader(lines[header_row:]):
        yr = int(row["year"])
        yd = int(row["yday"])
        cal = datetime.date(yr, 1, 1) + datetime.timedelta(days=yd - 1)
        years.append(yr)
        ydays.append(yd)
        months.append(cal.month)
        days.append(cal.day)
        tmax_l.append(float(row["tmax (deg c)"]))
        tmin_l.append(float(row["tmin (deg c)"]))
        prcp_l.append(float(row["prcp (mm/day)"]))
        vp_l.append(float(row["vp (Pa)"]))
        swe_l.append(float(row["swe (kg/m^2)"]))

    tmax = np.asarray(tmax_l, dtype=float)
    tmin = np.asarray(tmin_l, dtype=float)
    tmean = 0.5 * (tmax + tmin)
    vp_kpa = np.asarray(vp_l, dtype=float) * 1e-3
    es_kpa = saturation_vp_tetens_kpa(tmean)
    vpd_kpa = np.maximum(0.0, es_kpa - vp_kpa)
    rh_est = np.clip(100.0 * vp_kpa / np.maximum(es_kpa, 1e-9), 0.0, 100.0)

    series = {
        "year": np.asarray(years, dtype=int),
        "yday": np.asarray(ydays, dtype=int),
        "month": np.asarray(months, dtype=int),
        "day": np.asarray(days, dtype=int),
        "tmax": tmax,
        "tmin": tmin,
        "tmean": tmean,
        "prcp": np.asarray(prcp_l, dtype=float),
        "vp_kpa": vp_kpa,
        "swe": np.asarray(swe_l, dtype=float),
        "vpd_kpa": vpd_kpa,
        "rh_est": rh_est,
    }
    return series, meta


def parse_nasa_power_json(json_path: Path) -> dict:
    """Parse NASA POWER daily point JSON."""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    param = doc["properties"]["parameter"]
    date_keys = sorted(param["RH2M"].keys())
    yr, mo, dy = [], [], []
    cols = {k: [] for k in param}
    for dk in date_keys:
        yr.append(int(dk[:4]))
        mo.append(int(dk[4:6]))
        dy.append(int(dk[6:8]))
        for k, mapping in param.items():
            val = float(mapping[dk])
            cols[k].append(np.nan if val <= -900.0 else val)
    out = {
        "year": np.asarray(yr, dtype=int),
        "month": np.asarray(mo, dtype=int),
        "day": np.asarray(dy, dtype=int),
    }
    for k, vals in cols.items():
        out[k] = np.asarray(vals, dtype=float)
    return out


def compute_annual_covariates(daymet: dict[str, np.ndarray], power: dict | None = None) -> dict[int, dict[str, float]]:
    """Compute the six primary Daymet tick covariates and secondary cross-check covariates aligned to nymph year t."""
    yr = daymet["year"]
    mo = daymet["month"]
    tmax = daymet["tmax"]
    tmin = daymet["tmin"]
    tmean = daymet["tmean"]
    prcp = daymet["prcp"]
    swe = daymet["swe"]
    vpd = daymet["vpd_kpa"]
    rh_dm = daymet["rh_est"]

    y_start = int(yr.min()) + 1
    y_end = int(yr.max())
    by_year: dict[int, dict[str, float]] = {}

    for t in range(y_start, y_end + 1):
        w_oct_sep = ((yr == t - 1) & (mo >= 10)) | ((yr == t) & (mo <= 9))
        w_nov_mar = ((yr == t - 1) & (mo >= 11)) | ((yr == t) & (mo <= 3))
        w_may_sep = (yr == t) & (mo >= 5) & (mo <= 9)
        w_apr_sep = (yr == t) & (mo >= 4) & (mo <= 9)
        w_feb_apr = (yr == t) & (mo >= 2) & (mo <= 4)
        w_cal = yr == t

        rec = {
            "C1_dd6": float(np.sum(np.maximum(0.0, tmean[w_oct_sep] - 6.0))),
            "C2_hotdry": float(np.sum((tmax[w_may_sep] > 25.0) & (prcp[w_may_sep] == 0.0))),
            "C3_wmin": float(np.min(tmin[w_nov_mar])),
            "C4_snow": float(np.sum(swe[w_nov_mar] > 0.0)),
            "C5_gsprcp": float(np.sum(prcp[w_apr_sep])),
            "C6_satdef": float(np.mean(vpd[w_may_sep])),
            # Station cross-validation & sensitivity series
            "mat_cal": float(np.mean(tmean[w_cal])),
            "dd0_oct_sep": float(np.sum(np.maximum(0.0, tmean[w_oct_sep]))),
            "dd10_oct_sep": float(np.sum(np.maximum(0.0, tmean[w_oct_sep] - 10.0))),
            "cold_winter_frac": float(np.mean(tmin[w_nov_mar] < -10.0)),
            "feb_apr_warm_dry_frac": float(np.mean((tmax[w_feb_apr] > 15.0) & (rh_dm[w_feb_apr] < 50.0))),
        }
        if power is not None:
            pyr, pmo = power["year"], power["month"]
            p_gs = (pyr == t) & (pmo >= 5) & (pmo <= 9)
            p_jun = (pyr == t) & (pmo == 6)
            p_win = ((pyr == t - 1) & (pmo >= 11)) | ((pyr == t) & (pmo <= 3))
            rec["P_gwettop_gs"] = float(np.nanmean(power["GWETTOP"][p_gs]))
            rec["P_rh_gs"] = float(np.nanmean(power["RH2M"][p_gs]))
            rec["P_rh_jun_sub82"] = float(np.sum(power["RH2M"][p_jun] < 82.0))
            rec["P_ts_wmin"] = float(np.nanmin(power["TS"][p_win]))
        by_year[t] = rec
    return by_year


def daily_maturation_rates(tmean: np.ndarray, scheme: str = "wallace2019") -> tuple[np.ndarray, np.ndarray]:
    """Daily engorged L->N and N->A development rates (day^-1)."""
    t = np.asarray(tmean, dtype=float)
    if scheme == "wallace2019":
        gate = (t >= 15.0).astype(float)
        r_ln = 0.04001 * np.exp(-(((t - 26.68) / 9.533) ** 2)) * gate
        r_na = 0.03173 * np.exp(-(((t - 23.85) / 9.042) ** 2)) * gate
        return r_ln, r_na
    if scheme == "ogden2004_recalled":
        tp = np.maximum(t, 0.0)
        r_ln = np.where(t > 8.5, (tp ** 2.275) / 101181.0, 0.0)
        r_na = np.where(t > 8.0, (tp ** 1.210) / 1596.0, 0.0)
        return r_ln, r_na
    raise ValueError(f"Unknown scheme: {scheme}")


def first_crossing_days(rates: np.ndarray, start_pos: int, threshold: float = 1.0) -> int | None:
    """Smallest integer L >= 1 with sum_{u=0}^{L-1} rates[start_pos + u] >= threshold."""
    csum = np.cumsum(rates[start_pos:])
    hits = np.flatnonzero(csum >= threshold)
    return (int(hits[0]) + 1) if len(hits) > 0 else None


def evaluate_thermal_durations(
    daymet: dict[str, np.ndarray],
    larval_years: list[int],
    quest_hazard_rate: float = 0.02,
    scheme: str = "wallace2019",
    drop_doys: np.ndarray | None = None,
) -> dict:
    """Compute premoult L->N (15 Aug), premoult N->A (1 Jun), and full phenological L->N stay."""
    yr = daymet["year"]
    yd = daymet["yday"]
    mo = daymet["month"]
    dy = daymet["day"]
    tmean = daymet["tmean"]

    r_ln, r_na = daily_maturation_rates(tmean, scheme=scheme)
    q_rate = quest_hazard_rate * (tmean >= 10.0).astype(float)

    pos_by_yd = {(int(y), int(d)): idx for idx, (y, d) in enumerate(zip(yr, yd))}
    mar1_pos = {int(y): idx for idx, (y, m, d) in enumerate(zip(yr, mo, dy)) if m == 3 and d == 1}

    if drop_doys is None:
        drop_doys = np.arange(196, 257, 5, dtype=int)
    weights = np.exp(-0.5 * (((drop_doys - 227.0) / 20.0) ** 2))
    weights /= weights.sum()

    ln_premoult = {}
    na_premoult = {}
    phen_by_yr = {}
    all_stays, all_wts = [], []
    censored = 0

    for y in larval_years:
        p_aug15 = pos_by_yd.get((y, 227))
        p_jun1 = pos_by_yd.get((y, 152))
        if p_aug15 is not None:
            d_ln = first_crossing_days(r_ln, p_aug15, 1.0)
            if d_ln is not None:
                ln_premoult[y] = d_ln
        if p_jun1 is not None:
            d_na = first_crossing_days(r_na, p_jun1, 1.0)
            if d_na is not None:
                na_premoult[y] = d_na

        p_mar1 = mar1_pos.get(y + 1)
        if p_mar1 is None:
            continue
        yr_s, yr_w = [], []
        for doy, wt in zip(drop_doys, weights):
            p0 = pos_by_yd.get((y, int(doy)))
            if p0 is None:
                continue
            dm = first_crossing_days(r_ln, p0, 1.0)
            if dm is None:
                censored += 1
                continue
            p_moult = p0 + dm
            p_quest_start = max(p_moult, p_mar1)
            dq = first_crossing_days(q_rate, p_quest_start, 1.0)
            if dq is None:
                censored += 1
                continue
            stay_d = float((p_quest_start - p0) + dq)
            yr_s.append(stay_d)
            yr_w.append(float(wt))
            all_stays.append(stay_d)
            all_wts.append(float(wt))
        if yr_w and sum(yr_w) > 0.9:
            wn = np.asarray(yr_w) / sum(yr_w)
            phen_by_yr[y] = float(np.sum(np.asarray(yr_s) * wn))

    ln_arr = np.asarray(list(ln_premoult.values()), dtype=float)
    na_arr = np.asarray(list(na_premoult.values()), dtype=float)
    yr_arr = np.asarray(list(phen_by_yr.values()), dtype=float)
    s_arr = np.asarray(all_stays, dtype=float)
    w_arr = np.asarray(all_wts, dtype=float)
    w_arr /= w_arr.sum()

    phen_mean = float(np.sum(s_arr * w_arr))
    phen_total_sd = math.sqrt(max(0.0, float(np.sum(w_arr * ((s_arr - phen_mean) ** 2)))))
    phen_inter_sd = float(yr_arr.std(ddof=1))

    return {
        "premoult_ln_mean_d": float(ln_arr.mean()),
        "premoult_na_mean_d": float(na_arr.mean()),
        "phen_ln_mean_d": phen_mean,
        "phen_ln_interannual_sd_d": phen_inter_sd,
        "phen_ln_interannual_cv": phen_inter_sd / float(yr_arr.mean()),
        "phen_ln_total_sd_d": phen_total_sd,
        "phen_ln_total_cv": phen_total_sd / phen_mean,
        "phen_ln_by_year": phen_by_yr,
        "premoult_ln_by_year": ln_premoult,
        "censored_subcohorts": int(censored),
    }


def load_neon_biological(neon_raw: Path) -> dict:
    """Recompute NEON Peromyscus indices, sigma_site, r1, nymphal B. burgdorferi s.l. NIP, and plot habitat."""
    pat_re = re.compile(
        r"^NEON\.D\d\d\.([A-Z]{4})\.(DP1\.\d{5}\.\d{3})\.([A-Za-z_]+)\.(\d{4}-\d{2})\.basic\.(\d{8}T\d{6}Z)\.csv$"
    )

    def select_latest_files(prod: str, tbl: str, sites: list[str]) -> list[Path]:
        picked = {}
        for s in sites:
            sdir = neon_raw / prod / s
            if not sdir.is_dir():
                continue
            for p in sorted(sdir.glob("*.csv")):
                m = pat_re.match(p.name)
                if m and m.group(1) == s and m.group(3) == tbl:
                    key = (s, m.group(4))
                    if key not in picked or m.group(5) > picked[key][0]:
                        picked[key] = (m.group(5), p)
        return [v[1] for _, v in sorted(picked.items())]

    seen_mam = set()
    trap_nights = defaultdict(set)
    pero_tags = defaultdict(set)
    mam_plot_coords = defaultdict(dict)
    valid_trap_codes = {"2", "3", "4", "5", "6"}

    for csv_p in select_latest_files("DP1.10072.001", "mam_pertrapnight", NEON_13):
        with csv_p.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                uid = r["uid"]
                if uid in seen_mam:
                    continue
                seen_mam.add(uid)
                cdate = r["collectDate"][:10]
                if not cdate[:4].isdigit():
                    continue
                site = r["siteID"]
                yr = int(cdate[:4])
                tcode = r["trapStatus"].strip()[:1]
                if tcode in valid_trap_codes:
                    trap_nights[(site, yr)].add((r["plotID"], cdate, r["trapCoordinate"].strip()))
                sci = r["scientificName"].split()
                if sci and sci[0] == "Peromyscus":
                    tag = r["tagID"].strip()
                    if tag:
                        pero_tags[(site, yr)].add(tag)
                pid = (r.get("plotID") or "").strip()
                lat_s = (r.get("decimalLatitude") or "").strip()
                lon_s = (r.get("decimalLongitude") or "").strip()
                if pid and lat_s and lon_s and pid not in mam_plot_coords[site]:
                    mam_plot_coords[site][pid] = (float(lat_s), float(lon_s))

    effort = {k: len(v) for k, v in trap_nights.items()}
    mouse_index = {k: 100.0 * len(pero_tags.get(k, ())) / tn for k, tn in effort.items() if tn >= 500}

    sigma_site, r1_site, mean_idx, mean_log_tn = {}, {}, {}, {}
    for s in NEON_13:
        ys = sorted(y for (ss, y), val in mouse_index.items() if ss == s and val > 0.0)
        ln_series = {y: math.log(mouse_index[(s, y)]) for y in ys}
        sigma_site[s] = float(np.std(list(ln_series.values()), ddof=1)) if len(ln_series) >= 5 else None
        consec = [(ln_series[y], ln_series[y + 1]) for y in ys if (y + 1) in ln_series]
        r1_site[s] = float(np.corrcoef([a for a, _ in consec], [b for _, b in consec])[0, 1]) if len(consec) >= 3 else None
        mean_idx[s] = float(np.mean([mouse_index[(s, y)] for y in ys])) if ys else float("nan")
        mean_log_tn[s] = float(np.mean([math.log(effort[(s, y)]) for y in ys])) if ys else float("nan")

    seen_tax = set()
    sub_to_tax = {}
    for csv_p in select_latest_files("DP1.10093.001", "tck_taxonomyProcessed", NEON_13):
        with csv_p.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["uid"] in seen_tax:
                    continue
                seen_tax.add(r["uid"])
                sub_to_tax.setdefault(r["subsampleID"].strip(), (r["acceptedTaxonID"].strip(), r["sexOrAge"].strip()))

    seen_pat = set()
    by_tick = {}
    for csv_p in select_latest_files("DP1.10092.001", "tck_pathogen", NEON_13):
        with csv_p.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["uid"] in seen_pat:
                    continue
                seen_pat.add(r["uid"])
                if not r["collectDate"][:4].isdigit():
                    continue
                tid = r["testingID"].strip()
                rec = by_tick.setdefault(
                    tid,
                    {"site": r["siteID"], "year": int(r["collectDate"][:4]), "sub": r["subsampleID"].strip(), "tests": defaultdict(set)},
                )
                if r["testResult"] in ("Positive", "Negative"):
                    rec["tests"][r["testPathogenName"]].add(r["testResult"])

    target_pathogen = "Borrelia burgdorferi sensu lato"
    nymph_counts = defaultdict(lambda: [0, 0])
    for rec in by_tick.values():
        taxon, stage = sub_to_tax.get(rec["sub"], ("", ""))
        if taxon != "IXOSCA" or stage != "Nymph":
            continue
        p_res = rec["tests"].get(target_pathogen)
        if not p_res:
            continue
        dna_q = rec["tests"].get("HardTick DNA Quality")
        if dna_q is not None and "Positive" not in dna_q:
            continue
        cell = nymph_counts[(rec["site"], rec["year"])]
        cell[0] += 1
        cell[1] += 1 if "Positive" in p_res else 0

    tested, positive, nip_site, test_years, yearly_nip = {}, {}, {}, {}, {}
    for s in NEON_13:
        ys = sorted(y for (ss, y), c in nymph_counts.items() if ss == s and c[0] > 0)
        tot_n = sum(nymph_counts[(s, y)][0] for y in ys)
        tot_pos = sum(nymph_counts[(s, y)][1] for y in ys)
        tested[s] = tot_n
        positive[s] = tot_pos
        nip_site[s] = (tot_pos / tot_n) if tot_n > 0 else None
        test_years[s] = ys
        yearly_nip[s] = {y: nymph_counts[(s, y)][1] / nymph_counts[(s, y)][0] for y in ys if nymph_counts[(s, y)][0] >= 20}

    # Habitat & spatial summary for the 6 included sites
    forest_nlcd = {"deciduousForest", "mixedForest", "evergreenForest", "woodyWetlands"}
    habitat = {}
    for s in NEON_6:
        plot_meta = {}
        plot_coords = {}
        for csv_p in select_latest_files("DP1.10093.001", "tck_fielddata", [s]):
            with csv_p.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    pid = (r.get("plotID") or "").strip()
                    nlcd = (r.get("nlcdClass") or "").strip()
                    elev = (r.get("elevation") or "").strip()
                    la = (r.get("decimalLatitude") or "").strip()
                    lo = (r.get("decimalLongitude") or "").strip()
                    if pid and nlcd and elev:
                        plot_meta.setdefault(pid, (nlcd, float(elev)))
                    if pid and la and lo and pid not in plot_coords:
                        plot_coords[pid] = (float(la), float(lo))
        t_lat = float(np.mean([v[0] for v in plot_coords.values()]))
        t_lon = float(np.mean([v[1] for v in plot_coords.values()]))
        m_lat = float(np.mean([v[0] for v in mam_plot_coords[s].values()]))
        m_lon = float(np.mean([v[1] for v in mam_plot_coords[s].values()]))
        habitat[s] = {
            "forest_plot_frac": sum(1 for nlcd, _ in plot_meta.values() if nlcd in forest_nlcd) / len(plot_meta),
            "elevation_m": float(np.mean([el for _, el in plot_meta.values()])),
            "tick_lat": t_lat,
            "tick_lon": t_lon,
            "tick_to_mammal_centroid_km": great_circle_km(t_lat, t_lon, m_lat, m_lon),
        }

    return {
        "effort": effort,
        "index": mouse_index,
        "sigma": sigma_site,
        "r1": r1_site,
        "mean_index": mean_idx,
        "mean_log_trapnights": mean_log_tn,
        "cells": dict(nymph_counts),
        "tested": tested,
        "positive": positive,
        "nip_site": nip_site,
        "test_years": test_years,
        "yearly_nip": yearly_nip,
        "habitat": habitat,
    }


def load_cary_biological(cary_raw: Path) -> dict:
    """Load Cary Dryad primary (1991-2015) and Figshare holdout (2014-2022) datasets."""
    dry_csv = cary_raw / "dryad_d1c8046_forest_food_web_2018" / "Lyme_Study_Dataset.csv"
    hol_csv = cary_raw / "figshare_27871269_nymphal_infection_2025" / "allthedata_grid.csv"
    ind_csv = cary_raw / "figshare_27871269_nymphal_infection_2025" / "NymphalInfectionData.csv"

    with dry_csv.open(newline="", encoding="latin-1") as fh:
        dry_rows = [r for r in csv.DictReader(fh) if r["Grid"] not in ("", "NA")]
    with hol_csv.open(newline="", encoding="latin-1") as fh:
        hol_rows = [r for r in csv.DictReader(fh) if r["Grid"] not in ("", "NA")]
    with ind_csv.open(newline="", encoding="latin-1") as fh:
        ind_rows = list(csv.DictReader(fh))

    def extract_series(rows: list[dict], gcol: str, vcol: str) -> dict[str, dict[int, float]]:
        res: dict[str, dict[int, float]] = defaultdict(dict)
        for r in rows:
            val = (r.get(vcol) or "").strip()
            if val not in ("", "NA"):
                res[r[gcol]][int(r["Year"])] = float(val)
        return dict(res)

    dplots = sorted({r["Grid"] for r in dry_rows})
    hplots = sorted({r["Grid"] for r in hol_rows})

    station_by_yr: dict[int, dict[str, float]] = defaultdict(dict)
    pcr_share: dict[str, dict[int, float]] = defaultdict(dict)
    n_tested_dry: dict[tuple[str, int], int] = {}

    for r in dry_rows:
        g = r["Grid"]
        y = int(r["Year"])
        for scol in ("mat", "degree_days", "cold_winter_days", "feb_apr_warm_days", "snowfall_cm"):
            sv = (r.get(scol) or "").strip()
            if sv not in ("", "NA"):
                station_by_yr[y][scol] = float(sv)
        has_tfa = (r.get("TFA_NIP_CORR") or "").strip() not in ("", "NA")
        has_pcr = (r.get("PCR_NIP") or "").strip() not in ("", "NA")
        pf = (r.get("PCR_fraction") or "").strip()
        if pf not in ("", "NA"):
            pcr_share[g][y] = float(pf)
        elif has_pcr and not has_tfa:
            pcr_share[g][y] = 1.0
        elif has_tfa and not has_pcr:
            pcr_share[g][y] = 0.0
        tot = [float(r[c]) for c in ("TFA_total", "PCR_total") if (r.get(c) or "").strip() not in ("", "NA")]
        if tot:
            n_tested_dry[(g, y)] = int(sum(tot))

    # Cross-check holdout grid file against individual nymph rows
    ind_counts = Counter()
    ind_pos = Counter()
    ind_valid = Counter()
    for r in ind_rows:
        key = (r["Grid"], int(r["Year"]))
        ind_counts[key] += 1
        bb = r["B.burgdorferi"].strip()
        if bb in ("0", "1"):
            ind_valid[key] += 1
            ind_pos[key] += int(bb)

    hol_n = extract_series(hol_rows, "Grid", "n")
    hol_bb = extract_series(hol_rows, "Grid", "Bburgd")
    hol_mice = extract_series(hol_rows, "Grid", "PL.mnRawLyr")
    max_hol_diff = 0.0
    for g in hplots:
        for y, pval in hol_bb.get(g, {}).items():
            if ind_valid[(g, y)] > 0:
                max_hol_diff = max(max_hol_diff, abs(pval - ind_pos[(g, y)] / ind_valid[(g, y)]))

    dry_mice = extract_series(dry_rows, "Grid", "PL_grid")
    dry_nip = extract_series(dry_rows, "Grid", "NIP")
    dry_don = extract_series(dry_rows, "Grid", "DON")
    dry_lpm = extract_series(dry_rows, "Grid", "LPM")
    dry_chip = extract_series(dry_rows, "Grid", "TS_grid")

    cary_sigma = float(
        np.mean([np.std([math.log(v) for v in dry_mice[g].values() if v > 0.0], ddof=1) for g in dplots])
    )
    pbar_primary = float(np.mean([v for g in dplots for v in dry_nip[g].values()]))

    return {
        "dplots": dplots,
        "hplots": hplots,
        "dry_mice": dry_mice,
        "dry_nip": dry_nip,
        "dry_don": dry_don,
        "dry_lpm": dry_lpm,
        "dry_chip": dry_chip,
        "pcr_share": dict(pcr_share),
        "n_tested_dry": n_tested_dry,
        "station_dry": dict(station_by_yr),
        "hol_n": hol_n,
        "hol_bb": hol_bb,
        "hol_mice": hol_mice,
        "hol_ind_max_diff": max_hol_diff,
        "cary_sigma": cary_sigma,
        "pbar_primary": pbar_primary,
    }
