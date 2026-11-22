#!/usr/bin/env python3
"""Data loader and 28-day step aggregator for case study `voles`.

Reads untrusted raw files from `data/raw/voles/` using standard-library parsers
only (`csv`, `json`, `zipfile`, `xml.etree.ElementTree`, and `biff8_reader.py`),
following `protocol.md` (including Amendment 1) without modifying any raw file.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from biff8_reader import excel_serial_to_date, read_biff8_workbook, _decode_rk

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RAW_VOLES = os.path.join(REPO_ROOT, "data", "raw", "voles")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on WGS84 mean spherical radius (6371.0 km)."""
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(0.5 * dp) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(0.5 * dl) ** 2)
    return 2.0 * r_earth * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def verify_parsers() -> dict[str, Any]:
    """Rule 7 verification checks on BIFF8 decoding, Excel serial dates, and Haversine."""
    assert excel_serial_to_date(37370) == datetime.date(2002, 4, 24)
    assert excel_serial_to_date(39945) == datetime.date(2009, 5, 12)
    assert excel_serial_to_date(1) == datetime.date(1900, 1, 1)
    # RK integer 42 -> (42 << 2) | 0x02
    assert _decode_rk((42 << 2) | 0x02) == 42
    # RK integer -15 / 100 -> ((-15 & 0x3fffffff) << 2) | 0x03
    rk_neg = (((-15) << 2) & 0xFFFFFFFF) | 0x03
    assert abs(float(_decode_rk(rk_neg)) - (-0.15)) < 1e-12
    # Haversine 1 deg latitude ~ 111.195 km
    d1 = haversine_km(55.0, -2.5, 56.0, -2.5)
    assert abs(d1 - 111.1949) < 0.05
    return {
        "passed": True,
        "excel_date_2002_04_24": str(excel_serial_to_date(37370)),
        "excel_date_2009_05_12": str(excel_serial_to_date(39945)),
        "haversine_1deg_lat_km": round(d1, 4),
    }


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return None if math.isnan(f) else f
    s = str(val).strip()
    if s in ("", "NA", "NaN", "*", ".", "N/A", "null", "NULL"):
        return None
    try:
        f = float(s)
        return None if math.isnan(f) else f
    except ValueError:
        return None


def _parse_uk_or_iso_date(val: Any) -> datetime.date | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val > 10000:
            return excel_serial_to_date(float(val))
        return None
    s = str(val).strip()
    if not s or s == "*":
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 1. Weather Loader & 28-Day Step Matcher
# ---------------------------------------------------------------------------

def load_daily_ghcnd(relpath: str) -> dict[datetime.date, dict[str, float]]:
    path = os.path.join(RAW_VOLES, relpath)
    out: dict[datetime.date, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = _parse_uk_or_iso_date(row.get("DATE"))
            if dt is None:
                continue
            rec: dict[str, float] = {}
            for k in ("TMAX", "TMIN", "TAVG", "PRCP", "SNWD"):
                v = _parse_float(row.get(k))
                if v is not None and v > -900:
                    rec[k] = v
            if "TMAX" in rec and "TMIN" in rec:
                rec["TMID"] = 0.5 * (rec["TMAX"] + rec["TMIN"])
            elif "TAVG" in rec:
                rec["TMID"] = rec["TAVG"]
            out[dt] = rec
    return out


def load_daily_nasapower(relpath: str) -> tuple[dict[datetime.date, dict[str, float]], list[float]]:
    path = os.path.join(RAW_VOLES, relpath)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    coords = doc.get("geometry", {}).get("coordinates", [0.0, 0.0, 0.0])  # [lon, lat, elev]
    params = doc["properties"]["parameter"]
    dates_str = sorted(params["T2M"].keys())
    out: dict[datetime.date, dict[str, float]] = {}
    for ds in dates_str:
        dt = datetime.date(int(ds[0:4]), int(ds[4:6]), int(ds[6:8]))
        rec: dict[str, float] = {}
        for k in ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M"):
            v = _parse_float(params.get(k, {}).get(ds))
            if v is not None and v > -900:
                rec[k] = v
        out[dt] = rec
    return out, coords


def build_regional_daily_weather() -> dict[str, Any]:
    """Load and cross-check daily meteorology for Kielder Forest (UK) and Konnevesi (Finland)."""
    # Kielder Forest centroid from EIDC bounding box [-2.62, 55.21]..[-2.53, 55.25]
    kielder_lat, kielder_lon = 55.2300, -2.5750
    kc_daily = load_daily_ghcnd("weather/ghcnd_KielderCastle_UKE00105892_2007_2017.csv")
    esk_daily = load_daily_ghcnd("weather/ghcnd_Eskdalemuir_UK000003162_2007_2017.csv")
    kp_daily, kp_coords = load_daily_nasapower("weather/nasapower_Kielder_55.23N_2.575W_2007_2017.json")

    # Compute mean offset (Kielder Castle - Eskdalemuir) on overlapping days for rare gap-filling
    diffs_ke = [
        kc_daily[d]["TMID"] - esk_daily[d]["TMID"]
        for d in kc_daily
        if d in esk_daily and "TMID" in kc_daily[d] and "TMID" in esk_daily[d]
    ]
    offset_ke = float(np.mean(diffs_ke)) if diffs_ke else 0.0

    # Konnevesi Research Station (62.6280 N, 26.2880 E)
    konn_lat, konn_lon = 62.6280, 26.2880
    ves_daily = load_daily_ghcnd("weather/ghcnd_VesantoSonkari_FIE00144637_2001_2009.csv")
    jyv_daily = load_daily_ghcnd("weather/ghcnd_Jyvaskyla_FI000002401_2001_2009.csv")
    lah_daily = load_daily_ghcnd("weather/ghcnd_KonnevesiLahdenkyla_FIE00144657_2001_2007.csv")
    fp_daily, fp_coords = load_daily_nasapower("weather/nasapower_Konnevesi_62.628N_26.288E_2001_2009.json")

    diffs_vj = [
        ves_daily[d]["TMID"] - jyv_daily[d]["TMID"]
        for d in ves_daily
        if d in jyv_daily and "TMID" in ves_daily[d] and "TMID" in jyv_daily[d]
    ]
    offset_vj = float(np.mean(diffs_vj)) if diffs_vj else 0.0

    # Cross-check daily correlations (primary station TMID vs NASA POWER T2M)
    k_ov = [(kc_daily[d]["TMID"], kp_daily[d]["T2M"]) for d in kc_daily if d in kp_daily and "TMID" in kc_daily[d] and "T2M" in kp_daily[d]]
    f_ov = [(ves_daily[d]["TMID"], fp_daily[d]["T2M"]) for d in ves_daily if d in fp_daily and "TMID" in ves_daily[d] and "T2M" in fp_daily[d]]
    k_arr = np.array(k_ov)
    f_arr = np.array(f_ov)

    # Nearest 0.5x0.625 MERRA-2 grid cell centres
    kp_grid_lat, kp_grid_lon = round(kielder_lat * 2) / 2.0, round(kielder_lon / 0.625) * 0.625
    fp_grid_lat, fp_grid_lon = round(konn_lat * 2) / 2.0, round(konn_lon / 0.625) * 0.625

    provenance = {
        "kielder": {
            "site_centroid_lat_lon": [kielder_lat, kielder_lon],
            "primary_station": {
                "id": "UKE00105892",
                "name": "Kielder Castle",
                "source": "NOAA GHCN-Daily v1 (doi:10.7289/V5D21VHZ)",
                "lat_lon": [55.2339, -2.5789],
                "elevation_m": 201.0,
                "distance_km": round(haversine_km(kielder_lat, kielder_lon, 55.2339, -2.5789), 2),
                "n_days_loaded": len(kc_daily),
                "date_span": [str(min(kc_daily)), str(max(kc_daily))],
            },
            "reference_snow_station": {
                "id": "UK000003162",
                "name": "Eskdalemuir",
                "source": "NOAA GHCN-Daily v1 (doi:10.7289/V5D21VHZ)",
                "lat_lon": [55.3167, -3.2000],
                "elevation_m": 242.0,
                "distance_km": round(haversine_km(kielder_lat, kielder_lon, 55.3167, -3.2000), 2),
                "n_days_loaded": len(esk_daily),
                "mean_temp_offset_kc_minus_esk_C": round(offset_ke, 3),
            },
            "gridded_power": {
                "source": "NASA POWER Agroclimatology v2.0 daily (doi:10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY)",
                "requested_lat_lon": [kielder_lat, kielder_lon],
                "nearest_merra2_cell_centre_lat_lon": [kp_grid_lat, kp_grid_lon],
                "distance_to_cell_centre_km": round(haversine_km(kielder_lat, kielder_lon, kp_grid_lat, kp_grid_lon), 2),
                "grid_elevation_m": kp_coords[2] if len(kp_coords) > 2 else None,
                "n_days_loaded": len(kp_daily),
                "daily_r_vs_kielder_castle_T": round(float(np.corrcoef(k_arr[:, 0], k_arr[:, 1])[0, 1]), 4),
                "daily_rmse_vs_kielder_castle_C": round(float(np.sqrt(np.mean((k_arr[:, 0] - k_arr[:, 1]) ** 2))), 3),
            },
        },
        "konnevesi": {
            "site_centroid_lat_lon": [konn_lat, konn_lon],
            "primary_station": {
                "id": "FIE00144637",
                "name": "Vesanto Sonkari",
                "source": "NOAA GHCN-Daily v1 (doi:10.7289/V5D21VHZ)",
                "lat_lon": [62.9214, 26.4281],
                "elevation_m": 114.0,
                "distance_km": round(haversine_km(konn_lat, konn_lon, 62.9214, 26.4281), 2),
                "n_days_loaded": len(ves_daily),
                "date_span": [str(min(ves_daily)), str(max(ves_daily))],
            },
            "reference_snow_station": {
                "id": "FI000002401",
                "name": "Jyvaskyla Lentoasema",
                "source": "NOAA GHCN-Daily v1 (doi:10.7289/V5D21VHZ)",
                "lat_lon": [62.4019, 25.6783],
                "elevation_m": 140.0,
                "distance_km": round(haversine_km(konn_lat, konn_lon, 62.4019, 25.6783), 2),
                "n_days_loaded": len(jyv_daily),
                "mean_temp_offset_ves_minus_jyv_C": round(offset_vj, 3),
            },
            "local_precip_station": {
                "id": "FIE00144657",
                "name": "Konnevesi Lahdenkyla",
                "source": "NOAA GHCN-Daily v1 (doi:10.7289/V5D21VHZ)",
                "lat_lon": [62.5667, 26.2831],
                "elevation_m": 93.0,
                "distance_km": round(haversine_km(konn_lat, konn_lon, 62.5667, 26.2831), 2),
                "n_days_loaded": len(lah_daily),
                "date_span": [str(min(lah_daily)), str(max(lah_daily))],
            },
            "gridded_power": {
                "source": "NASA POWER Agroclimatology v2.0 daily (doi:10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY)",
                "requested_lat_lon": [konn_lat, konn_lon],
                "nearest_merra2_cell_centre_lat_lon": [fp_grid_lat, fp_grid_lon],
                "distance_to_cell_centre_km": round(haversine_km(konn_lat, konn_lon, fp_grid_lat, fp_grid_lon), 2),
                "grid_elevation_m": fp_coords[2] if len(fp_coords) > 2 else None,
                "n_days_loaded": len(fp_daily),
                "daily_r_vs_vesanto_T": round(float(np.corrcoef(f_arr[:, 0], f_arr[:, 1])[0, 1]), 4),
                "daily_rmse_vs_vesanto_C": round(float(np.sqrt(np.mean((f_arr[:, 0] - f_arr[:, 1]) ** 2))), 3),
            },
        },
    }

    return {
        "provenance": provenance,
        "kielder_daily": (kc_daily, esk_daily, kp_daily, offset_ke),
        "konnevesi_daily": (ves_daily, jyv_daily, lah_daily, fp_daily, offset_vj),
    }


def window_weather_28d(
    region_tuple: tuple,
    region_name: str,
    t_origin: datetime.date,
    step_k: int,
) -> dict[str, float]:
    """Aggregate 28-day weather over [t_origin + 28*k, t_origin + 28*(k+1) - 1]."""
    d_start = t_origin + datetime.timedelta(days=28 * step_k)
    temps_stn, temps_pwr, prcp_stn, prcp_pwr, rh_pwr, snow_ind, frost_ind = [], [], [], [], [], [], []

    if region_name == "kielder":
        kc_daily, esk_daily, kp_daily, offset_ke = region_tuple
        for i in range(28):
            d = d_start + datetime.timedelta(days=i)
            # Primary station temperature: Kielder Castle (0.5 km), fallback Eskdalemuir + offset, then POWER
            if d in kc_daily and "TMID" in kc_daily[d]:
                t_val = kc_daily[d]["TMID"]
                tmin_val = kc_daily[d].get("TMIN", t_val)
            elif d in esk_daily and "TMID" in esk_daily[d]:
                t_val = esk_daily[d]["TMID"] + offset_ke
                tmin_val = esk_daily[d].get("TMIN", t_val) + offset_ke
            elif d in kp_daily and "T2M" in kp_daily[d]:
                t_val = kp_daily[d]["T2M"]
                tmin_val = kp_daily[d].get("T2M_MIN", t_val)
            else:
                t_val = None
                tmin_val = None
            if t_val is not None:
                temps_stn.append(t_val)
                frost_ind.append(1.0 if (tmin_val is not None and tmin_val < 0.0) else 0.0)
            if d in kc_daily and "PRCP" in kc_daily[d]:
                prcp_stn.append(kc_daily[d]["PRCP"])
            elif d in esk_daily and "PRCP" in esk_daily[d]:
                prcp_stn.append(esk_daily[d]["PRCP"])
            if d in esk_daily and "SNWD" in esk_daily[d]:
                snow_ind.append(1.0 if esk_daily[d]["SNWD"] > 0 else 0.0)
            elif t_val is not None:
                snow_ind.append(0.0 if t_val > 3.0 else np.nan)
            if d in kp_daily:
                if "T2M" in kp_daily[d]:
                    temps_pwr.append(kp_daily[d]["T2M"])
                if "PRECTOTCORR" in kp_daily[d]:
                    prcp_pwr.append(kp_daily[d]["PRECTOTCORR"])
                if "RH2M" in kp_daily[d]:
                    rh_pwr.append(kp_daily[d]["RH2M"])
    else:
        ves_daily, jyv_daily, lah_daily, fp_daily, offset_vj = region_tuple
        for i in range(28):
            d = d_start + datetime.timedelta(days=i)
            if d in ves_daily and "TMID" in ves_daily[d]:
                t_val = ves_daily[d]["TMID"]
                tmin_val = ves_daily[d].get("TMIN", t_val)
            elif d in jyv_daily and "TMID" in jyv_daily[d]:
                t_val = jyv_daily[d]["TMID"] + offset_vj
                tmin_val = jyv_daily[d].get("TMIN", t_val) + offset_vj
            elif d in fp_daily and "T2M" in fp_daily[d]:
                t_val = fp_daily[d]["T2M"]
                tmin_val = fp_daily[d].get("T2M_MIN", t_val)
            else:
                t_val = None
                tmin_val = None
            if t_val is not None:
                temps_stn.append(t_val)
                frost_ind.append(1.0 if (tmin_val is not None and tmin_val < 0.0) else 0.0)
            if d in lah_daily and "PRCP" in lah_daily[d]:
                prcp_stn.append(lah_daily[d]["PRCP"])
            elif d in ves_daily and "PRCP" in ves_daily[d]:
                prcp_stn.append(ves_daily[d]["PRCP"])
            elif d in jyv_daily and "PRCP" in jyv_daily[d]:
                prcp_stn.append(jyv_daily[d]["PRCP"])
            if d in jyv_daily and "SNWD" in jyv_daily[d]:
                snow_ind.append(1.0 if jyv_daily[d]["SNWD"] > 0 else 0.0)
            if d in fp_daily:
                if "T2M" in fp_daily[d]:
                    temps_pwr.append(fp_daily[d]["T2M"])
                if "PRECTOTCORR" in fp_daily[d]:
                    prcp_pwr.append(fp_daily[d]["PRECTOTCORR"])
                if "RH2M" in fp_daily[d]:
                    rh_pwr.append(fp_daily[d]["RH2M"])

    sn_arr = [x for x in snow_ind if not math.isnan(x)]
    return {
        "date_start": str(d_start),
        "T_28d_C": float(np.mean(temps_stn)) if temps_stn else float("nan"),
        "T_power_28d_C": float(np.mean(temps_pwr)) if temps_pwr else float("nan"),
        "P_28d_mm": float(np.mean(prcp_stn) * 28.0) if prcp_stn else float("nan"),
        "P_power_28d_mm": float(np.mean(prcp_pwr) * 28.0) if prcp_pwr else float("nan"),
        "RH_28d_pct": float(np.mean(rh_pwr)) if rh_pwr else float("nan"),
        "frost_frac_28d": float(np.mean(frost_ind)) if frost_ind else float("nan"),
        "snow_frac_28d": float(np.mean(sn_arr)) if sn_arr else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Biological Loaders: K15, K08 (+ Arriero18), F02, F23
# ---------------------------------------------------------------------------

def load_kielder_2015_2017(weather_bundle: dict[str, Any]) -> dict[str, Any]:
    """Load Kielder Forest 2015-2017 EIDC dataset (`K15`) onto 28-day steps."""
    csv_path = os.path.join(
        RAW_VOLES, "kielder_eidc_2015_2017", "extracted", "data", "long_blood_qpcr_fv.2.csv"
    )
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_text = f.read()

    # Repair missing newline between column 77 `TOLLIP` and first row `A0001` in memory (Amendment 1)
    assert "TOLLIPA0001," in raw_text, "Expected EIDC header junction TOLLIPA0001, not found"
    repaired_text = raw_text.replace("TOLLIPA0001,", "TOLLIP\nA0001,", 1)

    reader = csv.DictReader(io.StringIO(repaired_text))
    cols = list(reader.fieldnames or [])
    assert len(cols) == 77 and cols[0] == "bloodNum" and cols[-1] == "TOLLIP", f"Unexpected K15 cols: {len(cols)}"

    parsed_rows = []
    all_dates = []
    grid_treat_by_site: dict[str, set[str]] = {}
    species_counts: dict[str, int] = {}
    n_untreated_rows = 0
    n_treated_rows = 0

    for row in reader:
        site = (row.get("site") or "").strip()
        dt = _parse_uk_or_iso_date(row.get("dateCapt"))
        if not site or dt is None:
            continue
        sp_code = (row.get("species") or "").strip() or "UNKNOWN"
        species_counts[sp_code] = species_counts.get(sp_code, 0) + 1
        all_dates.append(dt)
        gt = (row.get("gridTreat") or "").strip()
        if gt:
            grid_treat_by_site.setdefault(site, set()).add(gt)

        vid = (row.get("chip") or "").strip() or (row.get("bloodNum") or "").strip()
        sess = (row.get("session2") or "").strip() or f"{row.get('year')}_{row.get('session')}"
        tw = _parse_float(row.get("trapsWorking"))
        to = _parse_float(row.get("trapsOut"))
        eff_traps = tw if (tw is not None and tw > 0) else (to if (to is not None and to > 0) else 197.0)

        treat_flag = (row.get("treat") or "").strip().upper() == "TRUE"
        fl_val = _parse_float(row.get("frontline"))
        pr_val = _parse_float(row.get("profender"))
        tot_tr = _parse_float(row.get("totTreat"))
        is_untreated_now = (not treat_flag) and (fl_val is None or fl_val == 0.0) and (pr_val is None or pr_val == 0.0)
        is_never_treated = is_untreated_now and (tot_tr is None or tot_tr == 0.0)
        if is_untreated_now:
            n_untreated_rows += 1
        else:
            n_treated_rows += 1

        bar_rq = _parse_float(row.get("BAR"))
        bab_rq = _parse_float(row.get("BAB"))
        flea_cnt = _parse_float(row.get("flea"))
        tick_cnt = _parse_float(row.get("tick"))

        parsed_rows.append({
            "site": site,
            "date": dt,
            "session_key": sess,
            "vid": vid,
            "gridTreat": gt,
            "trapsWorking": eff_traps,
            "untreated_now": is_untreated_now,
            "never_treated": is_never_treated,
            "BAR": bar_rq,
            "BAB": bab_rq,
            "flea": flea_cnt,
            "tick": tick_cnt,
        })

    t_origin = min(all_dates)
    t_max = max(all_dates)
    max_step = (t_max - t_origin).days // 28

    # First aggregate biweekly sessions within (site, step_k, session_key) for effort-normalised host density
    by_site_step: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in parsed_rows:
        k = (r["date"] - t_origin).days // 28
        r["step_k"] = k
        by_site_step.setdefault((r["site"], k), []).append(r)

    sites = sorted({r["site"] for r in parsed_rows})
    weather_steps = {
        k: window_weather_28d(weather_bundle["kielder_daily"], "kielder", t_origin, k)
        for k in range(-2, max_step + 2)
    }

    site_series: dict[str, list[dict[str, Any]]] = {}
    for s in sites:
        windows = []
        steps_present = sorted({k for (st, k) in by_site_step if st == s})
        for k in steps_present:
            rows_sk = by_site_step[(s, k)]
            # Group by biweekly session_key inside this 28-day step
            by_sess: dict[str, list[dict[str, Any]]] = {}
            for r in rows_sk:
                by_sess.setdefault(r["session_key"], []).append(r)
            sess_indices = []
            sess_counts = []
            sess_traps = []
            for skey, srows in by_sess.items():
                n_ind = len({x["vid"] for x in srows})
                med_traps = float(np.median([x["trapsWorking"] for x in srows]))
                sess_indices.append(100.0 * n_ind / med_traps)
                sess_counts.append(float(n_ind))
                sess_traps.append(med_traps)

            h_per_100 = float(np.mean(sess_indices))
            raw_per_sess = float(np.mean(sess_counts))
            n_distinct_28d = len({x["vid"] for x in rows_sk})
            n_distinct_untreated = len({x["vid"] for x in rows_sk if x["untreated_now"]})

            # Deduplicate individuals within (site, k) for pathogen qPCR & ectoparasites
            def _agg_animals(subset_rows: list[dict[str, Any]]) -> dict[str, Any]:
                by_v: dict[str, list[dict[str, Any]]] = {}
                for x in subset_rows:
                    by_v.setdefault(x["vid"], []).append(x)
                bar_vals, bab_vals = [], []
                bar_first, bab_first = [], []
                bar_no_bab = []
                fleas, ticks = [], []
                for vid, vrows in by_v.items():
                    vrows_sorted = sorted(vrows, key=lambda z: z["date"])
                    b_list = [z["BAR"] for z in vrows_sorted if z["BAR"] is not None]
                    bb_list = [z["BAB"] for z in vrows_sorted if z["BAB"] is not None]
                    fl_list = [z["flea"] for z in vrows_sorted if z["flea"] is not None]
                    tk_list = [z["tick"] for z in vrows_sorted if z["tick"] is not None]
                    if b_list:
                        bar_vals.append(max(b_list))
                        bar_first.append(b_list[0])
                    if bb_list:
                        bab_vals.append(max(bb_list))
                        bab_first.append(bb_list[0])
                    if b_list and bb_list and max(bb_list) < 1.0:
                        bar_no_bab.append(max(b_list))
                    if fl_list:
                        fleas.append(float(np.mean(fl_list)))
                    if tk_list:
                        ticks.append(float(np.mean(tk_list)))
                return {
                    "bar_max": bar_vals,
                    "bab_max": bab_vals,
                    "bar_first": bar_first,
                    "bab_first": bab_first,
                    "bar_bab_neg": bar_no_bab,
                    "fleas": fleas,
                    "ticks": ticks,
                }

            untr_agg = _agg_animals([x for x in rows_sk if x["untreated_now"]])
            never_agg = _agg_animals([x for x in rows_sk if x["never_treated"]])
            all_agg = _agg_animals(rows_sk)

            def _cnt(arr: list[float], thr: float, strict_gt: bool = False) -> tuple[int, int]:
                n_t = len(arr)
                if strict_gt:
                    pos = sum(1 for v in arr if v > thr)
                else:
                    pos = sum(1 for v in arr if v >= thr)
                return pos, n_t

            bar_pos, bar_n = _cnt(untr_agg["bar_max"], 1.0)
            bab_pos, bab_n = _cnt(untr_agg["bab_max"], 1.0)

            mean_ord = float(np.mean([x["date"].toordinal() for x in rows_sk]))
            mean_dt = datetime.date.fromordinal(int(round(mean_ord)))
            w_cur = weather_steps[k]
            w_prev = weather_steps[k - 1]

            windows.append({
                "site": s,
                "gridTreat": sorted(grid_treat_by_site.get(s, set())),
                "step_k": k,
                "mean_date": str(mean_dt),
                "n_sessions_in_28d": len(by_sess),
                "mean_traps_working": float(np.mean(sess_traps)),
                "n_distinct_voles_28d": n_distinct_28d,
                "n_distinct_untreated_28d": n_distinct_untreated,
                "H_per_100_traps": h_per_100,
                "H_raw_per_session": raw_per_sess,
                # Primary untreated RQ >= 1.0
                "bar_pos": bar_pos,
                "bar_n": bar_n,
                "bab_pos": bab_pos,
                "bab_n": bab_n,
                # Sensitivity thresholds & subsets
                "bar_pos_rq05": _cnt(untr_agg["bar_max"], 0.5)[0],
                "bab_pos_rq05": _cnt(untr_agg["bab_max"], 0.5)[0],
                "bar_pos_rqgt0": _cnt(untr_agg["bar_max"], 0.0, strict_gt=True)[0],
                "bab_pos_rqgt0": _cnt(untr_agg["bab_max"], 0.0, strict_gt=True)[0],
                "bar_first_pos": _cnt(untr_agg["bar_first"], 1.0)[0],
                "bab_first_pos": _cnt(untr_agg["bab_first"], 1.0)[0],
                "bar_never_pos": _cnt(never_agg["bar_max"], 1.0)[0],
                "bar_never_n": len(never_agg["bar_max"]),
                "bab_never_pos": _cnt(never_agg["bab_max"], 1.0)[0],
                "bab_never_n": len(never_agg["bab_max"]),
                "bar_all_pos": _cnt(all_agg["bar_max"], 1.0)[0],
                "bar_all_n": len(all_agg["bar_max"]),
                "bab_all_pos": _cnt(all_agg["bab_max"], 1.0)[0],
                "bab_all_n": len(all_agg["bab_max"]),
                "bar_babneg_pos": _cnt(untr_agg["bar_bab_neg"], 1.0)[0],
                "bar_babneg_n": len(untr_agg["bar_bab_neg"]),
                # Ectoparasites on untreated voles
                "mean_fleas": float(np.mean(untr_agg["fleas"])) if untr_agg["fleas"] else float("nan"),
                "mean_ticks": float(np.mean(untr_agg["ticks"])) if untr_agg["ticks"] else float("nan"),
                "flea_prev": float(np.mean([v > 0 for v in untr_agg["fleas"]])) if untr_agg["fleas"] else float("nan"),
                "tick_prev": float(np.mean([v > 0 for v in untr_agg["ticks"]])) if untr_agg["ticks"] else float("nan"),
                # Weather matched to step k and preceding step k-1
                "T_28d_C": w_cur["T_28d_C"],
                "P_28d_mm": w_cur["P_28d_mm"],
                "RH_28d_pct": w_cur["RH_28d_pct"],
                "frost_frac_28d": w_cur["frost_frac_28d"],
                "snow_frac_28d": w_cur["snow_frac_28d"],
                "T_prev28d_C": w_prev["T_28d_C"],
                "P_prev28d_mm": w_prev["P_28d_mm"],
            })
        site_series[s] = windows

    return {
        "dataset_id": "K15",
        "doi": "10.5285/e5854431-6fa4-4ff0-aa02-3de68763c952",
        "licence": "UK Open Government Licence v3.0",
        "t_origin": str(t_origin),
        "t_max": str(t_max),
        "max_step_k": max_step,
        "total_capture_rows": len(parsed_rows),
        "untreated_capture_rows": n_untreated_rows,
        "treated_capture_rows": n_treated_rows,
        "species_counts": species_counts,
        "grid_treat_by_site": {k: sorted(v) for k, v in sorted(grid_treat_by_site.items())},
        "site_series": site_series,
        "weather_steps": weather_steps,
    }


def load_kielder_2008_2010(weather_bundle: dict[str, Any]) -> dict[str, Any]:
    """Load Kielder Forest 2008-2010 dataset (`K08`, Jackson et al. 2014) and verify Arriero 2018 subset."""
    xls_path = os.path.join(RAW_VOLES, "kielder_jackson_2008_2010", "Longitudinal database.xls")
    wb = read_biff8_workbook(xls_path)
    rows = wb["TS1-18"]
    hdr = [str(x).strip() if x is not None else "" for x in rows[0]]
    idx = {name: i for i, name in enumerate(hdr)}

    parsed = []
    all_dates = []
    for rw in rows[1:]:
        if not rw or all(x is None for x in rw):
            continue
        site = str(rw[idx["Site"]]).strip() if rw[idx["Site"]] is not None else ""
        dt = _parse_uk_or_iso_date(rw[idx["Date"]])
        yr = _parse_float(rw[idx["Year"]])
        ts = _parse_float(rw[idx["Trapping Session"]])
        if not site or site == "*" or dt is None or yr is None or ts is None:
            continue
        tag_raw = rw[idx["Tag ID"]]
        tag = str(int(tag_raw)) if isinstance(tag_raw, float) and tag_raw.is_integer() else (str(tag_raw).strip() if tag_raw is not None else "")
        if not tag or tag == "*":
            tag = f"untagged_row_{rw[idx['ID']]}"
        all_dates.append(dt)

        def _p01(col: str) -> int | None:
            v = _parse_float(rw[idx[col]])
            if v in (0.0, 1.0):
                return int(v)
            return None

        parsed.append({
            "site": site,
            "year": int(round(yr)),
            "ts": int(round(ts)),
            "date": dt,
            "vid": tag,
            "bartonella": _p01("Bartonella"),
            "cowpox": _p01("Cowpox"),
            "babesia": _p01("Babesia"),
            "fleas": _parse_float(rw[idx["Fleas"]]),
            "ticks": _parse_float(rw[idx["Ticks"]]),
        })

    t_origin = min(all_dates)
    t_max = max(all_dates)
    max_step = int(round((t_max - t_origin).days / 28.0)) + 1

    # Map each primary (site, year, ts) session to its mean capture date's 28-day step k (protocol §4.1)
    by_sess: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in parsed:
        by_sess.setdefault((r["site"], r["year"], r["ts"]), []).append(r)

    by_site_step: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sess_count_by_site_step: dict[tuple[str, int], int] = {}
    for (site, yr, ts), srows in by_sess.items():
        mean_ord = float(np.mean([x["date"].toordinal() for x in srows]))
        mean_dt = datetime.date.fromordinal(int(round(mean_ord)))
        k = int(round((mean_dt - t_origin).days / 28.0))
        by_site_step.setdefault((site, k), []).extend(srows)
        sess_count_by_site_step[(site, k)] = sess_count_by_site_step.get((site, k), 0) + 1

    weather_steps = {
        k: window_weather_28d(weather_bundle["kielder_daily"], "kielder", t_origin, k)
        for k in range(-2, max_step + 2)
    }

    sites = sorted({r["site"] for r in parsed})
    site_series: dict[str, list[dict[str, Any]]] = {}
    for s in sites:
        windows = []
        for k in sorted({kk for (st, kk) in by_site_step if st == s}):
            rsk = by_site_step[(s, k)]
            n_sess = sess_count_by_site_step[(s, k)]
            by_v: dict[str, list[dict[str, Any]]] = {}
            for x in rsk:
                by_v.setdefault(x["vid"], []).append(x)
            h_voles = len(by_v) / float(n_sess)
            # 150-trap longitudinal grids in Jackson et al. 2014
            h_per_100 = 100.0 * h_voles / 150.0

            bart_v, cow_v, bab_v, fl_v, tk_v = [], [], [], [], []
            bart_f, cow_f, bab_f = [], [], []
            for vid, vrows in by_v.items():
                vrows_s = sorted(vrows, key=lambda z: z["date"])
                bs = [z["bartonella"] for z in vrows_s if z["bartonella"] is not None]
                cs = [z["cowpox"] for z in vrows_s if z["cowpox"] is not None]
                bbs = [z["babesia"] for z in vrows_s if z["babesia"] is not None]
                fls = [z["fleas"] for z in vrows_s if z["fleas"] is not None]
                tks = [z["ticks"] for z in vrows_s if z["ticks"] is not None]
                if bs:
                    bart_v.append(max(bs))
                    bart_f.append(bs[0])
                if cs:
                    cow_v.append(max(cs))
                    cow_f.append(cs[0])
                if bbs:
                    bab_v.append(max(bbs))
                    bab_f.append(bbs[0])
                if fls:
                    fl_v.append(float(np.mean(fls)))
                if tks:
                    tk_v.append(float(np.mean(tks)))

            mean_ord = float(np.mean([x["date"].toordinal() for x in rsk]))
            mean_dt = datetime.date.fromordinal(int(round(mean_ord)))
            yrs_present = sorted({x["year"] for x in rsk})
            w_cur = weather_steps[k]
            w_prev = weather_steps[k - 1]
            windows.append({
                "site": s,
                "years": yrs_present,
                "step_k": k,
                "mean_date": str(mean_dt),
                "n_sessions_pooled": n_sess,
                "n_distinct_voles": len(by_v),
                "H_voles_per_session": h_voles,
                "H_per_100_traps": h_per_100,
                "bar_pos": int(sum(bart_v)),
                "bar_n": len(bart_v),
                "bar_first_pos": int(sum(bart_f)),
                "cow_pos": int(sum(cow_v)),
                "cow_n": len(cow_v),
                "cow_first_pos": int(sum(cow_f)),
                "bab_pos": int(sum(bab_v)),
                "bab_n": len(bab_v),
                "bab_first_pos": int(sum(bab_f)),
                "mean_fleas": float(np.mean(fl_v)) if fl_v else float("nan"),
                "mean_ticks": float(np.mean(tk_v)) if tk_v else float("nan"),
                "T_28d_C": w_cur["T_28d_C"],
                "P_28d_mm": w_cur["P_28d_mm"],
                "RH_28d_pct": w_cur["RH_28d_pct"],
                "T_prev28d_C": w_prev["T_28d_C"],
                "P_prev28d_mm": w_prev["P_28d_mm"],
            })
        site_series[s] = windows

    # Cross-check Arriero et al. (2018) xlsx (Zenodo 4941512) tag overlap with K08
    arr_path = os.path.join(RAW_VOLES, "kielder_arriero_2018", "vole_immunological_profiles_data.xlsx")
    arr_tags = set()
    n_arr_rows = 0
    with zipfile.ZipFile(arr_path) as zf:
        sst = [
            "".join(si.itertext())
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"
            )
        ]
        s1 = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        xrows = list(s1.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"))
        for xrw in xrows[1:]:
            vals = []
            for c in xrw:
                t = c.attrib.get("t")
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = sst[int(v.text)] if (t == "s" and v is not None) else (v.text if v is not None else None)
                vals.append(val)
            if vals and vals[0] is not None:
                n_arr_rows += 1
                tg = str(vals[0]).strip()
                if tg.endswith(".0"):
                    tg = tg[:-2]
                arr_tags.add(tg.replace(" ", "").lstrip("0"))
    k08_tags = {r["vid"].replace(" ", "").lstrip("0") for r in parsed}
    overlap_tags = len(arr_tags & k08_tags)

    # Also count Cross-sectional database.xls rows for completeness reporting
    cs_wb = read_biff8_workbook(os.path.join(RAW_VOLES, "kielder_jackson_2008_2010", "Cross-sectional database.xls"))
    n_cs_rows = sum(max(0, len(sh_rows) - 1) for sh_rows in cs_wb.values())

    return {
        "dataset_id": "K08",
        "doi": "10.5061/dryad.bk537 (Zenodo mirror 4971362) + 10.5061/dryad.110pk (Zenodo mirror 4941512)",
        "licence": "CC0 1.0",
        "t_origin": str(t_origin),
        "t_max": str(t_max),
        "total_longitudinal_rows": len(parsed),
        "cross_sectional_workbook_rows": n_cs_rows,
        "arriero_2018_rows": n_arr_rows,
        "arriero_distinct_tags": len(arr_tags),
        "arriero_tags_in_k08_longitudinal": overlap_tags,
        "site_series": site_series,
        "weather_steps": weather_steps,
    }


def load_finland_puumala(weather_bundle: dict[str, Any]) -> dict[str, Any]:
    """Load Konnevesi 2002-2009 60-session bank vole & PUUV series (`F02`) and 30-area `F23`."""
    prev_path = os.path.join(RAW_VOLES, "finland_puumala_2002_2009", "PUUV prevalence per trapping session.xls")
    sc_path = os.path.join(RAW_VOLES, "finland_puumala_2002_2009", "Seroconversion rate per trapping session.xls")

    wb_p = read_biff8_workbook(prev_path)
    prows = wb_p["PUUV prev per trapping session"]
    phdr = [str(x).strip() if x is not None else "" for x in prows[0]]
    pidx = {name: i for i, name in enumerate(phdr)}

    core_raw = []
    sat_by_date: dict[datetime.date, list[dict[str, Any]]] = {}
    for rw in prows[1:]:
        if not rw or all(x is None for x in rw):
            continue
        dt = _parse_uk_or_iso_date(rw[pidx["Date"]])
        site = str(rw[pidx["site"]]).strip() if rw[pidx["site"]] is not None else ""
        if dt is None or not site:
            continue
        tot_n = _parse_float(rw[pidx["Total N"]])
        tn = _parse_float(rw[pidx["trapnights"]])
        tested = _parse_float(rw[pidx["PUUV tested"]])
        inf_p = _parse_float(rw[pidx["Infected.P"]])
        cyc_phase = str(rw[pidx["cycle phase"]]).strip() if rw[pidx["cycle phase"]] is not None else ""
        rec = {
            "date": dt,
            "site": site,
            "cycle_phase": cyc_phase,
            "total_N": tot_n if tot_n is not None else 0.0,
            "trapnights": tn if (tn is not None and tn > 0) else float("nan"),
            "tested": int(round(tested)) if tested is not None else 0,
            "pos": int(round(inf_p)) if inf_p is not None else 0,
        }
        if site == "core":
            core_raw.append(rec)
        else:
            sat_by_date.setdefault(dt, []).append(rec)

    core_raw.sort(key=lambda x: x["date"])
    t_origin = core_raw[0]["date"]
    t_max = core_raw[-1]["date"]
    max_step = int(round((t_max - t_origin).days / 28.0)) + 2

    weather_steps = {
        k: window_weather_28d(weather_bundle["konnevesi_daily"], "konnevesi", t_origin, k)
        for k in range(-2, max_step + 2)
    }

    # Pool onto integer 28-day step k = round((date - t_origin)/28) per protocol §4.1
    by_k_core: dict[int, list[dict[str, Any]]] = {}
    for r in core_raw:
        k = int(round((r["date"] - t_origin).days / 28.0))
        r["step_k"] = k
        r["t_exact_28d"] = (r["date"] - t_origin).days / 28.0
        by_k_core.setdefault(k, []).append(r)

    core_windows = []
    for k in sorted(by_k_core):
        rk = by_k_core[k]
        tot_n_sum = sum(x["total_N"] for x in rk)
        tn_sum = sum(x["trapnights"] for x in rk if not math.isnan(x["trapnights"]))
        h_per_100 = 100.0 * tot_n_sum / tn_sum if tn_sum > 0 else float("nan")
        h_raw_mean = float(np.mean([x["total_N"] for x in rk]))
        pos_sum = sum(x["pos"] for x in rk)
        test_sum = sum(x["tested"] for x in rk)
        mean_ord = float(np.mean([x["date"].toordinal() for x in rk]))
        mean_dt = datetime.date.fromordinal(int(round(mean_ord)))
        t_ex = float(np.mean([x["t_exact_28d"] for x in rk]))
        w_cur = weather_steps[k]
        w_prev = weather_steps[k - 1]
        core_windows.append({
            "site": "core",
            "step_k": k,
            "t_exact_28d": round(t_ex, 4),
            "mean_date": str(mean_dt),
            "n_sessions_pooled": len(rk),
            "cycle_phase": rk[0]["cycle_phase"],
            "total_N_mean": h_raw_mean,
            "trapnights_sum": tn_sum,
            "H_per_100_trapnights": h_per_100,
            "puuv_pos": pos_sum,
            "puuv_n": test_sum,
            "T_28d_C": w_cur["T_28d_C"],
            "P_28d_mm": w_cur["P_28d_mm"],
            "RH_28d_pct": w_cur["RH_28d_pct"],
            "frost_frac_28d": w_cur["frost_frac_28d"],
            "snow_frac_28d": w_cur["snow_frac_28d"],
            "T_prev28d_C": w_prev["T_28d_C"],
            "P_prev28d_mm": w_prev["P_28d_mm"],
        })

    # Satellite pooled series (15 satellite grids A-P, 9 traps each)
    by_k_sat: dict[int, list[dict[str, Any]]] = {}
    for dt, srows in sorted(sat_by_date.items()):
        k = int(round((dt - t_origin).days / 28.0))
        by_k_sat.setdefault(k, []).extend(srows)
    sat_windows = []
    for k in sorted(by_k_sat):
        rk = by_k_sat[k]
        tot_n = sum(x["total_N"] for x in rk)
        tn = sum(x["trapnights"] for x in rk if not math.isnan(x["trapnights"]))
        pos_s = sum(x["pos"] for x in rk)
        test_s = sum(x["tested"] for x in rk)
        mean_ord = float(np.mean([x["date"].toordinal() for x in rk]))
        mean_dt = datetime.date.fromordinal(int(round(mean_ord)))
        w_cur = weather_steps.get(k, window_weather_28d(weather_bundle["konnevesi_daily"], "konnevesi", t_origin, k))
        sat_windows.append({
            "site": "satellite_pooled",
            "step_k": k,
            "mean_date": str(mean_dt),
            "n_site_sessions": len(rk),
            "total_N_sum": tot_n,
            "trapnights_sum": tn,
            "H_per_100_trapnights": 100.0 * tot_n / tn if tn > 0 else float("nan"),
            "puuv_pos": pos_s,
            "puuv_n": test_s,
            "T_28d_C": w_cur["T_28d_C"],
            "P_28d_mm": w_cur["P_28d_mm"],
            "snow_frac_28d": w_cur["snow_frac_28d"],
        })

    # Seroconversion intervals on core grid
    wb_sc = read_biff8_workbook(sc_path)
    sc_rows = wb_sc["Sc rates per trapping session"]
    schdr = [str(x).strip() if x is not None else "" for x in sc_rows[0]]
    scidx = {name: i for i, name in enumerate(schdr)}
    # Find flexible column names for exposure days and seroconverted
    col_sc_n = next(c for c in schdr if "Number seroconverted" in c)
    col_exp = next(c for c in schdr if "Days of exposure" in c)
    col_mid = next(c for c in schdr if "between" in c)
    sc_series = []
    for rw in sc_rows[1:]:
        if not rw or all(x is None for x in rw):
            continue
        dt_mid = _parse_uk_or_iso_date(rw[scidx[col_mid]])
        intv = _parse_float(rw[scidx["interval"]])
        n_sc = _parse_float(rw[scidx[col_sc_n]])
        exp_d = _parse_float(rw[scidx[col_exp]])
        rate = _parse_float(rw[scidx["Seroconversion rate"]])
        if dt_mid is None or exp_d is None or exp_d <= 0:
            continue
        k_mid = int(round((dt_mid - t_origin).days / 28.0))
        sc_series.append({
            "step_k": k_mid,
            "mid_date": str(dt_mid),
            "interval_days": intv,
            "n_seroconverted": n_sc if n_sc is not None else 0.0,
            "exposure_days": exp_d,
            "rate_per_28d": 28.0 * (rate if rate is not None else (n_sc / exp_d)),
        })

    # Load F23 (Wang et al. 2023, 30 areas annual comparator with FMI 5km climate)
    f23_path = os.path.join(RAW_VOLES, "finland_puumala_wang_2023", "Data_PUUV_in_bank_vole_Human_NE.csv")
    f23_rows = []
    with open(f23_path, "r", encoding="utf-8-sig", errors="replace") as f:
        r23 = csv.DictReader(f)
        for row in r23:
            area = (row.get("Area_no") or "").strip()
            yr = _parse_float(row.get("year"))
            mg = _parse_float(row.get("Myodes_glareolus"))
            ma = _parse_float(row.get("Microtus_agrestis"))
            n_scr = _parse_float(row.get("N_screened"))
            n_pos = _parse_float(row.get("N_screened__positive"))
            if not area or yr is None:
                continue
            f23_rows.append({
                "area": area,
                "year": int(round(yr)),
                "Myodes_glareolus": mg,
                "Microtus_agrestis": ma,
                "n_screened": int(round(n_scr)) if n_scr is not None else 0,
                "n_pos": int(round(n_pos)) if n_pos is not None else 0,
                "T_winter": _parse_float(row.get("T5KMwinter")),
                "T_spring": _parse_float(row.get("T5KMspring")),
                "T_summer": _parse_float(row.get("T5KMsummer")),
                "T_autumn": _parse_float(row.get("T5KMautumn")),
                "P_winter": _parse_float(row.get("P5KMwinter")),
                "P_summer": _parse_float(row.get("P5KMsummer")),
            })

    return {
        "dataset_id": "F02",
        "doi": "10.5061/dryad.g8140 (Zenodo mirror 4941743) + F23 10.5061/dryad.bg79cnpfh (Zenodo mirror 7796298)",
        "licence": "CC0 1.0",
        "t_origin": str(t_origin),
        "t_max": str(t_max),
        "n_core_raw_sessions": len(core_raw),
        "n_core_28d_windows": len(core_windows),
        "core_windows": core_windows,
        "satellite_windows": sat_windows,
        "seroconversion_series": sc_series,
        "f23_annual_rows": f23_rows,
        "weather_steps": weather_steps,
    }


def load_all_voles_data() -> dict[str, Any]:
    parser_chk = verify_parsers()
    weather_bundle = build_regional_daily_weather()
    k15 = load_kielder_2015_2017(weather_bundle)
    k08 = load_kielder_2008_2010(weather_bundle)
    f02 = load_finland_puumala(weather_bundle)
    return {
        "parser_verification": parser_chk,
        "weather_provenance": weather_bundle["provenance"],
        "kielder_daily_tuple": weather_bundle["kielder_daily"],
        "konnevesi_daily_tuple": weather_bundle["konnevesi_daily"],
        "K15": k15,
        "K08": k08,
        "F02": f02,
    }
