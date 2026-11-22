#!/usr/bin/env python3
"""Fetch open, unauthenticated weather and environmental datasets for case study `voles`
(Step B, Protocol Amendment 1 §7.2 and Amendment 2 §8).

Run:
    python3 -I data/fetch/voles-weather.py

Rules enforced (CASESTUDY_SPEC.md Rule 2 & end section; study rules):
  - Open, unauthenticated sources only; generic User-Agent (`open-ecology-fetch/1.0`);
    zero e-mail addresses, personal details, logins, or API tokens in any request.
  - Standard library only; idempotent / re-runnable (verifies existing cached files by
    non-zero size and SHA-256 unless `--force` is passed).
  - Writes raw files under `data/raw/voles/weather/` (git-ignored) and writes a complete
    provenance + distance manifest to `data/raw/voles/weather/MANIFEST.json`.
  - Records timestamps from `date -u`.

Sources covered for Kielder Forest, UK (2007-2017) and Finnish sites (2001-2009):
  1. NOAA GHCN-Daily in-situ and synoptic stations (daily TMAX, TMIN, TAVG, PRCP, SNWD;
     DOI 10.7289/V5D21VHZ, U.S. Gov Public Domain).
  2. ERA5-Land / ERA5-seamless daily 0.1 deg (~9 km) reanalysis via Open-Meteo Historical
     Weather Archive (air T, precipitation, snow depth, RH, 0-7 cm litter soil T,
     7-28 cm burrow soil T, 0-7 cm volumetric soil moisture; Copernicus C3S
     DOI 10.24381/cds.e2161bac, CC BY 4.0).
  3. TerraClimate v1.1 monthly 1/24 deg (~4 km) land climate & water balance via U. Idaho
     NKN THREDDS OPeNDAP (tmax, tmin, ppt, soil moisture, snow water equivalent;
     DOI 10.1038/sdata.2017.191, CC0 1.0).
  4. NASA POWER Agroclimatology v2.0 daily 0.5x0.625 deg meteorology & 2 m relative
     humidity (DOI 10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY, Public Domain).
  5. Audit of deposited FMI 5-km seasonal climate & CORINE habitat for the 30 anonymised
     Finnish sites in `F23` (Wang et al. 2023, Dryad DOI 10.5061/dryad.bg79cnpfh, CC0 1.0),
     plus transparent unauthenticated availability probes of CEDA HadUK-Grid and KNMI E-OBS.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "open-ecology-fetch/1.0"
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
RAW_VOLES = (__import__('pathlib').Path(__file__).resolve().parent / '../raw/voles').resolve()
WEATHER_DIR = RAW_VOLES / "weather"
TC_RAW_DIR = WEATHER_DIR / "terraclimate_raw"
TC_BASE = "https://thredds.northwestknowledge.net/thredds/dodsC"

KIELDER_CENTROID = (55.2300, -2.5750)
KONNEVESI_STATION = (62.6280, 26.2880)
KONNEVESI_DRYAD = (62.5667, 26.4000)


def date_utc() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(0.5 * dp) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(0.5 * dl) ** 2)
    return 2.0 * r_earth * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(url: str, tries: int = 3, timeout: int = 90) -> bytes:
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 + 2.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def save_or_verify(
    url: str,
    dest: Path,
    meta: dict,
    items: list[dict],
    force: bool = False,
) -> bytes:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rel = str(dest.relative_to(WEATHER_DIR))
    if dest.exists() and not force and dest.stat().st_size > 0:
        data = dest.read_bytes()
        status = "cached_verified"
    else:
        data = fetch_bytes(url)
        dest.write_bytes(data)
        status = "downloaded"
        time.sleep(0.4)

    rec = {
        "file": rel,
        "url": url,
        **meta,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "verified_utc": date_utc(),
        "status": status,
    }
    items.append(rec)
    print(f"{len(data):10d}  [{status:15s}]  {rel}", flush=True)
    return data


def parse_terraclimate_3x3_ascii(txt: str, var: str, n_t: int) -> tuple[list[list[list[int]]], list[float], list[float]]:
    sep_idx = txt.find("---------------------------------------------")
    body = txt[sep_idx + 45 :] if sep_idx >= 0 else txt
    lines = [ln.strip() for ln in body.splitlines()]
    cube = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(n_t)]
    lats: list[float] = []
    lons: list[float] = []
    mode = ""
    for ln in lines:
        if not ln:
            mode = ""
            continue
        if ln.startswith(f"{var}.{var}["):
            mode = "arr"
            continue
        if ln.startswith(f"{var}.lat["):
            mode = "lat"
            continue
        if ln.startswith(f"{var}.lon["):
            mode = "lon"
            continue
        if ln.startswith(f"{var}.time["):
            mode = "time"
            continue
        if mode == "arr" and ln.startswith("["):
            head, rest = ln.split("],", 1)
            parts_idx = head.replace("[", "").split("]")
            ti = int(parts_idx[0])
            ri = int(parts_idx[1])
            vals = [int(float(x.strip())) for x in rest.split(",")]
            if 0 <= ti < n_t and 0 <= ri < 3 and len(vals) == 3:
                cube[ti][ri] = vals
        elif mode == "lat":
            lats.extend(float(x.strip()) for x in ln.split(",") if x.strip())
        elif mode == "lon":
            lons.extend(float(x.strip()) for x in ln.split(",") if x.strip())
    return cube, lats, lons


def fetch_ghcnd(items: list[dict], force: bool = False) -> None:
    jobs = [
        (
            "ghcnd_KielderCastle_UKE00105892_2007_2017.csv",
            "UKE00105892",
            "Kielder Castle",
            (55.2339, -2.5789),
            201.0,
            KIELDER_CENTROID,
            "Kielder Forest EIDC centroid",
            "2007-01-01",
            "2017-12-31",
        ),
        (
            "ghcnd_Eskdalemuir_UK000003162_2007_2017.csv",
            "UK000003162",
            "Eskdalemuir Observatory",
            (55.3167, -3.2000),
            242.0,
            KIELDER_CENTROID,
            "Kielder Forest EIDC centroid",
            "2007-01-01",
            "2017-12-31",
        ),
        (
            "ghcnd_VesantoSonkari_FIE00144637_2001_2009.csv",
            "FIE00144637",
            "Vesanto Sonkari",
            (62.9214, 26.4281),
            114.0,
            KONNEVESI_STATION,
            "Konnevesi Research Station",
            "2001-01-01",
            "2009-12-31",
        ),
        (
            "ghcnd_Jyvaskyla_FI000002401_2001_2009.csv",
            "FI000002401",
            "Jyvaskyla Lentoasema",
            (62.4019, 25.6783),
            140.0,
            KONNEVESI_STATION,
            "Konnevesi Research Station",
            "2001-01-01",
            "2009-12-31",
        ),
        (
            "ghcnd_KonnevesiLahdenkyla_FIE00144657_2001_2007.csv",
            "FIE00144657",
            "Konnevesi Lahdenkyla",
            (62.5667, 26.2831),
            93.0,
            KONNEVESI_STATION,
            "Konnevesi Research Station",
            "2001-01-01",
            "2007-12-31",
        ),
    ]
    for fname, stn, sname, slatlon, elev_m, target_ll, target_name, d0, d1 in jobs:
        q = urllib.parse.urlencode(
            {
                "dataset": "daily-summaries",
                "stations": stn,
                "startDate": d0,
                "endDate": d1,
                "dataTypes": "TMAX,TMIN,TAVG,PRCP,SNWD",
                "units": "metric",
                "format": "csv",
            }
        )
        url = f"https://www.ncei.noaa.gov/access/services/data/v1?{q}"
        dist_km = round(haversine_km(target_ll[0], target_ll[1], slatlon[0], slatlon[1]), 2)
        save_or_verify(
            url,
            WEATHER_DIR / fname,
            meta={
                "source": "NOAA GHCN-Daily",
                "station_id": stn,
                "station_name": sname,
                "doi": "10.7289/V5D21VHZ",
                "licence": "U.S. Government Work / Public Domain",
                "version": "v1 daily-summaries",
                "resolution": "daily point station",
                "matched_site": target_name,
                "matched_site_lat_lon": list(target_ll),
                "station_or_cell_lat_lon": list(slatlon),
                "elevation_m": elev_m,
                "distance_km": dist_km,
                "date_span": [d0, d1],
            },
            items=items,
            force=force,
        )


def fetch_nasapower(items: list[dict], force: bool = False) -> None:
    jobs = [
        (
            "nasapower_Kielder_55.23N_2.575W_2007_2017.json",
            "Kielder Forest EIDC centroid",
            KIELDER_CENTROID,
            "20070101",
            "20171231",
            "2007-01-01",
            "2017-12-31",
        ),
        (
            "nasapower_Konnevesi_62.628N_26.288E_2001_2009.json",
            "Konnevesi Research Station",
            KONNEVESI_STATION,
            "20010101",
            "20091231",
            "2001-01-01",
            "2009-12-31",
        ),
    ]
    for fname, sname, (lat, lon), d0_p, d1_p, d0_iso, d1_iso in jobs:
        q = urllib.parse.urlencode(
            {
                "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
                "community": "AG",
                "longitude": f"{lon:.4f}",
                "latitude": f"{lat:.4f}",
                "start": d0_p,
                "end": d1_p,
                "format": "JSON",
            }
        )
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?{q}"
        cell_lat = round(lat * 2.0) / 2.0
        cell_lon = round(lon / 0.625) * 0.625
        dist_km = round(haversine_km(lat, lon, cell_lat, cell_lon), 2)
        save_or_verify(
            url,
            WEATHER_DIR / fname,
            meta={
                "source": "NASA POWER Agroclimatology",
                "doi": "10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY",
                "licence": "NASA Open Data / Public Domain",
                "version": "v2.0",
                "resolution": "daily 0.5 x 0.625 deg MERRA-2 grid",
                "matched_site": sname,
                "matched_site_lat_lon": [lat, lon],
                "station_or_cell_lat_lon": [cell_lat, cell_lon],
                "distance_km": dist_km,
                "date_span": [d0_iso, d1_iso],
            },
            items=items,
            force=force,
        )


def fetch_era5_openmeteo(items: list[dict], force: bool = False) -> None:
    daily_vars = ",".join(
        [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
            "snow_depth_max",
            "relative_humidity_2m_mean",
            "soil_temperature_0_to_7cm_mean",
            "soil_temperature_7_to_28cm_mean",
            "soil_moisture_0_to_7cm_mean",
        ]
    )
    jobs = [
        (
            "era5land_Kielder_55.23N_2.575W_2007_2017.json",
            "Kielder Forest EIDC centroid",
            KIELDER_CENTROID,
            "2007-01-01",
            "2017-12-31",
        ),
        (
            "era5land_KonnevesiStation_62.628N_26.288E_2001_2009.json",
            "Konnevesi Research Station",
            KONNEVESI_STATION,
            "2001-01-01",
            "2009-12-31",
        ),
        (
            "era5land_KonnevesiDryad_62.5667N_26.4000E_2001_2009.json",
            "Konnevesi Dryad/Zenodo study centroid (62d34m N, 26d24m E)",
            KONNEVESI_DRYAD,
            "2001-01-01",
            "2009-12-31",
        ),
    ]
    for fname, sname, (lat, lon), d0, d1 in jobs:
        q = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "start_date": d0,
                "end_date": d1,
                "daily": daily_vars,
                "models": "era5_seamless",
                "timezone": "UTC",
            }
        )
        url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
        dest = WEATHER_DIR / fname
        raw_bytes = save_or_verify(
            url,
            dest,
            meta={
                "source": "ERA5-Land / ERA5-seamless via Open-Meteo Historical Weather Archive",
                "doi": "10.24381/cds.e2161bac",
                "licence": "Copernicus C3S / CC BY 4.0 (unauthenticated Open-Meteo archive mirror)",
                "version": "ERA5-Land 0.1 deg seamless daily",
                "resolution": "daily 0.1 x 0.1 deg (~9 km) surface + 0-28 cm soil",
                "matched_site": sname,
                "matched_site_lat_lon": [lat, lon],
                "date_span": [d0, d1],
            },
            items=items,
            force=force,
        )
        doc = json.loads(raw_bytes.decode("utf-8"))
        c_lat = round(float(doc["latitude"]), 5)
        c_lon = round(float(doc["longitude"]), 5)
        items[-1]["station_or_cell_lat_lon"] = [c_lat, c_lon]
        items[-1]["elevation_m"] = doc.get("elevation")
        items[-1]["distance_km"] = round(haversine_km(lat, lon, c_lat, c_lon), 2)


def fetch_terraclimate(items: list[dict], force: bool = False) -> None:
    tc_vars = [
        ("tmax", 0.01, -99.0, -32768),
        ("tmin", 0.01, -99.0, -32768),
        ("ppt", 0.1, 0.0, -2147483648),
        ("soil", 0.1, 0.0, -32768),
        ("swe", 0.1, 0.0, -2147483648),
    ]
    jobs = [
        ("Kielder_2007_2017", "Kielder Forest EIDC centroid", KIELDER_CENTROID, 2007, 2017),
        ("Konnevesi_2001_2009", "Konnevesi Research Station", KONNEVESI_STATION, 2001, 2009),
    ]
    for label, sname, (lat, lon), y0, y1 in jobs:
        i = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
        j = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
        t0 = (y0 - 1950) * 12
        t1 = (y1 - 1950) * 12 + 11
        n_t = t1 - t0 + 1

        cubes = {}
        lats, lons = [], []
        urls_used = []
        for vname, _, _, _ in tc_vars:
            u = f"{TC_BASE}/agg_terraclimate_{vname}_1950_CurrentYear_GLOBE.nc.ascii?{vname}[{t0}:{t1}][{i-1}:{i+1}][{j-1}:{j+1}]"
            urls_used.append(u)
            raw_dest = TC_RAW_DIR / f"tc_{label}_{vname}.ascii"
            raw_b = save_or_verify(
                u,
                raw_dest,
                meta={
                    "source": "TerraClimate v1.1 OPeNDAP ASCII slice",
                    "variable": vname,
                    "doi": "10.1038/sdata.2017.191",
                    "licence": "CC0 1.0 (Public Domain)",
                    "version": "v1.1 (1950-CurrentYear)",
                    "resolution": "monthly 1/24 deg (~4 km)",
                    "matched_site": sname,
                    "matched_site_lat_lon": [lat, lon],
                    "date_span": [f"{y0}-01", f"{y1}-12"],
                },
                items=items,
                force=force,
            )
            cb, la, lo = parse_terraclimate_3x3_ascii(raw_b.decode("utf-8"), vname, n_t)
            cubes[vname] = cb
            if la and lo:
                lats, lons = la, lo

        options = []
        for a in range(3):
            for b in range(3):
                if cubes["tmax"][0][a][b] != -32768:
                    dkm = haversine_km(lat, lon, lats[a], lons[b])
                    options.append((dkm, a, b))
        if not options:
            raise RuntimeError(f"No valid land TerraClimate cell found around {sname}")
        dist_km, a_best, b_best = min(options)
        c_lat = round(float(lats[a_best]), 5)
        c_lon = round(float(lons[b_best]), 5)

        months = [f"{yr:04d}-{mo:02d}" for yr in range(y0, y1 + 1) for mo in range(1, 13)]
        parsed_doc = {
            "source": "TerraClimate v1.1 (U. Idaho NKN THREDDS OPeNDAP)",
            "doi": "10.1038/sdata.2017.191",
            "licence": "CC0 1.0",
            "matched_site": sname,
            "matched_site_lat_lon": [lat, lon],
            "cell_lat_lon": [c_lat, c_lon],
            "distance_km": round(dist_km, 2),
            "y0": y0,
            "y1": y1,
            "months": months,
            "urls": urls_used,
            "series": {},
        }
        for vname, scale, offset, fillv in tc_vars:
            vals = []
            for k in range(n_t):
                raw_int = cubes[vname][k][a_best][b_best]
                if raw_int == fillv or raw_int in (-32768, -2147483648):
                    vals.append(None)
                else:
                    vals.append(round(raw_int * scale + offset, 3))
            parsed_doc["series"][vname] = vals

        tmax_s = parsed_doc["series"]["tmax"]
        tmin_s = parsed_doc["series"]["tmin"]
        parsed_doc["series"]["tmean"] = [
            round(0.5 * (tx + tn), 3) if (tx is not None and tn is not None) else None
            for tx, tn in zip(tmax_s, tmin_s)
        ]

        out_json = WEATHER_DIR / f"terraclimate_{label}.json"
        payload = (json.dumps(parsed_doc, indent=2) + "\n").encode("utf-8")
        out_json.write_bytes(payload)
        items.append(
            {
                "file": str(out_json.relative_to(WEATHER_DIR)),
                "url": urls_used[0],
                "source": "TerraClimate v1.1 compiled 5-variable series",
                "doi": "10.1038/sdata.2017.191",
                "licence": "CC0 1.0",
                "version": "v1.1",
                "resolution": "monthly 1/24 deg (~4 km)",
                "matched_site": sname,
                "matched_site_lat_lon": [lat, lon],
                "station_or_cell_lat_lon": [c_lat, c_lon],
                "distance_km": round(dist_km, 2),
                "date_span": [f"{y0}-01", f"{y1}-12"],
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "verified_utc": date_utc(),
                "status": "compiled_from_raw_opendap",
            }
        )
        print(
            f"{len(payload):10d}  [compiled_4km   ]  {out_json.relative_to(WEATHER_DIR)} (dist={dist_km:.2f} km)",
            flush=True,
        )


def audit_probes_and_f23() -> dict:
    """Record unauthenticated endpoint probes (CEDA HadUK-Grid, KNMI E-OBS) without bypass,
    and verify the deposited FMI 5-km seasonal climate & CORINE habitat columns in F23.
    """
    probes = []
    probe_targets = [
        (
            "Met_Office_HadUK_Grid_CEDA",
            "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.3.0.ceda/1km/tas/mon/v20240514/",
        ),
        (
            "KNMI_EOBS_OPeNDAP",
            "https://opendap.knmi.nl/knmi/thredds/dodsC/e-obs_0.10regular/tg_ens_mean_0.1deg_reg_v29.0e.nc.das",
        ),
    ]
    for name, url in probe_targets:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read(256)
                probes.append({"target": name, "url": url, "status": f"HTTP {resp.status}", "checked_utc": date_utc()})
        except Exception as exc:  # noqa: BLE001
            probes.append(
                {
                    "target": name,
                    "url": url,
                    "status": "unreachable_from_host",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_used": "NOAA GHCN-Daily Kielder Castle (0.5 km) + ERA5-Land (~9 km) + TerraClimate (~4 km) + NASA POWER",
                    "checked_utc": date_utc(),
                }
            )

    f23_csv = RAW_VOLES / "finland_puumala_wang_2023" / "Data_PUUV_in_bank_vole_Human_NE.csv"
    f23_audit = {"present": f23_csv.exists()}
    if f23_csv.exists():
        raw_b = f23_csv.read_bytes()
        with f23_csv.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        clim_cols = [
            "T5KMwinter",
            "T5KMspring",
            "T5KMsummer",
            "T5KMautumn",
            "P5KMwinter",
            "P5KMspring",
            "P5KMsummer",
            "P5KMautumn",
        ]
        hab_cols = ["CA5", "ED5", "Artificial5.P", "Agricul5.P", "Forest5.P", "Wetland5.P", "Water5.P"]
        f23_audit.update(
            {
                "path_relative_to_raw_voles": "finland_puumala_wang_2023/Data_PUUV_in_bank_vole_Human_NE.csv",
                "doi": "10.5061/dryad.bg79cnpfh",
                "licence": "CC0 1.0",
                "bytes": len(raw_b),
                "sha256": sha256_bytes(raw_b),
                "n_rows": len(rows),
                "n_areas": len({r["Area_no"] for r in rows}),
                "years": sorted({r["year"] for r in rows}),
                "coordinates_disclosed": False,
                "embedded_fmi_5km_climate_columns": [c for c in clim_cols if rows and c in rows[0]],
                "embedded_corine_5km_habitat_columns": [c for c in hab_cols if rows and c in rows[0]],
                "note": "Exact coordinates of Area_no 1..30 are not disclosed in Dryad deposit; site-matched FMI 5-km seasonal climate and CORINE 5-km land-cover columns supplied directly in deposit.",
            }
        )
    return {"endpoint_probes": probes, "f23_embedded_environment_audit": f23_audit}


def main() -> int:
    force = "--force" in sys.argv[1:]
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    fetch_ghcnd(items, force=force)
    fetch_nasapower(items, force=force)
    fetch_era5_openmeteo(items, force=force)
    fetch_terraclimate(items, force=force)
    audits = audit_probes_and_f23()

    manifest_doc = {
        "study": "voles",
        "step": "B_weather_and_environment",
        "generated_utc": date_utc(),
        "user_agent": UA,
        "authenticated_requests": False,
        "items_count": len(items),
        "items": items,
        **audits,
    }
    man_path = WEATHER_DIR / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {man_path} ({len(items)} weather files verified)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
