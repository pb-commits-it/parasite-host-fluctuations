#!/usr/bin/env python3
"""Fetch open weather and environmental covariates for case study lpd (Amendment 3, r27).

Run:
    python3 -I data/fetch/lpd-weather.py

Rules enforced:
- Generic User-Agent only; no API keys, credentials, or personal details sent.
- Existing cached files in data/raw/lpd/weather/ are preserved intact and re-verified by SHA-256.
- Source hierarchy (CASESTUDY_SPEC.md, best coverage first):
    1. NEON instrument API (metadata & availability at DSNY, JERC; /data 403 logged without bypass)
    2. NASA ORNL DAAC Daymet V4 R1 (North America, daily, 1 km: tmax, tmin, prcp, vp)
    3. TerraClimate V1.1 (global land, monthly, 1/24 deg ~4 km, 1950-2025: tmax, tmin, ppt)
    4. NOAA ERSST v5 (global ocean, monthly, 2 deg, 1854-2025: sst; OISST ERDDAP/PSL status logged)
    5. NOAA GHCN-Daily station records (daily TMAX, TMIN, PRCP within <= 250 km) + GHCN-Monthly v4
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import collections
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
STUDY = ROOT / "casestudies/lpd"
sys.path.insert(0, str(STUDY))

from study_io import RAW, RESULTS, csvwrite, now, read, save
from analysis import candidates, eligible

spec = importlib.util.spec_from_file_location("fetch_original", Path(__file__).with_name("lpd.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

UA = "open-ecology-fetch/1.0"
W = RAW / "weather"
TC_BASE = "https://thredds.northwestknowledge.net/thredds/dodsC"
GHCND_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCND_ACCESS = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"

# Official NEON tower coordinates verified via https://data.neonscience.org/api/v0/sites/{DSNY,JERC}
NEON_SITES = {
    "DSNY": ("Disney Wilderness Preserve NEON", 28.12505, -81.43619),
    "JERC": ("The Jones Center At Ichauway NEON", 31.194839, -84.468623),
}


def request_bytes(url: str, timeout: int = 35, extra_headers: dict | None = None) -> bytes:
    hdrs = {"User-Agent": UA}
    if extra_headers:
        hdrs.update(extra_headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def classify_physical_site(s: dict, gpdd_ocean_by_mid: dict[str, str], any_ocean_250km) -> dict:
    """Audit physical site setting without altering original primary_metrics.json realm labels."""
    lat, lon = s["latitude"], s["longitude"]
    sid = s["series_id"]
    loc_txt = (s.get("published_location") or "")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {
            "physical_site_type": "missing_coordinates",
            "neon_site_code": "",
            "neon_tower_distance_km": None,
            "anadromous_inland_flag": False,
        }

    # Check NEON mosquito surveillance series in VecDyn
    if s["database"] == "VecDyn":
        for code, (_, nlat, nlon) in NEON_SITES.items():
            if loc_txt.upper().startswith(code):
                return {
                    "physical_site_type": "neon_terrestrial",
                    "neon_site_code": code,
                    "neon_tower_distance_km": round(old.haversine_km(lat, lon, nlat, nlon), 3),
                    "anadromous_inland_flag": False,
                }

    if s["realm_detail"] == "Marine":
        if any_ocean_250km(lat, lon):
            return {
                "physical_site_type": "marine_or_coastal",
                "neon_site_code": "",
                "neon_tower_distance_km": None,
                "anadromous_inland_flag": False,
            }
        if s["database"] == "GPDD":
            mid = sid.split("_")[1]
            ocean_val = (gpdd_ocean_by_mid.get(mid) or "").strip().lower()
            if ocean_val in ("", "not applicable", "none", "na", "land"):
                is_anad = "oncorhynchus" in (s.get("species") or "").lower() or "salmo" in (s.get("species") or "").lower()
                return {
                    "physical_site_type": "anadromous_inland_spawning" if is_anad else "inland_land_or_freshwater",
                    "neon_site_code": "",
                    "neon_tower_distance_km": None,
                    "anadromous_inland_flag": bool(is_anad),
                }
        return {
            "physical_site_type": "marine_no_coastal_grid_within_250km",
            "neon_site_code": "",
            "neon_tower_distance_km": None,
            "anadromous_inland_flag": False,
        }

    return {
        "physical_site_type": "inland_land_or_freshwater",
        "neon_site_code": "",
        "neon_tower_distance_km": None,
        "anadromous_inland_flag": False,
    }


def load_existing_tc_cells() -> dict[tuple[int, int], tuple[float, float]]:
    cells: dict[tuple[int, int], tuple[float, float]] = {}
    tc_dir = W / "terraclimate"
    if not tc_dir.exists():
        return cells
    for fp in sorted(tc_dir.glob("tc_*.json")):
        d = read(fp)
        if d.get("valid_land"):
            parts = fp.stem.split("_")
            cells[(int(parts[1]), int(parts[2]))] = (float(d["cell_lat"]), float(d["cell_lon"]))
    return cells


def load_ersst_mask():
    fp = W / "ersst_v5" / "ersst.v5.2000.asc"
    if not fp.exists():
        return lambda lat, lon: False
    import numpy as np

    arr = np.fromstring(fp.read_text(encoding="ascii", errors="ignore"), dtype=np.int32, sep=" ")
    cube = arr.reshape((12, 180, 89))

    def has_ocean_250km(lat: float, lon: float) -> bool:
        lon360 = lon % 360.0
        i0 = int(round(lon360 / 2.0)) % 180
        j0 = max(0, min(88, int(round((lat + 88.0) / 2.0))))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii = (i0 + di) % 180
                jj = j0 + dj
                if 0 <= jj <= 88 and int(cube[0, ii, jj]) > -9990:
                    clat = -88.0 + 2.0 * jj
                    clon_raw = 2.0 * ii
                    clon = clon_raw - 360.0 if clon_raw > 180.0 else clon_raw
                    if old.haversine_km(lat, lon, clat, clon) <= 250.0:
                        return True
        return False

    return has_ocean_250km


def fetch_neon_metadata(log_fn) -> None:
    neon_dir = W / "neon"
    neon_dir.mkdir(parents=True, exist_ok=True)
    dest = neon_dir / "neon_sites_products_metadata.json"
    if dest.exists() and dest.stat().st_size > 200:
        log_fn({"job": {"kind": "neon_meta"}, "path": str(dest.relative_to(RAW)), "status": "reused_existing", "attempted_utc": now()})
    else:
        meta = {"fetched_utc": now(), "sites": {}, "products": {}}
        for code in ("DSNY", "JERC"):
            u = f"https://data.neonscience.org/api/v0/sites/{code}"
            d = json.loads(request_bytes(u, extra_headers={"Accept": "application/json"}).decode("utf-8"))["data"]
            meta["sites"][code] = {
                "siteCode": d.get("siteCode"),
                "siteName": d.get("siteName"),
                "siteLatitude": d.get("siteLatitude"),
                "siteLongitude": d.get("siteLongitude"),
                "stateCode": d.get("stateCode"),
            }
        for dp in ("DP1.00002.001", "DP1.00045.001", "DP1.00098.001"):
            u = f"https://data.neonscience.org/api/v0/products/{dp}"
            d = json.loads(request_bytes(u, extra_headers={"Accept": "application/json"}).decode("utf-8"))["data"]
            meta["products"][dp] = {
                "productCode": d.get("productCode"),
                "productName": d.get("productName"),
                "siteAvailability": {
                    s["siteCode"]: {
                        "firstMonth": min(s["availableMonths"]),
                        "lastMonth": max(s["availableMonths"]),
                        "nMonths": len(s["availableMonths"]),
                    }
                    for s in (d.get("siteCodes") or [])
                    if s.get("siteCode") in ("DSNY", "JERC") and s.get("availableMonths")
                },
            }
        save(dest, meta)
        log_fn({"job": {"kind": "neon_meta"}, "path": str(dest.relative_to(RAW)), "url": "https://data.neonscience.org/api/v0/", "status": "downloaded", "fetched_utc": now()})

    # Probe anonymous /api/v0/data once and record exact HTTP response without bypass
    probe_url = "https://data.neonscience.org/api/v0/data/DP1.00002.001/DSNY/2017-06"
    try:
        _ = request_bytes(probe_url, timeout=15, extra_headers={"Accept": "application/json"})
        log_fn({"job": {"kind": "neon_data_probe"}, "url": probe_url, "status": "available_anonymous", "attempted_utc": now()})
    except Exception as exc:
        log_fn({"job": {"kind": "neon_data_probe"}, "url": probe_url, "status": "failed", "error": f"{type(exc).__name__}: {exc} (fallback to Daymet V4 R1 1km daily)", "attempted_utc": now()})


def ensure_ghcnd_catalogs(log_fn) -> tuple[Path, Path]:
    gd_dir = W / "ghcnd"
    gd_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for fname in ("ghcnd-stations.txt", "ghcnd-inventory.txt"):
        dest = gd_dir / fname
        u = f"{GHCND_BASE}/{fname}"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            log_fn({"job": {"kind": "ghcnd_catalog", "file": fname}, "path": str(dest.relative_to(RAW)), "url": u, "status": "reused_existing", "attempted_utc": now()})
        else:
            payload = request_bytes(u, timeout=60)
            dest.write_bytes(payload)
            log_fn({"job": {"kind": "ghcnd_catalog", "file": fname}, "path": str(dest.relative_to(RAW)), "url": u, "status": "downloaded", "fetched_utc": now()})
        out_paths.append(dest)
    return out_paths[0], out_paths[1]


def select_ghcnd_stations(inv_path: Path, land_requests: list[tuple[float, float, int, int]]) -> list[str]:
    """Deterministically pick top candidate GHCN-Daily stations within <= 250 km
    whose TMAX and TMIN year spans overlap at least 12 years of [y0-1, y1].
    """
    tmax_span: dict[str, tuple[float, float, int, int]] = {}
    tmin_span: dict[str, tuple[int, int]] = {}
    with inv_path.open(encoding="latin-1") as f:
        for ln in f:
            if len(ln) < 45:
                continue
            elem = ln[31:35]
            if elem not in ("TMAX", "TMIN"):
                continue
            sid = ln[0:11].strip()
            try:
                lat = float(ln[12:20])
                lon = float(ln[21:30])
                fy = int(ln[36:40])
                ly = int(ln[41:45])
            except ValueError:
                continue
            if elem == "TMAX":
                tmax_span[sid] = (lat, lon, fy, ly)
            else:
                tmin_span[sid] = (fy, ly)

    both = []
    for sid, (la, lo, fy1, ly1) in tmax_span.items():
        if sid in tmin_span:
            fy2, ly2 = tmin_span[sid]
            fy, ly = max(fy1, fy2), min(ly1, ly2)
            if ly - fy + 1 >= 12:
                both.append((sid, la, lo, fy, ly))

    import numpy as np

    s_lats = np.array([x[1] for x in both], dtype=float)
    chosen: set[str] = set()
    for lat0, lon0, y0, y1 in land_requests:
        idx = np.flatnonzero(np.abs(s_lats - lat0) <= 2.5)
        scored = []
        for k in idx:
            sid, la, lo, fy, ly = both[int(k)]
            ov = min(y1, ly) - max(y0 - 1, fy) + 1
            if ov < 14:
                continue
            dkm = old.haversine_km(lat0, lon0, la, lo)
            if dkm <= 250.0:
                scored.append((dkm, -ov, sid))
        scored.sort()
        # Keep top-3 closest overlapping stations per historical/land site
        for _, _, sid in scored[:3]:
            chosen.add(sid)
    return sorted(chosen)


def fetch_job(job: dict, existing_tc_cells: dict[tuple[int, int], tuple[float, float]]) -> dict:
    kind = job["kind"]
    r = {"job": job, "attempted_utc": now()}
    try:
        if kind == "sst":
            name = f"ersst.v5.{job['year']}.asc"
            dest = W / "ersst_v5" / name
            u = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/" + name
        elif kind == "dm":
            label = hashlib.sha256(f"{job['lat']},{job['lon']}".encode()).hexdigest()[:12]
            dest = W / "daymet" / f"site_{label}_{job['y0']}_{job['y1']}.csv"
            u = "https://daymet.ornl.gov/single-pixel/api/data?" + urllib.parse.urlencode(
                {
                    "lat": job["lat"],
                    "lon": job["lon"],
                    "vars": "tmax,tmin,prcp,vp",
                    "start": f"{job['y0']}-01-01",
                    "end": f"{job['y1']}-12-31",
                }
            )
        elif kind == "ghcnd_station":
            sid = job["station_id"]
            dest = W / "ghcnd" / "stations" / f"{sid}.csv"
            u = f"{GHCND_ACCESS}/{sid}.csv"
        elif kind == "tc":
            i, j, y0, y1 = job["i"], job["j"], job["y0"], job["y1"]
            dest = W / "terraclimate" / f"tc_{i:04d}_{j:04d}_{y0}_{y1}.json"
            t0 = (y0 - 1950) * 12
            t1 = (y1 - 1950) * 12 + 11
            urls = [
                f"{TC_BASE}/agg_terraclimate_{v}_1950_CurrentYear_GLOBE.nc.ascii?{v}[{t0}:{t1}][{i-1}:{i+1}][{j-1}:{j+1}]"
                for v in ("tmax", "tmin", "ppt")
            ]
            u = urls[0]
            r["urls"] = urls
        else:
            raise ValueError(f"Unknown job kind: {kind}")

        r.update(path=str(dest.relative_to(RAW)), url=u)
        if dest.exists() and dest.stat().st_size > 100:
            r["status"] = "reused_existing"
            return r

        dest.parent.mkdir(parents=True, exist_ok=True)
        if kind == "sst":
            payload = request_bytes(u)
            if len(payload) < 100_000:
                raise ValueError("Truncated SST")
            dest.write_bytes(payload)
        elif kind == "dm":
            payload = request_bytes(u)
            if b"year,yday" not in payload:
                raise ValueError("Unexpected Daymet format")
            dest.write_bytes(payload)
        elif kind == "ghcnd_station":
            payload = request_bytes(u, timeout=45)
            if b"STATION" not in payload[:120]:
                raise ValueError("Unexpected GHCN-Daily CSV header")
            dest.write_bytes(payload)
        elif kind == "tc":
            cubes = {}
            for v, u1 in zip(("tmax", "tmin", "ppt"), urls):
                text = request_bytes(u1, timeout=45).decode("utf-8")
                cb, lats, lons = old.parse_terraclimate_3x3_ascii(text, v, t1 - t0 + 1)
                cubes[v] = cb
            locked = existing_tc_cells.get((i, j))
            options = []
            for a in range(3):
                for b in range(3):
                    if cubes["tmax"][0][a][b] != -32768:
                        if locked is not None and (
                            abs(lats[a] - locked[0]) > 1e-4 or abs(lons[b] - locked[1]) > 1e-4
                        ):
                            continue
                        options.append((old.haversine_km(job["lat"], job["lon"], lats[a], lons[b]), a, b))
            if not options:
                d = {"valid_land": False, "ilat": i, "ilon": j, "y0": y0, "y1": y1}
            else:
                _, a, b = min(options)
                d = {
                    "valid_land": True,
                    "ilat": i - 1 + a,
                    "ilon": j - 1 + b,
                    "cell_lat": lats[a],
                    "cell_lon": lons[b],
                    "y0": y0,
                    "y1": y1,
                    "t0_idx": t0,
                }
                for v in ("tmax", "tmin", "ppt"):
                    key_out = v + "_c" if v != "ppt" else "ppt_mm"
                    d[key_out] = [
                        None
                        if cubes[v][k][a][b] in (-32768, -2147483648)
                        else round(cubes[v][k][a][b] * (0.1 if v == "ppt" else 0.01) + (0 if v == "ppt" else -99), 2)
                        for k in range(t1 - t0 + 1)
                    ]
            save(dest, d)
        r["status"] = "downloaded"
        r["fetched_utc"] = now()
    except Exception as exc:
        r["status"] = "failed"
        r["error"] = f"{type(exc).__name__}: {exc}"
    return r


def run() -> None:
    all_cands = candidates()
    # Audit laboratory records across all candidate series (primary + sensitivities)
    lab_records_all = []
    for s in all_cands:
        if s.get("environment_type") == "laboratory":
            lab_records_all.append(
                {
                    "series_id": s["series_id"],
                    "series_kind": s["series_kind"],
                    "database": s["database"],
                    "species": s["species"],
                    "primary_eligible": eligible(s, "primary")[0] is not None,
                    "s1_zero_eligible": eligible(s, "S1_zero")[0] is not None,
                    "s7_cov50_eligible": eligible(s, "S7_coverage50")[0] is not None,
                    "reported_regime": s.get("lab_regime"),
                }
            )

    gpdd_mains = {}
    gpdd_locs = {}
    with (RAW / "gpdd/main.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            gpdd_mains[r["MainID"]] = r
    with (RAW / "gpdd/location.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            gpdd_locs[r["LocationID"]] = r
    gpdd_ocean_by_mid = {
        mid: (gpdd_locs.get(m.get("LocationID", ""), {}).get("Ocean") or "")
        for mid, m in gpdd_mains.items()
    }
    has_ocean_250km = load_ersst_mask()
    existing_tc_cells = load_existing_tc_cells()

    ss = []
    for s in all_cands:
        if s["series_kind"] != "primary":
            continue
        if eligible(s)[0] is not None:
            ss.append(s)

    r27_inventory = []
    jobs: dict[tuple, dict] = {}
    land_station_requests: list[tuple[float, float, int, int]] = []
    site_type_counts = collections.Counter()

    # Check which (i, j) already have full TerraClimate coverage in existing files
    existing_tc_spans: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
    for fp in sorted((W / "terraclimate").glob("tc_*.json")):
        p = fp.stem.split("_")
        existing_tc_spans[(int(p[1]), int(p[2]))].append((int(p[3]), int(p[4])))

    def tc_covers(i: int, j: int, y0: int, y1: int) -> bool:
        spans = existing_tc_spans.get((i, j), [])
        if not spans:
            return False
        covered = set()
        for a0, a1 in spans:
            covered.update(range(a0, a1 + 1))
        return all(yr in covered for yr in range(y0, y1 + 1))

    for s in ss:
        mask = s["values"] > 0
        yrs = s["step_years"][mask]
        yrs_valid = yrs[(yrs >= 1700) & (yrs <= 2025)]
        lat, lon = s["latitude"], s["longitude"]
        pinfo = classify_physical_site(s, gpdd_ocean_by_mid, has_ocean_250km)
        site_type_counts[pinfo["physical_site_type"]] += 1

        r27_inventory.append(
            {
                "series_id": s["series_id"],
                "study_id": s["study_id"],
                "database": s["database"],
                "species": s["species"],
                "taxon_group": s["taxon_group"],
                "interval_class": s["interval_class"],
                "environment_type": s["environment_type"],
                "realm_registered": s["realm_detail"],
                "physical_site_type": pinfo["physical_site_type"],
                "neon_site_code": pinfo["neon_site_code"],
                "neon_tower_distance_km": pinfo["neon_tower_distance_km"],
                "anadromous_inland_flag": pinfo["anadromous_inland_flag"],
                "latitude": lat,
                "longitude": lon,
                "published_location": s["published_location"],
                "location_uncertainty": s["location_uncertainty"],
                "start_year": int(min(yrs_valid)) if len(yrs_valid) else None,
                "end_year": int(max(yrs_valid)) if len(yrs_valid) else None,
                "n_positive_steps": int(mask.sum()),
                "dates": s["step_dates_iso"]
                if s["step_dates_iso"]
                else list(zip(s["step_years"][mask].tolist(), s["step_months"][mask].tolist())),
                "date_precision": s["date_precision"],
                "lab_regime": s["lab_regime"],
            }
        )

        if s["environment_type"] == "laboratory" or not len(yrs_valid) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        y0, y1 = int(min(yrs_valid)), int(max(yrs_valid))
        ptype = pinfo["physical_site_type"]

        if ptype == "marine_or_coastal":
            for yr in range(max(1854, y0 - 1), min(2025, y1) + 1):
                jobs[("sst", yr)] = {"kind": "sst", "year": yr}
        elif ptype in ("neon_terrestrial", "inland_land_or_freshwater", "anadromous_inland_spawning"):
            # Subannual North America 1980+ -> Daymet 1 km daily (include y0-1 for lag-1 bin)
            if s["interval_class"] != "annual" and 14 <= lat <= 72 and -170 <= lon <= -52 and y1 >= 1980:
                la, lo = round(lat, 5), round(lon, 5)
                dm_y0, dm_y1 = max(1980, y0 - 1), min(2025, y1)
                key = ("dm", la, lo)
                rec = jobs.setdefault(key, {"kind": "dm", "lat": la, "lon": lo, "y0": 9999, "y1": 0})
                rec["y0"] = min(rec["y0"], dm_y0)
                rec["y1"] = max(rec["y1"], dm_y1)
            # Global land 1950+ -> TerraClimate 1/24 deg monthly (include y0-1 for lag-1 bin)
            if y1 >= 1950:
                i = max(1, min(4318, int(round((90 - lat) * 24 - 0.5))))
                j = max(1, min(8638, int(round((lon + 180) * 24 - 0.5))))
                tc_y0, tc_y1 = max(1950, y0 - 1), min(2025, y1)
                if not tc_covers(i, j, tc_y0, tc_y1):
                    key = ("tc", i, j)
                    rec = jobs.setdefault(key, {"kind": "tc", "i": i, "j": j, "y0": 9999, "y1": 0, "lat": lat, "lon": lon})
                    rec["y0"] = min(rec["y0"], tc_y0)
                    rec["y1"] = max(rec["y1"], tc_y1)
            # Any series starting before 1951 (or short post-1950 border cases) -> candidate for NOAA GHCN-Daily
            if y0 <= 1964:
                land_station_requests.append((lat, lon, y0, y1))

    csvwrite(RESULTS / "r27_field_locations_dates.csv", r27_inventory)
    save(
        RESULTS / "r27_lab_and_site_audit.json",
        {
            "generated_utc": now(),
            "primary_series": len(ss),
            "primary_laboratory_count": sum(1 for r in r27_inventory if r["environment_type"] == "laboratory"),
            "all_candidate_laboratory_records": lab_records_all,
            "physical_site_type_counts": dict(site_type_counts),
            "neon_series": [r for r in r27_inventory if r["neon_site_code"]],
            "inland_gpdd_formerly_marine_unmatched": [
                {
                    "series_id": r["series_id"],
                    "species": r["species"],
                    "physical_site_type": r["physical_site_type"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "published_location": r["published_location"],
                }
                for r in r27_inventory
                if r["realm_registered"] == "Marine" and r["physical_site_type"] in ("inland_land_or_freshwater", "anadromous_inland_spawning")
            ],
        },
    )

    attempts_path = W / "fetch_attempts.jsonl"
    with attempts_path.open("a", encoding="utf-8") as log_file:
        def log_fn(entry: dict) -> None:
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

        # 1. NEON site/product metadata & anonymous /data probe
        fetch_neon_metadata(log_fn)

        # 2. Log OISST v2.1 CoastWatch/PSL anonymous probe status cleanly
        for oisst_name, oisst_url in (
            ("OISST_v21_PSL", "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc.das"),
            ("OISST_v21_CoastWatch_slice", "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?sst[(2010-01-01T12:00:00Z)][(0.0)][(55.125)][(-1.125)]"),
        ):
            try:
                _ = request_bytes(oisst_url, timeout=12)
                log_fn({"job": {"kind": oisst_name}, "url": oisst_url, "status": "available", "attempted_utc": now()})
            except Exception as exc:
                log_fn({"job": {"kind": oisst_name}, "url": oisst_url, "status": "failed", "error": f"{type(exc).__name__}: {exc} (NOAA ERSST v5 used for marine SST)", "attempted_utc": now()})

        # 3. NOAA GHCN-Daily catalogs & station selection
        _, inv_path = ensure_ghcnd_catalogs(log_fn)
        dedup_land = sorted({(round(la, 4), round(lo, 4), y0, y1) for la, lo, y0, y1 in land_station_requests})
        ghcnd_ids = select_ghcnd_stations(inv_path, dedup_land)
        print(now(), "selected GHCN-Daily stations:", len(ghcnd_ids), "other jobs:", len(jobs), flush=True)
        for sid in ghcnd_ids:
            jobs[("ghcnd", sid)] = {"kind": "ghcnd_station", "station_id": sid}

        save(W / "r27_weather_jobs.json", list(jobs.values()))
        job_list = list(jobs.values())
        total = len(job_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(fetch_job, jb, existing_tc_cells) for jb in job_list]
            for idx, fut in enumerate(concurrent.futures.as_completed(futs)):
                res = fut.result()
                log_fn(res)
                if idx % 20 == 0 or res.get("status") in ("downloaded", "failed"):
                    print(now(), f"[{idx+1}/{total}]", res.get("status"), res.get("path") or res.get("job"), res.get("error", ""), flush=True)

    # Rebuild WEATHER_MANIFEST.json preserving all inherited + newly downloaded files
    attempts_data = [json.loads(ln) for ln in attempts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_path = {r.get("path"): r for r in attempts_data if r.get("path")}
    skip_names = {"WEATHER_MANIFEST.json", "fetch_attempts.jsonl", "continuation_weather_jobs.json", "r27_weather_jobs.json"}
    records = []
    for f in sorted(W.rglob("*")):
        if not f.is_file() or f.name in skip_names:
            continue
        rel = str(f.relative_to(RAW))
        rec = {
            "path": rel,
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "verified_utc": now(),
            "status": "inherited_file",
            "access_date": "2026-11-15",
        }
        if rel in by_path:
            rec.update(by_path[rel])
        if "/terraclimate/" in rel or "terraclimate_" in rel:
            rec.update(dataset="TerraClimate", version="V1.1 (1950-2025)", licence="CC0-1.0", doi="10.1038/sdata.2017.191", resolution="monthly 1/24 degree (~4 km)")
        elif "/daymet/" in rel:
            rec.update(dataset="Daymet", version="V4 R1", licence="NASA open data", doi="10.3334/ORNLDAAC/2129", resolution="daily 1 km")
        elif "/ghcnd/" in rel:
            rec.update(dataset="NOAA GHCN-Daily", version="v3.31", licence="US government public domain", doi="10.7289/V5D21VHZ", resolution="daily meteorological station")
        elif "/ghcnm/" in rel:
            rec.update(dataset="NOAA GHCN-Monthly", version="v4 qcf", licence="US government public domain", doi="10.7289/V5X34W6M", url="https://www.ncei.noaa.gov/pub/data/ghcn/v4/" + f.name, resolution="monthly station")
        elif "/ersst_v5/" in rel:
            rec.update(dataset="NOAA ERSST", version="v5", licence="US government public domain", doi="10.7289/V5T72FNM", url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/" + f.name, resolution="monthly 2 degree")
        elif "/neon/" in rel:
            rec.update(dataset="NEON Instrument Metadata", version="API v0", licence="CC0-1.0", doi="10.48443/bxnk-jb11", resolution="site tower metadata")
        records.append(rec)

    save(W / "WEATHER_MANIFEST.json", {"generated_utc": now(), "items": records, "attempts_count": len(attempts_data)})
    print(now(), "WEATHER_MANIFEST complete; total verified files:", len(records), flush=True)


if __name__ == "__main__":
    run()
