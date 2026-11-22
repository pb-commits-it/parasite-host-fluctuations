#!/usr/bin/env python3
"""Fetch open host-abundance time-series compilations for case study lpd.

Run:
    python3 -I data/fetch/lpd.py

Rules enforced (CASESTUDY_SPEC.md Rule 2):
  - Standard library only; run with `python3 -I`.
  - Generic User-Agent; no e-mail address, credential, login, or personal details in any request.
  - Stores raw files only under `data/raw/lpd/` (git-ignored).
  - Re-runnable / idempotent: skips files already present with non-zero size (and matching expected bytes where known).
  - Living Planet Database (`https://www.livingplanetindex.org/data_portal`) requires a manual
    R Shiny form asking for name, e-mail address, and intended use; per protocol and Rule 2,
    it is not circumvented and is recorded in MANIFEST.json as manual-form blocked.

Datasets fetched:
  1. BioTIME (Zenodo record 5026943, DOI 10.5281/zenodo.5026943, version 24_06_2021, CC BY 4.0):
     - BioTIMEMetadata_24_06_2021.csv
     - BioTIMECitations_24_06_2021.csv
     - BioTIMEQuery_24_06_2021.zip (134.4 MB zipped CSV)
     Plus BioTIME 2.0 metadata table (Zenodo record 15222193, DOI 10.5281/zenodo.15222193, v4 2025-04-15, CC BY 4.0):
     - biotime_v2_metadata_15April25.csv
  2. Global Population Dynamics Database v2 (KNB doi:10.5063/F1BZ63Z8, v2 2010, CC BY 4.0):
     - gpdd_eml_metadata.xml, data.csv, main.csv, timeperiod.csv, taxon.csv, datasource.csv, biotope.csv, location.csv
  3. VectorByte VecDyn (https://vectorbyte.crc.nd.edu/portal/api/, DOI 10.7274/10083626,
     default CC BY-NC 4.0 or per-dataset CC0 / CC BY):
     - Public provider catalogue (`vecdynbyprovider`) and CSV rows (`vecdyncsv`) for public,
       un-embargoed vector surveillance datasets with >= 25 collections and >= 2 years (capped
       at the 120 longest multi-year public datasets to stay polite to the API).
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "open-ecology-fetch/1.0 (python-urllib; reproducible-case-study)"
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
RAW_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '../raw/lpd').resolve()
TODAY_ISO = "2026-11-15"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stream_download(url: str, dest: Path, expected_bytes: int | None = None, tries: int = 3, timeout: int = 240) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_bytes is None or dest.stat().st_size == expected_bytes:
            size = dest.stat().st_size
            digest = sha256_file(dest)
            print(f"  [skip existing] {dest.relative_to(RAW_DIR)} ({size:,} B)")
            return {
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": size,
                "sha256": digest,
                "status": "reused_existing",
            }

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            h = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    h.update(buf)
                    total += len(buf)
            if expected_bytes is not None and total != expected_bytes:
                raise IOError(f"Size mismatch for {dest.name}: got {total}, expected {expected_bytes}")
            tmp.replace(dest)
            print(f"  [downloaded]    {dest.relative_to(RAW_DIR)} ({total:,} B)")
            return {
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": total,
                "sha256": h.hexdigest(),
                "fetched_utc": utc_now(),
                "status": "downloaded",
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Failed to download {url} -> {dest}: {last_err}")


def fetch_json(url: str, timeout: int = 60, tries: int = 3) -> dict | list:
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"Failed JSON GET {url}: {last_err}")


def fetch_biotime(manifest_items: list[dict]) -> None:
    print("1. Fetching BioTIME open files from Zenodo...")
    bt_dir = RAW_DIR / "biotime"
    bt_dir.mkdir(parents=True, exist_ok=True)

    # Record 5026943: BioTIME CSV release (2021-06-24), CC BY 4.0
    rec_v1 = fetch_json("https://zenodo.org/api/records/5026943")
    (bt_dir / "_zenodo_5026943.json").write_text(json.dumps(rec_v1, indent=2))
    wanted_v1 = {
        "BioTIMEMetadata_24_06_2021.csv",
        "BioTIMECitations_24_06_2021.csv",
        "BioTIMEQuery_24_06_2021.zip",
    }
    for fl in rec_v1.get("files", []):
        key = fl["key"]
        if key not in wanted_v1:
            continue
        url = fl["links"]["self"]
        info = stream_download(url, bt_dir / key, expected_bytes=int(fl["size"]))
        info.update({
            "dataset": "BioTIME",
            "doi": "10.5281/zenodo.5026943",
            "version": "24_06_2021",
            "licence": "CC-BY-4.0",
            "url": url,
        })
        manifest_items.append(info)

    # Record 15222193: BioTIME 2.0 metadata CSV (2025-04-15), CC BY 4.0
    rec_v2 = fetch_json("https://zenodo.org/api/records/15222193")
    (bt_dir / "_zenodo_15222193.json").write_text(json.dumps(rec_v2, indent=2))
    for fl in rec_v2.get("files", []):
        if fl["key"] == "biotime_v2_metadata_15April25.csv":
            url = fl["links"]["self"]
            info = stream_download(url, bt_dir / fl["key"], expected_bytes=int(fl["size"]))
            info.update({
                "dataset": "BioTIME_2.0_metadata",
                "doi": "10.5281/zenodo.15222193",
                "version": "v4_2025-04-15",
                "licence": "CC-BY-4.0",
                "url": url,
            })
            manifest_items.append(info)


def fetch_gpdd(manifest_items: list[dict]) -> None:
    print("2. Fetching Global Population Dynamics Database v2 (GPDD) from KNB...")
    gpdd_dir = RAW_DIR / "gpdd"
    mn = "https://knb.ecoinformatics.org/knb/d1/mn/v2/object/"
    objects = [
        ("doi:10.5063/F1BZ63Z8", "gpdd_eml_metadata.xml"),
        ("df35b.233.1", "data.csv"),
        ("df35b.234.1", "main.csv"),
        ("df35b.235.1", "timeperiod.csv"),
        ("df35b.236.1", "taxon.csv"),
        ("df35b.237.1", "datasource.csv"),
        ("df35b.238.1", "biotope.csv"),
        ("df35b.239.1", "location.csv"),
    ]
    for pid, name in objects:
        url = mn + pid
        info = stream_download(url, gpdd_dir / name)
        info.update({
            "dataset": "GPDD_v2",
            "doi": "10.5063/F1BZ63Z8",
            "version": "2.0 (2010)",
            "licence": "CC-BY-4.0",
            "url": url,
        })
        manifest_items.append(info)


def fetch_vecdyn(manifest_items: list[dict]) -> None:
    print("3. Fetching VectorByte VecDyn public multi-year vector surveillance series...")
    vd_dir = RAW_DIR / "vecdyn"
    vd_dir.mkdir(parents=True, exist_ok=True)
    cat_path = vd_dir / "providers_catalogue.json"

    if cat_path.exists() and cat_path.stat().st_size > 1000:
        catalogue = json.loads(cat_path.read_text())
        print(f"  [skip existing] vecdyn/providers_catalogue.json ({len(catalogue)} entries)")
    else:
        catalogue = []
        url = "https://vectorbyte.crc.nd.edu/portal/api/vecdynbyprovider/?format=json&page=1"
        pages = 0
        while url and pages < 40:
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            payload = fetch_json(url, timeout=45)
            data_block = payload.get("data", payload)
            results = data_block.get("results", [])
            # Strip personal email fields before writing metadata to disk
            for entry in results:
                entry.pop("contributoremail", None)
                entry.pop("email", None)
                catalogue.append(entry)
            nxt = data_block.get("next")
            url = nxt if nxt else None
            pages += 1
            time.sleep(0.15)
        cat_path.write_text(json.dumps(catalogue, indent=1))
        print(f"  [downloaded]    vecdyn/providers_catalogue.json ({len(catalogue)} entries across {pages} pages)")

    manifest_items.append({
        "dataset": "VecDyn_catalogue",
        "doi": "10.7274/10083626",
        "version": "accessed_" + TODAY_ISO,
        "licence": "CC-BY-NC-4.0 (default) or per-dataset licence",
        "path": "vecdyn/providers_catalogue.json",
        "bytes": cat_path.stat().st_size,
        "sha256": sha256_file(cat_path),
        "url": "https://vectorbyte.crc.nd.edu/portal/api/vecdynbyprovider/",
        "status": "ok",
    })

    # Select public, un-embargoed hematophagous vector datasets with >= 20 collections and >= 2 years,
    # paginating all 50-row pages for datasets with total row count <= 6000 (top 28 longest multi-year series)
    vec_kw = (
        "anopheles", "aedes", "ochlerotatus", "culex", "culiseta", "mansonia",
        "coquillettidia", "psorophora", "uranotaenia", "wyeomyia", "haemagogus",
        "simulium", "culicoides", "phlebotomus", "lutzomyia", "tabanus", "chrysops",
        "haematopota", "glossina", "stomoxys", "culicidae", "ixodes", "dermacentor",
        "amblyomma", "haemaphysalis", "rhipicephalus",
    )
    eligible = []
    for entry in catalogue:
        emb = str(entry.get("EmbargoReleaseDate") or "").strip()
        if emb and emb > TODAY_ISO:
            continue
        collections = int(entry.get("Collections") or 0)
        years = entry.get("Years") or []
        spp = entry.get("SpeciesName") or []
        if collections >= 20 and len(years) >= 2 and any(any(k in s.lower() for k in vec_kw) for s in spp):
            eligible.append((len(years), collections, int(entry["Id"]), entry.get("dataset_license")))
    # Prioritize longest year spans and moderate collection counts so full pagination stays fast and polite
    eligible.sort(key=lambda t: (-t[0], t[1], t[2]))
    full_dir = vd_dir / "full_datasets"
    full_dir.mkdir(parents=True, exist_ok=True)

    saved_full = 0
    for n_yrs, cols, did, dlic in eligible:
        if saved_full >= 28:
            break
        dest = full_dir / f"vecdyn_full_{did}.json"
        base_url = f"https://vectorbyte.crc.nd.edu/portal/api/vecdyncsv/?piids={did}"
        if dest.exists() and dest.stat().st_size > 100:
            size = dest.stat().st_size
            manifest_items.append({
                "dataset": "VecDyn_full_dataset",
                "doi": "10.7274/10083626",
                "version": "accessed_" + TODAY_ISO,
                "licence": str(dlic or "CC-BY-NC-4.0"),
                "dataset_id": did,
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": size,
                "sha256": sha256_file(dest),
                "url": base_url,
                "status": "reused_existing",
            })
            saved_full += 1
            continue

        try:
            p1 = fetch_json(base_url + "&page=1", timeout=45)
            total_rows = int(p1.get("count") or 0)
            if total_rows < 20 or total_rows > 6500:
                # Skip huge >6500-row tables (>130 pages) to remain polite to API
                continue
            cons = dict(p1.get("consistent_data") or {})
            cons.pop("email", None)
            cons.pop("contributoremail", None)
            all_rows = []
            for r in p1.get("results") or []:
                r.pop("email", None)
                r.pop("contributoremail", None)
                all_rows.append(r)
            n_pages = (total_rows + 49) // 50
            for pg in range(2, n_pages + 1):
                pj = fetch_json(f"{base_url}&page={pg}", timeout=45)
                for r in pj.get("results") or []:
                    r.pop("email", None)
                    r.pop("contributoremail", None)
                    all_rows.append(r)
                time.sleep(0.08)
            out_obj = {
                "dataset_id": did,
                "count": total_rows,
                "digitized_from_graph": p1.get("digitized_from_graph"),
                "dataset_license": str(dlic or "CC-BY-NC-4.0"),
                "consistent_data": cons,
                "results": all_rows,
            }
            raw_bytes = json.dumps(out_obj, separators=(",", ":")).encode("utf-8")
            dest.write_bytes(raw_bytes)
            manifest_items.append({
                "dataset": "VecDyn_full_dataset",
                "doi": "10.7274/10083626",
                "version": "accessed_" + TODAY_ISO,
                "licence": str(dlic or "CC-BY-NC-4.0"),
                "dataset_id": did,
                "rows_fetched": len(all_rows),
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "fetched_utc": utc_now(),
                "url": base_url,
                "status": "downloaded",
            })
            saved_full += 1
            print(f"  [downloaded full] vecdyn_full_{did}.json ({len(all_rows)} rows, {n_yrs} yrs)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {did}] {exc}", flush=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def parse_terraclimate_3x3_ascii(txt: str, var: str, n_t: int) -> tuple[list[list[list[int]]], list[float], list[float]]:
    """Parse 3x3 OPeNDAP ASCII grid response for var[n_t][3][3], lat[3], lon[3]."""
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
            # [t][r], v0, v1, v2
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


def fetch_weather() -> dict:
    """Download open site- and date-matched weather and sea-surface datasets for field series
    (Protocol Amendment 1, CASESTUDY_SPEC.md §'Weather and environment at the place and time of the data'):
      1. NOAA GHCN-Monthly v4 global station archive (1701-2026, ~26,000 stations, public domain)
      2. NOAA ERSST v5 global monthly 2x2 deg sea-surface temperature (1854-2025, public domain)
      3. NASA ORNL DAAC Daymet V4 R1 daily 1 km North American weather for sub-annual VecDyn sites
      4. TerraClimate 1/24 deg (~4 km) global monthly tmax, tmin, ppt via U. Idaho NKN OPeNDAP
    """
    t0 = time.monotonic()
    w_dir = RAW_DIR / "weather"
    w_dir.mkdir(parents=True, exist_ok=True)
    w_items: list[dict] = []

    # 1. NOAA GHCN-Monthly v4
    print("W1. Fetching NOAA GHCN-Monthly v4 global land station archive...")
    ghcn_dir = w_dir / "ghcnm"
    ghcn_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("readme.txt", "ghcnm.tavg.latest.qcf.tar.gz"):
        u = f"https://www.ncei.noaa.gov/pub/data/ghcn/v4/{fname}"
        info = stream_download(u, ghcn_dir / fname, timeout=240)
        info.update({
            "dataset": "NOAA_GHCN_Monthly_v4",
            "doi": "10.7289/V5X34W6M",
            "version": "v4.0.1 (latest qcf)",
            "licence": "U.S. Government Public Domain",
            "resolution": "Point station monthly TAVG (~26,000 stations, 1701-2026)",
            "url": u,
        })
        w_items.append(info)

    # Scan GPDD, BioTIME, and VecDyn metadata to collect exact marine ERSST years,
    # North American Daymet sites, and TerraClimate 1/24-deg land cells
    ersst_years: set[int] = set()
    tc_cells: dict[tuple[int, int], dict] = {}  # (ilat, ilon) -> {y0, y1, ref_lat, ref_lon}
    daymet_jobs: list[tuple[str, float, float, int, int]] = []

    fw_kw = ("limnetic", "fluvial", "wetland", "lake", "pond", "river", "stream", "freshwater", "marsh", "bog")
    gpdd_dir = RAW_DIR / "gpdd"
    if (gpdd_dir / "main.csv").exists():
        with (gpdd_dir / "taxon.csv").open(encoding="latin-1") as f:
            tax = {r["TaxonID"]: r for r in csv.DictReader(f)}
        with (gpdd_dir / "biotope.csv").open(encoding="latin-1") as f:
            bio = {r["BiotopeID"]: r for r in csv.DictReader(f)}
        with (gpdd_dir / "location.csv").open(encoding="latin-1") as f:
            loc = {r["LocationID"]: r for r in csv.DictReader(f)}
        with (gpdd_dir / "main.csv").open(encoding="latin-1") as f:
            for m in csv.DictReader(f):
                try:
                    dlen = int(float(m.get("DatasetLength") or 0))
                    y0 = int(float(m.get("StartYear") or -9999))
                    y1 = int(float(m.get("EndYear") or -9999))
                except ValueError:
                    continue
                if dlen < 15 or y0 < 1700 or y1 < 1854:
                    continue
                t_cls = (tax.get(m["TaxonID"], {}).get("TaxonomicClass") or "").strip().lower()
                if t_cls not in (
                    "mammalia", "aves", "actinopterygii", "osteichthyes", "chondrichthyes",
                    "chondrichtyhes", "cephalaspidomorphi", "hyperoartia", "petromyzontida",
                    "pisces", "insecta", "arachnida", "gastropoda", "bivalvia", "crustacea",
                    "rotifera", "oligochaeta", "hirudinea",
                ):
                    continue
                b = bio.get(m["BiotopeID"], {})
                l = loc.get(m["LocationID"], {})
                hab = ((b.get("HabitatName") or "") + " " + (b.get("BiotopeType") or "")).lower()
                ocean = (l.get("Ocean") or "").strip().lower()
                is_fw = any(k in hab for k in fw_kw)
                is_mar = ("marine" in hab) or ("tidal" in hab) or ("coastal" in hab and not is_fw) or bool(ocean and ocean not in ("none", "na", "unknown", "land"))
                try:
                    lat, lon = float(l.get("LatDD") or "nan"), float(l.get("LongDD") or "nan")
                except ValueError:
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
                    continue
                if is_mar:
                    for y in range(max(1854, y0), min(2025, y1) + 1):
                        ersst_years.add(y)
                elif y1 >= 1950:
                    ilat = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
                    ilon = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
                    rec = tc_cells.setdefault((ilat, ilon), {"y0": 9999, "y1": 0, "ref_lat": lat, "ref_lon": lon})
                    rec["y0"] = min(rec["y0"], max(1950, y0))
                    rec["y1"] = max(rec["y1"], min(2025, y1))

    bt_meta_path = RAW_DIR / "biotime" / "BioTIMEMetadata_24_06_2021.csv"
    if bt_meta_path.exists():
        with bt_meta_path.open(encoding="latin-1") as f:
            for st in csv.DictReader(f):
                if st["TAXA"] not in ("Mammals", "Birds", "Fish", "Freshwater invertebrates", "Benthos", "All"):
                    continue
                try:
                    y0, y1 = int(float(st["START_YEAR"])), int(float(st["END_YEAR"]))
                    lat, lon = float(st["CENT_LAT"]), float(st["CENT_LONG"])
                except ValueError:
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                if st["REALM"] == "Marine":
                    for y in range(max(1854, y0), min(2025, y1) + 1):
                        ersst_years.add(y)
                elif y1 >= 1950:
                    ilat = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
                    ilon = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
                    rec = tc_cells.setdefault((ilat, ilon), {"y0": 9999, "y1": 0, "ref_lat": lat, "ref_lon": lon})
                    rec["y0"] = min(rec["y0"], max(1950, y0))
                    rec["y1"] = max(rec["y1"], min(2025, y1))

    vd_full_dir = RAW_DIR / "vecdyn" / "full_datasets"
    if vd_full_dir.exists():
        for jf in sorted(vd_full_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            did = d["dataset_id"]
            cons = d.get("consistent_data") or {}
            cdoi = str(cons.get("curatedbydoi") or "")
            ainfo = str(cons.get("additional_sample_info") or "")
            if ("s41597-024-03482-y" in cdoi) or ("temporally downscaled" in ainfo.lower()):
                continue
            lats, lons, yrs = [], [], []
            for r in d.get("results") or []:
                la = r.get("sample_lat_dd") or cons.get("sample_lat_dd")
                lo = r.get("sample_long_dd") or cons.get("sample_long_dd")
                ds = (r.get("sample_end_date") or cons.get("sample_end_date") or "")[:4]
                try:
                    if la is not None and lo is not None:
                        lats.append(float(la))
                        lons.append(float(lo))
                    if len(ds) == 4:
                        yrs.append(int(ds))
                except ValueError:
                    pass
            if not lats or not lons or not yrs:
                continue
            lat, lon = lats[0], lons[0]
            y0, y1 = min(yrs), max(yrs)
            if y1 >= 1950:
                ilat = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
                ilon = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
                rec = tc_cells.setdefault((ilat, ilon), {"y0": 9999, "y1": 0, "ref_lat": lat, "ref_lon": lon})
                rec["y0"] = min(rec["y0"], max(1950, y0))
                rec["y1"] = max(rec["y1"], min(2025, y1))
            if 14.0 <= lat <= 72.0 and -170.0 <= lon <= -52.0 and y1 >= 1980:
                daymet_jobs.append((f"VD_{did}", lat, lon, max(1980, y0), min(2025, y1)))

    # 2. NOAA ERSST v5 ASCII
    print(f"W2. Fetching NOAA ERSST v5 monthly 2x2 deg SST for {len(ersst_years)} years...")
    ersst_dir = w_dir / "ersst_v5"
    ersst_dir.mkdir(parents=True, exist_ok=True)
    readme_info = stream_download(
        "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/Readme",
        ersst_dir / "Readme.txt",
        timeout=60,
    )
    readme_info.update({
        "dataset": "NOAA_ERSST_v5",
        "doi": "10.7289/V5T72FNM",
        "version": "v5",
        "licence": "U.S. Government Public Domain",
        "url": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/Readme",
    })
    w_items.append(readme_info)

    def dl_ersst_year(yr: int) -> dict:
        fname = f"ersst.v5.{yr}.asc"
        u = f"https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/{fname}"
        dest = ersst_dir / fname
        if dest.exists() and dest.stat().st_size > 100_000:
            return {
                "dataset": "NOAA_ERSST_v5",
                "doi": "10.7289/V5T72FNM",
                "version": "v5",
                "licence": "U.S. Government Public Domain",
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "url": u,
                "status": "reused_existing",
            }
        req = urllib.request.Request(u, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        tmp = dest.with_suffix(".asc.part")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        return {
            "dataset": "NOAA_ERSST_v5",
            "doi": "10.7289/V5T72FNM",
            "version": "v5",
            "licence": "U.S. Government Public Domain",
            "path": str(dest.relative_to(RAW_DIR)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_utc": utc_now(),
            "url": u,
            "status": "downloaded",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ersst_res = list(pool.map(dl_ersst_year, sorted(ersst_years)))
    w_items.extend(ersst_res)
    print(f"  [OK] ERSST v5: {len(ersst_res)} annual ASCII files ready", flush=True)

    # 3. NASA ORNL DAAC Daymet V4 R1 daily 1 km for North American sub-annual VecDyn sites
    print(f"W3. Fetching Daymet V4 R1 daily 1-km weather for {len(daymet_jobs)} North American VecDyn sites...")
    dm_dir = w_dir / "daymet"
    dm_dir.mkdir(parents=True, exist_ok=True)
    for label, lat, lon, y0, y1 in daymet_jobs:
        dest = dm_dir / f"{label}.csv"
        q = urllib.parse.urlencode({
            "lat": f"{lat:.5f}",
            "lon": f"{lon:.5f}",
            "vars": "tmax,tmin,prcp,vp",
            "start": f"{y0}-01-01",
            "end": f"{y1}-12-31",
        })
        u = f"https://daymet.ornl.gov/single-pixel/api/data?{q}"
        try:
            info = stream_download(u, dest, timeout=60)
            info.update({
                "dataset": "Daymet_V4_R1",
                "doi": "10.3334/ORNLDAAC/2129",
                "version": "V4R1",
                "licence": "NASA Open Data / Public Domain",
                "latitude": lat,
                "longitude": lon,
                "url": u,
            })
            w_items.append(info)
            time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN Daymet {label}] {exc}", flush=True)

    # 4. TerraClimate 1/24-deg (~4 km) monthly tmax, tmin, ppt via U. Idaho NKN OPeNDAP
    print(f"W4. Fetching TerraClimate 1/24-deg (~4 km) monthly tmax/tmin/ppt for {len(tc_cells)} land grid cells...")
    tc_dir = w_dir / "terraclimate"
    tc_dir.mkdir(parents=True, exist_ok=True)

    def dl_tc_cell(item: tuple[tuple[int, int], dict]) -> dict:
        (ilat, ilon), meta = item
        y0, y1 = int(meta["y0"]), int(meta["y1"])
        ref_lat, ref_lon = float(meta["ref_lat"]), float(meta["ref_lon"])
        dest = tc_dir / f"tc_{ilat:04d}_{ilon:04d}_{y0}_{y1}.json"
        t0_idx = max(0, (y0 - 1950) * 12)
        t1_idx = min(911, (y1 - 1950) * 12 + 11)
        n_t = t1_idx - t0_idx + 1
        base_dods = "https://thredds.northwestknowledge.net/thredds/dodsC"
        url_tmax = f"{base_dods}/agg_terraclimate_tmax_1950_CurrentYear_GLOBE.nc.ascii?tmax[{t0_idx}:{t1_idx}][{ilat-1}:{ilat+1}][{ilon-1}:{ilon+1}]"
        if dest.exists() and dest.stat().st_size > 100:
            return {
                "dataset": "TerraClimate_Monthly_4km",
                "doi": "10.1038/sdata.2017.191",
                "version": "V1.1/V2 (1950-CurrentYear)",
                "licence": "Public Domain / Freely Available (U. Idaho Climatology Lab)",
                "path": str(dest.relative_to(RAW_DIR)),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "url": url_tmax,
                "status": "reused_existing",
            }
        cubes = {}
        lats_3, lons_3 = [], []
        for var in ("tmax", "tmin", "ppt"):
            u = f"{base_dods}/agg_terraclimate_{var}_1950_CurrentYear_GLOBE.nc.ascii?{var}[{t0_idx}:{t1_idx}][{ilat-1}:{ilat+1}][{ilon-1}:{ilon+1}]"
            last_err = None
            for att in range(3):
                try:
                    req = urllib.request.Request(u, headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        txt = resp.read().decode("utf-8", "replace")
                    cb, la3, lo3 = parse_terraclimate_3x3_ascii(txt, var, n_t)
                    cubes[var] = cb
                    if la3 and lo3:
                        lats_3, lons_3 = la3, lo3
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(1.5 + 2.0 * att)
            else:
                raise RuntimeError(f"TerraClimate failed ({ilat},{ilon},{var}): {last_err}")

        # Pick nearest non-ocean-masked pixel in the 3x3 neighborhood
        best_rc = None
        best_d = 1e9
        for r_i in range(3):
            for c_i in range(3):
                # Check if pixel is valid land (tmax != -32768)
                if cubes["tmax"][0][r_i][c_i] != -32768:
                    d_km = haversine_km(ref_lat, ref_lon, lats_3[r_i], lons_3[c_i])
                    if d_km < best_d:
                        best_d = d_km
                        best_rc = (r_i, c_i)
        if best_rc is None:
            payload = {"valid_land": False, "ilat": ilat, "ilon": ilon, "y0": y0, "y1": y1}
        else:
            ri, ci = best_rc
            tmax_c = [round(cubes["tmax"][t][ri][ci] * 0.01 - 99.0, 2) if cubes["tmax"][t][ri][ci] != -32768 else None for t in range(n_t)]
            tmin_c = [round(cubes["tmin"][t][ri][ci] * 0.01 - 99.0, 2) if cubes["tmin"][t][ri][ci] != -32768 else None for t in range(n_t)]
            ppt_mm = [round(cubes["ppt"][t][ri][ci] * 0.1, 1) if cubes["ppt"][t][ri][ci] != -2147483648 else None for t in range(n_t)]
            payload = {
                "valid_land": True,
                "ilat": ilat - 1 + ri,
                "ilon": ilon - 1 + ci,
                "cell_lat": round(lats_3[ri], 6),
                "cell_lon": round(lons_3[ci], 6),
                "y0": y0,
                "y1": y1,
                "t0_idx": t0_idx,
                "tmax_c": tmax_c,
                "tmin_c": tmin_c,
                "ppt_mm": ppt_mm,
            }
        raw_b = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        tmp = dest.with_suffix(".json.part")
        tmp.write_bytes(raw_b)
        tmp.replace(dest)
        return {
            "dataset": "TerraClimate_Monthly_4km",
            "doi": "10.1038/sdata.2017.191",
            "version": "V1.1/V2 (1950-CurrentYear)",
            "licence": "Public Domain / Freely Available (U. Idaho Climatology Lab)",
            "path": str(dest.relative_to(RAW_DIR)),
            "bytes": len(raw_b),
            "sha256": hashlib.sha256(raw_b).hexdigest(),
            "fetched_utc": utc_now(),
            "url": url_tmax,
            "status": "downloaded",
        }

    tc_list = sorted(tc_cells.items())
    tc_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for idx_done, res in enumerate(pool.map(dl_tc_cell, tc_list), 1):
            tc_results.append(res)
            if idx_done % 20 == 0 or idx_done == len(tc_list):
                print(f"  [TerraClimate {idx_done}/{len(tc_list)}]", flush=True)
    w_items.extend(tc_results)

    w_manifest = {
        "study_id": "lpd",
        "component": "weather_and_environment",
        "generated_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "total_bytes": sum(int(x.get("bytes", 0)) for x in w_items),
        "items": w_items,
    }
    wm_path = w_dir / "WEATHER_MANIFEST.json"
    wm_path.write_text(json.dumps(w_manifest, indent=2))
    print(
        f"Completed weather fetch in {w_manifest['elapsed_seconds']}s: "
        f"{w_manifest['total_bytes'] / (1024 ** 2):.2f} MiB -> {wm_path}",
        flush=True,
    )
    return w_manifest


def main() -> int:
    t0 = time.monotonic()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    free_gb_before = shutil.disk_usage(RAW_DIR).free / (1024 ** 3)
    print(f"Starting fetch for case study lpd at {utc_now()} (free disk: {free_gb_before:.2f} GiB)")
    if free_gb_before < 6.0:
        raise RuntimeError(f"Insufficient disk space ({free_gb_before:.2f} GiB < 6.0 GiB safety floor)")

    if "--weather-only" in sys.argv:
        fetch_weather()
        return 0

    manifest_items: list[dict] = [
        {
            "dataset": "Living_Planet_Database_LPD",
            "url": "https://www.livingplanetindex.org/data_portal",
            "policy_url": "https://www.livingplanetindex.org/documents/LPI_Data_Use_Policy_2026.pdf",
            "licence": "Custom LPI Data Use Policy 2026 (research/conservation only; no redistribution of original data)",
            "version": "2026.1 (checked 2026-11-15)",
            "status": "not_downloaded_manual_form_required",
            "reason": (
                "The public download button (id='download_alldata') on livingplanetindex.org/data_portal "
                "is disabled inside an interactive R Shiny session until a human manually enters name, "
                "e-mail address, and intended use and ticks two policy checkboxes. Per CASESTUDY_SPEC.md "
                "Rule 2 (no e-mail/personal details; never circumvent a gate) and task instructions, "
                "this requirement is reported and open compilations (BioTIME, GPDD v2, VecDyn) are used instead."
            ),
            "checked_utc": utc_now(),
        }
    ]

    fetch_biotime(manifest_items)
    fetch_gpdd(manifest_items)
    fetch_vecdyn(manifest_items)
    if "--skip-weather" not in sys.argv:
        fetch_weather()

    free_gb_after = shutil.disk_usage(RAW_DIR).free / (1024 ** 3)
    total_bytes = sum(int(m.get("bytes", 0)) for m in manifest_items)
    manifest = {
        "study_id": "lpd",
        "generated_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "total_downloaded_bytes": total_bytes,
        "free_disk_gib_before": round(free_gb_before, 2),
        "free_disk_gib_after": round(free_gb_after, 2),
        "items": manifest_items,
    }
    manifest_path = RAW_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Completed fetch in {manifest['elapsed_seconds']}s: "
        f"{total_bytes / (1024 ** 2):.2f} MiB in {RAW_DIR} "
        f"(free disk: {free_gb_after:.2f} GiB). Wrote {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
