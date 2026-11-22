#!/usr/bin/env python3
"""Fetch open environmental covariates for case study envcov (Cary Institute + NEON sites).

Protocol: casestudies/envcov/protocol.md

Rules enforced (CASESTUDY_SPEC.md Rule 2 & task specification):
  - Open public endpoints only; generic User-Agent; NO email address, personal details, or API credential.
  - Explicitly ignores any environment credential and never sends Authorization / X-API-Token headers.
  - Respects server rate limits (sequential requests with polite pauses and exponential backoff).
  - Resumable: existing complete files on disk are skipped unless --force is passed.
  - Writes data/raw/envcov/MANIFEST.json with licence, version, DOI, URL, sha256, bytes, and UTC date.

Run:
    python3 -I data/fetch/envcov.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True

UA = "envcov-study-fetch/1.0 (python-urllib)"
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
RAW_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../raw').resolve()
OUT = RAW_ROOT / "envcov"
NEON_RAW = RAW_ROOT / "neon"

NEON_API = "https://data.neonscience.org/api/v0"
DAYMET_API = "https://daymet.ornl.gov/single-pixel/api/data"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
MODIS_RST_API = "https://modis.ornl.gov/rst/api/v1/products"

R14_INCLUDED_SITES = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]
R14_OTHER_SITES = ["DELA", "KONZ", "MLBS", "OSBS", "STEI", "TALL", "UKFS"]

NEON_PRODUCTS = {
    "DP1.00002.001": "Single aspirated air temperature",
    "DP1.00003.001": "Triple aspirated air temperature",
    "DP1.00006.001": "Precipitation",
    "DP1.00098.001": "Relative humidity",
    "DP1.00041.001": "Soil temperature",
    "DP1.00094.001": "Soil water content and water salinity",
    "DP3.30015.001": "Ecosystem structure (AOP lidar canopy height)",
}

# Published Cary Institute Lyme-study plot coordinates (Millbrook, NY):
# Burtis et al. (2016, Parasites & Vectors 9:606, PMC5124264, p. 3): 41°47'1.04"N, 73°43'56.49"W
CARY_COORD = (41.783622, -73.732358, "Cary Institute Lyme study plots, Millbrook, NY (Burtis et al. 2016 PMC5124264)")

# Fallback mean tick-plot coordinates from NEON DP1.10093.001 tck_fielddata (recomputed if local raw files exist)
FALLBACK_TICK_COORDS = {
    "BLAN": (39.08693, -77.96624),
    "HARV": (42.42286, -72.23639),
    "ORNL": (35.95903, -84.27526),
    "SCBI": (38.89096, -78.15290),
    "SERC": (38.88224, -76.54777),
    "TREE": (45.49146, -89.56240),
}

# Fallback official NEON site tower coordinates from /api/v0/sites/<site>
FALLBACK_TOWER_COORDS = {
    "BLAN": (39.03370, -78.04179),
    "HARV": (42.53691, -72.17265),
    "ORNL": (35.96413, -84.28259),
    "SCBI": (38.89293, -78.13949),
    "SERC": (38.89013, -76.56001),
    "TREE": (45.49369, -89.58571),
    "DELA": (32.54173, -87.80388),
    "KONZ": (39.10077, -96.56307),
    "MLBS": (37.37831, -80.52485),
    "OSBS": (29.68928, -81.99343),
    "STEI": (45.50895, -89.58637),
    "TALL": (32.95047, -87.39326),
    "UKFS": (39.04043, -95.19215),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, accept: str | None = None, timeout: int = 45, retries: int = 3, pause: float = 0.35) -> tuple[int, bytes, str]:
    """Unauthenticated GET with generic User-Agent, polite pause, and backoff."""
    hdr = {"User-Agent": UA}
    if accept:
        hdr["Accept"] = accept
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                status = int(r.status)
            time.sleep(pause)
            return status, body, ""
        except urllib.error.HTTPError as e:
            err_body = e.read(512).decode("utf-8", "replace")
            time.sleep(pause)
            if e.code in (401, 403, 404, 422):
                return int(e.code), b"", err_body
            last_err = f"HTTP {e.code}: {err_body[:160]}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        if attempt + 1 < retries:
            time.sleep(2.0 * (2 ** attempt))
    return 0, b"", last_err


def save_atomic(dest: Path, body: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    os.replace(tmp, dest)


def compute_tick_plot_coords() -> dict[str, tuple[float, float, str]]:
    """Compute mean tick-drag plot coordinates from local NEON DP1.10093.001 files if present."""
    coords = {}
    for site in R14_INCLUDED_SITES:
        pattern = str(NEON_RAW / "DP1.10093.001" / site / "*tck_fielddata*.csv")
        files = sorted(glob.glob(pattern))
        plot_ll = {}
        for fn in files:
            with open(fn, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = (row.get("plotID") or "").strip()
                    lat = (row.get("decimalLatitude") or "").strip()
                    lon = (row.get("decimalLongitude") or "").strip()
                    if pid and lat and lon and pid not in plot_ll:
                        plot_ll[pid] = (float(lat), float(lon))
        if plot_ll:
            mlat = round(sum(v[0] for v in plot_ll.values()) / len(plot_ll), 5)
            mlon = round(sum(v[1] for v in plot_ll.values()) / len(plot_ll), 5)
            coords[site] = (mlat, mlon, f"NEON {site} mean over {len(plot_ll)} DP1.10093.001 tick plots")
        else:
            fl = FALLBACK_TICK_COORDS[site]
            coords[site] = (fl[0], fl[1], f"NEON {site} tick-plot mean (embedded fallback)")
    return coords


def record_manifest(
    manifest: list[dict],
    dest: Path,
    source: str,
    version: str,
    licence: str,
    doi: str,
    url: str,
    lat: float | None = None,
    lon: float | None = None,
    status: str = "downloaded",
) -> None:
    manifest.append(
        {
            "path": str(dest.relative_to(OUT)),
            "source": source,
            "version": version,
            "licence": licence,
            "doi": doi,
            "url": url,
            "latitude": lat,
            "longitude": lon,
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "status": status,
            "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download files already present on disk")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    t0 = time.time()
    REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
    print(f"envcov fetch started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"Target directory: {OUT.relative_to(REPO_ROOT)} (no API tokens used)")

    # ------------------------------------------------------------------
    # 1. NEON public site & instrument product metadata + unauthenticated /data/ check
    # ------------------------------------------------------------------
    neon_dir = OUT / "neon_public"
    neon_dir.mkdir(parents=True, exist_ok=True)
    tower_coords: dict[str, tuple[float, float]] = dict(FALLBACK_TOWER_COORDS)

    print("\n1. NEON public site metadata (/api/v0/sites/<site>, no credential)")
    for site in R14_INCLUDED_SITES + R14_OTHER_SITES:
        dest = neon_dir / f"site_{site}.json"
        url = f"{NEON_API}/sites/{site}"
        if dest.exists() and not args.force:
            d = json.loads(dest.read_bytes())["data"]
            st_label = "present"
        else:
            code, body, err = http_get(url, accept="application/json", pause=0.25)
            if code != 200 or not body:
                print(f"  WARN site {site}: status={code} {err}")
                continue
            save_atomic(dest, body)
            d = json.loads(body)["data"]
            st_label = "downloaded"
        tower_coords[site] = (float(d["siteLatitude"]), float(d["siteLongitude"]))
        record_manifest(
            manifest, dest, "NEON Public API Site Metadata", "v0",
            "CC0 1.0", "https://data.neonscience.org/data-policy", url,
            tower_coords[site][0], tower_coords[site][1], st_label,
        )
        if site in R14_INCLUDED_SITES:
            print(f"  {site}: tower=({tower_coords[site][0]:.5f}, {tower_coords[site][1]:.5f}) [{st_label}]")

    print("\n2. NEON terrestrial instrument & AOP product catalogues + unauthenticated /data/ check")
    probe_results = []
    for pcode, pname in NEON_PRODUCTS.items():
        dest = neon_dir / f"product_{pcode}.json"
        url = f"{NEON_API}/products/{pcode}"
        if dest.exists() and not args.force:
            pdata = json.loads(dest.read_bytes())["data"]
            st_label = "present"
        else:
            code, body, err = http_get(url, accept="application/json", pause=0.35)
            if code != 200 or not body:
                print(f"  WARN product {pcode}: status={code} {err}")
                continue
            save_atomic(dest, body)
            pdata = json.loads(body)["data"]
            st_label = "downloaded"
        releases = pdata.get("releases") or []
        rel_tag = releases[0].get("release", "PROVISIONAL") if releases else "PROVISIONAL"
        rel_doi = releases[0].get("productDoi", {}).get("url", "https://data.neonscience.org") if releases else "https://data.neonscience.org"
        record_manifest(
            manifest, dest, f"NEON Product Metadata ({pname})", rel_tag,
            "CC0 1.0", rel_doi, url, None, None, st_label,
        )
        avail = {
            s["siteCode"]: sorted(s["availableMonths"])
            for s in (pdata.get("siteCodes") or [])
            if s["siteCode"] in R14_INCLUDED_SITES
        }
        # Probe unauthenticated /data/ endpoint on first available R14 site-month (if any)
        probe_site = next((s for s in R14_INCLUDED_SITES if avail.get(s)), None)
        if probe_site:
            probe_month = avail[probe_site][len(avail[probe_site]) // 2]
            data_url = f"{NEON_API}/data/{pcode}/{probe_site}/{probe_month}?package=basic"
            d_code, d_body, d_err = http_get(data_url, accept="application/json", pause=0.35)
            probe_results.append(
                {
                    "productCode": pcode,
                    "productName": pname,
                    "release": rel_tag,
                    "doi": rel_doi,
                    "r14SiteMonths": {k: len(v) for k, v in avail.items()},
                    "probedSiteMonth": f"{probe_site}/{probe_month}",
                    "unauthenticatedHttpStatus": d_code,
                    "serverResponse": (d_err or d_body.decode("utf-8", "replace"))[:240],
                }
            )
            print(f"  {pcode} ({pname[:38]:38s}): R14 months={ {k: len(v) for k, v in avail.items()} } -> /data/ without credential: HTTP {d_code}")
        else:
            probe_results.append(
                {
                    "productCode": pcode,
                    "productName": pname,
                    "release": rel_tag,
                    "doi": rel_doi,
                    "r14SiteMonths": {},
                    "probedSiteMonth": None,
                    "unauthenticatedHttpStatus": None,
                    "serverResponse": "No site-months listed for the 6 R14 sites under this product code",
                }
            )
            print(f"  {pcode} ({pname[:38]:38s}): 0 months listed at R14 sites")

    probe_path = neon_dir / "unauthenticated_data_probe.json"
    save_atomic(probe_path, json.dumps(probe_results, indent=2).encode("utf-8"))
    record_manifest(
        manifest, probe_path, "NEON Unauthenticated /data/ Probe Log", "v0",
        "CC0 1.0", "https://data.neonscience.org/data-api/endpoints/data/",
        f"{NEON_API}/data/", None, None, "recorded",
    )

    # ------------------------------------------------------------------
    # 2. NASA ORNL DAAC Daymet V4 R1 daily 1-km surface weather (1990-2025)
    # ------------------------------------------------------------------
    print("\n3. NASA ORNL DAAC Daymet V4 R1 daily 1-km weather (1990-01-01 to 2025-12-31)")
    daymet_dir = OUT / "daymet"
    daymet_dir.mkdir(parents=True, exist_ok=True)
    tick_coords = compute_tick_plot_coords()

    daymet_jobs: list[tuple[str, float, float, str]] = [
        ("CARY", CARY_COORD[0], CARY_COORD[1], CARY_COORD[2]),
    ]
    for site in R14_INCLUDED_SITES:
        lat, lon, desc = tick_coords[site]
        daymet_jobs.append((f"{site}_tick", lat, lon, desc))
    for site in R14_INCLUDED_SITES + R14_OTHER_SITES:
        lat, lon = tower_coords[site]
        daymet_jobs.append((f"{site}_tower", lat, lon, f"NEON {site} official site coordinates"))

    for label, lat, lon, desc in daymet_jobs:
        dest = daymet_dir / f"{label}.csv"
        params = urllib.parse.urlencode(
            {
                "lat": f"{lat:.6f}",
                "lon": f"{lon:.6f}",
                "vars": "tmax,tmin,prcp,vp,swe,srad,dayl",
                "start": "1990-01-01",
                "end": "2025-12-31",
            }
        )
        url = f"{DAYMET_API}?{params}"
        if dest.exists() and dest.stat().st_size > 100_000 and not args.force:
            st_label = "present"
        else:
            code, body, err = http_get(url, timeout=60, pause=0.45)
            if code != 200 or len(body) < 10_000:
                raise SystemExit(f"Daymet download failed for {label}: HTTP {code} {err}")
            save_atomic(dest, body)
            st_label = "downloaded"
        record_manifest(
            manifest, dest, f"Daymet V4 R1 Daily Surface Weather ({desc})",
            "Version 4 R1 (Software 4.0)", "NASA Open Data / CC0-equivalent public domain",
            "https://doi.org/10.3334/ORNLDAAC/2129", url, lat, lon, st_label,
        )
        print(f"  {label:11s} ({lat:8.5f}, {lon:9.5f}) -> {dest.stat().st_size / 1e3:7.1f} KB [{st_label}]")

    # ------------------------------------------------------------------
    # 3. NASA POWER Agroclimatology Daily (RH2M, GWETTOP soil wetness, TS skin temp, 1990-2025)
    # ------------------------------------------------------------------
    print("\n4. NASA POWER Agroclimatology daily RH, surface soil wetness, and skin temperature (1990-2025)")
    power_dir = OUT / "nasa_power"
    power_dir.mkdir(parents=True, exist_ok=True)
    power_jobs = [("CARY", CARY_COORD[0], CARY_COORD[1])] + [
        (f"{s}_tick", tick_coords[s][0], tick_coords[s][1]) for s in R14_INCLUDED_SITES
    ]
    for label, lat, lon in power_jobs:
        dest = power_dir / f"{label}.json"
        params = urllib.parse.urlencode(
            {
                "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,TS",
                "community": "AG",
                "longitude": f"{lon:.5f}",
                "latitude": f"{lat:.5f}",
                "start": "19900101",
                "end": "20251231",
                "format": "JSON",
            }
        )
        url = f"{POWER_API}?{params}"
        if dest.exists() and dest.stat().st_size > 100_000 and not args.force:
            st_label = "present"
        else:
            code, body, err = http_get(url, accept="application/json", timeout=60, pause=0.6)
            if code != 200 or len(body) < 10_000:
                print(f"  WARN NASA POWER {label}: HTTP {code} {err}")
                continue
            save_atomic(dest, body)
            st_label = "downloaded"
        record_manifest(
            manifest, dest, f"NASA POWER Agroclimatology Daily ({label})",
            "POWER API v2 / MERRA-2", "NASA Open Data / US Government Public Domain",
            "https://power.larc.nasa.gov/docs/methodology/", url, lat, lon, st_label,
        )
        print(f"  {label:11s} ({lat:8.5f}, {lon:9.5f}) -> {dest.stat().st_size / 1e3:7.1f} KB [{st_label}]")

    # ------------------------------------------------------------------
    # 4. Optional ORNL DAAC MODIS/GEDI Subset REST availability check
    # ------------------------------------------------------------------
    print("\n5. Optional ORNL DAAC MODIS/GEDI Subset REST probe")
    opt_dest = OUT / "optional_remotesensing_probe.json"
    m_code, m_body, m_err = http_get(MODIS_RST_API, accept="application/json", timeout=20, retries=1, pause=0.2)
    opt_info = {
        "endpoint": MODIS_RST_API,
        "httpStatus": m_code,
        "usableWithoutAuthInReasonableTime": bool(m_code == 200 and len(m_body) > 0),
        "detail": (m_err or f"received {len(m_body)} bytes")[:240],
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_atomic(opt_dest, json.dumps(opt_info, indent=2).encode("utf-8"))
    record_manifest(
        manifest, opt_dest, "ORNL DAAC MODIS/GEDI Subset Availability Check", "v1",
        "US Government Public Domain", "https://modis.ornl.gov/", MODIS_RST_API, None, None, "recorded",
    )
    print(f"  ORNL MODIS/GEDI RST: status={m_code} ({opt_info['detail']})")

    manifest_path = OUT / "MANIFEST.json"
    save_atomic(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
    tot_bytes = sum(m["bytes"] for m in manifest)
    print(f"\nWrote {manifest_path.relative_to(REPO_ROOT)} ({len(manifest)} entries, {tot_bytes / 1e6:.2f} MB total) in {time.time() - t0:.1f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
