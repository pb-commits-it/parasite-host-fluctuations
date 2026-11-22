#!/usr/bin/env python3
"""Fetch legitimately open field-vole, bank-vole, and small-mammal time-series datasets
for case study `voles`.

Run:
    python3 -I data/fetch/voles.py

Rules followed (CASESTUDY_SPEC.md Rule 2 & study rules):
  - Legitimately open data only; generic User-Agent; no e-mail address, API key,
    or personal details in any request.
  - Standard library only; re-runnable (skips existing files matching byte count
    unless --force is passed).
  - Stores raw files under `data/raw/voles/` (git-ignored) with `MANIFEST.json`
    recording URL, licence, version, DOI, byte count, sha256, and access date.
  - Also records open repositories checked where raw time-series files were
    unavailable or metadata-only (KNB `doi:10.5063/AA/knb.249.1` Kielder 2001-2007;
    Luis et al. 2015 Figshare collection 3307761 HTML figure supplement).

Datasets fetched:
  1. `kielder_eidc_2015_2017/` — Paterson, Jackson, Jackson, Bradley, Begon,
     Wanelik & Taylor (2022). Data on immunological expression and phenotypes in
     a natural population of field voles in Kielder Forest, UK 2015-2017. NERC
     EDS Environmental Information Data Centre.
     DOI: 10.5285/e5854431-6fa4-4ff0-aa02-3de68763c952
     Licence: UK Open Government Licence v3.0 (OGL).
  2. `kielder_jackson_2008_2010/` — Jackson et al. (2014). Data from: An
     immunological marker of tolerance to infection in wild rodents (Kielder
     Forest field voles 2008-2010: cowpox, Bartonella, Babesia, fleas, ticks).
     Dryad DOI: 10.5061/dryad.bk537 (via official Zenodo `dryad` mirror 4971362).
     Licence: CC0 1.0.
  3. `kielder_arriero_2018/` — Arriero, Wanelik, Birtles, Bradley, Jackson,
     Paterson & Begon (2018). Data from: From the animal house to the field: are
     there consistent individual differences in immunological profile in wild
     populations of field voles (Microtus agrestis)? Dryad DOI: 10.5061/dryad.110pk
     (via official Zenodo `dryad` mirror 4941512). Licence: CC0 1.0.
  4. `finland_puumala_2002_2009/` — Voutilainen, Kallio, Niemimaa, Vapalahti &
     Henttonen (2016). Data from: Temporal dynamics of Puumala hantavirus
     infection in cyclic populations of bank voles (Konnevesi, Finland, 60
     trapping sessions 2002-2009). Dryad DOI: 10.5061/dryad.g8140 (via official
     Zenodo `dryad` mirror 4941743). Licence: CC0 1.0.
  5. `finland_puumala_wang_2023/` — Wang et al. (2023). Data for: The impact of
     wildlife and environmental factors on hantavirus infection in host and its
     translation into human risk (30 areas in Finland). Dryad DOI:
     10.5061/dryad.bg79cnpfh (via Zenodo `dryad` mirror 7796298). Licence: CC0 1.0.
  6. `luis_2015_snv_appendix/` — Luis, Douglass, Mills & Bjornstad (2015).
     Environmental fluctuations lead to predictability in Sin Nombre hantavirus
     outbreaks: Appendix A (Ecological Archives E096-151-A1; Figshare article
     3562431 in collection 3307761). DOI: 10.6084/m9.figshare.3562431.v1.
     Licence: CC0 1.0. (Contains supplemental figures only; raw deer-mouse counts
     were not deposited.)
  7. `weather/` — Open daily station and gridded meteorological records matched to
     Kielder Forest, UK (55.2300 N, 2.5750 W; 2007-2017) and Konnevesi, Finland
     (62.6280 N, 26.2880 E; 2001-2009):
       - NOAA GHCN-Daily stations UKE00105892 (Kielder Castle, 0.5 km), UK000003162
         (Eskdalemuir, 40.7 km), FIE00144637 (Vesanto Sonkari, 33.4 km),
         FI000002401 (Jyvaskyla Lentoasema, 40.1 km), and FIE00144657 (Konnevesi
         Lahdenkyla, 6.8 km). DOI: 10.7289/V5D21VHZ (U.S. Gov / Public Domain).
       - NASA POWER Agroclimatology daily point series (T2M, T2M_MAX, T2M_MIN,
         PRECTOTCORR, RH2M) for Kielder and Konnevesi.
         DOI: 10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY (Public Domain).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

UA = "data-scout/1.0 (python-urllib; open-ecology-casestudy)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT_DIR = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/voles').resolve())


def fetch_bytes(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def save_file(
    url: str,
    relpath: str,
    manifest: list[dict],
    doi: str,
    licence: str,
    version: str,
    force: bool = False,
) -> bytes:
    dest = os.path.join(OUT_DIR, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
        with open(dest, "rb") as f:
            data = f.read()
        action = "cached"
    else:
        data = fetch_bytes(url)
        with open(dest, "wb") as f:
            f.write(data)
        action = "downloaded"
        time.sleep(0.3)

    entry = {
        "path": relpath,
        "url": url,
        "doi": doi,
        "licence": licence,
        "version": version,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": action,
    }
    manifest.append(entry)
    print(f"{len(data):10d}  [{action:10s}]  {relpath}")
    return data


def fetch_zenodo_record(
    record_id: str,
    subdir: str,
    doi: str,
    licence: str,
    manifest: list[dict],
    force: bool = False,
) -> None:
    meta_url = f"https://zenodo.org/api/records/{record_id}"
    meta_raw = save_file(
        meta_url,
        os.path.join(subdir, "_zenodo_metadata.json"),
        manifest,
        doi=doi,
        licence=licence,
        version=f"zenodo:{record_id}",
        force=force,
    )
    meta = json.loads(meta_raw.decode("utf-8", "replace"))
    ver = str(meta.get("metadata", {}).get("version", f"zenodo:{record_id}"))
    for fl in meta.get("files", []):
        fname = fl["key"].replace("/", "_")
        furl = fl["links"]["self"]
        save_file(
            furl,
            os.path.join(subdir, fname),
            manifest,
            doi=doi,
            licence=licence,
            version=ver,
            force=force,
        )


def fetch_eidc_kielder(manifest: list[dict], force: bool = False) -> None:
    subdir = "kielder_eidc_2015_2017"
    doi = "10.5285/e5854431-6fa4-4ff0-aa02-3de68763c952"
    lic = "UK Open Government Licence v3.0 (https://eidc.ac.uk/licences/ogl/plain)"
    save_file(
        "https://catalogue.ceh.ac.uk/id/e5854431-6fa4-4ff0-aa02-3de68763c952?format=json",
        os.path.join(subdir, "_eidc_metadata.json"),
        manifest,
        doi=doi,
        licence=lic,
        version="2022-03-25",
        force=force,
    )
    zip_bytes = save_file(
        "https://data-package.ceh.ac.uk/data/e5854431-6fa4-4ff0-aa02-3de68763c952.zip",
        os.path.join(subdir, "e5854431-6fa4-4ff0-aa02-3de68763c952.zip"),
        manifest,
        doi=doi,
        licence=lic,
        version="2022-03-25",
        force=force,
    )
    # Extract zip contents safely inside subdir/extracted/
    extract_root = os.path.join(OUT_DIR, subdir, "extracted")
    os.makedirs(extract_root, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            safe_rel = os.path.normpath(info.filename).lstrip(os.sep)
            if safe_rel.startswith(".."):
                continue
            target = os.path.join(extract_root, safe_rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            content = zf.read(info.filename)
            with open(target, "wb") as out_f:
                out_f.write(content)


def fetch_luis_appendix(manifest: list[dict], force: bool = False) -> None:
    subdir = "luis_2015_snv_appendix"
    doi = "10.6084/m9.figshare.3562431.v1"
    lic = "CC0 1.0"
    meta_raw = save_file(
        "https://api.figshare.com/v2/articles/3562431",
        os.path.join(subdir, "_figshare_metadata.json"),
        manifest,
        doi=doi,
        licence=lic,
        version="v1",
        force=force,
    )
    meta = json.loads(meta_raw.decode("utf-8", "replace"))
    for fl in meta.get("files", []):
        save_file(
            fl["download_url"],
            os.path.join(subdir, fl["name"].replace("/", "_")),
            manifest,
            doi=doi,
            licence=lic,
            version="v1",
            force=force,
        )


def record_unavailable_deposits(manifest: list[dict]) -> None:
    """Record metadata-only or restricted deposits checked during the search."""
    # Check KNB knb.249.1 EML metadata (public XML metadata, no data entity attached)
    knb_url = "https://knb.ecoinformatics.org/knb/d1/mn/v2/object/doi%3A10.5063%2FAA%2Fknb.249.1"
    try:
        save_file(
            knb_url,
            "knb_249_1_metadata_only/knb.249.1.xml",
            manifest,
            doi="10.5063/AA/knb.249.1",
            licence="Obtain permission from data set owner(s) — metadata only, zero data files attached",
            version="1",
        )
    except Exception as exc:  # noqa: BLE001
        manifest.append({
            "path": "knb_249_1_metadata_only/knb.249.1.xml",
            "url": knb_url,
            "doi": "10.5063/AA/knb.249.1",
            "status": f"not_downloaded: {exc}",
        })


def fetch_weather(manifest: list[dict], force: bool = False) -> None:
    """Fetch open daily meteorological station (NOAA GHCN-Daily) and gridded
    (NASA POWER) records matched to Kielder Forest (UK) and Konnevesi (Finland).
    """
    ghcn_jobs = [
        ("KielderCastle_UKE00105892_2007_2017", "UKE00105892", "2007-01-01", "2017-12-31"),
        ("Eskdalemuir_UK000003162_2007_2017", "UK000003162", "2007-01-01", "2017-12-31"),
        ("VesantoSonkari_FIE00144637_2001_2009", "FIE00144637", "2001-01-01", "2009-12-31"),
        ("Jyvaskyla_FI000002401_2001_2009", "FI000002401", "2001-01-01", "2009-12-31"),
        ("KonnevesiLahdenkyla_FIE00144657_2001_2007", "FIE00144657", "2001-01-01", "2007-12-31"),
    ]
    for label, stn, d0, d1 in ghcn_jobs:
        q = urllib.parse.urlencode({
            "dataset": "daily-summaries",
            "stations": stn,
            "startDate": d0,
            "endDate": d1,
            "dataTypes": "TMAX,TMIN,TAVG,PRCP,SNWD",
            "units": "metric",
            "format": "csv",
        })
        url = f"https://www.ncei.noaa.gov/access/services/data/v1?{q}"
        save_file(
            url,
            os.path.join("weather", f"ghcnd_{label}.csv"),
            manifest,
            doi="10.7289/V5D21VHZ",
            licence="U.S. Government Work / Public Domain (NOAA GHCN-Daily)",
            version="v1",
            force=force,
        )

    power_jobs = [
        ("Kielder_55.23N_2.575W_2007_2017", 55.2300, -2.5750, "20070101", "20171231"),
        ("Konnevesi_62.628N_26.288E_2001_2009", 62.6280, 26.2880, "20010101", "20091231"),
    ]
    for label, lat, lon, d0, d1 in power_jobs:
        q = urllib.parse.urlencode({
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
            "community": "AG",
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "start": d0,
            "end": d1,
            "format": "JSON",
        })
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?{q}"
        save_file(
            url,
            os.path.join("weather", f"nasapower_{label}.json"),
            manifest,
            doi="10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY",
            licence="NASA Open Data / Public Domain",
            version="v2.0",
            force=force,
        )


def main() -> int:
    force = "--force" in sys.argv[1:]
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest: list[dict] = []

    fetch_eidc_kielder(manifest, force=force)
    fetch_zenodo_record(
        "4971362",
        "kielder_jackson_2008_2010",
        doi="10.5061/dryad.bk537",
        licence="CC0 1.0",
        manifest=manifest,
        force=force,
    )
    fetch_zenodo_record(
        "4941512",
        "kielder_arriero_2018",
        doi="10.5061/dryad.110pk",
        licence="CC0 1.0",
        manifest=manifest,
        force=force,
    )
    fetch_zenodo_record(
        "4941743",
        "finland_puumala_2002_2009",
        doi="10.5061/dryad.g8140",
        licence="CC0 1.0",
        manifest=manifest,
        force=force,
    )
    fetch_zenodo_record(
        "7796298",
        "finland_puumala_wang_2023",
        doi="10.5061/dryad.bg79cnpfh",
        licence="CC0 1.0",
        manifest=manifest,
        force=force,
    )
    fetch_luis_appendix(manifest, force=force)
    record_unavailable_deposits(manifest)
    fetch_weather(manifest, force=force)

    manifest_path = os.path.join(OUT_DIR, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest ({len(manifest)} items) to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
