#!/usr/bin/env python3
"""Fetch the open Cary Institute of Ecosystem Studies (Millbrook, NY) Lyme-system data.

Run:  python3 -I data/fetch/fetch_cary_lyme.py

Four open deposits (no login, no credential), all from the same six 2.25-ha
mark-recapture grids / tick-drag plots on the Cary Institute property:

  A. Figshare 31872853 (CC BY 4.0)  - Ostfeld et al. 2026, plot-by-year table 1991/1995-2025
  B. Figshare 27871269 (CC BY-NC 4.0) - LaDeau & Ostfeld 2025, per-tick nymphal infection + grid-year table
  C. Figshare 31915557 (CC BY 4.0)  - Ostfeld et al. 2026, rodent/acorn/weather lagged table
  D. Dryad doi:10.5061/dryad.d1c8046 via Zenodo 5006160 (CC0) - Ostfeld et al. 2018 Ecology, 19-yr table

File lists are read from the repository APIs at run time, so new versions are picked up;
a MANIFEST.json with URL, size, sha256 and fetch time is written next to the files.
Standard library only.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

UA = "data-scout/1.0 (python-urllib; research data inventory)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/cary_lyme').resolve())

FIGSHARE = {
    "figshare_31872853_ecology_of_lyme_2026": 31872853,
    "figshare_27871269_nymphal_infection_2025": 27871269,
    "figshare_31915557_rodents_acorns_2026": 31915557,
}
ZENODO = {"dryad_d1c8046_forest_food_web_2018": 5006160}


def get(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * i)
    raise SystemExit("FAILED %s: %s" % (url, last))


def save(url, dest, manifest):
    data = get(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    manifest.append({
        "path": os.path.relpath(dest, OUT), "url": url, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print("%9d  %s" % (len(data), os.path.relpath(dest, OUT)))


def main():
    manifest = []
    for sub, art in FIGSHARE.items():
        meta_raw = get("https://api.figshare.com/v2/articles/%d" % art)
        meta = json.loads(meta_raw)
        d = os.path.join(OUT, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "_figshare_metadata.json"), "wb") as f:
            f.write(meta_raw)
        for fl in meta["files"]:
            save(fl["download_url"], os.path.join(d, fl["name"].replace("/", "_")), manifest)
    for sub, rec in ZENODO.items():
        meta_raw = get("https://zenodo.org/api/records/%d" % rec)
        meta = json.loads(meta_raw)
        d = os.path.join(OUT, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "_zenodo_metadata.json"), "wb") as f:
            f.write(meta_raw)
        for fl in meta["files"]:
            save(fl["links"]["self"], os.path.join(d, fl["key"].replace("/", "_")), manifest)
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(OUT, "MANIFEST.json"))


if __name__ == "__main__":
    sys.exit(main())
