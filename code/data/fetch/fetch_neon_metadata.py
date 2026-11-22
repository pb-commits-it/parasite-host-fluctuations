#!/usr/bin/env python3
"""NEON (US National Ecological Observatory Network): fetch OPEN product metadata; data need a credential.

Run:  python3 -I data/fetch/fetch_neon_metadata.py            # metadata only (open)
      NEON_ACCESS_CREDENTIAL=... python3 -I data/fetch/fetch_neon_metadata.py --data HARV SCBI   # needs a credential

STATUS 2026-11-14
  Open without a credential : GET https://data.neonscience.org/api/v0/products/<code>  and /sites/<code>
                         (product description, sites, months with data per site, releases, DOIs).
  Token required       : every /data/... endpoint (file listings and file downloads). Without a
                         credential they return HTTP 403 {"error":{"status":403,"detail":"Access Denied"}};
                         the API documentation (https://data.neonscience.org/data-api/endpoints/data/)
                         states "API Token required to utilize this endpoint". A credential is free but
                         needs a registered NEON account, so no NEON observations were downloaded.

What this script does
  default : saves the product JSON for the six vector/host/pathogen products and writes
            availability.csv (product, site, first/last year, number of years and months with data).
  --data  : ONLY if the environment variable NEON_ACCESS_CREDENTIAL is set (never stored in this file), walks
            /data/<product>/<site>/<month>?package=basic for the given sites and downloads the CSVs
            into raw/neon/<product>/<site>/. Without NEON_ACCESS_CREDENTIAL it stops with a message.

Licence: NEON data are CC0 1.0; NEON asks users to cite the product DOI of the release used
         (e.g. RELEASE-2026, small mammals https://doi.org/... given in each product JSON under
         releases[].productDoi). Standard library only.
"""
import csv
import json
import os
import sys
import time
import urllib.request

UA = "data-scout/1.0 (python-urllib; research data inventory)"
API = "https://data.neonscience.org/api/v0"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/neon').resolve())
PRODUCTS = {
    "DP1.10072.001": "Small mammal box trapping",
    "DP1.10093.001": "Ticks sampled using drag cloths",
    "DP1.10092.001": "Tick pathogen status",
    "DP1.10064.002": "Rodent pathogen status, tick-borne",
    "DP1.10064.001": "Rodent pathogen status, hantavirus",
    "DP1.10043.001": "Mosquitoes sampled from CO2 traps",
    "DP1.10041.001": "Mosquito pathogen status",
}


def get(url, credential=None, tries=3):
    last = None
    for i in range(tries):
        try:
            hdr = {"User-Agent": UA, "Accept": "application/json"}
            if credential:
                hdr["X-API-Token"] = credential
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * i)
    raise SystemExit("FAILED %s: %s" % (url, last))


def metadata():
    os.makedirs(os.path.join(OUT, "metadata"), exist_ok=True)
    rows = []
    for code in PRODUCTS:
        raw = get("%s/products/%s" % (API, code))
        with open(os.path.join(OUT, "metadata", code + ".json"), "wb") as f:
            f.write(raw)
        d = json.loads(raw)["data"]
        for s in d.get("siteCodes") or []:
            months = sorted(s["availableMonths"])
            years = sorted({m[:4] for m in months})
            rows.append([code, d["productName"], s["siteCode"], years[0], years[-1], len(years), len(months), " ".join(months)])
        print("%s  %-36s %2d sites" % (code, d["productName"], len(d.get("siteCodes") or [])))
        time.sleep(0.6)
    with open(os.path.join(OUT, "metadata", "availability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["productCode", "productName", "siteCode", "firstYear", "lastYear", "nYears", "nMonths", "months"])
        w.writerows(rows)
    print("wrote", os.path.join(OUT, "metadata", "availability.csv"))


def data(sites):
    credential = os.environ.get("NEON_ACCESS_CREDENTIAL")
    if not credential:
        raise SystemExit("NEON /data endpoints need an API credential (HTTP 403 without one). "
                         "Set NEON_ACCESS_CREDENTIAL in the environment; nothing downloaded.")
    for code in PRODUCTS:
        prod = json.loads(get("%s/products/%s" % (API, code)))["data"]
        for s in prod.get("siteCodes") or []:
            if s["siteCode"] not in sites:
                continue
            for month in sorted(s["availableMonths"]):
                listing = json.loads(get("%s/data/%s/%s/%s?package=basic" % (API, code, s["siteCode"], month), credential))
                for fl in listing["data"]["files"]:
                    if not fl["name"].endswith(".csv"):
                        continue
                    dest = os.path.join(OUT, code, s["siteCode"], fl["name"])
                    if os.path.exists(dest):
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(get(fl["url"]))
                time.sleep(0.2)
            print("done", code, s["siteCode"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--data":
        sys.exit(data(set(sys.argv[2:])))
    sys.exit(metadata())
