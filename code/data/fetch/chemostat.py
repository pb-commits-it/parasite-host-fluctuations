#!/usr/bin/env python3
"""Fetch open bacteria-phage chemostat and serial-transfer time-series datasets.

Study id: chemostat
Target directory: data/raw/chemostat/ (git-ignored)
Run with: python3 -I data/fetch/chemostat.py

Rules:
- Legitimately open datasets only (CC0 1.0 or CC BY 4.0).
- Standard library only; generic User-Agent; no credentials, tokens, or personal details.
- Re-runnable: skips files already present on disk when byte size matches, and records
  SHA-256, licence, version, DOI, URL, and UTC access date in MANIFEST.json.

Datasets fetched:
1. larsen2019_synechococcus_chemostat
   Larsen ML, Wilhelm SW, Lennon JT (2019) Nutrient stoichiometry shapes microbial
   coevolution. Ecology Letters 22:1009-1018, doi:10.1111/ele.13252.
   Deposit: Figshare article 7742819, doi:10.6084/m9.figshare.7742819.v3, version 3, CC BY 4.0.
2. reuter2025_phix174_serial_transfer
   Reuter M et al. (2026) Single capsid mutations modulating phage adsorption, persistence,
   and plaque morphology shape evolutionary trajectories in PhiX174. Virus Evolution,
   doi:10.1093/ve/veag022.
   Deposit: Zenodo record 17961407, doi:10.5281/zenodo.17961407, version v2, CC BY 4.0.
   (Fetches Data_sets_Phage_titers.zip and experiment_8_phage_titer_WT_30_180.csv; skips
   raw Sanger sequencing and raw plaque image archives.)
3. chaudhry2018_lambda_serial_transfer
   Chaudhry WN et al. (2018) Leaky resistance and the conditions for the existence of lytic
   bacteriophage. PLOS Biology 16(8):e2005971, doi:10.1371/journal.pbio.2005971.
   Deposit: Figshare article 6974783, doi:10.1371/journal.pbio.2005971, version 1, CC BY 4.0.
   (Fetches supplementary data workbooks pbio.2005971.s007.xlsx and pbio.2005971.s008.xlsx;
   skips large TIFF figures.)
4. duncan2017_sbw25_temp_fluctuation
   Duncan AB et al. (2017) Hot spots become cold spots: coevolution in variable temperature
   environments. J. Evol. Biol. 30:55-65, doi:10.1111/jeb.12985.
   Deposit: Zenodo record 4954973 (official mirror of Dryad doi:10.5061/dryad.vv29r),
   version 1, CC0 1.0.
5. harrison2013_sbw25_nutrient_pulses
   Harrison E, Laine A-L, Hietala M, Brockhurst MA (2013) Rapidly fluctuating environments
   constrain coevolutionary arms races by impeding selective sweeps. Proc. R. Soc. B
   280:20130937, doi:10.1098/rspb.2013.0937.
   Deposit: Zenodo record 4957209 (official mirror of Dryad doi:10.5061/dryad.8k015),
   version 1, CC0 1.0.
6. zhao2016_sbw25_productivity
   Zhao X, Hao Y, Zhang Q-G (2016) Stability of a coevolving host-parasite system peaks at
   intermediate productivity. PLOS ONE 11(12):e0168560, doi:10.1371/journal.pone.0168560.
   Deposit: Zenodo record 5012658 (official mirror of Dryad doi:10.5061/dryad.73s68),
   version 1, CC0 1.0.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request

DATASET = "chemostat"
UA = "data-scout/1.0 (python-urllib; open-data research fetch)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/chemostat').resolve())


def get_bytes(url: str, tries: int = 3, timeout: int = 180) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + 3 * i)
    raise SystemExit(f"FAILED {url}: {last}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_file(url: str, relpath: str, expected_size: int | None, meta: dict, manifest: list) -> None:
    dest = os.path.join(OUT, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and (expected_size is None or os.path.getsize(dest) == expected_size):
        size = os.path.getsize(dest)
        digest = sha256_file(dest)
        status = "cached"
    else:
        data = get_bytes(url)
        with open(dest, "wb") as f:
            f.write(data)
        size = len(data)
        digest = hashlib.sha256(data).hexdigest()
        status = "downloaded"
        time.sleep(0.4)
    entry = {
        "path": relpath,
        "url": url,
        "bytes": size,
        "sha256": digest,
        "downloaded": True,
        "status": status,
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    entry.update(meta)
    manifest.append(entry)
    print(f"{size:10d}  {relpath}  ({status})")


def fetch_zenodo(
    record_id: str,
    subdir: str,
    doi: str,
    paper_doi: str,
    manifest: list,
    only: set[str] | None = None,
    skip_reason: str = "non-tabular archive not needed for density analysis",
) -> None:
    raw = get_bytes(f"https://zenodo.org/api/records/{record_id}")
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    with open(os.path.join(OUT, subdir, "_zenodo_metadata.json"), "wb") as f:
        f.write(raw)
    rec = json.loads(raw.decode("utf-8"))
    md = rec.get("metadata", {})
    lic = md.get("license", {}).get("id") if isinstance(md.get("license"), dict) else str(md.get("license"))
    version = str(md.get("version") or rec.get("revision") or "1")
    common = {
        "repository": "Zenodo",
        "record_id": record_id,
        "dataset_doi": rec.get("doi") or doi,
        "primary_doi": doi,
        "paper_doi": paper_doi,
        "licence": lic,
        "version": version,
    }
    for fl in rec.get("files", []):
        name = fl["key"]
        safe_name = name.replace("/", "_")
        rel = os.path.join(subdir, safe_name)
        url = fl["links"]["self"]
        sz = int(fl["size"])
        if only is not None and name not in only:
            entry = {
                "path": rel,
                "url": url,
                "bytes": sz,
                "downloaded": False,
                "reason_skipped": skip_reason,
                "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            entry.update(common)
            manifest.append(entry)
            print(f"{sz:10d}  {rel}  (skipped: {skip_reason})")
            continue
        save_file(url, rel, sz, common, manifest)


def fetch_figshare(
    article_id: str,
    subdir: str,
    paper_doi: str,
    manifest: list,
    only: set[str] | None = None,
    skip_reason: str = "image or code file not needed for density analysis",
) -> None:
    raw = get_bytes(f"https://api.figshare.com/v2/articles/{article_id}")
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    with open(os.path.join(OUT, subdir, "_figshare_metadata.json"), "wb") as f:
        f.write(raw)
    art = json.loads(raw.decode("utf-8"))
    lic = (art.get("license") or {}).get("name", "unknown")
    version = str(art.get("version") or "1")
    doi = art.get("doi") or ""
    common = {
        "repository": "Figshare",
        "article_id": article_id,
        "dataset_doi": doi,
        "primary_doi": doi,
        "paper_doi": paper_doi,
        "licence": lic,
        "version": version,
    }
    for fl in art.get("files", []):
        name = fl["name"]
        safe_name = name.replace("/", "_")
        rel = os.path.join(subdir, safe_name)
        url = fl["download_url"]
        sz = int(fl["size"])
        if only is not None and name not in only:
            entry = {
                "path": rel,
                "url": url,
                "bytes": sz,
                "downloaded": False,
                "reason_skipped": skip_reason,
                "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            entry.update(common)
            manifest.append(entry)
            print(f"{sz:10d}  {rel}  (skipped: {skip_reason})")
            continue
        save_file(url, rel, sz, common, manifest)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    manifest: list[dict] = []

    # 1. Larsen, Wilhelm & Lennon (2019) Synechococcus - S-RIM8 chemostats (Figshare 7742819, CC BY 4.0)
    fetch_figshare(
        article_id="7742819",
        subdir="larsen2019_synechococcus_chemostat",
        paper_doi="10.1111/ele.13252",
        manifest=manifest,
        only={
            "README.md",
            "ts-counts.csv",
            "ts-counts_raw.csv",
            "cid-means.csv",
            "cstat-means.csv",
            "ccf-dataset-sas-abundances.csv",
            "inf-mat.csv",
            "PercChange.csv",
        },
        skip_reason="analysis script or processed web archive not needed",
    )

    # 2. Reuter et al. (2026) E. coli C - PhiX174 serial transfers (Zenodo 17961407, CC BY 4.0)
    fetch_zenodo(
        record_id="17961407",
        subdir="reuter2025_phix174_serial_transfer",
        doi="10.5281/zenodo.17961407",
        paper_doi="10.1093/ve/veag022",
        manifest=manifest,
        only={
            "Data_sets_Phage_titers.zip",
            "experiment_8_phage_titer_WT_30_180.csv",
        },
        skip_reason="raw sequencing (.ab1) or plaque image archive not needed",
    )

    # 3. Chaudhry et al. (2018) E. coli B - lambdaVIR serial transfers (Figshare 6974783, CC BY 4.0)
    fetch_figshare(
        article_id="6974783",
        subdir="chaudhry2018_lambda_serial_transfer",
        paper_doi="10.1371/journal.pbio.2005971",
        manifest=manifest,
        only={
            "pbio.2005971.s007.xlsx",
            "pbio.2005971.s008.xlsx",
        },
        skip_reason="high-resolution TIFF figure image not needed",
    )

    # 4. Duncan et al. (2017) P. fluorescens SBW25 - phi2 fluctuating temperature (Zenodo 4954973, CC0)
    fetch_zenodo(
        record_id="4954973",
        subdir="duncan2017_sbw25_temp_fluctuation",
        doi="10.5061/dryad.vv29r",
        paper_doi="10.1111/jeb.12985",
        manifest=manifest,
    )

    # 5. Harrison et al. (2013) P. fluorescens SBW25 - phi2 nutrient pulses (Zenodo 4957209, CC0)
    fetch_zenodo(
        record_id="4957209",
        subdir="harrison2013_sbw25_nutrient_pulses",
        doi="10.5061/dryad.8k015",
        paper_doi="10.1098/rspb.2013.0937",
        manifest=manifest,
    )

    # 6. Zhao, Hao & Zhang (2016) P. fluorescens SBW25 - phi2 productivity (Zenodo 5012658, CC0)
    fetch_zenodo(
        record_id="5012658",
        subdir="zhao2016_sbw25_productivity",
        doi="10.5061/dryad.73s68",
        paper_doi="10.1371/journal.pone.0168560",
        manifest=manifest,
    )

    manifest_path = os.path.join(OUT, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
