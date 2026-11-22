#!/usr/bin/env python3
"""Fetch verified laboratory environmental-regime records and open thermal-trait curves
for the `chemostat` bacteria-phage case study (`r27-weather-chemostat`).

Study id: chemostat
Target directory: data/raw/chemostat/weather_env/ (git-ignored under data/raw/)
Run with: python3 -I data/fetch/chemostat-weather.py

Scope and rationale (`CASESTUDY_SPEC.md`, "Weather and environment at the place and time of the data"):
- Field records in `chemostat`: ZERO (0). All six deposits (D1-D6) come from indoor,
  temperature-controlled laboratory chemostats or incubator microcosms. Outdoor weather
  grids and stations (NEON, Daymet, TerraClimate, NOAA SST, GHCN-Daily, ERA5) do not
  govern indoor thermostatted incubators and are therefore not queried.
- Per the laboratory rule in `CASESTUDY_SPEC.md` ("Laboratory studies: use the
  laboratory's reported temperature regime instead and say so"), this script fetches
  and archives the open-access primary Methods texts / metadata verifying the exact
  reported temperature, light/photoperiod, agitation, and medium regimes for D1-D6,
  together with open empirical thermal-response data and Arrhenius / Sharpe-Schoolfield
  references for the experimental host-phage systems:
    * D1 (Larsen, Wilhelm & Lennon 2019, Ecol. Lett. 10.1111/ele.13252): open author-hosted
      paper & supplement from LennonLab public repository + PubMed XML (PMID 30924583).
    * D2 (Reuter et al. 2026, Virus Evol. 10.1093/ve/veag022): Europe PMC OA JATS XML
      (PMC13108444, CC BY 4.0).
    * D3 (Chaudhry et al. 2018, PLOS Biol. 10.1371/journal.pbio.2005971): Europe PMC OA
      JATS XML (PMC6112682, CC BY 4.0).
    * D4 (Duncan et al. 2017, J. Evol. Biol. 10.1111/jeb.12985): PubMed XML (PMID 27711983)
      + HAL open bibliographic record (hal-01530834); replicate-level thermal schedules
      (28 C vs 32 C at switching periods 0, 2, 4, 8 d) are already archived in CC0
      `data/raw/chemostat/duncan2017_sbw25_temp_fluctuation/Duncan_JEB.xlsx` (publisher
      PDF returned HTTP 403 and is never circumvented per Rule 2).
    * D5 (Zhao, Hao & Zhang 2016, PLOS ONE 10.1371/journal.pone.0168560): Europe PMC OA
      JATS XML (PMC5226335, CC BY 4.0).
    * D6 (Harrison et al. 2013, Proc. R. Soc. B 10.1098/rspb.2013.0937): Europe PMC OA
      JATS XML (PMC3712419, CC BY 3.0).
    * Thermal trait source T1 (Padfield, Castledine & Buckling 2020, ISME J.
      10.1038/s41396-019-0526-5; open archive Zenodo 3492192, doi:10.5281/zenodo.3492192,
      GPL-3): independent empirical 4-h replication curve of ancestral phage SBW25phi2
      on P. fluorescens SBW25 across 15-37 C (`phage_replication.csv`, `bacteria_phage_tpcs.R`).
    * Thermal trait source T2 (De Paepe & Taddei 2006, PLOS Biol. 10.1371/journal.pbio.0040193,
      Europe PMC OA JATS XML PMC1475768, CC BY): Arrhenius virion thermal inactivation
      activation energies (90-150 kJ/mol) across lytic coliphages (including PhiX174 and lambda).

Rules:
- Standard library only; generic User-Agent; no personal details, tokens, or credentials.
- Re-runnable: caches existing files and writes SHA-256, licence, version, DOI, URL, and
  UTC access timestamp to `data/raw/chemostat/weather_env/WEATHER_ENV_MANIFEST.json`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request

UA = "data-scout/1.0 (python-urllib; open-data research fetch)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT_DIR = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/chemostat/weather_env').resolve())


def get_bytes(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    last_exc = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 + 2.0 * attempt)
    raise SystemExit(f"FAILED {url}: {last_exc}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_item(url: str, filename: str, meta: dict, manifest_items: list[dict]) -> None:
    dest = os.path.join(OUT_DIR, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        size = os.path.getsize(dest)
        digest = sha256_file(dest)
        status = "cached"
    else:
        raw = get_bytes(url)
        with open(dest, "wb") as f:
            f.write(raw)
        size = len(raw)
        digest = hashlib.sha256(raw).hexdigest()
        status = "downloaded"
        time.sleep(0.35)
    rec = {
        "file": filename,
        "url": url,
        "bytes": size,
        "sha256": digest,
        "status": status,
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec.update(meta)
    manifest_items.append(rec)
    print(f"{size:10d}  {filename}  ({status})")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    items: list[dict] = []

    sources = [
        # D1: Larsen, Wilhelm & Lennon (2019) Ecology Letters 22:1009
        (
            "https://raw.githubusercontent.com/jaytlennon/lennonlab.github.io/main/assets/publications/Larsen_etal_2019.pdf",
            "D1_Larsen_etal_2019.pdf",
            {
                "dataset_id": "D1_larsen2019",
                "paper_doi": "10.1111/ele.13252",
                "source_type": "author_open_repository_pdf",
                "repository": "GitHub (jaytlennon/lennonlab.github.io)",
                "licence": "open author archive (data CC BY 4.0 on Figshare 7742819 v3)",
                "version": "published_2019",
            },
        ),
        (
            "https://raw.githubusercontent.com/jaytlennon/lennonlab.github.io/main/assets/publications/Larsen_etal_2019_Suppl.pdf",
            "D1_Larsen_etal_2019_Suppl.pdf",
            {
                "dataset_id": "D1_larsen2019",
                "paper_doi": "10.1111/ele.13252",
                "source_type": "author_open_repository_supplement_pdf",
                "repository": "GitHub (jaytlennon/lennonlab.github.io)",
                "licence": "open author archive (data CC BY 4.0 on Figshare 7742819 v3)",
                "version": "published_2019",
            },
        ),
        (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=30924583&retmode=xml",
            "D1_pubmed_30924583.xml",
            {
                "dataset_id": "D1_larsen2019",
                "paper_doi": "10.1111/ele.13252",
                "source_type": "pubmed_bibliographic_xml",
                "repository": "NCBI PubMed",
                "licence": "public_metadata",
                "version": "PMID_30924583",
            },
        ),
        # D2: Reuter et al. (2026) Virus Evolution veag022
        (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13108444/fullTextXML",
            "D2_Reuter2026_PMC13108444.xml",
            {
                "dataset_id": "D2_reuter2026",
                "paper_doi": "10.1093/ve/veag022",
                "source_type": "europepmc_oa_jats_xml",
                "repository": "Europe PMC",
                "licence": "CC BY 4.0",
                "version": "PMC13108444",
            },
        ),
        # D3: Chaudhry et al. (2018) PLOS Biology e2005971
        (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6112682/fullTextXML",
            "D3_Chaudhry2018_PMC6112682.xml",
            {
                "dataset_id": "D3_chaudhry2018",
                "paper_doi": "10.1371/journal.pbio.2005971",
                "source_type": "europepmc_oa_jats_xml",
                "repository": "Europe PMC",
                "licence": "CC BY 4.0",
                "version": "PMC6112682",
            },
        ),
        # D4: Duncan et al. (2017) J. Evol. Biol. 30:55
        (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=27711983&retmode=xml",
            "D4_Duncan2017_pubmed_27711983.xml",
            {
                "dataset_id": "D4_duncan2017",
                "paper_doi": "10.1111/jeb.12985",
                "source_type": "pubmed_bibliographic_xml",
                "repository": "NCBI PubMed",
                "licence": "public_metadata (replicate thermal treatments CC0 1.0 in Zenodo 4954973 / Dryad vv29r)",
                "version": "PMID_27711983",
            },
        ),
        (
            "https://api.archives-ouvertes.fr/search/?q=halId_s:hal-01530834&fl=*&wt=json",
            "D4_Duncan2017_hal_01530834.json",
            {
                "dataset_id": "D4_duncan2017",
                "paper_doi": "10.1111/jeb.12985",
                "source_type": "hal_open_metadata_json",
                "repository": "HAL (Archive ouverte)",
                "licence": "public_metadata",
                "version": "hal-01530834v1",
            },
        ),
        # D5: Zhao, Hao & Zhang (2016) PLOS ONE e0168560
        (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5226335/fullTextXML",
            "D5_Zhao2016_PMC5226335.xml",
            {
                "dataset_id": "D5_zhao2016",
                "paper_doi": "10.1371/journal.pone.0168560",
                "source_type": "europepmc_oa_jats_xml",
                "repository": "Europe PMC",
                "licence": "CC BY 4.0",
                "version": "PMC5226335",
            },
        ),
        # D6: Harrison et al. (2013) Proc. R. Soc. B 20130937
        (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3712419/fullTextXML",
            "D6_Harrison2013_PMC3712419.xml",
            {
                "dataset_id": "D6_harrison2013",
                "paper_doi": "10.1098/rspb.2013.0937",
                "source_type": "europepmc_oa_jats_xml",
                "repository": "Europe PMC",
                "licence": "CC BY 3.0",
                "version": "PMC3712419",
            },
        ),
        # Independent thermal performance dataset & Sharpe-Schoolfield TPC for SBW25 + SBW25phi2:
        # Padfield, Castledine & Buckling (2020) ISME J. 14:389-398 (Zenodo 3492192, GPL-3)
        (
            "https://zenodo.org/api/records/3492192",
            "Padfield2020_zenodo_3492192_metadata.json",
            {
                "dataset_id": "thermal_trait_SBW25_phi2",
                "paper_doi": "10.1038/s41396-019-0526-5",
                "dataset_doi": "10.5281/zenodo.3492192",
                "source_type": "zenodo_record_metadata",
                "repository": "Zenodo",
                "licence": "GPL-3.0 / other-open",
                "version": "v1.0",
            },
        ),
        (
            "https://raw.githubusercontent.com/padpadpadpad/Padfield_2019_ISME_bact_phage_temperature/master/data/phage_replication.csv",
            "Padfield2020_SBW25phi2_phage_replication.csv",
            {
                "dataset_id": "thermal_trait_SBW25_phi2",
                "paper_doi": "10.1038/s41396-019-0526-5",
                "dataset_doi": "10.5281/zenodo.3492192",
                "source_type": "empirical_thermal_performance_csv",
                "repository": "GitHub / Zenodo 3492192 (padpadpadpad/Padfield_2019_ISME_bact_phage_temperature)",
                "licence": "other-open (Zenodo 3492192 open access)",
                "version": "v1.0",
            },
        ),
        (
            "https://raw.githubusercontent.com/padpadpadpad/Padfield_2019_ISME_bact_phage_temperature/master/scripts/bacteria_phage_tpcs.R",
            "Padfield2020_bacteria_phage_tpcs.R",
            {
                "dataset_id": "thermal_trait_SBW25_phi2",
                "paper_doi": "10.1038/s41396-019-0526-5",
                "dataset_doi": "10.5281/zenodo.3492192",
                "source_type": "sharpe_schoolfield_tpc_specification",
                "repository": "GitHub / Zenodo 3492192 (padpadpadpad/Padfield_2019_ISME_bact_phage_temperature)",
                "licence": "other-open (Zenodo 3492192 open access)",
                "version": "v1.0",
            },
        ),
        # Arrhenius phage virion thermal inactivation across coliphages:
        # De Paepe & Taddei (2006) PLOS Biology 4(7):e193 (PMC1475768, CC BY)
        (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1475768/fullTextXML",
            "DePaepe2006_PMC1475768.xml",
            {
                "dataset_id": "thermal_trait_coliphage_arrhenius",
                "paper_doi": "10.1371/journal.pbio.0040193",
                "source_type": "europepmc_oa_jats_xml",
                "repository": "Europe PMC",
                "licence": "CC BY",
                "version": "PMC1475768",
            },
        ),
    ]

    for url, fname, meta in sources:
        fetch_item(url, fname, meta, items)

    manifest = {
        "study_id": "chemostat",
        "task": "r27-weather-chemostat",
        "field_records_count": 0,
        "outdoor_weather_sources_queried": [],
        "outdoor_weather_omission_reason": (
            "All six datasets (D1-D6) are controlled indoor laboratory chemostat or "
            "incubator microcosm experiments. Per CASESTUDY_SPEC.md ('Laboratory studies: "
            "use the laboratory's reported temperature regime instead and say so'), "
            "outdoor weather station and gridded climate products (NEON, Daymet, "
            "TerraClimate, NOAA SST, GHCN-Daily, ERA5) do not apply."
        ),
        "publisher_blocks_respected": [
            {
                "dataset_id": "D4_duncan2017",
                "paper_doi": "10.1111/jeb.12985",
                "note": "Wiley PDF endpoint returned HTTP 403; not retried or circumvented per CASESTUDY_SPEC.md Rule 2. Exact replicate temperature regimes (28 C, 32 C, and 2/4/8-day switching) are recorded directly in CC0 deposit Zenodo 4954973 (`Duncan_JEB.xlsx`).",
            }
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": items,
    }
    manifest_path = os.path.join(OUT_DIR, "WEATHER_ENV_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
