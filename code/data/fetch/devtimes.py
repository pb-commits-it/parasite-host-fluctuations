#!/usr/bin/env python3
"""Fetch legitimately open stage-duration datasets for case study `devtimes`.

Run:
    python3 -I data/fetch/devtimes.py

Stores raw files under `data/raw/devtimes/` (git-ignored) and writes
`data/raw/devtimes/MANIFEST.json` recording dataset_id, DOI, licence, version,
URL, byte count, SHA-256 checksum, and UTC access timestamp.

Rules enforced:
  - Standard library only (`python3 -I`).
  - Generic User-Agent; no e-mail address, API credential, or personal details.
  - Re-runnable: existing non-empty files are hashed in place unless `--force`
    is passed on the command line.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "data-scout/1.0 (python-urllib; open-ecology-case-study)"
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
OUT = (__import__('pathlib').Path(__file__).resolve().parent / '../raw/devtimes').resolve()
FORCE = "--force" in sys.argv


def http_get(url: str, timeout: int = 45, tries: int = 3) -> bytes:
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 + 2.0 * attempt)
    raise RuntimeError(f"GET failed for {url}: {last_err}")


def save_file(
    dataset_id: str,
    rel_path: str,
    url: str,
    doi: str,
    licence: str,
    version: str,
    manifest: list[dict],
) -> bool:
    dest = OUT / dataset_id / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if dest.exists() and dest.stat().st_size > 0 and not FORCE:
        raw = dest.read_bytes()
        manifest.append({
            "dataset_id": dataset_id,
            "path": str(dest.relative_to(OUT)),
            "url": url,
            "doi": doi,
            "licence": licence,
            "version": version,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_utc": now_utc,
            "status": "cached",
        })
        return True
    try:
        raw = http_get(url)
        dest.write_bytes(raw)
        manifest.append({
            "dataset_id": dataset_id,
            "path": str(dest.relative_to(OUT)),
            "url": url,
            "doi": doi,
            "licence": licence,
            "version": version,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_utc": now_utc,
            "status": "downloaded",
        })
        print(f"  [OK] {dataset_id}/{rel_path} ({len(raw):,} B)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        manifest.append({
            "dataset_id": dataset_id,
            "path": str(dest.relative_to(OUT)),
            "url": url,
            "doi": doi,
            "licence": licence,
            "version": version,
            "bytes": 0,
            "sha256": None,
            "fetched_utc": now_utc,
            "status": f"failed: {exc}",
        })
        print(f"  [WARN] {dataset_id}/{rel_path}: {exc}", flush=True)
        return False


def fetch_zenodo_record(
    dataset_id: str,
    record_id: int,
    wanted_files: set[str] | None,
    manifest: list[dict],
) -> None:
    meta_url = f"https://zenodo.org/api/records/{record_id}"
    meta_bytes = http_get(meta_url)
    meta = json.loads(meta_bytes.decode("utf-8"))
    dest_dir = OUT / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"_zenodo_{record_id}.json").write_bytes(meta_bytes)

    doi = meta.get("doi") or meta.get("metadata", {}).get("doi") or f"zenodo.{record_id}"
    lic_obj = meta.get("metadata", {}).get("license") or {}
    licence = lic_obj.get("id", "unknown") if isinstance(lic_obj, dict) else str(lic_obj)
    version = str(meta.get("metadata", {}).get("version") or meta.get("revision") or "1")

    for fl in meta.get("files", []):
        key = fl["key"]
        if wanted_files is not None and key not in wanted_files:
            continue
        url = fl["links"]["self"]
        save_file(dataset_id, key.replace("/", "_"), url, doi, licence, version, manifest)


def fetch_figshare_article(
    dataset_id: str,
    article_id: int,
    wanted_files: set[str] | None,
    manifest: list[dict],
) -> None:
    meta_url = f"https://api.figshare.com/v2/articles/{article_id}"
    meta_bytes = http_get(meta_url)
    meta = json.loads(meta_bytes.decode("utf-8"))
    dest_dir = OUT / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"_figshare_{article_id}.json").write_bytes(meta_bytes)

    doi = meta.get("doi") or f"figshare.{article_id}"
    lic_obj = meta.get("license") or {}
    licence = lic_obj.get("name", "unknown") if isinstance(lic_obj, dict) else str(lic_obj)
    version = str(meta.get("version") or "1")

    for fl in meta.get("files", []):
        name = fl["name"]
        if wanted_files is not None and name not in wanted_files:
            continue
        save_file(dataset_id, name.replace("/", "_"), fl["download_url"], doi, licence, version, manifest)


def fetch_biostudies_malaria(manifest: list[dict]) -> None:
    dataset_id = "D1_malaria_embo2024"
    acc = "S-EPMC11473648"
    info_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}/info"
    study_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}"
    dest_dir = OUT / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    info_bytes = http_get(info_url)
    study_bytes = http_get(study_url)
    (dest_dir / "_biostudies_info.json").write_bytes(info_bytes)
    (dest_dir / "_biostudies_study.json").write_bytes(study_bytes)
    info = json.loads(info_bytes.decode("utf-8"))
    http_link = info["httpLink"].rstrip("/")
    doi = "10.1038/s44321-024-00127-w"
    licence = "CC BY 4.0"
    version = "PMC11473648"
    for fname in (
        "44321_2024_127_MOESM3_ESM.xlsx",
        "44321_2024_127_MOESM4_ESM.xlsx",
        "44321_2024_127_MOESM5_ESM.xlsx",
        "44321_2024_127_MOESM6_ESM.xlsx",
        "44321_2024_127_MOESM7_ESM.xlsx",
        "44321_2024_127_MOESM8_ESM.xlsx",
        "44321_2024_127_MOESM9_ESM.xlsx",
        "44321_2024_127_MOESM10_ESM.xlsx",
        "44321_2024_127_MOESM11_ESM.zip",
        "44321_2024_127_MOESM13_ESM.zip",
    ):
        furl = f"{http_link}/Files/{urllib.parse.quote(fname)}"
        save_file(dataset_id, fname, furl, doi, licence, version, manifest)
    save_file(
        dataset_id,
        "PMC11473648_fulltext.xml",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11473648/fullTextXML",
        doi,
        licence,
        version,
        manifest,
    )
    fetch_github_files(
        dataset_id,
        "macarrasq/falciparum_infection_length",
        "main",
        ["README.md", "Meta_to_infection_length_github.Rmd", "data_raw.csv"],
        doi,
        licence,
        manifest,
    )


def fetch_github_files(
    dataset_id: str,
    repo: str,
    branch: str,
    rel_paths: list[str],
    doi: str,
    licence: str,
    manifest: list[dict],
) -> None:
    meta_bytes = http_get(f"https://api.github.com/repos/{repo}")
    dest_dir = OUT / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "_github_repo.json").write_bytes(meta_bytes)
    meta = json.loads(meta_bytes.decode("utf-8"))
    version = str(meta.get("pushed_at") or branch)
    for rp in rel_paths:
        quoted = "/".join(urllib.parse.quote(part) for part in rp.split("/"))
        raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{quoted}"
        local_name = rp.replace("/", "__")
        save_file(dataset_id, local_name, raw_url, doi, licence, version, manifest)


def fetch_vectraits(manifest: list[dict]) -> None:
    dataset_id = "D16_vectraits_compilations"
    dest_dir = OUT / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    target_genera = {
        "Anopheles", "Aedes", "Culex", "Culiseta", "Mansonia", "Toxorhynchites",
        "Ixodes", "Dermacentor", "Rhipicephalus", "Amblyomma", "Haemaphysalis", "Hyalomma",
        "Biomphalaria", "Bulinus", "Oncomelania", "Lymnaea", "Helisoma", "Physa",
        "Plasmodium", "Schistosoma", "Lutzomyia", "Phlebotomus", "Glossina", "Triatoma", "Rhodnius", "Xenopsylla",
    }
    dur_words = (
        "development time", "longevity", "life span", "lifespan", "incubation",
        "gonotrophic", "hatch time", "molt", "moult", "emergence", "survival time",
    )

    cat_path = dest_dir / "_explorer_catalogue.json"
    if cat_path.exists() and cat_path.stat().st_size > 1000 and not FORCE:
        all_entries = json.loads(cat_path.read_text())
    else:
        first = json.loads(http_get("https://vectorbyte.crc.nd.edu/portal/api/vectraits-explorer/?format=json").decode("utf-8"))
        count = int(first["data"]["count"])
        per_page = len(first["data"]["results"])
        n_pages = (count + per_page - 1) // per_page
        all_entries = list(first["data"]["results"])

        def get_page(p: int) -> list[dict]:
            u = f"https://vectorbyte.crc.nd.edu/portal/api/vectraits-explorer/?format=json&page={p}"
            return json.loads(http_get(u).decode("utf-8"))["data"]["results"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for res in pool.map(get_page, range(2, n_pages + 1)):
                all_entries.extend(res)
        cat_path.write_text(json.dumps(all_entries, indent=1))

    matched = []
    for e in all_entries:
        tname = (e.get("OriginalTraitName") or "").lower()
        g1 = e.get("Interactor1Genus") or ""
        g2 = e.get("Interactor2Genus") or ""
        if (g1 in target_genera or g2 in target_genera) and any(w in tname for w in dur_words):
            matched.append(e)

    (dest_dir / "_matched_index.json").write_text(json.dumps(matched, indent=1))
    print(f"  VecTraits matched {len(matched)} duration datasets out of {len(all_entries)} explorer entries", flush=True)

    def dl_one(entry: dict) -> dict:
        dsid = str(entry["DatasetID"])
        url = f"https://vectorbyte.crc.nd.edu/portal/api/vectraits-dataset/{dsid}/?format=json"
        dest = dest_dir / f"vectraits_{int(dsid):04d}.json"
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doi = entry.get("DOI") or entry.get("CuratedByDOI") or f"vectraits.{dsid}"
        if dest.exists() and dest.stat().st_size > 0 and not FORCE:
            raw = dest.read_bytes()
            status = "cached"
        else:
            try:
                raw = http_get(url)
                dest.write_bytes(raw)
                status = "downloaded"
            except Exception as exc:  # noqa: BLE001
                return {
                    "dataset_id": dataset_id,
                    "path": str(dest.relative_to(OUT)),
                    "url": url,
                    "doi": doi,
                    "licence": "CC BY 4.0 (VecTraits)",
                    "version": str(entry.get("EmbargoReleaseDate") or "2026"),
                    "bytes": 0,
                    "sha256": None,
                    "fetched_utc": now_utc,
                    "status": f"failed: {exc}",
                }
        return {
            "dataset_id": dataset_id,
            "path": str(dest.relative_to(OUT)),
            "url": url,
            "doi": doi,
            "licence": "CC BY 4.0 (VecTraits)",
            "version": str(entry.get("EmbargoReleaseDate") or "2026"),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_utc": now_utc,
            "status": status,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        vt_records = list(pool.map(dl_one, matched))
    manifest.extend(vt_records)
    ok_vt = sum(1 for r in vt_records if r["bytes"] > 0)
    print(f"  [OK] {dataset_id}: {ok_vt}/{len(vt_records)} dataset JSONs saved", flush=True)


def fetch_weather_and_temp_docs(manifest: list[dict]) -> None:
    """Amendment 2: Fetch open weather for field studies (Daymet, TerraClimate, NOAA GHCN-Daily)
    and open EuropePMC study metadata/fulltext to verify laboratory temperature regimes."""
    # 1. NASA ORNL DAAC Daymet V4 R1 daily 1 km for D6 eastern North America sites (2017-2020)
    d6_sites = [
        ("FD_FortDrum_NY", 44.0, -75.7),
        ("WP_WestPoint_NY", 41.4, -74.1),
        ("CL_CampLejeune_NC", 34.7, -77.5),
    ]
    for label, lat, lon in d6_sites:
        q = urllib.parse.urlencode({
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "vars": "tmax,tmin,prcp,vp,swe",
            "start": "2017-01-01",
            "end": "2020-12-31",
        })
        url = f"https://daymet.ornl.gov/single-pixel/api/data?{q}"
        save_file(
            "weather",
            f"daymet_{label}.csv",
            url,
            "10.3334/ORNLDAAC/2129",
            "NASA Open Data / Public Domain",
            "V4R1",
            manifest,
        )

    # 2. NASA POWER Agroclimatology daily gridded weather (~0.5 deg, public domain, fallback where THREDDS 8080 is closed)
    power_sites = [
        ("Mali_Kalifabougou_D1", 12.95, -8.17, "20110601", "20130630"),
        ("Belgium_Boshoek_D5", 51.1331, 4.5169, "20160101", "20191231"),
    ]
    for label, lat, lon, d0, d1 in power_sites:
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
            "weather",
            f"nasapower_{label}.json",
            url,
            "10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY",
            "NASA Open Data / Public Domain",
            "v2.0",
            manifest,
        )

    # 3. NOAA GHCN-Daily nearest open station records (Bamako Senou MLM00061291; Uccle/Brussels BE000006447)
    ghcn_jobs = [
        ("Mali_BamakoSenou_MLM00061291", "MLM00061291", "2011-06-01", "2013-06-30"),
        ("Belgium_Uccle_BE000006447", "BE000006447", "2016-01-01", "2019-12-31"),
    ]
    for label, stn, d0, d1 in ghcn_jobs:
        q = urllib.parse.urlencode({
            "dataset": "daily-summaries",
            "stations": stn,
            "startDate": d0,
            "endDate": d1,
            "dataTypes": "TMAX,TMIN,TAVG,PRCP",
            "units": "metric",
            "format": "csv",
        })
        url = f"https://www.ncei.noaa.gov/access/services/data/v1?{q}"
        save_file(
            "weather",
            f"ghcnd_{label}.csv",
            url,
            "10.7289/V5D21VHZ",
            "U.S. Government Work / Public Domain (NOAA GHCN-Daily)",
            "v1",
            manifest,
        )

    # 4. Open EuropePMC metadata & OA full-text XML for laboratory studies lacking a temp column in CSV/ODS
    epmc_queries = [
        ("D5_fracasso2021", 'DOI:"10.1111/evo.14463"', "10.1111/evo.14463"),
        ("D7_leclech2019", 'DOI:"10.1186/s13071-019-3741-z"', "10.1186/s13071-019-3741-z"),
        ("D8_civitello2020", 'TITLE:"Resource supply and periodic starvation"', "10.1098/rspb.2019.2663"),
        ("D8_civitello2020_alt", 'AUTH:"Civitello" AND "Biomphalaria" AND PUB_YEAR:2020', "10.1098/rspb.2020.1963"),
        ("D11_taskinen2022", 'AUTH:"Taskinen" AND "Rhipidocotyle" AND PUB_YEAR:2022', "10.1017/S0031182022000518"),
        ("D12_gleichsner2016", 'TITLE:"One stimulus" AND "Gleichsner"', "10.1111/evo.12921"),
        ("D13_aleuy2023", 'AUTH:"Aleuy" AND "Marshallagia"', "10.5061/dryad.vmcvdnczv"),
    ]
    for tag, query, doi in epmc_queries:
        q = urllib.parse.urlencode({"query": query, "resultType": "core", "format": "json"})
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{q}"
        ok = save_file("study_temp_docs", f"{tag}_epmc.json", url, doi, "Open API metadata", "v6.9", manifest)
        if ok:
            p = OUT / "study_temp_docs" / f"{tag}_epmc.json"
            try:
                res = json.loads(p.read_text()).get("resultList", {}).get("result", [])
                if res and res[0].get("pmcid") and res[0].get("isOpenAccess") == "Y":
                    pmcid = res[0]["pmcid"]
                    xurl = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                    save_file("study_temp_docs", f"{tag}_{pmcid}.xml", xurl, doi, "Open Access PMC XML", pmcid, manifest)
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    print("Fetching open stage-duration datasets into data/raw/devtimes/ ...", flush=True)

    # D1: P. falciparum human clonal infection durations (Mali cohort, EMBO Mol Med 2024, CC BY 4.0)
    fetch_biostudies_malaria(manifest)

    # D2: USPHS Malariotherapy summary table (Johnston et al. 2013 PLOS Comp Biol, CC BY 4.0)
    fetch_figshare_article("D2_malariotherapy_plos2013", 685675, None, manifest)

    # D3: Anopheles stephensi survival & P. falciparum EIP across temperatures (Shapiro et al. 2017, CC0)
    fetch_zenodo_record(
        "D3_anopheles_pf_shapiro2017",
        4974691,
        {"temp.surv.csv", "infectious.data.csv", "PBiol.R.Code.R"},
        manifest,
    )

    # D4: mSOS Plasmodium sporogony & Anopheles survival compilation (Stopard et al. 2021, MIT)
    fetch_github_files(
        "D4_plasmodium_msos2021",
        "IsaacStopard/mSOS",
        "main",
        ["LICENSE", "README.md", "data/mSOS_all_parasite_data.csv", "data/mSOS_all_survival_data.csv"],
        "10.1371/journal.pcbi.1008658",
        "MIT",
        manifest,
    )

    # D5: Ixodes arboricola individual feeding, moulting & survival times (Fracasso et al. 2021, CC0)
    fetch_zenodo_record(
        "D5_ixodes_fracasso2021",
        5519771,
        {"All_ticks.csv", "Description_abbreviations.docx", "Number_hatched_eggs.csv"},
        manifest,
    )

    # D6: Ixodes scapularis multi-site field-enclosure survival & moulting (Brunner et al. 2022/2023, CC0 + MIT)
    fetch_zenodo_record(
        "D6_ixodes_scapularis_brunner2022",
        7461098,
        {
            "README.txt",
            "All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx",
            "Fed_Adult_Female_Survival_Summary_2017_and_2018_season_updated.xlsx",
            "Adult_holdingTemps.csv",
            "ibutton_data.csv",
        },
        manifest,
    )
    fetch_zenodo_record(
        "D6_ixodes_scapularis_brunner2022",
        6803051,
        {"TickClimate_Molting_FedNymphsLarvae.Rmd"},
        manifest,
    )

    # D7: Schistosoma mansoni in Biomphalaria glabrata individual prepatent/shedding/survival (Le Clec'h et al. 2019, CC BY 4.0)
    fetch_zenodo_record("D7_schisto_leclech2019", 3466005, None, manifest)

    # D8: Schistosoma mansoni in Biomphalaria glabrata longitudinal shedding/survival (Civitello et al. 2020, CC0)
    fetch_zenodo_record("D8_schisto_civitello2020", 3860566, None, manifest)

    # D9: Schistosoma mansoni thermal traits compilation (Nguyen et al. 2021 PNAS, CC BY 4.0)
    fetch_zenodo_record("D9_schisto_thermal_nguyen2021", 4550934, None, manifest)

    # D10: S. haematobium & S. mansoni thermal compilation (Aslan et al. 2024 PLOS NTD, MIT)
    fetch_github_files(
        "D10_schisto_thermal_aslan2024",
        "ibrahimhalilaslan/Thermal_sensitive_schistosomiasis_model",
        "main",
        [
            "LICENSE",
            "README.md",
            "s.haematobium/bootstrapping/lat_per_snail.R",
            "s.haematobium/bootstrapping/mort_snail_2.R",
            "s.mansoni/bootstrapping/lat_per_snail.R",
            "s.haematobium/ther_sen_par_bul_haemtobium/chester_kalinda_2017_bulinus_globosus_thermal sensitive demographic parameters_GADL.xlsx",
            "s.mansoni/ther_sen_par_biom_mansoni/McCreesh_2014_Biomphalaria_sudanica_thermal sensitive demographic parameters_GADL.xlsx",
            "s.mansoni/ther_sen_par_biom_mansoni/c_c_appleton_1977_Biomphalaria_pfeiffferi_thermal sensitive demographic parameters_GADL.xlsx",
        ],
        "10.1371/journal.pntd.0011836",
        "MIT",
        manifest,
    )

    # D11: Rhipidocotyle trematodes individual cercarial shedding start/end/duration across temperatures (Taskinen et al. 2022, CC BY 4.0)
    fetch_figshare_article("D11_trematode_taskinen2022", 19584286, None, manifest)

    # D12: Snail-trematode prepatent and patent durations (Gleichsner et al. 2016, CC0)
    fetch_zenodo_record(
        "D12_trematode_gleichsner2016",
        4993047,
        {"Gleichsner_et_al._RawData.csv", "AestivationDataRcode.R"},
        manifest,
    )

    # D13: Marshallagia marshalli trichostrongylid nematode egg hatch, L3 development & mortality across 5-38C (Aleuy et al. 2023, CC0)
    fetch_zenodo_record("D13_nematode_marshallagia2023", 8255690, None, manifest)

    # D14: Heligmosomoides polygyrus adult nematode survival in mice (Lippens et al. 2016, 2017, CC0)
    fetch_zenodo_record("D14_nematode_heligmosomoides", 4951702, None, manifest)
    fetch_zenodo_record("D14_nematode_heligmosomoides", 5013889, None, manifest)

    # D15: Benesh, Lafferty & Kuris (2017) helminth life-cycle database (via Figshare 30764579, CC BY 4.0)
    fetch_figshare_article("D15_helminth_clc_benesh2017", 30764579, {"CLC_database_lifehistory.csv"}, manifest)

    # D16: VecTraits curated vector/tick/parasite duration datasets (CC BY 4.0)
    fetch_vectraits(manifest)

    # Amendment 2: Open weather covariates for field studies & EuropePMC open study-temperature metadata
    fetch_weather_and_temp_docs(manifest)

    manifest_path = OUT / "MANIFEST.json"
    summary = {
        "study_id": "devtimes",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "total_files": len(manifest),
        "ok_files": sum(1 for r in manifest if r["bytes"] > 0),
        "total_bytes": sum(r["bytes"] for r in manifest),
        "files": manifest,
    }
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(
        f"\nCompleted fetch in {summary['elapsed_seconds']:.1f} s: "
        f"{summary['ok_files']}/{summary['total_files']} files, "
        f"{summary['total_bytes'] / (1024 * 1024):.2f} MB total -> {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
