#!/usr/bin/env python3
"""Fetch open datasets and source metadata for the Wood et al. (2023) case study.

Sources fetched into data/raw/wood/ (git-ignored):
1. Wood et al. (2023) PNAS dataset (doi:10.5061/dryad.fqz612jwf, CC0 1.0):
   - Queries Dryad API metadata (dataset + versions) and tries one direct file request
     (Dryad currently requires a bearer credential and returns HTTP 401; recorded, not retried).
   - Downloads all four dataset files from the official Zenodo 'dryad' community mirror
     (Zenodo record 7448240, doi:10.5061/dryad.fqz612jwf, CC0 1.0), verifying sizes and MD5s.
   - Downloads model/data-loading source text files (never executed) from the authors' open
     GitHub repository (github.com/wood-lab/Wood_et_al_2023_PNAS; renamed from Wood_et_al_2022_PNAS),
     plus open-access article XML from Europe PMC (PMC9934024, doi:10.1073/pnas.2211903120).
2. Benesh, Parker & Chubb (2021) / Benesh, Lafferty & Kuris (2017) Helminth Life-Cycle Database:
   - Official Zenodo 'dryad' community mirror (Zenodo record 4975516, doi:10.5061/dryad.gtht76hmm,
     CC0 1.0): parasite_stage_level.zip and host_traits.zip, verified by MD5.
   - Raw stage/host/life-history CSV tables from github.com/dbenesh82/propagule_transmission
     and github.com/dbenesh82/paratenics_as_bridges (derived from Ecology 98:882, doi:10.1002/ecy.1680, CC0).
3. Natural History Museum (London) Host-Parasite Database:
   - NHM Data Portal package 'host-parasite-database' (doi:10.5519/nplk7c5u, CC BY 4.0):
     project-host-parasites-csv-export.zip.
4. Global Mammal Parasite Database 2.0 (Stephens et al. 2017, Ecology 98:1476, doi:10.1002/ecy.1799):
   - GMPD_main.csv and schema.json via Global Biotic Interactions mirror.

Run:  python3 -I data/fetch/wood.py

Re-runnable: existing files with matching size (and MD5 where provided by repository) are skipped.
Generic User-Agent; no credentials, tokens, or personal details in any request. Standard library only.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "wood"
UA = "data-fetcher/1.0 (python-urllib; open-data research)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/wood').resolve())


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def http_get(url, tries=3, timeout=180):
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (401, 403, 404):
                body = b""
                try:
                    body = err.read()[:512]
                except Exception:
                    pass
                return err.code, body
            last_err = err
            time.sleep(2 + 3 * attempt)
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_digests(path):
    h_md5 = hashlib.md5()
    h_sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            h_md5.update(chunk)
            h_sha.update(chunk)
    return size, h_md5.hexdigest(), h_sha.hexdigest()


def save_file(url, relpath, manifest, meta, expected_size=None, expected_md5=None):
    dest = os.path.join(OUT, relpath)
    if os.path.exists(dest):
        sz, got_md5, got_sha = file_digests(dest)
        size_ok = (expected_size is None) or (sz == expected_size)
        md5_ok = (expected_md5 is None) or (got_md5 == expected_md5)
        if size_ok and md5_ok and sz > 0:
            entry = dict(meta)
            entry.update({
                "path": relpath,
                "url": url,
                "bytes": sz,
                "md5": got_md5,
                "sha256": got_sha,
                "downloaded": True,
                "cached": True,
                "checked_utc": utc_now(),
            })
            manifest.append(entry)
            print(f"{sz:10d}  {relpath}  (cached, verified)")
            return
    status, data = http_get(url)
    if status != 200:
        entry = dict(meta)
        entry.update({
            "path": relpath,
            "url": url,
            "http_status": status,
            "downloaded": False,
            "checked_utc": utc_now(),
            "note": data.decode("utf-8", errors="replace")[:300],
        })
        manifest.append(entry)
        print(f"  HTTP {status}  {relpath}  (not downloaded)")
        return
    got_md5 = md5_bytes(data)
    got_sha = sha256_bytes(data)
    if expected_size is not None and len(data) != expected_size:
        raise RuntimeError(f"Size mismatch for {relpath}: got {len(data)}, expected {expected_size}")
    if expected_md5 is not None and got_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {relpath}: got {got_md5}, expected {expected_md5}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    entry = dict(meta)
    entry.update({
        "path": relpath,
        "url": url,
        "bytes": len(data),
        "md5": got_md5,
        "sha256": got_sha,
        "downloaded": True,
        "cached": False,
        "fetched_utc": utc_now(),
    })
    manifest.append(entry)
    print(f"{len(data):10d}  {relpath}")


def check_dryad_once(doi, subdir, manifest):
    enc = urllib.parse.quote("doi:" + doi, safe="")
    ds_url = f"https://datadryad.org/api/v2/datasets/{enc}"
    status, raw = http_get(ds_url)
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    if status != 200:
        manifest.append({
            "source": "Dryad API metadata",
            "doi": doi,
            "url": ds_url,
            "http_status": status,
            "downloaded": False,
            "checked_utc": utc_now(),
        })
        return
    with open(os.path.join(OUT, subdir, "_dryad_dataset.json"), "wb") as f:
        f.write(raw)
    ds = json.loads(raw)
    ver_href = ds.get("_links", {}).get("stash:version", {}).get("href", "")
    version_num = ds.get("versionNumber")
    license_uri = ds.get("license")
    manifest.append({
        "source": "Dryad API metadata",
        "doi": doi,
        "version": version_num,
        "license": license_uri,
        "path": os.path.join(subdir, "_dryad_dataset.json"),
        "url": ds_url,
        "bytes": len(raw),
        "downloaded": True,
        "checked_utc": utc_now(),
    })
    if ver_href:
        files_url = f"https://datadryad.org{ver_href}/files"
        f_status, f_raw = http_get(files_url)
        if f_status == 200:
            with open(os.path.join(OUT, subdir, "_dryad_files.json"), "wb") as f:
                f.write(f_raw)
            files_obj = json.loads(f_raw)
            embedded = files_obj.get("_embedded", {}).get("stash:files", [])
            if embedded:
                first_dl = embedded[0].get("_links", {}).get("stash:download", {}).get("href", "")
                if first_dl:
                    dl_url = f"https://datadryad.org{first_dl}"
                    dl_status, dl_body = http_get(dl_url, tries=1, timeout=30)
                    manifest.append({
                        "source": "Dryad direct file probe (single request, no retry)",
                        "doi": doi,
                        "url": dl_url,
                        "http_status": dl_status,
                        "downloaded": False,
                        "note": dl_body.decode("utf-8", errors="replace")[:200] if dl_status != 200 else "200 OK",
                        "checked_utc": utc_now(),
                    })
                    print(f"  Dryad direct probe for {doi}: HTTP {dl_status} (using official Zenodo mirror)")


def fetch_zenodo_record(record_id, subdir, doi, manifest):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    status, raw = http_get(api_url)
    if status != 200:
        raise RuntimeError(f"Zenodo API HTTP {status} for record {record_id}")
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    meta_path = os.path.join(OUT, subdir, "_zenodo_record.json")
    with open(meta_path, "wb") as f:
        f.write(raw)
    rec = json.loads(raw)
    md = rec.get("metadata", {})
    lic = (md.get("license") or {}).get("id", "unknown")
    pub_date = md.get("publication_date", "")
    rec_doi = rec.get("doi") or md.get("doi") or doi
    for fl in rec.get("files", []):
        key = fl["key"]
        size = fl.get("size")
        checksum = fl.get("checksum", "")
        exp_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        dl_url = fl.get("links", {}).get("self")
        meta = {
            "source": f"Zenodo record {record_id} (official Dryad mirror)",
            "doi": rec_doi,
            "zenodo_record": str(record_id),
            "publication_date": pub_date,
            "license": lic,
        }
        save_file(dl_url, os.path.join(subdir, key), manifest, meta, expected_size=size, expected_md5=exp_md5)
        time.sleep(0.3)


def fetch_github_files(repo, subdir, paths, citation_note, manifest):
    info_status, info_raw = http_get(f"https://api.github.com/repos/{repo}")
    if info_status != 200:
        raise RuntimeError(f"GitHub API HTTP {info_status} for {repo}")
    info = json.loads(info_raw)
    branch = info.get("default_branch", "main")
    lic = (info.get("license") or {}).get("spdx_id") or "unspecified-public-academic-repo"
    br_status, br_raw = http_get(f"https://api.github.com/repos/{repo}/commits/{branch}")
    commit_sha = json.loads(br_raw).get("sha", branch) if br_status == 200 else branch
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    with open(os.path.join(OUT, subdir, "_github_repo.json"), "wb") as f:
        f.write(json.dumps({"repo": repo, "branch": branch, "commit_sha": commit_sha, "license": lic}, indent=2).encode("utf-8"))
    for rel in paths:
        raw_url = f"https://raw.githubusercontent.com/{repo}/{commit_sha}/{urllib.parse.quote(rel)}"
        meta = {
            "source": f"GitHub {repo}",
            "commit_sha": commit_sha,
            "license": lic,
            "citation": citation_note,
        }
        save_file(raw_url, os.path.join(subdir, rel), manifest, meta)
        time.sleep(0.2)


def fetch_nhm_csv(subdir, manifest):
    pkg_url = "https://data.nhm.ac.uk/api/3/action/package_show?id=host-parasite-database"
    status, raw = http_get(pkg_url)
    if status != 200:
        raise RuntimeError(f"NHM Data Portal API HTTP {status}")
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    with open(os.path.join(OUT, subdir, "_nhm_package.json"), "wb") as f:
        f.write(raw)
    res = json.loads(raw).get("result", {})
    doi = res.get("doi", "10.5519/nplk7c5u")
    lic = res.get("license_title", "CC BY 4.0")
    version = res.get("version") or res.get("metadata_modified", "")
    for item in res.get("resources", []):
        if item.get("name") == "project-host-parasites-csv-export.zip":
            meta = {
                "source": "Natural History Museum Data Portal (Host-Parasite Database)",
                "doi": doi,
                "version": version,
                "license": lic,
            }
            save_file(item["url"], os.path.join(subdir, item["name"]), manifest, meta, expected_size=item.get("size"))
            break


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []

    print("=== 1. Wood et al. (2023) PNAS dataset (Dryad metadata + Zenodo mirror 7448240) ===")
    check_dryad_once("10.5061/dryad.fqz612jwf", "wood_dryad_meta", manifest)
    fetch_zenodo_record("7448240", "wood_zenodo_7448240", "10.5061/dryad.fqz612jwf", manifest)

    print("=== 2. Wood et al. (2023) Europe PMC open-access article XML & SI (PMC9934024) ===")
    save_file(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9934024/fullTextXML",
        "wood_epmc/PMC9934024.xml",
        manifest,
        {
            "source": "Europe PMC open-access fullTextXML",
            "doi": "10.1073/pnas.2211903120",
            "pmcid": "PMC9934024",
            "license": "open-access-epmc",
        },
    )
    save_file(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9934024/supplementaryFiles",
        "wood_epmc/PMC9934024_supplementaryFiles.zip",
        manifest,
        {
            "source": "Europe PMC open-access supplementaryFiles (contains pnas.2211903120.sapp.pdf)",
            "doi": "10.1073/pnas.2211903120",
            "pmcid": "PMC9934024",
            "license": "open-access-epmc",
        },
    )

    print("=== 3. Wood et al. (2023) GitHub model & metadata text files (read-only, never executed) ===")
    wood_gh_paths = [
        "README.md",
        "data/final_dataset_metadata.csv",
        "data/parasite_species_codes.csv",
        "data/psite_lc_complexity.csv",
        "data/corresponding_psite_types_tim_to_universal_code.csv",
        "src/load_data.R",
        "src/configure_TMB_runs.R",
        "src/phase_1_models.R",
        "src/phase_2_models.R",
        "src/TMB/glmm.cpp",
        "src/TMB/glmm_byspecies.cpp",
        "src/TMB/glmm_byspecies_groups.cpp",
        "src/TMB/glmm_byspecies_groups_lh.cpp",
        "src/TMB/glmm_byspecies_lh.cpp",
        "src/TMB/glmm_byspecies_lh_predict.cpp",
        "src/TMB/glmm_no_pred.cpp",
    ]
    fetch_github_files(
        "wood-lab/Wood_et_al_2023_PNAS",
        "wood_github",
        wood_gh_paths,
        "Wood et al. (2023) PNAS 120(3):e2211903120 (repo renamed from Wood_et_al_2022_PNAS)",
        manifest,
    )

    print("=== 4. Benesh et al. (2021) Helminth Life-Cycle Database (Zenodo mirror 4975516 + raw CSVs) ===")
    check_dryad_once("10.5061/dryad.gtht76hmm", "benesh_dryad_meta", manifest)
    fetch_zenodo_record("4975516", "benesh_zenodo_4975516", "10.5061/dryad.gtht76hmm", manifest)
    fetch_github_files(
        "dbenesh82/propagule_transmission",
        "benesh_clc_raw",
        [
            "README.md",
            "data/Metadata.md",
            "data/CLC_database_hosts.csv",
            "data/CLC_database_lifehistory.csv",
        ],
        "Benesh, Lafferty & Kuris (2017) Ecology 98:882, doi:10.1002/ecy.1680 (CC0)",
        manifest,
    )
    fetch_github_files(
        "dbenesh82/paratenics_as_bridges",
        "benesh_paratenics",
        [
            "README.md",
            "data/stage_level_combined_noimputed.csv",
        ],
        "Benesh et al. stage-level life-cycle compilation (doi:10.1002/ecy.1680 / doi:10.5061/dryad.gtht76hmm)",
        manifest,
    )

    print("=== 5. Natural History Museum (London) Host-Parasite Database (CC BY 4.0) ===")
    fetch_nhm_csv("nhm_host_parasite", manifest)

    print("=== 6. Global Mammal Parasite Database 2.0 (Stephens et al. 2017, doi:10.1002/ecy.1799) ===")
    fetch_github_files(
        "globalbioticinteractions/global-mammal-parasite-database",
        "gmpd2",
        ["README.md", "globi.json", "schema.json", "GMPD_main.csv"],
        "Stephens et al. (2017) Ecology 98:1476, doi:10.1002/ecy.1799",
        manifest,
    )

    print("=== 7. WoRMS (CC BY 4.0) & GloBI (CC BY 4.0) taxonomic and host-association cache ===")
    wg_path = os.path.join(OUT, "worms_globi_cache.json")
    if os.path.exists(wg_path) and os.path.getsize(wg_path) > 0:
        sz, got_md5, got_sha = file_digests(wg_path)
        manifest.append({
            "source": "WoRMS REST API (marinespecies.org, CC BY 4.0) + GloBI API (globalbioticinteractions.org, CC BY 4.0)",
            "path": "worms_globi_cache.json",
            "bytes": sz,
            "md5": got_md5,
            "sha256": got_sha,
            "downloaded": True,
            "cached": True,
            "checked_utc": utc_now(),
        })
        print(f"{sz:10d}  worms_globi_cache.json  (cached, verified)")
    else:
        taxa_queries = [
            "Corynosoma", "Echinorhynchus gadi", "Clestobothrium crassiceps", "Nybelinia surmenicola", "Gyrocotyle",
            "Bomolochus bellones", "Bomolochus cuneatus", "Clavella", "Lepeophtheirus", "Parabrachiella",
            "Oceanobdella pallida", "Anthocotyle merluccii", "Gyrodactylus", "Chimaericola", "Microcotyle",
            "Ascarophis", "Cucullanellus kanabus", "Ascarophis sebastodis", "Anisakis", "Contracaecum",
            "Capillaria parophrysi", "Clavinema mariae", "Cucullanus annulatus",
            "Aporocotyle margolisi", "Lecithaster", "Podocotyle", "Pronoprymna petrowi",
            "Lepidapedon", "Gonocerca", "Lecithochirium", "Opechona", "Derogenes varicus",
        ]
        wg_cache = {}
        for t in taxa_queries:
            genus = t.split()[0]
            w_info = None
            for q in [t, genus]:
                st_w, raw_w = http_get(
                    "https://www.marinespecies.org/rest/AphiaRecordsByName/"
                    + urllib.parse.quote(q) + "?like=false&marine_only=false"
                )
                if st_w == 200 and raw_w:
                    res_w = json.loads(raw_w)
                    if res_w:
                        w_info = {
                            "query": q,
                            "AphiaID": res_w[0].get("AphiaID"),
                            "scientificname": res_w[0].get("scientificname"),
                            "valid_name": res_w[0].get("valid_name"),
                            "phylum": res_w[0].get("phylum"),
                            "class": res_w[0].get("class"),
                            "order": res_w[0].get("order"),
                            "family": res_w[0].get("family"),
                        }
                        break
            globi_hosts = []
            st_g, raw_g = http_get(
                "https://api.globalbioticinteractions.org/taxon/"
                + urllib.parse.quote(t) + "/parasiteOf?type=json.v2"
            )
            if st_g == 200 and raw_g:
                hosts = {
                    item.get("target", {}).get("name")
                    for item in (json.loads(raw_g) or [])
                    if item.get("target", {}).get("name")
                }
                globi_hosts = sorted(hosts)
            wg_cache[t] = {
                "worms": w_info,
                "globi_host_count": len(globi_hosts),
                "globi_sample_hosts": globi_hosts[:8],
            }
            time.sleep(0.15)
        raw_bytes = json.dumps(wg_cache, indent=2).encode("utf-8")
        with open(wg_path, "wb") as f:
            f.write(raw_bytes)
        manifest.append({
            "source": "WoRMS REST API (marinespecies.org, CC BY 4.0) + GloBI API (globalbioticinteractions.org, CC BY 4.0)",
            "path": "worms_globi_cache.json",
            "bytes": len(raw_bytes),
            "md5": md5_bytes(raw_bytes),
            "sha256": sha256_bytes(raw_bytes),
            "downloaded": True,
            "cached": False,
            "fetched_utc": utc_now(),
        })
        print(f"{len(raw_bytes):10d}  worms_globi_cache.json")

    manifest_obj = {
        "study_id": DATASET,
        "generated_utc": utc_now(),
        "items": manifest,
    }
    manifest_path = os.path.join(OUT, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
    print(f"Wrote {manifest_path} ({len(manifest)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
