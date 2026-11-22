#!/usr/bin/env python3
"""Fetch open experimental microcosm and parameter-calibration datasets for case study `microcosm`.

Datasets fetched into `data/raw/microcosm/` (git-ignored):
  1. duncan2013_paramecium_holospora:
     Duncan, Gonzalez & Kaltz (2013) Proc. R. Soc. B 280:20131747 (doi:10.1098/rspb.2013.1747)
     Official Zenodo "dryad" mirror record 4949969 of Dryad doi:10.5061/dryad.3hq2c (CC0 1.0, v1).
  2. ferris2020_phage_oscillations:
     Ferris, Wright, Brockhurst & Best (2020) Proc. R. Soc. B 287:20200787 (doi:10.1098/rspb.2020.0787)
     Figshare articles 12283799 (experiment data .xlsx) and 12283802 (appendices .pdf),
     collection doi:10.6084/m9.figshare.c.4971947 (CC BY 4.0, v1).
  3. harrison2013_phage_frequency:
     Harrison, Laine, Hietala & Brockhurst (2013) Proc. R. Soc. B 280:20130937 (doi:10.1098/rspb.2013.0937)
     Official Zenodo "dryad" mirror record 4957209 of Dryad doi:10.5061/dryad.8k015 (CC0 1.0, v1).
  4. dallas2018_metschnikowia_threshold:
     Dallas, Krkosek & Drake (2018) R. Soc. Open Sci. 5:171975 (doi:10.1098/rsos.171975)
     Figshare articles 5133001 (doi:10.6084/m9.figshare.5133001.v1, CC BY 4.0) and
     5745573 (supplement doi:10.6084/m9.figshare.5745573.v1, CC BY 4.0).
  5. dallas2016_metschnikowia_tempvar:
     Dallas & Drake (2016) Ecol. Evol. 6:7931-7938 (doi:10.1002/ece3.2539)
     Figshare article 3615816 (doi:10.6084/m9.figshare.3615816.v1, CC BY 4.0; includes sporeSurv.csv & filterRate.csv).
  6. davenport2024a_spores_outbreaks:
     Davenport et al. (2024) Ecology (doi:10.1002/ecy.4235)
     Official Zenodo "dryad" mirror record 10359990 of Dryad doi:10.5061/dryad.tqjq2bw5m (CC0 1.0, v1).
  7. clerc2015_pasteuria_stages:
     Clerc, Ebert & Hall (2015) Evolution (doi:10.1111/evo.12612)
     Official Zenodo "dryad" mirror record 5011860 of Dryad doi:10.5061/dryad.n1ks4 (CC0 1.0, v1).
  8. tadiri2023_pasteuria_epidemics:
     Tadiri (2023) Oikos (doi:10.1111/oik.10014)
     Official Zenodo "dryad" mirror record 8169416 of Dryad doi:10.5061/dryad.8931zcrwp (CC0 1.0, v1).
  9. krichel2023_ordospora_tempvar:
     Krichel et al. (2023) PLOS Biology 21:e3002260 (doi:10.1371/journal.pbio.3002260)
     Figshare articles 24111775 and 24111778 (CC BY 4.0, v1).
  10. dryad_blocked_metadata:
     Open Dryad API v2 metadata & file lists (plus single-request HTTP 401 check, not circumvented) for:
       - Davenport et al. (2024b) doi:10.5061/dryad.s7h44j1gb (CC0 1.0; plus Zenodo 13259720 code metadata)
       - Davenport et al. (2026) doi:10.5061/dryad.nvx0k6f7r (CC0 1.0)
       - Gutierrez et al. (2025) doi:10.5061/dryad.qjq2bvqrt (CC0 1.0)
  11. weather (for S6 field survey: 6 SE Michigan lakes, May-November 2021):
       - NASA ORNL DAAC Daymet V4R1 daily 1 km (doi:10.3334/ORNLDAAC/2129, Public Domain)
       - NOAA GHCN-Daily station summaries (doi:10.7289/V5D21VHZ, Public Domain)
  12. study_docs:
       - EuropePMC Open Access full-text XML for laboratory regime & methods verification
         (PMC10491407 Krichel et al. 2023; PMC6093173 Dallas & Drake 2016;
          PMC3712419 Harrison et al. 2013; PMC4523363 Vercken et al. 2015).

Run:  python3 -I data/fetch/microcosm.py
Standard library only. Generic User-Agent; no credentials, tokens, or personal details.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "microcosm"
UA = "data-scout/1.0 (python-urllib; research data inventory)"
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/microcosm').resolve())


def get(url, tries=3, timeout=180):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * i)
    raise SystemExit("FAILED %s: %s" % (url, last))


def save_bytes(data, relpath, url, meta, manifest):
    dest = os.path.join(OUT, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    entry = {
        "path": relpath,
        "url": url,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "downloaded": True,
    }
    entry.update(meta)
    manifest.append(entry)
    print("%10d  %s" % (len(data), relpath))


def save_or_reuse(url, relpath, meta, manifest):
    dest = os.path.join(OUT, relpath)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        with open(dest, "rb") as f:
            data = f.read()
        entry = {
            "path": relpath,
            "url": url,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "downloaded": True,
            "reused_cached": True,
        }
        entry.update(meta)
        manifest.append(entry)
        print("%10d  %s  (cached)" % (len(data), relpath))
        return
    data = get(url)
    save_bytes(data, relpath, url, meta, manifest)
    time.sleep(0.35)


def fetch_zenodo(record_id, subdir, source_doi, manifest, only=None, skip=()):
    api_url = "https://zenodo.org/api/records/%s" % record_id
    raw = get(api_url)
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    meta_path = os.path.join(OUT, subdir, "_zenodo_metadata.json")
    with open(meta_path, "wb") as f:
        f.write(raw)
    rec = json.loads(raw.decode("utf-8"))
    lic = (rec.get("metadata", {}).get("license") or {}).get("id", "unknown")
    ver = rec.get("metadata", {}).get("version") or str(rec.get("revision", "1"))
    doi = rec.get("doi") or source_doi
    for fl in rec.get("files", []):
        name = fl["key"]
        f_url = fl["links"]["self"]
        rel = os.path.join(subdir, name.replace("/", "_"))
        meta = {
            "repository": "Zenodo",
            "record_id": str(record_id),
            "doi": doi,
            "source_doi": source_doi,
            "license": lic,
            "version": ver,
        }
        if any(name.endswith(s) for s in skip) or (only is not None and name not in only):
            entry = {
                "path": rel,
                "url": f_url,
                "bytes": fl.get("size", 0),
                "downloaded": False,
                "reason": "skipped by filter",
            }
            entry.update(meta)
            manifest.append(entry)
            print("%10d  %s  (skipped)" % (fl.get("size", 0), rel))
            continue
        save_or_reuse(f_url, rel, meta, manifest)


def fetch_figshare(article_id, subdir, manifest, skip=()):
    api_url = "https://api.figshare.com/v2/articles/%s" % article_id
    raw = get(api_url)
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    with open(os.path.join(OUT, subdir, "_figshare_%s.json" % article_id), "wb") as f:
        f.write(raw)
    art = json.loads(raw.decode("utf-8"))
    lic = (art.get("license") or {}).get("name", "unknown")
    ver = str(art.get("version", "1"))
    doi = art.get("doi", "")
    for fl in art.get("files", []):
        name = fl["name"]
        f_url = fl["download_url"]
        rel = os.path.join(subdir, name.replace("/", "_"))
        meta = {
            "repository": "Figshare",
            "article_id": str(article_id),
            "doi": doi,
            "license": lic,
            "version": ver,
        }
        if any(name.endswith(s) for s in skip):
            entry = {
                "path": rel,
                "url": f_url,
                "bytes": fl.get("size", 0),
                "downloaded": False,
                "reason": "skipped by filter",
            }
            entry.update(meta)
            manifest.append(entry)
            print("%10d  %s  (skipped)" % (fl.get("size", 0), rel))
            continue
        save_or_reuse(f_url, rel, meta, manifest)


def check_dryad_blocked(dryad_doi, slug, manifest):
    subdir = os.path.join("dryad_blocked_metadata", slug)
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    enc = urllib.parse.quote("doi:" + dryad_doi, safe="")
    ds_url = "https://datadryad.org/api/v2/datasets/" + enc
    raw_ds = get(ds_url)
    with open(os.path.join(OUT, subdir, "dataset.json"), "wb") as f:
        f.write(raw_ds)
    ds = json.loads(raw_ds.decode("utf-8"))
    lic = ds.get("license", "unknown")
    ver_num = str(ds.get("versionNumber", "1"))
    ver_href = ds["_links"]["stash:version"]["href"]
    ver_id = ver_href.split("/")[-1]
    fl_url = "https://datadryad.org/api/v2/versions/%s/files" % ver_id
    raw_fl = get(fl_url)
    with open(os.path.join(OUT, subdir, "files.json"), "wb") as f:
        f.write(raw_fl)
    fls = json.loads(raw_fl.decode("utf-8")).get("_embedded", {}).get("stash:files", [])
    status_note = "not_tested"
    if fls:
        fid = fls[0]["_links"]["self"]["href"].split("/")[-1]
        dl_url = "https://datadryad.org/api/v2/files/%s/download" % fid
        req = urllib.request.Request(dl_url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                status_note = "HTTP %s" % resp.status
        except urllib.error.HTTPError as e:
            status_note = "HTTP %s (%s) - Dryad automated download refused; not circumvented" % (e.code, e.reason)
        except Exception as e:  # noqa: BLE001
            status_note = "Error: %s" % e
    for fobj in fls:
        fid = fobj["_links"]["self"]["href"].split("/")[-1]
        manifest.append({
            "path": os.path.join(subdir, fobj["path"]),
            "url": "https://datadryad.org/api/v2/files/%s/download" % fid,
            "bytes": fobj.get("size", 0),
            "repository": "Dryad",
            "doi": dryad_doi,
            "license": lic,
            "version": ver_num,
            "downloaded": False,
            "reason": status_note,
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    print("     meta   %s (%d files listed; %s)" % (subdir, len(fls), status_note))


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []

    # 1. Duncan, Gonzalez & Kaltz (2013) Paramecium - Holospora red vs white noise
    fetch_zenodo("4949969", "duncan2013_paramecium_holospora", "10.5061/dryad.3hq2c", manifest)

    # 2. Ferris et al. (2020) P. fluorescens - phage SBW25phi2 resource oscillation amplitudes at fixed mean
    fetch_figshare("12283799", "ferris2020_phage_oscillations", manifest)
    fetch_figshare("12283802", "ferris2020_phage_oscillations", manifest)

    # 3. Harrison et al. (2013) P. fluorescens - phage SBW25phi2 resource oscillation frequencies at fixed mean
    fetch_zenodo("4957209", "harrison2013_phage_frequency", "10.5061/dryad.8k015", manifest)

    # 4. Dallas, Krkosek & Drake (2018) D. dentifera - M. bicuspidata host density invasion threshold
    fetch_figshare("5133001", "dallas2018_metschnikowia_threshold", manifest)
    fetch_figshare("5745573", "dallas2018_metschnikowia_threshold", manifest)

    # 5. Dallas & Drake (2016) D. laevis - M. bicuspidata spore survival, filtering rate & thermal variability
    fetch_figshare("3615816", "dallas2016_metschnikowia_tempvar", manifest, skip=(".pdf",))

    # 6. Davenport et al. (2024a) Michigan lakes Daphnia/Ceriodaphnia 5-parasite spore banks & outbreaks
    fetch_zenodo("10359990", "davenport2024a_spores_outbreaks", "10.5061/dryad.tqjq2bw5m", manifest)

    # 7. Clerc, Ebert & Hall (2015) D. magna - Pasteuria ramosa within-host stage growth & castration
    fetch_zenodo("5011860", "clerc2015_pasteuria_stages", "10.5061/dryad.n1ks4", manifest)

    # 8. Tadiri (2023) D. magna - Pasteuria ramosa experimental jar epidemics & time to visible infection
    fetch_zenodo("8169416", "tadiri2023_pasteuria_epidemics", "10.5061/dryad.8931zcrwp", manifest)

    # 9. Krichel et al. (2023) D. magna - Ordospora colligata constant vs fluctuating temperature microcosms
    fetch_figshare("24111775", "krichel2023_ordospora_tempvar", manifest)
    fetch_figshare("24111778", "krichel2023_ordospora_tempvar", manifest)

    # 10. Dryad deposits without a Zenodo "dryad" data mirror (metadata saved; HTTP 401 recorded)
    check_dryad_blocked("10.5061/dryad.s7h44j1gb", "davenport2024b_lno", manifest)
    check_dryad_blocked("10.5061/dryad.nvx0k6f7r", "davenport2026_ordospora_pitchers", manifest)
    check_dryad_blocked("10.5061/dryad.qjq2bvqrt", "gutierrez2025_selective_culling", manifest)

    # 11. Weather for S6 field survey (6 SE Michigan lakes, 2021-05-01 to 2021-11-30)
    mi_lakes = [
        ("Bishop", 42.5004, -83.8411),
        ("Cedar", 42.3146, -84.0790),
        ("CrookedW", 42.3241, -84.1125),
        ("LittleAppleton", 42.5066, -83.8385),
        ("Mill", 42.3287, -84.0905),
        ("Walsh", 42.3375, -84.0798),
    ]
    for lk_name, lat, lon in mi_lakes:
        q = urllib.parse.urlencode({
            "lat": "%.4f" % lat,
            "lon": "%.4f" % lon,
            "vars": "tmax,tmin,prcp,srad,vp",
            "start": "2021-05-01",
            "end": "2021-11-30",
        })
        dm_url = "https://daymet.ornl.gov/single-pixel/api/data?" + q
        save_or_reuse(
            dm_url,
            os.path.join("weather", "daymet_%s_2021.csv" % lk_name),
            {
                "repository": "NASA ORNL DAAC Daymet",
                "doi": "10.3334/ORNLDAAC/2129",
                "license": "NASA Open Data / Public Domain",
                "version": "V4R1",
                "query_lat": lat,
                "query_lon": lon,
            },
            manifest,
        )

    ghcn_stations = [
        ("AnnArbor_USW00094889", "USW00094889"),
        ("Jackson_USW00014833", "USW00014833"),
    ]
    for stn_label, stn_id in ghcn_stations:
        q = urllib.parse.urlencode({
            "dataset": "daily-summaries",
            "stations": stn_id,
            "startDate": "2021-05-01",
            "endDate": "2021-11-30",
            "dataTypes": "TMAX,TMIN,TAVG,PRCP",
            "units": "metric",
            "includeStationLocation": "1",
            "includeStationName": "1",
            "format": "csv",
        })
        n_url = "https://www.ncei.noaa.gov/access/services/data/v1?" + q
        save_or_reuse(
            n_url,
            os.path.join("weather", "ghcnd_%s_2021.csv" % stn_label),
            {
                "repository": "NOAA NCEI GHCN-Daily",
                "doi": "10.7289/V5D21VHZ",
                "license": "U.S. Government Work / Public Domain",
                "version": "v1",
                "station_id": stn_id,
            },
            manifest,
        )

    # 12. Open-access EuropePMC XML & Zenodo 13259720 code metadata for laboratory/field documentation
    oa_docs = [
        ("PMC10491407_krichel2023.xml", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10491407/fullTextXML", "10.1371/journal.pbio.3002260", "CC BY 4.0"),
        ("PMC6093173_dallas2016.xml", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6093173/fullTextXML", "10.1002/ece3.2539", "CC BY 4.0"),
        ("PMC3712419_harrison2013.xml", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3712419/fullTextXML", "10.1098/rspb.2013.0937", "CC BY 3.0"),
        ("PMC4523363_vercken2015.xml", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4523363/fullTextXML", "10.1002/ece3.1505", "CC BY 4.0"),
        ("zenodo_13259720_davenport2024b_code.json", "https://zenodo.org/api/records/13259720", "10.5281/zenodo.13259720", "Open API metadata"),
    ]
    for fname, d_url, d_doi, d_lic in oa_docs:
        save_or_reuse(
            d_url,
            os.path.join("study_docs", fname),
            {
                "repository": "EuropePMC / Zenodo",
                "doi": d_doi,
                "license": d_lic,
                "version": "1",
            },
            manifest,
        )

    manifest_path = os.path.join(OUT, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote %s (%d entries)" % (manifest_path, len(manifest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
