#!/usr/bin/env python3
"""NEON: download the basic-package tables that protocol R14 needs (13 sites, 3 products).

Protocol: protocols/R14-neon-across-sites.md

Run (the credential is read from the environment and is never stored in this file):
    NEON_ACCESS_CREDENTIAL=... python3 -I data/fetch/fetch_neon_data.py --probe     # list one site-month per product, download nothing
    NEON_ACCESS_CREDENTIAL=... python3 -I data/fetch/fetch_neon_data.py             # download everything; safe to re-run (resumes)

What is fetched
    Sites    : BLAN DELA HARV KONZ MLBS ORNL OSBS SCBI SERC STEI TALL TREE UKFS
    Products : DP1.10072.001 small mammal box trapping   -> tables mam_pertrapnight, mam_perplotnight
               DP1.10093.001 ticks sampled by drag cloth -> tables tck_fielddata, tck_taxonomyProcessed
               DP1.10092.001 tick-borne pathogen status  -> table  tck_pathogen
    Package  : basic, every site-month the product catalogue lists, latest version on offer (released or
               provisional; the release tag of each site-month is written to the manifest).
    Also     : once per product, one `readme` and one `variables` file.
    Saved to : data/raw/neon/<product>/<site>/<file name as NEON gives it>   (git-ignored)
               data/raw/neon/<product>/<readme and variables>
               data/raw/neon/fetch_manifest.csv  (product, site, month, release, table, file, size, md5, status;
               no URLs, because the download URLs carry signatures)

How it behaves
    - GET https://data.neonscience.org/api/v0/products/<code> for the list of months per site, then
      GET .../data/<code>/<site>/<YYYY-MM>?package=basic for the file list of each site-month, then the
      pre-signed file URLs in that list. The credential goes only in the header X-API-Token and only to
      data.neonscience.org; the file URLs get no credential. Redirects are not followed.
    - At most 4 requests in flight (4 worker threads, one request each at a time), a short pause after
      each API call, 4 tries per request with back-off (2 s, 6 s, 18 s, or the server's Retry-After).
    - Stops (and says how far it got) if: the API answers 401 or 403; five requests in a row fail after
      their retries; the total would pass --max-gb (default 3); free disk space would fall under
      --min-free-gb (default 8); or --max-minutes (default 55) is used up.
    - A file already on disk with the size the listing gives is skipped, so a second run resumes.
      Files are written as <name>.part and renamed when complete; size and MD5 are checked when given.
    - Prints file names only: never the credential and never a download URL.

Licence and citation of the data: see the readme file saved with each product.
Standard library only.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "neon-data-fetch/1.0 (python-urllib)"
API_HOST = "data.neonscience.org"
API = "https://%s/api/v0" % API_HOST
HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
OUT = str((__import__('pathlib').Path(__file__).resolve().parent / '../raw/neon').resolve())
SITES = "BLAN DELA HARV KONZ MLBS ORNL OSBS SCBI SERC STEI TALL TREE UKFS".split()
TABLES = {
    "DP1.10072.001": ["mam_pertrapnight", "mam_perplotnight"],
    "DP1.10093.001": ["tck_fielddata", "tck_taxonomyProcessed"],
    "DP1.10092.001": ["tck_pathogen"],
}
WORKERS = 4
PAUSE = 0.15          # seconds after each API call, per worker
BACKOFF = [2, 6, 18]  # seconds between the 4 tries


class Refused(Exception):
    """The service refused (401/403) or kept failing: stop the whole run."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None  # a redirect becomes an HTTPError; headers are never forwarded to another host


OPENER = urllib.request.build_opener(NoRedirect)
STOP = threading.Event()
STOP_REASON = []
LOCK = threading.Lock()
STATE = {"bytes_new": 0, "bytes_seen": 0, "fail_streak": 0, "api_calls": 0, "file_gets": 0}
HOSTS = set()
ONCE = {}     # (product, kind) -> file name already taken for readme / variables
ROWS = []     # manifest rows


def stop(reason):
    with LOCK:
        if not STOP.is_set():
            STOP_REASON.append(reason)
        STOP.set()


def fetch(url, credential=None, what=""):
    """GET with retries. `what` is a label safe to print (never the URL). Returns bytes."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise RuntimeError("not https: %s" % what)
    hdr = {"User-Agent": UA}
    if credential is not None:
        if parts.hostname != API_HOST:
            raise RuntimeError("credential would leave %s: %s" % (API_HOST, what))
        hdr["X-API-Token"] = credential
        hdr["Accept"] = "application/json"
    with LOCK:
        HOSTS.add(parts.hostname)
    last = "no attempt"
    for attempt in range(len(BACKOFF) + 1):
        if STOP.is_set():
            raise Refused("stopped")
        wait = None
        try:
            with OPENER.open(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                body = r.read()
            with LOCK:
                STATE["fail_streak"] = 0
            return body
        except urllib.error.HTTPError as e:
            last = "HTTP %d" % e.code
            if e.code in (401, 403) and parts.hostname == API_HOST:
                stop("the API answered %s for %s" % (last, what))
                raise Refused(last)
            if e.code == 404:
                break
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra and ra.isdigit():
                wait = min(int(ra), 120)
        except Exception as e:  # noqa: BLE001  (class name only: a message could carry a URL)
            last = type(e).__name__
        if attempt < len(BACKOFF):
            time.sleep(wait if wait is not None else BACKOFF[attempt])
    with LOCK:
        STATE["fail_streak"] += 1
        streak = STATE["fail_streak"]
    if streak >= 5:
        stop("five requests in a row failed after retries (last: %s, %s)" % (last, what))
    raise RuntimeError("%s (%s)" % (last, what))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wanted(product, name):
    """Return ('table', table) or ('readme'|'variables', None) or None."""
    for t in TABLES[product]:
        if (".%s." % t) in name and name.endswith(".csv"):
            return ("table", t)
    if ".readme." in name and name.endswith(".txt"):
        return ("readme", None)
    if ".variables." in name and name.endswith(".csv"):
        return ("variables", None)
    return None


def save(product, site, month, release, fl, kind, table, args):
    name = os.path.basename(fl["name"])
    size = int(fl.get("size") or 0)
    md5 = fl.get("md5") or ""
    if kind == "table":
        dest = os.path.join(OUT, product, site, name)
    else:
        with LOCK:
            if (product, kind) in ONCE:
                return
            ONCE[(product, kind)] = name
        dest = os.path.join(OUT, product, name)
    row = [product, site, month, release, table or kind, name, size, md5]
    if os.path.exists(dest) and (size == 0 or os.path.getsize(dest) == size):
        with LOCK:
            STATE["bytes_seen"] += os.path.getsize(dest)
            ROWS.append(row + ["present"])
        return
    with LOCK:
        total = STATE["bytes_seen"] + STATE["bytes_new"] + size
    if total > args.max_gb * 1e9:
        stop("total would pass %.1f GB" % args.max_gb)
        raise Refused("size cap")
    if shutil.disk_usage(OUT).free - size < args.min_free_gb * 1e9:
        stop("free disk space would fall under %.1f GB" % args.min_free_gb)
        raise Refused("disk")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        body = fetch(fl["url"], None, "%s %s %s %s" % (product, site, month, name))
        with LOCK:
            STATE["file_gets"] += 1
        if size and len(body) != size:
            raise RuntimeError("size %d, listing says %d (%s)" % (len(body), size, name))
        if md5 and hashlib.md5(body).hexdigest() != md5.lower():
            raise RuntimeError("MD5 differs from the listing (%s)" % name)
        with open(dest + ".part", "wb") as f:
            f.write(body)
        os.replace(dest + ".part", dest)
        with LOCK:
            STATE["bytes_new"] += len(body)
            ROWS.append(row + ["downloaded"])
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001
        if kind != "table":
            with LOCK:
                ONCE.pop((product, kind), None)
        with LOCK:
            ROWS.append(row + ["FAILED: %s" % e])


def site_month(product, site, month, credential, args):
    if STOP.is_set():
        return "skipped"
    if time.time() > args.deadline:
        stop("time budget of %d minutes used up" % args.max_minutes)
        return "skipped"
    try:
        raw = fetch("%s/data/%s/%s/%s?package=basic" % (API, product, site, month), credential,
                    "listing %s %s %s" % (product, site, month))
        with LOCK:
            STATE["api_calls"] += 1
        time.sleep(PAUSE)
        d = json.loads(raw)["data"]
        release = d.get("release") or ""
        got = set()
        for fl in d.get("files") or []:
            w = wanted(product, os.path.basename(fl["name"]))
            if not w:
                continue
            if STOP.is_set():
                return "skipped"
            save(product, site, month, release, fl, w[0], w[1], args)
            if w[0] == "table":
                got.add(w[1])
        with LOCK:
            for t in TABLES[product]:
                if t not in got:
                    ROWS.append([product, site, month, release, t, "", 0, "", "not listed"])
        return "ok"
    except Refused:
        return "skipped"
    except Exception as e:  # noqa: BLE001
        with LOCK:
            ROWS.append([product, site, month, "", "", "", 0, "", "LISTING FAILED: %s" % e])
        return "failed"


def months_by_site(product, credential):
    d = json.loads(fetch("%s/products/%s" % (API, product), credential, "catalogue %s" % product))["data"]
    out = {}
    for s in d.get("siteCodes") or []:
        if s["siteCode"] in SITES:
            out[s["siteCode"]] = sorted(s["availableMonths"])
    return out


def write_manifest():
    ROWS.sort(key=lambda r: (r[0], r[1], r[2], r[4], r[5]))
    path = os.path.join(OUT, "fetch_manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "site", "month", "release", "table", "file", "size", "md5", "status"])
        w.writerows(ROWS)
    return path


def on_disk():
    out = {}
    for product in TABLES:
        n = b = 0
        for root, _dirs, files in os.walk(os.path.join(OUT, product)):
            for fn in files:
                if fn.endswith(".part") or fn.startswith("."):
                    continue
                n += 1
                b += os.path.getsize(os.path.join(root, fn))
        out[product] = (n, b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true", help="list one site-month per product; download nothing")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--max-minutes", type=int, default=55)
    args = ap.parse_args()
    credential = os.environ.get("NEON_ACCESS_CREDENTIAL")
    if not credential:
        raise SystemExit("NEON_ACCESS_CREDENTIAL is not set in the environment; nothing downloaded.")
    os.makedirs(OUT, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    print("free disk space: %.1f GB" % (free / 1e9))
    if free - args.max_gb * 1e9 < args.min_free_gb * 1e9 and not args.probe:
        print("note: free space minus the %.1f GB cap is under %.1f GB; the run stops if free space gets that low"
              % (args.max_gb, args.min_free_gb))
    t0 = time.time()
    args.deadline = t0 + 60 * args.max_minutes

    if args.probe:
        for product in TABLES:
            ms = months_by_site(product, credential)
            site = "HARV"
            month = ms[site][len(ms[site]) // 2]
            d = json.loads(fetch("%s/data/%s/%s/%s?package=basic" % (API, product, site, month), credential,
                                 "listing %s %s %s" % (product, site, month)))["data"]
            print("\n%s %s %s  release=%s  site-months in the 13 sites: %d"
                  % (product, site, month, d.get("release"), sum(len(v) for v in ms.values())))
            for fl in d.get("files") or []:
                name = os.path.basename(fl["name"])
                print("   %-9s %9d bytes  host=%s  md5=%s  %s"
                      % (wanted(product, name)[0] if wanted(product, name) else "-", int(fl.get("size") or 0),
                         urllib.parse.urlsplit(fl["url"]).hostname, "yes" if fl.get("md5") else "no", name))
        return 0

    jobs = []
    for product in TABLES:
        ms = months_by_site(product, credential)
        missing = [s for s in SITES if s not in ms]
        n = sum(len(v) for v in ms.values())
        print("%s: %d site-months at %d sites%s" % (product, n, len(ms), (" (not in catalogue: %s)" % missing) if missing else ""))
        for site in SITES:
            for month in ms.get(site, []):
                jobs.append((product, site, month))
    print("site-months to visit: %d" % len(jobs), flush=True)

    counts = {"ok": 0, "failed": 0, "skipped": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(lambda j: site_month(j[0], j[1], j[2], credential, args), jobs):
            counts[res] += 1
            done += 1
            if done % 200 == 0:
                with LOCK:
                    print("  %4d / %d site-months, %.1f MB new, %.0f s"
                          % (done, len(jobs), STATE["bytes_new"] / 1e6, time.time() - t0), flush=True)

    path = write_manifest()
    print("\nfinished in %.0f s; listings: %d ok, %d failed, %d not visited" % (time.time() - t0, counts["ok"], counts["failed"], counts["skipped"]))
    print("API listing calls: %d; file downloads: %d (%.1f MB new); hosts contacted: %s"
          % (STATE["api_calls"], STATE["file_gets"], STATE["bytes_new"] / 1e6, ", ".join(sorted(HOSTS))))
    bad = [r for r in ROWS if r[8].startswith(("FAILED", "LISTING FAILED"))]
    notl = [r for r in ROWS if r[8] == "not listed"]
    print("failed files or listings: %d" % len(bad))
    for r in bad[:60]:
        print("   %s %s %s %s  %s" % (r[0], r[1], r[2], r[5] or r[4], r[8]))
    print("site-months where a wanted table was not in the listing: %d (see the manifest, status 'not listed')" % len(notl))
    rel = {}
    for r in ROWS:
        if r[8] in ("present", "downloaded") and r[4] not in ("readme", "variables"):
            rel[(r[0], r[3])] = rel.get((r[0], r[3]), 0) + 1
    for k in sorted(rel):
        print("   %s  release tag %-14s %5d table files" % (k[0], k[1], rel[k]))
    tot = 0
    for product, (n, b) in on_disk().items():
        tot += b
        print("on disk  %s  %5d files  %8.1f MB" % (product, n, b / 1e6))
    print("on disk  total %.1f MB; manifest: %s" % (tot / 1e6, os.path.relpath(path, os.path.dirname(HERE))))
    if STOP.is_set():
        print("STOPPED EARLY: %s. Re-run the same command to resume (files on disk are skipped)." % "; ".join(STOP_REASON))
        return 2
    if bad:
        print("INCOMPLETE: re-run the same command to retry the failed items.")
        return 1
    print("COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
