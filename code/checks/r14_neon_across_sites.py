"""Run R14 (protocols/R14-neon-across-sites.md): across NEON sites, is the infection of questing
Ixodes scapularis nymphs with Borrelia burgdorferi lower where mice (genus Peromyscus) vary more
from year to year?

Data: NEON products DP1.10072.001 (small mammal box trapping), DP1.10093.001 (ticks sampled using
drag cloths) and DP1.10092.001 (tick pathogen status), basic package, 13 sites, fetched by
data/fetch/fetch_neon_data.py into ../../data/raw/neon/ (read there; not copied, not tracked).

Rule 10 of study rules governs: data set ranges and check patterns.  Nothing is fitted to the theory
and no parameter of the theory is tuned.  One pass.

What is computed, and nothing else
  - Mouse index per site and year; sigma_site and r1 per site (all 13 sites).
  - NIP_site and yearly NIP per site (share of tested I. scapularis nymphs positive for B. burgdorferi).
  - T1  Spearman rank correlation of NIP_site with sigma_site over the included sites; permutation p.
  - T2  sd of ln(yearly NIP) < sd of ln(mouse index) over the same years, site by site.
  - T3  pooled within-site slope of ln(NIP in year t) on ln(mouse index in year t - 1).
  - Without a hypothesis: sigma_site and r1 for every site; latitude and its rank correlations.

Readings of the protocol, fixed on 14 November 2026 at 10:17 CST, after the download was complete
and BEFORE any index, prevalence, sigma or correlation was computed.  What I had looked at by then:
the three readme files and the three variables files; the list of distinct values (no counts by
site, no results) of trapStatus, of the taxon codes near Peromyscus, of tagID shapes, fate, tick
sexOrAge, Ixodes taxon codes, identifier shapes, testResult, individualCount, sampleCondition and
laboratoryName; the number of pathogen-table rows per pathogen name and collection year summed
over all sites, tick species and results; that every pathogen row's subsampleID occurs in the tick
taxonomy table; and the header and first rows of one file per table.

  1. Genus Peromyscus = rows of mam_pertrapnight whose scientificName has "Peromyscus" as its first
     word.  In these files that is the taxonID codes PELE, PEMA, PEGO, PEPO, PELEPEMA, PEGOPELE and
     PESP (species pairs and "Peromyscus sp." count).  Podomys floridanus (POFL), Ochrotomys
     nuttalli (OCNU) and Reithrodontomys are other genera and do not count.
  2. A distinct tagged individual = a distinct non-empty tagID string (the variables file: "Domain-
     level unique identifier of tag used to mark the individual") on a Peromyscus row, within the
     site and calendar year of collectDate.  Rows without a tagID are not individuals.  An animal
     caught in two years counts in each.  No attempt is made to recognise odd-looking tagIDs.
  3. One trap-night of effort = one distinct (plotID, collectDate, trapCoordinate) whose trapStatus
     code is 2, 3, 4, 5 or 6 (the trap was set: disturbed, spoor left, capture(s), or set and
     empty).  A disturbed or sprung trap (codes 2, 3) counts as a FULL trap-night; no half-night
     correction.  Code 1 (trap not set), code 0 (no data) and a blank status are not effort.  Two
     rows for one trap on one night (several captures, code 4) are one trap-night.  The
     mam_perplotnight table has no count of traps, so effort comes from mam_pertrapnight only and
     the per-plot-night table is not used.
  4. Mouse index (site, year) = 100 x individuals / trap-nights, all plots and bouts of the calendar
     year pooled; it exists if trap-nights >= 500.  Provisional and partial years are treated like
     any other.  sigma_site = sd (n - 1) of ln(index) over the years with index > 0; needs at least
     five such years; years with a zero index are left out and counted.  r1 = ordinary correlation
     of ln(index) in year t with year t + 1 over the pairs of consecutive calendar years that both
     have a positive index; given only with at least three pairs.
  5. Rows with the same uid in a table are read once.  If a site-month has two files of one table,
     the one with the later publication stamp in its name is read.
  6. A tested tick = one testingID in tck_pathogen (individualCount is 1 on every row).  Its species
     and life stage come from tck_taxonomyProcessed through subsampleID: acceptedTaxonID == IXOSCA
     and sexOrAge == Nymph.  The codes written inside the subsampleID text (".IXOSCA.N") are only
     compared with that and disagreements counted; the taxonomy table governs.  The laboratory's own
     "Ixodes scapularis" test (2022 on) is not used.
  7. Borrelia burgdorferi sensu lato = testPathogenName exactly "Borrelia burgdorferi sensu lato"
     (the readme names it as the target for Ixodes nymphs; tested in every year 2014-2025).  This is
     the broader one and is used for T1 to T3.  Sensu stricto = testPathogenName exactly "Borrelia
     burgdorferi" (rows exist only for 2014), reported separately.  "Borrelia sp." (a genus-level
     test), B. mayonii, B. miyamotoi and B. lonestari are other tests and do not count.
  8. A nymph is "tested" for a name if it has a row of that name with testResult Positive or
     Negative; positive if any such row is Positive.  A nymph whose "HardTick DNA Quality" row is
     not Positive (the extraction control failed) is left out and counted; a nymph with no such row
     is kept.  sampleCondition is not used.  Year = calendar year of collectDate (UTC).
  9. NIP_site = positives / tested, all years pooled.  Yearly NIP exists if at least 20 nymphs were
     tested in the year.  A yearly NIP of zero is left out of logarithms and counted.
 10. Included site (T1): at least 100 nymphs tested in total, at least four calendar years with at
     least one nymph tested, and a sigma_site.
 11. T1: Spearman's rho with mid-ranks for ties.  Permutation p: every ordering if 9 sites or fewer
     (exact), otherwise 200,000 random orderings, seed (20261114, 1), p = (1 + count)/(1 + N).
     One-sided p for "negative" = share of orderings with rho <= observed; for "positive", >=.
     Fewer than five included sites: not run.
 12. T2: at an included site, the years used are those with a yearly NIP > 0 and a mouse index > 0
     in the same year; the site enters if there are at least four.  Violation = sd ln NIP >= sd ln
     index.  T2 fails if violations > a quarter of the sites entered; no site entered = not testable.
 13. T3: pairs (ln index in t - 1, ln NIP in t) at the included sites, both positive; a site enters
     with at least two pairs; least squares with one intercept per site.  Standard error = sd of the
     slope over 2,000 resamples of the pairs, drawn with replacement within each site, seed
     (20261114, 3); this standard error is the one the rule uses.  The textbook least-squares
     standard error is printed beside it as a check of the code only.  pbar = mean of the yearly NIP
     values (zeros included) of the included sites.  Holds if b + 2 se >= 0 and b - 2 se <= 1 - pbar.
     Needs (pairs - sites entered) >= 3, otherwise not run.
 14. Latitude of a site = mean decimalLatitude of its small-mammal plots (mam_pertrapnight).  Rank
     correlations of latitude with sigma_site (all sites with a sigma, and the included sites) and
     with NIP_site (included sites) are printed without a test.

Code checks (study rules rule 7; tolerances stated here, before the run; they do not touch the data):
  K1  sd and r1 against hand-worked values, 1e-12.
  K2  Spearman's rho against hand-worked values, with and without ties, 1e-12.
  K3  exact permutation test, 6 sites: p = 1/720 at rho = -1; the smallest |rho| with one-sided
      p < 0.05 is 0.829 (standard table; the protocol's "0.83"), 0.001.
  K4  within-site slope against numpy least squares with site dummies, synthetic numbers, 1e-9.
  K5  trap-night and individual counting on twelve hand-made trap rows: 8 trap-nights, 3 mice.
  K6  nymph counting on six hand-made ticks: 2 tested, 1 positive, 1 left out by the DNA control.

Standard library and numpy only.  No network and no token.

Run from problems/dobson-r0:
  python3 -I checks/r14_neon_across_sites.py > results/r14-neon-across-sites_20261114.txt
(an optional argument names another raw-data folder; used once for a dry run on made-up files)
"""
from __future__ import annotations

import csv
import datetime
import itertools
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

PROBLEM = (__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()
DEFAULT_RAW = (__import__('pathlib').Path(__file__).resolve().parent / '../data/raw/neon').resolve()
SITES = "BLAN DELA HARV KONZ MLBS ORNL OSBS SCBI SERC STEI TALL TREE UKFS".split()
MAM, TCK, PAT = "DP1.10072.001", "DP1.10093.001", "DP1.10092.001"
SEED = 20261114
NBOOT = 2000
NMC = 200000
SET_CODES = {"2", "3", "4", "5", "6"}
SL = "Borrelia burgdorferi sensu lato"
SS = "Borrelia burgdorferi"
DNAQ = "HardTick DNA Quality"
MIN_TN = 500
MIN_YEARS = 5
MIN_YEARLY = 20
MIN_TOTAL = 100
MIN_TEST_YEARS = 4
NAME_RE = re.compile(r"^NEON\.D\d\d\.([A-Z]{4})\.(DP1\.\d{5}\.\d{3})\.([A-Za-z_]+)\.(\d{4}-\d{2})\.basic\.(\d{8}T\d{6}Z)\.csv$")


# ----------------------------------------------------------------------------- statistics
def sd_ln(values):
    return float(np.std(np.log(np.asarray(values, float)), ddof=1))


def r1_of(series):
    """series: {year: value}.  Correlation between year t and t + 1 over consecutive pairs."""
    xs, ys = [], []
    for y in sorted(series):
        if y + 1 in series:
            xs.append(series[y])
            ys.append(series[y + 1])
    if len(xs) < 3:
        return None, len(xs)
    x, yv = np.asarray(xs, float), np.asarray(ys, float)
    if x.std() == 0 or yv.std() == 0:
        return None, len(xs)
    return float(np.corrcoef(x, yv)[0, 1]), len(xs)


def midranks(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = midranks(x), midranks(y)
    if len(rx) < 3 or rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_perm(x, y, rng):
    """rho, P(rho* <= rho), P(rho* >= rho), description of the permutation set."""
    rx, ry = midranks(x), midranks(y)
    n = len(rx)
    cx, cy = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((cx ** 2).sum() * (cy ** 2).sum()))
    if den == 0:
        return float("nan"), float("nan"), float("nan"), "no variation in one variable"
    rho = float((cx * cy).sum() / den)
    eps = 1e-12
    if n <= 9:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
        vals = (cy[perms] * cx).sum(axis=1) / den
        return rho, float(np.mean(vals <= rho + eps)), float(np.mean(vals >= rho - eps)), "exact, all %d orderings" % len(perms)
    low = high = 0
    left = NMC
    while left > 0:
        m = min(left, 20000)
        perms = rng.permuted(np.tile(np.arange(n), (m, 1)), axis=1)
        vals = (cy[perms] * cx).sum(axis=1) / den
        low += int(np.sum(vals <= rho + eps))
        high += int(np.sum(vals >= rho - eps))
        left -= m
    return rho, (1 + low) / (1 + NMC), (1 + high) / (1 + NMC), "%d random orderings" % NMC


def within_slope(groups):
    """groups: list of (x, y) arrays, one per site.  Least squares with one intercept per site.
    Returns (slope, Sxx, residual sum of squares, pairs, sites) or None."""
    sxx = sxy = 0.0
    n = k = 0
    for x, y in groups:
        if len(x) < 2:
            continue
        dx, dy = x - x.mean(), y - y.mean()
        sxx += float((dx * dx).sum())
        sxy += float((dx * dy).sum())
        n += len(x)
        k += 1
    if sxx <= 0:
        return None
    b = sxy / sxx
    rss = 0.0
    for x, y in groups:
        if len(x) < 2:
            continue
        rss += float((((y - y.mean()) - b * (x - x.mean())) ** 2).sum())
    return b, sxx, rss, n, k


# ----------------------------------------------------------------------------- reading
def files_for(raw, product, table):
    """[(site, month, path)] with one file per site-month (the latest stamp), and the number passed over."""
    chosen, passed = {}, 0
    for site in SITES:
        d = raw / product / site
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.csv")):
            m = NAME_RE.match(p.name)
            if not m or m.group(3) != table or m.group(1) != site:
                continue
            key = (site, m.group(4))
            if key in chosen:
                passed += 1
            if key not in chosen or m.group(5) > chosen[key][0]:
                chosen[key] = (m.group(5), p)
    return [(k[0], k[1], v[1]) for k, v in sorted(chosen.items())], passed


def rows_of(files):
    for _site, _month, path in files:
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)


# ----------------------------------------------------------------------------- mice
def is_peromyscus(scientific_name):
    w = scientific_name.split()
    return bool(w) and w[0] == "Peromyscus"


def mouse_counts(rows):
    """Trap-nights and distinct tagged Peromyscus per (site, year), by decisions 1 to 3 and 5."""
    seen = set()
    traps, tags = defaultdict(set), defaultdict(set)
    tally = Counter()
    plot_lat = {}
    for r in rows:
        if r["uid"] in seen:
            tally[(r["siteID"], "repeated uid, read once")] += 1
            continue
        seen.add(r["uid"])
        site = r["siteID"]
        date = r["collectDate"][:10]
        if not date[:4].isdigit():
            tally[(site, "rows without a collectDate, not read")] += 1
            continue
        year = int(date[:4])
        code = r["trapStatus"].strip()[:1]
        tally[(site, "rows")] += 1
        tally[(site, "status " + (code if code else "blank"))] += 1
        if code in SET_CODES:
            traps[(site, year)].add((r["plotID"], date, r["trapCoordinate"].strip()))
        if is_peromyscus(r["scientificName"]):
            tally[(site, "Peromyscus rows")] += 1
            tag = r["tagID"].strip()
            if tag:
                tags[(site, year)].add(tag)
                if code not in ("4", "5"):
                    tally[(site, "tagged Peromyscus on a row whose status is not a capture")] += 1
            else:
                tally[(site, "Peromyscus rows without a tagID")] += 1
        lat = (r.get("decimalLatitude") or "").strip()
        if lat and (site, r["plotID"]) not in plot_lat:
            plot_lat[(site, r["plotID"])] = float(lat)
    effort = {k: len(v) for k, v in traps.items()}
    mice = {k: len(tags.get(k, ())) for k in effort}
    for k in tags:
        mice.setdefault(k, len(tags[k]))
        effort.setdefault(k, 0)
    return effort, mice, tally, plot_lat


# ----------------------------------------------------------------------------- ticks
def nymph_counts(tax_rows, pat_rows):
    """Per (site, year): I. scapularis nymphs tested and positive, for SL and SS (decisions 6 to 8)."""
    tax = {}
    info = Counter()
    seen = set()
    for r in tax_rows:
        if r["uid"] in seen:
            info["taxonomy rows with a repeated uid, read once"] += 1
            continue
        seen.add(r["uid"])
        sub = r["subsampleID"].strip()
        val = (r["acceptedTaxonID"].strip(), r["sexOrAge"].strip())
        if sub in tax and tax[sub] != val:
            info["subsampleID with two different taxonomy records (first kept)"] += 1
            continue
        tax[sub] = val
    ticks = {}
    seen = set()
    names = Counter()
    for r in pat_rows:
        if r["uid"] in seen:
            info["pathogen rows with a repeated uid, read once"] += 1
            continue
        seen.add(r["uid"])
        if not r["collectDate"][:4].isdigit():
            info["pathogen rows without a collectDate, not read"] += 1
            continue
        info["pathogen rows"] += 1
        names[r["testPathogenName"]] += 1
        tid = r["testingID"].strip()
        t = ticks.get(tid)
        if t is None:
            t = ticks[tid] = {"site": r["siteID"], "year": int(r["collectDate"][:4]), "sub": r["subsampleID"].strip(), "res": defaultdict(set)}
        elif t["sub"] != r["subsampleID"].strip() or t["site"] != r["siteID"]:
            info["testingID seen with two subsampleIDs or sites (first kept)"] += 1
        if r["testResult"] in ("Positive", "Negative"):
            t["res"][r["testPathogenName"]].add(r["testResult"])
        else:
            info["rows with a testResult other than Positive or Negative (ignored)"] += 1
    out = {SL: defaultdict(lambda: [0, 0]), SS: defaultdict(lambda: [0, 0])}
    who = {SL: Counter(), SS: Counter()}
    qc_out = Counter()
    for tid, t in ticks.items():
        info["ticks (distinct testingID)"] += 1
        taxon, stage = tax.get(t["sub"], ("not in taxonomy table", "?"))
        if taxon == "not in taxonomy table":
            info["ticks whose subsampleID is not in the taxonomy table"] += 1
        parts = t["sub"].split(".")
        text = (parts[2], parts[3]) if len(parts) >= 4 else ("?", "?")
        if taxon != "not in taxonomy table" and (text[0] != taxon or text[1] != stage[:1]):
            info["ticks whose subsampleID text disagrees with the taxonomy table"] += 1
        for name in (SL, SS):
            res = t["res"].get(name)
            if not res:
                continue
            who[name][(taxon, stage)] += 1
            if not (taxon == "IXOSCA" and stage == "Nymph"):
                continue
            dq = t["res"].get(DNAQ)
            if dq is None:
                info["I. scapularis nymphs tested for %s with no DNA-quality row (kept)" % name] += 1
            elif "Positive" not in dq:
                qc_out[(name, t["site"])] += 1
                continue
            if len(res) > 1:
                info["nymphs with both a Positive and a Negative row for %s (counted positive)" % name] += 1
            cell = out[name][(t["site"], t["year"])]
            cell[0] += 1
            cell[1] += 1 if "Positive" in res else 0
    return out, who, qc_out, info, names


# ----------------------------------------------------------------------------- code checks
def code_checks():
    ok = True
    print("0. Code checks that do not touch the data")
    s = sd_ln(np.exp([1.0, 2.0, 4.0, 3.0]))
    r, n = r1_of({2001: 1.0, 2002: 2.0, 2003: 4.0, 2004: 3.0})
    hand_s, hand_r = math.sqrt(5 / 3), 1 / math.sqrt(2 * 42 / 9)
    good = abs(s - hand_s) < 1e-12 and abs(r - hand_r) < 1e-12 and n == 3
    ok &= good
    print("  K1  sd %.12f (hand %.12f); r1 %.12f from %d pairs (hand %.12f)   [%s]" % (s, hand_s, r, n, hand_r, "ok" if good else "FAILED"))
    a = spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    b = spearman([1, 2, 2, 4], [1, 2, 3, 4])
    hand_b = 4.5 / math.sqrt(22.5)
    good = abs(a - 0.8) < 1e-12 and abs(b - hand_b) < 1e-12
    ok &= good
    print("  K2  Spearman %.12f (hand 0.8); with a tie %.12f (hand %.12f)   [%s]" % (a, b, hand_b, "ok" if good else "FAILED"))
    rng = np.random.default_rng([SEED, 0])
    x = np.arange(6.0)
    rho, plow, _phigh, how = spearman_perm(x, x[::-1], rng)
    perms = np.array(list(itertools.permutations(range(6))))
    cx = x - x.mean()
    vals = np.round((cx[perms] * cx).sum(axis=1) / float((cx ** 2).sum()), 10)
    crit = min(v for v in set(vals.tolist()) if v > 0 and np.mean(vals >= v - 1e-12) < 0.05)
    good = abs(rho + 1) < 1e-12 and abs(plow - 1 / 720) < 1e-12 and abs(crit - 0.829) < 0.001
    ok &= good
    print("  K3  six sites: rho %.3f, p %.6f (1/720 = %.6f; %s); smallest |rho| with one-sided p < 0.05 is %.4f (table 0.829), p there %.4f   [%s]"
          % (rho, plow, 1 / 720, how, crit, float(np.mean(vals >= crit - 1e-12)), "ok" if good else "FAILED"))
    g = []
    cols_x, cols_y, dummies = [], [], []
    for i, m in enumerate((4, 6, 3)):
        xx = rng.normal(size=m)
        yy = 0.4 * xx + i + rng.normal(scale=0.3, size=m)
        g.append((xx, yy))
        cols_x += list(xx)
        cols_y += list(yy)
        dummies += [i] * m
    b_fast = within_slope(g)[0]
    design = np.zeros((len(cols_x), 4))
    design[:, 0] = cols_x
    for row, i in enumerate(dummies):
        design[row, 1 + i] = 1
    b_ls = float(np.linalg.lstsq(design, np.array(cols_y), rcond=None)[0][0])
    good = abs(b_fast - b_ls) < 1e-9
    ok &= good
    print("  K4  within-site slope, synthetic: %.12f against least squares with dummies %.12f   [%s]" % (b_fast, b_ls, "ok" if good else "FAILED"))

    def trap(uid, date, coord, status, name="", tag=""):
        return {"uid": uid, "siteID": "TEST", "plotID": "TEST_001", "collectDate": date, "trapCoordinate": coord,
                "trapStatus": status, "scientificName": name, "tagID": tag, "decimalLatitude": "40.0"}
    rows = [
        trap("1", "2020-06-01", "A1", "5 - capture", "Peromyscus leucopus", "T1"),
        trap("2", "2020-06-01", "A2", "6 - trap set and empty"),
        trap("3", "2020-06-01", "A3", "1 - trap not set"),
        trap("4", "2020-06-01", "A4", "2 - trap disturbed/door closed but empty"),
        trap("5", "2020-06-01", "A5", "4 - more than 1 capture in one trap", "Peromyscus leucopus", "T2"),
        trap("6", "2020-06-01", "A5", "4 - more than 1 capture in one trap", "Peromyscus maniculatus", "T3"),
        trap("7", "2020-06-02", "A1", "5 - capture", "Peromyscus leucopus", "T1"),
        trap("8", "2020-06-02", "A2", "5 - capture", "Microtus pennsylvanicus", "T9"),
        trap("9", "2020-06-02", "A3", "5 - capture", "Peromyscus sp.", ""),
        trap("10", "2020-06-02", "A4", "3 - trap door open or closed w/ spoor left"),
        trap("2", "2020-06-01", "A2", "6 - trap set and empty"),
        trap("12", "2020-06-02", "A5", ""),
    ]
    eff, mice, _tally, _lat = mouse_counts(rows)
    good = eff == {("TEST", 2020): 8} and mice == {("TEST", 2020): 3}
    ok &= good
    print("  K5  twelve hand-made trap rows: %d trap-nights (hand 8), %d tagged Peromyscus (hand 3)   [%s]"
          % (eff[("TEST", 2020)], mice[("TEST", 2020)], "ok" if good else "FAILED"))
    tax = [
        {"uid": "a", "subsampleID": "T_001.20200601.IXOSCA.N", "acceptedTaxonID": "IXOSCA", "sexOrAge": "Nymph"},
        {"uid": "b", "subsampleID": "T_001.20200601.IXOSCA.F", "acceptedTaxonID": "IXOSCA", "sexOrAge": "Female"},
        {"uid": "c", "subsampleID": "T_001.20200601.AMBAME.N", "acceptedTaxonID": "AMBAME", "sexOrAge": "Nymph"},
    ]

    def test(uid, sub, num, name, res):
        return {"uid": uid, "siteID": "TEST", "collectDate": "2020-06-01T15:00Z", "subsampleID": sub,
                "testingID": "%s.T.%s" % (sub, num), "testPathogenName": name, "testResult": res}
    n_, f_, a_ = tax[0]["subsampleID"], tax[1]["subsampleID"], tax[2]["subsampleID"]
    pat = [
        test("1", n_, "001", SL, "Positive"), test("2", n_, "001", DNAQ, "Positive"),
        test("3", n_, "002", SL, "Negative"), test("4", n_, "002", DNAQ, "Positive"),
        test("5", n_, "003", SL, "Negative"), test("6", n_, "003", DNAQ, "Negative"),
        test("7", n_, "004", "Anaplasma phagocytophilum", "Positive"), test("8", n_, "004", DNAQ, "Positive"),
        test("9", f_, "001", SL, "Positive"), test("10", f_, "001", DNAQ, "Positive"),
        test("11", a_, "001", SL, "Negative"), test("12", a_, "001", DNAQ, "Positive"),
    ]
    out, _who, qc, _info, _names = nymph_counts(tax, pat)
    cell = out[SL][("TEST", 2020)]
    good = cell == [2, 1] and qc[(SL, "TEST")] == 1 and len(out[SL]) == 1
    ok &= good
    print("  K6  six hand-made ticks: %d nymphs tested, %d positive, %d left out by the DNA control (hand 2, 1, 1)   [%s]"
          % (cell[0], cell[1], qc[(SL, "TEST")], "ok" if good else "FAILED"))
    return ok


# ----------------------------------------------------------------------------- report
def fmt(v, spec="%.3f", none="  -  "):
    return none if v is None or (isinstance(v, float) and math.isnan(v)) else spec % v


def main():
    raw = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RAW
    t0 = time.time()
    now = datetime.datetime.now().astimezone()
    if raw != DEFAULT_RAW:
        print("*** DRY RUN ON MADE-UP FILES in %s: NOT A RESULT ***" % raw)
    print("R14: NEON across sites.  Is infection of I. scapularis nymphs lower where Peromyscus vary more?  Started %s."
          % now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("A pattern check across sites; nothing is fitted.  Protocol: protocols/R14-neon-across-sites.md")
    print("Raw data read from: data/raw/neon/ (products %s, %s, %s; 13 sites)\n" % (MAM, TCK, PAT))
    if not code_checks():
        print("\nA CODE CHECK FAILED: the data were not read.")
        return 1
    sys.stdout.flush()

    # ---------------------------------------------------------------- 1. what was read
    mam_files, mam_passed = files_for(raw, MAM, "mam_pertrapnight")
    tax_files, tax_passed = files_for(raw, TCK, "tck_taxonomyProcessed")
    pat_files, pat_passed = files_for(raw, PAT, "tck_pathogen")
    print("\n1. What was read")
    print("  mam_pertrapnight      %5d site-month files (%d older duplicates passed over)" % (len(mam_files), mam_passed))
    print("  tck_taxonomyProcessed %5d site-month files (%d older duplicates passed over)" % (len(tax_files), tax_passed))
    print("  tck_pathogen          %5d site-month files (%d older duplicates passed over)" % (len(pat_files), pat_passed))
    print("  not read: mam_perplotnight (no count of traps in it) and tck_fielddata (no definition of the protocol uses drag counts)")
    effort, mice, tally, plot_lat = mouse_counts(rows_of(mam_files))
    out, who, qc_out, info, names = nymph_counts(rows_of(tax_files), rows_of(pat_files))
    lat = {}
    for s in SITES:
        v = [x for (ss, _p), x in plot_lat.items() if ss == s]
        lat[s] = float(np.mean(v)) if v else None

    labels = ["rows", "status 1", "status 2", "status 3", "status 4", "status 5", "status 6", "status blank", "status 0",
              "Peromyscus rows", "Peromyscus rows without a tagID", "tagged Peromyscus on a row whose status is not a capture",
              "repeated uid, read once"]
    short = ["rows", "st.1", "st.2", "st.3", "st.4", "st.5", "st.6", "blank", "st.0", "Pero rows", "no tag", "tag,no capt", "rep.uid"]
    print("\n  Small-mammal rows by site (st.N = rows with trapStatus code N: 1 not set, 2 disturbed, 3 spoor, 4 several captures,")
    print("  5 capture, 6 set and empty; 'no tag' = Peromyscus rows without a tagID; 'tag,no capt' = tagged Peromyscus on a row")
    print("  whose status is not 4 or 5; 'rep.uid' = rows with a repeated uid, read once)")
    print("  site " + " ".join("%10s" % h for h in short) + "  mammal plots")
    for s in SITES:
        print("  %s " % s + " ".join("%10d" % tally[(s, lab)] for lab in labels)
              + "  %d" % len([1 for (ss, _p) in plot_lat if ss == s]))
    other = sorted({lab for (_s, lab) in tally} - set(labels))
    if other:
        print("  other tallies: " + "; ".join("%s = %d" % (lab, sum(tally[(s, lab)] for s in SITES)) for lab in other))

    print("\n  Tick pathogen table: test names and number of rows (all sites, all ticks)")
    for nm in sorted(names):
        print("    %-34s %7d" % (nm, names[nm]))
    for k in sorted(info):
        print("  %s: %d" % (k, info[k]))
    for name in (SL, SS):
        print("  ticks with a result for '%s', by taxonomy-table species and stage: %s"
              % (name, ", ".join("%s %s %d" % (k[0], k[1], v) for k, v in sorted(who[name].items())) or "none"))
    print("  I. scapularis nymphs left out because the DNA-quality control was not Positive, by site (%s): %s"
          % (SL, ", ".join("%s %d" % (s, qc_out[(SL, s)]) for s in SITES if qc_out[(SL, s)]) or "none"))
    sys.stdout.flush()

    # ---------------------------------------------------------------- 2. mouse index
    years = sorted({y for (_s, y) in effort})
    index = {}
    for (s, y), tn in effort.items():
        if tn >= MIN_TN:
            index[(s, y)] = 100.0 * mice[(s, y)] / tn
    print("\n2. Mouse index: distinct tagged Peromyscus per 100 trap-nights, by site and calendar year")
    print("   each cell: individuals / trap-nights = index;  '*' = fewer than %d trap-nights, no index" % MIN_TN)
    for s in SITES:
        print("  %s" % s)
        for y in years:
            if (s, y) not in effort:
                continue
            tn, m = effort[(s, y)], mice[(s, y)]
            if tn >= MIN_TN:
                print("     %d  %5d / %6d = %7.3f" % (y, m, tn, index[(s, y)]))
            else:
                print("     %d  %5d / %6d   *" % (y, m, tn))
    sigma, r1, nyears, nzero, npairs, used_years = {}, {}, {}, {}, {}, {}
    for s in SITES:
        ys = sorted(y for (ss, y) in index if ss == s)
        pos = {y: math.log(index[(s, y)]) for y in ys if index[(s, y)] > 0}
        nyears[s] = len(ys)
        nzero[s] = len(ys) - len(pos)
        used_years[s] = sorted(pos)
        sigma[s] = float(np.std(list(pos.values()), ddof=1)) if len(pos) >= MIN_YEARS else None
        r1[s], npairs[s] = r1_of(pos)
    print("\n3. Host variability (no hypothesis): sigma_site = sd of ln(index) across years; r1 = correlation with the next year")
    print("  site  years with an index  zero years  years used   span        sigma_site     r1  (pairs)  mean index  latitude")
    for s in SITES:
        ys = used_years[s]
        mean_idx = float(np.mean([index[(s, y)] for (ss, y) in index if ss == s])) if nyears[s] else float("nan")
        print("  %s  %10d  %14d  %10d   %s   %10s  %6s  (%d)   %9.3f  %8s"
              % (s, nyears[s], nzero[s], len(ys), ("%d-%d" % (ys[0], ys[-1])) if ys else "    -    ",
                 fmt(sigma[s]), fmt(r1[s]), npairs[s], mean_idx, fmt(lat[s], "%.3f")))
    have = [s for s in SITES if sigma[s] is not None]
    if have:
        sv = [sigma[s] for s in have]
        print("  sites with a sigma_site: %d of 13; least %.3f, median %.3f, most %.3f" % (len(have), min(sv), float(np.median(sv)), max(sv)))
        rv = [r1[s] for s in have if r1[s] is not None]
        if rv:
            print("  r1 over %d sites: least %.3f, median %.3f, most %.3f; negative at %d" % (len(rv), min(rv), float(np.median(rv)), max(rv), sum(1 for v in rv if v < 0)))
    sys.stdout.flush()

    # ---------------------------------------------------------------- 4. infection
    print("\n4. Infection: I. scapularis nymphs tested for B. burgdorferi, positives / tested by collection year")
    nip, tested, positive, test_years, yearly = {}, {}, {}, {}, {}
    for name, label in ((SL, "sensu lato ('%s'), used for the tests" % SL), (SS, "sensu stricto ('%s'), reported only" % SS)):
        print("  %s" % label)
        cells = out[name]
        ys_all = sorted({y for (_s, y) in cells})
        if not cells:
            print("     no I. scapularis nymph has a result under this name")
            continue
        print("     site " + " ".join("%9d" % y for y in ys_all) + "      total    share")
        for s in SITES:
            tot = [sum(cells[(s, y)][0] for y in ys_all if (s, y) in cells), sum(cells[(s, y)][1] for y in ys_all if (s, y) in cells)]
            line = " ".join(("%4d/%-4d" % (cells[(s, y)][1], cells[(s, y)][0])) if (s, y) in cells else "    .    " for y in ys_all)
            print("     %s %s  %5d/%-5d  %s" % (s, line, tot[1], tot[0], ("%.4f" % (tot[1] / tot[0])) if tot[0] else "  -"))
            if name == SL:
                tested[s], positive[s] = tot[0], tot[1]
                nip[s] = tot[1] / tot[0] if tot[0] else None
                test_years[s] = sorted(y for y in ys_all if (s, y) in cells and cells[(s, y)][0] > 0)
                yearly[s] = {y: cells[(s, y)][1] / cells[(s, y)][0] for y in ys_all if (s, y) in cells and cells[(s, y)][0] >= MIN_YEARLY}
    for s in SITES:
        tested.setdefault(s, 0)
        positive.setdefault(s, 0)
        nip.setdefault(s, None)
        test_years.setdefault(s, [])
        yearly.setdefault(s, {})

    # ---------------------------------------------------------------- 5. per-site table
    included, why = [], {}
    for s in SITES:
        reasons = []
        if tested[s] < MIN_TOTAL:
            reasons.append("%d nymphs tested (needs %d)" % (tested[s], MIN_TOTAL))
        if len(test_years[s]) < MIN_TEST_YEARS:
            reasons.append("%d years with a tested nymph (needs %d)" % (len(test_years[s]), MIN_TEST_YEARS))
        if sigma[s] is None:
            reasons.append("no sigma_site (%d years with a positive index, needs %d)" % (len(used_years[s]), MIN_YEARS))
        why[s] = "; ".join(reasons)
        if not reasons:
            included.append(s)
    print("\n5. One row per site")
    print("  site  mouse years  sigma_site      r1   nymphs tested  positive  NIP_site  years tested  years with yearly NIP  latitude  included")
    for s in SITES:
        print("  %s  %7d     %9s  %6s   %9d   %7d    %7s   %7d   %12d         %8s   %s"
              % (s, len(used_years[s]), fmt(sigma[s]), fmt(r1[s]), tested[s], positive[s], fmt(nip[s], "%.4f"),
                 len(test_years[s]), len(yearly[s]), fmt(lat[s], "%.3f"), "yes" if s in included else "no: " + why[s]))
    print("  included sites: %d (%s)" % (len(included), ", ".join(included) or "none"))
    sys.stdout.flush()

    # ---------------------------------------------------------------- 6. T1
    print("\n6. T1 (main): across the included sites, NIP_site falls as sigma_site rises (Spearman's rho negative)")
    if len(included) < 5:
        print("  NOT RUN: %d included sites, fewer than five.  By the protocol that is the result." % len(included))
    else:
        xs = [sigma[s] for s in included]
        ys_ = [nip[s] for s in included]
        rho, p_neg, p_pos, how = spearman_perm(xs, ys_, np.random.default_rng([SEED, 1]))
        print("  sites: " + ", ".join("%s (sigma %.3f, NIP %.4f)" % (s, sigma[s], nip[s]) for s in included))
        print("  ranks of sigma_site: %s" % " ".join("%g" % v for v in midranks(xs)))
        print("  ranks of NIP_site  : %s" % " ".join("%g" % v for v in midranks(ys_)))
        print("  n = %d; Spearman's rho = %.4f; one-sided p (rho this low or lower) = %.4f; p for the opposite side (this high or higher) = %.4f; %s"
              % (len(included), rho, p_neg, p_pos, how))
        if math.isnan(rho):
            verdict = "NOT DECIDED (rho is not defined: one variable does not vary)"
        elif rho < 0 and p_neg < 0.05:
            verdict = "SUPPORTED"
        elif rho > 0 and p_pos < 0.05:
            verdict = "CONTRADICTED"
        else:
            verdict = "NOT DECIDED"
        print("  Outcome by the protocol's rule: %s" % verdict)

    # ---------------------------------------------------------------- 7. T2
    print("\n7. T2: at each included site with at least four yearly values, sd of ln(yearly NIP) < sd of ln(mouse index), same years")
    entered = viol = 0
    for s in included:
        yv = yearly[s]
        zero_nip = [y for y in yv if yv[y] == 0]
        no_idx = [y for y in yv if yv[y] > 0 and not ((s, y) in index and index[(s, y)] > 0)]
        use = sorted(y for y in yv if yv[y] > 0 and (s, y) in index and index[(s, y)] > 0)
        note = "yearly NIP values %d; zero NIP left out %d; no positive mouse index that year %d; years used %d" % (len(yv), len(zero_nip), len(no_idx), len(use))
        if len(use) < 4:
            print("  %s  not entered (%s)" % (s, note))
            continue
        a = sd_ln([yv[y] for y in use])
        b = sd_ln([index[(s, y)] for y in use])
        entered += 1
        bad = a >= b
        viol += bad
        print("  %s  years %s: sd ln NIP %.3f, sd ln index %.3f -> %s   (%s)"
              % (s, ",".join(str(y) for y in use), a, b, "VIOLATES" if bad else "holds", note))
    if entered == 0:
        print("  NOT TESTABLE: no included site has four years with both values.")
    else:
        print("  sites entered %d, violations %d (a quarter of the sites = %.2f).  Outcome by the protocol's rule: %s"
              % (entered, viol, entered / 4, "FAILS" if viol > entered / 4 else "HOLDS"))

    # ---------------------------------------------------------------- 8. T3
    print("\n8. T3: pooled within-site slope b of ln(NIP in year t) on ln(mouse index in year t - 1); 0 <= b <= 1 - pbar within two standard errors")
    groups, labels3 = [], []
    for s in included:
        pr = [(math.log(index[(s, y - 1)]), math.log(v), y) for y, v in sorted(yearly[s].items())
              if v > 0 and (s, y - 1) in index and index[(s, y - 1)] > 0]
        lost = len(yearly[s]) - len(pr)
        if len(pr) >= 2:
            groups.append((np.array([p[0] for p in pr]), np.array([p[1] for p in pr])))
            labels3.append("%s %d (NIP years %s)" % (s, len(pr), ",".join(str(p[2]) for p in pr)))
        else:
            labels3.append("%s %d, not entered" % (s, len(pr)))
        if lost:
            labels3[-1] += " [%d yearly NIP without a usable pair]" % lost
    print("  pairs by site: " + "; ".join(labels3) if labels3 else "  no included site")
    npair = sum(len(g[0]) for g in groups)
    pool = [v for s in included for v in yearly[s].values()]
    if not groups or npair - len(groups) < 3 or within_slope(groups) is None:
        print("  NOT RUN: %d pairs at %d sites; needs (pairs - sites) >= 3." % (npair, len(groups)))
    else:
        b, sxx, rss, n, k = within_slope(groups)
        se_ls = math.sqrt(rss / (n - k - 1) / sxx) if n - k - 1 > 0 else float("nan")
        rng = np.random.default_rng([SEED, 3])
        boots, skipped = [], 0
        for _ in range(NBOOT):
            res = within_slope([(x[i], y[i]) for x, y in groups for i in [rng.integers(0, len(x), len(x))]])
            if res is None:
                skipped += 1
            else:
                boots.append(res[0])
        se = float(np.std(boots, ddof=1))
        pbar = float(np.mean(pool))
        lo, hi = b - 2 * se, b + 2 * se
        print("  %d pairs at %d sites; b = %.4f; standard error %.4f from %d resamples within site (%d resamples had no spread in x and were skipped; mean of resampled b %.4f; 2.5th to 97.5th percentile %.4f to %.4f)"
              % (n, k, b, se, len(boots), skipped, float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
        print("  check of the code only: textbook least-squares standard error %.4f (%d residual degrees of freedom)" % (se_ls, n - k - 1))
        print("  pbar = %.4f (mean of %d yearly NIP values at the included sites), so 1 - pbar = %.4f" % (pbar, len(pool), 1 - pbar))
        print("  b - 2 se = %.4f, b + 2 se = %.4f; allowed band 0 to %.4f" % (lo, hi, 1 - pbar))
        print("  Outcome by the protocol's rule: %s" % ("HOLDS" if hi >= 0 and lo <= 1 - pbar else "FAILS"))

    # ---------------------------------------------------------------- 9. latitude
    print("\n9. Latitude (no hypothesis, no adjustment): Spearman rank correlations")
    if len(have) >= 3:
        print("  latitude with sigma_site, all %d sites with a sigma_site: rho = %s" % (len(have), fmt(spearman([lat[s] for s in have], [sigma[s] for s in have]), "%.4f")))
    if len(included) >= 3:
        print("  latitude with sigma_site, the %d included sites: rho = %s" % (len(included), fmt(spearman([lat[s] for s in included], [sigma[s] for s in included]), "%.4f")))
        print("  latitude with NIP_site,   the %d included sites: rho = %s" % (len(included), fmt(spearman([lat[s] for s in included], [nip[s] for s in included]), "%.4f")))
    else:
        print("  fewer than three included sites: no correlation with NIP_site")
    print("\nFinished in %.0f s." % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
