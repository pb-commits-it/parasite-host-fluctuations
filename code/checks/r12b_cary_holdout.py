"""Run R12b (protocols/R12b-cary-holdout.md): the holdout years 2014 to 2022 on the six Cary Institute
grids (Millbrook, New York).  Three predictions written before the infection data were opened:

  B1  On each grid the standard deviation across years of ln(prevalence of Borrelia burgdorferi in
      questing nymphs) is smaller than that of ln(mice during larval questing, previous year).
      Fails if the inequality fails on more than one grid of six (grids with fewer than five usable
      years are left out and counted).
  B2  The within-grid slope b2 of ln(prevalence in year t) on ln(mice in year t - 1) agrees with the
      first-period slope b1 = 0.038 (standard error 0.058): |b2 - b1| <= 2 sqrt(0.058^2 + SE2^2).
      Reported separately: whether b2 is inside [0, 1 - pbar2] within two standard errors.
  B3  B1 recomputed with the binomial sampling variance of each yearly prevalence subtracted
      ((1 - p)/(n p) for n nymphs with a share p infected).  Reported: whether B1 changes from holding
      to failing or the reverse.

This is a pattern check on one landscape, not a test of the theory's main prediction.  Nothing is fitted
to the theory (study rules, rule 10).  Nothing else is computed.

What I had seen when the readings below were fixed (14 November 2026, 10:06 CST), and what I had not:
  seen      the two column keys, the two method notes (their abstract gives the overall share infected
            with B. burgdorferi, 19.3%), the Figshare metadata file; the header of both data files; the
            grid labels and years; for all 54 rows the columns n, nPLhosts, nPLhostsN and the eight PL.*
            mouse columns (mouse numbers are not blind: protocol, "What is not blind"); the number of
            rows per grid and year and per month in the file of individual nymphs; PL_t of the long
            Figshare file used in run R12, beside the mouse columns.
  not seen  any infection value by grid or year (no column of proportions and no 0/1 result had been
            printed or entered any computation).

Readings of the protocol, fixed then, before the first run on the data (the protocol leaves these open):
  1. "Mice during larval questing, previous year" = the column PL.mnRawLyr (key: "mean MNA (minimum
     number alive) mice", in the group headed "total mice trapped during larval questing"), taken from
     the row of the year t in which the nymphs were collected, AS IT STANDS, with no shift by me.
     Reason: the column is already the value of year t - 1.  In every row PL.mnRawLyr equals
     PL.mnRawNyr of the same grid's row for the year before (checked below, C3), so the two columns
     are one yearly series written twice, one year apart; and ln PL.mnRawLyr of row t follows ln PL_t
     of the long Figshare file in year t - 1, not in year t (checked below, C4).  "Lyr" is read as
     "the year the nymphs were larvae", "Nyr" as "the year the nymphs were collected".
     Consequence: the first year of the file, 2014, keeps its previous-year value (mice of 2013 are in
     its row); no year is lost in forming the lag.
     Not used: PL.mnLogLyr (it is the mean of ln(count + 1), not ln of the mean), PL.SumLyr (a sum
     over a varying number of counts), nPLhosts (the number of mice examined for larvae).
  2. Prevalence = the column Bburgd and nymphs tested = the column n of allthedata_grid.csv (the file
     and columns the protocol names).  NymphalInfectionData.csv is used to verify both (C1, C2); if
     they disagree the grid file's values still decide the outcomes and the disagreement is printed.
  3. A grid-year is usable if n >= 20 (the protocol: fewer than 20 is left out), the prevalence is
     above zero and the mouse value is present and above zero.  A zero is left out of the logarithm
     and counted, as in R12.  A grid-year that is not usable is left out of B1, B2 and B3 entirely:
     in B1 both standard deviations are taken over the same usable years of that grid.  The standard
     deviation of ln mice over all nine years is printed beside it (R12 printed the all-year value
     in the same way); it decides nothing.
  4. sd = sample standard deviation (n - 1) of the ln values across years, as in R12.
  5. B1: a grid with fewer than five usable years is left out and counted.  B1 FAILS if the inequality
     fails on more than one of the grids that remain.
  6. B2: pairs (ln PL.mnRawLyr, ln Bburgd) from the same row, usable grid-years only (of every grid,
     also one that B1 leaves out: the five-year rule is written for B1); least squares
     with one intercept per grid.  b1 = 0.038 and its standard error 0.058 are taken as the protocol
     writes them.  SE2 = sd (n - 1) of the slope over 2,000 resamples of years: a block is two
     consecutive calendar years of nymph collection; ceil(9/2) = 5 blocks are drawn with replacement
     from the 8 possible starts; every grid gets the same years; seed 20261114 (R12's conventions).
     pbar2 = unweighted mean of Bburgd over the grid-years with n >= 20 (a zero counts here).  The
     mean over all 54 grid-years and the pooled share are printed beside it.
     "Inside [0, 1 - pbar2] within two standard errors" = -2 SE2 <= b2 <= 1 - pbar2 + 2 SE2 (R12).
  7. B3: for each grid that B1 uses, corrected variance = variance of ln prevalence over the usable years (n - 1)
     minus the mean over those years of (1 - p)/(n p); if the result is below zero the corrected sd is
     set to zero (all of the variation is within sampling error) and that is printed.  The mouse sd is
     left as it is (the protocol corrects prevalence only).  B1's rule is applied again.  B3 HOLDS if
     B1's outcome (holds or fails) is the same before and after; the number of grids on which the
     inequality changes is printed as well.

Code checks (study rules rule 7; tolerances stated here, before the run):
  K1  sd against a hand-worked value, 1e-12.
  K2  the within-grid slope against numpy least squares with grid dummies, 1e-9 (synthetic; and the data).
  K3  the block bootstrap of the slope on synthetic grids with y = a_grid + 0.3 x exactly and gaps:
      every resample that has a slope gives 0.3, 1e-9; more than half of the resamples have one.
  K4  the sampling-variance correction on synthetic numbers: 20,000 "years" with ln p ~ Normal(ln 0.3,
      0.3), n = 200 nymphs each, binomial counts; the corrected sd must come back within 5% of 0.3,
      and the uncorrected sd must be above it.
Checks of what the columns mean (on the data; C3 and C4 stop the run if they fail):
  C1  n equals the number of rows of that grid and year in NymphalInfectionData.csv (exactly).
  C2  Bburgd equals the share of rows with B.burgdorferi = 1 (within 5e-4).
  C3  PL.mnRawLyr of year t equals PL.mnRawNyr of year t - 1 on the same grid (within 1e-9).
  C4  the correlation of ln PL.mnRawLyr (row t) with ln PL_t of the long Figshare file is above 0.9 in
      year t - 1 and above its value in year t.  (Mouse numbers only.  Grid names are matched by
      their letters: greenC = GC, greenE = GE, henryC = HC, henryE = HE, teaC = TC, teaE = TE.)

Standard library and numpy only.  Raw data are read from ../../data/raw/ and are not copied; values by
grid and year are not printed, only summaries and the list of grid-years left out.

Run from problems/dobson-r0:
  python3 -I checks/r12b_cary_holdout.py > results/r12b-cary-holdout_20261114.txt
(A folder given as the first argument replaces the holdout folder; used once for a dry run on files of
random numbers, to catch programming errors.  C4 is skipped in that case.)
"""
from __future__ import annotations

import csv
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

PROBLEM = (__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()
RAW = (__import__('pathlib').Path(__file__).resolve().parent / '../data/raw/cary_lyme').resolve()
HOLDOUT = RAW / "figshare_27871269_nymphal_infection_2025"
LONG = RAW / "figshare_31872853_ecology_of_lyme_2026" / "Full_dataset_20260326.csv"

NBOOT = 2000
SEED = 20261114
MIN_NYMPHS = 20
MIN_YEARS = 5
B1_SLOPE = 0.038      # first-period slope, as the protocol writes it
B1_SE = 0.058         # its standard error, as the protocol writes it
MOUSE_COL = "PL.mnRawLyr"
LONG_NAME = {"greenC": "GC", "greenE": "GE", "henryC": "HC", "henryE": "HE", "teaC": "TC", "teaE": "TE"}
NAN = float("nan")


# ---------------------------------------------------------------- reading

def read_rows(path):
    with open(path, newline="", encoding="latin-1") as f:
        return list(csv.DictReader(f))


def num(v):
    v = v.strip()
    return None if v in ("NA", "") else float(v)


def span_text(years):
    """'2014, 2016-2018' for a sorted list of years."""
    if not years:
        return "none"
    out, a, b = [], years[0], years[0]
    for y in years[1:]:
        if y == b + 1:
            b = y
        else:
            out.append(f"{a}-{b}" if b > a else f"{a}")
            a = b = y
    out.append(f"{a}-{b}" if b > a else f"{a}")
    return ", ".join(out)


# ---------------------------------------------------------------- statistics (R12's conventions)

def sd(values):
    v = list(values)
    return float(np.std(v, ddof=1)) if len(v) >= 2 else NAN


def variance(values):
    v = list(values)
    return float(np.var(v, ddof=1)) if len(v) >= 2 else NAN


def to_array(d, y0, y1):
    a = np.full(y1 - y0 + 1, np.nan)
    for y, v in d.items():
        a[y - y0] = v
    return a


def draw_blocks(rng, n_years):
    """Start positions of ceil(n_years/2) blocks of two consecutive years, NBOOT times."""
    return rng.integers(0, n_years - 1, size=(NBOOT, math.ceil(n_years / 2)))


def fe_slope(xs, ys):
    """Least-squares slope with one intercept per grid.  xs, ys: lists (one per grid) of sequences."""
    sxy = sxx = 0.0
    for x, y in zip(xs, ys):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) == 0:
            continue
        sxy += float(((x - x.mean()) * (y - y.mean())).sum())
        sxx += float(((x - x.mean()) ** 2).sum())
    return sxy / sxx


def fe_slope_dummies(xs, ys):
    k = len(xs)
    rows, resp = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        for a, b in zip(x, y):
            rows.append([a] + [1.0 if j == i else 0.0 for j in range(k)])
            resp.append(b)
    coef = np.linalg.lstsq(np.array(rows), np.array(resp), rcond=None)[0]
    return float(coef[0])


def fe_slope_boot(xarrs, yarrs, starts):
    """The within-grid slope in each resample of year blocks (the same years for every grid)."""
    sxy = np.zeros(len(starts))
    sxx = np.zeros(len(starts))
    for xa, ya in zip(xarrs, yarrs):
        X = np.concatenate([xa[starts], xa[starts + 1]], axis=1)
        Y = np.concatenate([ya[starts], ya[starts + 1]], axis=1)
        m = np.isfinite(X) & np.isfinite(Y)
        n = np.maximum(m.sum(axis=1), 1)
        mx = np.where(m, X, 0.0).sum(axis=1) / n
        my = np.where(m, Y, 0.0).sum(axis=1) / n
        dx = np.where(m, X - mx[:, None], 0.0)
        dy = np.where(m, Y - my[:, None], 0.0)
        sxy += (dx * dy).sum(axis=1)
        sxx += (dx * dx).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sxy / sxx
    out[sxx < 1e-12] = np.nan
    return out


def corrected_sd(ps, ns):
    """sd of ln p across years, the mean binomial sampling variance (1-p)/(np), and the sd after subtraction."""
    v = variance([math.log(p) for p in ps])
    samp = float(np.mean([(1 - p) / (n * p) for p, n in zip(ps, ns)]))
    c = v - samp
    return math.sqrt(v), samp, c, (math.sqrt(c) if c > 0 else 0.0)


# ---------------------------------------------------------------- code checks (no data)

def code_checks():
    fails = []
    # K1
    s = sd([1.0, 2.0, 3.0, 4.0])
    ok = abs(s - math.sqrt(5 / 3)) < 1e-12
    print(f"  K1  sd of 1, 2, 3, 4: {s:.12f} (hand {math.sqrt(5 / 3):.12f})   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K1")
    # K2 (synthetic part)
    rng = np.random.default_rng(11)
    xs = [rng.normal(size=k) for k in (8, 5, 7)]
    ys = [0.4 * x + c + rng.normal(scale=0.3, size=len(x)) for x, c in zip(xs, (0.0, 2.0, -1.0))]
    b_a, b_b = fe_slope(xs, ys), fe_slope_dummies(xs, ys)
    ok = abs(b_a - b_b) < 1e-9
    print(f"  K2  within-grid slope, synthetic: {b_a:.12f} against least squares with dummies {b_b:.12f}   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K2")
    # K3
    rng = np.random.default_rng(7)
    xarrs, yarrs = [], []
    for g, a in enumerate((0.5, -1.0, 2.0)):
        x = rng.normal(size=9)
        y = a + 0.3 * x
        x[(g * 2) % 9] = np.nan          # a gap in a different year on each grid
        y[(g * 3 + 1) % 9] = np.nan
        xarrs.append(x)
        yarrs.append(y)
    bb = fe_slope_boot(xarrs, yarrs, draw_blocks(rng, 9))
    fin = np.isfinite(bb)
    ok = bool(np.all(np.abs(bb[fin] - 0.3) < 1e-9) and fin.sum() > NBOOT // 2)
    print(f"  K3  block bootstrap of the slope, exact line with gaps: slopes {bb[fin].min():.12f} to {bb[fin].max():.12f} "
          f"in {int(fin.sum())} of {NBOOT} resamples (exact 0.3)   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K3")
    # K4
    rng = np.random.default_rng(13)
    n_k4 = 200
    p_true = np.exp(rng.normal(math.log(0.3), 0.3, size=20000))
    p_true = np.minimum(p_true, 0.99)
    k = rng.binomial(n_k4, p_true)
    keep = k > 0
    p_hat = (k[keep] / n_k4).tolist()
    raw, samp, _, corr = corrected_sd(p_hat, [n_k4] * len(p_hat))
    sd_true = float(np.std(np.log(p_true[keep]), ddof=1))
    ok = abs(corr - sd_true) <= 0.05 * sd_true and raw > sd_true
    print(f"  K4  sampling-variance correction, synthetic ({len(p_hat)} years, n = {n_k4}): sd of ln p observed {raw:.4f}, "
          f"corrected {corr:.4f}, true {sd_true:.4f} (mean sampling variance {samp:.5f})   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K4")
    return fails


# ---------------------------------------------------------------- main

def main():
    t_start = time.time()
    now = datetime.datetime.now().astimezone()
    dry = len(sys.argv) > 1
    folder = Path(sys.argv[1]) if dry else HOLDOUT
    print(f"R12b: the holdout years at the Cary Institute grids (2014 to 2022).  Started {now:%Y-%m-%d %H:%M:%S %Z}.")
    print("A pattern check on one landscape; not a test of the predicted loss of R0.  Protocol: protocols/R12b-cary-holdout.md")
    if dry:
        print(f"DRY RUN ON FILES THAT ARE NOT THE DATA: {folder}")
    else:
        print(f"Files read: {(folder / 'allthedata_grid.csv').relative_to(RAW.parent.parent.parent)}")
        print(f"            {(folder / 'NymphalInfectionData.csv').relative_to(RAW.parent.parent.parent)}")
        print(f"            {LONG.relative_to(RAW.parent.parent.parent)}  (column PL_t only, for check C4)")

    print("\n0. Code checks that do not touch the data")
    fails = code_checks()
    if fails:
        print(f"FAILED code checks {fails}; stopping before the data are read.")
        return 1

    # ------------------------------------------------------------ 1. reading and what the columns mean
    grid_rows = read_rows(folder / "allthedata_grid.csv")
    tick_rows = read_rows(folder / "NymphalInfectionData.csv")
    grids = sorted({r["Grid"] for r in grid_rows})
    years = sorted({int(r["Year"]) for r in grid_rows})
    y0, y1 = years[0], years[-1]
    G = {(r["Grid"], int(r["Year"])): r for r in grid_rows}
    n_tested = {k: num(r["n"]) for k, r in G.items()}
    prev = {k: num(r["Bburgd"]) for k, r in G.items()}
    mice = {k: num(r[MOUSE_COL]) for k, r in G.items()}
    print("\n1. What was read, and checks of what the columns mean")
    print(f"  allthedata_grid.csv: {len(grid_rows)} rows, {len(grid_rows[0])} columns; grids {grids}; years {span_text(years)}; "
          f"{len(G)} distinct grid-years")
    ns = [v for v in n_tested.values() if v is not None]
    print(f"    n (nymphs tested): {len(ns)} values, total {sum(ns):.0f}, least {min(ns):.0f}, median {float(np.median(ns)):.0f}, most {max(ns):.0f}; "
          f"missing {sum(1 for v in n_tested.values() if v is None)}")
    print(f"    Bburgd (share infected with B. burgdorferi): missing {sum(1 for v in prev.values() if v is None)}, "
          f"zero {sum(1 for v in prev.values() if v == 0)}")
    print(f"    {MOUSE_COL} (mean minimum number of mice alive, year before collection): missing {sum(1 for v in mice.values() if v is None)}, "
          f"zero {sum(1 for v in mice.values() if v == 0)}")
    months = {}
    cnt, pos, valid = {}, {}, {}
    for r in tick_rows:
        k = (r["Grid"], int(r["Year"]))
        cnt[k] = cnt.get(k, 0) + 1
        v = r["B.burgdorferi"].strip()
        if v in ("0", "1"):
            valid[k] = valid.get(k, 0) + 1
            pos[k] = pos.get(k, 0) + int(v)
        months[r["month"].strip()] = months.get(r["month"].strip(), 0) + 1
    n_na = len(tick_rows) - sum(valid.values())
    print(f"  NymphalInfectionData.csv: {len(tick_rows)} rows (one per nymph), {len(cnt)} grid-years; B.burgdorferi is 0 or 1 in "
          f"{sum(valid.values())} rows, something else in {n_na}; rows by month of collection: "
          + ", ".join(f"{m}: {c}" for m, c in sorted(months.items(), key=lambda kv: (len(kv[0]), kv[0]))))

    # C1, C2
    bad_n = [k for k in G if (n_tested[k] is None) or abs(n_tested[k] - cnt.get(k, 0)) > 0]
    extra = [k for k in cnt if k not in G]
    diffs = {k: abs(prev[k] - pos.get(k, 0) / valid[k]) for k in G if prev[k] is not None and valid.get(k)}
    bad_p = [k for k, d in diffs.items() if d > 5e-4]
    print(f"  C1  n = number of rows in the file of individual nymphs: {len(G) - len(bad_n)} of {len(G)} grid-years agree"
          f"{'' if not bad_n else '; MISMATCH at ' + str(sorted(bad_n))}{'' if not extra else '; grid-years only in the individual file: ' + str(sorted(extra))}"
          f"   [{'ok' if not bad_n and not extra else 'MISMATCH'}]")
    print(f"  C2  Bburgd = share of those rows with B.burgdorferi = 1: largest difference {max(diffs.values()):.2e} over {len(diffs)} grid-years"
          f"{'' if not bad_p else '; MISMATCH at ' + str(sorted(bad_p))}   [{'ok' if not bad_p else 'MISMATCH'}]")
    if bad_n or extra or bad_p:
        print("      The grid file's values decide the outcomes below (reading 2).")
    # C3
    worst, n_cmp = 0.0, 0
    for (g, y), r in G.items():
        if (g, y - 1) in G:
            a, b = num(r["PL.mnRawLyr"]), num(G[(g, y - 1)]["PL.mnRawNyr"])
            if a is None or b is None:
                continue
            worst = max(worst, abs(a - b))
            n_cmp += 1
    ok3 = worst < 1e-9 and n_cmp > 0
    print(f"  C3  PL.mnRawLyr of year t = PL.mnRawNyr of year t - 1, same grid: largest difference {worst:.2e} over {n_cmp} pairs   [{'ok' if ok3 else 'FAIL'}]")
    # C4
    ok4 = True
    if dry:
        print("  C4  skipped in the dry run")
    else:
        pl = {}
        for r in read_rows(LONG):
            if r["Plot"] in ("NA", "") or r["PL_t"] in ("NA", ""):
                continue
            pl[(r["Plot"], int(r["Year"]))] = float(r["PL_t"])
        res = {}
        for lag in (-1, 0):
            a, b = [], []
            for (g, y), v in mice.items():
                w = pl.get((LONG_NAME[g], y + lag))
                if v and w:
                    a.append(math.log(v))
                    b.append(math.log(w))
            res[lag] = (float(np.corrcoef(a, b)[0, 1]), len(a), float(np.median(np.exp(np.array(a) - np.array(b)))))
        ok4 = res[-1][0] > 0.9 and res[-1][0] > res[0][0]
        print(f"  C4  correlation of ln {MOUSE_COL} (row of year t) with ln PL_t of the long Figshare file: in year t - 1: {res[-1][0]:.3f} "
              f"({res[-1][1]} grid-years; median ratio {res[-1][2]:.2f}); in year t: {res[0][0]:.3f} ({res[0][1]})   [{'ok' if ok4 else 'FAIL'}]")
    if not (ok3 and ok4):
        print("STOPPED: the mouse column does not mean what reading 1 takes it to mean.")
        return 1
    sys.stdout.flush()

    # ------------------------------------------------------------ 2. counts
    print(f"\n2. Counts.  A grid-year is usable if n >= {MIN_NYMPHS}, the prevalence is above zero and the mouse value is present and above zero.")
    usable = {}
    n_small = n_zero_p = n_bad_mouse = 0
    print(f"  {'grid':7s} {'usable years':28s} {'n':>2s}   left out: fewer than {MIN_NYMPHS} nymphs (n tested)        zero prevalence   mouse value missing or zero")
    for g in grids:
        small, zero_p, bad_m, use = [], [], [], []
        for y in years:
            k = (g, y)
            if k not in G or n_tested[k] is None or n_tested[k] < MIN_NYMPHS:
                small.append(y)
            elif prev[k] is None or prev[k] <= 0:
                zero_p.append(y)
            elif mice[k] is None or mice[k] <= 0:
                bad_m.append(y)
            else:
                use.append(y)
        usable[g] = use
        n_small += len(small)
        n_zero_p += len(zero_p)
        n_bad_mouse += len(bad_m)
        small_txt = ", ".join(f"{y} ({n_tested[(g, y)]:.0f})" if (g, y) in G and n_tested[(g, y)] is not None else f"{y} (no row)" for y in small) or "-"
        print(f"  {g:7s} {span_text(use):28s} {len(use):2d}   {small_txt:44s} {(span_text(zero_p) if zero_p else '-'):17s} {(span_text(bad_m) if bad_m else '-')}")
    n_use = sum(len(v) for v in usable.values())
    n_all = len(grids) * len(years)
    print(f"  grid-years: {n_all} in all; left out for fewer than {MIN_NYMPHS} nymphs {n_small}; for a zero prevalence {n_zero_p}; "
          f"for the mouse value {n_bad_mouse}; usable {n_use}")
    kept = [g for g in grids if len(usable[g]) >= MIN_YEARS]
    dropped = [g for g in grids if len(usable[g]) < MIN_YEARS]
    print(f"  grids with at least {MIN_YEARS} usable years: {len(kept)} of {len(grids)}"
          f"{'' if not dropped else '; left out: ' + ', '.join(f'{g} ({len(usable[g])} years)' for g in dropped)}")
    used_n = [n_tested[(g, y)] for g in grids for y in usable[g]]
    print(f"  nymphs tested in the usable grid-years: total {sum(used_n):.0f}, least {min(used_n):.0f}, median {float(np.median(used_n)):.0f}, most {max(used_n):.0f}")

    # ------------------------------------------------------------ 3. B1
    print("\n3. B1  sd of ln prevalence < sd of ln mice (year before), grid by grid, over the same usable years.")
    print("        Fails if the inequality fails on more than one grid.")
    print(f"  {'grid':7s} {'years':>5s} {'sd ln prevalence':>17s} {'sd ln mice':>11s}   {'prevalence varies less':22s}  [sd ln mice, all {len(years)} years]")
    b1 = {}
    for g in kept:
        lp = [math.log(prev[(g, y)]) for y in usable[g]]
        lm = [math.log(mice[(g, y)]) for y in usable[g]]
        lm_all = [math.log(mice[(g, y)]) for y in years if mice.get((g, y))]
        s_p, s_m = sd(lp), sd(lm)
        b1[g] = (s_p, s_m, s_p < s_m)
        print(f"  {g:7s} {len(lp):5d} {s_p:17.3f} {s_m:11.3f}   {('yes' if s_p < s_m else 'NO'):22s}  [{sd(lm_all):.3f}]")
    n_fail_b1 = sum(1 for g in kept if not b1[g][2])
    b1_out = "HOLDS" if n_fail_b1 <= 1 else "FAILS"
    print(f"  B1 {b1_out}: the inequality fails on {n_fail_b1} of {len(kept)} grids (criterion: fails if more than one); grids left out {len(dropped)}.")
    sys.stdout.flush()

    # ------------------------------------------------------------ 4. B2
    print("\n4. B2  slope b2 of ln prevalence in year t on ln mice in year t - 1, one intercept per grid, usable grid-years.")
    xs = [[math.log(mice[(g, y)]) for y in usable[g]] for g in grids]
    ys = [[math.log(prev[(g, y)]) for y in usable[g]] for g in grids]
    print("    pairs by grid: " + ", ".join(f"{g} {len(x)}" for g, x in zip(grids, xs)) + f"; total {n_use}")
    b2 = fe_slope(xs, ys)
    b2_dum = fe_slope_dummies(xs, ys)
    ok_k2 = abs(b2 - b2_dum) < 1e-9
    if not ok_k2:
        fails.append("K2 on the data")
    rng = np.random.default_rng(SEED)
    starts = draw_blocks(rng, y1 - y0 + 1)
    xarrs = [to_array({y: v for y, v in zip(usable[g], x)}, y0, y1) for g, x in zip(grids, xs)]
    yarrs = [to_array({y: v for y, v in zip(usable[g], yv)}, y0, y1) for g, yv in zip(grids, ys)]
    bb = fe_slope_boot(xarrs, yarrs, starts)
    bb_ok = bb[np.isfinite(bb)]
    se2 = float(np.std(bb_ok, ddof=1))
    print(f"    b2 = {b2:.4f}   (least squares with grid dummies {b2_dum:.4f}; K2 on the data [{'ok' if ok_k2 else 'FAIL'}])")
    print(f"    bootstrap standard error SE2 = {se2:.4f}   ({len(bb_ok)} of {NBOOT} resamples have a slope; their mean {float(bb_ok.mean()):.4f}; "
          f"2.5th to 97.5th percentile {np.percentile(bb_ok, 2.5):.4f} to {np.percentile(bb_ok, 97.5):.4f}; "
          f"{starts.shape[1]} blocks of two consecutive years per resample, seed {SEED})")
    diff = abs(b2 - B1_SLOPE)
    allow = 2 * math.sqrt(B1_SE ** 2 + se2 ** 2)
    b2_out = "HOLDS" if diff <= allow else "FAILS"
    print(f"    first period (protocol): b1 = {B1_SLOPE:.3f}, standard error {B1_SE:.3f}")
    print(f"    |b2 - b1| = {diff:.4f};  allowed 2 sqrt({B1_SE:.3f}^2 + SE2^2) = {allow:.4f};  the difference is {diff / math.sqrt(B1_SE ** 2 + se2 ** 2):.2f} combined standard errors")
    print(f"  B2 {b2_out}: |b2 - b1| is {'within' if diff <= allow else 'larger than'} the allowed difference.")
    p_kept = [prev[(g, y)] for g in grids for y in years if (g, y) in G and n_tested[(g, y)] is not None and n_tested[(g, y)] >= MIN_NYMPHS and prev[(g, y)] is not None]
    pbar2 = float(np.mean(p_kept))
    p_all = [v for v in prev.values() if v is not None]
    pooled = sum(prev[k] * n_tested[k] for k in G if prev[k] is not None and n_tested[k]) / sum(n_tested[k] for k in G if prev[k] is not None and n_tested[k])
    upper = 1 - pbar2
    inside = -2 * se2 <= b2 <= upper + 2 * se2
    print(f"    reported separately: pbar2 = {pbar2:.4f} (mean of the {len(p_kept)} yearly values with n >= {MIN_NYMPHS}; mean of all {len(p_all)} values {float(np.mean(p_all)):.4f}; "
          f"pooled share of all nymphs {pooled:.4f});  1 - pbar2 = {upper:.4f}")
    print(f"    the theory's bound with two standard errors either side: {-2 * se2:.4f} <= b2 <= {upper + 2 * se2:.4f}")
    print(f"    b2 is {'INSIDE' if inside else 'OUTSIDE'} that; it is {'inside' if 0 <= b2 <= upper else 'outside'} [0, {upper:.3f}] itself; "
          f"b2 / SE2 = {b2 / se2:.2f}; distance below 0: {max(0.0, -b2) / se2:.2f} SE2; distance above 1 - pbar2: {max(0.0, b2 - upper) / se2:.2f} SE2")
    sys.stdout.flush()

    # ------------------------------------------------------------ 5. B3
    print("\n5. B3  B1 again with the binomial sampling variance of ln prevalence, (1 - p)/(n p), subtracted (mean over the grid's usable years).")
    print(f"  {'grid':7s} {'years':>5s} {'var ln prev':>12s} {'sampling var':>13s} {'corrected var':>14s} {'corrected sd':>13s} {'sd ln mice':>11s}   {'prevalence varies less':22s} {'changed from B1':s}")
    b3 = {}
    n_neg = 0
    for g in kept:
        ps = [prev[(g, y)] for y in usable[g]]
        nn = [n_tested[(g, y)] for y in usable[g]]
        raw, samp, cvar, csd = corrected_sd(ps, nn)
        n_neg += 1 if cvar <= 0 else 0
        holds = csd < b1[g][1]
        b3[g] = holds
        print(f"  {g:7s} {len(ps):5d} {raw * raw:12.4f} {samp:13.4f} {cvar:14.4f} {csd:13.3f}{'*' if cvar <= 0 else ' '}{b1[g][1]:11.3f}   {('yes' if holds else 'NO'):22s} {('yes' if holds != b1[g][2] else 'no')}")
    if n_neg:
        print(f"  * corrected variance not above zero on {n_neg} grid(s): the sd is set to zero (reading 7).")
    n_fail_b3 = sum(1 for g in kept if not b3[g])
    b1c_out = "HOLDS" if n_fail_b3 <= 1 else "FAILS"
    n_changed = sum(1 for g in kept if b3[g] != b1[g][2])
    b3_out = "HOLDS" if b1c_out == b1_out else "FAILS"
    print(f"  B1 after the subtraction {b1c_out}: the inequality fails on {n_fail_b3} of {len(kept)} grids.  Grids on which it changed: {n_changed}.")
    print(f"  B3 {b3_out}: the outcome of B1 is {'the same' if b3_out == 'HOLDS' else 'DIFFERENT'} with and without the sampling variance "
          f"({b1_out} before, {b1c_out} after).")

    print(f"\nSummary: B1 {b1_out} ({n_fail_b1} of {len(kept)} grids against it; {len(dropped)} grid(s) left out); "
          f"B2 {b2_out} (b2 = {b2:.3f}, SE2 {se2:.3f}; |b2 - b1| = {diff:.3f}, allowed {allow:.3f}); "
          f"theory's bound [0, {upper:.3f}] within two SE2: {'inside' if inside else 'OUTSIDE'}; "
          f"B3 {b3_out} (B1 {b1c_out} after the subtraction, {n_changed} grid(s) changed).")
    print(f"Usable grid-years {n_use} of {n_all} ({n_small} left out for fewer than {MIN_NYMPHS} nymphs, {n_zero_p} for a zero prevalence, {n_bad_mouse} for the mouse value).")
    print(f"Finished {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}; wall time {time.time() - t_start:.1f} s.")
    print("CODE CHECKS: ALL PASSED" if not fails else f"CODE CHECKS FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
