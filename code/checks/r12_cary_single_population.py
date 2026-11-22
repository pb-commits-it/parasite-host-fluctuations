"""Run R12 (protocols/R12-cary-single-population.md): white-footed mice, ticks and the infection of
questing nymphs with Borrelia burgdorferi on the six Cary Institute plots (Millbrook, New York).

This is a pattern check on one landscape.  It is NOT a test of the theory's main prediction (the loss
of R0 when hosts fluctuate): R0 is not measured in these data.  Nothing is fitted to the theory and
no parameter of the theory is tuned to the data (study rules, rule 10).

What is computed, and nothing else
  A  Ranges (no hypothesis).  For each plot, and for the mean across plots in each year: the standard
     deviation across years of ln(mice per grid) and its correlation with the next year's value; the
     same for chipmunks and for ln(density of nymphs).  Intervals from a bootstrap over years.
  B  P1  sd of ln NIP < sd of ln mice, plot by plot (NIP = share of questing nymphs infected).
         Fails if it fails on more than one plot of six.
     P2  the within-plot slope b of ln NIP in year t+1 on ln mice in year t satisfies
         0 <= b <= 1 - pbar.  Fails if b is below 0 or above 1 - pbar by more than two standard errors.
  C  One recomputation of the exact loss D of ln R0 for the Borrelia cycle of draft 7 (stays of 1 and
     10/4.4 steps of 4.4 months), by the method of run R10, with sigma and the memory from A.

Readings of the protocol, fixed on 14 November 2026 at 09:45 CST, after reading the data owners'
README and variable key and before any statistic was computed (the protocol leaves these open):
  1. Source for A.  Primary: the Dryad file (PL_grid, TS_grid, DON).  The protocol's words "mice per
     grid" and "density of nymphs" are the words it attaches to those columns, the README defines
     them in full, and they are the measure used in part B.  The Figshare file (PL_t, TS_t,
     nymph_peak_t; to 2025) is reported beside it as the longer record; its key defers to the text
     of a paper that is not in this repository.  Part C uses the primary source only.
  2. sd = sample standard deviation (n - 1) of the ln values across years.  r1 = the ordinary
     correlation between the value in year t and in year t + 1, over the pairs of consecutive years
     that both have a value.
  3. A value of zero is left out of the logarithm and counted (mice, chipmunks, nymphs, and NIP).
  4. "Mean across plots" = ln of the arithmetic mean over the six plots, in the years in which all
     six have a value.
  5. Bootstrap: 2,000 resamples.  A block is two consecutive calendar years; ceil(N/2) blocks are
     drawn with replacement, N being the number of calendar years the variable spans; every plot gets
     the same years.  The sd is taken over the values in the drawn years, r1 over the drawn blocks
     (each block is one pair).  Interval = 2.5th to 97.5th percentile.  Seed 20261114.
  6. Part C: sigma = mean of the six within-plot sd of ln mice; r1 = mean of the six within-plot r1;
     memory per step = r1 ** (4.4/12) if r1 > 0, else 0.
  7. P1 compares, for each plot, sd of ln NIP (years with NIP > 0) with sd of ln PL_grid over that
     plot's years inside the NIP window (1995 to 2013).  The all-year sd from A is printed beside it.
  8. P2: pairs (ln PL_grid in year t, ln NIP in year t + 1) on the same plot; least squares with one
     intercept per plot; standard error = sd of the slope over 2,000 resamples of year blocks as in
     5 (blocks over the mouse years t).  pbar = mean of every NIP value in the file.

Code checks (study rules rule 7; tolerances stated here, before the run; these do not touch the data):
  K1  sd and r1 against hand-worked values, 1e-12.
  K2  the block bootstrap on a series 0,1,0,1,...: every resample has sd = sqrt(5/19) and r1 = -1, 1e-12.
  K3  the within-plot slope against numpy least squares with plot dummies, 1e-9 (synthetic and real).
  K4  gamma0 of run R10 at draft 7's inputs (sigma 0.668, memory 0.30) gives the draft's 0.10, +- 0.005.
  K5  solve_case of run R10 against the exact one-stage value (s = 0.5, sigma = 0.6, memory 0.75):
      within 3 standard errors or 2% (run R10's own tolerance).

Standard library and numpy only in this file.  Part C imports checks/r10_loss_at_large_noise.py,
which itself imports scipy.  Raw data are read from ../../data/raw/ and are not copied.

Run from problems/dobson-r0:
  python3 -I checks/r12_cary_single_population.py > results/r12-cary-single-population_20261114.txt
(about two to four minutes; nearly all of it is part C)
"""
from __future__ import annotations

import csv
import datetime
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

PROBLEM = (__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()
RAW = (__import__('pathlib').Path(__file__).resolve().parent / '../data/raw/cary_lyme').resolve()
DRYAD = RAW / "dryad_d1c8046_forest_food_web_2018" / "Lyme_Study_Dataset.csv"
FIGSHARE = RAW / "figshare_31872853_ecology_of_lyme_2026" / "Full_dataset_20260326.csv"
R10_PATH = PROBLEM / "checks" / "r10_loss_at_large_noise.py"

NBOOT = 2000
SEED = 20261114
SEED_K5 = 2026111401
SEED_C = 2026111412
STEP_MONTHS = 4.4
NAN = float("nan")


# ---------------------------------------------------------------- reading

def read_rows(path):
    with open(path, newline="", encoding="latin-1") as f:
        return list(csv.DictReader(f))


def column(rows, plot_col, col):
    """{plot: {year: value}} for the rows in which the value is present."""
    out = {}
    for r in rows:
        p = r[plot_col]
        if p in ("NA", ""):
            continue
        out.setdefault(p, {})
        v = r[col]
        if v in ("NA", ""):
            continue
        x = float(v)
        if x < 0:
            raise ValueError(f"negative value in {col}: {p} {r['Year']}")
        out[p][int(r["Year"])] = x
    return out


def logs(d):
    """ln of the positive values, and the years whose value is zero."""
    return {y: math.log(v) for y, v in d.items() if v > 0}, sorted(y for y, v in d.items() if v == 0)


def span_text(years):
    """'1996, 1998-2015' for a sorted list of years."""
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


# ---------------------------------------------------------------- statistics

def sd(d):
    v = list(d.values())
    return float(np.std(v, ddof=1)) if len(v) >= 2 else NAN


def lag1(d):
    """Correlation of the value in year t with the value in year t+1; number of pairs."""
    pairs = [(d[y], d[y + 1]) for y in sorted(d) if y + 1 in d]
    if len(pairs) < 3:
        return NAN, len(pairs)
    a = np.array(pairs)
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), len(pairs)


def pooled_mean(by_plot, plots):
    years = sorted(set.intersection(*(set(by_plot[p]) for p in plots)))
    return {y: sum(by_plot[p][y] for p in plots) / len(plots) for y in years}


def to_array(d, y0, y1):
    a = np.full(y1 - y0 + 1, np.nan)
    for y, v in d.items():
        a[y - y0] = v
    return a


def draw_blocks(rng, n_years):
    """Start positions of ceil(n_years/2) blocks of two consecutive years, NBOOT times."""
    return rng.integers(0, n_years - 1, size=(NBOOT, math.ceil(n_years / 2)))


def rowwise_corr(X, Y):
    m = np.isfinite(X) & np.isfinite(Y)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(m, X, 0.0).sum(axis=1) / n
        my = np.where(m, Y, 0.0).sum(axis=1) / n
        dx = np.where(m, X - mx[:, None], 0.0)
        dy = np.where(m, Y - my[:, None], 0.0)
        r = (dx * dy).sum(axis=1) / np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
    r[n < 3] = np.nan
    return r


def boot_sd_r1(arr, starts):
    a0, a1 = arr[starts], arr[starts + 1]
    both = np.concatenate([a0, a1], axis=1)
    cnt = np.isfinite(both).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(both, axis=1) / cnt
        ss = np.nansum((both - mean[:, None]) ** 2, axis=1)
        s = np.sqrt(ss / (cnt - 1))
    s[cnt < 2] = np.nan
    return s, rowwise_corr(a0, a1)


def interval(b):
    ok = b[np.isfinite(b)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), int(len(b) - len(ok))


def fe_slope(xs, ys):
    """Least-squares slope with one intercept per plot.  xs, ys: lists (one per plot) of arrays."""
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
    sxy = np.zeros(NBOOT)
    sxx = np.zeros(NBOOT)
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
        return sxy / sxx


# ---------------------------------------------------------------- code checks K1 to K3

def code_checks():
    fails = []
    # K1
    s = sd({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
    r, n = lag1({1: 1.0, 2: 2.0, 3: 4.0, 4: 3.0, 6: 10.0})
    ok1 = abs(s - math.sqrt(5 / 3)) < 1e-12 and n == 3 and abs(r - 1 / math.sqrt(28 / 3)) < 1e-12
    print(f"  K1  sd {s:.12f} (hand {math.sqrt(5 / 3):.12f}); r1 {r:.12f} from {n} pairs (hand {1 / math.sqrt(28 / 3):.12f})   [{'ok' if ok1 else 'FAIL'}]")
    if not ok1:
        fails.append("K1")
    # K2
    rng = np.random.default_rng(7)
    arr = np.array([float(i % 2) for i in range(20)])
    bs, br = boot_sd_r1(arr, draw_blocks(rng, 20))
    fin = np.isfinite(br)
    ok2 = bool(np.all(np.abs(bs - math.sqrt(5 / 19)) < 1e-12) and np.all(np.abs(br[fin] + 1) < 1e-12) and fin.sum() > NBOOT // 2)
    print(f"  K2  alternating series: bootstrap sd in [{bs.min():.12f}, {bs.max():.12f}] (exact {math.sqrt(5 / 19):.12f}); "
          f"r1 = {np.nanmin(br):.6f} to {np.nanmax(br):.6f} in {int(fin.sum())} resamples (exact -1)   [{'ok' if ok2 else 'FAIL'}]")
    if not ok2:
        fails.append("K2")
    # K3 (synthetic part)
    rng = np.random.default_rng(11)
    xs = [rng.normal(size=8) for _ in range(3)]
    ys = [0.4 * x + c + rng.normal(scale=0.3, size=8) for x, c in zip(xs, (0.0, 2.0, -1.0))]
    b1, b2 = fe_slope(xs, ys), fe_slope_dummies(xs, ys)
    ok3 = abs(b1 - b2) < 1e-9
    print(f"  K3  within-plot slope, synthetic: {b1:.12f} against least squares with dummies {b2:.12f}   [{'ok' if ok3 else 'FAIL'}]")
    if not ok3:
        fails.append("K3")
    return fails


# ---------------------------------------------------------------- part A

def part_a(title, by_plot, plots, rng):
    print(f"\n  {title}")
    y0 = min(min(by_plot[p]) for p in plots if by_plot[p])
    y1 = max(max(by_plot[p]) for p in plots if by_plot[p])
    n_years = y1 - y0 + 1
    starts = draw_blocks(rng, n_years)
    print(f"  calendar span {y0}-{y1} ({n_years} years); each resample draws {starts.shape[1]} blocks of two consecutive years")
    print(f"  {'plot':5s} {'years with a value':20s} {'n':>3s} {'zero':>5s} {'missing inside span':20s} {'sd of ln':>8s}  {'95% interval':16s} {'r1':>7s} {'pairs':>5s}  {'95% interval':16s}")
    res = {"sd": {}, "r1": {}, "n": {}, "zeros": {}}
    boot_sd, boot_r1 = [], []
    for p in plots:
        d = by_plot[p]
        ln, zeros = logs(d)
        years = sorted(d)
        missing = [y for y in range(years[0], years[-1] + 1) if y not in d]
        s = sd(ln)
        r, npairs = lag1(ln)
        bs, br = boot_sd_r1(to_array(ln, y0, y1), starts)
        boot_sd.append(bs)
        boot_r1.append(br)
        slo, shi, sbad = interval(bs)
        rlo, rhi, rbad = interval(br)
        res["sd"][p], res["r1"][p], res["n"][p], res["zeros"][p] = s, r, len(ln), zeros
        note = f"   (resamples without a value: sd {sbad}, r1 {rbad})" if sbad or rbad else ""
        print(f"  {p:5s} {span_text(years):20s} {len(ln):3d} {(span_text(zeros) if zeros else '-'):>5s} {(span_text(missing) if missing else '-'):20s} "
              f"{s:8.3f}  [{slo:.3f}, {shi:.3f}]   {r:7.3f} {npairs:5d}  [{rlo:.3f}, {rhi:.3f}]{note}")
    with np.errstate(invalid="ignore"):
        bsm = np.nanmean(np.array(boot_sd), axis=0)
        brm = np.nanmean(np.array(boot_r1), axis=0)
    res["sd_mean"] = float(np.mean([res["sd"][p] for p in plots]))
    res["r1_mean"] = float(np.mean([res["r1"][p] for p in plots]))
    slo, shi, _ = interval(bsm)
    rlo, rhi, _ = interval(brm)
    res["sd_mean_ci"], res["r1_mean_ci"] = (slo, shi), (rlo, rhi)
    print(f"  mean of the six within-plot values{'':30s} {res['sd_mean']:8.3f}  [{slo:.3f}, {shi:.3f}]   {res['r1_mean']:7.3f} {'':5s}  [{rlo:.3f}, {rhi:.3f}]")
    pm = pooled_mean(by_plot, plots)
    ln, zeros = logs(pm)
    s = sd(ln)
    r, npairs = lag1(ln)
    bs, br = boot_sd_r1(to_array(ln, y0, y1), starts)
    slo, shi, sbad = interval(bs)
    rlo, rhi, rbad = interval(br)
    res["sd_pool"], res["r1_pool"], res["sd_pool_ci"], res["r1_pool_ci"] = s, r, (slo, shi), (rlo, rhi)
    res["pool_years"] = sorted(pm)
    note = f"   (resamples without a value: sd {sbad}, r1 {rbad})" if sbad or rbad else ""
    print(f"  mean across the six plots, then ln; years with all six: {span_text(sorted(pm))} (n = {len(ln)}, zeros {len(zeros)})")
    print(f"  {'':64s} {s:8.3f}  [{slo:.3f}, {shi:.3f}]   {r:7.3f} {npairs:5d}  [{rlo:.3f}, {rhi:.3f}]{note}")
    return res


# ---------------------------------------------------------------- main

def main():
    t_start = time.time()
    now = datetime.datetime.now().astimezone()
    print(f"R12: Cary Institute plots, one population with hosts and infection measured together.  Started {now:%Y-%m-%d %H:%M:%S %Z}.")
    print("A pattern check on one landscape; not a test of the predicted loss of R0.  Protocol: protocols/R12-cary-single-population.md")
    print(f"Files read: {DRYAD.relative_to(RAW.parent.parent.parent)}")
    print(f"            {FIGSHARE.relative_to(RAW.parent.parent.parent)}")

    print("\n0. Code checks that do not touch the data")
    fails = code_checks()
    if fails:
        print(f"FAILED code checks {fails}; stopping before the data are read.")
        return 1

    rng = np.random.default_rng(SEED)
    dry = read_rows(DRYAD)
    fig = read_rows(FIGSHARE)
    dplots = sorted({r["Grid"] for r in dry if r["Grid"] not in ("NA", "")})
    fplots = sorted({r["Plot"] for r in fig if r["Plot"] not in ("NA", "")})
    n_noplot = sum(1 for r in fig if r["Plot"] in ("NA", ""))
    print("\n1. What was read")
    print(f"  Dryad file: {len(dry)} rows, plots {dplots}, years {min(int(r['Year']) for r in dry)}-{max(int(r['Year']) for r in dry)}")
    print(f"  Figshare file: {len(fig)} rows ({n_noplot} without a plot label, left out), plots {fplots}, "
          f"years {min(int(r['Year']) for r in fig)}-{max(int(r['Year']) for r in fig)}")

    # --- the NIP column: how it is put together (README p. 3: a weighted average of two assays)
    nip = column(dry, "Grid", "NIP")
    cols = {c: column(dry, "Grid", c) for c in ("TFA_infected", "TFA_total", "PCR_positive", "PCR_total", "TFA_NIP_CORR", "PCR_NIP", "TFA_fraction", "PCR_fraction")}
    kinds = {"fluorescence only": [], "PCR only": [], "both, weighted": [], "other": []}
    worst = 0.0
    ratio = []
    tested = []
    for p in dplots:
        for y, v in sorted(nip[p].items()):
            tfa, pcr = cols["TFA_NIP_CORR"][p].get(y), cols["PCR_NIP"][p].get(y)
            wt, wp = cols["TFA_fraction"][p].get(y), cols["PCR_fraction"][p].get(y)
            if tfa is not None and pcr is not None and wt is not None and wp is not None:
                kinds["both, weighted"].append(y)
                worst = max(worst, abs(v - (wt * tfa + wp * pcr)))
            elif tfa is not None and pcr is None:
                kinds["fluorescence only"].append(y)
                worst = max(worst, abs(v - tfa))
            elif pcr is not None and tfa is None:
                kinds["PCR only"].append(y)
                worst = max(worst, abs(v - pcr))
            else:
                kinds["other"].append(y)
            ti, tt = cols["TFA_infected"][p].get(y), cols["TFA_total"][p].get(y)
            if tfa is not None and ti and tt:
                ratio.append(tfa / (ti / tt))
            tested.append((cols["TFA_total"][p].get(y) or 0.0) + (cols["PCR_total"][p].get(y) or 0.0))
    print("  NIP (share of questing nymphs infected), by how the value was obtained (plot-years):")
    for k, ys in kinds.items():
        print(f"    {k:18s} {len(ys):3d}   years {span_text(sorted(set(ys)))}")
    print(f"    largest |NIP - value rebuilt from the assay columns| = {worst:.2e}")
    if ratio:
        print(f"    corrected fluorescence share / (infected / tested): {min(ratio):.4f} to {max(ratio):.4f} over {len(ratio)} plot-years")
    print(f"    nymphs tested per plot and year (both assays): least {min(tested):.0f}, median {float(np.median(tested)):.0f}, most {max(tested):.0f}")

    # --- part A
    print("\n2. Part A: ranges (no hypothesis).  sd = standard deviation of ln values across years; r1 = correlation with the next year.")
    print("   Intervals: 2,000 resamples of blocks of two consecutive years, the same years for every plot; seed 20261114.")
    A = {}
    print("\n  PRIMARY SOURCE: Dryad file (README pp. 1-2)")
    A["d_mice"] = part_a("A1  mice: PL_grid, Jolly-Seber estimate per 2.25-ha grid at 15 August", column(dry, "Grid", "PL_grid"), dplots, rng)
    A["d_chip"] = part_a("A2  chipmunks: TS_grid, Jolly-Seber estimate per 2.25-ha grid at 15 August", column(dry, "Grid", "TS_grid"), dplots, rng)
    A["d_nymph"] = part_a("A3  nymphs: DON, questing nymphs per m2 of drag area at the seasonal peak", column(dry, "Grid", "DON"), dplots, rng)
    print("\n  LONGER RECORD, FOR COMPARISON: Figshare file (variable key)")
    A["f_mice"] = part_a("A4  mice: PL_t, minimum number alive per 2.25-ha grid", column(fig, "Plot", "PL_t"), fplots, rng)
    A["f_chip"] = part_a("A5  chipmunks: TS_t, minimum number alive per 2.25-ha grid", column(fig, "Plot", "TS_t"), fplots, rng)
    A["f_nymph"] = part_a("A6  nymphs: nymph_peak_t, peak questing nymphs per m2", column(fig, "Plot", "nymph_peak_t"), fplots, rng)

    am = A["d_mice"]
    print("\n  Beside draft 7's inputs for this row (sigma 0.67, range 0.46 to 0.87; memory 0.30 per step of 4.4 months):")
    print(f"    mean within-plot sd of ln mice, Dryad file   {am['sd_mean']:.3f}  [{am['sd_mean_ci'][0]:.3f}, {am['sd_mean_ci'][1]:.3f}]"
          f"   (Figshare file {A['f_mice']['sd_mean']:.3f}  [{A['f_mice']['sd_mean_ci'][0]:.3f}, {A['f_mice']['sd_mean_ci'][1]:.3f}])")
    print(f"    mean within-plot r1 of ln mice, Dryad file   {am['r1_mean']:.3f}  [{am['r1_mean_ci'][0]:.3f}, {am['r1_mean_ci'][1]:.3f}]"
          f"   (Figshare file {A['f_mice']['r1_mean']:.3f}  [{A['f_mice']['r1_mean_ci'][0]:.3f}, {A['f_mice']['r1_mean_ci'][1]:.3f}])")
    print(f"    a memory of 0.30 per 4.4 months, if it decays geometrically, is a one-year correlation of 0.30 ** (12/4.4) = {0.30 ** (12 / STEP_MONTHS):.3f}")
    sys.stdout.flush()

    # --- part B
    print("\n3. Part B: two pattern checks (Dryad file: PL_grid and NIP on the same plots)")
    mice = column(dry, "Grid", "PL_grid")
    print("\n  P1  sd of ln NIP < sd of ln mice, plot by plot.  Fails if it fails on more than one plot of six.")
    print(f"  {'plot':5s} {'NIP years':12s} {'n':>3s} {'zero NIP':>9s} {'sd ln NIP':>10s}   {'mice years in window':22s} {'n':>3s} {'zero':>5s} {'sd ln mice':>10s}   {'NIP varies less':15s}  {'[sd ln mice, all years, A1]':s}")
    n_fail_p1 = 0
    for p in dplots:
        ln_n, z_n = logs(nip[p])
        w0, w1 = min(nip[p]), max(nip[p])
        m_win = {y: v for y, v in mice[p].items() if w0 <= y <= w1}
        ln_m, z_m = logs(m_win)
        s_n, s_m = sd(ln_n), sd(ln_m)
        holds = s_n < s_m
        n_fail_p1 += 0 if holds else 1
        print(f"  {p:5s} {span_text(sorted(nip[p])):12s} {len(ln_n):3d} {(span_text(z_n) if z_n else '-'):>9s} {s_n:10.3f}   {span_text(sorted(m_win)):22s} {len(ln_m):3d} "
              f"{(span_text(z_m) if z_m else '-'):>5s} {s_m:10.3f}   {('yes' if holds else 'NO'):15s}  [{am['sd'][p]:.3f}]")
    p1 = "HOLDS" if n_fail_p1 <= 1 else "FAILS"
    print(f"  P1 {p1}: the inequality fails on {n_fail_p1} of {len(dplots)} plots (criterion: fails if more than one).")

    print("\n  P2  slope b of ln NIP in year t+1 on ln mice in year t, one intercept per plot; 0 <= b <= 1 - pbar.")
    xs, ys, pair_years = [], [], []
    n_zero_pairs = 0
    for p in dplots:
        x, y, yrs = [], [], []
        for t in sorted(mice[p]):
            if t + 1 in nip[p]:
                if mice[p][t] > 0 and nip[p][t + 1] > 0:
                    x.append(math.log(mice[p][t]))
                    y.append(math.log(nip[p][t + 1]))
                    yrs.append(t)
                else:
                    n_zero_pairs += 1
        xs.append(x)
        ys.append(y)
        pair_years.append(yrs)
        print(f"    {p}: {len(x):2d} pairs, mouse years {span_text(yrs)}")
    n_pairs = sum(len(x) for x in xs)
    b = fe_slope(xs, ys)
    b_dum = fe_slope_dummies(xs, ys)
    ok3 = abs(b - b_dum) < 1e-9
    t0 = min(min(yrs) for yrs in pair_years)
    t1 = max(max(yrs) for yrs in pair_years)
    starts = draw_blocks(rng, t1 - t0 + 1)
    xarrs = [to_array(dict(zip(yrs, x)), t0, t1) for yrs, x in zip(pair_years, xs)]
    yarrs = [to_array(dict(zip(yrs, y)), t0, t1) for yrs, y in zip(pair_years, ys)]
    bb = fe_slope_boot(xarrs, yarrs, starts)
    bb_ok = bb[np.isfinite(bb)]
    se = float(np.std(bb_ok, ddof=1))
    all_nip = [v for p in dplots for v in nip[p].values()]
    pbar = float(np.mean(all_nip))
    pbar_used = float(np.mean([math.exp(v) for y in ys for v in y]))
    upper = 1 - pbar
    print(f"    pairs used {n_pairs}; pairs left out because of a zero {n_zero_pairs}; mouse years {t0}-{t1}; {starts.shape[1]} blocks per resample")
    print(f"    b = {b:.4f}   (least squares with plot dummies {b_dum:.4f}; K3 on the data [{'ok' if ok3 else 'FAIL'}])")
    print(f"    bootstrap standard error {se:.4f}   ({len(bb_ok)} resamples; their mean {float(bb_ok.mean()):.4f}; 2.5th to 97.5th percentile {np.percentile(bb_ok, 2.5):.4f} to {np.percentile(bb_ok, 97.5):.4f})")
    print(f"    pbar = {pbar:.4f} (mean of all {len(all_nip)} NIP values; mean of the {n_pairs} values used as response {pbar_used:.4f});  1 - pbar = {upper:.4f}")
    print(f"    allowed by the criterion: {-2 * se:.4f} <= b <= {upper + 2 * se:.4f}   (0 - 2 se and 1 - pbar + 2 se)")
    p2_fail = b < -2 * se or b > upper + 2 * se
    inside = 0 <= b <= upper
    p2 = "FAILS" if p2_fail else "HOLDS"
    print(f"  P2 {p2}: b is {'inside' if inside else 'outside'} the interval [0, {upper:.3f}]; "
          f"distance below 0: {max(0.0, -b) / se:.2f} se; distance above 1 - pbar: {max(0.0, b - upper) / se:.2f} se.")
    if not ok3:
        fails.append("K3 on the data")
    sys.stdout.flush()

    # --- part C
    print("\n4. Part C: one recomputation of the loss D of ln R0 for the Borrelia cycle of draft 7 (stays 1 and 10/4.4 steps), method of run R10")
    spec = importlib.util.spec_from_file_location("r10_loss_at_large_noise", R10_PATH)
    r10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r10)
    stays = [1, 10 / STEP_MONTHS]
    g_draft = r10.gamma0(stays, [0.668, 0], [0.30, 0])
    ok4 = abs(g_draft - 0.10) < 0.005
    print(f"  K4  second-order value at draft 7's inputs (sigma 0.668, memory 0.30): {g_draft:.4f}; the draft's discrete-time entry is 0.10   [{'ok' if ok4 else 'FAIL'}]")
    if not ok4:
        fails.append("K4")
    sys.stdout.flush()
    _, _, _, _, Dk, sek, _, dtk = r10.solve_case(("K5 one stage", [2.0], [0.6], [0.75], SEED_K5))
    exact = r10.one_stage_exact(0.5, 0.6)
    ok5 = abs(Dk - exact) <= max(3 * sek, 0.02 * exact)
    print(f"  K5  one stage, s = 0.5, sigma = 0.6, memory 0.75: D = {Dk:.5f} +- {sek:.5f}; exact {exact:.5f}   [{'ok' if ok5 else 'FAIL'}]   ({dtk:.0f} s)")
    if not ok5:
        fails.append("K5")
    sys.stdout.flush()

    sigma = am["sd_mean"]
    r1 = am["r1_mean"]
    if r1 > 0:
        phi = r1 ** (STEP_MONTHS / 12)
        print(f"  inputs from A1: sigma = {sigma:.4f}; r1 = {r1:.4f}; memory per step of 4.4 months = r1 ** (4.4/12) = {phi:.4f}")
    else:
        phi = 0.0
        print(f"  inputs from A1: sigma = {sigma:.4f}; r1 = {r1:.4f} is not positive, so the memory per step is set to zero (protocol, part C)")
    print(f"  (r1 of the plot-mean series, A1: {am['r1_pool']:.4f}; it {'has the same sign' if (am['r1_pool'] > 0) == (r1 > 0) else 'has the OTHER sign'})")
    print(f"  M = {r10.M} populations, T = {r10.T} steps, burn-in {r10.BURN}; seed {SEED_C}")
    sys.stdout.flush()
    _, _, _, _, D, se_D, g0, dt = r10.solve_case(("R12 Borrelia, Cary inputs", stays, [sigma, 0], [phi, 0], SEED_C))
    print(f"  D = {D:.4f} +- {se_D:.4f} (one standard error of the simulation)   second-order value {g0:.4f}   no-staying value sigma^2/2 = {sigma * sigma / 2:.4f}   ({dt:.0f} s)")
    print("  draft 7, this row: 0.05 to 0.10 (second-order values, continuous and discrete time) with sigma 0.67 and memory 0.30")

    print(f"\nSummary: P1 {p1} ({n_fail_p1} of {len(dplots)} plots against it); P2 {p2} (b = {b:.3f}, se {se:.3f}, allowed 0 to {upper:.3f} with two se either side); "
          f"D = {D:.4f} +- {se_D:.4f} at sigma {sigma:.3f}, memory {phi:.3f}.")
    print(f"Finished {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}; wall time {time.time() - t_start:.0f} s.")
    print("CODE CHECKS K1 TO K5: ALL PASSED" if not fails else f"CODE CHECKS FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
