"""Run R12c (protocols/R12c-cary-how-mice-enter.md): how strongly do mouse numbers enter the two steps
of the Borrelia cycle on the six Cary Institute plots (Millbrook, New York), and how far does the loss D
of ln R0 in the two-stage cycle of draft 7 move when the measured strength is put in?

The data set a range (study rules, rule 10).  Nothing is fitted to the theory and no mechanism is added.
This is not an estimate of the loss for Borrelia (see the protocol, "What this can and cannot show").

What is computed, and nothing else
  1  kappa2: the within-plot slope of ln(larvae per mouse, year t) on ln(mice, year t), one intercept per
     plot, standard error from R12's bootstrap over years.  Expectation stated in the protocol:
     -1 <= kappa2 <= 0.  It FAILS if the estimate is outside that interval by more than two standard errors.
  2  kappa1 = 1 + (the same slope for nymphs per mouse), only if the file has a per-mouse burden of nymphs.
  3  Reported, no hypothesis: the within-plot slopes of ln(density of nymphs, year t+1) and of
     ln(density of infected nymphs, year t+1) on ln(mice, year t).
  4  The loss D recomputed exactly (the shift y of one transfer at which long-run growth is zero) for a
     two-stage ring in which ONE host series z(t) multiplies both transfers, with the second-order value of
     Proposition 4a beside it.

What the data owners' README says (README_for_Lyme_Study_Dataset.pdf, four pages, read in full before
anything was computed):
  LPM      p. 2  "Larval tick body burden on white-footed mice: average number of larval ticks on mice for a
                 given grid and year, during the August trapping session"; units "Average number of larval
                 ticks per mouse".
  PL_grid  p. 2  Jolly-Seber estimate of white-footed mice per 2.25-ha trapping grid, interpolated between
                 trapping sessions to 15 August, the peak of larval ticks.
  DON      p. 1  density of nymphal blacklegged ticks at peak density for a grid, per m2 of drag area.
  NIP      p. 3  weighted average prevalence of B. burgdorferi in nymphs (two assays combined).
  TOTLAR   p. 3  estimated density of larvae per m2 of ground, combining drag-cloth larvae with the estimated
                 density of larvae found on mice.  A density of larvae, not a burden per mouse.
  No column of the README (59 names, pages 1 to 4) is a per-mouse burden of NYMPHS, and the file's header
  has no column beyond those 59 and an unnamed row number.  So part 2 cannot be done: kappa1 is not
  estimated, and the script says so.

Decisions the protocol leaves open, fixed on 14 November 2026 at 10:11 CST, after the README and the header
line of the file were read and before any value of the file was read by me or any statistic computed.
(What I knew of the values: only what R12's report states, namely that at HC in 1996 PL_grid is 0 while LPM
is 15.5, the years covered by PL_grid, DON and NIP, and that NIP is 0 at TX in 2007.)
   1. Years.  Only rows of the file with Year 1995 to 2013 are used, on both sides of every pair (the
      protocol's title and its Data section).  R12 read the same words as the span of NIP and also used
      earlier mouse years; here they are read as a restriction.  For the lagged slopes of part 3 the mouse
      year t therefore runs 1995 to 2012 and the tick year t+1 runs 1996 to 2013.
   2. Columns.  mice = PL_grid (R12's primary measure); larvae per mouse = LPM; density of nymphs = DON;
      density of infected nymphs = DON x NIP of the same plot and year.
   3. Zero rule (R12's): a pair with a zero in either variable is left out and counted.  A pair needs a
      value on both sides; plot-years with a value on one side only are counted too.
   4. Slope: R12's fe_slope (least squares, one intercept per plot), imported from
      checks/r12_cary_single_population.py, not rewritten.
   5. Standard error: R12's bootstrap, imported.  2,000 resamples; a block is two consecutive calendar
      years; ceil(N/2) blocks drawn with replacement, N being the calendar span of the mouse years of the
      pairs used; every plot gets the same years; standard error = standard deviation (n - 1) of the slope
      over the resamples.  One generator, seed 2026111431, used in the fixed order: kappa2, nymph density,
      infected-nymph density.  (R12's own random stream cannot be continued; a new seed is used.)
      The two slopes of part 3 are given with this standard error although the protocol asks only for
      the slopes.
   6. Expectation of part 1: FAILS if kappa2 < -1 - 2 se or kappa2 > 0 + 2 se; otherwise HOLDS.  Whether
      the estimate is inside [-1, 0] itself is printed as well.
   7. sigma.  "R12's values" are taken as R12's saved output gives them
      (results/r12-cary-single-population_20261114.txt): 1.0304 (line 142; Dryad PL_grid) and 0.772
      (line 109; Figshare PL_t, three decimals are all that was saved).  The protocol writes 1.03 and 0.77.
      They are typed in below, not recomputed.
   8. The cycle.  Stage 0 = mouse, stay 1 step (s = 0, g = 1); stage 1 = tick, stay 10/4.4 = 2.2727 steps
      (s = 0.56, g = 0.44).  The transfer out of stage 0 (mouse to tick: larvae feeding on a mouse) is
      multiplied by exp(kappa2 sigma z - (kappa2 sigma)^2/2); the transfer out of stage 1 (tick to mouse:
      nymphs finding a mouse) by exp(kappa1 sigma z - (kappa1 sigma)^2/2), with the SAME z(t) in the same
      step.  z(t) is standard Gaussian and independent from step to step.  exp(y) multiplies the transfer
      out of stage 0, as in `rates` of run R10.  This is the indexing of
      checks/real_systems_two_transfers.py: second-order loss per generation
      = sigma^2 x sum(tau) x closed_form(tau, 1, 0, kappa1, kappa2, 0).
   9. Simulation.  200 populations, 200,000 steps after a burn-in of 2,000; start with half in each stage;
      numpy default_rng(seed), z drawn 1,000 steps at a time, the generator re-seeded for every y so that
      every y meets the same numbers.  Secant method exactly as `solve_case` of run R10: first two points
      the second-order value g0 and 1.5 g0 + 0.005, at most 14 further evaluations, stop when the step is
      below 1e-6; standard error = (standard deviation over populations of the growth rate at the last y
      evaluated) / sqrt(200) / (last secant slope).  Seeds: 2026111451 (check C1), 2026111452 (check C2),
      2026111460 + k for the 18 cases, k = 0..17 in the order sigma (1.0304, 0.772), kappa2 (estimate - 2 se,
      estimate, estimate + 2 se), kappa1 (0, 0.5, 1).  At most 4 processes.
  10. Grid of part 4: kappa2 at the unrounded estimate and at estimate -+ 2 unrounded standard errors, run
      as they come (no clipping to [-1, 0]); kappa1 = 0, 0.5, 1; both sigma.  18 cases.
  11. Order.  Every code check, including the two simulation checks, runs BEFORE the data file is opened.
      If one fails the script stops there.

Code checks (study rules rule 7), tolerances stated here before the run:
  K1  R12's within-plot slope against numpy least squares with plot dummies, 1e-9 (synthetic numbers; and
      again on each slope from the data).
  K2  R12's bootstrap of the slope on exact lines y = c_plot + 0.7 x: every resample returns 0.7, 1e-9.
  K3  the second-order function at kappa1 = 1, kappa2 = 0, sigma = 1.0304 gives R12's printed 0.1622,
      +- 0.00005.
  C1  (protocol) the new simulation at kappa1 = 1, kappa2 = 0, sigma = 1.0304 (the protocol's 1.030; it is
      the sigma at which R12 got its number) reproduces R12's D = 0.2013 within three standard errors.
      R12's number is itself a simulation with standard error 0.0002 and another seed, so the criterion is
      |D - 0.2013| <= 3 sqrt(se^2 + 0.0002^2).  The stricter reading, |D - 0.2013| <= 3 se of the new run
      alone, is printed beside it.  (With no memory, a fluctuation on the transfer out of the tick is the
      same process as R12's fluctuation on the transfer out of the mouse, moved by one step.)
  C2  (protocol) at kappa1 = kappa2 = 0 the simulation gives D = 0 within three standard errors.  No noise
      enters, so the standard error is zero by construction; the criterion is |D| <= max(3 se, 1e-6), 1e-6
      being the tolerance of the secant method.

Note added at 10:13 CST on 14 November 2026, after one dry run of this script on a file of random numbers
(made-up values under the real column names, a short simulation; the data file had still not been opened)
and before the first run on the data.  The protocol's two checks use kappa2 = 0 only, so they never
exercise the multiplier on the transfer out of the mouse, which every case of part 4 uses.  In the dry run
the exact values lay far above the second-order ones when the two relative sizes had opposite signs, so
that path needs its own check before the table is believed.  Two checks are added; they are not hypotheses
and use no data:
  C3  kappa1 = 0, kappa2 = 1, sigma = 1.0304.  This is literally R12's configuration (the fluctuation on
      the transfer out of the mouse); it must give 0.2013 by the same criterion as C1.  Seed 2026111453.
  C4  kappa1 = kappa2 = 1, sigma = 0.1 (small noise): D within 5% of the second-order value
      4 sigma^2 / (2 sum(tau)) = 0.00611, or within 3 standard errors if that is larger (the tolerance of
      R10's check H1c).  Two independent series would give half of it and a wrong sign on one multiplier
      would give nearly zero, so this checks that both transfers meet the same z with the stated signs.
      Seed 2026111454.
If C1 or C2 (the protocol's) or C3 or C4 fails, the script stops before the data are read.

Standard library and numpy only.  Imports checks/r12_cary_single_population.py (numpy only) and
checks/one_host_two_transfers.py (which imports engine/theory.py, numpy only).  R10's module is not
imported.  Raw data are read from ../../data/raw/ and are not copied.

Run from problems/dobson-r0:
  python3 -I checks/r12c_cary_how_mice_enter.py > results/r12c-cary-how-mice-enter_20261114.txt
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import datetime
import importlib.util
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

PROBLEM = (__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()
RAW = (__import__('pathlib').Path(__file__).resolve().parent / '../data/raw/cary_lyme').resolve()
DRYAD = RAW / "dryad_d1c8046_forest_food_web_2018" / "Lyme_Study_Dataset.csv"
R12_PATH = PROBLEM / "checks" / "r12_cary_single_population.py"
W_PATH = PROBLEM / "checks" / "one_host_two_transfers.py"

Y0, Y1 = 1995, 2013
SEED_BOOT = 2026111431
SEED_C1, SEED_C2, SEED_GRID = 2026111451, 2026111452, 2026111460
SEED_C3, SEED_C4, SIGMA_C4 = 2026111453, 2026111454, 0.1
M, T, BURN, CHUNK = 200, 200_000, 2_000, 1_000
NPROC = 4
STEP_MONTHS = 4.4
STAYS = (1.0, 10 / STEP_MONTHS)          # stage 0 = mouse, stage 1 = tick
SIGMAS = ((1.0304, "R12, Dryad PL_grid (Jolly-Seber estimate); the protocol's 1.03"),
          (0.772, "R12, Figshare PL_t (minimum number alive); the protocol's 0.77"))
KAPPA1 = (0.0, 0.5, 1.0)
R12_D, R12_D_SE, R12_SECOND_ORDER = 0.2013, 0.0002, 0.1622

README_COLUMNS = [
    # page 1
    "Year", "Grid", "tot_days", "tot_warm_days", "feb_apr_warm_days", "winter_days", "cold_winter_days",
    "degree_days", "mat", "snowfall_cm", "DOL", "DON", "DOA",
    # page 2
    "LPM", "PL_grid", "PL_ha", "TS_grid", "TS_ha", "QURU", "QUVE", "QUAL", "QUPR", "ACSA", "REDOAK", "WHITEOAK",
    "TOT.MATURE.ACORNS", "Ave.Deer.Hr", "TFA_infected",
    # page 3
    "TFA_total", "PCR_positive", "PCR_total", "TFA_NIP_CORR", "PCR_NIP", "TFA_fraction", "PCR_fraction", "NIP",
    "TOTLAR", "grid_year", "grid_year_tminus1", "grid_year_tminus2", "deer_tminus2", "deer_tminus1", "CDD_tminus2",
    "acorns_tminus2", "acsa_tminus2",
    # page 4
    "mouse_tminus1", "mouse_t", "chip_tminus1", "chip_t", "rodents_tminus1", "rodents_t", "dol_tminus1", "don",
    "acorns_tminus1", "redoak_tminus1", "whiteoak_tminus1", "redoak_tminus2", "whiteoak_tminus2", "totlar_tminus1",
]
PER_MOUSE_BURDEN_COLUMNS = ["LPM"]       # the only column the README describes as a tick burden per mouse


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the simulation (part 4)

def rates2(sigma, k1, k2, y, seed, n_pop, n_steps, burn):
    """Growth rate per step of each of n_pop populations of the two-stage ring.

    Stage 0 (mouse, stay STAYS[0]) and stage 1 (tick, stay STAYS[1]); one Gaussian series z(t), independent
    from step to step; the transfer out of stage 0 carries exp(k2 sigma z - (k2 sigma)^2/2) and exp(y), the
    transfer out of stage 1 carries exp(k1 sigma z - (k1 sigma)^2/2), both with the same z(t).
    """
    s0, g0 = 1 - 1 / STAYS[0], 1 / STAYS[0]
    s1, g1 = 1 - 1 / STAYS[1], 1 / STAYS[1]
    a1, a2 = k1 * sigma, k2 * sigma
    ey = math.exp(y)
    rng = np.random.default_rng(seed)
    x0 = np.full(n_pop, 0.5)
    x1 = np.full(n_pop, 0.5)
    acc = np.zeros(n_pop)
    tots = np.empty((CHUNK, n_pop))
    assert burn % CHUNK == 0 and n_steps % CHUNK == 0
    for c in range((burn + n_steps) // CHUNK):
        z = rng.standard_normal((CHUNK, n_pop))
        out0 = (g0 * ey) * np.exp(a2 * z - a2 * a2 / 2)      # mouse -> tick, per unit in stage 0
        out1 = g1 * np.exp(a1 * z - a1 * a1 / 2)             # tick -> mouse, per unit in stage 1
        for t in range(CHUNK):
            n0 = s0 * x0 + out1[t] * x1
            n1 = s1 * x1 + out0[t] * x0
            tot = n0 + n1
            tots[t] = tot
            x0 = n0 / tot
            x1 = n1 / tot
        if c * CHUNK >= burn:
            acc += np.log(tots).sum(axis=0)
    return acc / n_steps


def solve(case):
    """The secant method of `solve_case` (run R10) on rates2.  Returns D, its standard error and bookkeeping."""
    label, sigma, k1, k2, seed, g0, n_pop, n_steps, burn = case
    t0 = time.time()
    y_prev = g0
    r_prev = rates2(sigma, k1, k2, y_prev, seed, n_pop, n_steps, burn)
    y = g0 * 1.5 + 0.005
    slope = None
    n_eval = 1
    converged = False
    for _ in range(14):
        r = rates2(sigma, k1, k2, y, seed, n_pop, n_steps, burn)
        n_eval += 1
        slope = (r.mean() - r_prev.mean()) / (y - y_prev)
        y_new = y - r.mean() / slope
        y_prev, r_prev = y, r
        if abs(y_new - y) < 1e-6:
            y = y_new
            converged = True
            break
        y = y_new
    se = float(r_prev.std(ddof=1) / math.sqrt(n_pop) / slope)
    return label, sigma, k1, k2, float(y), se, g0, n_eval, converged, time.time() - t0


def second_order(w, sigma, k1, k2):
    """Loss per generation to second order (Proposition 4a), no memory: stage 1 carries kappa1, stage 0 kappa2."""
    return sigma * sigma * sum(STAYS) * w.closed_form(list(STAYS), 1, 0, k1, k2, 0.0)


# ---------------------------------------------------------------- slopes (parts 1 and 3)

def within_plot_slope(r12, title, xby, yby, lag, plots, rng):
    """Slope of ln y(year t + lag) on ln x(year t), one intercept per plot; R12's bootstrap over years."""
    print(f"\n  {title}")
    xs, ys, pair_years, zeros = [], [], [], []
    x_only = y_only = 0
    for p in plots:
        x, y, yrs = [], [], []
        for t in range(Y0, Y1 + 1 - lag):
            xv, yv = xby[p].get(t), yby[p].get(t + lag)
            if xv is None and yv is None:
                continue
            if yv is None:
                x_only += 1
                continue
            if xv is None:
                y_only += 1
                continue
            if xv > 0 and yv > 0:
                x.append(math.log(xv))
                y.append(math.log(yv))
                yrs.append(t)
            else:
                zeros.append(f"{p} {t}" + (f"/{t + lag}" if lag else "") + f" ({'x' if xv == 0 else ''}{'y' if yv == 0 else ''} = 0)")
        xs.append(x)
        ys.append(y)
        pair_years.append(yrs)
        print(f"    {p}: {len(x):2d} pairs, mouse years {r12.span_text(yrs)}")
    n_pairs = sum(len(x) for x in xs)
    b = r12.fe_slope(xs, ys)
    b_dum = r12.fe_slope_dummies(xs, ys)
    ok = abs(b - b_dum) < 1e-9
    t0 = min(min(yrs) for yrs in pair_years if yrs)
    t1 = max(max(yrs) for yrs in pair_years if yrs)
    starts = r12.draw_blocks(rng, t1 - t0 + 1)
    xarrs = [r12.to_array(dict(zip(yrs, x)), t0, t1) for yrs, x in zip(pair_years, xs)]
    yarrs = [r12.to_array(dict(zip(yrs, y)), t0, t1) for yrs, y in zip(pair_years, ys)]
    bb = r12.fe_slope_boot(xarrs, yarrs, starts)
    bb_ok = bb[np.isfinite(bb)]
    se = float(np.std(bb_ok, ddof=1))
    print(f"    pairs used {n_pairs}; left out because of a zero {len(zeros)}{(': ' + '; '.join(zeros)) if zeros else ''}")
    print(f"    plot-years with a value of x but none of y: {x_only}; of y but none of x: {y_only}")
    print(f"    mouse years {t0}-{t1}; {starts.shape[1]} blocks of two years per resample")
    print(f"    slope = {b:.4f}   (least squares with plot dummies {b_dum:.4f}; K1 on the data [{'ok' if ok else 'FAIL'}])")
    print(f"    bootstrap standard error {se:.4f}   ({len(bb_ok)} resamples; their mean {float(bb_ok.mean()):.4f}; "
          f"2.5th to 97.5th percentile {np.percentile(bb_ok, 2.5):.4f} to {np.percentile(bb_ok, 97.5):.4f})")
    return b, se, n_pairs, ok


# ---------------------------------------------------------------- code checks that need no data

def cheap_checks(r12, w):
    fails = []
    rng = np.random.default_rng(11)
    xs = [rng.normal(size=8) for _ in range(3)]
    ys = [0.4 * x + c + rng.normal(scale=0.3, size=8) for x, c in zip(xs, (0.0, 2.0, -1.0))]
    b1, b2 = r12.fe_slope(xs, ys), r12.fe_slope_dummies(xs, ys)
    ok = abs(b1 - b2) < 1e-9
    print(f"  K1  within-plot slope, synthetic: {b1:.12f} against least squares with dummies {b2:.12f}   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K1")

    rng = np.random.default_rng(12)
    n_years = 19
    xa = [rng.normal(size=n_years) for _ in range(3)]
    for a in xa[1:]:
        a[rng.integers(0, n_years, size=3)] = np.nan          # plots with missing years, as in the data
    ya = [c + 0.7 * a for a, c in zip(xa, (0.0, 2.0, -1.0))]
    bb = r12.fe_slope_boot(xa, ya, r12.draw_blocks(rng, n_years))
    fin = np.isfinite(bb)
    ok = bool(fin.sum() > r12.NBOOT // 2 and np.all(np.abs(bb[fin] - 0.7) < 1e-9))
    print(f"  K2  bootstrap of the slope on exact lines of slope 0.7: {bb[fin].min():.12f} to {bb[fin].max():.12f} "
          f"in {int(fin.sum())} resamples   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K2")

    g = second_order(w, SIGMAS[0][0], 1.0, 0.0)
    ok = abs(g - R12_SECOND_ORDER) < 0.00005
    print(f"  K3  second-order value at kappa1 = 1, kappa2 = 0, sigma = {SIGMAS[0][0]}: {g:.5f}; R12 printed {R12_SECOND_ORDER}   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        fails.append("K3")
    return fails


# ---------------------------------------------------------------- main

def main(dryad=DRYAD, n_steps=T):
    t_start = time.time()
    now = datetime.datetime.now().astimezone()
    print(f"R12c: how mouse numbers enter the two steps of the Borrelia cycle, Cary Institute plots {Y0} to {Y1}.  Started {now:%Y-%m-%d %H:%M:%S %Z}.")
    print("Protocol: protocols/R12c-cary-how-mice-enter.md.  The data set a range; nothing is fitted; not an estimate of the loss for Borrelia.")
    r12 = load(R12_PATH, "r12_cary_single_population")
    w = load(W_PATH, "one_host_two_transfers")
    fails = []

    print("\n0. Code checks, run before the data file is opened")
    fails += cheap_checks(r12, w)
    print(f"  simulation: {M} populations, {n_steps} steps after a burn-in of {BURN}; stays {STAYS[0]:g} (mouse) and {STAYS[1]:.4f} (tick); no memory; at most {NPROC} processes")
    sys.stdout.flush()
    sig0 = SIGMAS[0][0]
    check_cases = [("C1", sig0, 1.0, 0.0, SEED_C1, second_order(w, sig0, 1.0, 0.0), M, n_steps, BURN),
                   ("C2", sig0, 0.0, 0.0, SEED_C2, second_order(w, sig0, 0.0, 0.0), M, n_steps, BURN),
                   ("C3", sig0, 0.0, 1.0, SEED_C3, second_order(w, sig0, 0.0, 1.0), M, n_steps, BURN),
                   ("C4", SIGMA_C4, 1.0, 1.0, SEED_C4, second_order(w, SIGMA_C4, 1.0, 1.0), M, n_steps, BURN)]
    with mp.Pool(NPROC) as pool:
        c1, c2, c3, c4 = pool.map(solve, check_cases, chunksize=1)
    D1 = se1 = None
    for name, what, c, seed in (("C1", "kappa1 = 1, kappa2 = 0", c1, SEED_C1), ("C3", "kappa1 = 0, kappa2 = 1 (added)", c3, SEED_C3)):
        _, _, _, _, D, se, g, ne, cv, dt = c
        diff = abs(D - R12_D)
        tol_comb = 3 * math.sqrt(se * se + R12_D_SE * R12_D_SE)
        ok = cv and diff <= tol_comb
        strict = diff <= 3 * se
        print(f"  {name}  {what}, sigma = {sig0}: D = {D:.5f} +- {se:.5f} (second order {g:.5f}); R12 got {R12_D} +- {R12_D_SE}")
        print(f"      |difference| = {diff:.5f}; allowed 3 sqrt(se^2 + 0.0002^2) = {tol_comb:.5f}   [{'ok' if ok else 'FAIL'}]"
              f"   (stricter reading, 3 se of this run alone = {3 * se:.5f}: {'also met' if strict else 'NOT met'})"
              f"   ({ne} evaluations, {'converged' if cv else 'NOT CONVERGED'}, {dt:.0f} s, seed {seed})")
        if not ok:
            fails.append(name)
        if name == "C1":
            D1, se1 = D, se
    _, _, _, _, D2, se2, g2, ne2, cv2, dt2 = c2
    tol2 = max(3 * abs(se2), 1e-6)
    ok2 = cv2 and abs(D2) <= tol2
    print(f"  C2  kappa1 = kappa2 = 0: D = {D2:.3e} +- {se2:.1e} (second order {g2:.1e}); allowed max(3 se, 1e-6) = {tol2:.1e}   [{'ok' if ok2 else 'FAIL'}]"
          f"   ({ne2} evaluations, {'converged' if cv2 else 'NOT CONVERGED'}, {dt2:.0f} s, seed {SEED_C2})")
    if not ok2:
        fails.append("C2")
    _, _, _, _, D4, se4, g4, ne4, cv4, dt4 = c4
    tol4 = max(0.05 * g4, 3 * se4)
    ok4 = cv4 and abs(D4 - g4) <= tol4
    print(f"  C4  kappa1 = kappa2 = 1, sigma = {SIGMA_C4} (added): D = {D4:.6f} +- {se4:.6f}; second order {g4:.6f}; ratio {D4 / g4:.3f}; "
          f"allowed max(5%, 3 se) = {tol4:.6f}   [{'ok' if ok4 else 'FAIL'}]   ({ne4} evaluations, {'converged' if cv4 else 'NOT CONVERGED'}, {dt4:.0f} s, seed {SEED_C4})")
    if not ok4:
        fails.append("C4")
    sys.stdout.flush()
    if fails:
        print(f"\nFAILED code checks {fails}; stopping before the data are read.")
        return 1

    # --- the data
    with open(dryad, newline="", encoding="latin-1") as f:
        header = next(csv.reader(f))
    rows_all = r12.read_rows(dryad)
    rows = [r for r in rows_all if Y0 <= int(r["Year"]) <= Y1]
    plots = sorted({r["Grid"] for r in rows if r["Grid"] not in ("NA", "")})
    named = [h for h in header if h != ""]
    extra = [h for h in named if h not in README_COLUMNS]
    absent = [h for h in README_COLUMNS if h not in named]
    print("\n1. What was read")
    print(f"  {Path(dryad).name}: {len(rows_all)} rows, years {min(int(r['Year']) for r in rows_all)}-{max(int(r['Year']) for r in rows_all)}; "
          f"used: the {len(rows)} rows of {Y0}-{Y1}, plots {plots}")
    print(f"  header: {len(header)} columns, {len(header) - len(named)} unnamed; named columns not in the README: {extra if extra else 'none'}; "
          f"README columns not in the file: {absent if absent else 'none'}")
    print(f"  columns the README describes as a tick burden per mouse: {PER_MOUSE_BURDEN_COLUMNS} (larvae; README p. 2, August trapping session)")
    mice = r12.column(rows, "Grid", "PL_grid")
    lpm = r12.column(rows, "Grid", "LPM")
    don = r12.column(rows, "Grid", "DON")
    nip = r12.column(rows, "Grid", "NIP")
    print(f"  {'plot':5s} {'PL_grid (mice): years':28s} {'n':>3s} {'zeros':>6s}   {'LPM (larvae per mouse): years':30s} {'n':>3s} {'zeros':>6s}   "
          f"{'DON: years':12s} {'n':>3s} {'zeros':>6s}   {'NIP: years':12s} {'n':>3s} {'zeros':>6s}")
    for p in plots:
        cells = []
        for d, wd in ((mice[p], 28), (lpm[p], 30), (don[p], 12), (nip[p], 12)):
            z = sorted(y for y, v in d.items() if v == 0)
            cells.append(f"{r12.span_text(sorted(d)):{wd}s} {len(d):3d} {(r12.span_text(z) if z else '-'):>6s}")
        print(f"  {p:5s} " + "   ".join(cells))
    for name, d in (("PL_grid", mice), ("LPM", lpm), ("DON", don), ("NIP", nip)):
        vals = [v for p in plots for v in d[p].values()]
        print(f"  {name:8s} {len(vals):3d} values in the window, {sum(1 for v in vals if v == 0)} of them zero; least positive {min(v for v in vals if v > 0):.4g}, median {float(np.median(vals)):.4g}, most {max(vals):.4g}")

    rng = np.random.default_rng(SEED_BOOT)
    print(f"\n2. Part 1: kappa2, the within-plot slope of ln(larvae per mouse, year t) on ln(mice, year t).  Bootstrap seed {SEED_BOOT}.")
    k2, k2_se, k2_n, okd = within_plot_slope(r12, "ln LPM (year t) on ln PL_grid (year t)", mice, lpm, 0, plots, rng)
    if not okd:
        fails.append("K1 on the data (kappa2)")
    below, above = max(0.0, -1.0 - k2), max(0.0, k2 - 0.0)
    k2_fail = k2 < -1 - 2 * k2_se or k2 > 0 + 2 * k2_se
    inside = -1 <= k2 <= 0
    verdict = "FAILS" if k2_fail else "HOLDS"
    print(f"  kappa2 = {k2:.4f}, standard error {k2_se:.4f}   (unrounded: {k2!r}, {k2_se!r})")
    print(f"  Expectation -1 <= kappa2 <= 0 (stated beforehand): allowed by the criterion {-1 - 2 * k2_se:.4f} to {2 * k2_se:.4f}.")
    print(f"  Expectation {verdict}: kappa2 is {'inside' if inside else 'outside'} [-1, 0]; distance below -1: {below / k2_se:.2f} se; distance above 0: {above / k2_se:.2f} se.")

    print("\n3. Part 2: kappa1")
    print("  The file has no per-mouse burden of nymphs (README pp. 1-4; header above).  kappa1 is NOT estimated; that is the result of part 2.")

    print("\n4. Part 3: reported, no hypothesis.  Within-plot slopes on ln(mice, year t); density of nymphs = DON; infected = DON x NIP.")
    inf = {p: {y: don[p][y] * nip[p][y] for y in don[p] if y in nip[p]} for p in plots}
    bN, bN_se, bN_n, okd = within_plot_slope(r12, "ln DON (year t+1) on ln PL_grid (year t)", mice, don, 1, plots, rng)
    if not okd:
        fails.append("K1 on the data (nymph density)")
    print(f"  slope for the density of nymphs = {bN:.4f}, standard error {bN_se:.4f}")
    bI, bI_se, bI_n, okd = within_plot_slope(r12, "ln (DON x NIP) (year t+1) on ln PL_grid (year t)", mice, inf, 1, plots, rng)
    if not okd:
        fails.append("K1 on the data (infected-nymph density)")
    print(f"  slope for the density of infected nymphs = {bI:.4f}, standard error {bI_se:.4f}")
    sys.stdout.flush()

    # --- part 4
    print("\n5. Part 4: the loss D of ln R0, recomputed.  Two-stage ring (mouse 1 step, tick 10/4.4 steps), one host series on both transfers, no memory.")
    print("   kappa1 on the transfer from tick to mouse; kappa2 on the transfer from mouse to tick.  D +- one standard error of the simulation; second-order value in brackets.")
    k2_vals = (("estimate - 2 se", k2 - 2 * k2_se), ("estimate", k2), ("estimate + 2 se", k2 + 2 * k2_se))
    cases = []
    for sigma, _ in SIGMAS:
        for name, kv in k2_vals:
            for k1 in KAPPA1:
                cases.append((f"{sigma}|{name}|{k1}", sigma, k1, kv, SEED_GRID + len(cases), second_order(w, sigma, k1, kv), M, n_steps, BURN))
    print(f"   {len(cases)} cases; seeds {SEED_GRID} to {SEED_GRID + len(cases) - 1}; {M} populations, {n_steps} steps, burn-in {BURN}")
    sys.stdout.flush()
    with mp.Pool(NPROC) as pool:
        res = pool.map(solve, cases, chunksize=1)
    by = {r[0]: r for r in res}
    not_conv = [r[0] for r in res if not r[8]]
    for sigma, what in SIGMAS:
        print(f"\n   sigma = {sigma}  ({what})")
        print(f"   {'kappa2':32s} " + " ".join(f"{('kappa1 = ' + format(k1, 'g')):^30s}" for k1 in KAPPA1))
        for name, kv in k2_vals:
            cells = []
            for k1 in KAPPA1:
                _, _, _, _, D, se, g0, ne, cv, dt = by[f"{sigma}|{name}|{k1}"]
                cells.append(f"{D:8.4f} +- {se:.4f} [{g0:.4f}]{'' if cv else ' NOT CONVERGED'}")
            print(f"   {(format(kv, '+.4f') + ' (' + name + ')'):32s} " + "   ".join(f"{c:^28s}" for c in cells))
    print("\n   every case, in the order run: label, kappa1, kappa2, D, se, second order, D / second order, evaluations, seconds, seed")
    for r, c in zip(res, cases):
        label, sigma, k1, kv, D, se, g0, ne, cv, dt = r
        ratio = f"{D / g0:.3f}" if g0 > 1e-12 else "   - "
        print(f"   sigma {sigma:<6} kappa1 {k1:<4} kappa2 {kv:+.4f}   D = {D:.5f} +- {se:.5f}   second order {g0:.5f}   ratio {ratio}   {ne} ev.   {dt:4.0f} s   seed {c[4]}{'' if cv else '   NOT CONVERGED'}")
    if not_conv:
        fails.append(f"secant not converged: {not_conv}")
    print("\n   For comparison: draft 7's entry for this row is 0.05 to 0.10 (second-order values, continuous and discrete time) at sigma = 0.67 with kappa1 = 1, kappa2 = 0")
    print(f"   and a memory of 0.30 per step; R12's exact value at sigma = 1.0304, kappa1 = 1, kappa2 = 0, no memory is {R12_D} +- {R12_D_SE} (reproduced in check C1 above: {D1:.4f} +- {se1:.4f}).")

    print(f"\nSummary: kappa2 = {k2:.3f} (se {k2_se:.3f}, {k2_n} pairs); expectation -1 <= kappa2 <= 0 {verdict}.  kappa1 not estimated (no nymphs-per-mouse column).")
    print(f"         slopes on ln mice (year t): density of nymphs (year t+1) {bN:.3f} (se {bN_se:.3f}, {bN_n} pairs); density of infected nymphs (year t+1) {bI:.3f} (se {bI_se:.3f}, {bI_n} pairs).")
    print(f"Finished {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}; wall time {time.time() - t_start:.0f} s.")
    print("CODE CHECKS K1 TO K3, C1 TO C4: ALL PASSED" if not fails else f"CODE CHECKS FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
