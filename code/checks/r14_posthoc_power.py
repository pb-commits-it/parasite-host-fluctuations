"""AFTER-THE-FACT analysis of run R14 (the comparison across NEON sites).  Not part of the protocol.

Run R14 found a rank correlation of +0.49 between the prevalence of B. burgdorferi in nymphs and the
variability of mice across six sites, where the protocol predicted a negative one.  The protocol said
how little six sites can decide, but it did not say how large the predicted effect is at these sites,
or what the test could have shown if the theory were exactly right.  This script computes both.  It
was written after the result was known, to judge the test and not to rescue the prediction; nothing
in it changes the outcome recorded in the R14 report ("not decided" for T1, "fails" for T2).

A.  What the theory predicts for these six sites.  The loss D of ln R0 for the two-stage Borrelia
    cycle of the manuscript (stays 1 and 2.273 steps, no host memory, which is what the data show),
    computed exactly by the method of run R10, at each site's sd of ln mouse index, for three
    assumptions about how strongly mouse numbers enter transmission (relative size kappa = 1, the
    manuscript's table; 0.5; 0.25).  A transfer of relative size kappa on a host of sd sigma is a
    transfer of relative size 1 on a host of sd kappa*sigma, so each case is one run.

B.  What the test could have shown.  Sites are given ln R0 = m + e, e Gaussian with sd tau standing
    for everything else that differs between sites, and ln R0* = ln R0 - D.  Prevalence in nymphs is
    p = 1 - exp(-f ln R0*) with f = 1/2 (for hosts of constant number the cycle satisfies
    sum over the two hosts of -ln(1 - p) = ln R0*; half is given to the tick).  m is set so that the
    mean prevalence is the observed one.  Tested nymphs are binomial with the numbers actually tested.
    For each scenario: the distribution of Spearman's rank correlation over 200,000 simulated sets of
    six sites; its mean; the chance of a value at least as positive as observed (an upper-tail
    probability, not a likelihood); the chance of the test reaching its own criterion for "supported".
    tau = 0.17 is what the observed spread of prevalence corresponds to if all of it were differences
    in R0; 0.1 and 0.3 bracket it.

    CORRECTION (14 November, after the twelfth referee report, point 1).  The registered criterion is
    a one-sided exact permutation p below 0.05.  With six untied ranks rho = 1 - S/35, S the sum of
    squared rank differences, and p < 0.05 exactly when S >= 64, that is rho <= -29/35 = -0.828571...
    (21 of the 720 orderings, 2.92%).  The first version of this script tested rho <= -0.829, which
    is past the attainable value -29/35 and keeps only the 12 orderings with S >= 66 (1.67%): it
    threw away 9 of the 21 orderings that count as support.  The cutoff is now -29/35, the count of
    orderings is checked by enumeration below, and the first output is kept as
    results/r14-posthoc-power_attempt1_20261114.txt.  The recorded outcome of R14 is not affected.

C.  The check T2 with sampling error allowed for.  For each site, the variance of ln(yearly
    prevalence) less the mean binomial sampling variance (1 - p)/(n p) of the years used, set to zero
    if negative, against the variance of ln mouse index over the same years.  This is the correction
    that was registered for the Cary holdout (R12b, B3) and was not registered for R14.

No pass or fail: this is a description of what the test was able to show.

Run: python3 -I checks/r14_posthoc_power.py   (about two minutes)
"""
from __future__ import annotations

import math
import pathlib
import re
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))
import r10_loss_at_large_noise as r10  # noqa: E402

SITES = ["BLAN", "HARV", "ORNL", "SCBI", "SERC", "TREE"]
OBS_RHO = 0.4857


def read_r14():
    """Per-site sigma, pooled prevalence and yearly positives/tested, from the saved R14 output."""
    text = (HERE.parent / "results" / "r14-neon-across-sites_20261114.txt").read_text().splitlines()
    sigma, nip, ntest, yearly, used_years, sd_mice = {}, {}, {}, {}, {}, {}
    years_header = None
    for line in text:
        t = line.split()
        if not t:
            continue
        if t[0] == "site" and "2014" in t and "total" in t and years_header is None:
            years_header = [int(y) for y in t[1:] if y.isdigit()]       # the first table (sensu lato) only
        if t[0] in SITES and years_header and len(t) >= len(years_header) + 3 and "/" in t[-2] and t[0] not in yearly:
            cells = t[1:1 + len(years_header)]
            yearly[t[0]] = {y: tuple(int(v) for v in c.split("/")) for y, c in zip(years_header, cells) if "/" in c}
            k, n = t[-2].split("/")
            ntest[t[0]] = int(n)
            nip[t[0]] = int(k) / int(n)
        m = re.match(r"\s*(\w{4})\s+years ([\d,]+): sd ln NIP ([\d.]+), sd ln index ([\d.]+)", line)
        if m and m.group(1) in SITES:
            used_years[m.group(1)] = [int(y) for y in m.group(2).split(",")]
            sd_mice[m.group(1)] = (float(m.group(3)), float(m.group(4)))
        m = re.match(r"\s*sites: (.*)", line)
        if m and "sigma" in line:
            for part in m.group(1).split("), "):
                mm = re.match(r"(\w{4}) \(sigma ([\d.]+), NIP ([\d.]+)", part)
                if mm:
                    sigma[mm.group(1)] = float(mm.group(2))
    return sigma, nip, ntest, yearly, used_years, sd_mice


def spearman(x, y):
    rx = np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)
    ry = np.argsort(np.argsort(y, axis=-1), axis=-1).astype(float)
    rx -= rx.mean(axis=-1, keepdims=True)
    ry -= ry.mean(axis=-1, keepdims=True)
    return (rx * ry).sum(axis=-1) / np.sqrt((rx ** 2).sum(axis=-1) * (ry ** 2).sum(axis=-1))


CUT = -29.0 / 35.0          # the registered criterion: S >= 64, one-sided exact p = 21/720 < 0.05


def check_cutoff():
    import itertools
    base = np.arange(6.0)
    S = np.array([sum((a - b) ** 2 for a, b in zip(base, perm)) for perm in itertools.permutations(range(6))])
    rho = 1 - S / 35.0
    n_reg = int(np.sum(rho <= CUT + 1e-9))
    n_old = int(np.sum(rho <= -0.829 + 1e-9))
    n_next = int(np.sum(S >= 62))
    print(f"cutoff check by enumeration of the 720 orderings: rho <= -29/35 in {n_reg} (p = {n_reg / 720:.4f}); "
          f"the first version's rho <= -0.829 kept {n_old} (p = {n_old / 720:.4f}); the next attainable value, S >= 62, has {n_next} (p = {n_next / 720:.4f}, above 0.05)")
    assert n_reg == 21 and n_old == 12 and n_next / 720 > 0.05


def main():
    check_cutoff()
    sigma, nip, ntest, yearly, used_years, sd_mice = read_r14()
    assert set(sigma) == set(SITES) and set(yearly) == set(SITES), (sigma, list(yearly))
    sig = np.array([sigma[s] for s in SITES])
    p_obs = np.array([nip[s] for s in SITES])
    n_obs = np.array([ntest[s] for s in SITES])
    check = float(spearman(sig[None, :], p_obs[None, :])[0])
    print(f"read from the R14 output: sigma {dict(zip(SITES, sig))}")
    print(f"                          prevalence {dict(zip(SITES, np.round(p_obs, 4)))}; Spearman rho recomputed {check:+.4f} (report: +0.4857)")

    # ---------------------------------------------------------------- A
    print("\nA. The loss D of ln R0 that the theory gives each site (exact, discrete Borrelia cycle, no memory)")
    r10.M, r10.T, r10.BURN = 200, 100_000, 2_000
    D = {}
    for kappa in (1.0, 0.5, 0.25):
        row = []
        for i, s in enumerate(SITES):
            lab, stays, sg, ph, d, se, g0, dt = r10.solve_case((f"{s} {kappa}", [1.0, 10 / 4.4], [kappa * sig[i], 0.0], [0.0, 0.0], 900 + i + int(100 * kappa)))
            row.append(d)
        D[kappa] = np.array(row)
        spread = D[kappa].max() - np.sort(D[kappa])[-2]
        print(f"   relative size {kappa:4.2f}: " + "  ".join(f"{s} {d:.4f}" for s, d in zip(SITES, row))
              + f"   (HARV exceeds the next site by {spread:.3f}; range among the other five {np.ptp(np.delete(D[kappa], 1)):.3f})")
    D[0.0] = np.zeros(6)

    # ---------------------------------------------------------------- B
    print("\nB. What a rank correlation across these six sites could have shown")
    print("   scenario                                  mean rho   P(rho >= +0.4857)   P(rho <= -29/35)   ratio of the upper tails, scenario / no effect")
    rng = np.random.default_rng(20261114)
    f = 0.5
    nsim = 200_000
    base = {}
    for tau in (0.10, 0.17, 0.30):
        for kappa in (0.0, 0.25, 0.5, 1.0):
            # set m so that the mean prevalence without site noise equals the observed mean
            target = float(p_obs.mean())
            lo, hi = 0.0, 5.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                pm = float(np.mean(1 - np.exp(-f * np.maximum(mid - D[kappa], 0))))
                lo, hi = (mid, hi) if pm < target else (lo, mid)
            m = 0.5 * (lo + hi)
            lnR = m + tau * rng.standard_normal((nsim, 6)) - D[kappa]
            p = 1 - np.exp(-f * np.maximum(lnR, 0))
            k = rng.binomial(n_obs, p)
            rho = spearman(np.broadcast_to(sig, k.shape), k / n_obs + 1e-9 * rng.standard_normal(k.shape))
            pr_obs = float(np.mean(rho >= OBS_RHO - 1e-9))
            pr_sup = float(np.mean(rho <= CUT + 1e-9))
            if kappa == 0.0:
                base[tau] = pr_obs
            name = "no effect of host variability" if kappa == 0 else f"theory, relative size {kappa}"
            print(f"   other differences sd {tau:.2f}; {name:30s} {rho.mean():+8.3f}   {pr_obs:14.3f}   {pr_sup:16.3f}   {pr_obs / base[tau]:14.2f}")
    print("   (m, the mean ln R0, came out between 0.4 and 0.6 in every scenario; published ln R0 for this parasite is 0.17 to 0.56)")

    # ---------------------------------------------------------------- C
    print("\nC. T2 with the sampling error of yearly prevalence allowed for")
    print("   site   years   sd ln prevalence   sampling sd   corrected sd   sd ln mouse index   prevalence varies less (raw / corrected)")
    for s in SITES:
        yrs = used_years[s]
        vals = [yearly[s][y] for y in yrs]
        lp = np.array([math.log(k / n) for k, n in vals])
        samp = float(np.mean([(1 - k / n) / k for k, n in vals]))          # (1 - p)/(n p) with p = k/n
        raw = float(lp.var(ddof=1))
        corr = max(raw - samp, 0.0)
        sd_nip_report, sd_mouse = sd_mice[s]
        print(f"   {s}   {len(yrs):>3}      {math.sqrt(raw):.3f} (report {sd_nip_report:.3f})    {math.sqrt(samp):.3f}        {math.sqrt(corr):.3f}          {sd_mouse:.3f}"
              f"            {'yes' if math.sqrt(raw) < sd_mouse else 'no'} / {'yes' if math.sqrt(corr) < sd_mouse else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
