"""Is the shortfall of secondary's no-staying ring (U5) from sigma^2/2 an effect of run length?

Claim checked: with no staying anywhere, a ring of n stages is n separate cohorts that never mix. Each
cohort's log size is a random walk with drift (mean ln g) - sigma^2/2, so the long-run cost is exactly
sigma^2/2. A finite run measures the growth of the SUM of the cohorts, which follows the luckiest one, so
the measured cost falls short of sigma^2/2 by an amount that shrinks as the run lengthens.

secondary's matrix simulation (`verification/check_independent_derivation.py`, ring U5: 3 stages, sigma = 0.2, 150,000
steps after 5,000 of burn-in, 400 paired replicates) gave 0.019643 +/- 0.000011 at phi = 0 and
0.019921 +/- 0.000007 at phi = 0.9, against sigma^2/2 = 0.02. This script simulates the three cohorts
directly, with no matrix, at the same run length, burn-in, sigma and pairing. Transmission is set equal in
all stages and the cohorts start equal, which changes the result by about 1e-5.

Result on 2026-11-13: 0.019617 +/- 0.000013 and 0.019905 +/- 0.000003. Same shortfall, so it is the run
length and not an error in either code. Takes about 10 s.
"""
import math, sys
import numpy as np
n, sigma, steps, burn, reps = 3, 0.2, 150000, 5000, 400
for phi in (0.0, 0.9):
    rng = np.random.default_rng(1234 + int(10 * phi))
    innov = sigma * math.sqrt(1 - phi * phi)
    z = rng.normal(0.0, sigma, size=(reps, n))          # one AR(1) series per stage
    Lp = np.zeros((reps, n)); Lm = np.zeros((reps, n))   # log size of cohort j, for +z and for -z
    idx = np.arange(n)
    start = None
    for t in range(steps + burn):
        if t == burn:
            start = (np.logaddexp.reduce(Lp, axis=1), np.logaddexp.reduce(Lm, axis=1))
        stage = (idx + t) % n                             # stage occupied by cohort j at time t
        Lp += z[:, stage] - sigma ** 2 / 2
        Lm += -z[:, stage] - sigma ** 2 / 2
        z = phi * z + innov * rng.normal(size=(reps, n))
    gp = (np.logaddexp.reduce(Lp, axis=1) - start[0]) / steps
    gm = (np.logaddexp.reduce(Lm, axis=1) - start[1]) / steps
    cost = -(gp + gm) / 2                                 # ln lambda0 = 0 here
    print(f"phi = {phi}: cohort simulation cost = {cost.mean():.6f} +/- {cost.std(ddof=1) / math.sqrt(reps):.6f}; "
          f"exact long-run value {sigma ** 2 / 2:.6f}")
