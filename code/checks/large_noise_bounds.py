"""Checks of notes/large-noise.md, Proposition 21 (bounds on the growth rate of a ring at any noise level).

Claims and tolerances, fixed before the first run:
  B1  (item 4 of section 3) For primitive rings with stays tau_k >= 1 of a line of descent and
      0 < phi < 1, the second-order coefficient of the loss per generation for one fluctuating
      host, (1 + 2 theta)/(2 T) - R/(1 - R), is at most 1/2.  200,000 random rings of 1 to 8
      stages.  No exception allowed (slack 1e-12).
  B2  (Proposition 21(a) and (c)) For 400 random rings of 1 to 4 stages away from their threshold,
      with every transfer fluctuating, the simulated long-run growth rate r satisfies
          r >= max( ln lambda - sum l_k / T ,  (ln R0 - sum l_k) / T0 ,  ln max s_k )
      within 3 standard errors.  Half the cases use lognormal multipliers with sigma between 0.3
      and 3 and autocorrelation between -0.9 and 0.99; half use two-state Markov multipliers
      with mean one, a low state between 0.001 and 0.5 times the mean, and persistence between
      0.05 and 0.98.  l_k = -E ln h_k is computed exactly for each.
  B3  (Proposition 21(b)) In the 100 or so of those cases whose multipliers are independent from
      step to step, r <= ln lambda within 3 standard errors.
  Reported, no criterion: the smallest margin in B2 in units of standard errors, and how close
  the bound comes (the largest ratio of the bound's loss to the realised loss).

Run: python3 -I checks/large_noise_bounds.py   (several minutes on six processes)
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys

import numpy as np

M, T, BURN = 60, 100_000, 2_000


def b1():
    rng = np.random.default_rng(1)
    worst = -math.inf
    for _ in range(200_000):
        n = int(rng.integers(1, 9))
        tau = 1 + np.exp(rng.uniform(-6, 5, size=n)) * (rng.random(n) < 0.8)
        if n >= 2 and np.all(tau == 1):
            tau[0] = 1 + math.exp(rng.uniform(-6, 5))
        phi = rng.uniform(1e-4, 1 - 1e-4)
        th = phi / (1 - phi)
        Tt = tau.sum()
        logP = float(np.sum(np.log1p(tau / th)))
        c = (1 + 2 * th) / (2 * Tt) - 1.0 / math.expm1(logP)
        worst = max(worst, c)
    ok = worst <= 0.5 + 1e-12
    print(f"[{'PASS' if ok else 'FAIL'}] B1 the second-order coefficient is at most 1/2: 200,000 rings, largest value {worst:.9f}")
    return ok


def one_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    s = rng.uniform(0.0, 0.97, size=n) * (rng.random(n) < 0.8)
    if n >= 2 and np.all(s == 0):
        s[0] = rng.uniform(0.2, 0.9)
    g = np.exp(rng.uniform(-2.0, 1.5, size=n))
    lognormal = seed % 2 == 0
    # the mean model
    lo, hi = float(s.max()) + 1e-12, float(s.max() + g.max() + 5)
    f = lambda lam: np.sum(np.log(lam - s)) - np.sum(np.log(g))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    Tgen = float(np.sum(1 / (1 - s / lam)))
    T0 = float(np.sum(1 / (1 - s)))
    lnR0 = float(np.sum(np.log(g / (1 - s))))
    X = np.full((M, n), 1.0 / n)
    acc = np.zeros(M)
    if lognormal:
        sig = rng.uniform(0.3, 3.0, size=n)
        phi = rng.uniform(-0.9, 0.99) if rng.random() < 0.75 else 0.0
        ell = float(np.sum(sig ** 2 / 2))
        z = rng.standard_normal((M, n))
        independent = phi == 0.0
        for t in range(BURN + T):
            z = phi * z + math.sqrt(1 - phi * phi) * rng.standard_normal((M, n))
            h = np.exp(sig * z - sig ** 2 / 2)
            Xn = s * X + np.roll(g * h * X, 1, axis=1)
            tot = Xn.sum(axis=1)
            if t >= BURN:
                acc += np.log(tot)
            X = Xn / tot[:, None]
        kind = f"lognormal, sigma up to {sig.max():.2f}, phi {phi:+.2f}"
    else:
        low = np.exp(rng.uniform(math.log(0.001), math.log(0.5), size=n))
        pi_low = rng.uniform(0.1, 0.9, size=n)
        high = (1 - pi_low * low) / (1 - pi_low)              # mean one
        stay = rng.uniform(0.05, 0.98) if rng.random() < 0.75 else None
        ell = float(np.sum(-(pi_low * np.log(low) + (1 - pi_low) * np.log(high))))
        state = rng.random((M, n)) < pi_low                   # True = low
        independent = stay is None
        for t in range(BURN + T):
            if stay is None:
                state = rng.random((M, n)) < pi_low
            else:
                redraw = rng.random((M, n)) > stay
                state = np.where(redraw, rng.random((M, n)) < pi_low, state)
            h = np.where(state, low, high)
            Xn = s * X + np.roll(g * h * X, 1, axis=1)
            tot = Xn.sum(axis=1)
            if t >= BURN:
                acc += np.log(tot)
            X = Xn / tot[:, None]
        kind = f"two-state, lowest state {low.min():.3f}, persistence {stay if stay is not None else 0}"
    r = acc / T
    r_mean, se = float(r.mean()), float(r.std(ddof=1) / math.sqrt(M))
    lower = max(math.log(lam) - ell / Tgen, (lnR0 - ell) / T0, math.log(s.max()) if s.max() > 0 else -math.inf)
    return dict(seed=seed, n=n, kind=kind, r=r_mean, se=se, lower=lower, lnlam=math.log(lam), independent=independent,
                tight=(math.log(lam) - r_mean) / (math.log(lam) - lower) if math.log(lam) > lower else float("nan"))


def main():
    ok1 = b1()
    with mp.Pool(6) as pool:
        res = pool.map(one_case, range(1000, 1400), chunksize=4)
    margins = [(d["r"] - d["lower"]) / max(d["se"], 1e-12) for d in res]
    bad = [d for d, m in zip(res, margins) if m < -3]
    i = int(np.argmin(margins))
    ok2 = not bad
    print(f"[{'PASS' if ok2 else 'FAIL'}] B2 lower bounds on growth: 400 rings, violations {len(bad)}; smallest margin {margins[i]:+.1f} standard errors "
          f"(seed {res[i]['seed']}, {res[i]['n']} stages, {res[i]['kind']}: r = {res[i]['r']:.5f} +- {res[i]['se']:.5f}, bound {res[i]['lower']:.5f})")
    for d in bad[:5]:
        print("      violation:", d)
    ind = [d for d in res if d["independent"]]
    bad3 = [d for d in ind if d["r"] > d["lnlam"] + 3 * d["se"]]
    ok3 = not bad3
    print(f"[{'PASS' if ok3 else 'FAIL'}] B3 upper bound without memory: {len(ind)} rings, violations {len(bad3)}")
    tights = np.array([d["tight"] for d in res if np.isfinite(d["tight"])])
    print(f"      realised loss as a fraction of the largest loss the bounds allow: median {np.median(tights):.2f}, largest {tights.max():.3f}")
    n_mem = sum(1 for d in res if (not d["independent"]) and d["r"] > d["lnlam"] + 3 * d["se"])
    print(f"      (with memory, growth exceeded ln lambda by more than 3 standard errors in {n_mem} of {len(res) - len(ind)} rings; Proposition 21(b) does not apply to them)")
    good = ok1 and ok2 and ok3
    print("\nALL CHECKS PASSED" if good else "\nFAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
