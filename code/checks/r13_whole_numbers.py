"""Run R13 (protocols/R13-whole-numbers.md): extinction of a ring life cycle counted in whole numbers.

Each step: stayers ~ Binomial(X_k, s_k); the leavers of stage k produce Poisson(b_k h_k(t) * leavers)
individuals of stage k + 1.  Claims and criteria are in the protocol: W0 is a check (extinct fraction
>= 0.99 below the threshold R0*, <= 0.90 above it with >= 0.10 escaped); W1 is open.

The first run stopped following a replicate once its total passed 10^6 and called it "escaped".  Its
check W0 failed (8.8% of replicates below the threshold reached 10^6 before dying), which showed that
rule to be unsafe; output kept in results/r13-whole-numbers_attempt1_20261114.txt.  Replicates are now
followed to the end of the horizon; "alive" means not extinct at the end (or above 10^15, where the
numbers no longer fit the sampler; with a positive growth rate the chance of dying from there is
negligible, and below the threshold no replicate gets there).

Run: python3 -I checks/r13_whole_numbers.py   (tens of minutes on six processes)
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import time

import numpy as np

CAP = 10**15      # a replicate is followed with exact whole-number sampling until its total passes this
REPS = 2000


def rates_for(stays, ln_r0):
    """s_k and lambda such that the stays of a line of descent are `stays` and prod g/(1-s) = exp(ln_r0)."""
    tau = np.asarray(stays, float)

    def r0_of(lam):
        s = lam * (1 - 1 / tau)
        if np.any(s >= 1):
            return math.inf
        return float(np.prod((lam / tau) / (1 - s)))
    lo, hi = 1e-9, 1.0
    while r0_of(hi) < math.exp(ln_r0):
        lo, hi = hi, hi * 1.0005 if hi >= 1 else 1.0
        if hi > 2:
            raise RuntimeError("no solution")
    lo = min(lo, 1.0) if r0_of(1.0) >= math.exp(ln_r0) else lo
    lo = 1e-9 if r0_of(lo) > math.exp(ln_r0) else lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if r0_of(mid) < math.exp(ln_r0):
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    s = lam * (1 - 1 / tau)
    g = lam / tau
    return s, g, lam


def run_case(case):
    label, s, g, sigma, phi, n0, steps, seed = case
    t0 = time.time()
    rng = np.random.default_rng(seed)
    s = np.asarray(s, float)
    g = np.asarray(g, float)
    n = len(s)
    b = g / (1 - s)                                  # offspring per leaver
    # stable stage distribution of the mean matrix
    A = np.diag(s)
    for k in range(n):
        A[(k + 1) % n, k] += g[k]
    vals, vecs = np.linalg.eig(A)
    v = np.abs(vecs[:, int(np.argmax(vals.real))].real)
    v = v / v.sum()
    X = rng.multinomial(n0, v, size=REPS).astype(np.int64)
    z = rng.standard_normal(REPS)
    innov = math.sqrt(1 - phi * phi)
    alive = np.arange(REPS)
    extinct_at = np.full(REPS, -1, dtype=np.int64)
    escaped = np.zeros(REPS, dtype=bool)
    for t in range(steps):
        if alive.size == 0:
            break
        z = phi * z + innov * rng.standard_normal(alive.size)
        h0 = np.exp(sigma * z - sigma * sigma / 2)   # the transfer out of stage 0 fluctuates
        stay = rng.binomial(X, s)
        leave = X - stay
        mean = leave * b
        mean[:, 0] = mean[:, 0] * h0
        born = rng.poisson(mean)
        X = stay + np.roll(born, 1, axis=1)
        tot = X.sum(axis=1)
        dead = tot == 0
        out = tot > CAP
        if dead.any() or out.any():
            extinct_at[alive[dead]] = t + 1
            escaped[alive[out]] = True
            keep = ~(dead | out)
            alive, X, z = alive[keep], X[keep], z[keep]
    n_ext = int(np.sum(extinct_at >= 0))
    n_esc = int(escaped.sum())
    med = float(np.median(extinct_at[extinct_at >= 0])) if n_ext else float("nan")
    return label, n_ext, n_esc, REPS - n_ext - n_esc, med, time.time() - t0


def build():
    cases = []
    seed = 20261114
    # W0
    D = 0.0526
    for tag, lr in (("below", D - 0.03), ("above", D + 0.03)):
        stays0 = [1.0, 7.5]
        # at the threshold of the MEAN model the stays (1, 7.5) are those of run R10; keep its s and scale g
        s = 1 - 1 / np.asarray(stays0)
        g = (1 / np.asarray(stays0)) * math.exp(lr / 2)
        seed += 1
        cases.append((f"W0 {tag} the threshold (ln R0 = D {'-' if tag == 'below' else '+'} 0.03)", s, g, 0.6, 0.75, 200, int(3000 * 8.5), seed))
    sigma = 0.15
    for name, phi, lr, bs in (("N1", 0.9, 0.2 * sigma ** 2, (None, 10, 40, 160)), ("N2", 0.0, 0.06 * sigma ** 2, (None, 10))):
        for bstay in bs:
            stays = [10.0] if bstay is None else [10.0, float(bstay)]
            s, g, lam = rates_for(stays, lr)
            if len(stays) == 2:                      # split the multiplication evenly between the two transfers
                root = math.exp(lr / 2)
                g = (1 - s) * root
            for n0 in (100, 1000, 10000):
                seed += 1
                cases.append((f"W1 {name} steady stay {bstay if bstay else 'none':>4}, start {n0:>5}", s, g, sigma, phi, n0, int(6000 * sum(stays)), seed))
    return cases


def main():
    t0 = time.time()
    cases = build()
    print(f"R13: {len(cases)} cases, {REPS} replicates each, every replicate followed to the end of its horizon (or to {CAP:.0e} individuals)")
    with mp.Pool(6) as pool:
        res = pool.map(run_case, cases, chunksize=1)
    fails = []
    for label, n_ext, n_esc, n_open, med, dt in res:
        p = n_ext / REPS
        se = math.sqrt(max(p * (1 - p), 1e-12) / REPS)
        print(f"  {label:48s} extinct {p:.4f} +- {se:.4f}   above 1e15 {n_esc / REPS:.4f}   alive at the end {n_open / REPS:.4f}   "
              f"median time to extinction {med:9.0f} steps   [{dt:.0f} s]")
    by = {r[0]: r for r in res}
    below = [r for r in res if r[0].startswith("W0 below")][0]
    above = [r for r in res if r[0].startswith("W0 above")][0]
    not_extinct_above = (above[2] + above[3]) / REPS
    ok = below[1] / REPS >= 0.99 and above[1] / REPS <= 0.90 and not_extinct_above >= 0.10
    print(f"[{'PASS' if ok else 'FAIL'}] W0 the threshold is where R0* says: extinct {below[1] / REPS:.4f} below, {above[1] / REPS:.4f} above (not extinct {not_extinct_above:.4f})")
    print(f"wall time {time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
