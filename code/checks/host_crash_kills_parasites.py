"""When a host population is hit by a bad year, the parasites inside the hosts that die are lost with
them.  What does that do to the cost of that host's fluctuation?  (Limits of draft 7: "crashes of a
host population that kill the parasites inside the hosts that die".)

Model.  A two-stage ring.  Stage 0 is in a steady host (or free-living), stay probability s0, transfer
g0 into stage 1.  Stage 1 is in a host whose log abundance is sigma*xi(t),
    xi(t+1) = phi xi(t) + sqrt(1 - phi^2) eps(t+1),
eps a standard Gaussian shock to the host (a negative shock is a bad year: hosts die).  In step t:
  - the transfer INTO stage 1 (out of stage 0) is multiplied by exp(sigma xi(t) - sigma^2/2)
    (more hosts, more transmission; relative size 1);
  - the staying in stage 1 is multiplied by exp(kappa sigma sqrt(1 - phi^2) eps(t+1) - v/2), v the
    variance of the exponent: the parasites inside share the host's shock, to the power kappa
    (kappa = 0: untouched; kappa = 1: they die in the proportion the host population does).  The
    host's later recovery, which is by births of uninfected hosts, does not bring them back; that is
    why the staying follows the shock and not the host's abundance.
Both multipliers have mean one, so the mean model is the same for every kappa: what is compared is
the cost of the fluctuation at fixed mean rates, the convention of the manuscript.

The second-order cost is C'' sigma^2 with C'' from Lemma 6'' of notes/error-bound-general.md,
section 13: two fluctuating entries, the transfer (driver xi(t)) and the staying (driver eps(t+1),
relative size kappa sqrt(1 - phi^2)).  The two drivers are correlated at positive lags only:
E[xi(t) eps(t - l + 1)] = sqrt(1 - phi^2) phi^(l - 1) for l >= 1.

History.  Two earlier versions of this script are kept as results/host-crash-kills-parasites_attempt1
and _attempt2_20261114.txt.  They let the parasites inside follow every change of host numbers, up
as well as down; the survival over a stay then telescopes while each step is separately normalised,
and the "cost" that came out was that normalisation and not biology.  Their numbers are not used.

Claims and tolerances:
  C1  With kappa = 0 the coefficient equals the closed form of the manuscript for one fluctuating
      transfer, (1/T)[(1 + 2 theta)/(2T) - R/(1 - R)] per unit sigma^2 per step, to 1e-10 relative
      (6 rings).  This is the known result the code must reproduce.
  C2  For four rings with kappa = 1 the coefficient agrees with simulation at sigma = 0.075 within the
      larger of 3% and 3 standard errors; and where the shortfalls RELATIVE to C'' sigma^2 at
      sigma = 0.075 and 0.15 both exceed 3 standard errors, their ratio lies between 2 and 8 (a term
      of order sigma^4 gives 4).
  C3  With no transfer fluctuation the staying term alone is e_s^2 kappa^2 (1 - phi^2)/2 per unit
      sigma^2 per step, e_s = (stay1 - 1)/T the elasticity of growth to the staying (the published
      result for uncorrelated fluctuation of one entry), to 1e-10 relative.  Tested by computing the
      (staying, staying) term of C'' separately.
  Reported, no criterion: the loss per generation in units of sigma^2 with kappa = 0, 0.5 and 1,
  for stays in the fluctuating host of 2, 5, 20 and 100 steps (stay 5 in the steady stage) and
  phi = 0, 0.5, 0.9, 0.99; and how much of it is the cross term.

Run: python3 -I checks/host_crash_kills_parasites.py   (a few minutes)
"""
from __future__ import annotations

import math
import sys

import numpy as np

FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def setup(stay0, stay1, growth=1.005):
    """A two-stage ring whose line of descent has stays stay0 and stay1 and whose mean growth factor is `growth`."""
    lam = growth
    s = np.array([lam * (1 - 1 / stay0), lam * (1 - 1 / stay1)])
    g = np.array([lam / stay0, lam / stay1])
    assert np.all(s < 1)
    A = np.array([[s[0], g[1]], [g[0], s[1]]])
    return A, s, g, lam


def perron(A):
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(vals.real))
    v = np.abs(vecs[:, i].real)
    vl, wl = np.linalg.eig(A.T)
    j = int(np.argmax(vl.real))
    w = np.abs(wl[:, j].real)
    v /= v.sum()
    w /= w @ v
    return float(vals[i].real), v, w


def rho(phi, e, e2, l):
    """E[driver_e(t) driver_e2(t - l)], l >= 0; driver 0 is xi(t), driver 1 is eps(t + 1)."""
    if e == 0 and e2 == 0:
        return phi ** l
    if e == 1 and e2 == 1:
        return 1.0 if l == 0 else 0.0
    if e == 0 and e2 == 1:                        # xi(t) with eps(t - l + 1)
        return 0.0 if l == 0 else math.sqrt(1 - phi * phi) * phi ** (l - 1)
    return 0.0                                    # eps(t + 1) with xi(t - l): the shock is in the future


def terms(A, kappa, phi):
    """C'' split into (transfer, transfer), (staying, staying) and the two cross terms."""
    lam, v, w = perron(A)
    n = 2
    M = A / lam
    Q = np.eye(n) - np.outer(v, w)
    T = M @ Q
    Bt = np.zeros((n, n)); Bt[1, 0] = A[1, 0] / lam        # the transfer into stage 1
    Bs = np.zeros((n, n)); Bs[1, 1] = A[1, 1] / lam        # the staying in stage 1
    B = [Bt, Bs]
    k = [1.0, kappa * math.sqrt(1 - phi * phi)]
    c = [float(w @ Be @ v) for Be in B]
    out = np.zeros((2, 2))
    for e in range(2):
        for e2 in range(2):
            acc = 0.0
            Tm = np.eye(n)
            for m in range(6000):
                r = rho(phi, e, e2, m + 1)
                if r != 0.0:
                    acc += r * float(w @ B[e] @ Tm @ Q @ B[e2] @ v)
                Tm = Tm @ T
            out[e, e2] = k[e] * k[e2] * (c[e] * c[e2] * rho(phi, e, e2, 0) / 2 - acc)
    return out, lam, c


def coefficient(A, kappa, phi):
    out, lam, _ = terms(A, kappa, phi)
    return float(out.sum()), lam


def closed_form_transfer(stay0, stay1, phi):
    T = stay0 + stay1
    if phi == 0:
        return 1 / (2 * T * T)
    th = phi / (1 - phi)
    R = 1 / ((1 + stay0 / th) * (1 + stay1 / th))
    return ((1 + 2 * th) / (2 * T) - R / (1 - R)) / T


def simulate(A, kappa, phi, sigma, seed, M=400, steps=200_000, burn=2000):
    rng = np.random.default_rng(seed)
    s0, s1, g0, g1 = A[0, 0], A[1, 1], A[1, 0], A[0, 1]
    xi = rng.standard_normal(M)
    X = np.full((M, 2), 0.5)
    acc = np.zeros(M)
    sd_s = kappa * sigma * math.sqrt(1 - phi * phi)
    for t in range(burn + steps):
        eps = rng.standard_normal(M)
        h_t = np.exp(sigma * xi - sigma * sigma / 2)
        h_s = np.exp(sd_s * eps - sd_s * sd_s / 2)
        x0 = s0 * X[:, 0] + g1 * X[:, 1]
        x1 = g0 * h_t * X[:, 0] + s1 * h_s * X[:, 1]
        tot = x0 + x1
        if t >= burn:
            acc += np.log(tot)
        X = np.stack([x0 / tot, x1 / tot], axis=1)
        xi = phi * xi + math.sqrt(1 - phi * phi) * eps
    r = acc / steps
    return float(r.mean()), float(r.std(ddof=1) / math.sqrt(M))


def main():
    worst = 0.0
    for stay0, stay1, phi in ((5, 2, 0.0), (5, 2, 0.5), (5, 20, 0.9), (3, 8, 0.3), (10, 10, 0.9), (2, 2, 0.7)):
        A, s, g, lam = setup(stay0, stay1)
        C, _ = coefficient(A, 0.0, phi)
        want = closed_form_transfer(stay0, stay1, phi)
        worst = max(worst, abs(C - want) / want)
    report("C1 hosts' deaths do not touch the parasites: the manuscript's closed form", worst < 1e-10, f"6 rings, largest relative difference {worst:.1e}")

    worst = 0.0
    for stay0, stay1, phi in ((5, 2, 0.0), (5, 20, 0.5), (5, 20, 0.9), (3, 8, 0.3)):
        A, s, g, lam = setup(stay0, stay1)
        out, _, c = terms(A, 1.0, phi)
        e_s = (stay1 - 1) / (stay0 + stay1)
        want = e_s ** 2 * (1 - phi * phi) / 2
        worst = max(worst, abs(out[1, 1] - want) / want, abs(c[1] - e_s) / e_s)
    report("C3 the staying term alone", worst < 1e-10, f"4 rings, largest relative difference {worst:.1e}")

    ok_all = True
    for idx, (stay0, stay1, phi) in enumerate(((5, 2, 0.5), (5, 20, 0.5), (5, 5, 0.9), (5, 5, 0.0))):
        A, s, g, lam = setup(stay0, stay1)
        C, _ = coefficient(A, 1.0, phi)
        rel = {}
        for sigma in (0.15, 0.075):
            growth, se = simulate(A, 1.0, phi, sigma, seed=700 + idx + int(1000 * sigma))
            cost = math.log(lam) - growth
            rel[sigma] = (cost / (C * sigma * sigma) - 1, se / (C * sigma * sigma))
        d1, se1 = rel[0.075]
        d2, se2 = rel[0.15]
        ok = abs(d1) <= max(0.03, 3 * se1)
        scaling = ""
        if abs(d1) > 3 * se1 and abs(d2) > 3 * se2:
            ratio = d2 / d1
            ok = ok and 2 <= ratio <= 8
            scaling = f"; ratio of relative shortfalls {ratio:.1f}"
        ok_all &= ok
        print(f"      stays ({stay0}, {stay1}), phi {phi}: cost / (C'' sigma^2) - 1 = {d1:+.4f} +- {se1:.4f} at sigma 0.075, "
              f"{d2:+.4f} +- {se2:.4f} at 0.15{scaling} [{'ok' if ok else 'FAIL'}]")
    report("C2 the coefficient with the hosts' deaths, against simulation", ok_all, "4 rings at sigma = 0.075 and 0.15")

    print("\n      loss per generation, in units of sigma^2 (a cycle with no staying loses 0.5)")
    print("      stay in the steady stage 5; rows: stay in the fluctuating host; each cell: kappa = 0 / 0.5 / 1 (share of the kappa = 1 value that is the cross term)")
    print("                      phi = 0                      phi = 0.5                    phi = 0.9                    phi = 0.99")
    for stay1 in (2, 5, 20, 100):
        row = []
        for phi in (0.0, 0.5, 0.9, 0.99):
            A, s, g, lam = setup(5, stay1)
            T = 5 + stay1
            vals = [T * coefficient(A, kap, phi)[0] for kap in (0.0, 0.5, 1.0)]
            out, _, _ = terms(A, 1.0, phi)
            cross = T * (out[0, 1] + out[1, 0])
            row.append(f"{vals[0]:.3f} / {vals[1]:.3f} / {vals[2]:.3f} ({cross / vals[2]:+.2f})")
        print(f"        stay {stay1:>3}:  " + "   ".join(row))
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
