"""An exactly solvable case of the parking lot, at any noise level.

Two stages.  Stage 0 is met through a fluctuating host and lasts exactly one step (no staying).  Stage 1
is steady and memoryless: a fraction s stays each step.  The host's relative abundance h(t) is
independent from step to step and gamma-distributed with mean one and shape p (variance 1/p):

    x0(t+1) = g1 x1(t),        x1(t+1) = s x1(t) + g0 h(t) x0(t).

Then x1(t+1) = s x1(t) + c h(t) x1(t-1), c = g0 g1, and with u(t) = x1(t+1)/x1(t) and
X(t) = c h(t)/u(t-1) one has u = s + X and the random continued fraction

    X' = Y/(s + X),      Y = c h ~ Gamma(shape p, scale c/p), independent of X.

Claim 1 (derived in notes/exact-gamma-parking.md).  The stationary density of X is
    f(x) proportional to x^(p-1) (s + x)^(-p) exp(-s p x / c),   x > 0.
Claim 2.  The growth rate is r = E ln(s + X) = ln s + (d/db) ln U(p, b, kappa) at b = 1, with
    kappa = s^2 p / c  and U Tricomi's confluent hypergeometric function.
Claim 3.  The loss D of ln R0 is the y at which r = 0 when c is replaced by c e^y, the mean matrix
    being at its threshold (s + c = 1).

Checks (tolerances fixed before the first run):
  G1  the density of Claim 1 is invariant: for 12 parameter sets, the density of Y/(s + X) computed by
      numerical integration from f agrees with f at 40 points to 1e-6 relative (after normalising).
  G2  r from Claim 2 by quadrature of ln(s + x) f(x) agrees with r from Tricomi's function (mpmath,
      numerical derivative in b) to 1e-8, 12 parameter sets.
  G3  r from Claim 2 agrees with the growth rate of the simulated matrix product, 8 parameter sets,
      within 4 standard errors.
  G4  D from Claim 3 agrees with D found by simulation with common random numbers, 6 parameter sets,
      within 4 standard errors.
  G5  small noise: D * (1 + tau) / (1/p) tends to 1/2 as p grows (the second-order value for no memory
      is variance/(2T) per generation, T = 1 + tau): within 2% at p = 400, four stays.
  G6  the bound: D <= l = ln p - digamma(p) in every case computed, and D -> l as s -> 0.
Reported, no criterion: D against the stay tau of the steady stage for hosts of variance 0.1, 0.5, 1, 2
and 4, beside the second-order value and the bound.

FIRST RUN (kept as results/exact-gamma-parking_attempt1_20261114.txt): G3 and G4, the comparisons with
simulation, passed.  G1 and G2 failed and G5 stopped the script, all three through the same fault of
this harness: the range of numerical integration was fixed in advance and missed the bulk of the
density for large and for small shape p (for p = 400 the normalising integral underflowed to zero).
The integrals are now taken around the located maximum of the integrand, on the range where it is
within e^-80 of its maximum, with the maximum factored out.  No tolerance and no formula was changed.

SECOND RUN (kept as results/exact-gamma-parking_attempt2_20261114.txt): G1 and G2 passed (2.8e-14 and
2.8e-17).  G3 to G6 failed for shape p <= 1 only, again through the harness: the new integrator
evaluated the logarithm of the density at x = exp(-800) = 0, where (p - 1) ln x is 0 times minus
infinity for p = 1 and plus infinity for p < 1.  The density in t = ln x is now written directly as
p t - p ln(s + e^t) - s p e^t / c, which is finite for every t.  No tolerance and no formula was changed.

THIRD RUN (kept as results/exact-gamma-parking_attempt3_20261114.txt): G1 to G4 passed.  G5 and G6
FAILED, and this time the fault was in the two claims as I had fixed them, not in the harness:
  G5 as fixed included a steady stay of 1.0001 steps (s = 1e-4).  There the loss at shape 400 is 41%
     above variance/(2T).  The second-order value is the limit of small noise at a fixed stay; it is
     not approached uniformly as the staying tends to zero.
  G6 as fixed required D to be within 0.1% of the bound at s = 1e-6.  It is 16% to 24% below it.  The
     bound is attained at s = 0 exactly (two cohorts that never mix), and for s > 0 it is approached
     only like 1/ln(1/s): Tricomi's function has a logarithm at its origin, U(p, 1, kappa) ~
     -[ln kappa + digamma(p) + 2 gamma]/Gamma(p).
Both are kept below as REPORTED lines with the numbers.  They are replaced by checks whose criteria
were fixed after seeing why the first ones failed and before these were run:
  G5b  for steady stays 2, 10 and 50 the ratio D/(variance/(2T)) is within 2% of one at shape 400.
  G5c  reported: the same ratio at s = 1e-4 for shapes 4e2, 4e4, 4e6.
  G6b  D <= l in every case computed, and (1 - D/l) ln(1/s) changes by less than a factor of two
       across s = 1e-12, 1e-24, 1e-48 while 1 - D/l falls: the approach to the bound is logarithmic.

Run: python3 -I checks/exact_gamma_parking.py   (about two minutes)
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import integrate, special

FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def log_f(x, s, c, p):
    return (p - 1) * np.log(x) - p * np.log(s + x) - s * p * x / c


def peak_integrals(logw, extras=(), t_lo=-700.0, t_hi=60.0):
    """Integrals of exp(logw(t)) * e(t) for e = 1 and each extra, with the maximum factored out.

    Returns (log of the maximum, integral of exp(logw - max), [integrals with each extra factor]).
    The range is where logw is within 80 of its maximum, found by bisection from the located maximum.
    """
    from scipy import optimize
    grid = np.linspace(t_lo, t_hi, 4001)
    vals = np.array([logw(t) for t in grid])
    k = int(np.argmax(vals))
    a0, b0 = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(lambda t: -logw(t), bounds=(a0, b0), method="bounded", options={"xatol": 1e-12})
    t0, m = float(res.x), float(-res.fun)

    def edge(direction):
        lo, hi = t0, t0 + direction * 1.0
        while logw(hi) > m - 80 and abs(hi - t0) < 1500:
            hi = t0 + 2 * (hi - t0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if logw(mid) > m - 80:
                lo = mid
            else:
                hi = mid
        return hi
    tl, tr = edge(-1), edge(+1)
    pts = [t0] + [t0 + f * (tl - t0) for f in (0.02, 0.1, 0.4)] + [t0 + f * (tr - t0) for f in (0.02, 0.1, 0.4)]
    I0 = integrate.quad(lambda t: math.exp(logw(t) - m), tl, tr, points=sorted(pts), limit=600, epsabs=0, epsrel=1e-13)[0]
    Is = [integrate.quad(lambda t, e=e: e(t) * math.exp(logw(t) - m), tl, tr, points=sorted(pts), limit=600, epsabs=0, epsrel=1e-13)[0] for e in extras]
    return m, I0, Is


def logw_t(t, s, c, p):
    """log of the density of ln X at t (unnormalised): p t - p ln(s + e^t) - s p e^t / c."""
    et = math.exp(t) if t < 700 else math.inf
    return p * t - p * math.log(s + et) - s * p * et / c


def moments(s, c, p):
    """log of the normalising constant and E ln(s + X), by quadrature in t = ln x."""
    logw = lambda t: logw_t(t, s, c, p)                                   # density in t
    m, I0, (I1,) = peak_integrals(logw, extras=(lambda t: math.log(s + math.exp(t)),))
    return m + math.log(I0), I1 / I0


def r_exact(s, c, p):
    return moments(s, c, p)[1]


def r_tricomi(s, c, p):
    import mpmath as mp
    mp.mp.dps = 40
    kappa = mp.mpf(s) ** 2 * p / c
    g = lambda b: mp.log(mp.hyperu(p, b, kappa))
    return float(mp.log(s) + mp.diff(g, 1))


def loss_exact(s, p):
    """D: the y with r = 0 when c = (1 - s) e^y."""
    c0 = 1 - s
    lo, hi = 0.0, 1.0
    while r_exact(s, c0 * math.exp(hi), p) < 0:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if r_exact(s, c0 * math.exp(mid), p) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_r(s, c, p, seed, M=400, steps=100_000, burn=500, ys=(0.0,)):
    """Growth rate of the matrix product for each birth tilt y in ys, same environments."""
    rng = np.random.default_rng(seed)
    x = [np.ones((M, 2)) for _ in ys]
    acc = [np.zeros(M) for _ in ys]
    for t in range(burn + steps):
        h = rng.gamma(p, 1.0 / p, M)
        for k, y in enumerate(ys):
            x0, x1 = x[k][:, 0], x[k][:, 1]
            n0 = x1.copy()                                   # g1 = 1; all of c is carried by the other transfer
            n1 = s * x1 + c * math.exp(y) * h * x0
            tot = n0 + n1
            x[k] = np.stack([n0 / tot, n1 / tot], axis=1)
            if t >= burn:
                acc[k] += np.log(tot)
    out = [a / steps for a in acc]
    return out


def main():
    sets = [(0.5, 0.5, 2.0), (0.9, 0.1, 1.0), (0.3, 0.7, 0.5), (0.8, 0.35, 4.0), (0.95, 0.05, 0.25), (0.6, 0.4, 10.0),
            (0.2, 0.8, 1.0), (0.99, 0.01, 2.0), (0.7, 0.2, 0.6), (0.4, 1.2, 3.0), (0.85, 0.15, 30.0), (0.5, 0.5, 0.15)]

    # ------------------------------------------------------------------ G1
    worst = 0.0
    for s, c, p in sets:
        logZ, _ = moments(s, c, p)
        scale = c / p
        xs = np.exp(np.linspace(math.log(1e-3 * c), math.log(8 * c + 2 * c / math.sqrt(p)), 40))
        for xv in xs:
            # density of Y/(s + X) at xv: integral over z of gamma_pdf(xv (s + z)) (s + z) f(z)
            def logint(t):
                z = math.exp(t)
                yv = xv * (s + z)
                lg = (p - 1) * math.log(yv) - yv / scale - special.gammaln(p) - p * math.log(scale)
                return float(lg + math.log(s + z) + logw_t(t, s, c, p))
            m, I0, _ = peak_integrals(logint)
            log_val = m + math.log(I0) - logZ
            log_ref = float(log_f(xv, s, c, p)) - logZ
            val, ref = math.exp(log_val - log_ref), 1.0
            worst = max(worst, abs(val / ref - 1))
    report("G1 the stated density is invariant under X' = Y/(s + X)", worst <= 1e-6, f"largest relative difference {worst:.1e} over {len(sets)} parameter sets and 40 points each")

    # ------------------------------------------------------------------ G2
    worst = 0.0
    for s, c, p in sets:
        worst = max(worst, abs(r_exact(s, c, p) - r_tricomi(s, c, p)))
    report("G2 growth rate: quadrature = ln s + d/db ln U(p, b, s^2 p/c) at b = 1", worst <= 1e-8, f"largest difference {worst:.1e}")

    # ------------------------------------------------------------------ G3
    ok = True
    lines = []
    for i, (s, c, p) in enumerate(sets[:8]):
        sim = simulate_r(s, c, p, 4100 + i)[0]
        m, se = sim.mean(), sim.std(ddof=1) / math.sqrt(len(sim))
        ex = r_exact(s, c, p)
        good = abs(m - ex) <= 4 * se
        ok = ok and good
        lines.append(f"      s {s}, c {c}, shape {p}: simulated {m:+.6f} +- {se:.6f}, exact {ex:+.6f} {'ok' if good else 'OUT'}")
    report("G3 exact growth rate against the simulated matrix product", ok, "eight parameter sets")
    print("\n".join(lines))

    # ------------------------------------------------------------------ G4
    ok = True
    lines = []
    for i, (s, p) in enumerate(((0.5, 1.0), (0.9, 0.5), (0.8, 2.0), (0.96, 1.0), (0.3, 4.0), (0.9, 0.25))):
        c0 = 1 - s
        D = loss_exact(s, p)
        dy = 0.02
        rs = simulate_r(s, c0, p, 5200 + i, ys=(D - dy, D + dy))
        slope = (rs[1] - rs[0]) / (2 * dy)
        root = D - 0.5 * (rs[0] + rs[1]) / slope            # per replicate, common random numbers
        m, se = root.mean(), root.std(ddof=1) / math.sqrt(len(root))
        good = abs(m - D) <= 4 * se
        ok = ok and good
        lines.append(f"      s {s} (stay {1 / (1 - s):.1f}), shape {p} (variance {1 / p:g}): simulated D {m:.5f} +- {se:.5f}, exact {D:.5f} {'ok' if good else 'OUT'}")
    report("G4 exact loss D against simulation", ok, "six parameter sets")
    print("\n".join(lines))

    # ------------------------------------------------------------------ G5 (as first fixed: a false claim, reported), G5b, G5c
    vals = []
    for sv in (0.0001, 0.5, 0.9, 0.98):
        pp = 400.0
        tau = 1 / (1 - sv)
        vals.append((tau, loss_exact(sv, pp) * (1 + tau) * pp / 0.5))
    print("[REPORTED] G5 as first fixed (false as a claim: not uniform as the staying tends to zero): D/(variance/(2T)) at shape 400: "
          + ", ".join(f"stay {t:.4f}: {v:.4f}" for t, v in vals))
    worst = 0.0
    for sv in (0.5, 0.9, 0.98):
        tau = 1 / (1 - sv)
        worst = max(worst, abs(loss_exact(sv, 400.0) * (1 + tau) * 400.0 / 0.5 - 1))
    report("G5b small noise at a fixed stay: D tends to variance/(2T), T = 1 + stay (stays 2, 10, 50)", worst <= 0.02, f"largest relative difference at shape 400: {worst:.4f}")
    print("[REPORTED] G5c the same ratio at s = 1e-4: " + ", ".join(f"shape {pp:g}: {loss_exact(1e-4, pp) * (1 + 1 / (1 - 1e-4)) * pp / 0.5:.4f}" for pp in (4e2, 4e4, 4e6)))

    # ------------------------------------------------------------------ G6 and the table
    bad = 0
    print("\n      Exact loss D of ln R0; host independent from step to step, gamma with the stated variance; fluctuating stage of one step")
    print("      each cell: exact D / second-order value variance/(2(1 + stay)) / share of the bound l = ln p - digamma(p) that is lost")
    print("      steady stay    variance 0.1            variance 0.5            variance 1              variance 2              variance 4")
    for tau in (1.0001, 2.0, 5.0, 10.0, 25.0, 100.0, 1000.0):
        s = 1 - 1 / tau
        cells = []
        for var in (0.1, 0.5, 1.0, 2.0, 4.0):
            p = 1 / var
            ell = math.log(p) - special.digamma(p)
            D = loss_exact(s, p)
            if D > ell + 1e-9:
                bad += 1
            cells.append(f"{D:.4f}/{var / (2 * (1 + tau)):.4f}/{D / ell:.3f}")
        print(f"      {tau:>9.0f}     " + "   ".join(cells))
    print("      bound l:        " + "   ".join(f"{math.log(1 / v) - special.digamma(1 / v):.4f}               " for v in (0.1, 0.5, 1.0, 2.0, 4.0)))
    lim = max(abs(loss_exact(1e-6, 1 / v) / (math.log(1 / v) - special.digamma(1 / v)) - 1) for v in (0.1, 1.0, 4.0))
    print(f"[REPORTED] G6 as first fixed (false as a claim: the approach is logarithmic): largest |D/l - 1| at s = 1e-6: {lim:.3f}")
    ok = bad == 0
    lines = []
    for v in (0.1, 1.0, 4.0):
        pp = 1 / v
        ell = math.log(pp) - special.digamma(pp)
        gaps = [(sv, 1 - loss_exact(sv, pp) / ell) for sv in (1e-6, 1e-12, 1e-24, 1e-48)]
        prods = [g * math.log(1 / sv) for sv, g in gaps]
        falling = all(gaps[i + 1][1] < gaps[i][1] for i in range(3))
        steady = max(prods[1:]) / min(prods[1:]) < 2
        ok = ok and falling and steady
        lines.append(f"      variance {v:g}: 1 - D/l = " + ", ".join(f"{g:.4f} at s = {sv:g}" for sv, g in gaps)
                     + "; times ln(1/s): " + ", ".join(f"{x:.3f}" for x in prods))
    report("G6b the bound holds, and is approached like 1/ln(1/s) as the staying tends to zero", ok, f"violations of the bound {bad}")
    print("\n".join(lines))
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
