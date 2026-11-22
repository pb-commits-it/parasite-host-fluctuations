"""First experiment on Dobson's stability claim (Unsolved Problems in Ecology, p. 64).

Claim under test: "this use of multiple hosts may provide an intrinsic form of stability
that allows persistence of species that would otherwise be susceptible to extinction
during the boom-bust cycles of individual host species in their life cycles."

Hypothesis (tertiary, 2026-11-11). "Complex" bundles three different things, and they act
in different directions when host abundances fluctuate:
  LENGTH    more obligate host stages in series        -> no buffering; harmful if each
                                                          added host brings its own fluctuation
  LONGEVITY a long-lived stage (the "parking lot")     -> buffers
  BREADTH   several alternative hosts at one stage     -> buffers (portfolio effect), and can
                                                          pay for itself even if it dilutes
                                                          mean transmission

Model. n stages in a ring. x_k(t) = parasites in stage k. Each time step a parasite in
stage k stays alive in that stage with probability s_k and produces g_k(t) parasites of
stage k+1:      x_{k+1}(t+1) = s_{k+1} x_{k+1}(t) + g_k(t) x_k(t).
g_k(t) = gbar_k * h_k(t), where h_k(t) is the abundance of the host needed for that step,
mean 1, lognormal, AR(1) with autocorrelation phi (boom-bust when phi is high).
Mean residence in stage k is 1/(1 - s_k); whole-cycle factor T = prod gbar_k/(1 - s_k).

Measures, per time step:
  log lam1  growth rate with constant hosts
  lam_s     long-run growth rate with fluctuating hosts
  cost      log lam1 - lam_s   (what fluctuation takes away; smaller = better buffered)
  var       variance rate of log population size (drives extinction risk)
"""
import numpy as np

rng = np.random.default_rng(20261111)
TAU, T_CYCLE = 12.0, 2.0          # mean time round the cycle; whole-cycle factor
STEPS, REPS, BURN = 30000, 300, 500


def host_series(m, sigma, phi):
    """(STEPS, REPS, m) lognormal AR(1) host abundances with mean 1."""
    z = np.zeros((REPS, m))
    out = np.empty((STEPS, REPS, m))
    innov = sigma * np.sqrt(1 - phi ** 2)
    z = rng.normal(0, sigma, (REPS, m))
    for t in range(STEPS):
        z = phi * z + innov * rng.normal(0, 1, (REPS, m))
        out[t] = np.exp(z - sigma ** 2 / 2)
    return out


def run(s, gbar, drivers):
    """s, gbar: length-n arrays. drivers: list (one per stage) of None or (weights, sigma, phi);
    stage k's transmission is gbar_k * sum_j weights_j h_j(t)."""
    n = len(s)
    A = np.diag(s) + np.roll(np.diag(gbar), 1, axis=0)
    lam1 = max(abs(np.linalg.eigvals(A)))
    series = [None if d is None else host_series(len(d[0]), d[1], d[2]) for d in drivers]
    x = np.ones((REPS, n)) / n
    logsum = np.zeros(REPS)
    for t in range(STEPS):
        g = np.empty((REPS, n))
        for k in range(n):
            g[:, k] = gbar[k] if drivers[k] is None else gbar[k] * (series[k][t] @ np.asarray(drivers[k][0]))
        x = x * s + np.roll(x * g, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= BURN:
            logsum += np.log(tot)
    L = STEPS - BURN
    return np.log(lam1), logsum.mean() / L, logsum.var() / L


def ring(n, residences=None):
    res = np.full(n, TAU / n) if residences is None else np.asarray(residences, float)
    s = 1 - 1 / res
    gbar = T_CYCLE ** (1 / n) * (1 - s)
    return s, gbar


def show(label, s, gbar, drivers):
    l1, ls, var = run(s, gbar, drivers)
    print(f"  {label:44s} log lam1 {l1:7.4f}   lam_s {ls:7.4f}   cost {l1 - ls:7.4f}   var {var:7.4f}")
    return l1 - ls, var


SIG = 0.8
for phi in (0.0, 0.9):
    kind = "uncorrelated years" if phi == 0 else "boom-bust (autocorrelation 0.9)"
    print(f"\n=== host fluctuations: sd {SIG} on log scale, {kind} ===")

    print("LENGTH, every obligate host fluctuates independently")
    for n in (1, 2, 3, 4, 6):
        s, g = ring(n)
        show(f"{n} stage(s), residence {TAU / n:.0f} each", s, g, [((1.0,), SIG, phi)] * n)

    print("LENGTH, only one host fluctuates, the rest constant")
    for n in (1, 2, 3, 4, 6):
        s, g = ring(n)
        show(f"{n} stage(s), residence {TAU / n:.0f} each", s, g, [((1.0,), SIG, phi)] + [None] * (n - 1))

    print("LONGEVITY, two stages; the step that leaves the long-lived stage depends on a fluctuating host")
    for r1 in (2.0, 6.0, 10.0):
        s, g = ring(2, [r1, TAU - r1])
        show(f"parking stage lives {r1:.0f}, other stage {TAU - r1:.0f}", s, g, [((1.0,), SIG, phi), None])

    print("BREADTH, two stages; the fluctuating step is shared among m alternative hosts")
    for m in (1, 2, 4, 8):
        s, g = ring(2)
        show(f"{m} alternative host(s), same mean transmission", s, g, [(tuple([1 / m] * m), SIG, phi), None])
    s, g = ring(2)
    show("2 hosts, second is poor: mean transmission -20%", s, g, [((0.5, 0.3), SIG, phi), None])
    show("2 hosts, second is poor: mean transmission -40%", s, g, [((0.5, 0.1), SIG, phi), None])
