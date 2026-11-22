"""Follow-up to stability_v0: remove two confounds.

(1) LENGTH with stage longevity held fixed. Every stage has the same residence R and the
    same per-stage factor f, so the growth rate with constant hosts is identical for every
    number of stages n (lam1 = 1 + (f - 1)/R). Only the number of obligate hosts changes.
(2) SLOWNESS. Two stages, whole-cycle factor fixed, all residences scaled together, against
    host fluctuations with a fixed correlation time. Is buffering just a slow life cycle?

Same model and notation as stability_v0.py. Extra column:
  safety = 2 * lam_s / var. In the diffusion approximation the chance of ever falling a
  log-distance d below the present size is exp(-safety * d); larger is safer, and a
  negative value means eventual decline.
"""
import numpy as np

rng = np.random.default_rng(20261112)
STEPS, REPS, BURN = 30000, 300, 500


def host_series(m, sigma, phi):
    out = np.empty((STEPS, REPS, m))
    z = rng.normal(0, sigma, (REPS, m))
    innov = sigma * np.sqrt(1 - phi ** 2)
    for t in range(STEPS):
        z = phi * z + innov * rng.normal(0, 1, (REPS, m))
        out[t] = np.exp(z - sigma ** 2 / 2)
    return out


def run(s, gbar, fluct, sigma, phi):
    """fluct: boolean per stage; a fluctuating stage's transmission is gbar_k * h_k(t)."""
    n = len(s)
    A = np.diag(s) + np.roll(np.diag(gbar), 1, axis=0)
    lam1 = max(abs(np.linalg.eigvals(A)))
    h = host_series(n, sigma, phi)
    mask = np.asarray(fluct, bool)
    x = np.ones((REPS, n)) / n
    logsum = np.zeros(REPS)
    for t in range(STEPS):
        g = np.where(mask, gbar * h[t], gbar)
        x = x * s + np.roll(x * g, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= BURN:
            logsum += np.log(tot)
    L = STEPS - BURN
    return np.log(lam1), logsum.mean() / L, logsum.var() / L


def show(label, s, gbar, fluct, sigma, phi):
    l1, ls, var = run(np.asarray(s, float), np.asarray(gbar, float), fluct, sigma, phi)
    print(f"  {label:40s} log lam1 {l1:7.4f}  lam_s {ls:7.4f}  cost {l1 - ls:7.4f}  var {var:7.4f}  safety {2 * ls / var:7.2f}")


SIG, R, F = 0.8, 3.0, 1.2
for phi in (0.0, 0.9):
    print(f"\n=== sd {SIG}, autocorrelation {phi} ===")
    print(f"LENGTH at fixed stage residence {R:.0f} and per-stage factor {F}: every host fluctuates")
    for n in (1, 2, 3, 4, 6, 8):
        s = np.full(n, 1 - 1 / R); g = F * (1 - s)
        show(f"{n} stage(s)", s, g, [True] * n, SIG, phi)
    print("LENGTH, same, but only one host fluctuates")
    for n in (1, 2, 3, 4, 6, 8):
        s = np.full(n, 1 - 1 / R); g = F * (1 - s)
        show(f"{n} stage(s)", s, g, [True] + [False] * (n - 1), SIG, phi)

print("\n=== SLOWNESS: two stages, whole-cycle factor 2, one fluctuating host, sd 0.8, autocorrelation 0.9 (correlation time about 10 steps) ===")
for tau in (4, 8, 16, 32, 64):
    s = np.full(2, 1 - 2 / tau); g = 2 ** 0.5 * (1 - s)
    l1, ls, var = run(s, g, [True, False], SIG, 0.9)
    print(f"  cycle time {tau:3d}: log lam1 {l1:7.4f}  lam_s {ls:7.4f}  fraction of growth lost {1 - ls / l1:6.2%}  var {var:7.4f}  safety {2 * ls / var:7.2f}")
