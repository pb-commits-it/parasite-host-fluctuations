"""Independent check of secondary's closed form for the cost of host fluctuation in a ring.

Claim (secondary, board/0002). Symmetric ring of n stages; each step a parasite stays in its
stage with probability s and produces gbar * h_k(t) parasites of the next stage; the n host
series h_k are independent, lognormal with mean 1, log-sd sigma, AR(1) with autocorrelation
phi. With lam0 = s + gbar and e = gbar / lam0, the growth rate lost to fluctuation is, to
second order in sigma,

  cost = (e^2 sigma^2 / n) * [ 1/2 + phi/(1-phi)
                               - n lam0 phi^n gbar^(n-1) / ((lam0 - phi s)^n - (phi gbar)^n) ].

Consequences claimed: no dependence on phi at n = 1; under strong autocorrelation the cost
first rises with n, peaks, then falls roughly as 1/n.

Method here: my own simulator (not secondary's), sigma = 0.15, a control variate that removes
the first-order noise (the mean of h - 1 is exactly zero), and a standard error from
replicate runs. The formula is second order, so agreement within a few per cent plus
sampling error is what would count as confirmation.
"""
import numpy as np

rng = np.random.default_rng(42)
S, GBAR = 2.0 / 3.0, 0.4          # stage residence 3, per-stage factor 1.2 (as in stability_v1)
LAM0 = S + GBAR
E = GBAR / LAM0
SIGMA, STEPS, REPS, BURN = 0.15, 120000, 400, 2000


def formula(n, phi):
    tail = n * LAM0 * phi ** n * GBAR ** (n - 1) / ((LAM0 - phi * S) ** n - (phi * GBAR) ** n)
    return E ** 2 * SIGMA ** 2 / n * (0.5 + phi / (1 - phi) - tail)


def simulate(n, phi):
    z = rng.normal(0, SIGMA, (REPS, n))
    innov = SIGMA * np.sqrt(1 - phi ** 2)
    x = np.ones((REPS, n)) / n
    acc = np.zeros(REPS)
    c = E / n                                   # elasticity of each transmission step
    for t in range(STEPS):
        z = phi * z + innov * rng.normal(0, 1, (REPS, n))
        h = np.exp(z - SIGMA ** 2 / 2)
        x = x * S + np.roll(x * GBAR * h, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= BURN:
            acc += np.log(tot) - c * (h - 1).sum(axis=1)
    lam_s = acc / (STEPS - BURN)
    cost = np.log(LAM0) - lam_s
    return cost.mean(), cost.std(ddof=1) / np.sqrt(REPS)


print(f"sigma = {SIGMA}; cost of fluctuation per time step (x 1e4)")
print("  phi    n   formula   simulated   s.e.   sim/formula")
for phi in (0.0, 0.9):
    for n in (1, 2, 4, 8, 16, 32):
        f = formula(n, phi)
        m, se = simulate(n, phi)
        print(f"  {phi:3.1f}  {n:3d}   {f * 1e4:7.3f}   {m * 1e4:8.3f}   {se * 1e4:5.3f}   {m / f:6.3f}")
