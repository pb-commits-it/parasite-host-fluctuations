"""R0 in a random environment for a loop (notes/random-R0.md): exact check for one stage, and the predictions of run R9.

Definition (Bacaer & Khaladi 2013, Proposition 2, taken here as the definition): R0* is the R for which the model
with the births divided by R has long-run growth rate zero. Claim of the note: ln R0 - m ln R0* = D, with m the
number of transfers counted as births, D independent of the mean transfers and of m, and D = Gamma_0 + O(sigma^4),
Gamma_0 being the per-generation closed form evaluated with the stays 1/(1 - s_k).

One stage: the growth rate is exactly E ln(s + (g/R) h), h = exp(sigma z - sigma^2/2), for any autocorrelation
(expectation by Gauss-Hermite quadrature, 200 nodes). Gamma_0 = (1 - s) sigma^2 / 2.

Claims, tolerances stated beforehand:
  B1  D does not depend on g: for 200 random (s, sigma) and three values of g each (R0 from 0.05 to 20),
      ln(g/(1 - s)) - ln R0* agrees across g to 1e-10.
  B2  D = (1 - s) sigma^2/2 + O(sigma^4): with Q(sigma) = (D - (1 - s) sigma^2/2)/sigma^4,
      max(|Q(0.05)|, |Q(0.025)|, |Q(0.0125)|) <= 2 max(|Q(0.2)|, |Q(0.1)|) + 1e-5 on 200 random s (the bounded-ratio
      form, chosen because of the two failed wordings recorded in checks/signed_kappa_remainder.py).
  B3  At sigma = 0.1 and R0 = 5 (so lambda = s + 5(1 - s)), on 200 random s in (0.1, 0.9): D is within 2% of
      Gamma_0; its distance from the value at the growing population's own stay, sigma^2 (1 - s/lambda)/2, is more
      than ten times its distance from Gamma_0 in every case, and more than 10% of that value whenever s >= 0.2.
      First wording, as run at 06:42 (output kept in results/random-r0_attempt1_20261114.txt): "differs by more
      than 10% from the value at the growing population's own stay" for all s in (0.1, 0.9). It failed: the
      smallest difference was 8.1%, at s near 0.1, where the two stays, 1/(1 - s) and 1/(1 - s/lambda), are both
      close to one and differ by 8%. The 10% was mine and ignored that. D was within 0.37% of Gamma_0 in every
      case, so the claim under test (which stay enters) was not in doubt; the restated criterion says the same
      thing without the bad threshold.

  B4  (added 06:55, after primary's blind derivation, which gives the next term for one stage) With p = 1 - s,
      D = p sigma^2/2 + s p (3s - 1) sigma^4/8 + O(sigma^6): on 200 random s, (D - p sigma^2/2)/sigma^4 at
      sigma = 0.025 is within 5e-4 of s p (3s - 1)/8.

Then the script prints, without any simulation, the predictions for run R9 (protocols/R9-random-R0.md).

Run: python3 -I checks/random_r0.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy import optimize

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402

X, W = np.polynomial.hermite_e.hermegauss(200)
W = W / math.sqrt(2 * math.pi)

# run R9: label, staying rates, sigma per transfer, phi per transfer, R0 of the model as given
R9_CASES = (
    ("V1", (0.5, 0.8), (0.25, 0.0), (0.8, 0.0), 4.0),
    ("V2", (0.5, 0.8), (0.25, 0.20), (0.8, 0.5), 0.3),
    ("V3", (0.3, 0.6, 0.9), (0.0, 0.30, 0.0), (0.0, 0.9, 0.0), 4.0),
    ("V4", (0.5, 0.8), (0.25, 0.25), (0.0, 0.0), 4.0),
)


def growth_one_stage(s, a, sig):
    """E ln(s + a h): the exact growth rate of a one-stage cycle with staying s and mean transfer a."""
    return float(np.sum(W * np.log(s + a * np.exp(sig * X - 0.5 * sig * sig))))


def d_one_stage(s, g, sig):
    r0 = g / (1 - s)
    rstar = optimize.brentq(lambda lr: growth_one_stage(s, g / math.exp(lr), sig), math.log(r0) - 5.0, math.log(r0) + 5.0, xtol=1e-14, rtol=1e-14)
    return math.log(r0) - rstar


def gamma_ring(s, g, sig, phi):
    """Per-generation second-order loss of a ring with mean rates (s, g): hosts with their own sigma and phi."""
    s = np.asarray(s, float); g = np.asarray(g, float); tot = 0.0
    for k in range(len(s)):
        if sig[k] > 0:
            var = np.zeros(len(s)); var[k] = sig[k] ** 2
            tot += float(theory.per_generation_cost_transfer(s, g, var, phi[k]).sum())
    return tot


def r9_predictions():
    rows = []
    for label, s, sig, phi, r0 in R9_CASES:
        s_ = np.asarray(s, float); n = len(s_)
        gam0 = gamma_ring(s_, 1 - s_, sig, phi)                           # at the threshold: stays 1/(1 - s_k)
        gam_l = gamma_ring(s_, (1 - s_) * r0 ** (1.0 / n), sig, phi)      # at the stays of the model as given
        t0 = float(np.sum(1 / (1 - s_)))
        rows.append(dict(case=label, s=s, sigma=sig, phi=phi, R0=r0, T0=t0, gamma0=gam0, gamma_given=gam_l, wrong_growth=(gam_l - gam0) / t0))
    return rows


def main():
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    rng = np.random.default_rng(20261114)
    worst = 0.0
    for trial in range(200):
        s = rng.uniform(0.05, 0.95); sig = rng.uniform(0.05, 0.6)
        ds = [d_one_stage(s, r0 * (1 - s), sig) for r0 in (0.05, 1.0, 20.0)]
        worst = max(worst, max(ds) - min(ds))
    report("B1 one stage: the loss of ln R0 does not depend on the mean transfer", worst < 1e-10, f"200 random (s, sigma), R0 = 0.05, 1, 20: largest spread {worst:.1e}")

    ok2, wr = True, 0.0
    for trial in range(200):
        s = rng.uniform(0.05, 0.95)
        Q = {sg: (d_one_stage(s, 3.0 * (1 - s), sg) - (1 - s) * sg * sg / 2) / sg ** 4 for sg in (0.2, 0.1, 0.05, 0.025, 0.0125)}
        small, large = max(abs(Q[0.05]), abs(Q[0.025]), abs(Q[0.0125])), max(abs(Q[0.2]), abs(Q[0.1]))
        ok2 &= small <= 2 * large + 1e-5; wr = max(wr, small / (large + 1e-9))
    report("B2 one stage: the loss is (1 - s) sigma^2 / 2 up to a term of order sigma^4", ok2, f"200 random s: largest ratio of the scaled remainder at sigma <= 0.05 to that at sigma >= 0.1: {wr:.3f}")

    ok3, w0, w1, wsep = True, 0.0, np.inf, np.inf
    for trial in range(200):
        s = rng.uniform(0.1, 0.9); sig = 0.1; lam = s + 5 * (1 - s)
        d = d_one_stage(s, 5 * (1 - s), sig); g0 = (1 - s) * sig * sig / 2; g1 = (1 - s / lam) * sig * sig / 2
        w0 = max(w0, abs(d / g0 - 1)); wsep = min(wsep, abs(d - g1) / abs(d - g0))
        ok3 &= abs(d / g0 - 1) < 0.02 and abs(d - g1) > 10 * abs(d - g0)
        if s >= 0.2:
            w1 = min(w1, abs(d / g1 - 1)); ok3 &= abs(d / g1 - 1) > 0.10
    report("B3 one stage, R0 = 5: the stay that enters is 1/(1 - s), not the stay of the growing population", ok3,
           f"200 random s at sigma = 0.1: largest |D/Gamma_0 - 1| = {w0:.4f}; D is at least {wsep:.0f} times nearer Gamma_0 than the own-stay value; for s >= 0.2 it differs from that value by at least {w1:.3f}")

    w4 = 0.0
    for trial in range(200):
        s = rng.uniform(0.05, 0.95); sg = 0.025; pp = 1 - s
        q4 = (d_one_stage(s, 3.0 * pp, sg) - pp * sg * sg / 2) / sg ** 4
        w4 = max(w4, abs(q4 - s * pp * (3 * s - 1) / 8))
    report("B4 one stage: the next term is s(1 - s)(3s - 1) sigma^4 / 8 (primary's coefficient)", w4 < 5e-4, f"200 random s at sigma = 0.025: largest difference {w4:.1e}")

    print()
    print("Predictions for run R9 (no simulation): loss of ln R0 = Gamma_0; growth per step of the model scaled with Gamma at the given model's own stays")
    for r in r9_predictions():
        print(f"  {r['case']}: s = {r['s']}, sigma = {r['sigma']}, phi = {r['phi']}, R0 = {r['R0']:g}: T0 = {r['T0']:.4f}; Gamma_0 = {r['gamma0']:.6f}; "
              f"Gamma at the given model's stays = {r['gamma_given']:.6f}; growth per step if that is used = {r['wrong_growth']:+.6f}")
    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
