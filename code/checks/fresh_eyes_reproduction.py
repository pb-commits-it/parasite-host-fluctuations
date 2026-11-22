#!/usr/bin/env python3
"""Independent reproduction of the auditor's examples C1 to C7, with three extensions.

Run from the repository root, always in isolated mode:

    python3 -I checks/fresh_eyes_reproduction.py

Only numpy, scipy and the standard library are used. Nothing is read from disk and
nothing is written; the report goes to standard output. No code of the auditor is
used. The constructions come from the text of the auditor's report and, where the
text is incomplete (C3, C5, C6, C7), from the auditor's printed output. The
auditor's numbers below were typed in from that printed output (nine or more
digits) and from the report text (fewer digits).

TOLERANCES, fixed before the first run and not changed afterwards
  deterministic : |mine - auditor| <= 1e-5 (absolute). Where the auditor's number is
                  not zero and below 1e-3 in absolute value, the relative difference
                  must also be <= 1e-6 (an absolute 1e-5 says nothing about 1e-33).
  report text   : a number quoted in the report with fewer digits passes if my number
                  rounds to it (half a unit in the last quoted place).
  Monte Carlo   : |mine - auditor| <= 4 standard errors of my estimate.

HISTORY
  Attempt 1 (output kept as results/fresh-eyes-reproduction_attempt1_20261114.txt)
  stopped in C6 on a defect of this script: it tried to print the number of digits of
  a 4900-digit integer, which Python refuses by default. All 108 comparisons made
  before that point (C1 to C5 and three of C6) had passed. Two changes followed, and
  neither touches a comparison or its tolerance: the size of that integer is now
  printed through its logarithm, and three quadratures that asked for more accuracy
  than double precision gives (which produced warnings) now ask for 1e-12 relative.
  At the same time two quadratures that had not yet run (stationary ages in C7 and
  E-c, used only as a cross-check of the finite sums) were set to 1e-10 relative, and
  one print format in the extra lines of C4 was made uniform. The saved output
  results/fresh-eyes-reproduction_20261114.txt is the second run.

METHODS, chosen to differ from the auditor's (stated in the auditor's own words as
"quadrature, finite transition matrices, generating functions, or periodic linear
systems")
  C1 scalar   adaptive quadrature and a uniform trapezoid rule (not a Gauss rule)
  C1 matrix   a closed form from cosh and sinh, and integration of the differential
              equation (not a product of matrix exponentials)
  C2          closed-form 2x2 exponentials and integration of the differential
              equation; scipy.linalg.expm as a third route
  C3          root of the characteristic (renewal) equation by bracketing, and the
              quadratic formula (not an eigenvalue routine); quadrature for the shock
  C4          the nonlinear ring itself: equilibrium found numerically, Jacobian by
              complex-step differentiation of the right-hand side (the auditor used a
              rescaled matrix with diagonal -1 and cycle gains 1 - kappa)
  C5          bracketing root of the generating-function equation, and simulation of
              the individual-based process; exact rational arithmetic for the packages
  C6          a closed form for the long block; exact integer arithmetic, backwards in
              time, for the alternating schedule; simulation of the long block
  C7          finite sums in closed form, the incomplete gamma function, and quadrature

EXTENSIONS
  E-a  exact growth rates of the gain-only and contact models on a grid of death rates
  E-b  allocation of regulation on 300 random rings, two forms of suppression
  E-c  survival through a gap for five residence-time distributions
"""

import datetime
import math
import sys
import time
from fractions import Fraction

import numpy as np
import scipy
from scipy import integrate, linalg, optimize, special

# --------------------------------------------------------------------------------------
# tolerances (fixed before the first run)
# --------------------------------------------------------------------------------------
DET_ABS = 1.0e-5
TINY = 1.0e-3
DET_REL_TINY = 1.0e-6
MC_NSE = 4.0
SEED = 20261114

RESULTS = []  # (example, label, ok)


def banner(title):
    print()
    print(title)
    print("-" * len(title))


def check(ex, label, auditor, mine):
    """Deterministic comparison against a number from the auditor's printed output."""
    diff = mine - auditor
    ok = abs(diff) <= DET_ABS
    tol = "1e-5 abs"
    shown = f"{diff:+.2e}"
    if auditor != 0.0 and abs(auditor) < TINY:
        rel = abs(diff) / abs(auditor)
        ok = ok and rel <= DET_REL_TINY
        tol = "1e-6 rel"
        shown = f"rel {rel:.1e}"
    RESULTS.append((ex, label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<50s} auditor {auditor:< .10g}  mine {mine:< .12g}"
          f"  diff {shown}  (tol {tol})")
    return ok


def check_text(ex, label, quoted, unit, mine):
    """A number quoted in the report text: mine must round to it."""
    diff = mine - quoted
    ok = abs(diff) <= 0.5 * unit * (1.0 + 1e-9)
    RESULTS.append((ex, label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<50s} text    {quoted:< .10g}  mine {mine:< .12g}"
          f"  diff {diff:+.2e}  (tol half of {unit:g})")
    return ok


def check_mc(ex, label, auditor, est, se):
    diff = est - auditor
    ok = abs(diff) <= MC_NSE * se
    RESULTS.append((ex, label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<50s} auditor {auditor:< .10g}  mine {est:< .8g}"
          f"  diff {diff:+.2e}  (s.e. {se:.2e}; {abs(diff) / se if se > 0 else float('nan'):.2f} s.e.; tol 4 s.e.)")
    return ok


def fmt_p(p):
    """A probability: six decimals if it is not small, three significant figures otherwise."""
    if p == 0.0:
        return "0"
    if p >= 1e-4:
        return f"{p:.6f}"
    return f"{p:.3e}"


# --------------------------------------------------------------------------------------
# linear periodic systems
# --------------------------------------------------------------------------------------
def expm2(A, t):
    """exp(A t) for a real 2x2 matrix with two distinct real eigenvalues, in closed form."""
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    m = 0.5 * (a + d)
    disc = (0.5 * (a - d)) ** 2 + b * c
    if disc <= 0.0:
        raise ValueError("eigenvalues not real and distinct")
    dl = math.sqrt(disc)
    lp, lm = m + dl, m - dl
    I = np.eye(2)
    return (math.exp(lp * t) * (A - lm * I) - math.exp(lm * t) * (A - lp * I)) / (lp - lm)


def dom_eig2(M):
    """Dominant eigenvalue of a non-negative 2x2 matrix."""
    tr = M[0, 0] + M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return 0.5 * (tr + math.sqrt(max(tr * tr - 4.0 * det, 0.0)))


def growth_closed(mats, T):
    M = np.eye(2)
    for A in mats:
        M = expm2(A, T) @ M
    return math.log(dom_eig2(M)) / (len(mats) * T)


def growth_expm(mats, T):
    M = np.eye(2)
    for A in mats:
        M = linalg.expm(A * T) @ M
    return math.log(max(abs(np.linalg.eigvals(M)))) / (len(mats) * T)


def growth_ode(mats, T):
    """Growth rate from the fundamental matrix, integrating the differential equation."""
    Phi = np.eye(2)
    for A in mats:
        sol = integrate.solve_ivp(lambda t, y, A=A: (A @ y.reshape(2, 2)).ravel(), (0.0, T), Phi.ravel(),
                                  method="DOP853", rtol=1e-12, atol=1e-15)
        Phi = sol.y[:, -1].reshape(2, 2)
    return math.log(max(abs(np.linalg.eigvals(Phi)))) / (len(mats) * T)


# --------------------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------------------
def gauss_density(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def scalar_growth(sigma, mean_one, s=0.9, g=0.1):
    """E log(s + g h), h = exp(sigma Z - sigma^2/2) (mean one) or exp(sigma Z) (median one)."""
    shift = -0.5 * sigma * sigma if mean_one else 0.0
    q, _ = integrate.quad(lambda z: math.log(s + g * math.exp(sigma * z + shift)) * gauss_density(z),
                          -14.0, 14.0, points=[-4.0, 0.0, 4.0], epsabs=1e-13, epsrel=1e-12, limit=500)
    z = np.linspace(-14.0, 14.0, 5601)
    t = float(np.sum(np.log(s + g * np.exp(sigma * z + shift)) * np.exp(-0.5 * z * z))
              * (z[1] - z[0]) / math.sqrt(2.0 * math.pi))
    return q, t


def c1_matrix_growth_closed(T, lo=0.5, hi=1.5):
    """A(H) = [[-1, 1/H], [H, -1]] has eigenvalues 0 and -2 for every H, so
    exp(A T) = e^{-T} (cosh T I + sinh T J(H)), J(H) = [[0, 1/H], [H, 0]].
    The product over the two states has determinant e^{-4T} and half-trace
    e^{-2T} [cosh^2 T + (1/2) sinh^2 T (hi/lo + lo/hi)]."""
    rr = hi / lo
    half_tr = math.cosh(T) ** 2 + 0.5 * math.sinh(T) ** 2 * (rr + 1.0 / rr)
    return -1.0 + math.acosh(half_tr) / (2.0 * T)


def example_c1():
    banner("C1. Mean rates, mean hosts, and the sign of the effect of variability")
    print("  (a) Scalar map x(t+1) = (0.9 + 0.1 h) x(t); growth = E log(0.9 + 0.1 h); h lognormal with")
    print("      arithmetic mean one, exp(sigma Z - sigma^2/2), or median one, exp(sigma Z).")
    aud = {0.30: (-0.000462687, 0.004090918, 1.046027860),
           0.80: (-0.003802417, 0.030567751, 1.377127764),
           1.00: (-0.006386136, 0.048885187, 1.648721271)}
    worst = 0.0
    mine = {}
    for sigma, (a_mean, a_med, a_eh) in aud.items():
        q1, t1 = scalar_growth(sigma, True)
        q2, t2 = scalar_growth(sigma, False)
        eh, _ = integrate.quad(lambda z: math.exp(sigma * z) * gauss_density(z), -14.0 + sigma, 14.0 + sigma,
                               points=[sigma], epsabs=1e-13, epsrel=1e-12, limit=500)
        worst = max(worst, abs(q1 - t1), abs(q2 - t2))
        mine[sigma] = (q1, q2, eh)
        check("C1", f"scalar, sigma={sigma:.2f}, mean-one growth", a_mean, q1)
        check("C1", f"scalar, sigma={sigma:.2f}, median-one growth", a_med, q2)
        check("C1", f"scalar, sigma={sigma:.2f}, median-one E[h]", a_eh, eh)
    print(f"      adaptive quadrature against trapezoid rule, largest difference: {worst:.1e}")
    check_text("C1", "text: mean-one growth at sigma=0.8", -0.003802, 1e-6, mine[0.80][0])
    check_text("C1", "text: median-one growth at sigma=0.8", 0.030568, 1e-6, mine[0.80][1])
    check_text("C1", "text: median-one E[h] at sigma=0.8", 1.377, 1e-3, mine[0.80][2])
    q = 0.1
    print("      Leading terms claimed in the text: -q^2 sigma^2/2 (mean one), +q(1-q) sigma^2/2 (median one), q=0.1.")
    for sigma in (0.05, 0.3, 0.8):
        g1, _ = scalar_growth(sigma, True)
        g2, _ = scalar_growth(sigma, False)
        print(f"        sigma={sigma:4.2f}: exact/leading = {g1 / (-q * q * sigma ** 2 / 2):.4f} (mean one), "
              f"{g2 / (q * (1 - q) * sigma ** 2 / 2):.4f} (median one)")

    print("  (b) Two stages, A(H) = [[-1, 1/H], [H, -1]]; H = 0.5 and 1.5 for equal times; mean H = 1.")
    print("      A(H) has eigenvalues 0 and -2 for EVERY constant H, so any constant host gives growth zero.")
    aud_r = {0.1: 0.154062545, 1.0: 0.112735305, 5.0: 0.028765937, 20.0: 0.007192052}
    aud_loss = {0.1: 0.000637993, 1.0: 0.041965234, 5.0: 0.125934601, 20.0: 0.147508487}
    mean_rate = np.array([[-1.0, 0.5 * (1 / 0.5 + 1 / 1.5)], [0.5 * (0.5 + 1.5), -1.0]])
    r_mean = float(max(np.linalg.eigvals(mean_rate).real))
    check("C1", "matrix of mean rates, growth", 0.154700538, r_mean)
    mats = [np.array([[-1.0, 1.0 / H], [H, -1.0]]) for H in (0.5, 1.5)]
    worst = 0.0
    r_at = {}
    for T in (0.1, 1.0, 5.0, 20.0):
        r_c = c1_matrix_growth_closed(T)
        r_o = growth_ode(mats, T)
        worst = max(worst, abs(r_c - r_o))
        r_at[T] = r_c
        check("C1", f"half-period {T:g}: growth", aud_r[T], r_c)
        check("C1", f"half-period {T:g}: loss against mean rates", aud_loss[T], r_mean - r_c)
    print(f"      closed form against integration of the differential equation, largest difference: {worst:.1e}")
    check_text("C1", "text: growth, one time unit in each state", 0.112735, 1e-6, r_at[1.0])
    check_text("C1", "text: growth of the matrix of mean rates", 0.154701, 1e-6, r_mean)
    check_text("C1", "text: loss against mean rates", 0.041965, 1e-6, r_mean - r_at[1.0])
    print("      Gain against a constant host (any constant) by half-period: "
          + ", ".join(f"T={T:g}: {r_at[T]:+.6f}" for T in (0.1, 1.0, 5.0, 20.0)))
    print(f"      Limits of the closed form: fast switching -> sqrt(4/3)-1 = {math.sqrt(4.0 / 3.0) - 1.0:.6f} (the matrix of mean rates);"
          f" slow switching -> 0. At T=0.001: {c1_matrix_growth_closed(0.001):.6f}; at T=300: {c1_matrix_growth_closed(300.0):.6f}.")
    print(f"      log(mean H x mean 1/H) = {math.log(4.0 / 3.0):.6f}; the text's leading term sigma^2 x sum(chi)/2 = "
          f"{0.25 * 2 / 2:.6f} (variance of H is 0.25, chi = 0 and 2).")


# --------------------------------------------------------------------------------------
# C2 and E-a
# --------------------------------------------------------------------------------------
A2 = 1.2


def gain_only(mu, H, c=1.0):
    a1 = c + mu
    return np.array([[-a1, A2 * a1 / c], [c * H, -A2]])


def contact(mu, H, c=1.0):
    a1 = c + mu
    return np.array([[-(c * H + mu), A2 * a1 / c], [c * H, -A2]])


def example_c2():
    banner("C2. Host availability as a contact hazard that also removes the donor")
    print("  gain-only: X' = -(1+mu) X + b Y,    Y' = H(t) X - 1.2 Y")
    print("  contact:   X' = -(H(t)+mu) X + b Y, Y' = H(t) X - 1.2 Y;  b = 1.2 (1+mu); H = 0.2 / 1.8, four time units each.")
    aud = {0.00: (-0.079175719, 0.0), 0.02: (-0.080518174, -0.002702601), 0.20: (-0.091154809, -0.026718630)}
    worst = 0.0
    mine = {}
    for mu, (a_g, a_k) in aud.items():
        G = [gain_only(mu, H) for H in (0.2, 1.8)]
        K = [contact(mu, H) for H in (0.2, 1.8)]
        g_o, k_o = growth_ode(G, 4.0), growth_ode(K, 4.0)
        g_c, k_c = growth_closed(G, 4.0), growth_closed(K, 4.0)
        g_e, k_e = growth_expm(G, 4.0), growth_expm(K, 4.0)
        worst = max(worst, abs(g_o - g_c), abs(k_o - k_c), abs(g_e - g_c), abs(k_e - k_c))
        mine[mu] = (g_c, k_c)
        check("C2", f"mu={mu:.2f}: gain-only growth", a_g, g_c)
        check("C2", f"mu={mu:.2f}: contact growth", a_k, k_c)
    print(f"      three routes (closed-form exponentials, differential equation, expm), largest difference: {worst:.1e}")
    check_text("C2", "text: gain-only growth at mu=0", -0.079176, 1e-6, mine[0.00][0])
    check_text("C2", "text: gain-only growth at mu=0.02", -0.080518, 1e-6, mine[0.02][0])
    check_text("C2", "text: contact growth at mu=0.02", -0.002703, 1e-6, mine[0.02][1])
    print("      Column sums of the contact matrix at mu = 0 (zero means X + Y is conserved for any H): "
          + ", ".join(f"H={H:g}: {contact(0.0, H).sum(axis=0)}" for H in (0.2, 1.8, 7.3)))
    print("      The two models have the same mean matrix: "
          + str(all(np.allclose(0.5 * (gain_only(mu, 0.2) + gain_only(mu, 1.8)),
                                0.5 * (contact(mu, 0.2) + contact(mu, 1.8))) for mu in aud)))
    for mu in (0.02, 0.20):
        print(f"      mu={mu:.2f}: gain-only loss / contact loss = {mine[mu][0] / mine[mu][1]:.2f}"
              + ("   (the text says 'roughly thirty-fold')" if mu == 0.02 else "   (not quoted in the text)"))


def ea_losses(mu, lo, hi, T, c=1.0, route="expm"):
    G = [gain_only(mu, H, c) for H in (lo, hi)]
    K = [contact(mu, H, c) for H in (lo, hi)]
    if route == "expm":
        return -growth_expm(G, T), -growth_expm(K, T)
    return -growth_closed(G, T), -growth_closed(K, T)


def extension_a():
    banner("E-a. Gain-only against contact forcing over a grid of death rates (exact growth rates)")
    print("  G (gain-only): X' = -(c+mu) X + b Y,        Y' = c H(t) X - a2 Y")
    print("  K (contact):   X' = -(c H(t)+mu) X + b Y,   Y' = c H(t) X - a2 Y")
    print("  a2 = 1.2; b = a2 (c+mu)/c, so the matrix at mean H = 1 has growth zero in both, and both have the same")
    print("  mean matrix. H is a square wave lo/hi with half-period T. loss = -(1/2T) log(dominant eigenvalue of")
    print("  expm(A_hi T) expm(A_lo T)), computed with scipy.linalg.expm and again with closed-form 2x2 exponentials.")
    print("  Nothing is fitted. Column 'death^2' is (mu/(1+mu))^2, the square of the fraction of X that die rather")
    print("  than transfer at mean host density. Column '2nd order' is (mu/(c+mu)) (1 + c/a2), the ratio K/G that a")
    print("  second-order expansion in the amplitude of H gives for any half-period (derived, not fitted; it is")
    print("  exact only as the amplitude tends to zero).")
    mus = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    waves = [(0.2, 1.8), (0.9, 1.1)]
    periods = [0.5, 4.0, 32.0]
    worst = 0.0
    ratio = {}
    print()
    print("  Table E-a.1 (c = 1)")
    for lo, hi in waves:
        for T in periods:
            print(f"  wave {lo:g}/{hi:g}, half-period {T:g}")
            print("      mu      loss G        loss K        K/G       death^2   2nd order")
            for mu in mus:
                lg, lk = ea_losses(mu, lo, hi, T)
                lg2, lk2 = ea_losses(mu, lo, hi, T, route="closed")
                worst = max(worst, abs(lg - lg2), abs(lk - lk2))
                r = lk / lg
                ratio[(lo, T, mu)] = r
                print(f"    {mu:6.2f}  {lg:12.6e}  {lk:12.6e}  {r:9.5f}  {(mu / (1 + mu)) ** 2:9.5f}  "
                      f"{mu / (1 + mu) * (1 + 1 / A2):9.5f}")
    print(f"  expm against closed-form exponentials over the whole grid, largest difference in a growth rate: {worst:.1e}")
    print()
    print("  Compact form of Table E-a.1: ratio K/G (contact loss over gain-only loss)")
    print("             wave 0.9/1.1                 wave 0.2/1.8")
    print("      mu    T=0.5    T=4      T=32       T=0.5    T=4      T=32      death^2  2nd order")
    for mu in mus:
        row = "  ".join(f"{ratio[(0.9, T, mu)]:7.4f}" for T in periods) + "    " \
              + "  ".join(f"{ratio[(0.2, T, mu)]:7.4f}" for T in periods)
        print(f"    {mu:6.2f}  {row}    {(mu / (1 + mu)) ** 2:7.4f}  {mu / (1 + mu) * (1 + 1 / A2):7.4f}")
    cross = {}
    for lo, hi in waves:
        for T in periods:
            f = lambda m: ea_losses(m, lo, hi, T, route="closed")[1] / ea_losses(m, lo, hi, T, route="closed")[0] - 1.0
            if f(0.05) * f(40.0) < 0.0:
                cross[(lo, T)] = f"{optimize.brentq(f, 0.05, 40.0, xtol=1e-10):.4f}"
            else:
                cross[(lo, T)] = "none in (0.05, 40)"
    print("  Death rate mu at which the two losses are equal (K/G = 1), found by bracketing; second order gives mu = a2 = 1.2:")
    print("    " + "; ".join(f"wave {lo:g}/{hi:g} T={T:g}: {cross[(lo, T)]}" for lo, hi in waves for T in periods))

    print()
    print("  Table E-a.2 (extra rows, c = 1): very small amplitude, and very large death rates")
    print("    wave        T       mu      loss G        loss K        K/G       2nd order   loss K - loss G")
    for lo, hi, T, mu in [(0.99, 1.01, 4.0, 0.02), (0.99, 1.01, 4.0, 0.5), (0.99, 1.01, 4.0, 5.0), (0.99, 1.01, 4.0, 50.0),
                          (0.2, 1.8, 4.0, 50.0), (0.2, 1.8, 4.0, 500.0), (0.2, 1.8, 4.0, 5000.0),
                          (0.9, 1.1, 4.0, 500.0), (0.9, 1.1, 4.0, 5000.0)]:
        lg, lk = ea_losses(mu, lo, hi, T, route="closed")
        print(f"    {lo:g}/{hi:g}".ljust(14) + f"{T:4g}  {mu:7.2f}  {lg:12.6e}  {lk:12.6e}  {lk / lg:9.5f}  "
              f"{mu / (1 + mu) * (1 + 1 / A2):9.5f}   {lk - lg:+.3e}")

    print()
    print("  Table E-a.3: is the gain-only model a limit of the contact model?")
    print("  Contact model with transfer rate c H(t) and death rate mu, against the gain-only model with the same")
    print("  total leaving rate a1 = c + mu and the same b = a2 a1/c. Half-period 4.")
    print("    wave      a1      c       mu      loss G        loss K        K/G      (1-c/a1)(1+c/a2)  loss K - loss G")
    for lo, hi in waves:
        for a1 in (2.0, 11.0):
            for c in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
                mu = a1 - c
                lg, lk = ea_losses(mu, lo, hi, 4.0, c=c, route="closed")
                print(f"    {lo:g}/{hi:g}".ljust(12) + f"{a1:5.1f}  {c:5.2f}  {mu:7.2f}  {lg:12.6e}  {lk:12.6e}  {lk / lg:8.5f}  "
                      f"{(1 - c / a1) * (1 + c / A2):12.5f}      {lk - lg:+.3e}")


# --------------------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------------------
def c3_growth(B, nu, mu=0.02, a=1.0):
    """Immature I' = -(nu+mu) I + B a A, adult A' = nu I - a A. Growth rate r solves
    B a nu / ((r + nu + mu)(r + a)) = 1 (renewal equation of the loop)."""
    f = lambda r: B * a * nu / ((r + nu + mu) * (r + a)) - 1.0
    r_root = optimize.brentq(f, -min(a, nu + mu) + 1e-12, 10.0, xtol=1e-15, rtol=1e-14)
    tr = -(nu + mu + a)
    det = (nu + mu) * a - B * a * nu
    r_quad = 0.5 * (tr + math.sqrt(tr * tr - 4.0 * det))
    return r_root, r_quad


def example_c3():
    banner("C3. A longer stage at fixed mortality and fecundity; and a death-only shock")
    print("  (a) Immature I' = -(nu+mu) I + B a A, adult A' = nu I - a A; B = 1.2 (lifetime output of an adult),")
    print("      a = 1, mu = 0.02, nu = 1/d. R0 = B nu/(nu+mu). Second comparison: B raised so that R0 stays 1.2.")
    aud = {2: (1.923077, 1.153846, 0.050925, 1.248000, 0.065591),
           5: (4.545455, 1.090909, 0.016179, 1.320000, 0.035058),
           10: (8.333333, 1.000000, 0.000000, 1.440000, 0.021034),
           20: (14.285714, 0.857143, -0.009429, 1.680000, 0.012928),
           50: (25.000000, 0.600000, -0.015619, 2.400000, 0.007636)}
    B, mu = 1.2, 0.02
    worst = 0.0
    mine = {}
    for d, (a_res, a_R0, a_r, a_B, a_r2) in aud.items():
        nu = 1.0 / d
        res = 1.0 / (nu + mu)
        R0 = B * nu / (nu + mu)
        r1, r1q = c3_growth(B, nu)
        Bneed = 1.2 * (nu + mu) / nu
        r2, r2q = c3_growth(Bneed, nu)
        worst = max(worst, abs(r1 - r1q), abs(r2 - r2q))
        mine[d] = (R0, r1, Bneed)
        check("C3", f"d={d}: mean residence of the immature stage", a_res, res)
        check("C3", f"d={d}: R0 at fixed B", a_R0, R0)
        check("C3", f"d={d}: growth rate at fixed B", a_r, r1)
        check("C3", f"d={d}: B needed for R0 = 1.2", a_B, Bneed)
        check("C3", f"d={d}: growth rate at fixed R0", a_r2, r2)
    print(f"      bracketed root of the renewal equation against the quadratic formula, largest difference: {worst:.1e}")
    for d, qR0, qr, qB in ((2, 1.154, 0.05093, 1.248), (10, 1.000, 0.0, 1.440), (50, 0.600, -0.01562, 2.400)):
        check_text("C3", f"text: R0 at d={d}", qR0, 1e-3, mine[d][0])
        check_text("C3", f"text: growth rate at d={d}", qr, 1e-5, mine[d][1])
        check_text("C3", f"text: B needed at d={d}", qB, 1e-3, mine[d][2])

    print("  (b) Death-only shock: survival multiplied by min(1, exp(sigma Z)), Z standard normal.")
    aud_s = {0.05: 0.980661646, 0.20: 0.929239808, 0.50: 0.849618835}
    worst = 0.0
    mine_s = {}
    for sigma, a_m in aud_s.items():
        low, _ = integrate.quad(lambda z: math.exp(sigma * z) * gauss_density(z), -14.0, 0.0, points=[-4.0],
                                epsabs=1e-13, epsrel=1e-12, limit=500)
        m_quad = low + 0.5
        m_closed = 0.5 + math.exp(0.5 * sigma * sigma) * special.ndtr(-sigma)
        worst = max(worst, abs(m_quad - m_closed))
        mine_s[sigma] = m_quad
        check("C3", f"shock, sigma={sigma:.2f}: mean multiplier", a_m, m_quad)
    print(f"      quadrature against the closed form 1/2 + exp(sigma^2/2) Phi(-sigma), largest difference: {worst:.1e}")
    check_text("C3", "text: mean multiplier at sigma=0.2", 0.92924, 1e-5, mine_s[0.20])
    check_text("C3", "text: 'a 7.1% reduction' at sigma=0.2", 0.071, 1e-3, 1.0 - mine_s[0.20])
    print("      loss / (sigma/sqrt(2 pi)), the claimed leading term: "
          + ", ".join(f"sigma={s:g}: {(1 - (0.5 + math.exp(0.5 * s * s) * special.ndtr(-s))) / (s / math.sqrt(2 * math.pi)):.4f}"
                      for s in (0.01, 0.05, 0.2, 0.5)))


# --------------------------------------------------------------------------------------
# rings with regulated transfers (C4 and E-b)
# --------------------------------------------------------------------------------------
# Model: dx_{k+1}/dt = b_k a_k f_k(x_k) x_k - a_{k+1} x_{k+1}, indices modulo n; R0 = prod b_k.
# f_k(x) = F(x / K_k); F = 1 for a stage that carries no regulation (1/K_k = 0).
#   BH        F(y) = 1/(1+y)            1 - kappa = F
#   EXP       F(y) = exp(-y)            kappa = y
#   HASSELL2  F(y) = (1+y)^-2           kappa = 2y/(1+y)
#   MSS2      F(y) = (1+y^2)^(-1/2)     1 - kappa = F^2
# kappa = -d ln f / d ln x at the equilibrium. Share of stage k: w_k = -ln f_k(x_k*) / ln R0 (they sum to one).

def f_val(form, y):
    if form == "BH":
        return 1.0 / (1.0 + y)
    if form == "EXP":
        return np.exp(-y)
    if form == "HASSELL2":
        return 1.0 / ((1.0 + y) * (1.0 + y))
    if form == "MSS2":
        return 1.0 / np.sqrt(1.0 + y * y)
    raise ValueError(form)


def ln_f(form, y):
    if form == "BH":
        return -np.log1p(y)
    if form == "EXP":
        return -y
    if form == "HASSELL2":
        return -2.0 * np.log1p(y)
    if form == "MSS2":
        return -0.5 * np.log1p(y * y)
    raise ValueError(form)


def kappa_of(form, y):
    if form == "BH":
        return y / (1.0 + y)
    if form == "EXP":
        return y
    if form == "HASSELL2":
        return 2.0 * y / (1.0 + y)
    if form == "MSS2":
        return y * y / (1.0 + y * y)
    raise ValueError(form)


def y_from_logsupp(form, L):
    """The value of x/K at which -ln F = L."""
    if form == "BH":
        return math.expm1(L)
    if form == "EXP":
        return L
    if form == "HASSELL2":
        return math.expm1(0.5 * L)
    if form == "MSS2":
        return math.sqrt(math.expm1(2.0 * L))
    raise ValueError(form)


def ring_rhs(x, a, b, invK, form):
    out = b * a * f_val(form, x * invK) * x
    return np.roll(out, 1) - a * x


def jac_complex_step(x, a, b, invK, form):
    n = x.size
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-20 * x[j]
        xc = x.astype(complex)
        xc[j] += 1j * h
        J[:, j] = ring_rhs(xc, a, b, invK, form).imag / h
    return J


def jac_formula(x, a, b, invK, form):
    n = x.size
    y = x * invK
    J = np.diag(-a.astype(float))
    g = b * a * f_val(form, y) * (1.0 - kappa_of(form, y))
    for k in range(n):
        J[(k + 1) % n, k] += g[k]
    return J


def return_map(s, a, lnb, invK, form):
    """Log of the flux out of stage 0 after one turn of the ring, minus its starting value s."""
    lp = np.array(s, dtype=float, copy=True)
    s0 = lp.copy()
    for k in range(a.size):
        if invK[k] != 0.0:
            y = np.exp(lp - math.log(a[k])) * invK[k]
            lp = lp + lnb[k] + ln_f(form, y)
        else:
            lp = lp + lnb[k]
    return lp - s0


def x_from_s(s, a, lnb, invK, form):
    n = a.size
    x = np.empty(n)
    lp = float(s)
    for k in range(n):
        x[k] = math.exp(lp) / a[k]
        lp += lnb[k] + float(ln_f(form, x[k] * invK[k]))
    return x


def newton_polish(x, a, b, invK, form, iters=4):
    for _ in range(iters):
        F = ring_rhs(x, a, b, invK, form)
        J = jac_complex_step(x, a, b, invK, form)
        try:
            xn = x - np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(xn)) or np.any(xn <= 0.0):
            break
        x = xn
    return x


S_GRID = np.linspace(-30.0137, 30.0211, 12001)


def find_equilibria(a, lnb, invK, form):
    """All positive equilibria seen by a scan of the one-turn map on a grid of 12001 starting fluxes
    (a lower bound on their number), each refined by bracketing and then by Newton steps on the full system."""
    b = np.exp(lnb)
    with np.errstate(over="ignore", invalid="ignore"):
        G = return_map(S_GRID, a, lnb, invK, form)
    sg = np.sign(G)
    out = []
    for i in np.nonzero(sg[:-1] * sg[1:] < 0)[0]:
        r = optimize.brentq(lambda t: float(return_map(np.array([t]), a, lnb, invK, form)[0]),
                            S_GRID[i], S_GRID[i + 1], xtol=1e-14, rtol=1e-14)
        out.append(newton_polish(x_from_s(r, a, lnb, invK, form), a, b, invK, form))
    return out


def design_for_shares(form, a, lnb, w, lnR0, ln_scale):
    """Carrying constants K_k for which an equilibrium with shares w exists, and that equilibrium
    (used only to recognise it among the equilibria found numerically)."""
    n = a.size
    lnphi = np.empty(n)
    lnphi[0] = ln_scale
    for k in range(n - 1):
        lnphi[k + 1] = lnphi[k] + lnb[k] - w[k] * lnR0
    x = np.exp(lnphi) / a
    y = np.array([y_from_logsupp(form, wk * lnR0) if wk > 0.0 else 0.0 for wk in w])
    return y / x, x


def target_equilibrium(form, a, lnb, w, lnR0, ln_scale):
    invK, x_t = design_for_shares(form, a, lnb, w, lnR0, ln_scale)
    eqs = find_equilibria(a, lnb, invK, form)
    best = None
    for x in eqs:
        d = float(np.max(np.abs(np.log(x) - np.log(x_t))))
        if best is None or d < best[0]:
            best = (d, x)
    fallback = False
    if best is None or best[0] > 1e-6:
        fallback = True
        x = newton_polish(x_t * (1.0 + 1e-4 * (-1.0) ** np.arange(a.size)), a, np.exp(lnb), invK, form, iters=8)
    else:
        x = best[1]
    return x, invK, len(eqs), fallback


def analyse(x, a, lnb, invK, form):
    b = np.exp(lnb)
    J = jac_complex_step(x, a, b, invK, form)
    Jf = jac_formula(x, a, b, invK, form)
    ev = np.linalg.eigvals(J)
    res = float(np.max(np.abs(ring_rhs(x, a, b, invK, form)) / (a * x)))
    jerr = float(np.max(np.abs(J - Jf)) / np.max(np.abs(Jf)))
    loop = float(np.prod(1.0 - kappa_of(form, x * invK)))
    return ev, loop, res, jerr


def char_newton_distance(ev, a, L):
    """Size of one Newton step from each eigenvalue towards a root of prod(1 + z/a_k) = L:
    an estimate of the distance from the computed eigenvalue to an exact root."""
    worst = 0.0
    for z in ev:
        terms = 1.0 + z / a
        P = np.prod(terms)
        dP = sum(np.prod(np.delete(terms, k)) / a[k] for k in range(a.size))
        if dP != 0:
            worst = max(worst, abs((P - L) / dP))
    return worst


def spectrum_distance(e1, e2):
    cost = np.abs(e1[:, None] - e2[None, :])
    r, c = optimize.linear_sum_assignment(cost)
    return float(cost[r, c].max())


def example_c4():
    banner("C4. Exponential against Beverton-Holt regulation of the transfers")
    print("  Ring dx_{k+1}/dt = b_k a_k f_k(x_k) x_k - a_{k+1} x_{k+1}; four stages, a_k = 1, b_k = e^1.5, ln R0 = 6.")
    print("  Regulation all in one stage, or in equal shares (each stage a quarter of ln R0 at the equilibrium).")
    print("  Equilibrium found numerically; Jacobian by complex-step differentiation of the right-hand side.")
    a = np.ones(4)
    lnb = np.full(4, 1.5)
    lnR0 = 6.0
    aud = {("EXP", "one"): (-5.0, 0.057371263), ("EXP", "equal"): (0.0625, -0.5),
           ("BH", "one"): (0.002478752, -0.776869840), ("BH", "equal"): (0.002478752, -0.776869840)}
    shares = {"one": np.array([1.0, 0.0, 0.0, 0.0]), "equal": np.full(4, 0.25)}
    name = {"EXP": "exponential", "BH": "Beverton-Holt"}
    mine = {}
    worst_closed = 0.0
    for (form, alloc), (a_loop, a_re) in aud.items():
        x, invK, neq, fb = target_equilibrium(form, a, lnb, shares[alloc], lnR0, 0.3)
        ev, loop, res, jerr = analyse(x, a, lnb, invK, form)
        maxre = float(ev.real.max())
        mine[(form, alloc)] = maxre
        roots = -1.0 + complex(loop) ** 0.25 * np.exp(0.5j * math.pi * np.arange(4))
        worst_closed = max(worst_closed, abs(maxre - float(roots.real.max())))
        check("C4", f"{name[form]}, {alloc}: loop gain prod(1-kappa)", a_loop, loop)
        check("C4", f"{name[form]}, {alloc}: largest real part", a_re, maxre)
        print(f"        equilibria found {neq}; residual {res:.1e}; Jacobian against formula {jerr:.1e}")
    print(f"      against the closed form -1 + L^(1/4) x (fourth roots of unity), largest difference: {worst_closed:.1e}")
    check_text("C4", "text: exponential, all in one stage", 0.057371, 1e-6, mine[("EXP", "one")])
    check_text("C4", "text: exponential, equally shared", -0.500000, 1e-6, mine[("EXP", "equal")])
    check_text("C4", "text: Beverton-Holt, all in one stage", -0.776870, 1e-6, mine[("BH", "one")])
    check_text("C4", "text: Beverton-Holt, equally shared", -0.776870, 1e-6, mine[("BH", "equal")])
    print("  Extra, not in the auditor's report: two other saturating-or-not forms on the same ring.")
    for form, label in (("HASSELL2", "f = (1+x/K)^-2 (over-compensating)"), ("MSS2", "f = (1+(x/K)^2)^(-1/2) (compensating)")):
        vals = []
        for alloc in ("one", "equal"):
            x, invK, neq, fb = target_equilibrium(form, a, lnb, shares[alloc], lnR0, 0.3)
            ev, loop, res, jerr = analyse(x, a, lnb, invK, form)
            vals.append((loop, float(ev.real.max())))
        print(f"      {label:<40s} one stage: loop gain {vals[0][0]:+.6e}, max Re {vals[0][1]:+.6f};"
              f"  equal shares: loop gain {vals[1][0]:+.6e}, max Re {vals[1][1]:+.6f}")


def extension_b(rng):
    banner("E-b. Allocation of regulation on random rings")
    print("  Model: dx_{k+1}/dt = b_k a_k f_k(x_k) x_k - a_{k+1} x_{k+1}, indices modulo n; R0 = prod b_k.")
    print("  f_k(x) = 1/(1 + x/K_k) (Beverton-Holt) or exp(-x/K_k) (exponential); f_k = 1 where a stage carries no regulation.")
    print("  Rings: n = 2..6, 60 rings for each n (300 in all). a_k = 10^U(-1,1) independently (two decades);")
    print("  ln R0 ~ U(0.2, 8); ln b_k = ln R0/n + (g_k - mean g), g_k standard normal (so single b_k may be below one).")
    print("  Share of stage k at an equilibrium: w_k = -ln f_k(x_k*)/ln R0; the shares sum to one.")
    print("  Allocations for each ring: A all in one stage (stage drawn at random); B equal shares;")
    print("  C random shares on all stages (flat Dirichlet); D uneven random shares (Dirichlet 0.3) on a random subset of")
    print("  at least two stages; E carrying constants K_k = 10^U(-1,1) on a random non-empty subset, shares not chosen.")
    print("  For A-D the K_k are set so that an equilibrium with those shares exists, with a random overall scale.")
    print("  The equilibrium is then found without using that knowledge: scan of the one-turn map of the flux on")
    print("  12001 starting values, bracketing, Newton steps on the full system; the root with the target shares is")
    print("  picked out afterwards. Jacobian: complex-step differentiation of the right-hand side (checked against")
    print("  the formula). Eigenvalues: numpy. Two spectra are compared by the best one-to-one matching.")
    ns = [2, 3, 4, 5, 6]
    per_n = 60
    alloc = ["A", "B", "C", "D"]
    stat = {f: {n: {al: {"unst": 0, "real": 0, "osc": 0, "multi": 0} for al in alloc + ["E"]} for n in ns} for f in ("BH", "EXP")}
    bh_abs, bh_rel, bh_newton, bh_neq, exp_rel = [], [], [], [], []
    exp_newton = []
    worst_res = worst_jerr = worst_share = 0.0
    fallbacks = 0
    missing_E = 0
    expE = {n: {"multi": 0, "all_unst": 0, "any_unst": 0} for n in ns}
    lnR0_gt4_n2 = 0
    samples = []
    for n in ns:
        for ring in range(per_n):
            a = 10.0 ** rng.uniform(-1.0, 1.0, n)
            lnR0 = rng.uniform(0.2, 8.0)
            g = rng.normal(size=n)
            lnb = lnR0 / n + (g - g.mean())
            if n == 2 and lnR0 > 4.0:
                lnR0_gt4_n2 += 1
            wA = np.zeros(n)
            wA[rng.integers(n)] = 1.0
            wB = np.full(n, 1.0 / n)
            wC = rng.dirichlet(np.ones(n))
            m = n if n == 2 else int(rng.integers(2, n))          # subset size 2..n-1 (n for n = 2)
            sub = rng.choice(n, size=m, replace=False)
            wD = np.zeros(n)
            wD[sub] = rng.dirichlet(np.full(m, 0.3))
            mE = int(rng.integers(1, n + 1))
            subE = rng.choice(n, size=mE, replace=False)
            invK_E = np.zeros(n)
            invK_E[subE] = 10.0 ** rng.uniform(-1.0, 1.0, mE)
            shares = {"A": wA, "B": wB, "C": wC, "D": wD}
            for form in ("BH", "EXP"):
                spectra = {}
                for al in alloc:
                    x, invK, neq, fb = target_equilibrium(form, a, lnb, shares[al], lnR0, rng.uniform(-3.0, 3.0))
                    fallbacks += int(fb)
                    ev, loop, res, jerr = analyse(x, a, lnb, invK, form)
                    worst_res, worst_jerr = max(worst_res, res), max(worst_jerr, jerr)
                    w_real = -ln_f(form, x * invK) / lnR0
                    worst_share = max(worst_share, float(np.max(np.abs(w_real - shares[al]))))
                    spectra[al] = ev
                    st = stat[form][n][al]
                    if neq > 1:
                        st["multi"] += 1
                    k = int(np.argmax(ev.real))
                    if ev[k].real > 1e-9 * a.max():
                        st["unst"] += 1
                        if abs(ev[k].imag) > 1e-9 * a.max():
                            st["osc"] += 1
                        else:
                            st["real"] += 1
                    if form == "BH":
                        bh_neq.append(neq)
                        bh_newton.append(char_newton_distance(ev, a, math.exp(-lnR0)) / a.max())
                    else:
                        L = float(np.prod(1.0 - shares[al] * lnR0))
                        exp_newton.append(char_newton_distance(ev, a, L) / a.max())
                # allocation E
                eqs = find_equilibria(a, lnb, invK_E, form)
                if not eqs:
                    missing_E += 1
                    continue
                if form == "BH":
                    bh_neq.append(len(eqs))
                    ev, loop, res, jerr = analyse(eqs[0], a, lnb, invK_E, form)
                    worst_res, worst_jerr = max(worst_res, res), max(worst_jerr, jerr)
                    spectra["E"] = ev
                    bh_newton.append(char_newton_distance(ev, a, math.exp(-lnR0)) / a.max())
                    if ev.real.max() > 1e-9 * a.max():
                        stat["BH"][n]["E"]["unst"] += 1
                    d = max(spectrum_distance(spectra["A"], spectra[al]) for al in ("B", "C", "D", "E"))
                    bh_abs.append(d)
                    bh_rel.append(d / a.max())
                    if ring == 0:
                        samples.append((n, lnR0, a.copy(), {al: float(spectra[al].real.max()) for al in spectra}))
                else:
                    flags = []
                    for xe in eqs:
                        ev, loop, res, jerr = analyse(xe, a, lnb, invK_E, form)
                        flags.append(ev.real.max() > 1e-9 * a.max())
                    if len(eqs) > 1:
                        expE[n]["multi"] += 1
                    if flags and all(flags):
                        expE[n]["all_unst"] += 1
                    if any(flags):
                        expE[n]["any_unst"] += 1
                    d = max(spectrum_distance(spectra["A"], spectra[al]) for al in ("B", "C", "D"))
                    exp_rel.append(d / a.max())
    nr = len(ns) * per_n
    print()
    print(f"  Rings: {nr}. Equilibria built: {nr * 4 * 2} with target shares, plus allocation E.")
    print(f"  Largest equilibrium residual |dx/dt| / (a x): {worst_res:.1e}; largest error of realised against target shares: {worst_share:.1e}")
    print(f"  Largest difference between complex-step Jacobian and formula (relative): {worst_jerr:.1e}")
    print(f"  Target equilibrium not caught by the scan and found by Newton steps from nearby instead: {fallbacks} cases")
    print(f"  Allocation E, no equilibrium inside the scanned range (ring and form left out of the E columns): {missing_E} cases")
    print()
    print("  Beverton-Holt")
    print(f"    positive equilibria found per ring and allocation: min {min(bh_neq)}, max {max(bh_neq)}")
    print(f"    LARGEST DIFFERENCE between the spectra of allocations A, B, C, D, E of one ring: {max(bh_abs):.2e} (absolute),"
          f" {max(bh_rel):.2e} (relative to the largest exit rate)")
    print(f"    largest distance of an eigenvalue from a root of prod(1 + z/a_k) = 1/R0 (relative to the largest exit rate): {max(bh_newton):.2e}")
    print("    first ring of each n, largest real part of an eigenvalue under A, B, C, D, E:")
    for n, lnR0, a, mr in samples:
        print(f"      n={n}, ln R0={lnR0:.4f}, a=[" + ", ".join(f"{v:.3f}" for v in a) + "]: "
              + ", ".join(f"{al} {mr[al]:+.12f}" for al in ("A", "B", "C", "D", "E")))
    print()
    print("  Exponential")
    er = np.array(exp_rel)
    print(f"    difference between the spectra of allocations A, B, C, D of one ring, relative to the largest exit rate:"
          f" median {np.median(er):.3f}, largest {er.max():.3f}, smallest {er.min():.2e}")
    print(f"    largest distance of an eigenvalue from a root of prod(1 + z/a_k) = prod(1 - w_k ln R0) (relative): {max(exp_newton):.2e}")
    print()
    print("  Number of rings (of 60 for each n) at which the equilibrium with the stated shares is unstable.")
    print("  'r' = a positive real eigenvalue (loop gain above one; other equilibria then exist), 'o' = a complex pair.")
    print("           Beverton-Holt       exponential")
    print("     n     A   B   C   D   E   A (one stage)   B (equal)       C (random)      D (uneven subset)")
    tot = {f: {al: 0 for al in alloc + ["E"]} for f in ("BH", "EXP")}
    for n in ns:
        row = f"     {n}   " + " ".join(f"{stat['BH'][n][al]['unst']:3d}" for al in alloc + ["E"]) + "   "
        for al in alloc:
            st = stat["EXP"][n][al]
            row += f"{st['unst']:3d} ({st['real']:2d}r,{st['osc']:2d}o)  "
            tot["EXP"][al] += st["unst"]
        for al in alloc + ["E"]:
            tot["BH"][al] += stat["BH"][n][al]["unst"]
        print(row)
    print("    all   " + " ".join(f"{tot['BH'][al]:3d}" for al in alloc + ["E"]) + "   "
          + "".join(f"{tot['EXP'][al]:3d} of {nr}".ljust(16) for al in alloc))
    print(f"    For comparison: with n = 2 and equal shares the manuscript's condition is ln R0 > 4; rings with n = 2 and ln R0 > 4: {lnR0_gt4_n2}")
    print()
    print("  Exponential: rings at which the scan found more than one positive equilibrium (a lower bound)")
    print("     n     A    B    C    D    E   |  E: every equilibrium found unstable / at least one unstable")
    for n in ns:
        print(f"     {n}   " + " ".join(f"{stat['EXP'][n][al]['multi']:4d}" for al in alloc)
              + f" {expE[n]['multi']:4d}   |  {expE[n]['all_unst']:3d} / {expE[n]['any_unst']:3d}")


# --------------------------------------------------------------------------------------
# C5
# --------------------------------------------------------------------------------------
def c5_q(b1, b2):
    """Smallest root of q = exp(b1 (exp(b2 (q-1)) - 1)): extinction from one stage-1 founder."""
    F = lambda q: math.exp(b1 * math.expm1(b2 * (q - 1.0))) - q
    q1 = optimize.brentq(F, 0.0, 1.0 - 1e-9, xtol=1e-16, rtol=1e-15)
    return q1, math.exp(b2 * (q1 - 1.0))


def branching_mc(b1, b2, n1_start, reps, rng, s=0.5, cap=100_000, max_steps=200_000):
    """Individual-based process in discrete time. Each individual stays with probability s; on leaving,
    a stage-1 individual leaves Poisson(b1) stage-2 individuals and a stage-2 individual Poisson(b2)
    stage-1 individuals. A run is stopped as surviving when the population reaches `cap`."""
    n1 = np.full(reps, n1_start, dtype=np.int64)
    n2 = np.zeros(reps, dtype=np.int64)
    extinct = escaped = steps = 0
    while n1.size and steps < max_steps:
        e1 = rng.binomial(n1, 1.0 - s)
        e2 = rng.binomial(n2, 1.0 - s)
        n1 = n1 - e1 + rng.poisson(b2 * e2)
        n2 = n2 - e2 + rng.poisson(b1 * e1)
        tot = n1 + n2
        dead = tot == 0
        big = tot >= cap
        extinct += int(dead.sum())
        escaped += int(big.sum())
        keep = ~(dead | big)
        n1, n2 = n1[keep], n2[keep]
        steps += 1
    return extinct, escaped, int(n1.size), steps


def example_c5(rng):
    banner("C5. The same mean life cycle, different extinction probabilities")
    print("  Two stages; each individual stays with probability 0.5 per step, then leaves and produces Poisson(b_k)")
    print("  of the next stage. b1 b2 = 2 in all three cases; the mean matrix is [[0.5, b2/2], [b1/2, 0.5]].")
    print("  Extinction from one stage-1 founder: smallest root of q = exp(b1 (exp(b2 (q-1)) - 1)).")
    cases = [("b1=b2=sqrt2", math.sqrt(2.0), math.sqrt(2.0), 0.477889492048, 8.565409388e-33),
             ("b1=0.02,b2=100", 0.02, 100.0, 0.984275771848, 2.049649018e-01),
             ("b1=100,b2=0.02", 100.0, 0.02, 0.207541737456, 5.135051637e-69)]
    qs = {}
    for label, b1, b2, a_q, a_e in cases:
        M = np.array([[0.5, 0.5 * b2], [0.5 * b1, 0.5]])
        lam = float(max(np.linalg.eigvals(M).real))
        q1, q2 = c5_q(b1, b2)
        qs[label] = (q1, q2)
        check("C5", f"{label}: growth factor of the mean matrix", 1.207106781, lam)
        check("C5", f"{label}: q, one stage-1 founder", a_q, q1)
        check("C5", f"{label}: extinction, 100 stage-1 founders", a_e, math.exp(100.0 * math.log(q1)))
    check_text("C5", "text: 0.205 (bottleneck cycle, 100 founders)", 0.205, 1e-3, math.exp(100.0 * math.log(qs["b1=0.02,b2=100"][0])))
    check_text("C5", "text: 8.6e-33 (symmetric cycle, 100 founders)", 8.6e-33, 1e-34, math.exp(100.0 * math.log(qs["b1=b2=sqrt2"][0])))
    print("  Simulation of the individual-based process (runs stopped as surviving at 100,000 individuals;")
    print("  from there extinction has probability below 0.999^100000 for any q below 0.999):")
    for label, b1, b2, a_q, a_e in cases:
        reps = 400_000
        ext, esc, unresolved, steps = branching_mc(b1, b2, 1, reps, rng)
        p = ext / reps
        se = math.sqrt(p * (1.0 - p) / reps)
        check_mc("C5", f"{label}: q by simulation ({reps} runs)", a_q, p, se)
        if unresolved:
            print(f"        WARNING: {unresolved} runs unresolved after {steps} steps")
    reps = 200_000
    ext, esc, unresolved, steps = branching_mc(0.02, 100.0, 100, reps, rng)
    p = ext / reps
    se = math.sqrt(p * (1.0 - p) / reps)
    check_mc("C5", f"b1=0.02,b2=100: 100 founders by simulation ({reps} runs)", 2.049649018e-01, p, se)
    if unresolved:
        print(f"        WARNING: {unresolved} runs unresolved after {steps} steps")
    print("  Extra, not in the auditor's report: where the 100 founders are placed.")
    print("    case              100 in stage 1   100 in stage 2   100 placed at the stable stage distribution")
    for label, b1, b2, a_q, a_e in cases:
        q1, q2 = qs[label]
        u1 = math.sqrt(b2) / (math.sqrt(b1) + math.sqrt(b2))
        mix = u1 * q1 + (1.0 - u1) * q2
        print(f"    {label:<16s}  {math.exp(100 * math.log(q1)):.3e}        {math.exp(100 * math.log(q2)):.3e}        "
              f"{math.exp(100 * math.log(mix)):.3e}   (stage-1 fraction {u1:.4f})")

    print("  Host-level packaging: 10,000 parasites in m hosts; each host (with all its parasites) survives a gap")
    print("  with probability 0.1, independently. P(extinction) = 0.9^m, in exact rational arithmetic.")
    for m, a_p, a_log in ((10, 0.34867844, -1.053605), (100, 2.65613989e-05, -10.536052), (10000, 0.0, -1053.605157)):
        exact = Fraction(9, 10) ** m
        p = exact.numerator / exact.denominator
        check("C5", f"packages m={m}: P(extinction)", a_p, p)
        check("C5", f"packages m={m}: log P(extinction)", a_log, m * math.log(0.9))
    print(f"      m=10000: log10 P = {10000 * math.log10(0.9):.4f} (below the smallest double; printed as zero)")


# --------------------------------------------------------------------------------------
# C6
# --------------------------------------------------------------------------------------
def c6_alternating_exact(K=100, pairs=50, bad_first=True):
    """P(extinct within `pairs` pairs of steps | start at K), exactly, as an integer over 10^(K pairs).
    Backward recursion on u_t(n) = P(extinct within t more pairs | n now)."""
    W = [[math.comb(n, j) * 9 ** j * 10 ** (K - n) for j in range(n + 1)] for n in range(K + 1)]  # Bin(n,0.9;j) x 10^K
    U = [1] + [0] * K
    for t in range(pairs):
        new = [0] * (K + 1)
        new[0] = 10 ** (K * (t + 1))
        for n in range(1, K + 1):
            if bad_first:     # bad step on n, then doubling with cap
                new[n] = sum(W[n][j] * U[min(2 * j, K)] for j in range(n + 1))
            else:             # doubling with cap, then bad step
                m = min(2 * n, K)
                new[n] = sum(W[m][j] * U[j] for j in range(m + 1))
        U = new
    return U[K], 10 ** (K * pairs)


def example_c6(rng):
    banner("C6. The same invasion exponent, different survival of a finite population")
    print("  Ceiling K = 100. A good step doubles N (capped at K); a bad step leaves Binomial(N, 0.9) survivors; zero is")
    print("  absorbing. Start at K. Schedules: 50 pairs (bad, good), against 50 bad steps followed by 50 good ones.")
    r = 0.5 * (math.log(2.0) + math.log(0.9))
    check("C6", "growth exponent when rare (both schedules)", 0.293893332, r)
    p_block = math.exp(100.0 * math.log1p(-(0.9 ** 50)))
    check("C6", "long block: extinction by step 100", 0.596479639, p_block)
    print("      closed form (1 - 0.9^50)^100: after 50 bad steps each of the 100 is alive with probability 0.9^50.")
    num, den = c6_alternating_exact()
    p_alt = num / den
    check("C6", "alternating: extinction by step 100", 1.797407701e-94, p_alt)
    print("      exact integer arithmetic, backward recursion over 50 pairs; the numerator is about "
          f"10^{math.log10(num):.2f}, over 10^5000.")
    num2, den2 = c6_alternating_exact(bad_first=False)
    print(f"      if each pair is read as (good, bad) instead: {num2 / den2:.9e}")
    check_text("C6", "text: 0.596 (long block)", 0.596, 1e-3, p_block)
    reps = 400_000
    n = np.full(reps, 100, dtype=np.int64)
    for _ in range(50):
        n = rng.binomial(n, 0.9)
    p = float(np.mean(n == 0))
    se = math.sqrt(p * (1.0 - p) / reps)
    check_mc("C6", f"long block by simulation ({reps} runs)", 0.596479639, p, se)
    n = np.full(reps, 100, dtype=np.int64)
    dead = np.zeros(reps, dtype=bool)
    for _ in range(50):
        n = rng.binomial(n, 0.9)
        dead |= n == 0
        n = np.minimum(2 * n, 100)
    print(f"      alternating by simulation: {int(dead.sum())} extinctions in {reps} runs (cannot resolve 1e-94; consistent only)")
    print(f"      expected number alive after the 50 bad steps of the long block: {100 * 0.9 ** 50:.4f}")


# --------------------------------------------------------------------------------------
# C7 and E-c
# --------------------------------------------------------------------------------------
def surv_fresh(shape, mean, t):
    """P(lifetime > t), lifetime gamma with integer shape and the given mean: a finite sum."""
    x = shape * t / mean
    return math.exp(-x) * math.fsum(x ** j / math.factorial(j) for j in range(shape))


def surv_stationary(shape, mean, t):
    """P(remaining lifetime > t) for an individual of stationary age (equilibrium excess life):
    (1/mean) x integral from t of the survival function = (1/shape) sum_{j<shape} P(Poisson(x) <= j)."""
    x = shape * t / mean
    cum, term, tot = 0.0, math.exp(-x), 0.0
    for j in range(shape):
        if j > 0:
            term *= x / j
        cum += term
        tot += cum
    return tot / shape


def p_any(S, N):
    if S <= 0.0:
        return 0.0
    return -math.expm1(N * math.log1p(-S)) if S < 1.0 else 1.0


def example_c7():
    banner("C7. The same mean residence, different chances of bridging a gap")
    print("  100 stored parasites, no reproduction or transfer during a gap of 60, independent deaths, mean lifetime 20.")
    print("  'Fresh' = all newly entered at the start of the gap; 'stationary' = ages drawn from the stationary age")
    print("  distribution, so that remaining life has the equilibrium excess-life distribution.")
    aud = {1: (0.049787068, 0.993945286, 0.993945286), 4: (0.002291791, 0.205023034, 0.069960296)}
    worst = 0.0
    mine = {}
    for shape, (a_s, a_f, a_st) in aud.items():
        S = surv_fresh(shape, 20.0, 60.0)
        Se = surv_stationary(shape, 20.0, 60.0)
        S2 = float(special.gammaincc(shape, shape * 60.0 / 20.0))
        Se2 = float(special.gammaincc(shape + 1, 3.0 * shape) - 3.0 * special.gammaincc(shape, 3.0 * shape))
        Se3 = integrate.quad(lambda u: float(special.gammaincc(shape, shape * u / 20.0)), 60.0, 1060.0,
                             points=[65.0, 80.0, 140.0, 300.0], epsabs=0.0, epsrel=1e-10, limit=400)[0] / 20.0
        worst = max(worst, abs(S - S2) / S, abs(Se - Se2) / Se, abs(Se - Se3) / Se)
        mine[shape] = (p_any(S, 100), p_any(Se, 100))
        check("C7", f"shape {shape}: survival of one fresh parasite", a_s, S)
        check("C7", f"shape {shape}: P(any of 100 fresh survive)", a_f, p_any(S, 100))
        check("C7", f"shape {shape}: P(any of 100 stationary survive)", a_st, p_any(Se, 100))
    check("C7", "fixed lifetime 20: P(any of 100 fresh survive)", 0.0, p_any(0.0, 100))
    check("C7", "fixed lifetime 20: P(any of 100 stationary survive)", 0.0, p_any(max(0.0, 1.0 - 60.0 / 20.0), 100))
    print(f"      finite sums against the incomplete gamma function and against quadrature, largest relative difference: {worst:.1e}")
    check_text("C7", "text: exponential, fresh", 0.99395, 1e-5, mine[1][0])
    check_text("C7", "text: shape 4, fresh", 0.20502, 1e-5, mine[4][0])
    check_text("C7", "text: shape 4, stationary ages", 0.06996, 1e-5, mine[4][1])
    print(f"      exact forms: shape 4 fresh survival = 373 e^-12 = {373 * math.exp(-12):.9e}; stationary = 118 e^-12 = {118 * math.exp(-12):.9e}")


def extension_c():
    banner("E-c. Survival through a gap, by distribution of the residence time (closed forms, no simulation)")
    print("  Mean residence 20 in every row. Lifetime gamma with shape 1 (exponential), 2, 4, 16, or fixed at 20.")
    print("  S = probability that one stored parasite is still alive at the end of the gap (lifetime strictly longer")
    print("  than the gap; a fixed lifetime of 20 therefore fails a gap of exactly 20, and survives any shorter gap).")
    print("  Fresh: S = Q(shape, shape x gap/20), the upper incomplete gamma ratio, here a finite sum.")
    print("  Stationary ages: S = (1/20) x integral of the fresh S from the gap onwards = (1/shape) sum_{j<shape} P(Poisson <= j);")
    print("  for a fixed lifetime the remaining life is uniform on (0, 20), S = max(0, 1 - gap/20).")
    print("  P(any of N) = 1 - (1 - S)^N.")
    shapes = [1, 2, 4, 16, None]
    worst = 0.0
    for gap in (20.0, 40.0, 60.0, 100.0, 10.0):
        extra = "   (extra row, shorter than the mean)" if gap == 10.0 else ""
        print(f"  gap {gap:g}{extra}")
        print("    shape    fresh: S        any of 100   any of 10,000  |  stationary: S   any of 100   any of 10,000")
        for shape in shapes:
            if shape is None:
                S = 1.0 if gap < 20.0 else 0.0
                Se = max(0.0, 1.0 - gap / 20.0)
                name = "fixed"
            else:
                S = surv_fresh(shape, 20.0, gap)
                Se = surv_stationary(shape, 20.0, gap)
                x = shape * gap / 20.0
                S2 = float(special.gammaincc(shape, x))
                Se3 = integrate.quad(lambda u: float(special.gammaincc(shape, shape * u / 20.0)), gap, gap + 1000.0,
                                     points=[gap + 5.0, gap + 20.0, gap + 80.0, gap + 240.0],
                                     epsabs=0.0, epsrel=1e-10, limit=400)[0] / 20.0
                worst = max(worst, abs(S - S2) / S, abs(Se - Se3) / Se)
                name = str(shape)
            print(f"    {name:>5s}    {fmt_p(S):>10s}    {fmt_p(p_any(S, 100)):>10s}    {fmt_p(p_any(S, 10000)):>10s}     |  "
                  f"{fmt_p(Se):>10s}    {fmt_p(p_any(Se, 100)):>10s}    {fmt_p(p_any(Se, 10000)):>10s}")
    print(f"  finite sums against the incomplete gamma function (fresh) and against quadrature (stationary),")
    print(f"  largest relative difference over the table: {worst:.1e}")


# --------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Independent reproduction of the auditor's examples C1 to C7, with extensions E-a, E-b, E-c")
    print(f"Started {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} (system clock)")
    print(f"Python {sys.version.split()[0]}; numpy {np.__version__}; scipy {scipy.__version__}; seed {SEED}")
    print("Tolerances, fixed before the first run: deterministic 1e-5 absolute (and 1e-6 relative where the auditor's")
    print("number is below 1e-3); text figures must be the rounding of mine; Monte Carlo 4 standard errors.")
    print("'auditor' = number in the auditor's printed output; 'text' = number quoted in the auditor's report.")
    rng = np.random.default_rng(SEED)
    example_c1()
    example_c2()
    example_c3()
    example_c4()
    example_c5(rng)
    example_c6(rng)
    example_c7()
    extension_a()
    extension_b(np.random.default_rng(SEED + 1))
    extension_c()

    banner("Summary of the reproduction")
    n_fail = 0
    for ex in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        rows = [ok for e, _, ok in RESULTS if e == ex]
        bad = [lab for e, lab, ok in RESULTS if e == ex and not ok]
        n_fail += len(bad)
        print(f"  {ex}: {sum(rows)} of {len(rows)} comparisons pass" + ("" if not bad else "; FAIL: " + "; ".join(bad)))
    print(f"  Total: {len(RESULTS) - n_fail} of {len(RESULTS)} pass; {n_fail} fail.")
    print(f"  Run time {time.time() - t0:.1f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
