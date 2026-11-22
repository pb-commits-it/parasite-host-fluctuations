"""Two published models set against the project's second-order formulas: Watts et al. (2021) and White et al. (2016).

Written 14 November 2026.  Both papers were read in full for this check (the public-access PDF of Watts et al.;
the Europe PMC full text, the equation images and the electronic supplement of White et al.).  Where a number
below is "read", it was read in the paper; where it is "inferred", the reason is given.

PART A.  Watts, Mishra, Nguyen & Tuong (2021), a host-vector loop under random switching.
  Linearised at the disease-free state (their eq. 22), with x = (infectious hosts, infectious vectors):
      dx/dt = [[-gamma1(s), A(s)], [C(s), -mu2]] x,   A = beta*omega1/mu1,  C = sigma*omega2/mu2,  gamma1 = gamma + mu1,
  s a two-state Markov chain with generator Q_eps = [[(u-1)/eps, (1-u)/eps], [u/eps, -u/eps]] (as printed, p. 14),
  so the rate of leaving state 1 is (1-u)/eps and of leaving state 2 is u/eps ("reading A").  The other reading
  ("reading B": rates eps(1-u), eps u) is also computed, because Examples 4.1 and 4.2 use eps = 25 and are
  described with "eps sufficiently small".  Their threshold lambda is the top Lyapunov exponent of this system.
  In the project's notation this is a ring of two stages in continuous time (Result 10): stage 1 the infectious
  host (loss rate gamma1), stage 2 the infectious vector (loss rate mu2), transfer out of stage 1 equal to C and
  out of stage 2 equal to A.
  Parameters.  Example 4.1 (read): omega1 = mu1 = 0.005, omega2 = mu2 = 0.025, beta = (0.15, 0.05),
  sigma(1) = 0.15, gamma = (0.3, 0.3), u = 0.5, eps = 25; sigma(2) = 0.15 is inferred (only beta is said to
  change, and it is what the printed 1.9672 and -0.00038 require).  Example 4.2 (read): beta = (0.15, 0.05),
  sigma = (0.05, 0.15), gamma(1) = 0.3, u = 0.5, eps = 25; gamma(2) = 0.3 is inferred (not printed; it is what the
  printed lambda_2 = -3.7922e-4 requires).  Example 4.3 (read): omega1 = mu1 = omega2 = mu2 = 1,
  beta = (1.1, 2.1), sigma = (1, 2), gamma = (0.01, 3), u = 0.5, eps = 0.05.

  Exact value: (i) the stationary law of the ratio y = x2/x1, which for two states has a closed density (their
  eqs. 27-29; re-derived here from the stationary forward equation), integrated with the end-point
  singularities handled by the quadrature weight; (ii) independently, simulation of the product of matrix
  exponentials over exponential sojourns, with the time spent in state 1 as a control variate.
  Our value: growth rate r0 of the mean-rate matrix plus the second-order term for a driver with exponential
  autocovariance pi1 pi2 exp(-alpha t), alpha = sum of the two switching rates,
      pi1 pi2 [ w'B (alpha + r0 - A0)^(-1) B v - (w'Bv)^2 / alpha ],      B = A_1 - A_2,
  the published second-order formula in continuous time (the matrix form behind eq. 9 of the manuscript), and
  the same thing written in the ring's notation (lineage leaving rates a_k = r0 + loss rate, rho_k = a_k/(a_k+alpha),
  R = prod rho_k, T = sum 1/a_k): own terms of each transfer (eq. 9), the cross term of two transfers on one
  driver (the continuous-time limit of Proposition 4a), the loss-rate term (Proposition 19), and the cross term
  of a loss rate with a transfer (not in the manuscript; derived here from the same lineage resolvent).

  Claims and tolerances, fixed before the first complete run:
  A0  The two exact methods agree: |quadrature - simulation| <= 4 standard errors, six cases (three examples,
      two readings of eps); so does the second simulation code, which Part B needs for larger matrices
      (Example 4.2, 4 standard errors).  And the quadrature approaches both limits at first order: the distance from the
      slow limit (the pi-average of the frozen growth rates) shrinks 10-fold when the rate does (ratio in
      [8, 12]), likewise the distance from the fast limit (the mean matrix).
  A1  The paper's printed frozen thresholds and averaged R0 are reproduced to half a unit of the last printed
      digit: 0.04018 and -0.00038 (Ex. 4.1), -3.7922e-4 twice (Ex. 4.2), 1.9672 (Ex. 4.1).
      Two printed numbers are known beforehand NOT to follow from the printed parameters and are reported as
      "DIFFERS FROM PAPER": the averaged R0 of Example 4.2 (printed 1.967; the printed parameters give
      0.01/0.007625 = 1.3115; 1.967 is Example 4.1's value), and the switched threshold of Example 4.1 (printed
      0.0500, which exceeds the larger frozen value 0.04018; since A_1 >= A_2 entrywise the switched system is
      dominated by the frozen system 1, so no value above 0.04018 is possible).
  A2  The stated behaviour is reproduced by both exact methods under reading A: Ex. 4.1 persistence (lambda > 0);
      Ex. 4.2 both frozen systems die out and the switched one persists at eps = 25; Ex. 4.3 both frozen systems
      persist and the switched one dies out at eps = 0.05.
  A3  The ring-notation closed forms equal the matrix formula on 300 random rings of 2 to 5 stages with random
      transfers and loss rates on one driver (relative 1e-9 of the gross size of the terms).
  A4  Small switching: with A_s replaced by mean + 0.1 (A_s - mean), |exact - (r0 + second-order term)| is at most
      5% of the gross second-order term (the sum of the absolute values of its parts); with 0.05, at most 2%.
      Three examples, reading A.  (For equal switching rates the law is unchanged by exchanging the states, so
      the expansion has even orders only.)
  A5  Example 4.2 (only transfers switch): the reproduction number in the switching environment, R0* (the number
      by which one transfer must be divided for the Lyapunov exponent to vanish), is never below the R0 of
      geometric-mean rates (the "two means" corollary of notes/large-noise.md, lower half), at
      eps = 0.25 ... 2500 (tolerance 1e-9).  Reported, not a pass/fail: whether R0* stays below the R0 of
      arithmetic-mean rates, and the second-order value of ln R0 - ln R0*.
  A7  (added after the second run) For a two-stage ring with both transfers on one host, relative sizes k1, k2,
      the second-order cost per generation and per unit variance is
      [(k1 + k2)^2 x1 x2 - k1 k2 (x1 + x2)^2] / [(x1 + x2)(x1 + x2 + x1 x2)], x_k = alpha/a_k; equal to the general
      closed form on 300 random cases (relative 1e-9).
  A6  A variant that is NOT in the paper: Example 4.3 with the removal rate held at its mean, so that only the
      two transfers switch, together.  Claim under test: the second-order formula says this is a GAIN (the
      switched growth rate exceeds that of the mean-rate matrix), and the exact value agrees in sign by both
      methods (simulation by at least 4 standard errors), at switching rates 10 and 0.1.

PART B.  White, Shirreff, Karl, Ghani & Mueller (2016), a periodic vector and a dormant liver stage.
  Linearised at the disease-free state, the force of infection on people is m(t) a b I_M and nothing else
  depends on time (their eqs. 2.6, S11, S13, S15, S17 and S18):
      m(t)/m0 = eps + (1 - eps) [pi / B(1/2, kappa + 1/2)] [(1 + cos(2 pi t/365))/2]^kappa,    eps = 0.001.
  "Seasonality" is the share of the year's m(t) in the peak three months.  R0 in the seasonal case is 1/xi0,
  xi0 the factor on m(t) at which the dominant Floquet multiplier is one (their section 6.5): the same
  definition as R0* of the manuscript with the one fluctuating transfer counted as the birth.
  Models: 1 (no relapse) is a ring of two stages (blood-stage infection, loss rate r; infectious mosquito, loss
  rate g).  Models 2 and 3 are NOT rings: a liver stage re-seeds blood-stage infection (relapse) without passing
  through the mosquito, so one inoculation gives several blood-stage episodes at different times.
  Parameters (read, Table 1): r = 1/60, gamma_L = 1/223, f = 1/72, gamma_D = 1/434, d = 162 days, K = 11,
  a = 0.21, g = 0.1, n = 12, c = 0.23, b = 0.5.  Inferred: m0 such that R0 of Model 1 is one without seasonality
  (Figure 4a's lowest curve starts at 1.0); "three months" is a quarter of 365 days; Figure 4a's Model 3 has
  gamma-distributed dormancy with K = 11 (the caption does not say; see B1).
  Figure 4a was digitised from the published image (pixel positions of the three curves, axes calibrated on
  the tick marks; error about 0.02 in R0).

  Exact value: (i) the paper's method, Floquet multiplier with root-finding on xi; (ii) independently, the
  spectral radius of the next-generation operator on periodic functions in Fourier space,
  M[k, n] = fhat(k - n) Khat(i n w), K the kernel of new inoculations (the method of Bacaer 2007).
  Our value: Result 7 of the manuscript summed over the harmonics of m(t), with g the transform of the time
  between successive uses of the fluctuating transfer by a line of descent at the threshold (Result 9; for a
  ring this is the time round the cycle):
      ln R0 - ln R0(constant) = sum_n (e_n^2 / 2) Re[ g(n w)/(1 - g(n w)) ],     e_n the amplitude of harmonic n.
  The manuscript states this on the scale of R0 for a ring only; its use for Models 2 and 3 is the extension
  through Result 9 and is itself under test in B2.

  Claims and tolerances.  (Before these were fixed, the exact R0 and the second-order value at the paper's own
  amplitude had been seen for Models 1 and 2 and for Model 3 with exponential dormancy, while reproducing
  Figure 4a.  The small-amplitude checks, the switching runs of B4 and everything in Part A after A2 had not.)
  B0  The forcing function gives a peak-three-month share of 0.35 at kappa = 0.37 and 0.95 at kappa = 12.3
      (both printed in the caption of Figure S4), within 0.005.  The constant-environment R0 from the linear
      system equals the printed closed forms (2.5), (2.7), (2.8) (relative 1e-10).
  B1  (a) The two exact methods agree to relative 1e-5 (four models at seasonality 35, 50, 75, 90%).
      (b) Figure 4a is reproduced: |computed - digitised| <= 0.04 for Models 1, 2 and 3 (gamma dormancy) at
      seasonality 35 to 95%.  (This tolerance was set after seeing that the unrounded parameters gave 0.02.)
      (c) The stated behaviour: R0 falls with seasonality in Model 1, changes little in Model 2 (less than 6% up
      to 95%), and rises in Model 3 (above its constant value at every level from 35 to 95%).
  B2  (a) The second-order coefficient: [D(s) + D(-s)]/(2 s^2) at s = 0.05, D(s) the exact change of ln R0 when
      the departure of m(t)/m0 from one is multiplied by s, is within 1% of the gross second-order sum
      (sum of |terms|) of the formula; and one-sided at s = 0.1, D(0.1) is within 5% of the gross sum times 0.01.
      Four models, four seasonality levels.
      (b) At the paper's amplitude (s = 1): reported; and the sign of the change of R0 with seasonality given by
      the formula agrees with the exact sign for Model 1 (loss) and Model 3 (gain) at every level.
  B3  For Model 1 the kernel's transform equals the ring's, r g / ((r + iw)(g + iw)) (1e-12).  Reported: mean,
      squared coefficient of variation of the generation interval, the slow-host coefficient (1 - CV^2)/2, the
      response at the annual frequency, and the second-order loss of ln R0 for a host with exponentially
      fading memory (the form of eq. 9 with R the Laplace transform of the generation interval).
  B4  An extension that is NOT in the paper: the vector switches at random between 1.5 and 0.5 times its mean
      with a memory of 100 days (two-state Markov), the cycle being at the threshold of the mean environment.
      Claim under test: the second-order formula gives the sign of the exact growth rate, a loss for Model 1
      and a GAIN for Model 3 (exponential and gamma dormancy); the simulated growth rate has that sign by at
      least 3 standard errors in each.  (Added later: for Model 1, which has two stages, the simulation agrees
      with the exact quadrature within 4 standard errors.)
  B5  Dormancy tuned to the season (Figure 4b, read by eye): with gamma dormancy (K = 11) the exact R0 is
      largest at a dormancy of at most 2 months at 35% seasonality and of 5 to 7 months at 95%.  Reported: the
      optimum according to the second-order formula.

Run: python3 -I checks/priorart_watts_white.py      (about two minutes)

First complete run, 14 November 2026, 11:41 (output kept as results/priorart-watts-white_attempt1_20261114.txt):
17 PASS, 1 FAIL.  The failure was A0: in the two fast-switching cases (reading B of Examples 4.1 and 4.2,
leaving rate 12.5) the simulation was 6.0 and 5.8 standard errors below the quadrature (differences of 2.5e-6
and 3.8e-6 in the growth rate).  Cause: the simulation discarded 200 sojourns at the start, which at that rate is
16 time units, about five relaxation times of the frozen systems; the start, far from the stationary direction,
biased the estimate downwards.  With 2,000 or 8,000 sojourns discarded the same simulation agreed with the
quadrature to 0.2 to 0.8 standard errors (run separately to find the cause).  Changed for the second run, and
nothing else in what was already tested: (1) the number of sojourns discarded is now at least 60 relaxation
times of the slower frozen system (function _burn_in); (2) the simulation estimate, which was a regression
through the origin of log growth on elapsed time and the control variate, is now a ratio estimate with the
same control variate (function _growth_estimate), because the regression has a bias of relative order
1/(number of sojourns) in the part of the growth rate that is due to transients, which is of the size of its
standard error.  No tolerance and no formula under test was changed.  Added after the first run (so its
tolerance was not fixed before every result was seen, though its own outcome was not known): A6b, the R0* of
the variant of A6.  Added after the second run (19 PASS, 0 FAIL): A7, an algebraic identity for the
two-stage ring with three illustrations; nothing else was touched, and with the seeds fixed every other line
of the output is the same as in the second run.  Added after the third run (20 PASS): in B4, the exact value
for Model 1 by the quadrature of Part A, as a check of the simulation in the setting of B4 (4 standard errors);
it had been computed separately first and agreed to 0.5 standard errors, so its outcome was known when added.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import integrate, optimize, special
from scipy.integrate import solve_ivp

COUNT = {"PASS": 0, "FAIL": 0, "DIFFERS FROM PAPER": 0}


def report(name, ok, detail):
    key = "PASS" if ok else "FAIL"
    COUNT[key] += 1
    print(f"[{key}] {name}: {detail}", flush=True)


def differs(name, detail):
    COUNT["DIFFERS FROM PAPER"] += 1
    print(f"[DIFFERS FROM PAPER] {name}: {detail}", flush=True)


# ----------------------------------------------------------------------------------------------------------
# general tools
# ----------------------------------------------------------------------------------------------------------
def perron(a):
    """Dominant eigenvalue r0, right vector v, left vector w with w.v = 1, of a matrix with non-negative
    off-diagonal entries."""
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    r0 = float(vals[k].real)
    v = np.abs(vecs[:, k].real)
    valsl, vecsl = np.linalg.eig(a.T)
    kl = int(np.argmax(valsl.real))
    w = np.abs(vecsl[:, kl].real)
    return r0, v, w / (w @ v)


def second_order_matrix(a0, parts, alpha, var):
    """Second-order change of the growth rate for A(t) = A0 + xi(t) sum(parts), xi with autocovariance
    var exp(-alpha t).  Returns (total, matrix of the bilinear terms between the parts)."""
    r0, v, w = perron(a0)
    n = a0.shape[0]
    res = np.linalg.inv((alpha + r0) * np.eye(n) - a0)
    q = np.array([[var * (w @ bi @ res @ bj @ v - (w @ bi @ v) * (w @ bj @ v) / alpha) for bj in parts] for bi in parts])
    return float(q.sum()), q


def ring_matrix(loss, gain):
    """Continuous-time ring: dx_{k+1}/dt = gain_k x_k - loss_{k+1} x_{k+1}."""
    n = len(loss)
    a = -np.diag(np.asarray(loss, float))
    for k in range(n):
        a[(k + 1) % n, k] += gain[k]
    return a


def ring_closed_form(a, alpha, var, transfers, losses):
    """The same second-order change in the ring's notation.  a: lineage leaving rates r0 + loss_k.
    transfers: list of (stage i the transfer leaves, relative size kappa_i).  losses: list of (stage j,
    absolute size delta_j of the loss rate).  All on one driver with autocovariance var exp(-alpha t).
    Returns a dict of parts; their sum is the change of the growth rate (positive = gain)."""
    a = np.asarray(a, float)
    n = len(a)
    rho = a / (a + alpha)
    big_r = float(np.prod(rho))
    gen = float(np.sum(1 / a))
    p = 1 / (a * gen)

    def path(first, last):                      # product of rho over stages first, first+1, ..., last (cyclic)
        length = (last - first) % n + 1
        return float(np.prod([rho[(first + q) % n] for q in range(length)]))

    def occupation(k, i):                       # E_k int exp(-alpha t) 1{line in stage i at t} dt
        pre = 1.0 if k == i else path(k, (i - 1) % n)
        return pre / ((a[i] + alpha) * (1 - big_r))

    out = {"transfers, own": 0.0, "transfers, cross": 0.0, "loss rates": 0.0, "loss rate x transfer": 0.0}
    for (i, ki) in transfers:
        for (j, kj) in transfers:
            val = var * ki * kj / gen * (path((j + 1) % n, i) / (1 - big_r) - 1 / (alpha * gen))
            out["transfers, own" if i == j else "transfers, cross"] += val
    for (j, dj) in losses:
        for (l, dl) in losses:
            out["loss rates"] += var * dj * dl * (p[l] * occupation(l, j) - p[j] * p[l] / alpha)
    for (i, ki) in transfers:
        for (j, dj) in losses:
            first = -ki * dj * p[j] * (a[i] * occupation(j, i) - 1 / (alpha * gen))
            second = -ki * dj * (occupation((i + 1) % n, j) / gen - p[j] / (alpha * gen))
            out["loss rate x transfer"] += var * (first + second)
    return out


def lyap_quad(a1, a2, k1, k2):
    """Top Lyapunov exponent of dx/dt = A_s x for 2 x 2 matrices [[-a, b], [c, -d]] with b, c > 0 and a two-state
    Markov chain s (k1, k2 the rates of leaving states 1 and 2), from the stationary law of y = x2/x1.
    In state l, dy/dt = h_l(y) = -b_l (y - q_l)(y - n_l), q_l > 0 > n_l.  Between the two roots q the stationary
    probability flux phi = h_U p_U = -h_L p_L (L the state with the lower root) satisfies
    d ln phi/dy = -(k_U/h_U + k_L/h_L), so
        p_L ~ (y - q_L)^(e_L - 1) (q_U - y)^(e_U) (y - n_L)^(-e_L - 1) (y - n_U)^(-e_U) / b_L,
        p_U ~ (y - q_L)^(e_L) (q_U - y)^(e_U - 1) (y - n_L)^(-e_L) (y - n_U)^(-e_U - 1) / b_U,
    e_l = k_l / (eigenvalue gap of A_l).  The growth rate of x1 is b_l y - a_l."""
    par = []
    for m in (a1, a2):
        a, b, c, d = -m[0, 0], m[0, 1], m[1, 0], -m[1, 1]
        gap = math.sqrt((a - d) ** 2 + 4 * b * c)
        par.append((a, b, gap, ((a - d) + gap) / (2 * b), ((a - d) - gap) / (2 * b)))
    pi1 = k2 / (k1 + k2)
    if abs(par[0][3] - par[1][3]) < 1e-13 * max(par[0][3], par[1][3]):
        return pi1 * (par[0][1] * par[0][3] - par[0][0]) + (1 - pi1) * (par[1][1] * par[1][3] - par[1][0])
    if par[0][3] < par[1][3]:
        (al, bl, gl, ql, nl), (au, bu, gu, qu, nu), kl, ku = par[0], par[1], k1, k2
    else:
        (al, bl, gl, ql, nl), (au, bu, gu, qu, nu), kl, ku = par[1], par[0], k2, k1
    el, eu = kl / gl, ku / gu

    def logsmooth(y):
        return -el * math.log(y - nl) - eu * math.log(y - nu)

    ys = np.linspace(ql, qu, 2003)[1:-1]
    lg = np.array([el * math.log(y - ql) + eu * math.log(qu - y) + logsmooth(y) for y in ys])
    scale = float(lg.max())
    mode = float(ys[int(lg.argmax())])
    mid = min(max(mode, ql + 0.05 * (qu - ql)), qu - 0.05 * (qu - ql))

    def integral(which, fun):
        if which == "L":
            ea, eb = el - 1, eu
            extra = lambda y: 1 / (bl * (y - nl))
        else:
            ea, eb = el, eu - 1
            extra = lambda y: 1 / (bu * (y - nu))
        wa, sa = (ea, 0.0) if ea < 1.0 else (0.0, ea)      # exponent given to the quadrature weight / kept in f
        wb, sb = (eb, 0.0) if eb < 1.0 else (0.0, eb)

        def f(y):
            val = logsmooth(y) - scale
            if sa:
                val += sa * math.log(y - ql)
            if sb:
                val += sb * math.log(qu - y)
            return math.exp(val) * extra(y) * fun(y)

        lower = (lambda y: f(y) * (qu - y) ** wb) if wb else f
        upper = (lambda y: f(y) * (y - ql) ** wa) if wa else f
        kw = dict(limit=400, epsabs=0, epsrel=1e-12)
        tot = integrate.quad(lower, ql, mid, weight="alg", wvar=(wa, 0.0), **kw)[0] if wa else integrate.quad(lower, ql, mid, **kw)[0]
        tot += integrate.quad(upper, mid, qu, weight="alg", wvar=(0.0, wb), **kw)[0] if wb else integrate.quad(upper, mid, qu, **kw)[0]
        return tot

    z = integral("L", lambda y: 1.0) + integral("U", lambda y: 1.0)
    g = integral("L", lambda y: bl * y - al) + integral("U", lambda y: bu * y - au)
    return g / z


def lyap_sim(a1, a2, k1, k2, chains, sojourns, seed, burn=None):
    """Top Lyapunov exponent by simulation: exact propagation over exponential sojourns (eigen-decomposition
    of each matrix), states alternating; independent chains; estimate by _growth_estimate."""
    rng = np.random.default_rng(seed)
    burn = _burn_in(a1, a2, k1, k2) if burn is None else burn
    dec = []
    for m in (a1, a2):
        vals, vecs = np.linalg.eig(m)
        top = float(np.max(vals.real))
        dec.append((vals - top, vecs, np.linalg.inv(vecs), top))
    n = a1.shape[0]
    x = np.full((chains, n), 1.0 / n)
    growth = np.zeros(chains)
    time = np.zeros(chains)
    time1 = np.zeros(chains)
    rates = (k1, k2)
    for step in range(burn + sojourns):
        s = step % 2
        dl, vecs, inv, top = dec[s]
        t = rng.exponential(1 / rates[s], chains)
        x = ((x @ inv.T) * np.exp(np.outer(t, dl))) @ vecs.T
        x = np.maximum(x.real, 0.0)
        norm = x.sum(axis=1)
        x /= norm[:, None]
        if step >= burn:
            growth += top * t + np.log(norm)
            time += t
            if s == 0:
                time1 += t
    return _growth_estimate(growth, time, time1, k2 / (k1 + k2))


def _growth_estimate(growth, time, time1, pi1):
    """Ratio estimate sum(G - c D)/sum(T) over independent chains, D = time in state 1 minus its expected
    share of the elapsed time (mean zero), c the least-squares slope of G - (sum G/sum T) T on D.  Standard
    error by the usual formula for a ratio."""
    d = time1 - pi1 * time
    plain = growth.sum() / time.sum()
    c = float(d @ (growth - plain * time) / (d @ d))
    est = float((growth - c * d).sum() / time.sum())
    resid = growth - c * d - est * time
    n = len(growth)
    return est, float(math.sqrt(resid @ resid / (n * (n - 1))) / time.mean())


def _burn_in(a1, a2, k1, k2):
    """Sojourns discarded at the start: at least 200, and at least 60 relaxation times of the slower frozen
    system (relaxation rate = gap between its two leading eigenvalues)."""
    gap = np.inf
    for m in (a1, a2):
        re = np.sort(np.linalg.eigvals(m).real)[::-1]
        gap = min(gap, re[0] - re[1])
    return max(200, 2 * int(math.ceil(60 / gap / (1 / k1 + 1 / k2))))


def lyap_sim_powers(a1, a2, k1, k2, chains, sojourns, seed, burn=None, bits=19):
    """The same estimate for matrices of any size, including defective ones (the chain of dormancy stages has
    repeated eigenvalues, so an eigen-decomposition cannot be used).  Each sojourn is an exponential time
    rounded to a multiple of q = (shorter mean sojourn)/4096 and applied through the binary powers of
    expm(A q); the time counted is the rounded time, so the simulated process is propagated exactly."""
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    burn = _burn_in(a1, a2, k1, k2) if burn is None else burn
    q = min(1 / k1, 1 / k2) / 4096
    powers = []
    for m in (a1, a2):
        e = expm(m * q)
        lst = [e]
        for _ in range(bits - 1):
            e = e @ e
            lst.append(e)
        powers.append(lst)
    n = a1.shape[0]
    x = np.full((chains, n), 1.0 / n)
    growth = np.zeros(chains)
    time = np.zeros(chains)
    time1 = np.zeros(chains)
    rates = (k1, k2)
    for step in range(burn + sojourns):
        s = step % 2
        kq = np.minimum(np.rint(rng.exponential(1 / rates[s], chains) / q).astype(np.int64), 2 ** bits - 1)
        for j in range(bits):
            mask = ((kq >> j) & 1) == 1
            if mask.any():
                x[mask] = x[mask] @ powers[s][j].T
        x = np.maximum(x, 0.0)
        norm = x.sum(axis=1)
        x /= norm[:, None]
        if step >= burn:
            growth += np.log(norm)
            time += kq * q
            if s == 0:
                time1 += kq * q
    return _growth_estimate(growth, time, time1, k2 / (k1 + k2))


# ----------------------------------------------------------------------------------------------------------
# PART A.  Watts et al. 2021
# ----------------------------------------------------------------------------------------------------------
def watts_matrix(gamma1, a, c, mu2):
    return np.array([[-gamma1, a], [c, -mu2]])


WATTS = {
    "4.1": dict(mu1=0.005, mu2=0.025, s1=1.0, s2=1.0, beta=(0.15, 0.05), sigma=(0.15, 0.15), gamma=(0.3, 0.3), u=0.5, eps=25.0),
    "4.2": dict(mu1=0.005, mu2=0.025, s1=1.0, s2=1.0, beta=(0.15, 0.05), sigma=(0.05, 0.15), gamma=(0.3, 0.3), u=0.5, eps=25.0),
    "4.3": dict(mu1=1.0, mu2=1.0, s1=1.0, s2=1.0, beta=(1.1, 2.1), sigma=(1.0, 2.0), gamma=(0.01, 3.0), u=0.5, eps=0.05),
}   # s1 = omega1/mu1 and s2 = omega2/mu2 are one in every example


def watts_pair(p, scale=1.0, gamma_fixed=False, divide_beta=1.0):
    """The two matrices of an example; scale shrinks their difference about the mean; gamma_fixed puts the
    removal rate at its mean in both states; divide_beta divides the vector-to-host transfer (for R0*)."""
    mats = []
    for l in (0, 1):
        g1 = p["gamma"][l] + p["mu1"]
        if gamma_fixed:
            g1 = p["u"] * p["gamma"][0] + (1 - p["u"]) * p["gamma"][1] + p["mu1"]
        mats.append(watts_matrix(g1, p["beta"][l] * p["s1"] / divide_beta, p["sigma"][l] * p["s2"], p["mu2"]))
    u = p["u"]
    mean = u * mats[0] + (1 - u) * mats[1]
    return mean + scale * (mats[0] - mean), mean + scale * (mats[1] - mean), mean


def watts_rates(p, reading, eps=None):
    eps = p["eps"] if eps is None else eps
    u = p["u"]
    return ((1 - u) / eps, u / eps) if reading == "A" else ((1 - u) * eps, u * eps)


def watts_second_order(a1, a2, mean, k1, k2):
    """Mean growth rate, second-order term (matrix form), and its parts in the ring's notation."""
    alpha = k1 + k2
    pi1 = k2 / alpha
    var = pi1 * (1 - pi1)
    b = a1 - a2
    r0, v, w = perron(mean)
    b_c = np.array([[0.0, 0.0], [b[1, 0], 0.0]])       # transfer out of stage 1 (host to vector)
    b_a = np.array([[0.0, b[0, 1]], [0.0, 0.0]])       # transfer out of stage 2 (vector to host)
    b_g = np.array([[b[0, 0], 0.0], [0.0, 0.0]])       # minus the difference of the host's loss rate
    total, q = second_order_matrix(mean, [b_c, b_a, b_g], alpha, var)
    lin = np.array([r0 - mean[0, 0], r0 - mean[1, 1]])   # lineage leaving rates
    transfers = [(0, b[1, 0] / mean[1, 0]), (1, b[0, 1] / mean[0, 1])]
    transfers = [(i, k) for (i, k) in transfers if k != 0.0]
    losses = [(0, -b[0, 0])] if b[0, 0] != 0.0 else []
    parts = ring_closed_form(lin, alpha, var, transfers, losses)
    gross = float(np.abs(q).sum())
    return r0, total, parts, gross, lin, alpha, var


def part_a():
    print("=" * 110)
    print("PART A.  Watts, Mishra, Nguyen & Tuong (2021): a host-vector loop under random switching")
    print("=" * 110)

    # ---- A1: printed numbers
    print("\nA1. Printed thresholds of the frozen systems and of the averaged system")
    ok = True
    rows = []
    printed = {"4.1": ((0.04018, 5e-6), (-0.00038, 5e-6)), "4.2": ((-3.7922e-4, 5e-9), (-3.7922e-4, 5e-9))}
    frozen = {}
    for name, p in WATTS.items():
        a1, a2, mean = watts_pair(p)
        l1, l2 = perron(a1)[0], perron(a2)[0]
        frozen[name] = (l1, l2)
        r0u = (mean[0, 1] * mean[1, 0]) / (mean[0, 0] * mean[1, 1])
        rows.append((name, l1, l2, perron(mean)[0], r0u))
        if name in printed:
            for val, (pr, tol) in zip((l1, l2), printed[name]):
                ok &= abs(val - pr) <= 1.02 * tol
        print(f"    Example {name}: frozen lambda_1 = {l1:+.7f}, lambda_2 = {l2:+.7f}; mean-rate matrix: growth rate {perron(mean)[0]:+.7f}, R0 = {r0u:.4f}")
    ok &= abs(rows[0][4] - 1.9672) <= 5.1e-5
    report("A1 printed frozen thresholds and the averaged R0 of Example 4.1", ok,
           "0.04018, -0.00038, -3.7922e-4 (twice) and 1.9672 reproduced to half a unit of the last printed digit")
    differs("A1 averaged R0 of Example 4.2", f"printed 1.967; the printed parameters give {rows[1][4]:.4f} "
            f"(beta-bar sigma-bar = 0.1 x 0.1 over mu2 (gamma + mu1) = 0.007625). 1.967 and the printed equilibrium (0.5317, 0.0077, 0.04403) are those of Example 4.1. "
            "The qualitative statement (averaged R0 above one) is unaffected.")
    p3 = WATTS["4.3"]
    a1, a2, mean = watts_pair(p3)
    c38 = a1[0, 1] * a1[1, 0] > -a1[0, 0] * 1.0 and a2[0, 1] * a2[1, 0] > -a2[0, 0] * 1.0
    c39 = mean[0, 1] * mean[1, 0] < -mean[0, 0] * 1.0
    print(f"    Example 4.3: A C against gamma1 mu2 in state 1: {a1[0,1]*a1[1,0]:.3f} > {-a1[0,0]:.3f}; state 2: {a2[0,1]*a2[1,0]:.3f} > {-a2[0,0]:.3f}; "
          f"mean rates: {mean[0,1]*mean[1,0]:.3f} < {-mean[0,0]:.3f}  (their conditions 38 and 39: {c38 and c39})")

    # ---- A0: exact methods
    print("\nA0. Exact Lyapunov exponent by two methods (quadrature of the stationary law; simulation)")
    exact = {}
    worst = 0.0
    for name, p in WATTS.items():
        a1, a2, mean = watts_pair(p)
        for reading in ("A", "B"):
            k1, k2 = watts_rates(p, reading)
            lq = lyap_quad(a1, a2, k1, k2)
            ls, se = lyap_sim(a1, a2, k1, k2, chains=4000, sojourns=4000, seed=20261114 + int(float(name) * 10) + (reading == "B"))
            exact[(name, reading)] = (lq, ls, se, k1, k2)
            worst = max(worst, abs(lq - ls) / se)
            print(f"    Example {name}, reading {reading} (eps = {p['eps']}: leaving rates {k1:g}, {k2:g}): quadrature {lq:+.8f}; simulation {ls:+.8f} +- {se:.1e}  ({abs(lq - ls) / se:.1f} s.e.)")
    report("A0 the two exact methods agree", worst <= 4.0, f"largest difference {worst:.1f} standard errors in six cases")
    lq, _, _, k1, k2 = exact[("4.2", "A")]
    a1, a2, _ = watts_pair(WATTS["4.2"])
    lp, sep = lyap_sim_powers(a1, a2, k1, k2, chains=4000, sojourns=2000, seed=2026111477)
    report("A0 the simulation code used in B4 (binary powers of the matrix exponential) agrees with the quadrature", abs(lp - lq) <= 4 * sep,
           f"Example 4.2, reading A: {lp:+.8f} +- {sep:.1e} against {lq:+.8f} ({abs(lp - lq) / sep:.1f} s.e.)")
    ok = True
    for name, p in WATTS.items():
        a1, a2, mean = watts_pair(p)
        s0 = min(-a1[0, 0], -a1[1, 1], -a2[0, 0], -a2[1, 1])
        lm = perron(mean)[0]
        lslow = 0.5 * (perron(a1)[0] + perron(a2)[0])
        slow = (lyap_quad(a1, a2, s0 / 100, s0 / 100) - lslow) / (lyap_quad(a1, a2, s0 / 1000, s0 / 1000) - lslow)
        fast = (lyap_quad(a1, a2, 100 * s0, 100 * s0) - lm) / (lyap_quad(a1, a2, 1000 * s0, 1000 * s0) - lm)
        ok &= 8 <= slow <= 12 and 8 <= fast <= 12
        print(f"    Example {name}: slow limit {lslow:+.7f}, fast limit {lm:+.7f}; ratio of distances for a ten-fold change of rate: slow {slow:.2f}, fast {fast:.2f}")
    report("A0 first-order approach to the slow and the fast limit", ok, "ratios within [8, 12] for the three examples")

    # ---- A2: stated behaviour
    print("\nA2. The behaviour the paper states (reading A, the generator as printed)")
    lq, ls, se, _, _ = exact[("4.1", "A")]
    ok1 = lq > 0 and ls - 4 * se > 0
    differs("A2 switched threshold of Example 4.1", f"printed 0.0500; exact {lq:+.5f} (reading A) and {exact[('4.1', 'B')][0]:+.5f} (reading B). "
            f"A value above the larger frozen threshold {frozen['4.1'][0]:.5f} is impossible here, because A_1 >= A_2 entrywise. Persistence (lambda > 0) is reproduced.")
    lq2, ls2, se2, _, _ = exact[("4.2", "A")]
    ok2 = frozen["4.2"][0] < 0 and frozen["4.2"][1] < 0 and lq2 > 0 and ls2 - 4 * se2 > 0
    lq3, ls3, se3, _, _ = exact[("4.3", "A")]
    ok3 = frozen["4.3"][0] > 0 and frozen["4.3"][1] > 0 and lq3 < 0 and ls3 + 4 * se3 < 0
    report("A2 stated behaviour of Examples 4.1, 4.2, 4.3", ok1 and ok2 and ok3,
           f"4.1: lambda = {lq:+.5f} > 0. 4.2: frozen {frozen['4.2'][0]:+.2e} twice, switched {lq2:+.6f} > 0. "
           f"4.3: frozen {frozen['4.3'][0]:+.4f} and {frozen['4.3'][1]:+.4f}, switched {lq3:+.5f} < 0.")
    p2 = WATTS["4.2"]
    a1, a2, _ = watts_pair(p2)
    f2 = lambda le: lyap_quad(a1, a2, *watts_rates(p2, "A", math.exp(le)))
    eps_star2 = math.exp(optimize.brentq(f2, math.log(25.0), math.log(2500.0), xtol=1e-10))
    a1, a2, _ = watts_pair(p3)
    f3 = lambda le: lyap_quad(a1, a2, *watts_rates(p3, "A", math.exp(le)))
    eps_star3 = math.exp(optimize.brentq(f3, math.log(0.05), math.log(50.0), xtol=1e-10))
    print(f"    Where the sign turns (exact): Example 4.2 persists for eps < {eps_star2:.2f} (leaving rate above {0.5 / eps_star2:.5f}); "
          f"Example 4.3 dies out for eps < {eps_star3:.3f} (leaving rate above {0.5 / eps_star3:.4f}).")

    # ---- A3: closed forms against the matrix formula
    print("\nA3. Ring-notation closed forms against the matrix formula")
    rng = np.random.default_rng(2026111431)
    worst = 0.0
    for trial in range(300):
        n = int(rng.integers(2, 6))
        loss = rng.uniform(0.1, 3.0, n)
        gain = rng.uniform(0.1, 3.0, n)
        alpha = float(rng.choice([0.03, 0.3, 3.0]) * rng.uniform(0.5, 2))
        a0 = ring_matrix(loss, gain)
        r0 = perron(a0)[0]
        nt = int(rng.integers(0, n + 1))
        tr = [(int(i), float(rng.uniform(-1.5, 1.5))) for i in rng.choice(n, nt, replace=False)]
        nl = int(rng.integers(0 if nt else 1, n + 1))
        ls = [(int(j), float(rng.uniform(-1.0, 1.0))) for j in rng.choice(n, nl, replace=False)]
        parts = []
        for (i, k) in tr:
            b = np.zeros((n, n)); b[(i + 1) % n, i] = k * gain[i]; parts.append(b)
        for (j, d) in ls:
            b = np.zeros((n, n)); b[j, j] = -d; parts.append(b)
        total, q = second_order_matrix(a0, parts, alpha, 0.37)
        cf = sum(ring_closed_form(r0 + loss, alpha, 0.37, tr, ls).values())
        worst = max(worst, abs(cf - total) / max(float(np.abs(q).sum()), 1e-300))
    report("A3 closed forms in the ring's notation = matrix formula", worst < 1e-9, f"300 random rings of 2 to 5 stages, transfers and loss rates on one driver: largest relative difference {worst:.1e}")

    # ---- A4 and the table at the paper's amplitude
    print("\nA4. Our formula for the three examples (reading A)")
    print("    r0: growth rate of the mean-rate matrix.  T: generation time of a line of descent.  alpha: sum of the switching rates.")
    ok = True
    table = []
    for name, p in WATTS.items():
        k1, k2 = watts_rates(p, "A")
        a1, a2, mean = watts_pair(p)
        r0, total, parts, gross, lin, alpha, var = watts_second_order(a1, a2, mean, k1, k2)
        lq, ls, se, _, _ = exact[(name, "A")]
        gen = float(np.sum(1 / lin))
        b = a1 - a2
        print(f"    Example {name}: r0 = {r0:+.7f}; lineage leaving rates {lin[0]:.5f}, {lin[1]:.5f}; T = {gen:.3f}; alpha = {alpha:g}; variance of the driver {var:g}")
        print(f"        relative size of the transfers: host to vector {b[1,0]/mean[1,0]:+.4f}, vector to host {b[0,1]/mean[0,1]:+.4f}; difference of the host's loss rate {-b[0,0]:+.4f}")
        print("        second-order parts (change of growth rate): " + "; ".join(f"{k} {v:+.3e}" for k, v in parts.items() if v != 0.0))
        print(f"        sum {sum(parts.values()):+.7f} (matrix form {total:+.7f}; gross {gross:.3e})")
        print(f"        at the paper's amplitude: exact {lq:+.7f}; ours r0 + second order = {r0 + total:+.7f}; difference (ours - exact) {r0 + total - lq:+.2e} "
              f"= {100 * (r0 + total - lq) / gross:+.1f}% of the gross second-order term, {100 * (r0 + total - lq) / abs(lq - r0):+.1f}% of the exact departure from r0")
        table.append((name, frozen[name], r0, lq, ls, se, r0 + total, r0 + total - lq))
        for s, tol in ((0.1, 0.05), (0.05, 0.02)):
            b1, b2, mean_s = watts_pair(p, scale=s)
            r0s, tot_s, _, gross_s, _, _, _ = watts_second_order(b1, b2, mean_s, k1, k2)
            ex = lyap_quad(b1, b2, k1, k2)
            err = abs(ex - (r0s + tot_s)) / gross_s
            ok &= err <= tol
            print(f"        difference between states x {s}: exact - r0 = {ex - r0s:+.6e}; second order {tot_s:+.6e}; error {100 * err:.2f}% of the gross term (tolerance {100 * tol:.0f}%)")
    report("A4 second-order formula at small switching", ok, "within 5% of the gross term at 0.1 and 2% at 0.05, three examples")
    print("\n    Table A (reading A).  Growth rates per unit time.")
    print("    example   frozen 1     frozen 2     mean matrix   exact (quadrature)   simulation            ours (2nd order)   ours - exact")
    for (name, fr, r0, lq, ls, se, ours, diff) in table:
        print(f"    {name}     {fr[0]:+.6f}   {fr[1]:+.6f}   {r0:+.6f}     {lq:+.6f}            {ls:+.6f} +- {se:.0e}   {ours:+.6f}          {diff:+.1e}")
    # where the second-order formula puts the change of sign
    for name, bracket in (("4.2", (25.0, 2500.0)), ("4.3", (0.05, 50.0))):
        p = WATTS[name]
        a1, a2, mean = watts_pair(p)
        def so(le, p=p, a1=a1, a2=a2, mean=mean):
            k1, k2 = watts_rates(p, "A", math.exp(le))
            r0, total, *_ = watts_second_order(a1, a2, mean, k1, k2)
            return r0 + total
        try:
            es = math.exp(optimize.brentq(so, math.log(bracket[0]), math.log(bracket[1]), xtol=1e-10))
            print(f"    Example {name}: the second-order formula puts the change of sign at eps = {es:.3f} (exact: {eps_star2 if name == '4.2' else eps_star3:.3f})")
        except ValueError:
            print(f"    Example {name}: the second-order formula has no change of sign for eps between {bracket[0]} and {bracket[1]} (exact: {eps_star2 if name == '4.2' else eps_star3:.3f})")

    # ---- A5: R0* and the two means (Example 4.2)
    print("\nA5. Example 4.2 on the scale of R0 (only transfers switch; the vector-to-host transfer counted as the birth)")
    p = WATTS["4.2"]
    _, _, mean = watts_pair(p)
    r0a = mean[0, 1] * mean[1, 0] / (mean[0, 0] * mean[1, 1])
    r0g = math.sqrt(p["beta"][0] * p["beta"][1]) * math.sqrt(p["sigma"][0] * p["sigma"][1]) / (mean[0, 0] * mean[1, 1])
    ell = math.log(r0a / r0g)
    print(f"    R0 at arithmetic-mean rates {r0a:.5f}; at geometric-mean rates {r0g:.5f} (equal to the R0 of each frozen state); ln of the ratio {ell:.5f} = sum of -E ln h_k")
    ok = True
    upper = True
    for eps in (0.25, 2.5, 25.0, 250.0, 2500.0):
        k1, k2 = watts_rates(p, "A", eps)
        def lam(lr, k1=k1, k2=k2):
            b1, b2, _ = watts_pair(p, divide_beta=math.exp(lr))
            return lyap_quad(b1, b2, k1, k2)
        lr = optimize.brentq(lam, math.log(r0g) - 0.2, math.log(r0a) + 0.2, xtol=1e-12)
        rstar = math.exp(lr)
        ok &= rstar >= r0g - 1e-9
        upper &= rstar <= r0a + 1e-9
        # second order at the threshold of the mean model: divide beta by R0A, so that r0 = 0
        b1, b2, mean_t = watts_pair(p, divide_beta=r0a)
        r0t, tot_t, _, _, lin_t, _, _ = watts_second_order(b1, b2, mean_t, k1, k2)
        d2 = -tot_t * float(np.sum(1 / lin_t))
        print(f"    eps = {eps:7g} (leaving rate {k1:g}): R0* = {rstar:.5f}; exact D = ln R0A - ln R0* = {math.log(r0a) - lr:.5f}; second-order D = {d2:.5f}; bound {ell:.5f}")
    report("A5 R0* is never below the R0 of geometric-mean rates (Example 4.2)", ok, f"five switching speeds; R0* also at or below the R0 of arithmetic-mean rates in all five: {upper}")

    # ---- A6: both transfers together, loss rate fixed (not in the paper)
    print("\nA6. A variant not in the paper: Example 4.3 with the removal rate held at its mean (2.505), both transfers switching together")
    p = WATTS["4.3"]
    ok = True
    for rate in (10.0, 0.1):
        a1, a2, mean = watts_pair(p, gamma_fixed=True)
        r0, total, parts, gross, lin, alpha, var = watts_second_order(a1, a2, mean, rate, rate)
        lq = lyap_quad(a1, a2, rate, rate)
        ls, se = lyap_sim(a1, a2, rate, rate, chains=4000, sojourns=4000, seed=20261199 + int(rate * 10))
        x = alpha / lin
        b = a1 - a2
        same = b[1, 0] / mean[1, 0]
        # closed form when both relative sizes are equal: -(var kappa^2 / T)(x1 - x2)^2 / ((x1 + x2)(x1 + x2 + x1 x2)) is the COST
        ok &= total > 0 and lq - r0 > 0 and (ls - r0) / se >= 4
        print(f"    leaving rate {rate:g}: mean matrix r0 = {r0:+.7f}; exact {lq:+.7f} (simulation {ls:+.7f} +- {se:.1e}); exact gain {lq - r0:+.3e}; "
              f"second-order gain {total:+.3e} (own {parts['transfers, own']:+.3e}, cross {parts['transfers, cross']:+.3e})")
        print(f"        for equal relative sizes k the two-stage closed form is a gain of (var k^2/T)(x1 - x2)^2/((x1 + x2)(x1 + x2 + x1 x2)), x_k = alpha/a_k: "
              f"{var * same ** 2 / float(np.sum(1 / lin)) * (x[0] - x[1]) ** 2 / ((x[0] + x[1]) * (x[0] + x[1] + x[0] * x[1])):+.3e} at k = {same:.4f} (here the sizes are {same:.4f} and {b[0,1]/mean[0,1]:.4f})")
    report("A6 two transfers that switch together are a gain over the mean-rate matrix", ok, "exact and second-order values both positive at both switching rates")
    _, _, mean = watts_pair(p, gamma_fixed=True)
    r0a = mean[0, 1] * mean[1, 0] / (mean[0, 0] * mean[1, 1])
    r0g = math.sqrt(p["beta"][0] * p["beta"][1] * p["sigma"][0] * p["sigma"][1]) / (mean[0, 0] * mean[1, 1])
    above = True
    for rate in (10.0, 0.1, 0.001):
        def lam(lr, rate=rate):
            b1, b2, _ = watts_pair(p, gamma_fixed=True, divide_beta=math.exp(lr))
            return lyap_quad(b1, b2, rate, rate)
        rstar = math.exp(optimize.brentq(lam, math.log(r0g) - 0.3, math.log(r0a) + 0.5, xtol=1e-12))
        above &= rstar > r0a
        print(f"    leaving rate {rate:g}: R0* = {rstar:.5f} against {r0a:.5f} at arithmetic-mean rates and {r0g:.5f} at geometric-mean rates")
    report("A6b in that variant R0* exceeds the R0 of arithmetic-mean rates", above,
           "so 'fluctuation never raises the reproduction number relative to hosts held at their average abundance' fails when two transfers follow one environment that has memory")

    # ---- A7: the two-stage ring with both transfers on one host, in closed form
    print("\nA7. Two-stage ring, both transfers on one host with exponentially fading memory: cost per generation and per unit variance")
    print("    = [ (k1 + k2)^2 x1 x2 - k1 k2 (x1 + x2)^2 ] / [ (x1 + x2)(x1 + x2 + x1 x2) ],   x_k = alpha/a_k, k1 and k2 the relative sizes")
    rng = np.random.default_rng(2026111433)
    worst = 0.0
    for trial in range(300):
        a = rng.uniform(0.05, 3.0, 2)
        alpha = float(rng.choice([0.03, 0.3, 3.0]) * rng.uniform(0.5, 2))
        k1, k2 = rng.uniform(-2, 2, 2)
        general = -sum(ring_closed_form(a, alpha, 1.0, [(0, float(k1)), (1, float(k2))], []).values()) * float(np.sum(1 / a))
        x = alpha / a
        two = ((k1 + k2) ** 2 * x[0] * x[1] - k1 * k2 * (x[0] + x[1]) ** 2) / ((x[0] + x[1]) * (x[0] + x[1] + x[0] * x[1]))
        worst = max(worst, abs(general - two) / max(abs(two), 1e-9))
    report("A7 the two-stage closed form equals the general one", worst < 1e-9, f"300 random cases: largest relative difference {worst:.1e}")
    print("    Illustration at the threshold with stays of 60 and 10 days (blood-stage infection and infectious mosquito, the Model 1 of Part B):")
    for memory in (30, 100, 365):
        x = (1 / memory) / np.array([1 / 60, 1 / 10])
        s_, p_ = x.sum(), x.prod()
        den = s_ * (s_ + p_)
        print(f"    memory {memory:3d} days: one transfer alone {p_ / den:+.4f} (a cost); both transfers together {-(x[0] - x[1]) ** 2 / den:+.4f} (a gain); "
              f"both, in opposite directions {s_ * s_ / den:+.4f} (a cost); two independent hosts {2 * p_ / den:+.4f}")
    return table


# ----------------------------------------------------------------------------------------------------------
# PART B.  White et al. 2016
# ----------------------------------------------------------------------------------------------------------
W_R, W_GL, W_F, W_GD, W_D, W_K = 1 / 60, 1 / 223, 1 / 72, 1 / 434, 162.0, 11
W_A, W_G, W_N, W_C, W_B = 0.21, 0.1, 12.0, 0.23, 0.5
W_HV = W_A * W_C * math.exp(-W_G * W_N)                               # person to mosquito, per unit of blood-stage prevalence
W_M0 = W_G * W_R / (W_A ** 2 * W_B * W_C * math.exp(-W_G * W_N))      # makes R0 of Model 1 equal to one
W_PHI = W_M0 * W_A * W_B                                              # mosquito to person at mean mosquito density
W_EPS = 0.001
OMEGA = 2 * math.pi / 365
# Figure 4a digitised: seasonality (%), Model 1, Model 2, Model 3
FIG4A = [(35, 0.975, 2.466, 2.311), (40, 0.963, 2.466, 2.363), (50, 0.925, None, 2.460), (60, 0.888, 2.460, 2.547),
         (65, 0.863, 2.453, 2.584), (75, 0.814, 2.435, 2.634), (85, 0.764, 2.407, 2.671), (90, 0.738, 2.385, 2.683), (95, 0.689, 2.343, 2.671)]


def white_model(kind, d=W_D, k_stages=W_K):
    """Loss matrix V of the infected compartments (dx/dt = (f(t) F0 - V) x), the compartment that receives new
    inoculations, and the compartment that makes them (infectious mosquitoes)."""
    r, gl, f, gd, hv, g = W_R, W_GL, W_F, W_GD, W_HV, W_G
    if kind == "M1":                                   # (I0, IM)
        v = np.array([[r, 0.0], [-hv, g]])
    elif kind == "M2":                                 # (IL, SL, I0, IM)
        v = np.array([[r + gl, -f, 0, 0], [-r, f + gl, 0, 0], [-gl, 0, r, 0], [-hv, 0, -hv, g]])
    else:                                              # (ID_1..K, SD_1..K, IL, SL, I0, IM); K = 1 is exponential dormancy
        k = 1 if kind == "M3exp" else k_stages
        n = 2 * k + 4
        v = np.zeros((n, n))
        kd = k / d
        il, sl, i0, im = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        for j in range(k):
            v[j, j] = r + gd + kd
            v[k + j, k + j] = gd + kd
            v[k + j, j] = -r
            if j > 0:
                v[j, j - 1] = -kd
                v[k + j, k + j - 1] = -kd
            v[i0, j] = -gd
            v[im, j] = -hv
        v[il, k - 1] = -kd
        v[sl, 2 * k - 1] = -kd
        v[il, il] = r + gl; v[il, sl] = -f; v[sl, il] = -r; v[sl, sl] = f + gl
        v[i0, il] = -gl; v[i0, i0] = r
        v[im, il] = -hv; v[im, i0] = -hv; v[im, im] = g
    return v, 0, v.shape[0] - 1


def khat(model, s):
    """Laplace transform at s (scalar or array) of the kernel K(x): expected rate of new inoculations at time
    x after one inoculation, at mean mosquito density."""
    v, tgt, src = model
    n = v.shape[0]
    s = np.atleast_1d(np.asarray(s, complex))
    sol = np.linalg.solve(v[None, :, :] + s[:, None, None] * np.eye(n)[None, :, :], np.broadcast_to(np.eye(n)[:, tgt][None, :, None], (len(s), n, 1)))
    return W_PHI * sol[:, src, 0]


def season(t, kappa):
    return W_EPS + (1 - W_EPS) * math.pi / special.beta(0.5, kappa + 0.5) * ((1 + np.cos(OMEGA * t)) / 2) ** kappa


def peak_share(kappa):
    return integrate.quad(lambda t: season(t, kappa), -365 / 8, 365 / 8, epsabs=1e-13, epsrel=1e-13)[0] / 365


def kappa_of(share):
    return optimize.brentq(lambda k: peak_share(k) - share, 0.0, 5000.0, xtol=1e-12)


def harmonics(kappa, m=8192):
    """Fourier coefficients c_j of m(t)/m0, j = 0, 1, ..., (real: the function is even)."""
    t = np.arange(m) * 365 / m
    return np.fft.fft(season(t, kappa)) / m


NH = 150   # harmonics kept


def r0_fourier(model, coef, scale=1.0):
    """R0 as the spectral radius of the next-generation operator on periodic functions."""
    idx = np.arange(-NH, NH + 1)
    kh = khat(model, 1j * idx * OMEGA)
    diff = idx[:, None] - idx[None, :]
    c = np.where(diff == 0, 1.0 + 0j, scale * coef[diff % len(coef)])
    return float(np.max(np.linalg.eigvals(c * kh[None, :]).real))


def r0_floquet(model, kappa, guess):
    """R0 = 1/xi0, xi0 the factor on m(t) at which the dominant Floquet multiplier is one (the paper's method)."""
    v, tgt, src = model
    n = v.shape[0]
    f0 = np.zeros((n, n)); f0[tgt, src] = W_PHI

    def log_multiplier(xi):
        def rhs(t, y):
            return ((xi * season(t, kappa) * f0 - v) @ y.reshape(n, n)).ravel()
        sol = solve_ivp(rhs, (0, 365), np.eye(n).ravel(), rtol=1e-10, atol=1e-13, method="DOP853")
        return math.log(np.max(np.abs(np.linalg.eigvals(sol.y[:, -1].reshape(n, n)))))

    try:
        xi0 = optimize.brentq(log_multiplier, 0.97 / guess, 1.03 / guess, xtol=1e-11, rtol=1e-11)
    except ValueError:                       # no root within 3% of the other method's value
        return float("nan")
    return 1 / xi0


def second_order_periodic(model, coef):
    """Sum over harmonics of (e_n^2/2) Re[g/(1 - g)], and the sum of the absolute values of the terms."""
    n = np.arange(1, NH + 1)
    g = khat(model, 1j * n * OMEGA) / khat(model, 0.0)[0]
    terms = (2 * np.abs(coef[n])) ** 2 / 2 * (g / (1 - g)).real
    return float(terms.sum()), float(np.abs(terms).sum())


def generation_interval(model):
    v, tgt, src = model
    e = np.eye(v.shape[0])[:, tgt]
    inv = np.linalg.inv(v)
    k0, k1, k2 = (inv @ e)[src], (inv @ inv @ e)[src], 2 * (inv @ inv @ inv @ e)[src]
    mean = k1 / k0
    return float(mean), float(k2 / k0 / mean ** 2 - 1)


def part_b():
    print()
    print("=" * 110)
    print("PART B.  White, Shirreff, Karl, Ghani & Mueller (2016): a periodic vector and a dormant liver stage")
    print("=" * 110)
    kinds = ("M1", "M2", "M3exp", "M3gam")
    label = {"M1": "Model 1 (no relapse)", "M2": "Model 2 (tropical)", "M3exp": "Model 3 (exponential dormancy)", "M3gam": "Model 3 (gamma dormancy, K = 11)"}
    models = {k: white_model(k) for k in kinds}
    r0c = {k: float(khat(models[k], 0.0)[0].real) for k in kinds}

    # ---- B0
    print("\nB0. The forcing function and the constant-environment R0")
    s37, s123 = peak_share(0.37), peak_share(12.3)
    f27 = 1 + W_F * W_R / (W_GL * (W_F + W_R + W_GL))
    dl = 1 / W_D
    f28 = 1 + dl * W_F * W_R * (dl + W_R + W_GL + W_GD) / (W_GL * (W_F + W_R + W_GL) * (dl + W_GD) * (dl + W_R + W_GD))
    ok = abs(s37 - 0.35) <= 0.005 and abs(s123 - 0.95) <= 0.005 and abs(r0c["M1"] - 1) < 1e-10 and abs(r0c["M2"] / f27 - 1) < 1e-10 and abs(r0c["M3exp"] / f28 - 1) < 1e-10
    report("B0 forcing function and closed forms (2.5), (2.7), (2.8)", ok,
           f"share of the peak three months {s37:.4f} at kappa = 0.37 and {s123:.4f} at kappa = 12.3 (printed: 35% and 95%); "
           f"constant R0: Model 1 {r0c['M1']:.6f}, Model 2 {r0c['M2']:.6f} (2.7: {f27:.6f}), Model 3 exponential {r0c['M3exp']:.6f} (2.8: {f28:.6f}), Model 3 gamma {r0c['M3gam']:.6f}")

    # ---- B1
    print("\nB1. R0 against seasonality: two exact methods, and Figure 4a")
    levels = [35, 40, 50, 60, 65, 75, 85, 90, 95]
    kap = {s: kappa_of(s / 100) for s in levels}
    coef = {s: harmonics(kap[s]) for s in levels}
    exact = {(k, s): r0_fourier(models[k], coef[s]) for k in kinds for s in levels}
    worst = 0.0
    for s in (35, 50, 75, 90):
        for k in kinds:
            fl = r0_floquet(models[k], kap[s], exact[(k, s)])
            worst = max(worst, abs(fl / exact[(k, s)] - 1) if fl == fl else float("inf"))
    report("B1a Floquet method (the paper's) = next-generation operator in Fourier space", worst <= 1e-5, f"16 cases: largest relative difference {worst:.1e}")
    print("    seasonality  kappa     Model 1: computed (figure)   Model 2: computed (figure)   Model 3 gamma: computed (figure)   Model 3 exponential: computed")
    wfig, wexp = 0.0, 0.0
    for (s, d1, d2, d3) in FIG4A:
        e1, e2, e3, e3x = exact[("M1", s)], exact[("M2", s)], exact[("M3gam", s)], exact[("M3exp", s)]
        wfig = max(wfig, abs(e1 - d1), abs(e3 - d3), abs(e2 - d2) if d2 is not None else 0.0)
        wexp = max(wexp, abs(e3x - d3))
        d2s = f"{d2:.3f}" if d2 is not None else "  -  "
        print(f"    {s:3d}%        {kap[s]:7.3f}   {e1:.4f} ({d1:.3f})              {e2:.4f} ({d2s})              {e3:.4f} ({d3:.3f})                    {e3x:.4f}")
    report("B1b Figure 4a reproduced (Model 3 with gamma-distributed dormancy)", wfig <= 0.04, f"largest |computed - digitised| = {wfig:.3f} over 26 points; with exponential dormancy the Model 3 curve is off by up to {wexp:.3f}")
    ok = all(exact[("M1", levels[i + 1])] < exact[("M1", levels[i])] for i in range(len(levels) - 1)) and exact[("M1", 35)] < r0c["M1"]
    ok &= all(abs(exact[("M2", s)] / r0c["M2"] - 1) < 0.06 for s in levels)
    ok &= all(exact[("M3gam", s)] > r0c["M3gam"] for s in levels) and all(exact[("M3exp", s)] > r0c["M3exp"] for s in levels)
    report("B1c stated behaviour: R0 falls in Model 1, changes little in Model 2, rises in Model 3", ok,
           f"at 95% seasonality R0/R0(constant) is {exact[('M1', 95)] / r0c['M1']:.3f}, {exact[('M2', 95)] / r0c['M2']:.3f}, {exact[('M3gam', 95)] / r0c['M3gam']:.3f} (gamma) and {exact[('M3exp', 95)] / r0c['M3exp']:.3f} (exponential)")

    # ---- B2
    print("\nB2. Our formula: Result 7 summed over the harmonics, g the transform of the generation interval")
    ok_a, worst_sym, worst_one = True, 0.0, 0.0
    for s in (35, 50, 75, 90):
        for k in kinds:
            c2, gross = second_order_periodic(models[k], coef[s])
            dp = math.log(r0_fourier(models[k], coef[s], 0.05) / r0c[k])
            dm = math.log(r0_fourier(models[k], coef[s], -0.05) / r0c[k])
            d1 = math.log(r0_fourier(models[k], coef[s], 0.1) / r0c[k])
            e_sym = abs((dp + dm) / (2 * 0.05 ** 2) - c2) / gross
            e_one = abs(d1 - 0.01 * c2) / (0.01 * gross)
            worst_sym, worst_one = max(worst_sym, e_sym), max(worst_one, e_one)
            ok_a &= e_sym <= 0.01 and e_one <= 0.05
    report("B2a the second-order coefficient at small amplitude", ok_a, f"16 cases: symmetric estimate at s = 0.05 within {100 * worst_sym:.3f}% of the gross sum (tolerance 1%); one-sided at s = 0.1 within {100 * worst_one:.2f}% (tolerance 5%)")
    print("    At the paper's amplitude.  D = ln R0 - ln R0(constant); 'ours' is the second-order sum; error as a percentage of the exact D")
    print("    seasonality   Model 1: exact D, ours (error)      Model 2: exact D, ours            Model 3 exponential: exact D, ours (error)   Model 3 gamma: exact D, ours (error)")
    ok_b = True
    table = []
    for s in levels:
        row = [f"    {s:3d}%       "]
        for k in kinds:
            c2, gross = second_order_periodic(models[k], coef[s])
            d = math.log(exact[(k, s)] / r0c[k])
            table.append((k, s, exact[(k, s)], r0c[k] * math.exp(c2), d, c2))
            if k == "M2":
                row.append(f"{d:+.4f}, {c2:+.4f}             ")
            else:
                row.append(f"{d:+.4f}, {c2:+.4f} ({100 * (c2 - d) / abs(d):+.0f}%)        ")
                ok_b &= (d < 0) == (c2 < 0) and ((d < 0) if k == "M1" else (d > 0))
        print("  ".join(row))
    report("B2b the formula gives the direction of the effect of seasonality in Models 1 and 3", ok_b, "a loss in Model 1 and a gain in Model 3 (both dormancy distributions) at all nine levels, as in the exact computation")

    # ---- B3
    print("\nB3. Why the sign differs between the models: the generation interval")
    w = OMEGA
    g1 = (khat(models["M1"], 1j * np.array([w, 2 * w, 5 * w])) / khat(models["M1"], 0.0)[0])
    ring = np.array([W_R * W_G / ((W_R + 1j * q) * (W_G + 1j * q)) for q in (w, 2 * w, 5 * w)])
    report("B3 Model 1 is the two-stage ring of the manuscript", float(np.max(np.abs(g1 - ring))) < 1e-12, f"largest difference of the transforms {float(np.max(np.abs(g1 - ring))):.1e}")
    print("    model                              mean (days)  CV^2    (1-CV^2)/2   Re[g/(1-g)] at periods of 12, 6, 4 months")
    for k in kinds:
        mean, cv2 = generation_interval(models[k])
        g = khat(models[k], 1j * np.array([w, 2 * w, 3 * w])) / khat(models[k], 0.0)[0]
        resp = (g / (1 - g)).real
        print(f"    {label[k]:34s} {mean:8.1f}     {cv2:.3f}   {(1 - cv2) / 2:+.3f}       {resp[0]:+.4f}, {resp[1]:+.4f}, {resp[2]:+.4f}")
    print("    Second-order loss of ln R0 per unit variance of ln(vector abundance), host with exponentially fading memory (formula only):")
    mems = (10, 30, 100, 365, 1000, 10000)
    print("    model                              memory (days): " + "  ".join(f"{m:>8d}" for m in mems))
    for k in kinds:
        mean, _ = generation_interval(models[k])
        vals = []
        for m in mems:
            rr = float((khat(models[k], 1 / m) / khat(models[k], 0.0)[0])[0].real)
            vals.append(1 / (mean / m) - rr / (1 - rr))
        print(f"    {label[k]:34s}                " + "  ".join(f"{x:+8.4f}" for x in vals))

    # ---- B4
    print("\nB4. An extension not in the paper: the vector switches at random between 1.5 and 0.5 times its mean, memory 100 days")
    ok = True
    for k, sign in (("M1", -1), ("M3exp", +1), ("M3gam", +1)):
        v, tgt, src = models[k]
        n = v.shape[0]
        f0 = np.zeros((n, n)); f0[tgt, src] = W_PHI / r0c[k]
        mean = f0 - v
        a1, a2 = 1.5 * f0 - v, 0.5 * f0 - v
        r0 = perron(mean)[0]
        total, _ = second_order_matrix(mean, [a1 - a2], 0.01, 0.25)
        gen, _ = generation_interval(models[k])
        rr = float((khat(models[k], 0.01) / khat(models[k], 0.0)[0])[0].real)
        renewal = 0.25 / gen * (rr / (1 - rr) - 1 / (0.01 * gen))
        chains, soj = (4000, 2000) if k != "M3gam" else (3000, 1200)
        ls, se = lyap_sim_powers(a1, a2, 0.005, 0.005, chains=chains, sojourns=soj, seed=20261150 + n)
        ok &= sign * ls / se >= 3 and sign * total > 0
        print(f"    {label[k]:34s} mean environment at its threshold (r0 = {r0:+.1e}); second-order change of growth rate {total:+.4e} per day "
              f"(renewal form {renewal:+.4e}); simulated {ls:+.4e} +- {se:.1e} ({ls / se:+.1f} s.e.); ratio simulated/second-order {ls / total:.2f}")
        if k == "M1":                                  # two stages: the quadrature of Part A gives the exact value
            lq = lyap_quad(a1, a2, 0.005, 0.005)
            m1_check = (abs(ls - lq) <= 4 * se, f"quadrature {lq:+.6e}, simulation {ls:+.6e} +- {se:.1e} ({abs(ls - lq) / se:.1f} s.e.)")
    report("B4 for Model 1 the simulation agrees with the exact quadrature", m1_check[0], m1_check[1])
    report("B4 a randomly switching vector: loss in Model 1, gain in Model 3", ok, "the simulated growth rate at the mean environment's threshold has the sign the formula gives, by at least 3 standard errors, in all three")

    # ---- B5
    print("\nB5. Dormancy tuned to the season (gamma dormancy, K = 11; dormancy d in months of 30 days)")
    ok = True
    for s in (35, 95):
        best = (None, -1.0)
        best2 = (None, -1.0)
        line = []
        for dm in np.arange(0.5, 12.01, 0.5):
            mod = white_model("M3gam", d=30.0 * dm)
            ex = r0_fourier(mod, coef[s])
            c2, _ = second_order_periodic(mod, coef[s])
            so = float(khat(mod, 0.0)[0].real) * math.exp(c2)
            if ex > best[1]:
                best = (dm, ex)
            if so > best2[1]:
                best2 = (dm, so)
            if dm in (1.0, 3.0, 6.0, 9.0, 12.0):
                line.append(f"d = {dm:g}: {ex:.3f} ({so:.3f})")
        ok &= (best[0] <= 2.0) if s == 35 else (5.0 <= best[0] <= 7.0)
        print(f"    seasonality {s}%: exact R0 (second-order value) at " + "; ".join(line))
        print(f"        exact optimum d = {best[0]:g} months (R0 = {best[1]:.3f}), time to first relapse d + 1/f = {best[0] + 72 / 30:.1f} months; second-order optimum d = {best2[0]:g} months (R0 = {best2[1]:.3f})")
    report("B5 the dormancy that maximises R0 lengthens with seasonality", ok, "optimum at most 2 months at 35% seasonality and between 5 and 7 months at 95% (Figure 4b)")
    return table


def main():
    part_a()
    part_b()
    print()
    print(f"SUMMARY: {COUNT['PASS']} PASS, {COUNT['FAIL']} FAIL, {COUNT['DIFFERS FROM PAPER']} printed numbers that DIFFER FROM THE PAPER (listed above; they are not failures of the check)")
    return 0 if COUNT["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
