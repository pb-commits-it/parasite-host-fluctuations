#!/usr/bin/env python3
"""Independent check of the note on hosts that fluctuate together because they interact
(studies/mechanism-linked-hosts/mechanism_note.md): predator and prey, two species under one
weather, competitors.  The note has one derivation and was checked only by its author's script.

Claim tested: every closed form of the note equals the general second-order coefficient computed
here on the model's own matrix; every quoted boundary and number is reproduced by code written
here; the statements said to hold at any noise survive a search for a counterexample.  Nothing in
this file is taken from the note's script, which was not read or run.  The note and its protocol
were read for the model definitions and the formulas only.

The model.  A ring of n stages.  Stage k keeps the share s_k = 1 - 1/tau_k each step and passes
g_k = 1/tau_k to stage k+1 ("transfer k"); tau_k is the stay.  The matrix of mean rates has column
sums one, so it is exactly at its threshold.  A fluctuating transfer is multiplied by
exp(sigma*xi(t) - sigma^2 var(xi)/2), xi(t) a linear combination of the state x(t) of a vector
autoregression x(t+1) = M x(t) + noise; both transfers use the hosts of the same step t.  Transfer 0
is also multiplied by e^y.  The loss D is the y at which the long-run growth rate is zero.
D = sigma^2 K + smaller terms; K is "the second-order coefficient".  "Order 0": host 1 on transfer
0 and host 2 on transfer 1.  "Order 1": the two hosts exchanged.

Own tools (section T checks them against known answers before they are used):
  (a) coef_general(): for any primitive matrix A0, perturbation matrices B_e and drivers
      xi_e = c_e . x,
        C'' = sum_{e,e'} { c_e c_e' rho_ee'(0)/2 - sum_{m>=0} rho_ee'(m+1) w^T B_e T^m Q B_e' v },
      rho_ee'(l) = E xi_e(t) xi_e'(t-l) = c_e^T M^l G c_e', G the stationary covariance, by a
      truncated sum over lags and by the closed form (M x I)(I - M x T)^(-1) (x: Kronecker
      product); K = C'' / (w^T B_birth v).  coef_ring_mp(): the same for a ring, in 40-digit
      arithmetic, for derivatives and limits.
  (b) floquet_root(): the exact loss in a periodic environment (matrix product over one period,
      largest eigenvalue, root in y).
  (c) sim1(): products of the actual random matrices; many variants (stays, y, host assignment)
      on the same host paths; antithetic pairs (hosts x and -x); solve_cases(): the root in y from
      three values of y on the same paths, with a standard error over pairs.
      sim2(): a second simulator written differently (rotation hosts as the real and imaginary part
      of a complex autoregression, competitor hosts from their two modes, each made with a linear
      filter and another random number generator; explicit matrices and batched matrix products;
      independent replicates, no antithetic pairs; normalisation every step; the root by a secant
      iteration on the same random numbers).  A ring without staying is started from one cohort.
  (d) gauss_E(): Gaussian expectations by Gauss-Hermite quadrature, for the one-stage laws.

TOLERANCES, fixed before the first run:
  TOL_ALG   = 1e-8   closed form against the general coefficient, relative to the larger of the
                     two values (absolute floor 1e-12); where the terms of a coefficient nearly
                     cancel, relative to 1e-3 of the sum of the magnitudes of its terms
  TOL_LIMIT = 1e-6   a limit or a leading coefficient obtained by extrapolation, relative
  TOL_FLOQ  = 1e-5   second-order value against the exact periodic loss at small amplitude,
                     relative, after one Richardson step in the amplitude squared
  a quoted number against my number: 0.6 of a unit of its last printed digit
  small-noise simulation: |D_sim - sigma^2 K| <= 4 SE + 5% of |sigma^2 K|
  a simulated number against an exact value: 4 SE; a simulated number of mine against a simulated
  number of the note: 4 combined SE
  empirical covariances: 4 SE + 0.002
  searches for a counterexample: a case counts against a claim when it is on the wrong side by
  more than 4 SE (exact periodic cases: by more than 1e-9); such a case is rerun ten times longer
  with a new seed before it is reported.

A check ends as PASS, FAIL, or DIFFERS FROM THE NOTE.  The last is used only where a number of the
note fails the tolerance above and the cause was traced to the note's printed digits (two cases:
the locations of two minima, wrong in the seventh and eighth decimal).  Simulation uses fixed seeds.

Sections: T tools; V host models; A algebra; Q quoted numbers; CT continuous time; S small noise;
QB boundaries in q; CH one host on two transfers; GRID the main grid; HEAD the order example;
G15 fixed generation time; LAG lag cancellation; PARK parking; B bounds at any noise.

Run:  python3 -I checks/verify_mechanism_linked_hosts.py
      (writes to standard output; the main process and two workers = 3 processes; about 4 minutes
      measured on a ten-core machine, about 10 minutes of processor time)
      --only NAME[,NAME] runs named sections; --list lists them.
Output: results/verify-mechanism-linked-hosts_20261114.txt.  Outputs of single sections that failed
while this file was being written, because of this harness, are kept in
results/verify-mechanism-linked-hosts_attempt1_20261114.txt.  Report:
notes/mechanism-linked-hosts-verification.md.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import math
import time
import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter
import mpmath as mp

TOL_ALG = 1e-8
ABS_FLOOR = 1e-12
TOL_LIMIT = 1e-6
TOL_FLOQ = 1e-5
NSE = 4.0
SMALL_REL = 0.05
COV_ABS = 0.002
PI = math.pi


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------
class Rep:
    def __init__(self, title):
        self.lines = ["", "=" * 100, title, "=" * 100]
        self.checks = []
        self.t0 = time.time()

    def p(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def check(self, key, ok, text):
        self.checks.append((key, bool(ok), text))
        self.p(("  PASS  " if ok else "  FAIL  ") + key + ": " + text)

    def differ(self, key, ok, text):
        """A number of the note compared at the tolerance fixed beforehand, where a failure has been traced
        to the note's printed digits and not to this harness."""
        self.checks.append((key, True if ok else None, text))
        self.p(("  PASS  " if ok else "  DIFFERS FROM THE NOTE  ") + key + ": " + text)

    def done(self):
        self.p("  [section time %.1f s]" % (time.time() - self.t0))
        return "\n".join(self.lines), self.checks


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def close_alg(a, b, mag=None):
    """TOL_ALG relative to the larger value, or to 1e-3 of the summed magnitude of the terms."""
    scale = max(abs(a), abs(b))
    if mag is not None:
        scale = max(scale, 1e-3 * mag)
    return abs(a - b) <= TOL_ALG * scale + ABS_FLOOR


def quoted(mine, text):
    """Compare my number with a number printed in the note: 0.6 of a unit of its last digit."""
    dec = len(text.split(".")[1]) if "." in text else 0
    return abs(mine - float(text)) <= 0.6 * 10.0 ** (-dec)


def fq(mine, text):
    dec = len(text.split(".")[1]) if "." in text else 0
    return ("%." + str(dec + 2) + "f") % mine


def sim_vs_note(mine, se, theirs, se_theirs):
    return abs(mine - theirs) <= NSE * math.hypot(se, se_theirs)


def within_small(Dsim, se, pred):
    return abs(Dsim - pred) <= NSE * se + SMALL_REL * abs(pred)


# --------------------------------------------------------------------------------------------
# hosts: vector autoregressions
# --------------------------------------------------------------------------------------------
def stat_cov(M, S):
    """Stationary covariance G = M G M^T + S of x(t+1) = M x(t) + noise, noise covariance S."""
    d = M.shape[0]
    G = np.linalg.solve(np.eye(d * d) - np.kron(M, M), S.reshape(-1)).reshape(d, d)
    return 0.5 * (G + G.T)


def psd_sqrt(S):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    if w.min() < -1e-10 * max(1.0, abs(w).max()):
        raise ValueError("matrix is not positive semidefinite: smallest eigenvalue %g" % w.min())
    return V * np.sqrt(np.clip(w, 0.0, None))


class Var:
    """x(t+1) = M x(t) + noise with covariance S, rescaled so that every coordinate has variance one."""

    def __init__(self, M, S, standardise=True):
        M = np.array(M, float)
        S = np.array(S, float)
        G = stat_cov(M, S)
        if standardise:
            sd = np.sqrt(np.diag(G))
            M = M * sd[None, :] / sd[:, None]
            S = S / np.outer(sd, sd)
            G = G / np.outer(sd, sd)
        self.M, self.S, self.G, self.d = M, S, G, M.shape[0]
        self.LS = psd_sqrt(S)
        self.LG = psd_sqrt(G)

    def rho(self):
        return float(np.abs(np.linalg.eigvals(self.M)).max())

    def gamma(self, lag):
        """E x(t+lag) x(t)^T for lag >= 0."""
        return np.linalg.matrix_power(self.M, lag) @ self.G


def rot(omega):
    return np.array([[math.cos(omega), -math.sin(omega)], [math.sin(omega), math.cos(omega)]])


def var_predator(r=0.9, omega=PI / 10):
    return Var(r * rot(omega), (1 - r * r) * np.eye(2))


def var_shared(p1=0.25, p2=0.85, gam=0.9):
    off = gam * math.sqrt((1 - p1 * p1) * (1 - p2 * p2))
    return Var(np.diag([p1, p2]), np.array([[1 - p1 * p1, off], [off, 1 - p2 * p2]]))


def var_competitors(a=0.55, c=0.35, dd=0.0):
    return Var(np.array([[a, -c], [-c, a]]), np.array([[1.0, dd], [dd, 1.0]]))


def var_ar1(phi):
    return Var(np.array([[phi]]), np.array([[1 - phi * phi]]))


def two_copies(vp):
    """Two independent copies (x, x') of a two-host autoregression as one of dimension four."""
    Z = np.zeros((2, 2))
    return Var(np.block([[vp.M, Z], [Z, vp.M]]), np.block([[vp.S, Z], [Z, vp.S]]), standardise=False)


def q_rows(q):
    """Host 1 = x_1; host 2 = q x_2 + sqrt(1 - q^2) x'_2: same marginal laws, cross-covariances times q."""
    return np.array([1.0, 0, 0, 0]), np.array([0, q, 0, math.sqrt(max(0.0, 1 - q * q))])


def random_var(rng, d, kind):
    """A random stable autoregression of dimension d, standardised.  kind: 'real' or 'complex' eigenvalues."""
    while True:
        P = np.eye(d) + 0.7 * rng.standard_normal((d, d))
        if np.linalg.cond(P) < 30:
            break
    if kind == "complex":
        r, om = rng.uniform(0.2, 0.97), rng.uniform(0.05, PI - 0.05)
        blocks = [r * rot(om)]
        rest = d - 2
    else:
        blocks = []
        rest = d
    D = np.zeros((d, d))
    pos = 0
    for b in blocks:
        D[pos:pos + 2, pos:pos + 2] = b
        pos += 2
    for _ in range(rest):
        D[pos, pos] = rng.uniform(-0.95, 0.97)
        pos += 1
    M = P @ D @ np.linalg.inv(P)
    Bm = rng.standard_normal((d, d))
    return Var(M, Bm @ Bm.T + 0.05 * np.eye(d))


# --------------------------------------------------------------------------------------------
# tool (a): the general second-order coefficient
# --------------------------------------------------------------------------------------------
def perron(A):
    """Growth factor, right and left Perron vectors (w^T v = 1) of a non-negative matrix."""
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.real))
    lam = float(ev[i].real)
    v = np.abs(V[:, i].real)
    ev2, W = np.linalg.eig(A.T)
    j = int(np.argmax(ev2.real))
    w = np.abs(W[:, j].real)
    v = v / v.sum()
    w = w / (w @ v)
    return lam, v, w


def coef_general(A0, Bs, Cs, M, G, Bbirth, method="resolvent", maxlag=2_000_000, tol=1e-19):
    """Matrix of terms K[e, e'] with D = sigma^2 * K.sum() for the matrix
         A(t) = A0 + sum_e B_e (exp(sigma xi_e(t) - sigma^2 var_e / 2) - 1),    xi_e = Cs[e] . x(t),
    x a stationary autoregression with matrix M and covariance G, and with B_birth multiplied by
    e^y.  Derived here from the first-order recursion of the stage structure:
         growth = log lam0 - sigma^2 C'',
         C'' = sum_{e,e'} { c_e c_e' rho_ee'(0)/2 - sum_{m>=0} rho_ee'(m+1) w^T B_e T^m Q B_e' v },
    with everything divided by lam0, Q = I - v w^T, T = A0 Q / lam0, c_e = w^T B_e v,
    rho_ee'(l) = E xi_e(t) xi_e'(t-l) = Cs[e]^T M^l G Cs[e'].  K = C'' / (w^T B_birth v)."""
    lam, v, w = perron(A0)
    n = A0.shape[0]
    d = M.shape[0]
    E = len(Bs)
    Q = np.eye(n) - np.outer(v, w)
    Tm = (A0 / lam) @ Q
    left = [Bs[e].T @ w / lam for e in range(E)]
    right = [Q @ (Bs[e] @ v) / lam for e in range(E)]
    c = [float(w @ Bs[e] @ v) / lam for e in range(E)]
    terms = np.zeros((E, E))
    if method == "resolvent":
        big = np.eye(d * n) - np.kron(M, Tm)
        MI = np.kron(M, np.eye(n))
        for e2 in range(E):
            sol = MI @ np.linalg.solve(big, np.kron(G @ Cs[e2], right[e2]))
            for e in range(E):
                terms[e, e2] = c[e] * c[e2] * float(Cs[e] @ G @ Cs[e2]) / 2 - float(np.kron(Cs[e], left[e]) @ sol)
    elif method == "lagsum":
        for e2 in range(E):
            u = right[e2].copy()
            gv = M @ (G @ Cs[e2])
            acc = np.zeros(E)
            for m in range(maxlag):
                for e in range(E):
                    acc[e] += float(Cs[e] @ gv) * float(left[e] @ u)
                u = Tm @ u
                gv = M @ gv
                if np.abs(u).max() * np.abs(gv).max() < tol:
                    break
            else:
                raise RuntimeError("lag sum did not converge")
            for e in range(E):
                terms[e, e2] = c[e] * c[e2] * float(Cs[e] @ G @ Cs[e2]) / 2 - acc[e]
    else:
        raise ValueError(method)
    cb = float(w @ Bbirth @ v) / lam
    return terms / cb


def ring(stays):
    stays = np.asarray(stays, float)
    n = len(stays)
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] += 1 - 1 / stays[k]
        A[(k + 1) % n, k] += 1 / stays[k]
    return A


def ring_B(stays, k):
    n = len(stays)
    B = np.zeros((n, n))
    B[(k + 1) % n, k] = 1 / stays[k]
    return B


def K_terms(stays, transfers, Cs, vp, method="resolvent"):
    """Terms of K for a ring: transfers[e] fluctuates with driver Cs[e] . x; birth factor on transfer 0."""
    stays = np.asarray(stays, float)
    Bs = [ring_B(stays, k) for k in transfers]
    Cs = [np.asarray(c, float) for c in Cs]
    return coef_general(ring(stays), Bs, Cs, vp.M, vp.G, ring_B(stays, 0), method=method)


def K_two(stays, vp, order=0, method="resolvent"):
    """The note's main arrangement: transfers 0 and 1 fluctuate; hosts 1, 2 in order 0, exchanged in order 1."""
    e = np.eye(vp.d)
    rows = [e[0], e[1]] if order == 0 else [e[1], e[0]]
    return K_terms(stays, [0, 1], rows, vp, method)


def K_two_q(stays, vp4, q, order=0):
    """As K_two with the cross-covariances multiplied by q (vp4 = two_copies(vp))."""
    h1, h2 = q_rows(q)
    rows = [h1, h2] if order == 0 else [h2, h1]
    return K_terms(stays, [0, 1], rows, vp4)


def mp_kron(A, B):
    out = mp.zeros(A.rows * B.rows, A.cols * B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            aij = A[i, j]
            if aij == 0:
                continue
            for k in range(B.rows):
                for l in range(B.cols):
                    out[i * B.rows + k, j * B.cols + l] = aij * B[k, l]
    return out


def coef_ring_mp(stays, transfers, Cs, M, G, dps=40):
    """The same coefficient for a ring at its threshold in multiple-precision arithmetic.  The
    column sums of the mean matrix are one, so its growth factor is 1 and its left vector is all
    ones; the right vector solves (A - I) v = 0 with sum one.  Returns the matrix of terms."""
    mp.mp.dps = dps
    n = len(stays)
    M = mp.matrix(M)
    G = mp.matrix(G)
    d = M.rows
    E = len(transfers)
    st = [mp.mpf(s) for s in stays]
    A = mp.zeros(n, n)
    for k in range(n):
        A[k, k] += 1 - 1 / st[k]
        A[(k + 1) % n, k] += 1 / st[k]
    Sys = A - mp.eye(n)
    for j in range(n):
        Sys[n - 1, j] = 1
    rhs = mp.zeros(n, 1)
    rhs[n - 1] = 1
    v = mp.lu_solve(Sys, rhs)
    w = mp.ones(1, n)
    Q = mp.eye(n) - v * w
    Tm = A * Q
    big = mp.eye(d * n) - mp_kron(M, Tm)
    MI = mp_kron(M, mp.eye(n))
    Cm = [mp.matrix([mp.mpf(x) for x in c]) for c in Cs]              # column vectors
    left, right, c = [], [], []
    for k in transfers:
        B = mp.zeros(n, n)
        B[(k + 1) % n, k] = 1 / st[k]
        left.append((w * B).T)
        right.append(Q * (B * v))
        c.append((w * B * v)[0])
    terms = mp.zeros(E, E)
    for e2 in range(E):
        sol = MI * mp.lu_solve(big, mp_kron(G * Cm[e2], right[e2]))
        for e in range(E):
            lagsum = (mp_kron(Cm[e], left[e]).T * sol)[0]
            terms[e, e2] = c[e] * c[e2] * (Cm[e].T * G * Cm[e2])[0] / 2 - lagsum
    cb = 1 / st[0] * v[0]
    return terms / cb


def mp_sum(Mx):
    return sum(Mx[i, j] for i in range(Mx.rows) for j in range(Mx.cols))


def mp_rot(r, omega):
    """r times the rotation by omega, and the identity, in multiple precision."""
    r, omega = mp.mpf(r), mp.mpf(omega)
    return mp.matrix([[r * mp.cos(omega), -r * mp.sin(omega)], [r * mp.sin(omega), r * mp.cos(omega)]]), mp.eye(2)


def Kmp_two(stays, M, G, order=0, dps=40):
    rows = [[1, 0], [0, 1]] if order == 0 else [[0, 1], [1, 0]]
    return mp_sum(coef_ring_mp(stays, [0, 1], rows, M, G, dps))


def mp_bisect(f, lo, hi, tol=mp.mpf(10) ** (-22)):
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError("no sign change")
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm == 0:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# --------------------------------------------------------------------------------------------
# the note's formulas, transcribed (these are the things under test)
# --------------------------------------------------------------------------------------------
def f_geo(z, tau):
    """Generating function of a geometric stay of mean tau."""
    return z / (tau - (tau - 1) * z)


def arc(stays, j, i):
    """Stages entered after transfer j through the stage left by transfer i (a full circuit if i = j)."""
    n = len(stays)
    out = []
    k = (j + 1) % n
    while True:
        out.append(k)
        if k == i:
            break
        k = (k + 1) % n
    return out


def note_K_modes(stays, transfers, Cs, vp, changed=None):
    """Formula (7): K = sum over modes n and transfers i, j of (H_n)_ij [ (1+z)/(2T(1-z)) - F_{j->i}(z)/(1-R(z)) ],
    Gamma(l) = sum_n H_n z_n^l.  With `changed`, formula (13): the derivative with respect to that stage's stay."""
    stays = np.asarray(stays, float)
    ev, V = np.linalg.eig(vp.M)
    Vi = np.linalg.inv(V)
    T = stays.sum()
    tot = 0.0 + 0.0j
    mag = 0.0
    for m in range(vp.d):
        z = ev[m]
        H = np.outer(V[:, m], Vi[m, :]) @ vp.G
        R = np.prod([f_geo(z, t) for t in stays])
        for a, i in enumerate(transfers):
            for b, j in enumerate(transfers):
                stages = arc(stays, j, i)
                F = np.prod([f_geo(z, stays[k]) for k in stages])
                h = np.asarray(Cs[a], float) @ H @ np.asarray(Cs[b], float)
                if changed is None:
                    val = h * ((1 + z) / (2 * T * (1 - z)) - F / (1 - R))
                else:
                    bb = stays[changed]
                    L = -(1 - z) / (bb - (bb - 1) * z)
                    Rp = R * L
                    Fp = F * L if changed in stages else 0.0
                    val = h * (-(1 + z) / (2 * T * T * (1 - z)) - (Fp * (1 - R) + F * Rp) / (1 - R) ** 2)
                tot += val
                mag += abs(val)
    return tot.real, tot.imag, mag


def note_Kplus(a, b, phi):
    th = phi / (1 - phi)
    return (2 * a * b + th * (2 * (a + b) - (a - b) ** 2)) / ((a + b) * (a * b + th * (a + b)))


def note_Kminus(a, b, phi):
    th = phi / (1 - phi)
    return th * (a + b) / (a * b + th * (a + b))


def note_gain_condition(a, b, phi):
    """(11): a common-driver gain only if (a-b)^2 > 2(a+b) and theta > 2ab / ((a-b)^2 - 2(a+b))."""
    th = phi / (1 - phi)
    den = (a - b) ** 2 - 2 * (a + b)
    return den > 0 and th > 2 * a * b / den


def note_long_stay(r, omega, a, c=5.0):
    """(15) for the ring (b, a, c), order 0, rotation hosts: K(b) = -Im A + J/b + ...; returns (-Im A, J, real part of J)."""
    z = r * np.exp(1j * omega)
    thz = z / (1 - z)
    A = f_geo(z, a)
    B = f_geo(z, c)
    re = ((1 + z) / (1 - z) - 2 * A * B * thz).real
    im = ((A * A - 1) * B * thz).imag
    return -A.imag, re - im, re


def manuscript_continuous(x1, x2, k1, k2):
    """The manuscript's continuous-time second-order cost per generation of one host on the two transfers
    of a two-stage ring, per unit sigma^2; x_k = (rate at which the host forgets) x (mean stay in stage k)."""
    return ((k1 + k2) ** 2 * x1 * x2 - k1 * k2 * (x1 + x2) ** 2) / ((x1 + x2) * (x1 + x2 + x1 * x2))


def renewal_u(stays, j, i, L):
    """u_{j->i}(1..L): coefficients of F_{j->i}(z) / (1 - R(z)), by polynomial division (my own route:
    the stay of mean tau has generating function g z / (1 - s z))."""
    stays = np.asarray(stays, float)
    n = len(stays)
    s, g = 1 - 1 / stays, 1 / stays
    inarc = set(arc(stays, j, i))
    num = np.array([1.0])
    den_a = np.array([1.0])
    den_b = np.array([1.0])
    for k in range(n):
        den_a = np.convolve(den_a, [1.0, -s[k]])
        den_b = np.convolve(den_b, [0.0, g[k]])
        num = np.convolve(num, [0.0, g[k]] if k in inarc else [1.0, -s[k]])
    den = den_a - den_b
    imp = np.zeros(L + 1)
    imp[0] = 1.0
    out = lfilter(num, den, imp)
    return out[1:]


# --------------------------------------------------------------------------------------------
# tool (b): exact loss in a periodic environment
# --------------------------------------------------------------------------------------------
def floquet_growth(A0, masks, hseq, birth_mask, y):
    """Growth rate per step of the periodic product; A(t) = A0 * prod_e h_e(t)^mask_e * e^(y birth_mask)."""
    P = len(hseq[0])
    n = A0.shape[0]
    Ay = A0 * np.exp(y * birth_mask)
    prod = np.eye(n)
    logscale = 0.0
    for t in range(P):
        A = Ay.copy()
        for mk, h in zip(masks, hseq):
            A = A * (1 + mk * (h[t] - 1))
        prod = A @ prod
        sc = np.abs(prod).max()
        prod /= sc
        logscale += math.log(sc)
    return (math.log(np.abs(np.linalg.eigvals(prod)).max()) + logscale) / P


def floquet_root(A0, masks, hseq, birth_mask, lo=-6.0, hi=6.0):
    f = lambda y: floquet_growth(A0, masks, hseq, birth_mask, y)
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=300)


def ring_masks(n, transfers):
    out = []
    for k in transfers:
        m = np.zeros((n, n))
        m[(k + 1) % n, k] = 1.0
        out.append(m)
    return out


def periodic_hosts(amp, logseqs):
    """Multipliers exp(amp * log sequence), each divided by its own mean over the period."""
    out = []
    for q in logseqs:
        h = np.exp(amp * np.asarray(q, float))
        out.append(h / h.mean())
    return out


# --------------------------------------------------------------------------------------------
# tool (c): simulators
# --------------------------------------------------------------------------------------------
def host_paths(vp, rng, X, m):
    """m further steps of P independent stationary paths; returns the states at steps t..t+m-1 and the next state."""
    P, d = X.shape
    Z = rng.standard_normal((m, P, d)) @ vp.LS.T
    Xp = np.empty((m, P, d))
    MT = vp.M.T
    for u in range(m):
        Xp[u] = X
        X = X @ MT + Z[u]
    return Xp, X


def sim1(vp, stays, ys, Cs, sigma, steps, burn, npairs, seed, start_one=False, every=4):
    """Products of the actual random matrices.  stays (V, n); ys (V,); Cs (V, n, d): the driver of
    transfer k in variant v is Cs[v, k] . x(t) (a zero row: steady transfer).  All V variants see
    the same host paths.  Each path is used twice, with x and with -x (an antithetic pair).
    Returns the growth rate per step, averaged over each pair: (npairs, V)."""
    rng = np.random.default_rng(seed)
    stays = np.asarray(stays, float)
    ys = np.asarray(ys, float)
    Cs = np.asarray(Cs, float)
    V, n = stays.shape
    P = npairs
    s = [(1 - 1 / stays[:, k])[None, :] for k in range(n)]
    g = [(1 / stays[:, k])[None, :] for k in range(n)]
    ey = np.exp(ys)[None, :]
    varc = np.einsum("vkd,de,vke->kv", Cs, vp.G, Cs)
    fluct = [bool(np.any(Cs[:, k, :] != 0)) for k in range(n)]
    uniq = []
    for k in range(n):
        if fluct[k]:
            u, inv = np.unique(Cs[:, k, :], axis=0, return_inverse=True)
            uniq.append((u, np.asarray(inv).reshape(-1)))
        else:
            uniq.append(None)
    const = [g[k] * (ey if k == 0 else 1.0) * np.exp(-sigma * sigma * varc[k][None, :] / 2) for k in range(n)]
    if start_one:
        x = [np.full((2 * P, V), 1.0 if k == 0 else 0.0) for k in range(n)]
    else:
        x = [np.full((2 * P, V), 1.0 / n) for k in range(n)]
    X = rng.standard_normal((P, vp.d)) @ vp.LG.T
    acc = np.zeros((2 * P, V))
    total = burn + steps
    assert burn % every == 0 and steps % every == 0
    chunk = max(every, min(1000, (3_000_000 // (2 * P * V)) // every * every))
    t = 0
    while t < total:
        m = min(chunk, total - t)
        Xp, X = host_paths(vp, rng, X, m)
        Gs = []
        for k in range(n):
            if fluct[k]:
                u, inv = uniq[k]
                e = np.exp(sigma * (Xp @ u.T))
                e = np.concatenate([e, 1.0 / e], axis=1)
                Gs.append(e[:, :, inv] * const[k])
            else:
                Gs.append(None)
        for q in range(m):
            new = []
            for k in range(n):
                km = k - 1 if k > 0 else n - 1
                inflow = Gs[km][q] * x[km] if fluct[km] else const[km] * x[km]
                new.append(s[k] * x[k] + inflow)
            x = new
            if (t + q + 1) % every == 0:
                tot = x[0]
                for k in range(1, n):
                    tot = tot + x[k]
                if t + q >= burn:
                    acc += np.log(tot)
                x = [xk / tot for xk in x]
        t += m
    r = acc / steps
    return 0.5 * (r[:P] + r[P:])


def quad_root(rm, centre, w):
    """Root of the parabola through (centre - w, centre, centre + w) nearest the centre; and the slope there."""
    r_lo, r_0, r_hi = rm[..., 0], rm[..., 1], rm[..., 2]
    c1 = (r_hi - r_lo) / (2 * w)
    c2 = (r_hi - 2 * r_0 + r_lo) / (2 * w * w)
    disc = c1 * c1 - 4 * c2 * r_0
    u = np.where(disc > 0, -2 * r_0 / (c1 + np.sqrt(np.abs(disc))), -r_0 / c1)
    return centre + u, c1 + 2 * c2 * u, u, c1, c2


class Roots:
    def __init__(self, D, se, infl, slope, centre, passes):
        self.D, self.se, self.infl, self.slope, self.centre, self.passes = D, se, infl, slope, centre, passes

    def diff(self, a, b):
        dd = self.infl[:, a] - self.infl[:, b]
        return float(self.D[a] - self.D[b]), float(dd.std(ddof=1) / math.sqrt(len(dd)))

    def combo(self, weights):
        """A linear combination of the losses of the cases, with its paired standard error."""
        wv = np.asarray(weights, float)
        dd = self.infl @ wv
        return float(self.D @ wv), float(dd.std(ddof=1) / math.sqrt(len(dd)))


def diff_runs(Ra, a, Rb, b):
    """Difference of two losses from two runs that used the same host paths (same seed and size)."""
    dd = Ra.infl[:, a] - Rb.infl[:, b]
    return float(Ra.D[a] - Rb.D[b]), float(dd.std(ddof=1) / math.sqrt(len(dd)))


def solve_cases(vp, cases, sigma, steps=40000, burn=4000, npairs=128, seed=1, start_one=False):
    """Loss D (the root in y) of each case, all on the same host paths.  A case is a dict with
    'stays' (n,), 'C' (n, d) and optionally 'guess'.  A short pilot on other random numbers centres
    three values of y; the root is that of the parabola through the pair-averaged growth rates."""
    nc = len(cases)
    stays = np.array([c["stays"] for c in cases], float)
    Cm = np.array([c["C"] for c in cases], float)
    centre = np.array([c.get("guess", 0.0) for c in cases], float)

    def run3(cen, w, st, npr, sd):
        ys = (cen[:, None] + np.array([-w, 0.0, w])[None, :]).reshape(-1)
        r = sim1(vp, np.repeat(stays, 3, axis=0), ys, np.repeat(Cm, 3, axis=0), sigma, st, burn, npr, sd, start_one)
        return r.reshape(r.shape[0], nc, 3)

    w = 0.3 * sigma * sigma
    pst = max(4000, (steps // 4) // 4 * 4)
    for _ in range(6):
        r = run3(centre, w, pst, max(16, npairs // 2), seed + 10007)
        root, _, u, _, _ = quad_root(r.mean(0), centre, w)
        ok = np.abs(u) <= w
        centre = centre + np.clip(u, -3 * w, 3 * w)
        if ok.all():
            break
    delta = 0.05 * sigma * sigma
    passes = 0
    for _ in range(5):
        passes += 1
        r = run3(centre, delta, steps, npairs, seed)
        root, slope, u, c1, c2 = quad_root(r.mean(0), centre, delta)
        ok = np.abs(u) <= 1.5 * delta
        if ok.all():
            break
        centre = np.where(ok, centre, centre + np.clip(u, -20 * delta, 20 * delta))
    if not ok.all():
        raise RuntimeError("root in y not bracketed by the three values of y after %d passes" % passes)
    rp_lo, rp_0, rp_hi = r[..., 0], r[..., 1], r[..., 2]
    c1p = (rp_hi - rp_lo) / (2 * delta)
    c2p = (rp_hi - 2 * rp_0 + rp_lo) / (2 * delta * delta)
    at_root = rp_0 + c1p * u[None, :] + c2p * u[None, :] ** 2
    infl = -(at_root - at_root.mean(0)[None, :]) / slope[None, :]
    se = infl.std(axis=0, ddof=1) / math.sqrt(r.shape[0])
    return Roots(root, se, infl, slope, centre, passes)


def case(stays, rows, guess=0.0):
    """A case for solve_cases: rows[k] is the driver row of transfer k (None: steady)."""
    d = max(len(r) for r in rows if r is not None)
    C = np.zeros((len(stays), d))
    for k, r in enumerate(rows):
        if r is not None:
            C[k] = r
    return {"stays": list(stays), "C": C, "guess": guess}


def two_host_case(stays, vp, order, sigma, kappa=(1.0, 1.0)):
    e = np.eye(vp.d)
    rows = [e[0] * kappa[0], e[1] * kappa[1]] if order == 0 else [e[1] * kappa[0], e[0] * kappa[1]]
    K = K_terms(stays, [0, 1], rows, vp).sum()
    return case(stays, rows + [None] * (len(stays) - 2), sigma * sigma * K), K


class ComplexAR:
    """Rotation hosts: Z(t+1) = r e^(i omega) Z(t) + sqrt(1 - r^2)(e1 + i e2), stationary, made chunk by
    chunk with a linear filter.  Real part: host 1.  Imaginary part: host 2."""

    def __init__(self, r_, omega, nrep, rng):
        self.zc = r_ * np.exp(1j * omega)
        self.sd = math.sqrt(1 - r_ * r_)
        self.rng, self.nrep = rng, nrep
        self.start = rng.standard_normal(nrep) + 1j * rng.standard_normal(nrep)

    def hosts(self, m):
        eps = self.sd * (self.rng.standard_normal((m, self.nrep)) + 1j * self.rng.standard_normal((m, self.nrep)))
        drive = np.concatenate([self.start[None, :], eps[:-1]], axis=0)       # Z[0] = start, Z[u] = zc Z[u-1] + eps[u-1]
        Z = lfilter([1.0], [1.0, -self.zc], drive, axis=0)
        self.start = self.zc * Z[-1] + eps[-1]
        return Z.real, Z.imag


class ModeAR:
    """Competitor hosts from their two modes: x1 = sqrt(w+) U + sqrt(w-) W, x2 = sqrt(w+) U - sqrt(w-) W, with U
    and W independent one-host series of memory a - c and a + c and the weights of the note's (4) (checked
    against the autoregression in sections V2 and V8).  Each mode is made with a linear filter."""

    def __init__(self, a, c, dd, nrep, rng):
        lp, lm = a - c, a + c
        vpl, vmi = (1 + dd) / (1 - lp * lp), (1 - dd) / (1 - lm * lm)
        self.w = (vpl / (vpl + vmi), vmi / (vpl + vmi))
        self.lam = (lp, lm)
        self.rng, self.nrep = rng, nrep
        self.start = [rng.standard_normal(nrep), rng.standard_normal(nrep)]

    def hosts(self, m):
        out = []
        for i in (0, 1):
            lam = self.lam[i]
            eps = math.sqrt(1 - lam * lam) * self.rng.standard_normal((m, self.nrep))
            drive = np.concatenate([self.start[i][None, :], eps[:-1]], axis=0)
            U = lfilter([1.0], [1.0, -lam], drive, axis=0)
            self.start[i] = lam * U[-1] + eps[-1]
            out.append(U)
        a_, b_ = math.sqrt(self.w[0]) * out[0], math.sqrt(self.w[1]) * out[1]
        return a_ + b_, a_ - b_


def rot_gen(r_, omega):
    return lambda nrep, rng: ComplexAR(r_, omega, nrep, rng)


def comp_gen(a, c, dd):
    return lambda nrep, rng: ModeAR(a, c, dd, nrep, rng)


def sim2(gen, order, stays, y, sigma, steps, burn, nrep, seed, start_one=False, fluct=(True, True)):
    """Second simulator, written differently.  Two hosts from `gen` (linear filters, another random
    number generator) on transfers 0 and 1 (order 1: exchanged; fluct: which of the two transfers
    fluctuate).  The n x n matrix of every replicate is written out each step and applied with a
    batched matrix product; the population is normalised every step; replicates are independent (no
    antithetic pairs).  Returns the growth rate per step of each replicate."""
    rng = np.random.Generator(np.random.Philox(seed))
    stays = np.asarray(stays, float)
    n = len(stays)
    base = ring(stays)
    A = np.broadcast_to(base, (nrep, n, n)).copy()
    x = np.zeros((nrep, n, 1))
    if start_one:
        x[:, 0, 0] = 1.0
    else:
        x[:] = 1.0 / n
    g = gen(nrep, rng)
    acc = np.zeros(nrep)
    total = burn + steps
    g0, g1 = base[1 % n, 0] * math.exp(y), base[2 % n, 1]
    half = sigma * sigma / 2
    t = 0
    while t < total:
        m = min(2000, total - t)
        x1, x2 = g.hosts(m)
        h1 = np.exp(sigma * x1 - half)
        h2 = np.exp(sigma * x2 - half)
        ha, hb = (h1, h2) if order == 0 else (h2, h1)
        for u in range(m):
            if fluct[0]:
                A[:, 1 % n, 0] = g0 * ha[u]
            else:
                A[:, 1 % n, 0] = g0
            if fluct[1]:
                A[:, 2 % n, 1] = g1 * hb[u]
            x = A @ x
            tot = x.sum(axis=1, keepdims=True)
            if t + u >= burn:
                acc += np.log(tot[:, 0, 0])
            x = x / tot
        t += m
    return acc / steps


def sim2_root(gen, order, stays, sigma, steps, burn, nrep, seed, y0=0.0, fluct=(True, True)):
    """Root in y by a secant iteration on the same random numbers; standard error over replicates.
    Returns D, its standard error, the number of passes, and each replicate's contribution to D."""
    T = float(np.sum(stays))
    f = lambda y: sim2(gen, order, stays, y, sigma, steps, burn, nrep, seed, fluct=fluct)
    ya = y0
    ra = f(ya)
    yb = ya - T * ra.mean()
    rb = f(yb)
    passes = 2
    for _ in range(6):
        slope = (rb.mean() - ra.mean()) / (yb - ya)
        yc = yb - rb.mean() / slope
        if abs(yc - yb) < 2e-5 * max(sigma * sigma, 1e-3):
            break
        ya, ra = yb, rb
        yb = yc
        rb = f(yb)
        passes += 1
    slope = (rb.mean() - ra.mean()) / (yb - ya)
    D = yb - rb.mean() / slope
    infl = -(rb - rb.mean()) / slope
    se = infl.std(ddof=1) / math.sqrt(nrep)
    return float(D), float(se), passes, infl


# --------------------------------------------------------------------------------------------
# tool (d): Gaussian expectation
# --------------------------------------------------------------------------------------------
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(200)
_GH_W = _GH_W / _GH_W.sum()


def gauss_E(f):
    return float(np.sum(_GH_W * f(_GH_X)))


def scalar_root(s, sigma, share=1.0):
    """Root y of E log( s + (1 - s) e^(share*y) exp(sigma Z - sigma^2/2) ) = 0."""
    f = lambda y: gauss_E(lambda z: np.log(s + (1 - s) * np.exp(share * y + sigma * z - sigma * sigma / 2)))
    return brentq(f, -20, 20, xtol=1e-14)


# --------------------------------------------------------------------------------------------
# random rings for the algebra
# --------------------------------------------------------------------------------------------
def rand_stays(rng, n, lo=1.0, hi=60.0, p_one=0.1):
    while True:
        st = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
        st[rng.random(n) < p_one] = 1.0
        if st.max() > 1.05:
            return st


# ============================================================================================
# SECTION T: the tools against known answers
# ============================================================================================
def sec_T():
    rp = Rep("T. The tools, against known answers, before they are used")
    rng = np.random.default_rng(101)

    # T1 lag sum against the closed (Kronecker) form on random dense matrices
    worst = 0.0
    kinds = {"real": 0, "complex": 0}
    for it in range(200):
        n = int(rng.integers(2, 6))
        A0 = rng.uniform(0.05, 1.0, (n, n))
        A0 /= perron(A0)[0]
        E = int(rng.integers(1, 4))
        d = int(rng.integers(1, 5))
        kind = "complex" if (d >= 2 and rng.random() < 0.5) else "real"
        kinds[kind] += 1
        vp = random_var(rng, d, kind)
        Bs = [A0 * (rng.random((n, n)) < 0.4) * rng.uniform(0.2, 1.0, (n, n)) for _ in range(E)]
        for B in Bs:
            if B.sum() == 0:
                B[0, 0] = A0[0, 0]
        Cs = [rng.standard_normal(d) for _ in range(E)]
        Bb = A0 * (rng.random((n, n)) < 0.5)
        Bb[1 % n, 0] = A0[1 % n, 0]
        a = coef_general(A0, Bs, Cs, vp.M, vp.G, Bb, "resolvent")
        b = coef_general(A0, Bs, Cs, vp.M, vp.G, Bb, "lagsum")
        worst = max(worst, np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    rp.check("T1", worst < 1e-9, "general coefficient, sum over lags against the closed Kronecker form, 200 random dense "
             "matrices (2 to 5 stages, 1 to 3 perturbed parts, autoregressions of dimension 1 to 4; %d with real and %d "
             "with complex eigenvalues): largest relative difference of any term %.1e (criterion 1e-9)"
             % (kinds["real"], kinds["complex"], worst))

    # T2 double precision against 40 digits on rings
    worst = 0.0
    for it in range(60):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n)
        vp = random_var(rng, 2, "complex" if it % 2 else "real")
        tr = sorted(rng.choice(n, 2, replace=False).tolist())
        rows = [np.array([1.0, 0]), np.array([0, 1.0])]
        a = K_terms(stays, tr, rows, vp)
        b = coef_ring_mp(stays, tr, rows, vp.M.tolist(), vp.G.tolist())
        bf = np.array([[float(b[i, j]) for j in range(2)] for i in range(2)])
        worst = max(worst, np.abs(a - bf).max() / np.abs(bf).max())
    rp.check("T2", worst < 1e-9, "general coefficient on rings, double precision against 40-digit arithmetic with the "
             "Perron vectors obtained differently, 60 random rings: largest relative difference of any term %.1e "
             "(criterion 1e-9)" % worst)

    # T3 general coefficient against the exact periodic loss at small amplitude (dense matrices and rings)
    worst = 0.0
    nring = 0
    keep = None
    for it in range(60):
        n = int(rng.integers(2, 5))
        per = int(rng.integers(3, 25))
        om = 2 * PI / per * int(rng.integers(1, max(2, per // 2)))
        if it % 2 == 0:
            A0 = rng.uniform(0.05, 1.0, (n, n))
            A0 /= perron(A0)[0]
            E = int(rng.integers(1, 4))
            cells = rng.permutation(n * n)
            masks = []
            for e in range(E):
                mk = np.zeros(n * n)
                mk[cells[e::E][: int(rng.integers(1, 3))]] = 1.0
                masks.append(mk.reshape(n, n))
            bmask = (rng.random((n, n)) < 0.4).astype(float)
            bmask[1 % n, 0] = 1.0
        else:
            nring += 1
            stays = rand_stays(rng, n, 1.0, 20.0)
            A0 = ring(stays)
            tr = rng.choice(n, min(n, int(rng.integers(1, 3))), replace=False).tolist()
            E = len(tr)
            masks = ring_masks(n, tr)
            bmask = ring_masks(n, [0])[0]
        Cs = [rng.standard_normal(2) for _ in range(E)]
        tt = np.arange(per)
        xs = np.stack([math.sqrt(2) * np.cos(om * tt), math.sqrt(2) * np.sin(om * tt)], axis=1)
        logs = [xs @ c for c in Cs]
        Ks = coef_general(A0, [A0 * mk for mk in masks], Cs, rot(om), np.eye(2), A0 * bmask).sum()
        est = []
        for amp in (0.02, 0.01):
            # the mean of the losses at +amp and -amp: a sinusoid sampled three times per period is not
            # symmetric under a change of sign and has a term of third order, which Gaussian hosts lack
            D = 0.5 * (floquet_root(A0, masks, periodic_hosts(amp, logs), bmask)
                       + floquet_root(A0, masks, periodic_hosts(-amp, logs), bmask))
            est.append(D / amp ** 2)
        rich = (4 * est[1] - est[0]) / 3
        err = abs(rich - Ks) / max(abs(Ks), 1e-3 * sum(abs(e) for e in est) / 2)
        if err > worst:
            worst = err
            keep = (A0, masks, logs, bmask, Ks, est[1])
    A0, masks, logs, bmask, Ks, e1 = keep
    e2 = 0.5 * (floquet_root(A0, masks, periodic_hosts(0.005, logs), bmask)
                + floquet_root(A0, masks, periodic_hosts(-0.005, logs), bmask)) / 0.005 ** 2
    finer = abs((4 * e2 - e1) / 3 - Ks) / abs(Ks)
    rp.check("T3", worst < TOL_FLOQ, "general coefficient against the exact loss in a periodic environment at small "
             "amplitude (two sinusoidal hosts a quarter period apart, random loadings; 30 random dense matrices and "
             "%d random rings, 1 to 3 fluctuating parts; mean of the losses at plus and minus the amplitude): largest "
             "relative difference %.1e (in that case amplitudes half as large give %.1e: what is left is the next "
             "order in the amplitude, not a difference in the coefficient). This fixes the direction of the lag and "
             "the conversion from growth per step to loss" % (nring, worst, finer))

    # T4 it does not matter which transfer of a ring carries the factor e^y (exact, periodic)
    worst = 0.0
    for it in range(40):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n, 1.0, 20.0)
        per = int(rng.integers(2, 20))
        tr = rng.choice(n, min(n, 2), replace=False).tolist()
        logs = [rng.standard_normal(per) for _ in tr]
        hs = periodic_hosts(rng.uniform(0.3, 1.5), logs)
        D0 = floquet_root(ring(stays), ring_masks(n, tr), hs, ring_masks(n, [0])[0])
        Dk = floquet_root(ring(stays), ring_masks(n, tr), hs, ring_masks(n, [int(rng.integers(0, n))])[0])
        worst = max(worst, abs(D0 - Dk))
    rp.check("T4", worst < 1e-10, "the loss does not depend on which transfer of the ring carries e^y: 40 random rings "
             "in random periodic environments of large amplitude, largest difference %.1e" % worst)

    # T5 simulator 1 against the one-stage law (exact by quadrature), with memory
    txt = []
    ok = True
    for sig, tau, phi, sd in ((1.0, 3.0, 0.7, 11), (0.3, 3.0, 0.7, 12), (1.0, 8.0, -0.5, 13)):
        exact = scalar_root(1 - 1 / tau, sig)
        R = solve_cases(var_ar1(phi), [case([tau], [np.array([1.0])], exact)], sig, seed=sd, npairs=1024)
        ok &= abs(R.D[0] - exact) <= NSE * R.se[0]
        txt.append("sigma %.1f stay %g memory %g: %.6f (SE %.6f) against %.6f" % (sig, tau, phi, R.D[0], R.se[0], exact))
    rp.check("T5", ok, "simulator 1 against the one-stage loss by quadrature (which does not depend on memory; 1,024 "
             "pairs x 40,000 steps): " + "; ".join(txt))

    # T6 simulator 1 against the manuscript's one-host ring formula at small noise
    txt = []
    ok = True
    for stays, phi, sd in (((5.0, 2.0), 0.5, 21), ((5.0, 20.0, 3.0), 0.9, 22)):
        T = sum(stays)
        th = phi / (1 - phi)
        Rr = np.prod([1 / (1 + t / th) for t in stays])
        want = (1 + 2 * th) / (2 * T) - Rr / (1 - Rr)
        mine = K_terms(stays, [0], [np.array([1.0])], var_ar1(phi)).sum()
        R = solve_cases(var_ar1(phi), [case(stays, [np.array([1.0])] + [None] * (len(stays) - 1), 0.0025 * mine)], 0.05, seed=sd)
        ok &= relerr(mine, want) < 1e-10 and within_small(R.D[0], R.se[0], 0.0025 * want)
        txt.append("stays %s memory %g: general coefficient %.10f, known closed form %.10f, simulated D/sigma^2 %.5f (SE %.5f)"
                   % (stays, phi, mine, want, R.D[0] / 0.0025, R.se[0] / 0.0025))
    rp.check("T6", ok, "one host on one transfer (the project's known formula (1 + 2 theta)/(2T) - R/(1 - R)): " + "; ".join(txt))

    # T7 both simulators on the ring without staying, where the loss is exactly the sum of sigma^2/2
    vp = var_predator()
    r1 = sim1(vp, np.ones((1, 3)), [1.0], np.array([[[1.0, 0], [0, 1.0], [0, 0]]]), 1.0, 12000, 0, 32, 5, start_one=True)
    r1b = sim1(vp, np.ones((1, 3)), [1.0], np.array([[[1.0, 0], [0, 1.0], [0, 0]]]), 1.0, 12000, 0, 32, 5, start_one=False)
    r2 = sim2(rot_gen(0.9, PI / 10), 0, [1.0, 1.0, 1.0], 1.0, 1.0, 12000, 0, 256, 6, start_one=True)
    se2 = r2.std(ddof=1) / math.sqrt(len(r2))
    rp.check("T7", abs(r1.mean()) < 1e-12 and abs(r2.mean()) <= NSE * se2,
             "ring without staying, stays (1,1,1), two linked hosts, sigma = 1: growth per step at y = 1 (the sum of "
             "sigma^2/2) started from one cohort: simulator 1 %.2e (antithetic pairs cancel exactly), simulator 2 "
             "%.2e (SE %.1e). Started in every stage simulator 1 gives %.5f (SE %.5f): the bias of a population "
             "of cohorts that never mix, which is why such a ring is started from one cohort"
             % (r1.mean(), r2.mean(), se2, r1b.mean(), r1b.std(ddof=1) / math.sqrt(len(r1b))))
    return rp.done()


# ============================================================================================
# SECTION V: the host models are stationary and their covariances are valid at all lags
# ============================================================================================
def note_gamma_predator(r, om, l):
    return r ** l * rot(l * om)


def note_gamma_shared(p1, p2, gam, l):
    c = gam * math.sqrt((1 - p1 * p1) * (1 - p2 * p2)) / (1 - p1 * p2)
    return np.diag([p1 ** l, p2 ** l]) @ np.array([[1.0, c], [c, 1.0]])


def note_gamma_comp(a, c, dd, l):
    lp, lm = a - c, a + c
    vpl, vmi = (1 + dd) / (1 - lp * lp), (1 - dd) / (1 - lm * lm)
    wp, wm = vpl / (vpl + vmi), vmi / (vpl + vmi)
    return wp * lp ** l * np.ones((2, 2)) + wm * lm ** l * np.array([[1.0, -1.0], [-1.0, 1.0]])


def host_models():
    out = [("predator and prey, radius 0.9, angle pi/10", var_predator(), lambda l: note_gamma_predator(0.9, PI / 10, l)),
           ("fast rotation, radius 0.9, angle pi/2", var_predator(0.9, PI / 2), lambda l: note_gamma_predator(0.9, PI / 2, l)),
           ("shared weather (0.25, 0.85, 0.9)", var_shared(), lambda l: note_gamma_shared(0.25, 0.85, 0.9, l)),
           ("competitors (0.55, 0.35, 0)", var_competitors(0.55, 0.35, 0.0), lambda l: note_gamma_comp(0.55, 0.35, 0.0, l)),
           ("fast competitors (0.02, 0.02, -0.8)", var_competitors(0.02, 0.02, -0.8), lambda l: note_gamma_comp(0.02, 0.02, -0.8, l)),
           ("competitors (0.15, 0.05, -0.8)", var_competitors(0.15, 0.05, -0.8), lambda l: note_gamma_comp(0.15, 0.05, -0.8, l))]
    return out


def spectral_min_eig(gam, rho, q=1.0, nfreq=720):
    """Smallest eigenvalue over a grid of frequencies of sum_l Gamma(l) e^(-i lambda l), Gamma(-l) = Gamma(l)^T,
    with the cross-covariances multiplied by q."""
    L = int(min(40000, math.ceil(40.0 / max(1e-3, 1 - rho))))
    lam = np.linspace(0, 2 * PI, nfreq, endpoint=False)
    f = np.zeros((nfreq, 2, 2), complex)
    for l in range(L + 1):
        Gl = np.array(gam(l), float)
        Gl[0, 1] *= q
        Gl[1, 0] *= q
        ph = np.exp(-1j * lam * l)[:, None, None]
        if l == 0:
            f += Gl[None]
        else:
            f += Gl[None] * ph + Gl.T[None] * np.conj(ph)
    herm = np.abs(f - np.conj(np.transpose(f, (0, 2, 1)))).max()
    return float(np.linalg.eigvalsh(0.5 * (f + np.conj(np.transpose(f, (0, 2, 1))))).min()), herm


def emp_cov(Xp, gam, maxlag=25):
    """Lag covariances from simulated paths Xp (steps, paths, d) against gam(l) = E x(t+l) x(t)^T."""
    m, P, d = Xp.shape
    worst_excess, worst_z = -1e9, 0.0
    for l in range(maxlag + 1):
        prod = np.einsum("tpi,tpj->pij", Xp[l:], Xp[: m - l]) / (m - l)
        mean = prod.mean(0)
        se = prod.std(0, ddof=1) / math.sqrt(P)
        dev = np.abs(mean - gam(l))
        worst_excess = max(worst_excess, float((dev - (NSE * se + COV_ABS)).max()))
        worst_z = max(worst_z, float((dev / np.maximum(se, 1e-300)).max()))
    return worst_excess, worst_z


def sec_V():
    rp = Rep("V. The host models: stationary, with a valid (positive semidefinite) covariance at all lags")
    models = host_models()
    fam = [(r, w) for r in (0.3, 0.6, 0.9, 0.98) for w in (0.1, 0.3, 0.5, 0.7, 0.9)]
    allv = [(nm, vp, gm) for nm, vp, gm in models]
    for r, w in fam:
        allv.append(("rotation %g, %g pi" % (r, w), var_predator(r, w * PI), (lambda l, r=r, w=w: note_gamma_predator(r, w * PI, l))))
    for phi in (0.0, 0.2, 0.8):
        allv.append(("one host, memory %g" % phi, var_ar1(phi), None))

    rho_max, s_min, diag_dev, lyap = 0.0, 1e9, 0.0, 0.0
    for nm, vp, gm in allv:
        rho_max = max(rho_max, vp.rho())
        s_min = min(s_min, float(np.linalg.eigvalsh(vp.S).min()))
        diag_dev = max(diag_dev, float(np.abs(np.diag(vp.G) - 1).max()))
        lyap = max(lyap, float(np.abs(vp.G - vp.M @ vp.G @ vp.M.T - vp.S).max()))
    rp.check("V1", rho_max < 1 and s_min > -1e-12 and diag_dev < 1e-12 and lyap < 1e-12,
             "%d autoregressions (the six named models, the 20 rotations of the note's wider family, three one-host "
             "series): largest modulus of an eigenvalue of M %.4f (< 1: stationary); smallest eigenvalue of the "
             "innovation covariance G - M G M^T %.3e (not negative); variances differ from one by at most %.1e; "
             "residual of G = M G M^T + S %.1e" % (len(allv), rho_max, s_min, diag_dev, lyap))

    worst = 0.0
    for nm, vp, gm in allv:
        if gm is None:
            continue
        for l in range(0, 61):
            worst = max(worst, float(np.abs(vp.gamma(l) - gm(l)).max()))
    rp.check("V2", worst < 1e-12, "the note's closed covariances (2), (3), (4) against M^l G from my own solution of the "
             "stationary equation, lags 0 to 60, all two-host models: largest difference %.1e" % worst)

    worst_block = 1e9
    for nm, vp, gm in allv:
        L = 40
        d = vp.d
        big = np.zeros(((L + 1) * d, (L + 1) * d))
        gl = [vp.gamma(l) for l in range(L + 1)]
        for i in range(L + 1):
            for j in range(L + 1):
                big[i * d:(i + 1) * d, j * d:(j + 1) * d] = gl[i - j] if i >= j else gl[j - i].T
        worst_block = min(worst_block, float(np.linalg.eigvalsh(big).min()))
    rp.check("V3", worst_block > -1e-10, "covariance matrix of 41 consecutive steps of the hosts, built from the lag "
             "covariances: smallest eigenvalue over all %d models %.3e (not negative)" % (len(allv), worst_block))

    qs = (-1.0, -0.5, 0.0, 0.03708065938, 0.1870806594, 0.2241613188, 0.33708065938, 0.3741613188, 0.5241613188,
          0.55464821794, 0.70524525726, 0.9501693176, 1.0)
    worst_spec, worst_herm = 1e9, 0.0
    for nm, vp, gm in allv:
        if gm is None:
            continue
        for q in qs:
            me, herm = spectral_min_eig(gm, vp.rho(), q)
            worst_spec = min(worst_spec, me)
            worst_herm = max(worst_herm, herm)
    over, _ = spectral_min_eig(models[0][2], 0.9, 1.3)
    rp.check("V4", worst_spec > -1e-9, "spectral density matrix (the sum over ALL lags of the note's closed covariances, "
             "720 frequencies) with the cross-covariances multiplied by q, 13 values of q from -1 to 1 including every "
             "boundary of the note, all two-host models: smallest eigenvalue %.3e (not negative; a valid process at "
             "every lag). For comparison q = 1.3 on the main predator model gives %.3f: not a valid process, as it "
             "should not be" % (worst_spec, over))

    rng = np.random.default_rng(303)
    worst = 0.0
    for nm, vp, gm in models:
        vp4 = two_copies(vp)
        for _ in range(10):
            q = rng.uniform(-1, 1)
            h1, h2 = q_rows(q)
            C = np.stack([h1, h2])
            for l in range(0, 30):
                got = C @ vp4.gamma(l) @ C.T
                want = np.array(gm(l), float)
                want[0, 1] *= q
                want[1, 0] *= q
                worst = max(worst, float(np.abs(got - want).max()))
    rp.check("V5", worst < 1e-12, "the construction for q (host 1 from one copy of the autoregression; host 2 = q times "
             "its partner plus sqrt(1 - q^2) times the same host of an independent copy) has the two marginal "
             "covariances unchanged and every cross-covariance multiplied by q: largest difference %.1e" % worst)

    vp = var_predator()
    lags = [(l, float(vp.gamma(l)[1, 0])) for l in range(0, 40)]
    lbest = max(lags, key=lambda t: t[1])
    rp.check("V6", lbest[0] == 4 and quoted(lbest[1], "0.6239881803"),
             "covariance of the predator with the prey l steps earlier is largest at lag %d, value %s (note: lag 4, "
             "0.6239881803); at lag 0 it is %.1e" % (lbest[0], fq(lbest[1], "0.6239881803"), lags[0][1]))
    c = float(var_shared().G[0, 1])
    rp.check("V7", quoted(c, "0.5829201647"), "shared weather (0.25, 0.85, 0.9): correlation at lag zero %s (note 0.5829201647)"
             % fq(c, "0.5829201647"))

    txt = []
    ok = True
    for i, (nm, vp, gm) in enumerate(models):
        rng = np.random.default_rng(310 + i)
        X0 = rng.standard_normal((64, vp.d)) @ vp.LG.T
        Xp, _ = host_paths(vp, rng, X0, 20000)
        ex, z = emp_cov(Xp, gm)
        ok &= ex <= 0
        txt.append("%s: largest deviation %.1f SE" % (nm.split(",")[0].split(" (")[0], z))
    rng = np.random.Generator(np.random.Philox(320))
    gen = ComplexAR(0.9, PI / 10, 64, rng)
    parts = [gen.hosts(2000) for _ in range(10)]
    ex, z = emp_cov(np.stack([np.concatenate([p_[0] for p_ in parts]), np.concatenate([p_[1] for p_ in parts])], axis=2), models[0][2])
    ok &= ex <= 0
    txt.append("second simulator's generator (predator and prey): %.1f SE" % z)
    gen = ModeAR(0.02, 0.02, -0.8, 64, rng)
    parts = [gen.hosts(2000) for _ in range(10)]
    ex, z = emp_cov(np.stack([np.concatenate([p_[0] for p_ in parts]), np.concatenate([p_[1] for p_ in parts])], axis=2), models[4][2])
    ok &= ex <= 0
    txt.append("second simulator's generator (fast competitors): %.1f SE" % z)
    rp.check("V8", ok, "covariances measured on the host paths that the simulators actually use (64 paths of 20,000 "
             "steps, lags 0 to 25, every entry) against the closed forms, criterion 4 SE + 0.002: " + "; ".join(txt))
    return rp.done()


# ============================================================================================
# SECTION A: every closed form against the general coefficient on random cases
# ============================================================================================
def sec_A():
    rp = Rep("A. The note's closed forms against the general coefficient (tolerance 1e-8 relative, fixed beforehand)")
    rng = np.random.default_rng(202)

    # A1 the mode formula (7), two fluctuating transfers, both orders
    worst, worst_plain, worst_im = 0.0, 0.0, 0.0
    cnt = {"real": 0, "complex": 0}
    bad = 0
    for it in range(300):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n)
        d = 2 if it % 3 else 3
        kind = "complex" if it % 2 else "real"
        cnt[kind] += 1
        vp = random_var(rng, d, kind)
        tr = rng.choice(n, 2, replace=False).tolist()
        kap = rng.choice([-1.0, 1.0], 2) * rng.uniform(0.3, 2.0, 2) if it % 5 < 3 else np.ones(2)
        e = np.eye(d)
        for order in (0, 1):
            rows = [e[0] * kap[0], e[1] * kap[1]] if order == 0 else [e[1] * kap[0], e[0] * kap[1]]
            mine = K_terms(stays, tr, rows, vp).sum()
            re, im, mag = note_K_modes(stays, tr, rows, vp)
            bad += not close_alg(mine, re, mag)
            worst = max(worst, abs(mine - re) / max(abs(mine), abs(re), 1e-3 * mag))
            worst_plain = max(worst_plain, relerr(mine, re))
            worst_im = max(worst_im, abs(im) / mag)
    rp.check("A1", bad == 0, "(7), the sum over modes: 300 random parameter sets x both orders of the hosts (rings of 2 "
             "to 5 stages, stays 1 to 60 with some exactly 1, the two fluctuating transfers anywhere; autoregressions "
             "of dimension 2 and 3, %d with real and %d with complex eigenvalues; relative sizes of either sign in 180): "
             "largest relative difference %.1e (plain relative difference, without allowance for cancelling terms, "
             "%.1e); largest imaginary part of the formula's sum %.1e of its terms"
             % (cnt["real"], cnt["complex"], worst, worst_plain, worst_im))

    worst, bad = 0.0, 0
    for it in range(100):
        n = int(rng.integers(3, 6))
        stays = rand_stays(rng, n)
        vp = random_var(rng, 3, "complex" if it % 2 else "real")
        tr = rng.choice(n, 3, replace=False).tolist()
        perm = rng.permutation(3)
        kap = rng.choice([-1.0, 1.0], 3) * rng.uniform(0.3, 2.0, 3)
        rows = [np.eye(3)[perm[k]] * kap[k] for k in range(3)]
        mine = K_terms(stays, tr, rows, vp).sum()
        re, im, mag = note_K_modes(stays, tr, rows, vp)
        bad += not close_alg(mine, re, mag)
        worst = max(worst, abs(mine - re) / max(abs(mine), abs(re), 1e-3 * mag))
    rp.check("A1b", bad == 0, "(7) read for three fluctuating transfers and three linked hosts (the note writes it for "
             "two): 100 random sets, largest relative difference %.1e" % worst)

    # A2 the lag form (6) and the renewal sequence u
    worst, bad, worst_u = 0.0, 0, 0.0
    for it in range(100):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n, 1.0, 25.0)
        vp = random_var(rng, 2, "complex" if it % 2 else "real")
        tr = rng.choice(n, 2, replace=False).tolist()
        kap = rng.choice([-1.0, 1.0], 2) * rng.uniform(0.3, 2.0, 2)
        rows = [np.eye(2)[0] * kap[0], np.eye(2)[1] * kap[1]]
        T = stays.sum()
        L = 6000
        Gl = np.empty((L + 1, 2, 2))
        Gl[0] = vp.G
        for l in range(1, L + 1):
            Gl[l] = vp.M @ Gl[l - 1]
        tot, mag = 0.0, 0.0
        A0 = ring(stays)
        for a_, i in enumerate(tr):
            for b_, j in enumerate(tr):
                u = renewal_u(stays, j, i, L)
                rho = np.einsum("i,lij,j->l", rows[a_], Gl, rows[b_])
                val = rho[0] / (2 * T) + float(np.sum(rho[1:] * (1 / T - u)))
                tot += val
                mag += abs(val)
                Apow = np.eye(n)
                for l in range(1, 81):
                    worst_u = max(worst_u, abs(Apow[i, (j + 1) % n] / stays[i] - u[l - 1]))
                    Apow = A0 @ Apow
        mine = K_terms(stays, tr, rows, vp).sum()
        bad += not close_alg(mine, tot, mag)
        worst = max(worst, abs(mine - tot) / max(abs(mine), abs(tot), 1e-3 * mag))
    rp.check("A2", bad == 0 and worst_u < 1e-11, "(6), the sum over lags with the renewal sequence u from (5) (computed "
             "here by polynomial division, 6000 lags): 100 random sets, largest relative difference %.1e; and the "
             "identity of the proof, w^T B_i A0^(l-1) B_j v = u(l)/T, lags 1 to 80: largest difference %.1e"
             % (worst, worst_u))

    # A3 the decomposition K = K1 + K2 + X and the boundaries (8)
    worst_dec, worst_q, worst_b, nb = 0.0, 0.0, 0.0, 0
    for it in range(100):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n)
        vp = random_var(rng, 2, "complex" if it % 2 else "real")
        vp4 = two_copies(vp)
        tr = rng.choice(n, 2, replace=False).tolist()
        order = it % 2
        e = np.eye(2)
        rows = [e[0], e[1]] if order == 0 else [e[1], e[0]]
        t = K_terms(stays, tr, rows, vp)
        K1h = K_terms(stays, [tr[0]], [rows[0]], vp).sum()
        K2h = K_terms(stays, [tr[1]], [rows[1]], vp).sum()
        X = t[0, 1] + t[1, 0]
        mag = abs(K1h) + abs(K2h) + abs(X)
        worst_dec = max(worst_dec, abs(t.sum() - (K1h + K2h + X)) / mag, abs(t[0, 0] - K1h) / mag, abs(t[1, 1] - K2h) / mag)
        q = rng.uniform(-1, 1)
        h1, h2 = q_rows(q)
        rq = [h1, h2] if order == 0 else [h2, h1]
        worst_q = max(worst_q, abs(K_terms(stays, tr, rq, vp4).sum() - (K1h + K2h + q * X)) / mag)
        for qb, target in ((-(K1h + K2h) / X, 0.0), (-K2h / X, K1h), (-K1h / X, K2h)):
            if abs(qb) <= 1:
                nb += 1
                h1, h2 = q_rows(qb)
                rq = [h1, h2] if order == 0 else [h2, h1]
                worst_b = max(worst_b, abs(K_terms(stays, tr, rq, vp4).sum() - target) / mag)
    rp.check("A3", max(worst_dec, worst_q, worst_b) < TOL_ALG, "(8): K = K1 + K2 + X with K1 and K2 the coefficients of "
             "each host fluctuating alone (the other transfer steady), 100 random sets: largest relative difference "
             "%.1e; with the cross-covariances multiplied by a random q (through the two-copy construction, an "
             "autoregression of dimension 4) K = K1 + K2 + qX: %.1e; at the boundaries q = -(K1 + K2)/X and q = -K2/X "
             "(and -K1/X) that fall in [-1, 1] (%d of them) the general coefficient equals 0 and K1 (K2): %.1e"
             % (worst_dec, worst_q, nb, worst_b))

    # A4 the order difference (9)
    worst, bad = 0.0, 0
    for it in range(100):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n, 1.0, 25.0)
        vp = random_var(rng, 2, "complex" if it % 2 else "real")
        tr = rng.choice(n, 2, replace=False).tolist()
        e = np.eye(2)
        t0 = K_terms(stays, tr, [e[0], e[1]], vp)
        t1 = K_terms(stays, tr, [e[1], e[0]], vp)
        L = 6000
        u12 = renewal_u(stays, tr[0], tr[1], L)
        u21 = renewal_u(stays, tr[1], tr[0], L)
        Gl = vp.G.copy()
        acc = 0.0
        for l in range(1, L + 1):
            Gl = vp.M @ Gl
            acc -= (Gl[1, 0] - Gl[0, 1]) * (u12[l - 1] - u21[l - 1])
        mag = np.abs(t0).sum() + np.abs(t1).sum()
        diff = t0.sum() - t1.sum()
        bad += not close_alg(diff, acc, mag)
        worst = max(worst, abs(diff - acc) / max(abs(diff), abs(acc), 1e-3 * mag))
    rp.check("A4", bad == 0, "(9), the difference between the two orders as a sum over lags of [rho21 - rho12] x "
             "[u(1->2) - u(2->1)]: 100 random sets, largest relative difference %.1e. The lag convention that this "
             "needs: rho_ij(l) is the covariance of the host of transfer i with the host of transfer j taken l steps "
             "EARLIER" % worst)

    e = np.eye(2)
    worst_rev = 0.0
    for it in range(30):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n)
        tr = rng.choice(n, 2, replace=False).tolist()
        a_, c_, d_ = rng.uniform(-0.4, 0.6), rng.uniform(-0.3, 0.3), rng.uniform(-0.9, 0.9)
        vc = var_competitors(a_, c_, d_)
        worst_rev = max(worst_rev, abs(K_terms(stays, tr, [e[0], e[1]], vc).sum() - K_terms(stays, tr, [e[1], e[0]], vc).sum()))
    worst_eq = 0.0
    vpp = var_predator()
    for it in range(30):
        a_, b_ = rng.uniform(1, 30, 2)
        for stays, tr in (([a_, a_], [0, 1]), ([a_, b_, a_, b_], [0, 2]), ([a_, b_, b_, a_], [1, 3])):
            worst_eq = max(worst_eq, abs(K_terms(stays, tr, [e[0], e[1]], vpp).sum() - K_terms(stays, tr, [e[1], e[0]], vpp).sum()))
    worst_ind = 0.0
    for it in range(30):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n)
        tr = rng.choice(n, 2, replace=False).tolist()
        p1, p2 = rng.uniform(-0.9, 0.95, 2)
        vi = Var(np.diag([p1, p2]), np.diag([1 - p1 * p1, 1 - p2 * p2]))
        worst_ind = max(worst_ind, abs(K_terms(stays, tr, [e[0], e[1]], vi).sum() - K_terms(stays, tr, [e[1], e[0]], vi).sum()))
    d_eqmean = K_two([5, 10, 5], vpp, 0).sum() - K_two([5, 10, 5], vpp, 1).sum()
    rp.check("A4b", max(worst_rev, worst_eq, worst_ind) < 1e-11 and abs(d_eqmean) > 1e-3,
             "the note's sufficient conditions for no effect of order at second order: cross-covariances the same in "
             "both directions (30 random symmetric competitor models on random rings) %.1e; the two directed arcs "
             "made of the same stays (30 random cases x 3 layouts, predator and prey) %.1e; independent hosts with "
             "different memories (30) %.1e. Equal MEAN arc times are not enough: stays (5, 10, 5), arcs of mean 10 "
             "each, predator and prey, order 0 minus order 1 = %.6f" % (worst_rev, worst_eq, worst_ind, d_eqmean))

    # A5 one host on the two transfers of a two-stage ring: (10)
    worst_p, worst_m, npos, nneg = 0.0, 0.0, 0, 0
    for it in range(240):
        a_, b_ = np.exp(rng.uniform(0, math.log(60), 2))
        if it % 10 == 0:
            a_ = 1.0
        phi = rng.uniform(0.0, 0.98) if it % 3 else rng.uniform(-0.95, 0.0)
        npos += phi >= 0
        nneg += phi < 0
        v1 = var_ar1(phi)
        kp = K_terms([a_, b_], [0, 1], [np.array([1.0]), np.array([1.0])], v1).sum()
        km = K_terms([a_, b_], [0, 1], [np.array([1.0]), np.array([-1.0])], v1).sum()
        worst_p = max(worst_p, relerr(kp, note_Kplus(a_, b_, phi)) if abs(kp) > 1e-6 else abs(kp - note_Kplus(a_, b_, phi)))
        worst_m = max(worst_m, relerr(km, note_Kminus(a_, b_, phi)) if abs(km) > 1e-6 else abs(km - note_Kminus(a_, b_, phi)))
    rp.check("A5", max(worst_p, worst_m) < TOL_ALG, "(10), one host on both transfers of a two-stage ring, K_+ (same "
             "sign) and K_- (opposite signs): 240 random (a, b, memory), %d with memory in [0, 0.98) as the note "
             "states and %d with negative memory, which the note does not cover: largest relative difference %.1e "
             "(K_+), %.1e (K_-). The formulas also hold for negative memory" % (npos, nneg, worst_p, worst_m))

    # A6 the sign condition (11)
    ngain, nmis, ngain_neg = 0, 0, 0
    for it in range(3000):
        a_, b_ = np.exp(rng.uniform(0, math.log(80), 2))
        phi = rng.uniform(0.0, 0.995) if it % 4 else rng.uniform(-0.99, 0.0)
        kp = K_terms([a_, b_], [0, 1], [np.array([1.0]), np.array([1.0])], var_ar1(phi)).sum()
        if phi >= 0:
            pred = note_gain_condition(a_, b_, phi)
            if abs(kp) > 1e-9:
                nmis += (kp < 0) != pred
            ngain += kp < 0
        else:
            ngain_neg += kp < -1e-12
    rp.check("A6", nmis == 0 and ngain_neg == 0, "(11): the general coefficient K_+ is negative exactly when "
             "(a - b)^2 > 2(a + b) and theta > 2ab/((a - b)^2 - 2(a + b)): 2250 random cases with memory in [0, 1), "
             "%d of them gains, no disagreement (so 'only if' in the note can read 'if and only if'); 750 cases with "
             "negative memory: %d gains (none: with negative memory moving together always costs)" % (ngain, ngain_neg))

    # A7 the discrete coefficient = the white-noise cost of the time step + the continuous-time expression
    worst = 0.0
    for it in range(240):
        a_, b_ = np.exp(rng.uniform(0, math.log(60), 2))
        phi = rng.uniform(0.02, 0.98) if it % 3 else rng.uniform(-0.9, -0.02)
        k1, k2 = rng.uniform(-2, 2, 2)
        th = phi / (1 - phi)
        mine = K_terms([a_, b_], [0, 1], [np.array([k1]), np.array([k2])], var_ar1(phi))
        want = (k1 + k2) ** 2 / (2 * (a_ + b_)) + manuscript_continuous(a_ / th, b_ / th, k1, k2)
        worst = max(worst, abs(mine.sum() - want) / max(abs(want), 1e-3 * np.abs(mine).sum()))
    rp.check("A7", worst < TOL_ALG, "what is exactly true in discrete time (found here, not in the note): for relative "
             "sizes k1, k2 of either sign, K = (k1 + k2)^2/(2T) + [the manuscript's continuous-time expression "
             "evaluated at x_k = tau_k (1 - phi)/phi], 240 random cases: largest relative difference %.1e. The first "
             "term is the cost of two transfers moving together within one time step and has no continuous "
             "counterpart" % worst)

    # A11 consistency: two hosts of equal memory and correlation c at lag zero; c = +1 or -1 is one host on two transfers
    worst_pm, worst_c, worst_ring = 0.0, 0.0, 0.0
    for it in range(120):
        a_, b_ = np.exp(rng.uniform(0, math.log(60), 2))
        phi = rng.uniform(-0.9, 0.98)
        e2 = np.eye(2)
        for c_, want in ((1.0, note_Kplus(a_, b_, phi)), (-1.0, note_Kminus(a_, b_, phi))):
            v2 = Var(phi * np.eye(2), (1 - phi * phi) * np.array([[1.0, c_], [c_, 1.0]]))
            two = note_K_modes([a_, b_], [0, 1], [e2[0], e2[1]], v2)[0]
            gen2 = K_terms([a_, b_], [0, 1], [e2[0], e2[1]], v2).sum()
            worst_pm = max(worst_pm, abs(two - want) / max(abs(want), 1e-3), abs(gen2 - want) / max(abs(want), 1e-3))
        c_ = rng.uniform(-1, 1)
        v2 = Var(phi * np.eye(2), (1 - phi * phi) * np.array([[1.0, c_], [c_, 1.0]]))
        mix = 0.5 * (1 + c_) * note_Kplus(a_, b_, phi) + 0.5 * (1 - c_) * note_Kminus(a_, b_, phi)
        worst_c = max(worst_c, abs(note_K_modes([a_, b_], [0, 1], [e2[0], e2[1]], v2)[0] - mix) / max(abs(mix), 1e-3))
        n = int(rng.integers(3, 6))
        stays = rand_stays(rng, n)
        tr = rng.choice(n, 2, replace=False).tolist()
        sgn = float(rng.choice([-1.0, 1.0]))
        v2 = Var(phi * np.eye(2), (1 - phi * phi) * np.array([[1.0, sgn], [sgn, 1.0]]))
        one_host = K_terms(stays, tr, [np.array([1.0]), np.array([sgn])], var_ar1(phi))
        two_host = note_K_modes(stays, tr, [e2[0], e2[1]], v2)
        worst_ring = max(worst_ring, abs(one_host.sum() - two_host[0]) / max(abs(two_host[0]), 1e-3 * two_host[2]))
    rp.check("A11", max(worst_pm, worst_c, worst_ring) < TOL_ALG, "consistency with one host on two transfers: the "
             "two-host formula (7) with equal memories and correlation +1 (-1) at lag zero equals K_+ (K_-) of (10) "
             "and the general coefficient of ONE host moving both transfers together (oppositely): 120 random two-stage "
             "rings, largest relative difference %.1e; on rings of 3 to 5 stages against the general coefficient of "
             "one host on two transfers: %.1e; and with correlation c between -1 and 1, K = (1 + c)/2 K_+ + (1 - c)/2 "
             "K_-: %.1e" % (worst_pm, worst_ring, worst_c))

    # A8 the derivative (13)
    worst, bad = 0.0, 0
    where = {"on one arc": 0, "other": 0}
    for it in range(120):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n, 1.0, 40.0)
        vp = random_var(rng, 2, "complex" if it % 2 else "real")
        tr = rng.choice(n, 2, replace=False).tolist()
        kap = rng.choice([-1.0, 1.0], 2) * rng.uniform(0.3, 2.0, 2)
        order = it % 2
        e = np.eye(2)
        rows = [e[0] * kap[0], e[1] * kap[1]] if order == 0 else [e[1] * kap[0], e[0] * kap[1]]
        ch = int(rng.integers(0, n))
        if stays[ch] < 1.2:
            stays[ch] = rng.uniform(1.5, 40)
        mp.mp.dps = 40
        b = mp.mpf(float(stays[ch]))
        h = b * mp.mpf(10) ** (-14)

        def at(x):
            st = [float(s) for s in stays]
            st[ch] = x
            return mp_sum(coef_ring_mp(st, tr, [list(r) for r in rows], vp.M.tolist(), vp.G.tolist()))
        deriv = float((at(b + h) - at(b - h)) / (2 * h))
        re, im, mag = note_K_modes(stays, tr, rows, vp, changed=ch)
        bad += not close_alg(deriv, re, mag)
        worst = max(worst, abs(deriv - re) / max(abs(deriv), abs(re), 1e-3 * mag))
    rp.check("A8", bad == 0, "(13), the derivative of K with respect to one stay, against a central difference of the "
             "general coefficient in 40-digit arithmetic: 120 random sets (the changed stage anywhere in the ring, "
             "both orders, sizes of either sign): largest relative difference %.1e" % worst)

    # A9 the long-stay limit (14), (15)
    mp.mp.dps = 40
    worst_lim, worst_J, worst_ind = 0.0, 0.0, 0.0
    b0 = mp.mpf(10) ** 9
    for it in range(100):
        r = float(rng.uniform(0.2, 0.97))
        om = float(rng.uniform(0.05, PI - 0.05))
        a_ = float(np.exp(rng.uniform(0, math.log(40))))
        c_ = float(np.exp(rng.uniform(0, math.log(20))))
        M2, G2 = mp_rot(r, om)
        lim_n, J_n, re_n = note_long_stay(r, om, a_, c_)
        im_n = re_n - J_n
        for order, lim_w, J_w in ((0, lim_n, J_n), (1, -lim_n, re_n + im_n)):
            Ks = [Kmp_two([b0 * f, a_, c_], M2, G2, order) for f in (1, 2, 4)]
            lim = float((8 * Ks[2] - 6 * Ks[1] + Ks[0]) / 3)
            J = float(2 * b0 * (5 * Ks[1] - 4 * Ks[2] - Ks[0]))
            worst_lim = max(worst_lim, relerr(lim, lim_w))
            worst_J = max(worst_J, relerr(J, J_w))
        if it < 30:
            Z = mp.zeros(2, 2)
            M4 = mp.zeros(4, 4)
            G4 = mp.eye(4)
            for i in range(2):
                for j in range(2):
                    M4[i, j] = M2[i, j]
                    M4[i + 2, j + 2] = M2[i, j]
            Ks = [mp_sum(coef_ring_mp([b0 * f, a_, c_], [0, 1], [[1, 0, 0, 0], [0, 0, 0, 1]], M4, G4)) for f in (1, 2, 4)]
            lim = float((8 * Ks[2] - 6 * Ks[1] + Ks[0]) / 3)
            J = float(2 * b0 * (5 * Ks[1] - 4 * Ks[2] - Ks[0]))
            worst_ind = max(worst_ind, abs(lim), relerr(J, re_n))
    rp.check("A9", max(worst_lim, worst_J, worst_ind) < TOL_LIMIT, "(14), (15), ring (b, a, c) with rotation hosts as "
             "b grows: K(b) = K_inf + J/b + ..., with K_inf and J fitted to the general coefficient at b = 1e9, 2e9, "
             "4e9 in 40-digit arithmetic. 100 random (radius, angle, a, c), both orders: largest relative difference "
             "of the limit -Im A (order 1: +Im A) %.1e; of J (order 1: the imaginary part reversed) %.1e. Independent "
             "hosts with the same two covariance functions (30 cases): limit 0 and 1/b coefficient equal to the real "
             "part of (15), largest difference %.1e. (Tolerance for extrapolated limits 1e-6.)"
             % (worst_lim, worst_J, worst_ind))

    # A10 the two parking interventions coincide at second order
    worst = 0.0
    for it in range(60):
        p_ = float(np.exp(rng.uniform(0, math.log(300))))
        vp = [var_predator(), var_shared(), random_var(rng, 2, "complex")][it % 3]
        for order in (0, 1):
            worst = max(worst, relerr(K_two([5, 5, p_], vp, order).sum(), K_two([p_, 5, 5], vp, order).sum()))
    rp.check("A10", worst < 1e-9, "stays (5, 5, p) and (p, 5, 5) have the same second-order coefficient (60 random p, "
             "three host models, both orders): largest relative difference %.1e. At finite noise they need not agree; "
             "the note says so" % worst)
    return rp.done()


# ============================================================================================
# SECTION Q: every quoted number and boundary that needs no random number
# ============================================================================================
NOTE_GRID = {
    ("predator", 1, 0): (0.233952, 0.021840, 0.000131, 0.303612, 0.001226),
    ("predator", 1, 1): (0.591590, 0.053453, 0.000074, 0.611780, 0.000748),
    ("predator", 5, 0): (0.144426, 0.013795, 0.000162, 0.219106, 0.001516),
    ("predator", 5, 1): (0.465254, 0.041830, 0.000086, 0.463261, 0.000964),
    ("predator", 50, 0): (0.208123, 0.018583, 0.000139, 0.200752, 0.001606),
    ("predator", 50, 1): (-0.094786, -0.008295, 0.000196, -0.059370, 0.002068),
    ("shared", 1, 0): (0.036594, 0.004062, 0.000102, 0.116797, 0.000973),
    ("shared", 1, 1): (0.328422, 0.029879, 0.000069, 0.365292, 0.000699),
    ("shared", 5, 0): (0.195500, 0.017890, 0.000081, 0.230625, 0.000822),
    ("shared", 5, 1): (0.332100, 0.030045, 0.000061, 0.352493, 0.000644),
    ("shared", 50, 0): (0.091697, 0.008389, 0.000106, 0.110961, 0.001106),
    ("shared", 50, 1): (-0.018174, -0.001440, 0.000125, 0.011708, 0.001307),
    ("competitors", 1, 0): (0.768273, 0.069183, 0.000020, 0.773731, 0.000214),
    ("competitors", 1, 1): (0.768273, 0.069163, 0.000026, 0.773516, 0.000273),
    ("competitors", 5, 0): (0.621334, 0.055960, 0.000033, 0.624128, 0.000358),
    ("competitors", 5, 1): (0.621334, 0.055941, 0.000031, 0.623943, 0.000344),
    ("competitors", 50, 0): (0.398834, 0.035993, 0.000075, 0.403398, 0.000859),
    ("competitors", 50, 1): (0.398834, 0.035996, 0.000096, 0.403564, 0.001057),
}


def grid_var(model):
    return {"predator": var_predator, "shared": var_shared, "competitors": var_competitors}[model]()


def sign_changes(f, grid):
    vals = [f(x) for x in grid]
    return [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0], vals


def sec_Q():
    rp = Rep("Q. Every quoted number and boundary that needs no random number; one example on each side of each boundary")
    mp.mp.dps = 40
    vp = var_predator()
    vp4 = two_copies(vp)
    Mp, Gp = mp_rot(mp.mpf("0.9"), mp.pi / 10)
    Mf, Gf = mp_rot(mp.mpf("0.9"), mp.pi / 2)

    def chk(key, mine, text, what):
        rp.check(key, quoted(float(mine), text), "%s: mine %s, note %s" % (what, fq(float(mine), text), text))

    # ---- (5,50,5), order 1: K1, K2, X and the two boundaries in q
    t = K_two([5, 50, 5], vp, 1)
    K1, K2, X = t[0, 0], t[1, 1], t[0, 1] + t[1, 0]
    chk("Q1a", K1, "0.02833423237", "predator and prey, stays (5,50,5), order 1: K1 (the host on transfer 0 alone)")
    chk("Q1b", K2, "0.02833423237", "the same: K2 (the host on transfer 1 alone)")
    chk("Q1c", X, "-0.15145463174", "the same: X (the two cross terms)")
    q_add, q_tot = -K2 / X, -(K1 + K2) / X
    chk("Q2a", q_add, "0.1870806594", "boundary in q above which the second host lowers the loss of one host alone, -K2/X")
    chk("Q2b", q_tot, "0.3741613188", "boundary in q above which the total loss is negative, -(K1+K2)/X")
    Kq = lambda q: K_two_q([5, 50, 5], vp4, q, 1).sum()
    # the note's two points are the boundary minus and plus 0.15; its printed q (10 digits) is that, rounded
    chk("Q2c", Kq(q_tot - 0.15), "0.02271819476", "K at q = boundary - 0.15 = %.13f (the note prints q = 0.2241613188; a loss)" % (q_tot - 0.15))
    chk("Q2d", Kq(q_tot + 0.15), "-0.02271819476", "K at q = boundary + 0.15 = %.13f (the note prints q = 0.5241613188; a gain)" % (q_tot + 0.15))
    rp.p("        (at the q exactly as printed, K = %.13f and %.13f: the same to the ten digits that the printed q carries)"
         % (Kq(0.2241613188), Kq(0.5241613188)))
    lo, hi = Kq(0.03708065938) - K1, Kq(0.33708065938) - K1
    rp.check("Q2e", lo > 0 and hi < 0 and abs(Kq(q_tot)) < 1e-12 and abs(Kq(q_add) - K1) < 1e-12,
             "sides of the added-host boundary by the general coefficient: K(q) - K1 = %+.8f at q = 0.03708065938 and "
             "%+.8f at q = 0.33708065938; at the two boundaries themselves K = %.1e and K - K1 = %.1e"
             % (lo, hi, Kq(q_tot), Kq(q_add) - K1))
    t = K_two([5, 5, 5], vp, 0)
    qa, qt = -t[1, 1] / (t[0, 1] + t[1, 0]), -(t[0, 0] + t[1, 1]) / (t[0, 1] + t[1, 0])
    chk("Q3a", qa, "0.9501693176", "stays (5,5,5), order 0: added-host boundary -K2/X")
    K5 = lambda q: K_two_q([5, 5, 5], vp4, q, 0)
    rp.check("Q3b", abs(qt) > 1 and K5(0.9).sum() - K5(0.9)[0, 0] > 0 and K5(1.0).sum() - K5(1.0)[0, 0] < 0,
             "the same ring: total-gain boundary -(K1+K2)/X = %.6f, outside [-1, 1] as the note says (K at q = 1 is "
             "%+.6f, a loss); sides of the added-host boundary: K - K1 = %+.6f at q = 0.90, %+.6f at q = 1"
             % (qt, K5(1.0).sum(), K5(0.9).sum() - K5(0.9)[0, 0], K5(1.0).sum() - K5(1.0)[0, 0]))

    # ---- one host on two transfers, stays (1, 10)
    gen = lambda a, b, phi, k2=1.0: K_terms([a, b], [0, 1], [np.array([1.0]), np.array([k2])], var_ar1(phi)).sum()
    phi_star = brentq(lambda ph: gen(1.0, 10.0, ph), 0.05, 0.6, xtol=1e-15)
    chk("Q4a", phi_star, "0.25316455696", "stays (1,10), one host moving both transfers together: memory at which the "
        "general coefficient changes sign (a root of my coefficient, not of the note's formula); 20/79 = %.14f" % (20 / 79))
    chk("Q4b", gen(1.0, 10.0, 0.2), "0.03743315508", "the same, memory 0.2 (below: a loss)")
    chk("Q4c", gen(1.0, 10.0, 0.8), "-0.36363636364", "the same, memory 0.8 (above: a gain)")
    rp.check("Q4d", abs(gen(4.0, 4.0, 0.3) - 0.25) < 1e-12 and abs(gen(4.0, 4.0, 0.9) - 0.25) < 1e-12 and abs(gen(1.0, 10.0, 0.0, -1.0)) < 1e-14
             and gen(1.0, 10.0, 0.4, -1.0) > 0,
             "equal stays a = b = 4: K_+ = %.12f at memory 0.3 and %.12f at 0.9 (the note: 1/a, whatever the memory); "
             "opposite signs, stays (1,10): K_- = %.1e at memory 0 and %.6f at memory 0.4 (positive)"
             % (gen(4.0, 4.0, 0.3), gen(4.0, 4.0, 0.9), gen(1.0, 10.0, 0.0, -1.0), gen(1.0, 10.0, 0.4, -1.0)))

    # ---- parking: the derivative and its zero; the long-stay limit
    def dK(stays, idx, M, G, order):
        b = mp.mpf(stays[idx])
        h = b * mp.mpf(10) ** (-14)
        st1, st2 = list(stays), list(stays)
        st1[idx], st2[idx] = b + h, b - h
        return (Kmp_two(st1, M, G, order) - Kmp_two(st2, M, G, order)) / (2 * h)

    grid = [1.0 + 0.5 * i for i in range(399)]                       # 1 to 200
    fl = lambda b, order: (K_two([b * (1 + 1e-6), 5, 5], vp, order).sum() - K_two([b * (1 - 1e-6), 5, 5], vp, order).sum()) / (2e-6 * b)
    sc1, v1 = sign_changes(lambda b: fl(b, 1), grid)
    sc0, v0 = sign_changes(lambda b: fl(b, 0), grid)
    roots1 = [float(mp_bisect(lambda b: dK([b, 5, 5], 0, Mp, Gp, 1), lo_, hi_)) for lo_, hi_ in sc1]
    rp.check("Q5a", len(roots1) == 1 and quoted(roots1[0], "5.83441888"),
             "predator and prey, stays (b,5,5), order 1: zeros of dK/db for b from 1 to 200: %s (note: one, at "
             "5.83441888); dK/db = %+.6f at b = 4 (a longer stay hurts) and %+.6f at b = 8 (it helps)"
             % (", ".join("%.10f" % r for r in roots1), float(dK([4, 5, 5], 0, Mp, Gp, 1)), float(dK([8, 5, 5], 0, Mp, Gp, 1))))
    rp.check("Q5b", len(sc0) == 0, "the same in order 0: no zero of dK/db for b from 1 to 200 (398 intervals of 0.5); "
             "the derivative is negative throughout, between %.3e and %.3e" % (min(v0), max(v0)))
    b0 = mp.mpf(10) ** 9
    Ks = [Kmp_two([5, 5, b0 * f], Mp, Gp, 0) for f in (1, 2, 4)]
    lim = (8 * Ks[2] - 6 * Ks[1] + Ks[0]) / 3
    chk("Q6a", lim, "-0.37361348649", "predator parking, stays (5,5,p), order 0, limit of K as p grows (from the general "
        "coefficient at p = 1e9, 2e9, 4e9)")
    chk("Q6b", note_long_stay(0.9, PI / 10, 5.0)[0], "-0.37361348649", "the same from the note's -Im f_5(0.9 e^(i pi/10))")
    M4 = mp.zeros(4, 4)
    for i in range(2):
        for j in range(2):
            M4[i, j] = Mp[i, j]
            M4[i + 2, j + 2] = Mp[i, j]
    Ki = [mp_sum(coef_ring_mp([5, 5, b0 * f], [0, 1], [[1, 0, 0, 0], [0, 0, 0, 1]], M4, mp.eye(4))) for f in (1, 2, 4)]
    rp.check("Q6c", abs(float((8 * Ki[2] - 6 * Ki[1] + Ki[0]) / 3)) < 1e-12,
             "its independent-host counterpart (same two covariance functions) tends to %.1e: zero, as the note says; "
             "at p = 200 it is %+.6f against %+.6f linked"
             % (float((8 * Ki[2] - 6 * Ki[1] + Ki[0]) / 3), K_two_q([5, 5, 200], vp4, 0.0, 0).sum(), K_two([5, 5, 200], vp, 0).sum()))

    # ---- the rotation family: J, the angle at which it vanishes, the fast rotation
    def fit(stf, M, G, order):
        Ks_ = [Kmp_two(stf(b0 * f), M, G, order) for f in (1, 2, 4)]
        return float((8 * Ks_[2] - 6 * Ks_[1] + Ks_[0]) / 3), float(2 * b0 * (5 * Ks_[1] - 4 * Ks_[2] - Ks_[0]))

    lim1, J1 = fit(lambda b: [b, 5, 5], Mp, Gp, 0)
    lim2, J2 = fit(lambda b: [b, 5, 5], Mf, Gf, 0)
    chk("Q7a", J1, "4.39839750358", "ring (b,5,5), order 0, radius 0.9, angle pi/10: coefficient J of 1/b, fitted to the "
        "general coefficient (positive: a long steady stay eventually helps)")
    chk("Q7b", J2, "-0.01776762426", "the same at angle pi/2 (negative: it eventually hurts)")
    chk("Q7c", note_long_stay(0.9, PI / 2, 5.0)[2], "0.07879016346", "angle pi/2: the 1/b coefficient for independent hosts "
        "with the same covariance functions (the real part of (15)); positive, so linking the hosts reverses the sign")
    chk("Q7d", lim2, "-0.11854583772", "angle pi/2: the limit of K as b grows")
    for key, b_, text in (("Q7e", 20, "-0.12131470722"), ("Q7f", 50, "-0.11927216124"), ("Q7g", 200, "-0.11866098865")):
        chk(key, Kmp_two([b_, 5, 5], Mf, Gf, 0), text, "angle pi/2: K at b = %d" % b_)
    Jn = lambda om: note_long_stay(0.9, om, 5.0)[1]
    og = [0.01 + i * (PI - 0.02) / 2000 for i in range(2001)]
    scJ, _ = sign_changes(Jn, og)
    rootsJ = [brentq(Jn, lo_, hi_, xtol=1e-15) for lo_, hi_ in scJ]
    om_star = rootsJ[0] if rootsJ else float("nan")
    MM, GG = mp_rot(mp.mpf("0.9"), mp.mpf(om_star))
    _, J_at = fit(lambda b: [b, 5, 5], MM, GG, 0)
    MM, GG = mp_rot(mp.mpf("0.9"), mp.mpf(om_star) - mp.mpf("0.1"))
    _, J_lo = fit(lambda b: [b, 5, 5], MM, GG, 0)
    dlo = float(Kmp_two([400, 5, 5], MM, GG, 0) - Kmp_two([200, 5, 5], MM, GG, 0))
    MM, GG = mp_rot(mp.mpf("0.9"), mp.mpf(om_star) + mp.mpf("0.1"))
    _, J_hi = fit(lambda b: [b, 5, 5], MM, GG, 0)
    dhi = float(Kmp_two([400, 5, 5], MM, GG, 0) - Kmp_two([200, 5, 5], MM, GG, 0))
    rp.check("Q8", len(rootsJ) >= 1 and quoted(om_star, "1.13321275604") and abs(J_at) < 1e-8 and J_lo > 0 and J_hi < 0
             and dlo < 0 and dhi > 0,
             "angle at which J = 0 (radius 0.9, a = 5): zeros of the note's J between 0 and pi: %s (note: first "
             "boundary 1.13321275604). J fitted to the general coefficient: %.1e at that angle, %+.6f at 0.1 below, "
             "%+.6f at 0.1 above. In terms of K itself: K(400) - K(200) = %+.3e at 0.1 below (lengthening helps) and "
             "%+.3e at 0.1 above (it hurts)" % (", ".join("%.11f" % r for r in rootsJ), J_at, J_lo, J_hi, dlo, dhi))

    harm = []
    for r_ in (0.3, 0.6, 0.9, 0.98):
        for w_ in (0.1, 0.3, 0.5, 0.7, 0.9):
            for a_ in (1.0, 5.0, 50.0):
                lim_, J0, re_ = note_long_stay(r_, w_ * PI, a_, 5.0)
                for order, J_ in ((0, J0), (1, 2 * re_ - J0)):
                    if J_ < 0:
                        harm.append("(radius %g, angle %g pi, a = %g, order %d: J = %+.4f)" % (r_, w_, a_, order, J_))
    rp.p("        the note's wider family (radius 0.3, 0.6, 0.9, 0.98; angle 0.1 to 0.9 pi; a = 1, 5, 50; both orders; 120 "
         "members), by formula (15) as verified in A9: J < 0, a long steady stay eventually hurts at second order, in %d: %s"
         % (len(harm), " ".join(harm)))

    # ---- the main grid of second-order coefficients
    worst, bad = 0.0, 0
    lines = []
    for (model, a, order), row in NOTE_GRID.items():
        mine = K_two([5, a, 5], grid_var(model), order).sum()
        bad += abs(mine - row[0]) > 0.6e-6
        worst = max(worst, abs(mine - row[0]))
        lines.append("%s a=%d order %d: %.6f" % (model, a, order, mine))
    rp.check("Q9", bad == 0, "the 18 coefficients K of the main grid, stays (5,a,5): all agree with the note to its six "
             "decimals (largest difference %.1e). Mine: %s" % (worst, "; ".join(lines)))

    vcp = var_competitors()
    xs = []
    for a_ in (1.0, 5.0, 50.0):
        t_ = K_two([5, a_, 5], vcp, 0)
        xs.append((a_, t_[0, 0] + t_[1, 1], t_[0, 1] + t_[1, 0]))
    rp.check("Q9b", all(x[2] > 0 for x in xs) and vcp.G[0, 1] < 0,
             "competitors (0.55, 0.35, 0): correlation at lag zero %+.4f, yet the cross terms X are positive at every a "
             "of the grid, so linking them ADDS to the cost of the two alone, as the note says: %s"
             % (vcp.G[0, 1], "; ".join("a = %g: K1 + K2 = %+.6f, X = %+.6f" % x for x in xs)))

    # ---- exact periodic example
    tt = np.arange(20)
    cs = math.sqrt(2) * np.cos(2 * PI * tt / 20)
    sn = math.sqrt(2) * np.sin(2 * PI * tt / 20)
    A0 = ring([5, 50, 5])
    D0 = floquet_root(A0, ring_masks(3, [0, 1]), periodic_hosts(1.0, [cs, sn]), ring_masks(3, [0])[0])
    D1 = floquet_root(A0, ring_masks(3, [0, 1]), periodic_hosts(1.0, [sn, cs]), ring_masks(3, [0])[0])
    chk("Q10a", D0, "0.15544338014", "exact loss with hosts sqrt(2)cos(2 pi t/20) and sqrt(2)sin(2 pi t/20) at amplitude 1, "
        "stays (5,50,5), host 1 = the cosine on transfer 0 and host 2 = the sine on transfer 1 (order 0)")
    chk("Q10b", D1, "-0.11865219418", "the same with the two hosts exchanged (order 1)")
    hs = periodic_hosts(1.0, [cs, sn])
    ell = sum(-np.log(h).mean() for h in hs)
    rp.p("        (their log penalties sum to %.6f, so the bound of the note holds for both: %.6f and %.6f)" % (ell, D0, D1))

    # ---- fixed generation time 15, fast competitors
    vf = var_competitors(0.02, 0.02, -0.8)
    vf4 = two_copies(vf)
    K_old = K_terms([10, 5], [0], [np.array([1.0, 0])], vf).sum()
    tn = K_two([5, 5, 5], vf, 0)
    K1n, K2n, Xn = tn[0, 0], tn[1, 1], tn[0, 1] + tn[1, 0]
    chk("Q11a", K_old, "0.03580286426", "fast competitors: one host, two stages (10,5)")
    chk("Q11b", tn.sum(), "0.02081380613", "fast competitors: two linked hosts, three stages (5,5,5)")
    q17 = (K_old - K1n - K2n) / Xn
    chk("Q11c", q17, "0.70524525726", "(17): q at which the two rings of generation time 15 have the same coefficient")
    chk("Q11d", -K2n / Xn, "0.70464821794", "same-ring added-host boundary on (5,5,5)")
    Kn = lambda q: K_two_q([5, 5, 5], vf4, q, 0).sum()
    rp.check("Q11e", Kn(0.55464821794) - K_old > 0 and Kn(1.0) - K_old < 0 and abs(Kn(q17) - K_old) < 1e-13,
             "sides of (17) by the general coefficient: new minus old = %+.8f at q = 0.55464821794 (the split raises "
             "the loss) and %+.8f at q = 1 (it lowers it); %.1e at the boundary"
             % (Kn(0.55464821794) - K_old, Kn(1.0) - K_old, Kn(q17) - K_old))
    vc = var_competitors(0.15, 0.05, -0.8)
    tc = K_two([5, 5, 5], vc, 0)
    chk("Q12a", tc[0, 0], "0.04900091644", "competitors (0.15, 0.05, -0.8) on (5,5,5): one fluctuating host")
    chk("Q12b", tc.sum(), "0.05853238492", "the same: both hosts")

    # ---- lag cancellation
    def Ka(a, ret, order):
        return Kmp_two([5, mp.mpf(a), ret], Mp, Gp, order)

    def dKa(a, ret, order):
        a = mp.mpf(a)
        h = a * mp.mpf(10) ** (-14)
        return (Ka(a + h, ret, order) - Ka(a - h, ret, order)) / (2 * h)

    agrid = [1.0 + 0.25 * i for i in range(797)]                     # 1 to 200
    out = {}
    for ret in (5, 45):
        for order in (0, 1):
            f = lambda a: K_two([5, a, ret], vp, order).sum()
            sc, vals = sign_changes(f, agrid)
            zeros = [float(mp_bisect(lambda a: Ka(a, ret, order), lo_, hi_)) for lo_, hi_ in sc]
            dv = [(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
            turn = [(agrid[i], agrid[i + 2]) for i in range(len(dv) - 1) if dv[i] * dv[i + 1] < 0]
            tz = [float(mp_bisect(lambda a: dKa(a, ret, order), lo_, hi_)) for lo_, hi_ in turn]
            out[(ret, order)] = (zeros, tz, min(vals), max(vals))
    z5, t5, mn5, mx5 = out[(5, 0)]
    rp.check("Q13a", len(z5) == 0 and len(t5) >= 1 and quoted(float(Ka(t5[0], 5, 0)), "0.14064244134"),
             "predator and prey, stays (5,a,5), ORDER 0, a from 1 to 200: zeros of K: %d (note: none); turning points "
             "of K: %s, where K = %s (note: a minimum with K = 0.14064244134; the second turning point is a maximum "
             "that the note does not mention)"
             % (len(z5), ", ".join("%.10f" % x for x in t5), ", ".join("%.13f" % float(Ka(x, 5, 0)) for x in t5)))
    d_note, d_mine = float(dKa("3.83665271", 5, 0)), float(dKa(t5[0], 5, 0))
    d13 = note_K_modes([5, 3.83665271, 5], [0, 1], [np.array([1.0, 0]), np.array([0, 1.0])], vp, changed=1)[0]
    rp.differ("Q13a-location", quoted(t5[0], "3.83665271"),
             "location of that minimum: mine %.10f (a zero of dK/da in 40-digit arithmetic), note 3.83665271: they "
             "differ by %.1e, in the seventh decimal. At the note's a the derivative is %+.3e by my coefficient and "
             "%+.3e by the note's own formula (13); at mine it is %.1e; K(mine) - K(note's a) = %.2e"
             % (t5[0], abs(t5[0] - 3.83665271), d_note, d13, d_mine, float(Ka(t5[0], 5, 0) - Ka("3.83665271", 5, 0))))
    z5b, t5b, _, _ = out[(5, 1)]
    rp.p("        the same ring in ORDER 1 does have a zero: K = 0 at a = %s (turning points: %s). The note's 'no "
         "coefficient zero' is true of order 0 only and does not say so"
         % (", ".join("%.8f" % x for x in z5b), ", ".join("%.6f" % x for x in t5b) if t5b else "none"))
    z45, t45, _, _ = out[(45, 0)]
    rp.check("Q13b", len(t45) >= 1 and quoted(float(Ka(t45[0], 45, 0)), "-0.32212318738"),
             "stays (5,a,45), order 0: turning points of K for a from 1 to 200: %s, K there %s (note: minimum "
             "K = -0.32212318738)"
             % (", ".join("%.10f" % x for x in t45), ", ".join("%.13f" % float(Ka(x, 45, 0)) for x in t45)))
    d_note, d_mine = float(dKa("3.00799334", 45, 0)), float(dKa(t45[0], 45, 0))
    rp.differ("Q13b-location", quoted(t45[0], "3.00799334"),
             "location of that minimum: mine %.10f, note 3.00799334: they differ by %.1e, in the eighth decimal. "
             "dK/da at the note's a %+.3e, at mine %.1e" % (t45[0], abs(t45[0] - 3.00799334), d_note, d_mine))
    rp.check("Q13c", len(z45) == 1 and quoted(z45[0], "67.95957981591"),
             "stays (5,a,45), order 0: zeros of K for a from 1 to 200: %s (note: 67.95957981591); K(1) = %+.6f"
             % (", ".join("%.13f" % x for x in z45), K_two([5, 1, 45], vp, 0).sum()))
    z45b, t45b, mn, mx = out[(45, 1)]
    beyond = float(mp_bisect(lambda a: Ka(a, 45, 1), 200, 400))
    rp.p("        the same ring in order 1: zeros of K for a from 1 to 200: %s; K falls from %+.4f to %+.6f at a = 200 "
         "and is zero at a = %.6f, just outside the range the note scanned. The note does not say which order its "
         "(5,a,45) numbers are for; they are order 0"
         % (", ".join("%.8f" % x for x in z45b) if z45b else "none", mx, mn, beyond))
    for key, a_, text in (("Q13d", 20, "-0.07464961695"), ("Q13e", 40, "-0.02110091130"), ("Q13f", 150, "0.01434040854")):
        chk(key, Ka(a_, 45, 0), text, "stays (5,%d,45), order 0: K" % a_)
    return rp.done()


# ============================================================================================
# simulation sections
# ============================================================================================
def burn_for(stays_list, floor=4000, cap=40000):
    """Burn-in: at least 40 relaxation times of the mean matrix (from its second eigenvalue)."""
    worst = 0.0
    for st in stays_list:
        if len(st) < 2:
            continue
        ev = np.sort(np.abs(np.linalg.eigvals(ring(st))))[::-1]
        worst = max(worst, float(ev[1]))
    need = 40.0 / max(1e-9, 1 - worst)
    return (int(min(cap, max(floor, need))) + 3) // 4 * 4


def pm(x, se, nd=6):
    return ("%+." + str(nd) + "f (SE %." + str(nd) + "f)") % (x, se)


def sec_S():
    rp = Rep("S. The closed forms against products of the actual random matrices at small noise (criterion 4 SE + 5%)")
    sig = 0.05
    s2 = sig * sig
    e = np.eye(2)
    rng = np.random.default_rng(404)

    # S1 the mode formula (7) on random host models, both orders
    done = {"real": 0, "complex": 0}
    lines = []
    ok = True
    it = 0
    while min(done.values()) < 2:
        kind = "complex" if it % 2 else "real"
        it += 1
        if done[kind] >= 2:
            continue
        stays = rand_stays(rng, 3, 1.0, 15.0, p_one=0.0)
        vp = random_var(rng, 2, kind)
        rows = lambda o: [e[0], e[1]] if o == 0 else [e[1], e[0]]
        Ks = [note_K_modes(stays, [0, 1], rows(o), vp)[0] for o in (0, 1)]
        if min(abs(k) for k in Ks) < 0.05:
            continue
        done[kind] += 1
        cases = [case(stays, rows(o) + [None], s2 * Ks[o]) for o in (0, 1)]
        R = solve_cases(vp, cases, sig, seed=4100 + it, burn=burn_for([stays]))
        ev = np.linalg.eigvals(vp.M)
        for o in (0, 1):
            ok &= within_small(R.D[o], R.se[o], s2 * Ks[o])
            lines.append("stays (%.2f, %.2f, %.2f), eigenvalues of M %s, order %d: formula (7) K = %+.5f, simulated "
                         "D/sigma^2 = %+.5f (SE %.5f)" % (stays[0], stays[1], stays[2],
                                                         np.array2string(ev, precision=3), o, Ks[o], R.D[o] / s2, R.se[o] / s2))
    rp.check("S1", ok, "(7) on four random host models (two with real, two with complex eigenvalues), both orders, "
             "sigma = 0.05:\n          " + "\n          ".join(lines))

    # S2 decomposition, order difference, q
    vp = var_predator()
    vp4 = two_copies(vp)
    st = [5.0, 50.0, 5.0]
    t = K_two(st, vp, 1)
    K1, K2, X = t[0, 0], t[1, 1], t[0, 1] + t[1, 0]
    t0 = K_two(st, vp, 0)
    h1, hq = q_rows(0.5)
    one = np.array([1.0, 0, 0, 0])
    two = np.array([0, 1.0, 0, 0])
    cases = [case(st, [two, one, None], s2 * t.sum()),              # order 1, linked
             case(st, [two, None, None], s2 * K1),                  # host 2 alone on transfer 0
             case(st, [None, one, None], s2 * K2),                  # host 1 alone on transfer 1
             case(st, [one, two, None], s2 * t0.sum()),             # order 0, linked
             case(st, [hq, h1, None], s2 * (K1 + K2 + 0.5 * X))]    # order 1, cross-covariances halved
    R = solve_cases(vp4, cases, sig, seed=4201, npairs=256)
    xs, xse = R.combo([1, -1, -1, 0, 0])
    od, odse = R.combo([-1, 0, 0, 1, 0])
    rp.check("S2", within_small(xs, xse, s2 * X) and within_small(R.D[1], R.se[1], s2 * K1) and within_small(R.D[2], R.se[2], s2 * K2)
             and within_small(R.D[0], R.se[0], s2 * t.sum()),
             "(8) on predator and prey, stays (5,50,5), order 1, sigma = 0.05, all on the same host paths: D/sigma^2 of "
             "the linked pair %+.5f (SE %.5f) against K = %+.5f; of each host alone %+.5f (SE %.5f) and %+.5f (SE %.5f) "
             "against K1 = K2 = %+.5f; linked minus the two alone %+.5f (SE %.5f) against X = %+.5f"
             % (R.D[0] / s2, R.se[0] / s2, t.sum(), R.D[1] / s2, R.se[1] / s2, R.D[2] / s2, R.se[2] / s2, K1, xs / s2, xse / s2, X))
    rp.check("S3", within_small(od, odse, s2 * (t0.sum() - t.sum())),
             "(9), order 0 minus order 1 on the same paths: %+.5f (SE %.5f) against %+.5f"
             % (od / s2, odse / s2, t0.sum() - t.sum()))
    rp.check("S4", within_small(R.D[4], R.se[4], s2 * (K1 + K2 + 0.5 * X)),
             "K(q) = K1 + K2 + qX at q = 0.5: simulated %+.5f (SE %.5f) against %+.5f"
             % (R.D[4] / s2, R.se[4] / s2, K1 + K2 + 0.5 * X))
    vs = var_shared()
    ts = [K_two([5, 1, 5], vs, o).sum() for o in (0, 1)]
    R = solve_cases(vs, [case([5, 1, 5], [e[0], e[1], None], s2 * ts[0]), case([5, 1, 5], [e[1], e[0], None], s2 * ts[1])], sig, seed=4202)
    od, odse = R.diff(0, 1)
    rp.check("S3b", within_small(od, odse, s2 * (ts[0] - ts[1])) and within_small(R.D[0], R.se[0], s2 * ts[0]),
             "the same for shared weather (hosts of different memory), stays (5,1,5): order 0 %+.5f (SE %.5f) against "
             "%+.5f; order 0 minus order 1 %+.5f (SE %.5f) against %+.5f"
             % (R.D[0] / s2, R.se[0] / s2, ts[0], od / s2, odse / s2, ts[0] - ts[1]))

    # S5 one host on two transfers
    lines = []
    ok = True
    for a_, b_, phi, k2, sd in ((1.0, 10.0, 0.2, 1.0, 4301), (1.0, 10.0, 0.8, 1.0, 4302), (3.0, 7.0, 0.6, -1.0, 4303), (1.0, 10.0, 0.5, -1.0, 4304)):
        want = note_Kplus(a_, b_, phi) if k2 > 0 else note_Kminus(a_, b_, phi)
        R = solve_cases(var_ar1(phi), [case([a_, b_], [np.array([1.0]), np.array([k2])], s2 * want)], sig, seed=sd, npairs=256)
        ok &= within_small(R.D[0], R.se[0], s2 * want)
        lines.append("stays (%g,%g), memory %g, %s: formula %+.5f, simulated %+.5f (SE %.5f)"
                     % (a_, b_, phi, "together" if k2 > 0 else "opposite", want, R.D[0] / s2, R.se[0] / s2))
    rp.check("S5", ok, "(10), one host on both transfers, sigma = 0.05 (the first two are the two sides of the boundary "
             "0.2532 of stays (1,10)): " + "; ".join(lines))

    # S6 long stays
    vf = var_predator(0.9, PI / 2)
    sts = [[50.0, 5, 5], [200.0, 5, 5]]
    Kf = [K_two(s_, vf, 0).sum() for s_ in sts]
    R = solve_cases(vf, [case(s_, [e[0], e[1], None], s2 * k) for s_, k in zip(sts, Kf)], sig, seed=4401, burn=burn_for(sts))
    R2 = solve_cases(vf, [case(s_, [e[0], e[1], None], 1e-4 * k) for s_, k in zip(sts, Kf)], 0.01, seed=4402, burn=burn_for(sts))
    d2, d2se = R2.diff(1, 0)
    Kp200 = K_two([5, 5, 200.0], vp, 0).sum()
    R3 = solve_cases(vp, [case([5, 5, 200.0], [e[0], e[1], None], s2 * Kp200)], sig, seed=4403, burn=burn_for([[5, 5, 200.0]]))
    rp.check("S6", within_small(R.D[1], R.se[1], s2 * Kf[1]) and within_small(d2, d2se, 1e-4 * (Kf[1] - Kf[0]))
             and within_small(R3.D[0], R3.se[0], s2 * Kp200),
             "long stays. Fast rotation (angle pi/2), stays (200,5,5), order 0, sigma = 0.05: %+.5f (SE %.5f) against "
             "K = %+.5f. The same hosts, loss at b = 200 minus loss at b = 50 at sigma = 0.01, in units of sigma^2: "
             "%+.3e (SE %.1e) against K(200) - K(50) = %+.3e (positive: the longer steady stay hurts, at this noise). "
             "Predator and prey, stays (5,5,200), order 0, sigma = 0.05: %+.5f (SE %.5f) against %+.5f"
             % (R.D[1] / s2, R.se[1] / s2, Kf[1], d2 / 1e-4, d2se / 1e-4, Kf[1] - Kf[0], R3.D[0] / s2, R3.se[0] / s2, Kp200))

    # S7 the two sides of the parking boundary b = 5.834 (order 1)
    sts = [[3.0, 5, 5], [4.5, 5, 5], [8.0, 5, 5], [12.0, 5, 5]]
    Kb = [K_two(s_, vp, 1).sum() for s_ in sts]
    R = solve_cases(vp, [case(s_, [e[1], e[0], None], s2 * k) for s_, k in zip(sts, Kb)], sig, seed=4501, npairs=256)
    da, dase = R.diff(1, 0)
    db, dbse = R.diff(3, 2)
    rp.check("S7", within_small(da, dase, s2 * (Kb[1] - Kb[0])) and within_small(db, dbse, s2 * (Kb[3] - Kb[2])) and da > 4 * dase and db < -4 * dbse,
             "(13), both sides of the turning stay 5.834 of (b,5,5), order 1, sigma = 0.05, same paths: lengthening b "
             "from 3 to 4.5 changes D/sigma^2 by %+.5f (SE %.5f) against %+.5f (it hurts); from 8 to 12 by %+.5f "
             "(SE %.5f) against %+.5f (it helps)" % (da / s2, dase / s2, Kb[1] - Kb[0], db / s2, dbse / s2, Kb[3] - Kb[2]))

    # S8 the two sides of (17)
    vfc = var_competitors(0.02, 0.02, -0.8)
    vf4 = two_copies(vfc)
    K_old = K_terms([10, 5], [0], [np.array([1.0, 0])], vfc).sum()
    qs = (0.4, 1.0)
    Kn = [K_two_q([5, 5, 5], vf4, q, 0).sum() for q in qs]
    Ro = solve_cases(vf4, [case([10.0, 5.0], [one, None], s2 * K_old)], sig, seed=4601, npairs=256)
    Rn = solve_cases(vf4, [case([5.0, 5, 5], [q_rows(q)[0], q_rows(q)[1], None], s2 * k) for q, k in zip(qs, Kn)], sig, seed=4601, npairs=256)
    dl = [diff_runs(Rn, i, Ro, 0) for i in range(2)]
    rp.check("S8", all(within_small(dl[i][0], dl[i][1], s2 * (Kn[i] - K_old)) for i in range(2)) and dl[0][0] > 4 * dl[0][1] and dl[1][0] < -4 * dl[1][1],
             "(17), fast competitors, three stages (5,5,5) with two hosts minus two stages (10,5) with one, sigma = "
             "0.05, same paths: at q = 0.4 %+.5f (SE %.5f) against %+.5f (the split raises the loss); at q = 1 %+.5f "
             "(SE %.5f) against %+.5f (it lowers it)"
             % (dl[0][0] / s2, dl[0][1] / s2, Kn[0] - K_old, dl[1][0] / s2, dl[1][1] / s2, Kn[1] - K_old))

    # S9 the two sides of the zero a = 67.96 of (5,a,45)
    sts = [[5, 40.0, 45], [5, 150.0, 45]]
    Kz = [K_two(s_, vp, 0).sum() for s_ in sts]
    R = solve_cases(vp, [case(s_, [e[0], e[1], None], s2 * k) for s_, k in zip(sts, Kz)], sig, seed=4701, npairs=384, burn=burn_for(sts))
    rp.check("S9", all(within_small(R.D[i], R.se[i], s2 * Kz[i]) for i in range(2)) and R.D[0] < -4 * R.se[0] and R.D[1] > 4 * R.se[1],
             "both sides of the zero of K at a = 67.96, stays (5,a,45), order 0, sigma = 0.05: a = 40 gives %+.5f "
             "(SE %.5f) against %+.5f (a gain); a = 150 gives %+.5f (SE %.5f) against %+.5f (a loss)"
             % (R.D[0] / s2, R.se[0] / s2, Kz[0], R.D[1] / s2, R.se[1] / s2, Kz[1]))
    return rp.done()


def sec_HEAD():
    rp = Rep("HEAD. The headline: the order of two linked hosts changes the sign of the loss. Predator and prey, stays (5,50,5)")
    vp = var_predator()
    st = [5.0, 50.0, 5.0]
    e = np.eye(2)
    K0, K1 = K_two(st, vp, 0).sum(), K_two(st, vp, 1).sum()
    rp.p("  second-order coefficients: order 0 (prey on transfer 0, predator on transfer 1, 50 steps later on average) K = %+.6f;"
         % K0)
    rp.p("  order 1 (predator on transfer 0, prey on transfer 1) K = %+.6f.  Bound of the note: D <= sigma^2." % K1)
    rp.p("  (if the lag were read the other way, predator before prey, the two values would simply be exchanged: "
         "%+.6f and %+.6f)" % (K_two(st, var_predator(0.9, -PI / 10), 0).sum(), K_two(st, var_predator(0.9, -PI / 10), 1).sum()))
    rp.p("  simulator 1: 256 antithetic pairs x 40,000 steps after 4,000; both orders on the same host paths.")
    rp.p("  simulator 2: 1,000 independent replicates x 20,000 steps after 3,000; other code, other generator, other root finder.")
    rp.p("")
    rp.p("   sigma |  order 0: sim 1          sim 2             sigma^2 K  |  order 1: sim 1          sim 2             sigma^2 K")
    res1, res2, tab = {}, {}, {}
    sig2 = (0.1, 0.3, 1.0, 1.5)
    for i, sg in enumerate((0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 1.75)):
        cases = [case(st, [e[0], e[1], None], sg * sg * K0), case(st, [e[1], e[0], None], sg * sg * K1)]
        R = solve_cases(vp, cases, sg, steps=40000, burn=4000, npairs=256, seed=900 + i)
        res1[sg] = R
        if sg in sig2:
            res2[sg] = [sim2_root(rot_gen(0.9, PI / 10), o, st, sg, 20000, 3000, 1000, 950 + i, y0=sg * sg * (K0, K1)[o]) for o in (0, 1)]
        cells = []
        for o in (0, 1):
            a = "%+.6f (%.6f)" % (R.D[o], R.se[o])
            b = "%+.6f (%.6f)" % (res2[sg][o][0], res2[sg][o][1]) if sg in sig2 else "        -          "
            cells.append("%s  %s  %+.6f" % (a, b, sg * sg * (K0, K1)[o]))
        tab[sg] = "   %5.2f |  %s  |  %s" % (sg, cells[0], cells[1])
    for sg in sorted(tab):
        rp.p(tab[sg])
    rp.p("   (standard errors in brackets)")
    rp.p("")
    ok = all(within_small(res1[sg].D[o], res1[sg].se[o], sg * sg * (K0, K1)[o]) for sg in (0.05, 0.1) for o in (0, 1))
    rp.check("HEAD1", ok, "at sigma = 0.05 and 0.1 simulator 1 agrees with sigma^2 K in both orders (4 SE + 5%)")
    note = {0.3: ((0.018583, 0.000139), (-0.008295, 0.000196)), 1.0: ((0.200752, 0.001606), (-0.059370, 0.002068))}
    ok = True
    txt = []
    for sg in (0.3, 1.0):
        for o in (0, 1):
            ok &= sim_vs_note(res1[sg].D[o], res1[sg].se[o], *note[sg][o])
            txt.append("sigma %.1f order %d: mine %+.6f (SE %.6f), note %+.6f (SE %.6f)" % (sg, o, res1[sg].D[o], res1[sg].se[o], note[sg][o][0], note[sg][o][1]))
    rp.check("HEAD2", ok, "simulator 1 against the note's four numbers (4 combined SE): " + "; ".join(txt))
    ok = True
    worst = 0.0
    for sg in sig2:
        for o in (0, 1):
            z = (res1[sg].D[o] - res2[sg][o][0]) / math.hypot(res1[sg].se[o], res2[sg][o][1])
            worst = max(worst, abs(z))
    rp.check("HEAD3", worst <= NSE, "the two simulators agree at sigma = 0.1, 0.3, 1.0, 1.5 in both orders: largest "
             "difference %.2f combined SE" % worst)
    ok = all(sim_vs_note(res2[sg][o][0], res2[sg][o][1], *note[sg][o]) for sg in (0.3, 1.0) for o in (0, 1))
    rp.check("HEAD4", ok, "simulator 2 against the note's four numbers (4 combined SE): order 0 %+.6f (SE %.6f) and order 1 "
             "%+.6f (SE %.6f) at sigma = 1" % (res2[1.0][0][0], res2[1.0][0][1], res2[1.0][1][0], res2[1.0][1][1]))
    sgs = sorted(res1)
    neg = [sg for sg in sgs if res1[sg].D[1] < -NSE * res1[sg].se[1]]
    pos = [sg for sg in sgs if res1[sg].D[1] > NSE * res1[sg].se[1]]
    pos0 = [sg for sg in sgs if res1[sg].D[0] > NSE * res1[sg].se[0]]
    rp.check("HEAD5", all(sg in neg for sg in (0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5)) and len(pos0) == len(sgs)
             and res2[1.5][1][0] < -NSE * res2[1.5][1][1]
             and all(res2[sg][1][0] < -NSE * res2[sg][1][1] for sg in (0.1, 0.3, 1.0)),
             "order 0 is a loss by more than 4 SE at every sigma tried (%s). Order 1 is a GAIN by more than 4 SE at "
             "sigma = %s (simulator 2 agrees at 0.1, 0.3 and 1.0), so the difference of sign is there from the "
             "smallest noise: it is a second-order effect, and it persists to sigma = 1.5. Order 1 is a loss by more "
             "than 4 SE at sigma = %s; not resolved from zero at: %s"
             % (", ".join("%g" % s_ for s_ in pos0), ", ".join("%g" % s_ for s_ in neg),
                ", ".join("%g" % s_ for s_ in pos) if pos else "none",
                ", ".join("%g" % s_ for s_ in sgs if s_ not in neg and s_ not in pos) or "no sigma tried"))
    d15 = res1[1.5].D[1], res1[1.5].se[1]
    d15b = res2[1.5][1]
    rp.p("  at sigma = 1.5, order 1: simulator 1 %+.6f (SE %.6f), simulator 2 %+.6f (SE %.6f): still a gain. The gain "
         "ends between sigma = 1.5 and 1.75 (simulator 1 only)." % (d15[0], d15[1], d15b[0], d15b[1]))
    dd = [res1[sg].diff(0, 1) for sg in sgs]
    rp.p("  order 0 minus order 1 on the same paths: " + "; ".join("sigma %g: %+.5f (SE %.5f)" % (sg, d[0], d[1]) for sg, d in zip(sgs, dd)))
    rp.p("  largest gain in order 1 among the sigmas tried: %+.6f at sigma = %g; second-order value there %+.6f"
         % (min(res1[sg].D[1] for sg in sgs), min(sgs, key=lambda s_: res1[s_].D[1]),
            min(sgs, key=lambda s_: res1[s_].D[1]) ** 2 * K1))
    return rp.done()


def sec_GRID():
    rp = Rep("GRID. The note's main grid, stays (5,a,5): loss at sigma = 0.3 and 1 (128 pairs x 40,000 steps, both orders on the same paths)")
    e = np.eye(2)
    worst = 0.0
    n2, n3 = 0, 0
    rows = []
    ex = []
    seed = 1000
    for model in ("predator", "shared", "competitors"):
        vp = grid_var(model)
        for a in (1, 5, 50):
            st = [5.0, float(a), 5.0]
            for col, sg in ((1, 0.3), (3, 1.0)):
                seed += 1
                cases = [two_host_case(st, vp, o, sg)[0] for o in (0, 1)]
                R = solve_cases(vp, cases, sg, seed=seed)
                for o in (0, 1):
                    nt = NOTE_GRID[(model, a, o)]
                    z = (R.D[o] - nt[col]) / math.hypot(R.se[o], nt[col + 1])
                    worst = max(worst, abs(z))
                    n2 += abs(z) > 2
                    n3 += abs(z) > 3
                    rows.append("   %-11s a=%-2d order %d sigma %.1f: mine %+.6f (SE %.6f)  note %+.6f (SE %.6f)  difference %+.1f SE"
                                % (model, a, o, sg, R.D[o], R.se[o], nt[col], nt[col + 1], z))
                if model == "competitors":
                    ex.append(R.diff(0, 1) + (a, sg))
    for r_ in rows:
        rp.p(r_)
    rp.check("GRID1", worst <= NSE, "all 36 simulated losses of the main grid agree with the note's within 4 combined "
             "SE: largest difference %.2f; %d of 36 beyond 2 and %d beyond 3 (about 1.6 and 0.1 expected by chance)" % (worst, n2, n3))
    rp.check("GRID2", all(abs(d) <= NSE * se for d, se, a, sg in ex),
             "competitors are exchangeable, so the two orders have the same loss at any noise (an exact statement): "
             "order 0 minus order 1 on the same paths: " + "; ".join("a=%d sigma %.1f: %+.6f (SE %.6f)" % (a, sg, d, se) for d, se, a, sg in ex))
    return rp.done()


def sec_QB():
    rp = Rep("QB. Both sides of the two boundaries in q; predator and prey, stays (5,50,5), order 1 (256 pairs x 40,000 steps)")
    vp = var_predator()
    vp4 = two_copies(vp)
    st = [5.0, 50.0, 5.0]
    t = K_two(st, vp, 1)
    K1, K2, X = t[0, 0], t[1, 1], t[0, 1] + t[1, 0]
    q_add, q_tot = -K2 / X, -(K1 + K2) / X
    one = np.array([1.0, 0, 0, 0])
    qs = [q_tot - 0.15, q_tot + 0.15, q_add - 0.15, q_add + 0.15, 0.65, 0.8, 1.0]
    res = {}
    for sg, sd in ((0.3, 1101), (1.0, 1102)):
        cases = [case(st, [q_rows(q)[1], one, None], sg * sg * (K1 + K2 + q * X)) for q in qs]
        cases.append(case(st, [None, one, None], sg * sg * K2))
        res[sg] = solve_cases(vp4, cases, sg, npairs=256, seed=sd)
        rp.p("  sigma = %.1f: " % sg + "; ".join("q = %.4f: %+.6f (SE %.6f)" % (q, res[sg].D[i], res[sg].se[i]) for i, q in enumerate(qs))
             + "; one host alone: %+.6f (SE %.6f)" % (res[sg].D[7], res[sg].se[7]))
    R3, R1 = res[0.3], res[1.0]
    note = [(R3, 0, 0.002428, 0.000145), (R3, 1, -0.001632, 0.000168), (R1, 0, 0.061490, 0.001512), (R1, 1, 0.018023, 0.001768),
            (R1, 7, 0.044585, 0.001025), (R3, 6, -0.008295, 0.000196), (R1, 6, -0.059370, 0.002068)]
    worst = max(abs(R.D[i] - d) / math.hypot(R.se[i], s_) for R, i, d, s_ in note)
    rp.check("QB1", worst <= NSE, "against the note (4 combined SE; largest difference %.2f): total-loss boundary "
             "q = 0.3742, sigma = 0.3: q - 0.15 gives %s (note +0.002428, SE 0.000145), q + 0.15 gives %s (note -0.001632, "
             "SE 0.000168); sigma = 1: %s (note +0.061490) and %s (note +0.018023); one host alone at sigma = 1 %s "
             "(note +0.044585, SE 0.001025)"
             % (worst, pm(R3.D[0], R3.se[0]), pm(R3.D[1], R3.se[1]), pm(R1.D[0], R1.se[0]), pm(R1.D[1], R1.se[1]), pm(R1.D[7], R1.se[7])))
    rp.check("QB2", R3.D[0] > NSE * R3.se[0] and R3.D[1] < -NSE * R3.se[1],
             "the two sides of the total-loss boundary have opposite signs by more than 4 SE at sigma = 0.3 (a loss "
             "below, a gain above). At sigma = 1 both are losses, as the note says: the gain needs q above a value "
             "between %s" % " and ".join("%.2f (D = %+.4f, SE %.4f)" % (qs[i], R1.D[i], R1.se[i]) for i in
                                         [max([j for j in (1, 4, 5, 6) if R1.D[j] > 0], key=lambda j: qs[j]),
                                          min([j for j in (1, 4, 5, 6) if R1.D[j] < 0], key=lambda j: qs[j])]))
    dl, dh = R3.diff(2, 7), R3.diff(3, 7)
    dl1, dh1 = R1.diff(2, 7), R1.diff(3, 7)
    t5 = K_two([5, 5, 5], vp, 0)
    two = np.array([0, 1.0, 0, 0])
    txt5 = []
    for sg, sd in ((0.1, 1110), (0.3, 1111), (1.0, 1112)):
        R5 = solve_cases(vp4, [case([5.0, 5, 5], [one, two, None], sg * sg * t5.sum()), case([5.0, 5, 5], [one, None, None], sg * sg * t5[0, 0])],
                         sg, npairs=256, seed=sd)
        txt5.append("sigma %.1f: %s" % (sg, pm(*R5.diff(0, 1))))
    rp.p("  stays (5,5,5), order 0, full linkage: second-order benefit of the second host K2 + X = %+.6f per sigma^2. Two hosts "
         "minus host 1 alone, same paths: %s. The note says this small benefit 'can disappear at finite noise'"
         % (t5.sum() - t5[0, 0], "; ".join(txt5)))
    rp.check("QB3", sim_vs_note(dl[0], dl[1], 0.002206, 0.000108) and sim_vs_note(dh[0], dh[1], -0.001820, 0.000115)
             and dl[0] > NSE * dl[1] and dh[0] < -NSE * dh[1],
             "added-host boundary q = 0.1871, two hosts minus one host alone on the same paths, sigma = 0.3: at "
             "q - 0.15 %s (note +0.002206, SE 0.000108), at q + 0.15 %s (note -0.001820, SE 0.000115). At sigma = 1: "
             "%s and %s (the note calls the second unresolved)" % (pm(dl[0], dl[1]), pm(dh[0], dh[1]), pm(dl1[0], dl1[1]), pm(dh1[0], dh1[1])))
    return rp.done()


def sec_CH():
    rp = Rep("CH. One host on both transfers of a two-stage ring (256 pairs x 40,000 steps)")
    one = np.array([1.0])
    lines = []
    ok = True
    signs = []
    for phi, k2, nd, ns, sd in ((0.2, 1.0, 0.126540, 0.000836, 1201), (0.8, 1.0, -0.166741, 0.002028, 1202), (0.0, -1.0, 0.116266, 0.000945, 1203)):
        vp = var_ar1(phi)
        K = K_terms([1.0, 10.0], [0, 1], [one, one * k2], vp).sum()
        R = solve_cases(vp, [case([1.0, 10.0], [one, one * k2], K)], 1.0, npairs=256, seed=sd)
        ok &= sim_vs_note(R.D[0], R.se[0], nd, ns)
        signs.append(R.D[0] / R.se[0])
        lines.append("memory %g, %s: K = %+.6f; D at sigma = 1 mine %s, note %+.6f (SE %.6f)"
                     % (phi, "together" if k2 > 0 else "opposite", K, pm(R.D[0], R.se[0]), nd, ns))
    rp.check("CH1", ok and signs[0] > NSE and signs[1] < -NSE and signs[2] > NSE,
             "stays (1,10), sigma = 1: " + "; ".join(lines) + ". So in discrete time one host moving two unequal stages "
             "together is a loss at memory 0.2 and a gain at memory 0.8, at sigma = 1 as at second order")
    # the same law written as two hosts with correlation -1 (a singular two-host autoregression)
    v2 = Var(np.zeros((2, 2)), np.array([[1.0, -1.0], [-1.0, 1.0]]))
    R = solve_cases(v2, [case([1.0, 10.0], [np.array([1.0, 0]), np.array([0, 1.0])], 0.1)], 1.0, npairs=256, seed=1203)
    rp.check("CH2", sim_vs_note(R.D[0], R.se[0], 0.116266, 0.000945),
             "two hosts without memory and with correlation -1 (the two-host model at its limit) on stays (1,10), "
             "sigma = 1: %s; the note's one-host value with opposite signs is +0.116266 (SE 0.000945)" % pm(R.D[0], R.se[0]))
    # (12): equal stays
    lines = []
    ok = True
    for sg, phi, sd in ((1.0, 0.0, 1211), (1.0, 0.8, 1212), (0.3, 0.8, 1213)):
        exact = scalar_root(1 - 1 / 4.0, sg, share=0.5)
        R = solve_cases(var_ar1(phi), [case([4.0, 4.0], [one, one], exact)], sg, npairs=256, seed=sd)
        ok &= abs(R.D[0] - exact) <= NSE * R.se[0]
        lines.append("sigma %.1f memory %g: simulated %s, (12) gives %+.6f" % (sg, phi, pm(R.D[0], R.se[0]), exact))
    rng = np.random.default_rng(1220)
    worst = 0.0
    for it in range(60):
        a_ = float(np.exp(rng.uniform(0, math.log(30))))
        per = int(rng.integers(2, 25))
        hs = periodic_hosts(rng.uniform(0.3, 2.0), [rng.standard_normal(per)])
        D = floquet_root(ring([a_, a_]), [ring_masks(2, [0])[0] + ring_masks(2, [1])[0]], hs, ring_masks(2, [0])[0])
        s_ = 1 - 1 / a_
        want = brentq(lambda y: np.mean(np.log(s_ + (1 - s_) * np.exp(y / 2) * hs[0])), -20, 20, xtol=1e-15)
        worst = max(worst, abs(D - want))
    rp.check("CH3", ok and worst < 1e-10, "(12), equal stays a = b = 4, one host on both transfers: the loss solves "
             "E log(s + (1 - s) e^(D/2) h) = 0 whatever the memory: " + "; ".join(lines) + ". Exact in 60 random "
             "periodic environments of large amplitude (the time average in place of E): largest difference %.1e" % worst)
    return rp.done()


def sec_G15():
    rp = Rep("G15. Fixed generation time 15: one host on two stages (10,5) against two linked hosts on three stages (5,5,5); fast competitors")
    vfc = var_competitors(0.02, 0.02, -0.8)
    vf4 = two_copies(vfc)
    one = np.array([1.0, 0, 0, 0])
    K_old = K_terms([10, 5], [0], [np.array([1.0, 0])], vfc).sum()
    tn = K_two([5, 5, 5], vfc, 0)
    qm = 0.55464821794
    res = {}
    for sg, sd in ((0.3, 1301), (0.5, 1302), (0.7, 1303), (1.0, 1304)):
        s2 = sg * sg
        Ro = solve_cases(vf4, [case([10.0, 5.0], [one, None], s2 * K_old)], sg, npairs=256, seed=sd)
        Rn = solve_cases(vf4, [case([5.0, 5, 5], [one, q_rows(1.0)[1], None], s2 * tn.sum()),
                               case([5.0, 5, 5], [one, q_rows(qm)[1], None], s2 * K_two_q([5, 5, 5], vf4, qm, 0).sum()),
                               case([5.0, 5, 5], [one, None, None], s2 * tn[0, 0])], sg, npairs=256, seed=sd)
        res[sg] = (Ro, Rn, diff_runs(Rn, 0, Ro, 0), diff_runs(Rn, 1, Ro, 0), Rn.diff(0, 2))
        rp.p("  sigma = %.1f: old ring %s; new ring, linked %s; new minus old %s; new ring at q = 0.5546 %s; new ring, "
             "host 2 held steady %s; linked minus host 2 steady %s"
             % (sg, pm(Ro.D[0], Ro.se[0]), pm(Rn.D[0], Rn.se[0]), pm(*res[sg][2]), pm(Rn.D[1], Rn.se[1]), pm(Rn.D[2], Rn.se[2]), pm(*res[sg][4])))
    Ro3, Rn3, d3, dq3, ds3 = res[0.3]
    Ro1, Rn1, d1, dq1, ds1 = res[1.0]
    rpt = []
    for sg, sd, nd, ns in ((0.3, 1351, -0.000915, 0.000042), (1.0, 1354, 0.023835, 0.000414)):
        Ro = solve_cases(vf4, [case([10.0, 5.0], [one, None], sg * sg * K_old)], sg, npairs=256, seed=sd)
        Rn = solve_cases(vf4, [case([5.0, 5, 5], [one, q_rows(1.0)[1], None], sg * sg * tn.sum())], sg, npairs=256, seed=sd)
        dd = diff_runs(Rn, 0, Ro, 0)
        first = res[sg][2]
        pooled = (0.5 * (dd[0] + first[0]), 0.5 * math.hypot(dd[1], first[1]))
        rpt.append((sg, dd, pooled, (pooled[0] - nd) / math.hypot(pooled[1], ns)))
    rp.p("  independent repeat of new minus old: " + "; ".join("sigma %.1f: %s; my two runs pooled %s, %+.1f combined SE from the note"
                                                             % (sg, pm(*dd), pm(*pl), z) for sg, dd, pl, z in rpt))
    cmp_ = [(Ro3.D[0], Ro3.se[0], 0.003348, 0.000039), (Rn3.D[0], Rn3.se[0], 0.002433, 0.000072), (d3[0], d3[1], -0.000915, 0.000042),
            (Ro1.D[0], Ro1.se[0], 0.051577, 0.000406), (Rn1.D[0], Rn1.se[0], 0.075412, 0.000723), (d1[0], d1[1], 0.023835, 0.000414),
            (Rn3.D[1], Rn3.se[1], 0.004282, 0.000061), (Rn3.D[2], Rn3.se[2], 0.003357, 0.000039), (Rn1.D[2], Rn1.se[2], 0.052292, 0.000402)]
    worst = max(abs(m - d) / math.hypot(se, s_) for m, se, d, s_ in cmp_)
    worst = max(worst, max(abs(z) for _, _, _, z in rpt))
    cg = comp_gen(0.02, 0.02, -0.8)
    x2 = []
    for sg, sd in ((0.3, 1361), (1.0, 1364)):
        Do, so, _, io = sim2_root(cg, 0, [10.0, 5.0], sg, 20000, 3000, 2000, sd, y0=sg * sg * K_old, fluct=(True, False))
        Dn, sn, _, in_ = sim2_root(cg, 0, [5.0, 5.0, 5.0], sg, 20000, 3000, 2000, sd, y0=sg * sg * tn.sum())
        dd = (Dn - Do, float((in_ - io).std(ddof=1) / math.sqrt(len(io))))
        first = res[sg]
        zz = [(Do - first[0].D[0]) / math.hypot(so, first[0].se[0]), (Dn - first[1].D[0]) / math.hypot(sn, first[1].se[0]),
              (dd[0] - first[2][0]) / math.hypot(dd[1], first[2][1])]
        x2.append((sg, Do, so, Dn, sn, dd, zz))
        rp.p("  second simulator (2,000 independent replicates x 20,000 steps, hosts from their two modes), sigma = %.1f: old "
             "ring %s; new ring %s; new minus old %s; differences from simulator 1: %+.1f, %+.1f, %+.1f combined SE"
             % (sg, pm(Do, so), pm(Dn, sn), pm(*dd), zz[0], zz[1], zz[2]))
    notes_d = {0.3: (-0.000915, 0.000042), 1.0: (0.023835, 0.000414)}
    for (sg, dd, _, _), row in zip(rpt, x2):
        ests = [res[sg][2], dd, row[5]]
        wts = [1 / e_[1] ** 2 for e_ in ests]
        pm3 = (sum(w * e_[0] for w, e_ in zip(wts, ests)) / sum(wts), 1 / math.sqrt(sum(wts)))
        z3 = (pm3[0] - notes_d[sg][0]) / math.hypot(pm3[1], notes_d[sg][1])
        worst = max(worst, abs(z3))
        rp.p("  new minus old at sigma = %.1f, my three independent estimates pooled: %s; note %+.6f (SE %.6f); difference "
             "%+.1f combined SE, that is %+.1f of the note's own standard errors"
             % (sg, pm(*pm3), notes_d[sg][0], notes_d[sg][1], z3, (pm3[0] - notes_d[sg][0]) / notes_d[sg][1]))
    rp.check("G15d", all(abs(z) <= NSE for row in x2 for z in row[6]) and x2[0][5][0] < -NSE * x2[0][5][1] and x2[1][5][0] > NSE * x2[1][5][1],
             "the second simulator agrees with the first on both rings and on their difference at sigma = 0.3 and 1 "
             "(4 combined SE), and shows the same reversal of sign")
    rp.check("G15a", worst <= NSE, "the note's nine numbers (old and new ring and their paired difference at sigma = 0.3 "
             "and 1; q = 0.5546 at 0.3; host 2 steady at 0.3 and 1) against mine, 4 combined SE: largest difference %.2f" % worst)
    rp.check("G15b", d3[0] < -NSE * d3[1] and d1[0] > NSE * d1[1] and dq3[0] > NSE * dq3[1],
             "the split with a linked host lowers the loss at sigma = 0.3 (new minus old %s) and raises it at "
             "sigma = 1 (%s): the reversal is reproduced. At q = 0.5546, below the boundary (17), it raises the loss "
             "already at sigma = 0.3 (%s). The sign changes between sigma = %s"
             % (pm(*d3), pm(*d1), pm(*dq3),
                " and ".join("%.1f (%+.6f, SE %.6f)" % (sg, res[sg][2][0], res[sg][2][1]) for sg in
                             [max(s_ for s_ in res if res[s_][2][0] < 0), min(s_ for s_ in res if res[s_][2][0] > 0)])))
    rp.check("G15c", ds3[0] < -NSE * ds3[1] and ds1[0] > NSE * ds1[1],
             "same ring, two linked hosts against one (host 2 held steady): %s at sigma = 0.3 (the pair beats one "
             "host) and %s at sigma = 1 (it does not)" % (pm(*ds3), pm(*ds1)))
    return rp.done()
def sec_LAG():
    rp = Rep("LAG. Lag cancellation, stays (5,a,45), predator and prey, order 0: the table, and the stay at which the loss is zero at sigma = 1")
    vp = var_predator()
    e = np.eye(2)
    az = 67.95957981591
    avals = [20.0, 40.0, az, 150.0]
    sts = [[5.0, a, 45.0] for a in avals]
    burn = burn_for(sts + [[5.0, 22.0, 45.0]])
    Ks = [K_two(s_, vp, 0).sum() for s_ in sts]
    note = {0.3: [(-0.006376, 0.000220), (-0.001664, 0.000205), (0.000165, 0.000199), (0.001364, 0.000194)],
            1.0: [(-0.034732, 0.002308), (0.005803, 0.002215), (0.018837, 0.002193), (0.022487, 0.002203)]}
    res = {}
    worst = 0.0
    for sg, sd in ((0.3, 1401), (1.0, 1402)):
        R = solve_cases(vp, [case(s_, [e[0], e[1], None], sg * sg * k) for s_, k in zip(sts, Ks)], sg, npairs=256, seed=sd, burn=burn)
        res[sg] = R
        for i, a in enumerate(avals):
            z = (R.D[i] - note[sg][i][0]) / math.hypot(R.se[i], note[sg][i][1])
            worst = max(worst, abs(z))
            rp.p("   a = %-8.4f sigma %.1f: K = %+.8f  mine %s  note %+.6f (SE %.6f)  difference %+.1f SE"
                 % (a, sg, Ks[i], pm(R.D[i], R.se[i]), note[sg][i][0], note[sg][i][1], z))
    rp.check("LAG1", worst <= NSE, "the eight losses of the note's table against mine (256 pairs x 40,000 steps after "
             "%d): largest difference %.2f combined SE" % (burn, worst))
    R3, R1 = res[0.3], res[1.0]
    rp.check("LAG2", abs(R3.D[2]) <= NSE * R3.se[2] and R1.D[2] > NSE * R1.se[2] and sim_vs_note(R1.D[2], R1.se[2], 0.016833, 0.001963),
             "at the zero of the coefficient, a = 67.96: the loss at sigma = 0.3 is %s (not resolved from zero); at "
             "sigma = 1 it is %s, positive by %.1f SE (note: +0.018837, SE 0.002193, and a repeat +0.016833, SE 0.001963). "
             "A zero of K is not a zero of the loss at sigma = 1"
             % (pm(R3.D[2], R3.se[2]), pm(R1.D[2], R1.se[2]), R1.D[2] / R1.se[2]))

    # the stay at which the loss is zero at sigma = 1: growth at y = 0 as a function of a, all a on the same host paths
    agrid = np.array([20.0, 24, 28, 32, 36, 40, 44, 47.57170587114, 52])
    nv = len(agrid)
    Cs = np.zeros((nv, 3, 2))
    Cs[:, 0, 0] = 1.0
    Cs[:, 1, 1] = 1.0
    stg = np.array([[5.0, a, 45.0] for a in agrid])
    r = sim1(vp, stg, np.zeros(nv), Cs, 1.0, 60000, burn, 256, 1410)
    mean, se = r.mean(0), r.std(0, ddof=1) / math.sqrt(r.shape[0])
    T = stg.sum(1)
    rp.p("   growth per generation at y = 0 and sigma = 1 (positive means the loss D is negative), 256 pairs x 60,000 steps:")
    rp.p("   " + "; ".join("a = %.4g: %+.5f (SE %.5f)" % (a, m * t, s_ * t) for a, m, s_, t in zip(agrid, mean, se, T)))
    gain = [a for a, m, s_ in zip(agrid, mean, se) if m > NSE * s_]
    loss = [a for a, m, s_ in zip(agrid, mean, se) if m < -NSE * s_]
    cross = [i for i in range(nv - 1) if mean[i] > 0 >= mean[i + 1]]
    a0 = agrid[cross[0]] + (agrid[cross[0] + 1] - agrid[cross[0]]) * mean[cross[0]] / (mean[cross[0]] - mean[cross[0] + 1])
    fgrid = a0 + np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    Cs = np.zeros((5, 3, 2))
    Cs[:, 0, 0] = 1.0
    Cs[:, 1, 1] = 1.0
    stf = np.array([[5.0, a, 45.0] for a in fgrid])
    est = []
    for sd in (1411, 1412):
        r = sim1(vp, stf, np.zeros(5), Cs, 1.0, 100000, burn, 256, sd)
        Xd = np.stack([np.ones(5), fgrid - a0, (fgrid - a0) ** 2], axis=1)
        coef = np.linalg.lstsq(Xd, r.T, rcond=None)[0]                 # (3, pairs)
        cm = coef.mean(1)
        disc = cm[1] ** 2 - 4 * cm[2] * cm[0]
        u = -2 * cm[0] / (cm[1] - math.sqrt(disc)) if cm[1] < 0 else -2 * cm[0] / (cm[1] + math.sqrt(disc))
        slope = cm[1] + 2 * cm[2] * u
        at = coef[0] + coef[1] * u + coef[2] * u * u
        est.append((a0 + u, at.std(ddof=1) / math.sqrt(len(at)) / abs(slope)))
    wts = [1 / s_ ** 2 for _, s_ in est]
    a_hat = sum(w * a for w, (a, _) in zip(wts, est)) / sum(wts)
    a_se = 1 / math.sqrt(sum(wts))
    rp.check("LAG3", sim_vs_note(a_hat, a_se, 35.48395, 0.88695) and abs(est[0][0] - est[1][0]) <= NSE * math.hypot(est[0][1], est[1][1]),
             "stay a at which the loss is zero at sigma = 1 (growth at y = 0 fitted by a parabola in a at five stays "
             "around the crossing, all on the same paths; two independent runs of 256 pairs x 100,000 steps): %.3f "
             "(SE %.3f) and %.3f (SE %.3f); together %.3f (SE %.3f). Note: 35.48395 (SE 0.88695). The second-order "
             "zero is 67.96" % (est[0][0], est[0][1], est[1][0], est[1][1], a_hat, a_se))
    rp.check("LAG4", len(gain) > 0 and len(loss) > 0 and max(gain) < a_hat < min(loss),
             "my own bracket by sign at 4 SE: the loss is negative at a = %s and positive at a = %s, so the zero lies "
             "between %.4g and %.4g (the note's bracket: 20 < a < 47.57)"
             % (", ".join("%.4g" % a for a in gain), ", ".join("%.4g" % a for a in loss), max(gain), min(loss)))
    return rp.done()


def sec_PARK():
    rp = Rep("PARK. Parking: a long stay in the predator, and a long steady stay with a fast rotation")
    vp = var_predator()
    vp4 = two_copies(vp)
    one = np.array([1.0, 0, 0, 0])
    ps = [1.0, 10.0, 50.0, 200.0]
    note = {1.0: ((0.423240, 0.001101), (0.427510, 0.000820)), 10.0: ((0.068791, 0.001983), (0.255105, 0.001165)),
            50.0: ((-0.199422, 0.002736), (0.097589, 0.001432)), 200.0: ((-0.320755, 0.003369), (0.033122, 0.001587))}
    cases = []
    for p_ in ps:
        for q in (1.0, 0.0):
            cases.append(case([5.0, 5.0, p_], [one, q_rows(q)[1], None], K_two_q([5, 5, p_], vp4, q, 0).sum()))
    burn = burn_for([c["stays"] for c in cases])
    R = solve_cases(vp4, cases, 1.0, npairs=128, seed=1501, burn=burn)
    worst = 0.0
    for i, p_ in enumerate(ps):
        for j, nm in enumerate(("linked", "independent")):
            k = 2 * i + j
            z = (R.D[k] - note[p_][j][0]) / math.hypot(R.se[k], note[p_][j][1])
            worst = max(worst, abs(z))
            rp.p("   stays (5,5,%g) %-11s sigma 1: K = %+.6f  mine %s  note %+.6f (SE %.6f)  difference %+.1f SE"
                 % (p_, nm, K_two_q([5, 5, p_], vp4, 1.0 - j, 0).sum(), pm(R.D[k], R.se[k]), note[p_][j][0], note[p_][j][1], z))
    Rb = solve_cases(vp4, cases, 1.0, npairs=128, seed=1502, burn=3 * burn)
    worst_b = max(abs(Rb.D[2 * i + j] - note[p_][j][0]) / math.hypot(Rb.se[2 * i + j], note[p_][j][1]) for i, p_ in enumerate(ps) for j in (0, 1))
    worst_ab = max(abs(R.D[k] - Rb.D[k]) / math.hypot(R.se[k], Rb.se[k]) for k in range(8))
    pooled = [(0.5 * (R.D[k] + Rb.D[k]), 0.5 * math.hypot(R.se[k], Rb.se[k])) for k in range(8)]
    zs = [(pooled[2 * i + j][0] - note[p_][j][0]) / math.hypot(pooled[2 * i + j][1], note[p_][j][1]) for i, p_ in enumerate(ps) for j in (0, 1)]
    rp.p("   an independent repeat with three times the burn-in: " + "; ".join("%+.6f (SE %.6f)" % (Rb.D[k], Rb.se[k]) for k in range(8)))
    rp.check("PARK1", max(worst, worst_b, worst_ab) <= NSE, "predator parking, order 0, sigma = 1, linked and "
             "independent hosts (128 pairs x 40,000 steps after %d): the note's eight losses against mine, largest "
             "difference %.2f combined SE; against an independent repeat of mine %.2f; my two runs against each other "
             "%.2f; my two runs pooled against the note: %s combined SE"
             % (burn, worst, worst_b, worst_ab, ", ".join("%+.1f" % z for z in zs)))
    rp.check("PARK2", all(R.diff(2 * i, 2 * i - 2)[0] < -NSE * R.diff(2 * i, 2 * i - 2)[1] for i in (1, 2, 3)) and R.D[2] > NSE * R.se[2]
             and R.D[4] < -NSE * R.se[4],
             "each lengthening of the predator stay lowers the linked loss by more than 4 SE; at p = 10 the loss is "
             "still positive at sigma = 1 (%s) though K is negative (%+.4f); at p = 50 and 200 it is negative"
             % (pm(R.D[2], R.se[2]), K_two([5, 5, 10], vp, 0).sum()))

    # the fast rotation: paired differences between steady stays
    vf = var_predator(0.9, PI / 2)
    e = np.eye(2)
    bs = [1.0, 5.0, 20.0, 50.0, 200.0]
    sts = [[b, 5.0, 5.0] for b in bs]
    Kb = [K_two(s_, vf, 0).sum() for s_ in sts]
    burn = burn_for(sts)
    pairs = ((1, 0, "1 to 5"), (3, 2, "20 to 50"), (4, 3, "50 to 200"))
    note = {0.05: ((0.000219422, 0.000000494), (0.000004678, 0.000000020), (0.000001200, 0.000000010)),
            0.3: ((0.007320750, 0.000015602), (-0.000349527, 0.000002925), (-0.000373157, 0.000002733)),
            1.0: ((0.036409439, 0.000065740), (-0.052456986, 0.000342788), (-0.057990991, 0.000600539))}
    res = {}
    for i, sg in enumerate((0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 1.0)):
        Rf = solve_cases(vf, [case(s_, [e[0], e[1], None], sg * sg * k) for s_, k in zip(sts, Kb)], sg, npairs=128, seed=1510 + i, burn=burn)
        res[sg] = [Rf.diff(a_, b_) for a_, b_, _ in pairs]
    rp.p("   fast rotation (radius 0.9, angle pi/2), stays (b,5,5), order 0: loss at the longer steady stay minus loss at the")
    rp.p("   shorter, same host paths; positive means the longer stay hurts.  K(5)-K(1) = %+.6f, K(50)-K(20) = %+.6f, "
         "K(200)-K(50) = %+.6f" % (Kb[1] - Kb[0], Kb[3] - Kb[2], Kb[4] - Kb[3]))
    for sg in sorted(res):
        rp.p("   sigma %-5g: " % sg + "; ".join("b %s: %+.3e (SE %.1e) [sigma^2 dK = %+.3e]" % (nm, d, se, sg * sg * (Kb[a_] - Kb[b_]))
                                                for (a_, b_, nm), (d, se) in zip(pairs, res[sg])))
    worst = 0.0
    wtxt = ""
    for sg in (0.05, 0.3, 1.0):
        for k in range(3):
            z = (res[sg][k][0] - note[sg][k][0]) / math.hypot(res[sg][k][1], note[sg][k][1])
            if abs(z) > worst:
                worst, wtxt = abs(z), "sigma %g, b %s: mine %+.4e (SE %.1e), note %+.4e (SE %.1e)" % (
                    sg, pairs[k][2], res[sg][k][0], res[sg][k][1], note[sg][k][0], note[sg][k][1])
    rp.check("PARK3", worst <= NSE, "the note's nine paired differences (sigma = 0.05, 0.3, 1) against mine, 4 combined "
             "SE: largest difference %.2f (%s)" % (worst, wtxt))
    hurt = [sg for sg in sorted(res) if res[sg][2][0] > NSE * res[sg][2][1]]
    help_ = [sg for sg in sorted(res) if res[sg][2][0] < -NSE * res[sg][2][1]]
    hurt2 = [sg for sg in sorted(res) if res[sg][1][0] > NSE * res[sg][1][1]]
    help2 = [sg for sg in sorted(res) if res[sg][1][0] < -NSE * res[sg][1][1]]
    rp.check("PARK4", all(res[sg][0][0] > NSE * res[sg][0][1] for sg in res) and 0.05 in hurt and 0.3 in help_ and 1.0 in help_,
             "the short extension (b from 1 to 5) hurts at every sigma tried. The long one (b from 50 to 200) hurts by "
             "more than 4 SE at sigma = %s and helps by more than 4 SE at sigma = %s: the second-order harm of a long "
             "steady stay is real but holds only below a noise between %g and %g. For b from 20 to 50: hurts at %s, "
             "helps at %s"
             % (", ".join("%g" % s_ for s_ in hurt), ", ".join("%g" % s_ for s_ in help_), max(hurt), min(help_),
                ", ".join("%g" % s_ for s_ in hurt2), ", ".join("%g" % s_ for s_ in help2)))
    r4 = res[0.05][2][0] / 0.05 ** 2 - (Kb[4] - Kb[3])
    rp.p("   at sigma = 0.05 the long difference is %.0f%% of its second-order value: the next term is already %.0f%% of it; "
         "it is why the small-noise check S6 of this difference uses sigma = 0.01"
         % (100 * res[0.05][2][0] / (0.0025 * (Kb[4] - Kb[3])), 100 * abs(r4) / (Kb[4] - Kb[3])))
    return rp.done()


def sec_B():
    rp = Rep("B. Statements said to hold at any noise: the bound D <= sum of sigma_e^2 / 2 with linked hosts, and D >= 0 without memory")
    rng = np.random.default_rng(606)

    # B1 exact, periodic environments with linked hosts
    worst, ratio_max, nneg, nroot = 1e9, 0.0, 0, 0
    for it in range(3000):
        n = int(rng.integers(2, 6))
        stays = rand_stays(rng, n, 1.0, 12.0, p_one=0.25)
        per = int(rng.integers(2, 25))
        nf = int(rng.integers(1, n + 1))
        tr = rng.choice(n, nf, replace=False).tolist()
        nfac = int(rng.integers(1, 3))
        F = rng.standard_normal((per, nfac))
        load = rng.standard_normal((nf, nfac))
        idio = rng.uniform(0, 0.5)
        logs = [F @ load[k] + idio * rng.standard_normal(per) for k in range(nf)]
        hs = periodic_hosts(rng.uniform(0.3, 2.0), logs)
        ell = sum(-np.log(h).mean() for h in hs)
        A0 = ring(stays)
        masks = ring_masks(n, tr)
        bm = ring_masks(n, [0])[0]
        worst = min(worst, floquet_growth(A0, masks, hs, bm, ell))
        if it % 10 == 0:
            D = floquet_root(A0, masks, hs, bm)
            nroot += 1
            ratio_max = max(ratio_max, D / ell)
            nneg += D < 0
    rp.check("B1", worst > -1e-9, "exact, no random numbers: 3,000 random rings (2 to 5 stages, stays 1 to 12, a quarter "
             "of the stays exactly 1) in random periodic environments (period 2 to 24, 1 to n fluctuating transfers "
             "sharing one or two common factors, log-amplitude 0.3 to 2), multipliers of mean one: growth per step at "
             "y = the sum of the log penalties is never below %.1e (the bound says it is not negative). In the 300 "
             "of them solved for D, the largest D/bound is %.6f and D is negative in %d" % (worst, ratio_max, nneg))

    # B2 no stays: equality when the period and the number of stages have no common factor
    worst_eq, min_ratio = 0.0, 1e9
    for it in range(120):
        n = int(rng.integers(2, 6))
        per = int(rng.integers(2, 25))
        nf = int(rng.integers(1, n + 1))
        tr = rng.choice(n, nf, replace=False).tolist()
        F = rng.standard_normal(per)
        logs = [F * rng.standard_normal() + 0.5 * rng.standard_normal(per) for k in range(nf)]
        hs = periodic_hosts(rng.uniform(0.3, 2.0), logs)
        ell = sum(-np.log(h).mean() for h in hs)
        D = floquet_root(ring(np.ones(n)), ring_masks(n, tr), hs, ring_masks(n, [0])[0], lo=-200, hi=200)
        if math.gcd(per, n) == 1:
            worst_eq = max(worst_eq, abs(D - ell))
        else:
            min_ratio = min(min_ratio, D / ell)
            worst = min(worst, ell - D)
    rp.check("B2", worst_eq < 1e-10 and worst > -1e-9, "no staying at all (every stay 1), 120 random periodic "
             "environments: when the period and the number of stages have no common factor the loss EQUALS the bound "
             "(largest difference %.1e); when they have one, cohorts see different parts of the cycle and the loss is "
             "below it, never above (D/bound goes down to %.2f: a cohort that always meets the favourable part of the "
             "cycle gains)" % (worst_eq, min_ratio))

    # B2b exact, aimed at the cases where the bound is nearly attained WITH staying: one slow host moving
    # several transfers with relative sizes that sum to zero (in a frozen environment the factors cancel
    # and the loss equals the bound)
    worst_slow, ratio_slow, nslow = 1e9, 0.0, 0
    for it in range(400):
        n = int(rng.integers(2, 5))
        stays = rand_stays(rng, n, 1.0, 12.0, p_one=0.2)
        per = int(rng.integers(60, 401))
        tt = np.arange(per)
        base = np.sin(2 * PI * tt / per + rng.uniform(0, 2 * PI)) + 0.3 * np.sin(4 * PI * tt / per + rng.uniform(0, 2 * PI))
        kap = rng.uniform(0.3, 1.5, n)
        kap[0] = -kap[1:].sum()
        if n == 2 and it % 2:
            kap = np.array([1.0, -1.0])
        hs = periodic_hosts(rng.uniform(0.5, 2.0), [kap[k] * base for k in range(n)])
        ell = sum(-np.log(h).mean() for h in hs)
        A0 = ring(stays)
        masks = ring_masks(n, list(range(n)))
        bm = ring_masks(n, [0])[0]
        worst_slow = min(worst_slow, floquet_growth(A0, masks, hs, bm, ell))
        if it % 5 == 0:
            nslow += 1
            ratio_slow = max(ratio_slow, floquet_root(A0, masks, hs, bm, lo=-50, hi=50) / ell)
    rp.check("B2b", worst_slow > -1e-9, "exact, aimed at near-equality with staying: 400 rings with every transfer moved "
             "by ONE slow periodic host (period 60 to 400 steps) with relative sizes that sum to zero, log-amplitude "
             "0.5 to 2: growth at y = the bound is never below %.1e; in the %d solved for D the largest D/bound is "
             "%.6f (the bound is approached, not passed)" % (worst_slow, nslow, ratio_slow))

    # B3 Gaussian hosts from random vector autoregressions, sigma = 1
    total, near, zmin, slack_min = 0, 0, 1e9, 1e9
    flagged = []
    kinds = {"real": 0, "complex": 0}
    for run in range(52):
        d = int(rng.choice([2, 3, 4]))
        n = int(rng.choice([2, 3, 4]))
        kind = "complex" if run % 2 else "real"
        kinds[kind] += 1
        while True:
            vp0 = random_var(rng, d, kind)
            rnk = int(rng.integers(1, d + 1))
            Bm = rng.standard_normal((d, rnk))
            try:
                vp = Var(vp0.M, Bm @ Bm.T + 1e-9 * np.eye(d))
                break
            except Exception:
                continue
        V = 24
        stays = np.empty((V, n))
        for v in range(V):
            if v % 2 == 0:
                while True:
                    st = 1 + rng.exponential(0.2, n)
                    st[rng.random(n) < 0.2] = 1.0
                    if st.max() > 1.05:
                        break
                stays[v] = st
            else:
                stays[v] = rand_stays(rng, n, 1.0, 20.0)
        Cs = np.zeros((V, n, d))
        for v in range(V):
            fl = rng.random(n) < 0.75
            if not fl.any():
                fl[int(rng.integers(0, n))] = True
            for k in range(n):
                if fl[k]:
                    kap = rng.choice([-1.0, 1.0]) * (1.0 if rng.random() < 0.5 else rng.uniform(0.3, 1.5))
                    Cs[v, k, int(rng.integers(0, d))] = kap
        bound = 0.5 * np.einsum("vkd,de,vke->v", Cs, vp.G, Cs)
        burn = burn_for(stays, floor=2000, cap=20000)
        r = sim1(vp, stays, bound, Cs, 1.0, 8000, burn, 48, 6000 + run)
        mean, se = r.mean(0), r.std(0, ddof=1) / math.sqrt(48)
        z = mean / se
        total += V
        near += int((z < NSE).sum())
        zmin = min(zmin, float(z.min()))
        slack_min = min(slack_min, float((mean * stays.sum(1) / bound).min()))
        for v in np.where(z < -NSE)[0]:
            r2 = sim1(vp, stays[v:v + 1], bound[v:v + 1], Cs[v:v + 1], 1.0, 80000, burn, 48, 9000 + run)
            z2 = float(r2.mean() / (r2.std(ddof=1) / math.sqrt(48)))
            flagged.append((run, int(v), float(z[v]), z2))
    nviol = sum(1 for f in flagged if f[0] < 100 and f[3] < -NSE)
    nflag = sum(1 for f in flagged if f[0] < 100)
    rp.check("B3", nviol == 0, "search with Gaussian hosts at sigma = 1: %d random cases (52 random stable vector "
             "autoregressions of dimension 2 to 4, %d with real and %d with complex eigenvalues, innovations of any "
             "rank, so the hosts range from loosely to rigidly linked; rings of 2 to 4 stages, half of them with "
             "every stay between 1 and about 1.5, where the bound is nearly attained; 1 to n fluctuating transfers "
             "with relative sizes of either sign). Growth at y = the bound: negative by more than 4 SE in %d cases "
             "(%d flagged at first, rerun ten times longer); most adverse %+.1f SE; within 4 SE of zero in %d (the "
             "near-tight ones); smallest (growth per generation)/bound %.4f"
             % (total, kinds["real"], kinds["complex"], nviol, nflag, zmin, near, slack_min))


    # aimed cases: slow hosts, relative sizes of opposite sign, short stays
    total_a, near_a, zmin_a, slack_a = 0, 0, 1e9, 1e9
    for run in range(24):
        n = int(rng.choice([2, 3, 4]))
        V = 24
        if run % 2 == 0:
            phi = float(rng.choice([0.9, 0.97, 0.99, 0.995]))
            vp = var_ar1(phi)
            d = 1
        else:
            vp = var_predator(float(rng.choice([0.97, 0.99, 0.995])), float(rng.choice([0.02, 0.05, 0.1])))
            d = 2
        stays = np.empty((V, n))
        Cs = np.zeros((V, n, d))
        for v in range(V):
            if v % 2 == 0:
                while True:
                    st = 1 + rng.exponential(0.2, n)
                    st[rng.random(n) < 0.2] = 1.0
                    if st.max() > 1.05:
                        break
                stays[v] = st
            else:
                stays[v] = rand_stays(rng, n, 1.0, 20.0)
            kap = rng.uniform(0.3, 1.5, n)
            if v % 3 != 2:
                kap[0] = -kap[1:].sum()
            else:
                kap = kap * rng.choice([-1.0, 1.0], n)
            for k in range(n):
                Cs[v, k, 0 if (d == 1 or v % 4 < 2) else int(rng.integers(0, 2))] = kap[k]
        bound = 0.5 * np.einsum("vkd,de,vke->v", Cs, vp.G, Cs)
        burn = burn_for(stays, floor=4000, cap=20000)
        r = sim1(vp, stays, bound, Cs, 1.0, 20000, burn, 48, 6500 + run)
        mean, se = r.mean(0), r.std(0, ddof=1) / math.sqrt(48)
        z = mean / se
        total_a += V
        near_a += int((z < NSE).sum())
        zmin_a = min(zmin_a, float(z.min()))
        slack_a = min(slack_a, float((mean * stays.sum(1) / bound).min()))
        for v in np.where(z < -NSE)[0]:
            r2 = sim1(vp, stays[v:v + 1], bound[v:v + 1], Cs[v:v + 1], 1.0, 200000, burn, 48, 9500 + run)
            z2 = float(r2.mean() / (r2.std(ddof=1) / math.sqrt(48)))
            flagged.append((100 + run, int(v), float(z[v]), z2))
    rp.check("B3b", sum(1 for f in flagged if f[0] >= 100 and f[3] < -NSE) == 0, "the same search aimed at near-equality: %d cases with slow hosts (one host of memory "
             "0.9 to 0.995, or a slow rotation of radius 0.97 to 0.995 and angle 0.02 to 0.1), every transfer "
             "fluctuating, relative sizes that sum to zero in two thirds of them, half with stays near 1; 48 pairs x "
             "20,000 steps: growth at y = the bound negative by more than 4 SE in %d; most adverse %+.1f SE; within "
             "4 SE of zero in %d; smallest (growth per generation)/bound %.5f"
             % (total_a, sum(1 for f in flagged if f[0] >= 100 and f[3] < -NSE), zmin_a, near_a, slack_a))

    other = []
    for sg, sd in ((0.5, 6700), (2.0, 6800)):
        tot_o, zmin_o, slack_o, nv_o = 0, 1e9, 1e9, 0
        for run in range(6):
            n = int(rng.choice([2, 3, 4]))
            V = 24
            vp = var_ar1(float(rng.choice([0.9, 0.99]))) if run % 2 == 0 else var_predator(float(rng.choice([0.9, 0.99])), float(rng.choice([0.05, 0.3])))
            d = vp.d
            stays = np.array([rand_stays(rng, n, 1.0, 20.0) if v % 2 else np.maximum(1.0, 1 + rng.exponential(0.2, n)) for v in range(V)])
            stays[:, 0] = np.maximum(stays[:, 0], 1.06)
            Cs = np.zeros((V, n, d))
            for v in range(V):
                kap = rng.uniform(0.3, 1.5, n)
                kap[0] = -kap[1:].sum() if v % 3 != 2 else kap[0]
                for k in range(n):
                    Cs[v, k, 0 if d == 1 else int(rng.integers(0, 2))] = kap[k]
            bound = 0.5 * sg * sg * np.einsum("vkd,de,vke->v", Cs, vp.G, Cs)
            r = sim1(vp, stays, bound, Cs, sg, 20000, burn_for(stays, floor=4000, cap=20000), 48, sd + run)
            mean, se = r.mean(0), r.std(0, ddof=1) / math.sqrt(48)
            z = mean / se
            tot_o += V
            zmin_o = min(zmin_o, float(z.min()))
            slack_o = min(slack_o, float((mean * stays.sum(1) / bound).min()))
            for v in np.where(z < -NSE)[0]:
                r2 = sim1(vp, stays[v:v + 1], bound[v:v + 1], Cs[v:v + 1], sg, 200000, 4000, 48, sd + 50 + run)
                nv_o += float(r2.mean() / (r2.std(ddof=1) / math.sqrt(48))) < -NSE
        other.append((sg, tot_o, nv_o, zmin_o, slack_o))
    rp.check("B3c", all(o[2] == 0 for o in other), "aimed cases at other noise levels: " + "; ".join(
        "sigma = %g: %d cases, %d below zero by 4 SE, most adverse %+.1f SE, smallest (growth per generation)/bound %.5f" % o for o in other))

    # B4 matrices independent in time (hosts without memory, any correlation within a step): D >= 0
    total, zmax, npos = 0, -1e9, 0
    flagged = []
    for run in range(10):
        d = int(rng.choice([2, 3]))
        n = int(rng.choice([2, 3, 4]))
        rnk = int(rng.integers(1, d + 1))
        Bm = rng.standard_normal((d, rnk))
        vp = Var(np.zeros((d, d)), Bm @ Bm.T + 1e-9 * np.eye(d))
        V = 24
        stays = np.array([rand_stays(rng, n, 1.0, 20.0) for _ in range(V)])
        Cs = np.zeros((V, n, d))
        for v in range(V):
            fl = rng.random(n) < 0.75
            if not fl.any():
                fl[int(rng.integers(0, n))] = True
            for k in range(n):
                if fl[k]:
                    Cs[v, k, int(rng.integers(0, d))] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
        burn = burn_for(stays, floor=2000, cap=20000)
        r = sim1(vp, stays, np.zeros(V), Cs, 1.0, 8000, burn, 48, 7000 + run)
        mean, se = r.mean(0), r.std(0, ddof=1) / math.sqrt(48)
        z = mean / se
        total += V
        zmax = max(zmax, float(z.max()))
        npos += int((z > NSE).sum())
    rp.check("B4", npos == 0, "hosts without memory (matrices independent in time) but correlated in any way within a "
             "step, sigma = 1, %d random cases: growth at y = 0 is positive by more than 4 SE in %d (the note: the "
             "loss cannot be negative); largest %+.1f SE" % (total, npos, zmax))
    return rp.done()


def sec_CT():
    rp = Rep("CT. One host on two transfers: discrete time against the manuscript's continuous-time statement")
    mp.mp.dps = 40

    def Kd(A, B, alpha, k1, k2, m):
        phi = mp.exp(-mp.mpf(alpha) / m)
        return mp_sum(coef_ring_mp([mp.mpf(A) * m, mp.mpf(B) * m], [0, 1], [[k1], [k2]], mp.matrix([[phi]]), mp.matrix([[1]])))

    A, B, al = 60, 10, mp.mpf(1) / 100
    rp.p("  The manuscript's example: stays of 60 and 10 days, the host forgets at rate 1/100 per day; continuous-time values per sigma^2:")
    cont = {nm: manuscript_continuous(0.6, 0.1, k1, k2) for nm, k1, k2 in (("together", 1, 1), ("opposite", 1, -1), ("one transfer", 1, 0))}
    rp.p("    together %+.6f, opposite %+.6f, one transfer alone %+.6f  (manuscript: a gain of 0.47, a cost of 0.92, a cost of 0.11)"
         % (cont["together"], cont["opposite"], cont["one transfer"]))
    rp.p("  General discrete coefficient with a time step of 1/m day (stays 60m and 10m steps, memory exp(-1/(100m)) per step):")
    rp.p("      m     together      (x m: error)   opposite      (x m: error)   one transfer  (x m: error)")
    errs = {nm: [] for nm in cont}
    for j in range(0, 11):
        m = 2 ** j
        row = "   %5d" % m
        for nm, k1, k2 in (("together", 1, 1), ("opposite", 1, -1), ("one transfer", 1, 0)):
            v = float(Kd(A, B, al, k1, k2, m))
            errs[nm].append(v - cont[nm])
            row += "   %+.8f  (%+.5f)" % (v, m * (v - cont[nm]))
        rp.p(row)
    for step, (a_, b_) in (("5 days", (12, 2)), ("10 days", (6, 1))):
        phi = math.exp(-float(al) * (60 / a_))
        rp.p("  with a time step of %s (stays %d and %d steps, memory %.4f): together %+.6f, opposite %+.6f, one transfer %+.6f"
             % (step, a_, b_, phi, K_terms([a_, b_], [0, 1], [np.array([1.0]), np.array([1.0])], var_ar1(phi)).sum(),
                K_terms([a_, b_], [0, 1], [np.array([1.0]), np.array([-1.0])], var_ar1(phi)).sum(),
                K_terms([a_, b_], [0], [np.array([1.0])], var_ar1(phi)).sum()))
    ratios = [errs[nm][j] / errs[nm][j + 1] for nm in errs for j in range(6, 10)]
    rp.check("CT1", all(1.9 < r_ < 2.1 for r_ in ratios),
             "the error of the discrete coefficient halves each time the step is halved (ratios between %.4f and "
             "%.4f for m from 64 to 1024, all three arrangements): convergence of first order in the time step"
             % (min(ratios), max(ratios)))
    rng = np.random.default_rng(808)
    worst = 0.0
    for it in range(100):
        A_, B_ = [float(x) for x in np.exp(rng.uniform(0, math.log(60), 2))]
        al_ = float(np.exp(rng.uniform(math.log(0.003), 0)))
        k1, k2 = [float(x) for x in rng.uniform(-2, 2, 2)]
        Ks = [Kd(A_, B_, al_, k1, k2, m) for m in (4096, 8192, 16384)]
        lim = float((8 * Ks[2] - 6 * Ks[1] + Ks[0]) / 3)
        want = manuscript_continuous(al_ * A_, al_ * B_, k1, k2)
        scale = max(abs(want), 1e-3 * (abs(k1) + abs(k2)) ** 2)
        worst = max(worst, abs(lim - want) / scale)
    rp.check("CT2", worst < TOL_LIMIT, "limit of the general discrete coefficient as the step shrinks (stays growing, "
             "the host's correlation time fixed in units of the stays; fitted at m = 4096, 8192, 16384 in 40-digit "
             "arithmetic) against the manuscript's continuous-time expression: 100 random (stays, forgetting rate, "
             "relative sizes k1, k2 of either sign), largest relative difference %.1e. The discrete formulas tend "
             "to the continuous one" % worst)
    # where the continuous statement fails in discrete time
    e1 = np.array([1.0])
    never = ((2.0, 3.0), (1.0, 4.0), (5.0, 9.0), (10.0, 16.0), (3.0, 7.0))
    assert all((a_ - b_) ** 2 <= 2 * (a_ + b_) for a_, b_ in never)
    nog = sum(K_terms([a_, b_], [0, 1], [e1, e1], var_ar1(phi)).sum() <= 0
              for a_, b_ in never for phi in (0.0, 0.3, 0.6, 0.9, 0.99, 0.999))
    exs = []
    for a_, b_ in ((1.0, 10.0), (5.0, 50.0), (60.0, 10.0), (2.0, 12.0)):
        den = (a_ - b_) ** 2 - 2 * (a_ + b_)
        th = 2 * a_ * b_ / den
        ph = th / (1 + th)
        lo = K_terms([a_, b_], [0, 1], [e1, e1], var_ar1(ph * 0.9)).sum()
        hi = K_terms([a_, b_], [0, 1], [e1, e1], var_ar1(ph + 0.1 * (1 - ph))).sum()
        exs.append("(%g,%g): gain only above memory %.6f (K = %+.5f just below, %+.5f above)" % (a_, b_, ph, lo, hi))
    rp.check("CT3", nog == 0, "in discrete time, moving together is NOT a gain 'unless the stages are left at equal "
             "rates'. It is a loss at every memory when (a - b)^2 <= 2(a + b): stays (2,3), (1,4), (5,9), (10,16), "
             "(3,7) at memories 0 to 0.999 give K > 0 in all 30 cases. Otherwise it is a gain only above a memory: "
             + "; ".join(exs))
    return rp.done()


SECTIONS = {
    "T": sec_T,
    "V": sec_V,
    "A": sec_A,
    "Q": sec_Q,
    "S": sec_S,
    "HEAD": sec_HEAD,
    "GRID": sec_GRID,
    "QB": sec_QB,
    "CH": sec_CH,
    "G15": sec_G15,
    "LAG": sec_LAG,
    "PARK": sec_PARK,
    "B": sec_B,
    "CT": sec_CT,
}
GROUPS = [["T", "V", "A", "Q", "CT", "S", "QB", "CH", "GRID"], ["HEAD", "G15"], ["LAG", "PARK", "B"]]


def run_section(name):
    try:
        return name, SECTIONS[name]()
    except Exception:
        import traceback
        return name, ("\n" + "=" * 100 + "\nSECTION %s CRASHED\n%s" % (name, traceback.format_exc()),
                      [(name + ".crash", False, "section crashed")])


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print(" ".join(SECTIONS))
        return 0
    names = list(SECTIONS)
    if "--only" in args:
        names = args[args.index("--only") + 1].split(",")
    t0 = time.time()
    print("verify_mechanism_linked_hosts.py   started %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("numpy %s, python %s" % (np.__version__, sys.version.split()[0]))
    print("tolerances: TOL_ALG %.0e, TOL_LIMIT %.0e, TOL_FLOQ %.0e, quoted numbers 0.6 of the last digit, "
          "simulation %g SE (+ %g%% at small noise), covariances %g SE + %g"
          % (TOL_ALG, TOL_LIMIT, TOL_FLOQ, NSE, 100 * SMALL_REL, NSE, COV_ABS))
    groups = [[n for n in g if n in names] for g in GROUPS]
    groups = [g for g in groups if g]
    results = {}
    if len(groups) <= 1:
        for g in groups:
            for n in g:
                results[n] = run_section(n)[1]
    else:
        from multiprocessing import Pool
        with Pool(2) as pool:                                   # main process + 2 workers = 3 processes
            pending = [pool.map_async(run_section, g, chunksize=len(g)) for g in groups[1:3]]
            for n in groups[0]:
                results[n] = run_section(n)[1]
            for pnd in pending:
                for n, res in pnd.get():
                    results[n] = res
    allchecks = []
    for n in SECTIONS:
        if n in results:
            text, checks = results[n]
            print(text)
            allchecks += checks
    nfail = sum(1 for c in allchecks if c[1] is False)
    ndiff = sum(1 for c in allchecks if c[1] is None)
    print("\n" + "=" * 100)
    print("SUMMARY: %d checks: %d pass, %d fail, %d differ from the note's printed digits.  Wall time %.1f s"
          % (len(allchecks), len(allchecks) - nfail - ndiff, nfail, ndiff, time.time() - t0))
    for key, ok, text in allchecks:
        if ok is False:
            print("  FAILED  " + key + ": " + text[:300])
    for key, ok, text in allchecks:
        if ok is None:
            print("  DIFFERS FROM THE NOTE  " + key + ": " + text[:300])
    print("finished %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
