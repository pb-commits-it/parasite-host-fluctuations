"""Numerical tests of the steps of the proof in `notes/error-bound.md`.

Claim checked: each inequality and identity used in that proof holds on the cases tried here.
These tests cannot prove the lemmas; they are there to catch an algebra slip or a wrong inequality.
Tolerances, stated beforehand:
  - an inequality counts as violated if left > right * (1 + 1e-12);
  - identities between two exact evaluations must agree to 1e-8 relative (derivatives up to order 4
    of quantities of order one, in double precision);
  - a Monte Carlo mean must lie within 4 standard errors of its exact value.

Run: python3 -I checks/error_bound_steps.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

ORDER = 4
FACT = np.array([1.0, 1.0, 2.0, 6.0, 24.0])


# ---------------------------------------------------------------- Taylor arithmetic in sigma
# A "jet" is an array whose last axis holds Taylor coefficients c_0..c_4 in (sigma - sigma0).

def jmul(a, b):
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape))
    for i in range(ORDER + 1):
        for j in range(ORDER + 1 - i):
            out[..., i + j] += a[..., i] * b[..., j]
    return out


def jrecip(a):
    out = np.zeros_like(a)
    out[..., 0] = 1.0 / a[..., 0]
    for k in range(1, ORDER + 1):
        acc = np.zeros_like(a[..., 0])
        for i in range(1, k + 1):
            acc += a[..., i] * out[..., k - i]
        out[..., k] = -acc / a[..., 0]
    return out


def jexp(a):
    out = np.zeros_like(a)
    out[..., 0] = np.exp(a[..., 0])
    for k in range(1, ORDER + 1):
        acc = np.zeros_like(a[..., 0])
        for i in range(1, k + 1):
            acc += i * a[..., i] * out[..., k - i]
        out[..., k] = acc / k
    return out


def jlog(a):
    out = np.zeros_like(a)
    out[..., 0] = np.log(a[..., 0])
    for k in range(1, ORDER + 1):
        acc = k * a[..., k]
        for i in range(1, k):
            acc -= i * out[..., i] * a[..., k - i]
        out[..., k] = acc / (k * a[..., 0])
    return out


def h_jet(xi, sigma0):
    """Taylor coefficients of exp(sigma*xi - sigma^2/2) about sigma0. xi has shape (..., n)."""
    e = np.zeros(xi.shape + (ORDER + 1,))
    e[..., 0] = sigma0 * xi - 0.5 * sigma0 ** 2
    e[..., 1] = xi - sigma0
    e[..., 2] = -0.5
    return jexp(e)


def ar1_paths(rng, reps, steps, n, phi):
    """Stationary Gaussian AR(1), unit variance: array (steps, reps, n)."""
    xi = np.empty((steps, reps, n))
    xi[0] = rng.normal(size=(reps, n))
    c = math.sqrt(1.0 - phi * phi)
    for t in range(1, steps):
        xi[t] = phi * xi[t - 1] + c * rng.normal(size=(reps, n))
    return xi


def run_jets(xi, s, gbar, sigma0):
    """Propagate the exact recursion with Taylor arithmetic.

    Returns (lnG, w): lnG[t] is the jet of ln G_t, shape (steps, reps, 5); w[t] the jet of the
    composition at time t, shape (steps + 1, reps, n, 5).
    """
    steps, reps, n = xi.shape
    w = np.zeros((steps + 1, reps, n, ORDER + 1))
    w[0, :, :, 0] = 1.0 / n
    lng = np.zeros((steps, reps, ORDER + 1))
    for t in range(steps):
        h = h_jet(xi[t], sigma0)
        moved = gbar * jmul(h, w[t])                       # from stage k
        new = s * w[t] + np.roll(moved, 1, axis=1)         # into stage k+1
        g = new.sum(axis=1)                                # jet of G_t
        lng[t] = jlog(g)
        w[t + 1] = jmul(new, jrecip(g)[:, None, :])
    return lng, w


# ---------------------------------------------------------------- pieces of the proof

def hermite_h_derivs(xi, sigma):
    """h^(l) = He_l(xi - sigma) * h for l = 0..4. Returns array (5,) + xi.shape."""
    x = xi - sigma
    h = np.exp(sigma * xi - 0.5 * sigma ** 2)
    he = [np.ones_like(x), x, x * x - 1, x ** 3 - 3 * x, x ** 4 - 6 * x * x + 3]
    return np.array([p * h for p in he])


def binom(a, b):
    return math.comb(a, b)


def explicit_recursion(xi, s, gbar, sigma):
    """Derivatives w_j(t), j = 0..4, from the recursion written in the proof (J(t) and f_j(t))."""
    steps, reps, n = xi.shape
    w = np.zeros((ORDER + 1, reps, n))
    w[0] = 1.0 / n
    lng4 = np.zeros((steps, reps))
    out_w = [w.copy()]
    for t in range(steps):
        hd = hermite_h_derivs(xi[t], sigma)                 # (5, reps, n)
        h = hd[0]
        G = s + gbar * np.sum(h * w[0], axis=1)             # (reps,)
        # derivatives of G that do not need w_j(t+1): G^(i) = gbar * sum_l C(i,l) h^(l).w_(i-l)
        Gd = np.zeros((ORDER + 1, reps))
        Gd[0] = G
        for i in range(1, ORDER + 1):
            for l in range(i + 1):
                Gd[i] += gbar * binom(i, l) * np.sum(hd[l] * w[i - l], axis=1)
        new = np.zeros_like(w)
        Aw = s * w[0] + np.roll(gbar * h * w[0], 1, axis=1)
        new[0] = Aw / G[:, None]
        for j in range(1, ORDER + 1):
            # J(t) w_j(t) = (1/G) (I - w(t+1) 1^T) A(t) w_j(t)
            Awj = s * w[j] + np.roll(gbar * h * w[j], 1, axis=1)
            Jw = (Awj - new[0] * Awj.sum(axis=1, keepdims=True)) / G[:, None]
            f = np.zeros((reps, n))
            for i in range(1, j + 1):
                f += binom(j, i) * np.roll(gbar * hd[i] * w[j - i], 1, axis=1)
            for i in range(1, j):
                f -= binom(j, i) * Gd[i][:, None] * new[j - i]
            gam = np.zeros(reps)
            for l in range(1, j + 1):
                gam += gbar * binom(j, l) * np.sum(hd[l] * w[j - l], axis=1)
            f -= gam[:, None] * new[0]
            new[j] = Jw + f / G[:, None]
        g = Gd / G[None, :]
        lng4[t] = g[4] - 4 * g[1] * g[3] - 3 * g[2] ** 2 + 12 * g[1] ** 2 * g[2] - 6 * g[1] ** 4
        w = new
        out_w.append(w.copy())
    return np.array(out_w), lng4


def c_closed(n, s, gbar, phi):
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    if phi == 0.0:
        return q * q / (2 * n)
    r = phi * q / (1 - phi * p)
    return q * q * (1 + phi) / (2 * n * (1 - phi)) - q * r ** n / (1 - r ** n)


def e_l2_finite(n, s, gbar, phi, t):
    """E l_2(t) from the finite sum of Lemma 6."""
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    om = np.exp(2j * np.pi * np.arange(1, n) / n)
    mu = p + q * om
    tot = 0.0
    for m in range(t):
        tot += phi ** (m + 1) * np.sum(mu ** m * om).real
    return -q * q / n + 2 * q * q / n * tot


def v_basis(n):
    """Orthonormal basis of the zero-sum subspace, as columns."""
    a = np.eye(n) - np.ones((n, n)) / n
    u, sv, _ = np.linalg.svd(a)
    return u[:, : n - 1]


# ---------------------------------------------------------------- the tests

def main():
    rng = np.random.default_rng(2026111401)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    # T1. Lemma 1: lam0/G <= exp(q*eta)
    worst, cnt = 0.0, 0
    for _ in range(2000):
        n = int(rng.integers(1, 13)); p = rng.uniform(0.01, 0.99); q = 1 - p
        sig = rng.uniform(-2.0, 2.0)
        xi = rng.normal(size=(100, n)) * rng.choice([1.0, 3.0])
        w = rng.dirichlet(np.ones(n) * rng.choice([0.2, 1.0, 5.0]), size=100)
        h = np.exp(sig * xi - 0.5 * sig ** 2)
        lhs = 1.0 / (p + q * np.sum(h * w, axis=1))
        eta = abs(sig) * np.abs(xi).max(axis=1) + 0.5 * sig ** 2
        worst = max(worst, float(np.max(lhs / np.exp(q * eta)))); cnt += 100
    report("T1 Lemma 1, lam0/G <= exp(q*eta)", worst <= 1 + 1e-12, f"{cnt} draws, largest left/right = {worst:.6f}")

    # T2. Lemma 2(a): ||J||_V <= beta~ * exp(c4*eta)
    worst, cnt = 0.0, 0
    for _ in range(6000):
        n = int(rng.integers(2, 11)); p = rng.uniform(0.01, 0.99); q = 1 - p
        sig = rng.uniform(-1.5, 1.5)
        xi = rng.normal(size=n) * rng.choice([1.0, 3.0])
        w = rng.dirichlet(np.ones(n) * rng.choice([0.2, 1.0, 5.0]))
        h = np.exp(sig * xi - 0.5 * sig ** 2)
        a = p * np.eye(n)
        for k in range(n):
            a[(k + 1) % n, k] += q * h[k]
        aw = a @ w; g = aw.sum(); w1 = aw / g
        jac = (np.eye(n) - np.outer(w1, np.ones(n))) @ a / g
        vb = v_basis(n)
        norm_v = np.linalg.norm(jac @ vb, 2)
        beta = math.sqrt(max(0.0, 1 - 4 * p * q * math.sin(math.pi / n) ** 2))
        bt = max(beta, 0.5)
        c4 = q + max(1.0, q * (1 + math.sqrt(n)) / bt)
        eta = abs(sig) * np.abs(xi).max() + 0.5 * sig ** 2
        worst = max(worst, norm_v / (bt * math.exp(c4 * eta))); cnt += 1
    report("T2 Lemma 2(a), ||J||_V <= beta~ exp(c4*eta)", worst <= 1 + 1e-12, f"{cnt} draws, largest left/right = {worst:.6f}")

    # T2b. the deterministic part: ||A0/lam0||_V equals beta
    worst = 0.0
    for n in range(2, 14):
        for p in (0.05, 0.3, 0.5, 0.625, 0.9):
            q = 1 - p
            b = p * np.eye(n) + q * np.roll(np.eye(n), 1, axis=0)
            beta = math.sqrt(max(0.0, 1 - 4 * p * q * math.sin(math.pi / n) ** 2))
            worst = max(worst, abs(np.linalg.norm(b @ v_basis(n), 2) - beta))
    report("T2b norm of A0/lam0 on V equals sqrt(1 - 4pq sin^2(pi/n))", worst < 1e-12, f"60 cases, largest difference {worst:.2e}")

    # T3. E exp(c|Z|) = 2 exp(c^2/2) Phi(c) <= exp(c + c^2/2)
    cs = np.linspace(0, 12, 2401)
    phi_c = 0.5 * (1 + np.array([math.erf(c / math.sqrt(2)) for c in cs]))
    ratio = 2 * phi_c / np.exp(cs)          # exact / bound, the factor exp(c^2/2) cancels
    zs = rng.normal(size=2_000_000)
    mc = np.mean(np.exp(0.7 * np.abs(zs))); exact = 2 * math.exp(0.245) * 0.5 * (1 + math.erf(0.7 / math.sqrt(2)))
    report("T3 E exp(c|Z|) <= exp(c + c^2/2)", ratio.max() <= 1 + 1e-12 and abs(mc / exact - 1) < 3e-3,
           f"2401 values of c in [0, 12], largest exact/bound = {ratio.max():.6f}; formula for the exact value against 2e6 draws at c = 0.7: {mc:.5f} vs {exact:.5f}")

    # T4. Lemma 2(b): E exp(a * sum |xi(u)|) <= exp(m (a' + a'^2/2))
    worst = 0.0; lines = []
    for phi in (0.0, 0.5, 0.9):
        for a in (0.02, 0.1):
            for m in (1, 5, 40):
                xi = ar1_paths(rng, 100_000, m, 1, phi)[:, :, 0]
                vals = np.exp(a * np.abs(xi).sum(axis=0))
                ap = a / (1 - phi)
                bound = math.exp(m * (ap + ap * ap / 2))
                worst = max(worst, vals.mean() / bound)
                lines.append(f"phi={phi} a={a} m={m}: {vals.mean():.4g} <= {bound:.4g}")
    report("T4 Lemma 2(b), exponential moment of an AR(1) path", worst <= 1.0, f"18 cases of 1e5 paths, largest estimate/bound = {worst:.4f}")

    # T5. Lemma 2(c): L^r norm of ||J...J||_V against rho_r(sigma)^m
    lines = []; worst = 0.0
    for (n, s, gbar, phi) in ((4, 2 / 3, 0.4, 0.9), (8, 2 / 3, 0.4, 0.5)):
        lam0 = s + gbar; p, q = s / lam0, gbar / lam0
        beta = math.sqrt(1 - 4 * p * q * math.sin(math.pi / n) ** 2); bt = max(beta, 0.5)
        c4 = q + max(1.0, q * (1 + math.sqrt(n)) / bt)
        vb = v_basis(n)
        for sig in (0.002, 0.02):
            for r in (2.0, 4.0):
                rho = bt * math.exp(c4 * sig ** 2 / 2 + n * c4 * sig / (1 - phi) + n * r * c4 ** 2 * sig ** 2 / (2 * (1 - phi) ** 2))
                m = 12; reps = 4000
                xi = ar1_paths(rng, reps, m, n, phi)
                w = np.full((reps, n), 1.0 / n)
                prod = np.broadcast_to(vb, (reps, n, n - 1)).copy()
                for t in range(m):
                    h = np.exp(sig * xi[t] - 0.5 * sig ** 2)
                    def amul(x):      # x has shape (reps, n, cols)
                        return s * x + np.roll(gbar * h[:, :, None] * x, 1, axis=1)
                    aw = s * w + np.roll(gbar * h * w, 1, axis=1)
                    g = aw.sum(axis=1); w1 = aw / g[:, None]
                    ap = amul(prod)
                    prod = (ap - w1[:, :, None] * ap.sum(axis=1, keepdims=True)) / g[:, None, None]
                    w = w1
                norms = np.linalg.norm(prod, 2, axis=(1, 2))
                lr = float(np.mean(norms ** r) ** (1 / r))
                worst = max(worst, lr / rho ** m)
                lines.append(f"n={n} phi={phi} sigma={sig} r={r:.0f}: L^r norm of 12-step product {lr:.4g}; beta^12 {beta**12:.4g}; bound rho^12 {rho**12:.4g}")
    for ln in lines:
        print("     " + ln)
    report("T5 Lemma 2(c), products of the linearised map", worst <= 1.0, f"8 cases of 4,000 paths, largest estimate/bound = {worst:.4g}")

    # T6. the recursion of Lemma 3 and the formula of Lemma 4 against Taylor arithmetic
    worst_w, worst_l, cnt = 0.0, 0.0, 0
    for (n, s, gbar, phi, sig) in ((1, 2 / 3, 0.4, 0.9, 0.3), (2, 0.5, 0.5, 0.9, 0.2), (3, 0.05, 1.0, 0.5, 0.4),
                                   (4, 2 / 3, 0.4, 0.9, 0.0), (4, 2 / 3, 0.4, 0.9, 0.25), (7, 0.9, 0.2, 0.0, -0.3),
                                   (12, 0.3, 0.8, 0.95, 0.15)):
        xi = ar1_paths(rng, 200, 25, n, phi)
        lng, wj = run_jets(xi, s, gbar, sig)
        w_exp, l4_exp = explicit_recursion(xi, s, gbar, sig)
        w_jet = np.moveaxis(wj, -1, 1) * FACT[None, :, None, None]        # (steps+1, 5, reps, n) derivatives
        scale = np.maximum(1.0, np.abs(w_jet))
        worst_w = max(worst_w, float(np.max(np.abs(w_jet - w_exp) / scale)))
        l4_jet = lng[:, :, 4] * 24.0
        worst_l = max(worst_l, float(np.max(np.abs(l4_jet - l4_exp) / np.maximum(1.0, np.abs(l4_jet)))))
        cnt += 1
    report("T6 Lemma 3 recursion (J and f_j, Hermite form of h^(l)) vs Taylor arithmetic", worst_w < 1e-8,
           f"{cnt} parameter sets x 200 paths x 25 steps, orders 0-4, largest relative difference {worst_w:.2e}")
    report("T6b Lemma 4 formula for the fourth derivative of ln G vs Taylor arithmetic", worst_l < 1e-8,
           f"same paths, largest relative difference {worst_l:.2e}")

    # T6c. the Taylor arithmetic itself against finite differences (orders 1 and 2), so that T6 is not circular
    n, s, gbar, phi, sig = 4, 2 / 3, 0.4, 0.9, 0.25
    xi = ar1_paths(rng, 50, 15, n, phi)
    lng, _ = run_jets(xi, s, gbar, sig)
    def lng_plain(sg):
        l, _w = run_jets(xi, s, gbar, sg)
        return l[:, :, 0]
    e = 1e-4
    d1 = (lng_plain(sig + e) - lng_plain(sig - e)) / (2 * e)
    d2 = (lng_plain(sig + e) - 2 * lng_plain(sig) + lng_plain(sig - e)) / e ** 2
    err1 = float(np.max(np.abs(d1 - lng[:, :, 1]))); err2 = float(np.max(np.abs(d2 - 2 * lng[:, :, 2])))
    report("T6c Taylor arithmetic vs finite differences", err1 < 1e-6 and err2 < 1e-5, f"first derivative off by at most {err1:.1e}, second by {err2:.1e}")

    # T7. Lemma 5: l_j(t)(-xi) = (-1)^j l_j(t)(xi)
    worst = 0.0
    for (n, s, gbar, phi) in ((3, 0.4, 0.7, 0.8), (6, 2 / 3, 0.4, 0.9)):
        xi = ar1_paths(rng, 300, 20, n, phi)
        a_, _ = run_jets(xi, s, gbar, 0.0); b_, _ = run_jets(-xi, s, gbar, 0.0)
        for j in range(1, 5):
            worst = max(worst, float(np.max(np.abs(b_[:, :, j] - (-1) ** j * a_[:, :, j]))))
    report("T7 Lemma 5, parity of the coefficients under xi -> -xi", worst < 1e-10, f"2 parameter sets x 300 paths x 20 steps, orders 1-4, largest violation {worst:.1e}")

    # T8. Lemma 6: finite-t sum against the average over paths, and its limit against the closed form
    ok = True; lines = []
    for (n, s, gbar, phi, t) in ((4, 2 / 3, 0.4, 0.9, 3), (4, 2 / 3, 0.4, 0.9, 12), (2, 0.5, 0.5, 0.6, 5), (9, 0.2, 0.9, 0.95, 20)):
        xi = ar1_paths(rng, 400_000, t + 1, n, phi)
        lng, _ = run_jets(xi, s, gbar, 0.0)
        l2 = 2 * lng[t, :, 2]
        mean, se = float(l2.mean()), float(l2.std(ddof=1) / math.sqrt(len(l2)))
        exact = e_l2_finite(n, s, gbar, phi, t)
        z = (mean - exact) / se
        ok &= abs(z) < 4
        lines.append(f"n={n} phi={phi} t={t}: paths {mean:+.6f} +- {se:.6f}; finite sum {exact:+.6f} (z = {z:+.2f}); limit -2C = {-2*c_closed(n, s, gbar, phi):+.6f}")
    for ln in lines:
        print("     " + ln)
    report("T8 Lemma 6, E l_2(t) from the finite sum vs 4e5 paths", ok, "4 cases, all within 4 standard errors")
    worst, worst_tr, cnt = 0.0, 0.0, 0
    for n in (1, 2, 3, 4, 5, 8, 16, 33, 64):
        for s in (1e-3, 0.2, 2 / 3, 0.98):
            for gbar in (0.05, 0.4, 2.5):
                for phi in (0.0, 0.01, 0.5, 0.9, 0.99):
                    lam0 = s + gbar; q = gbar / lam0
                    big = 4000 if phi > 0.95 else 600
                    lim = e_l2_finite(n, s, gbar, phi, big)
                    worst = max(worst, abs(lim + 2 * c_closed(n, s, gbar, phi)))
                    for t in (0, 1, 2, 5, 17):
                        gap = abs(e_l2_finite(n, s, gbar, phi, t) + 2 * c_closed(n, s, gbar, phi))
                        bound = 2 * q * q * phi ** (t + 1) / (1 - phi)
                        worst_tr = max(worst_tr, gap - bound)
                    cnt += 1
    report("T8b Lemma 6, limit of the finite sum equals -2 C_n", worst < 1e-9, f"{cnt} parameter sets, largest difference {worst:.2e}")
    report("T8c Lemma 6, transient within 2 q^2 phi^(t+1)/(1-phi)", worst_tr <= 1e-12, f"{cnt} x 5 times, largest excess over the bound {worst_tr:.2e}")

    # T9. Lemma 4: the mean absolute fourth derivative of ln G_t settles to a plateau and does not grow with t.
    # Times are multiples of the relaxation time 1/(1 - beta) of the composition. (A first version of this
    # test compared t = 80 with t = 320 and failed for 16 stages, where the relaxation time is 55 steps and
    # the value was still rising at t = 80: 8.86 against 11.25. The times were then tied to the relaxation
    # time. Tolerance: the last two values agree within 10% or 4 standard errors.)
    print("     E|d^4 ln G_t / d sigma^4| at 5, 10, 20 and 40 relaxation times (10,000 paths each):")
    ok = True
    for (n, s, gbar, phi) in ((4, 2 / 3, 0.4, 0.9), (16, 2 / 3, 0.4, 0.9), (4, 0.05, 31 / 30, 0.0)):
        lam0 = s + gbar; p_, q_ = s / lam0, gbar / lam0
        beta = math.sqrt(1 - 4 * p_ * q_ * math.sin(math.pi / n) ** 2)
        relax = 1.0 / (1.0 - beta)
        marks = [int(math.ceil(k * relax)) for k in (5, 10, 20, 40)]
        for sig in (0.0, 0.05):
            reps = 10_000
            xi_t = rng.normal(size=(reps, n)); c_ = math.sqrt(1 - phi * phi)
            w = np.zeros((reps, n, ORDER + 1)); w[:, :, 0] = 1.0 / n
            got = {}
            for t in range(marks[-1] + 1):
                h = h_jet(xi_t, sig)
                new = s * w + np.roll(gbar * jmul(h, w), 1, axis=1)
                g = new.sum(axis=1)
                if t in marks:
                    v = np.abs(24 * jlog(g)[:, 4])
                    got[t] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(reps)))
                w = jmul(new, jrecip(g)[:, None, :])
                xi_t = phi * xi_t + c_ * rng.normal(size=(reps, n))
            (m3, e3), (m4, e4) = got[marks[2]], got[marks[3]]
            close = abs(m4 - m3) <= max(0.10 * m3, 4 * math.hypot(e3, e4))
            ok &= close
            txt = "; ".join(f"t={t} {got[t][0]:.4g} +- {got[t][1]:.2g}" for t in marks)
            print(f"       n={n} s={s:.3g} phi={phi} sigma={sig} (relaxation time {relax:.1f} steps): {txt}")
    report("T9 Lemma 4, bounded in t", ok, "6 cases; the values at 20 and 40 relaxation times agree within 10% or 4 standard errors")

    # T10. the identity for (ln G)'''' and the two-stage comparison, symbolically
    try:
        import sympy as sp
        x = sp.symbols("x"); G = sp.Function("G")(x)
        g = [sp.diff(G, x, i) / G for i in range(5)]
        expr = g[4] - 4 * g[1] * g[3] - 3 * g[2] ** 2 + 12 * g[1] ** 2 * g[2] - 6 * g[1] ** 4
        d = sp.simplify(sp.diff(sp.log(G), x, 4) - expr)
        qq, ph = sp.symbols("q phi", positive=True)
        pp = 1 - qq; r = ph * qq / (1 - ph * pp)
        c1 = qq ** 2 * (1 + ph) / (2 * (1 - ph)) - qq * r / (1 - r)
        c2 = qq ** 2 * (1 + ph) / (4 * (1 - ph)) - qq * r ** 2 / (1 - r ** 2)
        d2 = sp.simplify(c2 - c1 - qq ** 2 * (ph * (3 - 2 * qq) - 1) / (4 * (1 - ph * (1 - 2 * qq))))
        d1 = sp.simplify(c1 - qq ** 2 / 2)
        report("T10 symbolic identities", d == 0 and d2 == 0 and d1 == 0,
               "fourth derivative of ln G in terms of G^(i)/G; C_1 = q^2/2 for every phi; C_2 - C_1 = q^2[phi(3-2q)-1] / (4[1-phi(1-2q)])")
    except ImportError:
        print("[skip] T10: sympy not available")

    print()
    print("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
