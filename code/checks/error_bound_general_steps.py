"""Numerical tests of the new steps of the proof in `notes/error-bound-general.md`.

Claim checked: for a general primitive projection matrix with lognormal AR(1) multipliers on some of its
entries (entries may share a driver), each inequality and identity of that proof holds on the cases tried.
These tests cannot prove the lemmas. Tolerances, stated beforehand:
  - an inequality counts as violated if left > right * (1 + 1e-12);
  - identities between two exact evaluations agree to 1e-8 relative;
  - a Monte Carlo mean lies within 4 standard errors of its exact value;
  - simulated cost divided by C*sigma^2 is within 3% of one at sigma = 0.05 (the fourth-order term is not
    known for these matrices, so this is a loose end-to-end check and not a test of the bound).

Run: python3 -I checks/error_bound_general_steps.py
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec = importlib.util.spec_from_file_location("ebs", HERE / "error_bound_steps.py")
ebs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebs)
ORDER, FACT = ebs.ORDER, ebs.FACT


def perron(a):
    ev, vr = np.linalg.eig(a)
    k = int(np.argmax(ev.real)); lam = float(ev[k].real)
    v = np.abs(vr[:, k].real); v /= v.sum()
    el, vl = np.linalg.eig(a.T)
    k2 = int(np.argmax(el.real)); w = np.abs(vl[:, k2].real); w /= w @ v
    return lam, v, w


def random_model(rng, d=None, ring=False):
    """A primitive matrix, a set of fluctuating entries, their drivers, relative sizes and autocorrelations."""
    d = d or int(rng.integers(2, 6))
    if ring:
        s = rng.uniform(0.0, 0.9, d)
        if d > 1:
            s[rng.integers(d)] = rng.uniform(0.2, 0.9)          # at least one stage with staying
            zero = rng.random(d) < 0.3; zero[np.argmax(s)] = False; s = np.where(zero, 0.0, s)
        a = np.diag(s)
        for k in range(d):
            a[(k + 1) % d, k] += rng.uniform(0.1, 1.5)
    else:
        a = rng.uniform(0.05, 1.0, (d, d)) * (rng.random((d, d)) < 0.6)
        a += np.diag(rng.uniform(0.05, 0.6, d)) + 0.1 * np.roll(np.eye(d), 1, axis=0)
    entries = [tuple(e) for e in np.argwhere(a > 0)]
    k = int(rng.integers(1, min(len(entries), 5) + 1))
    chosen = [entries[i] for i in rng.choice(len(entries), size=k, replace=False)]
    n_drv = int(rng.integers(1, k + 1))
    driver = [int(i % n_drv) for i in rng.permutation(k)]
    kappa = rng.uniform(0.3, 1.0, k)
    phi = rng.uniform(-0.6, 0.9, n_drv)
    return a, chosen, driver, kappa, phi


def coefficient(a, chosen, driver, kappa, phi):
    """C of Theorem G."""
    d = a.shape[0]
    lam, v, w = perron(a)
    m = a / lam; q = np.eye(d) - np.outer(v, w); t = m - np.outer(v, w)
    bs = []
    for (i, j) in chosen:
        b = np.zeros((d, d)); b[i, j] = m[i, j]; bs.append(b)
    c = np.array([w @ b @ v for b in bs])
    tot = 0.0
    for x in range(len(chosen)):
        for y in range(len(chosen)):
            if driver[x] != driver[y]:
                continue
            f = phi[driver[x]]
            lag = w @ bs[x] @ np.linalg.solve(np.eye(d) - f * t, q @ bs[y] @ v)
            tot += kappa[x] * kappa[y] * (0.5 * c[x] * c[y] - f * lag)
    return tot


def drivers_paths(rng, reps, steps, phi):
    out = np.empty((steps, reps, len(phi)))
    out[0] = rng.normal(size=(reps, len(phi)))
    c = np.sqrt(1 - phi ** 2)
    for t in range(1, steps):
        out[t] = phi[None, :] * out[t - 1] + c[None, :] * rng.normal(size=(reps, len(phi)))
    return out


def run_jets_general(a, chosen, driver, kappa, xi, sigma0):
    """Exact recursion with Taylor arithmetic, state normalised by the left eigenvector. Returns jets of ln G_t and of y(t)."""
    steps, reps, _ = xi.shape
    d = a.shape[0]
    lam, v, w = perron(a)
    y = np.zeros((steps + 1, reps, d, ORDER + 1)); y[0, :, :, 0] = v[None, :]
    lng = np.zeros((steps, reps, ORDER + 1))
    fixed = a.copy()
    for (i, j) in chosen:
        fixed[i, j] = 0.0
    for t in range(steps):
        new = np.einsum("ij,rjk->rik", fixed, y[t])
        for x, (i, j) in enumerate(chosen):
            e = np.zeros((reps, ORDER + 1))
            z = xi[t][:, driver[x]]
            e[:, 0] = sigma0 * kappa[x] * z - 0.5 * (sigma0 * kappa[x]) ** 2
            e[:, 1] = kappa[x] * z - sigma0 * kappa[x] ** 2
            e[:, 2] = -0.5 * kappa[x] ** 2
            new[:, i, :] += a[i, j] * ebs.jmul(ebs.jexp(e), y[t][:, j, :])
        g = np.einsum("i,rik->rk", w, new)
        lng[t] = ebs.jlog(g)
        y[t + 1] = ebs.jmul(new, ebs.jrecip(g)[:, None, :])
    return lng, y


def explicit_recursion_general(a, chosen, driver, kappa, xi, sigma):
    """Derivatives y_j(t) from the recursion in the proof: y_j(t+1) = J(t) y_j(t) + f_j(t)."""
    steps, reps, _ = xi.shape
    d = a.shape[0]
    lam, v, w = perron(a)
    y = np.zeros((ORDER + 1, reps, d)); y[0] = v[None, :]
    out = [y.copy()]
    lng4 = np.zeros((steps, reps))
    binom = math.comb
    for t in range(steps):
        # A^(l)(t) as arrays (reps, d, d), l = 0..4
        al = np.zeros((ORDER + 1, reps, d, d)); al[0] = a[None, :, :]
        for x, (i, j) in enumerate(chosen):
            z = xi[t][:, driver[x]]
            xx = z - sigma * kappa[x]
            h = np.exp(sigma * kappa[x] * z - 0.5 * (sigma * kappa[x]) ** 2)
            he = [np.ones_like(xx), xx, xx * xx - 1, xx ** 3 - 3 * xx, xx ** 4 - 6 * xx * xx + 3]
            for l in range(ORDER + 1):
                al[l][:, i, j] = a[i, j] * kappa[x] ** l * he[l] * h
        ay = lambda l, vec: np.einsum("rij,rj->ri", al[l], vec)
        G = np.einsum("i,ri->r", w, ay(0, y[0]))
        Gd = np.zeros((ORDER + 1, reps)); Gd[0] = G
        for i in range(1, ORDER + 1):
            for l in range(i + 1):
                Gd[i] += binom(i, l) * np.einsum("i,ri->r", w, ay(l, y[i - l]))
        new = np.zeros_like(y)
        new[0] = ay(0, y[0]) / G[:, None]
        for j in range(1, ORDER + 1):
            ayj = ay(0, y[j])
            Jy = (ayj - new[0] * np.einsum("i,ri->r", w, ayj)[:, None]) / G[:, None]
            f = np.zeros((reps, d))
            for i in range(1, j + 1):
                f += binom(j, i) * ay(i, y[j - i])
            for i in range(1, j):
                f -= binom(j, i) * Gd[i][:, None] * new[j - i]
            gam = np.zeros(reps)
            for l in range(1, j + 1):
                gam += binom(j, l) * np.einsum("i,ri->r", w, ay(l, y[j - l]))
            f -= gam[:, None] * new[0]
            new[j] = Jy + f / G[:, None]
        g = Gd / G[None, :]
        lng4[t] = g[4] - 4 * g[1] * g[3] - 3 * g[2] ** 2 + 12 * g[1] ** 2 * g[2] - 6 * g[1] ** 4
        y = new
        out.append(y.copy())
    return np.array(out), lng4


def e_l2_finite(a, chosen, driver, kappa, phi, t):
    d = a.shape[0]
    lam, v, w = perron(a)
    m = a / lam; q = np.eye(d) - np.outer(v, w)
    bs = []
    for (i, j) in chosen:
        b = np.zeros((d, d)); b[i, j] = m[i, j]; bs.append(b)
    c = np.array([w @ b @ v for b in bs])
    tot = 0.0
    mp = np.eye(d)
    lagsum = np.zeros((len(chosen), len(chosen)))
    for mm in range(t):
        for x in range(len(chosen)):
            for y in range(len(chosen)):
                if driver[x] == driver[y]:
                    lagsum[x, y] += phi[driver[x]] ** (mm + 1) * (w @ bs[x] @ mp @ q @ bs[y] @ v)
        mp = mp @ m
    for x in range(len(chosen)):
        for y in range(len(chosen)):
            if driver[x] == driver[y]:
                tot += kappa[x] * kappa[y] * (2 * lagsum[x, y] - c[x] * c[y])
    return tot


def simulate_cost(a, chosen, driver, kappa, phi, sigma, reps, steps, burn, rng):
    """Direct simulation with the elasticity control variate. Returns (cost, s.e.)."""
    d = a.shape[0]
    lam, v, w = perron(a)
    c = np.array([a[i, j] * w[i] * v[j] / lam for (i, j) in chosen])
    xi = rng.normal(size=(reps, len(phi)))
    cc = np.sqrt(1 - phi ** 2)
    x = np.tile(v, (reps, 1))
    tot = np.zeros(reps)
    fixed = a.copy()
    for (i, j) in chosen:
        fixed[i, j] = 0.0
    for t in range(burn + steps):
        new = x @ fixed.T
        cv = np.zeros(reps)
        for k, (i, j) in enumerate(chosen):
            h = np.exp(sigma * kappa[k] * xi[:, driver[k]] - 0.5 * (sigma * kappa[k]) ** 2)
            new[:, i] += a[i, j] * h * x[:, j]
            cv += c[k] * (h - 1.0)
        s_ = new.sum(axis=1)
        x = new / s_[:, None]
        if t >= burn:
            tot += np.log(s_) - cv
        xi = phi[None, :] * xi + cc[None, :] * rng.normal(size=(reps, len(phi)))
    rate = tot / steps
    return math.log(lam) - float(rate.mean()), float(rate.std(ddof=1) / math.sqrt(reps))


def main():
    rng = np.random.default_rng(2026111403)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    # G1. Lemma 1': lam0 / G <= exp(eta)
    worst, cnt = 0.0, 0
    for trial in range(3000):
        a, chosen, driver, kappa, phi = random_model(rng, ring=(trial % 2 == 0))
        d = a.shape[0]; lam, v, w = perron(a)
        sig = rng.uniform(-2, 2)
        z = rng.normal(size=len(phi)) * rng.choice([1.0, 3.0])
        y = rng.dirichlet(np.ones(d) * rng.choice([0.2, 1.0, 5.0])); y = y / (w @ y)
        at = a.copy()
        for k, (i, j) in enumerate(chosen):
            at[i, j] = a[i, j] * math.exp(sig * kappa[k] * z[driver[k]] - 0.5 * (sig * kappa[k]) ** 2)
        g = w @ at @ y
        eta = abs(sig) * kappa.max() * np.abs(z).max() + 0.5 * sig ** 2 * kappa.max() ** 2
        worst = max(worst, (lam / g) / math.exp(eta)); cnt += 1
    report("G1 Lemma 1', lam0/G <= exp(eta), with no lower bound on staying", worst <= 1 + 1e-12,
           f"{cnt} random models (half of them rings with some stages without staying), largest left/right = {worst:.6f}")

    # G2. the adapted norm and Lemma 2'(a)
    worst_m, worst_lo, worst_j, cnt = 0.0, 0.0, -1e300, 0
    for trial in range(400):
        a, chosen, driver, kappa, phi = random_model(rng, ring=(trial % 2 == 0))
        d = a.shape[0]
        if d < 2:
            continue
        lam, v, w = perron(a); m = a / lam
        ev = np.sort(np.abs(np.linalg.eigvals(m)))[::-1]
        rho2 = ev[1]
        beta = max(0.5 * (1 + rho2), 0.5)
        # basis of V = {z : w'z = 0}
        u_, s_, vt = np.linalg.svd(w[None, :]); vb = vt[1:].T
        # scaled powers (T/beta)^j with T = M - v w'; on V they equal beta^-j M^j, and they decay, so nothing overflows
        tb = (m - np.outer(v, w)) / beta
        powers = [np.eye(d)]
        while True:
            powers.append(powers[-1] @ tb)
            if np.linalg.norm(powers[-1] @ vb, 2) < 1e-13 or len(powers) > 20000:
                break
        def star(zz):
            return max(np.linalg.norm(pw @ zz) for pw in powers)
        c_star = max(np.linalg.norm(pw @ vb, 2) for pw in powers)
        b_bar = sum(m[i, j] for (i, j) in chosen)
        c5 = c_star * (1 + np.linalg.norm(w) / w.min()) * b_bar
        c4 = 1 + max(1.0, c5 / beta)
        for rep in range(25):
            zz = vb @ rng.normal(size=d - 1)
            worst_m = max(worst_m, star(m @ zz) / (beta * star(zz)))
            worst_lo = max(worst_lo, np.linalg.norm(zz) / star(zz), star(zz) / (c_star * np.linalg.norm(zz)))
            sig = rng.uniform(-1.5, 1.5); zdr = rng.normal(size=len(phi)) * rng.choice([1.0, 3.0])
            y = rng.dirichlet(np.ones(d)); y = y / (w @ y)
            at = a.copy()
            for k, (i, j) in enumerate(chosen):
                at[i, j] = a[i, j] * math.exp(sig * kappa[k] * zdr[driver[k]] - 0.5 * (sig * kappa[k]) ** 2)
            ay = at @ y; g = w @ ay; y1 = ay / g
            jz = (np.eye(d) - np.outer(y1, w)) @ at @ zz / g
            eta = abs(sig) * kappa.max() * np.abs(zdr).max() + 0.5 * sig ** 2 * kappa.max() ** 2
            worst_j = max(worst_j, math.log(star(jz)) - math.log(star(zz)) - math.log(beta) - c4 * eta)   # log of left/right
            cnt += 1
    report("G2 adapted norm: ||Mz||* <= beta ||z||*", worst_m <= 1 + 1e-9, f"{cnt} vectors, largest left/right = {worst_m:.6f}")
    report("G2b adapted norm equivalent to the Euclidean one", worst_lo <= 1 + 1e-9, f"largest of |z|/||z||* and ||z||*/(c*|z|) = {worst_lo:.6f}")
    report("G2c Lemma 2'(a), ||Jz||* <= beta exp(c4 eta) ||z||*", worst_j <= 1e-9, f"{cnt} draws, largest log(left/right) = {worst_j:.4f}")

    # G3. recursion of Lemma 3' and formula of Lemma 4' against Taylor arithmetic
    worst_y, worst_l, cnt = 0.0, 0.0, 0
    for trial in range(8):
        a, chosen, driver, kappa, phi = random_model(rng, ring=(trial % 2 == 0))
        sig = float(rng.uniform(-0.3, 0.4)) if trial else 0.0
        xi = drivers_paths(rng, 100, 20, phi)
        lng, yj = run_jets_general(a, chosen, driver, kappa, xi, sig)
        y_exp, l4_exp = explicit_recursion_general(a, chosen, driver, kappa, xi, sig)
        y_jet = np.moveaxis(yj, -1, 1) * FACT[None, :, None, None]
        worst_y = max(worst_y, float(np.max(np.abs(y_jet - y_exp) / np.maximum(1.0, np.abs(y_jet)))))
        l4 = 24 * lng[:, :, 4]
        worst_l = max(worst_l, float(np.max(np.abs(l4 - l4_exp) / np.maximum(1.0, np.abs(l4)))))
        cnt += 1
    report("G3 Lemma 3' recursion vs Taylor arithmetic", worst_y < 1e-8, f"{cnt} random models x 100 paths x 20 steps, orders 0-4, largest relative difference {worst_y:.2e}")
    report("G3b Lemma 4' fourth derivative of ln G vs Taylor arithmetic", worst_l < 1e-8, f"same paths, largest relative difference {worst_l:.2e}")

    # G4. parity with shared drivers
    worst = 0.0
    for trial in range(4):
        a, chosen, driver, kappa, phi = random_model(rng, ring=(trial % 2 == 0))
        xi = drivers_paths(rng, 100, 15, phi)
        p_, _ = run_jets_general(a, chosen, driver, kappa, xi, 0.0)
        m_, _ = run_jets_general(a, chosen, driver, kappa, -xi, 0.0)
        for j in range(1, 5):
            worst = max(worst, float(np.max(np.abs(m_[:, :, j] - (-1) ** j * p_[:, :, j]))))
    report("G4 Lemma 5', parity", worst < 1e-9, f"4 random models, orders 1-4, largest violation {worst:.1e}")

    # G5. Lemma 6': E l_2(t) and its limit
    ok = True
    for trial in range(4):
        a, chosen, driver, kappa, phi = random_model(rng, d=3 + trial % 2, ring=(trial % 2 == 0))
        t = 6
        xi = drivers_paths(rng, 300_000, t + 1, phi)
        lng, _ = run_jets_general(a, chosen, driver, kappa, xi, 0.0)
        l2 = 2 * lng[t, :, 2]
        mean, se = float(l2.mean()), float(l2.std(ddof=1) / math.sqrt(len(l2)))
        exact = e_l2_finite(a, chosen, driver, kappa, phi, t)
        z = (mean - exact) / se
        ok &= abs(z) < 4
        print(f"     model {trial} ({len(chosen)} fluctuating entries, {len(phi)} drivers): paths {mean:+.6f} +- {se:.6f}; finite sum {exact:+.6f}; z = {z:+.2f}")
    report("G5 Lemma 6', E l_2(t) from the finite sum vs 3e5 paths", ok, "4 random models, all within 4 standard errors")
    worst, cnt = 0.0, 0
    for trial in range(400):
        a, chosen, driver, kappa, phi = random_model(rng, ring=(trial % 2 == 0))
        lim = e_l2_finite(a, chosen, driver, kappa, phi, 3000)
        cval = coefficient(a, chosen, driver, kappa, phi)
        worst = max(worst, abs(lim + 2 * cval) / max(1e-6, abs(cval))); cnt += 1
    report("G5b Lemma 6', the limit equals -2C", worst < 1e-6, f"{cnt} random models, largest relative difference {worst:.2e}")

    # G6. end to end: simulated cost against C sigma^2 at small noise
    ok = True
    for trial in range(4):
        a, chosen, driver, kappa, phi = random_model(rng, d=3 + trial % 2, ring=(trial % 2 == 1))
        cval = coefficient(a, chosen, driver, kappa, phi)
        cost, se = simulate_cost(a, chosen, driver, kappa, phi, 0.05, 300, 60_000, 1_000, rng)
        ratio = cost / (cval * 0.05 ** 2)
        good = abs(ratio - 1) < max(0.03, 4 * se / abs(cval * 0.05 ** 2))
        ok &= good
        print(f"     model {trial}: C = {cval:+.5f}; simulated cost / (C sigma^2) at sigma = 0.05: {ratio:.4f} (s.e. {se / abs(cval * 0.05 ** 2):.4f})")
    report("G6 simulated cost against C sigma^2", ok, "4 random models at sigma = 0.05")

    print()
    print("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
