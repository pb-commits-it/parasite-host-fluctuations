"""Does host scarcity WASTE the infective stages that are waiting, or only DELAY them?

The ring of the manuscript multiplies the transfer out of stage j by the host's relative abundance
h(t) and leaves the staying rate of stage j alone ("gain-only": scarcity wastes the leavers).  The
alternative ("contact-limited"): an individual waiting in stage j is found by a host with probability
c h(t) per step, dies with probability mu, and otherwise waits, so the staying rate is
s_j - c (h(t) - 1): scarcity delays the transfer.  The two have the same mean matrix.

Closed form derived in notes/contact-limited-transfer.md (second order, host with geometric memory,
theta = phi/(1 - phi), any n and stays).  With tau_k the stays of a line of descent, T their sum,
q_k = 1/(1 + tau_k/theta), R = prod q_k, and
        d = (lambda - s_j - c)/(lambda - s_j)
(at the threshold lambda = 1: d = mu/(mu + c), the fraction of the waiting stage that dies waiting),
the loss per generation is

        Gamma_contact = sigma^2 { d^2 (1 + 2 theta)/(2T) - d [R - (1 - d) q_j]/(1 - R) },

which is the manuscript's Gamma at d = 1, sigma^2 d^2/(2T) without memory, and tends to
sigma^2 d (1 - d) q_j, not to zero, as a steady stage elsewhere becomes long.

Checks (tolerances fixed before the first run):
  T1  closed form = the general second-order formula summed over lags (the lag sum is written here
      from the formula C'' of notes/error-bound-general.md, not taken from engine/theory.py), on 400
      random rings, n = 1..5; relative difference <= 1e-8.
  T2  exact growth rate by simulation in a two-state Markov environment h = 1 +- delta (bounded, so the
      staying rate stays a probability), delta = 0.1 and 0.2: the loss per step is delta^2 C within
      4 standard errors + 6% (the next order is delta^4), for four rings, contact-limited and gain-only
      in the SAME environments.
  T3  Gamma_contact > 0 for 0 < d <= 1 on 20,000 random rings (it is d times a function linear in d
      that is positive at d = 0 and d = 1; the check is of the algebra).
  T4  the floor: with the steady stage's stay 1e7, Gamma_contact/sigma^2 = d (1 - d) q_j within 1e-4.
  T5  the reading of the floor: for a host frozen during the wait, the success probability of a
      waiting individual is p h/(p h + 1 - p), p = 1 - d; its arithmetic mean over hosts of variance
      sigma^2 is 1 - p (1 - p) sigma^2 + O(sigma^4): computed by quadrature for a lognormal h at
      sigma = 0.05, within 2%.
  Reported, no criterion: where the contact-limited loss EXCEEDS the gain-only loss; the two losses
  against the length of a steady stage.
  T6  (added after the first run, which passed T1 to T5; criterion fixed before T6 was run)
      CONTINUOUS TIME.  X' = -(c h(t) + mu) X + b a2 Y, Y' = c h(t) X - a2 Y (contact-limited) against
      X' = -(c + mu) X + b a2 Y, Y' = c h(t) X - a2 Y (gain-only), b = (c + mu)/c, h = 1 +- delta a
      random telegraph signal with switching rate nu (correlation exp(-t/theta), theta = 1/(2 nu)).
      Conjecture (the discrete proof carried over, not written out): the loss per generation is
          delta^2 { d^2 theta/T - d [R - (1 - d) q_X]/(1 - R) },  q_k = 1/(1 + tau_k/theta),
      tau_X = 1/(c + mu), tau_Y = 1/a2, T their sum, d = mu/(c + mu); at d = 1 the gain-only value.
      Exact growth by products of matrix exponentials over the random holding times; four parameter
      sets, delta = 0.15; both models within 4 standard errors + 6%.

Run: python3 -I checks/contact_limited_transfer.py   (a few minutes)
"""
from __future__ import annotations

import math
import sys

import numpy as np

FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def closed_form(taus, j, d, phi):
    """Gamma/sigma^2 per generation for a contact-limited transfer out of stage j."""
    T = float(np.sum(taus))
    if phi == 0:
        return d * d / (2 * T)
    th = phi / (1 - phi)
    q = 1 / (1 + np.asarray(taus, float) / th)
    R = float(np.prod(q))
    return d * d * (1 + 2 * th) / (2 * T) - d * (R - (1 - d) * q[j]) / (1 - R)


def ring(taus, lam=1.0):
    """Mean matrix of a ring with lineage stays taus and growth factor lam; transfers set so that it is exact."""
    taus = np.asarray(taus, float)
    n = len(taus)
    s = lam * (1 - 1 / taus)
    g = lam / taus                       # with these, w = v-weights are uniform up to scaling: any split works
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] += s[k]
        A[(k + 1) % n, k] += g[k]
    return A, s, g


def lag_sum(A, j, c, phi, s_j):
    """C (per step) from the general formula for the perturbation eps(t) [g_j E_{j+1,j} - c E_{jj}]."""
    n = A.shape[0]
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(vals.real))
    lam = float(vals[i].real)
    v = np.abs(vecs[:, i].real)
    vl, wl = np.linalg.eig(A.T)
    w = np.abs(wl[:, int(np.argmax(vl.real))].real)
    v /= v.sum()
    w /= w @ v
    B = np.zeros((n, n))
    if n > 1:
        B[(j + 1) % n, j] += A[(j + 1) % n, j]
        B[j, j] -= c
    else:
        # one stage: the transfer returns to the same stage; A[0,0] = s + g, perturbation (g - c) eps
        B[0, 0] = A[0, 0] - s_j - c
    B /= lam
    Q = np.eye(n) - np.outer(v, w)
    M = A / lam
    cc = float(w @ B @ v)
    if phi == 0:
        return cc * cc / 2, lam
    # sum_{m>=0} phi^{m+1} w B M^m Q B v  =  phi w B (I - phi M)^{-1} Q B v   (M Q = Q M, spectral radius of M Q < 1)
    X = np.linalg.solve(np.eye(n) - phi * (M @ Q), Q @ B @ v)
    return cc * cc / 2 - phi * float(w @ B @ X), lam


def simulate(taus, d, phi, delta, seed, M=600, steps=300_000, burn=3000):
    """Loss per step (minus the growth rate; the mean matrix has growth zero), contact-limited and gain-only."""
    rng = np.random.default_rng(seed)
    A, s, g = ring(taus)
    n = len(taus)
    j = 0
    c = (1 - s[j]) * (1 - d)
    assert s[j] - c * delta >= 0, "staying rate would go negative"
    p_sw = (1 - phi) / 2
    state = rng.integers(0, 2, M) * 2 - 1           # +-1
    xc = np.ones((M, n)) / n
    xg = np.ones((M, n)) / n
    accc = np.zeros(M)
    accg = np.zeros(M)
    for t in range(burn + steps):
        flip = rng.random(M) < p_sw
        state = np.where(flip, -state, state)
        eps = delta * state
        # gain-only
        new = xg * s
        new[:, 1:] += xg[:, :-1] * g[:-1]
        new[:, 0] += xg[:, -1] * g[-1]
        add = xg[:, j] * g[j] * eps
        new[:, (j + 1) % n] += add
        tot = new.sum(axis=1)
        xg = new / tot[:, None]
        if t >= burn:
            accg += np.log(tot)
        # contact-limited
        new = xc * s
        new[:, 1:] += xc[:, :-1] * g[:-1]
        new[:, 0] += xc[:, -1] * g[-1]
        new[:, (j + 1) % n] += xc[:, j] * g[j] * eps
        new[:, j] -= xc[:, j] * c * eps
        tot = new.sum(axis=1)
        xc = new / tot[:, None]
        if t >= burn:
            accc += np.log(tot)
    rc = -accc / steps
    rg = -accg / steps
    return rc.mean(), rc.std(ddof=1) / math.sqrt(M), rg.mean(), rg.std(ddof=1) / math.sqrt(M)


def ct_closed(c, mu, a2, theta, d=None):
    tx, ty = 1 / (c + mu), 1 / a2
    T = tx + ty
    d = mu / (c + mu) if d is None else d
    qx, qy = 1 / (1 + tx / theta), 1 / (1 + ty / theta)
    R = qx * qy
    return d * d * theta / T - d * (R - (1 - d) * qx) / (1 - R), T


def ct_simulate(c, mu, a2, nu, delta, seed, contact, M=400, n_intervals=150_000, burn=500):
    """Minus the growth rate per unit time for a telegraph host, by exact propagation over each holding time."""
    rng = np.random.default_rng(seed)
    b = (c + mu) / c
    mats = []
    for sgn in (+1, -1):
        h = 1 + sgn * delta
        if contact:
            A = np.array([[-(c * h + mu), b * a2], [c * h, -a2]])
        else:
            A = np.array([[-(c + mu), b * a2], [c * h, -a2]])
        lam, V = np.linalg.eig(A)
        mats.append((lam.real, V.real, np.linalg.inv(V.real)))
    x = np.ones((M, 2)) / 2
    logs = np.zeros(M)
    time_tot = np.zeros(M)
    start = rng.integers(0, 2, M)
    for k in range(burn + n_intervals):
        t = rng.exponential(1 / nu, M)
        st = (start + k) % 2
        new = np.empty_like(x)
        for sidx in (0, 1):
            m = st == sidx
            if not m.any():
                continue
            lam, V, Vi = mats[sidx]
            y = x[m] @ Vi.T
            y = y * np.exp(np.outer(t[m], lam))
            new[m] = y @ V.T
        tot = new.sum(axis=1)
        x = new / tot[:, None]
        if k >= burn:
            logs += np.log(tot)
            time_tot += t
    r = -logs / time_tot
    return r.mean(), r.std(ddof=1) / math.sqrt(M)


def main():
    rng = np.random.default_rng(20261114)

    # ------------------------------------------------------------------ T1
    worst = 0.0
    worst_case = None
    for it in range(400):
        n = int(rng.integers(1, 6))
        taus = np.exp(rng.uniform(math.log(1.2), math.log(200), n))
        phi = float(rng.choice([0.0, rng.uniform(0.05, 0.995)]))
        lam = float(rng.uniform(0.9, 1.1))
        A, s, g = ring(taus, lam)
        j = int(rng.integers(0, n))
        d = float(rng.uniform(0.0, 1.0))
        c = (lam - s[j]) * (1 - d)
        C, lam_num = lag_sum(A, j, c, phi, s[j])
        T = float(taus.sum())
        cf = closed_form(taus, j, d, phi) / T
        rel = abs(C - cf) / max(abs(cf), 1e-300)
        if rel > worst:
            worst, worst_case = rel, (n, np.round(taus, 3), round(phi, 4), round(lam, 4), round(d, 4))
    report("T1 closed form = the general formula summed over lags (400 random rings)", worst <= 1e-8,
           f"largest relative difference {worst:.2e} at {worst_case}")

    # ------------------------------------------------------------------ T2
    cases = [((4.0, 6.0), 0.5, 0.8), ((3.0, 40.0), 0.3, 0.9), ((5.0, 2.0, 8.0), 0.7, 0.5), ((2.0, 100.0), 0.9, 0.99)]
    ok = True
    lines = []
    for idx, (taus, d, phi) in enumerate(cases):
        T = sum(taus)
        cf_c = closed_form(taus, 0, d, phi) / T
        cf_g = closed_form(taus, 0, 1.0, phi) / T
        for delta in (0.1, 0.2):
            steps = 300_000 if phi < 0.95 else 1_500_000
            rc, sec, rg, seg = simulate(taus, d, phi, delta, 7000 + 10 * idx + int(100 * delta), steps=steps)
            pc, pg = delta * delta * cf_c, delta * delta * cf_g
            okc = abs(rc - pc) <= 4 * sec + 0.06 * abs(pc)
            okg = abs(rg - pg) <= 4 * seg + 0.06 * abs(pg)
            ok = ok and okc and okg
            lines.append(f"      stays {taus}, d {d}, memory {phi}, delta {delta}: contact {rc:.4e} +- {sec:.1e} (formula {pc:.4e}) {'ok' if okc else 'OUT'}; "
                         f"gain-only {rg:.4e} +- {seg:.1e} (formula {pg:.4e}) {'ok' if okg else 'OUT'}; ratio simulated {rc / rg:.3f}, formula {pc / pg:.3f}")
    report("T2 exact loss in a bounded two-state environment = delta^2 C", ok, "eight comparisons each way")
    print("\n".join(lines))

    # ------------------------------------------------------------------ T3
    neg = 0
    smallest = math.inf
    exceed = 0
    worst_ratio = 0.0
    worst_ex = None
    for it in range(20000):
        n = int(rng.integers(1, 6))
        taus = np.exp(rng.uniform(math.log(1.01), math.log(1000), n))
        phi = float(rng.uniform(0.0, 0.999))
        d = float(rng.uniform(1e-6, 1.0))
        j = int(rng.integers(0, n))
        gc = closed_form(taus, j, d, phi)
        gg = closed_form(taus, j, 1.0, phi)
        if gc <= 0:
            neg += 1
        smallest = min(smallest, gc / d)
        if gc > gg * (1 + 1e-12):
            exceed += 1
            if gc / gg > worst_ratio:
                worst_ratio, worst_ex = gc / gg, (np.round(taus, 2), j, round(phi, 4), round(d, 3), gc, gg)
    report("T3 the contact-limited loss is positive for 0 < d <= 1 (20,000 random rings)", neg == 0, f"non-positive in {neg}; smallest Gamma/(d sigma^2) = {smallest:.3e}")
    print(f"      reported: the contact-limited loss EXCEEDS the gain-only loss in {exceed} of 20,000 ({100 * exceed / 20000:.1f}%); "
          f"largest ratio {worst_ratio:.1f} at stays {worst_ex[0]}, waiting stage {worst_ex[1]}, memory {worst_ex[2]}, d {worst_ex[3]} "
          f"(contact {worst_ex[4]:.4f}, gain-only {worst_ex[5]:.4f}, units of sigma^2)")

    # ------------------------------------------------------------------ T4
    worst = 0.0
    for d in (0.1, 0.5, 0.9):
        for phi in (0.5, 0.9, 0.99):
            for tj in (2.0, 10.0):
                th = phi / (1 - phi)
                qj = 1 / (1 + tj / th)
                val = closed_form((tj, 1e7), 0, d, phi)
                worst = max(worst, abs(val - d * (1 - d) * qj))
    report("T4 with a very long steady stage the loss tends to d (1 - d) q_j, not to zero", worst <= 1e-4, f"largest difference {worst:.2e}")

    # ------------------------------------------------------------------ T5
    sig = 0.05
    z = np.linspace(-10, 10, 200001)
    wts = np.exp(-z * z / 2)
    wts /= wts.sum()
    h = np.exp(sig * z - sig * sig / 2)
    worst = 0.0
    for p in (0.1, 0.5, 0.9):
        mean_success = float(np.sum(wts * h / (p * h + 1 - p)))
        pred = p * (1 - p) * sig * sig
        worst = max(worst, abs((1 - mean_success) / pred - 1))
    report("T5 the floor is the curvature of the success probability of a wait in a frozen host", worst <= 0.02, f"largest relative difference {worst:.4f}")

    # ------------------------------------------------------------------ T6
    ok = True
    lines = []
    delta = 0.15
    for idx, (c, mu, a2, nu) in enumerate(((1.0, 0.5, 1.2, 1.0), (1.0, 0.1, 0.2, 0.2), (0.3, 1.0, 0.05, 0.05), (1.0, 1.0, 1.0, 5.0))):
        theta = 1 / (2 * nu)
        gc, T = ct_closed(c, mu, a2, theta)
        gg, _ = ct_closed(c, mu, a2, theta, d=1.0)
        pc, pg = delta * delta * gc / T, delta * delta * gg / T
        rc, sec = ct_simulate(c, mu, a2, nu, delta, 8100 + idx, True)
        rg, seg = ct_simulate(c, mu, a2, nu, delta, 8100 + idx, False)
        okc = abs(rc - pc) <= 4 * sec + 0.06 * abs(pc)
        okg = abs(rg - pg) <= 4 * seg + 0.06 * abs(pg)
        ok = ok and okc and okg
        lines.append(f"      c {c}, mu {mu} (d = {mu / (c + mu):.3f}), a2 {a2}, correlation time {theta:.2f}: contact {rc:.4e} +- {sec:.1e} (conjecture {pc:.4e}) {'ok' if okc else 'OUT'}; "
                     f"gain-only {rg:.4e} +- {seg:.1e} (formula {pg:.4e}) {'ok' if okg else 'OUT'}; ratio simulated {rc / rg:.3f}, predicted {pc / pg:.3f}")
    report("T6 continuous time, telegraph host: the loss equals the conjectured closed form", ok, "four parameter sets, both models")
    print("\n".join(lines))

    # ------------------------------------------------------------------ reported
    print("\n      Loss per generation in units of sigma^2; waiting stage of stay 3; host memory in the columns.")
    print("      rows: stay in a steady stage elsewhere; each cell: gain-only / contact-limited with half dying waiting (d = 0.5) / with a tenth dying waiting (d = 0.1)")
    print("      steady stay      memory 0            memory 0.5          memory 0.9          memory 0.99")
    for to in (1.0, 3.0, 10.0, 30.0, 100.0, 1000.0):
        cells = []
        for phi in (0.0, 0.5, 0.9, 0.99):
            cells.append("/".join(f"{closed_form((3.0, to), 0, d, phi):.4f}" for d in (1.0, 0.5, 0.1)))
        print(f"      {to:>8.0f}     " + "   ".join(cells))
    print("      floor d (1 - d) q_j:  " + "   ".join("/".join(f"{d * (1 - d) / (1 + 3.0 * (1 - phi) / phi) if phi > 0 else 0.0:.4f}" for d in (1.0, 0.5, 0.1)) for phi in (0.0, 0.5, 0.9, 0.99)))
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--t6-only":
        ok_all = True
        delta = 0.15
        for idx, (c, mu, a2, nu) in enumerate(((1.0, 0.5, 1.2, 1.0), (1.0, 0.1, 0.2, 0.2), (0.3, 1.0, 0.05, 0.05), (1.0, 1.0, 1.0, 5.0))):
            theta = 1 / (2 * nu)
            gc, T = ct_closed(c, mu, a2, theta)
            gg, _ = ct_closed(c, mu, a2, theta, d=1.0)
            pc, pg = delta * delta * gc / T, delta * delta * gg / T
            rc, sec = ct_simulate(c, mu, a2, nu, delta, 8100 + idx, True)
            rg, seg = ct_simulate(c, mu, a2, nu, delta, 8100 + idx, False)
            okc = abs(rc - pc) <= 4 * sec + 0.06 * abs(pc)
            okg = abs(rg - pg) <= 4 * seg + 0.06 * abs(pg)
            ok_all = ok_all and okc and okg
            print(f"c {c}, mu {mu} (d = {mu / (c + mu):.3f}), a2 {a2}, correlation time {theta:.2f}: contact {rc:.4e} +- {sec:.1e} (conjecture {pc:.4e}) {'ok' if okc else 'OUT'}; "
                  f"gain-only {rg:.4e} +- {seg:.1e} (formula {pg:.4e}) {'ok' if okg else 'OUT'}; ratio simulated {rc / rg:.3f}, predicted {pc / pg:.3f}")
        print("T6 PASS" if ok_all else "T6 FAIL")
        sys.exit(0 if ok_all else 1)
    sys.exit(main())
