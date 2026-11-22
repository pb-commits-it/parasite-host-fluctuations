"""One identity behind the mechanism results: the loss is the bound minus half the divergence of lineages.

Claim (second order in the noise; any primitive mean matrix A0 with growth factor lambda; any set of
stationary drivers xi_d with unit variance; entry (i, j) of the matrix multiplied to first order by
1 + sum_d K_d[i, j] xi_d(t)):

        C  =  J  -  V_between / 2,

where  growth = ln(lambda) - sigma^2 C + ...,
       J = (1/2) sum over edges e of c_e Var(reward on e)   (the bound of Proposition 21 to this order),
       c_e = w_i A0_ij v_j / lambda  the frequency with which a line of descent uses edge e = (j -> i),
       V_between = the rate at which the variance AMONG lines of descent of their accumulated reward
                   grows, in a fixed environment, averaged over environments.
The lines of descent are those of the mean matrix: a Markov chain that goes from j to i with
probability w_i A0_ij / (lambda w_j), independent of the environment.  The reward collected on a step
j -> i at time t is sum_d K_d[i, j] xi_d(t).

The identity follows from the law of total variance applied to the lag sum of the general formula
(notes/lineage-variance.md).  It is checked here in the only way that is not circular: C is computed
from the matrix formula (resolvent), J from the edge frequencies, and V_between is MEASURED by
simulating lines of descent through sampled environments.  Nothing in the measurement uses the formula.

Cases: (1) a three-stage ring, one fluctuating transfer, host memory 0.7;
       (2) a contact-limited transfer (the staying rate moves against the transfer), memory 0.8;
       (3) two independent hosts with different memories on a three-stage ring;
       (4) two patches joined by dispersal of one stage, independent hosts with memory 0.9 (the
           inflation case: C < 0 and V_between > 2J);
       (5) two alternative hosts in parallel for one step;
       (6) a ring without staying (lineages all share one fate: V_between = 0, C = J).
Criterion, fixed before the first run: |J - V/2 - C| <= 4 standard errors of V/2 + 0.01 |J| in every
case (the second part for the finite length of the simulated histories), and in case (6) V/2 <= 1e-12.

FIRST RUN (kept as results/lineage-variance-identity_attempt1_20261114.txt): cases 1 to 5 passed; case
6 failed, V/2 = 0.034.  The fault was in the design of case 6, not in the identity.  A ring without
staying is not primitive: lines of descent that start in different stages belong to three cohorts
that never mix, each meeting the host at its own times.  Their accumulated rewards differ by amounts
that grow like the square root of time, so the population is the LARGEST of three cohorts and not a
mixture to which a variance correction applies; the identity, like the growth formula, needs a
primitive mean matrix.  Within one cohort every line of descent has exactly the same history.  Case 6
now starts all lines in one stage and requires V/2 <= 1e-12 there; it also reports the value across
cohorts, with no criterion.  No tolerance was changed.

Run: python3 -I checks/lineage_variance_identity.py   (about two minutes)
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


def perron(A):
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(vals.real))
    lam = float(vals[i].real)
    v = np.abs(vecs[:, i].real)
    vl, wl = np.linalg.eig(A.T)
    w = np.abs(wl[:, int(np.argmax(vl.real))].real)
    v /= v.sum()
    w /= w @ v
    return lam, v, w


def coefficient(A, Ks, phis):
    """C from the matrix formula, independent AR(1) drivers with unit variance."""
    lam, v, w = perron(A)
    n = A.shape[0]
    M = A / lam
    Q = np.eye(n) - np.outer(v, w)
    C = 0.0
    for K, phi in zip(Ks, phis):
        B = K * A / lam
        c = float(w @ B @ v)
        C += c * c / 2
        if phi != 0.0:
            X = np.linalg.solve(np.eye(n) - phi * (M @ Q), Q @ B @ v)
            C -= phi * float(w @ B @ X)
    return C, lam, v, w


def jensen(A, Ks, lam, v, w):
    ce = (w[:, None] * A * v[None, :]) / lam          # ce[i, j]: frequency of the step j -> i
    return 0.5 * float(sum((ce * K * K).sum() for K in Ks)), ce


def measure_v_between(A, Ks, phis, lam, v, w, seed, n_env=160, n_lin=500, T=1500, burn=200, one_cohort=False):
    """Rate of growth of the between-lineage variance of accumulated reward, with a standard error over environments."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    P = (w[:, None] * A / (lam * w[None, :]))          # P[i, j]: probability of going from j to i
    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-10)
    cum = np.cumsum(P, axis=0)                         # by column
    pi = v * w
    state = rng.choice(n, size=(n_env, n_lin), p=pi / pi.sum())
    if one_cohort:
        state = np.zeros((n_env, n_lin), dtype=int)
    D = len(Ks)
    xi = rng.standard_normal((n_env, D))
    S = np.zeros((n_env, n_lin))
    var_at = {}
    phis = np.asarray(phis)
    Kst = np.stack(Ks)                                  # (D, n, n)
    for t in range(burn + 2 * T):
        xi = phis * xi + np.sqrt(1 - phis ** 2) * rng.standard_normal((n_env, D))
        u = rng.random((n_env, n_lin))
        nxt = (u[..., None] > cum.T[state]).sum(axis=-1)          # cum.T[j] is the cumulative column j
        nxt = np.minimum(nxt, n - 1)
        if t >= burn:
            k = Kst[:, nxt, state]                                  # (D, n_env, n_lin)
            S += np.einsum("dek,ed->ek", k, xi)
        state = nxt
        if t == burn + T - 1 or t == burn + 2 * T - 1:
            var_at[t] = S.var(axis=1, ddof=1).copy()
    rate = (var_at[burn + 2 * T - 1] - var_at[burn + T - 1]) / T   # differencing removes the transient
    return float(rate.mean()), float(rate.std(ddof=1) / math.sqrt(n_env))


def ring(taus, lam=1.0):
    taus = np.asarray(taus, float)
    n = len(taus)
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] += lam * (1 - 1 / taus[k])
        A[(k + 1) % n, k] += lam / taus[k]
    return A


def main():
    cases = []

    A = ring((2.0, 5.0, 3.0))
    K = np.zeros_like(A); K[1, 0] = 1.0
    cases.append(("1 ring, one fluctuating transfer, memory 0.7", A, [K], [0.7]))

    A = ring((4.0, 6.0))
    d = 0.5
    c = (1 - A[0, 0]) * (1 - d)
    K = np.zeros_like(A); K[1, 0] = 1.0; K[0, 0] = -c / A[0, 0]
    cases.append(("2 contact-limited transfer, half die waiting, memory 0.8", A, [K], [0.8]))

    A = ring((3.0, 2.0, 6.0))
    K1 = np.zeros_like(A); K1[1, 0] = 1.0
    K2 = np.zeros_like(A); K2[0, 2] = 1.0
    cases.append(("3 two independent hosts, memories 0.5 and 0.9", A, [K1, K2], [0.5, 0.9]))

    # two patches x two stages; state index = patch * 2 + stage; stage 1 disperses with m = 0.15
    base = ring((2.0, 2.0))
    m = 0.15
    A = np.zeros((4, 4))
    for p in range(2):
        A[2 * p, 2 * p] = base[0, 0]                       # staying in stage 0
        A[2 * p, 2 * p + 1] = base[0, 1]                   # transfer 1 -> 0, within the patch
        for q in range(2):                                  # stage 1 (stayers and arrivals) moves after demography
            share = (1 - m) if q == p else m
            A[2 * q + 1, 2 * p + 1] += share * base[1, 1]
            A[2 * q + 1, 2 * p] += share * base[1, 0]
    Ks = []
    for p in range(2):
        K = np.zeros_like(A)
        for q in range(2):
            K[2 * q + 1, 2 * p] = 1.0                       # the transfer 0 -> 1 of patch p follows host p
        Ks.append(K)
    cases.append(("4 two patches, one stage disperses (m = 0.15), independent hosts, memory 0.9", A, Ks, [0.9, 0.9]))

    # parallel: stage 0 -> host a (share 0.6) or host b (share 0.4) -> back to stage 0
    t0, ta, tb = 3.0, 4.0, 2.0
    A = np.zeros((3, 3))
    A[0, 0] = 1 - 1 / t0
    A[1, 0] = 0.6 / t0; A[2, 0] = 0.4 / t0
    A[1, 1] = 1 - 1 / ta; A[2, 2] = 1 - 1 / tb
    A[0, 1] = 1 / ta; A[0, 2] = 1 / tb
    Ka = np.zeros_like(A); Ka[1, 0] = 1.0
    Kb = np.zeros_like(A); Kb[2, 0] = 1.0
    cases.append(("5 two alternative hosts in parallel, memories 0.6 and 0.6", A, [Ka, Kb], [0.6, 0.6]))

    A = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    K = np.zeros_like(A); K[1, 0] = 1.0
    cases.append(("6 ring without staying, memory 0.7", A, [K], [0.7]))

    print("  case                                                              C (matrix formula)   J (bound)   V_between/2 (measured)   J - V/2      difference")
    for idx, (label, A, Ks, phis) in enumerate(cases):
        if label.startswith("6"):
            # not primitive: the growth formula needs a primitive matrix, but the lineages and J are defined; C = J is the known no-staying value
            lam, v, w = perron(A)
            J, _ = jensen(A, Ks, lam, v, w)
            V, se = measure_v_between(A, Ks, phis, lam, v, w, 9100 + idx, n_env=40, n_lin=200, T=600, one_cohort=True)
            Vx, sex = measure_v_between(A, Ks, phis, lam, v, w, 9200 + idx, n_env=40, n_lin=200, T=600)
            ok = abs(V / 2) <= 1e-12
            report(label, ok, f"J = {J:.6f} (one third of 1/2: the transfer is used once in three steps); within one cohort V_between/2 = {V / 2:.2e}: "
                   f"every line of descent shares one fate, and the loss is the whole bound.  Across the three cohorts, which never mix, the value is {Vx / 2:.4f} and the identity does not apply")
            continue
        C, lam, v, w = coefficient(A, Ks, phis)
        J, _ = jensen(A, Ks, lam, v, w)
        V, se = measure_v_between(A, Ks, phis, lam, v, w, 9100 + idx)
        diff = J - V / 2 - C
        ok = abs(diff) <= 4 * se / 2 + 0.01 * abs(J)
        report(label, ok, f"C = {C:+.6f}, J = {J:.6f}, V/2 = {V / 2:.6f} +- {se / 2:.6f}, J - V/2 = {J - V / 2:+.6f}, difference {diff:+.6f}"
               + ("   <- C is negative: lineages diverge faster than the bound" if C < 0 else ""))
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
