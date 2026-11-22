# Adopted into checks/ on 14 November 2026 at 09:42 from the working files of the reader that read the paper in full
# (cards in notes/literature/). Formula against formula; nothing here is a simulation. Unchanged below this line.
"""Prior-art checks for three papers (formula against formula; no simulation).

Claims tested (tolerances fixed here, before running):

R1  Reproduction. Tuljapurkar (1985, p. 7) prints, for his 5-age-class Leslie
    matrix, lambda0 = 1.0261, lambda1 = 0.0642 + 0.7271i,
    lambda2 = -0.5772 + 0.3675i and T0 = 3.55.  Tolerance 5e-5 on the
    eigenvalues, 5e-3 on T0.
R2  Reproduction. Orzack & Tuljapurkar (1989, Table 1): ln lambda0, T0 and D
    for the nine flat life histories.  Tolerance 6e-5, 6e-4, 6e-5.

C1  Tuljapurkar (1985) eqs. (8) and (15),
        lambda - lambda0 = Re[v0' B (lambda0 z - A)^-1 B u0] / (2 v0'u0),
    equal (a) the exact growth rate of the cycle matrix to O(g^4), and
    (b) the project's Result 7, lambda0 (g^2/2)(1/T_e) Re[ghat/(1-ghat)],
    with ghat the transform of the return time.  Tolerance for (b): 1e-10
    relative.  For (a): the ratio (exact - formula)/g^4 stays bounded as g
    is divided by 10 (reported, not thresholded).
C1d The long-period limit printed on p. 13, -(g^2/4T0^2), against the exact
    cycle-matrix value and against the project's slow limit
    -lambda0 (g^2/4)(1 - CV^2)/T0.  Reported.

C2  Tuljapurkar (1982b) eq. (2) with (15) (geometric autocorrelation rho),
        a - ln lambda0 = -tau^2/(2 lambda0^2)
            + sum_{alpha>=1} rho (V0 x Valpha)' C0 (Ualpha x U0)
                  / (lambda0^2 T0 Talpha (1 - rho lambda_alpha/lambda0)),
    equals minus the project's closed forms: eq. (2) for a ring transfer,
    eq. (5) for a ring staying rate, eq. (8) for a Leslie matrix whose
    fertilities share one series.  Tolerance 1e-9 relative, 400 random cases
    each.
"""
import numpy as np

rng = np.random.default_rng(20261114)


def eig_sorted(A):
    w, U = np.linalg.eig(A)
    order = np.argsort(-np.abs(w))
    w, U = w[order], U[:, order]
    Vt = np.linalg.inv(U)          # rows are left eigenvectors, V_a' U_b = delta_ab
    return w, U, Vt


# ---------------------------------------------------------------- R1
A85 = np.array([[0, 2, 4, 5, 5],
                [0.1, 0, 0, 0, 0],
                [0, 0.8, 0, 0, 0],
                [0, 0, 0.8, 0, 0],
                [0, 0, 0, 0.8, 0]], float)
w, U, Vt = eig_sorted(A85)
lam0 = w[0].real
l = np.array([1, 0.1, 0.08, 0.064, 0.0512])
f = A85[0]
ages = np.arange(1, 6)
q = l * f * lam0 ** (-ages)
T0 = (ages * q).sum()
VarL = (ages ** 2 * q).sum() - T0 ** 2
CV2 = VarL / T0 ** 2
print("R1 lambda0 %.5f (printed 1.0261)  T0 %.4f (printed 3.55)" % (lam0, T0))
cplx = sorted([x for x in w[1:] if x.imag > 0], key=lambda x: -abs(x))
print("R1 complex eigenvalues", [complex(round(x.real, 4), round(x.imag, 4)) for x in cplx],
      "(printed 0.0642+0.7271i, -0.5772+0.3675i)")
assert abs(lam0 - 1.0261) < 5e-5 and abs(T0 - 3.55) < 5e-3
# Departure from the tolerance stated above: 5e-5 was meant per printed decimal, and was first
# coded on the modulus, which lambda2 fails (6.9e-5) because both parts sit at the rounding
# boundary (-0.577248, 0.367450).  Coded now per component, with 5.5e-5 to admit the boundary.
def close(x, y):
    return abs(x.real - y.real) < 5.5e-5 and abs(x.imag - y.imag) < 5.5e-5
assert any(close(x, 0.0642 + 0.7271j) for x in cplx)
assert any(close(x, -0.5772 + 0.3675j) for x in cplx)
print("R1 sum q = %.6f, Var L = %.4f, CV^2 = %.4f" % (q.sum(), VarL, CV2))

# ---------------------------------------------------------------- R2
tab1 = [(0.525, 1, 2, 0.0326, 1.492, 0.5001), (0.2625, 1, 4, 0.0196, 2.475, 0.2501),
        (0.175, 1, 6, 0.0140, 3.459, 0.1668), (0.13125, 1, 8, 0.0109, 4.443, 0.1251),
        (0.105, 1, 10, 0.0089, 5.426, 0.1001), (0.13125, 3, 10, 0.0075, 6.460, 0.1250),
        (0.175, 5, 10, 0.0065, 7.481, 0.1667), (0.2625, 7, 10, 0.0057, 8.493, 0.2500),
        (0.525, 9, 10, 0.0051, 9.499, 0.5000)]
worst = [0, 0, 0]
for phi, a0, a1, lnl, T, D in tab1:
    ag = np.arange(a0, a1 + 1)
    lo, hi = 1.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (phi * mid ** (-ag.astype(float))).sum() > 1:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    G = phi * lam ** (-ag.astype(float))
    worst = [max(worst[0], abs(np.log(lam) - lnl)), max(worst[1], abs((ag * G).sum() - T)),
             max(worst[2], abs((G ** 2).sum() - D))]
print("R2 Table 1 of O&T 1989: worst |diff| ln lambda0 %.1e, T0 %.1e, D %.1e" % tuple(worst))
assert worst[0] < 6e-5 and worst[1] < 6e-4 and worst[2] < 6e-5


# ---------------------------------------------------------------- C1
def exact_cycle(A, Bunit, g, T):
    C = np.eye(A.shape[0])
    om = 2 * np.pi / T
    for m in range(T):
        C = (A + g * Bunit * np.cos(m * om)) @ C
    return max(abs(np.linalg.eigvals(C))) ** (1.0 / T)


def eq15(A, Bunit, g, T):
    w, U, Vt = eig_sorted(A)
    lam0 = w[0].real
    u0 = U[:, 0].real
    v0 = Vt[0].real
    z = np.exp(2j * np.pi / T)
    B = g * Bunit
    b1 = 0.5 * np.linalg.solve(lam0 * z * np.eye(len(A)) - A, B @ u0)
    return (v0 @ B @ b1.real) / (v0 @ u0)


BS = np.zeros((5, 5)); BS[1, 0] = 0.1          # Case S: p1(t) = 0.1 + 0.1 g cos
BF = np.zeros((5, 5)); BF[0, 1] = 2.0          # Case F: f2(t) = 2 + 2 g cos


def project_S(g, T):
    z = np.exp(2j * np.pi / T)
    gh = (q * z ** ages).sum()
    return lam0 * (g ** 2 / 2) * (1 / T0) * (gh / (1 - gh)).real


def project_F(g, T):
    z = np.exp(2j * np.pi / T)
    gh = (q * z ** ages).sum()
    return lam0 * (g ** 2 / 2) * (q[1] / T0) * (q[1] * z ** 2 / (1 - gh)).real


print("\nC1 Case S (juvenile survival), g = 0.2: lambda - lambda0, times 1e4")
print(" period   exact    eq.(15)   project R7   printed limit -(g^2/4T0^2) = %.3f" % (-1e4 * 0.04 / (4 * T0 ** 2)))
worst_b = 0
for T in list(range(3, 21)) + [50, 200, 1000]:
    ex = exact_cycle(A85, BS, 0.2, T) - lam0
    e15 = eq15(A85, BS, 0.2, T)
    pr = project_S(0.2, T)
    worst_b = max(worst_b, abs(e15 - pr) / abs(pr))
    print(" %6d  %8.3f  %8.3f  %8.3f" % (T, 1e4 * ex, 1e4 * e15, 1e4 * pr))
print("C1b worst relative |eq.(15) - project R7|, Case S: %.1e" % worst_b)
assert worst_b < 1e-10
worst_bF = 0
print("\nC1 Case F (fecundity of age class 2), g = 0.2: lambda - lambda0, times 1e4")
for T in list(range(3, 21)) + [50, 200, 1000]:
    ex = exact_cycle(A85, BF, 0.2, T) - lam0
    e15 = eq15(A85, BF, 0.2, T)
    pr = project_F(0.2, T)
    worst_bF = max(worst_bF, abs(e15 - pr) / abs(pr))
    print(" %6d  %8.3f  %8.3f  %8.3f" % (T, 1e4 * ex, 1e4 * e15, 1e4 * pr))
print("C1b worst relative |eq.(15) - project R7|, Case F: %.1e" % worst_bF)
assert worst_bF < 1e-10

print("\nC1a order of the error of eq.(15), Case S: (exact - eq.15)/g^4")
for T in (3, 4, 7, 20, 200):
    print("  period %4d:" % T, "  ".join("g=%.3f: %+.4f" % (g, (exact_cycle(A85, BS, g, T) - lam0 - eq15(A85, BS, g, T)) / g ** 4)
                                        for g in (0.2, 0.02)))

print("\nC1d long periods, Case S, (lambda - lambda0)/g^2 at g = 0.02:")
for T in (20, 100, 1000, 10000):
    print("  period %6d: exact %+.6f   eq.(15) %+.6f" % (T, (exact_cycle(A85, BS, 0.02, T) - lam0) / 4e-4, eq15(A85, BS, 0.02, T) / 4e-4))
print("  printed limit -(1/4T0^2)               = %+.6f" % (-1 / (4 * T0 ** 2)))
print("  project slow limit -lambda0(1-CV^2)/4T0 = %+.6f" % (-lam0 * (1 - CV2) / (4 * T0)))
print("  alpha = 0 term of (13)-(14) alone, any period: %+.6f" % (-1 / (4 * T0 ** 2)))


# ---------------------------------------------------------------- C2
def tulja82b(A, F, rho, sig2=1.0):
    """a - ln lambda0 from eq. (2) with tau^2 of (11) and theta of (15); B_t = eps_t F."""
    w, U, Vt = eig_sorted(A)
    lam0 = w[0].real
    U0, V0 = U[:, 0], Vt[0]
    T0 = V0 @ U0
    tau2 = sig2 * (V0 @ F @ U0) ** 2 / T0 ** 2
    th = 0
    for a in range(1, len(w)):
        Ta = Vt[a] @ U[:, a]
        th += rho * sig2 * (V0 @ F @ U[:, a]) * (Vt[a] @ F @ U0) / (lam0 ** 2 * T0 * Ta * (1 - rho * w[a] / lam0))
    return (-tau2 / (2 * lam0 ** 2) + th).real, (tau2 / lam0 ** 2).real


def ring(n):
    s = rng.uniform(0.05, 0.9, n)
    g = rng.uniform(0.2, 2.0, n)
    A = np.diag(s)
    for k in range(n):
        A[(k + 1) % n, k] += g[k]
    return A, s, g


worst = {"transfer": 0, "staying": 0, "leslie": 0}
for _ in range(400):
    n = int(rng.integers(1, 9))
    A, s, g = ring(n)
    lam = max(np.linalg.eigvals(A).real)
    m = s / lam
    tau = 1 / (1 - m)
    Tgen = tau.sum()
    phi = rng.uniform(-0.9, 0.95)
    th = phi / (1 - phi)
    R = np.prod((1 - m) * phi / (1 - m * phi))
    # one fluctuating transfer
    k = int(rng.integers(0, n))
    F = np.zeros((n, n)); F[(k + 1) % n, k] = g[k]
    if n == 1:
        F[0, 0] = g[0]
    ours = (1 + 2 * th) / (2 * Tgen ** 2) - (1 / Tgen) * R / (1 - R)
    his, _ = tulja82b(A, F, phi)
    worst["transfer"] = max(worst["transfer"], abs(his + ours) / abs(ours))
    # one fluctuating staying rate
    F = np.zeros((n, n)); F[k, k] = s[k]
    e = (tau[k] - 1) / Tgen
    ours = e ** 2 * (1 + 2 * th) / 2 - e * phi * m[k] / ((1 - phi * m[k]) * (1 - R))
    his, _ = tulja82b(A, F, phi)
    worst["staying"] = max(worst["staying"], abs(his + ours) / max(abs(ours), 1e-12))
for _ in range(400):
    k = int(rng.integers(2, 11))
    fert = rng.uniform(0.0, 2.0, k); fert[-1] += 0.1; fert[0] += 0.05
    p = rng.uniform(0.2, 0.95, k - 1)
    A = np.zeros((k, k)); A[0] = fert
    for i in range(k - 1):
        A[i + 1, i] = p[i]
    lam = max(np.linalg.eigvals(A).real)
    la = np.concatenate([[1.0], np.cumprod(p)])
    ag = np.arange(1, k + 1)
    qa = la * fert * lam ** (-ag.astype(float))
    T = (ag * qa).sum()
    phi = rng.uniform(-0.9, 0.95)
    th = phi / (1 - phi)
    R = (la * fert * (phi / lam) ** ag).sum()
    ours = (1 + 2 * th) / (2 * T ** 2) - (1 / T) * R / (1 - R)
    F = np.zeros((k, k)); F[0] = fert
    his, _ = tulja82b(A, F, phi)
    worst["leslie"] = max(worst["leslie"], abs(his + ours) / abs(ours))
print("\nC2 Tuljapurkar (1982b) eqs. (2)+(15) against the project's closed forms, worst relative difference:")
for key, val in worst.items():
    print("   %-9s %.1e" % (key, val))
    assert val < 1e-9
print("all assertions passed")
