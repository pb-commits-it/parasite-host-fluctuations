"""What happens beyond the stability boundaries of notes/coupled-hosts.md, K4 and K5?

The note claims linear instability only.  secondary integrated two unstable cases and both settled onto
cycles (verification/check-coupled-hosts.md, 1.2).  This script computes the first Lyapunov coefficient l1 at
the boundaries, which says whether a small stable cycle is born there (l1 < 0, a supercritical Hopf
bifurcation) or the loss of stability is abrupt (l1 > 0, subcritical, in which case large excursions
can occur even where the equilibrium is still linearly stable).

Formula (Kuznetsov, Elements of Applied Bifurcation Theory, section 5.4; the same as check S7 of
r5c_shared_regulation.py): with A the Jacobian at the critical point, A q = i w q, A' p = -i w p,
<p, q> = sum conj(p) q = 1, and B, C the second and third derivatives of the vector field,
    l1 = (1/(2w)) Re[ <p, C(q,q,q*)> - 2 <p, B(q, A^-1 B(q,q*))> + <p, B(q*, (2iw - A)^-1 B(q,q))> ].
Its sign does not depend on the scaling of q.

Systems.  One host block and one unregulated stage w of rate aL with static gain fixed:
  V:  dN/dt = Phi(N) - alpha I,  dI/dt = cL w (N - I) - a I,                 dw/dt = cV I - aL w
  M:  dH/dt = Phi(H) - alpha P,  dP/dt = cL w H - xi P - alpha eps P^2/H,    dw/dt = cM P - aL w
with Phi logistic (a self-regulated host held below its peak production; boundaries a_- and a_+ of
the island of instability) or Phi = r N (no self-regulation; boundary a_c).  Block V is sampled with
a >= alpha, the condition for it to be a possible host.

Checks and tolerances, fixed before the first run:
  L0  (known result reproduced) the routine gives l1 = -0.096225044865 for the three-stage ring with
      exp(-x) in every transfer at its critical point, the value three referees derived and check S7
      confirms.  Tolerance 1e-8 relative.
  L1  at every boundary used, the Jacobian has a purely imaginary pair: |Re| < 1e-8 times the largest
      eigenvalue modulus (this checks the boundary formulas of K4 and K5 once more).
  Reported, no criterion (an open question): the number of boundaries with l1 < 0 and with l1 > 0, for
  each of V and M, each kind of host, each end of the island.
  L2  (consistency) for six boundaries with l1 < 0, the nonlinear model integrated at a stage rate 3%
      inside the unstable side settles onto a cycle whose relative amplitude in the host block's
      parasite variable is within 25% of the prediction 2 sqrt(mu/(w |l1|)) |q_k| made from the normal form, mu
      being the real part of the critical pair there, converted to that variable.  (Loose, because 3%
      is not infinitesimal; it is a check that the sign and the size of l1 mean what they should.)

Run: python3 -I checks/coupled_hopf.py   (about a minute)
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

RNG = np.random.default_rng(20261114)
FAILS = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def logu(lo, hi):
    return float(np.exp(RNG.uniform(math.log(lo), math.log(hi))))


def lyapunov(A, B, C):
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.imag - 1e6 * np.abs(ev.real)))           # the eigenvalue i w with w > 0 nearest the axis
    w = float(ev[i].imag)
    q = V[:, i]
    evl, W = np.linalg.eig(A.T)
    j = int(np.argmin(np.abs(evl + 1j * w)))
    pvec = W[:, j]                                                # A' p = -i w p
    pvec = pvec / np.conj(np.vdot(pvec, q))                       # <p, q> = sum conj(p) q = 1
    n = A.shape[0]
    t1 = np.vdot(pvec, C(q, q, np.conj(q)))
    t2 = np.vdot(pvec, B(q, np.linalg.solve(A, B(q, np.conj(q)))))
    t3 = np.vdot(pvec, B(np.conj(q), np.linalg.solve(2j * w * np.eye(n) - A, B(q, q))))
    return float((t1 - 2 * t2 + t3).real / (2 * w)), w, float(abs(ev[i].real) / np.max(np.abs(ev))), q, pvec


# ---------------------------------------------------------------- the known case
def ring3():
    n = 3
    q_ = 1 / math.cos(math.pi / n)
    s = 1 + q_
    P = np.roll(np.eye(n), 1, axis=0)
    A = -np.eye(n) + (1 - s) * P
    g2, g3 = s - 2.0, 3.0 - s
    B = lambda a, b: g2 * (P @ (a * b))
    C = lambda a, b, c: g3 * (P @ (a * b * c))
    l1, w, res, q, p = lyapunov(A, B, C)
    # normalise as in S7: components of q of modulus one
    scale = abs(q[0])
    return l1 / scale ** 2, res                                   # l1 scales as |q|^2; S7 uses components of modulus one


# ---------------------------------------------------------------- V and M with one stage
def island_roots(Aq, Bc, J, q, rho):
    """Roots of F(a) = J a^2 + (A J + q rho) a + A B, when it has two positive roots."""
    c1 = Aq * J + q * rho
    disc = c1 * c1 - 4 * J * Aq * Bc
    if J <= 0 or c1 >= 0 or disc <= 0:
        return None
    r = math.sqrt(disc)
    return (-c1 - r) / (2 * J), (-c1 + r) / (2 * J)


def make_V(logistic):
    alpha = logu(0.05, 5)
    a = alpha + logu(0.02, 5)
    u = logu(0.02, 20)
    p = u / (u + a)
    if logistic:
        x = RNG.uniform(0.01, 0.5)
        r = alpha * p / (1 - x)
        K = 1.0
        N = K * x
        rho = -r * (1 - 2 * x)
        phi2 = -2 * r / K
    else:
        r = alpha * p
        N = 1.0
        rho = -r
        phi2 = 0.0
    I = p * N
    d, h, q = a / (1 - p), alpha * p, a
    Aq, Bc, J = rho + d, d * (rho + h), rho + d - q
    return dict(kind="V", a=a, alpha=alpha, u=u, N=N, I=I, rho=rho, phi2=phi2, r=r, logistic=logistic, K=1.0,
                d=d, h=h, q=q, Aq=Aq, Bc=Bc, J=J)


def make_M(logistic):
    alpha = logu(0.01, 1)
    xi = alpha + logu(0.02, 5)
    m = logu(0.1, 20)
    eps = RNG.uniform(1.0, 3.0)
    if logistic:
        x = RNG.uniform(0.01, 0.5)
        r = alpha * m / (1 - x)
        H = x
        rho = -r * (1 - 2 * x)
        phi2 = -2 * r
    else:
        r = alpha * m
        H = 1.0
        rho = -r
        phi2 = 0.0
    Pp = m * H
    h = alpha * m
    q, d = xi + eps * h, xi + 2 * eps * h
    u = xi * m + alpha * eps * m * m
    Aq, Bc, J = rho + d, d * (rho + h), rho + d - q
    return dict(kind="M", xi=xi, alpha=alpha, eps=eps, m=m, u=u, N=H, I=Pp, rho=rho, phi2=phi2, r=r, logistic=logistic, K=1.0,
                d=d, h=h, q=q, Aq=Aq, Bc=Bc, J=J)


def system(b, aL):
    """Jacobian, B, C and the right-hand side for the block b with one stage of rate aL (w* = 1)."""
    N, I, al = b["N"], b["I"], b["alpha"]
    cL = b["u"]                                   # cL w* = u*, w* = 1
    cO = aL / I                                   # output scale: cO I* = aL w*
    phi1 = -b["rho"]
    if b["kind"] == "V":
        a = b["a"]
        A = np.array([[phi1, -al, 0.0], [cL, -(cL + a), cL * (N - I)], [0.0, cO, -aL]])

        def B(x, y):
            return np.array([b["phi2"] * x[0] * y[0], cL * (x[2] * (y[0] - y[1]) + y[2] * (x[0] - x[1])), 0.0])

        def C(x, y, z):
            return np.zeros(3, dtype=complex)

        def rhs(t, s):
            n, i, w = s
            phi = b["r"] * n * (1 - n / b["K"]) if b["logistic"] else b["r"] * n
            return [phi - al * i, cL * w * (n - i) - a * i, cO * i - aL * w]
    else:
        xi, eps, m = b["xi"], b["eps"], b["m"]
        H, P = N, I
        ae = al * eps
        gPP, gPH, gHH = 2 / H, -2 * P / H ** 2, 2 * P ** 2 / H ** 3
        gPPH, gPHH, gHHH = -2 / H ** 2, 4 * P / H ** 3, -6 * P ** 2 / H ** 4
        A = np.array([[phi1, -al, 0.0], [cL + ae * m * m, -xi - 2 * ae * m, cL * H], [0.0, cO, -aL]])

        def B(x, y):
            g = gPP * x[1] * y[1] + gPH * (x[1] * y[0] + x[0] * y[1]) + gHH * x[0] * y[0]
            return np.array([b["phi2"] * x[0] * y[0], cL * (x[2] * y[0] + y[2] * x[0]) - ae * g, 0.0])

        def C(x, y, z):
            g = (gPPH * (x[1] * y[1] * z[0] + x[1] * y[0] * z[1] + x[0] * y[1] * z[1])
                 + gPHH * (x[1] * y[0] * z[0] + x[0] * y[1] * z[0] + x[0] * y[0] * z[1])
                 + gHHH * x[0] * y[0] * z[0])
            return np.array([0.0, -ae * g, 0.0])

        def rhs(t, s):
            hh, pp, w = s
            phi = b["r"] * hh * (1 - hh / b["K"]) if b["logistic"] else b["r"] * hh
            return [phi - al * pp, cL * w * hh - xi * pp - ae * pp * pp / hh, cO * pp - aL * w]
    return A, B, C, rhs, np.array([N, I, 1.0])


def main():
    val, res = ring3()
    report("L0 the routine reproduces the three-stage ring", abs(val / -0.096225044865 - 1) < 1e-8 and res < 1e-9,
           f"l1 = {val:.12f} (referees: -0.096225044865)")

    rows = {}
    worst_axis = 0.0
    examples = []
    for kind, maker in (("V", make_V), ("M", make_M)):
        for logistic in (True, False):
            found = tries = 0
            while found < 300 and tries < 400_000:
                tries += 1
                b = maker(logistic)
                if logistic:
                    roots = island_roots(b["Aq"], b["Bc"], b["J"], b["q"], b["rho"])
                    if roots is None:
                        continue
                    bounds = (("lower end of the island (long stage)", roots[0]), ("upper end of the island (short stage)", roots[1]))
                else:
                    if b["J"] <= 0 or b["Aq"] * b["J"] >= b["q"] * b["h"]:
                        continue
                    bounds = (("critical length", b["q"] * b["h"] / b["J"] - b["Aq"]),)
                found += 1
                for label, aL in bounds:
                    A, B, C, rhs, eq = system(b, aL)
                    l1, w, axis, q, p = lyapunov(A, B, C)
                    worst_axis = max(worst_axis, axis)
                    key = (kind, "self-regulated host below its peak" if logistic else "no self-regulation", label)
                    rows.setdefault(key, []).append(l1)
                    if l1 < 0 and len(examples) < 6 and RNG.random() < 0.2:
                        examples.append((b, aL, label, l1, w))
            print(f"      {kind}, {'logistic' if logistic else 'no self-regulation'}: {found} parameter sets with a boundary, from {tries} drawn")
    report("L1 a purely imaginary pair at every boundary used", worst_axis < 1e-8, f"largest |Re| / largest modulus = {worst_axis:.1e}")
    print("\n      sign of the first Lyapunov coefficient at the boundaries (negative: a small stable cycle is born)")
    for key, vals in rows.items():
        vals = np.array(vals)
        print(f"        {key[0]}, {key[1]}, {key[2]}: negative {int(np.sum(vals < 0))}, positive {int(np.sum(vals > 0))} of {len(vals)}")

    # L2: amplitude against the normal form, for six supercritical boundaries
    ok_all, lines = True, []
    for b, aL, label, l1, w in examples:
        inside = aL * (1.03 if "lower" in label else 0.97) if "island" in label else aL * 0.97
        A, B, C, rhs, eq = system(b, inside)
        ev = np.linalg.eigvals(A)
        lead = ev[np.argmax(ev.real)]
        mu = float(lead.real)
        if mu <= 0:
            lines.append(f"        skipped ({label}): 3% step did not land on the unstable side")
            continue
        A0, B0, C0, _, _ = system(b, aL)
        l1c, wc, _, q, p = lyapunov(A0, B0, C0)
        amp_pred = 2 * math.sqrt(mu / (wc * abs(l1c))) * abs(q[1]) / eq[1]     # relative amplitude of the block's parasite variable
        T = 60 / mu
        sol = solve_ivp(rhs, (0, T), eq * (1 + 1e-3 * np.array([1.0, -1.0, 0.5])), method="LSODA", rtol=1e-10, atol=1e-13, dense_output=True)
        tt = np.linspace(0.9 * T, T, 4000)
        yy = sol.sol(tt)[1]
        amp_sim = (yy.max() - yy.min()) / 2 / eq[1]
        ok = sol.success and abs(amp_sim / amp_pred - 1) < 0.25
        ok_all &= ok
        lines.append(f"        {b['kind']}, {label}: predicted relative amplitude {amp_pred:.3f}, integrated {amp_sim:.3f} [{'ok' if ok else 'FAIL'}]")
    print("\n      amplitude just inside the unstable side, six supercritical boundaries")
    for ln in lines:
        print(ln)
    report("L2 the cycle born at a supercritical boundary has the size the coefficient predicts", ok_all and len(lines) >= 4, f"{len(lines)} cases")
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
