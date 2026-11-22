"""Density dependence spread over the stages of a loop, against the same total in one stage. No stochastic simulation.

Dobson (2020, p. 64) argues that a parasite with n hosts has a reproduction number per stage of R0^(1/n), close to
one, "and thus ... rather stable dynamics". In a loop the nth root is a convention; the argument has content if
density dependence acts at every stage in proportion to that stage's share of ln R0.

Model: n stages in continuous time, dx_{k+1}/dt = b nu f(x_k) x_k - a x_{k+1}, a = nu + mu, f(x) = exp(-x/K)
(over-compensating), R0 = (b nu / a)^n.

Claims, tolerances stated beforehand:
  S1  Regulation in every transfer: the equilibrium is x_k = K ln(R0)/n and the eigenvalues of the Jacobian
      (computed by finite differences from the nonlinear right-hand side) are a[(1 - ln(R0)/n) w - 1], w the nth
      roots of unity (largest absolute difference below 1e-5 a).
  S2  So for ln R0 <= n the return to equilibrium is at rate (ln R0) a/n = ln R0 / tau exactly (tau = n/a the
      generation time), whatever n; and the equilibrium is unstable exactly when ln R0 > 2n (n even) or
      ln R0 > n (1 + sec(pi/n)) (n odd, n >= 3); a one-stage cycle is always stable. Tested on the sign of the
      largest real part either side of each threshold (0.5% below and above), n = 2 to 9.
  S3  Regulation in one transfer only (kappa = ln R0 there): unstable exactly when ln R0 > 1 + sec^n(pi/n), n >= 3
      (the result in notes/regulated-loop.md), tested the same way; and for every n from 3 to 9 that threshold is
      at or below the one for shared regulation, with equality only at n = 3.
  S4  Direct integration of the nonlinear model, n = 4, ln R0 = 6: with regulation shared among the four transfers
      the population settles (relative amplitude of the last quarter of the run below 1e-6); with the same ln R0
      regulated in one transfer it keeps oscillating (relative amplitude above 0.1). The thresholds are 8 and 5.

  S5  (added about 04:30, after primary's blind derivation arrived) primary's functions threshold_A, threshold_B,
      eigenvalues_A, eigenvalues_B (studies/shared-regulation/shared.py) equal ours: thresholds for n = 1 to 12
      (relative 1e-12, or both infinite); eigenvalues at 200 random (n, ln R0) as sets (1e-9 a). And primary's
      alternating steady state for even n: with per-stage strength s = ln R0 / n > 2 and d the positive root of
      d = s tanh(d/2), the levels K(s - d), K(s + d) alternately are an equilibrium of our nonlinear right-hand side
      (residual below 1e-10), and for s = 2.2 its Jacobian has all eigenvalues in the left half-plane (n = 2, 4, 6, 8).

  S6  (added about 06:05 on 14 November) Odd rings just beyond the critical value: a small stable oscillation.
      For n = 3 and n = 5 with shared regulation, at ln R0 = critical x 1.02 and x 1.08, the nonlinear model started
      very near the symmetric equilibrium and started far from it settles to an oscillation of the same amplitude
      (relative difference below 1%), and the amplitude at the larger distance is about twice that at the smaller
      (ratio between 1.7 and 2.3: the square-root law of a supercritical Hopf bifurcation). This is numerical
      support, for two ring sizes, for the referees' calculation that the first Lyapunov coefficient is negative.

  S7  (added 06:13 on 14 November) The first Lyapunov coefficient of that bifurcation, computed here from the
      standard formula (Kuznetsov, Elements of Applied Bifurcation Theory, section 5.4): with A the Jacobian at the
      critical value, A q = i w q, A' p = -i w p, <p, q> = 1, and B, C the second and third derivatives,
          l1 = (1/(2w)) Re[ <p, C(q,q,q*)> - 2 <p, B(q, A^-1 B(q,q*))> + <p, B(q*, (2iw - A)^-1 B(q,q))> ].
      In units a = K = 1 and with the components of q of modulus one, claims: (i) l1 < 0 for every odd n from 3 to
      31; (ii) l1 equals the fifth referee's expression N/(2 tan(pi/n)), N = [-2 - 11u - 6u^2 - 5u^3 - 2u^4(1 - u)] /
      [q^2 (q + 1)(5q - 4)], q = sec(pi/n), u = q - 1 (relative 1e-8; the referee's values -0.096225044865 at n = 3
      and -0.462090907898 at n = 5 to 1e-9); (iii) the amplitude it predicts at 2% beyond the critical value,
      2 sqrt(alpha/(w |l1|)) with alpha the real part of the critical pair, is within 6% of the amplitude found by
      integrating the nonlinear model (n = 3, 5).
      (iv) (added 06:35, after the sixth referee report gave the steps of its derivation) the referee's intermediate
      expression, Re{ xi [(2 - q) + 2(q - 1)^2/(q + 1)] + (q - 1)^2 xi^3 / (2 i w + 1 + q xi^2) } with
      xi = -(1 + i w)/q and w = tan(pi/n), equals N for the same n (relative 1e-10).

Run: python3 -I checks/r5c_shared_regulation.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

NU, MU, K = 1.0, 0.5, 1.0
A = NU + MU


def rhs_factory(n, ln_r0, shared):
    b = math.exp(ln_r0 / n) * A / NU

    def rhs(t, x):
        out = np.empty(n)
        for k in range(n):
            reg = math.exp(-x[k] / K) if (shared or k == 0) else 1.0
            out[(k + 1) % n] = b * NU * reg * x[k] - A * x[(k + 1) % n]
        return out
    return rhs


def equilibrium(n, ln_r0, shared):
    if shared or n == 1:          # with one stage the two versions are the same model
        return np.full(n, K * ln_r0 / n)
    b = math.exp(ln_r0 / n) * A / NU
    x = np.empty(n); x[0] = K * ln_r0
    x[1] = b * NU * math.exp(-x[0] / K) * x[0] / A
    for k in range(1, n - 1):
        x[k + 1] = b * NU * x[k] / A
    return x


def jac_eigs(n, ln_r0, shared):
    f = rhs_factory(n, ln_r0, shared); x0 = equilibrium(n, ln_r0, shared)
    assert np.max(np.abs(f(0, x0))) < 1e-9 * max(1.0, np.max(x0))
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-6 * max(1.0, abs(x0[j])); e = np.zeros(n); e[j] = h
        J[:, j] = (f(0, x0 + e) - f(0, x0 - e)) / (2 * h)
    return np.linalg.eigvals(J)


def main():
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    rng = np.random.default_rng(2026111410)
    worst, cnt = 0.0, 0
    for trial in range(300):
        n = int(rng.integers(1, 10)); ln_r0 = rng.uniform(0.2, 3 * n)
        ev = jac_eigs(n, ln_r0, True)
        pred = A * ((1 - ln_r0 / n) * np.exp(2j * np.pi * np.arange(n) / n) - 1)
        d = max(np.min(np.abs(pred - e)) for e in ev)
        worst = max(worst, d / A); cnt += 1
    report("S1 shared regulation: eigenvalues a[(1 - ln R0/n) w - 1]", worst < 1e-5, f"{cnt} random cases, n from 1 to 9: largest difference {worst:.1e} a")

    ok, rows = True, []
    for n in range(1, 10):
        for ln_r0 in (0.3 * n, 0.9 * n):
            rate = -np.max(jac_eigs(n, ln_r0, True).real)
            ok &= abs(rate - ln_r0 * A / n) < 1e-5
        if n == 1:
            ok &= np.max(jac_eigs(1, 50.0, True).real) < 0
            continue
        thr = 2 * n if n % 2 == 0 else n * (1 + 1 / math.cos(math.pi / n))
        below = np.max(jac_eigs(n, 0.995 * thr, True).real); above = np.max(jac_eigs(n, 1.005 * thr, True).real)
        ok &= below < 0 < above; rows.append(f"{n}: {thr:.2f}")
    report("S2 shared regulation: return rate ln R0/tau for ln R0 <= n; unstable above ln R0 = 2n (n even), n(1 + sec(pi/n)) (n odd)", ok, "thresholds in ln R0 by n: " + ", ".join(rows))

    ok, rows = True, []
    for n in range(3, 10):
        thr1 = 1 + 1 / math.cos(math.pi / n) ** n
        below = np.max(jac_eigs(n, 0.995 * thr1, False).real); above = np.max(jac_eigs(n, 1.005 * thr1, False).real)
        thr_sh = 2 * n if n % 2 == 0 else n * (1 + 1 / math.cos(math.pi / n))
        ok &= below < 0 < above and thr1 <= thr_sh + 1e-9 and ((abs(thr1 - thr_sh) < 1e-9) == (n == 3))
        rows.append(f"{n}: {thr1:.2f} against {thr_sh:.2f}")
    report("S3 one regulated transfer: unstable above ln R0 = 1 + sec^n(pi/n), never above the shared threshold", ok, "one transfer against shared, by n: " + ", ".join(rows))

    amps = []
    for shared in (True, False):
        n, ln_r0 = 4, 6.0
        x0 = equilibrium(n, ln_r0, shared) * (1 + 0.05 * np.array([1, -1, 0.5, -0.5]))
        sol = solve_ivp(rhs_factory(n, ln_r0, shared), (0, 600), x0, rtol=1e-10, atol=1e-13, dense_output=True)
        tt = np.linspace(450, 600, 6000); tot = sol.sol(tt).sum(axis=0)
        amps.append((tot.max() - tot.min()) / tot.mean())
    report("S4 nonlinear model, n = 4, ln R0 = 6: shared regulation settles, one-stage regulation oscillates", amps[0] < 1e-6 and amps[1] > 0.1, f"relative amplitude at the end of the run {amps[0]:.1e} (shared) and {amps[1]:.2f} (one transfer)")

    import importlib.util
    import pathlib
    from scipy import optimize
    spec = importlib.util.spec_from_file_location("primary_shared", pathlib.Path(__file__).resolve().parent.parent / "studies" / "shared-regulation" / "shared.py")
    cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
    ok = True
    for n in range(1, 13):
        mine_a = math.inf if n < 3 else 1 + 1 / math.cos(math.pi / n) ** n
        mine_b = math.inf if n == 1 else (2.0 * n if n % 2 == 0 else n * (1 + 1 / math.cos(math.pi / n)))
        for theirs, mine in ((cs.threshold_A(n), mine_a), (cs.threshold_B(n), mine_b)):
            ok &= (math.isinf(theirs) and math.isinf(mine)) or abs(theirs / mine - 1) < 1e-12
    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(1, 10)); ln_r0 = rng.uniform(0.2, 3 * n)
        for shared_reg, fn in ((True, cs.eigenvalues_B), (False, cs.eigenvalues_A)):
            ours = jac_eigs(n, ln_r0, shared_reg); theirs = fn(n, ln_r0, A)
            worst = max(worst, max(np.min(np.abs(theirs - e)) for e in ours) / A)
    res, stable = 0.0, True
    for n in (2, 4, 6, 8):
        s = 2.2; ln_r0 = s * n
        d = optimize.brentq(lambda x: x - s * math.tanh(x / 2), 1e-6, s)
        x0 = np.array([K * (s - d) if k % 2 == 0 else K * (s + d) for k in range(n)])
        f = rhs_factory(n, ln_r0, True)
        res = max(res, float(np.max(np.abs(f(0, x0)))))
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x0[j])); e = np.zeros(n); e[j] = h
            J[:, j] = (f(0, x0 + e) - f(0, x0 - e)) / (2 * h)
        stable &= np.max(np.linalg.eigvals(J).real) < 0
    report("S5 primary's blind thresholds and eigenvalues = ours; its alternating steady state exists and is stable just above the threshold", ok and worst < 1e-9 and res < 1e-10 and stable,
           f"thresholds n = 1 to 12 identical: {ok}; eigenvalues, 200 random cases, largest difference {worst:.1e} a; alternating state residual {res:.1e}, stable for n = 2, 4, 6, 8: {stable}")

    ok6, rows6 = True, []
    for n in (3, 5):
        thr = n * (1 + 1 / math.cos(math.pi / n)); amp = []
        for eps in (0.02, 0.08):
            ln_r0 = thr * (1 + eps); x0 = equilibrium(n, ln_r0, True)
            growth = float(np.max(jac_eigs(n, ln_r0, True).real)); tend = 60 / growth
            pair = []
            for scale in (1e-3, 0.3):
                y0 = x0 * (1 + scale * np.cos(2 * np.pi * np.arange(n) / n + 0.3))
                sol = solve_ivp(rhs_factory(n, ln_r0, True), (0, tend), y0, rtol=1e-9, atol=1e-12, dense_output=True)
                xs = sol.sol(np.linspace(0.9 * tend, tend, 4000))[0]
                pair.append((xs.max() - xs.min()) / (2 * x0[0]))
            ok6 &= abs(pair[0] / pair[1] - 1) < 0.01 and pair[0] > 0.05
            amp.append(pair[0])
        ratio = amp[1] / amp[0]; ok6 &= 1.7 < ratio < 2.3
        rows6.append(f"n = {n}: relative amplitude {amp[0]:.3f} and {amp[1]:.3f} at 2% and 8% beyond the critical value (ratio {ratio:.2f})")
    report("S6 odd rings just beyond the critical value settle to a small oscillation whose amplitude follows a square-root law", ok6, "; ".join(rows6))

    def lyapunov(n):
        q_ = 1 / math.cos(math.pi / n); s = 1 + q_; w = math.tan(math.pi / n)
        Pm = np.roll(np.eye(n), 1, axis=0)                      # (Pm v)[k+1] = v[k]
        Am = -np.eye(n) + (1 - s) * Pm
        g2, g3 = s - 2.0, 3.0 - s                               # derivatives of x exp(-x) c at the equilibrium, c exp(-s) = 1
        ev, V = np.linalg.eig(Am); i = int(np.argmin(np.abs(ev - 1j * w))); qv = V[:, i] / np.abs(V[0, i])
        evl, W = np.linalg.eig(Am.T); j = int(np.argmin(np.abs(evl + 1j * w))); pv = W[:, j]
        pv = pv / np.conj(np.vdot(pv, qv))                      # <p, q> = sum conj(p) q = 1
        Bf = lambda a, b: g2 * (Pm @ (a * b)); Cf = lambda a, b, c: g3 * (Pm @ (a * b * c))
        t1 = np.vdot(pv, Cf(qv, qv, np.conj(qv)))
        t2 = np.vdot(pv, Bf(qv, np.linalg.solve(Am, Bf(qv, np.conj(qv)))))
        t3 = np.vdot(pv, Bf(np.conj(qv), np.linalg.solve(2j * w * np.eye(n) - Am, Bf(qv, qv))))
        res = float(max(abs(ev[i] - 1j * w), abs(np.vdot(pv, qv) - 1), np.max(np.abs(np.abs(qv) - 1))))
        return float((t1 - 2 * t2 + t3).real / (2 * w)), res
    worst7, neg, resid = 0.0, True, 0.0
    vals = {}
    for n in range(3, 32, 2):
        l1, res = lyapunov(n); vals[n] = l1; resid = max(resid, res)
        q_ = 1 / math.cos(math.pi / n); u = q_ - 1
        ref = (-2 - 11 * u - 6 * u ** 2 - 5 * u ** 3 - 2 * u ** 4 * (1 - u)) / (q_ ** 2 * (q_ + 1) * (5 * q_ - 4)) / (2 * math.tan(math.pi / n))
        worst7 = max(worst7, abs(l1 / ref - 1)); neg &= l1 < 0
    worst_mid = 0.0
    for n in range(3, 32, 2):
        q_ = 1 / math.cos(math.pi / n); w = math.tan(math.pi / n); u = q_ - 1; xi = -(1 + 1j * w) / q_
        mid = (xi * ((2 - q_) + 2 * (q_ - 1) ** 2 / (q_ + 1)) + (q_ - 1) ** 2 * xi ** 3 / (2j * w + 1 + q_ * xi ** 2)).real
        rat = (-2 - 11 * u - 6 * u ** 2 - 5 * u ** 3 - 2 * u ** 4 * (1 - u)) / (q_ ** 2 * (q_ + 1) * (5 * q_ - 4))
        worst_mid = max(worst_mid, abs(mid / rat - 1))
    amp_ok, amp_rows = True, []
    for n in (3, 5):
        thr = n * (1 + 1 / math.cos(math.pi / n)); ln_r0 = thr * 1.02; x0 = equilibrium(n, ln_r0, True)
        alpha = float(np.max(jac_eigs(n, ln_r0, True).real)) / A; w = math.tan(math.pi / n)
        pred = 2 * math.sqrt(alpha / (w * abs(vals[n]))) * K / x0[0]
        tend = 60 / (alpha * A); y0 = x0 * (1 + 1e-3 * np.cos(2 * np.pi * np.arange(n) / n + 0.3))
        sol = solve_ivp(rhs_factory(n, ln_r0, True), (0, tend), y0, rtol=1e-9, atol=1e-12, dense_output=True)
        xs = sol.sol(np.linspace(0.9 * tend, tend, 4000))[0]; got = (xs.max() - xs.min()) / (2 * x0[0])
        amp_ok &= abs(got / pred - 1) < 0.06
        amp_rows.append(f"n = {n}: predicted relative amplitude {pred:.3f}, integrated {got:.3f}")
    report("S7 the first Lyapunov coefficient of the odd-ring bifurcation, computed from the standard formula, is negative and equals the referee's expression",
           neg and worst7 < 1e-8 and abs(vals[3] + 0.096225044865) < 1e-9 and abs(vals[5] + 0.462090907898) < 1e-9 and resid < 1e-9 and amp_ok and worst_mid < 1e-10,
           f"odd n = 3 to 31: all negative: {neg}; largest relative difference from the referee's expression {worst7:.1e}; l1(3) = {vals[3]:.12f}, l1(5) = {vals[5]:.12f}; intermediate expression against the rational one {worst_mid:.1e}; " + "; ".join(amp_rows))

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
