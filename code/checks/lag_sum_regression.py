"""Regression test for the sum over lags in engine/theory.py (lineage_cost, lineage_cost_shared).

Claims, tolerances stated beforehand:
  G1  primary's counterexample (second referee report on the manuscript, point 11): a matrix similar to the
      four-stage ring with staying share 1/2, entry (1, 0), autocorrelation 1/2 at lag 10 and zero at every other
      positive lag. The excess of returns is exactly zero at lag 9 and -1/128 at lag 10. Correct value
      0.00830078125 (absolute 1e-12); the rule in force until 14 November returned 0.0078125.
  G2  The same through lineage_cost_shared with the one entry (absolute 1e-12).
  G3  For an autocorrelation phi^l the lag sum equals the published matrix formula, theory.matrix_cost, on 300
      random primitive matrices including slowly mixing ones (relative 1e-9).
  G4  An autocorrelation that is non-zero only at one lag L, for every L from 1 to 40, on the ring of G1: the lag
      sum equals c^2/2 - c rho [u(L) - c] computed from the matrix power directly (absolute 1e-13). This is the
      family in which a stop on a single small term goes wrong.

  G5  (added after the third referee report) A matrix and a badly scaled copy of it give the same answer. primary's
      example: P = pi 1' + 0.99 S + 0.8 (I - pi 1' - S), pi = (0.49, 0.49, 0.02), S the projector on (1, -1, 0)/sqrt 2;
      entry (0, 0); autocorrelation 0.99^l; analytic coefficient c^2/2 - c 0.993 [0.5 phi/(1 - 0.99 phi)
      + 0.01 phi/(1 - 0.8 phi)] with c = 0.993 * 0.49, about -11.923. Both P and D P D^-1 with D = diag(1, 1, 1e16)
      must give it (relative 1e-9), through both routines. The repair of 04:10 gave -11.271 for the scaled copy.
  G6  Rescaling invariance in general: 200 random primitive matrices and random diagonal rescalings over 12 orders
      of magnitude, a random entry and autocorrelation phi^l with phi up to 0.97: the lag sum of the rescaled
      matrix equals that of the original (relative 1e-8).

  G7  The cap is honoured and reported: with lags = 4 the referee's first example raises, and with lags = L it
      uses exactly L terms before raising (a cap of L used to allow L - 1).
  G8  The declared tail bound is real: for 100 random primitive matrices and autocorrelation phi^l, the lag sum
      stopped at tolerance 1e-6 differs from the resolvent value of sum_l phi^l [u(l) - c] by less than 1e-6
      (the difference is measured on that sum, before it is multiplied by c).

  G9  (added after the fourth referee report) An extremely badly scaled copy, D P D^-1 with
      D = diag(1e-120, 1, 1e120), for the referee's 3 x 3 stochastic matrix, entry (0, 0), autocorrelation 0.5^l:
      the routine either returns the value for P itself (relative 1e-8) or raises; it never returns another number.
      (Before the check on the eigenvectors it returned -0.0438 where the answer is -0.00092.)
  G10 With relative sizes kappa = 100 on one entry, the shared-driver routine at tol = 1e-4 returns the cost
      within 1e-4 (absolute) of the resolvent value: the tolerance bounds the error of the cost, not only of the
      lag sum. (The referee's case: 0.0054 off before the tolerance was scaled.)

  G11 (added after the fifth referee report; corrected after the sixth, which pointed out that my first version
      of this test used a different matrix and entry from the referee's and quoted the referee's answer for it)
      The referee's case: A = [[1 - 1e-16, 0.5], [1e-16, 0.5]], the staying entry (1, 1), relative size 1e8,
      autocorrelation 0.5^l. Its cost is -0.33333333333333314 (the referee's 90-digit value). The routine must
      return that to 1e-6 or raise; before the guard it returned 0. As first written here the test used
      [[1, 0.5], [1e-16, 0.5]] with entry (1, 0), whose cost is +1.17e-16, and called it -1/3; it passed because the
      routine raised, but it did not test the number it printed.
  G12 (added after the sixth referee report) Cancellation with a very large relative size:
      [[0.5, 0.5], [0.5, 0.5]], entry (0, 0), relative size 1e8, autocorrelation 0.5^l. The cost is exactly 0
      (c = 1/4, u(1) - c = 1/4, u(l) = c for l >= 2: 1/32 - (1/4)(1/2)(1/4) = 0). The routine must return a value
      within 1e-6 of 0 or raise; before the rounding guard it returned 0.1875. With relative size 1 it returns a
      value within 1e-12 of 0, and with relative size 100 a value within 1e-8 of 0 or an error.
  G13 (same report) Ordinary models are not refused and are computed correctly: rings with stays (1, 1000),
      (10, 1000), (1000, 1000), a 20-stage ring of stays 2, and a 20-stage ring with stays spaced geometrically
      from 1 to 1000 (one fluctuating transfer, phi = 0.5, against theory.cost_transfer, absolute 1e-10); and
      30-age Leslie matrices with fertility and survival (0.2, 0.8) and (0.5, 0.5), all fertilities on one series,
      against the resolvent formula theory.matrix_cost (relative 1e-8).
  G14 (same report) A documented limit, not a virtue: the 30-age Leslie matrix with every fertility 1 and every
      survival 1/2, whose smallest stationary weight is 6.5e-15, is declined by the cutoff at 1e-12 although its
      cost is computable (the referee did so by recursion). The test records that the routine raises and does not
      return a number.

  G15 (added after the seventh referee report) A nearly absorbing state above the cutoff of G11:
      A = [[0.9998, 4e-16], [0.0002, 1 - 4e-16]], entry (0, 0), relative size 1e6, autocorrelation 0.5^l. For the
      exact binary entries the cost is -1.99880031992734 (the referee's 90-digit value; the stationary weight of
      state 0 is 2.0e-12, above the cutoff, and the eigensolver returns 2.2e-12). The routine returned -2.2190, 11%
      off. It must now return the right value to 1e-6 relative, or raise; and the same for the model with its two
      states renamed, which before the guard happened to return the right value. Also: the single-entry routine at
      relative size 1 must return a value within 1e-6 relative of -1.99880031992734e-12 or raise.

  G16 (added after the eighth referee check) The repeated squaring added for G15 is itself wrong for
      A = [[1, 2^-26], [2^-64, 1 - 2^-52]]: the second component of the right vector is 1.89990899441424872736e-6
      (the referee's 100-digit value); the eigen-solver has it right and the squaring gives 1.9055e-6, 0.29% off.
      perron must return that component to 1e-9 relative (it now arbitrates in multiple precision).
  G17 (same check) A = [[0.8, 1e-116], [1e103, 1]], entry (0, 0), relative size 1e4, autocorrelation 0.9^l: the
      cost is -0.00051428571427837 (the referee's value, the same at 100, 200 and 300 digits). The routine returned
      -0.00050818, 1.19% off, with every guard passing. It must return the right value to 1e-6 relative, or raise;
      the same with the two states renamed.

  G18 (added after the ninth referee check, report 10, point 6) The three-state matrix found by the referee's
      search, with entries over 471 orders of magnitude, entry (0, 0), relative size 1e4, autocorrelation 0.9^l.
      The cost is -0.50344709658782949 (the referee's value, the same at 100, 200 and 300 digits). The routine
      returned -0.5034465107738804, 1.16e-6 off in relative terms, with every guard passing: the eigenvalues of the
      matrix as given and of the balanced matrix differed by just over 1e-9, so the balanced solution was returned
      on its residual alone, and one component of its right vector was 1.16e-6 off. It must return the right value
      to 1e-9 relative, or raise.
  G19 (same check, point 2) What a float cannot hold: A = [[1.5, 1.5], [z, z]], z the smallest positive float.
      The right vector is (1.5, z) / (1.5 + z) and the left one (1, 1). The second component of the right vector,
      about 3.29e-324, is not a float: the nearest one is z itself, 50% off, and that is what the multiple-precision
      routine returned. _perron_high_precision must return None, and perron must raise or return both vectors to
      1e-9 relative in every component (compared in exact rational arithmetic). Raising is the expected outcome.
  G20 (same check, point 3) _vectors_agree must return False when a NaN is put in the left vector of its first
      argument and everything else agrees. It returned True, because Python's max(0.0, nan) is 0.0. The test also
      requires True for two identical valid arguments, so that it cannot pass by always refusing.
  G21 (same check, point 3) A = z [[1, 2], [3, 4]], z as in G19: the double-precision computations agree on the
      eigenvalue 5 z, where the exact one is (5 + sqrt 33) z / 2 = 5.37 z, which no float holds to better than
      6.9%. lineage_cost (entry (0, 0), autocorrelation 0.5^l) must raise and not return a number.
      (Until 11:10 perron returned the eigenvalue 5 z for it and the refusal came from the later test of the line
      of descent. Since 11:10 perron itself raises: the squaring now forms the eigenvalue as w'Av / w'v and gets
      4 z, so the three double-precision computations no longer agree. The printed line changed with that.)

  G22 (added 11:10 on 14 November, with the squaring of A + c I) A matrix that is irreducible but not primitive:
      the ring of three stages with no staying and transfers 0.5, 2 and 3 (period 3). From 09:47 it was refused,
      because the powers of such a matrix never become positive. perron must return lambda = (0.5 * 2 * 3)^(1/3),
      the right vector proportional to (1, g_0/lambda, g_0 g_1/lambda^2) and the left one w_k = 1/(3 v_k) (each
      stage carries a third of the line of descent), all to 1e-12 relative; and so must the multiple-precision
      routine by itself, which now squares A + c I.
  G23 (same change) A reducible matrix is still refused: the three-age life table with fertilities (0.5, 1, 0)
      and survivals (0.5, 0.5), whose last age is not fertile. perron must raise, and both squaring routines must
      return None; it must not return vectors. (Until 09:47 the eigen-solver's answer was returned for such a
      matrix without any check; that answer was right here, with reproductive value zero for the last age.)
  G24 (same change) A life table whose fertile ages share a divisor: four ages, fertile at ages 2 and 4 only
      (fertilities 1 and 1.5, survivals 0.5, 0.6, 0.7), period 2. perron must return what the module returned
      before 09:47 (lambda = 0.9297366758239806 and the vectors recorded below, from the eigen-solver alone) to
      1e-12 relative, and lambda must equal the root of the Euler-Lotka equation, a quadratic in lambda^-2, to
      1e-12. The same table with a fifth age that is not fertile is reducible and must be refused, as in G23.
      (The request for this test spoke of five ages with ages 2 and 4 fertile; that table is the reducible one.)

  G25 (added 11:22 on 14 November, with the sixth condition of the fast class of theory.perron) Matrices that are
      perfectly conditioned but whose entries are tiny or huge must be outside the fast class BY THE CONDITION ON
      THE SIZE OF THE ENTRIES (every positive entry between 1e-150 and 1e150), and not by an accident of
      rounding: z [[1, 2], [3, 4]] for z = 1e-160, 1e-300 and 1e160. For each, that condition must be the only one
      of the six that fails, perron must then return the eigenvalue (5 + sqrt 33) z / 2 to 1e-12 relative (it
      arbitrates in multiple precision; these numbers are ordinary floats), and the same matrix with z = 1 must
      be inside the class. For the z of G21, the smallest positive float, the condition must fail too.

Run: python3 -I checks/lag_sum_regression.py
     python3 -I checks/lag_sum_regression.py --always-arbitrate
The second form sets theory.ALWAYS_ARBITRATE = True, which switches off the fast path of theory.perron: every
call is then arbitrated in multiple precision or refused. Every test must pass both ways.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import theory  # noqa: E402


def main(argv=()):
    if "--always-arbitrate" in argv:
        theory.ALWAYS_ARBITRATE = True
        print("(run with theory.ALWAYS_ARBITRATE = True: no fast path; every call of perron is arbitrated or refused)")
    rng = np.random.default_rng(2026111412)
    fails = 0

    def report(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    A = np.array([[0.5, 0, 0, 0.015625], [32, 0.5, 0, 0], [0, 0.0625, 0.5, 0], [0, 0, 2, 0.5]])
    rho10 = lambda l: 0.5 if l == 10 else 0.0
    v1 = theory.lineage_cost(A, (1, 0), rho10)
    report("G1 the referee's counterexample", abs(v1 - 0.00830078125) < 1e-12, f"{v1:.11f} (correct 0.00830078125; old rule 0.0078125)")
    v2 = theory.lineage_cost_shared(A, [(1, 0)], [1.0], rho10)
    report("G2 the same through the shared-driver routine", abs(v2 - 0.00830078125) < 1e-12, f"{v2:.11f}")

    worst, cnt = 0.0, 0
    for trial in range(300):
        d = int(rng.integers(2, 7))
        M = rng.uniform(0, 1, (d, d)) * (rng.random((d, d)) < 0.6)
        M[np.arange(d), (np.arange(d) - 1) % d] += rng.uniform(0.05, 1, d)          # a cycle, so irreducible
        M[0, 0] += rng.uniform(0.05, 0.99)                                            # a self-loop, so primitive
        if trial % 3 == 0:
            M[np.arange(d), np.arange(d)] += rng.uniform(5, 60, d)                    # slowly mixing
        nz = np.argwhere(M > 0); i, j = nz[int(rng.integers(len(nz)))]
        phi = rng.uniform(-0.9, 0.97)
        lin = theory.lineage_cost(M, (int(i), int(j)), lambda l: phi ** l)
        mat = theory.matrix_cost(M, [(int(i), int(j))], [0], [1.0], [phi])
        worst = max(worst, abs(lin - mat) / max(abs(mat), 1e-12)); cnt += 1
    report("G3 geometric autocorrelation: lag sum = published matrix formula", worst < 1e-9, f"{cnt} random primitive matrices: largest relative difference {worst:.1e}")

    lam, v, w = theory.perron(A); M = A / lam; c = M[1, 0] * w[1] * v[0]
    worst = 0.0
    for L in range(1, 41):
        direct = 0.5 * c * c - c * 0.5 * (M[1, 0] * np.linalg.matrix_power(M, L - 1)[0, 1] - c)
        got = theory.lineage_cost(A, (1, 0), lambda l, L=L: 0.5 if l == L else 0.0)
        worst = max(worst, abs(got - direct))
    report("G4 autocorrelation at a single lag, every lag from 1 to 40", worst < 1e-13, f"largest absolute difference {worst:.1e}")

    pi = np.array([0.49, 0.49, 0.02]); slow = np.array([[0.5, -0.5, 0], [-0.5, 0.5, 0], [0, 0, 0.0]])
    proj = np.outer(pi, np.ones(3))
    P = proj + 0.99 * slow + 0.8 * (np.eye(3) - proj - slow)
    scale = np.array([1.0, 1.0, 1e16]); AA = scale[:, None] * P / scale[None, :]
    cc = 0.993 * 0.49; phi = 0.99
    direct = cc * cc / 2 - cc * 0.993 * (0.5 * phi / (1 - 0.99 * phi) + 0.01 * phi / (1 - 0.8 * phi))
    vals = []
    for mat in (P, AA):
        vals.append(theory.lineage_cost(mat, (0, 0), lambda l: phi ** l))
        vals.append(theory.lineage_cost_shared(mat, [(0, 0)], [1.0], lambda l: phi ** l))
    worst = max(abs(x / direct - 1) for x in vals)
    report("G5 a matrix and a badly scaled copy give the same answer (the third referee's example)", worst < 1e-9, f"analytic {direct:.9f}; the four computed values {', '.join(f'{x:.9f}' for x in vals)}; largest relative difference {worst:.1e}")

    worst, cnt = 0.0, 0
    for trial in range(200):
        d = int(rng.integers(2, 6))
        M = rng.uniform(0, 1, (d, d)) * (rng.random((d, d)) < 0.6)
        M[np.arange(d), (np.arange(d) - 1) % d] += rng.uniform(0.05, 1, d)
        M[0, 0] += rng.uniform(0.05, 0.99)
        nz = np.argwhere(M > 0); i, j = nz[int(rng.integers(len(nz)))]
        sc = 10.0 ** rng.uniform(-6, 6, d); M2 = sc[:, None] * M / sc[None, :]
        ph = rng.uniform(0.0, 0.97)
        a1 = theory.lineage_cost(M, (int(i), int(j)), lambda l: ph ** l); a2 = theory.lineage_cost(M2, (int(i), int(j)), lambda l: ph ** l)
        worst = max(worst, abs(a2 - a1) / max(abs(a1), 1e-12)); cnt += 1
    report("G6 the lag sum does not change when the states are rescaled", worst < 1e-8, f"{cnt} random matrices rescaled over 12 orders of magnitude: largest relative difference {worst:.1e}")

    raised, used = False, 0
    try:
        theory.lineage_cost(A, (1, 0), rho10, lags=4)
    except RuntimeError:
        raised = True
    lam, v, w = theory.perron(A)
    try:
        for lag, pm, e in theory._centered_powers(A / lam, v, w, 7, 0.0):
            used = lag
    except RuntimeError:
        pass
    report("G7 the cap on the number of lags is honoured and reported", raised and used == 7, f"raises at a cap of 4: {raised}; terms used at a cap of 7: {used}")

    worst, cnt = 0.0, 0
    for trial in range(100):
        d = int(rng.integers(2, 6))
        M = rng.uniform(0, 1, (d, d)) * (rng.random((d, d)) < 0.6)
        M[np.arange(d), (np.arange(d) - 1) % d] += rng.uniform(0.05, 1, d)
        M[0, 0] += rng.uniform(0.05, 0.99)
        if trial % 2 == 0:
            M[np.arange(d), np.arange(d)] += rng.uniform(5, 40, d)
        nz = np.argwhere(M > 0); i, j = nz[int(rng.integers(len(nz)))]; i, j = int(i), int(j)
        ph = rng.uniform(0.3, 0.97)
        lam, v, w = theory.perron(M); m = M / lam; c = m[i, j] * w[i] * v[j]
        exact = (c * c / 2 - theory.matrix_cost(M, [(i, j)], [0], [1.0], [ph])) / c          # = sum_l phi^l [u(l) - c]
        loose = (c * c / 2 - theory.lineage_cost(M, (i, j), lambda l: ph ** l, tol=1e-6)) / c
        worst = max(worst, abs(loose - exact)); cnt += 1
    report("G8 the declared bound on the omitted tail holds", worst < 1e-6, f"{cnt} random matrices at tolerance 1e-6: largest omitted part {worst:.1e}")

    P3 = np.array([[.394633, .133951, .575350], [.039270, .501922, .007965], [.566097, .364127, .416685]])
    ref = theory.matrix_cost(P3, [(0, 0)], [0], [1.0], [0.5])
    plain = theory.lineage_cost(P3, (0, 0), lambda l: 0.5 ** l)
    dd = np.array([1e-120, 1.0, 1e120])
    with np.errstate(over="ignore", under="ignore"):
        A3 = dd[:, None] * P3 / dd[None, :]
    try:
        got = theory.lineage_cost(A3, (0, 0), lambda l: 0.5 ** l); outcome = f"returned {got:.10f}"
        ok9 = abs(got / ref - 1) < 1e-8
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:60]; ok9 = True
    report("G9 an extremely badly scaled copy: the right value or an error, never another number", ok9 and abs(plain / ref - 1) < 1e-8, f"reference {ref:.10f}; unscaled {plain:.10f}; scaled by 1e-120, 1, 1e120: {outcome}")

    Pk = np.array([[0.5, 0.5], [0.5, 0.5]]); kap = 100.0; phk = 0.9
    exact = theory.matrix_cost(Pk, [(0, 0)], [0], [kap], [phk])
    got = theory.lineage_cost_shared(Pk, [(0, 0)], [kap], lambda l: phk ** l, tol=1e-4)
    slow2 = np.array([[0.999, 0.001], [0.001, 0.999]])
    exact2 = theory.matrix_cost(slow2, [(0, 0)], [0], [kap], [0.999])
    got2 = theory.lineage_cost_shared(slow2, [(0, 0)], [kap], lambda l: 0.999 ** l, tol=1e-4)
    report("G10 the tolerance bounds the error of the cost when the relative size is large", abs(got - exact) < 1e-4 and abs(got2 - exact2) < 1e-4, f"kappa = 100: errors {abs(got - exact):.1e} and {abs(got2 - exact2):.1e} (slowly mixing chain, cost {exact2:.3f})")

    Ar = np.array([[1.0 - 1e-16, 0.5], [1e-16, 0.5]])
    try:
        got = theory.lineage_cost_shared(Ar, [(1, 1)], [1e8], lambda l: 0.5 ** l); outcome = f"returned {got!r}"; ok11 = abs(got + 0.33333333333333314) < 1e-6
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:70]; ok11 = True
    report("G11 the fifth referee's rare-state case (cost -1/3): the right value or an error", ok11 and Ar[0, 0] < 1.0, outcome)

    Pc = np.array([[0.5, 0.5], [0.5, 0.5]]); outs, ok12 = [], True
    for kap, lim in ((1.0, 1e-12), (100.0, 1e-8), (1e8, 1e-6)):
        try:
            got = theory.lineage_cost_shared(Pc, [(0, 0)], [kap], lambda l: 0.5 ** l); outs.append(f"kappa = {kap:g}: returned {got:.3e}"); ok12 &= abs(got) < lim
        except RuntimeError as exc:
            outs.append(f"kappa = {kap:g}: raised"); ok12 &= kap > 1.0
    report("G12 exact cancellation with a large relative size (cost 0): the right value or an error", ok12, "; ".join(outs))

    worst13, ok13 = 0.0, True
    rings = [[1.0, 1000.0], [10.0, 1000.0], [1000.0, 1000.0], [2.0] * 20, list(np.geomspace(1.0, 1000.0, 20))]
    for stays in rings:
        mm = 1 - 1 / np.asarray(stays); mm = np.where(mm <= 0, 0.0, mm); gg = 1 - mm; nst = len(stays)
        Aring = theory.ring_matrix(mm, gg)
        try:
            got = theory.lineage_cost(Aring, (1 % nst, 0), lambda l: 0.5 ** l, tol=1e-12) if nst > 1 else None
            ref = float(theory.cost_transfer(mm, gg, 1.0, 0.5)[0])
            worst13 = max(worst13, abs(got - ref))
        except RuntimeError as exc:
            ok13 = False; worst13 = float("inf")
    relworst = 0.0
    for fert, surv in ((0.2, 0.8), (0.5, 0.5)):
        L = np.zeros((30, 30)); L[0, :] = fert
        for a_ in range(29):
            L[a_ + 1, a_] = surv
        ents = [(0, a_) for a_ in range(30)]
        try:
            got = theory.lineage_cost_shared(L, ents, [1.0] * 30, lambda l: 0.5 ** l)
            ref = theory.matrix_cost(L, ents, [0] * 30, [1.0] * 30, [0.5])
            relworst = max(relworst, abs(got / ref - 1))
        except RuntimeError as exc:
            ok13 = False; relworst = float("inf")
    report("G13 ordinary rings and Leslie matrices are computed, not refused", ok13 and worst13 < 1e-10 and relworst < 1e-8,
           f"five rings (stays up to 1,000; 20 stages): largest absolute error {worst13:.1e}; two 30-age Leslie matrices: largest relative error {relworst:.1e}")

    L = np.zeros((30, 30)); L[0, :] = 1.0
    for a_ in range(29):
        L[a_ + 1, a_] = 0.5
    try:
        got = theory.lineage_cost_shared(L, [(0, a_) for a_ in range(30)], [1.0] * 30, lambda l: 0.5 ** l); outcome = f"returned {got!r}"; ok14 = False
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:90]; ok14 = True
    report("G14 a documented limit: a Leslie matrix with a stationary weight of 6.5e-15 is declined", ok14, outcome)

    ref15 = -1.99880031992734; outs, ok15 = [], True
    A15 = np.array([[0.9998, 4e-16], [0.0002, 1.0 - 4e-16]])
    for label, mat, ent in (("as given", A15, (0, 0)), ("states renamed", A15[::-1, ::-1].copy(), (1, 1))):
        try:
            got = theory.lineage_cost_shared(mat, [ent], [1e6], lambda l: 0.5 ** l); outs.append(f"{label}: returned {got:.10f}"); ok15 &= abs(got / ref15 - 1) < 1e-6
        except RuntimeError as exc:
            outs.append(f"{label}: raised")
    try:
        got = theory.lineage_cost(A15, (0, 0), lambda l: 0.5 ** l); outs.append(f"single entry: returned {got:.6e}"); ok15 &= abs(got / (ref15 * 1e-12) - 1) < 1e-6
    except RuntimeError as exc:
        outs.append("single entry: raised")
    report("G15 a nearly absorbing state above the cutoff (cost -1.9988): the right value or an error", ok15, "; ".join(outs))

    A16 = np.array([[1.0, 2.0 ** -26], [2.0 ** -64, 1.0 - 2.0 ** -52]])
    v16 = theory.perron(A16)[1]; ref16 = 1.89990899441424872736e-6
    report("G16 the vectors where the double-precision squaring is wrong and the eigen-solver right", abs(v16[1] / ref16 - 1) < 1e-9,
           f"second component of the right vector {v16[1]:.15e} (reference {ref16:.15e}; relative difference {abs(v16[1] / ref16 - 1):.1e})")

    ref17 = -0.00051428571427837; outs, ok17 = [], True
    A17 = np.array([[0.8, 1e-116], [1e103, 1.0]])
    for label, mat, ent in (("as given", A17, (0, 0)), ("states renamed", A17[::-1, ::-1].copy(), (1, 1))):
        try:
            got = theory.lineage_cost_shared(mat, [ent], [1e4], lambda l: 0.9 ** l); outs.append(f"{label}: returned {got:.14f}"); ok17 &= abs(got / ref17 - 1) < 1e-6
        except RuntimeError as exc:
            outs.append(f"{label}: raised")
    report("G17 entries over 219 orders of magnitude (cost -0.000514286): the right value or an error", ok17, "; ".join(outs))

    ref18 = -0.50344709658782949
    A18 = np.array([[0.5939806914404566, 2.2502409188273124e-87, 3.788658900425089e-241],
                    [1.8314892145979668e85, 0.5939806914404566, 3.4180088324058337e-155],
                    [6.790013138809317e230, 7.526324528028247e144, 0.9999999974656335]])
    try:
        got = theory.lineage_cost_shared(A18, [(0, 0)], [1e4], lambda l: 0.9 ** l); outcome = f"returned {got!r} (relative difference {abs(got / ref18 - 1):.1e})"; ok18 = abs(got / ref18 - 1) < 1e-9
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:70]; ok18 = True
    report("G18 the ninth referee check's three-state matrix (cost -0.503447097): the right value or an error", ok18, f"reference {ref18!r}; {outcome}")

    from fractions import Fraction
    z = float(np.nextafter(0.0, 1.0))
    A19 = np.array([[1.5, 1.5], [z, z]])
    hp19 = theory._perron_high_precision(A19)
    try:
        lam19, v19, w19 = theory.perron(A19)
        exact = [Fraction(1.5) / (Fraction(1.5) + Fraction(z)), Fraction(z) / (Fraction(1.5) + Fraction(z)), Fraction(1), Fraction(1)]
        got19 = [float(v19[0]), float(v19[1]), float(w19[0]), float(w19[1])]
        worst19 = max((abs(Fraction(g) / e - 1) if np.isfinite(g) else Fraction(1)) for g, e in zip(got19, exact))
        outcome = f"perron returned v = {v19!r}, w = {w19!r} (largest relative error {float(worst19):.1e})"; ok19 = worst19 < Fraction(1, 10 ** 9)
    except RuntimeError as exc:
        outcome = "perron raised: " + str(exc)[:90]; ok19 = True
    report("G19 a vector component that no float can hold: no multiple-precision result, and the right vectors or an error", ok19 and hp19 is None,
           f"_perron_high_precision returned {hp19!r}; {outcome}")

    good = (1.0, np.array([0.5, 0.5]), np.array([1.0, 1.0]))
    outs, ok20 = [], True
    for label, left in (("both components NaN", np.array([np.nan, np.nan])), ("one component NaN", np.array([1.0, np.nan]))):
        res = theory._vectors_agree((1.0, np.array([0.5, 0.5]), left), good); outs.append(f"{label}: {res}"); ok20 &= res is False
    same = theory._vectors_agree(good, good); ok20 &= same is True
    report("G20 a NaN in the left vector of the first argument is not agreement", ok20, "; ".join(outs) + f"; two identical valid arguments: {same}")

    A21 = z * np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        got = theory.lineage_cost(A21, (0, 0), lambda l: 0.5 ** l); outcome = f"returned {got!r}"; ok21 = False
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:90]; ok21 = True
    report("G21 a matrix of subnormal entries whose eigenvalue no float can hold: an error, not a number", ok21, outcome)

    g22 = np.array([0.5, 2.0, 3.0]); A22 = theory.ring_matrix(np.zeros(3), g22)
    lam22 = float(np.prod(g22)) ** (1.0 / 3.0)
    v22 = np.array([1.0, g22[0] / lam22, g22[0] * g22[1] / lam22 ** 2]); v22 = v22 / v22.sum(); w22 = 1.0 / (3.0 * v22)
    outs, ok22 = [], True
    for label, fn in (("perron", theory.perron), ("multiple precision alone", theory._perron_high_precision)):
        try:
            got = fn(A22)
        except RuntimeError as exc:
            got = None; outs.append(f"{label}: raised"); ok22 = False
        if got is None:
            outs.append(f"{label}: no answer"); ok22 = False
        else:
            worst = max(abs(got[0] / lam22 - 1), float(np.max(np.abs(got[1] / v22 - 1))), float(np.max(np.abs(got[2] / w22 - 1))))
            outs.append(f"{label}: lambda {got[0]!r}, largest relative difference from the known eigenvalue and vectors {worst:.1e}"); ok22 &= worst < 1e-12
    report("G22 a ring with no staying (irreducible, period 3): the known eigenvalue and vectors", ok22, f"known lambda {lam22!r}; " + "; ".join(outs))

    A23 = np.array([[0.5, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    helpers23 = (theory._perron_by_squaring(A23), theory._perron_high_precision(A23))
    try:
        got = theory.perron(A23); outcome = f"perron returned {got!r}"; ok23 = False
    except RuntimeError as exc:
        outcome = "perron raised: " + str(exc)[:90]; ok23 = True
    report("G23 a reducible matrix (a life table with an age after the last fertile one) is refused", ok23 and helpers23 == (None, None),
           f"double-precision squaring returned {helpers23[0]!r}, multiple precision {helpers23[1]!r}; {outcome}")

    A24 = np.array([[0.0, 1.0, 0.0, 1.5], [0.5, 0.0, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0]])
    old24 = (0.9297366758239806, np.array([0.46595193835237875, 0.25058274588309015, 0.1617120755149371, 0.12175324024959404]),
             np.array([0.7548494348797017, 1.4036224086653286, 0.9169163291080825, 1.2178439140481068]))
    p2, p4 = 0.5 * 1.0, 0.5 * 0.6 * 0.7 * 1.5                      # l_2 f_2 and l_4 f_4: 1 = p2 y + p4 y^2, y = lambda^-2
    euler24 = float(np.sqrt((p2 + np.sqrt(p2 * p2 + 4 * p4)) / 2))
    try:
        got = theory.perron(A24)
        worst = max(abs(got[0] / old24[0] - 1), float(np.max(np.abs(got[1] / old24[1] - 1))), float(np.max(np.abs(got[2] / old24[2] - 1))))
        outcome = f"lambda {got[0]!r}; largest relative difference from the values of the module before 09:47: {worst:.1e}; from the Euler-Lotka root: {abs(got[0] / euler24 - 1):.1e}"
        ok24 = worst < 1e-12 and abs(got[0] / euler24 - 1) < 1e-12
    except RuntimeError as exc:
        outcome = "raised: " + str(exc)[:70]; ok24 = False
    A24b = np.zeros((5, 5)); A24b[:4, :4] = A24; A24b[4, 3] = 0.8
    try:
        theory.perron(A24b); five = "returned"; ok24 = False
    except RuntimeError:
        five = "refused"
    report("G24 a life table fertile at ages 2 and 4 only (irreducible, period 2): the values of the old module", ok24,
           f"{outcome}; with a fifth age that is not fertile (reducible): {five}")

    base25 = np.array([[1.0, 2.0], [3.0, 4.0]]); exact25 = (5.0 + np.sqrt(33.0)) / 2.0
    outs, ok25 = [], True
    for scale in (1.0, 1e-160, 1e-300, 1e160, z):
        A25 = scale * base25
        raw25 = theory._perron_raw(A25); lamb25, sq25 = theory._double_checks(A25)
        cond = theory._fast_class_conditions(A25, raw25, sq25, lamb25); inside = theory._in_fast_class(A25, raw25, sq25, lamb25)
        failed = [k for k, held in cond.items() if not held]
        if scale == 1.0:
            ok25 &= inside and not failed; outs.append(f"z = 1: inside ({inside})")
        elif scale == z:
            ok25 &= (not inside) and "vi" in failed; outs.append(f"z = {scale:.1e}: outside, fails ({', '.join(failed)})")
        else:
            try:
                err = abs(theory.perron(A25)[0] / (exact25 * scale) - 1); after = f"perron's eigenvalue off by {err:.1e}"; ok25 &= err < 1e-12
            except RuntimeError:
                after = "perron raised"; ok25 = False
            ok25 &= (not inside) and failed == ["vi"]; outs.append(f"z = {scale:g}: outside, fails ({', '.join(failed)}) only; {after}")
    report("G25 tiny and huge entries are outside the fast class by the condition on their size", ok25, "; ".join(outs))

    print()
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
