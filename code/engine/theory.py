"""Second-order theory for life cycles in fluctuating environments: closed forms and the matrix formula.

Everything here is a formula, with no simulation. The matrix formula is the published one (Tuljapurkar 1982;
Tuljapurkar & Haridas 2006, eq. 8). The closed forms for a ring are derived in `notes/general-ring.md` and
agree with the matrix formula to rounding (`checks/general_ring_closed_forms.py`).

Conventions. A ring has n stages; s[k] is the mean staying in stage k and g[k] the mean transfer from stage k
to stage k+1 (indices mod n). lam is the dominant eigenvalue of the mean matrix. m[k] = s[k]/lam is the staying
share of stage k along a line of descent, tau[k] = 1/(1 - m[k]) its mean stay, and T = sum(tau) the generation
time. Costs are per unit of sigma^2 unless a variance is passed in: cost = ln(lam) - long-run growth rate.
"""
from __future__ import annotations

import numpy as np

# perron returns the eigen-solver's answer without the multiple-precision check for matrices in a declared class of
# well-conditioned inputs (_in_fast_class). Set this to True to switch that fast path off: every call is then
# arbitrated in multiple precision, or refused, as every call was between 09:47 and 11:10 on 14 November 2026.
ALWAYS_ARBITRATE = False

# What "agrees" means wherever perron sets the eigen-solver's answer beside that of a squaring routine, in double
# or in multiple precision, before returning the eigen-solver's: relative agreement of the eigenvalue, and of every
# component of both vectors. The second number was 1e-9 until 12:23 on 14 November 2026. A sample of 21,773
# matrices from the calls of two checks then showed the eigen-solver 5.3e-10 off in a component of size 6.5e-7 of
# a 12-stage ring on which both squarings were right to 1e-15; at 1e-9 every branch returned the eigen-solver's
# vectors for it. See the history in perron.
AGREE_EIGENVALUE = 1e-12
AGREE_COMPONENTS = 1e-11


def ring_matrix(s, g):
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    a = np.diag(s).astype(float)
    for k in range(n):
        a[(k + 1) % n, k] += g[k]
    return a


def _perron_raw(a):
    ev, vr = np.linalg.eig(a)
    i = int(np.argmax(ev.real))
    lam = float(ev[i].real)
    v = np.abs(vr[:, i].real)
    v /= v.sum()
    el, vl = np.linalg.eig(a.T)
    j = int(np.argmax(el.real))
    w = np.abs(vl[:, j].real)
    w /= w @ v
    return lam, v, w


def _perron_residual(a, lam, v, w):
    """Largest relative residual of A v = lam v and w'A = lam w', over the components that carry weight.

    Componentwise, so that it does not depend on the units of the states; components of v or w below 1e-6 of the
    largest are left out, because an eigen-solver cannot give them to high relative accuracy and they do not
    decide the eigenvalue. Zero for exact vectors; of order one when the solver has failed.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        rv = np.abs(a @ v - lam * v) / np.maximum(np.abs(a) @ np.abs(v), 1e-300)
        rw = np.abs(w @ a - lam * w) / np.maximum(np.abs(w) @ np.abs(a), 1e-300)
        rv = rv[np.abs(v) >= 1e-6 * np.abs(v).max()]
        rw = rw[np.abs(w) >= 1e-6 * np.abs(w).max()]
    r = max(float(np.nanmax(rv)), float(np.nanmax(rw)))
    return r if np.isfinite(r) else np.inf


def _shift_for_squaring(a):
    """A positive number c, for squaring A + c I in place of A (_perron_by_squaring, _perron_high_precision).

    (Added 11:10 on 14 November 2026.) A + c I has the same right and left dominant vectors as A, and its
    dominant eigenvalue is that of A plus c. It is primitive whenever A is irreducible, so its powers converge to a
    multiple of v w' also when A is irreducible but not primitive (a ring in which no stage has any staying, a life
    table whose fertile ages share a divisor), where the powers of A itself never do. Any c > 0 is valid in exact
    arithmetic. The choice decides how fast the squares converge, at the rate of (second eigenvalue + c) /
    (dominant eigenvalue + c): c should be of the size of the dominant eigenvalue, and a c very much larger stops
    the convergence within the doublings allowed, which makes the routine decline.

    The largest column sum of a non-negative matrix is an upper bound for its dominant eigenvalue and needs no
    eigen-solver. Two such bounds are computed here and the smaller is returned: the largest column sum of A as
    given, and that of A balanced by a diagonal change of units (scipy.linalg.matrix_balance; the balanced matrix
    is similar to A, so the bound holds for it too). The second is there because the first is not of the size of
    the eigenvalue when the states are badly scaled: for the matrix of regression G18 it is 6.8e230 where the
    eigenvalue is 1, the squares would then need about 770 doublings, and the multiple-precision routine, which
    allows 400, declines that matrix and the one of G9 (tried on 14 November with the column sum of A alone: None
    for both, where with the smaller bound, about 2 for both, it returns their vectors).

    Returns None if neither bound is finite and positive.
    """
    a = np.asarray(a, float)
    import warnings
    from scipy.linalg import matrix_balance
    bounds = []
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bounds.append(float(a.sum(axis=0).max()))
        try:
            bounds.append(float(matrix_balance(a, permute=False)[0].sum(axis=0).max()))
        except Exception:                                # balancing is a convenience here; the first bound stands
            pass
    bounds = [c for c in bounds if np.isfinite(c) and c > 0]
    return min(bounds) if bounds else None


def _perron_by_squaring(a, doublings=64):
    """Dominant eigenvalue and eigenvectors of an irreducible non-negative matrix by repeated squaring, in double
    precision. A SECOND OPINION on the eigen-solver's vectors, not a more accurate method.

    What is squared is B = A + c I, with c from _shift_for_squaring (since 11:10 on 14 November 2026; before that
    it was A itself, whose powers do not converge when A is irreducible but not primitive). B has the vectors of A.
    The eigenvalue is NOT taken as that of B minus c: it is computed from the matrix as given and the converged
    vectors, as w'Av / w'v, a sum of non-negative terms. In double precision the sum a_ii + c drops the low digits
    of a diagonal entry much smaller than c, so B is A with its diagonal slightly changed; when that matters the
    result disagrees with the eigen-solver's, which is what a second opinion is for. A reducible matrix is
    declined: with a positive diagonal the pattern of positive entries of B^k stops growing at k = n - 1, so if
    the power reached has a zero entry by then, no power is positive.

    B^(2^k), rescaled, tends to a multiple of v w'. Every entry of every square is a sum of products of
    non-negative numbers, so no single step cancels, and this route found the error of the eigen-solver in the
    seventh referee's example ([[0.9998, 4e-16], [0.0002, 1 - 4e-16]]: weight 2.0e-12 returned as 2.2e-12). But it
    carries no bound on its own error: errors in a matrix enter both factors of its square and can double at each
    step, a small spectral gap amplifies them, and products can fall into the subnormal range (eighth referee
    report, points 2, 3 and 7: for [[1, 2^-26], [2^-64, 1 - 2^-52]] it is 0.29% wrong where the eigen-solver is
    right; for [[0.8, 1e-116], [1e103, 1]] it was 1.2% wrong). As first written, at 07:25, this docstring claimed a
    relative accuracy of "(number of squarings) x n x eps, however small" a component is. That was false. From
    09:47 to 11:10 perron consulted this routine only when there was no multiple-precision result; since 11:10 it is
    also the cross-check of the fast path (_in_fast_class).

    Returns (lam, v, w) with v summing to one and w @ v = 1, or None if the squares do not become strictly positive
    with proportional columns and rows to 1e-13 within the allowed doublings, or if one of the range tests below
    fails. None is not a diagnosis of anything about the matrix.

    Range tests (changed 14 November 2026 after the ninth referee check, report 10, point 4). Which entries are
    positive in exact arithmetic is tracked apart from the computed numbers, as a pattern of True and False that is
    squared along with the matrix. Each square is inspected BEFORE it is rescaled: the routine declines if an entry
    that must be positive has underflowed to zero, is below 1e-280 in absolute value, or is below 1e-280 of the
    largest entry of that square. After each rescaling to sum one (the first is that of B itself) it
    declines if an entry that must be positive is zero or below 1e-280. As written at 07:47 the only test was the
    one after rescaling, on entries still positive, and this docstring said that any positive entry below 1e-280
    "on the way" caused a refusal. That was wider than what the code did: for [[0.8, 1e-116], [1e103, 1]] the first
    square has a smallest entry of 1.8e-322, rescaling raised it back to 9.9e-220, and the damaged square went
    through. (That was for the squares of A. With A + c I the same entry of the first square is 5.6e-322, and the
    example is declined there.) The tests cover underflow in the squares as inspected, and nothing else: they do
    not look at the products formed when the eigenvalue is computed from the matrix as given, and they are not a
    bound on the error of the result, which can still be wrong for the other reasons given above.
    """
    a0 = np.array(a, float)
    if a0.ndim != 2 or a0.shape[0] < 2 or np.any(a0 < 0) or not np.all(np.isfinite(a0)):
        return None
    nn = a0.shape[0]
    shift = _shift_for_squaring(a0)
    if shift is None:
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        b = a0 + shift * np.eye(nn)                        # the matrix that is squared: same vectors as a0
        pat = b > 0                                        # the entries that are positive in exact arithmetic
        power = 1
        for _ in range(doublings):
            tot = b.sum()
            if not (np.isfinite(tot) and tot > 0):
                return None
            b = b / tot
            if np.any(pat & (b < 1e-280)):
                return None                                # a positive entry lost, or where products lose digits
            if np.all(b > 0):
                cols = b / b.sum(axis=0, keepdims=True)
                rows = b / b.sum(axis=1, keepdims=True)
                if max(float(np.max(np.abs(cols / cols[:, :1] - 1.0))), float(np.max(np.abs(rows / rows[:1, :] - 1.0)))) < 1e-13:
                    v = cols[:, 0].copy(); w = rows[0, :].copy()
                    break
            elif power >= nn - 1:
                return None                                # reducible: see the docstring
            b = b @ b
            power *= 2
            pat = (pat.astype(float) @ pat.astype(float)) > 0          # the pattern of the exact square
            big = float(b.max())
            if not np.isfinite(big) or np.any(pat & ((b <= 0) | (b < 1e-280) | (b < 1e-280 * big))):
                return None                                # the square, before rescaling: a product lost or subnormal
        else:
            return None
        v = v / v.sum()
        wv = w @ v
        lam = float(w @ (a0 @ v) / wv)                     # from the matrix as given, not from the shifted one
        w = w / wv
    if not (np.isfinite(lam) and lam > 0 and np.all(np.isfinite(v)) and np.all(np.isfinite(w)) and np.all(v > 0) and np.all(w > 0)):
        return None
    return lam, v, w


def _perron_high_precision(a, max_n=40, doublings=400):
    """The same repeated squaring in multiple-precision arithmetic (mpmath). perron uses it as the arbiter of the
    double-precision computations whenever it is consulted and gives a result.

    Like the double-precision routine it squares B = A + c I (c from _shift_for_squaring, added exactly; since
    11:10 on 14 November 2026), so that it also converges for a matrix that is irreducible but not primitive. The
    eigenvalue returned is w'Av / w'v computed in multiple precision from the matrix as given, not that of B minus
    c. A reducible matrix is declined at once, from the pattern of its positive entries: no power of B is positive
    then, and the squares could never pass the test of convergence.

    Two different things limit it, and they must not be confused (ninth referee check, report 10, point 2).

    Precision inside the computation: 60 digits, plus the number of decimal orders spanned by the positive entries,
    plus 0.31 per allowed doubling. By the usual model of rounding error (not a certified bound) that is far more
    than the doubling of errors needs, and mpmath has no limit on the exponent, so the range of the entries does not
    cause underflow or overflow inside the computation.

    Representability of the answer as floats: the routine returns floats, and a float cannot hold a component that
    is subnormal or overflows, however many digits were used to compute it. For [[1.5, 1.5], [z, z]], z the
    smallest positive float, the second component of the right vector is z / (1.5 + z), about 3.29e-324; as first
    written this routine returned z = 4.94e-324 for it, 50% off, and the earlier version of this docstring promised
    that the range of the entries did not matter. So the eigenvalue and every component of both vectors are
    converted to floats and back, and the routine declines unless each one returns to within 4 machine epsilons
    (relative) of its multiple-precision value. More digits cannot repair such a case.

    Returns floats (lam, v, w), or None if mpmath is not installed, the matrix is larger than max_n, has a
    negative or non-finite entry or is reducible, the squares do not converge within the allowed doublings, or
    the result is not representable as just described. Apart from reducibility, None is not a diagnosis of anything
    about the matrix.
    """
    a = np.asarray(a, float)
    nn = a.shape[0]
    if a.ndim != 2 or nn < 2 or nn > max_n or np.any(a < 0) or not np.all(np.isfinite(a)):
        return None
    try:
        import mpmath as mp
    except ImportError:
        return None
    pos = a[a > 0]
    if pos.size == 0:
        return None
    reach = ((a > 0) | np.eye(nn, dtype=bool)).astype(float)       # which entries of (A + c I)^k are positive
    power = 1
    while power < nn - 1:
        reach = ((reach @ reach) > 0).astype(float); power *= 2
    if not np.all(reach > 0):
        return None                                        # reducible: no power of A + c I is positive
    shift = _shift_for_squaring(a)
    if shift is None:
        return None
    span = float(np.log10(pos.max()) - np.log10(pos.min()))
    with mp.workdps(int(60 + span + 0.31 * doublings)):
        a_mp = mp.matrix(a.tolist()); b = a_mp.copy()
        for i in range(nn):
            b[i, i] += mp.mpf(shift)                       # the matrix that is squared: same vectors as A
        tol = mp.mpf(10) ** (-30)
        v = w = None
        for _ in range(doublings):
            tot = mp.fsum(b)
            if tot <= 0:
                return None
            b = b / tot
            if all(b[i, j] > 0 for i in range(nn) for j in range(nn)):
                cs = [mp.fsum(b[i, j] for i in range(nn)) for j in range(nn)]
                rs = [mp.fsum(b[i, j] for j in range(nn)) for i in range(nn)]
                dc = max(abs(b[i, j] / cs[j] / (b[i, 0] / cs[0]) - 1) for i in range(nn) for j in range(nn))
                dr = max(abs(b[i, j] / rs[i] / (b[0, j] / rs[0]) - 1) for i in range(nn) for j in range(nn))
                if max(dc, dr) < tol:
                    v = [b[i, 0] / cs[0] for i in range(nn)]; w = [b[0, j] / rs[0] for j in range(nn)]
                    break
            b = b * b
        if v is None:
            return None
        sv = mp.fsum(v); v = [x / sv for x in v]
        wv = mp.fsum(w[i] * v[i] for i in range(nn))
        lam = mp.fsum(w[i] * a_mp[i, j] * v[j] for i in range(nn) for j in range(nn)) / wv    # from A as given
        w = [x / wv for x in w]
        exact = [lam] + v + w
        try:
            floats = [float(x) for x in exact]
        except OverflowError:
            return None
        # representability: each float must return to within 4 machine epsilons of the multiple-precision value,
        # so that nothing subnormal, rounded to zero or overflowed is returned
        back = 4 * mp.mpf(float(np.finfo(float).eps))
        if not all(np.isfinite(f) and f > 0 and abs(mp.mpf(f) - x) <= back * abs(x) for f, x in zip(floats, exact)):
            return None
        out = floats[0], np.array(floats[1:1 + nn]), np.array(floats[1 + nn:])
    if not (np.isfinite(out[0]) and out[0] > 0 and np.all(np.isfinite(out[1])) and np.all(np.isfinite(out[2])) and np.all(out[1] > 0) and np.all(out[2] > 0)):
        return None
    return out


def _vectors_agree(x, y, tol=1e-9, lam_tol=None):
    """Whether two results (lam, v, w) agree: the eigenvalues to lam_tol (relative; tol if not given) and every
    component of both vectors to tol (relative). False if any number in either result is not finite or not
    positive. Agreement of two computations is evidence and not a certificate: an error common to both passes.
    """
    lam_tol = tol if lam_tol is None else lam_tol
    # Finiteness and positivity are tested first, on every number of both arguments, and not left to the
    # comparison. Until 14 November 2026 a comment here said that a NaN component always made the result False.
    # That was wrong: Python's max(0.0, nan) is 0.0, so a NaN in the left vector of the first argument was passed
    # over when the right vectors agreed (ninth referee check, report 10, point 3).
    for part in (x[0], x[1], x[2], y[0], y[1], y[2]):
        part = np.asarray(part, float)
        if not (np.all(np.isfinite(part)) and np.all(part > 0)):
            return False
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        dv = float(np.max(np.abs(x[1] / y[1] - 1.0)))
        dw = float(np.max(np.abs(x[2] / y[2] - 1.0)))
    return bool(abs(x[0] - y[0]) <= lam_tol * abs(y[0]) and dv <= tol and dw <= tol)


def _double_checks(a):
    """The two double-precision computations that perron sets beside the eigen-solver's answer: the eigenvalue of
    the matrix balanced by a diagonal change of units (scipy.linalg.matrix_balance), and the repeated squaring
    (_perron_by_squaring; None when it declines). Returned as (lam_balanced, sq)."""
    import warnings
    from scipy.linalg import matrix_balance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = matrix_balance(a, permute=False)[0]
    return _perron_raw(b)[0], _perron_by_squaring(a)


def _fast_class_conditions(a, raw, sq, lam_balanced):
    """The six conditions of the fast class of perron, one by one, as a dictionary of True and False with the keys
    "i" to "vi". _in_fast_class is True when all six are. raw is the eigen-solver's answer (lam, v, w) for the
    matrix a; sq and lam_balanced come from _double_checks. What each condition is, and what it is there for:

      (i)   The dimension is at most 200. A limit of scope: nothing larger has been examined.
      (ii)  Every positive entry is at least 1e-8 times the largest. Keeps out the badly scaled matrices and the
            nearly absorbing or rarely used transitions on which the eigen-solver's vectors were found wrong
            (fourth, fifth, seventh, eighth and ninth referee reports). It compares the entries with each other
            only, and says nothing about their absolute size; that is (vi).
      (iii) The eigen-solver's eigenvalue and that of the balanced matrix agree to 1e-12 relative. The
            eigen-solver's own consistency under a change of units (it failed that outright in the fourth
            report, and by 1.3e-9 in the ninth).
      (iv)  The double-precision squaring gave an answer, and the eigen-solver agrees with it: the eigenvalue to
            1e-12 relative and every component of both vectors to 1e-11 relative (AGREE_EIGENVALUE,
            AGREE_COMPONENTS, _vectors_agree; both were 1e-9 until 12:23). The cross-check by a second algorithm,
            and since 12:23 the test that sends to the multiple-precision routine any matrix on which the
            eigen-solver's small components are not good to 1e-11. It also keeps out a matrix with a negative
            entry and a reducible one, on which the squaring declines.
      (v)   The smallest of the normalised products v_i w_i, the stationary weights of the line of descent, is at
            least 1e-10. Keeps out states so rare that their vector components have few correct digits (fifth
            and seventh reports).
      (vi)  Every positive entry lies between 1e-150 and 1e150 (added 11:22 on 14 November 2026). Conditions (ii)
            to (v) are relative, or are computed from the matrix itself, so a matrix all of whose entries are
            subnormal, tiny or huge can pass them: z * [[1, 2], [3, 4]] with z the smallest positive float passed
            (ii), (iii) and (v), and failed (iv) only by an accident of rounding (ninth report, point 3;
            regressions G21 and G25). With (vi) the product of two entries can neither underflow nor overflow, and
            such a matrix is outside the class by design. The limit asked for together with (vi), that the largest
            positive entry be at most 1e12 times the smallest, is implied by (ii), which allows 1e8, and is not
            tested a second time.

    The numbers 200, 1e-8, 1e-12, 1e-11, 1e-10, 1e-150 and 1e150 were set by decision, not derived from an error
    bound.
    """
    a = np.asarray(a, float)
    out = dict.fromkeys(("i", "ii", "iii", "iv", "v", "vi"), False)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)) or np.any(a < 0):
        return out
    pos = a[a > 0]
    if pos.size == 0:
        return out
    lam, v, w = raw
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        weights = np.asarray(v, float) * np.asarray(w, float)
        weights = weights / weights.sum()
    out["i"] = a.shape[0] <= 200
    out["ii"] = bool(pos.min() >= 1e-8 * pos.max())
    out["iii"] = bool(np.isfinite(lam) and np.isfinite(lam_balanced) and abs(lam - lam_balanced) <= 1e-12 * abs(lam_balanced))
    out["iv"] = bool(sq is not None and _vectors_agree(raw, sq, tol=AGREE_COMPONENTS, lam_tol=AGREE_EIGENVALUE))
    out["v"] = bool(np.all(np.isfinite(weights)) and float(weights.min()) >= 1e-10)
    out["vi"] = bool(pos.min() >= 1e-150 and pos.max() <= 1e150)
    return out


def _in_fast_class(a, raw, sq, lam_balanced):
    """Whether perron may return the eigen-solver's answer `raw` for the matrix a without the multiple-precision
    check: True only if ALL six conditions of _fast_class_conditions hold. (Added 11:10 on 14 November 2026; the
    sixth condition at 11:22.)

    This is a declared class of well-conditioned inputs with a cross-check, and nothing more. Membership is not a
    certificate: (iii) and (iv) are agreements between double-precision computations, and an error common to them
    passes.
    """
    return all(_fast_class_conditions(a, raw, sq, lam_balanced).values())


def perron(a):
    """Dominant eigenvalue, right eigenvector v (sums to one), left eigenvector w (w @ v = 1).

    The branches as they are now (changed at 09:47 on 14 November 2026 after the ninth referee check, report 10,
    points 3, 6 and 8, and again at 11:10; what they replaced is under "History" below).

    1. Dimension one: the eigen-solver's answer (LAPACK, _perron_raw) is returned without any check.
    2. The fast path, for a declared class of well-conditioned inputs (_in_fast_class; skipped altogether when the
       module flag ALWAYS_ARBITRATE is True). A matrix is in the class when its dimension is at most 200, every
       positive entry is at least 1e-8 times the largest, the eigen-solver's eigenvalue agrees to 1e-12 with that of
       the matrix balanced by a diagonal change of units (scipy.linalg.matrix_balance), its eigenvalue agrees to
       1e-12 and every component of its vectors to 1e-11 with the repeated squaring in double precision
       (_perron_by_squaring), no stationary weight v_i w_i is below 1e-10, and every positive entry lies between
       1e-150 and 1e150 (the six
       conditions are set out, each with its purpose, in _fast_class_conditions). Inside the class the answer is
       the eigen-solver's, cross-checked against a second double-precision algorithm. This is not a certificate.
       The multiple-precision routine is not called.
    3. Outside the class a multiple-precision answer is computed (_perron_high_precision). When it gives a result
       it is the arbiter: the eigen-solver's eigenvalue and vectors are returned if they agree with it (eigenvalue
       to 1e-12 relative, every component of both vectors to 1e-11 relative), and otherwise the multiple-precision
       eigenvalue and vectors are returned, converted to floats.
    4. Outside the class and with no multiple-precision result (mpmath not installed; more than 40 states; a
       negative entry; a reducible matrix; squares that do not converge; or a result that floats cannot represent
       to 4 machine epsilons), a double-precision answer is returned only if three computations agree: the
       eigen-solver; the eigenvalue of the balanced matrix, to 1e-9 relative; and the repeated squaring in double
       precision, eigenvalue to 1e-12 and every vector component to 1e-11 relative. The eigen-solver's answer is
       then returned. In every other case a RuntimeError is raised, saying which computations were available and
       that reliability could not be established with this implementation. There is no falling back to a looser
       agreement.

    One rule of agreement serves branches 2, 3 and 4: the eigen-solver's answer is returned only if it agrees with
    a squaring (in double precision in 2 and 4, in multiple precision in 3) to 1e-12 in the eigenvalue and 1e-11
    in every component of both vectors.

    Why the class is what it is, and what it does not do. The limits on the spread of the entries and on the
    stationary weights were chosen so that the matrices on which referees found the EIGENVALUE OR THE VECTORS wrong
    lie outside the class: entries spanning more than eight orders of magnitude (fourth, fifth, seventh, eighth
    and ninth reports; regressions G9, G11, G15 to G19), stationary weights below 1e-10 (fifth and seventh).
    Inputs outside the class are arbitrated in multiple precision or refused. The class must NOT be described as
    excluding every input on which a referee found a failure: the sixth referee's matrix,
    [[0.5, 0.5], [0.5, 0.5]] (G12), is inside it. Its vectors are exact, and the failure was cancellation in the
    lag sum at a relative size of 1e8, which is the business of _rounding_guard and not of this routine. The limit
    on the spread compares the entries only with each other, so from 11:10 to 11:22 a matrix whose entries are all
    subnormal could pass it: z * [[1, 2], [3, 4]], z the smallest positive float (ninth report, point 3; G21),
    met every condition but the agreement with the squaring, and failed that one by an accident of rounding
    described below. Since 11:22 the limits of 1e-150 and 1e150 on the entries themselves keep such a matrix out
    by design (G25).

    What this is worth. In branch 3 the vectors are checked against one independent computation carried out with
    far more digits; that is strong evidence and not a proof (no interval arithmetic, no certified bound). In
    branches 2 and 4 the computations are all in double precision and an error common to them passes. For
    z * [[1, 2], [3, 4]] the exact eigenvalue is (5 + sqrt 33) z / 2 = 5.37 z, and the nearest float to that is
    5 z, 6.9% off; the eigen-solver and the balanced eigen-solver both give 5 z. Until 11:10 the squaring gave 5 z
    as well and perron returned it in branch 4 (lineage_cost refused the matrix at a later test). Since 11:10 the
    squaring forms the eigenvalue as w'Av / w'v, gets 4 z, and perron raises in branch 4, there being no
    multiple-precision result for a number that floats cannot hold. That the three no longer agree is an accident
    of rounding, not a protection; what keeps the matrix away from the fast path is the limit on the size of the
    entries. A returned answer is never certified, and nothing here covers what is computed from the vectors
    afterwards.

    What is refused, among inputs a user may well pass. A reducible matrix (a life table with an age after the
    last fertile one), a matrix with a negative entry, and a matrix of more than 40 states that is outside the
    class and on which the double computations do not all agree. From 09:47 to 11:10 a matrix that is
    irreducible but not primitive (a ring in which no stage has any staying, a life table whose fertile ages
    share a divisor) was refused too; both squaring routines now square A + c I and handle it (regressions G22
    and G24). Before 09:47 the eigen-solver's answer was returned unvalidated for all of these.

    History, all on 14 November 2026.
    After primary's fourth referee report on the manuscript: for D P D^-1 with D = diag(1e-120, 1, 1e120) the
    unchecked routine returned lam = 0.57 where the answer is 1, and everything computed from it was wrong without
    warning (regression test: checks/lag_sum_regression.py, G9). The repair compared the eigenvalue with that of the
    balanced matrix and, when they differed by more than 1e-9, returned the balanced solution mapped back, after a
    residual test in the balanced units (_perron_residual below 1e-7). Two earlier versions of that check, written
    the same hour, were wrong in opposite directions: one rejected ordinary rings whose eigenvectors have very small
    components, the other let the failure through to a later check. The full suite of checks and G9 caught them.
    At 07:25, after the seventh report: when the two eigenvalues agreed, the vectors were computed a second time by
    repeated squaring in double precision, and the squaring vectors were returned whenever the two disagreed. The
    eighth referee showed that this could replace right vectors by wrong ones.
    At 07:47, after the eighth: the multiple-precision computation was added, but consulted only when the
    eigen-solver and the squaring disagreed or the squaring gave no answer. When they agreed the eigen-solver's
    vectors were returned without it; when the squaring gave no answer and there was no arbiter they were returned
    unvalidated; and the balanced branch never consulted it at all.
    At 09:47, after the ninth check: the referee found an accepted cost with a relative error of 1.2e-6 that went
    through the balanced branch (report 10, point 6; regression test G18), whose residual test passed with a right
    vector 1.2e-6 off in one component. The balanced solution is no longer returned on any path (only its
    eigenvalue is used), _perron_residual is no longer called from here, and the multiple-precision result
    arbitrated on every call. The docstring before that change still said that when the two eigenvalues agree
    "the vectors from the matrix as given are returned, as they always were"; since 07:47 that had not been true
    of every case (report 10, point 8).
    At 11:10: arbitrating every call made the checks that call this routine hundreds of thousands of times
    impractical (checks/general_ring_closed_forms.py: an estimated 18 hours against 5 minutes), and matrices that
    are irreducible but not primitive were refused. The fast path of branch 2 and the squaring of A + c I were
    added. The fast path resembles the rule of 07:47 (agreement with the squaring returns the eigen-solver's
    answer); what is different is that it now applies only inside the declared class, and that everything outside
    the class goes to the multiple-precision arbiter or is refused.
    At 11:22: the sixth condition of the class, on the absolute size of the entries, was added.
    At 12:23: the agreement required of the vector components was tightened from 1e-9 to 1e-11, in all three
    branches, and that of the eigenvalue with the double-precision squaring from 1e-9 to 1e-12. The fast path had
    been compared with the multiple-precision routine on 21,773 matrices taken at random from the calls made by
    checks/general_ring_closed_forms.py and checks/referee_ms_points.py, at a tolerance of 1e-11 fixed beforehand.
    The class admitted 21,759, and 38 of those, all rings of 7 to 12 stages, exceeded it; the largest difference
    was 5.3e-10, in a component of size 6.5e-7 of a 12-stage ring. An independent solution of that ring in 80
    digits showed the eigen-solver to be the one in error; the multiple-precision squaring was right to 1e-16 and
    the double-precision squaring to 1e-15. At 1e-9 the arbitration of branch 3 returned the same eigen-solver's
    vectors, so the fault lay in the rule of agreement and not in the fast path as such. With 1e-11 such a matrix
    leaves the class and perron returns the multiple-precision vectors for it.
    """
    a = np.asarray(a, float)
    lam, v, w = _perron_raw(a)
    if a.shape[0] < 2:
        return lam, v, w
    raw = (lam, v, w)
    lam_b = sq = None
    if not ALWAYS_ARBITRATE:
        lam_b, sq = _double_checks(a)
        if _in_fast_class(a, raw, sq, lam_b):
            return raw                                   # the declared class: a cross-check, not a certificate
    hp = _perron_high_precision(a)
    if hp is not None:
        return raw if _vectors_agree(raw, hp, tol=AGREE_COMPONENTS, lam_tol=AGREE_EIGENVALUE) else hp   # the arbiter
    # No multiple-precision result: a double-precision answer only if three computations agree.
    if lam_b is None:
        lam_b, sq = _double_checks(a)
    bal_ok = bool(abs(lam - lam_b) <= 1e-9 * abs(lam_b))
    sq_ok = sq is not None and _vectors_agree(raw, sq, tol=AGREE_COMPONENTS, lam_tol=AGREE_EIGENVALUE)
    if bal_ok and sq_ok:
        return raw                                       # agreement of three computations; not a certificate
    raise RuntimeError(
        "perron: the reliability of the dominant eigenvalue and eigenvectors could not be established with this "
        "implementation. Multiple-precision check: no result (possible reasons: mpmath not installed; more than 40 "
        "states; a negative entry; a reducible matrix; squares that did not converge; or a result that floats cannot "
        "represent to 4 machine epsilons). Double-precision computations: "
        f"eigen-solver, eigenvalue {lam!r}; balanced eigen-solver, eigenvalue {lam_b!r}, which "
        f"{'agrees' if bal_ok else 'does not agree'} to 1e-9; repeated squaring, "
        + ("no answer" if sq is None else "which agrees (eigenvalue to 1e-12, vector components to 1e-11)" if sq_ok else
           "which does not agree to 1e-12 in the eigenvalue or to 1e-11 in some vector component")
        + ". A double-precision answer is returned only when all three agree")


def ring_quantities(s, g):
    """lam, m (staying shares), tau (mean stays), T (generation time) of a ring."""
    s = np.asarray(s, float)
    lam = perron(ring_matrix(s, g))[0]
    m = s / lam
    tau = 1.0 / (1.0 - m)
    return lam, m, tau, float(tau.sum())


def return_pgf(m, z):
    """E[z^L], L the time a line of descent takes to go once round the ring: a sum of geometric stays."""
    m = np.asarray(m, float)
    if z == 0:
        return 0.0
    val = np.prod(z * (1.0 - m) / (1.0 - z * m))
    # z may be complex (a point on the unit circle, for a periodic environment); keep the imaginary part then.
    # (Corrected 14 November 2026: a cast to float used to discard it. Found by primary's referee report.)
    return complex(val) if np.iscomplexobj(val) else float(val)


def cost_transfer(s, g, sigma2=1.0, phi=0.0):
    """Contribution of each stage's fluctuating transfer to the second-order cost.

    sigma2 and phi may be scalars or arrays of length n (variance and lag-one autocorrelation of the multiplier of
    g[k]). Closed form: sigma2 * [ (1+phi) / (2 T^2 (1-phi)) - (1/T) R/(1-R) ], R = E[phi^L].
    """
    lam, m, tau, T = ring_quantities(s, g)
    n = len(m)
    sigma2 = np.broadcast_to(np.asarray(sigma2, float), (n,))
    phi = np.broadcast_to(np.asarray(phi, float), (n,))
    out = np.zeros(n)
    for k in range(n):
        R = return_pgf(m, phi[k])
        out[k] = sigma2[k] * ((1 + phi[k]) / (2 * T * T * (1 - phi[k])) - (R / (1 - R)) / T)
    return out


def cost_survival(s, g, sigma2=1.0, psi=0.0):
    """Contribution of each stage's fluctuating survival (staying) to the second-order cost. Negative means a gain.

    Closed form: sigma2 * [ e^2 (1+psi) / (2(1-psi)) - e psi m_k / ((1 - psi m_k)(1 - R)) ], e = (tau_k - 1)/T.
    """
    lam, m, tau, T = ring_quantities(s, g)
    n = len(m)
    sigma2 = np.broadcast_to(np.asarray(sigma2, float), (n,))
    psi = np.broadcast_to(np.asarray(psi, float), (n,))
    out = np.zeros(n)
    for k in range(n):
        e = (tau[k] - 1.0) / T
        R = return_pgf(m, psi[k])
        out[k] = sigma2[k] * (e * e * (1 + psi[k]) / (2 * (1 - psi[k])) - e * psi[k] * m[k] / ((1 - psi[k] * m[k]) * (1 - R)))
    return out


def cost_equal_ring(n, s, gbar, phi):
    """Total second-order cost per unit variance of a ring of n equal stages, every transfer fluctuating."""
    lam0 = s + gbar
    p, q = s / lam0, gbar / lam0
    if phi == 0:
        return q * q / (2 * n)
    r = phi * q / (1 - phi * p)
    return q * q * (1 + phi) / (2 * n * (1 - phi)) - q * r ** n / (1 - r ** n)


def long_run_variance_transfer(s, g, sigma2=1.0, phi=0.0):
    """Leading-order variance of log growth per step caused by fluctuating transfers: sum sigma2 (1+phi)/((1-phi) T^2)."""
    lam, m, tau, T = ring_quantities(s, g)
    n = len(m)
    sigma2 = np.broadcast_to(np.asarray(sigma2, float), (n,))
    phi = np.broadcast_to(np.asarray(phi, float), (n,))
    return float(np.sum(sigma2 * (1 + phi) / (1 - phi)) / T ** 2)


def matrix_cost(a, entries, drivers, kappa, phi):
    """The published matrix formula for any primitive matrix a.

    entries: list of (i, j), the fluctuating entries; drivers[x]: index of the noise series driving entry x;
    kappa[x]: its relative noise size; phi[d]: autocorrelation of driver d. Returns C, the cost per unit sigma^2:
        C = sum over ordered pairs (x, y) with the same driver of
            kappa_x kappa_y [ c_x c_y / 2 - phi w'B_x (I - phi T)^-1 Q B_y v ].
    """
    a = np.asarray(a, float)
    d = a.shape[0]
    lam, v, w = perron(a)
    m = a / lam
    q = np.eye(d) - np.outer(v, w)
    t = m - np.outer(v, w)
    bs = []
    for (i, j) in entries:
        b = np.zeros((d, d))
        b[i, j] = m[i, j]
        bs.append(b)
    c = np.array([w @ b @ v for b in bs])
    tot = 0.0
    for x in range(len(entries)):
        for y in range(len(entries)):
            if drivers[x] != drivers[y]:
                continue
            f = phi[drivers[x]]
            lag = w @ bs[x] @ np.linalg.solve(np.eye(d) - f * t, q @ bs[y] @ v)
            tot += kappa[x] * kappa[y] * (0.5 * c[x] * c[y] - f * lag)
    return float(tot)


def _lineage_chain(m, v, w):
    """The line of descent as a Markov chain: P[i, j] = M[i, j] w[i] / w[j] is the probability of moving from state
    j to state i (columns sum to one), and pi = v * w is its stationary law. Unlike M, P does not change when the
    states are rescaled, so tests of convergence made on P cannot be fooled by a badly scaled matrix."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        pm = m * w[:, None] / w[None, :]
        pi = v * w
    if not (np.all(np.isfinite(pm)) and np.all(np.isfinite(pi))):
        raise RuntimeError("lineage chain: overflow in the change of units; rescale the states")
    colerr = float(np.abs(pm.sum(axis=0) - 1).max()); pi = pi / pi.sum()
    if pi.min() < 1e-12:
        # a state that the line of descent visits with a very small probability: the eigenvector components computed
        # here for it may have few or no correct digits, and a large relative noise size on a transition through
        # that state would turn that into a wrong cost without warning (primary's fifth referee report:
        # A = [[1 - 1e-16, 0.5], [1e-16, 0.5]], entry (1, 1), kappa = 1e8 returned 0 where the answer is -1/3).
        # The cutoff is conservative. It does not mean the matrix is reducible or that the cost cannot be computed:
        # a 30-age Leslie matrix with every fertility 1 and every survival 1/2 has a smallest weight of 6.5e-15 and
        # is declined, though a recursion for its vectors would be accurate (sixth referee report, point 6).
        raise RuntimeError(f"lineage chain: a state has stationary probability {pi.min():.1e}; this implementation declines "
                           "stationary weights below 1e-12. Use an independently validated calculation of the stationary "
                           "law, or higher precision. The cutoff does not imply that the matrix is reducible")
    staterr = float(np.abs(pm @ pi - pi).sum())
    if colerr > 1e-8 or staterr > 1e-8:
        # the chain must be stochastic with pi stationary; if it is not, the Perron vectors were wrong
        raise RuntimeError(f"lineage chain: not a stochastic matrix with stationary law pi (column sums off by {colerr:.1e}, "
                           f"stationarity off by {staterr:.1e}); the Perron vectors are inaccurate")
    pm = pm / pm.sum(axis=0, keepdims=True)          # remove rounding in the column sums
    return pm, pi


def _centered_powers(m, v, w, lags, tol):
    """Yield (lag, P, E_lag) for lag = 1, 2, ..., with E_lag = P^(lag-1) - pi 1', until later terms are negligible.

    P and pi are the chain of the line of descent (_lineage_chain). The excess of returns at any lag is an entry of
    P times an entry of E_lag: u(l) - c = P[i, j] (P^(l-1)[j, i] - pi[j]). E_lag is built as Q D^(lag-1) with
    Q = I - pi 1' and D = P - pi 1', so that it decays geometrically, at the rate of the second eigenvalue, and
    rounding does not accumulate in the stationary direction.

    Stop rule, with a bound on what is left out. Column a of E_lag is the law of the line of descent lag - 1 steps
    after state a, minus pi, so delta = (1/2) max over pairs of columns of the L1 distance between them is the
    Dobrushin coefficient of P^(lag-1), and the L1 norm of every column of E_m, m > lag, is at most
    2 delta^q with q the whole number of blocks of lag - 1 steps in m - 1. Hence the sum over all later lags of the
    largest column norm is at most 2 (lag - 1) delta / (1 - delta). The generator stops when that bound is below
    tol. For an autocorrelation bounded by one, the omitted part of sum_l rho(l) [u(l) - c] is then at most tol in
    absolute value (in exact arithmetic), whatever the scaling of the states. The rule is never applied to a single
    entry: an entry can pass through zero at a lag where the rest has not decayed. Raises if the bound has not
    fallen below tol within `lags` terms, so a truncated sum is never returned silently.

    (14 November 2026, two corrections after primary's referee reports on the manuscript. Second report, point 11:
    the earlier rule stopped when the single excess being summed fell below a tolerance, which is wrong when that
    excess is exactly zero at one lag. Third report, point 21: my first repair tested the norm of M^(lag-1) - v w',
    which is not invariant when the states are rescaled, and had no bound on the tail; for a matrix D P D^-1 with
    D = diag(1, 1, 1e16) it stopped after 145 lags and was 5% off. The test is now made in the coordinates of the
    line of descent, with the tail bound above. Regression tests for both: checks/lag_sum_regression.py.)
    """
    d = m.shape[0]
    pm, pi = _lineage_chain(m, v, w)
    proj = np.outer(pi, np.ones(d))
    q = np.eye(d) - proj
    dm = pm - proj
    e = q.copy()
    for lag in range(1, lags + 1):
        yield lag, pm, e
        e = e @ dm                                    # now E_(lag+1) = P^lag - pi 1'
        if lag >= 1:
            delta = 0.5 * max(np.abs(e[:, [a]] - e).sum(axis=0).max() for a in range(d)) if d > 1 else 0.0
            if delta < 1 and 2 * lag * delta / (1 - delta) <= tol:
                return
    raise RuntimeError(f"lag sum not converged within {lags} lags (tail bound above {tol}); the matrix mixes too slowly for this limit")


def _rounding_guard(result, scale, tol):
    """Refuse to return a number whose rounding error may be material.

    tol controls the part of the lag sum that is left out, in exact arithmetic. It says nothing about rounding. The
    cost is a difference of terms whose absolute values add up to `scale`; summing them in double precision leaves
    an error of the order of machine epsilon times scale. That is negligible for relative noise sizes of order one,
    and it is not when the sizes are very large and the terms nearly cancel (primary's sixth referee report:
    [[0.5, 0.5], [0.5, 0.5]], entry (0, 0), kappa = 1e8 returned 0.1875 where the answer is 0). The estimate is
    64 eps scale; the routine raises when it exceeds both 1000 tol and 1e-9 of the result. Computing with relative
    sizes of order one and rescaling is NOT a remedy: the cost is homogeneous of degree two in the sizes and so is
    its rounding error (seventh referee report, point 3). What is needed in such a case is higher precision.

    This is an estimate and not a certificate. It does not cover errors in the eigenvectors of rare states above
    the cutoff of _lineage_chain, or the growth of rounding error in long products of matrices.
    """
    est = 64 * float(np.finfo(float).eps) * scale
    if est > max(1e3 * tol, 1e-9 * abs(result)):
        raise RuntimeError(f"lag sum: rounding error may be material (terms of total size {scale:.3g} nearly cancel to "
                           f"{result:.3g}; estimated rounding error {est:.1e}). Rescaling the relative sizes does not help; "
                           "use higher precision or an independently validated calculation")


def _stability_guard(fn, a, result, tol):
    """Refuse to return a number when two probes fail to establish that this implementation computes it reliably.

    The routines below start from eigenvectors computed in double precision. When a state is nearly absorbing (a
    diagonal entry within rounding of its column's sum) or the model is otherwise ill-conditioned, small components
    of those vectors can be wrong by tens of per cent while every residual test passes (seventh referee report:
    [[0.9998, 4e-16], [0.0002, 1 - 4e-16]], entry (0, 0), relative size 1e6, returned -2.219 where the answer for
    those exact entries is -1.9988). No floor on the stationary weights catches that in general. This guard asks
    a different question: does the answer move when the entries of the matrix are changed at the level of rounding?
    The cost is recomputed for two copies of the matrix with every entry multiplied by 1 +- 4 eps (once with the
    diagonal lowered and the rest raised, once with a fixed pattern of random signs). If either copy fails, or
    gives a cost that differs from the first by more than the larger of 1000 tol and 1e-6 of the cost, the
    routine raises. An input for which this implementation's answer depends on the last bits is rejected whether
    or not the first answer happened to be right.

    What a refusal means, and what it does not (corrected 14 November 2026 after the ninth referee check, report
    10, point 8). The first line of this docstring and the two messages below used to say that the cost "is not
    determined by this matrix in double precision". A failed probe does not prove that: it shows that this
    algorithm failed or moved on a nearby input, which can be a defect of the algorithm and not a property of the
    model. The messages now say that the tests failed to establish reliability with this implementation.

    This is a test of sensitivity, not a certificate of accuracy: an error common to all three computations would
    pass it (in the example of report 10, point 6, the two probes were themselves 5.7e-7 and 4.6e-7 off and the
    differences stayed under the threshold).
    """
    a = np.asarray(a, float); eps = float(np.finfo(float).eps); nn = a.shape[0]
    patterns = (np.where(np.eye(nn, dtype=bool), -1.0, 1.0),
                np.random.default_rng(20261114).choice(np.array([-1.0, 1.0]), size=a.shape))
    for pat in patterns:
        try:
            other = fn(a * (1.0 + 4 * eps * pat))
        except RuntimeError as exc:
            raise RuntimeError("lag sum: a change of the matrix entries at the level of rounding makes the calculation fail "
                               f"({str(exc)[:80]}); the tests failed to establish that this implementation computes the "
                               "cost reliably for this matrix (that is not a proof that the cost is ill-determined)") from None
        if abs(other - result) > max(1e3 * tol, 1e-6 * abs(result)):
            raise RuntimeError(f"lag sum: the cost changes from {result:.6g} to {other:.6g} when the matrix entries are changed at "
                               "the level of rounding; the tests failed to establish that this implementation computes it "
                               "reliably (possible causes: a nearly absorbing state, severe cancellation, or a defect of the "
                               "algorithm; not a proof that the cost is ill-determined). Use higher precision or an "
                               "independently validated calculation")


def lineage_cost(a, entry, rho, lags=400_000, tol=1e-13, _check=True):
    """Cost per unit variance of one fluctuating entry (i, j), from the returns of a line of descent.

    cost = c^2/2 - c * sum_{l>=1} rho(l) [u(l) - c], with c the elasticity of the entry (the long-run frequency with
    which a line of descent uses the transition) and u(l) = M_ij (M^(l-1))_ji the probability that it uses it again
    l steps later. rho is a function giving the autocorrelation at lag l; it must be bounded by one in absolute value.
    The whole entry fluctuates: for a one-stage ring that is staying and transfer together. tol bounds the omitted
    part of the lag sum, and so (c <= 1) the error of the returned cost, in exact arithmetic.

    (Corrected 14 November 2026: first for an autocorrelation that is zero at lag one and not at lag two, found
    when testing a counterexample of secondary's, board 0014; then the stop rule, twice, see _centered_powers.)
    """
    a = np.asarray(a, float)
    lam, v, w = perron(a)
    m = a / lam
    i, j = entry
    c = m[i, j] * w[i] * v[j]
    tot, mag = 0.0, 0.0
    for lag, pm, e in _centered_powers(m, v, w, lags, tol):
        term = rho(lag) * pm[i, j] * e[j, i]            # u(lag) - c
        tot += term; mag += abs(term)
    result = float(0.5 * c * c - c * tot)
    _rounding_guard(result, float(0.5 * c * c + c * mag), tol)
    if _check:
        _stability_guard(lambda b: lineage_cost(b, entry, rho, lags, tol, _check=False), a, result, tol)
    return result


def lineage_cost_shared(a, entries, kappa, rho, lags=400_000, tol=1e-13, _check=True):
    """Cost per unit sigma^2 of several fluctuating entries that all follow ONE driver with autocorrelation rho.

    cost = (sum_a kappa_a c_a)^2 / 2 - sum over ordered pairs (a, b) of
           kappa_a kappa_b c_b * sum_{l>=1} rho(l) [u_{b->a}(l) - c_a],
    where u_{b->a}(l) = M_a (M^(l-1))[j_a, i_b] w[i_a] / w[i_b] is the probability that a line of descent which
    uses transition b now uses transition a l steps later. Both orders of each pair are included, and so is the
    lag-zero term, which is the square in front.

    tol bounds the omitted tail of the lag sums in exact arithmetic. Rounding and conditioning are handled
    separately and only by tests that can reject: see _rounding_guard and _stability_guard. Neither certifies the
    result. This routine is a checking tool for ordinary models with relative sizes of order one.
    """
    a = np.asarray(a, float)
    lam, v, w = perron(a)
    m = a / lam
    c = np.array([m[i, j] * w[i] * v[j] for (i, j) in entries])
    kappa = np.asarray(kappa, float)
    # tol is a bound on the error of the returned cost (for |rho| <= 1): the tail of each lag sum is multiplied by
    # |kappa_a kappa_b| c_b, so the tolerance passed down is divided by the sum of those weights
    weight = float(np.sum(np.abs(np.outer(kappa, kappa)) * c[None, :]))
    tot, mag = 0.0, 0.0
    for lag, pm, e in _centered_powers(m, v, w, lags, tol / max(weight, 1.0)):
        r = rho(lag)
        if r == 0.0:
            continue
        for x, (ia, ja) in enumerate(entries):
            for y, (ib, jb) in enumerate(entries):
                term = kappa[x] * kappa[y] * c[y] * r * pm[ia, ja] * e[ja, ib]      # u_{b->a}(lag) - c_a
                tot += term; mag += abs(term)
    result = float(0.5 * float(kappa @ c) ** 2 - tot)
    _rounding_guard(result, float(0.5 * float(np.abs(kappa) @ c) ** 2 + mag), tol)
    if _check:
        _stability_guard(lambda b: lineage_cost_shared(b, entries, kappa, rho, lags, tol, _check=False), a, result, tol)
    return result


def per_generation_cost_transfer(s, g, sigma2=1.0, phi=0.0):
    """T times the second-order cost of fluctuating transfers: the amount by which ln R0 must exceed zero.

    Per host: sigma2 * [ (1 + 2 theta)/(2 T) - R/(1 - R) ]. Use rates at the threshold (lam = 1) for a statement
    about R0; at other lam it is the per-step cost multiplied by the generation time.
    """
    T = ring_quantities(s, g)[3]
    return T * cost_transfer(s, g, sigma2, phi)
