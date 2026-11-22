"""Claim: the gamma-refresh diagonal and the finite cores of the generic
divergence proof agree with independent calculations. Tolerances and cases
were registered in plan.md. These are deterministic, non-interval checks;
they do not numerically prove an almost-everywhere or all-order assertion.
"""
import sys
sys.dont_write_bytecode = True
from collections import defaultdict
from fractions import Fraction as Q
from itertools import product
from math import comb, factorial
from pathlib import Path
import importlib.util
import json
import time
import mpmath as mp

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
PHIS = [Q(0), Q(1, 4), Q(1, 2), Q(3, 4)]
TOL = mp.mpf('1e-35')


def say(*args):
    print(*args, flush=True)


def asmp(x):
    return mp.mpf(x.numerator) / x.denominator if isinstance(x, Q) else mp.mpf(x)


def scaled(x, y):
    x, y = asmp(x), asmp(y)
    return abs(x-y) / max(1, abs(x), abs(y))


def trace_half(k, edges):
    """Half trace of adjacency^(2k) on a path of edges+1 vertices.
    Discrete Fourier/binomial formula, using only integer arithmetic.
    """
    if edges <= 0:
        return 0
    period = edges + 2
    bins = sum(comb(2*k, k+j*period)
               for j in range(-k//period-1, k//period+2)
               if 0 <= k+j*period <= 2*k)
    ans = period*bins - 4**k
    assert ans % 2 == 0
    return ans//2


def span_counts(k):
    a = [0] + [trace_half(k, length) for length in range(1, k+2)]
    counts = [a[l]-2*a[l-1]+(a[l-2] if l >= 2 else 0)
              for l in range(1, k+2)]
    assert counts[-1] == 0
    assert all(x >= 0 for x in counts)
    return counts[:-1]


def direct_walks(k):
    """Independent enumeration of all 2^(2k) nearest-neighbor paths."""
    counts = [0]*k
    for directions in product((-1, 1), repeat=2*k):
        pos = lo = hi = 0
        for step in directions:
            pos += step
            lo = min(lo, pos)
            hi = max(hi, pos)
        if pos == 0:
            counts[hi-lo-1] += 1
    assert all(x % 2 == 0 for x in counts)
    return [x//2 for x in counts]


def weights(phi, K):
    return [Q(0)] + [sum((Q(c)*phi**j for j, c in enumerate(span_counts(k))), Q(0))
                     for k in range(1, K+1)]


def power_series(a, alpha, K):
    """Coefficient recurrence from A C' = alpha A' C; A(0)=1."""
    z = a[0]*0
    c = [z+1]+[z]*K
    for m in range(1, K+1):
        c[m] = sum((((alpha+1)*j-m)*a[j]*c[m-j]
                    for j in range(1, m+1)), z)/m
    return c


def diagonal(w, K, precision=None):
    if precision:
        mp.mp.dps = precision
        a = [mp.mpf(1)] + [(-1)**j*asmp(w[j+1])*factorial(j)/(j+1)
                              for j in range(1, K+1)]
    else:
        a = [Q(1)] + [(-1)**j*w[j+1]*Q(factorial(j), j+1)
                         for j in range(1, K+1)]
    return [a[0]*0] + [power_series(a, -n, n)[n]/n for n in range(1, K+1)]


def multiply(a, b, K):
    z = a[0]*0
    return [sum((a[j]*b[n-j] for j in range(n+1)), z) for n in range(K+1)]


def exponential(a, K):
    z = a[0]*0
    out = [z+1]+[z]*K
    for n in range(1, K+1):
        out[n] = sum((j*a[j]*out[n-j] for j in range(1, n+1)), z)/n
    return out


def inverse_residual(w, b, K):
    e = exponential(b, K)
    z = [Q(0)]+e[:-1]
    powers = [Q(1)]+[Q(0)]*K
    out = [Q(0)]*(K+1)
    for k in range(1, K+1):
        powers = multiply(powers, z, K)
        fk = (-1)**(k+1)*w[k]*Q(factorial(k-1), k)
        for j in range(K+1):
            out[j] += fk*powers[j]
    out[1] -= 1
    return max(abs(x) for x in out)


def gamma_moments(K, centered):
    out = [[Q(0)]*(K+1) for _ in range(K+1)]
    out[0][0] = Q(1)
    if centered:
        for j in range(2, K+1):
            for n in range(1, K+1):
                out[j][n] = (j-1)*(out[j-1][n-1]+out[j-2][n-1])
    else:
        for j in range(1, K+1):
            out[j] = out[j-1][:]
            for n in range(1, K+1):
                out[j][n] += (j-1)*out[j-1][n-1]
    return out


def joint(exponents, phi, marginals):
    K = len(marginals[0])-1
    out = [Q(0)]*(K+1)
    for cuts in product((0, 1), repeat=len(exponents)-1):
        groups = []
        current = exponents[0]
        for cut, exponent in zip(cuts, exponents[1:]):
            if cut:
                groups.append(current)
                current = exponent
            else:
                current += exponent
        groups.append(current)
        prob = (1-phi)**sum(cuts)*phi**(len(cuts)-sum(cuts))
        pol = [Q(1)]+[Q(0)]*K
        for degree in groups:
            pol = multiply(pol, marginals[degree], K)
        for n in range(K+1):
            out[n] += prob*pol[n]
    return out


def finite_cores():
    centered, raw = gamma_moments(8, True), gamma_moments(8, False)
    count = 0
    for phi in PHIS:
        for length in range(1, 5):
            for powers in product((1, 2, 3), repeat=length):
                degree = sum(powers)
                if degree > 8:
                    continue
                c = joint(powers, phi, centered)
                assert all(0 <= x <= y for x, y in zip(c, centered[degree]))
                r = joint(powers, phi, raw)
                assert r[degree-1] == factorial(degree-1)*phi**(length-1)
                count += 1
    say('PASS exact refresh joint moment domination and raw top coefficient:', count, 'cases')
    mu = gamma_moments(40, True)
    count = 0
    for n in range(1, 21):
        for m in range(n+1, 2*n+1):
            assert mu[m][n] <= 8**n*factorial(n)
            count += 1
    say('PASS exact centered-moment factorial bound:', count, 'cases')


def geometry_and_tails():
    mp.mp.dps = 80
    worst_fraction = mp.mpf(0)
    count = 0
    for r in [mp.mpf('.25'), mp.mpf('.6')]:
        rho = (1+r)/2
        C = max(2*r*(1+r)/(1-r),
                (2*r*mp.log(1+r)+2)/mp.log(1/rho),
                2*r*(rho+r)/(rho-r))
        for zero in [mp.mpc('.1'), mp.mpc('.8'), mp.mpc('-.8'),
                     mp.mpc('.2', '.3'), mp.mpc(0, '.9')]:
            def green(x):
                return mp.log(abs((1-mp.conj(zero)*x)/(x-zero)))
            knots = [-r, r]
            if zero.imag == 0 and -r < zero.real < r:
                knots = [-r, zero.real, r]
            integral = mp.quad(green, knots)
            bound = C*mp.log(1/abs(zero))
            assert integral <= bound + TOL
            worst_fraction = max(worst_fraction, integral/bound)
            count += 1
        integral = mp.quad(lambda x: (1+x)/(1-x), [-r, r])
        assert integral <= C + TOL
        worst_fraction = max(worst_fraction, integral/C)
        count += 1
    say('PASS Green kernel / positive harmonic integral:', count,
        'cases; largest integral/bound', mp.nstr(worst_fraction, 12))
    count = 0
    for p in [20, 100]:
        for delta in [mp.mpf('.1'), mp.mpf('.25'), mp.mpf('.5')]:
            actual = (mp.gammainc(p, 0, p*(1-delta)) +
                      mp.gammainc(p, p*(1+delta), mp.inf))/mp.gamma(p)
            bound = (mp.exp(-p*(-delta-mp.log(1-delta))) +
                     mp.exp(-p*(delta-mp.log(1+delta))))
            assert actual <= bound + TOL
            count += 1
    say('PASS gamma concentration:', count, 'two-tail cases')
    count = 0
    for s, c, h0, h1, u in product([Q(1, 4), Q(3, 4)], repeat=5):
        # G(u)=s+c h1/(s+c h0/u); derivative in log coordinate.
        value = s+c*h1/(s+c*h0/u)
        derivative = c*c*h0*h1/(s*u+c*h0)**2
        elasticity = asmp(derivative*u/value)
        kap = (mp.sqrt(asmp(s*s+c*h1))-asmp(s))/(mp.sqrt(asmp(s*s+c*h1))+asmp(s))
        assert elasticity <= kap+TOL
        count += 1
    say('PASS real two-step contraction:', count, 'cases')
    count = 0
    for s, c, h0, h1, h2 in product([Q(1, 4), Q(3, 4)], repeat=5):
        entries = [s**3+s*c*(h2+h1), c*h0*(s*s+c*h2),
                   s*s+c*h1, s*c*h0]
        derivatives = [s*c*(h2+h1), c*h0*(s*s+2*c*h2), c*h1, s*c*h0]
        beta = c*min(h1, h2)/(s*s+c*min(h1, h2))
        assert all(d >= beta*x for d, x in zip(derivatives, entries))
        count += 1
    say('PASS exact triple-product elasticity:', count, 'cases, four entries each')


def supplementary(w_by_phi):
    mp.mp.dps = 80
    count = 0
    for n in range(1, 13):
        for cuts in product((0, 1), repeat=n-1):
            parts = []
            size = 1
            for cut in cuts:
                if cut:
                    parts.append(size)
                    size = 1
                else:
                    size += 1
            parts.append(size)
            lhs = factorial(len(parts))
            for size in parts:
                lhs *= factorial(size)
            assert lhs <= factorial(n)
            count += 1
    say('PASS exact factorial composition inequality:', count, 'compositions')
    for R in [mp.mpf('.25'), mp.mpf('.75'), mp.mpf('.95')]:
        delta = eta = (1-R)**2/100
        A = mp.exp(eta)*(1+delta)-1
        disc = (1-R)**2-4*R*A
        assert disc > 0
        Z = ((1-R)-mp.sqrt(disc))/2
        T = R*(A+Z)/(1-Z)
        contraction = R*(1+A)/(1-Z)**2
        assert scaled(T, Z) < TOL
        assert 0 < Z < 1 and contraction < 1
    say('PASS absolute-coefficient contraction choice: 3 cases')
    worst = mp.mpf(0)
    for k in range(1, 9):
        for L in range(1, 11):
            value = sum((4*mp.cos(j*mp.pi/(L+2))**2)**k
                        for j in range(1, (L+1)//2+1))
            error = scaled(value, trace_half(k, L))
            worst = max(worst, error)
            assert error < TOL
    say('PASS finite-path eigenvalues vs integer trace: 80 cases; worst', mp.nstr(worst, 8))
    count = 0
    for phi in PHIS[1:]:
        w = w_by_phi[str(phi)]
        p = asmp(phi)
        for k in range(1, 41):
            for L in [1, 2, 4, 8]:
                lower = (1-p)**2*p**(L-1)*(4*mp.cos(mp.pi/(L+2))**2)**k
                assert lower <= asmp(w[k])+TOL
                count += 1
    say('PASS fixed-path lower bound for W:', count, 'cases')

    import numpy as np
    from scipy.linalg import eigh_tridiagonal
    s, c, phi = Q(3, 5), Q(2, 5), Q(1, 2)
    mu = gamma_moments(4, True)
    pol = [Q(0)]*5
    for degree in range(1, 5):
        term = [Q(0)]*5
        for j in range(degree+1):
            refreshed = multiply(mu[j], mu[degree-j], 4)
            for k in range(5):
                moment = phi*mu[degree][k]+(1-phi)*refreshed[k]
                term[k] += comb(degree, j)*(s*c)**j*c**(degree-j)*moment
        for k in range(5):
            pol[k] += Q((-1)**(degree+1), degree)*(term[k]-c**degree*mu[degree][k])

    def integrate(v, order):
        p = 1/v
        index = np.arange(order, dtype=float)
        diagonal = (2*index+p)/p
        index_off = np.arange(1, order, dtype=float)
        off = np.sqrt(index_off*(index_off+p-1))/p
        hosts, eigenvectors = eigh_tridiagonal(diagonal, off)
        wts = eigenvectors[0]**2
        b = hosts-1
        numerator_held = 1+float(c*(1+s))*b
        held = np.dot(wts, np.log(numerator_held)-np.log(1+float(c)*b))
        values = np.log(1+float(s*c)*b[:, None]+float(c)*b[None, :])-np.log(1+float(c)*b[:, None])
        refreshed = wts @ values @ wts
        return float(phi)*held+float(1-phi)*refreshed

    for v in [.01, .005, .0025]:
        lo, hi = integrate(v, 40), integrate(v, 64)
        assert abs(lo-hi) <= 1e-12
        series = float(pol[0])+float(pol[1])*v+float(pol[2])*v*v
        cubic = abs(hi-series)/v**3
        assert cubic <= 100
        say('PASS two-step gamma quadrature vs formal expansion: v=', v,
            'quadrature difference=', f'{abs(lo-hi):.5g}',
            'absolute residual/v^3=', f'{cubic:.9g}')


def main():
    start = time.time()
    say('Registered tolerances: exact rationals; precision scaled 1e-35; epsilon-limit scaled 1e-8.')
    say('No Monte Carlo, no interval certification, no finite test of the exceptional set.')
    for k in range(1, 8):
        assert direct_walks(k) == span_counts(k)
    say('PASS direct walk enumeration vs finite-path trace: 7 orders')
    all_w = {str(phi): weights(phi, 81) for phi in PHIS}
    for phi in PHIS:
        w = all_w[str(phi)]
        assert w[1] == 1 and w[2] == 1+2*phi and w[3] == 1+6*phi+3*phi*phi
        assert all(1 <= w[k] <= Q(4**k, 2) for k in range(1, 82))
    say('PASS exact W identities and factorial-input bounds: 324 moments')
    finite_cores()

    # Known-result gates precede new high-order results.
    old_path = (__import__('pathlib').Path(__file__).resolve().parent / '../gap3holo2/checks.py').resolve()
    spec = importlib.util.spec_from_file_location('archived_gamma_oracle', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    worst = mp.mpf(0)
    for phi in PHIS:
        for s in [Q(3, 10), Q(6, 10), Q(9, 10)]:
            phi_s = str(float(phi))
            s_s = str(float(s))
            actual = old.gamma_refresh(s_s, phi_s, 1, 100)[0]
            expected = (1-s)*(1+(3-s)*phi)/(2*(2-s)*(1+(1-s)*phi))
            mp.mp.dps = 100
            error = scaled(str(actual), expected)
            worst = max(worst, error)
            assert error < TOL
    say('PASS archived recursion known first coefficient: 12 cases; worst', mp.nstr(worst, 8))

    out = {}
    for phi in PHIS:
        w = all_w[str(phi)]
        exact = diagonal(w, 9)
        assert exact[1] == Q(1, 2)+phi
        assert exact[2] == -2*(Q(1, 3)+2*phi+phi**2)+Q(3, 2)*(Q(1, 2)+phi)**2
        assert inverse_residual(w, exact, 9) == 0
        lo = diagonal(w, 80, 120)
        hi = diagonal(w, 80, 180)
        error = max(scaled(x, y) for x, y in zip(lo[1:], hi[1:]))
        exact_error = max(scaled(x, y) for x, y in zip(exact[1:], hi[1:]))
        assert max(error, exact_error) < TOL
        say('PASS diagonal phi=', str(phi), '80 coefficients; precision error', mp.nstr(error, 8),
            'exact error', mp.nstr(exact_error, 8))
        say('  finite diagnostics n, |b_n|, (|b_n|/n!)^(1/n):',
            [(n, mp.nstr(abs(hi[n]), 10),
              mp.nstr((abs(hi[n])/mp.factorial(n))**(mp.mpf(1)/n), 10)) for n in [10, 40, 80]])
        out[str(phi)] = {'diagonal': [mp.nstr(x, 180) for x in hi[1:]],
                         'W': [str(x) for x in w[1:]],
                         'exact_first_9': [str(x) for x in exact[1:]],
                         'precision_scaled_error': str(error)}
        (HERE/'results.json').write_text(json.dumps(out, indent=2)+'\n')
    say('PASS exact inverse substitution and known long-stay b1,b2: 4 phi values')

    worst = mp.mpf(0)
    for phi in PHIS:
        actual = old.gamma_refresh('0.999999999999', str(float(phi)), 6, 160)
        expected = diagonal(all_w[str(phi)], 6)
        mp.mp.dps = 160
        errors = [scaled(mp.mpf(str(x))/mp.mpf('1e-12')**n, expected[n])
                  for n, x in enumerate(actual, 1)]
        error = max(errors)
        assert error < mp.mpf('1e-8')
        worst = max(worst, error)
        say('PASS diagonal vs independent mixed-moment recursion phi=', str(phi),
            '6 coefficients; epsilon-limit error', mp.nstr(error, 10))
    say('PASS all 24 small-epsilon comparisons; worst', mp.nstr(worst, 10))
    geometry_and_tails()
    supplementary(all_w)
    say('PASS completed; elapsed seconds', round(time.time()-start, 3))


if __name__ == '__main__':
    main()
