"""Claim: the finite cores of gap3exc's positive factorial-type and capacity
arguments agree with independent computations. The archived diagonal is a
numerical oracle, not a proof of an infinite lower bound. Each check states
its claim. Grids/tolerances were registered in plan.md before execution.
No simulation, interval certification, or collaborator-code import is used.
"""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as Q
from itertools import product
from math import factorial, comb
from pathlib import Path
import json
import mpmath as mp

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
PHIS = [Q(0), Q(1, 4), Q(1, 2), Q(3, 4)]
K = 40


def say(*args):
    print(*args, flush=True)


def number(q):
    return mp.mpf(q.numerator) / q.denominator if isinstance(q, Q) else mp.mpf(q)


def product_series(a, b, N):
    out = [a[0]*0] * (N+1)
    for j in range(min(N, len(a)-1)+1):
        for k in range(min(N-j, len(b)-1)+1):
            out[j+k] += a[j]*b[k]
    return out


def power_series(u, p, N):
    """Differential equation u q'=p u' q; u[0]=1, integer p."""
    out = [u[0]] + [u[0]*0]*N
    for k in range(1, N+1):
        out[k] = sum((((p+1)*j-k)*u[j]*out[k-j]
                      for j in range(1, k+1)), u[0]*0) / k
    return out


def exponential(u, N):
    assert u[0] == 0
    out = [u[0]+1] + [u[0]*0]*N
    for k in range(1, N+1):
        out[k] = sum((j*u[j]*out[k-j] for j in range(1, k+1)), u[0]*0)/k
    return out


def matrix_walk_counts(N):
    """Half traces of actual powers of each path adjacency matrix.
    Independent of the archived binomial/Fourier trace formula.
    """
    traces = [[0]*(N+2) for _ in range(N+1)]
    for length in range(1, N+2):
        size = length+1
        mat = [[int(i == j) for j in range(size)] for i in range(size)]
        for step in range(1, 2*N+1):
            mat = [[(row[j-1] if j else 0) +
                    (row[j+1] if j+1 < size else 0)
                    for j in range(size)] for row in mat]
            if step % 2 == 0:
                trace = sum(mat[j][j] for j in range(size))
                assert trace % 2 == 0
                traces[step//2][length] = trace//2
    return [[traces[k][l]-2*traces[k][l-1]+
             (traces[k][l-2] if l >= 2 else 0)
             for l in range(1, k+2)] for k in range(N+1)]


def diagonal_checks():
    """Reproduce known W and b identities before inspecting higher orders."""
    counts = matrix_walk_counts(K+1)
    for k in range(1, 8):
        brute = [0]*k
        for steps in product((-1, 1), repeat=2*k):
            x = lo = hi = 0
            for step in steps:
                x += step
                lo, hi = min(lo, x), max(hi, x)
            if x == 0:
                brute[hi-lo-1] += 1
        assert [x//2 for x in brute] == counts[k][:-1]
        assert all(x % 2 == 0 for x in brute)
    say('PASS direct walk enumeration versus adjacency powers: 7 orders')
    archive = json.loads((HERE.parent/'gap3corr/results.json').read_text())
    mp.mp.dps = 120
    maxerr = mp.mpf(0)
    saved = {}
    for phi in PHIS:
        W = [Q(0)] + [sum((Q(c)*phi**j for j, c in enumerate(counts[k])), Q(0))
                       for k in range(1, K+2)]
        assert W[1:4] == [1, 1+2*phi, 1+6*phi+3*phi**2]
        F = [Q(0)] + [Q((-1)**(k+1)*factorial(k-1), k)*W[k]
                      for k in range(1, K+2)]
        U = F[1:]
        exact_b = [Q(0)] + [power_series(U, -n, n)[n]/n for n in range(1, 10)]
        assert exact_b[1] == Q(1, 2)+phi
        assert exact_b[2] == -2*(Q(1, 3)+2*phi+phi**2)+Q(3, 2)*(Q(1, 2)+phi)**2
        # Opposite inversion direction, using an exponential recurrence.
        for n in range(1, 10):
            predicted = exponential([-n*x for x in exact_b], n-1)[n-1]/n
            assert predicted == F[n]
        un = [number(x) for x in U]
        b = [power_series(un, -n, n)[n]/n for n in range(1, K+1)]
        for n, val in enumerate(b, 1):
            ref = mp.mpf(archive[str(phi)]['diagonal'][n-1])
            err = abs(val-ref)/max(1, abs(val), abs(ref))
            maxerr = max(maxerr, err)
            assert err < mp.mpf('1e-60'), (phi, n, err)
            if n <= 9:
                assert abs(val-number(exact_b[n]))/max(1, abs(val)) < mp.mpf('1e-60')
        saved[str(phi)] = [mp.nstr(x, 110) for x in b]
        say('DIAGNOSTIC phi=', str(phi), ' (|b_n|/n!)^(1/n) at n=10,20,40:',
            ', '.join(mp.nstr((abs(b[n-1])/factorial(n))**(mp.mpf(1)/n), 12)
                      for n in (10, 20, 40)))
    say('PASS known W_1..W_3 and b_1,b_2: 4 parameter cells, exact')
    say('PASS reverse Lagrange exponential identity: 36 exact coefficients')
    say('PASS archived diagonal comparison: 160 coefficients; max scaled error', mp.nstr(maxerr, 10))
    (HERE/'diagonal_checks.json').write_text(json.dumps(saved, indent=2)+'\n')


def compositions(n):
    if n == 0:
        yield ()
    else:
        for j in range(1, n+1):
            for rest in compositions(n-j):
                yield (j,)+rest


def type_zero_cores():
    """Check the coefficient majorants used in the type-zero contradiction."""
    count = 0
    for N in range(1, 13):
        counts = [0]*(N+1)
        for js in compositions(N):
            p = 1
            for j in js:
                p *= factorial(j)
            assert p <= factorial(N)
            counts[len(js)] += 1
            count += 1
        assert counts[1:] == [comb(N-1, k-1) for k in range(1, N+1)]
        assert sum(counts) == 2**(N-1)
    say('PASS factorial product and ordered-composition counts:', count, 'exact compositions')
    count = 0
    for N in range(1, 13):
        for c in (Q(1, 5), Q(1), Q(5), Q(25)):
            q = [c**l*Q(factorial(N-l), factorial(N)) for l in range(N+1)]
            assert all(x <= max(q[0], q[-1]) for x in q)
            count += len(q)
    say('PASS log-convex endpoint estimate:', count, 'exact values')
    mp.mp.dps = 80
    eps = Q(1, 7)
    count = 0
    largest = mp.mpf(0)
    for cutoff in (2, 5):
        for n in range(2, 17):
            N = n-1
            R = [Q(0)] + [(eps**j*factorial(j) if j >= cutoff else Q(0))
                           for j in range(1, N+1)]
            coeff = exponential([n*x for x in R], N)
            for m in range(N+1):
                bound = mp.exp(n)*number((2*eps)**m)*factorial(m)
                ratio = number(coeff[m])/bound
                assert ratio <= 1+mp.mpf('1e-60')
                largest = max(largest, ratio)
                count += 1
    say('PASS exponential-tail majorant:', count, 'coefficients; maximum ratio', mp.nstr(largest, 10))


def potential_cores():
    """Independently integrate the Green inequality for continuous measures."""
    mp.mp.dps = 60
    count = 0
    largest = mp.mpf(0)
    for r in (mp.mpf('.25'), mp.mpf('.6')):
        rho = (1+r)/2
        L = 1/r  # integral log+ kernel <=2, density=1/(2r)
        C = max((1+r)/(1-r), (mp.log(1+r)+L)/mp.log(1/rho), (rho+r)/(rho-r))
        for z in (mp.mpc('.1'), mp.mpc('.8'), mp.mpc('-.8'),
                  mp.mpc('.2', '.3'), mp.mpc(0, '.9')):
            knots = [-r, r]
            if z.imag == 0 and -r < z.real < r:
                knots = [-r, z.real, r]
            value = mp.quad(lambda x: mp.log(abs((1-mp.conj(z)*x)/(x-z))), knots)/(2*r)
            bound = C*mp.log(1/abs(z))
            assert value <= bound+mp.mpf('1e-30')
            largest = max(largest, value/bound)
            count += 1
        value = mp.quad(lambda x: (1+x)/(1-x), [-r, r])/(2*r)
        assert value <= C+mp.mpf('1e-30')
        largest = max(largest, value/C)
        count += 1
    say('PASS continuous-measure Green/harmonic bound:', count, 'quadratures; max ratio', mp.nstr(largest, 10))
    count = 0
    for iz in range(10):
        z = Q(iz-5, 2)
        x = min(Q(2, 3), max(Q(-1, 3), z))
        for iy in range(10):
            y = Q(-1, 3)+Q(iy, 9)
            assert abs(x-y) <= 2*abs(z-y)
            count += 1
    say('PASS nearest-support-point inequality:', count, 'exact triples')


def obstruction_checks():
    """Check the artificial comparison families, not the gamma host itself."""
    count = 0
    for star in (Q(1, 5), Q(2, 5), Q(3, 5), Q(4, 5)):
        for n in range(1, 21):
            b = Q((-1)**(n+1)*factorial(n), n+1)  # arbitrary nonzero diagonal
            # a^n*(b*(1-a/star)+(a/star)*n!*1_{n odd})
            leading = b
            next_coeff = (-b+(factorial(n) if n % 2 else 0))/star
            value = star**n*leading+star**(n+1)*next_coeff
            assert value == (star**n*factorial(n) if n % 2 else 0)
            assert b*(1-star/star) == 0
            count += 1
    say('PASS preserved diagonal and prescribed exceptional slice:', count, 'exact coefficients')


def main():
    say('gap3exc deterministic checks; registration: plan.md; no interval certification')
    diagonal_checks()
    type_zero_cores()
    potential_cores()
    obstruction_checks()
    say('ALL CHECKS PASSED. Finite checks do not prove emptiness or an eventual factorial lower bound.')


if __name__ == '__main__':
    main()
