"""Exact formal coefficients of the stationary log ratio and sign function.

Auxiliary derivation, not an invariant-law numerical solver. Sympy rational
polynomials implement the triangular conditional-moment recurrence.
"""
import math
import sympy as S
from pathlib import Path

d, p, q = S.symbols('delta phi q')
N = 8
M = {}
for k in range(N+1):
    M[k, 0] = (S.Integer(k == 0), S.Integer(k == 0))
for n in range(1,N+1):
    M[0,n] = (S.Integer(0),S.Integer(0))
    for k in range(1,N+1):
        out = []
        for j,z in enumerate((-1,1)):
            val = 0
            if k <= n:
                for m in range(n-k+1):
                    mm = M[m,n-k]
                    mix = ((1+p)*mm[j]+(1-p)*mm[1-j])/2
                    val += (-1)**m*math.comb(k+m-1,m)*mix
                val *= (1+z*d)**k
            out.append(S.expand(val))
        M[k,n] = tuple(out)
    print('moments degree',n,flush=True)
r = [S.Integer(0)] + [S.expand(sum(S.Rational((-1)**(k+1),2*k)*sum(M[k,n]) for k in range(1,n+1))) for n in range(1,N+1)]
# coefficients of exp(-R) by n e_n = -sum k r_k e_(n-k).
e = [S.Integer(1)]
for n in range(1,N+1):
    e.append(S.expand(-sum(k*r[k]*e[n-k] for k in range(1,n+1))/n))
g = [S.Integer(0)]
for n in range(1,N+1):
    g.append(S.expand(-e[n]-2*n*r[n]+sum(e[n-k]*k*r[k] for k in range(1,n+1))))
lines=['Exact rational coefficient derivation; N=8.','R(q)=sum r_n q^n; G(q)=sum g_n q^n.']
for n in range(1,N+1):
    lines.append(f'r_{n} = {r[n]}')
    lines.append(f'g_{n} = {g[n]}')
    if n>=2:
        poly = S.Poly(S.cancel(g[n]/d**2),d,p)
        bound = sum(abs(x) for x in poly.coeffs())
        lines.append(f'sum of absolute coefficients of g_{n}/delta^2 = {bound}')
Path(__file__).with_name('coefficients.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
