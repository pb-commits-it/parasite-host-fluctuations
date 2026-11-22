#!/usr/bin/env python3
"""Check the GAP 1 partial theorem and probe the unproved global sign.

Claims: G(q)>0 for q<=1/30 (proved in gap1_proof.md); globally G>0
(numerical question). Independent finite-history enumeration checks the formal
moment calculation. Deterministic Chebyshev interpolation of the Koopman
operator evaluates the stationary law, never assuming it has a density.
The Poisson equation differentiates that law. A separate high-precision
power series evaluates a fine grid in the proved interval.

Tolerances were registered in plan.md before running. No Monte Carlo, no
imports from project collaborators. mpmath precision does not certify the
collocation error. Outputs list resolution differences and unresolved cells.

Run: OPENBLAS_NUM_THREADS=1 python3 -I gap1/gap1_checks.py
Options: --mode pilot|full|hp|hpgrid|audit; --hp-n 48 (only hp mode).
"""
import argparse
import csv
import itertools
import math
import random
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 65
FAIL = []
FAST_HP = False
HP_RESIDUAL = mp.mpf(0)


def hp_solve(A,b):
    """65-digit iterative refinement; binary64 LU only supplies corrections.

    Every residual and accumulated solution is evaluated at mp precision.
    This is checked against a full mpmath LU solve before using the fast mode.
    A small residual is not an interval enclosure or a discretization bound.
    """
    global HP_RESIDUAL
    if not FAST_HP:
        return mp.lu_solve(A,b)
    factor=lu_factor(np.array(A.tolist(),dtype=float),check_finite=False)
    x=mp.matrix([0]*A.rows)
    for _ in range(12):
        residual=b-A*x
        norm=max(abs(v) for v in residual)
        if norm<mp.mpf('1e-57'):
            HP_RESIDUAL=max(HP_RESIDUAL,norm)
            return x
        corr=lu_solve(factor,np.array(list(residual),dtype=float),check_finite=False)
        x+=mp.matrix([mp.mpf(float(v)) for v in corr])
    raise ArithmeticError('high-precision iterative refinement did not converge')


def say(*args):
    print(*args, flush=True)


def check(name, ok, detail=''):
    say('PASS' if ok else 'FAIL', name, detail)
    if not ok:
        FAIL.append(name)


def fmt(x, n=18):
    return mp.nstr(x, n)


def moments(delta, phi, N):
    """Numerical conditional-moment triangular recurrence, not sympy output."""
    d, p = mp.mpmathify(delta), mp.mpf(phi)
    a, b = (1+p)/2, (1-p)/2
    M = [[[mp.mpf(0), mp.mpf(0)] for n in range(N+1)] for k in range(N+1)]
    M[0][0] = [mp.mpf(1), mp.mpf(1)]
    r = [mp.mpf(0)]*(N+1)
    for n in range(1, N+1):
        for k in range(1, n+1):
            sums = [mp.mpf(0), mp.mpf(0)]
            for m in range(n-k+1):
                C = (-1)**m * math.comb(k+m-1,m)
                u,v = M[m][n-k]
                sums[0] += C*(a*u+b*v)
                sums[1] += C*(b*u+a*v)
            M[k][n] = [sums[0]*(1-d)**k, sums[1]*(1+d)**k]
        r[n] = sum(mp.mpf((-1)**(k+1))*sum(M[k][n])/(2*k) for k in range(1,n+1))
    return r


def gcoeff(r):
    N = len(r)-1
    e, g = [mp.mpf(1)], [mp.mpf(0)]
    for n in range(1,N+1):
        e.append(-sum(k*r[k]*e[n-k] for k in range(1,n+1))/n)
        g.append(-e[n]-2*n*r[n]+sum(e[n-k]*k*r[k] for k in range(1,n+1)))
    return g


def finite_word_coefficients(delta,phi,N=8):
    """Independent enumeration of all 2^N histories; formal series via convolution.

    At q degree n the log ratio uses only the last n hosts. Starting w=0
    N steps back is therefore exact through degree N, without a burn-in claim.
    """
    out = [mp.mpf(0)]*(N+1)
    for word in itertools.product((-1,1), repeat=N):
        prob = mp.mpf('.5')
        for k in range(1,N):
            prob *= (1+phi*word[k]*word[k-1])/2
        w = [mp.mpf(0)]*(N+1)
        for z in word:
            inv = [mp.mpf(1)]+[mp.mpf(0)]*N
            for n in range(1,N+1):
                inv[n] = -sum(w[k]*inv[n-k] for k in range(1,n+1))
            w = [mp.mpf(0)]+[(1+delta*z)*inv[n-1] for n in range(1,N+1)]
        # d(log(1+w))/dq = w'/(1+w), using a new reciprocal.
        inv = [mp.mpf(1)]+[mp.mpf(0)]*N
        for n in range(1,N+1):
            inv[n] = -sum(w[k]*inv[n-k] for k in range(1,n+1))
            out[n] += prob*sum(k*w[k]*inv[n-k] for k in range(1,n+1))/n
    return out


def derived(R,V,q):
    if isinstance(R,mp.mpf):
        s = mp.exp(-R)
        a = -mp.expm1(-R)
        G = a-(1+a)*V
        D = mp.log(q)-2*R-mp.log(a)
    else:
        s = np.exp(-R)
        a = -np.expm1(-R)
        G = a-(1+a)*V
        D = np.log(q)-2*R-np.log(a)
    Ds = -G/(s*a*V)
    return s,D,Ds,G


def series_value(r,q):
    q = mp.mpf(q)
    R = mp.polyval(list(reversed(r)),q)
    V = mp.polyval(list(reversed([k*x for k,x in enumerate(r)])),q)
    return R,V,derived(R,V,q)


def bounds(q,d,high=False):
    # Extreme alternating cycle for z=1+w, at normalized s=1,c=q.
    sqrt,log = (mp.sqrt,mp.log) if high else (np.sqrt,np.log)
    B = 2*q*d-1
    lo = 2*q*(1+d)/(sqrt(B*B+4*q*(1+d))+B)
    hi = lo+2*q*d
    return log(lo),log(hi)


def spectral(q,d,p,n=48,high=False,extras=False):
    """R,V=qR' from invariant functional and differentiated Poisson equation.

    Coordinate nodes are held fixed when differentiating. This is legitimate
    because the continuum invariant rate is independent of the chosen mesh.
    The finite-mesh derivative is checked against independent finite differences.
    """
    if high:
        q,d,p = map(mp.mpf,(q,d,p))
        lo,hi = bounds(q,d,True)
        nodes = [mp.cos(mp.pi*k/(n-1)) for k in range(n)]
        weights = [mp.mpf((-1)**k)*(mp.mpf('.5') if k in (0,n-1) else 1) for k in range(n)]
        vs = [(lo+hi)/2+(hi-lo)*x/2 for x in nodes]
        DD = mp.matrix(n)
        for k in range(n):
            for l in range(n):
                if l!=k:
                    DD[k,l] = weights[l]/weights[k]/(vs[k]-vs[l])
            DD[k,k] = -sum(DD[k,l] for l in range(n) if l!=k)
        K,Ky = mp.matrix(2*n),mp.matrix(2*n)
        for j,z in enumerate((-1,1)):
            B = mp.matrix(n)
            Bd = mp.matrix(n)
            aa=[]
            for k,v in enumerate(vs):
                target = mp.log1p(q*(1+d*z)*mp.exp(-v))
                aa.append(-mp.expm1(-target))
                near = next((l for l in range(n) if abs(target-vs[l])<mp.mpf('1e-58')),None)
                if near is not None:
                    B[k,near]=1
                    for l in range(n): Bd[k,l]=DD[near,l]
                else:
                    vals = [weights[l]/(target-vs[l]) for l in range(n)]
                    norm=sum(vals)
                    for l in range(n): B[k,l]=vals[l]/norm
                    # Analytic barycentric derivative, O(n^2), avoids a
                    # high-precision dense matrix-matrix multiplication.
                    meaninv=sum(B[k,l]/(target-vs[l]) for l in range(n))
                    for l in range(n): Bd[k,l]=B[k,l]*(meaninv-1/(target-vs[l]))
            for i in range(2):
                prob=(1+p if i==j else 1-p)/2
                for k in range(n):
                    for l in range(n):
                        K[i*n+k,j*n+l]=prob*B[k,l]
                        Ky[i*n+k,j*n+l]=prob*aa[k]*Bd[k,l]
        A=K.T-mp.eye(2*n)
        rhs=mp.matrix(2*n,1)
        for l in range(2*n): A[2*n-1,l]=1
        rhs[2*n-1]=1
        mu=hp_solve(A,rhs)
        values=mp.matrix(vs+vs)
        R=(mu.T*values)[0]
        H=mp.eye(2*n)-K
        for k in range(2*n):
            for l in range(2*n): H[k,l]+=mu[l]
        f=hp_solve(H,values-mp.matrix([R]*(2*n)))
        V=(mu.T*Ky*f)[0]
        return R,V,derived(R,V,q)
    q,d,p=map(float,(q,d,p))
    lo,hi=bounds(q,d)
    nodes=np.cos(np.pi*np.arange(n)/(n-1))
    weights=(-1.)**np.arange(n)
    weights[[0,-1]]*=.5
    vs=(lo+hi)/2+(hi-lo)*nodes/2
    dif=vs[:,None]-vs[None,:]
    np.fill_diagonal(dif,1)
    DD=(weights[None,:]/weights[:,None])/dif
    np.fill_diagonal(DD,0)
    np.fill_diagonal(DD,-DD.sum(axis=1))
    K=np.empty((2*n,2*n));Ky=np.empty_like(K)
    for j,z in enumerate((-1,1)):
        target=np.log1p(q*(1+d*z)*np.exp(-vs))
        dif=target[:,None]-vs[None,:]
        hit=np.abs(dif)<1e-14*max(1,hi-lo)
        dif[hit]=1
        B=weights[None,:]/dif
        for k in np.where(hit.any(axis=1))[0]: B[k,:]=hit[k,:]
        B/=B.sum(axis=1)[:,None]
        Bd=(-np.expm1(-target))[:,None]*(B@DD)
        for i in range(2):
            prob=(1+p if i==j else 1-p)/2
            K[i*n:(i+1)*n,j*n:(j+1)*n]=prob*B
            Ky[i*n:(i+1)*n,j*n:(j+1)*n]=prob*Bd
    A=K.T-np.eye(2*n);A[-1,:]=1
    rhs=np.zeros(2*n);rhs[-1]=1
    mu=solve(A,rhs,check_finite=False)
    values=np.tile(vs,2)
    R=mu@values
    f=solve(np.eye(2*n)-K+np.ones((2*n,1))*mu[None,:],values-R,check_finite=False)
    V=mu@Ky@f
    result=(R,V,derived(R,V,q))
    if extras:
        Ew=mu@np.exp(values)
        cov=mu@(values*np.exp(values))-R*Ew
        return result+(cov-2*(Ew-np.exp(R)),)
    return result


def baseline():
    say('Registered tolerances: scalar identities 1e-40; known D 1e-12; refinement D 1e-12, Ds 1e-9.')
    # Constant host: series vs exact Perron root, independent of phi.
    r=moments('0','.7',80)
    q=mp.mpf('.02')
    R,V,_=series_value(r,q)
    lam=(1+mp.sqrt(1+4*q))/2
    check('constant host known result',abs(R-mp.log(lam))<mp.mpf('1e-40') and abs(V-(lam-1)/(2*lam-1))<mp.mpf('1e-40'),fmt(abs(R-mp.log(lam))))
    # Exact alternating product, at arbitrary q normalized s=1.
    q,d=mp.mpf('.02'),mp.mpf('.6')
    r=moments(d,'-1',80)
    R,V,(s,D,Ds,G)=series_value(r,q)
    alt=mp.log((1+s)/(1+mp.sqrt(d*d+(1-d*d)*s*s)))
    check('alternating-host known result',abs(D-alt)<mp.mpf('1e-40'),fmt(abs(D-alt)))
    s0=.5;d=.4;p=.7;D0=.0582995785286115578
    q=(1-s0)*math.exp(D0)/s0**2
    R,V,(s,D,Ds,G)=spectral(q,d,p,64)
    check('memA moderate-noise benchmark',abs(s-s0)<1e-12 and abs(D-D0)<1e-12,f's={s:.16g} D={D:.16g}')
    # Check the derivative by finite differences in log q, without Poisson differentiation.
    errs=[]
    for q,d,p in [(0.1,.4,.5),(1.,.8,.9),(10.,.4,0.)]:
        R,V,_=spectral(q,d,p,72)
        eps=2e-5
        fd=(spectral(q*math.exp(eps),d,p,72)[0]-spectral(q*math.exp(-eps),d,p,72)[0])/(2*eps)
        errs.append(abs(fd-V))
    check('Poisson derivative vs centered finite difference',max(errs)<1e-8,f'max abs={max(errs):.3g}')


def algebra_checks():
    rng=random.Random(20261115)
    maxerr=mp.mpf(0)
    for _ in range(5):
        d=mp.mpf(rng.randint(1,999))/1000
        p=mp.mpf(rng.randint(0,999))/1000
        r=moments(d,p,8)
        independent=finite_word_coefficients(d,p)
        maxerr=max(maxerr,max(abs(x-y) for x,y in zip(r,independent)))
    check('256-word enumeration vs moment coefficients: 5 cases',maxerr<mp.mpf('1e-40'),fmt(maxerr))
    worst=mp.inf; violations=0; tailratio=mp.mpf(0)
    for _ in range(100):
        d=mp.mpf(rng.randint(1,999999))/1000000
        p=mp.mpf(rng.randint(0,999999))/1000000
        q=mp.mpf(rng.randint(1,1000000))/30000000
        r=moments(d,p,24);g=gcoeff(r)
        R,V,(s,D,Ds,G)=series_value(r,q)
        A=d*d*(mp.mpf('.5')+p)
        if abs(g[2]-A)>mp.mpf('1e-50'): violations+=1
        if abs(g[3]+d*d*(4+8*p+2*p*p))>mp.mpf('1e-50'): violations+=1
        if min(g[4],g[6],g[8])<0: violations+=1
        if abs(g[5])>mp.mpf(3531)/4*d*d or abs(g[7])>mp.mpf(104991)/2*d*d: violations+=1
        low=mp.mpf('.5')-4*q-mp.mpf(3531)/4*q**3-mp.mpf(104991)/2*q**5-mp.mpf(25)/9*(10*q)**9/((1-10*q)*q*q)
        if G/(d*d*q*q)<low: violations+=1
        if R<mp.log((1+4*q)/(1+2*q))/2: violations+=1
        if q>(1-s*s)/(4*s*s-2): violations+=1
        poly=sum(g[n]*q**n for n in range(9))
        tail=mp.mpf(25)/9*d*d*(10*q)**9/(1-10*q)
        tailratio=max(tailratio,abs(G-poly)/tail)
        worst=min(worst,G/(d*d*q*q))
    check('100 random real points: all sign/coefficient/chord inequalities',violations==0,f'violations={violations}, min G/(delta^2 q^2)={fmt(worst)}, largest remainder/bound={fmt(tailratio)}')
    # Random complex histories: test invariance, derivative and resulting disk bounds.
    worstw=mp.mpf(0);worstb=mp.mpf(0);worstd=mp.mpf(0)
    for _ in range(100):
        q=mp.mpf('.1')*mp.exp(2j*mp.pi*rng.random())
        d=mp.mpf('1.2')*mp.exp(2j*mp.pi*rng.random())
        w=mp.mpc(0);der=mp.mpc(0)
        for t in range(100):
            h=1+d*rng.choice((-1,1))
            b=q*h/(1+w)**2
            new=q*h/(1+w)
            der=new-b*der
            w=new
            worstw=max(worstw,abs(w));worstb=max(worstb,abs(b));worstd=max(worstd,abs(der))
    check('100 complex histories x 100 steps: contraction and tangent bounds',worstw<mp.mpf(1)/3 and worstb<mp.mpf(1)/2 and worstd<mp.mpf(2)/3,f'max |w|={fmt(worstw)}, |b|={fmt(worstb)}, |q w_q|={fmt(worstd)}')
    # Exact rational arithmetic for the final constant.
    q=Fraction(1,30)
    margin=Fraction(1,2)-4*q-Fraction(3531,4)*q**3-Fraction(104991,2)*q**5-Fraction(25,9)*(10*q)**9/((1-10*q)*q*q)
    check('rational lower margin',margin==Fraction(185403593,1312200000) and margin>Fraction(14,100),str(margin))


def fine_series_grid():
    rows=[];maxdiff=mp.mpf(0);maxder=mp.mpf('-inf')
    qs=[mp.mpf(10)**(-k) for k in (12,10,8,6,4)] + [mp.mpf(k)/3000 for k in range(1,101)]
    for d in ('.1','.4','.8','.95'):
        for p in ('0','.5','.9'):
            r=moments(d,p,80)
            for q in qs:
                R,V,(s,D,Ds,G)=series_value(r,q)
                _,_,(_,D64,Ds64,_)=series_value(r[:65],q)
                maxdiff=max(maxdiff,abs(D-D64),abs(Ds-Ds64))
                maxder=max(maxder,Ds)
                rows.append([d,p,fmt(q,30),fmt(s,30),fmt(D,30),fmt(Ds,30),fmt(G,30)])
            say('65-digit series grid complete',d,p)
    with (HERE/'high_precision_grid.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['delta','phi','q','s','D','D_s','G']);w.writerows(rows)
    check(f'65-digit fine grid: {len(rows)} points, degrees 64 vs 80',maxder<0 and maxdiff<mp.mpf('1e-25'),f'max Ds={fmt(maxder)}, max refinement={fmt(maxdiff)}')
    say('near-s=1 sample',rows[0])


def global_grid():
    rows=[];unresolved=[];positive=[];minmargin=float('inf');maxdiffD=0.;maxdiffDs=0.
    qs=np.geomspace(.034,1e12,85)
    for d in (.1,.4,.8,.95):
        for p in (0.,.5,.9):
            for q in qs:
                n0=48 if q<1e2 else (80 if q<1e6 else 128)
                a=spectral(q,d,p,n0)
                b=spectral(q,d,p,n0+32)
                diffD=abs(a[2][1]-b[2][1]);diffDs=abs(a[2][2]-b[2][2])
                ok=diffD<1e-12 and diffDs<1e-9 and b[2][2]<-10*diffDs
                if not ok:
                    a=b;b=spectral(q,d,p,n0+96)
                    diffD=abs(a[2][1]-b[2][1]);diffDs=abs(a[2][2]-b[2][2])
                    ok=diffD<1e-12 and diffDs<1e-9 and b[2][2]<-10*diffDs
                s,D,Ds,G=b[2]
                rows.append([d,p,q,s,D,Ds,G,diffD,diffDs,ok])
                if not ok: unresolved.append(rows[-1])
                if Ds>=0: positive.append(rows[-1])
                minmargin=min(minmargin,-Ds)
                maxdiffD=max(maxdiffD,diffD);maxdiffDs=max(maxdiffDs,diffDs)
            say('global grid complete',d,p,'cumulative unresolved',len(unresolved))
    for name,data in [('global_grid.csv',rows),('unresolved_grid.csv',unresolved)]:
        with (HERE/name).open('w') as f:
            w=csv.writer(f);w.writerow(['delta','phi','q','s','D','D_s','G','D_refinement','Ds_refinement','resolved']);w.writerows(data)
    say('NUMERICAL RESULT global grid:',len(rows),'cases; nonnegative derivatives',len(positive),'unresolved at registered tolerances',len(unresolved))
    say('global minimum -Ds',minmargin,'max D refinement',maxdiffD,'max Ds refinement',maxdiffDs,'minimum s',min(x[3] for x in rows))
    for row in unresolved[:8]: say('UNRESOLVED example',row)
    # Count unresolved separately, rather than pretending refinement established the signs.
    check('resolved global cells negative',all(x[5]<0 for x in rows if x[-1]),f'{len(rows)-len(unresolved)} resolved cases')


def high_precision_collocation(n=32):
    rows=[]
    for q,d,p in [('.02','.8','.9'),('2.12','.4','.7'),('100','.4','.5'),('1e8','.4','.5')]:
        q,d,p=map(mp.mpf,(q,d,p))
        nn=n if q<1000 else max(n,80)
        t=time.monotonic()
        a=spectral(q,d,p,nn,True)
        b=spectral(q,d,p,nn+16,True)
        s,D,Ds,G=b[2]
        diffD=abs(a[2][1]-D);diffDs=abs(a[2][2]-Ds)
        say('65-digit collocation', 'q=',fmt(q),'delta=',fmt(d),'phi=',fmt(p),'nodes=',nn,nn+16,'s=',fmt(s),'D=',fmt(D),'Ds=',fmt(Ds),'Dref=',fmt(diffD),'Dsref=',fmt(diffDs),'seconds=',round(time.monotonic()-t,2))
        rows.append([fmt(x,32) for x in [q,d,p,s,D,Ds,diffD,diffDs]])
        if q<mp.mpf('.1'):
            r=moments(d,p,80);ss=series_value(r,q)
            check('independent collocation vs series R,V',abs(ss[0]-b[0])<mp.mpf('1e-30') and abs(ss[1]-b[1])<mp.mpf('1e-30'),fmt(max(abs(ss[0]-b[0]),abs(ss[1]-b[1]))))
        elif diffD>mp.mpf('1e-12') or diffDs>mp.mpf('1e-9'):
            say('UNRESOLVED high-precision discretization at registered tolerance')
    with (HERE/f'high_precision_collocation_{n}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['q','delta','phi','s','D','D_s','D_refinement','Ds_refinement']);w.writerows(rows)


def route_checks():
    # Pointwise coupling obstruction, at fixed trial D=0, entering U=1.
    check('simple pathwise s-order fails in opposite host states',1-(1-.4)>0 and 1-(1+.4)<0,'derivatives +0.4 and -0.4')
    for q,d,p in [(.1,.4,0.),(1.,.8,.5),(100.,.4,.9)]:
        R,V,vals,gap=spectral(q,d,p,96,extras=True)
        say('route-a diagnostic: q,d,p=',q,d,p,'Cov(X,expX)-2 JensenGap=',gap,'(not a gamma derivative identity)')


def extended_hp_grid():
    """Dated extension: independently diagnose near-zero derivative accuracy.

    49 logarithmically spaced q points, 0.1 through 1e12, at delta=.4,
    phi=.5; plus the worst original double-precision refinement cell.
    Registered tolerances stay D 1e-12 and Ds 1e-9. This is a refinement,
    not an expansion of the hypothesis or a change in its pass criteria.
    """
    global FAST_HP
    ref=spectral('2.12','.4','.7',32,True)
    FAST_HP=True
    fast=spectral('2.12','.4','.7',32,True)
    err=max(abs(x-y) for x,y in zip(ref[:2],fast[:2]))
    check('iterative refinement vs 65-digit full LU',err<mp.mpf('1e-50'),fmt(err))
    qs=[mp.mpf(10)**(-1+mp.mpf(13)*k/48) for k in range(49)]
    cases=[(q,mp.mpf('.4'),mp.mpf('.5')) for q in qs]
    cases.append((mp.mpf('691290300440.24'),mp.mpf('.95'),mp.mpf(0)))
    rows=[];unresolved=0;maxDd=mp.mpf(0);maxDsd=mp.mpf(0)
    for q,d,p in cases:
        n=32 if q<100 else (48 if q<mp.mpf('1e4') else (80 if q<mp.mpf('1e8') else 112))
        a=spectral(q,d,p,n,True);b=spectral(q,d,p,n+24,True)
        s,D,Ds,G=b[2]
        dd=abs(a[2][1]-D);dsd=abs(a[2][2]-Ds)
        ok=dd<mp.mpf('1e-12') and dsd<mp.mpf('1e-9') and Ds<-10*dsd
        if not ok:
            a=b;b=spectral(q,d,p,n+56,True)
            s,D,Ds,G=b[2];dd=abs(a[2][1]-D);dsd=abs(a[2][2]-Ds)
            ok=dd<mp.mpf('1e-12') and dsd<mp.mpf('1e-9') and Ds<-10*dsd
        unresolved+=not ok
        maxDd=max(maxDd,dd);maxDsd=max(maxDsd,dsd)
        rows.append([fmt(x,32) for x in [q,d,p,s,D,Ds,G,dd,dsd]]+[ok])
        say('HP grid',len(rows),'q=',fmt(q,9),'s=',fmt(s),'D=',fmt(D),'Ds=',fmt(Ds),'Dref=',fmt(dd,5),'Dsref=',fmt(dsd,5),'resolved=',ok)
    with (HERE/'extended_high_precision_grid.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['q','delta','phi','s','D','D_s','G','D_refinement','Ds_refinement','resolved']);w.writerows(rows)
    check('extended 65-digit grid',unresolved==0,f'cases={len(rows)}, unresolved={unresolved}, max Dref={fmt(maxDd)}, max Dsref={fmt(maxDsd)}')
    say('largest high-precision linear residual',fmt(HP_RESIDUAL))
    # Independent high-precision finite difference of the map q -> (s,D).
    q=mp.mpf('100');eps=mp.mpf('1e-12')
    left=spectral(q*mp.exp(-eps),'.4','.5',64,True)[2]
    right=spectral(q*mp.exp(eps),'.4','.5',64,True)[2]
    center=spectral(q,'.4','.5',64,True)[2]
    fd=(right[1]-left[1])/(right[0]-left[0])
    check('65-digit D_s vs independent threshold-curve secant',abs(fd-center[2])<mp.mpf('1e-20'),fmt(abs(fd-center[2])))


def supplementary_audit():
    rng=random.Random(20261116)
    maxcover=mp.mpf(0)
    for _ in range(20):
        q=mp.mpf(rng.randint(1,249000))/1000000
        d=mp.mpf(rng.randint(1,999999))/1000000
        intervals=[(mp.mpf(0),2*q)]
        for depth in range(1,11):
            intervals=[(q*(1+d*z)/(1+hi),q*(1+d*z)/(1+lo)) for lo,hi in intervals for z in (-1,1)]
            cover=sum(hi-lo for lo,hi in intervals)
            bound=2*q*(4*q)**depth
            maxcover=max(maxcover,cover/bound)
    check('singular-support covering lemma: 20 random cases, depths 1..10',maxcover<=1,fmt(maxcover))
    violations=0;maxsym=mp.mpf(0);maxquot=mp.mpf(0)
    for _ in range(20):
        q=mp.mpf('.04')*mp.exp(2j*mp.pi*rng.random())
        d=mp.mpf('1.19')*mp.exp(2j*mp.pi*rng.random())
        p=mp.mpf(rng.randint(0,999))/1000
        r=moments(d,p,50);rn=moments(-d,p,50)
        R=mp.polyval(list(reversed(r)),q)
        V=mp.polyval(list(reversed([k*x for k,x in enumerate(r)])),q)
        G=1-mp.exp(-R)-(2-mp.exp(-R))*V
        maxsym=max(maxsym,max(abs(x-y) for x,y in zip(r,rn)))
        maxquot=max(maxquot,abs(G)/(abs(d)**2))
        if abs(R)>mp.log(mp.mpf('1.5')) or abs(V)>1 or abs(G)>4 or abs(G)>mp.mpf(25)/9*abs(d)**2: violations+=1
    check('20 complex parameter points: expectation/delta symmetry/disk inequalities',violations==0 and maxsym<mp.mpf('1e-40'),f'violations={violations}, symmetry error={fmt(maxsym)}, max |G/delta^2|={fmt(maxquot)}; degree-50 evaluation, numerical only')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','full','hp','hpgrid','audit'],default='full');ap.add_argument('--hp-n',type=int,default=32);args=ap.parse_args()
    t=time.monotonic()
    say('gap1-1 deterministic numerical checks; mpmath dps=',mp.mp.dps,'mode=',args.mode)
    if args.mode=='audit': supplementary_audit()
    elif args.mode=='hpgrid': extended_hp_grid()
    elif args.mode=='hp': high_precision_collocation(args.hp_n)
    else:
        baseline()
        algebra_checks()
        route_checks()
        if args.mode=='full':
            fine_series_grid()
            global_grid()
            high_precision_collocation(32)
    say('FAILED CHECKS:',len(FAIL),FAIL,'elapsed seconds=',round(time.monotonic()-t,2))
    raise SystemExit(bool(FAIL))


if __name__=='__main__': main()
