#!/usr/bin/env python3
"""Independent checks of A13 formulas, controlled remainders, and outage bounds.

Each check function states its single claim. Protocol and tolerances were written
before execution. No imports from collaborators' code and no empirical data.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import time
import mpmath as mp
import numpy as np
from scipy.optimize import brentq

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 80
RESULTS = {}


def matrix(h, taus, d, high=False):
    """Column population convention, host on edge 0->1, tilt on last->0."""
    n = len(taus)
    if high:
        taus = list(map(mp.mpf, taus)); h = mp.mpf(h); d = mp.mpf(d)
        mat = mp.zeros(n)
        for j, tau in enumerate(taus):
            mat[j, j] = 1-1/tau
            mat[(j+1) % n, j] = (h if j == 0 else mp.exp(d) if j == n-1 else 1)/tau
    else:
        taus = np.asarray(taus, dtype=float)
        mat = np.diag(1-1/taus)
        for j, tau in enumerate(taus):
            mat[(j+1) % n, j] = (h if j == 0 else math.exp(d) if j == n-1 else 1)/tau
    return mat


def growth(host, taus, d, high=False):
    n = len(taus)
    if high:
        prod = mp.eye(n); scale = mp.mpf(0)
        for h in host:
            prod = matrix(h, taus, d, True)*prod
            z = max(abs(x) for x in prod)
            prod /= z; scale += mp.log(z)
        if n == 2:
            tr = prod[0,0]+prod[1,1]
            rho = (tr+mp.sqrt((prod[0,0]-prod[1,1])**2+4*prod[0,1]*prod[1,0]))/2
        else:
            rho = max(abs(x) for x in mp.eig(prod, left=False, right=False))
        return (mp.log(rho)+scale)/len(host)
    prod = np.eye(n); scale = 0.
    for h in host:
        prod = matrix(float(h), taus, d)@prod
        z = np.max(prod)
        prod /= z; scale += math.log(z)
    rho = max(abs(np.linalg.eigvals(prod)))
    return (math.log(rho)+scale)/len(host)


def root(host, taus, high=False):
    if not high:
        lo, hi = -1., 1.
        while growth(host, taus, lo) >= 0: lo *= 2
        while growth(host, taus, hi) <= 0: hi *= 2
        return brentq(lambda d: growth(host, taus, d), lo, hi, xtol=2e-13)
    guess = root(list(map(float,host)), list(map(float,taus)))
    return mp.findroot(lambda d: growth(host, taus, d, True),
                       (mp.mpf(guess)-mp.mpf('.001'), mp.mpf(guess)+mp.mpf('.001')),
                       tol=mp.mpf('1e-65'))


def alt(a,s):
    a,s = mp.mpf(a),mp.mpf(s)
    return mp.log((2*a-1)*(2*s-1)/(2*(a*(2*s-1)-(s-1))))


def bound(host, taus, use_min=False, high=False):
    pos = [h for h in host if h>0]
    if high:
        q = mp.mpf(len(host)-len(pos))/len(host)
        lg = mp.fsum(mp.log(h) for h in pos)/len(pos)
        ps = [1-1/mp.mpf(t) for t in taus]
        park = min(ps) if use_min else max(ps)
        lam = mp.exp(-q/(1-q)*mp.log(park))
        return mp.fsum(mp.log(mp.mpf(t)*(lam-p)) for t,p in zip(taus,ps))-lg
    q = (len(host)-len(pos))/len(host)
    lg = np.log(pos).mean()
    ps = 1-1/np.asarray(taus,dtype=float)
    park = min(ps) if use_min else max(ps)
    lam = math.exp(-q/(1-q)*math.log(park))
    return sum(math.log(t*(lam-p)) for t,p in zip(taus,ps))-lg


def check_benchmarks():
    """Reproduce the known constant and strictly-positive period-two results."""
    errors = [abs(root([1]*3, ts)) for ts in [(2,1),(2,2),(2,5),(2,2,20)]]
    target = mp.log((mp.mpf('.625')-mp.sqrt(mp.mpf('.300625')))/mp.mpf('.08'))
    errors.append(abs(root([1.6,.4],(2,2))-float(target)))
    assert max(errors)<1e-12, errors
    hp = abs(root([mp.mpf('1.6'),mp.mpf('.4')],(2,2),True)-target)
    assert hp<mp.mpf('1e-55')
    RESULTS['benchmarks'] = dict(cases=6,max_double_error=max(errors),mp_error=str(hp))
    print('BENCHMARKS',RESULTS['benchmarks'],flush=True)


def check_alternation():
    """The supplied outage threshold, its strict signs, and derivative are exact."""
    errors=[]; deriv_errors=[]
    for a,s in itertools.product(['1.01','1.25','2','5','50'],['1','1.1','2','5','20','1000']):
        d=alt(a,s)
        errors.append(abs(growth([2,0],(a,s),d,True)))
        expected = 1/((2*mp.mpf(s)-1)*((2*mp.mpf(a)-1)*mp.mpf(s)-mp.mpf(a)+1))
        deriv_errors.append(abs(mp.diff(lambda ss:alt(a,ss),mp.mpf(s))-expected))
        assert d<0 and expected>0
    assert max(errors)<mp.mpf('1e-55')
    assert max(deriv_errors)<mp.mpf('1e-45')
    RESULTS['alternation']=dict(cases=30,max_growth_residual=str(max(errors)),max_derivative_error=str(max(deriv_errors)))
    print('ALTERNATION',RESULTS['alternation'],flush=True)


def check_adversarial():
    """Attempt to falsify the p_max upper bound with a preregistered finite search."""
    rng=np.random.default_rng(13112026)
    best=(math.inf,None)
    candidate=None
    grids=[(2,1),(2,2),(2,5),(1.001,100),(100,1.001),
           (2,2,1),(1,100,1),(100,1,1),(1,1,100),
           (1.001,100,2),(2,100,1.001),(100,2,1.001),
           (1,1,1,100),(1,1,100,1),(100,1,1,1),(2,3,20,1)]
    started=time.monotonic()
    for case in range(5000):
        n=[4,8,16,32,64,128][case%6]
        nz=int(rng.integers(1,n))
        mode=(case//6)%5
        pos=np.ones(n-nz)
        if mode==1: pos=np.exp(rng.uniform(-20,20,n-nz))
        elif mode==2: pos=1+rng.pareto(.35,n-nz)
        elif mode==3: pos=np.where(rng.random(n-nz)<.5,1e-8,1e4)
        elif mode==4: pos=np.geomspace(1e-6,1e6,n-nz)
        host=np.r_[np.zeros(nz),pos]
        if case%3==1:rng.shuffle(host)
        elif case%3==2:host=np.roll(host, nz//2)
        host*=n/host.sum()
        taus=grids[(case//3)%len(grids)]
        ds=bound(host,taus)
        g=growth(host,taus,ds)
        if g<best[0]:best=(g,dict(case=case,taus=taus,host=host.tolist(),D_sup=ds))
        if g < -1e-8:
            hm=[mp.mpf(float(h)) for h in host]
            hm=[h*len(hm)/mp.fsum(hm) for h in hm]
            dm=bound(hm,taus,high=True); gm=growth(hm,taus,dm,True)
            if gm < -mp.mpf('1e-40'):
                dr=root(hm,taus,True)
                candidate=dict(case=case,taus=taus,host=[str(h) for h in hm],
                               D_sup=str(dm),growth_at_sup=str(gm),D=str(dr),violation=str(dr-dm),
                               root_residual=str(growth(hm,taus,dr,True)))
                assert abs(mp.mpf(candidate['root_residual']))<mp.mpf('1e-45')
                break
        if time.monotonic()-started>15*60: break
    RESULTS['adversarial']=dict(cases=case+1,minimum_growth=best[0],worst_case=best[1],
                                 counterexample=candidate,seconds=time.monotonic()-started)
    print('ADVERSARIAL',json.dumps(RESULTS['adversarial']),flush=True)


PATTERNS = [[2,0],[3,0,0],[2,2,0,0],[0,2,0,1,2],[0,0,0,0,0,0,0,1,2,7]]


def coefficients(host,a):
    h=list(map(mp.mpf,host)); a=mp.mpf(a); p=1-1/a; n=len(h)
    cov=[mp.fsum((h[t]-1)*(h[(t-l)%n]-1) for t in range(n))/n for l in range(n+1)]
    acov=cov[0]/2+mp.fsum(p**(l-1)*cov[l] for l in range(1,n+1))/(1-p**n)
    u=[-mp.fsum(p**(l-1)*(h[(t-l)%n]-1) for l in range(1,n+1))/(1-p**n) for t in range(n)]
    afilter=cov[0]/2-mp.fsum(h[t]*u[t] for t in range(n))/n
    return acov,afilter,cov,u


def constants(host,a):
    b=1/mp.mpf(a); H=(mp.fsum(mp.mpf(h)**4 for h in host)/len(host))**mp.mpf('.25')
    L=2*(4*H+1)/b; N=L*(6*H+1)/b
    Rg=2*H*N+H*L*(6*H+2)+mp.mpf(8)/3*(4*H+1)**3
    B0=4*H**2/b+2/b+2*H**2+3
    J=4*H**2/b+2*H**2+1/b+1
    K0=B0+Rg; C=J*K0+Rg+2*K0**2
    e0=min(b/2,1/(4*(B0+Rg+1)))
    return dict(H=H,L=L,N=N,Rg=Rg,B0=B0,J=J,K0=K0,C=C,e0=e0)


def check_longstay():
    """Raw covariance predicts the threshold with proved value/derivative bounds."""
    rows=[]; coefficient_error=mp.mpf(0); max_ratio=mp.mpf(0); max_rate_ratio=mp.mpf(0)
    max_derivative_ratio=mp.mpf(0); max_scaled_error=mp.mpf(0)
    for host,a0 in itertools.product(PATTERNS,['1.25','2','5']):
        a=mp.mpf(a0); b=1/a; p=1-b; n=len(host)
        A,Af,cov,u=coefficients(host,a)
        coefficient_error=max(coefficient_error,abs(A-Af))
        assert abs(A-Af)<mp.mpf('1e-45')
        cc=constants(host,a); C=cc['C']; e0=cc['e0']
        diagnostics=[]
        for s in [100,1000,10000]:
            d=root(host,(a,s),True)
            diagnostics.append(dict(s=s,D=str(d),scaled_error=str(s*d-A)))
            if s==10000:max_scaled_error=max(max_scaled_error,abs(s*d-A))
        enclosures=[]
        for e in [e0/2,e0/4]:
            d=root(host,(a,1/e),True)
            residual=abs(d-A*e); radius=C*e**2
            assert residual<=radius+mp.mpf('1e-40')
            max_ratio=max(max_ratio,residual/radius)
            enclosures.append(dict(epsilon=str(e),D=str(d),error=str(residual),radius=str(radius)))
        e=e0/2
        E2=mp.fsum(mp.mpf(x)**2 for x in host)/n
        S=mp.fsum(p**(l-1)*(cov[l]+1) for l in range(1,n+1))/(1-p**n)
        for c in [mp.mpf('.5'),mp.mpf(1),mp.mpf(2)]:
            B=c*c*S-c/b+(c*c*E2-2*c+1)/2
            r=growth(host,(a,1/e),mp.log(c),True)
            error=abs(r/e-(c-1)+e*B); radius=cc['Rg']*e**2
            assert error<=radius+mp.mpf('1e-40')
            max_rate_ratio=max(max_rate_ratio,error/radius)
        R=b/(256*mp.mpf(max(host))**2)
        de=None
        if abs(A)>mp.mpf('1e-50'):
            e=min(R/4,abs(A)*R**2/8); s=1/e
            d=root(host,(a,s),True)
            ds=-mp.diff(lambda ss:growth(host,(a,ss),d,True),s)/mp.diff(lambda dd:growth(host,(a,s),dd,True),d)
            error=abs(s*s*ds+A); radius=2*e/R**2
            assert error<=radius+mp.mpf('1e-40')
            assert mp.sign(ds)==-mp.sign(A) and mp.sign(d)==mp.sign(A)
            max_derivative_ratio=max(max_derivative_ratio,error/radius)
            de=dict(epsilon=str(e),scaled_derivative=str(s*s*ds),A=str(A),radius=str(radius))
        # The fixed-factor reversal conclusion uses only the general moment theorem.
        if abs(A)>mp.mpf('1e-50'):
            e=min(e0/2,abs(A)*mp.mpf('.5')/(C*mp.mpf('2.5')))
            d1=root(host,(a,1/e),True); d2=root(host,(a,2/e),True)
            assert mp.sign(d2-d1)==-mp.sign(A)
        rows.append(dict(host=host,a=a0,A=str(A),diagnostics=diagnostics,enclosures=enclosures,derivative=de))
    RESULTS['longstay']=dict(host_cells=len(rows),diagnostic_roots=45,value_enclosures=30,
        growth_enclosures=45,derivative_enclosures=sum(r['derivative'] is not None for r in rows),
        fixed_factor_orderings=sum(r['derivative'] is not None for r in rows),
        coefficient_error=str(coefficient_error),max_value_error_to_radius=str(max_ratio),
        max_rate_error_to_radius=str(max_rate_ratio),max_derivative_error_to_radius=str(max_derivative_ratio),
        max_scaled_error_at_10000=str(max_scaled_error),rows=rows)
    print('LONGSTAY',json.dumps({k:v for k,v in RESULTS['longstay'].items() if k!='rows'}),flush=True)
    for row in rows:print('COEFFICIENT',row['host'],row['a'],row['A'],'s=10000 error',row['diagnostics'][-1]['scaled_error'],flush=True)


def check_independence():
    """Independent nonnegative matrices have mean product equal to mean-matrix power."""
    errors=[]
    for q,taus in itertools.product([.25,.5,.9],[(2,1),(2,5),(5,2)]):
        on=1/(1-q); aa=[matrix(0,taus,0),matrix(on,taus,0)]
        pairs=[(1.,np.eye(2))]
        for _ in range(12):
            pairs=[(prob*w,aa[j]@prod) for prob,prod in pairs for j,w in enumerate([q,1-q])]
        actual=sum(prob*prod for prob,prod in pairs)
        expected=np.linalg.matrix_power(q*aa[0]+(1-q)*aa[1],12)
        err=float(np.max(abs(actual-expected))/max(1.,np.max(abs(expected))))
        errors.append(err); assert err<1e-12
    RESULTS['independence']=dict(cases=9,histories_per_case=4096,max_relative_error=max(errors))
    print('INDEPENDENCE',RESULTS['independence'],flush=True)


def check_iid_runs():
    """The exactly summed iid binary-outage stationary run law gives D>=0."""
    rows=[]
    for a0,q0 in itertools.product(['1.25','2','5'],['.1','.25','.5','.75','.9']):
        a,q=mp.mpf(a0),mp.mpf(q0); p=1-1/a; on=1/(1-q)
        # Uniform tail bound for D in [-2,60].
        envelope=max(abs(mp.log(p)),abs(mp.log(p+mp.exp(60)*on/(a*p))))
        M=int(mp.ceil(mp.log(mp.mpf('1e-58')/envelope)/mp.log(1-q)))
        tail=envelope*(1-q)**(M+1)
        assert tail<mp.mpf('1e-55')
        def run_growth(d):
            c=mp.exp(d); rr=mp.mpf(0); prob=q; total=q*mp.log(p)
            for m in range(1,M+1):
                rr=(on/a)/(p+c*rr); prob*=1-q
                total+=prob*mp.log(p+c*rr)
            return total
        # Secant at a double-precision bracket, then arbitrary-precision refinement.
        gd=brentq(lambda d:float(run_growth(mp.mpf(d))),-2,60,xtol=1e-12)
        d=mp.findroot(run_growth,(mp.mpf(gd)-mp.mpf('.001'),mp.mpf(gd)+mp.mpf('.001')),tol=mp.mpf('1e-65'))
        residual=abs(run_growth(d))
        assert residual<mp.mpf('1e-45') and d>=-mp.mpf('1e-40')
        rows.append(dict(a=a0,q0=q0,D=str(d),terms=M,tail_bound=str(tail),residual=str(residual)))
    RESULTS['iid_runs']=dict(cases=len(rows),max_tail_bound=str(max(mp.mpf(r['tail_bound']) for r in rows)),
                             max_residual=str(max(mp.mpf(r['residual']) for r in rows)),rows=rows)
    print('IID_RUNS',json.dumps(RESULTS['iid_runs']),flush=True)


def block_growth(m,taus,d):
    """Periodic growth from exact matrix powers, independently of path iteration."""
    zero=matrix(0,taus,d,True); on=matrix(2,taus,d,True)
    prod=(on**m)*(zero**m)
    z=max(abs(x) for x in prod); prod/=z
    if len(taus)==2:
        rho=(prod[0,0]+prod[1,1]+mp.sqrt((prod[0,0]-prod[1,1])**2+4*prod[0,1]*prod[1,0]))/2
    else:rho=max(abs(x) for x in mp.eig(prod,left=False,right=False))
    return (mp.log(z)+mp.log(rho))/(2*m)


def check_blocks_and_limit():
    """The bound is approached by long pure blocks and has the corrected mixed-on limit."""
    rows=[]
    for taus in [(2,1),(2,5),(1,2,20),(100,1,1)]:
        sup=bound([mp.mpf(0),mp.mpf(2)],taus,high=True)
        guess=sup
        for m in [1,8,64,512,4096]:
            d=mp.findroot(lambda dd:block_growth(m,taus,dd),(guess-mp.mpf('.01'),guess+mp.mpf('.01')),tol=mp.mpf('1e-65'))
            assert d<=sup+mp.mpf('1e-40')
            assert abs(block_growth(m,taus,d))<mp.mpf('1e-45')
            rows.append(dict(taus=taus,m=m,D=str(d),Dsup=str(sup),gap=str(sup-d)))
            guess=d
    limit=-mp.log(mp.sqrt(3)/2)
    actual=bound([mp.mpf(0),mp.mpf(0),mp.mpf(1),mp.mpf(3)],(2,10**8),high=True)
    assert abs(actual-limit)<mp.mpf('1e-6')
    RESULTS['blocks']=dict(cases=20,rows=rows,mixed_limit=str(limit),mixed_bound_at_1e8=str(actual),mixed_limit_error=str(abs(actual-limit)))
    print('BLOCKS',json.dumps(RESULTS['blocks']),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--part',choices=['search','theory','all'],default='all')
    args=parser.parse_args()
    if args.part=='theory':
        RESULTS.update(json.loads((HERE/'a13_results_search.json').read_text()))
        print('REUSING retained search run: 6 benchmarks, 30 alternating cells, 5000 adversarial cells; no violations.',flush=True)
    else:
        check_benchmarks(); check_alternation(); check_adversarial()
        (HERE/'a13_results_search.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    if args.part!='search':
        check_longstay(); check_independence(); check_iid_runs(); check_blocks_and_limit()
        (HERE/'a13_results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    print('PASS all requested checks in this invocation',flush=True)

if __name__=='__main__':main()
