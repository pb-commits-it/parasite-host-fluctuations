#!/usr/bin/env python3
"""GAP 4: rigorous enclosures of memory computations, with explicit tails.

Run python3 -I gap4/gap4_certify.py.
Requires python-flint==0.8.0, installed locally in _deps (or on sys.path).
No imports from collaborators' scripts. No random sampling. Decimal inputs
are exact mathematical decimals enclosed by Arb, never Python floats.
Every reported mathematical interval includes roundoff and its stated tail.
An OPEN result is intentional, not a passing numerical certificate.
"""
from pathlib import Path
import sys
import math
import json
import time
import csv
import hashlib
from collections import defaultdict

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE / '_deps'))
import flint
from flint import arb, arb_series, arb_mat, ctx

ctx.dps = 100
ctx.cap = 3
RECORDS = []


def ball(x, radius):
    return x + arb(0, radius.abs_upper())


def show(x):
    return x.str(24)


def record(name, method, values, detail='', certified=True):
    row = dict(claim=name, method=method,
               enclosure={k: show(v) if isinstance(v, arb) else v
                          for k, v in values.items()},
               detail=detail, certified=certified)
    RECORDS.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def F(s, phi):
    a = 1-s
    return a*(1+(a+2)*phi)/(2*(1+a)*(1+a*phi))


def Fs(s, phi):
    a = 1-s
    return -(1+2*phi+2*a*phi-a*a*phi*phi)/(2*(1+a)**2*(1+a*phi)**2)


def star(phi):
    return 1-(1-(2*(1+phi)).sqrt())/phi


def invariant_series(b, delta, phi, order, degree=2):
    """G(b,delta)=E log(U/lambda), with b=c/lambda^2.

    Scalar parity form of memA (26). At order n the one nonzero joint
    moment is E[z^k Z^(n mod 2)]. This halves the two-state calculation.
    arb_series supplies derivatives in b; its O(x^p) is NOT a tail bound.
    The separate Cauchy tails below provide that bound.
    """
    bjet = arb_series([b, 1], prec=degree+1)
    zero = arb_series([], prec=degree+1)
    mom = [[zero for _ in range(order+1)] for _ in range(order+1)]
    mom[0][0] = arb_series([1], prec=degree+1)
    bp = [bjet**k for k in range(order+1)]
    result = zero
    coefficient2 = None
    for n in range(1, order+1):
        for k in range(n, 0, -1):
            rhs = zero
            for r in range(k+1):
                transition = phi if (n-r) % 2 else arb(1)
                for m in range(n-k+1):
                    if r == m == 0:
                        continue
                    ell = k-r+m
                    if ell == 0 and n-r != 0:
                        continue
                    factor = math.comb(k, r)*(-1)**(k-r+m)*math.comb(k+m-1, m)
                    rhs += mom[ell][n-r]*(factor*transition)
            denominator = 1-(-1)**k*bp[k]*(phi if n % 2 else 1)
            mom[k][n] = bp[k]*rhs/denominator
        if n % 2 == 0:
            cn = sum((mom[k][n]*arb((-1)**(k+1))/k for k in range(1, n+1)), zero)
            result += cn*delta**n
            if n == 2:
                coefficient2 = cn[0]
    return [result[k]*math.factorial(k) for k in range(degree+1)], coefficient2


def series_with_tail(b, delta, phi, order, degree=2):
    """Extend P7 to |b_complex|<=B=b.upper()+h.

    h=min(b.lower(),1-b.upper())/100.

    R=(1-B)^2/(8B), q=|delta|/R, T=log(2) q^(N+1)/(1-q).
    The P7 contraction proof uses only |b|, hence holds uniformly on this
    complex disk. Cauchy's estimate bounds kth b-derivative of the omitted
    delta-series by k! T/h^k, including at all real points in the b ball.
    """
    assert b > 0 and b < 1
    h = min(b.lower(),1-b.upper())/100
    B = b.upper()+h
    radius = (1-B)**2/(8*B)
    q = delta.abs_upper()/radius
    assert q < 1, 'P7 radius is insufficient for this case'
    T = arb(2).log()*q**(order+1)/(1-q)
    raw, _ = invariant_series(b, delta, phi, order, degree)
    tails = [math.factorial(k)*T/h**k for k in range(degree+1)]
    return [ball(v, e) for v, e in zip(raw, tails)], radius, tails


def threshold(s, delta, phi, order=24, guess=None, derivative=True):
    """Bracket b via residual and a strictly positive derivative.

    r=log(s/(1-b))+G(b). P3 supplies existence and uniqueness. The final
    local sign certificate is also explicit: r_b>=m>0 on b0+-h, and
    |r(b0)|<m*h. This implies opposite endpoint signs by integration.
    Uncertified midpoint Newton iterates only choose the bracket centre.
    """
    if guess is None:
        guess = delta**2*F(s, phi)
    c = (1-s)*guess.exp()
    lam = (s+(s*s+4*c).sqrt())/2
    b0 = (c/lam**2).mid()
    # Candidate generation at modest order; not used as a certificate.
    for _ in range(5):
        g, _ = invariant_series(b0, delta, phi, min(order, 32), 1)
        r = (s/(1-b0)).log()+g[0]
        rb = 1/(1-b0)+g[1]
        b0 = (b0-r/rb).mid()
    g0, radius, tails = series_with_tail(b0, delta, phi, order, 0)
    r0 = (s/(1-b0)).log()+g0[0]
    # A preliminary disk allows a derivative bound independent of the root.
    outer_h = b0*arb('1e-10')
    outer = ball(b0, outer_h)
    gout, _, _ = series_with_tail(outer, delta, phi, order, 1)
    lower_slope = (1/(1-outer)+gout[1]).lower()
    assert lower_slope > 0
    root_h = (2*r0.abs_upper()/lower_slope + b0*arb('1e-85')).upper()
    assert root_h < outer_h
    assert r0.abs_upper() < lower_slope*root_h
    b = ball(b0, root_h)
    D = b.log()+2*(s/(1-b)).log()-(1-s).log()
    values = {'D': D, 'b': b, 'rate_tail': tails[0], 'delta_radius': radius,
              'root_halfwidth_b': root_h, 'rate_residual_at_center': r0,
              'lower_rate_slope_b': lower_slope}
    if derivative:
        g, _, errors = series_with_tail(b, delta, phi, order, 1)
        denom = 1+(1-b)*g[1]
        assert denom > 0
        J = b-1+s+b*(2-s)*(1-b)*g[1]
        Ds = J/(s*(1-s)*b*denom)
        values.update(Ds=Ds, derivative_tail_Gb=errors[1])
    return values


def basic_certificates():
    # Registered known-result checks precede any new stationary computation.
    for stext in ['0.35', '0.75', '0.99']:
        s = arb(stext)
        b = 1-s
        g, c2 = invariant_series(b, arb(0), arb('0.8'), 4)
        assert g[0].contains(0) and g[0].rad() < arb('1e-60')
        # Exact coefficient of rate at deterministic c=1-s.
        discrepancy = c2 + b/(1+b)*F(s, arb('0.8'))
        assert discrepancy.contains(0) and discrepancy.rad() < arb('1e-60')
    record('BASELINE', 'interval', {'tolerance': arb('1e-60')},
           '3 zero-noise cases and 3 known second-order rate coefficients passed.')
    R = arb(1)/256
    E = (arb(1)/4).exp()
    vals = {'R':R, 'invariant_ball_factor':1+2*E/(1-4*R),
            'contraction_factor':2*R*E/(1-4*R)**2,
            'f_error_coefficient':(8*E+8)/(1-4*R),
            '19R':19*R, 'Rouche_lower':arb(3)/2-E,
            'R2_over_4':R**2/4, 'R3_over_2':R**3/2}
    assert vals['invariant_ball_factor'] < 4
    assert vals['contraction_factor'] < 1
    assert vals['f_error_coefficient'] < 19
    assert 19*R < arb('0.075') < vals['Rouche_lower']
    assert vals['Rouche_lower'] > arb('0.21')
    record('P5 numerical constants', 'interval', vals)
    A = -arb(1)/16
    eps = arb('1e-8')
    derivative_bound = 2*eps/R**2
    assert eps < abs(A)/262144
    assert -A-derivative_bound > arb(1)/32
    record('CE1 analytic margin', 'interval plus truncation bound',
           {'A':A, 'allowed_epsilon':abs(A)/262144,
            'Ds':ball(-A, derivative_bound)},
           'P5 Cauchy derivative tail 2 epsilon/R^2; the counterexample itself is analytic.')
    phi = -arb('0.35')
    st = star(phi)
    record('Sign-table and ring critical values', 'interval',
           {'phi_crit':2-arb(5).sqrt(), 'phi_exception':-arb(1)/2,
            's_star(-.35)':st, 'F_at_s_star':F(st,phi),
            'ring_a_crit':(3+arb(17).sqrt())/4,
            'ring_staying_crit':1-4/(3+arb(17).sqrt()),
            'secondary_printed_staying_expression':(5-arb(17).sqrt())/4,
            'ring_c3_crit':arb(2), 'third_peak_c3_5':arb(46)/13},
           'Boundary locations and sign classification are analytic; these are decimal enclosures. '
           'secondary prints (5-sqrt(17))/4 beside .4384; the staying threshold 1-1/a_crit is (5-sqrt(17))/2.')
    for delta_text in ['.2','.5','.6','.9']:
        d = arb(delta_text)
        for st in ['0','.2','.7','.99']:
            s = arb(st)
            Q = (d*d+(1-d*d)*s*s).sqrt()
            D = ((1+s)/(1+Q)).log()
            Ds = 1/(1+s)-(1-d*d)*s/(Q*(1+Q))
            c = (1-s*s)/(1+Q)
            trace = s*s+2*c
            determinant = c*c*(1-d*d)
            perron = (trace+(trace*trace-4*determinant).sqrt())/2
            assert D < 0 and Ds > 0
            assert (perron-1).contains(0) and (perron-1).abs_upper() < arb('1e-90')
            record('CE4 delta='+delta_text+' s='+st, 'interval',
                   {'D':D,'Ds':Ds,'Perron_minus_1':perron-1,
                    'ell':-(1-d*d).log()/2},
                   'General signs and exact Perron identity are analytic (P8).')


def binary_certificates():
    cases = [
        ('CE1','0.99999999','.5','-.75',20),
        ('CE2','.9','.4','-.75',88),
        ('CE3','0.999999','.6','-.5',20),
    ]
    for name, s, delta, phi, n in cases:
        out = threshold(arb(s),arb(delta),arb(phi),n)
        assert out['D'] < 0 and out['Ds'] > 0
        if name == 'CE1': assert out['Ds'] > arb(1)/32
        if name == 'CE3':
            out['D_over_eps2'] = out['D']/arb('1e-12')
            out['Ds_over_eps'] = out['Ds']/arb('1e-6')
        record(name, 'interval plus truncation bound', out, 'P7 order '+str(n)+'; derivative tail by complex-b Cauchy estimate.')
    for ph in ['.8','-.75','-.5']:
        d,p = arb('.6'),arb(ph)
        A=d*d*(arb('.5')+p)
        B=-d*d*(arb('.5')+p+p*p)+arb('1.5')*A*A
        record('P5 coefficient phi='+ph,'interval',{'A':A,'B':B},'Coefficient formulas and limits are analytic.')
        for ep in ['.02','.005','.001']:
            eps=arb(ep)
            out=threshold(1-eps,d,p,30,derivative=False)
            out['extracted_B']=(out['D']-A*eps)/eps**2
            record('P5 epsilon='+ep+' phi='+ph,'interval plus truncation bound',out,'P7 order 30.')
        if not A.contains(0):
            eps=abs(A)/524288
            out=threshold(1-eps,d,p,20)
            out['epsilon']=eps
            assert out['Ds']*A < 0 and abs(out['Ds']+A) < abs(A)/2
            record('P5 explicit derivative test phi='+ph,'interval plus truncation bound',out,'P7 order 20.')


def stationary_j(b, delta, phi, order, rigorous=True):
    if rigorous:
        g, _, _ = series_with_tail(b, delta, phi, order, 2)
    else:
        g, _ = invariant_series(b, delta, phi, order, 2)
    s=(1-b)*(-g[0]).exp()
    sb=-(-g[0]).exp()*(1+(1-b)*g[1])
    C=b*(2-s)*(1-b)
    Cp=(1-2*b)*(2-s)-b*(1-b)*sb
    J=b-1+s+C*g[1]
    Jb=1+sb+Cp*g[1]+C*g[2]
    return J,Jb,s


def stationary_certificates():
    phi=arb('-.35')
    st=star(phi)
    roots=[]
    for dt,printed in [('.004','0.599487404570512478'),('.002','0.599495940314231187')]:
        d=arb(dt)
        b0=(1-arb(printed)).mid()
        for _ in range(5):
            j,jb,_=stationary_j(b0,d,phi,16,False)
            b0=(b0-j/jb).mid()
        j,_,_=stationary_j(b0,d,phi,24)
        _,jb,_=stationary_j(ball(b0,arb('1e-12')),d,phi,24)
        assert jb > 0 or jb < 0
        m=jb.abs_lower()
        h=(2*j.abs_upper()/m+arb('1e-70')).upper()
        assert h < arb('1e-12') and j.abs_upper() < m*h
        _,_,s=stationary_j(ball(b0,h),d,phi,24)
        roots.append(s)
        old=threshold(arb(printed),d,phi,24)
        assert not old['Ds'].contains(0)
        record('Actual stationary point delta='+dt,'interval plus truncation bound',
               {'s_stationary':s,'shift':s-st,'printed_s':arb(printed),
                'Ds_at_printed_s':old['Ds'],'J_slope_on_bracket':jb,
                'b_root_halfwidth':h},
               'P7 order 24, Cauchy first/second derivative tails, sign bracket. '
               'The published decimal is a finite-difference zero, not this derivative zero.')
    ratio=(roots[0]-st)/(roots[1]-st)
    assert ratio > arb('3.99') and ratio < arb('4.01')
    record('Stationary-point displacement ratio','interval plus truncation bound',{'ratio':ratio})


def cheb_values(x, n):
    out=[1, x]
    for k in range(2,n):
        out.append(2*x*out[-1]-out[-2])
    return out[:n]


def cheb_eval(coefficients, x):
    # Clenshaw evaluation; works for Arb balls and Arb Taylor jets.
    b1=0*x
    b2=0*x
    for c in coefficients[:0:-1]:
        b0=2*x*b1-b2+c
        b2,b1=b1,b0
    return coefficients[0]+x*b1-b2


def operator_corrector(s,c,delta,phi,n):
    """Find an arbitrary bounded polynomial test function, not an invariant law.

    Collocation chooses coefficients only. Its error is subsequently bounded
    on the ENTIRE invariant interval using Taylor's theorem with interval
    derivatives. No collocation quadrature accuracy is assumed.
    """
    width=2*c*delta/s
    lo=2*(s*width+c*(1-delta))/(((s+width)**2+4*c*(1-delta)).sqrt()+width-s)
    hi=lo+width
    # Slightly enlarge to exact dyadic endpoints and check invariance below.
    vl=lo.log().lower()-arb('1e-70')
    vh=hi.log().upper()+arb('1e-70')
    vm=((vl+vh)/2).mid()
    vr=((vh-vl)/2).upper()
    assert (s+c*(1-delta)*(-(vm+vr)).exp()).log() >= vm-vr
    assert (s+c*(1+delta)*(-(vm-vr)).exp()).log() <= vm+vr
    mat=[]
    rhs=[]
    for i in range(2):
        for k in range(n):
            x=(arb(k)/(n-1)).cos_pi()
            v=vm+vr*x
            now=cheb_values(x,n)
            row=[arb(0)]*(2*n+1)
            row[-1]=arb(1)
            for l in range(n): row[i*n+l]+=now[l]
            for j,z in enumerate([-1,1]):
                p=(1+(phi if i==j else -phi))/2
                xp=((s+c*(1+z*delta)*(-v).exp()).log()-vm)/vr
                future=cheb_values(xp,n)
                for l in range(n): row[j*n+l]-=p*future[l]
            mat.append(row)
            rhs.append([v])
    # Gauge: the constant coefficient of the minus-host test function is 0.
    mat.append([arb(int(k==0)) for k in range(2*n+1)])
    rhs.append([arb(0)])
    sol=arb_mat(mat).solve(arb_mat(rhs))
    # Midpoints are exact dyadic polynomial coefficients. Their accuracy is
    # immaterial to the proof; the residual test below certifies their quality.
    coeff=[[sol[i*n+k,0].mid() for k in range(n)] for i in range(2)]
    return coeff,vm,vr


def operator_residual(x, coeff, vm, vr, s,c,delta,phi):
    v=vm+vr*x
    future=[]
    for j,z in enumerate([-1,1]):
        xp=((s+c*(1+z*delta)*(-v).exp()).log()-vm)/vr
        future.append(cheb_eval(coeff[j],xp))
    return [v+(1+phi)/2*future[i]+(1-phi)/2*future[1-i]-cheb_eval(coeff[i],x)
            for i in range(2)]


def operator_rate(s,c,delta,phi,n=40,cells=64,degree=20):
    coeff,vm,vr=operator_corrector(s,c,delta,phi,n)
    h=arb(1)/cells
    extrema=None
    oldcap=ctx.cap
    ctx.cap=degree+1
    for k in range(cells):
        mid=-1+(2*k+1)*h
        center=arb_series([mid,1],prec=degree+1)
        domain=arb_series([ball(mid,h),1],prec=degree+1)
        fc=operator_residual(center,coeff,vm,vr,s,c,delta,phi)
        fd=operator_residual(domain,coeff,vm,vr,s,c,delta,phi)
        for a,z in zip(fc,fd):
            remainder=sum((a[j].abs_upper()*h**j for j in range(1,degree)),arb(0))
            remainder+=z[degree].abs_upper()*h**degree
            enclosure=ball(a[0],remainder)
            extrema=enclosure if extrema is None else extrema.union(enclosure)
    ctx.cap=oldcap
    assert extrema.is_finite()
    return extrema


def operator_certificate(label,s,delta,phi,guess,n=40,cells=64,degree=20):
    s,d,p,D0=map(arb,[s,delta,phi,guess])
    c=(1-s)*D0.exp()
    rate=operator_rate(s,c,d,p,n,cells,degree)
    # P3 lower slope holds on log-c interval D0 +- .01.
    cmin=(1-s)*(D0-arb('.01')).exp()
    beta=cmin*(1-d)/(s*s+cmin*(1-d))
    error=(3*rate.abs_upper()/beta).upper()
    assert error < arb('.01')
    D=ball(D0,error)
    record(label,'interval plus truncation bound',
           {'D':D,'rate':rate,'log_c_root_error':error,'beta':beta},
           f'Exact-operator coboundary; polynomial degree {n-1}, {cells} cells, '
           f'Taylor remainder order {degree}. Includes all discretization error.')
    return D


def operator_certificates():
    # Registered independent exact periodic benchmark, tolerance 1e-30.
    s,d=arb('.7'),arb('.6')
    c=(1-s*s)/(1+(d*d+(1-d*d)*s*s).sqrt())
    r=operator_rate(s,c,d,arb(-1),40,64,20)
    assert r.contains(0) and r.abs_upper() < arb('1e-30')
    record('OPERATOR BASELINE','interval plus truncation bound',{'rate_at_exact_alternating_threshold':r},
           'Registered tolerance 1e-30; independent closed-form benchmark passed.')
    operator_certificate('Moderate s=.5 delta=.4 phi=.7','.5','.4','.7','.0582995785286115578')
    operator_certificate('Moderate s=.7 delta=.6 phi=0','.7','.6','0','.0456067551059600241')
    other=operator_certificate('CE2 independent operator','.9','.4','-.75','-.004482971848658596862157')
    series=threshold(arb('.9'),arb('.4'),arb('-.75'),88)
    assert other.overlaps(series['D'])
    small=[]
    for args in [
        ('Small s=.1','.1','.4','.6','.0736141981986050458',64,128,24),
        ('Small s=.03','.03','.4','.6','.07646264616186288',96,128,24),
        ('Small s=.01','.01','.4','.6','.0781564790055211',128,128,28),
        ('Small s=.003','.003','.4','.6','.07985126015331216',160,128,28),
    ]:
        small.append(operator_certificate(*args))
    ell=-(1-arb('.4')**2).log()/2
    gaps=[small[k+1]-small[k] for k in range(3)]
    assert all(x > 0 for x in gaps) and small[-1] < ell
    record('Small-s sampled ordering','interval plus truncation bound',
           {'ell':ell,**{f'increment_{k+1}':v for k,v in enumerate(gaps)}},
           'All four sampled D values increase as s decreases, below ell. Endpoint limit is analytic.')


def small_noise_certificates():
    d=arb('.002')
    for ph in ['-.35','0','.8']:
        phi=arb(ph)
        points=[star(phi)-arb('.04'),star(phi),star(phi)+arb('.04')] if ph=='-.35' else [arb('.35'),arb('.75')]
        for j,s in enumerate(points):
            out=threshold(s,d,phi,32)
            out.update(s=s,D_over_delta2=out['D']/d**2,
                       true_Ds_over_delta2=out['Ds']/d**2,
                       F=F(s,phi),Fs=Fs(s,phi))
            h=arb('.00002')
            dp=threshold(s+h,d,phi,32,derivative=False)['D']
            dm=threshold(s-h,d,phi,32,derivative=False)['D']
            out['original_secant_over_delta2']=(dp-dm)/(2*h*d**2)
            assert abs(out['D']/d**2-F(s,phi)) < arb('0.00008')
            assert abs(out['Ds']/d**2-Fs(s,phi)) < arb('.0003')
            record(f'Small noise phi={ph} point={j+1}','interval plus truncation bound',out,
                   'P7 order 32; true derivative, not the original finite secant.')
    out=threshold(arb('.92'),arb('.5'),arb('.7'),88)
    record('P7 validation point s=.92 delta=.5 phi=.7','interval plus truncation bound',out,
           'True threshold with order 88 tail; original D16 and D28 were truncated approximations.')


def hermite(n,x):
    h0,h1=arb(1),x
    if n==0:return h0
    for k in range(1,n):h0,h1=h1,x*h1-k*h0
    return h1


def normal_quadrature(n):
    # NumPy supplies guesses only. Sign brackets and polynomial degree certify
    # all n distinct real roots; no floating-point root is trusted.
    import numpy as np
    seeds=np.polynomial.hermite_e.hermegauss(n)[0]
    xs=[]
    for seed in seeds:
        x=arb(str(seed))
        for _ in range(8):
            x=(x-hermite(n,x)/(n*hermite(n-1,x))).mid()
        h=arb('1e-80')
        a,b=x-h,x+h
        fa,fb=hermite(n,a),hermite(n,b)
        assert fa*fb < 0
        xx=ball(x,h)
        assert not hermite(n-1,xx).contains(0)
        if xs:assert xs[-1] < xx
        xs.append(xx)
    weights=[arb(math.factorial(n-1))/(n*hermite(n-1,x)**2) for x in xs]
    assert sum(weights).contains(1)
    assert sum((w*x for w,x in zip(weights,xs)),arb(0)).contains(0)
    assert sum((w*x*x for w,x in zip(weights,xs)),arb(0)).contains(1)
    return xs,weights


def gaussian_finite_chain(phi,n,order):
    xs,ws=normal_quadrature(n)
    kernel=[[ws[j]/(1-phi*phi).sqrt()*
             ((2*phi*xs[i]*xs[j]-phi*phi*(xs[i]**2+xs[j]**2))/(2*(1-phi*phi))).exp()
             for j in range(n)] for i in range(n)]
    rows=[sum(row) for row in kernel]
    pi=[ws[j]*rows[j] for j in range(n)]
    norm=sum(pi)
    pi=[p/norm for p in pi]
    P=arb_mat([[kernel[i][j]/rows[i] for j in range(n)] for i in range(n)])
    exponentials=[arb_series([0,x],prec=order+1).exp() for x in xs]
    normalization=sum((pi[j]*exponentials[j] for j in range(n)),arb_series([],prec=order+1))
    deviations=[e/normalization-1 for e in exponentials]
    powers=[[dev**r for dev in deviations] for r in range(order+1)]
    return xs,P,pi,powers


def finite_gaussian_rate(s,c,sigma,chain,order):
    xs,P,pi,powers=chain
    n=len(xs)
    lam=(s+(s*s+4*c).sqrt())/2
    b=c/(lam*lam)
    PT=P.transpose()
    eye=arb_mat([[int(i==j) for j in range(n)] for i in range(n)])
    inv={k:(eye-PT*((-b)**k)).solve(eye) for k in range(1,order+1)}
    mom={}
    pmom={(0,0):list(PT*arb_mat([[p] for p in pi]))}
    coeff=[]
    for total in range(1,order+1):
        for k in range(total,0,-1):
            rhs=[arb(0)]*n
            for r in range(k+1):
                for m in range(total-k+1):
                    if r==m==0:continue
                    ell=k-r+m
                    factor=b**k*math.comb(k,r)*(-1)**(k-r+m)*math.comb(k+m-1,m)
                    for power in ([0] if r==0 else range(r,total-ell+1)):
                        previous=pmom.get((ell,total-power))
                        if previous is None:continue
                        for j in range(n):rhs[j]+=factor*powers[r][j][power]*previous[j]
            v=inv[k]*arb_mat([[a] for a in rhs])
            mom[k,total]=list(v)
            pmom[k,total]=list(PT*v)
        coeff.append(sum((arb((-1)**(k+1))/k*sum(mom[k,total]) for k in range(1,total+1)),arb(0)))
    raw=lam.log()+sum((cn*sigma**(i+1) for i,cn in enumerate(coeff)),arb(0))
    rd=(1-b)**2/(8*b)
    radius=(1+rd/(2+rd)).log()/max(x.abs_upper() for x in xs)
    q=sigma.abs_upper()/radius
    assert q < 1
    tail=arb(2).log()*q**(order+1)/(1-q)
    return ball(raw,tail),tail,radius,raw,b


def gaussian_certificates():
    phi,sigma=arb('-.35'),arb('1e-5')
    points=[star(phi)-arb('.03'),star(phi),star(phi)+arb('.03')]
    results={}
    order=12
    oldcap=ctx.cap
    ctx.cap=order+1
    for n in [12,20]:
        chain=gaussian_finite_chain(phi,n,order)
        xs,P,pi,_=chain
        mean=sum((pi[i]*xs[i] for i in range(n)),arb(0))
        variance=sum((pi[i]*xs[i]**2 for i in range(n)),arb(0))-mean**2
        lag=sum((pi[i]*xs[i]*sum((P[i,j]*xs[j] for j in range(n)),arb(0)) for i in range(n)),arb(0))/variance
        record(f'Gaussian finite chain n={n} moments','interval',{'variance':variance,'lag1':lag},
               'Hermite nodes individually interval-bracketed; weights and all transitions enclosed.')
        results[n]=[]
        for i,s in enumerate(points):
            # Reproduce the original two deterministic-elasticity updates,
            # enclosing their arithmetic, to certify its residual diagnostic.
            D0=sigma**2*F(s,phi)
            for _ in range(2):
                _,_,_,raw,b=finite_gaussian_rate(s,(1-s)*D0.exp(),sigma,chain,8)
                D0-=raw*(1+b)/b
            _,tail8,_,raw8,_=finite_gaussian_rate(s,(1-s)*D0.exp(),sigma,chain,8)
            assert raw8.abs_upper() < arb('4e-46')
            assert tail8 < arb('1e-25')
            c=(1-s)*D0.exp()
            rate,tail,radius,_,_=finite_gaussian_rate(s,c,sigma,chain,order)
            actual_hosts=[(sigma*x).exp() for x in xs]
            normalizer=sum((pi[j]*actual_hosts[j] for j in range(n)),arb(0))
            minhost=min((h/normalizer).lower() for h in actual_hosts)
            cmin=(1-s)*(D0-arb('.01')).exp()
            beta=cmin*minhost/(s*s+cmin*minhost)
            error=3*rate.abs_upper()/beta
            assert error < arb('.01')
            normalized=ball(D0,error)/sigma**2
            results[n].append(normalized)
            record(f'Gaussian finite chain n={n} point={i+1}','interval plus truncation bound',
                   {'s':s,'D_over_sigma2':normalized,'rate_tail':tail,'sigma_radius':radius,
                    'rate_at_candidate_D':rate,'root_error_D':error,
                    'original_order8_rate_residual':raw8,'original_order8_tail':tail8},
                   'Order 12 finite-state series; DOES NOT enclose the infinite-state Gaussian host.')
    for i in range(3):
        diff=results[20][i]-results[12][i]
        assert abs(diff) < arb('2.4e-15')
        record(f'Gaussian finite-chain refinement point={i+1}','interval plus truncation bound',
               {'normalized_difference':diff},'A difference of certified finite-chain thresholds, not a Gaussian quadrature error bound.')
    left=(results[20][1]-results[20][0])/arb('.03')
    right=(results[20][2]-results[20][1])/arb('.03')
    assert left > 0 and right < 0
    record('Gaussian finite-chain secants','interval plus truncation bound',{'left':left,'right':right})
    record('Continuous Gaussian nonlinear threshold and secants','interval plus truncation bound',
           {'true_Gaussian_enclosure':'NOT ESTABLISHED'},
           'P7 bounds only each finite chain. No certified infinite-state transition/quadrature error has been supplied or derived.',False)
    ctx.cap=oldcap


def sampled_grid_tallies():
    path=(__import__('pathlib').Path(__file__).resolve().parent / '../studies/memory-memB/memB_tables.csv').resolve()
    groups=defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            groups[(row['model'],row['amplitude'],row['phi_input'],row['ring'])].append(row)
    counts=defaultdict(int)
    margins=[]
    for key,rows in groups.items():
        rows.sort(key=lambda r:arb(r['tau']).mid())
        up=[]
        down=[]
        for row in rows[:-1]:
            diff=arb(row['next_D_difference'])
            se=arb(row['next_difference_SE'])
            up.append(diff-4*se > 0)
            down.append(diff+4*se < 0)
            margins.append((abs(diff)-4*se).abs_lower())
        status='decrease' if all(down) else ('nonmonotone' if any(up) and any(down) else 'unresolved')
        assert status==rows[0]['status']
        if key[0]!='gamma_matched':counts[(key[3],status)]+=1
    assert counts['2','decrease']==112 and counts['3','nonmonotone']==15
    assert sum(v for (r,_),v in counts.items() if r=='3')==24
    record('Manuscript sampled-grid tallies','interval',
           {'two_stage_decreasing':arb(112),'three_stage_nonmonotone':arb(15),
            'three_stage_total':arb(24),'minimum_distance_from_4SE_decision':min(margins)},
           'Certified arithmetic on saved decimal differences and paired SEs only. '
           'Does not bound simulation roundoff, sampling error, or true thresholds.')


def main():
    start=time.monotonic()
    print('GAP4 CERTIFICATION; python-flint='+flint.__version__+'; decimal working precision='+str(ctx.dps), flush=True)
    basic_certificates()
    binary_certificates()
    stationary_certificates()
    small_noise_certificates()
    operator_certificates()
    gaussian_certificates()
    sampled_grid_tallies()
    (HERE/'gap4_results.json').write_text(json.dumps(RECORDS,indent=2,ensure_ascii=False)+'\n')
    files=[Path(__file__).resolve(),
           HERE.parent/'studies/memory-memA/memA_proofs.md',
           HERE.parent/'studies/memory-memA/memA_checks.py',
           HERE.parent/'studies/memory-memA/memA_checks_output.txt',
           HERE.parent/'verification/memory-check/memory-check.md',
           HERE.parent/'../paper/sections/06-exact.tex',
           HERE.parent/'../paper/appendices/D-exact.tex',
           HERE.parent/'studies/memory-memB/memB_tables.csv']
    (HERE/'gap4_inputs.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(HERE.parent))+'\n'
        for p in files))
    print('Certificate records:',len(RECORDS),'; certified:',sum(r['certified'] for r in RECORDS),
          '; explicitly open:',sum(not r['certified'] for r in RECORDS),flush=True)
    print('Elapsed seconds:',round(time.monotonic()-start,2),flush=True)


if __name__ == '__main__':
    main()
