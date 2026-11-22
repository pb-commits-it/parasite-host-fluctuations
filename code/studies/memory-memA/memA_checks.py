#!/usr/bin/env python3
"""Independent, deterministic checks for memA_proofs.md.

Run: python3 -I memA_checks.py > memA_checks_output.txt
Only numpy, scipy, mpmath and the standard library are used. No project imports.
The binary-host moment calculation is a convergent expansion of the exact
invariant equation, with an analytic truncation bound. Spectral calculations
are separately labelled numerical approximations, not interval certificates.
"""
import math
import time
import sys
import numpy as np
from scipy.linalg import solve as dsolve
import mpmath as mp

mp.mp.dps = 65
M = mp.mpf
FAILURES = []


def say(*args):
    print(*args, flush=True)


def fmt(x, n=18):
    return mp.nstr(x, n)


def check(name, condition, detail=""):
    say(("PASS " if condition else "FAIL ") + name, detail)
    if not condition:
        FAILURES.append(name)


def coef(s, phi):
    a = 1-s
    return a*(1+(a+2)*phi)/(2*(1+a)*(1+a*phi))


def coef_ds(s, phi):
    a = 1-s
    return -(1+2*phi+2*a*phi-a*a*phi*phi)/(2*(1+a)**2*(1+a*phi)**2)


def boundary(phi):
    if phi >= 2-mp.sqrt(5):
        return M(0)
    if phi <= -M('0.5'):
        return M(1)
    return 1-(1-mp.sqrt(2*(1+phi)))/phi


def exact_series_rate(s, c, delta, phi, order=14):
    """Taylor coefficients of the EXACT stationary ratio law, not of C(s).

    u=lambda*(1+z), z'=b*(delta*Z'-z)/(1+z).
    moments[k][n][j] = [delta^n] E[z^k 1{Z=j}].
    Descending k makes the equations triangular at each total order n.
    """
    lam = (s+mp.sqrt(s*s+4*c))/2
    b = c/(lam*lam)
    P = mp.matrix([[(1+phi)/2, (1-phi)/2],
                   [(1-phi)/2, (1+phi)/2]])
    mom = [[[M(0), M(0)] for _ in range(order+1)] for _ in range(order+1)]
    mom[0][0] = [M('0.5'), M('0.5')]
    signs = (-1, 1)
    coeff = [M(0)]*(order+1)
    for n in range(1, order+1):
        for k in range(n, 0, -1):
            rhs = mp.matrix([0, 0])
            for r in range(k+1):
                if r > n:
                    continue
                for m in range(n-k+1):
                    if r == 0 and m == 0:
                        continue
                    ell = k-r+m
                    common = (b**k * math.comb(k, r) * (-1)**(k-r+m)
                              * math.comb(k+m-1, m))
                    for j in range(2):
                        rhs[j] += common*signs[j]**r*sum(
                            P[i,j]*mom[ell][n-r][i] for i in range(2))
            q = (-b)**k
            # Inverse of I-q*P, diagonalized in the symmetric/antisymmetric basis.
            sym = (rhs[0]+rhs[1])/(2*(1-q))
            anti = (rhs[0]-rhs[1])/(2*(1-q*phi))
            mom[k][n] = [sym+anti, sym-anti]
        coeff[n] = sum((-1)**(k+1)*sum(mom[k][n])/k for k in range(1,n+1))
    value = mp.log(lam)+sum(coeff[n]*delta**n for n in range(1,order+1))
    radius = (1-b)**2/(8*b)
    ratio = abs(delta)/radius
    bound = (mp.log(2)*ratio**(order+1)/(1-ratio)) if ratio < 1 else mp.inf
    return value, bound, coeff, radius


def series_loss(s, delta, phi, order=14):
    f = lambda d: exact_series_rate(s, (1-s)*mp.exp(d), delta, phi, order)[0]
    d0 = delta**2*coef(s,phi)
    d = mp.findroot(f, (d0-M('0.00001'), d0+M('0.00001')), solver='secant', tol=M('1e-52'))
    r, err, _, radius = exact_series_rate(s, (1-s)*mp.exp(d), delta, phi, order)
    # A conservative growth slope on a c interval of relative width exp(+-0.01).
    cmin = (1-s)*mp.exp(d-M('.01'))
    beta = cmin*(1-delta)/(s*s+cmin*(1-delta))
    root_bound = 3*err/beta
    return d, root_bound, radius


def spectral_rate(s, c, delta, phi, n=24, high=True):
    """Chebyshev collocation of the joint invariant operator in log(u).

    This is deterministic quadrature of an exact operator. Refinement is
    reported; it is NOT called a rigorous numerical enclosure.
    """
    if high:
        d = 2*c*delta/s
        lo = 2*(s*d+c*(1-delta))/(mp.sqrt((s+d)**2+4*c*(1-delta))+d-s)
        hi = lo+d
        vl, vh = mp.log(lo), mp.log(hi)
        nodes = [mp.cos(mp.pi*k/(n-1)) for k in range(n)]
        vs = [(vl+vh)/2+(vh-vl)*x/2 for x in nodes]
        weights = [M((-1)**k)*(M('.5') if k in (0,n-1) else 1) for k in range(n)]
        K = mp.matrix(2*n)
        for i in range(2):
            for k,v in enumerate(vs):
                for j,z in enumerate((-1,1)):
                    target = mp.log(s+c*(1+delta*z)*mp.exp(-v))
                    x = (2*target-vl-vh)/(vh-vl)
                    near = next((l for l in range(n) if abs(x-nodes[l]) < M('1e-58')), None)
                    vals = [M(int(l == near)) for l in range(n)] if near is not None else [weights[l]/(x-nodes[l]) for l in range(n)]
                    norm = sum(vals)
                    prob = (1+phi if i==j else 1-phi)/2
                    for l in range(n):
                        K[i*n+k,j*n+l] = prob*vals[l]/norm
        A = K.T-mp.eye(2*n)
        rhs = mp.matrix(2*n,1)
        for j in range(2*n):
            A[2*n-1,j] = 1
        rhs[2*n-1] = 1
        w = mp.lu_solve(A,rhs)
        return sum(w[i*n+k]*vs[k] for i in range(2) for k in range(n))
    s,c,delta,phi = map(float,(s,c,delta,phi))
    d = 2*c*delta/s
    lo = 2*(s*d+c*(1-delta))/(np.sqrt((s+d)**2+4*c*(1-delta))+d-s)
    hi = lo+d
    vl,vh = np.log(lo),np.log(hi)
    nodes = np.cos(np.pi*np.arange(n)/(n-1))
    vs = (vl+vh)/2+(vh-vl)*nodes/2
    weights = (-1.)**np.arange(n)
    weights[[0,-1]] *= .5
    K = np.empty((2*n,2*n))
    for i in range(2):
        for j,z in enumerate((-1,1)):
            target = np.log(s+c*(1+delta*z)*np.exp(-vs))
            x = (2*target-vl-vh)/(vh-vl)
            dif = x[:,None]-nodes[None,:]
            hit = np.abs(dif) < 1e-14
            dif[hit] = 1
            vals = weights[None,:]/dif
            for k in np.where(hit.any(axis=1))[0]:
                vals[k,:] = hit[k,:]
            vals /= vals.sum(axis=1)[:,None]
            K[i*n:(i+1)*n,j*n:(j+1)*n] = ((1+phi if i==j else 1-phi)/2)*vals
    A = K.T-np.eye(2*n)
    A[-1,:] = 1
    rhs = np.zeros(2*n)
    rhs[-1] = 1
    w = dsolve(A,rhs)
    return float(w@np.tile(vs,2))


def spectral_loss(s, delta, phi, n=24, high=True):
    if high:
        f = lambda d: spectral_rate(s,(1-s)*mp.exp(d),delta,phi,n,True)
        d0 = delta**2*coef(s,phi)
        return mp.findroot(f,(d0-M('.01'),d0+M('.01')),tol=M('1e-42'),solver='secant')
    from scipy.optimize import brentq
    ell = -math.log1p(-float(delta)**2)/2
    return brentq(lambda d: spectral_rate(s,(1-float(s))*math.exp(d),delta,phi,n,False),
                  -math.log1p(float(delta))-.01, ell+.01, xtol=2e-13)


def alternating_loss(s,delta):
    return mp.log((1+s)/(1+mp.sqrt(delta*delta+(1-delta*delta)*s*s)))


def gaussian_chain(phi, n):
    """Reversible Nyström approximation to the Gaussian AR(1) kernel.

    The density relative to the N(0,1) invariant measure is symmetric.
    Row normalization has invariant weights proportional to w_i*row_sum_i.
    Quadrature refinement, not a rigorous infinite-state error bound, is used.
    """
    xs, ws = mp.gauss_quadrature(n, 'hermite')
    xs = [mp.sqrt(2)*x for x in xs]
    ws = [w/mp.sqrt(mp.pi) for w in ws]
    K = mp.matrix(n)
    for i in range(n):
        for j in range(n):
            K[i,j] = ws[j]/mp.sqrt(1-phi**2)*mp.exp(
                (2*phi*xs[i]*xs[j]-phi**2*(xs[i]**2+xs[j]**2))/(2*(1-phi**2)))
    rows = [sum(K[i,j] for j in range(n)) for i in range(n)]
    pi = [ws[i]*rows[i] for i in range(n)]
    norm = sum(pi)
    pi = [p/norm for p in pi]
    P = mp.matrix(n)
    for i in range(n):
        for j in range(n):
            P[i,j] = K[i,j]/rows[i]
    return xs,P,pi


def gaussian_series_setup(xs,P,pi,order):
    """Host exp(sigma*x)/E_pi exp(sigma*x), Taylor coefficients at zero."""
    nstate=len(xs)
    norm=[sum(pi[j]*xs[j]**n/mp.factorial(n) for j in range(nstate)) for n in range(order+1)]
    inv=[M(1)]+[M(0)]*order
    for n in range(1,order+1):
        inv[n]=-sum(norm[k]*inv[n-k] for k in range(1,n+1))
    g=[[M(0)]*nstate for _ in range(order+1)]
    for n in range(1,order+1):
        for j in range(nstate):
            g[n][j]=sum(xs[j]**k/mp.factorial(k)*inv[n-k] for k in range(n+1))
    powers=[[[M(0)]*nstate for _ in range(order+1)] for _ in range(order+1)]
    powers[0][0]=[M(1)]*nstate
    for r in range(1,order+1):
        for n in range(r,order+1):
            for j in range(nstate):
                powers[r][n][j]=sum(powers[r-1][n-k][j]*g[k][j] for k in range(1,n+1))
    return powers


def finite_state_lognormal_rate(s,c,sigma,xs,P,pi,powers,order=8):
    """Full nonlinear invariant series; arbitrary finite-state Markov host.

    Generalization of (26): replace (delta*j)^r by the Taylor coefficients
    of (h_j(sigma)-1)^r. No Gaussian second-order coefficient is used here.
    """
    ns=len(xs)
    lam=(s+mp.sqrt(s*s+4*c))/2
    b=c/(lam*lam)
    PT=P.T
    inv={k:(mp.eye(ns)-(-b)**k*PT)**-1 for k in range(1,order+1)}
    mom=[[[M(0)]*ns for _ in range(order+1)] for _ in range(order+1)]
    pmom=[[[M(0)]*ns for _ in range(order+1)] for _ in range(order+1)]
    mom[0][0]=pi
    pmom[0][0]=list(PT*mp.matrix(pi))
    coeff=[M(0)]*(order+1)
    for n in range(1,order+1):
        for k in range(n,0,-1):
            rhs=[M(0)]*ns
            for r in range(k+1):
                for m in range(n-k+1):
                    if r==0 and m==0:
                        continue
                    ell=k-r+m
                    factor=b**k*math.comb(k,r)*(-1)**(k-r+m)*math.comb(k+m-1,m)
                    for d in ([0] if r==0 else range(r,n-ell+1)):
                        for j in range(ns):
                            rhs[j]+=factor*powers[r][d][j]*pmom[ell][n-d][j]
            mom[k][n]=list(inv[k]*mp.matrix(rhs))
            pmom[k][n]=list(PT*mp.matrix(mom[k][n]))
        coeff[n]=sum((-1)**(k+1)*sum(mom[k][n])/k for k in range(1,n+1))
    rate=mp.log(lam)+sum(coeff[n]*sigma**n for n in range(1,order+1))
    # P7 extends to |h-1| <= R_delta. The following sigma disk ensures this.
    rd=(1-b)**2/(8*b)
    radius=mp.log(1+rd/(2+rd))/max(abs(x) for x in xs)
    q=abs(sigma)/radius
    tail=mp.log(2)*q**(order+1)/(1-q) if q<1 else mp.inf
    return rate,tail,b


def gaussian_checks():
    say('\nAdditional direct Gaussian AR(1) check (65 digits)')
    say('Nyström Gaussian transition quadrature: finite-state exact nonlinear invariant series, order 8.')
    say('Series tails are bounded analytically; Gaussian quadrature errors are assessed by 12/20-node refinement.')
    phi=-M('.35'); sigma=M('1e-5'); star=boundary(phi)
    points=[star-M('.03'),star,star+M('.03')]
    results={}
    for ng in [12,20]:
        xs,P,pi=gaussian_chain(phi,ng)
        powers=gaussian_series_setup(xs,P,pi,8)
        mean=sum(pi[i]*xs[i] for i in range(ng))
        var=sum(pi[i]*xs[i]**2 for i in range(ng))-mean**2
        corr=sum(pi[i]*xs[i]*sum(P[i,j]*xs[j] for j in range(ng)) for i in range(ng))/var
        say('  Gaussian nodes',ng,'variance',fmt(var,24),'lag-one correlation',fmt(corr,24))
        results[ng]=[]
        for s in points:
            # Newton updates use deterministic elasticity; its O(sigma^2)
            # discrepancy is removed by iteration. The final residual is printed.
            d=sigma**2*coef(s,phi)
            for _ in range(2):
                r,tail,b=finite_state_lognormal_rate(s,(1-s)*mp.exp(d),sigma,xs,P,pi,powers)
                d-=r*(1+b)/b
            r,tail,b=finite_state_lognormal_rate(s,(1-s)*mp.exp(d),sigma,xs,P,pi,powers)
            results[ng].append(d)
            say('  s',fmt(s,16),'D/sigma^2',fmt(d/sigma**2,24),'C',fmt(coef(s,phi),24),
                'rate residual',fmt(r,5),'series tail',fmt(tail,5))
            check('direct Gaussian quadrature coefficient',abs(d/sigma**2-coef(s,phi))<M('2e-8'))
            check('Gaussian nonlinear series residual and tail',abs(r)<M('1e-35') and tail<M('1e-22'))
    for k,s in enumerate(points):
        discrepancy=abs(results[20][k]-results[12][k])/sigma**2
        check('Gaussian quadrature refinement at s='+fmt(s,8),discrepancy<M('2e-8'),fmt(discrepancy))
    left=(results[20][1]-results[20][0])/(M('.03')*sigma**2)
    right=(results[20][2]-results[20][1])/(M('.03')*sigma**2)
    say('  nonlinear Gaussian secants on either side of s_star:',fmt(left),fmt(right))
    check('Gaussian small-noise increase/decrease across exact boundary',left>0 and right<0)


def main():
    start = time.monotonic()
    say('memA independent checks; mpmath digits =',mp.mp.dps)
    say('No Monte Carlo. Series errors are analytic truncation bounds; rounding is not interval-certified.')
    say('\nP1/P2: arbitrary-memory coefficient, matrix lag sum, and AR boundary')
    maxerr = M(0)
    for s in map(M,['.12','.4','.8']):
        a = 1-s
        A = mp.matrix([[s,a],[1,0]])
        v = mp.matrix([1,1])
        w = mp.matrix([1/(1+a),a/(1+a)])
        Q = mp.eye(2)-v*w.T
        B = mp.matrix([[0,a],[0,0]])
        T = A-v*w.T
        for rho in [lambda l:M('.7')**l,lambda l:(-M('.4'))**l,
                    lambda l:M('.3')*M('.9')**l+M('.7')*(-M('.6'))**l,
                    lambda l:1/mp.sqrt(l+1)]:
            scalar = a/(2*(1+a))*(1+2*sum((-a)**(l-1)*rho(l) for l in range(1,1200)))
            c_el = (w.T*B*v)[0]
            mat = c_el*c_el/2
            power = mp.eye(2)
            for j in range(1200):
                mat -= rho(j+1)*(w.T*B*power*Q*B*v)[0]
                power = power*T
            maxerr = max(maxerr,abs(scalar-mat*(1+a)/a))
    check('general lag formula versus independent matrix formula',maxerr<M('1e-55'),fmt(maxerr))
    maxerr = M(0)
    for phi in map(M,['-.9','-.5','-.35','-.2','0','.6','.95']):
        star = boundary(phi)
        for s in map(M,['.05','.25','.6','.95']):
            err = abs(mp.diff(lambda z:coef(z,phi),s)-coef_ds(s,phi))
            maxerr=max(maxerr,err)
            predicted = (s>star) if phi>-M('.5') else False
            check('boundary sign phi='+fmt(phi,5)+' s='+fmt(s,5), (coef_ds(s,phi)<0)==predicted)
        say('  phi',fmt(phi),'s_star',fmt(star))
    check('analytic derivative',maxerr<M('1e-58'),fmt(maxerr))
    for phi in map(M,['.2','.8','.99']):
        s=M('.37'); tau=1/(1-s); theta=phi/(1-phi)
        R=phi/(1+tau/theta)
        ring=(1+2*theta)/(2*(1+tau))-R/(1-R)
        check('ring reduction phi='+fmt(phi),abs(ring-coef(s,phi))<M('1e-60'))

    say('\nP2/P3/P7: exact Markov invariant expansion at small noise, same covariance phi^lag')
    say('This checks covariance universality; it is not a finite-noise Gaussian Lyapunov calculation.')
    for phi in map(M,['-.35','0','.8']):
        points = [boundary(phi)-M('.04'),boundary(phi),boundary(phi)+M('.04')] if phi<0 else [M('.35'),M('.75')]
        for s in points:
            delta=M('.002')
            d,err,radius=series_loss(s,delta,phi,14)
            predicted=delta**2*coef(s,phi)
            step=M('.00002')
            dp,ep,_=series_loss(s+step,delta,phi,14)
            dm,em,_=series_loss(s-step,delta,phi,14)
            slope=(dp-dm)/(2*step*delta**2)
            say('  phi',fmt(phi,6),'s',fmt(s,14),'D/delta^2',fmt(d/delta**2),
                'C',fmt(coef(s,phi)),'slope/delta^2',fmt(slope),'C_s',fmt(coef_ds(s,phi)),
                'root tail bound',fmt(err,5))
            check('small-noise coefficient',abs(d-predicted)<delta**4*20 and err<M('1e-15'))
            check('small-noise derivative',abs(slope-coef_ds(s,phi))<M('.0003'))
    phi=-M('.35'); star=boundary(phi)
    for delta in map(M,['.004','.002']):
        step=M('.00002')
        deriv=lambda s:(series_loss(s+step,delta,phi,12)[0]-series_loss(s-step,delta,phi,12)[0])/(2*step)
        root=mp.findroot(deriv,(star-M('.001'),star+M('.001')),tol=M('1e-30'))
        say('  finite-amplitude stationary point delta',fmt(delta),'s',fmt(root),'shift',fmt(root-star))
        check('boundary convergence O(delta^2)',abs(root-star)<10*delta**2)

    say('\nP3/P4: invariant-operator refinement and strict bounds at moderate noise')
    for s,delta,phi in [(M('.5'),M('.4'),M('.7')),(M('.7'),M('.6'),M('0')),
                        (M('.9'),M('.4'),M('-.75'))]:
        d1=spectral_loss(s,delta,phi,18)
        d2=spectral_loss(s,delta,phi,30)
        ell=-mp.log(1-delta**2)/2
        say('  s',fmt(s),'delta',fmt(delta),'phi',fmt(phi),'D18',fmt(d1),'D30',fmt(d2),'ell',fmt(ell),'refinement',fmt(abs(d2-d1),6))
        check('spectral refinement',abs(d2-d1)<M('2e-9'))
        check('upper bound',d2<ell)
        if phi>=0: check('positive loss for nonnegative persistence',d2>0)
        if s==M('.9'):
            ds,err,_=series_loss(s,delta,phi,36)
            say('  independent exact-series D',fmt(ds),'analytic tail',fmt(err,6))
            check('spectral versus convergent invariant series',abs(ds-d2)<err+M('1e-25'))

    say('\nP3: two-step projective contraction, direct deterministic paths')
    for s in map(M,['.2','.7']):
        c=M('.8'); delta=M('.6'); H=1+delta
        kappa=(mp.sqrt(s*s+c*H)-s)/(mp.sqrt(s*s+c*H)+s)
        u,v=M('.01'),M('100')
        old=abs(mp.log(u/v))
        worst=M(0)
        for n in range(100):
            for j in range(2):
                h=1+delta*(1 if (n*n+3*j+n)%7<3 else -1)
                u=s+c*h/u; v=s+c*h/v
            new=abs(mp.log(u/v))
            if old>M('1e-45'): worst=max(worst,new/old)
            old=new
        check('uniform two-step contraction s='+fmt(s),worst<=kappa+M('1e-25'),'observed '+fmt(worst)+' bound '+fmt(kappa))

    say('\nP5: all-noise long-stay expansion and explicit derivative threshold')
    for delta,phi in [(M('.6'),M('.8')),(M('.6'),M('-.75')),(M('.6'),-M('.5'))]:
        V=delta**2; A=V*(M('.5')+phi)
        B=-V*(M('.5')+phi+phi**2)+M('1.5')*A*A
        for eps in map(M,['.02','.005','.001']):
            d,err,_=series_loss(1-eps,delta,phi,24)
            scaled=(d-A*eps)/eps**2
            say('  delta',fmt(delta),'phi',fmt(phi),'eps',fmt(eps),'D',fmt(d),'D/eps',fmt(d/eps),'second coefficient',fmt(scaled),'B',fmt(B),'tail',fmt(err,4))
            check('long-stay second coefficient',abs(scaled-B)<20*eps and err<M('1e-20'))
        if A:
            eps=abs(A)/M(524288)
            slope=mp.diff(lambda ss:series_loss(ss,delta,phi,12)[0],1-eps)
            say('  explicit-threshold test eps',fmt(eps),'dD/ds',fmt(slope),'-A',fmt(-A))
            check('derivative in proved interval',slope*A<0 and abs(slope+A)<abs(A)/2)
        else:
            eps=M('1e-6')
            d,_,_=series_loss(1-eps,delta,phi,12)
            check('exception phi=-1/2, quadratic negative loss',abs(d/eps**2+V/4)<M('1e-5'))

    s=1-M('1e-8'); delta=M('.5'); phi=-M('.75')
    value,err,_=series_loss(s,delta,phi,12)
    deriv=mp.diff(lambda ss:series_loss(ss,delta,phi,12)[0],s)
    say('  exact aperiodic counterexample delta=1/2, phi=-3/4, s=1-1e-8:',
        'D',fmt(value,32),'dD/ds',fmt(deriv,32),'analytic tail',fmt(err,6))
    check('exact aperiodic counterexample exceeds proved slope 1/32',deriv>M(1)/32)

    say('\nP6: small-staying endpoint for the aperiodic Markov host')
    say('These collocation values use IEEE doubles with grid refinement, not arbitrary precision certificates.')
    delta=M('.4'); phi=M('.6'); ell=-mp.log(1-delta**2)/2
    previous=-1
    for s in [.1,.03,.01,.003]:
        d1=spectral_loss(s,delta,phi,128,False)
        d2=spectral_loss(s,delta,phi,256,False)
        say('  s',s,'D128',repr(d1),'D256',repr(d2),'ell',fmt(ell),'gap',fmt(ell-d2),'refinement',abs(d2-d1))
        check('small-s refinement',abs(d2-d1)<2e-7)
        check('approach to ell from below',d2>previous and d2<float(ell))
        previous=d2
    # High precision confirmation of one small-s cell.
    dm=spectral_loss(M('.1'),delta,phi,48,True)
    dd=spectral_loss(.1,delta,phi,256,False)
    say('  65-digit small-s cross-check at s=.1:',fmt(dm),'double difference',fmt(abs(dm-dd)))
    check('small-s high precision cross-check',abs(dm-dd)<M('2e-9'))

    say('\nP8: explicit alternating host, exact period product and endpoint obstruction')
    for delta in map(M,['.2','.6','.9']):
        for s in map(M,['0','.2','.7','.99']):
            d=alternating_loss(s,delta)
            c=(1-s)*mp.exp(d)
            Ap=mp.matrix([[s,c*(1+delta)],[1,0]])
            Am=mp.matrix([[s,c*(1-delta)],[1,0]])
            mat=Ap*Am
            trace=mat[0,0]+mat[1,1]
            root=(trace+mp.sqrt(trace**2-4*mp.det(mat)))/2
            slope=mp.diff(lambda ss:alternating_loss(ss,delta),s)
            say('  delta',fmt(delta),'s',fmt(s),'D',fmt(d),'derivative',fmt(slope),'Perron residual',fmt(root-1,4))
            check('exact period-two threshold and increase',abs(root-1)<M('1e-60') and d<0 and slope>0)
        check('s=0 differs from ell',abs(alternating_loss(0,delta)+mp.log(1+delta))<M('1e-60'))

    say('\nP7: series truncation validation at a fixed moderate-noise point')
    s,delta,phi=M('.92'),M('.5'),M('.7')
    d16,e16,_=series_loss(s,delta,phi,16)
    d28,e28,_=series_loss(s,delta,phi,28)
    say('  D16',fmt(d16,30),'D28',fmt(d28,30),'difference',fmt(abs(d16-d28)),'tail bound16',fmt(e16),'tail bound28',fmt(e28))
    check('convergent-series truncation enclosure consistency',abs(d16-d28)<=e16+e28)
    gaussian_checks()
    say('\nFailures:',len(FAILURES),FAILURES)
    say('Elapsed seconds:',round(time.monotonic()-start,2))
    if FAILURES:
        raise SystemExit(1)


if __name__ == '__main__':
    if '--gaussian-only' in sys.argv:
        start=time.monotonic()
        gaussian_checks()
        say('Additional check failures:',len(FAILURES),FAILURES)
        say('Additional elapsed seconds:',round(time.monotonic()-start,2))
        if FAILURES:
            raise SystemExit(1)
    else:
        main()
