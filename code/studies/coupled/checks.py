#!/usr/bin/env python3
"""Reproducible, offline checks for derivation.md. Run: python3 checks.py.

The characteristic-root tolerance, declared before running, is 3e-5 in
|lambda_poly-lambda_FD|/(1+|lambda_FD|), with optimal root matching.
Random searches test implications away from nonhyperbolic boundaries.
"""
import math
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.linalg import eigvals, solve_continuous_lyapunov
from scipy.optimize import brentq, linear_sum_assignment, minimize_scalar, root
from scipy.integrate import solve_ivp, quad
from scipy.signal import welch

SEED = 20261114
NCASE = 2500
FD_TOL = 3e-5
rng = np.random.default_rng(SEED)
failures = []


def report(name, count, bad, detail=""):
    print(f"{name}: cases={count}, counterexamples={bad}" + (f", {detail}" if detail else ""), flush=True)
    if bad:
        failures.append((name, bad))


def lr(d, q=None, beta=1.0, kind="L"):
    q = d if q is None else q
    return dict(kind=kind, d=d, q=q, beta=beta,
                den=np.array([d, 1.]), num=np.array([q*beta]))


def dyn(d, q, h, rho, kind="V"):
    return dict(kind=kind, d=d, q=q, h=h, rho=rho,
                den=np.array([d*(rho+h), rho+d, 1.]),
                num=np.array([q*rho, q]))


def characteristic(blocks):
    den = np.array([1.]); num = np.array([1.])
    for b in blocks:
        den = poly.polymul(den, b['den'])
        num = poly.polymul(num, b['num'])
    return poly.polysub(den, num)


def matrix(blocks):
    sizes = [2 if b['kind'] in ('V', 'M') else 1 for b in blocks]
    starts = np.cumsum([0]+sizes[:-1]); total = sum(sizes)
    mat = np.zeros((total, total))
    for k, b in enumerate(blocks):
        j = starts[k]
        prev = (k-1) % len(blocks)
        out = starts[prev]+sizes[prev]-1
        outfactor = blocks[prev].get('beta', 1.)
        if sizes[k] == 2:
            mat[j:j+2, j:j+2] = [[-b['rho'], -b['h']], [b['d'], -b['d']]]
            mat[j+1, out] += b['q']*outfactor
        else:
            mat[j, j] = -b['d']
            mat[j, out] += b['q']*outfactor
    return mat


def abscissa(blocks):
    return np.max(eigvals(matrix(blocks)).real)


def constants(b):
    d,q,h,rho = (b[k] for k in ('d','q','h','rho'))
    A = rho+d; B = d*(rho+h); J = A-q; C = B-q*rho
    U = rho*rho+d*d-2*d*h-q*q
    V = B*B-q*q*rho*rho
    return A,B,J,C,U,V


def contractive(b, strict=True):
    A,B,J,C,U,V = constants(b)
    if A <= 0 or B <= 0: return False
    if strict: return V > 0 and U+2*np.sqrt(V) > 0
    return V >= 0 and U+2*np.sqrt(max(V, 0)) >= 0


def random_dynamic(kind, rho_range=(-2.5, 3.)):
    h = 10**rng.uniform(-.5,.5)
    if kind == 'V':
        d = 10**rng.uniform(-.7,.8); q = d*rng.uniform(.08,.95)
    else:
        eps = rng.uniform(1.,4.); xi=10**rng.uniform(-.7,.8)
        d=xi+2*eps*h; q=xi+eps*h
    return dyn(d,q,h,h*rng.uniform(*rho_range),kind)


def exact_peak(b):
    """Independent direct frequency maximization, on log-frequency intervals."""
    A,B,*_ = constants(b)
    if A <= 0 or B <= 0: return np.inf
    scale=max(b['d'], abs(b['rho']), b['h'], .01)
    def val(logw):
        s=1j*scale*np.exp(logw)
        return abs(poly.polyval(s,b['num'])/poly.polyval(s,b['den']))**2
    edges=np.linspace(-18,12,31)
    values=[abs(b['num'][0]/b['den'][0])**2]
    for lo,hi in zip(edges[:-1],edges[1:]):
        fit=minimize_scalar(lambda t:-val(t),bounds=(lo,hi),method='bounded')
        values.append(-fit.fun)
    return np.sqrt(max(values))


def finite_difference_checks():
    worst=0.; residual=0.; counts={k:0 for k in 'LRSVM'}; bad=0
    for case in range(360):
        n=case%6+1
        kinds=list(rng.choice(list('LRSVM'),n))
        if n>=5:
            kinds[:5]=rng.permutation(list('LRSVM'))
        inputs=np.exp(rng.uniform(-.7,.7,n))
        blocks=[]; records=[]; eq=[]
        for k,kind in enumerate(kinds):
            counts[kind]+=1
            u=inputs[k]; v=inputs[(k+1)%n]
            a=np.exp(rng.uniform(-.7,.7))
            if kind in ('L','R'):
                w=u/a; kap=rng.uniform(0,3.2) if kind=='R' else 0.
                c=v/(w*np.exp(-kap))
                records.append((kind,len(eq),dict(a=a,w=w,kap=kap,c=c)))
                eq.append(w); blocks.append(lr(a,beta=1-kap,kind=kind))
            elif kind in ('S','V'):
                N=np.exp(rng.uniform(-.3,.3)); p=u/(a+u); I=N*p; c=v/I
                d=a+u
                if kind=='S':
                    records.append((kind,len(eq),dict(a=a,N=N,c=c)))
                    eq.append(I); blocks.append(lr(d,a,kind='S'))
                else:
                    h=np.exp(rng.uniform(-.6,.6)); rho=h*rng.uniform(-1.7,2.5)
                    records.append((kind,len(eq),dict(a=a,N=N,h=h,rho=rho,alpha=h/p,c=c)))
                    eq.extend([N,I]); blocks.append(dyn(d,a,h,rho,'V'))
            else:
                H=np.exp(rng.uniform(-.3,.3)); h=np.exp(rng.uniform(-.6,.4))
                eps=rng.uniform(1,2.5); xi=np.exp(rng.uniform(-.7,.7))
                q=xi+eps*h; d=xi+2*eps*h; m=u/q; P=m*H
                rho=h*rng.uniform(-1.7,2.5)
                records.append((kind,len(eq),dict(H=H,h=h,rho=rho,alpha=h/m,xi=xi,eps=eps,c=v/P)))
                eq.extend([H,P]); blocks.append(dyn(d,q,h,rho,'M'))
        eq=np.array(eq)
        def nonlinear(y):
            outs=[]
            for kind,j,r in records:
                if kind=='R': out=r['c']*np.exp(-r['kap']*y[j]/r['w'])*y[j]
                elif kind in ('L','S'): out=r['c']*y[j]
                else: out=r['c']*y[j+1]
                outs.append(out)
            ans=np.zeros_like(y)
            for k,(kind,j,r) in enumerate(records):
                u=outs[(k-1)%n]
                if kind in ('L','R'): ans[j]=u-r['a']*y[j]
                elif kind=='S': ans[j]=u*(r['N']-y[j])-r['a']*y[j]
                elif kind=='V':
                    N,I=y[j:j+2]
                    phi=N*(r['h']-(r['rho']+r['h'])*(N/r['N']-1))
                    ans[j]=phi-r['alpha']*I
                    ans[j+1]=u*(N-I)-r['a']*I
                else:
                    H,P=y[j:j+2]
                    phi=H*(r['h']-(r['rho']+r['h'])*(H/r['H']-1))
                    ans[j]=phi-r['alpha']*P
                    ans[j+1]=u*H-r['xi']*P-r['alpha']*r['eps']*P*P/H
            return ans
        residual=max(residual,np.max(abs(nonlinear(eq))))
        jac=np.empty((len(eq),len(eq)))
        for j in range(len(eq)):
            step=1e-5*max(1.,abs(eq[j])); dy=np.zeros_like(eq); dy[j]=step
            jac[:,j]=(nonlinear(eq+dy)-nonlinear(eq-dy))/(2*step)
        ef=eigvals(jac); ep=poly.polyroots(characteristic(blocks))
        cost=abs(ef[:,None]-ep[None,:])/(1+abs(ef[:,None]))
        ii,jj=linear_sum_assignment(cost); err=np.max(cost[ii,jj])
        worst=max(worst,err); bad+=err>FD_TOL
    report('Q1 polynomial versus nonlinear finite-difference Jacobian',360,bad,
           f'predeclared tolerance={FD_TOL:g}, worst={worst:.3g}, equilibrium residual={residual:.3g}, type counts={counts}')


def stability_checks():
    for family in ('RL','SL','mixed_contractive','nonnegative_rho'):
        bad=0; worst=-np.inf
        for _ in range(NCASE):
            n=int(rng.integers(1,8)); bs=[]
            for k in range(n):
                d=10**rng.uniform(-1,1)
                if family=='RL': bs.append(lr(d,beta=rng.uniform(-.98,.98),kind='R'))
                elif family=='SL': bs.append(lr(d,d*rng.uniform(.05,.98),kind='S'))
                else:
                    kind=rng.choice(['V','M','S','R'])
                    if kind=='S': bs.append(lr(d,d*rng.uniform(.05,.98),kind='S'))
                    elif kind=='R': bs.append(lr(d,beta=rng.uniform(-.98,.98),kind='R'))
                    else:
                        while True:
                            b=random_dynamic(kind, (0.,3.) if family=='nonnegative_rho' else (-.95,3.))
                            if contractive(b): break
                        bs.append(b)
                if rng.random()<.5: bs.append(lr(10**rng.uniform(-1,1)))
            mx=abscissa(bs); worst=max(worst,mx); bad+=mx>=1e-9
        report('Q2 arbitrary cycles '+family,NCASE,bad,f'largest real eigenvalue={worst:.4g}')
    for kind in ('V','M'):
        bad=0; count=0; maxamp=0
        while count<NCASE:
            b=random_dynamic(kind)
            A,B,*_=constants(b)
            if A<=0 or B<=0: continue
            peak=exact_peak(b)
            if abs(peak-1)<1e-7: continue
            bad+=(peak<1)!=contractive(b); count+=1; maxamp=max(maxamp,peak)
        report('Q2 exact frequency contraction '+kind,count,bad,f'largest sampled peak={maxamp:.4g}')


def cubic_checks():
    for kind in ('V','M'):
        bad=0; stable=0
        for _ in range(NCASE):
            b=random_dynamic(kind); a=10**rng.uniform(-2,2)
            A,B,J,C,*_=constants(b)
            c2=a+A; c1=B+a*J; c0=a*C
            prediction=c2>0 and c1>0 and c0>0 and c2*c1>c0
            obs=abscissa([b,lr(a)])<0
            bad+=prediction!=obs; stable+=obs
        report('Q3 full one-L Routh criterion '+kind,NCASE,bad,f'stable cases={stable}')
    for kind in ('V','M'):
        bad=0; count=0
        while count<NCASE:
            b=random_dynamic(kind,(-.99,-.001)); A,B,J,C,*_=constants(b)
            if A<=0 or B<=0: continue
            a=10**rng.uniform(-3,3)
            F=J*a*a+(A*J+b['q']*b['rho'])*a+A*B
            bad+=(F>0)!=(abscissa([b,lr(a)])<0); count+=1
        report('Q3 reduced F(a) test, internally stable '+kind,count,bad)
    bad=0; count=0
    while count<NCASE:
        b=random_dynamic('V',(-.99,-.01)); A,B,*_=constants(b)
        if not (A<=0 and B>0): continue
        a=10**rng.uniform(-3,3)
        bad+=abscissa([b,lr(a)])<=0; count+=1
    report('Q3 A<=0,B>0 cannot be stabilized by ONE L',count,bad)
    for kind in ('V','M'):
        bad=0; stable=0
        for _ in range(NCASE):
            b=random_dynamic(kind); b=dyn(b['d'],b['q'],b['h'],-b['h'],kind)
            a=10**rng.uniform(-3,3); A,B,J,C,*_=constants(b)
            pred=J>0 and (a+A)*J>b['q']*b['h']
            obs=abscissa([b,lr(a)])<0
            bad+=pred!=obs; stable+=obs
        report('Q3 unregulated host exact one-L criterion '+kind,NCASE,bad,f'stable cases={stable}')


def delay_converse_checks():
    """Construct unstable finite L cascades for amplifying stable host blocks.

    At a selected frequency, solve the pure-delay characteristic for a RHP
    root, then continue to a finite Erlang (equal-L) cascade. Verification is
    by the exact finite-cascade rational characteristic, not a delay surrogate.
    """
    bad=0; count=0; largest_m=0; worst_residual=0; skipped=0
    while count<NCASE:
        b=random_dynamic(rng.choice(['V','M']),(-.99,-.03))
        A,B,*_=constants(b)
        if A<=0 or B<=0: continue
        scale=max(b['d'],b['h'])
        ws=scale*np.logspace(-3,2,120)
        ts=poly.polyval(1j*ws,b['num'])/poly.polyval(1j*ws,b['den'])
        index=int(np.argmax(abs(ts))); w=ws[index]
        if abs(ts[index])<1.08: continue
        phase=np.angle(ts[index]) % (2*np.pi)
        if phase<.1: phase+=2*np.pi
        tau=phase/w
        def T(s): return poly.polyval(s,b['num'])/poly.polyval(s,b['den'])
        def delay_eq(x):
            s=complex(*x); val=T(s)*np.exp(-s*tau)-1
            return [val.real,val.imag]
        sol=root(delay_eq,[min(np.log(abs(ts[index]))/tau,.1*scale),w])
        if not sol.success or sol.x[0]<=1e-9*scale:
            # A failed numerical construction is recorded separately; no assertion
            # is made about these parameters unless a finite cascade is verified.
            skipped+=1; continue
        found=False
        for m in (16,64,256,1024,4096):
            a=m/tau
            def finite_eq(x):
                s=complex(*x); val=T(s)*np.exp(-m*np.log1p(s/a))-1
                return [val.real,val.imag]
            ss=root(finite_eq,sol.x)
            res=np.linalg.norm(finite_eq(ss.x))
            if ss.success and ss.x[0]>1e-9*scale and res<1e-7:
                found=True; largest_m=max(largest_m,m); worst_residual=max(worst_residual,res); break
        count+=1; bad+=not found
    report('Q3 amplification admits a destabilizing finite L cascade',count,bad,
           f'maximum stages used={largest_m}, characteristic residual={worst_residual:.3g}, failed pure-delay initializations retried={skipped}')


def slow_checks():
    bad=0; worst=0.; pos=0
    for _ in range(NCASE):
        bs=[]
        for k in range(int(rng.integers(1,5))):
            if rng.random()<.6:
                while True:
                    b=random_dynamic(rng.choice(['V','M']),(-.97,2.))
                    A,B,*_=constants(b)
                    if A>0 and B>0: break
                bs.append(b)
            else: bs.append(lr(10**rng.uniform(-.5,.5),beta=rng.uniform(-2,1),kind='R'))
        gain=np.prod([b['num'][0]/b['den'][0] for b in bs])
        poles=np.concatenate([poly.polyroots(b['den']) for b in bs])
        a=1e-9*min(abs(poles))/(1+abs(gain))**2
        vals=eigvals(matrix(bs+[lr(a)])); slow=vals[np.argmin(abs(vals))]
        rel=abs(slow/a-(gain-1))/(1+abs(gain-1)); worst=max(worst,rel)
        if abs(gain-1)>1e-5: bad+=(max(vals.real)<0)!=(gain<1)
        pos+=gain>1
    report('Q4 stable remaining blocks: slow root and signed DC test',NCASE,bad,
           f'max normalized slow-root error={worst:.3g}, Q(0)>1 cases={pos}')
    for kind in ('V','M'):
        bad=0; count=0
        while count<NCASE:
            b=random_dynamic(kind,(-2.5,-1.01)); A,B,J,C,*_=constants(b)
            if not (J>0 and C>0): continue
            a=1e-7*min(b['d'],b['h'])
            bad+=not (abscissa([b])<0 and abscissa([b,lr(a)])>0); count+=1
        report('Q4 rho<-h: stable instantaneous loop destabilized by long L '+kind,count,bad)
    for kind in ('V','M'):
        bad=0; err=0.; stable=0; count=0
        while count<NCASE:
            b=random_dynamic(kind); d,q,h=b['d'],b['q'],b['h']; A=d-h
            if A<.15*d: continue
            b=dyn(d,q,h,-h,kind)
            others=[lr(10**rng.uniform(-.3,.7)) for _ in range(int(rng.integers(0,3)))]
            # Include S blocks in some paths to test g<1 as well.
            if rng.random()<.5: others.append(lr(2.,rng.uniform(.3,1.8),kind='S'))
            g=np.prod([x['q']/x['d'] for x in others])
            tau=sum(1/x['d'] for x in others)
            K=g*q*h/A; C=g*q*d/A**2+K*tau
            if abs(C-1)<.015: continue
            a=1e-8*min([A]+[x['d'] for x in others])/(1+C)**2
            vals=eigvals(matrix([b]+others+[lr(a)]))
            pair=vals[np.argsort(abs(vals))[:2]]
            target=(C-1)/2
            err=max(err,abs(pair.real.mean()/a-target)/(1+abs(target)))
            bad+=(max(vals.real)<0)!=(C<1); stable+=C<1; count+=1
        report('Q4 no self-regulation: sqrt(a) pair and C<1 test '+kind,count,bad,
               f'max normalized damping error={err:.3g}, stable limit cases={stable}')
    bad=0; count=0; err=0
    while count<NCASE:
        b=random_dynamic(rng.choice(['V','M'])); d,q,h=b['d'],b['q'],b['h']; A=d-h
        if A<.2*d: continue
        b=dyn(d,q,h,-h,b['kind']); a=1e-9*min(A,h)
        vals=eigvals(matrix([b,lr(a),lr(a)]))
        expected=.5*(q*h/A)**(1/3)*a**(2/3)
        err=max(err,abs(max(vals.real)/expected-1)); bad+=max(vals.real)<=0; count+=1
    report('Q3/4 two sufficiently long L stages destabilize unregulated host',count,bad,
           f'max leading-growth relative error={err:.3g}')


def threshold(rates):
    if len(rates)<3: return np.inf, np.inf
    f=lambda w: np.arctan(w/rates).sum()-np.pi
    hi=max(rates)
    while f(hi)<0: hi*=2
    w=brentq(f,0.,hi,xtol=1e-13)
    return np.exp(.5*np.log1p((w/rates)**2).sum()),w


def ring_matrix(rates,G):
    n=len(rates); mat=-np.diag(rates)
    for k in range(n): mat[k,(k-1)%n]+=rates[k]
    mat[0,(n-1)%n]+=rates[0]*(G-1)
    return mat


def r_checks():
    bad=0
    for _ in range(NCASE):
        n=int(rng.integers(3,10)); rates=10**rng.uniform(-.8,.8,n); crit,w=threshold(rates)
        for factor in (.8,1.2):
            mx=max(eigvals(ring_matrix(rates,-factor*crit)).real)
            bad+=(mx<0)!=(factor<1)
    report('Q5 exact negative-gain stability threshold',NCASE,bad,'two sides per case')
    bad=0; mingap=np.inf
    for _ in range(NCASE):
        n=int(rng.integers(3,21)); rates=10**rng.uniform(-1,1,n); crit,w=threshold(rates)
        equal=(1/np.cos(np.pi/n))**n
        gap=crit/equal-1; mingap=min(mingap,gap); bad+=gap< -1e-10
    report('Q5 equal rates minimize critical gain',NCASE,bad,f'smallest relative gap={mingap:.3g}')
    bad=0; worst=0
    for _ in range(NCASE):
        n=int(rng.integers(3,9)); other=10**rng.uniform(-.5,.5,n-1)
        om=brentq(lambda w:np.arctan(w/other).sum()-np.pi/2,0,100*max(other))
        coeff=om*np.prod(np.sqrt(1+(om/other)**2)); a=1e-7*min(other)
        crit,w=threshold(np.r_[a,other]); err=abs(a*crit/coeff-1)
        worst=max(worst,err); bad+=err>1e-4
    report('Q5 one slow rate: critical gain proportional to 1/a',NCASE,bad,f'max relative error={worst:.3g}')
    bad=0; worst=0
    for _ in range(NCASE):
        aa=10**rng.uniform(-1,1,3); crit,w=threshold(aa)
        closed=aa.sum()*(aa[0]*aa[1]+aa[0]*aa[2]+aa[1]*aa[2])/aa.prod()-1
        err=abs(crit/closed-1); worst=max(worst,err); bad+=err>1e-9
    report('Q5 n=3 closed form',NCASE,bad,f'max relative error={worst:.3g}')
    bad=0
    for _ in range(NCASE):
        n=int(rng.integers(1,3)); aa=10**rng.uniform(-1,1,n); gain=-10**rng.uniform(-2,5)
        bad+=max(eigvals(ring_matrix(aa,gain)).real)>=0
    report('Q5 n=1,2 stable for every finite negative gain',NCASE,bad)


def edge_checks():
    """Additional searches at structural boundaries not hit by continuous draws."""
    bad=0
    for _ in range(NCASE):
        d,q,h=10**rng.uniform(-.5,.5,3); q=d*rng.uniform(.1,.9)
        for rho in (-h,0.,h):
            b=dyn(d,q,0.,rho,rng.choice(['V','M']))
            # q=d is required for the harmless M case.
            if b['kind']=='M': b=dyn(d,d,0.,rho,'M')
            bs=[b,lr(2.,1.,kind='S'),lr(.7)]
            vals=eigvals(matrix(bs))
            bad+=min(abs(vals+rho))>1e-8
            if rho>0: bad+=max(vals.real)>=0
            elif rho<0: bad+=max(vals.real)<=0
            else: bad+=abs(max(vals.real))>1e-8
    report('Q2 canceled host modes at h=0, including rho=0',NCASE,bad,
           'three host-slope signs per case')
    bad=0
    for _ in range(NCASE):
        n=int(rng.integers(1,10)); aa=10**rng.uniform(-1,1,n)
        signs=rng.choice([-1.,1.],n); bs=[lr(a,beta=s,kind='R') for a,s in zip(aa,signs)]
        mx=abscissa(bs)
        bad+=(mx< -1e-9)!=(np.prod(signs)<0)
        if np.prod(signs)>0: bad+=abs(mx)>1e-8
        bad+=abscissa(bs+[lr(2.,1.,kind='S')])>=0
    report('Q2 weak R/L endpoints and strict-stage proviso',NCASE,bad)
    bad=0
    for _ in range(NCASE):
        h=10**rng.uniform(-.5,.5); gamma=rng.uniform(.1,.9)
        d=h*rng.uniform(20,50); q=gamma*d; r=h/(1+gamma)
        b=dyn(d,q,h,-r)
        bad+=abs(abs(b['num'][0]/b['den'][0])-1)>1e-10
        bs=[b]+[lr(10**rng.uniform(-1,1)) for _ in range(int(rng.integers(1,5)))]
        bad+=abscissa(bs)>=0
    report('Q2/3 weak contraction at T(0)=-1 with finite L cascades',NCASE,bad)
    bad=0; count=0; worst=0
    while count<NCASE:
        h=10**rng.uniform(-.5,.5); xi=10**rng.uniform(-.5,.5)
        # M with epsilon=1 has an interior-frequency contraction boundary.
        d=xi+2*h; q=xi+h; gamma=q/d
        def test(r):
            _,_,_,_,U,V=constants(dyn(d,q,h,-r,'M'))
            return U+2*np.sqrt(max(V,0))
        r=brentq(test,0,h/(1+gamma)*(1-1e-12))
        b=dyn(d,q,h,-r,'M'); _,_,_,_,U,V=constants(b)
        w=np.sqrt(-U/2)
        peak=abs(poly.polyval(1j*w,b['num'])/poly.polyval(1j*w,b['den']))
        worst=max(worst,abs(peak-1)); bad+=abs(peak-1)>1e-8
        bs=[b]+[lr(10**rng.uniform(-1,1)) for _ in range(int(rng.integers(1,5)))]
        bad+=abscissa(bs)>=0; count+=1
    report('Q2/3 weak contraction at an interior frequency',count,bad,
           f'max boundary-gain error={worst:.3g}')
    for kind in ('V','M'):
        bad=0
        for _ in range(NCASE):
            b=random_dynamic(kind); A,B,J,C,*_=constants(b)
            bad+=(abscissa([b])<0)!=(J>0 and C>0)
        report('Q3 instantaneous dynamic-host quadratic criterion '+kind,NCASE,bad)
    bad=0; invariant_err=0
    for _ in range(NCASE):
        h,xi=10**rng.uniform(-.5,.5,2); d=xi+2*h; q=xi+h
        b=dyn(d,q,h,-h,'M'); vals=eigvals(matrix([b]))
        bad+=max(abs(vals.real))>1e-8
        bad+=abscissa([b,lr(10**rng.uniform(-2,2))])<=0
        H,m=10**rng.uniform(-.5,.5,2)
        Hdot=H*(h-h*m); mdot=m*(q*H-q)
        deriv=(q-q/H)*Hdot+(h-h/m)*mdot
        invariant_err=max(invariant_err,abs(deriv)/(1+abs(Hdot)+abs(mdot)))
        bad+=abs(deriv)>1e-8
    report('Q3 unregulated M epsilon=1: center invariant and L instability',NCASE,bad,
           f'max scaled invariant derivative={invariant_err:.3g}')
    bad=0
    for _ in range(NCASE):
        h,xi=10**rng.uniform(-.5,.5,2); eps=rng.uniform(2.,6.)
        b=dyn(xi+2*eps*h,xi+eps*h,h,-h,'M')
        bad+=abscissa([b])>=0
        bad+=abscissa([b,lr(10**rng.uniform(-3,3))])>=0
    report('Q3 unregulated M epsilon>=2: every single-L rate',NCASE,bad)
    bad=0; count=0
    while count<NCASE:
        if rng.random()<.4:
            h,xi=10**rng.uniform(-.5,.5,2)
            b=dyn(xi+2*h,xi+h,h,-h,'M')
        else:
            b=random_dynamic('V'); b=dyn(b['d'],b['q'],b['h'],-b['h'],'V')
        A,B,J,C,*_=constants(b)
        if J>1e-12: continue
        bs=[b]+[lr(10**rng.uniform(-1,1)) for _ in range(int(rng.integers(1,8)))]
        bad+=abscissa(bs)<=0; count+=1
    report('Q3 unregulated J<=0: EVERY nonempty L cascade unstable',count,bad)
    bad=0; count=0
    while count<NCASE:
        b=random_dynamic(rng.choice(['V','M']))
        b=dyn(b['d'],b['q'],b['h'],-b['h'],b['kind'])
        A,B,J,C,*_=constants(b)
        scale=max(b['d'],b['h'])
        if J<.05*scale: continue
        bs=[b]+[lr(scale*1e4*rng.uniform(1,3)) for _ in range(int(rng.integers(1,7)))]
        bad+=abscissa(bs)>=0; count+=1
    report('Q3 unregulated J>0: sufficiently fast L cascades stable',count,bad)
    bad=0; count=0; worst=0
    while count<NCASE:
        b=random_dynamic(rng.choice(['V','M']))
        d,q,h=b['d'],b['q'],b['h']; A0=d-h
        if A0<.2*d or q*d<1.05*A0*A0: continue
        delta=1e-7*h; b=dyn(d,q,h,-h+delta,b['kind'])
        A,B,J,C,*_=constants(b)
        rr=np.roots([J,A*J+q*b['rho'],A*B])
        positive=[v.real for v in rr if abs(v.imag)<1e-9 and v.real>0]
        if not positive:
            bad+=1; count+=1; continue
        critical=min(positive); predicted=A0*d*delta/(q*d-A0*A0)
        err=abs(critical/predicted-1); worst=max(worst,err)
        bad+=err>.003; count+=1
    report('Q4 near absent self-regulation: critical a proportional to rho+h',count,bad,
           f'max asymptotic relative error={worst:.3g}')
    bad=0; worst=0
    for _ in range(NCASE):
        d=10**rng.uniform(-.5,.5); s=rng.uniform(.72,.97)
        A=s*d; h=d-A; q=A*A/d
        # Also an admissible M: epsilon=1+s, xi=d*(2*s*s-1)>0.
        b=dyn(d,q,h,-h,'M'); J=A-q; a=1e-4*A
        vals=eigvals(matrix([b,lr(a)])); pair=vals[np.argsort(abs(vals))[:2]]
        predicted=-J*a*a/(2*A*A)
        err=abs(pair.real.mean()/predicted-1); worst=max(worst,err)
        bad+=max(vals.real)>=0 or err>.003
    report('Q4 unregulated C0=1: stable a^2 damping',NCASE,bad,
           f'max damping relative error={worst:.3g}')
    bad=0; worst=0; stabilizing=0
    for _ in range(NCASE):
        h=10**rng.uniform(-.3,.3); r=h*rng.uniform(.2,.8); d=r; q=d*rng.uniform(.2,.8)
        b=dyn(d,q,h,-r); B=d*(h-r); pole=1j*np.sqrt(B)
        fixed=[lr(10**rng.uniform(-.7,.7)) for _ in range(int(rng.integers(1,4)))]
        P=np.prod([x['q']/(pole+x['d']) for x in fixed])
        corr=q*(pole-r)*P/(2*pole*pole)
        a=1e-7*min([r,h-r]+[x['d'] for x in fixed])
        vals=eigvals(matrix([b]+fixed+[lr(a)]))
        close=vals[np.argmin(abs(vals-pole))]
        err=abs((close-pole)/a-corr)/(1+abs(corr)); worst=max(worst,err)
        bad+=err>1e-4
        if abs(corr.real)>1e-5:
            bad+=(max(vals.real)<0)!=(corr.real<0)
        stabilizing+=corr.real<0
    report('Q4 nonzero imaginary internal poles: perturbation formula',NCASE,bad,
           f'max correction error={worst:.3g}, stabilizing cases={stabilizing}')
    bad=0; worst=0
    for _ in range(NCASE):
        h=10**rng.uniform(-.5,.5); d=h; q=d*rng.uniform(.1,.9)
        b=dyn(d,q,h,-h); a=1e-10*h
        vals=eigvals(matrix([b,lr(a)])); expected=.5*(a*q*h)**(1/3)
        err=abs(max(vals.real)/expected-1); worst=max(worst,err)
        bad+=max(vals.real)<=0 or err>.003
    report('Q4 unregulated V A=0: unstable a^(1/3) pair',NCASE,bad,
           f'max leading-growth error={worst:.3g}')
    bad=0
    for _ in range(NCASE):
        m=int(rng.integers(2,5)); bs=[]
        for k in range(m):
            h,xi=10**rng.uniform(-.3,.3,2); eps=rng.uniform(1,3)
            bs.append(dyn(xi+2*eps*h,xi+eps*h,h,-h,'M'))
        bad+=abscissa(bs+[lr(1e-12)])<=0
    report('Q4 multiple zero-pole host blocks: unstable slow-stage limit',NCASE,bad)
    bad=0
    for _ in range(NCASE):
        h,xi=10**rng.uniform(-.3,.3,2); eps=rng.uniform(1,3)
        b=dyn(xi+2*eps*h,xi+eps*h,h,-h,'M')
        m=int(rng.integers(2,7)); a=1e-6*min(h,xi)
        bad+=abscissa([b]+[lr(a)]*m)<=0
    report('Q3/4 m>=2 simultaneously long L stages',NCASE,bad)
    bad=0
    for _ in range(NCASE):
        n=int(rng.integers(1,10)); aa=10**rng.uniform(-1,1,n)
        for G in (rng.uniform(.01,.99),1.,rng.uniform(1.01,5.)):
            mx=max(eigvals(ring_matrix(aa,G)).real)
            if G<1: bad+=mx>=0
            elif G>1: bad+=mx<=0
            else: bad+=abs(mx)>1e-8
    report('Q5 positive gain and zero-root boundary',NCASE,bad)
    bad=0; worst=0
    for _ in range(NCASE):
        n=int(rng.integers(3,12)); aa=10**rng.uniform(-1,1,n); factor=10**rng.uniform(-1,1)
        g,w=threshold(aa); gg,ww=threshold(rng.permutation(aa)*factor)
        err=max(abs(gg/g-1),abs(ww/w/factor-1)); worst=max(worst,err)
        bad+=err>1e-9
    report('Q5 permutation and common-rate-scale invariance',NCASE,bad,
           f'max relative error={worst:.3g}')
    for kind in ('V','M'):
        bad=0; stable=0
        for _ in range(NCASE):
            host=random_dynamic(kind); a,b=10**rng.uniform(-2,2,2)
            A,B,J,C,*_=constants(host)
            c3=A+a+b; c2=B+A*(a+b)+a*b; c1=B*(a+b)+J*a*b; c0=C*a*b
            pred=(min(c3,c2,c1,c0)>0 and c3*c2>c1
                  and c3*c2*c1>c1*c1+c3*c3*c0)
            obs=abscissa([host,lr(a),lr(b)])<0
            bad+=pred!=obs; stable+=obs
        report('Q3 two-L quartic check '+kind,NCASE,bad,f'stable cases={stable}')


def explicit_examples():
    print('\nEXPLICIT COUNTEREXAMPLES',flush=True)
    b=dyn(1.,.5,1.,-.49)
    print('V reentrant example: DC modulus=',abs(b['num'][0]/b['den'][0]),
          'frequency peak=',exact_peak(b),flush=True)
    b=dyn(10.,5.,1.,-.8)
    print('V one-L-safe but amplifying: eight L rates=.001, leading real part=',
          abscissa([b]+[lr(.001)]*8),flush=True)
    print('Two individually stable V feedback blocks: individual leading real part=',
          abscissa([b]),'combined=',abscissa([b,b]),flush=True)
    b=dyn(.499,.2495,1.,-.5)
    print('Internally unstable V stabilized by two L: internal real part=',
          max(poly.polyroots(b['den']).real),'two-L rate .1 leading real part=',
          abscissa([b,lr(.1),lr(.1)]),flush=True)
    print('Unregulated M eps=1.5,xi=h=1: a=1 leading=',
          abscissa([dyn(4,2.5,1,-1,'M'),lr(1)]),
          'a=3 leading=',abscissa([dyn(4,2.5,1,-1,'M'),lr(3)]),flush=True)


def nonlinear_integrations():
    print('\nNONLINEAR DETERMINISTIC INTEGRATIONS',flush=True)
    # V: p=.5, q=.5, d=1, h=1, rho=-.49. Logistic host.
    # x is the L state scaled to equilibrium x*=1.
    boundaries=np.sort(np.roots([.01,.51*.01-.5*.49,.51*.51]))
    print('V + L Hopf boundaries a =',boundaries,flush=True)
    for a in (.5,.8,1.,1.5,3.,6.):
        def rhs(t,y):
            N,I,x=y
            return [1.51*N-.51*N*N-2*I, .5*x*(N-I)-.5*I, a*(2*I-x)]
        J=np.array([[.49,-2,0],[.5,-1,.25],[0,2*a,-a]])
        vals,vecs=np.linalg.eig(J); ix=np.argmax(vals.real); lam=vals[ix]
        initial=np.array([1.,.5,1.])+1e-4*vecs[:,ix].real
        inv=np.linalg.inv(vecs); T=min(600.,5/max(abs(lam.real),.001))
        ts=np.linspace(0,T,6001)
        sol=solve_ivp(rhs,[0,T],initial,t_eval=ts,rtol=2e-10,atol=2e-12,method='DOP853')
        disp=sol.y-np.array([1.,.5,1.])[:,None]
        amp=abs((inv@disp)[ix]); mask=(amp>1e-9)&(amp<3e-3)&(ts<T*.8)
        slope=np.polyfit(ts[mask],np.log(amp[mask]),1)[0]
        ok=sol.success and slope*lam.real>0 and abs(slope-lam.real)<.12*abs(lam.real)
        print(f'a={a:g}: eigenvalue={lam.real:+.8f}{lam.imag:+.8f}i, fitted nonlinear growth={slope:+.8f}, final distance={np.linalg.norm(disp[:,-1]):.5g}, min N/I/x={sol.y.min():.5g}, PASS={ok}',flush=True)
        if not ok: failures.append(('nonlinear integration',a))


def stochastic_check():
    print('\nNONLINEAR STOCHASTIC CHECK (OU log-input, not exp of distributional white noise)',flush=True)
    srng=np.random.default_rng(SEED+900)
    a=.35; d=2.; q=1.; rate=.7; sig=.02; dt=.01
    n=1600000; burn=100000
    noises=srng.normal(size=n); decay=np.exp(-rate*dt)
    kick=sig*np.sqrt(1-decay*decay)
    x=y=1.; eta=0.; data=np.empty((n-burn,2))
    for i in range(n):
        eta2=decay*eta+kick*noises[i]
        forcing=np.exp(.5*(eta+eta2))
        # Explicit midpoint for host/infected ODE, exact OU transition.
        dx=a*(y-x); dy=x*forcing*(2-y)-y
        xm=x+.5*dt*dx; ym=y+.5*dt*dy
        x+=dt*a*(ym-xm); y+=dt*(xm*forcing*(2-ym)-ym); eta=eta2
        if i>=burn: data[i-burn]=[np.log(x),np.log(y)]
    A=np.array([[-a,a,0.],[q,-d,q],[0.,0.,-rate]])
    B=np.diag([0.,0.,2*rate*sig**2])
    cov=solve_continuous_lyapunov(A,-B)
    observed=data.var(axis=0)
    def density(w,k):
        L=a/(1j*w+a); S=q/(1j*w+d)
        H=L*S/(1-L*S) if k==0 else S/(1-L*S)
        return abs(H)**2*2*rate*sig**2/(rate*rate+w*w)
    expected=np.array([quad(lambda w:density(w,k),0,np.inf,epsabs=1e-12)[0]/np.pi for k in range(2)])
    print(f'seed={SEED+900}, dt={dt}, retained duration={(n-burn)*dt:g}, sigma_eta={sig}; variance acceptance tolerance=12%',flush=True)
    for k,name in enumerate(('slow L density','S infected density')):
        rel=observed[k]/expected[k]-1
        print(f'{name}: spectral integral={expected[k]:.8g}, Lyapunov={cov[k,k]:.8g}, nonlinear log variance={observed[k]:.8g}, relative error={rel:+.3%}',flush=True)
        if abs(rel)>.12: failures.append(('stochastic variance',name))
        freqs,power=welch(data[:,k],fs=1/dt,nperseg=32768,noverlap=16384)
        for lo,hi in ((0.,.04),(.04,.15),(.15,.6)):
            mask=(freqs>=lo)&(freqs<hi); df=freqs[1]-freqs[0]
            # Welch is one-sided per Hz; S(w) here is two-sided per rad/time.
            obs=power[mask].sum()*df
            pred=sum(2*density(2*np.pi*f,k)*df for f in freqs[mask])
            print(f'  PSD band [{lo:g},{hi:g}) cycles/time: empirical/theory={obs/pred:.4f}',flush=True)


if __name__=='__main__':
    print(f'OFFLINE INDEPENDENT CHECKS; seed={SEED}; random cases per claim={NCASE}; FD root tolerance fixed at {FD_TOL:g}',flush=True)
    finite_difference_checks()
    stability_checks()
    cubic_checks()
    delay_converse_checks()
    slow_checks()
    r_checks()
    edge_checks()
    explicit_examples()
    nonlinear_integrations()
    stochastic_check()
    print('\nFINAL:', 'PASS' if not failures else 'FAIL', failures,flush=True)
    raise SystemExit(1 if failures else 0)
