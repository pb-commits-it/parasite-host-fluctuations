"""Local, reproducible linked-host checks. Run: python3 -I mechanism_checks.py.

Only numpy/scipy and the standard library. Three processes, one BLAS thread each.
Every quantitative result is printed, including machine-readable JSON records.
Protocol: protocol.md. Baseline: every rate fixed at its arithmetic mean.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm

FAILS = []


def emit(kind, **kw):
    print(kind + ' ' + json.dumps(kw, sort_keys=True), flush=True)


def check(name, ok, **kw):
    emit('CHECK', name=name, status='PASS' if ok else 'FAIL', **kw)
    if not ok:
        FAILS.append(name)


def model(name):
    if name == 'predator' or name.startswith('rotation:'):
        r, om = (.9, np.pi / 10) if name=='predator' else tuple(map(float,name.split(':')[1:]))
        M = r * np.array([[np.cos(om), -np.sin(om)], [np.sin(om), np.cos(om)]])
        S = (1 - r*r) * np.eye(2)
    elif name == 'shared':
        p = np.array([.25, .85])
        M = np.diag(p)
        sd = np.sqrt(1-p*p)
        S = np.outer(sd, sd) * np.array([[1, .9], [.9, 1]])
    elif name in ('competitors', 'fast', 'oldfast'):
        a, c, cor = {'competitors': (.55, .35, 0), 'fast': (.02, .02, -.8), 'oldfast': (.15, .05, -.8)}[name]
        M = np.array([[a, -c], [-c, a]])
        S = np.array([[1., cor], [cor, 1.]])
    elif name.startswith('common'):
        ph = float(name.split(':')[1])
        M = np.eye(2)*ph
        S = np.ones((2, 2))*(1-ph*ph)
    else:
        raise ValueError(name)
    G = solve_discrete_lyapunov(M, S)
    d = np.sqrt(np.diag(G))
    M = M / d[:, None] * d[None, :]
    S = S / np.outer(d, d)
    G = G / np.outer(d, d)
    return M, S, G


def covariance_closed(name, lag):
    if name == 'predator':
        th = lag*np.pi/10
        return .9**lag*np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if name == 'shared':
        p, q, gam = .25, .85, .9
        c = gam*np.sqrt((1-p*p)*(1-q*q))/(1-p*q)
        return np.diag([p**lag, q**lag])@np.array([[1, c], [c, 1]])
    if name in ('competitors', 'fast', 'oldfast'):
        a, c, cor = {'competitors': (.55, .35, 0), 'fast': (.02, .02, -.8), 'oldfast': (.15, .05, -.8)}[name]
        lp, lm = a-c, a+c
        vp, vm = (1+cor)/(1-lp*lp), (1-cor)/(1-lm*lm)
        wp, wm = vp/(vp+vm), vm/(vp+vm)
        return wp*lp**lag*np.ones((2, 2))+wm*lm**lag*np.array([[1,-1],[-1,1]])
    return model(name)[2] * float(name.split(':')[1])**lag


def ring(tau):
    tau = np.asarray(tau, float)
    n = len(tau)
    A = np.diag(1-1/tau)
    B = []
    for j in range(n):
        b = np.zeros((n,n)); b[(j+1)%n,j] = 1/tau[j]
        A += b; B.append(b)
    return A, B


def transform(tau, z):
    ans = 1+0j
    for a in tau:
        ans *= z/(a-(a-1)*z)
    return ans


def arc(tau, j, i):
    """Stays after earlier transfer j through current transfer i, inclusive."""
    ids = []; k = (j+1)%len(tau)
    while True:
        ids.append(tau[k])
        if k == i:
            break
        k = (k+1)%len(tau)
    return ids


def coefficient(tau, M, G, order=0, coupling=1., active=(1.,1.)):
    """Return K and its ordered-pair terms by rational matrix functions of M.
    Uses the two edges 0,1, with driver assignment swapped if order=1.
    coupling rescales only cross-covariances, preserving both marginal laws.
    """
    T = sum(tau); I = np.eye(2)
    def fm(stays):
        F = I.copy()
        for a in stays:
            F = F @ np.linalg.solve(a*I-(a-1)*M, M)
        return F
    R = fm(tau)
    background = np.linalg.solve(I-M, (I+M)@G)/(2*T)
    terms = np.zeros((2,2))
    who = [0,1] if not order else [1,0]
    for i in range(2):
        for j in range(2):
            a, b = who[i], who[j]
            U = np.linalg.solve(I-R, fm(arc(tau,j,i)))@G
            terms[i,j] = active[a]*active[b]*(background[a,b]-U[a,b])*(1 if a==b else coupling)
    return float(terms.sum()), terms


def lag_coefficient(tau, M, G, order=0, coupling=1., active=(1.,1.), nlag=1200):
    A, B = ring(tau); n = len(tau); T0 = sum(tau)
    v = np.asarray(tau)/T0; w = np.ones(n)
    Q = np.eye(n)-np.outer(v,w); TT = A@Q
    out = np.zeros((2,2)); who = [0,1] if not order else [1,0]
    for i in range(2):
        for j in range(2):
            a,b = who[i],who[j]
            fac = active[a]*active[b]*(1 if a==b else coupling)
            rho = G.copy(); vec = Q@B[j]@v
            val = G[a,b]/(2*T0*T0)
            for _ in range(nlag):
                rho = M@rho
                val -= rho[a,b]*(w@B[i]@vec)
                vec = TT@vec
            out[i,j] = T0*fac*val
    return float(out.sum())


def common_formula(a,b,phi,sign):
    th = phi/(1-phi)
    if sign == 1:
        return (2*a*b+th*(2*(a+b)-(a-b)**2))/((a+b)*(a*b+th*(a+b)))
    return th*(a+b)/(a*b+th*(a+b))


def all_roots(f, lo=1, hi=200):
    xx = np.geomspace(lo, hi, 401); ans=[]
    for a,b in zip(xx[:-1],xx[1:]):
        if f(a)*f(b)<0:
            ans.append(float(brentq(f,a,b,xtol=1e-11)))
    return ans


def derivative(f,x):
    h=2e-4*max(1.,x)
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)


def exact_derivative(tau,stage,M,G,order=0):
    vals,V=np.linalg.eig(M.astype(complex)); Vi=np.linalg.inv(V); T=sum(tau)
    ans=0j; who=[0,1] if order==0 else [1,0]
    for m,z in enumerate(vals):
        H=np.outer(V[:,m],Vi[m])@G
        R=transform(tau,z)
        logder=-(1-z)/(tau[stage]-(tau[stage]-1)*z)
        Rp=R*logder
        for i in range(2):
            for j in range(2):
                ids=[]; k=(j+1)%len(tau)
                while True:
                    ids.append(k)
                    if k==i: break
                    k=(k+1)%len(tau)
                F=transform([tau[k] for k in ids],z)
                Fp=F*logder if stage in ids else 0
                ans+=H[who[i],who[j]]*(-(1+z)/(2*T*T*(1-z))-(Fp*(1-R)+F*Rp)/(1-R)**2)
    return float(ans.real)


def rotation_parking_asymptote(r,omega,a,order):
    z=r*np.exp(1j*omega); th=z/(1-z); A=transform([a],z); B=transform([5],z)
    independent=float(((1+z)/(1-z)-2*A*B*th).real)
    sign=1 if order==0 else -1
    J=independent-sign*((A*A-1)*B*th).imag
    return float(-sign*A.imag),float(J),independent


def floquet(tau, z, sigma, order=0):
    """Exact deterministic periodic product; normalized arithmetic mean rates."""
    n = len(tau); p = z.shape[0]
    h = np.exp(sigma*z); h /= h.mean(axis=0)
    who=[0,1] if order==0 else [1,0]
    A,B=ring(tau)
    def growth(y):
        P=np.eye(n); logscale=0.
        for t in range(p):
            At=A.copy()
            At += B[0]*(np.exp(y)*h[t,who[0]]-1)
            At += B[1]*(h[t,who[1]]-1)
            P=At@P
            sc=np.linalg.norm(P); P/=sc; logscale+=np.log(sc)
        return (logscale+np.log(max(abs(np.linalg.eigvals(P)))))/p
    return float(brentq(growth,-2,2,xtol=1e-13))


def periodic_coefficient(tau, z, order):
    A,B=ring(tau); T0=sum(tau); n=len(tau); p=len(z)
    v=np.asarray(tau)/T0; w=np.ones(n); Q=np.eye(n)-np.outer(v,w); TT=A@Q
    who=[0,1] if order==0 else [1,0]
    val=0.
    for i in range(2):
        for j in range(2):
            a,b=who[i],who[j]
            val+=np.mean(z[:,a]*z[:,b])/(2*T0)
            vec=Q@B[j]@v
            for lag in range(1,4000):
                val-=T0*np.mean(z[:,a]*np.roll(z[:,b],lag))*(w@B[i]@vec)
                vec=TT@vec
    return float(val)


def analytic():
    print('BASELINE: every rate fixed at its arithmetic mean; also same mean host abundance here.', flush=True)
    print('All parameters illustrative. K is D/sigma^2 in the small-noise limit.',flush=True)
    names=['predator','shared','competitors','fast','oldfast']
    for name in names:
        M,S,G=model(name)
        err=max(np.max(abs(np.linalg.matrix_power(M,l)@G-covariance_closed(name,l))) for l in range(26))
        block=np.block([[np.linalg.matrix_power(M,i-j)@G if i>=j else (np.linalg.matrix_power(M,j-i)@G).T for j in range(26)] for i in range(26)])
        check('stationarity '+name, max(abs(np.linalg.eigvals(M)))<1 and min(np.linalg.eigvalsh(S))>=-1e-11 and min(np.linalg.eigvalsh(block))>=-1e-11 and err<1e-11,
              M=M.tolist(), innovation=S.tolist(), G=G.tolist(), covariance_error=err, block_min_eigenvalue=float(min(np.linalg.eigvalsh(block))), lyapunov_error=float(np.max(abs(G-M@G@M.T-S))))
        for a in (1.,5.,50.):
            for order in (0,1):
                K,t=coefficient([5,a,5],M,G,order)
                emit('GRID',model=name,stay=a,order=order,K=K,K1=float(t[order,order]),K2=float(t[1-order,1-order]),cross=float(t[0,1]+t[1,0]))
    rng=np.random.default_rng(14112026); errors=[]
    for i in range(60):
        tau=rng.uniform(1,70,size=int(rng.integers(2,7))).tolist()
        M=rng.normal(size=(2,2)); M*=rng.uniform(.05,.96)/max(abs(np.linalg.eigvals(M)))
        L=rng.normal(size=(2,2)); S=L@L.T+.2*np.eye(2)
        G=solve_discrete_lyapunov(M,S); d=np.sqrt(np.diag(G)); M=M/d[:,None]*d[None,:]; G=G/np.outer(d,d)
        order=i%2; q=rng.uniform(-1,1)
        errors.append(abs(coefficient(tau,M,G,order,q)[0]-lag_coefficient(tau,M,G,order,q)))
    check('transform versus general matrix lag sum',max(errors)<2e-10,max_error=max(errors),cases=len(errors))
    for phi in (0.,.25,.8,.95):
        M,S,G=model('common:'+str(phi))
        for a,b in ((5,5),(1,10),(10,60)):
            for sign in (1,-1):
                K=coefficient([b,a],M,G,coupling=sign)[0]
                expected=common_formula(a,b,phi,sign)
                check('discrete common closed form',abs(K-expected)<2e-11,a=a,b=b,phi=phi,sign=sign,K=K)
    a,b=1.,10.; thcrit=2*a*b/((a-b)**2-2*(a+b)); phcrit=thcrit/(1+thcrit)
    emit('COMMON_BOUNDARY',a=a,b=b,theta=thcrit,phi=phcrit,below=common_formula(a,b,.2,1),above=common_formula(a,b,.8,1))
    M,S,G=model('predator')
    lags=np.arange(101); vals=np.array([(np.linalg.matrix_power(M,int(l))@G)[1,0] for l in lags]); peak=int(lags[np.argmax(vals)])
    emit('PREDATOR_LAG',peak_lag=peak,peak_covariance=float(vals[peak]),quarter_period=5)
    bounds={}
    for back in (10.,50.):
        f=lambda a:coefficient([5,a,back-5],M,G)[0]
        roots=all_roots(f); bounds['cancel'+str(int(back))]=roots
        from scipy.optimize import minimize_scalar
        opt=minimize_scalar(f,bounds=(1,200),method='bounded')
        emit('CANCELLATION',return_mean=back,roots=roots,minimum_stay=float(opt.x),minimum_K=float(opt.fun),endpoints=[f(1),f(200)])
        for rt in roots:
            emit('CANCELLATION_SIDES',return_mean=back,root=rt,low=rt*.7,K_low=f(rt*.7),high=rt*1.3,K_high=f(rt*1.3))
    for name in ('predator','shared','competitors','fast','oldfast'):
        M,S,G=model(name)
        for tau,order in (([5,5,5],0),([5,50,5],1),([5,5,45],0)):
            K,t=coefficient(tau,M,G,order); k1,k2=t[0,0],t[1,1]; X=t[0,1]+t[1,0]
            emit('CROSS_BOUNDARY',model=name,tau=tau,order=order,K1=float(k1),K2=float(k2),X=float(X),zero_q=float(-(k1+k2)/X) if X else None,helps_second_q=float(-k2/X) if X else None,beats_both_q=float(-max(k1,k2)/X) if X else None)
    M,S,G=model('predator')
    for which in ('predator','steady'):
        f=lambda b:coefficient([5,5,b] if which=='predator' else [b,5,5],M,G)[0]
        fi=lambda b:coefficient([5,5,b] if which=='predator' else [b,5,5],M,G,coupling=0)[0]
        roots=all_roots(lambda b:derivative(f,b),lo=1.001)
        rootsi=all_roots(lambda b:derivative(fi,b),lo=1.001)
        bounds['parking_'+which]=roots
        z=.9*np.exp(1j*np.pi/10)
        lim=-transform([5],z).imag
        emit('PARKING',stage=which,linked_derivative_roots=roots,independent_derivative_roots=rootsi,limit=lim,
             values=[dict(stay=b,K=f(b),independent=fi(b),derivative=derivative(f,b)) for b in (1.,5.,10.,50.,200.,1e6)])
    # Long dwell on the forward arc destroys the positive delayed association.
    f=lambda a:coefficient([5,a,5],M,G)[0]
    roots=all_roots(lambda a:derivative(f,a),lo=1.001)
    bounds['forward_extrema']=roots
    emit('FORWARD_PARKING',roots=roots,values=[dict(stay=a,K=f(a),derivative=derivative(f,a)) for a in (1.,5.,10.,50.,200.)],limit=float(transform([5,5],.9*np.exp(1j*np.pi/10)).imag))
    for name in ('predator','shared','competitors'):
        MH,SH,GH=model(name)
        for a in (1.,5.,50.):
            for order in (0,1):
                f=lambda b:coefficient([b,a,5],MH,GH,order)[0]
                rr=all_roots(lambda b:derivative(f,b),lo=1.001)
                emit('STEADY_SCAN',model=name,forward=a,order=order,derivative_roots=rr,values=[dict(stay=b,K=f(b),derivative=derivative(f,b)) for b in (1.,5.,50.,200.,10000.)])
    # Fixed-generation-time comparison: split a steady return stage, add linked transfer.
    for name in ('predator','fast','shared'):
        M,S,G=model(name)
        old=coefficient([10,5],M,G,active=(1,0))[0]
        new=coefficient([5,5,5],M,G)[0]
        _,terms=coefficient([5,5,5],M,G)
        X=terms[0,1]+terms[1,0]
        emit('FIXED_GENERATION',model=name,T=15,one_host_two_stages=old,two_hosts_three_stages=new,change=new-old,coupling_boundary=float((old-np.trace(terms))/X))
    # Registered exploratory scan of long steady-stage sign.
    harm=[]; isolated=[]; derr=[]
    for r in (.3,.6,.9,.98):
        for frac in (.1,.3,.5,.7,.9):
            om=np.pi*frac; name='rotation:'+str(r)+':'+str(om); M,S,G=model(name)
            for a in (1.,5.,50.):
                for order in (0,1):
                    lim,J,Ji=rotation_parking_asymptote(r,om,a,order)
                    f=lambda b:coefficient([b,a,5],M,G,order)[0]
                    d=exact_derivative([10000,a,5],0,M,G,order)
                    derr.append(abs(d-derivative(f,10000)))
                    if J<-.0001:
                        row=dict(model=name,r=r,omega=om,forward=a,order=order,limit=lim,inverse_stay_coefficient=J,independent_coefficient=Ji,derivative_at_10000=d)
                        harm.append(row)
                        if Ji>0: isolated.append(row)
    check('exact derivative versus finite difference',max(derr)<1e-10,max_error=max(derr))
    emit('EVENTUAL_HARM_SEARCH',found=len(harm),interaction_isolated=len(isolated),first=harm[0] if harm else None,first_isolated=isolated[0] if isolated else None)
    bounds['eventual_harm']=(isolated or harm)[:1]
    if bounds['eventual_harm']:
        row=bounds['eventual_harm'][0]; M,S,G=model(row['model']); a=row['forward']; order=row['order']
        f=lambda b:coefficient([b,a,5],M,G,order)[0]
        emit('EVENTUAL_HARM_EXAMPLE',**row,roots=all_roots(lambda b:exact_derivative([b,a,5],0,M,G,order),lo=1.001),values=[dict(stay=b,K=f(b),derivative=exact_derivative([b,a,5],0,M,G,order)) for b in (1.,5.,20.,50.,200.)])
        fn=lambda om:rotation_parking_asymptote(row['r'],om,a,order)[1]
        omega_roots=all_roots(fn,lo=np.pi*.1,hi=np.pi*.9)
        emit('EVENTUAL_HARM_BOUNDARY',r=row['r'],forward=a,order=order,omega_roots=omega_roots,
             low=dict(omega=np.pi/10,J=fn(np.pi/10)),high=dict(omega=np.pi/2,J=fn(np.pi/2)))
    # Exact periodic products independently test directed lag formulas and signs.
    period=20; t=np.arange(period); z=np.sqrt(2)*np.column_stack([np.cos(2*np.pi*t/period),np.sin(2*np.pi*t/period)])
    for tau in ([5,1,5],[5,5,5],[5,50,5],[5,5,45]):
        for order in (0,1):
            K=periodic_coefficient(tau,z,order)
            om=2*np.pi/period; MR=np.array([[np.cos(om),-np.sin(om)],[np.sin(om),np.cos(om)]])
            KT=coefficient(tau,MR,np.eye(2),order)[0]
            check('periodic transform versus lag sum',abs(K-KT)<2e-10,tau=tau,order=order,error=abs(K-KT))
            small=floquet(tau,z,.002,order)/.002**2
            check('Floquet versus covariance coefficient',abs(K-small)<3e-5,tau=tau,order=order,K=K,exact_small=small,D_sigma1=floquet(tau,z,1.,order))
    # Exact bound equality in an iid environment, calculated by scalar path sums.
    emit('BOUND_EQUALITY',sigma=1.,D=2*1.**2/2,description='two transfers, no stays, mixing Gaussian environment')
    return bounds


def build_cases(bounds):
    cases=[]
    def add(label,name,tau,sigma,order=0,q=1.,active=(1.,1.),replicate=0):
        M,S,G=model(name)
        K=coefficient(tau,M,G,order,q,active)[0]
        cases.append(dict(label=label,model=name,tau=list(tau),sigma=float(sigma),order=order,q=float(q),active=list(active),K=K,replicate=replicate))
    for name in ('predator','shared','competitors'):
        for a in (1.,5.,50.):
            for order in (0,1):
                for sig in (.3,1.):
                    add('main',name,[5,a,5],sig,order)
                    add('independent',name,[5,a,5],sig,order,q=0)
        for order in (0,1):
            add('small_noise',name,[5,5,5],.05,order)
        for active in ((1,0),(0,1)):
            for sig in (.3,1.):
                add('single',name,[5,5,5],sig,active=active)
    for name in ('fast','oldfast'):
        for sig in (.05,.3,1.):
            add('competition_contrast',name,[5,5,5],sig)
            add('competition_single',name,[5,5,5],sig,active=(1,0))
    for back in (10,50):
        for rt in bounds['cancel'+str(back)]:
            for factor in (.7,1.,1.3):
                for sig in (.3,1.):
                    add('cancel_'+str(factor),'predator',[5,rt*factor,back-5],sig)
            add('cancel_validation','predator',[5,rt,back-5],1.,replicate=1)
    for a in (10.,20.,30.,40.,150.):
        for sig in (.05,.3,1.):
            add('finite_cancellation_bracket','predator',[5,a,45],sig)
    for b in (1.,10.,50.,200.):
        for sig in (.3,1.):
            add('predator_parking','predator',[5,5,b],sig)
            add('parking_independent','predator',[5,5,b],sig,q=0)
    for b in (1.,5.,50.,200.):
        for sig in (.3,1.):
            add('steady_reversed','predator',[b,5,5],sig,order=1)
    for name in ('predator','fast','shared'):
        for sig in (.3,1.):
            add('fixed_generation_old',name,[10,5],sig,active=(1,0))
    # Valid coupling interpolation: two independent copies of the same VAR.
    for name,tau,order in (('predator',[5,50,5],1),('fast',[5,5,5],0)):
        M,S,G=model(name); K,t=coefficient(tau,M,G,order); X=t[0,1]+t[1,0]
        for typ,rt in (('total',-np.trace(t)/X),('add',-t[1,1]/X)):
            if 0<rt<1:
                for q in (max(0,rt-.15),min(1,rt+.15)):
                    for sig in (.05,.3,1.):
                        add('coupling_'+typ,name,tau,sig,order,q=q)
    for sig in (.05,.3,1.):
        add('coupling_single','predator',[5,50,5],sig,1,active=(1,0))
    for phi in (0.,.2,.8):
        for sign in (1,-1):
            for sig in (.05,.3,1.):
                add('common_control','common:'+str(phi),[10,1],sig,q=sign)
    for sig in (.05,.3,1.):
        add('common_equal','common:0.8',[5,5],sig)
    for name,order in (('predator',0),('predator',1),('shared',0),('competitors',0)):
        add('stress',name,[5,5,5],2.,order)
    for row in bounds.get('eventual_harm',[]):
        for b in (1.,5.,20.,50.,200.):
            for sig in (.05,.3,1.):
                add('eventual_harm',row['model'],[b,row['forward'],5],sig,row['order'])
                if b in (20.,50.,200.):
                    add('eventual_harm_independent',row['model'],[b,row['forward'],5],sig,row['order'],q=0)
    return cases


def sqrt_psd(A):
    d,U=np.linalg.eigh(A)
    return U@np.diag(np.sqrt(np.maximum(d,0)))


def simulate_batch(cases, y, steps, pairs, seed, burn=2000, empirical=False, parameter='y'):
    """Full random matrix products and their y derivatives. No K in recursion.
    Paths are antithetic; the independent sampling unit is the +/- pair.
    Cases within a batch share environmental histories for paired contrasts.
    """
    M,S,G=model(cases[0]['model']); L=sqrt_psd(S); LG=sqrt_psd(G)
    rng=np.random.default_rng(seed); nc=len(cases); nr=pairs*2
    n=np.array([len(c['tau']) for c in cases]); tau=np.ones((nc,3))
    for i,c in enumerate(cases): tau[i,:n[i]]=c['tau']
    valid=np.arange(3)[None,:]<n[:,None]
    s=(1-1/tau)*valid; g=valid/tau; g[:,0]*=np.exp(y)
    s=s[:,None,:]; g=g[:,None,:]
    who=np.array([[0,1] if c['order']==0 else [1,0] for c in cases])
    sig=np.array([c['sigma'] for c in cases])[:,None,None]
    active=np.array([c['active'] for c in cases])
    act=np.take_along_axis(active,who,axis=1)[:,None,:]
    q=np.array([c['q'] for c in cases])[:,None]
    prev=np.array([[(j-1)%ni for j in range(3)] for ni in n])
    # dummy stages are explicitly zeroed after indexing
    X=np.broadcast_to((tau*valid/tau.sum(axis=1)[:,None])[:,None,:],(nc,nr,3)).copy()
    X/=X.sum(axis=2)[:,:,None]
    V=np.zeros_like(X); logs=np.zeros((nc,nr)); slopes=np.zeros_like(logs)
    zz=rng.normal(size=(pairs,2,2))@LG.T
    normals=rng.normal(size=(steps+burn,pairs,2,2))
    hm=np.ones((nc,nr,3)); index=prev[:,None,:]
    if empirical:
        history=np.zeros((steps+burn,pairs,2))
    for t in range(steps+burn):
        zz=zz@M.T+normals[t]@L.T
        z1=zz[:,0,0][None,:]
        z2=q*zz[:,0,1][None,:]+np.sqrt(np.maximum(0,1-q*q))*zz[:,1,1][None,:]
        raw=np.stack([np.broadcast_to(z1,(nc,pairs)),z2],axis=2)
        raw=np.take_along_axis(raw,who[:,None,:],axis=2)
        ex=sig*act*raw; shift=.5*(sig*act)**2
        hm[:,:pairs,:2]=np.exp(ex-shift)
        hm[:,pairs:,:2]=np.exp(-ex-shift)
        out=g*hm*X; dv=g*hm*V
        if parameter=='y': dv[:,:,0]+=out[:,:,0]
        else: dv[:,:,1]-=hm[:,:,1]*X[:,:,1]/tau[:,1,None]**2
        Y=s*X+np.take_along_axis(out,index,axis=2)
        W=s*V+np.take_along_axis(dv,index,axis=2)
        if parameter!='y': W[:,:,1]+=X[:,:,1]/tau[:,1,None]**2
        Y*=valid[:,None,:]; W*=valid[:,None,:]
        norm=Y.sum(axis=2); normd=W.sum(axis=2); ds=normd/norm
        X=Y/norm[:,:,None]; V=W/norm[:,:,None]-X*ds[:,:,None]
        if t>=burn:
            logs+=np.log(norm); slopes+=ds
        if empirical: history[t]=zz[:,0,:]
    logs=(logs[:,:pairs]+logs[:,pairs:])/(2*steps)
    slopes=(slopes[:,:pairs]+slopes[:,pairs:])/(2*steps)
    cov=None
    if empirical:
        cov=np.array([np.einsum('tpi,tpj->pij',history[burn:],history[burn-lag:burn+steps-lag])/steps for lag in range(26)]).transpose(1,0,2,3)
    return logs,slopes,cov


def run_batch(item):
    bid,cases=item; start=time.monotonic()
    # Pilot Newton solve using independent paths from the final run.
    y=np.array([c['K']*c['sigma']**2 for c in cases])
    for it in range(3):
        r,d,_=simulate_batch(cases,y,12000,24,120000+bid)
        delta=r.mean(axis=1)/d.mean(axis=1); y-=delta
        if max(abs(delta))<2e-5: break
    # Independent long run. A Newton correction and same-path verification.
    saved_cov=None
    for it in range(3):
        r,d,cov=simulate_batch(cases,y,100000,24,880000+bid,empirical=(it==0 and cases[0]['model'] in ('predator','shared','competitors') and cases[0]['label']=='main'))
        if cov is not None: saved_cov=cov
        slope=d.mean(axis=1); delta=r.mean(axis=1)/slope
        se=r.std(axis=1,ddof=1)/np.sqrt(r.shape[1])/slope
        new=y-delta
        if np.all(abs(delta)<np.maximum(2e-5,.2*se)):
            y=new; break
        y=new
    records=[]
    for i,c in enumerate(cases):
        rec=dict(c,D=float(y[i]),SE=float(se[i]),last_correction=float(delta[i]),slope=float(slope[i]),steps=100000,pairs=24)
        rec['samples']=(y[i]-(r[i]-r[i].mean())/slope[i]).tolist()
        rec['root_resolved']=bool(abs(delta[i])<max(2e-5,.2*se[i]))
        records.append(rec)
    return records,saved_cov,float(time.monotonic()-start)


def scalar_controls():
    zz,ww=roots_hermitenorm(100); ww/=np.sqrt(2*np.pi)
    rng=np.random.default_rng(991); z=rng.normal(size=(24,200000))
    for tau in (2.,10.):
        for sig in (.3,1.):
            s=1-1/tau
            def f(y): return float(ww@np.log(s+np.exp(y+sig*zz-sig*sig/2)/tau))
            D=brentq(f,0,sig*sig/2)
            path=.5*(np.log(s+np.exp(D+sig*z-sig*sig/2)/tau)+np.log(s+np.exp(D-sig*z-sig*sig/2)/tau)).mean(axis=1)
            slope=float(ww@(np.exp(D+sig*zz-sig*sig/2)/(tau*s+np.exp(D+sig*zz-sig*sig/2))))
            se=float(path.std(ddof=1)/np.sqrt(24)/slope)
            err=float(path.mean()/slope)
            check('scalar exact quadrature',abs(err)<5*se+.001,tau=tau,sigma=sig,D_exact=D,MC_difference=err,SE=se)
    # No stays, one cohort; antithetic histories cancel the Gaussian part exactly.
    sig=1.; y=sig*sig; p=z[:,:20000]
    rates=.5*((y/2+sig*p-sig*sig/2).mean(axis=1)+(y/2-sig*p-sig*sig/2).mean(axis=1))
    check('single cohort bound equality',max(abs(rates))<1e-14,D=y,max_abs_rate=float(max(abs(rates))))


def report_contrasts(records):
    # paired only within a simulation batch; independent SE otherwise
    def contrast(a,b,label):
        diff=a['D']-b['D']
        if a['batch']==b['batch']:
            se=float(np.std(np.array(a['samples'])-np.array(b['samples']),ddof=1)/np.sqrt(a['pairs']))
        else: se=float(np.hypot(a['SE'],b['SE']))
        emit('CONTRAST',label=label,model=a['model'],tau=a['tau'],sigma=a['sigma'],order=a['order'],difference=diff,SE=se,sign_resolved=abs(diff)>5*se)
    for a in records:
        if a['label']=='main':
            bs=[b for b in records if b['model']==a['model'] and b['tau']==a['tau'] and b['sigma']==a['sigma']]
            for b in bs:
                if b['label']=='independent' and b['order']==a['order']: contrast(a,b,'linked minus independent')
                if b['label']=='main' and a['order']==0 and b['order']==1: contrast(a,b,'order 0 minus order 1')
                if b['label']=='single' and a['order']==0: contrast(a,b,'linked minus single '+str(b['active']))
        if a['label']=='cancel_validation':
            for b in records:
                if b['label']=='cancel_1.0' and b['tau']==a['tau'] and b['sigma']==a['sigma']: contrast(a,b,'independent repeat at cancellation')
        if a['label']=='fixed_generation_old':
            for b in records:
                if b['label'] in ('main','competition_contrast') and b['model']==a['model'] and b['tau']==[5.,5.,5.] and b['order']==0 and b['sigma']==a['sigma']: contrast(b,a,'fixed generation added host')
        if a['label']=='coupling_add' and a['model']=='predator':
            for b in records:
                if b['label']=='coupling_single' and b['sigma']==a['sigma']: contrast(a,b,'coupling added-host boundary q='+str(a['q']))
    for a in records:
        if a['label']=='predator_parking':
            for b in records:
                if b['label']=='parking_independent' and b['tau']==a['tau'] and b['sigma']==a['sigma']: contrast(a,b,'predator parking linked minus independent')
        if a['label'] in ('steady_reversed','eventual_harm'):
            for b in records:
                if b['label']==a['label'] and b['sigma']==a['sigma'] and (b['tau'][0],a['tau'][0]) in ((1.,5.),(20.,50.),(50.,200.)):
                    contrast(a,b,'longer steady stage '+a['label'])
        if a['label']=='common_equal':
            zz,ww=roots_hermitenorm(100); ww/=np.sqrt(2*np.pi); sig=a['sigma']; tau=a['tau'][0]
            f=lambda D:float(ww@np.log(1-1/tau+np.exp(D/2+sig*zz-sig*sig/2)/tau))
            exact=brentq(f,0,sig*sig)
            check('equal-stage common driver exact quadrature',abs(a['D']-exact)<5*a['SE']+.001,sigma=sig,D_exact=exact,D_simulated=a['D'],SE=a['SE'])
    finite=[a for a in records if a['label'] in ('finite_cancellation_bracket','cancel_0.7','cancel_1.0','cancel_1.3') and a['sigma']==1.]
    neg=[a for a in finite if a['D']+5*a['SE']<0]; pos=[a for a in finite if a['D']-5*a['SE']>0]
    emit('FINITE_CANCELLATION_BRACKET',negative_max_stay=max((a['tau'][1] for a in neg),default=None),positive_min_stay=min((a['tau'][1] for a in pos),default=None),sigma=1.)


def finite_stay_root():
    c=dict(label='direct_stay_root',model='predator',tau=[5.,35.,45.],sigma=1.,order=0,q=1.,active=[1.,1.])
    y=np.zeros(1)
    r,d,_=simulate_batch([c],y,5000,24,511,parameter='stay')
    c['tau'][1]+=.001; rp,_,_=simulate_batch([c],y,5000,24,511,parameter='stay')
    c['tau'][1]-=.002; rm,_,_=simulate_batch([c],y,5000,24,511,parameter='stay')
    c['tau'][1]+=.001
    err=abs(d.mean()-(rp.mean()-rm.mean())/.002)
    check('pathwise stay derivative',err<1e-8,error=float(err))
    for it in range(4):
        r,d,_=simulate_batch([c],y,12000,24,718,parameter='stay')
        delta=float(r.mean()/d.mean()); c['tau'][1]-=delta
        if abs(delta)<.001: break
    for it in range(4):
        r,d,_=simulate_batch([c],y,200000,48,719,parameter='stay')
        delta=float(r.mean()/d.mean()); se=float(r.std(ddof=1)/np.sqrt(48)/abs(d.mean()))
        c['tau'][1]-=delta
        if abs(delta)<max(.001,.1*se): break
    check('finite stay root residual',abs(delta)<max(.001,.1*se),correction=delta,SE=se)
    emit('FINITE_STAY_ROOT',sigma=1.,return_stays=[5.,45.],stay=c['tau'][1],SE=se,last_correction=delta,steps=200000,pairs=48,slope=float(d.mean()))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--analytic',action='store_true'); args=parser.parse_args()
    start=time.monotonic()
    emit('RUN',seed=14112026,processes=3,protocol_mtime=Path(__file__).with_name('protocol.md').stat().st_mtime,started=time.time())
    bounds=analytic()
    if args.analytic:
        emit('SUMMARY',failures=FAILS,seconds=time.monotonic()-start); return
    scalar_controls()
    cases=build_cases(bounds)
    # Moderate batch sizes keep vectorization efficient; retain main paired cases together.
    batches=[]
    for name in dict.fromkeys(c['model'] for c in cases):
        group=[c for c in cases if c['model']==name and c['replicate']==0]
        if len(group)>75:
            mainc=[c for c in group if c['label'] in ('main','independent','single','small_noise')]
            rest=[c for c in group if c not in mainc]
            batches.extend([mainc,rest])
        else: batches.append(group)
    for c in cases:
        if c['replicate']: batches.append([c])
    # derivative finite-difference control on identical simulated paths
    c=cases[0]; yy=np.array([.03]); rr,dd,_=simulate_batch([c],yy,5000,24,142)
    rp,_,_=simulate_batch([c],yy+.0001,5000,24,142); rm,_,_=simulate_batch([c],yy-.0001,5000,24,142)
    err=float(abs(dd.mean()-(rp.mean()-rm.mean())/.0002))
    check('pathwise y derivative',err<1e-8,error=err)
    emit('SIMULATION_PLAN',cases=len(cases),batches=len(batches),steps=100000,independent_pairs=24,burn=2000)
    records=[]
    with mp.get_context('spawn').Pool(3) as pool:
        for bid,(recs,cov,elapsed) in enumerate(pool.imap(run_batch,enumerate(batches))):
            for rec in recs:
                rec['batch']=bid; records.append(rec)
                emit('SIM',**{k:v for k,v in rec.items() if k!='samples'})
                if not rec['root_resolved']: FAILS.append('root '+str(rec))
                bound=rec['sigma']**2*sum(x*x for x in rec['active'])/2
                if rec['D']>bound+5*rec['SE']: FAILS.append('bound '+str(rec))
                if rec['sigma']==.05:
                    discrepancy=abs(rec['D']/.05**2-rec['K']); tol=5*rec['SE']/.05**2+.02
                    check('small-noise '+rec['model']+' '+rec['label'],discrepancy<=tol,tau=rec['tau'],order=rec['order'],q=rec['q'],K=rec['K'],estimated_K=rec['D']/.05**2,tolerance=tol)
            if cov is not None:
                name=recs[0]['model']; mean=cov.mean(axis=0); se=cov.std(axis=0,ddof=1)/np.sqrt(cov.shape[0]); exact=np.array([covariance_closed(name,l) for l in range(26)])
                worst=float(np.max(abs(mean-exact)/(5*se+.002)))
                check('empirical covariance '+name,worst<=1,max_tolerance_ratio=worst,max_error=float(np.max(abs(mean-exact))))
            emit('BATCH_DONE',batch=bid,seconds=elapsed)
    report_contrasts(records)
    finite_stay_root()
    emit('SUMMARY',failures=FAILS,simulations=len(records),seconds=time.monotonic()-start,within_15_minutes=time.monotonic()-start<900)


if __name__=='__main__': main()
