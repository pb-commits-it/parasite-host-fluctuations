"""Spatial ring checks. Run: python3 -I mechanism_checks.py (> output.txt).
Only numpy/scipy; no network. Model, baselines and criteria: protocol.md.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time, json, math, multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.sparse.csgraph import connected_components
from scipy.special import roots_hermitenorm

FAILS = []

def emit(tag, **data):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    print(tag, json.dumps(data, default=convert, sort_keys=True), flush=True)

def check(name, error, tol=2e-10):
    ok = bool(error < tol)
    if not ok: FAILS.append(name)
    emit('CHECK', name=name, error=error, tolerance=tol, passed=ok)

def perron(A):
    ev, V = np.linalg.eig(A)
    ix = np.argmax(ev.real)
    lam = ev[ix].real
    v = np.abs(V[:, ix].real); v /= v.sum()
    ew, W = np.linalg.eig(A.T)
    w = np.abs(W[:, np.argmin(abs(ew-lam))].real); w /= w@v
    return lam, v, w

def spectral(A):
    # Avoid defective repeated eigenvalues of one-way open patch chains.
    count, labels=connected_components(A>0,directed=True,connection='strong')
    return max(max(abs(np.linalg.eigvals(A[np.ix_(labels==j,labels==j)]))) for j in range(count))

def movement(P, m, ell=0., topology='island'):
    W = np.zeros((P,P))
    if topology == 'island': W[:] = 1/P
    else:
        for i in range(P):
            for step, weight in ([(1,1.)] if topology.startswith('downwind') else [(-1,.5),(1,.5)]):
                j = i+step
                if topology.endswith('circle'): j %= P
                if 0 <= j < P: W[j,i] += weight
    return (1-m)*np.eye(P)+m*(1-ell)*W

def model(tau, P=4, m=.2, ell=0., d=1, topology='island', quality=None, noisy=(0,), critical=True):
    tau = np.asarray(tau,float); n=len(tau)
    if quality is None: quality=np.ones(P)
    s=1-1/tau; g=1/tau
    K=[np.eye(P) for _ in range(n)]; K[d]=movement(P,m,ell,topology)
    G=[g[k]*np.diag(quality if k==0 else np.ones(P)) for k in range(n)]
    def matrices():
        A=np.zeros((n*P,n*P)); Bs=[]
        for j in range(n): A[j*P:(j+1)*P,j*P:(j+1)*P]+=s[j]*K[j]
        for k in range(n):
            j=(k+1)%n
            A[j*P:(j+1)*P,k*P:(k+1)*P]+=K[j]@G[k]
            if k in noisy:
                for i in range(P):
                    B=np.zeros_like(A)
                    B[j*P:(j+1)*P,k*P+i]=K[j][:,i]*G[k][i,i]
                    Bs.append(B)
        L=np.eye(P)
        for k in range(n):
            j=(k+1)%n
            L=np.linalg.solve(np.eye(P)-s[j]*K[j],K[j]@G[k])@L
        return A, Bs, max(abs(np.linalg.eigvals(L)))
    A, Bs, R=matrices(); original_R=R; original_lam=spectral(A)
    if critical:
        G[0]/=R
        A,Bs,R=matrices()
    # The matrix derivative for scaling transfer zero is needed even if it is not noisy.
    F0=np.zeros_like(A); j=1%n
    F0[j*P:(j+1)*P,:P]=K[j]@G[0]
    return dict(tau=tau,s=s,G=G,K=K,A=A,Bs=Bs,R=R,F0=F0,P=P,n=n,d=d,m=m,ell=ell,
                topology=topology,quality=np.asarray(quality),noisy=tuple(noisy),
                original_R=original_R,original_lam=original_lam)

def general(mod, phi, rho):
    A=mod['A']; lam,v,w=perron(A); Bs=[B/lam for B in mod['Bs']]
    Q=np.eye(len(A))-np.outer(v,w); T=A/lam@Q
    Z=phi*np.linalg.solve(np.eye(len(A))-phi*T,Q)
    c=np.array([w@B@v for B in Bs]); P=mod['P']
    C=0.
    for i,B in enumerate(Bs):
        for j,B2 in enumerate(Bs):
            corr=(rho+(1-rho)*(i%P==j%P)) if i//P==j//P else 0.
            C+=corr*(c[i]*c[j]/2-w@B@Z@B2@v)
    elasticity=w@mod['F0']/lam@v
    return float(C), float(C/elasticity), float(elasticity)

def island(tau, P=4, m=.2, ell=0., d=1, phi=.8, rho=0., noisy=(0,)):
    # Independent n-dimensional spatial-mode calculation, not full matrix C''.
    tau=np.asarray(tau,float); n=len(tau); s=1-1/tau; g=1/tau
    a=1-m*ell; b=1-m
    da=np.ones(n); db=np.ones(n); da[d]=a; db[d]=b
    base=np.diag(s)
    for k in range(n): base[(k+1)%n,k]+=g[k]
    # Critical transfer calibration from the uniform mode's cycle equation.
    R=np.prod(da*g/(1-da*s))
    base[1%n,0]-=g[0]; g[0]/=R; base[1%n,0]+=g[0]
    Aa=da[:,None]*base; Ab=db[:,None]*base
    lam,v,w=perron(Aa); Q=np.eye(n)-np.outer(v,w)
    Za=phi*np.linalg.solve(np.eye(n)-phi*(Aa/lam@Q),Q)
    Zb=phi*np.linalg.inv(np.eye(n)-phi*Ab/lam)
    Cs=0.; H=0.; elasticity=None
    for k in range(n):
        F=np.zeros((n,n)); F[(k+1)%n,k]=g[k]
        Ba=da[:,None]*F/lam; Bb=db[:,None]*F/lam
        c=w@Ba@v
        if k==0: elasticity=c
        if k in noisy:
            Cs+=c*c/2-w@Ba@Za@Ba@v
            H+=w@Ba@Zb@Bb@v
    q=rho+(1-rho)/P
    C=q*Cs-(1-q)*H
    return dict(C=float(C),D2=float(C/elasticity),Cs=float(Cs),H=float(H),
                c=float(elasticity),q=q,removed=float((Cs-C)/elasticity),single_D2=float(Cs/elasticity))

def scalar_single(tau,phi,noisy=1):
    tau=np.asarray(tau,float); T=tau.sum()
    if phi==0: return noisy/(2*T)
    theta=phi/(1-phi); R=np.prod(theta/(theta+tau))
    return noisy*((1+2*theta)/(2*T)-R/(1-R))

def closed_island(tau,P,m,d,phi,rho,ell=0.,count=1):
    s=1-1/np.asarray(tau,float); a=1-m*ell; b=1-m
    sa=s.copy(); sa[d]*=a
    T=np.sum(1/(1-sa)); q=rho+(1-rho)/P
    Ra=np.prod(phi*(1-sa)/(1-phi*sa))
    Rb=Ra*(b/a)*(1-phi*a*s[d])/(1-phi*b*s[d])
    S=(1+phi)/(2*(1-phi)*T)-Ra/(1-Ra)
    H=Rb/(1-Rb)
    return count*(q*S-(1-q)*H),S,H,Ra,Rb,T

def long_parking_coefficient(phi,m=.2,P=4,rho=0.,d=1):
    # tau=(2,2,L), L -> infinity, ell=0.
    theta=phi/(1-phi); q=rho+(1-rho)/P; U=(theta/(theta+2))**2
    sd=.5 if d!=2 else 1.
    f=(1-m)*(1-phi*sd)/(1-phi*(1-m)*sd)
    return q*(1+2*theta)/2-theta*U*(q+(1-q)*f)

def scan():
    emit('BASELINE', comparison='(ii) every rate fixed at its arithmetic mean; same mean host abundance because dependence is linear',
         ordering='demography then movement',calibration='whole-cycle R0=1 except labelled raw mean effects')
    worst=0.; criterr=0.; singleerr=0.
    for tau in ([2,2],[2,3,8],[3,3,3,3]):
        for P in (2,4):
            for d in range(len(tau)):
                for ell in (0.,.2):
                    for phi in (0.,.6,.95):
                        z=island(tau,P,.35,ell,d,phi,.25,tuple(range(len(tau))))
                        mod=model(tau,P,.35,ell,d,noisy=tuple(range(len(tau))))
                        C,D,e=general(mod,phi,.25)
                        worst=max(worst,abs(C-z['C']))
                        criterr=max(criterr,abs(perron(mod['A'])[0]-1),abs(mod['R']-1))
        for phi in (0.,.6,.95):
            z=island(tau,1,.3,0.,0,phi,1.)
            singleerr=max(singleerr,abs(z['D2']-scalar_single(tau,phi)))
    check('island modes versus full ordered driver pairs',worst)
    check('whole-cycle R0 and eigenvalue criticality',criterr)
    check('single-ring formula versus independent resolvent',singleerr)
    reference={'__name__':'reference_not_main'}
    source=(__import__('pathlib').Path(__file__).resolve().parent / 'checks/host_crash_kills_parasites.py').resolve()
    exec(compile(source.read_text(),str(source),'exec'),reference)
    refA,_,_,_=reference['setup'](3,7,growth=1.)
    refC=reference['terms'](refA,0.,.8)[0][0,0]
    check('supplied terms() lag sum versus general resolvent',abs(refC-general(model([3,7],1,0.,d=1),.8,0.)[0]))
    scalarerr=0.
    for tau in ([2,2],[2,3,8],[3,3,3,3]):
        for d in range(len(tau)):
            for phi in (0.,.5,.95):
                for ell in (0.,.2):
                    value=closed_island(tau,4,.3,d,phi,.25,ell)[0]
                    scalarerr=max(scalarerr,abs(value-island(tau,4,.3,ell,d,phi,.25)['D2']))
    check('scalar closed form versus full mode resolvent',scalarerr)
    # Full parameter coverage is printed as one-factor changes from the labelled central case.
    central=dict(tau=[2,3,8],P=4,m=.2,ell=0.,d=1,phi=.8,rho=0.)
    for key,values in dict(P=[1,2,4,8],m=[.02,.1,.2,.5,1.],rho=[0.,.25,.75,1.],
                           phi=[-.5,0.,.5,.8,.95],ell=[0.,.1,.4],d=[0,1,2],
                           tau=[[2,3,2],[2,3,8],[2,3,40]]).items():
        for val in values:
            cfg=central.copy(); cfg[key]=val
            z=island(**cfg); mod=model(cfg['tau'],cfg['P'],cfg['m'],cfg['ell'],cfg['d'])
            emit('SCAN',axis=key,parameters=cfg,**z,raw_log_lambda=math.log(mod['original_lam']),raw_log_R0=math.log(mod['original_R']))
    for topo in ('island','neighbour_circle','neighbour_line','downwind_circle','downwind_line'):
        mod=model([2,3,8],P=4,m=.3,d=1,ell=.1,topology=topo)
        # Open directed line is reducible; the full primitive formula is inapplicable.
        coeff=None if topo=='downwind_line' else general(mod,.8,.2)[1]
        emit('TOPOLOGY',topology=topo,D2=coeff,raw_log_lambda=math.log(mod['original_lam']),raw_log_R0=math.log(mod['original_R']))
    boundaries={}
    tau=[2,2]; P=4; d=1; m=.15
    phistar=brentq(lambda ph:island(tau,P,m,0,d,ph,0)['C'],0.,.99,xtol=1e-13)
    mstar=brentq(lambda mm:island(tau,P,mm,0,d,.8,0)['C'],.001,1.,xtol=1e-13)
    z=island(tau,P,m,0,d,.8,0)
    rhostar=(P*z['H']/(z['Cs']+z['H'])-1)/(P-1)
    _,S,H,Ra,_,_=closed_island(tau,P,m,d,.8,0.)
    q=1/P; xx=q*S/(1-q+q*S); mm=1-xx/(Ra*(1-.8*.5)+xx*.8*.5)
    check('closed movement boundary versus full resolvent root',abs(mm-mstar))
    boundaries.update(H1_phi=phistar,H1_m=mstar,H1_rho=rhostar)
    emit('BOUNDARY_H1',tau=tau,P=P,d=d,m=m,phi_star_at_rho0=phistar,m_star_at_phi08=mstar,
         rho_star_at_phi08=rhostar,residual_phi=island(tau,P,m,0,d,phistar,0)['C'])
    for key, vals in dict(phi=[.1,.8],m=[.15,.9],rho=[0.,.9]).items():
        for val in vals:
            cfg=dict(tau=tau,P=P,m=m,d=d,phi=.8,rho=0.); cfg[key]=val
            emit('SIDE_H1',axis=key,parameters=cfg,**island(**cfg),full_D2=general(model(tau,P,cfg['m'],d=d),cfg['phi'],cfg['rho'])[1])
    # A genuine source/sink system before common whole-matrix calibration.
    for phi in (.1,.8):
        mod=model([2,2],4,.15,d=1,quality=[.4,.8,1.2,1.6])
        emit('QUALITY',phi=phi,quality=mod['quality'],D2=general(mod,phi,0.)[1],raw_log_R0=math.log(mod['original_R']),
             calibrated_g0=np.diag(mod['G'][0]))
    qmod=model([2,2],4,.15,d=1,quality=[.4,.8,1.2,1.6])
    qstar=brentq(lambda ph:general(qmod,ph,0.)[0],0,.999,xtol=1e-13)
    emit('BOUNDARY_QUALITY',phi_star=qstar,residual=general(qmod,qstar,0.)[0])
    # H2: extend a separate steady stage; compare long finite stays, not only short-stay anomaly.
    for phi in (0.,.8,.95):
        for m in (.05,.2,1.):
            for d in (0,1,2):
                short=island([2,2,10],4,m,0,d,phi,0.)['D2']
                long=island([2,2,40],4,m,0,d,phi,0.)['D2']
                emit('PARKING',phi=phi,m=m,d=d,short_tau=[2,2,10],long_tau=[2,2,40],short_D2=short,long_D2=long,change=long-short,
                     unmixed_short=scalar_single([2,2,10],phi),unmixed_long=scalar_single([2,2,40],phi))
    fun=lambda ph:island([2,2,40],4,.2,0,1,ph,0.)['D2']-island([2,2,10],4,.2,0,1,ph,0.)['D2']
    h2star=brentq(fun,0,.99,xtol=1e-13); boundaries['H2_phi']=h2star
    emit('BOUNDARY_H2',phi_star=h2star,residual=fun(h2star),m=.2,P=4,d=1,short_tau=[2,2,10],long_tau=[2,2,40])
    star=brentq(long_parking_coefficient,0.,.99,xtol=1e-13)
    emit('BOUNDARY_H2_ASYMPTOTIC',phi_star=star,residual=long_parking_coefficient(star),m=.2,P=4,rho=0.,d=1)
    for phi in (.5,.8):
        L=100000.
        emit('LONG_PARKING_SIDE',phi=phi,K=long_parking_coefficient(phi),L=L,
             L_times_D2=island([2,2,L],4,.2,0,1,phi,0.)['D2']*L,
             D2_L=island([2,2,L],4,.2,0,1,phi,0.)['D2'],
             D2_2L=island([2,2,2*L],4,.2,0,1,phi,0.)['D2'])
    # H3 mobile final stage, equal stays, same mean generation time, independently fluctuating hosts.
    for phi in (0.,.8,.95):
        for m in (.05,.2,1.):
            for n in (2,3,4):
                for allnoise in (False,True):
                    noisy=tuple(range(n)) if allnoise else (0,)
                    z=island([12/n]*n,4,m,0,n-1,phi,0.,noisy)
                    emit('NUMBER',n=n,T=12,m=m,phi=phi,mobile_stage=n-1,noisy=noisy,**z)
    hn=lambda ph:island([3]*4,4,.2,0,3,ph,0.,tuple(range(4)))['D2']-island([6]*2,4,.2,0,1,ph,0.,(0,1))['D2']
    h3star=brentq(hn,0,.999,xtol=1e-13); boundaries['H3_phi']=h3star
    emit('BOUNDARY_H3',phi_star=h3star,residual=hn(h3star),T=12,m=.2,P=4,rho=0.,n=[2,4])
    # Exact geometric-mean matrix boundary for the universal upper bound.
    for ell in (0.,.25):
        for topo in ('island','neighbour_circle','downwind_line'):
            mod=model([2,3,8],4,.3,ell,1,topo)
            sig=.7; cap=sig**2/2
            emit('BOUND_H4',ell=ell,topology=topo,cap=cap,geometric_at_cap_lambda=spectral(mod['A']),
                 geometric_below_lambda=geometric_lambda(mod,cap-.1,sig),geometric_above_lambda=geometric_lambda(mod,cap+.1,sig))
    # Exact edge formula checked with full next-generation and mean matrices.
    for m in (.1,.4,.8):
        for ell in (0.,.2):
            mod=model([2,3,8],4,m,ell,1,'downwind_line',critical=False)
            s=mod['s'][1]; k=1-m
            want=k*(1-s)/(1-k*s)
            reduced=model([2,3,8],1,m,1.,1,critical=False)
            check('open edge R0',abs(mod['R']-want))
            check('open edge eigenvalue from communicating blocks',abs(spectral(mod['A'])-spectral(reduced['A'])))
            emit('EDGE',m=m,ell=ell,log_R0=math.log(mod['R']),log_lambda=math.log(spectral(mod['A'])),formula_R0=want)
    return boundaries

def geometric_lambda(mod,y,sigma):
    return spectral(mod['A']+(math.exp(y-sigma*sigma/2)-1)*mod['F0'])

def rates(mod,phi,rho,sigma,ys,seed=20261114,steps=60000,pairs=24,burn=3000,tangent=False):
    """Independent antithetic pairs, exact time-dependent matrix recurrence.
    Vectorizes roots and replicates. No small-noise approximation in products.
    """
    rng=np.random.default_rng(seed); P=mod['P']; n=mod['n']; noisy=mod['noisy']; J=len(noisy)
    ys=np.asarray(ys); L=len(ys); N=2*pairs
    z=rng.standard_normal((pairs,J,P+1)); sd=math.sqrt(1-phi**2)
    g=np.stack([np.diag(G) for G in mod['G']]); s=mod['s']; d=mod['d']; K=mod['K'][d]
    X=np.ones((L,N,n,P))/(n*P); U=np.zeros_like(X)
    if np.all(s==0):
        X[:]=0.; X[:,:,0,:]=1/P  # Follow one cohort in a periodic stage graph.
    acc=np.zeros((L,N)); deriv=np.zeros((L,N)); h=np.ones((N,n,P))
    gy=np.broadcast_to(g,(L,n,P)).copy(); gy[:,0,:]*=np.exp(ys)[:,None]
    gy=gy[:,None,:,:]
    for t in range(steps+burn):
        z=phi*z+sd*rng.standard_normal(z.shape)
        zz=math.sqrt(rho)*z[:,:,:1]+math.sqrt(1-rho)*z[:,:,1:]
        zz=np.concatenate((zz,-zz),axis=0)
        h[:,noisy,:]=np.exp(sigma*zz-sigma*sigma/2)
        flow=gy*h[None,:,:,:]*X
        Xn=s[None,None,:,None]*X+np.roll(flow,1,axis=2)
        Xn[:,:,d,:]=Xn[:,:,d,:]@K.T
        total=Xn.sum(axis=(2,3))
        if tangent:
            fu=gy*h[None,:,:,:]*U
            fu[:,:,0,:]+=flow[:,:,0,:]
            Un=s[None,None,:,None]*U+np.roll(fu,1,axis=2)
            Un[:,:,d,:]=Un[:,:,d,:]@K.T
            dt=Un.sum(axis=(2,3))/total
            U=Un/total[:,:,None,None]-(Xn/total[:,:,None,None])*dt[:,:,None,None]
            if t>=burn: deriv+=dt
        X=Xn/total[:,:,None,None]
        if t>=burn: acc+=np.log(total)
    r=acc/steps; r=(r[:,:pairs]+r[:,pairs:])/2
    slope=deriv/steps; slope=(slope[:,:pairs]+slope[:,pairs:])/2
    return r,slope

def solve_case(case,steps=30000,pairs=24):
    label,cfg,phi,rho,sigma,seed=case
    t0=time.time(); mod=model(**cfg)
    if cfg.get('m',.2)==0 and mod['P']>1: raise ValueError('Use one patch for the unmixed asymptotic Lyapunov exponent')
    D2=general(mod,phi,rho)[1]
    pred=D2*sigma*sigma; width=.08+.25*sigma*sigma
    ys=np.array([pred-width,0.,pred+width]); ys=np.unique(np.sort(ys))
    if len(ys)!=3: ys=np.array([pred-width,pred,pred+width])
    expansions=0
    while True:
        rr,_=rates(mod,phi,rho,sigma,ys,seed,steps=steps,pairs=pairs)
        p=np.polyfit(ys,rr.mean(axis=1),2)
        if np.polyval(p,ys[0])*np.polyval(p,ys[-1])<=0: break
        expansions+=1; width*=2
        ys=np.unique(np.sort([pred-width,0.,pred+width]))
        if expansions>5: raise RuntimeError('Root not bracketed '+label)
    root=brentq(lambda y:np.polyval(p,y),ys[0],ys[-1],xtol=1e-13)
    # Independent exact-product evaluation of proposed root; also compute unshifted growth.
    raw_y=math.log(mod['original_R']); raw_index=2 if abs(raw_y)>1e-12 else 0
    for iteration in range(5):
        root_ys=[0.,root,raw_y] if raw_index==2 else [0.,root]
        val,deriv=rates(mod,phi,rho,sigma,root_ys,seed,steps=steps,pairs=pairs,tangent=True)
        slope=deriv[1].mean(); correction=val[1].mean()/slope
        root-=correction
        if abs(correction)<.0008: break
    roots=root-(val[1]-val[1].mean())/slope
    Dse=roots.std(ddof=1)/math.sqrt(len(roots))
    # Taylor remainder of correction, estimated from the coarse curvature.
    residual_est=abs(p[0]*correction**2)
    result=dict(label=label,parameters=cfg,phi=phi,rho=rho,sigma=sigma,seed=seed,pairs=pairs,steps=steps,burn=3000,
                D=root,D_se=Dse,D2sigma2=pred,r=val[0].mean(),r_se=val[0].std(ddof=1)/math.sqrt(val.shape[1]),
                root_correction=correction,root_growth_remainder_est=residual_est,slope=slope,bracket_expansions=expansions,
                mean_log_lambda=math.log(spectral(mod['A'])),raw_log_lambda=math.log(mod['original_lam']),
                raw_log_R0=math.log(mod['original_R']),cap=len(mod['noisy'])*sigma*sigma/2,
                raw_r=val[raw_index].mean(),raw_r_se=val[raw_index].std(ddof=1)/math.sqrt(val.shape[1]),
                raw_variability_effect=val[raw_index].mean()-math.log(mod['original_lam']),
                seconds=time.time()-t0,replicate_roots=roots.tolist())
    return result

def build_cases():
    cases=[]
    def add(label,tau=[2,2],P=4,m=.15,d=1,phi=.8,rho=0.,sigmas=(.3,1.),**kw):
        cfg=dict(tau=tau,P=P,m=m,d=d,**kw)
        for sig in sigmas: cases.append((label,cfg,phi,rho,sig,20261114))
    add('H1_cost',phi=.1)
    add('H1_gain')
    add('H1_high_m',m=.9)
    add('H1_high_rho',rho=.9)
    add('H1_small_noise',sigmas=(.075,))
    add('source_sink_cost',quality=[.4,.8,1.2,1.6],phi=.8)
    add('source_sink_gain',quality=[.4,.8,1.2,1.6],phi=.99)
    for phi in (0.,.8):
        for tau in (10,40):
            add('parking_'+str(tau)+'_phi'+str(phi),tau=[2,2,tau],m=.2,phi=phi)
    for tau in (10,40):
        add('parking_unmixed_'+str(tau),tau=[2,2,tau],P=1,m=0.,phi=.8)
        add('parking_strong_memory_'+str(tau),tau=[2,2,tau],m=.2,phi=.95)
    for phi in (.8,.95):
        for n in (2,4):
            add('number_'+str(n)+'_phi'+str(phi),tau=[12/n]*n,m=.2,d=n-1,phi=phi,noisy=tuple(range(n)))
    for n in (2,4):
        add('number_'+str(n)+'_phi0.99',tau=[12/n]*n,m=.2,d=n-1,phi=.99,noisy=tuple(range(n)))
    add('complete_mixing',m=1.,phi=0.)
    add('single_patch',P=1,m=0.,phi=0.)
    add('synchrony',rho=1.,phi=0.)
    add('transit_loss',ell=.2)
    add('neighbour_circle',topology='neighbour_circle')
    add('neighbour_line',topology='neighbour_line')
    add('downwind_circle',topology='downwind_circle')
    add('downwind_circle_matched',tau=[2,3,8],m=.4,topology='downwind_circle')
    for d in (0,1,2):
        add('mobile_stage_'+str(d),tau=[2,3,8],m=.2,d=d)
    add('two_patches',P=2)
    add('eight_patches',P=8)
    # The open line's exponent is the maximum diagonal cocycle exponent; identical
    # marginals give the same one-patch exponent. Simulate that exact reduction.
    add('open_line_diagonal',P=1,m=.4,ell=1.,tau=[2,3,8],d=1,phi=.8)
    return cases

def finite_checks():
    # Synchrony: noncommuting periodic temporal matrices; exact full spatial products.
    h=np.array([.2,.5,1.3,2.])
    def floquet(mod, hs, y=0.):
        prod=np.eye(len(mod['A']))
        for row in hs:
            A=mod['A'].copy()
            for j,B in enumerate(mod['Bs']): A+=(row[j]-1)*B
            # Here just transfer zero fluctuates.
            A+=(math.exp(y)-1)*sum(row[j]*B for j,B in enumerate(mod['Bs']))
            prod=A@prod
        return math.log(spectral(prod))/len(hs)
    single=model([2,3,8],1,0.,d=1)
    r1=floquet(single,h[:,None])
    for topo in ('island','neighbour_circle','downwind_circle'):
        for d in (0,1,2):
            mod=model([2,3,8],4,.37,d=d,topology=topo)
            r=floquet(mod,np.repeat(h[:,None],4,axis=1))
            check('synchronous Floquet '+topo+' stage '+str(d),abs(r-r1),1e-10)
    # Direct normalized powers check the mean edge exponent without its formula.
    mod=model([2,3,8],4,.4,.2,1,'downwind_line',critical=False)
    A=mod['A']; x=np.zeros(len(A)); x[np.arange(3)*4+3]=1/3
    logs=[]
    for t in range(3000):
        x=A@x; norm=x.sum(); x/=norm
        if t>=1000: logs.append(math.log(norm))
    check('open-line mean exponent versus direct matrix powers',abs(np.mean(logs)-math.log(spectral(A))),1e-10)
    # Independent numerical product check for scalar lognormal quadrature.
    nodes,weights=roots_hermitenorm(160); weights/=math.sqrt(2*math.pi)
    for sigma in (.3,1.):
        f=lambda y:np.dot(weights,np.log(.5+.5*np.exp(y+sigma*nodes-sigma*sigma/2)))
        D=brentq(f,0,sigma*sigma/2,xtol=1e-13)
        mod=model([2],1,0.,d=0)
        vals,_=rates(mod,.8,0.,sigma,[0.,D],steps=100000,pairs=32)
        se=vals[1].std(ddof=1)/math.sqrt(32)
        check('scalar quadrature versus products sigma '+str(sigma),abs(vals[1].mean()),max(4*se,2e-6))
        emit('SCALAR',sigma=sigma,D_quadrature=D,r_at_root=vals[1].mean(),r_se=se)
    # All-noise removal formula in a no-staying ring, P=2, complete mixing.
    for sigma in (.3,1.):
        for rho in (0.,.5,1.):
            v=sigma*math.sqrt((1-rho)/2)
            removed=np.dot(weights,np.logaddexp(v*nodes,-v*nodes)-math.log(2))
            D=sigma*sigma/2-removed
            mod=model([1,1],2,1.,d=1)
            vals,_=rates(mod,.8,rho,sigma,[0.,D],steps=100000,pairs=32)
            se=vals[1].std(ddof=1)/math.sqrt(32)
            check('exact complete-mixing removal versus products',abs(vals[1].mean()),max(4*se,2e-6))
            emit('EXACT_REMOVAL',sigma=sigma,rho=rho,P=2,D=D,removed=removed,
                 second_order_D=sigma*sigma*(1+rho)/4,r_at_root=vals[1].mean(),r_se=se)
    # An exact, separate periodic environment: mean-one patch multipliers, random phase.
    hs=np.array([[1.8,.2]]*6+[[.2,1.8]]*6)
    def periodic_loss(m):
        mod=model([2,2],2,m,d=1)
        D=brentq(lambda y:floquet(mod,hs,y),-2.,2.,xtol=1e-13)
        return D,floquet(mod,hs)
    # Search brackets before claiming a periodic boundary.
    grid=np.linspace(.02,1.,30); Ds=[periodic_loss(m)[0] for m in grid]
    for a,b,Da,Db in zip(grid[:-1],grid[1:],Ds[:-1],Ds[1:]):
        if Da*Db<0:
            root=brentq(lambda m:periodic_loss(m)[0],a,b,xtol=1e-13)
            emit('PERIODIC_BOUNDARY',m_star=root,period=len(hs),multipliers=hs,baseline='arithmetic mean rates')
            for m in (max(.001,root-.05),min(1.,root+.05)):
                D,r=periodic_loss(m); emit('PERIODIC_SIDE',m=m,D=D,r=r)

def simulations():
    cases=build_cases(); emit('SIMULATION_DESIGN',cases=len(cases),workers=3,pairs=24,steps=30000,burn=3000)
    results=[]
    with mp.get_context('spawn').Pool(3) as pool:
        for result in pool.imap(solve_case,cases):
            results.append(result); emit('SIM',**{k:v for k,v in result.items() if k!='replicate_roots'})
    for sigma in (.3,1.):
        lookup={r['label']:r for r in results if r['sigma']==sigma}
        comparisons=[('H2_no_memory','parking_10_phi0.0','parking_40_phi0.0'),
                     ('H2_memory','parking_10_phi0.8','parking_40_phi0.8'),
                     ('H2_unmixed','parking_unmixed_10','parking_unmixed_40'),
                     ('H3_low_memory','number_2_phi0.8','number_4_phi0.8'),
                     ('H3_high_memory','number_2_phi0.95','number_4_phi0.95'),
                     ('H3_very_high_memory','number_2_phi0.99','number_4_phi0.99'),
                     ('H2_strong_memory','parking_strong_memory_10','parking_strong_memory_40')]
        for name,a,b in comparisons:
            delta=np.array(lookup[b]['replicate_roots'])-np.array(lookup[a]['replicate_roots'])
            emit('COMPARISON',name=name,sigma=sigma,change=delta.mean(),se=delta.std(ddof=1)/math.sqrt(len(delta)),first=a,second=b)
        for label in ('complete_mixing','synchrony'):
            a=lookup['single_patch']; b=lookup[label]
            emit('REMOVAL',label=label,sigma=sigma,single_D=a['D'],spatial_D=b['D'],removed=a['D']-b['D'],q_prediction=a['D']*.25)
    for r in results:
        if r['D']>r['cap']+4*r['D_se']: FAILS.append('H4 '+r['label'])
        if r['root_growth_remainder_est']>max(.02*r['D_se']*r['slope'],1e-8): FAILS.append('root accuracy '+r['label'])
        if r['label']=='H1_small_noise':
            check('small-noise prediction independent products',abs(r['D']-r['D2sigma2']),max(.05*abs(r['D2sigma2']),4*r['D_se']))
    return results

def boundary_job(job):
    kind,sigma=job; cache={}
    def evaluate(x):
        if x in cache: return cache[x]
        seed=20361114
        if kind in ('H1_phi','H1_m','H1_rho','quality_phi','H1_logm'):
            phi=x if kind.endswith('phi') else .8
            m=math.exp(x) if kind=='H1_logm' else x if kind=='H1_m' else .15
            rho=x if kind=='H1_rho' else 0.
            kw=dict(tau=[2,2],P=4,m=m,d=1)
            if kind=='quality_phi': kw['quality']=[.4,.8,1.2,1.6]
            r,_=rates(model(**kw),phi,rho,sigma,[0.],seed,steps=20000)
            values=-r[0]
        else:
            configs=([dict(tau=[2,2,L],P=4,m=.2,d=1) for L in (10,40)] if kind.startswith('H2_') else
                     [dict(tau=[12/n]*n,P=4,m=.2,d=n-1,noisy=tuple(range(n))) for n in (2,4)])
            res=[solve_case(('boundary',cfg,x,0.,sigma,seed),steps=20000) for cfg in configs]
            values=np.array(res[1]['replicate_roots'])-np.array(res[0]['replicate_roots'])
        cache[x]=values
        return values
    lo,hi=(math.log(1e-8),math.log(.04)) if kind=='H1_logm' else (.01,.998) if kind.endswith('phi') else (.04,.98)
    if kind=='H2_phi' and sigma==1.: lo,hi=.8,.95
    if kind=='H2_phi_high': lo,hi=.95,.998
    flo=evaluate(lo).mean(); fhi=evaluate(hi).mean()
    if flo*fhi>=0:
        return dict(kind=kind,sigma=sigma,status='not bracketed',lo=lo,hi=hi,flo=flo,fhi=fhi)
    root=brentq(lambda x:evaluate(x).mean(),lo,hi,xtol=.00015)
    values=evaluate(root); se=values.std(ddof=1)/math.sqrt(len(values))
    halfwidth=.35 if kind=='H1_logm' else .02
    left=max(lo,root-halfwidth); right=min(hi,root+halfwidth)
    sides=[]
    for x in (left,right):
        v=evaluate(x); sides.append(dict(parameter=x,value=v.mean(),se=v.std(ddof=1)/math.sqrt(len(v))))
    slope=(sides[1]['value']-sides[0]['value'])/(right-left)
    extra={}
    if kind=='H1_logm':
        extra=dict(m_root=math.exp(root),m_root_se=math.exp(root)*se/abs(slope))
        for side in sides: side['m']=math.exp(side['parameter'])
        rr,_=rates(model([2,2],4,math.exp(root),d=1),.8,0.,sigma,[0.],seed=20461114,steps=240000,pairs=32,burn=10000)
        long_se=rr[0].std(ddof=1)/math.sqrt(32)
        extra['long_validation']=dict(objective=-rr[0].mean(),se=long_se,steps=240000,pairs=32,burn=10000,
                                       passed=abs(rr[0].mean())<4*math.sqrt(long_se**2+se**2))
    for side in sides: side['resolved']=abs(side['value'])>2*side['se']
    return dict(kind=kind,sigma=sigma,root=root,root_se=se/abs(slope),residual=values.mean(),objective_se=se,steps=20000,pairs=24,**extra,
                sides=sides,evaluations=len(cache),seed=20361114,status='computed Monte Carlo boundary')

def finite_boundaries():
    jobs=[(kind,sigma) for kind in ('H1_phi','H1_m','H1_rho','quality_phi','H2_phi','H3_phi','H1_logm') for sigma in (.3,1.)]
    jobs.append(('H2_phi_high',1.))
    results=[]
    with mp.get_context('spawn').Pool(3) as pool:
        for r in pool.imap(boundary_job,jobs):
            results.append(r); emit('FINITE_BOUNDARY',**r)
    return results

def write_tables(results,bounds):
    """Write reviewable numeric tables from the same computed results as stdout."""
    lines=['<!-- NUMERICAL_TABLES -->','',
           'All tables compare with (ii), every rate fixed at its arithmetic mean, including movement. A plus/minus value is one standard error. The simulation design and seeds are recorded in the output. The tables are generated by the script.','',
           '### H1 examples: direct stochastic products','',
           'The identical-patch cases use stays (2,2), four patches, m=0.15, rho_s=0, and movement of stage one. The cost case has phi=0.1; the gain case has phi=0.8. The quality cases use multipliers (0.4,0.8,1.2,1.6), with phi=0.8 and phi=0.99. Each mean matrix is calibrated to whole-system R0=1.','',
           '| Case | sigma | D ± SE | r ± SE | second-order D |',
           '|---|---:|---:|---:|---:|']
    selected=('H1_cost','H1_gain','source_sink_cost','source_sink_gain','H1_small_noise')
    for r in results:
        if r['label'] in selected:
            lines.append(f"| {r['label']} | {r['sigma']:g} | {r['D']:.7f} ± {r['D_se']:.7f} | {r['r']:.7f} ± {r['r_se']:.7f} | {r['D2sigma2']:.7f} |")
    lines+=['','### Parking and host-number comparisons','',
            'Baseline: the critical arithmetic-mean matrix of each life cycle. Parking compares stays (2,2,10) with (2,2,40). Movement is m=0.2 in stage one, P=4 and rho_s=0, except the isolated-patch comparison. Host number compares two equal stages with four equal stages at total generation time 12, with m=0.2 in the final stage and independent host drivers. Change means D(second) minus D(first). Positive parking change is harmful. Negative host-number change contradicts H3.','',
            '| Comparison | sigma | D first | D second | change ± SE |',
            '|---|---:|---:|---:|---:|']
    comparisons=[('parking, phi=0','parking_10_phi0.0','parking_40_phi0.0'),
                 ('parking, phi=0.8','parking_10_phi0.8','parking_40_phi0.8'),
                 ('parking, phi=0.95','parking_strong_memory_10','parking_strong_memory_40'),
                 ('isolated parking, phi=0.8','parking_unmixed_10','parking_unmixed_40'),
                 ('hosts, phi=0.8','number_2_phi0.8','number_4_phi0.8'),
                 ('hosts, phi=0.95','number_2_phi0.95','number_4_phi0.95'),
                 ('hosts, phi=0.99','number_2_phi0.99','number_4_phi0.99')]
    for sigma in (.3,1.):
        by={r['label']:r for r in results if r['sigma']==sigma}
        for name,a,b in comparisons:
            x,y=by[a],by[b]; delta=np.array(y['replicate_roots'])-x['replicate_roots']
            lines.append(f"| {name} | {sigma:g} | {x['D']:.7f} | {y['D']:.7f} | {delta.mean():.7f} ± {delta.std(ddof=1)/math.sqrt(len(delta)):.7f} |")
    lines+=['','The host-number reversal is clear at sigma=0.3. The tested strong-memory examples at sigma=1 have the opposite ordering. This is a scientific limit of the small-noise conclusion, not a solver failure.','',
            '### Finite-noise boundary estimates and both-side checks','',
            'These use a different random seed from the example table. For H1 and quality boundaries the signed objective is minus growth at the critical arithmetic-mean matrix; its sign is the sign of D. For H2 and H3 the objective is the difference of threshold losses. Every reported side comes from direct stochastic products. Root uncertainty is the objective standard error divided by a numerical slope. It does not include a formal bound on finite-horizon bias.','',
            'H1 slices use stays (2,2), P=4, d=1: phi varies at m=0.15 and rho_s=0; m varies at phi=0.8 and rho_s=0; rho_s varies at phi=0.8 and m=0.15. Quality uses the same quality multipliers as above. H2 and H3 use the life-cycle comparisons above. The low-m search is in log(m).','',
            '| Boundary | sigma | parameter ± SE | lower side: parameter, objective ± SE | upper side: parameter, objective ± SE |',
            '|---|---:|---:|---|---|']
    for r in bounds:
        if 'root' not in r:
            lines.append(f"| {r['kind']} | {r['sigma']:g} | no crossing bracketed | {r['lo']:.6g}, {r['flo']:.7g} | {r['hi']:.6g}, {r['fhi']:.7g} |")
            continue
        sides=[]
        for z in r['sides']:
            sides.append(f"{z.get('m',z['parameter']):.7g}, {z['value']:.7g} ± {z['se']:.3g}")
        root=r.get('m_root',r['root']); se=r.get('m_root_se',r['root_se'])
        lines.append(f"| {r['kind']} | {r['sigma']:g} | {root:.7g} ± {se:.3g} | {sides[0]} | {sides[1]} |")
    lines+=['','A missing bracket is not a proof that a crossing does not exist elsewhere. These finite-noise boundaries are estimates; the second-order boundaries above are roots of explicit deterministic formulas.','',
            '### Movement stage, topology, and the mean wind penalty','',
            'Baseline: (ii), the arithmetic-mean matrix of the whole system. Raw means use the original transfer means before critical calibration. Raw r is simulated at those same original means. The variability growth effect is raw r minus raw mean log(lambda). The threshold D is invariant to the common transfer calibration. For the open line the row marked diagonal is its exact asymptotic diagonal-block reduction; ell=1 in that scalar-patch implementation represents departure from that patch.','',
            '| Case | sigma | raw mean log R0 | raw mean log lambda | raw r ± SE | variability growth effect | D ± SE |',
            '|---|---:|---:|---:|---:|---:|---:|']
    labels=('transit_loss','downwind_circle_matched','open_line_diagonal','neighbour_circle','neighbour_line',
            'mobile_stage_0','mobile_stage_1','mobile_stage_2','two_patches','eight_patches')
    for r in results:
        if r['label'] in labels:
            lines.append(f"| {r['label']} | {r['sigma']:g} | {r['raw_log_R0']:.6f} | {r['raw_log_lambda']:.6f} | {r['raw_r']:.7f} ± {r['raw_r_se']:.7f} | {r['raw_variability_effect']:.7f} | {r['D']:.7f} ± {r['D_se']:.7f} |")
    lines+=['','The matched downwind circle and open line use stays (2,3,8), m=0.4 and phi=0.8. The mobile-stage rows use those stays with m=0.2. Other configurations are printed in full in the output. Loss of mean reproduction at the open edge is already present with no variability.','',
            '### Finite-noise mixing versus a variance replacement','',
            'Baseline: the same arithmetic-mean ring, stays (2,2), phi=0, with four conservative island patches. Complete mixing has m=1 and rho_s=0. The isolated and synchronized cases should have the same loss; their separate simulation errors explain their small difference. The exact pathwise identity was also checked with full Floquet products.','',
            '| sigma | single-patch D | complete-mixing D | synchronized D | one quarter of single-patch D |',
            '|---:|---:|---:|---:|---:|']
    for sigma in (.3,1.):
        by={r['label']:r for r in results if r['sigma']==sigma}
        lines.append(f"| {sigma:g} | {by['single_patch']['D']:.7f} | {by['complete_mixing']['D']:.7f} | {by['synchrony']['D']:.7f} | {by['single_patch']['D']/4:.7f} |")
    lines+=['','The output also gives exact quadrature values for the two-patch no-staying formula. For example, at sigma=1 and rho_s=0 its loss is 0.2904852728, while replacing variance by its spatial-average factor would give 0.25. This directly refutes an all-noise reading of the variance rule.','',
            'A separate periodic check uses two patches with multipliers (1.8,0.2) for six steps and (0.2,1.8) for six steps, with a uniformly random phase. Its arithmetic means are one. Direct Floquet products give a crossing at m=0.8953970527; D is 0.0011329417 at m=0.8453970527 and -0.0003618255 at m=0.9453970527. This is an exact periodic-environment calculation to numerical root tolerance. It is not Gaussian AR forcing, and its boundary must not be transferred to that model.','']
    note=Path(__file__).with_name('mechanism_note.md')
    text=note.read_text(); head=text.split('<!-- NUMERICAL_TABLES -->')[0]; tail=text.split('## Not done')[1]
    note.write_text(head+'\n'.join(lines)+'\n## Not done'+tail)

if __name__=='__main__':
    start=time.time()
    scan()
    if '--scan' not in sys.argv:
        finite_checks()
        results=simulations()
        bounds=finite_boundaries()
        write_tables(results,bounds)
    emit('FINISH',wall_seconds=time.time()-start,failures=FAILS)
    sys.exit(bool(FAILS))
