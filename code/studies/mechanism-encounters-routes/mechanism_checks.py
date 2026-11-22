#!/usr/bin/env python3
"""Offline, reproducible environmental-encounter and route checks.

Run: python3 -I mechanism_checks.py > mechanism_checks_output.txt
Only numpy/scipy and the Python standard library. No imported project code.
Matrices act on column population vectors. All model constructors are critical
at their mean. A birth parameter multiplies returns to stage zero.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time
import math
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm

REPS, STEPS, BURN = 12, 40000, 2000
SEED = 20261114
N, E = 4, 4


@dataclass
class Model:
    name: str
    A: np.ndarray
    B: np.ndarray       # individual random contributions, including signed stays
    R: np.ndarray       # birth part of mean A
    RB: np.ndarray      # birth part of each random contribution
    kap: np.ndarray
    corr: np.ndarray
    phi: np.ndarray
    kind: str = 'gaussian'
    sigma: float = 0.3


def perron(A):
    vals, vec = np.linalg.eig(A)
    j = np.argmax(vals.real)
    lam = float(vals[j].real)
    v = np.abs(vec[:, j].real); v /= v.sum()
    vals, vec = np.linalg.eig(A.T)
    w = np.abs(vec[:, np.argmax(vals.real)].real); w /= w @ v
    return lam, v, w


def model(name, A, contributions, R, rb=None, kap=None, phi=0., corr=None,
          kind='gaussian', sigma=.3):
    d = len(contributions)
    B = np.array(contributions)
    if rb is None:
        rb = B * (R != 0)[None, :, :]
    return Model(name, np.array(A), B, np.array(R), np.array(rb),
                 np.ones(d) if kap is None else np.array(kap, float),
                 np.eye(d) if corr is None else np.array(corr, float),
                 np.full(d, phi) if np.isscalar(phi) else np.array(phi),
                 kind, sigma)


def ring(tau, active=None, phi=0., signs=None, shared=False, sigma=.3,
         name='ring'):
    tau = np.array(tau, float); p = 1/tau; n = len(tau)
    A = np.diag(1-p); R = np.zeros((n,n)); bs=[]
    if active is None: active = list(range(n))
    for k in range(n):
        b = np.zeros_like(A); b[(k+1)%n,k] = p[k]
        A += b
        if k == n-1: R += b
        bs.append(b if k in active else np.zeros_like(b))
    corr = np.ones((len(bs),len(bs))) if shared else np.eye(len(bs))
    return model(name,A,bs,R,kap=signs,phi=phi,corr=corr,sigma=sigma)


def contact(b=2., d=.1, phi=0., sigma=.3, gain=False):
    p=.2; c=(1-d)*p; mu=d*p
    # Similarity coordinates give A offdiagonals p and 1/b. Physical
    # entry is c, steady reproductive return is p/c times 1/b.
    A=np.array([[1-p,1/b],[p,1-1/b]])
    B=np.array([[0.,0.],[p,0.]])
    if not gain: B[0,0]=-c
    R=np.array([[0.,1/b],[0.,0.]])
    m=model(f"{'gain-bounded' if gain else 'contact'} b={b:g} d={d:g} phi={phi:g}",
            A,[B],R,phi=phi,kind='bounded',sigma=sigma)
    if not gain:
        hmax=1+2*np.sqrt(np.expm1(sigma*sigma))
        assert 0 <= 1-mu-c*hmax <= 1
    return m


def parallel(f=.4, newtau=1., newkap=1., rho=0., phi=0., sigma=.3):
    tau0=2.; t1=8.; p0=1/tau0; p1=1/t1; p2=1/newtau
    A=np.diag([1-p0,1-p1,1-p2]); R=np.zeros((3,3))
    A[1,0]=p0*(1-f); A[2,0]=p0*f
    A[0,1]=p1; A[0,2]=p2; R[0]=A[0]; R[0,0]=0
    b1=np.zeros_like(A); b1[1,0]=A[1,0]
    b2=np.zeros_like(A); b2[2,0]=A[2,0]
    return model(f"parallel f={f:g} tau2={newtau:g} k2={newkap:g} rho={rho:g} phi={phi:g}",
                 A,[b1,b2],R,kap=[1,newkap],corr=[[1,rho],[rho,1]],phi=phi,sigma=sigma)


def optional(f=.5,q=1.,tp=8.,phi=0.,sigma=.3):
    p=.5; pp=1/tp; Y=1-f+f*q
    A=np.diag([1-p,1-p,1-pp]); A[1,0]=p
    A[0,1]=(1-f)*p/Y; A[2,1]=f*p; A[0,2]=q*pp/Y
    R=np.zeros_like(A); R[0,1:]=A[0,1:]
    B=np.zeros_like(A); B[1,0]=p
    return model(f"optional q={q:g} phi={phi:g}",A,[B],R,phi=phi,sigma=sigma)


def bypass(f=.5,q=1.,phi=0.,sigma=.3):
    # Central exits: noisy obligatory-host route and direct reproductive route.
    p0=.5; p1=.25; Y=1-f+f*q
    A=np.diag([1-p0,1-p1]); A[1,0]=(1-f)*p0
    R=np.zeros_like(A); R[0,1]=p1/Y; R[0,0]=f*q*p0/Y
    A+=R; B=np.zeros_like(A); B[1,0]=A[1,0]
    return model(f"bypass f={f:g} q={q:g} phi={phi:g}",A,[B],R,phi=phi,sigma=sigma)


def coeff(m):
    """General finite-resolvent coefficient. Not used by product recursion."""
    lam,v,w=perron(m.A)
    assert abs(lam-1)<1e-10, (m.name,lam)
    Q=np.eye(len(v))-np.outer(v,w); T=m.A@Q
    B=m.B*m.kap[:,None,None]; c=np.einsum('i,eij,j->e',w,B,v)
    birth=float(w@m.R@v)
    C=0.
    for i in range(len(B)):
        Z=np.linalg.solve(np.eye(len(v))-m.phi[i]*T,Q)
        for j in range(len(B)):
            C+=m.corr[i,j]*(c[i]*c[j]/2-m.phi[i]*(w@B[i]@Z@B[j]@v))
    return float(C/birth), 1/birth


def kgain(tau,phi):
    G=sum(tau)
    if phi==0: return 1/(2*G)
    t=phi/(phi+(1-phi)*np.array(tau))
    P=np.prod(t)
    return (1+phi)/(2*(1-phi)*G)-P/(1-P)


def kcontact(tau,d,phi,k=0):
    Kg=kgain(tau,phi)
    if phi==0: return d*d*Kg
    t=phi/(phi+(1-phi)*np.array(tau)); P=np.prod(t)
    L=(t[k]-P)/(1-P)
    return d*d*Kg+d*(1-d)*L


def parking_deriv(a,b,d,phi):
    th=phi/(1-phi); tk=th/(th+a); U=th*tk; alpha=(1-d)*tk
    return -d*d*(th+.5)/(a+b)**2+d*U*(1-alpha)/(b+th-U)**2


def route_closed(m):
    """Renewal closed forms for the explicitly named networks."""
    K,G=coeff(m); ph=m.phi[0]
    AA=(1+ph)/(2*(1-ph))
    if m.name.startswith('parallel '):
        p0=1-m.A[0,0]; weights=m.A[1:,0]/p0
        tau=1/(1-np.diag(m.A)); t=ph/(ph+(1-ph)*tau)
        P=t[0]*np.dot(weights,t[1:]); u=weights*m.kap
        V=u@m.corr@u
        value=AA*V/G-t[0]*np.sum((u@m.corr)*u*t[1:])/(1-P)
    elif m.name.startswith('optional '):
        q=float(m.name.split('q=')[1].split()[0]); alpha=q/(1+q)
        t=ph/(ph+(1-ph)*np.array([2.,2.,8.]))
        P=t[0]*t[1]*(1-alpha+alpha*t[2])
        value=AA/G-P/(1-P)
    elif m.name.startswith('bypass '):
        f=float(m.name.split('f=')[1].split()[0]); q=float(m.name.split('q=')[1].split()[0])
        alpha=(1-f)/(1-f+f*q)
        t=ph/(ph+(1-ph)*np.array([2.,4.]))
        P=t[0]*(1-alpha+alpha*t[1])
        value=alpha*alpha*(AA/G-t[0]*t[1]/(1-P))
    else:raise ValueError(m.name)
    return value,K


def scalar_parallel(f,sigma,k=3.,rho=0.,nodes=100):
    z,w=roots_hermitenorm(nodes); w/=np.sqrt(2*np.pi)
    z1=z[:,None]; z2=rho*z1+np.sqrt(1-rho*rho)*z[None,:]
    h=(1-f)*np.exp(sigma*z1-sigma*sigma/2)+f*np.exp(k*sigma*z2-k*k*sigma*sigma/2)
    ww=w[:,None]*w[None,:]
    return brentq(lambda y:np.sum(ww*np.log(.75+.25*np.exp(y)*h)),-10,20,xtol=2e-13)


def scalar_bypass(f,q,sigma,nodes=160):
    z,w=roots_hermitenorm(nodes); w/=np.sqrt(2*np.pi)
    Y=1-f+f*q; alpha=(1-f)/Y
    h=1-alpha+alpha*np.exp(sigma*z-sigma*sigma/2)
    D=brentq(lambda y:w@np.log(.5+.5*np.exp(y)*h),-10,20,xtol=2e-13)
    return D,np.log(Y)+scalar_loss(.5,sigma)-D


def extra_checks():
    print('ADDITIONAL EXACT AND RENEWAL CHECKS')
    errs=[]
    for ph in (0.,.8):
        for m in (parallel(.4,1.,phi=ph),parallel(.1,8.,3.,.8,ph),optional(q=.8,phi=ph),optional(phi=ph),bypass(.5,.5,ph),bypass(.75,1.,ph)):
            closed,K=route_closed(m); errs.append(abs(closed-K))
            print(f'route_closed {m.name}: renewal={closed:.12g} resolvent={K:.12g}')
    print(f'route_algebra_max_error={max(errs):.4g} PASS={max(errs)<1e-9}')
    for sig in (.3,1.):
        old=scalar_loss(.75,sig)
        fstar=brentq(lambda f:scalar_parallel(f,sig)-old,1e-6,.999999,xtol=1e-11)
        rootcheck=scalar_parallel(fstar,sig,nodes=200)-scalar_parallel(0,sig,nodes=200)
        print(f'parallel_scalar_finite_boundary sigma={sig:g} fstar={fstar:.12g} local_fstar=.2 quadrature_residual={rootcheck:.3g}')
        for f in (.8*fstar,min(1,1.2*fstar)):
            D=scalar_parallel(f,sig)
            print(f'parallel_scalar_side sigma={sig:g} f={f:.12g} D={D:.12g} oldD={old:.12g} difference={D-old:.12g}')
        qstar=brentq(lambda q:scalar_bypass(.5,q,sig)[1],.001,1.,xtol=1e-12)
        print(f'bypass_scalar_finite_yield_boundary sigma={sig:g} qstar={qstar:.12g} residual={scalar_bypass(.5,qstar,sig,320)[1]:.3g}')
        for q in (qstar-.01,min(1.,qstar+.01)):
            D,adv=scalar_bypass(.5,q,sig)
            print(f'bypass_scalar_side sigma={sig:g} q={q:.12g} D={D:.12g} advantage_fixed_rates={adv:.12g}')
    fun=lambda a:3*kgain([a,(20-a)/2,(20-a)/2],.9)-2*kgain([10,10],.9)
    astar=brentq(fun,20/3,18,xtol=1e-12)
    print(f'H3_local_repartition_boundary phi=.9 astar={astar:.12g}')
    for a in (astar-.3,astar+.3):print(f'H3_local_a_side a={a:.12g} difference={fun(a):.12g}')
    fun=lambda ph:3*kgain([18,1,1],ph)-2*kgain([10,10],ph)
    phstar=brentq(fun,.1,.99,xtol=1e-12)
    print(f'H3_local_memory_boundary phistar={phstar:.12g}')
    for ph in (phstar-.03,phstar+.03):print(f'H3_local_phi_side phi={ph:.12g} difference={fun(ph):.12g}')


def floquet(m,amp=.5,period=2,curve='linear',scale=1.):
    # Fresh matrix construction. For even period: half high, half low.
    def growth(y):
        P=np.eye(len(m.A))
        for t in range(period):
            z=1 if t<period//2 else -1
            if curve=='linear': h=1+amp*m.kap*z
            else: h=np.exp(amp*m.kap*z)/np.cosh(amp*m.kap)
            A=m.A+np.einsum('e,eij->ij',h-1,m.B)
            R=m.R+np.einsum('e,eij->ij',h-1,m.RB)
            A=A+(np.exp(y)*scale-1)*R
            assert A.min()>-1e-12,(m.name,A)
            P=A@P
        return np.log(max(np.linalg.eigvals(P).real))/period
    y=brentq(growth,-8,8,xtol=2e-13)
    return y,growth(y)


def scalar_loss(s,sigma,eta=1.,nodes=160):
    z,w=roots_hermitenorm(nodes); w/=np.sqrt(2*np.pi)
    h=np.exp(eta*sigma*z-(eta*sigma)**2/2)
    return brentq(lambda y: w@np.log(s+(1-s)*np.exp(y)*h),-30,30,xtol=1e-13)


def curvature(kind,x,sigma,nodes=160):
    z,w=roots_hermitenorm(nodes); w/=np.sqrt(2*np.pi)
    H=np.exp(sigma*z-sigma*sigma/2)
    if kind=='II':
        F=lambda H: x*H/(1+x*H)
        eta=1/(1+x); chi=-2*x/(1+x)**2
    elif kind=='III':
        F=lambda H: (x*H)**2/(1+(x*H)**2)
        eta=2/(1+x*x); chi=2*(1-3*x*x)/(1+x*x)**2
    else:
        F=lambda H:H**x
        eta=x; chi=x*(x-1)
    h=F(H)/F(1.); mean=w@h
    y=brentq(lambda y:w@np.log(.75+.25*np.exp(y)*h),-30,30,xtol=1e-13)
    return y,y+np.log(mean),eta,chi,chi/2-eta*eta/8


def deterministic_checks():
    print('DETERMINISTIC CHECKS; all parameter values illustrative',flush=True)
    err=[]
    for tau in ([2,2],[5,2],[2,3,7],[1,4,9]):
        for phi in (-.8,0,.4,.8):
            m=ring(tau,active=[0],phi=phi)
            err.append(abs(coeff(m)[0]-kgain(tau,phi)))
    for b in (2.,20.,40.):
        for d in (0.,.1,.8):
            for phi in (0.,.8):
                m=contact(b,d,phi)
                err.append(abs(coeff(m)[0]-kcontact([5,b],d,phi)))
    print(f'algebra_resolvent_max_error={max(err):.12g} PASS={max(err)<1e-9}')
    # Alternating environments are ergodic after uniform random initial phase.
    for kind in ('gain','contact','opposite','same'):
        if kind=='gain': m=ring([2,2],active=[0],phi=-1.)
        elif kind=='contact': m=contact(2,.1,0); m.phi[:]=-1
        else: m=ring([2,2],phi=-1.,signs=[1,-1] if kind=='opposite' else [1,1],shared=True)
        K=coeff(m)[0]
        d1=floquet(m,.002,curve='exponential' if kind in ('opposite','same') else 'linear')[0]
        d2=floquet(m,.001,curve='exponential' if kind in ('opposite','same') else 'linear')[0]
        extrap=(4*d2/.001**2-d1/.002**2)/3
        ex,res=floquet(m,.6,curve='exponential' if kind in ('opposite','same') else 'linear')
        print(f'Floquet {kind}: K={K:.12g} extrap={extrap:.12g} error={extrap-K:.3g} '
              f'PASS={abs(extrap-K)<2e-5}; amplitude=.6 D={ex:.12g} residual={res:.3g}')
    # Positive-memory periodic blocks: independent full-product boundary tests.
    for period in (2,20,100):
        for d in (0.,.1,.8):
            vals=[floquet(contact(b,d,0),.6,period)[0] for b in (20.,40.)]
            print(f'periodic_contact period={period} d={d:g} b20={vals[0]:.12g} b40={vals[1]:.12g} difference={vals[1]-vals[0]:.12g}')
    # Periodic mortality boundary for the long-parking comparison.
    fun=lambda d:floquet(contact(40,d,0),.6,100)[0]-floquet(contact(20,d,0),.6,100)[0]
    grid=np.linspace(.01,.99,60); roots=[]
    for l,h in zip(grid[:-1],grid[1:]):
        if fun(l)*fun(h)<0: roots.append(brentq(fun,l,h,xtol=1e-11))
    print('periodic_parking_d_boundaries period=100',roots)
    for r in roots:
        for d in (r-.03,r+.03):print(f'periodic_parking_side d={d:.12g} difference={fun(d):.12g}')
    for d in (.1,.8):
        for b in (2.,20.,40.):
            Kg=kgain([5,b],.8); Kc=kcontact([5,b],d,.8)
            print(f'contact_closed b={b:g} d={d:g} phi=.8 Kg={Kg:.12g} Kc={Kc:.12g} ratio={Kc/Kg:.12g} derivative={parking_deriv(5,b,d,.8):.12g}')
    bstar=brentq(lambda b:parking_deriv(5,b,.8,.8),1,1000)
    print(f'parking_local_boundary a=5 d=.8 phi=.8 bstar={bstar:.12g}')
    for b in (bstar-1,bstar+1):print(f'parking_local_side b={b:.12g} derivative={parking_deriv(5,b,.8,.8):.12g}')
    for phi in (-.8,-.5,0.,.8):
        print(f'gain_local_sign_boundary tau=(2,2) phi={phi:g} K={kgain([2,2],phi):.12g}')
    for sig in (.1,.3,1.):
        for kind,x in (('II',1.),('III',.2),('III',.8),('power',3.),('power',1.),('power',-1.)):
            vals=curvature(kind,x,sig); vals2=curvature(kind,x,sig,320)
            print(f'curvature {kind} x={x:g} sigma={sig:g} baseline_i_threshold={vals[0]:.12g} '
                  f'baseline_ii_loss={vals[1]:.12g} eta={vals[2]:.12g} chi={vals[3]:.12g} '
                  f'advantage_coefficient={vals[4]:.12g} quadrature_error={max(abs(np.array(vals)-vals2)):.3g}')
        root=brentq(lambda x:curvature('III',x,sig)[0],.001,2.,xtol=1e-12)
        print(f'curvature_III_finite_boundary sigma={sig:g} x={root:.12g} local_boundary={np.sqrt(1/6):.12g}')
        for x in (.8*root,1.2*root):print(f'curvature_III_side sigma={sig:g} x={x:.12g} baseline_i_threshold={curvature("III",x,sig)[0]:.12g}')
    for newtau,k,rho in ((1.,1.,0.),(8.,3.,0.),(8.,3.,.8)):
        G=10.; V=1.; U=k*k; C=rho*k; delta=newtau-8
        boundary=(2*G*(V-C)+V*delta)/(G*(V-2*C+U))
        print(f'parallel_iid_boundary tau2={newtau:g} k2={k:g} rho={rho:g} fstar={boundary:.12g}')
        for f in (.1,.4,.8):
            m=parallel(f,newtau,k,rho); K,Gnew=coeff(m)
            Kform=((1-f)**2+2*rho*k*f*(1-f)+(k*f)**2)/(2*(10+f*delta))
            print(f'parallel_iid_side f={f:g} K={K:.12g} difference_vs_old={K-.05:.12g} algebra_error={K-Kform:.3g}')
    for period in (2,20):
        base=floquet(ring([2,2],active=[0]),.6,period)[0]
        fun=lambda q:np.log(.5+.5*q)+base-floquet(optional(q=q),.6,period)[0]
        # Positive saving can disappear for some periodic spectra. Report it.
        if fun(1)>0:
            qstar=brentq(fun,.01,1,xtol=1e-12)
            print(f'optional_periodic_boundary period={period} qstar={qstar:.12g} baseD={base:.12g}')
            for q in (qstar-.01,min(1.,qstar+.01)):
                print(f'optional_periodic_side period={period} q={q:.12g} advantage_fixed_rates={fun(q):.12g}')
        else:print(f'optional_periodic_no_survival_boundary period={period} advantage_q1={fun(1):.12g}')
    for sig in (.3,1.):
        print(f'H4_outside_unit_elasticity sigma={sig:g} host_bound={sig*sig/2:.12g} rate_bound={9*sig*sig/2:.12g} '
              f'exact_no_stay_power3_loss={scalar_loss(0,sig,3):.12g}')
    print('END DETERMINISTIC CHECKS',flush=True)


def cases():
    out=[]
    for sigma in (.1,.3,1.):
        for phi in (0.,.8):
            for d in (0.,.1,.8):
                for b in (2.,20.,40.) if sigma!=.1 else (2.,):
                    out.append(contact(b,d,phi,sigma))
            out.append(contact(2.,.1,phi,sigma,gain=True))
        if sigma==.1:
            out.append(ring([2,2],active=[0],phi=.8,sigma=sigma,name='single (2,2) phi=0.8'))
            out.append(parallel(.4,1.,phi=.8,sigma=sigma))
            out.append(optional(phi=.8,sigma=sigma))
            out.append(bypass(phi=.8,sigma=sigma))
            continue
        for phi in (-.8,0.,.8):
            for sign in (-1.,1.):
                out.append(ring([2,2],phi=phi,signs=[1,sign],shared=True,sigma=sigma,
                                name=f'shared sign={sign:g} phi={phi:g}'))
            out.append(ring([2,2],active=[0],phi=phi,sigma=sigma,name=f'single (2,2) phi={phi:g}'))
        for phi in (0.,.8):
            out.append(ring([2,8],active=[0],phi=phi,sigma=sigma,name=f'parallel-old phi={phi:g}'))
            for f,t,k,rho in ((.4,1.,1.,0.),(.8,1.,1.,0.),(.1,8.,3.,0.),(.4,8.,3.,0.),(.1,8.,3.,.8)):
                out.append(parallel(f,t,k,rho,phi,sigma))
            for q in (.8,.999,1.):out.append(optional(q=q,phi=phi,sigma=sigma))
            for q in (1.,.5):
                for f in (0.,.25,.5,.75,1.):out.append(bypass(f,q,phi,sigma))
            for tau in ([20.],[10.,10.],[20/3]*3,[18.,1.,1.]):
                label=','.join(f'{t:g}' for t in tau)
                out.append(ring(tau,phi=phi,sigma=sigma,name=f'series ({label}) all phi={phi:g}'))
                if len(tau)>1:
                    for j in range(len(tau)):
                        out.append(ring(tau,active=[j],phi=phi,sigma=sigma,name=f'series ({label}) only{j} phi={phi:g}'))
        for phi in (.9,.98):
            for tau in ([10.,10.],[18.,1.,1.]):
                label=','.join(f'{t:g}' for t in tau)
                out.append(ring(tau,phi=phi,sigma=sigma,name=f'series ({label}) all phi={phi:g}'))
                for j in range(len(tau)):
                    out.append(ring(tau,active=[j],phi=phi,sigma=sigma,name=f'series ({label}) only{j} phi={phi:g}'))
    return out


def pack(ms):
    C=len(ms)
    A=np.zeros((C,N,N)); B=np.zeros((C,E,N,N)); R=A.copy(); RB=B.copy()
    kap=np.zeros((C,E)); phi=kap.copy(); L=np.zeros((C,E,E)); bounded=np.zeros(C,bool)
    for c,m in enumerate(ms):
        n=len(m.A); e=len(m.B)
        A[c,:n,:n]=m.A; B[c,:e,:n,:n]=m.B; R[c,:n,:n]=m.R; RB[c,:e,:n,:n]=m.RB
        kap[c,:e]=m.kap; phi[c,:e]=m.phi
        # Stationary cross-correlations are inputs. Innovations must be PSD.
        cov=m.corr*(1-m.phi[:,None]*m.phi[None,:])
        vv,U=np.linalg.eigh(cov); assert vv.min()>-1e-10
        L[c,:e,:e]=U@np.diag(np.sqrt(np.maximum(vv,0)))
        bounded[c]=(m.kind=='bounded')
    return A,B,R,RB,kap,phi,L,bounded


def product_rates(ms,packed,y):
    """Direct random products and derivatives. No coefficient formula here."""
    A,B,R,RB,kap,phi,L,bounded=packed; C=len(ms)
    # Identical random innovations for each case allow paired comparisons.
    rng=np.random.default_rng(SEED)
    z=np.zeros((C,2*REPS,E))
    state=np.zeros((C,2*REPS,E),bool)
    x=np.ones((C,2*REPS,N))/N; dx=np.zeros_like(x)
    sig=np.array([m.sigma for m in ms])[:,None,None]
    y2=np.repeat(y,2,axis=1); ey=np.exp(y2)
    rate=np.zeros((C,2*REPS)); deriv=rate.copy()
    for t in range(STEPS+BURN):
        eps=rng.standard_normal((REPS,E)); eps=np.stack((eps,-eps),axis=1).reshape(2*REPS,E)
        z=phi[:,None,:]*z+np.einsum('cde,re->crd',L,eps,optimize=False)
        h=np.exp(sig*kap[:,None,:]*z-(sig*kap[:,None,:])**2/2)
        u=rng.random((2*REPS,E)); v=rng.random((2*REPS,E))
        if bounded.any():
            # P = phi I + (1-phi) 1 pi, common uniforms for coupling cases.
            state=np.where(u[None,:,:]<phi[:,None,:],state,v[None,:,:]<.2)
            hb=1+np.sqrt(np.expm1(sig**2))*np.where(state,2.,-.5)
            h[bounded]=hb[bounded]
        At=A[:,None,:,:]+np.einsum('cre,ceij->crij',h-1,B,optimize=False)
        Rt=R[:,None,:,:]+np.einsum('cre,ceij->crij',h-1,RB,optimize=False)
        assert np.min(At)>-1e-11
        At=At+(ey-1)[:,:,None,None]*Rt
        xn=np.einsum('crij,crj->cri',At,x,optimize=False)
        dn=np.einsum('crij,crj->cri',At,dx,optimize=False)+ey[:,:,None]*np.einsum('crij,crj->cri',Rt,x,optimize=False)
        norm=xn.sum(axis=-1); ds=dn.sum(axis=-1)/norm
        xn/=norm[:,:,None]; dx=dn/norm[:,:,None]-xn*ds[:,:,None]; x=xn
        if t>=BURN:
            rate+=np.log(norm); deriv+=ds
    return rate.reshape(C,REPS,2).mean(axis=2)/STEPS,deriv.reshape(C,REPS,2).mean(axis=2)/STEPS


def worker(job):
    global STEPS
    ms,STEPS=job
    start=time.time(); packed=pack(ms)
    K=np.array([coeff(m)[0] for m in ms]); sig=np.array([m.sigma for m in ms])
    y=np.repeat((K*sig**2)[:,None],REPS,axis=1)
    maxstep=np.inf
    for iteration in range(6):
        r,dr=product_rates(ms,packed,y)
        step=r/dr; y-=step; maxstep=np.max(np.abs(step))
        if maxstep<2e-5:break
    out=[]
    for c,m in enumerate(ms):
        out.append(dict(name=m.name,sigma=m.sigma,D=float(y[c].mean()),
                        se=float(y[c].std(ddof=1)/np.sqrt(REPS)),K=float(K[c]),
                        roots=y[c].tolist(),residual=float(np.max(abs(step[c]))),
                        iterations=iteration+1,kind=m.kind,steps=STEPS))
    return out,time.time()-start


def summarize(results):
    by={(r['name'],r['sigma']):r for r in results}
    def compare(label,terms,offset=0.):
        vals=sum(mult*np.array(by[key]['roots']) for mult,key in terms)+offset
        mean=vals.mean(); se=vals.std(ddof=1)/np.sqrt(REPS)
        print(f'COMPARE {label}: difference={mean:.12g} SE={se:.4g} fiveSE_sign={"positive" if mean>5*se else "negative" if mean < -5*se else "unresolved"}')
    for sig in (.3,1.):
        for phi in (0.,.8):
            for d in (.1,.8):
                compare(f'contact parking sigma={sig:g} phi={phi:g} d={d:g}',
                        [(1,(f'contact b=40 d={d:g} phi={phi:g}',sig)),(-1,(f'contact b=20 d={d:g} phi={phi:g}',sig))])
            for f,t,k,rho in ((.4,1.,1.,0.),(.8,1.,1.,0.),(.1,8.,3.,0.),(.4,8.,3.,0.),(.1,8.,3.,.8)):
                compare(f'parallel vs old sigma={sig:g} phi={phi:g} f={f:g} t={t:g} k={k:g} rho={rho:g}',
                        [(1,(parallel(f,t,k,rho,phi,sig).name,sig)),(-1,(f'parallel-old phi={phi:g}',sig))])
            for q in (.8,.999,1.):
                terms=[(1,(f'single (2,2) phi={phi:g}',sig)),(-1,(optional(q=q,phi=phi).name,sig))]
                compare(f'optional saving equal R0 sigma={sig:g} phi={phi:g} q={q:g}',terms)
                compare(f'optional advantage fixed rates sigma={sig:g} phi={phi:g} q={q:g}',terms,np.log(.5+.5*q))
            for q in (1.,.5):
                for f in (.25,.5,.75,1.):
                    compare(f'bypass advantage fixed rates sigma={sig:g} phi={phi:g} f={f:g} q={q:g}',
                        [(1,(bypass(0,q,phi).name,sig)),(-1,(bypass(f,q,phi).name,sig))],np.log(1-f+f*q))
            for label,n in (('10,10',2),('6.66667,6.66667,6.66667',3),('18,1,1',3)):
                terms=[(1,(f'series ({label}) all phi={phi:g}',sig))]+[(-1,(f'series ({label}) only{j} phi={phi:g}',sig)) for j in range(n)]
                compare(f'additivity sigma={sig:g} phi={phi:g} tau={label}',terms)
            compare(f'H3 unequal repartition sigma={sig:g} phi={phi:g}',
                    [(1,(f'series (18,1,1) all phi={phi:g}',sig)),(-1,(f'series (10,10) all phi={phi:g}',sig))])
        for phi in (-.8,0.,.8):
            r=by[(f'shared sign=1 phi={phi:g}',sig)]
            exact=2*scalar_loss(.5,sig)
            print(f'EXACT_SHARED_SAME sigma={sig:g} phi={phi:g} quadrature={exact:.12g} simulation={r["D"]:.12g} SE={r["se"]:.4g} PASS={abs(r["D"]-exact)<5*r["se"]+2e-5}')
        for phi in (.9,.98):
            compare(f'H3 unequal repartition sigma={sig:g} phi={phi:g}',
                    [(1,(f'series (18,1,1) all phi={phi:g}',sig)),(-1,(f'series (10,10) all phi={phi:g}',sig))])
            for label,n in (('10,10',2),('18,1,1',3)):
                compare(f'additivity sigma={sig:g} phi={phi:g} tau={label}',
                    [(1,(f'series ({label}) all phi={phi:g}',sig))]+[(-1,(f'series ({label}) only{j} phi={phi:g}',sig)) for j in range(n)])
    for r in results:
        if r['sigma']==.1:
            remainder=2*.1**(3 if r['kind']=='bounded' else 4)
            print(f'SMALL_NOISE {r["name"]} error={r["D"]-.01*r["K"]:.8g} tolerance={5*r["se"]+remainder:.8g} PASS={abs(r["D"]-.01*r["K"])<5*r["se"]+remainder}')


def memory_boundary_rates(ph,sig,opposite,birth=None):
    """Direct products at birth tilt zero; derivatives are with respect to phi."""
    steps=200000; burn=2000; C=len(sig)
    rng=np.random.default_rng(SEED+71)
    z=np.zeros((C,REPS)); dz=z.copy()
    x=np.full((C,REPS,2),.5); dx=np.zeros_like(x)
    acc=z.copy(); der=z.copy(); sig=sig[:,None]; op=opposite[:,None]
    noise_scale=np.sqrt(1-ph*ph)
    tilt=np.ones((C,1)) if birth is None else np.exp(birth[:,None]/2)
    for t in range(steps+burn):
        eps=rng.standard_normal((1,REPS))
        dz=z+ph*dz-ph/noise_scale*eps
        z=ph*z+noise_scale*eps
        hp=tilt*np.exp(sig*z-sig*sig/2)
        hm=tilt*np.where(op,np.exp(-sig*z-sig*sig/2),1.)
        dhp=hp*sig*dz; dhm=np.where(op,-hm*sig*dz,0.)
        xn=np.empty_like(x); dn=np.empty_like(x)
        xn[:,:,0]=.5*(x[:,:,0]+hm*x[:,:,1])
        xn[:,:,1]=.5*(hp*x[:,:,0]+x[:,:,1])
        dn[:,:,0]=.5*(dx[:,:,0]+hm*dx[:,:,1]+dhm*x[:,:,1])
        dn[:,:,1]=.5*(hp*dx[:,:,0]+dhp*x[:,:,0]+dx[:,:,1])
        total=xn.sum(axis=2); ds=dn.sum(axis=2)/total
        xn/=total[:,:,None]; dx=dn/total[:,:,None]-xn*ds[:,:,None]; x=xn
        if t>=burn: acc+=np.log(total); der+=ds
    return acc/steps,der/steps


def gain_sides():
    start=time.time(); phis=np.array([-.65,-.45])
    rr,_=memory_boundary_rates(np.repeat(phis[:,None],REPS,axis=1),np.full(2,.3),np.zeros(2,bool))
    for i,ph in enumerate(phis):
        mean=rr[i].mean(); se=rr[i].std(ddof=1)/np.sqrt(REPS)
        print(f'FINITE_AR_GAIN_WIDE_SIDE sigma=.3 phi={ph:g} growth_at_D0={mean:.12g} SE={se:.6g} fiveSE_sign={"positive" if mean>5*se else "negative" if mean < -5*se else "unresolved"}',flush=True)
    elapsed=time.time()-start
    print(f'GAIN_SIDE_SECONDS={elapsed:.3f}',flush=True)
    return elapsed


def memory_boundaries():
    start=time.time()
    ms=[]
    for sigma in (.1,.3,1.):
        m=parallel(.4,1.,1.,.4,0.,sigma)
        m.phi=np.array([.2,.8]); m.name='parallel unequal memory=(.2,.8) rho=.4'
        ms.append(m)
    rows,_=worker((ms,40000))
    for m,row in zip(ms,rows):
        _,G=coeff(m); weights=np.array([.6,.4]); closed=0.
        for i in range(2):
            ph=m.phi[i]; t=ph/(ph+(1-ph)*np.array([2.,8.,1.]))
            P=t[0]*np.dot(weights,t[1:])
            for j in range(2):
                closed+=m.corr[i,j]*weights[i]*weights[j]*((1+ph)/(2*(1-ph)*G)-t[0]*t[j+1]/(1-P))
        print(f'UNEQUAL_MEMORY sigma={m.sigma:g} closed_K={closed:.12g} general_K={row["K"]:.12g} D={row["D"]:.12g} SE={row["se"]:.6g} residual_y={row["residual"]:.3g} algebra_PASS={abs(closed-row["K"])<1e-9}',flush=True)
        if m.sigma==.1:
            print(f'UNEQUAL_MEMORY_SMALL_NOISE error={row["D"]-.01*closed:.12g} PASS={abs(row["D"]-.01*closed)<5*row["se"]+2*.1**4}',flush=True)
    for sigma in (.3,1.):
        for eta in (.5,1.,3.):
            print(f'H4_HOST_VARIANCE_SIDE sigma={sigma:g} eta={eta:g} exact_loss={scalar_loss(0,sigma,eta):.12g} host_variance_bound={sigma*sigma/2:.12g}',flush=True)
    sig=np.array([.3,1.,.3,1.]); op=np.array([False,False,True,True])
    endpoints=np.tile([-.97,.95],len(sig))
    er,_=memory_boundary_rates(np.repeat(endpoints[:,None],REPS,axis=1),np.repeat(sig,2),np.repeat(op,2))
    means=er.mean(axis=1).reshape(-1,2); valid=means[:,0]*means[:,1]<0
    for c in range(len(sig)):
        if not valid[c]:
            for side in range(2):
                rr=er[2*c+side]
                print(f'FINITE_AR_NO_BRACKET mode={"opposite" if op[c] else "gain"} sigma={sig[c]:g} phi={endpoints[2*c+side]:g} growth_at_D0={rr.mean():.12g} SE={rr.std(ddof=1)/np.sqrt(REPS):.6g}',flush=True)
    sig=sig[valid]; op=op[valid]
    ph=np.repeat(np.where(op,-.1,-.5)[:,None],REPS,axis=1)
    for iteration in range(10):
        r,dr=memory_boundary_rates(ph,sig,op)
        step=np.clip(r/dr,-.2,.2); ph=np.clip(ph-step,-.97,.95)
        if np.max(abs(step))<2e-5:break
    centers=ph.mean(axis=1)
    for c in range(len(sig)):
        se=ph[c].std(ddof=1)/np.sqrt(REPS)
        print(f'FINITE_AR_BOUNDARY mode={"opposite" if op[c] else "gain"} sigma={sig[c]:g} phi={centers[c]:.12g} SE={se:.6g} max_correction={max(abs(step[c])):.3g} PASS={max(abs(step[c]))<2e-5}',flush=True)
    side=np.stack((centers-.02,centers+.02),axis=1).reshape(-1)
    rr,_=memory_boundary_rates(np.repeat(side[:,None],REPS,axis=1),np.repeat(sig,2),np.repeat(op,2))
    for i in range(len(side)):
        mean=rr[i].mean(); se=rr[i].std(ddof=1)/np.sqrt(REPS)
        print(f'FINITE_AR_SIDE mode={"opposite" if op[i//2] else "gain"} sigma={sig[i//2]:g} phi={side[i]:.12g} growth_at_D0={mean:.12g} SE={se:.6g} fiveSE_sign={"positive" if mean>5*se else "negative" if mean < -5*se else "unresolved"}',flush=True)
    rsig=np.array([.3,.3,1.,1.]); rphi=np.array([0.,.8,0.,.8])
    rr,_=memory_boundary_rates(np.repeat(rphi[:,None],REPS,axis=1),rsig,np.ones(4,bool),birth=rsig**2)
    z,w=roots_hermitenorm(160); w/=np.sqrt(2*np.pi)
    for i in range(4):
        arg=rsig[i]*np.sqrt((1-rphi[i])/2)*z
        exact=w@(np.logaddexp(arg,-arg)-np.log(2.))
        mean=rr[i].mean(); se=rr[i].std(ddof=1)/np.sqrt(REPS)
        print(f'RAW_RECIPROCAL_BASELINE_I sigma={rsig[i]:g} phi={rphi[i]:g} exact_growth={exact:.12g} matrix_growth={mean:.12g} SE={se:.6g} PASS={abs(mean-exact)<5*se+2e-5}',flush=True)
    gain_sides()
    elapsed=time.time()-start
    print(f'MEMORY_BOUNDARY_SECONDS={elapsed:.3f}',flush=True)
    return elapsed


def main():
    start=time.time()
    print(f'Python={sys.version.split()[0]} numpy={np.__version__} seed={SEED} processes=3 reps={REPS} antithetic_paths={2*REPS} steps={STEPS} burn={BURN}',flush=True)
    deterministic_checks()
    extra_checks()
    ms=cases(); print(f'SIMULATION cases={len(ms)} baseline=(ii), threshold mean R0=1',flush=True)
    # Balance equal-size batches; all share innovations, even across workers.
    batches=[ms[i::3] for i in range(3)]
    with mp.get_context('spawn').Pool(3) as pool:
        data=pool.map(worker,[(batch,STEPS) for batch in batches])
        focus=[m for m in ms if (m.sigma==.3 and m.name.startswith('series (18,1,1)') and m.phi[0] in (0.,.8)) or
               (m.sigma==1. and m.phi[0]==.98 and ' all ' in m.name)]
        focused=pool.map(worker,[(focus[i::3],400000) for i in range(3)])
    results=[r for rs,_ in data for r in rs]
    replace={(r['name'],r['sigma']):r for rs,_ in focused for r in rs}
    results=[replace.get((r['name'],r['sigma']),r) for r in results]
    results.sort(key=lambda r:(r['sigma'],r['name']))
    for r in results:
        print(f'CASE sigma={r["sigma"]:g} {r["name"]}: D={r["D"]:.12g} SE={r["se"]:.6g} D2={r["K"]*r["sigma"]**2:.12g} K={r["K"]:.12g} residual_y={r["residual"]:.3g} iterations={r["iterations"]} steps={r["steps"]} root_PASS={r["residual"]<2e-5}')
    summarize(results)
    models={(m.name,m.sigma):m for m in ms}; excess=[]; failures=[]
    for r in results:
        m=models[(r['name'],r['sigma'])]
        if m.kind!='gaussian':continue
        _,v,w=perron(m.A); birth=w@m.R@v
        bound=sum((w@B@v)*(m.sigma*k)**2/2 for B,k in zip(m.B,m.kap))/birth
        excess.append(r['D']-bound)
        if r['D']-bound>5*r['se']+1e-10:failures.append(m.name)
    print(f'H4_weighted_bound cases={len(excess)} max_D_minus_bound={max(excess):.6g} failures={failures} PASS={not failures}')
    print(f'worker_seconds={[round(t,3) for _,t in data]}')
    print(f'precision_worker_seconds={[round(t,3) for _,t in focused]}')
    memory_boundaries()
    elapsed=time.time()-start
    print(f'TOTAL_SECONDS={elapsed:.3f} UNDER_15_MINUTES={elapsed<900}',flush=True)


if __name__=='__main__':
    if '--gain-sides-only' in sys.argv:
        elapsed=gain_sides()
        from pathlib import Path
        import re
        previous=re.findall(r'^RECORDED_COMPONENT_TOTAL_SECONDS=([0-9.]+)',Path('mechanism_checks_output.txt').read_text(),re.M)
        if previous:
            combined=float(previous[-1])+elapsed
            print(f'RECORDED_COMPONENT_TOTAL_SECONDS={combined:.3f} UNDER_15_MINUTES={combined<900}',flush=True)
    elif '--memory-boundaries-only' in sys.argv:
        elapsed=memory_boundaries()
        from pathlib import Path
        import re
        path=Path('mechanism_checks_output.txt')
        if path.exists():
            previous=re.findall(r'^TOTAL_SECONDS=([0-9.]+)',path.read_text(),re.M)
            if previous:
                combined=float(previous[-1])+elapsed
                print(f'RECORDED_COMPONENT_TOTAL_SECONDS={combined:.3f} UNDER_15_MINUTES={combined<900}',flush=True)
    else:main()
