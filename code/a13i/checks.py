#!/usr/bin/env python3
"""Claim: finite fourth moments suffice for the eventual A13 derivative sign.

Deterministic falsification checks of every finite algebraic step and seven
unbounded-law numerical cells; tolerances are in protocol.md, written first.
No collaborator imports, data, random simulation, or certified numerics.
The ratio Poisson discretization architecture is adapted, independently coded,
from verification/a13-check/verify_a13.py. Direct Perron and finite-product gradient
calculations provide different numerical routes from the proof's ratio algebra.
"""
from pathlib import Path
import json
import math
import time
import mpmath as mp
import numpy as np
from scipy.special import roots_genlaguerre, roots_hermitenorm, roots_legendre
from scipy.optimize import brentq

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 200
RECORDS = [[1], [2, 0], [2, 2, 0, 0], [0, 0, 1, 3], [16] + [0]*15]
ASTAYS = [mp.mpf('1.25'), mp.mpf(2), mp.mpf(5)]
COUNT = 0
FAILURES = []
METRICS = {}


def check(label, error, limit):
    global COUNT
    COUNT += 1
    if not (error <= limit):
        FAILURES.append(dict(label=label, error=str(error), limit=str(limit)))
        print('FAIL', label, 'error=', mp.nstr(error, 12), 'limit=', limit, flush=True)


def maximum(label, value):
    METRICS[label] = max(METRICS.get(label, 0.0), float(value))


def constants(a, H):
    b=1/a
    T=2*H+1; Q=4*H+1; U=T/b; L=2*Q/b
    N=L*(Q+2*H)/b; J=U*Q/b; Z=(Q*N+2*H*L**2)/b
    B0=4*H**2/b+2/b+2*H**2+3
    C0=2*H*J+2*H*T*U+T**3/3
    R4=2*H*Z+2*H*T*N+2*H**2*L**2+(2*H*L/3)*(Q**2+Q*T+T**2)+4*Q**4
    K=B0+C0+R4
    e0=min(b/4,mp.mpf(1)/16,1/(8*(K+1)))
    Mx=1+4*B0+9*C0; My=2+4*B0+9*C0
    Ex=Mx/2+3*R4; Ey=My/2+2*R4; E0=2*Ex+Ey
    J0=K+2*B0+3*C0+Ey
    N0=4*B0*K+2*C0+R4+E0
    estar=min(e0,1/(2*J0)); Cstar=2*(N0+B0*J0)
    return locals()


def mat(h, a, q, g):
    b=1/a
    return mp.matrix([[1-b,g],[b*h,q]])


def product(seq,a,q,g):
    prod=mp.eye(2)
    for h in seq:
        prod=mat(h,a,q,g)*prod
    return prod


def perron(prod):
    return (prod[0,0]+prod[1,1]+mp.sqrt((prod[0,0]-prod[1,1])**2+4*prod[0,1]*prod[1,0]))/2


def growth(seq,a,e,c):
    return mp.log(perron(product(seq,a,1-e,e*c)))/len(seq)


def root_and_derivative(seq,a,e):
    d=mp.findroot(lambda d:growth(seq,a,e,mp.exp(d)),(mp.mpf(0),e),tol=mp.mpf('1e-180'))
    c=mp.exp(d)
    re=mp.diff(lambda ee:growth(seq,a,ee,c),e)
    rd=mp.diff(lambda dd:growth(seq,a,e,mp.exp(dd)),d)
    return d,-re/rd


def average(xs):
    return mp.fsum(xs)/len(xs)


def norm(xs,k):
    return average([abs(x)**k for x in xs])**(1/k)


def linear_periodic(p,source):
    """Exact periodic solution z[t+1]=p*z[t]+source[t]."""
    n=len(source)
    start=mp.fsum(p**(n-1-j)*source[j] for j in range(n))/(1-p**n)
    out=[]; z=start
    for value in source:
        out.append(z); z=p*z+value
    return out


def coefficients(seq,a,c):
    p=1-1/a
    k=[c*h-1 for h in seq]
    u=linear_periodic(p,[-x for x in k])
    w=linear_periodic(p,[-c*seq[t]*u[t]-k[t]*u[(t+1)%len(seq)] for t in range(len(seq))])
    B=average([k[t]**2/2-c*seq[t]*u[t] for t in range(len(seq))])
    C=average([c*seq[t]*w[t]-k[t]*c*seq[t]*u[t]+k[t]**3/3 for t in range(len(seq))])
    return B,C,u,w


def raw_A(seq,a):
    p=1-1/a; n=len(seq)
    cov=[average([(seq[t]-1)*(seq[(t-l)%n]-1) for t in range(n)]) for l in range(n+1)]
    return cov[0]/2+mp.fsum(p**(l-1)*cov[l] for l in range(1,n+1))/(1-p**n)


def benchmark():
    print('Known-result gates: direct Perron calculations',flush=True)
    for a in ASTAYS:
        for e in map(mp.mpf,['.01','.001']):
            d,de=root_and_derivative([1],a,e)
            check('constant D',abs(d),mp.mpf('1e-65'))
            check('constant derivative',abs(de),mp.mpf('1e-65'))
            d,de=root_and_derivative([2,0],a,e)
            s=1/e
            exact=mp.log((2*a-1)*(2*s-1)/(2*((2*a-1)*s-a+1)))
            exactde=-1/(e**2*(2*s-1)*((2*a-1)*s-a+1))
            maximum('known_D_error',abs(d-exact))
            maximum('known_D_e_error',abs(de-exactde))
            check('alternating D',abs(d-exact),mp.mpf('1e-65'))
            check('alternating derivative',abs(de-exactde),mp.mpf('1e-65'))


def periodic_audit():
    print('Periodic algebra: 90 (record,a,c,e) cells; 15 interval cells',flush=True)
    for record in RECORDS:
        seq=list(map(mp.mpf,record)); n=len(seq)
        H=norm(seq,mp.mpf(4))
        for a in ASTAYS:
            b=1/a; p=1-b; co=constants(a,H)
            A=raw_A(seq,a)
            check('B(1)=A',abs(coefficients(seq,a,mp.mpf(1))[0]-A),mp.mpf('1e-60'))
            for index,key in [(0,'B0'),(1,'C0')]:
                deg=2+index
                vals=mp.matrix([coefficients(seq,a,mp.mpf(j))[index] for j in range(deg+1)])
                vand=mp.matrix([[mp.mpf(j)**k for k in range(deg+1)] for j in range(deg+1)])
                cc=mp.lu_solve(vand,vals)
                check('coefficient majorant',mp.fsum(abs(cc[j])*2**j for j in range(deg+1))/co[key],1+mp.mpf('1e-60'))
            for c in [mp.mpf('.5'),mp.mpf(1),mp.mpf(2)]:
                B,C,u,w=coefficients(seq,a,c)
                for e in [min(mp.mpf('.01'),b/8),min(mp.mpf('.01'),b/8)/10]:
                    prod=product(seq,a,1-e,e*c); lam=perron(prod)
                    v=(e*c/b)**(-1)*prod[0,1]/(lam-prod[0,0])
                    vs=[]
                    for h in seq:
                        vs.append(v); v=(p*v+b)/(1+e*(c*h*v-1))
                    delta=[v-1 for v in vs]
                    eta=[delta[t]-e*u[t] for t in range(n)]
                    zeta=[eta[t]-e**2*w[t] for t in range(n)]
                    for t in range(n):
                        j=(t+1)%n; k=c*seq[t]-1; W=c*seq[t]*vs[t]-1
                        identities=[delta[j]-p*delta[t]+e*vs[j]*W,
                          eta[j]-p*eta[t]+e*(delta[j]*W+c*seq[t]*delta[t]),
                          zeta[j]-p*zeta[t]+e*(c*seq[t]*eta[t]+k*eta[j]+c*seq[t]*delta[t]*delta[j])]
                        for residual in identities:
                            maximum('recursion_residual',abs(residual))
                            check('ratio recursion',abs(residual),mp.mpf('1e-60'))
                    for values,power,bound in [(u,4,co['U']),(w,2,co['J']),
                       (delta,4,co['L']*e),(eta,2,co['N']*e**2),(zeta,mp.mpf(4)/3,co['Z']*e**3)]:
                        ratio=norm(values,mp.mpf(power))/bound
                        maximum('moment_bound_ratio',ratio)
                        check('Lp bound',ratio,1+mp.mpf('1e-60'))
                    r=growth(seq,a,e,c); P=e*(c-1)-e**2*B+e**3*C
                    rr=abs(r-P)/(co['R4']*e**4)
                    maximum('growth_remainder_ratio',rr)
                    check('cubic growth remainder',rr,1+mp.mpf('1e-60'))
                    # Independent recovery of cubic coefficient from tiny-e Perron value.
                ee=mp.mpf('1e-25')
                c_est=(growth(seq,a,ee,c)-ee*(c-1)+ee**2*B)/ee**3
                check('cubic coefficient from matrix product',abs(c_est-C)/(1+abs(C)),mp.mpf('1e-18'))
            cutoff=co['estar'] if abs(A)<mp.mpf('1e-100') else min(co['estar'],(abs(A)/(2*co['Cstar']))**2)
            e=cutoff/2; d,de=root_and_derivative(seq,a,e); c=mp.exp(d)
            err=abs(de-A); radius=co['Cstar']*mp.sqrt(e)
            maximum('threshold_derivative_interval_ratio',err/radius)
            check('threshold derivative interval',err,radius+mp.mpf('1e-100'))
            if abs(A)>mp.mpf('1e-100'):
                check('derivative sign',-de*A,mp.mpf('1e-100'))
            def P_log(x,y):
                et=1-mp.exp(x); ct=mp.exp(y)/et
                B,C,_,_=coefficients(seq,a,ct)
                return et*(ct-1)-et**2*B+et**3*C
            x=mp.log(1-e); y=mp.log(e*c)
            def R_log(x,y):
                return mp.log(perron(product(seq,a,mp.exp(x),mp.exp(y))))/n
            for j,step,bound in [(0,e**2,co['Ex']*e**2),(1,e**mp.mpf('1.5'),co['Ey']*e**mp.mpf('2.5'))]:
                f=(lambda t:R_log(t,y)) if j==0 else (lambda t:R_log(x,t))
                poly=(lambda t:P_log(t,y)) if j==0 else (lambda t:P_log(x,t))
                z=x if j==0 else y
                der=mp.diff(f,z); pd=mp.diff(poly,z)
                ratio=abs(der-pd)/bound
                maximum('convex_derivative_bound_ratio',ratio)
                check('convex derivative interval',abs(der-pd),bound+mp.mpf('1e-100'))
                check('lower convex secant',(f(z)-f(z-step))/step-der,mp.mpf('1e-100'))
                check('upper convex secant',der-(f(z+step)-f(z))/step,mp.mpf('1e-100'))
            print(' interval',record,'a=',float(a),'A=',mp.nstr(A,9),'e=',mp.nstr(e,5),'D_e=',mp.nstr(de,9),flush=True)


def finite_gradient_audit():
    print('Finite-product edge probabilities and projective contraction',flush=True)
    for record in RECORDS:
        for a in ASTAYS:
            seq=record*3; q=mp.mpf('.97'); g=mp.mpf('.031'); n=len(seq)
            mats=[mat(h,a,q,g) for h in seq]
            forwards=[mp.matrix([1,1])]
            for m in mats: forwards.append(m*forwards[-1])
            backs=[None]*(n+1); backs[n]=mp.matrix([1,1])
            for t in range(n-1,-1,-1): backs[t]=mats[t].T*backs[t+1]
            total=mp.fsum(forwards[-1]); probs=[mp.mpf(0),mp.mpf(0)]
            for t in range(n):
                vals=[backs[t+1][1]*q*forwards[t][1]/total,backs[t+1][0]*g*forwards[t][1]/total]
                for j,val in enumerate(vals):
                    check('edge probability in [0,1]',max(-val,val-1),mp.mpf('1e-60'))
                    probs[j]+=val/n
            def finite(x,y): return mp.log(mp.fsum(product(seq,a,mp.exp(x),mp.exp(y))))/n
            x=mp.log(q); y=mp.log(g)
            for j in range(2):
                ff=(lambda z:finite(z,y)) if j==0 else (lambda z:finite(x,z))
                z=x if j==0 else y
                error=abs(mp.diff(ff,z)-probs[j])
                maximum('finite_gradient_error',error)
                check('edge derivative identity',error,mp.mpf('1e-60'))
                check('finite convexity',-mp.diff(ff,z,2),mp.mpf('1e-60'))
    for a in ASTAYS:
        p=1-1/a
        for h in map(mp.mpf,['0','1e-30','1','1e30']):
            u=p; v=mp.mpf('.01'); w=h/a; z=mp.mpf('.99')
            for t in map(mp.mpf,['1e-30','.001','1','1000','1e30']):
                val=abs((u*z-v*w)*t/((u*t+v)*(w*t+z)))
                bound=1 if h==0 else abs(mp.sqrt(u*z)-mp.sqrt(v*w))/(mp.sqrt(u*z)+mp.sqrt(v*w))
                check('projective contraction',val-bound,mp.mpf('1e-60'))
            if h>0:
                tstar=mp.sqrt(v*z/(u*w))
                val=abs((u*z-v*w)*tstar/((u*tstar+v)*(w*tstar+z)))
                check('sharp contraction formula',abs(val-bound),mp.mpf('1e-60'))


class Law:
    def __init__(self,name,q0,phi,kind,par):
        self.name=name; self.q0=q0; self.kind=kind; self.par=par
        self.pi=np.array([q0,1-q0]); self.phi=phi
        self.P=np.array([[q0+(1-q0)*phi,(1-q0)*(1-phi)],[q0*(1-phi),1-q0+q0*phi]])
    def positive_moment(self,k):
        mean=1/(1-self.q0)
        if self.kind=='fixed': return mean**k
        if self.kind=='gamma': return mean**k*math.prod(self.par+j for j in range(k))/self.par**k
        if self.kind=='lognormal': return mean**k*math.exp((k*k-k)*self.par**2/2)
        alpha=self.par; x0=mean*(alpha-1)/alpha
        return alpha*x0**k/(alpha-k)
    def A(self,a):
        return ((1-self.q0)*self.positive_moment(2)-1)/2+self.q0/(1-self.q0)*self.phi/(1-(1-1/a)*self.phi)
    def nodes(self,n):
        mean=1/(1-self.q0)
        if self.kind=='fixed': return np.array([mean]),np.array([1.]),None
        if self.kind=='gamma':
            nodes,weights=roots_genlaguerre(n,self.par-1)
            return nodes*mean/self.par,weights/math.gamma(self.par),None
        if self.kind=='lognormal':
            nodes,weights=roots_hermitenorm(n)
            return mean*np.exp(self.par*nodes-self.par**2/2),weights/math.sqrt(2*math.pi),None
        nodes,weights=roots_legendre(n); w=(nodes+1)/2; weights=weights/2
        alpha=self.par; x0=mean*(alpha-1)/alpha
        return x0/w**2,weights*2*alpha*w**(2*alpha-1),(w,x0)


def interp(nodes,weights,points):
    diff=points[...,None]-nodes
    exact=np.abs(diff)<2e-15
    ker=weights/np.where(exact,1.,diff)
    val=ker/ker.sum(axis=-1,keepdims=True)
    return np.where(exact,1.,np.where(exact.any(axis=-1,keepdims=True),0.,val))


def poisson_growth(law,a,e,d,m,nq):
    b=1/a; p=1-b; vmax=b/(b-e); c=math.exp(d)
    idx=np.arange(m); y=(1-np.cos(np.pi*idx/(m-1)))/2
    bary=(-1.)**idx; bary[[0,-1]]*=.5
    vs=vmax*y*y
    f0=(p*vs+b)/(1-e)
    L0=interp(y,bary,np.sqrt(f0/vmax))
    h,ww,extra=law.nodes(nq)
    den=1+e*(c*vs[:,None]*h-1)
    f1=(p*vs[:,None]+b)/den
    L1=np.einsum('iqj,q->ij',interp(y,bary,np.sqrt(f1/vmax)),ww)
    reward0=np.full(m,math.log1p(-e))
    if extra is None:
        reward1=np.log1p(e*(c*vs[:,None]*h-1))@ww
    else:
        w,x0=extra
        reward1=np.log((1-e)*w*w+e*c*x0*vs[:,None])@ww+1/law.par
        reward1[0]=math.log1p(-e)
    P=law.P
    K=np.block([[P[0,0]*L0,P[0,1]*L0],[P[1,0]*L1,P[1,1]*L1]])
    sys=np.eye(2*m)-K; sys[:,0]=1
    return float(np.linalg.solve(sys,np.r_[reward0,reward1])[0])


def quad_root(law,a,e,m,n):
    f=lambda d:poisson_growth(law,a,e,d,m,n)
    return brentq(f,-.25,.25,xtol=2e-14,rtol=2e-14)


def fivepoint(f,x,step):
    return (f(x-2*step)-8*f(x-step)+8*f(x+step)-f(x+2*step))/(12*step)


def quad_derivative(law,a,e,d,m,n,divisor=50):
    re=fivepoint(lambda ee:poisson_growth(law,a,ee,d,m,n),e,e/divisor)
    rd=fivepoint(lambda dd:poisson_growth(law,a,e,dd,m,n),d,.001)
    return -re/rd


def quadrature_audit():
    print('Quadrature known-result gates',flush=True)
    for e in [.01,.002,.0005]:
        for law in [Law('constant',0,0,'fixed',1),Law('alternating',.5,-1,'fixed',1)]:
            d=quad_root(law,2,e,40,80); de=quad_derivative(law,2,e,d,40,80)
            exact,exactde=root_and_derivative([1] if law.q0==0 else [2,0],mp.mpf(2),mp.mpf(e))
            check('quadrature known D',abs(d-float(exact)),2e-11)
            check('quadrature known derivative',abs(de-float(exactde)),2e-6)
    if FAILURES: return []
    laws=[Law('iid_exp',.4,0,'gamma',1),Law('switch_gamma8',.5,-.96,'gamma',8),
          Law('iid_pareto45',.5,0,'pareto',4.5),Law('switch_pareto45',.5,-.96,'pareto',4.5),
          Law('iid_lognormal',.3,0,'lognormal',.8)]
    cells=[(law,2.) for law in laws]+[(laws[1],a) for a in [1.5,4.]]
    for law in laws:
        for n in [80,128]:
            h,w,_=law.nodes(n)
            for k in [0,1,2,4]:
                target=law.positive_moment(k)
                error=abs(float(w@(h**k))-target)/max(1,target)
                maximum('quadrature_moment_relative_error',error)
                check('unbounded quadrature moment',error,2e-10)
    rows=[]
    print('Unbounded-law calculations: 7 cells x 3 epsilons',flush=True)
    for law,a in cells:
        As=law.A(a); des=[]
        for e in [.01,.002,.0005]:
            dc=quad_root(law,a,e,40,80); df=quad_root(law,a,e,64,128)
            dec=quad_derivative(law,a,e,dc,40,80)
            def_=quad_derivative(law,a,e,df,64,128)
            dehalf=quad_derivative(law,a,e,df,64,128,100)
            maximum('unbounded_D_refinement',abs(dc-df))
            maximum('unbounded_derivative_refinement',abs(dec-def_))
            maximum('unbounded_derivative_step_refinement',abs(def_-dehalf))
            check('unbounded D refinement',abs(dc-df),2e-10)
            check('unbounded derivative refinement',abs(dec-def_),2e-6)
            check('unbounded derivative step halving',abs(def_-dehalf),2e-6)
            rows.append(dict(law=law.name,a=a,e=e,A=As,D=df,D_e=dehalf,
                D_refinement=abs(dc-df),derivative_refinement=abs(dec-def_),step_refinement=abs(def_-dehalf)))
            des.append(dehalf)
            print(' ',law.name,'a=',a,'e=',e,'A=',f'{As:.9g}','D_e=',f'{dehalf:.9g}',flush=True)
        extrap=(.002*des[-1]-.0005*des[-2])/(.002-.0005)
        maximum('unbounded_derivative_extrapolation_error',abs(extrap-As))
        check('unbounded extrapolation to A',abs(extrap-As),2e-3)
        check('unbounded derivative sign at e=.0005',-As*des[-1],0.)
    return rows


def main():
    start=time.monotonic()
    print('A13i deterministic checks; protocol.md tolerances fixed before run.',flush=True)
    print('Precision:',mp.mp.dps,'decimal digits; numpy',np.__version__,flush=True)
    benchmark()
    if not FAILURES:
        periodic_audit(); finite_gradient_audit()
    rows=quadrature_audit() if not FAILURES else []
    result=dict(checks=COUNT,failures=FAILURES,metrics=METRICS,unbounded=rows,seconds=time.monotonic()-start)
    (HERE/'checks_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('METRICS',json.dumps(METRICS,sort_keys=True),flush=True)
    print('CHECKS',COUNT,'FAILURES',len(FAILURES),'SECONDS',round(result['seconds'],3),flush=True)
    if FAILURES: raise SystemExit(1)


if __name__=='__main__':
    main()
