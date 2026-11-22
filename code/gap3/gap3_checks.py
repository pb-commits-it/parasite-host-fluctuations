"""Check the finite-noise long-stay threshold expansion for two unbounded hosts.

Protocol and tolerances: protocol.md, written before the campaign. The nonlinear
ratio recursion is compiled from the C string below; no predicted coefficient is
used to compute growth or the numerical root. Python independently evaluates
closed-form moments, gamma quadrature, and normalized matrix products.
Run: python3 -I gap3/gap3_checks.py
Writes gap3_results.json beside this script; stdout is the complete run record.
"""
from pathlib import Path
import ctypes
import json
import math
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy import integrate, optimize, signal, special

ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
EPS = np.array([.02, .01, .005, .0025])
REPS, STEPS, BURN = 32, 131072, 1024
FAILS = []
C_SOURCE = r'''
#include <math.h>
void growth(const double *h, int nt, int nr, int nk, int burn,
            const double *eps, const double *d, double *out, double *der) {
  for(int j=0;j<nr;j++) for(int k=0;k<nk;k++) {
    int at=j*nk+k; double e=eps[k],q=exp(d[at]),v=0.,dy=0.,acc=0.,jac=0.;
    for(int t=0;t<nt;t++) {
      double hh=h[t*nr+j], old=1.+v;
      double nv=e*(q*hh/old-1.);
      double nd=(nv+e)/(1.+nv)*(1.-dy);
      v=nv; dy=nd;
      if(t>=burn) {acc+=log1p(v)-e*q*(hh-1.); jac+=dy-e*q*(hh-1.);}
    }
    out[at]=acc/(nt-burn); der[at]=jac/(nt-burn);
  }
}
'''

def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def coefficients(model, phi, var=.5):
    if model == 'gamma':
        m3=2*var**2
        c1,c2=var*phi,var*phi**2
        t12=t21=phi*m3
        t111=phi**2*m3
        ell=math.log(1/var)-special.digamma(1/var)
    elif model == 'binary':
        m3=t12=t21=t111=0.
        c1,c2=var*phi,var*phi**2
        ell=-.5*math.log(1-var)
    else:
        w=math.log1p(var)
        c1,c2=math.expm1(w*phi),math.expm1(w*phi**2)
        m3=math.exp(3*w)-3*math.exp(w)+2
        t12=t21=math.exp(w*(1+2*phi))-2*math.exp(w*phi)-math.exp(w)+2
        t111=math.exp(w*(2*phi+phi**2))-2*math.exp(w*phi)-math.exp(w*phi**2)+2
        ell=w/2
    a=var/2+c1
    b=-a-c2-t12-t21-t111-m3/3+1.5*a*a
    return float(a),float(b),float(ell)


def gamma_rate(s,c,p):
    # X=c*y/(s*p): the invariant density in y is proportional to
    # y^(p-1) exp(-y) (1+c*y/(s*s*p))^(-p). No narrow log-X window.
    z=c/(s*s*p)
    def weight(y):
        if y==0:
            return 0.
        return math.exp((p-1)*math.log(y)-y-p*math.log1p(z*y)-special.gammaln(p))
    den=integrate.quad(weight,0,np.inf,epsabs=2e-13,epsrel=2e-13)[0]
    num=integrate.quad(lambda y: weight(y)*math.log1p(z*y),0,np.inf,
                       epsabs=2e-13,epsrel=2e-13)[0]
    return math.log(s)+num/den


def gamma_root(e,p=2.):
    return optimize.brentq(lambda d: gamma_rate(1-e,e*math.exp(d),p),0,2,
                           xtol=5e-15)


def environments(model,phi,seed):
    rng=np.random.default_rng(seed)
    n=STEPS+BURN
    if model=='gamma':
        fresh=rng.gamma(2.,.5,(n,REPS))
        refresh=rng.random((n,REPS))>=phi
        refresh[0]=True
        idx=np.maximum.accumulate(np.where(refresh,np.arange(n)[:,None],0),axis=0)
        h=np.take_along_axis(fresh,idx,axis=0)
    else:
        z0=rng.standard_normal(REPS)
        z=signal.lfilter([math.sqrt(1-phi*phi)],[1.,-phi],
                         rng.standard_normal((n,REPS)),axis=0,zi=(phi*z0)[None,:])[0]
        w=math.log(1.5)
        h=np.exp(math.sqrt(w)*z-w/2)
    return np.ascontiguousarray(h)


def mean_se(x):
    return x.mean(axis=0), x.std(axis=0,ddof=1)/math.sqrt(len(x))


def make_lib(folder):
    src=Path(folder)/'growth.c'
    lib=Path(folder)/'growth.so'
    src.write_text(C_SOURCE)
    subprocess.run(['cc','-O3','-std=c99','-shared','-fPIC',str(src),'-o',str(lib),'-lm'],check=True)
    loaded=ctypes.CDLL(str(lib))
    f=loaded.growth
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr,ptr]
    f.restype=None
    return loaded,f


def solve(h,fn,eps=EPS,burn=BURN):
    d=np.zeros((h.shape[1],len(eps)))
    out=np.zeros_like(d);jac=np.zeros_like(d)
    for it in range(8):
        fn(h,len(h),h.shape[1],len(eps),burn,eps,d,out,jac)
        if np.max(np.abs(out)/eps)<1e-11:
            return d, float(np.max(np.abs(out)/eps)),it+1
        d-=out/jac
    raise RuntimeError('registered root-iteration budget exceeded')


def structural_checks():
    # Derivative of the positive two-step Möbius map versus analytic supremum.
    worst=0.
    for s in [.15,.7,.99]:
        for c in [.003,.2,4.]:
            for h in [.001,.4,2.,1000.]:
                for hp in [.01,1.,30.]:
                    u=np.exp(np.linspace(-20,20,1001))
                    aa=s*s+c*h;bb=s*c*hp;cc=s;dd=c*hp
                    derivative=u*(aa*dd-bb*cc)/((aa*u+bb)*(cc*u+dd))
                    bound=(math.sqrt(s*s+c*h)-s)/(math.sqrt(s*s+c*h)+s)
                    worst=max(worst,float(np.max(derivative)-bound))
    # 2026-11-15 audit: corrected count label; original campaign output retained.
    check('two-step derivative bound (108 grids)',worst<=1e-13,f'max excess={worst:.3g}')
    # Independently differentiate finite constant runs and geometrically weighted
    # Gaussian conditional pasts. Values need not stay near one.
    lows=[]
    for phi in [0.,.5,.9]:
        for z in np.linspace(-4,4,17):
            u=1.; du=0.
            for lag in range(99,-1,-1):
                h=math.exp(.7*phi**lag*z+.2*math.sin(lag))
                a=.6*h/(.4*u+.6*h)
                du=a*(.7*phi**lag-du)
                u=.4+.6*h/u
            lows.append(du)
    check('Gaussian backward-conditional order (51 paths)',min(lows)>=-1e-13,
          f'min derivative={min(lows):.6g}')
    lows=[]
    for h in np.exp(np.linspace(-4,4,17)):
        u=1.;q=0.
        for n in range(1,101):
            a=.6*h/(.4*u+.6*h);q=a*(1-q);u=.4+.6*h/u
            lows.append(q)
    check('refresh constant-run order (1700 iterates)',min(lows)>0 and max(lows)<1,
          f'range={min(lows):.6g},{max(lows):.6g}')
    # Lognormal negative memory: moment evaluation by Gaussian quadrature,
    # independent of the closed-form exponential moment expression.
    nodes,weights=special.roots_hermitenorm(50);weights/=math.sqrt(2*math.pi)
    w=.1;phi=-.75
    x=nodes[:,None];y=phi*x+math.sqrt(1-phi*phi)*nodes[None,:]
    hh=np.exp(math.sqrt(w)*x-w/2);hp=np.exp(math.sqrt(w)*y-w/2)
    c1=np.sum(weights[:,None]*weights[None,:]*(hh-1)*(hp-1))
    a=.5*math.expm1(w)+c1
    ref=.5*math.expm1(w)+math.expm1(w*phi)
    check('negative-memory counterexample coefficient',a<0 and abs(a-ref)<1e-12,
          f'sigma^2=.1 phi=-.75 A={a:.12g}; error={abs(a-ref):.3g}')
    # Trace lower bound underlying nonconvergence. Jacobi matrix, not recurrence.
    edges=np.array([.13,.8,1.7,4.2,.5,2.1])
    J=np.diag(np.sqrt(edges),1)+np.diag(np.sqrt(edges),-1)
    ev=np.linalg.eigvalsh(J)
    for k in range(1,9):
        moment=np.sum(ev**(2*k))/2
        check(f'Jacobi trace moment lower bound k={k}',
              moment+1e-10>=np.sum(edges**k),f'ratio={moment/np.sum(edges**k):.6g}')
    e=.02;z=.03
    q0,q1=1.,1.
    for h in edges:
        q0,q1=q1,q1+z*h*q0
    spectral=np.sum(np.log1p(z*ev[ev>1e-10]**2))
    check('continuant equals Jacobi log determinant',abs(math.log(q1)-spectral)<1e-12,
          f'error={abs(math.log(q1)-spectral):.3g}')


def matrix_check(h):
    s=.83;c=.23
    vector=np.array([1.,1.]);u=1.;lr=0.;lm=0.
    for hh in h[:4000,0]:
        u=s+c*hh/u;lr+=math.log(u)
        vector=np.array([s*vector[0]+c*hh*vector[1],vector[0]])
        scale=vector.sum();lm+=math.log(scale);vector/=scale
    # x_0=x_-1=1: normalized initial vector was deliberately not divided by 2.
    endpoint=math.log1p(1/u)
    err=abs(lm-lr-endpoint)
    check('matrix/ratio endpoint identity',err<1e-10,f'absolute discrepancy={err:.3g}')


def main():
    started=time.monotonic()
    print('gap3-1 protocol campaign; numpy',np.__version__,flush=True)
    print(f'REPS={REPS} STEPS={STEPS} BURN={BURN} epsilon={EPS.tolist()}',flush=True)
    # Reproduce established coefficients before new simulation results are used.
    err=0.
    for v in [.04,.25,.64]:
        for ph in [-.75,0,.5,.9]:
            a,b,_=coefficients('binary',ph,v)
            err=max(err,abs(a-v*(.5+ph)),abs(b-(-v*(.5+ph+ph*ph)+1.5*a*a)))
    check('known binary P5 coefficients (12 cases)',err<1e-12,f'max error={err:.3g}')
    err=0.
    for p in [.3,1.,2.,10.]:
        a,b,_=coefficients('gamma',0,1/p)
        err=max(err,abs(a-1/(2*p)),abs(b+(12*p+7)/(24*p*p)))
    check('known iid gamma R6 coefficients (4 cases)',err<1e-12,f'max error={err:.3g}')
    ref=gamma_root(.5)
    check('gamma quadrature known rounded threshold',abs(ref-.0849)<5e-5,f'D(.5,p=2)={ref:.12g}')
    exact=np.array([gamma_root(e) for e in EPS])
    print('iid gamma exact near-one roots:',exact.tolist(),flush=True)
    structural_checks()
    if FAILS:
        print('STOP: prerequisite checks failed',FAILS)
        return 1
    results=[]
    with tempfile.TemporaryDirectory(prefix='gap3-check-') as tmp:
        library,fn=make_lib(tmp)
        const=np.ones((4000,1))
        d,res,it=solve(const,fn,burn=100)
        check('constant-host compiled threshold',np.max(np.abs(d))<1e-11,
              f'max |D|={np.max(np.abs(d)):.3g}')
        if FAILS:
            return 1
        for cell,(model,phi) in enumerate((m,p) for m in ['gamma','lognormal'] for p in [0.,.5,.9]):
            print(f'BEGIN cell {cell+1}/6 {model} phi={phi}',flush=True)
            h=environments(model,phi,2026111500+cell)
            if cell==0:
                matrix_check(h)
                if FAILS:
                    return 1
            d,res,it=solve(h,fn)
            means,ses=mean_se(d)
            if cell==0:
                check('iid gamma simulation known-result gate (4 roots)',
                      np.all(np.abs(means-exact)<=5*ses+2e-7),
                      f'max z={np.max(np.abs(means-exact)/ses):.4g}')
                if FAILS:
                    return 1
            a,b,ell=coefficients(model,phi)
            # Columns are powers; each row has a replicate's q(epsilon) values.
            coeff=np.linalg.solve(np.vander(EPS,4,increasing=True), (d/EPS).T).T
            cm,cs=mean_se(coeff)
            ah,bh=float(cm[0]),float(cm[1]);ase,bse=float(cs[0]),float(cs[1])
            aok=abs(ah-a)<=5*ase+.01*max(abs(a),.1)
            bok=abs(bh-b)<=5*bse+.03*max(abs(b),.1)
            precise=ase<=.01*max(abs(a),.1) and bse<=.03*max(abs(b),.1)
            check(f'{model} phi={phi} A',aok,f'pred={a:.12g} estimate={ah:.12g} SE={ase:.5g}')
            check(f'{model} phi={phi} B',bok,f'pred={b:.12g} estimate={bh:.12g} SE={bse:.5g}')
            check(f'{model} phi={phi} precision',precise,f'A relSE={ase/max(abs(a),.1):.5g}, B relSE={bse/max(abs(b),.1):.5g}')
            check(f'{model} phi={phi} strict bounds on grid',
                  np.all(means-5*ses>0) and np.all(means+5*ses<ell),f'ell={ell:.12g}')
            print(f'root scaled residual={res:.3g}; iterations={it}; largest host={h.max():.6g}',flush=True)
            print('epsilon D SE D/epsilon (D-Aepsilon)/epsilon^2 residual/epsilon^3 SE(residual)',flush=True)
            rows=[]
            for e,dd,se in zip(EPS,means,ses):
                rr=(dd-a*e-b*e*e)/e**3
                print(f'{e:.5g} {dd:.12g} {se:.5g} {dd/e:.9g} {(dd-a*e)/e**2:.9g} {rr:.8g} {se/e**3:.5g}',flush=True)
                rows.append(dict(epsilon=float(e),D=float(dd),SE=float(se),scaled_remainder=float(rr),remainder_SE=float(se/e**3)))
            results.append(dict(model=model,phi=phi,A=a,B=b,ell=ell,Ahat=ah,A_SE=ase,Bhat=bh,B_SE=bse,
                                rows=rows,root_residual=res,iterations=it,max_host=float(h.max()),
                                tests=dict(A=aok,B=bok,precision=precise)))
            del h
    print('END one registered campaign; seconds=',round(time.monotonic()-started,2),flush=True)
    print('FAILURES:',FAILS,flush=True)
    (ROOT/'gap3_results.json').write_text(json.dumps(dict(results=results,failures=FAILS,
        seconds=time.monotonic()-started,replicates=REPS,steps=STEPS,burn=BURN),indent=2))
    return int(bool(FAILS))

def deterministic_audit():
    """Post-campaign algebra/solver audit; see dated protocol addendum."""
    edges=np.array([.13,.8,1.7,4.2,.5,2.1])
    J=np.diag(np.sqrt(edges),1)+np.diag(np.sqrt(edges),-1)
    ev=np.linalg.eigvalsh(J)
    positives=ev[ev>1e-10]**2
    explicit=[np.sum(edges),
              np.sum(edges**2)+2*np.sum(edges[:-1]*edges[1:]),
              np.sum(edges**3)+3*np.sum(edges[:-1]**2*edges[1:]+edges[:-1]*edges[1:]**2)
              +3*np.sum(edges[:-2]*edges[1:-1]*edges[2:])]
    for k,expected in enumerate(explicit,1):
        error=abs(np.sum(positives**k)-expected)
        check(f'finite spectral coefficient k={k}',error<=1e-11,f'error={error:.5g}')
    for z in [.0025,.02,.1,1.]:
        actual=np.sum(np.log1p(z*positives))
        approx=sum((-1)**(k+1)*z**k*np.sum(positives**k)/k for k in range(1,4))
        bound=z**4*np.sum(positives**4)/4
        check(f'spectral log remainder z={z}',abs(actual-approx)<=bound+1e-12,
              f'error={abs(actual-approx):.6g} bound={bound:.6g}')
    with tempfile.TemporaryDirectory(prefix='gap3-audit-') as tmp:
        library,fn=make_lib(tmp)
        h=np.ones((4000,1)); d=np.full((1,len(EPS)),.2)
        out=np.zeros_like(d);jac=np.zeros_like(d)
        fn(h,len(h),1,len(EPS),100,EPS,d,out,jac)
        s=1-EPS;c=EPS*np.exp(d[0])
        lam=(s+np.sqrt(s*s+4*c))/2
        err=float(np.max(np.abs(out[0]-np.log(lam))))
        check('nonzero constant-host compiled growth',err<=1e-11,f'error={err:.5g}')
        mm=[]
        for ss,cc in zip(s,c):
            vals=np.linalg.eigvals(np.array([[ss,cc],[1.,0.]]))
            mm.append(math.log(max(abs(vals))))
        err=float(np.max(np.abs(out[0]-mm)))
        check('nonzero constant-host matrix exponent',err<=1e-11,f'error={err:.5g}')
        pp=np.zeros_like(out);pm=np.zeros_like(out);unused=np.zeros_like(out)
        fn(h,len(h),1,len(EPS),100,EPS,d+1e-6,pp,unused)
        fn(h,len(h),1,len(EPS),100,EPS,d-1e-6,pm,unused)
        err=float(np.max(np.abs((pp-pm)/2e-6-jac)))
        check('compiled root derivative finite difference',err<=1e-8,f'error={err:.5g}')
    print('AUDIT FAILURES:',FAILS)
    return int(bool(FAILS))


if __name__=='__main__':
    raise SystemExit(deterministic_audit() if '--audit' in sys.argv else main())
