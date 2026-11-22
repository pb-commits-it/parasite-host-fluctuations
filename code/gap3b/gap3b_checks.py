#!/usr/bin/env python3
"""Numerically check the two GAP 3b claims; protocol.md predates all runs.

Claim (i): the explicit fourth-moment constants enclose D and D'.
Claim (ii): D->ell and log(1/s)*(ell-D)->Var(log h)*(1-phi)/(2*(1+phi)).
Deterministic checks use mpmath; correlated nonlinear paths use native MPFR.
No other project module is imported. RNG inputs are binary64; propagation is
192-bit. Sampling uncertainty is reported separately from arithmetic precision.
Output JSON is written incrementally; stdout is the human-readable run record.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import mpmath as mp
import numpy as np
from scipy import signal

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
mp.mp.dps = 60
FAILS = []
RECORDS = []
PASSES = 0
RESULT_FILE = HERE/'gap3b_results.json'


def fmt(x, n=32):
    return mp.nstr(x, n)


def check(name, ok, detail=''):
    global PASSES
    PASSES += bool(ok)
    if not ok:
        FAILS.append(name)
    print(('PASS ' if ok else 'FAIL ') + name + ' ' + str(detail), flush=True)


def save(record):
    RECORDS.append(record)
    RESULT_FILE.write_text(json.dumps(
        dict(passed=PASSES, failed=FAILS, records=RECORDS), indent=2)+'\n')


def softplus(x):
    return max(x, mp.mpf(0))+mp.log1p(mp.exp(-abs(x)))


def constants(model, par, phi):
    par, phi = mp.mpf(par), mp.mpf(phi)
    if model == 'gamma':
        V = 1/par
        A = V*(mp.mpf('.5')+phi)
        B = -V*(mp.mpf('.5')+phi+phi**2)-2*V**2*(mp.mpf(1)/3+2*phi+phi**2)+mp.mpf('1.5')*A**2
        h4 = (par+1)*(par+2)*(par+3)/par**3
        ell = mp.log(par)-mp.digamma(par)
        v = mp.polygamma(1, par)
    else:
        w = par
        V = mp.expm1(w)
        C1, C2 = mp.expm1(w*phi), mp.expm1(w*phi**2)
        m3 = mp.exp(3*w)-3*mp.exp(w)+2
        T = mp.exp(w*(1+2*phi))-mp.exp(w)-2*mp.exp(w*phi)+2
        U = mp.exp(w*(2*phi+phi**2))-2*mp.exp(w*phi)-mp.exp(w*phi**2)+2
        A = V/2+C1
        B = -A-C2-2*T-U-m3/3+mp.mpf('1.5')*A**2
        h4, ell, v = mp.exp(6*w), w/2, w
    a = 3+2*A
    b = 3*(mp.mpf('1.5')*A**2+5*A+mp.mpf(10)/3-B)
    out = derivative_constants(a, b, 128*h4)
    out.update(A=A, B=B, ell=ell, v=v, Vpi=v*(1-phi)/(1+phi), h4=h4)
    out['epsA'] = min(out['eps0'], abs(A)/(8*(abs(B)+1)), mp.sqrt(abs(A)/(2*out['K']))) if A else mp.mpf(0)
    return out


def derivative_constants(a, b, d):
    z0 = min(mp.mpf('.25'), 1/(2*a))
    M2, M3 = 1+a, 1+3*a+2*b
    M4 = 1+6*a+3*a*a+8*b+6*d
    T1 = 3*M2/2+2
    T2 = M3+9*M2*M2/4+2*a
    T3 = 3*M4/4+9*M2*M3/2+27*M2**3/4+4*b
    K = 27*T3+243*T2*M2+T1*(729*M2*M2+81*M3)
    return dict(a=a, b=b, d=d, K=K, eps0=z0/3)


def gamma_growth(s, c, p):
    """Independent stationary-density integrals, with y=t^2 and tail t<=16.

    The omitted gamma tail is O(exp(-256)) in this registered parameter grid.
    Returns growth and its derivative in log c from differentiated integrals.
    """
    s, c, p = map(mp.mpf, (s, c, p))
    z = c/(p*s*s)
    def terms(t, which):
        if not t:
            return 2 if p == mp.mpf('.5') and which == 0 else mp.mpf(0)
        y = t*t
        f = mp.log1p(z*y)
        a = z*y/(1+z*y)
        w = 2*t**(2*p-1)*mp.exp(-y-p*f)
        return w*(1, f, a, f*a)[which]
    vals = [mp.quad(lambda t: terms(t,k), [0, mp.mpf('.5'), 1, 2, 4, 8, 16]) for k in range(4)]
    mean = vals[1]/vals[0]
    deriv = vals[2]/vals[0]-p*(vals[3]/vals[0]-mean*vals[2]/vals[0])
    return mp.log(s)+mean, deriv


def gamma_threshold(e, p):
    D = mp.mpf(0)
    s = 1-e
    for _ in range(8):
        r, g = gamma_growth(s, e*mp.exp(D), p)
        if abs(r) < mp.mpf('1e-48'):
            return D, -1/(s*g)+1/e+2/s, r
        D -= r/g
    raise ArithmeticError('deterministic gamma root did not converge')


def gamma_q(L, p):
    """Exact iid-gamma growth from the supplied Tricomi representation.

    m=0, s=exp(-L), c=exp(ell), kappa=exp(psi(p)-2L), q=2r.
    This formula contains none of the proposed small-staying coefficient.
    """
    L, p = mp.mpf(L), mp.mpf(p)
    kappa = mp.exp(mp.digamma(p)-2*L)
    return 2*(-L+mp.diff(lambda b: mp.log(mp.hyperu(p,b,kappa)), 1))


def tail_bound(L, model, par, phi):
    par, phi, L = map(mp.mpf, (par, phi, L))
    cc = constants(model, par, phi)
    if model == 'gamma':
        a = par/4
        def G(t):
            return mp.gamma(par+t)/mp.gamma(par)*mp.exp(-t*mp.digamma(par))
        Na = mp.exp(a)*(G(2*a)+G(-2*a))
    else:
        a = 1/(2*max(1, mp.sqrt(par)))
        Na = 2*mp.exp(a+2*a*a*par)
    K = max(2,4/a)*mp.log(L)
    Cg, Cb, Ct = 6+2/(1+phi), 5+2/(1+phi), 8+4/(1+phi)
    delta = (Cb*K+16*Ct*Na/a**2)/L**2
    if 2*L <= Cg*K:
        return None
    return max(0,(cc['Vpi']-delta)/(2*L+Cg*K)), (cc['Vpi']+delta)/(2*L-Cg*K)


def controls():
    # A constant-host ratio gives the exact Perron root for every s,c.
    s, c = mp.mpf('.4'), mp.mpf('.7')
    u = mp.mpf(1)
    for _ in range(800):
        u = s+c/u
    known = (s+mp.sqrt(s*s+4*c))/2
    check('constant host ratio / Perron root', abs(u-known)<mp.mpf('1e-45'), fmt(abs(u-known)))
    for p in (mp.mpf('.5'),mp.mpf(2)):
        s, c = mp.mpf('.5'),mp.mpf('.5')
        r, g = gamma_growth(s,c,p)
        exact = mp.log(s)+mp.diff(lambda b:mp.log(mp.hyperu(p,b,s*s*p/c)),1)
        check('known gamma density / Tricomi p='+str(p), abs(r-exact)<mp.mpf('1e-25'), fmt(abs(r-exact)))
        num = mp.diff(lambda t:mp.log(s)+mp.diff(lambda b:mp.log(mp.hyperu(p,b,s*s*p/mp.exp(t))),1),mp.log(c))
        check('known gamma growth derivative p='+str(p),abs(g-num)<mp.mpf('1e-25'),fmt(abs(g-num)))
    if FAILS:
        raise SystemExit('Known-result control failed; research grid not run.')


def deterministic():
    controls()
    # Fresh quadrature of gamma moments in log(ph); no inverse-moment shortcut.
    for p in (mp.mpf('.5'),mp.mpf(2)):
        psi = mp.digamma(p)
        def integral(fun):
            return mp.quad(lambda t:mp.exp(p*t-mp.exp(t)-mp.loggamma(p))*fun(t),
                           [-400,-100,-40,-10,-3,0,2,4,8])
        for k in (1,2,3,4):
            val=integral(lambda t:mp.exp(k*t)/p**k)
            target=mp.rf(p,k)/p**k
            err=abs(val-target)/max(1,abs(target))
            check(f'gamma positive moment p={p} k={k}',err<mp.mpf('1e-30'),fmt(err))
        mean=integral(lambda t:t-psi)
        var=integral(lambda t:(t-psi)**2)
        check(f'gamma centered log moments p={p}',max(abs(mean),abs(var-mp.polygamma(1,p)))<mp.mpf('1e-30'),fmt(abs(var-mp.polygamma(1,p))))

    # Gaussian innovation quadrature independently evaluates joint host moments.
    ghx, ghw=mp.gauss_quadrature(48,'hermite')
    def normal_exp(t):
        return mp.fsum(w*mp.exp(mp.sqrt(2)*t*x) for x,w in zip(ghx,ghw))/mp.sqrt(mp.pi)
    for w,phi in [(mp.mpf('.5'),mp.mpf(p)) for p in ('-.5','0','.5')]+[(mp.mpf('.1'),mp.mpf('-.75'))]:
        def joint(powers):
            # xi_j=phi^j Z0+sqrt(1-phi^2)sum_(k=1)^j phi^(j-k)Zk.
            n=len(powers)
            ts=[mp.sqrt(w)*sum(powers[j]*phi**j for j in range(n))]
            ts += [mp.sqrt(w*(1-phi*phi))*sum(powers[j]*phi**(j-k) for j in range(k,n)) for k in range(1,n)]
            return mp.exp(-w*sum(powers)/2)*mp.fprod(normal_exp(t) for t in ts)
        a=joint([2])+2*joint([1,1])
        b=joint([3])+3*(joint([2,1])+joint([1,2])+joint([1,1,1]))
        h4=joint([4])
        cc=constants('lognormal',w,phi)
        err=max(abs(a-cc['a']),abs(b-cc['b']),abs(h4-cc['h4']))
        check(f'Gaussian quadrature joint moments w={w} phi={phi}',err<mp.mpf('1e-30'),fmt(err))

    # A=0 curve exists in (-1,0) for this w. It exercises the separate B case.
    w=mp.mpf('.1')
    phi0=mp.log((3-mp.exp(w))/2)/w
    cc=constants('lognormal',w,phi0)
    check('lognormal A=0 cancellation',abs(cc['A'])<mp.mpf('1e-45'),f'phi={fmt(phi0)} B={fmt(cc["B"])}')
    epsB=min(cc['eps0'],2*abs(cc['B'])/cc['K'])
    check('A=0 B-sign interval arithmetic',cc['B']<0 and cc['K']*epsB/2<=abs(cc['B'])+mp.mpf('1e-45'),fmt(epsB))
    save(dict(kind='degenerate',phi=fmt(phi0),B=fmt(cc['B']),epsB=fmt(epsB)))

    # Independent finite-path determinant and spectral calculations.
    hs=[mp.mpf(x) for x in ('.2','2','.7','3','.4','1.1','5')]
    J=mp.matrix(8)
    for i,h in enumerate(hs):J[i,i+1]=J[i+1,i]=mp.sqrt(h)
    eig=mp.eigsy(J,eigvals_only=True)
    for z in (mp.mpf('.001'),mp.mpf('.2'),mp.mpf(10)):
        q0=q1=mp.mpf(1)
        for h in hs:q0,q1=q1,q1+z*h*q0
        det=mp.fprod(1+z*v*v for v in eig if v>0)
        check('finite path spectral determinant z='+str(z),abs(det/q1-1)<mp.mpf('1e-40'),fmt(abs(det/q1-1)))

    # A scalar spectral measure with mu1=1 separately checks inverse calculus.
    xs=list(map(mp.mpf,('.1','1','4.9')))
    F=lambda z:mp.fsum(mp.log1p(z*x) for x in xs)/6
    ee=lambda z:-mp.expm1(-F(z))
    HH=lambda z:mp.log(z/ee(z))-2*F(z)
    a,b,d=[mp.fsum(x**j for x in xs)/6 for j in (2,3,4)]
    cc=derivative_constants(a,b,d)
    A=(a-3)/2;B=mp.mpf('1.5')*A*A+5*A+mp.mpf(10)/3-b/3
    for fraction in ('1','.1','.001'):
        e=cc['eps0']*mp.mpf(fraction)
        z=mp.findroot(lambda z:ee(z)-e,(e,2*e))
        e1,e2,e3=[mp.diff(ee,z,k) for k in (1,2,3)]
        H1,H2,H3=[mp.diff(HH,z,k) for k in (1,2,3)]
        chained=H3/e1**3-3*H2*e2/e1**4+H1*(3*e2**2/e1**5-e3/e1**4)
        direct=mp.diff(lambda v:HH(mp.findroot(lambda t:ee(t)-v,(z,z+mp.mpf('.001')))),e,3)
        check('inverse third derivative '+fraction,abs(chained-direct)<mp.mpf('1e-40'),fmt(abs(chained-direct)))
        check('spectral finite derivative bound '+fraction,abs(H1/e1-A-2*B*e)<=cc['K']*e*e/2 and abs(chained)<=cc['K'])

    # Pointwise boundary regions, with large positive and negative log hosts.
    maxslack=mp.mpf('-inf');count=0
    for phi in map(mp.mpf,('-.9','-.5','0','.5','.9')):
        Cg,Cb,Ct=6+2/(1+phi),5+2/(1+phi),8+4/(1+phi)
        for prev in map(mp.mpf,('-40','-2','0','1','30')):
            for eta in map(mp.mpf,('-40','-2','0','1','30')):
                R=1+abs(prev)+abs(eta)
                for L in map(mp.mpf,('1','8','128')):
                    for Y in [-2*L,-L,mp.mpf(-1),mp.mpf(0),max(prev,0)+mp.log(2)]:
                        Q=softplus(Y-eta)
                        integrand=abs((2*Y-2*eta/(1+phi)-Q)*Q)
                        maxslack=max(maxslack,abs(Y)*Q-2*R*R,Q-2*R,integrand-Ct*R*R)
                        for K in map(mp.mpf,('1','4','20','100')):
                            if R<=K:
                                bound=Cg*K*Q if Y>=-2*K else Cb*K*mp.exp(-K)
                                maxslack=max(maxslack,integrand-bound)
                        # Algebra before expectation, independent of stationarity.
                        X=L+Y;Xt=-X+eta+Q
                        resid=Xt**2-X**2-(eta**2-2*X*eta+2*(eta-X)*Q+Q**2)
                        maxslack=max(maxslack,abs(resid)-mp.mpf('1e-40'))
                        count+=1
    check(f'pointwise tail and square identities ({count} cases)',maxslack<=mp.mpf('1e-40'),fmt(maxslack))

    # Period-two counterexample: all moments exist but the endpoint differs.
    b=mp.mpf('.7');ell=mp.log(mp.cosh(b));hp=mp.exp(b-ell);hm=mp.exp(-b-ell)
    s=mp.mpf('1e-20')
    c=2*(1-s*s)/(hp+hm+mp.sqrt((hp+hm)**2-4*hp*hm*(1-s*s)))
    D=mp.log(c/(1-s))
    check('period-two endpoint counterexample',abs(D-(ell-b))<mp.mpf('2e-20') and abs(D-ell)>mp.mpf('.6'),fmt(D))

    for p in (mp.mpf('.5'),mp.mpf(2)):
        cc=constants('gamma',p,0)
        for tag,e in [('wide',cc['eps0']/2),('sign',cc['epsA']/2)]:
            D,Ds,res=gamma_threshold(e,p)
            rem=abs(D-cc['A']*e-cc['B']*e*e)
            der=abs(Ds+cc['A']+2*cc['B']*e)
            check(f'iid gamma explicit D remainder p={p} {tag}',rem<=cc['K']*e**3/6,fmt(rem))
            check(f'iid gamma explicit derivative p={p} {tag}',der<=cc['K']*e**2/2,fmt(der))
            if tag=='sign':check(f'iid gamma derivative sign p={p}',abs(Ds+cc['A'])<=abs(cc['A'])/2,fmt(Ds))
            save(dict(kind='gamma_derivative',p=str(p),tag=tag,epsilon=fmt(e),D=fmt(D),Ds=fmt(Ds),K=fmt(cc['K']),D_remainder=fmt(rem),derivative_remainder=fmt(der)))
        for L in (4,8,16,32,64,128,10000):
            q=gamma_q(L,p)
            scaled=2*L*q/cc['Vpi']
            print(f'GAMMA_RATE p={p} L={L} q={fmt(q)} scaled={fmt(scaled)}',flush=True)
            if L==128:check(f'iid gamma leading rate p={p}',abs(scaled-1)<mp.mpf('.1'),fmt(scaled))
            if L==10000:
                lo,hi=tail_bound(L,'gamma',p,0)
                check(f'iid gamma explicit tail interval p={p}',lo<=q<=hi,f'{fmt(lo)} <= {fmt(q)} <= {fmt(hi)}')
            save(dict(kind='gamma_rate',p=str(p),L=L,q=fmt(q),scaled=fmt(scaled)))


C_SOURCE = r'''
#include <mpfr.h>
#include <math.h>
#include <stddef.h>
#define R MPFR_RNDN
#define SZ 96
static void put(char *o,size_t k,mpfr_t x){mpfr_snprintf(o+SZ*k,SZ,"%.48Rg",x);}

void smallstay(const double *eta,int n,int nr,int burn,int bits,const char *ph,
               const int *Ls,int nl,char *out){
 mpfr_t x,xx,e,q,qq,tmp,m,d,sumq,summ,sumd,coef,phv;
 mpfr_inits2(bits,x,xx,e,q,qq,tmp,m,d,sumq,summ,sumd,coef,phv,(mpfr_ptr)0);
 mpfr_set_str(phv,ph,10,R);mpfr_add_ui(coef,phv,1,R);mpfr_ui_div(coef,2,coef,R);
 for(int j=0;j<nr;j++) for(int k=0;k<nl;k++){
  int L=Ls[k],coupled=0;
  mpfr_set_si(x,-L,R);mpfr_set_d(xx,L+fabs(eta[j*n])+1,R);
  mpfr_set_zero(sumq,0);mpfr_set_zero(summ,0);mpfr_set_zero(sumd,0);
  for(int t=0;t<n;t++){
   mpfr_set_d(e,eta[j*n+t],R);
   mpfr_sub(tmp,x,e,R);mpfr_sub_ui(tmp,tmp,L,R);mpfr_exp(q,tmp,R);mpfr_log1p(q,q,R);
   if(!coupled){
    mpfr_sub(tmp,xx,e,R);mpfr_sub_ui(tmp,tmp,L,R);mpfr_exp(qq,tmp,R);mpfr_log1p(qq,qq,R);
    if(t>=burn){mpfr_sub(d,q,qq,R);mpfr_abs(d,d,R);mpfr_add(sumd,sumd,d,R);}
    mpfr_sub(xx,e,xx,R);mpfr_add(xx,xx,qq,R);
   }
   if(t>=burn){
    mpfr_add(sumq,sumq,q,R);
    mpfr_mul_ui(m,x,2,R);mpfr_mul(tmp,e,coef,R);mpfr_sub(m,m,tmp,R);mpfr_sub(m,m,q,R);mpfr_mul(m,m,q,R);mpfr_add(summ,summ,m,R);
   }
   mpfr_sub(x,e,x,R);mpfr_add(x,x,q,R);
   if(!coupled){mpfr_sub(tmp,x,xx,R);mpfr_abs(tmp,tmp,R);if(mpfr_cmp_d(tmp,1e-45)<0)coupled=1;}
  }
  mpfr_div_ui(sumq,sumq,n-burn,R);mpfr_div_ui(summ,summ,n-burn,R);mpfr_div_ui(sumd,sumd,n-burn,R);
  size_t at=((size_t)j*nl+k)*3;put(out,at,sumq);put(out,at+1,summ);put(out,at+2,sumd);
 }
 mpfr_clears(x,xx,e,q,qq,tmp,m,d,sumq,summ,sumd,coef,phv,(mpfr_ptr)0);
}

void longstay(const double *hosts,int n,int nr,int burn,int bits,const char *es,char *out){
 mpfr_t ep,s,D,c,ed,v,u,t,fac,gd,ge,nd,ne,h,hm,tmp,r,rd,re,sr,sd,se,corr,ds;
 mpfr_inits2(bits,ep,s,D,c,ed,v,u,t,fac,gd,ge,nd,ne,h,hm,tmp,r,rd,re,sr,sd,se,corr,ds,(mpfr_ptr)0);
 mpfr_set_str(ep,es,10,R);mpfr_ui_sub(s,1,ep,R);
 for(int j=0;j<nr;j++){
  mpfr_set_zero(D,0);
  for(int it=0;it<6;it++){
   mpfr_exp(ed,D,R);mpfr_mul(c,ep,ed,R);
   mpfr_set_zero(v,0);mpfr_set_zero(gd,0);mpfr_set_zero(ge,0);
   mpfr_set_zero(sr,0);mpfr_set_zero(sd,0);mpfr_set_zero(se,0);
   for(int k=0;k<n;k++){
    mpfr_set_d(h,hosts[j*n+k],R);mpfr_sub_ui(hm,h,1,R);
    mpfr_add_ui(u,v,1,R);mpfr_mul(t,c,h,R);mpfr_div(t,t,u,R);
    mpfr_sub(v,t,ep,R);mpfr_add_ui(u,v,1,R);mpfr_div(fac,t,u,R);
    mpfr_ui_sub(nd,1,gd,R);mpfr_mul(nd,nd,fac,R);
    mpfr_div(ne,t,ep,R);mpfr_sub_ui(ne,ne,1,R);mpfr_div(ne,ne,u,R);
    mpfr_mul(tmp,fac,ge,R);mpfr_sub(ne,ne,tmp,R);
    mpfr_set(gd,nd,R);mpfr_set(ge,ne,R);
    if(k>=burn){
     mpfr_mul(corr,c,hm,R);mpfr_log1p(r,v,R);mpfr_sub(r,r,corr,R);mpfr_add(sr,sr,r,R);
     mpfr_sub(rd,gd,corr,R);mpfr_add(sd,sd,rd,R);
     mpfr_mul(corr,ed,hm,R);mpfr_sub(re,ge,corr,R);mpfr_add(se,se,re,R);
    }
   }
   mpfr_div_ui(sr,sr,n-burn,R);mpfr_div_ui(sd,sd,n-burn,R);mpfr_div_ui(se,se,n-burn,R);
   mpfr_abs(tmp,sr,R);if(mpfr_cmp_d(tmp,1e-30)<0 || it==5)break;
   mpfr_div(tmp,sr,sd,R);mpfr_sub(D,D,tmp,R);
  }
  mpfr_div(ds,se,sd,R);put(out,j*3,D);put(out,j*3+1,ds);put(out,j*3+2,sr);
 }
 mpfr_clears(ep,s,D,c,ed,v,u,t,fac,gd,ge,nd,ne,h,hm,tmp,r,rd,re,sr,sd,se,corr,ds,(mpfr_ptr)0);
}

typedef struct {mpfr_t eta,u,gd,ge,weight,h,t,fac,tmp,r;} GHState;
typedef struct {int n,depth;mpfr_t z[32],wt[32],phi,sigma,innov,halfvar,ep,c,sr,sd,se;GHState st[8];} GH;
static void ghvisit(GH *g,int lev){
 GHState *p=&g->st[lev-1],*s=&g->st[lev];
 for(int j=0;j<g->n;j++){
  if(lev==1){mpfr_mul(s->eta,g->sigma,g->z[j],R);}
  else {mpfr_mul(s->eta,g->phi,p->eta,R);mpfr_mul(s->tmp,g->innov,g->z[j],R);mpfr_add(s->eta,s->eta,s->tmp,R);}
  mpfr_sub(s->h,s->eta,g->halfvar,R);mpfr_exp(s->h,s->h,R);
  mpfr_mul(s->t,g->c,s->h,R);mpfr_div(s->t,s->t,p->u,R);
  mpfr_sub(s->u,s->t,g->ep,R);mpfr_add_ui(s->u,s->u,1,R);
  mpfr_div(s->fac,s->t,s->u,R);
  mpfr_ui_sub(s->gd,1,p->gd,R);mpfr_mul(s->gd,s->gd,s->fac,R);
  mpfr_div(s->ge,s->t,g->ep,R);mpfr_sub_ui(s->ge,s->ge,1,R);mpfr_div(s->ge,s->ge,s->u,R);
  mpfr_mul(s->tmp,s->fac,p->ge,R);mpfr_sub(s->ge,s->ge,s->tmp,R);
  mpfr_mul(s->weight,p->weight,g->wt[j],R);
  if(lev<g->depth)ghvisit(g,lev+1);
  else {
   mpfr_log(s->r,s->u,R);mpfr_mul(s->tmp,s->weight,s->r,R);mpfr_add(g->sr,g->sr,s->tmp,R);
   mpfr_mul(s->tmp,s->weight,s->gd,R);mpfr_add(g->sd,g->sd,s->tmp,R);
   mpfr_mul(s->tmp,s->weight,s->ge,R);mpfr_add(g->se,g->se,s->tmp,R);
  }
 }
}
void gaussian_tree(int n,int depth,int bits,const char **zs,const char **ws,
                   const char *var,const char *ph,const char *eps,const char *ds,char *out){
 GH g;g.n=n;g.depth=depth;
 mpfr_inits2(bits,g.phi,g.sigma,g.innov,g.halfvar,g.ep,g.c,g.sr,g.sd,g.se,(mpfr_ptr)0);
 mpfr_set_str(g.phi,ph,10,R);mpfr_set_str(g.halfvar,var,10,R);mpfr_sqrt(g.sigma,g.halfvar,R);
 mpfr_sqr(g.innov,g.phi,R);mpfr_ui_sub(g.innov,1,g.innov,R);mpfr_mul(g.innov,g.innov,g.halfvar,R);mpfr_sqrt(g.innov,g.innov,R);
 mpfr_div_ui(g.halfvar,g.halfvar,2,R);mpfr_set_str(g.ep,eps,10,R);
 mpfr_set_str(g.c,ds,10,R);mpfr_exp(g.c,g.c,R);mpfr_mul(g.c,g.c,g.ep,R);
 mpfr_set_zero(g.sr,0);mpfr_set_zero(g.sd,0);mpfr_set_zero(g.se,0);
 for(int j=0;j<n;j++){mpfr_init2(g.z[j],bits);mpfr_init2(g.wt[j],bits);mpfr_set_str(g.z[j],zs[j],10,R);mpfr_set_str(g.wt[j],ws[j],10,R);}
 for(int j=0;j<=depth;j++){
  GHState *s=&g.st[j];mpfr_inits2(bits,s->eta,s->u,s->gd,s->ge,s->weight,s->h,s->t,s->fac,s->tmp,s->r,(mpfr_ptr)0);
 }
 mpfr_set_ui(g.st[0].u,1,R);mpfr_set_zero(g.st[0].gd,0);mpfr_set_zero(g.st[0].ge,0);mpfr_set_ui(g.st[0].weight,1,R);
 ghvisit(&g,1);put(out,0,g.sr);put(out,1,g.sd);put(out,2,g.se);
 for(int j=0;j<=depth;j++){GHState *s=&g.st[j];mpfr_clears(s->eta,s->u,s->gd,s->ge,s->weight,s->h,s->t,s->fac,s->tmp,s->r,(mpfr_ptr)0);}
 for(int j=0;j<n;j++){mpfr_clear(g.z[j]);mpfr_clear(g.wt[j]);}
 mpfr_clears(g.phi,g.sigma,g.innov,g.halfvar,g.ep,g.c,g.sr,g.sd,g.se,(mpfr_ptr)0);
}
'''


def compile_kernel(tmp):
    source=Path(tmp)/'kernel.c';target=Path(tmp)/'kernel.dylib'
    source.write_text(C_SOURCE)
    subprocess.run(['cc','-O3','-dynamiclib',str(source),'-o',str(target),
                    '-I/opt/homebrew/opt/mpfr/include','-I/opt/homebrew/opt/gmp/include',
                    '-L/opt/homebrew/opt/mpfr/lib','-L/opt/homebrew/opt/gmp/lib','-lmpfr','-lgmp'],check=True)
    lib=ctypes.CDLL(str(target))
    dptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.smallstay.argtypes=[dptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,iptr,ctypes.c_int,ctypes.c_void_p]
    lib.longstay.argtypes=[dptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_void_p]
    lib.smallstay.restype=lib.longstay.restype=None
    cpp=ctypes.POINTER(ctypes.c_char_p)
    lib.gaussian_tree.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,cpp,cpp,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_void_p]
    lib.gaussian_tree.restype=None
    return lib


def parse_buffer(buf,n):
    return [mp.mpf(buf.raw[96*k:96*(k+1)].split(b'\0')[0].decode()) for k in range(n)]


def rate_kernel(lib,eta,phi,burn,Ls=(8,16,32),bits=192):
    nr,n=eta.shape
    ls=np.asarray(Ls,dtype=np.int32)
    buf=ctypes.create_string_buffer(96*nr*len(Ls)*3)
    lib.smallstay(eta,n,nr,burn,bits,str(phi).encode(),ls,len(Ls),buf)
    vals=parse_buffer(buf,nr*len(Ls)*3)
    return [[vals[(j*len(Ls)+k)*3:(j*len(Ls)+k+1)*3] for k in range(len(Ls))] for j in range(nr)]


def long_kernel(lib,h,e,burn=128,bits=192):
    nr,n=h.shape
    buf=ctypes.create_string_buffer(96*nr*3)
    lib.longstay(h,n,nr,burn,bits,fmt(e,58).encode(),buf)
    vals=parse_buffer(buf,nr*3)
    return [vals[j*3:(j+1)*3] for j in range(nr)]


def environments(model,par,phi,nr,n,seed):
    rng=np.random.default_rng(seed)
    par,phi=float(par),float(phi)
    if model=='gamma':
        fresh=rng.gamma(par,1/par,(nr,n))
        refresh=rng.random((nr,n))>=phi;refresh[:,0]=True
        ix=np.maximum.accumulate(np.where(refresh,np.arange(n)[None,:],0),axis=1)
        h=np.take_along_axis(fresh,ix,axis=1)
        ell=float(mp.log(par)-mp.digamma(par))
        eta=np.log(h)+ell
    else:
        z=rng.standard_normal((nr,n));xi=np.empty_like(z);xi[:,0]=z[:,0]
        xi[:,1:],_=signal.lfilter([math.sqrt(1-phi*phi)],[1,-phi],z[:,1:],axis=1,zi=(phi*z[:,0])[:,None])
        eta=math.sqrt(par)*xi
        h=np.exp(eta-par/2)
    return np.ascontiguousarray(h),np.ascontiguousarray(eta)


def stats(values):
    n=len(values);mean=mp.fsum(values)/n
    se=mp.sqrt(mp.fsum((v-mean)**2 for v in values)/(n*(n-1)))
    return mean,se


CELLS=[('gamma','2','0'),('gamma','2','.5'),('gamma','.5','.5'),
       ('lognormal','.5','-.5'),('lognormal','.5','0'),('lognormal','.5','.5'),
       ('lognormal','.1','-.75')]


def gaussian_deterministic(refine=False):
    """Nonlinear Gaussian-history quadrature, independent of A/B in root finding."""
    controls()
    with tempfile.TemporaryDirectory(prefix='gap3b-gh-') as tmp:
        lib=compile_kernel(tmp)
        cache={}
        def root(w,phi,e,n,depth):
            if n not in cache:
                x,wt=mp.gauss_quadrature(n,'hermite')
                arr=ctypes.c_char_p*n
                cache[n]=(arr(*[fmt(mp.sqrt(2)*u,60).encode() for u in x]),
                          arr(*[fmt(v/mp.sqrt(mp.pi),60).encode() for v in wt]))
            z,weights=cache[n]
            D=mp.mpf(0)
            for iteration in range(6):
                buf=ctypes.create_string_buffer(96*3)
                lib.gaussian_tree(n,depth,224,z,weights,fmt(w,60).encode(),fmt(phi,60).encode(),fmt(e,60).encode(),fmt(D,60).encode(),buf)
                r,rd,re=parse_buffer(buf,3)
                if abs(r)<mp.mpf('1e-43'):
                    return D,re/rd,r,iteration+1
                D-=r/rd
            raise ArithmeticError('Gaussian tree root failed')
        if refine:
            previous=json.loads((HERE/'gaussian_quadrature_results.json').read_text())['records']
            for phi in (mp.mpf('-.5'),mp.mpf('.5')):
                cc=constants('lognormal',mp.mpf('.5'),phi)
                e=cc['epsA']/2;start=time.monotonic()
                D,Ds,res,it=root(mp.mpf('.5'),phi,e,12,6)
                old=next(r for r in previous if r.get('kind')=='gaussian_tree' and r['depth']==5 and r['phi']==fmt(phi))
                difference=abs(Ds-mp.mpf(old['Ds']))
                check('Gaussian depth-five/six follow-up '+str(phi),difference<=mp.mpf('1e-12'),fmt(difference))
                check('Gaussian follow-up derivative interval '+str(phi),abs(Ds+cc['A']+2*cc['B']*e)<=cc['K']*e*e/2+mp.mpf('1e-12'),fmt(Ds))
                save(dict(kind='gaussian_history_followup',phi=fmt(phi),D=fmt(D),Ds=fmt(Ds),depth=6,nodes=12,history_difference=fmt(difference),residual=fmt(res),newton_passes=it,seconds=time.monotonic()-start))
            return
        phi0=mp.log((3-mp.exp(mp.mpf('.1')))/2)/mp.mpf('.1')
        for w,phi,zero in [(mp.mpf('.5'),mp.mpf('-.5'),False),(mp.mpf('.5'),mp.mpf('.5'),False),(mp.mpf('.1'),phi0,True)]:
            cc=constants('lognormal',w,phi)
            e=(min(cc['eps0'],2*abs(cc['B'])/cc['K']) if zero else cc['epsA'])/2
            results=[];start=time.monotonic()
            for n,depth in ((12,4),(16,4),(12,5)):
                D,Ds,res,it=root(w,phi,e,n,depth)
                results.append((D,Ds))
                check(f'Gaussian tree root w={w} phi={fmt(phi,8)} nodes={n} depth={depth}',abs(res)<mp.mpf('1e-43'),fmt(res))
                save(dict(kind='gaussian_tree',w=fmt(w),phi=fmt(phi),zero_A=zero,nodes=n,depth=depth,epsilon=fmt(e),D=fmt(D),Ds=fmt(Ds),residual=fmt(res),newton_passes=it,seconds=time.monotonic()-start))
            node_error=abs(results[1][1]-results[0][1]);depth_error=abs(results[2][1]-results[0][1])
            D,Ds=results[-1]
            check('Gaussian node refinement '+fmt(phi,8),node_error<=mp.mpf('1e-12'),fmt(node_error))
            check('Gaussian history refinement '+fmt(phi,8),depth_error<=mp.mpf('1e-12'),fmt(depth_error))
            check('Gaussian deterministic derivative interval '+fmt(phi,8),abs(Ds+cc['A']+2*cc['B']*e)<=cc['K']*e*e/2+mp.mpf('1e-12'),fmt(Ds))
            if zero:
                check('Gaussian A=0 positive derivative',Ds>mp.mpf('1e-12') and abs(Ds+2*cc['B']*e)<=abs(cc['B'])*e+mp.mpf('1e-12'),fmt(Ds))
            else:
                check('Gaussian deterministic A-sign interval '+fmt(phi,8),abs(Ds+cc['A'])<=abs(cc['A'])/2,fmt(Ds))


def stochastic(mode):
    with tempfile.TemporaryDirectory(prefix='gap3b-mpfr-') as tmp:
        lib=compile_kernel(tmp)
        # New MPFR implementation reproduces deterministic constant-host D=0.
        const=np.ones((1,1024),dtype=np.float64)
        val=long_kernel(lib,const,mp.mpf('.2'),128)[0]
        check('MPFR constant host threshold and derivative',max(abs(val[0]),abs(val[1]),abs(val[2]))<mp.mpf('1e-40'),str([fmt(v) for v in val]))
        if FAILS:raise SystemExit('Controls failed before MPFR campaign')
        for idx,(model,par,phi) in enumerate(CELLS):
            cc=constants(model,par,phi)
            label=f'{model} par={par} phi={phi}'
            if mode in ('all','long'):
                h,_=environments(model,par,phi,24,8192+128,172900+idx)
                for tag,e in [('wide',cc['eps0']/2),('sign',cc['epsA']/2)]:
                    start=time.monotonic()
                    vals=long_kernel(lib,h,e)
                    D,seD=stats([v[0] for v in vals]);Ds,seDs=stats([v[1] for v in vals])
                    residual=max(abs(v[2]) for v in vals)
                    errD=abs(D-cc['A']*e-cc['B']*e*e)
                    errDs=abs(Ds+cc['A']+2*cc['B']*e)
                    check('MPFR root '+label+' '+tag,residual<=mp.mpf('1e-30'),fmt(residual))
                    check('MC D interval '+label+' '+tag,errD<=cc['K']*e**3/6+6*seD+mp.mpf('1e-40'),f'error={fmt(errD,8)} bound={fmt(cc["K"]*e**3/6,8)} SE={fmt(seD,8)}')
                    check('MC derivative interval '+label+' '+tag,errDs<=cc['K']*e**2/2+6*seDs+mp.mpf('1e-40'),f'Ds={fmt(Ds,12)} SE={fmt(seDs,8)}')
                    if tag=='sign':check('MC derivative sign '+label,abs(Ds+cc['A'])+6*seDs<=abs(cc['A'])/2,f'A={fmt(cc["A"],10)} Ds={fmt(Ds,10)}')
                    if idx==0 and tag=='wide':
                        exact,exactDs,_=gamma_threshold(e,mp.mpf(par))
                        check('MC known iid gamma threshold',abs(D-exact)<=6*seD+mp.mpf('1e-12'),fmt(abs(D-exact)))
                        check('MC known iid gamma derivative',abs(Ds-exactDs)<=6*seDs+mp.mpf('1e-12'),fmt(abs(Ds-exactDs)))
                        again=long_kernel(lib,h[:1].copy(),e,bits=256)[0]
                        err=max(abs(a-b) for a,b in zip(vals[0],again))
                        check('MPFR 192/256-bit longstay comparison',err<mp.mpf('1e-40'),fmt(err))
                    save(dict(kind='mc_derivative',model=model,par=par,phi=phi,tag=tag,epsilon=fmt(e),D=fmt(D),D_se=fmt(seD),Ds=fmt(Ds),Ds_se=fmt(seDs),A=fmt(cc['A']),B=fmt(cc['B']),K=fmt(cc['K']),root_residual=fmt(residual),seconds=time.monotonic()-start))
            if mode in ('all','rate') and idx<6:
                start=time.monotonic()
                _,eta=environments(model,par,phi,24,131072+32768,372900+idx)
                vals=rate_kernel(lib,eta,phi,32768)
                for k,L in enumerate((8,16,32)):
                    q,seq=stats([v[k][0] for v in vals]);moment,sem=stats([v[k][1] for v in vals])
                    disagree=max(v[k][2] for v in vals)
                    scaled=2*L*q/cc['Vpi'];scaled_se=2*L*seq/cc['Vpi']
                    check('MC stationary identity '+label+f' L={L}',abs(moment-cc['Vpi'])<=6*sem+mp.mpf('1e-12'),f'residual={fmt(abs(moment-cc["Vpi"]),8)} SE={fmt(sem,8)}')
                    check('MC initial-condition control '+label+f' L={L}',disagree<=mp.mpf('1e-8'),fmt(disagree))
                    if L==32:check('MC inverse-log rate '+label,abs(scaled-1)<=mp.mpf('.25')+6*scaled_se,f'scaled={fmt(scaled,12)} SE={fmt(scaled_se,8)}')
                    if idx==0:
                        exact=gamma_q(L,mp.mpf(par))
                        check('MC exact iid gamma q '+f'L={L}',abs(q-exact)<=6*seq+mp.mpf('1e-12'),f'error={fmt(abs(q-exact),8)} SE={fmt(seq,8)}')
                    s=mp.exp(-L-q/2);D=cc['ell']-q-mp.log1p(-s)
                    print(f'RATE {label} L={L} scaled={fmt(scaled,12)} SE={fmt(scaled_se,8)}',flush=True)
                    save(dict(kind='mc_rate',model=model,par=par,phi=phi,L=L,q=fmt(q),q_se=fmt(seq),scaled=fmt(scaled),scaled_se=fmt(scaled_se),s=fmt(s),D=fmt(D),ell=fmt(cc['ell']),moment=fmt(moment),moment_se=fmt(sem),initial_disagreement=fmt(disagree),seconds=time.monotonic()-start))
                if idx==0:
                    # Recompute an identical full path; precision, not resampling.
                    again=rate_kernel(lib,eta[:1].copy(),phi,32768,Ls=(8,),bits=256)[0][0]
                    err=max(abs(a-b) for a,b in zip(vals[0][0],again))
                    check('MPFR 192/256-bit smallstay comparison',err<mp.mpf('1e-40'),fmt(err))


def main():
    global RESULT_FILE
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('all','deterministic','long','rate','gaussian','gaussian-refine'),default='all')
    args=parser.parse_args()
    started=time.monotonic()
    print('GAP3b registered checks: mpmath 60 dps; MPFR 192 bits; 53-bit RNG inputs.',flush=True)
    print('Sampling SE is not arithmetic error or a rigorous confidence certificate.',flush=True)
    if args.mode in ('gaussian','gaussian-refine'):
        refine=args.mode=='gaussian-refine'
        RESULT_FILE=HERE/('gaussian_history_followup_results.json' if refine else 'gaussian_quadrature_results.json')
        gaussian_deterministic(refine)
    else:
        if args.mode in ('all','deterministic'):deterministic()
        else:controls()
        if args.mode!='deterministic':stochastic(args.mode)
    save(dict(kind='summary',mode=args.mode,passed=PASSES,failures=FAILS,seconds=time.monotonic()-started))
    print(f'SUMMARY passed={PASSES} failed={len(FAILS)} seconds={time.monotonic()-started:.3f}',flush=True)
    if FAILS:raise SystemExit(1)


if __name__=='__main__':main()
