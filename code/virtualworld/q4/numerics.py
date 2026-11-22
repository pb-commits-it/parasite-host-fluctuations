"""Q4 numerical helpers: periodic engine products and covariance response.

These are computations, not a proof or independently reviewed new theory.
Known-result gates in validate_numerics.py precede all real-data outcomes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
sys.path.insert(0,str(P.parent))
import json, csv, subprocess, hashlib, time
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.stats import skew,kurtosis,norm
from world.model import ring
from world.invasion import ring_coefficient

def date():
    return subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip()

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def write_csv(path,rows):
    if not rows: Path(path).write_text(''); return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def norm_host(x):
    x=np.asarray(x,float)
    if np.any(~np.isfinite(x)) or np.any(x<0) or x.mean()<=0: raise ValueError('invalid host')
    return x/x.mean()

def means(k,s): return (2,s) if k==2 else (2,2,s)

@lru_cache(None)
def model(k,s): return ring(means(k,s))

def growth(m,h,y):
    matrices=m.matrix(np.asarray(h)[:,None],y)
    prod=np.eye(m.n); scale=0.
    for A in matrices:
        prod=A@prod
        mx=np.max(prod)
        if mx<=0 or not np.isfinite(mx): raise FloatingPointError('invalid period product')
        prod/=mx; scale+=np.log(mx)
    rho=float(max(np.linalg.eigvals(prod).real))
    if rho<=0: raise FloatingPointError('invalid Perron root')
    return float((np.log(rho)+scale)/len(h))

def loss(k,s,h):
    m=model(k,s); base=m.critical_tilt()
    lo,hi=-1.,1.
    while growth(m,h,lo)>0 and lo>-32: lo*=2
    while growth(m,h,hi)<0 and hi<32: hi*=2
    root=brentq(lambda y:growth(m,h,y),lo,hi,xtol=1e-12)
    residual=abs(growth(m,h,root)/((growth(m,h,root+1e-5)-growth(m,h,root-1e-5))/2e-5))
    if residual>1e-7: raise RuntimeError('root residual')
    return float(root-base),float(residual)

@lru_cache(None)
def kernel(k,s):
    m=model(k,s);y=m.critical_tilt();A=m.matrix(y=y)
    vals,vec=np.linalg.eig(A);v=vec[:,np.argmin(abs(vals-1))].real;v/=v.sum()
    vals,vec=np.linalg.eig(A.T);w=vec[:,np.argmin(abs(vals-1))].real;w/=w@v
    B=m.noise_parts[0,0]+np.exp(y)*m.noise_parts[1,0]
    Q=np.eye(m.n)-np.outer(v,w); M=A@Q
    b=float(w@(np.exp(y)*m.parts[1])@v);c=float(w@B@v)
    weights=np.empty(4096);u=Q@B@v
    for i in range(len(weights)):
        weights[i]=float(w@B@u)/b;u=M@u
    if np.linalg.norm(u)>1e-12: raise RuntimeError('kernel tail too large')
    return c*c/(2*b),weights

def cov(z):
    z=np.asarray(z);f=np.fft.fft(z-z.mean())
    return np.fft.ifft(abs(f)**2).real/len(z)

def predict(k,s,h):
    h=np.asarray(h)
    if np.any(h<=0): return None,None
    z=np.log(h);C=cov(z);var=C[0]
    if var<1e-25: return 0.,0.
    phi=C[1]/var
    ar=float(var*ring_coefficient(means(k,s),phi)) if abs(phi)<1 else None
    leading,weights=kernel(k,s)
    full=float(leading*var-np.dot(weights,C[np.arange(1,4097)%len(z)]))
    return ar,full

def tol(pred): return max(.01,.2*abs(pred))
def status(D,pred): return 'inapplicable' if pred is None else ('agree' if abs(D-pred)<=tol(pred) else 'departure')

def descriptors(h):
    h=np.asarray(h);n=len(h);ss=(h-1)**2
    rawshare=float(np.sort(ss)[-int(np.ceil(n*.1)):].sum()/ss.sum()) if ss.sum()>0 else 0.
    out=dict(n=n,zeros=int(np.sum(h==0)),raw_skew=float(skew(h)) if h.std()>0 else 0.,
             raw_excess_kurtosis=float(kurtosis(h)) if h.std()>0 else 0.,top_decile_variance_share=rawshare)
    if np.any(h<=0):
        out.update(sigma=None,phi=None,feature_flags='zero_support'+(';raw_tail' if rawshare>.75 else ''))
        return out
    z=np.log(h);z-=z.mean();C=cov(z);sd=np.sqrt(C[0]);phi=C[1]/C[0] if sd>1e-12 else 0.
    beta=np.dot(np.arange(n)-(n-1)/2,z)/np.sum((np.arange(n)-(n-1)/2)**2)
    trend=beta*(np.arange(n)-(n-1)/2);r2=float(np.mean(trend**2)/C[0]) if sd>1e-12 else 0.
    power=abs(np.fft.rfft(z))**2;power[0]=0
    power[1:-1 if n%2==0 else None]*=2
    peak=int(np.argmax(power));share=float(power[peak]/power.sum()) if power.sum()>1e-25 else 0.
    logsk=float(skew(z)) if sd>1e-12 else 0.; logku=float(kurtosis(z)) if sd>1e-12 else 0.
    lags=np.arange(1,min(10,n//2)+1)
    acfdiff=float(np.sqrt(np.mean((C[lags]/C[0]-phi**lags)**2))) if sd>1e-12 else 0.
    jump=float(abs(z[-1]-z[0])/sd) if sd>1e-12 else 0.
    flags=[name for name,test in [('skew',abs(logsk)>1),('heavy_log_tail',logku>3),('raw_tail',rawshare>.75),
        ('cycle',share>.4),('trend',r2>.25),('wrap',jump>3),('non_AR_ACF',acfdiff>.2)] if test] if sd>1e-12 else []
    out.update(sigma=float(sd),phi=float(phi),log_skew=logsk,log_excess_kurtosis=logku,
        dominant_power=share,dominant_period=float(n/peak) if peak else None,trend_R2=r2,
        wrap_jump_SD=jump,acf_discrepancy=acfdiff,feature_flags=';'.join(flags))
    return out

def variants(h,index):
    rng=np.random.default_rng(415500+index);h=np.asarray(h);n=len(h)
    if np.all(h>0):
        z=np.log(h);z-=z.mean()
        yield 'small',0,norm_host(np.exp(.1*z))
        f=np.fft.rfft(z)
        for j in range(8):
            phase=rng.uniform(0,2*np.pi,len(f));phase[0]=0
            if n%2==0: phase[-1]=rng.choice([0.,np.pi])
            ff=abs(f)*np.exp(1j*phase); ff[0]=0
            zz=np.fft.irfft(ff,n=n)
            if np.max(abs(cov(zz)-cov(z)))>1e-10: raise RuntimeError('phase spectrum changed')
            yield 'phase',j,norm_host(np.exp(zz-zz.max()))
    for j in range(8): yield 'shuffle',j,rng.permutation(h)
    if np.all(h>0):
        x=np.arange(n);trend=np.polyval(np.polyfit(x,z,1),x)
        yield 'detrend',0,norm_host(np.exp(z-trend))
        yield 'clip',0,norm_host(np.exp(np.clip(z,*np.quantile(z,[.05,.95]))))
    yield 'mirror',0,np.concatenate([h,h[::-1]])

def wilson(success,n=64):
    z=float(norm.ppf(.975));p=success/n;den=1+z*z/n
    ctr=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(ctr-half),float(ctr+half)
