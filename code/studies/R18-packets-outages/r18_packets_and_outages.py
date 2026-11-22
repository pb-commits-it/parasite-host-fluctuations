"""R18. Run only with python3 -I. See protocol.md for definitions and tolerances."""
from __future__ import annotations
import os
# Avoid hidden threaded BLAS in three-process runs.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, time, json, math, hashlib, pathlib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom
from scipy.optimize import brentq

ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
SWITCH=1e9
T=1700

def emit(tag, **kw):
    print(tag+' '+json.dumps(kw,allow_nan=False,separators=(',',':')),flush=True)

def bdraw(rng,n,p,limit=SWITCH):
    n=np.asarray(n); p=np.broadcast_to(p,n.shape)
    big=n>=limit
    if not np.any(big): return rng.binomial(n.astype(np.int64),p).astype(float)
    out=np.empty_like(n,dtype=float)
    out[~big]=rng.binomial(n[~big].astype(np.int64),p[~big])
    m=n[big]*p[big]
    out[big]=np.clip(np.rint(m+np.sqrt(m*(1-p[big]))*rng.standard_normal(m.shape)),0,n[big])
    return out

def pdraw(rng,m,limit=SWITCH):
    m=np.asarray(m); big=m>=limit
    if not np.any(big): return rng.poisson(m).astype(float)
    out=np.empty_like(m,dtype=float)
    out[~big]=rng.poisson(m[~big]); out[big]=np.maximum(0,np.rint(m[big]+np.sqrt(m[big])*rng.standard_normal(np.sum(big))))
    return out

def mdraw(rng,n,p,limit=SWITCH):
    n=np.asarray(n); big=n>=limit
    if not np.any(big): return rng.multinomial(n.astype(np.int64),p).astype(float)
    out=np.empty((len(n),len(p)))
    out[~big]=rng.multinomial(n[~big].astype(np.int64),p)
    mu=n[big,None]*p
    z=np.sqrt(mu)*rng.standard_normal(mu.shape)
    v=np.rint(mu+z-z.sum(axis=1)[:,None]*p)
    v=np.maximum(v,0)
    # Correct the largest category; the relative repair is logged by failure if material.
    j=np.argmax(p); correction=n[big]-v.sum(axis=1)
    if np.any(np.abs(correction)>np.maximum(1e-6*n[big],len(p))): raise RuntimeError('material multinomial repair')
    v[:,j]+=correction
    if np.any(v<0): raise RuntimeError('negative category after repair')
    out[big]=v
    return out

class HostLaw:
    def __init__(self,L,a,k,initial=100,clump=False,limit=SWITCH):
        self.L=L; self.a=a; self.k=k; self.clump=clump; self.limit=limit
        self.B=max(initial,19 if clump else k)
        self.weights=np.arange(self.B+1,dtype=float)
        self.sh=(1-1/L)**a; self.si=(1-1/L)**(1-a)
        self.di=1-self.si
        self.extra=max(0,(1/L-self.di)/self.si)
        self.probs=[None]+[binom.pmf(np.arange(b+1),b,self.si) for b in range(1,self.B+1)]
        for b in range(1,self.B+1): self.probs[b]/=self.probs[b].sum()
        self.cuts=[0]+[max(0,int(binom.ppf(1e-10,b,self.si))) for b in range(1,self.B+1)]
        self.pooled=[]
        for b in range(self.B+1):
            cut=self.cuts[b]
            if b and cut>=3:
                p=self.probs[b];self.pooled.append(np.r_[p[:cut].sum(),p[cut:]])
            else:self.pooled.append(None)
    def step(self,rng,F,C,h,gf,gs):
        Sold=C@self.weights
        Fborn=None
        # For modest numbers of burden>1 hosts, sample their burdens directly.
        # This is the same multinomial experiment expressed as individual trials.
        direct=(self.a<1 and self.B>=30 and C[:,2:].sum()<=3000000 and not getattr(self,'dense',False) and not getattr(self,'pooled_only',False) and getattr(self,'direct_allowed',self.k in [1,100]))
        if direct:
            Cnew=np.zeros_like(C)
            Cnew[:,1]=bdraw(rng,C[:,1],self.si*self.sh,self.limit)
            counts=C[:,2:].ravel()
            nz=np.flatnonzero(counts)
            loc=np.repeat(nz,counts[nz].astype(np.int64))
            if len(loc):
                rep=loc//(self.B-1)
                burden=loc%(self.B-1)+2
                after=rng.binomial(burden,self.si)
                if self.a>0:after*=rng.random(len(after))<self.sh
                keep=after>0
                Cnew+=np.bincount(rep[keep]*(self.B+1)+after[keep],minlength=C.size).reshape(C.shape)
        else:
            if self.a==1:
                I=C
                Isum=Sold
            else:
                I=np.zeros_like(C)
                # Only nonempty burden classes enter the multinomial draw.
                for b in np.flatnonzero(np.any(C[:,1:]>0,axis=0))+1:
                    nz=np.flatnonzero(C[:,b]>0)
                    if b==1:
                        I[nz,1]+=bdraw(rng,C[nz,1],self.si,self.limit)
                    else:
                        cut=self.cuts[b]
                        if cut>=3 and not getattr(self,'dense',False):
                            draws=mdraw(rng,C[nz,b],self.pooled[b],self.limit)
                            I[nz,cut:b+1]+=draws[:,1:]
                            rare=np.flatnonzero(draws[:,0]>0)
                            if len(rare):
                                pt=self.probs[b][:cut];pt=pt/pt.sum()
                                I[nz[rare],:cut]+=mdraw(rng,draws[rare,0],pt,self.limit)
                        else:
                            draws=mdraw(rng,C[nz,b],self.probs[b],self.limit)
                            I[nz,:b+1]+=draws
                I[:,0]=0
                Isum=I@self.weights
            if getattr(self,'early_shedding_draw',False):Fborn=pdraw(rng,Sold*gs,self.limit)
            if self.a==0: Cnew=I
            else: Cnew=bdraw(rng,I,self.sh,self.limit)
        if Fborn is None:Fborn=pdraw(rng,Sold*gs,self.limit)
        leaving=bdraw(rng,F,.1,self.limit)
        Fnew=F-leaving+Fborn
        pm=leaving*(gf/.1)*h/self.k
        if self.clump:
            Cnew[:,1]+=pdraw(rng,pm/2,self.limit)
            Cnew[:,19]+=pdraw(rng,pm/2,self.limit)
        else: Cnew[:,self.k]+=pdraw(rng,pm,self.limit)
        return Fnew,Cnew

def environment(rng,state,env):
    if env=='A':
        state=.9*state+math.sqrt(.19)*rng.standard_normal(state.shape)
        return state,np.exp(.15*state-.01125)
    if env=='constant': return state,np.ones(state.shape)
    G=int(env[1:]); state=np.where(rng.random(state.shape)<1/G,1-state,state)
    return state,2*state

def summary(ext,mins,teq):
    n=len(ext); d=ext>0; k=int(d.sum()); p=k/n
    pg=np.mean((ext>0)&(ext<=teq)); se=math.sqrt(p*(1-p)/n)
    z=1.96; den=1+z*z/n; mid=(p+z*z/(2*n))/den
    rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(n=n,extinct=k,p=p,se=se,wilson=[max(0,mid-rad),min(1,mid+rad)],p_gen=float(pg),se_gen=float(math.sqrt(pg*(1-pg)/n)),
       extinct_time_quartiles=np.quantile(ext[d],[.25,.5,.75]).tolist() if k else None,
       survivor_host_min_quartiles=np.quantile(mins[~d],[.25,.5,.75]).tolist() if np.any(~d) else None,
       survivors_with_zero_hosts=int(np.sum((mins==0)&~d)))

def simulate(spec,reps,seed,limit=SWITCH,steps=T,startN=10000,founderF=False):
    rng=np.random.default_rng(seed); L,a,k,env,b0=spec['L'],spec['a'],spec['k'],spec['env'],spec['b0']
    law=HostLaw(L,a,k,b0,spec.get('clump',False),limit)
    law.dense=spec.get('dense',False)
    # Preserve the exact RNG consumption of cells completed before the optimization.
    cached_pooled=(env=='A' and b0==1 and not spec.get('clump',False) and ((a==0 and k==100) or (a==.5 and k==1)))
    law.pooled_only=cached_pooled
    law.early_shedding_draw=cached_pooled and a==.5
    law.direct_allowed=(k in [1,100]) or (env=='A' and a==0 and k==10 and b0==100 and not spec.get('clump',False))
    C=np.zeros((reps,law.B+1)); F=np.zeros(reps)
    if founderF: F[:]=1
    else: C[:,b0]=startN/b0
    state=rng.standard_normal(reps) if env=='A' else rng.integers(0,2,reps).astype(float)
    ids=np.arange(reps); ext=np.full(reps,-1,dtype=np.int32); mins=C.sum(1); approximate=np.zeros(reps,bool)
    for t in range(1,steps+1):
        if not len(ids): break
        state,h=environment(rng,state,env)
        large=(F>=limit)|(C.max(1)>=limit)
        approximate[ids[large]]=True
        F,C=law.step(rng,F,C,h,spec['gf'],spec['gs'])
        hosts=C.sum(1)
        mins[ids]=np.minimum(mins[ids],hosts)
        dead=(F==0)&(hosts==0)
        if np.any(dead):
            ext[ids[dead]]=t; keep=~dead
            ids,F,C,state=ids[keep],F[keep],C[keep],state[keep]
        if np.any(~np.isfinite(F)) or np.any(~np.isfinite(C)): raise RuntimeError('nonfinite population')
    return ext,mins,approximate

def cell(spec,seed):
    t0=time.monotonic()
    e,m,a=simulate(spec,1000,seed)
    n0=int(np.sum(e>0)); extra=n0<20 or n0>980
    if extra:
        e2,m2,a2=simulate(spec,2000,seed+1000000)
        e=np.r_[e,e2];m=np.r_[m,m2];a=np.r_[a,a2]
    key=f"{spec['env']}_L{spec['L']}_a{spec['a']}_k{spec['k']}_b{spec['b0']}"+('_clump' if spec.get('clump') else '')
    np.savez_compressed(ROOT/('raw_'+key+'.npz'),extinction_time=e,minimum_occupied_hosts=m,approximated=a)
    return dict(key=key,**spec,**summary(e,m,10*(10+spec['L'])),approx_runs=int(a.sum()),seconds=time.monotonic()-t0,seed=seed)

def growth_grid(env,rs,seed,npaths=64,steps=80000,burn=2000):
    # rs shape (n_calibration_candidates,3); all candidates share environmental histories.
    rng=np.random.default_rng(seed); L=np.array([10.,40.,160.]); q=1/L
    shape=(len(rs),3,2*npaths); x=np.full(shape,.5); y=np.full(shape,.5)
    logsum=np.zeros(shape); state=rng.standard_normal(npaths) if env=='A' else rng.integers(0,2,npaths).astype(float)
    gf=(.1+rs)[...,None];gs=(q+rs)[...,None]
    for t in range(steps+burn):
        state,hbase=environment(rng,state,env)
        hant=np.exp(-.15*state-.01125) if env=='A' else 2-hbase
        h=np.r_[hbase,hant]
        u=.9*x+gs*y;v=gf*h*x+(1-q)[None,:,None]*y
        total=u+v; x=u/total;y=v/total
        if t>=burn: logsum+=np.log(total)
    # The two antithetic histories form ONE independent observation for SE.
    return (logsum/steps).reshape(len(rs),3,2,npaths).mean(axis=2)

def calibrate(env,seed):
    # A monotone solve using common histories, carried out in one pass per bisection round.
    lo=np.zeros(3);hi=np.full(3,.15);tg=np.array([20,50,170]);target=.005/tg
    for it in range(19):
        mid=(lo+hi)/2
        g=growth_grid(env,mid[None,:],seed)
        small=g[0].mean(1)<target
        lo=np.where(small,mid,lo);hi=np.where(small,hi,mid)
    r=(lo+hi)/2
    # Independent longer-path check and uncertainty across independent paths.
    check=growth_grid(env,r[None,:],seed+1)[0]
    mean=check.mean(1);se=check.std(1,ddof=1)/math.sqrt(check.shape[1]);m=mean*tg
    rows=[]
    for j,L in enumerate([10,40,160]):
        D=tg[j]*(math.log1p(r[j])-mean[j]); err=se[j]*tg[j]
        rows.append(dict(env=env,L=L,r=float(r[j]),gf=float(.1+r[j]),gs=float(1/L+r[j]),R0=float((1+10*r[j])*(1+L*r[j])),
            D=float(D),D_se=float(err),margin=float(m[j]),margin_se=float(err),pass_match=bool(abs(m[j]-.005)<=max(.002,4*err))))
    return rows

def pgf(a,k,L,gf,gs,steps=3000):
    q=1/L;sh=(1-q)**a
    f=s=0.
    for t in range(steps):
        # Poisson shedding is independent of individual and host death.
        nf=.9*f+.1*math.exp((gf/.1/k)*(s-1))
        if a==0: ns=(q+(1-q)*s)*math.exp(gs*(f-1))
        else: ns=((1-sh)+sh*s)*math.exp(k*gs*(f-1))
        f,s=nf,ns
    return f,s

def validation_pgf(seed=18001):
    out=[]
    for j,(a,k) in enumerate([(0,1),(1,10),(1,100)]):
        L=40;gf=.1*math.sqrt(1.35);gs=1/L*math.sqrt(1.35)
        spec=dict(L=L,a=a,k=k,env='constant',b0=k,gf=gf,gs=gs)
        finite=pgf(a,k,L,gf,gs,3000)[1];fixed=pgf(a,k,L,gf,gs,100000)[1]
        e,m,ap=simulate(spec,10000,seed+j,steps=3000,startN=k)
        p=np.mean(e>0);se=max(math.sqrt(finite*(1-finite)/10000),.0001)
        # Force Gaussian approximation at a lower switch and compare a separate sample.
        ea,ma,aa=simulate(spec,10000,seed+100+j,limit=1000,steps=3000,startN=k)
        pa=np.mean(ea>0); sea=math.sqrt(2*finite*(1-finite)/10000)
        out.append(dict(a=a,k=k,p=float(p),p_approx=float(pa),finite=finite,fixed=fixed,z=float((p-finite)/se),z_approx=float((pa-finite)/se),gap=abs(finite-fixed),approx_runs=int(aa.sum()),pass_check=bool(abs(p-finite)<=4*se and abs(pa-finite)<=4*se and abs(finite-fixed)<.001)))
    return out

def validation_means(seed=18002):
    rng=np.random.default_rng(seed);records=[]
    n=2000;N=100000;F0=100000
    for L in [10,40,160]:
      gf=.105;gs=1/L+.005
      for a in [0,.5,1]:
       for k in [1,10,100]:
        for b in [1,100]:
         law=HostLaw(L,a,k,b)
         for h in [0,1,2]:
          C=np.zeros((n,law.B+1));C[:,b]=N/b;F=np.full(n,float(F0))
          Fn,Cn=law.step(rng,F,C,np.full(n,h),gf,gs);Sn=Cn@law.weights
          means=np.array([.9*F0+gs*N,(1-1/L)*N+gf*h*F0])
          # Shedding count Binom(N,q), followed by Poisson burst.
          q=1/L;burst=gs/q
          varF=.09*F0+N*gs
          si=law.si;sh=law.sh
          varS=(N/b)*(sh*b*si*(1-si)+sh*(1-sh)*(b*si)**2)
          # F leavers: compound Poisson packet process mixed by Binomial count.
          c=gf/.1*h
          varS+=F0*(.1*k*c+.1*.9*c*c)
          ses=np.sqrt(np.array([varF,varS])/n)
          got=np.array([Fn.mean(),Sn.mean()]);z=np.divide(got-means,ses,out=np.zeros(2),where=ses>0)
          records.append(dict(L=L,a=a,k=k,b=b,h=h,z=z.tolist(),relative_error=((got-means)/(N+F0)).tolist(),pass_check=bool(np.all(np.abs(z)<=4) and np.all(np.abs(got-means)/(N+F0)<.005))))
    return records

def validate_approx(seed=18003):
    rng=np.random.default_rng(seed);n=10000.;p=np.array([.05,.15,.8]);draw=mdraw(rng,np.full(100000,n),p,1000)
    empirical=np.cov(draw,rowvar=False);exact=n*(np.diag(p)-np.outer(p,p))
    mean_z=(draw.mean(0)-n*p)/np.sqrt(np.diag(exact)/len(draw))
    rel=float(np.max(np.abs(empirical-exact)/np.maximum(abs(exact),1)))
    return dict(mean_z=mean_z.tolist(),max_cov_relative_error=rel,pass_check=bool(max(abs(mean_z))<4 and rel<.04))

def legacy(L,seed,reps=1000):
    rng=np.random.default_rng(seed);q=np.array([.1] if L==0 else [.1,1/L]);s=1-q
    b=np.exp(.0045/len(q));g=q*b
    A=np.diag(s)
    for i in range(len(q)): A[(i+1)%len(q),i]+=g[i]
    vals,v=np.linalg.eig(A);v=np.abs(v[:,np.argmax(vals.real)].real);v/=v.sum()
    X=rng.multinomial(10000,v,size=reps).astype(float);z=rng.standard_normal(reps)
    ids=np.arange(reps);ext=np.full(reps,-1,int);ap=np.zeros(reps,bool)
    steps=6000*(10+L)
    for t in range(1,steps+1):
        if not len(ids): break
        z=.9*z+math.sqrt(.19)*rng.standard_normal(len(z));h=np.exp(.15*z-.01125)
        stay=bdraw(rng,X,s);leave=X-stay;m=leave*b;m[:,0]*=h
        ap[ids[np.any((X>=SWITCH)|(m>=SWITCH),axis=1)]]=True
        births=pdraw(rng,m);X=stay+np.roll(births,1,axis=1)
        dead=X.sum(1)==0
        if np.any(dead):
            ext[ids[dead]]=t;keep=~dead;ids,X,z=ids[keep],X[keep],z[keep]
    p=float(np.mean(ext>0));ref={0:.368,10:.6285,40:.129}[L]
    se=math.sqrt(p*(1-p)/reps+ref*(1-ref)/2000)
    return dict(L=L,n=reps,steps=steps,p=p,reference=ref,z=(p-ref)/se,approx_runs=int(ap.sum()),pass_check=bool(abs(p-ref)<=4*se))

def retain(h,p,N,b):
    unit=h*(-math.expm1(b*math.log1p(-p))) if p<1 else h
    if unit>=1:return 1.
    return -math.expm1((N/b)*math.log1p(-unit))

def gap_experiments(seed=18004):
    rng=np.random.default_rng(seed);rows=[]
    for a in [0,.5,1]:
      for b in [1,100]:
       for x in [.5,2,5,10]:
        h=math.exp(-a*x);p=math.exp(-(1-a)*x);M=10000//b
        exact=retain(h,p,10000,b)
        # Marginal within-host survivors, then independent host survival. This uses the derived event decomposition.
        keepP=-math.expm1(b*math.log1p(-p)) if p<1 else 1.
        hs=rng.binomial(M,h,20000);alive=rng.binomial(hs,keepP)>0
        obs=float(alive.mean());se=max(math.sqrt(exact*(1-exact)/20000),1/20000)
        rows.append(dict(shape=1,age='fresh',a=a,b=b,ratio=x,exact=exact,observed=obs,z=(obs-exact)/se,pass_check=bool(abs(obs-exact)<=4*se)))
    for age in ['fresh','stationary']:
      for b in [1,100]:
       M=10000//b
       for x in [.5,2,5,10]:
        h=float(gammaincc(4,4*x)) if age=='fresh' else float(sum(gammaincc(j+1,4*x) for j in range(4))/4)
        exact=retain(h,1,10000,b)
        # Direct gamma remaining-lifetime sampling in chunks (stationary phase uniform).
        alive=[]
        for start in range(0,20000,100):
            shape=4 if age=='fresh' else rng.integers(1,5,(100,M))
            rem=rng.gamma(shape,1/4,size=(100,M))
            alive.extend(np.any(rem>x,axis=1).tolist())
        obs=float(np.mean(alive));se=max(math.sqrt(exact*(1-exact)/20000),1/20000)
        rows.append(dict(shape=4,age=age,a=1,b=b,ratio=x,exact=exact,observed=obs,z=(obs-exact)/se,pass_check=bool(abs(obs-exact)<=4*se)))
    thresholds=[]
    for shape in [1,4]:
      for age in ['fresh','stationary']:
       for M in [100,10000]:
        def survival(x):
            h=math.exp(-x) if shape==1 else (float(gammaincc(4,4*x)) if age=='fresh' else float(sum(gammaincc(j+1,4*x) for j in range(4))/4))
            return retain(h,1,M,1)
        thresholds.append(dict(shape=shape,age=age,units=M,ratios={str(target):brentq(lambda x:survival(x)-target,1e-8,100) for target in [.95,.5,.05]}))
    fixed=[]
    for L in [10,40,160]:
      for G in [5,20,60,200,800,1600,2400]:
       for b in [1,100]:
        for shape in [1,4]:
         x=G/L;h=math.exp(-x) if shape==1 else float(gammaincc(4,4*x))
         fixed.append(dict(L=L,G=G,units=10000//b,shape=shape,p=retain(h,1,10000,b)))
    return rows,thresholds,fixed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['attempt','final','stress','extension'],default='final');parser.add_argument('--resume',action='store_true');parser.add_argument('--fast-resume',action='store_true');args=parser.parse_args()
    start=time.monotonic();emit('START',shedding='independent_poisson',mode=args.mode,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest(),numpy=np.__version__,processes=3)
    if args.mode=='attempt':
        emit('APPROX',**validate_approx())
        # A balanced engineering sample, no claims from its estimates.
        for j,(a,k,b) in enumerate([(0,1,1),(.5,100,100),(1,100,100)]):
            spec=dict(L=40,a=a,k=k,b0=b,env='B20',gf=.11,gs=.035)
            t0=time.monotonic();e,m,ap=simulate(spec,100,19000+j,steps=200)
            emit('BENCHMARK',a=a,k=k,b=b,seconds=time.monotonic()-t0,extinct=int(sum(e>0)))
        emit('ATTEMPT_END',seconds=time.monotonic()-start);return
    if args.mode=='extension':
        extension_main(start)
        return
    if args.mode=='stress':
        stress_main(start)
        return
    if args.mode=='final' and not args.resume:
        (ROOT/'r18_campaign.json').write_text(json.dumps({'started_epoch':time.time()}))
    results={}
    if args.resume:
        resume(start,skip_pgf=args.fast_resume)
        return
    with ProcessPoolExecutor(max_workers=3) as pool:
        # Calibration is expensive but independent and must precede demographic comparisons.
        futures={pool.submit(calibrate,e,18100+i*10):('CALIBRATION',e) for i,e in enumerate(['A','B20','B200','B5','B60'])}
        for f in as_completed(futures):
            tag,e=futures[f];rows=f.result();results[e]=rows
            for row in rows:emit(tag,**row)
        (ROOT/'r18_calibration.json').write_text(json.dumps(results,indent=2))
        fs={pool.submit(legacy,L,18200+i):('LEGACY',L) for i,L in enumerate([0,10,40])}
        fs[pool.submit(validation_pgf)]=('PGF',None)
        fs[pool.submit(validation_means)]=('MEANS',None)
        fs[pool.submit(gap_experiments)]=('GAPS',None)
        fs[pool.submit(validate_approx)]=('APPROX',None)
        valid=True
        for f in as_completed(fs):
            tag,_=fs[f];out=f.result()
            if tag=='MEANS':
                worst=max(max(abs(x) for x in row['z']) for row in out);passed=all(row['pass_check'] for row in out)
                emit(tag,cases=len(out),worst_z=worst,pass_check=passed,failures=[r for r in out if not r['pass_check']]);valid &= passed
                (ROOT/'r18_mean_checks.json').write_text(json.dumps(out))
            elif tag=='GAPS':
                rows,thresholds,fixed=out
                for row in rows:emit('GAP_CHECK',**row)
                for row in thresholds:emit('GAP_THRESHOLD',**row)
                for row in fixed:emit('GAP_FIXED',**row)
                valid &= all(r['pass_check'] for r in rows)
            elif tag=='PGF':
                for row in out:emit(tag,**row)
                valid &= all(r['pass_check'] for r in out)
            else:emit(tag,**out);valid &= out['pass_check']
        valid &= all(r['pass_match'] for rows in results.values() for r in rows)
        emit('VALIDATION_COMPLETE',passed=valid,elapsed=time.monotonic()-start)
        if not valid: raise RuntimeError('validation failed; factorial not started')
        done=factorial(pool,results,start)
    emit('END',elapsed=time.monotonic()-start,cells=len(done),primary=sum(r['kind']=='primary' for r in done))

def factorial(pool,results,start,resume_run=False):
    jobs=[]
    for env in ['A','B20','B200']:
      for b in [1,100]:
       for a in [0,.5,1]:
        for k in [1,10,100]:
         for L in [10,40,160]:
          cal=next(r for r in results[env] if r['L']==L)
          jobs.append(dict(L=L,a=a,k=k,b0=b,env=env,gf=cal['gf'],gs=cal['gs'],kind='primary',dense=(env=='A' and b==1 and a==0 and k in [1,10])))
    for env in ['A','B20','B200']:
      for a in [0,1]:
       for L in [10,40,160]:
        cal=next(r for r in results[env] if r['L']==L)
        jobs.append(dict(L=L,a=a,k=10,b0=100,env=env,gf=cal['gf'],gs=cal['gs'],clump=True,kind='clumped'))
    for env in ['B5','B60']:
      for k in [1,100]:
       for L in [10,40,160]:
        cal=next(r for r in results[env] if r['L']==L)
        jobs.append(dict(L=L,a=1,k=k,b0=100,env=env,gf=cal['gf'],gs=cal['gs'],kind='extra_duration'))
    # Runtime-only priority: protect a complete two-level factorial before midpoint extensions.
    def priority(spec):
        if spec['kind']!='primary':return (1.5 if spec['kind']=='extra_duration' else 1.6, spec['env'],spec['a'],spec['k'],spec['b0'],spec['L'])
        tier=0 if spec['a'] in [0,1] and spec['k'] in [1,100] else (1 if spec['k'] in [1,100] else 2)
        return (tier,spec['a'],spec['k'],spec['b0'],spec['env'],spec['L'])
    # Keep the original cell seed despite changing execution order.
    for i,spec in enumerate(jobs):spec['job_seed']=18300+i
    jobs.sort(key=priority)
    # Queue at most one complete L triple per worker group. Never discard a running trajectory.
    done=json.loads((ROOT/'r18_cells.json').read_text()) if resume_run else []
    completed={r['key'] for r in done}
    for pos in range(0,len(jobs),3):
        if time.monotonic()-start>2100:
            emit('BUDGET_STOP',unrun=jobs[pos:],completed=len(done));break
        batch=jobs[pos:pos+3]
        futures=[]
        for j,spec in enumerate(batch):
            key=f"{spec['env']}_L{spec['L']}_a{spec['a']}_k{spec['k']}_b{spec['b0']}"+('_clump' if spec.get('clump') else '')
            if key not in completed:futures.append(pool.submit(cell,spec,spec['job_seed']))
        for f in as_completed(futures):
            row=f.result();done.append(row);emit('CELL',**row)
        (ROOT/'r18_cells.json').write_text(json.dumps(done,indent=2))
    return done

def resume(start,skip_pgf=False):
    # Count the whole campaign, including the brief engineering pause, against the budget.
    import datetime
    wall0=campaign_wall0()
    start=time.monotonic()-(time.time()-wall0)
    results=json.loads((ROOT/'r18_calibration.json').read_text())
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(validation_means):'MEANS_OPTIMIZED',pool.submit(validate_approx):'APPROX_OPTIMIZED'}
        if not skip_pgf:fs[pool.submit(validation_pgf)]='PGF_OPTIMIZED'
        passed=True
        for f in as_completed(fs):
            tag=fs[f];out=f.result()
            if tag=='MEANS_OPTIMIZED':
                ok=all(r['pass_check'] for r in out);passed &=ok
                emit(tag,cases=len(out),worst_z=max(max(abs(v) for v in r['z']) for r in out),pass_check=ok)
            elif tag=='PGF_OPTIMIZED':
                for row in out:emit(tag,**row)
                passed &=all(r['pass_check'] for r in out)
            else:emit(tag,**out);passed &=out['pass_check']
        if not passed:raise RuntimeError('optimized transition validation failed')
        emit('RESUME',elapsed=time.monotonic()-start,law_unchanged=True)
        done=factorial(pool,results,start,resume_run=True)
    emit('END',elapsed=time.monotonic()-start,cells=len(done),primary=sum(r['kind']=='primary' for r in done))


def stress_pair(a,k,seed):
    cal=next(r for r in json.loads((ROOT/'r18_calibration.json').read_text())['B20'] if r['L']==40)
    spec=dict(L=40,a=a,k=k,b0=100,env='B20',gf=cal['gf'],gs=cal['gs'])
    e,m,ap=simulate(spec,1000,seed,limit=1e18,steps=1000,startN=1000)
    f,mm,aq=simulate(spec,1000,seed+100,limit=1000,steps=1000,startN=1000)
    p=float(np.mean(e>0));q=float(np.mean(f>0));pbar=(p+q)/2
    se=max(math.sqrt(2*pbar*(1-pbar)/1000),.001)
    return dict(a=a,k=k,n_each=1000,p_exact=p,p_forced=q,z=(p-q)/se,
        reference_approximated=int(ap.sum()),forced_approximated=int(aq.sum()),
        forced_returned_to_extinction=int(np.sum(aq&(f>0))),pass_check=bool(abs(p-q)<=4*se and not np.any(ap)))


def cached_reproduction():
    rows=json.loads((ROOT/'r18_cells.json').read_text())
    row=next(r for r in rows if r['key']=='A_L10_a0.5_k1_b1')
    e,m,ap=simulate(row,1000,row['seed'])
    e2,m2,ap2=simulate(row,2000,row['seed']+1000000)
    old=np.load(ROOT/('raw_'+row['key']+'.npz'))
    ok=bool(np.array_equal(np.r_[e,e2],old['extinction_time']) and np.array_equal(np.r_[m,m2],old['minimum_occupied_hosts']))
    return dict(key=row['key'],replicates=3000,bitwise_times_and_minima=ok,pass_check=ok)

def stress_main(start):
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(stress_pair,a,k,19400+j) for j,(a,k) in enumerate([(0,1),(.5,100),(1,100)])]
        ok=True
        for f in as_completed(fs):
            r=f.result();emit('OUTAGE_STRESS',**r);ok &=r['pass_check']
    rec=cached_reproduction();emit('CACHED_REPRODUCTION',**rec);ok &=rec['pass_check']
    means=validation_means();(ROOT/'r18_mean_checks.json').write_text(json.dumps(means))
    mean_ok=all(r['pass_check'] for r in means);ok &=mean_ok
    emit('FINAL_MEANS',cases=len(means),worst_z=max(max(abs(v) for v in r['z']) for r in means),pass_check=mean_ok)
    emit('OUTAGE_STRESS_END',seconds=time.monotonic()-start,passed=ok)


def chain_growth(rs,seed):
    rng=np.random.default_rng(seed);L=np.array([10.,40.,160.]);q=(4/L)[:,None]
    rs=np.asarray(rs);gf=(.1+rs)[:,None];gs=(rs/(-np.expm1(-4*np.log1p(rs/(4/L)))))[:,None]
    F=np.full((3,128),.2);S=np.full((3,128,4),.2);state=rng.integers(0,2,64).astype(float);acc=np.zeros_like(F)
    for t in range(82000):
        state,h0=environment(rng,state,'B200');h=np.r_[h0,2-h0]
        Fn=.9*F+gs*S.sum(2);Sn=(1-q[:,:,None])*S
        Sn[:,:,1:]+=q[:,:,None]*S[:,:,:3];Sn[:,:,0]+=gf*h*F
        total=Fn+Sn.sum(2);F=Fn/total;S=Sn/total[:,:,None]
        if t>=2000:acc+=np.log(total)
    return (acc/80000).reshape(3,2,64).mean(1)

def chain_calibrate():
    lo=np.zeros(3);hi=np.full(3,.15);L=np.array([10.,40.,160.]);target=.005/(L+10)
    for _ in range(19):
        mid=(lo+hi)/2;v=chain_growth(mid,19500).mean(1);low=v<target
        lo=np.where(low,mid,lo);hi=np.where(low,hi,mid)
    r=(lo+hi)/2;v=chain_growth(r,19501);m=v.mean(1)*(L+10);se=v.std(1,ddof=1)/8*(L+10)
    out=[]
    for j,ll in enumerate([10,40,160]):
        q=4/ll;gf=.1+r[j];gs=r[j]/(-math.expm1(-4*math.log1p(r[j]/q)))
        A=np.zeros((5,5));A[0,0]=.9;A[0,1:]=gs;A[1,0]=gf
        for ph in range(4):
            A[ph+1,ph+1]=1-q
            if ph<3:A[ph+2,ph+1]=q
        u=np.r_[1.,[gs/r[j]*(1-(q/(q+r[j]))**remain) for remain in [4,3,2,1]]]
        resid=float(max(abs(u@A-(1+r[j])*u)))
        out.append(dict(env='B200',L=ll,a=1,k=100,b0=100,shape=4,kind='chain4',r=float(r[j]),gf=float(gf),gs=float(gs),R0=float(gf/.1*gs*ll),D=float((ll+10)*math.log1p(r[j])-m[j]),D_se=float(se[j]),margin=float(m[j]),margin_se=float(se[j]),rv_residual=resid,fresh_value=float(u[1]),pass_match=bool(abs(m[j]-.005)<=max(.002,4*se[j]) and resid<=1e-10)))
    return out

def chain_step(rng,F,C,h,L,gf,gs):
    shed=pdraw(rng,100*C.sum(1)*gs)
    move=bdraw(rng,C,4/L);Cn=C-move;Cn[:,1:]+=move[:,:3]
    depart=bdraw(rng,F,.1);Cn[:,0]+=pdraw(rng,depart*(gf/.1)*h/100)
    return F-depart+shed,Cn

def chain_sim(spec,n,seed,N=10000,steps=1700,constant=False):
    rng=np.random.default_rng(seed);F=np.zeros(n);C=np.zeros((n,4));C[:,0]=N/100
    env=rng.integers(0,2,n).astype(float);ids=np.arange(n);ext=np.full(n,-1,np.int32);mins=C.sum(1);ap=np.zeros(n,bool)
    for t in range(1,steps+1):
        if not len(ids):break
        if constant:h=np.ones(len(ids))
        else:env,h=environment(rng,env,'B200')
        ap[ids[(F>=SWITCH)|(C.max(1)>=SWITCH)|(100*C.sum(1)*spec['gs']>=SWITCH)]]=True
        F,C=chain_step(rng,F,C,h,spec['L'],spec['gf'],spec['gs'])
        hosts=C.sum(1);mins[ids]=np.minimum(mins[ids],hosts);dead=(hosts==0)&(F==0)
        if np.any(dead):
            ext[ids[dead]]=t;keep=~dead;ids,F,C,env=ids[keep],F[keep],C[keep],env[keep]
    return ext,mins,ap

def chain_cell(spec,seed):
    t0=time.monotonic();e,m,a=chain_sim(spec,1000,seed)
    if sum(e>0)<20 or sum(e>0)>980:
        e2,m2,a2=chain_sim(spec,2000,seed+1000000);e=np.r_[e,e2];m=np.r_[m,m2];a=np.r_[a,a2]
    key=f"chain4_B200_L{spec['L']}"
    np.savez_compressed(ROOT/('raw_'+key+'.npz'),extinction_time=e,minimum_occupied_hosts=m,approximated=a)
    return dict(key=key,**spec,**summary(e,m,10*(spec['L']+10)),seed=seed,approx_runs=int(sum(a)),seconds=time.monotonic()-t0)

def chain_validate():
    rng=np.random.default_rng(19600);zs=[]
    for L in [10,40,160]:
      q=4/L;gf=.105;gs=1/L+.005
      for h in [0,1,2]:
        n=2000;F=np.full(n,100000.);C=np.full((n,4),250.)
        fn,cn=chain_step(rng,F,C,np.full(n,h),L,gf,gs)
        expect=np.r_[90000+100000*gs,250*(1-q)+1000*gf*h,250,250,250]
        c=gf/.1*h/100
        var=np.r_[9000+100000*gs,250*q*(1-q)+100000*(.1*c+.09*c*c),np.repeat(500*q*(1-q),3)]
        got=np.r_[fn.mean(),cn.mean(0)];zs.extend(((got-expect)/np.sqrt(var/n)).tolist())
    L=40;gf=.1*math.sqrt(1.35);gs=.025*math.sqrt(1.35);q=.1
    def iterate(steps):
        f=0.;z=np.zeros(4)
        for _ in range(steps):
            nf=.9*f+.1*math.exp(gf/.1/100*(z[0]-1))
            z=math.exp(100*gs*(f-1))*((1-q)*z+q*np.r_[z[1:],1]);f=nf
        return float(z[0])
    finite=iterate(3000);fixed=iterate(100000)
    e,m,a=chain_sim(dict(L=L,gf=gf,gs=gs),10000,19601,N=100,steps=3000,constant=True)
    p=float(np.mean(e>0));se=max(math.sqrt(finite*(1-finite)/10000),.0001)
    return dict(mean_cases=9,worst_mean_z=max(abs(x) for x in zs),p=p,fixed=fixed,finite=finite,z=(p-finite)/se,pass_check=bool(max(abs(x) for x in zs)<=4 and abs(p-finite)<=4*se and abs(finite-fixed)<.001))

def extension_main(start):
    import datetime
    wall0=campaign_wall0()
    out=[];valid=True
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(chain_calibrate):'CHAIN_CALIBRATION',pool.submit(chain_validate):'CHAIN_VALIDATION'}
        for j,(a,k) in enumerate([(0,1),(.5,100),(1,100)]):fs[pool.submit(stress_pair,a,k,19400+j)]='OUTAGE_STRESS'
        cal=None
        for f in as_completed(fs):
            tag=fs[f];res=f.result()
            if tag=='CHAIN_CALIBRATION':
                cal=res
                for row in res:emit(tag,**row)
                valid &=all(r['pass_match'] for r in res)
            else:emit(tag,**res);valid &=res['pass_check']
        if valid and time.time()-wall0<2320:
            ff=[pool.submit(chain_cell,r,19700+j) for j,r in enumerate(cal)]
            for f in as_completed(ff):
                r=f.result();out.append(r);emit('CHAIN_CELL',**r)
            (ROOT/'r18_chain_cells.json').write_text(json.dumps(out,indent=2))
        else:emit('CHAIN_NOT_RUN',valid=valid,elapsed=time.time()-wall0)
    rec=cached_reproduction();emit('CACHED_REPRODUCTION',**rec);valid &=rec['pass_check']
    checks=validation_means();(ROOT/'r18_mean_checks.json').write_text(json.dumps(checks))
    emit('FINAL_MEANS',cases=len(checks),worst_z=max(max(abs(v) for v in r['z']) for r in checks),pass_check=all(r['pass_check'] for r in checks))
    valid &=all(r['pass_check'] for r in checks)
    emit('CAMPAIGN_END',extension_seconds=time.monotonic()-start,elapsed=time.time()-wall0,chain_cells=len(out),passed=valid)


def campaign_wall0():
    if (ROOT/'r18_campaign.json').exists():
        return json.loads((ROOT/'r18_campaign.json').read_text())['started_epoch']
    import datetime
    old=json.loads(next(x[6:] for x in (ROOT/'r18_output_segment1.txt').read_text().splitlines() if x.startswith('START ')))
    return datetime.datetime.strptime(old['utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()

if __name__=='__main__': main()
