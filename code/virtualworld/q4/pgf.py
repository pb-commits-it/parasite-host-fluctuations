"""Exact finite-horizon persistence for a replayed environment and engine law.

No population simulation is approximated: this computes branching extinction
probabilities from the engine's independent-parent event generating functions.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *

def extinction(m,h,initial,horizons,y=.05):
    h=np.asarray(h);rates=m.rates(h[:,None]);q=np.zeros((len(h),m.n))
    initial=np.asarray(initial);active=initial>0
    events=[];missing=np.ones_like(q)
    for e in m.events:
        if not np.array_equal(m.move,np.eye(m.n)):raise ValueError('PGF implementation scoped to nonmoving rings')
        prob,mu=m.event_values(e,rates,y)
        prob=np.broadcast_to(prob,(len(h),));mu=np.broadcast_to(mu,(len(h),))
        events.append((e.source,e.target,prob,mu,e.poisson));missing[:,e.source]-=prob
    if np.min(missing)<-1e-12:raise ValueError('invalid event sum')
    missing=np.maximum(missing,0);out={}
    for t in range(1,max(horizons)+1):
        future=np.roll(q,-1,axis=0);q=missing.copy()
        for src,dst,prob,mu,poisson in events:
            value=np.exp(mu*(future[:,dst]-1)) if poisson else future[:,dst]
            q[:,src]+=prob*value
        if np.any(q< -1e-12) or np.any(q>1+1e-12):raise FloatingPointError('invalid extinction probability')
        if t in horizons:
            with np.errstate(divide='ignore'):
                survival=-np.expm1(np.log(q[:,active])@initial[active])
            out[t]=survival
    return out

def direct_ring(h,means_,initial,horizon,y):
    """Independent specialized recurrence; does not use engine event compiler."""
    q=np.zeros((len(h),len(means_)))
    for t in range(horizon):
        nxt=np.roll(q,-1,axis=0);out=np.empty_like(q)
        for i,tau in enumerate(means_):
            mu=h if i==0 else (np.exp(y) if i==len(means_)-1 else 1.)
            out[:,i]=(1-1/tau)*nxt[:,i]+np.exp(mu*(nxt[:,(i+1)%len(means_)]-1))/tau
        q=out
    return 1-np.prod(q**np.asarray(initial),axis=1)

if __name__=='__main__':
    from world.simulation import simulate
    from world.design import allocate
    dest=P/'pgf-validation.json'
    if dest.exists():raise RuntimeError('append-only')
    start=time.monotonic();checks=[]
    def check(name,x,y,tol_):
        checks.append(dict(check=name,actual=float(x),expected=float(y),error=float(abs(x-y)),tolerance=float(tol_),passed=bool(abs(x-y)<=tol_)))
    for y in (-.1,0.,.1):
        m=model(2,5);out=extinction(m,np.ones(1),[10,0],(20,60,120),y)
        a=b=0.
        for t in range(1,121):
            a,b=.5*a+.5*np.exp(b-1),.8*b+.2*np.exp(np.exp(y)*(a-1))
            if t in out:check(f'phase1 branching y={y} t={t}',out[t][0],1-a**10,1e-12)
    h=norm_host(np.exp(np.array([.2,-.6,.3,1.,.4,-.2,.7,-.9,-.3,.6,-.8,-.4])))
    for k in (2,3):
        for s in (1,5,20):
            m=model(k,s);init=allocate(20,m.constant(.05)[1]);out=extinction(m,h,init,(120,))[120]
            ref=direct_ring(h,means(k,s),init,120,.05)
            check(f'periodic independent PGF k={k} s={s}',np.max(abs(out-ref)),0,1e-12)
    m=model(2,5);init=allocate(20,m.constant(.05)[1]);phases=np.random.default_rng(415900).integers(0,len(h),10000)
    hist=h[(np.arange(120)[:,None]+phases)%len(h)][:,:,None]
    sim=simulate(m,np.tile(init,(10000,1)),120,415901,hosts=hist,y=.05)
    probs=extinction(m,h,init,(120,))[120][phases]
    obs=float(np.mean(sim['extinction_time']>120));p=float(probs.mean());se_=np.sqrt(np.sum(probs*(1-probs)))/10000
    check('periodic engine Monte Carlo 10000',obs,p,4*se_+.002)
    result=dict(date=date(),seconds=time.monotonic()-start,passed=all(c['passed'] for c in checks),checks=checks,
        protocol_sha256=digest(P/'protocol.md'),code_sha256=digest(P/'pgf.py'))
    write_json(dest,result);print(len(checks),'checks',sum(c['passed'] for c in checks),'passed')
    if not result['passed']:sys.exit(1)
