"""Forward individual worlds; first extinction times are never survivor-filtered."""
import numpy as np
from .counts import integers, compact


def extinct_by(result, horizon):
    """Count observed extinctions; a censoring sentinel is never a later event."""
    if int(horizon) != horizon or not 0 <= horizon < len(result['totals']):
        raise ValueError('extinction horizon is outside observed steps')
    return int(np.sum(result['extinction_time'] <= horizon))


def simulate(model, initial, steps, seed, hosts=None, endogenous=None, y=0., checkpoints=()):
    """One environment per replicate; return full count history and censoring.

Initial is replicate x state. Host values are relative to their arithmetic
reference. Endogenous Ricker updates follow parasite demography and use the
new stage burdens; they are driven only by current/past values. Every time
step is retained here; phase 2 may use explicit checkpointing to save space.
"""
    n = integers(initial).copy()
    if n.ndim!=2 or n.shape[1]!=model.n or np.any(n<0):
        raise ValueError('invalid initial counts')
    if (hosts is None)==(endogenous is None):
        raise ValueError('supply exactly one exogenous or endogenous host driver')
    if hosts is not None and hosts.shape!=(steps,len(n),model.d):
        raise ValueError('host history shape mismatch')
    if endogenous is not None and endogenous.H.shape!=(len(n),model.d):
        raise ValueError('endogenous host shape mismatch')
    rng = np.random.default_rng(seed)
    total = np.empty((steps+1,len(n)),object); total[0]=n.sum(axis=1, dtype=object)
    extinct = np.full(len(n),steps+1,dtype=int); extinct[total[0]==0]=0
    states = {0:n.copy()}
    h = endogenous.H/endogenous.K if endogenous is not None else None
    env = np.empty((steps,len(n),model.d)) if endogenous is not None else hosts
    for t in range(steps):
        if hosts is not None:
            h = hosts[t]
        else:
            env[t] = h
        n = model.step(n,h,rng,y)
        total[t+1] = n.sum(axis=1, dtype=object)
        died = (total[t+1]==0)&(extinct==steps+1)
        extinct[died]=t+1
        if t+1 in checkpoints:
            states[t+1]=n.copy()
        if endogenous is not None:
            h = endogenous.step(model.burden(n))
    return dict(totals=compact(total),final=n,extinction_time=extinct,
                censored=extinct>steps,states=states,hosts=env)


def finite_growth(model, hosts, y=0., initial=None, endpoints=(100,500)):
    """Per-schedule conditional-mean growth from a specified initial composition.

There is no demographic noise in this observable. All schedules, including
ones whose paired individual population dies, remain in the growth sample.
"""
    v = model.constant(y)[1] if initial is None else np.asarray(initial,float)
    x = np.broadcast_to(v/v.sum(),(hosts.shape[1],model.n)).copy()
    logs = np.zeros(len(x)); out = {}
    for t,h in enumerate(hosts,1):
        x = np.einsum('rij,rj->ri',model.matrix(h,y),x)
        total=x.sum(axis=1)
        if np.any(total<=0):
            raise FloatingPointError('zero conditional expectation')
        logs+=np.log(total); x/=total[:,None]
        if t in endpoints:
            out[t]=logs.copy()/t
    return out
