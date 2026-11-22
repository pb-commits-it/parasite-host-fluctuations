"""Exact discrete-time individual model and its conditional expectation matrix.

An individual chooses one event. Conditional on that event it continues once
or leaves independent Poisson offspring. After demography, every individual
ending in the mobile stage moves independently, including continuing residents. No tau leap, population cap, density regulation or shared group
fate is hidden in the sampler. Two to four biological stages may expand into
many residence states. One designated transfer receives a log birth tilt.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import optimize
from .durations import Duration
from .counts import integers, compact, binomial, poisson_product


@dataclass
class Stage:
    name: str
    duration: Duration
    death: float = 0.
    host: int | None = None
    mode: str = 'waste'


@dataclass
class Route:
    source: int
    target: int
    host: int | None = None
    share: float = 1.
    yield_: float = 1.
    birth: bool = False
    poisson: bool = True


@dataclass
class Event:
    source: int
    target: int
    probability: float
    changes: np.ndarray
    yield_: float
    driver: int | None
    birth: bool
    poisson: bool
    # Hazard mode uses H-dependent survival/contact probabilities directly.
    hazard: tuple | None = None


class Model:
    """Compile residence kernels and routes into independent parental events.

mode='waste': independent death, then host-independent departure; host response
multiplies transfer offspring. mode='linear_delay': bounded contact q*h and
death compete, q=1/mean-death; invalid probabilities raise rather than clip.
mode='delay': geometric readiness with fixed independent death d, contact
probability (1-d)*(1-exp(-c*f(h))), where c is chosen to match total exit
1/mean at h=1. Delay modes require one route (no alternative-host ambiguity).
Gamma/empirical kernels apply to wasting/development stages; a delayed search
is represented as its own geometric stage. Both stages count toward k<=4.
"""
    def __init__(self, stages, routes, drivers=1, patches=1, movement=0.,
                 mobile_stage=None, saturation=False):
        self.stages, self.routes = stages,routes
        if not 2 <= len(stages) <= 4 or len({s.name for s in stages}) != len(stages):
            raise ValueError('require 2..4 uniquely named biological stages')
        if int(patches)!=patches or patches<1 or not 0<=movement<=1 or drivers<1:
            raise ValueError('invalid host/patch specification')
        if mobile_stage is not None and not 0<=mobile_stage<len(stages):
            raise ValueError('invalid mobile stage')
        self.drivers, self.patches = drivers,patches
        self.d = drivers*patches
        self.saturation = saturation
        self.spans = []
        n = 0
        for s in stages:
            if not 0<=s.death<1 or s.mode not in ('waste','delay','linear_delay'):
                raise ValueError('invalid stage death or scarcity rule')
            if s.host is not None and not 0<=s.host<drivers:
                raise ValueError('invalid stage host')
            self.spans.append(list(range(n,n+len(s.duration.exit))))
            n += len(s.duration.exit)
        self.k, self.n = n,n*patches
        for r in routes:
            if not 0<=r.source<len(stages) or not 0<=r.target<len(stages) or r.target!=(r.source+1)%len(stages):
                raise ValueError('routes must follow the fixed cyclic stage order')
            if r.share<0 or r.yield_<=0 or (r.host is not None and not 0<=r.host<drivers):
                raise ValueError('invalid route')
            if not r.poisson and (r.yield_!=1 or r.host is not None or r.birth):
                raise ValueError('single continuation needs unit yield, no host multiplier or birth tilt')
        if not any(r.birth for r in routes):
            raise ValueError('mark the circuit-completion birth transfer')
        self.events = []
        zero = np.zeros(self.d)
        for p in range(patches):
            for j,s in enumerate(stages):
                rr = [r for r in routes if r.source==j and r.share>0]
                if not rr or not np.isclose(sum(r.share for r in rr),1.,atol=1e-12):
                    raise ValueError('outgoing route shares must sum to one')
                ids = [p*n+i for i in self.spans[j]]
                if s.mode != 'waste':
                    if s.duration.label!='geometric' or len(rr)!=1 or rr[0].host is None:
                        raise ValueError('delayed search requires geometric residence and one host route')
                    r = rr[0]; h = p*drivers+r.host
                    q = 1/s.duration.mean
                    if not s.death<q<1:
                        raise ValueError('delay needs positive contact and positive staying')
                    dest = p*n+self.spans[r.target][0]
                    if s.mode=='linear_delay':
                        c = q-s.death
                        changes = zero.copy(); changes[h] = -c
                        self.events.append(Event(ids[0],ids[0],1-q,changes,1.,None,False,False))
                        changes = -changes
                        self.events.append(Event(ids[0],dest,c,changes,r.yield_,None,r.birth,r.poisson))
                    else:
                        c = -math.log((1-q)/(1-s.death))
                        self.events.append(Event(ids[0],ids[0],1-q,zero,1.,None,False,False,(h,c,s.death,False)))
                        self.events.append(Event(ids[0],dest,q-s.death,zero,r.yield_,None,r.birth,r.poisson,(h,c,s.death,True)))
                    continue
                for age,src in enumerate(ids):
                    for nxt in np.flatnonzero(s.duration.U[:,age]):
                        prob = (1-s.death)*s.duration.U[nxt,age]
                        self.events.append(Event(src,ids[nxt],prob,zero,1.,None,False,False))
                    for r in rr:
                        prob = (1-s.death)*s.duration.exit[age]*r.share
                        if prob:
                            h = None if r.host is None else p*drivers+r.host
                            self.events.append(Event(src,p*n+self.spans[r.target][0],prob,zero,r.yield_,h,r.birth,r.poisson))
        self.move = np.eye(self.n)
        if mobile_stage is not None:
            W = (1-movement)*np.eye(patches)+movement*np.ones((patches,patches))/patches
            for state in self.spans[mobile_stage]:
                ids = np.arange(patches)*n+state
                self.move[np.ix_(ids,ids)] = W
        self.groups = [[e for e in self.events if e.source==j] for j in range(self.n)]
        self.affine = not any(e.hazard for e in self.events)
        self.parts = np.zeros((2,self.n,self.n))
        self.noise_parts = np.zeros((2,self.d,self.n,self.n))
        for e in self.events:
            b = int(e.birth)
            target = self.move[:,e.target]
            self.parts[b,:,e.source] += target*e.probability*e.yield_
            changes = e.changes.copy()
            if e.driver is not None:
                if np.any(changes):
                    raise ValueError('quadratic host event unsupported')
                changes[e.driver] += e.probability
            self.noise_parts[b,:,:,e.source] += changes[:,None]*target[None,:]*e.yield_

    def rates(self, hosts):
        h = np.asarray(hosts,float)
        if h.shape[-1]!=self.d or np.any(h<0) or np.any(~np.isfinite(h)):
            raise ValueError('hosts must be finite nonnegative values for every driver')
        return 2*h/(1+h) if self.saturation else h

    def event_values(self, event, rates, y):
        e = event
        prob = e.probability+rates@e.changes-e.changes.sum()
        if e.hazard:
            h,c,d,exit_ = e.hazard
            survival = np.exp(-c*rates[...,h])
            prob = (1-d)*(1-survival if exit_ else survival)
        mu = e.yield_*np.exp(y if e.birth else 0.)
        if e.driver is not None:
            mu = mu*rates[...,e.driver]
        if np.any(prob < -1e-12) or np.any(prob > 1+1e-12):
            raise ValueError('invalid event probability; no clipping of host forcing')
        return np.clip(prob,0.,1.),mu

    def matrix(self, hosts=None, y=0.):
        rates = self.rates(np.ones(self.d) if hosts is None else hosts)
        if self.affine:
            A = self.parts[0]+np.exp(y)*self.parts[1]
            B = self.noise_parts[0]+np.exp(y)*self.noise_parts[1]
            out = A+np.einsum('...d,dij->...ij',rates-1,B)
            if np.any(out < -1e-12):
                raise ValueError('host forcing makes a negative matrix entry')
            return out
        out = np.zeros(rates.shape[:-1]+(self.n,self.n))
        for e in self.events:
            pr,mu = self.event_values(e,rates,y)
            out[..., :,e.source] += (pr*mu)[...,None]*self.move[:,e.target]
        return out

    def constant(self,y=0.):
        A = self.matrix(y=y)
        vals, vec = np.linalg.eig(A)
        i = np.argmax(vals.real)
        v = np.abs(vec[:,i].real); v/=v.sum()
        return float(vals[i].real),v

    def critical_tilt(self):
        return float(optimize.brentq(lambda y: np.log(self.constant(y)[0]),-20,20,xtol=1e-13))

    def step(self, counts, hosts, rng, y=0.):
        """Exact count aggregation of the independent-individual event law."""
        counts = integers(counts)
        if counts.ndim != 2 or counts.shape[1]!=self.n or np.any(counts<0):
            raise ValueError('counts must be a nonnegative integer replicate x state array')
        rates = self.rates(hosts)
        if rates.shape != (len(counts),self.d):
            raise ValueError('one complete environment per replicate required')
        new = np.zeros(counts.shape,dtype=object)
        for src,events in enumerate(self.groups):
            left = counts[:,src].copy()
            prob_left = np.ones(len(counts))
            for e in events:
                prob,mu = self.event_values(e,rates,y)
                if np.any(prob>prob_left+1e-10):
                    raise ValueError('parental event probabilities exceed one')
                conditional = np.divide(prob,prob_left,out=np.zeros_like(prob_left),where=prob_left>1e-14)
                chosen = binomial(left,np.clip(conditional,0,1),rng)
                left -= chosen; prob_left -= prob
                babies = poisson_product(chosen,mu,rng) if e.poisson else chosen
                dest_left = babies.copy(); mass = 1.
                targets = np.flatnonzero(self.move[:,e.target])
                for dest in targets[:-1]:
                    p = self.move[dest,e.target]
                    moved = binomial(dest_left,min(1.,p/mass),rng)
                    new[:,dest] += moved; dest_left -= moved; mass -= p
                new[:,targets[-1]] += dest_left
        return compact(new)

    def burden(self, counts):
        """Number in host-labelled stages, separately for each patch and host."""
        out = np.zeros((len(counts),self.d))
        for p in range(self.patches):
            for i,s in enumerate(self.stages):
                if s.host is not None:
                    out[:,p*self.drivers+s.host] += np.asarray(counts[:,np.array(self.spans[i])+p*self.k].sum(axis=1, dtype=object), float)
        return out


def ring(means, drivers=1, death=0., mode='waste', patches=1, movement=0., alternatives=False):
    """Convenience ring: one exposed transfer, Poisson offspring at every exit."""
    stages = [Stage(str(i),Duration.geometric(m),death if i==1 else 0.,
                    0 if i==1 else None,mode if i==0 else 'waste') for i,m in enumerate(means)]
    rr = []
    for i in range(len(means)):
        if i==0 and alternatives:
            rr.extend(Route(i,1,host=h,share=1/drivers) for h in range(drivers))
        else:
            rr.append(Route(i,(i+1)%len(means),host=0 if i==0 else None,birth=i==len(means)-1))
    return Model(stages,rr,drivers,patches,movement,1 if patches>1 else None)
