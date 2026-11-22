"""Data-described discrete life cycles. Run clients with python3 -I.

Columns are parents, rows are descendants. Demography precedes movement.
Stage mean is the development residence time before independent mortality;
for contact stages it is 1/(death+mean contact), as in the supplied note.
A chain is a sum of geometric stays. A fixed delay must be an integer.
Transfers choose a route, then produce offspring; internal progression and
staying preserve one individual. All noisy rates are arithmetic-mean centred.

D is the common birth tilt needed for stochastic criticality, minus the
constant-environment tilt. R0 is rho(F (I-U)^-1). Its birth convention is part
of the description. C and J2 are per step; D2, J and V_generation are per
reference birth (divide by b). No expansion is returned for periodic graphs.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from itertools import product
from math import gcd
import math
import numpy as np
from scipy import linalg, optimize, signal, sparse


class ModelError(ValueError):
    pass


class PeriodicModelError(ModelError):
    pass


@dataclass(frozen=True)
class Stage:
    name: str
    mean: float
    shape: str = 'memoryless'
    substages: int = 1
    death: float = 0.0
    contact: bool = False


@dataclass(frozen=True)
class Transfer:
    source: str
    target: str
    share: float = 1.0
    yield_: float = 1.0
    driver: str | None = None
    elasticity: float = 1.0
    birth: bool = False


@dataclass(frozen=True)
class Driver:
    name: str
    sigma: float
    phi: float
    kind: str = 'lognormal'
    low: float | None = None

    def levels(self):
        """Two levels match mean 1 and lognormal raw variance exp(sigma²)-1.

        low fixes the standardized shape of the small-amplitude family.
        The default symmetric law is available only when its low is positive.
        """
        variance = np.expm1(self.sigma**2)
        low = 1 - math.sqrt(variance) if self.low is None else self.low
        if variance == 0:
            return np.array([1., 1.]), np.array([.5, .5])
        if not 0 < low < 1:
            raise ModelError('bounded multiplier requires 0 < low < 1')
        a = 1-low
        high = 1 + variance/a
        phigh = a*a/(a*a+variance)
        return np.array([low, high]), np.array([1-phigh, phigh])


@dataclass(frozen=True)
class Patches:
    count: int = 1
    synchrony: float = 0.0
    mobile_stage: str | None = None
    movement: float = 0.0
    topology: str = 'island'
    transit_loss: float = 0.0


@dataclass
class Description:
    stages: list[Stage]
    transfers: list[Transfer]
    drivers: list[Driver] = field(default_factory=list)
    correlation: np.ndarray | None = None
    patches: Patches = field(default_factory=Patches)


def _psd_root(a):
    vals, vec = linalg.eigh(a)
    if vals.min(initial=0) < -1e-10:
        raise ModelError('infeasible covariance (not positive semidefinite)')
    return vec * np.sqrt(np.maximum(vals, 0))


def _perron(a):
    vals, left, right = linalg.eig(a, left=True, right=True)
    k = int(np.argmax(vals.real))
    lam = float(vals[k].real)
    v, w = np.abs(right[:, k].real), np.abs(left[:, k].real)
    if lam <= 0 or v.min() <= 0 or w.min() <= 0:
        raise ModelError('a positive irreducible recurrent life cycle is required')
    v /= v.sum()
    w /= w@v
    return lam, v, w


def _period(a):
    n = len(a)
    adj = a > 0
    count = sparse.csgraph.connected_components(sparse.csr_matrix(adj), directed=True,
                                               connection='strong', return_labels=False)
    if count != 1:
        raise ModelError('reducible mean cycle: split recurrent blocks explicitly')
    dist = [-1]*n
    dist[0] = 0
    queue = [0]
    for j in queue:
        for i in np.flatnonzero(adj[:, j]):
            if dist[i] < 0:
                dist[i] = dist[j]+1
                queue.append(int(i))
    period = 0
    for i, j in zip(*np.nonzero(adj)):
        period = gcd(period, abs(dist[j]+1-dist[i]))
    return max(1, period)


class Environment:
    """Causal diagonal AR model; covariance at lag l is diag(phi**l) R.

    Gaussian innovation covariance is R_ij(1-phi_i phi_j), checked PSD.
    Correlated two-state components use an explicitly fitted stationary joint
    law and a common-refresh chain. They require common nonnegative memory.
    Independent two-state components permit any feasible binary memory.
    Mixed types must have zero cross-correlation. Unsupported or infeasible
    specifications are refused, never approximated with a Gaussian copula.
    """
    def __init__(self, drivers, corr):
        self.drivers = drivers
        self.d = len(drivers)
        self.corr = np.asarray(corr, float)
        self.phi = np.array([d.phi for d in drivers])
        self.sigma = np.array([d.sigma for d in drivers])
        if any(not (-1 < d.phi < 1) or d.sigma < 0 for d in drivers):
            raise ModelError('require sigma >= 0 and -1 < memory < 1')
        if self.corr.shape != (self.d, self.d) or not np.allclose(self.corr, self.corr.T):
            raise ModelError('correlation has wrong shape or is not symmetric')
        if self.d and not np.allclose(np.diag(self.corr), 1):
            raise ModelError('correlation diagonal must be one')
        _psd_root(self.corr)
        self.gauss = np.array([i for i,d in enumerate(drivers) if d.kind == 'lognormal'], int)
        self.binary = np.array([i for i,d in enumerate(drivers) if d.kind == 'bounded'], int)
        if len(self.gauss)+len(self.binary) != self.d:
            raise ModelError('driver kind must be lognormal or bounded')
        if len(self.gauss) and len(self.binary) and np.any(abs(self.corr[np.ix_(self.gauss,self.binary)]) > 1e-12):
            raise ModelError('cross-correlated Gaussian and binary drivers are not specified')
        g = self.gauss
        if len(g):
            r = self.corr[np.ix_(g,g)]
            self.ginitial = _psd_root(r)
            self.ginnovation = _psd_root(r*(1-self.phi[g,None]*self.phi[None,g]))
        self.levels = np.ones((self.d,2))
        self.probs = np.full((self.d,2), .5)
        for i in self.binary:
            self.levels[i], self.probs[i] = drivers[i].levels()
            p = self.probs[i,1]
            if min((1-self.phi[i])*p, (1-self.phi[i])*(1-p)) < 0 or max((1-self.phi[i])*p, (1-self.phi[i])*(1-p)) > 1+1e-12:
                raise ModelError('memory infeasible for the two-state marginal')
        self.joint_states = self.joint_probs = None
        b = self.binary
        if len(b) and not np.allclose(self.corr[np.ix_(b,b)], np.eye(len(b))):
            if len(b) > 12:
                raise ModelError('exact correlated binary joint law limited to 12 components')
            if np.ptp(self.phi[b]) > 1e-12 or self.phi[b[0]] < 0:
                raise ModelError('correlated bounded drivers need common nonnegative memory')
            states = np.array(list(product([0,1], repeat=len(b))))
            p = self.probs[b,1]
            z = (states-p)/np.sqrt(p*(1-p))
            constraints = [np.ones(len(states))] + [z[:,i] for i in range(len(b))]
            targets = [1.] + [0.]*len(b)
            for i in range(len(b)):
                for j in range(i):
                    constraints.append(z[:,i]*z[:,j]); targets.append(self.corr[b[i],b[j]])
            fit = optimize.linprog(np.zeros(len(states)), A_eq=constraints, b_eq=targets,
                                   bounds=(0,None), method='highs')
            if not fit.success:
                raise ModelError('no two-state joint law has the requested correlation')
            keep = fit.x > 1e-13
            self.joint_states = states[keep]
            self.joint_probs = fit.x[keep]/fit.x[keep].sum()

    def sample(self, steps, replicates, seed, antithetic=False, standardized=False):
        rng = np.random.default_rng(seed)
        n = replicates*2 if antithetic else replicates
        out = np.ones((steps,n,self.d))
        def uniforms(shape):
            u = rng.random(shape)
            return np.concatenate([u,1-u], axis=1) if antithetic else u
        g = self.gauss
        if len(g):
            z0 = rng.standard_normal((replicates,len(g)))@self.ginitial.T
            eps = rng.standard_normal((steps,replicates,len(g)))@self.ginnovation.T
            if antithetic:
                z0=np.concatenate([z0,-z0]); eps=np.concatenate([eps,-eps],axis=1)
            for k,i in enumerate(g):
                z,_ = signal.lfilter([1.],[1.,-self.phi[i]], eps[:,:,k], axis=0,
                                     zi=(self.phi[i]*z0[:,k])[None,:])
                out[:,:,i] = z if standardized else np.exp(self.sigma[i]*z-self.sigma[i]**2/2)
        b = self.binary
        if len(b):
            groups = [list(b)] if self.joint_states is not None else [[int(i)] for i in b]
            for group in groups:
                phi = self.phi[group[0]]
                if self.joint_states is not None:
                    states, probs = self.joint_states, self.joint_probs
                else:
                    states, probs = np.array([[0],[1]]), self.probs[group[0]]
                if phi >= 0:
                    draw = np.searchsorted(np.cumsum(probs), uniforms((steps,replicates)))
                    fresh = uniforms((steps,replicates)) < 1-phi
                    fresh[0] = True
                    last = np.maximum.accumulate(np.where(fresh,np.arange(steps)[:,None],0),axis=0)
                    indices = draw[last,np.arange(n)[None,:]]
                else:
                    u = uniforms((steps,replicates))
                    indices = np.empty((steps,n),int)
                    indices[0] = u[0] > probs[0]
                    for t in range(1,steps):
                        phigh = phi*indices[t-1]+(1-phi)*probs[1]
                        indices[t] = u[t] < phigh
                state = states[indices]
                for k,i in enumerate(group):
                    h = self.levels[i,state[:,:,k]]
                    sd = math.sqrt(np.expm1(self.sigma[i]**2))
                    out[:,:,i] = (h-1)/sd if standardized and sd else (np.zeros_like(h) if standardized else h)
        return out

    def marginal_states(self, ids):
        """Exact small binary marginal, including requested dependence."""
        ids = list(ids)
        if any(i not in self.binary for i in ids):
            raise ModelError('only bounded drivers may change staying probabilities')
        if self.joint_states is not None:
            positions = [list(self.binary).index(i) for i in ids]
            return self.levels[np.array(ids)[None,:],self.joint_states[:,positions]], self.joint_probs
        states = np.array(list(product([0,1],repeat=len(ids))),int)
        return self.levels[np.array(ids)[None,:],states], np.prod(self.probs[np.array(ids)[None,:],states],axis=1)


class Engine:
    def __init__(self, description: Description, birth_scale=1.):
        self.description = description
        self.birth_scale = float(birth_scale)
        stages, routes, patch = description.stages, description.transfers, description.patches
        names = [s.name for s in stages]
        if len(set(names)) != len(names) or not stages:
            raise ModelError('stage names must be unique and nonempty')
        dnames = [d.name for d in description.drivers]
        if len(set(dnames)) != len(dnames):
            raise ModelError('driver names must be unique')
        if any(r.source not in names or r.target not in names or (r.driver is not None and r.driver not in dnames) for r in routes):
            raise ModelError('unknown stage or driver')
        if not any(r.birth for r in routes):
            raise ModelError('at least one transfer must be marked as birth')
        if any(r.share < 0 or r.yield_ <= 0 for r in routes):
            raise ModelError('shares nonnegative and yields positive required')
        if patch.count < 1 or int(patch.count) != patch.count or not 0<=patch.movement<=1 or not 0<=patch.transit_loss<=1:
            raise ModelError('invalid patch movement')
        if patch.mobile_stage is not None and patch.mobile_stage not in names:
            raise ModelError('unknown mobile stage')
        self.micro = []
        self.span = {}
        for s in stages:
            if s.mean < 1 or not 0<=s.death<1:
                raise ModelError('mean stay >= 1 and 0 <= death < 1 required')
            if s.shape == 'memoryless':
                k, hazard = 1, 1/s.mean
            elif s.shape == 'chain':
                k = int(s.substages)
                if k < 1 or k != s.substages or s.mean < k:
                    raise ModelError('chain mean cannot be shorter than its sub-stage count')
                hazard = k/s.mean
            elif s.shape == 'fixed':
                k = int(round(s.mean))
                if abs(k-s.mean)>1e-10:
                    raise ModelError('fixed delay must be an integer number of steps')
                hazard = 1.
            else:
                raise ModelError('unknown stay shape')
            if s.contact and (s.shape != 'memoryless' or s.mean <= 1 or s.death >= 1/s.mean):
                raise ModelError('contact requires memoryless timing, positive staying and positive contact')
            ids = list(range(len(self.micro),len(self.micro)+k))
            self.span[s.name] = ids
            self.micro += [(s,hazard,q==k-1) for q in range(k)]
        self.k = len(self.micro); self.n = patch.count*self.k
        nd = len(description.drivers)
        corr = np.eye(nd) if description.correlation is None else np.asarray(description.correlation,float)
        space = np.full((patch.count,patch.count),patch.synchrony); np.fill_diagonal(space,1.)
        drivers = [replace(d,name=f'{p}:{d.name}') for p in range(patch.count) for d in description.drivers]
        self.environment = Environment(drivers,np.kron(space,corr))
        self.rate_keys = []
        def key(p,r):
            if r.driver is None: return None
            item = (p*nd+dnames.index(r.driver),float(r.elasticity))
            if item not in self.rate_keys: self.rate_keys.append(item)
            return self.rate_keys.index(item)
        # Events: source, pre-movement target, mean probability, perturbations,
        # mean offspring yield, yield rate key, birth, deterministic continuation.
        self.events = []
        for p in range(patch.count):
            for j,(s,hazard,last) in enumerate(self.micro):
                src = p*self.k+j
                exits = [r for r in routes if r.source == s.name]
                if not np.isclose(sum(r.share for r in exits),1.):
                    raise ModelError('outgoing route shares must sum to one')
                stay = (1-s.death)*(1-hazard)
                stay_changes = {}
                if s.contact:
                    c = hazard-s.death
                    stay = 1-hazard
                    for r in exits:
                        ki = key(p,r)
                        if ki is not None:
                            di,_ = self.rate_keys[ki]
                            if drivers[di].kind != 'bounded':
                                raise ModelError('contact-limited stages require bounded drivers')
                            stay_changes[ki] = stay_changes.get(ki,0.)-c*r.share
                if stay > 0:
                    self.events.append((src,src,stay,stay_changes,1.,None,False,True))
                if not last:
                    self.events.append((src,src+1,(1-s.death)*hazard,{},1.,None,False,True))
                else:
                    for r in exits:
                        if r.share == 0: continue
                        ki = key(p,r)
                        prob = (hazard-s.death if s.contact else (1-s.death)*hazard)*r.share
                        changes = {ki:prob} if s.contact and ki is not None else {}
                        ykey = None if s.contact else ki
                        self.events.append((src,p*self.k+self.span[r.target][0],prob,changes,
                                            r.yield_*(birth_scale if r.birth else 1),ykey,r.birth,False))
        self.rate_normalizers = np.ones(len(self.rate_keys))
        for ki,(di,eta) in enumerate(self.rate_keys):
            if drivers[di].kind=='bounded':
                self.rate_normalizers[ki] = self.environment.probs[di]@(self.environment.levels[di]**eta)
        self.movement = np.eye(self.n)
        if patch.count>1 or patch.transit_loss:
            if patch.topology == 'island':
                W = np.ones((patch.count,patch.count))/patch.count
            elif patch.topology == 'nearest':
                W = np.zeros((patch.count,patch.count))
                for p in range(patch.count):
                    W[(p-1)%patch.count,p]+=.5; W[(p+1)%patch.count,p]+=.5
            else: raise ModelError('topology must be island or nearest (periodic boundary)')
            K = (1-patch.movement)*np.eye(patch.count)+patch.movement*(1-patch.transit_loss)*W
            for j,(s,_,_) in enumerate(self.micro):
                if s.name==patch.mobile_stage:
                    ids = np.arange(patch.count)*self.k+j
                    self.movement[np.ix_(ids,ids)] = K
        self.edges=[]
        for src,dst,prob,changes,yield_,ykey,birth,det in self.events:
            for target in np.flatnonzero(self.movement[:,dst]):
                base = prob*yield_*self.movement[target,dst]
                coeff = {key:amount/prob for key,amount in changes.items()}
                if ykey is not None: coeff[ykey]=1.
                self.edges.append((src,int(target),base,coeff,birth))
        self.base_parts=np.zeros((2,self.n,self.n))
        self.rate_parts=np.zeros((2,len(self.rate_keys),self.n,self.n))
        for src,dst,base,coeff,birth in self.edges:
            self.base_parts[int(birth),dst,src] += base
            for ki,value in coeff.items():
                self.rate_parts[int(birth),ki,dst,src] += base*value
            if coeff and any(value<0 for value in coeff.values()):
                ids=sorted(set(self.rate_keys[ki][0] for ki in coeff))
                hs,ps=self.environment.marginal_states(ids)
                mult=np.ones(len(ps))
                for ki,value in coeff.items():
                    di,eta=self.rate_keys[ki]
                    rate=hs[:,ids.index(di)]**eta/self.rate_normalizers[ki]
                    mult += value*(rate-1)
                if mult.min()<=1e-12:
                    raise ModelError('contact probability bound violated: staying must remain positive at every host state')
        self.mean_matrix = self.base_parts.sum(axis=0)
        self.period=_period(self.mean_matrix)
        self.growth_factor,self.v,self.w=_perron(self.mean_matrix)
        self.reference_chain=self.w[:,None]*self.mean_matrix/(self.growth_factor*self.w[None,:])
        self.stationary=self.v*self.w
        self.edge_frequencies=np.array([self.w[i]*a*self.v[j]/self.growth_factor for j,i,a,_,_ in self.edges])
        self.b=float(self.w@self.base_parts[1]@self.v/self.growth_factor)
        U,F=self.base_parts
        if max(abs(linalg.eigvals(U)))>=1-1e-12:
            raise ModelError('nonbirth subsystem must be transient for finite R0')
        self.R0=float(max(abs(linalg.eigvals(linalg.solve((np.eye(self.n)-U).T,F.T).T))))

    def critical(self):
        return Engine(self.description,self.birth_scale/self.R0)

    def rates(self, hosts):
        rates=np.empty(hosts.shape[:-1]+(len(self.rate_keys),))
        for ki,(di,eta) in enumerate(self.rate_keys):
            d=self.environment.drivers[di]
            if d.kind=='lognormal':
                rates[...,ki]=hosts[...,di]**eta*np.exp((eta-eta*eta)*d.sigma**2/2)
            else: rates[...,ki]=hosts[...,di]**eta/self.rate_normalizers[ki]
        return rates

    def matrix(self, hosts=None, y=0.):
        base=self.base_parts[0]+np.exp(y)*self.base_parts[1]
        if hosts is None:return base
        parts=self.rate_parts[0]+np.exp(y)*self.rate_parts[1]
        return base+np.einsum('...k,kij->...ij',self.rates(np.asarray(hosts))-1,parts)

    def sensitivities(self):
        B=np.zeros((self.environment.d,self.n,self.n))
        for ki,(di,eta) in enumerate(self.rate_keys):
            B[di]+=self.rate_parts[:,ki].sum(axis=0)*eta*self.environment.sigma[di]/self.growth_factor
        return B

    def coefficient(self, threshold=True):
        """General resolvent coefficient at the birth threshold by default.

        Set threshold=False for the growth correction at the supplied mean
        matrix. Away from threshold, C/b is not a loss of log R0, so that
        mode deliberately returns D2=None. The critical reference is obtained
        only by scaling birth edges, never by rescaling the entire matrix.
        """
        if threshold and abs(self.R0-1)>1e-10:
            answer=self.critical().coefficient()
            answer['calibrated_from_R0']=self.R0
            return answer
        if self.period != 1:
            raise PeriodicModelError(f'mean graph has period {self.period}; second-order mixing formula does not apply')
        B=self.sensitivities(); Q=np.eye(self.n)-np.outer(self.v,self.w)
        T=self.mean_matrix/self.growth_factor@Q
        c=np.einsum('i,dij,j->d',self.w,B,self.v)
        R=self.environment.corr
        C=float(c@R@c/2)
        for i,phi in enumerate(self.environment.phi):
            rhs=Q@np.einsum('d,dij,j->i',R[i],B,self.v)
            C-=phi*float(self.w@B[i]@linalg.solve(np.eye(self.n)-phi*T,rhs))
        return {'C':C,'D2':C/self.b if abs(self.R0-1)<1e-10 else None,
                'b':self.b,'J2':self.jensen(quadratic=True)}

    def edge_rewards(self):
        rewards=np.zeros((len(self.edges),self.environment.d))
        for e,(_,_,_,coeff,_) in enumerate(self.edges):
            for ki,val in coeff.items():
                di,eta=self.rate_keys[ki]
                rewards[e,di]+=val*eta*self.environment.sigma[di]
        return rewards

    def jensen(self, quadratic=False, transfer_only=False):
        """Per-step edge bound (divide by b for birth-log threshold bound)."""
        if quadratic:
            K=self.edge_rewards()
            return float(self.edge_frequencies@np.einsum('ed,df,ef->e',K,self.environment.corr,K)/2)
        penalties=[]
        for _,_,_,coeff,_ in self.edges:
            if not coeff or (transfer_only and any(v<0 for v in coeff.values())):
                penalties.append(0.);continue
            if len(coeff)==1 and next(iter(coeff.values()))==1:
                ki=next(iter(coeff));di,eta=self.rate_keys[ki];d=self.environment.drivers[di]
                if d.kind=='lognormal':penalty=eta*eta*d.sigma*d.sigma/2
                else:penalty=math.log(self.rate_normalizers[ki])-eta*(self.environment.probs[di]@np.log(self.environment.levels[di]))
            else:
                ids=sorted(set(self.rate_keys[ki][0] for ki in coeff))
                hs,ps=self.environment.marginal_states(ids);mult=np.ones(len(ps))
                for ki,val in coeff.items():
                    di,eta=self.rate_keys[ki]
                    mult+=val*(hs[:,ids.index(di)]**eta/self.rate_normalizers[ki]-1)
                penalty=-float(ps@np.log(mult))
            penalties.append(penalty)
        return float(self.edge_frequencies@penalties)

    def _products(self, rates, y, burn, pairs):
        reps=rates.shape[1];x=np.tile(self.v,(reps,1));dx=np.zeros_like(x)
        if self.period>1:x[:]=0.;x[:,0]=1.
        logs=np.zeros(reps);slopes=np.zeros(reps)
        A=self.base_parts[0]+np.exp(y)*self.base_parts[1]
        Ay=np.exp(y)*self.base_parts[1]
        parts=self.rate_parts[0]+np.exp(y)*self.rate_parts[1]
        parts_y=np.exp(y)*self.rate_parts[1]
        # For small stage systems, batched matrix multiplication beats edge loops.
        for t,rt in enumerate(rates):
            mat=A+np.einsum('rk,kij->rij',rt-1,parts,optimize=False)
            maty=Ay+np.einsum('rk,kij->rij',rt-1,parts_y,optimize=False)
            new=np.einsum('rij,rj->ri',mat,x,optimize=False)
            dn=np.einsum('rij,rj->ri',mat,dx,optimize=False)+np.einsum('rij,rj->ri',maty,x,optimize=False)
            total=new.sum(axis=1);der=dn.sum(axis=1)/total
            if np.any(total<=0):raise ModelError('matrix product lost every individual')
            x=new/total[:,None];dx=dn/total[:,None]-x*der[:,None]
            if t>=burn:logs+=np.log(total);slopes+=der
        logs/=len(rates)-burn;slopes/=len(rates)-burn
        if pairs:
            logs=(logs[:pairs]+logs[pairs:])/2;slopes=(slopes[:pairs]+slopes[pairs:])/2
        return logs,slopes

    def growth(self,y=0.,steps=20000,burn=1000,pairs=16,seed=1):
        h=self.environment.sample(steps+burn,pairs,seed,antithetic=True)
        r,s=self._products(self.rates(h),y,burn,pairs)
        return {'growth':float(r.mean()),'se':float(r.std(ddof=1)/np.sqrt(pairs)),
                'slope':float(s.mean()),'samples':r.tolist()}

    def loss(self,steps=20000,burn=1000,pairs=16,seed=1,tolerance=1e-7,max_iterations=12):
        # Calibrate first: this makes both coefficient and threshold bound valid.
        e=self if abs(self.R0-1)<1e-10 else self.critical()
        hosts=e.environment.sample(steps+burn,pairs,seed,antithetic=True)
        rates=e.rates(hosts)
        r0,s0=e._products(rates,0.,burn,pairs)
        y=float(np.clip(-r0.mean()/s0.mean(),-3,3))
        for iteration in range(max_iterations):
            r,s=e._products(rates,y,burn,pairs)
            correction=float(r.mean()/s.mean())
            if abs(correction)<tolerance:break
            y-=float(np.clip(correction,-2,2))
        else:raise ModelError('common-random-number birth root did not converge')
        samples=y-r/s.mean()
        return {'D':y,'raw_birth_root':y-math.log(self.R0),
                'reference_R0':e.R0,'se':float(samples.std(ddof=1)/np.sqrt(pairs)),
                'growth':float(r0.mean()),'growth_se':float(r0.std(ddof=1)/np.sqrt(pairs)),
                'root_residual':abs(correction),'iterations':iteration+1,
                'slope':float(s.mean()),'samples':samples.tolist(),
                'J':e.jensen()/e.b,'J_transfer':e.jensen(transfer_only=True)/e.b,
                'steps':steps,'burn':burn,'pairs':pairs,'seed':seed}

    def lineage_variance(self,steps=1000,environments=40,lines=200,seed=1):
        """Measure first-order path reward variance; never use C in this routine."""
        rng=np.random.default_rng(seed+197)
        z=self.environment.sample(2*steps,environments,seed,standardized=True)
        groups=[]
        for j in range(self.n):
            es=np.array([e for e,(src,*_) in enumerate(self.edges) if src==j])
            prob=np.array([self.edges[e][2]*self.w[self.edges[e][1]]/(self.growth_factor*self.w[j]) for e in es])
            groups.append((es,np.cumsum(prob)))
        width=max(len(es) for es,_ in groups)
        edge_ids=np.zeros((self.n,width),int);cum=np.ones((self.n,width))
        for j,(es,cs) in enumerate(groups):edge_ids[j,:len(es)]=es;cum[j,:len(es)]=cs
        dst=np.array([i for _,i,*_ in self.edges]);K=self.edge_rewards()
        states=rng.choice(self.n,(environments,lines),p=self.stationary/self.stationary.sum())
        if self.period>1:states[:]=0
        reward=np.zeros_like(states,dtype=float)
        for t in range(2*steps):
            pos=(rng.random(states.shape)[...,None]>cum[states]).sum(axis=-1)
            edges=edge_ids[states,np.minimum(pos,width-1)]
            reward+=np.einsum('eld,ed->el',K[edges],z[t],optimize=False)
            states=dst[edges]
            if t==steps-1:half=reward.var(axis=1,ddof=1)
        rate=(reward.var(axis=1,ddof=1)-half)/steps
        return {'V':float(rate.mean()),'se':float(rate.std(ddof=1)/np.sqrt(environments)),
                'V_generation':float(rate.mean()/self.b),'se_generation':float(rate.std(ddof=1)/np.sqrt(environments)/self.b),
                'steps':steps,'environments':environments,'lines':lines}

    def pgf_step(self,q,hosts=None,y=0.,offspring='poisson',dispersion=1.):
        """One-step offspring PGF, vectorized over environments.

        Each parent chooses stay/progress/route/death. Offspring then disperse
        independently. NB dispersion is per leaver. Shared group fate belongs
        to forward simulation; independent-parent PGFs do not apply to it.
        """
        return 1-self._pgf_survival(1-np.asarray(q),hosts,y,offspring,dispersion)

    def _pgf_survival(self,survival,hosts=None,y=0.,offspring='poisson',dispersion=1.):
        """Propagate 1-PGF directly, retaining very small survival chances."""
        if offspring not in ('poisson','negative_binomial'):
            raise ModelError('independent-parent PGF requires Poisson or negative binomial')
        if dispersion<=0:raise ModelError('negative-binomial dispersion must be positive')
        after=np.asarray(survival)@self.movement
        rates=None if hosts is None else self.rates(hosts)
        fail=np.zeros_like(survival)
        for src,dst,prob,changes,yield_,ykey,birth,det in self.events:
            pr=prob
            if rates is not None:
                for ki,a in changes.items():pr=pr+a*(rates[...,ki]-1)
            mu=yield_*np.exp(y*birth)
            if ykey is not None and rates is not None:mu=mu*rates[...,ykey]
            value=after[...,dst] if det else (-np.expm1(-mu*after[...,dst]) if offspring=='poisson' else -np.expm1(-dispersion*np.log1p(mu*after[...,dst]/dispersion)))
            fail[...,src]+=pr*value
        return np.clip(fail,0,1)

    def extinction(self,initial,steps,trials=256,seed=1,y=0.,offspring='poisson',dispersion=1.,constant=False):
        """Exact endpoint Bernoulli sampling via backward PGF composition.

        Returns the fraction of independent extinction indicators, with a
        binomial SE. Also returns the lower-variance conditional mean. Reverse
        arrays, not the transition kernel: works for nonreversible environments.

        Justification: conditional on one complete environmental history,
        descendants of distinct parents are independent. For a Poisson route
        with yield mu, its PGF is exp(mu*(q_move-1)); a deterministic
        continuation has PGF q_move. Weight these by the mutually exclusive
        stay/route/death probabilities. Conditioning on the first step proves
        the backward PGF composition by induction. For initial counts N_j,
        the extinction probability is product_j q_j**N_j. One Bernoulli draw
        with that probability therefore has the exact endpoint distribution.
        """
        initial=np.asarray(initial,dtype=np.int64)
        if initial.shape!=(self.n,) or np.any(initial<0):raise ModelError('invalid initial cohort')
        hosts=None if constant else self.environment.sample(steps,trials,seed)
        survival=np.ones((trials,self.n))
        for t in range(steps-1,-1,-1):
            survival=self._pgf_survival(survival,None if constant else hosts[t],y,offspring,dispersion)
        with np.errstate(divide='ignore',invalid='ignore'):
            lp=np.where(initial>0,initial*np.log1p(-survival),0).sum(axis=1)
        p=np.exp(lp);rng=np.random.default_rng(seed+413)
        extinct=rng.random(trials)<p;fraction=float(extinct.mean())
        return {'fraction':fraction,'se':math.sqrt(fraction*(1-fraction)/trials),
                'conditional_mean':float(p.mean()),'trials':trials,'steps':steps,
                'initial':int(initial.sum()),'method':'exact conditional PGF endpoint Bernoulli'}

    def offspring_fixed_point(self,y=0.,tolerance=1e-13,max_iterations=100000):
        q=np.zeros(self.n)
        for _ in range(max_iterations):
            nxt=self.pgf_step(q,y=y)
            if np.max(abs(nxt-q))<tolerance:return nxt
            q=nxt
        raise ModelError('offspring fixed point did not converge')

    def population(self,initial,steps,trials=100,seed=1,y=0.,offspring='poisson',dispersion=1.,group_size=10):
        """Forward whole-number trajectories, preserving the same mean matrix.

        Shared fate: independently group leavers within each chosen route;
        a group of size g produces g times a Poisson(yield) draw. The last
        smaller group uses its actual size. This preserves the first moment.
        Counts beyond safe int64 sampling are refused, never silently capped.
        """
        rng=np.random.default_rng(seed+97);x=np.tile(np.asarray(initial,np.int64),(trials,1))
        hosts=self.environment.sample(steps,trials,seed);rates=self.rates(hosts)
        bysource=[[event for event in self.events if event[0]==j] for j in range(self.n)]
        for t in range(steps):
            pre=np.zeros_like(x)
            for j,events in enumerate(bysource):
                remain=x[:,j].copy();mass=np.ones(trials)
                for src,dst,prob,changes,mean,ykey,birth,det in events:
                    pr=np.full(trials,prob)
                    for ki,a in changes.items():pr+=a*(rates[t,:,ki]-1)
                    chosen=rng.binomial(remain,np.clip(pr/np.maximum(mass,1e-300),0,1));remain-=chosen;mass-=pr
                    mu=mean*np.exp(y*birth)*(rates[t,:,ykey] if ykey is not None else np.ones(trials))
                    if np.any(chosen*mu>1e15):raise OverflowError('population too large for exact int64 sampling; use endpoint extinction')
                    if det:out=chosen
                    elif offspring=='poisson':out=rng.poisson(chosen*mu)
                    elif offspring=='negative_binomial':
                        out=np.zeros(trials,np.int64);ok=chosen>0
                        out[ok]=rng.negative_binomial(chosen[ok]*dispersion,dispersion/(dispersion+mu[ok]))
                    elif offspring=='shared_groups':
                        groups=chosen//group_size;rest=chosen%group_size
                        out=group_size*rng.poisson(groups*mu)+rest*rng.poisson(mu)
                    else:raise ModelError('unknown offspring law')
                    pre[:,dst]+=out
            x=np.zeros_like(pre)
            for j in range(self.n):
                remaining=pre[:,j].copy();mass=1.
                for i in np.flatnonzero(self.movement[:,j]):
                    p=self.movement[i,j]
                    moved=rng.binomial(remaining,min(1.,p/max(mass,1e-300)))
                    x[:,i]+=moved;remaining-=moved;mass-=p
        frac=float(np.mean(x.sum(axis=1)==0))
        return {'counts':x,'fraction':frac,'se':math.sqrt(frac*(1-frac)/trials)}
