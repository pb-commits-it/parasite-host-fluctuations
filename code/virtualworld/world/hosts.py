"""Exogenous and endogenous host drivers; no fitting to persistence data.

Forced lognormal phi is log correlation. Bounded/gamma refresh phi is raw
host correlation. Arrays are (time, replicate, patch*host). Real records must
be equally spaced in declared model steps; gaps are rejected, never filled.
"""
import numpy as np
from scipy import signal


def lognormal(steps, replicates, sigma, phi, seed, drivers=1, patches=1,
              synchrony=0., antithetic=False, mean=1.):
    """Stationary Gaussian AR(1), then arithmetic-mean-centered exponential.

Synchrony is a shared Gaussian innovation fraction across patches, separately
for each host type. Antithetic outputs put the positive half before its negative
half; pair means are the independent statistical units. No path normalization.
"""
    if not -1 < phi < 1 or sigma < 0 or mean <= 0 or not 0 <= synchrony <= 1:
        raise ValueError('invalid lognormal driver')
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(steps,replicates,patches,drivers))
    if synchrony:
        shared = rng.normal(size=(steps,replicates,1,drivers))
        z = np.sqrt(1-synchrony)*z+np.sqrt(synchrony)*shared
    z[1:] *= np.sqrt(1-phi*phi)
    z = signal.lfilter([1.], [1.,-phi], z, axis=0)
    z = z.reshape(steps,replicates,patches*drivers)
    if antithetic:
        z = np.concatenate((z,-z),axis=1)
    return mean*np.exp(sigma*z-sigma*sigma/2)


def bounded(steps, replicates, sigma, phi, seed, drivers=1, patches=1, low=.05):
    """R16's mean-one two-state refresh chain, raw variance exp(sigma²)-1."""
    if sigma < 0 or not 0 <= phi < 1 or not 0 < low < 1:
        raise ValueError('invalid bounded driver')
    rng = np.random.default_rng(seed)
    v = np.expm1(sigma*sigma)
    high = 1+v/(1-low)
    prob = (1-low)**2/((1-low)**2+v)
    shape = (steps,replicates,drivers*patches)
    draw = rng.random(shape) < prob
    fresh = rng.random(shape) < 1-phi
    fresh[0] = True
    last = np.maximum.accumulate(np.where(fresh,np.arange(steps)[:,None,None],0),axis=0)
    states = np.take_along_axis(draw,last,axis=0)
    return np.where(states,high,low)


def gamma(steps, replicates, shape, seed):
    """Independent gamma hosts of mean one and variance 1/shape."""
    if shape <= 0:
        raise ValueError('gamma shape must be positive')
    return np.random.default_rng(seed).gamma(shape,1/shape,(steps,replicates,1))


def recorded(times, log_abundance, baseline=None):
    """Drive one finite record without wrapping. Steps must be equally spaced.

Supply times already converted to one model-step units. Explicit baseline is
arithmetic abundance; if omitted the record's own mean is used and returned.
This finite-record normalization is distinct from stationary forcing.
"""
    t, z = np.asarray(times,float), np.asarray(log_abundance,float)
    if t.ndim != 1 or len(t)<2 or z.shape != t.shape or not np.all(np.isfinite(z)) or not np.all(np.isfinite(t)):
        raise ValueError('finite paired time and log-abundance vectors required')
    if not np.allclose(np.diff(t),1.,atol=1e-10,rtol=0):
        raise ValueError('convert units explicitly; missing/irregular steps refused')
    a = np.exp(z)
    base = float(a.mean()) if baseline is None else float(baseline)
    if not np.isfinite(base) or base <= 0 or not np.all(np.isfinite(a)):
        raise ValueError('invalid arithmetic abundance baseline')
    return a[:,None,None]/base, base


class Ricker:
    """Positive host abundance with environmental shocks and burden feedback.

H_next=H*exp(r*(1-H/K)+sigma*epsilon-sigma²/2-alpha*burden/H).
H is continuous abundance, not integer hosts. Mortality due to parasites acts
on the host update; parasites are not grouped into hosts and no co-death is
implied. Underflow/nonfinite abundance is refused. Co-death needs another law.
"""
    def __init__(self, initial, carrying=1000., r=.8, sigma=.3, alpha=.01, seed=1):
        self.H = np.asarray(initial,float).copy()
        if np.any(self.H<=0) or carrying<=0 or r<0 or sigma<0 or alpha<0:
            raise ValueError('invalid Ricker parameters')
        self.K, self.r, self.sigma, self.alpha = carrying,r,sigma,alpha
        self.rng = np.random.default_rng(seed)

    def step(self, burden):
        b = np.asarray(burden,float)
        if b.shape != self.H.shape or np.any(b<0):
            raise ValueError('burden must match host populations')
        e = self.sigma*self.rng.normal(size=self.H.shape)-self.sigma**2/2
        self.H *= np.exp(self.r*(1-self.H/self.K)+e-self.alpha*b/self.H)
        if np.any(self.H<=0) or np.any(~np.isfinite(self.H)):
            raise FloatingPointError('host abundance became nonpositive/nonfinite')
        return self.H/self.K
