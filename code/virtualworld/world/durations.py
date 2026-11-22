"""Transient residence kernels. Columns are current states, rows next states.

Death is applied separately by Model. Gamma durations are ceil(Gamma), with
the continuous scale solved so the discrete mean matches the requested mean.
Only the tail below 1e-12 is collapsed into the last age. Empirical durations
are a user-supplied probability mass on positive integer steps, not fitted.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats, optimize


@dataclass
class Duration:
    U: np.ndarray
    exit: np.ndarray
    label: str
    tail: float = 0.

    def __post_init__(self):
        self.U = np.asarray(self.U, float)
        self.exit = np.asarray(self.exit, float)
        n = len(self.exit)
        if self.U.shape != (n,n) or n == 0 or np.any(self.U < 0) or np.any(self.exit < 0):
            raise ValueError('invalid duration kernel')
        if not np.allclose(self.U.sum(axis=0)+self.exit, 1., atol=1e-12, rtol=0):
            raise ValueError('duration probabilities must sum to one')
        if max(abs(np.linalg.eigvals(self.U))) >= 1:
            raise ValueError('residence must end almost surely')

    @property
    def mean(self):
        entry = np.eye(len(self.exit))[:,0]
        return float(np.linalg.solve(np.eye(len(entry))-self.U, entry).sum())

    @classmethod
    def geometric(cls, mean):
        if not np.isfinite(mean) or mean < 1:
            raise ValueError('geometric mean must be at least one')
        return cls(np.array([[1-1/mean]]), np.array([1/mean]), 'geometric')

    @classmethod
    def chain(cls, mean, phases):
        if int(phases) != phases or phases < 1 or mean < phases:
            raise ValueError('chain requires integer phases <= mean')
        q = phases/mean
        U = np.eye(phases)*(1-q)
        U[np.arange(1,phases),np.arange(phases-1)] = q
        ex = np.zeros(phases); ex[-1] = q
        return cls(U, ex, f'chain({phases})')

    @classmethod
    def empirical(cls, probabilities):
        p = np.asarray(probabilities, float)
        if p.ndim != 1 or len(p) == 0 or np.any(~np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12):
            raise ValueError('supply normalized probability at each duration 1..n')
        p = p[:np.flatnonzero(p)[-1]+1]/p.sum()
        surv = np.cumsum(p[::-1])[::-1]
        hazard = p/surv
        U = np.zeros((len(p),len(p)))
        U[np.arange(1,len(p)),np.arange(len(p)-1)] = 1-hazard[:-1]
        return cls(U, hazard, 'empirical')

    @classmethod
    def gamma(cls, mean, shape, tail=1e-12):
        if not np.isfinite(mean) or not np.isfinite(shape) or mean < 1 or shape <= 0 or not 0 < tail <= 1e-10:
            raise ValueError('rounded gamma requires mean>=1, shape>0, small tail')
        # A positive integer duration with mean one is identically one.
        # This is the zero-scale limit, not a positive-scale gamma fit.
        if mean == 1:
            return cls(np.zeros((1,1)), np.ones(1), f'rounded gamma({shape}; degenerate)')
        # ceil(X) has mean sum_{j>=0} Pr(X>j). This bracket's upper mean
        # is at least the requested mean; an arbitrarily small scale gives 1.
        n = int(np.ceil(stats.gamma.ppf(1-tail, shape, scale=mean/shape)))+1
        if n > 2000:
            raise ValueError('age kernel too large; refine time units explicitly')
        ages = np.arange(n)
        scale = optimize.brentq(lambda a: stats.gamma.sf(ages, shape, scale=a).sum()-mean,
                                mean/shape*1e-8, mean/shape, xtol=1e-13)
        surv = stats.gamma.sf(ages, shape, scale=scale)
        p = np.r_[surv[:-1]-surv[1:],surv[-1]]
        obj = cls.empirical(p)
        obj.label = f'rounded gamma({shape})'
        obj.tail = float(stats.gamma.sf(n-1,shape,scale=scale))
        return obj
