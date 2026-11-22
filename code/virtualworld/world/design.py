"""Shared allocation, normal quantile and phase-2 design constants."""
from itertools import product
import numpy as np
from scipy.stats import norm

Z95 = float(norm.ppf(.975))
POWER80_FACTOR = (Z95 + float(norm.ppf(.8)))**2
GENERATIONS = (23, 50, 100)
LOG_R0_AXIS = (.02, .05, .1, .2, .4)
RICKER_AXES = ((.2, .8, 1.5), (1e3, 1e4), (0., .3, .6), (0., .01), (5, 20))
RICKER_SETTINGS = tuple(product(*RICKER_AXES))


def allocate(total, proportions):
    """Largest-remainder allocation, with stable state-index tie breaking."""
    p = np.asarray(proportions, float)
    if (not isinstance(total, (int, np.integer)) or total < 0 or
            p.ndim != 1 or not p.size or np.any(~np.isfinite(p)) or
            np.any(p < 0) or p.sum() <= 0):
        raise ValueError('nonnegative integer total and finite proportions required')
    quota = total * (p / p.sum())
    out = np.floor(quota).astype(np.int64)
    remainder = total - int(out.sum())
    order = np.argsort(-(quota - out), kind='stable')
    out[order[:remainder]] += 1
    return out


def generation_steps(stay, generations):
    """Reference exposed stay 2 plus steady stay; require exact step units."""
    steps = (2 + stay) * generations
    if steps != int(steps) or steps < 1:
        raise ValueError('matched horizon must be a positive integer step')
    return int(steps)


def matched_run_steps(stay):
    """Cover every registered primary and secondary observation horizon."""
    return max(500, generation_steps(stay, max(GENERATIONS)))
