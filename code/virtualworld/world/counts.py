"""Integer counts and independent-sum sampling without a population ceiling.

Binomial trials and Poisson rates are split at 2**50, below both binary64's
exact-integer limit and numpy's int64 sampling limit. Independent binomials
with common p and independent Poissons add to the original laws. Counts and
sums use Python integers when needed. No Gaussian/deterministic population
approximation is made; ordinary floating-point probabilities and numpy's
random variate algorithms remain. Work scales with count/rate divided by the
chunk size; memory per draw batch is bounded. Budget limits belong to callers.
"""
import numbers
import numpy as np

CHUNK = 2**50
BATCH = 4096
INT64_MAX = np.iinfo(np.int64).max


def compact(values):
    """Use int64 only when every value fits (never sum in int64 first)."""
    a = np.asarray(values)
    return a.astype(np.int64) if not a.size or a.max() <= INT64_MAX else a.astype(object)


def integers(values):
    a = np.asarray(values)
    if a.dtype == object:
        valid = all(isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))
                    for x in a.flat)
    else:
        valid = np.issubdtype(a.dtype, np.integer)
    if not valid or np.any(a < 0):
        raise ValueError('counts must be nonnegative integers')
    # Convert numpy integer objects to Python integers before any arithmetic.
    if a.dtype == object:
        a = np.array([int(x) for x in a.flat], dtype=object).reshape(a.shape)
    return compact(a)


def _sum_chunks(draw, count):
    total = 0
    while count:
        size = min(count, BATCH)
        total += sum(map(int, draw(size)))
        count -= size
    return total


def binomial(counts, probability, rng, chunk=CHUNK):
    n = np.asarray(counts)
    p = np.broadcast_to(np.asarray(probability, float), n.shape)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid binomial probability')
    if np.all(n <= chunk):
        return rng.binomial(n.astype(np.int64), p)
    out = np.empty(n.shape, object)
    for i in np.ndindex(n.shape):
        count, prob = int(n[i]), float(p[i])
        if prob == 0:
            out[i] = 0
        elif prob == 1:
            out[i] = count
        else:
            q, rem = divmod(count, chunk)
            out[i] = int(rng.binomial(rem, prob)) + _sum_chunks(
                lambda size: rng.binomial(chunk, prob, size=size), q)
    return compact(out)


def poisson_product(counts, per_individual, rng, chunk=CHUNK):
    """Poisson(n*mu), without first rounding n or overflowing a float product.

For large products, split the exact rational representation of binary64 mu.
Only the final remainder rate is rounded to binary64 (relative error <=2**-52
for normal numbers). Full chunk rates are exactly represented. This avoids
rounding an arbitrary-sized count through a float.
"""
    n = np.asarray(counts)
    mu = np.broadcast_to(np.asarray(per_individual, float), n.shape)
    if np.any(~np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError('offspring rate must be finite and nonnegative')
    if n.dtype != object and np.all(n <= chunk):
        with np.errstate(over='ignore'):
            lam = n * mu
        if np.all(lam <= chunk):
            return rng.poisson(lam)
    out = np.empty(n.shape, object)
    for i in np.ndindex(n.shape):
        num, den = float(mu[i]).as_integer_ratio()
        q, rem = divmod(int(n[i]) * num, den * chunk)
        out[i] = int(rng.poisson(rem / den)) + _sum_chunks(
            lambda size: rng.poisson(chunk, size=size), q)
    return compact(out)


def archive(values):
    """NPZ-safe exact count representation; decimal strings above int64."""
    a = compact(values)
    return a.astype(str) if a.dtype == object else a
