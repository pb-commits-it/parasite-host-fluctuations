"""Corrected check. Life cycle with n stages; stage k has w_k host types; block A_k
carries stage k -> stage k+1 (mod n). NGM K is block-cyclic. P = A_{n-1}...A_0 is the
'return map' to stage 0.  Claims tested:
 (1) rho(K)^n == rho(P)                        (always)
 (2) sum of all closed n-step routes == trace(P)
 (3) rho(P) == trace(P) when some block is rank one (bottleneck / separable mixing)
 (4) without that, trace(P) can be above OR below rho(P)
 (5) bounds: max_i P_ii <= rho(P)  and  trace(P)/w_min <= rho(P)
"""
import numpy as np, itertools
rng = np.random.default_rng(11)
rho = lambda M: max(abs(np.linalg.eigvals(M)))

def build(blocks):
    widths = [A.shape[1] for A in blocks]; n = len(blocks)
    off = np.concatenate([[0], np.cumsum(widths)]); N = off[-1]
    K = np.zeros((N, N))
    for k, A in enumerate(blocks):
        kn = (k + 1) % n
        K[off[kn]:off[kn+1], off[k]:off[k+1]] = A
    P = np.eye(widths[0])
    for A in blocks: P = A @ P
    return K, P, widths

def brute_routes(blocks):
    """explicit enumeration of every closed route that visits each stage once"""
    n = len(blocks); widths = [A.shape[1] for A in blocks]; tot = 0.0
    for path in itertools.product(*[range(w) for w in widths]):
        prod = 1.0
        for k in range(n): prod *= blocks[k][path[(k+1) % n], path[k]]
        tot += prod
    return tot

def random_cycle(rank_one_stage=None):
    n = int(rng.integers(2, 6)); widths = [int(rng.integers(2, 5)) for _ in range(n)]
    blocks = []
    for k in range(n):
        r, c = widths[(k+1) % n], widths[k]
        if rank_one_stage is not None and k == rank_one_stage % n:
            A = np.outer(rng.uniform(0.2, 3, r), rng.uniform(0.2, 3, c))   # separable mixing
        else:
            A = rng.uniform(0.0, 3, (r, c)) * (rng.random((r, c)) < 0.8)   # some zero links
        blocks.append(A)
    return blocks

T = 20000
c1 = c2 = c5a = c5b = 0; above = below = 0; ratios = []
for _ in range(T):
    b = random_cycle(); K, P, w = build(b); n = len(b)
    rK, rP, tr = rho(K)**n, rho(P), np.trace(P)
    if rP < 1e-12: continue
    c1 += abs(rK - rP) > 1e-7 * rP
    if _ < 300: c2 += abs(brute_routes(b) - tr) > 1e-9 * max(tr, 1)
    c5a += P.diagonal().max() > rP * (1 + 1e-9)
    c5b += tr / min(w) > rP * (1 + 1e-9)
    above += tr > rP * (1 + 1e-9); below += tr < rP * (1 - 1e-9); ratios.append(tr / rP)
print(f"(1) rho(K)^n != rho(P):                 {c1} failures / {T}")
print(f"(2) brute-force route sum != trace(P):  {c2} failures / 300")
print(f"(5) max P_ii > rho(P):                  {c5a} failures;  trace/w_min > rho(P): {c5b} failures")
print(f"(4) no bottleneck: sum-of-routes ABOVE true R0^n in {above/T:.1%}, BELOW in {below/T:.1%} of cycles")
r = np.array(ratios); print(f"    ratio sum/true: 5th pct {np.percentile(r,5):.3f}  median {np.median(r):.3f}  95th pct {np.percentile(r,95):.3f}  min {r.min():.3f}  max {r.max():.3f}")
c3 = 0
for _ in range(T):
    b = random_cycle(rank_one_stage=int(rng.integers(0, 5))); K, P, w = build(b); n = len(b)
    c3 += abs(rho(K)**n - np.trace(P)) > 1e-7 * max(np.trace(P), 1e-12)
print(f"(3) one rank-one (well-mixed) step: R0^n != sum of routes in {c3} / {T}")
# structured counterexamples
spec = [np.diag([2.0, 2.0]), np.diag([3.0, 3.0])]            # two host-specific cycles
K, P, _ = build(spec); print(f"host-specific cycles: true R0^2 = {rho(K)**2:.3f}, sum of routes = {np.trace(P):.3f}")
alt = [np.array([[0, 2.0], [2.0, 0]]), np.eye(2) * 3.0]      # routes only close after two laps
K, P, _ = build(alt); print(f"alternating cycle:    true R0^2 = {rho(K)**2:.3f}, sum of routes = {np.trace(P):.3f}")
