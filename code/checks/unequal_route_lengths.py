"""Check of Proposition 5 (notes/linear-theory.md).

Claim. If one compartment b lies on every cycle of the transmission graph, then
R0 = rho(K) is the unique positive root of  sum_r w_r * lam**(-len_r) = 1  over the simple
cycles r through b, and R0 > 1 exactly when sum_r w_r > 1.

Test. Random graphs built so that node 0 is on every cycle: the other nodes form a random
acyclic graph (edges only from lower to higher index), node 0 may send to any node and
receive from any node, and may have a self-loop. Routes are enumerated explicitly.
"""
import numpy as np
from scipy.optimize import brentq

rng = np.random.default_rng(5)


def random_case():
    N = int(rng.integers(2, 8))
    K = np.zeros((N, N))                       # K[i, j]: from j to i
    for j in range(1, N):
        for i in range(j + 1, N):
            if rng.random() < 0.5:
                K[i, j] = rng.uniform(0.1, 2.0)
    for i in range(1, N):
        if rng.random() < 0.7: K[i, 0] = rng.uniform(0.1, 2.0)   # out of the shared compartment
        if rng.random() < 0.7: K[0, i] = rng.uniform(0.1, 2.0)   # back into it
    if rng.random() < 0.3: K[0, 0] = rng.uniform(0.1, 2.0)
    return K


def routes_through_zero(K):
    N = len(K); out = []
    if K[0, 0] > 0: out.append((K[0, 0], 1))
    def walk(node, weight, length):
        if K[0, node] > 0: out.append((weight * K[0, node], length + 1))
        for nxt in range(node + 1, N):
            if K[nxt, node] > 0: walk(nxt, weight * K[nxt, node], length + 1)
    for first in range(1, N):
        if K[first, 0] > 0: walk(first, K[first, 0], 1)
    return out


bad_root = bad_threshold = tested = mixed = 0
for _ in range(20000):
    K = random_case(); r = routes_through_zero(K)
    if not r: continue
    tested += 1
    mixed += len({l for _, l in r}) > 1
    f = lambda lam: sum(w * lam ** (-l) for w, l in r) - 1
    root = brentq(f, 1e-9, 1e6)
    rho = max(abs(np.linalg.eigvals(K)))
    bad_root += abs(root - rho) > 1e-7 * max(rho, 1)
    total = sum(w for w, _ in r)
    if abs(total - 1) > 1e-6: bad_threshold += (rho > 1) != (total > 1)
print(f"cases with at least one route: {tested} (of which {mixed} mix route lengths)")
print(f"root of the route equation != rho(K): {bad_root}")
print(f"threshold disagreement (rho > 1 vs route sum > 1): {bad_threshold}")
