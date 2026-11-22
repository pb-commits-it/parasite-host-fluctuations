"""Run R10b (protocols/R10b-real-systems-stay-shape.md): the loss D of ln R0 for the rows of the manuscript's
table of real parasites (run R10, section H4) when the long steady stage is more regular than geometric.

Checks, with the criteria fixed in the protocol:
  R1  the 11 parameter sets of R10's H4 are reproduced within 4 combined standard errors (else the run stops);
  R2  a second simulator, written for this run, agrees with the ring simulator on eight sets (3.5 combined se);
  R3  the closed form with R = E[phi^L] equals the matrix formula for every shape (1e-9);
  R4  D <= sigma^2/2 everywhere, and a ring with every stay fixed gives sigma^2/2;
  R5  the growth rate at the root is the same in the two halves of every run (3 se), else the case is flagged.
Reported, with the rules for the verdicts fixed in the protocol: D for each row and shape of the long stage,
as a fraction of sigma^2/2 and of the all-geometric value; the coefficient of variation of the long stage at
which half of the protection is lost; and which stage's shape matters more.

Inputs are those of checks/r10_loss_at_large_noise.py (H4).  R10's own values are read from its results file.
Imports checks/r15a_lib.py (run R15a) for the ring simulator, the root finder and the matrix formula.

Run: python3 -I checks/r10b_real_systems_stay_shape.py     (three processes)
     add  --smoke N  to divide the lengths of the simulations by N (timing test only).
"""
from __future__ import annotations

import math
import os
import re
import sys
import time

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import multiprocessing as mp

import numpy as np

import r15a_lib as L

SMOKE = 1
if "--smoke" in sys.argv:
    SMOKE = int(sys.argv[sys.argv.index("--smoke") + 1])

R10_RESULTS = os.path.join(HERE, "..", "results", "r10-loss-at-large-noise_20261114.txt")

# The H4 parameter sets of checks/r10_loss_at_large_noise.py: label, stays (fluctuating stage first), hosts as
# (stage, sigma, phi per step).
A0_SETS = [
    ("1a S. haematobium", [1, 24], [(0, 0.589, 0.398)]),
    ("1b S. mansoni", [1, 24], [(0, 1.380, 0.367)]),
    ("2a E. multilocularis, transfer out of the egg", [1.25, 2, 1], [(0, 1.174, 0.75)]),
    ("2a E. multilocularis, transfer out of the fox", [1, 1.25, 2], [(0, 1.174, 0.75)]),
    ("2a E. multilocularis, transfer out of the vole", [2, 1, 1.25], [(0, 1.174, 0.75)]),
    ("2b E. multilocularis, fox and vole", [1.25, 2, 1], [(0, 1.174, 0.75), (2, 0.18, 0.50)]),
    ("3a B. burgdorferi, oak forest", [1, 10 / 4.4], [(0, 0.668, 0.30)]),
    ("3b B. burgdorferi, northern edge", [1, 10 / 4.4], [(0, 1.190, 0.50)]),
    ("5 T. tenuis in red grouse", [1, 12], [(0, 0.70, 0.85)]),
    ("6 P. falciparum", [1, 7.5], [(0, 1.40, 0.75)]),
    ("7 T. circumcincta in Soay sheep", [1, 3], [(0, 0.35, 0.80)]),
]

# The rows of the manuscript's table: fluctuating stage F and long steady stage, in R10's steps; r = refinement
# of the time step in Part B; `other` = further stages (row 2a: the fox, one step).
ROWS = {
    "1a": dict(label="S. haematobium (worm in people, 48 months)", F=1.0, Lg=24.0, other=0, sigma=0.589, phi=0.398, r=8, dt=2.0,
               mA=(1, 2, 3, 4, 6, 8, 12, 16, 20, 22, 23, 24), mB=(1, 2, 4, 16, 32, 64), gB=(128, 64, 32, 16, 8, 4)),
    "1b": dict(label="S. mansoni (worm in people, 48 months)", F=1.0, Lg=24.0, other=0, sigma=1.380, phi=0.367, r=8, dt=2.0,
               mA=(1, 2, 3, 4, 6, 8, 12, 16, 20, 22, 23, 24), mB=(1, 2, 4, 16, 32, 64), gB=(128, 64, 32, 16, 8, 4)),
    "6": dict(label="P. falciparum (infection in people, 6 months)", F=1.0, Lg=7.5, other=0, sigma=1.40, phi=0.75, r=8, dt=0.8,
              mA=(1, 2, 4, 7), mB=(1, 2, 4, 16, 30, 45), gB=(32, 16, 8, 4)),
    "3a": dict(label="B. burgdorferi, oak forest (tick, 10 months)", F=1.0, Lg=10 / 4.4, other=0, sigma=0.668, phi=0.30, r=11, dt=4.4,
               mA=(1, 2), mB=(1, 2, 4, 16, 20, 23), gB=(16, 8, 4, 2)),
    "3b": dict(label="B. burgdorferi, northern edge (tick, 10 months)", F=1.0, Lg=10 / 4.4, other=0, sigma=1.190, phi=0.50, r=11, dt=4.4,
               mA=(1, 2), mB=(1, 2, 4, 16, 20, 23), gB=(16, 8, 4, 2)),
    "2a": dict(label="E. multilocularis (larva in vole, 4 months; egg 2.5; fox 2)", F=1.25, Lg=2.0, other=1, sigma=1.174, phi=0.75, r=8, dt=2.0,
               mA=(1, 2), mB=(1, 2, 4, 8, 16), gB=()),
}
A0_OF_ROW = {"1a": "1a S. haematobium", "1b": "1b S. mansoni", "6": "6 P. falciparum", "3a": "3a B. burgdorferi, oak forest",
             "3b": "3b B. burgdorferi, northern edge", "2a": "2a E. multilocularis, transfer out of the egg"}


TCAP = max(320000 // SMOKE, 3000)        # longest run allowed when a case is lengthened (amendment 2)


def size(kind, r=1):
    M, T = {"A0": (400, 50000), "A": (400, 25000), "B": (200, 16000), "Bheavy": (100, 10000)}[kind]
    T = max((T * r) // SMOKE, 3000)
    return M, T + (T % 2)


# ----------------------------------------------------------------------------- building a case
def stage_substages(steps, shape):
    """A stage of mean `steps` as a list of sub-stage staying rates.  shape: "fixed", or the number m of equal
    sub-stages (m = 1 is geometric).  A sub-stage of exactly one step has staying rate 0."""
    if shape == "fixed":
        assert abs(steps - round(steps)) < 1e-12
        return [0.0] * int(round(steps))
    m = int(shape)
    per = steps / m
    assert per >= 1 - 1e-12, "a sub-stage cannot last less than one step"
    return [max(0.0, 1.0 - 1.0 / per)] * m


def stage_cv(steps, shape):
    if shape == "fixed":
        return 0.0
    m = int(shape)
    return math.sqrt(max(1.0 / m - 1.0 / steps, 0.0))


def build(row, part, F_shape, L_shape):
    """All sub-stages of a row.  part "A": R10's step; "B": r times finer.  Returns the staying rates of every
    sub-stage, the host memory per step of that part, and the coefficient of variation of the long stage."""
    R = ROWS[row]
    r = 1 if part == "A" else R["r"]
    phi = R["phi"] ** (1.0 / r)
    subs = stage_substages(R["F"] * r, F_shape) + stage_substages(R["Lg"] * r, L_shape) + [0.0] * (R["other"] * r)
    return subs, phi, stage_cv(R["Lg"] * r, L_shape)


def split(subs):
    """Sub-stages with staying (stepped explicitly) and the number of one-step sub-stages (a delay line)."""
    sA = [s for s in subs if s > 0]
    return sA, len(subs) - len(sA)


def second_order(subs, sigma, phi):
    """sigma^2 [(1 + 2 theta)/(2 T) - R/(1 - R)], R = E[phi^L] = product of (1 - s) phi/(1 - s phi)."""
    s = np.asarray(subs, float)
    TT = float(np.sum(1 / (1 - s)))
    th = phi / (1 - phi)
    R = float(np.prod((1 - s) * phi / (1 - s * phi)))
    return sigma ** 2 * ((1 + 2 * th) / (2 * TT) - R / (1 - R)), TT


def ring_matrix(subs):
    s = np.asarray(subs, float)
    n = s.size
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] += s[k]
        A[(k + 1) % n, k] += 1 - s[k]
    return A


def matrix_second_order(subs, sigma, phi):
    A = ring_matrix(subs)
    n = A.shape[0]
    dA = np.zeros((n, n))
    dA[1 % n, 0] = 1 - subs[0]
    return sigma ** 2 * float(np.sum(1 / (1 - np.asarray(subs)))) * L.cpp_resolvent(A, dA, phi)


def relax_time(subs):
    """-1/ln|lambda_2| of the mean matrix at its threshold, in steps (infinite for a ring with no staying)."""
    ev = np.sort(np.abs(np.linalg.eigvals(ring_matrix(subs))))[::-1]
    if len(ev) < 2:
        return 0.0
    if ev[1] >= 1 - 1e-12:
        return math.inf
    return -1.0 / math.log(ev[1])


# ----------------------------------------------------------------------------- simulators written for this run
def chain_sim(sA, b_fix, sigma, phi, ys, M, T, burn, seed, chunk=400):
    """Sub-stages with staying rates sA at the threshold (transfer 1 - s), then b_fix sub-stages of one step (a
    delay line) back to the first.  The transfer out of the last staying sub-stage carries exp(y) h(t),
    h = exp(sigma z - sigma^2/2), z an AR(1) series.  Returns the growth rate per step in the first and in the
    second half of the T steps after the burn-in, each (len(ys), M)."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ny = ys.size
    ey = np.exp(ys)[:, None]
    s = np.asarray(sA, float)[:, None, None]
    g = 1 - s
    mA = s.shape[0]
    G = float(g[-1, 0, 0])
    zlast = rng.standard_normal(M)
    X = np.ones((mA, ny, M))
    buf = np.ones((b_fix + 1, ny, M))
    logacc = np.zeros((ny, M))
    half = T // 2
    marks = {burn: None, burn + half: None}
    done, total = 0, burn + 2 * half
    while done < total:
        C = min(chunk, total - done)
        z, zlast = L.ar1_block(rng, zlast, phi, C, M)
        H = np.exp(sigma * z - sigma * sigma / 2) * G
        for t in range(C):
            tt = done + t
            if tt in marks:
                marks[tt] = logacc + np.log(X[-1])
            buf[tt % (b_fix + 1)] = ey * H[t] * X[-1]
            back = buf[(tt + 1) % (b_fix + 1)]
            Xn = X * s
            if mA > 1:
                Xn[1:] += X[:-1] * g[:-1]
            Xn[0] += back
            X = Xn
            if tt % 16 == 15:
                f = X.sum(axis=0)
                X /= f
                buf /= f
                logacc += np.log(f)
        done += C
    L2 = logacc + np.log(X[-1])
    L0, L1 = marks[burn], marks[burn + half]
    return (L1 - L0) / half, (L2 - L1) / half


def ring_multi(s, hosts, ys, M, T, burn, seed, chunk=400):
    """Ring at its threshold with several independent lognormal hosts (row 2b).  hosts: (stage, sigma, phi);
    exp(y) multiplies the transfer out of the first host's stage."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    ny = ys.size
    ey = np.exp(ys)[:, None]
    s = np.asarray(s, float)
    n = s.size
    s0 = s[:, None, None]
    g0 = (1 - s)[:, None, None]
    zl = [rng.standard_normal(M) for _ in hosts]
    X = np.full((n, ny, M), 1.0 / n)
    acc = np.zeros((ny, M))
    done, total = 0, burn + T
    ks = hosts[0][0]
    while done < total:
        C = min(chunk, total - done)
        Hs = []
        for i, (k, sig, ph) in enumerate(hosts):
            z, zl[i] = L.ar1_block(rng, zl[i], ph, C, M)
            Hs.append((k, (1 - s[k]) * np.exp(sig * z - sig * sig / 2)))
        tots = np.empty((C, ny, M))
        for t in range(C):
            SX = X * s0
            GX = X * g0
            for k, H in Hs:
                np.multiply(X[k], H[t], out=GX[k])
            GX[ks] *= ey
            SX[1:] += GX[:-1]
            SX[0] += GX[-1]
            tot = SX.sum(axis=0)
            tots[t] = tot
            X = SX / tot
        lo = max(burn - done, 0)
        if lo < C:
            acc += np.log(tots[lo:]).sum(axis=0)
        done += C
    return acc / T


def root(run, cap, M, T, burn, seed):
    """A wide, short pilot between 0.1 and 0.9 of the bound sigma^2/2, then the root finder of run R15a."""
    Tp = max(T // 16, 1500)
    ys = cap * np.array([0.1, 0.5, 0.9])
    r = run(ys, Tp, burn, seed + 4242)
    y1, _ = L.quad_root(ys, r.mean(axis=1))
    y1 = float(np.clip(y1, 0.005 * cap, 1.3 * cap))
    return L.find_root(run, y1, max(0.02, 0.15 * y1), M, T, burn, seed)


# ----------------------------------------------------------------------------- tasks
def task_ring(a):
    """A0: one of R10's parameter sets with the ring simulator."""
    t0 = time.time()
    s = 1 - 1 / np.asarray(a["stays"], float)
    hosts = a["hosts"]
    M, T = a["size"]
    cap = sum(h[1] ** 2 for h in hosts) / 2
    if len(hosts) == 1:
        k, sig, ph = hosts[0]
        g = 1 - s

        def vary(z):
            return [(1, k, g[k] * np.exp(sig * z - sig * sig / 2))]

        run = lambda ys, T_, b_, sd: L.sim_ring_A(s.size, s, g, k, vary, ph, ys, M, T_, b_, sd)
    else:
        run = lambda ys, T_, b_, sd: ring_multi(s, hosts, ys, M, T_, b_, sd)
    D, se, Dm, runs = root(run, cap, M, T, 2000, a["seed"])
    return a["key"], dict(D=D, se=se, runs=runs, secs=time.time() - t0)


def task_chain(a):
    """One shape with the simulator written for this run."""
    t0 = time.time()
    subs, sigma, phi = a["subs"], a["sigma"], a["phi"]
    M, T = a["size"]
    cap = sigma ** 2 / 2
    tau = relax_time(subs)
    so, TT = second_order(subs, sigma, phi)
    base = dict(tau=tau, TT=TT, so=so, cv=a.get("cv"), n=len(subs))
    sA, b_fix = split(subs)
    if not sA:
        return a["key"], dict(base, exact=True, D=cap, se=0.0, z=0.0, secs=0.0)
    if T < 30 * tau:
        if 30 * tau <= a.get("tcap", T):                      # amendment 2: lengthen the run to 30 relaxation times
            T = int(math.ceil(30 * tau))
            T += T % 2
        else:
            return a["key"], dict(base, skipped=True, secs=0.0)
    burn = max(2000, int(math.ceil(8 * tau)))
    store = {}

    def run(ys, T_, b_, sd):
        r1, r2 = chain_sim(sA, b_fix, sigma, phi, ys, M, T_, b_, sd)
        store["last"] = (r1, r2)
        return 0.5 * (r1 + r2)

    D, se, Dm, runs = root(run, cap, M, T, burn, a["seed"])
    r1, r2 = store["last"]
    d = r2[1] - r1[1]
    z = float(d.mean() / (d.std(ddof=1) / math.sqrt(M)))
    mso = matrix_second_order(subs, sigma, phi)
    return a["key"], dict(base, D=D, se=se, z=z, burn=burn, T=T, runs=runs, mso=mso, secs=time.time() - t0)


def task_pure(a):
    """R4: a ring with every stay fixed, simulated with the delay recursion of run R15a (one cohort followed)."""
    t0 = time.time()
    M, T = a["size"]
    run = lambda ys, T_, b_, sd: L.sim_delay([0.0], [1.0], 1.0, a["n"] - 1, a["sigma"], a["phi"], ys, M, T_, b_, sd)
    cap = a["sigma"] ** 2 / 2
    D, se, Dm, runs = L.find_root(run, cap, 0.05, M, T, 2000, a["seed"])
    return a["key"], dict(D=D, se=se, secs=time.time() - t0)


TASKS = {"ring": task_ring, "chain": task_chain, "pure": task_pure}


def run_task(a):
    return TASKS[a["fn"]](a)


def read_r10():
    """R10's D and standard error for each H4 parameter set, from its results file."""
    out = {}
    on = False
    for line in open(R10_RESULTS):
        if line.startswith("H4"):
            on = True
            continue
        if on:
            m = re.match(r"\s+(.*?)\s+D = ([0-9.]+) \+- ([0-9.]+)\s+second order ([0-9.]+)", line)
            if not m:
                if line.strip() == "":
                    break
                continue
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def chk(fails, name, ok, detail):
    if not ok:
        fails.append(name)
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def verdict(D, se, cap):
    thr = cap / 2
    return "survives" if D < thr - 3 * se else ("fails" if D > thr + 3 * se else "undecided")


def cv_half(points, D_geo, cap):
    """points: (cv, D) sorted by falling cv, D rising.  The cv at which D = (D_geo + cap)/2, by linear
    interpolation between the two neighbours that bracket it."""
    target = (D_geo + cap) / 2
    for (c1, d1), (c2, d2) in zip(points[:-1], points[1:]):
        if d1 <= target <= d2 and d2 > d1:
            return c1 + (target - d1) * (c2 - c1) / (d2 - d1), c1, c2, target
    return None, None, None, target


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    print("R10b: the table of real parasites when the long stage is not memoryless.  Protocol: protocols/R10b-real-systems-stay-shape.md")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (system clock); simulation lengths divided by {SMOKE}")
    print("Inputs: the H4 parameter sets of checks/r10_loss_at_large_noise.py.  ± is one standard error.  cap = sigma^2/2, the loss with no staying.")
    r10 = read_r10()
    fails = []
    seed = [18_000_000]

    def nxt():
        seed[0] += 1000
        return seed[0]

    # ---- phase 1: reproduction (A0) and the second simulator on the same sets (R2)
    p1 = []
    for label, stays, hosts in A0_SETS:
        p1.append(dict(fn="ring", key=("A0", label), stays=stays, hosts=hosts, size=size("A0"), seed=nxt()))
        if len(hosts) == 1 and "fox" not in label and "the vole" not in label:
            subs = [1 - 1 / t for t in stays]
            p1.append(dict(fn="chain", key=("A0c", label), subs=subs, sigma=hosts[0][1], phi=hosts[0][2], size=size("A0"), seed=nxt()))
    res = {}
    with mp.Pool(3) as pool:
        for k, r in pool.imap_unordered(run_task, p1):
            res[k] = r
        print("\n" + "=" * 118 + "\nR1  REPRODUCTION OF R10 (H4), every stay as R10 has it\n" + "=" * 118)
        print("      parameter set                                     this run              R10                difference / combined se   second order (R10's)")
        ok1 = True
        for label, stays, hosts in A0_SETS:
            r = res[("A0", label)]
            D0, se0, so0 = r10[label]
            z = (r["D"] - D0) / math.hypot(r["se"], se0)
            good = abs(r["D"] - D0) <= 4 * math.hypot(r["se"], se0) + 5e-5
            ok1 &= good
            print(f"      {label:48s}  {r['D']:.4f} ± {r['se']:.4f}     {D0:.4f} ± {se0:.4f}     {z:+5.1f}  [{'ok' if good else 'FAIL'}]              {so0:.4f}")
        print(chk(fails, "R1 the 11 parameter sets of R10's H4 are reproduced", ok1, "4 combined standard errors plus the rounding of R10's printed value"))
        if not ok1:
            print("\nR1 FAILED: the run stops here, as the protocol requires.  Nothing else was computed.")
            return 1
        ok2 = True
        det = []
        for label, stays, hosts in A0_SETS:
            if ("A0c", label) in res:
                ra, rc = res[("A0", label)], res[("A0c", label)]
                z = (ra["D"] - rc["D"]) / math.hypot(ra["se"], rc["se"])
                ok2 &= abs(z) <= 3.5
                det.append(f"{label.split()[0]} {rc['D']:.4f} ± {rc['se']:.4f} ({z:+.1f})")
        print(chk(fails, "R2 the simulator written for this run agrees with the ring simulator", ok2, "; ".join(det) + "  (new simulator; difference in combined se)"))

        # ---- phase 2: the shapes
        p2 = []
        for row, R in ROWS.items():
            for m in R["mA"]:
                if m == 1:
                    continue                                   # the all-geometric case is the A0c run
                L_shape = "fixed" if abs(m - R["Lg"]) < 1e-12 else m
                F_shape = "fixed" if R["F"] == 1.0 else 1
                subs, phi, cv = build(row, "A", F_shape, L_shape)
                p2.append(dict(fn="chain", key=("A", row, m), subs=subs, sigma=R["sigma"], phi=phi, cv=cv, size=size("A"), seed=nxt()))
            r = R["r"]
            Lf = R["Lg"] * r
            if row != "2a":
                for m in R["mB"]:
                    subs, phi, cv = build(row, "B", "fixed", m)
                    kind = "Bheavy" if m >= 16 else "B"
                    p2.append(dict(fn="chain", key=("B1", row, m), subs=subs, sigma=R["sigma"], phi=phi, cv=cv, size=size(kind, r), seed=nxt()))
                subs, phi, cv = build(row, "B", "fixed", "fixed")
                p2.append(dict(fn="chain", key=("B1", row, "fixed"), subs=subs, sigma=R["sigma"], phi=phi, cv=cv, size=size("B", r), seed=nxt()))
                for F_shape in (4, 1):
                    for L_shape in (1, "fixed"):
                        subs, phi, cv = build(row, "B", F_shape, L_shape)
                        p2.append(dict(fn="chain", key=("B2", row, F_shape, L_shape), subs=subs, sigma=R["sigma"], phi=phi, cv=cv,
                                       size=size("B", r), tcap=TCAP, seed=nxt()))
                for gg in R["gB"]:                                # amendment 3: fixed (L - g) steps + a geometric stay of mean g
                    Lf_i = int(round(Lf))
                    subs = [0.0] * int(round(R["F"] * r)) + [0.0] * (Lf_i - gg) + [1.0 - 1.0 / gg]
                    p2.append(dict(fn="chain", key=("B4", row, gg), subs=subs, sigma=R["sigma"], phi=R["phi"] ** (1.0 / r),
                                   cv=math.sqrt(gg * (gg - 1.0)) / Lf_i, size=size("B", r), tcap=TCAP, seed=nxt()))
            else:
                for m in R["mB"]:
                    L_shape = "fixed" if m == Lf else m
                    subs, phi, cv = build(row, "B", 1, L_shape)
                    p2.append(dict(fn="chain", key=("B1", row, m), subs=subs, sigma=R["sigma"], phi=phi, cv=cv, size=size("B", r), seed=nxt()))
                for F_shape, L_shape in ((4, 1), (4, "fixed"), ("fixed", 1), ("fixed", "fixed")):
                    subs, phi, cv = build(row, "B", F_shape, L_shape)
                    p2.append(dict(fn="chain", key=("B2", row, F_shape, L_shape), subs=subs, sigma=R["sigma"], phi=phi, cv=cv,
                                   size=size("B", r), seed=nxt()))
        p2.append(dict(fn="pure", key=("R4", "1b all fixed"), n=25, sigma=1.380, phi=0.367, size=size("A"), seed=nxt()))
        p2.sort(key=lambda a: -(a["size"][0] * a["size"][1] * (len(split(a["subs"])[0]) + 2 if "subs" in a else 3)))
        for k, r in pool.imap_unordered(run_task, p2):
            res[k] = r

    chains = {k: r for k, r in res.items() if k[0] in ("A", "B1", "B2", "B4", "A0c")}
    sim = {k: r for k, r in chains.items() if "D" in r and not r.get("exact")}
    worst = max(abs(r["mso"] - r["so"]) / abs(r["so"]) for r in sim.values())
    print(chk(fails, "R3 closed form with R = E[phi^L] = matrix formula, every simulated shape", worst < 1e-9, f"{len(sim)} shapes, largest relative difference {worst:.1e}"))
    capof = {}
    for k in chains:
        if k[0] == "A0c":
            capof[k] = [h[1] for lab, st, h_ in A0_SETS if lab == k[1] for h in h_][0] ** 2 / 2
        else:
            capof[k] = ROWS[k[1]]["sigma"] ** 2 / 2
    over = [(k, r) for k, r in sim.items() if r["D"] > capof[k] + 3 * r["se"]]
    rp = res[("R4", "1b all fixed")]
    okp = abs(rp["D"] - 1.380 ** 2 / 2) <= max(3 * rp["se"], 0.01 * 1.380 ** 2 / 2)
    print(chk(fails, "R4 D <= sigma^2/2 in every case; a ring with every stay fixed gives sigma^2/2", not over and okp,
              f"{len(sim)} simulated cases, {len(over)} above the bound by 3 se; row 1b with both stages fixed: {rp['D']:.4f} ± {rp['se']:.4f} against {1.380 ** 2 / 2:.4f}"))
    flagged = [k for k, r in sim.items() if abs(r["z"]) > 3]
    zmax = max(abs(r["z"]) for r in sim.values())
    print(chk(fails, "R5 the two halves of every run give the same growth rate at the root", zmax <= 4 and len(flagged) <= 3,
              f"{len(sim)} cases; largest |difference| {zmax:.1f} se (criterion, amendment 4: none above 4 and at most three above 3); "
              f"flagged above 3 se and left out of the interpolation: {flagged if flagged else 'none'}"))
    skipped = [k for k, r in chains.items() if r.get("skipped")]
    print("      cases not run because the run length is under 30 relaxation times: " +
          (", ".join(f"{k} (relaxation time {chains[k]['tau']:.0f} steps)" for k in skipped) if skipped else "none"))

    def years(r, row, part):
        R = ROWS[row]
        return r["tau"] * R["dt"] / (1 if part == "A" else R["r"]) / 12.0

    def line(tag, r, cap, Dgeo, row, part):
        if r.get("skipped"):
            return (f"      {tag:34s} cv {r['cv']:.3f}   not run: relaxation time {r['tau']:.0f} steps = {r['tau'] / r['TT']:.0f} generations = {years(r, row, part):.0f} years;"
                    f" second order {r['so']:.4f}")
        v = verdict(r["D"], r["se"], cap)
        tg = "  inf" if math.isinf(r["tau"]) else f"{r['tau'] / r['TT']:5.1f}"
        ty = "inf" if math.isinf(r["tau"]) else f"{years(r, row, part):.1f}"
        return (f"      {tag:34s} cv {r['cv']:.3f}   D = {r['D']:.4f} ± {r['se']:.4f}   D/cap = {r['D'] / cap:.3f}   D/D_geometric = {r['D'] / Dgeo:5.2f}   "
                f"second order {r['so']:.4f}   relaxation {tg} generations ({ty} years)   [{v}]{'  FLAGGED' if (not r.get('exact')) and abs(r['z']) > 3 else ''}")

    half = {}
    for part, title in (("A", "PART A: R10's OWN TIME STEP.  Fluctuating stage as in R10; the long stage a chain of m equal sub-stages with the same mean"),
                        ("B", "PART B: A TIME STEP r TIMES FINER (host memory phi^(1/r) per fine step, same sigma, same mean stays).  B1: fluctuating stage fixed (as in R10)")):
        print("\n" + "=" * 118 + "\n" + title + "\n" + "=" * 118)
        print("      'survives' = D below half of cap by more than 3 se; 'fails' = above by more than 3 se.  cv = coefficient of variation of the long stage.")
        for row, R in ROWS.items():
            cap = R["sigma"] ** 2 / 2
            D0 = r10[A0_OF_ROW[row]][0]
            if part == "A":
                geo = res[("A0c", A0_OF_ROW[row])]
                geo = dict(geo, cv=stage_cv(R["Lg"], 1))
                rows_ = [(1, geo)] + [(m, res[("A", row, m)]) for m in R["mA"] if m != 1]
                print(f"  {row} {R['label']}: stays ({R['F']:g}, {R['Lg']:.4g}{', 1' if R['other'] else ''}) steps, sigma {R['sigma']}, phi {R['phi']}; cap {cap:.4f}; R10's D {D0:.4f} (D/cap {D0 / cap:.3f})")
            else:
                r = R["r"]
                rows_ = [(m, res[("B1", row, m)]) for m in R["mB"]]
                if row != "2a":
                    rows_.append(("fixed", res[("B1", row, "fixed")]))
                geo = rows_[0][1]
                print(f"  {row} {R['label']}: r = {r}, stays ({R['F'] * r:g}, {R['Lg'] * r:g}{', ' + str(R['other'] * r) if R['other'] else ''}) fine steps, phi per fine step {R['phi'] ** (1 / r):.4f}; cap {cap:.4f}"
                      + ("; egg geometric, fox fixed" if row == "2a" else ""))
            pts = []
            for m, rr in rows_:
                tag = "fixed delay" if rr["cv"] == 0 else (f"m = {m}" + (" (geometric)" if m == 1 else ""))
                if rr.get("exact"):
                    tag += " (exact: no staying)"
                print(line(tag, rr, cap, geo["D"], row, part))
                if "D" in rr and (rr.get("exact") or abs(rr["z"]) <= 3):
                    pts.append((rr["cv"], rr["D"]))
            cvh, c1, c2, target = cv_half(pts, geo["D"], cap)
            half[(part, row)] = (cvh, c1, c2, target)
            if cvh is None:
                print(f"        half of the protection is lost at D = {target:.4f}: not reached by any shape computed (largest D {max(p[1] for p in pts):.4f})")
            else:
                print(f"        half of the protection is lost at D = {target:.4f}: between cv {c1:.3f} and {c2:.3f}; linear interpolation cv = {cvh:.3f}")
            if part == "B" and R["gB"]:
                print("        a second family (amendment 3): the long stage = a fixed part + a geometric stay of mean g fine steps, same total mean")
                pts = [(geo["cv"], geo["D"])]
                for gg in R["gB"]:
                    rr = res[("B4", row, gg)]
                    print(line(f"  fixed {int(round(R['Lg'] * R['r'])) - gg} + geometric {gg}", rr, cap, geo["D"], row, part))
                    if "D" in rr and abs(rr["z"]) <= 3:
                        pts.append((rr["cv"], rr["D"]))
                pts.append((0.0, cap))
                cvh, c1, c2, target = cv_half(pts, geo["D"], cap)
                half[("B4", row)] = (cvh, c1, c2, target)
                if cvh is None:
                    print(f"        half of the protection is lost at D = {target:.4f}: not bracketed by this family")
                else:
                    print(f"        half of the protection is lost at D = {target:.4f}: between cv {c1:.3f} and {c2:.3f}; linear interpolation cv = {cvh:.3f}")

    print("\n" + "=" * 118 + "\nPART B2: WHICH STAGE'S SHAPE MATTERS MORE (fine step).  Rows: shape of the fluctuating stage.  Columns: long stage geometric, long stage fixed\n" + "=" * 118)
    for row, R in ROWS.items():
        cap = R["sigma"] ** 2 / 2
        r = R["r"]
        print(f"  {row} {R['label']}: cap {cap:.4f}; fluctuating stage {R['F'] * r:g} fine steps, long stage {R['Lg'] * r:g}")
        for F_shape, nm in (("fixed", "fixed (as in R10)" if row != "2a" else "fixed"), (4, "chain of 4"), (1, "geometric" if row != "2a" else "geometric (as in B1)")):
            cells = []
            for L_shape in (1, "fixed"):
                if row != "2a" and F_shape == "fixed":
                    rr = res[("B1", row, 1 if L_shape == 1 else "fixed")]
                elif row == "2a" and F_shape == 1:
                    rr = res[("B1", row, 1 if L_shape == 1 else 16)]
                else:
                    rr = res[("B2", row, F_shape, L_shape)]
                if rr.get("skipped"):
                    cells.append(f"not run (relaxation {rr['tau']:.0f} fine steps = {rr['tau'] / rr['TT']:.0f} generations = {years(rr, row, 'B'):.0f} years; second order {rr['so']:.4f})")
                else:
                    cells.append(f"{rr['D']:.4f} ± {rr['se']:.4f} (D/cap {rr['D'] / cap:.3f}; second order {rr['so']:.4f})" + (" exact" if rr.get("exact") else ""))
            print(f"      fluctuating stage {nm:20s}:  long geometric {cells[0]}    long fixed {cells[1]}")

    print("\n" + "=" * 118 + "\nSUMMARY\n" + "=" * 118)
    print("  D as a fraction of cap = sigma^2/2, by shape of the long stage (A = R10's step, B = finer step; '-' = shape does not exist at that step or was not run)")
    print("      row    R10      A: m=1    m=2      m=4      m=16     fixed  |  B: m=1    m=2      m=4      m=16     fixed")
    for row, R in ROWS.items():
        cap = R["sigma"] ** 2 / 2

        def cell(key):
            rr = res.get(key)
            if rr is None or "D" not in rr:
                return "   -    "
            return f"{rr['D'] / cap:7.3f} "
        a1 = res[("A0c", A0_OF_ROW[row])]["D"] / cap
        fixedA = ("A", row, int(R["Lg"])) if abs(R["Lg"] - round(R["Lg"])) < 1e-12 else None
        fixedB = ("B1", row, "fixed") if row != "2a" else ("B1", row, 16)
        cellsA = f"{a1:7.3f} " + cell(("A", row, 2) if 2 != R["Lg"] else None) + cell(("A", row, 4)) + cell(("A", row, 16)) + cell(fixedA)
        cellsB = cell(("B1", row, 1)) + cell(("B1", row, 2)) + cell(("B1", row, 4)) + (cell(("B1", row, 16)) if row != "2a" else "   -    ") + cell(fixedB)
        print(f"      {row:4s} {r10[A0_OF_ROW[row]][0] / cap:7.3f}   {cellsA}|  {cellsB}")
    print("  coefficient of variation of the long stage at which half of the protection is lost (linear interpolation between neighbouring shapes)")
    for row in ROWS:
        txt = []
        for part, nm in (("A", "A, chains"), ("B", "B, chains"), ("B4", "B, fixed part + geometric")):
            if (part, row) not in half:
                continue
            cvh, c1, c2, target = half[(part, row)]
            txt.append(f"{nm}: " + (f"{cvh:.3f} (between {c2:.3f} and {c1:.3f})" if cvh is not None else "not reached"))
        print(f"      {row:4s} " + ";   ".join(txt))

    secs = sum(r.get("secs", 0.0) for r in res.values())
    print(f"\n{len(res)} tasks; process time {secs:.0f} s; wall time {time.time() - t_start:.0f} s; finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
