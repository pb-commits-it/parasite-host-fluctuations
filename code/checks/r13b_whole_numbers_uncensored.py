"""Run R13b (protocols/R13b-whole-numbers-uncensored.md): the cases of run R13 with no replicate stopped early.

Claim tested: none is assumed.  R13 stopped a replicate above 10^15 individuals and counted it as not
extinct, so its fractions were lower bounds.  Here every replicate is followed to the end of its horizon.

Sampler (protocol, part A), applied to each random draw separately, N_BIG = 1e9:
  binomial draw with n <  N_BIG: exact;  n >= N_BIG: n s + sqrt(n s (1 - s)) xi, rounded, kept in [0, n];
  offspring draw with mean m < N_BIG: exact;  m >= N_BIG: m + sqrt(variance) xi, rounded, not below 0.
A replicate whose total passes 1e200 is divided by 1e100 and the log of its scale is kept.

Order of work: (1) checks of the method H0, V1 to V4, with the tolerances of the protocol; the main cases
are run only if all pass; (2) R13's 20 cases, all W1 cases followed for T_EQ = 1,020,000 steps, read at
R13's horizon (equal generations) and at T_EQ (equal steps); (3) other starting conditions; (4) other
offspring laws with the same mean matrix.

Run:   python3 -I checks/r13b_whole_numbers_uncensored.py     (six processes)
       ... --smoke   runs toy rings that are not cases of the protocol, to see that the script runs.
"""
from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import pathlib
import re
import sys
import time

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
PROTOCOL = (__import__('pathlib').Path(__file__).resolve().parent / '../protocols/R13b-whole-numbers-uncensored.md').resolve()
R13_OUTPUT = (__import__('pathlib').Path(__file__).resolve().parent / '../results/r13-whole-numbers_20261114.txt').resolve()


def _load_r13():
    spec = importlib.util.spec_from_file_location("r13_whole_numbers", HERE / "r13_whole_numbers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R13 = _load_r13()

NPROC = 6
N_BIG = 1e9            # a draw at or above this size is approximated
N_STRESS = 1e3         # the same, lowered, to force the approximation to be used in the checks
SCALE_HI = 1e200       # a replicate whose total passes this is divided by SCALE_STEP
SCALE_STEP = 1e100
CAP15 = 1e15           # R13's stopping size, recorded here and never acted on
EXACT_MAX = 2.0 ** 62  # the exact reference sampler cannot go beyond this
T_EQ = 1_020_000       # equal number of steps for every W1 ring
REPS = 2000
K_NB = 0.1             # dispersion of the clumped offspring law
B_GROUP = 100          # parasites per host in the shared-fate law
D_R10 = 0.0526         # the exact loss of run R10 used by check W0
LN_1E30 = math.log(1e30)
BUDGET_S = 240 * 60    # raised from 90 minutes by the amendment to the protocol (shared machine); nothing else changed
SMOKE = "--smoke" in sys.argv


# ------------------------------------------------------------------------------------------------ rings

def ring(s, g):
    """Mean matrix, stable stage proportions v (as R13 computes them), reproductive value u (u_0 = 1), growth factor."""
    s = np.asarray(s, float)
    g = np.asarray(g, float)
    n = len(s)
    A = np.diag(s)
    for k in range(n):
        A[(k + 1) % n, k] += g[k]
    vals, vecs = np.linalg.eig(A)
    v = np.abs(vecs[:, int(np.argmax(vals.real))].real)
    v = v / v.sum()
    valsl, vecsl = np.linalg.eig(A.T)
    u = np.abs(vecsl[:, int(np.argmax(valsl.real))].real)
    u = u / u[0]
    return A, v, u, float(vals.real.max())


def w1_rates(stays, lr):
    """Rates as R13 sets them for its W1 cases."""
    s, g, _ = R13.rates_for(stays, lr)
    if len(stays) == 2:
        g = (1 - s) * math.exp(lr / 2)
    return np.asarray(s, float), np.asarray(g, float)


def w0_rates(lr):
    """Rates as R13 sets them for check W0 (stays 1 and 7.5)."""
    st = np.asarray([1.0, 7.5])
    return 1 - 1 / st, (1 / st) * math.exp(lr / 2)


def start_state(rng, c, s, g, reps):
    """Starting numbers (in units of state) and the starting environment, in R13's order of draws."""
    kind, val = c["start"]
    if kind == "r13":                                   # R13's own: multinomial in the stable proportions
        _, v, _, _ = ring(s, g)
        X = rng.multinomial(int(val), v, size=reps).astype(np.int64)
    else:                                               # "fixed" (parasites) or "state" (units of state)
        X = np.tile(np.asarray(val, dtype=np.int64), (reps, 1))
    z = rng.standard_normal(reps)
    X = X.astype(np.float64)
    if c.get("law") == "group" and kind != "state":     # parasites of the steady stage into whole groups
        q = np.floor(X[:, 1] / B_GROUP)
        X[:, 1] = q + (rng.random(reps) < (X[:, 1] / B_GROUP - q))
    return X, z


# ---------------------------------------------------------------------------------------- hybrid sampler

def _approx(rng, mean, var, Lb):
    """mean + sqrt(var) xi.  Lb is None when no replicate is scaled; then the result is rounded.
    With a scale exp(Lb) the true variance is var * exp(Lb), so the scaled noise is sqrt(var * exp(-Lb))."""
    xi = rng.standard_normal(mean.shape)
    if Lb is None:
        return np.rint(mean + np.sqrt(var) * xi)
    val = mean + np.sqrt(var * np.exp(-Lb)) * xi
    un = Lb == 0
    if un.any():
        val[un] = np.rint(val[un])
    return val


def simulate(c):
    """One case, every replicate followed for c['steps'] steps.  Returns per-replicate records."""
    t_start = time.time()
    rng = np.random.default_rng(c["seed"])
    s = np.asarray(c["s"], float)
    g = np.asarray(c["g"], float)
    n = len(s)
    law = c.get("law", "poisson")
    reps, steps = c["reps"], c["steps"]
    sigma, phi = c["sigma"], c["phi"]
    n_big = c.get("n_big", N_BIG)
    scale_hi = c.get("scale_hi", SCALE_HI)
    scale_step = c.get("scale_step", SCALE_STEP)
    ln_step = math.log(scale_step)
    deadline = c.get("deadline")
    h_fixed = c.get("h_fixed")
    X, z = start_state(rng, c, s, g, reps)
    b = g / (1 - s)                                    # offspring per leaver, as R13
    w = np.ones(n)                                     # parasites per unit of state
    if law == "group":
        b = b.copy()
        b[0] = b[0] / B_GROUP                          # new groups founded per leaver of stage 0
        b[1] = b[1] * B_GROUP                          # free individuals produced by a group that leaves
        w[1] = B_GROUP
    weighted = law == "group"
    nb = law == "nb"
    b0 = b[0]
    innov = math.sqrt(1 - phi * phi)
    alive = np.arange(reps)
    lev = np.zeros(reps, dtype=np.int64)               # number of times scaled down (aligned with alive)
    L = np.zeros(reps)                                 # log of the scale (aligned)
    capped = np.zeros(reps, dtype=bool)                # has passed 1e15 (aligned)
    peak_prev = np.full(reps, -np.inf)                 # log of the peak in earlier scale epochs (aligned)
    tot = X @ w if weighted else X.sum(axis=1)
    peak_raw = tot.copy()                              # peak in the present scale (aligned)
    ext_at = np.full(reps, -1, dtype=np.int64)
    first_cap = np.full(reps, -1, dtype=np.int64)
    ever_norm = np.zeros(reps, dtype=bool)
    ever_scaled = np.zeros(reps, dtype=bool)
    peak_ln = np.full(reps, -np.inf)
    final_ln = np.full(reps, np.nan)
    nscaled = 0
    n_en = 0                                           # living replicates that have used the approximation
    perm = np.roll(np.arange(n), 1)                    # column k of the new state receives from stage k - 1
    t_first_norm = -1
    anomalies = 0
    steps_done = steps
    unfinished = False
    sc = None
    for t in range(steps):
        m = alive.size
        if m == 0:
            break
        if deadline is not None and (t & 4095) == 0 and time.time() > deadline:
            unfinished, steps_done = True, t
            break
        z = phi * z + innov * rng.standard_normal(m)
        h0 = np.exp(sigma * z - sigma * sigma / 2) if h_fixed is None else np.full(m, float(h_fixed))
        # ---- stayers
        if nscaled == 0 and X.max() < n_big:
            stay = rng.binomial(X.astype(np.int64), s)             # exact, R13's own call
        else:
            if t_first_norm < 0:
                t_first_norm = t + 1
            big = X >= n_big
            if nscaled:
                big |= sc[:, None]
            if n_en < m:
                ever_norm[alive[big.any(axis=1)]] = True
                n_en = int(np.count_nonzero(ever_norm[alive]))
            if big.all():                                           # no masks needed; same draws, same order
                mu = X * s
                Lb = np.broadcast_to(L[:, None], X.shape) if nscaled else None
                stay = np.minimum(np.maximum(_approx(rng, mu, mu * (1 - s), Lb), 0.0), X)
            else:
                S = np.broadcast_to(s, X.shape)
                stay = np.empty_like(X)
                small = ~big
                stay[small] = rng.binomial(X[small].astype(np.int64), S[small])
                xb, sb = X[big], S[big]
                Lb = np.broadcast_to(L[:, None], X.shape)[big] if nscaled else None
                val = _approx(rng, xb * sb, xb * sb * (1 - sb), Lb)
                stay[big] = np.minimum(np.maximum(val, 0.0), xb)
        leave = X - stay
        mean = leave * b
        mean[:, 0] = mean[:, 0] * h0
        # ---- offspring
        if nscaled == 0 and mean.max() < n_big:
            if nb:
                lam = mean.copy()
                lam[:, 0] = rng.gamma(K_NB * leave[:, 0], (b0 * h0) / K_NB)
                born = rng.poisson(lam)
            else:
                born = rng.poisson(mean)                            # exact, R13's own call
        else:
            if t_first_norm < 0:
                t_first_norm = t + 1
            bigm = mean >= n_big
            if nscaled:
                bigm |= sc[:, None]
            if n_en < m:
                ever_norm[alive[bigm.any(axis=1)]] = True
                n_en = int(np.count_nonzero(ever_norm[alive]))
            if nb:
                vf = np.ones_like(mean)
                vf[:, 0] = 1 + (b0 * h0) / K_NB
            if bigm.all():                                          # no masks needed; same draws, same order
                Lb = np.broadcast_to(L[:, None], mean.shape) if nscaled else None
                born = np.maximum(_approx(rng, mean, mean * vf if nb else mean, Lb), 0.0)
            else:
                born = np.empty_like(mean)
                smallm = ~bigm
                if smallm.any():
                    if nb:
                        lam = mean.copy()
                        r0 = smallm[:, 0]
                        if r0.any():
                            lam[r0, 0] = rng.gamma(K_NB * leave[r0, 0], (b0 * h0[r0]) / K_NB)
                        born[smallm] = rng.poisson(lam[smallm])
                    else:
                        born[smallm] = rng.poisson(mean[smallm])
                if bigm.any():
                    mb = mean[bigm]
                    Lb = np.broadcast_to(L[:, None], mean.shape)[bigm] if nscaled else None
                    born[bigm] = np.maximum(_approx(rng, mb, mb * vf[bigm] if nb else mb, Lb), 0.0)
        X = stay + (born if n == 1 else born[:, perm])              # individuals move one stage round the ring
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        tot = X @ w if weighted else X.sum(axis=1)
        tmax = tot.max()
        # ---- guard against overflow
        if tmax > scale_hi:
            idx = tot > scale_hi
            peak_prev[idx] = np.maximum(peak_prev[idx], np.log(np.maximum(peak_raw[idx], tot[idx])) + L[idx])
            X[idx] /= scale_step
            tot[idx] /= scale_step
            lev[idx] += 1
            L = lev * ln_step
            peak_raw[idx] = tot[idx]
            ever_scaled[alive[idx]] = True
            sc = lev > 0
            nscaled = int(np.count_nonzero(sc))
        np.maximum(peak_raw, tot, out=peak_raw)
        if nscaled:
            up = sc & (tot < 1.0)
            if up.any():
                peak_prev[up] = np.maximum(peak_prev[up], np.log(peak_raw[up]) + L[up])
                X[up] *= scale_step
                tot[up] *= scale_step
                lev[up] -= 1
                L = lev * ln_step
                peak_raw[up] = tot[up]
                sc = lev > 0
                nscaled = int(np.count_nonzero(sc))
            if nscaled:
                mn = X.min(axis=1)
                bad = sc & (np.log(np.maximum(mn, 1e-300)) + L < LN_1E30)
                anomalies += int(np.count_nonzero(bad))
        # ---- R13's stopping size: recorded, not acted on
        if tmax > CAP15 or nscaled:
            over = tot > CAP15
            if nscaled:
                over |= sc
            new = over & ~capped
            if new.any():
                first_cap[alive[new]] = t + 1
                capped |= new
        # ---- extinction
        dead = tot == 0
        if dead.any():
            di = alive[dead]
            ext_at[di] = t + 1
            peak_ln[di] = np.maximum(peak_prev[dead], np.log(peak_raw[dead]) + L[dead])
            keep = ~dead
            alive, X, z, lev, L = alive[keep], X[keep], z[keep], lev[keep], L[keep]
            capped, peak_prev, peak_raw = capped[keep], peak_prev[keep], peak_raw[keep]
            n_en = int(np.count_nonzero(ever_norm[alive]))
            if nscaled:
                sc = lev > 0
    if alive.size:
        tot = X @ w if weighted else X.sum(axis=1)
        final_ln[alive] = np.log(tot) + L
        peak_ln[alive] = np.maximum(peak_prev, np.log(peak_raw) + L)
    out = dict(ext_at=ext_at, first_cap=first_cap, ever_norm=ever_norm, ever_scaled=ever_scaled, peak_ln=peak_ln,
               final_ln=final_ln, t_first_norm=t_first_norm, anomalies=anomalies, steps_done=steps_done,
               unfinished=unfinished, secs=time.time() - t_start)
    if c.get("return_state"):
        out["state"] = X * w
    return out


# ------------------------------------------------------------------------- exact reference (R13's algorithm)

def exact_run(c, cap):
    """R13's algorithm (exact whole-number sampling throughout), with the time of extinction and the peak
    recorded.  A replicate whose total passes `cap` is removed and flagged, as in R13."""
    t_start = time.time()
    rng = np.random.default_rng(c["seed"])
    s = np.asarray(c["s"], float)
    g = np.asarray(c["g"], float)
    reps, steps, sigma, phi = c["reps"], c["steps"], c["sigma"], c["phi"]
    b = g / (1 - s)
    X, z = start_state(rng, c, s, g, reps)
    X = X.astype(np.int64)
    innov = math.sqrt(1 - phi * phi)
    alive = np.arange(reps)
    ext_at = np.full(reps, -1, dtype=np.int64)
    flagged = np.zeros(reps, dtype=bool)
    peak = X.sum(axis=1).astype(np.float64)
    for t in range(steps):
        if alive.size == 0:
            break
        z = phi * z + innov * rng.standard_normal(alive.size)
        h0 = np.exp(sigma * z - sigma * sigma / 2)
        stay = rng.binomial(X, s)
        leave = X - stay
        mean = leave * b
        mean[:, 0] = mean[:, 0] * h0
        toobig = mean.max(axis=1) >= EXACT_MAX
        if toobig.any():                                # cannot be drawn exactly: flag before drawing
            flagged[alive[toobig]] = True
            keep = ~toobig
            alive, X, z, stay, mean = alive[keep], X[keep], z[keep], stay[keep], mean[keep]
            if alive.size == 0:
                break
        born = rng.poisson(mean)
        X = stay + np.roll(born, 1, axis=1)
        tot = X.sum(axis=1)
        peak[alive] = np.maximum(peak[alive], tot)
        dead = tot == 0
        out = tot > cap
        if dead.any() or out.any():
            ext_at[alive[dead]] = t + 1
            flagged[alive[out]] = True
            keep = ~(dead | out)
            alive, X, z = alive[keep], X[keep], z[keep]
    return dict(ext_at=ext_at, flagged=flagged, peak=peak, secs=time.time() - t_start)


def work(task):
    kind, key, payload = task
    if kind == "sim":
        return key, simulate(payload)
    if kind == "exact":
        return key, exact_run(payload[0], payload[1])
    if kind == "r13":
        return key, R13.run_case(payload)
    raise ValueError(kind)


def run_tasks(pool, tasks, what):
    res = {}
    t0 = time.time()
    for i, (key, r) in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
        res[key] = r
        print(f"    [{what}] {i + 1}/{len(tasks)} done after {time.time() - t0:6.0f} s: {key}", file=sys.stderr, flush=True)
    return res


# --------------------------------------------------------------------------------------- known answers

def gf_survival(s, b, law, t_max):
    """p[T, k]: probability that the line of one founder of type k is NOT extinct after T steps, constant
    environment (h = 1).  1 - p is the T-fold iterate of the offspring generating function from 0."""
    s = np.asarray(s, float)
    b = np.asarray(b, float)
    n = len(s)
    p = np.ones(n)
    out = np.empty((t_max + 1, n))
    out[0] = p
    for T in range(1, t_max + 1):
        nxt = np.roll(p, -1)
        off = -np.expm1(-b * nxt)                                    # Poisson: 1 - exp(-b p)
        if law == "nb":
            off[0] = -np.expm1(-K_NB * np.log1p(b[0] * nxt[0] / K_NB))  # 1 - (1 + (b/k) p)^(-k)
        p = s * p + (1 - s) * off
        out[T] = p
    return out


def gf_limit(s, b, law):
    """The smallest fixed point (probability of ever dying out, one founder of each type), by iteration."""
    s = np.asarray(s, float)
    b = np.asarray(b, float)
    p = np.ones(len(s))
    for _ in range(5_000_000):
        nxt = np.roll(p, -1)
        off = -np.expm1(-b * nxt)
        if law == "nb":
            off[0] = -np.expm1(-K_NB * np.log1p(b[0] * nxt[0] / K_NB))
        new = s * p + (1 - s) * off
        if np.max(np.abs(new - p)) < 1e-15:
            p = new
            break
        p = new
    return 1 - p


def one_step_moments(s, b, x, h, law):
    """Mean and variance of each stage after one step from x; units of state; worked out by hand."""
    n = len(s)
    m = np.array(b, float)
    m[0] *= h
    vf = np.ones(n)
    if law == "nb":
        vf[0] = 1 + m[0] / K_NB
    stay_mean, stay_var = s * x, x * s * (1 - s)
    born_mean = m * x * (1 - s)
    born_var = m * x * (1 - s) * vf + m * m * x * s * (1 - s)
    if n == 1:
        return stay_mean + born_mean, m * x * (1 - s) * vf + (1 - m) ** 2 * x * s * (1 - s)
    return stay_mean + np.roll(born_mean, 1), stay_var + np.roll(born_var, 1)


# ------------------------------------------------------------------------------------------- helpers

def frac(ext_at, t):
    n = len(ext_at)
    k = int(np.count_nonzero((ext_at > 0) & (ext_at <= t)))
    p = k / n
    return p, math.sqrt(p * (1 - p) / n), k


def cmp2(k1, n1, k2, n2):
    """Two simulated fractions: difference and whether it is within 4 standard errors (pooled)."""
    p1, p2 = k1 / n1, k2 / n2
    pb = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2))
    return p1, p2, se, abs(p1 - p2) <= 4 * se + 1e-12


def parse_r13():
    """R13's saved numbers, read from its output file (never typed by hand)."""
    rows = {}
    pat = re.compile(r"^\s+(W\d .*?)\s+extinct ([\d.]+) \+- ([\d.]+)\s+above 1e15 ([\d.]+)\s+alive at the end ([\d.]+)")
    for line in R13_OUTPUT.read_text().splitlines():
        mt = pat.match(line)
        if mt:
            e, se, c, a = (float(x) for x in mt.groups()[1:])
            rows[mt.group(1).strip()] = dict(ext=e, se=se, cens=c, alive=a, n_ext=round(e * 2000), n_cens=round(c * 2000))
    return rows


def se_of(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def judge(d, se):
    r = abs(d) / se if se > 0 else (math.inf if d != 0 else 0.0)
    if r >= 4:
        return r, "established"
    if r >= 2:
        return r, "suggested, not established"
    return r, "no difference detected"


def r13_verdict(d, se):
    """Fate of an ordering that R13 stated as 'first above second'."""
    r, word = judge(d, se)
    if r >= 4:
        return "SURVIVES" if d > 0 else "FAILS (reversed)"
    return "NOT ESTABLISHED (" + word + ")"


def case_dict(case, **over):
    label, s, g, sigma, phi, n0, steps, seed = case
    c = dict(label=label.strip(), s=np.asarray(s, float), g=np.asarray(g, float), sigma=sigma, phi=phi,
             start=("r13", n0), steps=steps, seed=seed, reps=REPS)
    c.update(over)
    return c


def stays_of(label):
    mt = re.search(r"steady stay\s+(\S+),", label)
    return [10.0] if mt.group(1) == "none" else [10.0, float(mt.group(1))]


# ----------------------------------------------------------------------------------- checks of the method

def method_checks(pool, r13):
    print("\n== 1. Checks of the method (tolerance 4 standard errors unless equality is required) ==", flush=True)
    seed = [20264000 + (777 if SMOKE else 0)]

    def nxt():
        seed[0] += 1
        return seed[0]

    toy_s = 0.3 if SMOKE else None
    tasks = []
    base = R13.build()
    # H0: R13's own function again, and my reference sampler, on five of R13's cases
    h0_cases = [] if SMOKE else [c for c in base if c[0].startswith("W0") or "N1 steady stay none" in c[0]]
    for c in h0_cases:
        tasks.append(("r13", "H0|r13|" + c[0].strip(), c))
        tasks.append(("exact", "H0|ref|" + c[0].strip(), (case_dict(c), CAP15)))
    # V1: constant environment, known answer
    lr1 = 0.05 if SMOKE else 0.3
    T1 = 300 if SMOKE else 3000
    reps1 = 200 if SMOKE else 20000
    s1, g1 = w1_rates([10.0, 40.0], lr1)
    b1 = g1 / (1 - s1)
    v1_plan = [("V1a", "poisson", [("one founder in stage 0", (1, 0)), ("one founder in stage 1", (0, 1)), ("one founder in each", (1, 1))]),
               ("V1b", "nb", [("one founder in stage 0", (1, 0)), ("one founder in stage 1", (0, 1)), ("one founder in each", (1, 1))]),
               ("V1c", "group", [("one free individual", (1, 0)), ("one group of 100", (0, 1)), ("100 free individuals", (100, 0))])]
    for tag, law, starts in v1_plan:
        for name, x in starts:
            for nbig in (N_BIG, N_STRESS):
                tasks.append(("sim", f"{tag}|{name}|{nbig:.0e}", dict(s=s1, g=g1, sigma=0.0, phi=0.0, law=law, start=("state", x),
                                                                     steps=T1, reps=reps1, n_big=nbig, seed=nxt())))
    # V1d: a fall through the switch, constant environment
    lrd = -0.2 if SMOKE else -0.5
    repsd = 100 if SMOKE else 10000
    sd_, gd_ = w1_rates([10.0, 40.0], lrd)
    bd_ = gd_ / (1 - sd_)
    _, vd_, _, _ = ring(sd_, gd_)
    xd = np.rint(2e10 * vd_)
    v1d = {}
    for law in ("poisson", "nb"):
        p = gf_survival(sd_, bd_, law, 60000)
        with np.errstate(divide="ignore"):
            P = np.exp(np.log1p(-p) @ xd)
        if P[-1] < 0.9:
            raise RuntimeError("V1d: the exact probability does not reach 0.9 within the steps computed")
        times =[int(np.argmax(P >= lev)) for lev in (0.1, 0.3, 0.5, 0.7, 0.9)]
        v1d[law] = (times, [float(P[t]) for t in times])
        tasks.append(("sim", f"V1d|{law}", dict(s=sd_, g=gd_, sigma=0.0, phi=0.0, law=law, start=("fixed", xd), steps=times[-1],
                                               reps=repsd, n_big=N_BIG, seed=nxt())))
    # V2: hybrid against exact in a random environment
    reps2 = 200 if SMOKE else 10000
    sa, ga = w0_rates(-0.2 if SMOKE else D_R10 - 0.3)
    sb_, gb_ = w0_rates(-0.2 if SMOKE else D_R10 - 0.03)
    sc_, gc_ = w1_rates([10.0, 160.0], -0.2 if SMOKE else -1.0)
    v2_plan = {"V2a": dict(s=sa, g=ga, sigma=toy_s or 0.6, phi=0.75, start=("r13", 20_000_000_000), steps=6000),
               "V2b": dict(s=sb_, g=gb_, sigma=toy_s or 0.6, phi=0.75, start=("r13", 200), steps=1500),
               "V2c": dict(s=sc_, g=gc_, sigma=toy_s or 0.15, phi=0.9, start=("r13", 20_000_000_000), steps=12000)}
    for tag, spec in v2_plan.items():
        tasks.append(("exact", f"{tag}|exact", (dict(spec, reps=reps2, seed=nxt()), EXACT_MAX)))
        for nbig in (N_BIG, N_STRESS):
            tasks.append(("sim", f"{tag}|{nbig:.0e}", dict(spec, reps=reps2, n_big=nbig, seed=nxt())))
    # V3: the overflow guard
    s3, g3 = w0_rates(0.05 if SMOKE else D_R10 + 0.3)
    spec3 = dict(s=s3, g=g3, sigma=toy_s or 0.6, phi=0.75, start=("r13", 200), steps=8000, reps=100 if SMOKE else 1000, seed=nxt())
    tasks.append(("sim", "V3|guard as in the main run", dict(spec3)))
    tasks.append(("sim", "V3|guard set to act early", dict(spec3, scale_hi=1e12 if SMOKE else 1e60, scale_step=1e6 if SMOKE else 1e30)))
    # V4: one step
    n1_40 = [c for c in base if c[0].strip() == "W1 N1 steady stay   40, start 10000"][0]
    n1_none = [c for c in base if c[0].strip() == "W1 N1 steady stay none, start 10000"][0]
    if SMOKE:
        s4, g4 = w1_rates([10.0, 40.0], 0.05)
        s4n, g4n = w1_rates([10.0], 0.05)
    else:
        s4, g4 = np.asarray(n1_40[1], float), np.asarray(n1_40[2], float)
        s4n, g4n = np.asarray(n1_none[1], float), np.asarray(n1_none[2], float)
    reps4 = 500 if SMOKE else 20000
    v4_plan = [("poisson, two stages", "poisson", s4, g4), ("clumped (E-i), two stages", "nb", s4, g4),
               ("shared host fate (E-ii), two stages", "group", s4, g4), ("poisson, one stage", "poisson", s4n, g4n)]
    for name, law, s_, g_ in v4_plan:
        for size in (1e6, 4e9):
            tasks.append(("sim", f"V4|{name}|{size:.0e}", dict(s=s_, g=g_, sigma=0.0, phi=0.0, law=law, start=("state", [size] * len(s_)),
                                                                steps=1, reps=reps4, h_fixed=1.3, return_state=True, seed=nxt())))
    order = {"r13": 0, "exact": 1, "sim": 2}
    tasks.sort(key=lambda tk: (order[tk[0]], -tk[2].get("steps", 0) * tk[2].get("reps", 0) if tk[0] == "sim" else 0))
    res = run_tasks(pool, tasks, "checks")
    all_ok = True

    def verdict(tag, ok, void, text):
        nonlocal all_ok
        word = "VOID" if void else ("PASS" if ok else "FAIL")
        if void or not ok:
            all_ok = False
        print(f"[{word}] {tag} {text}", flush=True)

    # ---- H0
    if not SMOKE:
        ok = True
        for c in h0_cases:
            lab = c[0].strip()
            saved = r13[lab]
            again = res["H0|r13|" + lab]
            ref = res["H0|ref|" + lab]
            n_ref_ext = int(np.count_nonzero(ref["ext_at"] > 0))
            n_ref_flag = int(ref["flagged"].sum())
            same = (again[1] == saved["n_ext"] and again[2] == saved["n_cens"] and n_ref_ext == saved["n_ext"] and n_ref_flag == saved["n_cens"])
            ok = ok and same
            print(f"  {lab:44s} saved by R13: extinct {saved['n_ext']:5d}, stopped {saved['n_cens']:4d} | R13's function again: {again[1]:5d}, {again[2]:4d} | "
                  f"reference sampler: {n_ref_ext:5d}, {n_ref_flag:4d}   {'same' if same else 'DIFFERENT'}")
        verdict("H0", ok, False, "R13's own function, run again, and the reference sampler both give R13's saved counts exactly (five cases)")
    # ---- V1a-c
    tfr = [T1 // 10, T1 // 4, T1 // 2, T1]
    for tag, law, starts in v1_plan:
        bb = b1.copy()
        if law == "group":
            bb[0], bb[1] = bb[0] / B_GROUP, bb[1] * B_GROUP
        gl = "nb" if law == "nb" else "poisson"
        p = gf_survival(s1, bb, gl, T1)
        qinf = gf_limit(s1, bb, gl)
        print(f"  {tag} ({law}; stays 10 and 40, ln R0 = {lr1}): smallest fixed point q = ({qinf[0]:.6f}, {qinf[1]:.6f}); "
              f"q({T1}) = ({1 - p[T1, 0]:.6f}, {1 - p[T1, 1]:.6f})")
        ok = True
        worst = 0.0
        for name, x in starts:
            exact = [float(np.exp(np.log1p(-p[T]) @ np.asarray(x, float))) for T in tfr]
            for nbig in (N_BIG, N_STRESS):
                r = res[f"{tag}|{name}|{nbig:.0e}"]
                cells = []
                for T, pe in zip(tfr, exact):
                    ps, _, _ = frac(r["ext_at"], T)
                    se = se_of(pe, reps1)
                    dev = abs(ps - pe) / se if se > 0 else (0.0 if ps == pe else math.inf)
                    worst = max(worst, dev)
                    ok = ok and dev <= 4
                    cells.append(f"T={T}: {ps:.4f} vs {pe:.4f} ({dev:.1f} se)")
                print(f"    {name:24s} N_big {nbig:.0e}, approximated in {int(r['ever_norm'].sum()):5d} replicates: " + "; ".join(cells))
        verdict(tag, ok, False, f"extinct fraction within 4 binomial standard errors of the generating-function value at every time (worst {worst:.1f})")
    # ---- V1d
    ok = True
    worst = 0.0
    for law in ("poisson", "nb"):
        times, exact = v1d[law]
        r = res[f"V1d|{law}"]
        cells = []
        for T, pe in zip(times, exact):
            ps, _, _ = frac(r["ext_at"], T)
            se = se_of(pe, repsd)
            dev = abs(ps - pe) / se if se > 0 else (0.0 if ps == pe else math.inf)
            worst = max(worst, dev)
            ok = ok and dev <= 4
            cells.append(f"T={T}: {ps:.4f} vs {pe:.4f} ({dev:.1f} se)")
        print(f"  V1d ({law}; ln R0 = {lrd}; start {xd[0]:.3e}, {xd[1]:.3e}; approximated in {int(r['ever_norm'].sum())} of {repsd} replicates): " + "; ".join(cells))
    verdict("V1d", ok, False, f"a fall from 2e10 through the switch: within 4 binomial standard errors of the exact value at five times (worst {worst:.1f})")
    # ---- V2
    for tag in ("V2a", "V2b", "V2c"):
        ex = res[f"{tag}|exact"]
        nflag = int(ex["flagged"].sum())
        ok = True
        worst = 0.0
        e_ext = ex["ext_at"]
        for nbig in (N_BIG, N_STRESS):
            hy = res[f"{tag}|{nbig:.0e}"]
            h_ext = hy["ext_at"]
            cells = []
            if tag == "V2b":
                steps = v2_plan[tag]["steps"]
                stats = [("extinct at the end", int((e_ext > 0).sum()), int((h_ext > 0).sum())),
                         ("ever above 1e6", int((ex["peak"] > 1e6).sum()), int((hy["peak_ln"] > math.log(1e6)).sum())),
                         ("ever above 1e10", int((ex["peak"] > 1e10).sum()), int((hy["peak_ln"] > math.log(1e10)).sum()))]
            else:
                te, th = e_ext[e_ext > 0], h_ext[h_ext > 0]
                stats = []
                if te.size > 1 and th.size > 1:
                    me, mh = te.mean(), th.mean()
                    sem = math.sqrt(te.var(ddof=1) / te.size + th.var(ddof=1) / th.size)
                    dev = abs(me - mh) / sem if sem > 0 else 0.0
                    worst = max(worst, dev)
                    ok = ok and dev <= 4
                    cells.append(f"mean time to extinction {mh:.1f} vs {me:.1f} ({dev:.1f} se)")
                    qs = np.quantile(np.concatenate([te, th]), [0.25, 0.5, 0.75])
                    for q in qs:
                        stats.append((f"extinct by step {q:.0f}", int(((e_ext > 0) & (e_ext <= q)).sum()), int(((h_ext > 0) & (h_ext <= q)).sum())))
                else:
                    ok = False
                    cells.append("too few extinctions to compare")
                stats.append(("extinct at the end", int((e_ext > 0).sum()), int((h_ext > 0).sum())))
            for name, ke, kh in stats:
                pe, ph, se, good = cmp2(ke, e_ext.size, kh, h_ext.size)
                dev = abs(pe - ph) / se if se > 0 else 0.0
                worst = max(worst, dev)
                ok = ok and good
                cells.append(f"{name} {ph:.4f} vs {pe:.4f} ({dev:.1f} se)")
            print(f"  {tag} hybrid N_big {nbig:.0e} (approximated in {int(hy['ever_norm'].sum())} of {h_ext.size}) vs exact: " + "; ".join(cells))
        desc = {"V2a": "a fall from 2e10, the ring of W0", "V2b": "rise and return from 200, W0 below the threshold, 1,500 steps",
                "V2c": "a fall from 2e10, stays 10 and 160"}[tag]
        verdict(tag, ok, nflag > 0, f"{desc}: hybrid against exact sampling (worst {worst:.1f} standard errors; exact arm flagged {nflag} replicates, must be 0)")
    # ---- V3
    ra, rb = res["V3|guard as in the main run"], res["V3|guard set to act early"]
    same_ext = bool(np.array_equal(ra["ext_at"], rb["ext_at"]))
    surv = (ra["ext_at"] < 0) & (rb["ext_at"] < 0)
    dmax = float(np.max(np.abs(ra["final_ln"][surv] - rb["final_ln"][surv]))) if surv.any() else 0.0
    n_a, n_b = int(ra["ever_scaled"].sum()), int(rb["ever_scaled"].sum())
    void = not (n_a == 0 and n_b >= 100)
    verdict("V3", same_ext and dmax <= 1e-6, void and not SMOKE,
            f"overflow guard: times of extinction identical: {same_ext}; largest difference in the log of the final total {dmax:.2e} (limit 1e-6); "
            f"replicates scaled {n_a} (guard as in the main run) and {n_b} (guard set to act early; at least 100 needed); "
            f"largest log10 size reached {np.nanmax(ra['peak_ln']) / math.log(10):.1f}")
    # ---- V4
    ok = True
    worst_m = worst_v = 0.0
    for name, law, s_, g_ in v4_plan:
        n = len(s_)
        b_ = g_ / (1 - s_)
        wv = np.ones(n)
        if law == "group":
            b_ = b_.copy()
            b_[0], b_[1] = b_[0] / B_GROUP, b_[1] * B_GROUP
            wv[1] = B_GROUP
        Ah = np.diag(s_)                                  # R13's mean matrix with h_0 = 1.3, in parasites
        for k in range(n):
            Ah[(k + 1) % n, k] += g_[k] * (1.3 if k == 0 else 1.0)
        for size in (1e6, 4e9):
            r = res[f"V4|{name}|{size:.0e}"]
            Y = r["state"]
            x_state = np.full(n, size)
            target = Ah @ (x_state * wv)
            _, var_th = one_step_moments(s_, b_, x_state, 1.3, law)
            var_th = var_th * wv * wv
            mean_s, var_s = Y.mean(axis=0), Y.var(axis=0, ddof=1)
            dev_m = np.abs(mean_s - target) / np.sqrt(var_s / Y.shape[0])
            dev_v = np.abs(var_s / var_th - 1) / math.sqrt(2 / (Y.shape[0] - 1))
            worst_m, worst_v = max(worst_m, float(dev_m.max())), max(worst_v, float(dev_v.max()))
            ok = ok and bool((dev_m <= 4).all()) and bool((dev_v <= 4).all())
            print(f"  V4 {name:36s} from {size:.0e} per stage ({'approximated' if r['ever_norm'].any() else 'exact'} draws): mean/(A x) - 1 = "
                  + ", ".join(f"{a / b - 1:+.2e}" for a, b in zip(mean_s, target)) + f" ({dev_m.max():.1f} se); variance/theory = "
                  + ", ".join(f"{a / b:.4f}" for a, b in zip(var_s, var_th)) + f" ({dev_v.max():.1f} se)")
    verdict("V4", ok, False, f"one step: mean equal to R13's mean matrix times the start for all three laws (worst {worst_m:.1f} se), "
                             f"variance equal to the hand formula (worst {worst_v:.1f} se)")
    return all_ok


# ------------------------------------------------------------------------------------------ main cases

def main_cases(pool, r13, deadline):
    base = R13.build()
    tasks = []
    info = {}
    for i, case in enumerate(base):
        label = case[0].strip()
        is_w1 = label.startswith("W1")
        c = case_dict(case)
        r13_steps = c["steps"]
        if SMOKE:
            if is_w1:
                c["s"], c["g"] = w1_rates(stays_of(label), 0.05)
            else:
                c["s"], c["g"] = w0_rates(-0.2 if "below" in label else 0.05)
            c["sigma"], c["reps"], c["seed"] = 0.3, 30, c["seed"] + 777
            r13_steps = max(1, r13_steps // 500)
            if is_w1:
                c["start"] = ("r13", c["start"][1] * 20000)        # toy: starts near the switching size
        horizon = (T_EQ // 500 if SMOKE else T_EQ) if is_w1 else r13_steps
        c["steps"], c["deadline"] = horizon, deadline
        n0 = case[5]
        batches = 3 if (is_w1 and n0 == 100) else 1
        info[label] = dict(r13_steps=r13_steps, horizon=horizon, batches=batches, n0=n0, s=c["s"], g=c["g"], is_w1=is_w1)
        for bt in range(batches):
            tasks.append(("sim", f"base|{label}|{bt}", dict(c, seed=c["seed"] + 1000 * bt)))
    # D and E: the headline comparison (N1 from 10,000)
    head = {}
    for stay in ("none", "10", "40", "160"):
        head[stay] = f"W1 N1 steady stay {stay:>4}, start 10000"
    dseed, eseed = 20262000 + (777 if SMOKE else 0), 20263000 + (777 if SMOKE else 0)
    dstarts = {}
    for stay in ("10", "40", "160"):
        lab = head[stay]
        s_, g_ = info[lab]["s"], info[lab]["g"]
        _, v, u, lam = ring(s_, g_)
        x2 = np.array([10000.0, 0.0])
        x3 = np.array([0.0, np.rint(10000.0 / u[1])])
        x4 = np.rint(10000.0 / float(u @ v) * v)
        dstarts[stay] = dict(u=u, v=v, lam=lam, S2=x2, S3=x3, S4=x4)
        for sname, x in (("S2", x2), ("S3", x3), ("S4", x4)):
            dseed += 1
            xs = x * (20000 if SMOKE else 1)
            tasks.append(("sim", f"D|{stay}|{sname}", dict(s=s_, g=g_, sigma=0.3 if SMOKE else 0.15, phi=0.9, start=("fixed", xs),
                                                          steps=info[lab]["horizon"], reps=30 if SMOKE else REPS, seed=dseed, deadline=deadline)))
    for law, stays in (("nb", ("none", "10", "40", "160")), ("group", ("10", "40", "160"))):
        for stay in stays:
            lab = head[stay]
            eseed += 1
            tasks.append(("sim", f"E|{law}|{stay}", dict(s=info[lab]["s"], g=info[lab]["g"], sigma=0.3 if SMOKE else 0.15, phi=0.9, law=law,
                                                        start=("r13", 10000 * (20000 if SMOKE else 1)), steps=info[lab]["horizon"],
                                                        reps=30 if SMOKE else REPS, seed=eseed, deadline=deadline)))

    def cost(tk):
        c = tk[2]
        n0 = c["start"][1] if c["start"][0] == "r13" else float(np.sum(c["start"][1]))
        return -(c["steps"] * len(c["s"]) * c["reps"] * (1.0 if n0 >= 1000 else 0.3) * (0.3 if len(c["s"]) == 1 else 1.0)
                 * (0.2 + float(c["s"].max() > 0.99) + 0.5 * float(c["s"].max() > 0.97)))
    tasks.sort(key=cost)
    print(f"\n== 2. Main cases: {len(tasks)} runs, {REPS if not SMOKE else 30} replicates each ==", flush=True)
    res = run_tasks(pool, tasks, "main")
    cpu = sum(r["secs"] for r in res.values())
    anomalies = sum(r["anomalies"] for r in res.values())
    unfinished = [k for k, r in res.items() if r["unfinished"]]

    def merged(label):
        rs = [res[f"base|{label}|{bt}"] for bt in range(info[label]["batches"])]
        return {k: np.concatenate([r[k] for r in rs]) for k in ("ext_at", "first_cap", "ever_norm", "ever_scaled", "peak_ln", "final_ln")}

    ln10 = math.log(10)
    # ---- B: R13 against R13b
    print("\n== B. R13 against R13b, every case (fraction extinct; +- is one binomial standard error) ==")
    print("   'R13 bounds' = R13's extinct fraction to that plus the fraction R13 stopped above 1e15.")
    print("   'equal generations' = within R13's own horizon; 'equal steps' = within 1,020,000 steps (W1 only).")
    summary = {}
    c1_ok = True
    for case in base:
        label = case[0].strip()
        inf = info[label]
        m = merged(label)
        n = m["ext_at"].size
        saved = r13[label]
        pg, seg, kg = frac(m["ext_at"], inf["r13_steps"])
        pe, see, ke = frac(m["ext_at"], inf["horizon"])
        b1 = res[f"base|{label}|0"]
        pg1, seg1, _ = frac(b1["ext_at"], inf["r13_steps"])
        summary[label] = dict(pg=pg, seg=seg, pe=pe, see=see, n=n, pg1=pg1, seg1=seg1, m=m)
        lo, hi = saved["ext"], saved["ext"] + saved["cens"]
        tol_lo = 4 * math.sqrt(se_of(lo, 2000) ** 2 + seg ** 2)
        tol_hi = 4 * math.sqrt(se_of(hi, 2000) ** 2 + seg ** 2)
        inside = (lo - tol_lo <= pg <= hi + tol_hi)
        c1_ok = c1_ok and inside
        extra = f"   (first batch alone, R13's seed: {pg1:.4f} +- {seg1:.4f})" if inf["batches"] > 1 else ""
        eq = f"   equal steps {pe:.4f} +- {see:.4f}" if inf["is_w1"] else ""
        print(f"  {label:44s} R13 {saved['ext']:.4f} +- {saved['se']:.4f}, bounds [{lo:.4f}, {hi:.4f}] | R13b equal generations {pg:.4f} +- {seg:.4f}{eq}"
              f"   n = {n}{'' if inside else '   OUTSIDE R13 BOUNDS'}{extra}")
    # ---- diagnostics on censoring
    print("\n== B2. What R13's stopping rule hid, and what the sampler had to do ==")
    print("   a: extinct within R13's horizon without ever passing 1e15 (what R13 could count);  b: passed 1e15 within R13's horizon (R13 stopped these);")
    print("   c: of b, extinct within R13's horizon;  d: of b, extinct within the whole run;  e: passed 1e15 at any time in the run;  f: of e, extinct by the end.")
    for case in base:
        label = case[0].strip()
        inf = info[label]
        m = summary[label]["m"]
        n = m["ext_at"].size
        ext, cap = m["ext_at"], m["first_cap"]
        h = inf["r13_steps"]
        died_h = (ext > 0) & (ext <= h)
        capped_h = (cap > 0) & (cap <= h)
        a = int((died_h & ~capped_h).sum())
        bq = int(capped_h.sum())
        cq = int((capped_h & died_h).sum())
        dq = int((capped_h & (ext > 0)).sum())
        eq_ = int((cap > 0).sum())
        fq = int(((cap > 0) & (ext > 0)).sum())
        dead = ext > 0
        peak_dead = float(np.max(m["peak_ln"][dead])) / ln10 if dead.any() else float("nan")
        fin = m["final_ln"][~dead]
        fin_med = float(np.median(fin)) / ln10 if fin.size else float("nan")
        fin_max = float(np.max(fin)) / ln10 if fin.size else float("nan")
        fin_min = float(np.min(fin)) / ln10 if fin.size else float("nan")
        print(f"  {label:44s} n {n:5d} | a {a:5d}  b {bq:5d}  c {cq:4d}  d {dq:4d}  e {eq_:5d}  f {fq:4d} | approximated in {int(m['ever_norm'].sum()):5d}, "
              f"scaled in {int(m['ever_scaled'].sum()):5d} | largest size from which a replicate later died 1e{peak_dead:.1f} | "
              f"survivors at the end: log10 size min {fin_min:.1f}, median {fin_med:.1f}, max {fin_max:.1f}")
    # ---- C: extinction by time
    print("\n== C. Extinction by time, W1 cases (fraction extinct by 10%, 25%, 50%, 100% of the equal-steps horizon) ==")
    for case in base:
        label = case[0].strip()
        inf = info[label]
        if not inf["is_w1"]:
            continue
        m = summary[label]["m"]
        cells = []
        for fr in (0.10, 0.25, 0.50, 1.00):
            T = int(round(fr * inf["horizon"]))
            p, se, _ = frac(m["ext_at"], T)
            cells.append(f"by step {T:>9,d}: {p:.4f} +- {se:.4f}")
        dead = m["ext_at"] > 0
        med = float(np.median(m["ext_at"][dead])) if dead.any() else float("nan")
        print(f"  {label:44s} " + " | ".join(cells) + f" | median time of those extinct {med:9.0f} steps")
    # ---- orderings
    def lab(case, stay, n0):
        return f"W1 {case} steady stay {stay:>4}, start {n0:>5}"

    pairs = [("N1", 10000, "10", "none"), ("N1", 10000, "none", "40"), ("N1", 10000, "40", "160"), ("N1", 10000, "none", "160"),
             ("N1", 1000, "10", "none"), ("N1", 1000, "none", "40"), ("N1", 1000, "40", "160"), ("N1", 1000, "none", "160"),
             ("N1", 100, "10", "none"), ("N1", 100, "40", "none"), ("N1", 100, "none", "160"), ("N1", 100, "10", "40"),
             ("N2", 10000, "none", "10"), ("N2", 1000, "none", "10"), ("N2", 100, "none", "10")]
    print("\n== Orderings stated by R13 ('first above second'), judged by the rule of the protocol (4 standard errors) ==")
    for cs, n0, a, b in pairs:
        A, Bq = lab(cs, a, n0), lab(cs, b, n0)
        ra, rb = r13[A], r13[Bq]
        lo, hi = ra["ext"], rb["ext"] + rb["cens"]
        se_r = math.sqrt(se_of(lo, 2000) ** 2 + se_of(hi, 2000) ** 2)
        own = (lo - hi) / se_r if se_r > 0 else 0.0
        sa, sb = summary[A], summary[Bq]
        dg, sdg = sa["pg"] - sb["pg"], math.sqrt(sa["seg"] ** 2 + sb["seg"] ** 2)
        de, sde = sa["pe"] - sb["pe"], math.sqrt(sa["see"] ** 2 + sb["see"] ** 2)
        note = "  [R13 said 'no effect' here]" if (cs == "N2" and n0 == 100) else ""
        print(f"  {cs} from {n0:>5}: stay {a:>4} above {b:>4} | R13's own numbers: {ra['ext']:.3f} against {rb['ext']:.3f}; lower bound less upper bound {lo - hi:+.4f} "
              f"({own:+.1f} se): {'established by R13' if own >= 4 else 'NOT established by R13'}")
        print(f"        R13b equal generations: {sa['pg']:.4f} - {sb['pg']:.4f} = {dg:+.4f} +- {sdg:.4f} ({dg / sdg if sdg else 0:+.1f} se)  {r13_verdict(dg, sdg)}")
        print(f"        R13b equal steps:       {sa['pe']:.4f} - {sb['pe']:.4f} = {de:+.4f} +- {sde:.4f} ({de / sde if sde else 0:+.1f} se)  {r13_verdict(de, sde)}{note}")
    A, Bq = lab("N1", "none", 10000), lab("N1", "40", 1000)
    print(f"  'Numbers needed' (R13: about equal): N1 none from 10,000 against stay 40 from 1,000: equal generations {summary[A]['pg']:.4f} +- {summary[A]['seg']:.4f} "
          f"against {summary[Bq]['pg']:.4f} +- {summary[Bq]['seg']:.4f}; equal steps {summary[A]['pe']:.4f} +- {summary[A]['see']:.4f} against "
          f"{summary[Bq]['pe']:.4f} +- {summary[Bq]['see']:.4f}")
    # ---- D and E
    def headline_rows(get, tag):
        rows = {}
        for stay in ("none", "10", "40", "160"):
            r = get(stay)
            if r is None:
                continue
            h = info[head[stay]]
            pg, seg, _ = frac(r["ext_at"], h["r13_steps"])
            pe, see, _ = frac(r["ext_at"], h["horizon"])
            rows[stay] = (pg, seg, pe, see)
        print(f"  {tag:58s} " + " | ".join(f"stay {st:>4}: {rows[st][0]:.4f} +- {rows[st][1]:.4f} / {rows[st][2]:.4f} +- {rows[st][3]:.4f}" for st in rows))
        return rows

    def headline_orderings(rows, tag):
        for a, b in (("10", "none"), ("none", "40"), ("40", "160"), ("none", "160")):
            cells = []
            for j, name in ((0, "equal generations"), (2, "equal steps")):
                d = rows[a][j] - rows[b][j]
                sd = math.sqrt(rows[a][j + 1] ** 2 + rows[b][j + 1] ** 2)
                cells.append(f"{name} {d:+.4f} +- {sd:.4f} ({d / sd if sd else 0:+.1f} se) {r13_verdict(d, sd)}")
            print(f"        {tag}: stay {a:>4} above {b:>4}: " + "; ".join(cells))

    print("\n== D. Starting conditions, N1 from 10,000 (each cell: equal generations / equal steps) ==")
    for stay in ("10", "40", "160"):
        d = dstarts[stay]
        print(f"  stay {stay:>4}: reproductive value u = (1, {d['u'][1]:.6f}); stable proportions v = ({d['v'][0]:.6f}, {d['v'][1]:.6f}); growth factor per step {d['lam']:.8f}; "
              f"S2 = ({d['S2'][0]:.0f}, {d['S2'][1]:.0f}), S3 = ({d['S3'][0]:.0f}, {d['S3'][1]:.0f}), S4 = ({d['S4'][0]:.0f}, {d['S4'][1]:.0f}), total {d['S4'].sum():.0f}")
    base_get = lambda stay: res[f"base|{head[stay]}|0"]
    rows_s1 = headline_rows(base_get, "S1  R13's own start (stable proportions, 10,000)")
    all_rows = {"S1": rows_s1}
    for sname, text in (("S2", "S2  all in the fluctuating stage, reproductive value 10,000"), ("S3", "S3  all in the steady stage, reproductive value 10,000"),
                        ("S4", "S4  stable proportions, reproductive value 10,000")):
        all_rows[sname] = headline_rows(lambda stay, sname=sname: base_get(stay) if stay == "none" else res[f"D|{stay}|{sname}"], text)
    print("   (the ring with no steady stage has one stage; its row entry is the same run in all four lines)")
    for sname in ("S1", "S2", "S3", "S4"):
        headline_orderings(all_rows[sname], sname)
    print("   differences from S1, same ring (equal generations / equal steps):")
    for sname in ("S2", "S3", "S4"):
        cells = []
        for stay in ("10", "40", "160"):
            r, r1 = all_rows[sname][stay], rows_s1[stay]
            dg, sdg = r[0] - r1[0], math.sqrt(r[1] ** 2 + r1[1] ** 2)
            de, sde = r[2] - r1[2], math.sqrt(r[3] ** 2 + r1[3] ** 2)
            cells.append(f"stay {stay:>4}: {dg:+.4f} +- {sdg:.4f} ({judge(dg, sdg)[1]}) / {de:+.4f} +- {sde:.4f} ({judge(de, sde)[1]})")
        print(f"        {sname} - S1: " + " | ".join(cells))
    print("\n== E. Offspring law, N1 from 10,000, start S1 (each cell: equal generations / equal steps) ==")
    rows_p = headline_rows(base_get, "Poisson offspring, independent individuals (R13's law)")
    rows_nb = headline_rows(lambda stay: res[f"E|nb|{stay}"], "E-i  clumped: negative binomial, k = 0.1, fluctuating transfer")
    rows_gr = headline_rows(lambda stay: base_get(stay) if stay == "none" else res[f"E|group|{stay}"], "E-ii shared host fate: groups of 100 in the steady stage")
    print("   (E-ii does not change the ring with no steady stage; its entry there is the Poisson run)")
    headline_orderings(rows_p, "Poisson")
    headline_orderings(rows_nb, "E-i   ")
    headline_orderings(rows_gr, "E-ii  ")
    print("   differences from the Poisson law, same ring (equal generations / equal steps):")
    for nm, rows in (("E-i ", rows_nb), ("E-ii", rows_gr)):
        cells = []
        for stay in ("none", "10", "40", "160"):
            if nm == "E-ii" and stay == "none":
                continue
            r, r1 = rows[stay], rows_p[stay]
            dg, sdg = r[0] - r1[0], math.sqrt(r[1] ** 2 + r1[1] ** 2)
            de, sde = r[2] - r1[2], math.sqrt(r[3] ** 2 + r1[3] ** 2)
            cells.append(f"stay {stay:>4}: {dg:+.4f} +- {sdg:.4f} ({judge(dg, sdg)[1]}) / {de:+.4f} +- {sde:.4f} ({judge(de, sde)[1]})")
        print(f"        {nm} - Poisson: " + " | ".join(cells))
    for key in sorted(k for k in res if k.startswith(("D|", "E|"))):
        r = res[key]
        print(f"   {key:14s} approximated in {int(r['ever_norm'].sum()):5d}, scaled in {int(r['ever_scaled'].sum()):5d}, passed 1e15: {int((r['first_cap'] > 0).sum()):5d}, "
              f"of which later extinct {int(((r['first_cap'] > 0) & (r['ext_at'] > 0)).sum()):4d}   [{r['secs']:.0f} s]")
    # ---- checks on the main run
    print("\n== Checks on the main run ==")
    ok = True
    below = summary[[c[0].strip() for c in base if c[0].startswith("W0 below")][0]]
    above = summary[[c[0].strip() for c in base if c[0].startswith("W0 above")][0]]
    w0_ok = below["pg"] >= 0.99 and above["pg"] <= 0.90 and (1 - above["pg"]) >= 0.10
    print(f"[{'PASS' if w0_ok else 'FAIL'}] W0 the threshold is where R0* says (R13's criterion, no replicate stopped): extinct {below['pg']:.4f} below, "
          f"{above['pg']:.4f} above (not extinct {1 - above['pg']:.4f})")
    print(f"[{'PASS' if c1_ok else 'FAIL'}] C1 every R13b fraction at R13's horizon lies within R13's bounds (allowance 4 standard errors)")
    print(f"[{'PASS' if anomalies == 0 else 'FAIL'}] no scaled replicate ever had a stage below 1e30 (anomalies: {anomalies})")
    print(f"[{'PASS' if not unfinished else 'FAIL'}] every case finished within the budget (unfinished: {len(unfinished)}{': ' + ', '.join(unfinished) if unfinished else ''})")
    # H0c: shared stream
    appl, good = 0, 0
    for case in base:
        label = case[0].strip()
        inf = info[label]
        r = res[f"base|{label}|0"]
        tf = r["t_first_norm"]
        if tf < 0 or tf > inf["r13_steps"]:
            appl += 1
            _, _, k = frac(r["ext_at"], inf["r13_steps"])
            same = (k == r13[label]["n_ext"])
            good += same
            print(f"      H0c {label:44s} no draw reached N_big within R13's horizon: R13b extinct {k}, R13 extinct {r13[label]['n_ext']}  {'same' if same else 'DIFFERENT'}")
        else:
            print(f"      H0c {label:44s} first approximated draw at step {tf} (R13's horizon {inf['r13_steps']}): not applicable")
    if SMOKE:
        print("[SMOKE] H0c not meaningful for toy rings")
    elif appl == 0:
        print("[VOID] H0c no case stayed below N_big within R13's horizon; the shared-stream check could not be made")
    else:
        print(f"[{'PASS' if good == appl else 'FAIL'}] H0c where no draw reached N_big within R13's horizon, R13b's count equals R13's exactly ({good} of {appl} cases)")
        ok = ok and good == appl
    ok = ok and w0_ok and c1_ok and anomalies == 0 and not unfinished
    print(f"   computing time summed over the main runs {cpu:.0f} s; replicates that needed scaling against overflow: "
          f"{sum(int(r['ever_scaled'].sum()) for r in res.values())} of {sum(r['ext_at'].size for r in res.values())}")
    return ok


def main():
    t0 = time.time()
    fmt = "%Y-%m-%d %H:%M:%S %Z"
    print(f"R13b{' SMOKE (toy rings, not results)' if SMOKE else ''}: R13's cases with no replicate stopped early")
    print(f"protocol file last modified {time.strftime(fmt, time.localtime(os.path.getmtime(PROTOCOL)))}; run started {time.strftime(fmt, time.localtime(t0))} "
          f"(both from the system clock); protocol older than the run: {os.path.getmtime(PROTOCOL) < t0}")
    print(f"numpy {np.__version__}; {NPROC} processes; N_big = {N_BIG:.0e}; overflow guard at {SCALE_HI:.0e} by {SCALE_STEP:.0e}; equal-steps horizon {T_EQ:,d} steps")
    r13 = parse_r13()
    print(f"R13's numbers read from {R13_OUTPUT.name}: {len(r13)} cases")
    with mp.Pool(NPROC) as pool:
        ok_method = method_checks(pool, r13)
        if not ok_method and not SMOKE:
            print("\nA check of the method did not pass. The main cases were NOT run.")
            print(f"total run time {time.time() - t0:.0f} s")
            return 1
        ok_main = main_cases(pool, r13, t0 + BUDGET_S)
    print(f"total run time {time.time() - t0:.0f} s")
    return 0 if (ok_method and ok_main) else 1


if __name__ == "__main__":
    sys.exit(main())
