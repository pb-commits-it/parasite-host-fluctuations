"""Run R2/R4: the closed forms for unequal stages and fluctuating survival against simulation.

Claim tested: hypotheses H1 to H7 of `protocols/R2-R4-general-ring.md`, with the cases, settings and pass
criteria fixed there before this script was first run. H8 (large noise) is reported without a criterion.
Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import environments, ring, theory  # noqa: E402

REPS, STEPS = 400, 120000
LAM = 1.05
rows = []
checks = []


def from_stays(tau):
    tau = np.asarray(tau, float)
    return LAM * (1 - 1 / tau), LAM / tau


def predict(s, g, sig_g, phi_g, sig_s, phi_s):
    """Closed-form cost for unit sigma (sig_g, sig_s are relative sizes) and the matrix formula as a cross-check."""
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    ct = theory.cost_transfer(s, g, np.asarray(sig_g, float) ** 2, phi_g)
    cs = theory.cost_survival(s, g, np.asarray(sig_s, float) ** 2, phi_s)
    closed = float(ct.sum() + cs.sum())
    entries, drivers, kappa, phis = [], [], [], []
    for k in range(n):
        if sig_g[k] > 0:
            entries.append(((k + 1) % n, k)); kappa.append(sig_g[k]); phis.append(phi_g[k]); drivers.append(len(phis) - 1)
        if sig_s[k] > 0 and s[k] > 0:
            entries.append((k, k)); kappa.append(sig_s[k]); phis.append(phi_s[k]); drivers.append(len(phis) - 1)
    a = theory.ring_matrix(s, g)
    if n == 1:
        return closed, closed
    # a transfer and a staying entry coincide only when n = 1, which is excluded above
    mat = theory.matrix_cost(a, entries, drivers, kappa, phis)
    return closed, mat


def simulate(name, s, g, sig_g, phi_g, sig_s, phi_s, sigma, seed, burn=2000):
    s, g = np.asarray(s, float), np.asarray(g, float)
    n = len(s)
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing(s, g, sigma * np.asarray(sig_g, float), phi_g, sigma * np.asarray(sig_s, float), phi_s, REPS, rng)
    out = ring.run(env, n, REPS, burn + STEPS, burn)
    m, se = ring.mean_se(out["rate_cv"])
    return math.log(env.lam0) - m, se


def case(name, hyp, s, g, sig_g=None, phi_g=None, sig_s=None, phi_s=None, burn=2000, seed=0, scale=None, sigmas=(0.05, 0.025)):
    n = len(s)
    z = [0.0] * n
    sig_g = z if sig_g is None else list(np.broadcast_to(np.asarray(sig_g, float), (n,)))
    sig_s = z if sig_s is None else list(np.broadcast_to(np.asarray(sig_s, float), (n,)))
    phi_g = z if phi_g is None else list(np.broadcast_to(np.asarray(phi_g, float), (n,)))
    phi_s = z if phi_s is None else list(np.broadcast_to(np.asarray(phi_s, float), (n,)))
    closed, mat = predict(s, g, sig_g, phi_g, sig_s, phi_s)
    res = {"case": name, "hyp": hyp, "closed_per_sigma2": closed, "matrix_per_sigma2": mat, "runs": {}}
    for i, sigma in enumerate(sigmas):
        cost, se = simulate(name, s, g, sig_g, phi_g, sig_s, phi_s, sigma, seed * 10 + i, burn)
        res["runs"][str(sigma)] = {"cost": cost, "se": se, "pred": closed * sigma ** 2}
    rows.append(res)
    r = res["runs"]
    if "0.025" in r:
        ref = abs(scale) if scale is not None else abs(closed)
        dep = r["0.025"]["cost"] - r["0.025"]["pred"]
        ok = abs(dep) <= max(3.5 * r["0.025"]["se"], 0.01 * ref * 0.025 ** 2)
        dep05 = r["0.05"]["cost"] - r["0.05"]["pred"]
        ratio = dep05 / dep if dep != 0 else float("nan")
        res["pass"] = bool(ok)
        txt = (f"closed form {closed:+.6f} (matrix {mat:+.6f}); at sigma 0.025: sim/sigma^2 {r['0.025']['cost'] / 0.025 ** 2:+.6f} "
               f"+- {r['0.025']['se'] / 0.025 ** 2:.6f}; at 0.05: {r['0.05']['cost'] / 0.05 ** 2:+.6f} +- {r['0.05']['se'] / 0.05 ** 2:.6f}; "
               f"departure ratio {ratio:+.1f}")
        checks.append((hyp, name, ok, txt))
        print(f"  [{'PASS' if ok else 'FAIL'}] {hyp} {name}: {txt}", flush=True)
    return res


def main():
    t0 = time.time()
    a1 = ([0.3, 0.85, 0.5], [0.6, 0.2, 0.9])
    print("H1  unequal stages, every transfer fluctuating")
    case("A1", "H1", *a1, sig_g=1.0, phi_g=0.7, seed=101)
    case("A2", "H1", [0.9, 0.0, 0.4, 0.0, 0.7, 0.2], [0.3, 1.2, 0.5, 0.9, 0.25, 0.8], sig_g=1.0, phi_g=0.8, seed=102)
    case("A3", "H1", [0.95, 0.1], [0.08, 1.5], sig_g=1.0, phi_g=0.9, seed=103)
    print("H2  a different sigma and phi for each host")
    case("B1", "H2", *a1, sig_g=[1.0, 0.5, 2.0], phi_g=[0.0, 0.5, 0.9], seed=104)
    print("H3  position of one fluctuating host")
    cring = ([0.2, 0.9, 0.5, 0.7], [0.9, 0.15, 0.6, 0.4])
    cres = []
    for k in range(4):
        sg = [0.0] * 4; sg[k] = 1.0
        cres.append(case(f"C{k + 1}", "H3", *cring, sig_g=sg, phi_g=0.8, seed=110 + k))
    costs = np.array([c["runs"]["0.025"]["cost"] for c in cres]); ses = np.array([c["runs"]["0.025"]["se"] for c in cres])
    i, j = int(np.argmax(costs)), int(np.argmin(costs))
    diff = costs[i] - costs[j]; sed = math.hypot(ses[i], ses[j])
    ok = diff <= max(3.5 * sed, 0.01 * abs(cres[0]["closed_per_sigma2"]) * 0.025 ** 2)
    txt = f"largest difference between positions at sigma 0.025: {diff / 0.025 ** 2:.6f} +- {sed / 0.025 ** 2:.6f} per sigma^2 (closed form {cres[0]['closed_per_sigma2']:.6f})"
    checks.append(("H3", "C positions agree", ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] H3 positions agree: {txt}")
    print("H4  same number of stages and generation time")
    dres = [case(nm, "H4", *from_stays(tau), sig_g=1.0, phi_g=0.9, seed=120 + q)
            for q, (nm, tau) in enumerate((("D1", [2, 2, 2, 2]), ("D2", [1.2, 1.6, 2.2, 3.0]), ("D3", [1, 1, 1, 5])))]
    for x, y in ((0, 1), (1, 2)):
        d = dres[x]["runs"]["0.025"]["cost"] - dres[y]["runs"]["0.025"]["cost"]
        sed = math.hypot(dres[x]["runs"]["0.025"]["se"], dres[y]["runs"]["0.025"]["se"])
        ok = d > 3.5 * sed
        txt = f"{dres[x]['case']} - {dres[y]['case']} = {d / 0.025 ** 2:+.6f} +- {sed / 0.025 ** 2:.6f} per sigma^2 (closed forms differ by {dres[x]['closed_per_sigma2'] - dres[y]['closed_per_sigma2']:+.6f})"
        checks.append(("H4", "order", ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] H4 order: {txt}")
    print("H5  fluctuating survival")
    e_s, e_g = from_stays([10, 1.25])
    e2_closed = float(theory.cost_survival(e_s, e_g, [1.0, 0.0], 0.0).sum())
    e1 = case("E1", "H5", e_s, e_g, sig_s=[1.0, 0.0], phi_s=0.8, seed=130)
    e2 = case("E2", "H5", e_s, e_g, sig_s=[1.0, 0.0], phi_s=0.0, seed=131)
    psi0 = brentq(lambda p: theory.cost_survival(e_s, e_g, [1.0, 0.0], p)[0], 1e-6, 0.8)
    print(f"      boundary for E3: psi = {psi0:.6f}")
    case("E3", "H5", e_s, e_g, sig_s=[1.0, 0.0], phi_s=psi0, seed=132, scale=e2_closed)
    e4 = case("E4", "H5", *a1, sig_s=[0.0, 1.0, 0.0], phi_s=0.7, seed=133)
    for res, sign in ((e1, -1), (e4, -1), (e2, +1)):
        r = res["runs"]["0.025"]
        ok = sign * r["cost"] > 3.5 * r["se"]
        txt = f"{res['case']}: simulated cost at sigma 0.025 is {r['cost'] / r['se']:+.1f} standard errors from zero"
        checks.append(("H5", "sign", ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] H5 sign: {txt}")
    print("H6  survival and transfer together")
    case("E5", "H6", *a1, sig_g=1.0, phi_g=0.7, sig_s=1.0, phi_s=0.5, seed=134)
    print("H7  slow environment")
    f1 = case("F1", "H7", *from_stays([2, 2, 2, 2]), sig_g=1.0, phi_g=0.98, burn=5000, seed=140)
    tau = np.array([2.0, 2, 2, 2]); T = tau.sum(); cv2 = np.sum(tau * (tau - 1)) / T ** 2
    print(f"      limit n (1 - CV^2)/(2T) = {4 * (1 - cv2) / (2 * T):.6f}; closed form at phi 0.98 = {f1['closed_per_sigma2']:.6f}")
    print("H8  larger noise (reported, no criterion)")
    big = []
    for nm, args, kw in (("A1", a1, dict(sig_g=1.0, phi_g=0.7)), ("D1", from_stays([2, 2, 2, 2]), dict(sig_g=1.0, phi_g=0.9)),
                         ("E1", (e_s, e_g), dict(sig_s=[1.0, 0.0], phi_s=0.8))):
        n = len(args[0]); z = [0.0] * n
        sg = list(np.broadcast_to(np.asarray(kw.get("sig_g", 0.0), float), (n,))); ss = list(np.broadcast_to(np.asarray(kw.get("sig_s", 0.0), float), (n,)))
        pg = list(np.broadcast_to(np.asarray(kw.get("phi_g", 0.0), float), (n,))); ps = list(np.broadcast_to(np.asarray(kw.get("phi_s", 0.0), float), (n,)))
        closed, _ = predict(*args, sg, pg, ss, ps)
        for i, sigma in enumerate((0.1, 0.2, 0.4)):
            cost, se = simulate(nm, *args, sg, pg, ss, ps, sigma, 900 + 10 * len(big) + i)
            big.append({"case": nm, "sigma": sigma, "cost": cost, "se": se, "pred": closed * sigma ** 2})
            print(f"      {nm} sigma {sigma}: simulated / closed form = {cost / (closed * sigma ** 2):.4f} +- {se / abs(closed * sigma ** 2):.4f}", flush=True)
    n_fail = sum(1 for c in checks if not c[2])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R2/R4 PASSED" if n_fail == 0 else "R2/R4 FAILED")
    out = (__import__('pathlib').Path(__file__).resolve().parent / '../results/R2-R4-general-ring_20261114.json').resolve()
    out.write_text(json.dumps({"cases": rows, "large_noise": big, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
