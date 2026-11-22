"""Run R1f: a steady stage near the threshold, with and without host memory.

Claim tested: checks C1 to C3 of `protocols/R1f-parking-near-threshold.md`, fixed there before this script was first
run, and its open question, which is reported without a criterion. Exit status is 0 only if C1 to C3 pass.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


pk = load("pk", "parking_stage_formulas.py")
r1e = load("r1e", "r1e_parking_stage.py")
r1c = load("r1c", "r1c_large_noise.py")

X0 = math.log(10.0)
GENERATIONS = 6000
SIGMA, A = 0.15, 10.0


def main():
    t0 = time.time()
    out, checks = [], []
    groups = {}
    for label, phi, ratio, bs in (("N1", 0.9, 0.2, (None, 10.0, 40.0, 160.0)), ("N2", 0.0, 0.06, (None, 10.0))):
        th = phi / (1 - phi); ln_r0 = ratio * SIGMA ** 2
        print(f"{label}: phi = {phi}, sigma = {SIGMA}, ln R0 = {ln_r0:.5f}")
        rows = []
        for idx, b in enumerate(bs):
            stays = [A] if b is None else [A, b]
            s, g, lam0 = r1e.rates_for(stays, ln_r0)
            T = float(sum(stays)); steps = int(round(GENERATIONS * T))
            sig = [SIGMA] + [0.0] * (len(stays) - 1)
            p, p_se, mu, v, mu_se = r1e.extinction(s, g, sig, phi, 12000 + 100 * idx + int(phi * 10), 2000, steps, 1000, X0)
            gam = (pk.gamma1(A) if b is None else pk.gamma2(A, b, phi)) * SIGMA ** 2
            tv = SIGMA ** 2 * (1 + 2 * th) / T
            E = 2 * (ln_r0 - gam) / tv
            pred = r1c.lande_orzack(ln_r0 - gam, tv, X0, GENERATIONS)
            row = dict(case=label, steady=b, T=T, p=p, p_se=p_se, growth=mu * T, growth_se=mu_se * T, var=v * T, gamma=gam, drift=ln_r0 - gam, tv=tv, index=E, predicted=pred)
            rows.append(row); out.append(row)
            ok1 = bool(abs(row["growth"] - row["drift"]) <= max(3.5 * row["growth_se"], 0.10 * gam))
            checks.append((f"C1 ({label}, steady stay {'none' if b is None else f'{b:g}'})", ok1,
                           f"growth per generation {row['growth']:+.5f} +- {row['growth_se']:.5f} against {row['drift']:+.5f} (Gamma {gam:.5f}); variance per generation {row['var']:.5f} against {tv:.5f}"))
            print(f"    steady stay {'none' if b is None else f'{b:g}'}: fell below the level {p:.4f} +- {p_se:.4f} (diffusion formula {pred:.3f}; index E {E:.3f}); "
                  f"growth per generation {row['growth']:+.5f} +- {row['growth_se']:.5f} (predicted {row['drift']:+.5f})", flush=True)
        groups[label] = rows
    n1, n2 = groups["N1"], groups["N2"]
    d = n1[0]["p"] - n1[3]["p"]; dse = math.hypot(n1[0]["p_se"], n1[3]["p_se"])
    checks.append(("C2", bool(d > 2 * dse), f"N1: {n1[0]['p']:.3f} with no steady stage, {n1[3]['p']:.3f} with a stay of 160 (difference {d:+.3f} +- {dse:.3f})"))
    d = n2[0]["p"] - n2[1]["p"]; dse = math.hypot(n2[0]["p_se"], n2[1]["p_se"])
    checks.append(("C3", bool(d > 2 * dse), f"N2: {n2[0]['p']:.3f} with no steady stage, {n2[1]['p']:.3f} with a stay of 10 (difference {d:+.3f} +- {dse:.3f})"))
    for name, ok, txt in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {txt}")
    d = n1[1]["p"] - n1[0]["p"]; dse = math.hypot(n1[0]["p_se"], n1[1]["p_se"])
    print(f"  OPEN QUESTION (N1, no criterion): with a steady stay of 10 the probability is {n1[1]['p']:.3f} +- {n1[1]['p_se']:.3f}, against {n1[0]['p']:.3f} +- {n1[0]['p_se']:.3f} with none: "
          f"a change of {d:+.3f} +- {dse:.3f} ({d / dse:+.1f} standard errors). The index predicted a rise.")
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R1f PASSED" if n_fail == 0 else "R1f FAILED")
    (HERE.parent / "results" / "R1f-parking-near-threshold_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
