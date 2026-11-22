"""Run R1e: a parking stage at a fixed R0.

Claim tested: checks K0 to K3 of `protocols/R1e-parking-stage.md`, fixed there before this script was first run.
Exit status is 0 only if all pass.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy import optimize

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE.parent))
from engine import environments, theory  # noqa: E402

spec = importlib.util.spec_from_file_location("r1c", HERE / "r1c_large_noise.py")
r1c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1c)

X0 = math.log(10.0)
GENERATIONS = 1500
TAU1 = 8.0 / 3.0


def rates_for(stays, ln_r0):
    m = 1 - 1 / np.asarray(stays, float)
    f = lambda lam: float(np.sum(np.log((1 - m) * lam / (1 - m * lam)))) - ln_r0
    lam0 = optimize.brentq(f, 0.5, 1 / m.max() - 1e-12)
    return m * lam0, (1 - m) * lam0, lam0


def extinction(s, g, sig, phi, seed, reps, steps, burn, depth):
    """As r1c.extinction, for any ring; sig[k] = 0 for a transfer that does not fluctuate."""
    n = len(s)
    rng = np.random.default_rng(seed)
    env = environments.FluctuatingRing(list(s), list(g), list(sig), [phi] * n, [0.0] * n, [0.0] * n, reps, rng)
    x = np.full((reps, n), 1.0 / n)
    cum = np.zeros(reps); low = np.zeros(reps)
    for t in range(burn + steps):
        s_t, g_t, cv = env.step()
        x = x * s_t + np.roll(x * g_t, 1, axis=1)
        tot = x.sum(axis=1)
        x /= tot[:, None]
        if t >= burn:
            cum += np.log(tot); low = np.minimum(low, cum)
    ext = low <= -depth
    p = float(ext.mean()); p_se = math.sqrt(max(p * (1 - p), 1e-12) / reps)
    return p, p_se, float(cum.mean() / steps), float(cum.var(ddof=1) / steps), float(cum.std(ddof=1) / math.sqrt(reps) / steps)


def main():
    t0 = time.time()
    out = []; checks = []
    r1d = {"K1": (0.410, 0.011), "K2": (0.221, 0.009)}
    for label, phi, sigma, ln_r0 in (("K1", 0.0, 0.25, 0.01572), ("K2", 0.9, 0.12, 0.03)):
        theta = phi / (1 - phi)
        print(f"{label}: phi = {phi}, sigma = {sigma}, ln R0 = {ln_r0}")
        rows = []
        for idx, park in enumerate((None, 4.0 / 3.0, 8.0 / 3.0, 8.0)):
            stays = [TAU1] if park is None else [TAU1, park]
            s, g, lam0 = rates_for(stays, ln_r0)
            T = float(sum(stays)); steps = int(round(GENERATIONS * T))
            sig = [sigma] + [0.0] * (len(stays) - 1)
            p, p_se, mu, v, mu_se = extinction(s, g, sig, phi, 9000 + 100 * idx + int(phi * 10), 2000, steps, 1000, X0)
            m = 1 - 1 / np.asarray(stays)
            sig2 = np.array([sigma ** 2] + [0.0] * (len(stays) - 1))
            gam = float(theory.per_generation_cost_transfer(m, 1 - m, sig2, phi).sum())
            tv = sigma ** 2 * (1 + 2 * theta) / T
            E = 2 * (ln_r0 - gam) / tv
            pred = r1c.lande_orzack(ln_r0 - gam, tv, X0, GENERATIONS)
            rows.append(dict(test=label, parking=park, T=T, lam0=lam0, p=p, p_se=p_se, growth_per_generation=mu * T, growth_se=mu_se * T, var_per_generation=v * T, gamma=gam, tv=tv, index=E, predicted=pred))
            print(f"    parking {'none' if park is None else f'{park:.3f}'}: extinct {p:.4f} +- {p_se:.4f}; growth per generation {mu * T:+.5f} +- {mu_se * T:.5f} (theory {ln_r0 - gam:+.5f}, Gamma {gam:.5f}); "
                  f"variance per generation {v * T:.5f} (theory {tv:.5f}); index E {E:.2f}; diffusion formula {pred:.4f}", flush=True)
        out += rows
        ref, ref_se = r1d[label]
        z = (rows[0]["p"] - ref) / math.hypot(rows[0]["p_se"], ref_se)
        checks.append((f"K0 ({label})", abs(z) < 3, f"one stage: {rows[0]['p']:.3f} +- {rows[0]['p_se']:.3f} against {ref:.3f} +- {ref_se:.3f} in run R1d (z = {z:+.1f})"))
        falls = all(rows[i]["p"] - rows[i + 1]["p"] > 2 * math.hypot(rows[i]["p_se"], rows[i + 1]["p_se"]) for i in range(3))
        ok = falls and rows[3]["p"] < 0.03
        checks.append((label, ok, f"extinction {', '.join(f'{r['p']:.3f}' for r in rows)} for parking none, 4/3, 8/3, 8; each fall by more than 2 s.e.: {falls}"))
        if label == "K2":
            d = rows[0]["growth_per_generation"] - rows[2]["growth_per_generation"]; dse = math.hypot(rows[0]["growth_se"], rows[2]["growth_se"])
            ok3 = d > 2 * dse and rows[2]["p"] < rows[0]["p"]
            checks.append(("K3", ok3, f"growth per generation {rows[0]['growth_per_generation']:+.5f} with no parking and {rows[2]['growth_per_generation']:+.5f} with a parking stay of 8/3 (difference {d:+.5f} +- {dse:.5f}), extinction {rows[0]['p']:.3f} and {rows[2]['p']:.3f}"))
    for name, ok, txt in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {txt}")
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R1e PASSED" if n_fail == 0 else "R1e FAILED")
    (HERE.parent / "results" / "R1e-parking-stage_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
