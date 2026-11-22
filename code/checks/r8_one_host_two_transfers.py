"""Run R8: one host series governing two transfers of a ring.

Claim tested: checks U1 to U6 of `protocols/R8-one-host-two-transfers.md`, fixed there before this script was first
run. Exit status is 0 only if all pass.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec = importlib.util.spec_from_file_location("w", HERE / "one_host_two_transfers.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

SIGMA, REPS, STEPS, BURN = 0.03, 128, 500_000, 2_000


def simulate(stays, k1, k2, phi, seed):
    """Cost per step in units of sigma^2 and its standard error; transfers out of stages 0 and 1 share one series."""
    tau = np.asarray(stays, float); n = len(tau); s = 1 - 1 / tau; g = 1 / tau; c = 1 / tau.sum()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(REPS); sd = math.sqrt(1 - phi * phi)
    x = np.full((REPS, n), 1.0 / n)
    cum = np.zeros(REPS); zsum = np.zeros(REPS)
    gg = np.tile(g, (REPS, 1))
    for t in range(BURN + STEPS):
        z = phi * z + sd * rng.standard_normal(REPS)
        gt = gg.copy()
        gt[:, 0] *= np.exp(k1 * SIGMA * z - (k1 * SIGMA) ** 2 / 2)
        gt[:, 1] *= np.exp(k2 * SIGMA * z - (k2 * SIGMA) ** 2 / 2)
        x = x * s + np.roll(x * gt, 1, axis=1)
        tot = x.sum(axis=1); x /= tot[:, None]
        if t >= BURN:
            cum += np.log(tot); zsum += z
    growth = cum / STEPS - (k1 + k2) * c * SIGMA * zsum / STEPS
    cost = -growth / SIGMA ** 2
    return float(cost.mean()), float(cost.std(ddof=1) / math.sqrt(REPS))


def main():
    t0 = time.time()
    cases = [("U1", [4.0, 4.0], 1, 1, 0.0, 0.031250), ("U2", [30.0, 1.0], 1, 1, 0.9, -0.023408), ("U3", [4.0, 4.0], 1, 1, 0.9, 0.031250),
             ("U4", [4.0, 4.0], 1, -1, 0.9, 0.102273), ("U5", [6.0, 2.0, 2.0], 1, 1, 0.7, 0.020649), ("U6", [4.0, 4.0], 1, -1, 0.0, 0.0)]
    out, checks = [], []
    for idx, (label, stays, k1, k2, phi, written) in enumerate(cases):
        pred = w.closed_form(stays, 0, 1, k1, k2, phi)
        assert abs(pred - written) < 6e-7, (label, pred, written)          # the prediction printed in the protocol
        est, se = simulate(stays, k1, k2, phi, 41000 + idx)
        if label == "U6":
            ok = abs(est) <= max(3.5 * se, 0.002)
            txt = f"simulated {est:+.6f} +- {se:.6f}, predicted 0"
        else:
            ok = abs(est - pred) <= max(3.5 * se, 0.015 * abs(pred))
            if label == "U2":
                ok = ok and est < -3.5 * se
            txt = f"simulated {est:+.6f} +- {se:.6f}, predicted {pred:+.6f} (z = {(est - pred) / se:+.1f}; {100 * (est / pred - 1):+.2f}%)"
        out.append(dict(case=label, stays=stays, kappa=[k1, k2], phi=phi, predicted=pred, simulated=est, se=se))
        ok = bool(ok)          # a numpy bool cannot be written to JSON (the first run printed all results and then failed on saving)
        checks.append((label, ok, txt)); print(f"  [{'PASS' if ok else 'FAIL'}] {label} stays {stays} kappa ({k1}, {k2}) phi {phi}: {txt}", flush=True)
    n_fail = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    print("R8 PASSED" if n_fail == 0 else "R8 FAILED")
    (HERE.parent / "results" / "R8-one-host-two-transfers_20261114.json").write_text(json.dumps({"runs": out, "checks": [list(c) for c in checks]}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
