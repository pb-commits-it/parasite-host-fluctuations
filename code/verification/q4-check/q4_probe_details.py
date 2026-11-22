#!/usr/bin/env python3
"""Supplemental deep-dive probe for Task 2(b), 2(c), and 2(d). Zero external imports from q4/world."""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4_check_run import (
    HERE, Q4_DIR, load_raw_records, stage_means, solve_replay_D,
    full_cov_prediction_spectral, ar1_ring_coeff, circular_autocov,
)


def main() -> None:
    records = load_raw_records()
    by_id = {r["series_id"]: r for r in records}
    with (Q4_DIR / "cells.csv").open() as f:
        saved_cells = {(r["series_id"], int(r["stages"]), int(r["stay"])): r for r in csv.DictReader(f)}
    with (Q4_DIR / "cell-attribution.csv").open() as f:
        saved_cell_attr = {(r["series_id"], int(r["stages"]), int(r["stay"])): r for r in csv.DictReader(f)}
    with (Q4_DIR / "monotonicity-attribution.csv").open() as f:
        saved_mono_attr = list(csv.DictReader(f))

    # 1. Deep check of ALL 211 H1-full departures at eps = 0.1, +-0.1, 0.01, 0.001
    # Why did 94 of 211 fail primary's one-sided eps=0.1 check?
    h1_rows = []
    for (sid, k, s), sc in saved_cells.items():
        if sc["H1_full"] != "departure":
            continue
        r = by_id[sid]
        h = r["h"]
        z = np.log(h)
        zc = z - float(z.mean())
        taus = stage_means(k, s)
        d_act = float(sc["D"])
        d2 = float(sc["prediction_full"])
        lim = max(0.001, 0.05 * abs(d2))

        vals = {}
        for eps in (0.1, -0.1, 0.01, -0.01, 0.001):
            he = np.exp(eps * zc)
            he /= float(he.mean())
            vals[eps] = solve_replay_D(k, s, he)[0]

        err_01 = abs(vals[0.1] / 0.01 - d2)
        err_sym01 = abs(0.5 * (vals[0.1] + vals[-0.1]) / 0.01 - d2)
        err_001 = abs(vals[0.01] / 1e-4 - d2)
        err_sym001 = abs(0.5 * (vals[0.01] + vals[-0.01]) / 1e-4 - d2)
        err_0001 = abs(vals[0.001] / 1e-6 - d2)

        # Decompose why D_actual differs from D2_full:
        # Compare phase-randomized mean D (preserves exact log spectrum/D2_full, destroys phase/skew/higher cumulants!)
        ca = saved_cell_attr[(sid, k, s)]
        phase_mean_D = float(ca["phase_mean_D"])
        phase_full_err = float(ca["phase_full_error"])  # mean |D(phase_j) - D2_full| across 8 Gaussianized phase draws!
        tol_h1 = float(sc["tolerance_full"])

        h1_rows.append({
            "series_id": sid,
            "stages": k,
            "stay": s,
            "sigma": float(np.std(zc)),
            "skew": float(stats.skew(zc)),
            "kurt": float(stats.kurtosis(zc)),
            "D": d_act,
            "D2_full": d2,
            "tol_h1": tol_h1,
            "err_actual": abs(d_act - d2),
            "phase_mean_D": phase_mean_D,
            "phase_full_err": phase_full_err,
            "phase_repairs_H1": phase_full_err <= tol_h1,
            "phase_material": ca["phase_material"] == "True",
            "primary_01_pass": err_01 <= lim,
            "sym_01_pass": err_sym01 <= lim,
            "one_001_pass": err_001 <= lim,
            "sym_001_pass": err_sym001 <= lim,
            "one_0001_pass": err_0001 <= lim,
            "max_0001_rel": err_0001 / max(1e-12, abs(d2)),
        })

    print("=== H1-full (211 cells) small-amplitude convergence ===")
    for key in ("primary_01_pass", "sym_01_pass", "one_001_pass", "sym_001_pass", "one_0001_pass"):
        print(f"  {key}: {sum(x[key] for x in h1_rows)} / {len(h1_rows)}")
    print("  max |D(0.001)/1e-6 - D2_full| vs lim:", max(abs(x["max_0001_rel"]) for x in h1_rows))

    # Cross-tabulate: does Gaussianized Fourier phase randomization (which preserves log variance & ALL circular autocovariances,
    # while making z approximately Gaussian and destroying log-skew/kurtosis/phase coupling) bring |D - D2_full| within H1 tolerance?
    ct_phase = collections.Counter(
        (x["primary_01_pass"], x["phase_material"], x["phase_repairs_H1"])
        for x in h1_rows
    )
    print("=== H1-full cross-tab (primary_01_pass, phase_material, phase_repairs_H1) ===")
    for k, v in sorted(ct_phase.items()):
        print(f"  {k}: {v}")

    # Check sigma distribution in each group!
    by_01 = {True: [x for x in h1_rows if x["primary_01_pass"]], False: [x for x in h1_rows if not x["primary_01_pass"]]}
    for flag, grp in by_01.items():
        sigs = [x["sigma"] for x in grp]
        skews = [abs(x["skew"]) for x in grp]
        kurts = [x["kurt"] for x in grp]
        print(f"  primary_01_pass={flag} (n={len(grp)}): median sigma={np.median(sigs):.3f} (min={min(sigs):.3f}, max={max(sigs):.3f}), "
              f"median |skew|={np.median(skews):.3f}, median kurt={np.median(kurts):.3f}")

    # Check if ANY H1-full departure has small sigma (< 0.3)
    small_sig = [x for x in h1_rows if x["sigma"] < 0.35]
    print(f"  H1-full departures with sigma < 0.35: {len(small_sig)}")
    for x in small_sig:
        print("   ", x["series_id"], x["stages"], x["stay"], f"sigma={x['sigma']:.3f}, skew={x['skew']:.2f}, kurt={x['kurt']:.2f}, D={x['D']:.4f}, D2={x['D2_full']:.4f}, phase_D={x['phase_mean_D']:.4f}")

    # 2. Check ALL 64 zero-containing series: compare the 32 Tribolium series that reverse vs 4 Tribolium series that don't vs 28 compilation series
    print("\n=== Zero-containing series comparison ===")
    rev_sids = {r["series_id"] for r in saved_mono_attr if "zero_support" in r["feature_flags"]}
    for grp_name, pred_fn in [
        ("Tribolium reversing (32)", lambda r: r["source"] == "Tribolium" and r["series_id"] in rev_sids),
        ("Tribolium non-reversing (4)", lambda r: r["source"] == "Tribolium" and np.any(r["h"] == 0) and r["series_id"] not in rev_sids),
        ("Compilation with zeros (28)", lambda r: r["source"] != "Tribolium" and np.any(r["h"] == 0)),
    ]:
        sub = [r for r in records if pred_fn(r)]
        rho1s, rho2s, q0s, mgaps, d2_1s, d2_5s = [], [], [], [], [], []
        for r in sub:
            h = r["h"]
            C = circular_autocov(h - 1.0)
            rho1s.append(float(C[1] / C[0]))
            rho2s.append(float(C[2] / C[0]))
            q0s.append(float(np.mean(h == 0)))
            # gaps
            zm = (h == 0)
            fp = int(np.argmax(~zm))
            rot = np.roll(zm, -fp)
            gaps, cur = [], 0
            for fl in rot:
                if fl: cur += 1
                elif cur > 0: gaps.append(cur); cur = 0
            if cur > 0: gaps.append(cur)
            mgaps.append(float(np.mean(gaps)))
            d2_1s.append(float(saved_cells[(r["series_id"], 2, 1)]["D"]))
            d2_5s.append(float(saved_cells[(r["series_id"], 2, 5)]["D"]))
        print(f"  {grp_name}: n={len(sub)}, q0=[{min(q0s):.2f}, {max(q0s):.2f}] (med {np.median(q0s):.2f}), "
              f"mean_gap=[{min(mgaps):.2f}, {max(mgaps):.2f}], rho1=[{min(rho1s):.3f}, {max(rho1s):.3f}] (med {np.median(rho1s):.3f}), "
              f"rho2_med={np.median(rho2s):.3f}, D(2,1)=[{min(d2_1s):.4f}, {max(d2_1s):.4f}], D(2,5)=[{min(d2_5s):.4f}, {max(d2_5s):.4f}]")
        if len(sub) == 4:
            for r in sub:
                h = r["h"]
                C = circular_autocov(h - 1.0)
                print(f"    {r['series_id']}: q0={np.mean(h==0):.2f}, rho1={C[1]/C[0]:.3f}, rho2={C[2]/C[0]:.3f}, "
                      f"D_k2={[float(saved_cells[(r['series_id'],2,s)]['D']) for s in (1,5,20)]}, "
                      f"D_k3={[float(saved_cells[(r['series_id'],3,s)]['D']) for s in (1,5,20)]}")

    # 3. Numerical verification of exact outage laws:
    # (a) Long-stay asymptotic s * D(a, s) -> Var(h) * [1/2 + sum_{ell>=1} rho_H(ell) (1 - 1/a)^{ell-1}]
    #     on pure outage hosts h in {0, 1/(1-q0)}!
    print("\n=== Long-stay asymptotic s * D(2, s) with zeros ===")
    for P, pattern in [
        (2, [0.0, 2.0]),
        (3, [0.0, 0.0, 3.0]),
        (4, [0.0, 0.0, 2.0, 2.0]),
        (5, [0.0, 1.25, 0.0, 1.25, 2.5]),
        (10, [0.0]*7 + [10.0/3.0]*3),
    ]:
        hp = np.array(pattern, dtype=np.float64)
        assert abs(hp.mean() - 1.0) < 1e-12
        # Exact theoretical limit of (2 + s) * D(2, s) as s -> inf:
        # Since T0 = 2 + s, D2_raw(2, s) * (2 + s) -> Var(h) * [1/2 + sum_{ell>=1} rho_H(ell) * (1/2)^{ell-1}]
        C = circular_autocov(hp - 1.0)
        lags = np.arange(1, 200)
        asymp = float(C[0] * (0.5 + np.sum((C[lags % P] / C[0]) * (0.5 ** (lags - 1)))))
        d_20 = solve_replay_D(2, 20, hp)[0]
        d_200 = solve_replay_D(2, 200, hp)[0]
        d_2000 = solve_replay_D(2, 2000, hp)[0]
        print(f"  pattern P={P}, q0={np.mean(hp==0):.2f}: asymp_coeff={asymp:.6f} | "
              f"(2+20)*D(20)={(22*d_20):.6f}, (2+200)*D(200)={(202*d_200):.6f}, (2+2000)*D(2000)={(2002*d_2000):.6f}")

    # (b) Exact i.i.d. pure-outage D(q0, s) via deterministic transfer-matrix / Markov chain or long replay?
    # Wait: for i.i.d. h_t in {0, 1/(1-q0)} with P(h_t = 0) = q0, in (a, s):
    # Check by exact 1D invariant-law Nyström / Koopman quadrature!
    # For (2, s): A(h, y) = [[1/2, e^y / s], [h/2, 1 - 1/s]].
    # Let u_t = x_{1,t} / x_{0,t} >= 0!
    # Next step: x_{0,t+1} = 0.5 x_{0,t} + (e^y / s) x_{1,t} = x_{0,t} * (0.5 + (e^y / s) u_t)
    #            x_{1,t+1} = (h_t / 2) x_{0,t} + (1 - 1/s) x_{1,t} = x_{0,t} * (h_t / 2 + (1 - 1/s) u_t)
    # So u_{t+1} = (h_t / 2 + (1 - 1/s) u_t) / (0.5 + (e^y / s) u_t)!!
    # Look at s = 1: p_2 = 1 - 1/s = 0!!
    # WAIT A SECOND!! AT s = 1, p_2 = 0, SO u_{t+1} = (h_t / 2) / (0.5 + e^y u_t)!!
    # When h_t = 0 (prob q0), u_{t+1} = 0 (AN ATOM AT ZERO!!)!
    # When h_t = 1/(1-q0) (prob 1-q0), u_{t+1} = [1/(2(1-q0))] / (0.5 + e^y u_t)!
    # That means for i.i.d. (or ANY 2-state Markov!) pure outages at (a, 1), the invariant law of u_t is supported on a COUNTABLE SET OF ATOMS {u_0, u_1, u_2, ...} where u_0 = 0 is the post-outage state and u_m is the state after m consecutive on-steps!!
    print("\n=== Exact countable-atom solution for (a, s=1) under Markov / i.i.d. pure outages! ===")
    from scipy import optimize as opt
    for q0 in (0.1, 0.25, 0.5, 0.75, 0.9):
        for phi in (-0.5, 0.0, 0.5, 0.9):
            # 2-state Markov chain on I_t in {0, 1} with P(I=0)=q0, P(I=1)=1-q0, lag-1 corr phi:
            # Transition matrix:
            # P(0 -> 0) = q0 + (1 - q0)*phi,  P(0 -> 1) = (1 - q0)*(1 - phi)
            # P(1 -> 0) = q0*(1 - phi),       P(1 -> 1) = (1 - q0) + q0*phi
            p00 = q0 + (1.0 - q0) * phi
            p01 = (1.0 - q0) * (1.0 - phi)
            p10 = q0 * (1.0 - phi)
            p11 = (1.0 - q0) + q0 * phi
            if min(p00, p01, p10, p11) <= 0.0:
                continue
            h_on = 1.0 / (1.0 - q0)
            a = 2.0
            p1_stay = 1.0 - 1.0 / a  # = 0.5
            # At s = 1: whenever I_t = 0 (total prob q0), u_{t+1} = 0, and log_growth_step = ln(p1_stay + e^y u_t)!
            # Wait: what is the stationary distribution of run-length m >= 0 of consecutive 1s prior to t?
            # Event {u_t = u_0 = 0} is {I_{t-1} = 0}, which has stationary probability q0!
            # For m >= 1, {u_t = u_m} is {I_{t-1}=1, ..., I_{t-m}=1, I_{t-m-1}=0}, which has stationary probability:
            # pi_m = q0 * p01 * (p11 ** (m - 1))! (Check: sum_{m>=1} pi_m = q0 * p01 / (1 - p11) = q0 * p01 / p10 = 1 - q0! Exact!!)
            # And u_0 = 0, u_m = (h_on / a) / (p1_stay + e^y u_{m-1}) for m >= 1!
            # Expected log growth per step is simply:
            # r(y) = sum_{m=0}^inf pi_m * ln(p1_stay + e^y u_m(y))!!
            def growth_s1(y: float) -> float:
                ey = math.exp(y)
                total = q0 * math.log(p1_stay)  # m = 0: u_0 = 0!
                u = 0.0
                weight = q0 * p01
                for _ in range(5000):
                    u = (h_on / a) / (p1_stay + ey * u)
                    total += weight * math.log(p1_stay + ey * u)
                    weight *= p11
                    if weight < 1e-16:
                        break
                return total

            d_s1 = float(opt.brentq(growth_s1, -5.0, 20.0, xtol=1e-13))
            print(f"  q0={q0:.2f}, phi={phi:+.2f} (mean_gap={1/(1-p00):.2f}): exact D(2, s=1) = {d_s1:+.6f}, "
                  f"asymp (2+s)*D(2, s->inf) = {(q0/(1-q0))*(0.5 + phi/(1 - 0.5*phi)):+.6f}")


if __name__ == "__main__":
    main()
