#!/usr/bin/env python3
"""Quantitative theoretical predictions for the Wood et al. (2023) case study.

Derives directional predictions and illustrative effect sizes (in units of D,
the reduction in whole-cycle ln R0 relative to arithmetic-mean rates) from
the project's proved/refereed formulas (notes/exact-limits.md, section_ring,
section_mech, section_exact) and studies/engine-R16/lifecycle_engine.py.

Respects both caveats documented in the engine verification report:
  1. For contact=True, Driver(kind='bounded', low=None) is used at sigma <= 0.6
     (and low=0.20 at sigma=1.0 where symmetric low < 0 is infeasible, with its
     exact non-Gaussian Jensen gap J reported alongside D2).
  2. No shared-group forward population call is used here.

Reproduces four known analytical results (T0a-T0d) to preset tolerances before
computing any study prediction (Rule 7). Run with:
  python3 -I casestudies/wood/theory_predictions.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
DOBSON_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(DOBSON_ROOT, "studies", "engine-R16"))

import lifecycle_engine as le  # noqa: E402

TOL_EXACT = 1e-11
TOL_SIM_SE = 3.5


def closed_form_B2(taus: list[float], sigma: float, phi: float) -> float:
    """Statement ring-B2: single gain-only transfer in a geometric ring."""
    T = sum(taus)
    if phi == 0.0:
        return sigma * sigma / (2.0 * T)
    theta = phi / (1.0 - phi)
    R = 1.0
    for tau in taus:
        R *= 1.0 / (1.0 + tau / theta)
    K = (1.0 + 2.0 * theta) / (2.0 * T) - R / (1.0 - R)
    return sigma * sigma * K


def closed_form_C1b(taus: list[float], contact_idx: int, d_frac: float, sigma_raw: float, phi: float) -> float:
    """Statement mech-C1b: single contact-limited transfer, variance sigma_raw^2."""
    T = sum(taus)
    if phi == 0.0:
        return sigma_raw * sigma_raw * (d_frac * d_frac) / (2.0 * T)
    theta = phi / (1.0 - phi)
    qs = [phi / (phi + (1.0 - phi) * tau) for tau in taus]
    R = 1.0
    for q in qs:
        R *= q
    qj = qs[contact_idx]
    Kc = d_frac * d_frac * (1.0 + 2.0 * theta) / (2.0 * T) - d_frac * (R - (1.0 - d_frac) * qj) / (1.0 - R)
    return sigma_raw * sigma_raw * Kc


def make_ring_description(
    taus: list[float],
    fluct_indices: list[int],
    sigmas: list[float],
    phis: list[float],
    shapes: list[tuple[str, int]] | None = None,
) -> le.Description:
    n = len(taus)
    if shapes is None:
        shapes = [("memoryless", 1)] * n
    stages = [
        le.Stage(name=f"s{i}", mean=float(taus[i]), shape=shapes[i][0], substages=shapes[i][1])
        for i in range(n)
    ]
    drivers = []
    transfers = []
    for i in range(n):
        nxt = f"s{(i + 1) % n}"
        drv_name = None
        if i in fluct_indices:
            idx = fluct_indices.index(i)
            if sigmas[idx] > 0:
                drv_name = f"d{i}"
                drivers.append(le.Driver(name=drv_name, sigma=float(sigmas[idx]), phi=float(phis[idx]), kind="lognormal"))
        transfers.append(le.Transfer(source=f"s{i}", target=nxt, driver=drv_name, birth=(i == n - 1)))
    return le.Description(stages=stages, transfers=transfers, drivers=drivers)


def run_known_checks(lines: list[str]) -> None:
    lines.append("=== 1. Verification against known analytical results (Rule 7) ===")
    lines.append(f"Preset tolerances: algebraic rel/abs <= {TOL_EXACT:.1e}; Monte Carlo <= {TOL_SIM_SE:.1f} SE")

    # T0a: Statement ring-B2 on stays (4, 6), sigma=0.8, phi=0.8
    desc_a = make_ring_description([4.0, 6.0], [0], [0.8], [0.8])
    eng_a = le.Engine(desc_a).critical()
    d2_eng = eng_a.coefficient()["D2"]
    d2_cf = closed_form_B2([4.0, 6.0], 0.8, 0.8)
    err_a = abs(d2_eng - d2_cf)
    assert err_a <= TOL_EXACT, f"T0a failed: {err_a}"
    lines.append(f"[PASS] T0a (ring-B2 single fluctuating transfer): engine D2={d2_eng:.12f}, formula={d2_cf:.12f}, |diff|={err_a:.2e}")

    # T0b: Statement mech-C2a sheltered segment kernel invariance:
    # Model 1: exposed s0 (mean 3) -> sheltered chain of 4 substages (total mean 8) [2 'hosts']
    # Model 2: exposed s0 (mean 3) -> 4 separate steady 'hosts' in series each of memoryless mean 2 [5 'hosts']
    desc_b1 = make_ring_description([3.0, 8.0], [0], [1.0], [0.7], shapes=[("memoryless", 1), ("chain", 4)])
    desc_b2 = make_ring_description([3.0, 2.0, 2.0, 2.0, 2.0], [0], [1.0], [0.7])
    e_b1 = le.Engine(desc_b1).critical()
    e_b2 = le.Engine(desc_b2).critical()
    d2_b1 = e_b1.coefficient()["D2"]
    d2_b2 = e_b2.coefficient()["D2"]
    sim_b1 = e_b1.loss(steps=25000, burn=1000, pairs=16, seed=42)
    sim_b2 = e_b2.loss(steps=25000, burn=1000, pairs=16, seed=42)
    err_b_d2 = abs(d2_b1 - d2_b2)
    err_b_sim = abs(sim_b1["D"] - sim_b2["D"])
    assert err_b_d2 <= TOL_EXACT and err_b_sim <= 1e-10, f"T0b failed: {err_b_d2}, {err_b_sim}"
    lines.append(
        f"[PASS] T0b (mech-C2a sheltered-host count invariance): 2-host chain(4) D={sim_b1['D']:.10f} vs "
        f"5-host series D={sim_b2['D']:.10f}, |diff|={err_b_sim:.2e} (D2 |diff|={err_b_d2:.2e})"
    )

    # T0c: Statement ring-B6 / mech-C4d: M=2 equal independent alternative hosts halves D2 at phi=0
    desc_c1 = le.Description(
        stages=[le.Stage("central", 4.0), le.Stage("host", 6.0)],
        transfers=[
            le.Transfer("central", "host", share=1.0, driver="h1"),
            le.Transfer("host", "central", share=1.0, birth=True),
        ],
        drivers=[le.Driver("h1", sigma=0.7, phi=0.0, kind="lognormal")],
    )
    desc_c2 = le.Description(
        stages=[le.Stage("central", 4.0), le.Stage("host", 6.0)],
        transfers=[
            le.Transfer("central", "host", share=0.5, driver="h1"),
            le.Transfer("central", "host", share=0.5, driver="h2"),
            le.Transfer("host", "central", share=1.0, birth=True),
        ],
        drivers=[
            le.Driver("h1", sigma=0.7, phi=0.0, kind="lognormal"),
            le.Driver("h2", sigma=0.7, phi=0.0, kind="lognormal"),
        ],
    )
    ratio_c = le.Engine(desc_c2).critical().coefficient()["D2"] / le.Engine(desc_c1).critical().coefficient()["D2"]
    assert abs(ratio_c - 0.5) <= TOL_EXACT, f"T0c failed: {ratio_c}"
    lines.append(f"[PASS] T0c (ring-B6 two equal independent alternative hosts): D2_alt / D2_single = {ratio_c:.12f}")

    # T0d: Statement mech-C1b & C1g: contact-limited (4, 6), d=0.5, phi=0.8, small symmetric bounded noise
    # Note: Driver(sigma=s, kind='bounded', low=None) has raw variance exp(s^2)-1, whereas engine.sensitivities()
    # scales by s; at s=0.02, exp(s^2)-1 = s^2 * (1 + s^2/2 + ...), matching C1b to O(s^4).
    s_small = 1e-4
    # mean=4 -> hazard=0.25; d=0.5 -> death = 0.125, c = 0.125
    desc_d = le.Description(
        stages=[le.Stage("wait", 4.0, death=0.125, contact=True), le.Stage("other", 6.0)],
        transfers=[
            le.Transfer("wait", "other", driver="h", birth=False),
            le.Transfer("other", "wait", birth=True),
        ],
        drivers=[le.Driver("h", sigma=s_small, phi=0.8, kind="bounded", low=None)],
    )
    d2_eng_d = le.Engine(desc_d).critical().coefficient()["D2"]
    d2_cf_d = closed_form_C1b([4.0, 6.0], 0, 0.5, s_small, 0.8)
    rel_err_d = abs(d2_eng_d - d2_cf_d) / d2_cf_d
    assert rel_err_d <= 1e-7, f"T0d failed: {rel_err_d}"
    # Also check C1g limit (death -> 0 => D -> 0)
    desc_d0 = le.Description(
        stages=[le.Stage("wait", 4.0, death=0.0, contact=True), le.Stage("other", 6.0)],
        transfers=[
            le.Transfer("wait", "other", driver="h", birth=False),
            le.Transfer("other", "wait", birth=True),
        ],
        drivers=[le.Driver("h", sigma=0.5, phi=0.8, kind="bounded", low=None)],
    )
    sim_d0 = le.Engine(desc_d0).critical().loss(steps=15000, burn=500, pairs=8, seed=11)
    assert abs(sim_d0["D"]) <= 1e-12, f"T0d C1g failed: {sim_d0['D']}"
    lines.append(
        f"[PASS] T0d (mech-C1b/C1g contact-limited): rel_diff={rel_err_d:.2e}; at d=0 (no waiting death), simulated D={sim_d0['D']:.2e}"
    )
    lines.append("")


def run_study_predictions(lines: list[str]) -> dict:
    out: dict = {}
    lines.append("=== 2. Quantitative predictions for illustrative marine life-cycle archetypes ===")
    lines.append(
        "All inputs below are illustrative ranges grounded in project records (studies.INPUTS, R12, R14: "
        "fluctuating host log-SD sigma_high in {0.6, 1.0}, buffered host sigma_low = 0.30; step = 1 month; "
        "AR(1) monthly memory phi in {0.0, 0.5, 0.75}). D is in natural-log units of whole-cycle R0."
    )

    # ------------------------------------------------------------------
    # Contrast 0: Pure host number vs fluctuating transfers (H3, C2a, C4c)
    # Fix 1 fluctuating transfer (sigma=1.0) and total mean generation time T=12 months:
    # 1-host (12), 2-host (2, 10) with 2nd host steady, 3-host (2, 4, 6) with 2nd & 3rd steady.
    # ------------------------------------------------------------------
    lines.append("\n--- Contrast H0: Number of obligate hosts in sequence with fixed 1 fluctuating transfer (T=12) ---")
    h0_rows = []
    for label, taus in [
        ("1-host (12), 1 fluct", [12.0]),
        ("2-host (2, 10), 1 fluct + 1 steady", [2.0, 10.0]),
        ("3-host (2, 4, 6), 1 fluct + 2 steady", [2.0, 4.0, 6.0]),
    ]:
        for phi in (0.0, 0.5, 0.75):
            eng = le.Engine(make_ring_description(taus, [0], [1.0], [phi])).critical()
            d2 = eng.coefficient()["D2"]
            sim = eng.loss(steps=30000, burn=1000, pairs=16, seed=101)
            row = {"case": label, "phi": phi, "D2": d2, "D": sim["D"], "se": sim["se"], "J": sim["J"]}
            h0_rows.append(row)
            lines.append(
                f"  {label:38s} | phi={phi:.2f} | D2={d2:.4f} | D={sim['D']:.4f} +- {sim['se']:.4f} | bound J={sim['J']:.4f}"
            )
    out["H0_host_number_alone"] = h0_rows

    # ------------------------------------------------------------------
    # Prediction (a): Number of fluctuating obligate transfers N_fluc in {0, 1, 2, 3}
    # 3-stage cycle with stays (2, 3, 4) [T=9 months]; non-fluctuating steps have buffered sigma_low=0.30 (or 0).
    # ------------------------------------------------------------------
    lines.append("\n--- Prediction (a): Number of fluctuating obligate transfers N_fluc (3-stage cycle, stays 2,3,4) ---")
    pa_rows = []
    for sigma_hi in (0.6, 1.0):
        for phi in (0.5, 0.75):
            for n_fluc in (0, 1, 2, 3):
                # buffered steps carry sigma_low=0.25, fluctuating steps carry sigma_hi
                sigs = [sigma_hi if j < n_fluc else 0.25 for j in range(3)]
                eng = le.Engine(make_ring_description([2.0, 3.0, 4.0], [0, 1, 2], sigs, [phi] * 3)).critical()
                d2 = eng.coefficient()["D2"]
                sim = eng.loss(steps=30000, burn=1000, pairs=16, seed=200 + n_fluc)
                row = {
                    "sigma_hi": sigma_hi,
                    "sigma_lo": 0.25,
                    "phi": phi,
                    "N_fluc": n_fluc,
                    "D2": d2,
                    "D": sim["D"],
                    "se": sim["se"],
                    "J": sim["J"],
                }
                pa_rows.append(row)
                lines.append(
                    f"  sigma_hi={sigma_hi:.1f}, phi={phi:.2f}, N_fluc={n_fluc} | D2={d2:.4f} | "
                    f"D={sim['D']:.4f} +- {sim['se']:.4f} | bound J={sim['J']:.4f}"
                )
    out["Pa_fluctuating_transfers"] = pa_rows

    # ------------------------------------------------------------------
    # Prediction (b): Storage stage with variable residence (H2, B5a, C2d, exact-P4)
    # Compare 3-host cycle without vertebrate tissue cyst (stays 1, 2, 3: snail/copepod/luminal GI adult)
    # vs with multi-year encysted vertebrate tissue larva (stays 1, 2, 24: memoryless vs chain(4)),
    # carrying 2 fluctuating transfers (sigma=0.8 or 1.0).
    # ------------------------------------------------------------------
    lines.append("\n--- Prediction (b): Long-lived variable-residence storage stage (stays 1,2,3 vs 1,2,24) ---")
    pb_rows = []
    for sigma in (0.6, 1.0):
        for phi in (0.5, 0.75):
            for label, taus, shapes in [
                ("no_storage (1,2,3) memoryless", [1.0, 2.0, 3.0], [("memoryless", 1)] * 3),
                ("storage (1,2,24) memoryless", [1.0, 2.0, 24.0], [("memoryless", 1)] * 3),
                ("storage (1,2,24) Erlang-4", [1.0, 2.0, 24.0], [("memoryless", 1), ("memoryless", 1), ("chain", 4)]),
            ]:
                eng = le.Engine(make_ring_description(taus, [0, 1], [sigma, sigma], [phi, phi], shapes=shapes)).critical()
                d2 = eng.coefficient()["D2"]
                sim = eng.loss(steps=35000, burn=1000, pairs=16, seed=303)
                row = {"sigma": sigma, "phi": phi, "model": label, "D2": d2, "D": sim["D"], "se": sim["se"]}
                pb_rows.append(row)
                lines.append(
                    f"  sigma={sigma:.1f}, phi={phi:.2f} | {label:30s} | D2={d2:.4f} | D={sim['D']:.4f} +- {sim['se']:.4f}"
                )
    out["Pb_storage"] = pb_rows

    # ------------------------------------------------------------------
    # Prediction (c): Alternative / paratenic hosts at a step (H3, B6, C4d, C4f)
    # 3-stage cycle (2, 3, 6) where transfer 1->2 uses M=1 specialist vs M=3 independent or rho=0.3 correlated hosts,
    # plus optional paratenic bridge of stay 12 (share f=0.5, survival q=1.0 at equal R0).
    # ------------------------------------------------------------------
    lines.append("\n--- Prediction (c): Alternative / paratenic host spreading (B6, C4d, C4f) ---")
    pc_rows = []
    for sigma in (0.6, 1.0):
        phi = 0.6
        # Specialist M=1 on transfer s1->s2, plus background fluctuating transfer s0->s1 (sigma=0.6)
        desc_m1 = le.Description(
            stages=[le.Stage("s0", 2.0), le.Stage("s1", 3.0), le.Stage("s2", 6.0)],
            transfers=[
                le.Transfer("s0", "s1", driver="d0"),
                le.Transfer("s1", "s2", share=1.0, driver="h1"),
                le.Transfer("s2", "s0", birth=True),
            ],
            drivers=[
                le.Driver("d0", sigma=sigma, phi=phi),
                le.Driver("h1", sigma=sigma, phi=phi),
            ],
        )
        # Generalist M=4 equal-share independent hosts on s1->s2
        desc_m4_ind = le.Description(
            stages=[le.Stage("s0", 2.0), le.Stage("s1", 3.0), le.Stage("s2", 6.0)],
            transfers=[
                le.Transfer("s0", "s1", driver="d0"),
                le.Transfer("s1", "s2", share=0.25, driver="h1"),
                le.Transfer("s1", "s2", share=0.25, driver="h2"),
                le.Transfer("s1", "s2", share=0.25, driver="h3"),
                le.Transfer("s1", "s2", share=0.25, driver="h4"),
                le.Transfer("s2", "s0", birth=True),
            ],
            drivers=[
                le.Driver("d0", sigma=sigma, phi=phi),
                le.Driver("h1", sigma=sigma, phi=phi),
                le.Driver("h2", sigma=sigma, phi=phi),
                le.Driver("h3", sigma=sigma, phi=phi),
                le.Driver("h4", sigma=sigma, phi=phi),
            ],
        )
        # Generalist M=4 partially correlated hosts (rho=0.35 among the 4 alternative hosts)
        corr = np.eye(5)
        for a in range(1, 5):
            for b in range(1, 5):
                if a != b:
                    corr[a, b] = 0.35
        desc_m4_corr = le.Description(
            stages=desc_m4_ind.stages,
            transfers=desc_m4_ind.transfers,
            drivers=desc_m4_ind.drivers,
            correlation=corr,
        )
        # Optional paratenic vertebrate bridge (share 0.5 through paratenic stage of stay 12, M=4 hosts)
        desc_par = le.Description(
            stages=[le.Stage("s0", 2.0), le.Stage("s1", 3.0), le.Stage("par", 12.0), le.Stage("s2", 6.0)],
            transfers=[
                le.Transfer("s0", "s1", driver="d0"),
                le.Transfer("s1", "s2", share=0.5, driver="h1"),
                le.Transfer("s1", "par", share=0.25, driver="h2"),
                le.Transfer("s1", "par", share=0.25, driver="h3"),
                le.Transfer("par", "s2", share=1.0),
                le.Transfer("s2", "s0", birth=True),
            ],
            drivers=[
                le.Driver("d0", sigma=sigma, phi=phi),
                le.Driver("h1", sigma=sigma, phi=phi),
                le.Driver("h2", sigma=sigma, phi=phi),
                le.Driver("h3", sigma=sigma, phi=phi),
            ],
        )
        for label, dsc in [
            ("specialist (M=1)", desc_m1),
            ("generalist (M=4, rho=0)", desc_m4_ind),
            ("generalist (M=4, rho=0.35)", desc_m4_corr),
            ("paratenic bridge + spreading", desc_par),
        ]:
            eng = le.Engine(dsc).critical()
            d2 = eng.coefficient()["D2"]
            sim = eng.loss(steps=30000, burn=1000, pairs=16, seed=404)
            row = {"sigma": sigma, "phi": phi, "model": label, "D2": d2, "D": sim["D"], "se": sim["se"]}
            pc_rows.append(row)
            lines.append(
                f"  sigma={sigma:.1f}, phi={phi:.2f} | {label:30s} | D2={d2:.4f} | D={sim['D']:.4f} +- {sim['se']:.4f}"
            )
    out["Pc_spreading"] = pc_rows

    # ------------------------------------------------------------------
    # Prediction (d): Active free-living waste (d=1) vs trophic encysted waiting (d in {0.05, 0.15, 0.50})
    # Fine time step (wait=4 steps), symmetric bounded driver at sigma=0.6 (raw SD = sqrt(exp(0.36)-1)=0.658).
    # Show BOTH short other stay (stay=6) and long steady stay (stay=40) to display C1e/C1g protection at low d
    # AND the C1h/C1i crossover when d=0.5 with long steady stay and high memory!
    # ------------------------------------------------------------------
    lines.append("\n--- Prediction (d): Active waste (d=1) vs trophic waiting (d < 1) [symmetric bounded host, sigma=0.6] ---")
    pd_rows = []
    sigma_b = 0.6
    raw_sd = math.sqrt(math.expm1(sigma_b * sigma_b))
    wait_mean = 4.0  # hazard = 0.25
    for other_stay in (6.0, 40.0):
        for phi in (0.0, 0.75):
            # Waste (d=1, contact=False)
            desc_w = le.Description(
                stages=[le.Stage("wait", wait_mean), le.Stage("other", other_stay)],
                transfers=[
                    le.Transfer("wait", "other", driver="h"),
                    le.Transfer("other", "wait", birth=True),
                ],
                drivers=[le.Driver("h", sigma=sigma_b, phi=phi, kind="bounded", low=None)],
            )
            eng_w = le.Engine(desc_w).critical()
            sim_w = eng_w.loss(steps=35000, burn=1000, pairs=16, seed=505)
            d2_w = closed_form_B2([wait_mean, other_stay], raw_sd, phi)
            pd_rows.append({
                "other_stay": other_stay, "phi": phi, "mode": "active_waste (d=1.00)",
                "d": 1.0, "D2_exact_var": d2_w, "D": sim_w["D"], "se": sim_w["se"],
            })
            lines.append(
                f"  other={other_stay:4.0f}, phi={phi:.2f} | active_waste (d=1.00)  | D2={d2_w:.4f} | D={sim_w['D']:.4f} +- {sim_w['se']:.4f}"
            )
            for d_val in (0.50, 0.15, 0.05):
                death = (1.0 / wait_mean) * d_val
                desc_c = le.Description(
                    stages=[le.Stage("wait", wait_mean, death=death, contact=True), le.Stage("other", other_stay)],
                    transfers=[
                        le.Transfer("wait", "other", driver="h"),
                        le.Transfer("other", "wait", birth=True),
                    ],
                    drivers=[le.Driver("h", sigma=sigma_b, phi=phi, kind="bounded", low=None)],
                )
                eng_c = le.Engine(desc_c).critical()
                sim_c = eng_c.loss(steps=35000, burn=1000, pairs=16, seed=505)
                d2_c = closed_form_C1b([wait_mean, other_stay], 0, d_val, raw_sd, phi)
                pd_rows.append({
                    "other_stay": other_stay, "phi": phi, "mode": f"trophic_wait (d={d_val:.2f})",
                    "d": d_val, "D2_exact_var": d2_c, "D": sim_c["D"], "se": sim_c["se"],
                })
                lines.append(
                    f"  other={other_stay:4.0f}, phi={phi:.2f} | trophic_wait (d={d_val:.2f}) | D2={d2_c:.4f} | D={sim_c['D']:.4f} +- {sim_c['se']:.4f}"
                )
    out["Pd_active_vs_trophic"] = pd_rows
    return out


def main() -> int:
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    lines: list[str] = []
    run_known_checks(lines)
    summary = run_study_predictions(lines)

    txt_path = os.path.join(res_dir, "theory_predictions.txt")
    json_path = os.path.join(res_dir, "theory_predictions.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n".join(lines))
    print(f"\nSaved {txt_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
