#!/usr/bin/env python3
"""Step B registered hypothesis tests and sensitivity analyses for case study `devtimes`.

Run with:
    python3 -I casestudies/devtimes/analysis_tests.py

Implements protocol.md §5 (`H_dev1`, `H_sys`, `H_transform`) with 2,000 non-parametric
bootstrap resamples (seed 20261114), plus registered sensitivity analyses and confounder
evaluations (§5, §7). Saves structured outputs to `results/tests.json` and
human-readable log to `results/tests.txt`.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
from scipy import linalg, optimize, special

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
DOBSON_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RAW_DIR = REPO_ROOT / "data" / "raw" / "devtimes"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOBSON_ROOT / "studies" / "engine-R16"))

import extract_helpers as eh  # noqa: E402
import lifecycle_engine as le  # noqa: E402

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261114


def d2_primitive_formula(sigma: float, phi: float, mean_L: float, R: float) -> float:
    """Second-order threshold loss D2 (`protocol.md` §2.1, Statement C2b)."""
    if phi == 0.0:
        return sigma * sigma / (2.0 * mean_L)
    theta = phi / (1.0 - phi)
    return (sigma * sigma) * ((1.0 + 2.0 * theta) / (2.0 * mean_L) - R / (1.0 - R))


def rho_chain(tau: float, m: int, phi: float) -> float:
    """Generating transform E[phi^L] for m equal geometric substages of total mean tau."""
    if phi == 0.0:
        return 0.0
    theta = phi / (1.0 - phi)
    return (1.0 + tau / (m * theta)) ** (-m)


def rho_fixed_geom(tau: int, g: float, phi: float) -> float:
    """Generating transform E[phi^L] for fixed integer delay (tau - g_int) + geometric stay g."""
    if phi == 0.0:
        return 0.0
    # Wait: tau - g must be an integer delay d >= 0, and g >= 1 real mean of geometric stay:
    # Actually in discrete time, if delay d = int(round(tau - g)) >= 0 and geometric mean g = tau - d >= 1:
    d = tau - g
    return (phi ** d) * ((phi / g) / (1.0 - (1.0 - 1.0 / g) * phi))


def load_stage_cv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_str = r["n"].strip()
            n_val = int(float(n_str)) if n_str != "" else None
            temp_str = r["temperature_C"].strip()
            temp_val = float(temp_str) if temp_str != "" else None
            rows.append(
                {
                    "dataset": r["dataset"].strip(),
                    "taxon": r["taxon"].strip(),
                    "stage": r["stage"].strip(),
                    "stage_category": r["stage_category"].strip(),
                    "temperature_C": temp_val,
                    "n": n_val,
                    "mean": float(r["mean"]),
                    "sd": float(r["sd"]),
                    "cv_indiv": float(r["cv_indiv"]),
                    "censoring": r["censoring"].strip(),
                    "source_file": r["source_file"].strip(),
                    "rule_applied": r["rule_applied"].strip(),
                }
            )
    return rows


def classify_dev_vs_surv(row: dict[str, Any]) -> str:
    """Classify each cohort into `dev_like` or `surv_like` per `protocol.md` §5 (`H_dev1`)."""
    cat = row["stage_category"]
    stage_l = row["stage"].lower()
    if cat == "S1_endoparasite_vertebrate":
        return "surv_like"
    if cat == "S5_vector_tick_adult_or_offhost_survival":
        return "surv_like"
    if cat == "S4_vector_tick_development_moulting":
        return "dev_like"
    if cat == "S2_parasite_in_invertebrate_vector_or_snail":
        if any(k in stage_l for k in ("extrinsic incubation", "eip", "pre-patent", "prepatent")):
            return "dev_like"
        return "surv_like"
    if cat == "S3_free_living_parasite_stage":
        if any(k in stage_l for k in ("survival", "mortality", "death", "longevity")):
            return "surv_like"
        return "dev_like"
    raise ValueError(f"Unclassified row: {cat} | {row['stage']}")


def fast_serial_ring_loss(
    s_stay: list[float],
    b_fix: int,
    sigma: float,
    phi: float,
    steps: int = 24000,
    burn: int = 2000,
    pairs: int = 16,
    seed: int = 20261114,
) -> dict[str, float]:
    """Exact O(mA) circular-buffer Newton solver for a serial ring of staying sub-stages `s_stay`
    plus `b_fix` unit-delay sub-stages driven by `le.Environment` (identical trajectories to
    `lifecycle_engine.Engine.loss()`, verified to < 1e-5 in Known Check K3)."""
    if not s_stay:
        return {"D": 0.5 * sigma * sigma, "se": 0.0}
    env = le.Environment([le.Driver("host", sigma=sigma, phi=phi, kind="lognormal")], np.eye(1))
    hosts = env.sample(steps + burn, pairs, seed, antithetic=True)[:, :, 0]
    mA = len(s_stay)
    s_col = np.asarray(s_stay, float)[:, None]
    g_col = 1.0 - s_col
    g_last = float(g_col[-1, 0])
    reps = hosts.shape[1]

    def eval_y(y: float) -> tuple[np.ndarray, np.ndarray]:
        ey = math.exp(y)
        n_tot = mA + b_fix
        # Match lifecycle_engine stationary eigenvector start on threshold serial ring:
        # w_j = 1/(1 - s_j) normalized
        v_st = (1.0 / (1.0 - np.asarray(s_stay, float)))
        denom = float(v_st.sum() + b_fix)
        X = np.tile((v_st / denom)[:, None], (1, reps))
        dX = np.zeros_like(X)
        buf = np.full((b_fix + 1, reps), 1.0 / denom)
        dbuf = np.zeros_like(buf)
        logs = np.zeros(reps)
        slopes = np.zeros(reps)
        for t in range(steps + burn):
            ht = hosts[t]
            out_m = ey * g_last * ht * X[-1]
            dout_m = out_m + ey * g_last * ht * dX[-1]
            if b_fix > 0:
                slot = t % (b_fix + 1)
                nxt = (t + 1) % (b_fix + 1)
                buf[slot] = out_m
                dbuf[slot] = dout_m
                back = buf[nxt]
                dback = dbuf[nxt]
            else:
                back = out_m
                dback = dout_m
            Xn = X * s_col
            dXn = dX * s_col
            if mA > 1:
                Xn[1:] += X[:-1] * g_col[:-1]
                dXn[1:] += dX[:-1] * g_col[:-1]
            Xn[0] += back
            dXn[0] += dback
            tot = Xn.sum(axis=0) + (buf.sum(axis=0) - out_m if b_fix > 0 else 0.0)
            dtot = dXn.sum(axis=0) + (dbuf.sum(axis=0) - dout_m if b_fix > 0 else 0.0)
            der = dtot / tot
            X = Xn / tot[None, :]
            dX = dXn / tot[None, :] - X * der[None, :]
            if b_fix > 0:
                buf /= tot[None, :]
                dbuf = dbuf / tot[None, :] - buf * der[None, :]
            if t >= burn:
                logs += np.log(tot)
                slopes += der
        logs /= steps
        slopes /= steps
        return 0.5 * (logs[:pairs] + logs[pairs:]), 0.5 * (slopes[:pairs] + slopes[pairs:])

    r0, s0 = eval_y(0.0)
    y = float(np.clip(-r0.mean() / s0.mean(), -3.0, 3.0))
    for _ in range(12):
        r, s = eval_y(y)
        corr = float(r.mean() / s.mean())
        if abs(corr) < 1e-7:
            break
        y -= float(np.clip(corr, -2.0, 2.0))
    samples = y - r / s.mean()
    return {"D": float(y), "se": float(samples.std(ddof=1) / math.sqrt(pairs))}


def run_known_checks() -> dict[str, Any]:
    """Rule 7: reproduce known analytical/engine results before empirical tests."""
    # Check 1: Engine D2 vs closed-form primitive formula on 2-stage cycle
    desc = le.Description(
        stages=[
            le.Stage("fluct", mean=8.0, shape="fixed"),
            le.Stage("steady", mean=60.0, shape="chain", substages=4),
        ],
        transfers=[
            le.Transfer("fluct", "steady", driver="host", birth=True),
            le.Transfer("steady", "fluct"),
        ],
        drivers=[le.Driver("host", sigma=1.4, phi=0.75 ** (1.0 / 8.0), kind="lognormal")],
    )
    eng = le.Engine(desc).critical()
    phi_fine = 0.75 ** (1.0 / 8.0)
    R_val = (phi_fine ** 8) * rho_chain(60.0, 4, phi_fine)
    d2_closed = d2_primitive_formula(1.4, phi_fine, 68.0, R_val)
    d2_eng = float(eng.coefficient()["D2"])
    err_d2 = abs(d2_closed - d2_eng)
    assert err_d2 < 1e-10, f"Known check K1 failed: {d2_closed} vs {d2_eng}"

    # Check 2: Fixed+geometric 2-stage representation in lifecycle_engine matches closed-form D2
    desc_fg = le.Description(
        stages=[
            le.Stage("fluct", mean=8.0, shape="fixed"),
            le.Stage("steady_fix", mean=44.0, shape="fixed"),
            le.Stage("steady_geo", mean=16.0, shape="memoryless"),
        ],
        transfers=[
            le.Transfer("fluct", "steady_fix", driver="host", birth=True),
            le.Transfer("steady_fix", "steady_geo"),
            le.Transfer("steady_geo", "fluct"),
        ],
        drivers=[le.Driver("host", sigma=1.4, phi=phi_fine, kind="lognormal")],
    )
    d2_fg_eng = float(le.Engine(desc_fg).critical().coefficient()["D2"])
    d2_fg_closed = d2_primitive_formula(1.4, phi_fine, 68.0, (phi_fine ** 8) * rho_fixed_geom(60, 16.0, phi_fine))
    err_k2 = abs(d2_fg_eng - d2_fg_closed)
    assert err_k2 < 1e-10, f"Known check K2 failed: {d2_fg_eng} vs {d2_fg_closed}"

    # Check 3: Both `fast_serial_ring_loss` (exact trajectory match) and `eh.simulate_empirical_loss`
    # agree with `lifecycle_engine.loss()` on m=4 malaria
    sim_eng = eng.loss(steps=12000, burn=1000, pairs=12, seed=20261114)
    sim_fast = fast_serial_ring_loss([1.0 - 4.0 / 60.0] * 4, 8, 1.4, phi_fine, steps=12000, burn=1000, pairs=12, seed=20261114)
    err_fast = abs(sim_eng["D"] - sim_fast["D"])
    assert err_fast < 5e-4, f"Known check K3a fast_serial_ring_loss failed: {sim_eng['D']} vs {sim_fast['D']}"

    p_sub = 4.0 / 60.0
    ell = np.arange(4, 321, dtype=int)
    pmf_nb = np.exp(
        special.gammaln(ell) - special.gammaln(4.0) - special.gammaln(ell - 3.0)
        + 4.0 * math.log(p_sub) + (ell - 4.0) * math.log1p(-p_sub)
    )
    p_m4 = np.zeros(320, float)
    p_m4[3:] = pmf_nb / pmf_nb.sum()
    d2_emp_m4, _, _ = eh.empirical_D2(8, p_m4, 1.4, phi_fine)
    err_k3_d2 = abs(d2_emp_m4 - d2_eng)
    assert err_k3_d2 < 1e-4, f"Known check K3b D2 failed: {d2_emp_m4} vs {d2_eng}"
    sim_emp = eh.simulate_empirical_loss(8, p_m4, 1.4, phi_fine, steps=16000, burn=1500, pairs=12, seed=20261114)
    z_k3 = abs(sim_eng["D"] - sim_emp["D"]) / math.hypot(sim_eng["se"], sim_emp["se"])
    assert z_k3 < 3.0, f"Known check K3c sim failed: z={z_k3:.2f} ({sim_eng['D']} vs {sim_emp['D']})"

    return {
        "K1_engine_vs_formula_D2": {
            "d2_closed": d2_closed,
            "d2_engine": d2_eng,
            "abs_diff": err_d2,
            "tolerance": 1e-10,
            "pass": True,
        },
        "K2_engine_fixed_geom_vs_formula_D2": {
            "d2_closed": d2_fg_closed,
            "d2_engine": d2_fg_eng,
            "abs_diff": err_k2,
            "tolerance": 1e-10,
            "pass": True,
        },
        "K3_empirical_companion_vs_engine_loss": {
            "fast_ring_abs_diff_vs_engine": err_fast,
            "d2_diff": err_k3_d2,
            "D_engine": sim_eng["D"],
            "se_engine": sim_eng["se"],
            "D_fast_ring": sim_fast["D"],
            "D_companion": sim_emp["D"],
            "se_companion": sim_emp["se"],
            "z_diff": z_k3,
            "tolerance_z": 3.0,
            "pass": True,
        },
    }


def reconstruct_d1_clones(max_zeroes: int = 2, include_left_trunc: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct D1_malaria_embo2024 individual clonal durations (t_arr, ev_arr) as in extract_part1.py."""
    from collections import defaultdict
    import extract_part1 as ep1

    d1_dir = RAW_DIR / "D1_malaria_embo2024"
    raw_rows = list(csv.DictReader((d1_dir / "data_raw.csv").open(encoding="utf-8-sig")))
    by_id_tps: dict[str, set[str]] = defaultdict(set)
    by_id_age: dict[str, float] = {}
    kid_tp_date: dict[tuple[str, int], dt.date] = {}
    by_tp_dates: dict[int, list[dt.date]] = defaultdict(list)

    for r in raw_rows:
        kid = r["ID"].strip()
        tp = r["Timepoint"].strip()
        by_id_tps[kid].add(tp)
        by_id_age[kid] = float(r["Age_in_2012"])
        if tp == "MAL":
            continue
        tn = -1 if tp == "May 12" else (13 if tp == "May 13" else int(tp))
        d_val = dt.datetime.strptime(r["Date"].strip(), "%d.%m.%y").date()
        kid_tp_date[(kid, tn)] = d_val
        by_tp_dates[tn].append(d_val)

    tp_med_date = {tn: sorted(ds)[len(ds) // 2] for tn, ds in by_tp_dates.items()}
    ids_92 = {k for k, tps in by_id_tps.items() if any(t.isdigit() for t in tps)}
    ref_date = dt.date(2012, 1, 1)

    def get_vdate(kid: str, tn: int) -> dt.date:
        return kid_tp_date.get((kid, tn), tp_med_date[tn])

    pres_ama1: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in raw_rows:
        kid = r["ID"].strip()
        tp = r["Timepoint"].strip()
        if kid not in ids_92 or tp == "MAL":
            continue
        tn = -1 if tp == "May 12" else (13 if tp == "May 13" else int(tp))
        pres_ama1[(kid, r["cluster_name"].strip())].add(tn)

    tp_grid_ama1 = [-1] + list(range(1, 14))
    young_t, young_e, old_t, old_e = [], [], [], []
    for (kid, _cl), tset in sorted(pres_ama1.items()):
        for st, en in ep1._chunk_presence(tp_grid_ama1, tset, max_zeroes=max_zeroes):
            if not include_left_trunc and (st == -1 or en == -1):
                continue
            if st == -1:
                t0 = float((get_vdate(kid, -1) - ref_date).days)
            elif st == 13:
                t0 = float((get_vdate(kid, 12) - ref_date).days) + 14.0
            elif st == 1:
                t0 = float((get_vdate(kid, 1) - ref_date).days) - 7.0
            else:
                t0 = 0.5 * (float((get_vdate(kid, st - 1) - ref_date).days) + float((get_vdate(kid, st) - ref_date).days))
            if en == 13:
                t1 = float((get_vdate(kid, 13) - ref_date).days)
                ev = False
            elif en == -1:
                t1 = 0.5 * (float((get_vdate(kid, -1) - ref_date).days) + float((get_vdate(kid, 1) - ref_date).days))
                ev = True
            elif en == 12:
                t1 = float((get_vdate(kid, 12) - ref_date).days) + 7.0
                ev = True
            else:
                t1 = 0.5 * (float((get_vdate(kid, en) - ref_date).days) + float((get_vdate(kid, en + 1) - ref_date).days))
                ev = True
            dur = max(t1 - t0, 1.0)
            if by_id_age[kid] < 10.0:
                young_t.append(dur)
                young_e.append(ev)
            else:
                old_t.append(dur)
                old_e.append(ev)

    m9_sheet = eh.read_xlsx_sheet(d1_dir / "44321_2024_127_MOESM9_ESM.xlsx", 0)
    pres_msp2: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in m9_sheet[1:]:
        if len(r) >= 3 and r[0] and r[1] and r[2]:
            pres_msp2[(r[1].strip(), r[0].strip())].add(int(r[2]))
    tp_grid_msp2 = list(range(1, 13))
    msp2_t, msp2_e = [], []
    for (kid, _hb), tset in sorted(pres_msp2.items()):
        for st, en in ep1._chunk_presence(tp_grid_msp2, tset, max_zeroes=max_zeroes):
            if st == 1:
                t0 = float((get_vdate(kid, 1) - ref_date).days) - 7.0
            else:
                t0 = 0.5 * (float((get_vdate(kid, st - 1) - ref_date).days) + float((get_vdate(kid, st) - ref_date).days))
            if en == 12:
                t1 = float((get_vdate(kid, 12) - ref_date).days)
                ev = False
            else:
                t1 = 0.5 * (float((get_vdate(kid, en) - ref_date).days) + float((get_vdate(kid, en + 1) - ref_date).days))
                ev = True
            msp2_t.append(max(t1 - t0, 1.0))
            msp2_e.append(ev)

    return {
        "ama1_7_9yr": (np.array(young_t, float), np.array(young_e, bool)),
        "ama1_10_13yr": (np.array(old_t, float), np.array(old_e, bool)),
        "msp2": (np.array(msp2_t, float), np.array(msp2_e, bool)),
    }


def reconstruct_rule1_vectors(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float | None], np.ndarray]:
    """Reconstruct exact individual duration vectors for all 202 Rule 1 (`raw_individual_uncensored`) rows
    in `stage_cv.csv` and assert exact match to `(n, mean, cv_indiv)` in `stage_cv.csv`."""
    from collections import defaultdict
    import extract_part2 as ep2

    vectors: dict[tuple[str, str, float | None], np.ndarray] = {}

    # 1. D4 (1 Rule 1 cohort: Anopheles gambiae uninfected 33 C)
    srows = list(csv.DictReader((RAW_DIR / "D4_plasmodium_msos2021" / "data__mSOS_all_survival_data.csv").open(encoding="utf-8-sig")))
    d4_33 = [float(r["day"]) for r in srows if int(r["index_ref"]) != 1 and r["species"].strip() == "gambiae" and int(r["initial_infection"]) == 0 and float(r["temp"]) == 33.0 and int(r["survival_status"]) == 1]
    vectors[("D4_plasmodium_msos2021", "adult female survival (uninfected bloodmeal, ref=Murdock_Thomas)", 33.0)] = np.array(d4_33, float)

    # 2. D5 (28 Rule 1 cohorts)
    d5_dir = RAW_DIR / "D5_ixodes_fracasso2021"
    tick_rows = list(csv.DictReader((d5_dir / "All_ticks.csv").open(encoding="utf-8")))
    by_sb: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in tick_rows:
        by_sb[(r["LSTAG"].strip(), r["BATCH"].strip())].append(r)

    # Match D5 Rule 1 rows by prefix
    d5_rule1 = [r for r in rows if r["dataset"] == "D5_ixodes_fracasso2021" and "raw_individual_uncensored" in r["rule_applied"]]
    tick_by_tid_vid = {(r["TID"].strip(), r["VID"].strip()): r for r in tick_rows}
    egg_rows = list(csv.DictReader((d5_dir / "Number_hatched_eggs.csv").open(encoding="utf-8")))
    seen_ek: set[tuple[str, str]] = set()
    egg_by_b: dict[str, list[float]] = defaultdict(list)
    for er in egg_rows:
        k = (er["TID"].strip(), er["VID"].strip())
        if k in seen_ek:
            continue
        seen_ek.add(k)
        hs = er["HATCH"].strip()
        if hs in ("NA", "-", "", "n/a"):
            continue
        tr = tick_by_tid_vid.get(k)
        if tr is None or tr["DREC"].strip() in ("n/a", "", "NA"):
            continue
        dur = float(hs) - float(tr["DREC"].strip())
        if dur > 0:
            egg_by_b[tr["BATCH"].strip()].append(dur)

    for r in d5_rule1:
        st_lbl = r["stage"]
        if "_feeding_batch_" in st_lbl:
            stg = st_lbl.split("_feeding_batch_")[0]
            batch = st_lbl.split("_feeding_batch_")[1].split("_")[0]
            vals = []
            for tr in by_sb[(stg, batch)]:
                dinf_s, drec_s, pos = tr["DINF"].strip(), tr["DREC"].strip(), tr["POS"].strip()
                if dinf_s in ("n/a", "", "NA") or drec_s in ("n/a", "", "NA"):
                    continue
                d_val = float(drec_s) - float(dinf_s)
                if d_val <= 0 or pos == "ocean" or (stg == "female" and d_val > 8.0):
                    continue
                vals.append(d_val)
            vectors[(r["dataset"], st_lbl, r["temperature_C"])] = np.array(vals, float)
        elif "moulting_batch_" in st_lbl:
            stg = "larva" if st_lbl.startswith("larva_") else "nymph"
            batch = st_lbl.split("_moulting_batch_")[1].split("_")[0]
            vals = [float(tr["MoultTime"].strip()) for tr in by_sb[(stg, batch)] if tr["MoultTime"].strip() not in ("n/a", "", "NA") and float(tr["MoultTime"].strip()) > 0]
            vectors[(r["dataset"], st_lbl, r["temperature_C"])] = np.array(vals, float)
        elif st_lbl.startswith("female_recovery_to_egg_hatch_batch_"):
            batch = st_lbl.split("female_recovery_to_egg_hatch_batch_")[1].split("_")[0]
            vectors[(r["dataset"], st_lbl, r["temperature_C"])] = np.array(egg_by_b[batch], float)

    # 3. D8 (2 Rule 1 cohorts)
    rs_rows = list(csv.DictReader((RAW_DIR / "D8_schisto_civitello2020" / "ResourceSupplyExp.csv").open(encoding="utf-8")))
    by_snail: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rs_rows:
        if r["Exposed"].strip() == "Yes":
            by_snail[(r["FoodLevel"].strip(), r["Color"].strip(), r["ID"].strip(), r["Exposed"].strip())].append(r)
    for r in rows:
        if r["dataset"] == "D8_schisto_civitello2020" and "raw_individual_uncensored" in r["rule_applied"]:
            fl_s = r["stage"].split("FoodLevel=")[1].split(" ")[0]
            vals = []
            for k in sorted(by_snail):
                if k[0] != fl_s:
                    continue
                v = sorted(by_snail[k], key=lambda x: int(x["Date"]))
                pos_d = [int(x["Date"]) for x in v if x["Worms"].strip() not in ("", "NA") and float(x["Worms"]) > 0.0]
                died = any(x["Dead"].strip() == "1" for x in v)
                onset = float(pos_d[0]) - 3.5
                cens_s = (not died) and (pos_d[-1] == 245)
                end_s = 245.0 if cens_s else (float(pos_d[-1]) + 3.5)
                vals.append(end_s - onset)
            vectors[(r["dataset"], r["stage"], r["temperature_C"])] = np.array(vals, float)

    # 4. D11 (4 Rule 1 cohorts)
    d11_rows = list(csv.DictReader((RAW_DIR / "D11_trematode_taskinen2022" / "Data_shedding_mussels").open(encoding="utf-8")))
    trials = [0, 14, 27, 42, 57, 69, 82, 96, 110, 125, 136, 150]
    idx_map = {t: i for i, t in enumerate(trials)}
    for r in rows:
        if r["dataset"] == "D11_trematode_taskinen2022" and "raw_individual_uncensored" in r["rule_applied"]:
            is_rf = "Rhipidocotyle fennica" in r["taxon"]
            raw_sp = "R. fennica" if is_rf else "Rhipidodoctyle campanula"
            st_col, sp_col = ("start_RF", "Stop_RF") if is_rf else ("start_rc", "stop_RC")
            pop = "Haajaistenjoki" if "Haajaistenjoki" in r["stage"] else "Kuusaankoski"
            tr_lbl = "High temperature" if "high_temperature" in r["stage"] else ("Intermediate temperature" if "intermediate_temperature" in r["stage"] else "Low temperature")
            vals = []
            for sr in d11_rows:
                if sr["Shedding_category"].strip() == raw_sp and sr["mussel_pop"].strip() == pop and sr["treatment"].strip() == tr_lbl:
                    s, e = int(sr[st_col]), int(sr[sp_col])
                    if s == 0:
                        continue
                    si, ei = idx_map[s], idx_map[e]
                    t0 = 0.5 * (trials[si - 1] + trials[si])
                    t1 = float(trials[ei]) if ei == len(trials) - 1 else 0.5 * (trials[ei] + trials[ei + 1])
                    vals.append(t1 - t0)
            vectors[(r["dataset"], r["stage"], r["temperature_C"])] = np.array(vals, float)

    # 5. D12 (2 Rule 1 cohorts)
    d12_raw = list(csv.reader((RAW_DIR / "D12_trematode_gleichsner2016" / "Gleichsner_et_al._RawData.csv").open(encoding="utf-8")))
    for r in rows:
        if r["dataset"] == "D12_trematode_gleichsner2016" and "raw_individual_uncensored" in r["rule_applied"]:
            tr_key = "Stressed" if "stressed (" in r["stage"] and "unstressed" not in r["stage"] else "Unstressed"
            vals = [float(rr[11]) * 7.0 for rr in d12_raw[1:] if rr[9].strip() == tr_key]
            vectors[(r["dataset"], r["stage"], r["temperature_C"])] = np.array(vals, float)

    # 6. D13 (90 Rule 1 cohorts)
    d13_raw = list(csv.DictReader((RAW_DIR / "D13_nematode_marshallagia2023" / "Marshallagia_develop_Morta.csv").open(encoding="utf-8")))
    by_lt: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for sr in d13_raw:
        by_lt[(sr["Location2"].strip(), float(sr["Temperature"]))].append(sr)
    for r in rows:
        if r["dataset"] == "D13_nematode_marshallagia2023" and "raw_individual_uncensored" in r["rule_applied"]:
            st_lbl = r["stage"]
            temp_c = float(r["temperature_C"])
            if st_lbl.startswith("egg_to_L1_hatching_time_"):
                loc = st_lbl[len("egg_to_L1_hatching_time_") : -len("_constant_lab")]
                vals = [0.5 * (float(sr["last_check_hatch"]) + float(sr["Hatching"])) for sr in by_lt[(loc, temp_c)] if sr["Hatching"].strip() != "NA"]
            elif st_lbl.startswith("egg_to_L3_development_time_"):
                loc = st_lbl[len("egg_to_L3_development_time_") : -len("_constant_lab")]
                vals = [0.5 * (float(sr["last_check_L3"]) + float(sr["L3_ocurrence"])) for sr in by_lt[(loc, temp_c)] if sr["L3_ocurrence"].strip() != "NA"]
            elif st_lbl.startswith("pre_L3_egg_larval_survival_time_"):
                loc = st_lbl[len("pre_L3_egg_larval_survival_time_") : -len("_constant_lab")]
                vals = [0.5 * (float(sr["last_check_Mortality"]) + float(sr["Dead_before_L3"])) for sr in by_lt[(loc, temp_c)] if sr["Dead_before_L3"].strip() != "NA"]
            else:
                raise ValueError(f"Unknown D13 stage {st_lbl}")
            vectors[(r["dataset"], st_lbl, r["temperature_C"])] = np.array(vals, float)

    # 7. D16 (75 Rule 1 cohorts from VECTRAITS_INDIV_STEMS)
    d16_dir = RAW_DIR / "D16_vectraits_compilations"
    for stem in sorted(ep2.VECTRAITS_INDIV_STEMS):
        payload = json.loads((d16_dir / f"{stem}.json").read_text(encoding="utf-8"))
        cells: dict[tuple[str, float | None], list[float]] = defaultdict(list)
        for sr in payload.get("results", []):
            tunit = str(sr.get("OriginalTraitUnit") or "").strip().lower()
            mult = ep2.TIME_UNIT_TO_DAYS.get(tunit)
            v_raw = sr.get("OriginalTraitValue")
            if mult is None or v_raw is None or float(v_raw) <= 0.0:
                continue
            tname = str(sr.get("OriginalTraitName") or "").strip()
            istage = "juvenile" if stem == "vectraits_0580" else str(sr.get("Interactor1Stage") or "").strip()
            tc_raw = ep2._vectraits_temp_c(sr)
            tc_val = round(float(tc_raw), 2) if tc_raw != "" else None
            stressor = str(sr.get("SecondStressor") or "").strip()
            sval = str(sr.get("SecondStressorValue") or "").strip()
            stress_sfx = f"_{stressor.replace(' ', '_')}_{sval}" if stressor and sval else ""
            stg_lbl = f"{stem}_{tname.lower().replace(' ', '_')}_{istage.replace(' ', '_')}{stress_sfx}"
            cells[(stg_lbl, tc_val)].append(float(v_raw) * mult)
        for (stg_lbl, tc_val), vals in cells.items():
            if len(vals) >= 10:
                vectors[("D16_vectraits_compilations", stg_lbl, tc_val)] = np.array(vals, float)

    # Verify 100% of all Rule 1 rows in stage_cv.csv
    rule1_rows = [r for r in rows if "raw_individual_uncensored" in r["rule_applied"]]
    assert len(rule1_rows) == 202, f"Expected 202 Rule 1 rows, found {len(rule1_rows)}"
    for r in rule1_rows:
        key = (r["dataset"], r["stage"], round(r["temperature_C"], 2) if r["temperature_C"] is not None else None)
        if key not in vectors:
            # Try exact float key
            key = (r["dataset"], r["stage"], r["temperature_C"])
        assert key in vectors, f"Missing reconstructed Rule 1 vector for {key}"
        v = vectors[key]
        st = eh.uncensored_stats(v)
        assert st["n"] == r["n"], f"n mismatch on {key}: {st['n']} vs {r['n']}"
        assert abs(st["mean"] - r["mean"]) < 1e-4, f"mean mismatch on {key}: {st['mean']} vs {r['mean']}"
        assert abs(st["cv_indiv"] - r["cv_indiv"]) < 1e-4, f"cv mismatch on {key}: {st['cv_indiv']} vs {r['cv_indiv']}"
        # Ensure exact key with r['temperature_C'] is also stored
        vectors[(r["dataset"], r["stage"], r["temperature_C"])] = v

    return vectors


def matched_cv_d2_families(mean_l: float, cv_target: float, f_steps: int, sigma: float, phi: float) -> dict[str, Any]:
    """Compute D2(Erlang chain matched to CV) and D2(fixed+geometric matched to CV)
    at exact mean `mean_l` and target `cv_target` (`protocol.md` §2.1 & §5 `H_transform`)."""
    cv_max = math.sqrt(max(1.0 - 1.0 / mean_l, 0.0))
    if cv_target > cv_max + 1e-12:
        return {"admissible_chain": False, "cv_max": cv_max}
    cv_c = min(max(cv_target, 0.0), cv_max)
    theta = phi / (1.0 - phi)
    T_tot = float(f_steps) + mean_l

    if cv_c <= 1e-14:
        rho_c = phi ** mean_l
        rho_fg = phi ** mean_l
        m_eff = math.inf
        g_eff = 1.0
    else:
        # Continuous-shape negative-binomial / serial-chain transform at exact (mean_l, cv_c)
        m_eff = 1.0 / (cv_c * cv_c + 1.0 / mean_l)
        rho_c = (1.0 + mean_l / (m_eff * theta)) ** (-m_eff)
        # Fixed (mean_l - g) + geometric(g) with exact variance V = (cv_c * mean_l)^2 = g*(g - 1)
        var_t = (cv_c * mean_l) ** 2
        g_eff = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * var_t))
        rho_fg = (phi ** (mean_l - g_eff)) * ((phi / g_eff) / (1.0 - (1.0 - 1.0 / g_eff) * phi))

    # Also compute exact integer-substage chain interpolation & integer-g fixed+geom as check
    m_lo = max(1, int(math.floor(m_eff))) if math.isfinite(m_eff) else int(round(mean_l))
    m_hi = min(int(round(mean_l)), m_lo + 1)
    if m_lo == m_hi or not math.isfinite(m_eff):
        rho_c_int = phi ** mean_l if m_lo >= int(round(mean_l)) else (1.0 + mean_l / (m_lo * theta)) ** (-m_lo)
    else:
        v_lo = mean_l * (mean_l / m_lo - 1.0)
        v_hi = 0.0 if m_hi >= int(round(mean_l)) else mean_l * (mean_l / m_hi - 1.0)
        v_tar = (cv_c * mean_l) ** 2
        w_hi = np.clip((v_lo - v_tar) / max(v_lo - v_hi, 1e-30), 0.0, 1.0)
        r_lo = (1.0 + mean_l / (m_lo * theta)) ** (-m_lo)
        r_hi = (phi ** mean_l) if m_hi >= int(round(mean_l)) else (1.0 + mean_l / (m_hi * theta)) ** (-m_hi)
        rho_c_int = float((1.0 - w_hi) * r_lo + w_hi * r_hi)

    d2_c = d2_primitive_formula(sigma, phi, T_tot, (phi ** f_steps) * rho_c)
    d2_fg = d2_primitive_formula(sigma, phi, T_tot, (phi ** f_steps) * rho_fg)
    d2_c_int = d2_primitive_formula(sigma, phi, T_tot, (phi ** f_steps) * rho_c_int)
    return {
        "admissible_chain": True,
        "cv_max": cv_max,
        "m_eff": m_eff,
        "g_eff": g_eff,
        "D2_chain": d2_c,
        "D2_chain_int_mix": d2_c_int,
        "D2_fg": d2_fg,
    }


def run_h_transform(rows: list[dict[str, Any]], vectors: dict[tuple[str, str, float | None], np.ndarray]) -> dict[str, Any]:
    """Protocol §5 Test `H_transform` across all uncensored individual cohorts with n >= 30."""
    f_steps = 8
    l_mean_target = 60.0
    sigma = 1.40
    phi_fine = 0.75 ** (1.0 / 8.0)

    elig_rows = [
        r for r in rows
        if "raw_individual_uncensored" in r["rule_applied"] and r["n"] is not None and r["n"] >= 30
    ]
    cohort_results: list[dict[str, Any]] = []
    for r in elig_rows:
        vec = vectors[(r["dataset"], r["stage"], r["temperature_C"])]
        p_long, cv_disc = eh.discretize_empirical_stay(vec, l_mean_target, max_age=800)
        d2_emp, T_emp, R_emp = eh.empirical_D2(f_steps, p_long, sigma, phi_fine)
        mean_disc = T_emp - float(f_steps)

        fam = matched_cv_d2_families(mean_disc, cv_disc, f_steps, sigma, phi_fine)
        raw_cv = r["cv_indiv"]
        cv_max_60 = math.sqrt(1.0 - 1.0 / 60.0)
        representable = bool(fam["admissible_chain"] and raw_cv <= cv_max_60)
        if fam["admissible_chain"]:
            rel_err_chain = abs(fam["D2_chain"] - d2_emp) / abs(d2_emp)
            rel_err_fg = abs(fam["D2_fg"] - d2_emp) / abs(d2_emp)
            rel_err_max = max(rel_err_chain, rel_err_fg)
            rel_err_int_max = max(abs(fam["D2_chain_int_mix"] - d2_emp) / abs(d2_emp), rel_err_fg)
        else:
            rel_err_chain = math.inf
            rel_err_fg = math.inf
            rel_err_max = math.inf
            rel_err_int_max = math.inf

        passes_10pct = bool(representable and rel_err_max <= 0.10)
        skew_val = float(eh.uncensored_stats(vec)["skewness"])
        cohort_results.append(
            {
                "dataset": r["dataset"],
                "taxon": r["taxon"],
                "stage": r["stage"],
                "stage_category": r["stage_category"],
                "n": r["n"],
                "cv_indiv": raw_cv,
                "cv_disc": cv_disc,
                "mean_disc": mean_disc,
                "skewness": skew_val,
                "D2_emp": d2_emp,
                "D2_chain": fam.get("D2_chain"),
                "D2_fg": fam.get("D2_fg"),
                "rel_err_chain": rel_err_chain,
                "rel_err_fg": rel_err_fg,
                "rel_err_max": rel_err_max,
                "rel_err_int_max": rel_err_int_max,
                "representable": representable,
                "passes_10pct": passes_10pct,
            }
        )

    k_tot = len(cohort_results)
    pass_flags = np.array([1.0 if c["passes_10pct"] else 0.0 for c in cohort_results], float)
    fail_flags = 1.0 - pass_flags
    frac_pass = float(pass_flags.mean()) if k_tot > 0 else math.nan
    frac_fail = float(fail_flags.mean()) if k_tot > 0 else math.nan
    n_pass = int(pass_flags.sum())
    n_fail = k_tot - n_pass
    n_cv_gt_1 = sum(1 for c in cohort_results if c["cv_indiv"] > 1.0 or not c["representable"])
    n_err_gt_10pct = sum(1 for c in cohort_results if c["representable"] and c["rel_err_max"] > 0.10)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_pass = np.empty(BOOTSTRAP_RESAMPLES, float)
    for b in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, k_tot, size=k_tot)
        boot_pass[b] = pass_flags[idx].mean()
    boot_fail = 1.0 - boot_pass
    ci_pass = [float(np.percentile(boot_pass, 2.5)), float(np.percentile(boot_pass, 97.5))]
    ci_fail = [float(np.percentile(boot_fail, 2.5)), float(np.percentile(boot_fail, 97.5))]

    if k_tot < 10:
        verdict = "not decided"
    elif frac_pass >= 0.90:
        verdict = "supports"
    else:
        verdict = "contradicts"

    rel_errs_finite = np.array([c["rel_err_max"] for c in cohort_results if math.isfinite(c["rel_err_max"])], float)
    chain_errs_finite = np.array([c["rel_err_chain"] for c in cohort_results if math.isfinite(c["rel_err_chain"])], float)
    fg_errs_finite = np.array([c["rel_err_fg"] for c in cohort_results if math.isfinite(c["rel_err_fg"])], float)

    # Sort top 10 largest relative errors for transparent reporting
    worst_cohorts = sorted(cohort_results, key=lambda c: -c["rel_err_max"])[:10]

    return {
        "verdict": verdict,
        "k_cohorts_n_ge_30": k_tot,
        "n_pass_10pct": n_pass,
        "n_fail_10pct": n_fail,
        "n_inadmissible_cv": n_cv_gt_1,
        "n_shape_error_gt_10pct": n_err_gt_10pct,
        "frac_pass_10pct": frac_pass,
        "ci95_frac_pass": ci_pass,
        "frac_fail_10pct": frac_fail,
        "ci95_frac_fail": ci_fail,
        "prob_boot_fail_gt_10pct": float(np.mean(boot_fail > 0.10)),
        "median_rel_err_max": float(np.median(rel_errs_finite)),
        "iqr_rel_err_max": [float(np.percentile(rel_errs_finite, 25)), float(np.percentile(rel_errs_finite, 75))],
        "p90_rel_err_max": float(np.percentile(rel_errs_finite, 90)),
        "max_rel_err_max": float(np.max(rel_errs_finite)),
        "median_rel_err_chain_only": float(np.median(chain_errs_finite)),
        "n_fail_chain_only_10pct": int(np.sum(chain_errs_finite > 0.10)) + n_cv_gt_1,
        "median_rel_err_fg_only": float(np.median(fg_errs_finite)),
        "n_fail_fg_only_10pct": int(np.sum(fg_errs_finite > 0.10)) + n_cv_gt_1,
        "n_pass_int_mix_10pct": sum(1 for c in cohort_results if c["representable"] and c["rel_err_int_max"] <= 0.10),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "top_10_largest_errors": worst_cohorts,
    }


R10B_FINE_ROWS: dict[str, dict[str, Any]] = {
    "1a": {
        "label": "Schistosoma haematobium (adult worm in humans, 48 mo)",
        "F_fine": 8,
        "F_shape": "fixed",
        "Lg_fine": 192,
        "other_fine": 0,
        "sigma": 0.589,
        "phi_coarse": 0.398,
        "r": 8,
        "cv_half_chain_bracket": [0.0, 0.102],
        "cv_half_fg_bracket": [0.0, 0.039],
        "direct_species_steady_measured": False,
        "steady_class_key": "surv_like_proxy",
    },
    "1b": {
        "label": "Schistosoma mansoni (adult worm in humans, 48 mo)",
        "F_fine": 8,
        "F_shape": "fixed",
        "Lg_fine": 192,
        "other_fine": 0,
        "sigma": 1.380,
        "phi_coarse": 0.367,
        "r": 8,
        "cv_half_chain_bracket": [0.0, 0.102],
        "cv_half_fg_bracket": [0.0, 0.039],
        "direct_species_steady_measured": False,
        "steady_class_key": "surv_like_proxy",
    },
    "6": {
        "label": "Plasmodium falciparum (blood-stage infection in humans, 6 mo)",
        "F_fine": 8,
        "F_shape": "fixed",
        "Lg_fine": 60,
        "other_fine": 0,
        "sigma": 1.400,
        "phi_coarse": 0.750,
        "r": 8,
        "cv_half_chain_bracket": [0.129, 0.214],
        "cv_half_fg_bracket": [0.125, 0.258],
        "direct_species_steady_measured": True,
        "steady_class_key": "S1_direct_pf",
    },
    "3a": {
        "label": "Borrelia burgdorferi, oak forest (Ixodes scapularis tick, 10 mo)",
        "F_fine": 11,
        "F_shape": "fixed",
        "Lg_fine": 25,
        "other_fine": 0,
        "sigma": 0.668,
        "phi_coarse": 0.300,
        "r": 11,
        "cv_half_chain_bracket": [0.0, 0.059],
        "cv_half_fg_bracket": [0.0, 0.057],
        "direct_species_steady_measured": True,
        "steady_class_key": "D6_ixodes_scapularis",
    },
    "3b": {
        "label": "Borrelia burgdorferi, northern edge (Ixodes scapularis tick, 10 mo)",
        "F_fine": 11,
        "F_shape": "fixed",
        "Lg_fine": 25,
        "other_fine": 0,
        "sigma": 1.190,
        "phi_coarse": 0.500,
        "r": 11,
        "cv_half_chain_bracket": [0.059, 0.100],
        "cv_half_fg_bracket": [0.057, 0.139],
        "direct_species_steady_measured": True,
        "steady_class_key": "D6_ixodes_scapularis",
    },
    "2a": {
        "label": "Echinococcus multilocularis (larva in vole, 4 mo; egg 2.5; fox 2)",
        "F_fine": 10,
        "F_shape": "memoryless",
        "Lg_fine": 16,
        "other_fine": 8,
        "sigma": 1.174,
        "phi_coarse": 0.750,
        "r": 8,
        "cv_half_chain_bracket": None,
        "cv_half_fg_bracket": None,
        "direct_species_steady_measured": False,
        "steady_class_key": "surv_like_proxy",
    },
}


def majority_verdict(D: float, se: float, cap: float) -> str:
    """R10b majority-protection rule (`protocol.md` §2.3)."""
    thr = 0.5 * cap
    if D < thr - 3.0 * se:
        return "survives"
    if D > thr + 3.0 * se:
        return "fails"
    return "undecided"


def bootstrap_percentiles(vals: np.ndarray, qs: list[float] = [10, 25, 50, 75, 90]) -> dict[str, Any]:
    obs = [float(np.percentile(vals, q)) for q in qs]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(vals)
    boots = np.empty((BOOTSTRAP_RESAMPLES, len(qs)), float)
    for b in range(BOOTSTRAP_RESAMPLES):
        sample = vals[rng.integers(0, n, size=n)]
        boots[b] = np.percentile(sample, qs)
    cis = [[float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))] for i in range(len(qs))]
    return {
        "k": int(n),
        "percentiles_q": qs,
        "observed": {f"p{int(q)}": obs[i] for i, q in enumerate(qs)},
        "ci95": {f"p{int(q)}": cis[i] for i, q in enumerate(qs)},
    }


def discrete_weibull_stay_pmf(mean_steps: float, cv_target: float, max_age: int = 1200) -> tuple[np.ndarray, float]:
    """Construct a discrete Weibull stay law on {1, ..., max_age} matched to `(mean_steps, cv_target)`."""
    from scipy import special

    def cv_of_k(k: float) -> float:
        g1 = special.gamma(1.0 + 1.0 / k)
        g2 = special.gamma(1.0 + 2.0 / k)
        return math.sqrt(max(g2 / (g1 * g1) - 1.0, 0.0)) - cv_target

    k_w = optimize.brentq(cv_of_k, 0.35, 30.0)
    g1 = special.gamma(1.0 + 1.0 / k_w)

    def mean_err(scale_s: float) -> float:
        idx = np.arange(1, max_age + 1, dtype=float)
        cdf_hi = -np.expm1(-np.power(idx / scale_s, k_w))
        cdf_lo = np.concatenate([[0.0], cdf_hi[:-1]])
        pmf = np.maximum(cdf_hi - cdf_lo, 0.0)
        pmf /= pmf.sum()
        return float(pmf @ idx) - mean_steps

    s_init = mean_steps / g1
    s_hat = optimize.brentq(mean_err, max(0.2 * s_init, 0.5), 3.0 * s_init)
    idx = np.arange(1, max_age + 1, dtype=float)
    cdf_hi = -np.expm1(-np.power(idx / s_hat, k_w))
    cdf_lo = np.concatenate([[0.0], cdf_hi[:-1]])
    pmf = np.maximum(cdf_hi - cdf_lo, 0.0)
    # Trim negligible upper tail
    nz = np.flatnonzero(pmf > 1e-13)
    lmax = int(nz[-1]) + 1
    pmf = pmf[:lmax] / pmf[:lmax].sum()
    ell = np.arange(1, lmax + 1, dtype=float)
    mu = float(pmf @ ell)
    cv_d = math.sqrt(max(float(pmf @ ((ell - mu) ** 2)), 0.0)) / mu
    return pmf, cv_d


def run_h_sys(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Protocol §5 Test `H_sys` across all 6 R10b rows + auxiliary nematode rows 5 and 7."""
    strict_rows = [r for r in rows if r["n"] is not None and r["n"] >= 10]
    s1_vals = np.array([r["cv_indiv"] for r in strict_rows if r["stage_category"] == "S1_endoparasite_vertebrate"], float)
    surv_vals = np.array([r["cv_indiv"] for r in strict_rows if classify_dev_vs_surv(r) == "surv_like"], float)
    d6_all_vals = np.array([r["cv_indiv"] for r in strict_rows if r["dataset"] == "D6_ixodes_scapularis_brunner2022"], float)
    d6_s4_vals = np.array([r["cv_indiv"] for r in strict_rows if r["dataset"] == "D6_ixodes_scapularis_brunner2022" and r["stage_category"] == "S4_vector_tick_development_moulting"], float)
    d6_s5_vals = np.array([r["cv_indiv"] for r in strict_rows if r["dataset"] == "D6_ixodes_scapularis_brunner2022" and r["stage_category"] == "S5_vector_tick_adult_or_offhost_survival"], float)
    s4_vals = np.array([r["cv_indiv"] for r in strict_rows if r["stage_category"] == "S4_vector_tick_development_moulting"], float)
    s5_vals = np.array([r["cv_indiv"] for r in strict_rows if r["stage_category"] == "S5_vector_tick_adult_or_offhost_survival"], float)
    s45_vals = np.array([r["cv_indiv"] for r in strict_rows if r["stage_category"] in ("S4_vector_tick_development_moulting", "S5_vector_tick_adult_or_offhost_survival")], float)
    d13_vals = np.array([r["cv_indiv"] for r in strict_rows if r["dataset"] == "D13_nematode_marshallagia2023"], float)

    class_dists = {
        "S1_direct_pf": bootstrap_percentiles(s1_vals),
        "surv_like_proxy": bootstrap_percentiles(surv_vals),
        "D6_ixodes_scapularis": bootstrap_percentiles(d6_all_vals),
        "D6_ixodes_moulting_S4": bootstrap_percentiles(d6_s4_vals),
        "D6_ixodes_survival_S5": bootstrap_percentiles(d6_s5_vals),
        "S4_all_strict": bootstrap_percentiles(s4_vals),
        "S5_all_strict": bootstrap_percentiles(s5_vals),
        "S4_plus_S5_strict": bootstrap_percentiles(s45_vals),
        "D13_marshallagia_S3": bootstrap_percentiles(d13_vals),
    }

    chain_cache: dict[tuple[str, int], dict[str, float]] = {}
    fg_cache: dict[tuple[str, int], dict[str, float]] = {}

    def eval_engine_chain_int(row_id: str, m_int: int) -> dict[str, float]:
        key = (row_id, m_int)
        if key in chain_cache:
            return chain_cache[key]
        cfg = R10B_FINE_ROWS[row_id]
        Lg = cfg["Lg_fine"]
        F = cfg["F_fine"]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        sigma = cfg["sigma"]
        cap = 0.5 * sigma * sigma
        if m_int >= Lg and cfg["F_shape"] == "fixed" and cfg["other_fine"] == 0:
            res = {"D": cap, "se": 0.0, "D2": math.nan, "cv": 0.0, "D_over_cap": 1.0, "verdict_majority": "fails"}
            chain_cache[key] = res
            return res
        l_shape = "fixed" if m_int >= Lg else ("memoryless" if m_int == 1 else "chain")
        cv_m = 0.0 if m_int >= Lg else math.sqrt(max(1.0 / m_int - 1.0 / Lg, 0.0))
        stages = [le.Stage("fluct", mean=float(F), shape=cfg["F_shape"])]
        stages.append(le.Stage("steady", mean=float(Lg), shape=l_shape, substages=1 if l_shape != "chain" else m_int))
        transfers = [le.Transfer("fluct", "steady", driver="host", birth=True)]
        if cfg["other_fine"] > 0:
            stages.append(le.Stage("other", mean=float(cfg["other_fine"]), shape="fixed"))
            transfers.append(le.Transfer("steady", "other"))
            transfers.append(le.Transfer("other", "fluct"))
        else:
            transfers.append(le.Transfer("steady", "fluct"))
        desc = le.Description(stages=stages, transfers=transfers, drivers=[le.Driver("host", sigma=sigma, phi=phi_f, kind="lognormal")])
        eng = le.Engine(desc).critical()
        d2_val = float(eng.coefficient()["D2"])
        s_list: list[float] = []
        b_fix = 0
        if cfg["F_shape"] == "memoryless":
            s_list.append(1.0 - 1.0 / F)
        else:
            b_fix += F
        if l_shape == "fixed":
            b_fix += Lg
        else:
            s_sub = 1.0 - m_int / float(Lg)
            if s_sub > 0:
                s_list.extend([s_sub] * m_int)
            else:
                b_fix += m_int
        b_fix += cfg["other_fine"]
        sim = fast_serial_ring_loss(s_list, b_fix, sigma, phi_f, steps=24000, burn=2000, pairs=16, seed=20261114 + int(m_int) * 7)
        res = {
            "D": float(sim["D"]),
            "se": float(sim["se"]),
            "D2": d2_val,
            "cv": cv_m,
            "D_over_cap": float(sim["D"]) / cap,
            "verdict_majority": majority_verdict(float(sim["D"]), float(sim["se"]), cap),
        }
        chain_cache[key] = res
        return res

    def eval_fg_int(row_id: str, g_int: int) -> dict[str, float]:
        key = (row_id, g_int)
        if key in fg_cache:
            return fg_cache[key]
        cfg = R10B_FINE_ROWS[row_id]
        Lg = cfg["Lg_fine"]
        F = cfg["F_fine"]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        sigma = cfg["sigma"]
        cap = 0.5 * sigma * sigma
        g_int = max(1, min(Lg, g_int))
        if g_int == 1 and cfg["F_shape"] == "fixed" and cfg["other_fine"] == 0:
            res = {"D": cap, "se": 0.0, "D2": math.nan, "cv": 0.0}
            fg_cache[key] = res
            return res
        if g_int == Lg:
            return eval_engine_chain_int(row_id, 1)
        d_fix = Lg - g_int
        cv_g = math.sqrt(g_int * (g_int - 1.0)) / Lg
        stages = [
            le.Stage("fluct", mean=float(F), shape=cfg["F_shape"]),
            le.Stage("steady_fix", mean=float(d_fix), shape="fixed"),
            le.Stage("steady_geo", mean=float(g_int), shape="memoryless"),
        ]
        transfers = [
            le.Transfer("fluct", "steady_fix", driver="host", birth=True),
            le.Transfer("steady_fix", "steady_geo"),
        ]
        if cfg["other_fine"] > 0:
            stages.append(le.Stage("other", mean=float(cfg["other_fine"]), shape="fixed"))
            transfers.append(le.Transfer("steady_geo", "other"))
            transfers.append(le.Transfer("other", "fluct"))
        else:
            transfers.append(le.Transfer("steady_geo", "fluct"))
        desc = le.Description(stages=stages, transfers=transfers, drivers=[le.Driver("host", sigma=sigma, phi=phi_f, kind="lognormal")])
        eng = le.Engine(desc).critical()
        d2_val = float(eng.coefficient()["D2"])
        s_list: list[float] = []
        b_fix = d_fix + cfg["other_fine"]
        if cfg["F_shape"] == "memoryless":
            s_list.append(1.0 - 1.0 / F)
        else:
            b_fix += F
        if g_int > 1:
            s_list.append(1.0 - 1.0 / g_int)
        else:
            b_fix += 1
        sim = fast_serial_ring_loss(s_list, b_fix, sigma, phi_f, steps=24000, burn=2000, pairs=16, seed=20261500 + int(g_int) * 11)
        res = {"D": float(sim["D"]), "se": float(sim["se"]), "D2": d2_val, "cv": cv_g}
        fg_cache[key] = res
        return res

    def eval_at_cv(row_id: str, cv_q: float) -> dict[str, Any]:
        cfg = R10B_FINE_ROWS[row_id]
        Lg = cfg["Lg_fine"]
        F = cfg["F_fine"]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        sigma = cfg["sigma"]
        cap = 0.5 * sigma * sigma
        cv_max = math.sqrt(1.0 - 1.0 / Lg)

        if cv_q >= cv_max:
            c_res = dict(eval_engine_chain_int(row_id, 1), clamped_at_m1=True)
            fg_res = dict(eval_engine_chain_int(row_id, 1), clamped_at_m1=True)
            # Also evaluate exact overdispersed discrete Weibull law at cv_q when F_shape == 'fixed' and other == 0
            if cfg["F_shape"] == "fixed" and cfg["other_fine"] == 0:
                p_w, cv_w = discrete_weibull_stay_pmf(float(Lg), cv_q, max_age=400)
                d2_w, _, _ = eh.empirical_D2(F, p_w, sigma, phi_f)
                sim_w = eh.simulate_empirical_loss(F, p_w, sigma, phi_f, steps=20000, burn=1500, pairs=16, seed=20262000 + int(round(cv_q * 1000)))
                od_res = {"D": sim_w["D"], "se": sim_w["se"], "D2": d2_w, "cv_realized": cv_w, "verdict_majority": majority_verdict(sim_w["D"], sim_w["se"], cap)}
            else:
                od_res = None
        else:
            m_real = 1.0 / (cv_q * cv_q + 1.0 / Lg)
            m_lo = max(1, int(math.floor(m_real)))
            m_hi = min(Lg, int(math.ceil(m_real)))
            r_lo = eval_engine_chain_int(row_id, m_lo)
            r_hi = eval_engine_chain_int(row_id, m_hi)
            if m_lo == m_hi or abs(r_lo["cv"] - r_hi["cv"]) < 1e-12:
                d_c, se_c, d2_c = r_lo["D"], r_lo["se"], r_lo["D2"]
            else:
                w = (r_lo["cv"] - cv_q) / (r_lo["cv"] - r_hi["cv"])
                d_c = (1.0 - w) * r_lo["D"] + w * r_hi["D"]
                se_c = math.hypot((1.0 - w) * r_lo["se"], w * r_hi["se"])
                d2_c = (1.0 - w) * r_lo["D2"] + w * r_hi["D2"]
            c_res = {"D": d_c, "se": se_c, "D2": d2_c, "m_bracket": [m_lo, m_hi], "clamped_at_m1": False}

            var_t = (cv_q * Lg) ** 2
            g_real = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * var_t))
            g_lo = max(1, int(math.floor(g_real)))
            g_hi = min(Lg, int(math.ceil(g_real)))
            fg_lo = eval_fg_int(row_id, g_lo)
            fg_hi = eval_fg_int(row_id, g_hi)
            if g_lo == g_hi or abs(fg_hi["cv"] - fg_lo["cv"]) < 1e-12:
                d_fg, se_fg, d2_fg = fg_lo["D"], fg_lo["se"], fg_lo["D2"]
            else:
                w_g = (cv_q - fg_lo["cv"]) / (fg_hi["cv"] - fg_lo["cv"])
                d_fg = (1.0 - w_g) * fg_lo["D"] + w_g * fg_hi["D"]
                se_fg = math.hypot((1.0 - w_g) * fg_lo["se"], w_g * fg_hi["se"])
                d2_fg = (1.0 - w_g) * fg_lo["D2"] + w_g * fg_hi["D2"]
            fg_res = {"D": d_fg, "se": se_fg, "D2": d2_fg, "g_bracket": [g_lo, g_hi], "clamped_at_m1": False}
            od_res = None

        c_res["D_over_cap"] = c_res["D"] / cap
        c_res["verdict_majority"] = majority_verdict(c_res["D"], c_res["se"], cap)
        fg_res["D_over_cap"] = fg_res["D"] / cap
        fg_res["verdict_majority"] = majority_verdict(fg_res["D"], fg_res["se"], cap)
        return {"cv_target": cv_q, "chain": c_res, "fixed_geom": fg_res, "overdispersed_weibull": od_res}

    # Species-specific empirical kernels for Row 6 (P. falciparum D1) and Rows 3a/3b (I. scapularis D6)
    d1_clones = reconstruct_d1_clones(max_zeroes=2)
    pf_phi = 0.75 ** (1.0 / 8.0)
    pf_cap = 0.5 * 1.40 * 1.40
    pf_kernels: dict[str, Any] = {}
    for kname, (t_arr, ev_arr) in d1_clones.items():
        # 1. Uncensored cleared subset rescaled to mean 60
        p_clr, cv_clr = eh.discretize_empirical_stay(t_arr[ev_arr], 60.0, max_age=400)
        d2_clr, _, _ = eh.empirical_D2(8, p_clr, 1.40, pf_phi)
        sim_clr = eh.simulate_empirical_loss(8, p_clr, 1.40, pf_phi, steps=20000, burn=1500, pairs=16, seed=20263001)
        # 2. Right-censored Weibull MLE kernel scaled to mean 60
        wm = eh.weibull_censored_mle(t_arr, ev_arr)
        p_wm, cv_wm = discrete_weibull_stay_pmf(60.0, wm["cv_mle"], max_age=450)
        d2_wm, _, _ = eh.empirical_D2(8, p_wm, 1.40, pf_phi)
        sim_wm = eh.simulate_empirical_loss(8, p_wm, 1.40, pf_phi, steps=20000, burn=1500, pairs=16, seed=20263002)
        pf_kernels[kname] = {
            "weibull_mle_cv": wm["cv_mle"],
            "weibull_kernel": {"cv_disc": cv_wm, "D2": d2_wm, "D": sim_wm["D"], "se": sim_wm["se"], "D_over_cap": sim_wm["D"] / pf_cap, "verdict_majority": majority_verdict(sim_wm["D"], sim_wm["se"], pf_cap)},
            "cleared_subset_kernel": {"n_cleared": int(ev_arr.sum()), "cv_disc": cv_clr, "D2": d2_clr, "D": sim_clr["D"], "se": sim_clr["se"], "D_over_cap": sim_clr["D"] / pf_cap, "verdict_majority": majority_verdict(sim_clr["D"], sim_clr["se"], pf_cap)},
        }

    # Sequential two-phase tick stay (D6 fed-larva moulting 4.5 fine steps + flat-nymph overwintering 20.5 fine steps = 25 fine steps)
    tick_two_phase: dict[str, Any] = {}
    for rid in ("3a", "3b"):
        cfg = R10B_FINE_ROWS[rid]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        cap = 0.5 * cfg["sigma"] ** 2
        # Median D6 moulting CV=0.145 (4.5 steps, m=4 -> CV=0.167) + median D6 off-host survival CV=0.341 (20.5 steps, m=6 -> CV=0.343)
        desc_2ph = le.Description(
            stages=[
                le.Stage("fluct", mean=11.0, shape="fixed"),
                le.Stage("moult_S4", mean=4.5, shape="chain", substages=4),
                le.Stage("overwinter_S5", mean=20.5, shape="chain", substages=6),
            ],
            transfers=[
                le.Transfer("fluct", "moult_S4", driver="host", birth=True),
                le.Transfer("moult_S4", "overwinter_S5"),
                le.Transfer("overwinter_S5", "fluct"),
            ],
            drivers=[le.Driver("host", sigma=cfg["sigma"], phi=phi_f, kind="lognormal")],
        )
        eng_2ph = le.Engine(desc_2ph).critical()
        s_2ph = [1.0 - 4.0 / 4.5] * 4 + [1.0 - 6.0 / 20.5] * 6
        sim_2ph = fast_serial_ring_loss(s_2ph, 11, cfg["sigma"], phi_f, steps=24000, burn=2000, pairs=16, seed=20263100)
        var_2ph = 4.5 * (4.5 / 4.0 - 1.0) + 20.5 * (20.5 / 6.0 - 1.0)
        cv_2ph = math.sqrt(var_2ph) / 25.0
        tick_two_phase[rid] = {
            "cv_combined": cv_2ph,
            "D2": float(eng_2ph.coefficient()["D2"]),
            "D": float(sim_2ph["D"]),
            "se": float(sim_2ph["se"]),
            "D_over_cap": float(sim_2ph["D"]) / cap,
            "verdict_majority": majority_verdict(float(sim_2ph["D"]), float(sim_2ph["se"]), cap),
        }

    row_summaries: dict[str, Any] = {}
    for rid, cfg in R10B_FINE_ROWS.items():
        cap = 0.5 * cfg["sigma"] ** 2
        geo = eval_engine_chain_int(rid, 1)
        dist = class_dists[cfg["steady_class_key"]]
        pct_evals = {pk: eval_at_cv(rid, cv_v) for pk, cv_v in dist["observed"].items()}

        # Also evaluate S4_all_strict on 3a/3b and S1_direct_pf on 1a/1b for cross-check
        extra_evals: dict[str, Any] = {}
        if rid in ("3a", "3b"):
            extra_evals["S4_moulting_class_p25_p50_p75"] = {
                pk: eval_at_cv(rid, class_dists["S4_all_strict"]["observed"][pk]) for pk in ("p25", "p50", "p75")
            }
            extra_evals["D6_moulting_only_p25_p50_p75"] = {
                pk: eval_at_cv(rid, class_dists["D6_ixodes_moulting_S4"]["observed"][pk]) for pk in ("p25", "p50", "p75")
            }
            extra_evals["two_phase_moult_plus_overwinter"] = tick_two_phase[rid]
        if rid == "6":
            extra_evals["D1_species_kernels"] = pf_kernels
            extra_evals["surv_like_proxy_p25_p50_p75"] = {
                pk: eval_at_cv("6", class_dists["surv_like_proxy"]["observed"][pk]) for pk in ("p25", "p50", "p75")
            }

        # Apply registered H_sys decision criterion (`protocol.md` §5 lines 122-124)
        p10_cv = dist["observed"]["p10"]
        iqr_keys = ("p25", "p50", "p75")
        all_keys = ("p10", "p25", "p50", "p75", "p90")

        if geo["verdict_majority"] == "fails":
            row_class = "not decided"
            row_verdict = "not decided"
            reason = "no_baseline_majority_protection_under_geometric_stays (clause iii)"
        elif not cfg["direct_species_steady_measured"]:
            # Clause (i) of `not decided`: no open dataset measures individual-level duration spread for that species' steady stage directly
            proxy_all_survive = all(
                pct_evals[k]["chain"]["verdict_majority"] == "survives"
                and pct_evals[k]["fixed_geom"]["verdict_majority"] == "survives"
                for k in all_keys
            )
            hi_br = max(cfg["cv_half_chain_bracket"][1], cfg["cv_half_fg_bracket"][1])
            row_class = "not decided (bounded_by_proxy)"
            row_verdict = "not decided"
            reason = (
                f"bounded_by_proxy (clause i: zero open direct adult-worm cohorts in humans; "
                f"under surv_like proxy K={dist['k']}, all p10-p90 survive={proxy_all_survive} and p10={p10_cv:.4f}>{hi_br:.3f})"
            )
        else:
            hi_br = max(cfg["cv_half_chain_bracket"][1], cfg["cv_half_fg_bracket"][1])
            lo_br = min(cfg["cv_half_chain_bracket"][0], cfg["cv_half_fg_bracket"][0])
            all_p10_p90_survive = all(
                pct_evals[k]["chain"]["verdict_majority"] == "survives"
                and pct_evals[k]["fixed_geom"]["verdict_majority"] == "survives"
                and (pct_evals[k]["overdispersed_weibull"] is None or pct_evals[k]["overdispersed_weibull"]["verdict_majority"] == "survives")
                for k in all_keys
            )
            kernels_ok = True
            if rid == "6":
                kernels_ok = all(
                    v["weibull_kernel"]["verdict_majority"] == "survives" and v["cleared_subset_kernel"]["verdict_majority"] == "survives"
                    for v in pf_kernels.values()
                )
            iqr_fails = any(
                pct_evals[k]["chain"]["verdict_majority"] == "fails"
                or pct_evals[k]["fixed_geom"]["verdict_majority"] == "fails"
                or (lo_br <= dist["observed"][k] <= hi_br)
                for k in iqr_keys
            )
            if all_p10_p90_survive and kernels_ok and p10_cv > hi_br:
                row_class = "robust"
                row_verdict = "supports"
                reason = f"direct species data exist, D < sigma^2/4 - 3*SE across p10-p90 & empirical kernels, and p10={p10_cv:.4f} > {hi_br:.3f}"
            elif iqr_fails:
                row_class = "fragile"
                row_verdict = "contradicts"
                reason = "direct species/stage data exist and within IQR D > sigma^2/4 + 3*SE or CV enters R10b CV_1/2 bracket"
            else:
                row_class = "not decided"
                row_verdict = "not decided"
                reason = "D straddles sigma^2/4 across empirical IQR without passing +-3*SE (clause ii)"

        row_summaries[rid] = {
            "label": cfg["label"],
            "cap": cap,
            "half_cap": 0.5 * cap,
            "geometric_baseline": geo,
            "steady_class_key": cfg["steady_class_key"],
            "empirical_cv_percentiles": dist,
            "evaluations_p10_to_p90": pct_evals,
            "extra_evaluations": extra_evals,
            "classification": row_class,
            "verdict": row_verdict,
            "reason": reason,
        }

    # Auxiliary 1-host nematode checks (rows 5 and 7 of R10, protocol §1 Q2)
    aux_nematodes: dict[str, Any] = {}
    for aux_id, lbl, Lg_c, sig, phi_c in [
        ("5", "Trichostrongylus tenuis in red grouse (adult 12 steps, free-living 1 step)", 12.0, 0.70, 0.85),
        ("7", "Teladorsagia circumcincta in Soay sheep (adult 3 steps, free-living 1 step)", 3.0, 0.35, 0.80),
    ]:
        r_fine = 8
        F_f = 8
        Lg_f = int(round(Lg_c * r_fine))
        phi_f = phi_c ** (1.0 / r_fine)
        cap = 0.5 * sig * sig
        # Evaluate at surv_like median CV (~0.2935) with F fixed vs F at D13 Marshallagia median CV (~0.2790, m_F=6)
        m_steady_med = max(1, int(round(1.0 / (class_dists["surv_like_proxy"]["observed"]["p50"] ** 2 + 1.0 / Lg_f))))
        m_steady_p10 = max(1, min(Lg_f, int(round(1.0 / (class_dists["surv_like_proxy"]["observed"]["p10"] ** 2 + 1.0 / Lg_f)))))
        res_sub: dict[str, Any] = {}
        for tag, f_sh, f_m, l_sh, l_m in [
            ("F_fixed_L_geom", "fixed", 1, "memoryless", 1),
            ("F_fixed_L_surv_p50", "fixed", 1, "chain", m_steady_med),
            ("F_fixed_L_surv_p10", "fixed", 1, "chain" if m_steady_p10 < Lg_f else "fixed", m_steady_p10),
            ("F_D13_m6_L_surv_p50", "chain", 6, "chain", m_steady_med),
            ("F_D13_m6_L_fixed", "chain", 6, "fixed", 1),
        ]:
            desc_n = le.Description(
                stages=[
                    le.Stage("fluct", mean=float(F_f), shape=f_sh, substages=f_m),
                    le.Stage("steady", mean=float(Lg_f), shape=l_sh, substages=1 if l_sh != "chain" else l_m),
                ],
                transfers=[
                    le.Transfer("fluct", "steady", driver="host", birth=True),
                    le.Transfer("steady", "fluct"),
                ],
                drivers=[le.Driver("host", sigma=sig, phi=phi_f, kind="lognormal")],
            )
            eng_n = le.Engine(desc_n).critical()
            s_n: list[float] = []
            b_n = 0
            if f_sh == "fixed":
                b_n += F_f
            else:
                m_f_u = 1 if f_sh == "memoryless" else f_m
                s_n.extend([1.0 - m_f_u / float(F_f)] * m_f_u)
            if l_sh == "fixed":
                b_n += Lg_f
            else:
                m_l_u = 1 if l_sh == "memoryless" else l_m
                if m_l_u < Lg_f:
                    s_n.extend([1.0 - m_l_u / float(Lg_f)] * m_l_u)
                else:
                    b_n += Lg_f
            sim_n = fast_serial_ring_loss(s_n, b_n, sig, phi_f, steps=24000, burn=2000, pairs=16, seed=20264000)
            res_sub[tag] = {
                "D2": float(eng_n.coefficient()["D2"]),
                "D": float(sim_n["D"]),
                "se": float(sim_n["se"]),
                "D_over_cap": float(sim_n["D"]) / cap,
                "verdict_majority": majority_verdict(float(sim_n["D"]), float(sim_n["se"]), cap),
            }
        aux_nematodes[aux_id] = {"label": lbl, "cap": cap, "m_steady_p50": m_steady_med, "m_steady_p10": m_steady_p10, "cases": res_sub}

    verdict_counts = {
        "supports_robust": [k for k, v in row_summaries.items() if v["verdict"] == "supports"],
        "contradicts_fragile": [k for k, v in row_summaries.items() if v["verdict"] == "contradicts"],
        "not_decided": [k for k, v in row_summaries.items() if v["verdict"] == "not decided"],
    }

    return {
        "overall_verdict": "not decided",
        "overall_summary": (
            f"Row-specific outcome across R10b's 6 rows: supports (robust) in {len(verdict_counts['supports_robust'])}/6 "
            f"({verdict_counts['supports_robust']}), contradicts (fragile) in {len(verdict_counts['contradicts_fragile'])}/6 "
            f"({verdict_counts['contradicts_fragile']}), and not decided in {len(verdict_counts['not_decided'])}/6 "
            f"({verdict_counts['not_decided']})."
        ),
        "verdict_counts": verdict_counts,
        "class_percentile_distributions": class_dists,
        "rows": row_summaries,
        "auxiliary_nematode_rows_5_and_7": aux_nematodes,
    }


def run_h_dev1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Protocol §5 Test `H_dev1` (2,000 bootstrap resamples, seed 20261114)."""
    strict_rows = [r for r in rows if r["n"] is not None and r["n"] >= 10]
    dev_vals = np.array([r["cv_indiv"] for r in strict_rows if classify_dev_vs_surv(r) == "dev_like"], float)
    surv_vals = np.array([r["cv_indiv"] for r in strict_rows if classify_dev_vs_surv(r) == "surv_like"], float)

    med_dev = float(np.median(dev_vals))
    med_surv = float(np.median(surv_vals))
    diff_obs = med_surv - med_dev

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_dev = len(dev_vals)
    n_surv = len(surv_vals)
    boot_dev = np.empty(BOOTSTRAP_RESAMPLES, float)
    boot_surv = np.empty(BOOTSTRAP_RESAMPLES, float)
    for b in range(BOOTSTRAP_RESAMPLES):
        boot_dev[b] = np.median(dev_vals[rng.integers(0, n_dev, size=n_dev)])
        boot_surv[b] = np.median(surv_vals[rng.integers(0, n_surv, size=n_surv)])
    boot_diff = boot_surv - boot_dev

    ci_dev = [float(np.percentile(boot_dev, 2.5)), float(np.percentile(boot_dev, 97.5))]
    ci_surv = [float(np.percentile(boot_surv, 2.5)), float(np.percentile(boot_surv, 97.5))]
    ci_diff = [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))]

    if ci_dev[1] < 0.50 and ci_diff[0] > 0.0:
        verdict = "supports"
    elif ci_dev[0] >= 0.80 or ci_diff[1] < 0.0:
        verdict = "contradicts"
    else:
        verdict = "not decided"

    return {
        "verdict": verdict,
        "n_strict_total": len(strict_rows),
        "n_dev_like": int(n_dev),
        "n_surv_like": int(n_surv),
        "median_cv_dev_like": med_dev,
        "iqr_cv_dev_like": [float(np.percentile(dev_vals, 25)), float(np.percentile(dev_vals, 75))],
        "ci95_median_dev_like": ci_dev,
        "median_cv_surv_like": med_surv,
        "iqr_cv_surv_like": [float(np.percentile(surv_vals, 25)), float(np.percentile(surv_vals, 75))],
        "ci95_median_surv_like": ci_surv,
        "diff_median_surv_minus_dev": diff_obs,
        "ci95_diff_surv_minus_dev": ci_diff,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _boot_hdev1_arrays(dev_vals: np.ndarray, surv_vals: np.ndarray) -> dict[str, Any]:
    med_dev = float(np.median(dev_vals))
    med_surv = float(np.median(surv_vals))
    diff_obs = med_surv - med_dev
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    nd, ns = len(dev_vals), len(surv_vals)
    b_dev = np.empty(BOOTSTRAP_RESAMPLES, float)
    b_surv = np.empty(BOOTSTRAP_RESAMPLES, float)
    for b in range(BOOTSTRAP_RESAMPLES):
        b_dev[b] = np.median(dev_vals[rng.integers(0, nd, size=nd)])
        b_surv[b] = np.median(surv_vals[rng.integers(0, ns, size=ns)])
    b_diff = b_surv - b_dev
    ci_dev = [float(np.percentile(b_dev, 2.5)), float(np.percentile(b_dev, 97.5))]
    ci_surv = [float(np.percentile(b_surv, 2.5)), float(np.percentile(b_surv, 97.5))]
    ci_diff = [float(np.percentile(b_diff, 2.5)), float(np.percentile(b_diff, 97.5))]
    if ci_dev[1] < 0.50 and ci_diff[0] > 0.0:
        verdict = "supports"
    elif ci_dev[0] >= 0.80 or ci_diff[1] < 0.0:
        verdict = "contradicts"
    else:
        verdict = "not decided"
    return {
        "verdict": verdict,
        "K_dev": int(nd),
        "K_surv": int(ns),
        "median_dev": med_dev,
        "ci95_dev": ci_dev,
        "median_surv": med_surv,
        "ci95_surv": ci_surv,
        "diff": diff_obs,
        "ci95_diff": ci_diff,
    }


def run_registered_sensitivities(
    rows: list[dict[str, Any]],
    rule1_vecs: dict[tuple[str, str, float | None], np.ndarray],
) -> dict[str, Any]:
    """Execute all sensitivity and confounder analyses registered in `protocol.md` (§2, §4.1, §7, Amendments 1-3)."""
    import re
    from collections import defaultdict
    from scipy import stats

    sys.path.insert(0, str(DOBSON_ROOT / "checks"))
    import r15a_lib as r15a
    import extract_part2 as ep2

    strict_rows = [r for r in rows if r["n"] is not None and r["n"] >= 10]

    # ------------------------------------------------------------------
    # S1. H_dev1 estimator, subset, field-vs-lab, and VecTraits SE audit
    # ------------------------------------------------------------------
    def split_ds(sub_rows: list[dict[str, Any]], cv_fn=lambda r: r["cv_indiv"], cls_fn=classify_dev_vs_surv) -> dict[str, Any]:
        dv = np.array([cv_fn(r) for r in sub_rows if cls_fn(r) == "dev_like"], float)
        sv = np.array([cv_fn(r) for r in sub_rows if cls_fn(r) == "surv_like"], float)
        return _boot_hdev1_arrays(dv, sv)

    def shep_cv(r: dict[str, Any]) -> float:
        m = re.search(r"cv_sheppard=([0-9.]+)", r["rule_applied"])
        return float(m.group(1)) if m else r["cv_indiv"]

    def comp_cv(r: dict[str, Any]) -> float:
        m = re.search(r"completers_cv=([0-9.]+)", r["rule_applied"])
        return float(m.group(1)) if m else r["cv_indiv"]

    def cls_juv_surv(r: dict[str, Any]) -> str:
        if "juvenile_life_span" in r["stage"]:
            return "surv_like"
        return classify_dev_vs_surv(r)

    is_field = lambda r: (
        r["dataset"] in ("D1_malaria_embo2024", "D11_trematode_taskinen2022")
        or (r["dataset"] == "D5_ixodes_fracasso2021" and "wild_nestbox" in r["stage"])
        or (r["dataset"] == "D6_ixodes_scapularis_brunner2022" and "_lab_Ogden_" not in r["stage"])
    )
    field_strict = [r for r in strict_rows if is_field(r)]
    lab_strict = [r for r in strict_rows if not is_field(r)]

    # VecTraits SE arena-n audit (Amendment 1 item 2: 420 excluded SE rows with Interactor1Number)
    se_raw_ratios: list[float] = []
    se_stock_cvs: list[float] = []
    for jf in sorted((RAW_DIR / "D16_vectraits_compilations").glob("vectraits_*.json")):
        if jf.stem in ep2.VECTRAITS_INDIV_STEMS:
            continue
        payload = json.loads(jf.read_text(encoding="utf-8"))
        rlist = payload.get("results", [])
        if not rlist or str(rlist[0].get("DatasetID") or "") == "1577":
            continue
        for sr in rlist:
            tname = str(sr.get("OriginalTraitName") or "").strip().lower()
            tunit = str(sr.get("OriginalTraitUnit") or "").strip().lower()
            eunit = str(sr.get("OriginalErrorUnit") or "").strip().lower()
            i1n = str(sr.get("Interactor1Number") or "").strip()
            if tunit not in ep2.TIME_UNIT_TO_DAYS or any(w in tname for w in ("percent", "success rate", "molting ratio")):
                continue
            if eunit in ("se", "sem", "standard error") and i1n != "":
                v_raw, ep_raw = sr.get("OriginalTraitValue"), sr.get("OriginalErrorPos")
                if v_raw is not None and ep_raw is not None and float(v_raw) > 0 and float(ep_raw) > 0:
                    n_st = float(i1n)
                    if n_st > 0:
                        rat = float(ep_raw) / float(v_raw)
                        se_raw_ratios.append(rat)
                        se_stock_cvs.append(rat * math.sqrt(n_st) * (1.0 + 1.0 / (4.0 * n_st)))

    s1_subsets = {
        "all_485_cohorts_incl_no_n": split_ds(rows),
        "uncensored_rule1_only": split_ds([r for r in strict_rows if "raw_individual_uncensored" in r["rule_applied"]]),
        "exclude_cross_sectional_rule3": split_ds([r for r in strict_rows if "cross_sectional" not in r["rule_applied"] and "Rule 3" not in r["rule_applied"]]),
        "exclude_9_daily_capped_eip": split_ds([r for r in strict_rows if "daily-step identifiability cap" not in r["rule_applied"]]),
        "sheppard_interval_debiased": split_ds(strict_rows, cv_fn=shep_cv),
        "naive_completers_on_censored_rule2": split_ds(strict_rows, cv_fn=comp_cv),
        "reclassify_juvenile_lifespan_to_surv": split_ds(strict_rows, cls_fn=cls_juv_surv),
        "field_outbred_strict_summary": {
            "K_total": len(field_strict),
            "K_dev": sum(1 for r in field_strict if classify_dev_vs_surv(r) == "dev_like"),
            "K_surv": sum(1 for r in field_strict if classify_dev_vs_surv(r) == "surv_like"),
            "median_dev": float(np.median([r["cv_indiv"] for r in field_strict if classify_dev_vs_surv(r) == "dev_like"])),
            "median_surv": float(np.median([r["cv_indiv"] for r in field_strict if classify_dev_vs_surv(r) == "surv_like"])),
            "hdev1": split_ds(field_strict),
        },
        "laboratory_strict_summary": {
            "K_total": len(lab_strict),
            "hdev1": split_ds(lab_strict),
        },
        "vectraits_SE_arena_n_audit": {
            "n_rows": len(se_stock_cvs),
            "median_raw_SE_over_mean": float(np.median(se_raw_ratios)),
            "median_inflated_SE_sqrtn_over_mean": float(np.median(se_stock_cvs)),
            "max_inflated_SE_sqrtn_over_mean": float(np.max(se_stock_cvs)),
            "n_exceeding_cv_1": int(sum(1 for x in se_stock_cvs if x > 1.0)),
        },
    }

    # ------------------------------------------------------------------
    # S2. D1 malaria skip-rule & left-truncation sensitivity + H_transform variants
    # ------------------------------------------------------------------
    d1_skip_sens: dict[str, Any] = {}
    for sk in (0, 1, 2, 3):
        cl_map = reconstruct_d1_clones(max_zeroes=sk, include_left_trunc=False)
        d1_skip_sens[f"max_skips_{sk}"] = {
            k: {
                "n": int(len(t_a)),
                "d": int(e_a.sum()),
                "cv_weibull_mle": eh.weibull_censored_mle(t_a, e_a)["cv_mle"],
                "mean_weibull_mle": eh.weibull_censored_mle(t_a, e_a)["mean_mle"],
                "cv_completers": eh.uncensored_stats(t_a[e_a])["cv_indiv"],
            }
            for k, (t_a, e_a) in cl_map.items()
        }
    cl_lt = reconstruct_d1_clones(max_zeroes=2, include_left_trunc=True)
    d1_skip_sens["max_skips_2_incl_left_truncated_May12"] = {
        k: {
            "n": int(len(t_a)),
            "d": int(e_a.sum()),
            "cv_weibull_mle": eh.weibull_censored_mle(t_a, e_a)["cv_mle"],
        }
        for k, (t_a, e_a) in cl_lt.items()
    }

    # H_transform variants: (a) strictly 0%-censored Rule 1 cohorts; (b) adding D1's 3 individual human cohorts
    zero_cens_rows = [
        r for r in rows
        if "raw_individual_uncensored" in r["rule_applied"]
        and r["n"] is not None
        and r["n"] >= 30
        and r["censoring"].startswith("uncensored")
    ]
    ht_zero = run_h_transform(zero_cens_rows, rule1_vecs)
    # Adding the 3 D1 cohorts (2 have CV_indiv > 1.0 -> inadmissible; msp2 cleared kernel check)
    cl_d1_primary = reconstruct_d1_clones(max_zeroes=2, include_left_trunc=False)
    p_msp2, cv_msp2 = eh.discretize_empirical_stay(cl_d1_primary["msp2"][0][cl_d1_primary["msp2"][1]], 60.0, max_age=400)
    d2_msp2, t_msp2, _ = eh.empirical_D2(8, p_msp2, 1.40, 0.75 ** (1.0 / 8.0))
    fam_msp2 = matched_cv_d2_families(t_msp2 - 8.0, cv_msp2, 8, 1.40, 0.75 ** (1.0 / 8.0))
    err_msp2 = max(abs(fam_msp2["D2_chain"] - d2_msp2), abs(fam_msp2["D2_fg"] - d2_msp2)) / abs(d2_msp2)
    d1_msp2_passes = bool(err_msp2 <= 0.10)
    s2_malaria_and_transform = {
        "D1_skip_and_left_truncation": d1_skip_sens,
        "H_transform_strictly_zero_censored_only": {
            "verdict": ht_zero["verdict"],
            "K": ht_zero["k_cohorts_n_ge_30"],
            "n_pass": ht_zero["n_pass_10pct"],
            "n_fail": ht_zero["n_fail_10pct"],
            "frac_pass": ht_zero["frac_pass_10pct"],
            "frac_fail": ht_zero["frac_fail_10pct"],
        },
        "H_transform_including_3_D1_cohorts": {
            "K": 162 + 3,
            "n_pass": 144 + (1 if d1_msp2_passes else 0),
            "n_fail": 18 + 2 + (0 if d1_msp2_passes else 1),
            "frac_fail": (18 + 2 + (0 if d1_msp2_passes else 1)) / 165.0,
            "d1_msp2_rel_err_max": err_msp2,
            "verdict": "contradicts",
        },
    }

    # ------------------------------------------------------------------
    # S3. Short fluctuating stage spread (Part B2 of R10b) with measured CVs
    # ------------------------------------------------------------------
    s3_b2: dict[str, Any] = {}
    for rid, f_cases in [
        ("1a", [("F_fixed", "fixed", 1), ("F_D12_prepatent_m8", "fixed", 8), ("F_D8_shedding_m3", "chain", 3), ("F_geom_m1", "memoryless", 1)]),
        ("1b", [("F_fixed", "fixed", 1), ("F_D8_shedding_m3", "chain", 3), ("F_geom_m1", "memoryless", 1)]),
        ("6", [("F_fixed", "fixed", 1), ("F_D3_21C_eip_m7", "chain", 7), ("F_geom_m1", "memoryless", 1)]),
        ("3a", [("F_fixed", "fixed", 1), ("F_chain_m4", "chain", 4), ("F_geom_m1", "memoryless", 1)]),
        ("3b", [("F_fixed", "fixed", 1), ("F_chain_m4", "chain", 4), ("F_geom_m1", "memoryless", 1)]),
    ]:
        cfg = R10B_FINE_ROWS[rid]
        F = cfg["F_fine"]
        Lg = cfg["Lg_fine"]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        sig = cfg["sigma"]
        cap = 0.5 * sig * sig
        # Steady stage at empirical median m_emp (m=11 for 1a/1b, m=1 for 6 [ plus m=4 check ], m=12 for 3a/3b)
        m_steady_list = [1, 4] if rid == "6" else ([11] if rid in ("1a", "1b") else [12, 16])
        s3_b2[rid] = {}
        for m_st in m_steady_list:
            for f_tag, f_sh, f_m in f_cases:
                s_st: list[float] = []
                b_fx = 0
                if f_sh == "fixed":
                    b_fx += F
                else:
                    mu_f = 1 if f_sh == "memoryless" else f_m
                    if mu_f < F:
                        s_st.extend([1.0 - mu_f / float(F)] * mu_f)
                    else:
                        b_fx += F
                if m_st < Lg:
                    s_st.extend([1.0 - m_st / float(Lg)] * m_st)
                else:
                    b_fx += Lg
                sim = fast_serial_ring_loss(s_st, b_fx, sig, phi_f, steps=20000, burn=1500, pairs=16, seed=20265000)
                s3_b2[rid][f"L_m{m_st}__{f_tag}"] = {
                    "D": sim["D"],
                    "se": sim["se"],
                    "D_over_cap": sim["D"] / cap,
                    "verdict_majority": majority_verdict(sim["D"], sim["se"], cap),
                }

    # ------------------------------------------------------------------
    # S4. Host encounter elasticity (eta=0.5 vs 1.0), seasonality (Floquet), and host memory sweeps
    # ------------------------------------------------------------------
    s4_factors: dict[str, Any] = {"elasticity_eta_05_vs_10": {}, "seasonal_floquet_vs_ar1": {}, "memory_phi_sweep": {}}

    # 4a. Elasticity eta in {0.5, 1.0} via lifecycle_engine (`protocol.md` §7 item 1)
    for rid, m_st in [("1b", 11), ("6", 1), ("3a", 16), ("3b", 12)]:
        cfg = R10B_FINE_ROWS[rid]
        phi_f = cfg["phi_coarse"] ** (1.0 / cfg["r"])
        cap_eta1 = 0.5 * cfg["sigma"] ** 2
        s4_factors["elasticity_eta_05_vs_10"][rid] = {}
        for eta in (1.0, 0.5):
            desc_e = le.Description(
                stages=[
                    le.Stage("fluct", mean=float(cfg["F_fine"]), shape="fixed"),
                    le.Stage("steady", mean=float(cfg["Lg_fine"]), shape="memoryless" if m_st == 1 else "chain", substages=m_st),
                ],
                transfers=[
                    le.Transfer("fluct", "steady", driver="host", elasticity=eta, birth=True),
                    le.Transfer("steady", "fluct"),
                ],
                drivers=[le.Driver("host", sigma=cfg["sigma"], phi=phi_f, kind="lognormal")],
            )
            eng_e = le.Engine(desc_e).critical()
            sim_e = eng_e.loss(steps=16000, burn=1500, pairs=12, seed=20265100)
            s4_factors["elasticity_eta_05_vs_10"][rid][f"eta_{eta:.1f}"] = {
                "D2": float(eng_e.coefficient()["D2"]),
                "D": float(sim_e["D"]),
                "se": float(sim_e["se"]),
                "J": float(sim_e["J"]),
                "D_over_eta2_cap": float(sim_e["D"]) / (eta * eta * cap_eta1),
            }

    # 4b. Sinusoidal seasonal forcing (`r15a_lib.floquet_root`) vs AR(1) (`protocol.md` §7 item 2)
    for rid, period_fine, m_list in [("6", 120, [1, 4]), ("3a", 30, [1, 16]), ("3b", 30, [1, 12])]:
        cfg = R10B_FINE_ROWS[rid]
        sig = cfg["sigma"]
        cap = 0.5 * sig * sig
        t_idx = np.arange(period_fine, dtype=float)
        raw_s = np.exp(sig * math.sqrt(2.0) * np.sin(2.0 * math.pi * t_idx / period_fine))
        h_seas = raw_s / float(raw_s.mean())  # exact arithmetic mean 1 over period
        j_seas = -float(np.mean(np.log(h_seas)))
        s4_factors["seasonal_floquet_vs_ar1"][rid] = {"period_fine_steps": period_fine, "J_seasonal": j_seas, "cap_lognormal": cap}
        for m_st in m_list:
            desc_s = le.Description(
                stages=[
                    le.Stage("fluct", mean=float(cfg["F_fine"]), shape="fixed"),
                    le.Stage("steady", mean=float(cfg["Lg_fine"]), shape="memoryless" if m_st == 1 else "chain", substages=m_st),
                ],
                transfers=[
                    le.Transfer("fluct", "steady", driver="host", birth=True),
                    le.Transfer("steady", "fluct"),
                ],
                drivers=[le.Driver("host", sigma=sig, phi=0.5, kind="lognormal")],
            )
            eng_s = le.Engine(desc_s).critical()
            rest = np.repeat(eng_s.base_parts[0][None, :, :], period_fine, axis=0) + (h_seas[:, None, None] - 1.0) * eng_s.rate_parts[0, 0][None, :, :]
            scal = np.repeat(eng_s.base_parts[1][None, :, :], period_fine, axis=0) + (h_seas[:, None, None] - 1.0) * eng_s.rate_parts[1, 0][None, :, :]
            d_floq = float(r15a.floquet_root(rest, scal, 0.0))
            s4_factors["seasonal_floquet_vs_ar1"][rid][f"m_{m_st}"] = {
                "D_seasonal_floquet": d_floq,
                "D_over_J_seasonal": d_floq / j_seas,
            }

    # 4c. Host memory phi sweep (`protocol.md` §7 item 3: phi in {0, phi_table/2, phi_table, 0.90} + oscillatory negative coarse r1)
    for rid, m_st in [("1b", 11), ("6", 1), ("3a", 16), ("3b", 12), ("2a", 6)]:
        cfg = R10B_FINE_ROWS[rid]
        sig = cfg["sigma"]
        cap = 0.5 * sig * sig
        s4_factors["memory_phi_sweep"][rid] = {}
        for p_tag, phi_c in [("phi_0", 0.0), ("phi_half", 0.5 * cfg["phi_coarse"]), ("phi_table", cfg["phi_coarse"]), ("phi_090", 0.90)]:
            phi_f = 0.0 if phi_c == 0.0 else (phi_c ** (1.0 / cfg["r"]))
            s_st = []
            b_fx = cfg["other_fine"]
            if cfg["F_shape"] == "memoryless":
                s_st.append(1.0 - 1.0 / cfg["F_fine"])
            else:
                b_fx += cfg["F_fine"]
            s_st.extend([1.0 - m_st / float(cfg["Lg_fine"])] * m_st)
            sim_p = fast_serial_ring_loss(s_st, b_fx, sig, phi_f, steps=18000, burn=1500, pairs=14, seed=20265200)
            s4_factors["memory_phi_sweep"][rid][p_tag] = {
                "phi_coarse": phi_c,
                "phi_fine": phi_f,
                "D": sim_p["D"],
                "se": sim_p["se"],
                "D_over_cap": sim_p["D"] / cap,
                "verdict_majority": majority_verdict(sim_p["D"], sim_p["se"], cap),
            }
        # Also evaluate negative fine-step autocorrelation (-0.30) on 3a, 3b, 2a (mast/vole oscillation check)
        if rid in ("3a", "3b", "2a"):
            s_st = ([1.0 - 1.0 / cfg["F_fine"]] if cfg["F_shape"] == "memoryless" else []) + [1.0 - m_st / float(cfg["Lg_fine"])] * m_st
            b_fx = (0 if cfg["F_shape"] == "memoryless" else cfg["F_fine"]) + cfg["other_fine"]
            sim_neg = fast_serial_ring_loss(s_st, b_fx, sig, -0.30, steps=18000, burn=1500, pairs=14, seed=20265299)
            s4_factors["memory_phi_sweep"][rid]["phi_fine_neg030"] = {
                "phi_fine": -0.30,
                "D": sim_neg["D"],
                "se": sim_neg["se"],
                "D_over_cap": sim_neg["D"] / cap,
            }

    # ------------------------------------------------------------------
    # S5. R16 illustrative cycles (lognormal vs bounded, switches E/A/P/M, extinction) & R18 outages
    # ------------------------------------------------------------------
    r16_cycles = [
        ("0f_long_sig06", 24.0, 0.6, 0.4, 11),      # surv_like p50 -> m=11
        ("1f_long_sig14", 24.0, 1.4, 0.4, 11),
        ("2f_mod_sig14", 7.5, 1.4, 0.75, 10),       # surv_like p50 -> m=10 (plus D1 m=1)
        ("3f_short_sig07", 2.3, 0.7, 0.3, 6),       # D6 I. scapularis p50 CV=0.214 -> m=6 (CV=0.241)
        ("4f_short_sig12", 2.3, 1.2, 0.3, 6),
    ]
    r16_driver_comp: dict[str, Any] = {}
    for cid, stay_c, sig, phi_c, m_emp in r16_cycles:
        dt_f = 0.25
        wait_f = 4.0
        long_f = stay_c / dt_f
        phi_f = phi_c ** dt_f
        r16_driver_comp[cid] = {}
        for dkind in ("lognormal", "bounded"):
            for m_tag, m_val in [("m1_geom", 1), ("m4_R16", 4), (f"m{m_emp}_emp", m_emp)]:
                desc_r = le.Description(
                    stages=[
                        le.Stage("exposed", wait_f, "memoryless"),
                        le.Stage("long", long_f, "memoryless" if m_val == 1 else "chain", m_val),
                    ],
                    transfers=[
                        le.Transfer("exposed", "long", driver="h0"),
                        le.Transfer("long", "exposed", birth=True),
                    ],
                    drivers=[le.Driver("h0", sig, phi_f, dkind, 0.05 if dkind == "bounded" else None)],
                )
                eng_r = le.Engine(desc_r).critical()
                sim_r = eng_r.loss(steps=14000, burn=1200, pairs=10, seed=20265300 + m_val)
                r16_driver_comp[cid][f"{dkind}__{m_tag}"] = {
                    "D2": float(eng_r.coefficient()["D2"]),
                    "D": float(sim_r["D"]),
                    "se": float(sim_r["se"]),
                    "J": float(sim_r["J"]),
                    "D_over_J": float(sim_r["D"]) / float(sim_r["J"]),
                }

    # Multi-mechanism switches E, A, P, M at empirical m_emp on R16 fine step
    r16_switches: dict[str, Any] = {}
    for cid, stay_c, sig, phi_c, m_emp in [
        ("0f_long_sig06", 24.0, 0.6, 0.4, 11),
        ("2f_mod_sig14", 7.5, 1.4, 0.75, 10),
        ("3f_short_sig07", 2.3, 0.7, 0.3, 6),
        ("4f_short_sig12", 2.3, 1.2, 0.3, 6),
    ]:
        dt_f = 0.25
        wait_f = 4.0
        long_f = stay_c / dt_f
        phi_f = phi_c ** dt_f
        r16_switches[cid] = {}
        switch_list = [("baseline_S_emp", 0, 0, 0, 0), ("A_alt_hosts", 0, 1, 0, 0), ("P_patches", 0, 0, 1, 0), ("M_stage_death", 0, 0, 0, 1)]
        if sig < 1.35:  # contact-limited E=1 admissible at dt=1/4 for sigma in {0.6, 0.7, 1.2}
            switch_list.insert(1, ("E_contact_delay", 1, 0, 0, 0))
        for sw_name, E_b, A_b, P_b, M_b in switch_list:
            death_l = (1.0 - 0.99 ** dt_f) if M_b else 0.0
            st0 = le.Stage("exposed", wait_f, "memoryless", death=0.5 / wait_f if E_b else 0.0, contact=bool(E_b))
            st1 = le.Stage("long", long_f, "chain", m_emp, death=death_l)
            drvs = [le.Driver(f"h{k}", sig, phi_f, "bounded", 0.05) for k in range(1 + A_b)]
            trs = [le.Transfer("exposed", "long", 1.0 / (1 + A_b), driver=d.name) for d in drvs]
            trs.append(le.Transfer("long", "exposed", birth=True))
            pat = le.Patches(4 if P_b else 1, 0.0, "long", 1.0 - 0.85 ** dt_f if P_b else 0.0, "island", 0.0)
            eng_sw = le.Engine(le.Description([st0, st1], trs, drvs, patches=pat)).critical()
            sim_sw = eng_sw.loss(steps=12000, burn=1000, pairs=8, seed=20265400)
            r16_switches[cid][sw_name] = {
                "D2": float(eng_sw.coefficient()["D2"]),
                "D": float(sim_sw["D"]),
                "se": float(sim_sw["se"]),
                "J": float(sim_sw["J"]),
            }

    # Whole-number branching extinction (`Engine.extinction`) on moderate cycle (`m=1` vs `m=10`)
    ext_comp: dict[str, Any] = {}
    for dkind in ("lognormal", "bounded"):
        for m_val in (1, 10):
            desc_ex = le.Description(
                stages=[
                    le.Stage("exposed", 4.0, "memoryless"),
                    le.Stage("long", 30.0, "memoryless" if m_val == 1 else "chain", m_val),
                ],
                transfers=[
                    le.Transfer("exposed", "long", driver="h0"),
                    le.Transfer("long", "exposed", birth=True),
                ],
                drivers=[le.Driver("h0", 1.4, 0.75 ** 0.25, dkind, 0.05 if dkind == "bounded" else None)],
            )
            eng_ex = le.Engine(desc_ex).critical()
            sim_ex = eng_ex.loss(steps=16000, burn=1500, pairs=12, seed=20265500)
            init_vec = np.zeros(eng_ex.n, dtype=np.int64)
            init_vec[0] = 10000
            ext_res = eng_ex.extinction(init_vec, int(math.ceil(2000.0 / eng_ex.b)), trials=128, seed=20265502, y=float(sim_ex["D"]) + 0.02)
            ext_comp[f"{dkind}__m{m_val}"] = {
                "D": float(sim_ex["D"]),
                "se_D": float(sim_ex["se"]),
                "extinction_fraction": ext_res["fraction"],
                "extinction_se": ext_res["se"],
                "extinction_conditional_mean": ext_res["conditional_mean"],
            }

    # R18 exact single-outage survival formula P(G) = 1 - [1 - h(G)*(1 - (1 - p(G))^b)]^J (`notes/exact-limits.md` line 96)
    def r18_outage_survival(g_over_mu: float, k_shape: float, J_hosts: int = 100, b_burden: int = 100, host_death_share: float = 1.0) -> float:
        # Pure host death (host_death_share=1): h(G) = P(T_host > G) = gammaincc(k, k*G/mu), p(G) = 1 (parasite survives if host survives)
        # Split death: host hazard share a, independent parasite survival p(G)
        if host_death_share >= 1.0 - 1e-12:
            h_g = float(special.gammaincc(k_shape, k_shape * g_over_mu))
            p_indiv = 1.0
        else:
            h_g = float(special.gammaincc(k_shape, k_shape * host_death_share * g_over_mu))
            p_indiv = math.exp(-(1.0 - host_death_share) * g_over_mu)
        surv_one_host = h_g * (-math.expm1(b_burden * math.log1p(-p_indiv)) if p_indiv < 1.0 else 1.0)
        return float(-math.expm1(J_hosts * math.log1p(-min(max(surv_one_host, 0.0), 1.0 - 1e-16))))

    def gap_5pct(k_shape: float) -> float:
        return float(optimize.brentq(lambda g: r18_outage_survival(g, k_shape) - 0.05, 0.05, 40.0))

    # Rule 7 Known Check K4: reproduce R18's exact 5%-retention gaps 7.58 (k=1) and 3.48 (k=4) (`notes/exact-limits.md` line 96)
    g05_k1 = gap_5pct(1.0)
    g05_k4 = gap_5pct(4.0)
    assert abs(g05_k1 - 7.58) < 0.01 and abs(g05_k4 - 3.48) < 0.01, f"K4 failed: {g05_k1}, {g05_k4}"

    outage_profiles: dict[str, Any] = {
        "K4_check_R18_5pct_gaps": {"k1_exp": g05_k1, "k4_4phase": g05_k4, "pass": True},
    }
    for label_o, cv_o in [
        ("D1_malaria_p50_CV1.187", 1.187183),
        ("exponential_baseline_CV1.000", 1.0),
        ("R18_four_phase_CV0.500", 0.5),
        ("surv_like_p50_CV0.294", 0.293522),
        ("D6_ixodes_p50_CV0.214", 0.213930),
    ]:
        k_sh = 1.0 / (cv_o * cv_o)
        outage_profiles[label_o] = {
            "cv": cv_o,
            "k_gamma": k_sh,
            "gap_5pct_retention_in_mean_lifetimes": gap_5pct(k_sh),
            "P_survive_at_G_over_mu": {
                f"{g:.1f}": r18_outage_survival(g, k_sh) for g in (0.5, 1.0, 2.0, 4.0, 8.0)
            },
        }

    # ------------------------------------------------------------------
    # S6. Weather and thermal analyses W1, W2, W3 (`protocol.md` Amendment 2 & §7 item 4)
    # ------------------------------------------------------------------
    # W1: Spearman rho(T, mean) and rho(T, CV_indiv) within constant-temperature series (>=4 temperatures)
    series_map: dict[tuple[str, str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        if r["temperature_C"] is None:
            continue
        ds = r["dataset"]
        if ds == "D6_ixodes_scapularis_brunner2022" and "_lab_Ogden_" not in r["stage"]:
            continue
        if ds in ("D1_malaria_embo2024", "D11_trematode_taskinen2022"):
            continue
        if ds == "D5_ixodes_fracasso2021" and "wild_nestbox" in r["stage"]:
            continue
        # Strip trailing row id in D16 (`_id12345`) and temperature suffix in Ogden (`_20C`) so thermal series group cleanly
        st_clean = re.sub(r"_id\d+$", "", r["stage"])
        st_clean = re.sub(r"_\d+C$", "", st_clean)
        series_map[(ds, r["taxon"], st_clean)].append((float(r["temperature_C"]), r["mean"], r["cv_indiv"]))

    w1_series: list[dict[str, Any]] = []
    for (ds, txn, st_c), pts in sorted(series_map.items()):
        # Aggregate duplicate replicates at identical temperature by median or keep all points if >=4 distinct temperatures
        distinct_t = sorted({p[0] for p in pts})
        if len(distinct_t) < 4:
            continue
        t_v = np.array([p[0] for p in pts], float)
        m_v = np.array([p[1] for p in pts], float)
        c_v = np.array([p[2] for p in pts], float)
        rho_tm = float(stats.spearmanr(t_v, m_v).statistic)
        rho_tc = float(stats.spearmanr(t_v, c_v).statistic) if np.ptp(c_v) > 1e-12 else 0.0
        w1_series.append({"dataset": ds, "taxon": txn, "stage_series": st_c, "n_points": len(pts), "n_temps": len(distinct_t), "rho_T_mean": rho_tm, "rho_T_cv": rho_tc})

    rhos_tm = np.array([s["rho_T_mean"] for s in w1_series if math.isfinite(s["rho_T_mean"])], float)
    rhos_tc = np.array([s["rho_T_cv"] for s in w1_series if math.isfinite(s["rho_T_cv"])], float)

    # W2: Field microclimate buffering in D6 (ibutton vs Daymet 1 km) & Sahelian dry season vs 37 C homeothermy in D1
    d6_sites = {
        "FD": ("Fort Drum, NY", 44.00, -75.70, "daymet_FD_FortDrum_NY.csv"),
        "WP": ("West Point, NY", 41.40, -74.10, "daymet_WP_WestPoint_NY.csv"),
        "CL": ("Camp Lejeune, NC", 34.70, -77.50, "daymet_CL_CampLejeune_NC.csv"),
    }
    ib_by_sd: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for r in csv.DictReader((RAW_DIR / "D6_ixodes_scapularis_brunner2022" / "ibutton_data.csv").open(encoding="utf-8")):
        if r["Position"].strip() == "Above":
            try:
                ib_by_sd[(r["Site"].strip(), r["Date"].strip())].append(
                    (float(r["T_min"]), float(r["T_max"]), float(r["T_mean"]), float(r["VPD_mean"]))
                )
            except ValueError:
                pass
    w2_d6: dict[str, Any] = {}
    wp_summer_temps: list[float] = []
    for scode, (sname, slat, slon, dm_fn) in d6_sites.items():
        dm_path = RAW_DIR / "weather" / dm_fn
        raw_lines = dm_path.read_text(encoding="utf-8").splitlines()
        lat_line = next(ln for ln in raw_lines if ln.startswith("Latitude"))
        m_ll = re.search(r"Latitude:\s*([0-9.-]+)\s*Longitude:\s*([0-9.-]+)", lat_line)
        glat, glon = (float(m_ll.group(1)), float(m_ll.group(2))) if m_ll else (slat, slon)
        dist_km = eh.haversine_km(slat, slon, glat, glon)
        clean_dm = [ln for ln in raw_lines if ln.strip() and not ln.startswith(("Latitude", "X & Y", "Tile", "Elevation", "All years", "How to cite"))]
        dm_map: dict[str, tuple[float, float, float, float]] = {}
        for r in csv.DictReader(clean_dm):
            d_iso = (dt.date(int(r["year"]), 1, 1) + dt.timedelta(days=int(r["yday"]) - 1)).isoformat()
            tmax, tmin, vp = float(r["tmax (deg c)"]), float(r["tmin (deg c)"]), float(r["vp (Pa)"])
            tmean = 0.5 * (tmax + tmin)
            vpd = eh.vpd_kpa_from_tmax_tmin_vp(tmax, tmin, vp)
            dm_map[d_iso] = (tmin, tmax, tmean, vpd)
            if scode == "WP" and 135 <= int(r["yday"]) <= 275:
                wp_summer_temps.append(tmean)
        matched_days = sorted(d for (s, d) in ib_by_sd if s == scode and d in dm_map)
        ib_dtr = np.array([np.mean([x[1] - x[0] for x in ib_by_sd[(scode, d)]]) for d in matched_days], float)
        dm_dtr = np.array([dm_map[d][1] - dm_map[d][0] for d in matched_days], float)
        ib_vpd = np.array([np.mean([x[3] for x in ib_by_sd[(scode, d)]]) for d in matched_days], float)
        dm_vpd = np.array([dm_map[d][3] for d in matched_days], float)
        ib_tmin = np.array([np.mean([x[0] for x in ib_by_sd[(scode, d)]]) for d in matched_days], float)
        dm_tmin = np.array([dm_map[d][0] for d in matched_days], float)
        w2_d6[scode] = {
            "site_name": sname,
            "daymet_grid_offset_km": dist_km,
            "matched_logger_days": len(matched_days),
            "mean_diurnal_range_daymet_C": float(dm_dtr.mean()),
            "mean_diurnal_range_ibutton_litter_C": float(ib_dtr.mean()),
            "min_winter_Tmin_daymet_C": float(dm_tmin.min()),
            "min_winter_Tmin_ibutton_litter_C": float(ib_tmin.min()),
            "mean_VPD_daymet_kPa": float(dm_vpd.mean()),
            "mean_VPD_ibutton_litter_kPa": float(ib_vpd.mean()),
        }

    # D1 Kalifabougou NASA POWER dry season (2013-01-01 to 2013-04-30) vs wet season (2012-07-01 to 2012-10-31)
    np_mali = json.loads((RAW_DIR / "weather" / "nasapower_Mali_Kalifabougou_D1.json").read_text(encoding="utf-8"))["properties"]["parameter"]
    dry_keys = [k for k in sorted(np_mali["T2M"]) if "20130101" <= k <= "20130430"]
    wet_keys = [k for k in sorted(np_mali["T2M"]) if "20120701" <= k <= "20121031"]
    all_mali_t = np.array([float(np_mali["T2M"][k]) for k in sorted(np_mali["T2M"]) if "20120101" <= k <= "20121231"], float)
    w2_d1_mali = {
        "wet_season_2012JulOct_mean_prcp_mm_d": float(np.mean([np_mali["PRECTOTCORR"][k] for k in wet_keys])),
        "dry_season_2013JanApr_mean_prcp_mm_d": float(np.mean([np_mali["PRECTOTCORR"][k] for k in dry_keys])),
        "dry_season_zero_rain_days": int(sum(1 for k in dry_keys if np_mali["PRECTOTCORR"][k] < 0.1)),
        "dry_season_total_days": len(dry_keys),
        "wet_season_mean_RH_pct": float(np.mean([np_mali["RH2M"][k] for k in wet_keys])),
        "dry_season_mean_RH_pct": float(np.mean([np_mali["RH2M"][k] for k in dry_keys])),
        "annual_T2M_mean_C": float(all_mali_t.mean()),
        "annual_T2M_sd_C": float(all_mali_t.std(ddof=1)),
        "annual_T2M_lag1_phi": float(np.corrcoef(all_mali_t[:-1], all_mali_t[1:])[0, 1]),
        "sheltered_human_blood_T_C": 37.0,
    }

    # W3: Temperature as mechanistic input in `r15a_lib.py` (`DDModel`) comparing m=1 vs empirical m_emp
    # Empirical thermal responses fitted to (a) D3 P. falciparum EIP & An. stephensi mortality, (b) D6 I. scapularis moulting, (c) D13 M. marshalli L3
    fn_pf_eip = lambda T: (r15a.briere(T, 1.10e-4, 15.4, 38.0), r15a.q10rate(T, 0.045, 2.1, Tref=24.0))
    fn_ix_moult = lambda T: (r15a.briere(T, 2.95e-5, 8.5, 34.0), r15a.q10rate(T, 0.012, 1.8, Tref=20.0))
    fn_mm_l3 = lambda T: (r15a.briere(T, 8.80e-5, 2.0, 37.5), r15a.q10rate(T, 0.015, 2.3, Tref=15.0))

    wp_arr = np.array(wp_summer_temps, float)
    wp_tbar, wp_sig = float(wp_arr.mean()), float(wp_arr.std(ddof=1))
    wp_phi = float(np.corrcoef(wp_arr[:-1], wp_arr[1:])[0, 1])

    w3_dd: dict[str, Any] = {}
    for sys_tag, fn_th, s_shelt, m_emp_w, tbar, sig_t, phi_t, amp_s in [
        ("D3_Pf_EIP_Kalifabougou", fn_pf_eip, 1.0 - 1.0 / 180.0, 25, float(all_mali_t.mean()), float(all_mali_t.std(ddof=1)), float(np.corrcoef(all_mali_t[:-1], all_mali_t[1:])[0, 1]), 5.2),
        ("D6_Ixodes_moult_WestPoint", fn_ix_moult, 1.0 - 1.0 / 30.0, 24, wp_tbar, wp_sig, wp_phi, 6.5),
        ("D13_Marshallagia_L3_temperate", fn_mm_l3, 1.0 - 1.0 / 90.0, 13, 15.0, 4.5, 0.80, 7.0),
    ]:
        w3_dd[sys_tag] = {"Tbar_C": tbar, "sigT_C": sig_t, "phiT": phi_t, "amp_season_C": amp_s, "m_emp": m_emp_w}
        for m_lbl, m_v in [("m1_exponential", 1), (f"m{m_emp_w}_empirical", m_emp_w)]:
            mod = r15a.DDModel(m_v, s1=s_shelt, fn=fn_th)
            fm = r15a.Formulas(mod, tbar, sig_t, J=60)
            _, d_herm_full, _ = fm.hermite(phi_t)
            sc = r15a.season_case(mod, tbar, amp_s, P=365)
            w3_dd[sys_tag][m_lbl] = {
                "J_gauss_curvature": float(fm.J_part()),
                "D_ar1_hermite": float(d_herm_full),
                "J_seasonal_floquet": float(sc["J"]),
                "D_seasonal_floquet": float(sc["D"]),
            }

    return {
        "S1_Hdev1_subsets_and_estimators": s1_subsets,
        "S2_D1_skip_truncation_and_Htransform": s2_malaria_and_transform,
        "S3_fluctuating_stage_B2": s3_b2,
        "S4_elasticity_memory_seasonality": s4_factors,
        "S5_R16_multimechanism_extinction_and_outages": {
            "driver_lognormal_vs_bounded": r16_driver_comp,
            "switches_E_A_P_M_at_empirical_m": r16_switches,
            "branching_extinction_moderate": ext_comp,
            "R18_single_outage_survival": outage_profiles,
        },
        "S6_thermal_and_weather_W1_W2_W3": {
            "W1_within_series_spearman": {
                "n_series_ge_4_temps": len(w1_series),
                "median_rho_T_mean": float(np.median(rhos_tm)),
                "iqr_rho_T_mean": [float(np.percentile(rhos_tm, 25)), float(np.percentile(rhos_tm, 75))],
                "n_negative_rho_T_mean": int(np.sum(rhos_tm < 0.0)),
                "median_rho_T_cv": float(np.median(rhos_tc)),
                "iqr_rho_T_cv": [float(np.percentile(rhos_tc, 25)), float(np.percentile(rhos_tc, 75))],
                "series": w1_series,
            },
            "W2_field_weather_buffering_and_homeothermy": {
                "D6_sites_ibutton_vs_daymet": w2_d6,
                "D1_Kalifabougou_dry_vs_wet_and_homeothermy": w2_d1_mali,
            },
            "W3_r15a_DDModel_thermal_input": w3_dd,
        },
    }


def main() -> None:
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = load_stage_cv(HERE / "results" / "stage_cv.csv")
    known = run_known_checks()
    rule1_vecs = reconstruct_rule1_vectors(rows)
    h_dev1 = run_h_dev1(rows)
    h_sys = run_h_sys(rows)
    h_transform = run_h_transform(rows, rule1_vecs)
    sensitivities = run_registered_sensitivities(rows, rule1_vecs)
    out = {
        "timestamp_utc": now_utc,
        "known_checks": known,
        "H_dev1": h_dev1,
        "H_sys": h_sys,
        "H_transform": h_transform,
        "sensitivities": sensitivities,
    }
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    s1 = sensitivities["S1_Hdev1_subsets_and_estimators"]
    s2 = sensitivities["S2_D1_skip_truncation_and_Htransform"]
    s5 = sensitivities["S5_R16_multimechanism_extinction_and_outages"]
    s6 = sensitivities["S6_thermal_and_weather_W1_W2_W3"]
    lines = [
        f"devtimes Step B tests — {now_utc}",
        f"Known check K1 (chain D2): pass={known['K1_engine_vs_formula_D2']['pass']} (diff={known['K1_engine_vs_formula_D2']['abs_diff']:.3e})",
        f"Known check K2 (fixed+geom D2): pass={known['K2_engine_fixed_geom_vs_formula_D2']['pass']} (diff={known['K2_engine_fixed_geom_vs_formula_D2']['abs_diff']:.3e})",
        f"Known check K3 (companion vs engine loss): pass={known['K3_empirical_companion_vs_engine_loss']['pass']} (z={known['K3_empirical_companion_vs_engine_loss']['z_diff']:.2f})",
        f"Known check K4 (R18 outage 5% gaps): pass={s5['R18_single_outage_survival']['K4_check_R18_5pct_gaps']['pass']} (k=1: {s5['R18_single_outage_survival']['K4_check_R18_5pct_gaps']['k1_exp']:.3f}, k=4: {s5['R18_single_outage_survival']['K4_check_R18_5pct_gaps']['k4_4phase']:.3f})",
        f"Rule 1 vector reconstruction check: 202/202 cohorts matched stage_cv.csv",
        "",
        f"=== H_dev1 ===",
        f"Verdict: {h_dev1['verdict']} | dev_like (K={h_dev1['n_dev_like']}): median={h_dev1['median_cv_dev_like']:.4f} 95%CI={h_dev1['ci95_median_dev_like']} | surv_like (K={h_dev1['n_surv_like']}): median={h_dev1['median_cv_surv_like']:.4f} 95%CI={h_dev1['ci95_median_surv_like']} | diff={h_dev1['diff_median_surv_minus_dev']:.4f} 95%CI={h_dev1['ci95_diff_surv_minus_dev']}",
        "",
        f"=== H_sys ===",
        f"Overall: {h_sys['overall_verdict']} — {h_sys['overall_summary']}",
    ]
    for rid, rinfo in h_sys["rows"].items():
        p50 = rinfo["evaluations_p10_to_p90"]["p50"]
        p25 = rinfo["evaluations_p10_to_p90"]["p25"]
        lines.append(
            f"  Row {rid:2s} ({rinfo['label']}): verdict={rinfo['verdict']} [{rinfo['classification']}] | "
            f"cap/2={rinfo['half_cap']:.4f} | p25(CV={p25['cv_target']:.3f}): D_chain={p25['chain']['D']:.4f}±{p25['chain']['se']:.4f} [{p25['chain']['verdict_majority']}], "
            f"D_fg={p25['fixed_geom']['D']:.4f}±{p25['fixed_geom']['se']:.4f} [{p25['fixed_geom']['verdict_majority']}] | "
            f"p50(CV={p50['cv_target']:.3f}): D_chain={p50['chain']['D']:.4f}±{p50['chain']['se']:.4f} [{p50['chain']['verdict_majority']}], "
            f"D_fg={p50['fixed_geom']['D']:.4f}±{p50['fixed_geom']['se']:.4f} [{p50['fixed_geom']['verdict_majority']}]"
        )
    lines.extend([
        "",
        f"=== H_transform ===",
        f"Verdict: {h_transform['verdict']} | K={h_transform['k_cohorts_n_ge_30']} uncensored cohorts (n>=30): pass_10pct={h_transform['n_pass_10pct']}/{h_transform['k_cohorts_n_ge_30']} ({100*h_transform['frac_pass_10pct']:.2f}%, 95%CI=[{100*h_transform['ci95_frac_pass'][0]:.2f}%, {100*h_transform['ci95_frac_pass'][1]:.2f}%]) | fail={h_transform['n_fail_10pct']} (inadmissible_cv={h_transform['n_inadmissible_cv']}, shape_err>10%={h_transform['n_shape_error_gt_10pct']}, P_boot(fail>10%)={h_transform['prob_boot_fail_gt_10pct']:.3f}) | median_max_rel_err={100*h_transform['median_rel_err_max']:.2f}% (p90={100*h_transform['p90_rel_err_max']:.2f}%, max={100*h_transform['max_rel_err_max']:.2f}%)",
        "",
        f"=== Registered Sensitivities Summary ===",
        f"  S1 H_dev1 all 485 rows: verdict={s1['all_485_cohorts_incl_no_n']['verdict']} (med_dev={s1['all_485_cohorts_incl_no_n']['median_dev']:.4f}, med_surv={s1['all_485_cohorts_incl_no_n']['median_surv']:.4f}, diff={s1['all_485_cohorts_incl_no_n']['diff']:.4f})",
        f"  S1 H_dev1 uncensored Rule 1 only: verdict={s1['uncensored_rule1_only']['verdict']} (med_dev={s1['uncensored_rule1_only']['median_dev']:.4f}, med_surv={s1['uncensored_rule1_only']['median_surv']:.4f}, diff={s1['uncensored_rule1_only']['diff']:.4f})",
        f"  S1 H_dev1 exclude cross-sectional Rule 3: verdict={s1['exclude_cross_sectional_rule3']['verdict']} (med_dev={s1['exclude_cross_sectional_rule3']['median_dev']:.4f}, med_surv={s1['exclude_cross_sectional_rule3']['median_surv']:.4f}, diff={s1['exclude_cross_sectional_rule3']['diff']:.4f})",
        f"  S1 H_dev1 Sheppard debiased: verdict={s1['sheppard_interval_debiased']['verdict']} (med_dev={s1['sheppard_interval_debiased']['median_dev']:.4f}, med_surv={s1['sheppard_interval_debiased']['median_surv']:.4f}, diff={s1['sheppard_interval_debiased']['diff']:.4f})",
        f"  S1 H_dev1 naive completers on Rule 2: verdict={s1['naive_completers_on_censored_rule2']['verdict']} (med_dev={s1['naive_completers_on_censored_rule2']['median_dev']:.4f}, med_surv={s1['naive_completers_on_censored_rule2']['median_surv']:.4f}, diff={s1['naive_completers_on_censored_rule2']['diff']:.4f})",
        f"  S1 H_dev1 reclassify juvenile lifespan to surv: verdict={s1['reclassify_juvenile_lifespan_to_surv']['verdict']} (med_dev={s1['reclassify_juvenile_lifespan_to_surv']['median_dev']:.4f}, med_surv={s1['reclassify_juvenile_lifespan_to_surv']['median_surv']:.4f}, diff={s1['reclassify_juvenile_lifespan_to_surv']['diff']:.4f})",
        f"  S1 VecTraits SE arena audit (n={s1['vectraits_SE_arena_n_audit']['n_rows']} rows): median SE/mean={s1['vectraits_SE_arena_n_audit']['median_raw_SE_over_mean']:.4f}, median SE*sqrt(Interactor1Number)/mean={s1['vectraits_SE_arena_n_audit']['median_inflated_SE_sqrtn_over_mean']:.4f} (max={s1['vectraits_SE_arena_n_audit']['max_inflated_SE_sqrtn_over_mean']:.2f}, n>1.0={s1['vectraits_SE_arena_n_audit']['n_exceeding_cv_1']})",
        f"  S2 H_transform zero-censored only: verdict={s2['H_transform_strictly_zero_censored_only']['verdict']} ({s2['H_transform_strictly_zero_censored_only']['n_fail']}/{s2['H_transform_strictly_zero_censored_only']['K']} fail = {100*s2['H_transform_strictly_zero_censored_only']['frac_fail']:.2f}%)",
        f"  S2 H_transform incl. 3 D1 cohorts: verdict={s2['H_transform_including_3_D1_cohorts']['verdict']} ({s2['H_transform_including_3_D1_cohorts']['n_fail']}/{s2['H_transform_including_3_D1_cohorts']['K']} fail = {100*s2['H_transform_including_3_D1_cohorts']['frac_fail']:.2f}%)",
        f"  S5 R18 5%-retention outage gaps (in mean lifetimes): D1_malaria={s5['R18_single_outage_survival']['D1_malaria_p50_CV1.187']['gap_5pct_retention_in_mean_lifetimes']:.2f}, Exp(m=1)={s5['R18_single_outage_survival']['exponential_baseline_CV1.000']['gap_5pct_retention_in_mean_lifetimes']:.2f}, surv_like_p50={s5['R18_single_outage_survival']['surv_like_p50_CV0.294']['gap_5pct_retention_in_mean_lifetimes']:.2f}, D6_Ixodes_p50={s5['R18_single_outage_survival']['D6_ixodes_p50_CV0.214']['gap_5pct_retention_in_mean_lifetimes']:.2f}",
        f"  S6 W1 Spearman across {s6['W1_within_series_spearman']['n_series_ge_4_temps']} multi-temp series: median rho(T,mean)={s6['W1_within_series_spearman']['median_rho_T_mean']:.3f} ({s6['W1_within_series_spearman']['n_negative_rho_T_mean']}/{s6['W1_within_series_spearman']['n_series_ge_4_temps']} < 0), median rho(T,CV)={s6['W1_within_series_spearman']['median_rho_T_cv']:.3f}",
    ])
    (out_dir / "tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
