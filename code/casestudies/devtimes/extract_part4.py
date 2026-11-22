#!/usr/bin/env python3
"""Extract stage-duration spreads (Part 4):
  - D11_trematode_taskinen2022
  - D12_trematode_gleichsner2016
  - D13_nematode_marshallagia2023
  - D14_nematode_heligmosomoides
  - D15_helminth_clc_benesh2017

Applies protocol.md §4, §4.2, and Amendments 1-5 (`python3 -I` safe).
Writes:
  - results/stage_cv_part4.csv
  - results/extraction_log_part4.txt
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import extract_helpers as eh  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "devtimes"
RESULTS_DIR = HERE / "results"
CSV_OUT = RESULTS_DIR / "stage_cv_part4.csv"
LOG_OUT = RESULTS_DIR / "extraction_log_part4.txt"

CSV_COLUMNS = [
    "dataset",
    "taxon",
    "stage",
    "stage_category",
    "temperature_C",
    "n",
    "mean",
    "sd",
    "cv_indiv",
    "censoring",
    "source_file",
    "rule_applied",
]

ASSIGNED_DATASETS = [
    "D11_trematode_taskinen2022",
    "D12_trematode_gleichsner2016",
    "D13_nematode_marshallagia2023",
    "D14_nematode_heligmosomoides",
    "D15_helminth_clc_benesh2017",
]


def fmt_float(val: float | str | None, digits: int = 6) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        return val
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def make_row(
    dataset: str,
    taxon: str,
    stage: str,
    stage_category: str,
    temperature_c: float | str,
    n: int | str | None,
    mean: float,
    sd: float,
    cv_indiv: float,
    censoring: str,
    source_file: str,
    rule_applied: str,
) -> dict[str, object]:
    temp_str = (
        f"{temperature_c:.2f}"
        if isinstance(temperature_c, (int, float)) and math.isfinite(float(temperature_c))
        else str(temperature_c)
    )
    n_out: int | str = "" if n in (None, "") else int(n)
    return {
        "dataset": dataset,
        "taxon": taxon,
        "stage": stage,
        "stage_category": stage_category,
        "temperature_C": temp_str,
        "n": n_out,
        "mean": fmt_float(mean, 6),
        "sd": fmt_float(sd, 6),
        "cv_indiv": fmt_float(cv_indiv, 6),
        "censoring": censoring,
        "source_file": source_file,
        "rule_applied": rule_applied,
    }


def verify_helpers_known_results() -> list[str]:
    """Rule 7: verify `extract_helpers` estimators on known analytical cases before extraction."""
    lines = ["=== Rule 7 Known-Result Self-Checks (tolerance stated beforehand) ==="]
    st = eh.uncensored_stats([10.0, 20.0, 30.0])
    expected_cv = 0.5 * (1.0 + 1.0 / 12.0)
    assert abs(st["mean"] - 20.0) < 1e-12
    assert abs(st["sd"] - 10.0) < 1e-12
    assert abs(st["cv_indiv"] - expected_cv) < 1e-12
    lines.append(
        f"  [PASS] uncensored_stats([10,20,30]): mean={st['mean']:.6f} (exp 20), "
        f"sd={st['sd']:.6f} (exp 10), cv_indiv={st['cv_indiv']:.8f} (exp {expected_cv:.8f}, tol 1e-12)"
    )

    rng = np.random.default_rng(20261114)
    t_true = rng.exponential(scale=50.0, size=4000)
    c_time = 69.3147
    t_obs = np.minimum(t_true, c_time)
    ev = t_true <= c_time
    wm = eh.weibull_censored_mle(t_obs, ev)
    assert abs(wm["cv_mle"] - 1.0) < 0.05
    assert abs(wm["mean_mle"] - 50.0) < 2.5
    lines.append(
        f"  [PASS] weibull_censored_mle(Exp(50), ~25% right-censored, n=4000): "
        f"mean_mle={wm['mean_mle']:.3f} (exp 50.0, tol 2.5), cv_mle={wm['cv_mle']:.4f} (exp 1.0000, tol 0.05)"
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Dataset D11: Rhipidocotyle fennica & R. campanula in Anodonta anatina
# (Taskinen et al. 2022 Parasitology, Figshare 19584286)
# ---------------------------------------------------------------------------

def extract_d11(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D11_trematode_taskinen2022"
    d11_dir = RAW_DIR / ds
    shed_path = d11_dir / "Data_shedding_mussels"
    all_path = d11_dir / "Data_all_mussels"
    temp_path = d11_dir / "Mean_shedding_vs_mean_temperature"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Taskinen et al. 2022 Parasitology, doi:10.1017/S0031182022000518)")
    log.append(
        "Source files: D11_trematode_taskinen2022/Data_shedding_mussels, Data_all_mussels, "
        "Mean_shedding_vs_mean_temperature, Mussels_trematodes_temperature_data_description.odt"
    )
    log.append(
        "Stage category: S2_parasite_in_invertebrate_vector_or_snail (patent cercarial shedding duration "
        "in individual Anodonta anatina mussel hosts across 12 biweekly trials, Days 1..151 / t=0..150 d)."
    )
    log.append("========================================================================")

    # Read seasonal tank temperature trajectories (12 trials per treatment)
    with temp_path.open(encoding="utf-8") as f:
        mrows = list(csv.DictReader(f))
    tr_temps: dict[str, float] = {}
    tr_ranges: dict[str, tuple[float, float]] = {}
    for tr in ("High temperature", "Intermediate temperature", "Low temperature"):
        ts = [
            float(r["Mean_temp"])
            for r in mrows
            if r["Treatments_shedding"].strip() == tr and r["Trematode_species"].strip() == "RF"
        ]
        tr_temps[tr] = float(np.mean(ts))
        tr_ranges[tr] = (min(ts), max(ts))

    with all_path.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_notshed = sum(1 for r in all_rows if r["Shedding_category"].strip() == "notshedding")
    n_coinf = sum(1 for r in all_rows if r["Shedding_category"].strip() == "R. campanula and R. fennica")

    log.append(
        f"[EXCLUSION] {ds} | Data_all_mussels ({n_notshed} 'notshedding' rows + {n_coinf} co-infected "
        "'R. campanula and R. fennica' rows; binary Survival column across 571 mussels) | "
        "reason=Uninfected mussels have no shedding duration, co-infected mussels lack species-resolved "
        "shedding start/stop dates in Data_shedding_mussels, and Survival is end-of-study binary status "
        "without individual mortality day (Protocol §4.1 Rule 6)."
    )
    log.append(
        f"[EXCLUSION] {ds} | Mean_shedding_vs_mean_temperature (72 rows) | "
        "reason=Treatment-mean cercarial emission rate curves without individual spread (Rule 6; "
        "used only for verified seasonal tank temperatures)."
    )
    log.append(
        f"[EXCLUSION] {ds} | Data_shedding_mussels columns start_RF, start_rc, Stop_RF, stop_RC | "
        "reason=Calendar phenology day measured from arbitrary tank-experiment Day 0 on wild-caught "
        "perennial mussels naturally infected in prior years (unknown infection date, not prepatent "
        "stage duration; Protocol §4.1 Rule 6)."
    )

    trials = [0, 14, 27, 42, 57, 69, 82, 96, 110, 125, 136, 150]
    idx_map = {t: i for i, t in enumerate(trials)}

    with shed_path.open(encoding="utf-8") as f:
        srows = list(csv.DictReader(f))
    assert len(srows) == 166

    species_meta = [
        ("R. fennica", "Rhipidocotyle fennica (in Anodonta anatina)", "start_RF", "Stop_RF", "RF_duration"),
        ("Rhipidodoctyle campanula", "Rhipidocotyle campanula (in Anodonta anatina)", "start_rc", "stop_RC", "RC_duration"),
    ]
    d11_added: list[dict[str, object]] = []

    for raw_sp, taxon_lbl, st_col, sp_col, du_col in species_meta:
        for pop in ("Haajaistenjoki", "Kuusaankoski"):
            for tr in ("High temperature", "Intermediate temperature", "Low temperature"):
                sub = [
                    r for r in srows
                    if r["Shedding_category"].strip() == raw_sp
                    and r["mussel_pop"].strip() == pop
                    and r["treatment"].strip() == tr
                ]
                valid: list[tuple[float, bool, str, int, int, float]] = []
                n_left_trunc = 0
                for r in sub:
                    s = int(r[st_col])
                    e = int(r[sp_col])
                    if s == 0:
                        n_left_trunc += 1
                        continue
                    si = idx_map[s]
                    ei = idx_map[e]
                    # Midpoint onset & termination across biweekly trials (matching D1 Amendment 3),
                    # avoiding artificial 0-day duration truncation on single-trial shedders (s == e).
                    t0 = 0.5 * (trials[si - 1] + trials[si])
                    if ei == len(trials) - 1:
                        t1 = float(trials[ei])
                        ev = False
                    else:
                        t1 = 0.5 * (trials[ei] + trials[ei + 1])
                        ev = True
                    valid.append((t1 - t0, ev, r["Mussel_ID"].strip(), s, e, float(r[du_col])))

                tr_short = tr.lower().replace(" ", "_")
                cell_lbl = f"patent_cercarial_shedding_duration_{pop}_{tr_short}_seasonal_tank"
                if n_left_trunc > 0:
                    log.append(
                        f"[EXCLUSION] {ds} | {taxon_lbl} {cell_lbl} ({n_left_trunc}/{len(sub)} rows with {st_col}==0) | "
                        "reason=Left-truncated shedding onset: mussels were already shedding cercariae on Day 1 (t=0) "
                        "prior to experimental tank regime (matching Amendment 3 left-truncation rule)."
                    )

                if len(valid) < 10:
                    log.append(
                        f"[EXCLUSION] {ds} | {taxon_lbl} {cell_lbl} (n={len(valid)} non-left-truncated, raw_total={len(sub)}) | "
                        "reason=Cohort sample size n < 10 (Protocol §4.1 Rule 1)."
                    )
                    continue

                t_arr = np.array([x[0] for x in valid], dtype=float)
                ev_arr = np.array([x[1] for x in valid], dtype=bool)
                cfrac = 1.0 - float(ev_arr.mean())
                temp_c = tr_temps[tr]
                tmin, tmax = tr_ranges[tr]
                raw_pos = [x[5] for x in valid if x[5] > 0.0]
                raw_pos_cv = eh.uncensored_stats(raw_pos)["cv_indiv"] if len(raw_pos) >= 2 else math.nan

                if cfrac <= 0.05:
                    st = eh.uncensored_stats(t_arr, dt_obs=14.0)
                    row = make_row(
                        dataset=ds,
                        taxon=taxon_lbl,
                        stage=f"{cell_lbl} (range {tmin:.1f}-{tmax:.1f}C)",
                        stage_category="S2_parasite_in_invertebrate_vector_or_snail",
                        temperature_c=temp_c,
                        n=st["n"],
                        mean=st["mean"],
                        sd=st["sd"],
                        cv_indiv=st["cv_indiv"],
                        censoring="uncensored_biweekly_midpoint",
                        source_file="D11_trematode_taskinen2022/Data_shedding_mussels",
                        rule_applied=(
                            f"Rule 1: raw_individual_uncensored (biweekly midpoint Sokal-Rohlf; "
                            f"cv_sheppard={st['cv_sheppard']:.4f}, multicheck_StopMinusStart_cv={raw_pos_cv:.4f})"
                        ),
                    )
                    rows_out.append(row)
                    d11_added.append(row)
                elif cfrac <= 0.50 and int(ev_arr.sum()) >= 10:
                    wm = eh.weibull_censored_mle(t_arr, ev_arr)
                    st_u = eh.uncensored_stats(t_arr[ev_arr], dt_obs=14.0)
                    row = make_row(
                        dataset=ds,
                        taxon=taxon_lbl,
                        stage=f"{cell_lbl} (range {tmin:.1f}-{tmax:.1f}C)",
                        stage_category="S2_parasite_in_invertebrate_vector_or_snail",
                        temperature_c=temp_c,
                        n=wm["n_total"],
                        mean=wm["mean_mle"],
                        sd=wm["sd_mle"],
                        cv_indiv=wm["cv_mle"],
                        censoring=f"right_censored_{wm['cens_frac']:.3f}_d{wm['n_events']}_at_t150",
                        source_file="D11_trematode_taskinen2022/Data_shedding_mussels",
                        rule_applied=(
                            f"Rule 2: raw_individual_censored (Weibull MLE, 1 mussel shedding at final t=150d; "
                            f"completers_cv={st_u['cv_indiv']:.4f})"
                        ),
                    )
                    rows_out.append(row)
                    d11_added.append(row)
                else:
                    log.append(
                        f"[EXCLUSION] {ds} | {taxon_lbl} {cell_lbl} (n={len(valid)}, cens={cfrac:.3f}) | "
                        "reason=Fails Protocol §4.1 Rule 2 right-censoring ceiling (>0.50) or <10 uncensored events."
                    )

    assert len(d11_added) == 5
    # Verify exact hand-checked arithmetic on 3 rows
    r_rf_hi = d11_added[1]  # R. fennica Haajaistenjoki Intermediate
    assert r_rf_hi["n"] == 19 and abs(float(r_rf_hi["mean"]) - (848.5 / 19.0)) < 1e-6
    r_rf_ki = d11_added[3]  # R. fennica Kuusaankoski Intermediate
    assert r_rf_ki["n"] == 13 and abs(float(r_rf_ki["cv_indiv"]) - 0.352768) < 1e-6
    r_rc_hl = d11_added[4]  # R. campanula Haajaistenjoki Low
    assert r_rc_hl["n"] == 10 and abs(float(r_rc_hl["mean"]) - 67.4) < 1e-6

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D11_trematode_taskinen2022) ---")
    log.append(
        "  [HAND CHECK D11-1] Rhipidocotyle campanula Haajaistenjoki Low temperature "
        "(Data_shedding_mussels lines 73..98): 20 R. campanula rows minus 10 left-truncated with start_rc==0 "
        "= 10 incident shedders (Mussel_ID 197,228,233,250,258,264,267,275,278,284); biweekly midpoint durations "
        "[97.0, 82.0, 29.0, 13.5, 68.5, 68.5, 110.0, 83.0, 40.0, 82.5] -> sum=674.0, sum(t^2)=53936.0, "
        "n=10, mean=67.400000 d, sd=sqrt((53936-674^2/10)/9)=30.746996 d, "
        "cv_indiv=(30.746996/67.4)*(1+1/40)=0.467592; 12-trial tank mean T=12.17 C. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D11-2] Rhipidocotyle fennica Haajaistenjoki Intermediate temperature "
        "(Data_shedding_mussels lines 43..72): 19 R. fennica rows, all start_RF>=42 and Stop_RF<=136 (0 censored); "
        "biweekly midpoint durations sum=848.5 d, sum(t^2)=47578.75 -> n=19, mean=848.5/19=44.657895 d, "
        "sd=sqrt((47578.75-848.5^2/19)/18)=23.197852 d, cv_indiv=(23.197852/44.657895)*(1+1/76)=0.526292; "
        "12-trial tank mean T=13.84 C. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D11-3] Rhipidocotyle fennica Kuusaankoski Intermediate temperature "
        "(Data_shedding_mussels lines 139..155): 13 R. fennica rows (12 cleared with Stop_RF<=136; 1 right-censored "
        "at final check Stop_RF=150 on Mussel_ID=445, start_RF=96 -> midpoint start 89.0 to 150.0 = 61.0+ d, "
        "cens=1/13=7.69% > 5% -> Rule 2 Weibull MLE): n=13 (d=12), mean_mle=51.662960 d, sd_mle=18.225022 d, "
        "cv_indiv=0.352768 (completers-only CV=0.3843). Matches CSV row."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D12: Schistosoma mansoni in Biomphalaria glabrata
# (Gleichsner, Cleveland & Minchella 2016 Evolution, Zenodo 4993047)
# ---------------------------------------------------------------------------

def extract_d12(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D12_trematode_gleichsner2016"
    taxon = "Schistosoma mansoni (in Biomphalaria glabrata)"
    csv_path = RAW_DIR / ds / "Gleichsner_et_al._RawData.csv"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Gleichsner et al. 2016 Evolution, doi:10.1111/evo.13061)")
    log.append(
        "Source files: D12_trematode_gleichsner2016/Gleichsner_et_al._RawData.csv, AestivationDataRcode.R"
    )
    log.append(
        "Applies Protocol Amendment 1 item 1 (cols 9-11 read independently of cols 0-5 and 6-8). "
        "Lab temperature is not stated in raw CSV, R script, Zenodo metadata, or open abstract -> temperature_C='' (study rules Rule 2)."
    )
    log.append("========================================================================")

    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))

    t1_cnt = sum(1 for r in raw_rows[1:] if any(c.strip() for c in r[:6]))
    t2_rows = [(i + 2, r[6].strip(), int(r[7]), float(r[8])) for i, r in enumerate(raw_rows[1:]) if r[6].strip()]
    t3_rows = [(i + 2, r[9].strip(), float(r[10]), float(r[11])) for i, r in enumerate(raw_rows[1:]) if r[9].strip()]
    assert t1_cnt == 655 and len(t2_rows) == 243 and len(t3_rows) == 81

    log.append(
        f"[EXCLUSION] {ds} | Gleichsner_et_al._RawData.csv cols 0-5 ({t1_cnt} rows: Growth, PrePatent/Patent/Total Egg Pouches) | "
        "reason=Host shell growth (mm) and egg-pouch fecundity counts, not stage durations (Protocol §4.1 Rule 6 & Amendment 1)."
    )
    log.append(
        f"[EXCLUSION] {ds} | Gleichsner_et_al._RawData.csv cols 6-8 ({len(t2_rows)} rows: weekly Cercariae at Time=4,5,6 wk) | "
        "reason=Cercarial counts assayed only across fixed Weeks 4-6 post-infection; experiment terminated at Week 6 "
        "while snails were actively shedding (100% right-truncated patent duration, Protocol §4.1 Rule 2 & Rule 6)."
    )
    log.append(
        f"[EXCLUSION] {ds} | Gleichsner_et_al._RawData.csv col 10 (Total Cercariae across 81 snails) | "
        "reason=Cumulative cercarial output count rather than a stage duration (Protocol §4.1 Rule 6)."
    )

    by_tr: dict[str, list[float]] = defaultdict(list)
    for _, tr, tot_cerc, pre_wk in t3_rows:
        assert tot_cerc > 0.0 and pre_wk > 0.0
        by_tr[tr].append(pre_wk * 7.0)  # weeks -> days (TIME_UNIT_TO_DAYS['weeks'] = 7.0)

    d12_added: list[dict[str, object]] = []
    for tr in ("Stressed", "Unstressed"):
        vals_d = by_tr[tr]
        st = eh.uncensored_stats(vals_d, dt_obs=7.0)
        tr_desc = "drought_aestivation_stressed" if tr == "Stressed" else "unstressed_control"
        row = make_row(
            dataset=ds,
            taxon=taxon,
            stage=f"intramolluscan_prepatent_period_{tr_desc} (weekly checks, Weeks 4-6)",
            stage_category="S2_parasite_in_invertebrate_vector_or_snail",
            temperature_c="",
            n=st["n"],
            mean=st["mean"],
            sd=st["sd"],
            cv_indiv=st["cv_indiv"],
            censoring="uncensored_weekly",
            source_file="D12_trematode_gleichsner2016/Gleichsner_et_al._RawData.csv",
            rule_applied=(
                f"Rule 1: raw_individual_uncensored (Amendment 1 cols 9-11, weeks*7d, Sokal-Rohlf; "
                f"cv_sheppard={st['cv_sheppard']:.4f})"
            ),
        )
        rows_out.append(row)
        d12_added.append(row)

    assert d12_added[0]["n"] == 56 and abs(float(d12_added[0]["mean"]) - 31.625) < 1e-6
    assert d12_added[1]["n"] == 25 and abs(float(d12_added[1]["mean"]) - 32.2) < 1e-6

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D12_trematode_gleichsner2016) ---")
    log.append(
        "  [HAND CHECK D12-1] intramolluscan_prepatent_period_drought_aestivation_stressed "
        "(Gleichsner_et_al._RawData.csv cols 9-11, lines 2..57): 56 Stressed snails with Pre-Patent period "
        "[4 wk x 28, 5 wk x 27, 6 wk x 1] = [28 d x 28, 35 d x 27, 42 d x 1] -> sum=1771 d (253 wk), "
        "sum(t^2)=56791 d^2, n=56, mean=1771/56=31.625000 d, sd=sqrt((56791-1771^2/56)/55)=3.773412 d, "
        "cv_indiv=(3.773412/31.625)*(1+1/224)=0.119850. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D12-2] intramolluscan_prepatent_period_unstressed_control "
        "(Gleichsner_et_al._RawData.csv cols 9-11, lines 58..82): 25 Unstressed snails with Pre-Patent period "
        "[4 wk x 12, 5 wk x 11, 6 wk x 2] = [28 d x 12, 35 d x 11, 42 d x 2] -> sum=805 d (115 wk), "
        "sum(t^2)=26411 d^2, n=25, mean=805/25=32.200000 d, sd=sqrt((26411-805^2/25)/24)=4.518481 d, "
        "cv_indiv=(4.518481/32.2)*(1+1/100)=0.141729. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D12-3] Excluded weekly cercarial table verification "
        "(Gleichsner_et_al._RawData.csv cols 6-8, lines 2..244): 81 snails x 3 weeks (Time=4,5,6) = 243 rows; "
        "final lines 242-244 ('Unstressed,6,786', 'Unstressed,6,802', 'Unstressed,6,835') confirm active shedding "
        "at final Week 6 without post-Week-6 follow-up, verifying exclusion of patent shedding duration."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D13: Marshallagia marshalli free-living egg hatching, L3 development,
# and pre-L3 larval survival across 5-38 C (Aleuy et al. 2023 GCB, Zenodo 8255690)
# ---------------------------------------------------------------------------

def extract_d13(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D13_nematode_marshallagia2023"
    taxon = "Marshallagia marshalli"
    cat = "S3_free_living_parasite_stage"
    csv_path = RAW_DIR / ds / "Marshallagia_develop_Morta.csv"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Aleuy et al. 2023 Global Change Biology, doi:10.1111/gcb.16918)")
    log.append(
        "Source files: D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv, README.md"
    )
    log.append(
        "Stage category: S3_free_living_parasite_stage (egg-to-L1 hatching time, egg-to-L3 development "
        "time, and pre-L3 egg/larval survival across 6 experimental location-batches x 5-38 C)."
    )
    log.append("========================================================================")

    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    assert len(raw_rows) == 2240

    by_lt: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    n_direct_l3 = 0
    n_dead_unhatched = 0
    n_dead_posthatch = 0
    for r in raw_rows:
        by_lt[(r["Location2"].strip(), float(r["Temperature"]))].append(r)
        if r["Hatching_L3_BINO"].strip() == "YES":
            n_direct_l3 += 1
        if r["Dead_before_L3"].strip() != "NA":
            if r["Hatching"].strip() == "NA":
                n_dead_unhatched += 1
            else:
                n_dead_posthatch += 1

    log.append(
        f"[EXCLUSION] {ds} | egg_to_L1_hatching_time ({n_direct_l3} direct-to-L3 eggs with Hatching_L3_BINO=='YES' "
        f"+ 1 row line 350 with Hatching=='NA', plus {n_dead_unhatched} eggs dying unhatched) | "
        "reason=Direct-L3 hatching bypasses free-living L1 emergence (included in egg_to_L3_development_time instead), "
        "and unhatched deaths failed to complete hatching (recorded in pre_L3_egg_larval_survival_time)."
    )
    log.append(
        f"[EXCLUSION] {ds} | egg_to_L3_development_time ({n_dead_unhatched + n_dead_posthatch} pre-L3 deaths: "
        f"{n_dead_unhatched} unhatched + {n_dead_posthatch} post-hatch L1/L2 deaths) | "
        "reason=Individuals died before completing L3 development; completion duration is conditional on reaching L3."
    )

    d13_added: list[dict[str, object]] = []
    surv_high_cens_cnt = 0
    n_warm_lost_wells = 0

    for (loc, temp_c), sub in sorted(by_lt.items()):
        obs_days_plate = [
            float(r[c])
            for r in sub
            for c in ("Hatching", "L3_ocurrence", "Dead_before_L3")
            if r[c].strip() != "NA"
        ]
        max_obs_plate = max(obs_days_plate) if obs_days_plate else 0.0

        # Identify wells lost mid-run on warm plates (T >= 15 C) where batch-level 5 C Final_day
        # (160 or 249 d, > 4x max_obs_plate of 15-33 d) was filled down without an actual censoring day.
        def _valid_end_censored(r: dict[str, str]) -> bool:
            if r["censored"].strip() != "YES":
                return False
            fd = float(r["Final_day"])
            if temp_c >= 15.0 and max_obs_plate > 0.0 and fd > 3.0 * max_obs_plate:
                return False
            return True

        n_warm_lost_wells += sum(
            1 for r in sub if r["censored"].strip() == "YES" and not _valid_end_censored(r)
        )

        # 1. Egg-to-L1 hatching time
        h_items: list[tuple[float, bool, float | None]] = []
        for r in sub:
            if r["Hatching"].strip() != "NA":
                a, b = float(r["last_check_hatch"]), float(r["Hatching"])
                h_items.append((0.5 * (a + b), True, b - a))
            elif _valid_end_censored(r):
                h_items.append((float(r["Final_day"]), False, None))

        d_h = sum(1 for _, ev, _ in h_items if ev)
        n_h = len(h_items)
        cf_h = 1.0 - (d_h / n_h) if n_h > 0 else 1.0
        if d_h < 10 or cf_h > 0.50:
            log.append(
                f"[EXCLUSION] {ds} | egg_to_L1_hatching_time_{loc} T={temp_c:g}C (n={n_h}, d={d_h}, cens={cf_h:.3f}) | "
                "reason=Fewer than 10 hatched eggs (100% lethal heat at 38 C) or right-censoring > 0.50 (Protocol §4.1 Rules 1-2)."
            )
        else:
            t_arr = np.array([x[0] for x in h_items], dtype=float)
            ev_arr = np.array([x[1] for x in h_items], dtype=bool)
            med_dt = float(np.median([x[2] for x in h_items if x[2] is not None]))
            if cf_h <= 0.05:
                st = eh.uncensored_stats(t_arr[ev_arr], dt_obs=med_dt)
                c_lbl = "uncensored" if cf_h == 0.0 else f"right_censored_{cf_h:.3f}_le_0.05_d{d_h}"
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"egg_to_L1_hatching_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=st["n"],
                    mean=st["mean"],
                    sd=st["sd"],
                    cv_indiv=st["cv_indiv"],
                    censoring=c_lbl,
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=f"Rule 1: raw_individual_uncensored (interval midpoint Sokal-Rohlf; cv_sheppard={st['cv_sheppard']:.4f})",
                )
            else:
                wm = eh.weibull_censored_mle(t_arr, ev_arr)
                st_u = eh.uncensored_stats(t_arr[ev_arr], dt_obs=med_dt)
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"egg_to_L1_hatching_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=wm["n_total"],
                    mean=wm["mean_mle"],
                    sd=wm["sd_mle"],
                    cv_indiv=wm["cv_mle"],
                    censoring=f"right_censored_{wm['cens_frac']:.3f}_d{wm['n_events']}",
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=f"Rule 2: raw_individual_censored (Weibull MLE; completers_cv={st_u['cv_indiv']:.4f})",
                )
            rows_out.append(row)
            d13_added.append(row)

        # 2. Egg-to-L3 development time
        l_items: list[tuple[float, bool, float | None]] = []
        for r in sub:
            if r["L3_ocurrence"].strip() != "NA":
                a, b = float(r["last_check_L3"]), float(r["L3_ocurrence"])
                l_items.append((0.5 * (a + b), True, b - a))
            elif _valid_end_censored(r):
                l_items.append((float(r["Final_day"]), False, None))

        d_l = sum(1 for _, ev, _ in l_items if ev)
        n_l = len(l_items)
        cf_l = 1.0 - (d_l / n_l) if n_l > 0 else 1.0
        if d_l < 10 or cf_l > 0.50:
            log.append(
                f"[EXCLUSION] {ds} | egg_to_L3_development_time_{loc} T={temp_c:g}C (n={n_l}, d={d_l}, cens={cf_l:.3f}) | "
                "reason=Fewer than 10 L3 completions or right-censoring > 0.50 (Protocol §4.1 Rules 1-2)."
            )
        else:
            t_arr = np.array([x[0] for x in l_items], dtype=float)
            ev_arr = np.array([x[1] for x in l_items], dtype=bool)
            dts_l = [x[2] for x in l_items if x[2] is not None]
            med_dt = float(np.median(dts_l))
            max_dt = float(np.max(dts_l))
            gap_note = f"; late_check_gap_max={max_dt:.0f}d" if max_dt > 14.0 else ""
            if cf_l <= 0.05:
                st = eh.uncensored_stats(t_arr[ev_arr], dt_obs=med_dt)
                c_lbl = "uncensored" if cf_l == 0.0 else f"right_censored_{cf_l:.3f}_le_0.05_d{d_l}"
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"egg_to_L3_development_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=st["n"],
                    mean=st["mean"],
                    sd=st["sd"],
                    cv_indiv=st["cv_indiv"],
                    censoring=c_lbl,
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=(
                        f"Rule 1: raw_individual_uncensored (interval midpoint Sokal-Rohlf; "
                        f"cv_sheppard={st['cv_sheppard']:.4f}{gap_note})"
                    ),
                )
            else:
                wm = eh.weibull_censored_mle(t_arr, ev_arr)
                st_u = eh.uncensored_stats(t_arr[ev_arr], dt_obs=med_dt)
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"egg_to_L3_development_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=wm["n_total"],
                    mean=wm["mean_mle"],
                    sd=wm["sd_mle"],
                    cv_indiv=wm["cv_mle"],
                    censoring=f"right_censored_{wm['cens_frac']:.3f}_d{wm['n_events']}",
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=(
                        f"Rule 2: raw_individual_censored (Weibull MLE; completers_cv={st_u['cv_indiv']:.4f}{gap_note})"
                    ),
                )
            rows_out.append(row)
            d13_added.append(row)

        # 3. Pre-L3 egg/larval survival time (right-censored upon L3 completion or Final_day)
        s_items: list[tuple[float, bool, float | None]] = []
        for r in sub:
            if r["Dead_before_L3"].strip() != "NA":
                a, b = float(r["last_check_Mortality"]), float(r["Dead_before_L3"])
                s_items.append((0.5 * (a + b), True, b - a))
            elif r["L3_ocurrence"].strip() != "NA":
                a, b = float(r["last_check_L3"]), float(r["L3_ocurrence"])
                s_items.append((0.5 * (a + b), False, None))
            elif _valid_end_censored(r):
                s_items.append((float(r["Final_day"]), False, None))

        d_s = sum(1 for _, ev, _ in s_items if ev)
        n_s = len(s_items)
        cf_s = 1.0 - (d_s / n_s) if n_s > 0 else 1.0
        if d_s < 10 or cf_s > 0.50:
            surv_high_cens_cnt += 1
        else:
            t_arr = np.array([x[0] for x in s_items], dtype=float)
            ev_arr = np.array([x[1] for x in s_items], dtype=bool)
            med_dt = float(np.median([x[2] for x in s_items if x[2] is not None]))
            st_u = eh.uncensored_stats(t_arr[ev_arr], dt_obs=med_dt)
            if st_u["sd"] == 0.0:
                log.append(
                    f"[EXCLUSION] {ds} | pre_L3_egg_larval_survival_time_{loc} T={temp_c:g}C (n={n_s}, d={d_s}) | "
                    "reason=All 40/40 eggs recorded dead in the identical check interval (5..6 d, SD=0.0); "
                    "within-interval survival spread unidentifiable (Protocol §4.1 Rule 6 & §6.2 P2a)."
                )
            elif cf_s <= 0.05:
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"pre_L3_egg_larval_survival_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=st_u["n"],
                    mean=st_u["mean"],
                    sd=st_u["sd"],
                    cv_indiv=st_u["cv_indiv"],
                    censoring="uncensored" if cf_s == 0.0 else f"right_censored_{cf_s:.3f}_le_0.05_d{d_s}",
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=f"Rule 1: raw_individual_uncensored (interval midpoint Sokal-Rohlf; cv_sheppard={st_u['cv_sheppard']:.4f})",
                )
                rows_out.append(row)
                d13_added.append(row)
            else:
                wm = eh.weibull_censored_mle(t_arr, ev_arr)
                row = make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"pre_L3_egg_larval_survival_time_{loc}_constant_lab",
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=wm["n_total"],
                    mean=wm["mean_mle"],
                    sd=wm["sd_mle"],
                    cv_indiv=wm["cv_mle"],
                    censoring=f"right_censored_{wm['cens_frac']:.3f}_d{wm['n_events']}_at_L3_or_end",
                    source_file="D13_nematode_marshallagia2023/Marshallagia_develop_Morta.csv",
                    rule_applied=f"Rule 2: raw_individual_censored (Weibull MLE; completers_cv={st_u['cv_indiv']:.4f})",
                )
                rows_out.append(row)
                d13_added.append(row)

    assert n_warm_lost_wells == 8
    log.append(
        f"[EXCLUSION] {ds} | Marshallagia_develop_Morta.csv ({n_warm_lost_wells} singleton warm-plate wells at "
        "T >= 15 C with censored=='YES': lines 108, 872, 1014, 1291, 2002, 2047, 2112, 2123) | "
        "reason=Lost/empty wells with unrecorded censoring date (batch 5 C Final_day=160 or 249 d filled down "
        "despite warm plates completing all active observations within Days 15-33; Protocol §4.1 Rule 6)."
    )
    log.append(
        f"[EXCLUSION] {ds} | pre_L3_egg_larval_survival_time ({surv_high_cens_cnt}/56 location x temperature cohorts) | "
        "reason=Fewer than 10 pre-L3 deaths or >50% of cohort exited observation alive upon reaching L3 "
        "(wells were not monitored for post-L3 longevity once L3_ocurrence was reached; Protocol §4.1 Rule 2)."
    )

    assert len(d13_added) == 105
    r_c15_h = next(
        r for r in d13_added
        if r["stage"] == "egg_to_L1_hatching_time_Colorado_constant_lab" and float(r["temperature_C"]) == 15.0
    )
    assert r_c15_h["n"] == 35 and abs(float(r_c15_h["mean"]) - (226.5 / 35.0)) < 1e-6
    r_d20_l = next(
        r for r in d13_added
        if r["stage"] == "egg_to_L3_development_time_Dawson_constant_lab" and float(r["temperature_C"]) == 20.0
    )
    assert r_d20_l["n"] == 39 and abs(float(r_d20_l["mean"]) - (375.5 / 39.0)) < 1e-6
    r_d35_s = next(
        r for r in d13_added
        if r["stage"] == "pre_L3_egg_larval_survival_time_Dawson_constant_lab" and float(r["temperature_C"]) == 35.0
    )
    assert r_d35_s["n"] == 40 and abs(float(r_d35_s["cv_indiv"]) - 0.359973) < 1e-6

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D13_nematode_marshallagia2023) ---")
    log.append(
        "  [HAND CHECK D13-1] egg_to_L1_hatching_time_Colorado_constant_lab at 15.00 C "
        "(Marshallagia_develop_Morta.csv lines 42..81): 40 wells (5 died unhatched, 35 hatched, 0 censored); "
        "interval midpoints 0.5*(last_check_hatch + Hatching) across 35 hatched eggs -> sum=226.5 d, "
        "sum(t^2)=1550.75 d^2, n=35, mean=226.5/35=6.471429 d, sd=sqrt((1550.75-226.5^2/35)/34)=1.580873 d, "
        "cv_indiv=(1.580873/6.471429)*(1+1/140)=0.246030. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D13-2] egg_to_L3_development_time_Dawson_constant_lab at 20.00 C "
        "(Marshallagia_develop_Morta.csv lines 482..521): 40 wells (1 died pre-L3, 39 reached L3, 0 censored); "
        "interval midpoints 0.5*(last_check_L3 + L3_ocurrence) across 39 L3 completers -> sum=375.5 d, "
        "sum(t^2)=3687.75 d^2, n=39, mean=375.5/39=9.628205 d, sd=sqrt((3687.75-375.5^2/39)/38)=1.379922 d, "
        "cv_indiv=(1.379922/9.628205)*(1+1/156)=0.144239. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D13-3] pre_L3_egg_larval_survival_time_Dawson_constant_lab at 35.00 C "
        "(Marshallagia_develop_Morta.csv lines 642..681): 40 wells (34 observed pre-L3 deaths, 6 reached L3 alive "
        "and right-censored at L3 midpoint, cens=6/40=15.0% in (5%, 50%] -> Rule 2 Weibull MLE): "
        "n=40 (d=34), mean_mle=8.450659 d, sd_mle=3.042007 d, cv_indiv=0.359973 (shape k=3.0323). Matches CSV row."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D14: Heligmosomoides polygyrus in Mus musculus
# (Lippens et al. 2016/2017, Zenodo 4951702 & 5013889)
# ---------------------------------------------------------------------------

def extract_d14(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D14_nematode_heligmosomoides"
    d14_dir = RAW_DIR / ds
    p_dryad = d14_dir / "dryad_data.xlsx"
    p_jeb = d14_dir / "data_lippens_et_al_jeb.xlsx"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Lippens et al. 2017 Biol. Lett. & 2017 J. Exp. Biol., Zenodo 4951702 & 5013889)")
    log.append("Source files: D14_nematode_heligmosomoides/dryad_data.xlsx, data_lippens_et_al_jeb.xlsx")
    log.append(
        "Applies Protocol §4.1 Rule 6 and Amendment 1 item 3: cross-sectional worm counts at fixed "
        "sacrifice days without individual survival or patency termination times -> 0 primary CV_indiv rows."
    )
    log.append("========================================================================")

    s_dryad = eh.read_xlsx_sheet(p_dryad, 0)
    assert len(s_dryad) == 121  # header + 120 rows (20 mice x 6 FEC days)
    dryad_worm_rows = [r for r in s_dryad[1:] if len(r) >= 5 and r[4].strip() != ""]
    assert len(dryad_worm_rows) == 20 and all(r[2].strip() == "28" for r in dryad_worm_rows)

    jeb_bm_larv = eh.read_xlsx_sheet(p_jeb, 0)
    jeb_bm_ad = eh.read_xlsx_sheet(p_jeb, 1)
    jeb_worms = eh.read_xlsx_sheet(p_jeb, 2)
    jeb_fec = eh.read_xlsx_sheet(p_jeb, 3)
    jeb_gene = eh.read_xlsx_sheet(p_jeb, 4)
    jeb_tgf_ad = eh.read_xlsx_sheet(p_jeb, 5)
    jeb_tgf_la = eh.read_xlsx_sheet(p_jeb, 6)
    assert len(jeb_worms) == 42  # header + 41 sacrificed mice

    log.append(
        f"[EXCLUSION] {ds} | dryad_data.xlsx Feuil1 (120 rows across 20 mice, Parental age 15/45/105/165 d) | "
        "reason=Fecal egg counts measured only across Days 11-28 post-infection while all 20 mice remained "
        "patent, and adult worm burden counted destructively at a single sacrifice day (Day 28 only, n=20 mice; "
        "Protocol §4.1 Rules 3 & 6, Amendment 1 item 3)."
    )
    log.append(
        f"[EXCLUSION] {ds} | data_lippens_et_al_jeb.xlsx sheet 'number of adult worms' (41 sacrificed mice: "
        "DSS+Hpoly Days 9,12 [n=9]; Hpoly Days 9,12,16,20 [n=20]; Hpoly+DSS Days 12,16,20 [n=12]) | "
        "reason=Cross-sectional worm counts per sacrificed mouse without known initial established adult count "
        "s0, N_mice < 50 per treatment, and non-monotonic worm counts across sacrifice days (e.g. Hpoly mean "
        "worms 68.4 at d9, 85.0 at d12, 92.4 at d16, 19.6 at d20; Hpoly+DSS 66.4 at d12, 92.2 at d16, 102.0 at d20), "
        "failing Protocol §4.1 Rule 3 & Amendment 1 item 3."
    )
    log.append(
        f"[EXCLUSION] {ds} | data_lippens_et_al_jeb.xlsx other sheets (bodymass: {len(jeb_bm_larv)-1}+{len(jeb_bm_ad)-1} rows; "
        f"fecal egg count Days 9-20: {len(jeb_fec)-1} rows; parasite gene expression: {len(jeb_gene)-1} rows; "
        f"mouseTGF: {len(jeb_tgf_ad)-1}+{len(jeb_tgf_la)-1} rows) | "
        "reason=Host body mass, fecal egg density over Days 9-20 without clearance, and gene/cytokine expression "
        "traits rather than stage durations (Protocol §4.1 Rule 6)."
    )

    # Verify 3 specific raw records across the two spreadsheets
    assert s_dryad[6][:5] == ["15", "ES11/A", "28", "19.239999999999998", "39"]
    assert jeb_worms[1][:4] == ["DSS+Hpoly", "DH12", "9", "101"]
    assert jeb_worms[-1][:4] == ["Hpoly+DSS", "HD34", "20", "106"]

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D14_nematode_heligmosomoides) ---")
    log.append(
        "  [HAND CHECK D14-1] dryad_data.xlsx Feuil1 rows 2..7 (mouse id='ES11/A', Parental age=15 d): "
        "fecal egg counts recorded at Days 11, 14, 18, 21, 25, 28 (14.62..19.24 eggs/mg, still positive at Day 28) "
        "with adult worms counted only on sacrifice row 7 (Day 28, number of adult worms=39; blank on Days 11-25). "
        "Confirms Amendment 1 item 3 exclusion."
    )
    log.append(
        "  [HAND CHECK D14-2] data_lippens_et_al_jeb.xlsx sheet 'number of adult worms' row 2 "
        "(treatment='DSS+Hpoly', mouse ID='DH12', time post-infection=9 d, number of adult worms=101; "
        "DSS+Hpoly sampled only at Days 9 [4 mice] and 12 [5 mice], 2 distinct days < 4). Confirms exclusion."
    )
    log.append(
        "  [HAND CHECK D14-3] data_lippens_et_al_jeb.xlsx sheet 'number of adult worms' row 42 "
        "(treatment='Hpoly+DSS', mouse ID='HD34', time post-infection=20 d, number of adult worms=106; "
        "Hpoly+DSS worm counts increase from mean 66.4 at Day 12 to 102.0 at Day 20 across 12 mice total). "
        "Confirms cross-sectional counts do not measure individual worm survival duration."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D15: Helminth complex life-cycle database
# (Benesh, Lafferty & Kuris 2017 Ecology 98:882, Figshare 30764579)
# ---------------------------------------------------------------------------

def extract_d15(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D15_helminth_clc_benesh2017"
    csv_path = RAW_DIR / ds / "CLC_database_lifehistory.csv"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Benesh, Lafferty & Kuris 2017 Ecology 98:882, Figshare 30764579)")
    log.append("Source file: D15_helminth_clc_benesh2017/CLC_database_lifehistory.csv")
    log.append(
        "Applies Protocol §3 & §4.1 Rule 6: species x stage x temperature point means without "
        "within-cohort individual SD/SE are flagged mean_only/range_only and never pooled into individual CVs."
    )
    log.append("========================================================================")

    with csv_path.open(encoding="latin1") as f:
        d15_rows = list(csv.DictReader(f))
    assert len(d15_rows) == 10010

    n_species = len({r["Parasite.species"].strip() for r in d15_rows})
    no_dt = [r for r in d15_rows if r["Development.time"].strip() in ("NA", "")]
    has_dt = [(i + 2, r) for i, r in enumerate(d15_rows) if r["Development.time"].strip() not in ("NA", "")]
    assert len(no_dt) == 7697 and len(has_dt) == 2313

    # Programmatic audit of Development.remarks & Size.remarks for any numeric SD / SE / +- spread
    num_pm_re = re.compile(r"\d+\s*(?:\+/-|\+\-|[\xb1])\s*\d+")
    sd_se_token_re = re.compile(r"(?:\bsd\b|\bs\.d\.|\bse\b|\bs\.e\.|\bsem\b|\bstd\b)", re.I)
    range_re = re.compile(r"(?:\bbetween\s+\d+|\b\d+\s*(?:-|to)\s*\d+\s*(?:d|day|dpi|dpe|wk|week|hr|hour))", re.I)

    n_pm_hits = 0
    n_sdse_hits = 0
    n_range_remarks = 0
    by_grp: dict[str, int] = defaultdict(int)
    for _, r in has_dt:
        by_grp[r["Parasite.group"].strip()] += 1
        rem = r["Development.remarks"].strip()
        if rem != "NA":
            if num_pm_hits_cond := bool(num_pm_re.search(rem)):
                n_pm_hits += int(num_pm_hits_cond)
            if sd_se_token_re.search(rem):
                n_sdse_hits += 1
            if range_re.search(rem):
                n_range_remarks += 1

    assert n_pm_hits == 0 and n_sdse_hits == 0
    n_mean_only = len(has_dt) - n_range_remarks

    log.append(
        f"[EXCLUSION] {ds} | CLC_database_lifehistory.csv rows with Development.time=='NA' ({len(no_dt)}/{len(d15_rows)} rows "
        f"across {n_species} helminth species) | reason=Morphometric/host-record entries without any reported development time (Protocol §4.1 Rule 6)."
    )
    log.append(
        f"[EXCLUSION] {ds} | CLC_database_lifehistory.csv point-estimate Development.time entries ({n_mean_only} mean_only rows "
        f"+ {n_range_remarks} range_only rows = {len(has_dt)} total: {by_grp['nematode']} nematode, {by_grp['cestode']} cestode, "
        f"{by_grp['acanthocephalan']} acanthocephalan) | "
        "reason=Database schema has no SD/SE column for Development.time (columns 'n' and 'Size.reported.as' refer to body "
        "Length/Width), programmatic audit of all 2,313 Development.remarks finds 0 entries with individual SD or SE "
        "(only point means/minima or min-max ranges), and max duplicate rows per (species, host, stage, temp, study) is 3 < 10; "
        "excluded as mean_only / range_only under Protocol §3 & §4.1 Rule 6."
    )

    # Verify 3 raw rows directly against CLC_database_lifehistory.csv (1-indexed file lines)
    r_line8 = d15_rows[6]      # file line 8
    r_line3398 = d15_rows[3396]  # file line 3398
    r_line4023 = d15_rows[4021]  # file line 4023
    assert r_line8["Parasite.species"] == "Abbreviata caucasica" and r_line8["Development.time"] == "55"
    assert r_line3398["Parasite.species"] == "Echinococcus multilocularis" and r_line3398["Development.time"] == "42.5"
    assert r_line4023["Parasite.species"] == "Glaridacris vogei" and r_line4023["Development.time"] == "54"

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D15_helminth_clc_benesh2017) ---")
    log.append(
        "  [HAND CHECK D15-1] CLC_database_lifehistory.csv line 8 (Abbreviata caucasica in Blatella germanica, "
        "Stage='1larv', Poinar & Quentin 1972): Development.time=55 d at Temp=20 C, "
        "Development.remarks='Physaloptera caucasica now?; achieve third stage sometime between 40 and 55 dpe', "
        "n=1 (for Length=1.7 mm, Size.reported.as='point'). Range/endpoint only without SD/SE -> excluded (Rule 6)."
    )
    log.append(
        "  [HAND CHECK D15-2] CLC_database_lifehistory.csv line 3398 (Echinococcus multilocularis in Microtus arvalis, "
        "Stage='1larv', Schmidt 2001): Development.time=42.5 d at Temp=38.5 C, Development.remarks='NA', "
        "n='NA'. Single point mean without within-cohort SD/SE -> excluded as mean_only (Protocol §3 & §4.1 Rule 6)."
    )
    log.append(
        "  [HAND CHECK D15-3] CLC_database_lifehistory.csv line 4023 (Glaridacris vogei, cestode Stage='1larv', "
        "Temp=21 C): Development.time=54 d, Development.remarks='DT was between 38 and 70 days, but source of this "
        "variation not stated', n='NA'. Range only (38-70 d) without SD/SE or sample size -> excluded as range_only (Rule 6)."
    )
    log.append("")


def write_outputs(rows: list[dict[str, object]], log_lines: list[str], done_count: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            out_row = {col: r.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(out_row)

    final_lines = list(log_lines)
    final_lines.append(
        f"Part 4: {len(rows)} rows from {done_count}/{len(ASSIGNED_DATASETS)} datasets"
    )
    LOG_OUT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, object]] = []
    log_lines: list[str] = [
        "# Extraction Log — Part 4 (D11_trematode_taskinen2022, D12_trematode_gleichsner2016, D13_nematode_marshallagia2023, D14_nematode_heligmosomoides, D15_helminth_clc_benesh2017)",
        "",
    ]
    log_lines.extend(verify_helpers_known_results())
    done_count = 0

    extract_d11(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D11: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d12(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D12: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d13(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D13: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d14(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D14: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d15(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D15: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")


if __name__ == "__main__":
    main()
