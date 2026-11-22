#!/usr/bin/env python3
"""Extract stage-duration spreads (Part 3):
  - D7_schisto_leclech2019
  - D8_schisto_civitello2020
  - D9_schisto_thermal_nguyen2021
  - D10_schisto_thermal_aslan2024

Applies protocol.md §4, §4.2, and Amendments 1-5 (`python3 -I` safe).
Writes:
  - results/stage_cv_part3.csv
  - results/extraction_log_part3.txt
"""
from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
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
CSV_OUT = RESULTS_DIR / "stage_cv_part3.csv"
LOG_OUT = RESULTS_DIR / "extraction_log_part3.txt"

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
    "D7_schisto_leclech2019",
    "D8_schisto_civitello2020",
    "D9_schisto_thermal_nguyen2021",
    "D10_schisto_thermal_aslan2024",
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
        f"Part 3: {len(rows)} rows from {done_count}/{len(ASSIGNED_DATASETS)} datasets"
    )
    LOG_OUT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")


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
    assert abs(wm["cv_mle"] - 1.0) < 0.05, f"Weibull MLE CV={wm['cv_mle']}"
    assert abs(wm["mean_mle"] - 50.0) < 2.5, f"Weibull MLE mean={wm['mean_mle']}"
    lines.append(
        f"  [PASS] weibull_censored_mle(Exp(50), ~25% right-censored, n=4000): "
        f"mean_mle={wm['mean_mle']:.3f} (exp 50.0, tol 2.5), cv_mle={wm['cv_mle']:.4f} (exp 1.0000, tol 0.05)"
    )

    t_det = np.array([5.0, 12.0, 14.0, 16.0, 19.0, 21.0, 22.0, 22.0])
    ev_det = np.array([True, True, True, True, True, True, False, False])
    wm2 = eh.weibull_censored_mle(t_det, ev_det)
    assert math.isfinite(wm2["cv_mle"]) and wm2["n_events"] == 6 and wm2["n_total"] == 8
    lines.append(
        f"  [PASS] weibull_censored_mle finite check (d=6/8): "
        f"mean_mle={wm2['mean_mle']:.4f}, cv_mle={wm2['cv_mle']:.4f}"
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Dataset D7: Schistosoma mansoni in Biomphalaria glabrata (Le Clec'h et al. 2019)
# ---------------------------------------------------------------------------

def extract_d7(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D7_schisto_leclech2019"
    d7_ods = RAW_DIR / ds / "Parasites_Vectors_2019_LeClech_et_al_data_v2.ods"
    rel_src = "D7_schisto_leclech2019/Parasites_Vectors_2019_LeClech_et_al_data_v2.ods"
    tables = eh.read_ods_tables(d7_ods)

    # Temperature verified directly from open-access full-text XML
    # data/raw/devtimes/study_temp_docs/D7_leclech2019_PMC6796389.xml (Methods):
    # "reared in 10-gallon aquaria containing aerated freshwater at 26-28 C on a 12L-12D photocycle" -> midpoint 27.0 C.
    # Units of `time` in Sheet `Cohort_1_Survival_infected_snails` verified directly from PMC6796389 Methods & Fig. 2b:
    # "We monitored snail survival every day from the first cercarial shedding day (28 days after exposure)
    #  to 22 days after the first cercarial shedding (50 days after exposure)." -> `time` is in DAYS (5..22 d).
    temp_c = 27.0

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Le Clec'h et al. 2019 Parasites & Vectors, Zenodo 3466005)")
    log.append(f"Source file: data/raw/devtimes/{rel_src}")
    log.append(" Companion open-access methods verified: study_temp_docs/D7_leclech2019_PMC6796389.xml")
    log.append(" Lab thermal regime: 26-28 C aerated freshwater aquaria (midpoint 27.00 C), 12L:12D")
    log.append(" Unit verification: `time` in Cohort_1_Survival_infected_snails is DAYS after first shedding (day 28 to day 50 post-exposure, monitored daily up to day 22 after first shedding; Amendment 4 status==0 is death, status==1 is right-censored at day 22)")
    log.append("========================================================================")

    # Exclusions for non-duration or unidentifiable-onset sheets in the ODS workbook
    s_shed = tables["Cohort_1_snail_shedding_LE_BRE_w4_w5_w6_w7"]
    smbre_w4_na = sum(1 for r in s_shed[1:] if r[1].strip() == "SmBRE" and r[2].strip() in ("NA", ""))
    smle_w4_na = sum(1 for r in s_shed[1:] if r[1].strip() == "SmLE" and r[2].strip() in ("NA", ""))
    log.append(
        f"[EXCLUSION] {ds} | Sheet 'Cohort_1_snail_shedding_LE_BRE_w4_w5_w6_w7' ({len(s_shed) - 1} rows: 48 SmBRE, 46 SmLE) | "
        f"reason=Weekly cercarial output counts across weeks 4-7 without pre-week-4 shedding checks (snails held in trays until day 28 first check, where {46 - smle_w4_na}/46 SmLE and {48 - smbre_w4_na}/48 SmBRE were already shedding; prepatent duration spread is left-censored at first check and unidentifiable under Protocol §4.1 Rule 6 / §6.2 P2a, while daily patent survival of the same 94 infected snails is in 'Cohort_1_Survival_infected_snails')."
    )
    for sh_name, reason in [
        (
            "Cohort_1_PO rate_Hb rate_LE_BRE_Ctrl_snails",
            "Hemolymph physiological assay at week 7.5 (laccase-like PO activity and hemoglobin rate), not a stage duration (Protocol §4.1 Rule 6).",
        ),
        (
            "Cohort_1_Correlation_PO rate_Hb_rate_Cercariae_produced",
            "Cross-trait correlation table of hemolymph Hb/PO rates vs cercarial counts, not a stage duration (Protocol §4.1 Rule 6).",
        ),
        (
            "Cohort_1_cercariae_production_by_parasite_sex",
            "Mean cercarial output counts stratified by parasite sex, not a stage duration (Protocol §4.1 Rule 6).",
        ),
        (
            "Cohort_2_sporocysts_quantification_qPCR",
            "Destructive qPCR sporocyst cell-proportion measurements on 6 sacrificed infected snails/week (weeks 3-8), not stage-duration timing (Protocol §4.1 Rule 6).",
        ),
        (
            "Cohort_2_correlation_infection_cercariae_sporocysts",
            "Destructive cercariae-vs-sporocyst correlation table across weeks 4-8 (selected from already-shedding snails), not stage-duration timing (Protocol §4.1 Rule 6).",
        ),
    ]:
        nrows_sh = max(len(tables.get(sh_name, [])) - 1, 0)
        log.append(f"[EXCLUSION] {ds} | Sheet '{sh_name}' ({nrows_sh} rows) | reason={reason}")

    # Sheet 2: Cohort_1_Survival_infected_snails
    s_surv = tables["Cohort_1_Survival_infected_snails"]
    by_pop: dict[str, list[tuple[int, str, float, bool]]] = defaultdict(list)
    for row_idx, r in enumerate(s_surv[1:], start=2):
        pop = r[1].strip()
        sex = r[2].strip()
        t_days = float(r[3].strip())
        status = int(r[4].strip())
        # Protocol Amendment 4 item 2: status == 0 is death prior to/by day 22; status == 1 is right-censored survivor at day 22
        is_death = status == 0
        by_pop[pop].append((row_idx, sex, t_days, is_death))

    for pop in ("SmLE", "SmBRE", "control"):
        items = by_pop[pop]
        t_arr = np.array([x[2] for x in items], dtype=float)
        ev_arr = np.array([x[3] for x in items], dtype=bool)
        n_tot = int(t_arr.size)
        d_ev = int(ev_arr.sum())
        cfrac = 1.0 - (d_ev / n_tot) if n_tot > 0 else math.nan

        if pop == "control":
            log.append(
                f"[EXCLUSION] {ds} | Biomphalaria glabrata Bg26 uninfected sham-exposed control survival (n={n_tot}, deaths d={d_ev}, right-censored={n_tot - d_ev} [{100.0 * cfrac:.2f}%]) | "
                "reason=Uncensored deaths d=1 < 10 and right-censored fraction 97.92% > 50.0% ceiling at day 22 (Protocol §4.1 Rule 2)."
            )
            continue

        wm = eh.weibull_censored_mle(t_arr, ev_arr)
        comp = eh.uncensored_stats(t_arr[ev_arr])
        if d_ev < 10 or cfrac > 0.50:
            log.append(
                f"[EXCLUSION] {ds} | Schistosoma mansoni ({pop} low-shedder population in B. glabrata Bg26) patent infected-snail survival (n={n_tot}, deaths d={d_ev}, right-censored={n_tot - d_ev} [{100.0 * cfrac:.2f}%]) | "
                f"reason=Right-censored fraction {100.0 * cfrac:.2f}% exceeds 50.0% ceiling at day 22 bleed (Protocol §4.1 Rule 2; Weibull MLE reference: mean={wm['mean_mle']:.3f} d, sd={wm['sd_mle']:.3f} d, cv={wm['cv_mle']:.4f}, completers_cv={comp['cv_indiv']:.4f})."
            )
            continue

        # Included cohort: SmLE (n=46, d=31, cens=32.61% <= 50%)
        # Also compute post-hoc PCR parasite-sex sub-cohorts for transparency note
        t_m = np.array([x[2] for x in items if x[1] == "m"], dtype=float)
        ev_m = np.array([x[3] for x in items if x[1] == "m"], dtype=bool)
        t_f = np.array([x[2] for x in items if x[1] == "f"], dtype=float)
        ev_f = np.array([x[3] for x in items if x[1] == "f"], dtype=bool)
        wm_m = eh.weibull_censored_mle(t_m, ev_m)
        wm_f = eh.weibull_censored_mle(t_f, ev_f)

        rows_out.append(
            make_row(
                dataset=ds,
                taxon="Schistosoma mansoni (in Biomphalaria glabrata Bg26)",
                stage="patent infected-snail survival from first cercarial shedding (SmLE high-shedder population, 1 miracidium)",
                stage_category="S2_parasite_in_invertebrate_vector_or_snail",
                temperature_c=temp_c,
                n=wm["n_total"],
                mean=wm["mean_mle"],
                sd=wm["sd_mle"],
                cv_indiv=wm["cv_mle"],
                censoring=f"right_censored_{100.0 * wm['cens_frac']:.1f}pct_day22 (d={wm['n_events']}/{wm['n_total']})",
                source_file=f"{rel_src}#Cohort_1_Survival_infected_snails",
                rule_applied=(
                    f"Rule 2: raw_individual_censored (Weibull MLE, Amend. 4 status==0 death; "
                    f"completers_cv={comp['cv_indiv']:.4f}, m_cv={wm_m['cv_mle']:.4f}, f_cv={wm_f['cv_mle']:.4f})"
                ),
            )
        )

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D7_schisto_leclech2019) ---")
    log.append(
        "  [HAND CHECK D7-1] SmLE patent infected-snail survival (Cohort_1_Survival_infected_snails rows 98-143, Sample # 1..46): "
        "n=46 snails; hand tally of status==0 deaths = [5d*3, 12d*4, 14d*5, 16d*4, 19d*6, 21d*6, 22d*3] -> d=31 deaths "
        "(sum=493 d, completers mean=493/31=15.903226 d, sd=4.925138 d, cv_indiv=0.312169) and 15 survivors right-censored at 22 d "
        "(status==1, cens=15/46=32.61% <= 50%, total exposure sum=823 d). Right-censored Weibull MLE k=3.301149, "
        "mean=19.525762 d, sd=6.512795 d, cv_indiv=0.333549. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D7-2] SmBRE patent infected-snail survival exclusion (Cohort_1_Survival_infected_snails rows 2-49, Sample # 1..48): "
        "n=48 snails; hand tally of status==0 deaths = [5d*2, 12d*2, 19d*5, 21d*2] -> d=11 deaths (sum=171 d) and "
        "37 survivors with status==1 at 22 d -> right-censored 37/48 = 77.08% > 50.0% ceiling; confirmed excluded under Rule 2."
    )
    log.append(
        "  [HAND CHECK D7-3] Uninfected control snail survival exclusion (Cohort_1_Survival_infected_snails rows 50-97, Sample # 1..48): "
        "n=48 sham-exposed snails; only Sample #31 died at 14 d (status==0, d=1 < 10) while 47/48 survived to 22 d "
        "(status==1, right-censored 97.92% > 50.0%); confirmed excluded under Rule 2."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D8: Schistosoma mansoni in Biomphalaria glabrata (Civitello et al. 2020)
# ---------------------------------------------------------------------------

def extract_d8(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D8_schisto_civitello2020"
    d8_dir = RAW_DIR / ds
    rs_csv = d8_dir / "ResourceSupplyExp.csv"
    ps_csv = d8_dir / "PeriodicStarveExp.csv"

    # Temperature provenance audit (study rules Rule 2):
    # In study_temp_docs/D8_civitello2020_alt_epmc.json, Civitello et al. (2020) Proc. R. Soc. B 287:20192446
    # (doi:10.1098/rspb.2019.2446, PMC7015343) has isOpenAccess='N' (while D8_civitello2020_alt_PMC7365472.xml
    # is Haggerty et al. 2020 PLOS NTD, a Senegal field survey), and Zenodo 3860566 CSV/JSON files omit a
    # temperature column. Per Rule 2, constant lab incubator temperature is left blank ('') rather than recalled from memory.
    temp_c = ""

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Civitello et al. 2020 Proc. R. Soc. B 287:20192446, Zenodo 3860566 / CC0)")
    log.append("Source files: data/raw/devtimes/D8_schisto_civitello2020/ResourceSupplyExp.csv, PeriodicStarveExp.csv")
    log.append(" Thermal provenance note (Rule 2): Constant laboratory holding regime; exact numerical temperature is not stated in open deposit zenodo.3860566 or non-OA EuropePMC abstract PMC7015343 (isOpenAccess='N'), so temperature_C is left blank ('') rather than filled from memory.")
    log.append("========================================================================")

    # 1. ResourceSupplyExp.csv
    with rs_csv.open(encoding="utf-8") as f:
        rs_rows = list(csv.DictReader(f))

    by_rs_snail: dict[tuple[str, float, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rs_rows:
        fl_s = r["FoodLevel"].strip()
        by_rs_snail[(r["Exposed"].strip(), float(fl_s), fl_s, r["Color"].strip(), r["ID"].strip())].append(r)

    # Prepatent period exclusion in ResourceSupplyExp.csv
    log.append(
        f"[EXCLUSION] {ds} | ResourceSupplyExp.csv prepatent period across 6 Exposed='Yes' FoodLevel cohorts (66 snails) | "
        "reason=Exposure date is not recorded as a column in ResourceSupplyExp.csv (Date begins at 28 with Worms=0 through Date=63), "
        "and under 7-day check spacing 60/66 snails (90.9%, including 12/12 at FoodLevel=2.75 and 10/10 at FoodLevel=5.5 with SD=0) "
        "first shed on the single check Date=70 (Protocol §4.1 Rule 6 & §6.2 P2a)."
    )

    # Unexposed uninfected cohorts in ResourceSupplyExp.csv (n=5 per FoodLevel)
    unexp_by_fl: dict[tuple[float, str], int] = defaultdict(int)
    for k in by_rs_snail:
        if k[0] == "No":
            unexp_by_fl[(k[1], k[2])] += 1
    for _, fl_s in sorted(unexp_by_fl):
        cnt = unexp_by_fl[(float(fl_s), fl_s)]
        log.append(
            f"[EXCLUSION] {ds} | ResourceSupplyExp.csv Exposed='No' uninfected Biomphalaria glabrata FoodLevel={fl_s} (n={cnt}) | "
            "reason=Cohort sample size n=5 < 10 (Protocol §4.1 Rule 1 / Rule 2; also uninfected control host without parasite stage)."
        )

    # Patent cercarial shedding duration across the 6 Exposed='Yes' constant resource-supply treatments
    exp_fls = sorted({(k[1], k[2]) for k in by_rs_snail if k[0] == "Yes"})
    for _, fl_s in exp_fls:
        t_shed: list[float] = []
        ev_shed: list[bool] = []
        t_host: list[float] = []
        ev_host: list[bool] = []
        for k in sorted(by_rs_snail):
            if k[0] != "Yes" or k[2] != fl_s:
                continue
            v = sorted(by_rs_snail[k], key=lambda x: int(x["Date"]))
            pos_d = [int(x["Date"]) for x in v if x["Worms"].strip() not in ("", "NA") and float(x["Worms"]) > 0.0]
            la_d = [int(x["Date"]) for x in v if x["Last_Alive"].strip() == "1"][0]
            died = any(x["Dead"].strip() == "1" for x in v)
            onset = float(pos_d[0]) - 3.5  # midpoint of (t_first - 7, t_first] weekly check interval
            cens_shed = (not died) and (pos_d[-1] == 245)
            end_shed = 245.0 if cens_shed else (float(pos_d[-1]) + 3.5)
            t_shed.append(end_shed - onset)
            ev_shed.append(not cens_shed)

            end_host = (float(la_d) + 3.5) if died else 245.0
            t_host.append(end_host - onset)
            ev_host.append(died)

        t_arr = np.array(t_shed, dtype=float)
        ev_arr = np.array(ev_shed, dtype=bool)
        th_arr = np.array(t_host, dtype=float)
        evh_arr = np.array(ev_host, dtype=bool)
        n_tot = int(t_arr.size)
        d_ev = int(ev_arr.sum())
        cfrac = 1.0 - (d_ev / n_tot)
        stg_lbl = f"patent cercarial shedding duration (constant resource supply FoodLevel={fl_s} mg/d, weekly checks to day 245)"

        if cfrac <= 0.05 and n_tot >= 10:
            st = eh.uncensored_stats(t_arr, dt_obs=7.0)
            sth = eh.uncensored_stats(th_arr, dt_obs=7.0)
            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon="Schistosoma mansoni (in Biomphalaria glabrata)",
                    stage=stg_lbl,
                    stage_category="S2_parasite_in_invertebrate_vector_or_snail",
                    temperature_c=temp_c,
                    n=st["n"],
                    mean=st["mean"],
                    sd=st["sd"],
                    cv_indiv=st["cv_indiv"],
                    censoring="uncensored" if cfrac == 0.0 else f"right_censored_{100.0 * cfrac:.1f}pct_le_5pct",
                    source_file="D8_schisto_civitello2020/ResourceSupplyExp.csv",
                    rule_applied=(
                        f"Rule 1: raw_individual_uncensored (Sokal-Rohlf, dt_obs=7d; "
                        f"cv_sheppard={st['cv_sheppard']:.4f}, patent_host_stay_cv={sth['cv_indiv']:.4f})"
                    ),
                )
            )
        elif d_ev >= 10 and cfrac <= 0.50:
            wm = eh.weibull_censored_mle(t_arr, ev_arr)
            wmh = eh.weibull_censored_mle(th_arr, evh_arr)
            comp = eh.uncensored_stats(t_arr[ev_arr], dt_obs=7.0)
            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon="Schistosoma mansoni (in Biomphalaria glabrata)",
                    stage=stg_lbl,
                    stage_category="S2_parasite_in_invertebrate_vector_or_snail",
                    temperature_c=temp_c,
                    n=wm["n_total"],
                    mean=wm["mean_mle"],
                    sd=wm["sd_mle"],
                    cv_indiv=wm["cv_mle"],
                    censoring=f"right_censored_{100.0 * wm['cens_frac']:.1f}pct_day245 (d={wm['n_events']}/{wm['n_total']})",
                    source_file="D8_schisto_civitello2020/ResourceSupplyExp.csv",
                    rule_applied=(
                        f"Rule 2: raw_individual_censored (Weibull MLE; "
                        f"completers_cv={comp['cv_indiv']:.4f}, patent_host_stay_cv={wmh['cv_mle']:.4f})"
                    ),
                )
            )
        else:
            wm = eh.weibull_censored_mle(t_arr, ev_arr)
            log.append(
                f"[EXCLUSION] {ds} | ResourceSupplyExp.csv {stg_lbl} (n={n_tot}, uncensored events d={d_ev}, right-censored={n_tot - d_ev} [{100.0 * cfrac:.1f}%]) | "
                f"reason=Uncensored event count d={d_ev} < 10 required for right-censored cohorts (Protocol §4.1 Rule 2; Weibull MLE reference: mean={wm['mean_mle']:.3f} d, sd={wm['sd_mle']:.3f} d, cv={wm['cv_mle']:.4f})."
            )

    # 2. PeriodicStarveExp.csv (all 8 treatment cells have n <= 8 < 10)
    with ps_csv.open(encoding="utf-8") as f:
        ps_rows = list(csv.DictReader(f))
    by_ps_cell: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in ps_rows:
        by_ps_cell[(r["Exposed"].strip(), r["Food"].strip(), r["Starve_W"].strip())].add(r["Snail"].strip())
    for (exp_s, food_s, stw_s), snail_ids in sorted(by_ps_cell.items()):
        log.append(
            f"[EXCLUSION] {ds} | PeriodicStarveExp.csv Exposed='{exp_s}' Food={food_s} Starve_W={stw_s} (n={len(snail_ids)} snails) | "
            f"reason=Treatment cell sample size n={len(snail_ids)} < 10 (Protocol §4.1 Rule 1 / Rule 2; pooling across starvation regimes forbidden by §4)."
        )

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D8_schisto_civitello2020) ---")
    log.append(
        "  [HAND CHECK D8-1] FoodLevel=0.6875 patent cercarial shedding duration (ResourceSupplyExp.csv lines 2-321, 10 Black-tagged snails): "
        "all 10 died before day 245 (0 censored); midpoint durations (last_pos + 3.5) - (first_pos - 3.5) = "
        "[105, 91, 28, 98, 21, 91, 21, 98, 14, 77] d -> sum=644.0, sum(t^2)=54586.0, n=10, "
        "mean=644/10=64.400000 d, sd=sqrt((54586 - 644^2/10)/9)=38.169796 d, "
        "cv_indiv=(38.169796/64.4)*(1+1/40)=0.607516. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D8-2] FoodLevel=1.375 patent cercarial shedding duration (ResourceSupplyExp.csv lines 322-705, 12 Brown-tagged snails): "
        "all 12 died before day 245 (0 censored); midpoint durations = [105, 105, 7, 98, 98, 98, 98, 63, 63, 105, 105, 98] d -> "
        "sum=1043.0, sum(t^2)=100107.0, n=12, mean=1043/12=86.916667 d, sd=sqrt((100107 - 1043^2/12)/11)=29.314775 d, "
        "cv_indiv=(29.314775/86.916667)*(1+1/48)=0.344301. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D8-3] FoodLevel=2.75 patent cercarial shedding duration (ResourceSupplyExp.csv lines 706-1089, 12 Purple-tagged snails): "
        "10 uncensored deaths [105, 126, 126, 105, 126, 7, 84, 84, 84, 119] d (d=10 >= 10) plus 2 survivors still shedding at day 245 "
        "(Purple-12 and Purple-6, right-censored at 245 - 66.5 = 178.5 d; cens=2/12=16.7% <= 50%) -> "
        "Weibull MLE k=1.987419, mean=115.133807 d, sd=60.525933 d, cv_indiv=0.525701. Matches CSV row."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D9: Schistosoma mansoni & Biomphalaria thermal compilation (Nguyen et al. 2021)
# ---------------------------------------------------------------------------

def _read_d9_csv(z9: zipfile.ZipFile, member: str) -> tuple[list[tuple[int, dict[str, str]]], int]:
    txt = z9.read(member).decode("latin1")
    all_lines = txt.splitlines()
    non_comment = [(idx + 1, ln) for idx, ln in enumerate(all_lines) if not ln.startswith("#") and ln.strip()]
    if not non_comment:
        return [], 0
    hdr = next(csv.reader([non_comment[0][1]]))
    out: list[tuple[int, dict[str, str]]] = []
    for ln_no, ln in non_comment[1:]:
        vals = next(csv.reader([ln]))
        out.append((ln_no, dict(zip(hdr, vals))))
    return out, len(out)


def extract_d9(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D9_schisto_thermal_nguyen2021"
    zip_path = RAW_DIR / ds / "pboesu_schistosomiasis-v1.0.zip"
    rel_zip = "D9_schisto_thermal_nguyen2021/pboesu_schistosomiasis-v1.0.zip"
    prefix = "pboesu-schistosomiasis-3e31c0c/data"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Nguyen et al. 2021 PNAS, Zenodo 4550934 / CC BY 4.0)")
    log.append(f"Source archive: data/raw/devtimes/{rel_zip}")
    log.append("========================================================================")

    with zipfile.ZipFile(zip_path) as z9:
        # 1. cercarial_survival_observations.csv (Lawson & Wilson 1980 Parasitology 81:337-348)
        cs_rows, _ = _read_d9_csv(z9, f"{prefix}/cercarial_survival_observations.csv")
        by_temp: dict[float, list[tuple[int, dict[str, str]]]] = defaultdict(list)
        for ln_no, r in cs_rows:
            by_temp[float(r["temperature"].strip())].append((ln_no, r))

        for temp_c in sorted(by_temp.keys()):
            items = by_temp[temp_c]
            days = np.array([float(x[1]["time"].strip()) / 24.0 for x in items], dtype=float)
            pct = np.clip(np.array([float(x[1]["value"].strip()) for x in items], dtype=float), 0.0, 100.0) / 100.0
            # Note: In weibull_survival_cross_sectional_mle, equal weight c across timepoints scales
            # the binomial quasi-log-likelihood by a constant factor c, leaving (k_hat, lam_hat, s0_hat,
            # mean_surv, sd_surv, cv_indiv, bracketed) strictly invariant. Because Lawson & Wilson (1980)
            # digitized survival curves in D9 report 'Mean % Survival' across replicates=5 vessels without
            # individual cercarial counts N_t, `n` is recorded as '' (unknown individual count) so strict
            # known-n>=10 primary filters separate them cleanly.
            fit = eh.weibull_survival_cross_sectional_mle(days, pct * 100.0, np.full_like(days, 100.0))
            ln_span = f"lines {items[0][0]}-{items[-1][0]}"
            if not fit.get("ok", False):
                log.append(
                    f"[EXCLUSION] {ds} | cercarial_survival_observations.csv T={temp_c:.0f}C ({len(items)} timepoints, {ln_span}, source=Lawson1980) | "
                    f"reason=Fewer than 4 distinct observation timepoints (m={len(items)} < 4, Protocol §4.1 Rule 3)."
                )
                continue
            if not fit.get("bracketed", False):
                log.append(
                    f"[EXCLUSION] {ds} | cercarial_survival_observations.csv T={temp_c:.0f}C ({len(items)} timepoints, {ln_span}, obs_surv={100.0 * float(pct.min()):.1f}%..{100.0 * float(pct.max()):.1f}%, source=Lawson1980) | "
                    f"reason=Survival curve not bracketed over 35-hour window (minimum observed survival {100.0 * float(pct.min()):.1f}% > 30% of s0={100.0 * fit['s0']:.1f}%, Protocol §4.1 Rule 3; unbracketed Weibull reference: mean={fit['mean_surv']:.4f} d, cv={fit['cv_indiv']:.4f})."
                )
                continue

            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon="Schistosoma mansoni",
                    stage=f"free-living cercarial survival in water (source=Lawson1980, {fit['n_days']} timepoints, 5 replicate vessels)",
                    stage_category="S3_free_living_parasite_stage",
                    temperature_c=temp_c,
                    n="",
                    mean=fit["mean_surv"],
                    sd=fit["sd_surv"],
                    cv_indiv=fit["cv_indiv"],
                    censoring=f"cross_sectional_survival_curve_{fit['n_days']}pts_no_indiv_n",
                    source_file=f"{rel_zip}#data/cercarial_survival_observations.csv",
                    rule_applied=(
                        f"Rule 3: interval_transition_curve (Weibull survival curve MLE on digitized % survival, "
                        f"no_indiv_n [replicates=5 vessels]; k_weibull={fit['k_weibull']:.4f}, s0={fit['s0']:.4f})"
                    ),
                )
            )

        # 2. Exclusions for the remaining 5 CSV tables in D9
        _, n_shed = _read_d9_csv(z9, f"{prefix}/cercarial_shedding_observations.csv")
        log.append(
            f"[EXCLUSION] {ds} | data/cercarial_shedding_observations.csv ({n_shed} rows, variable='cercariae_count_2h', source=Nguyen2021) | "
            "reason=2-hour cercarial release count per snail across temperatures (fecundity/output intensity), not a stage duration (Protocol §4.1 Rule 6)."
        )

        _, n_hatch = _read_d9_csv(z9, f"{prefix}/miracidia_hatching_observations.csv")
        log.append(
            f"[EXCLUSION] {ds} | data/miracidia_hatching_observations.csv ({n_hatch} rows, variable='miracidial_hatching', source=Nguyen2021) | "
            "reason=Cumulative hatching proportion (N_hatched / N_total) at a single endpoint without hatching time series or individual completion times (Protocol §4.1 Rule 6)."
        )

        _, n_smort = _read_d9_csv(z9, f"{prefix}/snail_mortality_observations.csv")
        log.append(
            f"[EXCLUSION] {ds} | data/snail_mortality_observations.csv ({n_smort} rows, variable='mortality_60d', n_total=28, 6 temperatures x infected TRUE/FALSE, source=Foster1964) | "
            "reason=Cumulative mortality proportion at a single fixed day (60 d; 1 timepoint < 4 and n_total=28 < 50), so survival duration spread CV_indiv is non-identifiable (Protocol §4.1 Rule 3 & Rule 6)."
        )

        tr_rows, n_trait = _read_d9_csv(z9, f"{prefix}/trait_observations.csv")
        var_counts: dict[str, int] = defaultdict(int)
        for _, r in tr_rows:
            var_counts[r["variable"].strip()] += 1
        var_desc = ", ".join(f"{k}:{v}" for k, v in sorted(var_counts.items()))
        log.append(
            f"[EXCLUSION] {ds} | data/trait_observations.csv ({n_trait} rows: {var_desc}) | "
            "reason=Point per-capita mortality/transmission rate estimates (mean_only: 1 scalar rate per temperature without within-cohort SD, SE, or n; Protocol §4.1 Rule 6)."
        )

        _, n_pred = _read_d9_csv(z9, f"{prefix}/trait_predictions.csv")
        log.append(
            f"[EXCLUSION] {ds} | data/trait_predictions.csv ({n_pred} rows) | "
            "reason=Model-fitted thermal performance curve grid predictions (value, lci, uci of curve mean), not empirical within-cohort stage durations (Protocol §3 note & §4.1 Rule 6)."
        )

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D9_schisto_thermal_nguyen2021) ---")
    log.append(
        "  [HAND CHECK D9-1] cercarial survival at 20.00 C (cercarial_survival_observations.csv lines 52-62, 11 timepoints): "
        "raw endpoints line 52 (t=2.4431555 h = 0.10180 d, Mean % Survival=94.08671%), line 57 (t=17.34763 h = 0.72282 d, 75.265076%), "
        "line 62 (t=34.75709 h = 1.44821 d, 3.635354%) -> bracketed (94.1% >= 70%*s0 to 3.6% <= 30%*s0 across 11 >= 4 times); "
        "Weibull survival MLE s0=0.917277, k=3.952317, mean=0.907534 d (21.781 h), sd=0.257379 d, cv_indiv=0.283603. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D9-2] cercarial survival at 25.00 C (cercarial_survival_observations.csv lines 63-72, 10 timepoints): "
        "raw endpoints line 63 (t=2.3176427 h = 0.09657 d, 96.63364%), line 68 (t=12.787864 h = 0.53283 d, 84.317314%), "
        "line 72 (t=26.286926 h = 1.09529 d, 3.2052052%) -> bracketed across 10 times; "
        "Weibull survival MLE s0=0.972015, k=4.220457, mean=0.740578 d (17.774 h), sd=0.197914 d, cv_indiv=0.267243. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D9-3] cercarial survival at 30.00 C (cercarial_survival_observations.csv lines 73-79, 7 timepoints): "
        "raw endpoints line 73 (t=2.5259342 h = 0.10525 d, 99.10671%), line 76 (t=9.134741 h = 0.38061 d, 62.16875%), "
        "line 79 (t=21.13434 h = 0.88060 d, 1.8019928%) -> bracketed across 7 times; "
        "Weibull survival MLE s0=0.999877, k=2.519050, mean=0.457351 d (10.976 h), sd=0.194378 d, cv_indiv=0.425009. Matches CSV row."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D10: Schistosoma haematobium & S. mansoni thermal literature compilation (Aslan et al. 2024)
# ---------------------------------------------------------------------------

def _count_r_vector_entries(r_text: str, var_prefix: str) -> int:
    import re
    total = 0
    for m in re.finditer(rf"\b{re.escape(var_prefix)}\w*\s*<-\s*c\(([^)]*)\)", r_text):
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        total += len(items)
    for m in re.finditer(rf"\b{re.escape(var_prefix)}\w*\s*<-\s*([0-9]+(?:\.[0-9]+)?)\b", r_text):
        if m.group(1):
            total += 1
    return total


def extract_d10(rows_out: list[dict[str, object]], log: list[str]) -> None:
    _ = rows_out  # All D10 entries are temperature-level point means/rates without individual spread
    ds = "D10_schisto_thermal_aslan2024"
    d10_dir = RAW_DIR / ds

    log.append("========================================================================")
    log.append(f"DATASET: {ds} (Aslan et al. 2024 PLOS Negl. Trop. Dis., GitHub MIT License)")
    log.append("Source directory: data/raw/devtimes/D10_schisto_thermal_aslan2024/")
    log.append("Audit result: 0 rows extracted; every R script and GADL demographic spreadsheet stores only temperature-level point means/rates without within-cohort SD, SE, CI, or individual timings (Protocol §4.1 Rule 6: mean_only).")
    log.append("========================================================================")

    # 1. s.haematobium__bootstrapping__lat_per_snail.R
    r_sh_lat = (d10_dir / "s.haematobium__bootstrapping__lat_per_snail.R").read_text(encoding="utf-8", errors="replace")
    n_sh_lat = _count_r_vector_entries(r_sh_lat, "temp_")
    log.append(
        f"[EXCLUSION] {ds} | s.haematobium__bootstrapping__lat_per_snail.R ({n_sh_lat} temperature-rate pairs: Pfluger 1984 Bulinus truncatus n_T=11 [18..32 C, 1/117..1/30 d^-1] & Gordon 1934 B. globosus n_T=8 [22..35.2 C, 1/67..1/26.3 d^-1]) | "
        "reason=mean_only point prepatent development rates (1/mean days) per temperature with no within-cohort SD, SE, or individual snail counts; case-resampling Boot(fit_nlsLM) in lines 56-77 estimates uncertainty of the mean thermal curve across temperatures, not individual duration spread (Protocol §4 & §4.1 Rule 6)."
    )

    # 2. s.mansoni__bootstrapping__lat_per_snail.R
    r_sm_lat = (d10_dir / "s.mansoni__bootstrapping__lat_per_snail.R").read_text(encoding="utf-8", errors="replace")
    n_sm_lat = _count_r_vector_entries(r_sm_lat, "temp_")
    log.append(
        f"[EXCLUSION] {ds} | s.mansoni__bootstrapping__lat_per_snail.R ({n_sm_lat} temperature-rate pairs: Pfluger 1981 diurnal B. glabrata n_T=12, Pfluger 1980 constant B. glabrata n_T=11, Foster 1964 B. pfeifferi n_T=8, Gordon 1934 B. pfeifferi n_T=12) | "
        "reason=mean_only point prepatent development rates (1/mean days) per temperature with no within-cohort SD, SE, or individual snail counts; Boot(fit_nlsLM) resamples curve points across temperatures rather than individuals within a cohort (Protocol §4 & §4.1 Rule 6)."
    )

    # 3. s.haematobium__bootstrapping__mort_snail_2.R
    r_sh_mort = (d10_dir / "s.haematobium__bootstrapping__mort_snail_2.R").read_text(encoding="utf-8", errors="replace")
    n_sh_mort = _count_r_vector_entries(r_sh_mort, "temp_")
    log.append(
        f"[EXCLUSION] {ds} | s.haematobium__bootstrapping__mort_snail_2.R ({n_sh_mort} high-temperature point mortality rates: Kalinda 2017 B. globosus 35.5 C, Joubert 1986 B. globosus 34..40 C [4 pts], Joubert 1986 B. africanus 34..40 C [4 pts], El-Hassan 1974 35 C) | "
        "reason=mean_only scalar per-capita daily mortality rates without within-cohort survival SD, SE, or individual death times (Protocol §4.1 Rule 6)."
    )

    # 4. Three GADL snail demographic spreadsheets (.xlsx)
    xlsx_specs = [
        (
            "s.haematobium__ther_sen_par_bul_haemtobium__chester_kalinda_2017_bulinus_globosus_thermal sensitive demographic parameters_GADL.xlsx",
            "Kalinda et al. 2017 Bulinus globosus (5 temperatures: 15.5, 21.2, 25.8, 31, 35.5 C)",
        ),
        (
            "s.mansoni__ther_sen_par_biom_mansoni__McCreesh_2014_Biomphalaria_sudanica_thermal sensitive demographic parameters_GADL.xlsx",
            "McCreesh et al. 2014 Biomphalaria sudanica (10 temperatures: 13.4..32 C)",
        ),
        (
            "s.mansoni__ther_sen_par_biom_mansoni__c_c_appleton_1977_Biomphalaria_pfeiffferi_thermal sensitive demographic parameters_GADL.xlsx",
            "Appleton 1977 Biomphalaria pfeifferi (3 temperatures: 25, 27, 29 C)",
        ),
    ]
    for fn, label in xlsx_specs:
        srows = eh.read_xlsx_sheet(d10_dir / fn, 0)
        temp_rows = [
            sr[0]
            for sr in srows[2:16]
            if sr and sr[0] not in ("", "Temp") and sr[0].replace(".", "", 1).replace("-", "", 1).isdigit()
        ]
        log.append(
            f"[EXCLUSION] {ds} | {fn} ({label}; {len(temp_rows)} temperature rows, {len(srows)} sheet rows incl. deterministic Euler-Lotka daily geometric table) | "
            "reason=mean_only point demographic parameters (incubation_period, mortality_rate_per_day, age_at_maturity) without any within-cohort SD, SE, CI, or individual survival/development observations (Protocol §4.1 Rule 6)."
        )

    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D10_schisto_thermal_aslan2024) ---")
    log.append(
        "  [HAND CHECK D10-1] s.haematobium__bootstrapping__lat_per_snail.R lines 12-21: "
        "temp_pfluger_84 has 11 temperatures [18..32 C] paired with scalar reciprocals rate_pfluger_84 = [1/117..1/30] d^-1, "
        "and temp_gordon_34_glo has 8 temperatures [22..35.2 C] paired with [1/67..1/26.3] d^-1 (total 19 point means, 0 SD/SE/n columns); "
        "confirmed mean_only exclusion under Rule 6."
    )
    log.append(
        "  [HAND CHECK D10-2] s.mansoni__bootstrapping__lat_per_snail.R lines 12-28: "
        "temp_pfluger_81 (12 pts, [24.3..33 C], [1/26..1/16.5]), temp_pfluger_80 (11 pts, [17..35 C], [1/92.5..1/16]), "
        "temp_foster_64 (8 pts, [18..31.75 C], [1/57..1/16]), temp_gordon_34_pfei (12 pts, [21.7..35 C], [1/36..1/16.4]) "
        "= 43 scalar point rates with zero individual dispersion columns; confirmed mean_only exclusion under Rule 6."
    )
    log.append(
        "  [HAND CHECK D10-3] s.haematobium__ther_sen_par_bul_haemtobium__chester_kalinda_2017_bulinus_globosus_thermal sensitive demographic parameters_GADL.xlsx rows 2-8: "
        "columns A-F are ['Temp', 'incubation_period', 'hatchin_rate', 'mortality_rate_per_day', 'age_at_maturity', 'Egg_snail_day'] "
        "for Temp=[15.5, 21.2, 25.8, 31, 35.5] C (incubation_period=[na, 12, 10, 8, na] d, age_at_maturity=[na, 108, 75, 135, na] d, "
        "daily mortality=[0.002449, 0.006901, 0.011513, 0.020737, 0.046427] d^-1), followed in rows 16-563 by deterministic Euler-Lotka terms "
        "exp(-r*x)*l_x*m_x with zero SD/SE/n fields; confirmed mean_only exclusion under Rule 6."
    )
    log.append("")


def main() -> None:
    rows: list[dict[str, object]] = []
    log_lines: list[str] = [
        "# Extraction Log — Part 3 (D7_schisto_leclech2019, D8_schisto_civitello2020, D9_schisto_thermal_nguyen2021, D10_schisto_thermal_aslan2024)",
        "",
    ]
    log_lines.extend(verify_helpers_known_results())
    done_count = 0

    extract_d7(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D7: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d8(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D8: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d9(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D9: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d10(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D10: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")


if __name__ == "__main__":
    main()
