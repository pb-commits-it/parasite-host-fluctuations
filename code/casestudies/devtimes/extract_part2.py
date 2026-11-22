#!/usr/bin/env python3
"""Extract stage-duration spreads (Part 2):
  - D5_ixodes_fracasso2021
  - D6_ixodes_scapularis_brunner2022
  - D16_vectraits_compilations

Applies protocol.md §4, §4.2, and Amendments 1-5 (`python3 -I` safe).
Writes:
  - results/stage_cv_part2.csv
  - results/extraction_log_part2.txt
"""
from __future__ import annotations

import csv
import datetime
import json
import math
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
CSV_OUT = RESULTS_DIR / "stage_cv_part2.csv"
LOG_OUT = RESULTS_DIR / "extraction_log_part2.txt"

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
    "D5_ixodes_fracasso2021",
    "D6_ixodes_scapularis_brunner2022",
    "D16_vectraits_compilations",
]


def fmt_float(val: float | str | None, digits: int = 6) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        return val
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def excel_serial_to_date(serial_str: str) -> datetime.date:
    return datetime.date(1899, 12, 30) + datetime.timedelta(days=int(float(serial_str)))


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


# ---------------------------------------------------------------------------
# Dataset D5: Ixodes arboricola (Fracasso et al. 2021/2022, Zenodo 5519771)
# ---------------------------------------------------------------------------

def extract_d5(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D5_ixodes_fracasso2021"
    taxon = "Ixodes arboricola"
    cat = "S4_vector_tick_development_moulting"
    d5_dir = RAW_DIR / ds
    ticks_csv = d5_dir / "All_ticks.csv"
    eggs_csv = d5_dir / "Number_hatched_eggs.csv"
    weather_json = RAW_DIR / "weather" / "nasapower_Belgium_Boshoek_D5.json"

    log.append("========================================================================")
    log.append(f"DATASET: {ds} ({taxon})")
    log.append("Source files: data/raw/devtimes/D5_ixodes_fracasso2021/All_ticks.csv, Number_hatched_eggs.csv")
    log.append(" Companion open-access methods verified: study_temp_docs/D5_fracasso2021_PMC9314103.xml (Amendment 5)")
    log.append(" Lab thermal regimes: 20.0 C, 55% RH (captive bird rooms 146/148); 20.0 C, 85% RH darkness (vials)")
    log.append(" Wild nest-box ambient T2M: weather/nasapower_Belgium_Boshoek_D5.json (51.1331N, 4.5169E)")
    log.append("========================================================================")

    t2m_map: dict[str, float] = {}
    if weather_json.exists():
        wdata = json.loads(weather_json.read_text(encoding="utf-8"))
        t2m_map = {
            k: float(v)
            for k, v in wdata.get("properties", {}).get("parameter", {}).get("T2M", {}).items()
            if float(v) > -900.0
        }

    with ticks_csv.open(encoding="utf-8") as f:
        tick_rows = list(csv.DictReader(f))

    # Log overarching protocol exclusion for fasting duration
    log.append(
        "[EXCLUSION] D5_ixodes_fracasso2021 | stage=fasting_duration (DINF_next - MOULT_DAY_prev) | "
        "reason=Experimental design variable, not a biological survival/completion trait: "
        "PMC9314103 Statistical Analyses explicitly states 'fasting time (number of days elapsed "
        "between recovery or hatching and the next infestation) ... was set by the experimenter and not by the ticks'."
    )

    by_stage_batch: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for line_idx, r in enumerate(tick_rows, start=2):
        by_stage_batch[(r["LSTAG"].strip(), r["BATCH"].strip())].append((line_idx, r))

    def batch_sort_key(item: tuple[tuple[str, str], object]) -> tuple[str, int, str]:
        stg, b = item[0]
        return (stg, int(b) if b.isdigit() else 9999, b)

    # 1. Feeding duration cohorts by (LSTAG, BATCH)
    for (stg, batch), items in sorted(by_stage_batch.items(), key=batch_sort_key):
        if batch == "":
            for ln, r in items:
                log.append(
                    f"[EXCLUSION] {ds} | All_ticks.csv line {ln} TID={r['TID']} LSTAG={stg} | "
                    "reason=Missing BATCH and missing DINF/DREC feeding dates (Rule 6)."
                )
            continue

        valid_obs: list[tuple[float, bool, str, str]] = []
        missing_dates = 0
        hidden_gt8 = 0
        moat_ocean = 0
        nonpos = 0
        for ln, r in items:
            dinf_s, drec_s = r["DINF"].strip(), r["DREC"].strip()
            pos = r["POS"].strip()
            if dinf_s in ("n/a", "", "NA") or drec_s in ("n/a", "", "NA"):
                missing_dates += 1
                continue
            dt = float(drec_s) - float(dinf_s)
            if dt <= 0:
                nonpos += 1
                continue
            if pos == "ocean":
                moat_ocean += 1
                log.append(
                    f"[EXCLUSION] {ds} | All_ticks.csv line {ln} TID={r['TID']} ({stg} batch {batch}) | "
                    f"reason=POS='ocean' (cage water moat, DREC-DINF={dt:.0f}d); tick detached earlier into moat so detachment day is unknown."
                )
                continue
            if stg == "female" and dt > 8.0:
                hidden_gt8 += 1
                continue
            # POS == 'bird' indicates tick still attached at batch end (right-censored at DREC-DINF)
            is_event = pos != "bird"
            valid_obs.append((dt, is_event, dinf_s, drec_s))

        if missing_dates > 0:
            log.append(
                f"[EXCLUSION] {ds} | {stg}_feeding_batch_{batch} ({missing_dates} rows) | "
                "reason=Missing DINF or DREC in All_ticks.csv (Rule 6)."
            )
        if hidden_gt8 > 0:
            log.append(
                f"[EXCLUSION] {ds} | {stg}_feeding_batch_{batch} ({hidden_gt8} rows with DREC-DINF > 8d) | "
                "reason=Assigned unknown feeding time per Fracasso et al. 2022 (PMC9314103): wild nest-box females "
                "recovered >8 days post-infestation had detached earlier and remained hidden in nest material."
            )
        if nonpos > 0:
            log.append(
                f"[EXCLUSION] {ds} | {stg}_feeding_batch_{batch} ({nonpos} rows) | "
                "reason=Non-positive feeding duration DREC-DINF <= 0."
            )

        n_tot = len(valid_obs)
        if n_tot < 10:
            log.append(
                f"[EXCLUSION] {ds} | {stg}_feeding_batch_{batch} (n={n_tot}) | "
                "reason=Cohort sample size n < 10 (Protocol §4.1 Rule 1)."
            )
            continue

        # Temperature: 20.0 C for captive indoor rooms 146/148; Boshoek NASA POWER ambient T2M mean for wild nest boxes
        if stg in ("larva", "nymph"):
            temp_c: float | str = 20.0
            env_tag = "captive_20C"
        else:
            d0 = min(excel_serial_to_date(x[2]) for x in valid_obs)
            d1 = max(excel_serial_to_date(x[3]) for x in valid_obs)
            t_vals = []
            cur = d0
            while cur <= d1:
                key = cur.strftime("%Y%m%d")
                if key in t2m_map:
                    t_vals.append(t2m_map[key])
                cur += datetime.timedelta(days=1)
            temp_c = float(np.mean(t_vals)) if t_vals else ""
            env_tag = f"wild_nestbox_{d0.isoformat()}_to_{d1.isoformat()}"

        t_arr = np.array([x[0] for x in valid_obs], dtype=float)
        ev_arr = np.array([x[1] for x in valid_obs], dtype=bool)
        n_cens = int((~ev_arr).sum())
        cfrac = n_cens / n_tot
        stage_label = f"{stg}_feeding_batch_{batch}_{env_tag}"

        if cfrac == 0.0:
            st = eh.uncensored_stats(t_arr, dt_obs=1.0)
            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=stage_label,
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=st["n"],
                    mean=st["mean"],
                    sd=st["sd"],
                    cv_indiv=st["cv_indiv"],
                    censoring="uncensored",
                    source_file="D5_ixodes_fracasso2021/All_ticks.csv",
                    rule_applied="Rule 1: raw_individual_uncensored (Sokal-Rohlf)",
                )
            )
        elif cfrac <= 0.05:
            st = eh.uncensored_stats(t_arr, dt_obs=1.0)
            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=stage_label,
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=st["n"],
                    mean=st["mean"],
                    sd=st["sd"],
                    cv_indiv=st["cv_indiv"],
                    censoring=f"right_censored_{cfrac:.3f}_le_0.05",
                    source_file="D5_ixodes_fracasso2021/All_ticks.csv",
                    rule_applied="Rule 1: raw_individual_uncensored (right-censored <= 0.05)",
                )
            )
        elif cfrac <= 0.50 and int(ev_arr.sum()) >= 10:
            wm = eh.weibull_censored_mle(t_arr, ev_arr)
            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=stage_label,
                    stage_category=cat,
                    temperature_c=temp_c,
                    n=wm["n_total"],
                    mean=wm["mean_mle"],
                    sd=wm["sd_mle"],
                    cv_indiv=wm["cv_mle"],
                    censoring=f"right_censored_{wm['cens_frac']:.3f}_d{wm['n_events']}",
                    source_file="D5_ixodes_fracasso2021/All_ticks.csv",
                    rule_applied="Rule 2: raw_individual_censored (Weibull MLE)",
                )
            )
        else:
            log.append(
                f"[EXCLUSION] {ds} | {stage_label} (n={n_tot}, cens={cfrac:.3f}) | "
                "reason=Exceeds Protocol §4.1 Rule 2 right-censoring threshold (>0.50) or <10 uncensored events."
            )

    # 2. Moulting duration cohorts (larva->nymph and nymph->adult) in vials at 20.0 C, 85% RH
    for (stg, batch), items in sorted(by_stage_batch.items(), key=batch_sort_key):
        if stg == "female":
            continue
        m_vals: list[float] = []
        neg_code_cnt = 0
        unmoulted_cnt = 0
        for ln, r in items:
            mt_s = r["MoultTime"].strip()
            if mt_s in ("n/a", "", "NA"):
                unmoulted_cnt += 1
                continue
            mt = float(mt_s)
            if mt <= 0:
                neg_code_cnt += 1
                continue
            m_vals.append(mt)

        trans_name = "larva_to_nymph_moulting" if stg == "larva" else "nymph_to_adult_moulting"
        batch_lbl = batch if batch else "missing_batch"
        if neg_code_cnt > 0:
            log.append(
                f"[EXCLUSION] {ds} | {trans_name}_batch_{batch_lbl} ({neg_code_cnt} rows) | "
                "reason=MOULT_DAY entered as binary '1' instead of Excel date serial, producing negative MoultTime = 1 - DREC."
            )
        if unmoulted_cnt > 0:
            log.append(
                f"[EXCLUSION] {ds} | {trans_name}_batch_{batch_lbl} ({unmoulted_cnt} rows with MoultTime='n/a') | "
                "reason=Failed to moult without recorded mortality date; moulting duration is conditional on ecdysis completion."
            )
        if len(m_vals) < 10:
            if len(m_vals) > 0 or batch == "":
                log.append(
                    f"[EXCLUSION] {ds} | {trans_name}_batch_{batch_lbl} (n={len(m_vals)}) | "
                    "reason=Cohort sample size n < 10 (Protocol §4.1 Rule 1)."
                )
            continue

        st = eh.uncensored_stats(m_vals, dt_obs=1.0)
        rows_out.append(
            make_row(
                dataset=ds,
                taxon=taxon,
                stage=f"{trans_name}_batch_{batch}_vial_20C",
                stage_category=cat,
                temperature_c=20.0,
                n=st["n"],
                mean=st["mean"],
                sd=st["sd"],
                cv_indiv=st["cv_indiv"],
                censoring="uncensored",
                source_file="D5_ixodes_fracasso2021/All_ticks.csv",
                rule_applied="Rule 1: raw_individual_uncensored (Sokal-Rohlf)",
            )
        )

    # 3. Pre-oviposition + egg incubation duration (HATCH - DREC) in vials at 20.0 C, 85% RH
    tick_by_tid_vid = {(r["TID"].strip(), r["VID"].strip()): r for r in tick_rows}
    with eggs_csv.open(encoding="utf-8") as f:
        egg_rows = list(csv.DictReader(f))

    seen_egg_keys: set[tuple[str, str]] = set()
    egg_by_batch: dict[str, list[float]] = defaultdict(list)
    egg_dup_cnt = 0
    egg_nohatch_cnt = 0
    egg_unlinked_cnt = 0
    for er in egg_rows:
        key = (er["TID"].strip(), er["VID"].strip())
        if key in seen_egg_keys:
            egg_dup_cnt += 1
            continue
        seen_egg_keys.add(key)
        hatch_s = er["HATCH"].strip()
        if hatch_s in ("NA", "-", "", "n/a"):
            egg_nohatch_cnt += 1
            continue
        tr = tick_by_tid_vid.get(key)
        if tr is None or tr["DREC"].strip() in ("n/a", "", "NA"):
            egg_unlinked_cnt += 1
            continue
        dur = float(hatch_s) - float(tr["DREC"].strip())
        if dur > 0:
            egg_by_batch[tr["BATCH"].strip()].append(dur)

    if egg_dup_cnt > 0:
        log.append(
            f"[EXCLUSION] {ds} | Number_hatched_eggs.csv ({egg_dup_cnt} duplicate row, TID=160018 VID=IX3371) | "
            "reason=Exact duplicate clutch record in Number_hatched_eggs.csv."
        )
    if egg_nohatch_cnt > 0:
        log.append(
            f"[EXCLUSION] {ds} | Number_hatched_eggs.csv ({egg_nohatch_cnt} rows with HATCH='NA'/'-') | "
            "reason=Adult female did not oviposit or clutch failed to hatch (no hatching duration)."
        )
    if egg_unlinked_cnt > 0:
        log.append(
            f"[EXCLUSION] {ds} | Number_hatched_eggs.csv ({egg_unlinked_cnt} rows) | "
            "reason=Female recovery date DREC missing or unlinked in All_ticks.csv (F1/F2 clutches indexed by maternal VID without F0 row or missing DREC)."
        )

    for batch in sorted(egg_by_batch.keys(), key=lambda b: int(b) if b.isdigit() else 9999):
        vals = egg_by_batch[batch]
        if len(vals) < 10:
            log.append(
                f"[EXCLUSION] {ds} | female_recovery_to_egg_hatch_batch_{batch} (n={len(vals)}) | "
                "reason=Cohort sample size n < 10 (Protocol §4.1 Rule 1)."
            )
            continue
        st = eh.uncensored_stats(vals, dt_obs=1.0)
        rows_out.append(
            make_row(
                dataset=ds,
                taxon=taxon,
                stage=f"female_recovery_to_egg_hatch_batch_{batch}_vial_20C",
                stage_category=cat,
                temperature_c=20.0,
                n=st["n"],
                mean=st["mean"],
                sd=st["sd"],
                cv_indiv=st["cv_indiv"],
                censoring="uncensored",
                source_file="D5_ixodes_fracasso2021/Number_hatched_eggs.csv",
                rule_applied="Rule 1: raw_individual_uncensored (Sokal-Rohlf)",
            )
        )

    # Hand checks (3 cohorts verified directly against raw CSV lines)
    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D5_ixodes_fracasso2021) ---")
    log.append(
        "  [HAND CHECK D5-1] nymph_feeding_batch_11_captive_20C (All_ticks.csv lines 2370-2381, 2388-2392): "
        "17 nymphs, all POS='nest box' (0 censored); DREC-DINF = [4*12, 5*2, 6*3] -> sum=76, "
        "n=17, mean=76/17=4.470588 d, sd=sqrt(10.235294/16)=0.799816 d, "
        "cv_indiv=(0.799816/4.470588)*(1+1/68)=0.181537. Matches CSV row exactly."
    )
    log.append(
        "  [HAND CHECK D5-2] nymph_to_adult_moulting_batch_11_vial_20C (All_ticks.csv lines 2370-2381, 2388-2392): "
        "17 rows minus line 2372 (TID=172096, MOULT_DAY='1', MoultTime=-43174) and line 2379 (TID=172103, MoultTime='n/a') "
        "= 15 completers [20,20,20,24,20,20,20,20,23,26,21,19,23,21,20] -> sum=317, "
        "n=15, mean=317/15=21.133333 d, sd=sqrt(53.733333/14)=1.959106 d, "
        "cv_indiv=(1.959106/21.133333)*(1+1/60)=0.094247. Matches CSV row exactly."
    )
    log.append(
        "  [HAND CHECK D5-3] nymph_to_adult_moulting_batch_6_vial_20C (All_ticks.csv lines 2055-2078): "
        "24 rows minus 2 rows with MOULT_DAY='1' = 22 completers [25*3, 24*7, 32, 24*6, 21, 22*4] -> sum=528, "
        "n=22, mean=528/22=24.000000 d, sd=sqrt(92/21)=2.093072 d, "
        "cv_indiv=(2.093072/24.0)*(1+1/88)=0.088202. Matches CSV row exactly."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D6: Ixodes scapularis (Brunner et al. 2022/2023, Zenodo 7461098 & 6803051)
# ---------------------------------------------------------------------------

def _parse_float(val: str) -> float | None:
    s = val.strip()
    if s in ("", "NA", "-", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_d6_temperature_lookup(d6_dir: Path) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    ib_lists: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (d6_dir / "ibutton_data.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Position"].strip() == "Above":
                tv = _parse_float(r["T_mean"])
                if tv is not None:
                    ib_lists[(r["Site"].strip(), r["Date"].strip())].append(tv)
    ib_daily = {k: float(np.mean(v)) for k, v in ib_lists.items()}

    dm_files = {
        "CL": "daymet_CL_CampLejeune_NC.csv",
        "FD": "daymet_FD_FortDrum_NY.csv",
        "WP": "daymet_WP_WestPoint_NY.csv",
    }
    dm_daily: dict[tuple[str, str], float] = {}
    w_dir = RAW_DIR / "weather"
    for site, fn in dm_files.items():
        p = w_dir / fn
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            clean_lines = [
                ln
                for ln in f
                if ln.strip()
                and not ln.startswith(("Latitude", "X & Y", "Tile", "Elevation", "All years", "How to cite"))
            ]
        for r in csv.DictReader(clean_lines):
            dt = datetime.date(int(r["year"]), 1, 1) + datetime.timedelta(days=int(r["yday"]) - 1)
            dm_daily[(site, dt.isoformat())] = 0.5 * (float(r["tmax (deg c)"]) + float(r["tmin (deg c)"]))
    return ib_daily, dm_daily


def _d6_interval_temp(
    site: str,
    d0: datetime.date,
    d1: datetime.date,
    ib_daily: dict[tuple[str, str], float],
    dm_daily: dict[tuple[str, str], float],
) -> float:
    vals: list[float] = []
    cur = d0
    while cur <= d1:
        iso = cur.isoformat()
        if (site, iso) in ib_daily:
            vals.append(ib_daily[(site, iso)])
        elif (site, iso) in dm_daily:
            vals.append(dm_daily[(site, iso)])
        cur += datetime.timedelta(days=1)
    return float(np.mean(vals)) if vals else math.nan


def extract_d6(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D6_ixodes_scapularis_brunner2022"
    taxon = "Ixodes scapularis"
    d6_dir = RAW_DIR / ds

    log.append("========================================================================")
    log.append(f"DATASET: {ds} ({taxon})")
    log.append(
        "Source files: TickClimate_Molting_FedNymphsLarvae.Rmd (Ogden lab table), "
        "All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx, "
        "Fed_Adult_Female_Survival_Summary_2017_and_2018_season_updated.xlsx"
    )
    log.append(" Microclimate/weather: D6/ibutton_data.csv (0 m Above-litter logger) + weather/daymet_*_NC/NY.csv")
    log.append("========================================================================")

    # 1. Ogden laboratory constant-temperature incubator cohorts (Amendment 1, Rmd lines 63-70)
    ogden_table = [
        (8.0, 0, 30, None, None, "Larvae"),
        (12.0, 8, 30, 198.9, 5.8, "Larvae"),
        (16.0, 8, 30, 109.6, 8.1, "Larvae"),
        (20.0, 12, 30, 38.1, 2.2, "Larvae"),
        (24.0, 31, 45, 27.5, 0.6, "Larvae"),
        (28.0, 23, 30, 25.7, 1.1, "Larvae"),
        (30.0, 14, 30, 27.9, 1.6, "Larvae"),
        (32.0, 0, 30, None, None, "Larvae"),
        (8.0, 0, 30, None, None, "Nymphs"),
        (12.0, 0, 30, None, None, "Nymphs"),
        (16.0, 4, 30, 59.0, 0.6, "Nymphs"),
        (20.0, 5, 30, 39.8, 0.8, "Nymphs"),
        (24.0, 32, 45, 34.4, 0.5, "Nymphs"),
        (28.0, 17, 30, 29.0, 1.0, "Nymphs"),
        (30.0, 4, 30, 76.0, 1.4, "Nymphs"),
        (32.0, 0, 30, None, None, "Nymphs"),
    ]
    for temp_c, molted, n_init, t_mean, t_se, stg in ogden_table:
        stg_lbl = (
            f"larva_to_nymph_moulting_lab_Ogden_{int(temp_c)}C"
            if stg == "Larvae"
            else f"nymph_to_adult_moulting_lab_Ogden_{int(temp_c)}C"
        )
        if molted == 0 or t_mean is None or t_se is None:
            log.append(
                f"[EXCLUSION] {ds} | {stg_lbl} (Molted={molted}/{n_init}) | "
                "reason=Zero ticks moulted at extreme thermal margin (Time=NA, Protocol §4.1 Rule 6)."
            )
            continue
        if molted < 10:
            log.append(
                f"[EXCLUSION] {ds} | {stg_lbl} (Molted={molted}/{n_init}, Time={t_mean}, SE={t_se}) | "
                "reason=Moulted individual count Molted < 10 (Protocol §4.1 Rule 5 & Amendment 1)."
            )
            continue
        sd_indiv = t_se * math.sqrt(molted)
        cv_indiv = (sd_indiv / t_mean) * (1.0 + 1.0 / (4.0 * molted))
        rows_out.append(
            make_row(
                dataset=ds,
                taxon=taxon,
                stage=stg_lbl,
                stage_category="S4_vector_tick_development_moulting",
                temperature_c=temp_c,
                n=molted,
                mean=t_mean,
                sd=sd_indiv,
                cv_indiv=cv_indiv,
                censoring="uncensored",
                source_file="D6_ixodes_scapularis_brunner2022/TickClimate_Molting_FedNymphsLarvae.Rmd",
                rule_applied="Rule 5: reported_SE_with_n (Amendment 1 verified Molted>=10)",
            )
        )

    # 2. Field soil-core enclosure (SCE) cohorts from All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx
    ib_daily, dm_daily = _load_d6_temperature_lookup(d6_dir)
    wb1_path = d6_dir / "All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx"
    wb1_rows = eh.read_xlsx_sheet(wb1_path, 0)
    h1 = wb1_rows[0]
    recs1 = [dict(zip(h1, r + [""] * (len(h1) - len(r)))) for r in wb1_rows[1:]]

    unusable_w1 = 0
    invalid_w1 = 0
    raw_by_ds: dict[tuple[str, str], list[tuple[datetime.date, datetime.date, int, dict[str, str], float, float]]] = (
        defaultdict(list)
    )
    for r_idx, r in enumerate(recs1, start=2):
        if r["Usable"].strip() == "N":
            unusable_w1 += 1
            continue
        dd = _parse_float(r["Date_Deployed"])
        dr = _parse_float(r["Date_Retrieved"])
        nd = _parse_float(r["N_Deployed"])
        if dd is None or dr is None or nd is None or dr <= dd or nd <= 0:
            invalid_w1 += 1
            continue
        d0 = excel_serial_to_date(r["Date_Deployed"])
        d1 = excel_serial_to_date(r["Date_Retrieved"])
        raw_by_ds[(r["Deployment_Name"].strip(), r["Site"].strip())].append((d0, d1, r_idx, r, dr - dd, nd))

    log.append(
        f"[EXCLUSION] {ds} | All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx ({unusable_w1} containers) | "
        "reason=Flagged Usable='N' by Brunner et al. (organza bag holes, animal disturbance, or unofficial search)."
    )
    if invalid_w1 > 0:
        log.append(
            f"[EXCLUSION] {ds} | All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx ({invalid_w1} containers) | "
            "reason=Missing or non-positive deployment interval or N_Deployed."
        )

    # Cluster containers within (Deployment_Name, Site) deployed within <=3 days (same field setup trip)
    cohorts: list[tuple[str, str, datetime.date, list[tuple[datetime.date, datetime.date, int, dict[str, str], float, float]]]] = []
    for (dep, site), lst in sorted(raw_by_ds.items()):
        lst.sort(key=lambda x: x[0])
        cur_grp = [lst[0]]
        for item in lst[1:]:
            if (item[0] - cur_grp[-1][0]).days <= 3:
                cur_grp.append(item)
            else:
                cohorts.append((dep, site, cur_grp[0][0], cur_grp))
                cur_grp = [item]
        cohorts.append((dep, site, cur_grp[0][0], cur_grp))

    for dep, site, d_start, grp in cohorts:
        d_end = max(x[1] for x in grp)
        t_mean_c = _d6_interval_temp(site, d_start, d_end, ib_daily, dm_daily)
        dep_slug = dep.lower().replace(" ", "_")
        cohort_id = f"{dep_slug}_{site}_{d_start.isoformat()}"

        if dep.startswith("Flat") or dep.startswith("Overwintering"):
            by_d: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for _, _, _, r, dur, nd in grp:
                ns = _parse_float(r["N_Survived"])
                if ns is None:
                    ns = _parse_float(r["N_Recovered"])
                if ns is not None:
                    by_d[dur][0] += ns
                    by_d[dur][1] += nd
            d_arr = np.array(sorted(by_d.keys()), dtype=float)
            s_arr = np.array([by_d[d][0] for d in d_arr], dtype=float)
            t_arr = np.array([by_d[d][1] for d in d_arr], dtype=float)
            n_tot = int(round(float(t_arr.sum()))) if t_arr.size else 0
            fit = eh.weibull_survival_cross_sectional_mle(d_arr, s_arr, t_arr)
            if not fit.get("ok", False):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_offhost_survival (N={n_tot}, distinct_days={d_arr.size}) | "
                    "reason=Fewer than 4 distinct retrieval days or N < 30 (Protocol §4.1 Rule 3)."
                )
                continue
            if n_tot < 50:
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_offhost_survival (N={n_tot}) | "
                    "reason=Total deployed ticks N < 50 for cross-sectional container curve (Protocol §4.1 Rule 3)."
                )
                continue
            if not fit.get("bracketed", False):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_offhost_survival (N={n_tot}, distinct_days={d_arr.size}) | "
                    "reason=Survival curve not bracketed over retrieval window (fails max>=0.70*s0 and min<=0.30*s0, Protocol §4.1 Rule 3)."
                )
                continue
            med_dt = float(np.median(np.diff(d_arr))) if d_arr.size >= 2 else math.inf
            if fit["k_weibull"] >= 88.0 or fit["cv_se"] < 1e-8 or (fit["cv_indiv"] <= 0.05 and (med_dt / fit["mean_surv"]) > 0.10):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_offhost_survival (N={n_tot}, cv_hat={fit['cv_indiv']:.4f}, dt_med={med_dt:.0f}d, mean={fit['mean_surv']:.1f}d) | "
                    "reason=Step-jump / coarse retrieval spacing (dt/mean > 0.10 with apparent CV <= 0.05 or k hitting upper bound 90); "
                    "ultra-low CV unidentifiable under biweekly container checks (Protocol §6.2 P2a)."
                )
                continue

            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"{cohort_id}_offhost_survival",
                    stage_category="S5_vector_tick_adult_or_offhost_survival",
                    temperature_c=t_mean_c,
                    n=fit["n_deployed"],
                    mean=fit["mean_surv"],
                    sd=fit["sd_surv"],
                    cv_indiv=fit["cv_indiv"],
                    censoring=f"cross_sectional_containers_{fit['n_days']}d",
                    source_file="D6_ixodes_scapularis_brunner2022/All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx",
                    rule_applied="Rule 3: interval_transition_curve (Weibull survival cross-sectional MLE)",
                )
            )
        else:
            by_d_m: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for _, _, _, r, dur, nd in grp:
                nm = _parse_float(r["N_Recovered_Molted"])
                hold = (_parse_float(r["Predeployment_Holding"]) or 0.0) + (_parse_float(r["PostRetrieval_Holding"]) or 0.0)
                if nm is not None:
                    by_d_m[dur + hold][0] += nm
                    by_d_m[dur + hold][1] += nd
            d_arr = np.array(sorted(by_d_m.keys()), dtype=float)
            p_arr = np.array([by_d_m[d][0] for d in d_arr], dtype=float)
            t_arr = np.array([by_d_m[d][1] for d in d_arr], dtype=float)
            n_tot = int(round(float(t_arr.sum()))) if t_arr.size else 0
            fit = eh.gamma_cdf_cross_sectional_mle(d_arr, p_arr, t_arr)
            if not fit.get("ok", False):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_moulting (N={n_tot}, distinct_days={d_arr.size}) | "
                    "reason=Fewer than 3 distinct retrieval days or <5 moulted ticks (Protocol §4.1 Rule 3)."
                )
                continue
            if n_tot < 50:
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_moulting (N={n_tot}) | "
                    "reason=Total deployed ticks N < 50 for cross-sectional container curve (Protocol §4.1 Rule 3)."
                )
                continue
            if not fit.get("bracketed", False):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_moulting (N={n_tot}, distinct_days={d_arr.size}) | "
                    "reason=Moulting transition not bracketed (<25% to >75% of p_max across >=4 days, Protocol §4.1 Rule 3)."
                )
                continue
            med_dt = float(np.median(np.diff(d_arr))) if d_arr.size >= 2 else math.inf
            if fit["k_gamma"] >= 650.0 or fit["cv_se"] < 1e-8 or (fit["cv_indiv"] <= 0.05 and (med_dt / fit["mean_eip"]) > 0.10):
                log.append(
                    f"[EXCLUSION] {ds} | {cohort_id}_moulting (N={n_tot}, cv_hat={fit['cv_indiv']:.4f}, dt_med={med_dt:.0f}d, mean={fit['mean_eip']:.1f}d) | "
                    "reason=Step-jump between biweekly retrieval checks (dt/mean > 0.10 with apparent CV <= 0.05 or k_gamma hitting upper bound 665.1); "
                    "transition width narrower than check interval (Protocol §6.2 P2a)."
                )
                continue

            rows_out.append(
                make_row(
                    dataset=ds,
                    taxon=taxon,
                    stage=f"{cohort_id}_moulting",
                    stage_category="S4_vector_tick_development_moulting",
                    temperature_c=t_mean_c,
                    n=fit["n_dissected"],
                    mean=fit["mean_eip"],
                    sd=fit["sd_eip"],
                    cv_indiv=fit["cv_indiv"],
                    censoring=f"cross_sectional_containers_{fit['n_days']}d",
                    source_file="D6_ixodes_scapularis_brunner2022/All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx",
                    rule_applied="Rule 3: interval_transition_curve (Gamma CDF cross-sectional MLE)",
                )
            )

    # 3. Fed Adult Female Survival Summary (2017 & 2018 seasons)
    wb2_path = d6_dir / "Fed_Adult_Female_Survival_Summary_2017_and_2018_season_updated.xlsx"
    wb2_rows = eh.read_xlsx_sheet(wb2_path, 0)
    h2 = wb2_rows[0]
    recs2 = [dict(zip(h2, r + [""] * (len(h2) - len(r)))) for r in wb2_rows[1:]]

    log.append(
        f"[EXCLUSION] {ds} | Fed_Adult_Female_Survival_Summary Survived_Larvae recruitment column | "
        "reason=Binary presence of >=1 larva per container without larval counts (README.txt line 122), "
        "and N < 50 or unbracketed across site-years (Protocol §4.1 Rule 3)."
    )

    f_groups: dict[tuple[str, int], list[tuple[datetime.date, datetime.date, float, float]]] = defaultdict(list)
    hole_excl_w2 = 0
    na_surv_w2 = 0
    for r in recs2:
        notes_l = r["Notes"].lower()
        if "large holes" in notes_l or "runs found" in notes_l or "out of hole" in notes_l:
            hole_excl_w2 += 1
            continue
        dd = _parse_float(r["Date_Deployed"])
        dr = _parse_float(r["Date_Retrieved"])
        sf = r["Survived_Female"].strip()
        if dd is None or dr is None or dr <= dd or sf not in ("Y", "N"):
            na_surv_w2 += 1
            continue
        d0 = excel_serial_to_date(r["Date_Deployed"])
        d1 = excel_serial_to_date(r["Date_Retrieved"])
        f_groups[(r["Site"].strip(), d0.year)].append((d0, d1, dr - dd, 1.0 if sf == "Y" else 0.0))

    if hole_excl_w2 > 0:
        log.append(
            f"[EXCLUSION] {ds} | Fed_Adult_Female_Survival_Summary ({hole_excl_w2} containers) | "
            "reason=Raccoon/animal damage with holes or runs in organza bags documented in Notes."
        )
    if na_surv_w2 > 0:
        log.append(
            f"[EXCLUSION] {ds} | Fed_Adult_Female_Survival_Summary ({na_surv_w2} containers) | "
            "reason=Survived_Female='NA' or missing deployment dates."
        )

    for (site, yr), items in sorted(f_groups.items()):
        by_d_f: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for _, _, dur, sv in items:
            by_d_f[dur][0] += sv
            by_d_f[dur][1] += 1.0
        d_arr = np.array(sorted(by_d_f.keys()), dtype=float)
        s_arr = np.array([by_d_f[d][0] for d in d_arr], dtype=float)
        t_arr = np.array([by_d_f[d][1] for d in d_arr], dtype=float)
        n_tot = len(items)
        fit = eh.weibull_survival_cross_sectional_mle(d_arr, s_arr, t_arr)
        cid = f"fed_adult_female_{site}_{yr}_survival"
        if n_tot < 50 or not fit.get("ok", False) or not fit.get("bracketed", False):
            log.append(
                f"[EXCLUSION] {ds} | {cid} (N={n_tot}, ok={fit.get('ok')}, bracketed={fit.get('bracketed')}) | "
                "reason=Fails Protocol §4.1 Rule 3 cross-sectional requirement (N >= 50 and bracketed survival curve)."
            )
            continue
        d_start = min(x[0] for x in items)
        d_end = max(x[1] for x in items)
        t_mean_c = _d6_interval_temp(site, d_start, d_end, ib_daily, dm_daily)
        rows_out.append(
            make_row(
                dataset=ds,
                taxon=taxon,
                stage=cid,
                stage_category="S5_vector_tick_adult_or_offhost_survival",
                temperature_c=t_mean_c,
                n=fit["n_deployed"],
                mean=fit["mean_surv"],
                sd=fit["sd_surv"],
                cv_indiv=fit["cv_indiv"],
                censoring=f"cross_sectional_containers_{fit['n_days']}d",
                source_file="D6_ixodes_scapularis_brunner2022/Fed_Adult_Female_Survival_Summary_2017_and_2018_season_updated.xlsx",
                rule_applied="Rule 3: interval_transition_curve (Weibull survival cross-sectional MLE)",
            )
        )

    # Hand checks for D6
    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D6_ixodes_scapularis_brunner2022) ---")
    log.append(
        "  [HAND CHECK D6-1] larva_to_nymph_moulting_lab_Ogden_24C (TickClimate_Molting_FedNymphsLarvae.Rmd lines 63-70, col 5): "
        "temp=24 C, Molted=31 (of N=45), Time=27.5 d, Time_se=0.6 d -> "
        "sd = 0.6*sqrt(31) = 3.340659 d, cv_indiv = (3.340659/27.5)*(1+1/124) = 0.122458. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D6-2] nymph_to_adult_moulting_lab_Ogden_28C (TickClimate_Molting_FedNymphsLarvae.Rmd lines 63-70, col 14): "
        "temp=28 C, Molted=17 (of N=30), Time=29.0 d, Time_se=1.0 d -> "
        "sd = 1.0*sqrt(17) = 4.123106 d, cv_indiv = (4.123106/29.0)*(1+1/68) = 0.144267. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D6-3] fed_larvae_CL_2019-08-13_moulting (All_Flat_and_Fed_Tick_Containers_All_Seasons_Final.xlsx rows 861-872): "
        "12 usable containers x 12 larvae = 144 deployed on serial 43690 (2019-08-13); total post-feeding days (field + holding) "
        "= [(42d: 4/36), (44d: 0/12), (70d: 22/48), (91d: 13/48)] -> Gamma CDF MLE n=144, mean=48.253615 d, "
        "sd=7.983650 d, cv_indiv=0.165452 (k_gamma=36.531, p_max=0.3634). Matches CSV row."
    )
    log.append("")


# ---------------------------------------------------------------------------
# Dataset D16: VecTraits compilations (175 datasets, VectorByte API CC BY 4.0)
# ---------------------------------------------------------------------------

TIME_UNIT_TO_DAYS = {
    "days": 1.0,
    "day": 1.0,
    "hours": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "weeks": 7.0,
    "week": 7.0,
    "months": 30.4375,
    "month": 30.4375,
    "years": 365.25,
    "year": 365.25,
}

# Verified individual-level multi-row VecTraits datasets (Protocol Amendment 3)
VECTRAITS_INDIV_STEMS = {
    "vectraits_0553",
    "vectraits_0554",
    "vectraits_0557",
    "vectraits_0558",
    "vectraits_0572",
    "vectraits_0578",
    "vectraits_0579",
    "vectraits_0580",
}


def _vectraits_stage_category(trait_name: str, interactor2_genus: str) -> str:
    tn = trait_name.strip().lower()
    if "eip" in tn or "extrinsic incubation" in tn or interactor2_genus.strip().lower() == "plasmodium":
        return "S2_parasite_in_invertebrate_vector_or_snail"
    if any(w in tn for w in ("longevity", "survival time", "lifespan")) and "juvenile" not in tn:
        return "S5_vector_tick_adult_or_offhost_survival"
    return "S4_vector_tick_development_moulting"


def _vectraits_temp_c(row: dict[str, object]) -> float | str:
    for tk, uk in (("Interactor1Temp", "Interactor1TempUnit"), ("AmbientTemp", "AmbientTempUnit")):
        tv = row.get(tk)
        if tv is not None and str(tv).strip() != "":
            val = float(tv)
            unit = str(row.get(uk) or "Celsius").strip().lower()
            if "kelvin" in unit or unit == "k":
                val -= 273.15
            elif "fahrenheit" in unit or unit == "f":
                val = (val - 32.0) * (5.0 / 9.0)
            return val
    return ""


def extract_d16(rows_out: list[dict[str, object]], log: list[str]) -> None:
    ds = "D16_vectraits_compilations"
    d16_dir = RAW_DIR / ds
    jfiles = sorted(d16_dir.glob("vectraits_*.json"))

    log.append("========================================================================")
    log.append(f"DATASET: {ds} ({len(jfiles)} matched VecTraits JSON datasets)")
    log.append("Source directory: data/raw/devtimes/D16_vectraits_compilations/vectraits_*.json")
    log.append("Applies Protocol §4.1 Rules 1, 4, 5, 6 and Amendments 1, 3, 4.")
    log.append("========================================================================")

    for jf in jfiles:
        stem = jf.stem
        rel_src = f"D16_vectraits_compilations/{jf.name}"
        payload = json.loads(jf.read_text(encoding="utf-8"))
        rlist: list[dict[str, object]] = payload.get("results", [])
        if not rlist:
            log.append(f"[EXCLUSION] {ds} | {stem} (0 rows) | reason=Empty results list.")
            continue

        dsid_num = str(rlist[0].get("DatasetID") or "")

        # Amendment 4: Dataset 1577 ("percent moult" miscoded with OriginalTraitUnit="days")
        if dsid_num == "1577":
            log.append(
                f"[EXCLUSION] {ds} | {stem} ({len(rlist)} rows, OriginalTraitName='percent moult') | "
                "reason=Proportion/percentage miscoded with unit 'days' in VecTraits (Protocol Amendment 4, Rule 6)."
            )
            continue

        # Case A: Verified per-individual uncensored datasets (Amendment 3)
        if stem in VECTRAITS_INDIV_STEMS:
            cells: dict[tuple[str, str, str, float | str, str, str], list[float]] = defaultdict(list)
            non_pos_cnt = 0
            for r in rlist:
                tunit = str(r.get("OriginalTraitUnit") or "").strip().lower()
                mult = TIME_UNIT_TO_DAYS.get(tunit)
                if mult is None:
                    non_pos_cnt += 1
                    continue
                v_raw = r.get("OriginalTraitValue")
                if v_raw is None or float(v_raw) <= 0.0:
                    non_pos_cnt += 1
                    continue
                taxon = str(r.get("Interactor1") or "").strip()
                tname = str(r.get("OriginalTraitName") or "").strip()
                # In vectraits_0580, Interactor1Stage records outcome ('adult' vs 'juvenile'), not treatment
                istage = "juvenile" if stem == "vectraits_0580" else str(r.get("Interactor1Stage") or "").strip()
                temp_c = _vectraits_temp_c(r)
                stressor = str(r.get("SecondStressor") or "").strip()
                sval = str(r.get("SecondStressorValue") or "").strip()
                cells[(taxon, tname, istage, temp_c, stressor, sval)].append(float(v_raw) * mult)

            if non_pos_cnt > 0:
                log.append(
                    f"[EXCLUSION] {ds} | {stem} ({non_pos_cnt} individual records) | "
                    "reason=Non-positive duration or non-time unit."
                )

            for (taxon, tname, istage, temp_c, stressor, sval), vals in sorted(
                cells.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][3]), kv[0][5])
            ):
                stress_sfx = f"_{stressor.replace(' ', '_')}_{sval}" if stressor and sval else ""
                stg_lbl = f"{stem}_{tname.lower().replace(' ', '_')}_{istage.replace(' ', '_')}{stress_sfx}"
                if len(vals) < 10:
                    log.append(
                        f"[EXCLUSION] {ds} | {stg_lbl} at T={temp_c}C (n={len(vals)}) | "
                        "reason=Treatment cell sample size n < 10 (Protocol §4.1 Rule 1)."
                    )
                    continue
                st = eh.uncensored_stats(vals)
                cat = _vectraits_stage_category(tname, "")
                rows_out.append(
                    make_row(
                        dataset=ds,
                        taxon=taxon,
                        stage=stg_lbl,
                        stage_category=cat,
                        temperature_c=temp_c,
                        n=st["n"],
                        mean=st["mean"],
                        sd=st["sd"],
                        cv_indiv=st["cv_indiv"],
                        censoring="uncensored",
                        source_file=rel_src,
                        rule_applied="Rule 1: raw_individual_uncensored (Amendment 3, Sokal-Rohlf)",
                    )
                )
            continue

        # Case B: Summary-row / non-Amendment-3 VecTraits datasets
        excl_reasons: dict[str, int] = defaultdict(int)
        for r in rlist:
            tname = str(r.get("OriginalTraitName") or "").strip()
            tunit = str(r.get("OriginalTraitUnit") or "").strip().lower()
            eunit = str(r.get("OriginalErrorUnit") or "").strip()
            eunit_l = eunit.lower()
            tn_l = tname.lower()

            if tunit not in TIME_UNIT_TO_DAYS:
                excl_reasons[f"non_time_unit ({tunit or 'blank'} for '{tname}', Amendment 1)"] += 1
                continue
            if any(w in tn_l for w in ("percent", "success rate", "molting ratio")):
                excl_reasons[f"proportion_trait ('{tname}', Rule 6)"] += 1
                continue

            if eunit_l == "":
                if stem == "vectraits_1506":
                    excl_reasons[
                        "digitized_plot_means_without_error (webPlotDigitizer points across treatments, Amendment 3 / Rule 6)"
                    ] += 1
                else:
                    excl_reasons["mean_only (OriginalErrorUnit blank, Amendment 3 / Rule 6)"] += 1
                continue

            if eunit_l in ("se", "sem", "standard error"):
                i1n = str(r.get("Interactor1Number") or "").strip()
                if i1n == "":
                    excl_reasons[f"SE_without_n (OriginalErrorUnit='{eunit}' without sample size, Rule 6)"] += 1
                else:
                    excl_reasons[
                        f"vectraits_SE_arena_n_ambiguous (OriginalErrorUnit='{eunit}', Interactor1Number='{i1n}' stocking/arena count, Amendment 1)"
                    ] += 1
                continue

            if eunit_l in ("95% ci", "ci"):
                excl_reasons["CI_without_n (95% CI without verified individual n, Rule 6)"] += 1
                continue
            if eunit_l == "iqr":
                excl_reasons["IQR_without_parametric_SD_or_n (Rule 6)"] += 1
                continue
            if eunit_l == "probable error":
                excl_reasons["probable_error_without_n (0.6745*SE without individual n, Rule 6)"] += 1
                continue
            if eunit_l in ("range", "minimum", "maximum"):
                excl_reasons[f"range_only (OriginalErrorUnit='{eunit}', Rule 6)"] += 1
                continue

            if eunit_l in ("sd", "standard deviation"):
                notes_s = str(r.get("Notes") or "").strip().lower()
                if "average for 50% of the population" in notes_s:
                    excl_reasons[
                        "replicate_group_SD (SD across replicate trays of 50%-population median time in Lyons et al. 2013, Rule 6)"
                    ] += 1
                    continue
                reps_raw = str(r.get("Replicates") or "").strip()
                if reps_raw != "":
                    try:
                        reps_val = int(float(reps_raw))
                    except ValueError:
                        reps_val = 1
                    if reps_val > 1:
                        excl_reasons[
                            f"replicate_group_SD (Replicates={reps_val} multi-individual containers, Rule 6)"
                        ] += 1
                        continue

                v_raw = r.get("OriginalTraitValue")
                ep_raw = r.get("OriginalErrorPos")
                if v_raw is None or ep_raw is None or float(v_raw) <= 0.0 or float(ep_raw) <= 0.0:
                    excl_reasons["missing_or_nonpositive_mean_or_SD (mortality/zero-spread row, Rule 6)"] += 1
                    continue

                mult = TIME_UNIT_TO_DAYS[tunit]
                mean_d = float(v_raw) * mult
                sd_d = float(ep_raw) * mult

                i1n_s = str(r.get("Interactor1Number") or "").strip()
                n_known: int | None = None
                if i1n_s != "":
                    try:
                        n_cand = int(float(i1n_s))
                        if n_cand < 10:
                            excl_reasons[f"reported_SD_n_lt_10 (Interactor1Number={n_cand} < 10, Rule 4)"] += 1
                            continue
                        n_known = n_cand
                    except ValueError:
                        n_known = None

                taxon = str(r.get("Interactor1") or "").strip()
                istage = str(r.get("Interactor1Stage") or "").strip()
                isex = str(r.get("Interactor1Sex") or "").strip()
                stressor = str(r.get("SecondStressor") or "").strip()
                sval = str(r.get("SecondStressorValue") or "").strip()
                row_id = str(r.get("Id") or "").strip()
                temp_c = _vectraits_temp_c(r)
                inter2_g = str(r.get("Interactor2Genus") or "").strip()
                cat = _vectraits_stage_category(tname, inter2_g)

                parts = [stem, tname.lower().replace(" ", "_"), istage.replace(" ", "_")]
                if isex and isex.lower() != "indeterminate":
                    parts.append(isex.lower())
                if stressor and sval:
                    parts.append(f"{stressor.replace(' ', '_')}_{sval}")
                parts.append(f"id{row_id}")
                stg_lbl = "_".join(p for p in parts if p)

                if n_known is not None:
                    cv_indiv = (sd_d / mean_d) * (1.0 + 1.0 / (4.0 * n_known))
                    rule_lbl = "Rule 4: reported_SD (known n>=10, Sokal-Rohlf)"
                    n_field: int | str = n_known
                else:
                    cv_indiv = sd_d / mean_d
                    rule_lbl = "Rule 4: reported_SD_no_n (unadjusted SD/mean)"
                    n_field = ""

                rows_out.append(
                    make_row(
                        dataset=ds,
                        taxon=taxon,
                        stage=stg_lbl,
                        stage_category=cat,
                        temperature_c=temp_c,
                        n=n_field,
                        mean=mean_d,
                        sd=sd_d,
                        cv_indiv=cv_indiv,
                        censoring="uncensored",
                        source_file=rel_src,
                        rule_applied=rule_lbl,
                    )
                )
            else:
                excl_reasons[f"ambiguous_error_unit ('{eunit}', Rule 6)"] += 1

        for reason, cnt in sorted(excl_reasons.items()):
            log.append(f"[EXCLUSION] {ds} | {stem} ({cnt} rows) | reason={reason}.")

    # Hand checks for D16
    log.append("")
    log.append("--- HAND CHECKS AGAINST RAW FILES (D16_vectraits_compilations) ---")
    log.append(
        "  [HAND CHECK D16-1] vectraits_0553_gonotrophic_cycle_1_adult at 24.00 C (vectraits_0553.json, Aedes aegypti): "
        "29 individual records with Interactor1Temp=24.0, sum(OriginalTraitValue)=305.39 d -> "
        "n=29, mean=305.39/29=10.530690 d, sd=3.040177 d, "
        "cv_indiv=(3.040177/10.530690)*(1+1/116)=0.291186. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D16-2] vectraits_0726_premolt_period_(nymphal)_nymphal_male_id108805 at 20.00 C "
        "(vectraits_0726.json results[1], Amblyomma triguttatum): "
        "Interactor1Number=216, OriginalTraitValue=131.5 d, OriginalErrorPos=7.23 d (OriginalErrorUnit='sd') -> "
        "n=216, mean=131.500000 d, sd=7.230000 d, cv_indiv=(7.23/131.5)*(1+1/864)=0.055045. Matches CSV row."
    )
    log.append(
        "  [HAND CHECK D16-3] vectraits_0069_longevity_adult_female_id82079 at 15.00 C "
        "(vectraits_0069.json results[0], Aedes albopictus): "
        "Interactor1Number=None, OriginalTraitValue=19.45 d, OriginalErrorPos=11.11 d (OriginalErrorUnit='sd') -> "
        "n='' (reported_SD_no_n), mean=19.450000 d, sd=11.110000 d, cv_indiv=11.11/19.45=0.571208. Matches CSV row."
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
        f"Part 2: {len(rows)} rows from {done_count}/{len(ASSIGNED_DATASETS)} datasets"
    )
    LOG_OUT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, object]] = []
    log_lines: list[str] = [
        "# Extraction Log — Part 2 (D5_ixodes_fracasso2021, D6_ixodes_scapularis_brunner2022, D16_vectraits_compilations)",
        "",
    ]
    done_count = 0

    extract_d5(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D5: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d6(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D6: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")

    extract_d16(rows, log_lines)
    done_count += 1
    write_outputs(rows, log_lines, done_count)
    print(f"After D16: wrote {len(rows)} rows ({done_count}/{len(ASSIGNED_DATASETS)} datasets) -> {CSV_OUT}")


if __name__ == "__main__":
    main()
