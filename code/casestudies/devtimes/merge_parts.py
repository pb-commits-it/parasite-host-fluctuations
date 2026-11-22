#!/usr/bin/env python3
"""Merge `stage_cv_part{1..4}.csv` into `results/stage_cv.csv` and audit Step A logs.

Step A merge script (`r46-devtimes`):
  1. Reproduces known results on synthetic merge/deduplication and estimator inputs
     before processing empirical outputs (study rules Rule 7).
  2. Reads `results/stage_cv_part{1..4}.csv`, verifies column-header consistency and
     row-level schema validity across all four parts, drops exact duplicate rows
     preserving part order, and writes `results/stage_cv.csv` with a single header.
  3. Parses `results/extraction_log_part{1..4}.txt` and `results/stage_cv.csv` to
     produce complete, traceable counts by dataset, stage category, estimator rule,
     strict-n subset, exclusions, zero-row datasets, and by-hand checks, saving
     `results/merge_summary.json` and `results/merge_summary.txt` (CASESTUDY_SPEC Rule 5).

Run with:
  python3 -I casestudies/devtimes/merge_parts.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import extract_helpers as eh  # noqa: E402

RESULTS_DIR = HERE / "results"
PART_CSVS = [RESULTS_DIR / f"stage_cv_part{i}.csv" for i in (1, 2, 3, 4)]
PART_LOGS = [RESULTS_DIR / f"extraction_log_part{i}.txt" for i in (1, 2, 3, 4)]
MERGED_CSV = RESULTS_DIR / "stage_cv.csv"
SUMMARY_JSON = RESULTS_DIR / "merge_summary.json"
SUMMARY_TXT = RESULTS_DIR / "merge_summary.txt"

EXPECTED_COLUMNS = [
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

ALL_ASSIGNED_DATASETS = [
    ("D1_malaria_embo2024", 1),
    ("D2_malariotherapy_plos2013", 1),
    ("D3_anopheles_pf_shapiro2017", 1),
    ("D4_plasmodium_msos2021", 1),
    ("D5_ixodes_fracasso2021", 2),
    ("D6_ixodes_scapularis_brunner2022", 2),
    ("D7_schisto_leclech2019", 3),
    ("D8_schisto_civitello2020", 3),
    ("D9_schisto_thermal_nguyen2021", 3),
    ("D10_schisto_thermal_aslan2024", 3),
    ("D11_trematode_taskinen2022", 4),
    ("D12_trematode_gleichsner2016", 4),
    ("D13_nematode_marshallagia2023", 4),
    ("D14_nematode_heligmosomoides", 4),
    ("D15_helminth_clc_benesh2017", 4),
    ("D16_vectraits_compilations", 2),
]

STAGE_CATEGORIES = [
    "S1_endoparasite_vertebrate",
    "S2_parasite_in_invertebrate_vector_or_snail",
    "S3_free_living_parasite_stage",
    "S4_vector_tick_development_moulting",
    "S5_vector_tick_adult_or_offhost_survival",
]


def merge_row_tables(
    headers: list[list[str]],
    tables: list[list[dict[str, str]]],
    expected_columns: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]], int]:
    """Validate identical headers across parts and concatenate rows, dropping exact duplicates."""
    if not headers or not tables or len(headers) != len(tables):
        raise ValueError("Non-empty matching headers and tables lists are required.")
    ref_header = headers[0]
    if expected_columns is not None and ref_header != expected_columns:
        raise ValueError(f"Part 1 header {ref_header} != expected {expected_columns}")
    for idx, hdr in enumerate(headers[1:], start=2):
        if hdr != ref_header:
            raise ValueError(f"Column mismatch in part {idx}: {hdr} != {ref_header}")

    seen: set[tuple[str, ...]] = set()
    merged: list[dict[str, str]] = []
    dropped_exact = 0
    for part_idx, rows in enumerate(tables, start=1):
        for row_idx, row in enumerate(rows, start=1):
            if set(row.keys()) != set(ref_header) or None in row:
                raise ValueError(f"Malformed row width in part {part_idx}, row {row_idx}")
            key = tuple(row[col] for col in ref_header)
            if key in seen:
                dropped_exact += 1
                continue
            seen.add(key)
            merged.append({col: row[col] for col in ref_header})
    return list(ref_header), merged, dropped_exact


def run_known_self_checks() -> list[str]:
    """study rules Rule 7: verify merge logic and helper math against known results first."""
    lines = ["=== Rule 7 Known-Result Self-Checks (tolerances stated beforehand) ==="]

    # Check 1: synthetic merge + exact deduplication (exact integer tolerance = 0)
    cols = ["a", "b"]
    t1 = [{"a": "x", "b": "1"}, {"a": "y", "b": "2"}]
    t2 = [{"a": "y", "b": "2"}, {"a": "z", "b": "3"}, {"a": "x", "b": "1"}]
    hdr, out, dropped = merge_row_tables([cols, cols], [t1, t2], expected_columns=cols)
    assert hdr == cols and len(out) == 3 and dropped == 2
    assert [r["a"] for r in out] == ["x", "y", "z"]
    mismatch_caught = False
    try:
        merge_row_tables([cols, ["a", "c"]], [t1, [{"a": "q", "c": "9"}]])
    except ValueError:
        mismatch_caught = True
    assert mismatch_caught, "Expected ValueError on inconsistent column names"
    lines.append(
        "  [PASS] merge_row_tables synthetic check: 5 rows -> 3 unique, 2 exact duplicates dropped "
        "(exp 3 unique, 2 dropped, tol 0); column-mismatch ValueError verified."
    )

    # Check 2: Sokal-Rohlf CV formula on [10, 20, 30] (tolerance 1e-12)
    st = eh.uncensored_stats([10.0, 20.0, 30.0])
    exp_cv = 0.5 * (1.0 + 1.0 / 12.0)
    assert abs(st["mean"] - 20.0) < 1e-12
    assert abs(st["sd"] - 10.0) < 1e-12
    assert abs(st["cv_indiv"] - exp_cv) < 1e-12
    lines.append(
        f"  [PASS] extract_helpers.uncensored_stats([10,20,30]): cv_indiv={st['cv_indiv']:.12f} "
        f"(exp {exp_cv:.12f}, tol 1e-12)."
    )
    lines.append("")
    return lines


def classify_rule(rule_str: str) -> str:
    s = rule_str.lower()
    if "reported_sd_no_n" in s:
        return "Rule 4: reported_SD_no_n (unstated n)"
    if "reported_sd" in s:
        return "Rule 4: reported_SD (known n >= 10)"
    if "reported_se_with_n" in s:
        return "Rule 5: reported_SE_with_n (verified indiv n >= 10)"
    if "raw_individual_uncensored" in s:
        return "Rule 1: raw_individual_uncensored"
    if "raw_individual_censored" in s:
        return "Rule 2: raw_individual_censored (Weibull MLE)"
    if "cross_sectional_cdf_mle" in s or "gamma cdf cross-sectional mle" in s:
        return "Rule 3: interval_transition_curve (Gamma CDF MLE)"
    if "weibull survival" in s:
        if "no_indiv_n" in s:
            return "Rule 3: interval_transition_curve (Weibull survival MLE, no_indiv_n)"
        return "Rule 3: interval_transition_curve (Weibull survival MLE, known N >= 50)"
    return f"Other ({rule_str})"


def classify_d16_exclusion(entry: str) -> tuple[str, int]:
    """Extract row count and canonical reason category for a D16 exclusion log entry."""
    m_rows = re.search(r"\((\d+)\s+(?:rows|individual records)", entry)
    n_rows = int(m_rows.group(1)) if m_rows else 1
    r_part = entry.split("reason=", 1)[1] if "reason=" in entry else entry
    for prefix in (
        "vectraits_SE_arena_n_ambiguous",
        "mean_only",
        "SE_without_n",
        "non_time_unit",
        "range_only",
        "replicate_group_SD",
        "probable_error_without_n",
        "digitized_plot_means_without_error",
        "missing_or_nonpositive_mean_or_SD",
        "CI_without_n",
        "IQR_without_parametric_SD_or_n",
    ):
        if r_part.startswith(prefix):
            return prefix, n_rows
    if "Proportion/percentage miscoded" in r_part:
        return "non_time_unit", n_rows
    if "Non-positive duration" in r_part:
        return "missing_or_nonpositive_mean_or_SD", n_rows
    if "n < 10" in r_part:
        return "cohort_n_lt_10", n_rows
    return "other", n_rows


def parse_logs() -> dict[str, object]:
    """Parse exclusion entries and by-hand check entries from `extraction_log_part{1..4}.txt`."""
    exclusions_by_dataset: dict[str, list[str]] = defaultdict(list)
    hand_checks_by_dataset: dict[str, list[str]] = defaultdict(list)
    part_footer_lines: dict[int, str] = {}

    ds_prefix_map = {
        f"D{i}": ds for ds, _ in ALL_ASSIGNED_DATASETS for i in range(1, 17) if ds.startswith(f"D{i}_")
    }

    for part_idx, log_path in enumerate(PART_LOGS, start=1):
        text = log_path.read_text(encoding="utf-8")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith(f"Part {part_idx}:"):
                part_footer_lines[part_idx] = line
            # Part 1 exclusion format: "- EX-D1-..."
            m_ex1 = re.match(r"^-\s+(EX-(D\d+)[^\s:]*):?\s*(.*)$", line)
            if m_ex1:
                ds_key = ds_prefix_map[m_ex1.group(2)]
                exclusions_by_dataset[ds_key].append(f"{m_ex1.group(1)}: {m_ex1.group(3)}")
                continue
            # Parts 2-4 exclusion format: "[EXCLUSION] <dataset> | <item> | reason=<reason>"
            m_ex2 = re.match(r"^\[EXCLUSION\]\s+([^|]+?)\s*\|\s*(.+)$", line)
            if m_ex2:
                ds_key = m_ex2.group(1).strip()
                exclusions_by_dataset[ds_key].append(m_ex2.group(2).strip())
                continue
            # Part 1 hand-check format: "- HC-D1-1 ..."
            m_hc1 = re.match(r"^-\s+(HC-(D\d+)-\d+\s+.*)$", line)
            if m_hc1:
                ds_key = ds_prefix_map[m_hc1.group(2)]
                hand_checks_by_dataset[ds_key].append(m_hc1.group(1))
                continue
            # Parts 2-4 hand-check format: "[HAND CHECK D5-1] ..."
            m_hc2 = re.match(r"^\[HAND CHECK\s+(D\d+)-\d+\]\s+(.*)$", line)
            if m_hc2:
                ds_key = ds_prefix_map[m_hc2.group(1)]
                hand_checks_by_dataset[ds_key].append(line)
                continue

    d16_excl_entries: Counter[str] = Counter()
    d16_excl_rows: Counter[str] = Counter()
    for ent in exclusions_by_dataset.get("D16_vectraits_compilations", []):
        cat, nrows = classify_d16_exclusion(ent)
        d16_excl_entries[cat] += 1
        d16_excl_rows[cat] += nrows

    d6_excl_cats: Counter[str] = Counter()
    for ent in exclusions_by_dataset.get("D6_ixodes_scapularis_brunner2022", []):
        if "Ogden" in ent:
            d6_excl_cats["ogden_lab_molted_lt_10_or_0"] += 1
        elif "Step-jump" in ent:
            d6_excl_cats["field_step_jump_coarse_check_P2a"] += 1
        elif "not bracketed" in ent:
            d6_excl_cats["field_curve_not_bracketed"] += 1
        elif "Fewer than 4 distinct" in ent or "N < 50" in ent or "Fails Protocol §4.1 Rule 3" in ent:
            d6_excl_cats["field_insufficient_days_or_N_lt_50"] += 1
        else:
            d6_excl_cats["damaged_na_or_binary_recruitment"] += 1

    d5_row_tallies: Counter[str] = Counter()
    for ent in exclusions_by_dataset.get("D5_ixodes_fracasso2021", []):
        m_r = re.search(r"\((\d+)\s+rows", ent)
        cnt = int(m_r.group(1)) if m_r else 1
        if "MoultTime='n/a'" in ent:
            d5_row_tallies["moulttime_na_rows"] += cnt
        elif "MOULT_DAY entered as binary '1'" in ent:
            d5_row_tallies["moult_day_binary_1_rows"] += cnt
        elif "DREC-DINF > 8d" in ent or "POS='ocean'" in ent:
            d5_row_tallies["unknown_detachment_day_rows"] += cnt
        elif "Number_hatched_eggs.csv" in ent:
            d5_row_tallies["egg_clutch_excluded_rows"] += cnt
        elif "Missing DINF or DREC" in ent or "Missing BATCH" in ent:
            d5_row_tallies["missing_dates_or_batch_rows"] += cnt
        elif "n < 10" in ent:
            d5_row_tallies["cohort_n_lt_10_cells"] += 1
        elif "fasting_duration" in ent:
            d5_row_tallies["experimenter_set_fasting_trait"] += 1

    d11_tallies: Counter[str] = Counter()
    for ent in exclusions_by_dataset.get("D11_trematode_taskinen2022", []):
        m_lt = re.search(r"\((\d+)/\d+\s+rows with start_rc==0\)", ent)
        if m_lt:
            d11_tallies["left_truncated_start0_mussels"] += int(m_lt.group(1))
        elif "Cohort sample size n < 10" in ent:
            d11_tallies["cohort_n_lt_10_cells"] += 1

    return {
        "exclusions_by_dataset": dict(exclusions_by_dataset),
        "hand_checks_by_dataset": dict(hand_checks_by_dataset),
        "part_footers": part_footer_lines,
        "d16_exclusion_entries_by_reason": dict(sorted(d16_excl_entries.items())),
        "d16_exclusion_rows_by_reason": dict(sorted(d16_excl_rows.items())),
        "d6_exclusion_entries_by_reason": dict(sorted(d6_excl_cats.items())),
        "d5_exclusion_tallies": dict(sorted(d5_row_tallies.items())),
        "d11_exclusion_tallies": dict(sorted(d11_tallies.items())),
    }


def verify_hand_checks_in_merged_csv(rows: list[dict[str, str]]) -> list[str]:
    """Cross-check specific hand-checked cohort values from Parts 1-4 against `stage_cv.csv`."""
    expected_spots = [
        ("D1_malaria_embo2024", "children 7-9 yr", 381, 183.095059, 239.278255, 1.306853),
        ("D1_malaria_embo2024", "children 10-13 yr", 471, 126.397128, 150.056493, 1.187183),
        ("D1_malaria_embo2024", "msp2 capillary", 332, 27.902102, 20.461123, 0.733318),
        ("D3_anopheles_pf_shapiro2017", "Block 1", 374, 13.929685, 1.766766, 0.126835),
        ("D3_anopheles_pf_shapiro2017", "Block 2", 356, 15.599622, 3.041896, 0.194998),
        ("D3_anopheles_pf_shapiro2017", "Block 2", 392, 11.651171, 1.117978, 0.095954),
        ("D4_plasmodium_msos2021", "uninfected bloodmeal", 191, 9.178010, 3.081308, 0.336167),
        ("D4_plasmodium_msos2021", "P. falciparum-exposed", 273, 12.430010, 4.151830, 0.334017),
        ("D4_plasmodium_msos2021", "Shapiro_L_Thomas", 1281, 10.821915, 0.419611, 0.038774),
        ("D5_ixodes_fracasso2021", "nymph_feeding_batch_11_captive_20C", 17, 4.470588, 0.799816, 0.181537),
        ("D5_ixodes_fracasso2021", "nymph_to_adult_moulting_batch_11_vial_20C", 15, 21.133333, 1.959106, 0.094247),
        ("D5_ixodes_fracasso2021", "nymph_to_adult_moulting_batch_6_vial_20C", 22, 24.000000, 2.093072, 0.088202),
        ("D6_ixodes_scapularis_brunner2022", "larva_to_nymph_moulting_lab_Ogden_24C", 31, 27.500000, 3.340659, 0.122458),
        ("D6_ixodes_scapularis_brunner2022", "nymph_to_adult_moulting_lab_Ogden_28C", 17, 29.000000, 4.123106, 0.144267),
        ("D6_ixodes_scapularis_brunner2022", "fed_larvae_CL_2019-08-13_moulting", 144, 48.253615, 7.983650, 0.165452),
        ("D7_schisto_leclech2019", "SmLE", 46, 19.525762, 6.512795, 0.333549),
        ("D8_schisto_civitello2020", "FoodLevel=0.6875", 10, 64.400000, 38.169796, 0.607516),
        ("D8_schisto_civitello2020", "FoodLevel=1.375", 12, 86.916667, 29.314775, 0.344301),
        ("D8_schisto_civitello2020", "FoodLevel=2.75", 12, 115.133807, 60.525933, 0.525701),
        ("D9_schisto_thermal_nguyen2021", "11 timepoints", None, 0.907534, 0.257379, 0.283603),
        ("D9_schisto_thermal_nguyen2021", "10 timepoints", None, 0.740578, 0.197914, 0.267243),
        ("D9_schisto_thermal_nguyen2021", "7 timepoints", None, 0.457351, 0.194378, 0.425009),
        ("D11_trematode_taskinen2022", "Rhipidocotyle campanula", 10, 67.400000, 30.746996, 0.467592),
        ("D11_trematode_taskinen2022", "Haajaistenjoki_intermediate", 19, 44.657895, 23.197852, 0.526292),
        ("D11_trematode_taskinen2022", "Kuusaankoski_intermediate", 13, 51.662960, 18.225022, 0.352768),
        ("D12_trematode_gleichsner2016", "stressed", 56, 31.625000, 3.773412, 0.119850),
        ("D12_trematode_gleichsner2016", "unstressed", 25, 32.200000, 4.518481, 0.141729),
        ("D13_nematode_marshallagia2023", "egg_to_L1_hatching_time_Colorado_constant_lab", 35, 6.471429, 1.580873, 0.246030),
        ("D13_nematode_marshallagia2023", "egg_to_L3_development_time_Dawson_constant_lab", 39, 9.628205, 1.379922, 0.144239),
        ("D13_nematode_marshallagia2023", "pre_L3_egg_larval_survival_time_Dawson_constant_lab", 40, 8.450659, 3.042007, 0.359973),
        ("D16_vectraits_compilations", "vectraits_0553_gonotrophic_cycle_1_adult", 29, 10.530690, 3.040177, 0.291186),
        ("D16_vectraits_compilations", "vectraits_0726_premolt_period_(nymphal)_nymphal_male_id108805", 216, 131.500000, 7.230000, 0.055045),
        ("D16_vectraits_compilations", "vectraits_0069_longevity_adult_female_id82079", None, 19.450000, 11.110000, 0.571208),
    ]

    verified = []
    for ds, substr, exp_n, exp_mean, exp_sd, exp_cv in expected_spots:
        matches = [
            r
            for r in rows
            if r["dataset"] == ds
            and (substr in r["stage"] or substr in r["taxon"])
            and ((exp_n is None and r["n"] == "") or (exp_n is not None and r["n"] == str(exp_n)))
            and abs(float(r["mean"]) - exp_mean) < 2e-6
        ]
        assert len(matches) == 1, f"Hand-check match failed for {ds} / {substr}: {len(matches)} matches"
        m = matches[0]
        assert abs(float(m["sd"]) - exp_sd) < 2e-6
        assert abs(float(m["cv_indiv"]) - exp_cv) < 2e-6
        verified.append(f"{ds} | {m['stage']} | n={m['n'] or 'NA'} | mean={m['mean']} | sd={m['sd']} | cv={m['cv_indiv']}")
    return verified


def cv_quantiles(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=float)
    q10, q25, q50, q75, q90 = np.percentile(arr, [10.0, 25.0, 50.0, 75.0, 90.0])
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "q10": round(float(q10), 6),
        "q25": round(float(q25), 6),
        "median": round(float(q50), 6),
        "q75": round(float(q75), 6),
        "q90": round(float(q90), 6),
        "max": round(float(np.max(arr)), 6),
    }


def main() -> None:
    report_lines = run_known_self_checks()

    headers: list[list[str]] = []
    tables: list[list[dict[str, str]]] = []
    part_counts: dict[str, int] = {}

    for idx, path in enumerate(PART_CSVS, start=1):
        if not path.exists():
            raise FileNotFoundError(f"Missing part CSV: {path}")
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV header in {path}")
            hdr = list(reader.fieldnames)
            rows = list(reader)
        headers.append(hdr)
        tables.append(rows)
        part_counts[f"part{idx}"] = len(rows)

    merged_header, merged_rows, dropped_exact = merge_row_tables(
        headers, tables, expected_columns=EXPECTED_COLUMNS
    )
    total_before_dedup = sum(part_counts.values())

    # Row-level integrity checks on merged table
    key_counter: Counter[tuple[str, str, str, str]] = Counter()
    for r_idx, r in enumerate(merged_rows, start=1):
        if r["stage_category"] not in STAGE_CATEGORIES:
            raise ValueError(f"Row {r_idx} invalid stage_category: {r['stage_category']}")
        for num_col in ("mean", "sd", "cv_indiv"):
            val = float(r[num_col])
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(f"Row {r_idx} non-positive/non-finite {num_col}: {r[num_col]}")
        if r["n"] != "":
            n_val = int(r["n"])
            if n_val < 10:
                raise ValueError(f"Row {r_idx} has n={n_val} < 10")
        key_counter[(r["dataset"], r["taxon"], r["stage"], r["temperature_C"])] += 1

    duplicate_natural_keys = {
        " | ".join(k): c for k, c in key_counter.items() if c > 1
    }

    # Write results/stage_cv.csv (single header, consistent columns, exact duplicates dropped)
    with MERGED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_header)
        writer.writeheader()
        writer.writerows(merged_rows)

    # Parse logs and verify hand checks against merged CSV
    log_info = parse_logs()
    exclusions_by_ds: dict[str, list[str]] = log_info["exclusions_by_dataset"]  # type: ignore[assignment]
    hand_checks_by_ds: dict[str, list[str]] = log_info["hand_checks_by_dataset"]  # type: ignore[assignment]
    verified_spots = verify_hand_checks_in_merged_csv(merged_rows)

    # Aggregate statistics by dataset, stage category, rule, and strict-n status
    rows_by_ds: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    rule_counts: Counter[str] = Counter()
    d16_source_files: set[str] = set()
    eip_step_cap_count = 0

    for r in merged_rows:
        rows_by_ds[r["dataset"]].append(r)
        rows_by_cat[r["stage_category"]].append(r)
        rule_counts[classify_rule(r["rule_applied"])] += 1
        if r["dataset"] == "D16_vectraits_compilations":
            d16_source_files.add(r["source_file"])
        if "daily-step identifiability cap" in r["rule_applied"]:
            eip_step_cap_count += 1

    dataset_summary: dict[str, dict[str, object]] = {}
    zero_row_datasets: list[str] = []
    for ds, part_no in ALL_ASSIGNED_DATASETS:
        drows = rows_by_ds.get(ds, [])
        strict_rows = [r for r in drows if r["n"] != ""]
        no_n_rows = [r for r in drows if r["n"] == ""]
        cats = Counter(r["stage_category"] for r in drows)
        rules = Counter(classify_rule(r["rule_applied"]) for r in drows)
        if len(drows) == 0:
            zero_row_datasets.append(ds)
        dataset_summary[ds] = {
            "part": part_no,
            "total_rows": len(drows),
            "strict_known_n_rows": len(strict_rows),
            "unstated_n_rows": len(no_n_rows),
            "stage_categories": dict(cats),
            "rules": dict(rules),
            "cv_all": cv_quantiles([float(r["cv_indiv"]) for r in drows]),
            "cv_strict_n": cv_quantiles([float(r["cv_indiv"]) for r in strict_rows]),
            "logged_exclusion_entries": len(exclusions_by_ds.get(ds, [])),
            "logged_hand_checks": len(hand_checks_by_ds.get(ds, [])),
        }

    category_summary: dict[str, dict[str, object]] = {}
    for cat in STAGE_CATEGORIES:
        crows = rows_by_cat.get(cat, [])
        strict_rows = [r for r in crows if r["n"] != ""]
        no_n_rows = [r for r in crows if r["n"] == ""]
        ds_counts = Counter(r["dataset"] for r in crows)
        category_summary[cat] = {
            "total_rows": len(crows),
            "strict_known_n_rows": len(strict_rows),
            "unstated_n_rows": len(no_n_rows),
            "datasets": dict(sorted(ds_counts.items())),
            "cv_all": cv_quantiles([float(r["cv_indiv"]) for r in crows]),
            "cv_strict_n": cv_quantiles([float(r["cv_indiv"]) for r in strict_rows]),
        }

    strict_all = [float(r["cv_indiv"]) for r in merged_rows if r["n"] != ""]
    unstated_all = [float(r["cv_indiv"]) for r in merged_rows if r["n"] == ""]
    all_cvs = [float(r["cv_indiv"]) for r in merged_rows]
    cv_gt_1_rows = [
        {
            "dataset": r["dataset"],
            "taxon": r["taxon"],
            "stage": r["stage"],
            "n": r["n"],
            "cv_indiv": float(r["cv_indiv"]),
        }
        for r in merged_rows
        if float(r["cv_indiv"]) > 1.0
    ]

    total_exclusion_entries = sum(len(v) for v in exclusions_by_ds.values())
    total_hand_checks = sum(len(v) for v in hand_checks_by_ds.values())

    summary_payload = {
        "columns": merged_header,
        "column_count": len(merged_header),
        "part_row_counts": part_counts,
        "total_rows_before_dedup": total_before_dedup,
        "exact_duplicates_dropped": dropped_exact,
        "merged_total_rows": len(merged_rows),
        "strict_known_n_rows": len(strict_all),
        "unstated_n_rows": len(unstated_all),
        "duplicate_natural_keys": duplicate_natural_keys,
        "assigned_datasets_total": len(ALL_ASSIGNED_DATASETS),
        "datasets_with_rows": len(ALL_ASSIGNED_DATASETS) - len(zero_row_datasets),
        "zero_row_datasets": zero_row_datasets,
        "d16_distinct_source_files_with_rows": len(d16_source_files),
        "eip_daily_step_cap_rows": eip_step_cap_count,
        "cv_gt_1_count": len(cv_gt_1_rows),
        "cv_gt_1_rows": cv_gt_1_rows,
        "overall_cv_all": cv_quantiles(all_cvs),
        "overall_cv_strict_n": cv_quantiles(strict_all),
        "rule_breakdown": dict(sorted(rule_counts.items())),
        "dataset_summary": dataset_summary,
        "stage_category_summary": category_summary,
        "logged_exclusion_entry_count": total_exclusion_entries,
        "d16_exclusion_entries_by_reason": log_info["d16_exclusion_entries_by_reason"],
        "d16_exclusion_rows_by_reason": log_info["d16_exclusion_rows_by_reason"],
        "d6_exclusion_entries_by_reason": log_info["d6_exclusion_entries_by_reason"],
        "d5_exclusion_tallies": log_info["d5_exclusion_tallies"],
        "d11_exclusion_tallies": log_info["d11_exclusion_tallies"],
        "logged_hand_check_count": total_hand_checks,
        "verified_csv_hand_check_rows": len(verified_spots),
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    n_d16_sd_no_n = rule_counts["Rule 4: reported_SD_no_n (unstated n)"]
    n_d9_no_n = rule_counts["Rule 3: interval_transition_curve (Weibull survival MLE, no_indiv_n)"]

    # Human-readable audit report
    report_lines.append("=== Merge Audit (`results/stage_cv.csv`) ===")
    report_lines.append(f"Columns ({len(merged_header)}): {', '.join(merged_header)}")
    for idx in (1, 2, 3, 4):
        report_lines.append(f"  part{idx}: {part_counts[f'part{idx}']} rows")
    report_lines.append(f"Total rows across parts 1-4 before deduplication: {total_before_dedup}")
    report_lines.append(f"Exact duplicate rows dropped: {dropped_exact}")
    report_lines.append(f"Final merged rows written to results/stage_cv.csv: {len(merged_rows)}")
    report_lines.append(
        f"  - Strict known-n (n >= 10) cohorts: {len(strict_all)} "
        f"(median CV_indiv = {summary_payload['overall_cv_strict_n']['median']:.4f}, "
        f"IQR = [{summary_payload['overall_cv_strict_n']['q25']:.4f}, {summary_payload['overall_cv_strict_n']['q75']:.4f}], "
        f"range = [{summary_payload['overall_cv_strict_n']['min']:.4f}, {summary_payload['overall_cv_strict_n']['max']:.4f}])"
    )
    report_lines.append(
        f"  - Unstated-n compilation/curve cohorts: {len(unstated_all)} "
        f"({n_d16_sd_no_n} D16 reported_SD_no_n + {n_d9_no_n} D9 digitized % survival curves without indiv n; "
        f"all-row median CV_indiv = {summary_payload['overall_cv_all']['median']:.4f}, "
        f"IQR = [{summary_payload['overall_cv_all']['q25']:.4f}, {summary_payload['overall_cv_all']['q75']:.4f}])"
    )
    report_lines.append(
        f"  - D16 distinct VecTraits JSON datasets contributing rows: {len(d16_source_files)} / 175"
    )
    report_lines.append(
        f"  - Cross-sectional EIP rows at daily check-spacing cap (log_k=6.5, CV=0.038774): {eip_step_cap_count}"
    )
    report_lines.append(
        f"  - Overdispersed cohorts with CV_indiv > 1.0: {len(cv_gt_1_rows)}"
    )
    report_lines.append("")

    report_lines.append("=== Counts by Dataset (D1-D16) ===")
    for ds, part_no in ALL_ASSIGNED_DATASETS:
        info = dataset_summary[ds]
        cv_info = info["cv_all"]  # type: ignore[assignment]
        med_str = f"{cv_info['median']:.4f} [{cv_info['min']:.4f}..{cv_info['max']:.4f}]" if info["total_rows"] else "NA (0 rows)"
        cats_str = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(info["stage_categories"].items())) or "-"
        report_lines.append(
            f"  {ds:36s} (Part {part_no}): total={info['total_rows']:3d} "
            f"(strict_n={info['strict_known_n_rows']:3d}, no_n={info['unstated_n_rows']:2d}) | "
            f"cats={cats_str:14s} | median[min..max]={med_str} | "
            f"excl_entries={info['logged_exclusion_entries']:3d} | hand_checks={info['logged_hand_checks']}"
        )
    report_lines.append("")

    report_lines.append("=== Counts by Stage Category (S1-S5) ===")
    for cat in STAGE_CATEGORIES:
        cinfo = category_summary[cat]
        ca = cinfo["cv_all"]  # type: ignore[assignment]
        cs = cinfo["cv_strict_n"]  # type: ignore[assignment]
        ds_str = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in cinfo["datasets"].items())  # type: ignore[union-attr]
        report_lines.append(
            f"  {cat:45s}: total={cinfo['total_rows']:3d} "
            f"(strict_n={cinfo['strict_known_n_rows']:3d}, no_n={cinfo['unstated_n_rows']:2d}) | "
            f"all_med={ca['median']:.4f} [{ca['q25']:.4f}, {ca['q75']:.4f}] | "
            f"strict_med={cs['median']:.4f} [{cs['q25']:.4f}, {cs['q75']:.4f}] | {ds_str}"
        )
    report_lines.append("")

    report_lines.append("=== Counts by Estimator / Dispersion Rule ===")
    for rname, cnt in sorted(rule_counts.items()):
        report_lines.append(f"  {rname:68s}: {cnt:3d}")
    report_lines.append("")

    report_lines.append("=== Zero-Row Assigned Datasets ===")
    for ds in zero_row_datasets:
        report_lines.append(f"  - {ds} ({len(exclusions_by_ds.get(ds, []))} logged exclusion entries)")
    report_lines.append("")

    report_lines.append("=== By-Hand Checks Cross-Verified Against `results/stage_cv.csv` ===")
    report_lines.append(
        f"Total hand checks logged across D1-D16: {total_hand_checks} (3 per dataset x 16 datasets); "
        f"extracted-cohort spot checks verified against stage_cv.csv: {len(verified_spots)} / {len(verified_spots)} PASS"
    )
    report_lines.append(f"Total logged exclusion entries across D1-D16: {total_exclusion_entries}")
    report_lines.append("  D16 VecTraits exclusion breakdown (entries / rows):")
    d16_ent_map = log_info["d16_exclusion_entries_by_reason"]  # type: ignore[assignment]
    d16_row_map = log_info["d16_exclusion_rows_by_reason"]  # type: ignore[assignment]
    for rk in sorted(d16_ent_map.keys()):
        report_lines.append(f"    - {rk:38s}: {d16_ent_map[rk]:3d} entries ({d16_row_map[rk]:4d} rows)")
    report_lines.append(
        f"    Total D16 excluded rows across {sum(d16_ent_map.values())} entries: {sum(d16_row_map.values())} rows"
    )
    report_lines.append("  D6 Ixodes scapularis exclusion breakdown (entries):")
    d6_ent_map = log_info["d6_exclusion_entries_by_reason"]  # type: ignore[assignment]
    for rk in sorted(d6_ent_map.keys()):
        report_lines.append(f"    - {rk:38s}: {d6_ent_map[rk]:3d} entries")
    report_lines.append("  D5 Ixodes arboricola exclusion tallies:")
    d5_map = log_info["d5_exclusion_tallies"]  # type: ignore[assignment]
    for rk in sorted(d5_map.keys()):
        report_lines.append(f"    - {rk:38s}: {d5_map[rk]:4d}")
    report_lines.append("  D11 Rhipidocotyle exclusion tallies:")
    d11_map = log_info["d11_exclusion_tallies"]  # type: ignore[assignment]
    for rk in sorted(d11_map.keys()):
        report_lines.append(f"    - {rk:38s}: {d11_map[rk]:4d}")

    out_txt = "\n".join(report_lines) + "\n"
    SUMMARY_TXT.write_text(out_txt, encoding="utf-8")
    print(out_txt, end="")


if __name__ == "__main__":
    main()
