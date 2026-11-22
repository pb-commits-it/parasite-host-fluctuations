#!/usr/bin/env python3
"""Amendment 3: describe laboratory regimes and D4 temperature associations.

Check: D4 temperature assignments match the original groups, whose saved means
are reproduced to 1e-10 before new descriptive rank correlations are used.
No causal or new T1 test, thermal-rate model, simulation, or outdoor weather join.
Run: python3 -I casestudies/chemostat/weather_environment.py
Verify append-only changes: same command with --verify-originals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import warnings

import openpyxl
from scipy.stats import spearmanr

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RESULTS = HERE / "results"
RAW = ROOT / "data/raw/chemostat"


def utc_now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_integrity() -> dict:
    original = json.loads((RESULTS / "weather_original_files.json").read_text())
    records = {}
    for name, before in original["files"].items():
        content = (HERE / name).read_bytes()
        append_only = name in {"protocol.md", "report.md"}
        checked = content[:before["bytes"]] if append_only else content
        assert hashlib.sha256(checked).hexdigest() == before["sha256"], name
        records[name] = "original prefix intact" if append_only else "unchanged"
    return {"checked_utc_from_date": utc_now(), "all_passed": True, "files": records}


def average_ranks(values: list[float]) -> list[float]:
    """Independent rank calculation, not scipy.stats.rankdata."""
    return [1 + sum(w < v for w in values) + (sum(w == v for w in values) - 1) / 2
            for v in values]


def rank_correlation(x: list[float], y: list[float]) -> dict:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "reason": "constant covariate or outcome"}
    rho = float(spearmanr(x, y).statistic)
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    independent = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(
        math.fsum((a - mx) ** 2 for a in rx) * math.fsum((b - my) ** 2 for b in ry))
    assert abs(rho - independent) <= 1e-12
    return {"n": len(x), "rho": rho, "independent_error": abs(rho - independent)}


def known_checks() -> dict:
    # For x ranks (1, 2.5, 2.5, 4) and y ranks (1, 2, 3, 4),
    # covariance numerator is 4.5; the squared denominator is 4.5 * 5.
    tied = rank_correlation([1, 2, 2, 4], [1, 2, 3, 4])["rho"]
    assert abs(tied - math.sqrt(0.9)) <= 1e-12
    assert abs(rank_correlation([1, 2, 3], [3, 2, 1])["rho"] + 1) <= 1e-12
    assert rank_correlation([1, 1, 1], [1, 2, 3])["rho"] is None
    return {"passed": True, "tolerance": 1e-12, "tied_example_rho": tied}


def load_assignments() -> tuple[dict, dict]:
    path = RAW / "duncan2017_sbw25_temp_fluctuation/Duncan_JEB.xlsx"
    # Read-only; unsupported Excel formatting is irrelevant to cell values.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown extension is not supported.*")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook["Bacteria_Phage_Density"]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        assert header[:5] == ("Population", "Fluctuation frequency", "Start temperature",
                              "Phage", "Time (days)")
        assignments = {}
        for row in rows:
            if row[0] is None:
                continue
            pop, freq, start, infection, day = row[:5]
            key = f"Pop_{int(pop)}"
            values = (int(freq), int(start), infection)
            entry = assignments.setdefault(key, {"values": values, "days": []})
            assert entry["values"] == values
            entry["days"].append(float(day))
        workbook.close()
    return assignments, {"file": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-originals", action="store_true")
    args = parser.parse_args()
    if args.verify_originals:
        result = original_integrity()
        save_json(RESULTS / "weather_integrity.json", result)
        print(f"PASS: {len(result['files'])} original files checked; report/protocol prefixes intact.")
        return

    lines = []
    try:
        known = known_checks()  # Before reading empirical outcomes.
        lines.append("Known rank benchmarks PASS; tolerance 1e-12.")
        original_integrity()
        sources_path = HERE / "weather_sources.json"
        inventory = json.loads(sources_path.read_text())
        source_checks = []
        for paper in inventory["papers"]:
            for source in paper["sources"]:
                path = ROOT / source["path"]
                assert path.is_relative_to(RAW)
                assert digest(path) == source["sha256"], path.name
                source_checks.append({"file": source["path"], "sha256": digest(path),
                                      "status": "supplied cache verified; no new download"})

        assignments, workbook_source = load_assignments()
        with (RESULTS / "unit_metrics.csv").open() as handle:
            all_units = list(csv.DictReader(handle))
        d4 = [r for r in all_units if r["dataset_id"] == "D4_duncan2017"]
        assert len(d4) == 30
        assert len({r["unit_id"] for r in d4}) == 30
        assert {r["unit_id"] for r in d4} == {
            k for k, v in assignments.items() if v["values"][2] == "Infected"}
        saved_sensitivity = json.loads((RESULTS / "sensitivity_summary.json").read_text())
        prior_d4 = saved_sensitivity["S4_parameter_sweeps"]["D4_thermal_confounding"]
        baseline_errors = []
        for treatment, saved in prior_d4["by_treatment"].items():
            subset = [r for r in d4 if r["treatment"] == treatment]
            assert len(subset) == saved["n"]
            for field in ["sigma_lnH", "y_mean_lnP"]:
                error = abs(statistics.mean(float(r[field]) for r in subset) - saved[field + "_mean"])
                assert error <= 1e-10
                baseline_errors.append(error)
            assert sum(int(r["went_extinct"]) for r in subset) == saved["extinctions"]
        lines.append("Original D4 means/counts reproduced for all 8 start-temperature groups; tolerance 1e-10.")

        annotated = []
        for row in d4:
            entry = assignments[row["unit_id"]]
            freq, start, infection = entry["values"]
            assert freq in {0, 2, 4, 8} and start in {28, 32}
            assert infection == "Infected"
            assert row["treatment"] == f"freq{freq}_T{start}"
            assert min(entry["days"]) == 0 and max(entry["days"]) == 16
            assert all(day in range(0, 17, 2) for day in entry["days"])
            # Reconstruct full programmed exposure, not the temperature at a sample.
            schedule = [start if freq == 0 or (day // freq) % 2 == 0 else 60 - start
                        for day in range(0, 16, 2)]
            if freq:
                assert schedule.count(28) == schedule.count(32) == 4
            annotated.append({
                "unit_id": row["unit_id"], "original_treatment": row["treatment"],
                "group": f"constant_{start}C" if freq == 0 else f"switch_{freq}d",
                "start_temperature_C": start, "switch_interval_d": freq,
                "assigned_temperature_mean_C": statistics.mean(schedule),
                "assigned_temperature_sd_C": statistics.pstdev(schedule),
                "assigned_2day_intervals_0_to_16d_C": ";".join(map(str, schedule)),
                "sigma_lnH": float(row["sigma_lnH"]),
                "y_mean_lnP": float(row["y_mean_lnP"]),
                "went_extinct": int(row["went_extinct"]),
            })

        groups = []
        for name in ["constant_28C", "constant_32C", "switch_2d", "switch_4d", "switch_8d"]:
            subset = [r for r in annotated if r["group"] == name]
            group = {"group": name, "n": len(subset)}
            for field in ["sigma_lnH", "y_mean_lnP"]:
                values = [r[field] for r in subset]
                group[field + "_mean"] = statistics.mean(values)
                group[field + "_range"] = [min(values), max(values)]
            group["final_zero_count"] = sum(r["went_extinct"] for r in subset)
            groups.append(group)

        correlations = []
        for covariate, subset in [
            ("assigned_temperature_mean_C", annotated),
            ("assigned_temperature_sd_C", annotated),
            ("switch_interval_d", [r for r in annotated if r["switch_interval_d"] > 0]),
        ]:
            for outcome in ["sigma_lnH", "y_mean_lnP", "went_extinct"]:
                stat = rank_correlation([r[covariate] for r in subset], [r[outcome] for r in subset])
                correlations.append({"covariate": covariate, "outcome": outcome, **stat})

        saved_tests = json.loads((RESULTS / "dataset_tests.json").read_text())
        units_by_dataset = {}
        for paper in inventory["papers"]:
            ds = paper["dataset_id"]
            count = sum(row["dataset_id"].startswith(ds + "_") for row in all_units)
            units_by_dataset[ds] = count
            lines.append(f"{ds}: {count} infected density trajectories; {paper['temperature_regime']} "
                         f"Light: {paper['light_regime']} Time: {paper['experimental_time_coverage']}")

        output = {
            "run_utc_from_date": utc_now(), "amendment": 3,
            "interpretation": "Descriptive associations only, no p-values or new T1 verdict.",
            "field_record_count": 0, "outdoor_weather_matches": 0,
            "new_network_requests": 0, "source_files_checked": source_checks,
            "assignment_workbook": workbook_source, "known_rank_checks": known,
            "original_group_mean_max_error": max(baseline_errors),
            "original_group_counts_match": True, "assignment_matches": len(annotated),
            "density_trajectories_by_dataset": units_by_dataset,
            "d4_groups": groups, "d4_correlations": correlations,
            "original_d4_T1": saved_tests["datasets"]["D4_duncan2017"]["T1_primary_y_mean"],
            "original_d4_sensitivities": prior_d4,
            "original_global_T2a": saved_tests["global_T2a"],
            "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [
                sources_path, RESULTS / "unit_metrics.csv", RESULTS / "sensitivity_summary.json",
                RESULTS / "dataset_tests.json", HERE / "protocol.md", Path(__file__).resolve()]},
        }
        save_csv(RESULTS / "weather_d4_units.csv", annotated)
        save_csv(RESULTS / "weather_lab_inventory.csv", [{
            key: p[key] for key in ["dataset_id", "constant_culture_temperature_C",
                                   "temperature_regime", "light_regime", "experimental_time_coverage",
                                   "temporal_resolution", "environment_context",
                                   "spatial_match_reason", "measurement_uncertainty"]
        } for p in inventory["papers"]])
        save_json(RESULTS / "weather_summary.json", output)
        lines.append(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        lines.append("STOP: one descriptive run completed; original analysis not rerun.")
        text = "\n".join(lines) + "\n"
        (RESULTS / "weather_output.txt").write_text(text)
        print(text)
    except Exception as exc:
        suffix = len(list(RESULTS.glob("weather_attempt*_failure.txt"))) + 1
        (RESULTS / f"weather_attempt{suffix}_failure.txt").write_text(
            utc_now() + "\n" + "\n".join(lines) + f"\n{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
