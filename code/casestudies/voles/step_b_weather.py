"""Step B(i): 28-day weather covariates aligned to host windows, plus B1 and B2."""

from __future__ import annotations

import json
from pathlib import Path

STUDY_DIR = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
WEATHER_RAW_DIR = REPO_ROOT / "data" / "raw" / "voles" / "weather"
RESULTS_DIR = STUDY_DIR / "results"


def build_weather_covariates():
    """Build 28-day weather covariates aligned to the host windows."""
    raise NotImplementedError("Skeleton: to be implemented after reading STEP_B_REGISTERED.md")


def b1(covariates_df):
    """Run B1: host-weather covariation analysis."""
    raise NotImplementedError("Skeleton: to be implemented after reading STEP_B_REGISTERED.md")


def b2(covariates_df):
    """Run B2: T1_mem re-evaluated with weather covariates."""
    raise NotImplementedError("Skeleton: to be implemented after reading STEP_B_REGISTERED.md")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cov_df = build_weather_covariates()
    cov_df.to_csv(RESULTS_DIR / "weather_covariates.csv", index=False)
    res_b1 = b1(cov_df)
    res_b2 = b2(cov_df)
    out = {"B1": res_b1, "B2": res_b2}
    (RESULTS_DIR / "weather_b1b2.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
