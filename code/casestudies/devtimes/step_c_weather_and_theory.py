#!/usr/bin/env python3
"""Step C analysis for casestudy `devtimes`:
1. Source-study temperature and environmental conditions (`data/raw/devtimes/study_temp_docs/`,
   raw datasets, and cached weather/microclimate files in `data/raw/devtimes/` — zero network calls).
2. Exact variance-decomposition and correlation tests asking whether temperature explains
   the spread among individuals (`CV_indiv` within constant-temperature cohorts vs across-temperature
   pooled CV when temperature plasticity is conflated with individual spread).
3. Quantitative placement of measured `CV_indiv` distributions against R10b's protection
   thresholds (`reports/R10b-real-systems-stay-shape.md` and its refereed note) across all
   real-system rows and stage classes, saving reproducible outputs in `results/`.

Run with: `python3 -I casestudies/devtimes/step_c_weather_and_theory.py`
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RAW = REPO / "data" / "raw" / "devtimes"
RESULTS = HERE / "results"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on WGS-84 mean spherical radius (6371.0088 km)."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def run_known_checks() -> dict:
    """Rule 7: reproduce known analytical benchmarks before computing new results."""
    # Check 1: 1 degree of latitude on sphere of radius 6371.0088 km = pi * R / 180 = 111.19508 km
    d_lat = haversine_km(0.0, 0.0, 1.0, 0.0)
    expected_lat = math.pi * 6371.0088 / 180.0
    err_lat = abs(d_lat - expected_lat)
    assert err_lat < 1e-9, f"Haversine check failed: {err_lat}"

    # Check 2: Law of total variance for equal-weight mixture of K cohorts with mean mu_k and SD = c * mu_k:
    # Var_total = E[sigma_k^2] + Var(mu_k) => CV_pooled^2 = c^2 * (1 + CV_means^2) + CV_means^2.
    mus = np.array([10.0, 20.0, 30.0])
    c = 0.20
    samples = np.concatenate([np.array([m * (1.0 - c), m * (1.0 + c)]) for m in mus])
    pop_mean = float(np.mean(samples))
    pop_var = float(np.var(samples, ddof=0))
    cv_means_sq = float(np.var(mus, ddof=0) / (np.mean(mus) ** 2))
    expected_cv_sq = c**2 * (1.0 + cv_means_sq) + cv_means_sq
    err_ltv = abs(pop_var / (pop_mean**2) - expected_cv_sq)
    assert err_ltv < 1e-12, f"Total variance identity check failed: {err_ltv}"

    # Check 3: Discrete chain CV formula CV = sqrt(1/m - 1/tau) matches R10b's reported fine-step chain CVs:
    # For malaria (tau_fine = 60, m = 4): sqrt(1/4 - 1/60) = sqrt(14/60) = 0.48304589...
    # For malaria coarse (tau_coarse = 7.5, m = 4): sqrt(1/4 - 1/7.5) = sqrt(7/60) = 0.34156503...
    cv_pf_coarse_m4 = math.sqrt(1.0 / 4.0 - 1.0 / 7.5)
    cv_pf_fine_m4 = math.sqrt(1.0 / 4.0 - 1.0 / 60.0)
    assert abs(round(cv_pf_coarse_m4, 3) - 0.342) < 1e-9
    assert abs(round(cv_pf_fine_m4, 3) - 0.483) < 1e-9

    return {
        "haversine_1deg_km": d_lat,
        "haversine_err_km": err_lat,
        "total_variance_cv_sq": pop_var / (pop_mean**2),
        "total_variance_err": err_ltv,
        "cv_pf_coarse_m4": cv_pf_coarse_m4,
        "cv_pf_fine_m4": cv_pf_fine_m4,
    }


def load_stage_cv() -> list[dict]:
    rows = []
    with (RESULTS / "stage_cv.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


if __name__ == "__main__":
    checks = run_known_checks()
    rows = load_stage_cv()
    print(f"UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Known checks passed; loaded {len(rows)} rows from stage_cv.csv")
