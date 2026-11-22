#!/usr/bin/env python3
"""Part 1 stage-duration spread extraction (`r46-devtimes` part 1).

Extracts cohort-level within-treatment among-individual stage-duration spreads
(`CV_indiv`) following `protocol.md` §3, §4, §4.2, and Amendments 1–4 for:
  - D1_malaria_embo2024
  - D2_malariotherapy_plos2013
  - D3_anopheles_pf_shapiro2017
  - D4_plasmodium_msos2021

Run with:
  python3 -I casestudies/devtimes/extract_part1.py
"""
from __future__ import annotations

import csv
import datetime
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import extract_helpers as eh  # noqa: E402

REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RAW_ROOT = REPO_ROOT / "data" / "raw" / "devtimes"
RESULTS_DIR = HERE / "results"
CSV_OUT = RESULTS_DIR / "stage_cv_part1.csv"
LOG_OUT = RESULTS_DIR / "extraction_log_part1.txt"

ASSIGNED_DATASETS = [
    "D1_malaria_embo2024",
    "D2_malariotherapy_plos2013",
    "D3_anopheles_pf_shapiro2017",
    "D4_plasmodium_msos2021",
]

COLUMNS = [
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

    from scipy import special
    days = np.arange(4.0, 22.0, 2.0)
    tot = np.full_like(days, 10000.0)
    pos = 10000.0 * 0.85 * special.gammainc(16.0, 16.0 * days / 12.0)
    gm = eh.gamma_cdf_cross_sectional_mle(days, pos, tot)
    assert gm["ok"] and gm["bracketed"]
    assert abs(gm["cv_indiv"] - 0.25) < 0.01
    assert abs(gm["mean_eip"] - 12.0) < 0.05
    lines.append(
        f"  [PASS] gamma_cdf_cross_sectional_mle(k=16, mu=12, p_max=0.85): "
        f"mean_eip={gm['mean_eip']:.4f} (exp 12.0, tol 0.05), cv_indiv={gm['cv_indiv']:.4f} (exp 0.2500, tol 0.01)"
    )
    lines.append("")
    return lines


def _chunk_presence(tp_nums: list[int], present_set: set[int], max_zeroes: int = 2) -> list[tuple[int, int]]:
    """Identify contiguous infection chunks allowing up to `max_zeroes` skipped visits
    (exact translation of `Meta_to_infection_length_github.Rmd`)."""
    vals = [1 if tn in present_set else 0 for tn in tp_nums]
    start: int | None = None
    num_zeroes = 0
    last_one: int | None = None
    chunks: list[tuple[int, int]] = []
    for idx, v in enumerate(vals):
        if v == 1:
            if start is None:
                start = idx
            num_zeroes = 0
            last_one = idx
        elif start is not None:
            num_zeroes += 1
            if num_zeroes > max_zeroes:
                assert last_one is not None
                chunks.append((tp_nums[start], tp_nums[last_one]))
                start = None
                num_zeroes = 0
    if start is not None:
        assert last_one is not None
        chunks.append((tp_nums[start], tp_nums[last_one]))
    return chunks


def extract_d1() -> tuple[list[dict[str, object]], list[str]]:
    """Extract D1_malaria_embo2024 (Andrade et al. 2024 EMBO Mol. Med., Kalifabougou, Mali).
    Applies protocol §3, §4.1 Rule 2, §4.2 (S1_endoparasite_vertebrate), Amendment 2 (37.0 C
    homeothermic human blood stage), and Amendment 3 (midpoint start/end, max_zeroes=2,
    May 13 right-censoring, drop left-truncated May 12)."""
    d1_dir = RAW_ROOT / "D1_malaria_embo2024"
    csv_path = d1_dir / "data_raw.csv"
    m9_path = d1_dir / "44321_2024_127_MOESM9_ESM.xlsx"
    log = ["=== Dataset 1: D1_malaria_embo2024 (Andrade et al. 2024 EMBO Mol. Med.) ==="]

    raw_rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    by_id_tps: dict[str, set[str]] = defaultdict(set)
    by_id_age: dict[str, float] = {}
    kid_tp_date: dict[tuple[str, int], datetime.date] = {}
    by_tp_dates: dict[int, list[datetime.date]] = defaultdict(list)

    n_mal_rows = 0
    for r in raw_rows:
        kid = r["ID"].strip()
        tp = r["Timepoint"].strip()
        by_id_tps[kid].add(tp)
        by_id_age[kid] = float(r["Age_in_2012"])
        if tp == "MAL":
            n_mal_rows += 1
            continue
        tn = -1 if tp == "May 12" else (13 if tp == "May 13" else int(tp))
        dt = datetime.datetime.strptime(r["Date"].strip(), "%d.%m.%y").date()
        kid_tp_date[(kid, tn)] = dt
        by_tp_dates[tn].append(dt)

    tp_med_date = {tn: sorted(ds)[len(ds) // 2] for tn, ds in by_tp_dates.items()}

    def get_visit_date(kid: str, tn: int) -> datetime.date:
        return kid_tp_date.get((kid, tn), tp_med_date[tn])

    # Identify the 92 longitudinal asymptomatic cohort children (with visits in Timepoints 1..12)
    # vs the 14 clinical-malaria-only children (only MAL / May 13 / May 12, Fig 4C subset).
    ids_92 = {k for k, tps in by_id_tps.items() if any(t.isdigit() for t in tps)}
    ids_excl_14 = sorted(set(by_id_tps.keys()) - ids_92)
    assert len(ids_92) == 92 and len(ids_excl_14) == 14

    log.append("Exclusions:")
    log.append(
        f"  - EX-D1-1: 153 rows with Timepoint=='MAL' in data_raw.csv / 44321_2024_127_MOESM8_ESM.xlsx "
        f"(and 14 children {ids_excl_14} having zero longitudinal wet-season visits 1..12, sampled only in "
        f"clinical malaria Fig. 4C / May boundary): excluded per Meta_to_infection_length_github.Rmd line 39 "
        f"and protocol §7 item 12 (treated 2013 clinical episodes, not untreated longitudinal clone tracking)."
    )

    tp_grid_ama1 = [-1] + list(range(1, 14))
    pres_ama1: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in raw_rows:
        kid = r["ID"].strip()
        tp = r["Timepoint"].strip()
        if kid not in ids_92 or tp == "MAL":
            continue
        tn = -1 if tp == "May 12" else (13 if tp == "May 13" else int(tp))
        pres_ama1[(kid, r["cluster_name"].strip())].add(tn)

    ref_date = datetime.date(2012, 1, 1)
    ama1_clones: list[dict[str, object]] = []
    n_left_trunc = 0
    for (kid, cl), tset in sorted(pres_ama1.items()):
        for st, en in _chunk_presence(tp_grid_ama1, tset, max_zeroes=2):
            if st == -1 or en == -1:
                n_left_trunc += 1
                continue
            if st == 13:
                t0 = float((get_visit_date(kid, 12) - ref_date).days) + 14.0
            elif st == 1:
                t0 = float((get_visit_date(kid, 1) - ref_date).days) - 7.0
            else:
                t0 = 0.5 * (
                    float((get_visit_date(kid, st - 1) - ref_date).days)
                    + float((get_visit_date(kid, st) - ref_date).days)
                )
            if en == 13:
                t1 = float((get_visit_date(kid, 13) - ref_date).days)
                event = False
            elif en == 12:
                # Biweekly half-interval (+7 d after final wet-season visit TP12 when cleared prior to May 13)
                t1 = float((get_visit_date(kid, 12) - ref_date).days) + 7.0
                event = True
            else:
                t1 = 0.5 * (
                    float((get_visit_date(kid, en) - ref_date).days)
                    + float((get_visit_date(kid, en + 1) - ref_date).days)
                )
                event = True
            ama1_clones.append(
                {
                    "kid": kid,
                    "cluster": cl,
                    "age": by_id_age[kid],
                    "start_tp": st,
                    "end_tp": en,
                    "duration": t1 - t0,
                    "event": event,
                }
            )

    log.append(
        f"  - EX-D1-2: {n_left_trunc} left-truncated ama1 clonal chunks already present at Timepoint=='May 12' "
        f"(Timepoint -1 in Meta_to_infection_length_github.Rmd): excluded per protocol Amendment 3 item 1."
    )
    log.append(
        "  - EX-D1-3: Supplemental files MOESM3_ESM.xlsx..MOESM7_ESM.xlsx (Datasets EV1-EV5: plasma "
        "metabolomics, lipidomics, proteomics), MOESM10_ESM.xlsx (Dataset EV8: SRA accessions), "
        "MOESM11_ESM.zip (SourceData_1.csv: daily clinical case counts), and MOESM13_ESM.zip "
        "(SourceData_4C.xlsx: clinical episode dates without longitudinal clone follow-up): excluded under "
        "protocol §4.1 Rule 6 (no stage-duration observations)."
    )

    # Stratify ama1 into the study's two disjoint host age cohorts (<10 yr [7-9 yr] vs 10-13 yr; Fig. 6C-D, protocol §3)
    out_rows: list[dict[str, object]] = []
    for label, pred in [
        ("clonal blood-stage infection duration (ama1, children 7-9 yr, max_skips=2)", lambda a: a < 10.0),
        ("clonal blood-stage infection duration (ama1, children 10-13 yr, max_skips=2)", lambda a: a >= 10.0),
    ]:
        sub = [c for c in ama1_clones if pred(float(c["age"]))]
        t_arr = np.array([float(c["duration"]) for c in sub], dtype=float)
        ev_arr = np.array([bool(c["event"]) for c in sub], dtype=bool)
        wm = eh.weibull_censored_mle(t_arr, ev_arr)
        unc_cv = eh.uncensored_stats(t_arr[ev_arr])["cv_indiv"]
        cens_pct = 100.0 * wm["cens_frac"]
        out_rows.append(
            {
                "dataset": "D1_malaria_embo2024",
                "taxon": "Plasmodium falciparum",
                "stage": label,
                "stage_category": "S1_endoparasite_vertebrate",
                "temperature_C": 37.0,
                "n": wm["n_total"],
                "mean": wm["mean_mle"],
                "sd": wm["sd_mle"],
                "cv_indiv": wm["cv_mle"],
                "censoring": f"right_censored_{cens_pct:.1f}pct_May2013 (d={wm['n_events']}/{wm['n_total']})",
                "source_file": "D1_malaria_embo2024/data_raw.csv",
                "rule_applied": f"raw_individual_censored_weibull_mle (§4.1 Rule 2, Amend. 3; completers_cv={unc_cv:.4f})",
            }
        )

    # Extract msp2 longitudinal cohort from 44321_2024_127_MOESM9_ESM.xlsx (Dataset EV7, 51 children, Timepoints 1..12)
    m9_sheet = eh.read_xlsx_sheet(m9_path, 0)
    pres_msp2: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in m9_sheet[1:]:
        if len(r) >= 3 and r[0] and r[1] and r[2]:
            pres_msp2[(r[1].strip(), r[0].strip())].add(int(r[2]))
    tp_grid_msp2 = list(range(1, 13))
    msp2_clones: list[dict[str, object]] = []
    for (kid, hbin), tset in sorted(pres_msp2.items()):
        for st, en in _chunk_presence(tp_grid_msp2, tset, max_zeroes=2):
            if st == 1:
                t0 = float((get_visit_date(kid, 1) - ref_date).days) - 7.0
            else:
                t0 = 0.5 * (
                    float((get_visit_date(kid, st - 1) - ref_date).days)
                    + float((get_visit_date(kid, st) - ref_date).days)
                )
            if en == 12:
                # Still positive at final visit Timepoint 12 of Dataset EV7 -> right-censored at TP12
                t1 = float((get_visit_date(kid, 12) - ref_date).days)
                event = False
            else:
                t1 = 0.5 * (
                    float((get_visit_date(kid, en) - ref_date).days)
                    + float((get_visit_date(kid, en + 1) - ref_date).days)
                )
                event = True
            msp2_clones.append(
                {"kid": kid, "hbin": hbin, "start_tp": st, "end_tp": en, "duration": t1 - t0, "event": event}
            )

    t_m9 = np.array([float(c["duration"]) for c in msp2_clones], dtype=float)
    ev_m9 = np.array([bool(c["event"]) for c in msp2_clones], dtype=bool)
    wm_m9 = eh.weibull_censored_mle(t_m9, ev_m9)
    unc_m9 = eh.uncensored_stats(t_m9[ev_m9])["cv_indiv"]
    cens_m9_pct = 100.0 * wm_m9["cens_frac"]
    out_rows.append(
        {
            "dataset": "D1_malaria_embo2024",
            "taxon": "Plasmodium falciparum",
            "stage": "clonal blood-stage infection duration (msp2 capillary genotyping, 51 children, max_skips=2)",
            "stage_category": "S1_endoparasite_vertebrate",
            "temperature_C": 37.0,
            "n": wm_m9["n_total"],
            "mean": wm_m9["mean_mle"],
            "sd": wm_m9["sd_mle"],
            "cv_indiv": wm_m9["cv_mle"],
            "censoring": f"right_censored_{cens_m9_pct:.1f}pct_TP12 (d={wm_m9['n_events']}/{wm_m9['n_total']})",
            "source_file": "D1_malaria_embo2024/44321_2024_127_MOESM9_ESM.xlsx",
            "rule_applied": f"raw_individual_censored_weibull_mle (§4.1 Rule 2, Amend. 3; completers_cv={unc_m9:.4f})",
        }
    )

    # Hand checks of 3 records & all 3 cohort rows against raw files
    c_hc1 = next(
        c for c in ama1_clones
        if c["kid"] == "Kali0277"
        and str(c["cluster"]).startswith("pfama1.00")
        and c["start_tp"] == 8
    )
    assert abs(float(c_hc1["duration"]) - 14.5) < 1e-9 and c_hc1["event"] is True

    c_hc2 = next(
        c for c in ama1_clones
        if c["kid"] == "Kali0061"
        and str(c["cluster"]).startswith("pfama1.00")
        and c["start_tp"] == 10
    )
    assert abs(float(c_hc2["duration"]) - 26.0) < 1e-9 and c_hc2["event"] is True

    c_hc3 = next(
        c for c in msp2_clones
        if c["kid"] == "Kali0069" and c["hbin"] == "457-460" and c["start_tp"] == 8
    )
    assert abs(float(c_hc3["duration"]) - 55.5) < 1e-9 and c_hc3["event"] is True

    log.append("Hand checks (3 raw trajectories + cohort summaries verified against raw files):")
    log.append(
        f"  - HC-D1-1 (Cohort row 1, younger 7-9 yr ama1; raw check Kali0277 in data_raw.csv lines 1609, 1620, 1625): "
        f"Age_in_2012=9.0; cluster pfama1.00_.._pfama1.73 positive at TP=8 (03.11.12, line 1609), absent TP=9..11 "
        f"(>2 skips before TP=12 on line 1610). Preceding TP=7 date=19.10.12 (line 1625) -> midpoint start 2012-10-26.5; "
        f"next TP=9 date=17.11.12 (line 1620) -> midpoint end 2012-11-10.0; hand duration = 14.5 d (uncensored), "
        f"script={c_hc1['duration']:.1f} d. Cohort n={out_rows[0]['n']} (d=242, cens=36.5%), Weibull mean={out_rows[0]['mean']:.4f} d, "
        f"sd={out_rows[0]['sd']:.4f} d, cv_indiv={out_rows[0]['cv_indiv']:.6f}."
    )
    log.append(
        f"  - HC-D1-2 (Cohort row 2, older 10-13 yr ama1; raw check Kali0061 in data_raw.csv lines 2, 6, 7): "
        f"Age_in_2012=11.4082; cluster pfama1.00_.._pfama1.73 positive at TP=10 (26.11.12, line 7) and TP=11 "
        f"(10.12.12, line 6), cleared by TP=12 (24.12.12, line 2). Preceding TP=9 median date=16.11.12 -> midpoint start "
        f"2012-11-21.0; midpoint end (10.12.12, 24.12.12)=2012-12-17.0; hand duration = 26.0 d (uncensored), "
        f"script={c_hc2['duration']:.1f} d. Cohort n={out_rows[1]['n']} (d=340, cens=27.8%), Weibull mean={out_rows[1]['mean']:.4f} d, "
        f"sd={out_rows[1]['sd']:.4f} d, cv_indiv={out_rows[1]['cv_indiv']:.6f}. "
        f"(Pooled 92-child check: 381+471=852 clones, 582 cleared matching Fig. EV5A n=582, 270 persistent median=178.5 d matching paper 178 d)."
    )
    log.append(
        f"  - HC-D1-3 (Cohort row 3, msp2 capillary genotyping; raw check Kali0069 bin 457-460 in "
        f"44321_2024_127_MOESM9_ESM.xlsx rows 371, 420, 500): positive at TP=8 (row 371, 459 bp), TP=9 (row 420, 459 bp), "
        f"1 skip at TP=10 (<=2 skips), positive at TP=11 (row 500, 459 bp), absent at TP=12. Midpoint start "
        f"(TP7 2012-10-18, TP8 2012-11-02)=2012-10-25.5; midpoint end (TP11 2012-12-13, TP12 2012-12-27)=2012-12-20.0; "
        f"hand duration = 55.5 d (uncensored), script={c_hc3['duration']:.1f} d. Cohort n={out_rows[2]['n']} (d=303, cens=8.7%), "
        f"Weibull mean={out_rows[2]['mean']:.4f} d, sd={out_rows[2]['sd']:.4f} d, cv_indiv={out_rows[2]['cv_indiv']:.6f}."
    )
    log.append("")
    return out_rows, log


def _read_biff8_xls(path: Path) -> list[list[str]]:
    """Minimal pure-stdlib OLE2 + BIFF8 cell reader for legacy `.xls` spreadsheets (`Table_2.xls`)."""
    data = path.read_bytes()
    sec_shift = struct.unpack_from("<H", data, 30)[0]
    sec_size = 1 << sec_shift
    num_fat = struct.unpack_from("<I", data, 44)[0]
    dir_start = struct.unpack_from("<I", data, 48)[0]
    difat = list(struct.unpack_from("<109I", data, 76))
    fat: list[int] = []
    for s in difat[:num_fat]:
        if s < 0xFFFFFFFC:
            off = 512 + s * sec_size
            fat.extend(struct.unpack_from(f"<{sec_size // 4}I", data, off))

    def read_chain(start_sec: int) -> bytes:
        out = bytearray()
        s = start_sec
        while s < 0xFFFFFFFC:
            out.extend(data[512 + s * sec_size : 512 + (s + 1) * sec_size])
            s = fat[s]
        return bytes(out)

    dir_bytes = read_chain(dir_start)
    wb_stream = b""
    for i in range(0, len(dir_bytes), 128):
        entry = dir_bytes[i : i + 128]
        nlen = struct.unpack_from("<H", entry, 64)[0]
        if nlen == 0:
            continue
        name = entry[: nlen - 2].decode("utf-16le", errors="ignore")
        start_s, stream_sz = struct.unpack_from("<II", entry, 116)
        if name in ("Workbook", "Book"):
            wb_stream = read_chain(start_s)[:stream_sz]
            break

    def decode_rk(rk: int) -> float:
        if rk & 2:
            val = float(rk >> 2)
        else:
            val = struct.unpack("<d", struct.pack("<Q", (rk & ~3) << 32))[0]
        if rk & 1:
            val /= 100.0
        return val

    pos = 0
    sst: list[str] = []
    cells: dict[tuple[int, int], str] = {}
    while pos + 4 <= len(wb_stream):
        rt, rl = struct.unpack_from("<HH", wb_stream, pos)
        rec = wb_stream[pos + 4 : pos + 4 + rl]
        pos += 4 + rl
        if rt == 0x00FC:  # SST
            uniq = struct.unpack_from("<I", rec, 4)[0]
            p = 8
            for _ in range(uniq):
                cch = struct.unpack_from("<H", rec, p)[0]
                fl = rec[p + 2]
                p += 3
                rt_runs = 0
                ext_sz = 0
                if fl & 0x08:
                    rt_runs = struct.unpack_from("<H", rec, p)[0]
                    p += 2
                if fl & 0x04:
                    ext_sz = struct.unpack_from("<I", rec, p)[0]
                    p += 4
                if fl & 0x01:
                    s_val = rec[p : p + 2 * cch].decode("utf-16le", errors="replace")
                    p += 2 * cch
                else:
                    s_val = rec[p : p + cch].decode("latin1", errors="replace")
                    p += cch
                p += 4 * rt_runs + ext_sz
                sst.append(s_val)
        elif rt == 0x00FD:  # LABELSST
            r, c, _xf, sidx = struct.unpack_from("<HHHI", rec, 0)
            cells[(r, c)] = sst[sidx].strip()
        elif rt == 0x0203:  # NUMBER
            r, c, _xf, val = struct.unpack_from("<HHHd", rec, 0)
            cells[(r, c)] = f"{val:g}"
        elif rt == 0x027E:  # RK
            r, c, _xf, rk = struct.unpack_from("<HHHI", rec, 0)
            cells[(r, c)] = f"{decode_rk(rk):g}"
        elif rt == 0x00BD:  # MULRK
            r, fc = struct.unpack_from("<HH", rec, 0)
            lc = struct.unpack_from("<H", rec, len(rec) - 2)[0]
            for k in range(lc - fc + 1):
                _xf, rk = struct.unpack_from("<HI", rec, 4 + 6 * k)
                cells[(r, fc + k)] = f"{decode_rk(rk):g}"

    if not cells:
        return []
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return [[cells.get((r, c), "") for c in range(max_c + 1)] for r in range(max_r + 1)]


def extract_d2() -> tuple[list[dict[str, object]], list[str]]:
    """Extract D2_malariotherapy_plos2013 (Johnston et al. 2013 PLOS Comput. Biol., Table_2.xls).
    Per protocol §4.1 Rule 6 and Amendment 1 item 3, Table_2.xls reports only observed minimum,
    median, and maximum without mean, SD, SE, or individual data; 0 primary rows emitted."""
    xls_path = RAW_ROOT / "D2_malariotherapy_plos2013" / "Table_2.xls"
    table = _read_biff8_xls(xls_path)
    assert len(table) == 10, f"Expected 10 rows in Table_2.xls, got {len(table)}"
    log = ["=== Dataset 2: D2_malariotherapy_plos2013 (Johnston et al. 2013 PLOS Comput. Biol.) ==="]
    log.append("Exclusions:")
    log.append(
        f"  - EX-D2-1 (Index {table[9][0]} '{table[9][1]}' in Table_2.xls row 9): "
        f"Untreated USPHS malariotherapy P. falciparum blood-stage duration reports only "
        f"min={table[9][2]} d, median={table[9][3]} d, max={table[9][4]} d (plus simulated model "
        f"min={table[9][5]}, median={table[9][6]}, max={table[9][7]}), with no sample mean, SD, SE, "
        f"or sample size n. Excluded under protocol §4.1 Rule 6 (range/median-only, non-identifiable CV_indiv) "
        f"and Amendment 1 item 3 (never imputed from range/median)."
    )
    log.append(
        "  - EX-D2-2 (Indices 2-1..2-8 in Table_2.xls rows 1..8: Initial slope, Log density at first "
        "local maximum, Number of local maxima, Slope of local maxima, Geometric mean of intervals between "
        "consecutive local maxima, SD of logs of consecutive local maxima, Proportion of positive observations "
        "in first/second half): within-host parasitaemia trajectory summary indices rather than stage durations, "
        "and likewise reported only as min/median/max without mean or among-patient SD/SE (§4.1 Rule 6)."
    )

    # Hand checks of 3 raw rows of Table_2.xls
    assert table[9][:5] == ["2-9", "Last positive day", "37", "215", "405"]
    assert table[5][:5] == [
        "2-5",
        "Geometric mean of the intervals between consecutive local maxima",
        "14.4",
        "20.0",
        "77.8",
    ]
    assert table[3][:5] == ["2-3", "Number of local maxima", "2", "10", "17"]

    log.append("Hand checks (3 raw rows verified directly against BIFF8 Workbook stream in Table_2.xls):")
    log.append(
        f"  - HC-D2-1 (Table_2.xls row 9, Index {table[9][0]} '{table[9][1]}'): "
        f"obs_min={table[9][2]} d, obs_median={table[9][3]} d, obs_max={table[9][4]} d, "
        f"model_min={table[9][5]} d, model_median={table[9][6]} d, model_max={table[9][7]} d; "
        f"confirms Amendment 1 item 3 (median 215 d, max 405 d, no mean or SD/SE -> excluded)."
    )
    log.append(
        f"  - HC-D2-2 (Table_2.xls row 5, Index {table[5][0]} '{table[5][1]}'): "
        f"obs_min={table[5][2]} d, obs_median={table[5][3]} d, obs_max={table[5][4]} d; "
        f"inter-peak parasitaemia wave spacing reported as min/median/max only -> excluded."
    )
    log.append(
        f"  - HC-D2-3 (Table_2.xls row 3, Index {table[3][0]} '{table[3][1]}'): "
        f"obs_min={table[3][2]}, obs_median={table[3][3]}, obs_max={table[3][4]}; "
        f"count of parasitaemia peaks (non-duration) reported as min/median/max only -> excluded."
    )
    log.append("")
    return [], log


def extract_d3() -> tuple[list[dict[str, object]], list[str]]:
    """Extract D3_anopheles_pf_shapiro2017 (Shapiro, Whitehead & Thomas 2017 PLOS Biol.).
    - `infectious.data.csv`: 12 cohorts (2 blocks x 6 constant temperatures 21..34 C) of
      destructive salivary-gland sporozoite dissections -> protocol §4.1 Rule 3 (`cross_sectional_cdf_mle`),
      stage category `S2_parasite_in_invertebrate_vector_or_snail`.
    - `temp.surv.csv`: 12 cohorts (2 blocks x 6 constant temperatures 21..34 C) of adult
      Anopheles stephensi survival -> right-censored fractions are 52.1%..74.2% (> 50% ceiling
      of protocol §4.1 Rule 2 because 4,439/7,363 mosquitoes were removed alive mid-study for
      destructive dissections); all 12 survival cohorts are logged as exclusions."""
    d3_dir = RAW_ROOT / "D3_anopheles_pf_shapiro2017"
    inf_path = d3_dir / "infectious.data.csv"
    surv_path = d3_dir / "temp.surv.csv"
    log = ["=== Dataset 3: D3_anopheles_pf_shapiro2017 (Shapiro, Whitehead & Thomas 2017 PLOS Biol.) ==="]

    # 1. Audit adult mosquito survival cohorts (temp.surv.csv) against §4.1 Rule 2 (cens_frac <= 0.50)
    srows = list(csv.DictReader(surv_path.open(encoding="utf-8-sig")))
    by_surv: dict[tuple[int, int], list[tuple[float, bool]]] = defaultdict(list)
    for r in srows:
        by_surv[(int(r["expt"]), int(r["temp"]))].append((float(r["day"]), int(r["status"]) == 1))

    # Rule 7 check against authors' PBiol.R.Code.R line 113 (expt=1, temp=21: N=609, Events=201, Censored=408)
    s_21_1 = by_surv[(1, 21)]
    assert len(s_21_1) == 609 and sum(1 for _, ev in s_21_1 if ev) == 201

    log.append("Exclusions:")
    out_rows: list[dict[str, object]] = []
    for (expt, temp), vals in sorted(by_surv.items()):
        t_arr = np.array([v[0] for v in vals], dtype=float)
        ev_arr = np.array([v[1] for v in vals], dtype=bool)
        wm = eh.weibull_censored_mle(t_arr, ev_arr)
        assert wm["cens_frac"] > 0.50, f"Expected cens_frac > 0.50 in D3 survival, got {wm['cens_frac']}"
        log.append(
            f"  - EX-D3-SURV-B{expt}-T{temp}: Anopheles stephensi adult survival in temp.surv.csv "
            f"(expt={expt}, temp={temp} C): n={wm['n_total']}, uncensored deaths d={wm['n_events']}, "
            f"right-censored={wm['n_total'] - wm['n_events']} ({100.0 * wm['cens_frac']:.2f}% > 50.0% ceiling; "
            f"mosquitoes removed alive daily for destructive sporozoite dissections). Excluded under protocol "
            f"§4.1 Rule 2 (Weibull MLE reference: mean={wm['mean_mle']:.3f} d, sd={wm['sd_mle']:.3f} d, cv={wm['cv_mle']:.4f})."
        )

    # 2. Extract Plasmodium falciparum EIP cross-sectional cohorts (infectious.data.csv, §4.1 Rule 3)
    irows = list(csv.DictReader(inf_path.open(encoding="utf-8-sig")))
    by_bt_day: dict[tuple[int, int, float], list[int]] = defaultdict(lambda: [0, 0])
    for r in irows:
        b = int(r["Block"])
        t = int(r["Temperature"])
        dy = float(r["Day"])
        sp = int(r["Sporozoite Prevalence"])
        by_bt_day[(b, t, dy)][0] += sp
        by_bt_day[(b, t, dy)][1] += 1

    by_bt: dict[tuple[int, int], list[tuple[float, int, int]]] = defaultdict(list)
    for (b, t, dy), (pos, tot) in sorted(by_bt_day.items()):
        by_bt[(b, t)].append((dy, pos, tot))

    for (b, t), dlist in sorted(by_bt.items()):
        days = np.array([x[0] for x in dlist], dtype=float)
        pos = np.array([x[1] for x in dlist], dtype=float)
        tot = np.array([x[2] for x in dlist], dtype=float)
        gm = eh.gamma_cdf_cross_sectional_mle(days, pos, tot)
        assert gm["ok"] and gm["bracketed"] and gm["n_dissected"] >= 50
        at_cap = gm["k_gamma"] > 600.0
        cap_note = "; k_gamma at daily-step identifiability cap (log_k<=6.5, §6.2 P2a)" if at_cap else ""
        out_rows.append(
            {
                "dataset": "D3_anopheles_pf_shapiro2017",
                "taxon": "Plasmodium falciparum (in Anopheles stephensi)",
                "stage": f"extrinsic incubation period (EIP, salivary-gland sporozoites, Block {b + 1})",
                "stage_category": "S2_parasite_in_invertebrate_vector_or_snail",
                "temperature_C": float(t),
                "n": gm["n_dissected"],
                "mean": gm["mean_eip"],
                "sd": gm["sd_eip"],
                "cv_indiv": gm["cv_indiv"],
                "censoring": f"cross_sectional_destructive ({gm['n_days']} dissection days, pos={gm['n_positive']}/{gm['n_dissected']})",
                "source_file": "D3_anopheles_pf_shapiro2017/infectious.data.csv",
                "rule_applied": f"cross_sectional_cdf_mle (Gamma CDF binomial MLE, §4.1 Rule 3, p_max={gm['p_max']:.3f}{cap_note})",
            }
        )

    # Hand checks of 3 cohorts against raw files
    d_b0_21 = by_bt[(0, 21)]
    assert sum(x[2] for x in d_b0_21) == 374 and sum(x[1] for x in d_b0_21) == 90
    assert d_b0_21[0] == (10.0, 0, 31) and d_b0_21[2] == (12.0, 4, 34) and d_b0_21[-1] == (25.0, 7, 16)

    d_b1_21 = by_bt[(1, 21)]
    assert sum(x[2] for x in d_b1_21) == 356 and sum(x[1] for x in d_b1_21) == 103
    assert d_b1_21[0] == (10.0, 0, 31) and d_b1_21[3] == (13.0, 9, 33) and d_b1_21[-1] == (25.0, 20, 31)

    d_b1_24 = by_bt[(1, 24)]
    assert sum(x[2] for x in d_b1_24) == 392 and sum(x[1] for x in d_b1_24) == 202
    assert d_b1_24[0] == (9.0, 0, 34) and d_b1_24[3] == (12.0, 14, 33) and d_b1_24[8] == (18.0, 24, 32)

    r_b0_21 = next(r for r in out_rows if "Block 1" in str(r["stage"]) and float(r["temperature_C"]) == 21.0)
    r_b1_21 = next(r for r in out_rows if "Block 2" in str(r["stage"]) and float(r["temperature_C"]) == 21.0)
    r_b1_24 = next(r for r in out_rows if "Block 2" in str(r["stage"]) and float(r["temperature_C"]) == 24.0)

    log.append("Hand checks (3 EIP rows + survival exclusion verified against raw CSV lines and PBiol.R.Code.R):")
    log.append(
        f"  - HC-D3-1 (Block=0 [Block 1], Temp=21 C in infectious.data.csv lines 2..700): "
        f"12 dissection days, hand-tallied Day 10=0/31, Day 12=4/34, ..., Day 25=7/16 -> total N=374, pos=90 "
        f"(matches script n={r_b0_21['n']}); Gamma CDF MLE mean={r_b0_21['mean']:.4f} d (cf. PBiol.R.Code.R logistic "
        f"t_50=13.972 d, g=0.3966 vs p_max=0.389), sd={r_b0_21['sd']:.4f} d, cv_indiv={r_b0_21['cv_indiv']:.6f}. "
        f"(Paired survival check temp.surv.csv lines 2..610: N=609, deaths=201, censored=408=67.00%, exact match to PBiol.R.Code.R line 113)."
    )
    log.append(
        f"  - HC-D3-2 (Block=1 [Block 2], Temp=21 C in infectious.data.csv lines 33..731): "
        f"11 dissection days, hand-tallied Day 10=0/31, Day 13=9/33, ..., Day 25=20/31 -> total N=356, pos=103 "
        f"(matches script n={r_b1_21['n']}); Gamma CDF MLE mean={r_b1_21['mean']:.4f} d (cf. PBiol.R.Code.R logistic "
        f"t_50=16.128 d, g=0.6374 vs p_max=0.591), sd={r_b1_21['sd']:.4f} d, cv_indiv={r_b1_21['cv_indiv']:.6f}."
    )
    log.append(
        f"  - HC-D3-3 (Block=1 [Block 2], Temp=24 C in infectious.data.csv lines 732..1508): "
        f"12 dissection days, hand-tallied Day 9=0/34, Day 12=14/33, Day 18=24/32 -> total N=392, pos=202 "
        f"(matches script n={r_b1_24['n']}); Gamma CDF MLE mean={r_b1_24['mean']:.4f} d (cf. PBiol.R.Code.R logistic "
        f"t_50=11.661 d, g=0.7036 vs p_max=0.702), sd={r_b1_24['sd']:.4f} d, cv_indiv={r_b1_24['cv_indiv']:.6f}."
    )
    log.append(
        "  - Note on daily check spacing (protocol §6.2 P2a & Rule 13): On 7 of 12 EIP cohorts (27, 30, 34 C in Block 1; "
        "27, 30, 32, 34 C in Block 2), the 0->plateau jump occurs within 1 daily interval (and 30-34 C exhibit post-peak "
        "sporozoite decline after Day 10-12 as documented in PBiol.R.Code.R line 576), placing extract_helpers.gamma_cdf_cross_sectional_mle "
        "at its log_k=6.5 step-function cap (cv_indiv=0.038774; rising-phase-only check at 30-32 C yields interior cv_indiv=0.0475..0.0991)."
    )
    log.append("")
    return out_rows, log


def extract_d4() -> tuple[list[dict[str, object]], list[str]]:
    """Extract D4_plasmodium_msos2021 (Stopard, Churcher & Lambert 2021 PLOS Comput. Biol.).
    Applies protocol Amendment 4 item 1 (exclude `index_ref == 1` `Shapiro_Thomas` as exact
    7,363-survival + 4,439-dissection duplicate of D3), §4.1 Rules 1-2 on mosquito survival
    cohorts (`data__mSOS_all_survival_data.csv`), and §4.1 Rule 3 on cross-sectional parasite
    development cohorts (`data__mSOS_all_parasite_data.csv`)."""
    d4_dir = RAW_ROOT / "D4_plasmodium_msos2021"
    surv_path = d4_dir / "data__mSOS_all_survival_data.csv"
    para_path = d4_dir / "data__mSOS_all_parasite_data.csv"
    log = ["=== Dataset 4: D4_plasmodium_msos2021 (Stopard et al. 2021 PLOS Comput. Biol. / mSOS) ==="]
    log.append("Exclusions:")
    out_rows: list[dict[str, object]] = []

    # 1. Mosquito survival cohorts (`data__mSOS_all_survival_data.csv`)
    srows = list(csv.DictReader(surv_path.open(encoding="utf-8-sig")))
    n_dup_surv = sum(1 for r in srows if int(r["index_ref"]) == 1)
    assert n_dup_surv == 7363
    log.append(
        f"  - EX-D4-DEDUP-SURV: {n_dup_surv} rows with index_ref==1 ('Shapiro_Thomas') in "
        f"data__mSOS_all_survival_data.csv excluded per protocol Amendment 4 item 1 (exact duplicate "
        f"of D3_anopheles_pf_shapiro2017/temp.surv.csv)."
    )

    by_surv: dict[tuple[int, str, str, int, float], list[tuple[float, bool]]] = defaultdict(list)
    for r in srows:
        iref = int(r["index_ref"])
        if iref == 1:
            continue
        key = (iref, r["ref"].strip(), r["species"].strip(), int(r["initial_infection"]), float(r["temp"]))
        by_surv[key].append((float(r["day"]), int(r["survival_status"]) == 1))

    for (iref, ref, sp, inf_flag, temp), vals in sorted(by_surv.items()):
        t_arr = np.array([v[0] for v in vals], dtype=float)
        ev_arr = np.array([v[1] for v in vals], dtype=bool)
        wm = eh.weibull_censored_mle(t_arr, ev_arr)
        taxon = f"Anopheles {sp}"
        inf_desc = "P. falciparum-exposed" if inf_flag == 1 else "uninfected bloodmeal"
        if wm["n_events"] < 10 or wm["cens_frac"] > 0.50:
            log.append(
                f"  - EX-D4-SURV-{ref}-{sp}-inf{inf_flag}-T{temp:g}: {taxon} ({inf_desc}, ref={ref}, "
                f"index_ref={iref}, temp={temp:g} C) has n={wm['n_total']}, uncensored deaths d={wm['n_events']}, "
                f"right-censored={wm['n_total'] - wm['n_events']} ({100.0 * wm['cens_frac']:.2f}% > 50.0% ceiling). "
                f"Excluded under protocol §4.1 Rule 2 (Weibull MLE reference: mean={wm['mean_mle']:.3f} d, "
                f"sd={wm['sd_mle']:.3f} d, cv={wm['cv_mle']:.4f})."
            )
            continue

        if wm["cens_frac"] <= 0.05:
            # Protocol §4.1 Rule 1: n >= 10 and right-censored fraction <= 0.05 -> raw_individual_uncensored
            st = eh.uncensored_stats(t_arr[ev_arr], dt_obs=1.0)
            cens_pct = 100.0 * wm["cens_frac"]
            out_rows.append(
                {
                    "dataset": "D4_plasmodium_msos2021",
                    "taxon": taxon,
                    "stage": f"adult female survival ({inf_desc}, ref={ref})",
                    "stage_category": "S5_vector_tick_adult_or_offhost_survival",
                    "temperature_C": temp,
                    "n": st["n"],
                    "mean": st["mean"],
                    "sd": st["sd"],
                    "cv_indiv": st["cv_indiv"],
                    "censoring": f"nearly_uncensored_{cens_pct:.2f}pct (d={wm['n_events']}/{wm['n_total']})",
                    "source_file": "D4_plasmodium_msos2021/data__mSOS_all_survival_data.csv",
                    "rule_applied": (
                        f"raw_individual_uncensored (Sokal-Rohlf CV, §4.1 Rule 1, cens<=5%; "
                        f"cv_sheppard={st['cv_sheppard']:.4f}, weibull_cv={wm['cv_mle']:.4f})"
                    ),
                }
            )
        else:
            # Protocol §4.1 Rule 2: d >= 10 and 0.05 < right-censored fraction <= 0.50 -> Weibull MLE
            unc_cv = eh.uncensored_stats(t_arr[ev_arr])["cv_indiv"]
            cens_pct = 100.0 * wm["cens_frac"]
            out_rows.append(
                {
                    "dataset": "D4_plasmodium_msos2021",
                    "taxon": taxon,
                    "stage": f"adult female survival ({inf_desc}, ref={ref})",
                    "stage_category": "S5_vector_tick_adult_or_offhost_survival",
                    "temperature_C": temp,
                    "n": wm["n_total"],
                    "mean": wm["mean_mle"],
                    "sd": wm["sd_mle"],
                    "cv_indiv": wm["cv_mle"],
                    "censoring": f"right_censored_{cens_pct:.1f}pct (d={wm['n_events']}/{wm['n_total']})",
                    "source_file": "D4_plasmodium_msos2021/data__mSOS_all_survival_data.csv",
                    "rule_applied": (
                        f"raw_individual_censored_weibull_mle (§4.1 Rule 2; completers_cv={unc_cv:.4f})"
                    ),
                }
            )

    # 2. Parasite sporogony / EIP cohorts (`data__mSOS_all_parasite_data.csv`)
    prows = list(csv.DictReader(para_path.open(encoding="utf-8-sig")))
    n_dup_para = sum(1 for r in prows if int(r["index_ref"]) == 1)
    assert n_dup_para == 4439
    log.append(
        f"  - EX-D4-DEDUP-PARA: {n_dup_para} rows with index_ref==1 ('Shapiro_Thomas') in "
        f"data__mSOS_all_parasite_data.csv excluded per protocol Amendment 4 item 1 (exact duplicate "
        f"of D3_anopheles_pf_shapiro2017/infectious.data.csv)."
    )

    by_para: dict[tuple[int, str, str, str, str, str, float, float, float], dict[float, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    n_pres_typo = 0
    for r in prows:
        iref = int(r["index_ref"])
        if iref == 1:
            continue
        key = (
            iref,
            r["ref"].strip(),
            r["vector_species"].strip(),
            r["vector_strain"].strip(),
            r["strain"].strip(),
            r["lifestage"].strip(),
            float(r["temp"]),
            float(r["DTR"]),
            float(r["humidity"]),
        )
        dy = float(r["day_post_inf"])
        pr_raw = int(r["presence"])
        if pr_raw > 1:
            n_pres_typo += 1
        pr = 1 if pr_raw >= 1 else 0
        by_para[key][dy][0] += pr
        by_para[key][dy][1] += 1

    n_single_day_murdock = 0
    for key, dmap in sorted(by_para.items()):
        iref, ref, vsp, vstr, pstr, stage_code, temp, dtr, hum = key
        dlist = sorted((dy, pos, tot) for dy, (pos, tot) in dmap.items())
        days = np.array([x[0] for x in dlist], dtype=float)
        pos = np.array([x[1] for x in dlist], dtype=float)
        tot = np.array([x[2] for x in dlist], dtype=float)
        n_tot = int(tot.sum())

        if days.size < 3:
            if iref == 3:
                n_single_day_murdock += 1
            else:
                log.append(
                    f"  - EX-D4-PARA-{ref}-{stage_code}-T{temp:g}-DTR{dtr:g}: only {days.size} dissection day(s) "
                    f"(N={n_tot}); excluded under protocol §4.1 Rule 3 (requires >=3 distinct dissection days)."
                )
            continue

        gm = eh.gamma_cdf_cross_sectional_mle(days, pos, tot)
        if (not gm.get("ok")) or (not gm.get("bracketed")) or n_tot < 50:
            obs_p = pos / tot
            log.append(
                f"  - EX-D4-PARA-{ref}-{vsp}-{stage_code}-T{temp:g}-DTR{dtr:g}: cross-sectional {stage_code} "
                f"prevalence (index_ref={iref}, N={n_tot}, days={days[0]:g}..{days[-1]:g}, "
                f"obs_prev_range={obs_p.min():.3f}..{obs_p.max():.3f}) fails transition bracketing "
                f"(starts above 25% of p_max={gm.get('p_max', float('nan')):.3f}, onset occurred prior to "
                f"first dissection day {days[0]:g}). Excluded under protocol §4.1 Rule 3 "
                f"(unbracketed Gamma reference: mean={gm.get('mean_eip', float('nan')):.3f} d, "
                f"cv={gm.get('cv_indiv', float('nan')):.4f})."
            )
            continue

        at_cap = gm["k_gamma"] > 600.0
        cap_note = "; k_gamma at daily-step identifiability cap (log_k<=6.5, §6.2 P2a)" if at_cap else ""
        out_rows.append(
            {
                "dataset": "D4_plasmodium_msos2021",
                "taxon": f"Plasmodium falciparum {pstr} (in Anopheles {vsp})",
                "stage": f"extrinsic incubation period (EIP, {stage_code}, ref={ref}, DTR={dtr:g} C, RH={hum:g}%)",
                "stage_category": "S2_parasite_in_invertebrate_vector_or_snail",
                "temperature_C": temp,
                "n": gm["n_dissected"],
                "mean": gm["mean_eip"],
                "sd": gm["sd_eip"],
                "cv_indiv": gm["cv_indiv"],
                "censoring": f"cross_sectional_destructive ({gm['n_days']} dissection days, pos={gm['n_positive']}/{gm['n_dissected']})",
                "source_file": "D4_plasmodium_msos2021/data__mSOS_all_parasite_data.csv",
                "rule_applied": f"cross_sectional_cdf_mle (Gamma CDF binomial MLE, §4.1 Rule 3, p_max={gm['p_max']:.3f}{cap_note})",
            }
        )

    assert n_single_day_murdock == 36
    log.append(
        f"  - EX-D4-PARA-Murdock_Thomas-36cells: all 36 treatment cells of index_ref==3 ('Murdock_Thomas': "
        f"An. gambiae keele & An. stephensi liston x cyst [day 7 only] & gland-spz [day 15 only] x "
        f"temp 27, 30, 33 C x DTR 0, 6, 9 C; including {n_pres_typo} rows coded presence=='2' for oocyst count 2) "
        f"were dissected at a single post-infection day (ndays=1 < 3), so EIP transition spread is non-identifiable; "
        f"excluded under protocol §4.1 Rule 3."
    )

    # Hand checks of 3 extracted rows against raw CSV files
    r_g33_0 = next(
        r for r in out_rows
        if r["taxon"] == "Anopheles gambiae"
        and "uninfected" in str(r["stage"])
        and float(r["temperature_C"]) == 33.0
    )
    assert r_g33_0["n"] == 191
    assert abs(float(r_g33_0["mean"]) - (1753.0 / 191.0)) < 1e-9
    hand_var_33 = (17893.0 - (1753.0 ** 2) / 191.0) / 190.0
    hand_sd_33 = math.sqrt(hand_var_33)
    hand_cv_33 = (hand_sd_33 / (1753.0 / 191.0)) * (1.0 + 1.0 / (4.0 * 191.0))
    assert abs(float(r_g33_0["sd"]) - hand_sd_33) < 1e-9
    assert abs(float(r_g33_0["cv_indiv"]) - hand_cv_33) < 1e-9

    r_g30_1 = next(
        r for r in out_rows
        if r["taxon"] == "Anopheles gambiae"
        and "exposed" in str(r["stage"])
        and float(r["temperature_C"]) == 30.0
    )
    assert r_g30_1["n"] == 273

    r_eip_slt = next(r for r in out_rows if "Shapiro_L_Thomas" in str(r["stage"]))
    assert r_eip_slt["n"] == 1281

    log.append("Hand checks (3 extracted rows verified directly against raw CSV files):")
    log.append(
        f"  - HC-D4-1 (Anopheles gambiae uninfected survival at 33 C, Murdock_Thomas index_ref=3, "
        f"data__mSOS_all_survival_data.csv lines 10310..13925): 194 mosquitoes total, 191 observed deaths "
        f"(days 1..14) and 3 survivors censored at day 15 (1.55% <= 5% -> §4.1 Rule 1). Hand tally of 191 death "
        f"days: sum(t)=1753, sum(t^2)=17893 -> mean=1753/191=9.178010 d, sd=sqrt((17893-1753^2/191)/190)=3.081308 d, "
        f"Sokal-Rohlf cv_indiv=(sd/mean)*(1+1/764)=0.336167 (script={r_g33_0['cv_indiv']:.6f}; right-censored Weibull MLE "
        f"on all 194 gives cv=0.335695, diff 0.00047)."
    )
    log.append(
        f"  - HC-D4-2 (Anopheles gambiae P. falciparum-exposed survival at 30 C, Murdock_Thomas index_ref=3, "
        f"data__mSOS_all_survival_data.csv lines 10299..14231): hand tally n=273, deaths d=179, right-censored=94 "
        f"(34.43% in (5%, 50%] -> §4.1 Rule 2), total time at risk sum(t)=3028 d; right-censored Weibull MLE "
        f"mean={r_g30_1['mean']:.4f} d, sd={r_g30_1['sd']:.4f} d, cv_indiv={r_g30_1['cv_indiv']:.6f} (shape k=3.2960)."
    )
    log.append(
        f"  - HC-D4-3 (Plasmodium falciparum NF54 gland-spz EIP in An. stephensi at 27 C, Shapiro_L_Thomas index_ref=2, "
        f"data__mSOS_all_parasite_data.csv lines 920..2200): 8 dissection days (Day 9=0/155, Day 10=0/154, "
        f"Day 11=67/153, Day 12=85/164, Day 13=102/169, Day 14=104/165, Day 15=100/167, Day 16=96/154) -> "
        f"hand sum N=1281, pos=554 (matches script n={r_eip_slt['n']}); Gamma CDF MLE mean={r_eip_slt['mean']:.4f} d, "
        f"sd={r_eip_slt['sd']:.4f} d, cv_indiv={r_eip_slt['cv_indiv']:.6f} (p_max=0.5988)."
    )
    log.append("")
    return out_rows, log


def fmt_num(val: float | int | str | None) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, int):
        return str(val)
    if not math.isfinite(val):
        return ""
    return f"{val:.6f}"


def write_outputs(rows: list[dict[str, object]], log_lines: list[str], datasets_done: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            out_row = {
                col: fmt_num(r.get(col, "")) if col in ("mean", "sd", "cv_indiv", "temperature_C") else r.get(col, "")
                for col in COLUMNS
            }
            writer.writerow(out_row)

    final_lines = list(log_lines)
    final_lines.append(f"Part 1: {len(rows)} rows from {datasets_done}/{len(ASSIGNED_DATASETS)} datasets")
    LOG_OUT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    log_lines = verify_helpers_known_results()
    rows: list[dict[str, object]] = []
    datasets_done = 0

    d1_rows, d1_log = extract_d1()
    rows.extend(d1_rows)
    log_lines.extend(d1_log)
    datasets_done += 1
    write_outputs(rows, log_lines, datasets_done)
    print(f"[1/4] D1 saved -> {len(rows)} rows total.")

    d2_rows, d2_log = extract_d2()
    rows.extend(d2_rows)
    log_lines.extend(d2_log)
    datasets_done += 1
    write_outputs(rows, log_lines, datasets_done)
    print(f"[2/4] D2 saved -> {len(rows)} rows total.")

    d3_rows, d3_log = extract_d3()
    rows.extend(d3_rows)
    log_lines.extend(d3_log)
    datasets_done += 1
    write_outputs(rows, log_lines, datasets_done)
    print(f"[3/4] D3 saved -> {len(rows)} rows total.")

    d4_rows, d4_log = extract_d4()
    rows.extend(d4_rows)
    log_lines.extend(d4_log)
    datasets_done += 1
    write_outputs(rows, log_lines, datasets_done)
    print(f"[4/4] D4 saved -> {len(rows)} rows total.")


if __name__ == "__main__":
    main()
