#!/usr/bin/env python3
"""Safe standard-library loader for open bacteria-phage datasets in `data/raw/chemostat/`.

Parses CSV, ZIP, and OpenXML `.xlsx` files using only Python's standard library (`csv`,
`zipfile`, `xml.etree.ElementTree`), treating all downloaded files as untrusted data.
Run under `python3 -I`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import os
import xml.etree.ElementTree as ET
import zipfile
import numpy as np


@dataclass
class UnitSeries:
    dataset_id: str
    unit_id: str
    treatment: str
    infected: bool
    times_d: np.ndarray          # observation times in days
    host_density: np.ndarray     # cells/mL (NaN where missing)
    phage_density: np.ndarray    # virions or PFU/mL (NaN where missing or uninfected)
    dilution_rate_d: np.ndarray  # effective washout rate per interval in d^-1 (length len(times_d))
    res_times_d: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    res_frac_anc: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    res_frac_contemp: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    meta: dict = field(default_factory=dict)


def read_xlsx_sheet_dicts(path: str, sheet_index: int = 0) -> list[dict[str, str]]:
    """Read an OpenXML worksheet into a list of {column_letter: string_value} dicts."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path, "r") as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("main:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        sh = wb.findall(".//main:sheet", ns)[sheet_index]
        target = rmap[sh.attrib[rel_ns]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sxml = ET.fromstring(zf.read(target))
        rows_out: list[dict[str, str]] = []
        for row in sxml.findall(".//main:sheetData/main:row", ns):
            rd: dict[str, str] = {}
            for c in row.findall("main:c", ns):
                col = "".join(ch for ch in c.attrib.get("r", "") if ch.isalpha())
                t = c.attrib.get("t", "")
                v = c.find("main:v", ns)
                is_el = c.find("main:is", ns)
                if t == "s" and v is not None and v.text is not None:
                    idx = int(v.text)
                    rd[col] = shared[idx] if 0 <= idx < len(shared) else ""
                elif t == "inlineStr" and is_el is not None:
                    rd[col] = "".join(x.text or "" for x in is_el.findall(".//main:t", ns))
                elif v is not None and v.text is not None:
                    rd[col] = v.text
            rows_out.append(rd)
        return rows_out


def _fval(s: str | None) -> float:
    if s is None:
        return math.nan
    s = s.strip()
    if s in ("", "NA", "NaN", "nan", "N/A", "."):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def load_d1_larsen2019(raw_root: str) -> list[UnitSeries]:
    """Load Larsen, Wilhelm & Lennon (2019) Synechococcus WH7803 - S-RIM8 continuous chemostats."""
    base = os.path.join(raw_root, "larsen2019_synechococcus_chemostat")
    cid_path = os.path.join(base, "cid-means.csv")
    inf_path = os.path.join(base, "inf-mat.csv")

    # Parse resistance from inf-mat.csv: 0 = infection (susceptible), 1 = no infection (resistant)
    res_anc_map: dict[tuple[str, float], list[float]] = {}
    res_con_map: dict[tuple[str, float], list[float]] = {}
    with open(inf_path, "r", encoding="utf-8", errors="replace") as f:
        r = list(csv.reader(f))
    host_cids = r[2][7:]
    host_days = [_fval(x) for x in r[3][7:]]

    # Ancestral phage rows have cID == 'RIM8-1'
    anc_rows = [row for row in r[7:] if len(row) > 7 and row[2].strip() == "RIM8-1"]
    # Contemporary phage rows indexed by (cID_short, day)
    phage_rows: dict[tuple[str, float], list[list[str]]] = {}
    for row in r[7:]:
        if len(row) > 7 and row[2].strip() != "RIM8-1":
            pcid = row[2].strip()  # e.g. 'N2', 'P4'
            pday = _fval(row[3])
            if not math.isnan(pday):
                phage_rows.setdefault((pcid, pday), []).append(row)

    cid_map_short = {"N1": "NL1", "N2": "NL2", "N3": "NL3", "N4": "NL4", "N5": "NL5",
                     "P1": "PL1", "P2": "PL2", "P3": "PL3", "P4": "PL4", "P5": "PL5"}

    for col_idx, (hcid_short, hday) in enumerate(zip(host_cids, host_days), start=7):
        if hcid_short not in cid_map_short or math.isnan(hday):
            continue
        full_cid = cid_map_short[hcid_short]
        # Challenge against ancestral phage RIM8-1
        vals_anc = [_fval(arow[col_idx]) for arow in anc_rows if col_idx < len(arow)]
        vals_anc = [v for v in vals_anc if not math.isnan(v)]
        if vals_anc:
            res_anc_map.setdefault((full_cid, hday), []).append(float(np.mean(vals_anc)))
        # Challenge against contemporary sympatric phage (same chemostat and same isolation day)
        sym_rows = phage_rows.get((hcid_short, hday), [])
        vals_con = [_fval(srow[col_idx]) for srow in sym_rows if col_idx < len(srow)]
        vals_con = [v for v in vals_con if not math.isnan(v)]
        if vals_con:
            res_con_map.setdefault((full_cid, hday), []).append(float(np.mean(vals_con)))

    # Parse density series from cid-means.csv (post-phage addition day >= 0; keep pre_equil in meta)
    by_cid: dict[str, dict] = {}
    with open(cid_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = _fval(row["day"])
            if math.isnan(day):
                continue
            cid = row["cID"].strip()
            lim = row["lim"].strip()
            utype = row["type"].strip()
            mic = row["microbe"].strip()
            abd = _fval(row["abd"])
            info = by_cid.setdefault(cid, {"lim": lim, "type": utype, "syn": {}, "phage": {}, "pre_syn": []})
            if day < 0:
                if mic == "Syn" and not math.isnan(abd) and day >= -30:
                    info["pre_syn"].append((day, abd))
            else:
                if mic == "Syn":
                    info["syn"][day] = abd
                elif mic == "Phage":
                    info["phage"][day] = abd

    units: list[UnitSeries] = []
    for cid in sorted(by_cid.keys()):
        info = by_cid[cid]
        infected = (info["type"] == "Infect")
        all_days = sorted(set(info["syn"].keys()) | set(info["phage"].keys()))
        t_arr = np.array(all_days, dtype=float)
        h_arr = np.array([info["syn"].get(d, math.nan) for d in all_days], dtype=float)
        p_arr = np.array([info["phage"].get(d, math.nan) for d in all_days], dtype=float)
        om_arr = np.full(len(t_arr), 1.0, dtype=float)  # 1.0 d^-1 chemostat dilution

        rdays = sorted(set(d for (c, d) in res_anc_map if c == cid and d >= -6))
        r_t = np.array(rdays, dtype=float)
        r_anc = np.array([float(np.mean(res_anc_map[(cid, d)])) for d in rdays], dtype=float)
        r_con = np.array([
            float(np.mean(res_con_map[(cid, d)])) if (cid, d) in res_con_map else float(np.mean(res_anc_map[(cid, d)]))
            for d in rdays
        ], dtype=float)

        units.append(UnitSeries(
            dataset_id="D1_larsen2019",
            unit_id=cid,
            treatment=f"{info['lim']}-limited",
            infected=infected,
            times_d=t_arr,
            host_density=h_arr,
            phage_density=p_arr,
            dilution_rate_d=om_arr,
            res_times_d=r_t,
            res_frac_anc=r_anc,
            res_frac_contemp=r_con,
            meta={"pre_addition_syn": info["pre_syn"], "mode": "continuous_chemostat"},
        ))
    return units


def load_d2_reuter2026(raw_root: str) -> tuple[list[UnitSeries], dict]:
    """Load Reuter et al. (2026) E. coli C - PhiX174 serial-transfer datasets + life-history assay checks."""
    zpath = os.path.join(raw_root, "reuter2025_phix174_serial_transfer", "Data_sets_Phage_titers.zip")
    units: list[UnitSeries] = []
    assay_checks: dict = {}

    with zipfile.ZipFile(zpath, "r") as zf:
        # Experiment 1: uniform 3-h transfers (0.125 d each), chloroform extraction + MOI ~0.1 reset
        e1_rows = list(csv.DictReader(
            zf.read("Data_sets_Phage_titers/experiment_1_3_h_transfers.csv").decode("utf-8-sig").splitlines()
        ))
        by_line1: dict[str, list[tuple[int, float, float]]] = {}
        for r in e1_rows:
            t_idx = int(r["Transfer"].strip().lstrip("T"))
            line = r["Line"].strip()
            cfu_ml = _fval(r["CFU"]) * _fval(r["Dilution_CFU"]) * 10.0
            pfu_ml = _fval(r["PFU"]) * _fval(r["Dilution_PFU"]) * 10.0
            by_line1.setdefault(line, []).append((t_idx, cfu_ml, pfu_ml))

        for line in sorted(by_line1.keys()):
            pts = sorted(by_line1[line], key=lambda x: x[0])
            K = len(pts)
            # Cumulative incubation time: 3 h = 0.125 d per transfer; include t=0 initial state (~0.1 MOI on T1 host)
            t_arr = np.arange(0, K + 1, dtype=float) * 0.125
            h_vals = [pts[0][1]] + [p[1] for p in pts]
            p0 = 0.1 * pts[0][1]  # target MOI ~ 0.1 at start of T1
            p_vals = [p0] + [p[2] for p in pts]
            # Between transfers, lysate is diluted back to ~0.1 * next_CFU_ml; effective per-day dilution over 0.125 d:
            # For transfer k>=1: input was ~0.1 * H_k, output was P_k, next input is ~0.1 * H_{k+1}, so fraction retained f_k = (0.1 * H_k) / P_{k-1}
            om_vals = [0.0]
            for k in range(K):
                prev_out = p_vals[k]
                in_k = 0.1 * pts[k][1]
                fk = min(max(in_k / max(prev_out, 1.0), 1e-9), 1.0)
                om_vals.append(-math.log(fk) / 0.125 if k > 0 else 0.0)
            units.append(UnitSeries(
                dataset_id="D2_reuter2026",
                unit_id=f"3h_{line}",
                treatment="3h_uniform",
                infected=True,
                times_d=t_arr,
                host_density=np.array(h_vals, dtype=float),
                phage_density=np.array(p_vals, dtype=float),
                dilution_rate_d=np.array(om_vals, dtype=float),
                meta={"regime": "3h", "n_transfers": K, "chloroform_each_transfer": True},
            ))

        # Experiment 2: pulsed 4 x 30-min (2% co-culture transfer, dt=1/48 d) + 1 x 3-h (2% transfer, dt=0.125 d)
        # Exclude aborted Passage 'P2', retain replacement 'P2b' per deposit documentation.
        e2_rows = list(csv.DictReader(
            zf.read("Data_sets_Phage_titers/experiment_2_30_min_transfers.csv").decode("utf-8-sig").splitlines()
        ))
        by_line2: dict[str, list[dict]] = {}
        for r in e2_rows:
            if r["Passage"].strip() == "P2":
                continue
            by_line2.setdefault(r["Line"].strip(), []).append(r)

        for line in sorted(by_line2.keys()):
            rows_l = sorted(by_line2[line], key=lambda x: int(x["Passage_ID"]))
            t_list = [0.0]
            h_list = [_fval(rows_l[0]["CFU_ml"])]
            p_list = [_fval(rows_l[0]["Real_virus_concentration"])]
            om_list = [0.0]
            cur_t = 0.0
            step_types = []
            f_dilutions = []
            for idx, r in enumerate(rows_l):
                tr = r["Transfer"].strip()
                dt = (30.0 / 1440.0) if tr in ("T1", "T2", "T3", "T4") else (180.0 / 1440.0)
                cur_t += dt
                h_k = _fval(r["CFU_ml"])
                p_out = _fval(r["Output_phage_ml"])
                v_in = _fval(r["Inoculation_volume_ml"])  # out of 5.0 mL culture
                # For T2..T5, 0.1 mL into 5.0 mL -> f = 0.02; for T1, v_in / 5.0 mL of previous T5 chloroform lysate
                fk = min(max((v_in / 5.0) if not math.isnan(v_in) else 0.02, 1e-8), 1.0)
                om_k = -math.log(fk) / dt if idx > 0 else 0.0
                t_list.append(cur_t)
                h_list.append(h_k)
                p_list.append(p_out)
                om_list.append(om_k)
                step_types.append(tr)
                f_dilutions.append(fk)
            units.append(UnitSeries(
                dataset_id="D2_reuter2026",
                unit_id=f"30min_{line}",
                treatment="30min_pulsed",
                infected=True,
                times_d=np.array(t_list, dtype=float),
                host_density=np.array(h_list, dtype=float),
                phage_density=np.array(p_list, dtype=float),
                dilution_rate_d=np.array(om_list, dtype=float),
                meta={"regime": "30min_pulsed", "step_types": step_types, "f_dilutions": f_dilutions},
            ))

        # Auxiliary direct check of Table 1 traits from experiment_3 (adsorption) and experiment_7 (OSGC)
        e3_rows = list(csv.DictReader(
            zf.read("Data_sets_Phage_titers/experiment_3_adsorption_assay.csv").decode("utf-8-sig").splitlines()
        ))
        wt_slopes = []
        by_run: dict[str, list[tuple[float, float, float]]] = {}
        for r in e3_rows:
            if r["Genotype"].strip() == "WT":
                by_run.setdefault(r["Run"].strip(), []).append(
                    (_fval(r["Sampling_time_min"]), _fval(r["PFU_ml"]), _fval(r["CFU_ml"]))
                )
        for run_id, pts in by_run.items():
            pts = sorted(pts, key=lambda x: x[0])
            ts = np.array([p[0] for p in pts])
            ys = np.log(np.array([p[1] for p in pts]))
            h0 = pts[0][2]
            slope = -float(np.polyfit(ts, ys, 1)[0]) / h0  # mL min^-1
            wt_slopes.append(slope)
        assay_checks["WT_adsorption_mean_ml_min"] = float(np.mean(wt_slopes))
        assay_checks["WT_adsorption_n_runs"] = len(wt_slopes)

    return units, assay_checks


def load_d3_chaudhry2018(raw_root: str) -> tuple[list[UnitSeries], dict]:
    """Load Chaudhry et al. (2018) E. coli B - lambdaVIR serial-transfer datasets."""
    s008 = os.path.join(raw_root, "chaudhry2018_lambda_serial_transfer", "pbio.2005971.s008.xlsx")
    rows_f2 = read_xlsx_sheet_dicts(s008, sheet_index=1)  # Figure 2
    rows_fs3 = read_xlsx_sheet_dicts(s008, sheet_index=6)  # Figure S3

    units: list[UnitSeries] = []
    om_daily = -math.log(0.01)  # 1:100 daily serial transfer -> 4.60517 d^-1

    # LB block: row index 1 has times in B..H (host) and K..Q (phage); rows 2..12 have Control + B_LB1..10 / P_LB1..10
    lb_cols_h = ["B", "C", "D", "E", "F", "G", "H"]
    lb_cols_p = ["K", "L", "M", "N", "O", "P", "Q"]
    t_lb = np.array([_fval(rows_f2[1].get(c)) for c in lb_cols_h], dtype=float)

    # Control LB
    h_ctrl_lb = np.array([_fval(rows_f2[2].get(c)) for c in lb_cols_h], dtype=float)
    units.append(UnitSeries(
        dataset_id="D3_chaudhry2018",
        unit_id="LB_Control",
        treatment="LB",
        infected=False,
        times_d=t_lb,
        host_density=h_ctrl_lb,
        phage_density=np.full(len(t_lb), math.nan),
        dilution_rate_d=np.where(t_lb == 0, 0.0, om_daily),
        meta={"medium": "LB"},
    ))

    host_lb: dict[int, np.ndarray] = {}
    phage_lb: dict[int, np.ndarray] = {}
    for r in rows_f2[2:13]:
        if r.get("A", "").startswith("B_LB"):
            idx = int(r["A"].replace("B_LB", ""))
            host_lb[idx] = np.array([_fval(r.get(c)) for c in lb_cols_h], dtype=float)
        if r.get("J", "").startswith("P_LB"):
            idx = int(r["J"].replace("P_LB", ""))
            phage_lb[idx] = np.array([_fval(r.get(c)) for c in lb_cols_p], dtype=float)
    for idx in sorted(host_lb.keys()):
        units.append(UnitSeries(
            dataset_id="D3_chaudhry2018",
            unit_id=f"LB_{idx}",
            treatment="LB",
            infected=True,
            times_d=t_lb,
            host_density=host_lb[idx],
            phage_density=phage_lb[idx],
            dilution_rate_d=np.where(t_lb == 0, 0.0, om_daily),
            meta={"medium": "LB"},
        ))

    # M9M block: row index 15 has times in B..G (host) and K..P (phage); rows 16..28 have Control + B_M9M1..12 / P_M9M1..12
    m9_cols_h = ["B", "C", "D", "E", "F", "G"]
    m9_cols_p = ["K", "L", "M", "N", "O", "P"]
    t_m9 = np.array([_fval(rows_f2[15].get(c)) for c in m9_cols_h], dtype=float)

    h_ctrl_m9 = np.array([_fval(rows_f2[16].get(c)) for c in m9_cols_h], dtype=float)
    units.append(UnitSeries(
        dataset_id="D3_chaudhry2018",
        unit_id="M9M_Control",
        treatment="M9M",
        infected=False,
        times_d=t_m9,
        host_density=h_ctrl_m9,
        phage_density=np.full(len(t_m9), math.nan),
        dilution_rate_d=np.where(t_m9 == 0, 0.0, om_daily),
        meta={"medium": "M9M"},
    ))

    host_m9: dict[int, np.ndarray] = {}
    phage_m9: dict[int, np.ndarray] = {}
    for r in rows_f2[16:29]:
        if r.get("A", "").startswith("B_M9M"):
            idx = int(r["A"].replace("B_M9M", ""))
            host_m9[idx] = np.array([_fval(r.get(c)) for c in m9_cols_h], dtype=float)
        if r.get("J", "").startswith("P_M9M"):
            idx = int(r["J"].replace("P_M9M", ""))
            phage_m9[idx] = np.array([_fval(r.get(c)) for c in m9_cols_p], dtype=float)
    for idx in sorted(host_m9.keys()):
        units.append(UnitSeries(
            dataset_id="D3_chaudhry2018",
            unit_id=f"M9M_{idx}",
            treatment="M9M",
            infected=True,
            times_d=t_m9,
            host_density=host_m9[idx],
            phage_density=phage_m9[idx],
            dilution_rate_d=np.where(t_m9 == 0, 0.0, om_daily),
            meta={"medium": "M9M"},
        ))

    # No-host washout check from Figure S3 (days 0..3 before hitting floor 1)
    nohost_slopes = []
    for r in rows_fs3:
        if r.get("A", "").startswith("No host"):
            vals = [_fval(r.get(c)) for c in ["B", "C", "D", "E"]]
            pos = [(t_i, v) for t_i, v in enumerate(vals) if not math.isnan(v) and v > 10.0]
            if len(pos) >= 3:
                ts = np.array([p[0] for p in pos], dtype=float)
                ys = np.log(np.array([p[1] for p in pos], dtype=float))
                nohost_slopes.append(-float(np.polyfit(ts, ys, 1)[0]))
    aux = {
        "nohost_loss_rate_mean_d": float(np.mean(nohost_slopes)),
        "expected_1to100_dilution_rate_d": om_daily,
    }
    return units, aux


def load_d4_duncan2017(raw_root: str) -> list[UnitSeries]:
    """Load Duncan et al. (2017) P. fluorescens SBW25 - phi2 variable-temperature microcosms."""
    fpath = os.path.join(raw_root, "duncan2017_sbw25_temp_fluctuation", "Duncan_JEB.xlsx")
    rows_d = read_xlsx_sheet_dicts(fpath, sheet_index=0)  # Bacteria_Phage_Density
    rows_r = read_xlsx_sheet_dicts(fpath, sheet_index=1)  # Sheet2 (time-shift resistance)

    # Contemporary resistance: Combination == 'CB/CP' at Centre_Time_Pt in {4, 8, 12}
    res_by_pop: dict[str, dict[float, list[float]]] = {}
    for r in rows_r[1:]:
        if r.get("G", "").strip() == "CB/CP":
            pop = str(int(float(r["A"])))
            t_pt = _fval(r.get("E"))
            prop = _fval(r.get("J"))
            if not math.isnan(t_pt) and not math.isnan(prop):
                res_by_pop.setdefault(pop, {}).setdefault(t_pt, []).append(prop)

    by_pop: dict[str, dict] = {}
    for r in rows_d[1:]:
        if not r.get("A"):
            continue
        pop = str(int(float(r["A"])))
        fluct = int(float(r["B"]))
        stemp = int(float(r["C"]))
        infected = (r.get("D", "").strip() == "Infected")
        t_d = _fval(r.get("E"))
        b_ml = _fval(r.get("F"))
        p_ml = _fval(r.get("H"))
        info = by_pop.setdefault(pop, {
            "fluct": fluct, "start_temp": stemp, "infected": infected, "pts": []
        })
        info["pts"].append((t_d, b_ml, p_ml))

    om_transfer = -math.log(0.01) / 2.0  # 1% serial transfer every 48 h -> 2.302585 d^-1
    units: list[UnitSeries] = []
    for pop in sorted(by_pop.keys(), key=lambda x: int(x)):
        info = by_pop[pop]
        pts = sorted(info["pts"], key=lambda x: x[0])
        t_arr = np.array([p[0] for p in pts], dtype=float)
        h_arr = np.array([p[1] for p in pts], dtype=float)
        p_arr = np.array([p[2] for p in pts], dtype=float)
        rmap = res_by_pop.get(pop, {})
        rt_arr = np.array(sorted(rmap.keys()), dtype=float)
        rc_arr = np.array([float(np.mean(rmap[k])) for k in sorted(rmap.keys())], dtype=float)
        units.append(UnitSeries(
            dataset_id="D4_duncan2017",
            unit_id=f"Pop_{pop}",
            treatment=f"freq{info['fluct']}_T{info['start_temp']}",
            infected=info["infected"],
            times_d=t_arr,
            host_density=h_arr,
            phage_density=p_arr,
            dilution_rate_d=np.where(t_arr == 0, 0.0, om_transfer),
            res_times_d=rt_arr,
            res_frac_anc=rc_arr,
            res_frac_contemp=rc_arr,
            meta={"fluct_freq_d": info["fluct"], "start_temp_C": info["start_temp"]},
        ))
    return units


def load_d5_zhao2016(raw_root: str, od_to_cfu: float = 5.0e8) -> tuple[list[UnitSeries], dict]:
    """Load Zhao, Hao & Zhang (2016) P. fluorescens SBW25 - phi2 across 3 nutrient levels."""
    base = os.path.join(raw_root, "zhao2016_sbw25_productivity")
    rows_den = read_xlsx_sheet_dicts(os.path.join(base, "Population_density.xlsx"), sheet_index=0)
    rows_tp = read_xlsx_sheet_dicts(os.path.join(base, "Three_productivities.xlsx"), sheet_index=0)
    rows_ch = read_xlsx_sheet_dicts(os.path.join(base, "Challenge_data.xlsx"), sheet_index=0)

    # Uninfected carrying-capacity OD from Three_productivities.xlsx
    uninf_od: dict[str, float] = {}
    for r in rows_tp[1:]:
        line = r.get("B", "").strip()
        lod = _fval(r.get("C"))
        if line and not math.isnan(lod):
            uninf_od[line] = (10.0 ** lod) * od_to_cfu

    # Contemporary resistance from Challenge_data.xlsx: PhageTransfer (col D) == 'N', Resistant Rate in col Y
    res_map: dict[str, dict[float, float]] = {}
    for r in rows_ch[1:]:
        if r.get("D", "").strip() == "N":
            nut_code = r.get("A", "").strip()  # H, M, L
            rep = r.get("C", "").strip()
            tr_num = _fval(r.get("B"))
            rr = _fval(r.get("Y"))
            if nut_code and rep and not math.isnan(tr_num) and not math.isnan(rr):
                line_id = f"{nut_code}{int(float(rep))}"
                res_map.setdefault(line_id, {})[2.0 * tr_num] = rr

    by_line: dict[str, dict] = {}
    for r in rows_den[1:]:
        if not r.get("C"):
            continue
        tr = _fval(r.get("A"))
        nut = r.get("B", "").strip()
        line = r.get("C", "").strip()
        lpfu1 = _fval(r.get("D"))
        od = _fval(r.get("E"))
        pfu = (10.0 ** lpfu1 - 1.0) if not math.isnan(lpfu1) else math.nan
        pfu = max(0.0, pfu) if not math.isnan(pfu) else math.nan
        cfu = od * od_to_cfu if not math.isnan(od) else math.nan
        info = by_line.setdefault(line, {"nut": nut, "pts": []})
        info["pts"].append((2.0 * tr, cfu, pfu))  # 48 h = 2 d per transfer

    om_transfer = -math.log(0.01) / 2.0
    units: list[UnitSeries] = []
    for line in sorted(by_line.keys(), key=lambda s: (s[0], int(s[1:]))):
        info = by_line[line]
        pts = sorted(info["pts"], key=lambda x: x[0])
        t_arr = np.array([p[0] for p in pts], dtype=float)
        h_arr = np.array([p[1] for p in pts], dtype=float)
        p_arr = np.array([p[2] for p in pts], dtype=float)
        rmap = res_map.get(line, {})
        rt_arr = np.array(sorted(rmap.keys()), dtype=float)
        rc_arr = np.array([rmap[k] for k in sorted(rmap.keys())], dtype=float)
        units.append(UnitSeries(
            dataset_id="D5_zhao2016",
            unit_id=line,
            treatment=info["nut"],
            infected=True,
            times_d=t_arr,
            host_density=h_arr,
            phage_density=p_arr,
            dilution_rate_d=np.where(t_arr == 0, 0.0, om_transfer),
            res_times_d=rt_arr,
            res_frac_anc=rc_arr,
            res_frac_contemp=rc_arr,
            meta={"uninfected_baseline_cfu": uninf_od.get(line, math.nan), "od_to_cfu": od_to_cfu},
        ))
    return units, {"uninfected_baseline_count": len(uninf_od), "od_to_cfu": od_to_cfu}
