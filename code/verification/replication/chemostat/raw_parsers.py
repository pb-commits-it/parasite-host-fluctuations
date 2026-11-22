#!/usr/bin/env python3
"""Independent raw-data parsers for `r36-rep-chemostat`.

Reads raw open files in `data/raw/chemostat/` directly using standard-library CSV,
ZIP, and OpenXML parsers, implementing the registered rules of `protocol.md`
(§3, §6.1, Amendment 1, and Amendment 3). Zero code imported or copied from
`casestudies/chemostat/`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np


@dataclass
class TrajectoryRecord:
    dataset_id: str
    unit_id: str
    treatment: str
    infected: bool
    times_d: np.ndarray
    host_cells_ml: np.ndarray
    phage_pfu_ml: np.ndarray
    washout_d: np.ndarray
    res_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    res_ancestral: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    res_contemporary: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    extra: dict = field(default_factory=dict)


def parse_num(text: str | None) -> float:
    if text is None:
        return math.nan
    t = text.strip()
    if t in ("", "NA", "NaN", "nan", "N/A", "."):
        return math.nan
    try:
        return float(t)
    except ValueError:
        return math.nan


def load_xlsx_workbook(path: Path) -> dict[str, list[dict[str, str]]]:
    """Parse all worksheets of an OpenXML `.xlsx` archive into `{sheet_name: [col_dict, ...]}`."""
    ns_m = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheets_out: dict[str, list[dict[str, str]]] = {}
    with zipfile.ZipFile(path, "r") as zf:
        sst: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in sroot.findall(f"{{{ns_m}}}si"):
                sst.append("".join(el.text or "" for el in item.iter(f"{{{ns_m}}}t")))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_lookup = {el.attrib["Id"]: el.attrib["Target"] for el in rels}

        for sh in wb.findall(f".//{{{ns_m}}}sheet"):
            title = sh.attrib["name"]
            target = rel_lookup[sh.attrib[f"{{{ns_r}}}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            tree = ET.fromstring(zf.read(target))
            rows: list[dict[str, str]] = []
            for row_el in tree.findall(f".//{{{ns_m}}}sheetData/{{{ns_m}}}row"):
                cells: dict[str, str] = {}
                for c_el in row_el.findall(f"{{{ns_m}}}c"):
                    coord = c_el.attrib.get("r", "")
                    col_let = "".join(ch for ch in coord if ch.isalpha())
                    kind = c_el.attrib.get("t", "")
                    v_el = c_el.find(f"{{{ns_m}}}v")
                    i_el = c_el.find(f"{{{ns_m}}}is")
                    if kind == "s" and v_el is not None and v_el.text is not None:
                        idx = int(v_el.text)
                        cells[col_let] = sst[idx] if 0 <= idx < len(sst) else ""
                    elif kind == "inlineStr" and i_el is not None:
                        cells[col_let] = "".join(x.text or "" for x in i_el.iter(f"{{{ns_m}}}t"))
                    elif v_el is not None and v_el.text is not None:
                        cells[col_let] = v_el.text
                rows.append(cells)
            sheets_out[title] = rows
    return sheets_out


def parse_d1_larsen(raw_dir: Path) -> list[TrajectoryRecord]:
    """Parse D1: Larsen, Wilhelm & Lennon (2019) Synechococcus WH7803 + S-RIM8 chemostats."""
    folder = raw_dir / "larsen2019_synechococcus_chemostat"

    # 1. Resistance matrix `inf-mat.csv`: 0 = susceptible, 1 = resistant
    short_to_full = {f"{p}{i}": f"{p}L{i}" for p in ("N", "P") for i in range(1, 6)}
    with (folder / "inf-mat.csv").open("r", encoding="utf-8", errors="replace") as fh:
        mat = list(csv.reader(fh))

    col_cids = mat[2][7:]
    col_days = [parse_num(x) for x in mat[3][7:]]
    anc_phage_rows = [r for r in mat[7:] if len(r) > 7 and r[2].strip() == "RIM8-1"]
    sym_phage_rows: dict[tuple[str, float], list[list[str]]] = {}
    for r in mat[7:]:
        if len(r) > 7 and r[2].strip() != "RIM8-1":
            d_val = parse_num(r[3])
            if not math.isnan(d_val):
                sym_phage_rows.setdefault((r[2].strip(), d_val), []).append(r)

    anc_by_cid_day: dict[tuple[str, float], list[float]] = {}
    con_by_cid_day: dict[tuple[str, float], list[float]] = {}
    for c_offset, (scid, hday) in enumerate(zip(col_cids, col_days)):
        col_i = 7 + c_offset
        if scid not in short_to_full or math.isnan(hday):
            continue
        fcid = short_to_full[scid]
        a_vals = [parse_num(r[col_i]) for r in anc_phage_rows if col_i < len(r)]
        a_vals = [v for v in a_vals if not math.isnan(v)]
        if a_vals:
            anc_by_cid_day.setdefault((fcid, hday), []).append(float(np.mean(a_vals)))

        s_rows = sym_phage_rows.get((scid, hday), [])
        c_vals = [parse_num(r[col_i]) for r in s_rows if col_i < len(r)]
        c_vals = [v for v in c_vals if not math.isnan(v)]
        if c_vals:
            con_by_cid_day.setdefault((fcid, hday), []).append(float(np.mean(c_vals)))

    # 2. Population counts `cid-means.csv` (post-addition day >= 0)
    grouped: dict[str, dict] = {}
    with (folder / "cid-means.csv").open("r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            day = parse_num(row["day"])
            if math.isnan(day) or day < 0.0:
                continue
            cid = row["cID"].strip()
            entry = grouped.setdefault(cid, {
                "lim": row["lim"].strip(),
                "infected": (row["type"].strip() == "Infect"),
                "syn": {},
                "phage": {},
            })
            val = parse_num(row["abd"])
            if row["microbe"].strip() == "Syn":
                entry["syn"][day] = val
            elif row["microbe"].strip() == "Phage":
                entry["phage"][day] = val

    records: list[TrajectoryRecord] = []
    for cid in sorted(grouped):
        g = grouped[cid]
        days = sorted(set(g["syn"]) | set(g["phage"]))
        t_vec = np.array(days, dtype=float)
        h_vec = np.array([g["syn"].get(d, math.nan) for d in days], dtype=float)
        p_vec = np.array([g["phage"].get(d, math.nan) for d in days], dtype=float)
        w_vec = np.ones(len(days), dtype=float)  # 1.0 d^-1 continuous chemostat dilution

        r_days = sorted(d for (c, d) in anc_by_cid_day if c == cid and d >= -6.0)
        r_anc = np.array([float(np.mean(anc_by_cid_day[(cid, d)])) for d in r_days], dtype=float)
        r_con = np.array([
            float(np.mean(con_by_cid_day[(cid, d)])) if (cid, d) in con_by_cid_day else float(np.mean(anc_by_cid_day[(cid, d)]))
            for d in r_days
        ], dtype=float)

        records.append(TrajectoryRecord(
            dataset_id="D1_larsen2019",
            unit_id=cid,
            treatment=f"{g['lim']}-limited",
            infected=g["infected"],
            times_d=t_vec,
            host_cells_ml=h_vec,
            phage_pfu_ml=p_vec,
            washout_d=w_vec,
            res_days=np.array(r_days, dtype=float),
            res_ancestral=r_anc,
            res_contemporary=r_con,
        ))
    return records


def parse_d2_reuter(raw_dir: Path) -> tuple[list[TrajectoryRecord], dict[str, float]]:
    """Parse D2: Reuter et al. (2026) E. coli C + PhiX174 uniform 3h and pulsed 30min transfers."""
    zfile = raw_dir / "reuter2025_phix174_serial_transfer" / "Data_sets_Phage_titers.zip"
    records: list[TrajectoryRecord] = []
    checks: dict[str, float] = {}

    with zipfile.ZipFile(zfile, "r") as zf:
        # 1. Uniform 3-hour transfers (dt = 0.125 d)
        text_3h = zf.read("Data_sets_Phage_titers/experiment_1_3_h_transfers.csv").decode("utf-8-sig")
        by_line_3h: dict[str, list[tuple[int, float, float]]] = {}
        for row in csv.DictReader(text_3h.splitlines()):
            tr_num = int(row["Transfer"].strip().lstrip("T"))
            ln = row["Line"].strip()
            cfu_ml = parse_num(row["CFU"]) * parse_num(row["Dilution_CFU"]) * 10.0
            pfu_ml = parse_num(row["PFU"]) * parse_num(row["Dilution_PFU"]) * 10.0
            by_line_3h.setdefault(ln, []).append((tr_num, cfu_ml, pfu_ml))

        for ln in sorted(by_line_3h):
            seq = sorted(by_line_3h[ln], key=lambda item: item[0])
            k_tr = len(seq)
            times = np.arange(0, k_tr + 1, dtype=float) * 0.125
            hosts = np.array([seq[0][1]] + [item[1] for item in seq], dtype=float)
            phages = np.array([0.1 * seq[0][1]] + [item[2] for item in seq], dtype=float)
            wash = [0.0]
            f_sched = []
            for idx in range(k_tr):
                f_ret = min(max((0.1 * seq[idx][1]) / max(phages[idx], 1.0), 1e-9), 1.0)
                f_sched.append(f_ret)
                wash.append(-math.log(f_ret) / 0.125 if idx > 0 else 0.0)

            records.append(TrajectoryRecord(
                dataset_id="D2_reuter2026",
                unit_id=f"3h_{ln}",
                treatment="3h_uniform",
                infected=True,
                times_d=times,
                host_cells_ml=hosts,
                phage_pfu_ml=phages,
                washout_d=np.array(wash, dtype=float),
                extra={"regime": "3h", "f_dilutions": f_sched, "step_types": [f"T{i+1}" for i in range(k_tr)]},
            ))

        # 2. Pulsed 4x30-min + 1x3-h transfers (50 transfers, P2 excluded in favor of P2b per Amendment 1)
        text_30m = zf.read("Data_sets_Phage_titers/experiment_2_30_min_transfers.csv").decode("utf-8-sig")
        by_line_30m: dict[str, list[dict[str, str]]] = {}
        for row in csv.DictReader(text_30m.splitlines()):
            if row["Passage"].strip() == "P2":
                continue
            by_line_30m.setdefault(row["Line"].strip(), []).append(row)

        for ln in sorted(by_line_30m):
            rows_ord = sorted(by_line_30m[ln], key=lambda r: int(r["Passage_ID"]))
            times_l = [0.0]
            hosts_l = [parse_num(rows_ord[0]["CFU_ml"])]
            phages_l = [parse_num(rows_ord[0]["Real_virus_concentration"])]
            wash_l = [0.0]
            f_sched = []
            stypes = []
            t_acc = 0.0
            for idx, r in enumerate(rows_ord):
                tr_code = r["Transfer"].strip()
                dt = (30.0 / 1440.0) if tr_code in ("T1", "T2", "T3", "T4") else (180.0 / 1440.0)
                t_acc += dt
                vin = parse_num(r["Inoculation_volume_ml"])
                f_k = min(max((vin / 5.0) if not math.isnan(vin) else 0.02, 1e-8), 1.0)
                wash_k = -math.log(f_k) / dt if idx > 0 else 0.0
                times_l.append(t_acc)
                hosts_l.append(parse_num(r["CFU_ml"]))
                phages_l.append(parse_num(r["Output_phage_ml"]))
                wash_l.append(wash_k)
                f_sched.append(f_k)
                stypes.append(tr_code)

            records.append(TrajectoryRecord(
                dataset_id="D2_reuter2026",
                unit_id=f"30min_{ln}",
                treatment="30min_pulsed",
                infected=True,
                times_d=np.array(times_l, dtype=float),
                host_cells_ml=np.array(hosts_l, dtype=float),
                phage_pfu_ml=np.array(phages_l, dtype=float),
                washout_d=np.array(wash_l, dtype=float),
                extra={"regime": "30min_pulsed", "f_dilutions": f_sched, "step_types": stypes},
            ))

        # 3. Auxiliary WT adsorption assay check (`experiment_3_adsorption_assay.csv`)
        text_ads = zf.read("Data_sets_Phage_titers/experiment_3_adsorption_assay.csv").decode("utf-8-sig")
        wt_runs: dict[str, list[tuple[float, float, float]]] = {}
        for row in csv.DictReader(text_ads.splitlines()):
            if row["Genotype"].strip() == "WT":
                wt_runs.setdefault(row["Run"].strip(), []).append((
                    parse_num(row["Sampling_time_min"]),
                    parse_num(row["PFU_ml"]),
                    parse_num(row["CFU_ml"]),
                ))
        ads_ests = []
        for pts in wt_runs.values():
            pts_s = sorted(pts, key=lambda x: x[0])
            t_min = np.array([x[0] for x in pts_s], dtype=float)
            ln_pfu = np.log(np.array([x[1] for x in pts_s], dtype=float))
            slope = float(np.polyfit(t_min, ln_pfu, 1)[0])
            ads_ests.append(-slope / pts_s[0][2])
        checks["wt_adsorption_mean_ml_min"] = float(np.mean(ads_ests))
        checks["wt_adsorption_n_runs"] = len(ads_ests)

    return records, checks


def parse_d3_chaudhry(raw_dir: Path) -> tuple[list[TrajectoryRecord], dict[str, float]]:
    """Parse D3: Chaudhry et al. (2018) E. coli + lambdaVIR in LB and M9M (`pbio.2005971.s008.xlsx`)."""
    wb = load_xlsx_workbook(raw_dir / "chaudhry2018_lambda_serial_transfer" / "pbio.2005971.s008.xlsx")
    fig2 = wb["Figure 2"]
    figs3 = wb["Figure S3"]
    wash_100 = -math.log(0.01)  # 1:100 daily serial transfer
    records: list[TrajectoryRecord] = []

    # LB block (days 0, 1, 2, 3, 5, 7, 9)
    cols_lb_b = ["B", "C", "D", "E", "F", "G", "H"]
    cols_lb_p = ["K", "L", "M", "N", "O", "P", "Q"]
    t_lb = np.array([parse_num(fig2[1].get(c)) for c in cols_lb_b], dtype=float)
    w_lb = np.where(t_lb == 0.0, 0.0, wash_100)

    records.append(TrajectoryRecord(
        dataset_id="D3_chaudhry2018",
        unit_id="LB_Control",
        treatment="LB",
        infected=False,
        times_d=t_lb,
        host_cells_ml=np.array([parse_num(fig2[2].get(c)) for c in cols_lb_b], dtype=float),
        phage_pfu_ml=np.full(len(t_lb), math.nan),
        washout_d=w_lb,
    ))
    b_lb_map: dict[int, np.ndarray] = {}
    p_lb_map: dict[int, np.ndarray] = {}
    for row in fig2[2:13]:
        lbl_b = row.get("A", "").strip()
        lbl_p = row.get("J", "").strip()
        if lbl_b.startswith("B_LB"):
            b_lb_map[int(lbl_b.replace("B_LB", ""))] = np.array([parse_num(row.get(c)) for c in cols_lb_b], dtype=float)
        if lbl_p.startswith("P_LB"):
            p_lb_map[int(lbl_p.replace("P_LB", ""))] = np.array([parse_num(row.get(c)) for c in cols_lb_p], dtype=float)
    for idx in sorted(b_lb_map):
        records.append(TrajectoryRecord(
            dataset_id="D3_chaudhry2018",
            unit_id=f"LB_{idx}",
            treatment="LB",
            infected=True,
            times_d=t_lb,
            host_cells_ml=b_lb_map[idx],
            phage_pfu_ml=p_lb_map[idx],
            washout_d=w_lb,
        ))

    # M9M block (days 0, 1, 2, 3, 5, 7)
    cols_m9_b = ["B", "C", "D", "E", "F", "G"]
    cols_m9_p = ["K", "L", "M", "N", "O", "P"]
    t_m9 = np.array([parse_num(fig2[15].get(c)) for c in cols_m9_b], dtype=float)
    w_m9 = np.where(t_m9 == 0.0, 0.0, wash_100)

    records.append(TrajectoryRecord(
        dataset_id="D3_chaudhry2018",
        unit_id="M9M_Control",
        treatment="M9M",
        infected=False,
        times_d=t_m9,
        host_cells_ml=np.array([parse_num(fig2[16].get(c)) for c in cols_m9_b], dtype=float),
        phage_pfu_ml=np.full(len(t_m9), math.nan),
        washout_d=w_m9,
    ))
    b_m9_map: dict[int, np.ndarray] = {}
    p_m9_map: dict[int, np.ndarray] = {}
    for row in fig2[16:29]:
        lbl_b = row.get("A", "").strip()
        lbl_p = row.get("J", "").strip()
        if lbl_b.startswith("B_M9M"):
            b_m9_map[int(lbl_b.replace("B_M9M", ""))] = np.array([parse_num(row.get(c)) for c in cols_m9_b], dtype=float)
        if lbl_p.startswith("P_M9M"):
            p_m9_map[int(lbl_p.replace("P_M9M", ""))] = np.array([parse_num(row.get(c)) for c in cols_m9_p], dtype=float)
    for idx in sorted(b_m9_map):
        records.append(TrajectoryRecord(
            dataset_id="D3_chaudhry2018",
            unit_id=f"M9M_{idx}",
            treatment="M9M",
            infected=True,
            times_d=t_m9,
            host_cells_ml=b_m9_map[idx],
            phage_pfu_ml=p_m9_map[idx],
            washout_d=w_m9,
        ))

    # Auxiliary cell-free washout check (`Figure S3`)
    slopes = []
    for row in figs3:
        if row.get("A", "").strip().startswith("No host"):
            pts = [(i, parse_num(row.get(c))) for i, c in enumerate(["B", "C", "D", "E"])]
            pts = [(t_i, v) for t_i, v in pts if not math.isnan(v) and v > 10.0]
            if len(pts) >= 3:
                slopes.append(-float(np.polyfit([x[0] for x in pts], np.log([x[1] for x in pts]), 1)[0]))
    return records, {"nohost_washout_mean_d": float(np.mean(slopes)), "dilution_1to100_d": wash_100}


def parse_d4_duncan(raw_dir: Path) -> list[TrajectoryRecord]:
    """Parse D4: Duncan et al. (2017) P. fluorescens SBW25 + phi2 under thermal switching."""
    wb = load_xlsx_workbook(raw_dir / "duncan2017_sbw25_temp_fluctuation" / "Duncan_JEB.xlsx")
    den_rows = wb["Bacteria_Phage_Density"]
    res_rows = wb["Sheet2"]

    res_lookup: dict[int, dict[float, list[float]]] = {}
    for row in res_rows[1:]:
        if row.get("G", "").strip() == "CB/CP":
            p_id = int(float(row["A"]))
            day = parse_num(row.get("E"))
            prop = parse_num(row.get("J"))
            if not math.isnan(day) and not math.isnan(prop):
                res_lookup.setdefault(p_id, {}).setdefault(day, []).append(prop)

    pop_data: dict[int, dict] = {}
    for row in den_rows[1:]:
        if not row.get("A"):
            continue
        p_id = int(float(row["A"]))
        freq = int(float(row["B"]))
        t_start = int(float(row["C"]))
        is_inf = (row.get("D", "").strip() == "Infected")
        day = parse_num(row.get("E"))
        b_den = parse_num(row.get("F"))
        p_den = parse_num(row.get("H"))
        rec = pop_data.setdefault(p_id, {"freq": freq, "start_C": t_start, "infected": is_inf, "obs": []})
        rec["obs"].append((day, b_den, p_den))

    wash_48h = -math.log(0.01) / 2.0
    records: list[TrajectoryRecord] = []
    for p_id in sorted(pop_data):
        info = pop_data[p_id]
        obs_s = sorted(info["obs"], key=lambda x: x[0])
        t_vec = np.array([x[0] for x in obs_s], dtype=float)
        h_vec = np.array([x[1] for x in obs_s], dtype=float)
        p_vec = np.array([x[2] for x in obs_s], dtype=float)
        w_vec = np.where(t_vec == 0.0, 0.0, wash_48h)

        rmap = res_lookup.get(p_id, {})
        rdays = np.array(sorted(rmap), dtype=float)
        rvals = np.array([float(np.mean(rmap[d])) for d in sorted(rmap)], dtype=float)

        # Reconstruct full 16-day (8 x 2-day) programmed incubator schedule per Amendment 3
        freq = info["freq"]
        start_c = info["start_C"]
        sched_2d = [start_c if (freq == 0 or (d // freq) % 2 == 0) else (60 - start_c) for d in range(0, 16, 2)]

        records.append(TrajectoryRecord(
            dataset_id="D4_duncan2017",
            unit_id=f"Pop_{p_id}",
            treatment=f"freq{freq}_T{start_c}",
            infected=info["infected"],
            times_d=t_vec,
            host_cells_ml=h_vec,
            phage_pfu_ml=p_vec,
            washout_d=w_vec,
            res_days=rdays,
            res_ancestral=rvals,
            res_contemporary=rvals,
            extra={
                "freq_d": freq,
                "start_C": start_c,
                "schedule_2d_C": sched_2d,
                "assigned_temp_mean_C": float(np.mean(sched_2d)),
                "assigned_temp_sd_C": float(np.std(sched_2d, ddof=0)),
                "weather_group": f"constant_{start_c}C" if freq == 0 else f"switch_{freq}d",
            },
        ))
    return records


def parse_d5_zhao(raw_dir: Path, od_cfu_factor: float = 5.0e8) -> list[TrajectoryRecord]:
    """Parse D5: Zhao, Hao & Zhang (2016) P. fluorescens SBW25 + phi2 across 3 nutrient levels."""
    folder = raw_dir / "zhao2016_sbw25_productivity"
    den_rows = load_xlsx_workbook(folder / "Population_density.xlsx")["density"]
    tp_rows = load_xlsx_workbook(folder / "Three_productivities.xlsx")["Sheet1"]
    ch_rows = load_xlsx_workbook(folder / "Challenge_data.xlsx")["Sheet1"]

    uninf_base: dict[str, float] = {}
    for row in tp_rows[1:]:
        ln = row.get("B", "").strip()
        lod = parse_num(row.get("C"))
        if ln and not math.isnan(lod):
            uninf_base[ln] = (10.0 ** lod) * od_cfu_factor

    res_by_line: dict[str, dict[float, float]] = {}
    for row in ch_rows[1:]:
        if row.get("D", "").strip() == "N":
            nut_c = row.get("A", "").strip()
            tr = parse_num(row.get("B"))
            rep = row.get("C", "").strip()
            rr = parse_num(row.get("Y"))
            if nut_c and rep and not math.isnan(tr) and not math.isnan(rr):
                lid = f"{nut_c}{int(float(rep))}"
                res_by_line.setdefault(lid, {})[2.0 * tr] = rr

    lines_map: dict[str, dict] = {}
    for row in den_rows[1:]:
        lid = row.get("C", "").strip()
        if not lid:
            continue
        tr = parse_num(row.get("A"))
        nut = row.get("B", "").strip()
        lpfu = parse_num(row.get("D"))
        od = parse_num(row.get("E"))
        pfu = max(0.0, 10.0 ** lpfu - 1.0) if not math.isnan(lpfu) else math.nan
        cfu = od * od_cfu_factor if not math.isnan(od) else math.nan
        rec = lines_map.setdefault(lid, {"nut": nut, "obs": []})
        rec["obs"].append((2.0 * tr, cfu, pfu))

    wash_48h = -math.log(0.01) / 2.0
    records: list[TrajectoryRecord] = []
    for lid in sorted(lines_map, key=lambda s: (s[0], int(s[1:]))):
        info = lines_map[lid]
        obs_s = sorted(info["obs"], key=lambda x: x[0])
        t_vec = np.array([x[0] for x in obs_s], dtype=float)
        h_vec = np.array([x[1] for x in obs_s], dtype=float)
        p_vec = np.array([x[2] for x in obs_s], dtype=float)
        w_vec = np.where(t_vec == 0.0, 0.0, wash_48h)

        rmap = res_by_line.get(lid, {})
        rdays = np.array(sorted(rmap), dtype=float)
        rvals = np.array([rmap[d] for d in sorted(rmap)], dtype=float)

        records.append(TrajectoryRecord(
            dataset_id="D5_zhao2016",
            unit_id=lid,
            treatment=info["nut"],
            infected=True,
            times_d=t_vec,
            host_cells_ml=h_vec,
            phage_pfu_ml=p_vec,
            washout_d=w_vec,
            res_days=rdays,
            res_ancestral=rvals,
            res_contemporary=rvals,
            extra={"uninfected_cfu": uninf_base.get(lid, math.nan)},
        ))
    return records
