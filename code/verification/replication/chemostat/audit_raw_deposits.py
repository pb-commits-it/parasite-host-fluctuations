#!/usr/bin/env python3
"""Independent raw-deposit schema, hash, and edge-case audit for `r36-rep-chemostat`.

Reads `data/raw/chemostat/` using standard-library CSV/ZIP/XML parsers only, verifies
manifest SHA-256 digests, and audits raw structure across D1-D6 and `weather_env/`.
Outputs summary metadata only (`raw_audit.json`); no raw data rows are copied.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW = ROOT / "data" / "raw" / "chemostat"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_xlsx_sheets(path: Path) -> dict[str, list[dict[str, str]]]:
    """Independent OpenXML `.xlsx` parser returning `{sheet_name: [row_col_dict, ...]}`."""
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    out: dict[str, list[dict[str, str]]] = {}
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall(f"{{{ns_main}}}si"):
                shared_strings.append("".join(t.text or "" for t in si.iter(f"{{{ns_main}}}t")))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {r.attrib["Id"]: r.attrib["Target"] for r in rels}

        for sh in wb.findall(f".//{{{ns_main}}}sheet"):
            name = sh.attrib["name"]
            rid = sh.attrib[f"{{{ns_rel}}}id"]
            target = rid_to_target[rid].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            ws = ET.fromstring(zf.read(target))
            rows_parsed: list[dict[str, str]] = []
            for row_el in ws.findall(f".//{{{ns_main}}}sheetData/{{{ns_main}}}row"):
                row_map: dict[str, str] = {}
                for cell in row_el.findall(f"{{{ns_main}}}c"):
                    ref = cell.attrib.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    ctype = cell.attrib.get("t", "")
                    val_el = cell.find(f"{{{ns_main}}}v")
                    inline_el = cell.find(f"{{{ns_main}}}is")
                    if ctype == "s" and val_el is not None and val_el.text is not None:
                        idx = int(val_el.text)
                        row_map[col] = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
                    elif ctype == "inlineStr" and inline_el is not None:
                        row_map[col] = "".join(t.text or "" for t in inline_el.iter(f"{{{ns_main}}}t"))
                    elif val_el is not None and val_el.text is not None:
                        row_map[col] = val_el.text
                rows_parsed.append(row_map)
            out[name] = rows_parsed
    return out


def to_float(val: str | None) -> float:
    if val is None:
        return math.nan
    s = val.strip()
    if s in ("", "NA", "NaN", "nan", "N/A", "."):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def main() -> None:
    utc_now = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()

    # 1. Verify MANIFEST.json and WEATHER_ENV_MANIFEST.json digests
    manifest = json.loads((RAW / "MANIFEST.json").read_text(encoding="utf-8"))
    dl_entries = [e for e in manifest if e.get("downloaded")]
    hash_mismatches = []
    for e in dl_entries:
        p = RAW / e["path"]
        if not p.exists() or sha256_file(p) != e["sha256"]:
            hash_mismatches.append(e["path"])

    w_manifest_path = RAW / "weather_env" / "WEATHER_ENV_MANIFEST.json"
    w_manifest = json.loads(w_manifest_path.read_text(encoding="utf-8")) if w_manifest_path.exists() else []
    w_entries = w_manifest if isinstance(w_manifest, list) else w_manifest.get("files", [])

    # 2. Audit D1 (`cid-means.csv` and `inf-mat.csv`)
    d1_dir = RAW / "larsen2019_synechococcus_chemostat"
    with (d1_dir / "cid-means.csv").open("r", encoding="utf-8") as f:
        cid_rows = list(csv.DictReader(f))
    d1_cids: dict[str, dict] = {}
    for r in cid_rows:
        day = to_float(r["day"])
        cid = r["cID"].strip()
        mic = r["microbe"].strip()
        abd = to_float(r["abd"])
        utype = r["type"].strip()
        lim = r["lim"].strip()
        rec = d1_cids.setdefault(cid, {"type": utype, "lim": lim, "post_syn": [], "post_phage": [], "pre_syn_n": 0})
        if day >= 0 and not math.isnan(abd):
            if mic == "Syn":
                rec["post_syn"].append((day, abd))
            elif mic == "Phage":
                rec["post_phage"].append((day, abd))
        elif day < 0 and mic == "Syn" and not math.isnan(abd):
            rec["pre_syn_n"] += 1

    with (d1_dir / "inf-mat.csv").open("r", encoding="utf-8") as f:
        inf_grid = list(csv.reader(f))
    anc_vals_distinct: dict[str, int] = {}
    evo_vals_distinct: dict[str, int] = {}
    host_days_d1 = sorted({to_float(x) for x in inf_grid[3][7:] if not math.isnan(to_float(x))})
    for row in inf_grid[7:]:
        if len(row) <= 7:
            continue
        is_anc = (row[2].strip() == "RIM8-1")
        target = anc_vals_distinct if is_anc else evo_vals_distinct
        for cell in row[7:]:
            k = cell.strip()
            target[k] = target.get(k, 0) + 1

    # 3. Audit D2 (`Data_sets_Phage_titers.zip`)
    zpath = RAW / "reuter2025_phix174_serial_transfer" / "Data_sets_Phage_titers.zip"
    d2_zip_info = {}
    with zipfile.ZipFile(zpath, "r") as zf:
        d2_names = zf.namelist()
        e1 = list(csv.DictReader(zf.read("Data_sets_Phage_titers/experiment_1_3_h_transfers.csv").decode("utf-8-sig").splitlines()))
        e2 = list(csv.DictReader(zf.read("Data_sets_Phage_titers/experiment_2_30_min_transfers.csv").decode("utf-8-sig").splitlines()))
        d2_zip_info = {
            "files": d2_names,
            "e1_cols": list(e1[0].keys()),
            "e1_rows": len(e1),
            "e2_cols": list(e2[0].keys()),
            "e2_rows_total": len(e2),
            "e2_rows_P2": sum(1 for r in e2 if r["Passage"].strip() == "P2"),
            "e2_rows_kept": sum(1 for r in e2 if r["Passage"].strip() != "P2"),
            "e2_first_row_per_line": {
                r["Line"].strip(): {
                    "Passage": r["Passage"],
                    "Transfer": r["Transfer"],
                    "CFU_ml": to_float(r["CFU_ml"]),
                    "Real_virus_concentration": to_float(r["Real_virus_concentration"]),
                    "Output_phage_ml": to_float(r["Output_phage_ml"]),
                    "Inoculation_volume_ml": to_float(r["Inoculation_volume_ml"]),
                }
                for r in e2 if r["Passage_ID"].strip() == "1"
            },
        }

    # 4. Audit D3 (`pbio.2005971.s007.xlsx` and `pbio.2005971.s008.xlsx`)
    s007_sheets = parse_xlsx_sheets(RAW / "chaudhry2018_lambda_serial_transfer" / "pbio.2005971.s007.xlsx")
    s008_sheets = parse_xlsx_sheets(RAW / "chaudhry2018_lambda_serial_transfer" / "pbio.2005971.s008.xlsx")
    f2_rows = s008_sheets["Figure 2"]
    lb_t = [to_float(f2_rows[1].get(c)) for c in ["B", "C", "D", "E", "F", "G", "H"]]
    lb_ctrl = [to_float(f2_rows[2].get(c)) for c in ["B", "C", "D", "E", "F", "G", "H"]]
    m9_t = [to_float(f2_rows[15].get(c)) for c in ["B", "C", "D", "E", "F", "G"]]
    m9_ctrl = [to_float(f2_rows[16].get(c)) for c in ["B", "C", "D", "E", "F", "G"]]

    # 5. Audit D4 (`Duncan_JEB.xlsx`)
    d4_sheets = parse_xlsx_sheets(RAW / "duncan2017_sbw25_temp_fluctuation" / "Duncan_JEB.xlsx")
    d4_den = d4_sheets["Bacteria_Phage_Density"]
    d4_ctrl_by_pop: dict[str, list[tuple[float, float]]] = {}
    d4_inf_by_pop: dict[str, list[tuple[float, float]]] = {}
    for r in d4_den[1:]:
        if not r.get("A"):
            continue
        pop = str(int(float(r["A"])))
        inf = (r.get("D", "").strip() == "Infected")
        t_d = to_float(r.get("E"))
        b_ml = to_float(r.get("F"))
        if not math.isnan(b_ml) and b_ml > 0:
            (d4_inf_by_pop if inf else d4_ctrl_by_pop).setdefault(pop, []).append((t_d, b_ml))

    d4_ctrl_sig_all = [float(np.std(np.log([x[1] for x in sorted(v)]), ddof=1)) for v in d4_ctrl_by_pop.values()]
    d4_ctrl_sig_post = [float(np.std(np.log([x[1] for x in sorted(v)[1:]]), ddof=1)) for v in d4_ctrl_by_pop.values()]
    d4_inf_sig_all = [float(np.std(np.log([x[1] for x in sorted(v)]), ddof=1)) for v in d4_inf_by_pop.values()]
    d4_inf_sig_post = [float(np.std(np.log([x[1] for x in sorted(v)[1:]]), ddof=1)) for v in d4_inf_by_pop.values()]
    d4_ctrl_t_sets = sorted({tuple(x[0] for x in sorted(v)) for v in d4_ctrl_by_pop.values()})
    d4_inf_t_sets = sorted({tuple(x[0] for x in sorted(v)) for v in d4_inf_by_pop.values()})

    # 6. Audit D5 (`Population_density.xlsx`, `Three_productivities.xlsx`, `Challenge_data.xlsx`)
    d5_dir = RAW / "zhao2016_sbw25_productivity"
    d5_den = parse_xlsx_sheets(d5_dir / "Population_density.xlsx")
    d5_tp = parse_xlsx_sheets(d5_dir / "Three_productivities.xlsx")
    d5_ch = parse_xlsx_sheets(d5_dir / "Challenge_data.xlsx")

    # 7. Audit D6 (`RAW DATA.xlsx`)
    d6_sheets = parse_xlsx_sheets(RAW / "harrison2013_sbw25_nutrient_pulses" / "RAW DATA.xlsx")

    summary = {
        "audited_utc_from_date": utc_now,
        "manifest_downloaded_files": len(dl_entries),
        "manifest_hash_mismatches": hash_mismatches,
        "D1_summary": {
            "cids": {
                k: {
                    "type": v["type"],
                    "lim": v["lim"],
                    "n_post_syn": len(v["post_syn"]),
                    "n_post_phage": len(v["post_phage"]),
                    "n_pre_syn": v["pre_syn_n"],
                    "sigma_syn_all_post": float(np.std(np.log([x[1] for x in sorted(v["post_syn"])]), ddof=1)) if len(v["post_syn"]) >= 2 else None,
                    "sigma_syn_excl_day0": float(np.std(np.log([x[1] for x in sorted(v["post_syn"])[1:]]), ddof=1)) if len(v["post_syn"]) >= 3 else None,
                }
                for k, v in sorted(d1_cids.items())
            },
            "inf_mat_isolation_days": host_days_d1,
            "inf_mat_ancestral_value_counts": anc_vals_distinct,
            "inf_mat_evolved_value_counts": evo_vals_distinct,
        },
        "D2_summary": d2_zip_info,
        "D3_summary": {
            "s007_sheets": list(s007_sheets.keys()),
            "s008_sheets": list(s008_sheets.keys()),
            "lb_times": lb_t,
            "lb_ctrl_sigma_incl_t0": float(np.std(np.log(lb_ctrl), ddof=1)),
            "lb_ctrl_sigma_excl_t0": float(np.std(np.log(lb_ctrl[1:]), ddof=1)),
            "m9_times": m9_t,
            "m9_ctrl_sigma_incl_t0": float(np.std(np.log(m9_ctrl), ddof=1)),
            "m9_ctrl_sigma_excl_t0": float(np.std(np.log(m9_ctrl[1:]), ddof=1)),
        },
        "D4_summary": {
            "sheets": list(d4_sheets.keys()),
            "header_row0": d4_den[0],
            "n_ctrl_pops": len(d4_ctrl_by_pop),
            "n_inf_pops": len(d4_inf_by_pop),
            "ctrl_time_grids": [list(t) for t in d4_ctrl_t_sets],
            "inf_time_grids": [list(t) for t in d4_inf_t_sets],
            "ctrl_sigma_mean_incl_first": float(np.mean(d4_ctrl_sig_all)),
            "ctrl_sigma_mean_excl_first": float(np.mean(d4_ctrl_sig_post)),
            "inf_sigma_mean_incl_first": float(np.mean(d4_inf_sig_all)),
            "inf_sigma_mean_excl_first": float(np.mean(d4_inf_sig_post)),
            "ratio_orig_mixed": float(np.mean(d4_inf_sig_post) / np.mean(d4_ctrl_sig_all)),
            "ratio_both_excl_first": float(np.mean(d4_inf_sig_post) / np.mean(d4_ctrl_sig_post)),
            "ratio_both_incl_first": float(np.mean(d4_inf_sig_all) / np.mean(d4_ctrl_sig_all)),
        },
        "D5_summary": {
            "den_sheets": {k: (len(v), v[0] if v else {}) for k, v in d5_den.items()},
            "tp_sheets": {k: (len(v), v[0] if v else {}) for k, v in d5_tp.items()},
            "ch_sheets": {k: (len(v), v[0] if v else {}) for k, v in d5_ch.items()},
        },
        "D6_summary": {
            "sheets": {k: (len(v), v[0] if v else {}) for k, v in d6_sheets.items()},
        },
    }

    (HERE / "raw_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
