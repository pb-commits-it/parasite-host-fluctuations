#!/usr/bin/env python3
"""Blind schema-only scout of downloaded datasets in data/raw/chemostat/.

Reads ONLY documentation, archive member names, worksheet names, column headers
(first non-empty row), and row counts. Never prints or computes any bacterial
density, phage density, variance, correlation, or outcome value.
Run with: python3 -I casestudies/chemostat/schema_scout.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = str((__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve())
RAW = os.path.join(ROOT, "data", "raw", "chemostat")
OUT_DIR = str((__import__('pathlib').Path(__file__).resolve().parent / 'results').resolve())


def xlsx_sheets_and_headers(path: str) -> list[tuple[str, int, list[str]]]:
    """Parse .xlsx XML safely via standard library without executing macros or reading data rows."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    out = []
    with zipfile.ZipFile(path, "r") as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared.append("".join(texts))
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        for sh in wb.findall(".//main:sheet", ns):
            name = sh.attrib.get("name", "?")
            rid = sh.attrib.get(rel_ns, "")
            target = rmap.get(rid, "").lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            if target not in zf.namelist():
                continue
            sxml = ET.fromstring(zf.read(target))
            rows = sxml.findall(".//main:sheetData/main:row", ns)
            nrows = len(rows)
            header = []
            for row in rows[:3]:
                vals = []
                for c in row.findall("main:c", ns):
                    t = c.attrib.get("t", "")
                    v = c.find("main:v", ns)
                    is_el = c.find("main:is", ns)
                    if t == "s" and v is not None and v.text is not None:
                        idx = int(v.text)
                        vals.append(shared[idx] if 0 <= idx < len(shared) else "")
                    elif t == "inlineStr" and is_el is not None:
                        vals.append("".join(x.text or "" for x in is_el.findall(".//main:t", ns)))
                    elif v is not None and v.text is not None:
                        vals.append(f"<cell:{c.attrib.get('r','?')}>")
                if any(vals):
                    header = vals[:20]
                    break
            out.append((name, nrows, header))
    return out


def inspect_csv_bytes(raw: bytes, label: str, buf: io.StringIO) -> None:
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        print(f"  [CSV] {label}: empty", file=buf)
        return
    sep = ";" if lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.reader([lines[0]], delimiter=sep)
    hdr = next(reader, [])
    print(f"  [CSV] {label}: {len(lines)-1} data rows, sep='{sep}', header={hdr[:20]}", file=buf)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    buf = io.StringIO()
    print("=== BLIND SCHEMA SCOUT (headers, sheet names, and row counts ONLY) ===", file=buf)
    readme_path = os.path.join(RAW, "larsen2019_synechococcus_chemostat", "README.md")
    if os.path.exists(readme_path):
        print("\n--- Larsen 2019 README.md ---", file=buf)
        with open(readme_path, "r", encoding="utf-8", errors="replace") as f:
            print(f.read(), file=buf)

    for sub in sorted(os.listdir(RAW)):
        subp = os.path.join(RAW, sub)
        if not os.path.isdir(subp):
            continue
        print(f"\n=== Deposit: {sub} ===", file=buf)
        for fn in sorted(os.listdir(subp)):
            if fn.startswith("_"):
                continue
            fp = os.path.join(subp, fn)
            if fn.endswith(".csv"):
                with open(fp, "rb") as f:
                    inspect_csv_bytes(f.read(), fn, buf)
            elif fn.endswith(".xlsx"):
                try:
                    sheets = xlsx_sheets_and_headers(fp)
                    print(f"  [XLSX] {fn}: {len(sheets)} sheets", file=buf)
                    for sname, nrows, hdr in sheets:
                        print(f"    sheet '{sname}': {nrows} non-empty XML rows, top-row={hdr}", file=buf)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [XLSX] {fn}: parse error {exc}", file=buf)
            elif fn.endswith(".zip"):
                with zipfile.ZipFile(fp, "r") as zf:
                    print(f"  [ZIP] {fn}: {len(zf.namelist())} entries", file=buf)
                    for zn in sorted(zf.namelist()):
                        if zn.endswith(".csv") and not os.path.basename(zn).startswith("."):
                            inspect_csv_bytes(zf.read(zn), zn, buf)

    out_text = buf.getvalue()
    out_path = os.path.join(OUT_DIR, "schema_scout.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
