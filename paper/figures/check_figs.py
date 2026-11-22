#!/usr/bin/env python3
"""Audit the used JMB figures against the pre-rebuild baseline, without writes.

Run from any directory: python3 -I ../paper/figures/check_figs.py
Optional: --output ../paper/figures/audit/checks.json
Requires PyMuPDF (fitz); pdffonts is used when available. Widths and actual
painted vector strokes come from MuPDF, including graphics transformations.
PDF units are bp (1/72 inch); TeX pt = bp * 72.27/72. Font sizes include
subscripts and superscripts. No inference from source styles alone is used.
Tolerance: 0.01 mm in page width; 0.0001 pt for stroke roundoff, 0.01 pt for
text-size roundoff. Checksums and table/analytic plot selections must match
exactly. The unused exact-test is explicitly outside this audit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

import fitz

FIG = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
PT_PER_BP = 72.27 / 72


def bindings(tex):
    """Resolve the aliases already used by the original figure sources."""
    macros = dict(re.findall(
        r"\\def\\(\w+)\{((?:figures/)?data/[\w-]+\.csv)\}", tex))
    result = []
    for opts, path in re.findall(r"table\s*\[([^\]]*)\]\s*\{([^}]*)\}", tex):
        path = path.strip()
        if path.startswith("\\mechdata"):
            path = "data/" + path[len("\\mechdata"):].strip()
        elif path.startswith("\\"):
            path = macros[path[1:]]
        result.append({"options": re.sub(r"\s+", "", opts),
                       "file": path.removeprefix("figures/")})
    return result


def analytic_plots(tex):
    # Preserve formulas as well as their domain and sampling count.
    return [{"formula": expr, "domain": re.findall(r"domain=([^,\]]+)", opts),
             "samples": re.findall(r"samples=([^,\]]+)", opts)}
            for opts, expr in re.findall(
                r"\\addplot\[([^\]]*)\]\s*\{([^}]+)\}", tex)]


def fonts_in(pdf, page):
    if shutil.which("pdffonts"):
        raw = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        fonts = []
        for line in raw.splitlines()[2:]:
            fields = line.split()
            if len(fields) >= 8:
                fonts.append({"name": fields[0], "embedded": fields[-5] == "yes",
                              "type": " ".join(fields[1:-6])})
        return fonts, raw, "pdffonts"
    # A font-file buffer provides direct embedding evidence if pdffonts is absent.
    fonts = []
    for xref, ext, kind, name, *_ in page.get_fonts(full=True):
        fonts.append({"name": name, "embedded": bool(page.parent.extract_font(xref)[3]),
                      "type": kind})
    return fonts, None, "PyMuPDF extract_font"


def inspect_pdf(pdf):
    with fitz.open(pdf) as doc:
        page = doc[0]
        widths = [d["width"] for d in page.get_drawings() if "s" in d["type"]]
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
                 for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        sizes = [s["size"] * PT_PER_BP for s in spans]
        fonts, raw, method = fonts_in(pdf, page)
        outside = [s["text"] for s in spans
                   if not (page.rect + (-0.1, -0.1, 0.1, 0.1)).contains(fitz.Rect(s["bbox"]))]
        return {"width_mm": page.rect.width * 25.4 / 72,
                "height_mm": page.rect.height * 25.4 / 72,
                "page_count": len(doc), "fonts": fonts, "font_method": method,
                "pdffonts": raw,
                "min_stroke_bp": min(widths) if widths else None,
                "min_stroke_pt": min(widths) * PT_PER_BP if widths else None,
                "min_font_pt": min(sizes) if sizes else None,
                "max_font_pt": max(sizes) if sizes else None,
                "text_outside_page": outside, "raster_images": len(page.get_images())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write full JSON audit")
    args = parser.parse_args()
    baseline = json.loads((FIG / "figure-spec.json").read_text())
    before = baseline["data_sha256"]
    after = {p.relative_to(FIG).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((FIG / "data").glob("*")) if p.is_file()}
    failures = []
    if before != after:
        failures.append("The data directory differs from the before-state")
    used = {}
    for folder in ("sections", "appendices"):
        for p in sorted((FIG.parent / folder).glob("*.tex")):
            for width, name in re.findall(
                    r"\\includegraphics\[width=(\d+)mm\]\{figures/([^}]+)\.pdf\}", p.read_text()):
                used[name] = int(width)
    if set(used) != set(baseline["figures"]):
        failures.append("Used figure inventory differs from baseline; review the new references")
    result = {"data_unchanged": before == after, "figures": {},
              "all_data": {p: {"before": before.get(p), "after": after.get(p),
                               "unchanged": before.get(p) == after.get(p)}
                           for p in sorted(before.keys() | after.keys())}}
    print("Figure      width mm   fonts embedded   stroke pt   font pt (min–max)   data")
    for name, old in baseline["figures"].items():
        pdf = FIG / f"{name}.pdf"
        if not pdf.exists():
            failures.append(f"{name}: PDF missing")
            continue
        row = inspect_pdf(pdf)
        tex = (FIG / f"{name}.tex").read_text()
        original = old["analytic_plots"]
        row["old_width_mm"] = old["old_width_mm"]
        row["target_mm"] = used.get(name, old["target_mm"])
        row["table_bindings_unchanged"] = bindings(tex) == old["table_bindings"]
        row["analytic_plots_unchanged"] = analytic_plots(tex) == original
        row["data"] = {p: result["all_data"][p]
                       for p in sorted({b["file"] for b in old["table_bindings"]})}
        row["data_unchanged"] = all(d["unchanged"] for d in row["data"].values())
        eps = FIG / f"{name}.eps"
        row["eps_exists"] = eps.exists()
        if eps.exists():
            eps_text = eps.read_text(errors="replace")
            bbox = re.search(r"^%%BoundingBox: ([-\d ]+)$", eps_text, re.M)
            row["eps_bounding_box_bp"] = list(map(int, bbox[1].split())) if bbox else None
            hires = re.search(r"^%%HiResBoundingBox: ([\d. ]+)$", eps_text, re.M)
            row["eps_hires_box_bp"] = list(map(float, hires[1].split())) if hires else None
            row["eps_embedded_fonts"] = re.findall(r"^%%BeginResource: font (.+)$", eps_text, re.M)
        checks = {
            "width": abs(row["width_mm"] - row["target_mm"]) <= 0.01,
            "single page": row["page_count"] == 1,
            "font embedding": bool(row["fonts"]) and all(f["embedded"] for f in row["fonts"]),
            "sans fonts": all(f["name"].split("+")[-1].startswith(
                ("TeXGyreHeros-", "ComputerModernSans-")) for f in row["fonts"]),
            "minimum stroke": row["min_stroke_pt"] is not None and row["min_stroke_pt"] >= 0.3 - 0.0001,
            "font sizes": row["min_font_pt"] >= 8 - 0.01 and row["max_font_pt"] <= 12 + 0.01,
            "unclipped text": not row["text_outside_page"],
            "vector artwork": row["raster_images"] == 0,
            "data": row["data_unchanged"],
            "table selection": row["table_bindings_unchanged"],
            "analytic plots": row["analytic_plots_unchanged"],
            "EPS": row["eps_exists"],
            "EPS embedded fonts": bool(row.get("eps_embedded_fonts")),
            "EPS precise width": bool(row.get("eps_hires_box_bp")) and abs(
                (row["eps_hires_box_bp"][2] - row["eps_hires_box_bp"][0]) * 25.4 / 72
                - row["target_mm"]) <= 0.01,
        }
        row["checks"] = checks
        failures.extend(f"{name}: {key}" for key, ok in checks.items() if not ok)
        result["figures"][name] = row
        print(f"{name:11} {row['width_mm']:8.3f}   "
              f"{sum(f['embedded'] for f in row['fonts'])}/{len(row['fonts']):<12} "
              f"{row['min_stroke_pt']:9.4f}   "
              f"{row['min_font_pt']:.2f}–{row['max_font_pt']:.2f}             "
              f"{'same' if row['data_unchanged'] else 'CHANGED'}")
        for p, digest in row["data"].items():
            print(f"  {p}\n    before {digest['before']}\n    after  {digest['after']}")
        if not row["data"]:
            print("  Schematic: no data file")
        print("  Fonts: " + ", ".join(f"{f['name']} (embedded={f['embedded']})" for f in row["fonts"]))
    result["failures"] = failures
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("\n" + ("FAIL: " + "; ".join(failures) if failures else
                    f"PASS: {len(result['figures'])} figures; all {len(before)} data files unchanged"))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
