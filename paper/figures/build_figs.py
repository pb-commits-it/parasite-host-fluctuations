#!/usr/bin/env python3
"""Compile the 14 used figures, export EPS, and verify immutable input data.

Run: python3 -I ../paper/figures/build_figs.py
Optional writable Tectonic cache: --cache-dir PATH
Requires tectonic, pdftops, and pdfinfo. No scientific Python dependencies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess

FIG = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()


def main():
    (FIG / "build").mkdir(exist_ok=True)
    baseline = json.loads((FIG / "figure-spec.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=list(baseline["figures"]))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    for exe in ("tectonic", "pdftops", "pdfinfo"):
        if not shutil.which(exe):
            raise SystemExit(f"Required executable unavailable: {exe}")
    def data_hashes():
        return {p.relative_to(FIG).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((FIG / "data").glob("*")) if p.is_file()}
    if data_hashes() != baseline["data_sha256"]:
        raise SystemExit("Data differ from the recorded before-state; refusing to rebuild")
    env = dict(os.environ)
    if env.get("TECTONIC_CACHE_DIR"):
        env["TECTONIC_CACHE_DIR"] = str(Path(env["TECTONIC_CACHE_DIR"]).resolve())
    if args.cache_dir:
        env["TECTONIC_CACHE_DIR"] = str(args.cache_dir.resolve())
    record_path = FIG / "build/build.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for number, name in enumerate(baseline["figures"], 1):
        if args.figures and name not in args.figures:
            continue
        proc = subprocess.run(["tectonic", "-X", "compile", "--keep-logs", name + ".tex"],
                              cwd=FIG, env=env, text=True, capture_output=True)
        (FIG / "build" / (name + "-build.txt")).write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise SystemExit(f"{name}: tectonic failed; see audit/{name}-build.txt")
        subprocess.run(["pdftops", "-eps", name + ".pdf", name + ".eps"], cwd=FIG, check=True)
        # DSC BoundingBox is integer-valued. Also preserve the exact PDF page
        # dimensions in HiResBoundingBox (without scaling or changing paths).
        info = subprocess.check_output(["pdfinfo", "-box", name + ".pdf"], cwd=FIG, text=True)
        box = re.search(r"^MediaBox:\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)", info, re.M)
        if not box:
            raise RuntimeError(f"{name}: cannot read MediaBox")
        # pdfinfo's page-size output retains more precision than -box.
        size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        if not size:
            raise RuntimeError(f"{name}: cannot read page size")
        eps = FIG / f"{name}.eps"
        text = re.sub(r"^%%HiResBoundingBox:.*\n", "", eps.read_text(), flags=re.M)
        hires = f"%%HiResBoundingBox: 0 0 {size[1]} {size[2]}"
        text = re.sub(r"^(%%BoundingBox:.*)$", lambda m: m[1] + "\n" + hires,
                      text, count=1, flags=re.M)
        eps.write_text(text)
        alias = FIG / f"Fig{number}.eps"
        shutil.copyfile(eps, alias)
        log = (FIG / f"{name}.log").read_text(errors="replace")
        record[name] = {"returncode": proc.returncode,
                        "artwork_width": re.findall(r"JMB artwork width: (.*)", log),
                        "eps": eps.name, "numbered_eps": alias.name,
                        "pdf_sha256": hashlib.sha256((FIG / f"{name}.pdf").read_bytes()).hexdigest(),
                        "eps_sha256": hashlib.sha256(eps.read_bytes()).hexdigest()}
        (FIG / "build" / f"{name}.log").write_text(log)
        (FIG / f"{name}.log").unlink()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Built {name}.pdf, {name}.eps, {alias.name}", flush=True)
    if data_hashes() != baseline["data_sha256"]:
        raise SystemExit("Data changed during compilation")
    print("All release data files match their SHA-256 checksums.")


if __name__ == "__main__":
    main()
