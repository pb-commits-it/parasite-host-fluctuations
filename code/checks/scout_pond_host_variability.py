"""SCOUTING ONLY.  No test is run.  No parasite outcome is related to any host quantity.

Question.  Can the open deposits from the California pond survey (Johnson & Wilber 2017, and later
deposits by the same group) give, pond by pond, the variability over years of host density (step 3b
of the scouting brief) and the marginal frequency of each parasite (step 3c)?

Claim checked.  The field data file deposited with Johnson & Wilber (2017) has no column for the
pond, the year, the host species, the parasite species or any host density.  If that is so, steps
3a, 3b and 3c cannot be computed from it, and the script says so instead of computing anything.

What the script reads.
  1. The MANIFEST.json files written by data/fetch/fetch_johnson_wilber_2017.py.
  2. The journal's metadata text for the 2017 field file (documentation, not data).
  3. The FIRST LINE, the column names, of each data file that is on disk.  For a zip it lists the
     members and reads the first line of each text member.  For the one spreadsheet it reads the
     first row of the first sheet and the sheet's stated size.
It never reads a second line of any data file, so no parasite count, prevalence or host density is
seen by the script or printed.  Nothing is sorted or filtered by any outcome.

What it prints.
  A. Which files of the 2017 deposit are on disk, and why the others are not.
  B. The columns of the 2017 field file, each put in a class.
  C. Steps 3a, 3b, 3c: computable or not, with the reason.
  D. The same classing of columns for the further deposits whose column names were fetched.

Classes: POND, YEAR, HOST (which host species), HOST DENSITY, PARASITE (which parasite),
PARASITE OUTCOME (anything measured on the parasite: counts, prevalence, infected-host density,
quantities derived from them), EFFORT (how much was sampled), OTHER.  The classing is the table
CLASS below, written by hand from the deposits' own documentation.  A name that is not in the table
is printed as UNCLASSED, so nothing is guessed silently.

Run: python3 -I checks/scout_pond_host_variability.py
     (an optional argument gives another raw-data directory; used once to try the script on
      made-up files, see data/INVENTORY3.md)
"""

import json
import os
import re
import sys
import zipfile

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
ROOT = str((__import__('pathlib').Path(__file__).resolve().parent / '..').resolve())
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "raw")
MAIN = os.path.join(RAW, "johnson_wilber_2017")
FIRST_LINE_CAP = 8192

P_OUT = "PARASITE OUTCOME"
CLASS = {
    # Johnson & Wilber 2017, field file (journal metadata, rspb20171388_si_005.txt)
    "logmean": P_OUT, "logvar": P_OUT, "P": P_OUT, "H": "EFFORT",
    "fs_var": P_OUT, "lower": P_OUT, "upper": P_OUT, "fs_var_mean": P_OUT,
    "fs_var_log_med": P_OUT, "lower_log": P_OUT, "upper_low": P_OUT, "fs_var_mean_log": P_OUT,
    # Johnson et al. 2024, Figshare 24982794 (README.docx)
    "fsite": "POND", "fyear": "YEAR", "fsiteyear": "POND+YEAR", "fsiteyearspecies": "POND+YEAR+HOST",
    "AmphSppCode": "HOST", "AmphParasite": "PARASITE", "AmphDissectN": "EFFORT",
    "NumAmphsInf": P_OUT, "TotalAmphMeta": P_OUT, "MeanMetaPerAmph": P_OUT,
    "SnailObserved": "HOST", "RichnessNum": "OTHER", "logZYGOPdensity": "OTHER",
    "logPSREdensity": "HOST DENSITY", "logAMPHdensity": "HOST DENSITY",
    "logCommunComp": "HOST DENSITY", "logMETAdensity": P_OUT,
    # Hobart et al. 2022, Figshare 15067254 (no README; classed from the names) [inferred]
    "Dissect.Code": "OTHER", "yearvis": "YEAR", "doy": "OTHER", "SnailSPPCode": "HOST",
    "Size": "OTHER", "stsize": "OTHER", "fchaeto": "OTHER", "ChaetogasterCount": "OTHER",
    "Infection": P_OUT, "fInfection": P_OUT, "MorphotypeCode1": "PARASITE",
    # Moss et al. 2020, Figshare 11809242 (no README) [inferred]
    "DissectionCode": "OTHER", "tCollectionDate": "YEAR", "SiteCode": "POND", "SpeciesCode": "HOST",
    "CollectionCode": "OTHER", "Visit": "OTHER", "Alaria": P_OUT, "Cephalogonimus": P_OUT,
    "Echinostoma": P_OUT, "Gorgoderid_Metacercaria": P_OUT, "Gyrinicola_batrachiensis": P_OUT,
    "Manodistomum_syntomentera": P_OUT, "Chabaudgolvania_sp": P_OUT, "Ribeiroia_ondatrae": P_OUT,
    "Nyctotherus": P_OUT, "Opalina": P_OUT, "Tritrichomonas": P_OUT,
    # Stewart Merrill & Johnson 2025, Figshare 29172116 (no README) [inferred]
    "Year": "YEAR", "Parasite": "PARASITE", "TotalCount": P_OUT, "Host": "HOST",
    "Susceptibility": "OTHER", "DoseSusceptibility": "OTHER", "Competence": "OTHER",
    # McDevitt-Galles et al. 2023, Zenodo 7062537 (MetaData sheet of the workbook)
    "SiteYear": "POND+YEAR", "nNet": "EFFORT", "Dist": "EFFORT", "SppCode": "HOST",
    "Count": "HOST DENSITY", "RACA": "HOST DENSITY", "Fish": "OTHER", "Cray": "OTHER",
    "Area": "OTHER", "Perim": "OTHER", "Date": "OTHER", "PropName": "OTHER", "County": "OTHER",
    "State": "OTHER", "FishPA": "OTHER", "RACAPA": "OTHER", "AMCA_JUV": "OTHER",
    "RADR_ALL": "OTHER", "RADR_JUV": "OTHER", "sPerim": "OTHER", "PDSI": "OTHER",
    "BdPosPSRE": P_OUT, "nSwabbedPSRE": "EFFORT",
}
# Columns of the 2024 deposit whose name carries a four-letter parasite code (MTST, MTAR, ...).
PATTERNS = [
    (re.compile(r"^MT[A-Z]{2}\.(SnailPrev|CombinedPrev)$"), P_OUT),
    (re.compile(r"^logMT[A-Z]{2}\.(DIS|CercDens)$"), P_OUT),
]
NOTE = {
    "H": "hosts dissected in the population",
    "logPSREdensity": "one host only, Pseudacris regilla; log10(1 + mean number per dipnet sweep)",
    "logAMPHdensity": "all amphibian species summed; log10(1 + mean number per dipnet sweep)",
    "logCommunComp": "amphibian density weighted by laboratory competence; not a plain density",
    "Count": "count of the focal species in seine hauls; divide by Dist; two threatened species only",
    "RACA": "count of bullfrogs in seine hauls; divide by Dist",
    "BdPosPSRE": "a fungus (Bd), not one of the four trematodes",
    "yearvis": "year and visit in one code [inferred]",
    "tCollectionDate": "a collection date; the year can be read from it [inferred]",
}
# What the documentation says about which rows a deposit holds.  Printed with the columns, because
# a column can exist and still be unusable if the rows were chosen by a parasite outcome.
ROWS = {
    "johnson_2024_density_richness": [
        "Which rows it holds (paper's supplementary methods, not checked against the files):",
        "only ponds in which the parasite was detected in a snail or an amphibian; at least 20",
        "snails dissected; at least 5 amphibians of the species dissected; amphibian larvae found",
        "in the dipnet survey; ponds without snails or without amphibians left out; Echinostoma",
        "from 2011 only.  So the rows are chosen by a parasite outcome and by host presence:",
        "a year in which the host count was zero, and a pond without the parasite, are not rows.",
        "Total snail density is not a column; only infected-snail density and prevalence are.",
    ],
}


def clean(name):
    name = name.strip().strip('"').lstrip("﻿")
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in name)[:60]


def classify(name):
    if name in CLASS:
        return CLASS[name]
    for rx, cls in PATTERNS:
        if rx.match(name):
            return cls
    return "UNCLASSED"


def split_header(line):
    return [clean(x) for x in line.split(",")]


def first_line_of(fh):
    raw = fh.readline(FIRST_LINE_CAP)
    return raw.decode("utf-8", "replace").rstrip("\r\n")


def show(names, indent="     "):
    seen = {}
    for n in names:
        cls = classify(n)
        seen.setdefault(cls, []).append(n)
        print("%s%-28s %-18s %s" % (indent, n, cls, NOTE.get(n, "")))
    return seen


def have(seen, cls):
    return any(k == cls or k.startswith(cls + "+") or ("+" + cls) in k for k in seen)


def verdict(seen):
    return "pond %s, year %s, host density %s, parasite outcome %s" % tuple(
        "YES" if have(seen, c) else "no" for c in ("POND", "YEAR", "HOST DENSITY", P_OUT))


def documented_columns(path):
    """Names in the journal's metadata text: lines of the form 'name: description'."""
    names = []
    started = False
    for line in open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n").split("\n"):
        if line.startswith("---"):
            started = True
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*):\s", line)
        if started and m:
            names.append(m.group(1))
    return names


def xlsx_first_row(path):
    """First row of the first sheet and the sheet's stated range.  No other row is read."""
    z = zipfile.ZipFile(path)
    for info in z.infolist():
        if info.file_size > 50_000_000:
            raise SystemExit("workbook member too large: %s" % info.filename)
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        s = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
        for si in re.findall(r"<si>(.*?)</si>", s, flags=re.S):
            strings.append("".join(re.findall(r"<t(?: [^>]*)?>(.*?)</t>", si, flags=re.S)))
    x = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "replace")
    dim = re.search(r'<dimension ref="([^"]*)"', x)
    row = re.search(r"<row [^>]*>(.*?)</row>", x, flags=re.S)
    names = []
    for attrs, body in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row.group(1) if row else "", flags=re.S):
        v = re.search(r"<v>(.*?)</v>", body or "", flags=re.S)
        if v is None:
            names.append("")
        elif 't="s"' in attrs:
            names.append(strings[int(v.group(1))])
        else:
            names.append(v.group(1))
    return [clean(n) for n in names], (dim.group(1) if dim else "not stated")


def part_a():
    print("A. The 2017 deposit: what is on disk")
    mpath = os.path.join(MAIN, "MANIFEST.json")
    if not os.path.exists(mpath):
        print("   no MANIFEST.json in %s; run data/fetch/fetch_johnson_wilber_2017.py first" % MAIN)
        return
    for m in json.load(open(mpath)):
        if m.get("kind") == "metadata":
            continue
        state = "on disk" if m.get("downloaded") else "NOT on disk"
        print("   %-48s %8s B  %s" % (m["path"], m.get("bytes"), state))
        if not m.get("downloaded"):
            print("        reason: %s" % m.get("reason"))


def part_b():
    print()
    print("B. The 2017 field file: its columns")
    doc = os.path.join(MAIN, "figshare_esm", "rspb20171388_si_005.txt")
    documented = []
    if os.path.exists(doc):
        documented = documented_columns(doc)
        print("   documented in the journal's metadata text (%d columns; the text says the file has" % len(documented))
        print("   971 rows and 12 columns and names it hostparasite_tpl_data_submission.csv):")
        seen = show(documented)
        print("   so, by the documentation: %s" % verdict(seen))
    else:
        print("   metadata text not on disk")
    found_any = False
    on_disk = []          # one dict of classes per data file or zip member actually read
    for rel in ("figshare_esm/rspb20171388_si_004.csv",):
        path = os.path.join(MAIN, rel)
        if os.path.exists(path):
            found_any = True
            names = split_header(first_line_of(open(path, "rb")))
            print("   first line of %s (%d columns):" % (rel, len(names)))
            seen = show(names)
            on_disk.append(seen)
            print("   so, by the file itself: %s" % verdict(seen))
            print("   same names as documented: %s" % (names == documented))
    for rel in ("dryad/prsb_datafiles.zip", "zenodo_4967765/prsb_datafiles.zip"):
        path = os.path.join(MAIN, rel)
        if os.path.exists(path):
            found_any = True
            z = zipfile.ZipFile(path)
            print("   members of %s:" % rel)
            for info in z.infolist():
                print("     %10d  %s" % (info.file_size, clean(info.filename)))
            for info in z.infolist():
                if info.filename.lower().endswith((".csv", ".txt")) and not info.is_dir():
                    names = split_header(first_line_of(z.open(info)))
                    print("   first line of member %s (%d fields):" % (clean(info.filename), len(names)))
                    seen = show(names)
                    on_disk.append(seen)
                    print("   so: %s" % verdict(seen))
    if not found_any:
        print("   no data file of the 2017 deposit is on disk, so no first line could be read.")
        print("   The statement above rests on the journal's metadata text alone.  The Dryad zip was")
        print("   not opened and may hold other files; that is not known.")
    return documented, found_any, on_disk


def part_c(documented, found_any, on_disk):
    print()
    print("C. Steps 3a, 3b, 3c on the 2017 field file")
    if any(have(d, "POND") and have(d, "YEAR") for d in on_disk):
        print("   A file of the deposit that is on disk HAS pond and year columns (see B).  The")
        print("   statement 'not computable' does not hold for that file.  This script computes none")
        print("   of steps 3a to 3c; they need a script written for that file's layout, which drops")
        print("   the parasite columns before anything is printed.  Nothing was computed.")
        return
    seen = {}
    for n in documented:
        seen.setdefault(classify(n), []).append(n)
    basis = "its first line" if found_any else "its documentation (file not on disk)"
    if not documented:
        print("   nothing known about the columns; no statement made")
        return
    pond, year, dens = have(seen, "POND"), have(seen, "YEAR"), have(seen, "HOST DENSITY")
    par = have(seen, "PARASITE")
    print("   basis: %s" % basis)
    print("   3a ponds with data in 1, 2, ... 7 years: %s" % (
        "computable" if pond and year else
        "NOT COMPUTABLE.  Pond identity in the file: %s.  Year in the file: %s." % (
            "yes" if pond else "no", "yes" if year else "no")))
    print("   3b sd over years of ln(host density), by pond: %s" % (
        "computable" if pond and year and dens else
        "NOT COMPUTABLE.  Host density in the file: %s.  No zeros were counted, no pseudocount "
        "was added, no lag-one correlation was computed: there is nothing to compute them from."
        % ("yes" if dens else "no")))
    print("   3c marginal frequency of each parasite species: %s" % (
        "computable" if par and pond and year else
        "NOT COMPUTABLE.  Parasite species in the file: %s.  A row is one host species by "
        "parasite by pond by year population that passed the paper's filter (at least 3 hosts and "
        "at least 4 parasites, as the article's Methods were summarised to me; not read verbatim), "
        "so a parasite that was absent is not a row." % ("yes" if par else "no")))
    print("   Nothing was computed from data in this section.")
    print("   From the deposit's abstract and metadata text, not from data: 971 populations, 11,987 hosts, 332,684")
    print("   parasites; 332684 / 11987 = %.1f trematodes per dissected host, the four species pooled," % (332684 / 11987))
    print("   among populations that passed the filter.  This is not a frequency of occurrence.")


def part_d():
    print()
    print("D. Further deposits: columns by class (first lines only)")
    groups = [
        ("Johnson, Stewart Merrill, Dean & Fenton 2024, Figshare 10.6084/m9.figshare.24982794",
         "johnson_2024_density_richness",
         ["rion_data.csv", "alar_data.csv", "ceph_data.csv", "ecsp_data.csv", "rion_data_community.csv",
          "alar_data_community.csv", "ceph_data_community.csv", "ecsp_data_community.csv"]),
        ("Hobart et al. 2022, Figshare 10.6084/m9.figshare.15067254", "johnson_lab_other_figshare",
         ["Hobart_et_al_JAnE.csv"]),
        ("Moss et al. 2020, Figshare 10.6084/m9.figshare.11809242", "johnson_lab_other_figshare",
         ["moss_JAE_data.csv"]),
        ("Stewart Merrill & Johnson 2025, Figshare 10.6084/m9.figshare.29172116",
         "johnson_lab_other_figshare", ["RION_infections.csv"]),
    ]
    for title, sub, files in groups:
        print("   %s" % title)
        shown = None
        for f in files:
            path = os.path.join(RAW, sub, f + ".header.txt")
            if not os.path.exists(path):
                print("     %s: first line not on disk" % f)
                continue
            names = split_header(first_line_of(open(path, "rb")))
            generic = [re.sub(r"MT[A-Z]{2}\.", "MTxx.", n) for n in names]
            if shown is not None and generic == shown:
                print("     %s: same columns as the file above, with its own parasite code" % f)
                continue
            shown = generic
            print("     %s (%d columns):" % (f, len(names)))
            seen = show(names, indent="       ")
            print("       so: %s" % verdict(seen))
        for line in ROWS.get(sub, []):
            print("     %s" % line)
    title = "McDevitt-Galles et al. 2023, Zenodo 10.5281/zenodo.7062537"
    path = os.path.join(RAW, "mcdevitt_galles_2023_drought_amphibians",
                        "FWS_Amphibian_responses_to_extreme_drought.xlsx")
    print("   %s" % title)
    if os.path.exists(path):
        names, dim = xlsx_first_row(path)
        print("     first row of the first sheet (%d columns; the sheet states its range as %s):" % (len(names), dim))
        seen = show(names, indent="       ")
        print("       so: %s" % verdict(seen))
        print("       The parasite outcome here is the fungus Bd in one frog species.  No trematode, no snail.")
    else:
        print("     workbook not on disk")


def main():
    print("Scout of pond host variability: which columns exist.  No data row was read.")
    print("Raw-data directory: %s" % ("data/raw" if len(sys.argv) < 2 else RAW))
    print()
    part_a()
    documented, found_any, on_disk = part_b()
    part_c(documented, found_any, on_disk)
    part_d()


if __name__ == "__main__":
    main()
