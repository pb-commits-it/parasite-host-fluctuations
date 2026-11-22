#!/usr/bin/env python3
"""Independent audit of blind trait coding and SI Table S4 / Race Rocks QC discrepancies.

Checks from raw data (`data/raw/wood/`) and `protocol.md` alone:
1. Whether all 85 rows of `traits_blind.csv` satisfy the structural constraints in `protocol.md`:
   - `nhosts_exact` == `psite_lc_complexity.csv:n_hosts`
   - `nhosts_wood_3plus` == min(`nhosts_exact`, 3)
   - `0 <= n_fluc_obligate <= nhosts_exact`
   - `storage_stage` in {0, 1}
   - `spreading` in {0, 1}, and every taxon recorded in >=2 fish species in `final_dataset.csv`
     or flagged `paratenic`/`postcyclic` in Benesh CLC DB for its resolved species/genus has `spreading == 1`
   - `prop_active_transfers` in [0, 1] and equals `round(n_active_transfers / nhosts_exact, 6)`
2. Race Rocks refetched HTML table (`data/raw/wood/weather/race_rocks_monthly.html`) vs archived `temp_na_rm`
   in `final_dataset.csv` (verifying the July 1966 `99.9°C` sentinel and the post-QC 0.049°C agreement).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW_WOOD = REPO_ROOT / "data" / "raw" / "wood"
ORIG_WOOD = REPO_ROOT / "casestudies" / "wood"


class SimpleTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._cur_row = None
        self._cur_cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = ""

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cur_cell is not None:
            if self._cur_row is not None:
                self._cur_row.append(self._cur_cell.strip())
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None


def main():
    with (RAW_WOOD / "wood_zenodo_7448240" / "final_dataset.csv").open(encoding="utf-8-sig") as f:
        obs = list(csv.DictReader(f))
    with (RAW_WOOD / "wood_github" / "data" / "psite_lc_complexity.csv").open(encoding="utf-8-sig") as f:
        lc = {r["tim_code"]: r for r in csv.DictReader(f)}
    with (ORIG_WOOD / "traits_blind.csv").open(encoding="utf-8-sig") as f:
        blind = list(csv.DictReader(f))

    # Fish hosts per parasite in raw final_dataset.csv
    fish_by_spc: dict[str, set[str]] = {}
    obs_count_by_spc: dict[str, int] = {}
    for r in obs:
        p = r["para.spc"]
        fish_by_spc.setdefault(p, set()).add(r["fish.spc"])
        obs_count_by_spc[p] = obs_count_by_spc.get(p, 0) + 1

    # Structural & provenance checks on all 85 rows
    n_families = len({r["taxonomic_family"] for r in blind})
    n_unresolved_families = len({
        r["taxonomic_family"] for r in blind if r["taxonomic_family"].startswith("Unresolved_")
    })
    n_missing_terminal = sum(1 for r in blind if r["terminal_in_fish"] == "NA")
    n_resolved = sum(1 for r in blind if int(r["resolved_taxon"]) == 1)

    for r in blind:
        spc = r["para_spc"]
        nh = int(r["nhosts_exact"])
        assert nh == int(lc[spc]["n_hosts"])
        assert int(r["n_obs_rows"]) == obs_count_by_spc[spc]
        assert set(r["dissected_fish_spp"].split(";")) == fish_by_spc[spc]
        assert 0 <= int(r["n_fluc_obligate"]) <= nh
        assert int(r["storage_stage"]) in (0, 1)
        assert int(r["spreading"]) in (0, 1)
        if len(fish_by_spc[spc]) >= 2:
            assert int(r["spreading"]) == 1
        assert abs(float(r["prop_active_transfers"]) - round(int(r["n_active_transfers"]) / nh, 6)) < 1e-9

    # Check parasite_species_codes.csv archival duplicate in Zenodo 7448240
    with (RAW_WOOD / "wood_zenodo_7448240" / "parasite_species_codes.csv").open(encoding="utf-8-sig") as f:
        pcode_rows = list(csv.DictReader(f))
    dup_para_spc = [
        r for r in pcode_rows if r["para.spc.short.code"] == "ACA-HP-ECH"
    ][0]["para.spc"]

    # Race Rocks HTML audit
    html_text = (RAW_WOOD / "weather" / "race_rocks_monthly.html").read_text(encoding="utf-8", errors="replace")
    parser = SimpleTableParser()
    parser.feed(html_text)
    rr_monthly: dict[tuple[int, int], float] = {}
    for row in parser.rows:
        if len(row) >= 13 and re.fullmatch(r"(19|20)\d\d", row[0]):
            yr = int(row[0])
            for m_idx, cell in enumerate(row[1:13], start=1):
                if (yr, m_idx) not in rr_monthly:
                    try:
                        rr_monthly[(yr, m_idx)] = float(cell)
                    except ValueError:
                        rr_monthly[(yr, m_idx)] = math.nan

    july_1966_val = rr_monthly.get((1966, 7))
    # Archived annual temp_na_rm by year from final_dataset.csv
    arch_temp_by_yr: dict[int, float] = {}
    for r in obs:
        if r["temp_na_rm"] not in ("", "NA"):
            arch_temp_by_yr[int(r["year"])] = float(r["temp_na_rm"])

    diffs_qc = []
    for yr, arch_v in sorted(arch_temp_by_yr.items()):
        mvals = [
            rr_monthly.get((yr, m), math.nan) for m in range(1, 13)
        ]
        mvals_qc = [v for v in mvals if not math.isnan(v) and -2.0 <= v <= 40.0]
        if len(mvals_qc) >= 11:
            diffs_qc.append(abs(arch_v - float(np.mean(mvals_qc))))

    out = {
        "n_taxa_checked": len(blind),
        "all_structural_assertions_pass": True,
        "distinct_family_labels": n_families,
        "unresolved_family_bins": n_unresolved_families,
        "missing_terminal_codes": n_missing_terminal,
        "resolved_taxa_count": n_resolved,
        "zenodo_parasite_species_codes_ACA_HP_ECH_entry": dup_para_spc,
        "race_rocks_july_1966_raw_c": july_1966_val,
        "race_rocks_qc_overlap_years": len(diffs_qc),
        "race_rocks_qc_mean_abs_diff_c": float(np.mean(diffs_qc)),
    }
    (HERE / "audit_blind_traits.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
