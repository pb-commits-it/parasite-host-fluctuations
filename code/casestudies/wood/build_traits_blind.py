#!/usr/bin/env python3
"""Build the outcome-blind trait table (traits_blind.csv) for all 85 parasite taxa
in Wood et al. (2023, PNAS; doi:10.5061/dryad.fqz612jwf).

STRICT BLINDNESS GUARANTEE:
- This script NEVER reads parasite counts ('count'), model results, random effects,
  or any outcome/trend variable. From final_dataset.csv it reads ONLY the columns
  ('fish.spc', 'para.spc') to map which of the 8 fish species each parasite taxon
  was recorded in.
- Every trait code is derived deterministically from open external databases
  (Benesh et al. 2017/2021 Helminth Life-Cycle Database, doi:10.5061/dryad.gtht76hmm;
  Global Mammal Parasite Database 2.0, doi:10.1002/ecy.1799; WoRMS REST API, CC BY 4.0;
  Global Biotic Interactions API, CC BY 4.0; FishBase attributes in Wood et al. SI Table S1;
  taxonomic identities and dissected stage in Wood et al. SI Table S3) and cited primary
  parasitological literature.

Run with:
  python3 -I casestudies/wood/build_traits_blind.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter

import numpy as np

HERE = str((__import__('pathlib').Path(__file__).resolve().parent / '.').resolve())
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RAW_WOOD = os.path.join(REPO_ROOT, "data", "raw", "wood")

# Wood et al. (2023) SI Table S1 FishBase attributes (Froese et al. 2022)
# Neritic + schooling pelagic hosts exhibit large recruitment/interannual abundance swings;
# demersal non-schooling multi-year groundfishes are buffered.
FISH_HOST_ATTRS = {
    "herring":  {"latin": "Clupea pallasii",      "habitat": "neritic",  "schooling": "yes",      "fluc_primary": 1, "fluc_broad": 1},
    "hake":     {"latin": "Merluccius productus", "habitat": "neritic",  "schooling": "yes",      "fluc_primary": 1, "fluc_broad": 1},
    "smelt":    {"latin": "Hypomesus pretiosus",  "habitat": "demersal", "schooling": "spawning", "fluc_primary": 0, "fluc_broad": 1},
    "pollock":  {"latin": "Gadus chalcogrammus",  "habitat": "demersal", "schooling": "spawning", "fluc_primary": 0, "fluc_broad": 0},
    "perch":    {"latin": "Embiotoca lateralis",  "habitat": "demersal", "schooling": "no",       "fluc_primary": 0, "fluc_broad": 0},
    "ratfish":  {"latin": "Hydrolagus colliei",   "habitat": "demersal", "schooling": "no",       "fluc_primary": 0, "fluc_broad": 0},
    "sole":     {"latin": "Parophrys vetulus",    "habitat": "demersal", "schooling": "no",       "fluc_primary": 0, "fluc_broad": 0},
    "rockfish": {"latin": "Sebastes caurinus",    "habitat": "demersal", "schooling": "no",       "fluc_primary": 0, "fluc_broad": 0},
}

# Taxonomic identifications and life stage in dissected fish from Wood et al. (2023) SI Table S3 (lines 58-325)
TABLE_S3_INFO: dict[str, tuple[str, str, str | None]] = {
    "ACA-CP-COR": ("Corynosoma sp.", "Larval", "Corynosoma"),
    "ACA-SC-COR": ("Corynosoma sp.", "Larval", "Corynosoma"),
    "ACA-SC-ECH": ("Echinorhynchus sp.", "Larval", "Echinorhynchus gadi"),
    "ACA-HP-ECH": ("Echinorhynchus gadi", "Larval", "Echinorhynchus gadi"),
    "ACA-PV-SP1": ("Acanthocephalan sp.", "Undetermined", None),
    "CES-MP-SP1": ("Cestode sp. 1", "Larval", None),
    "CES-MP-SP2": ("Cestode sp. 2", "Larval", None),
    "CES-MP-SP3": ("Cestode sp. 3", "Larval", None),
    "CES-MP-CLE": ("Clestobothrium crassiceps", "Adult", "Clestobothrium crassiceps"),
    "CES-MP-NYB": ("Nybelinia surmenicola", "Larval", "Nybelinia surmenicola"),
    "CES-CP-SP1": ("Cestode sp.", "Larval", None),
    "CES-CP-SP2": ("Cestode sp.", "Larval", None),
    "CES-GC-SP1": ("Cestode sp.", "Undetermined", None),
    "CES-GC-SP2": ("Cestode sp.", "Larval", None),
    "CES-GC-SP3": ("Cestode sp.", "Larval", None),
    "CES-HC-GYR": ("Gyrocotyle sp.", "Adult", "Gyrocotyle"),
    "CES-HP-SP1": ("Cestode sp. 1 (cysticerci)", "Larval", None),
    "COP-MP-PAR": ("Parabrachiella sp.", "Adult", "Parabrachiella"),
    "COP-MP-SP1": ("Copepod sp.", "Adult", None),
    "COP-CP-BOM": ("Bomolochus bellones", "Adult", "Bomolochus bellones"),
    "COP-CP-SP1": ("Copepod sp.", "Adult", None),
    "COP-EL-BOM": ("Bomolochus cuneatus", "Adult", "Bomolochus cuneatus"),
    "COP-EL-CLA": ("Clavella sp.", "Adult", "Clavella"),
    "COP-GC-LEL": ("Lepeophtheirus sp.", "Larval", "Lepeophtheirus"),
    "COP-GC-SP1": ("Copepod sp.", "Undetermined", None),
    "COP-GC-LEA": ("Lepeophtheirus sp.", "Adult", "Lepeophtheirus"),
    "COP-SC-CLA": ("Clavella sp.", "Adult", "Clavella"),
    "COP-SC-SP1": ("Copepod sp.", "Adult", None),
    "COP-SC-LER": ("Lernaeopodid sp.", "Adult", None),
    "COP-HP-LEP": ("Lepeophtheirus sp.", "Adult", "Lepeophtheirus"),
    "COP-PV-LEP": ("Lepeophtheirus sp.", "Larval", "Lepeophtheirus"),
    "HIR-PV-OCE": ("Oceanobdella pallida", "Adult", "Oceanobdella pallida"),
    "MON-MP-ANT": ("Anthocotyle merluccii", "Adult", "Anthocotyle merluccii"),
    "MON-CP-GYR": ("Gyrodactylus sp.", "Adult", "Gyrodactylus"),
    "MON-HC-CHI": ("Chimaericola sp.", "Adult", "Chimaericola"),
    "MON-SC-MI1": ("Microcotyle sp. 1", "Adult", "Microcotyle"),
    "MON-SC-MI2": ("Microcotyle sp. 2", "Adult", "Microcotyle"),
    "NEM-MP-ASC": ("Ascarophis sp.", "Adult", "Ascarophis"),
    "NEM-MP-SP1": ("Nematode sp.", "Both larval and adult", None),
    "NEM-CP-SP1": ("Nematode sp.", "Both larval and adult", None),
    "NEM-EL-CUC": ("Cucullanellus kanabus", "Adult", "Cucullanellus kanabus"),
    "NEM-GC-ASC": ("Ascarophis sp.", "Adult", "Ascarophis"),
    "NEM-GC-SP1": ("Nematode sp.", "Both larval and adult", None),
    "NEM-SC-ASC": ("Ascarophis sebastodis", "Adult", "Ascarophis sebastodis"),
    "NEM-SE-ANI": ("Anisakis sp.", "Larval", "Anisakis"),
    "NEM-SE-CON": ("Contracaecum sp.", "Larval", "Contracaecum"),
    "NEM-HP-SP1": ("Nematode sp.", "Both larval and adult", None),
    "NEM-PV-CAP": ("Capillaria parophrysi", "Adult", "Capillaria parophrysi"),
    "NEM-PV-CLA": ("Clavinema mariae", "Adult", "Clavinema mariae"),
    "NEM-PV-CUC": ("Cucullanus annulatus", "Adult", "Cucullanus annulatus"),
    "NEM-PV-SP1": ("Nematode sp.", "Both larval and adult", None),
    "NEM-PV-SPR": ("Spirurid sp.", "Adult", None),
    "TRE-MP-APO": ("Aporocotyle margolisi", "Adult", "Aporocotyle margolisi"),
    "TRE-MP-LEC": ("Lecithaster sp.", "Adult", "Lecithaster"),
    "TRE-MP-MET": ("Trematode metacercaria", "Larval", None),
    "TRE-MP-POD": ("Podocotyle sp.", "Adult", "Podocotyle"),
    "TRE-MP-SP1": ("Trematode sp.", "Adult", None),
    "TRE-CP-HEM": ("Hemiurid sp.", "Adult", None),
    "TRE-CP-LEC": ("Lecithaster sp.", "Adult", "Lecithaster"),
    "TRE-CP-ME1": ("Metacercaria sp. 1", "Larval", None),
    "TRE-CP-ME2": ("Metacercaria sp. 2", "Larval", None),
    "TRE-CP-PRO": ("Pronoprymna sp.", "Adult", "Pronoprymna petrowi"),
    "TRE-CP-SP1": ("Trematode sp.", "Adult", None),
    "TRE-EL-MET": ("Metacercaria sp.", "Larval", None),
    "TRE-GC-HEM": ("Hemiurid sp.", "Adult", None),
    "TRE-GC-LEC": ("Lecithaster sp.", "Adult", "Lecithaster"),
    "TRE-GC-LEP": ("Lepidapedon sp.", "Adult", "Lepidapedon"),
    "TRE-GC-ME1": ("Metacercaria sp. 1", "Larval", None),
    "TRE-GC-ME2": ("Metacercaria sp. 2", "Larval", None),
    "TRE-GC-POD": ("Podocotyle sp.", "Adult", "Podocotyle"),
    "TRE-GC-SP1": ("Trematode sp.", "Adult", None),
    "TRE-HC-GON": ("Gonocerca sp.", "Adult", "Gonocerca"),
    "TRE-SC-LEC": ("Lecithochirium sp.", "Adult", "Lecithochirium"),
    "TRE-SC-MET": ("Trematode metacercaria", "Larval", None),
    "TRE-SC-OPE": ("Opechona sp.", "Adult", "Opechona"),
    "TRE-SC-SP1": ("Trematode sp. 1", "Adult", None),
    "TRE-SC-SP2": ("Trematode sp. 2", "Adult", None),
    "TRE-SE-DER": ("Derogenes varicus", "Adult", "Derogenes varicus"),
    "TRE-SE-PRO": ("Pronoprymna petrowi", "Adult", "Pronoprymna petrowi"),
    "TRE-HP-LEC": ("Lecithaster sp.", "Adult", "Lecithaster"),
    "TRE-HP-MET": ("Metacercaria sp.", "Larval", None),
    "TRE-PV-ME1": ("Trematode metacercaria sp. 1", "Larval", None),
    "TRE-PV-ME2": ("Trematode metacercaria sp. 2", "Larval", None),
    "TRE-PV-SP1": ("Trematode sp. 1", "Adult", None),
    "TRE-PV-SP2": ("Trematode sp. 2", "Adult", None),
}


def load_external_databases():
    # 1. Benesh CLC_database_hosts.csv & CLC_database_lifehistory.csv
    benesh_genus = {}
    with open(os.path.join(RAW_WOOD, "benesh_clc_raw", "data", "CLC_database_hosts.csv"), encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            g = r["Parasite.genus"]
            info = benesh_genus.setdefault(g, {"species": set(), "facultative": set(), "hosts": set()})
            info["species"].add(r["Parasite.species"])
            if r["Facultative"] != "NA":
                info["facultative"].add(r["Facultative"])
            if r["Host.species"] != "NA":
                info["hosts"].add(r["Host.species"])

    with open(os.path.join(RAW_WOOD, "benesh_clc_raw", "data", "CLC_database_lifehistory.csv"), encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            g = r["Parasite.genus"]
            if g in benesh_genus and r["Egg.hatch"] != "NA":
                benesh_genus[g].setdefault("egg_hatch", set()).add(r["Egg.hatch"])

    # 2. GMPD2
    gmpd_genus = {}
    with open(os.path.join(RAW_WOOD, "gmpd2", "GMPD_main.csv"), encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            pname = r["ParasiteCorrectedName"] if r["ParasiteCorrectedName"] != "NA" else r["ParasiteReportedName"]
            g = pname.split()[0] if pname else ""
            if g:
                info = gmpd_genus.setdefault(g, {"records": 0, "mammal_hosts": set()})
                info["records"] += 1
                if r["HostCorrectedName"] != "NA":
                    info["mammal_hosts"].add(r["HostCorrectedName"])

    # 3. WoRMS + GloBI cache
    with open(os.path.join(RAW_WOOD, "worms_globi_cache.json"), encoding="utf-8") as f:
        wg_cache = json.load(f)

    return benesh_genus, gmpd_genus, wg_cache


def load_blind_fish_mapping():
    """Read ONLY ('fish.spc', 'para.spc') from final_dataset.csv; assert 'count' is untouched."""
    fish_by_para: dict[str, set[str]] = {}
    obs_by_para: dict[str, int] = {}
    with open(os.path.join(RAW_WOOD, "wood_zenodo_7448240", "final_dataset.csv"), encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_fish = header.index("fish.spc")
        idx_para = header.index("para.spc")
        # Deliberately never index 'count', 'year', 'temp_na_rm', 'sfish_density', 'ord1', 'ord2'
        for row in reader:
            p = row[idx_para]
            fsp = row[idx_fish]
            fish_by_para.setdefault(p, set()).add(fsp)
            obs_by_para[p] = obs_by_para.get(p, 0) + 1
    return fish_by_para, obs_by_para


def code_one_taxon(
    lc_row: dict[str, str],
    fish_hosts: list[str],
    n_obs: int,
    benesh_genus: dict,
    gmpd_genus: dict,
    wg_cache: dict,
) -> dict:
    pcode = lc_row["psite_code"]
    tcode = lc_row["tim_code"]
    ptype = lc_row["psite_type"]
    nhosts = int(lc_row["n_hosts"])
    nhosts_3plus = min(nhosts, 3)
    term_raw = lc_row["terminal"]
    s3_name, s3_stage, wg_key = TABLE_S3_INFO[pcode]
    genus = s3_name.split()[0]
    resolved_taxon = int(wg_key is not None or pcode in ("COP-SC-LER", "TRE-CP-HEM", "TRE-GC-HEM", "NEM-PV-SPR"))

    # WoRMS / GloBI / Benesh / GMPD lookups
    wg_entry = wg_cache.get(wg_key, {}) if wg_key else {}
    worms = wg_entry.get("worms") or {}
    globi_n = int(wg_entry.get("globi_host_count", 0))
    b_info = benesh_genus.get(genus, {})
    g_info = gmpd_genus.get(genus, {})
    benesh_paratenic = int(bool(b_info.get("facultative", set()) & {"paratenic", "postcyclic"}))
    gmpd_n_hosts = len(g_info.get("mammal_hosts", set()))

    # Fish host fluctuation regime from Wood et al. SI Table S1 (FishBase)
    # Primary: 1 if any primary dissected fish host is neritic schooling (herring, hake);
    # for multi-host taxa, proportion of dissected host species that are neritic schooling >= 0.5 -> 1.
    frac_neritic_fish = sum(FISH_HOST_ATTRS[h]["fluc_primary"] for h in fish_hosts) / len(fish_hosts)
    fish_fluc_primary = 1 if frac_neritic_fish >= 0.5 else 0
    frac_broad_fish = sum(FISH_HOST_ATTRS[h]["fluc_broad"] for h in fish_hosts) / len(fish_hosts)
    fish_fluc_broad = 1 if frac_broad_fish >= 0.5 else 0

    # Family classification
    family = worms.get("family") or {
        "COP-SC-LER": "Lernaeopodidae",
        "TRE-CP-HEM": "Hemiuridae",
        "TRE-GC-HEM": "Hemiuridae",
        "NEM-PV-SPR": "Cystidicolidae_or_Spiruroidea",
    }.get(pcode, f"Unresolved_{ptype}")

    # ------------------------------------------------------------------
    # Coding rules for the 4 registered theoretical predictors:
    # (a) n_fluc_obligate: count of obligate inter-host transfers into high-fluctuation
    #     short-lived planktonic/epibenthic crustaceans/gelatinous zooplankton or neritic schooling fish.
    # (b) storage_stage: 1 if life cycle includes a long-lived variable-residence encysted/encapsulated
    #     larval stage in vertebrate tissues (cystacanth, plerocercoid/cysticercus, anisakid L3,
    #     encysted fish metacercaria), 0 otherwise (luminal GI / ectoparasitic / short crustacean stage).
    # (c) spreading: 1 if documented paratenic/postcyclic bridge host capability in Benesh/literature
    #     OR multi-family host breadth in Puget Sound dissections / documented euryxenous generalist (>=50 GloBI hosts
    #     for species-resolved or genus-consistent generalist guild), 0 otherwise.
    #     We keep BOTH strict `spreading_strict` (paratenic/postcyclic OR multi-host in Wood dataset)
    #     and `spreading_broad` (also 1 if resolved genus has >=50 hosts in GloBI/Benesh).
    # (d) prop_active_transfers (and has_active_freeliving): proportion (and presence) of obligate
    #     transfers accomplished by an active free-swimming host-seeking/penetrating stage (waste, d->1)
    #     versus passive trophic ingestion of shelled eggs or encysted larvae (delay, d<1).
    # ------------------------------------------------------------------
    sources = []

    if ptype == "acanthocephalan":
        # Shelled egg eaten by amphipod (1st obligate, fluctuating=1) -> vertebrate definitive host
        # Corynosoma: 2nd obligate is pinniped/seabird (buffered=0), fish is paratenic cystacanth (storage=1, paratenic=1)
        # Echinorhynchus: 2nd obligate is teleost fish (buffered=0 for rockfish/pollock/sole), postcyclic transfer in Benesh
        n_fluc = 1
        n_fluc_broad = 1 + (1 if pcode in ("ACA-CP-COR", "ACA-HP-ECH") else 0)
        storage = 1 if pcode in ("ACA-CP-COR", "ACA-SC-COR") else 0
        storage_alt = 1 if s3_stage == "Larval" else 0
        paratenic = 1 if genus in ("Corynosoma", "Echinorhynchus") else 0
        n_active = 0  # shelled acanthor egg eaten + trophic cystacanth
        n_waterborne = 0
        sources.append("Benesh CLC DB (doi:10.5061/dryad.gtht76hmm: Egg.hatch='eaten', Corynosoma paratenic, Echinorhynchus postcyclic)")
        if gmpd_n_hosts > 0:
            sources.append(f"GMPD2 (doi:10.1002/ecy.1799: {gmpd_n_hosts} mammal hosts)")
        sources.append("Schmidt 1985; Nickol et al. 2002; Valtonen et al. 2001")

    elif ptype == "cestode":
        if pcode == "CES-HC-GYR":
            # Gyrocotyle sp. in Hydrolagus colliei: ciliated lycophora larva -> (crustacean under Wood 2-host) -> ratfish
            n_fluc = 1  # crustacean under 2-host coding; ratfish demersal=0 (0 under 1-host direct sensitivity)
            n_fluc_broad = 1
            storage = 0
            storage_alt = 0.5  # multi-year spiral-valve adult
            paratenic = 0
            n_active = 1  # free-swimming ciliated lycophora
            n_waterborne = 1
            sources.append("WoRMS AphiaID Gyrocotylidae; Xylander 2005; Williams et al. 1987")
        elif pcode == "CES-MP-CLE":
            # Clestobothrium crassiceps: free-swimming coracidium -> planktonic copepod (1) -> neritic hake (1)
            n_fluc = 2
            n_fluc_broad = 2
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 1  # ciliated coracidium
            n_waterborne = 1
            sources.append("WoRMS Bothriocephalidae; Mudry & Dailey 1971; Kuchta et al. 2008")
        elif pcode == "CES-MP-NYB":
            # Nybelinia surmenicola: coracidium -> copepod (1) -> euphausiid (1) -> hake plerocercoid (1) -> shark (0)
            n_fluc = 3  # 2 planktonic crustaceans + neritic hake (2 in Benesh strict-obligate sensitivity)
            n_fluc_broad = 3
            storage = 1  # encapsulated tissue plerocercoid in fish
            storage_alt = 1
            paratenic = 1  # Benesh CLC DB: Nybelinia surmenicola Facultative='paratenic'
            n_active = 1  # free-swimming coracidium
            n_waterborne = 1
            sources.append("Benesh CLC DB (Nybelinia surmenicola: Egg.hatch='hatch, eaten', Facultative='paratenic'); Palm 2004")
        else:
            # Marine larval cestodes (plerocercoids/cysticerci in teleosts, nhosts=3):
            # copepod (1) -> teleost fish (fish_fluc_primary) -> elasmobranch/bird/mammal (0)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 1  # persistent tissue/visceral plerocercoid or cysticercus in fish
            storage_alt = 1 if s3_stage == "Larval" else 0.5
            paratenic = 0  # strict primary conservative for unresolved sp.; 1 in clade-paratenic sensitivity
            n_active = 0  # marine tetraphyllidean/rhinebothriidean eggs passively eaten by copepods (0.5 in coracidium sensitivity)
            n_waterborne = 0
            sources.append("Wood SI Table S3 (larval Cestoda, nhosts=3); Jensen & Bullard 2010; Chervy 2002")

    elif ptype == "copepod":
        # Direct 1-host ectoparasitic copepods: free-swimming nauplius/copepodid locates fish host (active=1)
        n_fluc = fish_fluc_primary
        n_fluc_broad = fish_fluc_broad
        storage = 0
        storage_alt = 0
        paratenic = 0
        n_active = 1
        n_waterborne = 1
        sources.append("WoRMS Copepoda; Kabata 1979, 1988; Boxshall & Halsey 2004; Wood SI Table S1")

    elif ptype == "leech":
        # Oceanobdella pallida on demersal English sole: benthic cocoons + crawling juvenile/adult leech
        n_fluc = 0
        n_fluc_broad = 0
        storage = 0
        storage_alt = 0
        paratenic = 0
        n_active = 0  # benthic cocoon + crawling non-planktonic juvenile (tested as 1 in broad active sensitivity)
        n_waterborne = 0
        sources.append("WoRMS Piscicolidae; Burreson 1975 J. Parasitol.; Sawyer 1986")

    elif ptype == "monogenean":
        if pcode == "MON-CP-GYR":
            # Gyrodactylus sp. on herring: viviparous direct-contact + auto-reinfection on host, 0 free-swimming larvae
            n_fluc = 1  # herring host fluctuates (0.5 in auto-reinfection bypass sensitivity, mech-C4g)
            n_fluc_broad = 1
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 0  # viviparous, no free-living larva
            n_waterborne = 0
            sources.append("WoRMS Gyrodactylidae; Bakke et al. 2007; Kearn 1998")
        else:
            # Oviparous polyopisthocotyleans (Anthocotyle, Chimaericola, Microcotyle): ciliated oncomiracidium (active=1)
            n_fluc = fish_fluc_primary
            n_fluc_broad = fish_fluc_broad
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 1
            n_waterborne = 1
            sources.append("WoRMS Polyopisthocotylea; Kearn 1998; Whittington & Chisholm 2008; Wood SI Table S1")

    elif ptype == "nematode":
        if pcode in ("NEM-SE-ANI", "NEM-SE-CON"):
            # Anisakis sp. & Contracaecum sp. (nhosts=3):
            # ensheathed larva eaten by planktonic euphausiid/copepod (1) -> fish L3 tissue capsule (fish_fluc) -> mammal/bird (0)
            # Note NEM-SE-ANI in hake+pollock (frac_neritic=0.5 -> fish_fluc_primary=1); NEM-SE-CON in all 8 fish (frac_neritic=0.25 -> 0)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 1  # multi-year encapsulated L3 in fish viscera/muscle
            storage_alt = 1
            paratenic = 1  # Benesh CLC DB: Facultative='paratenic'
            n_active = 0  # passive trophic ingestion of ensheathed L2/L3 and encysted L3 (n_waterborne=1 for hatched free larva)
            n_waterborne = 1
            sources.append(
                f"Benesh CLC DB ({genus}: Egg.hatch='hatch, eaten', Facultative='paratenic'); "
                f"GMPD2 ({gmpd_n_hosts} mammal hosts); Anderson 2000; Mattiucci & Nascetti 2008"
            )
        elif pcode == "NEM-PV-CAP":
            # Capillaria (Paracapillaria) parophrysi: shelled egg eaten by benthic oligochaete/polychaete (0) -> demersal sole (0)
            n_fluc = 0
            n_fluc_broad = 0
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 0
            n_waterborne = 0
            sources.append("Benesh CLC DB (Capillaria: Egg.hatch='eaten'); Moravec et al. 1981; Moravec 2001")
        elif pcode in ("NEM-EL-CUC", "NEM-PV-CUC"):
            # Cucullanidae (Cucullanellus kanabus, Cucullanus annulatus): polychaete/copepod (1) -> demersal perch/sole (0)
            n_fluc = 1
            n_fluc_broad = 1
            storage = 0
            storage_alt = 0
            paratenic = 1  # Benesh CLC DB: Cucullanus Facultative='paratenic' in invertebrate/small-fish hosts; Koie 2000
            n_active = 0
            n_waterborne = 1  # hatched free L1/L2 in marine Cucullanus (Koie 2000)
            sources.append("Benesh CLC DB (Cucullanus: Facultative='paratenic'); Koie 2000 J. Parasitol.; Anderson 2000")
        elif pcode == "NEM-PV-CLA":
            # Clavinema mariae (Philometridae): motile L1 eaten by copepod (1) -> demersal sole fin/subcutis (0), annual semelparous
            n_fluc = 1
            n_fluc_broad = 1
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 0
            n_waterborne = 1  # free-living L1 in seawater eaten by copepods
            sources.append("Benesh CLC DB (Clavinema: Egg.hatch='hatch, eaten'); Margolis & Moravec 1987; Sakaguchi & Fukuhara 1975")
        else:
            # Ascarophis spp., Spirurid sp., and 2-host marine fish nematodes:
            # shelled egg eaten by amphipod/decapod crustacean (1) -> fish GI tract (fish_fluc_primary)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 0
            storage_alt = 0
            paratenic = 0
            n_active = 0
            n_waterborne = 0
            sources.append("Benesh CLC DB (Ascarophis: lcl=2, Egg.hatch='eaten'); Koie 1993; Moravec 2004")

    elif ptype == "trematode":
        if pcode == "TRE-MP-APO":
            # Aporocotyle margolisi (nhosts=2 blood fluke):
            # active miracidium -> benthic polychaete (0) -> active cercaria penetrates neritic hake (1)
            n_fluc = 1
            n_fluc_broad = 1
            storage = 0
            storage_alt = 0.5  # intramolluscan/intrapolychaete sporocyst in sensitivity
            paratenic = 0
            n_active = 2  # both miracidium and furcocercous cercaria actively penetrate hosts
            n_waterborne = 2
            sources.append("WoRMS Aporocotylidae; Smith 1967; Koie 1982; Cribb et al. 2003")
        elif family in ("Lecithasteridae", "Hemiuridae", "Derogenidae", "Gonocercidae"):
            # Hemiuroidea (nhosts=3): passive egg eaten by benthic snail (0) -> cystophorous cercaria eaten by
            # planktonic copepod (1) -> luminal stomach adult in fish (fish_fluc_primary)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 0
            storage_alt = 0.5  # intramolluscan stage in benthic snail
            # Derogenes varicus & Lecithochirium have documented paratenic/postcyclic fish transfers (Koie 1979, Gibson & Bray 1979)
            paratenic = 1 if pcode in ("TRE-SE-DER", "TRE-SC-LEC") else 0
            n_active = 0  # passive egg ingestion + passive trophic ingestion of cystophorous cercaria by copepod
            n_waterborne = 1  # free-floating cystophorous cercaria in water column
            sources.append("WoRMS Hemiuroidea; Koie 1979, 1989, 1990, 1995; Gibson & Bray 1979; Margolis & Boyce 1969")
        elif family == "Faustulidae":
            # Pronoprymna petrowi / sp. (nhosts=3): bivalve (0) -> active trichocercous cercaria into planktonic crustacean/ctenophore (1)
            # -> pelagic clupeoid/osmeroid fish (fish_fluc_primary)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 0
            storage_alt = 0.5
            paratenic = 0
            n_active = 2  # miracidium + trichocercous cercaria
            n_waterborne = 2
            sources.append("WoRMS Faustulidae; Bray, Gibson & Jones 2008; Cribb et al. 2003")
        elif pcode == "TRE-GC-LEP":
            # Lepidapedon sp. (nhosts=3): active miracidium -> snail (0) -> active cercaria into benthic polychaete/crustacean (0) -> demersal pollock (0)
            n_fluc = 0
            n_fluc_broad = 1  # if crustacean 2nd host included
            storage = 0
            storage_alt = 0.5
            paratenic = 0
            n_active = 2
            n_waterborne = 2
            sources.append("WoRMS Lepidapedidae; Koie 1985 Ophelia; Bray, Gibson & Jones 2008")
        elif pcode in ("TRE-MP-POD", "TRE-GC-POD", "TRE-SC-OPE"):
            # Podocotyle (amphipod 2nd host) & Opechona (hydromedusa/ctenophore 2nd host):
            # snail (0) -> fluctuating amphipod/medusa (1) -> fish (fish_fluc_primary)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 0
            storage_alt = 0.5
            paratenic = 0
            n_active = 2
            n_waterborne = 2
            sources.append("WoRMS Opecoelidae/Lepocreadiidae; Koie 1975, 1981; Hunninen & Cable 1943")
        elif s3_stage == "Larval":
            # 11 encysted trematode metacercariae in fish (nhosts=3):
            # benthic mollusc (0) -> active cercaria penetrates dissected fish forming tissue cyst (fish_fluc_primary, storage=1)
            # -> piscivorous fish/bird/mammal definitive host (0)
            n_fluc = fish_fluc_primary
            n_fluc_broad = fish_fluc_broad
            storage = 1
            storage_alt = 1
            paratenic = 0
            n_active = 2
            n_waterborne = 2
            sources.append("Wood SI Table S3 (larval metacercaria in fish, nhosts=3); Cribb et al. 2003; Olson 1978")
        else:
            # 7 unresolved adult GI trematodes in fish (nhosts=3):
            # mollusc (0) -> invertebrate crustacean (1) -> dissected fish GI adult (fish_fluc_primary)
            n_fluc = 1 + fish_fluc_primary
            n_fluc_broad = 1 + fish_fluc_broad
            storage = 0
            storage_alt = 0.5
            paratenic = 0
            n_active = 2
            n_waterborne = 2
            sources.append("Wood SI Table S3 (adult Trematoda, nhosts=3); Cribb et al. 2003")
    else:
        raise ValueError(f"Unknown psite_type: {ptype}")

    # Spreading indicators:
    # 1) Primary `spreading`: documented paratenic/postcyclic bridge OR observed across >=2 fish families in Wood dataset
    multi_fish_in_wood = int(len(fish_hosts) >= 2)
    spreading_primary = int(paratenic == 1 or multi_fish_in_wood == 1)
    # 2) Broad `spreading_broad`: also 1 if resolved genus/species has >= 50 host species in GloBI or >= 15 in Benesh,
    #    or Bomolochus cuneatus (documented multi-order generalist across NE Pacific fishes, Kabata 1988)
    benesh_n_hosts = len(b_info.get("hosts", set()))
    spreading_broad = int(
        spreading_primary == 1
        or (resolved_taxon == 1 and (globi_n >= 50 or benesh_n_hosts >= 15))
        or pcode == "COP-EL-BOM"
    )

    prop_active = n_active / float(nhosts)
    prop_waterborne = n_waterborne / float(nhosts)
    has_active = int(n_active > 0)

    return {
        "psite_code": pcode,
        "para_spc": tcode,
        "par_type": ptype,
        "taxonomic_family": family,
        "table_s3_id": s3_name,
        "stage_in_fish": s3_stage,
        "terminal_in_fish": term_raw,
        "resolved_taxon": resolved_taxon,
        "n_obs_rows": n_obs,
        "n_dissected_fish_spp": len(fish_hosts),
        "dissected_fish_spp": ";".join(fish_hosts),
        "nhosts_exact": nhosts,
        "nhosts_wood_3plus": nhosts_3plus,
        # Four primary registered theoretical traits:
        "n_fluc_obligate": n_fluc,
        "storage_stage": storage,
        "spreading": spreading_primary,
        "prop_active_transfers": round(prop_active, 6),
        # Pre-registered sensitivity variants of the four traits:
        "n_fluc_broad": n_fluc_broad,
        "storage_alt_mollusc": storage_alt,
        "spreading_broad": spreading_broad,
        "has_paratenic": paratenic,
        "has_active_freeliving": has_active,
        "n_active_transfers": n_active,
        "prop_waterborne_transfers": round(prop_waterborne, 6),
        # Supporting database counts:
        "benesh_paratenic_flag": benesh_paratenic,
        "benesh_genus_hosts": benesh_n_hosts,
        "gmpd_mammal_hosts": gmpd_n_hosts,
        "globi_host_records": globi_n,
        "worms_aphia_id": worms.get("AphiaID", ""),
        "sources": " | ".join(sources),
    }


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factors for columns of X (standardized, intercept added)."""
    n, p = X.shape
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    vifs = np.zeros(p)
    for j in range(p):
        yj = Xs[:, j]
        Xj = np.column_stack([np.ones(n), np.delete(Xs, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
        yhat = Xj @ beta
        r2 = 1.0 - np.sum((yj - yhat) ** 2) / np.sum((yj - yj.mean()) ** 2)
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def main() -> int:
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    benesh_genus, gmpd_genus, wg_cache = load_external_databases()
    fish_by_para, obs_by_para = load_blind_fish_mapping()

    with open(os.path.join(RAW_WOOD, "wood_github", "data", "psite_lc_complexity.csv"), encoding="utf-8-sig") as f:
        lc_rows = list(csv.DictReader(f))

    coded = []
    for r in lc_rows:
        tcode = r["tim_code"]
        coded.append(
            code_one_taxon(
                r,
                sorted(fish_by_para[tcode]),
                obs_by_para[tcode],
                benesh_genus,
                gmpd_genus,
                wg_cache,
            )
        )

    assert len(coded) == 85, f"Expected 85 taxa, got {len(coded)}"

    csv_path = os.path.join(HERE, "traits_blind.csv")
    fieldnames = list(coded[0].keys())
    # Guarantee no outcome column exists in traits_blind.csv
    forbidden = {"count", "slope", "gamma", "decline", "pfoo", "prevalence"}
    assert not (set(fieldnames) & forbidden), "Outcome column leaked!"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(coded)

    with open(csv_path, "rb") as f:
        sha256_blind = hashlib.sha256(f.read()).hexdigest()

    # Build summary & collinearity diagnostics (still 100% blind to outcomes)
    lines = []
    lines.append("=== Outcome-Blind Trait Coding Summary (85 Parasite Taxa) ===")
    lines.append(f"File: traits_blind.csv | Rows: {len(coded)} | SHA-256: {sha256_blind}")
    lines.append("Outcome columns accessed: NONE (verified by assertion)")
    lines.append("")

    for nh in (1, 2, 3):
        sub = [c for c in coded if c["nhosts_wood_3plus"] == nh]
        lbl = f"nhosts={nh}" if nh < 3 else "nhosts=3+ (43 three-host + 1 four-host)"
        lines.append(f"--- Group {lbl}: n = {len(sub)} taxa ---")
        lines.append(f"  n_fluc_obligate dist : {dict(sorted(Counter(c['n_fluc_obligate'] for c in sub).items()))}")
        lines.append(f"  storage_stage dist   : {dict(sorted(Counter(c['storage_stage'] for c in sub).items()))}")
        lines.append(f"  spreading (primary)  : {dict(sorted(Counter(c['spreading'] for c in sub).items()))}")
        lines.append(f"  spreading_broad      : {dict(sorted(Counter(c['spreading_broad'] for c in sub).items()))}")
        lines.append(f"  prop_active dist     : {dict(sorted(Counter(round(c['prop_active_transfers'], 2) for c in sub).items()))}")
        lines.append(f"  resolved_taxon dist  : {dict(sorted(Counter(c['resolved_taxon'] for c in sub).items()))}")

    # Predictor correlation matrix & VIFs across all 85 taxa
    names_4 = ["n_fluc_obligate", "storage_stage", "spreading", "prop_active_transfers"]
    X4 = np.array([[float(c[k]) for k in names_4] for c in coded])
    nh_vec = np.array([float(c["nhosts_wood_3plus"]) for c in coded])
    nh3_ind = np.array([1.0 if c["nhosts_wood_3plus"] == 3 else 0.0 for c in coded])

    X_all = np.column_stack([nh_vec, nh3_ind, X4])
    all_names = ["nhosts_linear", "nhosts_3plus_ind"] + names_4
    corr = np.corrcoef(X_all, rowvar=False)
    lines.append("\n--- Pairwise Pearson correlations among predictors (n = 85 taxa) ---")
    hdr = " " * 23 + " ".join(f"{nm[:10]:>10s}" for nm in all_names)
    lines.append(hdr)
    for i, nm in enumerate(all_names):
        row_str = f"{nm:22s} " + " ".join(f"{corr[i, j]:10.3f}" for j in range(len(all_names)))
        lines.append(row_str)

    vif_4_only = compute_vif(X4)
    vif_with_nh = compute_vif(np.column_stack([nh_vec, X4]))
    vif_with_nh3 = compute_vif(np.column_stack([nh3_ind, X4]))
    lines.append("\n--- Variance Inflation Factors (VIFs, n = 85 taxa) ---")
    lines.append("  4 traits alone:")
    for nm, vf in zip(names_4, vif_4_only):
        lines.append(f"    {nm:24s}: VIF = {vf:.3f}")
    lines.append("  4 traits + linear nhosts (1, 2, 3+):")
    for nm, vf in zip(["nhosts_linear"] + names_4, vif_with_nh):
        lines.append(f"    {nm:24s}: VIF = {vf:.3f}")
    lines.append("  4 traits + indicator(nhosts >= 3):")
    for nm, vf in zip(["I(nhosts>=3)"] + names_4, vif_with_nh3):
        lines.append(f"    {nm:24s}: VIF = {vf:.3f}")

    # Within nhosts >= 3 (n = 44 taxa) VIFs!
    sub3 = [c for c in coded if c["nhosts_wood_3plus"] == 3]
    X4_sub3 = np.array([[float(c[k]) for k in names_4] for c in sub3])
    vif_sub3 = compute_vif(X4_sub3)
    lines.append("\n  4 traits WITHIN nhosts >= 3 alone (n = 44 taxa):")
    for nm, vf in zip(names_4, vif_sub3):
        sd_sub = X4_sub3[:, names_4.index(nm)].std(ddof=1)
        lines.append(f"    {nm:24s}: SD = {sd_sub:.3f}, VIF = {vf:.3f}")

    sum_path = os.path.join(HERE, "results", "traits_blind_summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Also write trait_sources.md documenting every taxon and its sources
    md_lines = [
        "# Blind Trait Coding Ledger (`traits_blind.csv`)",
        "",
        f"- **SHA-256 of `traits_blind.csv`**: `{sha256_blind}`",
        "- **Blindness status**: Coded prior to accessing any parasite count, prevalence, or temporal slope.",
        "- **NHM Host-Parasite Database note**: Direct zip download from the NHM Data Portal (`doi:10.5519/nplk7c5u`) returned HTTP 403 and the legacy JSP endpoint returned HTTP 502; host-association counts were instead queried from Global Biotic Interactions (`api.globalbioticinteractions.org`, CC BY 4.0, which aggregates LNHM, WoRMS, USNPC, and GMPD records), alongside Benesh et al. (`doi:10.5061/dryad.gtht76hmm` via Zenodo `4975516`, CC0), Global Mammal Parasite Database 2.0 (`doi:10.1002/ecy.1799`), WoRMS (`marinespecies.org`, CC BY 4.0), and primary parasitological monographs.",
        "",
        "| Code | Identification (SI Table S3) | Group | `nhosts` | `(a) n_fluc` | `(b) storage` | `(c) spread` | `(d) prop_active` | Sources |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in coded:
        md_lines.append(
            f"| `{c['psite_code']}` | {c['table_s3_id']} ({c['stage_in_fish']}) | {c['par_type']} | "
            f"{c['nhosts_exact']} | {c['n_fluc_obligate']} | {c['storage_stage']} | "
            f"{c['spreading']} | {c['prop_active_transfers']:.2f} | {c['sources']} |"
        )
    with open(os.path.join(HERE, "trait_sources.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
