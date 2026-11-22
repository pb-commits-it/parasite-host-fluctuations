#!/usr/bin/env python3
"""Independent raw-data loader and eligibility filter for case study `lpd`.

Reads raw files directly from `data/raw/lpd/` (`gpdd/`, `biotime/`, `vecdyn/full_datasets/`)
and implements the registered inclusion, taxonomic, effort-normalization, and regularity
rules from `protocol.md` §§2.1–2.3 and Amendment 2. Also audits protocol-vs-script
edge cases (BioTIME seasonal cutoffs, GPDD log-transform flags, GPDD benthic insects).
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

REPO_ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../..').resolve()
RAW_LPD = REPO_ROOT / "data" / "raw" / "lpd"
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()

# ---------------------------------------------------------------------------
# Taxonomic sets built from protocol.md §2.2
# ---------------------------------------------------------------------------

SMALL_MAMMAL_ORDERS = {
    "rodentia", "lagomorpha", "eulipotyphla", "soricomorpha", "erinaceomorpha",
    "insectivora", "didelphimorphia", "scandentia", "macroscelidea",
    "dasyuromorphia", "afrosoricida", "microbiotheria", "paucituberculata",
}

# Genera of Rodentia, Lagomorpha, Eulipotyphla/Insectivora, Didelphimorphia, Dasyuromorphia, Macroscelidea
# (excluding Carnivora, Artiodactyla/Cetacea, Perissodactyla, Chiroptera, Primates, Proboscidea, Pilosa/Cingulata)
SMALL_MAMMAL_GENERA = {
    # Rodents (Muridae, Cricetidae, Sciuridae, Heteromyidae, Geomyidae, Dipodidae, Caviidae, Echimyidae, etc.)
    "abrocoma", "abrothrix", "acomys", "aethomys", "akodon", "alticola", "altilcola",
    "ammospermophilus", "apodemus", "arvicanthis", "arvicanthus", "arvicola",
    "baiomys", "bandicota", "bolomys", "callosciurus", "calomys", "castor",
    "cavia", "chaetodipus", "chelemys", "chetodipus", "chinchilla", "chionomys",
    "clethrionomys", "cricetulus", "cricetus", "ctenomys", "cynomys", "dasyprocta",
    "dendromus", "dipodomys", "dipus", "dryomys", "eliomys", "erethizon", "eutamias",
    "funisciurus", "galea", "geomys", "gerbilliscus", "gerbillurus", "gerbillus",
    "glaucomys", "glis", "grammomys", "graphiurus", "heliosciurus", "heteromys",
    "hydrochoerus", "hylomyscus", "jaculus", "lagidium", "lemmiscomys", "lemmus",
    "liomys", "lophuromys", "marmota", "mastomys", "meriones", "mesocricetus",
    "microcavia", "micromys", "microsciurus", "microtus", "mus", "muscardinus",
    "myocastor", "myodes", "myoprocta", "myopus", "mysateles", "napaeozapus",
    "nectomys", "neotoma", "neotomodon", "neozapus", "nesokia", "niviventer",
    "notomys", "ochrotomys", "octodon", "oligoryzomys", "ondatra", "onychomys",
    "oryzomys", "otomys", "oxymycterus", "paraxerus", "pedetes", "perognathus",
    "peromyscus", "petaurista", "phenacomys", "phyllotis", "pitymys", "podomys",
    "praomys", "proechimys", "pseudomys", "rattus", "reithrodontomys", "rhabdomys",
    "rhombomys", "saccostomus", "sciurus", "scotinomys", "sicista", "sigmodon",
    "spalacopus", "spalax", "spermophilus", "steatomys", "synaptomys", "tamias",
    "tamiasciurus", "tatera", "taterillus", "thomomys", "tylomys", "urocitellus",
    "zapus", "zelatomys", "zygodontomys",
    # Lagomorpha
    "lepus", "ochotona", "oryctolagus", "sylvilagus",
    # Shrews, moles, hedgehogs (Eulipotyphla / Insectivora)
    "atelerix", "blarina", "condylura", "crocidura", "cryptotis", "erinaceus",
    "hemiechinus", "neomys", "neurotrichus", "notiosorex", "parascalops", "scalopus",
    "scapanus", "sorex", "suncus", "talpa",
    # Didelphimorphia, Dasyuromorphia, Macroscelidea, Scandentia, Afrosoricida
    "antechinus", "caluromys", "chironectes", "dasyurus", "didelphis", "elephantulus",
    "gracilinanus", "lutreolina", "macroscelides", "marmosa", "marmosops",
    "metachirus", "micoureus", "monodelphis", "ningaui", "petrodromus", "phascogale",
    "philander", "planigale", "rhynchocyon", "sminthopsis", "tenrec", "thylamys", "tupaia",
}

LARGE_OR_EXCLUDED_MAMMAL_GENERA = {
    # Ungulates, carnivores, bats, primates, xenarthrans, peramelemorphs, non-mammals in mixed plots
    "alces", "antilocapra", "axis", "bison", "bos", "bubalus", "camelus", "canis",
    "capra", "capreolus", "cervus", "connochaetes", "crocuta", "dama", "dasypus",
    "diceros", "equus", "eudorcas", "felis", "gazella", "giraffa", "gulo", "halichoerus",
    "herpestes", "hippopotamus", "hyaena", "kobus", "lama", "leopardus", "loxodonta",
    "lontra", "lutra", "lynx", "macaca", "macropus", "meles", "mephitis", "mustela",
    "myotis", "nasua", "neovison", "nyctereutes", "odocoileus", "oreamnos", "oryx",
    "otocyon", "ovis", "panthera", "papio", "pecari", "phoca", "pipistrellus",
    "procyon", "puma", "rangifer", "redunca", "rhinolophus", "rupicapra", "spilogale",
    "sus", "syncerus", "tapirus", "taxidea", "tragelaphus", "trichosurus", "ursus",
    "vicugna", "vulpes", "wallabia", "zalophus",
    # Non-mammal intruders in BioTIME mammal studies
    "amphispiza", "campylorhynchus", "crotalus", "haliotis", "pipilio", "pipilo",
    "pooecetes", "sceloporus",
}

WATERFOWL_GULL_ORDERS = {
    "anseriformes", "charadriiformes", "pelecaniformes", "suliformes",
    "gaviiformes", "podicipediformes", "ciconiiformes",
}

WATERFOWL_GULL_GENERA = {
    "achmophorus", "actitis", "aechmophorus", "aix", "alle", "alopochen", "amazonetta",
    "anas", "anhinga", "anser", "ardea", "ardeola", "arenaria", "aythya", "bartramia",
    "biziura", "botaurus", "brachyramphus", "branta", "bubulcus", "bucephala", "burhinus",
    "butorides", "cairina", "calidris", "callonetta", "camptorhynchus", "casmerodius",
    "catharacta", "catoptrophorus", "cepphus", "cereopsis", "cerorhinca", "charadrius",
    "chen", "chenonetta", "chlidonias", "chloephaga", "chroicocephalus", "ciconia",
    "cladorhynchus", "clangula", "cochlearius", "coscoroba", "creagrus", "cursorius",
    "cyanochen", "cygnus", "dendrocygna", "dromas", "egretta", "eudocimus", "eudromias",
    "fratercula", "fregata", "gallinago", "gavia", "gelochelidon", "glareola",
    "haematopus", "heteronetta", "himantopus", "histrionicus", "hydrocoloeus",
    "hydrophasianus", "hydroprogne", "hymenolaimus", "ibidorhyncha", "ichthyaetus",
    "ixobrychus", "jabiru", "jacana", "larosterna", "larus", "leptoptilos", "leucophaeus",
    "limicola", "limnodromus", "limosa", "lophoetta", "lophodytes", "lunda",
    "lymnocryptes", "malacorhynchus", "mareca", "marmaronetta", "melanitta", "merganetta",
    "mergellus", "mergus", "microcarbo", "micropalama", "morus", "mycteria", "netta",
    "nettapus", "nomonyx", "numenius", "nychophus", "nyctanassa", "nycticorax",
    "onychoprion", "oxyura", "pagophila", "pelecanus", "phaetusa", "phalacrocorax",
    "phalaropus", "philomachus", "platalea", "plegadis", "pluvialis", "pluvianellus",
    "pluvianus", "podiceps", "podilymbus", "poliocephalus", "polysticta", "procelsterna",
    "pteronetta", "ptychoramphus", "ptycoramphus", "recurvirostra", "rhodonessa",
    "rhodostethia", "rollandia", "rostratula", "rychops", "rynchops", "salvadorina",
    "sarkidiornis", "scolopax", "somateria", "spatula", "stercorarius", "sterna",
    "sternula", "stictonetta", "sula", "synthliboramphus", "tachyeres", "tachybaptus",
    "tadorna", "thalasseus", "thalassornis", "thinocorus", "tigrisoma", "tringa",
    "tryngites", "uria", "urile", "vanellus", "xema", "xenus",
}

FISH_CLASSES = {
    "actinopterygii", "osteichthyes", "chondrichthyes", "chondrichtyhes",
    "cephalaspidomorphi", "hyperoartia", "petromyzontida", "myxini", "pisces",
    "elasmobranchii", "holocephali", "sarcopterygii",
}

SNAIL_GENERA = {
    "acella", "acroloxus", "amnicola", "ancylus", "anisus", "aplexa", "armiger",
    "assiminea", "bathyomphalus", "bellamya", "biomphalaria", "bithynia", "bulinus",
    "bythinella", "campeloma", "cipangopaludina", "cleopatra", "ecrobia", "elimia",
    "ferrissia", "fluminicola", "fontigens", "fossaria", "gabbiella", "galba",
    "goniobasis", "gyraulus", "hebetancylus", "helisoma", "hippeutis", "hydrobia",
    "juga", "laevapex", "lanistes", "leptoxis", "lioplax", "lymnaea", "marisa",
    "marstonia", "melanoides", "menetus", "neritina", "oncomelania", "oxyloma",
    "peringia", "physa", "physella", "pila", "planorbarius", "planorbella",
    "planorbis", "pleurocera", "pomacea", "potamopyrgus", "promenetus",
    "pseudosuccinea", "pyrgulopsis", "radix", "segmentina", "semisulcospira",
    "somatogyrus", "stagnicola", "succinea", "tarebia", "theodoxus", "thiara",
    "tricula", "valvata", "viviparus",
}

PHYTO_PLANT_GENERA = {
    "anabaena", "aphanizomenon", "aphanocapsa", "aphanothece", "asterionella",
    "aulacoseira", "belonastrum", "ceratium", "chlorella", "chroomonas",
    "closterium", "coelastrum", "cosmarium", "cryptomonas", "cyclotella",
    "cymbella", "dinobryon", "eudorina", "fragilaria", "gomphonema", "gymnodinium",
    "mallomonas", "melosira", "merismopedia", "microcystis", "mougeotia",
    "navicula", "nitzschia", "ochromonas", "oscillatoria", "pandorina",
    "pediastrum", "peridinium", "planktosphaeria", "rhodomonas", "scenedesmus",
    "schroederia", "staurastrum", "stephanodiscus", "synedra", "synura",
    "tabellaria", "ulothrix", "volvox", "woronichinia",
}

VECTOR_FAMILIES = {
    "culicidae", "simuliidae", "ceratopogonidae", "psychodidae",
    "tabanidae", "glossinidae", "pulicidae", "ceratophyllidae",
    "leptopsyllidae", "vermipsyllidae", "reduviidae", "pediculidae",
}

VECTOR_GENERA = {
    "aedes", "aedimorphus", "anopheles", "armigeres", "bironella", "chagasia",
    "chrysops", "coquillettidia", "culex", "culicoides", "culiseta", "deinocerites",
    "deinoceriites", "eretmapodites", "ficalbia", "glossina", "haemagogus",
    "haematobia", "haematopota", "heizmannia", "hodgesia", "limatus", "lutzia",
    "lutzomyia", "malaya", "mansonia", "melanoconion", "mimomyia", "ochlerotatus",
    "orthopodomyia", "panstrongylus", "phlebotomus", "psorophora", "pulex",
    "rhodnius", "sabethes", "simulium", "stomoxys", "tabanus", "toxorhynchites",
    "triatoma", "trichoprosopon", "tripteroides", "uranotaenia", "verrallina",
    "wyeomyia", "xenopsylla",
}

TICK_FAMILIES = {"ixodidae", "argasidae", "nuttalliellidae"}
TICK_GENERA = {
    "amblyomma", "argas", "boophilus", "dermacentor", "haemaphysalis",
    "hyalomma", "ixodes", "ornithodoros", "otobius", "rhipicephalus",
}

FW_KEYWORDS = ("limnetic", "fluvial", "wetland", "lake", "pond", "river", "stream", "freshwater", "marsh", "bog")


def _gpdd_month(tp: int) -> int:
    if 1 <= tp <= 12:
        return tp
    if tp in (13, 14, 15, 16):
        return {13: 4, 14: 7, 15: 10, 16: 1}[tp]
    if tp in (391, 392, 393, 394):
        return {391: 2, 392: 5, 393: 8, 394: 11}[tp]
    if tp in (70, 71):
        return {70: 1, 71: 7}[tp]
    if tp in (72, 73):
        return {72: 2, 73: 8}[tp]
    if 17 <= tp <= 68:
        return min(12, max(1, int((tp - 17) * 12 // 52) + 1))
    if 395 <= tp <= 407:
        return min(12, max(1, int((tp - 395) * 12 // 13) + 1))
    if 74 <= tp <= 388:
        return min(12, max(1, int((tp - 74) * 12 // 365) + 1))
    return 0


def classify_gpdd(t_row: dict, b_row: dict, strict_protocol_insects: bool = False) -> tuple[str | None, str | None]:
    cls = (t_row.get("TaxonomicClass") or "").strip().lower()
    ord_ = (t_row.get("TaxonomicOrder") or "").strip().lower()
    fam = (t_row.get("TaxonomicFamily") or "").strip().lower()
    gen = (t_row.get("TaxonomicGenus") or "").strip().lower()
    tname = (t_row.get("TaxonName") or "").strip().lower()
    g_use = gen if (gen and gen != "unknown") else (tname.split()[0] if tname else "")

    hab = ((b_row.get("HabitatName") or "") + " " + (b_row.get("BiotopeType") or "")).lower()
    is_marine = ("marine" in hab) or ("tidal" in hab) or ("coastal" in hab and "limnetic" not in hab and "wetland" not in hab)
    is_fw = any(w in hab for w in FW_KEYWORDS)

    if cls == "mammalia":
        if ord_ in SMALL_MAMMAL_ORDERS or g_use in SMALL_MAMMAL_GENERA or g_use == "perameles":
            return ("small_mammals", "rodents" if ord_ == "rodentia" else "other_small_mammals")
        return (None, None)

    if cls == "aves":
        if ord_ in WATERFOWL_GULL_ORDERS or g_use in WATERFOWL_GULL_GENERA:
            return ("birds", "birds_waterfowl_gulls")
        return ("birds", "birds_other")

    if cls in FISH_CLASSES:
        return ("fish", "freshwater_fish" if (is_fw and not is_marine) else "marine_or_diadromous_fish")

    if cls == "insecta":
        if fam in VECTOR_FAMILIES or g_use in VECTOR_GENERA or (strict_protocol_insects and ord_ in ("siphonaptera", "phthiraptera")):
            return ("insects_vectors", fam or "diptera_vector")
        if is_fw and not is_marine:
            if ord_ in ("odonata", "ephemeroptera", "plecoptera", "trichoptera"):
                return ("freshwater_invertebrates", "aquatic_insects_nonvector")
            if strict_protocol_insects and (fam == "chironomidae" or ord_ == "coleoptera"):
                return ("freshwater_invertebrates", "aquatic_insects_nonvector")
        return (None, None)

    if cls == "arachnida":
        if fam in TICK_FAMILIES or g_use in TICK_GENERA:
            return ("arachnid_vectors_ticks", "ticks")
        return (None, None)

    if cls in ("gastropoda", "bivalvia", "crustacea", "rotifera", "oligochaeta", "hirudinea"):
        if is_marine:
            return (None, None)
        if g_use in SNAIL_GENERA or (cls == "gastropoda" and is_fw):
            return ("freshwater_invertebrates", "snails_only")
        if is_fw or cls == "rotifera" or fam in ("cambaridae", "daphniidae", "bosminidae", "chydoridae", "sphaeriidae"):
            return ("freshwater_invertebrates", "other_freshwater_invertebrates")

    return (None, None)


def classify_biotime(st: dict, v2_realm: str, genus: str, sp_full: str) -> tuple[str | None, str | None]:
    taxa = (st.get("TAXA") or "").strip()
    orgs = (st.get("ORGANISMS") or "").strip().lower()
    hab = (st.get("HABITAT") or "").strip().lower()
    realm = (v2_realm or st.get("REALM") or "").strip()
    g = (genus or "").strip().lower()
    if not g and sp_full:
        g = sp_full.strip().split()[0].lower()

    if g in PHYTO_PLANT_GENERA or "phytoplankton" in orgs or "plants" in taxa.lower():
        return (None, None)
    if g in VECTOR_GENERA:
        return ("insects_vectors", "mosquitoes_and_biting_diptera")
    if g in TICK_GENERA:
        return ("arachnid_vectors_ticks", "ticks")

    if taxa == "Mammals":
        if realm == "Marine" or g in LARGE_OR_EXCLUDED_MAMMAL_GENERA:
            return (None, None)
        if g in SMALL_MAMMAL_GENERA or g == "perameles":
            return ("small_mammals", "small_mammals")
        return (None, None)

    marine_megafauna = {
        "balaenoptera", "delphinus", "phocoena", "arctocephalus", "caretta",
        "chelonia", "dermochelys", "stenella", "tursiops", "zalophus",
    }
    if taxa == "Birds" or (taxa == "All" and "seabird" in orgs and g not in marine_megafauna):
        if g in WATERFOWL_GULL_GENERA or "seabird" in orgs or "waterfowl" in orgs or "waterbird" in orgs:
            return ("birds", "birds_waterfowl_gulls")
        return ("birds", "birds_other")

    if taxa == "Fish":
        return ("fish", "freshwater_fish" if realm == "Freshwater" else "marine_or_estuarine_fish")

    if g in SNAIL_GENERA and (realm == "Freshwater" or any(w in hab for w in ("estuar", "lake", "river", "wetland"))):
        return ("freshwater_invertebrates", "snails_only")

    if realm == "Freshwater" and taxa in ("Freshwater invertebrates", "Benthos"):
        return ("freshwater_invertebrates", "other_freshwater_invertebrates")

    return (None, None)


def parse_gpdd(raw_dir: Path, audit_notes: dict) -> list[dict]:
    gpdd_dir = raw_dir / "gpdd"
    with (gpdd_dir / "taxon.csv").open(encoding="latin-1") as f:
        taxa = {r["TaxonID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "biotope.csv").open(encoding="latin-1") as f:
        biotopes = {r["BiotopeID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "location.csv").open(encoding="latin-1") as f:
        locs = {r["LocationID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "main.csv").open(encoding="latin-1") as f:
        mains = {r["MainID"]: r for r in csv.DictReader(f)}

    rows_by_mid: dict[str, list[dict]] = collections.defaultdict(list)
    with (gpdd_dir / "data.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            rows_by_mid[r["MainID"]].append(r)

    out: list[dict] = []
    strict_insect_extra = []
    log_equal_series = []

    for mid, m in mains.items():
        t_row = taxa.get(m["TaxonID"], {})
        b_row = biotopes.get(m["BiotopeID"], {})

        # Audit check: any extra series under strict protocol insect list (Chironomidae / aquatic Coleoptera / Siphonaptera / Phthiraptera)?
        g_strict, _ = classify_gpdd(t_row, b_row, strict_protocol_insects=True)
        g_std, sub_std = classify_gpdd(t_row, b_row, strict_protocol_insects=False)
        if g_strict is not None and g_std is None:
            strict_insect_extra.append((mid, t_row.get("TaxonName", ""), t_row.get("TaxonomicOrder", ""), t_row.get("TaxonomicFamily", "")))
        if g_std is None:
            continue

        sf = (m.get("SamplingFrequency") or "").strip()
        if sf == "1":
            iv_class, spy = "annual", 1.0
        elif sf in ("2", "3", "4"):
            iv_class, spy = "subannual_seasonal", float(sf)
        elif sf in ("6", "12"):
            iv_class, spy = "subannual_monthly", float(sf)
        elif sf in ("52", "52*"):
            iv_class, spy = "subannual_biweekly_weekly", 52.0
        elif sf == "7*":
            iv_class, spy = "subannual_biweekly_weekly", 13.0
        else:
            continue

        st_trans = (m.get("SourceTransform") or "").strip()
        if st_trans in ("Standardised", "Proportion"):
            continue

        drows = rows_by_mid.get(mid, [])
        if len(drows) < 12:
            continue

        notes_low = ((m.get("Notes") or "") + " " + (m.get("SamplingUnits") or "")).lower()
        step_vals: dict[int, list[float]] = collections.defaultdict(list)
        step_tp: dict[int, int] = {}
        step_yr: dict[int, int] = {}
        had_equal_log = False

        for r in drows:
            try:
                step = int(float(r["SeriesStep"]))
                p_untr = float(r["PopulationUntransformed"])
                p_raw = float(r["Population"])
            except (ValueError, TypeError):
                continue
            if p_untr <= -9990.0 or math.isnan(p_untr) or p_untr < 0.0:
                continue
            if st_trans == "Log" and abs(p_untr - p_raw) < 1e-9 and p_untr > 0.0:
                had_equal_log = True
            step_vals[step].append(p_untr)
            try:
                step_tp[step] = int(float(r.get("TimePeriodID") or 0))
            except ValueError:
                step_tp[step] = 0
            try:
                yr = int(float(r.get("SampleYear") or -9999))
                if 1700 <= yr <= 2026 and "generation numbers, not calendar" not in notes_low:
                    step_yr[step] = yr
                else:
                    step_yr.setdefault(step, -9999)
            except ValueError:
                step_yr.setdefault(step, -9999)

        if len(step_vals) < 12:
            continue
        if had_equal_log:
            log_equal_series.append((mid, t_row.get("TaxonName", ""), m.get("Notes", "")))

        steps_sorted = sorted(step_vals.keys())
        loc = locs.get(m.get("LocationID", ""), {})
        try:
            lat = float(loc.get("LatDD") or "nan")
            lon = float(loc.get("LongDD") or "nan")
        except ValueError:
            lat, lon = float("nan"), float("nan")
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            lat, lon = float("nan"), float("nan")
        try:
            rel = int(float(m.get("Reliability") or -1))
        except ValueError:
            rel = -1

        hab_low = ((b_row.get("HabitatName") or "") + " " + (b_row.get("BiotopeType") or "")).lower()
        ocean_str = (loc.get("Ocean") or "").strip().lower()
        is_fw = any(k in hab_low for k in FW_KEYWORDS)
        is_mar = ("marine" in hab_low) or ("tidal" in hab_low) or ("coastal" in hab_low and not is_fw) or bool(
            ocean_str and ocean_str not in ("none", "na", "unknown", "land", "not applicable")
        )
        realm_det = "Freshwater" if is_fw else ("Marine" if is_mar else "Terrestrial")

        lab_txt = " ".join([m.get("Notes", ""), m.get("SamplingUnits", ""), b_row.get("HabitatName", ""), b_row.get("BiotopeType", "")]).lower()
        is_lab = any(
            w in lab_txt
            for w in ("laboratory experiment", "laboratory population", "microcosm", "aquarium", "indoor", "laboratory culture")
        )

        out.append({
            "series_id": f"GPDD_{mid}",
            "database": "GPDD",
            "study_id": f"GPDD_SRC_{m.get('DataSourceID', mid)}",
            "licence": "CC-BY-4.0",
            "cc_by_compatible": True,
            "taxon_group": g_std,
            "taxon_subgroup": sub_std,
            "species": (t_row.get("TaxonName") or "Unknown").strip(),
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm_detail": realm_det,
            "gpdd_ocean": loc.get("Ocean", ""),
            "environment_type": "laboratory" if is_lab else "field",
            "reliability": rel,
            "manipulated": False,
            "series_kind": "primary",
            "units": m.get("SamplingUnits", "unspecified"),
            "steps": np.array(steps_sorted, dtype=int),
            "values": np.array([float(np.mean(step_vals[s])) for s in steps_sorted], dtype=float),
            "subperiod": np.array([step_tp[s] for s in steps_sorted], dtype=int),
            "step_years": np.array([step_yr.get(s, -9999) for s in steps_sorted], dtype=int),
            "step_months": np.array([_gpdd_month(step_tp[s]) for s in steps_sorted], dtype=int),
            "step_dates_iso": None,
            "bin_days": 0,
        })

    audit_notes["gpdd_strict_insect_candidates_missed_by_original"] = strict_insect_extra
    audit_notes["gpdd_log_equal_untransformed_series"] = log_equal_series
    return out


def parse_biotime(raw_dir: Path, audit_notes: dict) -> list[dict]:
    bt_dir = raw_dir / "biotime"
    v2_realm: dict[str, str] = {}
    v2_file = bt_dir / "biotime_v2_metadata_15April25.csv"
    if v2_file.exists():
        with v2_file.open(encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                v2_realm[str(r.get("STUDY_ID", "")).strip()] = (r.get("REALM") or "").strip()

    meta: dict[str, dict] = {}
    with (bt_dir / "BioTIMEMetadata_24_06_2021.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            meta[r["STUDY_ID"].strip()] = r

    # Pass 1: distinct months per study-year
    ym_months: dict[str, dict[int, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    with zipfile.ZipFile(bt_dir / "BioTIMEQuery_24_06_2021.zip") as zf:
        with zf.open("BioTIMEQuery_24_06_2021.csv") as bf:
            for r in csv.DictReader(io.TextIOWrapper(bf, encoding="latin-1")):
                sid = r["STUDY_ID"].strip()
                if sid not in meta:
                    continue
                try:
                    yr = int(r["YEAR"])
                except ValueError:
                    continue
                ms = (r.get("MONTH") or "").strip()
                if ms and ms != "NA":
                    try:
                        mo = int(ms)
                        if 1 <= mo <= 12:
                            ym_months[sid][yr].add(mo)
                    except ValueError:
                        pass

    bin_mode: dict[str, tuple[str, float]] = {}
    studies_med_2_to_25 = []
    for sid in meta:
        ym = ym_months.get(sid, {})
        med_m = float(np.median([len(s) for s in ym.values()])) if len(ym) >= 3 else 1.0
        if 2.0 <= med_m < 2.5:
            studies_med_2_to_25.append((sid, med_m, meta[sid].get("TAXA", "")))
        if med_m >= 6.0:
            bin_mode[sid] = ("subannual_monthly", 12.0)
        elif med_m >= 2.5:
            bin_mode[sid] = ("subannual_seasonal", 4.0)
        else:
            bin_mode[sid] = ("annual", 1.0)
    audit_notes["biotime_studies_median_months_in_2_to_2.5"] = studies_med_2_to_25

    # Pass 2: effort per (study, bin) and per (study, plot, bin), plus species sums
    study_events: dict[str, dict[int, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    plot_events: dict[tuple[str, str], dict[int, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    sp_sums: dict[tuple[str, str], dict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    plot_sp_sums: dict[tuple[str, str, str], dict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    sp_tax: dict[tuple[str, str], tuple[str, str]] = {}
    unmatched_mammal_genera: set[str] = set()

    with zipfile.ZipFile(bt_dir / "BioTIMEQuery_24_06_2021.zip") as zf:
        with zf.open("BioTIMEQuery_24_06_2021.csv") as bf:
            for r in csv.DictReader(io.TextIOWrapper(bf, encoding="latin-1")):
                sid = r["STUDY_ID"].strip()
                st = meta.get(sid)
                if st is None:
                    continue
                iv_class, _ = bin_mode[sid]
                try:
                    yr = int(r["YEAR"])
                except ValueError:
                    continue
                if iv_class == "annual":
                    t_bin = yr
                else:
                    ms = (r.get("MONTH") or "").strip()
                    if not ms or ms == "NA":
                        continue
                    try:
                        mo = int(ms)
                    except ValueError:
                        continue
                    if not (1 <= mo <= 12):
                        continue
                    t_bin = (yr * 12 + (mo - 1)) if iv_class == "subannual_monthly" else (yr * 4 + ((mo - 1) // 3))

                sdesc = r.get("SAMPLE_DESC") or f"{yr}"
                study_events[sid][t_bin].add(sdesc)
                plt = (r.get("PLOT") or "").strip()
                has_plt = bool(plt and plt != "NA")
                if has_plt:
                    plot_events[(sid, plt)][t_bin].add(sdesc)

                ab_s = (r.get("sum.allrawdata.ABUNDANCE") or "").strip()
                if not ab_s or ab_s == "NA":
                    continue
                try:
                    ab = float(ab_s)
                except ValueError:
                    continue
                if ab <= 0.0:
                    continue

                sp_name = (r.get("GENUS_SPECIES") or r.get("ID_SPECIES") or "").strip()
                skey = (sid, sp_name)
                if skey not in sp_tax:
                    grp, sub = classify_biotime(st, v2_realm.get(sid, ""), r.get("GENUS", ""), sp_name)
                    if grp is None and (st.get("TAXA") or "").strip() == "Mammals":
                        gtok = (r.get("GENUS") or (sp_name.split()[0] if sp_name else "")).strip().lower()
                        if gtok not in LARGE_OR_EXCLUDED_MAMMAL_GENERA and (v2_realm.get(sid) or st.get("REALM")) != "Marine":
                            unmatched_mammal_genera.add(gtok)
                    sp_tax[skey] = (grp or "", sub or "")
                if not sp_tax[skey][0]:
                    continue

                sp_sums[skey][t_bin] += ab
                if has_plt:
                    plot_sp_sums[(sid, plt, sp_name)][t_bin] += ab

    audit_notes["biotime_unmatched_nonmarine_mammal_genera"] = sorted(unmatched_mammal_genera)

    out: list[dict] = []
    # Pooled primary BioTIME series
    for (sid, sp_name), bins_ab in sp_sums.items():
        grp, sub = sp_tax[(sid, sp_name)]
        if not grp:
            continue
        evs = study_events[sid]
        if len(evs) < 15:
            continue
        all_b = sorted(evs.keys())
        vals = np.array([bins_ab.get(b, 0.0) / float(len(evs[b])) for b in all_b], dtype=float)
        if int(np.sum(vals > 0.0)) < 15:
            continue
        st = meta[sid]
        lic = (st.get("LICENSE") or "CC-BY").strip()
        iv_class, spy = bin_mode[sid]
        treat = ((st.get("GENERAL_TREAT") or "") + " " + (st.get("TREATMENT") or "")).strip().lower()
        manip = bool(treat and treat not in ("na", "none", "na na", "control", ""))
        try:
            lat = float(st.get("CENT_LAT") or "nan")
            lon = float(st.get("CENT_LONG") or "nan")
        except ValueError:
            lat, lon = float("nan"), float("nan")
        r_eff = v2_realm.get(sid) or (st.get("REALM") or "").strip()
        r_det = r_eff if r_eff in ("Marine", "Freshwater", "Terrestrial") else (
            "Marine" if "marine" in (st.get("HABITAT") or "").lower() else "Terrestrial"
        )
        txt = " ".join(st.get(k, "") for k in ("HABITAT", "METHODS", "COMMENTS")).lower()
        is_lab = any(w in txt for w in ("laboratory microcosm", "laboratory culture", "indoor aquarium"))

        sub_p = np.array(
            [b % 12 if iv_class == "subannual_monthly" else (b % 4 if iv_class == "subannual_seasonal" else 0) for b in all_b],
            dtype=int,
        )
        yrs = np.array(
            [b // 12 if iv_class == "subannual_monthly" else (b // 4 if iv_class == "subannual_seasonal" else b) for b in all_b],
            dtype=int,
        )
        mos = np.array(
            [(b % 12) + 1 if iv_class == "subannual_monthly" else ((b % 4) * 3 + 2 if iv_class == "subannual_seasonal" else 0) for b in all_b],
            dtype=int,
        )
        out.append({
            "series_id": f"BT_{sid}_{sp_name.replace(' ', '_')}",
            "database": "BioTIME",
            "study_id": f"BT_STUDY_{sid}",
            "licence": lic,
            "cc_by_compatible": ("nc" not in lic.lower()),
            "taxon_group": grp,
            "taxon_subgroup": sub,
            "species": sp_name,
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm_detail": r_det,
            "environment_type": "laboratory" if is_lab else "field",
            "reliability": 4,
            "manipulated": manip,
            "series_kind": "primary",
            "units": (st.get("ABUNDANCE_TYPE", "unspecified") + " per sampled event"),
            "steps": np.array(all_b, dtype=int),
            "values": vals,
            "subperiod": sub_p,
            "step_years": yrs,
            "step_months": mos,
            "step_dates_iso": None,
            "bin_days": 0,
        })

    # Fixed-plot BioTIME series for Sensitivity S6
    for (sid, plt, sp_name), bins_ab in plot_sp_sums.items():
        grp, sub = sp_tax.get((sid, sp_name), ("", ""))
        if not grp:
            continue
        pevs = plot_events[(sid, plt)]
        if len(pevs) < 15:
            continue
        all_b = sorted(pevs.keys())
        vals = np.array([bins_ab.get(b, 0.0) / float(len(pevs[b])) for b in all_b], dtype=float)
        if int(np.sum(vals > 0.0)) < 15:
            continue
        st = meta[sid]
        lic = (st.get("LICENSE") or "CC-BY").strip()
        iv_class, spy = bin_mode[sid]
        treat = ((st.get("GENERAL_TREAT") or "") + " " + (st.get("TREATMENT") or "")).strip().lower()
        manip = bool(treat and treat not in ("na", "none", "na na", "control", ""))
        try:
            lat = float(st.get("CENT_LAT") or "nan")
            lon = float(st.get("CENT_LONG") or "nan")
        except ValueError:
            lat, lon = float("nan"), float("nan")
        r_eff = v2_realm.get(sid) or (st.get("REALM") or "").strip()
        r_det = r_eff if r_eff in ("Marine", "Freshwater", "Terrestrial") else "Terrestrial"
        spy_int = int(spy)
        steps_arr = np.array(all_b, dtype=int)
        out.append({
            "series_id": f"BTPLOT_{sid}_{plt}_{sp_name.replace(' ', '_')}",
            "database": "BioTIME",
            "study_id": f"BT_STUDY_{sid}",
            "licence": lic,
            "cc_by_compatible": ("nc" not in lic.lower()),
            "taxon_group": grp,
            "taxon_subgroup": sub,
            "species": sp_name,
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm_detail": r_det,
            "environment_type": "field",
            "reliability": 4,
            "manipulated": manip,
            "series_kind": "biotime_fixed_plot",
            "units": (st.get("ABUNDANCE_TYPE", "unspecified") + " per sampled event"),
            "steps": steps_arr,
            "values": vals,
            "subperiod": steps_arr % spy_int,
            "step_years": steps_arr // spy_int,
            "step_months": np.array([(int(b) % spy_int) * (12 // spy_int) + 1 if spy_int > 1 else 0 for b in steps_arr], dtype=int),
            "step_dates_iso": None,
            "bin_days": 0,
        })

    return out


def parse_vecdyn(raw_dir: Path) -> list[dict]:
    vd_dir = raw_dir / "vecdyn" / "full_datasets"
    epoch = dt.date(1970, 1, 5)
    out: list[dict] = []

    for jf in sorted(vd_dir.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        cons = d.get("consistent_data") or {}
        did = d["dataset_id"]
        if "s41597-024-03482-y" in str(cons.get("curatedbydoi")) or "temporally downscaled" in str(cons.get("additional_sample_info")).lower():
            continue
        lic = str(cons.get("dataset_license") or d.get("dataset_license") or "CC-BY-NC-4.0")

        events: dict[tuple, dict[dt.date, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
        abund: dict[tuple, dict[dt.date, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
        tax_map: dict[str, tuple[str, str]] = {}
        coords_map: dict[tuple, tuple[float, float]] = {}
        unit_map: dict[tuple, str] = {}

        for r in d.get("results") or []:
            def fget(k: str, default: str = ""):
                v = r.get(k)
                return v if (v is not None and v != "") else cons.get(k, default)

            unit = str(fget("sample_unit")).lower()
            if any(u in unit for u in ("presence", "binary", "percent")):
                continue
            try:
                obs_d = dt.date.fromisoformat(str(fget("sample_end_date"))[:10])
                val = float(fget("sample_value"))
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val) or val < 0.0:
                continue

            try:
                lat = float(fget("sample_lat_dd"))
            except (ValueError, TypeError):
                lat = float("nan")
            try:
                lon = float(fget("sample_long_dd"))
            except (ValueError, TypeError):
                lon = float("nan")

            place = str(fget("sample_location") or fget("location_description") or "unspecified")
            loc_key = (round(lat, 3) if math.isfinite(lat) else None, round(lon, 3) if math.isfinite(lon) else None, place)
            skey = (loc_key, str(fget("sampling_method") or "unspecified"), str(fget("sample_stage") or "unspecified"))
            coords_map[skey] = (lat, lon)
            unit_map[skey] = unit

            ev_name = str(fget("sample_name") or f"{place}_{obs_d}")
            events[skey][obs_d].add(ev_name)

            gen = str(fget("genus"))
            sp = str(fget("species"))
            fam = str(fget("family")).lower()
            if sp.upper() in ("EMPTY", "BLANK", "NONE"):
                continue
            sp_full = (f"{gen} {sp}").strip() if (gen and not sp.lower().startswith(gen.lower())) else (sp or gen)
            tok0 = sp_full.split()[0].lower() if sp_full else ""
            if fam in TICK_FAMILIES or tok0 in TICK_GENERA:
                grp, sub = "arachnid_vectors_ticks", "ticks"
            elif fam in VECTOR_FAMILIES or tok0 in VECTOR_GENERA:
                grp, sub = "insects_vectors", "mosquitoes_and_biting_diptera"
            else:
                continue
            tax_map[sp_full] = (grp, sub)
            abund[(skey, sp_full)][obs_d] += val

        for skey, ev_by_date in events.items():
            dates = sorted(ev_by_date.keys())
            if len(dates) < 15:
                continue
            med_gap = float(np.median(np.diff([(dd - epoch).days for dd in dates])))
            if 4.0 <= med_gap <= 10.0:
                iv_class, spy, bdays = "subannual_biweekly_weekly", 52, 7
            elif 11.0 <= med_gap <= 20.0:
                iv_class, spy, bdays = "subannual_biweekly_weekly", 26, 14
            elif 21.0 <= med_gap <= 45.0:
                iv_class, spy, bdays = "subannual_monthly", 12, 0
            elif 60.0 <= med_gap <= 130.0:
                iv_class, spy, bdays = "subannual_seasonal", 4, 0
            elif 300.0 <= med_gap <= 400.0:
                iv_class, spy, bdays = "annual", 1, 0
            else:
                continue

            def to_bin(dd: dt.date) -> int:
                if bdays:
                    return (dd - epoch).days // bdays
                return dd.year * spy + (((dd.month - 1) // (12 // spy)) if spy > 1 else 0)

            eff_bin: dict[int, int] = collections.Counter()
            for dd, ev_set in ev_by_date.items():
                eff_bin[to_bin(dd)] += len(ev_set)
            bins = sorted(eff_bin.keys())
            lat, lon = coords_map[skey]

            for (skey2, sp_full), sp_dates in abund.items():
                if skey2 != skey:
                    continue
                sum_bin: dict[int, float] = collections.Counter()
                for dd, v in sp_dates.items():
                    sum_bin[to_bin(dd)] += v
                vals = np.array([sum_bin[b] / float(eff_bin[b]) for b in bins], dtype=float)
                if int(np.sum(vals > 0.0)) < 15:
                    continue
                h_tag = hashlib.sha256(repr((skey, sp_full)).encode()).hexdigest()[:12]
                iso_dates = [(epoch + dt.timedelta(days=b * bdays)).isoformat() for b in bins] if bdays else None
                out.append({
                    "series_id": f"VD_{did}_{h_tag}",
                    "database": "VecDyn",
                    "study_id": f"VD_SET_{did}",
                    "licence": lic,
                    "cc_by_compatible": ("nc" not in lic.lower()),
                    "taxon_group": tax_map[sp_full][0],
                    "taxon_subgroup": tax_map[sp_full][1],
                    "species": sp_full,
                    "interval_class": iv_class,
                    "samples_per_year": float(spy),
                    "latitude": lat,
                    "longitude": lon,
                    "realm_detail": "Terrestrial",
                    "environment_type": "field",
                    "reliability": 4,
                    "manipulated": False,
                    "series_kind": "primary",
                    "units": unit_map[skey],
                    "steps": np.array(bins, dtype=int),
                    "values": vals,
                    "subperiod": np.array([b % spy for b in bins], dtype=int),
                    "step_years": np.array([dt.date.fromisoformat(d0).year for d0 in iso_dates], dtype=int) if iso_dates else np.array([b // spy for b in bins], dtype=int),
                    "step_months": np.array([dt.date.fromisoformat(d0).month for d0 in iso_dates], dtype=int) if iso_dates else np.array([(b % spy) * (12 // spy) + 1 if spy > 1 else 0 for b in bins], dtype=int),
                    "step_dates_iso": iso_dates,
                    "bin_days": bdays,
                })

    return out


def check_primary_eligibility(s: dict, coverage_min: float = 0.75) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float] | None, str | None]:
    """Apply registered primary regularity and zero-handling rules (protocol.md §2.3)."""
    vals = s["values"]
    steps = s["steps"]
    pos_mask = np.isfinite(vals) & (vals > 0.0)
    if int(np.sum(pos_mask)) < 15:
        return None, "fewer_than_15_positive"
    if int(np.sum(pos_mask)) < 0.5 * len(vals):
        return None, "positive_fraction_below_50pct"
    t_pos = steps[pos_mask].astype(float)
    v_pos = vals[pos_mask].astype(float)
    span = float(t_pos[-1] - t_pos[0] + 1.0)
    cov = float(len(t_pos)) / span
    if cov < coverage_min:
        return None, "grid_coverage_below_threshold"
    if int(np.sum(np.diff(t_pos) == 1.0)) < 12:
        return None, "fewer_than_12_adjacent_pairs"
    y_log = np.log(v_pos)
    if float(np.var(y_log)) <= 1e-10:
        return None, "zero_log_variance"
    return (t_pos, y_log, pos_mask, cov), None


def load_all_compilations(raw_dir: Path = RAW_LPD) -> tuple[list[dict], dict]:
    audit_notes: dict = {}
    gpdd_s = parse_gpdd(raw_dir, audit_notes)
    bt_s = parse_biotime(raw_dir, audit_notes)
    vd_s = parse_vecdyn(raw_dir)
    all_s = gpdd_s + bt_s + vd_s
    return all_s, audit_notes


if __name__ == "__main__":
    series, notes = load_all_compilations()
    prim = []
    reasons = collections.Counter()
    for s in series:
        if s["series_kind"] != "primary":
            continue
        el, why = check_primary_eligibility(s)
        if el is not None:
            prim.append(s)
        else:
            reasons[why] += 1
    by_db = collections.Counter(s["database"] for s in prim)
    by_tax = collections.Counter(s["taxon_group"] for s in prim)
    clusters = len({s["study_id"] for s in prim})
    print(f"Primary retained: M={len(prim)}, K={clusters}")
    print("  By database:", dict(by_db))
    print("  By taxon:", dict(by_tax))
    print("  Exclusions:", dict(reasons))
    print("  Audit notes summary:", {k: len(v) for k, v in notes.items()})
    for k, v in notes.items():
        if v:
            print(f"    {k} (first 10):", v[:10])
