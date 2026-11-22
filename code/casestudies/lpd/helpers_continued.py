"""Taxonomic classification, compilation loaders, and time-series / theory helpers
for case study lpd. Imported by analysis.py and make_figures.py; run with python3 -I.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

# ---------------------------------------------------------------------------
# 1. Deterministic Taxonomic Classification Tables (fixed per protocol.md §2.2)
# ---------------------------------------------------------------------------

SMALL_MAMMAL_ORDERS = {
    "rodentia", "lagomorpha", "insectivora", "soricomorpha", "erinaceomorpha",
    "eulipotyphla", "didelphimorphia", "scandentia", "macroscelidea",
    "dasyuromorphia", "afrosoricida", "microbiotheria", "paucituberculata",
}

SMALL_MAMMAL_GENERA = {
    # Rodents, lagomorphs, shrews, moles, opossums, marsupial mice present in BioTIME/GPDD
    "abrocoma", "abrothrix", "acomys", "aethomys", "akodon", "alticola", "altilcola",
    "ammospermophilus", "apodemus", "arvicanthis", "arvicanthus", "arvicola",
    "baiomys", "bandicota", "blarina", "bolomys", "callosciurus", "calomys",
    "cavia", "chaetodipus", "chelemys", "chetodipus", "chinchilla", "chionomys",
    "clethrionomys", "cricetulus", "cricetus", "crocidura", "cryptotis", "ctenomys",
    "cyclopes", "cynomys", "dasyprocta", "dasyurus", "dendromus", "didelphis",
    "dipodomys", "dryomys", "eliomys", "elephantulus", "erethizon", "erinaceus",
    "eutamias", "funisciurus", "galea", "geomys", "gerbilliscus", "gerbillurus",
    "gerbillus", "glaucomys", "glis", "grammomys", "graphiurus", "heliosciurus",
    "heteromys", "hydrochoerus", "hylomyscus", "lemmiscomys", "lemmus", "lepus",
    "liomys", "lophuromys", "marmosa", "marmota", "mastomys", "meriones",
    "mesocricetus", "microcavia", "micromys", "microsciurus", "microtus",
    "monodelphis", "mus", "muscardinus", "myocastor", "myodes", "myoprocta",
    "myopus", "mysateles", "napaeozapus", "nectomys", "neomys", "neotoma",
    "neotomodon", "neozapus", "nesokia", "niviventer", "notomys", "ochotona",
    "ochrotomys", "octodon", "oligoryzomys", "ondatra", "onychomys", "oryzomys",
    "otocyon_exclude", "otomys", "oxymycterus", "paraxerus", "perameles",
    "perognathus", "peromyscus", "petaurista", "phenacomys", "phyllotis",
    "pitymys", "podomys", "praomys", "proechimys", "pseudomys", "rattus",
    "reithrodontomys", "rhabdomys", "rhombomys", "saccostomus", "scapanus",
    "sciurus", "scotinomys", "sicista", "sigmodon", "sorex", "spalacopus",
    "spalax", "spermophilus", "steatomys", "suncus", "sylvilagus", "synaptomys",
    "talpa", "tamias", "tamiasciurus", "tatera", "taterillus", "thomomys",
    "thylamys", "tylomys", "urocitellus", "utinam_exclude", "zapus", "zelatomys",
    "zygodontomys",
}

NON_MAMMAL_INTRUDERS_IN_BT_MAMMAL_STUDIES = {
    # Non-mammal genera occasionally recorded inside multi-taxa desert plots (e.g. BioTIME study 59, 382)
    "amphispiza", "campylorhynchus", "crotalus", "haliotis", "pipilio", "pipilo",
    "pooecetes", "sceloporus",
}

WATERFOWL_GULL_ORDERS = {
    "anseriformes", "charadriiformes", "pelecaniformes", "suliformes",
    "ciconiiformes", "gaviiformes", "podicipediformes",
}

WATERFOWL_GULL_GENERA = {
    # Anseriformes (ducks, geese, swans)
    "aix", "alopochen", "amazonetta", "anas", "anser", "aythya", "biziura",
    "branta", "bucephala", "cairina", "callonetta", "camptorhynchus", "cereopsis",
    "chen", "chenonetta", "chloephaga", "clangula", "coscoroba", "cyanochen",
    "cygnus", "dendrocygna", "heteronetta", "histrionicus", "hymenolaimus",
    "lophodytes", "lophoetta", "malacorhynchus", "mareca", "marmaronetta",
    "melanitta", "merganetta", "mergellus", "mergus", "netta", "nettapus",
    "nomonyx", "oxyura", "polysticta", "pteronetta", "rhodonessa", "salvadorina",
    "sarkidiornis", "somateria", "spatula", "stictonetta", "tachyeres", "tadorna",
    "thalassornis",
    # Charadriiformes (gulls, terns, skuas, alcids, waders/shorebirds)
    "actitis", "alle", "arenaria", "bartramia", "brachyramphus", "burhinus",
    "calidris", "catharacta", "catoptrophorus", "cepphus", "cerorhinca",
    "charadrius", "chlidonias", "chroicocephalus", "cladorhynchus", "creagrus",
    "cursorius", "dromas", "eudromias", "fratercula", "gallinago", "gelochelidon",
    "glareola", "haematopus", "himantopus", "hydrocoloeus", "hydrophasianus",
    "hydroprogne", "ibidorhyncha", "ichthyaetus", "jacana", "larosterna", "larus",
    "leucophaeus", "limicola", "limnodromus", "limosa", "lunda", "lymnocryptes",
    "micropalama", "numenius", "nychophus", "onychoprion", "pagophila", "phaetusa",
    "phalaropus", "philomachus", "pluvialis", "pluvianellus", "pluvianus",
    "procelsterna", "ptycoramphus", "ptychoramphus", "recurvirostra", "rhodostethia",
    "rynchops", "rosthramus_exclude", "rostratula", "rychops", "scolopax",
    "stercorarius", "sterna", "sternula", "synthliboramphus", "thalasseus",
    "thinocorus", "tringa", "tryngites", "uria", "vanellus", "xema", "xenus",
    # Gaviiformes, Podicipediformes, Pelecaniformes, Suliformes, Ciconiiformes (aquatic fish/snail-eating birds)
    "achmophorus", "aechmophorus", "anhinga", "ardea", "ardeola", "botaurus",
    "bubulcus", "butorides", "casmerodius", "ciconia", "cochlearius", "egretta",
    "eudocimus", "fregata", "gavia", "ixobrychus", "jabiru", "leptoptilos",
    "microcarbo", "morus", "mycteria", "nyctanassa", "nycticorax", "pelecanus",
    "phalacrocorax", "platalea", "plegadis", "podiceps", "podilymbus", "poliocephalus",
    "rollandia", "sula", "tachybaptus", "tigrisoma", "urile",
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

ALGAL_PLANT_GENERA_EXCLUDE = {
    "anabaena", "aphanizomenon", "aphanocapsa", "aphanothece", "asterionella",
    "aulacoseira", "belonastrum", "ceratium", "chlorella", "chroomonas",
    "closterium", "coelastrum", "cosmarium", "cryptomonas", "cyclotella",
    "cymbella", "dinobryon", "eudorina", "fragilaria", "gomphonema", "gymnodinium",
    "mallomonas", "melosira", "merismopedia", "microcystis", "mougeotia",
    "navicula", "nitzschia", "ochromonas", "oscillatoria", "pandorina",
    "pediastrum", "peridinium", "planktosphaeria", "rhodomonas", "scenedesmus",
    " Schroederia".strip().lower(), "staurastrum", "stephanodiscus", "synedra",
    "synura", "tabellaria", "ulothrix", "volvox", "woronichinia",
}

VECTOR_INSECT_FAMILIES = {
    "culicidae", "simuliidae", "ceratopogonidae", "psychodidae",
    "tabanidae", "glossinidae", "pulicidae", "ceratophyllidae",
    "leptopsyllidae", "vermipysllidae", "reduviidae", "pediculidae",
}

VECTOR_INSECT_GENERA = {
    "aedes", "aedimorphus", "anopheles", "armigeres", "bironella", "chagasia",
    "chrysops", "coquillettidia", "culex", "culicoides", "culiseta", "deinoceriites",
    "deinocerites", "eretmapodites", "ficalbia", "glossina", "haemagogus",
    "haematobia", "haematopota", "heizmannia", "hodgesia", "limatus", "lutzia",
    "lutzomyia", "malaya", "mansonia", "mimomyia", "ochlerotatus", "orthopodomyia",
    "panstrongylus", "phlebotomus", "psorophora", "pulex", "rhodnius", "sabethes",
    "simulium", "stomoxys", "tabanus", "toxorhynchites", "triatoma", "trichoprosopon",
    "tripteroides", "uranotaenia", "verrallina", "wyeomyia", "xenopsylla",
    "melanoconion",
}

TICK_FAMILIES = {"ixodidae", "argasidae", "nuttalliellidae"}
TICK_GENERA = {
    "amblyomma", "argas", "boophilus", "dermacentor", "haemaphysalis",
    "hyalomma", "ixodes", "ornithodoros", "otobius", "rhipicephalus",
}

FRESHWATER_BIOTOPE_KEYWORDS = (
    "limnetic", "fluvial", "wetland", "lake", "pond", "river", "stream", "freshwater", "marsh", "bog"
)


def classify_gpdd_taxon(taxon_row: dict, biotope_row: dict) -> tuple[str | None, str | None]:
    """Return (taxon_group, taxon_subgroup) or (None, None) if outside target host groups."""
    cls = (taxon_row.get("TaxonomicClass") or "").strip().lower()
    ord_ = (taxon_row.get("TaxonomicOrder") or "").strip().lower()
    fam = (taxon_row.get("TaxonomicFamily") or "").strip().lower()
    gen = (taxon_row.get("TaxonomicGenus") or "").strip().lower()
    name = (taxon_row.get("TaxonName") or "").strip().lower()
    first_tok = name.split()[0] if name else ""
    g_eff = gen if gen and gen != "unknown" else first_tok
    hab = ((biotope_row.get("HabitatName") or "") + " " + (biotope_row.get("BiotopeType") or "")).lower()
    is_marine = ("marine" in hab) or ("tidal" in hab) or ("coastal" in hab and "limnetic" not in hab and "wetland" not in hab)
    is_fw = any(k in hab for k in FRESHWATER_BIOTOPE_KEYWORDS)

    if cls == "mammalia":
        if ord_ in SMALL_MAMMAL_ORDERS or g_eff in SMALL_MAMMAL_GENERA:
            return ("small_mammals", "rodents" if ord_ == "rodentia" else "other_small_mammals")
        return (None, None)

    if cls == "aves":
        if ord_ in WATERFOWL_GULL_ORDERS or g_eff in WATERFOWL_GULL_GENERA:
            return ("birds", "birds_waterfowl_gulls")
        return ("birds", "birds_other")

    if cls in FISH_CLASSES:
        return ("fish", "freshwater_fish" if is_fw and not is_marine else "marine_or_diadromous_fish")

    if cls == "insecta":
        if fam in VECTOR_INSECT_FAMILIES or g_eff in VECTOR_INSECT_GENERA:
            return ("insects_vectors", fam or "diptera_vector")
        if is_fw and not is_marine and ord_ in ("odonata", "ephemeroptera", "plecoptera", "trichoptera"):
            return ("freshwater_invertebrates", "aquatic_insects_nonvector")
        return (None, None)

    if cls == "arachnida":
        if fam in TICK_FAMILIES or g_eff in TICK_GENERA:
            return ("arachnid_vectors_ticks", "ticks")
        return (None, None)

    if cls in ("gastropoda", "bivalvia", "crustacea", "rotifera", "oligochaeta", "hirudinea"):
        if is_marine:
            return (None, None)
        if g_eff in SNAIL_GENERA or (cls == "gastropoda" and is_fw):
            return ("freshwater_invertebrates", "snails_only")
        if is_fw or cls == "rotifera" or fam in ("cambaridae", "daphniidae", "bosminidae", "chydoridae", "sphaeriidae"):
            return ("freshwater_invertebrates", "other_freshwater_invertebrates")
    return (None, None)


def classify_biotime_record(study_row: dict, v2_realm: str | None, genus: str, species_full: str) -> tuple[str | None, str | None]:
    """Classify a BioTIME study+species into one of the five target host groups."""
    taxa = (study_row.get("TAXA") or "").strip()
    orgs = (study_row.get("ORGANISMS") or "").strip().lower()
    hab = (study_row.get("HABITAT") or "").strip().lower()
    realm = (v2_realm or study_row.get("REALM") or "").strip()
    g = (genus or "").strip().lower()
    if not g and species_full:
        g = species_full.strip().split()[0].lower()

    if g in ALGAL_PLANT_GENERA_EXCLUDE or "phytoplankton" in orgs or "plants" in taxa.lower():
        return (None, None)

    # Vector insects first (even if inside a freshwater/terrestrial invertebrate study)
    if g in VECTOR_INSECT_GENERA:
        return ("insects_vectors", "mosquitoes_and_biting_diptera")
    if g in TICK_GENERA:
        return ("arachnid_vectors_ticks", "ticks")

    if taxa == "Mammals":
        if realm == "Marine" or g in NON_MAMMAL_INTRUDERS_IN_BT_MAMMAL_STUDIES:
            return (None, None)
        if g in SMALL_MAMMAL_GENERA:
            return ("small_mammals", "small_mammals")
        return (None, None)

    if taxa == "Birds" or (taxa == "All" and "seabird" in orgs and g not in ("balaenoptera", "delphinus", "phocoena", "arctocephalus", "caretta", "chelonia", "dermochelys", "stenella", "tursiops", "zalophus")):
        if g in WATERFOWL_GULL_GENERA or "seabird" in orgs or "waterfowl" in orgs or "waterbird" in orgs:
            return ("birds", "birds_waterfowl_gulls")
        return ("birds", "birds_other")

    if taxa == "Fish":
        return ("fish", "freshwater_fish" if realm == "Freshwater" else "marine_or_estuarine_fish")

    # Snails in freshwater or brackish/estuarine mudflats (e.g. Everglades study 231, Baltic/estuarine Hydrobia)
    if g in SNAIL_GENERA and (realm == "Freshwater" or "estuar" in hab or "lake" in hab or "river" in hab or "wetland" in hab):
        return ("freshwater_invertebrates", "snails_only")

    if realm == "Freshwater" and taxa in ("Freshwater invertebrates", "Benthos"):
        return ("freshwater_invertebrates", "other_freshwater_invertebrates")

    return (None, None)


# ---------------------------------------------------------------------------
# 2. Core Estimators & Bootstrap Routines (identical to preanalysis_power.py)
# ---------------------------------------------------------------------------

_PHI_GRID = np.linspace(-0.95, 0.98, 194)
_WN_SPEC_BANK = np.random.default_rng(2026111509).standard_normal((200, 512))


def ols_detrend(t_idx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    t_mean = float(t_idx.mean())
    y_mean = float(y.mean())
    dt_vec = t_idx - t_mean
    dy_vec = y - y_mean
    denom = float(dt_vec @ dt_vec)
    beta = float(dt_vec @ dy_vec) / denom if denom > 0.0 else 0.0
    alpha = y_mean - beta * t_mean
    return y - (alpha + beta * t_idx), alpha, beta


def white_noise_r1_null_mean(t_idx: np.ndarray) -> float:
    n = len(t_idx)
    if n <= 3:
        return 0.0
    t_c = t_idx - t_idx.mean()
    denom_t = float(t_c @ t_c)
    adj = np.flatnonzero(np.diff(t_idx) == 1.0)
    m1 = len(adj)
    if m1 == 0:
        return 0.0
    sum_h = (m1 / float(n)) + (float(t_c[adj] @ t_c[adj + 1]) / denom_t if denom_t > 0.0 else 0.0)
    return -float(n) / float((n - 2) * m1) * sum_h


def white_noise_spec_null_mean(t_idx: np.ndarray) -> float:
    """Finite-sample linear-detrending null expectation of b_spec on grid t_idx."""
    n = len(t_idx)
    span = int(t_idx[-1] - t_idx[0]) + 1
    k_max = (span - 1) // 2
    if k_max < 4:
        return 0.0
    if n <= _WN_SPEC_BANK.shape[1]:
        W = _WN_SPEC_BANK[:, :n]
    else:
        W = np.random.default_rng(2026111510 + n).standard_normal((200, n))
    t_c = t_idx - t_idx.mean()
    denom_t = float(t_c @ t_c)
    W0 = W - W.mean(axis=1, keepdims=True)
    E = W0 - ((W0 @ t_c)[:, None] / denom_t) * t_c[None, :] if denom_t > 0.0 else W0
    grid = np.zeros((W.shape[0], span), dtype=float)
    grid[:, (t_idx - t_idx[0]).astype(int)] = E
    fft_vals = np.fft.rfft(grid, axis=1)[:, 1 : k_max + 1]
    pw = (np.abs(fft_vals) ** 2) / float(n)
    lx = np.log(np.arange(1, k_max + 1, dtype=float) / float(span))
    lx_c = lx - lx.mean()
    denom_x = float(lx_c @ lx_c)
    ly = np.log(np.maximum(pw, 1e-30))
    ly_c = ly - ly.mean(axis=1, keepdims=True)
    slopes = (ly_c @ lx_c) / denom_x
    return float(np.mean(slopes))


def orthonormal_basis_from_design(X: np.ndarray) -> np.ndarray:
    """Return an (n, p_eff) orthonormal column basis for the column space of X via SVD."""
    u_svd, s_vals, _ = np.linalg.svd(X, full_matrices=False)
    if len(s_vals) == 0 or s_vals[0] <= 1e-14:
        return np.zeros((X.shape[0], 0), dtype=float)
    keep = s_vals > (s_vals[0] * 1e-11)
    return u_svd[:, keep]


def projected_white_noise_r1_null_mean(t_idx: np.ndarray, Q_x: np.ndarray) -> float:
    """Exact first-order ratio-of-expectations white-noise lag-1 residual correlation
    after projecting out the orthonormal column basis Q_x of shape (n, p).
    """
    n = len(t_idx)
    p_rank = Q_x.shape[1]
    if n <= p_rank + 1:
        return 0.0
    adj = np.flatnonzero(np.diff(t_idx) == 1.0)
    m1 = len(adj)
    if m1 == 0:
        return 0.0
    sum_h = float(np.sum(Q_x[adj] * Q_x[adj + 1])) if p_rank > 0 else 0.0
    return -float(n) / float((n - p_rank) * m1) * sum_h


def ar1_projected_expected_r1_curve(
    t_idx: np.ndarray, Q_x: np.ndarray, phis: np.ndarray = _PHI_GRID
) -> np.ndarray:
    """Exact O(n * p) forward-backward evaluation of E_phi[num] / E_phi[den] for OLS
    residuals projected orthogonally to orthonormal basis Q_x (shape n x p).
    """
    n = len(t_idx)
    p_len = len(phis)
    p_rank = Q_x.shape[1]
    gaps = np.diff(t_idx)
    adj = np.flatnonzero(gaps == 1.0)
    if len(adj) == 0:
        return np.zeros(p_len, dtype=float)
    if p_rank == 0:
        return phis.copy()

    pow_gaps = phis[:, None] ** gaps[None, :]
    W = np.empty((p_len, n, p_rank), dtype=float)
    f = np.broadcast_to(Q_x[0], (p_len, p_rank)).copy()
    W[:, 0, :] = f
    for k in range(1, n):
        f = pow_gaps[:, k - 1 : k] * f + Q_x[k]
        W[:, k, :] = f
    b = np.zeros((p_len, p_rank), dtype=float)
    for k in range(n - 2, -1, -1):
        b = pow_gaps[:, k : k + 1] * (b + Q_x[k + 1])
        W[:, k, :] += b

    UtW = np.einsum("ka,pkb->pab", Q_x, W)
    tr_utw = np.einsum("paa->p", UtW)
    e_den = (float(n) - tr_utw) / float(n)

    ui = Q_x[adj]
    uj = Q_x[adj + 1]
    term_uw = np.einsum("ma,pma->pm", ui, W[:, adj + 1, :])
    term_wu = np.einsum("pma,ma->pm", W[:, adj, :], uj)
    term_quad = np.einsum("ma,pab,mb->pm", ui, UtW, uj)
    cov_ij = phis[:, None] - term_uw - term_wu + term_quad
    e_num = cov_ij.mean(axis=1)
    return e_num / np.maximum(e_den, 1e-15)


def invert_projected_bias_corrected_r1(t_idx: np.ndarray, Q_x: np.ndarray, r1_hat: float) -> float:
    curve = ar1_projected_expected_r1_curve(t_idx, Q_x, _PHI_GRID)
    return float(np.interp(r1_hat, curve, _PHI_GRID))


def ar1_detrended_expected_r1_curve(t_idx: np.ndarray, phis: np.ndarray = _PHI_GRID) -> np.ndarray:
    n = len(t_idx)
    t_c = t_idx - t_idx.mean()
    norm_t = math.sqrt(float(t_c @ t_c))
    u0 = np.full(n, 1.0 / math.sqrt(n))
    U = np.column_stack([u0, t_c / norm_t]) if norm_t > 0.0 else u0[:, None]
    return ar1_projected_expected_r1_curve(t_idx, U, phis)


def invert_bias_corrected_r1(t_idx: np.ndarray, r1_hat: float) -> float:
    curve = ar1_detrended_expected_r1_curve(t_idx, _PHI_GRID)
    return float(np.interp(r1_hat, curve, _PHI_GRID))


def compute_core_metrics(
    t_idx: np.ndarray,
    y: np.ndarray,
    detrend: bool = True,
    max_lag: int = 10,
) -> dict:
    n = len(t_idx)
    if detrend:
        resid, _, trend_slope = ols_detrend(t_idx, y)
        sigma2 = float((resid @ resid) / (n - 2))
    else:
        resid = y - float(y.mean())
        trend_slope = 0.0
        sigma2 = float((resid @ resid) / (n - 1))
    sigma2_raw = float(np.var(y, ddof=1))
    c0 = float((resid @ resid) / n)

    pos = {int(t): i for i, t in enumerate(t_idx)}
    lmax = min(max_lag, n // 3)
    acf = {}
    for lag in range(1, max_lag + 1):
        if lag > lmax:
            acf[lag] = None
            continue
        pairs = [(i, pos[int(t) + lag]) for i, t in enumerate(t_idx) if (int(t) + lag) in pos]
        if len(pairs) >= 6 and c0 > 1e-15:
            ia = np.array([p[0] for p in pairs], dtype=int)
            ib = np.array([p[1] for p in pairs], dtype=int)
            acf[lag] = float((resid[ia] @ resid[ib]) / (len(pairs) * c0))
        else:
            acf[lag] = None

    r1 = acf[1]
    span = int(t_idx[-1] - t_idx[0]) + 1
    grid = np.zeros(span, dtype=float)
    grid[(t_idx - t_idx[0]).astype(int)] = resid
    fft_vals = np.fft.rfft(grid)
    k_max = (span - 1) // 2
    if k_max >= 4:
        freqs = np.arange(1, k_max + 1, dtype=float) / float(span)
        pw = (np.abs(fft_vals[1 : k_max + 1]) ** 2) / float(n)
        valid = pw > 1e-30
        if int(valid.sum()) >= 4:
            lx = np.log(freqs[valid])
            ly = np.log(pw[valid])
            lx_c = lx - lx.mean()
            b_spec = float((lx_c @ (ly - ly.mean())) / (lx_c @ lx_c))
        else:
            b_spec = float("nan")
    else:
        b_spec = float("nan")

    return {
        "n": n,
        "span": span,
        "sigma2": sigma2,
        "sigma": math.sqrt(max(sigma2, 0.0)),
        "sigma2_raw": sigma2_raw,
        "sigma_raw": math.sqrt(max(sigma2_raw, 0.0)),
        "trend_slope": trend_slope,
        "r1": r1,
        "acf": acf,
        "b_spec": b_spec,
    }


def series_block_bootstrap_se(
    t_idx: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 200,
    force_loop: bool = False,
) -> dict[str, float]:
    """Moving-block bootstrap on the detrended residual grid (preserving trend removal on each draw)."""
    n = len(t_idx)
    span = int(t_idx[-1] - t_idx[0]) + 1
    resid, _, _ = ols_detrend(t_idx, y)
    # Place residuals on span grid with NaN at missing slots
    full = np.full(span, np.nan, dtype=float)
    rel_pos = (t_idx - t_idx[0]).astype(int)
    full[rel_pos] = resid

    blk = max(2, int(math.floor(n ** (1.0 / 3.0))))
    n_blocks = int(math.ceil(span / float(blk)))
    max_start = max(1, span - blk + 1)

    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(blk, dtype=int)[None, None, :]
    idx_grid = (starts[:, :, None] + offsets).reshape(n_boot, n_blocks * blk)[:, :span]
    drawn = full[idx_grid]  # (B, span)

    k_max = (span - 1) // 2
    lx = np.log(np.arange(1, k_max + 1, dtype=float) / float(span)) if k_max >= 4 else None
    lx_c = (lx - lx.mean()) if lx is not None else None
    denom_x = float(lx_c @ lx_c) if lx_c is not None else 1.0
    t_grid_full = np.arange(span, dtype=float)

    if span == n and n >= 8 and not force_loop:
        t_c = t_grid_full - t_grid_full.mean()
        denom_t = float(t_c @ t_c)
        y_c = drawn - drawn.mean(axis=1, keepdims=True)
        eb = y_c - ((y_c @ t_c)[:, None] / denom_t) * t_c[None, :] if denom_t > 0.0 else y_c
        ss = np.sum(eb * eb, axis=1)
        sig2_b = ss / float(max(n - 2, 1))
        c0 = ss / float(n)
        if n - 1 >= 4:
            num1 = np.sum(eb[:, :-1] * eb[:, 1:], axis=1) / float(n - 1)
            r1_b = np.where(c0 > 1e-15, num1 / np.maximum(c0, 1e-30), np.nan)
        else:
            r1_b = np.full(n_boot, np.nan, dtype=float)
        if lx_c is not None:
            pw = (np.abs(np.fft.rfft(eb, axis=1)[:, 1 : k_max + 1]) ** 2) / float(n)
            ly = np.log(np.maximum(pw, 1e-30))
            bs_b = ((ly - ly.mean(axis=1, keepdims=True)) @ lx_c) / denom_x
        else:
            bs_b = np.full(n_boot, np.nan, dtype=float)
        return {
            "se_sigma2": float(np.nanstd(sig2_b, ddof=1)),
            "se_r1": float(np.nanstd(r1_b, ddof=1)),
            "se_b_spec": float(np.nanstd(bs_b, ddof=1)) if lx_c is not None else float("nan"),
        }

    sig2_b = np.empty(n_boot, dtype=float)
    r1_b = np.empty(n_boot, dtype=float)
    bs_b = np.empty(n_boot, dtype=float)
    valid_draws = 0
    for b in range(n_boot):
        row = drawn[b]
        ok = ~np.isnan(row)
        nb = int(ok.sum())
        if nb < 8:
            sig2_b[b] = np.nan
            r1_b[b] = np.nan
            bs_b[b] = np.nan
            continue
        tb = t_grid_full[ok]
        yb = row[ok]
        eb, _, _ = ols_detrend(tb, yb)
        sig2_b[b] = float((eb @ eb) / max(nb - 2, 1))
        c0 = float((eb @ eb) / nb)
        adj = np.flatnonzero(np.diff(tb) == 1.0)
        if len(adj) >= 4 and c0 > 1e-15:
            r1_b[b] = float((eb[adj] @ eb[adj + 1]) / (len(adj) * c0))
        else:
            r1_b[b] = np.nan
        if lx_c is not None:
            g = np.zeros(span, dtype=float)
            g[ok] = eb
            pw = (np.abs(np.fft.rfft(g)[1 : k_max + 1]) ** 2) / float(nb)
            ly = np.log(np.maximum(pw, 1e-30))
            bs_b[b] = float((lx_c @ (ly - ly.mean())) / denom_x)
        else:
            bs_b[b] = np.nan
        valid_draws += 1

    return {
        "se_sigma2": float(np.nanstd(sig2_b, ddof=1)) if valid_draws >= 10 else float("nan"),
        "se_r1": float(np.nanstd(r1_b, ddof=1)) if valid_draws >= 10 else float("nan"),
        "se_b_spec": float(np.nanstd(bs_b, ddof=1)) if valid_draws >= 10 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 3. Exact Closed-Form & Multi-Lag ACF Loss Formulas
# ---------------------------------------------------------------------------

def gamma0_two_stage(a: float, b: float, sigma: float, phi: float) -> float:
    if sigma == 0.0:
        return 0.0
    t0 = a + b
    if abs(phi) < 1e-15:
        return (sigma * sigma) / (2.0 * t0)
    r1 = phi / (a - (a - 1.0) * phi)
    r2 = phi / (b - (b - 1.0) * phi)
    re = r1 * r2
    one_plus_2theta = (1.0 + phi) / (1.0 - phi)
    return (sigma * sigma) * (one_plus_2theta / (2.0 * t0) - re / (1.0 - re))


_GH_X, _GH_W = np.polynomial.hermite.hermgauss(64)
_GH_Z = _GH_X * math.sqrt(2.0)
_GH_PROB = _GH_W / math.sqrt(math.pi)


def one_stage_loss_quadrature(a: float, sigma: float) -> float:
    """Exact unexpanded threshold loss D(a; sigma) for a 1-stage memoryless geometric
    cycle of mean stay a >= 1 with mean-one lognormal multiplier exp(sigma*Z - sigma^2/2).
    Because 1x1 matrices commute, D(a; sigma) is independent of temporal autocorrelation phi.
    """
    if sigma <= 0.0:
        return 0.0
    if abs(a - 1.0) < 1e-14:
        return 0.5 * sigma * sigma
    from scipy.optimize import brentq

    s0 = 1.0 - 1.0 / a
    g0 = 1.0 / a
    log_h = sigma * _GH_Z - 0.5 * sigma * sigma

    def f_obj(d_val: float) -> float:
        return float(np.sum(_GH_PROB * np.log(s0 + g0 * np.exp(d_val + log_h))))

    hi = 0.5 * sigma * sigma + 1e-9
    return float(brentq(f_obj, 0.0, hi, xtol=1e-14, rtol=1e-14))


def gamma0_from_full_acf(
    a: float, b: float, sigma: float, acf_dict: dict[int, float | None], bartlett_taper: bool = True
) -> float:
    """Exact second-order threshold loss Gamma_0 = T_0 * C * sigma^2 from Proposition 1
    (Statement ring-B1a, eq. ring.2) using Bartlett-tapered empirical ACF lags:
        C = c^2 / 2 - c * sum_{l>=1} w_l * rho_hat(l) * (u(l) - c),
    where c = 1 / (a + b) and u(l) = (1 - s_0) (M^{l-1})_{0, 1} at the threshold matrix M.
    """
    if sigma == 0.0:
        return 0.0
    s0 = 1.0 - 1.0 / a
    s1 = 1.0 - 1.0 / b
    # Threshold mean matrix M (2x2):
    # [ s0,      1 - s1 ]
    # [ 1 - s0,  s1     ]
    M = np.array([[s0, 1.0 - s1], [1.0 - s0, s1]], dtype=float)
    t0 = a + b
    c = 1.0 / t0
    valid_lags = sorted(k for k, v in acf_dict.items() if v is not None and not math.isnan(v))
    if not valid_lags:
        return (sigma * sigma) / (2.0 * t0)
    l_max = max(valid_lags)
    # Bartlett taper w_l = 1 - l / (l_max + 1) guarantees positive semi-definiteness of tapered autocovariance
    M_pow = np.eye(2, dtype=float)
    serial_sum = 0.0
    for lag in range(1, l_max + 1):
        # Fluctuating transfer is s0 -> s1, i.e., entry (1, 0) with b_entry = 1 - s0
        # u(lag) = b_entry * (M^{lag-1})[0, 1]
        u_lag = (1.0 - s0) * float(M_pow[0, 1])
        rho_l = acf_dict.get(lag)
        if rho_l is not None and not math.isnan(rho_l):
            w_l = (1.0 - float(lag) / float(l_max + 1)) if bartlett_taper else 1.0
            serial_sum += w_l * float(rho_l) * (u_lag - c)
        M_pow = M_pow @ M
    C_val = 0.5 * c * c - c * serial_sum
    return float(t0 * C_val * sigma * sigma)


def persistence_index_delta_E(a: float, b: float, sigma: float, phi: float, ln_r0: float) -> float:
    """Exact E(a, b) - E(a) from Result 4 / Lean P1 (valid for all -1 < phi < 1):
        theta = phi / (1 - phi), 1 + 2*theta = (1 + phi)/(1 - phi).
    """
    if sigma <= 1e-12:
        return 1.0  # deterministic limit: longer T increases 2*T*ln_R0
    theta = phi / (1.0 - phi)
    one_plus_2theta = (1.0 + phi) / (1.0 - phi)
    ell = ln_r0 / (sigma * sigma)
    denom = a * theta + b * (a + theta)
    if abs(denom) < 1e-12:
        return 1.0
    return (2.0 * b / one_plus_2theta) * (ell - (a * theta) / denom)


def thermal_loss_d2_one_exposed(
    a: float, b: float, sigma_t: float, phi_t: float, pi_sens: float, chi_sens: float
) -> float:
    """Closed-form second-order thermal threshold loss D_{2,T} from
    notes/mechanism-temperature-incubation.md §3.2 (Statement C3c) for a two-stage
    geometric ring (stays a >= 1 exposed, b >= 1 sheltered) with AR(1) temperature
    SD sigma_t, lag-1 correlation phi_t in (-1, 1), completion thermal sensitivity
    pi_sens = d ln p / dT, and clock thermal sensitivity chi_sens = d ln q / dT:
        D_{2,T} = sigma_t^2 * [ pi^2 * K(phi_t) + pi * chi * X_0(phi_t) ].
    """
    if sigma_t <= 0.0:
        return 0.0
    t0 = a + b
    if abs(phi_t) < 1e-15:
        return (sigma_t * sigma_t) * (pi_sens * pi_sens) / (2.0 * t0)
    rho0 = phi_t / (a - (a - 1.0) * phi_t)
    rho1 = phi_t / (b - (b - 1.0) * phi_t)
    re = rho0 * rho1
    one_plus_2theta = (1.0 + phi_t) / (1.0 - phi_t)
    k_term = one_plus_2theta / (2.0 * t0) - re / (1.0 - re)
    x0_term = (rho0 - re) / (1.0 - re)
    return (sigma_t * sigma_t) * ((pi_sens * pi_sens) * k_term + (pi_sens * chi_sens) * x0_term)


def thermal_loss_d2_resolvent(
    a: float, b: float, sigma_t: float, phi_t: float, pi_sens: float, chi_sens: float
) -> float:
    """Independent matrix-resolvent calculation of D_{2,T} (notes/mechanism-temperature-incubation.md
    §3.1 & notes/error-bound-general.md §13) used in Rule 7 check K0f.
    """
    if sigma_t <= 0.0:
        return 0.0
    q0 = 1.0 / a
    q1 = 1.0 / b
    # Critical mean matrix A0 (lambda = 1):
    # [ 1 - q0,   q1     ]
    # [ q0,       1 - q1 ]
    A0 = np.array([[1.0 - q0, q1], [q0, 1.0 - q1]], dtype=float)
    t0 = a + b
    v = np.array([a / t0, b / t0], dtype=float)  # right eigenvector A0 v = v, w^T v = 1 with w = [1, 1]
    w = np.array([1.0, 1.0], dtype=float)
    # dA/dT on column 0: s0 = 1 - q0 -> ds0/dT = -q0 * chi; g0 = p0*q0 (at threshold p0=1) -> dg0/dT = q0*(pi + chi)
    N = sigma_t * np.array([[-q0 * chi_sens, 0.0], [q0 * (pi_sens + chi_sens), 0.0]], dtype=float)
    Q_proj = np.eye(2, dtype=float) - np.outer(v, w)
    nv = N @ v
    wnv = float(w @ nv)
    qnv = Q_proj @ nv
    sol = np.linalg.solve(np.eye(2, dtype=float) - phi_t * A0, qnv)
    c_pp = 0.5 * (wnv * wnv) - phi_t * float(w @ (N @ sol))
    return float(t0 * c_pp)


def gpdd_timeperiod_to_month(tp: int) -> int:
    """Map GPDD TimePeriodID (timeperiod.csv) to representative calendar month 1..12 (0 if annual/unknown)."""
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
    if 74 <= tp <= 407:
        if 395 <= tp <= 407:
            return min(12, max(1, int((tp - 395) * 12 // 13) + 1))
        if 74 <= tp <= 388:
            return min(12, max(1, int((tp - 74) * 12 // 365) + 1))
    return 0


# ---------------------------------------------------------------------------
# 4. Raw Compilation Parsers (GPDD v2, BioTIME, VecDyn)
# ---------------------------------------------------------------------------

def biotime_bin_mode(month_counts):
    """Registered median distinct-month thresholds; fewer than 3 years is annual."""
    med_mo=float(np.median(month_counts)) if len(month_counts)>=3 else 1.0
    if med_mo>=6.0:return "subannual_monthly",12.0
    if med_mo>=2.0:return "subannual_seasonal",4.0
    return "annual",1.0


def load_all_raw_series(raw_dir: Path) -> tuple[list[dict], dict]:
    """Load candidate host abundance series (prior to coverage/zero-handling filters)
    from GPDD v2, BioTIME, and VecDyn, plus a companion fixed-plot BioTIME list for S6.
    """
    series_list: list[dict] = []
    audit_counts = {
        "gpdd_total_main": 0,
        "gpdd_target_taxa_series": 0,
        "biotime_studies_total": 0,
        "biotime_studies_retained": 0,
        "biotime_species_series_pooled": 0,
        "biotime_plot_series_s6": 0,
        "vecdyn_files_total": 0,
        "vecdyn_vectabundance_excluded": 0,
        "vecdyn_raw_files_retained": 0,
        "vecdyn_candidate_series": 0,
    }

    # --- A. GPDD v2 ---
    gpdd_dir = raw_dir / "gpdd"
    with (gpdd_dir / "taxon.csv").open(encoding="latin-1") as f:
        gpdd_taxa = {r["TaxonID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "biotope.csv").open(encoding="latin-1") as f:
        gpdd_bio = {r["BiotopeID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "location.csv").open(encoding="latin-1") as f:
        gpdd_loc = {r["LocationID"]: r for r in csv.DictReader(f)}
    with (gpdd_dir / "main.csv").open(encoding="latin-1") as f:
        gpdd_mains = {r["MainID"]: r for r in csv.DictReader(f)}
    audit_counts["gpdd_total_main"] = len(gpdd_mains)

    gpdd_rows_by_main: dict[str, list[dict]] = {}
    with (gpdd_dir / "data.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            gpdd_rows_by_main.setdefault(r["MainID"], []).append(r)

    for mid, m in gpdd_mains.items():
        t_row = gpdd_taxa.get(m["TaxonID"], {})
        b_row = gpdd_bio.get(m["BiotopeID"], {})
        t_grp, t_sub = classify_gpdd_taxon(t_row, b_row)
        if t_grp is None:
            continue
        # Sampling frequency classification
        sf_raw = (m.get("SamplingFrequency") or "").strip()
        if sf_raw == "1":
            iv_class, spy = "annual", 1.0
        elif sf_raw in ("2", "3", "4"):
            iv_class, spy = "subannual_seasonal", float(sf_raw)
        elif sf_raw in ("6", "12"):
            iv_class, spy = "subannual_monthly", float(sf_raw)
        elif sf_raw in ("52", "52*"):
            iv_class, spy = "subannual_biweekly_weekly", 52.0
        elif sf_raw == "7*":
            # GPDD code 7* is ~biweekly/4-week seasonal insect trapping (verified on MainID 8145); none are target hosts,
            # but handle cleanly as subannual_biweekly_weekly (13/yr) if encountered
            iv_class, spy = "subannual_biweekly_weekly", 13.0
        else:
            continue

        st_trans = (m.get("SourceTransform") or "").strip()
        if st_trans in ("Standardised", "Proportion"):
            continue
        notes_low = ((m.get("Notes") or "") + " " + (m.get("SamplingUnits") or "")).lower()

        drows = gpdd_rows_by_main.get(mid, [])
        if len(drows) < 12:
            continue
        audit_counts["gpdd_target_taxa_series"] += 1

        step_vals: dict[int, list[float]] = {}
        step_periods: dict[int, int] = {}
        step_syears: dict[int, int] = {}
        for r in drows:
            try:
                step = int(float(r["SeriesStep"]))
                p_untr = float(r["PopulationUntransformed"])
                p_raw = float(r["Population"])
            except (ValueError, TypeError):
                continue
            if p_untr <= -9990:
                continue
            val = p_untr
            # protocol 2.3: supplied numeric PopulationUntransformed takes precedence,
            # even if Population is equal and the source paper used a log plot.
            if math.isnan(val) or val < 0.0:
                continue
            step_vals.setdefault(step, []).append(val)
            try:
                step_periods[step] = int(float(r.get("TimePeriodID") or 0))
            except ValueError:
                step_periods[step] = 0
            try:
                syr = int(float(r.get("SampleYear") or -9999))
                if 1700 <= syr <= 2026 and "generation numbers, not calendar" not in notes_low:
                    step_syears[step] = syr
                else:
                    step_syears.setdefault(step, -9999)
            except ValueError:
                step_syears.setdefault(step, -9999)

        if len(step_vals) < 12:
            continue
        steps_sorted = sorted(step_vals.keys())
        loc_row = gpdd_loc.get(m.get("LocationID", ""), {})
        try:
            lat = float(loc_row.get("LatDD") or "nan")
        except ValueError:
            lat = float("nan")
        try:
            lon = float(loc_row.get("LongDD") or "nan")
        except ValueError:
            lon = float("nan")
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            lat, lon = float("nan"), float("nan")
        try:
            rel = int(float(m.get("Reliability") or -1))
        except ValueError:
            rel = -1

        hab_low = ((b_row.get("HabitatName") or "") + " " + (b_row.get("BiotopeType") or "")).lower()
        ocean_str = (loc_row.get("Ocean") or "").strip().lower()
        is_fw_gpdd = any(k in hab_low for k in FRESHWATER_BIOTOPE_KEYWORDS)
        is_mar_gpdd = ("marine" in hab_low) or ("tidal" in hab_low) or ("coastal" in hab_low and not is_fw_gpdd) or bool(ocean_str and ocean_str not in ("none", "na", "unknown", "land", "not applicable"))
        realm_det = "Freshwater" if is_fw_gpdd else ("Marine" if is_mar_gpdd else "Terrestrial")
        is_lab = ("microcosm" in notes_low) or ("aquarium" in notes_low) or ("indoor" in notes_low)

        series_list.append({
            "series_id": f"GPDD_{mid}",
            "database": "GPDD",
            "study_id": f"GPDD_SRC_{m.get('DataSourceID', mid)}",
            "licence": "CC-BY-4.0",
            "cc_by_compatible": True,
            "taxon_group": t_grp,
            "taxon_subgroup": t_sub,
            "species": (t_row.get("TaxonName") or "Unknown").strip(),
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm": "Freshwater" if is_fw_gpdd else "Terrestrial_or_Marine",
            "realm_detail": realm_det,
            "environment_type": "laboratory" if is_lab else "field",
            "reliability": rel,
            "manipulated": False,
            "series_kind": "primary",
            "steps": np.array(steps_sorted, dtype=int),
            "values": np.array([float(np.mean(step_vals[s])) for s in steps_sorted], dtype=float),
            "subperiod": np.array([step_periods[s] for s in steps_sorted], dtype=int),
            "step_years": np.array([step_syears.get(s, -9999) for s in steps_sorted], dtype=int),
            "step_months": np.array([gpdd_timeperiod_to_month(step_periods[s]) for s in steps_sorted], dtype=int),
            "step_dates_iso": None,
            "bin_days": 0,
        })

    # --- B. BioTIME (Zenodo 5026943 CSV + v2 realm cross-check) ---
    bt_dir = raw_dir / "biotime"
    bt_v2_realm: dict[str, str] = {}
    v2_meta_path = bt_dir / "biotime_v2_metadata_15April25.csv"
    if v2_meta_path.exists():
        with v2_meta_path.open(encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                bt_v2_realm[str(r.get("STUDY_ID", "")).strip()] = (r.get("REALM") or "").strip()

    bt_meta: dict[str, dict] = {}
    with (bt_dir / "BioTIMEMetadata_24_06_2021.csv").open(encoding="latin-1") as f:
        for r in csv.DictReader(f):
            sid = r["STUDY_ID"].strip()
            bt_meta[sid] = r
    audit_counts["biotime_studies_total"] = len(bt_meta)

    # First pass over BioTIME zip: determine per-study temporal resolution and distinct SAMPLE_DESC effort per bin
    study_ym_months: dict[str, dict[int, set[int]]] = {}
    with zipfile.ZipFile(bt_dir / "BioTIMEQuery_24_06_2021.zip") as zf:
        with zf.open("BioTIMEQuery_24_06_2021.csv") as bf:
            rd = csv.DictReader(io.TextIOWrapper(bf, encoding="latin-1"))
            for r in rd:
                sid = r["STUDY_ID"].strip()
                if sid not in bt_meta:
                    continue
                try:
                    yr = int(r["YEAR"])
                except ValueError:
                    continue
                m_str = (r.get("MONTH") or "").strip()
                if m_str and m_str != "NA":
                    try:
                        mo = int(m_str)
                        if 1 <= mo <= 12:
                            study_ym_months.setdefault(sid, {}).setdefault(yr, set()).add(mo)
                    except ValueError:
                        pass

    study_bin_mode: dict[str, tuple[str, float]] = {}
    for sid, st in bt_meta.items():
        ym = study_ym_months.get(sid, {})
        study_bin_mode[sid] = biotime_bin_mode([len(ms) for ms in ym.values()])
    audit_counts["biotime_binning"] = {
        sid: dict(interval_class=mode[0],samples_per_year=mode[1],
                  median_distinct_months=float(np.median([len(ms) for ms in study_ym_months[sid].values()]))
                  if len(study_ym_months.get(sid,{}))>=3 else 1.0)
        for sid,mode in study_bin_mode.items()
    }

    # Second pass over BioTIME zip: accumulate effort (distinct SAMPLE_DESC per study & bin) and species sums
    study_events_per_bin: dict[str, dict[int, set[str]]] = {}
    sp_sum_per_bin: dict[tuple[str, str], dict[int, float]] = {}
    sp_info: dict[tuple[str, str], tuple[str, str, str]] = {}  # (t_grp, t_sub, species_name)
    # Companion fixed-plot accumulator for S6 (only for studies with meaningful plot IDs and <= 15 plots to keep RAM small)
    plot_events_per_bin: dict[tuple[str, str], dict[int, set[str]]] = {}
    plot_sp_sum: dict[tuple[str, str, str], dict[int, float]] = {}

    with zipfile.ZipFile(bt_dir / "BioTIMEQuery_24_06_2021.zip") as zf:
        with zf.open("BioTIMEQuery_24_06_2021.csv") as bf:
            rd = csv.DictReader(io.TextIOWrapper(bf, encoding="latin-1"))
            for r in rd:
                sid = r["STUDY_ID"].strip()
                st = bt_meta.get(sid)
                if st is None:
                    continue
                iv_class, _ = study_bin_mode[sid]
                try:
                    yr = int(r["YEAR"])
                except ValueError:
                    continue
                if iv_class == "annual":
                    t_bin = yr
                elif iv_class == "subannual_monthly":
                    m_str = (r.get("MONTH") or "").strip()
                    if not m_str or m_str == "NA":
                        continue
                    try:
                        mo = int(m_str)
                    except ValueError:
                        continue
                    if not (1 <= mo <= 12):
                        continue
                    t_bin = yr * 12 + (mo - 1)
                else:  # subannual_seasonal (quarters)
                    m_str = (r.get("MONTH") or "").strip()
                    if not m_str or m_str == "NA":
                        continue
                    try:
                        mo = int(m_str)
                    except ValueError:
                        continue
                    if not (1 <= mo <= 12):
                        continue
                    t_bin = yr * 4 + ((mo - 1) // 3)

                sdesc = r.get("SAMPLE_DESC") or f"{yr}"
                study_events_per_bin.setdefault(sid, {}).setdefault(t_bin, set()).add(sdesc)

                plt = (r.get("PLOT") or "").strip()
                if plt and plt != "NA":
                    plot_events_per_bin.setdefault((sid, plt), {}).setdefault(t_bin, set()).add(sdesc)

                ab_str = (r.get("sum.allrawdata.ABUNDANCE") or "").strip()
                if not ab_str or ab_str == "NA":
                    continue
                try:
                    ab = float(ab_str)
                except ValueError:
                    continue
                if ab <= 0.0:
                    continue

                sp_name = (r.get("GENUS_SPECIES") or r.get("ID_SPECIES") or "").strip()
                key = (sid, sp_name)
                if key not in sp_info:
                    t_grp, t_sub = classify_biotime_record(st, bt_v2_realm.get(sid), r.get("GENUS", ""), sp_name)
                    sp_info[key] = (t_grp or "", t_sub or "", sp_name)
                if not sp_info[key][0]:
                    continue

                sp_Bins = sp_sum_per_bin.setdefault(key, {})
                sp_Bins[t_bin] = sp_Bins.get(t_bin, 0.0) + ab

                plt = (r.get("PLOT") or "").strip()
                if plt and plt != "NA":
                    pk = (sid, plt)
                    if True:
                        plot_events_per_bin.setdefault(pk, {}).setdefault(t_bin, set()).add(sdesc)
                        psk = (sid, plt, sp_name)
                        p_bins = plot_sp_sum.setdefault(psk, {})
                        p_bins[t_bin] = p_bins.get(t_bin, 0.0) + ab

    retained_bt_studies = set()
    for (sid, sp_name), sp_bins in sp_sum_per_bin.items():
        t_grp, t_sub, _ = sp_info[(sid, sp_name)]
        if not t_grp:
            continue
        ev_map = study_events_per_bin.get(sid, {})
        if len(ev_map) < 15:
            continue
        st = bt_meta[sid]
        lic = (st.get("LICENSE") or "CC-BY").strip()
        cc_by_ok = ("nc" not in lic.lower())
        iv_class, spy = study_bin_mode[sid]
        treat_txt = ((st.get("GENERAL_TREAT") or "") + " " + (st.get("TREATMENT") or "")).strip().lower()
        manip = bool(treat_txt and treat_txt not in ("na", "none", "na na", "control", ""))
        try:
            lat = float(st.get("CENT_LAT") or "nan")
        except ValueError:
            lat = float("nan")
        try:
            lon = float(st.get("CENT_LONG") or "nan")
        except ValueError:
            lon = float("nan")
        realm_eff = bt_v2_realm.get(sid) or (st.get("REALM") or "").strip()
        realm_det = realm_eff if realm_eff in ("Marine", "Freshwater", "Terrestrial") else ("Marine" if "marine" in (st.get("HABITAT") or "").lower() else "Terrestrial")

        all_sampled_bins = sorted(ev_map.keys())
        vals = np.array([sp_bins.get(b, 0.0) / float(len(ev_map[b])) for b in all_sampled_bins], dtype=float)
        if int(np.sum(vals > 0.0)) < 15:
            continue
        retained_bt_studies.add(sid)
        audit_counts["biotime_species_series_pooled"] += 1
        sub_p = np.array(
            [b % 12 if iv_class == "subannual_monthly" else (b % 4 if iv_class == "subannual_seasonal" else 0) for b in all_sampled_bins],
            dtype=int,
        )
        step_yrs = np.array(
            [b // 12 if iv_class == "subannual_monthly" else (b // 4 if iv_class == "subannual_seasonal" else b) for b in all_sampled_bins],
            dtype=int,
        )
        step_mos = np.array(
            [(b % 12) + 1 if iv_class == "subannual_monthly" else ((b % 4) * 3 + 2 if iv_class == "subannual_seasonal" else 0) for b in all_sampled_bins],
            dtype=int,
        )
        series_list.append({
            "series_id": f"BT_{sid}_{sp_name.replace(' ', '_')}",
            "database": "BioTIME",
            "study_id": f"BT_STUDY_{sid}",
            "licence": lic,
            "cc_by_compatible": cc_by_ok,
            "taxon_group": t_grp,
            "taxon_subgroup": t_sub,
            "species": sp_name,
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm": realm_eff,
            "realm_detail": realm_det,
            "environment_type": "field",
            "reliability": 4,
            "manipulated": manip,
            "series_kind": "primary",
            "steps": np.array(all_sampled_bins, dtype=int),
            "values": vals,
            "subperiod": sub_p,
            "step_years": step_yrs,
            "step_months": step_mos,
            "step_dates_iso": None,
            "bin_days": 0,
        })

    audit_counts["biotime_studies_retained"] = len(retained_bt_studies)

    # Companion BioTIME fixed-plot series for Sensitivity S6
    for (sid, plt, sp_name), p_bins in plot_sp_sum.items():
        t_grp, t_sub, _ = sp_info.get((sid, sp_name), ("", "", sp_name))
        if not t_grp:
            continue
        pev = plot_events_per_bin.get((sid, plt), {})
        if len(pev) < 15:
            continue
        all_b = sorted(pev.keys())
        vals = np.array([p_bins.get(b, 0.0) / float(len(pev[b])) for b in all_b], dtype=float)
        if int(np.sum(vals > 0.0)) < 15:
            continue
        st = bt_meta[sid]
        lic = (st.get("LICENSE") or "CC-BY").strip()
        iv_class, spy = study_bin_mode[sid]
        treat_txt = ((st.get("GENERAL_TREAT") or "") + " " + (st.get("TREATMENT") or "")).strip().lower()
        manip = bool(treat_txt and treat_txt not in ("na", "none", "na na", "control", ""))
        try:
            lat = float(st.get("CENT_LAT") or "nan")
        except ValueError:
            lat = float("nan")
        try:
            lon = float(st.get("CENT_LONG") or "nan")
        except ValueError:
            lon = float("nan")
        realm_eff_p = bt_v2_realm.get(sid) or (st.get("REALM") or "").strip()
        realm_det_p = realm_eff_p if realm_eff_p in ("Marine", "Freshwater", "Terrestrial") else "Terrestrial"
        audit_counts["biotime_plot_series_s6"] += 1
        series_list.append({
            "series_id": f"BTPLOT_{sid}_{plt}_{sp_name.replace(' ', '_')}",
            "database": "BioTIME",
            "study_id": f"BT_STUDY_{sid}",
            "licence": lic,
            "cc_by_compatible": ("nc" not in lic.lower()),
            "taxon_group": t_grp,
            "taxon_subgroup": t_sub,
            "species": sp_name,
            "interval_class": iv_class,
            "samples_per_year": spy,
            "latitude": lat,
            "longitude": lon,
            "realm": realm_eff_p,
            "realm_detail": realm_det_p,
            "environment_type": "field",
            "reliability": 4,
            "manipulated": manip,
            "series_kind": "biotime_fixed_plot_only",
            "steps": np.array(all_b, dtype=int),
            "values": vals,
            "subperiod": np.zeros(len(all_b), dtype=int),
            "step_years": np.zeros(len(all_b), dtype=int),
            "step_months": np.zeros(len(all_b), dtype=int),
            "step_dates_iso": None,
            "bin_days": 0,
        })

    # VecDyn is loaded separately with the registered location key.
    return series_list, audit_counts
