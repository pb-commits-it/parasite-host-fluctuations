# Protocol: Environmental covariates in the Cary and NEON tick-pathogen studies (`envcov`)

**Timestamp:** `2026-11-14 22:11:07 CST` (from `date`).  
**Written before:** any run of `data/fetch/envcov.py` and before reading any environmental covariate values in `data/raw/envcov/` or the weather columns in `data/raw/cary_lyme/`.  
**Pre-analysis power & identifiability:** `casestudies/envcov/power_precheck.py`, saved in `casestudies/envcov/results/power_precheck_20261114.txt` (with first-attempt diagnostic log preserved in `casestudies/envcov/results/power_precheck_attempt1_20261114.txt`).  
**Literature log & cards:** `casestudies/envcov/literature/SEARCH_LOG.csv` and `casestudies/envcov/literature/cards/*.md`.

---

## 1. Question

The two earlier field evaluations of the mouse–tick–*Borrelia burgdorferi* system (`reports/R12-cary-single-population.md`, `reports/R12b-cary-holdout.md`, `reports/R12c-cary-how-mice-enter.md`, and `reports/R14-neon-across-sites.md` with `results/r14-posthoc-power_20261114.txt`) left their registered tests inconclusive: across six NEON sites, rank correlation between nymphal infection prevalence (`NIP_site`) and mouse log-abundance standard deviation (`sigma_site`) was `+0.4857` (against a predicted negative sign; exact one-sided `p = 0.8514` for negative, `0.1778` for positive; registered outcome "not decided"), while within the Cary Institute grids the slope of `ln NIP_{t+1}` on `ln mice_t` was `0.038 ± 0.058` in 1995–2013 (`0.051 ± 0.118` in 2014–2022), inside `[0, 1 - pbar]` yet indistinguishable from zero.

The standard ecological objection to both field comparisons is that abiotic conditions (temperature, humidity/saturation deficit, precipitation, winter cold, snow cover, and forest canopy habitat) directly govern *Ixodes scapularis* development, survival, and questing while also covarying with oak masting and rodent fluctuations across sites and years. This case study tests four specific questions:

1. **Covariate construction.** What are the site- and year-specific values of six biologically defined tick abiotic covariates across the six R14 NEON sites and across the Cary Institute time series?
2. **Confounding check.** How strongly do those abiotic covariates correlate with mouse variability (`sigma_site`) and pathogen prevalence (`NIP_site`) across the six NEON sites, and with annual mouse abundance and next-year `NIP` at Cary? When the registered R14 cross-site comparison and the Cary within-plot slope are re-estimated with pre-specified abiotic adjustments, how much do the point estimates and standard errors move—and what can and cannot be adjusted for with six sites (`N = 6`) or eighteen annual cohorts (`T = 18`)?
3. **Better theory inputs.** When the borrowed fixed 10-month tick stay and memoryless stay shape used in R12/R14 are replaced by site- and year-specific stage durations (and year-to-year stage-duration spread) computed from a published *I. scapularis* temperature-development model, how much does the predicted fluctuation loss `D` of `ln R0` move at Cary and at each NEON site?
4. **Masking and mimicking.** In the project's lifecycle engine (`studies/engine-R16/lifecycle_engine.py`), when site- and year-level weather drives tick survival and development while mice fluctuate with the empirically observed weather–mouse covariances and sample sizes, how often would the registered R14 and R12 tests show the theory's sign (or reach statistical support) when the host-variability effect is absent (`kappa = 0`) versus present at the predicted size (`kappa = 0.5` and `1.0`)?

All earlier registered outcomes in R12, R12b, R12c, and R14 stand unchanged. Adjustments and extensions beyond the registered R12/R14 rules are pre-specified here for `envcov` and labelled post hoc with respect to R12/R14 where they re-analyse those biological series.

---

## 2. Theory Predictions and Engine Calls

1. **Fluctuation loss `D` of `ln R0` against the arithmetic-mean-rate baseline.**
   - In a discrete-time two-stage *Borrelia* cycle with mouse infectious stay `tau_1` steps and tick larva-to-nymph stay `tau_2` steps (mean generation time `T_0 = tau_1 + tau_2`), where host abundance multiplies one obligate transfer with elasticity `kappa` (`h_t` lognormal with log-SD `kappa * sigma` and step-to-step autocorrelation `phi >= 0`), the second-order loss per generation (`notes/exact-limits.md`, H2; `notes/mechanism-temperature-incubation.md`, §3.3) is
     `D_2 = (kappa * sigma)^2 * [ (1 + 2 * theta) / (2 * T_0) - R / (1 - R) ]`,
     where `theta = phi / (1 - phi)` and `R = E[phi^L]` is the generating function of the threshold lineage return time `L` (`R = prod_k (1 + tau_k / (m_k * theta))^(-m_k)` for serial chains of `m_k` geometric sub-stages; `R = 0` at `phi = 0`, giving `D_2 = (kappa * sigma)^2 / (2 * T_0)`).
   - Exact finite-noise `D`, second-order `D_2`, and the visit-weighted edge Jensen upper bound `J` (`D <= J`; `notes/exact-limits.md`, H4; `notes/mechanism-temperature-incubation.md`, Proposition B) are computed via `Engine(Description(...)).critical().loss(...)` and `.coefficient()` in `studies/engine-R16/lifecycle_engine.py`.
   - **Engine caveats observed (`verification/engine-R16-check.md`):**
     - `Driver(kind='bounded', low=0.05)` holds the raw lower bound fixed rather than scaling a standardized shape as `sigma -> 0`; we use `kind='lognormal'` for standard host/weather multipliers and only `low=None` or explicitly documented bounded states when testing contact-limited leaving (`contact=True`).
     - `Engine.population(..., offspring='shared_groups')` groups outgoing route leavers (`det=False`) while keeping staying (`det=True`) binomial across individuals; we do not rely on `shared_groups` in `population()`.

2. **How temperature and abiotic covariates enter the theory (`notes/mechanism-temperature-incubation.md`, §§1–4; `notes/exact-limits.md`, H1–H4):**
   - **Through stage durations and their spread (Part 3):** Warmer sites accumulate moulting temperature faster, altering mean tick stage duration `tau_2` and `T_0`. Furthermore, seasonal gating of larval feeding (late summer) and spring nymphal questing (above `10 °C`) makes the realized larva-to-nymph stay much more regular (`CV << 1`, represented as an `m`-substage chain `shape='chain'` or fixed delay `shape='fixed'` on a fine time step) than the memoryless geometric stay (`m = 1`) assumed in R12/R14. By Proposition D (`notes/mechanism-temperature-incubation.md`, §2.4) and `reports/R10b-real-systems-stay-shape.md`, greater regularity raises `D` whenever `phi > 0`, and at finite noise raises `D` toward the cohort cap `(kappa * sigma)^2 / 2` as both stages approach fixed delays.
   - **Through abiotic fluctuations on tick survival/transfer and correlated host–weather drivers (Part 4):** If abiotic driver `W` alters tick survival/completion (elasticity `pi_W`) and covaries with mouse abundance `H` (correlation `rho_{HW}`), then:
     - Across sites, site-mean climate differences shift baseline `ln R0_site = m + beta_W * Z_site`, directly confounding `NIP_site = 1 - exp(-f * max(ln R0_site - D_site, 0))` whenever `Z_site` correlates with `sigma_site`.
     - Within a site over time, coupled abiotic and host drivers enter `Description(drivers=[Driver('host', ...), Driver('weather', ...)], correlation=[[1, rho], [rho, 1]])` in `lifecycle_engine.py`, adding a cross-covariance term `X` (`notes/exact-limits.md`, linked hosts) to `D` while directly perturbing annual nymphal survival and prevalence.

---

## 3. Data and Provenance

All data are legitimately open, fetched by `data/fetch/envcov.py` with generic `User-Agent` (`envcov-study-fetch/1.0 (python-urllib)`), no email or personal details, and **no API token** (no file under `[local working directory]Downloads` is read). Raw files are stored under `data/raw/envcov/` (verified git-ignored via `.gitignore:3:data/raw/`).

1. **NEON public metadata and unauthenticated data-endpoint check (`https://data.neonscience.org/api/v0/`):**
   - Sites: the six R14 included sites (`BLAN`, `HARV`, `ORNL`, `SCBI`, `SERC`, `TREE`).
   - Products queried via public `/products/<code>` and `/sites/<site>`: single-aspirated air temperature (`DP1.00002.001`), triple-aspirated air temperature (`DP1.00003.001`), precipitation (`DP1.00006.001`), relative humidity (`DP1.00098.001`), soil temperature (`DP1.00041.001`), soil water content (`DP1.00094.001`), and ecosystem structure / lidar canopy height (`DP3.30015.001`).
   - Unauthenticated `/data/<code>/<site>/<YYYY-MM>?package=basic` requests are attempted without a token and logged. Because NEON requires an `X-API-Token` header on `/data/` endpoints (returning `HTTP 403 Access Denied` without one, documented in `data/INVENTORY.md` §2), `envcov.py` records the exact HTTP status and does not circumvent it.
   - Already-downloaded NEON field tables in `data/raw/neon/` (`DP1.10072.001`, `DP1.10093.001`, `DP1.10092.001`, `RELEASE-2026`, CC0 1.0) supply the R14 biological series plus plot-level static habitat metadata (`nlcdClass` forest-cover class and `elevation` on the sampled tick-drag plots).

2. **Primary daily weather source — NASA ORNL DAAC Daymet Version 4 R1 (`https://daymet.ornl.gov/single-pixel/api/data`):**
   - Citation & DOI: Thornton et al. (2022), *Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1*, ORNL DAAC, `https://doi.org/10.3334/ORNLDAAC/2129`. Licence: NASA / US Government open public-domain earth science data (CC0-equivalent open reuse with citation).
   - Daily variables (`1990-01-01` through `2025-12-31`, covering every Cary and NEON study year plus lagged `t-1` winters): `tmax` (°C), `tmin` (°C), `prcp` (mm/day), `vp` (water vapor pressure, Pa), `swe` (snow water equivalent, kg/m²), `srad` (W/m²), `dayl` (s/day).
   - Coordinates:
     - **Cary Institute (Millbrook, NY):** `41.783622, -73.732358` (`41°47'1.04"N, 73°43'56.49"W`), the published coordinates of the Cary Lyme study plots in Burtis et al. (2016, `PMC5124264`, read in full; raw Cary Figshare/Zenodo metadata JSONs leave geospatial coordinates blank).
     - **Six R14 NEON sites (primary):** mean coordinates of each site's `DP1.10093.001` tick-drag plots (where the nymphs were actually collected), plus official NEON tower/site coordinates from `/api/v0/sites/<site>` as a spatial sensitivity check (important at `HARV`, where tick plots and main tower lie ~13 km apart), plus the remaining 7 R14 candidate NEON sites (`DELA, KONZ, MLBS, OSBS, STEI, TALL, UKFS`) for secondary 13-site climate–mouse context.

3. **Open daily supplement for relative humidity, surface soil moisture, and surface skin/soil temperature — NASA POWER Agroclimatology API (`https://power.larc.nasa.gov/api/temporal/daily/point`):**
   - Licence: NASA / US Government public domain (no token required).
   - Daily variables (`1990-01-01` to `2025-12-31`) at Cary and the six R14 NEON tick-plot coordinates: 2-m relative humidity (`RH2M`, %), surface soil wetness 0–5 cm (`GWETTOP`, unitless wetness fraction 0–1), and earth skin/surface temperature (`TS`, °C), used as open soil/humidity checks where NEON terrestrial instrument `/data/` requires a token, and cross-checked against Daymet vapor-pressure-derived relative humidity and saturation deficit.

4. **Ground-station cross-validation at Cary (`data/raw/cary_lyme/`):**
   - `Lyme_Study_Dataset.csv` (`doi:10.5061/dryad.d1c8046`, CC0) and `Full_dataset_20260326.csv` (`doi:10.25390/caryinstitute.31872853.v2`, CC BY 4.0) contain annual Cary weather-station summaries (`mat`, `degree_days` Oct–Sep base `0 °C`, `cold_winter_days` `T_min < -10 °C` Nov–Mar, `feb_apr_warm_days`, `snowfall_cm`). Daymet-derived counterparts at Cary are correlated against these ground-station columns to verify 1-km gridded fidelity before using Daymet across sites, and Cary ground-station covariates are run as a parallel sensitivity check for Cary's slope.

5. **Optional remote-sensing habitat products (MODIS LST, NEON AOP lidar, GEDI):**
   - NEON AOP (`DP3.30015.001`) `/data/` requires an API token (`HTTP 403` without one). ORNL DAAC MODIS/GEDI Subset REST (`modis.ornl.gov/rst/api/v1/products`) is probed once by `envcov.py` with a 25 s timeout; if unreachable without NASA Earthdata Login or timed out, it is not used and reported plainly. Plot-level forest cover fraction (`nlcdClass` in `deciduousForest`, `mixedForest`, `evergreenForest`, `woodyWetlands`) and elevation from NEON's `tck_fielddata` (`DP1.10093.001`) provide an exact, zero-overhead site habitat check across the six NEON sites.

---

## 4. Definitions of the Six Abiotic Tick Covariates and Thermal Stage-Duration Models

Let `T_mean,d = 0.5 * (T_max,d + T_min,d)` on day `d`. Every covariate is defined for biological year `t` aligned to the nymphal cohort collected in summer of year `t` (whose larvae fed in late summer of year `t - 1` and overwintered in winter `(t - 1) -> t`):

| Code | Covariate | Exact formula (primary, from Daymet) | Literature basis & read status |
|---|---|---|---|
| `C1_dd6` | **Degree-days above tick development threshold** | Cumulative `sum max(0, T_mean,d - 6.0)` (°C·day) over the 12-month tick year `1 Oct (t-1)` to `30 Sep t` (with `DD > 0 °C` and `DD > 10 °C` reported in sensitivity checks). | `6 °C` development base in Allen (2025, `PMC12040446`, Table 1, **read in full**, citing Ogden et al. 2004, **abstract read**); Oct–Sep window from Ostfeld et al. (2018 README, **read in full**); `DD > 0 °C` from Ogden et al. (2008 `PMC2412857`; 2014 `PMC4050516`, **read in full**). |
| `C2_hotdry` | **Hot dry days** | Count of days in `1 May`–`30 Sep` of year `t` (questing season; also split into `t-1` larval Aug–Sep and `t` nymphal May–Jul in sensitivity) with `T_max,d > 25.0 °C` and `prcp_d == 0 mm`. | Exact definition on the Cary plots in Burtis et al. (2016, `PMC5124264`, **read in full**). |
| `C3_wmin` | **Winter minimum temperature** | Coldest daily minimum temperature `min_d T_min,d` (°C) over `1 Nov (t-1)`–`31 Mar t` (with count of days `T_min,d < -10 °C` reported alongside). | Linske et al. (2019, `PMC6723576`, **read in full**); Ostfeld & Brunner (2015, `PMC4342967`, **read in full**); Ostfeld et al. (2018 README `cold_winter_days`, **read in full**). |
| `C4_snow` | **Snow days** | Count of winter days (`1 Nov (t-1)`–`31 Mar t`) with snow cover `swe_d > 0 kg/m^2` (`> 0 mm` water equivalent). | Snow-presence overwinter insulation in Linske et al. (2019, `PMC6723576`, **read in full**) and SWE predictor in Angell et al. (2026, `PMC13380151`, **read in full**). |
| `C5_gsprcp` | **Growing-season precipitation** | Total precipitation `sum prcp_d` (mm) over `1 Apr`–`30 Sep` of year `t` (with `t-1` growing-season precipitation also checked for larval-year carryover). | Clow et al. (2017, `PMC6086181`, **read in full**); Burtis et al. (2016, `PMC5124264`, **read in full**). |
| `C6_satdef` | **Questing-season saturation deficit** | Mean daily vapor pressure deficit `SD_d = max(0, e_s(T_mean,d) - vp_d / 1000)` in `kPa` (`1 kPa = 7.5006 mmHg`) over `1 May`–`30 Sep` of year `t`, where `e_s(T) = 0.61078 * exp(17.269 * T / (T + 237.3))` kPa (Tetens formula **[recalled standard meteorological formula]**). Cross-checked against June sub-82% `RH2M` days from NASA POWER (`PMC3991885`, Berger et al. 2014, **read in full**). | VPD (`kPa`) in Angell et al. (2026, `PMC13380151`, **read in full**); saturation deficit questing height mechanism in Ostfeld & Brunner (2015, `PMC4342967`, **read in full**) and Randolph & Storey (1999, **abstract read**). |

*(Secondary soil/humidity checks from NASA POWER: May–Sep mean surface soil wetness `GWETTOP`, winter mean surface temperature `TS`, and June days with `RH2M < 82%`; habitat check at NEON: tick-plot forest fraction and elevation.)*

### Published temperature-development models for Part 3 (stage durations)

1. **Primary verified full-text model — Wallace et al. (2019, `PMC6766261`, Table 1, read in full; fitted to Ogden et al. 2004 laboratory data):**
   - Daily engorged larva-to-nymph maturation rate:
     `m_LN(T) = 0.04001 * exp(-((T - 26.68) / 9.533)^2) * I(T >= 15.0)` (`day^-1`).
   - Daily engorged nymph-to-adult maturation rate:
     `m_NA(T) = 0.03173 * exp(-((T - 23.85) / 9.042)^2) * I(T >= 15.0)` (`day^-1`).
   - For an individual engorging on drop-off date `t_0` in year `y`, **physiological premoult duration** `M_stage(y, t_0)` is the minimal number of days `L >= 1` until `sum_{u=0}^{L-1} m_stage(T_{mean, t_0 + u}) >= 1.0`.
     - Standard reference drop-off dates: `15 August` (`DOY 227`, peak larval feeding in Ostfeld et al. 2018 README p. 2, read in full; cf. day 221 in Dunn et al. 2013 `PMC3913058`, read in full) for `L -> N`, and `1 June` (`DOY 152`; cf. day 148 in Dunn et al. 2013) for `N -> A`.
     - **Within-season drop-off spread:** weighted across larval drop-off dates every 5 days from `15 July` to `15 September` (`DOY 196` to `258`, Gaussian weights SD `20 d` centred at `15 August`) to capture both early cohorts that complete moulting before autumn cooling and late cohorts arrested over winter.
   - **Full phenological inter-feeding stay `S_LN(y, t_0)` (larval meal on mouse to nymphal meal on mouse — the *Borrelia* steady stage):**
     In nature, larvae that finish moulting in late summer/autumn enter behavioural diapause/winter quiescence rather than feeding immediately as nymphs in autumn, and resume questing in spring once temperatures reach `T_mean >= 10 °C` (Allen 2025, `PMC12040446`, Table 1, read in full), followed by host-finding at rate `0.02 day^-1` on active days `T_mean >= 10 °C` (mean active questing wait `50` active degree-threshold days, Allen 2025 Table 1; sensitivity `0.04 day^-1` = `25` active days). Specifically, a cohort dropping off at `t_0` completes moult at `t_moult = t_0 + M_LN(y, t_0)`, waits at least until `1 March` of `y + 1` if `t_moult` occurs before winter (post-solstice diapause, Allen 2025), and then accumulates active questing hazard `0.02 * I(T_mean,t >= 10 °C)` until cumulative hazard reaches `1` (median/mean host contact).
   - **Recalled power-law sensitivity check (labelled recalled):** Ogden et al. (2004, `10.1603/0022-2585-41.4.622`, abstract read; power-law constants `D_LN(T) = 101181 * T^(-2.275)` days for `T > 8.5 °C` and `D_NA(T) = 1596 * T^(-1.21)` days for `T > 8.0 °C` **[recalled, not read in full]**) is computed alongside Wallace et al. (2019) to verify that conclusions do not hinge on the `15 °C` Heaviside vs power-law tail.

2. **Replacing borrowed fixed stays in `D`:**
   - Mouse infectious stay is kept at the literature mean `133 days` (`4.4 months`, range `85–185 days`, Dunn et al. 2013 `PMC3913058`, read in full).
   - At each site (Cary and the 6 R14 NEON sites), we compute `D` on a fine discrete step `dt = 6 days` (`0.197 months`, so `tau_mouse = 22.17` steps and all serial chains are admissible) and compare against R12/R14's coarse-step (`dt = 4.4 months`, `tau = [1, 2.273]`) value across:
     - **(M0)** R12/R14 coarse benchmark (`dt = 4.4 mo`, `tau = [1, 2.273]`, geometric stay, `phi = 0`);
     - **(M1)** Fine-step benchmark with borrowed `304-day` (`10-mo`) tick stay (`shape='memoryless'` vs `shape='chain'`);
     - **(M2)** Site-specific **physiological premoult duration** `mean(M_LN)` only (testing what happens if diapause is ignored);
     - **(M3)** Site-specific **full phenological inter-feeding duration** `mean(S_LN)` with **memoryless** tick stay (`m = 1`) vs **chain stay matched to empirical year-to-year + cohort `CV`** (`m = min(round(1 / CV^2), floor(mean_steps))`), under both `phi = 0` (R12/R14 empirical lag-1 rule) and `phi_{4.4mo} = 0.30` (`phi_{6d} = 0.30^(6 / 133.8) = 0.9475`, manuscript oak-forest intra-annual memory).

---

## 5. Registered Estimands, Tests, and Criteria

### Part 2A: Across the six NEON sites (`N = 6`: BLAN, HARV, ORNL, SCBI, SERC, TREE)

- **What can and cannot be adjusted for (`power_precheck_20261114.txt`, Section 1):**
  - With `N = 6` sites, an unadjusted correlation has `4` residual degrees of freedom. Adjusting for **one** covariate leaves **`3` residual degrees of freedom**; adjusting for `2` or `3` leaves `2` or `1` df (fragile); adjusting for `4` or more covariates leaves **`0` df (saturated, unidentifiable)**.
  - Even when adjusting for the *exact* data-generating confounder (`r_{Z,sigma} = +0.60, beta_Z = +0.20, tau_res = 0.05–0.10`), pre-analysis simulation shows that Freedman–Lane 1-sided partial Spearman power at `alpha = 0.05` is only **`6.6%` to `7.4%`** (at best **`14.1%`** if `r_{Z,sigma} = 0` and `tau_res = 0.05`), while screening 6 unrelated covariates at `alpha = 0.05` yields a **`14.6%` familywise false-support rate**.
  - **Therefore:** Six sites can diagnose *confounding direction* (whether an abiotic gradient correlates positively with both `sigma_site` and `NIP_site`, and whether partial correlation shifts from `+0.49` toward zero or negative), **but cannot confirm the theory with high statistical power**. Simultaneous multi-covariate adjustment (`p >= 2`) is NOT identifiable and will not be used for inference.
- **Estimands & Primary Test (`E_NEON_PC1`):**
  1. Site-mean covariates `Z_{s,j}` averaged over the years in which ticks were tested at site `s` (`2014–2025`), standardized across the 6 sites; pairwise Spearman and Pearson correlations of each `Z_{.,j}` (plus NASA POWER soil wetness/RH and tick-plot forest fraction/elevation) with `sigma_site` and `NIP_site`.
  2. **Pre-specified primary adjusted test (`df = 3`):** Compute `PC1_env`, the first principal component of the 6 standardized primary Daymet covariates `(C1_dd6, C2_hotdry, C3_wmin, C4_snow, C5_gsprcp, C6_satdef)` across the 6 sites (sign oriented so higher `PC1_env` = warmer / higher `C1_dd6`, independent of `NIP` or `sigma`). Test the partial Spearman rank correlation `r_part(NIP_site, sigma_site | PC1_env)` by the exact `720`-permutation Freedman–Lane rank residual test (`power_precheck.py`):
     - **"Supports"** if `r_part < 0` and 1-sided Freedman–Lane `p < 0.05`;
     - **"Contradicts"** if `r_part > 0` and opposite 1-sided `p < 0.05`;
     - **"Not decided"** otherwise.
  3. **Secondary single-covariate adjustments (`df = 3` each, descriptive / multiplicity-controlled):** Report `r_part(NIP_site, sigma_site | Z_j)` and binomial logit-link WLS slope of `logit(NIP_site)` on `sigma_site + Z_j` (weighted by `n_site p_site (1 - p_site)`) for each of the 6 covariates one at a time, judged against Bonferroni `alpha = 0.05 / 6 = 0.0083` to prevent false discovery.
  4. **Within-site NEON pooled slope (`T3_adj`, `46` pairs across 6 sites):** Re-estimate R14's T3 pooled within-site slope `b` of `ln NIP_{s,t}` on `ln index_{s,t-1}` adding `PC1_env_{s,t}` (and each covariate `Z_{s,t,j}` one at a time), with site fixed effects and 2,000 within-site resamples.

### Part 2B: Within Cary across years (`1995–2013` Dryad primary, `102` plot-year pairs across `18` mouse-years; `2014–2022` holdout, `40` pairs across `9` years)

- **Identifiability note (`power_precheck_20261114.txt`, Section 3):** Weather at Cary is shared across all six 2.25-ha plots in each year, so annual weather confounding acts on the **`18` annual mouse-year cohorts** (`df_year = 16` unadjusted, `15` with 1 weather covariate, `10` with all 6 simultaneously). Pre-analysis simulation shows 1-covariate adjustment removes weather bias (`+0.058 -> +0.154` when `b_true = 0.15, r_{WM} = -0.5, gamma_W = +0.2`, restoring `89.9%` power vs `24.3%` unadjusted) with minor SE cost (`SD 0.048` vs `0.041`), whereas entering all 6 covariates at once inflates `SD` to `0.057`.
- **Estimands & Primary Test (`E_CARY_ADJ`):**
  1. Annual correlations (`Pearson r` and `Spearman rho`, with 2-year block bootstrap 95% CIs) of `C1_dd6`..`C6_satdef` (both in larval/mouse year `t` and nymphal collection year `t+1`) with plot-mean `ln PL_grid_t` and plot-mean `ln NIP_{t+1}` (plus Cary ground-station weather columns for cross-validation).
  2. **Pre-specified primary adjusted Cary slope (`df_year = 15`):** Re-estimate R12's within-plot slope `b` of `ln NIP_{p,t+1}` on `ln PL_grid_{p,t}` (one intercept per plot, SE from 2,000 resamples of 2-year calendar blocks, seed `20261114`) adjusting for `PC1_env_{t+1}` (nymphal-year weather PC1) and separately `PC1_env_t` (larval-year weather PC1), plus each single covariate `C1`–`C6` one at a time and all 6 simultaneously (reported with its `df_year = 10` SE inflation).
  3. **Criterion:**
     - R12's P2 band (`0 <= b_adj <= 1 - pbar` within `2 SE_adj`);
     - **Confounding-shift test:** Does adjustment move `b` by more than `1 SE_unadj` (`0.058`) or make `b_adj` distinguishable from zero (`|b_adj| > 2 SE_adj`)? Repeat on the `2014–2022` holdout (`R12b`, `40` pairs) and on `ln DON_{t+1}` and `ln LPM_t` (`R12c`) as labelled exploratory checks of tick density vs infection prevalence.

### Part 4: Simulation of Weather Masking and Mimicking in `lifecycle_engine.py`

- **Design:** Using `lifecycle_engine.py` (`Engine` calibrated at criticality alongside empirical site/year simulation), run a two-level simulation (`2,000` synthetic study realizations per cell, after calibrating `D` in `Engine` with `24` antithetic pairs × `40,000` steps):
  1. **Site-level cross-site study (`N = 6` NEON sites, matching R14 tested nymph counts and `sigma_site`):**
     - Weather `Z_s` across sites and annual weather shocks `w_{s,t}` within sites have the **empirically measured mean, variance, autocorrelation, and correlation with mouse variability/abundance** from Parts 1–2, driving tick stage duration (via Wallace et al. 2019) and off-host tick survival (`Linske et al. 2019` overwinter survival sensitivity, plus questing desiccation sensitivity bounded from the data).
     - Compare **Host-variability effect absent (`kappa = 0`, `D_host = 0`)** vs **Present at half strength (`kappa = 0.5`)** vs **Present at full strength (`kappa = 1.0`, `D_host` from Part 3)**, both with and without the observed cross-site climate gradient on baseline `ln R0`.
     - Report exact frequencies with which: (a) R14's unadjusted Spearman `rho < 0` (theory's sign), (b) `rho <= -29/35` (registered "supported"), (c) `rho >= +0.4857` (as positive as observed in R14!), and (d) the `PC1_env`-adjusted test shows negative partial correlation or `p < 0.05`.
  2. **Within-site multi-year study (`T = 18` years × `6` grids, matching Cary sample sizes and nymph counts):**
     - Simulate coupled annual/sub-annual weather `W_t` and mouse abundance `H_t` at Cary's observed `(sigma_H, sigma_W, r_1, rho_{HW})` in `lifecycle_engine.py` (`Description` with two correlated drivers `'mouse'` and `'weather'`), computing both how coupled weather shifts `D` (`D_coupled - D_mouse_only`) and how often R12's slope `b` and `PC1_env`-adjusted `b_adj` are positive, inside `[0, 1 - pbar]`, or `> 2 SE` when `kappa = 0` vs `kappa = 0.5, 1.0`.

---

## 6. Accounting for Every Confounding or Modifying Process

Per `CASESTUDY_SPEC.md` ("Account for every factor you can"), every candidate process in the mouse–tick–*Borrelia* system is classified beforehand as **(a)** included in the model, **(b)** controlled for in the analysis, **(c)** bounded / tested by sensitivity analysis, or **(d)** cannot be addressed with these data:

| # | Process | Class | How `envcov` handles it |
|---|---|---|---|
| 1 | **Mean host density** (not only variance `sigma`) | **(b) & (c)** | Controlled across NEON sites via single-covariate sensitivity adjusting for site mean `Peromyscus` index (and mean acorn/mouse level across Cary plots); bounded in Part 4 cross-site simulations. |
| 2 | **Seasonality and long-term trend** | **(a) & (b)** | Daily seasonal cycle explicitly drives tick moulting and diapause in Part 3 (Wallace et al. 2019 + spring `10 °C` threshold); linear year trend (`Year`) controlled in Cary sensitivity regression to separate secular warming from interannual shocks. |
| 3 | **Host memory beyond AR(1)** (mast-driven `r_1 < 0` / AR(2)) | **(a) & (c)** | Cary and NEON show negative annual `r_1` (`-0.11` to `-0.53`), while sub-annual mouse survival creates positive intra-annual memory; Part 3 and Part 4 evaluate `phi = 0`, positive sub-annual `phi_{4.4mo} = 0.30`, and negative step memory (`phi < 0`) in `lifecycle_engine.py`. |
| 4 | **Temperature & abiotic drivers of parasite rates** | **(a) & (b)** | Core subject of study: 6 abiotic covariates controlled in Part 2, temperature-driven stage durations computed in Part 3, and coupled weather–tick survival/development simulated in Part 4. |
| 5 | **Whether host scarcity wastes or delays waiting ticks** (`contact=True` vs `False`) | **(c)** | Sensitivity analysis in `lifecycle_engine.py` comparing wasting (`contact=False`) against contact-limited questing (`contact=True`, bounded two-state host `low=None` / `low=0.2` with questing death fraction `d = mu / (mu + c)`). |
| 6 | **Shape (spread) of stage durations** (not only means) | **(a) & (c)** | Explicitly modelled in Part 3 using `Stage(shape='chain', substages=m)` matched to empirical thermal duration `CV` and `Stage(shape='fixed')` vs `shape='memoryless'` on a 6-day step. |
| 7 | **Density dependence in parasite and host** | **(b) & (c)** | Host regulation enters empirically via observed bounded mouse series; parasite saturation (`K8`: `p = 1 - exp(-f ln R0*)`) links `ln R0*` to `NIP`, and dilution on larvae per mouse (`kappa_2 = -0.17`, R12c) is included in sensitivity. |
| 8 | **Host feedback** (*Borrelia* changing mouse abundance) | **(c)** | *B. burgdorferi* causes negligible mortality/fecundity loss in *P. leucopus* (`ostfeld2006`, `dunn2013`, read in full), so exogenous host dynamics hold to a tight approximation (bounded near zero). |
| 9 | **Alternative hosts & spatial spread / immigration** | **(b) & (c)** | Eastern chipmunks (`TS_grid`) controlled in Cary sensitivity regression; alternative competent host route (`Transfer.share = 0.5`, `A = 1`) and 4-patch asynchronous dispersal (`Patches(count=4, movement=0.15)`) run in `lifecycle_engine.py` sensitivity table. |
| 10 | **Demographic noise at small parasite numbers** | **(c)** | Bounded via `Engine.extinction(..., offspring='poisson')` in `lifecycle_engine.py` at site-relevant initial sizes alongside asymptotic `D`. |
| 11 | **Observation and sampling error** (binomial `NIP` counts, trapping effort, assay shift) | **(a) & (b)** | Binomial nymphal testing (`n_tested` per site/year) explicitly simulated in `power_precheck.py` and Part 4; assay indicator (`TFA` vs `PCR`, 2007 transition) controlled in Cary sensitivity; NEON annual trapping effort (`log trap-nights`) controlled in NEON sensitivity to check R14's effort-artifact caveat. |
| 12 | **Evolution during the study** (strain turnover, host resistance) | **(d)** | Cannot be addressed: neither Cary nor NEON public files resolve *B. burgdorferi* `ospC` genotypes across years. |
| 13 | **Unmeasured cross-site confounding** (deer density, predator community, northern vs southern tick questing behaviour) | **(c) & (d)** | Tick-plot forest cover and latitude/elevation are checked in Part 2; southern vs northern nymphal questing behaviour (questing beneath vs above leaf litter at `ORNL`, Ostfeld & Brunner 2015 `PMC4342967`) and deer density at NEON sites are unmeasured at NEON and modelled as residual site variance `tau_res` in Part 4. |

---

## 7. What Counts Against the Theory vs What Cannot Be Distinguished from No Effect

- **Counts against the theory:**
  1. Across NEON sites, `PC1_env`-adjusted partial Spearman correlation is **positive with 1-sided Freedman–Lane `p < 0.05`** (meaning abiotic adjustment *strengthens* rather than explains the positive `NIP`–`sigma` association); or
  2. At Cary, adjusting for weather pushes the lagged mouse slope `b_adj` outside `[0, 1 - pbar]` beyond `2 SE_adj` (for example `b_adj + 2 SE_adj < 0`).
- **Cannot be distinguished from no effect (expected under pre-analysis power):**
  - Any `PC1_env`-adjusted partial correlation across the 6 NEON sites with 1-sided `p >= 0.05` (since power is `6%–14%` even at `kappa = 1`), and any Cary `b_adj` inside `[0, 1 - pbar]` within `2 SE_adj` whose confidence interval still includes zero or small positive elasticities (`0` to `0.15`). If abiotic covariates correlate strongly with `NIP_site` or `sigma_site` and shift R14's `+0.49` toward zero/negative without crossing `p < 0.05`, the honest verdict is **"not decided (abiotic confounding accounts for part/all of the positive unadjusted rank correlation, while 3 residual degrees of freedom lack power to resolve the predicted loss)"**.

---

## 8. Execution Budget and Stopping Rule

1. Write and run `data/fetch/envcov.py` once (resumable; < 5 minutes, < 25 MB raw data in `data/raw/envcov/`).
2. Write and run `casestudies/envcov/analysis.py` once (reproducing all Rule 7 checks first, then Parts 1–4 and the factor sensitivity table; < 8 minutes wall time), saving all printed tables and JSON/CSV files in `casestudies/envcov/results/` and SVG + pgfplots `.tex` figures in `casestudies/envcov/figures/`.
3. Write `casestudies/envcov/report.md` progressively from the saved `results/` files, verify every quoted number, and stop.

---

## Dated Protocol Amendment 1 (`2026-11-15 01:32:21 CST`, before writing or running `analysis.py`)

Before writing `analysis.py` (and before computing any environmental covariates, correlations, or thermal durations), two technical clarifications in Part 3 are added so that discrete-step artifacts are not conflated with biological duration or stay-shape effects:

1. **Discrete-step serial-chain `CV` matching:** On a discrete time step of `dt = 6 days` with mean tick stay `B = mean_days / 6` steps, a serial chain of `m` geometric sub-stages on `{1, 2, ...}` has discrete squared coefficient of variation `CV_disc^2 = 1 / m - 1 / B` (whereas `1 / m` is the continuous-time Erlang formula). When `CV` is small, `round(1 / CV^2)` can exceed `B` and clamp to `floor(B)`, artificially collapsing discrete variance toward zero. Part 3 therefore computes **both** the exact discrete-CV-matched sub-stage count `m_disc = max(1, min(round(1 / (CV^2 + 1 / B)), floor(B)))` (taken as M3b/M3c primary chain shape) **and** the clamped continuous formula `m_cont = min(round(1 / CV^2), floor(B))` (reported alongside it, plus `fixed` integer delay `round(B)`).
2. **Same-time-step vs fine-time-step separation (`R10b` principle):** Changing `dt` from `4.4 months` (`133.8 days`) to `6 days` while holding `phi = 0` per step changes the host's correlation time by a factor of `22.3` (`notes/exact-limits.md` line 48; `reports/R10b-real-systems-stay-shape.md`). To isolate how much replacing the borrowed `10`-month duration moves `D` without changing the time step, Part 3 also evaluates the site-specific thermal durations `mean(S_LN)` and `mean(M_LN)` directly on R12/R14's coarse step (`dt = 4.4 months`, `tau_mouse = 1.0`, `tau_tick = mean_stay_days / (4.4 * 30.4375)`, `phi = 0` and `phi = 0.30`), alongside the `dt = 6 days` models (`phi_{6d} = 0`, `phi_{6d} = 0.30^(6 / 133.8) = 0.9475`, and `stage_mouse` fixed vs memoryless).



## Dated Protocol Amendment 2 (2026-11-15T14:07:28Z; continuation, before environmental analysis)

The copied folder has the registered protocol, helpers and prechecks, but no
`report.md`. Their hashes and missing background paths are recorded in
`results/continuation_inventory.json`. Existing files and prior results are kept.
The supplied R12/R14 reports and engine docstring were read; unavailable context
is not claimed read. There is no Git metadata or ignore file in this copy, so
`git check-ignore` cannot verify exclusion here. No raw data are copied to reports.

Implementation clarifications, with every registered decision criterion unchanged:

- R12's report specifies 102 lagged pairs with mouse years 1994–2012 (19 calendar
  cohorts), not 18. Reproduce those exact pairs in the primary adjustment and use
  their actual calendar span for common two-year blocks (seed 20261114).
  R12b uses its original 40 pairs with at least 20 tested nymphs; its already-lagged
  mouse column must not be shifted again. Part 4's registered 18-year design is
  retained, with missingness and nymph counts sampled from the observed primary
  records. Its difference from the exact primary calendar is disclosed.
- Cary PC1 is trained on unique collection years in the primary period and applied
  unchanged to larval years and holdout. NEON's site-mean PC1 is the registered
  six-site PCA; annual NEON scores use those same scaling and loading constants.
  Individual adjustment covariates are standardized only for numerical stability.
- Daymet has 365 records even in leap years (December 31 omitted; official Daymet
  overview read). Covariates use the supplied days without interpolated weather.
  Primary thermal cohorts retain the registered numeric day-of-year definitions;
  duration days mean supplied daily records, with at most one omitted winter day
  per crossed leap year. This is a disclosed calendar approximation. Annual
  nymph cohorts use previous-year larval drop-off; no end-of-file censored cohort
  is silently included. The point estimate of active waiting is a cumulative-hazard
  threshold, not a sampled exponential wait. Thus its spread is a model-implied
  environmental/cohort spread, not a measured individual duration distribution.
- Part 4 does not identify causal weather effects from prevalence correlations.
  Data supply climate, mouse variance, autocorrelation, cross-covariance, thermal
  durations, sampling counts and observed prevalence levels. Unknown weather
  survival/ln-R0 slopes are illustrative: 0, -0.20, +0.20 per standardized climate
  unit, with residual site SD 0.17 and annual residual log-prevalence SD 0.14
  (precheck settings); ±0.10 additionally bounds weather strength where feasible.
  Thermal survival sensitivity uses exp(-mu*duration) with mu=0.001/day (Wallace
  card), checked at 0 and 0.002/day; baseline climate and direct annual-weather
  terms are bounded scenarios, not fitted causal coefficients. A weather effect
  on tick abundance need not affect prevalence; zero is included explicitly.
- In Part 4 use the observed site climate scores directly (preserving their exact
  covariance with host variability). Within-site annual driver pairs use the
  project's Environment sampler at measured annual correlations/memories; if its
  innovation covariance is infeasible, use joint two-year block resampling and
  record the departure. Engine coupled-driver loss sensitivities keep subannual
  memory assumptions separate from annual estimates. Cary prevalence elasticities
  for kappa=0, 0.5, 1 are illustrative 0, (1-pbar)/4, (1-pbar)/2 as in the existing
  helper; they are not predictions of an R0-loss model for an unmeasured epidemic
  equilibrium. Simulated slope SEs use the same two-year blocks as the field test.
- Engine calibrations retain 24 antithetic pairs and 40,000 steps. A local sparse
  matrix-product accelerator may replace only Engine._products, after agreement
  with the unmodified routine within 1e-10 on synthetic memoryless, chain, fixed,
  correlated-driver and patch examples. The engine's rates, criticality, loss root,
  coefficient and Jensen bound remain its own. Save each cell on completion.
- Negative memory is an illustrative step-level sensitivity; negative annual
  correlation has no justified fractional-power conversion. Contact-limited,
  dilution, alternative-route, patches, mouse-stay, fixed-delay and extinction
  sensitivities are conditional examples, not measured mechanisms. A single
  finite campaign uses predeclared cells in analysis.py, without outcome-driven
  extension. Allow up to 45 minutes of computation, rather than the earlier
  optimistic eight-minute combined budget; stop and report any uncompleted cells.

No new novelty claim is proposed. Existing literature cards report the earlier
read status; continuation checks are recorded separately and do not silently
upgrade inaccessible sources. Registered outcomes from earlier studies stand.


## Dated continuation note (2026-11-15T14:11:32Z; after field comparisons, before engine calibration)

The first field-phase harness stopped at an over-strict no-censoring assertion in
the recalled power-law thermal sensitivity at TREE. The primary Wallace cohorts
are complete. Keep the failure log; allow the sensitivity to report its censored
subcohorts and missing yearly mean rather than dropping the primary analysis or
changing its window. Its pooled mean is conditional on completed subcohorts.
No registered inferential criterion changes.

Direct full-text Table 1 checks clarify an earlier citation overstatement: Allen
(2025) supplies the 10 C questing threshold and 0.02/day rate for **larvae**, not
nymphs. The registered application to nymphal host finding is therefore an
illustrative cross-stage assumption; retain its pre-specified 0.04/day sensitivity.
Its March gate and hazard-one deterministic wait are imposed scenario choices.
The 6 C degree-day base is for oviposition, not a universal moulting threshold.
Keep the registered covariate unchanged and label that biological limitation.
Wallace's two maturation curves and 0.001/day engorged mortality were verified
in its full-text Table 1. These corrections do not establish any novel result.


## Dated numerical correction note (2026-11-15T14:14:03Z; after first engine campaign)

All 160 admissible planned cells completed. The stopping assertion detected a
finite-horizon bias in a periodic, both-stages-fixed cell: antithetic sampling
cancels stochastic error, but an incomplete generation at the endpoints shifts
the numerical root. The exact stationary loss in this special model is sigma²/2.
Keep all original cell records. For the two both-fixed cases differing from that
known answer by more than 1e-6, check the same seed with burn and retained length
rounded upward to whole periods (still at least 40,000 steps and 24 pairs).
Report this as a numerical correction, not a new biological campaign; use the
known asymptotic value in any interpretation of those periodic examples. No
registered empirical criterion or nonperiodic cell changes.


## Simulation implementation detail (2026-11-15T14:17:18Z; before masking outcomes)

Part 4 uses an engine-calibrated observation model, not a fitted vector epidemic
model. Additional engine cells put the measured host-weather correlation on two
transfers at the primary coarse clock with constant site-mean thermal duration.
Annual observations then use measured annual AR covariances, site-specific daily
thermal-duration lookups, and the registered sensitivity coefficients. Annual
host losses allocate the engine's incremental host loss in proportion to squared
log-mouse anomalies (expectation one); this observation bridge is illustrative.
It does not identify simultaneous seasonal development and death inside the
linear engine. Report that approximation explicitly. No NIP-weather slope is
fitted to choose a scenario. NEON uses original per-year testing counts. The
18-year Cary simulation uses the observed 1995–2012 mouse-year missingness and
assay counts, excludes zero simulated prevalences from logs as R12 does, and
uses 2,000 common two-year block resamples in each synthetic study. The actual
field comparison retains its original 19-cohort calendar. Midranks handle ties;
report both the pre-specified rho cutoff and exact unadjusted permutation rule.
All event rates include counts and Monte Carlo uncertainty.


## Coupled-development completion (2026-11-15T14:21:27Z; before coupled-development outcomes)

The constant-duration engine bridge above does not fully carry out the registered
request for jointly changing development and survival. Complete that part with
`driven_weather.py`, retaining the bridge outputs as a sensitivity approximation.
This is implementation completion, not a changed field test. Each annual weather
state chooses its daily-model duration and stage mortality; each conditional
matrix is built by the project Engine, and its loss root/products are reused.
The arithmetic-mean baseline includes host-weather covariance analytically through
truncated Gaussian moments, rather than a product of mean rates. The Gaussian
annual host and weather pair has the observed correlation and AR memories, repeated
over 61 six-day steps; this subannual path is illustrative. Mortality is
1-exp(-0.001*6*exp(-gamma*climate)), gamma=-0.2,0,+0.2. Temperature changes stay
means in every scenario. Run six sites × three gamma × three kappa (54 cells),
then 2,000 synthetic studies per gamma/kappa with original testing counts and
residual site SD 0.17. Record 24-pair, 40,000-step loss error and exact frequency
counts. No parameter is chosen by the previous masking frequencies.
