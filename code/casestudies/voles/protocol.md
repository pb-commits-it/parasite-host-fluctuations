# Protocol: Kielder Forest field voles and comparable cyclic small mammals (`voles`)

**Registered before downloading or opening any empirical data rows**: 2026-11-14 23:01:45 CST (2026-11-15 05:01:45 UTC).  
**Pre-data identifiability and power script**: `casestudies/voles/protocol_power.py`, saved output `casestudies/voles/results/pre_data_power.json` and `pre_data_power_output.txt` (four earlier harness runs during tolerance/speed tuning preserved as `pre_data_power_attempt1_output.txt` through `attempt5_output.txt`; see Note on pre-data harness runs at the end of Section 5).

---

## 1. Question

The project's analytical theory derives its sharpest results for hosts that fluctuate independently from step to step or with simple first-order autoregressive (`AR(1)`) geometric memory (`notes/exact-limits.md`, `../paper/record/section_ring.html`), with extensions to Gaussian linear processes and quasi-cycles (`Statement ring-B8a`, `Statement ring-B8b`, `Statement ring-B10-Gpp` in `notes/error-bound-general.md` §13) and to dynamic endemic prevalence tracking under host fluctuations (`K9` in `notes/coupled-hosts.md`).

Using openly deposited longitudinal field series of cyclic small mammals sampled at approximately a 28-day step—primarily Kielder Forest field voles (*Microtus agrestis*) and their microparasites (cowpox virus, *Bartonella* spp., *Babesia microti*) alongside cyclic Finnish bank voles (*Myodes glareolus*) with Puumala hantavirus (PUUV)—this study asks three questions:

1. **Host memory regime at the 28-day step**: What are the within-site variance, autocorrelation function (ACF), and spectral density of log host abundance at the 28-day step, and is host memory adequately described by `AR(1)`, or does it require `AR(2)` quasi-cycles, annual seasonality plus `AR(1)`, or annual seasonality plus `AR(2)`?
2. **Parasite tracking, lag, and attenuation**: Does the project's linearised loop response (`K9` in `notes/coupled-hosts.md`, coupled with the residence-time kernel of the second-order theory) predict the lag and amplitude attenuation with which observed pathogen prevalence tracks host log-density across lags `0` to `12` steps (28-day units), and do pathogens with longer vector/chronic residence stages lag further behind host abundance than pathogens with shorter residence times?
3. **Empirical inputs for the memory-extension study**: What fitted 28-day host-memory regimes (log-density variance, ACF at lags `0..12`, parametric model coefficients, and second-order / finite-noise `R0` losses computed via `studies/engine-R16/lifecycle_engine.py` and the spectral formula `ring-B8b`) do these series supply to `casestudies/voles/results/memory_regimes.json`?

**Scope warning fixed in advance**: Observational time series of a fluctuating host and its endemic pathogens **cannot directly test H1** (that host fluctuation lowers `ln R0` relative to a constant-mean-host baseline), because `R0` is unmeasured and no control population holds host density constant at its arithmetic mean (`CASESTUDY_SPEC.md`; `reports/R12-cary-single-population.md`; `reports/R14-neon-across-sites.md`). What these series *can* directly test is: (i) whether `AR(1)` suffices for real vole memory (`H_mem`), and (ii) the `K9` frequency-response predictions for delayed prevalence tracking and between-pathogen lag ordering (`H_lag`, `H_order`).

---

## 2. Theoretical Predictions and Exact Formulas

All rates below are expressed per **28-day step** (`Delta t = 28` days = `1` step; `1` solar year = `365.25 / 28 = 13.0446` steps, angular frequency `omega_yr = 2*pi / 13.0446 = 0.4817` rad/step; a 3.2-year multiannual vole cycle has `P_cyc ~ 42` steps, `omega_cyc ~ 0.150` rad/step).

### 2.1 Host memory and spectral cost (`ring-B8a`, `ring-B8b`)
Let `z_H(t) = ln H(t) - E[ln H]` be centred log host abundance with stationary variance `sigma_H^2`, autocorrelation `rho_H(ell)` at integer 28-day lag `ell`, and normalised one-sided spectral density `f_H(omega)` on `(0, pi)` satisfying `(1/pi) int_0^pi f_H(omega) d omega = 1`. For a primitive discrete-time ring with threshold lineage stays `tau_k >= 1` (`m_k = 1 - 1/tau_k`, `T_0 = sum_k tau_k`) and one transfer scaling as `H(t)^eta`, Statement `ring-B8b` (`notes/continuous-time.md` §5; `notes/exact-limits.md` H4) gives the second-order loss of `ln R0` per circuit:

    Gamma_0 = T_0 * eta^2 * sigma_H^2 * (1 / pi) * int_0^pi f_H(omega) K(omega) d omega,
    K(omega) = -(1 / T_0) * Re[ g(omega) / (1 - g(omega)) ],
    g(omega) = prod_k [ (1 - m_k) e^{i*omega} / (1 - m_k e^{i*omega}) ].

For `AR(1)` geometric memory `rho_H(ell) = phi^ell` (`0 < phi < 1`, `theta = phi / (1 - phi)`), `Gamma_0` reduces to Statement `ring-B2`:

    Gamma_0 = eta^2 * sigma_H^2 * [ (1 + 2*theta) / (2 * T_0) - R / (1 - R) ],    R = prod_k 1 / (1 + tau_k / theta).

When `f_H(omega)` concentrates variance in both an annual peak (`P ~ 13` steps) and a multiannual peak (`P ~ 39-42` steps), `ring-B8b` predicts that `Gamma_0` differs systematically from a plug-in `AR(1)` match using only lag-1 correlation `rho_H(1)`. Finite-amplitude `AR(1)` benchmark losses `D` and exact edge bounds `J` are computed via `Engine.loss()` and `Engine.coefficient()` in `studies/engine-R16/lifecycle_engine.py` (noting the two caveats in `verification/engine-R16-check.md`: we use `kind='lognormal'` or `low=None` rather than a fixed raw `low=0.05` when comparing `D` to `D2`).

### 2.2 Dynamic prevalence response to fluctuating host density (`K9` in `notes/coupled-hosts.md`)
Consider an endemic parasite at stable equilibrium prevalence `p^*` (`0 < p^* < 1`) in host population `N(t) = N^* exp(z_H(t))`.

1. **Vector-borne / two-block `S + L` cycle (*Bartonella* via fleas; *Babesia* via *Ixodes trianguliceps* ticks)**:  
   In block `S` (`dI/dt = u(t)(N(t) - I(t)) - a_H I(t)`), `a_H` is the host clearance-plus-mortality rate per 28-day step, and `d_H = a_H / (1 - p^*)` (`E_S(s) = a_H / (s + d_H)`). In unregulated vector block `L` (`dV/dt = c_V I(t) N(t)^{eta_VH - 1} - a_V V(t)`), `a_V` is the vector-stage exit rate per 28-day step (`E_V(s) = a_V / (s + a_V)`), and per-susceptible force of infection is `u(t) = beta_V V(t) N(t)^{eta_HV - 1}` with net cycle host-density elasticity `eta_H = eta_VH + eta_HV - 1 > 0`. Linearising around `(N^*, I^* = p^* N^*, V^*)` gives the exact `K9` frequency response from `z_H(i*omega)` to log infected count `y_I(i*omega)` and to log active prevalence `y_p(i*omega) = ln(I(t)/N(t)) - ln p^* = y_I - z_H`:

       H_I(i*omega) = E_S(i*omega) * [ E_V(i*omega)*(eta_VH - 1) + (eta_HV - 1) + 1/(1 - p^*) ] / [ 1 - E_S(i*omega)*E_V(i*omega) ],
       H_p(i*omega) = H_I(i*omega) - 1.

2. **Direct transmission (`a_V -> infinity`, `E_V -> 1`) and serological assays (cowpox IF antibody; PUUV IgG)**:  
   Setting `E_V(i*omega) = 1` gives the direct-transmission active-prevalence filter:

       H_p(i*omega) = (a_H * eta_H - i*omega) / (i*omega + u^*),    u^* = a_H * p^* / (1 - p^*).

   When the field assay measures **seroprevalence** `M/N = (I + R)/N` for an infection conferring lifelong immunity (cowpox immunofluorescence in Kielder; `Smith et al. 2009`; `Burthe et al. 2008`) with recovery `gamma_I = a_H - mu_R` and host demographic turnover `mu_R` (`dI/dt = u(N - I - R) - a_H I`, `dR/dt = gamma_I I - mu_R R`, equilibrium seroprevalence `p_sero^*`), exact linearisation yields:

       H_{I,sir}(i*omega) = [ a_H*(eta_H - 1) + a_H / (1 - p_sero^*) ] / [ i*omega + u_sir + u_sir * gamma_I / (i*omega + mu_R) ],
       H_{sero}(i*omega)  = [ mu_R * (i*omega + a_H) / (a_H * (i*omega + mu_R)) ] * H_{I,sir}(i*omega) - 1,

   where `u_sir = mu_R * p_sero^* / (1 - p_sero^*)`.

3. **Two universal qualitative features of `H_p(i*omega)`**:
   - **Slow fluctuations (`omega -> 0`)**: `H_p(0) = eta_H * (1 - p^*) / p^* > 0` (positive tracking bounded by `(1 - p^*)/p^*` times `eta_H`, directly reflecting `K3` and `K8`).
   - **Fast fluctuations (`omega -> infinity`, e.g. rapid spring juvenile recruitment)**: `H_I(i*omega) -> 0`, so `H_p(i*omega) -> -1`! A sudden surge in host numbers dilutes prevalence `I/N` instantaneously (`CCF(0) <= 0` or small) before transmission and vector residence convert susceptible recruits into new infections at positive lags `ell > 0`.
   - **Cross-correlation prediction**: By Wiener-Khinchin, for host spectrum `S_z(omega)`,

         Cov(y_p(t + ell), z_H(t)) = (1 / pi) * int_0^inf Re[ H_p(i*omega) e^{i*omega*ell} ] S_z(omega) d omega.

### 2.3 Literature parameterisations fixed prior to seeing data (`protocol_power.py`)

All values below are cited literature inputs, never fitted to the data:

| Pathogen & system | Assay in open data | Host exit `a_H` (`/28d`) | Vector / turnover stage (`/28d`) | Mean prev `p^*` | Host elasticity `eta_H` | Predicted `CCF(0)` | Predicted peak lag (28d steps) | Predicted `SD(ln p) / SD(ln H)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Cowpox** (Kielder *M. agrestis*) — if active infection | — | `1.00` (`~28 d`; Smith et al. 2009) | direct (`E_V=1`) | `0.20` | `0.38` (`1 - q`, `q=0.62`, Smith et al. 2009) | `+0.78` | **`1` step** (0.9 mo) | `0.95` |
| **Cowpox** (Kielder *M. agrestis*) — **seroprevalence (actual assay)** | IF antibody (`I+R`) | `1.00` | `mu_R = 0.35` (`~80 d` vole turnover; Burthe et al. 2008) | `0.50` (`p_sero^*`) | `0.38` (`[0.26, 0.51]`) | `+0.32` (finite-window MC: `-0.07`) | **`3` steps** (IQR `2-4`) | `0.76` |
| ***Bartonella* spp.** (Kielder *M. agrestis*) | Blood PCR / qPCR (`RQ >= 1`, Taylor et al. 2023) | `0.70` (`~40 d` bacteraemia; Telfer et al. 2007a, 2010) | Flea `a_V = 0.45` (`~62 d` adult+nest flea; Telfer et al. 2007a) | `0.45` | `0.50` (`[0.25, 1.0]`) | `-0.09` (MC: `-0.14`) | **`4-5` steps** (MC median `4-7`) | `0.64` |
| ***Babesia microti*** (Kielder *M. agrestis*) | Blood PCR / qPCR (`RQ >= 1`, Taylor et al. 2023) | `0.35` (chronic; vole turnover `~80 d`, Bown et al. 2008) | Tick `a_V = 0.12` (`~8.3` steps *I. trianguliceps* stage; Randolph 1975, Bown et al. 2008) | `0.35` | `0.50` (`[0.25, 1.0]`) | `-0.58` (MC: `-0.54`) | **`6-8` steps** (MC median `8`) | `0.72` |
| **Puumala virus (PUUV)** (Konnevesi *M. glareolus*) | IgG antibody (lifelong chronic infection) | `0.30` (vole turnover `~93 d`; Voutilainen et al. 2016) | Shedding `a_V = 2.0` (`~14 d` env. survival; Kallio et al. 2006) | `0.25` | `0.70` (`[0.40, 1.0]`) | `-0.05` (MC: `-0.22`) | **`5-7` steps** (MC median `6`) | `1.41` (MC: `1.02`) |

---

## 3. Data and Provenance (Openness Audit Before Analysis)

Before writing this protocol, open repositories (Dryad, Zenodo, Figshare, UKCEH EIDC, KNB/DataONE, EuropePMC) were searched for every system named in the study brief (`study rules` Rule 2 & `CASESTUDY_SPEC.md` Rule 2):

1. **Kielder Forest 2001–2007 longitudinal series (Begon, Telfer, Smith, Carslake, Burthe)**:
   - KNB record `doi:10.5063/AA/knb.249.1` (*"Field voles and microparasite data, May 2001 – March 2007, 4 sites, 14,075 captures"*) is **metadata-only** with access constraint *"Obtain permission from data set owner(s)"* and zero data files attached. Neither `Telfer et al. (2007a, Parasitology)`, `Smith et al. (2009, PNAS)`, nor `Telfer et al. (2010, Science)` deposited raw population/prevalence tables in journal supplements or Dryad. **Cannot be used (not open).**
2. **Kielder Forest 2015–2017 longitudinal blood-pathogen & trapping dataset — OPEN (Primary Kielder dataset `K15`)**:
   - **Deposit**: Paterson, S., Jackson, J., Jackson, I., Bradley, J., Begon, M., Wanelik, K., & Taylor, C. (2022). *Data on immunological expression and phenotypes in a natural population of field voles in Kielder Forest, UK 2015-2017*. NERC EDS Environmental Information Data Centre. `https://doi.org/10.5285/e5854431-6fa4-4ff0-aa02-3de68763c952`.
   - **Licence**: UK Open Government Licence v3.0 (open, unrestricted reuse with citation).
   - **Contents (from `vole_summary.docx` and column header only)**: `long_blood_qpcr_fv.2.csv` (17,882 capture records across 46 biweekly sessions, March–October 2015–2017, 4 concurrent clear-cut grids out of 7 total sites, 150–197 Ugglan traps per grid; records `site`, `dateCapt`, `trapsWorking`, `gridTreat`, `treat`, `frontline`, `profender`, `species`, `chip`, `flea`, `tick`, and qPCR relative quantification `BAR` for *Bartonella* spp. and `BAB` for *Babesia microti*, where `Taylor et al. 2023, Mol. Ecol.` defines binary infection presence as `RQ >= 1.0`).
   - **Important design feature to control**: Some grids/individuals received antiparasitic treatments (`gridTreat`, `frontline`, `profender`). Our primary unperturbed analysis restricts to untreated control grids/animals (`gridTreat == 'C'` or untreated captures if all grids had a control subset—the exact structure of `gridTreat` will be tabulated from distinct codes first, and treated vs control grids will be separated; see Section 4).
3. **Kielder Forest 2008–2010 longitudinal & cross-sectional dataset — OPEN (Secondary Kielder dataset `K08`, includes cowpox, *Bartonella*, *Babesia*, fleas, ticks)**:
   - **Deposit**: Jackson, J. A. et al. (2014). *Data from: An immunological marker of tolerance to infection in wild rodents*. Dryad `https://doi.org/10.5061/dryad.bk537`, fetched from official Zenodo `dryad` community mirror record `4971362` (`Longitudinal database.xls`, `Cross-sectional database.xls`).
   - **Licence**: CC0 1.0.
   - **Companion deposit**: Arriero, E., Wanelik, K. M., Birtles, R. J., Bradley, J. E., Jackson, J. A., Paterson, S., & Begon, M. (2018). Dryad `https://doi.org/10.5061/dryad.110pk`, Zenodo mirror `4941512` (`vole_immunological_profiles_data.xlsx`, CC0 1.0).
   - **Contents (from `PLOS Biol.` Methods & BIFF8 header only)**: 4 Kielder clear-cut sites (2 in Feb 2008–Mar 2009, 2 in Apr 2009–Mar 2010), 150-trap grids trapped monthly (every ~28 days), with individual transponder `Tag ID`, `Date`, `Site`, `Trapping Session`, `Bartonella`, `Cowpox`, `Babesia`, `Fleas`, `Ticks`.
4. **Sin Nombre virus in Montana deer mice (*Peromyscus maniculatus*, Luis et al. 2015) — CHECKED, RAW SERIES NOT DEPOSITED**:
   - Figshare collection `https://doi.org/10.6084/m9.figshare.c.3307761.v1` (article `3562431`, CC0) holds `appendix-A.htm` (`Ecological Archives E096-151-A1`), which contains raster figures (`Fig. A1–A6`) and captions only (reporting Cascade peak `CCF = 0.348` at lag `16` months), **not** the underlying monthly capture or seroprevalence numbers. Sevilleta LTER / EDI PASTA packages return HTTP 403 to anonymous scripts (`INVENTORY2.md`). Reported honestly as unavailable in raw tabular form.
5. **Comparable cyclic small-mammal series — Finnish bank voles (*Myodes glareolus*) and Puumala hantavirus (`F02`) — OPEN**:
   - **Deposit**: Voutilainen, L., Kallio, E. R., Niemimaa, J., Vapalahti, O., & Henttonen, H. (2016). *Data from: Temporal dynamics of Puumala hantavirus infection in cyclic populations of bank voles*. Dryad `https://doi.org/10.5061/dryad.g8140`, fetched from official Zenodo `dryad` mirror `4941743` (`PUUV prevalence per trapping session.xls`, `Seroconversion rate per trapping session.xls`). **Licence**: CC0 1.0.
   - **Contents (from `Key` sheet only)**: 60 trapping sessions across 7 years (April 2002 – May 2009), spanning three complete 3-year vole cycles at Konnevesi, central Finland (1 large 246-station CMR `core` grid + 15 satellite sites `A–P`), giving `Total N`, `trapnights`, `PUUV tested`, `Infected.P`, `PUUV prevalence`, plus core-site seroconversion intervals (`17–166` days).
   - **Multi-area spatial companion (`F23`)**: Wang et al. (2023), Dryad `https://doi.org/10.5061/dryad.bg79cnpfh`, Zenodo mirror `7796298` (CC0 1.0; 30 areas × up to 5 annual points; used only as a multi-site annual variance comparator, not for 28-day memory).

---

## 4. Operational Definitions, Estimands, and Registered Tests

To place all series on a common **28-day step** (`Delta t = 28` days) without interpolating missing winter sessions:

1. **Binning to 28-day steps**:
   - For each dataset, define integer 28-day step index `k = floor((date - t_origin) / 28 days)` anchored at the first trapping date of that study (`K15`, `K08`, `F02` separately).
   - In `K15` (biweekly trapping, ~14-day intervals), two consecutive biweekly sessions fall into one 28-day primary window. Within each `(site, 28-day step k)`, **host abundance index `H_{s,k}`** is the number of distinct *Microtus agrestis* individuals (`species == 'FV'`, distinct `chip`, or distinct row if untagged dead/juvenile) caught per `100` working trap-checks (or if `trapsWorking` is missing/zero on a row, distinct tagged *M. agrestis* caught in that 28-day window divided by the number of trapping sessions in that window, normalised to per-session MNA/captures; both raw distinct-vole count per session and effort-normalised captures per 100 trap-checks will be computed, with effort-normalised captures per 100 working traps as primary where `trapsWorking > 0` on >=90% of rows, otherwise distinct voles per trapping session).
   - **Antiparasitic treatment rule in `K15` (fixed before opening rows)**: If `gridTreat` distinguishes control grids from treated grids, primary pathogen prevalence uses only untreated observations (`frontline` and `profender` zero/blank/FALSE, and `treat` not TRUE); we also compute host density both on control grids only and across all grids (since spot-on ivermectin/fipronil treats individual ectoparasites/nematodes rather than removing voles, and sensitivity analysis checks whether `gridTreat` alters host variance or pathogen CCF).
   - In `K08` (monthly ~28-day trapping), each primary `(Site, Year, Trapping Session)` maps to its mean capture date's 28-day step `k`; `H_{s,k}` is distinct *M. agrestis* individuals captured in that session on the longitudinal grid.
   - In `F02` (Konnevesi bank voles), `H_{core,k}` is `Total N` per `100` `trapnights` (and raw `Total N` as sensitivity) on the `core` CMR grid at `k = round((Date - Date_0) / 28 days)` (if two sessions ever round to the same 28-day index, counts/tests are pooled; gaps remain `NaN`). Satellite sites `A–P` (only 9 traps each) are pooled into one combined `satellite_pooled` series for sensitivity and kept separate from `core`.
2. **Log-transformation and minimum sample rules**:
   - Host log-density: `z_{s,k} = ln(H_{s,k})` for `H_{s,k} > 0`. Zero host counts are **excluded from `ln H` and recorded explicitly** (sensitivity check: `ln(H + 0.5 * min_pos)`). A site enters the 28-day memory analysis if it has **at least 12 non-zero 28-day observations** spanning at least 18 steps (`K15` long-running grids and `F02` core; `K08` site-years with 8–10 monthly sessions are analysed both per site-year and pooled with site-year mean centering).
   - Pathogen prevalence in `(s, k)`: `p_{s,k} = pos_{s,k} / n_{s,k}` for windows with **`n_{s,k} >= 8` tested animals** (sensitivity: `n >= 5` and `n >= 15`). For log-prevalence cross-correlations (matching `K9`'s `y_p = ln p`), use Haldane-Anscombe continuity-corrected `y_{s,k} = ln((pos_{s,k} + 0.5) / (n_{s,k} + 1.0))` so occasional sample zeros (`0/n`) are not dropped by selection bias; raw Pearson CCF on untransformed `p_{s,k}` vs `ln H_{s,k}` is reported alongside as a check.
3. **Host memory estimands (Goal 1)**:
   - Within-site variance `sigma_H^2` and SD `sigma_H` of `z_{s,k}` (`ddof = 1`).
   - Empirical ACF `rho_H(ell)` at 28-day lags `ell = 1..12` over observed pairs `(k, k + ell)` (requiring at least 6 pairs at lag `ell`), plus Lomb-Scargle / Bartlett periodogram across frequencies `(0, pi)` rad/step.
   - Exact Gaussian maximum-likelihood comparison on the actual (gapped) observation times `t_{s,i}` (in 28-day steps) across five models (`protocol_power.py`):
     - `M0`: White noise (`k = 2` parameters per site: mean + variance)
     - `M1`: Continuous/discrete `AR(1)` (`Cov(dt) = sigma^2 phi^{|dt|}`, `k = 3`)
     - `M2`: Damped oscillatory `CAR(2)/AR(2)` quasi-cycle (`Cov(dt) = sigma^2 [(1-c)*e^{-zeta|dt|}(cos(om_0|dt|) + (zeta/om_0)sin(om_0|dt|)) + c*I]`, `k = 5`)
     - `M3`: Annual seasonal harmonic (`cos` + `sin` at `P_yr = 13.0446` steps) + `AR(1)` (`k = 5`)
     - `M4`: Annual seasonal harmonic + damped oscillatory `CAR(2)/AR(2)` (`k = 7`).
4. **Registered Tests and Exact Decision Criteria (`CASESTUDY_SPEC.md` Rules 1 & 4)**:
   - **Test `T1_mem` (Host memory beyond `AR(1)`)**: On each multi-year series (`F02` core 7-yr series; `K15` pooled multi-grid Gaussian likelihood with site-specific means and shared covariance parameters, plus each individual `K15` grid with `>= 15` windows), compare `min(AICc(M2), AICc(M3), AICc(M4))` against `min(AICc(M0), AICc(M1))`.
     - *Supports structured memory beyond `AR(1)`*: `Delta AICc = min(M0, M1) - min(M2, M3, M4) >= 4.0`.
     - *Supports simple `AR(1)` / memoryless*: `Delta AICc <= -2.0` (i.e. `M0` or `M1` wins by at least 2 AICc units).
     - *Not decided*: `-2.0 < Delta AICc < 4.0`.
   - **Test `T2_track` (`K9` delayed positive tracking vs independent null)**: For each primary host-pathogen series with sufficient power (`F02` PUUV on Konnevesi core; `K15` *Bartonella* across Kielder grids; `K15` *Babesia* across Kielder grids; and `K08` cowpox and *Bartonella* reported with their lower power), compute pooled within-site `CCF(ell) = Corr(y_{s, t+ell}, z_{s, t})` for `ell = 0..12` steps, and test `S_pos = max_{ell in 1..6} CCF(ell)` and `S_delay = max_{ell in 1..6} CCF(ell) - CCF(0)` via circular/block phase-randomisation (surrogate host series preserving exact per-site sample locations and host autocorrelation spectrum, `2,000` surrogates, seed `20261114`).
     - *Supports `K9` delayed tracking*: `S_pos > 0` with one-sided surrogate `p < 0.05` **and** observed peak positive lag `ell^* = argmax_{0..8} CCF(ell)` falls within `+-2` steps of the `K9` pre-data interquartile range (`PUUV`: `[3, 8]`; `*Bartonella*`: `[2, 8]`; `*Babesia*`: `[5, 10]`; `cowpox sero`: `[1, 5]`).
     - *Contradicts `K9`*: `CCF(ell) < 0` across all `ell in 1..6` with two-sided surrogate `p < 0.05`, **or** `S_pos` is significant (`p < 0.05`) at `ell^* = 0` when `K9` predicts `CCF(0) < 0` and `ell^* >= 4` (e.g. *Babesia*).
     - *Not decided*: surrogate `p >= 0.05` for `S_pos`.
   - **Test `T3_order` (Residence-time lag ordering between co-occurring Kielder pathogens)**: On the same Kielder vole populations (`K15` primary; `K08` secondary), compare the positive-CCF centre-of-mass lag `L_com = sum_{ell=0}^8 ell * max(CCF(ell), 0) / sum_{ell=0}^8 max(CCF(ell), 0)` between tick-borne chronic *Babesia microti* (`T_0 ~ 11.2` steps) and flea-borne shorter-stay *Bartonella* spp. (`T_0 ~ 3.7` steps), using `2,000` within-site block-bootstrap resamples of 28-day windows.
     - *Supports residence-time lag ordering*: `L_com(Babesia) - L_com(Bartonella) > 0` with `95%` bootstrap CI excluding `<= 0` (one-sided bootstrap `p < 0.05`).
     - *Contradicts*: `L_com(Babesia) - L_com(Bartonella) < 0` with one-sided bootstrap `p < 0.05`.
     - *Not decided*: bootstrap `95%` CI straddles `0`.
   - **Descriptive check `D_atten` (Prevalence amplitude vs host amplitude)**: Report raw and binomial-sampling-corrected `SD(ln p) / SD(ln H)` alongside `K9`'s predicted ratio, explicitly noting the `R12`/`R14` caveat (`10:39` note) that vector abundance and seasonal contact also move infection pressure, so `SD(ln p) < SD(ln H)` is a descriptive check and **not** a standalone proof or refutation of `K3`.

---

## 5. Pre-Data Power and Identifiability Analysis (`protocol_power.py`)

Computed **before** downloading or inspecting any data rows (`results/pre_data_power.json`, runtime `49.4` s):

### 5.1 Identifiability of host memory models (`Delta AICc >= 2`)
| Study design (sample size & span) | True `iid` -> `M0` win | True `AR(1)` -> `M1` win (false `>AR1` rate) | True `AR(2)` cycle -> power to beat `AR(1)` | True `Seasonal + AR(2)` -> power to beat `AR(1)` | Identifiability verdict |
|---|---:|---:|---:|---:|---|
| Benchmark (`N=300` contiguous steps) | `65%` | `67.5%` (`15%`) | **`100%`** (`90% M2`, `10% M4`) | **`100%`** (`100% M4`) | Separates all 4 regimes cleanly |
| Konnevesi core (`N=60` over `88` steps, `7 yr`) | `90%` | `65.0%` (`15%`) | **`80.0%`** (`65% M2`, `20% M4`) | **`100%`** (`95% M4`, `5% M3`) | **High power (`80–100%`)** to detect both seasonality and multiannual cycles |
| Kielder `K15` single grid (`N=24` Mar–Oct over `34` steps, `3 yr`) | `100%` | `65.0%` (`17.5%`) | **`42.5%`** (`45%` absorbed into `AR(1)`) | **`100%`** (`52.5% M3`, `47.5% M4`) | **High power (`100%`)** to beat `AR(1)` via seasonality, **low single-grid power (`42.5%`)** to separate multiannual `AR(2)` from `AR(1)` over `2.6 yr` |
| Kielder `K08` single site-year (`N=9` over `1 yr`) | `100%` | `12.5%` (`0%`) | **`2.5%`** | **`15.0%`** | **Unidentifiable (`<= 15%` power)** on a 1-year series; cannot separate memory models |

*Implication fixed before seeing data*: A single 1-year (`K08`) series **cannot** identify host memory beyond `AR(1)` (`15%` power), and a 3-season (`K15`) series cleanly detects seasonal+AR memory over pure `AR(1)` (`100%` power) but splits `52.5% / 47.5%` between `Seasonal + AR(1)` and `Seasonal + AR(2)` because `2.6` years covers less than one full `3–4` year vole cycle. The 7-year Konnevesi `F02` series (`N=60`) has `95%` power to identify `Seasonal + AR(2)` specifically.

### 5.2 Power of `K9` cross-correlation and lag-ordering tests (at `alpha = 0.05` against independent AR(1)-prevalence null)
| Scenario (pre-data literature inputs + binomial testing error) | Sites × active windows (`n_test`/obs) | Null 95% cutoff (`S_pos`) | Power (`S_pos`: lagged positive tracking) | Power (`S_delay`: lagged > lag-0) | Simulated `K9` mean `CCF(0)` -> `max CCF(1..6)` | Simulated `K9` median peak lag (IQR) |
|---|---|---:|---:|---:|---:|---:|
| **Konnevesi `F02` PUUV** (`core`) | `1 × 60` (`n=50`) | `0.429` | **`98.0%`** | **`99.3%`** | `-0.22 -> +0.75` | `6.0` steps (`5–6`) |
| **Kielder `K15` *Bartonella*** | `4 × 24` (`n=20`) | `0.350` | **`97.7%`** | **`99.0%`** | `-0.14 -> +0.57` | `7.0` steps (`4–8`) |
| **Kielder `K15` *Babesia*** | `4 × 24` (`n=20`) | `0.352` | **`58.0%`** | **`100.0%`** | `-0.54 -> +0.38` | `8.0` steps (`7–8`) |
| **Kielder `K08` Cowpox serology** | `4 × 9` (`n=25`) | `0.535` | **`79.3%`** | **`40.7%`** | `-0.07 -> +0.62` | `3.0` steps (`2–4`) |
| **Kielder `K08` *Bartonella*** | `4 × 9` (`n=25`) | `0.546` | **`66.7%`** | **`80.7%`** | `-0.40 -> +0.59` | `4.0` steps (`4–5`) |

- **Pairwise lag-ordering power in Kielder `K15` (*Babesia* vs *Bartonella* on shared host trajectories)**: Discrete `argmax` lag is strictly greater for *Babesia* in `45.3%` of trials (`>=` in `84.3%`, because winter gaps discretise observed lags), whereas the **centre-of-mass lag `L_com`** (`T3_order`) orders `L_com(Babesia) > L_com(Bartonella)` in **`96.0%`** of trials (mean difference `+0.99` steps). That is why `L_com` (rather than discrete `argmax` alone) is registered for `T3_order`.
- **What the study cannot distinguish from no effect**: In `K08` (`4 × 9` monthly points, within-year spans), lags beyond `6–8` steps have almost no pairs, and moderate elasticities (`eta_H <= 0.25`) have `< 40%` power. And in *all* observational series, zero direct causal contrast in `ln R0` (`H1`) is identifiable without an experimental constant-host control.

*Note on pre-data harness runs*: During development of `protocol_power.py` (before any empirical data were downloaded), `attempt1` failed Check 2 (`rel_err = 1.7e-3 > 1e-4`) because `omega_max = 200` truncated the `O(omega^{-2})` OU tail (fixed by exact `tan(u)` quadrature on `[0, inf)`), `attempt2` failed Check 3 (`max_z = 5.3 > 3.5`) because forward Euler with `dt = 0.05` recorded `y_p` one sub-step after `z_H` (fixed by recording simultaneous state + Heun step), and `attempt3–4` set Check 4's nested `AICc` win threshold for `AR(1)` to `0.80` overlooking Wilks's asymptotic false-split rate (`P(chi^2_1 > 2) = 0.157` against `M2` plus `P(chi^2_2 > 4) = 0.135` against `M3`, theoretical `AR(1)` win rate `~0.71`; threshold set to `0.65`, observed `0.675`). All four failed outputs are preserved in `results/`.

---

## 6. Accounting for Every Confounding and Modifying Factor (`CASESTUDY_SPEC.md`)

Every biological and statistical process listed in `CASESTUDY_SPEC.md` is classified as **(a) included in the model**, **(b) controlled in the analysis**, **(c) bounded by sensitivity analysis**, or **(d) cannot be addressed with these data**:

| # | Factor / Process | Disposition | How handled in `analysis.py` / `lifecycle_engine.py` |
|---|---|---|---|
| 1 | **Mean host density (not only variance)** | **(b) controlled & (c) bounded** | Within-site centring removes between-site mean density differences in `CCF` and `ACF`; equilibrium prevalence `p^*` (which encodes mean `R0(N^*)` via `K8`: `1 - p^* = 1/R0`) enters `K9` explicitly and is varied across observed site means in sensitivity. |
| 2 | **Seasonality and long-term trend** | **(a) included & (b) controlled** | Annual seasonality (`P = 13.045` steps) is explicitly modelled in `M3/M4` and in the `K9` forcing spectrum; sensitivity analysis also computes **seasonally anomalies (detrended & deseasonalised)** `CCF` to test whether lagged pathogen tracking persists after removing common calendar-month seasonality! Linear trend is tested via detrending sensitivity. |
| 3 | **Host memory beyond `AR(1)`** | **(a) included** | Central focus of Goal 1: `AR(2)` / `CAR(2)` multiannual quasi-cycles and `Seasonal + AR(2)` spectra are fitted directly and propagated into `ring-B8b` spectral loss and `K9` `CCF` predictions. |
| 4 | **Temperature and abiotic drivers of parasite rates** | **(b) controlled via seasonal residual & (c) bounded** | Kielder ectoparasite activity (fleas, *I. trianguliceps* nymphs) and cowpox survival vary seasonally (`Smith et al. 2009`; `Telfer et al. 2007a`; `Bown et al. 2008`). Because temperature/photoperiod share the `13`-step annual frequency with vole breeding, raw `CCF` mixes host-density forcing with seasonal vector forcing; computing **within-month / seasonal-anomaly `CCF`** controls for calendar-date abiotic forcing, and thermal/seasonal staying modulation (`C3c`) is bounded in sensitivity. |
| 5 | **Whether host scarcity wastes or delays waiting stages (`contact=True` vs `False`)** | **(c) bounded via `lifecycle_engine.py`** | Free-living / questing fleas in burrows and *I. trianguliceps* ticks in nests wait for hosts (`contact-limited` transfer, `Statement C1b/C1i`). Sensitivity runs in `lifecycle_engine.py` compare wasting (`contact=False`) vs contact-limited waiting (`contact=True`, waiting mortality fraction `d in {0.25, 0.5, 0.75}`) on bounded host drivers matched to vole `sigma_H` and `phi`. |
| 6 | **Shape (spread) of stage durations (`memoryless` vs `chain` vs `fixed`)** | **(c) bounded via `lifecycle_engine.py`** | Cowpox has a narrow ~28-day infectious period (`Smith et al. 2009`), ticks have multi-month developmental minimums, whereas *Babesia* and PUUV are chronic. Sensitivity runs in `lifecycle_engine.py` compare `shape='memoryless'` (`k=1`) vs Erlang `shape='chain'` (`substages=4`) and `fixed` stays (`Statement C2d`). |
| 7 | **Density dependence in the parasite and in the host** | **(a) included** | Parasite susceptible depletion `(N - I)` (and `N - I - R` lifelong immunity for cowpox) is built directly into `K9` blocks `S` and `SIR`; sublinear contact `eta_H in [0.25, 1.0]` (`Smith et al. 2009` `q = 0.62 => eta_H = 0.38`) is included in `K9` and swept in sensitivity (`C4a`, `C4b`). Host density regulation enters the empirical `z_H(t)` spectrum directly. |
| 8 | **Host feedback (parasite changing host numbers)** | **(c) bounded via Block `V` (`K3/K9`)** | *Bartonella* and *Babesia* show no detectable survival or fecundity cost in Kielder voles (`Taylor et al. 2018, 2023`), so `alpha = 0` (block `S`) applies. Cowpox lowers monthly vole survival by ~10–22% (`Burthe et al. 2008`). Sensitivity replaces block `S` by harmed-host block `V` (`notes/coupled-hosts.md` §1, standing condition `a_H >= alpha`, with `alpha = 0.22 / step` and self-regulation `rho in {0.1, 0.5}`) to verify how host feedback alters `H_p(i*omega)`. |
| 9 | **Alternative hosts and spatial spread / immigration** | **(b) controlled & (c) bounded via `lifecycle_engine.py`** | Wood mice (*Apodemus sylvaticus*) and bank voles occur at low numbers on Kielder grass clear-cuts (`species == 'BV', 'WM'` in `K15` are filtered out of *M. agrestis* counts and tabulated to bound their share `w_alt`). In `lifecycle_engine.py`, sensitivity runs test `Patches(count=4, movement=0.10)` and a `15%` steady alternative-host route (`Statement ring-B6`, `C7a`). |
| 10 | **Demographic noise at small numbers (fade-out in troughs)** | **(c) bounded via `lifecycle_engine.py`** | During Kielder/Konnevesi vole crashes (`< 5` voles/grid), whole-number fade-out can extinguish a pathogen locally until re-immigration (`Luis et al. 2015`; `R18`). Sensitivity calls `Engine.extinction()` (`offspring='poisson'` and `'negative_binomial'`) at initial cohort sizes representative of vole troughs (`N_0 = 20, 100`). |
| 11 | **Observation and sampling error (trapability, binomial testing, qPCR cutoff, assay type)** | **(b) controlled & (c) bounded** | (i) Effort normalisation (`trapsWorking`, `trapnights`) controls variable trapping effort; (ii) plug-in binomial sampling variance `(1 - p)/(n*p)` is subtracted in `D_atten` (`Statement E6d`); (iii) in `K15`, qPCR threshold sensitivity tests `RQ >= 0.5`, `RQ >= 1.0` (registered primary from `Taylor et al. 2023`), and `RQ > 0`; (iv) seroprevalence (`I+R`) vs active infection (`I`) is modelled explicitly in `K9`. |
| 12 | **Evolution during the study (resistance, genotype turnover)** | **(d) cannot address dynamically / noted** | *M. agrestis* `FcER1A` (`GC` haplotype) polymorphism modulates *Babesia* susceptibility (`Wanelik et al. 2023`), and *Bartonella* comprises multiple species/strains (`B. taylorii`, `B. grahamii`, `B. doshiae`; `Telfer et al. 2007a`) pooled by genus-level 16S qPCR in `K15`. Over 2–7 years, evolutionary change in `R0` cannot be separated from ecological noise (`d`), and pooling *Bartonella* species is flagged as a key limitation. |
| 13 | **Confounding between sites or experimental grid treatments** | **(b) controlled** | In `K15`, antiparasitic treatment (`gridTreat`, `frontline`, `profender`) is controlled by restricting primary pathogen analyses to untreated control records and testing treated vs untreated grids separately. Between-site habitat differences are removed by within-site centring before computing `ACF` and `CCF`. Co-infection (`Babesia` suppressing `Bartonella`, `Telfer et al. 2010`; `Taylor et al. 2018`) is checked by computing *Bartonella* `CCF` both overall and restricted to *Babesia*-negative voles! |

---

## 7. Protocol Amendment 1 (Schema Verification and Weather & Environment Protocol)

**Dated**: 2026-11-14 23:48:50 CST (2026-11-15 05:48:50 UTC) — recorded after inspecting file documentation, column headers, and distinct non-outcome design codes only, and **before** computing any empirical abundance, variance, autocorrelation, or prevalence statistics.

### 7.1 Exact schema and column-parsing rules fixed prior to analysis
1. **`K15` (`long_blood_qpcr_fv.2.csv`, EIDC `10.5285/e5854431-6fa4-4ff0-aa02-3de68763c952`)**:
   - **Header repair in memory**: Byte inspection of line 0 showed a missing newline in the raw deposit between the 77th column name `TOLLIP` (`vole_summary.docx`) and the first `bloodNum` identifier `A0001` (`...SENS3,TOLLIPA0001,CHE,25/03/2015,...`). `analysis.py` splits `TOLLIPA0001` into `TOLLIP\nA0001` in memory (leaving the raw file on disk untouched), recovering all 77 columns and all `17,883` capture records across sites `BLB, CHE, COL, GRD, HAM, RAV, SCP` (`species == 'FV'` on all rows).
   - **Grid vs individual treatment**: `gridTreat` (`'C'`, `'S'`, or `''`) denotes grid-level supplementation, whereas `treat` (`'TRUE'`/`'FALSE'`), `frontline` (dose in µL), `profender` (dose in µL), and `totTreat` denote individual-level antiparasitic dosing within grids. Per Section 4.1, primary pathogen prevalence in `K15` restricts to untreated captures (`treat != 'TRUE'`, `frontline == ''`, `profender == ''`), with sensitivity checks restricting further to never-previously-treated captures (`totTreat in ('', '0')`) and comparing `gridTreat == 'C'` vs `gridTreat == 'S'` and all captures.
   - **Effort normalisation**: For each `(site, session)` biweekly trapping bout, distinct *M. agrestis* (`chip`, falling back to `bloodNum` if `chip` is blank) per `100` `trapsWorking` (median across rows of that `(site, session)`) is averaged across the sessions falling in 28-day step `k` (so windows with 1 vs 2 biweekly sessions have comparable effort-adjusted per-bout abundance; raw distinct individuals per session is reported in sensitivity).
2. **`K08` (`Longitudinal database.xls`, sheet `TS1-18`, Zenodo `4971362` / Dryad `10.5061/dryad.bk537`)**:
   - Sites `BLB, KTH, SCP, SQC` in `2008–2010`; pathogen columns `Bartonella`, `Cowpox`, and `Babesia` encode `1.0` = positive, `0.0` = negative, and `'*'` = untested/missing. Within each `(Site, 28-day step k)`, repeated captures of the same individual (`Tag ID`, or row `ID` if untagged) are collapsed to one record per animal per 28-day step (positive if positive on any capture in that window; first-capture-only checked in sensitivity). `Arriero et al. (2018)` (`vole_immunological_profiles_data.xlsx`, 248 rows) is checked for tag/session concordance with `K08`.
3. **`F02` (`PUUV prevalence per trapping session.xls` and `Seroconversion rate per trapping session.xls`, Zenodo `4941743` / Dryad `10.5061/dryad.g8140`)**:
   - `site == 'core'` (`246` trap stations) provides the primary 60-session 7-year bank-vole abundance (`Total N * 100 / trapnights`, plus raw `Total N` sensitivity) and PUUV seroprevalence (`Infected.P / PUUV tested`), supplemented by direct incidence (`Seroconversion rate` per exposure day) as a mechanistic check on active transmission vs cumulative IgG seroprevalence. Satellite sites `A–P` (`9` traps each) are pooled per session for sensitivity.

### 7.2 Weather and environment protocol (`CASESTUDY_SPEC.md` end section)
Because `K15`, `K08`, `F02`, and `F23` are field datasets from specific coordinates and dates, `data/fetch/voles.py` is extended to fetch open meteorological series matched to each study area and span (stored in `data/raw/voles/weather/`, git-ignored, with SHA-256 and provenance in `MANIFEST.json`):

1. **Matched open meteorological sources and station distances**:
   - **Kielder Forest, Northumberland, UK** (`55.2300°N, 2.5750°W` EIDC trapping bounding-box centroid, `2007-01-01` to `2017-12-31` covering both `K08` and `K15`):
     - *Primary in-forest station*: NOAA GHCN-Daily `UKE00105892` (**Kielder Castle**, `55.2339°N, 2.5789°W`, elev. `201 m`, **distance `0.5 km`**; daily `TMAX, TMIN, PRCP`, DOI `10.7289/V5D21VHZ`, Public Domain).
     - *Upland snow & `TAVG` reference station*: NOAA GHCN-Daily `UK000003162` (**Eskdalemuir**, `55.3167°N, 3.2000°W`, elev. `242 m`, **distance `40.7 km`**; daily `TMAX, TMIN, TAVG, PRCP, SNWD`).
     - *Gridded daily meteorology & humidity*: NASA POWER Agroclimatology (`T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M` at `55.2300°N, -2.5750°E`, `0.5°` grid centre `55.25°N, -2.50°E`, **distance `5.3 km`**, DOI `10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY`, Public Domain).
   - **Konnevesi, central Finland** (`62.6280°N, 26.2880°E` Konnevesi Research Station, `2001-01-01` to `2009-12-31` covering `F02`):
     - *Primary nearest full-span temperature + precipitation station*: NOAA GHCN-Daily `FIE00144637` (**Vesanto Sonkari**, `62.9214°N, 26.4281°E`, elev. `114 m`, **distance `33.4 km`**; daily `TMAX, TMIN, PRCP`).
     - *Regional synoptic & snow-depth reference station*: NOAA GHCN-Daily `FI000002401` (**Jyväskylä Lentoasema**, `62.4019°N, 25.6783°E`, elev. `140 m`, **distance `40.1 km`**; daily `TMAX, TMIN, TAVG, PRCP, SNWD`).
     - *Local precipitation check station*: NOAA GHCN-Daily `FIE00144657` (**Konnevesi Lahdenkylä**, `62.5667°N, 26.2831°E`, elev. `93 m`, **distance `6.8 km`**; daily `PRCP`, `2001–2007`).
     - *Gridded daily meteorology & humidity*: NASA POWER Agroclimatology (`T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M` at `62.6280°N, 26.2880°E`, `0.5°` grid centre `62.50°N, 26.25°E`, **distance `14.4 km`**).
   - **Multi-area Finland (`F23`, Wang et al. 2023)**: Finnish Meteorological Institute (FMI) 5 km buffer seasonal temperatures and precipitation (`T5KMwinter..autumn`, `P5KMwinter..autumn`) already present in `Data_PUUV_in_bank_vole_Human_NE.csv`.

2. **Weather as covariates (`W1_cov` — could weather produce or hide the `K9` host-tracking effect?)**:
   - Aggregate daily temperature (`T_28d`), total precipitation (`P_28d`), relative humidity (`RH_28d`), and snow-cover days (`SNWD > 0`) onto the exact 28-day step grid `k` of each study.
   - Compute partial cross-correlations `CCF_weather(ell) = Corr(y_p(t + ell), z_H(t) | T_28d(t+ell), P_28d(t+ell), T_28d(t), P_28d(t))` (plus `SNWD` in Konnevesi and `RH_28d` in Kielder) and non-seasonal weather-anomaly regressions, alongside direct ectoparasite regressions (`mean fleas/vole` and `mean ticks/vole` vs `T_28d`, `RH_28d`, and `z_H` in `K15` and `K08`).
   - *Decision rule*: weather is a **verdict-moving confounder** if conditioning on `(T_28d, P_28d)` changes a primary `T2_track` or `T3_order` verdict; a **partial modulator** if the verdict holds while peak `CCF` changes by `>= 0.15`; and **minor** if `|Delta CCF| < 0.15` with unchanged verdict.

3. **Weather as thermal stage-duration & survival inputs (`W2_thermal` — `notes/mechanism-temperature-incubation.md`)**:
   - Map observed 28-day temperatures `T_k` into the exposed-stage clock `q_V(T_k)` and completion probability `p_V(T_k)` (`notes/mechanism-temperature-incubation.md` §1, Proposition B §2.2, and §3.2) for the three ectoparasitic/environmental stages while keeping endotherm intra-host stages (`37–38 °C`) sheltered:
     - *Tick vector (`I. trianguliceps` for `Babesia` in Kielder)*: burrow/litter damped temperature `T_eff = T_bar + m_buf*(T_k - T_bar)` (`m_buf = 0.5` primary, `[0.35, 1.0]` sensitivity), development `Q10_dev = 2.5` (with cold diapause floor below `6 °C`, `Randolph 1975` `[literature/illustrative]`), off-host mortality `Q10_mort = 1.8` `[illustrative]` (`pi_V > 0, chi_V > 0`: cold winter slows tick development and lowers completion).
     - *Flea vector (`Bartonella` in Kielder)*: insulated nest microclimate (`m_buf = 0.35`, `T_nest_base = 12 °C`, `Q10_dev = 2.2`, `Q10_mort = 2.0`, `Krasnov et al. 2001` `[literature/illustrative]`).
     - *Indirectly shed PUUV virions (`F02` in Konnevesi)*: environmental persistence decays faster in warm weather (`Q10_decay = 3.0`, `Kallio et al. 2006` `[literature]`) while uptake rate per host contact is temperature-flat (`Q10_uptake = 1.0` -> `pi_V < 0, chi_V > 0`, empirically instantiating Case `E4` of `notes/mechanism-temperature-incubation.md` §4.2!).
   - Compute along the empirical Kielder and Konnevesi weather trajectories: (i) seasonal range and CV of thermally driven stage stay `tau_V(T_k)`, (ii) curvature shift `J_temp = ln R0_ii - ln R0(T_bar)`, (iii) pure thermal fluctuation loss `D_temp` and Proposition B staying-inclusive bound `ell^g + (tau_0 - 1) ell^s`, and (iv) joint `(H(t), T(t))` fluctuation loss `D_joint` vs host-only `D_host`.

---

## 8. Protocol Amendment 2 (Step B: Multi-Source Open Weather & Subsurface/Environmental Covariates)

**Dated**: 2026-11-15 09:02:19 CST (`2026-11-15T15:02:19Z` by `date`) — registered after Step A (`analysis.py`, `results/tests.json`) completed the unadjusted `§4` tests and **before** running any Step B weather-host or weather-parasite statistical or thermal analyses.

### 8.1 Why this amendment is added
Step A intentionally ran only the unadjusted `§4` tests (`cov_keys=None`) and deferred weather covariates (`W1_cov`) and thermal stage-duration runs (`W2_thermal`) to Step B (`report.md` §5 Confounder #4). To ensure complete, multi-resolution open environmental coverage for Kielder Forest, UK (`K08`, `K15`) and the Finnish sites (`F02` Konnevesi `core` + `15` `satellite` sites, plus the `30` Finnish sites in `F23`) without any API token or personal account, Step B adds a standalone weather fetch script `data/fetch/voles-weather.py` (`data/raw/voles/weather/MANIFEST.json`) that preserves and SHA-256-verifies the Amendment 1 NOAA GHCN-Daily and NASA POWER series while adding ~9 km daily ERA5-Land/ERA5-seamless surface-and-soil reanalysis and ~4 km monthly TerraClimate water-balance and snow-water-equivalent grids, and fixes the exact covariate operationalisations before running `step_b_weather.py`.

### 8.2 Registered open weather and environmental sources and covariates (`data/fetch/voles-weather.py`)
All HTTP requests use generic `User-Agent: open-ecology-fetch/1.0` with no email address, personal details, or API token (`CASESTUDY_SPEC.md` Rule 2):

1. **NOAA GHCN-Daily station records** (DOI `10.7289/V5D21VHZ`, U.S. Gov / Public Domain, daily):
   - *Kielder Forest (`55.2300°N, 2.5750°W`, `2007–2017`)*: `UKE00105892` (Kielder Castle, `0.50 km`, `TMAX, TMIN, PRCP`) and `UK000003162` (Eskdalemuir, `40.66 km`, `TMAX, TMIN, TAVG, PRCP, SNWD`).
   - *Konnevesi (`62.6280°N, 26.2880°E`, `2001–2009`)*: `FIE00144637` (Vesanto Sonkari, `33.41 km`, `TMAX, TMIN, PRCP`), `FI000002401` (Jyväskylä Lentoasema, `40.15 km`, `TMAX, TMIN, TAVG, PRCP, SNWD`), and `FIE00144657` (Konnevesi Lahdenkylä, `6.82 km`, `PRCP`, `2001–2007`).
   - *Registered 28-day covariates*: `T_28d_C` (mean daily station temperature, °C), `P_28d_mm` (28-day precipitation sum, mm), `frost_frac_28d` (fraction of days with `TMIN < 0 °C`), and `snow_frac_28d` (fraction of days with `SNWD > 0`).
2. **ERA5-Land / ERA5-seamless daily reanalysis via Open-Meteo Historical Weather Archive** (`archive-api.open-meteo.com/v1/archive`, `0.1° × 0.1°` ~`9 km`, Copernicus C3S ERA5-Land DOI `10.24381/cds.e2161bac`, CC BY 4.0, unauthenticated open mirror):
   - Matched to Kielder Forest (`55.2300°N, -2.5750°E`, `2007-01-01..2017-12-31`), Konnevesi Research Station (`62.6280°N, 26.2880°E`, `2001-01-01..2009-12-31`), and the Konnevesi Dryad/Zenodo metadata coordinate (`62°34′N, 26°24′E` = `62.5667°N, 26.4000°E`, `2001-01-01..2009-12-31`).
   - *Registered 28-day covariates*: `ERA5_T2m_28d_C` (2 m mean air temperature, °C), `ERA5_P_28d_mm` (28-day precipitation sum, mm), `ERA5_RH_28d_pct` (2 m relative humidity, %), `ERA5_snow_depth_28d_m` (mean daily max snow depth, m), **`ERA5_Tsoil0_7_28d_C`** (`0–7 cm` litter / subnivean runway soil temperature, °C), **`ERA5_Tsoil7_28_28d_C`** (`7–28 cm` vole burrow / nest soil temperature, °C), and **`ERA5_smoist0_7_28d_m3m3`** (`0–7 cm` volumetric soil moisture, $\text{m}^3\,\text{m}^{-3}$).
3. **TerraClimate v1.1 monthly high-resolution land climate (`1/24°`, ~`4 km`)** (`thredds.northwestknowledge.net/thredds/dodsC`, Abatzoglou et al. 2018 DOI `10.1038/sdata.2017.191`, CC0 1.0):
   - Nearest valid ~4 km land grid cell to Kielder (`2007–2017`) and Konnevesi (`2001–2009`), mapped onto each 28-day step `[d_start, d_start + 27]` by exact day-count weighting across overlapping calendar months.
   - *Registered 28-day covariates*: `TC_tmean_28d_C` (`0.5*(tmax + tmin)`, °C), `TC_ppt_28d_mm` (scaled to 28 days, mm), **`TC_soil_28d_mm`** (root-zone soil moisture storage, mm), and **`TC_swe_28d_mm`** (snow water equivalent, mm).
4. **NASA POWER Agroclimatology v2.0 daily** (`0.5° × 0.625°`, DOI `10.5067/TERRA+AQUA/CERES/POWER_SYN1DEG_DAILY`, Public Domain):
   - Matched to Kielder (`26.02 km` to MERRA-2 `0.5°×0.625°` cell `55.0°N, 2.5°W`; `5.26 km` to `0.5°` grid centre `55.25°N, 2.50°W`) and Konnevesi (`14.37 km` to `62.5°N, 26.25°E`), supplying `T_power_28d_C`, `P_power_28d_mm`, and primary `RH_28d_pct`.
5. **Deposited 5-km FMI seasonal weather and CORINE habitat for the 30 Finnish sites in `F23`** (`Wang et al. 2023`, Dryad `10.5061/dryad.bg79cnpfh`, CC0 1.0):
   - Because `Data_PUUV_in_bank_vole_Human_NE.csv` anonymises site locations (`Area_no = 1..30` across `24` municipalities, `2001–2005`, `119` site-years) without coordinates, Step B uses the deposit's site-matched Finnish Meteorological Institute 5-km seasonal temperatures and precipitation (`T5KMwinter..autumn`, `P5KMwinter..autumn`) and CORINE 2006 5-km habitat composition (`Forest5.P, Wetland5.P, Water5.P, Agricul5.P, Artificial5.P, ED5`) rather than guessing municipality coordinates.
   - Unauthenticated endpoint probes to Met Office HadUK-Grid (`dap.ceda.ac.uk`) and KNMI E-OBS (`opendap.knmi.nl`) are logged transparently in `data/raw/voles/weather/MANIFEST.json` without bypass if restricted or unreachable.

### 8.3 Registered Step B analyses and decision criteria (`step_b_weather.py`)
1. **Cross-source concordance and burrow thermal buffering (`B1_verify`)**:
   - Report daily and 28-day Pearson `r` and `RMSE` across GHCN-Daily, ERA5-Land, TerraClimate, and NASA POWER, plus Konnevesi station vs Dryad-centroid ERA5-Land concordance, and compare observed `SD(ERA5_Tsoil0_7_28d_C) / SD(T_28d_C)` and `SD(ERA5_Tsoil7_28_28d_C) / SD(T_28d_C)` against the pre-registered `m_buf = 0.50` and `0.35` assumptions (`§7.2.3`).
2. **Covariation with host abundance, variability, and `T1_mem` (`B2_host`)**:
   - Compute pooled within-site correlations (`r`, two-sided `t`-distribution `p`, and pair counts) of `z_H = ln H`, 28-day growth rate `Delta z_H(k) = z_H(k) - z_H(k - 1)`, and squared host deviation `(z_H - mean_s(z_H))^2` with each 28-day weather/environment covariate at lags `0` and `1` step (both raw and deseasonalised calendar-harmonic anomalies) across `F02` `core` (+ `satellite_pooled`), `K15`, and `K08`.
   - Re-evaluate **`T1_mem`** (`Delta AICc >= +4.0` supports structured memory beyond `AR(1)`) on `F02` `core` (`N=60`) and `K15` pooled (`G=4`, `N=79`) after adding (a) Amendment 1 surface weather `(T_28d_C, P_28d_mm, snow_frac_28d / RH_28d_pct)` and (b) ERA5-Land subsurface covariates `(ERA5_Tsoil7_28_28d_C, ERA5_smoist0_7_28d_m3m3, ERA5_snow_depth_28d_m)` to the Gaussian MLE mean design matrix `X`.
   - Across the `30` Finnish areas in `F23` (sites with `>= 4` annual rows for `SD(ln H)`, and all `119` site-years for `ln H`), test correlations between bank-vole variability `SD(ln H)` / abundance and FMI 5-km seasonal climate means/SDs and CORINE forest/wetland/edge-density covariates.
3. **Covariation with parasite outcomes, ectoparasite burdens, and weather-partial `T2_track` & `T3_order` (`B3_parasite`, Amendment 1 `W1_cov`)**:
   - Report within-site raw and deseasonalised correlations of pathogen log-prevalence `y_p` (`F02` PUUV on `core` + `satellite_pooled`, plus `F02` `Seroconversion rate`; `K15` *Bartonella*, *Babesia*; `K08` Cowpox, *Bartonella*, *Babesia*) and Kielder vector burdens (`mean_fleas`, `mean_ticks` in `K15` and `K08`) with each weather/environment covariate.
   - Re-run **`T2_track`** (`2,000` phase surrogates, seed `20261114`) on all six Step A series under three pre-specified adjustments:
     - `W1a` (Amendment 1 primary surface weather): partial out `[T_28d_C, P_28d_mm, snow_frac_28d]` in `F02` and `[T_28d_C, P_28d_mm, RH_28d_pct]` in `K15`/`K08` at `(t, t + ell)`;
     - `W1b` (ERA5-Land physical subsurface/microclimate): partial out `[ERA5_Tsoil7_28_28d_C, ERA5_smoist0_7_28d_m3m3, ERA5_snow_depth_28d_m]` at `(t, t + ell)`;
     - `W1c` (Calendar-harmonic seasonal residual, `§6` #2 & #4): partial out annual `cos(omega_yr t), sin(omega_yr t)` within each site before computing `CCF` and phase surrogates.
   - Re-run **`T3_order`** (`2,000` within-site block bootstraps, `block_len=3`, seed `20261114`) in `K15` and `K08` on `W1a`-, `W1b`-, and `W1c`-residualised `z_H`, `y_Bartonella`, and `y_Babesia`.
   - Test across-site and within-site associations in `F23` (`30` Finnish areas) between `PUUV` seroprevalence (`n_screened >= 8`) and bank-vole abundance / `SD(ln H)` before vs after controlling FMI 5-km seasonal climate and CORINE habitat.
   - Classify weather's confounding role on each test by the Amendment 1 `§7.2.2` criterion (`verdict-moving confounder`, `partial modulator` if `|Delta S_pos| >= 0.15` with unchanged verdict, or `minor` if `|Delta S_pos| < 0.15` with unchanged verdict).
4. **Weather as thermal stage-duration & survival inputs (`B4_thermal`, Amendment 1 `W2_thermal`)**:
   - Evaluate `evaluate_thermal_trajectory_loss` (`notes/mechanism-temperature-incubation.md` §§1, 2.2) along the empirical 28-day trajectories of `K15` (`2015–2017`), `K08` (`2008–2010`), and `F02` (`2002–2009`) for Kielder *Babesia* ticks, Kielder *Bartonella* fleas, and Konnevesi `PUUV` shed virions, comparing Amendment 1 buffered air temperature (`m_buf = 0.50` / `0.35`, sensitivity `[0.35, 1.0]`) against direct ERA5-Land physical `0–7 cm` litter and `7–28 cm` burrow temperatures (`m_buf = 1.0`), reporting `tau_V(T_k)` range/CV, `J_temp`, `D_temp`, `D_host`, `D_joint`, `Cov(g_V(T), H^eta)` mean shift, and Proposition B bound verification.

### 8.4 Pre-execution operational note (before running `step_b_weather.py`)

**Dated**: 2026-11-15 10:20:21 CST (`2026-11-15T16:20:21Z` by `date`) — recorded prior to executing any Step B statistical or thermal analysis:
1. **Exact Haversine distances (`r_earth = 6371.0 km`):** Exact computed station distances recorded in `data/raw/voles/weather/MANIFEST.json` (`UK000003162` Eskdalemuir `40.75 km`, `FIE00144637` Vesanto Sonkari `33.39 km`, `FI000002401` Jyväskylä `40.14 km`, `UKE00105892` Kielder Castle `0.50 km`, `FIE00144657` Konnevesi Lahdenkylä `6.82 km`, ERA5-Land Kielder `3.69 km` / Konnevesi Station `3.17 km` / Konnevesi Dryad `3.70 km`, TerraClimate Kielder `0.80 km` / Konnevesi `2.17 km`) supersede the two-decimal hand-rounded figures in §7.2.1/§8.2.1.
2. **Degrees-of-freedom rule for weather-adjusted `T2_track` and `T3_order` (`§8.3.3`):** Code inspection of `voles_models.compute_pooled_ccf` showed that fitting `2*K = 6` lagged weather columns `[W(t), W(t + ell)]` pair-by-pair would leave `< 4` residual degrees of freedom (and fall back to unadjusted `corrcoef`) at sparse Kielder lags (`6–12` pairs at `ell = 6..8`). Therefore: (i) primary `W1a`, `W1b`, and `W1c` tests residualise `z_H(t)` on within-site centred `W(t)` across all observed host windows and `y_p(t)` on `W(t)` across all observed pathogen windows (`n >= 8`) before running the registered `2,000`-surrogate `T2_track` and `2,000`-bootstrap `T3_order` tests (`seed = 20261114`); and (ii) companion pair-wise `CCF(ell | W(t), W(t + ell))` is reported alongside at every lag with `n_pairs - 1 - 2*K >= 5` (returning `None` rather than unadjusted `corrcoef` where `df < 5`).
3. **Reference rate splits fixed for `B4_thermal` (`§7.2.3` & `§8.3.4`, all `[illustrative]` splits of Table 2.3 total exit rates `a_V = nu_ref + mu_ref`):**
   - *Babesia* tick (`a_H = 0.35`, `a_V = 0.12 / 28d`, `eta_H = 0.50`): `T_ref_C = 8.0 °C`, `nu_ref = 0.06`, `mu_ref = 0.06` (`p_ref = 0.50`, sensitivity `p_ref in {0.25, 0.75}`), `Q10_dev = 2.5`, `Q10_mort = 1.8`, `cold_floor_C = 6.0 °C`.
   - *Bartonella* flea (`a_H = 0.70`, `a_V = 0.45 / 28d`, `eta_H = 0.50`): `T_ref_C = 12.0 °C`, `nu_ref = 0.25`, `mu_ref = 0.20` (`p_ref = 0.556`), `Q10_dev = 2.2`, `Q10_mort = 2.0`, `cold_floor_C = None`.
   - *PUUV* shed virion (`a_H = 0.30`, `a_V = 2.00 / 28d`, `eta_H = 0.70`): `T_ref_C = 4.0 °C`, `nu_ref = 0.50`, `mu_ref = 1.50` (`p_ref = 0.25`, sensitivity `p_ref in {0.10, 0.50}`), `Q10_dev = 1.0`, `Q10_mort = 3.0` (`pi_V < 0, chi_V > 0`, Case `E4`), `cold_floor_C = None`, `m_buf = 0.75` (`[0.50, 1.0]` sensitivity + direct ERA5-Land `0–7 cm` litter T).
4. **Rule 7 pre-analysis code verification tolerances for `step_b_weather.py` (`2026-11-15T17:41:29Z` by `date`, fixed before running `step_b_weather.py`):**
   - `SV1` (tol `1e-12`): Day-weighted monthly-to-28-day TerraClimate aggregator reproduces exact constant temperature `c` and exact day-prorated precipitation sum across single-month and straddling-month 28-day windows.
   - `SV2` (tol `1e-10` on NLL/AICc; `1e-5` on shift-invariance): Covariate-extended Gaussian MLE (`fit_host_memory_panel_with_covs`) reproduces `voles_models.fit_host_memory_panel` across `M0..M4` when `W = None`, and preserves GLS residual `sigma^2` and correlation parameters (`phi`, `zeta`, `omega_0`, `nugget`) under exact linear covariate shifts `y' = y + W gamma`.
   - `SV3` (tol `1e-12`): Within-site OLS residualisation + Pearson `r` matches the exact precision-matrix Schur-complement partial correlation `-P_{zy}/sqrt(P_{zz} P_{yy})` and exact two-sided Student-`t` tail probabilities (`df = n - 2 - p_cov`).
   - `SV4` (tol `1e-10` for Proposition A; `1e-7` for Proposition B; `1e-3` abs against `notes/mechanism-temperature-incubation.md` §4.2 Case `E4` Floquet benchmark): `evaluate_thermal_trajectory_loss` reproduces the independent published Floquet benchmarks of `notes/mechanism-temperature-incubation.md` §4.2 (`Case E4`, 20-day sheltered + 1 exposed stage with flat development and `Q10_mort = 3`, sinusoid amplitude `7.07 °C`: `D = +0.0017` at period `7 d` and `D = -0.0170` at period `365 d`, confirming negative `D` = fluctuation gain under slow thermal memory) in addition to `verify_parsers()` and `verify_voles_models()`.




