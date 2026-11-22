# Protocol: Parasite Decline and Life-Cycle Traits in Puget Sound Museum Fishes (Reanalysis of Wood et al. 2023)

- **Study ID**: `wood`
- **Protocol locked**: `Sat Nov 14 22:31:19 CST 2026` (prior to accessing any parasite count outcome, temporal slope, or fitted model output from Wood et al. 2023)
- **Pre-analysis blind artifact hashes (SHA-256)**:
  - `traits_blind.csv`: `f4d0202392697dd3cff69f10eecdcd97f6331c27386e3c82e94a5a394332fe35`
  - `results/theory_predictions.txt`: `c76ba5c968c6376653723e68963b739d4dd98196a99e51f05459cf4a4a6def46`
  - `results/pre_analysis_power.txt`: `688e92311d81a3a246535e5535bcc5096fe930dcca7c64e4a1a55d23ce339f89`

---

## 1. Question

Wood et al. (2023, *PNAS* 120:e2211903120) dissected 699 fluid-preserved museum specimens of eight marine fish species collected in Puget Sound, Washington, between 1880 and 2019, and reported that parasite taxa with three or more obligately required host species declined in abundance by about 10.9% per decade whereas taxa with one or two obligate hosts showed no net change.

**Question**: Does the project's stochastic life-cycle theory add predictive value beyond sequential obligate host number (`nhosts`) in explaining which parasite taxa declined over the century?

---

## 2. Theoretical Predictions and Expected Effect Sizes

### 2.1 Definitions and core theoretical premise (`notes/exact-limits.md`)

Consider a discrete-time parasite life cycle with mean projection matrix $A_0$ (holding every demographic rate at its arithmetic mean, baseline A4) and deterministic whole-cycle basic reproduction number $R_0 = \rho(F(I - U)^{-1})$. In a stationary ergodic fluctuating environment, let $R_0^*$ denote the divisor applied to a single counted birth transfer per cycle at which the long-run stochastic growth rate $r$ equals zero (`section_ring`, Statement `ring-B13a`). The stochastic threshold loss is
$$D = \ln R_0 - \ln R_0^*,$$
measured in natural-log units of whole-cycle $R_0$. Persistence when rare requires $\ln R_0 > D$.

**Why sequential host number $n$ is not itself the cause (`notes/exact-limits.md` H3)**:
1. **Sheltered-segment invariance (Statement `mech-C2a`, proved and refereed)**: Any sheltered segment of consecutive stages with constant rates enters $r$ and $D$ solely through its survival-and-timing kernel $K(\ell)$ (the expected outflow $\ell$ steps after entry). Relabelling sub-stages inside a single steady host as separate steady hosts in series yields the exact same matrix and identical $D$ (`theory_predictions.py` check `T0b`: difference `0.00e+00`).
2. **Steady-host addition (Statements `ring-B5a`, `exact-P4`)**: Adding an obligate host whose abundance does not fluctuate ($\sigma_k = 0$) adds zero to the all-noise transfer bound $D \le \sum_k \ell_k$ (`ring-B11`) and, when its stay is variable and long relative to host memory, reduces $D$.
3. **Where sequential stages can raise $D$ indirectly (`ring-B3`, `mech-C4c`)**:
   - First, if each added obligate host itself fluctuates independently with log standard deviation $\sigma_j > 0$ and AR(1) autocorrelation $\phi_j \in [0, 1)$, the second-order loss sums across fluctuating transfers (`mech-C4c`):
     $$D_2 = \sum_{j} \sigma_j^2 K_g(\tau, \phi_j), \qquad K_g(\tau, \phi) = \frac{1 + 2\theta}{2\mathcal{T}} - \frac{R(\phi)}{1 - R(\phi)} \ge \frac{1}{2\mathcal{T}},$$
     where $\tau_i = 1/(1 - s_i)$ is the mean threshold lineage stay in stage $i$, $\mathcal{T} = \sum_i \tau_i$ is the mean generation time, $\theta = \phi/(1 - \phi)$, and $R(\phi) = \mathbb{E}[\phi^L] = \prod_i (1 + \tau_i/\theta)^{-1}$.
   - Second, even with a single fluctuating transfer at fixed total generation time $\mathcal{T}$ and positive memory $\phi > 0$, splitting $\mathcal{T}$ into more equal sequential geometric stages lowers generation-time variance $\operatorname{Var}(L)$ and raises $D_2$ (`Contrast H0` in `results/theory_predictions.txt`: at $\mathcal{T}=12, \sigma=1.0, \phi=0.75$, $D$ rises from `0.059` for `(12)` to `0.145` for `(2,10)` and `0.206` for `(2,4,6)`, whereas at $\phi=0$ all three give $D_2 = 0.0417$). Even there, an asymmetric 3-host cycle `(18,1,1)` has *smaller* $D_2$ than a symmetric 2-host cycle `(10,10)` (`mech-C4c`). Thus host count alone does not order $D$.

### 2.2 Link from threshold loss $D_k$ to empirical temporal trend $\gamma_k$

Wood et al. estimate a taxon-specific slope $\gamma_k$ of log parasite count per fish specimen per standard deviation of collection year ($s_t = 30.2681\text{ yr}$ over 1880–2019; fractional change per decade $= \exp(10\gamma_k / s_t) - 1$).

The project's theory describes stationary ergodic fluctuations around a baseline, not a century-long deterministic trend. A directional prediction for $\gamma_k$ follows under the comparative hypothesis that environmental and host-abundance variability $\sigma_H^2$ in Puget Sound increased over 1880–2019 by $\Delta\sigma_H^2 > 0$ (through warming, harvesting, and food-web perturbation) or that baseline transmission margins $\ln R_{0,k} - D_k$ narrowed toward criticality:
- Under regulated endemic transmission (`ring-B15b`, exponential or Beverton–Holt transfer regulation), log equilibrium burden increases monotonically with $\ln R_{0,k}^* = \ln R_{0,k} - D_k$, and near criticality finite-population fadeout rises with $D_k$ (`studies.E5c`, `studies.R18-order`).
- Holding mean rates fixed, a rise in variability shifts $\ln R_{0,k}^*$ by $-\Delta D_k$. Taxa whose life-cycle traits create **larger $D_k$ (and larger $\partial D_k / \partial \sigma^2$)** suffer a larger drop in $\ln R_{0,k}^*$ and therefore a **more negative temporal slope $\gamma_k$**.
- **Scope limit on magnitude**: Neither the century-scale change $\Delta\sigma_H^2$ across Puget Sound invertebrate and vertebrate guilds nor the burden elasticity $\partial \ln \mu_k / \partial \ln R_{0,k}^*$ is measured in these data. The theory fixes exact differences $\Delta D$ in units of $\ln R_0$ (computed in `results/theory_predictions.txt` across illustrative inputs grounded in `studies.INPUTS`, R12, and R14: $\sigma_{\text{high}} \in \{0.6, 1.0\}$, $\sigma_{\text{low}} = 0.25$, monthly $\phi \in \{0.0, 0.5, 0.75\}$), fixing **signs and relative ordering**, while expected empirical slope magnitudes $\beta = -\kappa \Delta D$ depend on the unmeasured scaling $\kappa > 0$ (where $\kappa \approx 0.5\text{--}1.0$ per SD-year corresponds to variability losses explaining half to all of Wood et al.'s $-0.35$ SD-year 3+-host decline).

### 2.3 Registered directional predictions (P1–P4)

| ID | Trait predictor | Formula / engine basis (`notes/exact-limits.md`) | Predicted sign on slope $\gamma_k$ | Illustrative theoretical size $\Delta D$ (`results/theory_predictions.txt`) | Exact theoretical limits where sign can stop or reverse |
|---|---|---|---|---|---|
| **P1 (a)** | `n_fluc_obligate`: count of obligate transfers into high-fluctuation host guilds (short-lived planktonic/epibenthic crustaceans, gelatinous zooplankton, or neritic schooling fish) | `H1`, `H4`, `mech-C4c`, `ring-B11`: $D_2 = \sum_j \sigma_j^2 K_g(\tau,\phi_j)$; $D \le \frac12\sum_j \eta_j^2\sigma_{H,j}^2$ | **Negative ($\beta_a < 0$)**: more fluctuating transfers $\implies$ steeper decline | $+\mathbf{0.046}$ to $+\mathbf{0.236}$ in $\ln R_0$ per added fluctuating transfer (`stays 2,3,4`, $\sigma_{\text{hi}} \in [0.6, 1.0]$, $\phi \in [0.5, 0.75]$) | Linked predator–prey hosts in favourable cycle order (`mech-C6b`); asynchronous patch dispersal with positive memory (`mech-C7a`); negative host memory (`mech-C4h`). |
| **P2 (b)** | `storage_stage`: presence (`0/1`) of a long-lived variable-residence encysted/encapsulated larval stage in vertebrate tissue | `H2`, `ring-B5a`, `mech-C2d`, `exact-P4`: $\partial_\tau D < 0$ strictly in i.i.d. gamma model; $D_2 \to 0$ as steady stay $b \to \infty$ under waste | **Positive ($\beta_b > 0$)**: storage buffers host fluctuations $\implies$ less decline | $-\mathbf{0.112}$ to $-\mathbf{0.391}$ in $\ln R_0$ ($51\%\text{--}74\%$ reduction comparing `(1,2,24)` vs `(1,2,3)`; $-\mathbf{0.107}$ to $-\mathbf{0.301}$ for Erlang-4 stay) | Steady stay short relative to host memory (`ring-B5b`); every stay in cycle fixed (`mech-C2f`, $D=\sigma^2/2$); contact-limited floor $d(1-d)q_j\sigma^2$ (`mech-C1i`); shared host mortality raises extinction level though longer stays still reduce it in 44/54 R18 pairs (`studies.R18-length`). |
| **P3 (c)** | `spreading`: presence (`0/1`) of paratenic/postcyclic transport hosts or multi-family host breadth at a step | `H3`, `ring-B6`, `mech-C4d`, `mech-C4f`: $M$ equal hosts multiply $D_2$ by $[1+\rho(M-1)]/M$; paratenic bridge lengthens $G$ at equal $R_0$ | **Positive ($\beta_c > 0$)**: spreading across alternative/paratenic hosts $\implies$ less decline | $-\mathbf{0.028}$ to $-\mathbf{0.119}$ in $\ln R_0$ ($M=4$ hosts, $\rho \in [0, 0.35]$); $-\mathbf{0.060}$ to $-\mathbf{0.167}$ ($51\%\text{--}53\%$ drop) with paratenic bridge | Synchronous hosts ($\rho \to 1$ removes benefit); much more variable newcomer above finite-noise share $f^*$ (`mech-C4e`); optional steady host with low survival $q<1$ or alternating memory (`mech-C4f(b)`). |
| **P4 (d)** | `prop_active_transfers`: fraction (`[0, 1]`) of obligate transfers via an active free-living host-seeking stage (proxy for scarcity wastes, $d \to 1$) vs passive trophic stage (proxy for scarcity delays, $d < 1$) | `H1`, `H2`, `mech-C1a`–`C1j`: at $d=0$, $D=0$ (`C1g`); at $\phi=0$, $K_c = d^2 K_g < K_g$ (`C1e(i)`); crossover at `C1h`–`C1i` | **Negative ($\beta_d < 0$)**: active free-living waste $\implies$ larger $D$ and steeper decline than low-$d$ trophic waiting *(with registered `C1h` interaction caveat)* | At `other_stay = 6`, $\phi=0.75$: $D = 0.093$ (`d=1` waste) vs $0.024$ (`d=0.15`) and $0.008$ (`d=0.05` trophic cyst). **Qualification (`C1h`/`C1i`)**: at `other_stay = 40`, $\phi=0.75$, `d=0.05` still lowers $D$ (`0.010` vs `0.023`), whereas moderate waiting death `d=0.50` *raises* $D$ (`0.056` vs `0.023`). | Reverses when waiting mortality $d$ is moderate ($0.15\text{--}0.50$), host memory $\phi > 0$ is strong, and another stage has very long residence (`mech-C1h`, `C1i`, `studies.E5b`). |

---

## 3. Data, Provenance, and Blind Trait Coding

All raw data were fetched via `python3 -I data/fetch/wood.py` into `data/raw/wood/` (verified git-ignored via `git check-ignore`; `MANIFEST.json` records sizes, MD5, SHA-256, URLs, and UTC timestamps; access date `2026-11-14`):

1. **Wood et al. (2023) Puget Sound museum dataset**:
   - Primary DOI: `10.5061/dryad.fqz612jwf` (version 6 on Dryad; licence **CC0 1.0**). Direct automated download from Dryad returned HTTP 401 (bearer token required; recorded in `MANIFEST.json`, not retried). Downloaded from the official Zenodo `dryad` community mirror (**Zenodo record `7448240`**, CC0 1.0, MD5-verified): `final_dataset.csv` (`8,754` rows, `699` individual fish across `8` fish species collected `1880–2019`, `85` common parasite taxa, `97` host–parasite pairs), `final_dataset_metadata.csv`, `parasite_species_codes.csv`, `README.md`.
   - Authors' open code/metadata repository: `github.com/wood-lab/Wood_et_al_2023_PNAS` (renamed on GitHub from `Wood_et_al_2022_PNAS`; commit SHA recorded in `data/raw/wood/wood_github/_github_repo.json`), read as plain text and **never executed** (`data/psite_lc_complexity.csv`, `src/load_data.R`, `src/configure_TMB_runs.R`, `src/phase_1_models.R`, `src/TMB/*.cpp`).
   - Open-access article text and SI Tables S1–S3: Europe PMC `PMC9934024` (`doi:10.1073/pnas.2211903120`); only Materials & Methods and SI Tables S1–S3 (lines 1–325 of `pnas.2211903120.sapp.pdf`) were read prior to locking this protocol.
2. **External open trait databases (coded 100% blind to parasite count/trend outcomes in `build_traits_blind.py` $\to$ `traits_blind.csv`)**:
   - **Benesh, Parker & Chubb (2021) / Benesh, Lafferty & Kuris (2017) Helminth Life-Cycle Database**: `doi:10.5061/dryad.gtht76hmm` via official Zenodo mirror **record `4975516`** (**CC0 1.0**, MD5-verified) and raw stage/host/life-history CSVs (`CLC_database_hosts.csv`, `CLC_database_lifehistory.csv`, `stage_level_combined_noimputed.csv`, derived from *Ecology* 98:882, `doi:10.1002/ecy.1680`, CC0).
   - **Natural History Museum (London) Host-Parasite Database**: NHM Data Portal (`doi:10.5519/nplk7c5u`, CC BY 4.0) metadata fetched (`HTTP 200`), while direct zip download returned `HTTP 403` and legacy `results.jsp` returned `HTTP 502` (recorded in `MANIFEST.json`, not circumvented). Host-range records were instead queried via **Global Biotic Interactions (`api.globalbioticinteractions.org`, CC BY 4.0)**, which aggregates LNHM, WoRMS, USNPC, and GMPD host–parasite associations (`worms_globi_cache.json`).
   - **Global Mammal Parasite Database 2.0**: Stephens et al. (2017, *Ecology* 98:1476, `doi:10.1002/ecy.1799`), `GMPD_main.csv` via GloBI mirror.
   - **World Register of Marine Species (WoRMS)**: `marinespecies.org/rest` (**CC BY 4.0**), `worms_globi_cache.json`.
   - **FishBase host ecology**: Wood et al. SI Table S1 (Froese et al. 2022), classifying the 8 fish species objectively into neritic schooling (`Clupea pallasii`, `Merluccius productus`) vs demersal (`Embiotoca lateralis`, `Hydrolagus colliei`, `Hypomesus pretiosus`, `Parophrys vetulus`, `Sebastes caurinus`, `Gadus chalcogrammus`).
   - Every single taxon's codes and literature citations are locked in `traits_blind.csv` and `trait_sources.md` before any outcome is opened.

---

## 4. Estimands, Statistical Models, and Decision Criteria

### 4.1 Step 1: Independent reproduction of Wood et al. (2023) Phase 1 models

In `analysis.py` (run with `python3 -I`, without executing any downloaded code), we implement the exact hierarchical Negative Binomial GLMM likelihood of Wood et al. (`src/TMB/glmm*.cpp`, `src/phase_1_models.R`) via Laplace approximation over the random effects:
- Observation $i \in \{1, \dots, 8754\}$ (parasite taxon $k(i) \in \{1, \dots, 85\}$, host–parasite pair $j(i) \in \{1, \dots, 97\}$, fish specimen $m(i) \in \{1, \dots, 699\}$, taxonomic group $p(k) \in \{1, \dots, 7\}$):
  $$y_i \sim \operatorname{NegBin}(\mu_i, \phi_{k(i)}), \qquad \operatorname{Var}(y_i) = \mu_i + \mu_i^2 / \phi_{k(i)},$$
  $$\ln \mu_i = \beta_{0, j(i)} + \beta_{\text{lat}, k(i)} X_{\text{lat}, i} + \beta_{\text{len}, j(i)} X_{\text{len}, i} + \zeta_{m(i)} + \gamma_{k(i)} u_i,$$
  where $u_i = (t_i - \bar{t}) / s_t$ ($s_t = 30.2681\text{ yr}$), $X_{\text{lat}, i}$ is standardized latitude (`NA` set to `0` as in `load_data.R`), $X_{\text{len}, i}$ is `length` from `final_dataset.csv`, $\zeta_m \sim \mathcal{N}(0, \sigma_\zeta^2)$, $\beta_{\text{lat}, k} \sim \mathcal{N}(\bar{\beta}_{\text{lat}}, \sigma_{\text{lat}}^2)$, $\beta_{\text{len}, j} \sim \mathcal{N}(\bar{\beta}_{\text{len}}, \sigma_{\text{len}}^2)$, and $\ln \phi_k \sim \mathcal{N}(\overline{\ln \phi}, \sigma_{\ln \phi}^2)$.
- Four Phase 1 models reproduced:
  - `M1_null`: $\gamma_k \equiv 0$.
  - `M2_year`: $\gamma_k \sim \mathcal{N}(\bar{\gamma}, \sigma_\gamma^2)$.
  - `M3_nhosts`: $\gamma_k \sim \mathcal{N}(\mu_{\gamma, h(k)}, \sigma_\gamma^2)$ for $h(k) \in \{1, 2, 3+\}$ (with group intercept $\beta_{0, g} \sim \mathcal{N}(\bar{\beta}_0, \sigma_{0,g}^2)$).
  - `M4_partype`: $\gamma_k = a_{p(k)} + g_k$, $g_k \sim \mathcal{N}(0, \sigma_\gamma^2)$ across the 7 broad taxonomic groups.
- **Preset reproduction tolerance (Rule 7)**: Before running registered trait tests, our independent solver must reproduce Wood et al.'s published `3+`-host decline rate (`-10.9%/decade`, i.e. $\hat{\mu}_{\gamma, 3+} \approx -0.349$ per SD-year) within `0.03` per SD-year (`1.0%/decade`), confirm near-zero slopes for `1`- and `2`-host groups, and recover the published Phase 1 AICc ordering (`M3_nhosts` best, beating `M4_partype` and `M2_year` and `M1_null`).

### 4.2 Step 2: Registered trait models and phylogenetic/taxonomic non-independence

Let $T_k = (T_{k,a}, T_{k,b}, T_{k,c}, T_{k,d})$ be the four primary blind-coded traits from `traits_blind.csv`. We fit the registered models **both** (i) in the unified one-stage 8,754-observation Laplace NB-GLMM (replacing $\mu_{\gamma, h(k)}$ in `M3_nhosts` by linear predictors in `nhosts` and $T_k$, with and without taxonomic random slope $a_{p(k)} \sim \mathcal{N}(0, \sigma_{\text{tax}}^2)$ across the 7 higher taxa `par.type`, and taxonomic family random slope in sensitivity) **and** (ii) in precision-weighted mixed-effects meta-regression (ML for AICc/parametric-bootstrap LRT; REML with $t_{\text{df}}$ for contrast inference) on the taxon-level temporal slopes $(\hat{\gamma}_k, \widehat{\text{SE}}_k^2)$ extracted from `M2_year`:

1. **`M3_nhosts_tax` (Taxonomic-controlled host-number baseline)**:
   $$\gamma_k = \alpha_0 + \beta_{\text{nh},2}\mathbb{I}(h_k=2) + \beta_{\text{nh},3+}\mathbb{I}(h_k\ge 3) + a_{p(k)} + \varepsilon_k, \qquad a_p \sim \mathcal{N}(0,\sigma_{\text{tax}}^2),\ \varepsilon_k \sim \mathcal{N}(0,\sigma_\gamma^2).$$
2. **`M5_traits_tax` (Theory traits alone + taxonomic random effect)**:
   $$\gamma_k = \alpha_0 + \beta_a T_{k,a} + \beta_b T_{k,b} + \beta_c T_{k,c} + \beta_d T_{k,d} + a_{p(k)} + \varepsilon_k.$$
3. **`M6_joint_notax` (Competing model without taxonomic slope RE — exact Wood Model 3 architecture)**:
   $$\gamma_k = \alpha_0 + \beta_{\text{nh},2}\mathbb{I}(h_k=2) + \beta_{\text{nh},3+}\mathbb{I}(h_k\ge 3) + \beta_a T_{k,a} + \beta_b T_{k,b} + \beta_c T_{k,c} + \beta_d T_{k,d} + \varepsilon_k.$$
4. **`M7_joint_tax` (Primary Registered Competing Model — `nhosts` + 4 traits + taxonomic random effect)**:
   $$\gamma_k = \alpha_0 + \beta_{\text{nh},2}\mathbb{I}(h_k=2) + \beta_{\text{nh},3+}\mathbb{I}(h_k\ge 3) + \beta_a T_{k,a} + \beta_b T_{k,b} + \beta_c T_{k,c} + \beta_d T_{k,d} + a_{p(k)} + \varepsilon_k.$$
5. **`M8_within_3plus` (Within-3+-host test, $n = 44$ taxa, free of any 1/2 vs 3+ host-count or two-vertebrate confound)**:
   $$\gamma_k = \alpha_0 + \beta_a T_{k,a} + \beta_b T_{k,b} + \beta_c T_{k,c} + \beta_d T_{k,d} + a_{p(k)} + \varepsilon_k \quad (h_k \ge 3).$$

### 4.3 Registered decision criteria

*(Note from `pre_analysis_power.py`: at $N = 85$ with 7 taxonomic clusters, asymptotic $\chi^2_4$ LRT has finite-sample Type I error `~0.10`, whereas $\Delta\text{AICc} \le -2.0$ has empirical Type I error `0.049–0.053` and parametric-bootstrap LRT calibrates size at `0.05`. We therefore require parametric-bootstrap LRT + $\Delta\text{AICc}$ for omnibus support, and REML $t$-tests with `df = N - p` confirmed by the full Laplace NB-GLMM for individual contrasts.)*

- **Omnibus test ("Does the theory add predictive value beyond host number?")**:
  - **`supports`**: `M7_joint_tax` improves over `M3_nhosts_tax` by **$\Delta\text{AICc} \le -2.0$ AND parametric-bootstrap LRT $p < 0.05$** ($B = 2,000$ bootstrap draws under `M3_nhosts_tax`), with at least one individually supported contrast (`p < 0.05`) in its predicted direction (also verified in the unified 8,754-row NB-GLMM `M6_joint_notax` vs `M3_nhosts`).
  - **`contradicts`**: $\Delta\text{AICc} \le -2.0$ driven by one or more contrasts (`(a)`, `(b)`, or `(c)`) significant at Holm-adjusted two-sided $p < 0.05$ in the **opposite** direction to the theory.
  - **`not decided`**: $\Delta\text{AICc} > -2.0$ or parametric-bootstrap LRT $p \ge 0.05$.
- **Individual contrasts (`P1/a`: $\beta_a < 0$; `P2/b`: $\beta_b > 0$; `P3/c`: $\beta_c > 0$; `P4/d`: $\beta_d < 0$)** in `M7_joint_tax` (and reported in parallel for `M8_within_3plus`):
  - **`supports`**: estimated slope has the registered sign AND REML two-sided $p < 0.05$ ($|t| \ge t_{\text{df}=78, 0.975} = 1.9908$).
  - **`contradicts`**: estimated slope has the opposite sign AND REML two-sided $p < 0.05$ (for `P4/d`, an opposite positive sign when `storage_stage = 1` is additionally checked against the registered `C1h` `storage * prop_active` interaction before classifying as contradicting `C1b` vs supporting `C1h`).
  - **`not decided`**: REML two-sided $p \ge 0.05$. Both unadjusted and Holm–Bonferroni 4-contrast adjusted $p$-values are reported.

---

## 5. Pre-Analysis Identifiability, Power, Falsifiers, and Indistinguishable Zone

Computed prior to opening any outcome data (`traits_blind_summary.txt`, `pre_analysis_power.txt`):

1. **Collinearity / Identifiability (`VIF` across $N = 85$ taxa)**:
   - In `M6`/`M7` (`nhosts` + 4 traits), **all Variance Inflation Factors are below `1.80`**: `nhosts_linear` `1.787` (`I(nhosts>=3)` `1.534`), `n_fluc_obligate` `1.532`, `storage_stage` `1.380`, `spreading` `1.143`, `prop_active_transfers` `1.538`.
   - Within `nhosts >= 3` ($n = 44$ taxa), `storage_stage` splits **22 vs 22** (`VIF = 1.085`), `n_fluc_obligate` spans `{0: 8, 1: 21, 2: 14, 3: 1}` (`VIF = 1.315`), `spreading` splits `{0: 38, 1: 6}` (`VIF = 1.099`), and `prop_active_transfers` spans `{0.00: 20, 0.25: 1, 0.67: 23}` (`VIF = 1.278`). Every registered contrast is algebraically well-identified even within 3+-host parasites alone.
2. **Minimum Detectable Effects (`MDE_80` at 80% power, two-sided $\alpha = 0.05$, at representative noise $\sigma_\gamma = 0.30, \sigma_{\text{tax}} = 0.15$)**:
   - **`P1 (a) n_fluc_obligate`**: `SE = 0.080`, **`MDE_80 = 0.223` per SD-year (`7.7%/decade` per fluctuating transfer)** across all 85 taxa (`0.277` / `9.6%/decade` within `3+`).
   - **`P2 (b) storage_stage`**: `SE = 0.137`, **`MDE_80 = 0.384` per SD-year (`13.5%/decade`)** across all 85 taxa (`0.434` / `15.4%/decade` within `3+`).
   - **`P3 (c) spreading`**: `SE = 0.136`, **`MDE_80 = 0.382` per SD-year (`13.5%/decade`)** across all 85 taxa (`0.515` / `18.6%/decade` within `3+`).
   - **`P4 (d) prop_active_transfers`**: `SE = 0.172`, **`MDE_80 = 0.483` per SD-year (`17.3%/decade` across `[0, 1]`)** across all 85 taxa (`0.637` / `23.4%/decade` within `3+`).
3. **Monte Carlo Power (`1,000` simulated datasets per scenario, $\sigma_\gamma \in \{0.25, 0.35\}, \sigma_{\text{tax}} \in \{0.10, 0.15\}$)**:
   - **Full-scale theory ($\kappa = 1.0$: $\beta_a=-0.20, \beta_b=+0.30, \beta_c=+0.15, \beta_d=-0.15$, matching Wood et al.'s `-0.35` 3+-host decline scale)**: Omnibus $\Delta\text{AICc} \le -2$ power = **`85.2%–95.0%`**; individual contrast power = **`66.7%–82.7%`** for `(a)`, **`66.8%–76.9%`** for `(b)`, **`20.9%–27.6%`** for `(c)`, and **`18.3%–22.2%`** for `(d)`.
   - **Half-scale theory ($\kappa = 0.5$)**: Omnibus $\Delta\text{AICc} \le -2$ power = **`28.4%–37.3%`**; individual power = `30.1%–35.6%` `(a)`, `27.7%–35.3%` `(b)`, `11.9%–12.2%` `(c)`, `10.4%–12.3%` `(d)`.
   - **Quarter-scale theory ($\kappa = 0.25$)**: Omnibus $\Delta\text{AICc} \le -2$ power = **`8.4%–12.9%`**.
4. **What counts against the theory vs what cannot be distinguished from no effect**:
   - **Counts against**: (i) Significant positive slope on `n_fluc_obligate` ($\hat{\beta}_a \ge +0.16$, $p < 0.05$), or significant negative slope on `storage_stage` ($\hat{\beta}_b \le -0.27$, $p < 0.05$) — e.g., if encysted tissue larvae in fish declined *faster* than unencysted luminal adult worms with the same `nhosts` — or significant negative slope on `spreading` ($\hat{\beta}_c \le -0.27$, $p < 0.05$). (ii) If `nhosts >= 3` retains its full `~ -0.35` effect unattenuated (`< 20%` attenuation) when `(a)–(d)` are entered and the 95% CIs on `(a)` and `(b)` exclude full-scale effects ($|\beta_a| < 0.20$, $\beta_b < 0.30$), that rules out the project's stationary-fluctuation mechanisms as the primary driver of Wood et al.'s century-scale 3+-host decline (pointing instead to secular mean shifts in unmeasured definitive hosts or phenology).
   - **Indistinguishable from no effect**: Any trait effect smaller than `~0.15` per SD-year (`~5%/decade`), and in particular moderate effects (`0.08–0.15` per SD-year) on `spreading` (`P3/c`) and `prop_active_transfers` (`P4/d`), where power is `10%–28%`.

---

## 6. Accounting for Confounding and Biological Processes

Every process listed in `CASESTUDY_SPEC.md` is classified below as **(a) included in model**, **(b) controlled in analysis**, **(c) bounded / sensitivity-tested**, or **(d) cannot be addressed**, with planned sensitivity runs `S1`–`S7`:

| Factor / Process | Handling | Details and Registered Sensitivity Check |
|---|---|---|
| **1. Mean host density (not only variance)** | **(b) / (c) / (d)** | Dissected fish host density (`sfish_density`, 1946–2011 for 6 of 8 fish spp.) is tested in **Sensitivity `S5`** on the subset of years/hosts with fish-density data. Abundances of *non-fish* obligate hosts (copepods, amphipods, benthic snails, seabirds, pinnipeds, sharks) are **unmeasured (d)**: a secular decline in mean apex-predator or snail density could produce decline in `3+`-host taxa without any host-fluctuation effect (`D`). |
| **2. Seasonality and century-scale trend** | **(b) / (d)** | Century trend is the primary estimand $\gamma_k$. Within-year collection month/season is **not** in `final_dataset.csv` (only integer `year` is archived), so seasonal collection shifts across decades **cannot be addressed (d)** except insofar as encysted larvae (`storage_stage = 1`) integrate across seasons while short-lived GI adults fluctuate seasonally. |
| **3. Host memory beyond AR(1) (quasi-cycles, decadal PDO)** | **(c)** | Northeast Pacific hosts experience multi-year PDO/ENSO regimes (`ring-B8a`/`B8b`). In `theory_predictions.py`, varying $\phi$ from `0.0` to `0.75` preserves the directional signs of `(a)`, `(b)`, and `(c)` while triggering the `C1h` crossover for `(d)` when `storage_stage = 1` (**Sensitivity `S4`** tests the `storage_stage * prop_active_transfers` interaction). |
| **4. Temperature and abiotic drivers (SST, hypoxia/pollutants)** | **(c)** | `mech-C3a` splits thermal variability into curvature $J$ and fluctuation loss $D$; secular Puget Sound warming (`temp_na_rm`, 1921–2019) and sediment pollutants (`ord1`, `ord2`, 1880–2005) can shift mean rates directly. **Sensitivity `S5`** refits the trait contrasts controlling for `temp_na_rm` and `ord1`+`ord2`. |
| **5. Whether host scarcity wastes or delays (`d` parameter)** | **(a) / (c)** | Included directly via `prop_active_transfers` (`P4/d`), plus **Sensitivity `S3`** replacing active-penetration `prop_active_transfers` with `prop_waterborne_transfers` (counting passive waterborne stages such as hemiurid cystophorous cercariae and hatched *Anisakis*/*Clavinema* larvae) and binary `has_active_freeliving`. |
| **6. Shape (spread) of stage durations, not only means** | **(c)** | Checked in `theory_predictions.py` (`memoryless` vs `Erlang-4` `chain(4)` reduces storage protection modestly from `66%–74%` to `51%–70%` without changing sign; only all-fixed `CV=0` removes protection, `mech-C2f`). Also **Sensitivity `S2`** tests `storage_alt_mollusc` (giving `0.5` weight to multi-year intramolluscan clonal stages in snails and spiral-valve *Gyrocotyle*). |
| **7. Density dependence in parasite and host** | **(c) / (d)** | Negative-binomial overdispersion $\phi_k$ captures static aggregation; `ring-B15a`/`B15b` shows local equilibrium stability depends on $G = \prod(1-\kappa_k)$, and regulated burden tracks $\ln R_0^*$. Nonlinear density-dependent fecundity/mortality parameters are unmeasured **(d)**. |
| **8. Host feedback (parasite regulating host numbers)** | **(d)** | `studies.E4` (`R11`) shows predator–prey–parasite feedback can destabilize or shift equilibria when parasite-induced host mortality $\alpha$ is strong; unmeasured in museum dissections **(d)**. |
| **9. Alternative hosts and spatial spread / immigration** | **(a) / (b) / (c)** | Alternative/paratenic hosts included via `spreading` (`P3/c`), with **Sensitivity `S1`** testing `spreading_broad` (including resolved genera with $\ge 50$ GloBI or $\ge 15$ Benesh hosts). Spatial variation within Puget Sound is controlled via taxon-specific latitude slopes $\beta_{\text{lat},k} X_{\text{lat},i}$ **(b)**; open-ocean immigration of migratory hake/herring/marine mammals is discussed as a bound **(c)** (`mech-C7a` asynchronous patch dispersal can turn host variance into a net gain!). |
| **10. Demographic noise at small numbers & shared host death** | **(c)** | `studies.R18` shows parasite aggregation in packets and shared host mortality raise extinction probability at equal $D$, though longer steady stays still lower extinction (`44/54` pairs). `foo >= 0.04` filter excludes ultra-rare vagrants; zero-rich rare taxa (`10` taxa absent for $\ge 40$ yr) are retained in the exact Negative Binomial likelihood. |
| **11. Observation, preservation, and sampling error** | **(b) / (c)** | Host body length (`X_len` per host–parasite pair), collection latitude (`X_lat` per parasite), individual fish random effect ($\zeta_m$ across co-dissected taxa), and taxon-specific NB dispersion $\phi_k$ are **controlled in every model (b)**. Fiorenza et al. (2020) tested fluid-preservation fixation effects experimentally. **Sensitivity `S6`** restricts to the `49` taxonomically genus/species-resolved taxa (`resolved_taxon == 1`) to ensure unresolved morphotaxa (`sp.`) do not bias trait slopes. |
| **12. Evolution during the 140-year span** | **(d)** | Host resistance or parasite life-history evolution over 1880–2019 cannot be separated from ecological abundance change in preserved specimens **(d)**. |
| **13. Confounding with taxonomy, dissected life stage (`terminal`), and two-vertebrate hosts** | **(b) / (c)** | Phylogenetic/taxonomic non-independence across the 7 higher taxa (`par.type`) is **controlled in `M7_joint_tax` (b)** and extended to taxonomic family (`27` families) in **Sensitivity `S7`**. Confounding with `nhosts >= 3` (all of which use $\ge 2$ vertebrates in Wood et al.) is eliminated in **`M8_within_3plus` (b)**. Confounding between `storage_stage` and being sampled as a larva in the dissected fish (`terminal == 0`) is tested explicitly in **Sensitivity `S2b`**. |


## Dated implementation amendment — 2026-11-14 23:46:49 CST

The unfinished helper exponentiates the dispersion standard deviation twice. The reference C++ uses `logsigma_phi = exp(loglogsigma_phi)` as the standard deviation of `logphi`. A new helper corrects this implementation error without changing a registered model or criterion. Before fitting, negative-binomial likelihood/derivatives must agree with SciPy and finite differences to absolute error 1e-5; the dispersion transform must agree to 1e-12. Inner gradients and outer gradients and Hessian eigenvalues will be reported; an iteration limit alone does not count as convergence. Original helpers and locked artifacts remain unchanged.


## Weather and environment amendment — 2026-11-14 23:49:53 CST (registered before environmental analysis)

**Scope and temporal precision.** The archived field records give a year and usually coordinates; no collection month or day. Export one record per unique fish with the original site, coordinates, year interval January–December, and location uncertainty marked unreported. Unknown coordinates remain unknown. Never substitute exact dates or silently impute sites.

**Sources, chosen by marine relevance and coverage before outcome comparison.** Fetch NOAA ERSST version 6 monthly SST (2-degree grid, 1850 onward) from the public NOAA PSL NetCDF subset service for 1880–2019 and the coastal rectangle 46–50 N, 122–128 W. Assign the geographically nearest valid ocean cell to each known coordinate, report great-circle distance, and retain all available monthly values. These are offshore/regional proxies; coarse cells do not resolve Puget Sound. For unknown coordinates leave the ERSST match missing. Supplement with the original archived Race Rocks SST (`temp_na_rm`, coastal point near 48.2996 N, 123.5318 W), and fetch the station's openly published monthly archive to independently document its seasonal range. Record any failed request. DFO's relocated station description is public; its linked Open Canada dataset returned HTTP 403 in the browser and will not be circumvented. Daymet/TerraClimate are land products and do not measure seawater; NEON is not applicable; OISST begins too late for the century-scale contrast. No account-based reanalysis.

**Covariates and mechanism.** Primary new adjustment: annual mean ERSST (all 12 months required) and within-year SD of monthly SST (seasonal amplitude, not daily variability). Both can affect development and survival; shared temperature can confound any trait–decline association. Secondary adjustment: archived Race Rocks annual SST, sediment ordination axes `ord1`, `ord2` and standardized fish density on available subsets, corresponding to original S5. Retain year and length/latitude/specimen structure; enter environmental covariates as additive fixed effects by host-number group to permit different thermal responses. Compare otherwise identical trait and host-number-only models on the same rows. No outcome-driven variable selection. Report fits with and without each environment set. Environmental results remain amendments and never overwrite the original decision.

**Associations and host variability.** Calculate Pearson correlations on unique fish-year/species records (and unique years for region-wide series), plus correlations of annual fish-species mean log(1+count) with each environmental series. These descriptive correlations do not test causality. For the archived host-density index, report correlations with its level and absolute first differences within species at consecutive observed years; this is only a change-magnitude proxy. Standardized, interpolated fish density cannot recover log-abundance variance of the obligate host guilds, and no such measurement will be claimed. Avoid treating repeated parasite rows as independent environmental samples. Obtain descriptive 95% intervals by resampling years (1,000 draws, seed 20261115); cross-year dependence remains a limitation.

**Identifiability before outcomes.** After fetching covariates but before their association with parasite counts, save matched fish/years, missingness, year correlations, and design ranks/condition numbers. Simulate slope noise with SD 0.30 for the count-weighted design, four illustrative true environmental coefficients 0, 0.1, 0.3, 0.5 per standardized covariate, 1,000 replicates, seed 20261115. This is a design diagnostic, not calibrated NB-GLMM power; number of unique years, not parasite rows, limits independent thermal information. If year plus environment is rank deficient, report non-identifiability and do not drop terms to obtain significance.

**Temperature as an input.** Use the exponential development-rate relation in `notes/mechanism-temperature-incubation.md` (EXP family): nu(T)/nu(10)=Q10^((T-10)/10), equivalently tau(T)=tau(10)*Q10^(-(T-10)/10). The Q10 values 1.5, 2, 3 are illustrative sensitivity settings, not measured for these taxa. For the registered no-storage/storage archetypes, rescale the exposed stays (baseline 2 and 3 months), leave the illustrative storage stay (6 or 24 months) fixed, and calculate D2 with host SD 0.6 and memory 0 or 0.75 using monthly ERSST temperatures. Verify formula against the engine to 1e-10 before use. Evaluate mean-temperature and mean-monthly-rate durations separately (nonlinearity); record that death, host dynamics and ectotherm storage temperatures are not calibrated. Also rescale every stage as an alternative sensitivity. This is an illustration of possible timing shifts, not taxon-specific prediction or evidence for the fitted trait coefficients.

**What changes the conclusion.** Apply the original signs and decision thresholds to complete amended comparisons; a change in classification is reported as environmental sensitivity, not replacement of the original result. A shift of at least 20% in the host-number or either primary trait coefficient is a material confounding warning (use absolute shift if the original coefficient is near zero). If a full amended omnibus bootstrap is not completed, no amended supports/contradicts verdict is assigned. Thermal illustration changes the mechanistic ordering only if storage has greater D2 than the matched no-storage case or the effect of extra exposed transfers reverses. No finite-noise or causal claim follows from D2 alone.

**Budget and stop.** One data fetch per source (cache successes), the declared covariate sets and Q10 grid only; one report of all matches, correlations, adjustments and limitations. No exploratory search over lags, thresholds or outcomes. Preserve any failed first output with an attempt suffix.


## Dated audit note — 2026-11-15 00:04:35 CST

The locked family column contains 33 distinct labels, including 5 unresolved taxonomic bins; the protocol's prose count of 27 was inaccurate. S7 uses the locked labels unchanged. Seven `terminal_in_fish` codes are missing, so S2b uses complete cases. These are implementation disclosures, not changed criteria.

The refetched Race Rocks HTML table contains 99.9 degrees C for July 1966. This physically invalid surface-seawater value affects an unfiltered source-comparison summary, not any registered fit (which uses the original archived annual temperature) or the ERSST thermal illustration. Retain that first comparison. A separately labelled post hoc quality-control comparison will treat surface SST outside [-2,40] C as missing and continue requiring at least 11 valid months. This does not replace any outcome or input of a registered fit.


## Dated identifiability clarification — 2026-11-15 00:20:34 CST

The initial prose saying all VIFs in M6/M7 were below 1.80 used the inherited summary of separate linear-host-count and single-3+-indicator designs. The actual categorical host-number design is full rank, with VIFs saved in results/categorical_design_audit.json: approximately 2.93 and 3.01 for the two host contrasts and 2.16 for active-transfer fraction. The fluctuation/storage/spreading VIFs remain below 1.54. The pre-analysis power code already used both categorical indicators, so its saved results and every registered criterion remain unchanged. This clarification was made after analysis and is not a fresh preregistration.
