# Protocol: `microcosm` — Experimental Microcosms of Spore-Forming and Viral Microparasites Under Varying Host Availability

- **Study ID**: `microcosm` (`casestudies/microcosm/`)
- **Registered timestamp (`date`)**: `2026-11-15T04:15:30Z` (`Sat Nov 14 22:15:30 CST 2026`), written and saved to disk after running the data-free benchmark and power script (`results/preanalysis_power_20261115.txt` at `2026-11-15T04:14:55Z`) and **before** opening any data rows or outcome values in `data/raw/microcosm/`.
- **Pre-protocol inspection boundary**: Only repository API metadata, dataset `README` texts, supplementary model parameter appendices, and row-0 column headers / row counts were inspected prior to writing this protocol. Inevitable exposure to published headline prose in paper abstracts/appendices during literature verification is disclosed explicitly per dataset in Section 3.

---

## 1. Question

Do open experimental microcosm datasets in which host availability (or the environmental resource/thermal schedule driving host density) varied over time support the theory's quantitative predictions for:
1. **Fluctuation cost at fixed mean (`H1`)**: whether fluctuating host availability lowers rare parasite growth ($\ln R_0^* = \ln R_0 - D$, $D > 0$), with size governed by stage residence times ($\tau_{\text{out}}, \tau_{\text{in}}$), stay-time regularity ($k_{\text{out}}, k_{\text{in}}$), host autocorrelation ($\phi$), and whether host scarcity wastes ($d = 1$) or delays ($0 \le d < 1$) the free-living transmission stage;
2. **Memory and spatial asynchrony (`B2`, `C1`, `C7a`)**: whether positive temporal autocorrelation (red vs. white noise at matched mean and variance) raises the single-patch rare-invasion loss $D$ while enabling spatial buffering across asynchronous connected patches; and
3. **Empirical calibration of candidate microcosm systems**: where experimental designs cannot isolate temporal host variability at a fixed mean, what exact values do they supply for free-living spore survival ($\mu_{\text{out}}$ and stay shape), within-host development ($\tau_{\text{in}}$ and Erlang shape $k_{\text{in}}$), per-host encounter/filtering rates, contact-limitation mortality fraction $d = \mu_{\text{out}} / (\mu_{\text{out}} + c \bar{H})$, and baseline $R_0(\bar{H})$?

---

## 2. Theoretical Predictions, Formulas, and Engine Calls

Write $h(t) = H(t) / \bar{H}$ for relative host abundance (mean $1$ over the stationary law or full periodic cycle), $\tau_{\text{out}}$ for mean residence time of the free-living stage at mean host density, $\tau_{\text{in}}$ for mean infectious/developmental residence time inside the host, and $\mathcal{T} = \tau_{\text{out}} + \tau_{\text{in}}$ for threshold generation time.

1. **Continuous-time periodic schedules (exact Floquet threshold loss $D$)**:
   For a two-stage Erlang chain ($k_{\text{out}}$ sub-stages outside, $k_{\text{in}}$ sub-stages in hosts) under a periodic host schedule $h(t)$ of period $P$ (piecewise constant or step-wise across transfers), the instant generator $Q(h, y)$ has exit rate $r_{\text{out}} = k_{\text{out}}/\tau_{\text{out}}$ per outside sub-stage (with last sub-stage leaving at rate $r_{\text{out}}$ for **gain-only waste** $d = 1$, or $r_{\text{out}}[d + (1 - d)h]$ for **contact-limited waiting** $0 \le d < 1$, where $d = \mu_{\text{out}} / (\mu_{\text{out}} + c\bar{H})$ is the threshold fraction of searching stages dying before host contact), entry to hosts $r_{\text{out}} e^y h$, and host sub-stage progression $r_{\text{in}} = k_{\text{in}}/\tau_{\text{in}}$ (`notes/decisive-experiment-v2.md` §3; `notes/contact-limited-transfer.md` §7). Over one period $P = \sum_m \Delta t_m$, the monodromy matrix is $M(y) = \prod_m \exp(Q(h_m, y)\Delta t_m)$, Floquet growth is $r_{\text{Floq}}(y) = P^{-1}\ln\rho(M(y))$, and the exact threshold loss $D$ solves $r_{\text{Floq}}(D) = 0$.
   - *Key limit (`Statement ring-B9a`)*: When $\tau_{\text{in}} \to 0$ relative to $\tau_{\text{out}}$ (one-stage limit, e.g., fast-lysis bacteriophage), $D \to 0$ at every amplitude and period under proportional encounter; any observed loss at equal mean resource then comes from mean-host shift $\Delta \ln \bar{H} < 0$ (concave host growth) or saturating encounter curvature, plus a small two-stage residual.

2. **Stochastic schedules with geometric memory $\phi$ (closed forms + simulation + `lifecycle_engine.py`)**:
   - *Discrete second-order closed form (`Proposition C1`, `notes/contact-limited-transfer.md` eq. C1; reduces to `Statement ring-B2` eq. ring.5 at $d = 1$)*:
     $$\Gamma_c = \sigma^2 \left\{ d^2 \frac{1 + 2\theta}{2\mathcal{T}} - d \frac{R - (1 - d)q_{\text{out}}}{1 - R} \right\}, \qquad \theta = \frac{\phi}{1 - \phi}, \quad q_k = \frac{1}{1 + \tau_k/\theta}, \quad R = \prod_k q_k.$$
   - *Continuous-time second-order closed form (`C2`, `notes/contact-limited-transfer.md` §7)*:
     $$\Gamma_c^{\text{ct}} = \sigma^2 \left\{ d^2 \frac{\theta_{\text{ct}}}{\mathcal{T}} - d \frac{R - (1 - d)q_{\text{out}}}{1 - R} \right\}.$$
   - *Finite-noise daily Markov / AR(1) schedules and multi-patch / bypass extensions*: Computed both by continuous-time daily matrix-exponential propagation (`markov_daily_loss_D`) and by `studies/engine-R16/lifecycle_engine.py` (`Engine.coefficient()` and `Engine.loss()`, noting the `Driver.levels()` raw-`low` caveat documented in `verification/engine-R16-check.md`: we use `low=None` symmetric two-state drivers when $\sigma$ permits and report exact `levels()` alongside lognormal drivers for `d = 1`).
   - *Sign predictions for `Study 2` (*Paramecium*–*Holospora*, $\tau_{\text{out}} \approx 2\,\text{d}, \tau_{\text{in}} \approx 7\,\text{d}$)*:
     - **Single-patch / synchronous memory contrast**: For $0 < d \le 1$ and $\phi_{\text{red}} = 0.8 > \phi_{\text{white}} = 0$, $D_{\text{red, sync}} > D_{\text{white, sync}}$ across both memoryless and Erlang stays (`preanalysis_power.py` §2: $+0.0501$ vs. $+0.0168$ at $d=1$ memoryless; $+0.1212$ vs. $+0.0211$ for Erlang-4; engine $2$-patch synchronous $+0.0925$ vs. $+0.0251$). Adding a vertical-transmission bypass share $w_{\text{vert}} \in (0, 1)$ reduces absolute $D$ (`Statement C4g`) without reversing $D_{\text{red, sync}} > D_{\text{white, sync}}$.
     - **Two-patch asynchronous buffering (`Statement C7a`)**: Under `Patches(count=2, synchrony=0.0, mobile_stage='host', movement=0.0667)`, spatial averaging strongly reduces the red-noise loss ($D$ drops from $+0.0925 \pm 0.0030$ in `Red | Sync | High` to $-0.0042 \pm 0.0014$ in `Red | Async | High`, $\Delta D = +0.0967$), compared with a much smaller reduction under white noise ($+0.0251$ to $+0.0125$, $\Delta D = +0.0126$).

---

## 3. Verified Candidate Datasets, Provenance, and Roles

All files were fetched by `python3 -I data/fetch/microcosm.py` into `data/raw/microcosm/` (verified git-ignored; SHA-256 hashes, licences, versions, DOIs, and UTC timestamps recorded in `data/raw/microcosm/MANIFEST.json`).

| ID | Study & System | Source, DOI, Licence, Version | One Record & Sample Size | Design Role & Prior Exposure Disclosure |
|---|---|---|---|---|
| **S1** | **Ferris et al. (2020)** *Proc. R. Soc. B* 287:20200787 — *Pseudomonas fluorescens* SBW25 + lytic phage SBW25$\Phi$2 | Figshare `12283799` & `12283802`, `doi:10.6084/m9.figshare.12283799.v1`, **CC BY 4.0**, v1 | 1 microcosm $\times$ transfer (`Population Dynamics`: 270 rows = 5 arms $\times$ 6 reps $\times$ 9 transfers `0..24`; `Growth Data`: 338 rows) | **Primary Test T1** (periodic resource oscillation amplitudes `0, 0.5, 0.8, 0.9, 0.98` at **exact arithmetic mean `0.50 KB`**). *Disclosed*: paper text & Appendix A.2 reported phage densities and survival were not significantly related to resource amplitude ($p = 0.27$, $p = 0.85$). Raw `.xlsx` values unread before protocol. |
| **S2** | **Duncan, Gonzalez & Kaltz (2013)** *Proc. R. Soc. B* 280:20131747 — *Paramecium caudatum* + *Holospora undulata* | Zenodo `4949969` (official `"dryad"` mirror of `doi:10.5061/dryad.3hq2c`), **CC0 1.0**, v1 | 1 metapopulation $\times$ day (`Paramecium_Time`: 1,440 rows = 96 metapops $\times$ 15 days; `Parasite_time`: 384 rows = 48 infected $\times$ 8 days) | **Primary Test T2** (white $\phi=0$ vs. red $\phi=0.8$ noise $\times$ sync/async $\times$ high/low migration at **matched driver mean & variance**, with 48 uninfected controls). *Disclosed*: `sweep2/duncan2013.md` reported higher Day-56 prevalence and later peaks under red noise, and $p > 0.77$ for synchrony; **early rare-invasion growth (Days 0–16)** was never analysed. Raw `.xlsx` values unread before protocol. |
| **S3** | **Dallas, Krkošek & Drake (2018)** *R. Soc. Open Sci.* 5:171975 — *Daphnia dentifera* + *Metschnikowia bicuspidata* | Figshare `5133001` & `5745573`, `doi:10.6084/m9.figshare.5133001.v1`, **CC BY 4.0**, v1 | 1 microcosm $\times$ census (`expData.csv`: 1,800 rows = 90 microcosms across 6 initial host densities `20..640/L`, $n=15$/arm $\times$ 20 censuses) | **Primary Calibration + Secondary Observational Check T3** (calibrates $H_c$, per-host transmission $\beta$, and within-microcosm $\sigma_{\ln H}, r_1$; cannot isolate temporal variance at fixed mean). *Disclosed*: paper abstract reported clear threshold between 80 and 160 hosts/L. Raw `.csv` unread before protocol. |
| **S4** | **Dallas & Drake (2016)** *Ecol. Evol.* 6:7931–7938 — *Daphnia laevis* + *M. bicuspidata* | Figshare `3615816`, `doi:10.6084/m9.figshare.3615816.v1`, **CC BY 4.0**, v1 | `sporeSurv.csv` (800 FOV rows over 19 d); `filterRate.csv` (85 hosts); `indivExpData0..4.csv` (320 hosts) | **Parameter Extraction** (cannot test multi-generation host density fluctuation; directly measures free-spore mortality $\mu_{\text{out}}$ & stay shape, filtering rate $F$, host death time $\tau_{\text{in}}$ & CV, and contact parameter $d$). |
| **S5** | **Clerc, Ebert & Hall (2015)** (`doi:10.5061/dryad.n1ks4`) & **Tadiri (2023)** (`doi:10.5061/dryad.8931zcrwp`) — *Daphnia magna* + *Pasteuria ramosa* | Zenodo `5011860` & `8169416` (official `"dryad"` mirrors), **CC0 1.0**, v1 | `clerc_exp_1.csv` (297 sacrificed hosts across DPI); `temp.csv` (300 epidemic jars across 5 constant temperatures, days 0–76) | **Parameter Extraction** (Ebert et al. 2000 / Pulkkinen & Ebert 2004 have no open deposits; S5 supplies empirical *Pasteuria* within-host maturation duration, CV, spore load, and thermal `tti` + jar host $\sigma_{\ln H}, r_1$). |
| **S6** | **Davenport et al. (2024a)** *Ecology* `10.1002/ecy.4235` — 6 Michigan lakes, *Daphnia*/*Ceriodaphnia* + 5 spore-forming parasites | Zenodo `10359990` (official `"dryad"` mirror of `doi:10.5061/dryad.tqjq2bw5m`), **CC0 1.0**, v1 | `Clean_Data_2021_Lakes_All_Host_Densities.csv` (576 lake-date-species rows) + spore dPCR CSVs + `AvgDaysBetween...csv` (13 outbreaks) | **Parameter Extraction** (field observational single season; supplies biweekly host $\sigma_{\ln H}, r_1$, waterborne spore concentrations, and spore-to-outbreak lags across 5 parasites). |
| **S7** | **Krichel et al. (2023)** *PLOS Biol.* 21:e3002260 — *D. magna* + *Ordospora colligata* | Figshare `24111775` & `24111778`, **CC BY 4.0**, v1 | `s016.csv` (12,565 dissected hosts: 12 pops $\times$ 76 obs days, 12 hosts/day); `s017.csv` (229 daily temps) | **Confounded Thermal + Immigration Contrast** (host density unrecorded; 3 infected hosts added every 3 d preventing fade-out; documents thermal-acclimation + immigration reversal). |
| **S8** | **Harrison et al. (2013)** (`Zenodo 4957209` = `dryad.8k015`, CC0); **Vercken et al. (2015)** (`PMC4523363`); **Dryad HTTP-401 sets** (`s7h44j1gb`, `nvx0k6f7r`, `qjq2bvqrt`) | Verified via APIs on 2026-11-15 | Harrison `RAW DATA.xlsx` holds resistance/range assays only (no density time series); Vercken has no deposited data; 3 recent Dryad sets return HTTP 401 with no Zenodo data mirror | **Documented Verification & System Comparison** (`report.md` and `systems.md`). |

---

## 4. Accounting for Every Confounding and Modifying Process

Classification codes required by `CASESTUDY_SPEC.md`:
- **(a)** Included explicitly in the theoretical model / engine calculation;
- **(b)** Controlled for statistically or by experimental design in the analysis;
- **(c)** Bounded only via sensitivity range;
- **(d)** Cannot be addressed with the available data.

| # | Process / Factor | Study S1 (Ferris Phage) | Study S2 (Duncan *Holospora*) | Study S3/S4 (Dallas *Metschnikowia*) | Sensitivity / Implementation Plan |
|---|---|---|---|---|---|
| 1 | **Mean host density $\bar{H}$ (not only variance)** | **(a) & (b)** Measured across even/odd transfers; separated into mean-host shift $\Delta\ln\bar{H}$ vs. Floquet $D$ | **(b)** Measured directly in 48 uninfected & 48 infected metapops; regression adjustment for $\ln\bar{H}$ | **(b)** Controlled explicitly as covariate $\ln\bar{H}$ across the 6 initial density arms | Compute both unadjusted treatment contrasts and $\ln\bar{H}$-adjusted contrasts. |
| 2 | **Seasonality and secular trend** | **(b)** Strictly periodic 2-transfer square wave in incubator; detrended vs. raw host variance compared | **(b)** Stationary 56-d 28/28-d balanced binary temperature sequences; early (0–16 d) vs. late (48–56 d) separated | **(b)** Single epidemic wave over ~50 d; primary invasion window (day $\le 11$) separated from full wave | Report raw and linearly detrended $\sigma_{\ln H}$ and $r_1$. |
| 3 | **Host memory beyond AR(1)** | **(a)** Exact periodic Floquet monodromy matrix (no AR(1) approximation) | **(c)** Imposed driver is binary Markov ($\phi \in \{0, 0.8\}$); empirical host $r_1, r_2$ checked in uninfected series | **(c)** Empirical lag-1 and lag-2 autocorrelations measured per microcosm | Compare AR(1)/Markov engine loss against Floquet and empirical lag structure. |
| 4 | **Temperature & abiotic drivers of parasite rates** | **(b)** Constant $28\,^\circ\text{C}$ across all 5 arms (only KB nutrient oscillated) | **(a) & (c)** $5\,^\circ\text{C}$ arrests *Holospora* development as well as depressing host growth; modelled both as host-only (M2A) and joint thermal-arrest + host (M2B) | **(b)** S3 held at constant temp/food; S4 manipulates diurnal thermal pulses and directly measures spore survival & filtering | Run M2A (host-driven only) vs. M2B (joint $5\,^\circ\text{C}$ developmental arrest) in S2; extract S7 thermal contrast. |
| 5 | **Whether host scarcity wastes ($d=1$) or delays ($0 \le d < 1$) waiting** | **(a) & (c)** Sweep $d \in \{1.0, 0.5, 0.1, 0.03\}$ (48-h 99% serial dilution sets baseline waste floor, fast adsorption lowers $d$) | **(a) & (c)** Sweep $d \in \{1.0, 0.5, 0.1\}$ in both C1/C2 and Markov simulations | **(a) & (b)** Estimate $d = \mu_{\text{out}}/(\mu_{\text{out}} + F\bar{H})$ directly from S4 `sporeSurv.csv` + `filterRate.csv`! | Report both gain-only ($d=1$) and contact-limited ($d < 1$) predictions side by side for every system. |
| 6 | **Shape (spread) of stage durations** | **(a) & (c)** Compare memoryless `(1,1)` vs. Erlang `(1,16)` for narrow phage lysis time | **(a) & (c)** Compare memoryless `(1,1)` vs. Erlang `(4,4)` and `(1,16)` for ~7-d *Holospora* latency | **(a) & (b)** Fit empirical Weibull/Erlang shape to S4 `sporeSurv.csv` and infected host `death.day` ( plus S5 Clerc *Pasteuria*) | Include Erlang sub-stage chains ($k = 1, 4, 16$) in continuous-time Floquet/Markov and `lifecycle_engine.py`. |
| 7 | **Density dependence in parasite and host** | **(c)** Inoculated at high density ($10^7\text{ PFU/ml}$); carrying capacity compresses abundance ratios except near fade-out | **(b)** Day 0 starts rare ($\sim 2.5\%$ prevalence), so Days 0–16 probe near-linear rare invasion while Days 32–56 are density-dependent | **(b)** Day 0 starts from $10\text{ spores/ml}$ in naive hosts; primary/early invasion probes rare regime | Pre-register early rare-growth window (`T1b` Transfers 3–6; `T2a` Days 0–16) separately from late endemic endpoints. |
| 8 | **Host feedback (parasite changing host numbers)** | **(c)** No uninfected phage-free control in S1 `Population Dynamics`; host growth without phage calibrated from `Growth Data` | **(b)** Exact 1:1 matched uninfected control metapopulations ($n = 48$) for every treatment cell! | **(c)** No uninfected tubes in S3; host crash feedback modelled in `T3a` power check | In S2, estimate unconfounded host $(\bar{H}, \sigma_{\ln H}, r_1)$ from uninfected controls and quantify parasite suppression of $\bar{H}$. |
| 9 | **Alternative hosts and spatial spread / immigration** | **(b)** Single clonal host, closed tubes (no immigration) | **(a) & (b)** Single host, factorial 2-patch island dispersal (`Low` $0.67\%$ vs. `High` $6.67\%$) $\times$ `Sync`/`Async` modelled in `lifecycle_engine.py`; weekly 50-host immigration noted | **(b)** Single clonal host, closed 50-ml tubes in S3 (S7 has 3 infected + 9 susceptible added every 3 d, disqualifying S7 as a threshold test) | Explicitly model 2-patch dispersal in S2 (`T2c`) and vertical-transmission route bypass (`Statement C4g`). |
| 10 | **Demographic noise at small numbers** | **(c)** Severe 1% serial-transfer bottleneck when $\text{PFU/ml} \le 10^3$ ($< 60$ virions transferred in $60\,\mu\text{l}$) | **(c)** Moderate (~7,500 initial hosts/metapop, ~180 infected at $2.5\%$) | **(a) & (b)** Dominant at $H_0 = 1, 2, 4$ hosts/tube ($20\text{--}80\text{ hosts/L}$); quantified via stochastic invasion curve | Account for bottlenecks and whole-number extinction in S1 and S3. |
| 11 | **Observation and sampling error** | **(b)** Detection limit $200\text{ PFU/ml}$ and transient zero recovery (`Phage amplified later?`) tracked explicitly | **(a) & (b)** Binomial testing error on $18\text{--}110$ dissected *Paramecium* per sample included in `T2a` power and weighted analyses | **(b)** Complete whole-tube census every 3–4 d under low light in S3 (minimal sampling error; early pre-day-11 infections visually undetectable) | Propagate binomial sampling error in S2 and detection-limit censoring in S1. |
| 12 | **Evolution during the study (resistance, adaptation)** | **(b) & (c)** Strong coevolution documented in `RBG Data`; controlled partly by restricting `T1b` to Transfers 3–6 and measuring RBG shift | **(c)** Clonal host VEN over 56 d (~30–50 host divisions); resistance evolution minor over 16 d, possible by 56 d | **(b)** Single parthenogenetic *D. dentifera* clone in S3; no genetic variation for resistance evolution | Restrict primary growth tests to early pre-coevolution windows (`T1b`, `T2a`) and report `RBG` evolution in S1. |
| 13 | **Confounding between treatments** | **(b)** Mean KB resource strictly matched (`0.50 KB`), temperature/shaking identical; only host nonlinear growth shifts $\bar{H}$ | **(c)** Variable treatments have matched mean ($14\,^\circ\text{C}$) and variance, but constant controls sit at extremes ($5, 23\,^\circ\text{C}$), not $14\,^\circ\text{C}$ | **(d)** In S6 (Michigan lakes), lake morphometry, predators, and community composition covary across lakes | Never compare S2 variable arms to $23\,^\circ\text{C}$ as if $23\,^\circ\text{C}$ were the mean; restrict S2 causal test to `Red` vs. `White` $\times$ `Sync`/`Async`. |

---

## 5. Pre-Registered Estimands, Tests, Criteria, and Pre-Computed Power

All power and false-positive figures below were computed prior to data inspection by `preanalysis_power.py` (`results/preanalysis_power_20261115.txt`, 4,000 Monte Carlo trials per scenario; benchmark checks `B0a`–`B0d` all passed within pre-stated tolerances).

### Test Group T1 — Ferris et al. (2020): Periodic Resource Oscillation Amplitudes at Fixed Mean (`N = 30` microcosms, `5` arms $\times$ `6` replicates)
- **Data calibration step (non-outcome)**: From `Population Dynamics` (`Bacteria Density`), compute realized high-phase (`Transfers 3, 9, 15, 21` vs. `6, 12, 18, 24`, matching phase parity from `Growth Data` and paper Fig. 1a) and low-phase bacterial densities, arithmetic mean $\bar{H}_{\text{trt}}$, log-SD $\sigma_{\ln H}$, and exact continuous-time Floquet loss $D_{\text{trt}}$ for both gain-only ($d=1$) and contact-limited ($d \in \{0.5, 0.1, 0.03\}$) phage cycles, plus mean-host shift $\Delta\ln\bar{H}_{\text{trt}} = \ln(\bar{H}_0 / \bar{H}_{\text{trt}})$.
- **Primary estimands & criteria**:
  - **T1a (Phage persistence)**: One-sided Spearman rank correlation $\rho_S$ across $N = 30$ microcosms between nominal oscillation amplitude $\delta_{\text{res}} \in \{0, 0.50, 0.80, 0.90, 0.98\}$ (and monotonically ordered $D_{\text{trt}}$) and terminal phage persistence transfer $T_{\text{surv}} \in \{3, 6, \dots, 24\}$ (the last transfer with detectable PFU or `Phage amplified later? == Yes`; permanent extinction scored at first unrecovered zero).
    - *Criterion*: **Supports** a detectable fluctuation persistence cost if $\rho_S < 0$ with exact permutation one-sided $p < 0.05$; **contradicts** if $\rho_S > 0$ with opposite one-sided $p < 0.05$; **not decided** otherwise.
  - **T1b (Early two-phase mean phage abundance)**: Ordinary least-squares slope $\beta_1$ of early two-phase mean $\bar{y}_{3,6} = \frac{1}{2}[\ln(\text{PFU}_3 + 1) + \ln(\text{PFU}_6 + 1)]$ on predicted total loss $D_{\text{trt}}$ across the $N = 30$ microcosms (with secondary $\ln\bar{H}$-adjusted slope `T1b-adj`).
    - *Criterion*: **Supports** if $\beta_1 < 0$ with one-sided permutation $p < 0.05$; **contradicts** if $\beta_1 > 0$ with opposite one-sided $p < 0.05$; **not decided** otherwise.
- **Pre-computed power & identifiability (`results/preanalysis_power_20261115.txt` §1)**:
  - Because phage intracellular lysis time ($\tau_{\text{in}} \approx 0.03\,\text{d}$) is tiny beside free-virion residence ($\tau_{\text{out}} \approx 0.35\,\text{d}$), the system sits near the continuous-time one-stage zero-loss limit (`Statement ring-B9a`): even at maximum host swing $\pm 92\%$, predicted Floquet $D$ is only **$0.0661$** under gain-only ($d=1$) and **$0.0021\text{--}0.0069$** under realistic fast-adsorption contact limitation ($d = 0.03\text{--}0.10$).
  - Consequently, pre-data power at $N = 30$ is **7.9%** (`T1a`) and **7.4%–12.0%** (`T1b`) under gain-only ($d=1$), and **5.0%–5.6%** (indistinguishable from the **4.7%–5.2%** null rate) under contact limitation ($d \le 0.10$).
  - *What counts against the theory vs. indistinguishable from no effect*: A **not decided** outcome in S1 is **predicted by the continuous-time phage theory itself** (`D <= 0.066`, power $\le 12\%$) and cannot distinguish small positive $D$ from $D = 0$. Conversely, a large positive surge in phage persistence/density with amplitude ($\rho_S > 0$ or $\beta_1 > 0$, $p < 0.05$ after accounting for $\bar{H}$), OR an observed threshold loss exceeding the gain-only bound $-\frac{1}{2}[\ln(1+\delta_H)+\ln(1-\delta_H)]$, would count against the model.

### Test Group T2 — Duncan, Gonzalez & Kaltz (2013): Red vs. White Noise and Spatial Asynchrony (`N = 32` variable infected + `32` variable uninfected metapopulations)
- **Host schedule calibration step**: From the `32` variable **Uninfected** metapopulations (`Duncan_Paramecium_Time_7_13.xlsx`, `Infection == Uninfected`), estimate unconfounded mean host abundance $\bar{H}$, within-series $\sigma_{\ln H}$, and 4-day lag-1 autocorrelation $r_1$ by noise colour (`White` vs. `Red`) and synchrony (`Sync` vs. `Async`). Compute predicted $D$ in `lifecycle_engine.py` and continuous-time Markov switching across both gain-only ($d=1$) and contact-limited ($d=0.5, 0.1$), memoryless vs. Erlang-4 stays, vertical-transmission shares $w_{\text{vert}} \in \{0, 0.3, 0.6\}$, and joint $5\,^\circ\text{C}$ developmental arrest.
- **Primary estimands & criteria**:
  - **T2a (Primary rare-invasion growth contrast: Red vs. White, Days 0 to 16)**: For each of the $32$ variable infected metapopulations, compute early per-day rare-invasion growth rate of prevalence $r_{\text{prev},0\to 16} = \frac{1}{16}[\ln(\frac{k_{16}+0.5}{n_{16}+1}) - \ln(\frac{k_0+0.5}{n_0+1})]$ (and infected host abundance growth $r_{I,0\to 16}$ combining `Paramecium` total and prevalence), where $k_t, n_t$ are `Number infected` and `Total` at Day $t$. Test `Red` ($n=16$) minus `White` ($n=16$), stratified by the 4 `Synchrony × Migration` blocks ($10,000$ exact block permutations; also reported separately within `Synchronous`, $n=8$ vs. $8$).
    - *Criterion*: **Supports** the positive-memory penalty (`B2`/`C1`) if $\text{mean}(r_{\text{Red}} - r_{\text{White}}) < 0$ with one-sided stratified permutation $p < 0.05$; **contradicts** if $> 0$ with opposite one-sided $p < 0.05$; **not decided** otherwise.
  - **T2b (Late post-peak endpoint contrast: Red vs. White at Day 56)**: Stratified block-permutation test of Day-56 log-prevalence $\ln(\frac{k_{56}+0.5}{n_{56}+1})$ (`Red` minus `White`), testing whether post-peak epidemic turnover reverses the early rare-growth ordering.
    - *Criterion*: Same two-tailed directional thresholds ($p < 0.05$ each side) against the naive extrapolation of asymptotic $D$ to a 56-day post-peak epidemic.
  - **T2c (Spatial asynchrony buffering under Red noise, `Statement C7a`)**: Stratified block-permutation test (blocking on `Migration`) of `Asynchronous` ($n=8$) minus `Synchronous` ($n=8$) under `Red` noise on both $r_{\text{prev},0\to 16}$ and mean log-prevalence across Days 16–56, plus the interaction `(Async - Sync)_Red - (Async - Sync)_White`.
    - *Criterion*: **Supports** spatial rescue (`C7a`) if `(Async - Sync)_Red > 0` with one-sided $p < 0.05$; **contradicts** if `< 0` with opposite one-sided $p < 0.05$; **not decided** otherwise.
- **Pre-computed power & identifiability (`results/preanalysis_power_20261115.txt` §2)**:
  - For **T2a** ($n = 16$ vs. $16$, including unblocked 16-day Markov temperature realization variance and $n_{\text{exam}} \approx 45$ binomial error):
    - Null ($\Delta D = 0$): $P(\text{supports}) = \mathbf{4.8\%}$, $P(\text{contradicts}) = \mathbf{4.6\%}$.
    - Pure host-density fluctuation ($\Delta D = +0.11$): $P(\text{supports}) = \mathbf{13.7\%}$ (underpowered for host-only effect alone).
    - Host fluctuation + $5\,^\circ\text{C}$ developmental arrest ($\Delta D = +0.35\text{ to }+0.80$): $P(\text{supports}) = \mathbf{49.8\%\text{ to }97.7\%}$ (well-powered!).
  - For **T2c** ($n = 8$ vs. $8$ under Red noise, engine $\Delta D_{\text{spatial}} = +0.0967$): $P(\text{supports}) = \mathbf{18.0\%}$ because unblocked random red-noise realizations have high path-to-path spread over 56 days.
  - *What counts against the theory vs. indistinguishable from no effect*: Faster early rare-invasion growth under Red noise than White noise ($r_{\text{prev},0\to 16,\text{Red}} > r_{\text{prev},0\to 16,\text{White}}$, $p < 0.05$) in `Synchronous` units would **contradict** the single-patch memory prediction unless over-dispersed relapse/dormancy is active. A **not decided** result on `T2c` ($18\%$ power) or small `T2a` difference ($13.7\%$ power under host-only forcing) cannot be distinguished from zero effect without recorded per-replicate temperature sequences.

### Test Group T3 & Parameter Extractions — *Daphnia* Microparasite Systems (S3–S7)
- **Primary role**: Extract empirical parameter distributions with standard errors / confidence intervals (no tuning to theory):
  1. **S4 (Dallas & Drake 2016)**: Fit exponential ($\mu_{\text{out}}$ constant, memoryless) vs. Weibull ($S(t) = \exp[-(\lambda t)^k]$) survival to `sporeSurv.csv` by maximum likelihood to measure free-spore mean stay $\tau_{\text{out}}$ and test memoryless stay shape ($k = 1$); compute mean filtering rate $F$ (`ml/host/day`) from `filterRate.csv`; compute infected host lifespan $\tau_{\text{in}}$, CV, implied Erlang shape $k_{\text{in}} = 1/\text{CV}^2$, and spore yield $\theta$ from `indivExpData0..4.csv`; combine $\mu_{\text{out}}$ and $F$ to obtain the empirical **contact-limitation curve $d(\bar{H}) = \mu_{\text{out}} / (\mu_{\text{out}} + F\bar{H})$** across host densities $\bar{H} \in [20, 640]\text{ hosts/L}$.
  2. **S3 (Dallas, Krkošek & Drake 2018)**: Measure primary/secondary invasion fractions, empirical threshold density $H_c$, per-host transmission rate $\beta$, and within-microcosm $\sigma_{\ln H}$ and $r_1$ across all 90 microcosms; feed S3 + S4 calibrated parameters into continuous-time Floquet and telegraph calculations (`notes/decisive-experiment-v2.md` framework) to produce an **empirically calibrated prediction curve** for *Daphnia*–*Metschnikowia*.
  3. **Secondary observational check T3a (S3 supercritical arms `160, 320, 640 hosts/L`, $n = 45$)**: Partial Pearson/Spearman correlation of log(epidemic prevalence AUC + 0.01) with within-microcosm $\sigma_{\ln H}$ controlling linearly for $\ln\bar{H}$.
     - *Criterion*: negative partial correlation with one-sided $p < 0.05$ = directionally consistent (labelled observational, non-causal); positive with one-sided $p < 0.05$ = dominated by reverse-causal epidemic host suppression; otherwise **not decided**.
     - *Pre-computed power (`results/preanalysis_power_20261115.txt` §3)*: Only **10.5%** power under true theory ($\sigma_{\ln H}$ spread $\approx 0.12$), with **21.5%** probability of positive sign if large epidemics depress hosts (+feedback). Thus `T3a` cannot causally test `H1` and is included purely as an honest diagnostic of endogenous confounding.
  4. **S5 (Clerc et al. 2015; Tadiri 2023)**, **S6 (Davenport et al. 2024a)**, and **S7 (Krichel et al. 2023)**: Extract *Pasteuria ramosa* within-host maturation schedule & thermal `tti` across 5 temperatures, Michigan lake free-spore-to-host ratios & outbreak lags (`n = 13`), and Krichel's constant-vs-variable $20\,^\circ\text{C}$ prevalence/burden summary alongside its daily AR(1) temperature driver (`s017.csv`).

---

## 6. Deliverables Planned Under This Protocol

1. `casestudies/microcosm/analysis.py`: Single self-contained `python3 -I` script (standard library + `numpy` + `scipy` + `studies/engine-R16/lifecycle_engine.py`) that runs code benchmarks first, executes all registered analyses `T1`–`T3` and parameter extractions `S1`–`S7`, writes machine-readable tables to `casestudies/microcosm/results/`, and writes vector `.svg` and pgfplots `.tex` + `.csv` figures to `casestudies/microcosm/figures/`.
2. `casestudies/microcosm/systems.md`: Comparative table of every candidate experimental system evaluated (open and non-open), detailing stage durations, spore survival, $1-d$, controllability of host density without parasite co-removal, evolution risk, and concrete suitability for the redesigned experiment of `notes/decisive-experiment-v2.md`.
3. `casestudies/microcosm/report.md`: Complete, plain-English case study report reporting every pre-registered test outcome against its exact criterion (`supports`, `contradicts`, `not decided`) with uncertainties, sensitivity to confounders, extracted biological parameters, and limits.

---

## 7. Protocol Amendment 1 — Weather and Environment Module (`2026-11-15T05:06:00Z`)

- **Timestamp (`date`)**: `2026-11-15T05:06:00Z` (`Sat Nov 14 23:06:00 CST 2026`), added **before** opening or analysing any outcome data rows in `data/raw/microcosm/` (continuation from interrupted pre-analysis run).
- **Reason**: Incorporate the `CASESTUDY_SPEC.md` requirement (*"Weather and environment at the place and time of the data"*) inside the registered protocol for both the field dataset (`S6`) and all laboratory microcosm studies (`S1`–`S5`, `S7`), and record the verification of `Zenodo 13259720` (the open R Markdown code archive of `Davenport et al. 2024b`, confirming that its CSV data files exist only on Dryad `10.5061/dryad.s7h44j1gb` behind HTTP 401).
- **Additions**:
  1. **Laboratory studies (`S1`–`S5`, `S7`)**: Record each experiment's controlled incubator temperature regime from deposited data files (`s017.csv` daily `14–26 °C` AR(1) sequences vs. constant `20 °C` in `S7`; `temp.csv` five constant treatments in `S5` Tadiri; `indivExpData0..4.csv` and `filterRate.csv` static `12, 20, 28 °C` and diurnal pulse `v1, v2, v4` around `20 °C` in `S4` Dallas & Drake) and open-access primary methods (`PMC6093173`, `PMC10491407`, `rspb20200787_si_001.pdf`: constant `28 °C` in `S1`; daily `5 °C / 23 °C` switching in `S2`; constant `20 °C` in `S3` and `S5` Clerc).
  2. **Field weather matched to `S6` (Davenport et al. 2024a, 6 southeastern Michigan lakes, May–November 2021)**:
     - Fetch daily meteorological series (`2021-05-01` to `2021-11-30`) via `data/fetch/microcosm.py` from:
       - **NASA ORNL DAAC Daymet V4R1** (`doi:10.3334/ORNLDAAC/2129`, daily, `1 km` Lambert Conformal Conic grid: `tmax`, `tmin`, `prcp`, `srad`, `vp`) at the verified coordinates of all six lakes (`Bishop` `42.5004, -83.8411`; `Cedar` `42.3146, -84.0790`; `Crooked W` `42.3241, -84.1125`; `Little Appleton` `42.5066, -83.8385`; `Mill` `42.3287, -84.0905`; `Walsh` `42.3375, -84.0798`), recording exact Daymet 1-km pixel offsets (`<= 0.71 km`);
       - **NOAA GHCN-Daily** (`doi:10.7289/V5D21VHZ`, daily `TMAX`, `TMIN`, `PRCP`) at the two closest active first-order stations bracketing the Waterloo and Brighton lake clusters (`USW00094889` Ann Arbor Municipal Airport, `42.2239, -83.7398`, and `USW00014833` Jackson Reynolds Field, `42.2637, -84.4564`), recording exact Haversine distances (`km`) to each lake.
  3. **Registered weather estimands (`W1` covariate checks & `W2` stage-duration input)**:
     - **W1a (Spore-to-outbreak lag vs. temperature, `n = 13` outbreaks)**: Spearman rank correlation across the 13 `S6` outbreaks between `DaysBetweenSporeIncreaseAndOutbreakStart` and trailing 14-day Daymet mean temperature $T_{14\text{d}}$ prior to outbreak start (testing whether cooler temperatures lengthen the lag from environmental spore rise to $\ge 1\%$ host prevalence).
     - **W1b (Seasonal weather confounding of host–infection associations in `S6`)**: Compare unadjusted vs. $(T_{14\text{d}}, \text{srad}_{14\text{d}})$-partialled correlations between biweekly $\ln(\text{host density})$ and parasite infection prevalence across `S6` lake-dates, quantifying whether late-summer/autumn cooling produces or masks apparent host-density effects.
     - **W2 (Thermal modulation of within-host duration $\tau_{\text{in}}(T)$ and fluctuation loss $D$)**: Fit log-linear thermal clock sensitivity $\chi_{\text{in}} = -d\ln \tau_{\text{in}}/dT$ to `S5` Tadiri's empirical *Pasteuria* `tti(T)` across temperatures (`temp.csv`), map `S6`'s June–November 2021 Daymet lake temperatures into $\tau_{\text{in}}(T)$, and compute the resulting range of predicted fluctuation loss $D$ (`notes/mechanism-temperature-incubation.md` §3.2 and `Statement C1c`) across summer (`~22 °C`) versus late-autumn (`~6–10 °C`) regimes.


## Continuation amendment 2 — implementation and environmental interpretation (2026-11-15T14:07:03Z)

The inherited protocol, report and helpers are retained byte for byte; their original
lengths and hashes are in `results/continuation_audit.json`. This amendment follows
inspection of file schemas, the first two rows of datasets, and deposited methods,
but precedes all new statistical tests, survival fits and weather associations.
No registered criterion or outcome window is changed.

- The promised “10,000 exact block permutations” is a random subset, not exhaustive
  enumeration. T2 uses 10,000 seeded block permutations and reports Monte Carlo
  uncertainty; exact enumeration is an independent check where tractable. T1 uses
  100,000 seeded permutations (with add-one correction), explicitly an approximation
  to its registered exact test. If an exact tail can be enumerated, report it as well.
  Report both directional p-values and unadjusted individual criteria; these were
  not registered as a familywise 5% test. Descriptive 95% intervals use experimental
  units, never census rows as independent replicates.
- Apply logarithms to positive host counts only, report excluded zeros, and retain
  original time indices for trend and lag pairs. Extinct S3 tubes have prevalence
  zero for AUC, but undefined log host density is not replaced by an arbitrary value.
  S3 primary infection is at day 11 (also report any infection by day 11); secondary
  infection is day >18, as defined in the deposited analysis. No causal threshold or
  transmission coefficient will be fitted to epidemic outcomes.
- The registered S4 exponential and Weibull fits use binomial dead/total counts,
  treating fields of view as subsamples. Their likelihood intervals are conditional
  on the sampled spore suspension, not biological replication. Dye viability is not
  infectivity. Filtering measures encounters, not successful transmission. S3 and S4
  use different host species; combining them yields a labelled sensitivity scenario.
  Clerc cross-sectional sacrifices cannot identify an individual maturation-time CV.
- Bound the continuation simulations: S2 daily Markov grid d={1,.5,.1}, shapes
  {(1,1),(4,4),(1,16)}, colours {0,.8}, host-only versus cold arrest (36 cells),
  2,000 days and 12 antithetic pairs per cell, discard 200 days. Host amplitudes come
  from control-series coefficient of variation; cap symmetric amplitude at .9 and
  report the cap. A 2-day outside / 7-day inside clock remains a literature-informed
  scenario; free-stage residence is not measured. Cold background loss .02/day is
  illustrative. Reuse the completed pre-analysis spatial simulations; add engine
  coefficient sensitivities for d, stay shape, and vertical shares {0,.3,.6}.
  S3/S4 periodic and telegraph grid uses scales {7,14,28,56,112} days, amplitude .8,
  memoryless outside and k_inside={1,4,16}, waste and calibrated waiting; telegraph
  2,000 cycles, 12 antithetic pairs, discard 100. Quantile-approximated telegraph
  code will be independently checked or replaced by exact unbinned propagation.
  Benchmark tolerances: deterministic constant/zero-death 1e-9, Floquet versus
  independent ODE integration 1e-7; simulation comparisons within four combined SE.
  Stop after this grid and one numerical audit; no outcome-driven parameter search.
- Weather amendment 1 remains the primary weather specification. W1a retains the
  signed published lag (negative means spores rose before infection); the opposite
  sign is the elapsed lead time. W1b uses temperature and shortwave radiation as
  registered adjustments; also show within lake/species correlations and linear
  day-of-year adjustment to expose seasonal confounding. Describe temperature,
  precipitation, shortwave and vapor-pressure associations with squared deviations
  of log host density and prevalence. These extra checks are amendment diagnostics.
- W2 fits the registered log-linear `tti(T)` summary, also stratifies hatch origin
  and reports hatch-adjusted values. It is a visible-infection proxy, not a validated
  physiological development relation; lake air temperature is not water temperature.
  Mapping to lakes is therefore only an illustrative sensitivity, with extrapolation
  flagged, not an estimate of field stage duration. Use tau_out={2,10,50} days,
  d={1,.5,.1}, sigma=.5, daily phi=.8 for the C1 coefficient, and the observed
  temperature range. These nuisance values are illustrative and never fitted to
  lake outcomes. No such mapping to other parasite species is biologically calibrated.
- A weather adjustment that changes the sign of a host–infection association, or
  reduces its magnitude by at least half, changes our assessment of that descriptive
  association. Neither it nor the illustrative W2 clock can change the original
  registered T1–T3 labels. Missing actual thermal paths or unreliable site coordinates
  preclude a quantitative causal weather correction. Site coordinates inherited from
  amendment 1 will be checked against open primary material; unverified coordinates
  must be labelled approximate, and grid-centre offsets must not be invented.
