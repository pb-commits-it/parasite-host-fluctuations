# Protocol: Bacteria–Phage Chemostats and Serial-Transfer Microcosms (`chemostat`)

- **Registered:** 14 November 2026, 22:15 CST (`2026-11-15T04:15Z`, recorded from `date`).
- **Chronology guarantee:** Written and locked after the blind schema scout (`schema_scout.py`, output `results/schema_scout.txt`: worksheet names, column headers, and row counts only) and the synthetic pre-analysis verification and power simulation (`power_precheck.py`, output `results/power_precheck.txt`), and **before** any bacterial density, phage titer, variance, autocorrelation, or empirical outcome in `data/raw/chemostat/` was read or analysed.

---

## 1. Question

Lytic bacteriophages alternate between two stages: an intracellular stage inside an infected bacterial cell (the latent period) and an extracellular free-virion stage that decays, washes out, or adsorbs to a host cell upon contact. Does temporal fluctuation in host-cell density lower a phage's growth rate when rare and its effective reproduction number relative to a constant host at the same arithmetic mean density, as predicted by the project's two-stage contact-limited life-cycle theory (`notes/contact-limited-transfer.md`)? Second, in laboratory chemostats and serial-transfer microcosms, is that linear fluctuation effect detectable in observed phage trajectories, or is it masked by contact-limited waiting protection, host phage-resistance evolution, density-dependent host depletion, and predator–prey feedback? Third, how does chemostat dilution—which removes both host cells and free phage—alter the theory's fixed-mean-rate comparison and the upper bound on fluctuation loss (`notes/mechanism-temperature-incubation.md`, Proposition B)?

---

## 2. Mapping onto the Two-Stage Theory and Coupled-Dilution Note

### 2.1 Two-stage linearized phage model

Let $I(t)$ denote the density of phage-infected bacterial cells ($\text{cells mL}^{-1}$) and $P(t)$ the density of infectious free virions ($\text{virions mL}^{-1}$ or plaque-forming units $\text{PFU mL}^{-1}$). When the phage is rare relative to susceptible hosts of density $H(t)$ ($\text{cells mL}^{-1}$), the continuous-time linear operator on $x(t) = (I(t), P(t))^\top$ is

$$\frac{d}{dt}\begin{pmatrix} I(t) \\ P(t) \end{pmatrix} = \begin{pmatrix} -(\nu + \omega_I(t)) & \alpha H(t) \\ \beta \nu & -(\alpha H(t) + \mu_0 + \omega_P(t)) \end{pmatrix}\begin{pmatrix} I(t) \\ P(t) \end{pmatrix}, \tag{1}$$

where every symbol is defined as follows:
- $\alpha$: irreversible phage adsorption rate constant ($\text{mL cell}^{-1}\text{ d}^{-1}$);
- $\nu = 1/L_{\text{lat}}$: lysis (maturation) rate ($\text{d}^{-1}$), where $L_{\text{lat}}$ is the mean latent period ($\text{d}$); for an Erlang latent period of $m_I \ge 1$ equal sub-stages (`shape='chain'`), $I$ is split into $I_1, \dots, I_{m_I}$ each exited by progression at rate $m_I \nu$ and by washout at rate $\omega_I(t)$, with burst $\beta$ released from $I_{m_I}$;
- $\beta$: burst size (infectious progeny virions released per lysed cell, dimensionless);
- $\mu_0$: intrinsic free-phage decay (inactivation) rate in cell-free medium ($\text{d}^{-1}$);
- $\omega_P(t), \omega_I(t)$: dilution (washout) loss rates ($\text{d}^{-1}$) acting on free virions and infected cells, respectively. In a well-mixed continuous-flow chemostat without cell retention, $\omega_P(t) = \omega_I(t) = \omega(t)$ (flow rate divided by vessel volume). In a serial-transfer protocol where a volume fraction $f_k \in (0, 1)$ of whole culture is transferred after incubation duration $\Delta T_k$, dilution acts as a discrete scalar survival multiplier $f_k$ on $(I, P)$ at transfer $k$; if chloroform extraction is applied at transfer $k$, surviving infected cells without mature virions are killed ($I \mapsto 0$) while free phage $P$ is diluted to the target inoculum.

Write $\bar{H}$, $\bar{\omega}_P$, and $\bar{\omega}_I$ for the time-weighted arithmetic means over an observed series. At constant arithmetic-mean inputs $(\bar{H}, \bar{\omega}_P, \bar{\omega}_I)$:
- The deterministic basic reproduction number (counting lysis $I \to P$ as the single birth edge, `birth=True`) is

$$R_0 = \underbrace{\frac{\alpha \bar{H}}{\alpha \bar{H} + \mu_0 + \bar{\omega}_P}}_{1 - d} \cdot \beta S_I(\bar{\omega}_I), \qquad S_I(\bar{\omega}_I) = \left(\frac{m_I \nu}{m_I \nu + \bar{\omega}_I}\right)^{m_I}, \tag{2}$$

- The threshold contact-loss parameter of `notes/contact-limited-transfer.md` (§1 and §7) is

$$d = \frac{\mu_0 + \bar{\omega}_P}{\alpha \bar{H} + \mu_0 + \bar{\omega}_P} \in [0, 1], \tag{3}$$

  which is the fraction of free virions lost to decay or washout before adsorbing at mean host density $\bar{H}$ and at the persistence threshold ($r = 0$). Notably, adsorption is **contact-limited** (`contact=True`): host scarcity lowers the transfer rate $\alpha H(t)$ into $I$ and lowers the exit rate from $P$ by the same amount, so virions wait rather than vanish unless $\mu_0 + \omega_P(t) > 0$.
- Mean stage residence times at the threshold are $\tau_P = 1/(\alpha \bar{H} + \mu_0 + \bar{\omega}_P)$ and $\tau_I = \sum_{j=1}^{m_I} \text{lineage stay} = L_{\text{lat}} / (1 + \bar{\omega}_I / (m_I \nu))$ (equal to $1/(\nu + \bar{\omega}_I)$ for $m_I = 1$), with threshold generation time $\mathcal{T}_0 = \tau_P + \tau_I$.

### 2.2 Predictions computed for every observed trajectory

For each replicate unit $i$ with observed host series $H_i(t_k)$ over intervals $\Delta t_k$ ($k = 1, \dots, K_i$, total span $T_i = \sum_k \Delta t_k$):
1. **Constant-mean rare growth rate $r_{\text{const},i}$ ($\text{d}^{-1}$):** largest real part of the eigenvalues of the constant generator $M(\bar{H}_i, \bar{\omega}_{P,i}, \bar{\omega}_{I,i})$.
2. **Exact linear-model rare growth rate $r_{\text{obs},i}$ ($\text{d}^{-1}$):** top Floquet/Lyapunov growth rate of the piecewise-constant trajectory $\prod_{k=1}^{K_i} \exp(M_k \Delta t_k)$ (computed in log-normalized form to machine precision, and cross-checked against fine-step discrete propagation on step $\delta t$ chosen so all per-step exit probabilities are below $0.05$). The **rare-growth fluctuation penalty** is

$$\Delta r_{\text{fluct},i} = r_{\text{const},i} - r_{\text{obs},i}. \tag{4}$$

3. **Exact threshold reproduction-number loss $D_{\text{sim},i}$ (dimensionless $\ln R_0$ units):** unique root $D$ solving $r_{\text{obs},i}(\beta \mapsto \beta e^{-\ln R_{0,i} + D}) = 0$, matching `lifecycle_engine.py`'s `Engine.loss()['D']`. Phage invades from rarity on the cycled host trajectory if and only if $\ln R_{0,i} > D_{\text{sim},i}$.
4. **Second-order contact-limited closed form $\Gamma_{c,i}$ (`notes/contact-limited-transfer.md`, eq. C2 in continuous time and eq. C1 / `Engine.coefficient()['D2']` in discrete time):**
   Let $\hat{\sigma}_{H,i}$ be the sample standard deviation of $\ln H_i$ (also evaluated using the relative-host standard deviation $\operatorname{SD}(H_i/\bar{H}_i)$ as a finite-amplitude check) and $\hat{\varphi}_i$ the lag-1 autocorrelation of $\ln H_i$ across regular steps $\Delta t$. For $\hat{\varphi}_i \in (0, 1)$, let $\hat{\theta}_{\text{CT},i} = -\Delta t / \ln \hat{\varphi}_i$, $q_P = 1/(1 + \tau_P/\hat{\theta}_{\text{CT},i})$, $q_I = (1 + \tau_I/(m_I \hat{\theta}_{\text{CT},i}))^{-m_I}$, and $R = q_P q_I$. Then equation (C2) gives

$$\Gamma_{c,i} = \hat{\sigma}_{H,i}^2 \left\{ d_i^2 \frac{\hat{\theta}_{\text{CT},i}}{\mathcal{T}_{0,i}} - d_i \frac{R - (1 - d_i) q_P}{1 - R} \right\}, \tag{5}$$

   compared alongside the gain-only ("waste", $d_i = 1$) formula $\Gamma_{\text{waste},i} = \hat{\sigma}_{H,i}^2 [\hat{\theta}_{\text{CT},i}/\mathcal{T}_{0,i} - R/(1 - R)]$. When $\hat{\varphi}_i \le 0$ (oscillatory/anti-persistent host series, where continuous-time exponential covariance does not apply), we evaluate the discrete-time formula (C1) and `Engine.coefficient()['D2']` directly at $\hat{\varphi}_i \in (-1, 1)$ on step $\Delta t$ where admissible, and flag non-exponential covariance explicitly.
5. **Engine caveats (`verification/engine-R16-check.md`):** Whenever `studies/engine-R16/lifecycle_engine.py` is called with `Driver(kind='bounded')`, `low=None` (symmetric two-state driver) is used for small-noise `coefficient()` checks because fixing `low=0.05` alters the small-$\sigma$ standardized limit; and whole-number extinction checks use `Engine.extinction(..., offspring='poisson')` (exact backward PGF composition) rather than `population(..., offspring='shared_groups')`.

### 2.3 Pre-registered theoretical note: coupled dilution in a chemostat

In a chemostat, dilution $\omega(t)$ removes uninfected host cells, free virions, and (unless retained on vessel walls) infected cells. This modifies the fixed-mean-rate comparison in three distinct ways (verified numerically in `power_precheck.py`, check `K4`):

1. **Uniform continuous-time washout ($\omega_P(t) = \omega_I(t) = \omega(t)$):**
   In equation (1), $-\omega(t)\mathbf{I}_2$ is a scalar multiple of the identity matrix and commutes with $M_{\text{no-wash}}(H(t))$ at every instant $t$. Setting $\tilde{x}(t) = \exp(\int_0^t \omega(u)\,du) x(t)$ gives $r(\omega(\cdot), H(\cdot)) = r_{\text{no-wash}}(H(\cdot)) - \bar{\omega}$ (`K4` pathwise error $3.55 \times 10^{-15}$). Consequently, **at fixed arithmetic-mean $(\bar{H}, \bar{\omega})$ in continuous time**, the dynamic fluctuation $\omega(t) - \bar{\omega}$ cancels identically out of $\Delta r_{\text{fluct}}$! However, $\bar{\omega} > 0$ still shifts the threshold root from $r_{\text{no-wash}} = 0$ to $r_{\text{no-wash}} = +\bar{\omega} > 0$, raising the effective waiting-death fraction from $d_0 = \mu_0/(\alpha\bar{H}+\mu_0)$ to $d(\bar{\omega}) = (\mu_0 + \bar{\omega})/(\alpha\bar{H} + \mu_0 + \bar{\omega})$. Thus, even a virion with zero intrinsic decay ($\mu_0 = 0$, which would otherwise suffer zero fluctuation loss $D = 0$ at any amplitude) acquires $d(\bar{\omega}) > 0$ and strictly positive fluctuation loss in a chemostat. Furthermore, against a constant-dilution chemostat (Baseline i: holding pump average $\bar{\omega}$ fixed while letting Monod host dynamics respond), fluctuating $\omega(t)$ depresses mean host density $\bar{H}$ by curvature, adding a mean-rate shift $J < 0$.
2. **Differential washout ($\omega_P(t) \ne \omega_I(t)$, e.g. wall-adherent infected cells or selective filtration) and discrete serial transfers:**
   When dilution acts unequally on $P$ and $I$, $-\operatorname{diag}(\omega_I(t), \omega_P(t))$ no longer commutes with $M_{\text{no-wash}}(H(t))$. Because high dilution $\omega(t)$ drives host scarcity ($\operatorname{Cov}(H, \omega) < 0$), free phage experiences peak washout $\omega_P(t)$ **during host busts**, when reduced adsorption $\alpha H(t)$ forces virions to accumulate in $P$. In `power_precheck.py` (`K4` illustration at identical $\bar{H}=1, \bar{\omega}_P=0.5$), negative chemostat-like coupling ($\operatorname{Cov}(H, \omega) < 0$) raises $D$ from $0.05037$ (constant $\omega_P$) to $0.06128$ ($+21.7\%$), whereas positive coupling lowers $D$ to $0.01131$. In discrete serial transfers where each transfer $k$ multiplies $(I, P)$ by a dilution fraction $f_k$ at fixed arithmetic mean $\bar{f}$, Jensen's inequality adds the exact scalar dilution penalty $-\mathbb{E}\ln(f_k/\bar{f}) > 0$ per transfer (`K4`: $28/100$ discrete fluctuating-dilution schedules strictly violate the naive transfer-only bound $D \le \sum \ell^g$).
3. **Corrected staying-rate bound (`notes/mechanism-temperature-incubation.md`, Proposition B):**
   On any discrete step $\Delta t$ with positive staying probabilities $s_k(t) > 0$ and positive transfers $g_k(t) > 0$, both contact-limited adsorption and fluctuating dilution make $s_P(t)$ (and $s_I(t)$) fluctuate, invalidating the transfer-only bound $D \le \sum_k \ell_k^g$. Proposition B supplies the exact valid bound (`K4`: $0/100$ violations):

$$D \le \ell_P^g + (\tau_{P,0} - 1)\ell_P^s + (\tau_{I,0} - 1)\ell_I^s, \qquad \ell_k^s = -\mathbb{E}\ln\!\left(\frac{s_k(t)}{\bar{s}_k}\right). \tag{6}$$

   As $\Delta t \to 0$ in a continuous-flow chemostat with $s_k(t) = e^{-r_k(t)\Delta t}$ and finite second moments of exit rates $r_k(t)$, $(\tau_{k,0} - 1)\ell_k^s = \frac{\Delta t \, \operatorname{Var}(r_k)}{2\,\mathbb{E}[r_k]} + O(\Delta t^2) \to 0$ (`K4`: halving $\Delta t$ from $0.02$ to $0.01$ gives ratio $1.9850 \approx 2$). Hence the staying-rate penalty is strictly positive and duration-dependent for discrete serial transfers, and vanishes in the continuous-flow limit where $D_{\text{CT}} \le \ell_P^g = -\mathbb{E}\ln(H(t)/\bar{H})$.

---

## 3. Data and Provenance

All raw files were fetched on 15 November 2026 (`04:00–04:01 UTC`) by `data/fetch/chemostat.py` into `data/raw/chemostat/` (verified git-ignored via `.gitignore:3:data/raw/`), with SHA-256 digests recorded in `data/raw/chemostat/MANIFEST.json`.

| ID | Citation & Paper DOI | Deposit DOI & Version | Repo & Licence | System & Regime | Usable for density analysis? |
|---|---|---|---|---|---|
| **D1** | Larsen, Wilhelm & Lennon (2019) *Ecol. Lett.* 22:1009 (`10.1111/ele.13252`) | `10.6084/m9.figshare.7742819.v3` (v3) | Figshare `7742819`, **CC BY 4.0** | *Synechococcus* WH7803 + cyanomyovirus S-RIM8; 40 mL continuous chemostats ($\omega = 1.0\text{ d}^{-1}$, ~166 d, N- vs P-limited, 6 infected + 4 control chemostats + `inf-mat.csv` resistance) | **Yes** (primary continuous-flow chemostat study) |
| **D2** | Reuter et al. (2026) *Virus Evol.* (`10.1093/ve/veag022`) | `10.5281/zenodo.17961407` (v2) | Zenodo `17961407`, **CC BY 4.0** | *E. coli* C + lytic phage $\Phi\text{X174}$; serial transfers comparing uniform 3-h transfers (5 lines) vs pulsed 4$\times$30-min (2% transfer) + 3-h schedule (5 lines, 50 transfers), fresh susceptible host each transfer + measured traits | **Yes** (primary controlled-schedule serial-transfer study) |
| **D3** | Chaudhry et al. (2018) *PLOS Biol.* 16:e2005971 (`10.1371/journal.pbio.2005971`) | `10.1371/journal.pbio.2005971` (v1) | Figshare `6974783`, **CC BY 4.0** | *E. coli* B + virulent phage $\lambda^{\text{VIR}}$; daily 1:100 serial transfers in maltose-limited M9M (12 lines, persistence vs loss) and LB (10 lines) + uninfected & phage-washout controls | **Yes** (persistence vs extinction + resistance transition) |
| **D4** | Duncan et al. (2017) *J. Evol. Biol.* 30:55 (`10.1111/jeb.12985`) | `10.5061/dryad.vv29r` (v1) | Zenodo `4954973` (Dryad mirror), **CC0 1.0** | *P. fluorescens* SBW25 + lytic phage SBW25$\Phi$2; 48-h 1% serial transfers (days 0–16), 30 infected + 20 uninfected lines across constant (28/32 °C) vs 2, 4, 8-d thermal switching + `Sheet2` resistance | **Yes** (multi-frequency environmental switching + resistance) |
| **D5** | Zhao, Hao & Zhang (2016) *PLOS ONE* 11:e0168560 (`10.1371/journal.pone.0168560`) | `10.5061/dryad.73s68` (v1) | Zenodo `5012658` (Dryad mirror), **CC0 1.0** | *P. fluorescens* SBW25 + phage SBW25$\Phi$2; 48-h 1% serial transfers across 3 nutrient levels (`Population_density.xlsx`: bacterial OD and $\log_{10}(\text{PFU}+1)$) | **Yes** (secondary productivity gradient; host in OD units converted via literature calibration with wide sensitivity) |
| **D6** | Harrison et al. (2013) *Proc. R. Soc. B* 280:20130937 (`10.1098/rspb.2013.0937`) | `10.5061/dryad.8k015` (v1) | Zenodo `4957209` (Dryad mirror), **CC0 1.0** | *P. fluorescens* SBW25 + $\Phi$2; alternating $1\times / 0.1\times$ KB nutrient pulses vs $0.55\times$ matched mean | **No**: blind header check (`results/schema_scout.txt`) showed `RAW DATA.xlsx` contains only `TIME-SHIFT DATA` and `RANGE DATA` cross-infection sheets, with no density columns. Reported as an unanalyzable deposit per Rule 4. |

---

## 4. Literature Life-History Parameters and Sensitivity Ranges

No parameter is fitted to any dataset. Every input is fixed beforehand from the dataset's paper (`[paper]`), published literature on the same or congeneric host–phage system (`[literature]`), or a bounded illustrative baseline accompanied by a wide sensitivity sweep (`[illustrative]`):

| System (Datasets) | Latent period $L_{\text{lat}}$ | Burst size $\beta$ | Adsorption rate $\alpha$ | Free-phage decay $\mu_0$ | Dilution / transfer loss $\omega$ | Provenance & Sensitivity Range |
|---|---|---|---|---|---|---|
| ***Synechococcus* WH7803 – S-RIM8** (D1) | $9.0\text{ h} = 0.375\text{ d}$ `[literature]` (range $3\text{–}14\text{ h}$) | $50\text{ virions cell}^{-1}$ `[literature]` (range $22\text{–}95$) | $6.06 \times 10^{-9}\text{ mL min}^{-1} = 8.73 \times 10^{-6}\text{ mL d}^{-1}$ `[literature]` (range $10^{-6}\text{–}3\times 10^{-5}\text{ mL d}^{-1}$) | $0.03\text{ d}^{-1}$ `[literature]` (range $0.01\text{–}0.38\text{ d}^{-1}$) | $\omega = 1.0\text{ d}^{-1}$ `[paper]` (continuous chemostat) | Wilson et al. 1996 / Mann 2003 *FEMS Microbiol. Rev.* 27:17 (`S-PM2` cyanomyovirus on WH7803: $\alpha = 6.06\times 10^{-9}\text{ mL min}^{-1}$, $L_{\text{lat}} \approx 9\text{ h}$, $\beta = 22$); Traving et al. / Mruwat et al. 2022 *ISME J.* (`WH7803` cyanophages: $L_{\text{lat}} = 3\text{–}8\text{ h}$, $\beta = 53.5\text{–}95.3$, $\mu_0 = 0.024\text{–}0.034\text{ d}^{-1}$); Traving / Maudlin acidification study ($\mu_0 = 0.35\text{–}0.38\text{ d}^{-1}$ at high light; Larsen 2019 uses low light $20\,\mu\text{E m}^{-2}\text{ s}^{-1}$). |
| ***E. coli* C – $\Phi\text{X174}$** (D2) | $17\text{ min} = 0.01181\text{ d}$ `[paper]` (range $14\text{–}24\text{ min}$) | $200\text{ PFU cell}^{-1}$ `[paper]` (range $100\text{–}300$) | $4.0 \times 10^{-9}\text{ mL min}^{-1} = 5.76 \times 10^{-6}\text{ mL d}^{-1}$ `[paper]` (evolved variants: $2\times 10^{-9}$ and $6\times 10^{-9}$) | $1.7 \times 10^{-2}\text{ min}^{-1} = 24.48\text{ d}^{-1}$ `[paper]` (evolved: $0.006\text{–}0.014\text{ min}^{-1}$) | Recorded per transfer `[paper]` (2% transfer $f=0.02$ every $30\text{ min}$ in `T1–T4`, plus $3\text{ h}$ `T5` and daily MOI-0.1 reset) | Reuter et al. (2026) *Virus Evol.* `10.1093/ve/veag022`, Table 1 (`[paper]`, read in full text). |
| ***E. coli* B – $\lambda^{\text{VIR}}$** (D3) | $25\text{ min} = 0.01736\text{ d}$ `[paper]` (range $20\text{–}40\text{ min}$) | $55\text{ PFU cell}^{-1}$ `[paper]` ($50$ in Fig. S4, $60$ in Fig. 1; range $40\text{–}80$) | $2.0 \times 10^{-7}\text{ mL h}^{-1} = 4.80 \times 10^{-6}\text{ mL d}^{-1}$ `[paper]` (range $10^{-6}\text{–}1.5\times 10^{-5}\text{ mL d}^{-1}$) | $0.10\text{ d}^{-1}$ `[illustrative]` (checked directly against host-free washout curve in `Figure S3`, range $0.01\text{–}1.0\text{ d}^{-1}$) | $1:100$ daily serial transfer ($\omega_{\text{eff}} = -\ln(0.01)/\text{d} = 4.605\text{ d}^{-1}$) `[paper]` | Chaudhry et al. (2018) *PLOS Biol.* `10.1371/journal.pbio.2005971`, Fig. 1, Fig. S1, Fig. S4 (`[paper]`, read in full text). |
| ***P. fluorescens* SBW25 – $\Phi$2** (D4, D5) | $35\text{ min} = 0.02431\text{ d}$ `[illustrative]` (range $20\text{–}60\text{ min}$) | $90\text{ PFU cell}^{-1}$ `[literature]` (range $40\text{–}180$) | $3.0 \times 10^{-9}\text{ mL min}^{-1} = 4.32 \times 10^{-6}\text{ mL d}^{-1}$ `[illustrative]` (range $10^{-6}\text{–}1.5\times 10^{-5}\text{ mL d}^{-1}$) | $0.15\text{ d}^{-1}$ `[illustrative]` (range $0.02\text{–}1.5\text{ d}^{-1}$; thermal sensitivity in D4 tested at 32 °C) | $1:100$ transfer every $48\text{ h}$ ($\omega_{\text{eff}} = -\ln(0.01)/2\text{ d} = 2.303\text{ d}^{-1}$) `[paper]` | No one-step growth measurement specific to SBW25$\Phi$2 was found in our full-text search; $L_{\text{lat}} = 45\text{ min}$ and $\beta = 90$ are measured for lytic ~40-kb *P. fluorescens* phage VSW-3 (Zhang et al. 2017 *Can. J. Microbiol.* `10.1139/cjm-2016-0368`), combined with lytic *Pseudomonas* podo/myovirus ranges (`20–60 min`, `40–180`, $\alpha \in [10^{-6}, 1.5\times 10^{-5}]\text{ mL d}^{-1}$). For D5 (`Bacteria_OD`), standard 600 nm conversion $1.0\text{ OD} \approx 5\times 10^8\text{ CFU mL}^{-1}$ `[illustrative]` is swept over $[10^8, 2\times 10^9]\text{ CFU mL}^{-1}\text{ OD}^{-1}$. |

---

## 5. Accounting for Every Confounding and Dynamical Factor

Per `CASESTUDY_SPEC.md`, every process capable of producing or masking the predicted fluctuation effect is classified beforehand as **(a)** included in the model, **(b)** controlled for in the analysis, **(c)** bounded via sensitivity analysis, or **(d)** unaddressable with these data:

| # | Process | Class | How handled in model, analysis, or sensitivity sweep |
|---|---|---|---|
| 1 | **Mean host density $\bar{H}_i$** (not only variance) | **(a), (b)** | Included directly in $r_{\text{const},i}$ and $R_{0,i}(\bar{H}_i)$; controlled via rank-residualization on $r_{\text{const},i}$ in `T1` and compared against uninfected host controls in D1 and D4. |
| 2 | **Seasonality / periodic forcing & secular trend** | **(a), (c)** | Periodic dilution/thermal schedules (D2, D4) are propagated step-by-step on their exact schedule; secular drift (e.g. post-inoculation host recovery) is tested via linear-detrended vs raw $\ln H_i$ sensitivity. |
| 3 | **Host memory beyond AR(1)** (quasi-cycles, multi-lag covariance) | **(a)** | Exact simulation `r_obs,i` and `D_sim,i` propagate the linear operator along the **observed** sequence $H_i(t_k)$, capturing the full empirical spectrum and higher moments without AR(1) truncation; comparing `D_sim,i` with AR(1)-based $\Gamma_{c,i}$ in `T2` directly measures the non-AR(1) / finite-noise gap. |
| 4 | **Temperature and abiotic drivers** (light:dark, stoichiometry, thermal switching) | **(c)** | In D1, N- vs P-limitation can alter phage burst size $\beta$ and latent period $L_{\text{lat}}$ (bounded across $\beta \in [22, 95], L_{\text{lat}} \in [3, 14]\text{ h}$). In D4, 32 °C heat stress directly raises phage decay / lowers replication on SBW25 (bounded by testing a thermal-mortality penalty at 32 °C vs temperature-independent rates). |
| 5 | **Whether host scarcity wastes or delays the waiting stage** | **(a), (c)** | Primary model is contact-limited (`contact=True`, $d_i < 1$ from eq. 3, scarcity delays adsorption). Explicitly contrasted against the gain-only ("waste", $d_i = 1$) model in every dataset. |
| 6 | **Shape (spread) of stage durations** (latent period regularity) | **(a), (c)** | Primary continuous-time/engine model uses memoryless stays ($m_I = 1$) and is swept across Erlang latent-period chains $m_I \in \{1, 4, 16\}$ (`Stage(shape='chain', substages=m_I)`) and discrete fixed delay (`shape='fixed'`) to test whether narrow lysis timing alters $D_{\text{sim}}$ or triggers period resonance. |
| 7 | **Coupled dilution / washout** removing host, free phage, and infected cells | **(a)** | Explicitly modelled in both continuous chemostats ($\omega_P = \omega_I = 1.0\text{ d}^{-1}$ vs wall-retained $\omega_I = 0$) and serial transfers (exact dilution factor $f_k$ at each transfer, with vs without chloroform lysis of $I$), plus Proposition B bound check. |
| 8 | **Density dependence in parasite and host** (MOI saturation, superinfection loss, Monod ceiling) | **(c)** | Linear rare-growth predictions omit self-shading; we quantify the density-dependent gap $r_{\text{obs},i} - r_{\text{phage,emp},i}$ and bound saturating encounter kinetics $f(h) = 2h/(1 + h/h_{50})$ and secondary adsorption onto infected cells in D2 (`Real_MOI`). |
| 9 | **Host feedback** (phage crashing host numbers — reverse causality) | **(b), (c)** | Explicitly diagnosed by comparing host variance $\sigma_H$ in **phage-infected vs phage-free control vessels** (available in D1 and D4!) and separating pre-crash invasion windows from endemic predator–prey cycles (`power_precheck.py` scenario `S3`). |
| 10 | **Alternative hosts, spatial structure, wall biofilm refuges** | **(c), (d)** | Static microcosms (D4, D5) and chemostat glass walls (D1, D3) permit spatial/wall refuges (`(d)` unmeasured spatially, `(c)` bounded via minority susceptible subpopulation / selective washout $\omega_I \ne \omega_P$). |
| 11 | **Demographic noise at small numbers** (serial-transfer bottlenecks) | **(a), (c)** | Evaluated via `Engine.extinction(..., offspring='poisson')` at the actual bottleneck sizes ($1\%$ of $6\text{ mL}$ in D4/D5, $2\%$ of $5\text{ mL}$ in D2, $1\%$ of $10\text{ mL}$ in D3) whenever titers approach detection thresholds. |
| 12 | **Observation and sampling error** (CFU/PFU counts, OD, zeros/detection limits) | **(b), (c)** | Zeros in PFU/CFU occur at finite assay detection limits: pre-registered rule replaces zero counts only in log-summary statistics by half the stated detection limit (recording exact zero counts separately as phage loss/below-detection events, and testing sensitivity to dropping zeros vs half-LOD imputation). Measurement variance is subtracted to report noise-corrected host $\hat{\sigma}_{H,\text{corr}}^2 = \max(0, \hat{\sigma}_H^2 - \sigma_{\text{meas}}^2)$ in sensitivity checks. |
| 13 | **Evolution during the study** (host resistance sweeps & phage counter-evolution) | **(b), (c)** | Directly tracked where measured! In D1 (`inf-mat.csv` across 6 time points) and D4 (`Sheet2` resistant fraction), we recompute $r_{\text{const}}, r_{\text{obs}}, D_{\text{sim}}$ on **estimated susceptible host trajectories $H_S(t) = (1 - q_{\text{res}}(t)) H(t)$** alongside total $H(t)$. In D2, fresh ancestral *E. coli* C prevents host evolution while evolved phage traits ($\Phi_L, \Phi_S$ in Table 1) are evaluated directly. |
| 14 | **Confounding between treatments** | **(b), (c)** | Controlled by within-treatment replicate comparisons where available and flagged wherever treatments alter mean physiology alongside $\sigma_H$ (N vs P stoichiometry in D1; 28 vs 32 °C in D4; M9M vs LB in D3; 3 productivities in D5). |

---

## 6. Estimands, Registered Tests, and Decision Criteria

### 6.1 Pre-registered data-handling rules (fixed before viewing data values)
1. **Units:** All rates and growth exponents are reported in **per day ($\text{d}^{-1}$)**, and all threshold reproduction-number losses $D_{\text{sim}}, \Gamma_c$, and Proposition B bounds are dimensionless ($\ln R_0$ units).
2. **Zeros and missing values:** Missing entries (`NA`/blank) are omitted with exact interval lengths $\Delta t_k = t_k - t_{k-1}$ preserved between consecutive valid observations. If $H_i(t_k) = 0$, it is imputed at half the assay detection limit ($0.5 \times \min_{H>0} H$ in that dataset) for $\ln H$ and piecewise trajectory integration, with sensitivity checked at $0.1\times$ and $1.0\times$ minimum positive $H$. Phage zeros ($P_i(t_k) = 0$) define empirical below-detection / extinction events; for net log-phage slope $r_{\text{phage,emp},i}$, half the minimum positive PFU/mL in that dataset is used as the censoring floor. In D5, `Phage_log(PFU+1)` is converted back via $10^x - 1$.
3. **Empirical phage observables $y_i$:**
   - **Primary full-series observable $y_{\text{mean},i}$:** time-averaged $\ln P_i$ over post-inoculation observations (measuring sustained phage abundance / survival; extinct lines carry low time-averaged $\ln P_i$), supplemented by **net production rate** $r_{\text{phage,emp},i} = \frac{\ln P_i(t_{\text{end}}) - \ln P_i(t_{\text{start}})}{t_{\text{end}} - t_{\text{start}}} + \bar{\omega}_{P,i}$.
   - **Early pre-resistance invasion observable $y_{\text{early},i}$:** initial net phage growth rate over the first post-inoculation interval (or pre-resistance window $t \le 9\text{ d}$ in D1, first transfer in D2–D5), where phage is closest to invading ancestral susceptible hosts.

### 6.2 Registered tests and exact outcomes

- **Test `T1` (Primary empirical test across replicates within each dataset D1–D5):**
  Does predicted linear fluctuation loss $D_{\text{sim},i}$ lower empirical phage performance ($y_{\text{mean},i}$ primary; $r_{\text{phage,emp},i}$ and $y_{\text{early},i}$ secondary) after controlling for the constant-mean baseline rare growth rate $r_{\text{const},i}$?
  - **Statistic:** Freedman–Lane rank-residual partial Spearman correlation $\rho_{\text{part}}(y_i, D_{\text{sim},i} \mid r_{\text{const},i})$ across infected replicate units $i = 1, \dots, n$ within each dataset (exact over all $n!$ permutations for $n \le 7$; $10,000$ permutations with seed `20261114` for $n > 7$).
  - **Decision rule (exact, per dataset):**
    - **`supports`**: $\rho_{\text{part}} < 0$ and one-sided $p_{\text{neg}} < 0.05$;
    - **`contradicts`**: $\rho_{\text{part}} > 0$ and one-sided $p_{\text{pos}} < 0.05$ **and** host fluctuation is externally imposed rather than endogenously phage-driven (tested by checking whether infected $\sigma_H$ exceeds uninfected-control $\sigma_H$ by $> 2\times$; if infected $\sigma_H > 2\times$ control $\sigma_H$, a positive $\rho_{\text{part}}$ is classified **`not decided (confounded by endogenous predator–prey feedback)`**, as pre-established by scenario `S3` below);
    - **`not decided`**: otherwise ($p_{\text{neg}} \ge 0.05$ and $p_{\text{pos}} \ge 0.05$).

- **Test `T2` (Mathematical & structural tests of the theory on empirical trajectories):**
  - **`T2a` (Proposition B staying-rate bound):** Across every admissible discrete-time replicate model (both constant-dilution and fluctuating-dilution), test $D_{\text{sim},i} \le \ell_{P,i}^g + (\tau_{P,0,i}-1)\ell_{P,i}^s + (\tau_{I,0,i}-1)\ell_{I,i}^s$. **Criterion:** **`supports`** if 0 violations occur beyond root-finding tolerance $10^{-6}$; **`contradicts`** if any violation exceeds $10^{-6}$.
  - **`T2b` (Contact-limited waiting protection vs waste, and second-order $\Gamma_c$ accuracy):** Quantify the ratio $D_{\text{sim},i}(\text{contact}) / D_{\text{sim},i}(\text{waste}, d=1)$ and the relative error $|\Gamma_{c,i} - D_{\text{sim},i}| / \max(|D_{\text{sim},i}|, 10^{-6})$ across all empirical trajectories.

- **Test `T3` (Magnitude comparison of competing mechanisms):**
  Compare four quantities in identical $\ln R_0$ units for every dataset: (i) deterministic baseline margin $\ln R_0(\bar{H}_i)$, (ii) linear host-fluctuation loss $D_{\text{sim},i}$ on total $H_i(t)$, (iii) resistance-evolution loss $\Delta \ln R_{0,\text{res},i} = \ln R_0(\bar{H}_i) - \ln R_0(\bar{H}_{S,i})$ (in D1, D3, D4), and (iv) coupled-dilution loss shift.

---

## 7. Pre-Analysis Power and Identifiability (`results/power_precheck.txt`)

Before touching any data file, `power_precheck.py` simulated $1,000$ synthetic datasets per cell at the sample sizes, step counts, and measurement noise of Designs A–D under four data-generating worlds:
- **`S0_null`**: Phage responds only to deterministic mean $\ln R_0(\bar{H}_i)$ plus biological/assay noise ($\text{SD} = 0.35$).
- **`S1_linear_highH_d0.04`**: Linear rare-invasion theory holds exactly at typical dense culture concentrations ($d = 0.04$, where contact-limited waiting strongly buffers host fluctuations).
- **`S2_linear_lowH_d0.70`**: Linear rare-invasion theory holds at scarce susceptible host concentrations ($d = 0.70$, weaker waiting buffer).
- **`S3_nonlinear_ecoevo`**: Endogenous nonlinear predator–prey cycles plus a $98\%$ resistance sweep, where higher phage attack drives larger host crashes ($\sigma_H$ rises with phage productivity).

| Design ($n$, steps $K$, $\Delta t$) | Scenario | Mean $\Delta D$ range ($\ln R_0$) | Mean $\rho_{\text{part}}$ | `supports` (%) | `contradicts` (%) | `not decided` (%) |
|---|---|---:|---:|---:|---:|---:|
| **A: Chemostat** (D1 scale: $n=6, K=60, \Delta t=2\text{ d}$) | `S0_null` | 0.0927 | $+0.040$ | 4.8% | 5.8% | 89.4% |
| | `S1_linear_highH` ($d=0.04$) | 0.0922 | $-0.039$ | **5.8%** | 4.1% | 90.1% |
| | `S2_linear_lowH` ($d=0.70$) | 0.3128 | $-0.196$ | **10.1%** | 2.4% | 87.5% |
| | `S3_nonlinear_ecoevo` | 0.1135 | $+0.589$ | 0.2% | **30.8%** | 69.0% |
| **B: Pulsed transfer** (D2 scale: $n=10, K=20, \Delta t=0.125\text{ d}$) | `S0_null` | 0.0294 | $-0.010$ | 6.0% | 5.5% | 88.5% |
| | `S1_linear_highH` ($d=0.04$) | 0.0293 | $-0.032$ | **5.4%** | 3.3% | 91.3% |
| | `S2_linear_lowH` ($d=0.70$) | 0.0504 | $-0.021$ | **5.8%** | 5.5% | 88.7% |
| | `S3_nonlinear_ecoevo` | 0.0374 | $+0.479$ | 0.1% | **39.9%** | 60.0% |
| **C: Nutrient transfer** (D3/D5 scale: $n=18, K=12, \Delta t=1\text{ d}$) | `S0_null` | 0.0803 | $+0.012$ | 5.0% | 5.5% | 89.5% |
| | `S1_linear_highH` ($d=0.04$) | 0.0794 | $-0.028$ | **7.7%** | 3.4% | 88.9% |
| | `S2_linear_lowH` ($d=0.70$) | 0.2551 | $-0.103$ | **11.5%** | 2.3% | 86.2% |
| | `S3_nonlinear_ecoevo` | 0.0997 | $+0.436$ | 0.0% | **58.3%** | 41.7% |
| **D: Freq. microcosms** (D4 scale: $n=30, K=9, \Delta t=2\text{ d}$) | `S0_null` | 0.0945 | $+0.036$ | 3.2% | 8.1% | 88.7% |
| | `S1_linear_highH` ($d=0.04$) | 0.0951 | $-0.019$ | **5.3%** | 4.0% | 90.7% |
| | `S2_linear_lowH` ($d=0.70$) | 0.3101 | $-0.110$ | **13.3%** | 1.1% | 85.6% |
| | `S3_nonlinear_ecoevo` | 0.1186 | $+0.520$ | 0.0% | **93.0%** | 7.0% |

### What this pre-analysis calculation establishes beforehand

1. **Why power for `T1` is intrinsically low ($\approx 5\%\text{–}13\%$) in standard phage cultures:** At dense bacterial concentrations ($\bar{H} \sim 10^7\text{–}10^8\text{ mL}^{-1}$), rapid adsorption makes the threshold waiting-loss fraction $d$ very small ($d \approx 0.01\text{–}0.10$), so contact-limited waiting shrinks the predicted fluctuation loss $D_{\text{sim}}$ to $\sim 0.01\text{–}0.09$ $\ln R_0$ units. Even when $d = 0.70$ (scarce susceptible hosts), $\Delta D \approx 0.25\text{–}0.31$ yields only $10.1\%\text{–}13.3\%$ power at $n = 6\text{–}30$ against realistic biological and plating noise ($\text{SD} \approx 0.35$).
2. **Severe reverse-causality hazard (`S3_nonlinear_ecoevo`):** In coevolving or resource-fed chemostats and microcosms where the phage itself drives host crashes, replicates with stronger phage exploitation generate *larger* host fluctuations $\sigma_H$, producing positive partial rank correlations ($\bar{\rho} = +0.44\text{ to }+0.59$) $31\%\text{–}93\%$ of the time. Comparing infected $\sigma_H$ with phage-free control $\sigma_H$ (available in D1 and D4) is therefore essential to separate exogenous host fluctuation from phage-driven host crashes.
3. **What would count against the theory:**
   - Any admissible trajectory violating Proposition B's bound (`T2a`);
   - In an experiment where host density is **externally set** (host resistance blocked and host feedback absent, closest to D2 or an early pre-crash window with exogenous forcing) and calibrated $d_i$ is large enough that predicted $\Delta r_{\text{fluct}} \cdot T$ exceeds $4$ combined measurement-and-replicate standard errors, an observed phage growth contrast in the **opposite direction** ($\rho_{\text{part}} > 0, p_{\text{pos}} < 0.05$) not explained by measured trait evolution or encounter curvature.
4. **What the study cannot distinguish from no effect:**
   - Any outcome in `T1` when predicted $\Delta D$ across units is $< 0.15$ $\ln R_0$ units (indistinguishable from `S0_null` at $> 85\%$ probability);
   - Endemic chemostat or coevolving serial-transfer trajectories in which phage-induced host crashes ($\sigma_{H,\text{infected}} \gg \sigma_{H,\text{control}}$) and $> 90\%$ bacterial resistance sweeps dominate total host variance and phage equilibrium titers. Even where `T1` is `not decided`, these datasets supply empirical host variance-and-memory regimes ($\hat{\sigma}_H, \hat{\varphi}$), susceptible-vs-total host contrasts, and real-trajectory benchmarks of $\Gamma_c$ and Proposition B for the wider programme.

---

## Amendment 1 (14 November 2026, 22:18 CST / `2026-11-15T04:18Z`)

Recorded after inspecting worksheet coordinate layouts and deposit column encodings to write the data parsers, and before computing any host statistic, model prediction, or test outcome:

1. **D2 (`Reuter et al. 2026`):** `experiment_2_30_min_transfers.csv` contains both the aborted `P2` passage (missing phage counts at `T4`–`T5`) and its replacement `P2b`. Following the deposit specification verbatim ("Passage P2b worked and was used as the second passage day. In total, 10 passage days (40 30-minute transfers + 10 amplification steps) were done"), `P2` is omitted and `P2b` is retained, giving the complete 50-transfer chronological series (`Passage_ID` 1..50) for each of lines `L1`–`L5`. In `experiment_1_3_h_transfers.csv`, `CFU_ml = CFU * Dilution_CFU * 10` and `PFU_ml = PFU * Dilution_PFU * 10` ($100\,\mu\text{L}$ plated, per the deposit key).
2. **D3 (`Chaudhry et al. 2018`):** `pbio.2005971.s008.xlsx` sheet `Figure 2` provides 10 `LB` replicate pairs (`B_LB1..10`, `P_LB1..10`, days 0, 1, 2, 3, 5, 7, 9) and 12 `M9M` replicate pairs (`B_M9M1..12`, `P_M9M1..12`, days 0, 1, 2, 3, 5, 7) alongside phage-free `Control` series; sheet `Figure S3` (`No host 1..4`) provides cell-free phage serial-transfer washout series used solely as an empirical check on $(\mu_0 + \omega_{\text{eff}})$ (not fitted).
3. **D5 (`Zhao et al. 2016`):** `Population_density.xlsx` contains **36** phage-infected lines (`H1..H12` at `1*`, `M1..M12` at `0.1*`, `L1..L12` at `0.01*` KB across transfers 1..20; `Transfer = 0` has initial phage inoculum and `NA` bacterial OD), rather than the 18 estimated before opening rows; `Three_productivities.xlsx` provides the 36 uninfected bacterial `logOD` baselines, and `Challenge_data.xlsx` column `Y` (`Resistant Rate` at `PhageTransfer == 'N'`) provides contemporary bacterial resistance fractions, extending the susceptible-host $H_S(t)$ sensitivity check to D5 alongside D1 and D4.
4. **D1 (`Larsen et al. 2019`):** Per `README.md`, `cid-means.csv` provides post-addition (`day >= 0`) and pre-addition (`day < 0`) densities in $\text{cells mL}^{-1}$ (`Syn`) and $\text{particles mL}^{-1}$ (`Phage`) across the 6 infected chemostats (`NL2, NL3, NL5, PL2, PL4, PL5`) and 4 uninfected controls (`NL1, NL4, PL1, PL3`); in `inf-mat.csv`, `0` encodes infection (susceptible host) and `1` encodes no infection (resistant host). Post-inoculation dynamics (`day >= 0`) form the analysis span.

---

## Amendment 2: Laboratory Temperature, Light, and Abiotic Environment (`r27-weather-chemostat`)

- **Registered:** 14 November 2026, 23:32 CST (`2026-11-15T05:32Z`, recorded from `date`), after `CASESTUDY_SPEC.md` added `"Weather and environment at the place and time of the data"`, after fetching primary open-access Methods texts and independent thermal-response trait files via `data/fetch/chemostat-weather.py` into `data/raw/chemostat/weather_env/` (`WEATHER_ENV_MANIFEST.json`), and **before** running any environmental-covariate correlation, mechanistic thermal trajectory propagation, or amended test on D1–D5.

### A2.1 Inventory of field records (zero) and verified laboratory environmental regimes

All six deposits in `chemostat` (**D1–D6**) come from indoor controlled-environment laboratory chambers, shaking incubators, or static microplate/bottle incubators; the study uses **zero (`0`) outdoor field records**. Per `CASESTUDY_SPEC.md` (*"Laboratory studies: use the laboratory's reported temperature regime instead and say so"*), outdoor meteorological station and gridded climate products (NEON, Daymet, TerraClimate, NOAA SST, GHCN-Daily, ERA5) do not govern thermostatted indoor incubators and are not matched. None of D1–D6 reports calendar dates per sampling point (time is indexed in experimental days or transfers from phage inoculation); institutional locations and publication/deposit dates are recorded below for provenance only, alongside the verified laboratory temperature, light/photoperiod, agitation, and medium regimes:

| ID | Performing Laboratory (Provenance Only) & Dates | Reported Temperature Regime (`[paper]` verified) | Reported Light / Photoperiod & Agitation Regime (`[paper]` verified) | Abiotic Medium Regime & Uncertainty |
|---|---|---|---|---|
| **D1** | Indiana Univ., Bloomington, IN, USA ($\approx 39.165^\circ\text{N}, 86.523^\circ\text{W}$); analysis files stamped `2013-02`, deposit `2019-03` | Constant **`25 °C`** in a Percival growth chamber (Perry, IA, USA) over `166 d` post-addition (`-125` to `166 d` total) | **`14 : 10 h light : dark` diel cycle** at **`20 µE m⁻² s⁻¹`** photosynthetically active radiation (PAR); continuous magnetic stir-bar mixing (`40 mL` continuous chemostat, $\omega = 1.0\text{ d}^{-1}$) | Modified AN artificial seawater: **N-limited** (`N:P = 10:1`, $220\,\mu\text{M}\text{ KNO}_3$) vs **P-limited** (`N:P = 40:1`, $11\,\mu\text{M}\text{ K}_2\text{HPO}_4$); chamber setpoint tolerance unlogged per vessel (typical Percival $\pm 0.5\,^\circ\text{C}$). *(Source: `D1_Larsen_etal_2019.pdf`, p. 1010.)* |
| **D2** | Max Planck Inst. for Evolutionary Biology, Plön, Germany ($\approx 54.162^\circ\text{N}, 10.418^\circ\text{E}$); deposit `2025-12` / `2026-03` | Constant **`37 °C`** across all `3 h` (`4–7` transfers) and `30 min` (`50` transfers) selection lines (`4 °C` centrifugation during chloroform extraction) | Enclosed orbital shaking incubator (**dark / non-photosynthetic**, light unmonitored); **`250 rpm`** (`2.5 cm` orbital diameter, `5 mL` culture; `200 rpm` water bath for adsorption assays) | Rich LB (Miller) supplemented with `10 mM MgCl₂` + `5 mM CaCl₂`; fresh exponential ancestral *E. coli* C supplied every transfer. *(Source: `D2_Reuter2026_PMC13108444.xml`.)* |
| **D3** | Emory Univ., Atlanta, GA, USA ($\approx 33.792^\circ\text{N}, 84.323^\circ\text{W}$); deposit `2018-08` | Constant **`37 °C`** across all `22` infected and `2` control lines (`7–9 d`, daily `1:100` transfers) | Enclosed shaking incubator (**dark / non-photosynthetic**, light unmonitored); **vigorous shaking** (`10 mL` flask cultures) | **M9M** (`500 µg mL⁻¹` / `0.4%` maltose-limited M9 + `1 mM MgSO₄`, `0.1 mM CaCl₂`, `0.2%` B1; $n=12$) vs **LB** broth ($n=10$). *(Source: `D3_Chaudhry2018_PMC6112682.xml`; note host is *E. coli* K12 MG1655 rather than *E. coli* B.)* |
| **D4** | ISEM, Univ. Montpellier, Montpellier, France ($\approx 43.632^\circ\text{N}, 3.863^\circ\text{E}$); Dryad deposit `2016-10` | **Experimental thermal treatments (`28 °C` vs `32 °C`)** across `8` two-day transfers (`days 0–16`): constant **`28 °C`** (`freq0_T28`), constant **`32 °C`** (`freq0_T32`), or square-wave switching **`28 °C <-> 32 °C`** every **`2 d`**, **`4 d`**, or **`8 d`** started at `28 °C` or `32 °C` | Enclosed **static** incubators (**dark / non-photosynthetic**, light unmonitored); unshaken microcosms transferred `1%` every `48 h` | Standard King's B (`KB`) medium across all `30` infected and `20` uninfected populations; exact step-by-step incubator assignment $T_i(t_k) \in \{28, 32\}\,^\circ\text{C}$ recorded in `Duncan_JEB.xlsx` (`CC0 1.0`; publisher PDF HTTP 403 not circumvented per Rule 2). |
| **D5** | Beijing Normal Univ., Beijing, China ($\approx 39.962^\circ\text{N}, 116.366^\circ\text{E}$); Dryad deposit `2017-12` | Constant **`28 °C`** across all `36` infected and `36` uninfected baseline microcosms (`20` transfers = `40 d`) | Enclosed **static** incubator (**dark / non-photosynthetic**, light unmonitored); unshaken `1.5 mL` batch cultures in `24-well` microplates (`1%` transfer every `48 h`) | Three KB nutrient dilutions in M9 buffer: **`1×` KB** ($n=12$), **`0.1×` KB** ($n=12$), and **`0.01×` KB** ($n=12$). *(Source: `D5_Zhao2016_PMC5226335.xml`.)* |
| **D6** | Univ. of York, York, UK ($\approx 53.946^\circ\text{N}, 1.054^\circ\text{W}$); Dryad deposit `2013-06` | Constant **`28 °C`** (`1%` transfer every `48 h`; overnight founder clones shaken at `200 rpm`, selection microcosms static) | Enclosed **static** incubator (**dark / non-photosynthetic**); `6 mL` medium in `30 mL` loose-capped glass universals | Alternating `1× / 0.1×` KB every `1, 2, 4` transfers vs constant `0.55×` KB (**unanalyzable**: `RAW DATA.xlsx` lacks density columns). *(Source: `D6_Harrison2013_PMC3712419.xml`.)* |

### A2.2 Registered covariates, mechanisms, and how they enter the model and tests

Because incubator temperature and light are held strictly constant across replicates within **D1** ($25\,^\circ\text{C}$, `14:10 L:D` at $20\,\mu\text{E m}^{-2}\text{ s}^{-1}$), **D2** ($37\,^\circ\text{C}$, dark/shaken), **D3** ($37\,^\circ\text{C}$, dark/shaken), and **D5** ($28\,^\circ\text{C}$, dark/static), within-study replicate variation in temperature exists **only in D4** ($T_i(t_k) \in \{28, 32\}\,^\circ\text{C}$) and across studies ($T_{\text{lab}} \in \{25, 28, 30, 32, 37\}\,^\circ\text{C}$), while within-day photoperiod cycling operates in photosynthetic **D1** and experimental abiotic nutrient chemistry varies in **D1, D3, D5**. We register three ways these environmental variables enter (`M1`–`M3`), with zero parameters fitted to D1–D5:

1. **`M1` — Statistical adjustment in `T1` (`T1_env`):**
   - For every dataset where an environmental covariate varies across replicates, compute marginal Spearman correlations between the environmental covariate and (i) host log-SD $\hat{\sigma}_{\ln H,i}$ and lag-1 autocorrelation $\hat{\varphi}_i$, and (ii) phage performance ($y_{\text{mean},i}$, $r_{\text{phage,emp},i}$, extinction indicator).
   - Re-evaluate the Freedman–Lane rank-residual partial Spearman test $\rho_{\text{part}}(y_{\text{mean},i}, D_{\text{sim},i} \mid r_{\text{const},i}, Z_i)$ (`10,000` permutations, seed `20261114`; exact $6! = 720$ permutations for D1):
     - In **D4** ($n = 30$): $Z_i = (\bar{T}_i, \sigma_{T,i})$, where $\bar{T}_i = \frac{1}{8}\sum_{k=1}^8 T_i(t_k) \in \{28, 30, 32\}\,^\circ\text{C}$ is mean incubator temperature and $\sigma_{T,i} \in \{0, 2.0\}\,^\circ\text{C}$ is within-replicate thermal SD (plus secondary check adding switching period rank among the $n = 18$ switching lines);
     - In **D1, D2, D3, D5**: within-study temperature and light covariates are constant ($Z_{\text{temp}} \equiv \text{const}$, leaving `T1` algebraically unchanged); we additionally test adjustment for the experimental abiotic medium/schedule covariate $Z_{\text{abio},i}$ (`P-limited` indicator in D1; `30min_pulsed` indicator in D2; `LB` indicator in D3; $\log_{10}[\text{KB}]$ in D5), and across all $N = 104$ units pooled across D1–D5 controlling for $(r_{\text{const},i}, T_{\text{lab},i})$.

2. **`M2` — Mechanistic driver of phage stage durations and thermal/light/stoichiometric vital rates (`notes/mechanism-temperature-incubation.md`):**
   - **`M2a` (Temperature-driven stage duration $L_{\text{lat}}(T)$, adsorption $\alpha(T)$, and thermal inactivation $\mu_0(T)$ in D4 across $28\,^\circ\text{C} \leftrightarrow 32\,^\circ\text{C}$):**
     Independent 4-hour thermal performance measurements on ancestral *P. fluorescens* SBW25 and lytic phage SBW25$\Phi$2 by Padfield, Castledine & Buckling (2020) (*ISME J.* `10.1038/s41396-019-0526-5`, open deposit Zenodo `10.5281/zenodo.3492192`) show that phage $\Phi$2 has a thermal optimum $T_{\text{opt}} \approx 27.1\,^\circ\text{C}$, half-inactivation temperature $T_h = 302.59\text{ K}$ ($29.44\,^\circ\text{C}$), rising activation energy $E = 0.2619\text{ eV}$, high-temperature inactivation energy $E_h = 13.0007\text{ eV}$, and critical thermal maximum $CT_{\max} \approx 29.22\,^\circ\text{C}$ above which net replication turns negative (`+1.47 h⁻¹` at $28\,^\circ\text{C}$ vs `-1.52 h⁻¹` at $30\,^\circ\text{C}$ and `-2.02 h⁻¹` at $33\,^\circ\text{C}$), whereas host SBW25 grows positively up to $\sim 34\,^\circ\text{C}$ (Zhang & Buckling 2011; Chu 2025). With Boltzmann constant $k_B = 8.6173 \times 10^{-5}\text{ eV K}^{-1}$ and $T_K = T + 273.15$, we drive the two-stage phage operator along each D4 replicate's exact $(H_i(t_k), T_i(t_k))$ sequence via:
     1. **Sharpe–Schoolfield intracellular maturation clock** $\nu(T) = \nu_{28}\,\phi_{\text{SS}}(T)$ (so stage duration $L_{\text{lat}}(T) = L_{\text{lat},28}/\phi_{\text{SS}}(T)$), where $\phi_{\text{SS}}(T) = \frac{\exp[(E/k_B)(1/301.15 - 1/T_K)](1 + \exp[(E_h/k_B)(1/T_h - 1/301.15)])}{1 + \exp[(E_h/k_B)(1/T_h - 1/T_K)]}$ uses $(E, E_h, T_h) = (0.2619\text{ eV}, 13.0007\text{ eV}, 302.59\text{ K})$ from Padfield et al. (2020) `[literature/fetch]` (giving $L_{\text{lat}}(28\,^\circ\text{C}) = 35\text{ min}$, $L_{\text{lat}}(30\,^\circ\text{C}) = 103\text{ min}$, and $L_{\text{lat}}(32\,^\circ\text{C}) = 29.8\text{ h}$, swept across $E \in [0.20, 0.65]\text{ eV}$ and $T_h \in [29.0, 31.5]\,^\circ\text{C}$);
     2. **Arrhenius adsorption scaling** $\alpha(T) = \alpha_{28}\exp[(E_\alpha/k_B)(1/301.15 - 1/T_K)]$ with $E_\alpha = 0.20\text{ eV}$ `[literature]` (diffusion/viscosity-limited microbial encounter range $0\text{–}0.45\text{ eV}$, Dell et al. 2011);
     3. **Arrhenius free-virion thermal inactivation** $\mu_0(T) = \mu_{0,28}\exp[(E_\mu/k_B)(1/301.15 - 1/T_K)]$ with $E_\mu = 120\text{ kJ mol}^{-1} = 1.244\text{ eV}$ `[literature/fetch]` (midpoint of $90\text{–}150\text{ kJ mol}^{-1}$ measured across 16 phages in De Paepe & Taddei 2006 *PLOS Biol.* `10.1371/journal.pbio.0040193`, swept across $[90, 150]\text{ kJ mol}^{-1}$ plus a high-heat denaturation regime $\mu_0(32\,^\circ\text{C}) = 12.0\text{ d}^{-1}$ matching Padfield 2020's above-$T_h$ virion loss).
     Following `notes/mechanism-temperature-incubation.md` (§1), we compute for every D4 replicate: (i) constant-$(\bar{H}_i, \bar{T}_i)$ reproduction number $\ln R_{0,\text{i}}$, (ii) arithmetic-mean-rate reproduction number $\ln R_{0,\text{ii}}$ (where entries $\alpha(T)H, \nu(T), \mu_0(T)$ are time-averaged before forming $R_0$ and $r_{\text{const},\text{ii}}$), (iii) curvature shift $J_i = \ln R_{0,\text{ii},i} - \ln R_{0,\text{i},i}$, (iv) exact joint $(H, T)$ threshold fluctuation loss $D_{\text{sim},H+T,i}$ relative to Baseline (ii), (v) Proposition B's discrete staying-rate bound when both $s_P(t)$ and $s_I(t)$ fluctuate with $(H(t), T(t))$, and (vi) **`T1_M2a`**: $\rho_{\text{part}}(y_{\text{mean},i}, D_{\text{sim},H+T,i} \mid r_{\text{const},\text{ii},i})$ across all $n = 30$ D4 populations.
   - **`M2b` (Diel `14:10` light:dark maturation modulation and P-limitation stage-duration shift in D1):**
     In **D1**, we test two documented chamber/medium mechanisms on stage duration $L_{\text{lat}}$: (i) subdividing each $2\text{ d}$ step into `14 h` light ($20\,\mu\text{E m}^{-2}\text{ s}^{-1}$) and `10 h` dark sub-steps with dark-phase maturation factor $\eta_{\text{dark}} \in \{1.0, 0.5, 0.1\}$ (cyanomyovirus dark slowdown on WH7803; Clokie & Mann 2006), and (ii) applying the P-limitation `+18 h` latent-period delay ($L_{\text{lat}}: 9\text{ h} \to 27\text{ h}$) and $80\%$ burst-size reduction ($\beta: 50 \to 10$) measured on P-starved WH7803 by Wilson, Carr & Mann (1996) (`[literature]`, cited in Larsen et al. 2019 p. 1015) to `PL2, PL4, PL5`.

3. **`M3` — Sensitivity analysis in `studies/engine-R16/lifecycle_engine.py` and second-order thermal clock-completion theory:**
   - Call `lifecycle_engine.py` on a two-driver phage cycle (`Driver("host", ..., kind="bounded", low=None)` on $P \to I$ contact adsorption and `Driver("temp", ..., kind="bounded", low=None)` on $I \to P$ maturation yield, coupled via `Description(..., correlation=np.array([[1.0, rho_HT], [rho_HT, 1.0]]))` across the empirical D4 range $\rho_{H,T} \in [-0.6, +0.6]$ and Erlang latent shapes $m_I \in \{1, 4, 16\}$) to quantify how host–temperature correlation shifts `Engine.coefficient()['D2']` and `Engine.loss()['D']`.
   - Evaluate the analytical second-order thermal coefficient $D_2/\sigma_T^2$ (`notes/mechanism-temperature-incubation.md` §3.2, splitting completion sensitivity $\pi_k = d\ln p_k/dT$ and clock sensitivity $\chi_k = d\ln q_k/dT$) at each study's laboratory reference temperature ($25, 28, 30, 37\,^\circ\text{C}$).

### A2.3 Criterion for what would change the study's conclusion

1. **Change of empirical `T1` conclusion to `supports`:** `chemostat`'s overall conclusion on `T1` changes to `supports` only if incorporating verified laboratory environmental covariates (`M1`) or cited temperature/light/stoichiometry stage-duration relations (`M2`) produces $\rho_{\text{part}} < 0$ with one-sided $p_{\text{neg}} < 0.05$ on total host density $H_i(t)$ in at least one dataset not confounded by endogenous phage-induced host crashes ($\sigma_{H,\text{inf}} / \sigma_{H,\text{ctrl}} \le 2$), robustly across the registered thermal parameter range.
2. **Change of D4's full-sample ($n = 30$) verdict from `contradicts` to `not decided`:** If accounting for D4's experimental temperature regime either by covariate adjustment (`M1`) or by the independent Padfield et al. (2020) + De Paepe & Taddei (2006) thermal stage-duration operator (`M2a`) removes the positive partial correlation ($p_{\text{pos}} \ge 0.05$) on the **complete $n = 30$ dataset** without dropping `freq0_T32`, then D4's apparent `contradicts` in the temperature-blind baseline is formally resolved as an omitted-thermal-driver artifact, strengthening (rather than reversing) the study's conclusion that none of D1–D5 contradicts the theory and bulk microcosm tests are `not decided`.
3. **Change of structural `T2a` conclusion:** Any violation ($> 10^{-6}$) of Proposition B's bound under temperature- or diel-light-modulated staying probabilities $s_P(t), s_I(t)$ changes `T2a` from `supports` to `contradicts`.



## Amendment 3: Laboratory environment completion — 2026-11-15T14:06:57Z

This dated addition registers the continuation before any new environmental
association is computed. The supplied copy already contains Amendment 2 and cached
source files, but no environmental analysis output or report section. Their earlier
chronology is not independently certified here. Original sections and results remain
unchanged; `results/weather_original_files.json` records their byte counts and hashes.

**Scope and departure from Amendment 2.** The continuation explicitly permits the
laboratory route: record reported temperature and light regimes, then report their
relevance. All six deposits are laboratory systems; there are no field observations
to locate or match to outdoor weather. This addition supersedes the unexecuted M1–M3
campaign in Amendment 2 for this task. No new thermal model, pooled cross-species
regression, adjusted T1 hypothesis test, or simulation campaign will run. The already
registered T1 and structural results retain their original labels.

**Covariates and mechanism.** Inventory the reported culture temperature and
illumination for D1–D6, distinguishing population experiments from preparation,
storage and assay conditions. Temperature can affect bacterial growth and phage
replication, adsorption and survival; light can affect the photosynthetic D1 host
and infection. Record medium, mixing and transfer schedules as context because they
alter resource supply or loss. These are laboratory treatment assignments, not
measured environmental time series. Missing light regimes, clock times, calibration
uncertainty or calendar dates remain unknown; never infer darkness from incubation
or experiment coordinates from affiliations. Report source sections, raw file hashes,
experimental time coverage and treatment resolution. Grid/station distances are not
applicable. No outdoor products are fetched under the laboratory exception.

**Limited descriptive calculation for step 4.** Reuse the original per-replicate
`results/unit_metrics.csv`; do not refit or overwrite it. Only D4 varies temperature
within a dataset. Match its unit IDs and treatment labels to the deposited workbook.
For its 30 infected populations, tabulate the five treatment groups' sample sizes,
mean and range of host log-density standard deviation (`sigma_lnH`), original mean
log phage density (`y_mean_lnP`) and original final-zero indicator (`went_extinct`).
Compute descriptive Spearman rank correlations of (a) assigned mean temperature
(28, 30, 32 degrees C) and (b) assigned temperature standard deviation (0 or 2 degrees
C) with those three outcomes. Among the 18 switching populations only, also compute
correlations with switching interval (2, 4 or 8 days). The balanced square-wave
summaries concern the full day 0–16 experimental exposure, not instantaneous
measurement temperatures. Confirm that balance from workbook labels and schedules;
if it cannot be verified, omit numerical exposure correlations and report group
summaries only. No phase-resolved temperature–density correlation is registered.
Use average ranks for ties and report null, not zero, for an undefined correlation.
Show ranges rather than claiming inferential confidence or causal effects. Do not
pool species. D1, D2, D3 and D5 have constant assigned temperature within each study;
D6 supplies no density trajectory. D1's shared diel light cycle has no logged
sampling phase; light associations therefore are not identifiable from this audit.

**Verification before new numbers.** Reproduce D4's saved treatment means and
extinction counts in `results/sensitivity_summary.json` from `unit_metrics.csv` at
absolute tolerance 1e-10 for means and exact equality for counts. Verify a known
Spearman example (including ties) against independent average-rank/Pearson
calculation at tolerance 1e-12 before applying it. Check every calculated D4
correlation against that independent implementation at the same tolerance. Preserve
any failed attempt and explain it; do not revise the planned statistic after results.

**How inputs enter, and what could change the conclusion.** Temperature and light
enter this amendment as descriptive treatment covariates only. Quote the original
D4 restriction/thermal sensitivity results explicitly as existing results, without
rerunning or reclassifying T1. No cited relation in the material verified for this
limited audit identifies the latent period as a function of temperature for these
exact evolving cultures. In particular, a net replication curve is not a measured
intracellular development clock: Amendment 2's proposed conversion requires an
additional assumption and is not applied. There is no new adjustment, stage-duration
driver or engine sensitivity run in this laboratory inventory. Discovery of
unaccounted field data or a mismatch between culture assignments and the original
D4 grouping would change this audit's applicability or interpretation and must be
reported as a discrepancy. Descriptive associations alone cannot change the theory
verdict: that would require an independently justified, adequately identified
thermal-rate analysis satisfying the original empirical criteria. A strong
association can reinforce the existing confounding limitation, not rescue T1.

**Budget and stop.** One source audit and one descriptive run, no simulations,
followed by a dated report section with limitations and original-file integrity
checks. Existing Amendment 2 assertions of darkness, precise laboratory coordinates
and a typical chamber tolerance are not treated as measurements. Any further
corrections are added here or in the new report section, never silently rewritten.
