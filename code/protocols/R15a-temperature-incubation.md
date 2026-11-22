# Protocol R15a: temperature and incubation time

Written 14 November 2026. System clock (`date`) when writing began: 11:33:55. Time stamp of the file when first saved: 11:35:20. No code for this run existed then. One run, one results file (`results/r15a-temperature-incubation_20261114.txt`), one note (`notes/mechanism-temperature-incubation.md`), then stop. Amendments, if any, are added at the end with the clock time and the reason.

## 1. Question

Four conclusions of the manuscript (`manuscript/draft-8.md`) are tested against two mechanisms it leaves out.

- H1 (cost). A fluctuating obligate transfer lowers ln R0 compared with the same cycle with that rate at its arithmetic mean.
- H2 (parking). Lengthening the stay in a stage whose rates do not fluctuate lowers that loss, for long enough stays.
- H3 (number). What matters is time and exposure, not the number of hosts.
- H4 (bound). With fixed staying rates, D ≤ Σ σ_k²/2 over the fluctuating transfers.

Mechanism 1 is temperature: one driver, several rates, curved responses, and staying rates that move with the transfers. Mechanism 2 is incubation time: stays that are nearly fixed and not geometric, with or without death during the stay. For each conclusion the aim is the condition at which it stops holding, with one exactly computed example on each side.

## 2. Definitions used throughout

The ring: x_{k+1}(t+1) = s_{k+1}(t) x_{k+1}(t) + g_k(t) x_k(t). One step is one day in Part 1 and in Part 2(c). In Part 2(a) and 2(b) a step has no fixed length.

- r: long-run growth rate per step.
- R0 of a constant matrix: Π g_k/(1 − s_k).
- ln R0*: the log of the factor by which one transfer must be divided for r to be zero.
- Baseline (i): every rate at its value at the mean temperature Tbar. R0_i.
- Baseline (ii): every matrix entry at its arithmetic mean over the temperature distribution (or over one period). R0_ii. This is the convention of the manuscript.
- Curvature part J = ln R0_ii − ln R0_i. Fluctuation part = −D, with D = ln R0_ii − ln R0*. Total = ln R0* − ln R0_i = J − D. The identity is exact.
- An exposed stage k is described by two functions of temperature: q_k(T), the chance of leaving the stage in one step (by development or death), and p_k(T), the chance that a leaver goes on to the next stage. Then s_k = 1 − q_k and g_k = b_k p_k q_k. With a development (or contact) rate ν and a death rate μ per day, both constant within a day: q = 1 − exp(−(ν + μ)), p = ν/(ν + μ). I call q the clock and p the completion probability.
- π_k = d ln p_k/dT, χ_k = d ln q_k/dT, at Tbar.
- τ_k = 1/(1 − s_k) mean stay, 𝒯 their sum, θ = φ/(1 − φ), ρ_k = E[φ^(stay in k)] = 1/(1 + τ_k/θ) for a geometric stay, R = Π ρ_k, R_{k→j} = product of ρ over the stages after k up to and including j, going round the ring.

## 3. Models and parameter values

Every thermal parameter below is ILLUSTRATIVE. The repository holds no measured thermal response. Two scales are set to lie inside ranges that are in the repository: development of about 12 days at 25 °C (extrinsic incubation 10 to 14 days) and a vector life of about 13 days at 25 °C (10 to 20 days); both from `verification/systems/SYSTEMS.md`, the *P. falciparum* row, which cites the source read in full there. They fix a scale only. The shapes, limits and Q10 values are mine.

**Temperature.** T(t) = Tbar + σ_T z(t), z a Gaussian AR(1) series with unit variance and coefficient φ per day. φ = 0.8 stands for a memory of days (θ = 4 days), φ = 0.99 for a memory of a season (θ = 99 days), φ = 0 for none. σ_T = 2, 5, 8 °C. Seasonal case: T(t) = Tbar + A_T sin(2πt/P), P = 365 days, A_T = √2 σ_T (same variance): 2.83, 7.07, 11.31 °C. All illustrative.

**Model TPC (three stages).**
1. Sheltered, in a warm-blooded definitive host: stay 30 days (s = 1 − 1/30). No rate depends on temperature. Illustrative.
2. Free-living or in a cold-blooded host. Development ν₂(T) = c T (T − 10) √(35 − T) between 10 and 35 °C and zero outside (Briere), c = 7.0e-5, so ν₂(25) = 0.083 per day. Death μ₂(T) = 0.03 exp(ln 2 · (T − 20)/10) per day (Q10 = 2). Illustrative.
3. Vector or cold-blooded intermediate host. Contact a₃(T) = c T (T − 8) √(40 − T) between 8 and 40 °C, c = 1.2e-4, so a₃(25) = 0.198 per day. Death μ₃(T) = 0.05 exp(ln 2.5 · (T − 20)/10) per day (Q10 = 2.5), 0.079 per day at 25 °C. Illustrative.

Mean temperatures for the table: 14, 23 and 30 °C. Chosen by hand before any run: near the lower limit, near the optimum of R0(T) (p₂ peaks near 24 °C and p₃ near 22 °C by hand calculation), and on the hot limb.

**Model EXP (two stages), for clean tests.** Stage 1 sheltered, stay 20 days. Stage 2 exposed: ν = 0.10 exp(β_ν (T − Tbar)), μ = 0.05 exp(β_μ (T − Tbar)) per day. Cases: E1 β_ν = β_μ = ln 2/10; E2 β_ν = ln 2/10, β_μ = ln 3/10; E3 β_ν = ln 2/10, β_μ = 0; E4 β_ν = 0, β_μ = ln 3/10. All rates stay positive, so every −E ln h is finite. Illustrative.

**Model CHAIN (Part 2a, 2b).** Stage A: geometric stay a = 4 steps, its transfer multiplied by a lognormal host h = exp(σz − σ²/2), z AR(1) with coefficient φ. Stage B: steady, mean duration b, a chain of m identical sub-stages (m = b is a fixed delay of b steps). σ = 1.0, inside the range 0.6 to 1.4 that the manuscript reports for real hosts (`manuscript/draft-8.md`, Result 15). a, b, μ illustrative.

**Model DD (Part 2c).** Stage 1 sheltered, stay 30 days. Stage 2: incubation as a chain of m sub-stages, each left at rate m ν₂(T) per day, death at μ₂(T) per day meanwhile, with ν₂ and μ₂ of model TPC. Temperature is constant within a day and the one-day map of the chain is exact (several sub-stages can be passed in a day). m = 1, 4, 16.

## 4. Methods

**Exact computations.**
- Gaussian means of rates: trapezoid rule on 40,001 points of z in [−10, 10].
- Random environment: simulation of the product of random matrices, the state renormalised every step, the growth rate being the mean log growth per step after a burn-in. The root in y (the log multiplier of one sheltered transfer) is found with common random numbers at three values of y and a quadratic fit, repeated until the root lies inside the three values. Standard error: spread across replicate populations of the growth rate at the root, divided by √(replicates) and by the slope dr/dy. Sizes are fixed after a timing test and recorded in an amendment; the target is a standard error of 1e-3 or less on D in ln R0 units.
- Two code paths. Path A: stage-wise update, noise generated in blocks with a recursive filter. Path B: explicit matrices for every replicate, AR(1) by its recursion one step at a time, another seed, the state normalised by its largest entry. For a fixed delay there is a third: the delay recursion x(t+1) = s x(t) + g h(t−b) x(t−b).
- Seasonal case: Floquet. The product of the 365 daily matrices, its dominant eigenvalue, root in y by bracketing. Second method: iterate the periodic system 400 years and read the growth per year.
- One stage: E ln(s(T) + g(T) e^y) = 0 by quadrature, at any memory.

**Second-order formula.** C″ of `notes/error-bound-general.md`, section 13, coded twice: as a sum over lags (as in `checks/host_crash_kills_parasites.py`, function `terms`), and in closed form through a resolvent, C″ = (wᵀNv)²/2 − φ wᵀN (I − φT)⁻¹ Q N v with N = Σ κ_e B_e. Three versions for temperature:
- (a) tangent: A0 = the matrix at Tbar, κ_e = σ_T d ln(rate_e)/dT at Tbar;
- (b) first Hermite term: A0 = the matrix of mean rates, κ_e = Cov(rate_e, z)/mean(rate_e);
- (c) all Hermite terms: A0 = the matrix of mean rates, and κ_e κ_e′ ρ(m) replaced by the exact covariance of the two rates at lag m, which is Σ_j α_ej α_e′j φ^(jm) with α_ej the j-th normalised Hermite coefficient of rate_e/mean. Version (b) is its first term.
D is taken as 𝒯₀ C″ at the threshold of the matrix used. For (c) I also report the root of ln λ(y) − C″(y) = 0, the same formula applied at the root model. For the seasonal case the same with cos(2πm/P) in place of φ^m (tangent), and with all Fourier harmonics of the daily rates (full).

## 5. Checks of the code against known results (must pass before anything else is used)

| # | Statement | Tolerance |
|---|---|---|
| K1 | One stage, lognormal host, s = 0.75, σ = 1, φ = 0 and 0.9: both simulation paths give the exact D from quadrature | larger of 3 standard errors and 2% |
| K2 | The matrix C″ (both codings) equals the manuscript's equation (2) for one fluctuating transfer, 6 rings, and the two codings agree with each other on 50 random models with several entries on one driver | 1e-10 and 1e-9 relative |
| K3 | Model CHAIN at σ = 0.2, m = 4 and m = 32 (b = 32), φ = 0.5 and 0.9: simulated D equals the second-order value | larger of 5% and 3 standard errors |
| K4 | One exposed stage alone (stage 2 of E4, σ_T = 5, φ = 0.8): both paths give the exact D from quadrature | larger of 3 standard errors and 2% |
| K5 | Seasonal model TPC, 9 cases: Floquet and brute-force iteration agree on ln R0* | 1e-8 |
| K6 | Paths A and B agree on D for model TPC at σ_T = 5, three Tbar, φ = 0.8 and 0.99 | 3.5 combined standard errors, each of 6 cases |
| K7 | Model DD at constant temperature: the completion probability from the one-day map equals (mν/(mν + μ))^m; and for m = 1 the map is the q, p form of section 2 | 1e-12 |

## 6. Claims, Part 1 (temperature)

| # | Claim | Basis | Kind and criterion |
|---|---|---|---|
| T1 | If p_k does not depend on temperature in every exposed stage (death and development respond in the same proportion), then ln R0* = ln R0_ii = ln R0_i at any σ_T and any memory: J = 0 and D = 0. | Derived by hand: at the threshold the reproductive value w satisfies wᵀA(t) = wᵀ for every t. Proof to be written in the note. | Check of a proof. (a) wᵀx(t) constant to 1e-12 along a random path; (b) seasonal, Floquet: \|D\| and \|J\| ≤ 1e-10; (c) E1, σ_T = 8, φ = 0.8 and 0.99: \|D\| ≤ 3 standard errors. |
| T2 | At the threshold and to second order, with one driver of AR(1) memory, D = σ_T² × [ Σ_j π_j² K + Σ_{j<k} π_j π_k K_jk + Σ_k π_k χ_k X_k − Σ_{j≠k} π_j χ_k Y_jk ], with K = (1 + 2θ)/(2𝒯) − R/(1 − R), K_jk = (1 + 2θ)/𝒯 − (R_{k→j} + R_{j→k})/(1 − R), X_k = (ρ_k − R)/(1 − R), Y_jk = R_{k→j}(1 − ρ_k)/(1 − R). The clock enters only through cross terms with a completion probability. | Derived by hand from C″ and the line of descent. | Check of a derivation against the matrix formula, which does not use it: 200 random rings of 2 to 6 stages with 1 to 3 exposed stages, sensitivities of either sign, φ from 0 to 0.99; relative difference ≤ 1e-9. |
| T3 | The matrix formula is the small-variance limit of the exact D, cross terms included. | Theorem G″ gives the coefficient; no remainder bound is proved for thermal responses. | Check. Seasonal, exact: for E2, E3, E4 and model TPC at Tbar = 14 and 30, D_exact/D_formula at A_T = 0.5 °C within 2% of one, and the relative error at A_T = 0.5 between 3 and 5 times that at 0.25 where both exceed 1e-6. Random: E3 and TPC at 14 °C, σ_T = 1, φ = 0.8: within the larger of 5% and 3 standard errors; a case whose 3 standard errors exceed half the formula value is reported as uninformative, not as a pass. |
| T4 | Without memory D ≥ 0 exactly. | Known inequality for independent environments (manuscript, Proposition 21(b)). | Check: D ≥ −3 standard errors in every φ = 0 case. |
| T5 | With memory the fluctuation part can be a gain (D < 0). For one exposed stage, at second order, D < 0 exactly when π χ < 0 and \|χ/π\| > K/X. In E2 and E4 (death more sensitive than development) the formula gives a gain at φ = 0.8 and 0.99 and a loss at φ = 0; in E3 a loss at every φ. In E4 the sign changes near φ = 0.17 (hand estimate; the script computes it). | T2. | Check: the simulated D has the predicted sign by at least 3 standard errors in each of the 9 cases (E2, E3, E4 at φ = 0, 0.8, 0.99, σ_T = 5). Seasonal counterpart, exact: sign of D for E4 as the period goes from 2 to 365 days, reported. |
| T6 | H4 as it stands fails when staying rates fluctuate: D can exceed the sum over fluctuating transfers of −E ln h, and also that sum with the staying rates added unweighted. A corrected bound holds: D ≤ Σ_k ℓᵍ_k + Σ_k (τ_k − 1) ℓˢ_k, with ℓ = −E ln(rate/mean) and τ_k the mean stay at the mean rates. | Corrected bound: the proof of Proposition 21(a) with the staying multipliers kept; to be written in the note. Counterexample: hand estimate for E4. | Check. Counterexample: one exposed stage of E4 alone, exact by quadrature; and E4 as a two-stage ring at φ = 0 by simulation, D above each of the two uncorrected sums by more than 3 standard errors. Corrected bound: D ≤ bound + 3 standard errors in every EXP case. If the counterexample does not materialise the claim is reported as not shown. |
| T7 | Decomposition table: J, −D and the total for model TPC at Tbar = 14, 23, 30; σ_T = 2, 5, 8; φ = 0.8 and 0.99; and the three seasonal amplitudes. Accuracy of formula versions (a), (b), (c). | none | Open. Numbers reported. No criterion. |
| T8 | Boundary (a). To second order, more variable temperature raises ln R0 against baseline (i) exactly when Σ_k [p_k″/p_k + 2π_kχ_k] > 2 D₂/σ_T², with D₂ of T2; without memory the right side is (Σπ_k)²/𝒯. For small rates per step the left side is Σ_k (1 − p_k)(ν_k″/ν_k − μ_k″/μ_k). For model TPC there is a lower boundary near 16 to 17 °C (hand estimate) and an upper one below the upper limit of 35 °C; variability raises ln R0 outside them and lowers it between. | Hand derivation. | Check and open. Seasonal: the exact boundaries by Floquet and root-finding at each amplitude, with the exact total at ±1 °C either side. Random, σ_T = 5, φ = 0.8 and 0.99: the boundary is predicted from exact J and formula (c); the total is simulated at the prediction −2, −1, +1, +2 °C; the boundary counts as bracketed if the total at −2 and at +2 have opposite signs, each by more than 3 standard errors. The second-order boundary is reported beside the exact one; the gap is a result, not a criterion. |
| T9 | Shelter. (i) Replace exposed stage 2 by a sheltered stage with its mean rates: D becomes that of stage 3 alone. Reported as the share of the loss that goes. (ii) Split sheltered stage 1 into two sheltered stages of 15 days each. Second-order prediction: D unchanged without memory; with memory it changes only through R, the transform of the time round the ring. (iii) Sheltered stages enter only through the distribution of the time spent in them between exposed stages; two hosts in a row and two sub-stages in one host are the same matrix. | T2; (iii) by a short argument to be written. | Open for (i) and (ii): model TPC at Tbar = 14 and 30, σ_T = 5, φ = 0.8 and 0.99 by simulation, and φ = 0 at 14 °C; all three Tbar and amplitudes by Floquet. The second-order values are printed beside them. No criterion. |

## 7. Claims, Part 2 (incubation)

| # | Claim | Basis | Kind and criterion |
|---|---|---|---|
| I1 | The manuscript's closed form (2) extends to stays that are not geometric when R is read as E[φ^L], L the time round the ring: for a chain of m sub-stages with mean stay τ, ρ = (1 + τ/(mθ))^(−m); for a fixed delay, ρ = φ^τ. With death at a fixed rate the stays are those of the sub-stages with their staying rates. | A chain of sub-stages is a ring with more stages, so this is equation (2) itself. | Check against the matrix formula: the grid of I3 and 100 random chains; 1e-10 relative. |
| I2 | At second order and fixed mean stays, D does not fall as m rises, and rises strictly if φ > 0. Without memory it does not depend on m. | Proof in two lines ((1 + x/m)^m increases with m), to be written. | Check on the grid of I3: monotone to 1e-12. |
| I3 | Exact D at σ = 1 for a = 4, b = 32, m = 1, 2, 4, 16, 32 (fixed), φ = 0, 0.5, 0.9, 0.99, without death and with death at 0.02 per step in stage B. Second-order prediction for the parking benefit (the exact one-stage D minus D): positive for every m at φ = 0 and 0.5; at φ = 0.9 and 0.99 positive for m = 1 and negative for m ≥ 2. | I1. Nothing at σ = 1. | Open. The sign of the exact benefit, with its standard error, is reported for each cell beside the predicted sign. |
| I4 | If every stay is fixed the ring has no staying and D = σ²/2 exactly, whatever the durations: no parking benefit at all. | Manuscript, Proposition 21(a), equality case. | Check: stage A a fixed delay of 4 and stage B a fixed delay of 32, σ = 1, φ = 0 and 0.9, one cohort followed; larger of 3 standard errors and 2%. Also reported: D with stage A as a chain of 2 and 4 sub-stages and B fixed. |
| I5 | H4 holds for chains: D ≤ σ²/2 in every case of Part 2(a) and 2(b). | Proposition 21(a); staying rates are fixed. | Check: 3 standard errors. |
| I6 | D is the same whichever transfer of the ring fluctuates, at any noise: the transfer out of stage A or the transfer out of the last sub-stage of B. | Short renewal argument, to be written. | Check: two cases (m = 4 and m = 32, φ = 0.9), difference within 3 combined standard errors. |
| I7 | Death at rate μ per step in the steady stage. Net margin N(b) = ln R0(b) − D(b). Fixed delay: ln R0 falls by μ per step of delay, and at second order without memory N is largest at b* = σ/√(2μ) − a (no gain from any delay if that is negative). Geometric stay with the same death rate: b* is the positive root of 2ε c² b² + (4εac − σ²(1+ε)ε) b + (2εa² − σ²(1+ε)) = 0, ε = e^μ − 1, c = aε + 1 + ε. With memory, b* is the largest root of dN/db = 0 with D from I1. | Hand derivation. | Check of the algebra: the closed forms against a direct numerical maximum of the second-order N(b), 1e-6 (geometric) and nearest integers (fixed delay). |
| I8 | The exact b* at σ = 1, a = 4, μ = 0.005: fixed delay at φ = 0 (second-order value 6) and φ = 0.5 (second-order value about 13); geometric at φ = 0 and 0.5. | none at σ = 1 | Open. N(b) on a grid with common random numbers; b* is the grid maximum; reported with the set of b whose N is within 2 standard errors of the maximum. A boundary is claimed only if N at b* exceeds N at one smaller and one larger grid value by more than 3 paired standard errors. |
| I9 | Model DD: effect of temperature variability on the probability of completing incubation, on the mean incubation time of those that complete, and on ln R0 against both baselines, at Tbar = 14, 23, 30, σ_T = 5, φ = 0.8 and 0.99, m = 1, 4, 16, and the seasonal amplitude 7.07. Expectation: below the inflection of ν₂ (near 20 °C) variability shortens incubation and raises the completion probability; near and above the optimum it lowers it. | Rate summation; curvature. | Open. Numbers reported. |

## 8. Budget and stop

At most 3 processes. The final script must finish in under 15 minutes of wall time. Sizes are set from a timing test (recorded in an amendment before the run). If a check of section 5 fails, the method is fixed, the failed output is kept as `results/r15a-temperature-incubation_attemptN_20261114.txt`, and the whole run is repeated. If a claim fails because it is false, that is the result. No parameter value is changed after the full run.

## 9. Not covered by this protocol

Encounters, transmission routes and dispersal (other collaborators). Temperature acting on the sheltered stage through the host. Hosts whose abundance also follows temperature. Density dependence. Any measured thermal response. Continuous time as such (the day step is fine compared with the rates, but no limit is taken). Extinction in finite populations.

## 10. Amendments made before the full run

Written at 12:01:54 by the system clock, after three smoke tests of the whole script at one-tenth length (started 11:51:18, 11:56:39 and 11:59:51 by the clock; outputs kept in the session scratchpad, not in `results/`). The smoke tests were for timing and for the harness. They also showed one-tenth-length values of the open quantities; no claim of sections 6 and 7 was reworded because of them, except where listed here.

1. **Sizes** (replicates × steps after a burn-in of 2,000 steps; a pilot of one-eighth length locates the root first). Temperature rings: 500 × 20,000; the table T7 and the shelter runs T9: 500 × 40,000; small-variance checks T3 and the φ = 0 cases of T5: 500 × 160,000; path B: 250 × 20,000; lognormal-host rings and delays (Part 2): 400 × 25,000 (K3: 400 × 50,000); model DD: 200 × 20,000, cohorts 200 × 10,000 after 3,000. Estimated 1,200 process-seconds. The target of a standard error of 1e-3 on D is not met where φ = 0.99 (expected 2e-3 to 5e-3); the budget of 15 minutes decides.
2. **Two more checks of the code.** K8: the fixed delay of 32 as a ring of 33 stages through path A against the delay recursion (path C), φ = 0.9, σ = 1; 3 combined standard errors. K9: model DD with m = 1 through the chain code against the same ring through path A, 14 °C, σ_T = 5, φ = 0.8; 3 combined standard errors. The runs of I6 and K8 use the seed of the I3 runs at φ = 0.9, so that paired differences can be printed beside the unpaired criterion of the protocol.
3. **T6, a second two-stage example.** In the smoke test the two-stage ring E4 (sheltered stay 20 days, φ = 0) exceeded the first uncorrected bound by 10 standard errors and the second by only 2.5. The one-stage example, which is exact, exceeded both. The protocol's case is kept and judged as written (T6c). A second case is added (T6c′): the same exposed stage with a sheltered stay of 5 days, where the hand estimate of the margin is wider. This case was chosen after seeing the smoke test and is labelled so.
4. **T8.** (a) The grid runs from 10.5 to 34.98 °C, with extra points above 34.5. (b) A boundary closer than 2.5 °C to the upper limit of development (35 °C) cannot be bracketed at +1 and +2 °C; it is simulated at −2 and −1 °C and at one third and two thirds of the way to the limit. (c) The small-variance calculation prints the intervals of mean temperature on which J₂ − D₂ > 0 instead of assuming the pattern "raises outside, lowers between" that the protocol wrote down by hand; the smoke test showed that the hand pattern is wrong for the small-variance limit with long memory (no interval at all at φ = 0.99). That is a result and is reported as one. (d) The boundary of the curvature part alone (J = 0) at σ_T = 2, 5, 8 is added, exact.
5. **T5, seasonal counterpart.** Periods 3, 5, 7, 14, 30, 90, 365 days. Period 2 is dropped: a sinusoid of period 2 sampled at whole days with this phase is identically zero.
6. **Quadrature.** 20,001 points on [−10, 10] (spacing 0.001), not 40,001. Hermite terms kept: 200 (120 for model DD); the share of the variance they capture is computed.
7. **Harness faults found in the smoke tests and fixed.** (a) T1a took its threshold from a numerical root (1e-13); over 5,000 steps that error accumulated to 8e-12, above the tolerance of 1e-12. It now uses the exact threshold e^y = 1/p. The tolerance is unchanged. (b) −E ln(s/mean s) was computed from ln s, which underflows at very high temperature when σ_T = 8; it is now E(ν + μ) + ln(mean s), which is the same quantity. (c) I4: with no staying anywhere the cohorts never mix, and the logarithm of the sum of four of them grows faster than one cohort over a finite run (the bias met in run R10). The delay recursion now reads growth from one sub-stage and, for a pure cycle, between two times a whole number of cycles apart. The first smoke test gave 0.443 ± 0.007 and 0.475 ± 0.008 against 0.5 before this fix. (d) T6 printed "holds" where a bound was infinite through (b); fixed with (b).
8. **I2** is checked without death only. With death the mean stay of the line of descent changes with m, so the comparison at fixed mean stays does not apply.
9. **T3, seasonal:** a period of 30 days is reported as well, as an extra.
10. **I4:** with a = 4, "stage A as a chain of 4 sub-stages" is the fixed delay itself; the intermediate case reported is a chain of 2.
11. **K5:** the brute-force iteration runs 400 years and discards the first 100.

## 11. After the full run

12. **Full run:** started 12:02:03, finished 12:08:25 by the system clock; wall time 381 s on three processes, 1,137 process-seconds; 194 tasks; every check of sections 5 to 7 passed at the first full run, so there is no `_attempt` file. **Two descriptive blocks were added afterwards** (written at 12:08 by the clock), in a helper script `checks/r15a_extra.py` whose output is appended to the same results file under its own heading: T8b, the mean temperatures at which the fluctuation part is a loss or a gain in model TPC (second order, formula (c), and exactly by Floquet for the seasonal case, with an exact value on each side of each sign change); and I3b, the steady stays at which a steady stage lowers the second-order loss, by the shape of the stay. Neither has a pass/fail criterion, neither uses simulation, and neither changes a number of the full run. They were added because the report asks for the boundary of each conclusion and the full run gave these two only at a few points.

13. **Correction to item 12, added at 12:16:40 by the system clock.** The helper script was written while the full run was still in progress (its file time stamp is 12:07), after the smoke tests and before I had seen the output of the full run. It was run once into the session scratchpad at 12:08:38 and then again, appended to the results file, at 12:09:00. So "added afterwards" in item 12 is true of when it was run, not of when it was thought of. Also for the record: after the full run, one scratch diagnostic was run to see whether simulation paths A and B differ systematically (the K6 check had passed with two differences above two standard errors); it showed no difference. It is not part of the results and none of its numbers is quoted.

14. **Added at 12:17:00 by the system clock.** The output of the first smoke test (one-tenth length, started 11:51:19) is now kept as `results/r15a-temperature-incubation_attempt1_20261114.txt`, because it holds the two harness failures of amendment 7 (T1a and I4) as they were printed. Its other two failures (T5 and T6c) were for lack of precision at one-tenth length. Item 12 said there is no `_attempt` file; that was true of full-length runs only.
