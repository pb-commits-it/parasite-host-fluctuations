# Protocol R17: the decisive experiment, predictions redone

Written 14 November 2026. System clock (`date`) when writing began: 16:11:14. No code for this run existed then. One script (`checks/r17_experiment_predictions.py`), one results file (`results/r17-experiment-predictions_20261114.txt`; a failed first output is kept as `..._attempt1_...`), one note (`notes/decisive-experiment-v2.md`), then stop. Amendments are added at the end with the clock time and the reason. The first design (`notes/decisive-experiment.md`, `checks/experiment_predictions.py`) is not edited.

## 1. Question

The first design predicts what a microcosm experiment would show when host density is switched between a high and a low level at the same arithmetic mean. It assumes three things that are now known to matter:

1. scarcity **wastes** the stage that waits for a host (`notes/contact-limited-transfer.md` gives the other reading, in which it **waits**);
2. every stay is memoryless (`notes/mechanism-temperature-incubation.md`, as corrected after referee 14: a regular stay protects less);
3. the schedule is a square wave (Results 7 and 8 of `manuscript/draft-9.md`: a periodic schedule can resonate with the generation and give a gain).

The question: what does the experiment show under each alternative, and which treatments tell the alternatives apart, at a measurement error of 20% on a count?

## 2. The model (one engine for everything)

**Time.** Continuous time is the primary form. Reasons: (i) the waiting model needs no bound on host density in continuous time, and in daily steps it needs μ + c h ≤ 1; (ii) the refereed second-order formula for the waiting model that is used here is the continuous one; (iii) a nearly fixed stay of 2 days cannot be a chain of 16 sub-stages of at least one day each. Host density is constant between switches, so the solution between switches is a matrix exponential and nothing is integrated numerically. The daily-step form is the same generator A used as I + AΔ with a step of Δ days; with Δ = 1 it is the model of the first design.

**Stages.** In hosts: mean stay a = 5 days, a chain of m_a equal sub-stages, each left at rate m_a/a. Outside hosts: mean stay b days at the mean host density, a chain of m_b equal sub-stages, each left at rate m_b/b at the mean density. Only the last sub-stage outside meets hosts. m = 1 is a memoryless stay, m = 4 a chain of 4, m = 16 "nearly fixed" (coefficient of variation 1/4). Deaths inside the other sub-stages are absorbed in the one constant e^y below.

**The exposed sub-stage** (the last one outside), with h(t) the host density divided by its mean and f the encounter function:

- **waste**: it is left at the fixed rate m_b/b, and the flow into the first sub-stage in hosts is e^y (m_b/b) f(h) per head;
- **wait**, with d the fraction that dies waiting at the mean density: it is left at rate (m_b/b)[d + (1 − d) f(h)], and the flow into hosts is e^y (m_b/b) f(h) per head. With d = 1 this is waste. d = 0.99, 0.9, 0.5, 0.1.

For a chain outside the host this means: sub-stages 1 to m_b − 1 are development, the last one is the search. The time outside at constant mean density then has exactly the stated chain distribution.

**Encounter function.** Proportional: f(h) = h. Saturating (type II, half-saturation at the mean density, scaled so that f(1) = 1): f(h) = 2h/(1 + h).

**y** is ln R0 of the constant treatment (host density at its mean): with h = 1 every variant has R0 = e^y, and the generation time at the threshold is T = a + b.

**Schedules**, all with arithmetic mean density one:

- constant: h = 1;
- square wave of period P: h = 1 + δ for the first half of each period, 1 − δ for the second;
- random telegraph with mean dwell m: h = 1 ± δ, each level left at rate 1/m (dwell times exponential). Its autocorrelation is exp(−2|t|/m), a correlation time θ = m/2;
- three-level random schedule (robustness check): levels 1 − δ₃, 1, 1 + δ₃ with δ₃ = δ√(3/2); each level is left at rate 4/(3m) for either other level with equal chance. It has the same mean, the same variance δ² and the same autocorrelation exp(−2|t|/m) as the telegraph with mean dwell m; its fourth moment is 1.5 times larger.

**Loss.** r(y) is the long-run growth rate per day. D is the y at which r = 0: the parasite invades the fluctuating treatment only if ln R0 > D. D does not depend on R0. For the saturating function D is split: D_curv = −ln of the R0 of a constant treatment held at the mean *rate* E f(h) in place of the mean density, D_dyn = D − D_curv.

## 3. Grid

- Outside stay b = 2, 10, 50 days (short, moderate, long). a = 5 days. Reason: the first design's grid.
- Swings δ = 0.8 (1.8 and 0.2 times the mean) and δ = 0.5 (1.5 and 0.5).
- Time scale k = 1/4, 1/2, 1, 2, 4, 8 generations: square-wave period P = kT; telegraph mean dwell m = kT. (A square wave of period P stays P/2 at each level; this is said wherever the two are set side by side.)
- Variants: d ∈ {1 (waste), 0.99, 0.9, 0.5, 0.1} × (m_b, m_a) ∈ {1, 4, 16}² × f ∈ {proportional, saturating}: 90 variants, full factorial, for the square wave and for the telegraph.
- Three-level schedule: d ∈ {1, 0.5}, shapes (1, 1) and (16, 16), proportional, all b, δ, k: 144 cells. D only.
- A finer set of d (0.1, 0.2, …, 0.9, 0.95, 0.99, 1) for memoryless stays, proportional, telegraph and square wave, D only.

## 4. Methods

**Square wave.** Floquet: the growth rate is ln(dominant eigenvalue of exp(A_low P/2) exp(A_high P/2))/P; D by bracketing root-finding in y (tolerance 1e-12).

**Telegraph, memoryless stays (two dimensions).** Deterministic. The fraction u of the parasites that is outside hosts follows a one-dimensional flow in each level; the stationary law of (level, u) has zero net flux, which gives its density in closed form up to a constant, with algebraic end-point singularities. The growth rate is the stationary mean of the per-head growth of the total, computed with a quadrature rule for algebraic end-point weights. D by root-finding in y. "Computed", not proved here.

**Telegraph with chains, and the three-level schedule.** Simulation of the model without any expansion: the level sequence and the exponential dwell times are drawn, and the population vector is carried exactly through each dwell (eigen-decomposition of the generator of each level; checked against the matrix exponential, V6). Units of account: cycles of two consecutive dwells. Estimator of the growth rate: [mean log growth per cycle − β·(mean of the centred dwell-time controls)]/(mean cycle length), the controls being the time spent at each level in the cycle minus its known expectation, β by least squares. Log growth is measured with the left dominant eigenvector of the constant-treatment matrix. Standard error from 20 batch means. The root in y: a pilot (short run, y on a wide grid, polynomial fit), then the full run at three values of y spaced 0.04 round the pilot root with the same random numbers, a quadratic through the three; the standard error of D is the standard error of the growth rate at the middle value divided by the slope. A fourth value, y = 0.5, gives the growth rate at the stated R0. If the root falls outside the three values the cell is rerun centred on the new root. The numbers of cycles are fixed after a timing test (speed and standard errors only) and recorded in an amendment before the final run. Target: standard error of D at most the larger of 0.003 and 3% of D; cells that miss it are marked in the output, not dropped.

**Growth rate at a stated R0.** r_const − r_fluct per day at ln R0 = 0.5 (R0 = 1.65), the value used in the first design.

**What an experimenter would see: actual runs, not extrapolated rates.** The model is run for 60 days and read at days 30 and 60.

- Start: the stage composition of an infection that has been growing in the constant treatment at ln R0 = 0.5 (its stable composition), the same in every treatment. In practice: establish the infection at constant mean density, then split the culture.
- Count: all parasites, in hosts and outside (the script also carries the count in hosts alone).
- Square wave: 16 starting phases ψ = jP/16, j = 0 … 15; phase 0 is the start of the high half. Reported: the geometric mean over phases of N_const/N_fluct, and its smallest and largest value.
- Telegraph: 400 schedules per cell (200 start high, 200 start low, dwell times exponential from day 0). Reported: the geometric mean of N_const/N_fluct, its 10th and 90th percentiles, and the standard deviation of its logarithm.
- Sensitivity, small tables only: the same at ln R0 = 1.5; and a start with every parasite in the first sub-stage outside the host.

**Second-order values used in the checks.**
- Gain-only, random schedule with exponential covariance (Result 1 summed for a ring, in continuous time; the manuscript's equation (9) per generation): D₂ = δ²[θ/T − R/(1 − R)], R = Π_k 1/(1 + τ_k/θ) over all sub-stages, τ_k their mean stays.
- Gain-only, square wave (Result 7): D₂ = −Σ over odd j of (ε_j²/2) Re[g_j/(1 − g_j)], ε_j = 4δ/(πj), g_j = Π_k 1/(1 + iω_jτ_k), ω_j = 2πj/P; summed to j = 20,001.
- Contact-limited, random schedule (C2 of `notes/contact-limited-transfer.md`): D₂ = δ²{d²θ/T − d[R − (1 − d)q_j]/(1 − R)}, q_j = 1/(1 + τ_j/θ), j the exposed sub-stage.

## 5. Validation checks (tolerances fixed here, before any run)

| # | Statement | Tolerance |
|---|---|---|
| V1 | The engine in daily steps (Δ = 1, waste, memoryless, proportional, period rounded to an even number of days as in the first script) reproduces three numbers printed in `results/experiment-predictions_20261114.txt` for swings 1.8/0.2: D = 0.1511 (b = 2, period 2 generations), 0.0634 (b = 10, 1 generation), 0.0619 (b = 50, 8 generations) | every printed digit: difference below 0.00005 |
| V2a | Gain-only, square wave, δ = 0.05: D against Result 7. Six designs (shape m_b, m_a; b; P): (1,1; 10; T/2), (1,1; 2; 2T), (1,1; 50; 4T), (4,4; 10; 2T), (16,16; 10; T), (16,1; 2; 4T) | 2% relative, or 2e-7 absolute, each |
| V2b | Gain-only, telegraph, δ = 0.05, memoryless stays (quadrature): D against Result 1 in the form above. b = 2, 10, 50 × m = T/2, 2T | 2% relative each |
| V2c | Contact-limited, telegraph, δ = 0.05, memoryless stays (quadrature): D against (C2). d = 0.1, 0.5, 0.9 × b = 2, 10, 50 × m = T/2, 2T: 18 designs | 2% relative each |
| V2d | Chains, telegraph, by simulation at δ = 0.2: shape (4, 4), b = 10, m = 2T; gain-only against Result 1 and contact-limited d = 0.5 against (C2) with j the last sub-stage outside | 4 standard errors + 6%, each (the criterion of check T6 of the contact-limited note) |
| V3 | The bound D ≤ ℓ = −½[ln(1 + δ) + ln(1 − δ)] for every gain-only variant with fixed staying (waste, proportional; 9 shapes × 3 b × 2 δ × 6 time scales) under the square wave and the telegraph; and D ≤ −⅓[ln(1 + δ₃) + ln(1 − δ₃)] for the waste cells of the three-level schedule | no violation; deterministic values to 1e-9, simulated values to 4 standard errors |
| V4 | Continuous time against steps, two cases, square wave, δ = 0.8, b = 10, P = 2T = 30 days: (1) waste; (2) wait, d = 0.5. The step form with Δ = 1, ½, …, 1/64 day tends to the continuous value | Δ = 1/64 within 2% of the continuous D; the extrapolation 2D(1/64) − D(1/32) within 0.2% |
| V5 | Simulation against quadrature, telegraph, memoryless stays, δ = 0.8: waste and wait d = 0.5 × b = 2, 10, 50 × m = T/2, 4T: 12 cells, the simulation code being the one used for chains | 4 standard errors + 0.0005 each |
| V6 | Internal: (a) δ = 0 gives D = 0 (six variants, 1e-10); (b) eigen-propagation equals the matrix exponential for every matrix used in simulation and in the 60-day runs (1e-8 relative); (c) in the quadrature the two levels carry equal mass (1e-6 relative); (d) a 60-day-run routine started on the Floquet eigenvector grows by exactly the Floquet rate over 40 periods (1e-9 relative, three cells) | as stated |
| V7 | A known exact result: with nobody dying while waiting (d = 0, proportional) the loss is zero at any amplitude (`notes/exact-limits.md`, H1 table). Square wave, quadrature, and one simulated chain | 1e-8 (deterministic); 1e-6 (simulated: the conserved total makes every path exact) |

The bound for the saturating function, D ≤ −E ln f(h) (Proposition 21 applied to the rate), is reported for the waste variants and is not one of the gates the task names.

A check that fails stops the use of the numbers it covers. The output is kept as attempt 1, the cause is found, an amendment is written here, and the script is rerun.

## 6. What is open (no pass or fail)

| # | Question | What is reported |
|---|---|---|
| O1 | Do the three predictions of the first design survive? (i) a loss in every cell; (ii) the loss grows with the time scale; (iii) at one generation or more a 50-day outside stage loses less than a 2-day one | counts of variants and cells for and against, by schedule |
| O2 | Where does the square wave give a gain (D < 0), and does the telegraph ever | every cell with D < 0 listed |
| O3 | How far does D move with d, with the shapes, with the encounter function | the tables of section 3 |
| O4 | Does the three-level schedule give the telegraph's D | the difference, with standard errors |
| O5 | The design question (section 7) | best pairs and replicate numbers |

## 7. The design question: definitions

- **Treatment**: the constant schedule, or one fluctuating schedule (square or telegraph, δ, k), on one parasite (one b).
- **Outcome of a treatment**: the mean over replicates of ln(count) at day 30 or at day 60. Replicates of a square-wave treatment are spread evenly over the 16 starting phases; replicates of a telegraph treatment have independent schedules. Write L(X, treatment, day) for the predicted mean of ln(N_const/N_fluct) under alternative X; L = 0 for the constant treatment.
- **Measurement error**: coefficient of variation 20% on a count, so a standard deviation σ_m = √ln(1.04) = 0.198 of ln(count) per replicate.
- **Two alternatives A and B, two treatments T1 and T2** (either may be the constant one): the predicted contrast is L(X, T2) − L(X, T1). The pair of treatments sought is the one for which the contrast differs most between A and B; call the difference Δ.
- **Replicates**: the smallest n per treatment with Δ ≥ 4 σ_m √(2/n), n = ⌈32 σ_m²/Δ²⌉. This counts measurement error only and assumes each replicate's own schedule is used in the prediction. A second number adds the spread between starting phases or between schedules as if it were error: n = ⌈16 (2σ_m² + s₁² + s₂²)/Δ²⌉, s the standard deviation of ln N across phases or schedules in each treatment (the larger of the two alternatives' values).
- **Pairs of alternatives**: (a) waste against wait with d = 0.1, 0.5, 0.9, 0.99 (memoryless, proportional; also with shapes (16, 16)); (b) memoryless against (16, 16), (4, 4), (16, 1), (1, 16), under waste and under wait d = 0.5; (c) proportional against saturating, under waste and under wait d = 0.5, memoryless; (d) each variant against "no effect of variability at equal mean" (L = 0 in every treatment). Each for b = 2, 10, 50.
- **Cannot be told from no effect**: a treatment whose |L| is below the Δ that 12 replicates resolve (0.32), at both days.
- For (c) and (d) a second, pre-specified contrast is reported: a fast against a slow telegraph with the same levels (k = 1/4 against k = 8). The two have the same distribution of density, so the change of the mean rate under a curved encounter function cancels; what is left is the part the theory is about.

## 8. Candidate systems

Only from files in the repository (`data/INVENTORY.md`, `INVENTORY2.md`, `INVENTORY3.md`; cards in `notes/literature/sweep2/`). Each statement carries the label the file gives it (read in full; abstract; from the fetched data; inferred; recalled). Anything recalled and not in the files is listed apart.

## 9. Budget and stop

At most 3 processes. The final script under 15 minutes of wall time. Seeds fixed in the script. One run after the timing test; one report. If a validation check fails: attempt kept, amendment, rerun. No number enters the note that the script did not print.

## 10. Words

"Proved" only where a proof is written and cited. Everything this script produces is "computed" (deterministic) or "simulated" (with a standard error).

## Amendment 1 (system clock 16:34, 14 November 2026; written before the final run)

Three timing tests of the script were made. The first ran 3 of the 18 (b, k) tasks at 2,000 cycles. Its output is kept as `results/r17-experiment-predictions_attempt1_20261114.txt`: it ends in "FAILED" because the 12 cells of check V5 were not among the three tasks run, so V5 had nothing to compare; every other check passed. The second and third ran the whole grid at 4,000 and 3,000 cycles; all checks passed in both. Their outputs are not kept and no number from any of the three is used. I did read them, and items 4 to 6 below were decided after reading them. That is said here so that it is not mistaken for a blind choice.

1. **Method (section 4).** The eigen-decomposition named there is ill-conditioned for chains of 16 sub-stages: in a test made before any result was computed it differed from the matrix exponential by 5% (condition number of the eigenvectors about 6e15). It is replaced by products of precomputed matrix exponentials. For the simulations the dwell times are exponential and rounded to m/65536. For the 60-day telegraph runs the switching times are rounded to 1/1024 day (84 seconds). The square wave's 60-day runs use the matrix exponential of each piece, with no rounding. Check V6(b) now reads: these products equal the matrix exponential, same tolerance (1e-8).
2. **Sizes.** Pilot 1,000 cycles, final 60,000 cycles, burn-in 100 cycles, 20 batches. Check V2d uses the same final size.
3. **Number of telegraph schedules in the 60-day runs: 2,000 per cell (1,000 start high, 1,000 start low), not 400.** With 400 the geometric mean wobbled by several per cent between neighbouring cells.
4. **Check V7, simulated part.** It is checked as the simulated growth rate at ln R0 = 0, times the generation time, below 1e-6 in absolute value (the total is conserved on every path), and not as a root in y, whose interpolation error is larger than that.
5. **The design question (section 7), additions.** The search of section 7 is still reported, marked "any". Added, because the unrestricted search chose schedules that hardly switch within a 60-day run, which compares constant densities and not variability:
   - *admissible* schedules: a square wave with at least two periods in 60 days (period ≤ 30 days), a telegraph with mean dwell ≤ 15 days; the best pair is reported among these, by the largest Δ and by the smallest n*;
   - *held densities*: 1.8, 0.2, 1.5, 0.5 times the mean for the whole run, as explicit treatments of their own (not variability treatments), for pairs (a) to (c);
   - a third replicate number, n_s = ⌈32 σ_m²/(Δ² + s_Δ²)⌉, for an analysis in which every replicate is compared with the prediction for its own phase or schedule; s_Δ is the standard deviation over phases or schedules of the difference between the two alternatives;
   - everything for two counts: all parasites, and parasites in hosts;
   - "cannot be told from no effect" is counted among admissible treatments;
   - the fast-against-slow contrast is also given for the slowest admissible schedule.
6. **Added tables.** The first design's whole table recomputed in daily steps beside the continuous-time values at the same periods (Table 0b). For the three-level schedule, D as a fraction of that schedule's own bound −E ln h. The list of gains under the square wave is limited to D < −0.001; the count of all negative cells is still given.
7. The three-level schedule is as in section 2. Its levels for the 1.8/0.2 swing are 1.980, 1, 0.020.

## Amendment 2 (system clock 16:46, 14 November 2026; written before the second complete run)

The first complete run (16:34 to 16:39, 308 seconds, all checks passed) is kept as `results/r17-experiment-predictions_attempt2_20261114.txt` and is superseded. It did not fail. Two things were found on reading it.

1. **The telegraph means of the 60-day runs were too noisy at slow schedules.** With 2,000 unpaired schedules the standard deviation of ln(N_const/N_fluct) across schedules reached 7.31 (its table (e): b = 2, dwell 8 generations), so the geometric means in its Tables 2 and 3 carried a sampling error of up to about a sixth, and none was printed. The schedules are now drawn in **mirrored pairs**: the same switching times, once starting high and once starting low. Over the 2,000 runs of a cell the time spent high and low is then balanced exactly, and the standard error of each mean, from the 1,000 pair means, is printed.
2. **The design question.** A replicate number for mirrored pairs (nm) is added: pairs drawn independently, their spread treated as error. The search by "smallest n*" becomes a search by "smallest nm". n_s is no longer printed for pair (d): compared replicate by replicate with "no effect", a schedule that stays high or low only shows that growth follows the level of density.

Nothing else is changed: the model, the grid, the checks and their tolerances, the simulation sizes and the seeds are the same. D, the growth rates and every check should come out as in the first complete run; this is verified after the run by comparing the two outputs, and the result is stated in the note.
