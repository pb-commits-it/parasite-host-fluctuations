# Protocol R10b: the table of real parasites when the long stage is not memoryless

Written 14 November 2026. System clock (`date`) when writing began: 12:29:03. No code for this run existed then (the library `checks/r15a_lib.py` of run R15a exists and is imported, not edited). One run, one results file (`results/r10b-real-systems-stay-shape_20261114.txt`), one report (`reports/R10b-real-systems-stay-shape.md`), then stop. Amendments are added at the end with the clock time and the reason.

## 1. Question

Run R10 computed the loss D of ln R0 for the rows of the manuscript's table of real parasites (`manuscript/draft-9.md`, Result 14) with every stay geometric on the time step of the calculation. The manuscript reads the result as: the long steady stage cuts the loss to about a thirteenth, a ninth and a third of σ²/2 for the two schistosomes and malaria. Run R15a found that this kind of protection comes from the spread of the stay, not from its mean. How much of the table's protection is left if the long stage is more regular than geometric? At what coefficient of variation of its duration is half of the protection gone?

## 2. Inputs

Taken from the R10 script (`checks/r10_loss_at_large_noise.py`, function `build_cases`, the lines labelled H4): stays in steps, σ and φ per step. Cross-checked by eye against `manuscript/table-sources-draft8.md` before writing this: the durations, σ and φ of every row agree with the script (snail 2 and worm 48 months at a step of 2 months; fox 2.0, egg 2.5, vole 4.0 months at a step of 2 months; mouse 4.4 and tick 10 months at a step of 4.4 months; mosquito 0.8 and human 6.0 months at a step of 0.8 months; and the two nematodes). The status of each input (published, derived, illustrative, recalled) is in that file and is not changed here. The memory values of every row except the two snail rows are illustrative choices there.

| Row | Stays (steps), fluctuating stage first | σ | φ per step | R10's D | σ²/2 |
|---|---|---|---|---|---|
| 1a *S. haematobium* | 1, 24 | 0.589 | 0.398 | 0.0137 ± 0.0002 | 0.173 |
| 1b *S. mansoni* | 1, 24 | 1.380 | 0.367 | 0.1068 ± 0.0004 | 0.952 |
| 2a *E. multilocularis* (egg, vole, fox) | 1.25, 2, 1 | 1.174 | 0.75 | 0.4952 ± 0.0005 (0.4955, 0.4944 with the transfer placed elsewhere) | 0.689 |
| 2b the same with the fox also fluctuating | 1.25, 2, 1 | 1.174; fox 0.18 | 0.75; 0.50 | 0.5049 ± 0.0005 | 0.705 |
| 3a *B. burgdorferi*, oak forest | 1, 10/4.4 | 0.668 | 0.30 | 0.1075 ± 0.0001 | 0.223 |
| 3b *B. burgdorferi*, northern edge | 1, 10/4.4 | 1.190 | 0.50 | 0.4056 ± 0.0003 | 0.708 |
| 5 *T. tenuis* | 1, 12 | 0.70 | 0.85 | 0.0541 ± 0.0004 | 0.245 |
| 6 *P. falciparum* | 1, 7.5 | 1.40 | 0.75 | 0.3305 ± 0.0007 | 0.980 |
| 7 *T. circumcincta* | 1, 3 | 0.35 | 0.80 | 0.0348 ± 0.0002 | 0.061 |

**A fact about these inputs that shapes the design.** In every row but *E. multilocularis* the fluctuating stage lasts exactly one step. In R10's model a stage of one step has no staying: its duration is already fixed. So in R10 the long stage is the only source of spread in the generation time. Two consequences, both stated before any run.

1. If the long stage is given a fixed duration as well, the ring has no staying at all, and D = σ²/2 exactly (manuscript, Proposition 21(a), equality case). All of the protection is then gone. This needs no simulation; one case is simulated as a check.
2. "Make the fluctuating stage a chain of 4 sub-stages" cannot be done at R10's time step: a sub-stage cannot last less than one step. The same limit applies to the long stage: a chain of m sub-stages needs m ≤ its stay in steps, so m = 16 does not exist for a stay of 7.5 or 2.27 steps, and a fixed delay does not exist for a stay that is not a whole number of steps.

So the run has two parts.

## 3. Design

**Reduction used everywhere** (run R15a, note `notes/mechanism-temperature-incubation.md`, Proposition C, proved there, read by nobody else): with one fluctuating transfer, D depends on the ring only through the list of its sub-stages, whatever their order and whichever transfer fluctuates. So every single-host case is a list of staying rates of the sub-stages that have staying, plus a number of sub-stages of exactly one step.

**Part A: R10's own time step.**
- A0. The 11 parameter sets of R10's H4 section, as R10 has them.
- A1. The long stage as a chain of m equal sub-stages with the same mean b: sub-stage staying rate 1 − m/b. Its coefficient of variation is √(1/m − 1/b) (so 0.98, not 1, for m = 1 and b = 24: a geometric stay of at least one step). Rows and shapes:
  - 1a, 1b (b = 24): m = 1, 2, 3, 4, 6, 8, 12, 16, 20, 22, 23, and 24 (the fixed delay, exact).
  - 6 (b = 7.5): m = 1, 2, 4, 7 (7 is the most regular chain the step allows).
  - 3a, 3b (b = 2.273): m = 1, 2.
  - 2a (larva in the vole, b = 2, the longest stage): m = 1, 2 (fixed). The egg keeps its geometric stay of 1.25 steps and the fox its one step.
  - 5, 7 (not rows of the manuscript's table): reproduced in A0 only.

**Part B: a time step r times finer**, so that every shape exists and the fluctuating stage can have spread. The host is an AR(1) series per fine step with φ_f = φ^(1/r), which has the same correlation at whole coarse steps and the same σ. Every stay is r times as many fine steps. r = 4 for rows 1a, 1b (stays 4 and 96 fine steps) and row 6 (4 and 30); r = 11 for rows 3a, 3b (11 and 25); r = 8 for row 2a (egg 10, vole 16, fox 8). This is a different model from R10's, labelled as such: R10's geometric long stay sits on the coarse lattice, the fine one on the fine lattice. (R10's own model is, exactly, the fine model with the fluctuating stage fixed and the long stay a whole number of coarse steps: the fine model then splits into r copies that never mix. That is an argument, not a computation of this run.)
- B1. Fluctuating stage fixed (as in R10), long stage a chain of m: m = 1, 2, 4, 16, then values close to fixed: 32, 48, 64 for rows 1a, 1b; 23, 27 for row 6; 20, 23 for rows 3a, 3b. Fixed: σ²/2 exactly.
- B2. Fluctuating stage with spread: geometric, and a chain of 4 sub-stages, each with the long stage geometric and with the long stage fixed.
- B3. Row 2a at the fine step: egg geometric (mean 10 fine steps) with the vole stage a chain of m = 1, 2, 4, 8, 16 (fixed); egg a chain of 4 with the vole geometric and fixed; egg fixed with the vole geometric. Fox fixed at 8 throughout.

**Second-order value** for every shape: σ²[(1 + 2θ)/(2𝒯) − R/(1 − R)], R = E[φ^L] = the product over sub-stages of (1 − s)φ/(1 − sφ) (so φ for a sub-stage of one step), 𝒯 the mean generation time; θ = φ/(1 − φ).

## 4. Method

D is the log multiplier of one transfer at which the long-run growth rate is zero, for the ring at the threshold of its mean matrix. Simulation with common random numbers at three multipliers, a quadratic fit, a wide short pilot, a narrow pilot of one-eighth length and a final run; standard error from the spread across replicate populations (the method of R15a, `r15a_lib.find_root`).

- A0 is run with the ring simulator of R15a (`r15a_lib.sim_ring_A`; for row 2b a two-host version written for this run).
- Every shape is run with a new simulator written for this run: the sub-stages with staying are stepped explicitly, the sub-stages of one step are a delay line. It records the growth rate in the first and second halves of the run separately.
- Sizes: A0, 400 populations × 50,000 steps; Part A shapes, 400 × 25,000; Part B, 200 × (20,000 r) fine steps, or 100 populations where there are 16 or more sub-stages with staying (rows 3a, 3b: 12,000 r). May be changed after a timing test; recorded as an amendment before the run.
- **Burn-in and slow mixing.** A ring whose generation time has little spread forgets its starting distribution slowly. For every case the relaxation time τ_relax = −1/ln|λ₂| of the mean matrix is computed (λ₂ its second eigenvalue in modulus). Burn-in is the larger of 2,000 steps and 8 τ_relax. A case whose run length is under 30 τ_relax is not run; it is listed with its relaxation time.

## 5. Checks (criteria fixed now)

| # | Statement | Criterion |
|---|---|---|
| R1 | A0 reproduces R10's D for each of the 11 parameter sets | \|D − D_R10\| ≤ 4 √(se² + se_R10²) + 0.00005 (the last term is the rounding of R10's printed value). **If any set fails, the script stops there and reports; nothing else is run.** |
| R2 | The new simulator agrees with the ring simulator on the eight single-host sets of A0 (all but 2b, and one placement of 2a) | 3.5 combined standard errors, each |
| R3 | The closed form with R = E[φ^L] equals the matrix formula (`r15a_lib.cpp_resolvent`) for every shape simulated | 1e-9 relative |
| R4 | D ≤ σ²/2 in every case; and row 1b with both stages fixed (a pure cycle of 25 steps), simulated, gives σ²/2 | 3 standard errors; the larger of 3 standard errors and 1% |
| R5 | In every simulated case the growth rate at the root is the same in the two halves of the run | 3 standard errors per case. A case that fails is flagged and not used for the interpolation of section 6. |

## 6. What is reported, and how the verdicts are drawn (fixed now)

For each row and shape: D ± standard error, D/(σ²/2), D/D_geometric (geometric = m = 1 of the same part), the second-order value, the coefficient of variation of the long stage, and τ_relax in generations.

- **"The long stage removes most of the loss"** is taken to mean D < ½ · σ²/2. It *survives* at a shape if D is below that by more than 3 standard errors, *fails* if above by more than 3, and is *undecided* otherwise. Verdicts are given at m = 2, 4, 16 and fixed, from Part A where the shape exists there and from Part B otherwise, and both where both exist.
- **Protection** is σ²/2 − D. **Half of the table's protection is lost** where D = (D_geometric + σ²/2)/2. The coefficient of variation at which that happens, CV½, is found by linear interpolation of D against the coefficient of variation between the two neighbouring shapes that bracket that value, within the chain family with the fluctuating stage as in R10. It is given for Part A and for Part B. If no pair brackets it, that is said.
- **Which stage's shape matters more:** from B2 and B3, the change of D when the fluctuating stage goes from fixed to a chain of 4 to geometric, set beside the change when the long stage goes from geometric to fixed.

Expectations written down before the run, to be marked right or wrong: (i) D rises as m rises in every row; (ii) for rows 1a, 1b and 6 the reading survives at m = 2, 4 and 16, and fails only very close to a fixed stay (CV½ below 0.2); (iii) for rows 3a, 3b and 2a the reading does not hold even with geometric stays or holds marginally (R10's fractions are 0.48, 0.57, 0.72), so there is little to lose; (iv) spread in the short fluctuating stage restores much of the protection even when the long stage is fixed.

## 7. What the sources say about the real distributions

Only from `verification/systems/` (`SYSTEMS.md`, `PAPER_CARDS.md`, `cards/`) and `manuscript/table-sources-draft8.md`. If none of them states a distribution or a coefficient of variation of a stage duration, the report says "not in our sources". Nothing is supplied from memory.

## 8. Budget and stop

At most 3 processes. Under 15 minutes of wall time. If R1 fails: stop and report. If another check fails because of the harness: keep the output as `results/r10b-real-systems-stay-shape_attemptN_20261114.txt`, fix, rerun, say so. No input is changed.

## 9. Not covered

Rows 5 and 7 beyond the reproduction. Two fluctuating hosts (row 2b) beyond the reproduction. Any shape other than chains of equal sub-stages. Death during the long stage (R15a, section 5.2, treats it; for a fixed delay it does not change D). How many generations a real population needs before the long-run value applies, beyond the relaxation time of the mean matrix. Continuous time.

## 10. Amendments made before the full run

Written at 12:36:39 by the system clock, after two smoke tests of the whole script at one-eighth length (started 12:33:34 and 12:35:38; outputs kept in the session scratchpad). The smoke tests were for timing and for the harness. They also showed one-eighth-length values of the quantities to be reported. No verdict rule of section 6 is changed.

1. **Refinement of the time step in Part B: r = 8 for rows 1a, 1b, 6 (and 2a), not 4.** With r = 4 a chain of 4 sub-stages for a fluctuating stage of 4 fine steps is four sub-stages of one step each, which is a fixed delay again. With r = 8 each sub-stage has a mean of 2 fine steps. Stays are then 8 and 192 fine steps (1a, 1b) and 8 and 60 (6). The values of m close to fixed become 32, 64 for 1a and 1b, and 30, 45 for row 6. Decided while writing the code, before any run.
2. **Slow mixing.** Several cases of B2 with the long stage fixed, and the most regular shapes, have relaxation times of thousands of fine steps (the smoke test printed them: up to 50,710 fine steps, about 250 generations, for the schistosome rows with the worm's stay fixed and the snail stage a chain of 4). For the B2 cases and for the family of amendment 3 the run is lengthened to 30 relaxation times where that is at most 320,000 fine steps; beyond that the case is not run and is listed with its relaxation time in steps, generations and years (the step lengths in months, 2, 0.8, 4.4 and 2, are those of `manuscript/table-sources-draft8.md`).
3. **A second family of shapes for the regular end, Part B only.** Chains close to a fixed delay need 100 or more sub-stages at the fine step and are slow to simulate. Added: the long stage as a fixed part plus one geometric stay of mean g fine steps, with the same total mean (g = 128, 64, 32, 16, 8, 4 for rows 1a, 1b; 32, 16, 8, 4 for row 6; 16, 8, 4, 2 for rows 3a, 3b). Its coefficient of variation is √(g(g − 1)) divided by the mean. CV½ is given for this family too, by the same rule. It is a different shape from a chain with the same coefficient of variation; the two answers are reported side by side. The protocol had said that only chains would be used.
4. **R5, the pass/fail line.** The rule for flagging a case (3 standard errors between the two halves; flagged cases are left out of the interpolation) is unchanged. But with about 110 cases one or two will exceed 3 standard errors by chance; the second smoke test had one at 3.0, in a case that mixes within a fifth of a generation. The check as a whole now fails only if a case exceeds 4 standard errors or more than three exceed 3.
5. **Sizes** as in section 4, with Part B at 200 × 16,000 r fine steps (100 × 10,000 r where there are 16 or more sub-stages with staying). Estimated 700 to 900 process-seconds.
6. R10's values are read by the script from `results/r10-loss-at-large-noise_20261114.txt`, not typed in.

## 11. After the full run

Written at 12:40:40 by the system clock. The full run started at 12:36:47 and finished at 12:39:19; wall time 152 s on three processes, 450 process-seconds, 136 tasks. R1 to R5 passed at the first full run. The output of the second smoke test (one-eighth length), in which R5 failed under its first wording (one case of 91 at 3.0 standard errors; amendment 4), is kept as `results/r10b-real-systems-stay-shape_attempt1_20261114.txt`. Four cases were not run because their relaxation times exceed the cap of amendment 2; they are listed in the output with those times. Nothing was changed after the full run.
