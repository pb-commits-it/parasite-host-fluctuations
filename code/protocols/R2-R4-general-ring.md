# Protocol R2/R4 — the closed forms for unequal stages and fluctuating survival, at finite noise

Written 2026-11-14 by tertiary, before `checks/r2r4_general_ring.py` was run on any case below. Rungs R2 and R4 of the plan. The simulator is `engine/ring.py` with `engine/environments.py`, both verified (R0 and R0b).

## Questions

1. Do the closed forms of `notes/general-ring.md` give the cost of fluctuation in rings with unequal stages, with a different noise level and memory in each stage, and with fluctuating survival, when the noise is small?
2. Do four statements read off those formulas hold in simulation: the position of a fluctuating host does not matter; at a given number of stages and generation time the equal ring costs most; positively autocorrelated survival can turn the cost into a gain; and contributions of different hosts add?

## What is already known, and what is not

The closed forms equal the published matrix formula to rounding (`checks/general_ring_closed_forms.py`), and `notes/error-bound-general.md` argues that the matrix formula is the true cost up to a term of order σ⁴. So each hypothesis below is a **check**: a failure would mean an error in the algebra, in the proof or in the simulator. What is **open** is the size of the fourth-order term for these rings, that is, how small σ must be.

## Hypotheses

| | Statement | Kind | Basis |
|---|---|---|---|
| H1 | Unequal stages, every transfer fluctuating: simulated cost equals the closed form | check | closed form = matrix formula; Theorem G |
| H2 | A different σ and φ for each host: the contributions add | check | the formula is a sum over hosts |
| H3 | One fluctuating host, moved from stage to stage: the cost does not change | check | the formula depends on the host only through σ² and φ |
| H4 | Same number of stages and same generation time: equal stages cost most, and a ring with all but one stage without staying costs least | check | the product in R is largest for equal stays |
| H5 | Fluctuating survival with positive autocorrelation in a long-lived stage: the contribution is negative (a gain) where the formula says so, zero at the predicted boundary, and positive without autocorrelation | check | closed form for survival |
| H6 | Survival and transfer fluctuating together, independently: contributions add | check | independence |
| H7 | Slow environment (φ = 0.98): closed form holds, and is near (1 − CV²)/(2𝒯) per host | check | limit of the closed form |
| H8 | How far the closed form is off at σ = 0.1, 0.2, 0.4 | open | — |

## Cases

Rings marked λ = 1.05 are built from chosen mean stays τ_k as s_k = λ(1 − 1/τ_k), g_k = λ/τ_k, which makes the dominant eigenvalue exactly λ.

| Case | Ring | Noise | Tests |
|---|---|---|---|
| A1 | n = 3, s = (0.3, 0.85, 0.5), g = (0.6, 0.2, 0.9) | all transfers, φ = 0.7 | H1 |
| A2 | n = 6, s = (0.9, 0, 0.4, 0, 0.7, 0.2), g = (0.3, 1.2, 0.5, 0.9, 0.25, 0.8) | all transfers, φ = 0.8 | H1, with two stages without staying |
| A3 | n = 2, s = (0.95, 0.1), g = (0.08, 1.5) | all transfers, φ = 0.9 | H1, very unequal |
| B1 | ring of A1 | transfers with σ_k = σ × (1, 0.5, 2), φ_k = (0, 0.5, 0.9) | H2 |
| C1–C4 | n = 4, s = (0.2, 0.9, 0.5, 0.7), g = (0.9, 0.15, 0.6, 0.4) | one transfer fluctuating, φ = 0.8, at stage 1, 2, 3, 4 in turn | H3 |
| D1 | λ = 1.05, τ = (2, 2, 2, 2) | all transfers, φ = 0.9 | H4 |
| D2 | λ = 1.05, τ = (1.2, 1.6, 2.2, 3.0) | all transfers, φ = 0.9 | H4 |
| D3 | λ = 1.05, τ = (1, 1, 1, 5) | all transfers, φ = 0.9 | H4 |
| E1 | λ = 1.05, τ = (10, 1.25) | survival of stage 1, ψ = 0.8 | H5, gain predicted |
| E2 | ring of E1 | survival of stage 1, ψ = 0 | H5, cost predicted |
| E3 | ring of E1 | survival of stage 1, ψ at the value where the closed form is zero | H5, boundary |
| E4 | ring of A1 | survival of stage 2, ψ = 0.7 | H5, gain predicted |
| E5 | ring of A1 | every transfer (φ = 0.7) and every survival (ψ = 0.5), all independent | H6 |
| F1 | ring of D1 | all transfers, φ = 0.98 | H7 |

Reasons for the ranges: stays from 1 step (no staying) to 10 steps span a transmission stage and a long-lived stage; autocorrelation from 0 to 0.98 spans uncorrelated years to an environment whose memory (about 50 steps) is several generations; rings of 2 to 6 stages cover the range in which real cycles lie and in which the second-order formula is expected to be accurate at these noise levels.

## What is run and measured

Each case at σ = 0.05 and σ = 0.025: 400 replicates of 120,000 steps after 2,000 discarded (5,000 discarded for F1), with the control variate. Measured: the cost, ln λ minus the mean log growth per step, with its standard error over replicates. Seeds are fixed in the script. For H8, cases A1, D1 and E1 are also run at σ = 0.1, 0.2 and 0.4.

## Passes if

- **Each case (H1, H2, H5, H6, H7):** at σ = 0.025 the simulated cost is within the larger of 3.5 standard errors and 1% of the closed form. For E3, where the closed form is zero, within the larger of 3.5 standard errors and 1% of the cost of E2. The result at σ = 0.05 is reported with the ratio of the two departures (four is expected if the next term is of fourth order); it is not a pass criterion.
- **H3:** each of C1 to C4 passes as above, and the largest difference between any two of them at σ = 0.025 is within the larger of 3.5 standard errors of that difference and 1% of the closed form.
- **H4:** D1, D2, D3 pass as above, and simulated D1 > D2 > D3 with each difference more than 3.5 standard errors.
- **H5 signs:** simulated E1 and E4 are negative by more than 3.5 standard errors, and E2 positive by more than 3.5.
- **H7:** F1 passes as above. The comparison with the limit is reported only.

One failure fails the hypothesis it belongs to. The cause is then found before anything is built on it, and the run is reported as it came out.

## Budget and stop

Fifteen minutes of computing on one machine. One report, `reports/R2-R4-general-ring.md`, then stop. Departures are listed in the report.

## Amendment 1, 2026-11-14, written after attempt 1 stopped and before any of the cases below was run

**What happened.** Attempt 1 (`results/R2-R4-general-ring_attempt1_20261114.txt`) ran cases A1 to A3, B1, C1 to C4, D1 to D3, E1 and E2, all of which met their criteria, and then stopped with an error while setting up E3.

**The cause is an error of mine in this protocol, not in the formula or the simulator.** I labelled E1 "gain predicted" without evaluating the closed form for that ring. For stays (10, 1.25) and fluctuating survival of the long stage at ψ = 0.8 the closed form gives a cost of +0.2504 σ², smaller than the +0.3200 σ² at ψ = 0 but positive, at every ψ below one. The long stage holds 80% of the elasticity of that ring, and the condition for a gain in `notes/general-ring.md`, section 5, needs the elasticity to be small. The simulation agreed with the closed form (0.25048 ± 0.00010 at σ = 0.025). E3, the boundary case, does not exist for that ring, which is what stopped the run. Had the run continued, the sign criterion "simulated E1 is negative" would have failed, correctly.

**What is kept.** The results of attempt 1 for A1 to E2 stand as run. E1 is relabelled: it is a check that positively autocorrelated survival lowers the cost without reversing it when the fluctuating stage dominates the cycle.

**What replaces the rest of block E.** Defined now, before running:

| Case | Ring | Noise | Tests |
|---|---|---|---|
| E3a | λ = 1.05, τ = (4, 4, 4) | survival of stage 1, ψ = 0.8 | H5, gain predicted (closed form −0.1473 σ²) |
| E3b | same ring | survival of stage 1, ψ = 0.2168, where the closed form is zero | H5, boundary |
| E3c | same ring | survival of stage 1, ψ = 0 | H5, cost predicted (+0.03125 σ²) |
| E4 | ring of A1 | survival of stage 2, ψ = 0.7 | H5, gain predicted (−0.0327 σ²), unchanged |
| E5 | ring of A1 | every transfer (φ = 0.7) and every survival (ψ = 0.5) | H6, unchanged |
| F1 | ring of D1 | all transfers, φ = 0.98 | H7, unchanged |

Pass criteria as before, with these changes: the sign criterion applies to E3a and E4 (negative by more than 3.5 standard errors) and to E3c and E2 (positive); E3b must lie within the larger of 3.5 standard errors and 1% of the cost of E3c. H8 is run on A1, D1 and E3a.

The remaining cases are run by `checks/r2r4_general_ring_part2.py`, with new seeds.
