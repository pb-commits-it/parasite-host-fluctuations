# Protocol R1f — can a steady stage raise the risk? A test near the threshold

Written 2026-11-14 at 05:05 by tertiary, before `checks/r1f_parking_near_threshold.py` was run on any case below. The table of predictions was written into this file by the script that computed them (the closed forms of `checks/parking_stage_formulas.py` and the diffusion formula `lande_orzack` of `checks/r1c_large_noise.py`); none was typed by hand.

## Why

Run R1e found that time in a steady host lowered the probability of falling ten-fold in all eight cases tried. The third referee of the manuscript pointed out that the diffusion approximation does not say this must always happen: with host memory, adding a steady stage can raise the loss of growth Γ by more than it divides the variance, and near the threshold the index E = 2(ln R0 − Γ)/(𝒯V) then falls. That is a statement about an approximation. Whether the simulated process does the same is an open question, and it bears directly on how far the "parking lot" can be relied on.

## Model

As run R1e: one fluctuating host governs the transfer out of a stage with lineage stay a = 10 steps; a second stage with stay b has a constant transfer ("none" is the one-stage cycle). σ = 0.15. R0 is the same along each row; rates are set as in R1e so that the lineage stays are exactly a and b.

## Hypotheses and predictions

Γ, drift (ln R0 − Γ) and 𝒯V per generation, from the second-order closed forms; E the index; "ever" is exp(−E ln 10); the last column is the diffusion formula for 6,000 generations.

| Case | φ | ln R0 | Steady stay | 𝒯 | Γ | Drift | 𝒯V | E | Ever | Within 6,000 generations |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 | 0.9 | 0.00450 | none | 10 | 0.00112 | +0.00337 | 0.04275 | 0.158 | 0.695 | 0.684 |
| N1 | 0.9 | 0.00450 | 10 | 20 | 0.00418 | +0.00032 | 0.02138 | 0.030 | 0.933 | 0.809 |
| N1 | 0.9 | 0.00450 | 40 | 50 | 0.00213 | +0.00237 | 0.00855 | 0.554 | 0.279 | 0.276 |
| N1 | 0.9 | 0.00450 | 160 | 170 | 0.00068 | +0.00382 | 0.00251 | 3.042 | 0.001 | 0.001 |
| N2 | 0.0 | 0.00135 | none | 10 | 0.00112 | +0.00022 | 0.00225 | 0.200 | 0.631 | 0.411 |
| N2 | 0.0 | 0.00135 | 10 | 20 | 0.00056 | +0.00079 | 0.00112 | 1.400 | 0.040 | 0.036 |

- **N1 (open question).** Host with memory, ln R0 = 0.2σ². The index says the risk rises when a steady stage of stay 10 is added, then falls for stays of 40 and 160. For a stay of 160 the fall is certain by the sufficient condition (160 × ln R0 = 0.72 > θσ² = 0.2025); for a stay of 10 it is the case in doubt. Run R1c found the diffusion formula to overstate the risk with long memory, and at σ = 0.15 the neglected fourth-order term of Γ may be of the size of the small drift at a stay of 10, so the outcome is not known in advance.
- **N2 (check).** Host without memory, ln R0 = 0.06σ². The risk falls when a steady stage of stay 10 is added (E rises from 0.20 to 1.40).

## What is run and measured

Each population starts 10 times above a fixed level and runs for 6,000 generations after 1,000 steps of burn-in; 2,000 replicates per case. Measured: the fraction that fall below the level, with its binomial standard error; the mean growth and the variance per generation. The loop is `extinction` of `checks/r1e_parking_stage.py`.

## Passes if

- C1 (all six cases): the simulated growth per generation is within the larger of 3.5 standard errors and 10% of Γ of the predicted drift.
- C2 (N1): the simulated probability at a steady stay of 160 is below that with no steady stage, by more than 2 standard errors of the difference.
- C3 (N2): the simulated probability at a steady stay of 10 is below that with no steady stage, by more than 2 standard errors of the difference.
- The open question is reported, with no pass criterion: is the simulated probability at a steady stay of 10 in case N1 above or below the one with no steady stage, and by how many standard errors?

## Budget and stop

Fifteen minutes of computing. One short report, `reports/R1f-parking-near-threshold.md`. Then stop.
