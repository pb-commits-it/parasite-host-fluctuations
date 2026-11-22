# Protocol R1b — does the closed form hold in long rings at very small noise?

Written 2026-11-13 by tertiary, after the results of R1a were seen and before any case below was run. Rung R1.

## Question

R1a left one thing open. In long rings with long spells, the excess of simulation over the closed form shrinks with σ², but a straight line through the four noise levels did not reach zero at σ = 0: it left 1.0% at 32 stages and 4.5% at 64. A curve with one more term, excess = cσ² + dσ⁴, fits and does reach zero. That curve was chosen after seeing the data, so it needs a test at noise levels below any used so far.

## Hypothesis and prediction

| | Hypothesis | Basis | Kind | Prediction, fixed here |
|---|---|---|---|---|
| H1b | The closed form is the exact second-order coefficient for long rings; the excess is cσ² + dσ⁴ with no constant | R1a fits: c = 8.52, d = −86 at 32 stages; c = 30.57, d = −363 at 64 stages (φ = 0.9). A third model's independent run gave c = 8.8 at 32 stages | open | n = 32, σ = 0.02: ratio 1.0034. n = 64, σ = 0.012: ratio 1.0044 |

**Passes** if each simulated ratio is within 3 standard errors of its prediction and each standard error is at most 0.25% of the cost. **Fails** if either ratio is more than 3 standard errors above its prediction. A constant of the size the straight line suggested would put the ratios near 1.014 and 1.049.

If it fails, the closed form is not the limit of the simulation in long rings, and claim a is restricted to short rings until the reason is found.

## Grid, with reasons

- s = 2/3, ḡ = 0.4, φ = 0.9, as in R1a.
- σ = 0.02 at 32 stages and σ = 0.012 at 64 stages. Both lie below the R1a range, and each makes the predicted excess (0.3% to 0.4%) small against the constant being tested for.
- 6,400 replicates per case, run as 8 independent batches of 800 with different seeds; 120,000 steps; 5,000 discarded, more than five mixing times of the 64-stage ring.

## Budget and stop

Ninety minutes of processor time across the machine's cores. R1a overran its budget of 60 minutes (it used 98); this budget is set from R1a's measured speed. One section added to `reports/R1a-large-ring-gap.md`, then stop.
