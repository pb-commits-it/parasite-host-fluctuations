# Protocol R1a — why simulation and the closed form part company in long rings

Written 2026-11-13 by tertiary, after R0 passed and before any case below was run. Rung R1, first part.

## Question

At σ = 0.15 and φ = 0.9 the simulated cost of fluctuation exceeds the closed form by 0.2% at 8 stages, 2% at 16 and 14.5% at 32 (`results/ring_cost_formula_20261113.txt`). With uncorrelated years it exceeds it by 4.8% at 32 stages. Is the closed form wrong for long rings, or is it right to second order and the difference a higher-order term that grows with the number of stages?

## Why it matters

The closed form is the one result confirmed by separate code. If it fails for long rings, the statement that cost falls as 1/n beyond the peak is unsupported.

## Hypotheses

Write ratio = simulated cost / closed form, and excess = ratio − 1.

| | Hypothesis | Basis | Kind | Prediction |
|---|---|---|---|---|
| H1 | The excess is a higher-order term in σ | The closed form is second order. The next terms are of order σ⁴, so the excess should be proportional to σ² | open | At fixed n and φ, halving σ² halves the excess. Passes if excess/σ² at the four noise levels agrees within 25% of its mean, or within 2.5 s.e. where the excess is small, and the excess extrapolated to σ = 0 is within 2 s.e. of zero |
| H2 | The higher-order term grows with n | In a ring of n stages the second eigenvalue of the mean matrix approaches the first as n grows (modulus ratio 0.9955 at n = 32 here), so stage structure forgets a disturbance slowly and fluctuations compound | open | excess/σ² rises with n at fixed φ |
| H3 | The simulation is not biased by run length or burn-in | 120,000 steps is hundreds of mixing times | check | At n = 32, φ = 0.9, σ = 0.15, a run four times longer with four times the burn-in agrees with the shorter one within 3 s.e. |
| H4 | With little staying the same happens at small n | At s = 0 the ring is cyclic, cohorts never mix, and the exact cost is σ²/2 whatever n and φ are, while the closed form gives less for n > 1. So for small s the closed form should need smaller σ | open | At s = 0.05, n = 4, φ = 0: the excess is large at σ = 0.15 and falls in proportion to σ² |

If H1 fails, with the excess not vanishing as σ → 0, the closed form is wrong for long rings. Claim a is then withdrawn until the derivation is repaired, and the plan's week-3 gate is not met.

## Grid, with reasons

- Stay probability s = 2/3 and production ḡ = 0.4, as in every earlier check, so results are comparable.
- σ = 0.15, 0.106, 0.075, 0.053: each step halves σ². The top value is the one at which the gap was seen. Below 0.05 the excess expected under H1 falls under one standard error.
- (n, φ) = (16, 0.9), (32, 0.9), (32, 0), (64, 0.9). The first three are where the gap was seen. 64 tests H2 beyond the range already looked at.
- H4: s = 0.05, ḡ = 1.0167 (same λ0 = 1.0667), n = 4, φ = 0, the same four σ. One value of s; this is a probe, not a map.
- H3: n = 32, φ = 0.9, σ = 0.15 at 480,000 steps with 8,000 discarded.

## What is measured

Simulated cost with the control variate, its standard error across replicates, the closed form, the ratio. Replicates: 1,600 for φ = 0.9 at n = 32 and 64 (the excess expected at the smallest σ is about 2%, so the standard error must be well under 1%); 800 elsewhere. 120,000 steps, 2,000 discarded.

## Budget and stop

Sixty minutes of processor time in total, spread over the machine's cores. One report, `reports/R1a-large-ring-gap.md`, then stop. No further cases are added after seeing the results; anything they suggest goes into a new protocol.
