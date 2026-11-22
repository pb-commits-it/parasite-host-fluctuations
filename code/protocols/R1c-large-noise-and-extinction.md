# Protocol R1c — large fluctuations, and the risk of extinction

Written 2026-11-14 by tertiary, before `checks/r1c_large_noise.py` was run on any case below. Work package 1.4 of the plan. Simulator: `engine/ring.py` and `engine/environments.py` (verified in R0 and R0b). Formulas: `engine/theory.py`; the fourth-order coefficient is `studies/h4/h4.py`, which agrees with secondary's independent one.

## Why

Everything proved so far is about small fluctuations. The abundance of real hosts varies with a standard deviation of the logarithm, σ, of order one. Three things need to be known before the small-noise results are used to say anything about real parasites: how large σ can be before the closed form is off; whether the pattern in the number of stages survives beyond that; and whether the mean and variance of growth from the theory predict the risk of extinction.

## Model for this run

A ring of n equal stages, staying s and mean transfer ḡ in every stage, every transfer following its own host with the same σ and φ. Two settings of the rates:

- **Part A and B:** s = 2/3, ḡ = 0.4, as in all earlier runs on the equal ring (λ0 = 1.0667, mean stay 2.67 steps).
- **Part C:** s = 0.63125, ḡ = 0.37875, the same staying share with λ0 = 1.01, so that the population grows by 1% per step without fluctuation and extinction is neither certain nor negligible.

## Questions and hypotheses

| | Statement | Kind | Basis |
|---|---|---|---|
| L1 | For each n and φ, the simulated cost divided by the second-order closed form, at σ = 0.1, 0.2, 0.4, 0.8 | open | — |
| L2 | Adding the exact fourth-order term brings the prediction closer to simulation wherever the fourth-order correction is itself below 20% of the second-order term | check | two independent derivations of the coefficient |
| L3 | With one stage the simulated cost equals the exact one-dimensional integral at every σ and φ | check | exact |
| L4 | At σ = 0.4 and 0.8: with φ = 0 the cost still falls from one stage onward; with φ = 0.9 it still has a maximum above one stage | open | small-noise result; nothing is known at this noise |
| L5 | The variance of total log growth per step, at σ = 0.1, equals σ² n (1 + 2θ)/𝒯² within 5% | check | leading-order formula (`notes/general-ring.md`, section 6) |
| L6 | The same at σ = 0.4 and 0.8 | open | — |
| L7 | The probability that the total falls below one hundredth of its starting value within 2,000 steps, against the diffusion formula of Lande & Orzack (1988, eq. 11) fed with the mean and the long-run variance of log growth measured in the same simulation | open | Lande & Orzack tested their formula for uncorrelated years only |
| L8 | The same formula fed with the theory's mean (second-order closed form) and variance (leading order) | open | — |

## Grid

- Parts A and B: n = 1, 2, 4, 8, 16; φ = 0, 0.5, 0.9; σ = 0.1, 0.2, 0.4, 0.8. Sixty runs. Reasons: n up to 16 covers every known parasite cycle several times over and reaches the range where the fourth-order term is known to dominate; φ = 0.9 is a host whose state persists for about ten steps; σ = 0.8 is a host whose abundance commonly varies five-fold either way.
- Part C: n = 1, 2, 4, 8; φ = 0 and 0.9; σ = 0.2 and 0.5. Sixteen runs.

## What is run and measured

- Parts A and B: 400 replicates of 60,000 steps after 3,000 discarded, with the control variate. Measured: the cost with its standard error over replicates; and the long-run variance of log growth per step, estimated as the variance of the total log growth over batches of 6,000 steps (ten batches per replicate, 4,000 batch totals in all), divided by 6,000. A batch is 100 or more times longer than the memory of the environment or of the composition, so the bias of this estimate is of the order of 1%. Its standard error is taken from the spread of the squared deviations.
- Part C: 4,000 replicates of 2,000 steps after 1,000 steps of burn-in in which the composition settles but the total is not tracked. Each replicate starts at log size 0 and is counted extinct if its log total size falls below −ln 100 at any step. Measured: the fraction extinct, with its binomial standard error; the mean log growth per step and the variance per step of the total log growth over the 2,000 steps.
- Seeds are fixed in the script.

## Passes if

- L2: in every run where |H_n σ²/C_n| < 0.2, the absolute departure of simulation from C_n σ² + H_n σ⁴ is smaller than its departure from C_n σ², or both are within 3.5 standard errors.
- L3: all twelve one-stage runs within 3.5 standard errors of the exact integral.
- L5: all fifteen runs at σ = 0.1 within the larger of 5% and 3.5 standard errors.
- L1, L4, L6, L7 and L8 are open: the outcome is reported as it comes, with no criterion.

## Budget and stop

Thirty minutes of computing on one machine. One report, `reports/R1c-large-noise-and-extinction.md`, then stop. Departures are listed in the report.
