# Protocol R7 — a host that cycles: does the parasite's generation time resonate with it?

Written 2026-11-14 about 04:10 by tertiary, before `checks/r7_cyclic_hosts.py` was run on any case below. The predictions in the table were computed first from the closed form, with no simulation.

## Why

"Boom and bust" in real hosts is usually a cycle: voles, grouse, mast-driven mice, seasonal snails and mosquitoes. All runs so far used hosts with geometric memory (AR(1)), which have no period. The lineage form of the theory covers any autocorrelation. For a host whose log abundance is a stationary quasi-cycle (an AR(2) series with complex roots re^{±iω}, period P = 2π/ω), its autocorrelation is ρ(ℓ) = Re[B z^ℓ], z = re^{iω}, and the same renewal argument as for geometric memory gives, for one fluctuating transfer of a ring,

    cost/σ² = S₀/(2𝒯²) − (1/𝒯) Re[ B R(z)/(1 − R(z)) ],        S₀ = 1 + 2 Re[B z/(1 − z)],

with R(z) = Π_k (1 − m_k)z/(1 − m_k z) the generating function of the time round the ring at the complex number z. I checked this against the direct sum over lags on 300 random rings (agreement to 10⁻¹³). It has not been derived by anyone else.

What it predicts is a resonance. If the generation time is close to the host's period and not too variable, a line of descent returns to the host at the same phase of its cycle, returns are in excess where the autocorrelation is positive, and the cost falls, to the point of becoming a gain. If the generation time is half the period it returns at the opposite phase and the cost is several times that of an uncorrelated host. A cycle of many stages has a less variable generation time than a cycle of one stage with the same mean, so here, for the first time in this programme, the number of stages matters by itself, through the regularity of the generation.

## Model

A ring of n equal stages with lineage stay τ each (rates s = 1 − 1/τ, g = 1/τ, so λ = 1). The transfer out of stage 1 is multiplied by exp(σz − σ²/2), z a stationary Gaussian AR(2) series with unit variance, roots 0.9e^{±2πi/12} (period 12 steps, damping 0.9 per step). All other transfers are constant. σ = 0.03.

## Hypotheses and predictions

Γ = 𝒯 × cost in units of σ²; the value for an uncorrelated host of the same variance is 1/(2𝒯).

| | Stages | Stay | 𝒯 | Predicted Γ/σ² | Uncorrelated host | Kind |
|---|---|---|---|---|---|---|
| Q0 | 4 | 3 | 12 | (AR(1) host, φ = 0.5: the validated closed form `theory.cost_transfer`) | | check of the new simulation code |
| Q1 | 1 | 12 | 12 | +0.0417 | 0.0417 | check: one stage ignores any memory |
| Q2 | 4 | 3 | 12 | +0.1047 | 0.0417 | prediction: generation equal to the period but variable (CV² = 0.17): a cost |
| Q3 | 8 | 1.5 | 12 | −0.0764 | 0.0417 | prediction: generation equal to the period and regular (CV² = 0.04): a GAIN |
| Q4 | 2 | 3 | 6 | +0.2561 | 0.0833 | prediction: half the period: about three times the uncorrelated cost |
| Q5 | 4 | 1.5 | 6 | +0.3903 | 0.0833 | prediction: half the period, regular: nearly five times |
| Q6 | 4 | 6 | 24 | +0.0259 | 0.0208 | prediction: twice the period: close to uncorrelated |

## What is run and measured

For each case 128 replicate populations run for 500,000 steps after 2,000 steps of burn-in. The cost is ln λ minus the mean growth of the logarithm of total numbers per step. To reduce noise, the first-order term (1/𝒯) × (mean of σz), whose expectation is zero, is subtracted from each replicate's growth; this does not use the prediction. The standard error is the standard deviation across replicates divided by √128. Seeds are fixed in the script.

## Passes if

Each simulated cost is within the larger of 3.5 standard errors and 1.5% of its prediction; and, for Q3, the simulated cost is negative by more than 3.5 standard errors.

## Budget and stop

Fifteen minutes of computing. One short report, `reports/R7-cyclic-hosts.md`. Then stop.
