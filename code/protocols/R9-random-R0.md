# Protocol R9 — the loss of ln R0 in a random environment is Γ at the ordinary stays

Written 2026-11-14 at 06:42 (clock) by tertiary, before `checks/r9_random_r0.py` existed or was run on any case below. The predictions were printed by `checks/random_r0.py` (no simulation; output `results/random-r0_20261114.txt`) and copied here.

## Why

`notes/random-R0.md` derives that, with Bacaër & Khaladi's (2013) definition of R0 in a random environment (R0* is the R for which the model with births divided by R has zero growth), ln R0 − ln R0* = Γ₀ to second order, where Γ₀ is our closed form evaluated with the stays 1/(1 − s_k), whatever the value of R0. For one stage this was checked exactly (checks B1 to B3). For several stages with memory the growth rate is not available in closed form, so the statement is tested by simulation. primary is deriving the statement blind at the same time; I have not seen its result.

## Model

The ring of the manuscript with staying rates s_k (constant) and transfers g_k h_k(t), h_k = exp(z_k − σ_k²/2), z_k Gaussian AR(1) with variance σ_k² and autocorrelation φ_k, independent across k. For a model with reproduction number R0 at mean rates, "dividing the births by R" gives a ring with the same s_k and Π g_k/(1 − s_k) = R0/R; only that product matters, and it is shared equally among the transfers.

## Hypotheses and predictions

| | Staying rates | σ per transfer | φ per transfer | R0 as given | Γ₀ | Γ at the given model's own stays | Kind |
|---|---|---|---|---|---|---|---|
| V1 | (0.5, 0.8) | (0.25, 0) | (0.8, –) | 4 | 0.013863 | 0.019231 | prediction |
| V2 | (0.5, 0.8) | (0.25, 0.20) | (0.8, 0.5) | 0.3 | 0.020081 | 0.011809 | prediction |
| V3 | (0.3, 0.6, 0.9) | (0, 0.30, 0) | (–, 0.9, –) | 4 | 0.019046 | 0.027288 | prediction |
| V4 | (0.5, 0.8) | (0.25, 0.25) | (0, 0) | 4 | 0.008929 | 0.014793 | check of the engine: no memory, Γ₀ = Σσ²/(2𝒯₀) |

- **H1 (prediction).** Run A: the model with Π g_k/(1 − s_k) = exp(Γ₀), that is, the given model with its births divided by R0 exp(−Γ₀), has zero long-run growth up to a term of order σ⁴.
- **H2 (prediction).** Run B: the model with Π g_k/(1 − s_k) = exp(Γ at the given model's own stays) does not. Its growth per step is (that Γ − Γ₀)/𝒯₀ to leading order: +0.000767, −0.001182, +0.000592 and +0.000838 for V1 to V4 (𝒯₀ = 7, 7, 13.93, 7).

The ranges: R0 = 4 and 0.3 are far from one on either side, so that the stays of the given model differ from 1/(1 − s_k) by enough to tell the two candidates apart; σ = 0.2 to 0.3 keeps the neglected fourth-order term at a few per cent of Γ₀ (it was 3 to 4% at this noise in run R1c) while making Γ₀ large beside the simulation error.

## What is run and measured

For each case and each of runs A and B: 400 replicate populations, 200,000 steps after 2,000 of burn-in, the total renormalised each step. Measured: the mean growth of the logarithm of the total per step. The first-order term Σ_k e_k × (mean of z_k), whose expectation is zero (e_k the elasticity of transfer k in the simulated model), is subtracted from each replicate's growth; this does not use the prediction. Standard error: standard deviation across replicates divided by √400. Results are reported multiplied by 𝒯₀, that is, in units of ln R0.

## Passes if

- **C1 (run A, all four cases):** |growth × 𝒯₀| is at most the larger of 3.5 standard errors and 6% of Γ₀.
- **C2 (run B, all four cases):** growth has the predicted sign, is more than 5 standard errors from zero, and is within 25% of the predicted value. (The prediction is of leading order, and the slope of growth against ln R is 1/𝒯 of the simulated model, which is within a few per cent of 1/𝒯₀.)

## Budget and stop

Eight runs, about ten minutes. One attempt. If a criterion fails, report it and stop; no parameter is changed after seeing a result.
