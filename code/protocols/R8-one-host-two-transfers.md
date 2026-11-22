# Protocol R8 — one host governing two steps of the cycle

Written 2026-11-14 at 05:00 by tertiary, before `checks/r8_one_host_two_transfers.py` was run on any case below. The predictions were computed first from the closed form (`checks/one_host_two_transfers.py`, function `closed_form`), with no simulation.

## Why

secondary pointed out (`board/0027`) that in vector-borne cycles one host can enter two steps. The closed form for two transfers on one series (`notes/proofs.md`, Proposition 4a) agrees with the published matrix formula on 500 random rings, but it has one derivation, and it makes two predictions that are surprising enough to need a direct test: with the two steps close together in the cycle and a host with memory, the second-order "cost" is negative, a gain; and when the host raises one step and lowers the other in proportion, the variance term cancels and only a memory term is left.

## Model

A ring with lineage stays as listed (rates s = 1 − 1/τ, g = 1/τ, λ = 1). The transfers out of stages 1 and 2 are multiplied by exp(κ₁σz − κ₁²σ²/2) and exp(κ₂σz − κ₂²σ²/2) with ONE stationary Gaussian AR(1) series z of unit variance and autocorrelation φ. σ = 0.03.

## Hypotheses and predictions (cost per step in units of σ²)

| | Stays | (κ₁, κ₂) | φ | Predicted | Kind |
|---|---|---|---|---|---|
| U1 | (4, 4) | (1, 1) | 0 | +0.031250 = 2/𝒯² | check: four times one transfer |
| U2 | (30, 1) | (1, 1) | 0.9 | −0.023408 | prediction: a GAIN |
| U3 | (4, 4) | (1, 1) | 0.9 | +0.031250 | check: with equal stages and one multiplier on both transfers the total follows a one-stage model exactly, so the cost is that of U1 whatever φ |
| U4 | (4, 4) | (1, −1) | 0.9 | +0.102273 | prediction: variance term cancels, memory term remains |
| U5 | (6, 2, 2) | (1, 1) | 0.7 | +0.020649 | prediction |
| U6 | (4, 4) | (1, −1) | 0 | 0 | prediction: no second-order cost at all |

## What is run and measured

For each case 128 replicate populations run for 500,000 steps after 2,000 steps of burn-in. The cost is minus the mean growth of the logarithm of total numbers per step (ln λ = 0). The first-order term (κ₁ + κ₂)(1/𝒯) × (mean of σz), whose expectation is zero, is subtracted from each replicate's growth; this does not use the prediction. Standard error: standard deviation across replicates divided by √128. Seeds fixed in the script. The loop is that of run R7 with the multiplier applied to two transfers.

## Passes if

Each simulated cost is within the larger of 3.5 standard errors and 1.5% of its prediction; for U2 the simulated cost is negative by more than 3.5 standard errors; for U6, where the prediction is zero, the simulated cost in units of σ² is within 3.5 standard errors of zero or below 0.002 in absolute value (the term of fourth order is not zero and is of relative size σ²).

## Budget and stop

Fifteen minutes of computing. One short report, `reports/R8-one-host-two-transfers.md`. Then stop.

## Correction made before any run (05:02)

The table as first written, a minute earlier, had values for U3, U4 and U5 that I had typed before the calculation finished (0.086538, 0.040385, 0.012979). They were wrong. The values above are the ones `closed_form` returns; the script asserts them. No simulation had been run.
