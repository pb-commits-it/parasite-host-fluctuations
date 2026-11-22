# Protocol R1e — a "parking" stage: time away from the fluctuating host, at a fixed R0

Written 2026-11-14 about 03:55 by tertiary, before `checks/r1e_parking_stage.py` was run on any case below. The predictions in the table were computed from the closed forms first (`engine/theory.py`), with no simulation.

## Why

Dobson's sentence (2020, p. 64) has a specific mechanism in it: a parasite such as *Plasmodium* uses its long-lived host "as a parking lot" while the abundance of the other host is low, and this is what is said to carry it through the boom and bust of that host. Runs so far added hosts that all fluctuate (R1d) or stages at fixed rates per step (R1c). Neither isolates the parking lot: one fluctuating host, and a second stage that does not depend on a fluctuating host, lengthened at a fixed R0.

The theory says (i) the variance of growth per generation, σ²(1 + 2θ)/𝒯, falls in proportion to the generation time, whatever stage the time is spent in; (ii) the loss of growth per generation, Γ, stays in the band between σ²/(2𝒯) and σ²(1 + 2θ)/(2𝒯), and with memory it first RISES when a parking stage is added to a one-stage cycle (the one-stage cycle sits on the lower edge of its band); (iii) in the diffusion approximation the probability of ever falling by x in log size is exp(−E x) with E = 2(ln R0 − Γ)/(𝒯V), and E rises nearly in proportion to 𝒯. So the prediction is that parking lowers the risk of extinction through the variance, even where it raises Γ.

## Model

Two stages in a ring. Stage 1: lineage staying share 0.625 (stay 8/3 steps); its transfer follows a host with log standard deviation σ and autocorrelation φ. Stage 2, the parking stage: stay τ₂, staying share 1 − 1/τ₂; its transfer is constant. "No parking" is the one-stage ring with stay 8/3. Rates are s_k = m_k λ0, g_k = (1 − m_k) λ0, with λ0 solved so that R0 = Π g_k/(1 − s_k) has the stated value; the lineage stays are then exactly 8/3 and τ₂.

## Hypotheses and predictions

| | Hosts | Parking stay | Γ | 𝒯V | E | Diffusion formula, 1,500 generations |
|---|---|---|---|---|---|---|
| K1 | no memory, σ = 0.25, ln R0 = 0.01572 | none | 0.01172 | 0.02344 | 0.34 | 0.41 |
| | | 4/3 | (printed by the script) | | | |
| | | 8/3 | 0.00586 | 0.01172 | 1.68 | 0.02 |
| | | 8 | 0.00293 | 0.00586 | 4.37 | 0.00 |
| K2 | φ = 0.9, σ = 0.12, ln R0 = 0.03 | none | 0.00270 | 0.1026 | 0.53 | 0.29 |
| | | 4/3 | (printed by the script) | | | |
| | | 8/3 | 0.00449 | 0.0513 | 0.99 | 0.10 |
| | | 8 | 0.00288 | 0.0257 | 2.11 | 0.01 |

Both are checks of a prediction. Run R1c found the diffusion formula to overstate extinction under strong memory, so its values are a guide and not a criterion.

## What is run and measured

For each of the 8 cases 2,000 replicates, each starting 10 times above the extinction threshold and running for 1,500 generations (1,500 𝒯 steps) after 1,000 steps of burn-in. Measured: the fraction that fall below the threshold, with its binomial standard error; the mean and variance of growth per generation. The loop is that of `extinction` in `checks/r1c_large_noise.py`, generalised to unequal stages and to transfers that do not all fluctuate; it is first run on the one-stage cases of R1d and must reproduce them (0.410 ± 0.011 and 0.221 ± 0.009) within 3 standard errors of the difference. Seeds are fixed in the script.

## Passes if

- K0: the two one-stage cases reproduce run R1d as stated.
- K1 and K2, each: the simulated probability of extinction falls at each step as the parking stay goes none → 4/3 → 8/3 → 8, each fall by more than 2 standard errors of the difference, and is below 0.03 at a parking stay of 8.
- K3: in case K2 the simulated growth per generation at a parking stay of 8/3 is lower than with no parking (Γ is higher), by more than 2 standard errors, while the extinction probability is lower.

## Budget and stop

Ten minutes of computing. One short report, `reports/R1e-parking-stage.md`. Then stop.
