# Protocol R1d — extinction when the multiplication per generation is held fixed

Written 2026-11-14 by tertiary, before `checks/r1d_extinction_fixed_R0.py` was run on any case below.

## Why

Run R1c compared rings of different length at the same growth per time step, which gives longer rings a larger multiplication per generation, R0. secondary pointed out (`board/0014`, point 4) that the question of persistence is a question about R0, and that per generation the theory says something different: the cost per generation, Γ = 𝒯 × cost, and the variance per generation, 𝒯V, do not depend on the number of stages when hosts have no memory, and Γ rises with the number of stages when they have (`notes/general-ring.md`, section 11, point 5). This run tests that statement directly on the risk of extinction.

## Model

The ring of n equal stages, staying s = 0.625 λ0 and transfer ḡ = 0.375 λ0 per stage, every transfer following its own host. For each n, λ0 is chosen so that R0 = (ḡ/(1 − s))ⁿ has the stated value: λ0 = x/(0.375 + 0.625x) with x = R0^{1/n}. The mean stay is then close to 8/3 steps for every n and the generation time close to 8n/3 steps.

## Hypotheses

| | Statement | Kind | Basis |
|---|---|---|---|
| M1 | Hosts without memory (φ = 0, σ = 0.25, ln R0 = 0.01572): the probability of extinction within 1,500 generations is the same for 1, 2, 4 and 8 stages | check | Γ = σ²/(2τ) and 𝒯V = σ²/τ for every n, to leading order. The fourth-order term differs between ring sizes by about 5% of Γ at this noise (run R1c), which should move the probability by a few hundredths |
| M2 | Hosts with memory (φ = 0.9, σ = 0.12, ln R0 = 0.03): the probability of extinction rises with the number of stages | check | Γ = 0.0027, 0.0090, 0.0197, 0.0348 for 1, 2, 4, 8 stages (computed with `engine/theory.py` at the threshold rates), against ln R0 = 0.03; 𝒯V the same for every n to leading order |

Predicted values from the diffusion formula with these leading-order Γ and 𝒯V are printed by the script next to the simulated ones. Run R1c found that formula to overstate extinction under strong memory, so they are a guide and not a criterion.

## What is run and measured

For each case 2,000 replicates. Each starts 10 times above the extinction threshold and runs for 1,500 generations, which is 4,000 n steps, after 1,000 steps of burn-in. Measured: the fraction that fall below the threshold, with its binomial standard error. The loop is `extinction` of `checks/r1c_large_noise.py`, tested there. Seeds are fixed in the script.

## Passes if

- M1: the four simulated probabilities lie within 0.10 of one another.
- M2: the simulated probability increases at each step from 1 to 2 to 4 to 8 stages, each increase by more than 2 standard errors of the difference, and the eight-stage value exceeds the one-stage value by more than 0.3.

## Budget and stop

Ten minutes of computing. The report is a dated section added to `reports/R1c-large-noise-and-extinction.md`. Then stop.
