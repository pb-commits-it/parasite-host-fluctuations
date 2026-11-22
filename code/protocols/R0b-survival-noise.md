# Protocol R0b — verification of fluctuating survival and of stage-specific noise

Written 2026-11-14 by tertiary, before `engine/environments.py` was run on any case below. It extends `protocols/R0-verification.md`, whose checks all put the fluctuation in transfer with one σ and one φ for every stage. secondary asked for this before rung R4 (`verification/derivation_check.md`, section 3.1).

## Question

Does the new environment class reproduce results known without it, when survival fluctuates and when each stage has its own noise level and autocorrelation? Nothing new is learned here.

## What is new in the code

`engine/environments.py`, class `FluctuatingRing`: G_k(t) = g_k h_k(t) and S_k(t) = s_k u_k(t), every series an independent lognormal AR(1) with mean one, with a log standard deviation and an autocorrelation given per stage and per kind (zero switches a series off). `engine/ring.py` is not changed, so the R0 run still stands for it.

In this model S_k(t) is a multiplier and can exceed one when u_k is large. That is harmless for the mathematics. It is noted here because a survival probability cannot.

## Checks, each with its tolerance

All are checks. None is open. s.e. is the standard error across independent replicates.

| | Known result | Basis | Cases | Passes if |
|---|---|---|---|---|
| S1 | One stage, survival fluctuating, transfer constant: rate = E ln(s u + g) by Gauss–Hermite quadrature. Autocorrelation cannot matter | exact; a scalar product of stationary factors | s = 2/3, g = 0.4; σ = 0.05, 0.15, 0.4, 0.8; φ = 0, 0.9 | every case within 3.5 s.e. of the exact value, with and without the control variate, and s.e. no more than 1% of the exact cost |
| S2 | One stage, both fluctuating independently: rate = E ln(s u + g h) by a two-dimensional quadrature (80 × 80 nodes) | exact | s = 2/3, g = 0.4; (σ_s, σ_g) = (0.3, 0.2); (φ_s, φ_g) = (0, 0.9) and (0.9, 0.5) | within 3.5 s.e. |
| S3 | One stage, no transfer at all (g = 0): log growth per step is ln s + z − σ²/2, so its mean is ln s − σ²/2 at any horizon and the variance of the total over T steps is σ²[T + 2 Σ_{j=1}^{T−1} (T − j) φ^j] | exact | s = 0.9; σ = 0.3; φ = 0, 0.5, 0.9; 4,000 replicates of 20,000 steps, no burn-in | mean within 3.5 s.e.; variance within 8% |
| S4 | No staying, transfer noise with a different σ_k in each stage, one cohort: mean log growth per step is (1/n) Σ_k (ln g_k − σ_k²/2) over whole turns of the ring | exact, as V4 | n = 3, g = (1.3, 0.8, 1.1), σ = (0.1, 0.3, 0.5), φ = (0, 0.5, 0.9); 4,000 replicates of 20,001 steps (6,667 turns) | mean within 3.5 s.e. |
| S5 | Uncorrelated years, small noise: cost ≈ ½ Σ e_k² Var, summed over the fluctuating rates, e_k the elasticity of λ0 to the rate and Var = exp(σ²) − 1 | published small-noise approximation (Tuljapurkar 1982, recalled), as V6 | σ = 0.05, φ = 0: (a) equal stages n = 2, survival noise in both; (b) equal stages n = 4, survival noise in all; (c) unequal n = 3, survival noise in stage 2 only; (d) unequal n = 5, survival and transfer noise in every stage | simulated cost within the larger of 3.5 s.e. and 2% of the approximation |
| S6 | The checks can fail: (m3) survival noise applied to the stage after the intended one must fail S5(c); (m4) a survival series without the mean correction must fail S1 at σ = 0.4 | — | two deliberately broken copies | both fail |

Simulation settings for S1, S2 and S5: 400 replicates, 120,000 steps, 2,000 discarded, with the control variate (the elasticity-weighted sum of h_k − 1 and u_k − 1, whose mean is exactly zero). Seeds are fixed in the script.

## What counts as passing

S1 to S6 all meet their criteria. One failure fails the run: the environment class is then not used for any new result, the cause is found, and the whole run is repeated and reported as a second attempt.

## Not in this run

V7, the comparison with Jonsson (2021). It needs that paper's own parameter values and is run separately.

## Budget and stop

Fifteen minutes of computing on one machine. One report, appended to `reports/R0-verification.md` as a dated section, then stop. Departures are listed there.
