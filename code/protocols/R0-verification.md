# Protocol R0 — verification run for the ring simulator

Written 2026-11-13 by tertiary, before `engine/ring.py` was run on any case below. Plan package 1.0, rung R0.

## Question

Does the simulator that rungs R1 to R4 will use reproduce results that are known without it? Nothing new is learned here.

## The model the engine must implement

n stages in a ring, indices mod n. Per time step, with S_k(t) the survival in stage k and G_k(t) the number of stage-(k+1) parasites produced per stage-k parasite:

    x_{k+1}(t+1) = S_{k+1}(t) x_{k+1}(t) + G_k(t) x_k(t)

The long-run growth rate is the time average of the log of the one-step growth of the total. "Cost" is ln λ0 minus that rate, where λ0 is the dominant eigenvalue of the matrix of mean rates.

## Checks, each with its tolerance

All are checks. None is open. s.e. means the standard error across independent replicates.

| | Known result | Basis | Cases | Passes if |
|---|---|---|---|---|
| V1 | Constant rates: growth per step is ln ρ(Ā), ρ the dominant eigenvalue from `numpy.linalg.eigvals` | definition | equal stages, s = 2/3, ḡ = 0.4, n = 1, 2, 4, 8, 16, 32; and 20 rings with random unequal s_k in (0.1, 0.95), g_k in (0.05, 1), n drawn from 2 to 8 | last-step growth within 1e-9 after 20,000 steps, every case |
| V2 | One stage: rate = E ln(s + ḡ h), h lognormal with mean one, by Gauss–Hermite quadrature (200 nodes, cross-checked with `scipy.integrate.quad`). Autocorrelation cannot matter | exact; a scalar product of stationary factors | s = 2/3, ḡ = 0.4; σ = 0.05, 0.15, 0.4, 0.8; φ = 0, 0.9 | every case within 3.5 s.e. of the exact value, and s.e. no more than 1% of the exact cost |
| V3 | One stage, small fluctuations: cost → ½ e² Var(h), e = ḡ/λ0, with a relative gap of order σ². My own expansion gives a gap of −2e(1 − ¾e)σ², which is −0.539σ² here | scalar case of the small-noise approximation (Tuljapurkar 1982, recalled, and re-derived by Taylor expansion) | the exact integral of V2 at σ = 0.2, 0.1, 0.05; no simulation | gap shrinks by a factor between 3.5 and 4.5 each time σ halves; gap/σ² at σ = 0.05 within 15% of −0.539 |
| V4 | No staying (s = 0), all parasites start in stage 1: mean log growth per step is exactly ln ḡ − σ²/2 at any horizon; the variance of total log growth over T steps is σ²[T + 2 Σ_{j≥1, jn<T} (T − jn) φ^{jn}] | exact; the population is one cohort that meets host k every n steps | ḡ = 1.1, σ = 0.3; n = 1, 2, 4, 8; φ = 0, 0.9; 4,000 replicates of 20,000 steps, no burn-in | mean within 3.5 s.e.; variance within 8% (3.5 times the sampling error of a variance from 4,000 replicates) |
| V5 | Rates that repeat with period p: rate = (1/p) ln ρ(A_p ⋯ A_1), matrices built entry by entry from the model above | exact; periodic matrix products (Caswell 2001, recalled) | 10 random rings, n from 2 to 6, p in {2, 3, 5}; survival and production both vary | mean growth over the last 10 periods within 1e-9 after 20,000 steps, every case |
| V6 | Uncorrelated years, several stages: cost ≈ Var(vᵀ A_t u) / (2 λ0²), with u and v the right and left eigenvectors of Ā scaled so vᵀu = 1 | published small-noise approximation (Tuljapurkar 1982, recalled) | σ = 0.05, φ = 0; equal stages n = 2, 4, 8; two rings with unequal stages (n = 3 and n = 5, fixed in the script) | simulated cost within the larger of 3.5 s.e. and 2% of the approximation, every case |
| V7 | Two-stage result of Jonsson (2021) | published | only once the paper has been read in full | not run if the full text is not in hand; recorded as outstanding |
| V8 | The checks can fail: (m1) hosts without the mean correction, h = exp(z), must fail V2 at σ = 0.4; (m2) one host series shared by all stages must fail the variance part of V4 at n = 4, φ = 0.9 | — | two deliberately broken copies of the environment | both broken copies fail |

Simulation settings for V2 and V6: 400 replicates, 120,000 steps, 2,000 discarded, and a control variate (the elasticity-weighted sum of h_k − 1, whose mean is exactly zero) to remove first-order sampling noise. Seeds are fixed in the script.

## What counts as passing

V1 to V6 and V8 all meet their criteria. One failure fails the run: the engine is then not used for any new result, the cause is found, and the whole run is repeated and reported as a second attempt. Every attempt is reported.

## Budget and stop

Ten minutes of computing on one machine. One report, `reports/R0-verification.md`, then stop. Departures from this protocol are listed in the report.
