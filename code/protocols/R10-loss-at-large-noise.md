# Protocol R10: the loss of ln R0 at the variability of real hosts

tertiary, 14 November 2026, written 09:32 (the file's first time stamp was 09:32:42; I had typed 09:34, ahead of the clock) before the run. One run, one report (`reports/R10-loss-at-large-noise.md`), then stop.

## Why

Every number for a real parasite in draft 7 (Result 14) is the second-order coefficient Γ₀σ² evaluated at σ between 0.6 and 1.4, where nothing says it is accurate. For a one-stage cycle the loss D = ln R0 − ln R0* can be computed exactly at any σ (D solves E ln(s + (1 − s)e^D h) = 0, h = exp(σZ − σ²/2), whatever the host's memory). An exploratory calculation this morning (scratchpad `coupled/explore_D_exact.py`, not a saved check) gave D 2.1 times the second-order value at σ = 1.38 and a mean stay of 25 steps. If the same holds for cycles of two and three stages, the table of Result 14 understates the loss about two-fold.

## What is computed

The model is the discrete ring of draft 7 at its threshold: stage k has stay τ_k ≥ 1 steps, s_k = 1 − 1/τ_k, mean transfer g_k = 1/τ_k (so R0 = 1 and λ = 1), and the transfer out of stage k is multiplied by h_k(t) = exp(σ_k z_k(t) − σ_k²/2), z_k a stationary Gaussian AR(1) series of unit variance and autocorrelation φ_k. One transfer is also multiplied by e^y. D is the y at which the long-run growth rate is zero. By Proposition 20(b) (proved, refereed) D does not depend on the mean transfers or on which transfer carries y, so this D is the loss of ln R0 for every R0.

**Method.** M = 200 populations in parallel, T = 200,000 steps each after a burn-in of 2,000, renormalised every step; the growth rate is the mean log growth per step. The same random numbers are used for every y (fixed seed per case), and the root is found by the secant method to 1e-6 in y. Standard error of D: the standard deviation across the 200 populations of their growth rates, divided by √200 and by the slope dr/dy from the last two secant points.

## Hypotheses

| # | Statement | Basis | Kind |
|---|---|---|---|
| H1 | The simulation reproduces (a) the exact one-stage D from quadrature, for s in {0.5, 0.9, 0.96}, σ in {0.6, 1.0, 1.4}, φ in {0, 0.75}; (b) D = Σσ_k²/2 for a three-stage ring with no staying, σ = 1.0, φ in {0, 0.75}; (c) Γ₀ within 5% at σ = 0.2 for stays (1, 24) and (1, 1.25, 2) at φ = 0.4. | exact results; R9 | check. Tolerance for (a) and (b): the larger of 3 standard errors and 2% of D. |
| H2 | In every case D ≤ Σσ_k²/2, and D ≥ 0 when φ = 0. | an argument written this morning (a lower bound on growth from Jensen's inequality along a reference line of descent; to be Proposition 21), not yet checked by anyone | check of a derivation. Tolerance 3 standard errors. Any violation is reported as a failure of the argument. |
| H3 | D/Γ₀ rises with σ and at σ = 1.4 lies between 1.5 and 3 for cycles with one long stay. | the one-stage exploration | open |
| H4 | For the seven parameter sets of `verification/systems/SYSTEMS.md`, Table A, D exceeds the tabulated second-order value with memory. | H3 | open |
| H5 | For a fluctuating stage of stay 10 with host memory φ = 0.9, adding a steady stage of stay b changes D with the same sign as the second-order formula predicts, at σ = 0.3; at σ = 1.0 and 1.4 the sign is not predicted. | Result 4; nothing at large σ | open |

## Grid, with the reason for each range

- **σ scan (H3):** stays (1, 24) (the schistosome set), (1, 7.5) (malaria), (1, 2.273) (*Borrelia*), (1, 1.25, 2) with the middle transfer fluctuating (*Echinococcus*); φ in {0, 0.4, 0.75} (none, and the two ends of secondary's range); σ in {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0} (0.2 is inside the range where the formula was verified; 1.4 is the largest value in the table; 2.0 is beyond it).
- **Real sets (H4):** the seven of Table A as secondary parameterised them (1a, 1b, 2a, 3a, 3b, 5, 6, 7; 2b adds a second fluctuating host and is included). For the three-stage set the fluctuating transfer is taken out of stage 2 (the egg), and also out of stage 1 and out of stage 3, to see whether position matters at large σ (at second order it does not).
- **Parking (H5):** stay a = 10, φ = 0.9; b in {2, 5, 19, 50, 200} (19 is the second-order critical stay of run R1f); σ in {0.3, 0.6, 1.0, 1.4}. The one-stage value is exact.

## Budget and stop

About 190 cases, each a few seconds to a minute; eight processes; stop at 30 minutes of wall time and report what finished. No rerun with other settings unless H1 fails, in which case the method is fixed and the whole run repeated, and the failed output kept.

## What is not tested

Continuous time; hosts whose fluctuation is not lognormal (a two-state boom and bust in particular); more than three stages; growth away from the threshold.

## Amendments made before the run (09:34 by the clock, after a smoke test at one-tenth length on three cases; the run started at 09:34:19)

1. **Method, ring with no staying.** Its n cohorts never mix, so the log of their sum exceeds the log of one cohort by a term of order σ√T, and the estimate of D converges like 1/√T (the smoke test gave 1.4815 ± 0.0016 against 1.5 at T = 20,000; the expected excess there is 0.018). For that ring only, the populations start in one stage, so that a single cohort is followed.
2. **H1c tolerance.** At σ = 0.2 the smoke test's standard error was a tenth of D. At full length it will be about 3% of D, and the fourth-order term is expected to add 2 to 4%. The tolerance is the larger of 5% and 3 standard errors. This widens a check; it is recorded for that reason.
