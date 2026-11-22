# Protocol R12c: how strongly do mouse numbers enter the two steps of the *Borrelia* cycle? (Cary Institute plots, 1995 to 2013)

tertiary, 14 November 2026 (the file's time stamp is the record of when). Written after run R12 and before any of the quantities below was computed by anyone on this project. One pass, one report (`reports/R12c-cary-how-mice-enter.md`), then stop.

## Why

Draft 7's row for *Borrelia* assumes that the step from tick to mouse rises in proportion to mouse numbers (relative size κ₁ = 1) and that the step from mouse to tick does not depend on them (κ₂ = 0), and says (Result 14, from secondary's reading of the transmission models) that the truth lies in 0 ≤ κ₁ ≤ 1 and −1 ≤ κ₂ ≤ 0: more mice means a larger share of nymphs find a mouse, and fewer larvae on each mouse. The loss of ln R0 scales with the squares and the product of these two numbers (proofs file, Proposition 4a). Run R12 found that infection in nymphs barely follows mouse numbers (elasticity 0.04 ± 0.06). The same file records larvae per mouse, which measures κ₂ directly. This protocol uses the data to set the range of κ₂, as rule 10 allows, and recomputes the loss. The mechanism (one host, two steps, opposite signs) was in the manuscript before these data were opened; nothing is added because of them.

## Data

`data/raw/cary_lyme/dryad_d1c8046_forest_food_web_2018/Lyme_Study_Dataset.csv`, years 1995 to 2013, with the column meanings established in R12 from the owners' README (to be re-read for any column not used in R12: larvae per mouse and, if present, nymphs per mouse or any other measure of tick burden).

## What is computed

1. **κ₂.** The within-plot slope of ln(larvae per mouse in year t) on ln(mice in year t), ordinary least squares with an intercept for each plot, standard error by the bootstrap over years of R12. Years with a zero in either variable are left out and counted. *Expectation stated beforehand (secondary's reading):* −1 ≤ κ₂ ≤ 0. *It fails* if the estimate lies outside that interval by more than two standard errors.
2. **If the file has nymphs per mouse** (or another per-mouse burden of nymphs): the same slope for it, s_N. Then the share of nymphs that feed on a mouse varies as mice^(1 + s_N) when the density of questing nymphs is held fixed, so κ₁ = 1 + s_N, reported with its standard error and with the caveat that nymph density is not held fixed in the data. If no such column exists, κ₁ is not estimated and that is the result.
3. **Reported, no hypothesis:** the within-plot slopes of ln(density of nymphs in year t + 1) and of ln(density of infected nymphs in year t + 1, density × prevalence) on ln(mice in year t).
4. **The loss, recomputed.** For the cycle of draft 7 (stays of 1 step in the mouse and 2.273 steps in the tick, steps of 4.4 months), both transfers follow one host series with relative sizes κ₁ on the transfer from tick to mouse and κ₂ on the transfer from mouse to tick, no memory (R12 found none), σ at both of R12's values (1.03 and 0.77). D is computed exactly by the method of run R10 (the shift at which growth is zero; 200 populations, 200,000 steps) for κ₂ at its estimate and at the estimate ± two standard errors, each with κ₁ = 0, 0.5 and 1 (and at the estimate of κ₁ if step 2 gave one). The second-order value from Proposition 4a (`checks/one_host_two_transfers.py`, function `closed_form`) is given beside each. *Check of the new code before use:* with κ₂ = 0 and κ₁ = 1 it must reproduce R12's D = 0.2013 at σ = 1.030 within three standard errors, and with κ₁ = κ₂ = 0 it must give D = 0 within three standard errors.

## What this can and cannot show

It can show how far the entry in draft 7's table moves when one of its two assumed relative sizes is replaced by a measured one. It cannot give the loss for *Borrelia*: κ₁ is at best indirectly measured, the stage durations and the reading of the cycle as two stages are secondary's parameterisation, and the cycle's true timing (larvae in August, nymphs the next June) is seasonal, which the model ignores.

## Budget

Thirty minutes. One pass.
