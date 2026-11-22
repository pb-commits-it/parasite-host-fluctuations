# Protocol R12b: the holdout years at the Cary Institute plots (2014 to 2022)

tertiary, 14 November 2026 (the file's time stamp is the record of when). Written after run R12 (`reports/R12-cary-single-population.md`, years 1995 to 2013) and before the holdout file was opened by anyone on this project. The repository owner has cleared the file for this use (non-commercial research). One pass, one report (`reports/R12b-cary-holdout.md`), then stop.

## The holdout

`data/raw/cary_lyme/figshare_27871269_nymphal_infection_2025/`: `allthedata_grid.csv` (six grids by nine years: nymphs tested, proportion infected with *B. burgdorferi*, mean minimum number of mice alive during larval and during nymphal questing) and `NymphalInfectionData.csv` (one row per nymph tested). The two method notes and column keys in the same folder are to be read first. The assay (PCR on individual nymphs) differs from the one used before 2007, so levels of prevalence are not compared between periods; only variation within the holdout period is used.

What is not blind: the mouse numbers. Run R12 computed the standard deviation of ln mice from a file that runs to 2025. What is blind: every infection value after 2013.

## Predictions, fixed now

| # | Prediction for 2014 to 2022 | Basis | Fails if |
|---|---|---|---|
| B1 | On each grid the standard deviation across years of ln(prevalence in nymphs) is smaller than that of ln(mice during larval questing, previous year). | Claim K3 of `notes/coupled-hosts.md` (a host population passes on at most the fraction 1 − prevalence of a proportional change); held on 6 of 6 plots in R12. | it fails on more than one grid of six (grids with fewer than five usable years are left out and counted) |
| B2 | The within-grid slope b₂ of ln(prevalence in year t) on ln(mice during larval questing in year t − 1) agrees with the first-period slope b₁ = 0.038 (standard error 0.058): \|b₂ − b₁\| ≤ 2√(0.058² + SE₂²). | Replication of R12; and the theory's bound 0 ≤ b ≤ 1 − p̄. | the difference is larger than that; reported separately: whether b₂ is inside the theory's bound [0, 1 − p̄₂] within two standard errors |
| B3 | Prevalence is no more variable where it is measured more precisely: the result of B1 does not depend on sampling error. Checked by recomputing B1 with the binomial sampling variance of each yearly prevalence subtracted (the variance of ln prevalence from n nymphs with p infected is about (1 − p)/(np)). | Sampling error inflates the variation of prevalence and so works against B1; this makes the test fair in the other direction. | B1 changes from holding to failing or the reverse (reported; no criterion) |

Years in which fewer than 20 nymphs were tested on a grid are left out of B1 and B2 and counted. Standard errors by a bootstrap over years (blocks of two consecutive years, the same years for all grids), 2,000 resamples, fixed seed, as in R12. Ordinary least squares with a separate intercept for each grid.

## What this can and cannot show

It can show whether the two pattern checks of R12 hold again on infection data nobody on the project had seen. It cannot test the main prediction of the theory (more variable hosts, lower R0): that needs sites that differ in host variability (run R14).

## Budget

Twenty minutes. If a column does not mean what the inventory inferred, use the correct meaning and report the difference.
