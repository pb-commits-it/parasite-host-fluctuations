# Protocol R12: one population with hosts and infection measured together (Cary Institute plots)

tertiary, 14 November 2026 (the file's time stamp is the record of when; written before any of the data files were opened by me or analysed by anyone on this project. The data scout described the files' columns and coverage in `data/INVENTORY.md` and computed nothing). One pass, one report (`reports/R12-cary-single-population.md`), then stop.

## Why

Draft 7 says: "For no parasite with a complex life cycle did we find all the quantities the theory needs measured in one population", and its row for *Borrelia burgdorferi* in oak forest takes the variability of mice (σ = 0.67) from a study elsewhere. Open data from six plots at the Cary Institute of Ecosystem Studies, Millbrook, New York, give white-footed mouse numbers and questing-tick density for 30 to 35 years and the infection prevalence of questing nymphs for 1995 to 2013, on the same plots (`data/INVENTORY.md`, section 1). This protocol says what will be computed from them and what will not.

Rule 10 of `study rules` governs: data set ranges and check patterns. Nothing is fitted, no parameter of the theory is tuned to these data, and no mechanism is added because of them.

## Data and its limits (from the inventory, not yet checked by analysis)

- `data/raw/cary_lyme/dryad_d1c8046_forest_food_web_2018/Lyme_Study_Dataset.csv`: per plot and year, mice per grid (`PL_grid`), chipmunks (`TS_grid`), density of questing larvae, nymphs and adults (`DOL`, `DON`, `DOA`), larvae per mouse (`LPM`), and nymphal infection prevalence with *B. burgdorferi* (`NIP`), 1995 to 2013 on six plots. The column meanings were inferred by the scout from names; the README is a PDF in the same folder and **must be read first**, and any meaning that differs from the inference reported.
- `data/raw/cary_lyme/figshare_31872853_ecology_of_lyme_2026/Full_dataset_20260326.csv`: mice (`PL_t`), chipmunks (`TS_t`), peak questing larvae and nymphs, 1991 or 1995 to 2025.
- Yearly values only. The parasite's shortest stage lasts months, so variation within a year is not in these data.
- Six plots in three pairs on one property: replicates in one landscape, not independent populations.
- Infection is measured in ticks, not in mice. Two assays were combined around 2007.
- Stage durations and R0 are not in the data.

## What is computed

**A. Ranges for the theory's inputs (no hypothesis; reported).**
For each plot, and for the mean across plots in each year: the standard deviation of ln(mice per grid) across years, and its autocorrelation at a lag of one year; the same for chipmunks and for ln(density of nymphs). Years with a count of zero are left out of logarithms and counted. Reported with a bootstrap over years (2,000 resamples of blocks of two consecutive years) for an interval. Set beside the values draft 7 used for this row: σ = 0.67 (range 0.46 to 0.87) and a memory of 0.30 per step of 4.4 months.

**B. Two pattern checks derived from the theory before looking (each can fail).**

| # | Prediction | Where it comes from | Fails if |
|---|---|---|---|
| P1 | The prevalence of infection in nymphs varies less, on the log scale, than the abundance of mice: sd of ln NIP < sd of ln mice, plot by plot. | Claim C3 of the sealed derivation: a population of hosts that are susceptible or infected passes on at most the fraction 1 − prevalence of a proportional change in infection pressure. | it fails on more than one plot of six |
| P2 | Within plots, the slope b of ln NIP in year t + 1 on ln mice in year t (the mice that fed the larvae that became those nymphs) satisfies 0 ≤ b ≤ 1 − p̄, with p̄ the mean prevalence in nymphs. | The same claim, with the further assumption (stated, not tested) that the infection pressure on larvae rises no faster than in proportion to mouse numbers. | b is below 0 or above 1 − p̄ by more than two standard errors (ordinary least squares with a separate intercept for each plot; standard error from a bootstrap over years, because plots in one landscape share their years) |

P2 is a weak test: it can only reject an elasticity above about 0.6 or a negative one. It is stated because it is what the theory says about these variables and nothing sharper can be said without a model of the other hosts ticks feed on, which the data do not contain.

**C. One recomputation.** The exact loss D of ln R0 for the *Borrelia* cycle of draft 7 (stays of 1 and 2.273 steps of 4.4 months), by the method of run R10, with σ replaced by the value from A (mean within-plot standard deviation of ln mice) and the memory per step taken as r₁^(4.4/12), r₁ being the one-year autocorrelation from A. That conversion assumes the memory decays geometrically within the year, which yearly data cannot test; if r₁ is not positive the memory is set to zero and that is said.

## What is not computed, and why

- R0, or any test of the predicted loss of R0: R0 is not measured here, and one landscape gives no contrast in host variability.
- Any relation involving acorns, deer or weather: those are the data owners' subject and not ours.
- Anything from the 2014 to 2022 individual-nymph file, whose licence is non-commercial and whose assay differs; it is left for a decision by the repository owner.

## Budget and stop

About twenty minutes. One pass. If the README shows that a column does not mean what the inventory inferred, the analysis is done with the correct meaning and the difference reported.

## Credit

The data are from Ostfeld, LaDeau, Keesing, Canham, Oggenfuss, Winchcombe and colleagues; the citations requested by the owners are in `data/INVENTORY.md` and go into the report and into any manuscript that uses the numbers.
