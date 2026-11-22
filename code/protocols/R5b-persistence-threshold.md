# Protocol R5b — does the cost of fluctuation set the threshold for persistence when the parasite is regulated?

Written 2026-11-14 by tertiary, before `checks/r5b_persistence.py` was run on any case below. Part of rung R5.

## Question

All results so far are for a parasite whose numbers do not affect its own rates, and are stated as a loss of long-run growth. A real parasite population is held down by density dependence and does not grow in the long run. The standard argument is that fluctuation then decides whether the parasite can persist at all: it persists when its long-run growth while rare is positive (recalled: Chesson & Ellner 1989; Schreiber, Benaïm & Atchadé 2011; not checked against the sources). If so, the cost computed here is the amount by which ln λ0 must exceed zero. This run tests that for the ring.

## Model

The ring of n equal stages (staying s, mean transfer ḡ, each transfer following its own host with σ and φ), with one change: the transfer out of stage n is divided by 1 + x_n/K, with K = 1. When the parasite is rare this is the linear model. Staying share s/λ0 = 0.625 as in earlier runs; λ0 is set case by case by scaling s and ḡ together.

## Hypotheses

| | Statement | Kind | Basis |
|---|---|---|---|
| P0 | Without fluctuation the number in stage n settles at K(R0 − 1), R0 = (ḡ/(1 − s))ⁿ | check, known answer | the equilibrium of the model |
| P1 | If ln λ0 is below the cost by 0.01, the log of the total falls at 0.01 per step in the long run | check | when rare the model is linear |
| P2 | If ln λ0 is above the cost by 0.01, the total neither grows nor dies: its mean log size settles to the same value from a start at 10⁻⁶ and from a start at 10 | check | invasion criterion (recalled) |
| P3 | How the settled mean log size depends on the margin above the threshold | open | — |

The cost used to set λ0 is the two-term formula C_n σ² + H_n σ⁴, which run R1c found accurate to 0.3% at this noise for up to four stages.

## Cases

σ = 0.4; n = 1 and 4; φ = 0 and 0.9. For each of the four, margins −0.01 and +0.01 (P1, P2), and for n = 4, φ = 0.9 also margins +0.002, +0.005, +0.02, +0.05 (P3). Reasons: one stage and four stages bracket the hump; the margin of 0.01 is half the cost itself in these cases, so the threshold is tested to within half its own size.

## What is run and measured

200 replicates of 200,000 steps. The state is carried as a composition and a log size, so that very small totals do not underflow. Measured: the slope of the mean log total over the second half of the run (with a standard error from the replicate slopes), and the mean log total over the last 20,000 steps, for the start at 10⁻⁶ and, where P2 applies, for the start at 10.

## Passes if

- P0: within 10⁻⁸ relative after 20,000 steps, for n = 1, 2, 4, 8 at λ0 = 1.05.
- P1: the slope is within the larger of 3.5 standard errors and 0.0005 of −0.01.
- P2: the slope is within 3.5 standard errors of zero or below 0.0002 in size, and the settled mean log sizes from the two starts differ by less than 3.5 standard errors of their difference.

## Budget and stop

Fifteen minutes of computing. The report is `reports/R5b-persistence-threshold.md`. Then stop.
