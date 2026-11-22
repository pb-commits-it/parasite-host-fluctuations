# Protocol R3 — several alternative hosts at one stage

Written 2026-11-14 by tertiary, before `checks/r3_alternative_hosts.py` was run on any case below. Rung R3 of the plan, narrowed after the prior-art search to unequal shares, correlated hosts and hosts with different memory.

## Question

When a stage can pass through any of M alternative hosts, its transfer is g_k times a weighted mean of their multipliers, Σ_m π_m h_m, with shares π_m that sum to one. Does the cost follow the simple rule that the second-order theory gives?

## What the theory says

The second-order cost is linear in the autocovariance function of the multiplier of each rate (`notes/general-ring.md`, section 3), and Theorem G′ (`notes/error-bound-general.md`, section 9) covers several contributions to one matrix entry. For hosts with variances σ_m², autocorrelations φ_m and correlation ρ between the log abundances of any two of them (shared through a common component with autocorrelation φ_c), the contribution of stage k is

    Σ_m π_m² σ_m² [(1 − ρ) F(φ_m)] + (Σ_m π_m σ_m)² ρ F(φ_c),        F(φ) = (1 + 2θ)/(2𝒯²) − (1/𝒯) R(φ)/(1 − R(φ)),

with F the cost per unit variance of a single host (`engine/theory.cost_transfer`). For M independent hosts with equal shares and equal noise this is 1/M of the single-host cost, which is the published portfolio effect (Doak et al. 1998, eq. 1; Frank & Slatkin 1990). The rule with correlation, for equal noise, is (1 + ρ(M − 1))/M (Doak et al., eq. 5).

## Hypotheses (all checks; basis: the formula above)

| | Statement | Cases |
|---|---|---|
| K0 | Known answer for the new code: one stage, two alternative hosts, against the two-dimensional integral E ln(s + g(π_1h_1 + π_2h_2)) | s = 2/3, g = 0.4, π = (0.7, 0.3), σ = 0.4, φ = (0, 0.9) |
| K1 | M equal independent hosts at every stage: 1/M of the single-host cost | ring A1 of protocol R2/R4 (s = (0.3, 0.85, 0.5), g = (0.6, 0.2, 0.9)), φ = 0.7, M = 1, 2, 4 |
| K2 | Unequal shares: the cost goes as Σ π_m² | same ring, π = (0.7, 0.2, 0.1) at every stage, φ = 0.7 |
| K3 | Correlated hosts: two hosts with equal shares and correlation ρ = 0.5 through a common component with the same φ | same ring, φ = 0.7 |
| K4 | Hosts with different memory: π = (0.5, 0.5), φ = (0, 0.9) | same ring |

## What is run and measured

Each case at σ = 0.05: 400 replicates of 120,000 steps after 2,000 discarded, with the control variate (the elasticity-weighted sum of the aggregate multipliers minus one). K0 at σ = 0.4. Measured: the cost with its standard error over replicates. New code: `engine/environments_r3.py`. Seeds are fixed in the script.

## Passes if

K0 within 3.5 standard errors of the integral. K1 to K4: simulated cost within the larger of 3.5 standard errors and 1.5% of the formula (at σ = 0.05 the fourth-order term of this ring is known from run R2/R4 to be of order 0.05%, so 1.5% is generous; a failure would mean the rule is wrong).

## Budget and stop

Ten minutes of computing. The report is a dated section appended to `reports/R2-R4-general-ring.md`. Then stop.
