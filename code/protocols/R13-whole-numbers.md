# Protocol R13: extinction with whole numbers of parasites

tertiary, 14 November 2026 (the file's time stamp is the record of when; written before the script was run). One run, one report (`reports/R13-whole-numbers.md`), then stop.

## Why

Every statement about risk in draft 7 is about a continuous population falling ten-fold (runs R1c to R1f), and its Limits say so: "no demographic noise". Dobson's sentence is about persistence "of species that would otherwise be susceptible to extinction". This run asks whether the parking-lot pattern of run R1f survives when parasites are counted in whole numbers and extinction means the last one dying.

## Model

The ring of the manuscript read as a branching process in a random environment. In each step every individual of stage k stays with probability s_k; each one that does not stay leaves, and the leavers of stage k together produce a Poisson number of individuals of stage k + 1 with mean b_k h_k(t) per leaver. The expected numbers follow the matrix model with g_k = (1 − s_k) b_k. h_k(t) = exp(σ z(t) − σ²/2) for the fluctuating transfer, z a Gaussian AR(1) series; 1 for the steady one. Each replicate has its own environment. With two stages the multiplication is split evenly, b_k = √R0 (R0 is within half a percent of one in the cases below, so the split hardly matters; it is fixed here so that it is not a choice made afterwards).

A replicate is **extinct** when every stage is empty, and **escaped** when the total passes 10⁶, after which it is not followed.

## Cases

**W0 (check that the threshold is where R0* says).** Stays (1, 7.5), σ = 0.6, memory 0.75: the exact loss is D = 0.0526 (run R10). ln R0 = D − 0.03 and D + 0.03. 2,000 replicates from 200 individuals, 3,000 generations. Expected: below the threshold every replicate dies; above it some escape. *Criteria:* extinct fraction at least 0.99 below; at most 0.90 above, with at least 0.10 escaped. (The theorem behind this, that a branching process in a random environment dies out with certainty exactly when its growth rate is not positive, is Athreya & Karlin 1971 and Tanny 1981; recalled, not verified.)

**W1 (open): the cases of run R1f with whole numbers.** σ = 0.15; rates set as in R1f so that the stays of a line of descent are exactly a = 10 and b.
- N1: memory 0.9, ln R0 = 0.2σ² = 0.0045; steady stay none, 10, 40, 160.
- N2: no memory, ln R0 = 0.06σ² = 0.00135; steady stay none, 10.
Starting numbers 100, 1,000 and 10,000, placed in the stages in the proportions of the stable distribution. 2,000 replicates each; 6,000 generations (6,000 × 𝒯 steps), as in R1f.

For reference, R1f's probabilities of a ten-fold fall of a continuous population were 0.642, 0.789, 0.267 and 0.0005 for N1 and 0.411 and 0.036 for N2 (the last two are the diffusion values of the protocol table; the report will quote the simulated ones from `reports/R1f-parking-near-threshold.md`).

*Questions, no criterion:* (i) Is the order of the four N1 cases the same as in R1f at each starting number, in particular is a steady stay of 10 worse than none? (ii) How does the probability of extinction fall with the starting number in each case? (iii) Does the steady stage change the starting number needed for a given safety?

*One prediction stated beforehand:* at the largest starting number the order of the N1 cases is that of R1f (10 worst, then none, then 40, then 160). At the smallest I do not predict the order: there the deaths and births of individuals should matter more than the hosts.

## Measured

Fractions extinct and escaped with binomial standard errors; the median time to extinction among those that die.

## Budget and stop

Twenty minutes of computing on six processes. If a case has not finished by then it is reported as unfinished. No second run with other numbers.

## Not tested

Other ways of turning rates into individuals (for example a parasite that both stays and transmits in a step); aggregation of parasites among hosts; mating; hosts counted in whole numbers.

## Amendment after the first run (recorded when made; the clock time is the file's time stamp)

The first run counted a replicate as "escaped" once its total passed 10⁶ and stopped following it. Check W0 failed: below the threshold 8.8% of replicates reached 10⁶ before dying. So that rule was unsafe, and it biased every extinction fraction downward. The whole run is repeated with every replicate followed to the end of its horizon (and stopped only above 10¹⁵). W0's criterion is unchanged, reading "escaped" as "not extinct at the end of the horizon". The first output is kept (`results/r13-whole-numbers_attempt1_20261114.txt`). This is the repeat the protocol allows for a failed check of the method, not a second run with other numbers; the budget is raised to forty minutes because replicates are followed for longer.
