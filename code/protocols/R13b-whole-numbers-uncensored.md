# Protocol R13b: extinction with whole numbers, with no replicate stopped early

14 November 2026. Written and saved before the script `checks/r13b_whole_numbers_uncensored.py` existed. The file's time stamp is the record of when; the script prints this file's time stamp and its own starting time, both read from the system clock, at the top of its output. One run, one report (`reports/R13b-whole-numbers-uncensored.md`), then stop.

This run corrects run R13 (`protocols/R13-whole-numbers.md`, `reports/R13-whole-numbers.md`). R13's files are not edited; the corrections are made by the new report.

## Why

A referee and an auditor raised six objections to R13. I checked the first against R13's saved output and it is right.

1. **Censoring.** R13 stopped a replicate when its total passed 10¹⁵ and counted it as not extinct. So each of its fractions is only a lower bound. The upper bound is that fraction plus the fraction stopped. From `results/r13-whole-numbers_20261114.txt` (arithmetic on the saved numbers, done before this protocol was saved):

| Case | Steady stay | from 100 | from 1,000 | from 10,000 |
|---|---|---|---|---|
| N1 | none | 0.753 to 0.904 | 0.525 to 0.818 | 0.359 to 0.773 |
| N1 | 10 | 0.934 to 0.941 | 0.799 to 0.816 | 0.631 to 0.663 |
| N1 | 40 | 0.826 to 0.831 | 0.389 to 0.425 | 0.121 to 0.193 |
| N1 | 160 | 0.709 to 0.727 | 0.086 to 0.202 | 0.001 to 0.292 |
| N2 | none, 10 | no replicate stopped | no replicate stopped | no replicate stopped |

   Check W0: 0.998 to 1.000 below the threshold, 0.5005 to 1.000 above it. The sentence "with a positive growth rate the chance of dying from there is negligible" was asserted and not shown.
2. **Horizon.** R13 followed every ring for 6,000 generations. That is 60,000 steps for the ring with no steady stage and 1,020,000 steps for the ring with a steady stay of 160. The rings were not compared over the same stretch of time.
3. **Starting conditions.** Equal totals of individuals placed in different stages are not equal starts.
4. **Offspring law and independence.** R13's individuals are independent and their offspring numbers are Poisson. Parasites that share a host share its death. The auditor built two cycles with the same mean matrix whose probabilities of extinction from 100 founders are 0.205 and 8.6 × 10⁻³³. So R13's conclusions hold for its offspring scheme only, unless shown otherwise.
5. The sentence that a branching process dies out "exactly when" its growth rate is not positive needs its conditions. This is dealt with in words in the report (section "The threshold statement and its conditions"), from memory and marked so.
6. At 100 the fall with the long stage (0.0445) is about 3.2 standard errors. "Does nothing" is not supported as an exact statement. This run gives the cases that start from 100 three times the replicates and reports the difference with its uncertainty.

## What had been run before this protocol was saved

Only two things. (a) A timing of the random number generator on arrays of fixed numbers (no model, no case). (b) The arithmetic on R13's saved output shown above. No ring was simulated.

## Model

As R13. A ring of stages. In each step every individual of stage k stays with probability s_k. Each one that does not stay leaves, and the leavers of stage k together produce a Poisson number of individuals of stage k + 1 with mean b_k h_k(t) per leaver. h_0(t) = exp(σ z(t) − σ²/2) for the transfer out of stage 0 (the fluctuating one), where z is a Gaussian series with memory φ from one step to the next and variance one. h = 1 for the other transfer. Each replicate has its own z. A replicate is **extinct** when every stage is empty. The **mean matrix** A has s_k on the diagonal and g_k = (1 − s_k) b_k below it (around the ring).

The 20 cases of R13 are taken from R13's own function `build()`, imported from its file, so the rates, starting numbers, horizons and seeds cannot differ from R13's.

## A. The sampler: no replicate is stopped

Every replicate is followed to the end of its horizon, whatever its size.

**Rule, applied to each random draw separately.** N_big = 10⁹.
- A binomial draw (the stayers of a stage that holds n individuals) is exact when n < N_big. When n ≥ N_big the number of stayers is n s + √(n s (1 − s)) ξ with ξ a standard normal number, rounded to a whole number and kept between 0 and n.
- An offspring draw with mean m is an exact Poisson draw when m < N_big. When m ≥ N_big it is m + √m ξ, rounded, not below 0.
- So a stage that falls below N_big is sampled exactly again from the next step. Extinction can only be reached through exact sampling.
- Counts are held in double precision. They are exact whole numbers up to 2⁵³ and rounded to a relative 1.1 × 10⁻¹⁶ above that.

**Error of the approximation, stated.** The mean and the variance of every approximated draw are exact. What differs is the shape. The skewness of a binomial count is (1 − 2s)/√(n s (1 − s)), and that of a Poisson count is 1/√m. At the switching size these are at most about 4 × 10⁻⁴ (the stage with a stay of 160, s near 0.994) and 3 × 10⁻⁵. The fluctuation they belong to has a relative size of at most about 4 × 10⁻⁴ of the number leaving, so the shape error moves a stage by a relative amount of order 10⁻⁷ in a step. Rounding adds a variance of 1/12 to a variance of at least 6 × 10⁶. These are my estimates, not a theorem; the checks V1 to V4 below are the evidence, including with N_big lowered to 10³, where the approximation is a million times coarser in n.

**Guard against overflow.** A replicate whose total passes 10²⁰⁰ is divided by 10¹⁰⁰ and the logarithm of its scale is kept. It is multiplied back if its scaled total falls below one (a true total below 10¹⁰⁰). While scaled, every draw of that replicate is approximated as above, with the noise scaled to match, and without rounding (the relative demographic noise is then below 10⁻⁵⁰, less than the rounding of double precision, so in that state the noise formula does not matter and cannot be tested). The script reports how many replicates needed scaling. It counts as an **anomaly** any step at which a scaled replicate has a stage whose true size is below 10³⁰. Criterion: no anomaly in any case.

**The same streams as R13 where that means something.** Each R13 case uses R13's seed, R13's starting draw, and R13's order of draws for as long as no draw of the case has reached N_big. From that step on the order differs. So R13b is not an independent replicate of R13: early extinctions are the same events.

## Checks of the method (all must pass before the main cases are run)

Tolerance everywhere: 4 standard errors, unless equality is required. "Binomial standard error" is √(p(1 − p)/n). For two simulated fractions the standard error of the difference is √(p̄(1 − p̄)(1/n₁ + 1/n₂)) with p̄ the pooled fraction. A check whose own condition of validity is not met is **void**. Void counts as not passed.

There are about 140 comparisons. A correct method fails at least one of them by chance about once in a hundred runs. A failure is not waved through: the output is kept, the cause is looked for, and it is reported.

**H0. The exact sampler reproduces R13.** R13's own function `run_case`, run again on five of its cases (both W0 cases; N1 with no steady stage from 100, 1,000 and 10,000), must give the counts in R13's saved output exactly (extinct, stopped above 10¹⁵). My reference sampler (`exact_run`: R13's algorithm with the times of extinction and the peak size recorded) must give the same counts on the same five cases. *Criterion: equality.*

**H0c. Shared stream.** In any main case in which no draw reaches N_big within R13's horizon, R13b's count of extinctions within R13's horizon must equal R13's. *Criterion: equality in every such case. If there is no such case the check is void and is reported as void; this one check does not block the run, because it can only be known afterwards.*

**V1. Known answer in a constant environment.** With σ = 0 the probability that the line of one founder of stage k is extinct within T steps is q_k(T), where q(0) = 0 and q(T + 1) = f(q(T)), and f is the offspring generating function of one step: f_k(q) = s_k q_k + (1 − s_k) exp(b_k (q_{k+1} − 1)) for Poisson offspring. From several founders the probabilities multiply. As T grows q(T) rises to the smallest fixed point of f, which is the probability of ever dying out; the script prints both, and compares the simulation with q(T) at the T simulated, which is exact for that T.
- Ring: stays 10 and 40, rates set as R13 sets them, ln R0 = 0.3, T = 3,000 steps, 20,000 replicates per start.
- V1a, Poisson offspring. Starts: one founder in stage 0; one in stage 1; one in each.
- V1b, clumped offspring (law E-i below; f_0(q) = s_0 q_0 + (1 − s_0)(1 + (b_0/k)(1 − q_1))^(−k)). Same three starts.
- V1c, shared host fate (law E-ii below; the types are a free individual and a group). Starts: one free individual; one group; 100 free individuals.
- Each is run twice: with N_big = 10⁹ (as in the main run) and with N_big = 10³ (to force the approximation and the switch back to be used).
- Compared at T/10, T/4, T/2 and T. *Criterion: every simulated fraction within 4 binomial standard errors of q(T), the standard error computed from the exact probability.*
- V1d, a fall through the switch. Same ring, ln R0 = −0.5, start 2 × 10¹⁰ individuals in the stable proportions (both stages above N_big), N_big = 10⁹, 10,000 replicates, Poisson and clumped offspring. The exact probability of extinction by T is the product of q_k(T) to the power of the starting numbers. Compared at the five times at which that probability first reaches 0.1, 0.3, 0.5, 0.7 and 0.9. *Criterion: within 4 binomial standard errors at each.*

**V2. Hybrid against exact sampling in a random environment, where exact sampling is possible throughout.** The exact arm is `exact_run`. It flags a replicate if a stage passes 2⁶² (the largest count the exact sampler can take). *Validity: no replicate flagged.* Hybrid arms: N_big = 10⁹ and N_big = 10³. 10,000 replicates per arm, each arm with its own seed.
- V2a, fall. The ring of check W0 (stays 1 and 7.5, σ = 0.6, memory 0.75) with ln R0 = D − 0.3, D = 0.0526; start 2 × 10¹⁰ in the stable proportions; 6,000 steps.
- V2c, fall with the longest stay. Stays 10 and 160, σ = 0.15, memory 0.9, ln R0 = −1; start 2 × 10¹⁰; 12,000 steps.
- For V2a and V2c, each hybrid arm against the exact arm: (1) the mean time to extinction, within 4 standard errors of the difference; (2) the fraction extinct by each of the three quartiles of the pooled times to extinction, within 4 standard errors of the difference; (3) the fraction extinct at the end, the same.
- V2b, rise and return. Check W0 below the threshold exactly as in R13 (ln R0 = D − 0.03, start 200) for 1,500 steps. Each hybrid arm against the exact arm: the fraction extinct at the end; the fraction whose total ever passed 10⁶; the fraction whose total ever passed 10¹⁰. *Criterion: each within 4 standard errors of the difference.*

**V3. The overflow guard changes nothing.** The ring of check W0 with ln R0 = D + 0.3, start 200, 8,000 steps, 1,000 replicates, run twice with the same seed: once with the guard as in the main run (10²⁰⁰, 10¹⁰⁰; not expected to act) and once with the guard set to act early (scale at 10⁶⁰ by 10³⁰). *Criterion: the time of extinction of every replicate identical; the logarithm of the final total of every survivor equal within 10⁻⁶. Validity: at least 100 replicates scaled in the second run and none in the first.*

**V4. One step: mean and variance, and the mean matrix of the alternative laws.** For each of the three offspring laws, with the rates of the N1 ring with a steady stay of 40 and h_0 fixed at 1.3: one step from 10⁶ units in each stage (exact branch) and from 4 × 10⁹ (approximated branch), 20,000 replicates. For Poisson offspring also the one-stage N1 ring.
- *Mean:* the sample mean of each stage, in parasites, within 4 standard errors of A(h) x, where A(h) is **R13's** mean matrix with h_0 = 1.3. This is the numerical check that laws E-i and E-ii leave the mean matrix unchanged.
- *Variance:* the sample variance of each stage within 4 √(2/(n − 1)) (4.0%) of the variance worked out by hand for that law.

## B. The cases of R13 again, uncensored

All 20 cases, R13's parameters and seeds, 2,000 replicates. The cases that start from 100 (six of them) get two further batches of 2,000 with new seeds, so 6,000; this is for objection 6. More replicates for the other cases are not run, for lack of time.

- **W0** (check, criterion unchanged from R13): 25,500 steps. Extinct fraction at least 0.99 below the threshold; at most 0.90 above it, with at least 0.10 not extinct.
- **W1** (open): N1 (memory 0.9, ln R0 = 0.0045; steady stay none, 10, 40, 160) and N2 (no memory, ln R0 = 0.00135; none, 10), from 100, 1,000 and 10,000.

Reported next to each R13 number: the R13b fraction extinct within R13's horizon, with its binomial standard error; how many replicates passed 10¹⁵ within R13's horizon and how many of those were extinct by the end of it; how many replicates used the approximation and how many needed scaling; the largest size from which a replicate later died.

**C1 (check of consistency).** Each R13b fraction at R13's horizon lies between R13's lower and upper bound, with an allowance of 4 standard errors (of the two fractions combined) at each end.

## C. Equal calendar time

Every W1 replicate is followed for **T_eq = 1,020,000 steps**, whatever its ring. Reason for the number: it is the horizon of the longest ring in R13 (6,000 generations of the ring with a steady stay of 160). It is the smallest equal number of steps that contains every horizon of R13, so the reading at equal generations and the reading at equal steps come from the same replicates, and no ring is followed for less than R13 followed it. For the ring with no steady stage it is 102,000 generations. N2 is followed for the same number of steps, for simplicity.

Two readings for each case: extinct within R13's horizon (**equal generations**: 60,000, 120,000, 300,000 or 1,020,000 steps) and extinct within T_eq (**equal steps**). Also the fraction extinct by 10%, 25%, 50% and 100% of T_eq (102,000, 255,000, 510,000 and 1,020,000 steps).

## D. Starting conditions (headline comparison only: N1, the four rings, from 10,000)

Reproductive value: the left eigenvector u of the mean matrix (u A = λ u). It is defined within one ring up to a factor. To compare rings the factor is fixed by giving an individual of the fluctuating stage (stage 0, which every ring has) the value one. This is a choice; another choice would rescale a whole ring.

- S1: R13's own. 10,000 individuals drawn into the stages in the stable proportions v (A v = λ v).
- S2: equal total reproductive value, all in the fluctuating stage: (10,000, 0).
- S3: equal total reproductive value, all in the steady stage: (0, 10,000/u_1 rounded).
- S4: the stable proportions scaled to the same total reproductive value: c v rounded, with c = 10,000/(u·v). No random draw.

The ring with no steady stage has one stage; all four starts are the same 10,000 individuals there and it is run once. I expect u_1 to be within one percent of 1, because R0 is within half a percent of 1; if so S4 will be nearly S1 and the informative contrast is S2 against S3. The script prints u and the starting numbers.

## E. Offspring law (headline comparison only: N1, the four rings, from 10,000, start S1)

Two alternatives with the same mean matrix, the same R0 and the same distribution of stays.

- **E-i, clumped offspring.** Each individual that leaves the fluctuating stage (stage 0) leaves, independently, a negative-binomial number of individuals of the next stage, with mean b_0 h_0(t) and dispersion k = 0.1 (variance mean + mean²/k; about 79% of leavers then leave none). The offspring of L leavers together are negative binomial with mean L b_0 h_0 and dispersion k L, drawn exactly as a Poisson number whose mean is a gamma number of shape k L and scale b_0 h_0/k. When L b_0 h_0 ≥ N_big the draw is approximated by a normal number with that mean and variance L b_0 h_0 (1 + b_0 h_0/k). The other transfer stays Poisson. Why this transfer: R13's rings have no amplifying transfer (each transfer multiplies by e^(ln R0/2), within 0.3% of one), and the fluctuating transfer is the one every ring has, so each ring gets exactly one clumped transfer per generation. *Mean unchanged, in one line:* a negative-binomial number with mean m has mean m, so the expected arrivals in the next stage are (1 − s_0) b_0 h_0 x_0 as before.
- **E-ii, shared host fate.** The parasites of the steady stage sit in hosts, B = 100 to a host. In each step a whole group stays (probability s_1) or leaves (probability 1 − s_1) together. A group that leaves produces a Poisson number of free individuals of stage 0 with mean 100 b_1. Infection arrives in whole groups: the leavers of stage 0 together found a Poisson number of new groups with mean b_0 h_0(t)/100 per leaver. Individuals of stage 0 are independent as before. So 10,000 parasites in the steady stage are 100 independent units. *Mean unchanged, in one line:* with G groups, E[100 G′] = 100 (s_1 G + (b_0 h_0/100)(1 − s_0) x_0) = s_1 (100 G) + b_0 h_0 (1 − s_0) x_0, and E[x_0′] = s_0 x_0 + 100 b_1 (1 − s_1) G = s_0 x_0 + b_1 (1 − s_1)(100 G). Start: R13's draw; the steady-stage number x_1 becomes ⌊x_1/100⌋ groups plus one more with probability equal to the remainder over 100. The ring with no steady stage is unchanged by this law and is not run again.

Both are checked numerically in V4 and against a known answer in V1b and V1c.

## How an ordering is judged (fixed now)

For two cases with extinct fractions p_A and p_B: d = p_A − p_B, with standard error √(se_A² + se_B²).
- |d| at least 4 standard errors: **established**.
- between 2 and 4: **suggested, not established**.
- below 2: **no difference detected**; the report gives d and its standard error, never "equal".

An ordering stated by R13 **survives** if it is established in R13's direction, **fails** if it is established in the other direction, and is otherwise **not established**. This is judged twice, at equal generations and at equal steps. Separately the script says whether R13's own numbers had established it: the lower bound of the larger less the upper bound of the smaller, at least 4 standard errors.

The orderings of R13 that are judged (R13's direction first):
- N1 from 10,000 and from 1,000: stay 10 above none; none above 40; 40 above 160; none above 160.
- N1 from 100: stay 10 above none; 40 above none; none above 160 (R13: "0.75 to 0.71", and "does nothing"); 10 above 40.
- N2: none above stay 10 from 10,000 and from 1,000; from 100, R13 said "no effect" (0.932 against 0.928).
- R13's "numbers needed" (N1, none from 10,000 about equal to stay 40 from 1,000): both numbers are reported at both horizons.

By the rule above, R13's own numbers had **not** established: stay 10 above none (at any starting number; from 100 the ratio is 3.5), 40 above 160 from 10,000, 40 above none from 100, none above 160 from 100. They had established: none above 40 and none above 160 from 10,000 and 1,000; 40 above 160 from 1,000; 10 above 40 from 100; none above 10 in N2 from 10,000 and 1,000.

## Predictions, stated beforehand

Rough, by hand, from the growth rates and the index E of run R1f; not computed by a script.
- P1. In the ring with no steady stage from 10,000, a population at 10¹⁵ still has a chance of roughly 0.1 of dying out eventually (the index gives a fall of about 0.07 in log probability per unit of log size). So of the 0.41 that R13 stopped, a few hundredths die within R13's horizon: I expect 0.37 to 0.42 at equal generations, and so "stay 10 above none" to survive at equal generations.
- P2. At equal steps the ring with no steady stage rises further, towards 0.55 to 0.6, and the ring with a stay of 10 rises too (its growth is nine times slower); I expect the gap to narrow and not to close.
- P3. The ring with a stay of 160 from 10,000 stays below 0.01.
- P4. Shared host fate raises extinction in every ring with a steady stage. I do not predict whether a stay of 160 still protects under it.
- I make no prediction for the starts from 100.

## Budget and stop

REPS as stated. Six processes. Estimated 35 to 50 minutes. The main cases have a deadline of 90 minutes from the start of the run; a case not finished by then is reported as unfinished, with the step it reached. No second run with other numbers. If a check fails because of the harness, the failed output is kept as `results/r13b-whole-numbers-uncensored_attempt1_20261114.txt`, the fault is fixed and said, and the run is repeated.

Before the run the script is exercised only in a mode `--smoke`, which replaces every ring by toy rings that are not cases of this protocol (σ = 0.3, ln R0 = 0.05 or −0.2, horizons 500 times shorter, 30 to 200 replicates, other seeds), to see that it runs and to time it. Its numbers are not results and are not kept.

## Not tested

Group sizes other than 100, or unequal; parasites that enter hosts one by one and accumulate; hosts counted in whole numbers with their own births and deaths; dispersion other than 0.1; clumping on the other transfer or on both; clumping and shared fate together; mating; starting conditions and offspring laws at starting numbers other than 10,000; horizons beyond 1,020,000 steps (the probability of ever dying out); rings of more than two stages; values of N_big other than 10⁹ and 10³.

## Amendment 1, made after the first run was stopped (the clock times are from the system clock)

The protocol above was saved at 11:26:41. The first run started at 11:38:08. Its checks of the method all passed (H0, V1a to V1d, V2a to V2c, V3, V4) within the first minute. I stopped it by hand at 11:46:28, during the main cases, before any main case had finished or been seen (0 of 48 runs done). Reason: the machine turned out to be shared with ten other heavy processes (16 busy processes on 10 processors; each of the six workers of this run was getting about 65% of a processor). At that rate the 90-minute deadline written into the run would probably have cut cases off unfinished, and a case started after the deadline would have been cut at step 0.

Change: the deadline for the main cases is raised from 90 minutes to 240 minutes (the constant `BUDGET_S` in the script). The estimate of the running time becomes 60 to 100 minutes while the machine stays shared. Nothing else is changed: same cases, same seeds, same replicates, same checks and tolerances, same rules for judging. The first output is kept as `results/r13b-whole-numbers-uncensored_attempt1_20261114.txt` with a note at its top. This is not a second run with other numbers, and it was decided with no main result known.
