# Memory experiment protocol

Written before simulation, 14 November 2026. Local files only. No network.
Run Python with `python3 -I`. Use numpy, scipy and mpmath. At most four
worker processes. The final production invocation must take under 60 minutes.
Pilot and failed validation output are separate from production output.

## Models and matching

The two-stage recurrence is u[t] = s + (1-s) exp(y) h[t]/u[t-1].
D is the zero of the stationary mean of log(u). tau = 1/(1-s).

* Lognormal: z is a stationary unit Gaussian AR(1) with parameter phi;
  h = exp(a z - a^2/2), a in {0.3,0.6,1,1.5}. Thus phi describes LOG
  correlation. Host variance V = exp(a^2)-1; host lag-one correlation
  rho = (exp(a^2 phi)-1)/V. All lags are reported by this formula with
  phi replaced by phi^lag.
* Telegraph: match that lognormal mean, V and HOST rho. Set
  d=min(sqrt(V),0.8), low=1-d, high=1+V/d, Pr(high)=d^2/(V+d^2).
  The Markov transition is Pr(high next | state)=w+rho*(state-w).
  This keeps both levels positive even when V>1. Its covariance is V rho^lag.
* Gamma refresh: shape p=1/a^2, scale 1/p. At each step retain the old
  value with probability phi; otherwise draw a fresh gamma value.
  This is an explicitly specified stationary gamma Markov host, with
  covariance (1/p) phi^lag. It is not a beta-gamma autoregression.
  Additional gamma cells with p=1/V and memory rho provide exact matching
  to the lognormal and telegraph moments, subject to the time budget.
* Seasonal: [1+0.5 cos(2 pi t/12)] times the same lognormal AR host.
  One time step is one month, one year is 12 steps. The cycle has mean one.
  Seasonal phase is fixed, and production lengths are complete years.
  Its variance averaged over phase is 1.125 exp(a^2)-1. This is an
  illustrative cycle, not a calibrated host data set.

Memory grid: {0,0.3,0.6,0.8,0.9,0.95,0.99}. Initial stay grid: 25
log-spaced values from 1/0.99 to 1000, with 2,5,10,25,100 added for
interpretability. Refine any resolved turning point if resources permit.
If a pilot requires reduction, reduce stay points or supplementary cells
before reducing path length. Record the actual grid and omissions.

## Sampling and roots

Use stationary host starts, at least 4096 discarded steps, and at least
16 independent pairs of antithetic paths. Gaussian innovations have
opposite signs; Markov uniforms and gamma quantile uniforms are paired
as U and 1-U. Pair averages, not the two individual paths, are the
independent units for standard errors. Common environments are used
for every stay and every root iteration within a parameter cell.

Start with 131076 recorded steps per path (a multiple of 12). Use the
known mean-one control variate h-1 with a deterministic coefficient
1/T, where T is the total mean cycle duration. Its expectation is zero;
for the seasonal case subtract the known mean for each phase instead.
Solve the sample estimating equation by Newton iteration with a
pathwise derivative. Report the final estimating-equation residual,
the raw exponent and its standard error, and the controlled exponent.
Root standard errors use the independent pair means and the delta
method. Save pair root influences so differences and maximum locations
use their common-random-number covariance.

Target one-standard-error precision: max(5e-5, 0.02*abs(D)). A cell
which misses this target must be extended or clearly marked unresolved;
never label it accurate by simply weakening the target. Root numerical
error must be below max(1e-9,0.05*SE_D). Compare half-run estimates and
longer burn-in at selected worst corners. Sampling errors do not certify
burn-in error or every feature between sampled stays.

## Mandatory validation before the production map

1. At phi=0, simulate gamma at the six G4 cases in
   results/exact-gamma-parking_20261114.txt. Recompute full precision
   reference roots by invariant-density quadrature. Require agreement
   within 4 Monte Carlo standard errors, and agreement with the printed
   exact values to their rounding precision.
2. At log sd 0.03, check stays 2,10,100 and phi=0,0.6,0.95 against
   the threshold coefficient in notes/proofs-v2.md. Require point
   discrepancies below 5% and report Monte Carlo error relative to the
   coefficient. Do not use the singular no-staying limit as this check.
3. For every production telegraph two-stage cell, independently compute
   its exponent/root using the invariant joint (ratio, host-state)
   distribution with linear deposition on a log-ratio grid. Refine
   grid size until the root changes by less than max(1e-7,0.25*SE_D).
   Require agreement with the simulation root within 4 SE_D. Disclose
   discretisation errors separately. An unresolved grid check is not a pass.
4. Check deterministic hosts and the three-stage matrix update against
   their deterministic eigenvalue/periodic-product growth rates.

Preserve the first failed output as memB_output_attempt1.txt. Repair
the cause, state changes, and rerun the failed criteria without relaxing
their limits. A production validation failure must be visible in the report.

## Interpretation and three-stage extension

The second-order threshold is variance times
K=(1+2 theta)/(2T)-R/(1-R), theta=phi/(1-phi),
R=product theta/(theta+tau_k). For the actual two-stage stays (1,tau),
K=[phi+(1+2phi)tau]/[2(1+tau)(phi+tau)]. Its derivative has numerator
phi^2-2phi*tau-(1+2phi)*tau^2, which is negative for tau>=1.
There is therefore NO interior second-order boundary in this specific
two-stage model. H2 in exact-limits.md has wider scope.

For lognormal also evaluate the covariance-based second-order expression
by its positive exponential-series mixture, so that changes in higher-lag
covariance are separated from use of log variance. For seasonal hosts
include the deterministic covariance mode and its products with that
mixture. Label these finite-amplitude substitutions as approximations.

Classify monotonicity only on the computed grid. An adjacent rise greater
than 4 paired standard errors rejects decrease. A fully resolved decreasing
grid requires every adjacent difference below minus 4 paired standard
errors. Otherwise use 'unresolved, compatible with decrease'. Give a
candidate maximum set from paired 4-SE comparisons and bracket it by
neighbouring grid points. A decreasing tail is only certified on the grid
and only through tau=1000. The 4-SE intervals are pointwise, not simultaneous
confidence bands or mathematical proofs.

Report D/[V/(2(1+tau))] and D divided by the memory-aware coefficient
at long stays. For independent gamma the first ratio is 2p(1+tau)D.
Do not assume the memory-aware all-noise limit before computing it.

Three-stage ring: stays (1,tau,5), one fluctuating transfer out of the
one-step stage. The other stages are steady. Transfers have product
exp(y)/(5*tau). Compare the same root and ring coefficient. Use a subset
of amplitudes and memory values if needed, and state it explicitly.

Differences between matched lognormal and telegraph hosts cannot by
themselves identify higher-lag memory: their marginal distributions and
higher-order joint moments also differ. Matched gamma refresh and
telegraph have the SAME entire covariance sequence, a useful extra check.

## Deliverables

memB.py, memB_output.txt, memB_tables.csv, and memB_report.md.
The report includes a table for every computed (model,a,phi), uncertainty,
validation status, 'what this means for the parking lot with real host
memory', and 'not done'. Retain machine-readable replicate evidence and
intermediate checkpoints if useful. No uncomputed cell is reported as a result.

## Allocation fixed after the first run, before the final run

The initial 144-cell run took 176.6 seconds. Mandatory validation passed,
but 14 cells missed the precision target. Its output is retained as
memB_output_attempt1.txt, with all its arrays in memB_data_attempt1/.
The final run keeps the full 30-point stay grid, all 112 primary cells,
8 matched-gamma cells, and 24 three-stage cells. No primary grid was reduced.

The frozen allocation in memB_sampling_plan.json uses B=ceil(4 r^2)
independent batches, at least one, where r is the worst initial SE divided
by its precision target. Each batch has 16 independent antithetic pairs
and 131076 recorded steps per path. The largest allocation is 80 batches.
The final seeds differ from the allocation pilot. Thus the allocation does
not stop the final samples according to their observed errors. Every batch
uses the same candidate root; their estimating equations are pooled before
each Newton step. Recreating the same batch seeds at every step preserves
common random numbers without storing all environments in memory.

The first-run roots are starting guesses only. Final values use fresh
simulation data. The accuracy limits and validation criteria are unchanged.

## Final audit repairs

The larger run took 826.5 seconds. One seasonal group (a=1.5, phi=0.6)
missed the SE target by 0.6%. One telegraph group (a=1, phi=0.99)
missed the independent-operator comparison, at 4.23 SE after an independent
root solve (4.25 SE using the initial local linearisation). Their replacements
use five and four batches respectively, with fresh seeds fixed beforehand in
memB_repair_plan.json. Their original arrays are in memB_data_attempt2/.
The second failed output and audit are retained. No comparison limit changes.

The first audit also incorrectly included three-stage gamma rows in a
two-stage exact-formula check. That filter is corrected. Its half-run
diagnostic split 10923 seasonal years into two pieces with half a year at
the boundary. The corrected seasonal diagnostic uses the first 5461
complete years against the full run. It is recomputed on the same original
paths. Full-run means, roots and standard errors are unaffected by these
two diagnostic fixes. Future runs use the corrected window directly.

A roundoff warning in a near-zero gamma integral motivates an additional
check of 12 quadrature roots against independent 50-digit Tricomi roots.
This does not change the original validation criteria or SE targets.
