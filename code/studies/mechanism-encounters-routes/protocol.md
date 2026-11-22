# Preregistered mechanism checks

This file is written before inspecting numerical results or running a check.
All work uses this directory. There is no network access. All parameters below
are illustrative. The two comparisons are named explicitly throughout:

* baseline (i): a constant environment at its mean host abundance;
* baseline (ii): every rate fixed at its own arithmetic mean.

The loss is the log multiplier on a designated reproductive return edge needed
to make the top Lyapunov exponent zero, minus that required by the deterministic
mean matrix. This defines a threshold loss also for networks. For a ring it is
the loss of log R0 at the mean-matrix threshold. Away from threshold, a local
growth-rate correction is not an exact threshold loss. A negative loss is allowed.
Network R0 will be explicitly defined by expected lifetime reproductive output.

## Claims and methods, before computation

1. Re-derive the general second-order coefficient using the local notes. Check
   the supplied formula against direct matrix products, never against itself.
   Derive contact-limited ring coefficients with independent and AR(1) temporal
   noise, including the mortality fraction, gain-only ratio, and mortality
   limits. Do not use an unbounded encounter probability. The principal discrete
   encounter model uses a bounded two-state multiplier. Its variance is matched
   to exp(sigma^2)-1, and its lag correlation is phi^lag. Small-noise derivatives
   use the same covariance kernel as Gaussian AR(1). Finite-noise statements
   about this model are not called lognormal statements.
2. Test H1 and H2 for contact limitation, including zero waiting mortality,
   positive mortality, long steady parking, and persistent/alternating noise.
   A sign counterexample from an exact periodic matrix product overrides a
   universal claim. Differentiate or solve the explicit coefficient to locate
   local boundaries; verify a point on either side by direct products.
3. Derive the curvature/elasticity decomposition for type II, type III, powers,
   and inverse host dependence, separately for baselines (i) and (ii). Test
   the predicted sign boundary with independent direct products. Distinguish
   second-order boundaries from finite-amplitude boundaries.
4. Derive signed shared-driver coefficients for two transfers, for both signs,
   with and without memory. Check the cancellation criterion, not merely the
   sum of elasticities. Include an exact periodic example where applicable.
5. Specify a parallel-route network with host-specific residence times. Derive
   its coefficient and the newcomer-benefit quadratic criterion, including
   different variances, durations and admissible correlations. Compare with
   series at a specified mean generation time. Report a boundary and points on
   either side. Do not infer a universal finite-noise rule from a quadratic.
6. Specify optional steady-host and bypass networks. Compare optional hosts at
   equal deterministic R0, and separately at fixed other rates. Locate the
   survival-cost boundary and evaluate both sides. For bypasses specify mean
   yield and distinguish allocation fraction from reproductive route share.
7. Test repeated use of one host and independent hosts in sequence, including
   fixed generation time, memory, and sigma = 0.3 and 1.0. Compare joint losses
   with sums of one-driver losses using paired independent-replicate errors.
   Additivity at second order is not presumed exact at finite noise.
8. Check the scope of H4 from the supplied note. Give a proof for any extension
   claimed and a counterexample outside its assumptions if one is found. For
   gain-only lognormal rings compare computed losses with sum sigma_k^2/2.

## Numerical design and acceptance criteria

Use Python 3, numpy and scipy, invoked as `python3 -I mechanism_checks.py`.
At most three worker processes. Final full-script wall time must be below
15 minutes. Script output includes parameters, seeds, sample sizes, runtime,
threshold roots, standard errors, differences and diagnostic pass/fail labels.
All numbers quoted in the note or final answer must occur in this output.

Deterministic tests use independent construction of the matrix and its
eigenvalue, or a full-period Floquet product. Root residual tolerance is 1e-10;
closed-form algebra comparisons have absolute tolerance 1e-9. Periodic
second-order checks use central/symmetric amplitudes and extrapolation, with
absolute tolerance 2e-5. Exact finite-amplitude sign examples require a margin
of 1e-8. Periodic environments can be stationary and ergodic under random phase;
they are not asserted to be Gaussian AR(1).

Stochastic checks use the actual matrix cocycle and solve its threshold, with
common random numbers across nearby roots and comparisons. Use at least 12
independent replicates, at least 40,000 retained steps each, and at least 2,000
burn-in steps (increase if mixing requires it). Report replicate standard
errors, and a five-standard-error sign criterion. Unresolved signs are labelled
unresolved. Root interpolation is checked by direct growth residuals and
estimated derivative; its tolerance is 2e-5 in log multiplier or 0.2 Monte Carlo
standard errors, whichever is larger. Small-noise coefficients are additionally
checked at a small amplitude with paired comparisons; allow 5 SE plus a
declared order-amplitude remainder. No finite simulation is called an exact
real number: “exact model simulation” means no small-noise approximation.

For finite-noise boundary finding, common-random-number interpolation or root
solving is permitted; report its Monte Carlo uncertainty and verify separated
points on both sides. If too costly, give the exact boundary as an implicit
Lyapunov/Floquet equation and clearly mark any unevaluated boundary as not done.
Do not advertise an uncomputed boundary as computed.

## Failure handling and interpretation

Keep the first failed harness output as `mechanism_checks_output_attempt1.txt`
(and subsequent numbered attempts if needed). Explain fixes. A false claim is
a result, not a harness failure. Any new test or numerical-design change is
appended here before its first execution. The note supplies proofs for all
statements called proved. Otherwise use “computed pattern” or “conjecture”.
Literature comments, if any, say “recalled, not verified”. End the note with
“Not done”. No names of people.

## Implementation specification, before the first numerical run

Use the mean-matrix threshold throughout: lambda = R0 = 1. The varied birth
multiplier acts on all returns to the common initial stage. For rings it acts
on the final transfer. Use 12 independent antithetic pairs for Gaussian runs,
40,000 retained steps, 2,000 burn-in, and pathwise tangent Newton iterations.
An antithetic pair shares a root of its average growth. Its two paths are not
treated as independent replicates. Bounded contact runs use 24 independent
paths, paired only to fit the same array layout; SE uses 12 independent pairs.
The bounded multiplier is 1 + sqrt(exp(sigma^2)-1) X, with X = -1/2 or 2,
probabilities 4/5 and 1/5. Transition matrix is phi I + (1-phi) 1 pi.
Use it for nonnegative phi. Its third-order remainder need not vanish.
For alternating contact checks use h = 1 +/- amplitude instead.

Illustrative models: contact stages (5,b), mortality fraction d in {0,0.1,0.8,1},
b in {2,20,40}, phi in {0,0.8}; d=1 is a gain-only limiting comparison, not a
contact process with zero mean transfer. Signed shared-driver checks use a
two-stage ring with stays (2,2), phi in {-0.8,0,0.8}. Series use total mean
generation time 20, with stays (20), (10,10), (20/3,20/3,20/3), and (18,1,1).
Additivity checks compare a joint-noise ring with its separate one-noise rings.
Parallel networks have a central stage of stay 2 and host stages of stays 8
and 1 or 8. The newcomer has share f, noise elasticity 1 or 3, and correlation
0 or 0.8 where admissible. Optional-host checks use ring stays (2,2), f=1/2,
extra stay 8, q in {0.8,0.999,1}. Bypass checks use central stay 2, host stay 4,
allocation f in {0,0.25,0.5,0.75,1}, and relative bypass yield 1 or 1/2.
All route families are checked at sigma = 0.3 and 1; selected algebra checks
use sigma = 0.1. Exact periodic examples use full matrix-product roots.

Curvature is checked by scalar Gaussian quadrature (stay 1/2) for type II,
type III, power, and inverse responses. Quadrature is doubled to check
convergence. The type III finite-noise abundance boundary is solved directly,
and points on each side are evaluated. This scalar check is independent of
the small-noise curvature formula. Additional periodic contact, parking,
correlation, and optional-survival boundaries are solved directly with full
Floquet products. They test the declared periodic environment, not AR(1).

The full script prints all simulation cases, including unresolved tests.
Small-noise Gaussian consistency tolerance is 5 SE + 2 sigma^4; bounded
contact tolerance is 5 SE + 2 sigma^3. These are screening tolerances, not
proved remainder constants. A mismatch triggers investigation, not deletion.

## Amendment after harness attempt 1

The first run stopped because the type III root was not bracketed. With scalar
stay 1/2 its small-noise advantage is never positive at positive abundance;
the proposed finite boundary did not exist in that example. The retained output
is `mechanism_checks_output_attempt1.txt`. To test a nontrivial type III boundary,
change the scalar stay to 3/4 and use its correct coefficient 1/8. Preserve the
failed example as evidence that a boundary cannot be assumed to exist. The
period-20 contact comparison also had no mortality boundary. Add a period-100
square-wave check, labelled an additional exploratory spectrum, before running
it. Neither periodic environment is a surrogate for AR(1).

## Additional exact boundary checks specified during the simulation run

Add scalar Gaussian quadrature benchmarks for parallel alternatives with equal
residence (stay 3/4). The old host has elasticity 1; the newcomer has elasticity
3, shares f, and correlation 0. Solve D(f)=D(0) at sigma 0.3 and 1, checking
both sides and doubling quadrature order. This is an exact scalar benchmark,
not the unequal-residence network. Add a scalar bypass with stay 1/2 and
allocation f=1/2. Solve the relative-yield boundary at which log mean yield
plus loss saving vanishes, with both sides evaluated. Add analytic renewal
forms for parallel, optional and bypass networks and compare them with the
general resolvent and the already specified direct simulations. For series,
solve the second-order boundary between (10,10) and (a,(20-a)/2,(20-a)/2)
 at phi=0.9; the existing symmetric and (18,1,1) simulations test either side.
These are distinct local and finite-noise boundaries; do not conflate them.

The completed first stochastic run is preserved as
`mechanism_checks_output_pre_extension.txt`. It took less than the runtime
limit and had no failed numerical checks. Its phi=0.8 repartition did not lower
the loss. The additional series-boundary specification above originally said
phi=0.8; that was an incorrect anticipation of a crossing. Before computing
the new boundary, change it to phi=0.9, and add series simulations at phi=0.9
and 0.98. This is a declared adaptive search, not a prior prediction. Improve
paired additivity precision by retaining each stage's driver coordinate in
single-driver cases, with zero contributions for the other stages. This does
not change their marginal laws. Print all outcomes, including failures to
resolve exact additivity. Add the local phi boundary for the (18,1,1) versus
(10,10) comparison and evaluate both sides algebraically.

## Precision extension before the final run

The extended run left weak-noise additivity unresolved and gave a marginal
sign for the strong-noise, phi=0.98 repartition. Increase retained steps to
400,000 for the (18,1,1) joint and three single-driver cases at sigma=0.3,
phi=0 and 0.8, and for the two joint rings (10,10), (18,1,1) at sigma=1,
phi=0.98. Keep 12 independent pairs and the same five-SE rule. This is a
precision increase after an unresolved check, not a changed sign criterion.
The final output will identify the longer runs. Add an automated H4 comparison
for every positive lognormal contribution in the existing Gaussian models,
using the birth-frequency-weighted bound already proved in the note. Zero
routes receive a numerical tolerance of 1e-10. Preserve the preceding complete
output as `mechanism_checks_output_pre_precision.txt`.

## Finite AR cancellation and cost boundaries

Add a direct zero-growth solve for the memory coefficient in the two-stage
ring with stays (2,2), for a single noisy transfer and for two opposite-sign
transfers, at sigma=0.3 and 1. These compare with baseline (ii). Use 200,000
steps, 2,000 burn-in, 12 independent replicate environments, and propagate
the derivative through both the AR recursion and the normalized population.
Initialize the memory roots at -0.5 for gain-only and -0.1 for opposite signs.
Use a safeguarded Newton iteration inside (-0.97,0.95). Report replicate
root SEs; require last root correction below 2e-5. Independently evaluate mean
growth at the estimated root plus and minus 0.02, with the five-SE sign rule.
This computes finite-noise boundaries, rather than labeling the second-order
ones as exact. It does not prove uniqueness of all possible memory roots.
Before Newton solving, test the two allowed endpoint memories. If their mean
growths do not bracket zero, report their growths and SEs and do not manufacture
a boundary. This matters because the existing sigma=1, gain-only phi=-0.8
case has positive loss despite the negative second-order coefficient.
This independent block can be run with `--memory-boundaries-only` and appended
to the same output; the default final script runs it as well. Record both
component runtimes if they were executed separately.

Also check the unequal-memory parallel formula with central stay 2, host
stays 8 and 1, share 0.4, unit elasticities, log-host correlation 0.4, and
memories 0.2 and 0.8. Run direct products at sigma 0.1, 0.3 and 1 with the
original 40,000-step design. Verify innovation covariance admissibility and
compare the explicit unequal-memory expression with the general inverse.

For raw reciprocal responses H and 1/H against baseline (i), add an independent
check of the rank-one identity in the symmetric two-stage ring with stay 1/2:
r = E log cosh(sigma (z_t-z_(t-1))/2). Check Gaussian quadrature against actual
matrix products at sigma=0.3 and 1 and phi=0 and 0.8, using the same 200,000-step
design as the finite-memory boundary block. This is a growth identity, not an
identification of growth times generation length with a threshold loss.

The finite gain-only root at sigma=0.3 was found, but its immediately adjacent
plus/minus 0.02 growth checks were unresolved at five SE. Keep them. Add two
wider, fixed side points phi=-0.65 and -0.45 with the same 200,000-step design.
They are examples on either side of the estimated root, not a tighter boundary
estimate. Run this independent block with `--gain-sides-only` if needed; the
default script also includes it. Add its elapsed time to the recorded total.
