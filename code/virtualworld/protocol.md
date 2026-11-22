# vworld-1: registered protocol

Registered 2026-11-15 08:28:30 CST (shell `date`), before model execution,
new numerical results, or new summaries of the supplied empirical metrics.
Phase 1 only. This file is frozen; corrections go in dated addenda.

## Questions and hypotheses

Q1: Which predictions survive demographic noise, nongeometric development,
host feedback, contact saturation, delaying scarcity, mortality, alternative
hosts and dispersal? Q2: What effects and replication does a controlled
factorial experiment have? Q3: Can a virtual ecologist recover the effect
after sparse, noisy field-style observation? Q4: What changes when actual
host log-abundance trajectories drive the world? Q3 and Q4, and the full
realism grid of Q1, are planned here, not run in phase 1.

Hypotheses (basis is the supplied notes, read locally; no novelty claim):

| ID | Hypothesis | Status and scope |
|---|---|---|
| H1 | Small-noise loss per circuit is sigma² K, with K between 1/(2T) and (1+2theta)/(2T) | Check: primitive geometric ring, one fluctuating transfer, wasting scarcity, phi>=0; general-ring-v2 B1 and corrected I.2 |
| H2 | Gamma-host threshold loss strictly decreases with steady staying probability s | Check: Roerdink recurrence, independent gamma hosts, fluctuating stage exactly one step; exact-limits dated addition and exact-gamma-parking section 8 |
| H3 | Positive log-host AR memory increases the two-stage small-noise coefficient; extending its steady stage decreases it when the exposed stay is one | Check: memA coefficient; finite-noise decrease is a numerical memB check, not a general theorem |
| H4 | Demographic extinction agrees with its specified multitype branching law under constant hosts | Check: conditional independent individuals, no density feedback |
| H5 | Longer steady stays reduce extinction under the pilot law | Open: D does not order extinction across demographic laws (exact-limits R18 correction); both calendar time and generation count matter |
| H6 | Mortality, delaying scarcity, regular stays, correlated alternative hosts and dispersal can change the size or direction of protection | Open beyond the separately checked limiting cases; exact-limits H1–H4 and R16 |
| H7 | Sparse sampling, observation errors, unknown kernels and feedback obscure the controlled effect | Open; virtual-ecologist question, motivated by the Refereed section of decisive-experiment-v2, not a claim established by it |
| H8 | Driving with actual series differs from an AR surrogate matched only in variance and lag-one memory | Open; higher lags, irregular spacing, nonstationarity and tails need separate checks |

Here sigma is log-host standard deviation; phi is log-host AR(1) memory,
theta=phi/(1-phi), T is total reference residence time. For one noisy
transfer K=(1+2theta)/(2T)-R/(1-R), where
R=product[phi/(tau_i+(1-tau_i)phi)] and tau_i are geometric means.
At phi=0 the return term is zero. The all-unit periodic ring is excluded.
Neither a finite-noise bound nor an extinction theorem follows from this band.

## Model and estimands

Use a discrete-time individual branching world with 2–4 ordered biological
stages. Exact multinomial/binomial sampling groups exchangeable individuals;
Poisson offspring sums are exact conditional on the numbers leaving. This is
an individual-based model, not a continuous-time approximation. Each individual
chooses one continuation, development, transfer or death event in a step.
Stage durations: geometric; sums of geometric phases; gamma rounded to time
steps and numerically mean-matched; or an empirical integer-duration mass
function. Gamma tails are truncated below 1e-12 probability and disclosed.
Independent per-step death precedes development in wasting stages. All
transfers can have Poisson offspring, or deterministic single continuation;
the latter requires yield one. Offspring then move independently among
patches. No shared host-death packets, evolution or parasite regulation are
assumed. Stage host labels specify where burdens enter feedback.

Forced hosts: H=Hbar exp(sigma z-sigma²/2), stationary unit Gaussian z with
AR parameter phi. Different hosts/patches are independent unless synchrony
is explicitly requested. Thus the stationary arithmetic mean is fixed;
individual finite schedules are NOT rescaled to a fixed sample mean. Record
each schedule's realized mean and log variance. Correlation of H is not phi.
Constant hosts are the sigma=0 case. Validation also uses independent gamma
and bounded Markov hosts with their actual marginal laws.

Endogenous option: positive Ricker host abundance, environmental log noise,
carrying capacity and parasite-induced host mortality proportional to burden
per host. This is an illustrative abundance process, not fitted to data.
Its uninfected and rare-parasite counterparts must be run separately in phase 2.
Mass-action response is H/Hbar; saturation is 2h/(1+h), normalized at h=1,
not silently centered to mean rate. Wasting scarcity multiplies offspring
yield. Delaying scarcity changes the exit probability: a geometric ready
stage with independent death and contact hazard, giving valid probabilities
even for unbounded hosts. A separate bounded linear-contact compatibility
mode reproduces R16's mutually exclusive contact/death convention and refuses
invalid probabilities. Alternative hosts are mutually exclusive weighted
routes at one step; dispersal follows demography on an island graph.

Outputs: first extinction time, survival/right censoring at horizon, extinction
fraction, conditional-on-environment linear invasion growth, and threshold
D. D is the common log offspring multiplier restoring zero long-run invasion
growth minus that for the arithmetic-mean-rate operator. It is found from
growth roots, never identified with a finite endpoint or with T times an
arbitrary growth difference. Only one designated circuit-completion transfer
receives the tilt. For saturation and feedback report separately the reference
at mean host, the averaged-rate reference, and the dynamic residual. The
phase-1 mass-action forced arm makes the first two references coincide.
Finite individual log growth is undefined after extinction: retain zeros,
report extinction separately, and label survivor-only growth if displayed.
The primary rare-growth observable uses the conditional expectation operator
for all schedules and has no survivor selection.

## Validation gates (fixed tolerances)

Run unit/structural tests, then gates i–v in order. Independent algebra or
numerical implementations supply references. Do not import the reference
engine as the world implementation. Archive every gate as it finishes.

1. **i, coefficient and band:** stays (2,5), (1,2,7), (2,3,4,5), each at
   phi=0 and 0.6, sigma=0.03. Compare the direct matrix resolvent coefficient
   to the ring formula: absolute error <=1e-10; band slack >=-1e-10.
   Simulate the linear conditional-mean limit for 65,536 recorded steps,
   burn 2,048, 32 independent antithetic pairs. Use the known mean-zero
   control variate (h-1)/T. Compare -T*r/sigma² to K:
   absolute error <=0.05*K+4*SE; require SE<=0.03*K. Also check forward
   individual one-step means against the same matrices (4 SE + 0.002).
2. **ii, six R16 cells:** IDs 0c-00000, 0c-01000, 0c-00100, 0c-00010,
   0c-00001, 0f-10000. These respectively cover base, shape, alternatives,
   patches, mortality and delaying contact. Read parameters and archived D
   from R16_results.csv. Compile them independently; compare matrices at
   bounded host states to the reference engine to 1e-11. Use fresh seeds,
   32 independent paths, 32,768 recorded steps and burn 2,048 (both times
   multiplied by four for the fine cell). Require |D-D_archive| <=
   4*sqrt(SE²+SE_archive²)+0.002 and SE<=max(0.003,0.05*abs(D)).
   Root correction <=1e-7. Compare the full and last-half estimating
   equations: root-equivalent difference <=4 paired SE +0.002. These
   allowances cover sampling and a stated numerical floor, not certified bias.
3. **iii, exact gamma case:** shape p=2,1,0.25 (host variance 0.5,1,4),
   staying s=0.2,0.5,0.8,0.95. Independently integrate the invariant
   density in log coordinates and solve its zero-growth root; double
   integration range and tighten quadrature tolerances, root change <=1e-8.
   All nine adjacent exact-root decreases must exceed 1e-8. Compare model
   linear products (65,536 steps, burn 2,048, 64 independent gamma paths,
   common hosts across s) to quadrature: error <=4 SE+0.0005; sampled
   adjacent decreases must exceed four paired SE. Root correction <=1e-7.
4. **iv, memory:** lognormal sigma=0.3,0.6; phi=0.6,0.9; exposed stay one,
   steady stays 2,5,10. Use 65,536 steps, burn 2,048, 32 antithetic pairs,
   common Gaussian innovations across stays and phi. All eight adjacent
   decreases must exceed four paired SE. Compare the 12 roots to matching
   archived memB rows within 4 combined SE +0.001. Independently verify
   memA's coefficient against the ring coefficient to 1e-10. Do not claim
   global finite-noise monotonicity from these twelve cells.
5. **v, demographic persistence:** constant hosts, stays (2,5), Poisson
   transfers, independent initial ten individuals in exposed stage,
   log R0=-0.1,0,0.1. Run 10,000 independent trajectories per setting to
   step 120; compare extinction at steps 20,60,120 to a separately written
   multitype probability-generating-function recursion. Error <=
   4*sqrt(p*(1-p)/10000)+0.002 in all nine comparisons. This predicts the
   finite-horizon persistence curve and handles demographic variance exactly.

Seeds: validation base 151100; pilot base 152000; deterministic offsets
recorded in code and outputs. Antithetic pairs are the sampling units, not
their members. All confidence intervals are pointwise unless stated otherwise.
No tolerance is changed after seeing a result. A failed scientific gate stops
this job before the pilot; preserve its output and explain failure. Software
unit-test corrections before gate execution are recorded and not hidden.

## Registered pilot

Exactly 18 cells: sigma={0,0.3,0.6}, phi={0,0.6}, steady mean={1,5,20}.
200 independent complete host/demographic replicates per cell (3,600 total).
Two geometric stages; exposed stay 2 (keeps the steady-stay-one endpoint
primitive), mass action, wasting scarcity, independent Poisson transfers,
no stage death, patches or alternatives. One host affects exposed-to-steady
transfer. Final transfer is the designated birth tilt. Set log R0=0.05 in
every cell; start 20 individuals allocated by the constant-environment stable
stage proportions, with deterministic rounding. Horizon 500 steps; this is
a calendar-time comparison, not matched generations. No population cap or
replacement of extinct trajectories; integer overflow is a run failure.
The zero-noise phi labels are duplicate negative controls with independent
demography; do not pool them without saying so.

Each replicate also carries the conditional-mean invasion trajectory. Report
its log growth over steps 0–100 (primary early growth) and 0–500. The individual
endpoint and first extinction time are archived. To compare with theory,
estimate D in each cell on an additional 32 independent antithetic pairs,
32,768 steps and burn 2,048, with a root correction <=1e-7. Report second-order
sigma² K beside D; finite-noise discrepancies are results, not gate failures.

Per cell: Wilson 95% extinction intervals and t intervals for mean early
growth. Main registered contrast: the noise-by-stay interaction at phi=0.6,
[Pext(sigma=.6,stay=1)-Pext(sigma=0,stay=1)] minus
[Pext(sigma=.6,stay=20)-Pext(sigma=0,stay=20)]. Positive means buffering of
the noise-associated extinction increment. Report its independent-cell Wald
interval, plus a prespecified independent Jeffreys beta posterior simulation
interval (100,000 draws) to disclose boundary sensitivity. Report the analogous
early-growth interaction with signs oriented to loss, and simple noise and
stay contrasts. No data-dependent selection of the most favorable contrast.

80% power: equal allocation, two-sided alpha=.05, primary four-cell
interaction, planning n=ceil((z_.975+z_.8)^2*sum[p_i(1-p_i)]/delta²).
Also report n for the continuous-growth interaction using the sum of its
four variances. Draw 10,000 independent Jeffreys beta parameter sets for
binary planning sensitivity and report 10th–90th percentile n. If delta's
interval contains zero, state that a finite upper replication requirement is
not established. These are conditional model-based normal approximations,
not executable real-experiment recommendations or measurements of power.
A real trial also needs nuisance calibration, observation error and independent
biological blocks, as required by decisive-experiment-v2's Refereed section.

## Empirical ranges and phase-2 plan (not fitted)

The supplied LPD summary already reports 1,842 series in 187 study clusters.
Its sigma 10/25/50/75/90 percentiles are 0.156/0.330/0.581/0.907/1.238;
raw r1 are -0.133/0.053/0.281/0.492/0.673; bias-corrected r1 are
-0.054/0.139/0.386/0.631/0.876. These are read values, not new computations.
The pilot's 0.3 and 0.6 span approximately the lower quartile and median;
phi=0.6 probes substantial positive memory. Phase 2 adds sigma 0.15,0.9,1.25
and phi -0.1,0.3,0.9 for tails and boundary cases. r1_bc is a scenario proxy
for phi, not proof of an AR(1) law. Extract metrics n, sampling interval,
coverage, sigma, r1/r1_bc, acf_1 through acf_10, trend and study_id before
designing series blocks. Preserve annual/subannual units; do not equate an
annual host lag to a parasite day. Data determines ranges and checks patterns,
never parameters fitted to parasite persistence or a mechanism chosen to fit it.

Phase 2 proposal, requiring a new protocol: forced core 2–4 stages x
6 sigma x 5 phi x 3 steady means =270 cells (500 trajectories each).
One-at-a-time realism screens at 18 pilot cells: gamma shapes 1/4/16,
empirical kernels once supplied, death 0/.005/.02 per step, saturation,
delay, 1/4 patches with dispersal 0/.05/.2, and 1/2 alternative hosts with
shares .1/.5 and correlation 0/.8. Cap total forced cells at 1,000, then
select interactions by a separately frozen mechanistic rule, not significant
pilot outcomes. These life-history ranges are illustrative, pending the
separate measured stage-duration study; require complete stay/survival kernels,
units, sampling uncertainty and provenance as its input.

Endogenous arm: 108 prespecified Ricker settings crossing r=.2,.8,1.5,
K=1e3,1e4, host noise=0,.3,.6, parasite mortality coefficient=0,.01,
steady means=5,20; 500 replicates and 5,000 steps each, plus parasite-free
counterparts. Recheck invasion linearization before interpreting persistence.
Virtual ecologist: sample archived paths at 1/5/20 steps, detection .2/.8,
and log observation SD 0/.3, retaining nondetections; 100 observation draws
per world, no new population simulation. Compare measured versus latent
contrasts and bias/coverage rather than refitting a mechanism.
Real-series driving: up to 60 series sampled by study cluster and interval
strata before outcome computation, preserving dates and gaps; compare actual
paths, shuffled order, and AR surrogates. Normalize only the input's declared
arithmetic baseline and disclose finite record length. No circular wrap or
unregistered interpolation. Actual trajectories are not supplied in the
metrics CSV; acquisition/licensing and time-scale alignment are phase-2 gates.

## Budget and stop

Phase 1: one process, at most 2 GiB working arrays, at most 45 minutes total
scientific compute (validation <=30, pilot <=10, summaries <=5); no installs
unless essential (numpy/scipy plus standard library suffice). Record wall
times. Only the registered validation and one pilot; no tuning biological
parameters, extending a failed gate, or starting the full grid. Stop at the
first failed gate, missing essential reference, invalid probability, overflow,
or compute limit. Write validation.md, pilot.md (explicitly not run if gated),
results and report.md regardless. Phase-2 provisional ceiling: 24 single-core
hours, 8 GiB RAM, 10 GiB outputs, maximum 500,000 forced and 54,000 endogenous
trajectories, up to 30,000 real-series trajectories; timing feasibility must
be checked before approval. This is a budget, not measured runtime.

## 2026-11-15 addendum — vworld-fix1 corrections and phase-2 registration

Registered before fix1 scientific runs. This dated addendum supersedes the
specific statements below; all original text and phase-1 results are retained.
The executable-run protocol is ../protocols/vworld-fix1.md (internal record; not distributed).

1. Initial allocation uses largest remainders after flooring total times
   normalized stable proportions, with ties by state index. At log R0=.05
   the three pilot allocations are [13,7], [6,14], [2,18]. The original
   modal-stage rule was biased. The original pilot remains a historical run;
   `pilot-rerun-fix1.md` is the corrected sensitivity with the same grid/seeds.
2. Remove the 1e15 and int64 population limits. The sampler stores Python
   integers where needed and sums independent binomial/Poisson chunks no
   larger than 2**50 trials/rate. This preserves the count law by additivity;
   no large-population approximation or population cap is introduced. Floating
   probabilities and numpy sampling retain ordinary numerical limitations.
   For large Poisson products, the binary64 per-parent rate is split using its
   exact rational representation; only the final remainder rate is rounded
   (relative error <=2**-52 for normal values). Full chunks are exactly
   represented. Histories and reductions also use Python integers before
   optional safe int64 conversion; NPZ exports use decimal strings if needed.
   Work is proportional to count/rate divided by chunk size, so extreme
   supercritical cells may exhaust the declared compute budget; report such
   a cell incomplete rather than censored or extinct. Nonfinite model rates
   still stop a run. There is no new ecological carrying capacity.
3. Movement correction to the Model and estimands paragraph: after demography,
   **all individuals ending in the mobile stage disperse independently**,
   including continuing residents, developing individuals, and offspring.
   Movement preserves residence age/phase. The destination probabilities are
   W=(1-m)I+m*11'/P across P patches. Other stages do not move. This aligns
   the wording with the existing operator; the implemented biological law
   has not changed.
4. At steady mean 1, any positive-integer residence is identically one step.
   `Duration.gamma(1, shape)` now denotes that zero-scale limiting kernel for
   every positive finite shape, with zero tail; it is not a positive-scale
   gamma distribution. Phase-2 shapes {1,4,16} apply to means 5 and 20; at
   mean 1 retain one shared degenerate control per sigma/phi, not three shape
   treatments. This is 12*3+6=42 distinct screen settings across the 18 pilot
   labels (before overlapping controls are deduplicated with the core).
5. A control variate must be centered on the estimand's host expectation.
   `products` and `threshold` default to finite-record centering: subtract
   each record's measured-window mean, with the last half centered on its
   own window. The summed correction is exactly zero, implemented by retaining
   raw growth. This gives no variance reduction for a fixed record but cannot
   shift growth, the root, or the half-run diagnostic. For stationary generated
   hosts, explicitly pass `control_mean=1` (or another known normalized host
   expectation). An externally chosen abundance baseline is not that known
   expectation. Never assume externally normalized records have mean one;
   do not rescale their supplied hosts. Non-unit affine host means must also
   use an explicit averaged reference if D is to be relative to mean rates.
6. All normal 95% intervals use the shared Z95=norm.ppf(.975); t intervals
   retain their stated degrees of freedom. The Ricker grid has
   3*2*3*2*2=72 infected settings, not 108. As noted in audit_notes.md,
   36 parasite-free counterpart blocks yield 108 total blocks and 54,000
   trajectories at 500/block; the biological host control law has only
   18 distinct (r,K,noise) settings, duplicated by steady-stay label.

Audit 2(a) changes phase-2 extinction design. A baseline subtraction removes
only the constant-host truncation difference; it does not match exposure to
noisy generations. The primary extinction outcomes are at matched reference
G={23,50,100}, where T_gen=2+steady_mean and horizon=G*T_gen steps. These are
reference circuit generations, not realized lineage-generation counts under
noise or altered life histories. For later stage structures use the sum of
reference stage means, declaring any noninteger time rounding before running.
Keep extinction at 500 steps as a secondary calendar-time outcome. At stays
1 and 20, the primary horizons are (69,506), (150,1100), and (300,2200).
G=50 is the prespecified single confirmatory contrast; G=23 and 100 are primary
time-course estimates with pointwise intervals. Reporting three estimates
is not a familywise significance test.

Compute planning power now from 400 independent trajectories in each of the
four main cells (sigma {0,.6}, stay {1,20}, phi=.6, log R0=.05), retaining the
same trajectories across horizons. For each G use the four-cell interaction
and equal-allocation normal formula from the original pilot at two-sided
alpha=.05, 80% target power; report Jeffreys-parameter 10th–90th percentile
planning sensitivity. Report n/cell for each G and their maximum as the
planning allocation for marginal 80% power at all three; it is not joint
power. No use of the inflated fixed-step effect for this primary planning.
The computed estimates will be added in a separate dated results note.

Audit 2(b): the extinction interaction is regime-dependent and non-monotone
across ln R0; the supplied audit's .05 and .2 comparisons do not map the whole
curve. Register ln R0={.02,.05,.1,.2,.4} for phase 2, spanning near-critical to
supercritical regimes. Do not promote a favorable pilot contrast to a universal
buffering law. Cross this axis with the two-stage forced core (6 sigma * 5 phi
* 3 steady means * 5 ln R0=450 settings); retain the three- and four-stage
core at ln R0=.05 (180 settings), giving 630 core settings. Reserve up to
370 of the existing 1,000-cell cap for the predeclared screens. This replaces
the older 270-cell core; a full 1,350-cell cross would exceed the existing
budget and is not registered. Phase-2 implementation, matched-time host
correlation interpretation, timing feasibility, and realism-screen selection
still require the separate frozen phase-2 protocol and human gate. No phase-2
campaign is run in fix1.

## 2026-11-15 addendum — fix1 observed power and completion record

The registered sensitivity is complete: 18 cells * 200 trajectories, corrected
allocations, 500 steps. Its main interaction is 0.435 [0.32085,0.54915] (95%
Wald), compared with the frozen pilot's 0.430. See pilot-rerun-fix1.md. These
are calendar-time results and do not set the primary phase-2 replication.

The matched-generation run has 400 independent replicates per main cell,
1,600 total. At G=23,50,100, the respective interactions are
0.3325 [0.2446,0.4204], 0.3275 [0.2383,0.4167], and
0.3425 [0.2571,0.4279] (pointwise 95% Wald). The two-sided alpha=.05,
80% normal planning estimates are **58, 61, and 51 replicates per cell**.
Their Jeffreys-parameter 10th–90th percentile planning ranges are 41–85,
44–91, and 38–73/cell. Use 61/cell as the maximum point estimate for marginal
80% power across these three outcomes, conditional on this model and these
observed effects; it does not guarantee joint power, familywise error control,
or achieved power. G=50 remains the prespecified confirmatory horizon.
These estimates are recomputed from the matched data, not scaled from the
500-step contrast. See matched-generations-fix1.md and fix1-report.md.

Implementation departure: the first matched runner stopped the stay-one paths
at step 300 (G=100) and incorrectly treated their sentinel 301 as extinction
by the secondary step 500. Its secondary row was withdrawn in a dated note.
The three primary outcomes were unaffected. Before completing the secondary
horizon, an observation-horizon guard and a regression test were added; all
22 tests passed. The same two seeded stay-one cells were replayed through
step 500. Host values and count totals through step 300 matched bit for bit;
no new independent replicates were added. Corrected secondary outputs are
separately archived in results/fix1/matched-secondary-completion/ and described
in matched-secondary-completion-fix1.md. No acceptance tolerance was changed.
The original incomplete output is preserved for audit. No phase-2 campaign
or extra ln R0 sweep was run.
