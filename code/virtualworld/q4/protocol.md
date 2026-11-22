# Q4-1: registered real-trajectory forcing experiment

Registered 2026-11-15 09:55 CST (shell `date`; minute precision), before any
new model execution, data selection, or empirical outcome computation.
The task explicitly authorizes this Q4 run. The earlier full phase-2 campaign
is not run. This protocol is frozen; changes require dated additions.
The requested q4/ location takes precedence over the general protocols/
convention. Work follows study rules rules 7, 8, 10, 13 and the read
`[local working directory] This repository copy
has no board or plan.md; A6 is defined by the task (stationarity and AR(1)).
No sources/ files will be opened. No biological parameter is fitted.

## Question and hypotheses

Does the extra log reproduction needed for invasion, D, agree with a
second-order prediction under observed host trajectories? Do longer steady
stays buffer it? These are numerical tests, not novelty claims or propositions.

* H1 (open finite-amplitude check): D agrees with the prediction using each
  series' own log variance and lag-one correlation (AR(1)), and with the
  prediction using its full circular log autocovariance. Report these two
  checks separately. Agreement tolerance: max(0.01, 0.20*abs(prediction)) in
  log reproduction per circuit, per cell. Zero-containing records have no
  finite log variance: compute their world outcomes but mark H1 inapplicable,
  never insert pseudocounts or count them as agreement.
* H2 (open): for each record and stage count, D decreases at both adjacent
  stays 1→5 and 5→20. A difference >1e-6 supports strict decrease; abs(delta)
  <=1e-6 is unresolved/flat; delta < -1e-6 is a reversal. A single reversal
  defeats the universal wording; report counts rather than majority proof.
* H3 (open): fluctuating minus constant habitat treatment differences in
  mean D agree with differences in predicted D, within
  max(0.01, 0.20*abs(predicted contrast)). Report both prediction versions
  and directions. Equal *flour* means do not imply equal beetle means;
  normalization imposes equal host means in our world.

The basis is the supplied phase-1 protocol, all its addenda, fix1-report.md,
README.md and the inspected world code, plus leads.md Lead 4. Phase 1 proved
neither finite-amplitude H1 nor universal H2, especially at negative memory.

## Drivers and frozen sampling

1. Download Zenodo 7972066 and 4976742 metadata and files with a generic
   User-Agent, without credentials. Preserve URLs, checksums and README
   interpretation. Store downloaded/raw working material only in ignored
   q4/raw/. Primary Tribolium hosts are larvae and adults, separately, from
   Data Sets 7, 8 and 10 (constant versus periodic habitat). For 7 and 10 use
   weeks >=18; for 8 use all available observations. Require regular biweekly
   spacing, at least 16 observations and positive arithmetic mean. Include
   zeros exactly. Keep initial-condition blocks in set 8. Archive overlap
   audits; shared controls are not independent new experimental replicates.
   The 2018 synchrony record has no matched constant-habitat contrast and
   is provenance/context only, not an additional primary campaign.
2. The open compilation sampling frame is the intersection of the supplied
   candidate JSON and series_metrics.csv. Require finite, nonnegative values,
   positive mean, at least 16 observations, and all adjacent `steps` exactly
   one apart. Do not interpolate gaps or compress time. Require compatible
   open licence per supplied metadata. Include zero-containing series.
   Sample exactly min(120, eligible count) without replacement, seed 415111,
   stratified by taxon_group × interval_class. Allocate one per nonempty
   stratum, then distribute remaining slots proportionally to remaining
   capacities by largest remainders (lexicographic tie break). Sort series
   IDs within each stratum before RNG sampling. Save selection, exclusions,
   and original supplied metrics before outcomes. No replacement by outcome.

One model step equals one observed sampling interval, not one day. Parasite
stays are generic multiples of that interval and are not measured life
histories. Divide every complete input record by its arithmetic mean. For
positive records compute z=log(H)-mean(log(H)) without detrending. Population
variance uses divisor n. Recompute circular autocovariances for this exact
driver; original supplied detrended metrics are descriptive provenance only.

Extend short records by **circular replay**. This explicitly supersedes the
earlier phase-1 prohibition for this Q4 registration. Repetition preserves
each finite record's marginal distribution and circular autocorrelation,
but imposes periodicity and a wrap boundary. Interpret D as the exact
periodic-environment threshold, not a consistent estimate of the unknown
wild host process. Randomize initial phase for persistence. Quantify wrap
jump and linear trend and test a mirrored replay for departures.

## Parasites, loss and persistence

Six cells per record: geometric stage means [2,s] and [2,2,s], s={1,5,20}.
These primitive rings keep the exposed stay 2 as in the corrected pilot;
the added stage has fixed stay 2. One fluctuating transfer, mass action,
wasting scarcity, Poisson offspring at exits, no feedback, mortality,
alternative hosts or dispersal. Use world.model.ring unchanged.

D is the log birth tilt that makes the dominant growth rate zero, minus
the constant mean-one critical tilt. Compute a normalized one-period product
of the engine's matrices and its Perron eigenvalue, then bracket/solve the
root. This is an infinite circular replay calculation, not an endpoint loss.
Require root-equivalent residual <=1e-7 and constant-host root abs(D)<=1e-10.
Record all D, predictions, residuals and invasion at log R0=.05. When using
engine products/threshold, use control=1/T and control_mean=None: the fix1
finite-window centering has identically zero summed adjustment. Never use
known-mean stationary variance reduction on these recorded windows.

Run engine simulate at log R0=.05, initial 20 individuals allocated by fix1
largest remainders from the constant stable proportions. Use 64 independent
phase/demographic draws per record/cell; phase sampled uniformly from the
finite record, shared across stays and stage counts. Demographic seeds
415200+1000*stage_count+10*series_index+stay_index. Phase seed
415150+series_index. Horizons G={23,50,100} reference generations,
T=sum(stage means); G=50 primary. Also report 500-step survival. Run through
max(500,100*T); report survival fractions and Wilson 95% intervals. These
are conditional replay Monte Carlo intervals, not uncertainty in field data.
Retain extinction times, censoring and phase offsets. No population cap.

## Predictions and attribution diagnostics

AR prediction is sigma²*world.invasion.ring_coefficient(means,phi), using
phi=circular log covariance at lag 1 / sigma², including negative phi.
Handle a constant record as zero; at |phi|=1 label AR outside its stationary
domain. Full-covariance prediction uses the same matrix perturbation formula
expanded as an impulse sum: with critical mean matrix A, left/right Perron
vectors w,v normalized wv=1, Q=I-vw, noisy-transfer matrix B, c=wBv and birth
sensitivity b, D2=[c² C(0)/2-sum(k>=1) C(k) wB(AQ)^(k-1)QBv]/b.
Here C is the circular autocovariance of centered log abundance. Sum 4096
lags and verify the kernel tail is <1e-12. This is a computational extension
of the supplied resolvent, checked below, not an independently reviewed new
theorem. Do not claim the phase-1 nonnegative-phi band at negative memory.

Compute diagnostic descriptors before interpreting departures: log skew,
log excess kurtosis, raw skew/kurtosis, largest 10% contribution to raw
variance, dominant Fourier power fraction/period, linear trend R², wrap
jump divided by log SD, and RMS discrepancy of full ACF from phi^k over
lags 1..min(10,n/2). Flag abs(log skew)>1, excess kurtosis>3, top-decile raw
variance share>.75, dominant power>.40, trend R²>.25, wrap jump>3 SD,
ACF discrepancy>.20. These thresholds classify features, not significance.

For **every** positive-record H1 departure or H2 reversal, evaluate all six
cells of these prespecified diagnostics, seed 415500+series_index:
* small amplitude: exp(0.1*z), re-normalized; prediction scales by .01;
* eight Fourier phase randomizations of z, each preserving its exact log
  periodogram/circular covariance, re-normalized after exponentiation;
* eight random permutations of original abundances (marginal preserved);
* remove the least-squares linear trend of z, then re-normalize;
* clip z to its empirical 5th/95th percentiles, then re-normalize;
* mirrored replay [H,reverse(H)] to remove the endpoint jump.

Recompute predictions for transformed series. AR failure repaired by full
ACF identifies non-AR memory at second order. Full-ACF failure that contracts
at small amplitude identifies finite-amplitude terms. Phase changes at fixed
log covariance establish sensitivity beyond second order; they do not alone
distinguish skew from higher-order temporal dependence. Permutations test
ordering, detrending tests trend sensitivity, clipping tests tail sensitivity,
mirroring tests boundary sensitivity. These interventions can alter multiple
features, so label mixed/unresolved cases explicitly; no unsupported exclusive
causal labels. Diagnostic material change: >max(.005,.20*abs(original D)).
For small amplitude require abs(Dsmall/.01-D2)<=max(.001,.05*abs(D2)); a
failure is disclosed as unresolved nonlinear/implementation scope, not ignored.
Zero-containing departures are support violations of the finite-log theory;
permutation and mirror diagnostics still apply, log transforms do not.

## Validation before new outcomes

First reproduce phase-1 gate i, stays [2,5], sigma=.03, phi=.6, seed 151120,
65,536 recorded steps, burn 2048, 32 antithetic pairs, known control_mean=1.
Require |estimated K-K|<=.05*K+4*SE and SE<=.03*K; ring versus matrix
resolvent absolute error <=1e-10. Archive before any real-series run.
Run the supplied 22-test suite without editing world/. Then validate new
periodic code against engine products on a synthetic positive period at
small amplitude and constant hosts (root tolerance 1e-7 after burn), its
covariance kernel against the AR resolvent at phi=-.5,0,.6,.9 (1e-10),
and symmetric +/- small-amplitude root differences against D2 (1e-5).
Require all checks before real-data outcomes. Finite-record centered and raw
engine products must agree to 1e-12. These are independent numerical
implementations, not an independent second-party review.

## H3 comparisons, budget and stop

Contrast fluctuating against constant within data set and host stage, and
within initial-condition block in set 8, keeping amplitude treatments separate.
Equal weight per culture. Report measured mean host abundances, normalized
means and imposed habitat means separately. H3 requires finite predictions
for every culture in a contrast; otherwise report it inapplicable with the
zero-containing cultures named. Descriptive contrast counts are not replicated
experiments or a multiplicity-adjusted test. No parasite data are present.

One campaign, <=120 compilation records plus <=72 Tribolium trajectories;
<=73,728 primary demographic paths, <=30 minutes scientific compute,
<=2 GiB RAM and <=1 GiB outputs. Diagnostic roots <=192*6*21; no demographic
diagnostic campaign. Fetch budget <=5 minutes with public mirror fallbacks.
Stop on failed validation, nonfinite/invalid rates, unresolved root, resource
limit, or unavailable required input; preserve partial outputs and explain.
Scientific departures do not trigger retuning or extra unregistered runs.
After the registered diagnostics and one report, stop. Save a manifest with
dates from date, code/input/protocol hashes, software versions and wall time;
one primary run summary, every departure as a CSV row, and q4/report.md with
H1–H3 counts, scope limits and the required final status line. Post a new
board message before finishing. No commit, push or service.

## 2026-11-15 09:54 CST — registration timestamp correction

The opening minute `09:55` was a transcription error: shell date immediately
before writing returned **2026-11-15 09:53:48 CST**. The complete protocol was
written before the validation commands launched at 09:54:09. This correction
does not change a method, tolerance or budget.

## 2026-11-15 09:58 CST — invalid source count, before empirical outcomes

Input validation stopped on Data Set 8, HS=2, IC=2, replicate=5, week=48:
the workbook's adult count is **-183**. The README supplies no encoding rule
for negative counts. Do not change its sign, impute it, drop its time point,
or substitute another culture. Exclude this entire adult trajectory from
the valid-host campaign; keep its larvae. Record it in exclusions. H3 for
that adult treatment/block has only 2 valid fluctuating cultures versus 3
controls and is explicitly a partial-source contrast, not a complete original
experimental contrast. Other cells and frozen tolerances are unchanged.
This resolves a source-validity ambiguity before any empirical outcomes;
it is a disclosed protocol departure, not an outcome-based exclusion.

## 2026-11-15 09:59 CST — compute batching, before empirical outcomes

Run demographic replicates in consecutive batches of at most 24 series in
the frozen selection order, using one engine simulate call per stage/stay
and batch. This avoids repeated Python time loops and reduces runtime and
memory. Replace the earlier demographic seed formula by
415200+1000*stage_count+10*batch_index+stay_index (zero-based indices).
The phase seeds and all 64 draws per record/cell remain unchanged; draws
within a batch use the engine's independent branching law. Archive each
batch membership and seed. This changes RNG assignment, not its law.
Also run the already specified diagnostic variants for all members of any
H3 contrast that departs, if not already triggered by H1/H2; the existing
worst-case diagnostic budget covers every series and does not increase.

## 2026-11-15 10:01:42 CST — count-computation departure and exact completion

The initial run completed all 1,146 threshold cells and five demographic
batches (120 records, two stages, stay 1). It stalled in exact chunked
binomial draws in the first Tribolium batch. Observed conditional growth
rates reach 0.1063 per step; the secondary 500-step horizon permits enormous
counts. Interrupted the process at the shell date above, after approximately
128 seconds. Preserve the log, partial persistence.csv and all completed
demographic archives. An incomplete trajectory is neither extinct nor
right-censored at its intended endpoint; do not classify those paths.

Complete persistence under the **same branching law** using its probability
generating function (PGF). For each finite record compute extinction by
23T,50T,100T and 500 from the registered initial allocation, averaging over
**every** initial phase. Recursion starts with extinction vector zero and
applies the engine's per-parent event generating functions with the next
phase shifted forward. Thus geometric stays, Poisson exits and the host
sequence are unchanged. No count cap, Gaussian approximation, ecological
mechanism, or parasite parameter is introduced. This changes the numerical
estimator from 64 Monte Carlo paths to an exact finite-phase mean; there is
no demographic Monte Carlo confidence interval on that mean. Record phase
min/max and phase-specific probabilities, which are not confidence limits.

Before any new real-data PGF results: reproduce the phase-1 constant-host
branching predictions at log R0=-.1,0,.1 and times 20,60,120, to 1e-12;
compare a synthetic 12-step periodic driver against an independently written
ring PGF and engine simulate (10,000 draws, 120 steps), tolerance 1e-12 for
the deterministic recursion and 4 binomial SE+.002 for Monte Carlo. Retain
the earlier 78 numerical checks. After completion compare the 480 saved
Monte Carlo endpoints to PGF means over their *same* sampled phases with
tolerance 4*sqrt(sum[p_j(1-p_j)])/64+.002; report every discrepancy.

The completed PGF table is the primary common persistence table for all
series; partial Monte Carlo output is an archived validation comparison.
Run the originally specified diagnostics once, with the same seeds. No
threshold or hypothesis is retuned after these observed outcomes. The
original 30-minute total scientific budget still applies, including the
interrupted run. This is an implementation departure and will appear in
the report and run record.
