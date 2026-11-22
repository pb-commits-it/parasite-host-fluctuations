"""Record continuation state before computing environmental associations."""
from pathlib import Path
import subprocess, hashlib, json
p=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
now=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name!='continuation_setup.py'}
state={'timestamp':now,'report_present':(p/'report.md').exists(),'original_sha256':files,'missing_context':['../paper/record/','verification/engine-R16-check.md','results/r14-posthoc-power_20261114.txt','.git','.gitignore']}
(p/'results/continuation_inventory.json').write_text(json.dumps(state,indent=2)+'\n')
amend=f'''

## Dated Protocol Amendment 2 ({now}; continuation, before environmental analysis)

The copied folder has the registered protocol, helpers and prechecks, but no
`report.md`. Their hashes and missing background paths are recorded in
`results/continuation_inventory.json`. Existing files and prior results are kept.
The supplied R12/R14 reports and engine docstring were read; unavailable context
is not claimed read. There is no Git metadata or ignore file in this copy, so
`git check-ignore` cannot verify exclusion here. No raw data are copied to reports.

Implementation clarifications, with every registered decision criterion unchanged:

- R12's report specifies 102 lagged pairs with mouse years 1994–2012 (19 calendar
  cohorts), not 18. Reproduce those exact pairs in the primary adjustment and use
  their actual calendar span for common two-year blocks (seed 20261114).
  R12b uses its original 40 pairs with at least 20 tested nymphs; its already-lagged
  mouse column must not be shifted again. Part 4's registered 18-year design is
  retained, with missingness and nymph counts sampled from the observed primary
  records. Its difference from the exact primary calendar is disclosed.
- Cary PC1 is trained on unique collection years in the primary period and applied
  unchanged to larval years and holdout. NEON's site-mean PC1 is the registered
  six-site PCA; annual NEON scores use those same scaling and loading constants.
  Individual adjustment covariates are standardized only for numerical stability.
- Daymet has 365 records even in leap years (December 31 omitted; official Daymet
  overview read). Covariates use the supplied days without interpolated weather.
  Primary thermal cohorts retain the registered numeric day-of-year definitions;
  duration days mean supplied daily records, with at most one omitted winter day
  per crossed leap year. This is a disclosed calendar approximation. Annual
  nymph cohorts use previous-year larval drop-off; no end-of-file censored cohort
  is silently included. The point estimate of active waiting is a cumulative-hazard
  threshold, not a sampled exponential wait. Thus its spread is a model-implied
  environmental/cohort spread, not a measured individual duration distribution.
- Part 4 does not identify causal weather effects from prevalence correlations.
  Data supply climate, mouse variance, autocorrelation, cross-covariance, thermal
  durations, sampling counts and observed prevalence levels. Unknown weather
  survival/ln-R0 slopes are illustrative: 0, -0.20, +0.20 per standardized climate
  unit, with residual site SD 0.17 and annual residual log-prevalence SD 0.14
  (precheck settings); ±0.10 additionally bounds weather strength where feasible.
  Thermal survival sensitivity uses exp(-mu*duration) with mu=0.001/day (Wallace
  card), checked at 0 and 0.002/day; baseline climate and direct annual-weather
  terms are bounded scenarios, not fitted causal coefficients. A weather effect
  on tick abundance need not affect prevalence; zero is included explicitly.
- In Part 4 use the observed site climate scores directly (preserving their exact
  covariance with host variability). Within-site annual driver pairs use the
  project's Environment sampler at measured annual correlations/memories; if its
  innovation covariance is infeasible, use joint two-year block resampling and
  record the departure. Engine coupled-driver loss sensitivities keep subannual
  memory assumptions separate from annual estimates. Cary prevalence elasticities
  for kappa=0, 0.5, 1 are illustrative 0, (1-pbar)/4, (1-pbar)/2 as in the existing
  helper; they are not predictions of an R0-loss model for an unmeasured epidemic
  equilibrium. Simulated slope SEs use the same two-year blocks as the field test.
- Engine calibrations retain 24 antithetic pairs and 40,000 steps. A local sparse
  matrix-product accelerator may replace only Engine._products, after agreement
  with the unmodified routine within 1e-10 on synthetic memoryless, chain, fixed,
  correlated-driver and patch examples. The engine's rates, criticality, loss root,
  coefficient and Jensen bound remain its own. Save each cell on completion.
- Negative memory is an illustrative step-level sensitivity; negative annual
  correlation has no justified fractional-power conversion. Contact-limited,
  dilution, alternative-route, patches, mouse-stay, fixed-delay and extinction
  sensitivities are conditional examples, not measured mechanisms. A single
  finite campaign uses predeclared cells in analysis.py, without outcome-driven
  extension. Allow up to 45 minutes of computation, rather than the earlier
  optimistic eight-minute combined budget; stop and report any uncompleted cells.

No new novelty claim is proposed. Existing literature cards report the earlier
read status; continuation checks are recorded separately and do not silently
upgrade inaccessible sources. Registered outcomes from earlier studies stand.
'''
with (p/'protocol.md').open('a') as f:f.write(amend)
if not (p/'report.md').exists():
 (p/'report.md').write_text(f'''# Environmental covariates in the mouse–tick–Borrelia studies

Continuation opened {now}. No report.md was present in this copy. This report
is being built progressively from the retained protocol, helpers and saved outputs.
The original protocol and pre-analysis results remain intact. New findings follow
below; earlier sections will not be rewritten.

## Question, data and prediction

Could weather explain or conceal the association between mouse variability and
infection prevalence at six NEON sites, or change the lagged mouse slope at Cary?
Could temperature-based tick durations materially change the model's fluctuation
loss D, the reduction of log reproduction number relative to arithmetic-mean rates?
Neither field series measures that reproduction number. These are conditional
pattern checks, not a direct causal test of its loss.

One weather record is one grid cell-day. The retained Daymet files supply daily
weather at Cary and tick-plot and tower coordinates; NASA POWER supplies coarse
humidity, surface wetness and skin temperature checks. The raw manifests record
source URLs, dates, versions, licences and hashes. NEON biological records are
trap-nights, identified ticks and individual pathogen tests; Cary records are
plot-years. Detailed counts and matching checks will be appended from the run.

The registered covariates are development degree-days, hot dry days, winter
minimum temperature, snow-cover days, growing-season precipitation and saturation
deficit. Definitions and literature read status are preserved in protocol.md.
Temperature-dependent maturation uses the published Wallace model, with an
explicit spring gate and active questing wait. These model-derived durations are
not measured residence times. The R16 engine computes finite-noise D; the analytic
small-noise coefficient and its Jensen bound provide checks.

## Pre-analysis power and identifiability (retained)

The completed precheck is results/power_precheck_20261114.txt and its JSON.
It reproduced the earlier R14 simulation and checked the permutation implementation
before environmental analysis. Its six-site scenarios gave low power even at full
host coupling. One adjustment leaves only three residual degrees of freedom.
Testing six covariates separately requires the registered multiplicity control.
Cary weather is shared by plots within a year; plots do not supply independent
weather replicates. Annual blocks, not independent plot-years, set uncertainty.
No new pre-analysis campaign is needed. The registered support, contradiction and
confounding-shift rules are unchanged.
''')
print(json.dumps({'timestamp':now,'files_preserved':len(files),'report_present_before':state['report_present']}))
