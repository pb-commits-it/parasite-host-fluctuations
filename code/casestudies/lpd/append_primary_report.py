"""Append final host/theory sections, retaining earlier report text verbatim."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_io import *
def fmt(d,percent=False):
    f=100 if percent else 1;dec=1 if percent else 4
    return f"{d['estimate']*f:.{dec}f} [{d['ci95'][0]*f:.{dec}f}, {d['ci95'][1]*f:.{dec}f}]"
def run():
    tables=read(RESULTS/'summary_tables.json');r=tables['overall'];sens=read(RESULTS/'sensitivity_tables.json');acf=read(RESULTS/'full_acf_sensitivity.json');ref=read(RESULTS/'refinement_audit.json');t=now()
    text=f'''\n\n## 6. Final registered results and dated corrections ({t})

**This section supersedes the preliminary counts and estimates in section 5.** The GPDD metadata correction restored four bird series and corrected land/ocean classification. The final primary set contains **{r['M']} series in {r['K']} study/source clusters**. The database counts are in `results/eligibility_audit.json`. The strict BioTIME + GPDD subset contains {tables['CC_BY_only']['M']} series; it excludes VecDyn even when a particular VecDyn licence is permissive. No registered criterion was rewritten. Earlier results remain in `results/attempt3_before_metadata_correction/`.

**T1: {r['T1']}.** The corrected overall excess correlation is {fmt(r['mean_delta_r1'])}; median corrected memory is {r['median_r1_bc']:.4f}. This satisfies the registered interval and positive-median conditions. The following rows give all registered taxon × annual/subannual cells, with their own cluster counts and decisions. Spectral and ACF distributions, quantiles and their cluster intervals are also in `results/summary_tables.json`.

| Taxon and interval | Series / clusters | Mean excess lag-1 correlation [95% interval] | T1 |
|---|---:|---:|---|
'''
    for k,g in tables.items():
        if not k.startswith('taxon_interval:'):continue
        text+=f"| {k.replace('taxon_interval:','').replace('_',' ').replace(':',', ')} | {g['M']} / {g['K']} | {fmt(g['mean_delta_r1'])} | {g['T1']} |\n"
    text+='\nThe interval distribution is not a common biological clock. Longer sampled intervals can hide shorter positive memory or alias cycles. Positive AR(1) memory on a coarse interval is converted to a fourth root only in the registered illustrative fine-grid model. Negative memory has no such real conversion and is excluded from that comparison, with its denominator shown.\n\n'
    text+=f"**T2: {r['T2']}.** The long-cycle strict-accuracy failure fraction is {fmt(r['fractions']['long_strict_failure'],True)}%, and the fraction above the registered short-stage memory boundary is {fmt(r['fractions']['short_penalty_zone_T2'],True)}%. Each lower bound exceeds the registered 20% support threshold. This supports departure from the memoryless/small-noise premise in these retained series; it does not validate a causal parasite effect.\n\n"
    text+='The primary coarse-cycle finite-amplitude losses below were computed directly with the project engine after the registered numerical precision amendment. Brackets resample study clusters. They do not include all within-series estimation error, simulation error or finite-horizon bias.\n\n| Illustrative cycle | Parking: D ≤ σ²/4 (%) | Small-noise accuracy within 30% (%) | Within 10% (%) |\n|---|---:|---:|---:|\n'
    for c in ['long','moderate','short']:text+=f"| {c} | {fmt(r['fractions'][c+'_i_half_bound'],True)} | {fmt(r['fractions'][c+'_iii_30pct'],True)} | {fmt(r['fractions'][c+'_iii_10pct'],True)} |\n"
    fine=r['fractions']['short_ii_loss_fine_unexpanded'];idx=r['fractions']['short_ii_index_P2_0.35'];pl=r['fractions']['short_plugin_penalty_zone_T2']
    text+=f'''\nOn the admissible fine grid, the short steady stage raises the unexpanded loss in {fmt(fine,True)}% of {fine['denominator']} nonnegative-memory regimes. That comparison changes the exposed stage from a fixed coarse stay to a memoryless fine stay, as specified in R16; it is a different model, not merely a more accurate discretization. The persistence-index reversal at illustrative ln R0 = 0.35 occurs in {fmt(idx,True)}% of all primary regimes. An index reversal and a threshold-loss reversal are different statements. The plug-in, uncorrected-memory penalty-zone fraction is {fmt(pl,True)}%; its difference from the corrected fraction shows why finite-series estimation matters.

Numerical precision matters too. Direct coarse-cycle evaluation changes {ref['strict_long_changes']} long-cycle strict-accuracy classifications from the amplitude-aware interpolation result. `results/primary_interpolation_predictions.json.gz` preserves that approximation, and `results/primary_direct_cells.json` records {ref['n_cells']} distinct parameter/cycle evaluations (identical inputs share a cell). Raw-D interpolation, amplitude-aware interpolation and direct evaluation must not be conflated. The registered T2 conclusion also holds through the memory fraction, independently of either loss interpolation. The {ref['extreme_lognormal_series']} series with log-SD above 3 remain included under the original rule, but rare lognormal peaks are poorly identified by these simulation budgets. Their loss errors are not certified by a small Monte Carlo standard error.

## 7. Registered sensitivity analyses

The host-memory premise remains supported in each completed registered sensitivity below. These comparisons change eligible samples as well as transformations; they are not causal estimates. “Memory zone” is the registered φ > 0.1919 fraction and is independent of loss interpolation. Sensitivity finite-amplitude losses use the validated surfaces and direct outside-grid calls, rather than the all-primary direct numerical refinement.

| Sensitivity | Series / clusters | T1 | T2 | Memory zone (%) [95% interval] |
|---|---:|---|---|---:|
'''
    for label,g in sens.items():text+=f"| {label.replace('_',' ')} | {g['M']} / {g['K']} | {g['T1']} | {g['T2']} | {fmt(g['fractions']['short_penalty_zone_T2'],True)} |\n"
    text+='\nS1 retains zeros through the registered half-minimum offset; S2 removes seasonal means; S3 retains trend; S5 applies the high-abundance threshold; S6 uses fixed BioTIME plots and the registered quality/manipulation exclusions; S7 relaxes coverage; S8b changes the assumed observation variance and, where allowed, uses the lag-2/lag-1 memory proxy; S9a halves encounter elasticity. S5 is only a noise sensitivity because abundance indices and densities do not all have literal individual-count units. The lag-ratio correction in S8b needs an AR(1) process and independent observation error; it is not an identified biological correction for arbitrary cyclic hosts.\n\n'
    text+='S4 evaluates the tapered, finite-lag empirical ACF instead of an AR(1) extrapolation. The coefficient changes sign in '+', '.join(str(acf[c]['sign_changes'])+' of '+str(acf[c]['n'])+' '+c+'-cycle cases' for c in ['long','moderate','short'])+'. These are coefficient comparisons, not finite-amplitude loss measurements. The registered ACF stops at the lesser of ten lags and one third of the series length. Pair-normalized estimates with missing observations are not guaranteed to define a positive-semidefinite covariance sequence, even after tapering. Longer memory remains unmeasured.\n\n'
    text+='The engine mechanism sensitivities below use a common numerical benchmark D ≤ σ²/4. It is half the Jensen bound only in the original mean-one lognormal transfer model. It is not half the exact bound under a bounded host or fluctuating staying rates. Entries are conditional on each saved sensitivity grid; out-of-grid and mathematically inadmissible cases are omitted, never clipped.\n\n| Mechanism / host law | Eligible series | Long (%) | Moderate (%) | Short (%) |\n|---|---:|---:|---:|---:|\n'
    for mode,label in [('bounded_coarse','Bounded host'),('chain4_fine','Four-substage steady stay, fine grid'),('alt2_coarse','Two independent alternatives'),('patches4_coarse','Four asynchronous patches'),('contact_fine','Contact-limited waiting with death, bounded host')]:
        ds=[r['fractions']['S9_'+mode+'_'+c+'_half_bound'] for c in ['long','moderate','short']]
        text+=f"| {label} | {ds[0]['denominator']} | "+' | '.join(fmt(d,True) for d in ds)+' |\n'
    text+='\nThe prevalence of predicted buffering is therefore sensitive to host law, waiting mechanism, duration spread, alternative hosts and patches. The contact comparison also changes host law and cannot isolate contact limitation by itself. No such mechanism was fitted to abundance data. Mean-abundance associations within unit types are saved in `results/mean_abundance_correlations.json`; units and study composition limit their interpretation as density effects. They cannot determine a natural baseline R0.\n'
    save(RESULTS/'primary_report_numbers.json',dict(generated_utc=t,summary=tables,sensitivities=sens,full_acf=acf,refinement=ref))
    print(text,flush=True)
    with (HERE/'report.md').open('a') as f:f.write(text)
if __name__=='__main__':run()
