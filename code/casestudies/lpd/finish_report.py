"""Append the final weather, interpretation and reproducibility sections once.
All reported empirical numbers come from saved results; no new hypothesis is tested.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import csv,collections,re
import numpy as np
from study_io import *

def ci(d,scale=1,digits=3,key='estimate'):
    v=d[key]*scale;lo,hi=[x*scale for x in d['ci95']]
    return f'{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'

def run():
    dest=HERE/'report.md'
    assert '\n## 8. Weather and environment' not in dest.read_text(), 'Final sections already appended'
    audit=read(RESULTS/'final_audit.json');assert audit['passed']
    for name in ['memory_map','acf_by_interval','regime_fractions','sensitivity_fractions']:
        log=(RESULTS/('tectonic_'+name+'_final.txt')).read_text()
        assert 'Writing' in log and (HERE/'figures'/f'{name}.pdf').exists(),name
    w=read(RESULTS/'weather_summary.json');summary=read(RESULTS/'summary_tables.json')
    wm=read(RESULTS/'weather_metrics.json');matched={r['series_id']:r for r in wm if r['matched']}
    rows=list(csv.DictReader((RESULTS/'weather_record_matches.csv').open()))
    sources={}
    for src,count in w['coverage']['by_source'].items():
        group=[r for r in rows if r['series_id'] in matched and r['source']==src]
        distances=[r['distance_km'] for r in matched.values() if r['source']==src and r['distance_km'] is not None]
        sources[src]=dict(series=count,earliest=min(r['start_date'] for r in group),latest=max(r['end_date'] for r in group),median_distance_km=float(np.median(distances)) if distances else None,max_distance_km=max(distances) if distances else None)
    numbers=dict(generated_utc=now(),coverage=w['coverage'],sources=sources,weather_summary=w,final_audit=audit,record_inventory=read(RESULTS/'record_inventory.json'))
    save(RESULTS/'final_report_numbers.json',numbers)
    lines=['\n\n## 8. Weather and environment ('+numbers['generated_utc']+') — amendment\n']
    def add(s=''):lines.append(s+'\n')
    add('This section implements the weather amendment registered before weather analysis. It supplements the original host result; it does not replace its criteria. The dated implementation and metadata corrections are appended to `protocol.md`. The comparison uses identical host dates before and after adjustment. No missing temperature is replaced by zero, and no marine record receives land air temperature as its ocean proxy.')
    add('### Places, dates and coverage\n')
    inv=numbers['record_inventory']
    add('`results/field_locations_dates.csv` lists each primary trajectory’s published location, coordinates, uncertainty, dates and date precision. The complete source-event ledger is `data/raw/lpd/derived/source_field_record_locations.csv.gz`; it also includes assemblage events needed to define BioTIME effort. Its source-row counts are '+', '.join(f'{k}: {v:,}' for k,v in inv['source_record_counts'].items())+'. `data/raw/lpd/derived/analysis_bin_locations.csv.gz` links the retained primary and sensitivity trajectories to those calendar bins. Original GPDD TimePeriodID values are retained rather than presented as exact dates. No final primary trajectory is identified as a laboratory experiment; the laboratory/institution-name correction is recorded explicitly.')
    add('BioTIME pooled series use published study centroids for weather. Individual sample coordinates and each study’s spatial bounds are retained in the source ledger and `results/record_inventory.json`. GPDD coordinates sometimes describe a region or province rather than a trap or plot. A small grid-to-centroid distance does not resolve that location uncertainty. The supplied metadata do not identify these as NEON instrument sites, so no NEON product match was asserted.')
    c=w['coverage']
    add(f"Weather adjustment is possible for **{c['matched_series']} of {c['primary_series']} series**, in **{c['matched_clusters']} study/source clusters**. `results/weather_record_matches.csv` supplies every matched host bin’s source, start/end date, site and source coordinates, distance, temperature and available precipitation. Source-specific coverage among included trajectories is:")
    add('| Source | Series | Resolution | Matched date range | Median / maximum distance (km) |\n|---|---:|---|---|---:|')
    resolutions={'TerraClimate_V1.1':'monthly, 1/24° land grid','NOAA_ERSST_v5':'monthly, 2° ocean grid','NOAA_GHCNm_v4':'monthly station air temperature','Daymet_V4R1':'daily, 1 km land grid'}
    for src,d in sources.items():
        distance='unknown actual pixel centre' if d['median_distance_km'] is None else f"{d['median_distance_km']:.2f} / {d['max_distance_km']:.2f}"
        add(f"| {src} | {d['series']} | {resolutions[src]} | {d['earliest']}–{d['latest']} | {distance} |")
    add('Daymet echoes request coordinates; that is not a measured zero distance to its pixel centre. Air temperature is the mean of reported maximum and minimum temperatures in Daymet/TerraClimate. Precipitation is summed over each observation bin. SST is a surface-water proxy, not a measured fish or parasite body temperature. Annual gridded bins need the registered coverage fraction; the station fallback uses its separately declared annual-month threshold.')
    add('The source access and SHA-256 records are in `data/raw/lpd/weather/WEATHER_MANIFEST.json`, with failed/reused requests retained in `weather/fetch_attempts.jsonl`. Downloaded TerraClimate V1.1 attributes specify CC0 and the extended start in 1950, rather than the older start in the general case-study spec. NOAA products are US government public data. Daymet was obtained through its documented anonymous single-pixel API. The registered GHCN-Monthly fallback departs from the spec’s GHCN-Daily example; it cannot supply daily extremes. [TerraClimate documentation](https://www.climatologylab.org/terraclimate.html), [ERSST ASCII format](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/Readme), [Daymet web services](https://daymet.ornl.gov/web_services).')
    add('Unmatched reasons are '+ '; '.join(f"{reason.replace('_',' ')}: {n}" for reason,n in c['unmatched_reasons'].items())+'. These are analysis-eligibility reasons, not all download failures. An otherwise matched host bin can be lost when lagged weather or adjacent residual pairs are unavailable. There is no substitution of a shorter or more favourable inclusion threshold.')
    add('### W1: adjustment for weather\n')
    add('The adjustment includes intercept, linear time, current temperature, temperature one observation step earlier, and precipitation when complete. The finite-sample white-noise reference uses this full projection. Brackets below are study-cluster bootstrap intervals; they do not make the adjustment causal.')
    add('| Matched sample | Series / clusters | Mean excess host correlation [95% interval] | Median corrected memory | T1 / T2 |\n|---|---:|---:|---:|---|')
    for key,label in [('weather_paired','All, before adjustment'),('weather_adjusted','All, after adjustment'),('weather_paired_annual','Annual, before'),('weather_adjusted_annual','Annual, after'),('weather_paired_subannual','Subannual, before'),('weather_adjusted_subannual','Subannual, after')]:
        g=w['W1'][key]
        add(f"| {label} | {g['M']} / {g['K']} | {ci(g['mean_delta_r1'])} | {g['median_r1_bc']:.3f} | {g['T1']} / {g['T2']} |")
    wd=w['weather_distribution']
    add(f"Within a series, median detrended host–temperature correlation is {wd['corr_host_temp']['median']:.3f}, with 10th–90th percentiles {wd['corr_host_temp']['q10_q90'][0]:.3f} to {wd['corr_host_temp']['q10_q90'][1]:.3f}. For precipitation it is {wd['corr_host_prcp']['median']:.3f} among {wd['corr_host_prcp']['n']} series. The median fraction of detrended log-abundance variation explained in sample by the weather design is {100*wd['r2_weather_explained']['median']:.1f}%. This is an in-sample association, without a causal or out-of-sample claim.")
    add('The following compares like-for-like weather mappings. Both use the same amplitude-aware interpolation and outside-grid engine evaluation. Their strict-accuracy fractions are less numerically resolved than the direct primary values in section 6.')
    add('| Cycle | Parking before / after (%) | 30% accuracy before / after (%) | 10% accuracy before / after (%) |\n|---|---:|---:|---:|')
    before=w['W1']['weather_paired']['fractions'];after=w['W1']['weather_adjusted']['fractions']
    for cycle in ['long','moderate','short']:
        pairs=[ci(before[cycle+'_'+key],100,1)+' / '+ci(after[cycle+'_'+key],100,1) for key in ['i_half_bound','iii_30pct','iii_10pct']]
        add('| '+cycle+' | '+' | '.join(pairs)+' |')
    add(f"The registered short-stage memory-zone fraction changes from {ci(before['short_penalty_zone_T2'],100,1)}% to {ci(after['short_penalty_zone_T2'],100,1)}%. **W1 does not change the registered conclusion:** positive host memory remains supported overall and in annual and subannual subsets. This rules out this particular linear weather adjustment as a complete statistical explanation, not all environmental explanations. Nonlinear weather effects, other lags, season–weather interactions and imperfect weather proxies remain untested.")
    add('### W2: climate associations with hosts and predicted parasite loss\n')
    add('The table reports cross-series Spearman correlations, with study-cluster intervals. Host variability is log-abundance standard deviation; memory is corrected lag-1 correlation. Parasite loss is the model output D. No parasite abundance, prevalence, infection intensity or R0 was observed in these records.')
    corr={r['label']:r for r in w['W2']}
    add('| Climate covariate | Host log-SD | Host memory | Predicted long-cycle D | Predicted short-cycle D |\n|---|---:|---:|---:|---:|')
    for key,label in [('mean_temp_c','Mean temperature'),('sigma_temp_c','Temperature SD'),('r1_bc_temp','Temperature memory'),('mean_prcp_mm','Mean precipitation per bin'),('sigma_prcp_mm','Precipitation SD')]:
        add('| '+label+' | '+' | '.join(ci(corr['overall:'+key+':'+y],key='rho') for y in ['sigma','r1_bc','long_D','short_D'])+' |')
    add('Overall temperature intervals include zero for these host and predicted-loss associations. Precipitation–memory associations are negative, but precipitation per bin mixes sampling durations and climates. These descriptive comparisons have no multiplicity-adjusted causal decision rule. The registered realm/interval stratification is therefore material:')
    add('| Realm and interval | Series / clusters | Temperature SD vs host log-SD | Temperature memory vs host memory |\n|---|---:|---:|---:|')
    for realm in ['Terrestrial','Freshwater','Marine']:
        for iv in ['annual','subannual']:
            stem=realm+':'+iv;g=corr[stem+':sigma_temp_c:sigma'];h=corr[stem+':r1_bc_temp:r1_bc']
            add(f"| {realm}, {iv} | {g['n']} / {g['K']} | {ci(g,key='rho')} | {ci(h,key='rho')} |")
    add('The full matrix, including moderate-cycle predictions, is in `results/weather_summary.json`. Differences between these strata warn against treating a global pooled association as a universal biological mechanism. Multiple species sharing a weather record are not independent climate replicates; uncertainty resamples study/source clusters.')
    add('### W3: temperature as a model input\n')
    add('The previously registered development sensitivity is a(T) = Q10^((20 − T)/10), measured in observation steps, paired with sheltered adult durations from the original model. Q10 is illustrative, not estimated from these abundance data. A physiological basis for testing temperature-dependent development is [Zuo et al., *A general model for effects of temperature on ectotherm ontogenetic growth and development*](https://web.mst.edu/_disabled/houch/Publications/Temperature%20size%20rule%20PRSB.pdf), DOI 10.1098/rspb.2011.2000. The abstract, introduction, model and empirical-evaluation sections were read; the card records partial reading and an unsuccessful DOI-resolver request. This supports a sensitivity exercise, not a parasite-specific rate calibration or novelty claim.')
    add('| Q10 | Matched / admissible series | Median exposed duration a | Median short-stage ratio b/a | Short-stage second-order penalty cases / admissible |\n|---|---:|---:|---:|---:|')
    for q,g in w['W3a'].items():
        d=g['all'];add(f"| {q} | {d['n']} / {d['admissible']} | {d['median_a']:.3f} | {d['median_short_ratio']:.3f} | {d['short_penalty_count']} / {d['admissible']} |")
    add('A coarse discrete duration below one is inadmissible and is not clamped. The displayed duration/ratio medians describe all matched temperatures; penalty counts use only admissible cases. These cycles vary with temperature and differ from the original fixed-duration comparison. They should not be read as revised primary fractions.')
    add('| Latitude band, Q10 = 2.5 | Matched / admissible | Median a | Short-stage penalty count |\n|---|---:|---:|---:|')
    for band in ['tropical','temperate','boreal_polar']:
        d=w['W3a']['2.5'][band];add(f"| {band.replace('_',' ')} | {d['n']} / {d['admissible']} | {d['median_a']:.3f} | {d['short_penalty_count']} |")
    add('The tropical sample is especially sparse and mostly outside the coarse model’s admissible duration range. Anchoring a to one annual versus one monthly observation is not the same physiological timescale. These are conditional illustrations, not predicted natural stage lengths.')
    add('W3b evaluates the project’s second-order thermal clock/completion coefficient at the amended interior exposed duration a = 4, with π = ±0.05 per °C and χ = 0.08 per °C. The original boundary a = 1 cannot sustain a two-sided clock perturbation; its formal coefficients are saved separately, not treated as admissible simulations. The following medians use measured temperature variability and memory but illustrative thermal sensitivities:')
    add('| Cycle | Same-sign median thermal D2 | Opposite-sign median thermal D2 | Negative thermal coefficients, same / opposite |\n|---|---:|---:|---:|')
    for cycle in ['long','moderate','short']:
        a=w['W3b']['D2_thermal_same_'+cycle];b=w['W3b']['D2_thermal_opposite_'+cycle]
        add(f"| {cycle} | {a['median']:.6f} | {b['median']:.6f} | {a['negative_count']} / {b['negative_count']} |")
    add('A negative coefficient denotes a second-order growth benefit in this thermal model. It does not prove a finite-amplitude benefit. The host and thermal calculations use different exposed stays and omit their joint cross-covariance terms; they must not be added as if they were a calibrated combined loss. The saved thermal/host magnitude comparisons are descriptive only. Temperature can change a model’s sign, but these data do not identify the parasite’s thermal response.')
    add('### What the weather amendment leaves unresolved\n')
    add('Weather adjustment leaves the registered host-memory result supported. Direct temperature effects make the parasite interpretation conditional on unknown stage rates. Air grids cannot resolve freshwater temperature, burrow or canopy microclimate, coastal extremes, host migration or within-study spatial footprints. Vapour pressure may be present in downloaded Daymet records but is not an analysed covariate. There is no account-free ERA5 extraction, no new weather station observation, and no observed parasite outcome. Residual environmental confounding remains possible.')
    add('## 9. Interpretation, final descriptive distributions, and work not done\n')
    add('**Supports:** the registered host-side statements T1 and T2, in the retained compilations and under their stated sampling assumptions. **Limits:** treating host fluctuations as memoryless, using a common small-noise approximation without checking its error, and exporting one life-cycle buffering fraction across host laws and waiting mechanisms. **Says nothing directly:** whether a particular natural parasite’s R0 declines or rises, or whether host richness or population size is regulated by an intrinsic process. The inherited word “intrinsic” is too strong for this observational test.')
    add('The final descriptive table below supersedes section 5’s preliminary distribution counts. Entries are medians. The full distributions, 10th/25th/75th/90th percentiles, cluster intervals and taxon × interval breakdowns are saved in `results/summary_tables.json`; within-series moving-block uncertainty and ACFs are in `results/series_metrics.csv`.')
    add('| Taxon or sampling interval | Series / clusters | Log variance | Raw lag-1 correlation | Corrected memory | Spectral slope |\n|---|---:|---:|---:|---:|---:|')
    keys=[k for k in summary if k.startswith('taxon_group:') or k.startswith('interval_class:')]
    for key in keys:
        g=summary[key];vals=[g['distributions'][m]['quantiles']['0.5']['estimate'] for m in ['sigma2','r1','r1_bc','b_spec']]
        add('| '+key.split(':',1)[1].replace('_',' ')+' | '+str(g['M'])+' / '+str(g['K'])+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    add('A spectral slope describes the fitted log spectrum over the registered band, not a validated power-law process. The “full ACF” requested here is reported through the registered finite lag limit; unobserved longer lags are not inferred. Annual, monthly and seasonal steps retain their physical interval labels in the exported regimes.')
    add('Remaining limits are substantial. This is a convenience sample of published and monitored hosts, geographically uneven and dominated by birds. Compilation overlap or shared original surveys may leave dependence beyond the study/source clusters. Short records, zeros, sampling effort, detection, nonstationarity and the approximate AR(1) bias calibration limit biological interpretation. Raw observations are not used to fit parasite mechanisms. Noise-share, elasticity, stage spread, waiting, alternative-host and patch sensitivities are illustrative bounds rather than identified parameters. Parasite density dependence, host–parasite feedback, evolution, true mean encounter density, correlated alternative hosts, and natural baseline R0 remain unmeasured.')
    add('The supplied copy lacks `../paper/record/` and `verification/engine-R16-check.md`; those references could not be read. The available exact-limits, R16 protocol, temperature notes, engine implementation and numerical checks were used, and absent caveats cannot be claimed reviewed. LPD itself remains unavailable under the no-personal-details condition. No literature-wide novelty claim is made. The temperature paper was not read in full and cannot clear a novelty claim.')
    add('The empirical export `results/memory_regimes.json` is ready for the memory-extension study. Each record carries variance, memory, finite ACF, spectral and bootstrap information, physical sampling interval, cluster/source/units/licence metadata, and labelled model predictions. Use the source-specific licence and preserve provenance when redistributing derived VecDyn records. A strict BioTIME + GPDD summary is supplied separately. The export supplies plausible empirical regimes; it is not a fitted natural parasite model.')
    add('## 10. Reproduction and verification\n')
    add('The inherited files and the original prefixes of `protocol.md` and `report.md` are unchanged byte for byte, checked against `results/continuation_original_files.json`. Corrections are dated additions. Failed harness outputs and earlier estimates remain on disk. The final audit passed '+str(audit['manifest_hashes_checked'])+' original-data manifest hashes and '+str(audit['weather_manifest_hashes_checked'])+' weather-manifest hashes. The final field inventory contains no laboratory trajectories. This copy has no Git metadata, so git-ignore status was not verifiable and no commit or push was attempted.')
    add('`results/continuation_checks.json` records the independent known-result checks: accelerated matrix products against the original engine, the ring formula against the engine, the ACF sum against AR(1), the thermal formula against an independent resolvent, vectorized bootstrap against a loop, projection-specific white-noise simulation, and synthetic interpolation checks. Tolerances were declared before empirical use. Direct primary coarse losses have separate provenance in `results/primary_direct_cells.json`; Newton/bracket fallbacks are logged. These checks do not certify extreme-lognormal tail sampling or remove Monte Carlo and finite-horizon uncertainty.')
    add('The following runs from the repository root with the existing cached data and saved pre-analysis surfaces. Network refresh is optional and separate. Checkpoint reuse is deliberate; final summaries read the direct-primary file written by `refine_primary.py`.')
    add('```sh\npython3 -I casestudies/lpd/extract_series.py\npython3 -I casestudies/lpd/analysis.py metrics\npython3 -I casestudies/lpd/analysis.py predict\npython3 -I casestudies/lpd/refine_primary.py\npython3 -I casestudies/lpd/summarize.py\npython3 -I casestudies/lpd/record_inventory.py\npython3 -I casestudies/lpd/weather_continued.py\npython3 -I casestudies/lpd/weather_summary.py\npython3 -I casestudies/lpd/make_figures.py\npython3 -I casestudies/lpd/audit_results.py\n```')
    add('The continuation weather fetcher is `data/fetch/lpd-weather.py`; it requires no account. If refreshed inputs change, archive and invalidate their dependent metric/prediction checkpoints before rerunning. The original fetcher and pre-analysis scripts are retained unchanged. The optional C accelerator is local source in `products.c`; `numerics.py` builds it if absent. Saved outputs remain inspectable without recompilation.')
    add('Four standalone pgfplots figures, their CSVs and compiled PDFs are in `figures/`: `memory_map`, `acf_by_interval`, `regime_fractions`, and `sensitivity_fractions`. `figures/plotting_note.txt` explains intervals, denominators and the memory-map viewport. Final output logs are `results/analysis_output_final.txt`, `summary_output_final.txt`, `refinement_output_final.txt`, `weather_analysis_output_final.txt`, `weather_summary_output_final.txt`, and `audit_output_final.txt`. `results/final_report_numbers.json` saves the inputs used for these final sections; `results/final_report_output.txt` prints this appended text.')
    add('The registered empirical case study and weather amendment are complete within the stated coverage and identifiability limits. No fieldwork, fitted parasite response, inaccessible LPD download or unregistered simulation campaign was performed. Further work would need an independently registered protocol and parasite-specific observations or rate measurements.')
    add('Status: done')
    text=re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)', r'\1\n', '\n'.join(lines))
    with dest.open('a') as f:f.write(text)
    print(text,flush=True)

if __name__=='__main__':run()
