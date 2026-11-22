"""Append the fix1 comparison once; preserve every byte of the earlier report.

All numbers are read from outputs of the registered rerun or explicitly recomputed
from archived predictions using the same summary function and seed convention.
"""
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collections,csv,hashlib,json
import numpy as np
from study_io import *
from summarize import grouping,summarize

OLD=RESULTS/'before_fix1'
TITLE='## Corrected registered results (fix1, 2026-11-15)'


def repr_value(v):
    return json.dumps(clean(v),ensure_ascii=False,separators=(',',':')).replace('|',r'\|')


def differences(a,b,path=''):
    """Preserve complete interval/statistic objects in side-by-side cells."""
    if a==b:return []
    if isinstance(a,dict) and isinstance(b,dict):
        if 'estimate' in a or 'rho' in a:return [(path,a,b)]
        out=[]
        for k in sorted(a.keys()|b.keys()):
            if k in ['timestamp','generated_utc','finished_utc','started_utc']:continue
            out.extend(differences(a.get(k),b.get(k),path+'/'+str(k)))
        return out
    if isinstance(a,list) and isinstance(b,list) and a and b and isinstance(a[0],dict) and isinstance(b[0],dict):
        for field in ['label','series_id','check','path']:
            if all(field in r for r in a+b):
                return differences({r[field]:r for r in a},{r[field]:r for r in b},path)
    return [(path,a,b)]


def full_weather_cells(root):
    w=read(root/'weather_summary.json')
    for mode in ['weather_paired','weather_adjusted']:
        for label,rr in grouping(read(root/(mode+'_predictions.json.gz'))).items():
            if label.startswith(('taxon_interval:','taxon_detailed:','taxon_subgroup:')):
                key=mode+':'+label
                if key not in w['W1']:w['W1'][key]=summarize(rr,key,False)
    return w


def counts(rows):
    out=dict(M=len(rows),K=len({r['study_id'] for r in rows}))
    for key in ['database','taxon_group','interval_class','series_kind']:
        out[key]=dict(collections.Counter(r[key] for r in rows))
    return out


def run():
    report=HERE/'report.md';prefix=(OLD/'original_files/report.md').read_bytes()
    assert report.read_bytes()==prefix,'Append only once; earlier text must be untouched'
    assert read(RESULTS/'fix1_independent_audit.json')['passed']
    compilation=read(RESULTS/'fix1_figure_compile_log.json')
    assert len(compilation)==4 and all(r['returncode']==0 for r in compilation)
    oldsummary=read(OLD/'summary_tables.json');newsummary=read(RESULTS/'summary_tables.json')
    oldweather=full_weather_cells(OLD);newweather=full_weather_cells(RESULTS)
    save(RESULTS/'fix1_old_weather_cells.json',oldweather['W1'])
    pairs={'primary':(oldsummary,newsummary),'weather':(oldweather['W1'],newweather['W1']),
           'sensitivity':(read(OLD/'sensitivity_tables.json'),read(RESULTS/'sensitivity_tables.json'))}
    labels=[]
    for scope,(a,b) in pairs.items():
        for key in sorted(a.keys()|b.keys()):
            for test in ['T1','T2']:
                x=a.get(key,{}).get(test,'not decided');y=b.get(key,{}).get(test,'not decided')
                if x!=y:labels.append(dict(scope=scope,stratum=key,test=test,old=x,new=y))
    save(RESULTS/'fix1_label_changes.json',labels)
    oldrows=read(OLD/'primary_predictions.json.gz');newrows=read(RESULTS/'primary_predictions.json.gz')
    om={r['series_id']:r for r in oldrows};nm={r['series_id']:r for r in newrows}
    removed=sorted(om.keys()-nm.keys());added=sorted(nm.keys()-om.keys())
    oldcan=read(RAW/'derived/before_fix1/candidate_series.json.gz');newcan=read(RAW/'derived/candidate_series.json.gz')
    bins=read(RESULTS/'extraction_audit.json')['counts']['biotime_binning']
    changed_studies=sorted([k for k,v in bins.items() if 2<=v['median_distinct_months']<2.5],key=int)
    bincounts=dict(old_study_bins=dict(collections.Counter('subannual_monthly' if v['median_distinct_months']>=6 else 'subannual_seasonal' if v['median_distinct_months']>=2.5 else 'annual' for v in bins.values())),
        new_study_bins=dict(collections.Counter(v['interval_class'] for v in bins.values())),
        old_candidates=counts(oldcan),new_candidates=counts(newcan),old_primary=counts(oldrows),new_primary=counts(newrows),
        changed_studies=changed_studies,removed_primary=removed,added_primary=added,study_counts={})
    for sid in changed_studies:
        k='BT_STUDY_'+sid
        bincounts['study_counts'][k]={label:counts([r for r in rows if r['study_id']==k])
             for label,rows in [('old_candidates',oldcan),('new_candidates',newcan),('old_primary',oldrows),('new_primary',newrows)]}
    save(RESULTS/'fix1_binning_counts.json',bincounts)
    # Restore current versions of derivative tables formerly made during reporting.
    fields=list(next(csv.DictReader((OLD/'field_locations_dates.csv').open())).keys())
    candidates_by_id={r['series_id']:r for r in newcan}
    locations=[]
    for r in newrows:
        s=candidates_by_id[r['series_id']]
        locations.append(dict(r,realm=s['realm_detail'],dates=list(zip(s['step_years'],s['step_months']))))
    csvwrite(RESULTS/'field_locations_dates.csv',locations,fields)
    wm=read(RESULTS/'weather_metrics.json');matched={r['series_id']:r for r in wm if r['matched']}
    with (RESULTS/'weather_record_matches.csv').open() as f:ledger=list(csv.DictReader(f))
    sources={}
    for source,n in newweather['coverage']['by_source'].items():
        rr=[r for r in ledger if r['series_id'] in matched and r['source']==source]
        distances=[r['distance_km'] for r in matched.values() if r['source']==source and r['distance_km'] is not None]
        sources[source]=dict(series=n,earliest=min(r['start_date'] for r in rr),latest=max(r['end_date'] for r in rr),
            median_distance_km=float(np.median(distances)) if distances else None,max_distance_km=max(distances) if distances else None)
    save(RESULTS/'primary_report_numbers.json',dict(generated_utc=now(),summary=newsummary,sensitivities=pairs['sensitivity'][1],
         full_acf=read(RESULTS/'full_acf_sensitivity.json'),refinement=read(RESULTS/'refinement_audit.json')))
    finalnumbers=dict(generated_utc=now(),coverage=newweather['coverage'],sources=sources,weather_summary=newweather,
        final_audit=read(RESULTS/'final_audit.json'),record_inventory=read(RESULTS/'record_inventory.json'))
    save(RESULTS/'final_report_numbers.json',finalnumbers)
    comparisons={
        'eligibility_audit.json':(read(OLD/'eligibility_audit.json'),read(RESULTS/'eligibility_audit.json')),
        'extraction_audit.json':(read(OLD/'extraction_audit.json'),read(RESULTS/'extraction_audit.json')),
        'summary_tables.json':(oldsummary,newsummary),
        'sensitivity_tables.json':pairs['sensitivity'],
        'full_acf_sensitivity.json':(read(OLD/'full_acf_sensitivity.json'),read(RESULTS/'full_acf_sensitivity.json')),
        'mean_abundance_correlations.json':(read(OLD/'mean_abundance_correlations.json'),read(RESULTS/'mean_abundance_correlations.json')),
        'weather_summary.json':(oldweather,newweather),
        'weather_source_coverage':(read(OLD/'final_report_numbers.json')['sources'],sources),
        'record_inventory.json':(read(OLD/'record_inventory.json'),read(RESULTS/'record_inventory.json')),
        'refinement_audit.json':(read(OLD/'refinement_audit.json'),read(RESULTS/'refinement_audit.json')),
        'final_audit.json':(read(OLD/'final_audit.json'),read(RESULTS/'final_audit.json'))}
    # Binning metadata is new; old cutoff counts are reconstructed explicitly above.
    comparisons['extraction_audit.json'][1]['counts'].pop('biotime_binning')
    changes=[]
    for name,(a,b) in comparisons.items():
        changes.extend(dict(output=name,path=k,old=x,new=y) for k,x,y in differences(a,b))
    save(RESULTS/'fix1_aggregate_changes.json',changes)
    csvwrite(RESULTS/'fix1_aggregate_changes.csv',changes)
    record_changes={}
    names=['primary_metrics.json','primary_predictions.json.gz','primary_interpolation_predictions.json.gz',
           'weather_metrics.json','weather_paired_metrics.json','weather_adjusted_metrics.json',
           'weather_paired_predictions.json.gz','weather_adjusted_predictions.json.gz']
    for mode in ['S1_zero','S2_season','S3_no_detrend','S6_quality_fixedplot','S7_coverage50']:
        names.extend([mode+'_metrics.json',mode+'_predictions.json.gz'])
    names.extend(['S8b_observation_variance25_predictions.json.gz','S9a_elasticity_half_predictions.json.gz'])
    for name in names:
        a={r['series_id']:r for r in read(OLD/name)};b={r['series_id']:r for r in read(RESULTS/name)}
        record_changes[name]=dict(removed=sorted(a.keys()-b.keys()),added=sorted(b.keys()-a.keys()),
            changed={k:[dict(path=p,old=x,new=y) for p,x,y in differences(a[k],b[k])]
                     for k in sorted(a.keys()&b.keys()) if a[k]!=b[k]})
    save(RESULTS/'fix1_record_changes.json.gz',record_changes)
    figure_changes=[]
    for path in sorted((HERE/'figures').glob('*.csv')):
        a=list(csv.DictReader((OLD/'figures'/path.name).open())) if (OLD/'figures'/path.name).exists() else []
        b=list(csv.DictReader(path.open()))
        fields=list(b[0] if b else a[0]) if a or b else []
        keys=[k for k in ['series_id','taxon','interval','lag','cycle','metric','label','x'] if k in fields]
        identity=lambda r,i:' / '.join(r[k] for k in keys) if keys else str(i)
        aa={identity(r,i):r for i,r in enumerate(a)};bb={identity(r,i):r for i,r in enumerate(b)}
        figure_changes.extend(dict(figure_table=path.name,path=k,old=x,new=y) for k,x,y in differences(aa,bb))
    save(RESULTS/'fix1_figure_changes.json',figure_changes)
    lines=['\n\n'+TITLE+'\n']
    def add(x=''):lines.append(x)
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
        for row in rows:add('| '+' | '.join(str(x).replace('|',r'\|') for x in row)+' |')
        add()
    def stat(d):
        if not d or d.get('estimate') is None:return '—'
        return f"{d['estimate']:.6f} [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]"
    def count(g):return f"{g.get('M',0)} / {g.get('K',0)}"
    def scalar(v):return '—' if v is None else f'{v:.6f}'
    add('These are the **corrected registered results**. They implement the registered BioTIME seasonal cutoff and physical weather lag. All earlier affected results are superseded; the earlier report text and correction notes remain unchanged as a dated record. Old outputs, figures and original code are in `results/before_fix1/`. Raw-derived trajectories and source ledgers are preserved under `data/raw/lpd/derived/before_fix1/`, without exporting raw abundance into results.')
    add()
    add('**Verified [read/run]:** `protocol.md`, the replication at `verification/replication/lpd/replication.md`, the supplied raw BioTIME zip and weather archive, `test_fix1.py`, `numerics.py`, the phases in `results/fix1_run_log.json`, and `verify_fix1.py`. No result below relies on recalled literature or on opening `sources/`. This supplied study rules ends at rule 12; rule 13 is not present. The initial board was absent in this partial copy.')
    add()
    add('The proposed `lpd.diff` was checked and applied with two safeguards: GHCN station eligibility still requires 15 **current** weather windows, and a predecessor uses the full host grid (including zeros), or the preceding physical interval when that grid step is absent. Published GPDD representative periods are retained. First observations may use pre-series weather. Missing weather remains missing. Bootstrap counts, seeds, eligibility thresholds and decision rules are unchanged. All series-ID checkpoints were invalidated. The only reused calculations are pre-analysis surfaces and engine cells keyed by their numerical parameters; the primary direct-refinement phase was rerun.')
    add()
    if labels:
        add('**Registered labels changed.** The complete list, including subgroup and weather-adjusted strata, is:')
        add();table(['Analysis','Stratum','Test','Old','Corrected'],[[r[k] for k in ['scope','stratum','test','old','new']] for r in labels])
        add('For `taxon_subgroup:small_mammals`, the primary count changes from 50 series / 5 clusters to 34 / 4; the paired and weather-adjusted subgroup changes from 47 / 5 to 34 / 4. Its positive intervals and median memory still meet the sign conditions, but four clusters fail the registered minimum of five. Hence these labels are **not decided**. The encompassing `taxon_group:small_mammals` and all primary taxon × annual/subannual and detailed-interval labels retain their earlier decisions. No label changes to contradicts.');add()
    else:add('**No registered label changed**, overall or in any primary, subgroup, sensitivity or weather-adjusted stratum.');add()
    add('Overall T1 and T2: **'+newsummary['overall']['T1']+' / '+newsummary['overall']['T2']+'**. The tables below expose every broad and detailed taxon × interval decision, with counts and intervals. `M` is series count, `K` is independent study/source clusters. T1 requires M ≥ 25 and K ≥ 5, a strictly positive 95% interval for mean excess correlation Δr1, and positive median corrected memory. Empty cells are not decided. T2 uses the registered fraction thresholds; it has no additional M/K cutoff in the registered rule.')
    add('\n### Eligibility and binning\n')
    table(['Quantity','Old','Corrected'],[
        ['Primary series / clusters',count(oldsummary['overall']),count(newsummary['overall'])],
        ['Candidate trajectories, including fixed plots',len(oldcan),len(newcan)],
        ['Study bin counts',repr_value(bincounts['old_study_bins']),repr_value(bincounts['new_study_bins'])],
        ['Primary interval counts',repr_value(bincounts['old_primary']['interval_class']),repr_value(bincounts['new_primary']['interval_class'])],
        ['Primary database counts',repr_value(bincounts['old_primary']['database']),repr_value(bincounts['new_primary']['database'])],
        ['Primary taxon counts',repr_value(bincounts['old_primary']['taxon_group']),repr_value(bincounts['new_primary']['taxon_group'])]])
    add(f'{len(changed_studies)} studies move from annual to seasonal bins at the extraction stage. All {len(removed)} removed primary series belong to these re-binned studies; {len(added)} primary series are newly eligible. They are rebuilt from assemblage events and re-tested, not simply transferred to a new stratum. The old note saying raw events were absent is superseded for this supplied copy. Exact membership and all candidate/primary counts are in `results/fix1_binning_counts.json`.')
    add()
    table(['BioTIME study','Old candidate M/K','Corrected candidate M/K','Old primary M/K','Corrected primary M/K'],
          [[sid,*[count(v[k]) for k in ['old_candidates','new_candidates','old_primary','new_primary']]] for sid,v in bincounts['study_counts'].items()])
    add('\n### T1 and T2: complete primary comparisons\n')
    def hypothesis_table(a,b,keys):
        table(['Cell','Old M/K','Corrected M/K','Old Δr1 [95% CI]','Corrected Δr1 [95% CI]','Old median memory','Corrected median memory','Old T1 → corrected T1'],
          [[k,count(a.get(k,{})),count(b.get(k,{})),stat(a.get(k,{}).get('mean_delta_r1')),stat(b.get(k,{}).get('mean_delta_r1')),
            scalar(a.get(k,{}).get('median_r1_bc')),scalar(b.get(k,{}).get('median_r1_bc')),
            a.get(k,{}).get('T1','not decided')+' → '+b.get(k,{}).get('T1','not decided')] for k in keys])
        table(['Cell','Old long failure [95% CI]','Corrected long failure [95% CI]','Old memory zone [95% CI]','Corrected memory zone [95% CI]','Old T2 → corrected T2'],
          [[k,stat(a.get(k,{}).get('fractions',{}).get('long_strict_failure')),stat(b.get(k,{}).get('fractions',{}).get('long_strict_failure')),
            stat(a.get(k,{}).get('fractions',{}).get('short_penalty_zone_T2')),stat(b.get(k,{}).get('fractions',{}).get('short_penalty_zone_T2')),
            a.get(k,{}).get('T2','not decided')+' → '+b.get(k,{}).get('T2','not decided')] for k in keys])
    primarykeys=['overall','CC_BY_only']+sorted((oldsummary.keys()|newsummary.keys())-{'overall','CC_BY_only'})
    # Include zero-series combinations instead of silently omitting taxa/intervals.
    for tax in sorted({r['taxon_group'] for r in oldrows+newrows}):
        for kind,ivs in [('taxon_interval',['annual','subannual']),('taxon_detailed',['annual','subannual_seasonal','subannual_monthly','subannual_biweekly_weekly'])]:
            for iv in ivs:
                key=kind+':'+tax+':'+iv
                if key not in primarykeys:primarykeys.append(key)
    hypothesis_table(oldsummary,newsummary,primarykeys)
    add('Fractions are proportions (multiply by 100 for percent). The long-failure fraction uses direct primary coarse-cycle losses; the short-stage memory zone is corrected memory > 0.1919. Broad annual and detailed annual cells have distinct label-derived bootstrap seeds, retained from the original scripts; their finite-bootstrap endpoints can differ despite identical members.')
    add('\n### Weather amendment and adjusted comparisons\n')
    add('The old column is the archived final report on its original weather coverage; the corrected column uses the supplied expanded weather archive. No weather was fetched during fix1. Thus weather differences combine the two code corrections with the already expanded archive; they are not attributed wholly to the lag fix. The earlier replication’s 1,765 matches applied the physical-lag correction before the seasonal-binning correction and is not the corrected joint sample here. W1 linear adjustment, W2 associations and the executed W3 thermal summaries were all rerun. This correction does not claim completion of the separate, unfinished R27 quadratic/VPD/engine extensions.')
    control=read(RESULTS/'fix1_weather_patch_comparison.json')
    add('Patch verification on the same corrected host sample and current weather archive gave '+str(control['old_lookup_corrected_bins']['n_matched'])+' matches with the old lookup, '+str(control['proposed_patch']['n_matched'])+' with `lpd.diff` alone, and '+str(control['final_fix1']['n_matched'])+' with the reviewed correction. The unmodified proposal prematurely selects some GHCN stations by counting current plus lag windows toward the 15-current-window minimum. Requiring 15 current windows recovers 14 matches from later qualifying stations. `results/fix1_weather_patch_comparison.json` records every membership and adjusted count; the synthetic regression test also catches this failure.')
    add()
    table(['Coverage','Old','Corrected'],[[k,repr_value(oldweather['coverage'].get(k)),repr_value(newweather['coverage'].get(k))] for k in sorted(oldweather['coverage'].keys()|newweather['coverage'].keys())])
    weatherkeys=sorted(oldweather['W1'].keys()|newweather['W1'].keys())
    hypothesis_table(oldweather['W1'],newweather['W1'],weatherkeys)
    add('Old weather taxon × interval and subgroup rows not originally tabulated were computed here from the archived old predictions, using the same labels, 1,000 study-cluster draws and seed rule as the corrected rows. They do not replace the archived old values. Full precision, denominators, all W2 correlations and W3 summaries are compared below.')
    add('\n### Sensitivities, descriptive tables and figures\n')
    hypothesis_table(*pairs['sensitivity'],sorted(pairs['sensitivity'][0].keys()|pairs['sensitivity'][1].keys()))
    add('All four figures were regenerated and compiled: `figures/memory_map.pdf`, `acf_by_interval.pdf`, `regime_fractions.pdf`, and `sensitivity_fractions.pdf`; their plotting CSVs and TeX are current. The old versions are in `results/before_fix1/figures/`. Exact changed plotted cells, including removed points, are in `results/fix1_figure_changes.json`. `memory_regimes.json`, `series_metrics.csv`, both report-number exports and the field inventory were rebuilt. Detailed per-series metric/prediction changes and removed IDs are in `results/fix1_record_changes.json.gz`; these contain derived results, not original abundance.')
    add('Run departure: the default Tectonic network client failed in the sandbox. `compile_fix1_figures.py` compiled the same TeX offline using unchanged resources in the existing local cache; `results/fix1_figure_compile_log.json` records the bundle and successful commands. Failed attempt logs are retained. This changes no plotted data.')
    add('\n### Every changed aggregate number, side by side\n')
    add('The following tables are exhaustive for changed aggregate outputs: eligibility/extraction, every summary distribution and quantile/interval, regime fractions, sensitivities, weather W1–W3, climate correlations, source/date coverage, inventories and numerical audits. JSON-shaped cells retain full saved precision, denominators and interval endpoints. `null` means absent or unavailable, never zero. Unchanged quantities are omitted here; the hypothesis tables above show unchanged labels too. Machine-readable copies are `results/fix1_aggregate_changes.csv` and `.json`. Dates/hashes in audit comparisons document provenance rather than ecological effects.')
    add()
    for name in comparisons:
        rr=[r for r in changes if r['output']==name]
        if not rr:continue
        add('<details>');add(f'<summary>{name}: {len(rr)} changed entries</summary>');add()
        table(['Statistic path','Old','Corrected'],[[r['path'],repr_value(r['old']),repr_value(r['new'])] for r in rr])
        add('</details>');add()
    add('\n### Verification and reproduction\n')
    audit=read(RESULTS/'fix1_independent_audit.json')
    add(f'`test_fix1.py` passed all 9 regression tests, including an independently solved synthetic least-squares fit (tolerance 1e-10). All seven existing known-result checks in `numerics.py` passed their predeclared tolerances. `verify_fix1.py` independently checked {audit["primary_series"]} primary core estimates, {audit["weather_series"]} weather regression designs, and {audit["checks"]["summary_labels"]} summary cells using the replication’s independent estimators. Estimates and bootstrap endpoints agreed within the predeclared tolerances; labels agreed exactly. `results/fix1_independent_audit.json` gives the observed errors. All {audit["archive_hashes"]} archived-file hashes passed; the original report prefix ({audit["report_prefix_bytes"]} bytes) and the whole protocol are preserved. `results/final_audit.json` records the input-manifest checks. No new fieldwork, hypotheses or model-fitting was introduced.')
    add()
    add('Commands: run `python3 -I casestudies/lpd/test_fix1.py`, `numerics.py`, `run_fix1.py`, `verify_fix1.py`, then `append_fix1_report.py` in the same study directory path from the repository root. The report appender refuses a second append. Before recomputing with changed extraction code, archive and invalidate all data-dependent checkpoints as in `fix1_plan.md`; parameter-keyed engine caches can remain. Logs record each actual command and completion. The changes retain the original observational and numerical limitations: supported host-side premises are not measured parasite effects.')
    add()
    final='none' if not labels else '; '.join(f"{r['scope']} {r['stratum']} {r['test']} ({r['old']} -> {r['new']})" for r in labels)
    add('fix1: labels changed: '+final)
    text='\n'.join(lines)+'\n'
    (RESULTS/'fix1_report_output.txt').write_text(text)
    with report.open('ab') as f:f.write(text.encode())
    assert report.read_bytes().startswith(prefix)
    print('Appended',len(text),'characters;',len(changes),'aggregate changes;',len(labels),'label changes')
    print('fix1: labels changed: '+final)

if __name__=='__main__':run()
