"""Independent numerical audit of the corrected registered run.

Tolerances declared before execution: primary core metrics 1e-10 absolute or
relative; weather OLS metrics 1e-9 absolute or relative; bootstrap estimates and
endpoints 1e-12. Dates, memberships, labels and archived bytes must match exactly.
Uses the second party's independently written estimators without editing them.
"""
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collections,csv,hashlib,math
import numpy as np
from study_io import *
from analysis import candidates
from summarize import grouping
sys.path.insert(0,str(HERE.parents[1]/'verification/replication/lpd'))
import core_math as independent


def run():
    old=RESULTS/'before_fix1';ss={s['series_id']:s for s in candidates()}
    errors=collections.defaultdict(float);counts=collections.Counter()
    def close(key,a,b,tol):
        if a is None or b is None:
            assert a is None and b is None,(key,a,b);return
        err=abs(a-b);errors[key]=max(errors[key],err);counts[key]+=1
        assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(key,a,b,tol)
    rows=read(RESULTS/'primary_predictions.json.gz')
    for r in rows:
        s=ss[r['series_id']];mask=s['values']>0;t=s['steps'][mask].astype(float);y=np.log(s['values'][mask])
        assert len(t)>=15 and len(t)/(t[-1]-t[0]+1)>=.75 and (np.diff(t)==1).sum()>=12
        m=independent.estimate_series_quantities(t,y)
        for key in ['sigma2','sigma2_raw','r1','b_spec']:
            close('primary_'+key,r[key],m[key],1e-10)
        close('primary_delta_r1',r['delta_r1'],m['r1']-independent.wn_null_r1_linear(t),1e-10)
    print('Independent primary core metrics:',len(rows),flush=True)
    # Rebuild the W1 design independently from the weather ledger and raw host data.
    ledger=collections.defaultdict(dict)
    with (RESULTS/'weather_record_matches.csv').open() as f:
        for r in csv.DictReader(f):ledger[r['series_id']][(int(r['host_step']),r['weather_role'])]=r
    matched=[r for r in read(RESULTS/'weather_metrics.json') if r['matched']]
    for w in matched:
        s=ss[w['series_id']];records=ledger[w['series_id']]
        steps=[int(t) for t,v in zip(s['steps'],s['values']) if v>0 and (int(t),'current') in records and (int(t),'lag') in records]
        assert len(steps)==w['n_adjusted']
        pos={int(t):i for i,t in enumerate(s['steps'])};t=np.array(steps,float)
        y=np.log([s['values'][pos[k]] for k in steps]);tc=[];tl=[];p=[]
        for k in steps:
            c=records[k,'current'];l=records[k,'lag'];assert int(l['step'])==k-1
            tc.append(float(c['temperature_c']));tl.append(float(l['temperature_c']))
            if w['precipitation_in_design']:p.append(float(c['precipitation_mm']))
        cols=[np.ones(len(t)),t-t.mean(),np.array(tc)-np.mean(tc),np.array(tl)-np.mean(tl)]
        if p:cols.append(np.array(p)-np.mean(p))
        X=np.column_stack(cols);Q=independent.orthonormal_columns(X)
        e=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
        m=independent.estimate_series_quantities(t,e)
        close('weather_sigma2',w['residual_metrics']['sigma2'],e@e/(len(t)-Q.shape[1]),1e-9)
        close('weather_r1',w['residual_metrics']['r1'],m['r1'],1e-9)
        close('weather_delta_r1',w['residual_metrics']['delta_r1'],m['r1']-independent.wn_null_r1_projected(t,Q),1e-9)
        close('weather_r1_bc',w['residual_metrics']['r1_bc'],independent.calibrate_ar1_phi(t,m['r1'],Q),1e-9)
    print('Independent weather OLS designs:',len(matched),flush=True)
    def verify_summary(rr,s):
        if not rr:return
        seed=int(hashlib.sha256(s['label'].encode()).hexdigest()[:8],16)
        chk=independent.cluster_bootstrap_summary(rr,['long_strict_failure','short_penalty_zone_T2'],seed=seed)
        assert (chk['M'],chk['K'],chk['T1'],chk['T2'])==(s['M'],s['K'],s['T1'],s['T2']),s['label']
        for key in ['mean_delta_r1']:
            for metric in ['estimate']:
                close('bootstrap_'+metric,chk[key][metric],s[key][metric],1e-12)
            for a,b in zip(chk[key]['ci95'],s[key]['ci95']):close('bootstrap_ci',a,b,1e-12)
        for key in ['long_strict_failure','short_penalty_zone_T2']:
            close('bootstrap_fraction',chk['fractions'][key]['estimate'],s['fractions'][key]['estimate'],1e-12)
            for a,b in zip(chk['fractions'][key]['ci95'],s['fractions'][key]['ci95']):close('bootstrap_fraction_ci',a,b,1e-12)
        counts['summary_labels']+=1
    summaries=read(RESULTS/'summary_tables.json')
    for label,rr in grouping(rows).items():verify_summary(rr,summaries[label])
    weather=read(RESULTS/'weather_summary.json')['W1']
    for mode in ['weather_paired','weather_adjusted']:
        rr=read(RESULTS/(mode+'_predictions.json.gz'));gg=grouping(rr)
        for key,s in weather.items():
            if key==mode:group=rr
            elif key.startswith(mode+':'):group=gg[key.removeprefix(mode+':')]
            elif key==mode+'_annual':group=[r for r in rr if r['interval_class']=='annual']
            elif key==mode+'_subannual':group=[r for r in rr if r['interval_class']!='annual']
            elif key.startswith(mode+'_'):group=[r for r in rr if r['taxon_group']==key.removeprefix(mode+'_')]
            else:continue
            verify_summary(group,s)
    for label,s in read(RESULTS/'sensitivity_tables.json').items():
        if label=='S5_high_abundance':rr=[r for r in rows if r['median_abundance']>=25 and r['minimum_positive_abundance']>=5]
        elif label=='S2_season_subannual':rr=[r for r in read(RESULTS/'S2_season_predictions.json.gz') if r['interval_class']!='annual']
        else:rr=read(RESULTS/(label+'_predictions.json.gz'))
        verify_summary(rr,s)
    manifest=read(old/'preservation_manifest.json')
    for name,digest in manifest.items():
        path=(old/name.removeprefix('results/')) if name.startswith('results/') else old/name if name.startswith('figures/') else old/'original_files'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    prefix=(old/'original_files/report.md').read_bytes()
    assert (HERE/'report.md').read_bytes().startswith(prefix)
    assert (HERE/'protocol.md').read_bytes()==(old/'original_files/protocol.md').read_bytes()
    save(RESULTS/'fix1_independent_audit.json',dict(generated_utc=now(),passed=True,
        primary_series=len(rows),weather_series=len(matched),checks=dict(counts),
        maximum_absolute_errors=dict(errors),archive_hashes=len(manifest),
        report_prefix_bytes=len(prefix),protocol_unchanged=True))
    print('Independent audit passed',dict(counts),dict(errors),flush=True)

if __name__=='__main__':run()
