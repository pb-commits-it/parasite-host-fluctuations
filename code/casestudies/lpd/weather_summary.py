"""Registered W1–W3 summaries; parasite outcomes here are predictions only."""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collections,math
import numpy as np
from study_io import *
from analysis import map_one,CYCLES
from summarize import summarize,correlation,finite,grouping
from numerics import Surfaces

def run():
    sf=Surfaces('weather_direct_engine_predictions.json');mapped={}
    for mode in ['weather_paired','weather_adjusted']:
        dest=RESULTS/(mode+'_predictions.json.gz')
        if dest.exists():mapped[mode]=read(dest);continue
        rows=read(RESULTS/(mode+'_metrics.json'));arr=[]
        for i,m in enumerate(rows):
            arr.append(map_one(m,sf))
            if i%300==0:print(now(),mode,i,len(rows),flush=True)
        save(dest,arr);mapped[mode]=arr;sf.flush()
    out={'generated_utc':now(),'W1':{}}
    for mode,rows in mapped.items():
        out['W1'][mode]=summarize(rows,mode,False)
        for iv in ['annual','subannual']:
            rr=[r for r in rows if (r['interval_class']=='annual')==(iv=='annual')];out['W1'][mode+'_'+iv]=summarize(rr,mode+'_'+iv,False)
        for tax in sorted({r['taxon_group'] for r in rows}):out['W1'][mode+'_'+tax]=summarize([r for r in rows if r['taxon_group']==tax],mode+'_'+tax,False)
        # The amendment's decision check includes every taxon x interval cell.
        for label,rr in grouping(rows).items():
            if label.startswith(('taxon_interval:','taxon_detailed:','taxon_subgroup:')):
                key=mode+':'+label
                out['W1'][key]=summarize(rr,key,False)
    weather=read(RESULTS/'weather_metrics.json');baseline={r['series_id']:r for r in mapped['weather_paired']};rr=[]
    for w in weather:
        if not w['matched']:continue
        r=baseline[w['series_id']].copy();r.update({k:v for k,v in w.items() if k not in ['residual_metrics','paired_host_metrics']});rr.append(r)
    out['coverage']=dict(primary_series=len(weather),matched_series=len(rr),matched_clusters=len({r['study_id'] for r in rr}),by_source=dict(collections.Counter(w.get('source','unknown') for w in weather if w['matched'])),unmatched_reasons=dict(collections.Counter(w.get('reason','unknown') for w in weather if not w['matched'])))
    out['W2']=[]
    strata={'overall':rr}
    for r in rr:
        key=r['realm']+':'+('annual' if r['interval_class']=='annual' else 'subannual');strata.setdefault(key,[]).append(r)
    for label,group in strata.items():
        for x in ['mean_temp_c','sigma_temp_c','r1_bc_temp','mean_prcp_mm','sigma_prcp_mm']:
            for y in ['sigma','r1_bc','long_D','moderate_D','short_D']:
                if label!='overall' and x in ['mean_prcp_mm','sigma_prcp_mm']:continue
                out['W2'].append(correlation(group,x,y,label+':'+x+':'+y))
    out['weather_distribution']={}
    for key in ['mean_temp_c','sigma_temp_c','r1_bc_temp','corr_host_temp','corr_host_prcp','r2_weather_explained','distance_km']:
        a=[r[key] for r in rr if finite(r.get(key))];out['weather_distribution'][key]=dict(n=len(a),median=float(np.median(a)) if a else None,q10_q90=np.quantile(a,[.1,.9]).tolist() if a else None)
    out['W3a']={}
    for q in ['2.0','2.5','3.0']:
        out['W3a'][q]={}
        for band,group in {'all':rr,**{b:[r for r in rr if r['latitude_band']==b] for b in ['tropical','temperate','boreal_polar']}}.items():
            if not group:continue
            aa=[r['thermal_duration'][q]['a'] for r in group];good=[r for r in group if r['thermal_duration'][q]['admissible_coarse']]
            out['W3a'][q][band]=dict(n=len(group),admissible=len(good),median_a=float(np.median(aa)),median_short_ratio=float(np.median([2.3/a for a in aa])),short_penalty_count=sum(bool(r['thermal_duration'][q]['short_loss_penalty']) for r in good))
    out['W3b']={}
    for c in CYCLES:
        for sign in ['same','opposite']:
            key='D2_thermal_'+sign+'_'+c;vals=np.array([r[key] for r in rr]);out['W3b'][key]=dict(n=len(vals),median=float(np.median(vals)),q10_q90=np.quantile(vals,[.1,.9]).tolist(),negative_count=int((vals<0).sum()),larger_magnitude_than_host_count=sum(abs(r[key])>abs(r[c+'_D']) for r in rr))
    save(RESULTS/'weather_summary.json',out);print(now(),'WEATHER RESULTS',out['coverage'],{k:v['T1'] for k,v in out['W1'].items()},flush=True)
if __name__=='__main__':run()
