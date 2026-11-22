#!/usr/bin/env python3
"""Registered host-memory analysis, continuation of lpd. Run with python3 -I.
Known-result checks in numerics.py must pass before empirical computation.
No test criterion or illustrative parameter is fitted to the host observations.
Phases checkpoint extraction, per-series metrics, predictions, summaries separately.
"""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,collections,math,functools,hashlib
import numpy as np
from scipy.stats import spearmanr
import helpers_continued as h
from study_io import *

CYCLES={'long':24.,'moderate':7.5,'short':2.3}

def candidates():
    ss=read(RAW/'derived/candidate_series.json.gz')
    for s in ss:
        for k in ['steps','values','subperiod','step_years','step_months']:s[k]=np.array(s[k])
        for k in ['latitude','longitude']:s[k]=float(s[k]) if s[k] is not None else float('nan')
    return ss

def eligible(s,mode='primary'):
    val=s['values'];t=s['steps'];pos=np.isfinite(val)&(val>0)
    mask=np.isfinite(val)&(val>=0) if mode=='S1_zero' else pos
    if pos.sum()<15:return None,'fewer_than_15_positive'
    if mode!='S1_zero' and pos.sum()<.5*len(val):return None,'positive_fraction'
    tt=t[mask].astype(float);vv=val[mask].astype(float)
    coverage=len(tt)/(tt[-1]-tt[0]+1)
    if coverage<(.5 if mode=='S7_coverage50' else .75):return None,'grid_coverage'
    if (np.diff(tt)==1).sum()<12:return None,'fewer_than_12_adjacent_pairs'
    yy=np.log(vv+.5*min(val[pos])) if mode=='S1_zero' else np.log(vv)
    if np.var(yy)<=1e-10:return None,'constant'
    return (tt,yy,mask,coverage),None

@functools.lru_cache(maxsize=6000)
def curve(t):return h.ar1_detrended_expected_r1_curve(np.array(t,float))
@functools.lru_cache(maxsize=6000)
def specnull(t):return h.white_noise_spec_null_mean(np.array(t,float))

def metadata(s):
    omit={'steps','values','subperiod','step_years','step_months','step_dates_iso'}
    return {k:v for k,v in s.items() if k not in omit}

def metric(s,mode='primary',bootstrap=False):
    e,why=eligible(s,mode)
    if e is None:return None,why
    t,y,mask,cov=e;detrend=mode!='S3_no_detrend';Q=None
    if mode=='S2_season' and s['interval_class']!='annual':
        seasons=s['subperiod'][mask];levels=np.unique(seasons)
        X=np.column_stack([np.ones(len(t)),t-t.mean()]+[(seasons==v).astype(float) for v in levels[1:]])
        Q=h.orthonormal_basis_from_design(X)
        if len(t)-Q.shape[1]<8:return None,'seasonal_design_insufficient_df'
        y=y-Q@(Q.T@y)
    m=h.compute_core_metrics(t,y,detrend=detrend)
    if m['r1'] is None or m['sigma2']<=1e-10:return None,'zero_residual_variance'
    tt=tuple((t-t[0]).astype(int))
    if Q is not None:
        null=h.projected_white_noise_r1_null_mean(t,Q);bc=h.invert_projected_bias_corrected_r1(t,Q,m['r1'])
        m['sigma2']*= (len(t)-2)/(len(t)-Q.shape[1]);m['sigma']=math.sqrt(m['sigma2'])
    elif detrend:
        null=h.white_noise_r1_null_mean(t);bc=float(np.interp(m['r1'],curve(tt),h._PHI_GRID))
    else:
        Q=h.orthonormal_basis_from_design(np.ones((len(t),1)));null=h.projected_white_noise_r1_null_mean(t,Q);bc=h.invert_projected_bias_corrected_r1(t,Q,m['r1'])
    m.update(metadata(s));m.update(mode=mode,coverage=cov,mean_log_abundance=float(np.mean(np.log(s['values'][s['values']>0]))),median_abundance=float(np.median(s['values'])),minimum_positive_abundance=float(min(s['values'][s['values']>0])),r1_bc=bc,mu0_r1=null,delta_r1=m['r1']-null,mu0_b_spec=specnull(tt),excess_reddening=specnull(tt)-m['b_spec'])
    lat=s['latitude'];m['latitude_band']='unknown' if not math.isfinite(lat) else 'tropical' if abs(lat)<23.5 else 'temperate' if abs(lat)<=50 else 'boreal_polar'
    years=s['step_years'][mask];valid=years[years>=1700];m['start_year']=int(min(valid)) if len(valid) else None;m['end_year']=int(max(valid)) if len(valid) else None
    m['n_adjacent_pairs']=int((np.diff(t)==1).sum())
    for lag,v in m['acf'].items():m['acf_'+str(lag)]=v
    if bootstrap:
        seed=int(hashlib.sha256(s['series_id'].encode()).hexdigest()[:8],16)
        m.update(h.series_block_bootstrap_se(t,y,np.random.default_rng(seed),200))
        # ACF sampling errors, using the same block construction, all admissible lags.
        rng=np.random.default_rng(seed);span=int(t[-1]-t[0]+1);e0=h.ols_detrend(t,y)[0];full=np.full(span,np.nan);full[(t-t[0]).astype(int)]=e0;L=max(2,int(len(t)**(1/3)));nb=math.ceil(span/L)
        starts=rng.integers(0,max(1,span-L+1),size=(200,nb));ix=(starts[:,:,None]+np.arange(L)).reshape(200,-1)[:,:span];draws=full[ix];acfs=[]
        for row in draws:
            ok=np.isfinite(row);tb=np.flatnonzero(ok).astype(float)
            if len(tb)<8:continue
            st=h.compute_core_metrics(tb,row[ok]);acfs.append([st['acf'].get(l) for l in range(1,11)])
        ac=np.array(acfs,float)
        for j in range(10):m['se_acf_'+str(j+1)]=float(np.nanstd(ac[:,j],ddof=1)) if np.isfinite(ac[:,j]).sum()>1 else None
    return m,None

def estimate():
    checks=read(RESULTS/'continuation_checks.json');assert all(c['passed'] for c in checks['checks'])
    ss=candidates();primary=[];audit=collections.Counter();checkpoint=RESULTS/'metrics_checkpoint.jsonl'
    done={}
    if checkpoint.exists():
        import json
        with checkpoint.open() as f:
            for line in f:
                if line.strip():
                    r=json.loads(line);done[r['series_id']]=r
    with checkpoint.open('a') as out:
        import json
        for i,s in enumerate(ss):
            if s['series_kind']!='primary':continue
            if s['series_id'] in done:m=done[s['series_id']];why=None
            else:
                m,why=metric(s,bootstrap=True)
                if m:out.write(json.dumps(clean(m),allow_nan=False)+'\n');out.flush()
            if m:primary.append(m)
            else:audit[why]+=1
            if i%250==0:print(now(),'primary',i,'retained',len(primary),flush=True)
    save(RESULTS/'primary_metrics.json',primary)
    csvwrite(RESULTS/'series_metrics.csv',primary)
    save(RESULTS/'eligibility_audit.json',dict(timestamp=now(),primary_retained=len(primary),exclusions=dict(audit),by_database=dict(collections.Counter(r['database'] for r in primary)),by_taxon=dict(collections.Counter(r['taxon_group'] for r in primary))))
    # Sensitivities that change the actual time series. Bootstrap uncertainty later clusters by study.
    sens={};pmap={r['series_id']:r for r in primary}
    for mode in ['S1_zero','S2_season','S3_no_detrend','S6_quality_fixedplot','S7_coverage50']:
        path=RESULTS/(mode+'_metrics.json')
        if path.exists():sens[mode]=read(path);continue
        arr=[]
        for i,s in enumerate(ss):
            if mode=='S6_quality_fixedplot':
                if s['database']=='BioTIME' and (s['series_kind']=='primary' or s['manipulated']):continue
                if s['database']=='GPDD' and s['reliability']<3:continue
            elif s['series_kind']!='primary':continue
            usemode=mode if mode!='S6_quality_fixedplot' else 'primary'
            if mode=='S2_season' and s['series_id'] not in pmap:continue
            if mode=='S2_season' and s['interval_class']=='annual':m=pmap[s['series_id']].copy()
            elif mode=='S6_quality_fixedplot' and s['series_id'] in pmap:m=pmap[s['series_id']].copy()
            else:m,_=metric(s,usemode)
            if m:m['mode']=mode;arr.append(m)
            if i%2000==0:print(now(),mode,i,len(arr),flush=True)
        save(path,arr);sens[mode]=arr
    print(now(),'metric phases complete',flush=True)

def map_one(m,surface,do_sensitivity=False):
    out=m.copy();sig=m['sigma'];phi=m['r1_bc'];acf={int(k):v for k,v in m['acf'].items()}
    for suffix,ph in [('',phi),('_plugin',float(np.clip(m['r1'],-.95,.98)))]:
        for cname,b in CYCLES.items():
            ev=surface.eval('primary_coarse_'+cname,sig,ph);D=ev['D'];D2=h.gamma0_two_stage(1,b,sig,ph)
            base=cname+suffix+'_';out[base+'D']=D;out[base+'D2']=D2;out[base+'D_se']=ev['se'];out[base+'D_method']=ev['method'];out[base+'i_half_bound']=D<=sig*sig/4
            same=D*D2>0 or (abs(D)<1e-4 and abs(D2)<1e-4)
            for frac,label in [(.30,'iii_30pct'),(.10,'iii_10pct')]:
                out[base+label]=bool(same and abs(D-D2)<=frac*abs(D))
                raw=ev['D_raw_linear'];out[base+label+'_raw_linear']=bool((raw*D2>0 or (abs(raw)<1e-4 and abs(D2)<1e-4)) and abs(raw-D2)<=frac*abs(raw))
                # Simulation-only ambiguity; finite-time bias and interpolation error are not enclosed.
                margin=abs(D-D2)-frac*abs(D);out[base+label+'_simulation_ambiguous']=abs(margin)<=4*ev['se']*(1+frac)
            for lnr in [.20,.35,1.]:
                dE=h.persistence_index_delta_E(1,b,sig,ph,lnr)
                out[base+f'i_index_P2_{lnr:.2f}']=dE>0
                out[base+f'ii_index_P2_{lnr:.2f}']=ph>0 and dE<0
            if ph>=0:
                fine=surface.eval('primary_fine_'+cname,sig,ph);one=h.one_stage_loss_quadrature(4,sig);df=h.gamma0_two_stage(4,4*b,sig,ph**.25 if ph>0 else 0)
                out[base+'fine_D']=fine['D'];out[base+'fine_D2']=df;out[base+'i_loss_fine']=fine['D']<one;out[base+'ii_loss_fine_unexpanded']=ph>0 and fine['D']>one;out[base+'ii_loss_fine_second_order']=ph>0 and df>sig*sig/8
            else:
                for k in ['fine_D','fine_D2','i_loss_fine','ii_loss_fine_unexpanded','ii_loss_fine_second_order']:out[base+k]=None
            if not suffix:out[base+'D2_full_acf']=h.gamma0_from_full_acf(1,b,sig,acf)
        out['short'+suffix+'_penalty_zone_T2']=ph>.1919
    out['long_strict_failure']=not out['long_iii_10pct'];out['long_strict_failure_raw_linear']=not out['long_iii_10pct_raw_linear']
    if do_sensitivity:
        for cname,b in CYCLES.items():
            for mode in ['bounded_coarse','chain4_fine','alt2_coarse','patches4_coarse','contact_fine']:
                ev=surface.eval(mode+'_'+cname,sig,phi,direct_outside=False)
                out['S9_'+mode+'_'+cname+'_D']=ev['D']
                out['S9_'+mode+'_'+cname+'_half_bound']=None if ev['D'] is None else ev['D']<=sig*sig/4
    return out

def predict():
    from numerics import Surfaces
    sf=Surfaces()
    paths=['primary','S1_zero','S2_season','S3_no_detrend','S6_quality_fixedplot','S7_coverage50']
    for label in paths:
        dest=RESULTS/(label+'_predictions.json.gz')
        if dest.exists():continue
        metrics=read(RESULTS/(label+'_metrics.json'));out=[]
        for i,m in enumerate(metrics):
            out.append(map_one(m,sf,do_sensitivity=label=='primary'))
            if i%250==0:print(now(),'predictions',label,i,len(metrics),flush=True)
        sf.flush();save(dest,out)
    print(now(),'predictions complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['metrics','predict']);a=p.parse_args()
    (estimate if a.phase=='metrics' else predict)()
