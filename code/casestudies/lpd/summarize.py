"""Clustered uncertainty and literal registered decisions, with explicit denominators."""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collections,math,hashlib
import numpy as np
from scipy.stats import spearmanr
from study_io import *
from analysis import map_one,CYCLES

DIST=['sigma2','sigma','sigma2_raw','r1','r1_bc','delta_r1','b_spec','excess_reddening','n']+['acf_'+str(i) for i in range(1,11)]+[c+'_D' for c in CYCLES]

def ci(x):
    x=np.array(x,float);x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975]).tolist() if len(x) else [None,None]
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)

def summarize(rows,label,distributions=True):
    if not rows:return dict(label=label,M=0,K=0,T1='not decided',T2='not decided')
    groups=sorted({r['study_id'] for r in rows});K=len(groups);M=len(rows);gid={s:i for i,s in enumerate(groups)};g=np.array([gid[r['study_id']] for r in rows]);rng=np.random.default_rng(int(hashlib.sha256(label.encode()).hexdigest()[:8],16));draw=rng.multinomial(K,np.full(K,1/K),size=1000)
    sums={};out=dict(label=label,M=M,K=K,distributions={},fractions={})
    bools=sorted({k for r in rows for k,v in r.items() if isinstance(v,bool) and (k.startswith(tuple(CYCLES)) or k.startswith('S9_'))})
    for key in ['delta_r1']+bools:
        x=np.array([float(r.get(key)) if finite(r.get(key)) else np.nan for r in rows]);ok=np.isfinite(x);ng=np.bincount(g,weights=ok,minlength=K);sx=np.bincount(g,weights=np.where(ok,x,0),minlength=K);den=draw@ng;boot=np.divide(draw@sx,den,out=np.full(1000,np.nan),where=den>0);d=dict(estimate=float(x[ok].mean()) if ok.any() else None,ci95=ci(boot),denominator=int(ok.sum()),eligible_clusters=int((ng>0).sum()))
        if key=='delta_r1':out['mean_delta_r1']=d
        else:out['fractions'][key]=d
    med=float(np.median([r['r1_bc'] for r in rows]));out['median_r1_bc']=med
    lo,hi=out['mean_delta_r1']['ci95'];out['T1']='not decided'
    if K>=5 and M>=25:
        if lo>0 and med>0:out['T1']='supports'
        elif hi<0:out['T1']='contradicts'
    a=out['fractions'].get('long_strict_failure');b=out['fractions'].get('short_penalty_zone_T2');out['T2']='not decided'
    if a and b:
        if a['ci95'][0]>.20 or b['ci95'][0]>.20:out['T2']='supports'
        elif a['ci95'][1]<.10 and b['ci95'][1]<.10:out['T2']='contradicts'
    if distributions:
        # Cluster multiplicities preserve all within-study dependence; quantiles are series-weighted.
        for key in DIST:
            x=np.array([r.get(key) if finite(r.get(key)) else np.nan for r in rows],float);ok=np.isfinite(x)
            if not ok.any():continue
            order=np.argsort(x[ok]);xs=x[ok][order];gs=g[ok][order];qs=[.1,.25,.5,.75,.9];boots=[]
            # Inverse weighted empirical CDF; use same definition for estimates and resamples.
            for start in range(0,1000,100):
                w=draw[start:start+100,gs];cum=np.cumsum(w,axis=1);tot=cum[:,-1];vals=np.empty((len(w),len(qs)))
                for j,q in enumerate(qs):
                    ix=np.argmax(cum>=q*tot[:,None],axis=1);vals[:,j]=xs[ix];vals[tot==0,j]=np.nan
                boots.extend(vals.tolist())
            boots=np.array(boots);out['distributions'][key]=dict(n=int(ok.sum()),mean=float(x[ok].mean()),quantiles={str(q):dict(estimate=float(np.quantile(xs,q,method='inverted_cdf')),ci95=ci(boots[:,j])) for j,q in enumerate(qs)})
    return out

def grouping(rows):
    groups={'overall':rows,'CC_BY_only':[r for r in rows if r['database'] in ['BioTIME','GPDD'] and r['cc_by_compatible']]}
    for key in ['database','taxon_group','interval_class','taxon_subgroup','latitude_band','realm_detail']:
        for v in sorted({r.get(key,'unknown') for r in rows}):groups[key+':'+v]=[r for r in rows if r.get(key,'unknown')==v]
    for r in rows:
        broad='annual' if r['interval_class']=='annual' else 'subannual'
        groups.setdefault('taxon_interval:'+r['taxon_group']+':'+broad,[]).append(r)
        groups.setdefault('taxon_detailed:'+r['taxon_group']+':'+r['interval_class'],[]).append(r)
    return groups

def correlation(rows,x,y,label):
    rr=[r for r in rows if finite(r.get(x)) and finite(r.get(y))];K=len({r['study_id'] for r in rr})
    if len(rr)<5 or K<2:return dict(label=label,n=len(rr),K=K,rho=None,ci95=[None,None])
    arr=np.array([[r[x],r[y]] for r in rr]);cl=sorted({r['study_id'] for r in rr});ix=[np.array([j for j,r in enumerate(rr) if r['study_id']==c]) for c in cl];rng=np.random.default_rng(777);boot=[]
    for _ in range(1000):
        jj=np.concatenate([ix[k] for k in rng.integers(0,K,K)]);a=arr[jj]
        if np.ptp(a[:,0])==0 or np.ptp(a[:,1])==0:continue
        boot.append(float(spearmanr(a[:,0],a[:,1]).statistic))
    return dict(label=label,n=len(rr),K=K,rho=float(spearmanr(arr[:,0],arr[:,1]).statistic) if np.ptp(arr[:,0])>0 and np.ptp(arr[:,1])>0 else None,ci95=ci(boot))

def run():
    primary=read(RESULTS/'primary_predictions.json.gz');summaries={}
    for label,rr in grouping(primary).items():
        summaries[label]=summarize(rr,label);print(now(),label,len(rr),summaries[label]['T1'],summaries[label]['T2'],flush=True)
    save(RESULTS/'summary_tables.json',summaries)
    if '--primary-only' in sys.argv:return
    # Additional registered sensitivities to transform theoretical inputs only.
    from numerics import Surfaces
    sf=Surfaces();sens={}
    for label in ['S1_zero','S2_season','S3_no_detrend','S6_quality_fixedplot','S7_coverage50']:
        rr=read(RESULTS/(label+'_predictions.json.gz'));sens[label]=summarize(rr,label,distributions=False)
        if label=='S2_season':sens[label+'_subannual']=summarize([r for r in rr if r['interval_class']!='annual'],label+'_subannual',False)
    sens['S5_high_abundance']=summarize([r for r in primary if r['median_abundance']>=25 and r['minimum_positive_abundance']>=5],'S5_high_abundance',False)
    for label in ['S8b_observation_variance25','S9a_elasticity_half']:
        rr=[]
        for i,m in enumerate(primary):
            n={k:v for k,v in m.items() if not k.startswith('S9_')};n['sigma']=m['sigma']*(math.sqrt(.75) if label.startswith('S8') else .5);n['sigma2']=n['sigma']**2
            if label.startswith('S8') and m['r1']>.15 and finite(m.get('acf_2')):n['r1_bc']=float(np.clip(m['acf_2']/m['r1'],0,.98))
            rr.append(map_one(n,sf))
        save(RESULTS/(label+'_predictions.json.gz'),rr);sens[label]=summarize(rr,label,False)
    # S4 is a coefficient comparison, never an unexpanded accuracy test.
    s4={}
    for c in CYCLES:
        rr=[dict(r,acf_sign_changed=(r[c+'_D2']*r[c+'_D2_full_acf']<0),acf_minus_ar1=r[c+'_D2_full_acf']-r[c+'_D2']) for r in primary]
        s4[c]=dict(n=len(rr),sign_changes=sum(r['acf_sign_changed'] for r in rr),median_difference=float(np.median([r['acf_minus_ar1'] for r in rr])),difference_quantiles=np.quantile([r['acf_minus_ar1'] for r in rr],[.1,.9]).tolist())
    sf.flush();save(RESULTS/'sensitivity_tables.json',sens);save(RESULTS/'full_acf_sensitivity.json',s4)
    corr=[]
    for unit in sorted({r.get('units','') for r in primary}):
        rr=[r for r in primary if r.get('units','')==unit]
        if len(rr)>=25:
            for key in ['sigma','r1_bc']:corr.append(correlation(rr,'mean_log_abundance',key,unit+':'+key))
    save(RESULTS/'mean_abundance_correlations.json',corr)
    manifest=dict(study_id='lpd',generated_utc=now(),schema_version='1.0',description='Empirical detrended log-abundance regimes. Variance and memory are descriptive estimates, with moving-block SE. Model losses are illustrative, not parasite measurements. Cluster by study_id. A coarse step equals one recorded sampling interval; fine-grid assumptions cannot be identified from these observations.',n_series=len(primary),records=primary)
    save(RESULTS/'memory_regimes.json',manifest)
    print(now(),'summary complete',flush=True)
if __name__=='__main__':run()
