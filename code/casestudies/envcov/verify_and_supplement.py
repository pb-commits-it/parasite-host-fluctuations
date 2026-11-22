"""Verify completed numerics and finish pre-specified ancillary covariate checks.
No primary criterion is changed; outputs are descriptive sensitivities.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
from driven_weather import make_model
import engine_accelerator as A
from scipy import integrate,special

def main():
 started=now();d=json.loads((RESULTS/'field_analysis.json').read_text());plan=json.loads((RESULTS/'driven_weather_plan.json').read_text());A.install(check=False)
 maxerr=0.
 for jb in plan['jobs'][::9]:
  e,meta=make_model(jb);rr=e.environment.sample(100,3,123,antithetic=True);K=len(meta['raw_matrices']);qh=np.array(meta['exact_host_state_prob']);R0=meta['mean_R0']
  for t in [0,49,99]:
   mat=e.base_parts.sum(0)+np.einsum('rk,kij->rij',rr[t]-1,e.rate_parts.sum(0))
   for rep in range(6):
    k=int(np.argmax(rr[t,rep,:K]));H=rr[t,rep,K+k]*qh[k]
    raw=meta['raw_matrices'][k];expected=raw[0]+raw[1]/R0;expected[0,1]*=H
    maxerr=max(maxerr,float(np.max(abs(mat[rep]-expected))))
 assert maxerr<1e-10,maxerr
 # Conditional arithmetic means checked by quadrature in the most variable site, kappa=1.
 jb=next(j for j in plan['jobs'] if j['site']=='HARV' and j['kappa']==1 and j['gamma']==0)
 e,meta=make_model(jb);K=len(meta['exact_host_state_prob']);cuts=special.ndtri(np.arange(K+1)/K);a=jb['kappa']*jb['sigma']*jb['joint']['rho'];qerr=0
 for k in range(K):
  quad=integrate.quad(lambda z:math.exp(-.5*(z-a)**2)/math.sqrt(2*math.pi),cuts[k],cuts[k+1],epsabs=1e-12)[0]
  qerr=max(qerr,abs(quad-meta['exact_host_state_prob'][k]))
 assert qerr<1e-10
 # Constant weather and host = deterministic Engine, known zero loss.
 j0=dict(jb,sigma=0.,kappa=0.,weather=[0.,0.],duration=[300.,300.],gamma=0.)
 e,meta=make_model(j0);zero=e.loss(steps=4000,burn=2000,pairs=4,seed=123)['D'];assert abs(zero)<1e-10
 print('VERIFY driven state matrices',maxerr,'quadrature',qerr,'constant case D',zero,flush=True)
 # Site single-covariate sensitivities already defined in the registered covariate section.
 sites=d['neon_sites'];sig=np.array([r['sigma'] for r in sites]);p=np.array([r['nip'] for r in sites]);rows=[]
 keys=['dd0_oct_sep','dd10_oct_sep','hotdry_nymph_may_jul','hotdry_larv_prev_aug_sep','P_rh_jun_sub82','P_rh_gs','P_gwettop_gs']
 for k in keys:
  z=np.array([r[k] for r in sites]);r,pn,pp=S.freedman_lane_partial6(p,sig,z)
  rows.append(dict(covariate=k,partial_rho=r,p_negative=pn,p_positive=pp,pearson_sigma=S.pearson_r(sig,z),pearson_nip=S.pearson_r(p,z)))
 table('ancillary_neon_adjustments.csv',rows)
 # Cary additional seasonal definitions and previous growing-season precipitation.
 rs=d['cary_pairs']['primary'];plots=sorted({r['plot'] for r in rs});t0=min(r['mouse_year'] for r in rs);t1=max(r['mouse_year'] for r in rs)
 starts=np.random.default_rng(20261114).integers(0,t1-t0,size=(2000,(t1-t0+2)//2));cr=[]
 for k in ['dd0_oct_sep_nymph','dd10_oct_sep_nymph','hotdry_nymph_may_jul_nymph','hotdry_larv_prev_aug_sep_nymph','C5_gsprcp_larva','P_rh_jun_sub82_nymph','P_gwettop_gs_nymph','P_rh_gs_nymph']:
  rr=[dict(r) for r in rs];vals=standardized([r[k] for r in rr])
  for r,v in zip(rr,vals):r[k]=float(v)
  b,bz,se,ci,df=S.cary_block_boot_slope(rr,plots,[k],starts,t0,t1)
  cr.append(dict(covariate=k,b=b,se=se,ci_low=ci[0],ci_high=ci[1]))
 table('ancillary_cary_slopes.csv',cr)
 # Bootstrap Spearman CIs, supplementary to Pearson intervals in the main file.
 ys=sorted({r['mouse_year'] for r in rs});draw=np.concatenate([starts,starts+1],axis=1);rho_rows=[]
 for lag in ['nymph','larva']:
  for k in L.COV_KEYS+['PC1_env']:
   zz=np.array([next(r[k+'_'+lag] for r in rs if r['mouse_year']==y) for y in ys])
   for target,field in [('log_mice','x'),('log_nip_next','y')]:
    xx=np.array([np.mean([r[field] for r in rs if r['mouse_year']==y]) for y in ys]);rx=stats.rankdata(xx[draw],axis=1);rz=stats.rankdata(zz[draw],axis=1);rx-=rx.mean(1,keepdims=True);rz-=rz.mean(1,keepdims=True);rr=(rx*rz).sum(1)/np.sqrt((rx*rx).sum(1)*(rz*rz).sum(1))
    lo,hi=np.quantile(rr,[.025,.975]);rho_rows.append(dict(covariate=k,lag=lag,target=target,rho=S.spearman_rho(xx,zz),rho_low=float(lo),rho_high=float(hi)))
 table('cary_spearman_intervals.csv',rho_rows)
 # Diagnose insufficient weather-year rank in heavily adjusted bootstrap samples.
 rank_rows=[]
 for per in ['primary','holdout']:
  rr=d['cary_pairs'][per];t0=min(r['mouse_year'] for r in rr);t1=max(r['mouse_year'] for r in rr);starts=np.random.default_rng(20261114).integers(0,t1-t0,size=(2000,(t1-t0+2)//2));ds=np.concatenate([starts,starts+1],axis=1)+t0
  for pp in [1,6]:
   insufficient=sum(len(set(row))<pp+2 for row in ds)
   rank_rows.append(dict(period=per,covariates=pp,draws=2000,insufficient_distinct_years=insufficient,fraction=insufficient/2000,interpretation='Necessary year-count diagnostic; field slope estimator may still use within-year grid contrast'))
 table('bootstrap_weather_rank_diagnostic.csv',rank_rows)
 # Weather validation between two independently produced gridded products.
 weather=[]
 for lab in ['CARY']+[s+'_tick' for s in SITES]:
  dm=L.read_daymet_daily(RAW/'envcov/daymet'/f'{lab}.csv');pw=L.read_power_daily(RAW/'envcov/nasa_power'/f'{lab}.json');c=L.build_annual_covariates(dm,pw)
  ys=list(range(1995,2014)) if lab=='CARY' else next(r['test_years'] for r in sites if r['site']==lab[:4])
  for a,b in [('C6_satdef','P_rh_gs'),('C3_wmin','P_ts_wmin'),('C5_gsprcp','P_gwettop_gs')]:weather.append(dict(location=lab,daymet=a,power=b,n=len(ys),pearson=S.pearson_r([c[y][a] for y in ys],[c[y][b] for y in ys])))
 table('power_daymet_crosschecks.csv',weather)
 # Twelve-month autocorrelation is not evidence for fine-step AR(1); return higher-lag inputs.
 acf=[]
 for s,j in d['joint'].items():
  xx=np.array(j['log_mouse']);ww=np.array(j['weather']);years=np.array(j['years'])
  for lag in [1,2,3]:
   pairs=[(i,k) for i,y in enumerate(years) for k,z in enumerate(years) if z==y+lag]
   acf.append(dict(site=s,lag_years=lag,n_pairs=len(pairs),mouse_r=S.pearson_r([xx[i] for i,k in pairs],[xx[k] for i,k in pairs]),weather_r=S.pearson_r([ww[i] for i,k in pairs],[ww[k] for i,k in pairs])))
 table('annual_memory_inputs.csv',acf)
 save('verification_and_supplement.json',dict(started=started,completed=now(),state_matrix_check_error=maxerr,conditional_mean_quadrature_error=qerr,constant_model_loss=zero,neon=rows,cary=cr,rank_diagnostics=rank_rows,power_crosschecks=weather,annual_memory=acf))
if __name__=='__main__':main()
