#!/usr/bin/env python3
"""Reproduce the registered Wood NB-GLMMs, then gate further trait analyses.

Run python3 -I .../analysis.py validate|phase1|traits. Downloaded code is never
executed. Inherited helpers remain intact; glmm_checked.py fixes a reviewed
implementation error. All model outputs are checkpointed individually.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import sys, json, csv, time, subprocess, hashlib
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"_deps"))
import numpy as np
import pandas as pd
from scipy import stats, optimize
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
sys.path.insert(0,str(HERE))
from glmm_checked import HierarchicalNBData, LaplaceNBGLMM, nb_loglik_and_derivs
RES=HERE/'results'
RAW=ROOT/'data/raw/wood'
TRAITS=['n_fluc_obligate','storage_stage','spreading','prop_active_transfers']
def stamp(): return subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip()
def plain(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [plain(v) for v in x]
 return x
def save(name,obj):
 (RES/(name+'.json')).write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+'\n')
def load():
 df=pd.read_csv(RAW/'wood_zenodo_7448240/final_dataset.csv')
 tr=pd.read_csv(HERE/'traits_blind.csv').set_index('para_spc').sort_index()
 assert set(df['para.spc'])==set(tr.index)
 return df,tr

def data_object(df,tr,taxcol='par_type'):
 df=df.copy(); sp=list(tr.index)
 hp=(df['fish.spc']+' - '+df['para.spc']).astype('category')
 fish=df['fish.id'].astype('category')
 si=np.array([sp.index(v) for v in df['para.spc']],int)
 ti=pd.Categorical(tr[taxcol]).codes.astype(int)
 nh=np.minimum(tr['nhosts_exact'].to_numpy(),3).astype(int)-1
 lat=df['lat'].to_numpy(); lat=(lat-np.nanmean(lat))/np.nanstd(lat,ddof=1); lat=np.nan_to_num(lat)
 year=df['year'].to_numpy(); uy=(year-year.mean())/year.std(ddof=1)
 d=HierarchicalNBData(df['count'].to_numpy(float),hp.cat.codes.to_numpy(dtype=int),si,fish.cat.codes.to_numpy(dtype=int),ti,ti[si],nh[si],lat,df['length'].to_numpy(),uy,len(hp.cat.categories),len(sp),len(fish.cat.categories),len(set(ti)))
 return d,nh

def validate():
 y=np.array([0.,1.,4.,20.]); eta=np.array([-2.,0.2,1.2,2.8]); psi=np.array([-1.,0.,1.,-0.5])
 out=nb_loglik_and_derivs(y,eta,psi); mu=np.exp(eta); ph=np.exp(psi)
 errs={'scipy_nb':abs(out[0]+stats.nbinom.logpmf(y,ph,ph/(ph+mu)).sum())}
 eps=1e-4
 for j in range(4):
  de=np.eye(4)[j]*eps
  fp=nb_loglik_and_derivs(y,eta+de,psi); fm=nb_loglik_and_derivs(y,eta-de,psi)
  gp=nb_loglik_and_derivs(y,eta,psi+de); gm=nb_loglik_and_derivs(y,eta,psi-de)
  errs[f'eta_grad_{j}']=abs((fp[0]-fm[0])/(2*eps)-out[1][j])
  errs[f'psi_grad_{j}']=abs((gp[0]-gm[0])/(2*eps)-out[2][j])
  errs[f'eta_hess_{j}']=abs((fp[1][j]-fm[1][j])/(2*eps)-out[3][j])
  errs[f'cross_hess_{j}']=abs((gp[1][j]-gm[1][j])/(2*eps)-out[4][j])
  errs[f'psi_hess_{j}']=abs((gp[2][j]-gm[2][j])/(2*eps)-out[5][j])
 df,tr=load(); d,nh=data_object(df,tr)
 m=LaplaceNBGLMM(d,X_slope=np.ones((len(tr),1)))
 p=m.unpack_theta(m.initial_theta()); _,s=m._mu_and_sig(p)
 errs['dispersion_sd']=abs(s[m.sl_psi][0]-np.exp(p['loglogsigma_phi']))
 assert max(errs.values())<1e-5,errs
 summary={'timestamp':stamp(),'errors':errs,'pass':True,'n_observations':len(df),'n_taxa':len(tr),'n_fish':d.n_fish,'n_host_parasite':d.n_hp,'n_fish_species':df['fish.spc'].nunique(),'year_min':int(df.year.min()),'year_max':int(df.year.max()),'year_sd':df.year.std(ddof=1),'hashes':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['traits_blind.csv','results/theory_predictions.txt','results/pre_analysis_power.txt']}}
 save('validation',summary); print(json.dumps(summary,indent=2),flush=True)

def fit_save(name,m,theta=None):
 path=RES/(name+'.json')
 if path.exists():
  r=json.loads(path.read_text()); print('CACHED',name,r['nll'],flush=True); return r
 print('START',name,stamp(),flush=True); start=time.monotonic()
 r=m.fit(theta0=theta,maxiter=220)
 r['elapsed_seconds']=time.monotonic()-start;r['timestamp']=stamp()
 save(name,r)
 print('DONE',name,{k:r[k] for k in ['nll','aicc','converged','outer_grad_max','inner_grad_max','hessian_min_eigenvalue','elapsed_seconds']},'slopes',r['pars'].get('beta_slope'),flush=True)
 return plain(r)

def phase1():
 assert json.loads((RES/'validation.json').read_text())['pass']
 df,tr=load();d,nh=data_object(df,tr);n=len(tr)
 spec=[('M3_nhosts',dict(X_slope=np.eye(3)[nh],bog_obs_idx=d.nh_of_obs,n_bog=3)),('M2_year',dict(X_slope=np.ones((n,1)))),('M4_partype',dict(X_slope=np.eye(d.n_tax)[d.tax_of_spc],bog_obs_idx=d.tax_of_obs,n_bog=d.n_tax)),('M1_null',dict(include_year=False))]
 fits={}
 for name,kw in spec: fits[name]=fit_save(name,LaplaceNBGLMM(d,**kw))
 b=np.array(fits['M3_nhosts']['pars']['beta_slope']); se=np.array(fits['M3_nhosts']['se_theta'])[10:13]
 order=sorted(fits,key=lambda x:fits[x]['aicc'])
 checks={'three_plus_tolerance':bool(abs(b[2]+.349)<=.03),'one_two_include_zero':bool(np.all(np.abs(b[:2])<1.96*se[:2])),'M3_best':order[0]=='M3_nhosts','numerical_convergence':all(r['converged'] and r['inner_grad_max']<1e-5 and r['outer_grad_max']<.02 for r in fits.values())}
 result={'timestamp':stamp(),'checks':checks,'pass':all(checks.values()),'aicc_order':order,'slopes':b,'slope_se':se,'percent_per_decade':100*np.expm1(10*b/df.year.std(ddof=1))}
 save('reproduction_gate',result);print(json.dumps(plain(result),indent=2),flush=True)
 pd.DataFrame({'para_spc':tr.index,'slope':fits['M2_year']['spc_slopes'],'se':fits['M2_year']['spc_slopes_se']}).to_csv(RES/'M2_taxon_slopes.csv',index=False)

if __name__=='__main__':
 action=sys.argv[1] if len(sys.argv)>1 else 'validate'
 {'validate':validate,'phase1':phase1}[action]()
