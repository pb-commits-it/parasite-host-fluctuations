#!/usr/bin/env python3
"""Run only after registered weather_pre_analysis.json exists."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from unified_traits import warm
from meta_analysis import designs
from weather import haversine
import numpy as np
from scipy import stats

def corr_rows(df,x,y,label,seed=20261115):
 sub=df[['year',x,y]].dropna();n=len(sub)
 if n<4 or sub[x].std()==0 or sub[y].std()==0:return {'label':label,'x':x,'y':y,'n':n,'r':None}
 r=float(sub[x].corr(sub[y]));rng=np.random.default_rng(seed);years=np.unique(sub.year);vals=[]
 blocks={yr:sub[sub.year==yr] for yr in years}
 for b in range(1000):
  ss=a.pd.concat([blocks[yr] for yr in rng.choice(years,len(years),replace=True)])
  if ss[x].std()>0 and ss[y].std()>0: vals.append(ss[x].corr(ss[y]))
 lo,hi=np.quantile(vals,[.025,.975])
 return {'label':label,'x':x,'y':y,'n':n,'n_years':len(years),'r':r,'ci_lo':float(lo),'ci_hi':float(hi),'bootstrap':'1000 independent year resamples; serial dependence unaccounted'}

def descriptives():
 assert (a.RES/'weather_pre_analysis.json').exists()
 fish=a.pd.read_csv(a.RES/'weather_matches.csv');df,tr=a.load();rs=[]
 # Count totals within fish only across the retained published taxon pairs.
 totals=df.groupby('fish.id')['count'].sum().rename('retained_parasites')
 fish=fish.join(totals,on='fish.id');fish['log1p_burden']=np.log1p(fish.retained_parasites)
 cov=['sst_mean','sst_monthly_sd','temp_na_rm','ord1','ord2']
 annual=fish.groupby(['fish.spc','year']).agg({**{c:'mean' for c in cov},'sfish_density':'first','log1p_burden':'mean'}).reset_index()
 annual['host_abs_change']=np.nan
 for sp,s in annual.groupby('fish.spc'):
  order=s.sort_values('year');change=abs(order.sfish_density.diff());change[order.year.diff()!=1]=np.nan
  annual.loc[order.index,'host_abs_change']=change
  for c in cov:
   for y in ['log1p_burden','sfish_density','host_abs_change']:
    ss=annual.loc[order.index];rs.append(corr_rows(ss,c,y,sp))
 # Pooled year associations after centering outcomes/host proxies within species.
 for y in ['log1p_burden','sfish_density','host_abs_change']:
  ss=annual.copy();ss[y]=ss[y]-ss.groupby('fish.spc')[y].transform('mean')
  yr=ss.groupby('year')[[*cov,y]].mean().reset_index()
  for c in cov:rs.append(corr_rows(yr,c,y,'pooled_year_species_centered'))
 a.pd.DataFrame(rs).to_csv(a.RES/'environment_correlations.csv',index=False)
 annual.to_csv(a.RES/'annual_host_environment_outcome.csv',index=False)
 race=a.pd.read_csv(a.RES/'race_rocks_monthly_refetched.csv')
 rr=race.groupby('year').sst_c.agg(['mean','count','std']);rr=rr[rr['count']>=11]
 existing=fish[['year','temp_na_rm']].dropna().drop_duplicates('year').set_index('year')
 cmp=existing.join(rr,how='inner');cmp['difference']=cmp.temp_na_rm-cmp['mean'];cmp.to_csv(a.RES/'race_archive_comparison.csv')
 # Source coverage and range; numbers are emitted, never hand transcribed from memory.
 summary={'timestamp':a.stamp(),'coverage':{c:{'fish':int(fish[c].notna().sum()),'years':int(fish.loc[fish[c].notna(),'year'].nunique()),'min':float(fish[c].min()),'max':float(fish[c].max())} for c in cov+['sfish_density']},'race_refetch_overlap_years':len(cmp),'race_mean_abs_difference':float(cmp.difference.abs().mean()),'race_max_abs_difference':float(cmp.difference.abs().max()),'pooled_correlations':[r for r in rs if r['label']=='pooled_year_species_centered'],'host_variability':'not measured; absolute consecutive-year change in a standardized interpolated index is a proxy only'}
 a.save('weather_descriptive_summary',summary);print(json.dumps(a.plain(summary),indent=2),flush=True)

def fits():
 assert (a.RES/'weather_pre_analysis.json').exists()
 df,tr=a.load();fish=a.pd.read_csv(a.RES/'weather_matches.csv');df=df.merge(fish[['fish.id','sst_mean','sst_monthly_sd']],on='fish.id',validate='many_to_one')
 base=json.loads((a.RES/'M3_nhosts.json').read_text());b=np.array(base['pars']['beta_slope']);bn=np.r_[b[0],b[1:]-b[0]]
 sets={'W1_ersst':['sst_mean','sst_monthly_sd'],'W2_race':['temp_na_rm'],'S5_pollutants':['ord1','ord2'],'S5_fish_density':['sfish_density'],'S5_all_archived':['temp_na_rm','ord1','ord2','sfish_density']}
 summary={}
 for name,cols in sets.items():
  sub=df.dropna(subset=cols).copy();tri=tr.loc[sorted(sub['para.spc'].unique())];N,T=designs(tri)
  d,nh=a.data_object(sub,tri);d.uyear=(sub.year.to_numpy()-df.year.mean())/df.year.std(ddof=1);d.slat=np.nan_to_num((sub.lat.to_numpy()-df.lat.mean())/df.lat.std(ddof=1))
  E=np.column_stack([(sub[c]-sub[c].mean())/sub[c].std(ddof=1) for c in cols]);G=np.eye(3)[d.nh_of_obs]
  extra=np.column_stack([E[:,j]*G[:,k] for j in range(len(cols)) for k in range(3)])
  test=np.column_stack([G,G*d.uyear[:,None],extra]);rank=np.linalg.matrix_rank(test)
  result={'covariates':cols,'n_rows':len(sub),'n_fish':sub['fish.id'].nunique(),'n_taxa':len(tri),'n_years':sub.year.nunique(),'design_rank':int(rank),'design_columns':test.shape[1],'design_condition':float(np.linalg.cond(test)),'covariate_means':{c:sub[c].mean() for c in cols},'covariate_sd':{c:sub[c].std(ddof=1) for c in cols}}
  # Rank is checked before using count outcomes in these fits.
  if rank<test.shape[1]:
   result['not_fit']='rank deficient with retained year and group effects';summary[name]=result;a.save('weather_fit_summary',summary);continue
  for label,X in [('baseline',N),('joint',np.column_stack([N,T]))]:
   if np.linalg.matrix_rank(X)<X.shape[1]:result[label]={'estimable':False};continue
   m=a.LaplaceNBGLMM(d,X_slope=X,has_tax_re=True,bog_obs_idx=d.nh_of_obs,n_bog=3,extra_fixed_X=extra)
   r=a.fit_save(name+'_'+label,m,warm(m,base,np.r_[bn,np.zeros(X.shape[1]-3)]))
   offset=10;result[label]={'aicc':r['aicc'],'nll':r['nll'],'beta':r['pars']['beta_slope'],'se':r['se_theta'][offset:offset+X.shape[1]],'environment_beta':r['pars']['beta_extra'],'environment_se':r['se_theta'][-extra.shape[1]:],'converged':r['converged'],'outer_grad_max':r['outer_grad_max'],'inner_grad_max':r['inner_grad_max'],'hessian_min_eigenvalue':r['hessian_min_eigenvalue']}
   summary[name]=result;a.save('weather_fit_summary',summary)
  result['delta_aicc']=result['joint']['aicc']-result['baseline']['aicc'];summary[name]=result;a.save('weather_fit_summary',summary)
 print(json.dumps(a.plain(summary),indent=2),flush=True)

if __name__=='__main__':{'descriptives':descriptives,'fits':fits}[sys.argv[1]]()
